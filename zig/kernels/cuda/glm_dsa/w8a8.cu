// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 W8A8 prefill projection: e4m3 activations times the checkpoint's FP8B e4m3 weights on
// mma.sync m16n8k32, fp32 accumulators. TF_GLM53_PREFILL_W8A8=1 launches this for o_proj.
// =2 reuses the same quant and GEMM for every other FP8B prefill projection whose K fits the level-2 scratch.
//
// Provenance:
// - The e4m3 m16n8k32 MMA, the per-(row, 128-K) activation scale with amax / 448 and the FP32 promotion of block
//   scales outside the MMA follow zig/kernels/cuda/glm_dsa/fp8_mma.cu, which is adapted from
//   ashhart/TensorFold@d1018b31:zig/kernels/cuda/ck_lane.cu and mma4.cuh (Apache-2.0).
// - The cp.async multistage pipeline, the ldmatrix A path and the banded tile order (qmm_frag::tile_of) follow
//   zig/kernels/cuda/qmmf.cu and qmm_frag.cuh, adapted from bertholomus/TensorFold@ddfad35 (Apache-2.0).
// - The 64-byte-row A swizzle is mma4.cuh's A8 chunk rotation (ashhart/TensorFold, Apache-2.0).
// Changes: FP8B weight tiles are read in place (no repack): the activation quantizer writes each 16-K group in the
// K order the FP8B packer gives each B lane, so one 16-byte shared load per lane is both k32 B fragments of a 64-K
// step. 128x128x64 tiles, 4 stages, 8 warps (2 x 4), FP32 promotion once per 64-K weight tile in ascending K order.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../qmm_frag.cuh"

namespace tf_glm53_w8a8 {

constexpr int BM = 128;      // rows of a block tile
constexpr int BN = 128;      // output columns of a block tile (two FP8B 64-row weight tiles)
constexpr int BK = 64;       // K of one stage: one FP8B 64x64 weight tile and its one expanded block scale
constexpr int QK = 128;      // activation scale block (row-local)
constexpr int STAGES = 4;
constexpr int WM = 2, WN = 4, THREADS = WM * WN * 32;
constexpr int MT = BM / WM / 16;  // m16 tiles a warp (4)
constexpr int NT = BN / WN / 8;   // n8 tiles a warp (4)
constexpr int A_BYTES = BM * BK;  // 8 KiB of e4m3 activations
constexpr int W_TILE = 64 * BK;   // one FP8B tile: [8 row groups][32 lanes][2 k32 halves][8 bytes]
constexpr int B_BYTES = BN * BK;  // 8 KiB of e4m3 weights
constexpr int STAGE = A_BYTES + B_BYTES;
constexpr int SMEM = STAGES * STAGE;  // 65536: cuda/w8a8.zig shared_bytes
constexpr int RASTER = 8;             // row tiles a band (L2 reuse of both operands)
static_assert(SMEM == 65536, "cuda/w8a8.zig shared_bytes");
static_assert(QK == 2 * BK, "two stages share one activation scale");

// 16-byte chunk c of 64-byte row r: rotate by row pairs so ldmatrix's 8 rows and cp.async's stores are conflict-free.
__device__ __forceinline__ int aswz(int r, int c) { return c ^ ((r >> 1) & 3); }

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mma0_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
#else
    __trap();
#endif
}

}  // namespace tf_glm53_w8a8

// Activation quantizer: x bf16 [M][ldx] -> xq e4m3 [M][K] and xs fp32 [K/128][M].
// One warp per (row, 128-K block), 8 blocks a CTA. scale = amax / 448 (1 when the block is all zero),
// q = cvt.rn.satfinite.e4m3(x / scale) with IEEE division. The row's scale depends on that row alone.
// Byte order in each 16-K group: byte 4q + j holds K 2q + (j & 1) + 8 (j >> 1) (fp8_mma.cu tf_glm53_xoff), the
// order the FP8B packer (cuda/fp8.zig kin) gives B lane q, so both operands see the same K permutation.
extern "C" __global__ void __launch_bounds__(256)
tf_glm53_w8a8_quant(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int K, uint8_t* __restrict__ xq,
                    float* __restrict__ xs) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.y;
    const int kb = blockIdx.x * 8 + warp;
    if (row >= M || kb >= K / tf_glm53_w8a8::QK) return;  // whole warps leave together
    const int g = lane >> 2, q = lane & 3;
    const int k0 = kb * tf_glm53_w8a8::QK + g * 16 + 2 * q;
    const __nv_bfloat16* src = x + static_cast<long long>(row) * ldx;
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(src + k0);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(src + k0 + 8);
    const float v[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[i] / scale, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
    *reinterpret_cast<uint32_t*>(xq + static_cast<long long>(row) * K + kb * tf_glm53_w8a8::QK + g * 16 + 4 * q) = packed;
    if (lane == 0) xs[static_cast<long long>(kb) * M + row] = scale;
}

// y [M][N] (fp32, or bf16 when out_f32 == 0) = sum over K of e4m3(xq) e4m3(w), scaled.
// w / ws are an FP8B projection as qmmf reads it: codes [npad/64][K/64][4096], expanded scales [npad/64][K/64].
// Each 64-K step: two k32 MMAs per (m16, n8) tile into a zeroed fp32 partial, then acc = fma(partial, ws * xs, acc)
// with the step's weight scale and the row's 128-K activation scale. Steps run in ascending K; no split-K, no
// atomics, and a row's result does not depend on the other rows of its tile, so it is row-invariant.
extern "C" __global__ void __launch_bounds__(256, 1)
tf_glm53_w8a8_gemm(const uint8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w,
                   const float* __restrict__ ws, void* __restrict__ out, int M, int N, int K, int out_f32) {
    using namespace tf_glm53_w8a8;
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, t = lane & 3;
    const int2 at = qmm_frag::tile_of(blockIdx.x, M, N, BM, BN, RASTER);
    const int m0 = at.x, n0 = at.y;
    const int KT = K / BK;
    const int t0 = n0 / 64;  // first FP8B 64-row tile of this block

    auto load = [&](int s, int kt) {
        unsigned char* pa = smem + s * STAGE;
        unsigned char* pb = pa + A_BYTES;
        for (int c = tid; c < BM * (BK / 16); c += THREADS) {
            const int r = c >> 2, ch = c & 3;
            const bool live = m0 + r < M;
            const size_t row = static_cast<size_t>(live ? m0 + r : 0);
            qmm_frag::cp16z(pa + r * BK + aswz(r, ch) * 16, xq + row * K + static_cast<size_t>(kt) * BK + ch * 16, live);
        }
        for (int c = tid; c < B_BYTES / 16; c += THREADS) {
            const int tt = c / (W_TILE / 16), off = c % (W_TILE / 16);
            qmm_frag::cp16(pb + c * 16, w + (static_cast<size_t>(t0 + tt) * KT + kt) * W_TILE + off * 16);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    float xsr[MT][2];
#pragma unroll
    for (int i = 0; i < MT; ++i) xsr[i][0] = xsr[i][1] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        qmm_frag::commit();
    }
    const float* wrow = ws + static_cast<size_t>(t0 + (wn >> 1)) * KT;  // this warp's 32 columns sit in one tile
#pragma unroll 1
    for (int it = 0; it < KT; ++it) {
        qmm_frag::wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KT) load(next % STAGES, next);
        qmm_frag::commit();
        const float wsc = __ldg(wrow + it);
        if ((it & 1) == 0) {
            const size_t kb = static_cast<size_t>(it >> 1);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + wm * (BM / WM) + i * 16 + g + h * 8;
                    xsr[i][h] = row < M ? __ldg(xs + kb * static_cast<size_t>(M) + row) : 0.0f;
                }
        }
        const unsigned char* pa = smem + (it % STAGES) * STAGE;
        const uint4* pb = reinterpret_cast<const uint4*>(pa + A_BYTES);
        uint4 bv[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) bv[j] = pb[(wn * NT + j) * 32 + lane];  // n8 tile = FP8B row group
        float part[MT][NT][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = 2 * h + (lane >> 4);
                qmm_frag::ldmatrix4(a[i], pa + r * BK + aswz(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = h == 0 ? bv[j].x : bv[j].z;
                const uint32_t b1 = h == 0 ? bv[j].y : bv[j].w;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (h == 0) mma0_e4m3(part[i][j], a[i], b0, b1);
                    else mma_e4m3(part[i][j], a[i], b0, b1);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const float s0 = wsc * xsr[i][0];
            const float s1 = wsc * xsr[i][1];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[i][j][0] = __fmaf_rn(part[i][j][0], s0, acc[i][j][0]);
                acc[i][j][1] = __fmaf_rn(part[i][j][1], s0, acc[i][j][1]);
                acc[i][j][2] = __fmaf_rn(part[i][j][2], s1, acc[i][j][2]);
                acc[i][j][3] = __fmaf_rn(part[i][j][3], s1, acc[i][j][3]);
            }
        }
    }
    qmm_frag::wait<0>();

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + 2 * t;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g + h * 8;
                if (row >= M) continue;
                const size_t o = static_cast<size_t>(row) * N + col;
                if (out_f32) {
                    *reinterpret_cast<float2*>(static_cast<float*>(out) + o) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(out) + o) =
                        __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
            }
        }
}
