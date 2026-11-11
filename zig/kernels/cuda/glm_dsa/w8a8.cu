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

// Bench-only copies of tf_glm53_w8a8_gemm. The release kernel above is unchanged.
// One template. The switch is the mode: knock-outs (dx_*) may tear or ignore K, and the same-bits
// modes (sc, s6, pf8, pf16, o2, m64) keep each output's products, K order and __fmaf_rn chain.
// A row's chain does not read its tile-mates, the ring depth, or which CTA owns the row.
//
// smem, GB10 cap 101376 B a CTA:
//   s6 ring is 6 * 16384 = 98304. Scales stay on __ldg (0 B). 98304 <= 101376, so 6 stages, not 5.
//   o2 ring is 3 * 16384 = 49152. Two CTAs are 98304. Scales stay on __ldg. 98304 <= 101376.
//   sc stages the tile's xs rows and ws entries beside the ring, 4 * (128 * 4 + 16) = 2112 B,
//   total 67648. cp.async has no register result, so the stage that issues a scale does not consume it.
//   m64 ring is 4 * (64 * 64 + 128 * 64) = 49152.
namespace tf_glm53_w8f {

enum Mode : int {
    DX_L2 = 1,
    DX_NOSCALE = 2,
    DX_MMA = 3,
    DX_NOSYNC = 4,
    SC = 5,
    S6 = 6,
    PF8 = 7,
    PF16 = 8,
    O2 = 9,
    M64 = 10,
};

static_assert(6 * 16384 == 98304 && 98304 <= 101376, "s6 ring fits, scales not in smem");
static_assert(3 * 16384 == 49152 && 2 * 49152 <= 101376, "two o2 CTAs fit, scales not in smem");
static_assert(4 * (128 * 4 + 16) == 2112 && 65536 + 2112 == 67648 && 67648 <= 101376, "sc scale bytes");
static_assert(4 * (64 * 64 + 128 * 64) == 49152, "m64 ring");

// 4-byte cp.async. src-size 0 writes zeros and does not read global (cp16z's predicate, for a float).
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(qmm_frag::smem(dst)), "l"(src), "r"(read ? 4 : 0));
}

// prefetch.global.L2 is the form exl3_experts.cu and l2pf.cu already compile. Not L2::evict_* and not
// a cp.async cache hint: those fault on GB10. One thread issues each 64-byte line. K is clamped to the
// last stage. A row at or past M is clamped to the last row, which is inside the activation buffer.
template <int BM, int BN, int BK>
__device__ __forceinline__ void prefetch_ahead(const uint8_t* xq, const uint8_t* w, int m0, int t0, int M, int K, int KT,
                                              int it, int ahead, int tid) {
    if (M <= 0 || KT <= 0 || ahead <= 0) return;
    int kt = it + ahead;
    if (kt >= KT) kt = KT - 1;
    if (kt < 0) kt = 0;
    constexpr int W_TILE = 64 * BK;
    constexpr int per = W_TILE / 64;
    constexpr int wlines = (BN / 64) * per;
    if (tid < BM) {
        int row = m0 + tid;
        if (row >= M) row = M - 1;
        const uint8_t* line = xq + static_cast<size_t>(row) * static_cast<size_t>(K) + static_cast<size_t>(kt) * static_cast<size_t>(BK);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
    } else if (tid < BM + wlines) {
        const int wline = tid - BM;
        const int tt = wline / per;
        const int line = wline - tt * per;
        const uint8_t* addr = w + (static_cast<size_t>(t0 + tt) * static_cast<size_t>(KT) + static_cast<size_t>(kt)) * W_TILE +
                              static_cast<size_t>(line) * 64;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
    }
}

template <int MODE>
__device__ __forceinline__ void body(const uint8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w,
                                    const float* __restrict__ ws, void* __restrict__ out, int M, int N, int K, int out_f32,
                                    unsigned char* smem) {
    namespace base = tf_glm53_w8a8;
    constexpr bool dx_l2 = MODE == DX_L2;
    constexpr bool dx_noscale = MODE == DX_NOSCALE;
    constexpr bool dx_mma = MODE == DX_MMA;
    constexpr bool dx_nosync = MODE == DX_NOSYNC;
    constexpr bool scale_pipe = MODE == SC;
    constexpr int prefetch = MODE == PF8 ? 8 : MODE == PF16 ? 16 : 0;
    constexpr int BM = MODE == M64 ? 64 : base::BM;
    constexpr int BN = base::BN;
    constexpr int BK = base::BK;
    constexpr int STAGES = MODE == S6 ? 6 : MODE == O2 ? 3 : base::STAGES;
    constexpr int WM = base::WM, WN = base::WN, THREADS = base::THREADS;
    constexpr int MT = BM / WM / 16;
    constexpr int NT = BN / WN / 8;
    constexpr int A_BYTES = BM * BK;
    constexpr int B_BYTES = BN * BK;
    constexpr int STAGE = A_BYTES + B_BYTES;
    constexpr int W_TILE = base::W_TILE;
    // xs tile (BM floats) plus two weight scales, padded so the next slot stays 16-byte aligned.
    constexpr int SCALE_SLOT = BM * static_cast<int>(sizeof(float)) + 16;
    static_assert(BM % (WM * 16) == 0, "m16 tiles");
    static_assert(STAGES >= 3, "wait_group distance");
    static_assert(SCALE_SLOT % 16 == 0, "scale slot alignment");
    static_assert(BM + (BN / 64) * (W_TILE / 64) <= THREADS, "one thread a 64-byte line");
    if constexpr (scale_pipe) static_assert(STAGES * SCALE_SLOT == 2112, "sc bytes");
    if constexpr (MODE == S6) static_assert(STAGES * STAGE == 98304, "s6 bytes");
    if constexpr (MODE == O2) static_assert(STAGES * STAGE == 49152, "o2 bytes");
    if constexpr (MODE == M64) static_assert(BM == 64 && MT == 2 && STAGES == 4, "m64 tile");

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, t = lane & 3;
    const int2 at = qmm_frag::tile_of(blockIdx.x, M, N, BM, BN, base::RASTER);
    const int m0 = at.x, n0 = at.y;
    const int KT = K / BK;
    const int t0 = n0 / 64;

    auto load = [&](int s, int kt) {
        int src = kt;
        if constexpr (dx_l2) {
            src = kt & 3;
            if (src >= KT) src = 0;
        }
        unsigned char* pa = smem + s * STAGE;
        unsigned char* pb = pa + A_BYTES;
        for (int c = tid; c < BM * (BK / 16); c += THREADS) {
            const int r = c >> 2, ch = c & 3;
            const bool live = m0 + r < M;
            const size_t row = static_cast<size_t>(live ? m0 + r : 0);
            qmm_frag::cp16z(pa + r * BK + base::aswz(r, ch) * 16, xq + row * K + static_cast<size_t>(src) * BK + ch * 16, live);
        }
        for (int c = tid; c < B_BYTES / 16; c += THREADS) {
            const int tt = c / (W_TILE / 16), off = c % (W_TILE / 16);
            qmm_frag::cp16(pb + c * 16, w + (static_cast<size_t>(t0 + tt) * KT + src) * W_TILE + off * 16);
        }
        if constexpr (scale_pipe) {
            unsigned char* slotp = smem + STAGES * STAGE + s * SCALE_SLOT;
            for (int r = tid; r < BM; r += THREADS) {
                const bool live = m0 + r < M;
                const int row = live ? m0 + r : 0;
                const size_t kb = static_cast<size_t>(kt >> 1);
                cp4z(slotp + r * 4, xs + kb * static_cast<size_t>(M) + static_cast<size_t>(row), live);
            }
            if (tid < 2) {
                const size_t wi = static_cast<size_t>(t0 + tid) * static_cast<size_t>(KT) + static_cast<size_t>(kt);
                cp4z(slotp + BM * 4 + tid * 4, ws + wi, true);
            }
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
    // The loop commits an empty group when it does not copy. Drain the prologue first so those empty
    // groups are not what retires the real copies. dx_mma then multiplies the drained slots only.
    if constexpr (dx_mma) {
        qmm_frag::wait<0>();
        __syncthreads();
    }
    const float* wrow = ws + static_cast<size_t>(t0 + (wn >> 1)) * KT;
    float wsc_hold = 0.0f;
    if constexpr (dx_mma) {
        wsc_hold = 1.0f;
        asm volatile("" : "+f"(wsc_hold));
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) xsr[i][h] = wsc_hold;
        (void)xs;
        (void)ws;
        (void)wrow;
    } else if constexpr (dx_noscale) {
        if (KT > 0) wsc_hold = __ldg(wrow + 0);
        const size_t kb0 = 0;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g + h * 8;
                xsr[i][h] = row < M ? __ldg(xs + kb0 * static_cast<size_t>(M) + row) : 0.0f;
            }
    } else if constexpr (scale_pipe) {
        (void)wrow;
        (void)wsc_hold;
    } else {
        (void)wsc_hold;
    }
#pragma unroll 1
    for (int it = 0; it < KT; ++it) {
        if constexpr (prefetch > 0) prefetch_ahead<BM, BN, BK>(xq, w, m0, t0, M, K, KT, it, prefetch, tid);
        qmm_frag::wait<STAGES - 2>();
        if constexpr (!dx_nosync) __syncthreads();
        const int next = it + STAGES - 1;
        if constexpr (!dx_mma) {
            if (next < KT) load(next % STAGES, next);
        }
        qmm_frag::commit();
        // Slot it % STAGES holds K step it. It was committed STAGES-1 steps ago. The copy just issued
        // is next % STAGES, a different slot while STAGES >= 3, so this stage does not consume its own load.
        float wsc = 0.0f;
        if constexpr (scale_pipe) {
            const unsigned char* slotp = smem + STAGES * STAGE + (it % STAGES) * SCALE_SLOT;
            const float* xs_s = reinterpret_cast<const float*>(slotp);
            const float* ws_s = reinterpret_cast<const float*>(slotp + BM * 4);
            wsc = ws_s[wn >> 1];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int local = wm * (BM / WM) + i * 16 + g + h * 8;
                    xsr[i][h] = xs_s[local];
                }
        } else if constexpr (dx_mma || dx_noscale) {
            wsc = wsc_hold;
        } else {
            wsc = __ldg(wrow + it);
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
        }
        int slot = it % STAGES;
        if constexpr (dx_mma) slot = it % (STAGES - 1);
        const unsigned char* pa = smem + slot * STAGE;
        const uint4* pb = reinterpret_cast<const uint4*>(pa + A_BYTES);
        uint4 bv[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) bv[j] = pb[(wn * NT + j) * 32 + lane];
        float part[MT][NT][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = 2 * h + (lane >> 4);
                qmm_frag::ldmatrix4(a[i], pa + r * BK + base::aswz(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = h == 0 ? bv[j].x : bv[j].z;
                const uint32_t b1 = h == 0 ? bv[j].y : bv[j].w;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (h == 0) base::mma0_e4m3(part[i][j], a[i], b0, b1);
                    else base::mma_e4m3(part[i][j], a[i], b0, b1);
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

}  // namespace tf_glm53_w8f

#define W8F_KERNEL(sym, mode, minb)                                                                                                    \
    extern "C" __global__ void __launch_bounds__(256, minb) sym(const uint8_t* __restrict__ xq, const float* __restrict__ xs,         \
                                                                const uint8_t* __restrict__ w, const float* __restrict__ ws,         \
                                                                void* __restrict__ out, int M, int N, int K, int out_f32) {          \
        extern __shared__ __align__(128) unsigned char smem[];                                                                        \
        tf_glm53_w8f::body<tf_glm53_w8f::mode>(xq, xs, w, ws, out, M, N, K, out_f32, smem);                                          \
    }

W8F_KERNEL(tf_glm53_w8a8_dx_l2, DX_L2, 1)
W8F_KERNEL(tf_glm53_w8a8_dx_noscale, DX_NOSCALE, 1)
W8F_KERNEL(tf_glm53_w8a8_dx_mma, DX_MMA, 1)
W8F_KERNEL(tf_glm53_w8a8_dx_nosync, DX_NOSYNC, 1)
W8F_KERNEL(tf_glm53_w8a8_gemm_sc, SC, 1)
W8F_KERNEL(tf_glm53_w8a8_gemm_s6, S6, 1)
W8F_KERNEL(tf_glm53_w8a8_gemm_pf8, PF8, 1)
W8F_KERNEL(tf_glm53_w8a8_gemm_pf16, PF16, 1)
W8F_KERNEL(tf_glm53_w8a8_gemm_o2, O2, 2)
W8F_KERNEL(tf_glm53_w8a8_gemm_m64, M64, 1)

#undef W8F_KERNEL
