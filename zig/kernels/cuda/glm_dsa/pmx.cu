// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 width-specialised EXL3 prompt experts. Structure ported from this repo's zig/kernels/cuda/exl3_block/x3gm.cu (TensorFold deepseek_v41 x3gm v2). Per-width (K2/K3) specialisation after greatyingzi/GLM-5.3-Flash-EXL3-3bit-TensorFold k3-kernel/exl3k3.cu (Apache-2.0); expert-major prompt-expert design after MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patches 0106-glm-full-prompt-experts (drowzeys) and 0108-glm-full-mia-prompt-experts (Apache-2.0, Copyright 2026 Mia's AI Lab). Independent CUDA; no upstream code copied. EXL3 format after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp); mcg_e4m3 decode after bertholomus/TensorFold@ddfad35 (Apache-2.0).
#include "../tf_vblock.cuh"
#include "../exl3_decode.cuh"
#include "../exl3_block/experts_grouped.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace tf_glm53_pmx {
using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::mma16816;
constexpr int BM = 128, NB = 8, KS = 2, STAGES = 4, TWMAX = 24;
constexpr float HAD_SCALE = 0.08838834764831845f;

struct PmxArgs {
    const half *x0, *x1;
    const long long *tp0, *tp1;
    const int *k2_0, *k2_1;
    const int *items, *counts, *members;
    const half *sv0, *sv1, *sd;
    half *xd;
    float *y;
    int K, N, E;
    float limit;
    int act_mode;
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}
template<int K2> __device__ __forceinline__ int swz(int r, int c) { return c ^ ((r >> 1) & 3); }

template<int K2, int CB>
__device__ __forceinline__ void decode_a(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&a)[4]) {
    constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
    uint32_t st[8];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const uint32_t whi = tile[m.hi[g]], wlo = tile[m.lo[g]];
        const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
        for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
    }
    a[0] = tf_exl3::decode2<CB>(st[0], st[1]);
    a[1] = tf_exl3::decode2<CB>(st[4], st[5]);
    a[2] = tf_exl3::decode2<CB>(st[2], st[3]);
    a[3] = tf_exl3::decode2<CB>(st[6], st[7]);
}

__device__ __forceinline__ float upTo(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template<int K2G, int K2U, int CB>
__device__ __forceinline__ void gu_body(const PmxArgs a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TWG = 4 * K2G, TWU = 4 * K2U, LDA = 32, W = 16, THREADS = 512;
    constexpr size_t ABYTES = 2ull * BM * LDA * sizeof(half);
    constexpr size_t WBYTES = 2ull * KS * NB * TWMAX * sizeof(uint32_t);
    constexpr size_t STAGE_BYTES = ABYTES + WBYTES;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp >> 3, slice = warp & 7, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int KT = a.K >> 4, NT = a.N >> 4, S = KT / KS;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % STAGES) * STAGE_BYTES;
        half* ra = reinterpret_cast<half*>(stg);
        const int k0 = s * KS * 16;
        for (int i = threadIdx.x; i < 2 * BM * 4; i += THREADS) {
            const int m = i / (BM * 4), r = (i / 4) % BM, ch = i % 4;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const half* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * a.K + k0 + ch * 8;
                cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<4>(r, ch) * 8, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + ABYTES);
        for (int i = threadIdx.x; i < 2 * KS * NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % NB, kk = (i / (6 * NB)) % KS, m = i / (6 * NB * KS);
            const int local_tw = m ? TWU : TWG;
            const uint32_t* src = (m ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : reinterpret_cast<const uint32_t*>(a.tp0[e])) + ((size_t)(s * KS + kk) * NT + nb * NB + n) * local_tw + q * 4;
            const bool valid = q < (m ? K2U : K2G);
            if (valid) cp_async16(rw + ((m * KS + kk) * NB + n) * TWMAX + q * 4, src, true);
        }
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < S) load_stage(s); cp_commit(); }
    cp_wait<STAGES - 2>(); __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + STAGES - 1 < S) load_stage(s + STAGES - 1);
        cp_commit();
        const unsigned char* stg = smem + (size_t)(s % STAGES) * STAGE_BYTES;
        const half* A = reinterpret_cast<const half*>(stg) + (size_t)mat * BM * LDA;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(stg + ABYTES) + (size_t)mat * KS * NB * TWMAX + slice * TWMAX;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[4];
            if (mat == 0) { const LaneMap<K2G> map(lane); decode_a<K2G, CB>(Wd + kk * NB * TWMAX, map, af); }
            else { const LaneMap<K2U> map(lane); decode_a<K2U, CB>(Wd + kk * NB * TWMAX, map, af); }
            const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
#pragma unroll
            for (int np = 0; np < 8; ++np) if (16 * np < cnt) {
                uint32_t b[4]; ldsm4(b, A + (16 * np + lr) * LDA + swz<4>(lr, kk * 2 + lc) * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
                mma16816(acc[2 * np], af, b01); mma16816(acc[2 * np + 1], af, b23);
            }
        }
        cp_wait<STAGES - 2>(); __syncthreads();
    }
    // Four rounds spill 32 member rows at a time into the shared ring, as in x3gm.
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (rd) __syncthreads();
        // As x3gm's epilogue: round rd holds accumulator groups 4*rd..4*rd+3 (member rows 32*rd..32*rd+31); c0/c2
        // are member 2t at columns g / g+8, c1/c3 member 2t+1 (the next ep row, +132).
        if (32 * rd < cnt) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int ng = 4 * rd + n;
                float* E = ep + ((size_t)mat * 32 + 8 * n + 2 * t) * 132 + slice * 16 + g;
                E[0] = acc[ng][0]; E[132] = acc[ng][1]; E[8] = acc[ng][2]; E[132 + 8] = acc[ng][3];
            }
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += W) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const int c0 = 4 * lane;
            float v[4]; *reinterpret_cast<float4*>(v) = *reinterpret_cast<float4*>(ep + (size_t)r * 132 + c0);
            tf_exl3::fwht128(v, lane);
            float u[4]; *reinterpret_cast<float4*>(u) = *reinterpret_cast<float4*>(ep + (size_t)(32 * 132 + r * 132 + c0));
            tf_exl3::fwht128(u, lane);
            const size_t col = (size_t)e * a.N + nb * 128 + c0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = v[j] * HAD_SCALE * __half2float(a.sv0[col + j]);
                float uu = u[j] * HAD_SCALE * __half2float(a.sv1[col + j]);
                // As =1's tgateup (exl3_prompt.cuh) and =2's wup: act_mode 0 rounds gg and uu to bf16 before the clamp.
                if (a.act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
                float act = gg / (1.f + expf(-gg));
                if (a.act_mode == 0) act = bf16r(act);
                act *= uu;
                if (a.act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(a.sd[col + j]);
            }
            tf_exl3::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) a.xd[(size_t)row * a.N + nb * 128 + c0 + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

template<int CB>
__device__ __forceinline__ void gu_task(PmxArgs a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (a.items[item * 3 + 2] <= 64) return; // the preceding =1 gu_s launch owns these items
    const int e = a.items[item * 3];
    const int k2g = a.k2_0[e], k2u = a.k2_1[e];
    if (k2g == 4 && k2u == 4) gu_body<4,4,CB>(a,item,nb,smem,rows_sh);
    else if (k2g == 4 && k2u == 6) gu_body<4,6,CB>(a,item,nb,smem,rows_sh);
    else if (k2g == 6 && k2u == 4) gu_body<6,4,CB>(a,item,nb,smem,rows_sh);
    else if (k2g == 6 && k2u == 6) gu_body<6,6,CB>(a,item,nb,smem,rows_sh);
    else __trap();
}

template<int K2, int CB>
__device__ __forceinline__ void dequant_task(const uint32_t* T, half* out, int N, uint32_t* tile) {
    constexpr int TW = 4 * K2;
    const int kt = TF_VBID.x, nt = TF_VBID.y, lane = threadIdx.x, NT = N >> 4;
    for (int i = lane; i < TW; i += 32) tile[i] = T[((size_t)kt * NT + nt) * TW + i];
    __syncwarp();
    const LaneMap<K2> map(lane); uint32_t af[4]; decode_a<K2,CB>(tile,map,af);
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const half2 h = *reinterpret_cast<const half2*>(&af[q]);
        const int col = nt * 16 + g + 8 * (q & 1), row = kt * 16 + 2 * t + 8 * (q >> 1);
        out[(size_t)row * N + col] = __low2half(h); out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

// decode2<CB_MCG_E4M3> returns f16x2, so the release m16n8k16 fp16 MMA stays. The ring stores e4m3
// activations (half the cp.async bytes of gu_body). One fp16 buffer is rewritten each K stage into the
// swizzle ldmatrix already uses. Seven stages is the most that stays within the sm_89 101376 byte cap
// once that expand buffer is added. Rows stay 128: the epilogue and the cnt<=64 split are gu_body's.
struct PmxA8Args {
    const uint8_t *x0, *x1;
    const float *s0, *s1;
    const long long *tp0, *tp1;
    const int *k2_0, *k2_1;
    const int *items, *counts, *members;
    const half *sv0, *sv1, *sd;
    half *xd;
    float *y;
    int K, N, E;
    float limit;
    int act_mode;
    // gu_w8s only. 0: X row r at byte r * K, its scale at s[r]. Else X row r at byte r * xs of x and its scale at
    // byte r * xs of s (the engine keeps both inside the pair's fp16 slot).
    int xs;
    // gu_w8s only. Nonzero: items of <= 64 members run here too (TF_GLM53_GU_W8S=2; no p5a gate/up launch).
    int all;
};

constexpr int A8_STAGES = 7;
constexpr int A8_X = 2 * BM * 32;
constexpr int A8_W = 2 * KS * NB * TWMAX * (int)sizeof(uint32_t);
constexpr int A8_STAGE = A8_X + A8_W;
constexpr int A8_EXPAND = 2 * BM * 32 * (int)sizeof(half);
constexpr int A8_SMEM = A8_STAGES * A8_STAGE + A8_EXPAND;
static_assert(A8_X == 8192, "e4m3 bytes of both matrices in one K stage");
static_assert(A8_W == 3072, "trellis bytes of one gu_body stage");
static_assert(A8_EXPAND == 16384, "one fp16 stage for ldmatrix");
static_assert(A8_SMEM == 95232, "exl3_pmx.zig gu_a8_smem");
static_assert(A8_STAGE % 16 == 0, "each stage stays 16-byte aligned for cp.async");
static_assert((A8_STAGES * A8_STAGE) % 16 == 0, "the expand buffer stays 16-byte aligned");

__device__ __forceinline__ float a8_unit(uint8_t code) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3));
#else
    return tf_exl3::e4m3_value(code);
#endif
}

template<int K2G, int K2U>
__device__ __forceinline__ void gu_body_a8(const PmxA8Args& a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TWG = 4 * K2G, TWU = 4 * K2U, LDA = 32, W = 16, THREADS = 512;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp >> 3, slice = warp & 7, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int KT = a.K >> 4, NT = a.N >> 4, S = KT / KS;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    unsigned char* expand = smem + (size_t)A8_STAGES * A8_STAGE;
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % A8_STAGES) * A8_STAGE;
        uint8_t* ra = reinterpret_cast<uint8_t*>(stg);
        const int k0 = s * KS * 16;
        for (int i = threadIdx.x; i < 2 * BM * 2; i += THREADS) {
            const int m = i / (BM * 2), r = (i / 2) % BM, ch = i % 2;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const uint8_t* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * a.K + k0 + ch * 16;
                cp_async16(ra + ((size_t)m * BM + r) * 32 + ch * 16, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + A8_X);
        for (int i = threadIdx.x; i < 2 * KS * NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % NB, kk = (i / (6 * NB)) % KS, m = i / (6 * NB * KS);
            const int local_tw = m ? TWU : TWG;
            const uint32_t* src = (m ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : reinterpret_cast<const uint32_t*>(a.tp0[e])) + ((size_t)(s * KS + kk) * NT + nb * NB + n) * local_tw + q * 4;
            const bool valid = q < (m ? K2U : K2G);
            if (valid) cp_async16(rw + ((m * KS + kk) * NB + n) * TWMAX + q * 4, src, true);
        }
    };
    // (r>>1)&3 equals (lr>>1)&3 for a row inside a 16-aligned group, so this is gu_body's swizzle.
    auto expand_stage = [&](int s) {
        const uint8_t* ra = reinterpret_cast<const uint8_t*>(smem + (size_t)(s % A8_STAGES) * A8_STAGE);
        half* dst = reinterpret_cast<half*>(expand);
        const int k0 = s * KS * 16;
        const int blk = k0 >> 7;
        const int blocks = a.K >> 7;
        for (int i = threadIdx.x; i < 2 * BM * 4; i += THREADS) {
            const int m = i / (BM * 4), r = (i / 4) % BM, ch = i % 4;
            if (r >= cnt16) continue;
            half* out = dst + ((size_t)m * BM + r) * LDA + swz<4>(r, ch) * 8;
            const int row = rows_sh[r];
            if (row < 0) {
                *reinterpret_cast<uint4*>(out) = make_uint4(0, 0, 0, 0);
                continue;
            }
            const uint8_t* sm = ra + ((size_t)m * BM + r) * 32 + ch * 8;
            const float scale = __ldg((m ? a.s1 : a.s0) + (size_t)row * blocks + blk);
            uint32_t outb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const half h0 = __float2half_rn(a8_unit(sm[2 * j]) * scale);
                const half h1 = __float2half_rn(a8_unit(sm[2 * j + 1]) * scale);
                outb[j] = static_cast<uint32_t>(__half_as_ushort(h0)) | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
            }
            *reinterpret_cast<uint4*>(out) = make_uint4(outb[0], outb[1], outb[2], outb[3]);
        }
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < A8_STAGES - 1; ++s) { if (s < S) load_stage(s); cp_commit(); }
    cp_wait<A8_STAGES - 2>(); __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + A8_STAGES - 1 < S) load_stage(s + A8_STAGES - 1);
        cp_commit();
        expand_stage(s);
        __syncthreads();
        const half* A = reinterpret_cast<const half*>(expand) + (size_t)mat * BM * LDA;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(smem + (size_t)(s % A8_STAGES) * A8_STAGE + A8_X) + (size_t)mat * KS * NB * TWMAX + slice * TWMAX;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[4];
            if (mat == 0) { const LaneMap<K2G> map(lane); decode_a<K2G, tf_exl3::CB_MCG_E4M3>(Wd + kk * NB * TWMAX, map, af); }
            else { const LaneMap<K2U> map(lane); decode_a<K2U, tf_exl3::CB_MCG_E4M3>(Wd + kk * NB * TWMAX, map, af); }
            const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
#pragma unroll
            for (int np = 0; np < 8; ++np) if (16 * np < cnt) {
                uint32_t b[4]; ldsm4(b, A + (16 * np + lr) * LDA + swz<4>(lr, kk * 2 + lc) * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
                mma16816(acc[2 * np], af, b01); mma16816(acc[2 * np + 1], af, b23);
            }
        }
        cp_wait<A8_STAGES - 2>(); __syncthreads();
    }
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (rd) __syncthreads();
        if (32 * rd < cnt) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int ng = 4 * rd + n;
                float* E = ep + ((size_t)mat * 32 + 8 * n + 2 * t) * 132 + slice * 16 + g;
                E[0] = acc[ng][0]; E[132] = acc[ng][1]; E[8] = acc[ng][2]; E[132 + 8] = acc[ng][3];
            }
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += W) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const int c0 = 4 * lane;
            float v[4]; *reinterpret_cast<float4*>(v) = *reinterpret_cast<float4*>(ep + (size_t)r * 132 + c0);
            tf_exl3::fwht128(v, lane);
            float u[4]; *reinterpret_cast<float4*>(u) = *reinterpret_cast<float4*>(ep + (size_t)(32 * 132 + r * 132 + c0));
            tf_exl3::fwht128(u, lane);
            const size_t col = (size_t)e * a.N + nb * 128 + c0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = v[j] * HAD_SCALE * __half2float(a.sv0[col + j]);
                float uu = u[j] * HAD_SCALE * __half2float(a.sv1[col + j]);
                if (a.act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
                float act = gg / (1.f + expf(-gg));
                if (a.act_mode == 0) act = bf16r(act);
                act *= uu;
                if (a.act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(a.sd[col + j]);
            }
            tf_exl3::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) a.xd[(size_t)row * a.N + nb * 128 + c0 + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

__device__ __forceinline__ void gu_task_a8(const PmxA8Args& a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (a.items[item * 3 + 2] <= 64) return;
    const int e = a.items[item * 3];
    const int k2g = a.k2_0[e], k2u = a.k2_1[e];
    if (k2g == 4 && k2u == 4) gu_body_a8<4, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 4 && k2u == 6) gu_body_a8<4, 6>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 4) gu_body_a8<6, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 6) gu_body_a8<6, 6>(a, item, nb, smem, rows_sh);
    else __trap();
}

// e4m3 x e4m3 gate/up. One stage is one 128-K activation-scale block (four k32 steps).
// Weights: decode2_e4m3_codes, then a quad shuffle into the m16n8k32 A fragment
// (row g, K 4*q..4*q+3 in af[0], row g+8 in af[1], K+16 in af[2] and af[3]).
// That is the Q fragment map in mla.cu. The shuffle is registers, once per 16-K tile.
// Activations stay in rot_in_a8 K order. ldmatrix.x4 reads them the way mla reads a pair of
// k32 steps (lanes 0-7 at byte 0, 8-15 at +16, 16-23 at +32, 24-31 at +48). No rot_in_w8
// permutation: w8a8's byte swap matches an FP8B raw load, and applying it here would
// disagree with this A map.
// Row stride is 128 bytes, one shared-memory bank period, so the chunk slot is c ^ (r & 7).
// The four k32 partials sum in registers, then one fma by that member row's scale.
// Shared memory: 2 * (32768 + 12288) = 90112 dynamic, plus rows[128] = 512. One __syncthreads
// per stage, after the cp.async wait. Release has one per 32 K. Nothing between the four k32 steps.
constexpr int W8_STAGES = 2;
constexpr int W8_K = 128;
constexpr int W8_TILES = W8_K / 16;
constexpr int W8_X = 2 * BM * W8_K;
constexpr int W8_W = 2 * W8_TILES * NB * TWMAX * (int)sizeof(uint32_t);
constexpr int W8_STAGE = W8_X + W8_W;
constexpr int W8_SMEM = W8_STAGES * W8_STAGE;
static_assert(W8_X == 32768, "e4m3 bytes of both matrices in one 128-K stage");
static_assert(W8_W == 12288, "trellis bytes of eight 16-K tiles, both matrices");
static_assert(W8_STAGE == 45056, "one scale block staged");
static_assert(W8_SMEM == 90112, "exl3_pmx.zig gu_w8_smem");
static_assert(W8_SMEM + 128 * (int)sizeof(int) <= 101376, "stage ring plus the member-row table");
static_assert(W8_STAGE % 16 == 0, "stage stays 16-byte aligned for cp.async");

__device__ __forceinline__ int w8_swz(int r, int c) { return c ^ (r & 7); }

__device__ __forceinline__ void mma_w8(float (&d)[4], const uint32_t (&frag)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(frag[0]), "r"(frag[1]), "r"(frag[2]), "r"(frag[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)frag; (void)b0; (void)b1;
    __trap();
#endif
}

__device__ __forceinline__ void mma0_w8(float (&d)[4], const uint32_t (&frag)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
    const float z = 0.0f;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(frag[0]), "r"(frag[1]), "r"(frag[2]), "r"(frag[3]), "r"(b0), "r"(b1), "f"(z));
#else
    (void)d; (void)frag; (void)b0; (void)b1;
    __trap();
#endif
}

// low_pair is K[2q], K[2q+1]. high_pair is K[2q+8], K[2q+9]. Both are this lane's decode pairs
// at one output row of a 16-K tile. The A register wants K[4q]..K[4q+3], which sits across the quad.
__device__ __forceinline__ uint32_t w8_pack4(uint32_t low_pair, uint32_t high_pair, int lane) {
    const int q = lane & 3;
    const int base = lane & ~3;
    const int src = (q & 1) << 1;
    const uint32_t low0 = __shfl_sync(0xffffffffu, low_pair, base | src);
    const uint32_t low1 = __shfl_sync(0xffffffffu, low_pair, base | (src + 1));
    const uint32_t high0 = __shfl_sync(0xffffffffu, high_pair, base | src);
    const uint32_t high1 = __shfl_sync(0xffffffffu, high_pair, base | (src + 1));
    const uint32_t lo = q < 2 ? low0 : high0;
    const uint32_t hi = q < 2 ? low1 : high1;
    return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

template<int K2>
__device__ __forceinline__ void w8_codes(const uint32_t* tile, const LaneMap<K2>& map, uint32_t (&c)[4]) {
    constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
    uint32_t st[8];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const uint32_t whi = tile[map.hi[g]], wlo = tile[map.lo[g]];
        const uint64_t bits = ((((uint64_t)wlo) << 32) | whi) >> map.sh[g];
#pragma unroll
        for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(bits >> Fmt<K2>::off(j)) & 0xffffu;
    }
    c[0] = tf_exl3::decode2_e4m3_codes(st[0], st[1]);
    c[1] = tf_exl3::decode2_e4m3_codes(st[4], st[5]);
    c[2] = tf_exl3::decode2_e4m3_codes(st[2], st[3]);
    c[3] = tf_exl3::decode2_e4m3_codes(st[6], st[7]);
}

template<int K2>
__device__ __forceinline__ void w8_load_afs(const uint32_t* tile0, int lane, uint32_t (&afs)[4][4]) {
    const LaneMap<K2> map(lane);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        uint32_t lo[4], hi[4];
        w8_codes<K2>(tile0 + (size_t)(2 * h) * NB * TWMAX, map, lo);
        w8_codes<K2>(tile0 + (size_t)(2 * h + 1) * NB * TWMAX, map, hi);
        afs[h][0] = w8_pack4(lo[0], lo[2], lane);
        afs[h][1] = w8_pack4(lo[1], lo[3], lane);
        afs[h][2] = w8_pack4(hi[0], hi[2], lane);
        afs[h][3] = w8_pack4(hi[1], hi[3], lane);
    }
}

__device__ __forceinline__ float w8_row_scale(const PmxA8Args& a, const int* rows_sh, int mat, int blocks, int blk, int r, int cnt) {
    if (r >= cnt) return 0.f;
    const int row = rows_sh[r];
    if (row < 0) return 0.f;
    const float* sc = mat ? a.s1 : a.s0;
    return __ldg(sc + (size_t)row * (size_t)blocks + (size_t)blk);
}

template<int K2G, int K2U>
__device__ __forceinline__ void gu_body_w8(const PmxA8Args& a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TWG = 4 * K2G, TWU = 4 * K2U, W = 16, THREADS = 512;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp >> 3, slice = warp & 7, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int NT = a.N >> 4, S = a.K >> 7, blocks = a.K >> 7;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % W8_STAGES) * W8_STAGE;
        uint8_t* xa = reinterpret_cast<uint8_t*>(stg);
        const int k0 = s * W8_K;
        for (int i = threadIdx.x; i < 2 * BM * 8; i += THREADS) {
            const int m = i / (BM * 8), r = (i / 8) % BM, ch = i % 8;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const uint8_t* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * a.K + k0 + ch * 16;
                cp_async16(xa + ((size_t)m * BM + r) * W8_K + w8_swz(r, ch) * 16, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + W8_X);
        for (int i = threadIdx.x; i < 2 * W8_TILES * NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % NB, kk = (i / (6 * NB)) % W8_TILES, m = i / (6 * NB * W8_TILES);
            const int local_tw = m ? TWU : TWG;
            const uint32_t* src = (m ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : reinterpret_cast<const uint32_t*>(a.tp0[e])) + ((size_t)(s * W8_TILES + kk) * NT + nb * NB + n) * local_tw + q * 4;
            const bool valid = q < (m ? K2U : K2G);
            if (valid) cp_async16(rw + ((m * W8_TILES + kk) * NB + n) * TWMAX + q * 4, src, true);
        }
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
        for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < W8_STAGES - 1; ++s) { if (s < S) load_stage(s); cp_commit(); }
    cp_wait<W8_STAGES - 2>();
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + W8_STAGES - 1 < S) load_stage(s + W8_STAGES - 1);
        cp_commit();
        const unsigned char* stg = smem + (size_t)(s % W8_STAGES) * W8_STAGE;
        const uint8_t* Xb = reinterpret_cast<const uint8_t*>(stg) + (size_t)mat * BM * W8_K;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(stg + W8_X) + (size_t)(mat * W8_TILES * NB + slice) * TWMAX;
        uint32_t afs[4][4];
        if (mat == 0) w8_load_afs<K2G>(Wd, lane, afs);
        else w8_load_afs<K2U>(Wd, lane, afs);
#pragma unroll
        for (int grp = 0; grp < 16; ++grp) {
            if (8 * grp >= cnt) continue;
            float part[4];
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int row = 8 * grp + (lane & 7);
                const int chunk = pair * 4 + (lane >> 3);
                uint32_t b[4];
                ldsm4(b, Xb + (size_t)row * W8_K + w8_swz(row, chunk) * 16);
                if (pair == 0) mma0_w8(part, afs[0], b[0], b[1]);
                else mma_w8(part, afs[2], b[0], b[1]);
                mma_w8(part, afs[pair * 2 + 1], b[2], b[3]);
            }
            const int r0 = 8 * grp + 2 * t;
            const int r1 = r0 + 1;
            const float s0 = w8_row_scale(a, rows_sh, mat, blocks, s, r0, cnt);
            const float s1 = w8_row_scale(a, rows_sh, mat, blocks, s, r1, cnt);
            acc[grp][0] = __fmaf_rn(part[0], s0, acc[grp][0]);
            acc[grp][1] = __fmaf_rn(part[1], s1, acc[grp][1]);
            acc[grp][2] = __fmaf_rn(part[2], s0, acc[grp][2]);
            acc[grp][3] = __fmaf_rn(part[3], s1, acc[grp][3]);
        }
        cp_wait<W8_STAGES - 2>();
        __syncthreads();
    }
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (rd) __syncthreads();
        if (32 * rd < cnt) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int ng = 4 * rd + n;
                float* E = ep + ((size_t)mat * 32 + 8 * n + 2 * t) * 132 + slice * 16 + g;
                E[0] = acc[ng][0]; E[132] = acc[ng][1]; E[8] = acc[ng][2]; E[132 + 8] = acc[ng][3];
            }
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += W) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const int c0 = 4 * lane;
            float v[4]; *reinterpret_cast<float4*>(v) = *reinterpret_cast<float4*>(ep + (size_t)r * 132 + c0);
            tf_exl3::fwht128(v, lane);
            float u[4]; *reinterpret_cast<float4*>(u) = *reinterpret_cast<float4*>(ep + (size_t)(32 * 132 + r * 132 + c0));
            tf_exl3::fwht128(u, lane);
            const size_t col = (size_t)e * a.N + nb * 128 + c0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = v[j] * HAD_SCALE * __half2float(a.sv0[col + j]);
                float uu = u[j] * HAD_SCALE * __half2float(a.sv1[col + j]);
                if (a.act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
                float act = gg / (1.f + expf(-gg));
                if (a.act_mode == 0) act = bf16r(act);
                act *= uu;
                if (a.act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(a.sd[col + j]);
            }
            tf_exl3::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) a.xd[(size_t)row * a.N + nb * 128 + c0 + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

__device__ __forceinline__ void gu_task_w8(const PmxA8Args& a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (a.items[item * 3 + 2] <= 64) return;
    const int e = a.items[item * 3];
    const int k2g = a.k2_0[e], k2u = a.k2_1[e];
    if (k2g == 4 && k2u == 4) gu_body_w8<4, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 4 && k2u == 6) gu_body_w8<4, 6>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 4) gu_body_w8<6, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 6) gu_body_w8<6, 6>(a, item, nb, smem, rows_sh);
    else __trap();
}

// GU-W8S: the e4m3 x e4m3 gate/up with nothing but decode, ldmatrix and MMA in the K loop.
// NCU on gu_w8 (GB10): long scoreboard is the top stall, from the two scale loads per row group per stage.
// Here rot_in_w8s gives one scale per (pair, matrix), applied once in the epilogue, so every k32 MMA
// accumulates straight into acc. rot_in_w8s also stores the codes K-permuted inside each 16-K tile
// (byte 4*t+i = K 2*t + (i & 1) + 8 * (i >> 1)), which is one lane's two decode pairs in order:
// the A register is low pair | high pair << 16 with no warp shuffle. Stage ring and shared memory as gu_w8.
template<int K2>
__device__ __forceinline__ void w8s_load_afs(const uint32_t* tile0, int lane, uint32_t (&afs)[4][4]) {
    const LaneMap<K2> map(lane);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        uint32_t lo[4], hi[4];
        w8_codes<K2>(tile0 + (size_t)(2 * h) * NB * TWMAX, map, lo);
        w8_codes<K2>(tile0 + (size_t)(2 * h + 1) * NB * TWMAX, map, hi);
        afs[h][0] = (lo[0] & 0xffffu) | (lo[2] << 16);
        afs[h][1] = (lo[1] & 0xffffu) | (lo[3] << 16);
        afs[h][2] = (hi[0] & 0xffffu) | (hi[2] << 16);
        afs[h][3] = (hi[1] & 0xffffu) | (hi[3] << 16);
    }
}

template<int K2G, int K2U>
__device__ __forceinline__ void gu_body_w8s(const PmxA8Args& a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TWG = 4 * K2G, TWU = 4 * K2U, W = 16, THREADS = 512;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp >> 3, slice = warp & 7, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int NT = a.N >> 4, S = a.K >> 7;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    const size_t xstride = a.xs ? (size_t)a.xs : (size_t)a.K;
    const size_t sstride = a.xs ? (size_t)a.xs : sizeof(float);
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % W8_STAGES) * W8_STAGE;
        uint8_t* xa = reinterpret_cast<uint8_t*>(stg);
        const int k0 = s * W8_K;
        for (int i = threadIdx.x; i < 2 * BM * 8; i += THREADS) {
            const int m = i / (BM * 8), r = (i / 8) % BM, ch = i % 8;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const uint8_t* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * xstride + k0 + ch * 16;
                cp_async16(xa + ((size_t)m * BM + r) * W8_K + w8_swz(r, ch) * 16, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + W8_X);
        for (int i = threadIdx.x; i < 2 * W8_TILES * NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % NB, kk = (i / (6 * NB)) % W8_TILES, m = i / (6 * NB * W8_TILES);
            const int local_tw = m ? TWU : TWG;
            const uint32_t* src = (m ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : reinterpret_cast<const uint32_t*>(a.tp0[e])) + ((size_t)(s * W8_TILES + kk) * NT + nb * NB + n) * local_tw + q * 4;
            const bool valid = q < (m ? K2U : K2G);
            if (valid) cp_async16(rw + ((m * W8_TILES + kk) * NB + n) * TWMAX + q * 4, src, true);
        }
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
        for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < W8_STAGES - 1; ++s) { if (s < S) load_stage(s); cp_commit(); }
    cp_wait<W8_STAGES - 2>();
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + W8_STAGES - 1 < S) load_stage(s + W8_STAGES - 1);
        cp_commit();
        const unsigned char* stg = smem + (size_t)(s % W8_STAGES) * W8_STAGE;
        const uint8_t* Xb = reinterpret_cast<const uint8_t*>(stg) + (size_t)mat * BM * W8_K;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(stg + W8_X) + (size_t)(mat * W8_TILES * NB + slice) * TWMAX;
        uint32_t afs[4][4];
        if (mat == 0) w8s_load_afs<K2G>(Wd, lane, afs);
        else w8s_load_afs<K2U>(Wd, lane, afs);
#pragma unroll
        for (int grp = 0; grp < 16; ++grp) {
            if (8 * grp >= cnt) continue;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int row = 8 * grp + (lane & 7);
                const int chunk = pair * 4 + (lane >> 3);
                uint32_t b[4];
                ldsm4(b, Xb + (size_t)row * W8_K + w8_swz(row, chunk) * 16);
                mma_w8(acc[grp], afs[pair * 2], b[0], b[1]);
                mma_w8(acc[grp], afs[pair * 2 + 1], b[2], b[3]);
            }
        }
        cp_wait<W8_STAGES - 2>();
        __syncthreads();
    }
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (rd) __syncthreads();
        if (32 * rd < cnt) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int ng = 4 * rd + n;
                float* E = ep + ((size_t)mat * 32 + 8 * n + 2 * t) * 132 + slice * 16 + g;
                E[0] = acc[ng][0]; E[132] = acc[ng][1]; E[8] = acc[ng][2]; E[132 + 8] = acc[ng][3];
            }
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += W) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const int c0 = 4 * lane;
            const float q0 = HAD_SCALE * __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(a.s0) + (size_t)row * sstride));
            const float q1 = HAD_SCALE * __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(a.s1) + (size_t)row * sstride));
            float v[4]; *reinterpret_cast<float4*>(v) = *reinterpret_cast<float4*>(ep + (size_t)r * 132 + c0);
            tf_exl3::fwht128(v, lane);
            float u[4]; *reinterpret_cast<float4*>(u) = *reinterpret_cast<float4*>(ep + (size_t)(32 * 132 + r * 132 + c0));
            tf_exl3::fwht128(u, lane);
            const size_t col = (size_t)e * a.N + nb * 128 + c0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = v[j] * q0 * __half2float(a.sv0[col + j]);
                float uu = u[j] * q1 * __half2float(a.sv1[col + j]);
                if (a.act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
                float act = gg / (1.f + expf(-gg));
                if (a.act_mode == 0) act = bf16r(act);
                act *= uu;
                if (a.act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(a.sd[col + j]);
            }
            tf_exl3::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) a.xd[(size_t)row * a.N + nb * 128 + c0 + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

__device__ __forceinline__ void gu_task_w8s(const PmxA8Args& a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (!a.all && a.items[item * 3 + 2] <= 64) return;
    const int e = a.items[item * 3];
    const int k2g = a.k2_0[e], k2u = a.k2_1[e];
    if (k2g == 4 && k2u == 4) gu_body_w8s<4, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 4 && k2u == 6) gu_body_w8s<4, 6>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 4) gu_body_w8s<6, 4>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 6) gu_body_w8s<6, 6>(a, item, nb, smem, rows_sh);
    else __trap();
}
} // namespace tf_glm53_pmx

extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_e4m3(tf_glm53_pmx::PmxArgs a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task<tf_exl3::CB_MCG_E4M3>(a,smem,rows); }
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_mcg(tf_glm53_pmx::PmxArgs a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task<tf_exl3::CB_MCG>(a,smem,rows); }
#define PMX_DQ(K,CB,NAME) extern "C" __global__ void __launch_bounds__(32) tf_glm53_pmx_dequant_##K##_##NAME(const uint32_t* t, half* o, int n) { TF_VBLOCK_ENTER(); __shared__ __align__(16) uint32_t tile[24]; tf_glm53_pmx::dequant_task<K,tf_exl3::CB_##CB>(t,o,n,tile); }
PMX_DQ(4,MCG,mcg) PMX_DQ(6,MCG,mcg) PMX_DQ(4,MCG_E4M3,e4m3) PMX_DQ(6,MCG_E4M3,e4m3)
#undef PMX_DQ

namespace tf_glm53_pmx {
// GU-MB. Same stage bytes, MMA order and epilogue as gu_body. The per-stage CTA barrier is replaced
// by a full/empty mbarrier pair per ring slot so warps are not locked to one phase.
// One stage is 19456 B. Each barrier is 8 B. Slot s%RING holds stage s, lap s/RING.
constexpr size_t MB_STAGE = 2ull * BM * 32 * sizeof(half) + 2ull * KS * NB * TWMAX * sizeof(uint32_t);
constexpr size_t MB_PAIR = 16;
static_assert(MB_STAGE == 19456ull, "one gu_body stage");
static_assert(5 * MB_STAGE + 5 * MB_PAIR == 97360ull, "exl3_pmx.zig gu_mb5_smem");
static_assert(5 * MB_STAGE + 5 * MB_PAIR + 128 * sizeof(int) <= 101376ull, "five slots plus barriers and the row table");

__device__ __forceinline__ void mb_init(uint64_t* bar, int count) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}

// CUTLASS PipelineState (sm90_pipeline.hpp make_producer_start_state): parity 0 blocks until the
// first completion, parity 1 does not block on a fresh barrier, and each completion flips that.
__device__ __forceinline__ void mb_wait(uint64_t* bar, unsigned parity) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    const unsigned ticks = 0x989680u;
    unsigned ready = 0u;
    while (ready == 0u) {
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2, %3;\n\t"
            "selp.b32 %0, 1, 0, P1;\n\t"
            "}\n"
            : "=&r"(ready)
            : "r"(addr), "r"(parity), "r"(ticks)
            : "memory");
    }
}

// Every lane executes this. Only lane 0 of the warp arrives, so empty's count is 16.
// mma.sync before the call has already converged the warp, so the arrive is after that warp's reads.
__device__ __forceinline__ void mb_arrive_lane(uint64_t* bar, int do_arrive) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %1, 0;\n\t"
        "@p mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}\n"
        :: "r"(addr), "r"(do_arrive) : "memory");
}

__device__ __forceinline__ void mb_cp_arrive(uint64_t* bar) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(addr) : "memory");
}

template<int K2G, int K2U, int CB, int RING, int LEAD>
__device__ __forceinline__ void gu_body_mb(const PmxArgs a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TWG = 4 * K2G, TWU = 4 * K2U, LDA = 32, W = 16, THREADS = 512;
    constexpr int FULL_COUNT = THREADS;
    constexpr int EMPTY_COUNT = THREADS / 32;
    constexpr size_t ABYTES = 2ull * BM * LDA * sizeof(half);
    constexpr size_t WBYTES = 2ull * KS * NB * TWMAX * sizeof(uint32_t);
    constexpr size_t STAGE_BYTES = ABYTES + WBYTES;
    static_assert(STAGE_BYTES == MB_STAGE, "stage layout matches gu_body");
    // LEAD is the prefetch distance in stages. Producing stage s+LEAD reuses the slot of stage s+LEAD-RING,
    // so a warp may run RING-LEAD-1 whole stages ahead of the slowest warp. LEAD = RING-1 would be lockstep.
    static_assert(LEAD >= 1 && LEAD < RING - 1, "at least one stage of slack between warps");
    static_assert(FULL_COUNT == 512, "every thread arrives on full");
    static_assert(EMPTY_COUNT == 16, "one lane of each warp arrives on empty");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp >> 3, slice = warp & 7, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int KT = a.K >> 4, NT = a.N >> 4, S = KT / KS;
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + (size_t)RING * STAGE_BYTES);
    uint64_t* empty = full + RING;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < RING; ++slot) {
            mb_init(full + slot, FULL_COUNT);
            mb_init(empty + slot, EMPTY_COUNT);
        }
    }
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % RING) * STAGE_BYTES;
        half* ra = reinterpret_cast<half*>(stg);
        const int k0 = s * KS * 16;
        for (int i = threadIdx.x; i < 2 * BM * 4; i += THREADS) {
            const int m = i / (BM * 4), r = (i / 4) % BM, ch = i % 4;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const half* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * a.K + k0 + ch * 8;
                cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<4>(r, ch) * 8, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + ABYTES);
        for (int i = threadIdx.x; i < 2 * KS * NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % NB, kk = (i / (6 * NB)) % KS, m = i / (6 * NB * KS);
            const int local_tw = m ? TWU : TWG;
            const uint32_t* src = (m ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : reinterpret_cast<const uint32_t*>(a.tp0[e])) + ((size_t)(s * KS + kk) * NT + nb * NB + n) * local_tw + q * 4;
            const bool valid = q < (m ? K2U : K2G);
            if (valid) cp_async16(rw + ((m * KS + kk) * NB + n) * TWMAX + q * 4, src, true);
        }
    };
    // commit_group puts the copies in a group so the epilogue cp_wait<0> can see them.
    // The arrive still fires only after this thread's prior cp.async ops have landed.
    auto produce = [&](int s) {
        if (s >= S) return;
        const int slot = s % RING;
        const int use = s / RING;
        if (use > 0) mb_wait(empty + slot, static_cast<unsigned>((use - 1) & 1));
        load_stage(s);
        cp_commit();
        mb_cp_arrive(full + slot);
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < LEAD; ++s) produce(s);
    for (int s = 0; s < S; ++s) {
        produce(s + LEAD);
        const int slot = s % RING;
        const int use = s / RING;
        mb_wait(full + slot, static_cast<unsigned>(use & 1));
        const unsigned char* stg = smem + (size_t)slot * STAGE_BYTES;
        const half* A = reinterpret_cast<const half*>(stg) + (size_t)mat * BM * LDA;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(stg + ABYTES) + (size_t)mat * KS * NB * TWMAX + slice * TWMAX;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[4];
            if (mat == 0) { const LaneMap<K2G> map(lane); decode_a<K2G, CB>(Wd + kk * NB * TWMAX, map, af); }
            else { const LaneMap<K2U> map(lane); decode_a<K2U, CB>(Wd + kk * NB * TWMAX, map, af); }
            const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
#pragma unroll
            for (int np = 0; np < 8; ++np) if (16 * np < cnt) {
                uint32_t b[4]; ldsm4(b, A + (16 * np + lr) * LDA + swz<4>(lr, kk * 2 + lc) * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
                mma16816(acc[2 * np], af, b01); mma16816(acc[2 * np + 1], af, b23);
            }
        }
        mb_arrive_lane(empty + slot, lane == 0);
    }
    cp_wait<0>();
    __syncthreads();
    // Four rounds spill 32 member rows at a time into the shared ring, as in x3gm.
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (rd) __syncthreads();
        // As x3gm's epilogue: round rd holds accumulator groups 4*rd..4*rd+3 (member rows 32*rd..32*rd+31); c0/c2
        // are member 2t at columns g / g+8, c1/c3 member 2t+1 (the next ep row, +132).
        if (32 * rd < cnt) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int ng = 4 * rd + n;
                float* E = ep + ((size_t)mat * 32 + 8 * n + 2 * t) * 132 + slice * 16 + g;
                E[0] = acc[ng][0]; E[132] = acc[ng][1]; E[8] = acc[ng][2]; E[132 + 8] = acc[ng][3];
            }
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += W) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const int c0 = 4 * lane;
            float v[4]; *reinterpret_cast<float4*>(v) = *reinterpret_cast<float4*>(ep + (size_t)r * 132 + c0);
            tf_exl3::fwht128(v, lane);
            float u[4]; *reinterpret_cast<float4*>(u) = *reinterpret_cast<float4*>(ep + (size_t)(32 * 132 + r * 132 + c0));
            tf_exl3::fwht128(u, lane);
            const size_t col = (size_t)e * a.N + nb * 128 + c0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = v[j] * HAD_SCALE * __half2float(a.sv0[col + j]);
                float uu = u[j] * HAD_SCALE * __half2float(a.sv1[col + j]);
                // As =1's tgateup (exl3_prompt.cuh) and =2's wup: act_mode 0 rounds gg and uu to bf16 before the clamp.
                if (a.act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
                float act = gg / (1.f + expf(-gg));
                if (a.act_mode == 0) act = bf16r(act);
                act *= uu;
                if (a.act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(a.sd[col + j]);
            }
            tf_exl3::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) a.xd[(size_t)row * a.N + nb * 128 + c0 + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

template<int CB, int RING, int LEAD>
__device__ __forceinline__ void gu_task_mb(PmxArgs a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (a.items[item * 3 + 2] <= 64) return;
    const int e = a.items[item * 3];
    const int k2g = a.k2_0[e], k2u = a.k2_1[e];
    if (k2g == 4 && k2u == 4) gu_body_mb<4, 4, CB, RING, LEAD>(a, item, nb, smem, rows_sh);
    else if (k2g == 4 && k2u == 6) gu_body_mb<4, 6, CB, RING, LEAD>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 4) gu_body_mb<6, 4, CB, RING, LEAD>(a, item, nb, smem, rows_sh);
    else if (k2g == 6 && k2u == 6) gu_body_mb<6, 6, CB, RING, LEAD>(a, item, nb, smem, rows_sh);
    else __trap();
}
} // namespace tf_glm53_pmx

// TF_GLM53_GU_MB=1: 5 ring slots, prefetch lead 2 (kernel bench glm53-gmb: 0.92x of tf_glm53_pmx_gu_e4m3, same bits).
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_mb5l2_e4m3(tf_glm53_pmx::PmxArgs a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task_mb<tf_exl3::CB_MCG_E4M3, 5, 2>(a,smem,rows); }
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_a8_e4m3(tf_glm53_pmx::PmxA8Args a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task_a8(a,smem,rows); }
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_w8_e4m3(tf_glm53_pmx::PmxA8Args a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task_w8(a,smem,rows); }
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_w8s_e4m3(tf_glm53_pmx::PmxA8Args a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task_w8s(a,smem,rows); }

namespace tf_glm53_pmx {
// DOWN-W8S (TF_GLM53_DOWN_W8S=1, only with TF_GLM53_GU_W8S): the routed down of items > 64 members as e4m3 x e4m3,
// the GU-W8S recipe applied to down. Two kernels on the routed stream after gate/up:
//  1. tf_glm53_pmx_xd_w8s: one CTA a pair. Reads the pair's fp16 xd row (K = width values), one fp32 scale
//     amax / 448 (1 when all zero) for the whole row, e4m3 codes (RN, satfinite) of x / scale stored K-permuted
//     inside every 16-K tile exactly as rot_in_w8s stores gate/up codes (byte 4t+i = K 2t + (i & 1) + 8 (i >> 1)),
//     row p at byte p * xs of out, its scale at byte p * xs + K. Pairs with pick < 0 (other rank's under EP) skip.
//  2. tf_glm53_pmx_down_w8s_e4m3: gu_body_w8s with one matrix. 16 warps x one 16-column n-tile = 256 output
//     columns a CTA, grid (D / 256, items). Weights are the MMA A operand from w8_codes (K-permuted fragments, no
//     shuffle), codes the B operand by ldmatrix, every k32 MMA accumulates straight into acc. Epilogue as wdown:
//     y[pair][col] = fwht128(acc) * HAD_SCALE * svh_d[e][col], times the pair's scale.
constexpr int DW_NB = 16;
constexpr int DW_X = BM * W8_K;
constexpr int DW_W = W8_TILES * DW_NB * TWMAX * (int)sizeof(uint32_t);
constexpr int DW_STAGE = DW_X + DW_W;
constexpr int DW_STAGES = 3;
constexpr int DW_SMEM = DW_STAGES * DW_STAGE;
constexpr int DW_ES = 260;
static_assert(DW_X == 16384 && DW_W == 12288 && DW_STAGE == 28672, "one 128-K down stage");
static_assert(DW_SMEM == 86016, "exl3_pmx.zig down_w8_smem");
static_assert(DW_SMEM + 128 * (int)sizeof(int) <= 101376, "stage ring plus the member-row table");
static_assert(32 * DW_ES * (int)sizeof(float) <= DW_SMEM, "epilogue tile fits the ring");
static_assert(DW_STAGE % 16 == 0 && DW_X % 16 == 0, "16-byte cp.async slots");

struct DownW8Args {
    const uint8_t* xq;     // codes: row p at byte p * xs, its fp32 scale at byte p * xs + K
    const long long* td;   // per-expert down trellis
    const int* k2;         // per-expert down K2 (4 or 6)
    const int *items, *counts, *members;
    const half* svh;       // fp16 [E][D]
    float* y;              // fp32 [pairs][D]
    int K, D, xs;
};

template<int K2, int NBS>
__device__ __forceinline__ void w8s_load_afs_n(const uint32_t* tile0, int lane, uint32_t (&afs)[4][4]) {
    const LaneMap<K2> map(lane);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        uint32_t lo[4], hi[4];
        w8_codes<K2>(tile0 + (size_t)(2 * h) * NBS * TWMAX, map, lo);
        w8_codes<K2>(tile0 + (size_t)(2 * h + 1) * NBS * TWMAX, map, hi);
        afs[h][0] = (lo[0] & 0xffffu) | (lo[2] << 16);
        afs[h][1] = (lo[1] & 0xffffu) | (lo[3] << 16);
        afs[h][2] = (hi[0] & 0xffffu) | (hi[2] << 16);
        afs[h][3] = (hi[1] & 0xffffu) | (hi[3] << 16);
    }
}

template<int K2>
__device__ __forceinline__ void down_body_w8s(const DownW8Args& a, int item, int nb, unsigned char* smem, int* rows_sh) {
    constexpr int TW = 4 * K2, THREADS = 512;
    static_assert(TW <= TWMAX, "a tile fits its stage slot");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int e = a.items[item * 3], off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    const int NT = a.D >> 4, S = a.K >> 7;
    for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.members[off + i] : -1;
    __syncthreads();
    const int cnt16 = (cnt + 15) & ~15;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(a.td[e]);
    auto load_stage = [&](int s) {
        unsigned char* stg = smem + (size_t)(s % DW_STAGES) * DW_STAGE;
        const int k0 = s * W8_K;
        for (int i = threadIdx.x; i < BM * 8; i += THREADS) {
            const int r = i >> 3, ch = i & 7;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const uint8_t* src = a.xq + (size_t)(row < 0 ? 0 : row) * a.xs + k0 + ch * 16;
                cp_async16(stg + (size_t)r * W8_K + w8_swz(r, ch) * 16, src, row >= 0);
            }
        }
        uint32_t* rw = reinterpret_cast<uint32_t*>(stg + DW_X);
        for (int i = threadIdx.x; i < W8_TILES * DW_NB * 6; i += THREADS) {
            const int q = i % 6, n = (i / 6) % DW_NB, kk = i / (6 * DW_NB);
            if (q < K2) cp_async16(rw + (kk * DW_NB + n) * TWMAX + q * 4, T + ((size_t)(s * W8_TILES + kk) * NT + nb * DW_NB + n) * TW + q * 4, true);
        }
    };
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
        for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int s = 0; s < DW_STAGES - 1; ++s) { if (s < S) load_stage(s); cp_commit(); }
    cp_wait<DW_STAGES - 2>();
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + DW_STAGES - 1 < S) load_stage(s + DW_STAGES - 1);
        cp_commit();
        const unsigned char* stg = smem + (size_t)(s % DW_STAGES) * DW_STAGE;
        const uint8_t* Xb = reinterpret_cast<const uint8_t*>(stg);
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(stg + DW_X) + (size_t)warp * TWMAX;
        uint32_t afs[4][4];
        w8s_load_afs_n<K2, DW_NB>(Wd, lane, afs);
#pragma unroll
        for (int grp = 0; grp < 16; ++grp) {
            if (8 * grp >= cnt) continue;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int row = 8 * grp + (lane & 7);
                const int chunk = pair * 4 + (lane >> 3);
                uint32_t b[4];
                ldsm4(b, Xb + (size_t)row * W8_K + w8_swz(row, chunk) * 16);
                mma_w8(acc[grp], afs[pair * 2], b[0], b[1]);
                mma_w8(acc[grp], afs[pair * 2 + 1], b[2], b[3]);
            }
        }
        cp_wait<DW_STAGES - 2>();
        __syncthreads();
    }
    float* ep = reinterpret_cast<float*>(smem);
    for (int rd = 0; rd < 4; ++rd) {
        if (32 * rd >= cnt) break;
        if (rd) __syncthreads();
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int ng = 4 * rd + n;
            float* E = ep + (size_t)(8 * n + 2 * t) * DW_ES + warp * 16 + g;
            E[0] = acc[ng][0]; E[DW_ES] = acc[ng][1]; E[8] = acc[ng][2]; E[DW_ES + 8] = acc[ng][3];
        }
        __syncthreads();
        for (int r = warp; r < 32 && rd * 32 + r < cnt; r += 16) {
            const int row = rows_sh[rd * 32 + r];
            if (row < 0) continue;
            const float xsc = HAD_SCALE * __ldg(reinterpret_cast<const float*>(a.xq + (size_t)row * a.xs + a.K));
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                float v[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * DW_ES + blk * 128 + 4 * lane);
                tf_exl3::fwht128(v, lane);
                const int col = nb * 256 + blk * 128 + 4 * lane;
                float* o = a.y + (size_t)row * a.D + col;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * xsc * __half2float(a.svh[(size_t)e * a.D + col + j]);
            }
        }
    }
}

__device__ __forceinline__ void down_task_w8s(const DownW8Args& a, unsigned char* smem, int* rows_sh) {
    const int nb = TF_VBID.x, item = TF_VBID.y;
    if (item >= a.counts[0]) return;
    if (a.items[item * 3 + 2] <= 64) return;
    const int k2 = a.k2[a.items[item * 3]];
    if (k2 == 4) down_body_w8s<4>(a, item, nb, smem, rows_sh);
    else if (k2 == 6) down_body_w8s<6>(a, item, nb, smem, rows_sh);
    else __trap();
}

// One CTA (128 threads) a pair; thread t quantises 16-K tiles t and t + 128 (K <= 4096).
__device__ __forceinline__ void xd_w8s_task(const half* __restrict__ xd, const int* __restrict__ pick, uint8_t* __restrict__ out, int K, int xs, float* red) {
    const int p = TF_VBID.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (pick[p] < 0) return;
    const int tiles = K >> 4;
    float v[2][16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int tt = tid + 128 * i;
#pragma unroll
        for (int j = 0; j < 16; ++j) v[i][j] = 0.0f;
        if (tt < tiles) {
            const uint4* src = reinterpret_cast<const uint4*>(xd + (size_t)p * K + tt * 16);
            const uint4 a0 = src[0], a1 = src[1];
            const half* h0 = reinterpret_cast<const half*>(&a0);
            const half* h1 = reinterpret_cast<const half*>(&a1);
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[i][j] = __half2float(h0[j]); v[i][8 + j] = __half2float(h1[j]); }
#pragma unroll
            for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(v[i][j]));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) red[warp] = amax;
    __syncthreads();
    const float all = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float scale = all > 0.0f ? all / 448.0f : 1.0f;
    const float inv = 1.0f / scale;
    uint8_t* row = out + (size_t)p * xs;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int tt = tid + 128 * i;
        if (tt < tiles) {
            uint32_t w[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t c0 = __nv_cvt_float_to_fp8(v[i][2 * q] * inv, __NV_SATFINITE, __NV_E4M3);
                const uint32_t c1 = __nv_cvt_float_to_fp8(v[i][2 * q + 1] * inv, __NV_SATFINITE, __NV_E4M3);
                const uint32_t c2 = __nv_cvt_float_to_fp8(v[i][2 * q + 8] * inv, __NV_SATFINITE, __NV_E4M3);
                const uint32_t c3 = __nv_cvt_float_to_fp8(v[i][2 * q + 9] * inv, __NV_SATFINITE, __NV_E4M3);
                w[q] = (c0 & 0xffu) | ((c1 & 0xffu) << 8) | ((c2 & 0xffu) << 16) | ((c3 & 0xffu) << 24);
            }
            *reinterpret_cast<uint4*>(row + tt * 16) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    if (tid == 0) *reinterpret_cast<float*>(row + K) = scale;
}
} // namespace tf_glm53_pmx

extern "C" __global__ void __launch_bounds__(512, 1) tf_glm53_pmx_down_w8s_e4m3(tf_glm53_pmx::DownW8Args a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::down_task_w8s(a, smem, rows); }
extern "C" __global__ void __launch_bounds__(128) tf_glm53_pmx_xd_w8s(const half* __restrict__ xd, const int* __restrict__ pick, uint8_t* __restrict__ out, int K, int xs) { TF_VBLOCK_ENTER(); __shared__ float red[4]; tf_glm53_pmx::xd_w8s_task(xd, pick, out, K, xs, red); }
