// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 width-specialised EXL3 prompt experts. Structure ported from this repo's zig/kernels/cuda/exl3_block/x3gm.cu (TensorFold deepseek_v41 x3gm v2). Per-width (K2/K3) specialisation after greatyingzi/GLM-5.3-Flash-EXL3-3bit-TensorFold k3-kernel/exl3k3.cu (Apache-2.0); expert-major prompt-expert design after MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patches 0106-glm-full-prompt-experts (drowzeys) and 0108-glm-full-mia-prompt-experts (Apache-2.0, Copyright 2026 Mia's AI Lab). Independent CUDA; no upstream code copied. EXL3 format after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp); mcg_e4m3 decode after bertholomus/TensorFold@ddfad35 (Apache-2.0).
#include "../tf_vblock.cuh"
#include "../exl3_decode.cuh"
#include "../exl3_block/experts_grouped.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
} // namespace tf_glm53_pmx

extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_e4m3(tf_glm53_pmx::PmxArgs a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task<tf_exl3::CB_MCG_E4M3>(a,smem,rows); }
extern "C" __global__ void __launch_bounds__(512,1) tf_glm53_pmx_gu_mcg(tf_glm53_pmx::PmxArgs a) { TF_VBLOCK_ENTER(); extern __shared__ __align__(16) unsigned char smem[]; __shared__ int rows[128]; tf_glm53_pmx::gu_task<tf_exl3::CB_MCG>(a,smem,rows); }
#define PMX_DQ(K,CB,NAME) extern "C" __global__ void __launch_bounds__(32) tf_glm53_pmx_dequant_##K##_##NAME(const uint32_t* t, half* o, int n) { TF_VBLOCK_ENTER(); __shared__ __align__(16) uint32_t tile[24]; tf_glm53_pmx::dequant_task<K,tf_exl3::CB_##CB>(t,o,n,tile); }
PMX_DQ(4,MCG,mcg) PMX_DQ(6,MCG,mcg) PMX_DQ(4,MCG_E4M3,e4m3) PMX_DQ(6,MCG_E4M3,e4m3)
#undef PMX_DQ
