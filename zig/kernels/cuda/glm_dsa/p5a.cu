// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 TF_GLM53_PROMPT_EXPERTS=10: =3's small-item (<= 64-member) routed prompt experts with a compile-time trellis
// width and member count, hoisted cp.async addressing and the next k16 step's decode issued ahead of the current MMAs.
// Every output keeps the bits of tf_exl3_prompt_gu_s / tf_exl3_prompt_down_s (TF_GLM53_PROMPT_EXPERTS=1 and 3).
// Structure after @BobClawblaw (#976) and MiaAI-Lab (Apache-2.0, patches 0004 and 0009), as exl3_prompt.cuh. EXL3 format
// and trellis decode after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp); mcg_e4m3 decode after
// bertholomus/TensorFold@ddfad35 (Apache-2.0); the paired E4M3 conversion as in gr8 (MIT, see THIRD_PARTY_NOTICES.md).
// Ours otherwise; no upstream code copied.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "../tf_vblock.cuh"
#include "../exl3_decode.cuh"
namespace tf_exl3 {
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float upTo(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }
}
#include "../exl3_prompt.cuh"
namespace tf_glm53_p5a {
using namespace tf_exl3;
using namespace tf_exl3_prompt;
constexpr int SMALL_MAX = TPM;
constexpr int SMEM = tsmem_bytes();
static_assert(SMEM == 43008, "exl3_p5a.zig smem_static");
static_assert(TT == 256 && TMT == 2 && TKT == 2 && TSTAGES == 3 && TPH == 32, "the =1 tile geometry");
static_assert(16 * TKT * 8 == TT, "one weight cp.async slot a thread");
__host__ __device__ constexpr bool pipe_for(int nb8) { return nb8 > 0; }

template <int CB> __device__ __forceinline__ uint32_t decode2_nv(uint32_t s0, uint32_t s1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (CB == CB_MCG_E4M3) {
        const uint32_t out = decode2<CB_MCG>(s0, s1);
        uint16_t packed; uint32_t restored;
        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(packed) : "r"(out));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(restored) : "h"(packed));
        return restored;
    }
#endif
    return decode2<CB>(s0, s1);
}
__device__ __forceinline__ void mma_nv(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int K2, int CB> __device__ __forceinline__ void decode_a(const uint32_t* tile, int lane, uint32_t (&a)[4]) {
    uint32_t w[lane_words<K2>()]; int word, offset; lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
    uint32_t s[8]; lane_states<K2>(w, lane, s);
    a[0] = decode2_nv<CB>(s[0], s[1]); a[1] = decode2_nv<CB>(s[4], s[5]);
    a[2] = decode2_nv<CB>(s[2], s[3]); a[3] = decode2_nv<CB>(s[6], s[7]);
}
template <int K2, int CB> __device__ __forceinline__ void decode_w(const uint32_t* tile, int k2, int lane, uint32_t (&a)[4]) {
    if constexpr (K2 != 0) decode_a<K2, CB>(tile, lane, a);
    else switch (k2) {
        case 2: decode_a<2, CB>(tile, lane, a); break; case 3: decode_a<3, CB>(tile, lane, a); break;
        case 4: decode_a<4, CB>(tile, lane, a); break; case 5: decode_a<5, CB>(tile, lane, a); break;
        case 6: decode_a<6, CB>(tile, lane, a); break; case 7: decode_a<7, CB>(tile, lane, a); break;
        case 8: decode_a<8, CB>(tile, lane, a); break;
        default: a[0] = a[1] = a[2] = a[3] = 0;
    }
}
template <int K2, int NB8, int CB> __device__ __forceinline__ void pstage(const half* xs, const uint32_t* ws, int k2, int lane, float (&acc)[TMT][NB8][4]) {
    constexpr int NB16 = (NB8 + 1) / 2; constexpr bool PIPE = pipe_for(NB8);
    const int row = (lane & 7) + ((lane >> 4) & 1) * 8, col = ((lane >> 3) & 1) * 8;
    uint32_t a[2][TMT][4];
#pragma unroll
    for (int i = 0; i < TMT; ++i) decode_w<K2, CB>(ws + (size_t)(i * TKT) * 32, k2, lane, a[0][i]);
#pragma unroll
    for (int kk = 0; kk < TKT; ++kk) {
        uint32_t b[2][4]; ldmatrix_x4(b[0], xs + (size_t)row * TXS + kk * 16 + col);
        if (PIPE && kk + 1 < TKT) {
#pragma unroll
            for (int i = 0; i < TMT; ++i) decode_w<K2, CB>(ws + (size_t)(i * TKT + kk + 1) * 32, k2, lane, a[(kk + 1) & 1][i]);
        }
#pragma unroll
        for (int jb = 0; jb < NB16; ++jb) {
            if (jb + 1 < NB16) ldmatrix_x4(b[(jb + 1) & 1], xs + (size_t)((jb + 1) * 16 + row) * TXS + kk * 16 + col);
            const uint32_t bl[2] = {b[jb & 1][0], b[jb & 1][1]}, bh[2] = {b[jb & 1][2], b[jb & 1][3]};
#pragma unroll
            for (int i = 0; i < TMT; ++i) {
                mma_nv(acc[i][2 * jb], a[kk & 1][i], bl);
                if (2 * jb + 1 < NB8) mma_nv(acc[i][2 * jb + 1], a[kk & 1][i], bh);
            }
        }
        if (!PIPE && kk + 1 < TKT) {
#pragma unroll
            for (int i = 0; i < TMT; ++i) decode_w<K2, CB>(ws + (size_t)(i * TKT + kk + 1) * 32, k2, lane, a[(kk + 1) & 1][i]);
        }
    }
}

template <int MATS, int XR, int NB8, int CB> __device__ __forceinline__ void pmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, const int* rows_sh, int K, int NTILES, int cb, int k2a, int k2b, half* stage, float (&acc)[TMT][NB8][4]) {
    static_assert(XR == TPM || XR == 2 * TPM); constexpr int SH = tstage_halves<XR>(); constexpr int NB16 = (NB8 + 1) / 2;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x; int mat, nt_own;
    ttile<MATS>(warp * TMT, cb, mat, nt_own); const int k2w = mat ? k2b : k2a; const int chunks = K / TBK;
#pragma unroll
    for (int i = 0; i < TMT; ++i) for (int j = 0; j < NB8; ++j) for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    const int xi = tid >> 2, xj = tid & 3; const bool xload = xi < NB16 * 16; const int xr = rows_sh[xi]; const bool xvalid = xr >= 0;
    const uint32_t xoff = (uint32_t)(xvalid ? xr : 0) * (uint32_t)K + (uint32_t)(xj * 8); const int xdst = xi * TXS + xj * 8;
    const int wq = tid / (TKT * 8), wkk = (tid / 8) % TKT, wpart = tid % 8; int wm, wnt; ttile<MATS>(wq, cb, wm, wnt);
    const int wk2 = wm ? k2b : k2a; const bool wload = wpart < wk2;
    const uint32_t* wsrc = (wm ? T1 : T0) + ((size_t)wkk * NTILES + wnt) * (4 * wk2) + wpart * 4;
    const size_t wstep = (size_t)TKT * NTILES * (4 * wk2); const int wdst = (wq * TKT + wkk) * 32 + wpart * 4;
    auto load = [&](int chunk) { half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        if (xload) { const size_t k0 = (size_t)chunk * TBK; cp_async16(dst + xdst, X0 + xoff + k0, xvalid); if constexpr (XR == 2 * TPM) cp_async16(dst + TPM * TXS + xdst, X1 + xoff + k0, xvalid); }
        if (wload) cp_async16(reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS) + wdst, wsrc + (size_t)chunk * wstep, true);
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) { if (c < chunks) load(c); cp_async_commit(); }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>(); TF_TASK_SYNC(); if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1); cp_async_commit();
        half* base = stage + (size_t)(c % TSTAGES) * SH; const half* xs = base + (XR == TPM ? 0 : mat) * TPM * TXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + XR * TXS) + warp * TMT * TKT * 32;
        if (k2w == 4) pstage<4, NB8, CB>(xs, ws, 4, lane, acc); else if (k2w == 6) pstage<6, NB8, CB>(xs, ws, 6, lane, acc); else pstage<0, NB8, CB>(xs, ws, k2w, lane, acc);
    }
    cp_async_wait<0>(); TF_TASK_SYNC();
}

template <int NB8, int HF> __device__ __forceinline__ void pspill(float* E, int ES, int col0, const float (&acc)[TMT][NB8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i) for (int q = 0; q < TPM / 16; ++q) { const int nb = HF * (TPM / 16) + q, tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
        if (nb < NB8) { E[(size_t)tok * ES + col] = acc[i][nb][0]; E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1]; E[(size_t)tok * ES + col + 8] = acc[i][nb][2]; E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3]; }
    }
}

template <int HF, int NB8> __device__ __forceinline__ void gu_half(const float (&acc)[TMT][NB8][4], float* Eg, int mat, int col0, int cnt, int e, int n, const int* rows_sh, const half* svh_g, const half* svh_u, const half* suh_d, half* xd, int N, float limit, int act_mode) {
    if (HF) TF_TASK_SYNC(); pspill<NB8, HF>(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc); TF_TASK_SYNC(); const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = warp; i < TPH && HF * TPH + i < cnt; i += TNW) { const int p = rows_sh[HF * TPH + i]; float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j]; uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j]; }
        fwht128(gv, lane); fwht128(uv, lane); float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]); float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]); if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); } if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); } float act = gg / (1.f + expf(-gg)); if (act_mode == 0) act = bf16r(act); act *= uu; if (act_mode == 0) act = bf16r(act); v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]); }
        fwht128(v, lane); half* o = xd + (size_t)p * N + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
    }
}
template <int NB8, int CB> __device__ __forceinline__ void gu_body(const half* Xg, const half* Xu, const uint64_t* Tg, const uint64_t* Tu, int cnt, int e, const half* svh_g, const half* svh_u, const half* suh_d, half* xd, int K, int N, float limit, int act_mode, int k2g_e, int k2u_e, unsigned char* smem_raw, const int* rows_sh) {
    const int cb = TF_VBID.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31; float acc[TMT][NB8][4];
    pmain<2, 2 * TPM, NB8, CB>(Xg, Xu, reinterpret_cast<const uint32_t*>(Tg[e]), reinterpret_cast<const uint32_t*>(Tu[e]), rows_sh, K, N >> 4, cb, k2g_e, k2u_e, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw); int mat, nt; ttile<2>(warp * TMT, cb, mat, nt); const int col0 = (nt - cb * 8) * 16, n = cb * 128 + 4 * lane;
    gu_half<0, NB8>(acc, Eg, mat, col0, cnt, e, n, rows_sh, svh_g, svh_u, suh_d, xd, N, limit, act_mode); if (TPH < cnt) gu_half<1, NB8>(acc, Eg, mat, col0, cnt, e, n, rows_sh, svh_g, svh_u, suh_d, xd, N, limit, act_mode);
}
template <int HF, int NB8> __device__ __forceinline__ void down_half(const float (&acc)[TMT][NB8][4], float* Ed, int cb, int cnt, int e, const int* rows_sh, const half* svh_d, float* y, int D) {
    if (HF) TF_TASK_SYNC(); const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31; pspill<NB8, HF>(Ed, DN_ES, warp * TMT * 16, acc); TF_TASK_SYNC();
    for (int i = warp; i < TPH && HF * TPH + i < cnt; i += TNW) { const int p = rows_sh[HF * TPH + i];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) { const int n = cb * 256 + blk * 128 + 4 * lane; float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j]; fwht128(v, lane); float* o = y + (size_t)p * D + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
        }
    }
}
template <int NB8, int CB> __device__ __forceinline__ void down_body(const half* Xd, const uint64_t* Td, int cnt, int e, const half* svh_d, float* y, int K, int D, int k2, unsigned char* smem_raw, const int* rows_sh) {
    const int cb = TF_VBID.x, warp = threadIdx.x >> 5; float acc[TMT][NB8][4];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(Td[e]);
    pmain<1, TPM, NB8, CB>(Xd, Xd, T, T, rows_sh, K, D >> 4, cb, k2, k2, reinterpret_cast<half*>(smem_raw), acc); float* Ed = reinterpret_cast<float*>(smem_raw);
    down_half<0>(acc, Ed, cb, cnt, e, rows_sh, svh_d, y, D); if (TPH < cnt) down_half<1>(acc, Ed, cb, cnt, e, rows_sh, svh_d, y, D);
}
template <int CB> __device__ __forceinline__ void gu_entry(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ k2g, const int* __restrict__ k2u) {
    __shared__ __align__(16) unsigned char smem_raw[SMEM]; __shared__ int rows_sh[TPM];
    const int item = TF_VBID.y; if (item >= counts[0]) return; const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (cnt > SMALL_MAX) return; if (xdiv != 1) __trap();
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1; TF_TASK_SYNC();
    const int k2g_e = k2g[e], k2u_e = k2u[e];
    // One n8 fragment per 8 members in use; every output is one k16 chain, so the bits do not depend on NB.
    switch ((cnt + 7) >> 3) {
#define P5A_GU_CASE(NB) case NB: gu_body<NB, CB>(Xg, Xu, Tg, Tu, cnt, e, svh_g, svh_u, suh_d, xd, K, N, limit, act_mode, k2g_e, k2u_e, smem_raw, rows_sh); break
        P5A_GU_CASE(1); P5A_GU_CASE(2); P5A_GU_CASE(3); P5A_GU_CASE(4); P5A_GU_CASE(5); P5A_GU_CASE(6); P5A_GU_CASE(7); P5A_GU_CASE(8);
#undef P5A_GU_CASE
        default: break;
    }
}
template <int CB> __device__ __forceinline__ void down_entry(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    (void)ys; __shared__ __align__(16) unsigned char smem_raw[SMEM]; __shared__ int rows_sh[TPM];
    const int item = TF_VBID.y; if (item >= counts[0]) return; const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2]; if (cnt > SMALL_MAX) return;
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1; TF_TASK_SYNC(); const int k2 = k2d[e];
    switch ((cnt + 7) >> 3) {
#define P5A_DN_CASE(NB) case NB: down_body<NB, CB>(Xd, Td, cnt, e, svh_d, y, K, D, k2, smem_raw, rows_sh); break
        P5A_DN_CASE(1); P5A_DN_CASE(2); P5A_DN_CASE(3); P5A_DN_CASE(4); P5A_DN_CASE(5); P5A_DN_CASE(6); P5A_DN_CASE(7); P5A_DN_CASE(8);
#undef P5A_DN_CASE
        default: break;
    }
}
} // namespace tf_glm53_p5a

extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_p5a_gu_mcg(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode,
        const int* __restrict__ k2g, const int* __restrict__ k2u) { TF_VBLOCK_ENTER(); tf_glm53_p5a::gu_entry<tf_exl3::CB_MCG>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u); }
extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_p5a_gu_e4m3(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode,
        const int* __restrict__ k2g, const int* __restrict__ k2u) { TF_VBLOCK_ENTER(); tf_glm53_p5a::gu_entry<tf_exl3::CB_MCG_E4M3>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u); }
extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_p5a_down_mcg(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d, float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER(); tf_glm53_p5a::down_entry<tf_exl3::CB_MCG>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_p5a_down_e4m3(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d, float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER(); tf_glm53_p5a::down_entry<tf_exl3::CB_MCG_E4M3>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}
