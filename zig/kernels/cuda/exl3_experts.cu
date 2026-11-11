// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
#include "tf_vblock.cuh"
// EXL3 routed experts: grouping, grouped GEMMs, epilogues; port of tensorfold's experts.cu (ExLlamaV3, Turboderp).
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/cuda/exl3/ (Apache-2.0). Changes: add the CB 3 grouped instantiation.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "exl3_decode.cuh"

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

namespace tf_exl3 {

// One projection's grouped arguments; the fields must stay in launch.zig add order.
struct GroupedArgs {
    const half* x0; // fp16 [P, K], the rotated gate (or down) inputs
    const half* x1; // fp16 [P, K], the rotated up inputs
    const long long* tp0; // int64 [E], one trellis pointer per expert
    const long long* tp1;
    const int* k2_0; // int32 [E], half-bits a value
    const int* k2_1;
    const int* uids; // int32 [nexp_max], distinct routed experts in id order
    const int* ucount; // int32 [1]
    const int* members; // int32 [nexp_max, maxm], row * 32 + slot, -1 after the last member
    float* z; // fp32 partials, [(mat * sk + split) * P + p] * N + col
    int k, n, p, sk, maxm, slots, nexp_max, mats, ns, standard, skip, l2d;
};

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc: per-lane loads and the shared decode (same bits).
template <int CB, int K2, int NT, bool NOALLOC = false>
__device__ __forceinline__ void warp_tiles(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0, const half* x0,
                                           const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4], int l2d) {
    // L2 read-ahead after MiaAI-Lab GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patch 0128-glm-full-expert-prefetch (Apache-2.0, Copyright 2026 Mia's AI Lab); independent CUDA. Changes: depth argument, GLM-only default.
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr int LINES = (NT * TW + 31) / 32;
    static_assert(LINES <= 32);
    const size_t kstride = (size_t)ntiles * TW;
    if (l2d > 0) {
        const int ahead = l2d < nkt ? l2d : nkt;
        const int count = ahead * LINES;
        for (int j = lane; j < count; j += 32)
            prefetch_l2(T + (size_t)(kt0 + j / LINES) * kstride + (size_t)nt0 * TW + (size_t)(j % LINES) * 32);
    }
    for (int i = 0; i < nkt; ++i) {
        const uint32_t* t0 = T + (size_t)(kt0 + i) * kstride + (size_t)nt0 * TW;
        if (l2d > 0 && i + l2d < nkt && lane < LINES)
            prefetch_l2(T + (size_t)(kt0 + i + l2d) * kstride + (size_t)nt0 * TW + (size_t)lane * 32);
        uint32_t w[NT][LW];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if constexpr (NOALLOC) ldg_lane_words_noalloc<K2>(t0 + j * TW, lane, w[j]);
            else ldg_lane_words<K2>(t0 + j * TW, lane, w[j]);
        }
        const int k = (kt0 + i) * 16;
        const uint32_t a[4] = {ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
                               ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w[j], lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
    }
}

// Program (expert u, n block, split, member tile): 16 members times W_q over the split's K range, summed in warp order.
template <int CB, int NT, int W, bool NOALLOC = false>
__device__ __forceinline__ void grouped_body(const GroupedArgs a, int u, int nblock, int zz) {
    if (u >= a.ucount[0]) return;
    const int MT = (a.maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % a.sk;
    const int mat = zz / MT / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) return;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = a.ns > 0 ? a.ns : (a.n >> 4); // row stride of the trellis, in tiles

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
        const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
        const int code = m < count ? a.members[first + m] : -1;
        rows_sh[threadIdx.x] = a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
    }
    TF_TASK_SYNC();
    if (rows_sh[0] < 0) return; // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    __shared__ __align__(16) float red[W][16][NT * 16];
    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                   \
    case K2_:                                                                                                \
        warp_tiles<CB, K2_, NT, NOALLOC>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc, a.l2d);  \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    TF_TASK_SYNC();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

}  // namespace tf_exl3

// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void tf_exl3_grouped_0_task(tf_exl3::GroupedArgs a) {  // 3inst, any width
    tf_exl3::grouped_body<0, 8, 4>(a, TF_VBID.x, TF_VBID.y, TF_VBID.z);
}
extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_0(tf_exl3::GroupedArgs a) {
    TF_VBLOCK_ENTER(); tf_exl3_grouped_0_task(a); }

__device__ __forceinline__ void tf_exl3_grouped_1_task(tf_exl3::GroupedArgs a) {  // mcg, any width
    tf_exl3::grouped_body<1, 8, 4>(a, TF_VBID.x, TF_VBID.y, TF_VBID.z);
}
extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_1(tf_exl3::GroupedArgs a) {
    TF_VBLOCK_ENTER(); tf_exl3_grouped_1_task(a); }

__device__ __forceinline__ void tf_exl3_grouped_2_task(tf_exl3::GroupedArgs a) {  // mul1, any width (RED-SNOW's experts)
    tf_exl3::grouped_body<2, 8, 4>(a, TF_VBID.x, TF_VBID.y, TF_VBID.z);
}
extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_2(tf_exl3::GroupedArgs a) {
    TF_VBLOCK_ENTER(); tf_exl3_grouped_2_task(a); }

__device__ __forceinline__ void tf_exl3_grouped_3_task(tf_exl3::GroupedArgs a) {  // mcg rounded through E4M3
    tf_exl3::grouped_body<3, 8, 4>(a, TF_VBID.x, TF_VBID.y, TF_VBID.z);
}
extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_3(tf_exl3::GroupedArgs a) {
    TF_VBLOCK_ENTER(); tf_exl3_grouped_3_task(a); }

#define TF_EXL3_NOALLOC_KERNEL(CB_) \
extern "C" __global__ void __launch_bounds__(128) tf_exl3_grouped_noalloc_##CB_(tf_exl3::GroupedArgs a) { \
    TF_VBLOCK_ENTER(); tf_exl3::grouped_body<CB_, 8, 4, true>(a, TF_VBID.x, TF_VBID.y, TF_VBID.z); }
TF_EXL3_NOALLOC_KERNEL(0)
TF_EXL3_NOALLOC_KERNEL(1)
TF_EXL3_NOALLOC_KERNEL(2)
TF_EXL3_NOALLOC_KERNEL(3)
#undef TF_EXL3_NOALLOC_KERNEL

// Small parity entry points use the exact decoder template that grouped_3 calls.
template <int K2>
__device__ __forceinline__ void mcg_e4m3_tile(const uint32_t* tile, half* out) {
    const int lane = threadIdx.x;
    uint32_t w[tf_exl3::lane_words<K2>()];
    tf_exl3::load_lane_words<K2>(tile, lane, w);
    uint32_t b0[2], b1[2];
    tf_exl3::decode_lane<K2, tf_exl3::CB_MCG_E4M3>(w, lane, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    const uint32_t v[4] = { b0[0], b0[1], b1[0], b1[1] };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const half2 h = *reinterpret_cast<const half2*>(&v[q]);
        const int col = g + 8 * (q >> 1), row = 2 * t + 8 * (q & 1);
        out[(size_t)row * 16 + col] = __low2half(h);
        out[(size_t)(row + 1) * 16 + col] = __high2half(h);
    }
}

extern "C" __global__ void tf_exl3_mcg_e4m3_tile_2(const uint32_t* tile, half* out) {
    TF_VBLOCK_ENTER(); mcg_e4m3_tile<2>(tile, out); }
extern "C" __global__ void tf_exl3_mcg_e4m3_tile_3(const uint32_t* tile, half* out) {
    TF_VBLOCK_ENTER(); mcg_e4m3_tile<3>(tile, out); }
extern "C" __global__ void tf_exl3_mcg_e4m3_tile_4(const uint32_t* tile, half* out) {
    TF_VBLOCK_ENTER(); mcg_e4m3_tile<4>(tile, out); }
extern "C" __global__ void tf_exl3_mcg_e4m3_tile_6(const uint32_t* tile, half* out) {
    TF_VBLOCK_ENTER(); mcg_e4m3_tile<6>(tile, out); }
extern "C" __global__ void tf_exl3_mcg_e4m3_tile_8(const uint32_t* tile, half* out) {
    TF_VBLOCK_ENTER(); mcg_e4m3_tile<8>(tile, out); }

namespace tf_exl3 {

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

struct GroupArgs {
    const int* pick; // int32 [rows * slots]
    int* uids; // int32 [nexp_max]
    int* ucount; // int32 [1]
    int* members; // int32 [nexp_max, maxm]
    int rows, slots, experts, maxm;
};

__device__ __forceinline__ void tf_exl3_group_task(const GroupArgs a, int* sh_pick) {
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = a.rows * a.slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = a.pick[i];
    TF_TASK_SYNC();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < a.experts)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    TF_TASK_SYNC();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v; // exclusive per warp
        if (lane == 31) a.ucount[0] = s;
    }
    TF_TASK_SYNC();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        a.uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < a.maxm; ++i)
            if (sh_pick[i] == e) a.members[place * a.maxm + j++] = (i / a.slots) * 32 + (i % a.slots);
        for (; j < a.maxm; ++j) a.members[place * a.maxm + j] = -1;
        ++place;
    }
}

extern "C" __global__ void __launch_bounds__(GROUP_THREADS) tf_exl3_group(const GroupArgs a) {
    TF_VBLOCK_ENTER();
    extern __shared__ int sh_pick[];
    tf_exl3_group_task(a, sh_pick);
}

// Program (slot, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
struct RotInArgs {
    const void* x; // bf16 or fp16 [rows, x_stride]
    int x_stride;
    int x_dtype; // 0: fp16, 1: bf16 (exl3.zig DType)
    const int* pick; // int32 [rows * slots]
    const half* suh0; // fp16 [E, K], gate
    const half* suh1; // fp16 [E, K], up
    half* out0; // fp16 [P, K]
    half* out1;
    int k, slots, experts;
};

__device__ __forceinline__ void tf_exl3_experts_rot_in_task(const RotInArgs a) {
    const int p = TF_VBID.x, blk = TF_VBID.y, mat = TF_VBID.z;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int row = p / a.slots;
    const half* suh = (mat ? a.suh1 : a.suh0) + (size_t)e * a.k + blk * 128 + 4 * lane;
    const uint16_t* xr = reinterpret_cast<const uint16_t*>(a.x) + (size_t)row * a.x_stride + blk * 128 + 4 * lane;
    float v[4];
    if (a.x_dtype == 1) {
        const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
    } else {
        const half* xv = reinterpret_cast<const half*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(xv[j]) * __half2float(suh[j]);
    }
    fwht128(v, lane);
    half* o = (mat ? a.out1 : a.out0) + (size_t)p * a.k + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

extern "C" __global__ void tf_exl3_experts_rot_in(const RotInArgs a) {
    TF_VBLOCK_ENTER();
    tf_exl3_experts_rot_in_task(a);
}

// ROT-W4: one CTA a pair, block 128. Warp w rotates matrix (w & 1) and the 128-K blocks (w >> 1), (w >> 1) + 2, ...
// U blocks a pass: every load of the pass first, then the transforms, then the stores. Same per-element arithmetic as rot_in.
template <int U>
__device__ __forceinline__ void tf_exl3_experts_rot_in_w4_pair(const RotInArgs& a, const int p) {
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp & 1, first = warp >> 1;
    const int nblk = a.k / 128;
    const int row = p / a.slots;
    const half* suh_base = (mat ? a.suh1 : a.suh0) + (size_t)e * a.k + 4 * lane;
    const uint16_t* x_base = reinterpret_cast<const uint16_t*>(a.x) + (size_t)row * a.x_stride + 4 * lane;
    half* o_base = (mat ? a.out1 : a.out0) + (size_t)p * a.k + 4 * lane;
    for (int b0 = first; b0 < nblk; b0 += 2 * U) {
        float v[U][4];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int blk = b0 + 2 * u;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[u][j] = 0.f;
            if (blk < nblk) {
                const half* suh = suh_base + blk * 128;
                const uint16_t* xr = x_base + blk * 128;
                if (a.x_dtype == 1) {
                    const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(xr);
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[u][j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
                } else {
                    const half* xv = reinterpret_cast<const half*>(xr);
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[u][j] = __half2float(xv[j]) * __half2float(suh[j]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (b0 + 2 * u < nblk) fwht128(v[u], lane);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int blk = b0 + 2 * u;
            if (blk < nblk) {
                half* o = o_base + blk * 128;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[u][j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void tf_exl3_experts_rot_in_w4u4(const RotInArgs a) { TF_VBLOCK_ENTER(); tf_exl3_experts_rot_in_w4_pair<4>(a, TF_VBID.x); }
// TF_GLM53_GU_W8S companion: rot_in_w4's fp16 rotation for the pairs of small items only (<= 64 members, the p5a
// gate/up). Grid (items, 4): CTA y takes members y, y + 4, ... of its item. Same per-element arithmetic as rot_in.
struct RotInSmallArgs {
    RotInArgs r;
    const int* items; // int32 [items][3]: expert, first member, count
    const int* counts; // counts[0]: items
    const int* members; // int32 pair indices
};
extern "C" __global__ void tf_exl3_experts_rot_in_w4s(const RotInSmallArgs a) {
    TF_VBLOCK_ENTER();
    const int item = TF_VBID.x;
    if (item >= a.counts[0]) return;
    const int off = a.items[item * 3 + 1], cnt = a.items[item * 3 + 2];
    if (cnt > 64) return;
    for (int j = TF_VBID.y; j < cnt; j += 4) tf_exl3_experts_rot_in_w4_pair<4>(a.r, a.members[off + j]);
}
// Same Hadamard as rot_in, then one e4m3 code per element and one fp32 scale per 128-K block.
// scale = amax/448 of the fp32 values still in registers after fwht128 (1 when the block is zero).
// Byte j of a lane's word is K element 4*lane+j, in K order. The gate/up consumer widens that order.
struct RotInA8Args {
    const void* x;
    int x_stride;
    int x_dtype;
    const int* pick;
    const half* suh0;
    const half* suh1;
    uint8_t* out0;
    uint8_t* out1;
    float* sc0;
    float* sc1;
    int k, slots, experts;
    // rot_in_w8s only. 0: codes of pair p at byte p * k, its scale at sc[p]. Else codes at byte p * ostride of out
    // and the scale at byte p * ostride of sc (the engine keeps both inside the pair's fp16 slot).
    int ostride;
};

__device__ __forceinline__ uint8_t ra8_quant(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
#else
    return round_e4m3(x);
#endif
}

__device__ __forceinline__ void tf_exl3_experts_rot_in_a8_task(const RotInA8Args a) {
    const int p = TF_VBID.x, blk = TF_VBID.y, mat = TF_VBID.z;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int row = p / a.slots;
    const half* suh = (mat ? a.suh1 : a.suh0) + (size_t)e * a.k + blk * 128 + 4 * lane;
    const uint16_t* xr = reinterpret_cast<const uint16_t*>(a.x) + (size_t)row * a.x_stride + blk * 128 + 4 * lane;
    float v[4];
    if (a.x_dtype == 1) {
        const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
    } else {
        const half* xv = reinterpret_cast<const half*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(xv[j]) * __half2float(suh[j]);
    }
    fwht128(v, lane);
    float scaled[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) scaled[j] = v[j] * HAD_SCALE;
    float amax = fmaxf(fmaxf(fabsf(scaled[0]), fabsf(scaled[1])), fmaxf(fabsf(scaled[2]), fabsf(scaled[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    const int blocks = a.k >> 7;
    if (lane == 0) (mat ? a.sc1 : a.sc0)[(size_t)p * blocks + blk] = scale;
    uint32_t code_word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) code_word |= static_cast<uint32_t>(ra8_quant(scaled[j] / scale)) << (8 * j);
    uint8_t* o = (mat ? a.out1 : a.out0) + (size_t)p * a.k + blk * 128 + 4 * lane;
    *reinterpret_cast<uint32_t*>(o) = code_word;
}

extern "C" __global__ void tf_exl3_experts_rot_in_a8(const RotInA8Args a) {
    TF_VBLOCK_ENTER();
    tf_exl3_experts_rot_in_a8_task(a);
}

// GU-W8S rotation. One 512-thread CTA per (pair, matrix): warp w rotates 128-K blocks w, w+16, w+32 and
// keeps them in registers, so the CTA sees the whole row before it rounds. One fp32 scale per (pair, matrix):
// amax/448 over all K (1 when the row is zero), written to sc[p]. The gate/up epilogue applies it once.
// Codes are stored K-permuted inside every 16-K tile: byte 4*t+i holds K 2*t + (i & 1) + 8 * (i >> 1).
// That is the order one lane's two decode pairs come out in, so the gate/up A register needs no shuffle.
constexpr int W8S_WARPS = 16;
constexpr int W8S_BLK = 3;

__device__ __forceinline__ void tf_exl3_experts_rot_in_w8s_task(const RotInA8Args a, float* amax_sh) {
    const int p = TF_VBID.x, mat = TF_VBID.y;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int blocks = a.k >> 7;
    if (blocks > W8S_WARPS * W8S_BLK) __trap();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = p / a.slots;
    const half* suh_row = (mat ? a.suh1 : a.suh0) + (size_t)e * a.k;
    const uint16_t* x_row = reinterpret_cast<const uint16_t*>(a.x) + (size_t)row * a.x_stride;
    float v[W8S_BLK][4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < W8S_BLK; ++i) {
        const int blk = warp + W8S_WARPS * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[i][j] = 0.0f;
        if (blk < blocks) {
            const half* suh = suh_row + blk * 128 + 4 * lane;
            const uint16_t* xr = x_row + blk * 128 + 4 * lane;
            if (a.x_dtype == 1) {
                const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(xr);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[i][j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
            } else {
                const half* xv = reinterpret_cast<const half*>(xr);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[i][j] = __half2float(xv[j]) * __half2float(suh[j]);
            }
            fwht128(v[i], lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[i][j] *= HAD_SCALE;
                amax = fmaxf(amax, fabsf(v[i][j]));
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) amax_sh[warp] = amax;
    __syncthreads();
    float all = lane < W8S_WARPS ? amax_sh[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) all = fmaxf(all, __shfl_xor_sync(0xffffffffu, all, o));
    const float scale = all > 0.0f ? all / 448.0f : 1.0f;
    if (threadIdx.x == 0) {
        float* sc = mat ? a.sc1 : a.sc0;
        if (a.ostride) *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sc) + (size_t)p * a.ostride) = scale;
        else sc[p] = scale;
    }
    const float inv = 1.0f / scale;
    uint8_t* out_row = (mat ? a.out1 : a.out0) + (size_t)p * (a.ostride ? a.ostride : a.k);
    const int w = lane & 3, base = lane & ~3, sh = (lane & 1) * 16;
#pragma unroll
    for (int i = 0; i < W8S_BLK; ++i) {
        const int blk = warp + W8S_WARPS * i;
        if (blk < blocks) {
            uint32_t code_word = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) code_word |= static_cast<uint32_t>(ra8_quant(v[i][j] * inv)) << (8 * j);
            // Word w of a tile is K 2w, 2w+1 (lane w>>1, half w&1) then K 2w+8, 2w+9 (lane 2 + (w>>1), same half).
            const uint32_t wa = __shfl_sync(0xffffffffu, code_word, base | (w >> 1));
            const uint32_t wb = __shfl_sync(0xffffffffu, code_word, base | 2 | (w >> 1));
            const uint32_t word = ((wa >> sh) & 0xffffu) | (((wb >> sh) & 0xffffu) << 16);
            *reinterpret_cast<uint32_t*>(out_row + blk * 128 + 4 * lane) = word;
        }
    }
}

extern "C" __global__ void __launch_bounds__(512) tf_exl3_experts_rot_in_w8s(const RotInA8Args a) {
    TF_VBLOCK_ENTER();
    __shared__ float amax_sh[W8S_WARPS];
    tf_exl3_experts_rot_in_w8s_task(a, amax_sh);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// The SwiGLU clamps as torch.clamp has them: a NaN stays a NaN, where fminf and fmaxf hand back the limit.
__device__ __forceinline__ float upTo(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }

// Program (slot, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: bf16, 1: fp32), then Xd.
struct GateUpArgs {
    const float* z; // [(mat * sk + split) * P + p] * N + col
    const int* pick; // int32 [P]
    const half* svh_g; // fp16 [E, N]
    const half* svh_u; // fp16 [E, N]
    const half* suh_d; // fp16 [E, N]
    half* xd; // fp16 [P, N]
    int p, n, sk, experts;
    float limit;
    int act_mode;
};

__device__ __forceinline__ void tf_exl3_gateup_epilogue_task(const GateUpArgs a) {
    const int p = TF_VBID.x, blk = TF_VBID.y;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < a.sk; ++s) {
            sg += a.z[((size_t)(0 * a.sk + s) * a.p + p) * a.n + n + j];
            su += a.z[((size_t)(1 * a.sk + s) * a.p + p) * a.n + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (a.act_mode == 0) {
            float gg = bf16r(gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]));
            float uu = bf16r(uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]));
            if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]);
            float uu = uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]);
            if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(a.suh_d[(size_t)e * a.n + n + j]);
    }
    fwht128(v, lane);
    half* o = a.xd + (size_t)p * a.n + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

extern "C" __global__ void tf_exl3_gateup_epilogue(const GateUpArgs a) {
    TF_VBLOCK_ENTER();
    tf_exl3_gateup_epilogue_task(a);
}

// Program (member row, 128-block of the width): down epilogue and combine in one launch, the same bits as the two.
struct DownArgs {
    const float* z; // [split * P + p] * D + col
    const int* pick; // int32 [P]
    const half* svh_d; // fp16 [E, D]
    float* y; // fp32 [P, D]
    const float* wts; // fp32 [rows, slots]
    float* out; // fp32 [rows, D]
    int p, d, sk, experts, slots;
};

__device__ __forceinline__ void tf_exl3_down_combine_task(const DownArgs a, char* smem) {
    float4 (*part)[32] = reinterpret_cast<float4 (*)[32]>(smem); // [slot][lane]: the slot's 4 outputs of the lane
    const int r = TF_VBID.x, blk = TF_VBID.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * a.slots + k;
    const int e = a.pick[p];
    float o[4];
    if (e >= 0 && e < a.experts) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < a.sk; ++q) s += a.z[((size_t)q * a.p + p) * a.d + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(a.svh_d[(size_t)e * a.d + n + j]);
            a.y[(size_t)p * a.d + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = a.y[(size_t)p * a.d + n + j];
    }
    if (!a.wts) return; // runSlots reads the same post-rotation per-slot FP32 outputs without reducing them.
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    TF_TASK_SYNC();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < a.slots; ++q) {
        const float w = a.wts[r * a.slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) a.out[(size_t)r * a.d + n + j] = acc[j];
}

extern "C" __global__ void tf_exl3_down_combine(const DownArgs a) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(16) char smem[];
    tf_exl3_down_combine_task(a, smem);
}

}  // namespace tf_exl3

// The shared Grouped plan supplies triples (expert, first, count) and direct pair indices.
struct PlanStage {
    const int *items, *counts, *members;
    const __nv_bfloat16* x;
    const half *su0, *su1, *sv0, *sv1;
    half *x0, *x1;
    const float* z;
    void* out;
    int k, n, p, sk, slots, stride, skip, epi, act_mode;
    float limit;
};

extern "C" __global__ void tf_exl3_plan_rotate(const PlanStage a) {
    TF_VBLOCK_ENTER();
    const int it = TF_VBID.x, blk = TF_VBID.y, mat = TF_VBID.z;
    if (it >= a.counts[0]) return;
    const int e = a.items[3 * it], first = a.items[3 * it + 1], count = a.items[3 * it + 2];
    if (e == a.skip) return;
    const int lane = threadIdx.x;
    const half* su = (mat ? a.su1 : a.su0) + (size_t)e * a.k + blk * 128 + 4 * lane;
    for (int j = 0; j < count; ++j) {
        const int p = a.members[first + j], row = a.slots ? p / a.slots : p;
        const __nv_bfloat16* x = a.x + (size_t)row * a.stride + blk * 128 + 4 * lane;
        float v[4];
        for (int q = 0; q < 4; ++q) v[q] = __bfloat162float(x[q]) * __half2float(su[q]);
        tf_exl3::fwht128(v, lane);
        half* out = (mat ? a.x1 : a.x0) + (size_t)p * a.k + blk * 128 + 4 * lane;
        for (int q = 0; q < 4; ++q) out[q] = __float2half_rn(v[q] * tf_exl3::HAD_SCALE);
    }
}

extern "C" __global__ void tf_exl3_plan_finish(const PlanStage a) {
    TF_VBLOCK_ENTER();
    const int it = TF_VBID.x, blk = TF_VBID.y, lane = threadIdx.x;
    if (it >= a.counts[0]) return;
    const int e = a.items[3 * it], first = a.items[3 * it + 1], count = a.items[3 * it + 2];
    if (e == a.skip) return;
    const int n = blk * 128 + 4 * lane;
    for (int j = 0; j < count; ++j) {
        const int p = a.members[first + j];
        float g[4] = {}, u[4] = {};
        for (int s = 0; s < a.sk; ++s)
            for (int q = 0; q < 4; ++q) {
                g[q] += a.z[((size_t)s * a.p + p) * a.n + n + q];
                if (a.epi == 2) u[q] += a.z[((size_t)(a.sk + s) * a.p + p) * a.n + n + q];
            }
        tf_exl3::fwht128(g, lane);
        if (a.epi == 2) tf_exl3::fwht128(u, lane);
        for (int q = 0; q < 4; ++q) {
            float v = g[q] * tf_exl3::HAD_SCALE * __half2float(a.sv0[(size_t)e * a.n + n + q]);
            if (a.epi == 2) {
                float up = u[q] * tf_exl3::HAD_SCALE * __half2float(a.sv1[(size_t)e * a.n + n + q]);
                if (a.act_mode == 0) { v = tf_exl3::bf16r(v); up = tf_exl3::bf16r(up); }
                if (a.limit > 0) { v = tf_exl3::upTo(v, a.limit); up = tf_exl3::within(up, a.limit); }
                v = v / (1.f + expf(-v));
                if (a.act_mode == 0) v = tf_exl3::bf16r(v);
                v *= up;
            }
            const size_t at = (size_t)p * a.n + n + q;
            if (a.epi == 0) reinterpret_cast<float*>(a.out)[at] = v;
            else reinterpret_cast<__nv_bfloat16*>(a.out)[at] = __float2bfloat16_rn(v);
        }
    }
}

// Transposed fp32 prompt accumulators in the shared EXL3 module; decode kernels above stay unchanged.
#include "exl3_prompt.cuh"
using namespace tf_exl3_prompt;
extern "C" __global__ void tf_exl3_prompt_rotate(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
        half* __restrict__ out, int K) {
    TF_VBLOCK_ENTER();
    const int row = TF_VBID.x, blk = TF_VBID.y;
    const int lane = threadIdx.x;
    const half* sh = suh + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(sh[j]);
    fwht128(v, lane);
    half* o = out + (size_t)row * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

template <int CB>
__device__ __forceinline__ void tf_exl3_prompt_gu_body(const half* Xg, const half* Xu, const uint64_t* Tg, const uint64_t* Tu,
        const int* items, const int* counts, const int* members, const half* svh_g, const half* svh_u, const half* suh_d,
        half* xd, int K, int N, int xdiv, float limit, int act_mode, const int* k2g, const int* k2u) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    if (xdiv > 1) tgateup<TPM, CB>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u, smem_raw, rows_sh);
    else tgateup<2 * TPM, CB>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 1, limit, act_mode, k2g, k2u, smem_raw, rows_sh);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ k2g, const int* __restrict__ k2u) {
    TF_VBLOCK_ENTER();
    tf_exl3_prompt_gu_body<CB_MCG>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u);
}
// The same with GLM-5.3's mcg codebook rounded through E4M3 (decode_lane<K2, CB_MCG_E4M3>).
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu_e4m3(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ k2g, const int* __restrict__ k2u) {
    TF_VBLOCK_ENTER();
    tf_exl3_prompt_gu_body<CB_MCG_E4M3>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u);
}

// Block (256 model-width columns, item): y[row][slot][:] = the down epilogue of the item's pairs (this rank's share).
template <int CB>
__device__ __forceinline__ void tf_exl3_prompt_down_body(const half* Xd, const uint64_t* Td, const int* items, const int* counts,
        const int* members, const half* svh_d, float* y, int K, int D, int ys, const int* k2d) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    const int cb = TF_VBID.x, item = TF_VBID.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    TF_TASK_SYNC();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    const int k2 = k2d[e];
    tmain<1, TPM, CB>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, 1, k2, k2, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) TF_TASK_SYNC();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        TF_TASK_SYNC();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + (size_t)p * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER();
    tf_exl3_prompt_down_body<CB_MCG>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down_e4m3(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER();
    tf_exl3_prompt_down_body<CB_MCG_E4M3>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}

// TF_GLM53_PROMPT_EXPERTS=2 over a 128-member plan (exl3_prompt.cuh's wide path); every output has the bits of
// tf_exl3_prompt_gu/down. The _s entries are those kernels unchanged for items of 64 members or fewer (larger items
// return at once); wup, wgate and wdown take the larger items (dynamic shared memory: wsmem_bytes(), exl3_prompt.zig).
__device__ __forceinline__ bool tf_exl3_prompt_small(const int* items, const int* counts) {
    const int item = TF_VBID.y;
    return item < counts[0] && items[3 * item + 2] <= TPM;
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu_s(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ k2g, const int* __restrict__ k2u) {
    TF_VBLOCK_ENTER();
    if (!tf_exl3_prompt_small(items, counts)) return;
    tf_exl3_prompt_gu_body<CB_MCG>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu_s_e4m3(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, const int* __restrict__ k2g, const int* __restrict__ k2u) {
    TF_VBLOCK_ENTER();
    if (!tf_exl3_prompt_small(items, counts)) return;
    tf_exl3_prompt_gu_body<CB_MCG_E4M3>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, k2g, k2u);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down_s(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER();
    if (!tf_exl3_prompt_small(items, counts)) return;
    tf_exl3_prompt_down_body<CB_MCG>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down_s_e4m3(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, const int* __restrict__ k2d) {
    TF_VBLOCK_ENTER();
    if (!tf_exl3_prompt_small(items, counts)) return;
    tf_exl3_prompt_down_body<CB_MCG_E4M3>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, k2d);
}

#define TF_EXL3_PROMPT_WIDE(SUFFIX, CBV)                                                                                   \
    extern "C" __global__ void __launch_bounds__(WTHREADS, 2) tf_exl3_prompt_wup##SUFFIX(const half* __restrict__ Xu,       \
            const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,                   \
            const int* __restrict__ members, const half* __restrict__ svh_u, float* __restrict__ uu, int K, int N, int xdiv,  \
            int act_mode, const int* __restrict__ k2u) {                                                                     \
        TF_VBLOCK_ENTER();                                                                                                 \
        extern __shared__ __align__(16) unsigned char tf_exl3_wide_smem[];                                                 \
        __shared__ int rows_sh[WPM];                                                                                       \
        wup<CBV>(Xu, Tu, items, counts, members, svh_u, uu, K, N, xdiv, act_mode, k2u, tf_exl3_wide_smem, rows_sh);       \
    }                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(WTHREADS, 2) tf_exl3_prompt_wgate##SUFFIX(const half* __restrict__ Xg,     \
            const uint64_t* __restrict__ Tg, const int* __restrict__ items, const int* __restrict__ counts,                   \
            const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ suh_d,                  \
            const float* __restrict__ uu, half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode,           \
            const int* __restrict__ k2g) {                                                                                   \
        TF_VBLOCK_ENTER();                                                                                                 \
        extern __shared__ __align__(16) unsigned char tf_exl3_wide_smem[];                                                 \
        __shared__ int rows_sh[WPM];                                                                                       \
        wgate<CBV>(Xg, Tg, items, counts, members, svh_g, suh_d, uu, xd, K, N, xdiv, limit, act_mode, k2g, tf_exl3_wide_smem, rows_sh); \
    }                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(WTHREADS, 2) tf_exl3_prompt_wdown##SUFFIX(const half* __restrict__ Xd,     \
            const uint64_t* __restrict__ Td, const int* __restrict__ items, const int* __restrict__ counts,                   \
            const int* __restrict__ members, const half* __restrict__ svh_d, float* __restrict__ y, int K, int D,             \
            const int* __restrict__ k2d) {                                                                                   \
        TF_VBLOCK_ENTER();                                                                                                 \
        extern __shared__ __align__(16) unsigned char tf_exl3_wide_smem[];                                                 \
        __shared__ int rows_sh[WPM];                                                                                       \
        wdown<CBV>(Xd, Td, items, counts, members, svh_d, y, K, D, k2d, tf_exl3_wide_smem, rows_sh);                       \
    }
TF_EXL3_PROMPT_WIDE(, CB_MCG)
TF_EXL3_PROMPT_WIDE(_e4m3, CB_MCG_E4M3)
#undef TF_EXL3_PROMPT_WIDE
