// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#include "../tf_vblock.cuh"
// Adapted from ashhart/TensorFold@d1018b31:zig/kernels/cuda/deepseek_v41/topk_cuda.cu (blob 839ee8a85f0e, Apache-2.0).
// Original contributors: @jayleaton, PRs #671, #674, #727.
// GLM indexer math adapts bertholomus/bertholomus/TensorFold@ddfad35 `select.py` and `dcp.py` (Apache-2.0).
// Changes: GLM-5.3 DCP2 index selection, 32 entries/thread, and key/score kernels.
// DCP exchange layout also follows NVIDIA flare-local-inference@47581cc's packed-candidate design (Apache-2.0); no code copied.
// Indexer decode top-k: top K of unique keys (score image, then lower position), exact.

// Radix select of the K-th 32-bit image over a cluster of <= 8 CTAs a row (DSMEM histograms), then scan-order pick.

// Image matches Python select._order_key: s+0 collapses signed zero, then transform IEEE bits to unsigned order.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cooperative_groups.h>
#include <mma.h>

#include "rope.cuh"

namespace glm53_topk {
namespace cg = cooperative_groups;

constexpr int NT = 512, NW = NT / 32, MAXCL = 8, MAXEPT = 33;
constexpr unsigned FULL = 0xffffffffu;

struct Candidate { float score; int token; };

struct Job {
    const float* s; long long ss;    // scores [R, n] (unit stride)
    const int* cand; long long cs;   // mode 3: ascending candidate blocks [R, cols] (-1 last)
    int* out; long long os;          // [R, k]
    int* cnt;                        // [R] visible counts or null
    Candidate* packed;               // [R, k] score/position records for one exchange
    int nk, k, mode, ept;            // entries, K, 0 dense / 2 blocks / 3 candidate positions, entries a thread
};

struct Args {
    Job job[2];
    const long long* pos;            // int64 [R], global position for each row
    int ratio, bs, owner;
};

__device__ __forceinline__ uint32_t image(float s) {
    const uint32_t bits = __float_as_uint(s + 0.0f);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// exclusive scan of one int a thread over the CTA (512 threads); returns the prefix, *total the sum
__device__ int block_scan(int v, int* ws, int* total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) ws[w] = x;
    TF_TASK_SYNC();
    if (w == 0) {
        int z = lane < NW ? ws[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(FULL, z, o);
            if (lane >= o) z += y;
        }
        if (lane < NW) ws[NW + lane] = z;
    }
    TF_TASK_SYNC();
    const int pre = (w ? ws[NW + w - 1] : 0) + x - v;
    *total = ws[2 * NW - 1];
    TF_TASK_SYNC();
    return pre;
}

__device__ __forceinline__ void tf_glm53_topk_task(Args a, uint32_t* img) {
    cg::cluster_group cl = cg::this_cluster();
    const Job J = TF_VBID.z == 0 ? a.job[0] : a.job[1];        // (no dynamic indexing: no local memory)
    const int CL = static_cast<int>(cl.num_blocks()), rank = static_cast<int>(cl.block_rank());
    const int r = TF_VBID.y, tid = threadIdx.x, lane = tid & 31;
    // SL = 512 x ept images (0: missing), supplied by the launch wrapper.
    __shared__ int hist[2][256];
    __shared__ int tot[256];
    __shared__ int ws[2 * NW];
    __shared__ int ex[2];                                        // (above, equal) of this CTA, read by the cluster
    __shared__ int dec[4];
    const int SL = NT * J.ept, lo = rank * SL, n = max(0, min(SL, J.nk - lo));
    const long long q = a.pos[r];
    const long long local_visible = q < a.owner ? 0 : (q - a.owner) / a.ratio + 1;
    const int nvis = static_cast<int>(local_visible);
    if (rank == 0 && tid == 0 && J.cnt) J.cnt[r] = 0;            // before the first cluster barrier (release)
    const float* srow = J.s + static_cast<long long>(r) * J.ss;
    const int* crow = J.cand ? J.cand + static_cast<long long>(r) * J.cs : nullptr;
    for (int i = tid; i < SL; i += NT) {                         // images, coalesced
        const int e = lo + i;
        uint32_t u = 0;
        if (i < n) {
            if (J.mode == 2) {                                   // block e: max over its visible members, newest +inf
                float m = 0.f;                                   // tl.max (maxnumf) over the members, masked -inf
                for (int j = 0; j < a.bs; ++j) {
                    const int p = e * a.bs + j;
                    const float v = p < nvis ? srow[p] : -INFINITY;
                    m = j ? fmaxf(m, v) : v;
                }
                if (e == (nvis - 1) / a.bs) m = INFINITY;
                if (e * a.bs < nvis) u = image(m);
            } else if (J.mode == 3) {
                if (crow[e / a.bs] >= 0) u = image(srow[e]);
            } else if (e < nvis) {
                u = image(srow[e]);
            }
        }
        img[i] = u;
    }
    TF_TASK_SYNC();
    uint32_t prefix = 0, pmask = 0;
    int above = 0, quota_mode = 0;                               // 0: every match selected; 1: equal images capped
    for (int p = 0; p < 4; ++p) {
        int* h = hist[p & 1];
        if (tid < 256) h[tid] = 0;
        TF_TASK_SYNC();
        const int shift = 24 - 8 * p;
        for (int i0 = 0; i0 < SL; i0 += NT) {
            const uint32_t v = img[i0 + tid];
            const bool m = v != 0 && (v & pmask) == prefix;
            const unsigned act = __ballot_sync(FULL, m);
            if (m) {
                const int d = (v >> shift) & 255;
                const unsigned peers = __match_any_sync(act, d);
                if (lane == __ffs(peers) - 1) atomicAdd(&h[d], __popc(peers));
            }
        }
        cl.sync();
        if (tid < 256) {
            int v[MAXCL];
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) v[x] = x < CL ? cl.map_shared_rank(h, x)[tid] : 0;
            int sum = 0;
#pragma unroll
            for (int x = 0; x < MAXCL; ++x) sum += v[x];
            tot[tid] = sum;
        }
        TF_TASK_SYNC();
        if (tid < 32) {                                          // suffix sums over 256 digits, 8 a lane; the digit
            int loc[8], run = 0;
            for (int x = 7; x >= 0; --x) { run += tot[lane * 8 + x]; loc[x] = run; }
            int suf = run;                                       // inclusive suffix over lanes >= this one
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_down_sync(FULL, suf, o);
                if (lane + o < 32) suf += y;
            }
            const int after = suf - run;                         // lanes above this one
            int best = -1, bge = 0;
            for (int x = 7; x >= 0; --x)
                if (best < 0 && above + after + loc[x] >= J.k) { best = lane * 8 + x; bge = after + loc[x]; }
            const unsigned has = __ballot_sync(FULL, best >= 0);
            const int src = has ? 31 - __clz(has) : 0;
            const int dg = __shfl_sync(FULL, best, src), ge = __shfl_sync(FULL, bge, src);
            const int all = __shfl_sync(FULL, suf, 0);
            if (lane == 0) { dec[0] = dg; dec[1] = ge; dec[2] = all; }
        }
        TF_TASK_SYNC();
        const int dg = dec[0], ge = dec[1], all = dec[2];
        if (p == 0 && all <= J.k) break;                         // every valid entry (prefix / mask stay 0)
        const int hd = tot[dg];
        above += ge - hd;
        prefix |= static_cast<uint32_t>(dg) << shift;
        pmask |= 255u << shift;
        if (above + hd == J.k) break;                            // the prefix's entries complete the top K
        if (p == 3) quota_mode = 1;                              // ties at the K-th image: the first ones win
    }
    // classify: 2 above (selected), 1 equal (selected up to the quota, in scan order), 0 out
    const int e0 = tid * J.ept;
    int gt = 0, eq = 0;
    for (int x = 0; x < J.ept; ++x) {
        const uint32_t v = img[e0 + x];
        if (v == 0) continue;
        const uint32_t mv = v & pmask;
        gt += mv > prefix;
        eq += mv == prefix;
    }
    int tgt, teq;
    const int pre_eq = block_scan(eq, ws, &teq);
    block_scan(gt, ws, &tgt);
    if (tid == 0) { ex[0] = tgt; ex[1] = teq; }
    cl.sync();
    int base = 0, eq_before = 0, total = 0, quota = teq;
    {
        const int need = J.k - above;                            // equal images to take (quota mode)
        for (int x = 0; x < CL; ++x) {
            const int* rx = cl.map_shared_rank(ex, x);
            const int xg = rx[0], xe = rx[1];
            const int xq = quota_mode ? max(0, min(xe, need - eq_before)) : xe;
            if (x < rank) base += xg + xq;
            if (x == rank) quota = xq;
            eq_before += xe;
            total += xg + xq;
        }
    }
    cl.sync();                                                   // no DSMEM access after this: CTAs may exit
    int sel = 0;
    {
        int er = pre_eq;
        for (int x = 0; x < J.ept; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            if (mv > prefix) ++sel;
            else if (mv == prefix) { sel += er < quota; ++er; }
        }
    }
    int tsel;
    const int off = base + block_scan(sel, ws, &tsel);
    int* orow = J.out + static_cast<long long>(r) * J.os;
    int vis = 0;
    {
        int er = pre_eq, o = off;
        for (int x = 0; x < J.ept; ++x) {
            const uint32_t v = img[e0 + x];
            if (v == 0) continue;
            const uint32_t mv = v & pmask;
            bool s = mv > prefix;
            if (!s && mv == prefix) { s = er < quota; ++er; }
            if (!s) continue;
            const int e = lo + e0 + x;
            const int pp = J.mode == 3 ? crow[e / a.bs] * a.bs + e % a.bs : e * a.ratio + a.owner;
            orow[o++] = pp;
            if (J.packed) J.packed[static_cast<long long>(r) * J.k + (o - 1)] = { srow[e] + 0.0f, pp };
            vis += pp <= q;
        }
    }
    if (J.cnt && q + 1 > 2048) {
        int tv;
        block_scan(vis, ws, &tv);
        if (tid == 0 && tv) atomicAdd(J.cnt + r, tv);
    }
    if (rank == 0) for (int i = total + tid; i < J.k; i += NT) {
        orow[i] = -1;
        if (J.packed) J.packed[static_cast<long long>(r) * J.k + i] = { -CUDART_INF_F, -1 };
    }
}

__global__ void __launch_bounds__(NT, 1) tf_glm53_topk(Args a) {
    TF_VBLOCK_ENTER();
    extern __shared__ uint32_t img[];
    tf_glm53_topk_task(a, img);
}

// Same task, two CTAs a SM: the host fits ept to nkeys so a single cluster CTA a row stays resident.
extern "C" __global__ void __launch_bounds__(NT, 2) tf_glm53_topk_fit(Args a) {
    TF_VBLOCK_ENTER();
    extern __shared__ uint32_t img[];
    tf_glm53_topk_task(a, img);
}

// Shared memory a CTA for ``ept`` entries a thread (the images; the rest is static).
inline size_t smem(int ept) { return static_cast<size_t>(NT) * ept * 4; }

inline cudaError_t launch(const Args& a, int jobs, int R, int CL, int ept, cudaStream_t stream) {
    if (R < 1 || jobs < 1 || jobs > 2 || CL < 1 || CL > MAXCL || ept < 1 || ept > MAXEPT)
        return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        const cudaError_t e = cudaFuncSetAttribute(tf_glm53_topk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(smem(MAXEPT)));
        if (e != cudaSuccess) return e;
        init = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CL, R, jobs);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = smem(ept);
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, tf_glm53_topk, a);
}

// Key projection has already been rounded to bf16. Apply the Python LayerNorm convention and RoPE,
// then keep the interleaved 128-wide key in the bf16 cache.
__device__ __forceinline__ void tf_glm53_index_q_task(const __nv_bfloat16* in, const float* cos, const float* sin, __nv_bfloat16* out) {
    const int h = TF_VBID.x, r = TF_VBID.y, d = threadIdx.x;
    const long long base = (static_cast<long long>(r) * 32 + h) * 128;
    if (d < 32) {
        const int pair = d;
        const float c = cos[r * 32 + pair], s = sin[r * 32 + pair];
        const float x0 = __bfloat162float(in[base + pair * 2]);
        const float x1 = __bfloat162float(in[base + pair * 2 + 1]);
        out[base + pair * 2] = __float2bfloat16_rn(glm53_rope::even(x0, c, x1, s));
        out[base + pair * 2 + 1] = __float2bfloat16_rn(glm53_rope::odd(x1, c, x0, s));
    } else if (d >= 64) {
        out[base + d] = in[base + d];
    }
}

extern "C" __global__ void tf_glm53_index_q(const __nv_bfloat16* in, const float* cos, const float* sin, __nv_bfloat16* out) {
    TF_VBLOCK_ENTER();
    tf_glm53_index_q_task(in, cos, sin, out);
}

__device__ __forceinline__ void tf_glm53_index_key_task(const __nv_bfloat16* proj, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                           const float* cos, const float* sin, const long long* positions, __nv_bfloat16* out,
                           int stride, int col, int rank, int world, float eps) {
    const int r = TF_VBID.x, t = threadIdx.x;
    const long long position = positions[r];
    if (position % world != rank) return;
    const long long slot = position / world;
    __shared__ float x[128], red[128];
    x[t] = __bfloat162float(proj[static_cast<long long>(r) * stride + col + t]);
    red[t] = x[t];
    TF_TASK_SYNC();
    for (int d = 64; d; d >>= 1) { if (t < d) red[t] += red[t + d]; TF_TASK_SYNC(); }
    const float mean = red[0] * (1.0f / 128.0f);
    TF_TASK_SYNC();   // every thread has read red[0] before thread 0 overwrites it (the race made a few keys' means wrong)
    red[t] = (x[t] - mean) * (x[t] - mean);
    TF_TASK_SYNC();
    for (int d = 64; d; d >>= 1) { if (t < d) red[t] += red[t + d]; TF_TASK_SYNC(); }
    const float denom = sqrtf(red[0] * (1.0f / 128.0f) + eps);
    const float y = (x[t] - mean) / denom * __bfloat162float(weight[t]) + __bfloat162float(bias[t]);
    if (t < 32) {
        const int pair = t;
        const float c = cos[r * 32 + pair], s = sin[r * 32 + pair];
        const float y0 = __bfloat162float(__float2bfloat16_rn((x[pair * 2] - mean) / denom * __bfloat162float(weight[pair * 2]) + __bfloat162float(bias[pair * 2])));
        const float y1 = __bfloat162float(__float2bfloat16_rn((x[pair * 2 + 1] - mean) / denom * __bfloat162float(weight[pair * 2 + 1]) + __bfloat162float(bias[pair * 2 + 1])));
        out[slot * 128 + pair * 2] = __float2bfloat16_rn(glm53_rope::even(y0, c, y1, s));
        out[slot * 128 + pair * 2 + 1] = __float2bfloat16_rn(glm53_rope::odd(y1, c, y0, s));
    } else if (t >= 64) {
        out[slot * 128 + t] = __float2bfloat16_rn((x[t] - mean) / denom * __bfloat162float(weight[t]) + __bfloat162float(bias[t]));
    }
}

extern "C" __global__ void tf_glm53_index_key(const __nv_bfloat16* proj, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                           const float* cos, const float* sin, const long long* positions, __nv_bfloat16* out,
                           int stride, int col, int rank, int world, float eps) {
    TF_VBLOCK_ENTER();
    tf_glm53_index_key_task(proj, weight, bias, cos, sin, positions, out, stride, col, rank, world, eps);
}

// Fixed head-major, dimension-major fp32 accumulation from bf16 Q and K.
__device__ __forceinline__ void tf_glm53_index_scores_oracle_task(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, char* smem) {
    const int r = TF_VBID.y, k = TF_VBID.x * TF_VBDIM.x + threadIdx.x;
    __nv_bfloat16* qtile = reinterpret_cast<__nv_bfloat16*>(smem);
    float* wtile = reinterpret_cast<float*>(qtile + 32 * 128);
    for (int i = threadIdx.x; i < 32 * 128; i += TF_VBDIM.x) qtile[i] = qi[r * 32 * 128 + i];
    if (threadIdx.x < 32) wtile[threadIdx.x] = __bfloat162float(iw[static_cast<long long>(r) * iw_stride + iw_col + threadIdx.x]) * 0.1767766952966369f;
    TF_TASK_SYNC();
    if (k >= nkeys) return;
    const long long global_pos = static_cast<long long>(k) * world + rank;
    if (global_pos > pos[r]) { scores[static_cast<long long>(r) * key_stride + k] = -CUDART_INF_F; return; }
    float total = 0.0f;
    for (int h = 0; h < 32; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < 128; ++d) {
            const float q = __bfloat162float(qtile[h * 128 + d]);
            dot = fmaf(q, __bfloat162float(keys[static_cast<long long>(k) * 128 + d]), dot);
        }
        total = fmaf(wtile[h],
                     fmaxf(dot * 0.08838834764831845f, 0.0f), total);
    }
    scores[static_cast<long long>(r) * key_stride + k] = total;
}

extern "C" __global__ void tf_glm53_index_scores_oracle(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_oracle_task(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, smem);
}

// Tensor-core score path. Each CTA covers 16 keys and computes the 32 head dots in two
// 16x16x128 BF16 WMMA products. The reduction order is head 0..31 for every key, then
// one weighted ReLU FMA per head, matching decode and prefill regardless of row batch size.
__device__ __forceinline__ void tf_glm53_index_scores_task(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, char* smem) {
    // Decode index-score tile skip after bertholomus/TensorFold glm-dsa-tp4 commit 4ec5c5d (Apache-2.0); independent CUDA.
    using namespace nvcuda;
    const int r = TF_VBID.y, key0 = TF_VBID.x * 16;
    const long long hi = pos[r];
    const long long vis = hi < rank ? 0 : (hi - rank) / world + 1;
    if (key0 >= vis) return;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    __nv_bfloat16* qtile = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* ktile = qtile + 16 * 128;
    float* dots = reinterpret_cast<float*>(ktile + 16 * 128);
    float* weights = dots + 32 * 16;
    float* tile = weights + 32;
    for (int h = lane; h < 32; h += TF_VBDIM.x) weights[h] = __bfloat162float(iw[static_cast<long long>(r) * iw_stride + iw_col + h]) * 0.1767766952966369f;
    TF_TASK_SYNC();
    for (int hg = 0; hg < 2; ++hg) {
        for (int i = lane; i < 16 * 128; i += TF_VBDIM.x) {
            const int h = i / 128, d = i % 128;
            const int base = r * 32 * 128 + (h + 16 * hg) * 128;
            qtile[i] = qi[base + d];
        }
        for (int i = lane; i < 16 * 128; i += TF_VBDIM.x) {
            const int key = key0 + i / 128, d = i % 128;
            ktile[i] = key < nkeys ? keys[static_cast<long long>(key) * 128 + d] : __float2bfloat16_rn(0.0f);
        }
        TF_TASK_SYNC();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        if (warp == 0) {
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < 128; d += 16) {
                wmma::load_matrix_sync(a_frag, qtile + d, 128);
                wmma::load_matrix_sync(b_frag, ktile + d, 128);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
            for (int h = 0; h < 16; ++h)
                for (int k = 0; k < 16; ++k) dots[(hg * 16 + h) * 16 + k] = tile[h * 16 + k];
        }
        TF_TASK_SYNC();
    }
    for (int key = lane; key < 16; key += TF_VBDIM.x) {
        const int k = key0 + key;
        if (k >= nkeys) continue;
        if (static_cast<long long>(k) * world + rank > pos[r]) {
            scores[static_cast<long long>(r) * key_stride + k] = -CUDART_INF_F;
            continue;
        }
        float total = 0.0f;
        for (int h = 0; h < 32; ++h)
            total = fmaf(weights[h], fmaxf(dots[h * 16 + key] * 0.08838834764831845f, 0.0f), total);
        scores[static_cast<long long>(r) * key_stride + k] = total;
    }
}

extern "C" __global__ void tf_glm53_index_scores(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_task(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, smem);
}

// Tiled indexer scores: one block = 64 keys x 4 rows (M = 4 rows x 32 heads = 128, one m16 tile per warp), so each
// key tile is read once for 4 rows and all 8 warps run the dots on the tensor cores. The arithmetic is the scalar
// kernel's: each 128-wide dot accumulates d = 0..127 in 16-wide MMA steps (fp32), relu(dot * 0.0884), then the row's
// total fmaf(w_h, ., total) over h = 0..31 in order, w_h = iw * 0.1768; masking as tf_glm53_index_scores.
constexpr int ISM_KEYS = 64, ISM_TILE_ROWS = 4, ISM_STR = 128 + 8;
constexpr int ISM_SMEM = ISM_TILE_ROWS * 32 * ISM_STR * 2 + ISM_KEYS * ISM_STR * 2 + ISM_TILE_ROWS * 32 * ISM_KEYS * 4 + ISM_TILE_ROWS * 32 * 4;

__device__ __forceinline__ void ism_mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int ISM_ROWS>
__device__ __forceinline__ void tf_glm53_index_scores_mma_task(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, char* smem) {
    const int key0 = TF_VBID.x * ISM_KEYS, row0 = TF_VBID.y * ISM_ROWS;
    // Decode index-score tile skip after bertholomus/TensorFold glm-dsa-tp4 commit 4ec5c5d (Apache-2.0); independent CUDA.
    long long hi = row0 < rows ? pos[row0] : -1;
    for (int r = row0 + 1; r < rows && r < row0 + ISM_ROWS; ++r) hi = max(hi, pos[r]);
    const long long vis = hi < rank ? 0 : (hi - rank) / world + 1;
    if (key0 >= vis) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);                  // [4 * 32][STR]
    __nv_bfloat16* ks = qs + ISM_TILE_ROWS * 32 * ISM_STR;                      // [KEYS][STR]
    float* rel = reinterpret_cast<float*>(ks + ISM_KEYS * ISM_STR);              // [4 * 32][KEYS]: relu(dot * scale)
    float* w = rel + ISM_TILE_ROWS * 32 * ISM_KEYS;                              // [4 * 32]
    for (int x = tid; x < ISM_KEYS * 16; x += 256) {
        const int k = x / 16, c = x % 16, key = key0 + k;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (key < nkeys) v = reinterpret_cast<const uint4*>(keys + static_cast<long long>(key) * 128)[c];
        reinterpret_cast<uint4*>(ks + k * ISM_STR)[c] = v;
    }
    TF_TASK_SYNC();
    // The CTA owns ISM_ROWS query rows, but reuses the 4-row score scratch in order. This keeps the
    // key tile resident once while preserving every row's original MMA and head-reduction chain.
    for (int row_batch = row0; row_batch < row0 + ISM_ROWS; row_batch += ISM_TILE_ROWS) {
    for (int x = tid; x < ISM_TILE_ROWS * 32 * 16; x += 256) {   // 16 16-byte pieces a head row
        const int rh = x / 16, c = x % 16, r = row_batch + rh / 32;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < rows) v = reinterpret_cast<const uint4*>(qi + (static_cast<long long>(r) * 32 + rh % 32) * 128)[c];
        reinterpret_cast<uint4*>(qs + rh * ISM_STR)[c] = v;
    }
    if (tid < ISM_TILE_ROWS * 32) {
        const int r = row_batch + tid / 32;
        w[tid] = r < rows ? __bfloat162float(iw[static_cast<long long>(r) * iw_stride + iw_col + tid % 32]) * 0.1767766952966369f : 0.f;
    }
    TF_TASK_SYNC();
    {   // warp = m tile: (row, head) pairs warp * 16 .. + 15; 8 n tiles of 8 keys; 8 K steps
        float acc[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        const __nv_bfloat16* a0 = qs + (warp * 16 + g) * ISM_STR;
        const __nv_bfloat16* a1 = a0 + 8 * ISM_STR;
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            const int k0 = st * 16 + 2 * q;
            const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(a0 + k0), *reinterpret_cast<const uint32_t*>(a1 + k0),
                                   *reinterpret_cast<const uint32_t*>(a0 + k0 + 8), *reinterpret_cast<const uint32_t*>(a1 + k0 + 8)};
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const __nv_bfloat16* b0 = ks + (t * 8 + g) * ISM_STR;
                const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(b0 + k0), *reinterpret_cast<const uint32_t*>(b0 + k0 + 8)};
                ism_mma(acc[t], a, b);
            }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int col = t * 8 + 2 * q;
            rel[(warp * 16 + g) * ISM_KEYS + col] = fmaxf(acc[t][0] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g) * ISM_KEYS + col + 1] = fmaxf(acc[t][1] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * ISM_KEYS + col] = fmaxf(acc[t][2] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * ISM_KEYS + col + 1] = fmaxf(acc[t][3] * 0.08838834764831845f, 0.0f);
        }
    }
    TF_TASK_SYNC();
    {   // thread = (row, key)
        const int rr = tid / ISM_KEYS, k = tid % ISM_KEYS, r = row_batch + rr, key = key0 + k;
        if (r < rows && key < nkeys) {
            if (static_cast<long long>(key) * world + rank > pos[r]) {
                scores[static_cast<long long>(r) * key_stride + key] = -CUDART_INF_F;
            } else {
                float total = 0.0f;
                for (int h = 0; h < 32; ++h) total = fmaf(w[rr * 32 + h], rel[(rr * 32 + h) * ISM_KEYS + k], total);
                scores[static_cast<long long>(r) * key_stride + key] = total;
            }
        }
    }
    TF_TASK_SYNC();
    }
}

template <int ISM_ROWS>
__device__ __forceinline__ void tf_glm53_index_scores_mma_entry(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, char* smem) {
    TF_VBLOCK_ENTER();
    tf_glm53_index_scores_mma_task<ISM_ROWS>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, smem);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_mma_4(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_mma_entry<4>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, smem);
}
extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_mma_8(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_mma_entry<8>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, smem);
}
extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_mma_16(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_mma_entry<16>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, smem);
}

constexpr int IQS_KEYS = 64, IQS_STR = 136, IQS_ROWS = 4;
constexpr int IQS_REL_BYTES = IQS_ROWS * 32 * IQS_KEYS * sizeof(float);
constexpr int IQS_WEIGHT_BYTES = IQS_ROWS * 32 * sizeof(float);
constexpr int IQS_SMEM(int stages) { return stages * IQS_KEYS * IQS_STR * (int)sizeof(__nv_bfloat16) + IQS_REL_BYTES + IQS_WEIGHT_BYTES; }
static_assert(IQS_SMEM(3) <= 101376, "qs s3 must fit in 99 KB");

__device__ __forceinline__ void iqs_cp16(__nv_bfloat16* dst, const __nv_bfloat16* src, bool valid) {
    const unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(shared), "l"(src), "r"(valid ? 16 : 0));
}

template <int STAGES>
__device__ __forceinline__ void iqs_prefetch(__nv_bfloat16* stage, const __nv_bfloat16* keys, int nkeys, int tile, int tid) {
    for (int piece = tid; piece < IQS_KEYS * 16; piece += 256) {   // the 8-half row pad is never read
        const int key = tile * IQS_KEYS + piece / 16, chunk = piece % 16;
        const bool valid = key < nkeys;
        const __nv_bfloat16* src = valid ? keys + static_cast<long long>(key) * 128 + chunk * 8 : keys;
        iqs_cp16(stage + (piece / 16) * IQS_STR + chunk * 8, src, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void iqs_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ int iqs_vis(int row, int rows, int vis_rows, const long long* pos, int rank, int world) {
    if (row >= rows) return 0;
    const int first = (row / vis_rows) * vis_rows;
    long long hi = -1;
    for (int r = first; r < rows && r < first + vis_rows; ++r) hi = max(hi, pos[r]);
    return hi < rank ? 0 : static_cast<int>((hi - rank) / world + 1);
}

template <int STAGES>
__device__ __forceinline__ void tf_glm53_index_scores_qs_task(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles, char* smem) {
    const int row0 = TF_VBID.y * IQS_ROWS, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
    const int total_tiles = (nkeys + IQS_KEYS - 1) / IQS_KEYS;
    const int first_tile = TF_VBID.x * tiles;
    __nv_bfloat16* stage_base = reinterpret_cast<__nv_bfloat16*>(smem);
    float* rel = reinterpret_cast<float*>(smem + STAGES * IQS_KEYS * IQS_STR * sizeof(__nv_bfloat16));
    float* w = reinterpret_cast<float*>(reinterpret_cast<char*>(rel) + IQS_REL_BYTES);
    __shared__ int vis_sh[IQS_ROWS];   // its own cells: w is rewritten below without a barrier after these reads
    if (tid < IQS_ROWS) vis_sh[tid] = iqs_vis(row0 + tid, rows, vis_rows, pos, rank, world);
    __syncthreads();
    int max_vis = 0;
    for (int rr = 0; rr < IQS_ROWS; ++rr) max_vis = max(max_vis, vis_sh[rr]);
    const int row_vis = vis_sh[tid / IQS_KEYS];
    const int end_tile = min(min(total_tiles, (max_vis + IQS_KEYS - 1) / IQS_KEYS), first_tile + tiles);
    if (first_tile >= end_tile) return;

    for (int x = tid; x < IQS_ROWS * 32; x += 256) {
        const int row = row0 + x / 32;
        w[x] = row < rows ? __bfloat162float(iw[static_cast<long long>(row) * iw_stride + iw_col + x % 32]) * 0.1767766952966369f : 0.0f;
    }
    __syncthreads();

    const int pair0 = warp * 16 + g, pair1 = pair0 + 8;
    const int row_a0 = row0 + pair0 / 32, row_a1 = row0 + pair1 / 32;
    const int head_a0 = pair0 % 32, head_a1 = pair1 % 32;
    const __nv_bfloat16* a0 = row_a0 < rows ? qi + (static_cast<long long>(row_a0) * 32 + head_a0) * 128 : nullptr;
    const __nv_bfloat16* a1 = row_a1 < rows ? qi + (static_cast<long long>(row_a1) * 32 + head_a1) * 128 : nullptr;
    uint32_t af[8][4];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
        const int k0 = st * 16 + 2 * q;
        af[st][0] = a0 ? *reinterpret_cast<const uint32_t*>(a0 + k0) : 0;
        af[st][1] = a1 ? *reinterpret_cast<const uint32_t*>(a1 + k0) : 0;
        af[st][2] = a0 ? *reinterpret_cast<const uint32_t*>(a0 + k0 + 8) : 0;
        af[st][3] = a1 ? *reinterpret_cast<const uint32_t*>(a1 + k0 + 8) : 0;
    }

    for (int ahead = 0; ahead < STAGES; ++ahead) {
        const int prefetch_tile = first_tile + ahead < end_tile ? first_tile + ahead : total_tiles;
        iqs_prefetch<STAGES>(stage_base + ahead * IQS_KEYS * IQS_STR, keys, nkeys, prefetch_tile, tid);
    }

    for (int tile = first_tile; tile < end_tile; ++tile) {
        iqs_wait<STAGES - 1>();
        __syncthreads();
        const int slot = (tile - first_tile) % STAGES;
        __nv_bfloat16* ks = stage_base + slot * IQS_KEYS * IQS_STR;
        float acc[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            const int k0 = st * 16 + 2 * q;
            const uint32_t a[4] = { af[st][0], af[st][1], af[st][2], af[st][3] };
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const __nv_bfloat16* b0 = ks + (nt * 8 + g) * IQS_STR;
                const uint32_t b[2] = { *reinterpret_cast<const uint32_t*>(b0 + k0), *reinterpret_cast<const uint32_t*>(b0 + k0 + 8) };
                ism_mma(acc[nt], a, b);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int col = nt * 8 + 2 * q;
            rel[(warp * 16 + g) * IQS_KEYS + col] = fmaxf(acc[nt][0] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g) * IQS_KEYS + col + 1] = fmaxf(acc[nt][1] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * IQS_KEYS + col] = fmaxf(acc[nt][2] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * IQS_KEYS + col + 1] = fmaxf(acc[nt][3] * 0.08838834764831845f, 0.0f);
        }
        __syncthreads();
        // One group is committed every tile (empty past the end), so wait_group<STAGES - 1> always covers the next tile.
        if (tile + STAGES < end_tile) iqs_prefetch<STAGES>(stage_base + slot * IQS_KEYS * IQS_STR, keys, nkeys, tile + STAGES, tid);
        else asm volatile("cp.async.commit_group;\n" ::: "memory");
        const int key0 = tile * IQS_KEYS;
        const int rr = tid / IQS_KEYS, key = key0 + tid % IQS_KEYS, row = row0 + rr;
        if (row < rows && key < nkeys && key0 < ((row_vis + IQS_KEYS - 1) / IQS_KEYS) * IQS_KEYS) {
            if (static_cast<long long>(key) * world + rank > pos[row]) {
                scores[static_cast<long long>(row) * key_stride + key] = -CUDART_INF_F;
            } else {
                float total = 0.0f;
                for (int h = 0; h < 32; ++h) total = fmaf(w[rr * 32 + h], rel[(rr * 32 + h) * IQS_KEYS + tid % IQS_KEYS], total);
                scores[static_cast<long long>(row) * key_stride + key] = total;
            }
        }
        // no barrier here: the next tile's wait + barrier comes before any thread rewrites rel
    }
    iqs_wait<0>();
    __syncthreads();
}

template <int STAGES>
__device__ __forceinline__ void tf_glm53_index_scores_qs_entry(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles, char* smem) {
    TF_VBLOCK_ENTER();
    tf_glm53_index_scores_qs_task<STAGES>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_qs_s2(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_qs_entry<2>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_qs_s3(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_qs_entry<3>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

constexpr int IQS2_SMEM(int stages, int keys) {
    return stages * keys * IQS_STR * (int)sizeof(__nv_bfloat16) + IQS_ROWS * 32 * keys * (int)sizeof(float) + IQS_WEIGHT_BYTES;
}
static_assert(IQS2_SMEM(3, 32) <= 50176, "qs2 k32 s3 must fit two CTAs per SM");

template <int STAGES, int KEYS>
__device__ __forceinline__ void tf_glm53_index_scores_qs2_task(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles, char* smem) {
    const int row0 = TF_VBID.y * IQS_ROWS, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
    const int total_tiles = (nkeys + KEYS - 1) / KEYS;
    const int first_tile = TF_VBID.x * tiles;
    __nv_bfloat16* stage_base = reinterpret_cast<__nv_bfloat16*>(smem);
    float* rel = reinterpret_cast<float*>(smem + STAGES * KEYS * IQS_STR * sizeof(__nv_bfloat16));
    float* w = reinterpret_cast<float*>(reinterpret_cast<char*>(rel) + IQS_ROWS * 32 * KEYS * sizeof(float));
    __shared__ int vis_sh[IQS_ROWS];
    if (tid < IQS_ROWS) vis_sh[tid] = iqs_vis(row0 + tid, rows, vis_rows, pos, rank, world);
    __syncthreads();
    int max_vis = 0;
    for (int rr = 0; rr < IQS_ROWS; ++rr) max_vis = max(max_vis, vis_sh[rr]);
    const int rounded_vis = ((max_vis + 63) / 64) * 64;
    const int end_tile = min(min(total_tiles, (rounded_vis + KEYS - 1) / KEYS), first_tile + tiles);
    if (first_tile >= end_tile) return;

    for (int x = tid; x < IQS_ROWS * 32; x += 256) {
        const int row = row0 + x / 32;
        w[x] = row < rows ? __bfloat162float(iw[static_cast<long long>(row) * iw_stride + iw_col + x % 32]) * 0.1767766952966369f : 0.0f;
    }
    __syncthreads();

    const int pair0 = warp * 16 + g, pair1 = pair0 + 8;
    const int row_a0 = row0 + pair0 / 32, row_a1 = row0 + pair1 / 32;
    const int head_a0 = pair0 % 32, head_a1 = pair1 % 32;
    const __nv_bfloat16* a0 = row_a0 < rows ? qi + (static_cast<long long>(row_a0) * 32 + head_a0) * 128 : nullptr;
    const __nv_bfloat16* a1 = row_a1 < rows ? qi + (static_cast<long long>(row_a1) * 32 + head_a1) * 128 : nullptr;
    uint32_t af[8][4];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
        const int k0 = st * 16 + 2 * q;
        af[st][0] = a0 ? *reinterpret_cast<const uint32_t*>(a0 + k0) : 0;
        af[st][1] = a1 ? *reinterpret_cast<const uint32_t*>(a1 + k0) : 0;
        af[st][2] = a0 ? *reinterpret_cast<const uint32_t*>(a0 + k0 + 8) : 0;
        af[st][3] = a1 ? *reinterpret_cast<const uint32_t*>(a1 + k0 + 8) : 0;
    }

    for (int ahead = 0; ahead < STAGES; ++ahead) {
        const int prefetch_tile = first_tile + ahead < end_tile ? first_tile + ahead : total_tiles;
        for (int piece = tid; piece < KEYS * 16; piece += 256) {
            const int key = prefetch_tile * KEYS + piece / 16, chunk = piece % 16;
            const bool valid = key < nkeys;
            const __nv_bfloat16* src = valid ? keys + static_cast<long long>(key) * 128 + chunk * 8 : keys;
            iqs_cp16(stage_base + ahead * KEYS * IQS_STR + (piece / 16) * IQS_STR + chunk * 8, src, valid);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    for (int tile = first_tile; tile < end_tile; ++tile) {
        iqs_wait<STAGES - 1>();
        __syncthreads();
        const int slot = (tile - first_tile) % STAGES;
        __nv_bfloat16* ks = stage_base + slot * KEYS * IQS_STR;
        float acc[KEYS / 8][4];
#pragma unroll
        for (int nt = 0; nt < KEYS / 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            const uint32_t a[4] = { af[st][0], af[st][1], af[st][2], af[st][3] };
#pragma unroll
            for (int np = 0; np < KEYS / 16; ++np) {
                const int key_row = np * 16 + ((lane >> 4) << 3) + (lane & 7);
                const int half = st * 16 + ((lane >> 3) & 1) * 8;
                const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ks + key_row * IQS_STR + half));
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr) : "memory");
                ism_mma(acc[2 * np], a, b);
                ism_mma(acc[2 * np + 1], a, b + 2);
            }
        }
#pragma unroll
        for (int nt = 0; nt < KEYS / 8; ++nt) {
            const int col = nt * 8 + 2 * q;
            rel[(warp * 16 + g) * KEYS + col] = fmaxf(acc[nt][0] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g) * KEYS + col + 1] = fmaxf(acc[nt][1] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * KEYS + col] = fmaxf(acc[nt][2] * 0.08838834764831845f, 0.0f);
            rel[(warp * 16 + g + 8) * KEYS + col + 1] = fmaxf(acc[nt][3] * 0.08838834764831845f, 0.0f);
        }
        __syncthreads();
        if (tile + STAGES < end_tile) {
            const int prefetch_tile = tile + STAGES;
            for (int piece = tid; piece < KEYS * 16; piece += 256) {
                const int key = prefetch_tile * KEYS + piece / 16, chunk = piece % 16;
                const bool valid = key < nkeys;
                const __nv_bfloat16* src = valid ? keys + static_cast<long long>(key) * 128 + chunk * 8 : keys;
                iqs_cp16(stage_base + slot * KEYS * IQS_STR + (piece / 16) * IQS_STR + chunk * 8, src, valid);
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        } else {
            asm volatile("cp.async.commit_group;\n" ::: "memory");  // one group a tile, as the qs task: wait_group<STAGES - 1> then always covers the next tile
        }
        if (tid < 4 * KEYS) {
            const int rr = tid / KEYS, key_offset = tid % KEYS, key = tile * KEYS + key_offset, row = row0 + rr;
            const int row_vis = vis_sh[min(tid / KEYS, IQS_ROWS - 1)];
            const int key64 = (key / 64) * 64;
            if (row < rows && key < nkeys && key64 < ((row_vis + 63) / 64) * 64) {
                if (static_cast<long long>(key) * world + rank > pos[row]) {
                    scores[static_cast<long long>(row) * key_stride + key] = -CUDART_INF_F;
                } else {
                    float total = 0.0f;
                    for (int h = 0; h < 32; ++h) total = fmaf(w[rr * 32 + h], rel[(rr * 32 + h) * KEYS + key_offset], total);
                    scores[static_cast<long long>(row) * key_stride + key] = total;
                }
            }
        }
    }
    iqs_wait<0>();
    __syncthreads();
}

template <int STAGES, int KEYS>
__device__ __forceinline__ void tf_glm53_index_scores_qs2_entry(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles, char* smem) {
    TF_VBLOCK_ENTER();
    tf_glm53_index_scores_qs2_task<STAGES, KEYS>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_index_scores_qs2_k64_s2(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_qs2_entry<2, 64>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_index_scores_qs2_k32_s2(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_qs2_entry<2, 32>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_glm53_index_scores_qs2_k32_s3(const __nv_bfloat16* qi, const __nv_bfloat16* iw,
                              int iw_stride, int iw_col, const __nv_bfloat16* keys, const long long* pos, float* scores,
                              int nkeys, int key_stride, int rank, int world, int rows, int vis_rows, int tiles) {
    extern __shared__ __align__(32) char smem[];
    tf_glm53_index_scores_qs2_entry<3, 32>(qi, iw, iw_stride, iw_col, keys, pos, scores, nkeys, key_stride, rank, world, rows, vis_rows, tiles, smem);
}

__device__ __forceinline__ bool before(Candidate a, Candidate b) {
    const float as = a.score + 0.0f, bs = b.score + 0.0f;
    uint32_t ai = __float_as_uint(as), bi = __float_as_uint(bs);
    ai = (ai & 0x80000000u) ? ~ai : (ai ^ 0x80000000u);
    bi = (bi & 0x80000000u) ? ~bi : (bi ^ 0x80000000u);
    return ai > bi || (ai == bi && a.token < b.token);
}

// Shared historical sort and output path, also used for the rare NaN ordering fallback in merge2.
__device__ __forceinline__ void tf_glm53_merge_bitonic_task(Candidate* values, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int k, int stride) {
    const int r = TF_VBID.x, t = threadIdx.x;
    for (int size = 2; size <= 4096; size <<= 1) {
        for (int half = size >> 1; half; half >>= 1) {
            for (int i = t; i < 4096; i += 512) {
                const int partner = i ^ half;
                if (partner > i) {
                    const bool up = (i & size) == 0;
                    const Candidate a = values[i], b = values[partner];
                    const bool swap = up ? before(b, a) : before(a, b);
                    if (swap) { values[i] = b; values[partner] = a; }
                }
            }
            TF_TASK_SYNC();
        }
    }
    for (int i = t; i < 4096; i += 512) {
        if (i >= k || values[i].token == 0x7fffffff) values[i] = { -CUDART_INF_F, 0x7fffffff };
    }
    __shared__ int nvalid;
    if (t == 0) {
        nvalid = 0;
        while (nvalid < k && values[nvalid].token != 0x7fffffff) ++nvalid;
        global_counts[r] = nvalid;
    }
    TF_TASK_SYNC();
    // Sort the selected prefix by token ascending. Scores are now irrelevant.
    for (int size = 2; size <= 4096; size <<= 1) {
        for (int half = size >> 1; half; half >>= 1) {
            for (int i = t; i < 4096; i += 512) {
                const int partner = i ^ half;
                if (partner > i) {
                    const bool up = (i & size) == 0;
                    Candidate a = values[i], b = values[partner];
                    const int ka = a.token, kb = b.token;
                    const bool swap = up ? kb < ka : ka < kb;
                    if (swap) { values[i] = b; values[partner] = a; }
                }
            }
            TF_TASK_SYNC();
        }
    }
    for (int i = t; i < k; i += 512) {
        const int token = i < nvalid ? values[i].token : -1;
        global[static_cast<long long>(r) * stride + i] = token;
    }
    __shared__ int scan_ws[2 * NW], selected_count;
    int total = 0;
    for (int base = 0; base < k; base += 512) {
        const int token = base + t < k ? values[base + t].token : 0x7fffffff;
        const int keep = pos[r] + 1 > 2048 && token != 0x7fffffff && token % world == rank;
        int pass_total = 0;
        const int dst = block_scan(keep, scan_ws, &pass_total);
        if (keep) local[static_cast<long long>(r) * stride + total + dst] = token / world;
        total += pass_total;
    }
    if (t == 0) selected_count = pos[r] + 1 > 2048 ? total : 0;
    TF_TASK_SYNC();
    for (int i = selected_count + t; i < k; i += 512) local[static_cast<long long>(r) * stride + i] = -1;
    if (t == 0 && counts) counts[r] = selected_count;
}

// One packed [rank][row][k] exchange; global top-k, sparse filter and rank-slot remap in one CTA.
__device__ __forceinline__ void tf_glm53_merge_task(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int stride, unsigned char* storage) {
    const int r = TF_VBID.x, t = threadIdx.x;
    const int candidates = 2 * k;
    Candidate* values = reinterpret_cast<Candidate*>(storage);
    for (int i = t; i < candidates; i += 512) {
        const int which = i / k, col = i % k;
        const long long j = static_cast<long long>(which) * rows * k + static_cast<long long>(r) * k + col;
        const Candidate candidate = packed[j];
        values[i] = candidate.token < 0 || (isinf(candidate.score) && candidate.score < 0) ? Candidate{ -CUDART_INF_F, 0x7fffffff } : candidate;
    }
    for (int i = candidates + t; i < 4096; i += 512) values[i] = { -CUDART_INF_F, 0x7fffffff };
    TF_TASK_SYNC();
    tf_glm53_merge_bitonic_task(values, global, global_counts, local, counts, pos, rank, world, k, stride);
}

extern "C" __global__ void tf_glm53_merge(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int stride) {
    TF_VBLOCK_ENTER();
    extern __shared__ unsigned char storage[];
    tf_glm53_merge_task(packed, global, global_counts, local, counts, pos, rank, world, rows, k, stride, storage);
}

// Exact threshold select followed by stable compaction and a merge-path rank calculation. Input lists are
// already token-ascending; only the selected prefix is compacted, so the historical output order is preserved.
__device__ __forceinline__ void tf_glm53_merge2_task(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int block_rows, int stride,
                                            unsigned char* storage) {
    const int r = TF_VBID.x, t = threadIdx.x;
    Candidate* values = reinterpret_cast<Candidate*>(storage);
    __shared__ unsigned hist[256];
    __shared__ unsigned threshold, tie_token, radix_prefix, radix_mask;
    __shared__ int valid_count, take_equal, radix_target, equal_count, n_a, n_b, scan_ws[2 * NW], tile_total, offset, negative_nan;
    const int block = r / block_rows, within = r % block_rows;
    const int block_count = min(block_rows, rows - block * block_rows);
    const long long block_base = static_cast<long long>(block) * 2 * block_rows * k;
    for (int i = t; i < 2 * k; i += NT) {
        const int which = i / k, col = i % k;
        const long long src = block_base + static_cast<long long>(which) * block_count * k + static_cast<long long>(within) * k + col;
        Candidate c = packed[src];
        if (c.token < 0 || (isinf(c.score) && c.score < 0)) c = { -CUDART_INF_F, 0x7fffffff };
        values[which * k + col] = c;
    }
    TF_TASK_SYNC();
    if (t == 0) { valid_count = 0; negative_nan = 0; }
    TF_TASK_SYNC();
    for (int i = t; i < 2 * k; i += NT) if (values[i].token != 0x7fffffff) atomicAdd(&valid_count, 1);
    TF_TASK_SYNC();
    for (int i = t; i < 2 * k; i += NT) {
        if (values[i].token != 0x7fffffff) {
            const unsigned bits = __float_as_uint(values[i].score + 0.0f);
            const unsigned image = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
            if (image < 0x007fffffu) atomicExch(&negative_nan, 1);
        }
    }
    TF_TASK_SYNC();
    if (negative_nan) {
        for (int i = 2 * k + t; i < 4096; i += NT) values[i] = { -CUDART_INF_F, 0x7fffffff };
        TF_TASK_SYNC();
        tf_glm53_merge_bitonic_task(values, global, global_counts, local, counts, pos, rank, world, k, stride);
        return;
    }
    if (t == 0) tie_token = 0xffffffffu;
    if (valid_count <= k) {
        if (t == 0) { threshold = 0; take_equal = k; }
    } else {
        if (t == 0) { radix_prefix = 0; radix_mask = 0; radix_target = k - 1; }
        TF_TASK_SYNC();
        for (int pass = 0; pass < 4; ++pass) {
            if (t < 256) hist[t] = 0;
            TF_TASK_SYNC();
            const int shift = 24 - pass * 8;
            for (int i = t; i < 2 * k; i += NT) {
                const Candidate c = values[i];
                if (c.token != 0x7fffffff) {
                    const unsigned image = __float_as_uint(c.score + 0.0f);
                    const unsigned ordered = (image & 0x80000000u) ? ~image : (image ^ 0x80000000u);
                    if ((ordered & radix_mask) == radix_prefix) atomicAdd(&hist[(ordered >> shift) & 255u], 1);
                }
            }
            TF_TASK_SYNC();
            if (t == 0) {
                int b = 255;
                for (; b >= 0; --b) { if (radix_target < static_cast<int>(hist[b])) break; radix_target -= hist[b]; }
                radix_prefix |= static_cast<unsigned>(b) << shift;
                radix_mask |= 255u << shift;
            }
            if (t == 0) equal_count = hist[(radix_prefix >> shift) & 255u];
            TF_TASK_SYNC();
        }
        if (t == 0) threshold = radix_prefix;
        if (t == 0) {
            // After all four digits radix_target is the zero-based rank within the exact threshold image.
            take_equal = radix_target + 1;
            // The equal-score boundary uses the old total order's lower-token-first tie break.
            if (take_equal < equal_count) {
                for (int b = 0; b < 256; ++b) hist[b] = 0;
                for (int i = 0; i < 2 * k; ++i) {
                    const Candidate c = values[i];
                    if (c.token == 0x7fffffff) continue;
                    const unsigned bits = __float_as_uint(c.score + 0.0f);
                    const unsigned image = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
                    if (image == threshold) ++hist[(static_cast<unsigned>(c.token) >> 24) & 255u];
                }
                int target_token = take_equal - 1, digit = 0;
                for (; digit < 256; ++digit) { if (target_token < static_cast<int>(hist[digit])) break; target_token -= hist[digit]; }
                unsigned token_prefix = static_cast<unsigned>(digit) << 24;
                for (int shift = 16; shift >= 0; shift -= 8) {
                    for (int b = 0; b < 256; ++b) hist[b] = 0;
                    for (int i = 0; i < 2 * k; ++i) {
                        const Candidate c = values[i];
                        if (c.token == 0x7fffffff) continue;
                        const unsigned bits = __float_as_uint(c.score + 0.0f);
                        const unsigned image = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
                        if (image == threshold && (static_cast<unsigned>(c.token) >> (shift + 8)) == (token_prefix >> (shift + 8)))
                            ++hist[(static_cast<unsigned>(c.token) >> shift) & 255u];
                    }
                    digit = 0;
                    for (; digit < 256; ++digit) { if (target_token < static_cast<int>(hist[digit])) break; target_token -= hist[digit]; }
                    token_prefix |= static_cast<unsigned>(digit) << shift;
                }
                tie_token = token_prefix;
            }
        }
    }
    TF_TASK_SYNC();
    const unsigned boundary = threshold;
    const int quota = take_equal;
    const unsigned tie = tie_token;
    // Stable, in-place compaction by rank list. Each 512-record tile is fully loaded before its writes can overlap it.
    if (t == 0) { n_a = 0; n_b = 0; }
    TF_TASK_SYNC();
    for (int which = 0; which < 2; ++which) {
        if (t == 0) offset = 0;
        TF_TASK_SYNC();
        for (int base = 0; base < k; base += NT) {
            const int i = base + t;
            Candidate c = i < k ? values[which * k + i] : Candidate{ -CUDART_INF_F, 0x7fffffff };
            int keep = 0;
            if (i < k && c.token != 0x7fffffff) {
                const unsigned bits = __float_as_uint(c.score + 0.0f);
                const unsigned image = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
                keep = valid_count <= k || image > boundary || (image == boundary && (quota >= valid_count || static_cast<unsigned>(c.token) <= tie));
            }
            const int dst = block_scan(keep, scan_ws, &tile_total);
            const int before = offset;
            TF_TASK_SYNC();
            if (keep) values[which * k + before + dst] = c;
            if (t == 0) offset += tile_total;
            TF_TASK_SYNC();
        }
        if (t == 0) { if (which == 0) n_a = offset; else n_b = offset; }
        TF_TASK_SYNC();
    }
    if (t == 0) {
        global_counts[r] = n_a + n_b;
    }
    TF_TASK_SYNC();
    for (int i = t; i < n_a; i += NT) {
        const int token = values[i].token;
        int lo = 0, hi = n_b;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (values[k + mid].token < token) lo = mid + 1; else hi = mid; }
        global[static_cast<long long>(r) * stride + i + lo] = token;
    }
    for (int j = t; j < n_b; j += NT) {
        const int token = values[k + j].token;
        int lo = 0, hi = n_a;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (values[mid].token < token) lo = mid + 1; else hi = mid; }
        global[static_cast<long long>(r) * stride + j + lo] = token;
    }
    for (int i = t; i < k; i += NT) if (i >= n_a + n_b) global[static_cast<long long>(r) * stride + i] = -1;
    TF_TASK_SYNC();
    const int own_n = rank == 0 ? n_a : n_b;
    const int own_base = rank == 0 ? 0 : k;
    const int selected_count = pos[r] + 1 > 2048 ? own_n : 0;
    for (int i = t; i < k; i += NT) local[static_cast<long long>(r) * stride + i] = i < selected_count ? values[own_base + i].token / world : -1;
    if (t == 0 && counts) counts[r] = selected_count;
}

extern "C" __global__ void tf_glm53_merge2(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int block_rows, int stride) {
    TF_VBLOCK_ENTER();
    extern __shared__ unsigned char storage[];
    tf_glm53_merge2_task(packed, global, global_counts, local, counts, pos, rank, world, rows, k, block_rows, stride, storage);
}

// One warp scans the 256 bins as eight a lane; both helpers pick the bin that holds the target rank.
__device__ __forceinline__ void radix_pick_desc(const unsigned* h, int target, int* out_digit, int* out_greater) {
    const int lane = threadIdx.x & 31;
    int loc[8], run = 0;
    for (int x = 7; x >= 0; --x) { run += static_cast<int>(h[lane * 8 + x]); loc[x] = run; }
    int suf = run;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_down_sync(FULL, suf, o);
        if (lane + o < 32) suf += y;
    }
    const int after = suf - run;
    int best = -1, greater = 0;
    for (int x = 7; x >= 0; --x) {
        if (best < 0 && after + loc[x] >= target + 1) {
            best = lane * 8 + x;
            greater = after + loc[x] - static_cast<int>(h[best]);
        }
    }
    const unsigned has = __ballot_sync(FULL, best >= 0);
    const int src = has ? 31 - __clz(has) : 0;
    *out_digit = __shfl_sync(FULL, best, src);
    *out_greater = __shfl_sync(FULL, greater, src);
}

__device__ __forceinline__ void radix_pick_asc(const unsigned* h, int target, int* out_digit, int* out_below) {
    const int lane = threadIdx.x & 31;
    int run = 0;
    for (int x = 0; x < 8; ++x) run += static_cast<int>(h[lane * 8 + x]);
    int pre = run;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, pre, o);
        if (lane >= o) pre += y;
    }
    const int before = pre - run;
    int best = -1, below = 0, acc = before;
    for (int x = 0; x < 8; ++x) {
        const int cnt = static_cast<int>(h[lane * 8 + x]);
        if (best < 0 && acc + cnt >= target + 1) { best = lane * 8 + x; below = acc; }
        acc += cnt;
    }
    const unsigned has = __ballot_sync(FULL, best >= 0);
    const int src = has ? __ffs(has) - 1 : 0;
    *out_digit = __shfl_sync(FULL, best, src);
    *out_below = __shfl_sync(FULL, below, src);
}

// Same select and compaction as tf_glm53_merge2, with the ordered image cached in the record beside the token.
// Missing records keep their raw score so the NaN fallback sees exactly the input merge2 saw.
__device__ __forceinline__ void tf_glm53_merge2_fast_task(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int block_rows, int stride,
                                            unsigned char* storage) {
    const int r = TF_VBID.x, t = threadIdx.x, lane = t & 31;
    Candidate* values = reinterpret_cast<Candidate*>(storage);
    __shared__ unsigned hist[256];
    __shared__ unsigned threshold, tie_token, radix_prefix, radix_mask, tie_prefix;
    __shared__ int valid_count, take_equal, radix_target, equal_count, n_a, n_b, scan_ws[2 * NW], tile_total, offset, negative_nan, tie_target;
    const int block = r / block_rows, within = r % block_rows;
    const int block_count = min(block_rows, rows - block * block_rows);
    const long long block_base = static_cast<long long>(block) * 2 * block_rows * k;
    for (int i = t; i < 2 * k; i += NT) {
        const int which = i / k, col = i % k;
        const long long src = block_base + static_cast<long long>(which) * block_count * k + static_cast<long long>(within) * k + col;
        Candidate c = packed[src];
        if (c.token < 0 || (isinf(c.score) && c.score < 0)) {
            c = { -CUDART_INF_F, 0x7fffffff };
        } else if (c.token != 0x7fffffff) {
            const unsigned bits = __float_as_uint(c.score + 0.0f);
            c.score = __uint_as_float((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u));
        }
        values[which * k + col] = c;
    }
    TF_TASK_SYNC();
    if (t == 0) { valid_count = 0; negative_nan = 0; }
    TF_TASK_SYNC();
    int warp_valid = 0;
    unsigned warp_nan = 0;
    for (int base = 0; base < 2 * k; base += NT) {
        const int i = base + t;
        int valid = 0;
        unsigned nan_here = 0;
        if (i < 2 * k) {
            const Candidate c = values[i];
            valid = c.token != 0x7fffffff;
            nan_here = valid && __float_as_uint(c.score) < 0x007fffffu;
        }
        warp_valid += __popc(__ballot_sync(FULL, valid != 0));
        warp_nan |= __ballot_sync(FULL, nan_here != 0);
    }
    if (lane == 0) {
        if (warp_valid) atomicAdd(&valid_count, warp_valid);
        if (warp_nan) atomicExch(&negative_nan, 1);
    }
    TF_TASK_SYNC();
    if (negative_nan) {
        for (int i = t; i < 2 * k; i += NT) {
            Candidate c = values[i];
            if (c.token != 0x7fffffff) {
                const unsigned image = __float_as_uint(c.score);
                c.score = __uint_as_float((image & 0x80000000u) ? (image ^ 0x80000000u) : ~image);
            }
            values[i] = c;
        }
        for (int i = 2 * k + t; i < 4096; i += NT) values[i] = { -CUDART_INF_F, 0x7fffffff };
        TF_TASK_SYNC();
        tf_glm53_merge_bitonic_task(values, global, global_counts, local, counts, pos, rank, world, k, stride);
        return;
    }
    if (t == 0) tie_token = 0xffffffffu;
    if (valid_count <= k) {
        if (t == 0) { threshold = 0; take_equal = k; }
    } else {
        if (t == 0) { radix_prefix = 0; radix_mask = 0; radix_target = k - 1; }
        TF_TASK_SYNC();
        for (int pass = 0; pass < 4; ++pass) {
            if (t < 256) hist[t] = 0;
            TF_TASK_SYNC();
            const int shift = 24 - pass * 8;
            for (int base = 0; base < 2 * k; base += NT) {
                const int i = base + t;
                if (i < 2 * k) {
                    const Candidate c = values[i];
                    const unsigned image = __float_as_uint(c.score);
                    if (c.token != 0x7fffffff && (image & radix_mask) == radix_prefix)
                        atomicAdd(&hist[(image >> shift) & 255u], 1);
                }
            }
            TF_TASK_SYNC();
            if (t < 32) {
                int dg = 0, greater = 0;
                radix_pick_desc(hist, radix_target, &dg, &greater);
                if (t == 0) {
                    radix_prefix |= static_cast<unsigned>(dg) << shift;
                    radix_mask |= 255u << shift;
                    radix_target -= greater;
                    equal_count = static_cast<int>(hist[dg]);
                }
            }
            TF_TASK_SYNC();
        }
        if (t == 0) threshold = radix_prefix;
        TF_TASK_SYNC();
        if (t == 0) {
            take_equal = radix_target + 1;
            tie_target = take_equal - 1;
            tie_prefix = 0;
        }
        TF_TASK_SYNC();
        if (take_equal < equal_count) {
            for (int shift = 24; shift >= 0; shift -= 8) {
                if (t < 256) hist[t] = 0;
                TF_TASK_SYNC();
                for (int base = 0; base < 2 * k; base += NT) {
                    const int i = base + t;
                    if (i < 2 * k) {
                        const Candidate c = values[i];
                        const unsigned token = static_cast<unsigned>(c.token);
                        const bool match = c.token != 0x7fffffff && __float_as_uint(c.score) == threshold &&
                            (shift == 24 || (token >> (shift + 8)) == (tie_prefix >> (shift + 8)));
                        if (match) atomicAdd(&hist[(token >> shift) & 255u], 1);
                    }
                }
                TF_TASK_SYNC();
                if (t < 32) {
                    int dg = 0, below = 0;
                    radix_pick_asc(hist, tie_target, &dg, &below);
                    if (t == 0) {
                        tie_prefix |= static_cast<unsigned>(dg) << shift;
                        tie_target -= below;
                    }
                }
                TF_TASK_SYNC();
            }
            if (t == 0) tie_token = tie_prefix;
        }
    }
    TF_TASK_SYNC();
    const unsigned boundary = threshold;
    const int quota = take_equal;
    const unsigned tie = tie_token;
    if (t == 0) { n_a = 0; n_b = 0; }
    TF_TASK_SYNC();
    for (int which = 0; which < 2; ++which) {
        if (t == 0) offset = 0;
        TF_TASK_SYNC();
        for (int base = 0; base < k; base += NT) {
            const int i = base + t;
            Candidate c = i < k ? values[which * k + i] : Candidate{ -CUDART_INF_F, 0x7fffffff };
            int keep = 0;
            if (i < k && c.token != 0x7fffffff) {
                const unsigned image = __float_as_uint(c.score);
                keep = valid_count <= k || image > boundary || (image == boundary && (quota >= valid_count || static_cast<unsigned>(c.token) <= tie));
            }
            const int dst = block_scan(keep, scan_ws, &tile_total);
            const int before = offset;
            TF_TASK_SYNC();
            if (keep) values[which * k + before + dst] = c;
            if (t == 0) offset += tile_total;
            TF_TASK_SYNC();
        }
        if (t == 0) { if (which == 0) n_a = offset; else n_b = offset; }
        TF_TASK_SYNC();
    }
    if (t == 0) {
        global_counts[r] = n_a + n_b;
    }
    TF_TASK_SYNC();
    for (int i = t; i < n_a; i += NT) {
        const int token = values[i].token;
        int lo = 0, hi = n_b;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (values[k + mid].token < token) lo = mid + 1; else hi = mid; }
        global[static_cast<long long>(r) * stride + i + lo] = token;
    }
    for (int j = t; j < n_b; j += NT) {
        const int token = values[k + j].token;
        int lo = 0, hi = n_a;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (values[mid].token < token) lo = mid + 1; else hi = mid; }
        global[static_cast<long long>(r) * stride + j + lo] = token;
    }
    for (int i = t; i < k; i += NT) if (i >= n_a + n_b) global[static_cast<long long>(r) * stride + i] = -1;
    TF_TASK_SYNC();
    const int own_n = rank == 0 ? n_a : n_b;
    const int own_base = rank == 0 ? 0 : k;
    const int selected_count = pos[r] + 1 > 2048 ? own_n : 0;
    for (int i = t; i < k; i += NT) local[static_cast<long long>(r) * stride + i] = i < selected_count ? values[own_base + i].token / world : -1;
    if (t == 0 && counts) counts[r] = selected_count;
}

extern "C" __global__ void tf_glm53_merge2_fast(const Candidate* packed, int* global, int* global_counts,
                                            int* local, int* counts, const long long* pos,
                                            int rank, int world, int rows, int k, int block_rows, int stride) {
    TF_VBLOCK_ENTER();
    extern __shared__ unsigned char storage[];
    tf_glm53_merge2_fast_task(packed, global, global_counts, local, counts, pos, rank, world, rows, k, block_rows, stride, storage);
}

}  // namespace glm53_topk
