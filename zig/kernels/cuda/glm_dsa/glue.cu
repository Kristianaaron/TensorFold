// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#include "../tf_vblock.cuh"
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/glue.py:_swiglu_plain,embed_pick
// (Apache-2.0). Changes: GLM-5.3
// FP32 low-rank, dense SwiGLU, residual and vocab owner-pick kernels.
#include "tasks.cuh"

extern "C" __global__ void tf_glm53_swiglu_plain(const __nv_bfloat16* gu, __nv_bfloat16* out, int rows, int width) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_swiglu_plain_params p{gu, out, rows, width};
  tf_glm53_swiglu_plain_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_lowrank_partial(const __nv_bfloat16* x, const float* a, float* partials,
    int rows, int hidden, int rank, int x_stride, int a_stride, int splits) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_lowrank_partial_params p{x, a, partials, rows, hidden, rank, x_stride, a_stride, splits};
  tf_glm53_lowrank_partial_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_lowrank_decode(const __nv_bfloat16* x, const float* a, float* partials,
    int rows, int hidden, int rank, int x_stride, int a_stride, int splits) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_lowrank_decode_params p{x, a, partials, rows, hidden, rank, x_stride, a_stride, splits};
  tf_glm53_lowrank_decode_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_lowrank_reduce(const float* partials, float* mid, int total, int splits) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_lowrank_reduce_params p{partials, mid, total, splits};
  tf_glm53_lowrank_reduce_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_add_partial(const float* add, float* acc, int count) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_add_partial_params p{add, acc, count};
  tf_glm53_add_partial_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_lowrank_combine_bf16(const float* routed, const float* shared,
    const float* b, const float* mid, __nv_bfloat16* out, int rows, int hidden, int rank, int b_stride) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_lowrank_combine_bf16_params p{routed, shared, b, mid, out, rows, hidden, rank, b_stride};
  tf_glm53_lowrank_combine_bf16_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_lowrank_combine_bf16_decode(const float* routed, const float* shared,
    const float* b, const float* mid, __nv_bfloat16* out, int rows, int hidden, int rank, int b_stride) {
  TF_VBLOCK_ENTER();
  const tf_glm53_lowrank_combine_bf16_params p{routed, shared, b, mid, out, rows, hidden, rank, b_stride};
  tf_glm53_lowrank_combine_bf16_decode_task(p, TF_VBID.x);
}

extern "C" __global__ void tf_glm53_moe_post(const float* y, const float* weights, const float* shared,
    const float* b, const float* mid, __nv_bfloat16* out, int rows, int hidden, int rank, int b_stride) {
  TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_moe_post_params p{y, weights, shared, b, mid, out, rows, hidden, rank, b_stride};
  tf_glm53_moe_post_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_moe_post_ep(const float* y, const float* weights, const int* picks, const float* shared,
                                                  const float* b, const float* mid, __nv_bfloat16* out,
                                                  int rows, int hidden, int rank, int b_stride) {
  TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_moe_post_ep_params p{y, weights, picks, shared, b, mid, out, rows, hidden, rank, b_stride};
  tf_glm53_moe_post_ep_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_residual_rmsnorm_6144(__nv_bfloat16* x, const float* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_residual_rmsnorm_6144_params p{x, parts, w, normed, rows, eps};
  tf_glm53_residual_rmsnorm_6144_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_residual_rmsnorm_6144_reg(__nv_bfloat16* x, const float* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_residual_rmsnorm_6144_params p{x, parts, w, normed, rows, eps};
  tf_glm53_residual_rmsnorm_6144_reg_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_residual_rmsnorm_6144_sum(__nv_bfloat16* x, const float* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_residual_rmsnorm_6144_sum_params p{x, parts, w, normed, rows, eps};
  tf_glm53_residual_rmsnorm_6144_sum_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_residual_rmsnorm_6144_sum_reg(__nv_bfloat16* x, const float* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
  TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_residual_rmsnorm_6144_sum_params p{x, parts, w, normed, rows, eps};
  tf_glm53_residual_rmsnorm_6144_sum_reg_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_ablate_6144(__nv_bfloat16* x, const void* parts, int kind, const float* r, const float* a,
    const __nv_bfloat16* w, __nv_bfloat16* normed, int rows, float eps) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_ablate_6144_params p{x, parts, kind, r, a, w, normed, rows, eps};
  tf_glm53_ablate_6144_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_moe_residual_add(const __nv_bfloat16* x, __nv_bfloat16* xout,
    const __nv_bfloat16* parts, int count) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_moe_residual_add_params p{x, xout, parts, count};
  tf_glm53_moe_residual_add_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_moe_residual_add_sum(const __nv_bfloat16* x, __nv_bfloat16* xout,
    const __nv_bfloat16* parts, int count) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_moe_residual_add_sum_params p{x, xout, parts, count};
  tf_glm53_moe_residual_add_sum_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

// TF_GLM53_ADD_NORM_FUSE=1: tf_glm53_moe_residual_add on one row, then the next layer's input norm on the stored
// bf16 row, in one launch. Each element's add uses the same arithmetic; the norm uses glm_glue's RMSNorm template.
extern "C" __global__ void __launch_bounds__(256)
tf_glm53_moe_residual_rmsnorm_6144(__nv_bfloat16* x, const __nv_bfloat16* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
  TF_VBLOCK_ENTER();
  const int row = TF_VBID.x;
  const int count = rows * 6144;
  for (int j = threadIdx.x; j < 6144; j += TF_VBDIM.x) {
    const int i = row * 6144 + j;
    const float rank_sum = __fadd_rn(__bfloat162float(parts[i]), __bfloat162float(parts[count + i]));
    const float residual = __fadd_rn(__bfloat162float(x[i]), __bfloat162float(__float2bfloat16_rn(rank_sum)));
    x[i] = __float2bfloat16_rn(residual);
  }
  TF_TASK_SYNC();
  glm_glue::rmsnorm<6144, 8192, 8, 8>(x, 6144, w, normed, 6144, nullptr, eps, 0, row);
}

extern "C" __global__ void tf_glm53_lowrank_finish(const float* b, const float* mid, float* acc,
    int rows, int hidden, int rank, int b_stride) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_lowrank_finish_params p{b, mid, acc, rows, hidden, rank, b_stride};
  tf_glm53_lowrank_finish_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void tf_glm53_embed_pick(const __nv_bfloat16* gathered, const int* owners,
    __nv_bfloat16* out, int rows, int hidden, int world) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_embed_pick_params p{gathered, owners, out, rows, hidden, world};
  tf_glm53_embed_pick_task(p, TF_VBID.x, TF_VNBLK.x, smem);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_unpack_bf16_rows(const unsigned short* packed, unsigned short* out, long long k, long long row0, long long rows) {
  long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= rows * k) return;
  long long row = row0 + idx / k, c = idx % k, rr = row % 64, cc = c % 64, q = cc % 16, rest = q % 8;
  long long lane = (rr % 8) * 4 + rest / 2, e = (q / 8) * 2 + rest % 2;
  long long tg = (row / 64) * (k / 64) + c / 64;
  out[idx] = packed[tg * 4096 + (rr / 8) * 512 + (cc / 16) * 128 + lane * 4 + e];
}

// TF_GLM53_QA_SHARD=1: q_a_norm over the two ranks' gathered 1024-column halves, bit-identical to
// tf_glm_rmsnorm_2048 on the assembled [rows][2048] row: glm_glue::rmsnorm<2048, 2048, 4, 8> with rep 0 (columns
// 0..1023) read from x0 and rep 1 (columns 1024..2047) from x1; the same values a thread holds and the same sum order.
extern "C" __global__ void __launch_bounds__(128)
tf_glm53_rmsnorm_2048_halves(const __nv_bfloat16* x0, const __nv_bfloat16* x1, int64_t half_stride,
    const __nv_bfloat16* w, __nv_bfloat16* out, int64_t o_stride, float eps) {
  TF_VBLOCK_ENTER();
  constexpr int D = 2048, NW = 4, S = 8, T = NW * 32, TILE = S * T, REPS = D / TILE;
  static_assert(REPS == 2 && TILE == 1024, "one rep per rank half");
  __shared__ float smem[NW];
  const int64_t r = TF_VBID.x;
  const int t = threadIdx.x;
  const bool xv0 = glm_glue::packed<S>(x0, half_stride), xv1 = glm_glue::packed<S>(x1, half_stride);
  const bool wv = glm_glue::packed<S>(w, 0), ov = glm_glue::packed<S>(out, o_stride);
  float v[REPS * S];
  glm_glue::load_s<S>(x0 + r * half_stride + t * S, v, xv0);
  glm_glue::load_s<S>(x1 + r * half_stride + t * S, v + S, xv1);
  float acc = __fmul_rn(v[1], v[1]);
  acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
  for (int k = 2; k < REPS * S; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  const float total = glm_triton::block_sum<NW>(acc, smem);
  const float rinv = glm_triton::div_full(1.0f, glm_triton::sqrt_approx(eps + glm_triton::div_full(total, (float)D)));
#pragma unroll
  for (int rep = 0; rep < REPS; ++rep) {
    const int d = rep * TILE + t * S;
    float wf[S];
    __nv_bfloat16 o[S];
    glm_glue::load_s<S>(w + d, wf, wv);
#pragma unroll
    for (int j = 0; j < S; ++j) o[j] = __float2bfloat16_rn(__fmul_rn(wf[j], glm_triton::rbf(__fmul_rn(rinv, v[rep * S + j]))));
    glm_glue::store_s<S>(out + r * o_stride + d, o, ov);
  }
}
