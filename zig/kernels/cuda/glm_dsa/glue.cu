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

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_residual_rmsnorm_6144(__nv_bfloat16* x, const float* parts, const __nv_bfloat16* w,
    __nv_bfloat16* normed, int rows, float eps) {
    TF_VBLOCK_ENTER();
  extern __shared__ char smem[];
  const tf_glm53_residual_rmsnorm_6144_params p{x, parts, w, normed, rows, eps};
  tf_glm53_residual_rmsnorm_6144_task(p, TF_VBID.x, TF_VNBLK.x, smem);
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
