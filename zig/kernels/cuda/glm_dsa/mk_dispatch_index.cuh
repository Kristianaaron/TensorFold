// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 index adapters. The task math is adapted from bertholomus/TensorFold@ddfad35.
#pragma once
#include "index.cu"
// the index kernels live in namespace glm53_topk (extern "C" sets their linkage, not their lookup scope)
using glm53_topk::tf_glm53_index_q;
using glm53_topk::tf_glm53_index_key;
using glm53_topk::tf_glm53_index_scores;

struct tf_glm53_mk_index_query_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* input; const float *cos, *sin; __nv_bfloat16* output;
};
struct tf_glm53_mk_index_key_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16 *proj, *weight, *bias; const float *cos, *sin;
  const long long* positions; __nv_bfloat16* output;
  int stride, col, rank, world; float eps;
};
struct tf_glm53_mk_index_scores_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16 *qi, *iw; int iw_stride, iw_col;
  const __nv_bfloat16* keys; const long long* positions; float* scores;
  int nkeys, key_stride, rank, world;
};
static_assert(sizeof(tf_glm53_mk_index_query_params) == 48, "index query payload ABI");
static_assert(offsetof(tf_glm53_mk_index_query_params, input) == 16, "index query input ABI");
static_assert(offsetof(tf_glm53_mk_index_key_params, proj) == 16, "index key projection ABI");
static_assert(offsetof(tf_glm53_mk_index_scores_params, qi) == 16, "index scores query ABI");
static_assert(sizeof(tf_glm53_mk_index_key_params) == 96, "index key size ABI");
static_assert(sizeof(tf_glm53_mk_index_scores_params) == 80, "index scores size ABI");
struct tf_glm53_mk_sig_index_q { const __nv_bfloat16 *input; const float *cos, *sin; __nv_bfloat16* output; };
struct tf_glm53_mk_sig_index_key { const __nv_bfloat16 *proj, *weight, *bias; const float *cos, *sin; const long long* positions; __nv_bfloat16* output; int stride, col, rank, world; float eps; };
struct tf_glm53_mk_sig_index_scores { const __nv_bfloat16 *qi, *iw; int iw_stride, iw_col; const __nv_bfloat16* keys; const long long* positions; float* scores; int nkeys, key_stride, rank, world; };
TF_GLM53_MK_SIG(tf_glm53_index_q, tf_glm53_mk_sig_index_q, TF_GLM53_MK_T(tf_glm53_mk_sig_index_q, input), TF_GLM53_MK_T(tf_glm53_mk_sig_index_q, cos), TF_GLM53_MK_T(tf_glm53_mk_sig_index_q, sin), TF_GLM53_MK_T(tf_glm53_mk_sig_index_q, output));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_index_query_params, tf_glm53_mk_sig_index_q);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_query_params, tf_glm53_mk_sig_index_q, input); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_query_params, tf_glm53_mk_sig_index_q, cos); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_query_params, tf_glm53_mk_sig_index_q, sin); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_query_params, tf_glm53_mk_sig_index_q, output);
TF_GLM53_MK_SIG(tf_glm53_index_key, tf_glm53_mk_sig_index_key, TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, proj), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, weight), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, bias), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, cos), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, sin), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, positions), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, output), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, stride), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, col), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, rank), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, world), TF_GLM53_MK_T(tf_glm53_mk_sig_index_key, eps));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, proj); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, weight); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, bias); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, cos); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, sin); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, positions); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, output); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, stride); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, col); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, rank); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, world); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_key_params, tf_glm53_mk_sig_index_key, eps);
TF_GLM53_MK_SIG(tf_glm53_index_scores, tf_glm53_mk_sig_index_scores, TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, qi), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, iw), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, iw_stride), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, iw_col), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, keys), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, positions), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, scores), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, nkeys), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, key_stride), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, rank), TF_GLM53_MK_T(tf_glm53_mk_sig_index_scores, world));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, qi); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, iw); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, iw_stride); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, iw_col); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, keys); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, positions); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, scores); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, nkeys); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, key_stride); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, rank); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_index_scores_params, tf_glm53_mk_sig_index_scores, world);

__device__ __forceinline__ void tf_glm53_mk_index_vblock(uint32_t block, uint32_t gx, uint32_t gy,
                                                          uint32_t gz, uint32_t threads) {
  tf_vblock_set(make_uint3(block % gx, (block / gx) % gy, block / (gx * gy)),
                make_uint3(gx, gy, gz), make_uint3(threads, 1, 1));
}
__device__ __forceinline__ void tf_glm53_mk_run_index_query(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_index_query_params*>(raw);
  tf_glm53_mk_index_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  glm53_topk::tf_glm53_index_q_task(p.input, p.cos, p.sin, p.output);
}
__device__ __forceinline__ void tf_glm53_mk_run_index_key(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_index_key_params*>(raw);
  tf_glm53_mk_index_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  glm53_topk::tf_glm53_index_key_task(p.proj, p.weight, p.bias, p.cos, p.sin, p.positions, p.output,
                          p.stride, p.col, p.rank, p.world, p.eps);
}
__device__ __forceinline__ void tf_glm53_mk_run_index_scores(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_index_scores_params*>(raw);
  tf_glm53_mk_index_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  glm53_topk::tf_glm53_index_scores_task(p.qi, p.iw, p.iw_stride, p.iw_col, p.keys, p.positions,
                             p.scores, p.nkeys, p.key_stride, p.rank, p.world, smem);
}
