// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 MoE task adapters. Each payload is a plain C ABI for one task kind.
#pragma once
#include "tasks.cuh"
#include "../glm_glue_router.cuh"
#include "../glm_glue_bmm.cuh"
#include "../glm_glue_moe.cuh"
#include "../glm_glue_rows.cuh"

#define TF_MK_MOE_HEAD uint32_t gx, gy, gz, threads
struct tf_glm53_mk_moe_post_norm_params { TF_MK_MOE_HEAD; tf_glm53_residual_rmsnorm_6144_params task; };
struct tf_glm53_mk_moe_router_part_params { tf_glm53_mk_hdr h; const __nv_bfloat16 *x, *w; float* parts; int64_t rows, stride, hidden, experts; int ks; };
struct tf_glm53_mk_moe_router_topk_params { tf_glm53_mk_hdr h; const float *logits, *bias; int* picks; float* weights; float scale; int experts, top_k, norm; };
struct tf_glm53_mk_moe_lowrank_partial_params { TF_MK_MOE_HEAD; tf_glm53_lowrank_partial_params task; };
struct tf_glm53_mk_moe_lowrank_reduce_params { TF_MK_MOE_HEAD; tf_glm53_lowrank_reduce_params task; };
struct tf_glm53_mk_moe_lowrank_finish_params { TF_MK_MOE_HEAD; tf_glm53_lowrank_finish_params task; };
struct tf_glm53_mk_moe_residual_add_params { TF_MK_MOE_HEAD; tf_glm53_moe_residual_add_params task; };
struct tf_glm53_mk_moe_swiglu_params { TF_MK_MOE_HEAD; tf_glm53_swiglu_plain_params task; };
struct tf_glm53_mk_moe_slice_sum_params { tf_glm53_mk_hdr h; const float* part; void* out; int64_t total; int ks, f32; };
#undef TF_MK_MOE_HEAD
static_assert(offsetof(tf_glm53_mk_moe_post_norm_params, task) == 16, "moe post payload ABI");
static_assert(offsetof(tf_glm53_mk_moe_router_part_params, x) == 16, "moe router payload ABI");
static_assert(offsetof(tf_glm53_mk_moe_router_topk_params, logits) == 16, "moe topk payload ABI");
static_assert(sizeof(tf_glm53_mk_moe_post_norm_params) == 56, "moe post size ABI");
static_assert(sizeof(tf_glm53_mk_moe_router_part_params) == 80, "moe router size ABI");
static_assert(sizeof(tf_glm53_mk_moe_router_topk_params) == 64, "moe topk size ABI");
static_assert(sizeof(tf_glm53_mk_moe_lowrank_partial_params) == 64, "moe lowrank partial size ABI");
static_assert(sizeof(tf_glm53_mk_moe_lowrank_reduce_params) == 40, "moe lowrank reduce size ABI");
static_assert(sizeof(tf_glm53_mk_moe_lowrank_finish_params) == 56, "moe lowrank finish size ABI");
static_assert(sizeof(tf_glm53_mk_moe_residual_add_params) == 48, "moe residual size ABI");
static_assert(sizeof(tf_glm53_mk_moe_swiglu_params) == 40, "moe swiglu size ABI");
static_assert(sizeof(tf_glm53_mk_moe_slice_sum_params) == 48, "moe slice sum size ABI");
static_assert(offsetof(tf_glm53_mk_moe_post_norm_params, task) == 16, "moe post task offset");
static_assert(offsetof(tf_glm53_mk_moe_lowrank_partial_params, task) == 16, "moe partial task offset");
static_assert(offsetof(tf_glm53_mk_moe_lowrank_reduce_params, task) == 16, "moe reduce task offset");
static_assert(offsetof(tf_glm53_mk_moe_lowrank_finish_params, task) == 16, "moe finish task offset");
static_assert(offsetof(tf_glm53_mk_moe_residual_add_params, task) == 16, "moe residual task offset");
static_assert(offsetof(tf_glm53_mk_moe_swiglu_params, task) == 16, "moe swiglu task offset");
struct tf_glm53_mk_sig_router_part { const __nv_bfloat16 *x, *w; float* parts; int64_t rows, stride, hidden, experts; int ks; };
struct tf_glm53_mk_sig_router_topk { const float *logits, *bias; int* picks; float* weights; float scale; int experts, top_k, norm; };
struct tf_glm53_mk_sig_slice_sum { const float* part; void* out; int64_t total; int ks, f32; };
TF_GLM53_MK_SIG(tf_glm_router_part, tf_glm53_mk_sig_router_part, TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, x), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, w), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, parts), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, stride), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, hidden), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, experts), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, ks));
TF_GLM53_MK_SIG(tf_glm_router_tile, tf_glm53_mk_sig_router_part, TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, x), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, w), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, parts), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, stride), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, hidden), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, experts), TF_GLM53_MK_T(tf_glm53_mk_sig_router_part, ks));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, x); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, w); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, parts); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, rows); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, stride); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, hidden); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, experts); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_part_params, tf_glm53_mk_sig_router_part, ks);
TF_GLM53_MK_SIG(tf_glm_topk, tf_glm53_mk_sig_router_topk, TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, logits), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, bias), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, picks), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, weights), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, scale), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, experts), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, top_k), TF_GLM53_MK_T(tf_glm53_mk_sig_router_topk, norm));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, logits); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, bias); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, picks); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, weights); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, scale); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, experts); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, top_k); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_router_topk_params, tf_glm53_mk_sig_router_topk, norm);
TF_GLM53_MK_SIG(tf_glm_slice_sum, tf_glm53_mk_sig_slice_sum, TF_GLM53_MK_T(tf_glm53_mk_sig_slice_sum, part), TF_GLM53_MK_T(tf_glm53_mk_sig_slice_sum, out), TF_GLM53_MK_T(tf_glm53_mk_sig_slice_sum, total), TF_GLM53_MK_T(tf_glm53_mk_sig_slice_sum, ks), TF_GLM53_MK_T(tf_glm53_mk_sig_slice_sum, f32));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum, part); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum, out); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum, total); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum, ks); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_moe_slice_sum_params, tf_glm53_mk_sig_slice_sum, f32);

__device__ __forceinline__ void tf_glm53_mk_run_moe_post_norm(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_post_norm_params*>(raw);
  tf_vblock_set(make_uint3(block % p.gx, (block / p.gx) % p.gy, block / (p.gx * p.gy)),
                make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_residual_rmsnorm_6144_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_router_part(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_router_part_params*>(raw);
  tf_vblock_set(make_uint3(block % p.h.gx, (block / p.h.gx) % p.h.gy, block / (p.h.gx * p.h.gy)),
                make_uint3(p.h.gx, p.h.gy, p.h.gz), make_uint3(p.h.threads, 1, 1));
  if (threadIdx.x >= p.h.threads) return;
  const bool tile = (reinterpret_cast<uintptr_t>(p.x) & 15u) == 0 && (p.stride & 7) == 0;
  if (tile) tf_glm_router_tile_task(p.x, p.w, p.parts, p.rows, p.stride, p.hidden, p.experts, p.ks, smem);
  else tf_glm_router_part_task(p.x, p.w, p.parts, p.rows, p.stride, p.hidden, p.experts, p.ks);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_router_topk(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_router_topk_params*>(raw);
  tf_vblock_set(make_uint3(block % p.h.gx, (block / p.h.gx) % p.h.gy, block / (p.h.gx * p.h.gy)),
                make_uint3(p.h.gx, p.h.gy, p.h.gz), make_uint3(p.h.threads, 1, 1));
  if (threadIdx.x >= p.h.threads) return;
  tf_glm_topk_task(p.logits, p.bias, p.picks, p.weights, p.scale, p.experts, p.top_k, p.norm);
}
__device__ __forceinline__ uint3 tf_glm53_mk_moe_bid(uint32_t block, uint32_t gx, uint32_t gy, uint32_t gz) {
  return make_uint3(block % gx, (block / gx) % gy, block / (gx * gy));
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_lowrank_partial(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_lowrank_partial_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.gx, p.gy, p.gz), make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_lowrank_partial_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_lowrank_reduce(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_lowrank_reduce_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.gx, p.gy, p.gz), make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_lowrank_reduce_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_lowrank_finish(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_lowrank_finish_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.gx, p.gy, p.gz), make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_lowrank_finish_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_residual_add(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_residual_add_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.gx, p.gy, p.gz), make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_moe_residual_add_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_swiglu(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_swiglu_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.gx, p.gy, p.gz), make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_glm53_swiglu_plain_task(p.task, TF_VBID.x, TF_VNBLK.x, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_moe_slice_sum(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_moe_slice_sum_params*>(raw);
  tf_vblock_set(tf_glm53_mk_moe_bid(block, p.h.gx, p.h.gy, p.h.gz), make_uint3(p.h.gx, p.h.gy, p.h.gz), make_uint3(p.h.threads, 1, 1));
  if (threadIdx.x >= p.h.threads) return;
  tf_glm_slice_sum_task(p.part, p.out, p.total, p.ks, p.f32);
}
