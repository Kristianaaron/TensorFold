// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Fixed-layout payloads for the megakernel's nine original utility operations.
// Task math is reused unchanged from tasks.cuh, adapted from bertholomus/TensorFold@ddfad35 (Apache-2.0).
#pragma once
#include "tasks.cuh"
// The __global__ wrappers themselves, so the ROUND 4 proofs below check decltype(&kernel) against the real code.
#include "glue.cu"

// ROUND 4 (MK8 deferred launch): swiglu, low-rank partial/reduce and the MoE residual add are
// { tf_glm53_mk_hdr h; <__global__ parameters in signature order> }, proven below against signature mirrors.
struct tf_glm53_mk_sig_swiglu_plain { const __nv_bfloat16* gu; __nv_bfloat16* out; int rows; int width; };
struct tf_glm53_mk_swiglu_params { tf_glm53_mk_hdr h; const __nv_bfloat16* gu; __nv_bfloat16* out; int rows; int width; };
struct tf_glm53_mk_sig_lowrank_partial {
  const __nv_bfloat16* x; const float* a; float* partials;
  int rows; int hidden; int rank; int x_stride; int a_stride; int splits;
};
struct tf_glm53_mk_lowrank_partial_params {
  tf_glm53_mk_hdr h; const __nv_bfloat16* x; const float* a; float* partials;
  int rows; int hidden; int rank; int x_stride; int a_stride; int splits;
};
struct tf_glm53_mk_sig_lowrank_reduce { const float* partials; float* mid; int total; int splits; };
struct tf_glm53_mk_lowrank_reduce_params { tf_glm53_mk_hdr h; const float* partials; float* mid; int total; int splits; };
struct tf_glm53_mk_sig_moe_residual_add { const __nv_bfloat16* x; __nv_bfloat16* xout; const __nv_bfloat16* parts; int count; };
struct tf_glm53_mk_residual_add_params { tf_glm53_mk_hdr h; const __nv_bfloat16* x; __nv_bfloat16* xout; const __nv_bfloat16* parts; int count; };

#define TF_MK_S tf_glm53_mk_sig_swiglu_plain
TF_GLM53_MK_SIG(tf_glm53_swiglu_plain, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, gu), TF_GLM53_MK_T(TF_MK_S, out),
                TF_GLM53_MK_T(TF_MK_S, rows), TF_GLM53_MK_T(TF_MK_S, width));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_swiglu_params, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_swiglu_params, TF_MK_S, gu);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_swiglu_params, TF_MK_S, out);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_swiglu_params, TF_MK_S, rows);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_swiglu_params, TF_MK_S, width);
#undef TF_MK_S
#define TF_MK_S tf_glm53_mk_sig_lowrank_partial
TF_GLM53_MK_SIG(tf_glm53_lowrank_partial, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, x), TF_GLM53_MK_T(TF_MK_S, a),
                TF_GLM53_MK_T(TF_MK_S, partials), TF_GLM53_MK_T(TF_MK_S, rows), TF_GLM53_MK_T(TF_MK_S, hidden),
                TF_GLM53_MK_T(TF_MK_S, rank), TF_GLM53_MK_T(TF_MK_S, x_stride), TF_GLM53_MK_T(TF_MK_S, a_stride),
                TF_GLM53_MK_T(TF_MK_S, splits));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_lowrank_partial_params, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, x);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, a);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, partials);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, rows);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, hidden);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, rank);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, x_stride);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, a_stride);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_partial_params, TF_MK_S, splits);
#undef TF_MK_S
#define TF_MK_S tf_glm53_mk_sig_lowrank_reduce
TF_GLM53_MK_SIG(tf_glm53_lowrank_reduce, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, partials), TF_GLM53_MK_T(TF_MK_S, mid),
                TF_GLM53_MK_T(TF_MK_S, total), TF_GLM53_MK_T(TF_MK_S, splits));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_lowrank_reduce_params, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_reduce_params, TF_MK_S, partials);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_reduce_params, TF_MK_S, mid);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_reduce_params, TF_MK_S, total);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_reduce_params, TF_MK_S, splits);
#undef TF_MK_S
#define TF_MK_S tf_glm53_mk_sig_moe_residual_add
TF_GLM53_MK_SIG(tf_glm53_moe_residual_add, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, x), TF_GLM53_MK_T(TF_MK_S, xout),
                TF_GLM53_MK_T(TF_MK_S, parts), TF_GLM53_MK_T(TF_MK_S, count));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_residual_add_params, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_add_params, TF_MK_S, x);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_add_params, TF_MK_S, xout);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_add_params, TF_MK_S, parts);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_add_params, TF_MK_S, count);
#undef TF_MK_S

#define TF_MK_UTIL_HEAD uint32_t gx, gy, gz, threads
struct tf_glm53_mk_add_partial_params { TF_MK_UTIL_HEAD; tf_glm53_add_partial_params task; };
struct tf_glm53_mk_lowrank_combine_bf16_params { tf_glm53_mk_hdr h; const float *routed, *shared, *b, *mid; __nv_bfloat16* out; int rows, hidden, rank, b_stride; };
struct tf_glm53_mk_residual_rmsnorm_6144_params { tf_glm53_mk_hdr h; __nv_bfloat16* x; const float* parts; const __nv_bfloat16* w; __nv_bfloat16* normed; int rows; float eps; };
struct tf_glm53_mk_lowrank_finish_params { tf_glm53_mk_hdr h; const float *b, *mid; float* acc; int rows, hidden, rank, b_stride; };
struct tf_glm53_mk_embed_pick_params { tf_glm53_mk_hdr h; const __nv_bfloat16* gathered; const int* owners; __nv_bfloat16* out; int rows, hidden, world; };
#undef TF_MK_UTIL_HEAD

static_assert(sizeof(tf_glm53_mk_swiglu_params) == 40, "MK swiglu payload ABI");
static_assert(sizeof(tf_glm53_mk_lowrank_partial_params) == 64, "MK lowrank partial payload ABI");
static_assert(sizeof(tf_glm53_mk_lowrank_reduce_params) == 40, "MK lowrank reduce payload ABI");
static_assert(sizeof(tf_glm53_mk_residual_add_params) == 48, "MK residual add payload ABI");
static_assert(sizeof(tf_glm53_mk_add_partial_params) == 40, "MK add payload ABI");
static_assert(sizeof(tf_glm53_mk_lowrank_combine_bf16_params) == 72, "MK combine payload ABI");
static_assert(sizeof(tf_glm53_mk_residual_rmsnorm_6144_params) == 56, "MK residual norm payload ABI");
static_assert(sizeof(tf_glm53_mk_lowrank_finish_params) == 56, "MK lowrank finish payload ABI");
static_assert(sizeof(tf_glm53_mk_embed_pick_params) == 56, "MK embed payload ABI");
static_assert(offsetof(tf_glm53_mk_add_partial_params, task) == 16, "MK add task offset");
struct tf_glm53_mk_sig_lowrank_combine_bf16 { const float *routed, *shared, *b, *mid; __nv_bfloat16* out; int rows, hidden, rank, b_stride; };
struct tf_glm53_mk_sig_residual_rmsnorm_6144 { __nv_bfloat16* x; const float* parts; const __nv_bfloat16* w; __nv_bfloat16* normed; int rows; float eps; };
struct tf_glm53_mk_sig_lowrank_finish { const float *b, *mid; float* acc; int rows, hidden, rank, b_stride; };
struct tf_glm53_mk_sig_embed_pick { const __nv_bfloat16* gathered; const int* owners; __nv_bfloat16* out; int rows, hidden, world; };
#define TF_MK_UTIL_SIG(k, m, ...) TF_GLM53_MK_SIG(k, m, __VA_ARGS__)
#define TF_MK_UTIL_ABI(p, m, ...) static_assert(sizeof(p) == sizeof(tf_glm53_mk_hdr) + sizeof(m), "payload size"); __VA_ARGS__
TF_MK_UTIL_SIG(tf_glm53_lowrank_combine_bf16, tf_glm53_mk_sig_lowrank_combine_bf16,
  TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, routed), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, shared), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, b), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, mid), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, out), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, hidden), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, rank), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_combine_bf16, b_stride));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, routed); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, shared); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, b); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, mid); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, out); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, rows); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, hidden); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, rank); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_mk_sig_lowrank_combine_bf16, b_stride);
TF_GLM53_MK_SIG(tf_glm53_residual_rmsnorm_6144, tf_glm53_mk_sig_residual_rmsnorm_6144, TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, x), TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, parts), TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, w), TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, normed), TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_residual_rmsnorm_6144, eps));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, x); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, parts); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, w); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, normed); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, rows); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_mk_sig_residual_rmsnorm_6144, eps);
TF_GLM53_MK_SIG(tf_glm53_lowrank_finish, tf_glm53_mk_sig_lowrank_finish, TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, b), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, mid), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, acc), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, hidden), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, rank), TF_GLM53_MK_T(tf_glm53_mk_sig_lowrank_finish, b_stride));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, b); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, mid); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, acc); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, rows); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, hidden); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, rank); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_lowrank_finish_params, tf_glm53_mk_sig_lowrank_finish, b_stride);
TF_GLM53_MK_SIG(tf_glm53_embed_pick, tf_glm53_mk_sig_embed_pick, TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, gathered), TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, owners), TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, out), TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, rows), TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, hidden), TF_GLM53_MK_T(tf_glm53_mk_sig_embed_pick, world));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, gathered); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, owners); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, out); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, rows); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, hidden); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_embed_pick_params, tf_glm53_mk_sig_embed_pick, world);
#undef TF_MK_UTIL_SIG
#undef TF_MK_UTIL_ABI

__device__ __forceinline__ void tf_glm53_mk_util_vblock(uint32_t block, uint32_t gx, uint32_t gy,
                                                         uint32_t gz, uint32_t threads) {
  tf_vblock_set(make_uint3(block % gx, (block / gx) % gy, block / (gx * gy)),
                make_uint3(gx, gy, gz), make_uint3(threads, 1, 1));
}
#define TF_MK_UTIL_RUN(name, type, task_fn) \
  __device__ __forceinline__ void tf_glm53_mk_run_##name(const void* raw, uint32_t block, char* smem) { \
    const auto& p = *static_cast<const type*>(raw); \
    tf_glm53_mk_util_vblock(block, p.gx, p.gy, p.gz, p.threads); \
    if (threadIdx.x >= p.threads) return; \
    task_fn(p.task, TF_VBID.x, TF_VNBLK.x, smem); }
TF_MK_UTIL_RUN(add_partial, tf_glm53_mk_add_partial_params, tf_glm53_add_partial_task)
#undef TF_MK_UTIL_RUN

#define TF_MK_UTIL_SIG_RUN(name, type, task_type, task_fn, ...) \
  __device__ __forceinline__ void tf_glm53_mk_run_##name(const void* raw, uint32_t block, char* smem) { \
    const auto& p = *static_cast<const type*>(raw); \
    tf_glm53_mk_util_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads); \
    if (threadIdx.x >= p.h.threads) return; \
    const task_type t{__VA_ARGS__}; \
    task_fn(t, TF_VBID.x, TF_VNBLK.x, smem); }
TF_MK_UTIL_SIG_RUN(lowrank_combine_bf16, tf_glm53_mk_lowrank_combine_bf16_params, tf_glm53_lowrank_combine_bf16_params, tf_glm53_lowrank_combine_bf16_task, p.routed, p.shared, p.b, p.mid, p.out, p.rows, p.hidden, p.rank, p.b_stride)
TF_MK_UTIL_SIG_RUN(residual_rmsnorm_6144, tf_glm53_mk_residual_rmsnorm_6144_params, tf_glm53_residual_rmsnorm_6144_params, tf_glm53_residual_rmsnorm_6144_task, p.x, p.parts, p.w, p.normed, p.rows, p.eps)
TF_MK_UTIL_SIG_RUN(lowrank_finish, tf_glm53_mk_lowrank_finish_params, tf_glm53_lowrank_finish_params, tf_glm53_lowrank_finish_task, p.b, p.mid, p.acc, p.rows, p.hidden, p.rank, p.b_stride)
TF_MK_UTIL_SIG_RUN(embed_pick, tf_glm53_mk_embed_pick_params, tf_glm53_embed_pick_params, tf_glm53_embed_pick_task, p.gathered, p.owners, p.out, p.rows, p.hidden, p.world)
#undef TF_MK_UTIL_SIG_RUN

// ROUND 4 entries: the task parameters are rebuilt exactly as each __global__ wrapper in glue.cu builds them.
#define TF_MK_SIG_RUN(name, type, task_type, task_fn, ...) \
  __device__ __forceinline__ void tf_glm53_mk_run_##name(const void* raw, uint32_t block, char* smem) { \
    const auto& p = *static_cast<const type*>(raw); \
    tf_glm53_mk_util_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads); \
    if (threadIdx.x >= p.h.threads) return; \
    const task_type t{__VA_ARGS__}; \
    task_fn(t, TF_VBID.x, TF_VNBLK.x, smem); }
TF_MK_SIG_RUN(swiglu, tf_glm53_mk_swiglu_params, tf_glm53_swiglu_plain_params, tf_glm53_swiglu_plain_task,
              p.gu, p.out, p.rows, p.width)
TF_MK_SIG_RUN(lowrank_partial, tf_glm53_mk_lowrank_partial_params, tf_glm53_lowrank_partial_params,
              tf_glm53_lowrank_partial_task, p.x, p.a, p.partials, p.rows, p.hidden, p.rank, p.x_stride, p.a_stride,
              p.splits)
TF_MK_SIG_RUN(lowrank_reduce, tf_glm53_mk_lowrank_reduce_params, tf_glm53_lowrank_reduce_params,
              tf_glm53_lowrank_reduce_task, p.partials, p.mid, p.total, p.splits)
TF_MK_SIG_RUN(moe_residual_add_utility, tf_glm53_mk_residual_add_params, tf_glm53_moe_residual_add_params,
              tf_glm53_moe_residual_add_task, p.x, p.xout, p.parts, p.count)
#undef TF_MK_SIG_RUN
