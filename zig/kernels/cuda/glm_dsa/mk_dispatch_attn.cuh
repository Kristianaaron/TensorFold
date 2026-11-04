// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 attention adapters. Each payload mirrors one task's plain C ABI.
// Math is adapted from bertholomus/bertholomus/TensorFold@ddfad35 (Apache-2.0), mla.cu.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include "mla.cu"
#include "../tf_vblock.cuh"

// ROUND 4 (MK8 deferred launch): the input norm (tf_glm_rmsnorm_6144), q_pe pack and attend payloads are
// { tf_glm53_mk_hdr h; <__global__ parameters in signature order> }, proven below against signature mirrors.
struct tf_glm53_mk_sig_rmsnorm_6144 {
  const __nv_bfloat16* x; int64_t x_stride; const __nv_bfloat16* w; __nv_bfloat16* out;
  int64_t o_stride; float* xs; float eps; int sums;
};
struct tf_glm53_mk_attn_input_norm_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* x; int64_t x_stride; const __nv_bfloat16* w; __nv_bfloat16* out;
  int64_t o_stride; float* xs; float eps; int sums;
};
struct tf_glm53_mk_sig_qpe_pack {
  const __nv_bfloat16* q; const float* cos; const float* sin; __nv_bfloat16* qpe; int rows; int local_heads;
};
struct tf_glm53_mk_attn_qpe_pack_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* q; const float* cos; const float* sin; __nv_bfloat16* qpe; int rows; int local_heads;
};
struct tf_glm53_mk_sig_attend {
  const __nv_bfloat16* qa; const __nv_bfloat16* qpe; const __nv_bfloat16* latent; const __nv_bfloat16* rope;
  const __nv_bfloat16* stage_latent; const __nv_bfloat16* stage_rope; const int64_t* positions;
  const int32_t* global_tokens; const int32_t* local_slots; const int32_t* global_counts;
  const int32_t* local_counts; const int32_t* dense_len; __nv_bfloat16* partial;
  int rows; int local_heads; int rank; int world; int prefill;
};
struct tf_glm53_mk_attn_attend_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* qa; const __nv_bfloat16* qpe; const __nv_bfloat16* latent; const __nv_bfloat16* rope;
  const __nv_bfloat16* stage_latent; const __nv_bfloat16* stage_rope; const int64_t* positions;
  const int32_t* global_tokens; const int32_t* local_slots; const int32_t* global_counts;
  const int32_t* local_counts; const int32_t* dense_len; __nv_bfloat16* partial;
  int rows; int local_heads; int rank; int world; int prefill;
};

#define TF_MK_S tf_glm53_mk_sig_rmsnorm_6144
#define TF_MK_P tf_glm53_mk_attn_input_norm_params
TF_GLM53_MK_SIG(tf_glm_rmsnorm_6144, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, x), TF_GLM53_MK_T(TF_MK_S, x_stride),
                TF_GLM53_MK_T(TF_MK_S, w), TF_GLM53_MK_T(TF_MK_S, out), TF_GLM53_MK_T(TF_MK_S, o_stride),
                TF_GLM53_MK_T(TF_MK_S, xs), TF_GLM53_MK_T(TF_MK_S, eps), TF_GLM53_MK_T(TF_MK_S, sums));
TF_GLM53_MK_ABI_SIZE(TF_MK_P, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, x);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, x_stride);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, w);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, out);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, o_stride);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, xs);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, eps);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, sums);
#undef TF_MK_P
#undef TF_MK_S
#define TF_MK_S tf_glm53_mk_sig_qpe_pack
#define TF_MK_P tf_glm53_mk_attn_qpe_pack_params
TF_GLM53_MK_SIG(tf_glm53_qpe_pack, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, q), TF_GLM53_MK_T(TF_MK_S, cos),
                TF_GLM53_MK_T(TF_MK_S, sin), TF_GLM53_MK_T(TF_MK_S, qpe), TF_GLM53_MK_T(TF_MK_S, rows),
                TF_GLM53_MK_T(TF_MK_S, local_heads));
TF_GLM53_MK_ABI_SIZE(TF_MK_P, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, q);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, cos);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, sin);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, qpe);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, rows);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, local_heads);
#undef TF_MK_P
#undef TF_MK_S
#define TF_MK_S tf_glm53_mk_sig_attend
#define TF_MK_P tf_glm53_mk_attn_attend_params
TF_GLM53_MK_SIG(tf_glm53_attend, TF_MK_S, TF_GLM53_MK_T(TF_MK_S, qa), TF_GLM53_MK_T(TF_MK_S, qpe),
                TF_GLM53_MK_T(TF_MK_S, latent), TF_GLM53_MK_T(TF_MK_S, rope), TF_GLM53_MK_T(TF_MK_S, stage_latent),
                TF_GLM53_MK_T(TF_MK_S, stage_rope), TF_GLM53_MK_T(TF_MK_S, positions),
                TF_GLM53_MK_T(TF_MK_S, global_tokens), TF_GLM53_MK_T(TF_MK_S, local_slots),
                TF_GLM53_MK_T(TF_MK_S, global_counts), TF_GLM53_MK_T(TF_MK_S, local_counts),
                TF_GLM53_MK_T(TF_MK_S, dense_len), TF_GLM53_MK_T(TF_MK_S, partial), TF_GLM53_MK_T(TF_MK_S, rows),
                TF_GLM53_MK_T(TF_MK_S, local_heads), TF_GLM53_MK_T(TF_MK_S, rank), TF_GLM53_MK_T(TF_MK_S, world),
                TF_GLM53_MK_T(TF_MK_S, prefill));
TF_GLM53_MK_ABI_SIZE(TF_MK_P, TF_MK_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, qa);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, qpe);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, latent);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, rope);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, stage_latent);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, stage_rope);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, positions);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, global_tokens);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, local_slots);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, global_counts);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, local_counts);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, dense_len);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, partial);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, rows);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, local_heads);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, rank);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, world);
TF_GLM53_MK_ABI_FIELD(TF_MK_P, TF_MK_S, prefill);
#undef TF_MK_P
#undef TF_MK_S
struct tf_glm53_mk_attn_kv_write_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16 *proj, *norm; const float *cos, *sin; const int64_t* positions;
  __nv_bfloat16 *latent, *rope, *stage_latent, *stage_rope; int rows, stride, rank, world;
};
struct tf_glm53_mk_attn_absorb_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* q; const uint8_t* kvb_codes; const float* kvb_scales;
  __nv_bfloat16* qa; int rows, local_heads, head0;
};
struct tf_glm53_mk_attn_merge_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16 *partial_local, *partial_peer; __nv_bfloat16* out; float* lse_out;
  int rows, local_heads, rank;
};
struct tf_glm53_mk_attn_expand_v_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* ol; const uint8_t* kvb_codes; const float* kvb_scales;
  __nv_bfloat16* out; int rows, local_heads, head0;
};
struct tf_glm53_mk_attn_q_a_norm_params {
  tf_glm53_mk_hdr h;
  const __nv_bfloat16* x; int64_t xs; const __nv_bfloat16* w;
  __nv_bfloat16* out; int64_t os; float* sums; float eps; int do_sums;
};
struct tf_glm53_mk_sig_kv_write { const __nv_bfloat16 *proj, *norm; const float *cos, *sin; const int64_t* positions; __nv_bfloat16 *latent, *rope, *stage_latent, *stage_rope; int rows, stride, rank, world; };
struct tf_glm53_mk_sig_absorb { const __nv_bfloat16* q; const uint8_t* kvb_codes; const float* kvb_scales; __nv_bfloat16* qa; int rows, local_heads, head0; };
struct tf_glm53_mk_sig_mla_merge { const __nv_bfloat16 *partial_local, *partial_peer; __nv_bfloat16* out; float* lse_out; int rows, local_heads, rank; };
struct tf_glm53_mk_sig_expand_v { const __nv_bfloat16* ol; const uint8_t* kvb_codes; const float* kvb_scales; __nv_bfloat16* out; int rows, local_heads, head0; };
#define TF_MK_KV_S tf_glm53_mk_sig_kv_write
#define TF_MK_KV_P tf_glm53_mk_attn_kv_write_params
TF_GLM53_MK_SIG(tf_glm53_kv_write, TF_MK_KV_S, TF_GLM53_MK_T(TF_MK_KV_S, proj), TF_GLM53_MK_T(TF_MK_KV_S, norm), TF_GLM53_MK_T(TF_MK_KV_S, cos), TF_GLM53_MK_T(TF_MK_KV_S, sin), TF_GLM53_MK_T(TF_MK_KV_S, positions), TF_GLM53_MK_T(TF_MK_KV_S, latent), TF_GLM53_MK_T(TF_MK_KV_S, rope), TF_GLM53_MK_T(TF_MK_KV_S, stage_latent), TF_GLM53_MK_T(TF_MK_KV_S, stage_rope), TF_GLM53_MK_T(TF_MK_KV_S, rows), TF_GLM53_MK_T(TF_MK_KV_S, stride), TF_GLM53_MK_T(TF_MK_KV_S, rank), TF_GLM53_MK_T(TF_MK_KV_S, world));
TF_GLM53_MK_ABI_SIZE(TF_MK_KV_P, TF_MK_KV_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, proj); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, norm);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, cos); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, sin);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, positions); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, latent);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, rope); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, stage_latent);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, stage_rope); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, rows);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, stride); TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, rank);
TF_GLM53_MK_ABI_FIELD(TF_MK_KV_P, TF_MK_KV_S, world);
#undef TF_MK_KV_S
#undef TF_MK_KV_P
#define TF_MK_ATTN_S tf_glm53_mk_sig_absorb
#define TF_MK_ATTN_P tf_glm53_mk_attn_absorb_params
TF_GLM53_MK_SIG(tf_glm53_absorb, TF_MK_ATTN_S, TF_GLM53_MK_T(TF_MK_ATTN_S, q), TF_GLM53_MK_T(TF_MK_ATTN_S, kvb_codes), TF_GLM53_MK_T(TF_MK_ATTN_S, kvb_scales), TF_GLM53_MK_T(TF_MK_ATTN_S, qa), TF_GLM53_MK_T(TF_MK_ATTN_S, rows), TF_GLM53_MK_T(TF_MK_ATTN_S, local_heads), TF_GLM53_MK_T(TF_MK_ATTN_S, head0));
TF_GLM53_MK_ABI_SIZE(TF_MK_ATTN_P, TF_MK_ATTN_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, q); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, kvb_codes); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, kvb_scales); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, qa); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, rows); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, local_heads); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, head0);
#undef TF_MK_ATTN_S
#undef TF_MK_ATTN_P
#define TF_MK_ATTN_S tf_glm53_mk_sig_mla_merge
#define TF_MK_ATTN_P tf_glm53_mk_attn_merge_params
TF_GLM53_MK_SIG(tf_glm53_mla_merge, TF_MK_ATTN_S, TF_GLM53_MK_T(TF_MK_ATTN_S, partial_local), TF_GLM53_MK_T(TF_MK_ATTN_S, partial_peer), TF_GLM53_MK_T(TF_MK_ATTN_S, out), TF_GLM53_MK_T(TF_MK_ATTN_S, lse_out), TF_GLM53_MK_T(TF_MK_ATTN_S, rows), TF_GLM53_MK_T(TF_MK_ATTN_S, local_heads), TF_GLM53_MK_T(TF_MK_ATTN_S, rank));
TF_GLM53_MK_ABI_SIZE(TF_MK_ATTN_P, TF_MK_ATTN_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, partial_local); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, partial_peer); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, out); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, lse_out); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, rows); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, local_heads); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, rank);
#undef TF_MK_ATTN_S
#undef TF_MK_ATTN_P
#define TF_MK_ATTN_S tf_glm53_mk_sig_expand_v
#define TF_MK_ATTN_P tf_glm53_mk_attn_expand_v_params
TF_GLM53_MK_SIG(tf_glm53_expand_v, TF_MK_ATTN_S, TF_GLM53_MK_T(TF_MK_ATTN_S, ol), TF_GLM53_MK_T(TF_MK_ATTN_S, kvb_codes), TF_GLM53_MK_T(TF_MK_ATTN_S, kvb_scales), TF_GLM53_MK_T(TF_MK_ATTN_S, out), TF_GLM53_MK_T(TF_MK_ATTN_S, rows), TF_GLM53_MK_T(TF_MK_ATTN_S, local_heads), TF_GLM53_MK_T(TF_MK_ATTN_S, head0));
TF_GLM53_MK_ABI_SIZE(TF_MK_ATTN_P, TF_MK_ATTN_S);
TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, ol); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, kvb_codes); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, kvb_scales); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, out); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, rows); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, local_heads); TF_GLM53_MK_ABI_FIELD(TF_MK_ATTN_P, TF_MK_ATTN_S, head0);
#undef TF_MK_ATTN_S
#undef TF_MK_ATTN_P
static_assert(sizeof(tf_glm53_mk_attn_input_norm_params) == 72, "attn norm payload ABI");
static_assert(offsetof(tf_glm53_mk_attn_input_norm_params, eps) == 64, "attn norm eps ABI");
static_assert(sizeof(tf_glm53_mk_attn_qpe_pack_params) == 56, "attn qpe payload ABI");
static_assert(offsetof(tf_glm53_mk_attn_qpe_pack_params, rows) == 48, "attn qpe rows ABI");
static_assert(sizeof(tf_glm53_mk_attn_attend_params) == 144, "attn attend payload ABI");
static_assert(offsetof(tf_glm53_mk_attn_attend_params, rows) == 120, "attn attend rows ABI");
static_assert(sizeof(tf_glm53_mk_attn_kv_write_params) == 104, "attn kv-write payload ABI");
static_assert(sizeof(tf_glm53_mk_attn_absorb_params) == 64, "attn absorb payload ABI");
static_assert(sizeof(tf_glm53_mk_attn_merge_params) == 64, "attn merge payload ABI");
static_assert(sizeof(tf_glm53_mk_attn_expand_v_params) == 64, "attn expand payload ABI");
static_assert(sizeof(tf_glm53_mk_attn_q_a_norm_params) == 72, "attn q-a norm payload ABI");

__device__ __forceinline__ void tf_glm53_mk_vblock(uint32_t block, uint32_t gx, uint32_t gy,
                                                    uint32_t gz, uint32_t threads) {
  tf_vblock_set(make_uint3(block % gx, (block / gx) % gy, block / (gx * gy)),
                make_uint3(gx, gy, gz), make_uint3(threads, 1, 1));
}
__device__ __forceinline__ void tf_glm53_mk_run_attn_input_norm(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_attn_input_norm_params*>(raw);
  tf_glm53_mk_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  tf_glm_rmsnorm_6144_task(p.x, p.x_stride, p.w, p.out, p.o_stride, p.xs, p.eps, p.sums);
  (void)smem;
}
__device__ __forceinline__ void tf_glm53_mk_run_attn_qpe_pack(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_attn_qpe_pack_params*>(raw);
  tf_glm53_mk_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  tf_glm53_qpe_pack_task(p.q, p.cos, p.sin, p.qpe, p.rows, p.local_heads);
}
__device__ __forceinline__ void tf_glm53_mk_run_attn_attend(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_attn_attend_params*>(raw);
  tf_glm53_mk_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  tf_glm53_attend_task(p.qa, p.qpe, p.latent, p.rope, p.stage_latent, p.stage_rope,
    p.positions, p.global_tokens, p.local_slots, p.global_counts, p.local_counts, p.dense_len,
    p.partial, p.rows, p.local_heads, p.rank, p.world, p.prefill);
}
#define TF_MK_ATTN_RUN(name, type, call) \
  __device__ __forceinline__ void tf_glm53_mk_run_attn_##name(const void* raw, uint32_t block, char*) { \
    const auto& p = *static_cast<const type*>(raw); \
    tf_glm53_mk_vblock(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads); \
    if (threadIdx.x >= p.h.threads) return; call; }
TF_MK_ATTN_RUN(kv_write, tf_glm53_mk_attn_kv_write_params,
  tf_glm53_kv_write_task(p.proj,p.norm,p.cos,p.sin,p.positions,p.latent,p.rope,p.stage_latent,p.stage_rope,p.rows,p.stride,p.rank,p.world))
TF_MK_ATTN_RUN(absorb, tf_glm53_mk_attn_absorb_params,
  tf_glm53_absorb_task(p.q,p.kvb_codes,p.kvb_scales,p.qa,p.rows,p.local_heads,p.head0))
TF_MK_ATTN_RUN(merge, tf_glm53_mk_attn_merge_params,
  tf_glm53_mla_merge_task(p.partial_local,p.partial_peer,p.out,p.lse_out,p.rows,p.local_heads,p.rank))
TF_MK_ATTN_RUN(expand_v, tf_glm53_mk_attn_expand_v_params,
  tf_glm53_expand_v_task(p.ol,p.kvb_codes,p.kvb_scales,p.out,p.rows,p.local_heads,p.head0))
TF_MK_ATTN_RUN(q_a_norm, tf_glm53_mk_attn_q_a_norm_params,
  tf_glm_rmsnorm_2048_task(p.x,p.xs,p.w,p.out,p.os,p.sums,p.eps,p.do_sums))
#undef TF_MK_ATTN_RUN
