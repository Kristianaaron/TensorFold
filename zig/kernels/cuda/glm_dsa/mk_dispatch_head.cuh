// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 head adapters. Payloads are independent, fixed-layout C structs.
#pragma once
#include "head.cu"
#include "tasks.cuh"

#define TF_MK_HEAD tf_glm53_mk_hdr h
struct tf_glm53_mk_head_norm_params { TF_MK_HEAD; const __nv_bfloat16* x; int64_t xs; const __nv_bfloat16* w; __nv_bfloat16* out; int64_t os; float* sums; float eps; int do_sums; };
struct tf_glm53_mk_head_top1_params { TF_MK_HEAD; const float* x; uint32_t vocab, offset; float* values; uint32_t* ids; };
struct tf_glm53_mk_head_cand_fill_params { TF_MK_HEAD; float* values; uint32_t* ids; uint32_t count; };
struct tf_glm53_mk_head_cand_part_params { TF_MK_HEAD; const float* x; uint32_t vocab, offset, k; unsigned long long* scratch; };
struct tf_glm53_mk_head_cand_merge_params { TF_MK_HEAD; const unsigned long long* scratch; uint32_t vocab, k; float* values; uint32_t* ids; };
struct tf_glm53_mk_head_lse_params { TF_MK_HEAD; const float* x; uint32_t vocab; float* partial; };
struct tf_glm53_mk_head_gate_gather_params { TF_MK_HEAD; const float* logits; const uint32_t* ids; uint32_t vocab, offset; float* values; uint32_t probes, rows; };
struct tf_glm53_mk_head_gate_mass_params { TF_MK_HEAD; const float *probe_logits, *lse; float *logp, *mass_outside; uint32_t probes, rows; };
struct tf_glm53_mk_head_embed_pick_params { TF_MK_HEAD; tf_glm53_embed_pick_params task; };
#undef TF_MK_HEAD
static_assert(offsetof(tf_glm53_mk_head_norm_params, x) == 16, "head norm payload ABI");
static_assert(offsetof(tf_glm53_mk_head_top1_params, x) == 16, "head top1 payload ABI");
static_assert(offsetof(tf_glm53_mk_head_cand_part_params, x) == 16, "head candidate payload ABI");
static_assert(sizeof(tf_glm53_mk_head_norm_params) == 72, "head norm size ABI");
static_assert(sizeof(tf_glm53_mk_head_top1_params) == 48, "head top1 size ABI");
static_assert(sizeof(tf_glm53_mk_head_cand_fill_params) == 40, "head fill size ABI");
static_assert(sizeof(tf_glm53_mk_head_cand_part_params) == 48, "head part size ABI");
static_assert(sizeof(tf_glm53_mk_head_cand_merge_params) == 48, "head merge size ABI");
static_assert(sizeof(tf_glm53_mk_head_lse_params) == 40, "head lse size ABI");
static_assert(sizeof(tf_glm53_mk_head_gate_gather_params) == 56, "head gather size ABI");
static_assert(sizeof(tf_glm53_mk_head_gate_mass_params) == 56, "head mass size ABI");
static_assert(sizeof(tf_glm53_mk_head_embed_pick_params) == 56, "head embed size ABI");
static_assert(offsetof(tf_glm53_mk_head_cand_fill_params, values) == 16, "head fill values ABI");
static_assert(offsetof(tf_glm53_mk_head_cand_merge_params, scratch) == 16, "head merge scratch ABI");
static_assert(offsetof(tf_glm53_mk_head_lse_params, x) == 16, "head lse input ABI");
static_assert(offsetof(tf_glm53_mk_head_gate_gather_params, logits) == 16, "head gather logits ABI");
static_assert(offsetof(tf_glm53_mk_head_gate_mass_params, probe_logits) == 16, "head mass input ABI");
static_assert(offsetof(tf_glm53_mk_head_embed_pick_params, task) == 16, "head embed task ABI");
struct tf_glm53_mk_sig_head_top1 { const float* x; unsigned vocab, offset; float* values; unsigned* ids; };
struct tf_glm53_mk_sig_head_fill { float* values; unsigned* ids; unsigned count; };
struct tf_glm53_mk_sig_head_part { const float* x; unsigned vocab, offset, k; unsigned long long* scratch; };
struct tf_glm53_mk_sig_head_merge { const unsigned long long* scratch; unsigned vocab, k; float* values; unsigned* ids; };
struct tf_glm53_mk_sig_head_lse { const float* x; unsigned vocab; float* partial; };
struct tf_glm53_mk_sig_head_gather { const float* logits; const unsigned* ids; unsigned vocab, offset; float* values; unsigned probes, rows; };
struct tf_glm53_mk_sig_head_mass { const float *probe_logits, *lse; float *logp, *mass_outside; unsigned probes, rows; };
TF_GLM53_MK_SIG(glm53_head::tf_glm53_top1_f32, tf_glm53_mk_sig_head_top1, TF_GLM53_MK_T(tf_glm53_mk_sig_head_top1, x), TF_GLM53_MK_T(tf_glm53_mk_sig_head_top1, vocab), TF_GLM53_MK_T(tf_glm53_mk_sig_head_top1, offset), TF_GLM53_MK_T(tf_glm53_mk_sig_head_top1, values), TF_GLM53_MK_T(tf_glm53_mk_sig_head_top1, ids));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1, x); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1, vocab); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1, offset); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1, values); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_top1_params, tf_glm53_mk_sig_head_top1, ids);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_cand_fill_f32, tf_glm53_mk_sig_head_fill, TF_GLM53_MK_T(tf_glm53_mk_sig_head_fill, values), TF_GLM53_MK_T(tf_glm53_mk_sig_head_fill, ids), TF_GLM53_MK_T(tf_glm53_mk_sig_head_fill, count));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_cand_fill_params, tf_glm53_mk_sig_head_fill);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_fill_params, tf_glm53_mk_sig_head_fill, values); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_fill_params, tf_glm53_mk_sig_head_fill, ids); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_fill_params, tf_glm53_mk_sig_head_fill, count);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_cand_part_f32, tf_glm53_mk_sig_head_part, TF_GLM53_MK_T(tf_glm53_mk_sig_head_part, x), TF_GLM53_MK_T(tf_glm53_mk_sig_head_part, vocab), TF_GLM53_MK_T(tf_glm53_mk_sig_head_part, offset), TF_GLM53_MK_T(tf_glm53_mk_sig_head_part, k), TF_GLM53_MK_T(tf_glm53_mk_sig_head_part, scratch));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part, x); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part, vocab); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part, offset); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part, k); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_part_params, tf_glm53_mk_sig_head_part, scratch);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_cand_merge_f32, tf_glm53_mk_sig_head_merge, TF_GLM53_MK_T(tf_glm53_mk_sig_head_merge, scratch), TF_GLM53_MK_T(tf_glm53_mk_sig_head_merge, vocab), TF_GLM53_MK_T(tf_glm53_mk_sig_head_merge, k), TF_GLM53_MK_T(tf_glm53_mk_sig_head_merge, values), TF_GLM53_MK_T(tf_glm53_mk_sig_head_merge, ids));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge, scratch); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge, vocab); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge, k); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge, values); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_cand_merge_params, tf_glm53_mk_sig_head_merge, ids);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_lse_f32, tf_glm53_mk_sig_head_lse, TF_GLM53_MK_T(tf_glm53_mk_sig_head_lse, x), TF_GLM53_MK_T(tf_glm53_mk_sig_head_lse, vocab), TF_GLM53_MK_T(tf_glm53_mk_sig_head_lse, partial));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_lse_params, tf_glm53_mk_sig_head_lse);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_lse_params, tf_glm53_mk_sig_head_lse, x); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_lse_params, tf_glm53_mk_sig_head_lse, vocab); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_lse_params, tf_glm53_mk_sig_head_lse, partial);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_gate_gather_f32, tf_glm53_mk_sig_head_gather, TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, logits), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, ids), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, vocab), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, offset), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, values), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, probes), TF_GLM53_MK_T(tf_glm53_mk_sig_head_gather, rows));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, logits); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, ids); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, vocab); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, offset); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, values); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, probes); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_gather_params, tf_glm53_mk_sig_head_gather, rows);
TF_GLM53_MK_SIG(glm53_head::tf_glm53_gate_mass_f32, tf_glm53_mk_sig_head_mass, TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, probe_logits), TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, lse), TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, logp), TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, mass_outside), TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, probes), TF_GLM53_MK_T(tf_glm53_mk_sig_head_mass, rows));
TF_GLM53_MK_ABI_SIZE(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass);
TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, probe_logits); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, lse); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, logp); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, mass_outside); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, probes); TF_GLM53_MK_ABI_FIELD(tf_glm53_mk_head_gate_mass_params, tf_glm53_mk_sig_head_mass, rows);

__device__ __forceinline__ void tf_glm53_mk_head_coords(uint32_t block, uint32_t gx, uint32_t gy,
                                                         uint32_t gz, uint32_t threads) {
  tf_vblock_set(make_uint3(block % gx, (block / gx) % gy, block / (gx * gy)),
                make_uint3(gx, gy, gz), make_uint3(threads, 1, 1));
}
#define TF_MK_HEAD_RUN(name, type, call) \
  __device__ __forceinline__ void tf_glm53_mk_run_##name(const void* raw, uint32_t block, char* smem) { \
    const auto& p = *static_cast<const type*>(raw); \
    tf_glm53_mk_head_coords(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads); \
    if (threadIdx.x >= p.h.threads) return; call; }
TF_MK_HEAD_RUN(head_norm, tf_glm53_mk_head_norm_params,
  tf_glm_rmsnorm_6144_task(p.x, p.xs, p.w, p.out, p.os, p.sums, p.eps, p.do_sums))
TF_MK_HEAD_RUN(head_top1, tf_glm53_mk_head_top1_params,
  glm53_head::tf_glm53_top1_f32_task(p.x, p.vocab, p.offset, p.values, p.ids))
TF_MK_HEAD_RUN(head_cand_fill, tf_glm53_mk_head_cand_fill_params,
  glm53_head::tf_glm53_cand_fill_f32_task(p.values, p.ids, p.count))
__device__ __forceinline__ void tf_glm53_mk_run_head_cand_part(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_head_cand_part_params*>(raw);
  tf_glm53_mk_head_coords(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  glm53_head::tf_glm53_cand_part_f32_task(p.x, p.vocab, p.offset, p.k, p.scratch, smem);
}
__device__ __forceinline__ void tf_glm53_mk_run_head_cand_merge(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_head_cand_merge_params*>(raw);
  tf_glm53_mk_head_coords(block, p.h.gx, p.h.gy, p.h.gz, p.h.threads);
  if (threadIdx.x >= p.h.threads) return;
  glm53_head::tf_glm53_cand_merge_f32_task(p.scratch, p.vocab, p.k, p.values, p.ids, smem);
}
TF_MK_HEAD_RUN(head_lse, tf_glm53_mk_head_lse_params,
  glm53_head::tf_glm53_lse_f32_task(p.x, p.vocab, p.partial))
TF_MK_HEAD_RUN(head_gate_gather, tf_glm53_mk_head_gate_gather_params,
  glm53_head::tf_glm53_gate_gather_f32_task(p.logits, p.ids, p.vocab, p.offset, p.values, p.probes, p.rows))
TF_MK_HEAD_RUN(head_gate_mass, tf_glm53_mk_head_gate_mass_params,
  glm53_head::tf_glm53_gate_mass_f32_task(p.probe_logits, p.lse, p.logp, p.mass_outside, p.probes, p.rows))
TF_MK_HEAD_RUN(head_embed_pick, tf_glm53_mk_head_embed_pick_params,
  tf_glm53_embed_pick_task(p.task, block, p.h.gx * p.h.gy * p.h.gz, nullptr))
#undef TF_MK_HEAD_RUN
