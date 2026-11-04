// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// TensorFold GLM-5.3 megakernel runtime. Scheduler code is original to TensorFold.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

// Byte-for-byte layout of graph.zig's Task (size 16, alignment 4).
struct tf_glm53_mk_task {
  uint32_t successor_offset;
  uint16_t layer;
  uint16_t aux;
  uint16_t dependency_count;
  uint16_t successor_count;
  uint8_t kind;
  uint8_t row;
  uint16_t reserved;
};
static_assert(sizeof(tf_glm53_mk_task) == 16, "Task ABI size");
static_assert(offsetof(tf_glm53_mk_task, layer) == 4, "Task ABI layer");
static_assert(offsetof(tf_glm53_mk_task, aux) == 6, "Task ABI aux");
static_assert(offsetof(tf_glm53_mk_task, successor_offset) == 0, "Task ABI successors");
static_assert(offsetof(tf_glm53_mk_task, successor_count) == 10, "Task ABI successor count");
static_assert(offsetof(tf_glm53_mk_task, kind) == 12, "Task ABI kind");
static_assert(offsetof(tf_glm53_mk_task, row) == 13, "Task ABI row");
static_assert(offsetof(tf_glm53_mk_task, reserved) == 14, "Task ABI dispatch extension");

enum tf_glm53_mk_error : uint32_t {
  TF_GLM53_MK_OK = 0,
  TF_GLM53_MK_TIMEOUT = 1,
  TF_GLM53_MK_BAD_TASK = 2,
  TF_GLM53_MK_UNSUPPORTED = 3,
};

enum tf_glm53_mk_op : uint8_t {
  TF_GLM53_MK_SWIGLU = 0,
  TF_GLM53_MK_LOWRANK_PARTIAL = 1,
  TF_GLM53_MK_LOWRANK_REDUCE = 2,
  TF_GLM53_MK_ADD_PARTIAL = 3,
  TF_GLM53_MK_LOWRANK_COMBINE_BF16 = 4,
  TF_GLM53_MK_RESIDUAL_RMSNORM_6144 = 5,
  TF_GLM53_MK_MOE_RESIDUAL_ADD_UTILITY = 6,
  TF_GLM53_MK_LOWRANK_FINISH = 7,
  TF_GLM53_MK_EMBED_PICK = 8,
  TF_GLM53_MK_ATTN_BASE = 9,
  TF_GLM53_MK_ATTN_INPUT_NORM = 9,
  TF_GLM53_MK_ATTN_QPE_PACK = 10,
  TF_GLM53_MK_ATTN_ATTEND = 11,
  TF_GLM53_MK_MOE_BASE = 12,
  TF_GLM53_MK_MOE_POST_NORM = 12,
  TF_GLM53_MK_MOE_ROUTER_PART = 13,
  TF_GLM53_MK_MOE_ROUTER_TOPK = 14,
  TF_GLM53_MK_MOE_LOWRANK_PARTIAL = 15,
  TF_GLM53_MK_MOE_LOWRANK_REDUCE = 16,
  TF_GLM53_MK_MOE_LOWRANK_FINISH = 17,
  TF_GLM53_MK_MOE_RESIDUAL_ADD = 18,
  TF_GLM53_MK_MOE_SWIGLU = 19,
  TF_GLM53_MK_HEAD_BASE = 20,
  TF_GLM53_MK_HEAD_FINAL_NORM = 20,
  TF_GLM53_MK_HEAD_TOP1_F32 = 21,
  TF_GLM53_MK_HEAD_CANDIDATE_FILL_F32 = 22,
  TF_GLM53_MK_HEAD_CANDIDATE_PART_F32 = 23,
  TF_GLM53_MK_HEAD_CANDIDATE_MERGE_F32 = 24,
  TF_GLM53_MK_HEAD_LSE_F32 = 25,
  TF_GLM53_MK_HEAD_GATE_GATHER_F32 = 26,
  TF_GLM53_MK_HEAD_GATE_MASS_F32 = 27,
  TF_GLM53_MK_HEAD_EMBED_PICK = 28,
  TF_GLM53_MK_INDEX_QUERY = 29,
  TF_GLM53_MK_INDEX_KEY = 30,
  TF_GLM53_MK_INDEX_SCORES = 31,
  TF_GLM53_MK_ATTN_KV_WRITE = 32,
  TF_GLM53_MK_ATTN_ABSORB = 33,
  TF_GLM53_MK_ATTN_MLA_MERGE = 34,
  TF_GLM53_MK_ATTN_EXPAND_V = 35,
  TF_GLM53_MK_MOE_ROUTER_SLICE_SUM = 36,
  TF_GLM53_MK_ATTN_Q_A_NORM = 37,
  TF_GLM53_MK_EXL3_ROT_IN = 38,
  TF_GLM53_MK_EXL3_DOWN_COMBINE = 39,
  TF_GLM53_MK_QMMF_PARTIAL = 40,
  TF_GLM53_MK_QMMF_REDUCE = 41,
};

struct tf_glm53_mk_state {
  const tf_glm53_mk_task* tasks;
  uint32_t* counters;
  const uint32_t* initial_counters;
  const uint32_t* successors;
  void* const* params;
  const uint32_t* blocks;
  uint32_t* next_block;
  uint32_t* finished_blocks;
  uint32_t* active;
  uint32_t* ready;
  uint32_t* ready_head;
  uint32_t* ready_tail;
  uint32_t* ready_lock;
  uint32_t* completed;
  uint32_t* error;
  uint32_t task_count;
  uint32_t edge_count;
  uint32_t ready_capacity;
  uint32_t timeout_cycles;
  // Test-only timing (null in production): per task {first start, last end, summed block ns, ready time} in
  // %globaltimer ns, then one record {first worker entry, last worker exit, 0, 0}. The host initialises it.
  unsigned long long* trace;
  // Chain scheduler (tf_glm53_mk_run_chain*) only: TF_GLM53_MK_SM_WORDS words the reset kernel zeroes, the CTAs per
  // SM it was sized for (`chain_slots`, with `chain_sms` SMs) and the worker mapping (TF_GLM53_MK_MAP_*).
  uint32_t* sm_words;
  uint32_t chain_sms;
  uint32_t chain_slots;
  uint32_t chain_map;
  uint32_t reserved;
};
static_assert(sizeof(tf_glm53_mk_state) == 168 && offsetof(tf_glm53_mk_state, trace) == 136 &&
              offsetof(tf_glm53_mk_state, sm_words) == 144 && offsetof(tf_glm53_mk_state, chain_map) == 160,
              "MK state ABI (launch.zig RuntimeState)");

// sm_words layout: [0, 1024) dense SM index + 1 per %smid (0: unseen, ~0: being assigned), [1024, 2048) CTAs that
// started on each dense SM, [2048] the next dense SM index, [2049] CTAs that have picked a worker index, [2050]
// nonzero when any CTA's SM slot fell outside the chain_sms x chain_slots grid.
#define TF_GLM53_MK_SM_WORDS 2051u
enum tf_glm53_mk_map : uint32_t {
  TF_GLM53_MK_MAP_BLOCK = 0,  // worker w = blockIdx.x
  TF_GLM53_MK_MAP_SM = 1,     // worker w = (CTA's slot on its SM) * chain_sms + dense SM index
};

__device__ __forceinline__ uint32_t tf_glm53_mk_load_acquire(const uint32_t* p) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}

__device__ __forceinline__ uint32_t tf_glm53_mk_load_acquire_sys(const uint32_t* p) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}

__device__ __forceinline__ void tf_glm53_mk_store_release(uint32_t* p, uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t tf_glm53_mk_fetch_add(uint32_t* p, uint32_t value) {
  uint32_t prior;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(prior) : "l"(p), "r"(value) : "memory");
  return prior;
}

__device__ __forceinline__ uint32_t tf_glm53_mk_fetch_sub(uint32_t* p, uint32_t value) {
  uint32_t prior;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(prior) : "l"(p), "r"(~(value - 1u)) : "memory");
  return prior;
}

__device__ __forceinline__ uint32_t tf_glm53_mk_claim_ready(uint32_t* head, const uint32_t* tail_ptr) {
  uint32_t observed = atomicAdd(head, 0u);
  for (;;) {
    const uint32_t tail = tf_glm53_mk_load_acquire(tail_ptr);
    if (observed >= tail) return UINT32_MAX;
    const uint32_t prior = atomicCAS(head, observed, observed + 1u);
    if (prior == observed) return observed;
    observed = prior;
  }
}

__device__ __forceinline__ unsigned long long tf_glm53_mk_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ bool tf_glm53_mk_enqueue(uint32_t id, uint32_t* ready, uint32_t* tail,
                                                     uint32_t* lock, uint32_t capacity, uint32_t count) {
  while (atomicCAS(lock, 0u, 1u) != 0u) __nanosleep(32);
  const uint32_t slot = *tail;
  if (slot > capacity || count > capacity - slot) { __threadfence(); atomicExch(lock, 0u); return false; }
  for (uint32_t i = 0; i < count; ++i) ready[slot + i] = id;
  __threadfence();
  tf_glm53_mk_store_release(tail, slot + count);
  atomicExch(lock, 0u);
  return true;
}

__device__ __forceinline__ void tf_glm53_mk_fail(uint32_t* error, uint32_t code) {
  atomicCAS(error, TF_GLM53_MK_OK, code);
}

__device__ __forceinline__ bool tf_glm53_mk_poll_flag(const uint32_t* flag, uint32_t expected,
                                                       uint32_t* error, uint32_t timeout_cycles) {
  uint32_t backoff = 32;
  for (uint32_t i = 0; i < timeout_cycles; ++i) {
    if (tf_glm53_mk_load_acquire_sys(flag) == expected) return true;
    __nanosleep(backoff);
    if (backoff < 2048) backoff <<= 1;
  }
  tf_glm53_mk_fail(error, TF_GLM53_MK_TIMEOUT);
  return false;
}

// ROUND 4 payload layout (MK8 deferred launch): a 16-byte header, then the kernel's parameters exactly as the CUDA
// parameter ABI lays out its __global__ signature (natural alignment, offsets from the end of the header). The
// launch sink writes the header from the launch grid/block and copies the packed parameter bytes behind it.
struct tf_glm53_mk_hdr {
  uint32_t gx, gy, gz, threads;
};
static_assert(sizeof(tf_glm53_mk_hdr) == 16 && alignof(tf_glm53_mk_hdr) == 4, "MK payload header ABI");

// Proof that payload P = { tf_glm53_mk_hdr h; <the parameters of mirror M> }: M is written from the __global__
// signature (TF_GLM53_MK_SIG ties its member types, in order, to decltype(&kernel)); every parameter sits at
// 16 + its offset in M, and P is exactly header + M.
#define TF_GLM53_MK_ABI_SIZE(P, M) \
  static_assert(offsetof(P, h) == 0 && sizeof(P) == sizeof(tf_glm53_mk_hdr) + sizeof(M), #P " is not header + " #M)
#define TF_GLM53_MK_ABI_FIELD(P, M, f) \
  static_assert(offsetof(P, f) == sizeof(tf_glm53_mk_hdr) + offsetof(M, f), #P "." #f " is not at 16 + its ABI offset")
#define TF_GLM53_MK_T(M, f) decltype(M::f)
#define TF_GLM53_MK_SIG(kernel, M, ...) \
  static_assert(std::is_same<decltype(&kernel), void (*)(__VA_ARGS__)>::value, #M " does not mirror " #kernel)

#include "../tf_vblock.cuh"
#include "tasks.cuh"
#include "../qmmf.cu"
#include "mk_dispatch_qmmf.cuh"

#include "mk_dispatch_attn.cuh"
#include "mk_dispatch_moe.cuh"
#include "mk_dispatch_head.cuh"
#include "mk_dispatch_index.cuh"
#include "mk_dispatch_exl3.cuh"
#include "mk_dispatch_util.cuh"

__device__ __forceinline__ bool tf_glm53_mk_dispatch(uint8_t kind, const void* raw,
                                                      uint32_t block, char* smem) {
  switch (kind) {
    case TF_GLM53_MK_SWIGLU:
      tf_glm53_mk_run_swiglu(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_PARTIAL:
      tf_glm53_mk_run_lowrank_partial(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_REDUCE:
      tf_glm53_mk_run_lowrank_reduce(raw, block, smem); return true;
    case TF_GLM53_MK_ADD_PARTIAL:
      tf_glm53_mk_run_add_partial(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_COMBINE_BF16:
      tf_glm53_mk_run_lowrank_combine_bf16(raw, block, smem); return true;
    case TF_GLM53_MK_RESIDUAL_RMSNORM_6144:
      tf_glm53_mk_run_residual_rmsnorm_6144(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_RESIDUAL_ADD_UTILITY:
      tf_glm53_mk_run_moe_residual_add_utility(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_FINISH:
      tf_glm53_mk_run_lowrank_finish(raw, block, smem); return true;
    case TF_GLM53_MK_EMBED_PICK:
      tf_glm53_mk_run_embed_pick(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_INPUT_NORM: tf_glm53_mk_run_attn_input_norm(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_QPE_PACK: tf_glm53_mk_run_attn_qpe_pack(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_ATTEND: tf_glm53_mk_run_attn_attend(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_KV_WRITE: tf_glm53_mk_run_attn_kv_write(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_ABSORB: tf_glm53_mk_run_attn_absorb(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_MLA_MERGE: tf_glm53_mk_run_attn_merge(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_EXPAND_V: tf_glm53_mk_run_attn_expand_v(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_Q_A_NORM: tf_glm53_mk_run_attn_q_a_norm(raw, block, smem); return true;
    case TF_GLM53_MK_EXL3_ROT_IN: tf_glm53_mk_run_exl3_rot_in(raw, block, smem); return true;
    case TF_GLM53_MK_EXL3_DOWN_COMBINE: tf_glm53_mk_run_exl3_down_combine(raw, block, smem); return true;
    case TF_GLM53_MK_QMMF_PARTIAL: tf_glm53_mk_run_qmmf_partial(raw, block, smem); return true;
    case TF_GLM53_MK_QMMF_REDUCE: tf_glm53_mk_run_qmmf_reduce(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_POST_NORM: tf_glm53_mk_run_moe_post_norm(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_PART: tf_glm53_mk_run_moe_router_part(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_TOPK: tf_glm53_mk_run_moe_router_topk(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_LOWRANK_PARTIAL: tf_glm53_mk_run_moe_lowrank_partial(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_LOWRANK_REDUCE: tf_glm53_mk_run_moe_lowrank_reduce(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_LOWRANK_FINISH: tf_glm53_mk_run_moe_lowrank_finish(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_SWIGLU: tf_glm53_mk_run_moe_swiglu(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_SLICE_SUM: tf_glm53_mk_run_moe_slice_sum(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_RESIDUAL_ADD: tf_glm53_mk_run_moe_residual_add(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_FINAL_NORM: tf_glm53_mk_run_head_norm(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_TOP1_F32: tf_glm53_mk_run_head_top1(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_FILL_F32: tf_glm53_mk_run_head_cand_fill(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_PART_F32: tf_glm53_mk_run_head_cand_part(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_MERGE_F32: tf_glm53_mk_run_head_cand_merge(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_LSE_F32: tf_glm53_mk_run_head_lse(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_GATE_GATHER_F32: tf_glm53_mk_run_head_gate_gather(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_GATE_MASS_F32: tf_glm53_mk_run_head_gate_mass(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_EMBED_PICK: tf_glm53_mk_run_head_embed_pick(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_QUERY: tf_glm53_mk_run_index_query(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_KEY: tf_glm53_mk_run_index_key(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_SCORES: tf_glm53_mk_run_index_scores(raw, block, smem); return true;
    default: return false;
  }
}

// The ops the MK8 deferred-launch table (defer.zig) can record: the same task bodies as tf_glm53_mk_dispatch, without
// the qmmf/EXL3/utility-only entries (the chain scheduler below runs only recorded launches).
__device__ __forceinline__ bool tf_glm53_mk_dispatch_deferred(uint8_t kind, const void* raw,
                                                               uint32_t block, char* smem) {
  switch (kind) {
    case TF_GLM53_MK_SWIGLU: tf_glm53_mk_run_swiglu(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_PARTIAL: tf_glm53_mk_run_lowrank_partial(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_REDUCE: tf_glm53_mk_run_lowrank_reduce(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_COMBINE_BF16: tf_glm53_mk_run_lowrank_combine_bf16(raw, block, smem); return true;
    case TF_GLM53_MK_RESIDUAL_RMSNORM_6144: tf_glm53_mk_run_residual_rmsnorm_6144(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_RESIDUAL_ADD_UTILITY: tf_glm53_mk_run_moe_residual_add_utility(raw, block, smem); return true;
    case TF_GLM53_MK_LOWRANK_FINISH: tf_glm53_mk_run_lowrank_finish(raw, block, smem); return true;
    case TF_GLM53_MK_EMBED_PICK: tf_glm53_mk_run_embed_pick(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_INPUT_NORM: tf_glm53_mk_run_attn_input_norm(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_QPE_PACK: tf_glm53_mk_run_attn_qpe_pack(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_ATTEND: tf_glm53_mk_run_attn_attend(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_KV_WRITE: tf_glm53_mk_run_attn_kv_write(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_ABSORB: tf_glm53_mk_run_attn_absorb(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_MLA_MERGE: tf_glm53_mk_run_attn_merge(raw, block, smem); return true;
    case TF_GLM53_MK_ATTN_EXPAND_V: tf_glm53_mk_run_attn_expand_v(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_PART: tf_glm53_mk_run_moe_router_part(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_TOPK: tf_glm53_mk_run_moe_router_topk(raw, block, smem); return true;
    case TF_GLM53_MK_MOE_ROUTER_SLICE_SUM: tf_glm53_mk_run_moe_slice_sum(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_TOP1_F32: tf_glm53_mk_run_head_top1(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_FILL_F32: tf_glm53_mk_run_head_cand_fill(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_PART_F32: tf_glm53_mk_run_head_cand_part(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_CANDIDATE_MERGE_F32: tf_glm53_mk_run_head_cand_merge(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_LSE_F32: tf_glm53_mk_run_head_lse(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_GATE_GATHER_F32: tf_glm53_mk_run_head_gate_gather(raw, block, smem); return true;
    case TF_GLM53_MK_HEAD_GATE_MASS_F32: tf_glm53_mk_run_head_gate_mass(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_QUERY: tf_glm53_mk_run_index_query(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_KEY: tf_glm53_mk_run_index_key(raw, block, smem); return true;
    case TF_GLM53_MK_INDEX_SCORES: tf_glm53_mk_run_index_scores(raw, block, smem); return true;
    default: return false;
  }
}

__device__ __forceinline__ void tf_glm53_mk_persistent(tf_glm53_mk_state s, char* task_smem,
                                                        uint32_t* shared_task_id,
                                                        uint32_t* shared_task_block) {
  const uint32_t lane = threadIdx.x;
  uint32_t spins = 0;
  unsigned long long* const trace = s.trace;
  if (trace && lane == 0) atomicMin(trace + 4ull * s.task_count, tf_glm53_mk_now());
  while (tf_glm53_mk_load_acquire(s.error) == TF_GLM53_MK_OK) {
    if (lane == 0) {
      *shared_task_id = UINT32_MAX;
      *shared_task_block = UINT32_MAX;
      if (tf_glm53_mk_load_acquire(s.completed) == s.task_count) *shared_task_id = s.task_count;
      else {
        const uint32_t head = tf_glm53_mk_claim_ready(s.ready_head, s.ready_tail);
        if (head != UINT32_MAX) {
          const uint32_t id = s.ready[head];
          if (id >= s.task_count) tf_glm53_mk_fail(s.error, TF_GLM53_MK_BAD_TASK);
          else {
            const uint32_t block = tf_glm53_mk_fetch_add(s.next_block + id, 1u);
            if (block >= s.blocks[id]) tf_glm53_mk_fail(s.error, TF_GLM53_MK_BAD_TASK);
            else { *shared_task_id = id; *shared_task_block = block; }
          }
        }
      }
    }
    __syncthreads();
    if (*shared_task_id == s.task_count) {
      if (trace && lane == 0) atomicMax(trace + 4ull * s.task_count + 1, tf_glm53_mk_now());
      return;
    }
    if (*shared_task_id != UINT32_MAX) {
      const uint32_t id = *shared_task_id;
      const uint32_t block = *shared_task_block;
      const unsigned long long started = trace && lane == 0 ? tf_glm53_mk_now() : 0ull;
      const tf_glm53_mk_task task = s.tasks[id];
      if (task.reserved > TF_GLM53_MK_QMMF_REDUCE) { if (lane == 0) tf_glm53_mk_fail(s.error, TF_GLM53_MK_UNSUPPORTED); return; }
      // Virtual block coordinates for task bodies that read TF_VBID/TF_VNBLK (MK2 round 4). Every thread stores the
      // same value and reads its own store; the barrier after the previous task already ordered earlier reads.
      if (task.reserved <= TF_GLM53_MK_EMBED_PICK)
        tf_vblock_set(make_uint3(block, 0, 0), make_uint3(s.blocks[id], 1, 1), make_uint3(256, 1, 1));
      if (!tf_glm53_mk_dispatch(static_cast<uint8_t>(task.reserved), s.params[id], block, task_smem)) {
        if (lane == 0) tf_glm53_mk_fail(s.error, TF_GLM53_MK_UNSUPPORTED);
        return;
      }
      __syncthreads();
      if (trace && lane == 0) {
        const unsigned long long ended = tf_glm53_mk_now();
        atomicMin(trace + 4ull * id, started);
        atomicMax(trace + 4ull * id + 1, ended);
        atomicAdd(trace + 4ull * id + 2, ended - started);
      }
      if (lane == 0 && tf_glm53_mk_fetch_add(s.finished_blocks + id, 1u) + 1u == s.blocks[id]) {
        const tf_glm53_mk_task* completed_task = s.tasks + id;
        for (uint32_t e = 0; e < completed_task->successor_count; ++e) {
          const uint32_t successor = s.successors[completed_task->successor_offset + e];
          if (successor >= s.task_count) { tf_glm53_mk_fail(s.error, TF_GLM53_MK_BAD_TASK); break; }
          if (tf_glm53_mk_fetch_sub(s.counters + successor, 1u) == 1u) {
            if (trace) atomicMax(trace + 4ull * successor + 3, tf_glm53_mk_now());
            atomicExch(s.active + successor, 1u);
            if (!tf_glm53_mk_enqueue(successor, s.ready, s.ready_tail, s.ready_lock,
                                     s.ready_capacity, s.blocks[successor])) {
              tf_glm53_mk_fail(s.error, TF_GLM53_MK_BAD_TASK); break;
            }
          }
        }
        atomicExch(s.active + id, 0u);
        tf_glm53_mk_fetch_add(s.completed, 1u);
      }
      __syncthreads();
      spins = 0;
    } else {
      if (++spins >= s.timeout_cycles) { tf_glm53_mk_fail(s.error, TF_GLM53_MK_TIMEOUT); return; }
      __nanosleep(min(32u + spins, 2048u));
    }
  }
}

// MK8 chain scheduler. The deferred launch sink records a linear chain (task i+1 depends on task i), so no ready queue
// is needed: block b of every task runs on worker b % workers, exactly once, as soon as the previous task's last block
// has finished. Same task bodies, same arguments, same order as the per-kernel launches; nothing is claimed
// through a shared queue head (the queue's CAS claim serialises tasks with many short blocks), and with
// TF_GLM53_MK_MAP_SM block b and block b + chain_sms share an SM the way two waves of a normal launch do.
// The SM mapping holds only if every SM runs exactly chain_slots of the grid's CTAs (an idle GPU and a grid of
// SMs x resident CTAs). Every CTA checks its own slot, then all agree: if any slot fell outside the grid, every CTA
// uses blockIdx.x instead, so the worker indices are always a permutation of 0..gridDim.x-1. Lane 0 only.
__device__ __forceinline__ uint32_t tf_glm53_mk_sm_slot(const tf_glm53_mk_state& s) {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (smid >= 1024u || s.chain_sms == 0u || s.chain_sms > 1024u) return UINT32_MAX;
  uint32_t* const map = s.sm_words;
  uint32_t dense = atomicCAS(map + smid, 0u, UINT32_MAX);
  if (dense == 0u) {
    dense = atomicAdd(map + 2048, 1u) + 1u;
    tf_glm53_mk_store_release(map + smid, dense);
  } else {
    while (dense == UINT32_MAX) {
      __nanosleep(32);
      dense = tf_glm53_mk_load_acquire(map + smid);
    }
  }
  dense -= 1u;
  if (dense >= s.chain_sms) return UINT32_MAX;
  const uint32_t slot = atomicAdd(map + 1024 + dense, 1u);
  if (slot >= s.chain_slots) return UINT32_MAX;
  return slot * s.chain_sms + dense;
}

__device__ __forceinline__ uint32_t tf_glm53_mk_chain_worker(const tf_glm53_mk_state& s) {
  if (s.chain_map != TF_GLM53_MK_MAP_SM) return blockIdx.x;
  uint32_t* const map = s.sm_words;
  const uint32_t mapped = tf_glm53_mk_sm_slot(s);
  if (mapped >= gridDim.x) atomicExch(map + 2050, 1u);
  __threadfence();
  atomicAdd(map + 2049, 1u);
  for (uint32_t spins = 0; tf_glm53_mk_load_acquire(map + 2049) != gridDim.x; ++spins) {
    if (spins >= s.timeout_cycles) { tf_glm53_mk_fail(s.error, TF_GLM53_MK_TIMEOUT); return UINT32_MAX; }
    __nanosleep(64);
  }
  return tf_glm53_mk_load_acquire(map + 2050) != 0u ? blockIdx.x : mapped;
}

__device__ __forceinline__ void tf_glm53_mk_chain(tf_glm53_mk_state s, char* task_smem, uint32_t* shared_word) {
  const uint32_t lane = threadIdx.x;
  unsigned long long* const trace = s.trace;
  if (lane == 0) {
    if (trace) atomicMin(trace + 4ull * s.task_count, tf_glm53_mk_now());
    const uint32_t w = tf_glm53_mk_chain_worker(s);
    if (w >= gridDim.x) tf_glm53_mk_fail(s.error, TF_GLM53_MK_UNSUPPORTED);
    *shared_word = w;
  }
  __syncthreads();
  const uint32_t worker = *shared_word;
  if (worker >= gridDim.x) return;
  const uint32_t workers = gridDim.x;
  for (uint32_t id = 0; id < s.task_count; ++id) {
    const uint32_t blocks = s.blocks[id];
    // A worker with no block of task id goes straight on: task id+1 cannot be ready before task id is.
    if (worker >= blocks) continue;
    if (id > 0) {
      __syncthreads();
      if (lane == 0) {
        uint32_t ok = 1, backoff = 32, spins = 0;
        while (tf_glm53_mk_load_acquire(s.finished_blocks + id - 1) != s.blocks[id - 1]) {
          if (tf_glm53_mk_load_acquire(s.error) != TF_GLM53_MK_OK) { ok = 0; break; }
          if (++spins >= s.timeout_cycles) { tf_glm53_mk_fail(s.error, TF_GLM53_MK_TIMEOUT); ok = 0; break; }
          __nanosleep(backoff);
          if (backoff < 256) backoff <<= 1;
        }
        *shared_word = ok;
      }
      __syncthreads();
      if (*shared_word == 0) return;
    }
    const tf_glm53_mk_task task = s.tasks[id];
    for (uint32_t block = worker; block < blocks; block += workers) {
      const unsigned long long started = trace && lane == 0 ? tf_glm53_mk_now() : 0ull;
      if (!tf_glm53_mk_dispatch_deferred(static_cast<uint8_t>(task.reserved), s.params[id], block, task_smem)) {
        if (lane == 0) tf_glm53_mk_fail(s.error, TF_GLM53_MK_UNSUPPORTED);
        return;
      }
      __syncthreads();
      if (lane == 0) {
        __threadfence();
        const uint32_t done = atomicAdd(s.finished_blocks + id, 1u) + 1u;
        if (trace) {
          const unsigned long long ended = tf_glm53_mk_now();
          atomicMin(trace + 4ull * id, started);
          atomicMax(trace + 4ull * id + 1, ended);
          atomicAdd(trace + 4ull * id + 2, ended - started);
          if (done == blocks && id + 1 < s.task_count) atomicMax(trace + 4ull * (id + 1) + 3, ended);
        }
        if (done == blocks) atomicAdd(s.completed, 1u);
      }
    }
  }
  if (trace && lane == 0) atomicMax(trace + 4ull * s.task_count + 1, tf_glm53_mk_now());
}
