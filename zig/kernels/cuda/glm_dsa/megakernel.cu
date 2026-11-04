// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// TensorFold GLM-5.3 persistent task scheduler and device interpreter.
#include <math_constants.h>
#include <cuda_runtime.h>
#include "mk_runtime.cuh"

// Kept as separate kernels so a captured step is reset -> persistent execution.
extern "C" __global__ void tf_glm53_mk_reset(tf_glm53_mk_task*, uint32_t* counters,
    const uint32_t* initial_counters, const uint32_t* blocks, uint32_t* next_block,
    uint32_t* finished_blocks, uint32_t* active, uint32_t* ready, uint32_t* ready_head,
    uint32_t* ready_tail, uint32_t* completed, uint32_t* error, uint32_t* ready_lock,
    uint32_t task_count, uint32_t ready_capacity, uint32_t* sm_words) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (sm_words && i < TF_GLM53_MK_SM_WORDS) sm_words[i] = 0;
  if (i < task_count) {
    counters[i] = initial_counters[i];
    next_block[i] = 0;
    finished_blocks[i] = 0;
    active[i] = 0;
  }
  if (i == 0) {
    *ready_head = 0;
    *ready_tail = 0;
    *completed = 0;
    *error = TF_GLM53_MK_OK;
    *ready_lock = 0;
  }
  __syncthreads();
  if (i < task_count && initial_counters[i] == 0) {
    const uint32_t slot = atomicAdd(ready_tail, blocks[i]);
    if (slot > ready_capacity || blocks[i] > ready_capacity - slot) {
      atomicCAS(error, TF_GLM53_MK_OK, TF_GLM53_MK_BAD_TASK);
    } else {
      active[i] = 1;
      for (uint32_t b = 0; b < blocks[i]; ++b) ready[slot + b] = i;
    }
  }
}

extern "C" __global__ void tf_glm53_mk_run(tf_glm53_mk_state state) {
  extern __shared__ __align__(32) char task_smem[];
  __shared__ uint32_t shared_task_id;
  __shared__ uint32_t shared_task_block;
  tf_glm53_mk_persistent(state, task_smem, &shared_task_id, &shared_task_block);
}

// MK8 chain scheduler over the deferrable ops (tf_glm53_mk_chain): what the deferred launch sink runs.
extern "C" __global__ void tf_glm53_mk_run_chain(tf_glm53_mk_state state) {
  extern __shared__ __align__(32) char task_smem[];
  __shared__ uint32_t shared_word;
  tf_glm53_mk_chain(state, task_smem, &shared_word);
}

// MK8 deferred launch: one segment's payload arena refresh. The bytes travel as the kernel parameter, so a CUDA
// graph capture bakes them into the node (no host staging buffer has to outlive the capture).
#define TF_GLM53_MK_STORE_BYTES 2032
struct tf_glm53_mk_store_chunk {
  uint8_t* dst;
  uint32_t len;
  uint32_t reserved;
  uint8_t bytes[TF_GLM53_MK_STORE_BYTES];
};
static_assert(sizeof(tf_glm53_mk_store_chunk) == 2048, "MK8 store chunk ABI (Zig defer.zig StoreChunk)");
static_assert(offsetof(tf_glm53_mk_store_chunk, bytes) == 16, "MK8 store chunk bytes offset");

extern "C" __global__ void tf_glm53_mk_store(const tf_glm53_mk_store_chunk chunk) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < chunk.len; i += gridDim.x * blockDim.x)
    chunk.dst[i] = chunk.bytes[i];
}

// Per-task reference entry used by tf-cuda-test's scheduler parity run.
extern "C" __global__ void tf_glm53_mk_add_partial_ref(tf_glm53_add_partial_params params) {
  tf_glm53_add_partial_task(params, blockIdx.x, 0, nullptr);
}
