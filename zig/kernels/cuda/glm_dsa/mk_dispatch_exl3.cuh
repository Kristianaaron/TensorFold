// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31:zig/kernels/cuda/exl3_experts.cu (Apache-2.0).
// Changes: add fixed-layout persistent-task payloads and virtual-block adapters; task math remains upstream-local.
#pragma once
#include "../exl3_experts.cu"

struct tf_glm53_mk_exl3_rot_in_params { uint32_t gx, gy, gz, threads; tf_exl3::RotInArgs task; };
struct tf_glm53_mk_exl3_down_combine_params { uint32_t gx, gy, gz, threads; tf_exl3::DownArgs task; };
static_assert(sizeof(tf_glm53_mk_exl3_rot_in_params) == 88, "EXL3 rot-in payload ABI");
static_assert(sizeof(tf_glm53_mk_exl3_down_combine_params) == 88, "EXL3 down payload ABI");
static_assert(offsetof(tf_glm53_mk_exl3_rot_in_params, task) == 16, "EXL3 rot-in argument ABI");
static_assert(offsetof(tf_glm53_mk_exl3_down_combine_params, task) == 16, "EXL3 down argument ABI");

__device__ __forceinline__ void tf_glm53_mk_run_exl3_rot_in(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_exl3_rot_in_params*>(raw);
  tf_vblock_set(make_uint3(block % p.gx, (block / p.gx) % p.gy, block / (p.gx * p.gy)),
                make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_exl3::tf_exl3_experts_rot_in_task(p.task);
}

__device__ __forceinline__ void tf_glm53_mk_run_exl3_down_combine(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_exl3_down_combine_params*>(raw);
  tf_vblock_set(make_uint3(block % p.gx, (block / p.gx) % p.gy, block / (p.gx * p.gy)),
                make_uint3(p.gx, p.gy, p.gz), make_uint3(p.threads, 1, 1));
  if (threadIdx.x >= p.threads) return;
  tf_exl3::tf_exl3_down_combine_task(p.task, smem);
}
