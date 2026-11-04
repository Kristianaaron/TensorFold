// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Persistent qmmf adapters. Partial MMA is the same qmmf task with a 256-thread
// tile; its output accumulator sequence is unchanged from the 128-thread cluster CTA.
#pragma once

struct tf_glm53_mk_qmmf_partial_params {
  uint32_t gx, gy, gz, threads;
  const __nv_bfloat16* x; const unsigned char* w; const uint8_t* scales;
  float scale; void* out; float* partial;
  int m, n, k, sk, npad, ldx, group;
  int mode, f32;
};
struct tf_glm53_mk_qmmf_reduce_params {
  uint32_t gx, gy, gz, threads;
  const float* partial; void* out; long long total; int sk; float scale; int f32;
};

static_assert(sizeof(tf_glm53_mk_qmmf_partial_params) == 104, "MK qmmf partial ABI");
static_assert(sizeof(tf_glm53_mk_qmmf_reduce_params) == 56, "MK qmmf reduce ABI");
static_assert(offsetof(tf_glm53_mk_qmmf_partial_params, x) == 16, "MK qmmf partial offset");
static_assert(offsetof(tf_glm53_mk_qmmf_reduce_params, partial) == 16, "MK qmmf reduce offset");

__device__ __forceinline__ void tf_glm53_mk_qmmf_vblock(uint32_t block, uint32_t gx,
                                                         uint32_t gy, uint32_t gz) {
  tf_vblock_set(make_uint3(block % gx, (block / gx) % gy, block / (gx * gy)),
                make_uint3(gx, gy, gz), make_uint3(256, 1, 1));
}

template <int MODE, bool F32>
__device__ __forceinline__ void tf_glm53_mk_qmmf_mma(const tf_glm53_mk_qmmf_partial_params& p, char* smem) {
  // BM=32, WM=2 preserves the GLM BM=16 CTA's row/column accumulator order,
  // while allowing the persistent scheduler's full 256-thread block to sync.
  tf_qmmf::qmmf_task<MODE, 32, 64, 2, 4, 4, F32, false, false>(
      p.x, p.w, p.scales, p.scale, p.out, p.partial, p.m, p.n, p.k, p.sk, p.npad, p.ldx, p.group,
      reinterpret_cast<unsigned char*>(smem));
}

__device__ __forceinline__ void tf_glm53_mk_run_qmmf_partial(const void* raw, uint32_t block, char* smem) {
  const auto& p = *static_cast<const tf_glm53_mk_qmmf_partial_params*>(raw);
  tf_glm53_mk_qmmf_vblock(block, p.gx, p.gy, p.gz);
  if (p.mode == tf_qmmf::FP8B) {
    if (p.f32) tf_glm53_mk_qmmf_mma<tf_qmmf::FP8B, true>(p, smem);
    else tf_glm53_mk_qmmf_mma<tf_qmmf::FP8B, false>(p, smem);
  } else {
    if (p.f32) tf_glm53_mk_qmmf_mma<tf_qmmf::BF16, true>(p, smem);
    else tf_glm53_mk_qmmf_mma<tf_qmmf::BF16, false>(p, smem);
  }
}

__device__ __forceinline__ void tf_glm53_mk_run_qmmf_reduce(const void* raw, uint32_t block, char*) {
  const auto& p = *static_cast<const tf_glm53_mk_qmmf_reduce_params*>(raw);
  if (p.sk == 1) return; // The partial task wrote the ordinary qmmf epilogue directly.
  tf_vblock_set(make_uint3(block, 0, 0), make_uint3(p.gx, 1, 1), make_uint3(256, 1, 1));
  if (p.f32) tf_qmmf::reduce_task<true>(p.partial, p.out, p.total, p.sk, p.scale);
  else tf_qmmf::reduce_task<false>(p.partial, p.out, p.total, p.sk, p.scale);
}
