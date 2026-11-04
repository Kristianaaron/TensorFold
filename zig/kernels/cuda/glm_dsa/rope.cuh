// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Shared GLM-5.3 RoPE arithmetic follows bertholomus/bertholomus/TensorFold@ddfad35
// `src/tensorfold/families/glm_moe_dsa/cuda/rope.py:_apply_heads` (Apache-2.0).
#pragma once

#include "../tf_vblock.cuh"

#include <cuda_runtime.h>

namespace glm53_rope {
__device__ __forceinline__ float even(float x, float c, float y, float s) {
  return __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
}
__device__ __forceinline__ float odd(float y, float c, float x, float s) {
  return __fadd_rn(__fmul_rn(y, c), __fmul_rn(x, s));
}
}  // namespace glm53_rope
