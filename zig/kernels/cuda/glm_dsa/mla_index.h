// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#ifndef TF_GLM53_MLA_INDEX_H
#define TF_GLM53_MLA_INDEX_H

#include <stdint.h>

// Shared by the CUDA kernel and the Zig host reference: peer rows are stored at absolute DCP slots.
#ifdef __CUDACC__
#define TF_GLM53_HD __host__ __device__
#else
#define TF_GLM53_HD
#endif
static TF_GLM53_HD inline int glm53_stage_plane_slot(int64_t position, int world) {
  return (int)(position / world);
}

#ifndef __CUDACC__
int glm53_stage_plane_slot_host(int64_t position, int world);
#endif

#endif
