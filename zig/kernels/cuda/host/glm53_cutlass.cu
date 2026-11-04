// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted from NVIDIA/cutlass@91843b71d90737e0a9b86f7d9df213cff5a3526f:
 * examples/87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu.
 * Changes: expose a small C ABI launcher for TensorFold row-major FP8 weights and contiguous block scales.
 **************************************************************************************************/

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {
using namespace cute;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128, UMMA::Major::K, UMMA::Major::K>;
using LayoutSFA = typename ScaleConfig::LayoutSFA;
using LayoutSFB = typename ScaleConfig::LayoutSFB;
constexpr int AlignA = 16;
constexpr int AlignB = 16;
constexpr int AlignD = 8;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute, ElementD, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignB,
    ElementAccumulator, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;

Gemm::Arguments make_arguments(const void *a, const void *b, const float *sfa, const float *sfb,
                               void *d, int m, int n, int k) {
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const ElementA *>(a), stride_a,
       static_cast<const ElementB *>(b), stride_b,
       sfa, layout_sfa, sfb, layout_sfb},
      {{}, static_cast<ElementD *>(d), stride_d, static_cast<ElementD *>(d), stride_d}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  return args;
}
} // namespace

extern "C" int tf_glm53_cutlass_init() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 1;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 2;
#ifdef TF_GLM53_ALLOW_SM120
  // Test builds for RTX 50-series (sm_120, same Blackwell GeForce family) compile with -DTF_GLM53_ALLOW_SM120 and
  // -gencode=arch=compute_120a,code=sm_120a; production (GB10, sm_121a) keeps the strict 12.1 check.
  if (prop.major != 12 || (prop.minor != 0 && prop.minor != 1)) return 3;
#else
  if (prop.major != 12 || prop.minor != 1) return 3;
#endif
  // CUTLASS's adapter explicitly opts into its compiled dynamic shared-memory requirement (above 48 KiB).
  // maximum_active_blocks() calls cudaFuncSetAttribute and returns -1 if the opt-in fails.
  return Gemm::maximum_active_blocks() < 0 ? 4 : 0;
}

extern "C" size_t tf_glm53_cutlass_dynamic_smem_bytes() {
  return Gemm::GemmKernel::SharedStorageSize;
}

extern "C" size_t tf_glm53_cutlass_workspace(int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || n % 4 || k % 128) return SIZE_MAX;
  auto args = make_arguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k);
  return Gemm::get_workspace_size(args);
}

namespace {
struct CachedPlan {
  const void *a;
  const void *b;
  const float *sfa;
  const float *sfb;
  void *d;
  void *workspace;
  int m, n, k;
  size_t required_workspace;
  std::unique_ptr<Gemm> gemm;
};
std::mutex plan_mutex;
std::vector<CachedPlan> plans;
} // namespace

extern "C" int tf_glm53_cutlass_gemm(const void *a, const void *b, const float *sfa,
                                     const float *sfb, void *d, int m, int n, int k,
                                     void *workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (!a || !b || !sfa || !sfb || !d || m <= 0 || n <= 0 || k <= 0 || n % 4 || k % 128) return 10;
  if (((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(d)) & 15u) ||
      ((reinterpret_cast<uintptr_t>(sfa) | reinterpret_cast<uintptr_t>(sfb)) & 3u)) return 13;
  std::lock_guard<std::mutex> lock(plan_mutex);
  for (auto &plan : plans) {
    if (plan.a == a && plan.b == b && plan.sfa == sfa && plan.sfb == sfb && plan.d == d &&
        plan.workspace == workspace && plan.m == m && plan.n == n && plan.k == k) {
      if (workspace_bytes < plan.required_workspace || (plan.required_workspace && !workspace)) return 11;
      auto status = plan.gemm->run(stream);
      return status == cutlass::Status::kSuccess ? 0 : 52 + static_cast<int>(status);
    }
  }
  auto args = make_arguments(a, b, sfa, sfb, d, m, n, k);
  const size_t need = Gemm::get_workspace_size(args);
  if (need == SIZE_MAX || workspace_bytes < need || (need && !workspace)) return 11;
  auto gemm = std::make_unique<Gemm>();
  auto status = gemm->can_implement(args);
  if (status != cutlass::Status::kSuccess) return 12 + static_cast<int>(status);
  status = gemm->initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) return 32 + static_cast<int>(status);
  status = gemm->run(stream);
  if (status == cutlass::Status::kSuccess) plans.push_back({a, b, sfa, sfb, d, workspace, m, n, k, need, std::move(gemm)});
  return status == cutlass::Status::kSuccess ? 0 : 52 + static_cast<int>(status);
}
