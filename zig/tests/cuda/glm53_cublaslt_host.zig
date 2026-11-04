// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");
const cuda = @import("cuda");

test "cuBLASLt FP8 block-scale ABI declarations are fully analyzed" {
    std.testing.refAllDecls(cuda.cublaslt);
    std.testing.refAllDecls(cuda.cublaslt.Linear);
    std.testing.refAllDecls(cuda.cublaslt.Fp8BlockLinear);
    const mapping = cuda.cublaslt.fp8TnLayouts();
    try std.testing.expectEqual(cuda.cublaslt.Op.t, mapping.transa);
    try std.testing.expectEqual(cuda.cublaslt.Op.n, mapping.transb);
    try std.testing.expectEqual(cuda.cublaslt.Order.col, mapping.order);
    try std.testing.expectEqual(cuda.cublaslt.scale_blk128x128_32f, mapping.scale_a);
    try std.testing.expectEqual(cuda.cublaslt.scale_vec128_32f, mapping.scale_b);
}
