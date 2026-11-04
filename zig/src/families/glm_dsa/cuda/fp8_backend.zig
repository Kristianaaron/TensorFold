// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Runtime selection and shared interfaces for GLM-5.3 block-scaled FP8 GEMM backends.
const std = @import("std");

pub const Backend = enum { mma, cutlass, cublaslt };

pub const Storage = enum { packed_fp8b, row_major };

pub const ShapeSupport = struct {
    cutlass: bool,
    cublaslt: bool,
};

/// Choose storage before conversion. Any shape that might use the MMA fallback
/// keeps its original packed copy; conversion is safe only when the chosen
/// backend covers every M the caller will issue.
pub fn chooseStorage(support: ShapeSupport, requested: Backend) struct { backend: Backend, storage: Storage } {
    const backend: Backend = switch (requested) {
        .mma => .mma,
        .cutlass => if (support.cutlass) .cutlass else .mma,
        .cublaslt => if (support.cublaslt) .cublaslt else .mma,
    };
    return .{ .backend = backend, .storage = if (backend == .mma) .packed_fp8b else .row_major };
}

pub fn shapeSupport(n: usize, k: usize, all_m_divisible_by_four: bool) ShapeSupport {
    return .{
        .cutlass = n % 128 == 0 and k % 128 == 0,
        .cublaslt = n % 128 == 0 and k % 128 == 0 and all_m_divisible_by_four,
    };
}

pub fn selected(value: ?[]const u8) error{UnknownBackend}!Backend {
    const name = value orelse return .mma;
    if (std.mem.eql(u8, name, "mma")) return .mma;
    if (std.mem.eql(u8, name, "cutlass")) return .cutlass;
    if (std.mem.eql(u8, name, "cublaslt")) return .cublaslt;
    return error.UnknownBackend;
}

pub fn fromEnvironment() error{UnknownBackend}!Backend {
    const raw = std.c.getenv("TF_GLM53_FP8_GEMM") orelse return .mma;
    return selected(std.mem.span(raw));
}

test "backend selector defaults to mma and accepts both optional paths" {
    try std.testing.expectEqual(Backend.mma, try selected(null));
    try std.testing.expectEqual(Backend.cutlass, try selected("cutlass"));
    try std.testing.expectEqual(Backend.cublaslt, try selected("cublaslt"));
    try std.testing.expectError(error.UnknownBackend, selected("unknown"));
}

test "backend storage selection preserves packed fallback for unsupported shapes" {
    const fast = shapeSupport(6144, 6144, true);
    try std.testing.expectEqual(Storage.row_major, chooseStorage(fast, .cutlass).storage);
    try std.testing.expectEqual(Backend.cublaslt, chooseStorage(fast, .cublaslt).backend);
    const ragged = shapeSupport(576, 6144, true);
    try std.testing.expectEqual(Backend.mma, chooseStorage(ragged, .cutlass).backend);
    try std.testing.expectEqual(Storage.packed_fp8b, chooseStorage(ragged, .cutlass).storage);
    const decode_m = shapeSupport(6144, 6144, false);
    try std.testing.expectEqual(Backend.mma, chooseStorage(decode_m, .cublaslt).backend);
    try std.testing.expectEqual(Storage.packed_fp8b, chooseStorage(decode_m, .cublaslt).storage);
}

test {
    std.testing.refAllDecls(@This());
}
