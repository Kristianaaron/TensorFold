// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host ABI and grid planning for MK qmmf split-K tasks.
const std = @import("std");

pub const Mode = enum(i32) { fp8b = 5, bf16 = 4 };

pub const PartialParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    x: u64,
    w: u64,
    scales: u64,
    scale: f32,
    out: u64,
    partial: u64,
    m: i32,
    n: i32,
    k: i32,
    sk: i32,
    npad: i32,
    ldx: i32,
    group: i32,
    mode: i32,
    f32: i32,
};

pub const ReduceParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    partial: u64,
    out: u64,
    total: i64,
    sk: i32,
    scale: f32,
    f32: i32,
};

pub const Shape = struct { rows: usize, n: usize, k: usize, npad: usize, ldx: usize, mode: Mode, f32: bool };
pub const Plan = struct { partial: PartialParams, reduce: ReduceParams, partial_blocks: u32, reduce_blocks: u32, dynamic_smem: u32 };
pub const TaskPayloads = struct { partial: PartialParams, reduce: ReduceParams };

/// Payload order matches graph.qmmf_partial -> graph.qmmf_reduce. Both records retain
/// the caller's x/weight/output/scratch addresses and the planner's split-K shape.
pub fn taskPayloads(plan: Plan) TaskPayloads {
    return .{ .partial = plan.partial, .reduce = plan.reduce };
}

/// GLM decode uses b16_split_k, even for the native FP8B mode.
pub fn splitK(n: usize, k: usize) !usize {
    if (n == 0 or k == 0 or k % 64 != 0) return error.BadShape;
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) : (sk *= 2) {}
    return sk;
}

pub fn bind(shape: Shape, x: u64, w: u64, scales: u64, out: u64, partial: u64) !Plan {
    if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.k == 0 or shape.npad < shape.n or
        shape.ldx < shape.k or x == 0 or w == 0 or out == 0 or partial == 0 or (shape.mode == .fp8b and scales == 0)) return error.BadShape;
    const sk = try splitK(shape.n, shape.k);
    const row_tiles = (shape.rows + 31) / 32;
    const n_tiles = (shape.n + 63) / 64;
    const gx = row_tiles * n_tiles;
    const blocks = gx * sk;
    const total = shape.rows * shape.n;
    const group: usize = @max(1, @min(row_tiles, (12 * 1024 * 1024) / (32 * shape.k * 2)));
    return .{
        .partial = .{ .gx = @intCast(gx), .gy = 1, .gz = @intCast(sk), .threads = 256, .x = x, .w = w, .scales = scales, .scale = 1, .out = out, .partial = partial, .m = @intCast(shape.rows), .n = @intCast(shape.n), .k = @intCast(shape.k), .sk = @intCast(sk), .npad = @intCast(shape.npad), .ldx = @intCast(shape.ldx), .group = @intCast(group), .mode = @backingInt(shape.mode), .f32 = @intFromBool(shape.f32) },
        .reduce = .{ .gx = @intCast((total + 255) / 256), .gy = 1, .gz = 1, .threads = 256, .partial = partial, .out = out, .total = @intCast(total), .sk = @intCast(sk), .scale = 1, .f32 = @intFromBool(shape.f32) },
        .partial_blocks = @intCast(blocks),
        .reduce_blocks = @intCast((total + 255) / 256),
        .dynamic_smem = 49_152,
    };
}

test "qmmf ABI and GLM decode projection grids" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Mode);
    std.testing.refAllDecls(Shape);
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(TaskPayloads);
    try std.testing.expectEqual(@as(usize, 104), @sizeOf(PartialParams));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(ReduceParams));
    const shapes = [_][2]usize{ .{ 2048, 6144 }, .{ 4096, 2048 }, .{ 576, 6144 }, .{ 3072, 8192 }, .{ 6144, 6144 }, .{ 77440, 6144 } };
    for (shapes) |dims| for ([_]usize{ 1, 4 }) |rows| {
        const n = dims[0];
        const p = try bind(.{ .rows = rows, .n = n, .k = dims[1], .npad = (n + 127) / 128 * 128, .ldx = dims[1], .mode = .fp8b, .f32 = false }, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000);
        try std.testing.expectEqual(@as(i32, @intCast(rows)), p.partial.m);
        try std.testing.expectEqual(@as(i32, @intCast(try splitK(n, dims[1]))), p.partial.sk);
        try std.testing.expectEqual(@as(u32, 256), p.partial.threads);
        try std.testing.expectEqual(@as(u32, 49_152), p.dynamic_smem);
        try std.testing.expectEqual(@as(u32, @intCast((rows * n + 255) / 256)), p.reduce_blocks);
        const payloads = taskPayloads(p);
        try std.testing.expectEqual(p.partial.x, payloads.partial.x);
        try std.testing.expectEqual(p.partial.partial, payloads.reduce.partial);
        try std.testing.expectEqual(p.partial.out, payloads.reduce.out);
        try std.testing.expectEqual(p.partial.sk, payloads.reduce.sk);
    };
    const head = try bind(.{ .rows = 4, .n = 77440, .k = 6144, .npad = 77440, .ldx = 6144, .mode = .bf16, .f32 = false }, 0x1000, 0x2000, 0, 0x4000, 0x5000);
    try std.testing.expectEqual(@as(u64, 0), head.partial.scales);
    try std.testing.expectEqual(@as(i32, @intCast(try splitK(77440, 6144))), head.partial.sk);
}
