// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Synthetic GLM-5.3 q_a/kv_a tensor-parallel split parity.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fp8 = cuda.fp8;

test "sharded qkv-a declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var failed = false;
    try case(gpu, &lane, stream, 2048, 1024, 0, "q_a", &failed);
    try case(gpu, &lane, stream, 576, 256, 0, "kv_a", &failed);
    try case(gpu, &lane, stream, 576, 320, 256, "kv_a", &failed);
    if (failed) return error.TestUnexpectedResult;
    pass("glm53-shard-qkv-a: full/half slice parity and full GEMM determinism", .{});
}

fn case(gpu: check.Gpu, lane: *const cuda.qmmf.Lane, stream: cuda.Stream, full_n: usize, half_n: usize, row0: usize, label: []const u8, failed: *bool) !void {
    const k: usize = 6144;
    if (cuda.qmmf.b16SplitK(full_n, k) != cuda.qmmf.b16SplitK(half_n, k)) failed.* = true;
    const rows_set = [_]usize{ 1, 8, 2048 };
    const a = gpu.gpa;
    const codes = try a.alloc(u8, full_n * k);
    defer a.free(codes);
    const scales = try a.alloc(f32, ((full_n + 127) / 128) * ((k + 127) / 128));
    defer a.free(scales);
    var seed: u32 = 0x53a_771;
    for (codes) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @truncate(0x20 + ((seed >> 24) % 0x60));
        if (v.* & 0x7f == 0x7f) v.* = 0x38;
    }
    for (scales) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(1 + ((seed >> 24) % 8))) / 8;
    }
    const packed_full = try a.alloc(u8, fp8.padded(full_n) * k);
    defer a.free(packed_full);
    fp8.packCodes(packed_full, codes, full_n, k);
    const packed_scales_full = try a.alloc(f32, fp8.padded(full_n) / fp8.group * (k / fp8.group));
    defer a.free(packed_scales_full);
    try fp8.packBlockScales(packed_scales_full, scales, full_n, k, 0, 0, full_n, k);
    const codes_half = try a.alloc(u8, half_n * k);
    defer a.free(codes_half);
    @memcpy(codes_half, codes[row0 * k ..][0 .. half_n * k]);
    const packed_half = try a.alloc(u8, fp8.padded(half_n) * k);
    defer a.free(packed_half);
    fp8.packCodes(packed_half, codes_half, half_n, k);
    const packed_scales_half = try a.alloc(f32, fp8.padded(half_n) / fp8.group * (k / fp8.group));
    defer a.free(packed_scales_half);
    try fp8.packBlockScales(packed_scales_half, scales, full_n, k, row0, 0, half_n, k);
    var dw_full = try cuda.DeviceBuffer.fromHost(gpu.d, packed_full);
    defer dw_full.free();
    var ds_full = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales_full));
    defer ds_full.free();
    var dw_half = try cuda.DeviceBuffer.fromHost(gpu.d, packed_half);
    defer dw_half.free();
    var ds_half = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales_half));
    defer ds_half.free();
    const w_full = fp8.blockWeight(dw_full.ptr, ds_full.ptr, full_n, k);
    const w_half = fp8.blockWeight(dw_half.ptr, ds_half.ptr, half_n, k);
    for (rows_set) |rows| {
        const x = try a.alloc(u16, rows * k);
        defer a.free(x);
        for (x) |*v| {
            seed = seed *% 1664525 +% 1013904223;
            const sample: i32 = @as(i32, @intCast((seed >> 16) % 129)) - 64;
            v.* = @truncate(@as(u32, @bitCast(fp8.roundBf16(@as(f32, @floatFromInt(sample)) / 64))) >> 16);
        }
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var full = try cuda.DeviceBuffer.alloc(gpu.d, rows * full_n * 2);
        defer full.free();
        var full_again = try cuda.DeviceBuffer.alloc(gpu.d, rows * full_n * 2);
        defer full_again.free();
        var half = try cuda.DeviceBuffer.alloc(gpu.d, rows * half_n * 2);
        defer half.free();
        var part_full = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane.partBytes(rows, w_full), 4));
        defer part_full.free();
        var part_half = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane.partBytesPinned(rows, w_half, full_n), 4));
        defer part_half.free();
        try lane.matmulOut(stream, dx.ptr, k, rows, w_full, full.ptr, .bf16, part_full.ptr);
        try lane.matmulOut(stream, dx.ptr, k, rows, w_full, full_again.ptr, .bf16, part_full.ptr);
        try lane.matmulOutPinned(stream, dx.ptr, k, rows, w_half, half.ptr, .bf16, part_half.ptr, full_n);
        try stream.synchronize();
        const full_bytes = try download(gpu, full);
        defer a.free(full_bytes);
        const full_again_bytes = try download(gpu, full_again);
        defer a.free(full_again_bytes);
        const half_bytes = try download(gpu, half);
        defer a.free(half_bytes);
        if (!std.mem.eql(u8, full_again_bytes, full_bytes)) {
            failed.* = true;
            reportDiff(label, rows, "full-repeat", full_bytes, full_again_bytes, full_n, full_n, full_n, w_full, lane.clusters, packed_scales_full[0], 0);
        }
        const full_slice = try a.alloc(u8, rows * half_n * 2);
        defer a.free(full_slice);
        for (0..rows) |r| @memcpy(full_slice[r * half_n * 2 ..][0 .. half_n * 2], full_bytes[r * full_n * 2 + row0 * 2 ..][0 .. half_n * 2]);
        if (!std.mem.eql(u8, half_bytes, full_slice)) {
            failed.* = true;
            reportDiff(label, rows, "full-vs-half", full_slice, half_bytes, half_n, full_n, full_n, w_full, lane.clusters, packed_scales_half[0], row0);
        } else {
            std.debug.print("RESULT glm53-shard-qkv-a {s} slice_row={d} rows={d}: byte-identical\n", .{ label, row0, rows });
        }
    }
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn reportDiff(label: []const u8, rows: usize, kind: []const u8, want: []const u8, got: []const u8, output_n: usize, full_plan_n: usize, other_plan_n: usize, weight: cuda.qmmf.Weight, clusters: bool, first_scale: f32, weight_row: usize) void {
    var first: ?usize = null;
    var max_abs: f32 = 0;
    var count: usize = 0;
    for (0..@min(want.len, got.len) / 2) |i| {
        const a = std.mem.readInt(u16, want[i * 2 ..][0..2], .little);
        const b = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
        if (a == b) continue;
        if (first == null) first = i;
        count += 1;
        max_abs = @max(max_abs, @abs(bf16Float(a) - bf16Float(b)));
    }
    const idx = first orelse 0;
    const p_full = cuda.qmmf.planForMode(rows, full_plan_n, weight.k, clusters, weight.mode);
    const actual_other_n = @divExact(got.len, rows * 2);
    const p_other = cuda.qmmf.planForMode(rows, other_plan_n, weight.k, clusters, weight.mode);
    std.debug.print("FAIL glm53-shard-qkv-a {s} rows={d} {s}: first=(row={d},col={d}) bf16=0x{x:0>4} ({d:.8}) vs 0x{x:0>4} ({d:.8}); max_abs={d:.8} differing={d}\n", .{ label, rows, kind, idx / output_n, idx % output_n, std.mem.readInt(u16, want[idx * 2 ..][0..2], .little), bf16Float(std.mem.readInt(u16, want[idx * 2 ..][0..2], .little)), std.mem.readInt(u16, got[idx * 2 ..][0..2], .little), bf16Float(std.mem.readInt(u16, got[idx * 2 ..][0..2], .little)), max_abs, count });
    printPlan("full", rows, full_plan_n, p_full);
    printPlan(if (std.mem.eql(u8, kind, "full-repeat")) "repeat" else "half", rows, actual_other_n, p_other);
    std.debug.print("weight slice first_row={d} first_scale_block={d} scale={d:.8}\n", .{ weight_row, weight_row / 128, first_scale });
}

fn printPlan(name: []const u8, m: usize, n: usize, p: cuda.qmmf.Plan) void {
    const grid_x = ((m + p.bm - 1) / p.bm) * ((n + 63) / 64);
    const mode: []const u8 = if (p.fused) "fused" else if (p.cluster) "cluster" else if (p.reduce) "reduce" else "plain";
    std.debug.print("  {s}: sk={d} mode={s} tile={d}x64 grid={d}x1x{d}\n", .{ name, p.sk, mode, p.bm, grid_x, if (p.fused) 1 else p.sk });
}

fn bf16Float(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}
