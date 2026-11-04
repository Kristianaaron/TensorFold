// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! `tf-cuda-test glm53-prefill-bf16`: BF16 scratch/cuBLASLt vs native FP8B MMA and row split invariance.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fp8 = cuda.fp8;
const workspace_bytes = 64 << 20;

const Shape = struct { n: usize, k: usize };
const shapes = [_]Shape{ .{ .n = 2048, .k = 6144 }, .{ .n = 8192, .k = 2048 }, .{ .n = 12288, .k = 6144 } };
const row_counts = [_]usize{ 2048, 4096, 8192 };

pub fn run(gpu: check.Gpu) !void {
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    var workspace = try cuda.DeviceBuffer.alloc(gpu.d, workspace_bytes);
    defer workspace.free();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var side = try cuda.Stream.init(gpu.d, true);
    defer side.deinit();
    var ready = try cuda.Event.init(gpu.d, false);
    defer ready.deinit();
    var part = try cuda.Event.init(gpu.d, false);
    defer part.deinit();
    try part.record(stream);

    for (shapes) |shape| try runShape(gpu, &lane, &lt, workspace.ptr, stream, side, ready, part, shape);
}

fn runShape(gpu: check.Gpu, lane: *const cuda.qmmf.Lane, lt: *cuda.cublaslt.Library, workspace: u64, stream: cuda.Stream, side: cuda.Stream, ready: cuda.Event, consumed: cuda.Event, shape: Shape) !void {
    const gpa = gpu.gpa;
    const n = shape.n;
    const k = shape.k;
    const npad = fp8.padded(n);
    const codes = try gpa.alloc(u8, npad * k);
    defer gpa.free(codes);
    const raw = try gpa.alloc(u8, n * k);
    defer gpa.free(raw);
    const raw_scales = try gpa.alloc(f32, ((n + 127) / 128) * (k / 128));
    defer gpa.free(raw_scales);
    const scales = try gpa.alloc(f32, (npad / fp8.group) * (k / fp8.group));
    defer gpa.free(scales);
    var rng = std.Random.DefaultPrng.init(0x53_2026 ^ @as(u64, @intCast(n)) ^ (@as(u64, @intCast(k)) << 20));
    const random = rng.random();
    for (raw) |*v| {
        v.* = random.int(u8) & 0x7e;
        if (v.* & 0x7f == 0x7f) v.* = 0x38;
    }
    for (raw_scales) |*v| v.* = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -5, 5));
    fp8.packCodes(codes, raw, n, k);
    try fp8.packBlockScales(scales, raw_scales, n, k, 0, 0, n, k);

    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    const weight = fp8.blockWeight(dw.ptr, ds.ptr, n, k);
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
    defer scratch.free();
    var linear = try cuda.cublaslt.Linear.init(lt, 2048, n, k, .bf16, workspace_bytes);
    defer linear.deinit();
    try side.wait(consumed);
    try lane.dequantizeFp8bBf16RowMajor(side, weight, scratch.ptr);
    try ready.record(side);
    try stream.wait(ready);

    for (row_counts) |m| {
        const x = try gpa.alloc(u16, m * k);
        defer gpa.free(x);
        for (x) |*v| v.* = bf16Bits((@as(f32, @floatFromInt(random.intRangeAtMost(i32, -64, 64))) / 64.0));
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var d_bf16 = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer d_bf16.free();
        var d_mma = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer d_mma.free();
        var scratch_part = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane.partBytes(m, weight), 4));
        defer scratch_part.free();

        try linear.runRows(m, dx.ptr, scratch.ptr, d_bf16.ptr, workspace, workspace_bytes, stream.handle);
        try lane.matmul(stream, dx.ptr, k, m, weight, d_mma.ptr, scratch_part.ptr);
        try stream.synchronize();
        const got = try check.download(gpu, d_bf16);
        defer gpa.free(got);
        const ref = try check.download(gpu, d_mma);
        defer gpa.free(ref);
        var max_abs: f32 = 0;
        for (std.mem.bytesAsSlice(u16, got), std.mem.bytesAsSlice(u16, ref)) |a, b| max_abs = @max(max_abs, @abs(bf16Float(a) - bf16Float(b)));
        std.debug.print("BF16 prefill M={d} N={d} K={d}: max_abs_diff={e:.6}\n", .{ m, n, k, max_abs });

        var chunk = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer chunk.free();
        var at: usize = 0;
        while (at < m) : (at += @min(2048, m - at)) {
            const rows = @min(2048, m - at);
            try linear.runRows(rows, dx.ptr + at * k * 2, scratch.ptr, chunk.ptr + at * n * 2, workspace, workspace_bytes, stream.handle);
        }
        try stream.synchronize();
        const split = try check.download(gpu, chunk);
        defer gpa.free(split);
        const equal = std.mem.eql(u8, got, split);
        std.debug.print("BF16 prefill M={d} N={d} fixed 2048-row splits: bit_equal={}\n", .{ m, n, equal });
        if (!equal) return error.PrefillChunkChangedBits;
    }
    try consumed.record(stream);
    try stream.synchronize();
}

fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn bf16Float(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

test "test references the full GPU prefill parity path" {
    _ = &run;
    _ = &runShape;
}

test {
    std.testing.refAllDecls(@This());
}
