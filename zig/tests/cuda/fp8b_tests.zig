// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Adapted from bertholomus/TensorFold@ddfad35: glm5_next/cuda/qmm.py (Apache-2.0).
//! Changes: host-only references and fixture dispatch for native block-FP8 W8A16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const fp8 = cuda.fp8;

test "fp8b GPU fixture and benchmark entry points type-check on the host" {
    _ = &lane;
    _ = &bench;
}

test "fp8b no-argument smoke case uses native one-block scales and expanded tile scales" {
    const n = 64;
    const k = 128;
    const scale_inv = [_]f32{0.75};
    var packed_scales: [4]f32 = undefined;
    try fp8.packBlockScales(&packed_scales, &scale_inv, n, k, 0, 0, n, k);
    try std.testing.expectEqualSlices(f32, &.{ 0.75, 0.75, 1, 1 }, &packed_scales);
}

test "fp8b reusable BF16 scratch dequant is bit exact to the in-kernel round point" {
    const n = 128;
    const k = 64;
    var codes: [n * k]u8 = @splat(0);
    for (0..n) |r| {
        for (0..k) |c| codes[fp8.packedCodeIndex(r, c, k)] = 0x38;
    }
    const scales = [_]f32{ 0.5, 0.5 };
    var out: [n * k]u16 = undefined;
    try fp8.dequantizeBf16Packed(&out, &codes, &scales, n, k);
    for (out) |v| try std.testing.expectEqual(@as(u16, 0x3f00), v);
}

test "fp8b scratch launcher is part of the loaded qmmf lane" {
    _ = &cuda.qmmf.Lane.dequantizeFp8bBf16;
    _ = &cuda.qmmf.Lane.dequantizeFp8bBf16RowMajor;
}

/// Run a Python fixture written by oracle/fp8b_lane.py through the `.fp8b` qmmf lane.
pub fn lane(gpu: Gpu, dir: ?[]const u8) !void {
    if (dir == null) return defaultLane(gpu);
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir.?);
    defer fx.deinit();
    const a = gpu.gpa;
    const n: usize = @intCast(try fx.int("n"));
    const k: usize = @intCast(try fx.int("k"));
    const row0: usize = @intCast(try fx.int("row0"));
    const col0: usize = @intCast(try fx.int("col0"));
    const full_n: usize = @intCast(try fx.int("full_n"));
    const full_k: usize = @intCast(try fx.int("full_k"));
    const codes = try fx.bytes("codes");
    defer a.free(codes);
    const inv_bytes = try fx.bytes("scale_inv");
    defer a.free(inv_bytes);
    const inv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, inv_bytes));
    const inv_aligned = try a.alloc(f32, inv.len);
    defer a.free(inv_aligned);
    for (inv, inv_aligned) |v, *o| o.* = v;

    const packed_codes = try a.alloc(u8, fp8.padded(n) * k);
    defer a.free(packed_codes);
    fp8.packCodes(packed_codes, codes, n, k);
    const packed_scales = try a.alloc(f32, fp8.padded(n) / fp8.group * (k / fp8.group));
    defer a.free(packed_scales);
    try fp8.packBlockScales(packed_scales, inv_aligned, full_n, full_k, row0, col0, n, k);

    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales));
    defer ds.free();
    var lane_ = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane_.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const w = fp8.blockWeight(dw.ptr, ds.ptr, n, k);
    var it = std.mem.tokenizeScalar(u8, try fx.string("rows"), ',');
    while (it.next()) |tok| {
        const m = try std.fmt.parseInt(usize, tok, 10);
        var name: [16]u8 = undefined;
        const x = try fx.bytes(try std.fmt.bufPrint(&name, "x{d}", .{m}));
        defer a.free(x);
        const want = try fx.bytes(try std.fmt.bufPrint(&name, "y{d}", .{m}));
        defer a.free(want);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
        defer dx.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer dy.free();
        var dp = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane_.partBytes(m, w), 4));
        defer dp.free();
        try lane_.matmul(stream, dx.ptr, k, m, w, dy.ptr, dp.ptr);
        try stream.synchronize();
        const got = try check.download(gpu, dy);
        defer a.free(got);
        try sameBf16Ulp(m, n, got, want, 0);
        const want32 = try fx.bytes(try std.fmt.bufPrint(&name, "y32_{d}", .{m}));
        defer a.free(want32);
        var dy32 = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 4);
        defer dy32.free();
        try lane_.matmulF32(stream, dx.ptr, k, m, w, dy32.ptr, dp.ptr);
        try stream.synchronize();
        const got32 = try check.download(gpu, dy32);
        defer a.free(got32);
        try sameF32(m, n, got32, want32);
        check.pass("fp8b lane: {d} rows x [{d}, {d}], K slices {d}", .{ m, n, k, cuda.qmmf.planForMode(m, n, k, lane_.clusters, .fp8b).sk });
    }
}

/// No-argument smoke/parity case for a Spark: deterministic native E4M3 inputs and the same host oracle as fixtures.
fn defaultLane(gpu: Gpu) !void {
    const n = 64;
    const k = 128;
    const codes = try gpu.gpa.alloc(u8, n * k);
    defer gpu.gpa.free(codes);
    const x = try gpu.gpa.alloc(u16, 4 * k);
    defer gpu.gpa.free(x);
    const scale_inv = [_]f32{0.75};
    const packed_scales = try gpu.gpa.alloc(f32, fp8.padded(n) / fp8.group * (k / fp8.group));
    defer gpu.gpa.free(packed_scales);
    var seed: u32 = 0x8f31a;
    for (codes) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @truncate(0x28 + ((seed >> 24) % 0x50));
        if (v.* & 0x7f == 0x7f) v.* = 0x38;
    }
    for (x) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        const sample: i32 = @as(i32, @intCast((seed >> 16) % 129)) - 64;
        v.* = bf16Bits(@as(f32, @floatFromInt(sample)) / 64);
    }
    const packed_codes = try gpu.gpa.alloc(u8, fp8.padded(n) * k);
    defer gpu.gpa.free(packed_codes);
    fp8.packCodes(packed_codes, codes, n, k);
    try fp8.packBlockScales(packed_scales, &scale_inv, n, k, 0, 0, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales));
    defer ds.free();
    var lane_ = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane_.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const weight = fp8.blockWeight(dw.ptr, ds.ptr, n, k);
    for ([_]usize{ 1, 2, 4 }) |rows| {
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x[0 .. rows * k]));
        defer dx.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
        defer dy.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane_.partBytes(rows, weight), 4));
        defer part.free();
        try lane_.matmul(stream, dx.ptr, k, rows, weight, dy.ptr, part.ptr);
        try stream.synchronize();
        const got = try check.download(gpu, dy);
        defer gpu.gpa.free(got);
        const sums = try gpu.gpa.alloc(f32, rows * n);
        defer gpu.gpa.free(sums);
        reference(x, codes, &scale_inv, n, k, rows, cuda.qmmf.b16SplitK(n, k), sums);
        const expected = try gpu.gpa.alloc(u16, sums.len);
        defer gpu.gpa.free(expected);
        for (sums, expected) |v, *dst| dst.* = @truncate(@as(u32, @bitCast(fp8.roundBf16(v))) >> 16);
        try sameBf16Ulp(rows, n, got, std.mem.sliceAsBytes(expected), 0);
        check.pass("fp8b lane default: seeded E4M3 64x128 projection, rows={d}", .{rows});
    }
}

/// Benchmark the fixture's decode projection for every R in [1,16], reporting weight GB/s.
pub fn bench(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const n: usize = @intCast(try fx.int("n"));
    const k: usize = @intCast(try fx.int("k"));
    const row0: usize = @intCast(try fx.int("row0"));
    const col0: usize = @intCast(try fx.int("col0"));
    const full_n: usize = @intCast(try fx.int("full_n"));
    const full_k: usize = @intCast(try fx.int("full_k"));
    const raw_codes = try fx.bytes("codes");
    defer a.free(raw_codes);
    const raw_inv = try fx.bytes("scale_inv");
    defer a.free(raw_inv);
    const unaligned = std.mem.bytesAsSlice(f32, @as([]align(1) u8, raw_inv));
    const inv = try a.alloc(f32, unaligned.len);
    defer a.free(inv);
    for (unaligned, inv) |v, *o| o.* = v;
    const codes = try a.alloc(u8, fp8.padded(n) * k);
    defer a.free(codes);
    fp8.packCodes(codes, raw_codes, n, k);
    const scales = try a.alloc(f32, fp8.padded(n) / fp8.group * (k / fp8.group));
    defer a.free(scales);
    try fp8.packBlockScales(scales, inv, full_n, full_k, row0, col0, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    var lane_ = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane_.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const w = fp8.blockWeight(dw.ptr, ds.ptr, n, k);
    const bytes: f64 = @floatFromInt(n * k + (fp8.padded(n) / fp8.group) * (k / fp8.group) * @sizeOf(f32));
    for (1..17) |m| {
        const x = try a.alloc(u16, m * k);
        defer a.free(x);
        for (x, 0..) |*v, i| v.* = @truncate(0x3f00 + i % 128);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer dy.free();
        var dp = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane_.partBytes(m, w), 4));
        defer dp.free();
        for (0..4) |_| try lane_.matmul(stream, dx.ptr, k, m, w, dy.ptr, dp.ptr);
        try stream.synchronize();
        var start = try cuda.Event.init(gpu.d, true);
        defer start.deinit();
        var end = try cuda.Event.init(gpu.d, true);
        defer end.deinit();
        const reps = 20;
        try start.record(stream);
        for (0..reps) |_| try lane_.matmul(stream, dx.ptr, k, m, w, dy.ptr, dp.ptr);
        try end.record(stream);
        try end.synchronize();
        const seconds = @as(f64, try cuda.Event.elapsedMs(start, end)) / 1e3 / reps;
        std.debug.print("RESULT fp8b decode R={d:2} N={d} K={d}: {d:.2} ms, {d:.1} GB/s\n", .{ m, n, k, seconds * 1e3, bytes / seconds / 1e9 });
    }
}

fn sameBf16Ulp(rows: usize, n: usize, got: []const u8, want: []const u8, max_ulp: u16) !void {
    if (got.len != rows * n * 2 or want.len != got.len) return error.BadFixture;
    const ga = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got));
    const wa = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, want));
    for (ga, wa, 0..) |g, w, i| {
        const gd: i32 = @intCast(if (g & 0x8000 != 0) ~g else g | 0x8000);
        const wd: i32 = @intCast(if (w & 0x8000 != 0) ~w else w | 0x8000);
        const diff = @abs(gd - wd);
        if (diff > max_ulp) {
            std.debug.print("fp8b mismatch at {d}: got {x:0>4}, want {x:0>4}, ulp {d}\n", .{ i, g, w, diff });
            return error.Bf16Mismatch;
        }
    }
}

fn sameF32(rows: usize, n: usize, got: []const u8, want: []const u8) !void {
    if (got.len != rows * n * 4 or want.len != got.len) return error.BadFixture;
    const ga = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    const wa = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, want));
    for (ga, wa, 0..) |g, w, i| if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(w))) {
        std.debug.print("fp8b fp32 mismatch at {d}: got {d}, want {d}\n", .{ i, g, w });
        return error.F32Mismatch;
    };
}

fn bf16(x: u16) f32 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

fn bf16Bits(x: f32) u16 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn reference(x: []const u16, codes: []const u8, inv: []const f32, n: usize, k: usize, m: usize, sk: usize, out: []f32) void {
    const kg = k / fp8.group;
    const groups_per_slice = kg / sk;
    for (0..m) |r| for (0..n) |row| {
        var partials: [8]f32 = @splat(0);
        for (0..sk) |slice| {
            var partial: f32 = 0;
            const g0 = slice * groups_per_slice;
            // The CUDA lane chains m16n8k16 MMA accumulators through each 64-wide stage.
            for (g0..g0 + groups_per_slice) |group_idx| for (0..fp8.group / 16) |kt| {
                for (0..16) |i| {
                    const c = group_idx * fp8.group + kt * 16 + i;
                    const scale = inv[(row / fp8.block) * (k / fp8.block) + c / fp8.block];
                    const w = fp8.fp8bWeight(codes[row * k + c], scale);
                    partial = @mulAdd(f32, bf16(x[r * k + c]), w, partial);
                }
            };
            partials[slice] = partial;
        }
        // reduce_kernel, cluster, and fused paths all add slice partials in increasing order.
        var sum = partials[0];
        for (1..sk) |slice| sum += partials[slice];
        out[r * n + row] = sum;
    };
}

test "fp8b scale tiles retain 128x128 scales across partial TP rows and columns" {
    // Rank one of a 1152-row kv_a tensor starts halfway through a 128-row source block.
    const rows = 1152;
    const cols = 256;
    var source: [9 * 2]f32 = undefined;
    for (&source, 0..) |*s, i| s.* = @floatFromInt(i + 1);
    var packed_scales: [10 * 4]f32 = undefined; // padded 640 / 64 by 256 / 64
    try fp8.packBlockScales(&packed_scales, &source, rows, cols, 576, 0, 576, cols);
    try std.testing.expectEqual(@as(f32, 9), packed_scales[0]); // global row 576 -> scale block 4
    try std.testing.expectEqual(@as(f32, 9), packed_scales[1]); // next 64 K inputs still share the first scale
    try std.testing.expectEqual(@as(f32, 11), packed_scales[4]); // global row 640 -> scale block 5
    try std.testing.expectEqual(@as(f32, 18), packed_scales[8 * 4 + 3]); // global row 1088, K block 1

    // TP column cuts also preserve the original K-block origin.
    var col_slice: [10 * 2]f32 = undefined;
    try fp8.packBlockScales(&col_slice, &source, rows, cols, 0, 64, 576, 128);
    try std.testing.expectEqual(@as(f32, 1), col_slice[0]);
    try std.testing.expectEqual(@as(f32, 2), col_slice[1]);
}

test "fp8b kv_a tail rows 512 through 575 use native scale row 4" {
    const n = 576;
    const k = 128;
    const inv: [5]f32 = .{ 0.5, 0.75, 1, 1.25, 1.5 };
    var packed_scales: [20]f32 = undefined;
    try fp8.packBlockScales(&packed_scales, &inv, n, k, 0, 0, n, k);
    try std.testing.expectEqual(@as(f32, 1.5), packed_scales[16]);

    const x: [k]u16 = @splat(0x3f80);
    const codes: [n * k]u8 = @splat(0x38);
    var output: [n]f32 = undefined;
    reference(&x, &codes, &inv, n, k, 1, 1, &output);
    const expected = fp8.roundBf16(1.0 * inv[4]) * @as(f32, @floatFromInt(k));
    for (512..576) |row| try std.testing.expectEqual(expected, output[row]);
}

test "fp8b scale pack accepts the 32-row tail and refuses invalid layouts" {
    const rows = 2784;
    const cols = 64;
    var source: [22]f32 = undefined;
    for (&source, 0..) |*s, i| s.* = @floatFromInt(i + 1);
    var packed_scales: [44]f32 = undefined;
    try fp8.packBlockScales(&packed_scales, &source, rows, cols, 0, 0, rows, cols);
    try std.testing.expectEqual(@as(f32, 22), packed_scales[43]); // rows 2752..2783, the real 32-row tail
    try std.testing.expectError(error.InvalidShape, fp8.packBlockScales(&packed_scales, &source, rows, cols, 0, 0, rows + 64, cols));
}

fn pythonB16SplitK(n: usize, k: usize) u32 {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return @intCast(sk);
}

test "fp8b split K matches Python b16_split_k for every K1 shape" {
    const shapes = [_][2]usize{
        .{ 2048, 6144 },  .{ 576, 6144 },  .{ 8192, 2048 }, .{ 6144, 8192 },
        .{ 4096, 2048 },  .{ 128, 6144 },  .{ 2048, 6144 }, .{ 6144, 1024 },
        .{ 12288, 6144 }, .{ 6144, 6144 }, .{ 2048, 1536 }, .{ 576, 512 },
    };
    for (shapes) |shape| try std.testing.expectEqual(pythonB16SplitK(shape[0], shape[1]), cuda.qmmf.b16SplitK(shape[0], shape[1]));
}

test "fp8b host reference chains K16 MMA steps after bf16 weight dequantization" {
    const n = 130;
    const k = 256;
    const m = 2;
    const x = try std.testing.allocator.alloc(u16, m * k);
    defer std.testing.allocator.free(x);
    const codes = try std.testing.allocator.alloc(u8, n * k);
    defer std.testing.allocator.free(codes);
    const inv = try std.testing.allocator.alloc(f32, 2 * 2);
    defer std.testing.allocator.free(inv);
    for (x, 0..) |*v, i| v.* = @truncate(0x3f00 + i % 128);
    for (codes, 0..) |*v, i| {
        v.* = @truncate(i * 37 + 0x28);
        if (v.* & 0x7f == 0x7f) v.* = 0x38;
    }
    for (inv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) / 16;
    const got = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(got);
    reference(x, codes, inv, n, k, m, 1, got);
    for (0..m) |r| for (0..n) |row| {
        var want: f32 = 0;
        var precise: f64 = 0;
        for (0..k / 16) |kt| {
            for (0..16) |i| {
                const c = kt * 16 + i;
                const s = inv[(row / 128) * 2 + c / 128];
                const w = fp8.roundBf16(@as(f32, @floatCast(core.quant.fp8.e4m3(codes[row * k + c]))) * s);
                const xv = bf16(x[r * k + c]);
                want = @mulAdd(f32, xv, w, want);
                precise += @as(f64, xv) * @as(f64, w);
            }
        }
        try std.testing.expectEqual(want, got[r * n + row]);
        try std.testing.expect(@abs(@as(f64, got[r * n + row]) - precise) <= @max(1e-4, @abs(precise) * 1e-6));
    };
    try std.testing.expectEqual(fp8.roundBf16(1.00390625), fp8.fp8bWeight(0x38, 1.00390625));
}

test "fp8b split K sums four partials left to right" {
    const n = 1;
    const k = 1024;
    var x: [k]u16 = undefined;
    for (0..256) |i| x[i] = 0x5c00; // large positive partial
    for (256..512) |i| x[i] = 0xdc00; // cancellation with slice zero
    for (512..768) |i| x[i] = 0x3d00; // partial 8
    for (768..1024) |i| x[i] = 0x3c80; // partial 4
    const codes: [n * k]u8 = @splat(0x38);
    const inv: [k / 128]f32 = @splat(1);
    var got: [n]f32 = undefined;
    reference(&x, &codes, &inv, n, k, 1, 4, &got);
    try std.testing.expectEqual(@as(f32, 12), got[0]);
    const p0 = bf16(0x5c00) * 256;
    const p1 = bf16(0xdc00) * 256;
    const p2 = bf16(0x3d00) * 256;
    const p3 = bf16(0x3c80) * 256;
    const changed_order = (p0 + p2) + p1 + p3;
    try std.testing.expect(changed_order != got[0]);
    var unsplit: [n]f32 = undefined;
    reference(&x, &codes, &inv, n, k, 1, 1, &unsplit);
    try std.testing.expect(@abs(unsplit[0] - got[0]) <= 16);
}

test {
    std.testing.refAllDecls(@This());
}

test "fp8b launch plan uses Python split K, including the 576x512 boundary" {
    try std.testing.expectEqual(@as(u32, 2), cuda.qmmf.planForMode(1, 576, 512, false, .fp8b).sk);
    try std.testing.expectEqual(cuda.qmmf.splitK(576, 512), cuda.qmmf.plan(1, 576, 512, false).sk);
}

test "fp8b decode bench plan reports the R=1..16 window and native byte count" {
    const n = 8192;
    const k = 2048;
    const weight_bytes: usize = n * k + (fp8.padded(n) / fp8.group) * (k / fp8.group) * @sizeOf(f32);
    try std.testing.expectEqual(n * k + 128 * 32 * @sizeOf(f32), weight_bytes);
    for (1..17) |rows| try std.testing.expect(rows >= 1 and rows <= 16);
}
