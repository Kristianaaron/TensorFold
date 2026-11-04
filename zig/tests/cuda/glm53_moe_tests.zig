// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 router, SwiGLU and split-K low-rank GPU parity (WP8).
const std = @import("std");
const cuda = @import("cuda");
const moe = @import("moe.zig");
const mlp = @import("mlp.zig");

test "GLM-5.3 MoE declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var glue_module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_module.unload();
    var local_module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer local_module.unload();
    const router: moe.RouterKernels = .{
        .router_part = try glue_module.function("tf_glm_router_part"),
        .router_tile = try glue_module.function("tf_glm_router_tile"),
        .topk = try glue_module.function("tf_glm_topk"),
        .slice_sum = try glue_module.function("tf_glm_slice_sum"),
    };
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try routerInvariant(gpu, stream, router);
    const swiglu = try local_module.function("tf_glm53_swiglu_plain");
    try swigluCase(gpu, stream, swiglu);
    const kernels = try moe.Kernels.load(local_module);
    try lowRankCase(gpu, stream, kernels.lowrank);
    try moePostCase(gpu, stream, kernels);
    try moePostShortRingCase(gpu, stream, kernels);
}

pub fn runLowRank(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels = try moe.Kernels.load(module);
    kernels.decode_fast = true;
    try lowRankDecodeCases(gpu, stream, kernels);
}

fn lowRankDecodeCases(gpu: anytype, stream: cuda.Stream, kernels: moe.Kernels) !void {
    const ranks = [_]usize{ 16, 32, 64 };
    const row_counts = [_]usize{ 1, 2, 8, 16, 64 };
    var seed: u32 = 0x53d3c0de;
    for (ranks) |rank| for (row_counts) |rows| {
        const x = try gpu.gpa.alloc(u16, rows * moe.hidden);
        defer gpu.gpa.free(x);
        const a = try gpu.gpa.alloc(f32, moe.hidden * rank);
        defer gpu.gpa.free(a);
        for (x) |*v| {
            seed = seed *% 1664525 +% 1013904223;
            const sample: i32 = @as(i32, @intCast((seed >> 16) % 129)) - 64;
            v.* = bf16Bits(@as(f32, @floatFromInt(sample)) / 64);
        }
        for (a) |*v| {
            seed = seed *% 1664525 +% 1013904223;
            const sample: i32 = @as(i32, @intCast((seed >> 16) % 257)) - 128;
            v.* = @as(f32, @floatFromInt(sample)) / 2048;
        }
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var da = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(a));
        defer da.free();
        const split_count: usize = if (rows <= 16) moe.decode_splits else 8;
        var fast_mid = try cuda.DeviceBuffer.alloc(gpu.d, split_count * rows * rank * 4);
        defer fast_mid.free();
        var old_mid = try cuda.DeviceBuffer.alloc(gpu.d, 8 * rows * rank * 4);
        defer old_mid.free();
        var fast_k = kernels;
        fast_k.decode_fast = true;
        try moe.lowrankPre(fast_k, stream, dx.ptr, moe.hidden, rows, .{ .a = da.ptr, .b = 0, .mid = fast_mid.ptr, .rank_local = rank });
        try stream.synchronize();
        const fast_bytes = try download(gpu, fast_mid);
        defer gpu.gpa.free(fast_bytes);
        const fast = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, fast_bytes));
        var max_abs: f64 = 0;
        var max_rel: f64 = 0;
        var within_tolerance = true;
        for (0..rows) |row| for (0..rank) |rr| {
            var expected: f64 = 0;
            for (0..moe.hidden) |k| expected += @as(f64, @floatCast(bf16Value(x[row * moe.hidden + k]))) * @as(f64, a[k * rank + rr]);
            const actual = @as(f64, fast[row * rank + rr]);
            const abs_error = @abs(actual - expected);
            const rel_error = abs_error / @max(@abs(expected), 1e-12);
            max_abs = @max(max_abs, abs_error);
            max_rel = @max(max_rel, rel_error);
            if (abs_error > 2e-4 + 2e-4 * @abs(expected)) within_tolerance = false;
        };
        std.debug.print("RESULT glm53-lowrank rows={d} rank={d} max_abs={e:.6} max_rel={e:.6}\n", .{ rows, rank, max_abs, max_rel });
        try expect(within_tolerance, "decode low-rank fp32 tolerance abs<=2e-4+2e-4*abs(ref) rows={d} rank={d}", .{ rows, rank });
        try moe.lowrankPre(fast_k, stream, dx.ptr, moe.hidden, rows, .{ .a = da.ptr, .b = 0, .mid = fast_mid.ptr, .rank_local = rank });
        try stream.synchronize();
        const repeat = try download(gpu, fast_mid);
        defer gpu.gpa.free(repeat);
        try expect(std.mem.eql(u8, fast_bytes, repeat), "decode low-rank repeat bits rows={d} rank={d}", .{ rows, rank });
        if (rows == 8) {
            var alone_mid = try cuda.DeviceBuffer.alloc(gpu.d, moe.decode_splits * rank * 4);
            defer alone_mid.free();
            var alone_x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x[3 * moe.hidden ..][0..moe.hidden]));
            defer alone_x.free();
            try moe.lowrankPre(fast_k, stream, alone_x.ptr, moe.hidden, 1, .{ .a = da.ptr, .b = 0, .mid = alone_mid.ptr, .rank_local = rank });
            try stream.synchronize();
            const alone_bytes = try download(gpu, alone_mid);
            defer gpu.gpa.free(alone_bytes);
            const alone = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, alone_bytes));
            for (0..rank) |rr| try expect(@as(u32, @bitCast(alone[rr])) == @as(u32, @bitCast(fast[3 * rank + rr])), "row-alone bits rank={d}", .{rank});
        }
        var old_k = kernels;
        old_k.decode_fast = false;
        try moe.lowrankPre(old_k, stream, dx.ptr, moe.hidden, rows, .{ .a = da.ptr, .b = 0, .mid = old_mid.ptr, .rank_local = rank });
        try stream.synchronize();
        const old_bytes = try download(gpu, old_mid);
        defer gpu.gpa.free(old_bytes);
        const old = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, old_bytes));
        for (0..rows * rank) |i| try expect(@abs(old[i] - fast[i]) <= 3e-4, "decode vs old kernels rows={d} rank={d}", .{ rows, rank });
        pass("glm53-lowrank: rows={d} rank={d} reference, old path, repeat and row invariance", .{ rows, rank });
    };
    const rank = moe.lowrank;
    const routed = try gpu.gpa.alloc(f32, moe.hidden);
    defer gpu.gpa.free(routed);
    const shared = try gpu.gpa.alloc(f32, moe.hidden);
    defer gpu.gpa.free(shared);
    const b = try gpu.gpa.alloc(f32, rank * moe.hidden);
    defer gpu.gpa.free(b);
    const mid = try gpu.gpa.alloc(f32, rank);
    defer gpu.gpa.free(mid);
    for (routed) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((seed >> 16) % 129)) - 64)) / 128;
    }
    for (shared) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((seed >> 16) % 129)) - 64)) / 128;
    }
    for (b) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((seed >> 16) % 129)) - 64)) / 4096;
    }
    for (mid) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((seed >> 16) % 129)) - 64)) / 32;
    }
    var dr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(routed));
    defer dr.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(shared));
    defer ds.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(b));
    defer db.free();
    var dm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mid));
    defer dm.free();
    var fast_out = try cuda.DeviceBuffer.alloc(gpu.d, moe.hidden * 2);
    defer fast_out.free();
    var old_out = try cuda.DeviceBuffer.alloc(gpu.d, moe.hidden * 2);
    defer old_out.free();
    const combine_weights: moe.LowRankWeights = .{ .a = 0, .b = db.ptr, .mid = dm.ptr, .rank_local = rank };
    var fast = kernels;
    fast.decode_fast = true;
    try moe.blockPost(fast, stream, dr.ptr, ds.ptr, combine_weights, 1, fast_out.ptr);
    var old = kernels;
    old.decode_fast = false;
    try moe.blockPost(old, stream, dr.ptr, ds.ptr, combine_weights, 1, old_out.ptr);
    try stream.synchronize();
    const fast_result = try download(gpu, fast_out);
    defer gpu.gpa.free(fast_result);
    const old_result = try download(gpu, old_out);
    defer gpu.gpa.free(old_result);
    try expect(std.mem.eql(u8, fast_result, old_result), "decode combine BF16 matches legacy bits", .{});
    pass("glm53-lowrank: decode combine uses a 32-thread output tile and preserves BF16 bits", .{});
}

fn routerInvariant(gpu: anytype, stream: cuda.Stream, kernels: moe.RouterKernels) !void {
    const rows_max = 64;
    const x = try gpu.gpa.alloc(u16, rows_max * moe.hidden);
    defer gpu.gpa.free(x);
    const weights = try gpu.gpa.alloc(u16, moe.experts * moe.hidden);
    defer gpu.gpa.free(weights);
    const bias = try gpu.gpa.alloc(f32, moe.experts);
    defer gpu.gpa.free(bias);
    for (x, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32);
    for (weights, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 17)) - 8)) / 64);
    for (bias, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 100;
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weights));
    defer dw.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bias));
    defer db.free();
    var first_picks: [moe.top_k]i32 = undefined;
    var first_weights: [moe.top_k]f32 = undefined;
    for ([_]usize{ 1, 3, 16, 64 }) |rows| {
        var logits = try cuda.DeviceBuffer.alloc(gpu.d, rows * moe.experts * 4);
        defer logits.free();
        var parts = try cuda.DeviceBuffer.alloc(gpu.d, rows * moe.experts * 8 * 4);
        defer parts.free();
        var picks = try cuda.DeviceBuffer.alloc(gpu.d, rows * moe.slots * 4);
        defer picks.free();
        var probs = try cuda.DeviceBuffer.alloc(gpu.d, rows * moe.slots * 4);
        defer probs.free();
        try moe.route(kernels, stream, dx.ptr, moe.hidden, dw.ptr, db.ptr, logits.ptr, parts.ptr, picks.ptr, probs.ptr, rows);
        try stream.synchronize();
        const gotp = try download(gpu, picks);
        defer gpu.gpa.free(gotp);
        const gotw = try download(gpu, probs);
        defer gpu.gpa.free(gotw);
        const gotlog = try download(gpu, logits);
        defer gpu.gpa.free(gotlog);
        const pi = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, gotp));
        const pw = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, gotw));
        const pl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, gotlog));
        if (rows == 1) {
            for (&first_picks, pi[0..moe.top_k]) |*dst, src| dst.* = src;
            for (&first_weights, pw[0..moe.top_k]) |*dst, src| dst.* = src;
        } else {
            for (first_picks, pi[0..moe.top_k]) |a, b| try expect(a == b, "router row-invariance picks rows={d}", .{rows});
            for (first_weights, pw[0..moe.top_k]) |a, b| try expect(@as(u32, @bitCast(a)) == @as(u32, @bitCast(b)), "router row-invariance weights rows={d}", .{rows});
        }
        for (0..rows) |r| {
            var row_logits: [moe.experts]f32 = undefined;
            for (&row_logits, pl[r * moe.experts ..][0..moe.experts]) |*dst, src| dst.* = src;
            const expected = try moe.routeRef(&row_logits, bias, 2.5);
            for (0..moe.top_k) |k| {
                try expect(pi[r * moe.slots + k] == expected.pick[k], "router host pick row={d} slot={d}", .{ r, k });
                try expect(@abs(pw[r * moe.slots + k] - expected.weights[k]) < 2e-6, "router host weight row={d} slot={d}", .{ r, k });
            }
        }
        try expect(pi[moe.top_k] == moe.experts and pw[moe.top_k] == 1, "router shared slot rows={d}", .{rows});
        pass("glm53-moe: fixed router tile rows={d} preserves first-row picks and weights", .{rows});
    }
}

fn swigluCase(gpu: anytype, stream: cuda.Stream, kernel: cuda.Function) !void {
    const rows = 8;
    const width = 64;
    const gu = try gpu.gpa.alloc(u16, rows * width * 2);
    defer gpu.gpa.free(gu);
    var seed: u32 = 0x53a17;
    for (gu) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        const sample: i32 = @as(i32, @intCast((seed >> 16) % 1025)) - 512;
        v.* = bf16Bits(@as(f32, @floatFromInt(sample)) / 64);
    }
    var dgu = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(gu));
    defer dgu.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, rows * width * 2);
    defer dout.free();
    try mlp.swigluPlain(kernel, stream, dgu.ptr, dout.ptr, rows, width);
    try stream.synchronize();
    const got = try download(gpu, dout);
    defer gpu.gpa.free(got);
    const actual = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got));
    // The GPU uses ex2.approx + div.full, which a host libm cannot reproduce bitwise. Both paths
    // round SiLU and the product to BF16; one adjacent BF16 value is the bounded exp approximation delta.
    var mismatches: usize = 0;
    var max_ulp: u16 = 0;
    for (0..rows * width) |i| {
        const row = i / width;
        const col = i % width;
        // qmmf writes each row as [gate | up], matching the checkpoint's
        // stacked gate_proj then up_proj matrix.
        const want = mlp.swigluRef(bf16Value(gu[row * 2 * width + col]), bf16Value(gu[row * 2 * width + width + col]));
        const want_bits = bf16Bits(want);
        const ulp = bf16Ulp(actual[i], want_bits);
        max_ulp = @max(max_ulp, ulp);
        if (ulp > 1) {
            if (mismatches < 8) std.debug.print("SwiGLU mismatch item={d}: GPU={d} (0x{x:0>4}), host={d} (0x{x:0>4}), ulp={d}\n", .{ i, bf16Value(actual[i]), actual[i], bf16Value(want_bits), want_bits, ulp });
            mismatches += 1;
        }
    }
    try expect(mismatches == 0, "SwiGLU exceeded 1 bf16 ULP at {d} items (max {d})", .{ mismatches, max_ulp });
    for (0..rows) |r| {
        var one = try cuda.DeviceBuffer.alloc(gpu.d, width * 2);
        defer one.free();
        try mlp.swigluPlain(kernel, stream, dgu.ptr + r * width * 4, one.ptr, 1, width);
        try stream.synchronize();
        const alone = try download(gpu, one);
        defer gpu.gpa.free(alone);
        try expect(std.mem.eql(u8, alone, got[r * width * 2 ..][0 .. width * 2]), "SwiGLU row-alone vs batch row={d}", .{r});
    }
    pass("glm53-moe: plain bf16 SwiGLU within 1 bf16 ULP over {d} seeded values", .{rows * width});
}

fn lowRankCase(gpu: anytype, stream: cuda.Stream, kernels: moe.LowRankKernels) !void {
    const rows = 1;
    const d = moe.hidden;
    const rank = moe.lowrank / 2;
    const xf = try gpu.gpa.alloc(f32, d);
    defer gpu.gpa.free(xf);
    const xb = try gpu.gpa.alloc(u16, d);
    defer gpu.gpa.free(xb);
    const a = try gpu.gpa.alloc(f32, d * rank);
    defer gpu.gpa.free(a);
    const b = try gpu.gpa.alloc(f32, rank * d);
    defer gpu.gpa.free(b);
    for (xf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 32;
    for (0..d) |i| xb[i] = bf16Bits(xf[i]);
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 128;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 256;
    for (xf, 0..) |*v, i| v.* = bf16Value(xb[i]);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
    defer dx.free();
    var da = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(a));
    defer da.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(b));
    defer db.free();
    var acc = try cuda.DeviceBuffer.alloc(gpu.d, d * 4);
    defer acc.free();
    var mid = try cuda.DeviceBuffer.alloc(gpu.d, rank * 8 * 4);
    defer mid.free();
    try acc.fill8(0, stream.handle);
    try moe.lowRankAdd(kernels, stream, dx.ptr, da.ptr, db.ptr, acc.ptr, mid.ptr, rows, rank, d, rank, d);
    try stream.synchronize();
    const got = try download(gpu, acc);
    defer gpu.gpa.free(got);
    const actual = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    const expected = try moe.lowRankRef(gpu.gpa, xf, a, b, rows, d, rank);
    defer gpu.gpa.free(expected);
    for (actual, expected) |xv, yv| try expect(@abs(xv - yv) <= @max(2e-4, @abs(yv) * 2e-5), "low-rank output", .{});
    pass("glm53-moe: low-rank split-K 8 matches host reference for rank width {d}", .{rank});
}

fn moePostCase(gpu: anytype, stream: cuda.Stream, kernels: moe.Kernels) !void {
    const rows: usize = 2;
    const rank: usize = 16;
    const d = moe.hidden;
    const y = try gpu.gpa.alloc(f32, rows * moe.top_k * d);
    defer gpu.gpa.free(y);
    const weights = try gpu.gpa.alloc(f32, rows * moe.top_k);
    defer gpu.gpa.free(weights);
    const routed = try gpu.gpa.alloc(f32, rows * d);
    defer gpu.gpa.free(routed);
    const shared = try gpu.gpa.alloc(f32, rows * d);
    defer gpu.gpa.free(shared);
    const b = try gpu.gpa.alloc(f32, rank * d);
    defer gpu.gpa.free(b);
    const mid = try gpu.gpa.alloc(f32, rows * rank);
    defer gpu.gpa.free(mid);
    var seed: u32 = 0x6d6f6570;
    for (y, 0..) |*v, i| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((seed >> 16) % 127)) - 63)) / 128;
        if (i % 997 == 1) v.* = @bitCast(@as(u32, 1));
        if (i % 991 == 2) v.* = -0.0;
        if (i % 983 == 3) v.* = std.math.inf(f32);
        if (i % 977 == 4) v.* = -std.math.inf(f32);
        if (i % 971 == 5) v.* = std.math.nan(f32);
    }
    for (weights, 0..) |*v, i| v.* = if (i % 3 == 0) -0.0 else @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 16;
    for (routed, 0..) |*v, i| {
        const row = i / d;
        const col = i % d;
        var acc: f32 = 0;
        for (0..moe.top_k) |q| acc = @mulAdd(f32, weights[row * moe.top_k + q], y[(row * moe.top_k + q) * d + col], acc);
        v.* = acc;
    }
    for (shared, 0..) |*v, i| v.* = if (i % 613 == 0) -0.0 else @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 2048;
    for (mid, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16;
    var dy = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(y));
    defer dy.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weights));
    defer dw.free();
    var dr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(routed));
    defer dr.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(shared));
    defer ds.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(b));
    defer db.free();
    var dm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mid));
    defer dm.free();
    var old_out = try cuda.DeviceBuffer.alloc(gpu.d, rows * d * 2);
    defer old_out.free();
    var fused_out = try cuda.DeviceBuffer.alloc(gpu.d, rows * d * 2);
    defer fused_out.free();
    const lr: moe.LowRankWeights = .{ .a = 0, .b = db.ptr, .mid = dm.ptr, .rank_local = rank };
    try moe.blockPost(kernels, stream, dr.ptr, ds.ptr, lr, rows, old_out.ptr);
    try moe.postPrompt(kernels, stream, dy.ptr, dw.ptr, ds.ptr, lr, rows, fused_out.ptr);
    try stream.synchronize();
    const old_bytes = try download(gpu, old_out);
    defer gpu.gpa.free(old_bytes);
    const fused_bytes = try download(gpu, fused_out);
    defer gpu.gpa.free(fused_bytes);
    try expect(std.mem.eql(u8, old_bytes, fused_bytes), "moePost matches combinePrompt + blockPost byte for byte", .{});
    pass("glm53-moe: fused prompt post is byte-equal to the legacy combine and blockPost path", .{});
}

fn moePostShortRingCase(gpu: anytype, stream: cuda.Stream, kernels: moe.Kernels) !void {
    const piece_rows: usize = 512;
    const rows: usize = 1200;
    const pieces = [_]usize{ piece_rows, piece_rows, rows - 2 * piece_rows };
    try std.testing.expectEqual(rows, pieces[0] + pieces[1] + pieces[2]);
    for (pieces) |short_rows| {
        const local_bytes = short_rows * moe.hidden * 2;
        const rank0_host = try gpu.gpa.alloc(u16, short_rows * moe.hidden);
        defer gpu.gpa.free(rank0_host);
        const rank1_host = try gpu.gpa.alloc(u16, short_rows * moe.hidden);
        defer gpu.gpa.free(rank1_host);
        for (rank0_host, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 32);
        for (rank1_host, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 32);
        var rank0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rank0_host));
        defer rank0.free();
        var rank1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rank1_host));
        defer rank1.free();
        var separate = try cuda.DeviceBuffer.alloc(gpu.d, 2 * local_bytes);
        defer separate.free();
        try separate.copyFrom(0, rank0.ptr, local_bytes, stream.handle);
        try separate.copyFrom(local_bytes, rank1.ptr, local_bytes, stream.handle);
        var in_place = try cuda.DeviceBuffer.alloc(gpu.d, 2 * piece_rows * moe.hidden * 2);
        defer in_place.free();
        try in_place.fill8(0, stream.handle);
        // NCCL in-place all-gather places rank r at r * n, including the 176-row tail.
        try in_place.copyFrom(0, rank0.ptr, local_bytes, stream.handle);
        try in_place.copyFrom(local_bytes, rank1.ptr, local_bytes, stream.handle);
        const residual_host = try gpu.gpa.alloc(u16, short_rows * moe.hidden);
        defer gpu.gpa.free(residual_host);
        for (residual_host, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 64);
        var separate_out = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(residual_host));
        defer separate_out.free();
        var in_place_out = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(residual_host));
        defer in_place_out.free();
        try moe.residualAddBf16(kernels.residual_add_bf16, stream, separate_out.ptr, separate.ptr, short_rows);
        try moe.residualAddBf16(kernels.residual_add_bf16, stream, in_place_out.ptr, in_place.ptr, short_rows);
        try stream.synchronize();
        const separate_bytes = try download(gpu, separate_out);
        defer gpu.gpa.free(separate_bytes);
        const in_place_bytes = try download(gpu, in_place_out);
        defer gpu.gpa.free(in_place_bytes);
        try expect(std.mem.eql(u8, separate_bytes, in_place_bytes), "short in-place ring matches separate gather bytes", .{});
    }
    pass("glm53-moe: 1200-row ring (512 + 512 + 176) matches separate gather bytes", .{});
}

fn bf16Bits(x: f32) u16 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn bf16Value(x: u16) f32 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

fn bf16Ulp(a: u16, b: u16) u16 {
    const ao: i32 = @intCast(if (a & 0x8000 != 0) ~a else a | 0x8000);
    const bo: i32 = @intCast(if (b & 0x8000 != 0) ~b else b | 0x8000);
    return @intCast(@abs(ao - bo));
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) error{TestFailed}!void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}

test "MoE partial uses fp32 routed plus shared plus low-rank order" {
    try std.testing.expectEqual(@as(f32, 1.0e20), moe.partialRef(1.0e20, 4, -4));
    try std.testing.expectEqual(@as(f32, 0), moe.partialRef(1.0e20, -1.0e20, 3));
}
