// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 MoE combine and post-attention residual/RMSNorm fusion parity.
const std = @import("std");
const cuda = @import("cuda");

test "GLM-5.3 fusion declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

test "fused epilogue host reference keeps both BF16 residual boundaries" {
    const residual = try residualRmsnormRef(std.testing.allocator, &.{ 0x3f80, 0xbf80 }, &.{ 0.5, 1.0, -0.5, 0.5 }, &.{ 0x3f80, 0x3f80 }, 1, 2, 1e-5);
    defer std.testing.allocator.free(residual.x);
    defer std.testing.allocator.free(residual.normed);
    try std.testing.expectEqualSlices(u16, &.{ bf16Bits(1), bf16Bits(0.5) }, residual.x);
    try std.testing.expectEqual(@as(usize, 2), residual.normed.len);
}

fn residualRmsnormRef(a: std.mem.Allocator, x: []const u16, parts: []const f32, w: []const u16, rows: usize, d: usize, eps: f32) !struct { x: []u16, normed: []u16 } {
    if (x.len != rows * d or parts.len != rows * d * 2 or w.len != d) return error.BadReferenceShape;
    const residual = try a.alloc(u16, x.len);
    errdefer a.free(residual);
    const normed = try a.alloc(u16, x.len);
    errdefer a.free(normed);
    for (0..rows) |r| {
        var squares: f32 = 0;
        for (0..d) |c| {
            const i = r * d + c;
            const rank_sum = parts[i] + parts[rows * d + i];
            const sum_bf = bf16Bits(rank_sum);
            const old = bf16Value(x[i]);
            const updated = bf16Bits(old + bf16Value(sum_bf));
            residual[i] = updated;
            const value = bf16Value(updated);
            squares = @mulAdd(f32, value, value, squares);
        }
        const rinv = 1.0 / @sqrt(squares / @as(f32, @floatFromInt(d)) + eps);
        for (0..d) |c| {
            const i = r * d + c;
            const normalized = bf16Value(bf16Bits(bf16Value(residual[i]) * rinv));
            normed[i] = bf16Bits(bf16Value(w[c]) * normalized);
        }
    }
    return .{ .x = residual, .normed = normed };
}

pub fn run(gpu: anytype) !void {
    var base = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer base.unload();
    var fused = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer fused.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try moeCombine(gpu, stream, fused);
    try attentionEpilogue(gpu, stream, base, fused);
    try moeAddNormEpilogue(gpu, stream, base, fused);
    try qaShardNorm(gpu, stream, base, fused);
}

fn qaShardNorm(gpu: anytype, stream: cuda.Stream, base: cuda.Module, fused: cuda.Module) !void {
    const full_norm = try base.function("tf_glm_rmsnorm_2048");
    const halves_norm = try fused.function("tf_glm53_rmsnorm_2048_halves");
    var prng = std.Random.DefaultPrng.init(0x5a4d_2048);
    const random = prng.random();
    for ([_]usize{ 1, 2, 3, 4, 8, 16, 512 }) |rows| {
        const full = try gpu.gpa.alloc(u16, rows * 2048);
        defer gpu.gpa.free(full);
        const x0 = try gpu.gpa.alloc(u16, rows * 1024);
        defer gpu.gpa.free(x0);
        const x1 = try gpu.gpa.alloc(u16, rows * 1024);
        defer gpu.gpa.free(x1);
        const weight = try gpu.gpa.alloc(u16, 2048);
        defer gpu.gpa.free(weight);
        for (0..rows) |r| {
            for (0..2048) |c| {
                const i = r * 2048 + c;
                const value: f32 = if (rows > 1 and r == 0) 0 else if (rows > 1 and r == 1) signed(i, 31) * 8192 else signed(random.int(usize), 127) / 32;
                full[i] = bf16Bits(value);
                if (c < 1024) x0[r * 1024 + c] = full[i] else x1[r * 1024 + c - 1024] = full[i];
            }
        }
        for (weight) |*v| v.* = bf16Bits(0.5 + @as(f32, @floatFromInt(random.intRangeAtMost(i16, -256, 256))) / 512);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(full));
        defer dx.free();
        var d0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x0));
        defer d0.free();
        var d1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x1));
        defer d1.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weight));
        defer dw.free();
        var out_full = try cuda.DeviceBuffer.alloc(gpu.d, rows * 2048 * 2);
        defer out_full.free();
        var out_halves = try cuda.DeviceBuffer.alloc(gpu.d, rows * 2048 * 2);
        defer out_halves.free();
        var full_args: cuda.Args = .{};
        inline for (.{ dx.ptr, @as(i64, 2048), dw.ptr, out_full.ptr, @as(i64, 2048), @as(u64, 0), @as(f32, 1e-5), @as(c_int, 0) }) |v| full_args.add(v);
        try cuda.launch.launch(full_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &full_args);
        var halves_args: cuda.Args = .{};
        inline for (.{ d0.ptr, d1.ptr, @as(i64, 1024), dw.ptr, out_halves.ptr, @as(i64, 2048), @as(f32, 1e-5) }) |v| halves_args.add(v);
        try cuda.launch.launch(halves_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &halves_args);
        try stream.synchronize();
        const expected = try gpu.gpa.alloc(u8, out_full.len);
        defer gpu.gpa.free(expected);
        const actual = try gpu.gpa.alloc(u8, out_halves.len);
        defer gpu.gpa.free(actual);
        try out_full.download(0, expected);
        try out_halves.download(0, actual);
        try std.testing.expectEqualSlices(u8, expected, actual);
        pass("glm53-fuse: q_a halves RMSNorm rows={d}", .{rows});
    }
}

fn moeCombine(gpu: anytype, stream: cuda.Stream, module: cuda.Module) !void {
    const kernel = try module.function("tf_glm53_lowrank_combine_bf16");
    const separate_finish = try module.function("tf_glm53_lowrank_finish");
    const separate_add = try module.function("tf_glm53_add_partial");
    const rank = 32;
    for ([_]usize{ 1, 3, 8, 64 }) |rows| {
        const count = rows * 6144;
        const routed = try gpu.gpa.alloc(f32, count);
        defer gpu.gpa.free(routed);
        const shared = try gpu.gpa.alloc(f32, count);
        defer gpu.gpa.free(shared);
        const mid = try gpu.gpa.alloc(f32, rows * rank);
        defer gpu.gpa.free(mid);
        const b = try gpu.gpa.alloc(f32, rank * 6144);
        defer gpu.gpa.free(b);
        for (0..count) |i| {
            const a: f32 = @floatFromInt(@as(i32, @intCast((i * 19 + 3) % 257)) - 128);
            const sv: f32 = @floatFromInt(@as(i32, @intCast((i * 7 + 9) % 193)) - 96);
            routed[i] = a / 64;
            shared[i] = sv / 32;
        }
        for (mid, 0..) |*v, i| v.* = signed(i, 67) / 128;
        for (b, 0..) |*v, i| v.* = signed(i, 71) / 256;
        var dr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(routed));
        defer dr.free();
        var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(shared));
        defer ds.free();
        var dm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mid));
        defer dm.free();
        var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(b));
        defer db.free();
        var dseparate = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(shared));
        defer dseparate.free();
        var dseparate_out = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(routed));
        defer dseparate_out.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
        defer dout.free();
        var args: cuda.Args = .{};
        inline for (.{ dr.ptr, ds.ptr, db.ptr, dm.ptr, dout.ptr, @as(c_int, @intCast(rows)), @as(c_int, 6144), @as(c_int, rank), @as(c_int, 6144) }) |v| args.add(v);
        try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
        var finish: cuda.Args = .{};
        inline for (.{ db.ptr, dm.ptr, dseparate.ptr, @as(c_int, @intCast(rows)), @as(c_int, 6144), @as(c_int, rank), @as(c_int, 6144) }) |v| finish.add(v);
        try cuda.launch.launch(separate_finish, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &finish);
        var add: cuda.Args = .{};
        inline for (.{ dseparate.ptr, dseparate_out.ptr, @as(c_int, @intCast(count)) }) |v| add.add(v);
        try cuda.launch.launch(separate_add, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &add);
        try stream.synchronize();
        const got = try download(gpu, dout);
        defer gpu.gpa.free(got);
        const separate = try download(gpu, dseparate_out);
        defer gpu.gpa.free(separate);
        for (0..count) |i| {
            const separate_f32 = @as(f32, @bitCast(std.mem.readInt(u32, separate[i * 4 ..][0..4], .little)));
            try std.testing.expectEqual(bf16Bits(separate_f32), std.mem.readInt(u16, got[i * 2 ..][0..2], .little));
        }
        pass("glm53-fuse: fixed-order MoE BF16 combine rows={d}", .{rows});
    }
}

fn attentionEpilogue(gpu: anytype, stream: cuda.Stream, base: cuda.Module, fused: cuda.Module) !void {
    const separate_add = try base.function("tf_glm_residual_add");
    const separate_norm = try base.function("tf_glm_rmsnorm_6144");
    const fused_epilogue = try fused.function("tf_glm53_residual_rmsnorm_6144");
    for ([_]usize{ 1, 3, 8, 64 }) |rows| {
        const count = rows * 6144;
        const x = try gpu.gpa.alloc(u16, count);
        defer gpu.gpa.free(x);
        const parts = try gpu.gpa.alloc(f32, count * 2);
        defer gpu.gpa.free(parts);
        const weight = try gpu.gpa.alloc(u16, 6144);
        defer gpu.gpa.free(weight);
        for (x, 0..) |*v, i| v.* = bf16Bits(signed(i, 37) / 64);
        for (parts, 0..) |*v, i| v.* = signed(i, 53) / 512;
        for (weight, 0..) |*v, i| v.* = bf16Bits(1 + signed(i, 29) / 2048);
        var dx0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx0.free();
        var dx1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx1.free();
        var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(parts));
        defer dp.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weight));
        defer dw.free();
        var dn0 = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
        defer dn0.free();
        var dn1 = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
        defer dn1.free();

        var add: cuda.Args = .{};
        inline for (.{ dx0.ptr, dx0.ptr, dp.ptr, @as(i64, @intCast(count)), @as(i64, 6144), @as(c_int, 2) }) |v| add.add(v);
        try cuda.launch.launch(separate_add, .{ .grid = .{ .x = @intCast(rows), .y = 6 }, .block = .{ .x = 128 } }, stream, &add);
        var norm: cuda.Args = .{};
        inline for (.{ dx0.ptr, @as(i64, 6144), dw.ptr, dn0.ptr, @as(i64, 6144), @as(u64, 0), @as(f32, 1e-5), @as(c_int, 0) }) |v| norm.add(v);
        try cuda.launch.launch(separate_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &norm);
        var epilogue: cuda.Args = .{};
        inline for (.{ dx1.ptr, dp.ptr, dw.ptr, dn1.ptr, @as(c_int, @intCast(rows)), @as(f32, 1e-5) }) |v| epilogue.add(v);
        try cuda.launch.launch(fused_epilogue, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &epilogue);
        try stream.synchronize();
        const residual0 = try download(gpu, dx0);
        defer gpu.gpa.free(residual0);
        const residual1 = try download(gpu, dx1);
        defer gpu.gpa.free(residual1);
        const norm0 = try download(gpu, dn0);
        defer gpu.gpa.free(norm0);
        const norm1 = try download(gpu, dn1);
        defer gpu.gpa.free(norm1);
        try std.testing.expectEqualSlices(u8, residual0, residual1);
        try std.testing.expectEqualSlices(u8, norm0, norm1);
        pass("glm53-fuse: rank-sum residual and post-attention RMSNorm rows={d}", .{rows});
    }
}

fn moeAddNormEpilogue(gpu: anytype, stream: cuda.Stream, base: cuda.Module, fused: cuda.Module) !void {
    const separate_add = try fused.function("tf_glm53_moe_residual_add");
    const separate_norm = try base.function("tf_glm_rmsnorm_6144");
    const fused_epilogue = try fused.function("tf_glm53_moe_residual_rmsnorm_6144");
    for ([_]usize{ 1, 2, 3, 4, 8, 64 }) |rows| {
        const count = rows * 6144;
        const x = try gpu.gpa.alloc(u16, count);
        defer gpu.gpa.free(x);
        const parts = try gpu.gpa.alloc(u16, count * 2);
        defer gpu.gpa.free(parts);
        const weight = try gpu.gpa.alloc(u16, 6144);
        defer gpu.gpa.free(weight);
        for (x, 0..) |*v, i| {
            const row = i / 6144;
            v.* = if (row == 0) 0 else if (row == 1) bf16Bits(if (i % 2 == 0) 3.0e38 else -3.0e38) else if (row == 2) (if (i % 2 == 0) 0x0001 else 0x8001) else bf16Bits(signed(i, 37) / 64);
        }
        for (parts, 0..) |*v, i| {
            const row = i % count / 6144;
            v.* = if (row < 2) 0 else if (row == 2) (if (i % 2 == 0) 0x0001 else 0x8001) else bf16Bits(signed(i, 53) / 512);
        }
        for (weight, 0..) |*v, i| v.* = bf16Bits(1 + signed(i, 29) / 2048);
        var dx0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx0.free();
        var dx1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx1.free();
        var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(parts));
        defer dp.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weight));
        defer dw.free();
        var dn0 = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
        defer dn0.free();
        var dn1 = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
        defer dn1.free();

        var add: cuda.Args = .{};
        inline for (.{ dx0.ptr, dx0.ptr, dp.ptr, @as(c_int, @intCast(count)) }) |v| add.add(v);
        try cuda.launch.launch(separate_add, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &add);
        var norm: cuda.Args = .{};
        inline for (.{ dx0.ptr, @as(i64, 6144), dw.ptr, dn0.ptr, @as(i64, 6144), @as(u64, 0), @as(f32, 1e-5), @as(c_int, 0) }) |v| norm.add(v);
        try cuda.launch.launch(separate_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &norm);
        var epilogue: cuda.Args = .{};
        inline for (.{ dx1.ptr, dp.ptr, dw.ptr, dn1.ptr, @as(c_int, @intCast(rows)), @as(f32, 1e-5) }) |v| epilogue.add(v);
        try cuda.launch.launch(fused_epilogue, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &epilogue);
        try stream.synchronize();
        const residual0 = try download(gpu, dx0);
        defer gpu.gpa.free(residual0);
        const residual1 = try download(gpu, dx1);
        defer gpu.gpa.free(residual1);
        const norm0 = try download(gpu, dn0);
        defer gpu.gpa.free(norm0);
        const norm1 = try download(gpu, dn1);
        defer gpu.gpa.free(norm1);
        try std.testing.expectEqualSlices(u8, residual0, residual1);
        try std.testing.expectEqualSlices(u8, norm0, norm1);
        pass("glm53-fuse: MoE residual and next-layer RMSNorm rows={d}", .{rows});
    }
}

fn signed(i: usize, modulus: usize) f32 {
    return @floatFromInt(@as(i32, @intCast(i % modulus)) - @as(i32, @intCast(modulus / 2)));
}

fn bf16Bits(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn bf16Value(x: u16) f32 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

fn expectU16Bytes(expected: []const u16, actual: []const u8) !void {
    try std.testing.expectEqual(expected.len * 2, actual.len);
    for (expected, 0..) |value, i| try std.testing.expectEqual(value, std.mem.readInt(u16, actual[i * 2 ..][0..2], .little));
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn pass(comptime format: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ format ++ "\n", args);
}
