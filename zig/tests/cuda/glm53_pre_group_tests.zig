// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Grouped GLM-5.3 FP8B pre-attention projections versus their independent qmmf launches.
const std = @import("std");
const cuda = @import("cuda");
const pre_names = [_][]const u8{ "q_a", "kv_a", "wk" };
const path_names = [_][]const u8{ "separate", "grouped", "grouped_prefetch" };
const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;

test { std.testing.refAllDecls(@This()); }

const PackedWeight = struct { codes: cuda.DeviceBuffer, scales: cuda.DeviceBuffer, w: qmmf.Weight };

pub fn run(gpu: anytype) !void {
    var lane = try qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    try lane.loadGroup();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var prefetch_buffer = try cuda.DeviceBuffer.alloc(gpu.d, 16 * 1024 * 1024);
    defer prefetch_buffer.free();
    var prng = std.Random.DefaultPrng.init(0x6e7ca8e4);
    const random = prng.random();
    var weights: [3]PackedWeight = undefined;
    const widths = [_]usize{ 2048, 576, 128 };
    for (widths, 0..) |n, wi| {
        const npad = fp8.padded(n);
        const raw = try gpu.gpa.alloc(u8, n * 6144);
        defer gpu.gpa.free(raw);
        for (raw) |*v| { v.* = random.int(u8); while (v.* == 0x7f or v.* == 0xff) v.* = random.int(u8); }
        const packed_codes = try gpu.gpa.alloc(u8, npad * 6144);
        defer gpu.gpa.free(packed_codes);
        fp8.packCodes(packed_codes, raw, n, 6144);
        const sn = ((n + 127) / 128) * ((6144 + 127) / 128);
        const scales = try gpu.gpa.alloc(f32, sn);
        defer gpu.gpa.free(scales);
        for (scales) |*v| v.* = @as(f32, 1) / @as(f32, @floatFromInt(@as(u32, 1) << random.intRangeAtMost(u5, 2, 12)));
        const packed_scales = try gpu.gpa.alloc(f32, npad / 64 * (6144 / 64));
        defer gpu.gpa.free(packed_scales);
        try fp8.packBlockScales(packed_scales, scales, n, 6144, 0, 0, n, 6144);
        const dc = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
        const ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales));
        weights[wi] = .{ .codes = dc, .scales = ds, .w = .{ .mode = .fp8b, .codes = dc.ptr, .scales = ds.ptr, .n = @intCast(n), .k = 6144, .npad = @intCast(npad) } };
    }
    defer for (&weights) |*w| { w.codes.free(); w.scales.free(); };

    const rows_list = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 32, 33, 200, 256, 512, 2048 };
    for (rows_list) |rows| for ([_]bool{ false, true }) |split| {
        const ldx: usize = if ((rows == 4 or rows == 512) and split) 6208 else 6144;
        const xh = try gpu.gpa.alloc(u16, rows * ldx);
        defer gpu.gpa.free(xh);
        for (xh) |*v| v.* = random.int(u16);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer x.free();
        var refs: [3]cuda.DeviceBuffer = undefined;
        var outs: [3]cuda.DeviceBuffer = undefined;
        var parts: [3]cuda.DeviceBuffer = undefined;
        for (weights, 0..) |pw, i| {
            const bytes = rows * pw.w.n * 2;
            refs[i] = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
            outs[i] = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
            const fill = try gpu.gpa.alloc(u8, bytes);
            defer gpu.gpa.free(fill);
            @memset(fill, 0xa5);
            try outs[i].upload(0, fill);
            parts[i] = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, 8 * rows * pw.w.n * 4));
            try lane.matmulOut(stream, x.ptr, ldx, rows, pw.w, refs[i].ptr, .bf16, parts[i].ptr);
        }
        const grouped = [_]qmmf.GroupOut{ .{ .w = weights[0].w, .y = outs[0].ptr }, .{ .w = weights[1].w, .y = outs[1].ptr }, .{ .w = weights[2].w, .y = outs[2].ptr } };
        try lane.matmulGroup(stream, x.ptr, ldx, rows, &grouped, .{ .split = split });
        if (rows == 4 and !split) try lane.matmulGroup(stream, x.ptr, ldx, rows, &grouped, .{ .prefetch = prefetch_buffer.ptr, .prefetch_bytes = prefetch_buffer.len });
        if (!split and (rows == 1 or rows == 4 or rows == 8 or rows == 512)) {
            try benchPaths(gpu.d, lane, stream, x.ptr, ldx, rows, weights, refs, outs, prefetch_buffer);
        }
        try stream.synchronize();
        for (&refs, 0..) |*ref, i| {
            const expected = try gpu.gpa.alloc(u8, ref.len);
            defer gpu.gpa.free(expected);
            try ref.download(0, expected);
            const got = try gpu.gpa.alloc(u8, outs[i].len);
            defer gpu.gpa.free(got);
            try outs[i].download(0, got);
            if (!std.mem.eql(u8, expected, got)) {
                const exp16 = std.mem.bytesAsSlice(u16, expected);
                const got16 = std.mem.bytesAsSlice(u16, got);
                var max_abs: f32 = 0;
                var first_diff: usize = 0;
                var found_diff = false;
                for (exp16, got16, 0..) |a, b, at| if (a != b) {
                    if (!found_diff) { first_diff = at; found_diff = true; }
                    const av: f32 = @bitCast(@as(u32, a) << 16);
                    const bv: f32 = @bitCast(@as(u32, b) << 16);
                    max_abs = @max(max_abs, @abs(av - bv));
                };
                std.debug.print("FAIL glm53-pre-group {s} rows={d} split={} max_abs={d} first_diff={d}\n", .{ pre_names[i], rows, split, max_abs, first_diff });
                return error.PreGroupMismatch;
            }
            ref.free(); outs[i].free(); parts[i].free();
        }
        std.debug.print("PASS glm53-pre-group rows={d} split={}\n", .{ rows, split });
    };

    const bf_raw = try gpu.gpa.alloc(u16, 32 * 6144);
    defer gpu.gpa.free(bf_raw);
    for (bf_raw) |*v| v.* = random.int(u16);
    const bf_packed = try gpu.gpa.alloc(u16, 128 * 6144);
    defer gpu.gpa.free(bf_packed);
    qmmf.packBf16(bf_packed, bf_raw, 32, 6144);
    var bf_codes = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bf_packed));
    defer bf_codes.free();
    const bf_weight: qmmf.Weight = .{ .mode = .bf16, .codes = bf_codes.ptr, .scales = 0, .n = 32, .k = 6144, .npad = 128 };
    for ([_]usize{ 1, 17, 512, 2048 }) |rows| {
        const xh = try gpu.gpa.alloc(u16, rows * 6144);
        defer gpu.gpa.free(xh);
        for (xh) |*v| v.* = random.int(u16);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer x.free();
        var ref = try cuda.DeviceBuffer.alloc(gpu.d, rows * 32 * 2);
        defer ref.free();
        var actual = try cuda.DeviceBuffer.alloc(gpu.d, rows * 32 * 2);
        defer actual.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, 8 * rows * 32 * 4));
        defer part.free();
        try lane.matmulOut(stream, x.ptr, 6144, rows, bf_weight, ref.ptr, .bf16, part.ptr);
        try lane.matmulOutSplit(stream, x.ptr, 6144, rows, bf_weight, actual.ptr, .bf16, part.ptr);
        try stream.synchronize();
        const expected = try gpu.gpa.alloc(u8, ref.len);
        defer gpu.gpa.free(expected);
        try ref.download(0, expected);
        const got = try gpu.gpa.alloc(u8, actual.len);
        defer gpu.gpa.free(got);
        try actual.download(0, got);
        if (!std.mem.eql(u8, expected, got)) return error.PreGroupMismatch;
    }
}

fn benchPaths(driver: *const cuda.Driver, lane: qmmf.Lane, stream: cuda.Stream, x: u64, ldx: usize, rows: usize, weights: [3]PackedWeight, refs: [3]cuda.DeviceBuffer, outs: [3]cuda.DeviceBuffer, prefetch: cuda.DeviceBuffer) !void {
    const reps = 50;
    const grouped = [_]qmmf.GroupOut{ .{ .w = weights[0].w, .y = outs[0].ptr }, .{ .w = weights[1].w, .y = outs[1].ptr }, .{ .w = weights[2].w, .y = outs[2].ptr } };
    const bytes = weights[0].codes.len + weights[0].scales.len + weights[1].codes.len + weights[1].scales.len + weights[2].codes.len + weights[2].scales.len;
    var start = try cuda.Event.init(driver, true);
    defer start.deinit();
    var stop = try cuda.Event.init(driver, true);
    defer stop.deinit();
    for (0..3) |path| {
        // Warm the selected path and its code/data addresses before timing.
        if (path == 0) {
            for (weights, 0..) |pw, i| try lane.matmulOut(stream, x, ldx, rows, pw.w, refs[i].ptr, .bf16, 0);
        } else if (path == 1) {
            try lane.matmulGroup(stream, x, ldx, rows, &grouped, .{});
        } else {
            try lane.matmulGroup(stream, x, ldx, rows, &grouped, .{ .prefetch = prefetch.ptr, .prefetch_bytes = prefetch.len });
        }
        try stream.synchronize();
        try start.record(stream);
        for (0..reps) |_| {
            if (path == 0) {
                for (weights, 0..) |pw, i| try lane.matmulOut(stream, x, ldx, rows, pw.w, refs[i].ptr, .bf16, 0);
            } else if (path == 1) {
                try lane.matmulGroup(stream, x, ldx, rows, &grouped, .{});
            } else {
                try lane.matmulGroup(stream, x, ldx, rows, &grouped, .{ .prefetch = prefetch.ptr, .prefetch_bytes = prefetch.len });
            }
        }
        try stop.record(stream);
        try stop.synchronize();
        const us = @as(f64, try start.elapsedMs(stop)) * 1000.0 / reps;
        const gbps = @as(f64, @floatFromInt(bytes)) / us / 1000.0;
        std.debug.print("RESULT glm53-pre-group rows={d} path={s} us={d:.3} GB/s={d:.3}\n", .{ rows, path_names[path], us, gbps });
    }
}
