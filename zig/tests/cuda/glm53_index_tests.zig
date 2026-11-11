// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 DSA indexer and DCP merge GPU parity checks (WP7).
const std = @import("std");
const cuda = @import("cuda");
const index = @import("index.zig");

test "GLM-5.3 index declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_index);
    defer module.unload();
    const kernels = try index.Index.load(module, 4);
    try qsResources(kernels);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try scoresCase(gpu, stream, &kernels);
    try scoresMmaLargeRowTiles(gpu, stream, &kernels);
    try scoresQsParityCases(gpu, stream, &kernels);
    try tileSkipCases(gpu, stream, &kernels);
    try topkCases(gpu, stream, &kernels);
    try mergeCase(gpu, stream, &kernels);
    try merge2Cases(gpu, stream, &kernels);
    try merge2BlockRowsCase(gpu, stream, &kernels, 130, 64);
    try merge2BlockRowsCase(gpu, stream, &kernels, 513, 64); // INDEX_PIPE schedule with a 513-row prompt tail.
    try topkMerge2RandomCase(gpu, stream, &kernels);
    try merge2EdgeCases(gpu, stream, &kernels);
}

fn qsResources(kernels: index.Index) !void {
    const Variant = struct { name: []const u8, function: cuda.Function, shared: u32 };
    const variants = [_]Variant{
        .{ .name = "qs_s2", .function = kernels.score_qs_2, .shared = index.scoreQsSmem(2, 64) },
        .{ .name = "qs_s3", .function = kernels.score_qs_3, .shared = index.scoreQsSmem(3, 64) },
        .{ .name = "qs2_k64_s2", .function = kernels.score_qs2_k64_s2, .shared = index.scoreQsSmem(2, 64) },
        .{ .name = "qs2_k32_s2", .function = kernels.score_qs2_k32_s2, .shared = index.scoreQsSmem(2, 32) },
        .{ .name = "qs2_k32_s3", .function = kernels.score_qs2_k32_s3, .shared = index.scoreQsSmem(3, 32) },
    };
    for (variants) |variant| {
        const regs = try variant.function.attribute(.num_regs);
        const static_shared = try variant.function.attribute(.shared_size_bytes);
        const active = try variant.function.occupancy(256, @intCast(variant.shared));
        std.debug.print("INFO glm53-index resources {s}: {d} registers, {d} B static + {d} B dynamic shared, {d} CTAs/SM\n", .{ variant.name, regs, static_shared, variant.shared, active });
    }
    pass("glm53-index: query-stationary kernel registers, shared memory, and active CTAs/SM reported", .{});
}

fn scoresCase(gpu: anytype, stream: cuda.Stream, kernels_in: *const index.Index) !void {
    var scalar_k = kernels_in.*;
    scalar_k.score_kernel = .scalar;
    try scoresMmaCases(gpu, stream, kernels_in);
    const kernels = &scalar_k;
    const rows = 3;
    const n = 16;
    const q_count = rows * index.heads * index.dim;
    const key_count = n * index.dim;
    const q = try gpu.gpa.alloc(u16, q_count);
    defer gpu.gpa.free(q);
    const iw = try gpu.gpa.alloc(u16, rows * index.heads);
    defer gpu.gpa.free(iw);
    const keys = try gpu.gpa.alloc(u16, key_count);
    defer gpu.gpa.free(keys);
    const qf = try gpu.gpa.alloc(f32, q_count);
    defer gpu.gpa.free(qf);
    const iwf = try gpu.gpa.alloc(f32, iw.len);
    defer gpu.gpa.free(iwf);
    const keyf = try gpu.gpa.alloc(f32, key_count);
    defer gpu.gpa.free(keyf);
    for (q, 0..) |*v, i| {
        v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 32);
        qf[i] = bf16Value(v.*);
    }
    for (iw, 0..) |*v, i| {
        v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 16);
        iwf[i] = bf16Value(v.*);
    }
    for (keys, 0..) |*v, i| {
        v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 32);
        keyf[i] = bf16Value(v.*);
    }
    var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dqi.free();
    var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
    defer diw.free();
    var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(keys));
    defer dkeys.free();
    var positions: [rows]i64 = .{ n - 1, n - 1, n - 1 };
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&positions));
    defer dpos.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer out.free();
    try kernels.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, rows, n, n, 0, 1);
    try stream.synchronize();
    const got = try download(gpu, out);
    defer gpu.gpa.free(got);
    const actual = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    for (0..rows) |r| {
        var expected: [n]f32 = undefined;
        try index.scoresRef(qf, iwf, index.heads, 0, r, keyf, n - 1, &expected, 0, 1);
        for (0..n) |j| try expect(@abs(actual[r * n + j] - expected[j]) <= @max(2e-3, @abs(expected[j]) * 2e-3), "index score row={d} key={d}", .{ r, j });
        var one = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer one.free();
        try kernels.scores(stream, dqi.ptr + r * index.heads * index.dim * 2, diw.ptr + r * index.heads * 2, index.heads, 0, dkeys.ptr, dpos.ptr + r * 8, one.ptr, 1, n, n, 0, 1);
        try stream.synchronize();
        const alone = try download(gpu, one);
        defer gpu.gpa.free(alone);
        try expect(std.mem.eql(u8, alone, got[r * n * 4 ..][0 .. n * 4]), "index score row invariance row={d}", .{r});
    }
    pass("glm53-index: fp32 indexer scores match host tolerance and stay row-invariant", .{});
}

/// Tile skipping preserves exact top-k tokens and counts against the host oracle at the listed positions and both ranks.
fn tileSkipCases(gpu: anytype, stream: cuda.Stream, kernels_in: *const index.Index) !void {
    const n = 131072;
    const rows = 5;
    const positions_host = [_]i64{ 2048, 70000, 131071, 200000, 262143 };
    var prng = std.Random.DefaultPrng.init(0x51a7);
    const rnd = prng.random();
    const q = try gpu.gpa.alloc(u16, rows * index.heads * index.dim);
    defer gpu.gpa.free(q);
    const qf = try gpu.gpa.alloc(f32, q.len);
    defer gpu.gpa.free(qf);
    const iw = try gpu.gpa.alloc(u16, rows * index.heads);
    defer gpu.gpa.free(iw);
    const iwf = try gpu.gpa.alloc(f32, iw.len);
    defer gpu.gpa.free(iwf);
    const keys = try gpu.gpa.alloc(u16, n * index.dim);
    defer gpu.gpa.free(keys);
    const keyf = try gpu.gpa.alloc(f32, keys.len);
    defer gpu.gpa.free(keyf);
    for (q, 0..) |*v, i| {
        v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        qf[i] = bf16Value(v.*);
    }
    for (iw, 0..) |*v, i| {
        v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        iwf[i] = bf16Value(v.*);
    }
    for (keys, 0..) |*v, i| {
        v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        keyf[i] = bf16Value(v.*);
    }
    var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dqi.free();
    var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
    defer diw.free();
    var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(keys));
    defer dkeys.free();
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&positions_host));
    defer dpos.free();
    var scored = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer scored.free();
    var tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 4);
    defer tokens.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer counts.free();
    var records = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 8);
    defer records.free();
    var mma_kernels = kernels_in.*;
    mma_kernels.score_kernel = .mma;
    const kernels = &mma_kernels;
    for ([_]usize{ 0, 1 }) |rank| {
        try kernels.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, scored.ptr, rows, n, n, rank, 2);
        try kernels.topk(stream, scored.ptr, n, tokens.ptr, counts.ptr, records.ptr, dpos.ptr, rows, n, index.top_k, rank, 2);
        try stream.synchronize();
        const got_tokens = try download(gpu, tokens);
        defer gpu.gpa.free(got_tokens);
        const got_counts_bytes = try download(gpu, counts);
        defer gpu.gpa.free(got_counts_bytes);
        const got_counts = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, got_counts_bytes));
        const got_token_values = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, got_tokens));
        for (0..rows) |r| {
            const expected_scores: []f32 = try gpu.gpa.alloc(f32, n);
            defer gpu.gpa.free(expected_scores);
            try index.scoresRef(qf, iwf, index.heads, 0, r, keyf, @intCast(positions_host[r]), expected_scores, rank, 2);
            const expected = try index.topkDcpRef(gpu.gpa, expected_scores, @intCast(positions_host[r]), index.top_k, rank, 2);
            defer gpu.gpa.free(expected);
            var expected_count: i32 = 0;
            for (expected) |token| expected_count += @intFromBool(token >= 0);
            try expect(got_counts[r] == expected_count, "index tile skip top-k count row={d} rank={d}: got {d}, expected {d}", .{ r, rank, got_counts[r], expected_count });
            // The selection is a set: topkDcpRef sorts ascending (its -1 padding first), so compare sorted copies.
            const got_row = try gpu.gpa.alloc(i32, index.top_k);
            for (got_row, got_token_values[r * index.top_k ..][0..index.top_k]) |*d, v| d.* = v;
            defer gpu.gpa.free(got_row);
            std.mem.sort(i32, got_row, {}, std.sort.asc(i32));
            try expect(std.mem.eql(i32, expected, got_row), "index tile skip top-k tokens row={d} rank={d}", .{ r, rank });
        }
    }
    pass("glm53-index: tile skip matches host top-k tokens and counts at positions 2048/70000/131071/200000/262143 on both ranks", .{});

    // Benchmark the decode bucket with a single row at position 140000.
    var timing_pos: i64 = 140000;
    var dtime_pos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&timing_pos));
    defer dtime_pos.free();
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    for (0..1) |_| try kernels.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dtime_pos.ptr, scored.ptr, 1, n, n, 0, 2);
    try e0.record(stream);
    for (0..2) |_| try kernels.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dtime_pos.ptr, scored.ptr, 1, n, n, 0, 2);
    try e1.record(stream);
    try stream.synchronize();
    std.debug.print("BENCH index scores bucket=262144 position=140000: {d:.1} us\n", .{(try e0.elapsedMs(e1)) * 500.0});
}

fn topkCases(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    const n = 4096;
    const rows = 3;
    const score_values = try gpu.gpa.alloc(f32, n * rows);
    defer gpu.gpa.free(score_values);
    for (score_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 29) % 1021)) / 17;
    score_values[0] = -0.0;
    score_values[1] = 0.0;
    score_values[2] = std.math.nan(f32);
    var dscore = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(score_values));
    defer dscore.free();
    var positions_host: [rows]i64 = .{ n - 1, n - 1, n - 1 };
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&positions_host));
    defer dpos.free();
    for ([_]usize{ 1, 8, 64, 2048 }) |k| {
        var tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer tokens.free();
        var counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
        defer counts.free();
        try kernels.topk(stream, dscore.ptr, n, tokens.ptr, counts.ptr, 0, dpos.ptr, rows, n, k, 0, 1);
        try stream.synchronize();
        const got = try download(gpu, tokens);
        defer gpu.gpa.free(got);
        const got_counts = try download(gpu, counts);
        defer gpu.gpa.free(got_counts);
        const actual_counts = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, got_counts));
        for (0..rows) |r| {
            const expected = try index.topkRef(gpu.gpa, score_values[r * n ..][0..n], n - 1, k);
            defer gpu.gpa.free(expected);
            try expect(actual_counts[r] == k, "index top-k count row={d} k={d}", .{ r, k });
            try expect(std.mem.eql(u8, got[r * k * 4 ..][0 .. k * 4], std.mem.sliceAsBytes(expected)), "index top-k row={d} k={d}", .{ r, k });
        }
        pass("glm53-index: exact top-k k={d} across {d} rows (NaN and signed zero included)", .{ k, rows });
    }
}

fn mergeCase(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    const k = 64;
    const records = try gpu.gpa.alloc(index.PackedCandidate, 2 * k);
    defer gpu.gpa.free(records);
    for (records, 0..) |*r, i| r.* = .{ .score = @as(f32, @floatFromInt((i * 13) % 97)), .token = @intCast((i % k) * 2 + i / k) };
    var drecords = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(records));
    defer drecords.free();
    var global = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global.free();
    var global_count = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count.free();
    var local = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local.free();
    var local_count = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count.free();
    var global2 = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global2.free();
    var global_count2 = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count2.free();
    var local2 = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local2.free();
    var local_count2 = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count2.free();
    var position: i64 = 4095;
    var dposition = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&position));
    defer dposition.free();
    try kernels.merge(stream, drecords.ptr, global.ptr, global_count.ptr, local.ptr, local_count.ptr, dposition.ptr, 0, 2, 1, k);
    try kernels.merge2(stream, drecords.ptr, global2.ptr, global_count2.ptr, local2.ptr, local_count2.ptr, dposition.ptr, 0, 2, 1, k, 1);
    try stream.synchronize();
    const got_global = try download(gpu, global);
    defer gpu.gpa.free(got_global);
    const got_local = try download(gpu, local);
    defer gpu.gpa.free(got_local);
    const got_global2 = try download(gpu, global2);
    defer gpu.gpa.free(got_global2);
    const got_local2 = try download(gpu, local2);
    defer gpu.gpa.free(got_local2);
    const got_global_count = try download(gpu, global_count);
    defer gpu.gpa.free(got_global_count);
    const got_global_count2 = try download(gpu, global_count2);
    defer gpu.gpa.free(got_global_count2);
    const got_local_count = try download(gpu, local_count);
    defer gpu.gpa.free(got_local_count);
    const got_local_count2 = try download(gpu, local_count2);
    defer gpu.gpa.free(got_local_count2);
    const want = try index.mergePackedDcp(gpu.gpa, records, 2, 0, k, 4095);
    defer gpu.gpa.free(want.global);
    defer gpu.gpa.free(want.local);
    try expect(std.mem.eql(u8, got_global, std.mem.sliceAsBytes(want.global)), "DCP merge global tokens", .{});
    try expect(std.mem.eql(u8, got_local, std.mem.sliceAsBytes(want.local)), "DCP merge local remap", .{});
    try expect(std.mem.eql(u8, got_global, got_global2) and std.mem.eql(u8, got_local, got_local2) and std.mem.eql(u8, got_global_count, got_global_count2) and std.mem.eql(u8, got_local_count, got_local_count2), "merge2 byte parity vs merge", .{});
    pass("glm53-index: rank_buf two-rank merge2 byte parity and host oracle", .{});
}

fn merge2Cases(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    const k = index.top_k;
    const records = try gpu.gpa.alloc(index.PackedCandidate, 2 * k);
    defer gpu.gpa.free(records);
    for (records, 0..) |*r, i| r.* = .{ .score = @as(f32, @floatFromInt((i * 17) % 251)), .token = @intCast((i % k) * 2 + i / k) };
    var drecords = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(records));
    defer drecords.free();
    var global = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global.free();
    var global2 = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global2.free();
    var global_count = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count.free();
    var global_count2 = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count2.free();
    var local = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local.free();
    var local2 = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local2.free();
    var local_count = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count.free();
    var local_count2 = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count2.free();
    var position: i64 = 8191;
    var dposition = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&position));
    defer dposition.free();
    try kernels.merge(stream, drecords.ptr, global.ptr, global_count.ptr, local.ptr, local_count.ptr, dposition.ptr, 1, 2, 1, k);
    try kernels.merge2(stream, drecords.ptr, global2.ptr, global_count2.ptr, local2.ptr, local_count2.ptr, dposition.ptr, 1, 2, 1, k, 1);
    try stream.synchronize();
    try compareBuffers(gpu, global, global2, "merge2 k=2048 global parity");
    try compareBuffers(gpu, global_count, global_count2, "merge2 k=2048 global count parity");
    try compareBuffers(gpu, local, local2, "merge2 k=2048 local parity");
    try compareBuffers(gpu, local_count, local_count2, "merge2 k=2048 local count parity");
    pass("glm53-index: k=2048 merge2 parity", .{});
}

fn merge2BlockRowsCase(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index, rows: usize, sr: usize) !void {
    const k = 64;
    const rank_major = try gpu.gpa.alloc(index.PackedCandidate, rows * 2 * k);
    defer gpu.gpa.free(rank_major);
    const pos_host = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(pos_host);
    @memset(pos_host, 4095);
    if (rows == 513) {
        // Exercise score -> top-k -> gathered block-major merge for a non-512 prompt length.
        const nkeys = 1024;
        var prng = std.Random.DefaultPrng.init(0x51_3_64);
        const rnd = prng.random();
        const scores = try gpu.gpa.alloc(f32, rows * nkeys);
        defer gpu.gpa.free(scores);
        for (scores) |*v| v.* = rnd.float(f32) * 8 - 4;
        var dscores = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scores));
        defer dscores.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_host));
        defer dpos.free();
        var records = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 8);
        defer records.free();
        var tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer tokens.free();
        var counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
        defer counts.free();
        for (0..2) |which| {
            try kernels.topk(stream, dscores.ptr, nkeys, tokens.ptr, counts.ptr, records.ptr, dpos.ptr, rows, nkeys, k, which, 2);
            try stream.synchronize();
            const bytes = try download(gpu, records);
            @memcpy(rank_major[which * rows * k ..][0 .. rows * k], std.mem.bytesAsSlice(index.PackedCandidate, @as([]align(1) const u8, bytes)));
            gpu.gpa.free(bytes);
        }
    } else {
        for (0..2) |which| for (0..rows) |r| for (0..k) |col| {
            const i = (which * rows + r) * k + col;
            rank_major[i] = .{ .score = @floatFromInt((r * 23 + col * 11 + which * 5) % 101), .token = @intCast(col * 2 + which) };
        };
    }
    const block_major = try gpu.gpa.alloc(index.PackedCandidate, rows * 2 * k);
    defer gpu.gpa.free(block_major);
    var base: usize = 0;
    while (base < rows) : (base += sr) {
        const nr = @min(sr, rows - base);
        for (0..2) |which| for (0..nr) |r| {
            const src = (which * rows + base + r) * k;
            const dst = (base / sr * 2 * sr + which * nr + r) * k;
            @memcpy(block_major[dst..][0..k], rank_major[src..][0..k]);
        };
    }
    var rank_buf = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rank_major));
    defer rank_buf.free();
    var piped = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(block_major));
    defer piped.free();
    var positions = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_host));
    defer positions.free();
    var global = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer global.free();
    var global2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer global2.free();
    var gc = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer gc.free();
    var gc2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer gc2.free();
    var local = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer local.free();
    var local2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer local2.free();
    var lc = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer lc.free();
    var lc2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer lc2.free();
    try kernels.merge(stream, rank_buf.ptr, global.ptr, gc.ptr, local.ptr, lc.ptr, positions.ptr, 0, 2, rows, k);
    try kernels.merge2(stream, piped.ptr, global2.ptr, gc2.ptr, local2.ptr, lc2.ptr, positions.ptr, 0, 2, rows, k, sr);
    try stream.synchronize();
    try compareBuffers(gpu, global, global2, "block-major merge2 global byte parity");
    try compareBuffers(gpu, gc, gc2, "block-major merge2 global counts byte parity");
    try compareBuffers(gpu, local, local2, "block-major merge2 local byte parity");
    try compareBuffers(gpu, lc, lc2, "block-major merge2 local counts byte parity");
}

fn topkMerge2RandomCase(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    const rows = 64;
    const nkeys = 16384;
    const k = index.top_k;
    var prng = std.Random.DefaultPrng.init(0x53_16_64);
    const rnd = prng.random();
    const scores = try gpu.gpa.alloc(f32, rows * nkeys);
    defer gpu.gpa.free(scores);
    for (scores) |*v| v.* = rnd.float(f32) * 8 - 4;
    const pos_host = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(pos_host);
    @memset(pos_host, 32767);
    var dscores = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scores));
    defer dscores.free();
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_host));
    defer dpos.free();
    var records = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 8);
    defer records.free();
    var tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer tokens.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer counts.free();
    const rank_records = try gpu.gpa.alloc(index.PackedCandidate, 2 * rows * k);
    defer gpu.gpa.free(rank_records);
    for (0..2) |rank| {
        try kernels.topk(stream, dscores.ptr, nkeys, tokens.ptr, counts.ptr, records.ptr, dpos.ptr, rows, nkeys, k, rank, 2);
        try stream.synchronize();
        const bytes = try download(gpu, records);
        @memcpy(rank_records[rank * rows * k ..][0 .. rows * k], std.mem.bytesAsSlice(index.PackedCandidate, @as([]align(1) const u8, bytes)));
        gpu.gpa.free(bytes);
    }
    const rank_major = try gpu.gpa.alloc(index.PackedCandidate, rows * 2 * k);
    defer gpu.gpa.free(rank_major);
    const block_major = try gpu.gpa.alloc(index.PackedCandidate, rows * 2 * k);
    defer gpu.gpa.free(block_major);
    for (0..2) |rank| for (0..rows) |r| {
        @memcpy(rank_major[(rank * rows + r) * k ..][0..k], rank_records[(rank * rows + r) * k ..][0..k]);
        @memcpy(block_major[(rank * rows + r) * k ..][0..k], rank_records[(rank * rows + r) * k ..][0..k]);
    };
    var rank_buf = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rank_major));
    defer rank_buf.free();
    var tiled = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(block_major));
    defer tiled.free();
    var global = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer global.free();
    var global2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer global2.free();
    var gc = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer gc.free();
    var gc2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer gc2.free();
    var local = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer local.free();
    var local2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
    defer local2.free();
    var lc = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer lc.free();
    var lc2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer lc2.free();
    try kernels.merge(stream, rank_buf.ptr, global.ptr, gc.ptr, local.ptr, lc.ptr, dpos.ptr, 0, 2, rows, k);
    try kernels.merge2(stream, tiled.ptr, global2.ptr, gc2.ptr, local2.ptr, lc2.ptr, dpos.ptr, 0, 2, rows, k, rows);
    try stream.synchronize();
    try compareBuffers(gpu, global, global2, "random topk->merge global byte parity");
    try compareBuffers(gpu, gc, gc2, "random topk->merge global-count byte parity");
    try compareBuffers(gpu, local, local2, "random topk->merge local byte parity");
    try compareBuffers(gpu, lc, lc2, "random topk->merge local-count byte parity");
}

fn compareBuffers(gpu: anytype, old_buffer: cuda.DeviceBuffer, new_buffer: cuda.DeviceBuffer, comptime label: []const u8) !void {
    const old = try download(gpu, old_buffer);
    defer gpu.gpa.free(old);
    const new = try download(gpu, new_buffer);
    defer gpu.gpa.free(new);
    try expect(std.mem.eql(u8, old, new), label, .{});
}

fn merge2EdgeCases(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    try merge2EdgeCase(gpu, stream, kernels, 2048, 0, 4095, 0);
    try merge2EdgeCase(gpu, stream, kernels, 2048, 1, 8191, 1);
    try merge2EdgeCase(gpu, stream, kernels, 2048, 2, 4095, 0);
    try merge2EdgeCase(gpu, stream, kernels, 2048, 3, 4095, 1);
    try merge2EdgeCase(gpu, stream, kernels, 8, 0, 2047, 0);
    try merge2EdgeCase(gpu, stream, kernels, 2048, 4, 4095, 0);
}

fn merge2EdgeCase(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index, k: usize, pattern: u8, position: i64, rank: usize) !void {
    const nan_negative: f32 = @bitCast(@as(u32, 0xffc00001));
    const records = try gpu.gpa.alloc(index.PackedCandidate, 2 * k);
    defer gpu.gpa.free(records);
    for (0..2) |which| for (0..k) |col| {
        const i = which * k + col;
        const valid = switch (pattern) {
            2 => col < 17,
            3 => which == 1,
            4 => col < 17,
            else => true,
        };
        const token: i32 = if (valid) @intCast(col * 2 + which) else -1;
        const score: f32 = if (!valid) -std.math.inf(f32) else switch (pattern) {
            0 => 3,
            4 => if (token == 4) nan_negative else 1.0,
            1 => switch (col % 8) { 0 => -0.0, 1 => 0.0, 2 => std.math.nan(f32), 3 => nan_negative, 4 => std.math.inf(f32), 5 => -std.math.inf(f32), 6 => 5, else => -2 },
            else => @floatFromInt((col * 11 + which * 7) % 31),
        };
        records[i] = .{ .score = score, .token = token };
    };
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(records));
    defer input.free();
    var global_old = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global_old.free();
    var global_new = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer global_new.free();
    var global_count_old = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count_old.free();
    var global_count_new = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer global_count_new.free();
    var local_old = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local_old.free();
    var local_new = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer local_new.free();
    var local_count_old = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count_old.free();
    var local_count_new = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer local_count_new.free();
    var dposition = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&position));
    defer dposition.free();
    try kernels.merge(stream, input.ptr, global_old.ptr, global_count_old.ptr, local_old.ptr, local_count_old.ptr, dposition.ptr, rank, 2, 1, k);
    try kernels.merge2(stream, input.ptr, global_new.ptr, global_count_new.ptr, local_new.ptr, local_count_new.ptr, dposition.ptr, rank, 2, 1, k, 1);
    try stream.synchronize();
    try compareBuffers(gpu, global_old, global_new, "merge2 edge global parity");
    try compareBuffers(gpu, global_count_old, global_count_new, "merge2 edge global count parity");
    try compareBuffers(gpu, local_old, local_new, "merge2 edge local parity");
    try compareBuffers(gpu, local_count_old, local_count_new, "merge2 edge local count parity");
    if (pattern == 4) {
        const global_bytes = try download(gpu, global_old);
        defer gpu.gpa.free(global_bytes);
        const count_bytes = try download(gpu, global_count_old);
        defer gpu.gpa.free(count_bytes);
        const count = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, count_bytes))[0];
        const tokens = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, global_bytes));
        // Both merges rank on `score + 0.0f`, and GPU arithmetic returns the canonical +NaN (0x7fffffff) for any NaN
        // input, so a negative NaN ranks first and is kept: 34 of 34 valid. merge2 must match it (parity above).
        try expect(count == 34, "negative NaN legacy merge count: {d}", .{count});
        var has_nan_token = false;
        for (tokens) |token| has_nan_token = has_nan_token or token == 4;
        try expect(has_nan_token, "negative NaN token 4 must be kept (canonical +NaN ranks first)", .{});
    }
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

fn bf16Bits(value: f32) u16 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn bf16Value(value: u16) f32 {
    return @as(f32, @bitCast(@as(u32, value) << 16));
}

test "top-k host oracle sorts ascending, pads, and canonicalizes NaN ties" {
    const scores = [_]f32{ 0, -0.0, std.math.nan(f32), 5, 5, -1 };
    const got = try index.topkRef(std.testing.allocator, &scores, 5, 8);
    defer std.testing.allocator.free(got);
    for (0..6) |i| try std.testing.expectEqual(@as(i32, @intCast(i)), got[i]);
    try std.testing.expectEqual(@as(i32, -1), got[7]);
    const best = try index.topkRef(std.testing.allocator, &scores, 5, 2);
    defer std.testing.allocator.free(best);
    try std.testing.expectEqual(@as(i32, 2), best[0]);
    try std.testing.expectEqual(@as(i32, 3), best[1]);
}

/// The tiled tensor-core scores against the scalar kernel, bit for bit, on random bf16 rows and keys (rows 1, 3, 8,
/// 37; keys 1, 63, 64, 1000, 4100; causal masking across both ranks), and rows alone == rows in a batch.
fn scoresMmaCases(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    var scalar_k = kernels.*;
    scalar_k.score_kernel = .scalar;
    var prng = std.Random.DefaultPrng.init(0x1d3c5);
    const rnd = prng.random();
    for ([_]usize{ 4, 8, 16 }) |rows_per_cta| {
    var mma_k = kernels.*;
    mma_k.score_kernel = .mma;
    mma_k.score_mma_rows = rows_per_cta;
    for ([_]usize{ 1, 3, 8, 37 }) |rows| for ([_]usize{ 1, 63, 64, 1000, 4100 }) |n| for ([_]usize{ 0, 1 }) |rank| {
        const q = try gpu.gpa.alloc(u16, rows * index.heads * index.dim);
        defer gpu.gpa.free(q);
        const iw = try gpu.gpa.alloc(u16, rows * 40);
        defer gpu.gpa.free(iw);
        const keys = try gpu.gpa.alloc(u16, n * index.dim);
        defer gpu.gpa.free(keys);
        for (q) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (iw) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (keys) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        const positions = try gpu.gpa.alloc(i64, rows);
        defer gpu.gpa.free(positions);
        for (positions, 0..) |*p, r| p.* = @intCast((2 * n * (r + 1)) / (rows + 1)); // some keys past each row's position
        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(keys));
        defer dkeys.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
        defer dpos.free();
        const stride = n + 3;
        var a = try cuda.DeviceBuffer.alloc(gpu.d, rows * stride * 4);
        defer a.free();
        var b = try cuda.DeviceBuffer.alloc(gpu.d, rows * stride * 4);
        defer b.free();
        try gpu.d.check(gpu.d.api.cuMemsetD8_v2(a.ptr, 0, a.len), "cuMemsetD8");
        try gpu.d.check(gpu.d.api.cuMemsetD8_v2(b.ptr, 0, b.len), "cuMemsetD8");
        try scalar_k.scores(stream, dqi.ptr, diw.ptr, 40, 5, dkeys.ptr, dpos.ptr, a.ptr, rows, n, stride, rank, 2);
        try mma_k.scores(stream, dqi.ptr, diw.ptr, 40, 5, dkeys.ptr, dpos.ptr, b.ptr, rows, n, stride, rank, 2);
        try stream.synchronize();
        const ga = try download(gpu, a);
        defer gpu.gpa.free(ga);
        const gb = try download(gpu, b);
        defer gpu.gpa.free(gb);
        // Compare only the entries top-k reads (e < visible slots of the row): both kernels skip whole blocks past
        // every row's visible slots (decode index-score tile skip), at different block sizes, so unread tails differ.
        const fa = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, ga));
        const fb = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, gb));
        var bad: ?usize = null;
        for (0..rows) |r| {
            const p: i64 = @intCast(positions[r]);
            const vis_i: i64 = if (p < @as(i64, @intCast(rank))) 0 else @divTrunc(p - @as(i64, @intCast(rank)), 2) + 1;
            const vis: usize = @min(@as(usize, @intCast(vis_i)), n);
            for (0..vis) |e| {
                const i = r * stride + e;
                if (@as(u32, @bitCast(fa[i])) != @as(u32, @bitCast(fb[i]))) {
                    bad = i;
                    break;
                }
            }
            if (bad != null) break;
        }
        if (bad) |i| {
            std.debug.print("FAIL glm53-index mma scores rows={d} keys={d} rank={d}: first diff at {d}: scalar {e} mma {e}\n", .{ rows, n, rank, i, fa[i], fb[i] });
            return error.TestFailed;
        }
        if (rows > 1) {
            var one = try cuda.DeviceBuffer.alloc(gpu.d, stride * 4);
            defer one.free();
            try gpu.d.check(gpu.d.api.cuMemsetD8_v2(one.ptr, 0, one.len), "cuMemsetD8");
            const r = rows - 1;
            try mma_k.scores(stream, dqi.ptr + r * index.heads * index.dim * 2, diw.ptr + r * 40 * 2, 40, 5, dkeys.ptr, dpos.ptr + r * 8, one.ptr, 1, n, stride, rank, 2);
            try stream.synchronize();
            const alone = try download(gpu, one);
            defer gpu.gpa.free(alone);
            try expect(std.mem.eql(u8, alone[0 .. n * 4], gb[r * stride * 4 ..][0 .. n * 4]), "index mma scores row invariance rows={d} keys={d}", .{ rows, n });
        }
    };
    }
    pass("glm53-index: 4/8/16-row CTA tensor-core scores == scalar bit for bit (rows 1/3/8/37, keys 1..4100, both ranks), row-invariant", .{});
}

/// Prefill row-tile contract: exact scores and radix top-k at 2k/16k contexts for both DCP ranks.
fn scoresMmaLargeRowTiles(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    var prng = std.Random.DefaultPrng.init(8_160_2048);
    const rnd = prng.random();
    for ([_]usize{ 1, 7, 64, 512 }) |rows| for ([_]usize{ 1024, 8192 }) |nkeys| for ([_]usize{ 0, 1 }) |rank| {
        const q = try gpu.gpa.alloc(u16, rows * index.heads * index.dim);
        defer gpu.gpa.free(q);
        const iw = try gpu.gpa.alloc(u16, rows * index.heads);
        defer gpu.gpa.free(iw);
        const key_data = try gpu.gpa.alloc(u16, nkeys * index.dim);
        defer gpu.gpa.free(key_data);
        for (q) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (iw) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (key_data) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        const positions = try gpu.gpa.alloc(i64, rows);
        defer gpu.gpa.free(positions);
        @memset(positions, @intCast(2 * nkeys));
        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(key_data));
        defer dkeys.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
        defer dpos.free();
        var expected_scores = try cuda.DeviceBuffer.alloc(gpu.d, rows * nkeys * 4);
        defer expected_scores.free();
        var expected_tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 4);
        defer expected_tokens.free();
        var expected_counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
        defer expected_counts.free();
        var expected_records = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 8);
        defer expected_records.free();
        var reference = kernels.*;
        reference.score_kernel = .mma;
        reference.score_mma_rows = 4;
        try reference.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, expected_scores.ptr, rows, nkeys, nkeys, rank, 2);
        try reference.topk(stream, expected_scores.ptr, nkeys, expected_tokens.ptr, expected_counts.ptr, expected_records.ptr, dpos.ptr, rows, nkeys, index.top_k, rank, 2);
        try stream.synchronize();
        const expected_score_bytes = try download(gpu, expected_scores);
        defer gpu.gpa.free(expected_score_bytes);
        const expected_token_bytes = try download(gpu, expected_tokens);
        defer gpu.gpa.free(expected_token_bytes);
        const expected_count_bytes = try download(gpu, expected_counts);
        defer gpu.gpa.free(expected_count_bytes);
        for ([_]usize{ 8, 16 }) |rows_per_cta| {
            var actual_scores = try cuda.DeviceBuffer.alloc(gpu.d, rows * nkeys * 4);
            defer actual_scores.free();
            var actual_tokens = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 4);
            defer actual_tokens.free();
            var actual_counts = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
            defer actual_counts.free();
            var actual_records = try cuda.DeviceBuffer.alloc(gpu.d, rows * index.top_k * 8);
            defer actual_records.free();
            var actual = kernels.*;
            actual.score_kernel = .mma;
            actual.score_mma_rows = rows_per_cta;
            try actual.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, actual_scores.ptr, rows, nkeys, nkeys, rank, 2);
            try actual.topk(stream, actual_scores.ptr, nkeys, actual_tokens.ptr, actual_counts.ptr, actual_records.ptr, dpos.ptr, rows, nkeys, index.top_k, rank, 2);
            try stream.synchronize();
            const actual_score_bytes = try download(gpu, actual_scores);
            defer gpu.gpa.free(actual_score_bytes);
            const actual_token_bytes = try download(gpu, actual_tokens);
            defer gpu.gpa.free(actual_token_bytes);
            const actual_count_bytes = try download(gpu, actual_counts);
            defer gpu.gpa.free(actual_count_bytes);
            try expect(std.mem.eql(u8, expected_score_bytes, actual_score_bytes), "index MMA scores rows={d} keys={d} rank={d} CTA={d}", .{ rows, nkeys, rank, rows_per_cta });
            try expect(std.mem.eql(u8, expected_token_bytes, actual_token_bytes) and std.mem.eql(u8, expected_count_bytes, actual_count_bytes), "index MMA top-k rows={d} keys={d} rank={d} CTA={d}", .{ rows, nkeys, rank, rows_per_cta });
        }
    };
    pass("glm53-index: 8/16-row CTA scores and top-k bit-exact vs 4 rows (rows 1/7/64/512, contexts 2k/16k, both ranks)", .{});
}

fn scoresQsParityCases(gpu: anytype, stream: cuda.Stream, kernels: *const index.Index) !void {
    const Case = struct { rows: usize, keys: usize, causal: bool };
    const cases = [_]Case{
        .{ .rows = 1, .keys = 64, .causal = false },
        .{ .rows = 7, .keys = 1000, .causal = false },
        .{ .rows = 64, .keys = 8192, .causal = true },
        .{ .rows = 100, .keys = 4097, .causal = true },
        .{ .rows = 512, .keys = 16384, .causal = true },
        .{ .rows = 512, .keys = 65536, .causal = true },
    };
    var prng = std.Random.DefaultPrng.init(0x51535);
    const rnd = prng.random();
    for ([_]usize{ 4, 8, 16 }) |rows_per_cta| {
        var release = kernels.*;
        release.score_kernel = .mma;
        release.score_mma_rows = rows_per_cta;
        release.score_qs = 0;
        for ([_]u8{ 2, 3, 4, 5, 6 }) |mode| {
            var qs = kernels.*;
            qs.score_kernel = .mma;
            qs.score_mma_rows = rows_per_cta;
            qs.score_qs = mode;
            for (cases) |case| for ([_]usize{ 0, 1 }) |rank| {
                const q = try gpu.gpa.alloc(u16, case.rows * index.heads * index.dim);
                defer gpu.gpa.free(q);
                const iw = try gpu.gpa.alloc(u16, case.rows * index.heads);
                defer gpu.gpa.free(iw);
                const key_data = try gpu.gpa.alloc(u16, case.keys * index.dim);
                defer gpu.gpa.free(key_data);
                for (q) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
                for (iw) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
                for (key_data) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
                const positions = try gpu.gpa.alloc(i64, case.rows);
                defer gpu.gpa.free(positions);
                if (case.causal) {
                    const p0 = 2 * case.keys - case.rows;
                    for (positions, 0..) |*position, row| position.* = @intCast(p0 + row);
                } else {
                    @memset(positions, @intCast(2 * case.keys));
                }
                var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
                defer dqi.free();
                var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
                defer diw.free();
                var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(key_data));
                defer dkeys.free();
                var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
                defer dpos.free();
                const score_bytes = case.rows * case.keys * 4;
                var release_scores = try cuda.DeviceBuffer.alloc(gpu.d, score_bytes);
                defer release_scores.free();
                var qs_scores = try cuda.DeviceBuffer.alloc(gpu.d, score_bytes);
                defer qs_scores.free();
                var release_tokens = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * index.top_k * 4);
                defer release_tokens.free();
                var qs_tokens = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * index.top_k * 4);
                defer qs_tokens.free();
                var release_counts = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * 4);
                defer release_counts.free();
                var qs_counts = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * 4);
                defer qs_counts.free();
                var release_records = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * index.top_k * 8);
                defer release_records.free();
                var qs_records = try cuda.DeviceBuffer.alloc(gpu.d, case.rows * index.top_k * 8);
                defer qs_records.free();
                try gpu.d.check(gpu.d.api.cuMemsetD8_v2(release_scores.ptr, 0x7f, release_scores.len), "cuMemsetD8");
                try gpu.d.check(gpu.d.api.cuMemsetD8_v2(qs_scores.ptr, 0x7f, qs_scores.len), "cuMemsetD8");
                try release.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, release_scores.ptr, case.rows, case.keys, case.keys, rank, 2);
                try release.topk(stream, release_scores.ptr, case.keys, release_tokens.ptr, release_counts.ptr, release_records.ptr, dpos.ptr, case.rows, case.keys, index.top_k, rank, 2);
                try qs.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, qs_scores.ptr, case.rows, case.keys, case.keys, rank, 2);
                try qs.topk(stream, qs_scores.ptr, case.keys, qs_tokens.ptr, qs_counts.ptr, qs_records.ptr, dpos.ptr, case.rows, case.keys, index.top_k, rank, 2);
                try stream.synchronize();
                const release_score_bytes = try download(gpu, release_scores);
                defer gpu.gpa.free(release_score_bytes);
                const qs_score_bytes = try download(gpu, qs_scores);
                defer gpu.gpa.free(qs_score_bytes);
                const release_token_bytes = try download(gpu, release_tokens);
                defer gpu.gpa.free(release_token_bytes);
                const qs_token_bytes = try download(gpu, qs_tokens);
                defer gpu.gpa.free(qs_token_bytes);
                const release_count_bytes = try download(gpu, release_counts);
                defer gpu.gpa.free(release_count_bytes);
                const qs_count_bytes = try download(gpu, qs_counts);
                defer gpu.gpa.free(qs_count_bytes);
                try expect(std.mem.eql(u8, release_score_bytes, qs_score_bytes), "index QS scores rows={d} keys={d} R={d} mode={d} rank={d}", .{ case.rows, case.keys, rows_per_cta, mode, rank });
                try expect(std.mem.eql(u8, release_token_bytes, qs_token_bytes) and std.mem.eql(u8, release_count_bytes, qs_count_bytes), "index QS top-k rows={d} keys={d} R={d} mode={d} rank={d}", .{ case.rows, case.keys, rows_per_cta, mode, rank });
                if (case.rows == 512 and case.keys == 65536 and mode == 5) {
                    try gpu.d.check(gpu.d.api.cuMemsetD8_v2(qs_scores.ptr, 0x7f, qs_scores.len), "cuMemsetD8");
                    try qs.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, qs_scores.ptr, case.rows, case.keys, case.keys, rank, 2);
                    try stream.synchronize();
                    const repeated_score_bytes = try download(gpu, qs_scores);
                    defer gpu.gpa.free(repeated_score_bytes);
                    try expect(std.mem.eql(u8, qs_score_bytes, repeated_score_bytes), "index QS repeated scores rows=512 keys=65536 R={d} mode=5 rank={d}", .{ rows_per_cta, rank });
                }
            };
        }
    }
    pass("glm53-index: QS modes 2/3/4/5/6 scores, top-k tokens/counts, sentinels, and mode-5 repeat parity for (1,64), (7,1000), (64,8192), (100,4097), (512,16384), (512,65536), both ranks and R=4/8/16", .{});
}

/// `tf-cuda-test glm53-index-bench`: scalar vs tiled scores and the radix top-k; decode at 262k (131072 rank-local
/// keys, rows 1 and 8) and a prefill piece at 32k (16384 keys, 256 rows).
pub fn bench(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_index);
    defer module.unload();
    const base = try index.Index.load(module, 4);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const Case = struct { rows: usize, keys: usize };
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    for ([_]Case{ .{ .rows = 1, .keys = 131072 }, .{ .rows = 8, .keys = 131072 }, .{ .rows = 256, .keys = 16384 } }) |c| {
        const q = try gpu.gpa.alloc(u16, c.rows * index.heads * index.dim);
        defer gpu.gpa.free(q);
        const iw = try gpu.gpa.alloc(u16, c.rows * index.heads);
        defer gpu.gpa.free(iw);
        const keys = try gpu.gpa.alloc(u16, c.keys * index.dim);
        defer gpu.gpa.free(keys);
        for (q) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (iw) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (keys) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        const positions = try gpu.gpa.alloc(i64, c.rows);
        defer gpu.gpa.free(positions);
        for (positions) |*p| p.* = @intCast(2 * c.keys);
        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(keys));
        defer dkeys.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
        defer dpos.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, c.rows * c.keys * 4);
        defer out.free();
        for ([_]index.ScoreKernel{ .scalar, .mma }) |kind| {
            var k = base;
            k.score_kernel = kind;
            const reps: usize = if (c.rows == 256 and kind == .scalar) 3 else 20;
            for (0..3) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, c.rows, c.keys, c.keys, 0, 2);
            try e0.record(stream);
            for (0..reps) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, c.rows, c.keys, c.keys, 0, 2);
            try e1.record(stream);
            try stream.synchronize();
            std.debug.print("BENCH index scores {t} rows={d} keys={d}: {d:.1} us\n", .{ kind, c.rows, c.keys, (try e0.elapsedMs(e1)) * 1000.0 / @as(f32, @floatFromInt(reps)) });
        }
        // the radix top-2048 over the scores just written (one cluster of <= 8 CTAs a row)
        var tokens = try cuda.DeviceBuffer.alloc(gpu.d, c.rows * index.top_k * 4);
        defer tokens.free();
        var counts = try cuda.DeviceBuffer.alloc(gpu.d, c.rows * 4);
        defer counts.free();
        var records = try cuda.DeviceBuffer.alloc(gpu.d, c.rows * index.top_k * 8);
        defer records.free();
        for (0..3) |_| try base.topk(stream, out.ptr, c.keys, tokens.ptr, counts.ptr, records.ptr, dpos.ptr, c.rows, c.keys, index.top_k, 0, 2);
        try e0.record(stream);
        for (0..20) |_| try base.topk(stream, out.ptr, c.keys, tokens.ptr, counts.ptr, records.ptr, dpos.ptr, c.rows, c.keys, index.top_k, 0, 2);
        try e1.record(stream);
        try stream.synchronize();
        std.debug.print("BENCH index topk-2048 rows={d} keys={d}: {d:.1} us\n", .{ c.rows, c.keys, (try e0.elapsedMs(e1)) * 1000.0 / 20.0 });
    }
    for ([_]usize{1024, 8192}) |keys| for ([_]usize{4, 8, 16}) |rows_per_cta| {
        const rows: usize = 512;
        const q = try gpu.gpa.alloc(u16, rows * index.heads * index.dim);
        defer gpu.gpa.free(q);
        const iw = try gpu.gpa.alloc(u16, rows * index.heads);
        defer gpu.gpa.free(iw);
        const key_data = try gpu.gpa.alloc(u16, keys * index.dim);
        defer gpu.gpa.free(key_data);
        for (q) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (iw) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (key_data) |*v| v.* = bf16Bits(rnd.float(f32) * 2 - 1);
        const positions = try gpu.gpa.alloc(i64, rows);
        defer gpu.gpa.free(positions);
        @memset(positions, @intCast(2 * keys));
        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(key_data));
        defer dkeys.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
        defer dpos.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * keys * 4);
        defer out.free();
        var k = base;
        k.score_mma_rows = rows_per_cta;
        for (0..2) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, rows, keys, keys, 0, 2);
        try e0.record(stream);
        for (0..5) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, rows, keys, keys, 0, 2);
        try e1.record(stream);
        try stream.synchronize();
        std.debug.print("BENCH index scores rows=512 context={d} mma_rows={d}: {d:.1} us\n", .{ keys * 2, rows_per_cta, (try e0.elapsedMs(e1)) * 1000.0 / 5.0 });
    };
    for ([_]usize{ 64, 512 }) |rows| for ([_]usize{ 8192, 32768, 65536 }) |keys| {
        const q = try gpu.gpa.alloc(u16, rows * index.heads * index.dim);
        defer gpu.gpa.free(q);
        const iw = try gpu.gpa.alloc(u16, rows * index.heads);
        defer gpu.gpa.free(iw);
        const key_data = try gpu.gpa.alloc(u16, keys * index.dim);
        defer gpu.gpa.free(key_data);
        for (q) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (iw) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
        for (key_data) |*value| value.* = bf16Bits(rnd.float(f32) * 2 - 1);
        const positions = try gpu.gpa.alloc(i64, rows);
        defer gpu.gpa.free(positions);
        @memset(positions, @intCast(2 * keys));
        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dkeys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(key_data));
        defer dkeys.free();
        var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
        defer dpos.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * keys * 4);
        defer out.free();
        var us: [6]f64 = @splat(0.0);
        for (0..6) |variant| {
            var k = base;
            k.score_kernel = .mma;
            k.score_mma_rows = 16;
            k.score_qs = switch (variant) {
                0 => 0,
                1 => 2,
                2 => 3,
                3 => 4,
                4 => 5,
                else => 6,
            };
            for (0..2) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, rows, keys, keys, 0, 2);
            try e0.record(stream);
            for (0..10) |_| try k.scores(stream, dqi.ptr, diw.ptr, index.heads, 0, dkeys.ptr, dpos.ptr, out.ptr, rows, keys, keys, 0, 2);
            try e1.record(stream);
            try stream.synchronize();
            us[variant] = @as(f64, @floatCast((try e0.elapsedMs(e1)) * 100.0));
        }
        const flops: f64 = @floatFromInt(rows * keys * index.heads * index.dim * 2);
        const rate0 = flops / (us[0] * 1.0e-6) / 1.0e12;
        const rate1 = flops / (us[1] * 1.0e-6) / 1.0e12;
        const rate2 = flops / (us[2] * 1.0e-6) / 1.0e12;
        const rate3 = flops / (us[3] * 1.0e-6) / 1.0e12;
        const rate4 = flops / (us[4] * 1.0e-6) / 1.0e12;
        const rate5 = flops / (us[5] * 1.0e-6) / 1.0e12;
        std.debug.print("BENCH glm53-index-qs rows={d} keys={d} | mma16 {d:.1} us ({d:.1} TFLOP/s) | qs2 {d:.1} ({d:.1}) | qs3 {d:.1} ({d:.1})\n", .{ rows, keys, us[0], rate0, us[1], rate1, us[2], rate2 });
        std.debug.print("BENCH glm53-index-qs2 rows={d} keys={d} | qs3 {d:.1} us ({d:.1} TFLOP/s) | k64s2 {d:.1} ({d:.1}) | k32s2 {d:.1} ({d:.1}) | k32s3 {d:.1} ({d:.1})\n", .{ rows, keys, us[2], rate2, us[3], rate3, us[4], rate4, us[5], rate5 });
    };
}
