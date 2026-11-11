// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 fp32 head GPU parity and host oracle checks (WP10).
const std = @import("std");
const cuda = @import("cuda");
const sampler = @import("sampler.zig");

test "GLM-5.3 head module declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_head);
    defer module.unload();
    const head = try sampler.Head.resolve(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const vocab = 8192;
    const rows = 3;
    const logits = try gpu.gpa.alloc(f32, vocab * rows);
    defer gpu.gpa.free(logits);
    for (logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 101)) - 50)) / 8;
    logits[11] = 7;
    logits[vocab + 7] = 9;
    logits[vocab * 2 + 19] = 11;
    var dlogits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(logits));
    defer dlogits.free();
    var values = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer values.free();
    var ids = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer ids.free();
    try head.top1(stream, dlogits.ptr, vocab, 0, values.ptr, ids.ptr, rows);
    try stream.synchronize();
    const got_values = try download(gpu, values);
    defer gpu.gpa.free(got_values);
    const got_ids = try download(gpu, ids);
    defer gpu.gpa.free(got_ids);
    const va = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got_values));
    const ia = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, got_ids));
    for (0..rows) |r| {
        const want = sampler.top1Ref(logits[r * vocab ..][0..vocab], 0);
        try expect(ia[r] == want.id and @as(u32, @bitCast(va[r])) == @as(u32, @bitCast(want.value)), "head top1 row {d}", .{r});
        try expect(ia[r] == ([_]u32{ 11, 7, 19 })[r], "head synthetic winner row {d}", .{r});
    }
    pass("glm53-head: fp32 top1 on {d} rows x {d} logits matches host bits", .{ rows, vocab });
    for (0..rows) |r| {
        var alone_values = try cuda.DeviceBuffer.alloc(gpu.d, 4);
        defer alone_values.free();
        var alone_ids = try cuda.DeviceBuffer.alloc(gpu.d, 4);
        defer alone_ids.free();
        try head.top1(stream, dlogits.ptr + r * vocab * 4, vocab, 0, alone_values.ptr, alone_ids.ptr, 1);
        try stream.synchronize();
        const av = try download(gpu, alone_values);
        defer gpu.gpa.free(av);
        const ai = try download(gpu, alone_ids);
        defer gpu.gpa.free(ai);
        try expect(std.mem.eql(u8, av, got_values[r * 4 ..][0..4]) and std.mem.eql(u8, ai, got_ids[r * 4 ..][0..4]), "head top1 row-alone vs batch row={d}", .{r});
    }
    try candidateCases(gpu, stream, &head, dlogits.ptr, logits, vocab);
    try candidateEdgeCases(gpu, stream, &head);
    try candidateV2Cases(gpu, stream, &head);
    try lseCase(gpu, stream, &head, dlogits.ptr, logits, vocab);
    try gateCase(gpu, stream, &head, dlogits.ptr, logits, vocab);
    try maskCase(gpu, stream, &head, logits, vocab);
}

fn maskCase(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head, logits: []const f32, vocab: usize) !void {
    const rows = 3;
    const offset = 8192;
    const words = 512;
    var bits = try gpu.gpa.alloc(u32, rows * words);
    defer gpu.gpa.free(bits);
    @memset(bits, 0);
    for (0..rows) |r| for (0..vocab) |i| {
        if ((i * 17 + r * 13) % 31 < 9) {
            const token = offset + i;
            bits[r * words + token / 32] |= @as(u32, 1) << @intCast(token % 32);
        }
    };
    for (0..rows) |r| {
        const first = offset + r;
        bits[r * words + first / 32] |= @as(u32, 1) << @intCast(first % 32);
    }
    var dlogits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(logits));
    defer dlogits.free();
    var dbits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bits));
    defer dbits.free();
    try head.mask(stream, dlogits.ptr, vocab, offset, dbits.ptr, @intCast(words), rows);
    try stream.synchronize();
    const got = try download(gpu, dlogits);
    defer gpu.gpa.free(got);
    const actual = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    var masked = try gpu.gpa.alloc(f32, rows * vocab);
    defer gpu.gpa.free(masked);
    for (0..rows) |r| for (0..vocab) |i| {
        const token = offset + i;
        const allowed = (bits[r * words + token / 32] >> @intCast(token % 32)) & 1 != 0;
        masked[r * vocab + i] = if (allowed) logits[r * vocab + i] else -std.math.inf(f32);
        try expect(@as(u32, @bitCast(actual[r * vocab + i])) == @as(u32, @bitCast(masked[r * vocab + i])), "head mask row={d} token={d}", .{ r, token });
    };
    var values = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer values.free();
    var ids = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer ids.free();
    try head.top1(stream, dlogits.ptr, vocab, offset, values.ptr, ids.ptr, rows);
    try stream.synchronize();
    const gotv = try download(gpu, values);
    defer gpu.gpa.free(gotv);
    const goti = try download(gpu, ids);
    defer gpu.gpa.free(goti);
    const actual_v = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, gotv));
    const actual_i = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, goti));
    for (0..rows) |r| {
        const want = sampler.top1Ref(masked[r * vocab ..][0..vocab], offset);
        try expect(actual_i[r] == want.id and @as(u32, @bitCast(actual_v[r])) == @as(u32, @bitCast(want.value)), "masked head top1 row={d}", .{r});
    }
    const empty = try gpu.gpa.alloc(u32, words);
    defer gpu.gpa.free(empty);
    @memset(empty, 0);
    var dempty = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(empty));
    defer dempty.free();
    var one_value = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer one_value.free();
    var one_id = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer one_id.free();
    try head.mask(stream, dlogits.ptr, vocab, offset, dempty.ptr, @intCast(words), 1);
    try head.top1(stream, dlogits.ptr, vocab, offset, one_value.ptr, one_id.ptr, 1);
    try stream.synchronize();
    const empty_value = try download(gpu, one_value);
    defer gpu.gpa.free(empty_value);
    try expect(std.mem.bytesToValue(f32, empty_value) == -std.math.inf(f32), "fully masked shard has -inf top1", .{});
    pass("glm53-head: row masks preserve allowed logits and restrict top1 on a rank-1 shard", .{});
}

fn candidateCases(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head, logits_ptr: u64, logits: []const f32, vocab: usize) !void {
    const rows = 3;
    // Baselines per k live across the variant loop (variant 0 writes them; 1 and 2 compare), as candidateEdgeCases.
    const ks = [_]usize{ 1, 8, 64, 2048 };
    var baseline_values: [ks.len]?[]u8 = @splat(null);
    var baseline_ids: [ks.len]?[]u8 = @splat(null);
    defer for (baseline_values, baseline_ids) |v, i| {
        if (v) |b| gpu.gpa.free(b);
        if (i) |b| gpu.gpa.free(b);
    };
    for (0..3) |variant| {
        const fast = variant == 1;
        const v2 = variant == 2;
        var selected_head = head.*;
        selected_head.fast = fast;
        selected_head.v2 = v2;
        for (ks, 0..) |k, ki| {
        var dvalues = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer dvalues.free();
        var dids = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer dids.free();
        var scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(rows, k));
        defer scratch.free();
        try selected_head.candidates(stream, logits_ptr, vocab, 0, k, dvalues.ptr, dids.ptr, scratch.ptr, scratch.len, rows);
        try stream.synchronize();
        const gotv = try download(gpu, dvalues);
        defer gpu.gpa.free(gotv);
        const goti = try download(gpu, dids);
        defer gpu.gpa.free(goti);
            if (variant == 0) {
                baseline_values[ki] = try gpu.gpa.dupe(u8, gotv);
                baseline_ids[ki] = try gpu.gpa.dupe(u8, goti);
            } else {
                try expect(std.mem.eql(u8, baseline_values[ki].?, gotv) and std.mem.eql(u8, baseline_ids[ki].?, goti), "head candidate variant byte parity variant={d} k={d}", .{ variant, k });
            }
        const gv = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, gotv));
        const gi = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, goti));
        for (0..rows) |r| {
            const want = try sampler.topCandidatesRef(gpu.gpa, logits[r * vocab ..][0..vocab], 0, k);
            defer gpu.gpa.free(want);
            for (0..k) |i| try expect(gi[r * k + i] == want[i].id and @as(u32, @bitCast(gv[r * k + i])) == @as(u32, @bitCast(want[i].value)), "head candidates row={d} k={d} item={d}", .{ r, k, i });
            var alone_values = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
            defer alone_values.free();
            var alone_ids = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
            defer alone_ids.free();
            var alone_scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(1, k));
            defer alone_scratch.free();
            const row_ptr = logits_ptr + r * vocab * @sizeOf(f32);
            try selected_head.candidates(stream, row_ptr, vocab, 0, k, alone_values.ptr, alone_ids.ptr, alone_scratch.ptr, alone_scratch.len, 1);
            try stream.synchronize();
            const alone_v = try download(gpu, alone_values);
            defer gpu.gpa.free(alone_v);
            const alone_i = try download(gpu, alone_ids);
            defer gpu.gpa.free(alone_i);
            try expect(std.mem.eql(u8, alone_v, gotv[r * k * 4 ..][0 .. k * 4]) and std.mem.eql(u8, alone_i, goti[r * k * 4 ..][0 .. k * 4]), "head candidate row invariance row={d} k={d}", .{ r, k });
        }
        pass("glm53-head: {s} candidates k={d}, {d} entries match host order", .{ if (v2) (if (k >= 64) "v2" else "v2 fallback") else if (fast) "fast" else "baseline", k, k });
        }
    }
}

fn candidateEdgeCases(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head) !void {
    const vocab = 512;
    const max_rows = 64;
    var ties: [max_rows * vocab]f32 = @splat(1.0);
    var nan_mix: [max_rows * vocab]f32 = undefined;
    for (&nan_mix, 0..) |*value, i| {
        value.* = if (i % 11 == 0) std.math.nan(f32) else @as(f32, @floatFromInt((i * 17) % 23)) / 4;
    }
    for ([_][]const f32{ &ties, &nan_mix }) |fixture| {
        var dlogits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fixture));
        defer dlogits.free();
        for ([_]usize{ 1, 2, 8, 64 }) |rows| for ([_]usize{ 1, 2, 8, 64 }) |k| {
            var baseline_values: ?[]u8 = null;
            var baseline_ids: ?[]u8 = null;
            for (0..3) |variant| {
                const fast = variant == 1;
                const v2 = variant == 2;
                var selected_head = head.*;
                selected_head.fast = fast;
                selected_head.v2 = v2;
                var dvalues = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
                defer dvalues.free();
                var dids = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
                defer dids.free();
                var scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(rows, k));
                defer scratch.free();
                try selected_head.candidates(stream, dlogits.ptr, vocab, 0, k, dvalues.ptr, dids.ptr, scratch.ptr, scratch.len, rows);
                try stream.synchronize();
                const got_values = try download(gpu, dvalues);
                defer gpu.gpa.free(got_values);
                const got_ids = try download(gpu, dids);
                defer gpu.gpa.free(got_ids);
                for (0..rows) |r| {
                    const want = try sampler.topCandidatesRef(gpu.gpa, fixture[r * vocab ..][0..vocab], 0, k);
                    defer gpu.gpa.free(want);
                    for (0..k) |i| {
                        const got_value = std.mem.readInt(u32, got_values[(r * k + i) * 4 ..][0..4], .little);
                        const got_id = std.mem.readInt(u32, got_ids[(r * k + i) * 4 ..][0..4], .little);
                        try expect(got_id == want[i].id and got_value == @as(u32, @bitCast(want[i].value)), "head {s} edge rows={d} k={d} row={d} item={d}", .{ if (fast) "fast" else "baseline", rows, k, r, i });
                    }
                }
                if (variant == 0) {
                    baseline_values = try gpu.gpa.dupe(u8, got_values);
                    baseline_ids = try gpu.gpa.dupe(u8, got_ids);
                } else {
                    try expect(std.mem.eql(u8, baseline_values.?, got_values) and std.mem.eql(u8, baseline_ids.?, got_ids), "head candidate edge output byte parity variant={d} rows={d} k={d}", .{ variant, rows, k });
                }
            }
            gpu.gpa.free(baseline_values.?);
            gpu.gpa.free(baseline_ids.?);
        };
    }
    pass("glm53-head: baseline, fast and v2 (k >= 64) match ties, NaNs, short local lists and row windows", .{});
}

fn candidateV2Cases(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head) !void {
    const vocab = 77_440;
    for ([_]usize{ 1, 2, 4, 8, 64, 65 }) |rows| for ([_]usize{ 64, 256, 2048 }) |k| for (0..7) |fixture_kind| {
        const logits = try gpu.gpa.alloc(f32, rows * vocab);
        defer gpu.gpa.free(logits);
        for (0..rows) |row| for (0..vocab) |i| {
            const at = row * vocab + i;
            logits[at] = switch (fixture_kind) {
                0 => @as(f32, @floatFromInt(@as(i64, @intCast((i *% 2_654_435_761 +% row *% 97_531) % 3_501)))) / 100 - 10,
                1 => 4.25,
                2 => if (i < 5_000) 0.0 else -@as(f32, @floatFromInt(i - 4_999)),
                3 => switch (i % 4) { 0 => 0.0, 1 => -0.0, 2 => 2.0, else => -2.0 },
                4 => switch (i % 19) { 0 => std.math.inf(f32), 1 => -std.math.inf(f32), 2 => std.math.nan(f32), else => @as(f32, @floatFromInt(i % 37)) / 8 },
                5 => std.math.nan(f32),
                6 => if (i < 3) @as(f32, @floatFromInt(5 - @as(i32, @intCast(i)))) else std.math.nan(f32),
                else => unreachable,
            };
        };
        var dlogits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(logits));
        defer dlogits.free();
        var baseline_values = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer baseline_values.free();
        var baseline_ids = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer baseline_ids.free();
        var baseline_scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(rows, k));
        defer baseline_scratch.free();
        var v2_values = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer v2_values.free();
        var v2_ids = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 4);
        defer v2_ids.free();
        var v2_scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(rows, k));
        defer v2_scratch.free();
        var baseline_head = head.*;
        baseline_head.fast = false;
        baseline_head.v2 = false;
        var v2_head = head.*;
        v2_head.fast = true;
        v2_head.v2 = true;
        try baseline_head.candidates(stream, dlogits.ptr, vocab, 0, k, baseline_values.ptr, baseline_ids.ptr, baseline_scratch.ptr, baseline_scratch.len, rows);
        try v2_head.candidates(stream, dlogits.ptr, vocab, 0, k, v2_values.ptr, v2_ids.ptr, v2_scratch.ptr, v2_scratch.len, rows);
        try stream.synchronize();
        const base_values = try download(gpu, baseline_values);
        defer gpu.gpa.free(base_values);
        const base_ids = try download(gpu, baseline_ids);
        defer gpu.gpa.free(base_ids);
        const got_values = try download(gpu, v2_values);
        defer gpu.gpa.free(got_values);
        const got_ids = try download(gpu, v2_ids);
        defer gpu.gpa.free(got_ids);
        try expect(std.mem.eql(u8, base_values, got_values) and std.mem.eql(u8, base_ids, got_ids), "CAND_V2 matches baseline bytes rows={d} k={d} fixture={d}", .{ rows, k, fixture_kind });
        for (0..rows) |row| {
            if (fixture_kind == 5) {
                for (0..k) |i| {
                    const value_bits = std.mem.readInt(u32, got_values[(row * k + i) * 4 ..][0..4], .little);
                    const token = std.mem.readInt(u32, got_ids[(row * k + i) * 4 ..][0..4], .little);
                    try expect(value_bits == @as(u32, @bitCast(-std.math.inf(f32))) and token == std.math.maxInt(u32), "CAND_V2 all-NaN sentinel row={d} k={d} item={d}", .{ row, k, i });
                }
                continue;
            }
            const want = try sampler.topCandidatesRef(gpu.gpa, logits[row * vocab ..][0..vocab], 0, k);
            defer gpu.gpa.free(want);
            for (0..want.len) |i| {
                const value_bits = std.mem.readInt(u32, got_values[(row * k + i) * 4 ..][0..4], .little);
                const token = std.mem.readInt(u32, got_ids[(row * k + i) * 4 ..][0..4], .little);
                try expect(token == want[i].id and value_bits == @as(u32, @bitCast(want[i].value)), "CAND_V2 host reference row={d} k={d} fixture={d} item={d}", .{ row, k, fixture_kind, i });
            }
            for (want.len..k) |i| {
                const value_bits = std.mem.readInt(u32, got_values[(row * k + i) * 4 ..][0..4], .little);
                const token = std.mem.readInt(u32, got_ids[(row * k + i) * 4 ..][0..4], .little);
                try expect(value_bits == @as(u32, @bitCast(-std.math.inf(f32))) and token == std.math.maxInt(u32), "CAND_V2 short-row sentinel row={d} k={d} fixture={d} item={d}", .{ row, k, fixture_kind, i });
            }
        }
    };
    try candidateV2GraphCase(gpu, stream, head);
    pass("glm53-head: CAND_V2 matches baseline and host order at k>=64, including the 64-row tile boundary", .{});
}

fn candidateV2GraphCase(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head) !void {
    const vocab = 77_440;
    const k = 64;
    const logits = try gpu.gpa.alloc(f32, vocab);
    defer gpu.gpa.free(logits);
    for (logits, 0..) |*value, i| value.* = @as(f32, @floatFromInt((i *% 17) % 101)) / 8;
    var dlogits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(logits));
    defer dlogits.free();
    var eager_values = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer eager_values.free();
    var eager_ids = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer eager_ids.free();
    var eager_scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(1, k));
    defer eager_scratch.free();
    var graph_values = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer graph_values.free();
    var graph_ids = try cuda.DeviceBuffer.alloc(gpu.d, k * 4);
    defer graph_ids.free();
    var graph_scratch = try cuda.DeviceBuffer.alloc(gpu.d, try sampler.Head.candidateScratchBytes(1, k));
    defer graph_scratch.free();
    var v2_head = head.*;
    v2_head.fast = true;
    v2_head.v2 = true;
    try v2_head.candidates(stream, dlogits.ptr, vocab, 0, k, eager_values.ptr, eager_ids.ptr, eager_scratch.ptr, eager_scratch.len, 1);
    try stream.synchronize();
    try cuda.graph.beginCapture(stream, .thread_local);
    try v2_head.candidates(stream, dlogits.ptr, vocab, 0, k, graph_values.ptr, graph_ids.ptr, graph_scratch.ptr, graph_scratch.len, 1);
    var captured = try cuda.graph.endCapture(stream);
    defer captured.deinit();
    var graph_exec = try captured.instantiate();
    defer graph_exec.deinit();
    try graph_exec.launchOn(stream);
    try stream.synchronize();
    const eager_value_bytes = try download(gpu, eager_values);
    defer gpu.gpa.free(eager_value_bytes);
    const eager_id_bytes = try download(gpu, eager_ids);
    defer gpu.gpa.free(eager_id_bytes);
    const graph_value_bytes = try download(gpu, graph_values);
    defer gpu.gpa.free(graph_value_bytes);
    const graph_id_bytes = try download(gpu, graph_ids);
    defer gpu.gpa.free(graph_id_bytes);
    try expect(std.mem.eql(u8, eager_value_bytes, graph_value_bytes) and std.mem.eql(u8, eager_id_bytes, graph_id_bytes), "CAND_V2 graph replay equals eager output", .{});
}

fn lseCase(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head, logits_ptr: u64, logits: []const f32, vocab: usize) !void {
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 3 * 8);
    defer out.free();
    try head.lse(stream, logits_ptr, vocab, out.ptr, 3);
    try stream.synchronize();
    const got = try download(gpu, out);
    defer gpu.gpa.free(got);
    const pair = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    // The host oracle matches CUDA's 256 strided lane accumulators and binary tree. At 8192
    // values each lane has 32 sequential positive additions (gamma_31), then the tree has 8
    // levels (gamma_8); allowing 1.5u for expf vs host exp gives about 41u relative error.
    // For this fixture's LSE sums (about 45 and 327), 32 output ULPs cover that bound.
    var mismatches: usize = 0;
    var max_ulp: u32 = 0;
    for (0..3) |r| {
        const host = sampler.partialLse(logits[r * vocab ..][0..vocab]);
        const max_equal = @as(u32, @bitCast(pair[r * 2])) == @as(u32, @bitCast(host.max));
        const ulp = f32Ulp(pair[r * 2 + 1], host.sum);
        max_ulp = @max(max_ulp, ulp);
        if (!max_equal or ulp > 32) {
            if (mismatches < 8) std.debug.print("head LSE mismatch row={d}: GPU max={d} (0x{x:0>8}), sum={d} (0x{x:0>8}); host max={d} (0x{x:0>8}), sum={d} (0x{x:0>8}), sum ulp={d}\n", .{ r, pair[r * 2], @as(u32, @bitCast(pair[r * 2])), pair[r * 2 + 1], @as(u32, @bitCast(pair[r * 2 + 1])), host.max, @as(u32, @bitCast(host.max)), host.sum, @as(u32, @bitCast(host.sum)), ulp });
            mismatches += 1;
        }
    }
    try expect(mismatches == 0, "head LSE exceeded 32 fp32 ULP at {d} rows (max sum distance {d})", .{ mismatches, max_ulp });
    for (0..3) |r| {
        var alone = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 4);
        defer alone.free();
        try head.lse(stream, logits_ptr + r * vocab * 4, vocab, alone.ptr, 1);
        try stream.synchronize();
        const alone_bytes = try download(gpu, alone);
        defer gpu.gpa.free(alone_bytes);
        try expect(std.mem.eql(u8, alone_bytes, got[r * 8 ..][0..8]), "head LSE row-alone vs batch row={d}", .{r});
    }
    pass("glm53-head: CUDA-tree LSE sum within 32 fp32 ULP on {d} rows", .{3});
}

fn f32Ordered(value: f32) i64 {
    const bits: u32 = @bitCast(value);
    const ordered = if (bits & 0x80000000 != 0) ~bits else bits ^ 0x80000000;
    return @intCast(ordered);
}

fn f32Ulp(a: f32, b: f32) u32 {
    return @intCast(@abs(f32Ordered(a) - f32Ordered(b)));
}

fn gateCase(gpu: anytype, stream: cuda.Stream, head: *const sampler.Head, logits_ptr: u64, logits: []const f32, vocab: usize) !void {
    const rows = 3;
    const probes = 4;
    const ids_host = [_]u32{ 11, 7, 8191, 9000, 7, 11, 13, 9000, 19, 7, 11, 9000 };
    var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ids_host));
    defer dids.free();
    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, rows * probes * 4);
    defer gathered.free();
    try head.gateGather(stream, logits_ptr, dids.ptr, vocab, 0, gathered.ptr, probes, rows);
    try stream.synchronize();
    const got_gathered = try download(gpu, gathered);
    defer gpu.gpa.free(got_gathered);
    const gathered_host = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got_gathered));
    for (0..rows * probes) |i| {
        const id = ids_host[i];
        const expected = if (id < vocab) logits[(i / probes) * vocab + id] else 0;
        try expect(@as(u32, @bitCast(gathered_host[i])) == @as(u32, @bitCast(expected)), "head gate gather probe={d}", .{i});
    }
    const lse_host = try gpu.gpa.alloc(f32, rows);
    defer gpu.gpa.free(lse_host);
    for (0..rows) |r| {
        const partial = sampler.partialLse(logits[r * vocab ..][0..vocab]);
        lse_host[r] = partial.max + @log(partial.sum);
    }
    var dlse = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(lse_host));
    defer dlse.free();
    var logp = try cuda.DeviceBuffer.alloc(gpu.d, rows * probes * 4);
    defer logp.free();
    var mass = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer mass.free();
    try head.gateMass(stream, gathered.ptr, dlse.ptr, logp.ptr, mass.ptr, probes, rows);
    try stream.synchronize();
    const got_logp = try download(gpu, logp);
    defer gpu.gpa.free(got_logp);
    const got_mass = try download(gpu, mass);
    defer gpu.gpa.free(got_mass);
    const actual_logp = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got_logp));
    const actual_mass = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got_mass));
    for (0..rows) |r| {
        var sum: f32 = 0;
        for (0..probes) |j| {
            const expected = gathered_host[r * probes + j] - lse_host[r];
            try expect(@abs(actual_logp[r * probes + j] - expected) < 1e-6, "head gate logp row={d} probe={d}", .{ r, j });
            sum += @exp(expected);
        }
        try expect(@abs(actual_mass[r] - @max(0, 1 - sum)) < 2e-6, "head gate mass row={d}", .{r});
    }
    for (0..rows) |r| {
        var alone_gather = try cuda.DeviceBuffer.alloc(gpu.d, probes * 4);
        defer alone_gather.free();
        try head.gateGather(stream, logits_ptr + r * vocab * 4, dids.ptr + r * probes * 4, vocab, 0, alone_gather.ptr, probes, 1);
        try stream.synchronize();
        const gathered_row = try download(gpu, alone_gather);
        defer gpu.gpa.free(gathered_row);
        try expect(std.mem.eql(u8, gathered_row, got_gathered[r * probes * 4 ..][0 .. probes * 4]), "head gate gather row-alone vs batch row={d}", .{r});
        var alone_lse = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(lse_host[r .. r + 1]));
        defer alone_lse.free();
        var alone_logp = try cuda.DeviceBuffer.alloc(gpu.d, probes * 4);
        defer alone_logp.free();
        var alone_mass = try cuda.DeviceBuffer.alloc(gpu.d, 4);
        defer alone_mass.free();
        try head.gateMass(stream, alone_gather.ptr, alone_lse.ptr, alone_logp.ptr, alone_mass.ptr, probes, 1);
        try stream.synchronize();
        const logp_row = try download(gpu, alone_logp);
        defer gpu.gpa.free(logp_row);
        const mass_row = try download(gpu, alone_mass);
        defer gpu.gpa.free(mass_row);
        try expect(std.mem.eql(u8, logp_row, got_logp[r * probes * 4 ..][0 .. probes * 4]) and std.mem.eql(u8, mass_row, got_mass[r * 4 ..][0..4]), "head gate mass row-alone vs batch row={d}", .{r});
    }
    pass("glm53-head: gate probe gather, logp and outside mass match host math", .{});
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

test "candidate host ordering has sentinel and tie behavior" {
    const scores = [_]f32{ 1, 1, -0.0, 0.0, -2 };
    const got = try sampler.candidateOutputRef(std.testing.allocator, &scores, 7, 8);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(u32, 7), got[0].id);
    try std.testing.expectEqual(@as(u32, 8), got[1].id);
    try std.testing.expectEqual(@as(u32, 9), got[2].id);
    try std.testing.expectEqual(@as(u32, 10), got[3].id);
    try std.testing.expectEqual(@as(u32, 0xffffffff), got[7].id);
}
