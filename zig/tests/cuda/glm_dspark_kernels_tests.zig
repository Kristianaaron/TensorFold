// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! DSpark-sized CUDA kernel checks. This file is registered in tf-cuda-test as glm-dspark-kernels.
const std = @import("std");
const cuda = @import("cuda");
/// The runner's pass line and byte comparison (zig/tests/cuda/check.zig, which lives in the runner's root module;
/// a second module over the same file is a Zig 0.17 error). The runner's `check.Gpu` arrives as `anytype`.
const check = struct {
    const Failed = error{TestFailed};

    fn pass(comptime fmt: []const u8, args: anytype) void {
        std.debug.print("PASS " ++ fmt ++ "\n", args);
    }

    fn sameBytes(what: []const u8, got: []const u8, want: []const u8) Failed!void {
        if (got.len != want.len) {
            std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ what, got.len, want.len });
            return error.TestFailed;
        }
        if (std.mem.eql(u8, got, want)) return;
        const first = std.mem.indexOfDiff(u8, got, want).?;
        std.debug.print("FAIL {s}: first difference at byte {d}\n", .{ what, first });
        return error.TestFailed;
    }
};

const Rows = 3;
const HalfVocab = 77_440;
const Chunk = 4096;
const Chunks = (HalfVocab + Chunk - 1) / Chunk;
const Top = 64;

fn bf(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn f32bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn better(_: void, a: Entry, b: Entry) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}
const Entry = struct { value: f32, id: i32 };

fn hostTop(a: std.mem.Allocator, row: []const u16, offset: i32, out: []i32) !void {
    const entries = try a.alloc(Entry, row.len);
    defer a.free(entries);
    for (row, 0..) |v, i| entries[i] = .{ .value = f32bf(v), .id = offset + @as(i32, @intCast(i)) };
    std.mem.sort(Entry, entries, {}, better);
    for (0..Top) |i| out[i] = entries[i].id;
    for (0..Top) |i| out[Top + i] = @bitCast(entries[i].value);
}

fn upload(gpu: anytype, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

fn randomBf16(a: std.mem.Allocator, n: usize, seed: u64) ![]u16 {
    const out = try a.alloc(u16, n);
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*x| x.* = bf(@as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -256, 256))) / 64.0);
    return out;
}

fn expectBf16Near(what: []const u8, got: []const u16, want: []const u16, tolerance: f32) !void {
    if (got.len != want.len) return error.BadShape;
    for (got, want, 0..) |g, w, i| {
        // Absolute below 1, relative above: one bf16 step at 8 is 0.0625 (drafter maths only; verify stays exact).
        if (@abs(f32bf(g) - f32bf(w)) > tolerance * @max(1.0, @abs(f32bf(w)))) {
            std.debug.print("FAIL {s} at {d}: got {d:.7}, expected {d:.7}\n", .{ what, i, f32bf(g), f32bf(w) });
            return error.TestFailed;
        }
    }
}

fn normRope64Host(x: []const u16, weight: []const u16, pos: i32, out: []u16) void {
    var sum: f32 = 0;
    for (x) |v| {
        const f = f32bf(v);
        sum += f * f;
    }
    const rinv = 1.0 / @sqrt(sum / 64.0 + 1e-5);
    for (0..32) |d| {
        const a = f32bf(x[d]);
        const b = f32bf(x[d + 32]);
        const na = f32bf(bf(f32bf(weight[d]) * f32bf(bf(a * rinv))));
        const nb = f32bf(bf(f32bf(weight[d + 32]) * f32bf(bf(b * rinv))));
        const inv = std.math.pow(f32, 8_000_000.0, -@as(f32, @floatFromInt(2 * d)) / 64.0);
        const angle = @as(f32, @floatFromInt(pos)) * inv;
        const c = @cos(angle);
        const sn = @sin(angle);
        out[d] = bf(na * c - nb * sn);
        out[d + 32] = bf(nb * c + na * sn);
    }
}

fn blockQkv64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const f = try module.function("df_block_qkv_glm");
    const R: usize = 8;
    const QH: usize = 64;
    const KVH: usize = 64;
    const HD: usize = 64;
    const KVW = KVH * HD;
    const ldq = (QH + 2 * KVH) * HD;
    const qkv_h = try randomBf16(a, R * ldq, 0xd5a4_201);
    defer a.free(qkv_h);
    const qn_h = try randomBf16(a, HD, 0xd5a4_202);
    defer a.free(qn_h);
    const kn_h = try randomBf16(a, HD, 0xd5a4_203);
    defer a.free(kn_h);
    const pos_h = [_]i32{ 0, 1, 127, 511, 1023, 1537, 2046, 3000 };
    var qkv = try upload(gpu, std.mem.sliceAsBytes(qkv_h));
    defer qkv.free();
    var qn = try upload(gpu, std.mem.sliceAsBytes(qn_h));
    defer qn.free();
    var kn = try upload(gpu, std.mem.sliceAsBytes(kn_h));
    defer kn.free();
    var pos = try upload(gpu, std.mem.sliceAsBytes(&pos_h));
    defer pos.free();
    var q = try cuda.DeviceBuffer.alloc(gpu.d, R * QH * HD * @sizeOf(u16));
    defer q.free();
    var k = try cuda.DeviceBuffer.alloc(gpu.d, R * KVW * @sizeOf(u16));
    defer k.free();
    var v = try cuda.DeviceBuffer.alloc(gpu.d, R * KVW * @sizeOf(u16));
    defer v.free();
    var args: cuda.Args = .{};
    args.add(qkv.ptr);
    args.add(@as(i32, ldq));
    args.add(qn.ptr);
    args.add(kn.ptr);
    args.add(pos.ptr);
    args.add(@as(f32, 1e-5));
    args.add(@as(f32, 8_000_000));
    args.add(@as(i32, QH));
    args.add(@as(i32, KVH));
    args.add(@as(i32, HD));
    args.add(@as(i32, KVW));
    args.add(q.ptr);
    args.add(k.ptr);
    args.add(v.ptr);
    try cuda.launch.launch(f, .{ .grid = .{ .x = R, .y = QH + KVH }, .block = .{ .x = HD / 2 } }, s, &args);
    try s.synchronize();
    const q_got = try a.alloc(u16, R * QH * HD);
    defer a.free(q_got);
    const k_got = try a.alloc(u16, R * KVW);
    defer a.free(k_got);
    const v_got = try a.alloc(u16, R * KVW);
    defer a.free(v_got);
    try q.download(0, std.mem.sliceAsBytes(q_got));
    try k.download(0, std.mem.sliceAsBytes(k_got));
    try v.download(0, std.mem.sliceAsBytes(v_got));
    const q_want = try a.alloc(u16, q_got.len);
    defer a.free(q_want);
    const k_want = try a.alloc(u16, k_got.len);
    defer a.free(k_want);
    const v_want = try a.alloc(u16, v_got.len);
    defer a.free(v_want);
    for (0..R) |r| {
        for (0..QH) |h| normRope64Host(qkv_h[r * ldq + h * HD ..][0..HD], qn_h, pos_h[r], q_want[r * QH * HD + h * HD ..][0..HD]);
        for (0..KVH) |h| {
            const kr = r * ldq + QH * HD + h * HD;
            normRope64Host(qkv_h[kr..][0..HD], kn_h, pos_h[r], k_want[r * KVW + h * HD ..][0..HD]);
            const vr = r * ldq + QH * HD + KVW + h * HD;
            @memcpy(v_want[r * KVW + h * HD ..][0..HD], qkv_h[vr..][0..HD]);
        }
    }
    try expectBf16Near("DSpark block Q norm/RoPE", q_got, q_want, 0.02);
    try expectBf16Near("DSpark block K norm/RoPE", k_got, k_want, 0.02);
    try check.sameBytes("DSpark block V copy", std.mem.sliceAsBytes(v_got), std.mem.sliceAsBytes(v_want));
    check.pass("DSpark 64-head QKV normalization/RoPE and V copy vs host", .{});
}

fn contextKv64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const f = try module.function("df_ctx_kv_glm");
    const R: usize = 8;
    const KVW: usize = 4096;
    const kv_h = try randomBf16(a, R * 2 * KVW, 0xd5a4_251);
    defer a.free(kv_h);
    const kn_h = try randomBf16(a, 64, 0xd5a4_252);
    defer a.free(kn_h);
    const pos_h = [_]i64{ 1, 39, 40, 2046, 2047, 2048, 3000, 3001 };
    var kv = try upload(gpu, std.mem.sliceAsBytes(kv_h));
    defer kv.free();
    var kn = try upload(gpu, std.mem.sliceAsBytes(kn_h));
    defer kn.free();
    var pos = try upload(gpu, std.mem.sliceAsBytes(&pos_h));
    defer pos.free();
    const ring_bytes = 2 * 2048 * KVW * @sizeOf(u16);
    var ring = try cuda.DeviceBuffer.alloc(gpu.d, ring_bytes);
    defer ring.free();
    try ring.fill8(0, s.handle);
    const ring_ptrs: [R]u64 = @splat(ring.ptr);
    var rings = try upload(gpu, std.mem.sliceAsBytes(&ring_ptrs));
    defer rings.free();
    var args: cuda.Args = .{};
    args.add(kv.ptr);
    args.add(kn.ptr);
    args.add(pos.ptr);
    args.add(rings.ptr);
    args.add(@as(i32, 0));
    args.add(@as(f32, 1e-5));
    args.add(@as(f32, 8_000_000));
    args.add(@as(i32, 64));
    args.add(@as(i32, 64));
    args.add(@as(i32, KVW));
    args.add(@as(i32, 2048));
    try cuda.launch.launch(f, .{ .grid = .{ .x = R, .y = 64 }, .block = .{ .x = 32 } }, s, &args);
    try s.synchronize();
    const ring_got = try a.alloc(u16, 2 * 2048 * KVW);
    defer a.free(ring_got);
    try ring.download(0, std.mem.sliceAsBytes(ring_got));
    for (0..R) |r| {
        const slot: usize = @intCast(@mod(pos_h[r], 2048));
        for (0..64) |h| {
            var expected: [64]u16 = undefined;
            normRope64Host(kv_h[r * 2 * KVW + h * 64 ..][0..64], kn_h, @intCast(pos_h[r]), &expected);
            try expectBf16Near("DSpark context K norm/RoPE", ring_got[slot * KVW + h * 64 ..][0..64], &expected, 0.02);
            try check.sameBytes("DSpark context V copy", std.mem.sliceAsBytes(ring_got[2048 * KVW + slot * KVW + h * 64 ..][0..64]), std.mem.sliceAsBytes(kv_h[r * 2 * KVW + KVW + h * 64 ..][0..64]));
        }
    }
    check.pass("DSpark context KV write at short and wrapped positions vs host", .{});
}

fn contextKvStage64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const stage_fn = try module.function("df_ctx_kv_stage_glm");
    const ring_fn = try module.function("df_ctx_kv_glm");
    const R: usize = 8;
    const KVW: usize = 4096;
    const S: usize = 16;
    const kv_h = try randomBf16(a, R * 2 * KVW, 0xd5a4_251);
    defer a.free(kv_h);
    const kn_h = try randomBf16(a, 64, 0xd5a4_252);
    defer a.free(kn_h);
    const pos_h = [_]i64{ 1, 39, 40, 2046, 2047, 2048, 3000, 3001 };
    var kv = try upload(gpu, std.mem.sliceAsBytes(kv_h));
    defer kv.free();
    var kn = try upload(gpu, std.mem.sliceAsBytes(kn_h));
    defer kn.free();
    var pos = try upload(gpu, std.mem.sliceAsBytes(&pos_h));
    defer pos.free();
    var stage = try cuda.DeviceBuffer.alloc(gpu.d, 3 * 2 * S * KVW * @sizeOf(u16));
    defer stage.free();
    const ring_len = 3 * 2 * 2048 * KVW * @sizeOf(u16);
    var ring = try cuda.DeviceBuffer.alloc(gpu.d, ring_len);
    defer ring.free();
    try ring.fill8(0, s.handle);
    const ring_ptrs: [R]u64 = @splat(ring.ptr);
    var rings = try upload(gpu, std.mem.sliceAsBytes(&ring_ptrs));
    defer rings.free();
    var side = try cuda.Stream.init(gpu.d, true);
    defer side.deinit();
    var shared_workspace = try cuda.DeviceBuffer.alloc(gpu.d, 64 * 1024);
    defer shared_workspace.free();
    try s.synchronize();

    for ([_]i32{ 0, 2 }) |layer| {
        var staged: cuda.Args = .{};
        staged.add(kv.ptr);
        staged.add(kn.ptr);
        staged.add(pos.ptr);
        staged.add(stage.ptr);
        staged.add(layer);
        staged.add(@as(f32, 1e-5));
        staged.add(@as(f32, 8_000_000));
        staged.add(@as(i32, 64));
        staged.add(@as(i32, 64));
        staged.add(@as(i32, KVW));
        staged.add(@as(i32, S));
        try cuda.launch.launch(stage_fn, .{ .grid = .{ .x = R, .y = 64 }, .block = .{ .x = 32 } }, side, &staged);
        for (0..32) |_| try shared_workspace.fill8(0xa5, s.handle);
        var ordinary: cuda.Args = .{};
        ordinary.add(kv.ptr);
        ordinary.add(kn.ptr);
        ordinary.add(pos.ptr);
        ordinary.add(rings.ptr);
        ordinary.add(layer);
        ordinary.add(@as(f32, 1e-5));
        ordinary.add(@as(f32, 8_000_000));
        ordinary.add(@as(i32, 64));
        ordinary.add(@as(i32, 64));
        ordinary.add(@as(i32, KVW));
        ordinary.add(@as(i32, 2048));
        try cuda.launch.launch(ring_fn, .{ .grid = .{ .x = R, .y = 64 }, .block = .{ .x = 32 } }, s, &ordinary);
    }
    try s.synchronize();
    try side.synchronize();
    const stage_h = try a.alloc(u16, 3 * 2 * S * KVW);
    defer a.free(stage_h);
    const ring_h = try a.alloc(u16, ring_len / @sizeOf(u16));
    defer a.free(ring_h);
    try stage.download(0, std.mem.sliceAsBytes(stage_h));
    try ring.download(0, std.mem.sliceAsBytes(ring_h));
    for ([_]usize{ 0, 2 }) |layer| for (0..R) |r| {
        const slot: usize = @intCast(@mod(pos_h[r], 2048));
        const stage_k = ((layer * 2) * S + r) * KVW;
        const stage_v = ((layer * 2 + 1) * S + r) * KVW;
        const ring_k = (layer * 2 * 2048 + slot) * KVW;
        const ring_v = (layer * 2 * 2048 + 2048 + slot) * KVW;
        try check.sameBytes("DSpark staged context K equals ring K", std.mem.sliceAsBytes(stage_h[stage_k..][0..KVW]), std.mem.sliceAsBytes(ring_h[ring_k..][0..KVW]));
        try check.sameBytes("DSpark staged context V equals ring V", std.mem.sliceAsBytes(stage_h[stage_v..][0..KVW]), std.mem.sliceAsBytes(ring_h[ring_v..][0..KVW]));
    };
    check.pass("DSpark staged context KV matches ring rows byte for byte at layers 0 and 2", .{});
}

fn ringCommit64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const stage_fn = try module.function("df_ctx_kv_stage_glm");
    const ring_fn = try module.function("df_ctx_kv_glm");
    const commit_fn = try module.function("df_ring_commit_glm");
    const R: usize = 8;
    const KVW: usize = 4096;
    const S: usize = 16;
    const rows_h = try randomBf16(a, R * 2 * KVW, 0xd5a4_261);
    defer a.free(rows_h);
    const kn_h = try randomBf16(a, 64, 0xd5a4_262);
    defer a.free(kn_h);
    const pos_h = [_]i64{ 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053 };
    var rows = try upload(gpu, std.mem.sliceAsBytes(rows_h));
    defer rows.free();
    var kn = try upload(gpu, std.mem.sliceAsBytes(kn_h));
    defer kn.free();
    var pos = try upload(gpu, std.mem.sliceAsBytes(&pos_h));
    defer pos.free();
    var stage = try cuda.DeviceBuffer.alloc(gpu.d, 3 * 2 * S * KVW * @sizeOf(u16));
    defer stage.free();
    const ring_len = 3 * 2 * 2048 * KVW * @sizeOf(u16);
    var committed = try cuda.DeviceBuffer.alloc(gpu.d, ring_len);
    defer committed.free();
    var expected = try cuda.DeviceBuffer.alloc(gpu.d, ring_len);
    defer expected.free();
    try committed.fill8(0, s.handle);
    try expected.fill8(0, s.handle);
    const ring_ptrs: [R]u64 = @splat(expected.ptr);
    var rings = try upload(gpu, std.mem.sliceAsBytes(&ring_ptrs));
    defer rings.free();

    for (0..3) |layer| {
        var staged: cuda.Args = .{};
        staged.add(rows.ptr);
        staged.add(kn.ptr);
        staged.add(pos.ptr);
        staged.add(stage.ptr);
        staged.add(@as(i32, @intCast(layer)));
        staged.add(@as(f32, 1e-5));
        staged.add(@as(f32, 8_000_000));
        staged.add(@as(i32, 64));
        staged.add(@as(i32, 64));
        staged.add(@as(i32, KVW));
        staged.add(@as(i32, S));
        try cuda.launch.launch(stage_fn, .{ .grid = .{ .x = R, .y = 64 }, .block = .{ .x = 32 } }, s, &staged);
        var ordinary: cuda.Args = .{};
        ordinary.add(rows.ptr);
        ordinary.add(kn.ptr);
        ordinary.add(pos.ptr);
        ordinary.add(rings.ptr);
        ordinary.add(@as(i32, @intCast(layer)));
        ordinary.add(@as(f32, 1e-5));
        ordinary.add(@as(f32, 8_000_000));
        ordinary.add(@as(i32, 64));
        ordinary.add(@as(i32, 64));
        ordinary.add(@as(i32, KVW));
        ordinary.add(@as(i32, 2048));
        try cuda.launch.launch(ring_fn, .{ .grid = .{ .x = 5, .y = 64 }, .block = .{ .x = 32 } }, s, &ordinary);
    }
    var commit: cuda.Args = .{};
    commit.add(stage.ptr);
    commit.add(committed.ptr);
    commit.add(@as(i64, 2046));
    commit.add(@as(i32, 5));
    commit.add(@as(i32, KVW));
    commit.add(@as(i32, 2048));
    commit.add(@as(i32, S));
    try cuda.launch.launch(commit_fn, .{ .grid = .{ .x = 5, .y = 6 }, .block = .{ .x = 256 } }, s, &commit);
    try s.synchronize();
    const got = try a.alloc(u16, ring_len / @sizeOf(u16));
    defer a.free(got);
    const want = try a.alloc(u16, ring_len / @sizeOf(u16));
    defer a.free(want);
    try committed.download(0, std.mem.sliceAsBytes(got));
    try expected.download(0, std.mem.sliceAsBytes(want));
    try check.sameBytes("DSpark staged ring commit writes only the kept prefix", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    check.pass("DSpark ring commit matches ordinary context ingest for five kept wrapped rows", .{});
}

fn attentionRowHost(q: []const u16, kb: []const u16, vb: []const u16, ring: []const u16, anchor: i64, from: i64, row: usize, head: usize, out: []u16) void {
    const qrow = row * 64 * 64 + head * 64;
    var lo = anchor + @as(i64, @intCast(row)) - 2048;
    if (lo < from) lo = from;
    if (lo < 0) lo = 0;
    const kvw: usize = 4096;
    const vbase = 2048 * kvw;
    var max_score: f32 = -std.math.inf(f32);
    var scores: [2056]f32 = undefined;
    var keys: usize = 0;
    var p = lo;
    while (p < anchor) : (p += 1) {
        const slot: usize = @intCast(@mod(p, 2048));
        const base = slot * kvw + head * 64;
        var score: f32 = 0;
        for (0..64) |d| score += f32bf(q[qrow + d]) * f32bf(ring[base + d]) * 0.125;
        scores[keys] = score;
        max_score = @max(max_score, score);
        keys += 1;
    }
    const block_row = row;
    for (0..block_row + 1) |b| {
        const base = (row + b) * 4096 + head * 64;
        var score: f32 = 0;
        for (0..64) |d| score += f32bf(q[qrow + d]) * f32bf(kb[base + d]) * 0.125;
        scores[keys] = score;
        max_score = @max(max_score, score);
        keys += 1;
    }
    var denom: f32 = 0;
    var acc: [64]f32 = @splat(0);
    var k: usize = 0;
    p = lo;
    while (p < anchor) : (p += 1) {
        const slot: usize = @intCast(@mod(p, 2048));
        const base = vbase + slot * kvw + head * 64;
        const w = @exp(scores[k] - max_score);
        denom += w;
        for (0..64) |d| acc[d] += w * f32bf(ring[base + d]);
        k += 1;
    }
    for (0..block_row + 1) |b| {
        const base = (row + b) * 4096 + head * 64;
        const w = @exp(scores[k] - max_score);
        denom += w;
        for (0..64) |d| acc[d] += w * f32bf(vb[base + d]);
        k += 1;
    }
    for (0..64) |d| out[d] = bf(acc[d] / denom);
}

fn blockAttention64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const part = try module.function("df_attn_part_glm");
    const merge = try module.function("df_attn_merge_glm");
    const seqs: usize = 2;
    const rows = seqs * 8;
    const q_h = try randomBf16(a, rows * 64 * 64, 0xd5a4_301);
    defer a.free(q_h);
    const kb_h = try randomBf16(a, rows * 4096, 0xd5a4_302);
    defer a.free(kb_h);
    const vb_h = try randomBf16(a, rows * 4096, 0xd5a4_303);
    defer a.free(vb_h);
    const ring_h = try randomBf16(a, 2 * 2048 * 4096, 0xd5a4_304);
    defer a.free(ring_h);
    const starts_h = [_]i64{ 3000, 40 };
    const froms_h = [_]i64{ 952, 0 };
    var q = try upload(gpu, std.mem.sliceAsBytes(q_h));
    defer q.free();
    var kb = try upload(gpu, std.mem.sliceAsBytes(kb_h));
    defer kb.free();
    var vb = try upload(gpu, std.mem.sliceAsBytes(vb_h));
    defer vb.free();
    var ring = try upload(gpu, std.mem.sliceAsBytes(ring_h));
    defer ring.free();
    const ring_ptrs = [_]u64{ ring.ptr, ring.ptr };
    var rings = try upload(gpu, std.mem.sliceAsBytes(&ring_ptrs));
    defer rings.free();
    var starts = try upload(gpu, std.mem.sliceAsBytes(&starts_h));
    defer starts.free();
    var froms = try upload(gpu, std.mem.sliceAsBytes(&froms_h));
    defer froms.free();
    const partial_count = rows * 64 * 16;
    var pm = try cuda.DeviceBuffer.alloc(gpu.d, partial_count * @sizeOf(f32));
    defer pm.free();
    var pl = try cuda.DeviceBuffer.alloc(gpu.d, partial_count * @sizeOf(f32));
    defer pl.free();
    var pacc = try cuda.DeviceBuffer.alloc(gpu.d, partial_count * 64 * @sizeOf(f32));
    defer pacc.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * 64 * 64 * @sizeOf(u16));
    defer out.free();
    var pa: cuda.Args = .{};
    pa.add(q.ptr);
    pa.add(kb.ptr);
    pa.add(vb.ptr);
    pa.add(rings.ptr);
    pa.add(starts.ptr);
    pa.add(froms.ptr);
    pa.add(@as(i32, 0));
    pa.add(@as(f32, 0.125));
    pa.add(@as(i32, 2048));
    pa.add(@as(i32, 8));
    pa.add(@as(i32, 64));
    pa.add(@as(i32, 64));
    pa.add(@as(i32, 64));
    pa.add(@as(i32, 4096));
    pa.add(pm.ptr);
    pa.add(pl.ptr);
    pa.add(pacc.ptr);
    try cuda.launch.launch(part, .{ .grid = .{ .x = seqs, .y = 64, .z = 16 }, .block = .{ .x = 256 } }, s, &pa);
    var ma: cuda.Args = .{};
    ma.add(pm.ptr);
    ma.add(pl.ptr);
    ma.add(pacc.ptr);
    ma.add(out.ptr);
    ma.add(@as(i32, 64));
    ma.add(@as(i32, 64));
    try cuda.launch.launch(merge, .{ .grid = .{ .x = rows, .y = 64 }, .block = .{ .x = 64 } }, s, &ma);
    try s.synchronize();
    const out_h = try a.alloc(u16, rows * 64 * 64);
    defer a.free(out_h);
    try out.download(0, std.mem.sliceAsBytes(out_h));
    for (0..seqs) |seq| for ([_]usize{ 0, 17 }) |head| {
        var expected: [64]u16 = undefined;
        const start = seq * 8;
        attentionRowHost(q_h[start * 64 * 64 ..], kb_h[start * 4096 ..], vb_h[start * 4096 ..], ring_h, starts_h[seq], froms_h[seq], 0, head, &expected);
        try expectBf16Near(if (seq == 0) "DSpark window-cut attention anchor 3000" else "DSpark short-context attention anchor 40", out_h[start * 64 * 64 + head * 64 ..][0..64], &expected, 0.025);
    };
    check.pass("DSpark attention host parity at 2048-window cut and short context", .{});
}

fn smallOps(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const add = try module.function("df_add_glm");
    const tap = try module.function("df_tap_glm");
    const swiglu = try module.function("df_swiglu");
    const sum2 = try module.function("df_sum2_glm");
    const rows: usize = 8;
    const hidden: usize = 6144;

    const xh = try randomBf16(a, rows * hidden, 0xd5a4_101);
    defer a.free(xh);
    const yh = try randomBf16(a, rows * hidden, 0xd5a4_102);
    defer a.free(yh);
    var x = try upload(gpu, std.mem.sliceAsBytes(xh));
    defer x.free();
    var y = try upload(gpu, std.mem.sliceAsBytes(yh));
    defer y.free();
    var aa: cuda.Args = .{};
    aa.add(x.ptr);
    aa.add(y.ptr);
    aa.add(@as(i32, rows));
    aa.add(@as(i32, hidden));
    try cuda.launch.launch(add, .{ .grid = .{ .x = rows, .y = (hidden + 255) / 256 }, .block = .{ .x = 256 } }, s, &aa);
    try s.synchronize();
    const add_got = try a.alloc(u16, rows * hidden);
    defer a.free(add_got);
    try x.download(0, std.mem.sliceAsBytes(add_got));
    const add_want = try a.alloc(u16, rows * hidden);
    defer a.free(add_want);
    for (add_want, 0..) |*v, i| v.* = bf(f32bf(xh[i]) + f32bf(yh[i]));
    try check.sameBytes("DSpark bf16 residual add", std.mem.sliceAsBytes(add_got), std.mem.sliceAsBytes(add_want));

    // Tap five residual inputs from plain hidden rows; only the requested slot changes.
    var tap_x = try upload(gpu, std.mem.sliceAsBytes(xh));
    defer tap_x.free();
    var tap_out = try cuda.DeviceBuffer.alloc(gpu.d, rows * 5 * hidden * @sizeOf(u16));
    defer tap_out.free();
    try tap_out.fill8(0, s.handle);
    var ta: cuda.Args = .{};
    ta.add(tap_x.ptr);
    ta.add(tap_out.ptr);
    ta.add(@as(i32, 2));
    ta.add(@as(i32, 1));
    ta.add(@as(i32, hidden));
    ta.add(@as(i32, 5));
    try cuda.launch.launch(tap, .{ .grid = .{ .x = rows, .y = (hidden + 1023) / 1024 }, .block = .{ .x = 256 } }, s, &ta);
    try s.synchronize();
    const tap_got = try a.alloc(u16, rows * 5 * hidden);
    defer a.free(tap_got);
    try tap_out.download(0, std.mem.sliceAsBytes(tap_got));
    const tap_want = try a.alloc(u16, rows * 5 * hidden);
    defer a.free(tap_want);
    @memset(tap_want, 0);
    for (0..rows) |r| @memcpy(tap_want[(r * 5 + 2) * hidden ..][0..hidden], xh[r * hidden ..][0..hidden]);
    try check.sameBytes("DSpark five-tap plain residual copy", std.mem.sliceAsBytes(tap_got), std.mem.sliceAsBytes(tap_want));

    // SwiGLU preserves the existing bf16 SiLU intermediate rounding at FF=12288.
    const ff: usize = 12288;
    const gu_h = try randomBf16(a, rows * 2 * ff, 0xd5a4_103);
    defer a.free(gu_h);
    var gu = try upload(gpu, std.mem.sliceAsBytes(gu_h));
    defer gu.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, rows * ff * @sizeOf(u16));
    defer act.free();
    var sa: cuda.Args = .{};
    sa.add(gu.ptr);
    sa.add(act.ptr);
    sa.add(@as(i32, ff));
    try cuda.launch.launch(swiglu, .{ .grid = .{ .x = rows, .y = ff / 1024 }, .block = .{ .x = 256 } }, s, &sa);
    try s.synchronize();
    const act_got = try a.alloc(u16, rows * ff);
    defer a.free(act_got);
    try act.download(0, std.mem.sliceAsBytes(act_got));
    const act_want = try a.alloc(u16, rows * ff);
    defer a.free(act_want);
    for (0..rows) |r| for (0..ff) |c| {
        const g = f32bf(gu_h[r * 2 * ff + c]);
        const u = f32bf(gu_h[r * 2 * ff + ff + c]);
        const silu = f32bf(bf(g / (1.0 + @exp(-g))));
        act_want[r * ff + c] = bf(silu * u);
    };
    try check.sameBytes("DSpark SwiGLU bf16 reference", std.mem.sliceAsBytes(act_got), std.mem.sliceAsBytes(act_want));

    // Split MLP partials are accumulated in fp32 and rounded once to bf16.
    const partial_h = try a.alloc(f32, 2 * rows * hidden);
    defer a.free(partial_h);
    var prng = std.Random.DefaultPrng.init(0xd5a4_104);
    for (partial_h) |*v| v.* = @as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -100, 100))) / 256.0;
    var partial = try upload(gpu, std.mem.sliceAsBytes(partial_h));
    defer partial.free();
    var summed = try cuda.DeviceBuffer.alloc(gpu.d, rows * hidden * @sizeOf(u16));
    defer summed.free();
    var su: cuda.Args = .{};
    su.add(partial.ptr);
    su.add(summed.ptr);
    su.add(@as(i32, rows));
    su.add(@as(i32, hidden));
    try cuda.launch.launch(sum2, .{ .grid = .{ .x = rows, .y = (hidden + 1023) / 1024 }, .block = .{ .x = 256 } }, s, &su);
    try s.synchronize();
    const sum_got = try a.alloc(u16, rows * hidden);
    defer a.free(sum_got);
    try summed.download(0, std.mem.sliceAsBytes(sum_got));
    const sum_want = try a.alloc(u16, rows * hidden);
    defer a.free(sum_want);
    for (sum_want, 0..) |*v, i| v.* = bf(partial_h[i] + partial_h[rows * hidden + i]);
    try check.sameBytes("DSpark split sum2 bf16 reference", std.mem.sliceAsBytes(sum_got), std.mem.sliceAsBytes(sum_want));
    check.pass("DSpark add/tap/SwiGLU/sum2 host-reference parity", .{});
}

fn top64(gpu: anytype, module: cuda.Module, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const part = try module.function("df_top16_part_glm");
    const merge = try module.function("df_top16_merge_glm");
    // Two rank halves, each Rows x HalfVocab logits: rank q's rows start at q * Rows * HalfVocab.
    const count = 2 * Rows * HalfVocab;
    var host = try a.alloc(u16, count);
    defer a.free(host);
    var prng = std.Random.DefaultPrng.init(0xd5a4_0064);
    for (host, 0..) |*x, i| {
        // Deliberately include quantized ties; host and device both resolve them by low id.
        const n: i32 = prng.random().intRangeAtMost(i32, -4000, 4000);
        x.* = bf(@as(f32, @floatFromInt(n)) / 256.0 + @as(f32, @floatFromInt(i % 17)) / 4096.0);
    }
    var input = try upload(gpu, std.mem.sliceAsBytes(host));
    defer input.free();
    const partial_items = Rows * Chunks * Top;
    var pv = try cuda.DeviceBuffer.alloc(gpu.d, partial_items * @sizeOf(f32));
    defer pv.free();
    var pi = try cuda.DeviceBuffer.alloc(gpu.d, partial_items * @sizeOf(i32));
    defer pi.free();
    var got = try cuda.DeviceBuffer.alloc(gpu.d, 2 * Rows * 2 * Top * @sizeOf(i32));
    defer got.free();
    for (0..2) |rank| {
        var pa: cuda.Args = .{};
        pa.add(input.ptr + @as(u64, @intCast(rank * Rows * HalfVocab * @sizeOf(u16))));
        pa.add(@as(i32, HalfVocab));
        pa.add(@as(i32, @intCast(rank * HalfVocab)));
        pa.add(@as(i32, Top));
        pa.add(pv.ptr);
        pa.add(pi.ptr);
        try cuda.launch.launch(part, .{ .grid = .{ .x = Rows, .y = Chunks }, .block = .{ .x = 256 } }, s, &pa);
        var ma: cuda.Args = .{};
        ma.add(pv.ptr);
        ma.add(pi.ptr);
        ma.add(@as(i32, Chunks));
        ma.add(@as(i32, Top));
        ma.add(got.ptr + @as(u64, @intCast(rank * Rows * 2 * Top * @sizeOf(i32))));
        try cuda.launch.launch(merge, .{ .grid = .{ .x = Rows }, .block = .{ .x = 256 } }, s, &ma);
        try s.synchronize();

        var expected = try a.alloc(i32, Rows * 2 * Top);
        defer a.free(expected);
        for (0..Rows) |r| {
            const offset: i32 = @intCast(rank * HalfVocab);
            try hostTop(a, host[(rank * Rows + r) * HalfVocab ..][0..HalfVocab], offset, expected[r * 2 * Top ..][0 .. 2 * Top]);
        }
        const actual_bytes = try a.alloc(u8, expected.len * @sizeOf(i32));
        defer a.free(actual_bytes);
        try got.download(@as(u64, @intCast(rank * actual_bytes.len)), actual_bytes);
        try check.sameBytes(if (rank == 0) "DSpark top-64 rank 0" else "DSpark top-64 rank 1", actual_bytes, std.mem.sliceAsBytes(expected));
    }
    check.pass("DSpark TOP=64 exact ids and bf16-derived values, three rows x two rank halves", .{});
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_dflash);
    defer module.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try smallOps(gpu, module, s);
    try blockQkv64(gpu, module, s);
    try contextKv64(gpu, module, s);
    try contextKvStage64(gpu, module, s);
    try ringCommit64(gpu, module, s);
    try blockAttention64(gpu, module, s);
    try top64(gpu, module, s);
}
