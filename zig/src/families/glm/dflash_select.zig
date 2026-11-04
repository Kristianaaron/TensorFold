// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! DFlash2's path selector on rank 0: each draft among a row's 16 likeliest by the previous draft's codebook row.
const std = @import("std");
const df = @import("dflash.zig");
const RANK = df.RANK;
const MAX_TOPK = 64;

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, count: usize, stream: *anyopaque) usize;
extern "c" fn fflush(stream: *anyopaque) c_int;
extern "c" fn fclose(stream: *anyopaque) c_int;

const Cand = struct { v: f32, id: u32 };

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Stream `i`'s 7 drafts: each row's best of every rank's 16 by logit plus codebook score; greedy, chance kept.
pub fn select(dr: *const df.Drafter, i: usize, M: usize, q: df.Req) void {
    const out = q.out.?;
    var prev: usize = q.anchor;
    const g = dr.geometry;
    const rank = g.selector_rank;
    const top_k = g.selector_top_k;
    for (1..g.block_size) |t| {
        const r = i * g.block_size + t;
        var c: [MAX_TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const cw = df.cw(g);
            const blk = dr.cand[rk * M * cw + r * cw ..][0..cw];
            for (0..top_k) |j| {
                if (blk[j] < 0) continue;
                const cand: Cand = .{ .v = @bitCast(blk[top_k + j]), .id = @intCast(blk[j]) };
                if (n < top_k) {
                    c[n] = cand;
                    n += 1;
                    continue;
                }
                var lo: usize = 0;
                for (1..top_k) |m| if (c[m].v < c[lo].v) {
                    lo = m;
                };
                if (cand.v > c[lo].v) c[lo] = cand;
            }
        }
        if (n == 0) { // no candidate (cannot happen with a whole vocabulary): the anchor repeated, chance 0
            for (t - 1..g.block_size - 1) |k| {
                out[k] = @intCast(prev);
                if (q.probs) |p| p[k] = 0;
            }
            return;
        }
        const gvec = dr.gh[r * rank ..][0..rank];
        const P = dr.pred[prev * rank ..][0..rank];
        var pg: [RANK]f32 = undefined;
        for (pg[0..rank], P, gvec) |*o, p, x| o.* = bf(p) * x;
        var best: usize = 0;
        var sc: [MAX_TOPK]f32 = undefined;
        for (0..n) |m| {
            const Sc = dr.succ[@as(usize, c[m].id) * rank ..][0..rank];
            var dot: f32 = 0;
            for (pg[0..rank], Sc) |a, b| dot += a * bf(b);
            sc[m] = c[m].v + dot;
            if (sc[m] > sc[best]) best = m;
        }
        var z: f64 = 0;
        for (sc[0..n]) |x| z += @exp(@as(f64, x - sc[best]) / dr.temp);
        out[t - 1] = c[best].id;
        if (q.probs) |p| p[t - 1] = 1.0 / z;
        prev = c[best].id;
    }
}

/// DSpark's vanilla rank-256 Markov selector and conditional confidence head.
pub fn selectDspark(dr: *const df.Drafter, i: usize, M: usize, q: df.Req) void {
    const out = q.out.?;
    // GLM-5.3 (glm_dsa) passes probabilities only when TF_GLM53_DRAFT_CHANCES=1; they remain optional, as in select().
    const probs: ?[]f64 = q.probs;
    const g = dr.geometry;
    const rank = g.selector_rank;
    const top_k = g.selector_top_k;
    var prev: usize = q.anchor;
    var drafts: [8]u32 = undefined;
    var confs: [8]f64 = undefined;
    var top1s: [8]u32 = undefined;
    for (0..8) |slot| {
        const r = i * g.block_size + slot;
        const cw = df.cw(g);
        var candidates: [MAX_TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const blk = dr.cand[rk * M * cw + r * cw ..][0..cw];
            for (0..top_k) |j| {
                if (blk[j] < 0) continue;
                const cand: Cand = .{ .v = @bitCast(blk[top_k + j]), .id = @intCast(blk[j]) };
                if (n < top_k) {
                    candidates[n] = cand;
                    n += 1;
                } else {
                    var lo: usize = 0;
                    for (1..n) |m| if (candidates[m].v < candidates[lo].v or
                        (candidates[m].v == candidates[lo].v and candidates[m].id > candidates[lo].id)) { lo = m; };
                    if (cand.v > candidates[lo].v or (cand.v == candidates[lo].v and cand.id < candidates[lo].id)) candidates[lo] = cand;
                }
            }
        }
        if (n == 0) {
            for (0..8) |slot0| {
                if (slot0 < out.len) out[slot0] = @intCast(prev);
                if (probs) |p| if (slot0 < p.len) {
                    p[slot0] = 0;
                };
                drafts[slot0] = @intCast(prev);
                confs[slot0] = 0;
                top1s[slot0] = @intCast(prev);
            }
            traceDspark(q.pos, q.anchor, &drafts, &confs, &top1s);
            return;
        }
        var base_best: usize = 0;
        for (1..n) |m| if (candidates[m].v > candidates[base_best].v or
            (candidates[m].v == candidates[base_best].v and candidates[m].id < candidates[base_best].id)) { base_best = m; };
        top1s[slot] = candidates[base_best].id;
        const w1 = dr.markov_w1[prev * rank ..][0..rank];
        var best = base_best;
        var best_score: f32 = -std.math.inf(f32);
        // TF_GLM53_ROUND_TRACE / TF_GLM53_DRAFT_RECAL: a 32-slot probs buffer also takes z, margin and logp per slot.
        const feats = if (probs) |ps| ps.len >= 32 else false;
        var scores: [MAX_TOPK]f32 = undefined;
        for (0..n) |m| {
            const w2 = dr.markov_w2[@as(usize, candidates[m].id) * rank ..][0..rank];
            var score = candidates[m].v;
            for (w1, w2) |a, b| score += bf(a) * bf(b);
            if (feats) scores[m] = score;
            if (score > best_score or (score == best_score and candidates[m].id < candidates[best].id)) {
                best = m;
                best_score = score;
            }
        }
        const draft = candidates[best].id;
        drafts[slot] = draft;
        // DSpark proposes 8; the glm_dsa backend holds 7 (its depth cap), so write only what the caller holds.
        if (slot < out.len) out[slot] = draft;
        const conf_prev = if (slot == 0) q.anchor else drafts[slot - 1];
        const conf_w1 = dr.markov_w1[conf_prev * rank ..][0..rank];
        const h = dr.hrows[r * g.hidden ..][0..g.hidden];
        var z: f32 = dr.conf_b;
        for (h, 0..) |v, j| z += bf(v) * bf(dr.conf_w[j]);
        for (conf_w1, 0..) |v, j| z += bf(v) * bf(dr.conf_w[g.hidden + j]);
        const p: f64 = 1.0 / (1.0 + @exp(-@as(f64, z)));
        confs[slot] = p;
        if (probs) |ps| if (slot < ps.len) {
            ps[slot] = if (dr.conf_temp == 1.0) p else 1.0 / (1.0 + @exp(-@as(f64, z) / dr.conf_temp));
        };
        if (feats) {
            const ps = probs.?;
            var second: f32 = -std.math.inf(f32);
            var mass: f64 = 0;
            for (scores[0..n], 0..) |sc, m| {
                if (m != best and sc > second) second = sc;
                mass += @exp(@as(f64, sc) - @as(f64, best_score));
            }
            ps[8 + slot] = z;
            ps[16 + slot] = if (n > 1) @as(f64, best_score) - @as(f64, second) else 0;
            ps[24 + slot] = -@log(mass);
        }
        prev = draft;
    }
    traceDspark(q.pos, q.anchor, &drafts, &confs, &top1s);
}

fn traceDspark(pos: i64, anchor: u32, drafts: *const [8]u32, conf: *const [8]f64, top1: *const [8]u32) void {
    const path = std.c.getenv("TF_GLM_DSPARK_TRACE") orelse return;
    if (path[0] == 0) return;
    const zpath = std.heap.page_allocator.dupeSentinel(u8, std.mem.span(path), 0) catch return;
    defer std.heap.page_allocator.free(zpath);
    var line: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer line.deinit();
    line.writer.print("{{\"pos\":{d},\"anchor\":{d},\"drafts\":[", .{ pos, anchor }) catch return;
    for (drafts.*, 0..) |v, j| line.writer.print("{s}{d}", .{ if (j == 0) "" else ",", v }) catch return;
    line.writer.writeAll("],\"conf\":[") catch return;
    for (conf.*, 0..) |v, j| line.writer.print("{s}{d:.9}", .{ if (j == 0) "" else ",", v }) catch return;
    line.writer.writeAll("],\"top1\":[") catch return;
    for (top1.*, 0..) |v, j| line.writer.print("{s}{d}", .{ if (j == 0) "" else ",", v }) catch return;
    line.writer.writeAll("]}\n") catch return;
    const file = fopen(zpath.ptr, "a") orelse return;
    defer _ = fclose(file);
    const bytes = line.written();
    if (fwrite(bytes.ptr, 1, bytes.len, file) != bytes.len) return;
    _ = fflush(file);
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

test "DSpark selector Markov chain, confidence, and lower-id tie" {
    const gpa = std.testing.allocator;
    var cand: [2 * 8 * 8]i32 = @splat(-1);
    for (0..8) |row| {
        const ids = [_]u32{ 2, 3, 4, 5 };
        const vals = if (row == 2) [_]f32{ 1, 0, 10, 10 } else [_]f32{ 10, 9, 8, 7 };
        for (ids, vals, 0..) |id, value, j| {
            cand[row * 8 + j] = @intCast(id);
            cand[row * 8 + 4 + j] = @bitCast(value);
        }
        const peer = 8 * 8 + row * 8;
        for ([_]u32{ 6, 7, 8, 9 }, [_]f32{ 1, 0, -1, -2 }, 0..) |id, value, j| {
            cand[peer + j] = @intCast(id);
            cand[peer + 4 + j] = @bitCast(value);
        }
    }
    var w1: [32 * 4]u16 = @splat(0);
    var w2: [32 * 4]u16 = @splat(0);
    w1[1 * 4] = bf16(1);
    w1[3 * 4] = bf16(1);
    w2[3 * 4] = bf16(10);
    w2[4 * 4] = bf16(10);
    var conf_w: [6]u16 = @splat(0);
    var hrows: [8 * 2]u16 = @splat(0);
    var out: [8]u32 = undefined;
    var probs: [8]f64 = undefined;
    var dr: df.Drafter = .{
        .gpa = gpa,
        .d = undefined,
        .module = undefined,
        .k = undefined,
        .weights = undefined,
        .geometry = .{ .kind = .dspark, .hidden = 2, .selector_rank = 4, .selector_top_k = 4, .block_size = 8 },
        .split = false,
        .ff = 0,
        .rank = 0,
        .world = 2,
        .cand = &cand,
        .markov_w1 = &w1,
        .markov_w2 = &w2,
        .conf_w = &conf_w,
        .hrows = &hrows,
    };
    const req: df.Req = .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out, .probs = &probs };
    selectDspark(&dr, 0, 8, req);
    try std.testing.expectEqual(@as(u32, 3), out[0]); // Markov score reverses the base argmax (2).
    try std.testing.expectEqual(@as(u32, 4), out[1]); // W1[3] changes the next slot's choice.
    try std.testing.expectEqual(@as(u32, 4), out[2]); // Equal scores choose the lower vocab id.
    for (probs) |p| try std.testing.expectApproxEqAbs(@as(f64, 0.5), p, 1e-6);
}

test "glm dflash: DSpark confidence temperature changes chances, never drafts" {
    const gpa = std.testing.allocator;
    var cand: [2 * 8 * 8]i32 = @splat(-1);
    for (0..8) |row| {
        const ids = [_]u32{ 2, 3, 4, 5 };
        const vals = if (row == 2) [_]f32{ 1, 0, 10, 10 } else [_]f32{ 10, 9, 8, 7 };
        for (ids, vals, 0..) |id, value, j| {
            cand[row * 8 + j] = @intCast(id);
            cand[row * 8 + 4 + j] = @bitCast(value);
        }
        const peer = 8 * 8 + row * 8;
        for ([_]u32{ 6, 7, 8, 9 }, [_]f32{ 1, 0, -1, -2 }, 0..) |id, value, j| {
            cand[peer + j] = @intCast(id);
            cand[peer + 4 + j] = @bitCast(value);
        }
    }
    var w1: [32 * 4]u16 = @splat(0);
    var w2: [32 * 4]u16 = @splat(0);
    w1[1 * 4] = bf16(1);
    w1[3 * 4] = bf16(1);
    w2[3 * 4] = bf16(10);
    w2[4 * 4] = bf16(10);
    var conf_w: [6]u16 = @splat(0);
    var hrows: [8 * 2]u16 = @splat(0);
    var out_one: [8]u32 = undefined;
    var out_two: [8]u32 = undefined;
    var out_null: [8]u32 = undefined;
    var probs_one: [8]f64 = undefined;
    var probs_two: [8]f64 = undefined;
    var dr: df.Drafter = .{
        .gpa = gpa,
        .d = undefined,
        .module = undefined,
        .k = undefined,
        .weights = undefined,
        .geometry = .{ .kind = .dspark, .hidden = 2, .selector_rank = 4, .selector_top_k = 4, .block_size = 8 },
        .split = false,
        .ff = 0,
        .rank = 0,
        .world = 2,
        .cand = &cand,
        .markov_w1 = &w1,
        .markov_w2 = &w2,
        .conf_w = &conf_w,
        .conf_b = 1.0,
        .hrows = &hrows,
    };
    dr.conf_temp = 1.0;
    selectDspark(&dr, 0, 8, .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out_one, .probs = &probs_one });
    dr.conf_temp = 2.0;
    selectDspark(&dr, 0, 8, .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out_two, .probs = &probs_two });
    dr.conf_temp = 1.0;
    selectDspark(&dr, 0, 8, .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out_null, .probs = null });
    try std.testing.expectEqualSlices(u32, &out_one, &out_two);
    try std.testing.expectEqualSlices(u32, &out_one, &out_null);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 4 }, out_one[0..3]);
    for (probs_one) |p| try std.testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), p, 1e-12);
    for (probs_two) |p| try std.testing.expectApproxEqAbs(@as(f64, 0.6224593312018546), p, 1e-12);
}

test "glm dflash: DSpark features fill a 32-slot buffer, never change drafts" {
    const gpa = std.testing.allocator;
    var cand: [2 * 8 * 8]i32 = @splat(-1);
    for (0..8) |row| {
        const ids = [_]u32{ 2, 3, 4, 5 };
        const vals = if (row == 2) [_]f32{ 1, 0, 10, 10 } else [_]f32{ 10, 9, 8, 7 };
        for (ids, vals, 0..) |id, value, j| {
            cand[row * 8 + j] = @intCast(id);
            cand[row * 8 + 4 + j] = @bitCast(value);
        }
        const peer = 8 * 8 + row * 8;
        for ([_]u32{ 6, 7, 8, 9 }, [_]f32{ 1, 0, -1, -2 }, 0..) |id, value, j| {
            cand[peer + j] = @intCast(id);
            cand[peer + 4 + j] = @bitCast(value);
        }
    }
    var w1: [32 * 4]u16 = @splat(0);
    var w2: [32 * 4]u16 = @splat(0);
    w1[1 * 4] = bf16(1);
    w1[3 * 4] = bf16(1);
    w2[3 * 4] = bf16(10);
    w2[4 * 4] = bf16(10);
    var conf_w: [6]u16 = @splat(0);
    var hrows: [8 * 2]u16 = @splat(0);
    var out_8: [8]u32 = undefined;
    var out_32: [8]u32 = undefined;
    var probs_8: [8]f64 = undefined;
    var probs_32: [32]f64 = @splat(0);
    var dr: df.Drafter = .{
        .gpa = gpa,
        .d = undefined,
        .module = undefined,
        .k = undefined,
        .weights = undefined,
        .geometry = .{ .kind = .dspark, .hidden = 2, .selector_rank = 4, .selector_top_k = 4, .block_size = 8 },
        .split = false,
        .ff = 0,
        .rank = 0,
        .world = 2,
        .cand = &cand,
        .markov_w1 = &w1,
        .markov_w2 = &w2,
        .conf_w = &conf_w,
        .conf_b = 1.0,
        .hrows = &hrows,
    };
    dr.conf_temp = 1.0;
    selectDspark(&dr, 0, 8, .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out_8, .probs = &probs_8 });
    selectDspark(&dr, 0, 8, .{ .slot = 0, .pos = 17, .n = 0, .anchor = 1, .out = &out_32, .probs = &probs_32 });
    try std.testing.expectEqualSlices(u32, &out_8, &out_32);
    try std.testing.expectEqualSlices(f64, &probs_8, probs_32[0..8]);
    for (0..8) |slot| {
        try std.testing.expectEqual(@as(f64, 1.0), probs_32[8 + slot]);
        try std.testing.expect(probs_32[16 + slot] >= 0);
        try std.testing.expect(probs_32[24 + slot] >= -@log(4.0) - 1e-12 and probs_32[24 + slot] <= 1e-12);
        try std.testing.expect(probs_32[24 + slot] <= -@log(1.0 + @exp(-probs_32[16 + slot])) + 1e-12);
    }
    // Slot 0 rescored values are id2=10, id3=9+10=19, id4=8+10=18, id5=7; the runner-up margin is 1.
    try std.testing.expectEqual(@as(f64, 1.0), probs_32[16]);
    try std.testing.expectApproxEqAbs(-@log(1.0 + @exp(-1.0) + @exp(-9.0) + @exp(-12.0)), probs_32[24], 1e-9);
}
