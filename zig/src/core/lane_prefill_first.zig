// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Bounded prompt-piece priority planner for hosts that can keep several sequences alive.
const std = @import("std");
const api = @import("engine_api.zig");

pub const Settings = struct {
    piece_rows: u32 = 512,
    burst: u32 = 4,
    idle_burst: u32 = 8,
    max_stall_ms: u32 = 800,
    aging: u32 = 8,

    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        s.burst = try envInt("TF_GLM53_PREFILL_BURST", s.burst, 0, 64);
        s.idle_burst = try envInt("TF_GLM53_PREFILL_IDLE_BURST", s.idle_burst, 1, 64);
        s.max_stall_ms = try envInt("TF_GLM53_DECODE_STALL_MS", s.max_stall_ms, 0, 60000);
        s.aging = try envInt("TF_GLM53_PREFILL_AGING", s.aging, 1, 64);
        return s;
    }
};

fn envInt(comptime name: [:0]const u8, fallback: u32, min: u32, max: u32) !u32 {
    const raw = std.c.getenv(name) orelse return fallback;
    const text = std.mem.trim(u8, std.mem.span(raw), " \t\r\n");
    const n = std.fmt.parseInt(u32, text, 10) catch return error.InvalidPrefillSetting;
    if (n < min or n > max) return error.InvalidPrefillSetting;
    return n;
}

pub const Planner = struct {
    const Seq = struct {
        key: usize,
        prompt: []const u32,
        at: u32,
        fill_wait: u32 = 0,
        decoding: bool = false,
    };

    gpa: std.mem.Allocator,
    settings: Settings = .{},
    start_of: ?*const fn (ctx: *anyopaque, prompt: []const u32) u32 = null,
    start_ctx: ?*anyopaque = null,
    seqs: std.ArrayList(Seq) = .empty,
    stalled_s: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, settings: Settings) Planner {
        return .{ .gpa = gpa, .settings = settings };
    }
    pub fn deinit(p: *Planner) void { p.seqs.deinit(p.gpa); }
    fn self(ctx: *anyopaque) *Planner { return @ptrCast(@alignCast(ctx)); }
    pub fn rounds(p: *Planner) api.Rounds {
        return .{ .ctx = p, .vtable = &.{ .begin_admit = beginAdmit, .admit = admit, .admitted = admitted, .arm = arm, .began = began, .plan = plan, .left = left, .after = after } };
    }
    fn beginAdmit(_: *anyopaque) void {}
    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        const p = self(ctx);
        const start = if (p.start_of) |f| f(p.start_ctx orelse ctx, ask.prompt) else 0;
        p.seqs.append(p.gpa, .{ .key = ask.key, .prompt = ask.prompt, .at = start }) catch return .refuse;
        return .admit;
    }
    fn admitted(_: *anyopaque) anyerror!void {}
    fn arm(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}

    fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        const p = self(ctx);
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        var decoding = false;
        for (p.seqs.items) |q| decoding = decoding or q.decoding;
        const budget: u32 = if (!decoding) p.settings.idle_burst else if (p.settings.burst == 0 or p.stalled_s * 1000 >= @as(f64, @floatFromInt(p.settings.max_stall_ms))) 0 else p.settings.burst;
        for (0..budget) |_| {
            const idx = p.choose() orelse break;
            const q = &p.seqs.items[idx];
            const end: u32 = @intCast(@min(@as(u64, q.at) + p.settings.piece_rows, @as(u64, @intCast(q.prompt.len - 1))));
            if (end <= q.at) {
                q.decoding = true;
                try finals.append(p.gpa, q.key);
                continue;
            }
            try pieces.append(p.gpa, .{ .key = q.key, .start = q.at, .end = end, .save = false });
            q.at = end;
            if (q.at == q.prompt.len - 1) {
                q.decoding = true;
                try finals.append(p.gpa, q.key);
            }
        }
    }
    fn choose(p: *Planner) ?usize {
        var best: ?usize = null;
        for (p.seqs.items, 0..) |q, i| {
            if (q.decoding or q.at >= q.prompt.len - 1) continue;
            const b = best orelse { best = i; continue; };
            const old = p.seqs.items[b];
            const aged = q.fill_wait >= p.settings.aging;
            const old_aged = old.fill_wait >= p.settings.aging;
            if ((aged and !old_aged) or (aged == old_aged and ((aged and q.fill_wait > old.fill_wait) or (!aged and q.prompt.len < old.prompt.len)))) best = i;
        }
        if (best) |chosen| {
            for (p.seqs.items, 0..) |*q, i| q.fill_wait = if (i == chosen) 0 else q.fill_wait +| 1;
        }
        return best;
    }
    fn left(ctx: *anyopaque, key: usize) void {
        const p = self(ctx);
        for (p.seqs.items, 0..) |q, i| if (q.key == key) { _ = p.seqs.orderedRemove(i); return; };
    }
    fn after(ctx: *anyopaque, piece_s: f64, decode_s: f64) void {
        const p = self(ctx);
        if (decode_s > 0) p.stalled_s = 0 else p.stalled_s += piece_s;
    }
};

test "bounded prefill planner emits aligned pieces around fake backend decode rounds" {
    const lanes = @import("lanes");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var fake: lanes.fake.Fake = .{ .gpa = gpa };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, fake.pieceBackend(), clock.clock());
    defer core.deinit();
    var p = Planner.init(gpa, .{ .piece_rows = 4, .burst = 1, .idle_burst = 1, .max_stall_ms = 50, .aging = 2 });
    defer p.deinit();
    var pieces: std.ArrayList(api.Rounds.Piece) = .empty;
    defer pieces.deinit(gpa);
    var finals: std.ArrayList(usize) = .empty;
    defer finals.deinit(gpa);
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    const short = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const rd = p.rounds();
    var why: []const u8 = "";
    try std.testing.expectEqual(api.Rounds.Verdict.admit, rd.vtable.admit(rd.ctx, .{ .key = 1, .prompt = &long, .max_tokens = 20, .submitted_s = 0 }, &why));
    try std.testing.expectEqual(api.Rounds.Verdict.admit, rd.vtable.admit(rd.ctx, .{ .key = 2, .prompt = &short, .max_tokens = 20, .submitted_s = 0 }, &why));
    // Short foreground prompt is selected first, then every interval stays aligned and excludes the final token.
    try rd.vtable.plan(rd.ctx, &pieces, &finals);
    try std.testing.expectEqual(@as(usize, 1), pieces.items.len);
    try std.testing.expectEqual(@as(usize, 2), pieces.items[0].key);
    try std.testing.expectEqual(@as(u64, 0), pieces.items[0].start);
    try std.testing.expectEqual(@as(u64, 4), pieces.items[0].end);
    try std.testing.expectEqual(@as(usize, 0), finals.items.len);
    // Simulate elapsed prefill with no decode; max stall disables pieces while a stream decodes.
    rd.vtable.after(rd.ctx, 0.06, 0);
    p.seqs.items[1].decoding = true;
    try rd.vtable.plan(rd.ctx, &pieces, &finals);
    try std.testing.expectEqual(@as(usize, 0), pieces.items.len);
    // Cancellation removes its slot and deterministic planning continues for the remaining prompt.
    rd.vtable.left(rd.ctx, 2);
    p.stalled_s = 0;
    try rd.vtable.plan(rd.ctx, &pieces, &finals);
    try std.testing.expectEqual(@as(usize, 1), pieces.items.len);
    try std.testing.expectEqual(@as(usize, 1), pieces.items[0].key);
    try std.testing.expectEqual(@as(u64, 0), pieces.items[0].start);
    try std.testing.expectEqual(@as(u64, 4), pieces.items[0].end);
}

test "bounded prefill planner preserves piece invariants across fixed-seed workloads" {
    const lanes = @import("lanes");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var fake: lanes.fake.Fake = .{ .gpa = gpa };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, fake.pieceBackend(), clock.clock());
    defer core.deinit();

    // Deterministic LCG workloads keep this property test reproducible on every host.
    var seed: u32 = 0x53_77_2026;
    for (0..64) |case| {
        seed = seed *% 1664525 +% 1013904223;
        const n1: usize = 513 + (seed % 3584);
        seed = seed *% 1664525 +% 1013904223;
        const n2: usize = 513 + (seed % 3584);
        const p1 = try gpa.alloc(u32, n1);
        defer gpa.free(p1);
        const p2 = try gpa.alloc(u32, n2);
        defer gpa.free(p2);
        @memset(p1, 1);
        @memset(p2, 2);
        var planner = Planner.init(gpa, .{ .piece_rows = 512, .burst = 2, .idle_burst = 2, .max_stall_ms = 800, .aging = 8 });
        defer planner.deinit();
        var pieces: std.ArrayList(api.Rounds.Piece) = .empty;
        defer pieces.deinit(gpa);
        var finals: std.ArrayList(usize) = .empty;
        defer finals.deinit(gpa);
        const rd = planner.rounds();
        var why: []const u8 = "";
        try std.testing.expectEqual(api.Rounds.Verdict.admit, rd.vtable.admit(rd.ctx, .{ .key = 1, .prompt = p1, .max_tokens = 1, .submitted_s = 0 }, &why));
        try std.testing.expectEqual(api.Rounds.Verdict.admit, rd.vtable.admit(rd.ctx, .{ .key = 2, .prompt = p2, .max_tokens = 1, .submitted_s = 0 }, &why));
        var next = [_]u64{ 0, 0 };
        var finish_round = [_]usize{ 0, 0 };
        var rounds: usize = 0;
        var decode_ready_seen = [_]bool{ false, false };
        while (finish_round[0] == 0 or finish_round[1] == 0) : (rounds += 1) {
            try std.testing.expect(rounds < 32);
            try rd.vtable.plan(rd.ctx, &pieces, &finals);
            for (pieces.items) |piece| {
                const ix = piece.key - 1;
                const prompt = if (ix == 0) p1 else p2;
                try std.testing.expectEqual(next[ix], piece.start);
                try std.testing.expectEqual(@as(u64, 0), piece.start % 512);
                try std.testing.expect(piece.end <= prompt.len - 1);
                try std.testing.expect(piece.end % 512 == 0 or piece.end == prompt.len - 1);
                next[ix] = piece.end;
            }
            for (finals.items) |key| {
                const ix = key - 1;
                finish_round[ix] = rounds + 1;
                decode_ready_seen[ix] = true;
                try std.testing.expectEqual(@as(u64, (if (ix == 0) p1.len else p2.len) - 1), next[ix]);
            }
            // In idle filling there are no decode-ready streams, so each round spends at most
            // the configured burst. Once a stream becomes ready it cannot accumulate prefill debt.
            try std.testing.expect(pieces.items.len <= 2);
            rd.vtable.after(rd.ctx, 0, if (finals.items.len > 0) 0.001 else 0);
        }
        _ = case;
        try std.testing.expect(decode_ready_seen[0] and decode_ready_seen[1]);
        try std.testing.expect(finish_round[0] <= 16 and finish_round[1] <= 16);
        // No pending prefill means subsequent plans contain no pieces; decode-ready streams
        // remain eligible for decode without a planner delay.
        try rd.vtable.plan(rd.ctx, &pieces, &finals);
        try std.testing.expectEqual(@as(usize, 0), pieces.items.len);
    }
}
