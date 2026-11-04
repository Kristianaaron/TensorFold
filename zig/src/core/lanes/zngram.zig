// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Per-request suffix-automaton n-gram proposer.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sam = @import("zngram_sam.zig").Sam;

pub const Params = struct { min_match: u32 = 8, max_draft: u32 = 7, adaptive: bool = false };
pub const Stats = struct { proposals: u64 = 0, proposed: u64 = 0, accepted: u64 = 0, rounds: u64 = 0, match_sum: u64 = 0, silent: u64 = 0, suffix_rounds: u64 = 0, suffix_accepted: u64 = 0, dflash_rounds: u64 = 0, dflash_accepted: u64 = 0 };
pub const ZNgram = struct {
    gpa: Allocator, sam: Sam, params: Params, stats: Stats = .{}, last: i64 = 0,
    log: bool = false, last_proposed: u32 = 0, last_accepted: u32 = 0,
    acceptance_rate: f64 = 0.75,
    landed: f64 = 4.5,
    missed: f64 = 0.5,
    pub fn init(gpa: Allocator, params: Params) !ZNgram { return .{ .gpa = gpa, .sam = try Sam.init(gpa), .params = params }; }
    pub fn deinit(z: *ZNgram) void { z.sam.deinit(); }
    pub fn proposer(z: *ZNgram) Proposer { return .{ .ptr = z, .vtable = &vtable }; }
};

const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch, .observe = observe, .landing = landing, .round = round };
fn cast(ptr: *anyopaque) *ZNgram { return @ptrCast(@alignCast(ptr)); }
fn propose(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
    const z = cast(ptr);
    _ = try z.sam.syncAppend(context);
    z.last = z.sam.matchLen();
    z.last_proposed = 0;
    z.last_accepted = 0;
    if (max_draft <= 0 or z.last < z.params.min_match) { z.stats.silent += 1; return &.{}; }
    // A barely qualifying match gets a short trial. Longer matches can use the full verify window;
    // recent acceptance then trims or expands that ceiling without changing target verification.
    const limit = if (z.params.adaptive) blk: {
        const match_extra: u32 = @intCast(@max(0, z.last - @as(i64, z.params.min_match)));
        const match_cap = @min(z.params.max_draft, 4 + match_extra / 2);
        const acceptance_scale = 0.5 + z.acceptance_rate;
        const adaptive_cap: u32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(z.params.max_draft)) * acceptance_scale));
        break :blk @min(@min(@as(u32, @intCast(max_draft)), match_cap), adaptive_cap);
    } else @min(@as(u32, @intCast(max_draft)), z.params.max_draft);
    const out = z.sam.propose(limit, z.params.min_match);
    if (out.len != 0) { z.stats.proposals += 1; z.stats.proposed += out.len; z.stats.match_sum += @intCast(z.last); }
    z.last_proposed = @intCast(out.len);
    return out;
}
fn lastMatch(ptr: *anyopaque) i64 { return cast(ptr).last; }
fn landing(ptr: *anyopaque) f64 { const z = cast(ptr); return z.landed / (z.landed + z.missed); }
fn observe(ptr: *anyopaque, proposed: i64, accepted: i64) void {
    const z = cast(ptr);
    const p: u32 = @intCast(@max(0, proposed));
    const a: u32 = @intCast(@min(@max(0, accepted), @max(0, proposed)));
    z.last_proposed = p;
    z.last_accepted = a;
    z.stats.accepted += a;
    if (p > 0) {
        z.landed = 0.95 * z.landed + @as(f64, @floatFromInt(a));
        z.missed = 0.95 * z.missed + @as(f64, @floatFromInt(@intFromBool(a < p)));
    }
    if (p > 0) z.acceptance_rate = 0.75 * z.acceptance_rate + 0.25 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(p));
}
fn round(ptr: *anyopaque, _: u32, kept: u32, suffix_used: bool) void {
    const z = cast(ptr);
    z.stats.rounds += 1;
    if (suffix_used) {
        z.stats.suffix_rounds += 1;
        z.stats.suffix_accepted += @min(z.last_accepted, kept);
    } else {
        z.stats.dflash_rounds += 1;
        z.stats.dflash_accepted += kept;
    }
    if (z.log) std.log.info("draft round: drafter={s} draft={d} accepted={d}", .{ if (suffix_used) "suffix" else "dflash2", if (suffix_used) z.last_proposed else 0, if (suffix_used) z.last_accepted else kept });
    z.last_proposed = 0;
    z.last_accepted = 0;
}

pub const Totals = struct { requests: std.atomic.Value(u64) = .init(0), proposals: std.atomic.Value(u64) = .init(0), proposed: std.atomic.Value(u64) = .init(0), accepted: std.atomic.Value(u64) = .init(0) };
var process_totals: Totals = .{};
pub fn totals() *Totals { return &process_totals; }
pub const Factory = struct {
    params: Params, log: bool = false, totals: *Totals = &process_totals,
    pub fn make(ctx: *anyopaque, gpa: Allocator, prompt: []const u32, _: []const u32) anyerror!Proposer {
        const f: *Factory = @ptrCast(@alignCast(ctx));
        const z = try gpa.create(ZNgram);
        errdefer gpa.destroy(z);
        z.* = try ZNgram.init(gpa, f.params);
        z.log = f.log;
        errdefer z.deinit();
        _ = try z.sam.syncAppend(prompt);
        return z.proposer();
    }
    pub fn free(ctx: *anyopaque, p: Proposer) void {
        const f: *Factory = @ptrCast(@alignCast(ctx));
        const z = cast(p.ptr);
        if (f.log) std.log.info("draft summary: suffix mean accepted/round {d:.2} ({d}/{d}); DFlash2 mean accepted/round {d:.2} ({d}/{d}); suffix proposals {d}, drafted {d}, accepted {d} ({d:.1}%), mean match {d:.1}; all rounds {d}", .{ if (z.stats.suffix_rounds == 0) 0 else @as(f64, @floatFromInt(z.stats.suffix_accepted)) / @as(f64, @floatFromInt(z.stats.suffix_rounds)), z.stats.suffix_accepted, z.stats.suffix_rounds, if (z.stats.dflash_rounds == 0) 0 else @as(f64, @floatFromInt(z.stats.dflash_accepted)) / @as(f64, @floatFromInt(z.stats.dflash_rounds)), z.stats.dflash_accepted, z.stats.dflash_rounds, z.stats.proposals, z.stats.proposed, z.stats.accepted, if (z.stats.proposed == 0) 0 else 100.0 * @as(f64, @floatFromInt(z.stats.accepted)) / @as(f64, @floatFromInt(z.stats.proposed)), if (z.stats.proposals == 0) 0 else @as(f64, @floatFromInt(z.stats.match_sum)) / @as(f64, @floatFromInt(z.stats.proposals)), z.stats.rounds });
        _ = f.totals.requests.fetchAdd(1, .monotonic);
        _ = f.totals.proposals.fetchAdd(z.stats.proposals, .monotonic);
        _ = f.totals.proposed.fetchAdd(z.stats.proposed, .monotonic);
        _ = f.totals.accepted.fetchAdd(z.stats.accepted, .monotonic);
        z.deinit();
        z.gpa.destroy(z);
    }
};

test "zngram proposer syncs append-only context, proposes repeats and records outcomes" {
    var z = try ZNgram.init(std.testing.allocator, .{ .min_match = 3, .max_draft = 4 });
    defer z.deinit();
    const context = [_]u32{ 1, 2, 3, 4, 5, 1, 2, 3 };
    _ = try z.sam.sync(&context);
    const p = z.proposer();
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 1, 2 }, try p.propose(&context, 4));
    try std.testing.expectEqual(@as(u64, 1), z.stats.proposals);
    try std.testing.expectEqual(@as(u64, 4), z.stats.proposed);
    try std.testing.expectEqual(@as(u64, 3), z.stats.match_sum);
    try std.testing.expectEqual(@as(usize, context.len), z.sam.len());
    p.observe(2, 1);
    p.round(3, 2, true);
    try std.testing.expectEqual(@as(u64, 1), z.stats.accepted);
    try std.testing.expectEqual(@as(u64, 1), z.stats.rounds);
    const longer = [_]u32{ 1, 2, 3, 4, 5, 1, 2, 3, 4 };
    _ = try p.propose(&longer, 4);
    try std.testing.expectEqual(@as(usize, longer.len), z.sam.len());
    const changed = [_]u32{ 9, 8, 7, 6 };
    _ = try p.propose(&changed, 4);
    try std.testing.expectEqual(@as(usize, changed.len), z.sam.len());
    try std.testing.expectEqualSlices(u32, &changed, z.sam.toks.items);
    try std.testing.expectEqual(@as(usize, 0), (try p.propose(&changed, 4)).len);
    try std.testing.expect(z.last < z.params.min_match);
    try std.testing.expectEqual(@as(u64, 2), z.stats.silent);
}

test "zngram factory primes prompt" {
    var f: Factory = .{ .params = .{ .min_match = 2 } };
    const p = try Factory.make(&f, std.testing.allocator, &.{ 1, 2, 1, 2 }, &.{});
    const z = cast(p.ptr);
    try std.testing.expectEqual(@as(usize, 4), z.sam.len());
    Factory.free(&f, p);
}

test "suffix draft ceiling adapts to the recent suffix acceptance rate" {
    var z = try ZNgram.init(std.testing.allocator, .{ .min_match = 3, .max_draft = 7 });
    defer z.deinit();
    const initial = z.acceptance_rate;
    for (0..4) |_| z.proposer().observe(7, 0);
    try std.testing.expect(z.acceptance_rate < initial);
    for (0..4) |_| z.proposer().observe(7, 7);
    try std.testing.expect(z.acceptance_rate > 0.5);
}

test "landing estimate starts at the Recall prior and tracks outcomes" {
    var z = try ZNgram.init(std.testing.allocator, .{});
    defer z.deinit();
    const p = z.proposer();
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), p.landing().?, 1e-12);
    p.observe(7, 7);
    try std.testing.expect(p.landing().? > 0.9);
    for (0..12) |_| p.observe(7, 0);
    try std.testing.expect(p.landing().? < 0.5);
}
