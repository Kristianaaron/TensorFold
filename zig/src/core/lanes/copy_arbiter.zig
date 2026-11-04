// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Copy vs drafter per round by expected tokens a millisecond (TF_GLM53_COPYD, plan D3). The pricing follows the lane allocator's rule (allocate.zig, ported from upstream TensorFold); the per-match acceptance idea comes from SuffixDecoding (Snowflake ArcticInference, Apache-2.0). No code is copied.
const std = @import("std");
const Table = @import("table.zig").Table;

pub const Pick = struct { drafts: usize, rate: f64 };
pub const Decision = struct { copy: Pick, head_rate: ?f64, copy_wins: bool };

fn widthOk(k: usize) bool {
    return k <= 7 or k == 11 or k == 15;
}

fn rowMs(costs: *const Table, rows: usize) f64 {
    return costs.get(@intCast(rows)) orelse @floatFromInt(rows);
}

pub fn at(chain: []const f64, k: usize, costs: *const Table) f64 {
    var expected: f64 = 1;
    for (chain[0..k]) |chance| expected += chance;
    return expected / rowMs(costs, k + 1);
}

pub fn best(chain: []const f64, costs: *const Table, least: usize, widths: bool, most: usize) Pick {
    var pick: Pick = .{ .drafts = 0, .rate = at(chain, 0, costs) };
    var found = false;
    const end: usize = @min(chain.len, most);
    var count = least;
    while (count <= end) : (count += 1) {
        if (widths and !widthOk(count)) continue;
        const candidate = at(chain, count, costs);
        if (!found or candidate > pick.rate) {
            pick = .{ .drafts = count, .rate = candidate };
            found = true;
        }
    }
    return pick;
}

pub fn copyWins(copy_rate: f64, head_rate: f64) bool {
    return copy_rate >= head_rate;
}

pub fn decide(copy_chain: []const f64, least: usize, head_chain: []const f64, head_trims: bool, costs: *const Table, widths: bool, max_copy: usize) Decision {
    const copy = best(copy_chain, costs, least, widths, max_copy);
    if (head_chain.len == 0) return .{ .copy = copy, .head_rate = null, .copy_wins = true };
    const head = if (head_trims) best(head_chain, costs, 1, widths, head_chain.len).rate else at(head_chain, head_chain.len, costs);
    return .{ .copy = copy, .head_rate = head, .copy_wins = copyWins(copy.rate, head) };
}

fn cumulative(a: std.mem.Allocator, q: f64, n: usize) ![]f64 {
    const chain = try a.alloc(f64, n);
    var chance = q;
    for (chain) |*item| {
        item.* = chance;
        chance *= q;
    }
    return chain;
}

fn measured(gpa: std.mem.Allocator) !Table {
    var costs: Table = .{};
    errdefer costs.deinit(gpa);
    const pairs = [_]struct { rows: i64, ms: f64 }{
        .{ .rows = 1, .ms = 88 }, .{ .rows = 2, .ms = 104.1 }, .{ .rows = 3, .ms = 120.2 }, .{ .rows = 4, .ms = 130.3 },
        .{ .rows = 5, .ms = 145.1 }, .{ .rows = 6, .ms = 159.8 }, .{ .rows = 7, .ms = 175.8 }, .{ .rows = 8, .ms = 191.8 },
        .{ .rows = 12, .ms = 274.3 }, .{ .rows = 16, .ms = 332.2 },
    };
    for (pairs) |entry| try costs.put(gpa, entry.rows, entry.ms);
    return costs;
}

test "copy pricing picks the best supported prefix" {
    const gpa = std.testing.allocator;
    var costs = try measured(gpa);
    defer costs.deinit(gpa);
    const q94 = try cumulative(gpa, 0.94, 15);
    defer gpa.free(q94);
    const seven = best(q94, &costs, 2, true, 7);
    try std.testing.expectEqual(@as(usize, 7), seven.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.033927), seven.rate, 1e-5);
    const limited = best(q94, &costs, 2, true, 15);
    try std.testing.expectEqual(@as(usize, 7), limited.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.033927), limited.rate, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.031844), at(q94, 11, &costs), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.031528), at(q94, 15, &costs), 1e-5);
    const q99 = try cumulative(gpa, 0.99, 15);
    defer gpa.free(q99);
    const full = best(q99, &costs, 2, true, 15);
    try std.testing.expectEqual(@as(usize, 15), full.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.044715), full.rate, 1e-5);
    const capped = best(q99, &costs, 2, true, 7);
    try std.testing.expectEqual(@as(usize, 7), capped.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.040279), capped.rate, 1e-5);
    const q30 = try cumulative(gpa, 0.3, 7);
    defer gpa.free(q30);
    const two = best(q30, &costs, 2, true, 7);
    try std.testing.expectEqual(@as(usize, 2), two.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.011564), two.rate, 1e-5);
    try std.testing.expectEqual(@as(usize, 1), best(q30, &costs, 1, true, 7).drafts);
    const empty: Table = .{};
    const qshort = try cumulative(gpa, 0.94, 3);
    defer gpa.free(qshort);
    const linear = best(qshort, &empty, 2, true, 3);
    try std.testing.expectEqual(@as(usize, 2), linear.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9412), linear.rate, 1e-5);
}

test "copy and held head prices decide a round" {
    const gpa = std.testing.allocator;
    var costs = try measured(gpa);
    defer costs.deinit(gpa);
    const q94 = try cumulative(gpa, 0.94, 7);
    defer gpa.free(q94);
    const q60 = try cumulative(gpa, 0.6, 7);
    defer gpa.free(q60);
    const first = decide(q94, 2, q60, true, &costs, true, 7);
    try std.testing.expect(first.copy_wins);
    try std.testing.expectEqual(@as(usize, 7), first.copy.drafts);
    try std.testing.expectApproxEqAbs(@as(f64, 0.016700), first.head_rate.?, 1e-5);
    const q30 = try cumulative(gpa, 0.3, 7);
    defer gpa.free(q30);
    const q90 = try cumulative(gpa, 0.9, 7);
    defer gpa.free(q90);
    const second = decide(q30, 2, q90, true, &costs, true, 7);
    try std.testing.expect(!second.copy_wins);
    try std.testing.expectApproxEqAbs(@as(f64, 0.029694), second.head_rate.?, 1e-5);
    const solo = decide(q30, 2, &.{}, true, &costs, true, 7);
    try std.testing.expect(solo.copy_wins and solo.head_rate == null);
    try std.testing.expectEqual(@as(usize, 2), solo.copy.drafts);
    const whole = decide(q30, 2, q60, false, &costs, true, 7);
    try std.testing.expect(!whole.copy_wins);
    try std.testing.expectApproxEqAbs(@as(f64, 0.012816), whole.head_rate.?, 1e-5);
    try std.testing.expect(copyWins(0.5, 0.5));
}
