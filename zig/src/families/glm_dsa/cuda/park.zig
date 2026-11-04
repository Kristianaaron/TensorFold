// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Side-call park for the single resident GLM-5.3 conversation (TF_GLM53_SIDE_ROWS). Host bookkeeping; Engine copies the rows.
const std = @import("std");
const reusableRows = @import("engine.zig").Engine.reusableRows;

pub const Range = struct { lo: usize, hi: usize };
pub const Slots = struct { first: usize, count: usize };
pub const Plan = struct { unpark: bool = false, park: ?Range = null };
pub const Restored = struct { frontier: usize, copy: ?Slots };

pub fn slotsOf(r: Range) Slots {
    return .{ .first = r.lo / 2, .count = (r.hi + 1) / 2 - r.lo / 2 };
}

pub fn effectiveFrontier(frontier: usize, lo: usize, hi: usize, write_high: usize, adapt_floor: ?usize) usize {
    var f = frontier;
    if (write_high > hi) f = @min(f, hi);
    if (adapt_floor) |a| f = if (a <= lo) @min(f, lo) else @min(f, @max(hi, a));
    return f;
}

pub const Park = struct {
    rows: usize,
    long_from: usize = 0,
    long_drops: u64 = 0,
    adapt_limit: usize = 0,
    held: bool = false,
    lo: usize = 0,
    hi: usize = 0,
    frontier: usize = 0,
    write_high: usize = 0,
    adapt_hit: bool = false,
    history: std.ArrayList(u32) = .empty,
    saves: u64 = 0,
    restores: u64 = 0,

    pub fn beforeWrite(p: *Park, end: usize) bool {
        if (p.long_from != 0 and p.held and end > p.long_from) {
            p.held = false;
            p.history.clearRetainingCapacity();
            p.long_drops += 1;
            return true;
        }
        return false;
    }

    pub fn deinit(p: *Park, gpa: std.mem.Allocator) void {
        p.history.deinit(gpa);
    }

    pub fn notePrefill(p: *Park, start: usize, end: usize) void {
        if (!p.held) return;
        p.write_high = @max(p.write_high, end);
        if (p.adapt_limit != 0 and start < p.adapt_limit) p.adapt_hit = true;
    }

    pub fn noteDecode(p: *Park, end: usize) void {
        if (p.held) p.write_high = @max(p.write_high, end);
    }

    pub fn effective(p: *const Park) usize {
        return effectiveFrontier(p.frontier, p.lo, p.hi, p.write_high, if (p.adapt_hit) p.adapt_limit else null);
    }

    pub fn plan(p: *const Park, history: []const u32, frontier: usize, prompt: []const u32, piece_rows: usize) Plan {
        if (p.rows == 0 or piece_rows == 0 or prompt.len == 0) return .{};
        var h = history;
        var f = frontier;
        var out: Plan = .{};
        if (p.held) {
            const pf = p.effective();
            const ra = reusableRows(history, frontier, prompt, piece_rows);
            const rp = reusableRows(p.history.items, pf, prompt, piece_rows);
            if (rp <= ra and ra >= p.lo) return .{};
            out.unpark = true;
            h = p.history.items[0..pf];
            f = pf;
        }
        const r = reusableRows(h, f, prompt, piece_rows);
        if (f >= r + piece_rows) {
            const hi = @min(f, r + p.rows);
            if (p.long_from == 0 or hi <= p.long_from) out.park = .{ .lo = r, .hi = hi };
        }
        return out;
    }

    pub fn save(p: *Park, gpa: std.mem.Allocator, history: []const u32, pos: usize, frontier: usize, r: Range, piece_rows: usize) !Slots {
        if (p.held) return error.ParkHeld;
        if (pos != frontier) return error.ParkNotSettled;
        if (piece_rows == 0 or r.lo % piece_rows != 0 or r.hi <= r.lo or r.hi > frontier or r.hi - r.lo > p.rows or frontier > history.len) return error.BadParkRange;
        p.history.clearRetainingCapacity();
        try p.history.appendSlice(gpa, history[0..frontier]);
        p.lo = r.lo;
        p.hi = r.hi;
        p.frontier = frontier;
        p.write_high = 0;
        p.adapt_hit = false;
        p.held = true;
        p.saves += 1;
        return slotsOf(r);
    }

    pub fn restore(p: *Park, history: *std.ArrayList(u32)) !Restored {
        if (!p.held) return error.NothingParked;
        const f = p.effective();
        std.mem.swap(std.ArrayList(u32), history, &p.history);
        history.shrinkRetainingCapacity(f);
        p.history.clearRetainingCapacity();
        p.held = false;
        p.write_high = 0;
        p.adapt_hit = false;
        p.restores += 1;
        const top: usize = @min(p.hi, f);
        return .{ .frontier = f, .copy = if (top > p.lo) slotsOf(.{ .lo = p.lo, .hi = top }) else null };
    }
};

test "park selection keeps active prefixes and saves deltas" {
    var h: [9000]u32 = undefined;
    for (&h, 0..) |*v, i| v.* = @intCast(i + 1);
    var pk: Park = .{ .rows = 2048 };
    try std.testing.expectEqual(Plan{}, pk.plan(&h, 8300, h[0..], 512));
    var other: [1025]u32 = @splat(0);
    try std.testing.expectEqual(Range{ .lo = 0, .hi = 2048 }, pk.plan(&h, 8300, &other, 512).park.?);
    for (&other, 0..) |*v, i| v.* = h[i];
    other[1024] = 77;
    try std.testing.expectEqual(@as(usize, 1024), reusableRows(&h, 8300, &other, 512));
}

test "park effective frontier accounts for writes and adapt tails" {
    try std.testing.expectEqual(@as(usize, 9000), effectiveFrontier(9000, 1024, 3072, 3072, null));
    try std.testing.expectEqual(@as(usize, 3072), effectiveFrontier(9000, 1024, 3072, 3073, null));
    try std.testing.expectEqual(@as(usize, 4096), effectiveFrontier(9000, 1024, 3072, 0, 4096));
    try std.testing.expectEqual(@as(usize, 3072), effectiveFrontier(9000, 1024, 3072, 0, 2048));
    try std.testing.expectEqual(@as(usize, 1024), effectiveFrontier(9000, 1024, 3072, 0, 512));
}

test "park selection preserves the held conversation and evicts in the specified order" {
    const gpa = std.testing.allocator;
    var main: std.ArrayList(u32) = .empty;
    defer main.deinit(gpa);
    var side: std.ArrayList(u32) = .empty;
    defer side.deinit(gpa);
    var next: std.ArrayList(u32) = .empty;
    defer next.deinit(gpa);
    try main.appendNTimes(gpa, 1, 4096);
    try side.appendNTimes(gpa, 2, 1024);
    try next.appendNTimes(gpa, 3, 1024);
    var pk: Park = .{ .rows = 2048 };
    defer pk.deinit(gpa);
    _ = try pk.save(gpa, main.items, 4096, 4096, .{ .lo = 0, .hi = 2048 }, 512);
    // Another unrelated side prompt leaves the parked main conversation alone.
    try std.testing.expectEqual(Plan{}, pk.plan(side.items, 1024, next.items, 512));
    // The main prompt is a strictly better match for the parked history.
    try std.testing.expect(pk.plan(side.items, 1024, main.items, 512).unpark);
    // If active has a larger match than the parked range, ties stay on active.
    try std.testing.expectEqual(Plan{}, pk.plan(main.items, 4096, main.items, 512));
}

test "park save restore swaps the parked history and bounds copied slots" {
    const gpa = std.testing.allocator;
    var pk: Park = .{ .rows = 1024 };
    defer pk.deinit(gpa);
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendNTimes(gpa, 7, 4096);
    try std.testing.expectError(error.ParkNotSettled, pk.save(gpa, h.items, 1, 4096, .{ .lo = 0, .hi = 512 }, 512));
    try std.testing.expectError(error.BadParkRange, pk.save(gpa, h.items, 4096, 4096, .{ .lo = 1, .hi = 512 }, 512));
    try std.testing.expectError(error.BadParkRange, pk.save(gpa, h.items, 4096, 4096, .{ .lo = 0, .hi = 2048 }, 512));
    _ = try pk.save(gpa, h.items, 4096, 4096, .{ .lo = 0, .hi = 1024 }, 512);
    try std.testing.expectError(error.ParkHeld, pk.save(gpa, h.items, 4096, 4096, .{ .lo = 0, .hi = 512 }, 512));
    var side: std.ArrayList(u32) = .empty;
    defer side.deinit(gpa);
    try side.appendNTimes(gpa, 9, 512);
    const got = try pk.restore(&side);
    try std.testing.expectEqual(@as(usize, 4096), got.frontier);
    try std.testing.expectEqualSlices(u32, h.items, side.items);
    try std.testing.expect(!pk.held);
    try std.testing.expectEqual(@as(usize, 512), slotsOf(.{ .lo = 0, .hi = 1023 }).count);
    try std.testing.expectEqual(@as(usize, 512), slotsOf(.{ .lo = 0, .hi = 1024 }).count);
    try std.testing.expectError(error.NothingParked, pk.restore(&h));

    var clipped: Park = .{ .rows = 1024, .adapt_limit = 512 };
    defer clipped.deinit(gpa);
    _ = try clipped.save(gpa, h.items, 4096, 4096, .{ .lo = 1024, .hi = 2048 }, 512);
    clipped.notePrefill(0, 4096);
    const clipped_result = try clipped.restore(&h);
    try std.testing.expect(clipped_result.frontier <= clipped.lo);
    try std.testing.expect(clipped_result.copy == null);
}
