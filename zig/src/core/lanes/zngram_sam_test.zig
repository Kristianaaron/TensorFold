// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");
const sam = @import("zngram_sam.zig");

fn oracle(s: *const sam.Sam) !void {
    const t = s.toks.items;
    const n = t.len;
    var best: usize = 0;
    var e: usize = 0;
    while (e + 1 < n) : (e += 1) {
        var matched: usize = 0;
        while (matched <= e and t[e - matched] == t[n - matched - 1]) : (matched += 1) {}
        best = @max(best, matched);
    }
    try std.testing.expectEqual(@as(u32, @intCast(best)), s.matchLen());
    if (best == 0) {
        try std.testing.expectEqualSlices(u32, &.{}, s.propose(5, 0));
        return;
    }
    const earlier_end: usize = @intCast(s.matchEnd());
    try std.testing.expect(earlier_end < n - 1);
    try std.testing.expectEqualSlices(u32, t[n - best ..], t[earlier_end + 1 - best .. earlier_end + 1]);
    const ks = [_]usize{ 0, 1, 5 };
    const ms = [_]u32{ 1, @intCast(best), @intCast(best + 1) };
    for (ks) |k| for (ms) |m| {
        if (k == 0 or m > best) {
            try std.testing.expectEqualSlices(u32, &.{}, s.propose(k, m));
        } else {
            const count = @min(k, n - 1 - earlier_end);
            try std.testing.expectEqualSlices(u32, t[earlier_end + 1 .. earlier_end + 1 + count], s.propose(k, m));
        }
    };
}

fn run(tokens: []const u32) !void {
    var s = try sam.Sam.init(std.testing.allocator);
    defer s.deinit();
    for (tokens) |tok| { try s.append(tok); try oracle(&s); }
}

test "examples and repeated token" {
    try run(&.{ 1, 2, 1, 2, 1 });
    var s = try sam.Sam.init(std.testing.allocator);
    defer s.deinit();
    try std.testing.expectEqual(@as(u32, 0), s.matchLen());
    try std.testing.expectEqual(@as(i32, -1), s.matchEnd());
    try std.testing.expectEqualSlices(u32, &.{}, s.propose(2, 1));
    const pairs = [_]struct { u32, i32 }{ .{ 0, -1 }, .{ 0, -1 }, .{ 1, 0 }, .{ 2, 1 }, .{ 3, 2 } };
    for ([_]u32{ 1, 2, 1, 2, 1 }, pairs) |tok, pair| {
        try s.append(tok);
        try std.testing.expectEqual(pair[0], s.matchLen());
        try std.testing.expectEqual(pair[1], s.matchEnd());
    }
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, s.propose(2, 1));
    s.reset();
    for ([_]u32{ 10, 11, 12, 13, 10, 11, 12, 14, 10, 11, 12 }) |tok| try s.append(tok);
    try std.testing.expectEqual(@as(u32, 3), s.matchLen());
    try std.testing.expectEqual(@as(i32, 6), s.matchEnd());
    try std.testing.expectEqualSlices(u32, &.{ 14, 10, 11, 12 }, s.propose(4, 1));
    try std.testing.expectEqualSlices(u32, &.{}, s.propose(4, 4));
    s.reset();
    var k: u32 = 1;
    while (k <= 50) : (k += 1) {
        try s.append(9);
        try std.testing.expectEqual(k - 1, s.matchLen());
        try std.testing.expectEqual(@as(i32, @intCast(k)) - 2, s.matchEnd());
        try std.testing.expect(s.stateCount() <= 2 * s.len());
    }
}

test "random oracle, sync, and large token ids" {
    var seed: u64 = 0x9e3779b97f4a7c15;
    var trial: usize = 0;
    while (trial < 100) : (trial += 1) {
        var buf: [150]u32 = undefined;
        const size = trial % 150 + 1;
        const alphabet: u64 = switch (trial % 4) { 0 => 2, 1 => 3, 2 => 5, else => 300 };
        for (buf[0..size]) |*v| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            v.* = @intCast(seed % alphabet);
        }
        try run(buf[0..size]);
    }
    try run(&.{ 0xffffffff, 1, 0xffffffff, 1 });
    var s = try sam.Sam.init(std.testing.allocator);
    defer s.deinit();
    var ctx: [55]u32 = undefined;
    for (&ctx, 0..) |*v, i| v.* = @intCast(i);
    _ = try s.sync(ctx[0..40]);
    try std.testing.expectEqual(@as(usize, 15), try s.sync(ctx[0..55]));
    ctx[10] = 999;
    try std.testing.expectEqual(@as(usize, 55), try s.sync(&ctx));
    try std.testing.expectEqual(@as(usize, 20), try s.sync(ctx[0..20]));
    for (&ctx, 0..) |*v, i| v.* = @intCast(i);
    _ = try s.sync(ctx[0..55]);
    try std.testing.expectEqual(@as(usize, 0), try s.sync(&ctx)); // same context: nothing to append
    ctx[10] = 999;
    try std.testing.expectEqual(@as(usize, 55), try s.sync(&ctx)); // spec: a changed context rebuilds
}

test "large stream state bound and recent scan" {
    var s = try sam.Sam.init(std.testing.allocator);
    defer s.deinit();
    var seed: u64 = 1234567;
    var i: usize = 0;
    while (i < 200000) : (i += 1) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        const tok: u32 = if (i % 1000 < 100) @intCast(i % 37) else @intCast(seed % 5000);
        try s.append(tok);
    }
    try std.testing.expect(s.stateCount() <= 2 * s.len());
    const n = s.len();
    const ml: usize = @intCast(s.matchLen());
    const e: usize = if (ml > 0) @intCast(s.matchEnd()) else 0;
    if (ml > 0) {
        try std.testing.expectEqualSlices(u32, s.toks.items[n - ml ..], s.toks.items[e - ml + 1 .. e + 1]);
        try std.testing.expect(e + 1 < n);
    }
    if (ml > 0) {
        const limit = @min(64, ml);
        var size: usize = 1;
        while (size <= limit) : (size += 1) {
            try std.testing.expectEqualSlices(u32, s.toks.items[n - size ..], s.toks.items[e - size + 1 .. e + 1]);
        }
    }
}

test "long session index uses linear host memory and append sync adds only the new suffix" {
    var s = try sam.Sam.init(std.testing.allocator);
    defer s.deinit();
    const n: usize = 20_000;
    const history = try std.testing.allocator.alloc(u32, n + 3);
    defer std.testing.allocator.free(history);
    for (history[0..n], 0..) |*token, i| token.* = @intCast((i * 31 + i / 7) % 997);
    try std.testing.expectEqual(n, try s.syncAppend(history[0..n]));
    try std.testing.expect(s.stateCount() <= 2 * n);
    try std.testing.expect(s.edges.items.len <= 3 * n);
    try std.testing.expectEqual(@as(usize, 0), try s.syncAppend(history[0..n]));
    history[n..][0] = 12;
    history[n..][1] = 13;
    history[n..][2] = 14;
    try std.testing.expectEqual(@as(usize, 3), try s.syncAppend(history));
    try std.testing.expectEqual(n + 3, s.len());
}
