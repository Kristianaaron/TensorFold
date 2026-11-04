// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Pure per-round allocation for multi-stream verification; currently tested independently and not wired into the GLM single-window adapter.
const std = @import("std");

pub const Want = struct { id: u32, drafts: u32, rate: f64 };

pub fn allocate(wants: []const Want, cap: u32, floor_rate: f64, out: []u32) void {
    @memset(out, 0);
    if (out.len != wants.len or cap < wants.len) return;
    var remaining = cap - @as(u32, @intCast(wants.len));
    while (remaining > 0) {
        var best: ?usize = null;
        var best_value: f64 = -1;
        for (wants, 0..) |w, i| {
            if (out[i] >= w.drafts) continue;
            const value = std.math.pow(f64, std.math.clamp(w.rate, 0, 1), @as(f64, @floatFromInt(out[i] + 1)));
            if (value < floor_rate) continue;
            if (best == null or value > best_value) { best = i; best_value = value; }
        }
        const i = best orelse break;
        out[i] += 1;
        remaining -= 1;
    }
}

test "verify budget respects cap, ranks marginal yield, cutoff and determinism" {
    const wants = [_]Want{ .{ .id = 1, .drafts = 4, .rate = 0.9 }, .{ .id = 2, .drafts = 4, .rate = 0.5 } };
    var a: [2]u32 = undefined;
    var b: [2]u32 = undefined;
    allocate(&wants, 5, 0.2, &a);
    allocate(&wants, 5, 0.2, &b);
    try std.testing.expectEqualSlices(u32, &a, &b);
    try std.testing.expectEqual(@as(u32, 3), a[0] + a[1]);
    try std.testing.expect(a[0] > a[1]);
    allocate(&wants, 10, 0.85, &a);
    try std.testing.expectEqual(@as(u32, 1), a[0]);
    try std.testing.expectEqual(@as(u32, 0), a[1]);
    allocate(&wants, 1, 0, &a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, &a);
    const tied = [_]Want{ .{ .id = 9, .drafts = 2, .rate = 0.5 }, .{ .id = 1, .drafts = 2, .rate = 0.5 } };
    allocate(&tied, 3, 0, &a);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, &a);
}
