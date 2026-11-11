// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 DSA taps: copy selected residual hidden rows into the drafter buffer.
const cuda = @import("cuda");

/// TF_GLM53_DRAFT_INGEST_SIDE: the drafter's verify ingest, forked after the last tap layer, joined after the head.
pub const Ingest = struct {
    ctx: *anyopaque,
    fork_fn: *const fn (ctx: *anyopaque, s: cuda.Stream, positions: u64, rows: usize) anyerror!void,
    join_fn: *const fn (ctx: *anyopaque, s: cuda.Stream) anyerror!void,
};

pub const default_layers = [_]usize{ 5, 19, 33, 47, 61, 75 };

pub const Taps = struct {
    f: cuda.Function,
    out: u64,
    hidden: usize = 6144,
    layers: []const usize = &default_layers,
    from: usize = 0,
    cap_rows: usize,
    ingest: ?Ingest = null,
};

pub fn at(t: Taps, s: cuda.Stream, layer: usize, x: u64, first: usize, rows: usize) !void {
    const slot = slotOf(t.layers, layer) orelse return;
    const lo = t.from -| first;
    if (lo >= rows) return;
    const n = rows - lo;
    const dst = first + lo - t.from;
    if (dst + n > t.cap_rows) return error.TapRowsOverflow;
    var a: cuda.Args = .{};
    a.add(x + lo * t.hidden * 2);
    a.add(t.out + dst * t.layers.len * t.hidden * 2);
    a.add(@as(i32, @intCast(slot)));
    a.add(@as(i32, 1));
    a.add(@as(i32, @intCast(t.hidden)));
    a.add(@as(i32, @intCast(t.layers.len)));
    try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(n), .y = @intCast(t.hidden / 1024) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn slotOf(layers: []const usize, layer: usize) ?usize {
    for (layers, 0..) |target, slot| if (target == layer) return slot;
    return null;
}

test "glm dsa dflash taps map configured layer ids to slots" {
    try @import("std").testing.expectEqual(@as(?usize, 0), slotOf(&default_layers, 5));
    try @import("std").testing.expectEqual(@as(?usize, 5), slotOf(&default_layers, 75));
    try @import("std").testing.expectEqual(@as(?usize, null), slotOf(&default_layers, 14));
}

test "glm dsa dflash taps skip rows before the requested tap start" {
    const t: Taps = .{ .f = undefined, .out = 0, .from = 100, .cap_rows = 50 };
    try at(t, undefined, 5, 0, 0, 50);
}

test "glm dsa dflash taps reject output row overflow before launching" {
    const t: Taps = .{ .f = undefined, .out = 0, .cap_rows = 32 };
    try @import("std").testing.expectError(error.TapRowsOverflow, at(t, undefined, 5, 0, 0, 40));
}

test {
    @import("std").testing.refAllDecls(@This());
}
