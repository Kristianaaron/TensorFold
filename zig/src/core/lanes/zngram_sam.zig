// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");

const Edge = struct { tok: u32, to: u32, next: i32 };

pub const Sam = struct {
    gpa: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    length: std.ArrayList(u32) = .empty,
    link: std.ArrayList(i32) = .empty,
    endp: std.ArrayList(i32) = .empty,
    head: std.ArrayList(i32) = .empty,
    edges: std.ArrayList(Edge) = .empty,
    map: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    last: u32 = 0,
    ml: u32 = 0,
    me: i32 = -1,

    fn key(state: u32, tok: u32) u64 {
        return (@as(u64, state) << 32) | tok;
    }
    fn addState(s: *Sam, state_len: u32, lk: i32, ep: i32) !u32 {
        const id: u32 = @intCast(s.length.items.len);
        try s.length.append(s.gpa, state_len);
        try s.link.append(s.gpa, lk);
        try s.endp.append(s.gpa, ep);
        try s.head.append(s.gpa, -1);
        return id;
    }
    fn trans(s: *const Sam, state: u32, tok: u32) ?u32 {
        const idx = s.map.get(key(state, tok)) orelse return null;
        return s.edges.items[@intCast(idx)].to;
    }
    fn put(s: *Sam, state: u32, tok: u32, to: u32) !void {
        const k = key(state, tok);
        if (s.map.get(k)) |idx| {
            s.edges.items[@intCast(idx)].to = to;
            return;
        }
        const idx: u32 = @intCast(s.edges.items.len);
        try s.edges.append(s.gpa, .{ .tok = tok, .to = to, .next = s.head.items[@intCast(state)] });
        s.head.items[@intCast(state)] = @intCast(idx);
        try s.map.put(s.gpa, k, idx);
    }
    pub fn init(gpa: std.mem.Allocator) !Sam {
        var s = Sam{ .gpa = gpa };
        _ = try s.addState(0, -1, -1);
        return s;
    }
    pub fn deinit(s: *Sam) void {
        s.toks.deinit(s.gpa); s.length.deinit(s.gpa); s.link.deinit(s.gpa);
        s.endp.deinit(s.gpa); s.head.deinit(s.gpa); s.edges.deinit(s.gpa);
        s.map.deinit(s.gpa);
    }
    pub fn reset(s: *Sam) void {
        s.toks.clearRetainingCapacity(); s.length.clearRetainingCapacity();
        s.link.clearRetainingCapacity(); s.endp.clearRetainingCapacity();
        s.head.clearRetainingCapacity(); s.edges.clearRetainingCapacity();
        s.map.clearRetainingCapacity();
        s.length.append(s.gpa, 0) catch unreachable;
        s.link.append(s.gpa, -1) catch unreachable;
        s.endp.append(s.gpa, -1) catch unreachable;
        s.head.append(s.gpa, -1) catch unreachable;
        s.last = 0; s.ml = 0; s.me = -1;
    }
    pub fn append(s: *Sam, tok: u32) !void {
        const i: i32 = @intCast(s.toks.items.len);
        try s.toks.append(s.gpa, tok);
        const cur = try s.addState(s.length.items[@intCast(s.last)] + 1, 0, i);
        var p: i32 = @intCast(s.last);
        while (p != -1 and s.trans(@intCast(p), tok) == null) {
            try s.put(@intCast(p), tok, cur);
            p = s.link.items[@intCast(p)];
        }
        if (p == -1) {
            s.link.items[@intCast(cur)] = 0;
        } else {
            const q = s.trans(@intCast(p), tok).?;
            if (s.length.items[@intCast(p)] + 1 == s.length.items[@intCast(q)]) {
                s.link.items[@intCast(cur)] = @intCast(q);
            } else {
                const clone = try s.addState(s.length.items[@intCast(p)] + 1, s.link.items[@intCast(q)], s.endp.items[@intCast(q)]);
                var edge = s.head.items[@intCast(q)];
                while (edge != -1) : (edge = s.edges.items[@intCast(edge)].next) {
                    const e = s.edges.items[@intCast(edge)];
                    try s.put(clone, e.tok, e.to);
                }
                while (p != -1 and s.trans(@intCast(p), tok) == q) {
                    try s.put(@intCast(p), tok, clone);
                    p = s.link.items[@intCast(p)];
                }
                s.link.items[@intCast(q)] = @intCast(clone);
                s.link.items[@intCast(cur)] = @intCast(clone);
            }
        }
        s.last = cur;
        const v = s.link.items[@intCast(cur)];
        if (v == 0) { s.ml = 0; s.me = -1; } else {
            s.ml = s.length.items[@intCast(v)]; s.me = s.endp.items[@intCast(v)];
            s.endp.items[@intCast(v)] = i;
        }
    }
    pub fn len(s: *const Sam) usize { return s.toks.items.len; }
    pub fn matchLen(s: *const Sam) u32 { return s.ml; }
    pub fn matchEnd(s: *const Sam) i32 { return s.me; }
    pub fn stateCount(s: *const Sam) usize { return s.length.items.len; }
    pub fn propose(s: *const Sam, k: usize, min_match: u32) []const u32 {
        if (k == 0 or s.ml == 0 or s.ml < min_match) return &.{};
        const start: usize = @intCast(s.me + 1);
        const count = @min(k, s.toks.items.len - start);
        return s.toks.items[start .. start + count];
    }
    pub fn sync(s: *Sam, context: []const u32) !usize {
        const n = s.len();
        var keep = n <= context.len;
        if (keep) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                if (s.toks.items[j] != context[j]) keep = false;
            }
            const check = @min(16, n);
            j = 0;
            while (j < check) : (j += 1) {
                if (s.toks.items[n - check + j] != context[n - check + j]) keep = false;
            }
        }
        var start = n;
        if (!keep) { s.reset(); start = 0; }
        for (context[start..]) |tok| try s.append(tok);
        return context.len - start;
    }

    /// Fast path for a session history that only grows by accepted tokens. Checking its old tail detects
    /// rollback/replacement without rescanning the full prompt on every draft round.
    pub fn syncAppend(s: *Sam, context: []const u32) !usize {
        const n = s.len();
        const check = @min(16, n);
        var keep = n <= context.len;
        if (keep) {
            for (0..check) |j| {
                if (s.toks.items[n - check + j] != context[n - check + j]) keep = false;
            }
        }
        var start = n;
        if (!keep) { s.reset(); start = 0; }
        for (context[start..]) |tok| try s.append(tok);
        return context.len - start;
    }
};
