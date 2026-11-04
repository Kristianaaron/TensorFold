// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! MK8 launch-shape policy: how the deferred executor (defer.zig) runs a segment, chosen once per plan key.
//! Every shape gives the same bits (the task bodies, arguments and order are those of the per-kernel launches);
//! the choice changes only the speed. No shape wins on every GPU and row count: on GB10 (48 SMs, sm_121) the
//! fastest shape differs for each of rows 1, 2, 4 and 8, and at rows=2 the plain launches are the fastest.
//!
//! Sources, in priority order:
//!   1. TF_GLM53_MK_SHAPE=<name> forces one shape (tests and A/B runs).
//!   2. The per-device table: (SM count, compute capability) and the rows bucket. The built-in entries are the
//!      measured winners of tf-cuda-test glm53-mk-policy.
//!   3. Calibration (TF_GLM53_MK_CALIBRATE=1): for a key with no table entry, the executor times each shape over a few
//!      eager windows (outside any graph capture) and keeps the fastest. TF_GLM53_MK_CALIBRATE=force also calibrates
//!      keys that have a table entry; =N (2..16) sets the windows per shape (default 3).
//!   4. The default: the chain scheduler with the SM worker mapping.
//! Original to TensorFold (MK8).
const std = @import("std");

/// The ways a deferred segment can run.
pub const Shape = enum(u8) {
    /// No fusion: the recorded launches are issued one by one, unchanged.
    plain,
    /// The dependency ready queue (tf_glm53_mk_run), the 48 KiB arena, one CTA per SM.
    queue,
    /// The ready queue at the plan's own arena, as many CTAs per SM as are resident.
    queue_occ,
    /// The chain scheduler (tf_glm53_mk_run_chain), worker = blockIdx.x.
    chain_block,
    /// The chain scheduler, block b and b + SMs on one SM (the default).
    chain_sm,

    /// The TF_GLM53_MK_SHAPE spelling.
    pub fn name(s: Shape) []const u8 {
        return switch (s) {
            .plain => "plain",
            .queue => "queue",
            .queue_occ => "queue-occ",
            .chain_block => "chain-block",
            .chain_sm => "chain-sm",
        };
    }

    pub fn parse(text: []const u8) ?Shape {
        for (all) |s| if (std.ascii.eqlIgnoreCase(text, s.name())) return s;
        return null;
    }
};

pub const all = [_]Shape{ .plain, .queue, .queue_occ, .chain_block, .chain_sm };
pub const shape_count = all.len;
pub const default_shape: Shape = .chain_sm;

/// What the table is keyed by: the SM count and the compute capability (10 * major + minor).
pub const Device = struct {
    sms: u32 = 0,
    cc: u32 = 0,
};

/// Row-count buckets: 1, 2, 3-4, 5 and more (decode windows hold at most 8 rows).
pub const bucket_rows = [_]u32{ 1, 2, 4, 8 };
pub const bucket_count = bucket_rows.len;

/// The bucket of a segment's row count; null when the row count is unknown (0).
pub fn bucket(rows: u32) ?u8 {
    if (rows == 0) return null;
    for (bucket_rows, 0..) |top, i| if (rows <= top) return @intCast(i);
    return bucket_count - 1;
}

pub const Entry = struct {
    sms: u32,
    cc: u32,
    /// One shape per rows bucket (bucket_rows).
    shapes: [bucket_count]Shape,
    /// Where the numbers come from (printed by tf-cuda-test glm53-mk-policy).
    note: []const u8 = "",
};

/// Measured with tf-cuda-test glm53-mk-defer / glm53-mk-policy (the eight-kernel decode slice, deferred + graph).
pub const builtin = [_]Entry{
    // GB10, step time in us per shape (plain / queue / queue-occ / chain-block / chain-sm):
    // rows=1 1670/1552/1549/1465/1454, rows=2 2831/3098/3189/3005/2968, rows=4 5240/5533/5055/4948/5678,
    // rows=8 13526/12450/11521/14670/14561.
    .{ .sms = 48, .cc = 121, .shapes = .{ .chain_sm, .plain, .chain_block, .queue_occ }, .note = "GB10" },
    // RTX 5070 Ti: chain-sm wins every bucket (rows=1/2/4/8: 1295/1487/3036/7197 us against 1540/1726/3092/7516 plain).
    .{ .sms = 70, .cc = 120, .shapes = @splat(.chain_sm), .note = "RTX 5070 Ti" },
};

pub fn lookup(table: []const Entry, device: Device, rows: u32) ?Shape {
    const b = bucket(rows) orelse return null;
    for (table) |e| if (e.sms == device.sms and e.cc == device.cc) return e.shapes[b];
    return null;
}

pub const Source = enum { override, table, calibration, default };

pub const Calibrate = enum { off, on, force };

pub const Decision = union(enum) {
    fixed: struct { shape: Shape, source: Source },
    calibrate,
};

pub const default_windows: u8 = 3;
/// A calibrated shape replaces the table or default shape only when it is faster by more than this fraction.
pub const keep_margin: f32 = 0.01;
pub const max_windows: u8 = 16;

pub const Policy = struct {
    table: []const Entry = &builtin,
    override: ?Shape = null,
    calibrate: Calibrate = .off,
    /// Eager windows per shape when calibrating.
    windows: u8 = default_windows,

    /// What a new plan key runs with.
    pub fn decide(p: Policy, device: Device, rows: u32) Decision {
        if (p.override) |s| return .{ .fixed = .{ .shape = s, .source = .override } };
        if (p.calibrate == .force) return .calibrate;
        if (lookup(p.table, device, rows)) |s| return .{ .fixed = .{ .shape = s, .source = .table } };
        if (p.calibrate == .on) return .calibrate;
        return .{ .fixed = .{ .shape = default_shape, .source = .default } };
    }

    /// The shape without calibration (table, else the default): what a key that has no measurement yet runs with.
    pub fn fallback(p: Policy, device: Device, rows: u32) Shape {
        if (p.override) |s| return s;
        return lookup(p.table, device, rows) orelse default_shape;
    }

    /// The policy from the environment's two values (null = unset).
    pub fn parse(shape: ?[]const u8, calibrate: ?[]const u8) error{ BadShape, BadCalibrate }!Policy {
        var p: Policy = .{};
        if (shape) |text| if (text.len != 0) {
            p.override = Shape.parse(text) orelse return error.BadShape;
        };
        if (calibrate) |text| {
            if (text.len == 0 or std.mem.eql(u8, text, "0")) {
                p.calibrate = .off;
            } else if (std.mem.eql(u8, text, "1")) {
                p.calibrate = .on;
            } else if (std.ascii.eqlIgnoreCase(text, "force")) {
                p.calibrate = .force;
            } else {
                const n = std.fmt.parseInt(u8, text, 10) catch return error.BadCalibrate;
                if (n < 2 or n > max_windows) return error.BadCalibrate;
                p.calibrate = .on;
                p.windows = n;
            }
        }
        return p;
    }

    /// TF_GLM53_MK_SHAPE and TF_GLM53_MK_CALIBRATE.
    pub fn fromEnv() error{ BadShape, BadCalibrate }!Policy {
        const shape: ?[]const u8 = if (std.c.getenv("TF_GLM53_MK_SHAPE")) |v| std.mem.span(v) else null;
        const cal: ?[]const u8 = if (std.c.getenv("TF_GLM53_MK_CALIBRATE")) |v| std.mem.span(v) else null;
        return parse(shape, cal) catch |err| {
            std.log.err("GLM-5.3 MK8: bad launch-shape environment (TF_GLM53_MK_SHAPE={s}, TF_GLM53_MK_CALIBRATE={s}); shapes: plain, queue, queue-occ, chain-block, chain-sm; calibrate: 0, 1, force, 2..16", .{ shape orelse "", cal orelse "" });
            return err;
        };
    }
};

/// One plan key's calibration: the shapes take turns (round-robin, so drift hits them alike) for `windows` eager
/// windows each; the winner has the smallest median.
pub const Calibration = struct {
    windows: u8,
    samples: [shape_count][max_windows]f32 = undefined,
    counts: [shape_count]u8 = @splat(0),
    issued: u32 = 0,

    pub fn init(windows: u8) Calibration {
        return .{ .windows = std.math.clamp(windows, 1, max_windows) };
    }

    /// The shape for the next eager window, or null when every shape has been issued `windows` times.
    pub fn next(c: *Calibration) ?Shape {
        if (c.issued >= @as(u32, c.windows) * shape_count) return null;
        const s = all[c.issued % shape_count];
        c.issued += 1;
        return s;
    }

    pub fn record(c: *Calibration, shape: Shape, us: f32) void {
        const i = @backingInt(shape);
        if (c.counts[i] >= c.windows) return;
        c.samples[i][c.counts[i]] = us;
        c.counts[i] += 1;
    }

    pub fn complete(c: *const Calibration) bool {
        for (c.counts) |n| if (n < c.windows) return false;
        return true;
    }

    pub fn median(c: *const Calibration, shape: Shape) ?f32 {
        const i = @backingInt(shape);
        const n = c.counts[i];
        if (n == 0) return null;
        var v: [max_windows]f32 = undefined;
        @memcpy(v[0..n], c.samples[i][0..n]);
        std.mem.sort(f32, v[0..n], {}, std.sort.asc(f32));
        return if (n % 2 == 1) v[n / 2] else (v[n / 2 - 1] + v[n / 2]) / 2;
    }

    /// The winner: the smallest median, unless `preferred` (the table or default shape) is within `margin` of it
    /// (window-to-window noise is about 0.5%, so near-ties keep the known shape). Null before any sample.
    pub fn pick(c: *const Calibration, preferred: Shape, margin: f32) ?Shape {
        const b = c.best() orelse return null;
        const keep = c.median(preferred) orelse return b;
        return if (c.median(b).? >= keep * (1 - margin)) preferred else b;
    }

    /// The measured shape with the smallest median (ties: the earlier shape in `all`); null before any sample.
    pub fn best(c: *const Calibration) ?Shape {
        var out: ?Shape = null;
        var t: f32 = std.math.inf(f32);
        for (all) |s| if (c.median(s)) |m| if (m < t) {
            t = m;
            out = s;
        };
        return out;
    }
};

const testing = std.testing;

test "rows buckets: 1, 2, 3-4, 5 and more; 0 is unknown" {
    try testing.expectEqual(@as(?u8, null), bucket(0));
    const want = [_]struct { u32, u8 }{ .{ 1, 0 }, .{ 2, 1 }, .{ 3, 2 }, .{ 4, 2 }, .{ 5, 3 }, .{ 8, 3 }, .{ 16, 3 }, .{ 1000, 3 } };
    for (want) |w| try testing.expectEqual(@as(?u8, w[1]), bucket(w[0]));
}

test "table lookup: GB10 picks a different shape per bucket; 70-SM sm_120 is chain-sm everywhere; others miss" {
    const gb10: Device = .{ .sms = 48, .cc = 121 };
    try testing.expectEqual(@as(?Shape, .chain_sm), lookup(&builtin, gb10, 1));
    try testing.expectEqual(@as(?Shape, .plain), lookup(&builtin, gb10, 2));
    try testing.expectEqual(@as(?Shape, .chain_block), lookup(&builtin, gb10, 3));
    try testing.expectEqual(@as(?Shape, .chain_block), lookup(&builtin, gb10, 4));
    try testing.expectEqual(@as(?Shape, .queue_occ), lookup(&builtin, gb10, 8));
    try testing.expectEqual(@as(?Shape, null), lookup(&builtin, gb10, 0));
    for ([_]u32{ 1, 2, 4, 8 }) |r| try testing.expectEqual(@as(?Shape, .chain_sm), lookup(&builtin, .{ .sms = 70, .cc = 120 }, r));
    // The key is both values: same SMs on another architecture, or another sm_120 part, is not in the table.
    try testing.expectEqual(@as(?Shape, null), lookup(&builtin, .{ .sms = 48, .cc = 120 }, 1));
    try testing.expectEqual(@as(?Shape, null), lookup(&builtin, .{ .sms = 170, .cc = 120 }, 1));
    // A custom table replaces the built-in one.
    const mine = [_]Entry{.{ .sms = 170, .cc = 120, .shapes = .{ .queue, .queue, .plain, .plain } }};
    try testing.expectEqual(@as(?Shape, .plain), lookup(&mine, .{ .sms = 170, .cc = 120 }, 8));
}

test "override and calibration parsing" {
    try testing.expectEqual(@as(?Shape, .queue_occ), Shape.parse("queue-occ"));
    try testing.expectEqual(@as(?Shape, .chain_block), Shape.parse("CHAIN-BLOCK"));
    try testing.expectEqual(@as(?Shape, null), Shape.parse("chain"));
    for (all) |s| try testing.expectEqual(@as(?Shape, s), Shape.parse(s.name()));
    var p = try Policy.parse("plain", null);
    try testing.expectEqual(@as(?Shape, .plain), p.override);
    try testing.expectEqual(Calibrate.off, p.calibrate);
    p = try Policy.parse(null, "1");
    try testing.expectEqual(@as(?Shape, null), p.override);
    try testing.expectEqual(Calibrate.on, p.calibrate);
    try testing.expectEqual(default_windows, p.windows);
    p = try Policy.parse("", "5");
    try testing.expectEqual(Calibrate.on, p.calibrate);
    try testing.expectEqual(@as(u8, 5), p.windows);
    p = try Policy.parse(null, "force");
    try testing.expectEqual(Calibrate.force, p.calibrate);
    p = try Policy.parse(null, "0");
    try testing.expectEqual(Calibrate.off, p.calibrate);
    try testing.expectError(error.BadShape, Policy.parse("fast", null));
    try testing.expectError(error.BadCalibrate, Policy.parse(null, "yes"));
    try testing.expectError(error.BadCalibrate, Policy.parse(null, "17"));
}

test "decide: override, then the table, then calibration, then the default" {
    const gb10: Device = .{ .sms = 48, .cc = 121 };
    const other: Device = .{ .sms = 84, .cc = 89 };
    const fixed = struct {
        fn f(d: Decision) ?struct { Shape, Source } {
            return switch (d) {
                .fixed => |x| .{ x.shape, x.source },
                .calibrate => null,
            };
        }
    }.f;
    try testing.expectEqual(.{ Shape.plain, Source.table }, fixed((Policy{}).decide(gb10, 2)).?);
    try testing.expectEqual(.{ Shape.chain_sm, Source.default }, fixed((Policy{}).decide(other, 2)).?);
    try testing.expectEqual(.{ Shape.queue, Source.override }, fixed((Policy{ .override = .queue, .calibrate = .force }).decide(gb10, 2)).?);
    // Calibration only where the table has nothing, unless forced.
    try testing.expectEqual(.{ Shape.plain, Source.table }, fixed((Policy{ .calibrate = .on }).decide(gb10, 2)).?);
    try testing.expect((Policy{ .calibrate = .on }).decide(other, 2) == .calibrate);
    try testing.expect((Policy{ .calibrate = .force }).decide(gb10, 2) == .calibrate);
    try testing.expect((Policy{ .calibrate = .on }).decide(gb10, 0) == .calibrate); // unknown rows: no table bucket
    try testing.expectEqual(Shape.plain, (Policy{}).fallback(gb10, 2));
    try testing.expectEqual(default_shape, (Policy{}).fallback(other, 2));
}

test "calibration: round-robin turns, median per shape, the smallest median wins" {
    var c = Calibration.init(3);
    var turns: [3 * shape_count]Shape = undefined;
    for (&turns) |*t| t.* = c.next().?;
    try testing.expectEqual(@as(?Shape, null), c.next());
    try testing.expectEqualSlices(Shape, &(all ++ all ++ all), &turns);
    try testing.expectEqual(@as(?Shape, null), c.best());
    // plain 10/11/90 (one outlier: median 11), queue 12, queue-occ 20, chain-block 10.5/11.5/9 (median 10.5), chain-sm 13.
    const times = [_][3]f32{ .{ 10, 11, 90 }, .{ 12, 12, 12 }, .{ 20, 20, 20 }, .{ 10.5, 11.5, 9 }, .{ 13, 13, 13 } };
    for (0..3) |w| for (all, times) |s, t| {
        try testing.expect(!c.complete());
        c.record(s, t[w]);
    };
    try testing.expect(c.complete());
    try testing.expectEqual(@as(?f32, 11), c.median(.plain));
    try testing.expectEqual(@as(?Shape, .chain_block), c.best());
    c.record(.plain, 1); // past `windows` samples: ignored
    try testing.expectEqual(@as(?f32, 11), c.median(.plain));
    // chain-block (10.5) is 4.5% faster than plain (11): it replaces it. Near-ties (within 1%) keep the preferred shape.
    try testing.expectEqual(@as(?Shape, .chain_block), c.pick(.plain, keep_margin));
    var near = Calibration.init(1);
    near.record(.chain_block, 1000);
    near.record(.chain_sm, 1005);
    near.record(.plain, 1200);
    try testing.expectEqual(@as(?Shape, .chain_sm), near.pick(.chain_sm, keep_margin));
    try testing.expectEqual(@as(?Shape, .chain_block), near.pick(.plain, keep_margin));
    near.record(.queue, 980); // 2.5% faster than chain-sm
    try testing.expectEqual(@as(?Shape, .queue), near.pick(.chain_sm, keep_margin));
    // A partial calibration picks among what it has.
    var p = Calibration.init(2);
    p.record(.queue, 5);
    p.record(.chain_sm, 4);
    try testing.expectEqual(@as(?Shape, .chain_sm), p.best());
    try testing.expectEqual(@as(?f32, 4.5), blk: {
        p.record(.chain_sm, 5);
        break :blk p.median(.chain_sm);
    });
}
