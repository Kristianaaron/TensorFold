// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Width-specialised GLM-5.3 EXL3 prompt expert kernels (P1 kernel package).
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;

pub const bm: usize = 128;
pub const gu_smem: usize = 77_824;
pub const dn_smem: usize = 38_912;

pub fn eligibleWidths(k2: []const i32) bool {
    if (k2.len == 0) return false;
    for (k2) |w| if (w != 4 and w != 6) return false;
    return true;
}

pub fn eligible(l: anytype) bool {
    return (l.codebook == .mcg or l.codebook == .mcg_e4m3) and l.pmx_ok and l.gu_stride == 0 and
        !l.shared_input and l.dims % bm == 0 and l.width % bm == 0 and (l.act_mode == 0 or l.act_mode == 1);
}

pub const Scratch = struct {
    xg: u64,
    xu: u64,
    xd: u64,
    y: u64,
    members: u64,
    items: u64,
    counts: u64,
    rank: u64,
    hist: u64,
    rows: usize,
    slots: usize,
};

pub fn bytes(l: anytype, rows: usize, slots: usize) !usize {
    const p = try std.math.mul(usize, rows, slots);
    const max_items = grouped.maxItems(p, l.experts, bm);
    const rank_bytes = try std.math.mul(usize, p, 4);
    const hist_bytes = try std.math.mul(usize, try std.math.mul(usize, (p + 1023) / 1024, l.experts), 4);
    return region(try std.math.mul(usize, p, l.dims * 2)) +
        region(try std.math.mul(usize, p, l.dims * 2)) +
        region(try std.math.mul(usize, p, l.width * 2)) +
        region(try std.math.mul(usize, p, l.dims * 4)) +
        region(try std.math.mul(usize, p, 4)) +
        region(try std.math.mul(usize, max_items, 12)) + region(16) + region(rank_bytes) + region(hist_bytes);
}

fn region(n: usize) usize { return (n + 255) / 256 * 256; }

pub fn scratch(base: u64, l: anytype, rows: usize, slots: usize) !Scratch {
    const p = try std.math.mul(usize, rows, slots);
    const max_items = grouped.maxItems(p, l.experts, bm);
    const hist_bytes = try std.math.mul(usize, try std.math.mul(usize, (p + 1023) / 1024, l.experts), 4);
    var at = base;
    const take = struct {
        fn f(ptr: *u64, n: usize) u64 {
            const out = ptr.*;
            ptr.* += region(n);
            return out;
        }
    }.f;
    const xg = take(&at, p * l.dims * 2);
    const xu = take(&at, p * l.dims * 2);
    const xd = take(&at, p * l.width * 2);
    const y = take(&at, p * l.dims * 4);
    const members = take(&at, p * 4);
    const items = take(&at, max_items * 12);
    const counts = take(&at, 16);
    const rank = take(&at, p * 4);
    const hist = take(&at, hist_bytes);
    return .{ .xg = xg, .xu = xu, .xd = xd, .y = y, .members = members, .items = items, .counts = counts, .rank = rank, .hist = hist, .rows = rows, .slots = slots };
}

pub const PmxArgs = extern struct {
    x0: u64,
    x1: u64,
    tp0: u64,
    tp1: u64,
    k2_0: u64,
    k2_1: u64,
    items: u64,
    counts: u64,
    members: u64,
    sv0: u64,
    sv1: u64,
    sd: u64,
    xd: u64,
    y: u64,
    K: i32,
    N: i32,
    E: i32,
    limit: f32,
    act_mode: i32,
};

pub const Pmx = struct {
    mod: Module,
    plan_mod: Module,
    router: grouped.Router,
    gu: [2]Function,
    dn: [2]Function,
    dequant: [4]Function,

    pub fn load(d: *const Driver) !Pmx {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var p: Pmx = .{ .mod = try Module.load(d, kernels.glm53_pmx), .plan_mod = undefined, .router = undefined, .gu = undefined, .dn = undefined, .dequant = undefined };
        errdefer p.mod.unload();
        p.plan_mod = try Module.load(d, kernels.experts);
        errdefer p.plan_mod.unload();
        p.router = try grouped.Router.resolve(p.plan_mod);
        p.gu = .{ try p.mod.function("tf_glm53_pmx_gu_mcg"), try p.mod.function("tf_glm53_pmx_gu_e4m3") };
        for (p.gu) |f| try f.allowDynamicShared(gu_smem);
        // TODO(P2): resolve down-kernel entries when the down/combine path is added.
        p.dn = .{ undefined, undefined };
        p.dequant = .{
            try p.mod.function("tf_glm53_pmx_dequant_4_mcg"),
            try p.mod.function("tf_glm53_pmx_dequant_6_mcg"),
            try p.mod.function("tf_glm53_pmx_dequant_4_e4m3"),
            try p.mod.function("tf_glm53_pmx_dequant_6_e4m3"),
        };
        return p;
    }

    pub fn unload(p: *Pmx) void {
        p.plan_mod.unload();
        p.mod.unload();
    }
};

test "width table accepts only nonempty 4 and 6 widths" {
    try std.testing.expect(eligibleWidths(&.{ 4, 6, 4 }));
    try std.testing.expect(!eligibleWidths(&.{ 4, 2 }));
    try std.testing.expect(!eligibleWidths(&.{8}));
    try std.testing.expect(!eligibleWidths(&.{}));
}

test "scratch regions align and cover allocation" {
    const Layer = struct { experts: usize, dims: usize, width: usize };
    const l: Layer = .{ .experts = 12, .dims = 256, .width = 384 };
    const sc = try scratch(0x1000, l, 17, 8);
    const ptrs = [_]u64{ sc.xg, sc.xu, sc.xd, sc.y, sc.members, sc.items, sc.counts, sc.rank, sc.hist };
    for (ptrs) |ptr| try std.testing.expectEqual(@as(u64, 0), ptr % 256);
    try std.testing.expectEqual(@as(u64, 0x1000), sc.xg);
    try std.testing.expectEqual(@as(usize, 0), (try bytes(l, 17, 8)) % 256);
    try std.testing.expect(sc.hist + @as(u64, @intCast(region(((17 * 8 + 1023) / 1024) * l.experts * 4))) <= 0x1000 + @as(u64, @intCast(try bytes(l, 17, 8))));
}

test "gate-up dynamic shared memory fits sm_89 limit" {
    try std.testing.expect(gu_smem <= 101_376);
    try std.testing.expectEqual(@as(usize, 384), grouped.maxItems(16384, 256, 128));
}

test "PmxArgs layout matches CUDA ABI" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(PmxArgs, "x0"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(PmxArgs, "x1"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(PmxArgs, "tp0"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(PmxArgs, "tp1"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(PmxArgs, "k2_0"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(PmxArgs, "k2_1"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(PmxArgs, "items"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(PmxArgs, "counts"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(PmxArgs, "members"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(PmxArgs, "sv0"));
    try std.testing.expectEqual(@as(usize, 80), @offsetOf(PmxArgs, "sv1"));
    try std.testing.expectEqual(@as(usize, 88), @offsetOf(PmxArgs, "sd"));
    try std.testing.expectEqual(@as(usize, 96), @offsetOf(PmxArgs, "xd"));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(PmxArgs, "y"));
    try std.testing.expectEqual(@as(usize, 112), @offsetOf(PmxArgs, "K"));
    try std.testing.expectEqual(@as(usize, 116), @offsetOf(PmxArgs, "N"));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(PmxArgs, "E"));
    try std.testing.expectEqual(@as(usize, 124), @offsetOf(PmxArgs, "limit"));
    try std.testing.expectEqual(@as(usize, 128), @offsetOf(PmxArgs, "act_mode"));
}
