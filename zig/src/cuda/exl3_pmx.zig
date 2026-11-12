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
/// tf_glm53_pmx_gu_mb5l2_e4m3: 5 ring slots of one 19,456 B gate/up stage plus a full/empty mbarrier pair a slot.
pub const gu_mb5_smem: usize = 5 * 19_456 + 5 * 16;
/// Seven stages of e4m3 X (2*128*32) and trellis words (2*2*8*24*4), plus one fp16 expand buffer (2*128*32*2).
pub const gu_a8_smem: usize = 7 * (2 * bm * 32 + 2 * 2 * 8 * 24 * 4) + 2 * bm * 32 * 2;
/// Two stages. One stage is a 128-K scale block: e4m3 X (2*128*128) and eight trellis tiles (2*8*8*24*4).
pub const gu_w8_smem: usize = 2 * (2 * bm * 128 + 2 * 8 * 8 * 24 * 4);
pub const dn_smem: usize = 38_912;
/// TF_GLM53_DOWN_W8S: three stages. One stage is a 128-K block: e4m3 X (128*128) and 8 x 16 trellis tiles (8*16*24*4).
pub const down_w8_smem: usize = 3 * (bm * 128 + 8 * 16 * 24 * 4);
/// gu_w8s knockout masks, in the order the glm53-ra8-ko bench prints them.
pub const gu_w8s_ko_masks = [_]u8{ 1, 2, 4, 8, 16, 3, 9, 32, 35, 39, 64, 103, 119, 128 };
const gu_w8s_ko_syms = [_][:0]const u8{
    "tf_glm53_pmx_gu_w8s_ko1_e4m3",
    "tf_glm53_pmx_gu_w8s_ko2_e4m3",
    "tf_glm53_pmx_gu_w8s_ko4_e4m3",
    "tf_glm53_pmx_gu_w8s_ko8_e4m3",
    "tf_glm53_pmx_gu_w8s_ko16_e4m3",
    "tf_glm53_pmx_gu_w8s_ko3_e4m3",
    "tf_glm53_pmx_gu_w8s_ko9_e4m3",
    "tf_glm53_pmx_gu_w8s_ko32_e4m3",
    "tf_glm53_pmx_gu_w8s_ko35_e4m3",
    "tf_glm53_pmx_gu_w8s_ko39_e4m3",
    "tf_glm53_pmx_gu_w8s_ko64_e4m3",
    "tf_glm53_pmx_gu_w8s_ko103_e4m3",
    "tf_glm53_pmx_gu_w8s_ko119_e4m3",
    "tf_glm53_pmx_gu_w8s_ko128_e4m3",
};

/// down_w8s knockout masks, in the order the glm53-dn-ko bench prints them: KO1 NODECODE,
/// KO2 NOMMA, KO4 NOLOADS, KO16 NOEPI, KO32 NOLDSM, KO64 NOBAR, KO35 = 1|2|32, KO39 = 1|2|4|32.
pub const dn_ko_masks = [_]u8{ 1, 2, 4, 16, 32, 64, 35, 39 };
const dn_ko_syms = [_][:0]const u8{
    "tf_glm53_pmx_down_w8s_ko1_e4m3",
    "tf_glm53_pmx_down_w8s_ko2_e4m3",
    "tf_glm53_pmx_down_w8s_ko4_e4m3",
    "tf_glm53_pmx_down_w8s_ko16_e4m3",
    "tf_glm53_pmx_down_w8s_ko32_e4m3",
    "tf_glm53_pmx_down_w8s_ko64_e4m3",
    "tf_glm53_pmx_down_w8s_ko35_e4m3",
    "tf_glm53_pmx_down_w8s_ko39_e4m3",
};

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

pub const PmxA8Args = extern struct {
    x0: u64 = 0,
    x1: u64 = 0,
    s0: u64 = 0,
    s1: u64 = 0,
    tp0: u64 = 0,
    tp1: u64 = 0,
    k2_0: u64 = 0,
    k2_1: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    members: u64 = 0,
    sv0: u64 = 0,
    sv1: u64 = 0,
    sd: u64 = 0,
    xd: u64 = 0,
    y: u64 = 0,
    K: i32 = 0,
    N: i32 = 0,
    E: i32 = 0,
    limit: f32 = 0,
    act_mode: i32 = 0,
    /// gu_w8s only. 0: X row r at r * K, scale at s[r]. Else X row and scale at byte r * xs of x and s.
    xs: i32 = 0,
    /// gu_w8s only. Nonzero: items of <= 64 members run here too.
    all: i32 = 0,
};

/// tf_glm53_pmx_down_w8s_e4m3 (pmx.cu DownW8Args): codes row p and its fp32 scale at byte p * xs and p * xs + K of xq.
pub const DownW8Args = extern struct {
    xq: u64 = 0,
    td: u64 = 0,
    k2: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    members: u64 = 0,
    svh: u64 = 0,
    y: u64 = 0,
    K: i32 = 0,
    D: i32 = 0,
    xs: i32 = 0,
};

pub const Pmx = struct {
    mod: Module,
    plan_mod: Module,
    router: grouped.Router,
    gu: [2]Function,
    /// Dynamic shared memory of each `gu` entry.
    gu_shared: [2]usize = .{ gu_smem, gu_smem },
    dn: [2]Function,
    dequant: [4]Function,
    gu_a8: Function = undefined,
    gu_w8: Function = undefined,
    gu_w8s: Function = undefined,
    /// gu_w8s with the k32 decode split into two pairs, interleaved with the MMA sweep. Same bits and smem.
    gu_w8s_pf: Function = undefined,
    /// gu_w8s timing ablations indexed by gu_w8s_ko_masks: KO1, KO2, KO4, KO8, KO16, KO3, KO9,
    /// KO32 (NOLDSM), KO35, KO39, KO64 (NOBAR), KO103, KO119, KO128 (NOLOOPADDR).
    gu_w8s_ko: [gu_w8s_ko_masks.len]Function = undefined,
    /// gu_w8s with only the item/count reads: launch and grid cost.
    gu_w8s_empty: Function = undefined,
    /// TF_GLM53_DOWN_W8S: tf_glm53_pmx_xd_w8s (xd -> e4m3 codes + one scale a pair) and the e4m3 down.
    xd_w8s: Function = undefined,
    down_w8s: Function = undefined,
    /// TF_GLM53_DN_EPI=1: down_w8s with the paired-FWHT, vector-store epilogue (same bytes).
    down_w8s_epi: Function = undefined,
    /// down_w8s timing ablations indexed by dn_ko_masks.
    down_w8s_ko: [dn_ko_masks.len]Function = undefined,
    /// down_w8s with only the item/count reads: launch and grid cost.
    down_w8s_empty: Function = undefined,

    pub fn load(d: *const Driver) !Pmx {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var p: Pmx = .{ .mod = try Module.load(d, kernels.glm53_pmx), .plan_mod = undefined, .router = undefined, .gu = undefined, .dn = undefined, .dequant = undefined };
        errdefer p.mod.unload();
        p.plan_mod = try Module.load(d, kernels.experts);
        errdefer p.plan_mod.unload();
        p.router = try grouped.Router.resolve(p.plan_mod);
        p.gu = .{ try p.mod.function("tf_glm53_pmx_gu_mcg"), try p.mod.function("tf_glm53_pmx_gu_e4m3") };
        for (p.gu) |f| try f.allowDynamicShared(gu_smem);
        p.gu_a8 = try p.mod.function("tf_glm53_pmx_gu_a8_e4m3");
        try p.gu_a8.allowDynamicShared(@as(u32, @intCast(gu_a8_smem)));
        p.gu_w8 = try p.mod.function("tf_glm53_pmx_gu_w8_e4m3");
        try p.gu_w8.allowDynamicShared(@as(u32, @intCast(gu_w8_smem)));
        p.gu_w8s = try p.mod.function("tf_glm53_pmx_gu_w8s_e4m3");
        try p.gu_w8s.allowDynamicShared(@as(u32, @intCast(gu_w8_smem)));
        p.gu_w8s_pf = try p.mod.function("tf_glm53_pmx_gu_w8s_pf_e4m3");
        try p.gu_w8s_pf.allowDynamicShared(@as(u32, @intCast(gu_w8_smem)));
        for (gu_w8s_ko_syms, 0..) |sym, i| {
            p.gu_w8s_ko[i] = try p.mod.function(sym);
            try p.gu_w8s_ko[i].allowDynamicShared(@as(u32, @intCast(gu_w8_smem)));
        }
        p.gu_w8s_empty = try p.mod.function("tf_glm53_pmx_gu_w8s_empty_e4m3");
        try p.gu_w8s_empty.allowDynamicShared(@as(u32, @intCast(gu_w8_smem)));
        p.xd_w8s = try p.mod.function("tf_glm53_pmx_xd_w8s");
        p.down_w8s = try p.mod.function("tf_glm53_pmx_down_w8s_e4m3");
        try p.down_w8s.allowDynamicShared(@as(u32, @intCast(down_w8_smem)));
        p.down_w8s_epi = try p.mod.function("tf_glm53_pmx_down_w8s_epi_e4m3");
        try p.down_w8s_epi.allowDynamicShared(@as(u32, @intCast(down_w8_smem)));
        for (dn_ko_syms, 0..) |sym, i| {
            p.down_w8s_ko[i] = try p.mod.function(sym);
            try p.down_w8s_ko[i].allowDynamicShared(@as(u32, @intCast(down_w8_smem)));
        }
        p.down_w8s_empty = try p.mod.function("tf_glm53_pmx_down_w8s_empty_e4m3");
        try p.down_w8s_empty.allowDynamicShared(@as(u32, @intCast(down_w8_smem)));
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

    /// TF_GLM53_GU_MB=1: the e4m3 gate/up entry becomes the 5-slot mbarrier ring with prefetch lead 2 (same bits).
    pub fn useGuMb(p: *Pmx) !void {
        const f = try p.mod.function("tf_glm53_pmx_gu_mb5l2_e4m3");
        try f.allowDynamicShared(gu_mb5_smem);
        p.gu[1] = f;
        p.gu_shared[1] = gu_mb5_smem;
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
    try std.testing.expectEqual(@as(usize, 97_360), gu_mb5_smem);
    try std.testing.expect(gu_mb5_smem + 128 * @sizeOf(i32) <= 101_376);
    try std.testing.expectEqual(@as(usize, 384), grouped.maxItems(16384, 256, 128));
}

test "a8 gate-up shared memory matches the kernel and fits sm_89" {
    try std.testing.expectEqual(@as(usize, 95_232), gu_a8_smem);
    try std.testing.expect(gu_a8_smem <= 101_376);
}

test "w8 gate-up shared memory matches the kernel and fits sm_89" {
    try std.testing.expectEqual(@as(usize, 90_112), gu_w8_smem);
    try std.testing.expect(gu_w8_smem <= 101_376);
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

test "PmxA8Args layout matches the CUDA struct" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(PmxA8Args, "x0"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(PmxA8Args, "x1"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(PmxA8Args, "s0"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(PmxA8Args, "s1"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(PmxA8Args, "tp0"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(PmxA8Args, "tp1"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(PmxA8Args, "k2_0"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(PmxA8Args, "k2_1"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(PmxA8Args, "items"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(PmxA8Args, "counts"));
    try std.testing.expectEqual(@as(usize, 80), @offsetOf(PmxA8Args, "members"));
    try std.testing.expectEqual(@as(usize, 88), @offsetOf(PmxA8Args, "sv0"));
    try std.testing.expectEqual(@as(usize, 96), @offsetOf(PmxA8Args, "sv1"));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(PmxA8Args, "sd"));
    try std.testing.expectEqual(@as(usize, 112), @offsetOf(PmxA8Args, "xd"));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(PmxA8Args, "y"));
    try std.testing.expectEqual(@as(usize, 128), @offsetOf(PmxA8Args, "K"));
    try std.testing.expectEqual(@as(usize, 132), @offsetOf(PmxA8Args, "N"));
    try std.testing.expectEqual(@as(usize, 136), @offsetOf(PmxA8Args, "E"));
    try std.testing.expectEqual(@as(usize, 140), @offsetOf(PmxA8Args, "limit"));
    try std.testing.expectEqual(@as(usize, 144), @offsetOf(PmxA8Args, "act_mode"));
    try std.testing.expectEqual(@as(usize, 148), @offsetOf(PmxA8Args, "xs"));
    try std.testing.expectEqual(@as(usize, 160), @sizeOf(PmxA8Args));
}

test "DownW8Args mirrors pmx.cu DownW8Args" {
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(DownW8Args, "y"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(DownW8Args, "K"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(DownW8Args, "xs"));
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(DownW8Args));
    try std.testing.expectEqual(@as(usize, 86_016), down_w8_smem);
    try std.testing.expect(down_w8_smem + 128 * 4 <= 101_376);
}
