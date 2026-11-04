// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 TF_GLM53_PROMPT_EXPERTS=2 (the wide 128-member prompt experts) against =1 at the deployed rank shapes.
//! `tf-cuda-test glm53-prompt-wide`: one routed layer of 256 experts, D 6144, rank width 1024, 8 slots, one 4096-row
//! piece; per-expert, per-matrix widths K1/K2/K3 (half-bits 2/4/6) with random trellis codes under mcg and mcg_e4m3;
//! skewed routing (experts with 0 rows, 1 row, 65-128 rows and hundreds of rows) and uniform routing.
//! Checks: =2's per-slot y and the weighted combine bit-equal to =1's; a token alone and smaller calls bit-equal to the
//! 4096-row call, including an explicit comparison with two 2048-row batches; then BENCH lines for both modes per phase,
//! gate/up, down) with TFLOP/s and modelled DRAM GB/s, on skewed and uniform pieces and one rows-per-expert size at a time.
const std = @import("std");
const cuda = @import("cuda");
const moe = @import("moe.zig");
const quant_ref = @import("quant_ref_tests.zig");
const exl3e = cuda.exl3_experts;
const Rig = quant_ref.RigT;

const E = 256;
const D = 6144;
const I = 1024; // this rank's half of moe_intermediate 2048
const SLOTS = 8;
const ROWS = 4096; // compare one batch with two 2048-row batches
const PAIRS = ROWS * SLOTS;
const lone_row = 777; // the only row routed to expert E - 7
const MoeLayers = 75;

test "GLM-5.3 wide prompt experts declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

/// Half-bits of expert e's matrix m (0 gate, 1 up, 2 down): mostly K2, a quarter K3, one in sixteen K1. Gate, up and
/// down of one expert differ, so a matrix or width mix-up changes the bits.
fn halvesOf(e: usize, m: usize) i32 {
    const h = (e * 7 + m * 3) % 16;
    return if (h == 0) 2 else if (h < 5) 6 else 4;
}

/// Bytes of one 16x16 trellis tile: 256 values of halves / 2 bits.
fn tileBytes(halves: i32) usize {
    return @as(usize, @intCast(halves)) * 16;
}

const Weights = struct { ptrs: [3]u64, k2: [3]u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64 };

/// Every expert's own random trellis (no sharing, so a wrong expert index changes the bits) and distinct scales.
fn buildWeights(r: *Rig) !Weights {
    var w: Weights = undefined;
    for (0..3) |m| {
        var offs: [E]u64 = undefined;
        var k2s: [E]i32 = undefined;
        var total: usize = 0;
        for (0..E) |e| {
            k2s[e] = halvesOf(e, m);
            offs[e] = total;
            total += (D / 16) * (I / 16) * tileBytes(k2s[e]);
        }
        const host = try r.gpu.gpa.alloc(u8, total);
        defer r.gpu.gpa.free(host);
        r.rng.bytes(host);
        const base = try r.dev(host);
        for (&offs) |*o| o.* += base;
        w.ptrs[m] = try r.dev(std.mem.sliceAsBytes(offs[0..]));
        w.k2[m] = try r.dev(std.mem.sliceAsBytes(k2s[0..]));
    }
    w.suh_g = try scaleRows(r, E * D);
    w.suh_u = try scaleRows(r, E * D);
    w.svh_g = try scaleRows(r, E * I);
    w.svh_u = try scaleRows(r, E * I);
    w.suh_d = try scaleRows(r, E * I);
    w.svh_d = try scaleRows(r, E * D);
    return w;
}

/// fp16 scales in [-0.125, 0.125): with half-range x the SwiGLU inputs stay moderate.
fn scaleRows(r: *Rig, n: usize) !u64 {
    const out = try r.a.alloc(u16, n);
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 0.25 - 0.125)));
    return r.dev(std.mem.sliceAsBytes(out));
}

fn layerOf(w: Weights, cb: exl3e.Codebook, shared_input: bool, limit: f32) exl3e.Layer {
    return layerOfMode(w, cb, shared_input, limit, 0);
}

fn layerOfMode(w: Weights, cb: exl3e.Codebook, shared_input: bool, limit: f32, act_mode: i32) exl3e.Layer {
    return .{ .gate = w.ptrs[0], .up = w.ptrs[1], .down = w.ptrs[2], .gate_k2 = w.k2[0], .up_k2 = w.k2[1], .down_k2 = w.k2[2], .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = E, .dims = D, .width = I, .codebook = cb, .widths_ok = true, .shared_input = shared_input, .limit = limit, .act_mode = act_mode };
}

/// 8 distinct experts a row, drawn by weight: 4 hot experts (hundreds of rows each), 8 warm (about 200), the rest
/// 1-3 (about 26-78 rows), the last 7 never, then expert E - 7 forced into exactly one row (lone_row).
fn skewedPicks(r: *Rig) ![]i32 {
    var weight: [E]u32 = undefined;
    for (&weight, 0..) |*v, e| v.* = if (e < 4) 24 else if (e < 12) 8 else if (e >= E - 7) 0 else @as(u32, @intCast(1 + e % 3));
    var cum: [E + 1]u32 = undefined;
    cum[0] = 0;
    for (0..E) |e| cum[e + 1] = cum[e] + weight[e];
    const out = try r.a.alloc(i32, PAIRS);
    for (0..ROWS) |row| {
        var k: usize = 0;
        while (k < SLOTS) {
            const u = r.rng.uintLessThan(u32, cum[E]);
            var e: usize = 0;
            while (cum[e + 1] <= u) e += 1;
            var dup = false;
            for (out[row * SLOTS ..][0..k]) |prev| dup = dup or prev == @as(i32, @intCast(e));
            if (!dup) {
                out[row * SLOTS + k] = @intCast(e);
                k += 1;
            }
        }
    }
    out[lone_row * SLOTS + 5] = E - 7;
    return out;
}

/// 8 distinct uniform experts a row (about 64 rows an expert).
fn uniformPicks(r: *Rig) ![]i32 {
    const out = try r.a.alloc(i32, PAIRS);
    for (0..ROWS) |row| {
        var idx: [E]u32 = undefined;
        for (&idx, 0..) |*v, n| v.* = @intCast(n);
        var left: usize = E;
        for (0..SLOTS) |k| {
            const j = r.rng.uintLessThan(usize, left);
            out[row * SLOTS + k] = @intCast(idx[j]);
            idx[j] = idx[left - 1];
            left -= 1;
        }
    }
    return out;
}

const Spread = struct { zero: usize = 0, one: usize = 0, mid: usize = 0, many: usize = 0, most: usize = 0, items64: usize = 0, items128: usize = 0 };

fn spreadOf(pick: []const i32) Spread {
    var counts: [E]usize = @splat(0);
    for (pick) |e| counts[@intCast(e)] += 1;
    var s: Spread = .{};
    for (counts) |c| {
        if (c == 0) s.zero += 1;
        if (c == 1) s.one += 1;
        if (c > 64 and c <= 128) s.mid += 1;
        if (c > 128) s.many += 1;
        s.most = @max(s.most, c);
        s.items64 += (c + 63) / 64;
        s.items128 += (c + 127) / 128;
    }
    return s;
}

fn planOf(r: *Rig, tile: usize) !cuda.grouped.Plan {
    return .{ .members = try r.zeros(PAIRS * 4), .items = try r.zeros(cuda.grouped.maxItems(PAIRS, E, tile) * 12), .counts = try r.zeros(16), .rank = try r.zeros(PAIRS * 4), .hist = try r.zeros((PAIRS + 1023) / 1024 * E * 4) };
}

const Mode = enum { tiles, wide, pmx };

fn slot(mode: Mode) usize {
    return switch (mode) {
        .tiles => 0,
        .wide => 1,
        .pmx => 2,
    };
}

/// One layer's inputs and a scratch, plan and combined output per mode (allocated once, reused by every call).
const Lab = struct {
    r: *Rig,
    exp: *const exl3e.Experts,
    pmx: *const exl3e.pmx.Pmx,
    x: u64, // bf16 [ROWS][D]
    wts: u64, // fp32 [ROWS][SLOTS]
    sb: [3][2]u64,
    sb_len: [2]usize,
    plans: [3]cuda.grouped.Plan,
    out: [3]u64, // fp32 [ROWS][D]
    ids: u64,
    glue53: moe.Kernels,

    /// One prompt call over rows [first, first + rows) of the piece; y lands at pair 0 of the mode's scratch.
    fn call(lab: *const Lab, mode: Mode, layer: exl3e.Layer, pick: u64, first: usize, rows: usize) !exl3e.prompt.Scratch {
        const m = slot(mode);
        const sc = try exl3e.prompt.scratch(lab.sb[m][0], layer, ROWS, SLOTS, false);
        const xs = lab.x + first * D * 2;
        const ps = pick + first * SLOTS * 4;
        switch (mode) {
            .tiles => try lab.exp.prompt.runTile(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m], exl3e.prompt.tile),
            .wide => try lab.exp.prompt.runWide(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m]),
            .pmx => if (exl3e.pmx.eligible(layer))
                try lab.exp.prompt.runPmx(lab.pmx, lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m])
            else
                try lab.exp.prompt.runWide(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m]),
        }
        return sc;
    }

    fn callOverlay(lab: *const Lab, mode: Mode, layer: exl3e.Layer, pick: u64, first: usize, rows: usize) !exl3e.prompt.Scratch {
        const m = slot(mode);
        const sc = try exl3e.prompt.scratch(lab.sb[m][1], layer, ROWS, SLOTS, true);
        const off = try exl3e.prompt.scratch(lab.sb[m][0], layer, ROWS, SLOTS, false);
        if (sc.xg - lab.sb[m][1] != off.xg - lab.sb[m][0] or sc.xu - lab.sb[m][1] != off.xu - lab.sb[m][0] or sc.xd - lab.sb[m][1] != off.xd - lab.sb[m][0] or sc.y - lab.sb[m][1] != off.y - lab.sb[m][0] or sc.uu - lab.sb[m][1] != off.uu - lab.sb[m][0] or sc.ids - lab.sb[m][1] != off.ids - lab.sb[m][0]) return error.PromptOverlayLayoutMismatch;
        const xs = lab.x + first * D * 2;
        const ps = pick + first * SLOTS * 4;
        switch (mode) {
            .tiles => try lab.exp.prompt.runTile(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m], exl3e.prompt.tile),
            .wide => try lab.exp.prompt.runWide(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m]),
            .pmx => if (exl3e.pmx.eligible(layer))
                try lab.exp.prompt.runPmx(lab.pmx, lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m])
            else
                try lab.exp.prompt.runWide(lab.exp, lab.exp.router, lab.r.s, layer, xs, D, ps, rows, sc, lab.plans[m]),
        }
        return sc;
    }

    fn combine(lab: *const Lab, mode: Mode, layer: exl3e.Layer, y: u64, first: usize, rows: usize) !void {
        try lab.exp.combinePrompt(lab.r.s, layer, y, lab.wts + first * SLOTS * 4, rows, SLOTS, lab.out[slot(mode)] + first * D * 4, lab.ids);
    }

    /// 0xff canaries in both scratches: an output a call does not write cannot match.
    fn fresh(lab: *const Lab) !void {
        for (lab.sb) |modes| {
            for (modes, 0..) |p, layout| {
                const view: cuda.DeviceBuffer = .{ .d = lab.r.gpu.d, .ptr = p, .len = lab.sb_len[layout] };
                try view.fill8(0xff, lab.r.s.handle);
            }
        }
    }

    fn fetch(lab: *const Lab, ptr: u64, len: usize) ![]u8 {
        const out = try lab.r.gpu.gpa.alloc(u8, len);
        errdefer lab.r.gpu.gpa.free(out);
        try lab.r.s.synchronize();
        const view: cuda.DeviceBuffer = .{ .d = lab.r.gpu.d, .ptr = ptr, .len = len };
        try view.download(0, out);
        return out;
    }
};

const Diff = struct { max_abs: f64 = 0, max_rel: f64 = 0, differ: usize = 0, nonfinite: usize = 0, first: ?usize = null };

/// fp32 rows of `width`: largest absolute difference, the same relative to its row's largest |want|, differing count.
fn diffOf(got: []const u8, want: []const u8, width: usize) Diff {
    var d: Diff = .{};
    const g = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
    const w = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, want));
    var row: usize = 0;
    while (row * width < w.len) : (row += 1) {
        var scale: f64 = 0;
        for (w[row * width ..][0..width]) |v| scale = @max(scale, @abs(@as(f64, v)));
        for (g[row * width ..][0..width], w[row * width ..][0..width], 0..) |gv, wv, col| {
            const a: f64 = gv;
            const b: f64 = wv;
            if (!std.math.isFinite(a)) d.nonfinite += 1;
            if (@as(u32, @bitCast(gv)) == @as(u32, @bitCast(wv))) continue;
            d.differ += 1;
            if (d.first == null) d.first = row * width + col;
            const diff = @abs(a - b);
            d.max_abs = @max(d.max_abs, diff);
            if (scale > 0) d.max_rel = @max(d.max_rel, diff / scale);
        }
    }
    return d;
}

const Case = struct { name: []const u8, cb: exl3e.Codebook, shared_input: bool, limit: f32, act_mode: i32 = 0, skewed: bool, invariance: bool };

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.exl3_experts.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    if (exp.prompt.wide == null) return error.Exl3PromptWideMissing;
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x5302_0128);
    var rig: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const r = &rig;
    var glue53_module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer glue53_module.unload();
    const glue53 = try moe.Kernels.load(glue53_module);
    const w = try buildWeights(r);
    const x = try r.bfs(ROWS * D, 0.5);
    const wts = try r.a.alloc(f32, PAIRS);
    for (wts) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    const skew = try skewedPicks(r);
    const uni = try uniformPicks(r);
    const dskew = try r.dev(std.mem.sliceAsBytes(skew));
    const duni = try r.dev(std.mem.sliceAsBytes(uni));
    // The non-shared layout is the larger; a shared-input layer's scratch fits inside it.
    const bench_shape = layerOf(w, .mcg_e4m3, false, 0);
    const sb_len = @max(try exl3e.prompt.bytes(bench_shape, ROWS, SLOTS, false), try exl3e.pmx.bytes(bench_shape, ROWS, SLOTS));
    const overlay_len = try exl3e.prompt.bytes(bench_shape, ROWS, SLOTS, true);
    if (overlay_len != try exl3e.prompt.bytes(bench_shape, ROWS, SLOTS, false)) return error.PromptOverlayLayoutMismatch;
    const lab: Lab = .{
        .r = r,
        .exp = &exp,
        .pmx = &pmx,
        .x = try r.dev(std.mem.sliceAsBytes(x)),
        .wts = try r.dev(std.mem.sliceAsBytes(wts)),
        .sb = .{ .{ try r.zeros(sb_len), try r.zeros(overlay_len) }, .{ try r.zeros(sb_len), try r.zeros(overlay_len) }, .{ try r.zeros(sb_len), try r.zeros(overlay_len) } },
        .sb_len = .{ sb_len, overlay_len },
        .plans = .{ try planOf(r, exl3e.prompt.tile), try planOf(r, exl3e.prompt.wide_tile), try planOf(r, exl3e.prompt.wide_tile) },
        .out = .{ try r.zeros(ROWS * D * 4), try r.zeros(ROWS * D * 4), try r.zeros(ROWS * D * 4) },
        .ids = try r.zeros(PAIRS * 4),
        .glue53 = glue53,
    };
    const ss = spreadOf(skew);
    const us = spreadOf(uni);
    std.debug.print("INFO glm53-prompt-wide skewed routing: {d} experts with 0 rows, {d} with 1, {d} with 65-128, {d} above 128 (most {d}); items {d} at tile 64, {d} at tile 128\n", .{ ss.zero, ss.one, ss.mid, ss.many, ss.most, ss.items64, ss.items128 });
    std.debug.print("INFO glm53-prompt-wide uniform routing: {d} with 65-128 rows, {d} above 128 (most {d}); items {d} at tile 64, {d} at tile 128\n", .{ us.mid, us.many, us.most, us.items64, us.items128 });
    if (ss.zero == 0 or ss.one == 0 or ss.mid == 0 or ss.many == 0) return error.RoutingNotSpread;

    const cases = [_]Case{
        .{ .name = "mcg_e4m3 K1/K2/K3 skewed", .cb = .mcg_e4m3, .shared_input = false, .limit = 0, .skewed = true, .invariance = true },
        .{ .name = "mcg_e4m3 act_mode=1 skewed", .cb = .mcg_e4m3, .shared_input = false, .limit = 0, .act_mode = 1, .skewed = true, .invariance = false },
        .{ .name = "mcg K1/K2/K3 skewed, clamp 10", .cb = .mcg, .shared_input = false, .limit = 10, .skewed = true, .invariance = true },
        .{ .name = "mcg act_mode=1 skewed", .cb = .mcg, .shared_input = false, .limit = 0, .act_mode = 1, .skewed = true, .invariance = false },
        .{ .name = "mcg_e4m3 K1/K2/K3 uniform", .cb = .mcg_e4m3, .shared_input = false, .limit = 0, .skewed = false, .invariance = false },
        .{ .name = "mcg_e4m3 shared input skewed", .cb = .mcg_e4m3, .shared_input = true, .limit = 0, .skewed = true, .invariance = false },
    };
    for (cases) |c| try checkCase(&lab, w, c, if (c.skewed) dskew else duni);
    var ev: [4]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    const bench_layer = layerOf(w, .mcg_e4m3, false, 0);
    try bench(&lab, &ev, bench_layer, "skewed", skew, dskew);
    try bench(&lab, &ev, bench_layer, "uniform", uni, duni);
    try sizeBench(&lab, &ev, bench_layer);
}

fn checkCase(lab: *const Lab, w: Weights, c: Case, pick: u64) !void {
    const gpa = lab.r.gpu.gpa;
    const layer = layerOfMode(w, c.cb, c.shared_input, c.limit, c.act_mode);
    if (!exl3e.prompt.eligible(layer)) return error.PromptNotEligible;
    const y_bytes = PAIRS * D * 4;
    try lab.fresh();
    const tiles = try lab.call(.tiles, layer, pick, 0, ROWS);
    const wide = try lab.call(.wide, layer, pick, 0, ROWS);
    const pmx_sc = try lab.call(.pmx, layer, pick, 0, ROWS);
    const overlay_tiles = try lab.callOverlay(.tiles, layer, pick, 0, ROWS);
    const overlay_wide = try lab.callOverlay(.wide, layer, pick, 0, ROWS);
    const want = try lab.fetch(tiles.y, y_bytes);
    defer gpa.free(want);
    const got = try lab.fetch(wide.y, y_bytes);
    defer gpa.free(got);
    const pmx_got = try lab.fetch(pmx_sc.y, y_bytes);
    defer gpa.free(pmx_got);
    const overlay_tile_y = try lab.fetch(overlay_tiles.y, y_bytes);
    defer gpa.free(overlay_tile_y);
    const overlay_wide_y = try lab.fetch(overlay_wide.y, y_bytes);
    defer gpa.free(overlay_wide_y);
    if (!std.mem.eql(u8, want, overlay_tile_y) or !std.mem.eql(u8, got, overlay_wide_y)) return error.PromptOverlayMismatch;
    const yd = diffOf(got, want, D);
    std.debug.print("RESULT glm53-prompt-wide {s}: per-slot y =2 vs =1 max_abs {e:.3} max_rel {e:.3}, {d} of {d} floats differ, {d} non-finite\n", .{ c.name, yd.max_abs, yd.max_rel, yd.differ, PAIRS * D, yd.nonfinite });
    if (yd.differ != 0 or yd.nonfinite != 0) {
        if (yd.first) |at| {
            const g = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
            const v = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, want));
            std.debug.print("FAIL glm53-prompt-wide {s}: first difference at pair {d} col {d}: =2 {e} =1 {e}\n", .{ c.name, at / D, at % D, g[at], v[at] });
        }
        return error.PromptWideBits;
    }
    const pd = diffOf(pmx_got, want, D);
    std.debug.print("RESULT glm53-prompt-wide {s}: per-slot y =3 vs =1 max_abs {e:.3} max_rel {e:.3}, {d} of {d} floats differ, {d} non-finite\n", .{ c.name, pd.max_abs, pd.max_rel, pd.differ, PAIRS * D, pd.nonfinite });
    if (pd.differ != 0 or pd.nonfinite != 0) return error.PromptPmxBits;
    // The weighted combine (one kernel, fed equal y) must agree too; it is what the forward keeps.
    try lab.combine(.tiles, layer, tiles.y, 0, ROWS);
    try lab.combine(.wide, layer, wide.y, 0, ROWS);
    try lab.combine(.pmx, layer, pmx_sc.y, 0, ROWS);
    const out_want = try lab.fetch(lab.out[0], ROWS * D * 4);
    defer gpa.free(out_want);
    const out_got = try lab.fetch(lab.out[1], ROWS * D * 4);
    defer gpa.free(out_got);
    const pmx_out = try lab.fetch(lab.out[2], ROWS * D * 4);
    defer gpa.free(pmx_out);
    try lab.combine(.tiles, layer, overlay_tiles.y, 0, ROWS);
    try lab.combine(.wide, layer, overlay_wide.y, 0, ROWS);
    const overlay_out_tiles = try lab.fetch(lab.out[0], ROWS * D * 4);
    defer gpa.free(overlay_out_tiles);
    const overlay_out_wide = try lab.fetch(lab.out[1], ROWS * D * 4);
    defer gpa.free(overlay_out_wide);
    if (!std.mem.eql(u8, out_want, overlay_out_tiles) or !std.mem.eql(u8, out_got, overlay_out_wide)) return error.PromptOverlayCombineMismatch;
    const od = diffOf(out_got, out_want, D);
    if (od.differ != 0 or od.nonfinite != 0) {
        std.debug.print("FAIL glm53-prompt-wide {s}: combined out differs ({d} floats, max_abs {e:.3}, {d} non-finite)\n", .{ c.name, od.differ, od.max_abs, od.nonfinite });
        return error.PromptWideCombine;
    }
    const pod = diffOf(pmx_out, out_want, D);
    if (pod.differ != 0 or pod.nonfinite != 0) return error.PromptPmxCombine;
    if (std.mem.eql(u8, c.name, "mcg_e4m3 K1/K2/K3 skewed")) {
        try moePostParity(lab, .tiles, tiles.y);
        try moePostParity(lab, .wide, wide.y);
        try moePostParity(lab, .pmx, pmx_sc.y);
    }
    for ([_]usize{ 1, 177, 512 }) |short_rows| {
        try lab.fresh();
        const short_tiles = try lab.call(.tiles, layer, pick, 0, short_rows);
        const short_wide = try lab.call(.wide, layer, pick, 0, short_rows);
        const short_y_bytes = short_rows * SLOTS * D * 4;
        const short_tile_y = try lab.fetch(short_tiles.y, short_y_bytes);
        defer gpa.free(short_tile_y);
        const short_wide_y = try lab.fetch(short_wide.y, short_y_bytes);
        defer gpa.free(short_wide_y);
        if (!std.mem.eql(u8, short_tile_y, short_wide_y)) return error.PromptWideShortRowsY;
        try lab.combine(.tiles, layer, short_tiles.y, 0, short_rows);
        try lab.combine(.wide, layer, short_wide.y, 0, short_rows);
        const short_out_bytes = short_rows * D * 4;
        const short_tile_out = try lab.fetch(lab.out[0], short_out_bytes);
        defer gpa.free(short_tile_out);
        const short_wide_out = try lab.fetch(lab.out[1], short_out_bytes);
        defer gpa.free(short_wide_out);
        if (!std.mem.eql(u8, short_tile_out, short_wide_out)) return error.PromptWideShortRowsCombine;
    }
    std.debug.print("PASS glm53-prompt-wide {s}: =2 and =3 y and combined out bit-equal to TF_GLM53_PROMPT_EXPERTS=1 (4096 rows)\n", .{c.name});
    // The 4096-row batch must be bit-identical to executing the same rows in two 2048-row batches.
    const half_y_bytes = 2048 * SLOTS * D * 4;
    const half_out_bytes = 2048 * D * 4;
    for (0..2) |half| {
        const first = half * 2048;
        try lab.fresh();
        const split_sc = try lab.call(.wide, layer, pick, first, 2048);
        const split_y = try lab.fetch(split_sc.y, half_y_bytes);
        defer gpa.free(split_y);
        if (!std.mem.eql(u8, split_y, got[first * SLOTS * D * 4 ..][0..half_y_bytes])) return error.PromptWide4096YMismatch;
        try lab.combine(.wide, layer, split_sc.y, first, 2048);
        const split_out = try lab.fetch(lab.out[1] + first * D * 4, half_out_bytes);
        defer gpa.free(split_out);
        if (!std.mem.eql(u8, split_out, out_got[first * D * 4 ..][0..half_out_bytes])) return error.PromptWide4096CombineMismatch;
    }
    std.debug.print("PASS glm53-prompt-wide {s}: 4096-row y and combined output bit-equal to two 2048-row batches\n", .{c.name});
    if (!c.invariance) return;

    // Row invariance: a token alone (rows = 1) and row ranges as smaller calls give the 4096-row call's bits.
    const row_bytes = SLOTS * D * 4;
    for ([_]usize{ 0, lone_row, ROWS - 1 }) |t| {
        try lab.fresh();
        const alone = try lab.call(.wide, layer, pick, t, 1);
        const one = try lab.fetch(alone.y, row_bytes);
        defer gpa.free(one);
        if (!std.mem.eql(u8, one, got[t * row_bytes ..][0..row_bytes])) {
            const rd = diffOf(one, got[t * row_bytes ..][0..row_bytes], D);
            std.debug.print("FAIL glm53-prompt-wide {s}: row {d} alone differs from row {d} of 4096 ({d} floats, max_abs {e:.3})\n", .{ c.name, t, t, rd.differ, rd.max_abs });
            return error.PromptWideRowInvariance;
        }
        try lab.fresh();
        const pmx_alone = try lab.call(.pmx, layer, pick, t, 1);
        const pmx_one = try lab.fetch(pmx_alone.y, row_bytes);
        defer gpa.free(pmx_one);
        if (!std.mem.eql(u8, pmx_one, pmx_got[t * row_bytes ..][0..row_bytes])) return error.PromptPmxRowInvariance;
    }
    const split = 1000;
    for ([_][2]usize{ .{ 0, split }, .{ split, ROWS - split } }) |part| {
        try lab.fresh();
        const sc = try lab.call(.wide, layer, pick, part[0], part[1]);
        const piece = try lab.fetch(sc.y, part[1] * row_bytes);
        defer gpa.free(piece);
        if (!std.mem.eql(u8, piece, got[part[0] * row_bytes ..][0 .. part[1] * row_bytes])) {
            std.debug.print("FAIL glm53-prompt-wide {s}: rows {d}..{d} as their own call differ from the 4096-row call\n", .{ c.name, part[0], part[0] + part[1] });
            return error.PromptWideRowInvariance;
        }
        try lab.fresh();
        const pmx_piece_sc = try lab.call(.pmx, layer, pick, part[0], part[1]);
        const pmx_piece = try lab.fetch(pmx_piece_sc.y, part[1] * row_bytes);
        defer gpa.free(pmx_piece);
        if (!std.mem.eql(u8, pmx_piece, pmx_got[part[0] * row_bytes ..][0 .. part[1] * row_bytes])) return error.PromptPmxRowInvariance;
    }
    std.debug.print("PASS glm53-prompt-wide {s}: rows 0, {d}, {d} alone and the piece as {d} + {d} rows bit-equal to the 4096-row call\n", .{ c.name, lone_row, ROWS - 1, split, ROWS - split });
}

fn moePostParity(lab: *const Lab, mode: Mode, y: u64) !void {
    const r = lab.r;
    const rank: usize = 16;
    const shared = try r.a.alloc(f32, D);
    const b = try r.a.alloc(f32, rank * D);
    const mid = try r.a.alloc(f32, rank);
    for (shared, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 128;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 2048;
    for (mid, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16;
    const ds = try r.dev(std.mem.sliceAsBytes(shared));
    const db = try r.dev(std.mem.sliceAsBytes(b));
    const dm = try r.dev(std.mem.sliceAsBytes(mid));
    const old = try r.zeros(D * 2);
    const fused = try r.zeros(D * 2);
    const lr: moe.LowRankWeights = .{ .a = 0, .b = db, .mid = dm, .rank_local = rank };
    var reference_kernels = lab.glue53;
    reference_kernels.decode_fast = false;
    try moe.blockPost(reference_kernels, r.s, lab.out[slot(mode)], ds, lr, 1, old);
    try moe.postPrompt(lab.glue53, r.s, y, lab.wts, ds, lr, 1, fused);
    const old_bytes = try lab.fetch(old, D * 2);
    defer r.gpu.gpa.free(old_bytes);
    const fused_bytes = try lab.fetch(fused, D * 2);
    defer r.gpu.gpa.free(fused_bytes);
    if (!std.mem.eql(u8, old_bytes, fused_bytes)) return error.MoePostPromptParity;
    std.debug.print("PASS glm53-prompt-wide: fused post byte-equal for {s} prompt y\n", .{@tagName(mode)});
}

const Phase = enum { prep, gu, down };

fn tileOf(mode: Mode) usize {
    return switch (mode) {
        .tiles => exl3e.prompt.tile,
        .wide => exl3e.prompt.wide_tile,
        .pmx => exl3e.prompt.wide_tile,
    };
}

/// One phase of one mode over rows [0, rows): prep (plan + rotation), gate/up, or down; each reads only what the
/// phases before it wrote, so a phase may repeat.
fn phaseCall(lab: *const Lab, mode: Mode, phase: Phase, layer: exl3e.Layer, pick: u64, rows: usize) !void {
    const m = slot(mode);
    const sc = try exl3e.prompt.scratch(lab.sb[m][0], layer, ROWS, SLOTS, false);
    const k = lab.exp.prompt;
    const s = lab.r.s;
    const plan = lab.plans[m];
    switch (phase) {
        .prep => try k.prepare(lab.exp, lab.exp.router, s, layer, lab.x, D, pick, rows, sc, plan, tileOf(mode)),
        .gu => switch (mode) {
            .tiles => try k.gateUpTiles(s, layer, rows, sc, plan, exl3e.prompt.tile),
            .wide => try k.gateUpWide(s, layer, rows, sc, plan),
            .pmx => if (exl3e.pmx.eligible(layer)) try pmxGateUp(lab, layer, rows, sc, plan) else try k.gateUpWide(s, layer, rows, sc, plan),
        },
        .down => switch (mode) {
            .tiles => try k.downTiles(s, layer, rows, sc, plan, exl3e.prompt.tile),
            .wide => try k.downWide(s, layer, rows, sc, plan),
            .pmx => try k.downWide(s, layer, rows, sc, plan),
        },
    }
}

fn pmxGateUp(lab: *const Lab, layer: exl3e.Layer, rows: usize, sc: exl3e.prompt.Scratch, plan: cuda.grouped.Plan) !void {
    const k = lab.exp.prompt;
    const items = cuda.grouped.maxItems(rows * sc.slots, layer.experts, exl3e.prompt.wide_tile);
    const c: usize = if (layer.codebook == .mcg_e4m3) 1 else 0;
    const small = k.wide.?.gu_small[c];
    var small_args: cuda.launch.Args = .{};
    inline for (.{ sc.xg, sc.xu, layer.gate, layer.up, plan.items, plan.counts, plan.members, layer.svh_g, layer.svh_u, layer.suh_d, sc.xd, @as(i32, @intCast(layer.dims)), @as(i32, @intCast(layer.width)), @as(i32, @intCast(if (layer.shared_input) sc.slots else 1)), layer.limit, layer.act_mode, layer.gate_k2, layer.up_k2 }) |v| small_args.add(v);
    try cuda.launch.launch(small, .{ .grid = .{ .x = @intCast(layer.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, lab.r.s, &small_args);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.PmxArgs{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = layer.gate, .tp1 = layer.up, .k2_0 = layer.gate_k2, .k2_1 = layer.up_k2, .items = plan.items, .counts = plan.counts, .members = plan.members, .sv0 = layer.svh_g, .sv1 = layer.svh_u, .sd = layer.suh_d, .xd = sc.xd, .y = sc.y, .K = @intCast(layer.dims), .N = @intCast(layer.width), .E = @intCast(layer.experts), .limit = layer.limit, .act_mode = layer.act_mode });
    try cuda.launch.launch(lab.pmx.gu[c], .{ .grid = .{ .x = @intCast(layer.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @intCast(exl3e.pmx.gu_smem) }, lab.r.s, &args);
}

const Times = struct { prep: f64, gu: f64, down: f64 }; // us a call

fn timePhases(lab: *const Lab, ev: *const [4]cuda.Event, mode: Mode, layer: exl3e.Layer, pick: u64, rows: usize) !Times {
    const s = lab.r.s;
    const reps = 5;
    for ([_]Phase{ .prep, .gu, .down }) |phase| try phaseCall(lab, mode, phase, layer, pick, rows); // warm
    try ev[0].record(s);
    for (0..reps) |_| try phaseCall(lab, mode, .prep, layer, pick, rows);
    try ev[1].record(s);
    for (0..reps) |_| try phaseCall(lab, mode, .gu, layer, pick, rows);
    try ev[2].record(s);
    for (0..reps) |_| try phaseCall(lab, mode, .down, layer, pick, rows);
    try ev[3].record(s);
    try s.synchronize();
    return .{
        .prep = @as(f64, try ev[0].elapsedMs(ev[1])) * 1000.0 / reps,
        .gu = @as(f64, try ev[1].elapsedMs(ev[2])) * 1000.0 / reps,
        .down = @as(f64, try ev[2].elapsedMs(ev[3])) * 1000.0 / reps,
    };
}

fn matrixBytes(halves: i32) f64 {
    return @floatFromInt((D / 16) * (I / 16) * tileBytes(halves));
}

/// Modelled DRAM bytes a phase moves: each touched expert's trellis once, activations once (the 8 column CTAs of an
/// item share their X through L2). =2 adds wup's fp32 uu, written and read, for the pairs of items over 64 members.
const Traffic = struct { prep: f64, gu: f64, down: f64, combine: f64, big_pairs: usize };

fn trafficOf(pick: []const i32, rows: usize, mode: Mode) Traffic {
    var counts: [E]usize = @splat(0);
    for (pick) |e| counts[@intCast(e)] += 1;
    var gu_w: f64 = 0;
    var down_w: f64 = 0;
    var big: usize = 0;
    for (counts, 0..) |c, e| {
        if (c == 0) continue;
        gu_w += matrixBytes(halvesOf(e, 0)) + matrixBytes(halvesOf(e, 1));
        down_w += matrixBytes(halvesOf(e, 2));
        const rem = c % exl3e.prompt.wide_tile;
        big += c - rem + (if (rem > exl3e.prompt.tile) rem else 0);
    }
    const p: f64 = @floatFromInt(pick.len);
    const n: f64 = @floatFromInt(rows);
    const uu: f64 = if (mode == .wide) 2.0 * @as(f64, @floatFromInt(big)) * I * 4 else 0;
    return .{
        .prep = n * D * 2 + 2 * p * D * 2,
        .gu = gu_w + 2 * p * D * 2 + p * I * 2 + uu,
        .down = down_w + p * I * 2 + p * D * 4,
        .combine = p * D * 4 + n * D * 4 + p * 8,
        .big_pairs = big,
    };
}

fn tflops(flop: f64, us: f64) f64 {
    return flop / us / 1e6;
}

fn gbps(bytes: f64, us: f64) f64 {
    return bytes / us / 1e3;
}

fn printMode(label: []const u8, mode: Mode, t: Times, tr: Traffic, pairs: usize) void {
    const p: f64 = @floatFromInt(pairs);
    const gu_flop = 2.0 * p * 2.0 * D * I;
    const down_flop = 2.0 * p * D * I;
    const total = t.prep + t.gu + t.down;
    std.debug.print("BENCH glm53-prompt-wide {s} ={d}: prep {d:.0} us ({d:.0} GB/s) | gate/up {d:.0} us ({d:.1} TFLOP/s, {d:.0} GB/s) | down {d:.0} us ({d:.1} TFLOP/s, {d:.0} GB/s) | total {d:.0} us ({d:.1} TFLOP/s, {d:.0} GB/s)\n", .{ label, slot(mode) + 1, t.prep, gbps(tr.prep, t.prep), t.gu, tflops(gu_flop, t.gu), gbps(tr.gu, t.gu), t.down, tflops(down_flop, t.down), gbps(tr.down, t.down), total, tflops(gu_flop + down_flop, total), gbps(tr.prep + tr.gu + tr.down, total) });
}

/// A 4096-row batch in both modes, phase by phase, plus the shared combine (timed once, it is the same kernel).
fn bench(lab: *const Lab, ev: *const [4]cuda.Event, layer: exl3e.Layer, name: []const u8, pick_host: []const i32, pick: u64) !void {
    const sp = spreadOf(pick_host);
    const t1 = try timePhases(lab, ev, .tiles, layer, pick, ROWS);
    const t2 = try timePhases(lab, ev, .wide, layer, pick, ROWS);
    const t3 = try timePhases(lab, ev, .pmx, layer, pick, ROWS);
    const tr1 = trafficOf(pick_host, ROWS, .tiles);
    const tr2 = trafficOf(pick_host, ROWS, .wide);
    const tr3 = trafficOf(pick_host, ROWS, .pmx);
    const s = lab.r.s;
    const y = (try exl3e.prompt.scratch(lab.sb[1][0], layer, ROWS, SLOTS, false)).y;
    try lab.combine(.wide, layer, y, 0, ROWS);
    try ev[0].record(s);
    for (0..5) |_| try lab.combine(.wide, layer, y, 0, ROWS);
    try ev[1].record(s);
    try s.synchronize();
    const combine_us = @as(f64, try ev[0].elapsedMs(ev[1])) * 1000.0 / 5.0;
    var label_buf: [96]u8 = undefined;
    const label = std.fmt.bufPrint(&label_buf, "{s} {d} rows", .{ name, ROWS }) catch name;
    std.debug.print("BENCH glm53-prompt-wide {s} (256 x 6144 x 1024 rank half, mcg_e4m3 K1/K2/K3, 8 slots): items {d} at tile 64, {d} at tile 128; {d} of {d} pairs in items over 64 members\n", .{ label, sp.items64, sp.items128, tr2.big_pairs, PAIRS });
    printMode(label, .tiles, t1, tr1, PAIRS);
    printMode(label, .wide, t2, tr2, PAIRS);
    printMode(label, .pmx, t3, tr3, PAIRS);
    const total1 = t1.prep + t1.gu + t1.down;
    const total2 = t2.prep + t2.gu + t2.down;
    const per_token = @as(f64, MoeLayers) / @as(f64, ROWS);
    std.debug.print("BENCH glm53-prompt-wide {s}: speedup total {d:.3}x (gate/up {d:.3}x, down {d:.3}x); combine {d:.0} us ({d:.0} GB/s); stage over {d} MoE layers incl. combine: =1 {d:.0} us/token/rank, =2 {d:.0} us/token/rank\n", .{ label, total1 / total2, t1.gu / t2.gu, t1.down / t2.down, combine_us, gbps(tr1.combine, combine_us), MoeLayers, (total1 + combine_us) * per_token, (total2 + combine_us) * per_token });
}

/// Rows an expert, one size at a time: `active` experts share `rows` rows evenly (row r slot k -> (8r + k) % active).
fn sizeBench(lab: *const Lab, ev: *const [4]cuda.Event, layer: exl3e.Layer) !void {
    const Size = struct { rows: usize, active: usize };
    const sizes = [_]Size{ .{ .rows = 512, .active = 256 }, .{ .rows = 1024, .active = 256 }, .{ .rows = 2048, .active = 256 }, .{ .rows = 2048, .active = 192 }, .{ .rows = 2048, .active = 128 }, .{ .rows = 2048, .active = 64 } };
    for (sizes) |sz| {
        const host = try lab.r.a.alloc(i32, sz.rows * SLOTS);
        for (0..sz.rows) |row| {
            for (0..SLOTS) |k| host[row * SLOTS + k] = @intCast((row * SLOTS + k) % sz.active);
        }
        const pick = try lab.r.dev(std.mem.sliceAsBytes(host));
        const t1 = try timePhases(lab, ev, .tiles, layer, pick, sz.rows);
        const t2 = try timePhases(lab, ev, .wide, layer, pick, sz.rows);
        const tr1 = trafficOf(host, sz.rows, .tiles);
        const tr2 = trafficOf(host, sz.rows, .wide);
        const per = sz.rows * SLOTS / sz.active;
        var label_buf: [96]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "size {d} rows/expert ({d} rows, {d} experts)", .{ per, sz.rows, sz.active }) catch "size";
        printMode(label, .tiles, t1, tr1, sz.rows * SLOTS);
        printMode(label, .wide, t2, tr2, sz.rows * SLOTS);
        std.debug.print("BENCH glm53-prompt-wide {s}: speedup gate/up {d:.3}x, down {d:.3}x, gate/up + down {d:.3}x\n", .{ label, t1.gu / t2.gu, t1.down / t2.down, (t1.gu + t1.down) / (t2.gu + t2.down) });
    }
}
