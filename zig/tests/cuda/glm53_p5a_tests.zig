// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 TF_GLM53_PROMPT_EXPERTS=10 routed prompt expert parity and benchmark.
const std = @import("std");
const cuda = @import("cuda");
const quant_ref = @import("quant_ref_tests.zig");
const exl3e = cuda.exl3_experts;
const Rig = quant_ref.RigT;

const E = 256;
const D = 6144;
const I = 1024;
const SLOTS = 8;
const ROWS_MAX = 4096;
const PAIRS_MAX = ROWS_MAX * SLOTS;
const MoeLayers = 75;
const sizes = [_]usize{ 1, 7, 8, 9, 16, 17, 32, 33, 63, 64, 65, 108, 129, 512 };

test "GLM-5.3 p5a GPU declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

const Weights = struct { ptrs: [3]u64, k2: [3]u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64 };
const Halves = *const fn (usize, usize) i32;

fn w46(e: usize, m: usize) i32 {
    const wide = if (m == 0) e % 2 == 0 else if (m == 1) (e / 2) % 2 == 1 else (e * 7 + m * 3) % 4 == 0;
    return if (wide) 6 else 4;
}
fn w246(e: usize, m: usize) i32 {
    const h = (e * 7 + m * 3) % 16;
    return if (h == 0) 2 else if (h < 5) 6 else 4;
}
fn wbench(e: usize, m: usize) i32 {
    return if ((e * 7 + m * 3) % 9 == 0) 6 else 4;
}
fn everyWant() [E]usize {
    var want: [E]usize = @splat(0);
    for (0..64) |expert| want[expert] = expert + 1;
    const extra = [_]usize{ 65, 96, 127, 128, 129, 192, 255, 256 };
    @memcpy(want[64..72], &extra);
    return want;
}
fn everyTotalWant() usize {
    var total: usize = 0;
    for (everyWant()) |count| total += count;
    return total;
}
fn tileBytes(k: i32) usize {
    return @as(usize, @intCast(k)) * 16;
}
fn scaleRows(r: *Rig, n: usize) !u64 {
    const out = try r.a.alloc(u16, n);
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 0.25 - 0.125)));
    return r.dev(std.mem.sliceAsBytes(out));
}
fn buildWeights(r: *Rig, halves: Halves) !Weights {
    var out: Weights = undefined;
    for (0..3) |m| {
        var offsets: [E]u64 = undefined;
        var widths: [E]i32 = undefined;
        var total: usize = 0;
        for (0..E) |e| {
            widths[e] = halves(e, m);
            offsets[e] = total;
            total += (D / 16) * (I / 16) * tileBytes(widths[e]);
        }
        const host = try r.gpu.gpa.alloc(u8, total);
        defer r.gpu.gpa.free(host);
        r.rng.bytes(host);
        const base = try r.dev(host);
        for (&offsets) |*v| v.* += base;
        out.ptrs[m] = try r.dev(std.mem.sliceAsBytes(offsets[0..]));
        out.k2[m] = try r.dev(std.mem.sliceAsBytes(widths[0..]));
    }
    out.suh_g = try scaleRows(r, E * D);
    out.suh_u = try scaleRows(r, E * D);
    out.svh_g = try scaleRows(r, E * I);
    out.svh_u = try scaleRows(r, E * I);
    out.suh_d = try scaleRows(r, E * I);
    out.svh_d = try scaleRows(r, E * D);
    return out;
}
fn layerOf(w: Weights, cb: exl3e.Codebook, limit: f32, act_mode: i32) exl3e.Layer {
    return .{ .gate = w.ptrs[0], .up = w.ptrs[1], .down = w.ptrs[2], .gate_k2 = w.k2[0], .up_k2 = w.k2[1], .down_k2 = w.k2[2], .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = E, .dims = D, .width = I, .codebook = cb, .widths_ok = true, .pmx_ok = true, .shared_input = false, .limit = limit, .act_mode = act_mode };
}
fn layerGeneric(w: Weights, cb: exl3e.Codebook, limit: f32, act_mode: i32) exl3e.Layer {
    var l = layerOf(w, cb, limit, act_mode);
    l.pmx_ok = false;
    return l;
}

fn planOf(r: *Rig) !cuda.grouped.Plan {
    return .{ .members = try r.zeros(PAIRS_MAX * 4), .items = try r.zeros(cuda.grouped.maxItems(PAIRS_MAX, E, exl3e.prompt.tile) * 12), .counts = try r.zeros(16), .rank = try r.zeros(PAIRS_MAX * 4), .hist = try r.zeros((PAIRS_MAX + 1023) / 1024 * E * 4) };
}

/// Size routing: each of experts 0..251 receives the requested count; zero-pair experts remain unused.
fn sizesPicks(r: *Rig) ![]i32 {
    var flat = std.ArrayList(i32).empty;
    for (0..E - 4) |e| for (0..sizes[e % sizes.len]) |_| try flat.append(r.a, @intCast(e));
    for (0..flat.items.len) |i| {
        const j = i + r.rng.uintLessThan(usize, flat.items.len - i);
        std.mem.swap(i32, &flat.items[i], &flat.items[j]);
    }
    const rows = flat.items.len / SLOTS;
    if (flat.items.len % SLOTS != 0) return error.BadSizesRouting;
    return flat.items[0 .. rows * SLOTS];
}
fn skewedPicks(r: *Rig, rows: usize) ![]i32 {
    var weight: [E]u32 = undefined;
    for (&weight, 0..) |*v, e| v.* = if (e < 4) 24 else if (e < 12) 8 else if (e >= E - 7) 0 else @as(u32, @intCast(1 + e % 3));
    var cum: [E + 1]u32 = undefined;
    cum[0] = 0;
    for (0..E) |e| cum[e + 1] = cum[e] + weight[e];
    const out = try r.a.alloc(i32, rows * SLOTS);
    for (0..rows) |row| {
        var k: usize = 0;
        while (k < SLOTS) {
            const u = r.rng.uintLessThan(u32, cum[E]);
            var e: usize = 0;
            while (cum[e + 1] <= u) e += 1;
            var dup = false;
            for (out[row * SLOTS ..][0..k]) |prev| dup = dup or prev == @as(i32, @intCast(e));
            if (!dup) { out[row * SLOTS + k] = @intCast(e); k += 1; }
        }
    }
    return out;
}
fn uniformPicks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * SLOTS);
    for (0..rows) |row| {
        var idx: [E]u32 = undefined;
        for (&idx, 0..) |*v, e| v.* = @intCast(e);
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
fn benchPicks(a: std.mem.Allocator, rows: usize) ![]i32 {
    var weight: [E]u64 = undefined;
    for (&weight, 0..) |*w, e| {
        const h: u32 = @truncate(@as(u64, e) *% 2654435761);
        const u = @as(f64, @floatFromInt(h)) / 4294967296.0;
        w.* = 50 + @as(u64, @intFromFloat(@floor(1000.0 * u * u)));
    }
    var cum: [E + 1]u64 = undefined;
    cum[0] = 0;
    for (0..E) |e| cum[e + 1] = cum[e] + weight[e];
    var prng = std.Random.DefaultPrng.init(0x5305_0000 + @as(u64, rows));
    const rng = prng.random();
    const out = try a.alloc(i32, rows * SLOTS);
    for (0..rows) |row| {
        var k: usize = 0;
        while (k < SLOTS) {
            const v = rng.uintLessThan(u64, cum[E]);
            var e: usize = 0;
            while (cum[e + 1] <= v) e += 1;
            var dup = false;
            for (out[row * SLOTS ..][0..k]) |prev| dup = dup or prev == @as(i32, @intCast(e));
            if (!dup) { out[row * SLOTS + k] = @intCast(e); k += 1; }
        }
    }
    return out;
}

const Lab = struct {
    r: *Rig,
    exp: *const exl3e.Experts,
    pmx: *const exl3e.pmx.Pmx,
    p5a: *const exl3e.p5a.P5a,
    x: u64,
    weights: u64,
    sb: [2]u64,
    sb_len: usize,
    plans: [2]cuda.grouped.Plan,
    out: [2]u64,
    ids: u64,

    fn sc(lab: *const Lab, mode: usize, l: exl3e.Layer) !exl3e.prompt.Scratch {
        return exl3e.prompt.scratch(lab.sb[mode], l, ROWS_MAX, SLOTS, false);
    }
    fn fresh(lab: *const Lab) !void {
        for (lab.sb) |p| {
            const buf: cuda.DeviceBuffer = .{ .d = lab.r.gpu.d, .ptr = p, .len = lab.sb_len };
            try buf.fill8(0xff, lab.r.s.handle);
        }
    }
    fn fetch(lab: *const Lab, ptr: u64, len: usize) ![]u8 {
        const out = try lab.r.gpu.gpa.alloc(u8, len);
        errdefer lab.r.gpu.gpa.free(out);
        try lab.r.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = lab.r.gpu.d, .ptr = ptr, .len = len };
        try buf.download(0, out);
        return out;
    }
    fn runPmx(lab: *const Lab, mode: usize, l: exl3e.Layer, picks: u64, rows: usize) !exl3e.prompt.Scratch {
        return lab.runPmxAt(mode, l, picks, 0, rows);
    }
    fn runPmxAt(lab: *const Lab, mode: usize, l: exl3e.Layer, picks: u64, first: usize, rows: usize) !exl3e.prompt.Scratch {
        const s = try lab.sc(mode, l);
        try lab.exp.prompt.runPmx(lab.pmx, lab.exp, lab.exp.router, lab.r.s, l, lab.x + first * D * 2, D, picks, rows, s, lab.plans[mode]);
        return s;
    }
    fn runP5a(lab: *const Lab, mode: usize, l: exl3e.Layer, picks: u64, first: usize, rows: usize) !exl3e.prompt.Scratch {
        const s = try lab.sc(mode, l);
        try lab.exp.prompt.runP5a(lab.pmx, lab.p5a, lab.exp, lab.exp.router, lab.r.s, l, lab.x + first * D * 2, D, picks, rows, s, lab.plans[mode]);
        return s;
    }
    fn prepare(lab: *const Lab, mode: usize, l: exl3e.Layer, picks: u64, rows: usize) !exl3e.prompt.Scratch {
        const s = try lab.sc(mode, l);
        try lab.exp.prompt.prepare(lab.exp, lab.exp.router, lab.r.s, l, lab.x, D, picks, rows, s, lab.plans[mode], exl3e.prompt.wide_tile);
        return s;
    }
    fn combine(lab: *const Lab, mode: usize, l: exl3e.Layer, y: u64, rows: usize) !void {
        try lab.exp.combinePrompt(lab.r.s, l, y, lab.weights, rows, SLOTS, lab.out[mode], lab.ids);
    }
};

fn resources(gpu: anytype, p: *const exl3e.p5a.P5a) !void {
    const fns = [_]cuda.Function{ p.gu[0], p.gu[1], p.down[0], p.down[1] };
    const names = [_][]const u8{ "tf_glm53_p5a_gu_mcg", "tf_glm53_p5a_gu_e4m3", "tf_glm53_p5a_down_mcg", "tf_glm53_p5a_down_e4m3" };
    for (fns, names) |f, name| {
        const local = try f.attribute(.local_size_bytes);
        const regs = try f.attribute(.num_regs);
        const active = try f.occupancy(exl3e.p5a.threads, 0);
        std.debug.print("INFO glm53-p5a {s}: regs {d}, local {d} B, {d} CTAs an SM\n", .{ name, regs, local, active });
        if (local != 0 or regs > 128 or active < 2) return error.P5aResources;
    }
    std.debug.print("PASS glm53-p5a resources: all four entries have no local memory and allow at least 2 CTAs/SM\n", .{});
    _ = gpu;
}

fn makeLab(r: *Rig, exp: *const exl3e.Experts, pmx: *const exl3e.pmx.Pmx, p5a: *const exl3e.p5a.P5a, w: Weights) !Lab {
    const x = try r.bfs(ROWS_MAX * D, 0.5);
    const mix = try r.a.alloc(f32, PAIRS_MAX);
    for (mix) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    const shape = layerOf(w, .mcg_e4m3, 0, 0);
    const bytes = @max(try exl3e.prompt.bytes(shape, ROWS_MAX, SLOTS, false), try exl3e.pmx.bytes(shape, ROWS_MAX, SLOTS));
    return .{ .r = r, .exp = exp, .pmx = pmx, .p5a = p5a, .x = try r.dev(std.mem.sliceAsBytes(x)), .weights = try r.dev(std.mem.sliceAsBytes(mix)), .sb = .{ try r.zeros(bytes), try r.zeros(bytes) }, .sb_len = bytes, .plans = .{ try planOf(r), try planOf(r) }, .out = .{ try r.zeros(ROWS_MAX * D * 4), try r.zeros(ROWS_MAX * D * 4) }, .ids = try r.zeros(PAIRS_MAX * 4) };
}

fn expectBytes(label: []const u8, got: []const u8, want: []const u8, width: usize, pair_width: usize) !void {
    if (std.mem.eql(u8, got, want)) return;
    var at: usize = 0;
    while (at < @min(got.len, want.len) and got[at] == want[at]) : (at += 1) {}
    const element_bytes: usize = if (width == D) 4 else 2;
    const index = at / element_bytes;
    const pair = index / width;
    const col = index % width;
    if (element_bytes == 4 and at + 4 <= @min(got.len, want.len)) {
        const g = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got));
        const w = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, want));
        std.debug.print("FAIL glm53-p5a {s}: first difference at pair {d} col {d}: got {e} expected {e}\n", .{ label, pair, col, g[index], w[index] });
    } else {
        const g = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got));
        const w = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, want));
        const half: usize = at / 2;
        std.debug.print("FAIL glm53-p5a {s}: first difference at pair {d} col {d}: got 0x{x:0>4} expected 0x{x:0>4}\n", .{ label, pair, col, g[half], w[half] });
    }
    _ = pair_width;
    return error.P5aBits;
}

fn checkFull(lab: *const Lab, w: Weights, cb: exl3e.Codebook, act: i32, limit: f32, label: []const u8, host_pick: []const i32) !void {
    const rows = host_pick.len / SLOTS;
    const pick = try lab.r.dev(std.mem.sliceAsBytes(host_pick));
    const l = layerOf(w, cb, limit, act);
    const xd_len = rows * SLOTS * I * 2;
    const y_len = rows * SLOTS * D * 4;
    try lab.fresh();
    const ref = try lab.runPmx(0, l, pick, rows);
    const new = try lab.runP5a(1, l, pick, 0, rows);
    const rx = try lab.fetch(ref.xd, xd_len);
    defer lab.r.gpu.gpa.free(rx);
    const nx = try lab.fetch(new.xd, xd_len);
    defer lab.r.gpu.gpa.free(nx);
    try expectBytes(label, nx, rx, I, rows * SLOTS);
    const ry = try lab.fetch(ref.y, y_len);
    defer lab.r.gpu.gpa.free(ry);
    const ny = try lab.fetch(new.y, y_len);
    defer lab.r.gpu.gpa.free(ny);
    try expectBytes(label, ny, ry, D, rows * SLOTS);
    try lab.combine(0, l, ref.y, rows);
    try lab.combine(1, l, new.y, rows);
    const ro = try lab.fetch(lab.out[0], rows * D * 4);
    defer lab.r.gpu.gpa.free(ro);
    const no = try lab.fetch(lab.out[1], rows * D * 4);
    defer lab.r.gpu.gpa.free(no);
    try expectBytes(label, no, ro, D, rows);
    std.debug.print("RESULT glm53-p5a {s}: y, xd and combined output compared over {d} rows\n", .{ label, rows });
    std.debug.print("PASS glm53-p5a {s}: =10 bit-equal to production =3\n", .{label});
}

fn compareSmall(lab: *const Lab, w: Weights, cb: exl3e.Codebook, label: []const u8, host_pick: []const i32) !void {
    const rows = host_pick.len / SLOTS;
    const pick = try lab.r.dev(std.mem.sliceAsBytes(host_pick));
    const l = layerGeneric(w, cb, 0, 0);
    const y_len = rows * SLOTS * D * 4;
    const xd_len = rows * SLOTS * I * 2;
    try lab.fresh();
    const ref = try lab.prepare(0, l, pick, rows);
    const items = cuda.grouped.maxItems(rows * SLOTS, E, exl3e.prompt.wide_tile);
    const c: usize = if (cb == .mcg_e4m3) 1 else 0;
    var gu_args = smallGuArgs(l, ref, lab.plans[0]);
    var down_args = smallDownArgs(l, ref, lab.plans[0]);
    try cuda.launch.launch(lab.exp.prompt.wide.?.gu_small[c], .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, lab.r.s, &gu_args);
    try cuda.launch.launch(lab.exp.prompt.wide.?.down_small[c], .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, lab.r.s, &down_args);
    const new = try lab.prepare(1, l, pick, rows);
    var p5a_gu_args = smallGuArgs(l, new, lab.plans[1]);
    var p5a_down_args = smallDownArgs(l, new, lab.plans[1]);
    try cuda.launch.launch(lab.p5a.gu[c], .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = exl3e.p5a.threads } }, lab.r.s, &p5a_gu_args);
    try cuda.launch.launch(lab.p5a.down[c], .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = exl3e.p5a.threads } }, lab.r.s, &p5a_down_args);
    const got_x = try lab.fetch(new.xd, xd_len);
    defer lab.r.gpu.gpa.free(got_x);
    const ref_x = try lab.fetch(ref.xd, xd_len);
    defer lab.r.gpu.gpa.free(ref_x);
    try expectBytes(label, got_x, ref_x, I, rows * SLOTS);
    const got_y = try lab.fetch(new.y, y_len);
    defer lab.r.gpu.gpa.free(got_y);
    const ref_y = try lab.fetch(ref.y, y_len);
    defer lab.r.gpu.gpa.free(ref_y);
    try expectBytes(label, got_y, ref_y, D, rows * SLOTS);
    std.debug.print("PASS glm53-p5a {s}: p5a small items bit-equal to =1 kernels\n", .{label});
}

fn smallGuArgs(l: exl3e.Layer, sc: exl3e.prompt.Scratch, p: cuda.grouped.Plan) cuda.launch.Args {
    var args: cuda.launch.Args = .{};
    inline for (.{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, @as(i32, D), @as(i32, I), @as(i32, 1), l.limit, l.act_mode, l.gate_k2, l.up_k2 }) |v| args.add(v);
    return args;
}
fn smallDownArgs(l: exl3e.Layer, sc: exl3e.prompt.Scratch, p: cuda.grouped.Plan) cuda.launch.Args {
    var args: cuda.launch.Args = .{};
    // tf_exl3_prompt_down_s takes ys before k2d; keep both trailing parameters in signature order.
    inline for (.{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, @as(i32, I), @as(i32, D), @as(i32, SLOTS), l.down_k2 }) |v| args.add(v);
    return args;
}

/// =3 (PMX for items above 64 members) against =1 (tile-64 kernels) on one routing: prints the differing xd halves and
/// y floats and returns whether both are bit-equal. It never returns early, so every case is reported.
fn pmxGap(lab: *const Lab, w: Weights, cb: exl3e.Codebook, act: i32, limit: f32, label: []const u8, host_pick: []const i32) !bool {
    const rows = host_pick.len / SLOTS;
    const pick = try lab.r.dev(std.mem.sliceAsBytes(host_pick));
    const l = layerOf(w, cb, limit, act);
    const xd_len = rows * SLOTS * I * 2;
    const y_len = rows * SLOTS * D * 4;
    try lab.fresh();
    const s1 = try lab.sc(0, l);
    try lab.exp.prompt.runTile(lab.exp, lab.exp.router, lab.r.s, l, lab.x, D, pick, rows, s1, lab.plans[0], exl3e.prompt.tile);
    const s3 = try lab.runPmx(1, l, pick, rows);
    const x1 = try lab.fetch(s1.xd, xd_len);
    defer lab.r.gpu.gpa.free(x1);
    const x3 = try lab.fetch(s3.xd, xd_len);
    defer lab.r.gpu.gpa.free(x3);
    const y1 = try lab.fetch(s1.y, y_len);
    defer lab.r.gpu.gpa.free(y1);
    const y3 = try lab.fetch(s3.y, y_len);
    defer lab.r.gpu.gpa.free(y3);
    var xd_diff: usize = 0;
    for (std.mem.bytesAsSlice(u16, @as([]align(1) const u8, x1)), std.mem.bytesAsSlice(u16, @as([]align(1) const u8, x3))) |a, b| {
        if (a != b) xd_diff += 1;
    }
    var y_diff: usize = 0;
    for (std.mem.bytesAsSlice(u32, @as([]align(1) const u8, y1)), std.mem.bytesAsSlice(u32, @as([]align(1) const u8, y3))) |a, b| {
        if (a != b) y_diff += 1;
    }
    std.debug.print("INFO glm53-p5a =3 vs =1 {s}: xd {d} of {d} halves differ, y {d} of {d} floats differ\n", .{ label, xd_diff, xd_len / 2, y_diff, y_len / 4 });
    return xd_diff == 0 and y_diff == 0;
}

fn checkDeterminism(lab: *const Lab, w: Weights, pick_host: []const i32) !void {
    const rows = pick_host.len / SLOTS;
    const pick = try lab.r.dev(std.mem.sliceAsBytes(pick_host));
    const l = layerOf(w, .mcg_e4m3, 0, 0);
    try lab.fresh();
    const a = try lab.runP5a(0, l, pick, 0, rows);
    const ay = try lab.fetch(a.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(ay);
    const ax = try lab.fetch(a.xd, rows * SLOTS * I * 2);
    defer lab.r.gpu.gpa.free(ax);
    try lab.fresh();
    const b = try lab.runP5a(1, l, pick, 0, rows);
    const by = try lab.fetch(b.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(by);
    const bx = try lab.fetch(b.xd, rows * SLOTS * I * 2);
    defer lab.r.gpu.gpa.free(bx);
    if (!std.mem.eql(u8, ay, by) or !std.mem.eql(u8, ax, bx)) return error.P5aNondeterministic;
    std.debug.print("PASS glm53-p5a determinism: repeated =10 y and xd bytes match\n", .{});
}

fn rowInvariance(lab: *const Lab, w: Weights, pick_host: []const i32) !void {
    const rows = pick_host.len / SLOTS;
    const pick = try lab.r.dev(std.mem.sliceAsBytes(pick_host));
    const l = layerOf(w, .mcg_e4m3, 0, 0);
    try lab.fresh();
    const all = try lab.runP5a(0, l, pick, 0, rows);
    const full = try lab.fetch(all.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(full);
    const row_bytes = SLOTS * D * 4;
    const split: usize = @min(@as(usize, 1000), rows - 1);
    try lab.fresh();
    const ref3_full_sc = try lab.runPmx(0, l, pick, rows);
    const ref3_full = try lab.fetch(ref3_full_sc.y, rows * row_bytes);
    defer lab.r.gpu.gpa.free(ref3_full);
    try lab.fresh();
    const ref3_piece_sc = try lab.runPmxAt(0, l, pick + split * SLOTS * 4, split, rows - split);
    const ref_got = try lab.fetch(ref3_piece_sc.y, (rows - split) * row_bytes);
    defer lab.r.gpu.gpa.free(ref_got);
    std.debug.print("INFO glm53-p5a =3 piece == =3 full batch: {}\n", .{std.mem.eql(u8, ref_got, ref3_full[split * row_bytes ..][0 .. (rows - split) * row_bytes])});
    std.debug.print("INFO glm53-p5a =3 full batch == =10 full batch: {}\n", .{std.mem.eql(u8, ref3_full, full)});
    var counts: [E]usize = @splat(0);
    for (pick_host) |e| counts[@intCast(e)] += 1;
    var single_row: ?usize = null;
    for (pick_host, 0..) |e, pair| {
        if (counts[@intCast(e)] == 1) {
            single_row = pair / SLOTS;
            break;
        }
    }
    for ([_]usize{ 0, single_row orelse 0, rows - 1 }) |row| {
        try lab.fresh();
        const one = try lab.runP5a(1, l, pick + row * SLOTS * 4, row, 1);
        const got = try lab.fetch(one.y, row_bytes);
        defer lab.r.gpu.gpa.free(got);
        // Control: production =3 on the same single row through the same harness.
        try lab.fresh();
        const s3 = try lab.sc(1, l);
        try lab.exp.prompt.runPmx(lab.pmx, lab.exp, lab.exp.router, lab.r.s, l, lab.x + row * D * 2, D, pick + row * SLOTS * 4, 1, s3, lab.plans[1]);
        const ref3 = try lab.fetch(s3.y, row_bytes);
        defer lab.r.gpu.gpa.free(ref3);
        const want = full[row * row_bytes ..][0..row_bytes];
        const p10 = std.mem.eql(u8, got, want);
        const p3 = std.mem.eql(u8, ref3, want);
        const same = std.mem.eql(u8, got, ref3);
        std.debug.print("INFO glm53-p5a row {d}: =10 alone == =10 batch {}, =3 alone == =10 batch {}, =10 alone == =3 alone {}\n", .{ row, p10, p3, same });
        try expectBytes("single row =10 vs =10 batch", got, want, D, SLOTS);
        if (!p10) {
            const gf = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, got));
            const wf = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, want));
            var shown: usize = 0;
            for (gf, wf, 0..) |g, w_, i| if (g != w_ and shown < 4) {
                std.debug.print("INFO glm53-p5a row {d} diff slot {d} col {d}: got {x:0>8} want {x:0>8}\n", .{ row, i / D, i % D, g, w_ });
                shown += 1;
            };
        }
        try expectBytes("single row =10 vs =3", got, ref3, D, SLOTS);
    }
    try lab.fresh();
    const piece = try lab.runP5a(1, l, pick + split * SLOTS * 4, split, rows - split);
    const got = try lab.fetch(piece.y, (rows - split) * row_bytes);
    defer lab.r.gpu.gpa.free(got);
    try expectBytes("split piece =10 vs =10 full batch", got, full[split * row_bytes ..][0 .. (rows - split) * row_bytes], D, (rows - split) * SLOTS);
    std.debug.print("PASS glm53-p5a row invariance: =10 split piece == =10 full batch (rows 0, 1, {d}, split {d}+{d})\n", .{ rows - 1, split, rows - split });
}

const Counts = struct { le16: usize = 0, n17_32: usize = 0, n33_64: usize = 0, n65_128: usize = 0 };
fn itemCounts(pick: []const i32) Counts {
    var per: [E]usize = @splat(0);
    for (pick) |e| per[@intCast(e)] += 1;
    var out: Counts = .{};
    for (per) |count| {
        var rem = count;
        while (rem != 0) {
            const n = @min(rem, exl3e.prompt.wide_tile);
            if (n <= 16) out.le16 += 1 else if (n <= 32) out.n17_32 += 1 else if (n <= 64) out.n33_64 += 1 else out.n65_128 += 1;
            rem -= n;
        }
    }
    return out;
}
fn pmxGateUp(lab: *const Lab, l: exl3e.Layer, sc: exl3e.prompt.Scratch, p: cuda.grouped.Plan, rows: usize) !void {
    const items = cuda.grouped.maxItems(rows * SLOTS, E, exl3e.prompt.wide_tile);
    const c: usize = if (l.codebook == .mcg_e4m3) 1 else 0;
    var small_args = smallGuArgs(l, sc, p);
    try cuda.launch.launch(lab.exp.prompt.wide.?.gu_small[c], .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, lab.r.s, &small_args);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.PmxArgs{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = l.gate, .tp1 = l.up, .k2_0 = l.gate_k2, .k2_1 = l.up_k2, .items = p.items, .counts = p.counts, .members = p.members, .sv0 = l.svh_g, .sv1 = l.svh_u, .sd = l.suh_d, .xd = sc.xd, .y = sc.y, .K = D, .N = I, .E = E, .limit = l.limit, .act_mode = l.act_mode });
    try cuda.launch.launch(lab.pmx.gu[c], .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = exl3e.pmx.gu_smem }, lab.r.s, &args);
}
const BenchPhase = enum { ref_gu, ref_down, new_gu, new_down };
fn benchCall(lab: *const Lab, phase: BenchPhase, l: exl3e.Layer, sc: exl3e.prompt.Scratch, plan: cuda.grouped.Plan, rows: usize) !void {
    switch (phase) {
        .ref_gu => try pmxGateUp(lab, l, sc, plan, rows),
        .ref_down => try lab.exp.prompt.downWide(lab.r.s, l, rows, sc, plan),
        .new_gu => try lab.exp.prompt.gateUpP5a(lab.pmx, lab.p5a, lab.r.s, l, rows, sc, plan),
        .new_down => try lab.exp.prompt.downP5a(lab.p5a, lab.r.s, l, rows, sc, plan, false),
    }
}
fn timed(lab: *const Lab, ev: *const [2]cuda.Event, phase: BenchPhase, l: exl3e.Layer, sc: exl3e.prompt.Scratch, plan: cuda.grouped.Plan, rows: usize) !f64 {
    for (0..2) |_| try benchCall(lab, phase, l, sc, plan, rows);
    try ev[0].record(lab.r.s);
    for (0..10) |_| try benchCall(lab, phase, l, sc, plan, rows);
    try ev[1].record(lab.r.s);
    try lab.r.s.synchronize();
    return @as(f64, try ev[0].elapsedMs(ev[1])) * 100.0;
}
fn flops(rows: usize) f64 {
    return 2.0 * @as(f64, @floatFromInt(rows * SLOTS)) * D * I;
}
fn tflops(rows: usize, us: f64) f64 {
    return flops(rows) / us / 1e6;
}
fn benchRows(lab: *const Lab, w: Weights, rows: usize, ev: *const [2]cuda.Event) !void {
    const host = try benchPicks(lab.r.a, rows);
    const pick = try lab.r.dev(std.mem.sliceAsBytes(host));
    const l = layerOf(w, .mcg_e4m3, 0, 0);
    const c = itemCounts(host);
    const ref_sc = try lab.prepare(0, l, pick, rows);
    const new_sc = try lab.prepare(1, l, pick, rows);
    const ref_gu = try timed(lab, ev, .ref_gu, l, ref_sc, lab.plans[0], rows);
    const ref_down = try timed(lab, ev, .ref_down, l, ref_sc, lab.plans[0], rows);
    const new_gu = try timed(lab, ev, .new_gu, l, new_sc, lab.plans[1], rows);
    const new_down = try timed(lab, ev, .new_down, l, new_sc, lab.plans[1], rows);
    const stage_ref = (ref_gu + ref_down) * MoeLayers / @as(f64, @floatFromInt(rows));
    const stage_new = (new_gu + new_down) * MoeLayers / @as(f64, @floatFromInt(rows));
    std.debug.print("BENCH glm53-p5 cand=p5a mode=10 rows={d} items<=16 {d} 17-32 {d} 33-64 {d} 65-128 {d} | ref=3 gu {d:.1} us ({d:.2} TFLOP/s) down {d:.1} us ({d:.2} TFLOP/s) | new gu {d:.1} us ({d:.2} TFLOP/s) down {d:.1} us ({d:.2} TFLOP/s) | gu+down {d:.3}x | us/token/rank ref {d:.1} new {d:.1} saved {d:.1}\n", .{ rows, c.le16, c.n17_32, c.n33_64, c.n65_128, ref_gu, tflops(rows, ref_gu), ref_down, tflops(rows, ref_down), new_gu, tflops(rows, new_gu), new_down, tflops(rows, new_down), (ref_gu + ref_down) / (new_gu + new_down), stage_ref, stage_new, stage_ref - stage_new });
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_p5a.len == 0 or cuda.kernels.exl3_experts.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var p5a = try exl3e.p5a.P5a.load(gpu.d);
    defer p5a.unload();
    if (exp.prompt.wide == null) return error.Exl3PromptWideMissing;
    try resources(gpu, &p5a);
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const rig = try arenaRig(gpu, &arena);
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const w = try buildWeights(rig, w46);
    const lab = try makeLab(rig, &exp, &pmx, &p5a, w);
    const size_pick = try sizesPicks(lab.r);
    const skew = try skewedPicks(lab.r, 3072);
    const uni = try uniformPicks(lab.r, 2048);
    const routings = [_]struct { name: []const u8, picks: []const i32 }{ .{ .name = "sizes", .picks = size_pick }, .{ .name = "skewed", .picks = skew }, .{ .name = "uniform", .picks = uni } };
    const cases = [_]struct { cb: exl3e.Codebook, act: i32, limit: f32, name: []const u8 }{ .{ .cb = .mcg_e4m3, .act = 0, .limit = 0, .name = "mcg_e4m3 act=0" }, .{ .cb = .mcg_e4m3, .act = 1, .limit = 0, .name = "mcg_e4m3 act=1" }, .{ .cb = .mcg, .act = 0, .limit = 10, .name = "mcg act=0 clamp=10" }, .{ .cb = .mcg, .act = 1, .limit = 0, .name = "mcg act=1" } };
    for (cases) |case| for (routings) |routing| {
        var label: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&label, "{s} {s}", .{ case.name, routing.name });
        try checkFull(&lab, w, case.cb, case.act, case.limit, name, routing.picks);
    };
    try checkDeterminism(&lab, w, size_pick);
    var pmx_exact = true;
    for (cases) |case| {
        var gap_label: [128]u8 = undefined;
        const gap_name = try std.fmt.bufPrint(&gap_label, "{s} sizes", .{case.name});
        pmx_exact = (try pmxGap(&lab, w, case.cb, case.act, case.limit, gap_name, size_pick)) and pmx_exact;
    }
    if (!pmx_exact) {
        std.debug.print("FAIL glm53-p5a =3 vs =1: PMX gate/up items above 64 members are not bit-equal to the =1 kernels\n", .{});
        return error.P5aBits;
    }
    std.debug.print("PASS glm53-p5a =3 vs =1: xd and y bit-equal on the sizes routing, 4 cases\n", .{});
    try rowInvariance(&lab, w, size_pick);
    const wg = try buildWeights(lab.r, w246);
    for ([_]exl3e.Codebook{ .mcg_e4m3, .mcg }) |cb| {
        try compareSmall(&lab, wg, cb, "sizes K1/K2/K3", size_pick);
        try compareSmall(&lab, wg, cb, "skewed K1/K2/K3", skew);
        try compareSmall(&lab, wg, cb, "uniform K1/K2/K3", uni);
    }
    var ev: [2]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    const wb = try buildWeights(lab.r, wbench);
    try benchRows(&lab, wb, 2048, &ev);
    try benchRows(&lab, wb, 4096, &ev);
}

fn arenaRig(gpu: anytype, arena: *std.heap.ArenaAllocator) !*Rig {
    var prng = std.Random.DefaultPrng.init(0x5305_1280);
    const r = try arena.allocator().create(Rig);
    r.* = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    return r;
}

fn same8Picks(a: std.mem.Allocator, rows: usize) ![]i32 {
    const out = try a.alloc(i32, rows * SLOTS);
    const first = (rows * 37) % 248;
    for (0..rows) |row| {
        for (0..SLOTS) |slot| {
            out[row * SLOTS + slot] = @intCast(first + slot);
        }
    }
    return out;
}

fn everyPicks(a: std.mem.Allocator, rows: usize) ![]i32 {
    std.debug.print("INFO glm53-p5ax everyPicks rows={d} total_want={d} SLOTS={d} E={d} ROWS_MAX={d}\n", .{ rows, everyTotalWant(), SLOTS, E, ROWS_MAX });
    const out = try a.alloc(i32, rows * SLOTS);
    @memset(out, -1);
    const want = everyWant();
    var filled: [ROWS_MAX]usize = @splat(0);
    var cursor: usize = 0;
    for (0..72) |expert| {
        for (0..want[expert]) |_| {
            const row = cursor % rows;
            cursor += 1;
            if (filled[row] == SLOTS) {
                std.debug.print("INFO glm53-p5ax stage=overflow row={d} expert={d} rows={d} SLOTS={d} E={d} ROWS_MAX={d}\n", .{ row, expert, rows, SLOTS, E, ROWS_MAX });
                return error.P5axRouting;
            }
            out[row * SLOTS + filled[row]] = @intCast(expert);
            filled[row] += 1;
        }
    }
    var next_expert: usize = 72;
    for (0..rows) |row| {
        while (filled[row] < SLOTS) {
            var found = false;
            var attempts: usize = 0;
            while (attempts < E - 72) : (attempts += 1) {
                const expert = next_expert;
                next_expert = if (next_expert + 1 == E) 72 else next_expert + 1;
                var present = false;
                for (out[row * SLOTS ..][0..filled[row]]) |v| {
                    present = present or v == @as(i32, @intCast(expert));
                }
                if (!present) {
                    out[row * SLOTS + filled[row]] = @intCast(expert);
                    filled[row] += 1;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.debug.print("INFO glm53-p5ax stage=fill row={d} filled={d} rows={d} SLOTS={d} E={d} ROWS_MAX={d}\n", .{ row, filled[row], rows, SLOTS, E, ROWS_MAX });
                return error.P5axRouting;
            }
        }
    }
    var actual: [E]usize = @splat(0);
    for (out, 0..) |expert, pair| {
        if (expert < 0) {
            std.debug.print("INFO glm53-p5ax stage=negative row={d} rows={d} SLOTS={d} E={d} ROWS_MAX={d}\n", .{ pair / SLOTS, rows, SLOTS, E, ROWS_MAX });
            return error.P5axRouting;
        }
        const index: usize = @intCast(expert);
        actual[index] += 1;
    }
    if (!std.mem.eql(usize, actual[0..72], want[0..72])) {
        std.debug.print("INFO glm53-p5ax stage=counts rows={d} SLOTS={d} E={d} ROWS_MAX={d}\n", .{ rows, SLOTS, E, ROWS_MAX });
        for (0..72) |expert| {
            if (actual[expert] != want[expert]) {
                std.debug.print("INFO glm53-p5ax stage=counts expert={d} actual={d} want={d}\n", .{ expert, actual[expert], want[expert] });
                break;
            }
        }
        return error.P5axRouting;
    }
    return out;
}

fn compareScratch(lab: *const Lab, label: []const u8, ref: exl3e.prompt.Scratch, got: exl3e.prompt.Scratch, rows: usize, case_rows: usize, routing: []const u8, rank: usize, first: usize) !void {
    const xd = try lab.fetch(got.xd, rows * SLOTS * I * 2);
    defer lab.r.gpu.gpa.free(xd);
    const ref_xd = try lab.fetch(ref.xd, rows * SLOTS * I * 2);
    defer lab.r.gpu.gpa.free(ref_xd);
    std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ case_rows, routing, label, rank, first });
    try expectBytes(label, xd, ref_xd, I, rows * SLOTS);
    const y = try lab.fetch(got.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(y);
    const ref_y = try lab.fetch(ref.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(ref_y);
    std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ case_rows, routing, label, rank, first });
    try expectBytes(label, y, ref_y, D, rows * SLOTS);
}

fn compareP5axCase(lab: *const Lab, l: exl3e.Layer, pick: u64, rows: usize, routing: []const u8, rank: usize, label: []const u8) !void {
    try lab.fresh();
    const ref = try lab.runPmx(0, l, pick, rows);
    const got = try lab.runP5a(1, l, pick, 0, rows);
    std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ rows, routing, label, rank, @as(usize, 0) });
    try compareScratch(lab, label, ref, got, rows, rows, routing, rank, 0);
    try lab.combine(0, l, ref.y, rows);
    try lab.combine(1, l, got.y, rows);
    const ref_out = try lab.fetch(lab.out[0], rows * D * 4);
    defer lab.r.gpu.gpa.free(ref_out);
    const got_out = try lab.fetch(lab.out[1], rows * D * 4);
    defer lab.r.gpu.gpa.free(got_out);
    std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ rows, routing, label, rank, @as(usize, 0) });
    try expectBytes(label, got_out, ref_out, D, rows);

    const split_points = [_]usize{ 1, 63, 64, 65, 127, 128, 129, rows / 2, rows - 1 };
    var splits: usize = 0;
    for (split_points) |first| {
        if (first == 0 or first >= rows) continue;
        try lab.fresh();
        const piece_pick = pick + first * SLOTS * 4;
        const piece_rows = rows - first;
        const ref_piece = try lab.runPmxAt(0, l, piece_pick, first, piece_rows);
        const got_piece = try lab.runP5a(1, l, piece_pick, first, piece_rows);
        std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ rows, routing, label, rank, first });
        try compareScratch(lab, label, ref_piece, got_piece, piece_rows, rows, routing, rank, first);
        try lab.combine(0, l, ref_piece.y, piece_rows);
        try lab.combine(1, l, got_piece.y, piece_rows);
        {
            const rp = try lab.fetch(lab.out[0], piece_rows * D * 4);
            defer lab.r.gpu.gpa.free(rp);
            const gp = try lab.fetch(lab.out[1], piece_rows * D * 4);
            defer lab.r.gpu.gpa.free(gp);
            std.debug.print("INFO glm53-p5ax check rows {d} {s} {s} rank {d} first {d}\n", .{ rows, routing, label, rank, first });
            try expectBytes(label, gp, rp, D, piece_rows);
        }
        splits += 1;
    }
    std.debug.print("RESULT glm53-p5ax rows {d} {s} rank {d}: xd/y/out bytes equal, splits {d}\n", .{ rows, routing, rank, splits });
}

fn reportTileGap(lab: *const Lab, l: exl3e.Layer, pick: u64, rows: usize, routing: []const u8) !void {
    try lab.fresh();
    const ref = try lab.runPmx(0, l, pick, rows);
    const ref_y = try lab.fetch(ref.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(ref_y);
    try lab.fresh();
    const sc = try lab.sc(0, l);
    try lab.exp.prompt.runTile(lab.exp, lab.exp.router, lab.r.s, l, lab.x, D, pick, rows, sc, lab.plans[0], exl3e.prompt.tile);
    const tile_y = try lab.fetch(sc.y, rows * SLOTS * D * 4);
    defer lab.r.gpu.gpa.free(tile_y);
    const ref_f = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, ref_y));
    const tile_f = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, tile_y));
    var different: usize = 0;
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    for (tile_f, ref_f) |got, want| {
        if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(want))) different += 1;
        const diff = @abs(got - want);
        max_abs = @max(max_abs, diff);
        const rel = diff / @max(@abs(want), 1.0e-30);
        max_rel = @max(max_rel, rel);
    }
    std.debug.print("INFO glm53-p5ax =1 vs =3 rows {d} {s}: {d} of {d} y floats differ, max |diff| {e}, max rel {e}\n", .{ rows, routing, different, tile_f.len, max_abs, max_rel });
}

pub fn sweep(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_p5a.len == 0 or cuda.kernels.exl3_experts.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var p5a = try exl3e.p5a.P5a.load(gpu.d);
    defer p5a.unload();
    if (exp.prompt.wide == null) return error.Exl3PromptWideMissing;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const rig = try arenaRig(gpu, &arena);
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const w0 = try buildWeights(rig, w46);
    const w1 = try buildWeights(rig, w46);
    var pairs: [4]bool = @splat(false);
    for (0..E) |expert| {
        const gate_narrow = w46(expert, 0) == 4;
        const up_narrow = w46(expert, 1) == 4;
        const ix = @as(usize, if (gate_narrow) 0 else 2) + @as(usize, if (up_narrow) 0 else 1);
        pairs[ix] = true;
    }
    for (pairs) |found| {
        if (!found) return error.P5axCoverage;
    }
    std.debug.print("INFO glm53-p5ax weight bytes per rank: {d} (two ranks total {d})\n", .{ (D / 16) * (I / 16) * 10 * 16 * 3 * E, (D / 16) * (I / 16) * 10 * 16 * 3 * E * 2 });
    const lab0 = try makeLab(rig, &exp, &pmx, &p5a, w0);
    const lab1 = try makeLab(rig, &exp, &pmx, &p5a, w1);
    const row_values = [_]usize{ 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 1536, 2047, 2048 };
    var configs: usize = 0;
    for (row_values) |rows| {
        const uniform = try uniformPicks(rig, rows);
        const skewed = try skewedPicks(rig, rows);
        const same = try same8Picks(rig.a, rows);
        var every: ?[]i32 = null;
        if (rows >= 256 and rows * SLOTS >= everyTotalWant()) {
            every = try everyPicks(rig.a, rows);
        }
        const routing_set = [_]struct { name: []const u8, picks: []const i32 }{
            .{ .name = "uniform", .picks = uniform }, .{ .name = "skewed", .picks = skewed }, .{ .name = "same8", .picks = same },
        };
        const extended = [_]struct { cb: exl3e.Codebook, act: i32, limit: f32, name: []const u8 }{
            .{ .cb = .mcg_e4m3, .act = 0, .limit = 0, .name = "mcg_e4m3" },
            .{ .cb = .mcg_e4m3, .act = 1, .limit = 0, .name = "mcg_e4m3_act1" },
            .{ .cb = .mcg, .act = 0, .limit = 10, .name = "mcg_limit10" },
        };
        for (routing_set) |routing| for (0..2) |rank| {
            const w = if (rank == 0) w0 else w1;
            const lab = if (rank == 0) &lab0 else &lab1;
            var production: [128]u8 = undefined;
            const prod_label = try std.fmt.bufPrint(&production, "{s}", .{routing.name});
            try compareSweepEntry(lab, w, rows, routing.name, routing.picks, .mcg_e4m3, 0, 0, prod_label, rank);
            configs += 1;
            if (rows == 1 or rows == 64 or rows == 65 or rows == 513 or rows == 2048) {
                for (extended[1..]) |case| {
                var label_buf: [128]u8 = undefined;
                const label = try std.fmt.bufPrint(&label_buf, "{s}", .{case.name});
                try compareSweepEntry(lab, w, rows, routing.name, routing.picks, case.cb, case.act, case.limit, label, rank);
                configs += 1;
                }
            }
            if (rank == 0 and (rows == 64 or rows == 512 or rows == 2048)) {
                if (std.mem.eql(u8, routing.name, "same8") or std.mem.eql(u8, routing.name, "uniform")) {
                    try reportTileGap(lab, layerOf(w, .mcg_e4m3, 0, 0), try lab.r.dev(std.mem.sliceAsBytes(routing.picks)), rows, routing.name);
                }
            }
        };
        if (every) |picks| {
        for (0..2) |rank| {
            const w = if (rank == 0) w0 else w1;
            const lab = if (rank == 0) &lab0 else &lab1;
            try compareSweepEntry(lab, w, rows, "every", picks, .mcg_e4m3, 0, 0, "every", rank);
            configs += 1;
            if (rows == 513 or rows == 2048) {
                for (extended[1..]) |case| {
                    try compareSweepEntry(lab, w, rows, "every", picks, case.cb, case.act, case.limit, case.name, rank);
                    configs += 1;
                }
            }
        }
        }
    }
    std.debug.print("PASS glm53-p5ax: =10 byte-equal to =3 over {d} configurations\n", .{configs});
}

fn compareSweepEntry(lab: *const Lab, w: Weights, rows: usize, routing: []const u8, host_pick: []const i32, cb: exl3e.Codebook, act: i32, limit: f32, label: []const u8, rank: usize) !void {
    const pick = try lab.r.dev(std.mem.sliceAsBytes(host_pick));
    const l = layerOf(w, cb, limit, act);
    try compareP5axCase(lab, l, pick, rows, routing, rank, label);
}

pub fn bench(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_p5a.len == 0 or cuda.kernels.exl3_experts.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var p5a = try exl3e.p5a.P5a.load(gpu.d);
    defer p5a.unload();
    if (exp.prompt.wide == null) return error.Exl3PromptWideMissing;
    try resources(gpu, &p5a);
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const rig = try arenaRig(gpu, &arena);
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const weights = try buildWeights(rig, wbench);
    const lab = try makeLab(rig, &exp, &pmx, &p5a, weights);
    var ev: [2]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    try benchRows(&lab, weights, 2048, &ev);
    try benchRows(&lab, weights, 4096, &ev);
}
