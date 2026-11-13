// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! ROUTED-A8 bench: e4m3 rotated activations against release pmx gate/up.
//! glm53_had_tests.zig is not in this tree. The shape, wbench, and benchPicks are the p5a bench.
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

test "GLM-5.3 ra8 declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

const Weights = struct {
    ptrs: [3]u64 = .{ 0, 0, 0 },
    k2: [3]u64 = .{ 0, 0, 0 },
    suh_g: u64 = 0,
    suh_u: u64 = 0,
    svh_g: u64 = 0,
    svh_u: u64 = 0,
    suh_d: u64 = 0,
    svh_d: u64 = 0,
};
const Halves = *const fn (usize, usize) i32;

fn wbench(e: usize, m: usize) i32 {
    return if ((e * 7 + m * 3) % 9 == 0) 6 else 4;
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
fn planOf(r: *Rig) !cuda.grouped.Plan {
    return .{ .members = try r.zeros(PAIRS_MAX * 4), .items = try r.zeros(cuda.grouped.maxItems(PAIRS_MAX, E, exl3e.prompt.tile) * 12), .counts = try r.zeros(16), .rank = try r.zeros(PAIRS_MAX * 4), .hist = try r.zeros((PAIRS_MAX + 1023) / 1024 * E * 4) };
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
            if (!dup) {
                out[row * SLOTS + k] = @intCast(e);
                k += 1;
            }
        }
    }
    return out;
}

const Call = struct {
    exp: *const exl3e.Experts = undefined,
    pmx: *const exl3e.pmx.Pmx = undefined,
    stream: cuda.Stream = undefined,
    layer: exl3e.Layer = undefined,
    x: u64 = 0,
    picks: u64 = 0,
    rows: usize = 0,
    sc: exl3e.prompt.Scratch = undefined,
    route_plan: cuda.grouped.Plan = undefined,
    code0: u64 = 0,
    code1: u64 = 0,
    scale0: u64 = 0,
    scale1: u64 = 0,
};

const Which = enum { rot_in, gu, rot_a8, gu_a8, gu_w8, rot_w8s, rot_dd, gu_w8s, gu_w8s_pf, gu_w8s_m, down, xd_w8s, down_w8s };

const Metrics = struct {
    max_abs: f64 = 0,
    max_rel: f64 = 0,
    mean_abs: f64 = 0,
    mean_rel: f64 = 0,
    nonfinite: usize = 0,
};

const PmxSet = struct {
    rows: []i32 = &.{},
    items: usize = 0,
    pairs: usize = 0,
};

const Outcome = struct {
    failed: bool = false,
    slow: bool = false,
};

fn wipe(r: *Rig, ptr: u64, len: usize) !void {
    const buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
    try buf.fill8(0xff, r.s.handle);
}

fn fetch(r: *Rig, ptr: u64, len: usize) ![]u8 {
    const out = try r.gpu.gpa.alloc(u8, len);
    errdefer r.gpu.gpa.free(out);
    try r.s.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
    try buf.download(0, out);
    return out;
}

fn copyI32(a: std.mem.Allocator, bytes: []const u8) ![]i32 {
    if (bytes.len % 4 != 0) return error.Ra8Plan;
    const out = try a.alloc(i32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn fetchHalves(r: *Rig, ptr: u64, len: usize) ![]u16 {
    const bytes = try fetch(r, ptr, len);
    defer r.gpu.gpa.free(bytes);
    if (bytes.len % 2 != 0) return error.Ra8Plan;
    const out = try r.gpu.gpa.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn pmxMembers(a: std.mem.Allocator, triples: []const i32, members: []const i32, nitems: usize) !PmxSet {
    const seen = try a.alloc(u8, members.len);
    @memset(seen, 0);
    var list: std.ArrayList(i32) = .empty;
    var n_items: usize = 0;
    for (0..nitems) |item| {
        const cnt = triples[item * 3 + 2];
        if (cnt < 0) return error.Ra8Plan;
        if (cnt <= 64) continue;
        const off: usize = @intCast(triples[item * 3 + 1]);
        const n: usize = @intCast(cnt);
        if (off > members.len or n > members.len - off) return error.Ra8Plan;
        n_items += 1;
        for (0..n) |j| {
            const row = members[off + j];
            if (row < 0) return error.Ra8Plan;
            const idx: usize = @intCast(row);
            if (idx >= members.len or seen[idx] != 0) return error.Ra8Plan;
            seen[idx] = 1;
            try list.append(a, row);
        }
    }
    const rows = try list.toOwnedSlice(a);
    return .{ .rows = rows, .items = n_items, .pairs = rows.len };
}

fn launchRot(c: *const Call) !void {
    const pairs = c.rows * SLOTS;
    var args: cuda.launch.Args = .{};
    inline for (.{ c.x, @as(i32, D), @as(i32, @intFromEnum(exl3e.DType.bf16)), c.picks, c.layer.suh_g, c.layer.suh_u, c.sc.xg, c.sc.xu, @as(i32, D), @as(i32, SLOTS), @as(i32, E) }) |v| args.add(v);
    try cuda.launch.launch(c.exp.rot_in, .{ .grid = .{ .x = @intCast(pairs), .y = D / 128, .z = 2 }, .block = .{ .x = 32 } }, c.stream, &args);
}

fn launchGu(c: *const Call) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.PmxArgs{
        .x0 = c.sc.xg,
        .x1 = c.sc.xu,
        .tp0 = c.layer.gate,
        .tp1 = c.layer.up,
        .k2_0 = c.layer.gate_k2,
        .k2_1 = c.layer.up_k2,
        .items = c.route_plan.items,
        .counts = c.route_plan.counts,
        .members = c.route_plan.members,
        .sv0 = c.layer.svh_g,
        .sv1 = c.layer.svh_u,
        .sd = c.layer.suh_d,
        .xd = c.sc.xd,
        .y = c.sc.y,
        .K = D,
        .N = I,
        .E = E,
        .limit = c.layer.limit,
        .act_mode = c.layer.act_mode,
    });
    try cuda.launch.launch(c.pmx.gu[1], .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @as(u32, @intCast(exl3e.pmx.gu_smem)) }, c.stream, &args);
}

fn launchRotA8(c: *const Call) !void {
    const pairs = c.rows * SLOTS;
    var args: cuda.launch.Args = .{};
    args.add(exl3e.RotA8Args{
        .x = c.x,
        .x_stride = D,
        .x_dtype = @intFromEnum(exl3e.DType.bf16),
        .pick = c.picks,
        .suh0 = c.layer.suh_g,
        .suh1 = c.layer.suh_u,
        .out0 = c.code0,
        .out1 = c.code1,
        .sc0 = c.scale0,
        .sc1 = c.scale1,
        .k = D,
        .slots = SLOTS,
        .experts = E,
    });
    try cuda.launch.launch(c.exp.rot_in_a8, .{ .grid = .{ .x = @intCast(pairs), .y = D / 128, .z = 2 }, .block = .{ .x = 32 } }, c.stream, &args);
}

fn launchGuA8(c: *const Call) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.PmxA8Args{
        .x0 = c.code0,
        .x1 = c.code1,
        .s0 = c.scale0,
        .s1 = c.scale1,
        .tp0 = c.layer.gate,
        .tp1 = c.layer.up,
        .k2_0 = c.layer.gate_k2,
        .k2_1 = c.layer.up_k2,
        .items = c.route_plan.items,
        .counts = c.route_plan.counts,
        .members = c.route_plan.members,
        .sv0 = c.layer.svh_g,
        .sv1 = c.layer.svh_u,
        .sd = c.layer.suh_d,
        .xd = c.sc.xd,
        .y = c.sc.y,
        .K = D,
        .N = I,
        .E = E,
        .limit = c.layer.limit,
        .act_mode = c.layer.act_mode,
    });
    try cuda.launch.launch(c.pmx.gu_a8, .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @as(u32, @intCast(exl3e.pmx.gu_a8_smem)) }, c.stream, &args);
}

// One 512-thread CTA per (pair, matrix). Writes K-permuted codes and one scale per pair (scale0/scale1[pair]).
fn launchRotW8s(c: *const Call) !void {
    const pairs = c.rows * SLOTS;
    var args: cuda.launch.Args = .{};
    args.add(exl3e.RotA8Args{
        .x = c.x,
        .x_stride = D,
        .x_dtype = @intFromEnum(exl3e.DType.bf16),
        .pick = c.picks,
        .suh0 = c.layer.suh_g,
        .suh1 = c.layer.suh_u,
        .out0 = c.code0,
        .out1 = c.code1,
        .sc0 = c.scale0,
        .sc1 = c.scale1,
        .k = D,
        .slots = SLOTS,
        .experts = E,
    });
    try cuda.launch.launch(c.exp.rot_in_w8s, .{ .grid = .{ .x = @intCast(pairs), .y = 2, .z = 1 }, .block = .{ .x = 512 } }, c.stream, &args);
}

// TF_GLM53_ROT_DD: one 512-thread CTA a row, the same RotA8Args and output bytes as launchRotW8s.
fn launchRotW8sDd(c: *const Call) !void {
    var args: cuda.launch.Args = .{};
    args.add(exl3e.RotA8Args{
        .x = c.x,
        .x_stride = D,
        .x_dtype = @intFromEnum(exl3e.DType.bf16),
        .pick = c.picks,
        .suh0 = c.layer.suh_g,
        .suh1 = c.layer.suh_u,
        .out0 = c.code0,
        .out1 = c.code1,
        .sc0 = c.scale0,
        .sc1 = c.scale1,
        .k = D,
        .slots = SLOTS,
        .experts = E,
    });
    try cuda.launch.launch(c.exp.rot_in_w8s_row, .{ .grid = .{ .x = @intCast(c.rows), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, c.stream, &args);
}

fn launchGuW8(c: *const Call) !void {
    return launchGuCodes(c, c.pmx.gu_w8);
}

fn launchGuW8s(c: *const Call) !void {
    return launchGuCodes(c, c.pmx.gu_w8s);
}

fn launchGuW8sPf(c: *const Call) !void {
    return launchGuCodes(c, c.pmx.gu_w8s_pf);
}

fn launchGuW8sM(c: *const Call) !void {
    return launchGuCodes(c, c.pmx.gu_w8s_m);
}

fn launchGuCodes(c: *const Call, fun: anytype) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.PmxA8Args{
        .x0 = c.code0,
        .x1 = c.code1,
        .s0 = c.scale0,
        .s1 = c.scale1,
        .tp0 = c.layer.gate,
        .tp1 = c.layer.up,
        .k2_0 = c.layer.gate_k2,
        .k2_1 = c.layer.up_k2,
        .items = c.route_plan.items,
        .counts = c.route_plan.counts,
        .members = c.route_plan.members,
        .sv0 = c.layer.svh_g,
        .sv1 = c.layer.svh_u,
        .sd = c.layer.suh_d,
        .xd = c.sc.xd,
        .y = c.sc.y,
        .K = D,
        .N = I,
        .E = E,
        .limit = c.layer.limit,
        .act_mode = c.layer.act_mode,
    });
    try cuda.launch.launch(fun, .{ .grid = .{ .x = I / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @as(u32, @intCast(exl3e.pmx.gu_w8_smem)) }, c.stream, &args);
}

// Release down of the items > 64 members (=2's wide down on fp16 xd), as exl3_prompt.downP5a launches it.
fn launchDown(c: *const Call) !void {
    const w = c.exp.prompt.wide orelse return error.Exl3PromptWideMissing;
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    inline for (.{ c.sc.xd, c.layer.down, c.route_plan.items, c.route_plan.counts, c.route_plan.members, c.layer.svh_d, c.sc.y, @as(i32, I), @as(i32, D), c.layer.down_k2 }) |v| args.add(v);
    try cuda.launch.launch(w.down[1], .{ .grid = .{ .x = D / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = exl3e.prompt.wide_threads }, .shared = exl3e.prompt.wide_smem }, c.stream, &args);
}

// TF_GLM53_DOWN_W8S phase 1, as exl3_prompt.downW8s: xd -> e4m3 codes + one scale a pair into the sc.xg pair slots.
fn launchXdW8s(c: *const Call) !void {
    var args: cuda.launch.Args = .{};
    inline for (.{ c.sc.xd, c.picks, c.sc.xg, @as(i32, I), @as(i32, 2 * D) }) |v| args.add(v);
    try cuda.launch.launch(c.pmx.xd_w8s, .{ .grid = .{ .x = @intCast(c.rows * SLOTS), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, c.stream, &args);
}

fn launchDownW8s(c: *const Call) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.DownW8Args{ .xq = c.sc.xg, .td = c.layer.down, .k2 = c.layer.down_k2, .items = c.route_plan.items, .counts = c.route_plan.counts, .members = c.route_plan.members, .svh = c.layer.svh_d, .y = c.sc.y, .K = I, .D = D, .xs = 2 * D });
    try cuda.launch.launch(c.pmx.down_w8s, .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @as(u32, @intCast(exl3e.pmx.down_w8_smem)) }, c.stream, &args);
}

fn launchDownW8sFun(c: *const Call, fun: anytype) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    args.add(exl3e.pmx.DownW8Args{ .xq = c.sc.xg, .td = c.layer.down, .k2 = c.layer.down_k2, .items = c.route_plan.items, .counts = c.route_plan.counts, .members = c.route_plan.members, .svh = c.layer.svh_d, .y = c.sc.y, .K = I, .D = D, .xs = 2 * D });
    try cuda.launch.launch(fun, .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @as(u32, @intCast(exl3e.pmx.down_w8_smem)) }, c.stream, &args);
}

// p5a small-item down on the release fp16 xd, at =10's launch shape (dims / 256, 256 threads).
fn launchP5aDownFun(c: *const Call, fun: anytype) !void {
    const items = cuda.grouped.maxItems(c.rows * SLOTS, E, exl3e.prompt.wide_tile);
    var args: cuda.launch.Args = .{};
    inline for (.{ c.sc.xd, c.layer.down, c.route_plan.items, c.route_plan.counts, c.route_plan.members, c.layer.svh_d, c.sc.y, @as(i32, I), @as(i32, D), @as(i32, SLOTS), c.layer.down_k2 }) |v| args.add(v);
    try cuda.launch.launch(fun, .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, c.stream, &args);
}

fn fetchF32(r: *Rig, ptr: u64, len: usize) ![]f32 {
    const bytes = try fetch(r, ptr, len);
    defer r.gpu.gpa.free(bytes);
    if (bytes.len % 4 != 0) return error.Ra8Plan;
    const out = try r.gpu.gpa.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn scoreY(live: []const i32, got: []const f32, ref: []const f32) Metrics {
    var out: Metrics = .{};
    var sum_abs: f64 = 0;
    var sum_rel: f64 = 0;
    var finite: usize = 0;
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * D;
        for (0..D) |col| {
            const g: f64 = @floatCast(got[start + col]);
            const w: f64 = @floatCast(ref[start + col]);
            if (!std.math.isFinite(g) or !std.math.isFinite(w)) {
                out.nonfinite += 1;
                continue;
            }
            const diff = @abs(g - w);
            const rel = diff / @max(@abs(w), @as(f64, 1.0e-6));
            if (diff > out.max_abs) out.max_abs = diff;
            if (rel > out.max_rel) out.max_rel = rel;
            sum_abs += diff;
            sum_rel += rel;
            finite += 1;
        }
    }
    if (finite != 0) {
        const denom: f64 = @floatFromInt(finite);
        out.mean_abs = sum_abs / denom;
        out.mean_rel = sum_rel / denom;
    }
    return out;
}

// Relative L2 error of the live rows: |got - ref| / |ref| over every element (the per-element mean_rel is dominated
// by near-zero outputs).
fn relL2(live: []const i32, got: []const f32, ref: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * D;
        for (0..D) |col| {
            const g: f64 = @floatCast(got[start + col]);
            const w: f64 = @floatCast(ref[start + col]);
            num += (g - w) * (g - w);
            den += w * w;
        }
    }
    return if (den > 0) @sqrt(num / den) else 0;
}

// DOWN-W8S check: the release fp16 xd of rot_in + gu, then the release wide down vs xd_w8s + down_w8s on that xd.
// Same live rows as the gate/up checks (the items > 64 members). Reference for scale: the e4m3 gate/up's own xd
// error against release (gu w8s line above).
fn downCheck(r: *Rig, c: *const Call, ev: *const [2]cuda.Event, live: []const i32) !bool {
    const y_len = c.rows * SLOTS * D * 4;
    try launchRot(c);
    try launchGu(c);
    try wipe(r, c.sc.y, y_len);
    try launchDown(c);
    const yref = try fetchF32(r, c.sc.y, y_len);
    defer r.gpu.gpa.free(yref);
    try wipe(r, c.sc.y, y_len);
    try launchXdW8s(c);
    try launchDownW8s(c);
    const yw = try fetchF32(r, c.sc.y, y_len);
    defer r.gpu.gpa.free(yw);
    try wipe(r, c.sc.y, y_len);
    try launchXdW8s(c);
    try launchDownW8s(c);
    const yw2 = try fetchF32(r, c.sc.y, y_len);
    defer r.gpu.gpa.free(yw2);
    var failed = false;
    const m = scoreY(live, yw, yref);
    std.debug.print("INFO glm53-ra8 rows={d} down_w8s max_abs={e} max_rel={e} mean_rel={e} mean_abs={e} rel_l2={e} nonfinite={d}\n", .{ c.rows, m.max_abs, m.max_rel, m.mean_rel, m.mean_abs, relL2(live, yw, yref), m.nonfinite });
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * D;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(yw[start .. start + D]), std.mem.sliceAsBytes(yw2[start .. start + D]))) {
            std.debug.print("FAIL glm53-ra8 rows={d} down_w8s repeat mismatch at row {d}\n", .{ c.rows, row });
            failed = true;
            break;
        }
    }
    if (!failed) std.debug.print("INFO glm53-ra8 rows={d} down_w8s repeat=equal\n", .{c.rows});
    if (m.nonfinite != 0) {
        std.debug.print("FAIL glm53-ra8 rows={d} down_w8s nonfinite={d}\n", .{ c.rows, m.nonfinite });
        failed = true;
    }
    if (relL2(live, yw, yref) > 0.1) {
        std.debug.print("FAIL glm53-ra8 rows={d} down_w8s rel_l2 above 0.1 (layout or scale bug)\n", .{c.rows});
        failed = true;
    }
    const us_down = try timed(c, ev, .down);
    const us_q = try timed(c, ev, .xd_w8s);
    const us_dw = try timed(c, ev, .down_w8s);
    std.debug.print("BENCH glm53-ra8 rows={d} | down {d:.1} us | down_w8s xd {d:.1} + down {d:.1} = {d:.1} us {d:.2}x\n", .{ c.rows, us_down, us_q, us_dw, us_q + us_dw, us_down / (us_q + us_dw) });
    return failed;
}

fn once(c: *const Call, which: Which) !void {
    switch (which) {
        .rot_in => try launchRot(c),
        .gu => try launchGu(c),
        .rot_a8 => try launchRotA8(c),
        .gu_a8 => try launchGuA8(c),
        .gu_w8 => try launchGuW8(c),
        .rot_w8s => try launchRotW8s(c),
        .rot_dd => try launchRotW8sDd(c),
        .gu_w8s => try launchGuW8s(c),
        .gu_w8s_pf => try launchGuW8sPf(c),
        .gu_w8s_m => try launchGuW8sM(c),
        .down => try launchDown(c),
        .xd_w8s => try launchXdW8s(c),
        .down_w8s => try launchDownW8s(c),
    }
}

fn timed(c: *const Call, ev: *const [2]cuda.Event, which: Which) !f64 {
    for (0..2) |_| try once(c, which);
    try ev[0].record(c.stream);
    for (0..10) |_| try once(c, which);
    try ev[1].record(c.stream);
    try c.stream.synchronize();
    return @as(f64, try ev[0].elapsedMs(ev[1])) * 100.0;
}

fn timedFun(c: *const Call, ev: *const [2]cuda.Event, fun: anytype) !f64 {
    for (0..2) |_| try launchGuCodes(c, fun);
    try ev[0].record(c.stream);
    for (0..10) |_| try launchGuCodes(c, fun);
    try ev[1].record(c.stream);
    try c.stream.synchronize();
    return @as(f64, try ev[0].elapsedMs(ev[1])) * 100.0;
}

fn timedDownFun(c: *const Call, ev: *const [2]cuda.Event, fun: anytype) !f64 {
    for (0..2) |_| try launchDownW8sFun(c, fun);
    try ev[0].record(c.stream);
    for (0..10) |_| try launchDownW8sFun(c, fun);
    try ev[1].record(c.stream);
    try c.stream.synchronize();
    return @as(f64, try ev[0].elapsedMs(ev[1])) * 100.0;
}

fn timedP5aDown(c: *const Call, ev: *const [2]cuda.Event, fun: anytype) !f64 {
    for (0..2) |_| try launchP5aDownFun(c, fun);
    try ev[0].record(c.stream);
    for (0..10) |_| try launchP5aDownFun(c, fun);
    try ev[1].record(c.stream);
    try c.stream.synchronize();
    return @as(f64, try ev[0].elapsedMs(ev[1])) * 100.0;
}

fn releaseUnread(live: []const i32, ref: []const u16) bool {
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * I;
        for (0..I) |col| {
            if (ref[start + col] != 0xffff) return false;
        }
    }
    return true;
}

fn score(live: []const i32, got: []const u16, ref: []const u16) Metrics {
    var out: Metrics = .{};
    var sum_abs: f64 = 0;
    var sum_rel: f64 = 0;
    var finite: usize = 0;
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * I;
        for (0..I) |col| {
            const idx = start + col;
            const g: f32 = @as(f32, @floatCast(@as(f16, @bitCast(got[idx]))));
            const w: f32 = @as(f32, @floatCast(@as(f16, @bitCast(ref[idx]))));
            if (!std.math.isFinite(g) or !std.math.isFinite(w)) {
                out.nonfinite += 1;
                continue;
            }
            const diff = @abs(@as(f64, @floatCast(g)) - @as(f64, @floatCast(w)));
            const rel = diff / @max(@abs(@as(f64, @floatCast(w))), @as(f64, 1.0e-6));
            if (diff > out.max_abs) out.max_abs = diff;
            if (rel > out.max_rel) out.max_rel = rel;
            sum_abs += diff;
            sum_rel += rel;
            finite += 1;
        }
    }
    if (finite != 0) {
        const denom: f64 = @floatFromInt(finite);
        out.mean_abs = sum_abs / denom;
        out.mean_rel = sum_rel / denom;
    }
    return out;
}

fn firstMismatch(live: []const i32, a: []const u16, b: []const u16) ?usize {
    for (live) |row| {
        const start = @as(usize, @intCast(row)) * I;
        for (0..I) |col| {
            const idx = start + col;
            if (a[idx] != b[idx]) return idx;
        }
    }
    return null;
}

fn firstByteDiff(a: []const u8, b: []const u8) ?usize {
    const n = @min(a.len, b.len);
    for (0..n) |at| {
        if (a[at] != b[at]) return at;
    }
    return if (a.len == b.len) null else n;
}

fn rotddCompare(rows: usize, mat: usize, kind: []const u8, stride: usize, ref: []const u8, got: []const u8) bool {
    const at = firstByteDiff(ref, got) orelse return false;
    if (at >= ref.len or at >= got.len) {
        std.debug.print("FAIL glm53-rotdd rows={d} mat={d} {s} len={d}/{d}\n", .{ rows, mat, kind, got.len, ref.len });
        return true;
    }
    std.debug.print("FAIL glm53-rotdd rows={d} mat={d} {s} pair={d} byte={d} got={d} ref={d}\n", .{ rows, mat, kind, at / stride, at % stride, got[at], ref[at] });
    return true;
}

const DownLive = struct { large: []i32, small: []i32 };

// The pair indices the two down paths write: pmx down_w8s owns items > 64 members, p5a owns items <= 64.
fn downLive(r: *Rig, route_plan: cuda.grouped.Plan, pairs: usize) !DownLive {
    const count_bytes = try fetch(r, route_plan.counts, 16);
    defer r.gpu.gpa.free(count_bytes);
    const counts = try copyI32(r.a, count_bytes);
    if (counts[0] < 0) return error.Ra8Plan;
    if (counts[0] == 0) return error.Ra8Empty;
    const nitems: usize = @intCast(counts[0]);
    const cap = cuda.grouped.maxItems(PAIRS_MAX, E, exl3e.prompt.tile);
    if (nitems > cap) return error.Ra8Plan;
    const item_bytes = try fetch(r, route_plan.items, nitems * 12);
    defer r.gpu.gpa.free(item_bytes);
    const triples = try copyI32(r.a, item_bytes);
    const member_bytes = try fetch(r, route_plan.members, pairs * 4);
    defer r.gpu.gpa.free(member_bytes);
    const members = try copyI32(r.a, member_bytes);
    const seen = try r.a.alloc(u8, pairs);
    @memset(seen, 0);
    var large: std.ArrayList(i32) = .empty;
    var small: std.ArrayList(i32) = .empty;
    for (0..nitems) |item| {
        const cnt = triples[item * 3 + 2];
        if (cnt < 0) return error.Ra8Plan;
        const off: usize = @intCast(triples[item * 3 + 1]);
        const n: usize = @intCast(cnt);
        if (off > members.len or n > members.len - off) return error.Ra8Plan;
        for (0..n) |j| {
            const row = members[off + j];
            if (row < 0) return error.Ra8Plan;
            const idx: usize = @intCast(row);
            if (idx >= pairs or seen[idx] != 0) return error.Ra8Plan;
            seen[idx] = 1;
            if (cnt > 64) {
                try large.append(r.a, row);
            } else {
                try small.append(r.a, row);
            }
        }
    }
    return .{ .large = try large.toOwnedSlice(r.a), .small = try small.toOwnedSlice(r.a) };
}

fn firstDownMismatch(live: []const i32, a: []const u8, b: []const u8, dims: usize) ?struct { pair: i32, col: usize } {
    for (live) |pair| {
        const row: usize = @intCast(pair);
        for (0..dims) |col| {
            const off: usize = (row * dims + col) * 4;
            if (a[off] != b[off] or a[off + 1] != b[off + 1] or a[off + 2] != b[off + 2] or a[off + 3] != b[off + 3]) {
                return .{ .pair = pair, .col = col };
            }
        }
    }
    return null;
}

fn readPlan(r: *Rig, route_plan: cuda.grouped.Plan, pairs: usize) !PmxSet {
    const count_bytes = try fetch(r, route_plan.counts, 16);
    defer r.gpu.gpa.free(count_bytes);
    const counts = try copyI32(r.a, count_bytes);
    if (counts[0] < 0) return error.Ra8Plan;
    if (counts[0] == 0) return error.Ra8Empty;
    const nitems: usize = @intCast(counts[0]);
    const cap = cuda.grouped.maxItems(PAIRS_MAX, E, exl3e.prompt.tile);
    if (nitems > cap) return error.Ra8Plan;
    const item_bytes = try fetch(r, route_plan.items, nitems * 12);
    defer r.gpu.gpa.free(item_bytes);
    const triples = try copyI32(r.a, item_bytes);
    const member_bytes = try fetch(r, route_plan.members, pairs * 4);
    defer r.gpu.gpa.free(member_bytes);
    const members = try copyI32(r.a, member_bytes);
    const set = try pmxMembers(r.a, triples, members, nitems);
    if (set.pairs == 0) return error.Ra8Empty;
    return set;
}

fn oneRow(r: *Rig, c: *const Call, ev: *const [2]cuda.Event) !Outcome {
    const pairs = c.rows * SLOTS;
    const xd_len = pairs * I * 2;
    try c.exp.router.route(c.stream, c.picks, pairs, E, exl3e.prompt.wide_tile, c.route_plan);
    const set = try readPlan(r, c.route_plan, pairs);
    std.debug.print("INFO glm53-ra8 rows={d} pmx_items={d} pmx_pairs={d}\n", .{ c.rows, set.items, set.pairs });
    try wipe(r, c.sc.xd, xd_len);
    try launchRot(c);
    try launchGu(c);
    const ref = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(ref);
    if (releaseUnread(set.rows, ref)) {
        std.debug.print("FAIL glm53-ra8 rows={d} release xd unread\n", .{c.rows});
        return error.Ra8Unread;
    }
    try wipe(r, c.sc.xd, xd_len);
    try launchRotA8(c);
    try launchGuA8(c);
    const got = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(got);
    try wipe(r, c.sc.xd, xd_len);
    try launchRotA8(c);
    try launchGuA8(c);
    const again = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(again);
    const m = score(set.rows, got, ref);
    std.debug.print("INFO glm53-ra8 rows={d} max_abs={e} max_rel={e} mean_rel={e} mean_abs={e} nonfinite={d}\n", .{ c.rows, m.max_abs, m.max_rel, m.mean_rel, m.mean_abs, m.nonfinite });
    var failed = m.nonfinite != 0;
    if (firstMismatch(set.rows, got, again)) |at| {
        std.debug.print("FAIL glm53-ra8 rows={d} repeat mismatch at {d}\n", .{ c.rows, at });
        failed = true;
    } else {
        std.debug.print("INFO glm53-ra8 rows={d} repeat=equal\n", .{c.rows});
    }
    if (m.nonfinite != 0) std.debug.print("FAIL glm53-ra8 rows={d} nonfinite={d}\n", .{ c.rows, m.nonfinite });
    try wipe(r, c.sc.xd, xd_len);
    try launchRotA8(c);
    try launchGuW8(c);
    const got_w = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(got_w);
    try wipe(r, c.sc.xd, xd_len);
    try launchRotA8(c);
    try launchGuW8(c);
    const again_w = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(again_w);
    const mw = score(set.rows, got_w, ref);
    std.debug.print("INFO glm53-ra8 rows={d} w8 max_abs={e} max_rel={e} mean_rel={e} mean_abs={e} nonfinite={d}\n", .{ c.rows, mw.max_abs, mw.max_rel, mw.mean_rel, mw.mean_abs, mw.nonfinite });
    if (firstMismatch(set.rows, got_w, again_w)) |at| {
        std.debug.print("FAIL glm53-ra8 rows={d} w8 repeat mismatch at {d}\n", .{ c.rows, at });
        failed = true;
    } else {
        std.debug.print("INFO glm53-ra8 rows={d} w8 repeat=equal\n", .{c.rows});
    }
    if (mw.nonfinite != 0) {
        std.debug.print("FAIL glm53-ra8 rows={d} w8 nonfinite={d}\n", .{ c.rows, mw.nonfinite });
        failed = true;
    }
    try wipe(r, c.sc.xd, xd_len);
    try launchRotW8s(c);
    try launchGuW8s(c);
    const got_s = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(got_s);
    try wipe(r, c.sc.xd, xd_len);
    try launchRotW8s(c);
    try launchGuW8s(c);
    const again_s = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(again_s);
    const ms = score(set.rows, got_s, ref);
    std.debug.print("INFO glm53-ra8 rows={d} w8s max_abs={e} max_rel={e} mean_rel={e} mean_abs={e} nonfinite={d}\n", .{ c.rows, ms.max_abs, ms.max_rel, ms.mean_rel, ms.mean_abs, ms.nonfinite });
    if (firstMismatch(set.rows, got_s, again_s)) |at| {
        std.debug.print("FAIL glm53-ra8 rows={d} w8s repeat mismatch at {d}\n", .{ c.rows, at });
        failed = true;
    } else {
        std.debug.print("INFO glm53-ra8 rows={d} w8s repeat=equal\n", .{c.rows});
    }
    if (ms.nonfinite != 0) {
        std.debug.print("FAIL glm53-ra8 rows={d} w8s nonfinite={d}\n", .{ c.rows, ms.nonfinite });
        failed = true;
    }
    // TF_GLM53_GU_M: same args and smem as gu_w8s, so every live xd half must be byte-equal.
    try wipe(r, c.sc.xd, xd_len);
    try launchRotW8s(c);
    try launchGuW8sM(c);
    const got_m = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(got_m);
    if (firstMismatch(set.rows, got_m, got_s)) |at| {
        std.debug.print("FAIL glm53-gum rows={d} differs at {d}\n", .{ c.rows, at });
        failed = true;
    } else {
        std.debug.print("INFO glm53-gum rows={d} xd byte=equal\n", .{c.rows});
    }
    // TF_GLM53_ROT_DD: the row kernel must write the pair kernel's code and scale bytes. benchPicks has no negative
    // ids, so every pair of this run is live.
    {
        const code_len = pairs * D;
        const scale_len = pairs * 4;
        try wipe(r, c.code0, code_len);
        try wipe(r, c.code1, code_len);
        try wipe(r, c.scale0, scale_len);
        try wipe(r, c.scale1, scale_len);
        try launchRotW8s(c);
        const ref_code0 = try fetch(r, c.code0, code_len);
        const ref_code1 = try fetch(r, c.code1, code_len);
        const ref_scale0 = try fetch(r, c.scale0, scale_len);
        const ref_scale1 = try fetch(r, c.scale1, scale_len);
        try wipe(r, c.code0, code_len);
        try wipe(r, c.code1, code_len);
        try wipe(r, c.scale0, scale_len);
        try wipe(r, c.scale1, scale_len);
        try launchRotW8sDd(c);
        {
            const got_code0 = try fetch(r, c.code0, code_len);
            defer r.gpu.gpa.free(got_code0);
            if (rotddCompare(c.rows, 0, "codes", D, ref_code0, got_code0)) failed = true;
        }
        {
            const got_code1 = try fetch(r, c.code1, code_len);
            defer r.gpu.gpa.free(got_code1);
            if (rotddCompare(c.rows, 1, "codes", D, ref_code1, got_code1)) failed = true;
        }
        {
            const got_scale0 = try fetch(r, c.scale0, scale_len);
            defer r.gpu.gpa.free(got_scale0);
            if (rotddCompare(c.rows, 0, "scale", 4, ref_scale0, got_scale0)) failed = true;
        }
        {
            const got_scale1 = try fetch(r, c.scale1, scale_len);
            defer r.gpu.gpa.free(got_scale1);
            if (rotddCompare(c.rows, 1, "scale", 4, ref_scale1, got_scale1)) failed = true;
        }
        r.gpu.gpa.free(ref_code0);
        r.gpu.gpa.free(ref_code1);
        r.gpu.gpa.free(ref_scale0);
        r.gpu.gpa.free(ref_scale1);
        if (!failed) std.debug.print("INFO glm53-ra8 rows={d} rotdd byte=equal pairs={d}\n", .{ c.rows, pairs });
    }
    try wipe(r, c.sc.xd, xd_len);
    try launchRotW8s(c);
    try launchGuW8sPf(c);
    const got_pf = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(got_pf);
    try wipe(r, c.sc.xd, xd_len);
    try launchRotW8s(c);
    try launchGuW8sPf(c);
    const again_pf = try fetchHalves(r, c.sc.xd, xd_len);
    defer r.gpu.gpa.free(again_pf);
    if (firstMismatch(set.rows, got_pf, got_s)) |at| {
        std.debug.print("FAIL glm53-ra8 rows={d} w8s_pf differs at {d}\n", .{ c.rows, at });
        failed = true;
    } else if (firstMismatch(set.rows, again_pf, got_s)) |at| {
        std.debug.print("FAIL glm53-ra8 rows={d} w8s_pf differs at {d}\n", .{ c.rows, at });
        failed = true;
    } else {
        std.debug.print("INFO glm53-ra8 rows={d} w8s_pf byte=equal\n", .{c.rows});
    }
    const us_rot = try timed(c, ev, .rot_in);
    const us_gu = try timed(c, ev, .gu);
    const us_rot_a8 = try timed(c, ev, .rot_a8);
    const us_gu_a8 = try timed(c, ev, .gu_a8);
    const us_gu_w8 = try timed(c, ev, .gu_w8);
    const release_us = us_rot + us_gu;
    const a8_us = us_rot_a8 + us_gu_a8;
    const w8_us = us_rot_a8 + us_gu_w8;
    const a8_speed = release_us / a8_us;
    const w8_speed = release_us / w8_us;
    std.debug.print("BENCH glm53-ra8 rows={d} | rot_in {d:.1} + gu {d:.1} = {d:.1} us | a8 {d:.1} + {d:.1} = {d:.1} us {d:.2}x | w8 {d:.1} + {d:.1} = {d:.1} us {d:.2}x\n", .{ c.rows, us_rot, us_gu, release_us, us_rot_a8, us_gu_a8, a8_us, a8_speed, us_rot_a8, us_gu_w8, w8_us, w8_speed });
    // The w8s and rot_dd runs come last: they overwrite the code and scale buffers the a8 and w8 kernels read.
    const us_rot_w8s = try timed(c, ev, .rot_w8s);
    const us_rot_dd = try timed(c, ev, .rot_dd);
    std.debug.print("BENCH glm53-rotdd rows={d} | rot_in_w8s {d:.1} | rot_dd {d:.1} us {d:.3}x\n", .{ c.rows, us_rot_w8s, us_rot_dd, us_rot_w8s / us_rot_dd });
    const us_gu_w8s = try timed(c, ev, .gu_w8s);
    const w8s_us = us_rot_w8s + us_gu_w8s;
    const w8s_speed = release_us / w8s_us;
    std.debug.print("BENCH glm53-ra8 rows={d} | w8s {d:.1} + {d:.1} = {d:.1} us {d:.2}x of release {d:.1} us\n", .{ c.rows, us_rot_w8s, us_gu_w8s, w8s_us, w8s_speed, release_us });
    const us_gu_w8s_pf = try timed(c, ev, .gu_w8s_pf);
    std.debug.print("BENCH glm53-ra8 rows={d} | gu_w8s {d:.1} | gu_w8s_pf {d:.1} us {d:.3}x\n", .{ c.rows, us_gu_w8s, us_gu_w8s_pf, us_gu_w8s / us_gu_w8s_pf });
    const us_gu_w8s_m = try timed(c, ev, .gu_w8s_m);
    std.debug.print("BENCH glm53-gum rows={d} | gu_w8s {d:.1} | gu_m {d:.1} us {d:.3}x\n", .{ c.rows, us_gu_w8s, us_gu_w8s_m, us_gu_w8s / us_gu_w8s_m });
    if (try downCheck(r, c, ev, set.rows)) failed = true;
    var slow = false;
    if (c.rows == 4096 and w8s_speed < 1.25) {
        std.debug.print("INFO glm53-ra8 rows={d} w8s kill under 1.25x\n", .{c.rows});
        slow = true;
    }
    return .{ .failed = failed, .slow = slow };
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.exl3_experts.len == 0 or cuda.kernels.glm53_pmx.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0xA86144);
    const rig = try arena.allocator().create(Rig);
    rig.* = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const weights = try buildWeights(rig, wbench);
    const layer = layerOf(weights, .mcg_e4m3, 0, 0);
    const scratch_bytes = try exl3e.prompt.bytes(layer, ROWS_MAX, SLOTS, false);
    const scratch_ptr = try rig.zeros(scratch_bytes);
    const sc = try exl3e.prompt.scratch(scratch_ptr, layer, ROWS_MAX, SLOTS, false);
    const x_host = try rig.bfs(ROWS_MAX * D, 0.5);
    const x = try rig.dev(std.mem.sliceAsBytes(x_host));
    const code_bytes = PAIRS_MAX * D;
    const scale_bytes = PAIRS_MAX * (D / 128) * 4;
    const code0 = try rig.zeros(code_bytes);
    const code1 = try rig.zeros(code_bytes);
    const scale0 = try rig.zeros(scale_bytes);
    const scale1 = try rig.zeros(scale_bytes);
    const route_plan = try planOf(rig);
    var ev: [2]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    var failed = false;
    var slow = false;
    const row_counts = [_]usize{ 2048, 4096 };
    for (row_counts) |rows| {
        const host_pick = try benchPicks(rig.a, rows);
        const call = Call{
            .exp = &exp,
            .pmx = &pmx,
            .stream = rig.s,
            .layer = layer,
            .x = x,
            .picks = try rig.dev(std.mem.sliceAsBytes(host_pick)),
            .rows = rows,
            .sc = sc,
            .route_plan = route_plan,
            .code0 = code0,
            .code1 = code1,
            .scale0 = scale0,
            .scale1 = scale1,
        };
        const outcome = try oneRow(rig, &call, &ev);
        failed = failed or outcome.failed;
        slow = slow or outcome.slow;
    }
    if (failed) return error.Ra8Check;
    if (slow) return error.Ra8Slow;
}

/// Timing only: rot_w8s runs once for each row set, then the full gu_w8s and every KO ablation.
/// The KO variants write wrong output by design, so there is no correctness check here.
pub fn runKo(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.exl3_experts.len == 0 or cuda.kernels.glm53_pmx.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x4B4F);
    const rig = try arena.allocator().create(Rig);
    rig.* = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const weights = try buildWeights(rig, wbench);
    const layer = layerOf(weights, .mcg_e4m3, 0, 0);
    const scratch_bytes = try exl3e.prompt.bytes(layer, ROWS_MAX, SLOTS, false);
    const scratch_ptr = try rig.zeros(scratch_bytes);
    const sc = try exl3e.prompt.scratch(scratch_ptr, layer, ROWS_MAX, SLOTS, false);
    const x_host = try rig.bfs(ROWS_MAX * D, 0.5);
    const x = try rig.dev(std.mem.sliceAsBytes(x_host));
    const code_bytes = PAIRS_MAX * D;
    const scale_bytes = PAIRS_MAX * (D / 128) * 4;
    const code0 = try rig.zeros(code_bytes);
    const code1 = try rig.zeros(code_bytes);
    const scale0 = try rig.zeros(scale_bytes);
    const scale1 = try rig.zeros(scale_bytes);
    const route_plan = try planOf(rig);
    var ev: [2]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    const row_counts = [_]usize{ 2048, 4096 };
    for (row_counts) |rows| {
        const host_pick = try benchPicks(rig.a, rows);
        const call = Call{
            .exp = &exp,
            .pmx = &pmx,
            .stream = rig.s,
            .layer = layer,
            .x = x,
            .picks = try rig.dev(std.mem.sliceAsBytes(host_pick)),
            .rows = rows,
            .sc = sc,
            .route_plan = route_plan,
            .code0 = code0,
            .code1 = code1,
            .scale0 = scale0,
            .scale1 = scale1,
        };
        const pairs = rows * SLOTS;
        try exp.router.route(rig.s, call.picks, pairs, E, exl3e.prompt.wide_tile, route_plan);
        try launchRotW8s(&call);
        const full = try timedFun(&call, &ev, call.pmx.gu_w8s);
        const nodecode = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[0]);
        const nomma = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[1]);
        const noload = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[2]);
        const noguard = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[3]);
        const noepi = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[4]);
        const nodecode_nomma = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[5]);
        const nodecode_noguard = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[6]);
        const noldsm = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[7]);
        const noldsm_nodecode_nomma = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[8]);
        const noldsm_nodecode_nomma_noload = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[9]);
        const nobar = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[10]);
        const loop_epi = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[11]);
        const loop_only = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[12]);
        const noloopaddr = try timedFun(&call, &ev, call.pmx.gu_w8s_ko[13]);
        const empty_grid = try timedFun(&call, &ev, call.pmx.gu_w8s_empty);
        std.debug.print("BENCH glm53-ra8-ko rows={d} | full {d:.1} | ko1 nodecode {d:.1} | ko2 nomma {d:.1} | ko4 noload {d:.1} | ko8 noguard {d:.1} | ko16 noepi {d:.1} | ko3 nodecode+nomma {d:.1} | ko9 nodecode+noguard {d:.1} | ko32 noldsm {d:.1} | ko35 noldsm+nodecode+nomma {d:.1} | ko39 noldsm+nodecode+nomma+noload {d:.1} | ko64 nobar {d:.1} | ko103 noldsm+nodecode+nomma+noload+nobar {d:.1} | ko119 ko103+noepi {d:.1} | ko128 noloopaddr {d:.1} | empty {d:.1} us\n", .{ rows, full, nodecode, nomma, noload, noguard, noepi, nodecode_nomma, nodecode_noguard, noldsm, noldsm_nodecode_nomma, noldsm_nodecode_nomma_noload, nobar, loop_epi, loop_only, noloopaddr, empty_grid });
    }
}

/// Timing only: the release rotation and gate/up run once to fill fp16 xd, then xd_w8s, the full
/// down_w8s, every down KO ablation and the p5a small-item down (full, no-epilogue, epi and epi_sw). No checks:
/// the KO variants write wrong output by design.
pub fn runDnKo(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.exl3_experts.len == 0 or cuda.kernels.glm53_pmx.len == 0 or cuda.kernels.glm53_p5a.len == 0) return error.KernelsNotEmbedded;
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();
    var pmx = try exl3e.pmx.Pmx.load(gpu.d);
    defer pmx.unload();
    var small = try exl3e.p5a.P5a.load(gpu.d);
    defer small.unload();
    const p5a_noepi = try small.mod.function("tf_glm53_p5a_down_e4m3_noepi");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x444E4B4F);
    const rig = try arena.allocator().create(Rig);
    rig.* = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const weights = try buildWeights(rig, wbench);
    const layer = layerOf(weights, .mcg_e4m3, 0, 0);
    const scratch_bytes = try exl3e.prompt.bytes(layer, ROWS_MAX, SLOTS, false);
    const scratch_ptr = try rig.zeros(scratch_bytes);
    const sc = try exl3e.prompt.scratch(scratch_ptr, layer, ROWS_MAX, SLOTS, false);
    const x_host = try rig.bfs(ROWS_MAX * D, 0.5);
    const x = try rig.dev(std.mem.sliceAsBytes(x_host));
    const route_plan = try planOf(rig);
    var ev: [2]cuda.Event = undefined;
    for (&ev) |*event| event.* = try cuda.Event.init(gpu.d, true);
    defer {
        for (&ev) |*event| event.deinit();
    }
    const row_counts = [_]usize{ 2048, 4096 };
    var failed = false;
    for (row_counts) |rows| {
        const host_pick = try benchPicks(rig.a, rows);
        const call = Call{
            .exp = &exp,
            .pmx = &pmx,
            .stream = rig.s,
            .layer = layer,
            .x = x,
            .picks = try rig.dev(std.mem.sliceAsBytes(host_pick)),
            .rows = rows,
            .sc = sc,
            .route_plan = route_plan,
        };
        const pairs = rows * SLOTS;
        try exp.router.route(rig.s, call.picks, pairs, E, exl3e.prompt.wide_tile, route_plan);
        const live = try downLive(rig, route_plan, pairs);
        // The gate/up path fills fp16 xd: release rotation then release PMX gate/up, once.
        try launchRot(&call);
        try launchGu(&call);
        // xd_w8s first: it writes the e4m3 codes and scales the down kernels read from sc.xg.
        const us_xd = try timed(&call, &ev, .xd_w8s);
        const us_full = try timedDownFun(&call, &ev, call.pmx.down_w8s);
        const ko1 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[0]);
        const ko2 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[1]);
        const ko4 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[2]);
        const ko16 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[3]);
        const ko32 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[4]);
        const ko64 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[5]);
        const ko35 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[6]);
        const ko39 = try timedDownFun(&call, &ev, call.pmx.down_w8s_ko[7]);
        const us_empty = try timedDownFun(&call, &ev, call.pmx.down_w8s_empty);
        const us_p5a = try timedP5aDown(&call, &ev, small.down[1]);
        const us_p5a_noepi = try timedP5aDown(&call, &ev, p5a_noepi);
        const us_epi = try timedDownFun(&call, &ev, call.pmx.down_w8s_epi);
        const us_p5a_epi = try timedP5aDown(&call, &ev, small.down_epi);
        const us_p5a_epi_sw = try timedP5aDown(&call, &ev, small.down_epi_sw);
        std.debug.print("BENCH glm53-dn-ko rows={d} | xd {d:.1} | full {d:.1} | ko1 nodecode {d:.1} | ko2 nomma {d:.1} | ko4 noload {d:.1} | ko16 noepi {d:.1} | ko32 noldsm {d:.1} | ko64 nobar {d:.1} | ko35 nodecode+nomma+noldsm {d:.1} | ko39 nodecode+nomma+noload+noldsm {d:.1} | empty {d:.1} | p5a {d:.1} | p5a noepi {d:.1} us\n", .{ rows, us_xd, us_full, ko1, ko2, ko4, ko16, ko32, ko64, ko35, ko39, us_empty, us_p5a, us_p5a_noepi });
        std.debug.print("BENCH glm53-dn-epi rows={d} | down {d:.1} | down_epi {d:.1} | p5a {d:.1} | p5a_epi {d:.1} us\n", .{ rows, us_full, us_epi, us_p5a, us_p5a_epi });
        std.debug.print("BENCH glm53-p5sw rows={d} | p5a_epi {d:.1} | p5a_epi_sw {d:.1} us\n", .{ rows, us_p5a_epi, us_p5a_epi_sw });
        const y_len = pairs * D * 4;
        var row_failed = false;
        {
            try wipe(rig, call.sc.y, y_len);
            try launchXdW8s(&call);
            try launchDownW8sFun(&call, call.pmx.down_w8s);
            const y_down = try fetch(rig, call.sc.y, y_len);
            defer rig.gpu.gpa.free(y_down);
            try wipe(rig, call.sc.y, y_len);
            try launchXdW8s(&call);
            try launchDownW8sFun(&call, call.pmx.down_w8s_epi);
            const y_epi = try fetch(rig, call.sc.y, y_len);
            defer rig.gpu.gpa.free(y_epi);
            if (firstDownMismatch(live.large, y_down, y_epi, D)) |m| {
                std.debug.print("FAIL glm53-dn-epi rows={d} down_w8s pair={d} col={d}\n", .{ rows, m.pair, m.col });
                row_failed = true;
            }
        }
        {
            try wipe(rig, call.sc.y, y_len);
            try launchP5aDownFun(&call, small.down[1]);
            const y_p5a = try fetch(rig, call.sc.y, y_len);
            defer rig.gpu.gpa.free(y_p5a);
            try wipe(rig, call.sc.y, y_len);
            try launchP5aDownFun(&call, small.down_epi);
            const y_p5a_epi = try fetch(rig, call.sc.y, y_len);
            defer rig.gpu.gpa.free(y_p5a_epi);
            if (firstDownMismatch(live.small, y_p5a, y_p5a_epi, D)) |m| {
                std.debug.print("FAIL glm53-dn-epi rows={d} p5a pair={d} col={d}\n", .{ rows, m.pair, m.col });
                row_failed = true;
            }
            try wipe(rig, call.sc.y, y_len);
            try launchP5aDownFun(&call, small.down_epi_sw);
            const y_p5a_epi_sw = try fetch(rig, call.sc.y, y_len);
            defer rig.gpu.gpa.free(y_p5a_epi_sw);
            if (firstDownMismatch(live.small, y_p5a_epi, y_p5a_epi_sw, D)) |m| {
                std.debug.print("FAIL glm53-p5sw rows={d} pair={d} col={d}\n", .{ rows, m.pair, m.col });
                row_failed = true;
            }
        }
        if (!row_failed) std.debug.print("INFO glm53-dn-epi rows={d} large_pairs={d} small_pairs={d} byte=equal\n", .{ rows, live.large.len, live.small.len });
        failed = failed or row_failed;
    }
    if (failed) return error.Ra8Mismatch;
}
