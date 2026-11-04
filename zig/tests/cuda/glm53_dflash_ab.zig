// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! DFlash2 GLM kernel parity gate. Run on Spark with the glm_dflash CUDA image.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

fn bf(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn bfToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
fn launch(f: cuda.Function, grid: cuda.Dim3, threads: u32, s: cuda.Stream, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}
fn compare(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, base: cuda.Function, glm: cuda.Function, grid: cuda.Dim3, threads: u32, base_args: cuda.Args, glm_args: cuda.Args, out0: cuda.DeviceBuffer, out1: cuda.DeviceBuffer, what: []const u8) !void {
    var x = base_args;
    var y = glm_args;
    try launch(base, grid, threads, s, &x);
    try launch(glm, grid, threads, s, &y);
    try s.synchronize();
    const left = try a.alloc(u8, out0.len);
    defer a.free(left);
    const right = try a.alloc(u8, out1.len);
    defer a.free(right);
    try out0.download(0, left);
    try out1.download(0, right);
    try check.sameBytes(what, left, right);
    _ = d;
}
fn randomWords(a: std.mem.Allocator, n: usize, seed: u64) ![]u16 {
    const out = try a.alloc(u16, n);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (out, 0..) |*v, i| v.* = bf((@as(f32, @floatFromInt(@as(i32, @intCast(r.intRangeAtMost(i32, -128, 128))))) / 512) + @as(f32, @floatFromInt(i % 13)) / 2048);
    return out;
}
fn deviceWords(gpu: check.Gpu, a: std.mem.Allocator, n: usize, seed: u64) !cuda.DeviceBuffer {
    const host = try randomWords(a, n, seed);
    defer a.free(host);
    return cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}
fn zeroPair(gpu: check.Gpu, n: usize) ![2]cuda.DeviceBuffer {
    const host = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(host);
    @memset(host, 0);
    var a = try cuda.DeviceBuffer.fromHost(gpu.d, host);
    errdefer a.free();
    var b = try cuda.DeviceBuffer.fromHost(gpu.d, host);
    errdefer b.free();
    return .{ a, b };
}
fn f32Pair(gpu: check.Gpu, a: std.mem.Allocator, n: usize, seed: u64) ![3]cuda.DeviceBuffer {
    const host = try a.alloc(f32, n);
    defer a.free(host);
    var prng = std.Random.DefaultPrng.init(seed);
    for (host) |*v| v.* = @as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -20, 20))) / 64;
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    errdefer input.free();
    var out0 = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer out0.free();
    var out1 = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer out1.free();
    return .{ input, out0, out1 };
}

/// Nine ordinary `_glm` kernels against their Flash twins at the Flash geometry.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_dflash);
    defer module.unload();
    const ctx_h8 = try module.function("df_ctx_kv_h8_glm");
    const pairs = [_][2][:0]const u8{
        .{ "df_tap", "df_tap_glm" },   .{ "df_block_qkv", "df_block_qkv_glm" },   .{ "df_ctx_kv", "df_ctx_kv_glm" },
        .{ "df_conv", "df_conv_glm" }, .{ "df_attn_part", "df_attn_part_glm" },   .{ "df_attn_merge", "df_attn_merge_glm" },
        .{ "df_sum2", "df_sum2_glm" }, .{ "df_top16_part", "df_top16_part_glm" }, .{ "df_top16_merge", "df_top16_merge_glm" },
    };
    var f: [pairs.len][2]cuda.Function = undefined;
    for (pairs, 0..) |p, i| {
        f[i] = .{ try module.function(p[0]), try module.function(p[1]) };
    }
    const R: usize = 8;
    const D: usize = 4096;
    const QH: usize = 32;
    const KVH: usize = 8;
    const HD: usize = 128;
    const KVW: usize = KVH * HD;
    const CH: usize = 16;

    // Tap: the GLM generalization in mode 0 must preserve Flash stream averaging.
    var tap_x = try deviceWords(gpu, a, R * 4 * D, 10);
    defer tap_x.free();
    var tap_outs = try zeroPair(gpu, R * 5 * D * 2);
    defer tap_outs[0].free();
    defer tap_outs[1].free();
    var ta: cuda.Args = .{};
    ta.add(tap_x.ptr);
    ta.add(tap_outs[0].ptr);
    ta.add(@as(i32, 2));
    ta.add(@as(i32, 0));
    var tb: cuda.Args = .{};
    tb.add(tap_x.ptr);
    tb.add(tap_outs[1].ptr);
    tb.add(@as(i32, 2));
    tb.add(@as(i32, 0));
    tb.add(@as(i32, D));
    tb.add(@as(i32, 5));
    try compare(a, gpu.d, s, f[0][0], f[0][1], .{ .x = R, .y = D / 1024 }, 256, ta, tb, tap_outs[0], tap_outs[1], "df_tap Flash vs GLM");
    ta = .{};
    ta.add(tap_x.ptr);
    ta.add(tap_outs[0].ptr);
    ta.add(@as(i32, 2));
    ta.add(@as(i32, 1));
    tb = .{};
    tb.add(tap_x.ptr);
    tb.add(tap_outs[1].ptr);
    tb.add(@as(i32, 2));
    tb.add(@as(i32, 1));
    tb.add(@as(i32, D));
    tb.add(@as(i32, 5));
    try compare(a, gpu.d, s, f[0][0], f[0][1], .{ .x = R, .y = D / 1024 }, 256, ta, tb, tap_outs[0], tap_outs[1], "df_tap plain-copy Flash vs GLM");

    // Block QKV: q/k norm and interleaved RoPE with identical runtime values.
    var qkv = try deviceWords(gpu, a, R * (QH + 2 * KVH) * HD, 11);
    defer qkv.free();
    var qn = try deviceWords(gpu, a, HD, 12);
    defer qn.free();
    var kn = try deviceWords(gpu, a, HD, 13);
    defer kn.free();
    const pos_h = [_]i32{ 0, 1, 127, 511, 1023, 1537, 2046, 2047 };
    var pos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&pos_h));
    defer pos.free();
    var qout = try zeroPair(gpu, R * QH * HD * 2);
    defer qout[0].free();
    defer qout[1].free();
    var kout = try zeroPair(gpu, R * KVW * 2);
    defer kout[0].free();
    defer kout[1].free();
    var vout = try zeroPair(gpu, R * KVW * 2);
    defer vout[0].free();
    defer vout[1].free();
    var qa: cuda.Args = .{};
    qa.add(qkv.ptr);
    qa.add(@as(i32, (QH + 2 * KVH) * HD));
    qa.add(qn.ptr);
    qa.add(kn.ptr);
    qa.add(pos.ptr);
    qa.add(@as(f32, 1e-5));
    qa.add(@as(f32, 10000));
    qa.add(qout[0].ptr);
    qa.add(kout[0].ptr);
    qa.add(vout[0].ptr);
    var qb: cuda.Args = .{};
    qb.add(qkv.ptr);
    qb.add(@as(i32, (QH + 2 * KVH) * HD));
    qb.add(qn.ptr);
    qb.add(kn.ptr);
    qb.add(pos.ptr);
    qb.add(@as(f32, 1e-5));
    qb.add(@as(f32, 10000));
    qb.add(@as(i32, QH));
    qb.add(@as(i32, KVH));
    qb.add(@as(i32, HD));
    qb.add(@as(i32, KVW));
    qb.add(qout[1].ptr);
    qb.add(kout[1].ptr);
    qb.add(vout[1].ptr);
    try compare(a, gpu.d, s, f[1][0], f[1][1], .{ .x = R, .y = QH + KVH }, 64, qa, qb, qout[0], qout[1], "df_block_qkv q Flash vs GLM");
    try sameDevice(a, kout[0], kout[1], "df_block_qkv k Flash vs GLM");
    try sameDevice(a, vout[0], vout[1], "df_block_qkv v Flash vs GLM");

    // Context KV writes and chunked attention partials/merge at Flash shape.
    const ring_bytes = 2 * 2048 * KVW * 2;
    var ring0 = try cuda.DeviceBuffer.alloc(gpu.d, ring_bytes);
    defer ring0.free();
    var ring1 = try cuda.DeviceBuffer.alloc(gpu.d, ring_bytes);
    defer ring1.free();
    try ring0.fill8(0, s.handle);
    try ring1.fill8(0, s.handle);
    var kv_in = try deviceWords(gpu, a, R * 2 * KVW, 20);
    defer kv_in.free();
    const pos64_h = [_]i64{ 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052 };
    var pos64 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&pos64_h));
    defer pos64.free();
    var rings0_h: [R]u64 = @splat(ring0.ptr);
    var rings1_h: [R]u64 = @splat(ring1.ptr);
    var rings0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&rings0_h));
    defer rings0.free();
    var rings1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&rings1_h));
    defer rings1.free();
    var ctx_a: cuda.Args = .{};
    ctx_a.add(kv_in.ptr);
    ctx_a.add(kn.ptr);
    ctx_a.add(pos64.ptr);
    ctx_a.add(rings0.ptr);
    ctx_a.add(@as(i32, 0));
    ctx_a.add(@as(f32, 1e-5));
    ctx_a.add(@as(f32, 10000));
    var ctx_b: cuda.Args = .{};
    ctx_b.add(kv_in.ptr);
    ctx_b.add(kn.ptr);
    ctx_b.add(pos64.ptr);
    ctx_b.add(rings1.ptr);
    ctx_b.add(@as(i32, 0));
    ctx_b.add(@as(f32, 1e-5));
    ctx_b.add(@as(f32, 10000));
    ctx_b.add(@as(i32, KVH));
    ctx_b.add(@as(i32, HD));
    ctx_b.add(@as(i32, KVW));
    ctx_b.add(@as(i32, 2048));
    try launch(f[2][0], .{ .x = R, .y = KVH }, 64, s, &ctx_a);
    try launch(f[2][1], .{ .x = R, .y = KVH }, 64, s, &ctx_b);
    try s.synchronize();
    try sameDevice(a, ring0, ring1, "df_ctx_kv Flash vs GLM");
    var ring_h8 = try cuda.DeviceBuffer.alloc(gpu.d, ring_bytes);
    defer ring_h8.free();
    try ring_h8.fill8(0, s.handle);
    var rings_h8_host: [R]u64 = @splat(ring_h8.ptr);
    var rings_h8 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&rings_h8_host));
    defer rings_h8.free();
    var ctx_h8_args: cuda.Args = .{};
    ctx_h8_args.add(kv_in.ptr);
    ctx_h8_args.add(kn.ptr);
    ctx_h8_args.add(pos64.ptr);
    ctx_h8_args.add(rings_h8.ptr);
    ctx_h8_args.add(@as(i32, 0));
    ctx_h8_args.add(@as(f32, 1e-5));
    ctx_h8_args.add(@as(f32, 10000));
    ctx_h8_args.add(@as(i32, KVH));
    ctx_h8_args.add(@as(i32, HD));
    ctx_h8_args.add(@as(i32, KVW));
    ctx_h8_args.add(@as(i32, 2048));
    try launch(ctx_h8, .{ .x = R, .y = KVH }, 64, s, &ctx_h8_args);
    try s.synchronize();
    try sameDevice(a, ring0, ring_h8, "df_ctx_kv_h8_glm Flash-shape twin");

    var starts_h = [_]i64{2047};
    var froms_h = [_]i64{2045};
    var starts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&starts_h));
    defer starts.free();
    var froms = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&froms_h));
    defer froms.free();
    const pm_n = R * QH * CH;
    const pacc_n = pm_n * HD;
    var p0 = try zeroPair(gpu, pm_n * 4);
    defer p0[0].free();
    defer p0[1].free();
    var l0 = try zeroPair(gpu, pm_n * 4);
    defer l0[0].free();
    defer l0[1].free();
    var acc0 = try zeroPair(gpu, pacc_n * 4);
    defer acc0[0].free();
    defer acc0[1].free();
    var attn0 = try zeroPair(gpu, R * QH * HD * 2);
    defer attn0[0].free();
    defer attn0[1].free();
    var aa: cuda.Args = .{};
    aa.add(qout[0].ptr);
    aa.add(kout[0].ptr);
    aa.add(vout[0].ptr);
    aa.add(rings0.ptr);
    aa.add(starts.ptr);
    aa.add(froms.ptr);
    aa.add(@as(i32, 0));
    aa.add(@as(f32, 1.0 / @sqrt(128.0)));
    aa.add(p0[0].ptr);
    aa.add(l0[0].ptr);
    aa.add(acc0[0].ptr);
    var ab: cuda.Args = .{};
    ab.add(qout[1].ptr);
    ab.add(kout[1].ptr);
    ab.add(vout[1].ptr);
    ab.add(rings1.ptr);
    ab.add(starts.ptr);
    ab.add(froms.ptr);
    ab.add(@as(i32, 0));
    ab.add(@as(f32, 1.0 / @sqrt(128.0)));
    ab.add(@as(i32, 2048));
    ab.add(@as(i32, 8));
    ab.add(@as(i32, QH));
    ab.add(@as(i32, KVH));
    ab.add(@as(i32, HD));
    ab.add(@as(i32, KVW));
    ab.add(p0[1].ptr);
    ab.add(l0[1].ptr);
    ab.add(acc0[1].ptr);
    try launch(f[4][0], .{ .x = 1, .y = KVH, .z = CH }, 256, s, &aa);
    try launch(f[4][1], .{ .x = 1, .y = KVH, .z = CH }, 256, s, &ab);
    try s.synchronize();
    try sameDevice(a, p0[0], p0[1], "df_attn_part max Flash vs GLM");
    try sameDevice(a, l0[0], l0[1], "df_attn_part sum Flash vs GLM");
    try sameDevice(a, acc0[0], acc0[1], "df_attn_part accumulator Flash vs GLM");
    var am: cuda.Args = .{};
    am.add(p0[0].ptr);
    am.add(l0[0].ptr);
    am.add(acc0[0].ptr);
    am.add(attn0[0].ptr);
    var bm: cuda.Args = .{};
    bm.add(p0[1].ptr);
    bm.add(l0[1].ptr);
    bm.add(acc0[1].ptr);
    bm.add(attn0[1].ptr);
    bm.add(@as(i32, QH));
    bm.add(@as(i32, HD));
    try compare(a, gpu.d, s, f[5][0], f[5][1], .{ .x = R, .y = QH }, 128, am, bm, attn0[0], attn0[1], "df_attn_merge Flash vs GLM");

    // Convolution, including add mode and the last tile boundary.
    var z = try deviceWords(gpu, a, R * D, 14);
    defer z.free();
    var dyn = try deviceWords(gpu, a, R * 4 * (D / 16), 15);
    defer dyn.free();
    var base = try deviceWords(gpu, a, 2 * D, 16);
    defer base.free();
    var conv_out = try zeroPair(gpu, R * D * 2);
    defer conv_out[0].free();
    defer conv_out[1].free();
    var ca: cuda.Args = .{};
    ca.add(z.ptr);
    ca.add(dyn.ptr);
    ca.add(@as(i32, 1));
    ca.add(base.ptr);
    ca.add(conv_out[0].ptr);
    ca.add(@as(i32, 1));
    var cb: cuda.Args = .{};
    cb.add(z.ptr);
    cb.add(dyn.ptr);
    cb.add(@as(i32, 1));
    cb.add(base.ptr);
    cb.add(conv_out[1].ptr);
    cb.add(@as(i32, 1));
    cb.add(@as(i32, D));
    cb.add(@as(i32, 16));
    cb.add(@as(i32, 2));
    cb.add(@as(i32, 8));
    try compare(a, gpu.d, s, f[3][0], f[3][1], .{ .x = R, .y = D / 1024 }, 256, ca, cb, conv_out[0], conv_out[1], "df_conv Flash vs GLM");

    // Split-K sum, with fp32 input values and BF16 output.
    var sum = try f32Pair(gpu, a, 2 * R * D, 17);
    defer sum[0].free();
    defer sum[1].free();
    defer sum[2].free();
    var sa: cuda.Args = .{};
    sa.add(sum[0].ptr);
    sa.add(sum[1].ptr);
    sa.add(@as(i32, R));
    var sb: cuda.Args = .{};
    sb.add(sum[0].ptr);
    sb.add(sum[2].ptr);
    sb.add(@as(i32, R));
    sb.add(@as(i32, D));
    try compare(a, gpu.d, s, f[6][0], f[6][1], .{ .x = R, .y = D / 1024 }, 256, sa, sb, sum[1], sum[2], "df_sum2 Flash vs GLM");

    // Top-16 part/merge, preserving the lower-id tie behavior with repeated BF16 logits.
    const cols: usize = 6144;
    var logits = try deviceWords(gpu, a, R * cols, 18);
    defer logits.free();
    const chunks = (cols + 4095) / 4096;
    var pv = try zeroPair(gpu, R * chunks * 16 * 4);
    defer pv[0].free();
    defer pv[1].free();
    var pi = try zeroPair(gpu, R * chunks * 16 * 4);
    defer pi[0].free();
    defer pi[1].free();
    var pa: cuda.Args = .{};
    pa.add(logits.ptr);
    pa.add(@as(i32, cols));
    pa.add(@as(i32, 77440));
    pa.add(pv[0].ptr);
    pa.add(pi[0].ptr);
    var pb: cuda.Args = .{};
    pb.add(logits.ptr);
    pb.add(@as(i32, cols));
    pb.add(@as(i32, 77440));
    pb.add(@as(i32, 16));
    pb.add(pv[1].ptr);
    pb.add(pi[1].ptr);
    try compare(a, gpu.d, s, f[7][0], f[7][1], .{ .x = R, .y = chunks }, 256, pa, pb, pv[0], pv[1], "df_top16_part values Flash vs GLM");
    try sameDevice(a, pi[0], pi[1], "df_top16_part ids Flash vs GLM");
    var merge = try zeroPair(gpu, R * 16 * 4);
    defer merge[0].free();
    defer merge[1].free();
    var ma: cuda.Args = .{};
    ma.add(pv[0].ptr);
    ma.add(pi[0].ptr);
    ma.add(@as(i32, chunks));
    ma.add(merge[0].ptr);
    var mb: cuda.Args = .{};
    mb.add(pv[1].ptr);
    mb.add(pi[1].ptr);
    mb.add(@as(i32, chunks));
    mb.add(@as(i32, 16));
    mb.add(merge[1].ptr);
    try compare(a, gpu.d, s, f[8][0], f[8][1], .{ .x = R }, 256, ma, mb, merge[0], merge[1], "df_top16_merge Flash vs GLM");

    try runH8Oracle(gpu, s, module);
    _ = gpu.io;
    check.pass("GLM DFlash Flash-shape parity: all nine generalized kernels, with fp64 h8 attention oracle reserved for GLM shape", .{});
}
fn runH8Oracle(gpu: check.Gpu, s: cuda.Stream, module: cuda.Module) !void {
    const a = gpu.gpa;
    const qh: usize = 64;
    const kvh: usize = 8;
    const hd: usize = 128;
    const kvw = kvh * hd;
    const rows: usize = 8;
    const ctx: usize = 2;
    const from: i64 = 2045;
    const start: i64 = 2047;
    const qh_data = try randomWords(a, rows * qh * hd, 91);
    defer a.free(qh_data);
    const kv_data = try randomWords(a, (ctx + rows) * kvh * hd, 92);
    defer a.free(kv_data);
    const value_data = try randomWords(a, kv_data.len, 93);
    defer a.free(value_data);
    const want = try a.alloc(u16, rows * qh * hd);
    defer a.free(want);
    try attentionH8Reference(qh_data, kv_data, value_data, from, start, rows, want);
    var q = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qh_data));
    defer q.free();
    var kb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(kv_data[ctx * kvw ..]));
    defer kb.free();
    var vb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(value_data[ctx * kvw ..]));
    defer vb.free();
    const ring_words = 2 * 2048 * kvw;
    const ring_h = try a.alloc(u16, ring_words);
    defer a.free(ring_h);
    @memset(ring_h, 0);
    for (0..ctx) |n| {
        const slot = @as(usize, @intCast(from)) + n;
        for (0..kvw) |d| {
            ring_h[slot * kvw + d] = kv_data[n * kvw + d];
            ring_h[2048 * kvw + slot * kvw + d] = value_data[n * kvw + d];
        }
    }
    var ring = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ring_h));
    defer ring.free();
    const rings_h = [_]u64{ring.ptr};
    var rings = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&rings_h));
    defer rings.free();
    const starts_h = [_]i64{start};
    const froms_h = [_]i64{from};
    var starts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&starts_h));
    defer starts.free();
    var froms = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&froms_h));
    defer froms.free();
    const pcount = rows * qh * 16;
    var pm = try cuda.DeviceBuffer.alloc(gpu.d, pcount * 4);
    defer pm.free();
    var pl = try cuda.DeviceBuffer.alloc(gpu.d, pcount * 4);
    defer pl.free();
    var pacc = try cuda.DeviceBuffer.alloc(gpu.d, pcount * hd * 4);
    defer pacc.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * qh * hd * 2);
    defer out.free();
    const part = try module.function("df_attn_part_h8_glm");
    const merge = try module.function("df_attn_merge_glm");
    var args: cuda.Args = .{};
    args.add(q.ptr);
    args.add(kb.ptr);
    args.add(vb.ptr);
    args.add(rings.ptr);
    args.add(starts.ptr);
    args.add(froms.ptr);
    args.add(@as(i32, 0));
    args.add(@as(f32, 1.0 / @sqrt(128.0)));
    args.add(@as(i32, 2048));
    args.add(@as(i32, rows));
    args.add(@as(i32, qh));
    args.add(@as(i32, kvh));
    args.add(@as(i32, hd));
    args.add(@as(i32, kvw));
    args.add(pm.ptr);
    args.add(pl.ptr);
    args.add(pacc.ptr);
    try launch(part, .{ .x = 1, .y = kvh, .z = 16 }, 256, s, &args);
    var ma: cuda.Args = .{};
    ma.add(pm.ptr);
    ma.add(pl.ptr);
    ma.add(pacc.ptr);
    ma.add(out.ptr);
    ma.add(@as(i32, qh));
    ma.add(@as(i32, hd));
    try launch(merge, .{ .x = rows, .y = qh }, @intCast(hd), s, &ma);
    try s.synchronize();
    const bytes = try cudaBytes(a, out);
    defer a.free(bytes);
    const got = std.mem.bytesAsSlice(u16, bytes);
    for (got, want, 0..) |g, w, i| {
        const actual = bfToF32(g);
        const expected = bfToF32(w);
        const tol = @max(@as(f32, 0.00390625), @abs(expected) * 0.02);
        if (!(@abs(actual - expected) <= tol)) {
            std.debug.print("GLM DFlash h8 attention FP64 mismatch at {d}: got {d}, expected {d}, tolerance {d}\n", .{ i, actual, expected, tol });
            return error.TestFailed;
        }
    }
    check.pass("GLM DFlash h8 attention against FP64 at GLM shape (64/8, short wrapped context)", .{});
}

fn sameDevice(a: std.mem.Allocator, x: cuda.DeviceBuffer, y: cuda.DeviceBuffer, label: []const u8) !void {
    const xb = try cudaBytes(a, x);
    defer a.free(xb);
    const yb = try cudaBytes(a, y);
    defer a.free(yb);
    try check.sameBytes(label, xb, yb);
}
fn cudaBytes(a: std.mem.Allocator, b: cuda.DeviceBuffer) ![]u8 {
    const out = try a.alloc(u8, b.len);
    errdefer a.free(out);
    try b.download(0, out);
    return out;
}

/// Host-side FP64 oracle for the h8 attention case; each row attends the short context plus block keys.
pub fn attentionH8Reference(q: []const u16, keys: []const u16, values: []const u16, from: i64, start: i64, rows: usize, out: []u16) !void {
    const q_heads = 64;
    const kv_heads = 8;
    const hd = 128;
    const ctx = @as(usize, @intCast(@max(0, start - from)));
    if (q.len < rows * q_heads * hd or keys.len < (ctx + rows) * kv_heads * hd or values.len < keys.len or out.len < rows * q_heads * hd) return error.BadAttentionOracleShape;
    const scale: f64 = 1.0 / @sqrt(128.0);
    for (0..rows) |r| for (0..q_heads) |h| {
        const kh = h / 8;
        const nkeys = ctx + rows;
        var scores = try std.heap.page_allocator.alloc(f64, nkeys);
        defer std.heap.page_allocator.free(scores);
        var max_score: f64 = -std.math.inf(f64);
        for (0..nkeys) |n| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, bfToF32(q[(r * q_heads + h) * hd + d])) * bfToF32(keys[(n * kv_heads + kh) * hd + d]);
            scores[n] = dot * scale;
            max_score = @max(max_score, scores[n]);
        }
        var denom: f64 = 0;
        for (scores) |*v| {
            v.* = @exp(v.* - max_score);
            denom += v.*;
        }
        for (0..hd) |d| {
            var acc: f64 = 0;
            for (0..nkeys) |n| acc += scores[n] * bfToF32(values[(n * kv_heads + kh) * hd + d]);
            out[(r * q_heads + h) * hd + d] = bf(@floatCast(acc / denom));
        }
    };
}

pub fn main(init: std.process.Init) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io };
    try run(gpu);
}

test "GLM DFlash AB harness declarations and FP64 attention oracle analyze" {
    std.testing.refAllDecls(@This());
}

test "GLM DFlash h8 attention FP64 oracle handles a wrapped short context" {
    const q = try std.testing.allocator.alloc(u16, 2 * 64 * 128);
    defer std.testing.allocator.free(q);
    const kv = try std.testing.allocator.alloc(u16, 5 * 8 * 128);
    defer std.testing.allocator.free(kv);
    const out = try std.testing.allocator.alloc(u16, q.len);
    defer std.testing.allocator.free(out);
    @memset(q, bf(0.125));
    @memset(kv, bf(0.25));
    try attentionH8Reference(q, kv, kv, 2045, 2047, 2, out);
    for (out) |v| try std.testing.expect(v == bf(0.25));
}
