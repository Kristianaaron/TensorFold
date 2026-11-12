// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! `tf-cuda-test glm53-iqf`: TF_GLM53_IQ_FOLD. The wq_b W8A8 projection plus tf_glm53_index_q is the reference;
//! w8a8's `_iq` entry does the same K loop and rounds each accumulator through bf16 before index_q's pair math.
//! Checks byte-equal indexer queries at rows 1 / 17 / 127 / 128 / 129 / 1024 on both GEMM tile heights.
//! BENCH: gemm + index_q vs fused, rows 1024, CUDA event timing.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fp8 = cuda.fp8;
const w8a8 = cuda.w8a8;

const heads: usize = 32;
const head_dim: usize = 128;
const n: usize = heads * head_dim;
const k: usize = 2048;
const rope_half: usize = 32;
const rows_list = [_]usize{ 1, 17, 127, 128, 129, 1024 };
const bench_rows: usize = 1024;
const rounds: usize = 10;
const warmup: usize = 2;

const Env = struct {
    gpu: check.Gpu,
    stream: cuda.Stream,
    start: cuda.Event,
    stop: cuda.Event,
    query: cuda.Function,
    weight: cuda.qmmf.Weight,
    x: u64,
    positions: u64,
    cos: u64,
    sin: u64,
    scratch: u64,
    scratch_len: usize,
    iq_ref: cuda.DeviceBuffer,
    qi_ref: cuda.DeviceBuffer,
    qi_fused: cuda.DeviceBuffer,
};

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    var kern = try w8a8.Kernels.load(gpu.d, cuda.kernels.glm53_w8a8);
    defer kern.unload();
    var kern_m64 = try w8a8.Kernels.load(gpu.d, cuda.kernels.glm53_w8a8);
    defer kern_m64.unload();
    try kern_m64.useM64();
    var index_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm53_index);
    defer index_mod.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();

    // FP8B weights through the checkpoint's packer, e4m3 codes and native block scales.
    const table = try gpa.alloc(u8, 1 << 16);
    defer gpa.free(table);
    for (table, 0..) |*code, bits| code.* = w8a8.e4m3Nearest(bf16Float(@intCast(bits)));
    var rng = std.Random.DefaultPrng.init(0x1f_0143_2026);
    const random = rng.random();
    const raw = try gpa.alloc(u8, n * k);
    defer gpa.free(raw);
    for (raw) |*code| code.* = table[bf16Bits(random.floatNorm(f32) * 96.0)];
    const block_scales = try gpa.alloc(f32, (n / 128) * (k / 128));
    defer gpa.free(block_scales);
    for (block_scales) |*s| s.* = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -14, -8));
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    fp8.packCodes(codes, raw, n, k);
    const scales = try gpa.alloc(f32, (n / fp8.group) * (k / fp8.group));
    defer gpa.free(scales);
    try fp8.packBlockScales(scales, block_scales, n, k, 0, 0, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    const weight = fp8.blockWeight(dw.ptr, ds.ptr, n, k);
    try check.expect(w8a8.supports(weight), "glm53-iqf: weight not FP8B-shaped", .{});

    // Real-shaped row inputs: one bf16 row per prompt position, ascending positions a DCP slot apart, and the
    // per-row 32 rope angles the forward uploads as ws.cos / ws.sin.
    const x = try gpa.alloc(u16, bench_rows * k);
    defer gpa.free(x);
    for (x) |*v| v.* = bf16Bits(random.floatNorm(f32) * 2.0);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    const positions = try gpa.alloc(i64, bench_rows);
    defer gpa.free(positions);
    for (positions, 0..) |*p, row| p.* = @intCast(1024 + 2 * row);
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
    defer dpos.free();
    const cos = try gpa.alloc(f32, bench_rows * rope_half);
    defer gpa.free(cos);
    const sin = try gpa.alloc(f32, bench_rows * rope_half);
    defer gpa.free(sin);
    for (0..bench_rows * rope_half) |i| {
        const angle: f32 = random.float(f32) * @as(f32, 2.0 * std.math.pi);
        cos[i] = @cos(angle);
        sin[i] = @sin(angle);
    }
    var dcos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cos));
    defer dcos.free();
    var dsin = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sin));
    defer dsin.free();

    const scratch_len = w8a8.scratchBytes(bench_rows, k);
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, scratch_len);
    defer scratch.free();
    var iq_ref = try cuda.DeviceBuffer.alloc(gpu.d, bench_rows * n * 2);
    defer iq_ref.free();
    var qi_ref = try cuda.DeviceBuffer.alloc(gpu.d, bench_rows * n * 2);
    defer qi_ref.free();
    var qi_fused = try cuda.DeviceBuffer.alloc(gpu.d, bench_rows * n * 2);
    defer qi_fused.free();

    const env: Env = .{
        .gpu = gpu,
        .stream = stream,
        .start = start,
        .stop = stop,
        .query = try index_mod.function("tf_glm53_index_q"),
        .weight = weight,
        .x = dx.ptr,
        .positions = dpos.ptr,
        .cos = dcos.ptr,
        .sin = dsin.ptr,
        .scratch = scratch.ptr,
        .scratch_len = scratch_len,
        .iq_ref = iq_ref,
        .qi_ref = qi_ref,
        .qi_fused = qi_fused,
    };
    for (rows_list) |rows| {
        try compareRows(&env, kern.fns, rows, "release");
        try compareRows(&env, kern_m64.fns, rows, "m64");
    }

    const ref_us = try timePath(&env, kern.fns, false);
    const fused_us = try timePath(&env, kern.fns, true);
    std.debug.print("BENCH glm53-iqf rows={d} | gemm+index_q {d:.1} | fused {d:.1} us\n", .{ bench_rows, ref_us, fused_us });
}

fn compareRows(env: *const Env, fns: w8a8.Fns, rows: usize, label: []const u8) !void {
    try fns.run(env.stream, env.x, k, rows, env.weight, env.iq_ref.ptr, .bf16, env.scratch, env.scratch_len);
    try launchQuery(env.stream, env.query, env.iq_ref.ptr, env.cos, env.sin, env.qi_ref.ptr, rows);
    try fns.runIq(env.stream, env.x, k, rows, env.weight, env.qi_fused.ptr, env.positions, env.cos, env.sin, env.scratch, env.scratch_len);
    try env.stream.synchronize();
    const gpa = env.gpu.gpa;
    const out_bytes = rows * n * 2;
    const ref_all = try check.download(env.gpu, env.qi_ref);
    defer gpa.free(ref_all);
    const fused_all = try check.download(env.gpu, env.qi_fused);
    defer gpa.free(fused_all);
    var what_buf: [96]u8 = undefined;
    const what = try std.fmt.bufPrint(&what_buf, "glm53-iqf rows={d} {s} fused vs gemm+index_q", .{ rows, label });
    try check.sameBytes(what, fused_all[0..out_bytes], ref_all[0..out_bytes]);
    check.pass("glm53-iqf rows={d} {s}: indexer query byte-equal ({d} bytes)", .{ rows, label, out_bytes });
}

fn launchQuery(s: cuda.Stream, query: cuda.Function, input: u64, cos: u64, sin: u64, out: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(input);
    a.add(cos);
    a.add(sin);
    a.add(out);
    try cuda.launch.launch(query, .{ .grid = .{ .x = 32, .y = @intCast(rows), .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

fn timePath(env: *const Env, fns: w8a8.Fns, fused: bool) !f64 {
    var times: [rounds]f64 = undefined;
    for (0..warmup + rounds) |i| {
        try env.start.record(env.stream);
        if (fused) {
            try fns.runIq(env.stream, env.x, k, bench_rows, env.weight, env.qi_fused.ptr, env.positions, env.cos, env.sin, env.scratch, env.scratch_len);
        } else {
            try fns.run(env.stream, env.x, k, bench_rows, env.weight, env.iq_ref.ptr, .bf16, env.scratch, env.scratch_len);
            try launchQuery(env.stream, env.query, env.iq_ref.ptr, env.cos, env.sin, env.qi_ref.ptr, bench_rows);
        }
        try env.stop.record(env.stream);
        try env.stop.synchronize();
        if (i >= warmup) times[i - warmup] = @as(f64, try cuda.Event.elapsedMs(env.start, env.stop)) * 1000.0;
    }
    return check.median(&times);
}

fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn bf16Float(value: u16) f32 {
    return @bitCast(@as(u32, value) << 16);
}

test "test references the GLM-5.3 IQ fold report" {
    _ = &run;
    _ = &compareRows;
    _ = &timePath;
}

test {
    std.testing.refAllDecls(@This());
}
