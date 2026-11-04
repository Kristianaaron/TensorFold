// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! `tf-cuda-test glm53-w8a8`: TF_GLM53_PREFILL_W8A8's o_proj (cuda.w8a8, kernels/cuda/glm_dsa/w8a8.cu) against the
//! default FP8B qmmf lane (its fused64 plan from 256 rows) on the per-rank o_proj shape, N 6144 x K 8192, fp32 out.
//! Reported for the KLD gate: NMSE, max relative error and cosine vs the default, at 1 / 512 / 2048 rows.
//! Hard checks: the device quantizer equals the host quantizer bit for bit; sampled outputs equal an FP64 sum of the
//! same e4m3 products and scales (operand layout); a row alone equals that row inside 512 / 2048 rows bit for bit.
//! BENCH: default vs W8A8 (quantize + GEMM, and each alone), microseconds and TFLOP/s, CUDA event timing.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const fp8 = cuda.fp8;
const w8a8 = cuda.w8a8;

const n: usize = 6144;
const k: usize = 8192;
const blocks_k: usize = k / w8a8.block_k;
const rows_list = [_]usize{ 1, 512, 2048 };
const samples: usize = 256;
const rounds: usize = 10;
const warmup: usize = 2;
const layout_tolerance: f64 = 1e-3; // |W8A8 - FP64 of its own operands| / sum |terms|; a layout error gives ~1e-2
const min_cosine: f64 = 0.995; // sanity floor vs the default only; the KLD gate decides acceptance

/// The FP8B projection on the device plus its row-major codes and native block scales for the host sums.
const Weights = struct {
    raw: []u8, // e4m3 [n][k]
    block_scales: []f32, // [n/128][k/128]
    weight: cuda.qmmf.Weight,
};

const Kit = struct {
    gpu: check.Gpu,
    lane: *const cuda.qmmf.Lane,
    fns: w8a8.Fns,
    stream: cuda.Stream,
    start: cuda.Event,
    stop: cuda.Event,
    wt: Weights,
};

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var kern = try w8a8.Kernels.load(gpu.d, cuda.kernels.glm53_w8a8);
    defer kern.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();

    // Gaussian weights through e4m3 (most codes mid-range, a few saturated), block scales 2^-14 .. 2^-8.
    const table = try gpa.alloc(u8, 1 << 16);
    defer gpa.free(table);
    for (table, 0..) |*code, bits| code.* = w8a8.e4m3Nearest(bf16Float(@intCast(bits)));
    var rng = std.Random.DefaultPrng.init(0x53_a8_2026);
    const random = rng.random();
    const raw = try gpa.alloc(u8, n * k);
    defer gpa.free(raw);
    for (raw) |*code| code.* = table[bf16Bits(random.floatNorm(f32) * 96)];
    const block_scales = try gpa.alloc(f32, (n / 128) * blocks_k);
    defer gpa.free(block_scales);
    for (block_scales) |*s| s.* = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -14, -8));
    const npad = fp8.padded(n);
    const codes = try gpa.alloc(u8, npad * k);
    defer gpa.free(codes);
    fp8.packCodes(codes, raw, n, k);
    const scales = try gpa.alloc(f32, (npad / fp8.group) * (k / fp8.group));
    defer gpa.free(scales);
    try fp8.packBlockScales(scales, block_scales, n, k, 0, 0, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    const kit: Kit = .{
        .gpu = gpu,
        .lane = &lane,
        .fns = kern.fns,
        .stream = stream,
        .start = start,
        .stop = stop,
        .wt = .{ .raw = raw, .block_scales = block_scales, .weight = fp8.blockWeight(dw.ptr, ds.ptr, n, k) },
    };
    try check.expect(w8a8.supports(kit.wt.weight), "glm53-w8a8: o_proj weight not FP8B-shaped", .{});
    for (rows_list) |rows| try runRows(&kit, rows, random);
}

const Stats = struct {
    nmse: f64,
    cosine: f64,
    max_rel: f64, // over |ref| >= rms(ref)
    max_abs_norm: f64, // max |diff| / max |ref|
    worst_row_cosine: f64,
};

fn compare(got: []align(1) const f32, ref: []align(1) const f32, rows: usize) Stats {
    var diff_sq: f64 = 0;
    var ref_sq: f64 = 0;
    var got_sq: f64 = 0;
    var dot: f64 = 0;
    var ref_max: f64 = 0;
    var diff_max: f64 = 0;
    for (got, ref) |g, r| {
        const gv: f64 = g;
        const rv: f64 = r;
        diff_sq += (gv - rv) * (gv - rv);
        ref_sq += rv * rv;
        got_sq += gv * gv;
        dot += gv * rv;
        ref_max = @max(ref_max, @abs(rv));
        diff_max = @max(diff_max, @abs(gv - rv));
    }
    const rms = @sqrt(ref_sq / @as(f64, @floatFromInt(ref.len)));
    var max_rel: f64 = 0;
    for (got, ref) |g, r| {
        const rv: f64 = r;
        if (@abs(rv) >= rms) max_rel = @max(max_rel, @abs(@as(f64, g) - rv) / @abs(rv));
    }
    var worst: f64 = 1;
    for (0..rows) |row| {
        var rd: f64 = 0;
        var rg: f64 = 0;
        var rr: f64 = 0;
        for (got[row * n ..][0..n], ref[row * n ..][0..n]) |g, r| {
            rd += @as(f64, g) * r;
            rg += @as(f64, g) * g;
            rr += @as(f64, r) * r;
        }
        worst = @min(worst, rd / @sqrt(@max(rg * rr, 1e-300)));
    }
    return .{
        .nmse = diff_sq / @max(ref_sq, 1e-300),
        .cosine = dot / @sqrt(@max(got_sq * ref_sq, 1e-300)),
        .max_rel = max_rel,
        .max_abs_norm = diff_max / @max(ref_max, 1e-300),
        .worst_row_cosine = worst,
    };
}

fn runRows(kit: *const Kit, rows: usize, random: std.Random) !void {
    const gpu = kit.gpu;
    const gpa = gpu.gpa;
    const w = kit.wt.weight;
    // Attention-output-like rows: Gaussian per 256-wide head, each (row, head) scaled by 2^-3 .. 2^3.
    const x = try gpa.alloc(u16, rows * k);
    defer gpa.free(x);
    const xf = try gpa.alloc(f32, rows * k);
    defer gpa.free(xf);
    for (0..rows) |row| for (0..k / 256) |head| {
        const gain = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -3, 3));
        for (0..256) |d| {
            const idx = row * k + head * 256 + d;
            x[idx] = bf16Bits(random.floatNorm(f32) * gain);
            xf[idx] = bf16Float(x[idx]);
        }
    };
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var ref_out = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer ref_out.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(kit.lane.partBytes(rows, w), 4));
    defer part.free();
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, w8a8.scratchBytes(rows, k));
    defer scratch.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer out.free();
    try kit.lane.matmulF32(kit.stream, dx.ptr, k, rows, w, ref_out.ptr, part.ptr);
    try kit.fns.run(kit.stream, dx.ptr, k, rows, w, out.ptr, .f32, scratch.ptr, scratch.len);
    try kit.stream.synchronize();
    const ref_bytes = try check.download(gpu, ref_out);
    defer gpa.free(ref_bytes);
    const out_bytes = try check.download(gpu, out);
    defer gpa.free(out_bytes);
    const scratch_bytes = try check.download(gpu, scratch);
    defer gpa.free(scratch_bytes);
    const ref = std.mem.bytesAsSlice(f32, ref_bytes);
    const got = std.mem.bytesAsSlice(f32, out_bytes);
    const at = w8a8.scratchAt(0, rows, k);
    const xq = scratch_bytes[@intCast(at.xq)..][0 .. rows * k];
    const xs = std.mem.bytesAsSlice(f32, scratch_bytes[@intCast(at.xs)..][0 .. blocks_k * rows * 4]);

    // 1. The device quantizer is the host quantizer, bit for bit (up to 16 rows, first and last included).
    const host_codes = try gpa.alloc(u8, k);
    defer gpa.free(host_codes);
    var host_scales: [blocks_k]f32 = undefined;
    const quant_rows: usize = @min(rows, 16);
    for (0..quant_rows) |i| {
        const row = if (quant_rows == 1) 0 else i * (rows - 1) / (quant_rows - 1);
        try w8a8.quantizeRow(xf[row * k ..][0..k], host_codes, &host_scales);
        try check.expect(std.mem.eql(u8, host_codes, xq[row * k ..][0..k]), "glm53-w8a8 M={d}: row {d} e4m3 codes differ from the host quantizer", .{ rows, row });
        for (0..blocks_k) |b| {
            const device: u32 = @bitCast(xs[b * rows + row]);
            const host: u32 = @bitCast(host_scales[b]);
            try check.expect(device == host, "glm53-w8a8 M={d}: row {d} block {d} scale {x} vs host {x}", .{ rows, row, b, device, host });
        }
    }

    // 2. Sampled outputs vs FP64 sums: W8A8 against its own operands (layout), and both paths' exact targets.
    var layout_worst: f64 = 0;
    var default_worst: f64 = 0;
    var sample_diff_sq: f64 = 0;
    var sample_ref_sq: f64 = 0;
    for (0..samples) |i| {
        const row = if (i == 0) 0 else if (i == 1) rows - 1 else random.uintLessThan(usize, rows);
        const col = if (i == 0) 0 else if (i == 1) n - 1 else random.uintLessThan(usize, n);
        var own: f64 = 0;
        var own_abs: f64 = 0;
        var exact: f64 = 0;
        var exact_abs: f64 = 0;
        for (0..blocks_k) |b| {
            const ws: f64 = kit.wt.block_scales[(col / 128) * blocks_k + b];
            const s: f64 = @as(f64, xs[b * rows + row]) * ws;
            var partial: f64 = 0;
            var partial_abs: f64 = 0;
            for (0..w8a8.block_k) |j| {
                const kk = b * w8a8.block_k + j;
                const code = kit.wt.raw[col * k + kk];
                const prod = @as(f64, w8a8.e4m3Value(xq[row * k + w8a8.activationOffset(kk)])) * w8a8.e4m3Value(code);
                partial += prod;
                partial_abs += @abs(prod);
                const term = @as(f64, xf[row * k + kk]) * fp8.fp8bWeight(code, @floatCast(ws));
                exact += term;
                exact_abs += @abs(term);
            }
            own += partial * s;
            own_abs += partial_abs * @abs(s);
        }
        const g: f64 = got[row * n + col];
        const r: f64 = ref[row * n + col];
        layout_worst = @max(layout_worst, @abs(g - own) / @max(own_abs, 1e-300));
        default_worst = @max(default_worst, @abs(r - exact) / @max(exact_abs, 1e-300));
        sample_diff_sq += (g - exact) * (g - exact);
        sample_ref_sq += exact * exact;
    }
    try check.expect(layout_worst <= layout_tolerance, "glm53-w8a8 M={d}: W8A8 vs FP64 of its own operands {e:.3} of sum |terms| (> {e:.1}): operand layout", .{ rows, layout_worst, layout_tolerance });

    // 3. Gate inputs vs the default lane.
    const st = compare(got, ref, rows);
    std.debug.print("RESULT glm53-w8a8 M={d} N={d} K={d}: vs default nmse={e:.4} cosine={d:.8} worst_row_cosine={d:.8} max_rel(|ref|>=rms)={e:.4} max_abs/max|ref|={e:.4}; sampled vs exact bf16-weight sum nmse={e:.4}; W8A8 vs FP64 own operands {e:.3}, default vs FP64 own {e:.3} (of sum |terms|); quantizer bit-exact on {d} rows\n", .{ rows, n, k, st.nmse, st.cosine, st.worst_row_cosine, st.max_rel, st.max_abs_norm, sample_diff_sq / @max(sample_ref_sq, 1e-300), layout_worst, default_worst, quant_rows });
    try check.expect(st.cosine >= min_cosine, "glm53-w8a8 M={d}: cosine {d:.6} vs default below the {d} sanity floor", .{ rows, st.cosine, min_cosine });

    // 4. Row invariance: a row alone (M = 1, its own quantize and GEMM) equals the same row inside the batch.
    if (rows >= 512) {
        var one_scratch = try cuda.DeviceBuffer.alloc(gpu.d, w8a8.scratchBytes(1, k));
        defer one_scratch.free();
        var one = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer one.free();
        const probes = [_]usize{ 0, 1, 127, 128, 255, 383, rows / 2 + 3, rows - 1 };
        for (probes) |row| {
            try kit.fns.run(kit.stream, dx.ptr + row * k * 2, k, 1, w, one.ptr, .f32, one_scratch.ptr, one_scratch.len);
            try kit.stream.synchronize();
            const alone = try check.download(gpu, one);
            defer gpa.free(alone);
            try check.sameBytes("glm53-w8a8 row alone vs inside the batch", alone, out_bytes[row * n * 4 ..][0 .. n * 4]);
        }
        std.debug.print("RESULT glm53-w8a8 row_invariance M={d}: {d} rows alone bit-exact with the batch\n", .{ rows, probes.len });
    }

    // 5. BENCH: default lane vs W8A8.
    const xq_dev = scratch.ptr + at.xq;
    const xs_dev = scratch.ptr + at.xs;
    const default_us = try timeDefault(kit, dx.ptr, rows, ref_out.ptr, part.ptr);
    const w8a8_us = try timeW8a8(kit, .total, dx.ptr, rows, out.ptr, scratch.ptr, scratch.len, xq_dev, xs_dev);
    const quant_us = try timeW8a8(kit, .quant, dx.ptr, rows, out.ptr, scratch.ptr, scratch.len, xq_dev, xs_dev);
    const gemm_us = try timeW8a8(kit, .gemm, dx.ptr, rows, out.ptr, scratch.ptr, scratch.len, xq_dev, xs_dev);
    const flops = 2.0 * @as(f64, @floatFromInt(rows * n * k));
    std.debug.print("BENCH glm53-w8a8 M={d}: default {s} {d:.1} us {d:.1} TFLOP/s | W8A8 {d:.1} us {d:.1} TFLOP/s (quantize {d:.1} us, GEMM {d:.1} us {d:.1} TFLOP/s) | speedup {d:.2}x\n", .{ rows, if (rows >= cuda.qmmf.fused_rows) "fused64" else "lane", default_us, flops / (default_us * 1e6), w8a8_us, flops / (w8a8_us * 1e6), quant_us, gemm_us, flops / (gemm_us * 1e6), default_us / w8a8_us });
}

fn timeDefault(kit: *const Kit, x: u64, rows: usize, y: u64, part: u64) !f64 {
    var times: [rounds]f64 = undefined;
    for (0..warmup + rounds) |i| {
        try kit.start.record(kit.stream);
        try kit.lane.matmulF32(kit.stream, x, k, rows, kit.wt.weight, y, part);
        try kit.stop.record(kit.stream);
        try kit.stop.synchronize();
        if (i >= warmup) times[i - warmup] = @as(f64, try cuda.Event.elapsedMs(kit.start, kit.stop)) * 1000.0;
    }
    return check.median(&times);
}

const Part = enum { total, quant, gemm };

fn timeW8a8(kit: *const Kit, which: Part, x: u64, rows: usize, y: u64, scratch: u64, scratch_len: usize, xq: u64, xs: u64) !f64 {
    var times: [rounds]f64 = undefined;
    for (0..warmup + rounds) |i| {
        try kit.start.record(kit.stream);
        switch (which) {
            .total => try kit.fns.run(kit.stream, x, k, rows, kit.wt.weight, y, .f32, scratch, scratch_len),
            .quant => try kit.fns.quantize(kit.stream, x, k, rows, k, xq, xs),
            .gemm => try kit.fns.gemmRows(kit.stream, xq, xs, kit.wt.weight, y, rows, .f32),
        }
        try kit.stop.record(kit.stream);
        try kit.stop.synchronize();
        if (i >= warmup) times[i - warmup] = @as(f64, try cuda.Event.elapsedMs(kit.start, kit.stop)) * 1000.0;
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

test "test references the GLM-5.3 W8A8 gate report" {
    _ = &run;
    _ = &runRows;
    _ = &compare;
}

test "compare: identical rows are exact, a scaled copy keeps cosine one" {
    var a: [n]f32 = undefined;
    for (&a, 0..) |*o, i| o.* = ([_]f32{ 1, -2, 3, 4 })[i % 4];
    var b: [n]f32 = undefined;
    for (&b, a) |*o, v| o.* = v * 2;
    const same = compare(&a, &a, 1);
    try std.testing.expectEqual(@as(f64, 0), same.nmse);
    const scaled = compare(&b, &a, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1), scaled.cosine, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), scaled.nmse, 1e-12);
}

test {
    std.testing.refAllDecls(@This());
}
