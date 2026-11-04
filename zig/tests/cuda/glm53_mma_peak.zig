// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! PF0 probe for GLM-5.3 W8A8 MMA. Run on GB10 after building CUDA fatbins.
const std = @import("std");
const cuda = @import("cuda");
const glm53 = @import("fp8_mma");
const backend_probe = @import("fp8_backend_probe");

const AccumulatorRounding = enum { nearest_even, truncate, align_trunc };

fn simulateAccumulator(seed: f64, addend: f64, fraction_bits: u6, rounding: AccumulatorRounding) f64 {
    const exponent: i32 = @intFromFloat(@floor(@log2(@abs(seed))));
    const ulp = std.math.scalbn(@as(f64, 1), exponent - @as(i32, fraction_bits));
    const effective_addend = if (rounding == .align_trunc) @trunc(addend / ulp) * ulp else addend;
    const sum = seed + effective_addend;
    const scaled = sum / ulp;
    const base = @floor(scaled);
    const fraction = scaled - base;
    const rounded = if (rounding != .nearest_even) @trunc(scaled) else base + @as(f64, @floatFromInt(@intFromBool(fraction > 0.5 or (fraction == 0.5 and @mod(base, 2) != 0))));
    return rounded * ulp;
}

fn measuredFractionBits(seed: f64, fraction_bits: u6, rounding: AccumulatorRounding) u6 {
    var measured: u6 = 0;
    for (10..31) |k| {
        const addend = std.math.scalbn(@as(f64, 1), 12 - @as(i32, @intCast(k)));
        if (simulateAccumulator(seed, addend, fraction_bits, rounding) != seed) measured = @intCast(k);
    }
    return measured;
}

const RoundingMode = enum { nearest_even, truncate, align_trunc, other };

fn classifyRounding(got: [8]f64, seed: f64, fraction_bits: u6) RoundingMode {
    const ulp = std.math.scalbn(@as(f64, 1), 12 - @as(i32, fraction_bits));
    const addends = [8]f64{ ulp / 2, ulp * 1.5, ulp * 0.75, ulp * 0.25, -ulp / 2, -ulp * 1.5, -ulp * 0.75, -ulp * 0.25 };
    const models = [_]AccumulatorRounding{ .nearest_even, .truncate, .align_trunc };
    inline for (models) |model| {
        var matches = true;
        inline for (0..8) |i| matches = matches and got[i] == simulateAccumulator(seed, addends[i], fraction_bits, model);
        if (matches) return switch (model) {
            .nearest_even => .nearest_even,
            .truncate => .truncate,
            .align_trunc => .align_trunc,
        };
    }
    return .other;
}

fn probeRoundingAtWidth(_: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, mma: *const glm53.Fp8Mma, bf16: bool, p: u6) !RoundingMode {
    const seed: f32 = 6144;
    const ulp = std.math.scalbn(@as(f64, 1), 12 - @as(i32, p));
    const targets = [8]f64{ ulp / 2, ulp * 1.5, ulp * 0.75, ulp * 0.25, -ulp / 2, -ulp * 1.5, -ulp * 0.75, -ulp * 0.25 };
    var a: [8]u32 = undefined;
    var b: [8]u32 = undefined;
    var seeds: [8]f32 = @splat(seed);
    var da: cuda.DeviceBuffer = undefined;
    var db: cuda.DeviceBuffer = undefined;
    if (bf16) {
        for (targets, 0..) |target, i| {
            a[i] = bf16Bits(1);
            b[i] = bf16Bits(@floatCast(@abs(target)));
            if (target < 0) a[i] |= 0x8000;
        }
    } else {
        for (targets, 0..) |target, i| {
            const pair = fp8Pair(@abs(target));
            a[i] = pair.a | (if (target < 0) @as(u32, 0x80) else 0);
            b[i] = pair.b;
        }
    }
    da = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&a));
    defer da.free();
    db = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&b));
    defer db.free();
    var ds = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&seeds));
    defer ds.free();
    var dout = try cuda.DeviceBuffer.alloc(driver, 8 * @sizeOf(f32));
    defer dout.free();
    try mma.runRawAccumulatorProbe(stream, bf16, da.ptr, db.ptr, ds.ptr, 1, dout.ptr, 8);
    try stream.synchronize();
    var observed: [8]f32 = undefined;
    try dout.download(0, std.mem.sliceAsBytes(&observed));
    var widened: [8]f64 = undefined;
    for (observed, 0..) |v, i| widened[i] = v;
    return classifyRounding(widened, seed, p);
}

test "rounding classifier separates RNE, full-sum truncation, and aligned truncation" {
    const seed = 6144.0;
    inline for (.{ @as(u6, 23), 22, 13 }) |p| {
        var observed: [8]f64 = undefined;
        const ulp = std.math.scalbn(@as(f64, 1), 12 - @as(i32, p));
        const addends = [8]f64{ ulp / 2, ulp * 1.5, ulp * 0.75, ulp * 0.25, -ulp / 2, -ulp * 1.5, -ulp * 0.75, -ulp * 0.25 };
        const models = [_]AccumulatorRounding{ .nearest_even, .truncate, .align_trunc };
        inline for (models) |model| {
            inline for (0..8) |i| observed[i] = simulateAccumulator(seed, addends[i], p, model);
            const expected: RoundingMode = switch (model) {
                .nearest_even => .nearest_even,
                .truncate => .truncate,
                .align_trunc => .align_trunc,
            };
            try std.testing.expectEqual(expected, classifyRounding(observed, seed, p));
        }
        try std.testing.expect(simulateAccumulator(seed, -ulp * 0.75, p, .truncate) != simulateAccumulator(seed, -ulp * 0.75, p, .align_trunc));
    }
    try std.testing.expectEqual(@as(u6, 23), measuredFractionBits(seed, 23, .nearest_even));
    try std.testing.expectEqual(@as(u6, 13), measuredFractionBits(seed, 13, .nearest_even));
}

test "bf16 accumulator control scaling preserves the requested product" {
    const targets = [_]f32{ 0.25, 0.375, 0.125 };
    for (targets) |target| {
        const a = @as(f32, @bitCast(@as(u32, bf16Bits(std.math.scalbn(@as(f32, 1), -6))) << 16));
        const b = @as(f32, @bitCast(@as(u32, bf16Bits(target * 64)) << 16));
        try std.testing.expectEqual(target, a * b);
    }
}

const Bench = struct {
    mma: *const glm53.Fp8Mma,
    xq: u64,
    xs: u64,
    w: u64,
    ws: u64,
    out: u64,
    m: usize,
    n: usize,
    k: usize,
    scalar: bool = false,
    label: []const u8 = "block_128x128_fp8",

    fn launch(self: Bench, stream: cuda.Stream) !void {
        if (self.scalar) {
            try self.mma.runScalar(stream, self.xq, self.xs, self.w, self.ws, self.out, self.m, self.n, self.k, true);
        } else {
            try self.mma.run(stream, self.xq, self.xs, self.w, self.ws, self.out, self.m, self.n, self.k, true);
        }
    }
};

fn median(values: []f64) f64 {
    for (1..values.len) |i| {
        const value = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > value) : (j -= 1) values[j] = values[j - 1];
        values[j] = value;
    }
    return values[values.len / 2];
}

fn timeBench(driver: *const cuda.Driver, stream: cuda.Stream, bench: Bench) !f64 {
    var start = try cuda.Event.init(driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(driver, true);
    defer end.deinit();
    for (0..3) |_| try bench.launch(stream);
    try stream.synchronize();
    var times: [5]f64 = undefined;
    for (&times) |*ms| {
        try start.record(stream);
        for (0..20) |_| try bench.launch(stream);
        try end.record(stream);
        try end.synchronize();
        ms.* = (try cuda.Event.elapsedMs(start, end)) / 20;
    }
    return median(&times);
}

fn randomCode(random: std.Random) u8 {
    const sign: u8 = if (random.boolean()) 0x80 else 0;
    const magnitude = @min(random.int(u8) & 0x7f, 0x7e); // excludes NaN 0x7f/0xff
    return sign | magnitude;
}

fn fp8Pair(target: f64) struct { a: u8, b: u8 } {
    var best_a: u8 = 0;
    var best_b: u8 = 0;
    var best_error = std.math.inf(f64);
    for (0..127) |ai| for (0..127) |bi| {
        const a: u8 = @intCast(ai);
        const b: u8 = @intCast(bi);
        const product = glm53.e4m3(a) * glm53.e4m3(b);
        if (!std.math.isNan(product) and @abs(product - target) < best_error) {
            best_a = a;
            best_b = b;
            best_error = @abs(product - target);
        }
    };
    return .{ .a = best_a, .b = best_b };
}

fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate(bits >> 16);
}

fn runAccumulatorProbe(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, mma: *const glm53.Fp8Mma) !void {
    // Use a non-power-of-two seed so signed inexact addends remain in one binade.
    const count = 27;
    const seed: f32 = 6144;
    const a8 = try gpa.alloc(u32, count);
    defer gpa.free(a8);
    const b8 = try gpa.alloc(u32, count);
    defer gpa.free(b8);
    const a16 = try gpa.alloc(u32, count);
    defer gpa.free(a16);
    const b16 = try gpa.alloc(u32, count);
    defer gpa.free(b16);
    const seeds = try gpa.alloc(f32, count);
    defer gpa.free(seeds);
    const ks = try gpa.alloc(u8, count);
    defer gpa.free(ks);
    for (0..count) |i| {
        seeds[i] = seed;
        const k: u8 = if (i < 21) @intCast(i + 10) else if (i == 21) 13 else 13;
        ks[i] = k;
        const tie = std.math.scalbn(@as(f64, 1), -12);
        const target: f64 = if (i < 21) std.math.scalbn(@as(f64, 1), 12 - @as(i32, k)) else switch (i) {
            21, 24, 26 => tie,
            22 => 3 * std.math.scalbn(@as(f64, 1), -13),
            23 => std.math.scalbn(@as(f64, 1), -14),
            25 => 3 * std.math.scalbn(@as(f64, 1), -13),
            else => unreachable,
        };
        const pair = fp8Pair(target);
        const negative = i == 24 or i == 25;
        a8[i] = pair.a | (if (negative) @as(u32, 0x80) else 0);
        b8[i] = pair.b;
        const bf_a: f32 = if (i < 21) 1 else std.math.scalbn(@as(f32, 1), -6);
        const bf_b: f32 = if (i < 21) std.math.scalbn(@as(f32, 1), 12 - @as(i32, k)) else @floatCast(target * 64);
        a16[i] = bf16Bits(bf_a);
        b16[i] = bf16Bits(bf_b);
        const ulp = std.math.scalbn(@as(f32, 1), -11);
        seeds[i] = if (i == 26) seed + ulp else seed;
    }
    var da8 = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(a8));
    defer da8.free();
    var db8 = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(b8));
    defer db8.free();
    var da16 = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(a16));
    defer da16.free();
    var db16 = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(b16));
    defer db16.free();
    var ds = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(seeds));
    defer ds.free();
    var dout8 = try cuda.DeviceBuffer.alloc(driver, count * @sizeOf(f32));
    defer dout8.free();
    var dout16 = try cuda.DeviceBuffer.alloc(driver, count * @sizeOf(f32));
    defer dout16.free();
    try mma.runRawAccumulatorProbe(stream, false, da8.ptr, db8.ptr, ds.ptr, 1, dout8.ptr, count);
    try mma.runRawAccumulatorProbe(stream, true, da16.ptr, db16.ptr, ds.ptr, 1, dout16.ptr, count);
    try stream.synchronize();
    const got8 = try gpa.alloc(f32, count);
    defer gpa.free(got8);
    const got16 = try gpa.alloc(f32, count);
    defer gpa.free(got16);
    try dout8.download(0, std.mem.sliceAsBytes(got8));
    try dout16.download(0, std.mem.sliceAsBytes(got16));
    var max_k8: i32 = 0;
    var max_k16: i32 = 0;
    for (0..21) |i| {
        if (got8[i] != seed) max_k8 = @max(max_k8, ks[i]);
        if (got16[i] != seed) max_k16 = @max(max_k16, ks[i]);
    }
    const mode8 = if (max_k8 > 0) try probeRoundingAtWidth(gpa, driver, stream, mma, false, @intCast(max_k8)) else .other;
    const mode16 = if (max_k16 > 0) try probeRoundingAtWidth(gpa, driver, stream, mma, true, @intCast(max_k16)) else .other;
    if (max_k8 == 0 or max_k16 == 0) {
        std.debug.print("raw mma.sync accumulator: e4m3 stored_mantissa_bits={s}, bf16 stored_mantissa_bits={s} (<10 means below sweep range)\n", .{ if (max_k8 == 0) "<10" else "measured", if (max_k16 == 0) "<10" else "measured" });
    }
    std.debug.print("raw mma.sync accumulator: e4m3 stored_mantissa_bits={d}, significand_bits={d} (fp32=23/24), rounding={s}; bf16 stored_mantissa_bits={d}, significand_bits={d} (fp32=23/24), rounding={s}\n", .{ max_k8, max_k8 + 1, @tagName(mode8), max_k16, max_k16 + 1, @tagName(mode16) });

    // C=0 with two active k positions measures the reduction adder inside one
    // mma.sync; the previous sweep measures a product added to existing C.
    const large = fp8Pair(6144);
    for (0..21) |i| {
        const k: u8 = @intCast(i + 10);
        const small = fp8Pair(std.math.scalbn(@as(f64, 1), 12 - @as(i32, k)));
        a8[i] = @as(u32, large.a) | (@as(u32, small.a) << 8);
        b8[i] = @as(u32, large.b) | (@as(u32, small.b) << 8);
        seeds[i] = 0;
    }
    var d_inmma_a = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(a8[0..21]));
    defer d_inmma_a.free();
    var d_inmma_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(b8[0..21]));
    defer d_inmma_b.free();
    const zero_seeds = try gpa.alloc(f32, 21);
    defer gpa.free(zero_seeds);
    @memset(zero_seeds, 0);
    var d_zero_seeds = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(zero_seeds));
    defer d_zero_seeds.free();
    try mma.runRawAccumulatorProbe(stream, false, d_inmma_a.ptr, d_inmma_b.ptr, d_zero_seeds.ptr, 1, dout8.ptr, 21);
    try stream.synchronize();
    try dout8.download(0, std.mem.sliceAsBytes(got8[0..21]));
    var in_mma_max_k: usize = 0;
    for (0..21) |i| if (got8[i] != 6144) {
        in_mma_max_k = i + 10;
    };
    std.debug.print("raw mma.sync in-MMA adder: stored_mantissa_bits={d} (C=0, products 6144 + small); C-accumulate path reported above\n", .{in_mma_max_k});
    std.debug.print("PF0 SUMMARY\nbest backend per shape: run g mode for CUTLASS vs MMA timings\naccumulator width/rounding e4m3: {d} bits/{s}; bf16: {d} bits/{s}\ncuBLASLt 128x128 block-scaling support: run h mode\n", .{ max_k8, @tagName(mode8), max_k16, @tagName(mode16) });
}

fn checkSamples(gpa: std.mem.Allocator, got: []const f32, xq: []const u8, xs: []const f32, w: []const u8, ws: []const f32, m: usize, n: usize, k: usize) !struct { abs: f64, normalized: f64 } {
    const sample_x = try gpa.alloc(f32, 64 * k);
    defer gpa.free(sample_x);
    const sample_w = try gpa.alloc(f32, 64 * k);
    defer gpa.free(sample_w);
    var rows: [64]usize = undefined;
    var cols: [64]usize = undefined;
    for (0..64) |i| {
        rows[i] = (i * 67) % m;
        cols[i] = (i * 131) % n;
        for (0..k) |ki| {
            sample_x[i * k + ki] = @floatCast(glm53.e4m3(xq[rows[i] * k + glm53.activationFragmentIndex(rows[i], ki)]));
            sample_w[i * k + ki] = @floatCast(glm53.e4m3(w[glm53.weightFragmentIndex(cols[i], ki, k)]));
        }
    }
    var max_abs: f64 = 0;
    var max_ref: f64 = 0;
    for (0..64) |i| for (0..64) |j| {
        const row = rows[i];
        const col = cols[j];
        var expected: f32 = 0;
        for (0..k / 128) |g| {
            var partial: f32 = 0;
            for (0..128) |t| {
                const ki = g * 128 + t;
                partial = partial + sample_x[i * k + ki] * sample_w[j * k + ki];
            }
            const scale = ws[(col / 128) * (k / 128) + g] * xs[row * (k / 128) + g];
            expected = @mulAdd(f32, partial, scale, expected);
        }
        const actual = got[row * n + col];
        const err = @abs(@as(f64, actual) - expected);
        max_abs = @max(max_abs, err);
        max_ref = @max(max_ref, @abs(@as(f64, expected)));
    };
    return .{ .abs = max_abs, .normalized = max_abs / @max(max_ref, 1e-30) };
}

fn reportCase(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, bench: Bench, host_xq: []const u8, host_xs: []const f32, host_w: []const u8, host_ws: []const f32) !void {
    const ms = try timeBench(driver, stream, bench);
    try stream.synchronize();
    const got = try gpa.alloc(f32, bench.m * bench.n);
    defer gpa.free(got);
    const output: cuda.DeviceBuffer = .{ .d = driver, .ptr = bench.out, .len = bench.m * bench.n * @sizeOf(f32), .borrowed = true };
    try output.download(0, std.mem.sliceAsBytes(got));
    const err = try checkSamples(gpa, got, host_xq, host_xs, host_w, host_ws, bench.m, bench.n, bench.k);
    const flops = 2.0 * @as(f64, @floatFromInt(bench.m)) * @as(f64, @floatFromInt(bench.n)) * @as(f64, @floatFromInt(bench.k));
    const pass = err.normalized <= 2e-3;
    std.debug.print("{s}: {s}, M={d} N={d} K={d}, median5x20={d:.4} ms, {d:.2} TFLOPS, max_abs={e:.6}, normalized_error={e:.6} (gate 2e-3)\n", .{
        if (pass) "PASS" else "FAIL",
        bench.label,
        bench.m,
        bench.n,
        bench.k,
        ms,
        flops / (ms / 1000.0) / 1e12,
        err.abs,
        err.normalized,
    });
}

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    if (args.next()) |mode| {
        if (std.mem.eql(u8, mode, "g") or std.mem.eql(u8, mode, "h")) return backend_probe.runMode(init, mode);
    } else if (std.c.getenv("TF_GLM53_FP8_GEMM")) |raw| {
        const selected = std.mem.span(raw);
        if (std.mem.eql(u8, selected, "cutlass")) return backend_probe.runMode(init, "g");
        if (std.mem.eql(u8, selected, "cublaslt")) return backend_probe.runMode(init, "h");
        if (!std.mem.eql(u8, selected, "mma")) return error.InvalidMode;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var mma = try glm53.Fp8Mma.load(&driver);
    defer mma.unload();
    try runAccumulatorProbe(init.gpa, &driver, stream, &mma);

    const gpa = init.gpa;
    const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
    const m: usize = 4096;
    const n: usize = 6144;
    const k: usize = 6144;
    var prng = std.Random.DefaultPrng.init(0x20261021);
    const random = prng.random();

    // Register-only BF16 and E4M3 peak loops; four independent C fragments per warp.
    var peak = try cuda.DeviceBuffer.alloc(&driver, sms * 4 * 256 * @sizeOf(f32));
    defer peak.free();
    const peak_modes = [_]bool{ false, true };
    for (peak_modes) |fp8| {
        var events_start = try cuda.Event.init(&driver, true);
        defer events_start.deinit();
        var events_end = try cuda.Event.init(&driver, true);
        defer events_end.deinit();
        for (0..3) |_| try mma.runPeak(stream, fp8, 4096, sms, peak.ptr);
        try stream.synchronize();
        var times: [20]f64 = undefined;
        for (&times) |*ms| {
            try events_start.record(stream);
            try mma.runPeak(stream, fp8, 4096, sms, peak.ptr);
            try events_end.record(stream);
            try events_end.synchronize();
            ms.* = try cuda.Event.elapsedMs(events_start, events_end);
        }
        const elapsed = median(&times);
        const flops_each: usize = if (fp8) 8192 else 4096;
        const flops: f64 = @floatFromInt(sms * 4 * 8 * 4 * 4096 * flops_each);
        const expected: f32 = @as(f32, if (fp8) 128 else 64) * 4096;
        var got_peak: [1]f32 = undefined;
        const peak_head: cuda.DeviceBuffer = .{ .d = &driver, .ptr = peak.ptr, .len = @sizeOf(f32), .borrowed = true };
        try peak_head.download(0, std.mem.sliceAsBytes(&got_peak));
        const abs_error = @abs(@as(f64, got_peak[0]) - expected);
        const rel_error = abs_error / expected;
        std.debug.print("{s}_mma_peak: {d:.2} TFLOPS, max_abs={e:.6}, max_rel={e:.6}\n", .{ if (fp8) "e4m3" else "bf16", flops / (elapsed / 1000.0) / 1e12, abs_error, rel_error });
    }

    const xq = try gpa.alloc(u8, m * k);
    defer gpa.free(xq);
    const xs = try gpa.alloc(f32, m * (k / 128));
    defer gpa.free(xs);
    const w = try gpa.alloc(u8, n * k);
    defer gpa.free(w);
    const ws = try gpa.alloc(f32, (n / 128) * (k / 128));
    defer gpa.free(ws);
    for (xq) |*v| v.* = randomCode(random);
    for (w) |*v| v.* = randomCode(random);
    for (xs) |*v| v.* = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -8, 8));
    for (ws) |*v| v.* = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -8, 8));

    var dx = try cuda.DeviceBuffer.fromHost(&driver, xq);
    defer dx.free();
    var ds = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(xs));
    defer ds.free();
    var dw = try cuda.DeviceBuffer.fromHost(&driver, w);
    defer dw.free();
    var dws = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(ws));
    defer dws.free();
    var dout = try cuda.DeviceBuffer.alloc(&driver, m * n * @sizeOf(f32));
    defer dout.free();

    const block = Bench{ .mma = &mma, .xq = dx.ptr, .xs = ds.ptr, .w = dw.ptr, .ws = dws.ptr, .out = dout.ptr, .m = m, .n = n, .k = k };
    try reportCase(gpa, &driver, stream, block, xq, xs, w, ws);
    const per_tensor_xs = try gpa.alloc(f32, xs.len);
    defer gpa.free(per_tensor_xs);
    const per_tensor_ws = try gpa.alloc(f32, ws.len);
    defer gpa.free(per_tensor_ws);
    @memset(per_tensor_xs, 1);
    @memset(per_tensor_ws, 1);
    var dptx = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(per_tensor_xs));
    defer dptx.free();
    var dptw = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(per_tensor_ws));
    defer dptw.free();
    try reportCase(gpa, &driver, stream, .{ .mma = &mma, .xq = dx.ptr, .xs = dptx.ptr, .w = dw.ptr, .ws = dptw.ptr, .out = dout.ptr, .m = m, .n = n, .k = k, .label = "per_tensor_fp8" }, xq, per_tensor_xs, w, per_tensor_ws);
    try reportCase(gpa, &driver, stream, .{ .mma = &mma, .xq = dx.ptr, .xs = ds.ptr, .w = dw.ptr, .ws = dptw.ptr, .out = dout.ptr, .m = m, .n = n, .k = k, .label = "activation_1x128_fp8" }, xq, xs, w, per_tensor_ws);
    try reportCase(gpa, &driver, stream, .{ .mma = &mma, .xq = dx.ptr, .xs = ds.ptr, .w = dw.ptr, .ws = dws.ptr, .out = dout.ptr, .m = m, .n = 2048, .k = k, .label = "block_fp8_N2048" }, xq, xs, w, ws);

    // Tail cases exercise the last partial CTA in M and N. The weight buffer
    // has padded rows, matching the fp8b load layout used by the CUDA kernel.
    for ([_]usize{ 1, 7, 130 }) |tail_m| {
        try reportCase(gpa, &driver, stream, .{ .mma = &mma, .xq = dx.ptr, .xs = ds.ptr, .w = dw.ptr, .ws = dws.ptr, .out = dout.ptr, .m = tail_m, .n = 200, .k = k, .label = "tail_M_N200" }, xq, xs, w, ws);
    }

    // Same activation row and weights, launched alone and inside the large
    // grid. This must stay bit-identical if CTA scheduling is row invariant.
    const one_row = Bench{ .mma = &mma, .xq = dx.ptr, .xs = ds.ptr, .w = dw.ptr, .ws = dws.ptr, .out = dout.ptr, .m = 1, .n = n, .k = k };
    try one_row.launch(stream);
    try stream.synchronize();
    const row_one = try gpa.alloc(f32, n);
    defer gpa.free(row_one);
    const row_out: cuda.DeviceBuffer = .{ .d = &driver, .ptr = dout.ptr, .len = n * @sizeOf(f32), .borrowed = true };
    try row_out.download(0, std.mem.sliceAsBytes(row_one));
    try block.launch(stream);
    try stream.synchronize();
    const full_out: cuda.DeviceBuffer = .{ .d = &driver, .ptr = dout.ptr, .len = m * n * @sizeOf(f32), .borrowed = true };
    const all_rows = try gpa.alloc(f32, m * n);
    defer gpa.free(all_rows);
    try full_out.download(0, std.mem.sliceAsBytes(all_rows));
    const row_invariant = std.mem.sliceAsBytes(row_one).len == std.mem.sliceAsBytes(all_rows[0..n]).len and
        std.mem.eql(u8, std.mem.sliceAsBytes(row_one), std.mem.sliceAsBytes(all_rows[0..n]));
    std.debug.print("row_invariance M=1 vs M=4096: {s}\n", .{if (row_invariant) "PASS" else "FAIL"});

    // The scalar oracle is intentionally limited to a small shape.
    const small_m: usize = 16;
    const small_n: usize = 128;
    const small_k: usize = 256;
    const small_xq = try gpa.alloc(u8, small_m * small_k);
    defer gpa.free(small_xq);
    const small_xs = try gpa.alloc(f32, small_m * 2);
    defer gpa.free(small_xs);
    const small_w = try gpa.alloc(u8, small_n * small_k);
    defer gpa.free(small_w);
    const small_ws = try gpa.alloc(f32, 2);
    defer gpa.free(small_ws);
    for (small_xq) |*v| v.* = randomCode(random);
    for (small_w) |*v| v.* = randomCode(random);
    @memset(small_xs, 1);
    @memset(small_ws, 1);
    var dsx = try cuda.DeviceBuffer.fromHost(&driver, small_xq);
    defer dsx.free();
    var dss = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(small_xs));
    defer dss.free();
    var dsw = try cuda.DeviceBuffer.fromHost(&driver, small_w);
    defer dsw.free();
    var dsws = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(small_ws));
    defer dsws.free();
    var dso = try cuda.DeviceBuffer.alloc(&driver, small_m * small_n * @sizeOf(f32));
    defer dso.free();
    const small = Bench{ .mma = &mma, .xq = dsx.ptr, .xs = dss.ptr, .w = dsw.ptr, .ws = dsws.ptr, .out = dso.ptr, .m = small_m, .n = small_n, .k = small_k, .scalar = true, .label = "scalar_oracle" };
    try reportCase(gpa, &driver, stream, small, small_xq, small_xs, small_w, small_ws);

    std.debug.print("mxf8f6f4: not available; nvcc 13.0 acceptance for sm_121a must be confirmed by the Spark compile before enabling this PTX kind.\n", .{});
}

test "PF0 probe declarations are all analyzed on the host" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Bench);
    std.testing.refAllDecls(AccumulatorRounding);
}
