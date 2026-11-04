// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 FP8 GEMM probe: g selects CUTLASS, h selects cuBLASLt.
const std = @import("std");
const cuda = @import("cuda");
const cutlass = @import("glm53_cutlass");
const fp8 = @import("glm53_fp8");

const Backend = enum { cutlass, cublaslt };
const k_dim = 6144;

// bf16 unit roundoff is 2^-8; 5e-4 covers accumulation-order differences.
// With the probe's 2^+/-8 scales, a layout or scale-index error is order one.
const bf16_gate: f64 = 4.4e-3;

const SummaryShape = struct { m: usize, n: usize, backend: []const u8, tflops: f64 };

fn formatSummary(out: []u8, shapes: []const SummaryShape, e4m3_bits: []const u8, e4m3_rounding: []const u8, bf16_bits: []const u8, bf16_rounding: []const u8) ![]const u8 {
    var used = (try std.fmt.bufPrint(out, "PF0 SUMMARY\nbest FP8 GEMM backend per shape:\n", .{})).len;
    for (shapes) |shape| used += (try std.fmt.bufPrint(out[used..], "  M={d} N={d}: {s}, {d:.2} TFLOPS\n", .{ shape.m, shape.n, shape.backend, shape.tflops })).len;
    used += (try std.fmt.bufPrint(out[used..], "accumulator e4m3: {s} bits, {s}; bf16: {s} bits, {s}\n", .{ e4m3_bits, e4m3_rounding, bf16_bits, bf16_rounding })).len;
    return out[0..used];
}

test "PF0 summary formats backend winners, TFLOPS, and both accumulator reports" {
    const fake = [_]SummaryShape{
        .{ .m = 4096, .n = 6144, .backend = "mma", .tflops = 101.25 },
        .{ .m = 1, .n = 6144, .backend = "cublaslt", .tflops = 2.5 },
    };
    var output: [512]u8 = undefined;
    const formatted = try formatSummary(&output, &fake, "23", "nearest_even", "7", "truncate");
    try std.testing.expectEqualStrings("PF0 SUMMARY\nbest FP8 GEMM backend per shape:\n  M=4096 N=6144: mma, 101.25 TFLOPS\n  M=1 N=6144: cublaslt, 2.50 TFLOPS\naccumulator e4m3: 23 bits, nearest_even; bf16: 7 bits, truncate\n", formatted);
}

fn numericGate(normalized: f64) bool {
    return normalized <= bf16_gate;
}

test "bf16 output gate includes nearest rounding and rejects scale-layout errors" {
    try std.testing.expect(numericGate(3.9e-3));
    try std.testing.expect(numericGate(4.4e-3));
    try std.testing.expect(!numericGate(5e-3));
}

test "cuBLASLt M=1 activation scales pack the padded execution rows" {
    const exec_m = (1 + 3) / 4 * 4;
    const groups = k_dim / 128;
    const source = try std.testing.allocator.alloc(f32, exec_m * groups);
    defer std.testing.allocator.free(source);
    const packed_scales = try std.testing.allocator.alloc(f32, exec_m * groups);
    defer std.testing.allocator.free(packed_scales);
    for (0..exec_m) |row| {
        for (0..groups) |group| source[row * groups + group] = @floatFromInt(row * groups + group + 1);
    }
    try fp8.packCublasLtActivationScales(packed_scales, source, exec_m, k_dim);
    try std.testing.expectEqual(@as(usize, 4), exec_m);
    for (0..groups) |group| for (0..exec_m) |row| try std.testing.expectEqual(source[row * groups + group], packed_scales[group * exec_m + row]);
}

fn rowRandom(row: usize, stream: u64) std.Random.DefaultPrng {
    return std.Random.DefaultPrng.init(0x2053_1021 ^ (stream *% 0x9e37_79b9_7f4a_7c15) ^ @as(u64, @intCast(row)));
}

fn fillActivationRows(xq: []u8, xs: []f32, m: usize, k: usize) void {
    const groups = k / 128;
    for (0..m) |row| {
        var prng = rowRandom(row, 0);
        const random = prng.random();
        for (0..k) |col| xq[row * k + col] = code(random);
        for (0..groups) |group| xs[row * groups + group] = std.math.scalbn(@as(f32, 1), random.intRangeAtMost(i32, -8, 8));
    }
}

test "probe activation codes and scales are row invariant across batch sizes" {
    const small_x = try std.testing.allocator.alloc(u8, k_dim);
    defer std.testing.allocator.free(small_x);
    const large_x = try std.testing.allocator.alloc(u8, 4096 * k_dim);
    defer std.testing.allocator.free(large_x);
    const small_s = try std.testing.allocator.alloc(f32, k_dim / 128);
    defer std.testing.allocator.free(small_s);
    const large_s = try std.testing.allocator.alloc(f32, 4096 * (k_dim / 128));
    defer std.testing.allocator.free(large_s);
    fillActivationRows(small_x, small_s, 1, k_dim);
    fillActivationRows(large_x, large_s, 4096, k_dim);
    try std.testing.expectEqualSlices(u8, small_x, large_x[0..k_dim]);
    try std.testing.expectEqualSlices(f32, small_s, large_s[0 .. k_dim / 128]);
}

fn code(random: std.Random) u8 {
    return (if (random.boolean()) @as(u8, 0x80) else 0) | @min(random.int(u8) & 0x7f, 0x7e);
}

const CaseResult = struct { output: []u16, tflops: f64, err: f64, passed: bool, best_backend: []const u8 };

fn runCase(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, backend: Backend, lt: ?*const cuda.cublaslt.Library, cutlass_lib: ?*const cutlass.Library, mma: ?*const fp8.Fp8Mma, m: usize, n: usize) !?CaseResult {
    const exec_m = if (backend == .cublaslt) (m + 3) / 4 * 4 else m;
    const groups = k_dim / 128;
    const n_blocks = n / 128;
    const xq = try gpa.alloc(u8, exec_m * k_dim);
    defer gpa.free(xq);
    const w = try gpa.alloc(u8, n * k_dim);
    defer gpa.free(w);
    const xs = try gpa.alloc(f32, exec_m * groups);
    defer gpa.free(xs);
    const ws = try gpa.alloc(f32, n_blocks * groups);
    defer gpa.free(ws);
    var weights_prng = std.Random.DefaultPrng.init(0x2053_1021);
    const weights_random = weights_prng.random();
    for (w) |*v| v.* = code(weights_random);
    for (ws) |*v| v.* = std.math.scalbn(@as(f32, 1), weights_random.intRangeAtMost(i32, -8, 8));
    fillActivationRows(xq, xs, exec_m, k_dim);
    if (exec_m > m) {
        @memset(xq[m * k_dim ..], 0);
        for (xs[m * groups ..]) |*scale| scale.* = 1;
    }

    var d_x = try cuda.DeviceBuffer.fromHost(driver, xq);
    defer d_x.free();
    var d_w = try cuda.DeviceBuffer.fromHost(driver, w);
    defer d_w.free();
    var d_xs: cuda.DeviceBuffer = undefined;
    var d_ws: cuda.DeviceBuffer = undefined;
    var packed_xs: ?[]f32 = null;
    var packed_ws: ?[]f32 = null;
    defer if (packed_xs) |p| gpa.free(p);
    defer if (packed_ws) |p| gpa.free(p);
    if (backend == .cublaslt) {
        packed_xs = try gpa.alloc(f32, xs.len);
        packed_ws = try gpa.alloc(f32, n_blocks * ((groups + 3) / 4 * 4));
        try fp8.packCublasLtActivationScales(packed_xs.?, xs, exec_m, k_dim);
        try fp8.packCublasLtWeightScales(packed_ws.?, ws, n, k_dim);
        d_xs = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(packed_xs.?));
        d_ws = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(packed_ws.?));
    } else {
        d_xs = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(xs));
        d_ws = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ws));
    }
    defer d_xs.free();
    defer d_ws.free();

    var d_out = try cuda.DeviceBuffer.alloc(driver, exec_m * n * @sizeOf(u16));
    defer d_out.free();
    var d_repeat = try cuda.DeviceBuffer.alloc(driver, exec_m * n * @sizeOf(u16));
    defer d_repeat.free();
    var workspace_bytes: usize = 256;
    var lt_plan: ?cuda.cublaslt.Fp8BlockLinear = null;
    defer if (lt_plan) |*plan| plan.deinit();
    if (backend == .cublaslt) {
        lt_plan = cuda.cublaslt.Fp8BlockLinear.init(lt.?, exec_m, n, k_dim, 64 * 1024 * 1024) catch |e| {
            std.debug.print("cuBLASLt M={d} N={d} K={d} UNSUPPORTED TFLOPS=n/a err=n/a gate={d:.4}: {s}\n", .{ m, n, k_dim, bf16_gate, @errorName(e) });
            std.debug.print("UNSUPPORTED cuBLASLt M={d} N={d} K={d}: {s}\n", .{ m, n, k_dim, @errorName(e) });
            return null;
        };
        workspace_bytes = @max(workspace_bytes, lt_plan.?.workspace_need);
    } else {
        workspace_bytes = try cutlass_lib.?.workspace(m, n, k_dim);
    }
    var d_workspace = try cuda.DeviceBuffer.alloc(driver, @max(workspace_bytes, 1));
    defer d_workspace.free();

    const launch = struct {
        fn call(be: Backend, cutlass_: ?*const cutlass.Library, plan: ?*const cuda.cublaslt.Fp8BlockLinear, s: cuda.Stream, dx: u64, dw: u64, dxs: u64, dws: u64, dout: u64, m_: usize, n_: usize, work: u64, work_len: usize) !void {
            if (be == .cutlass) return cutlass_.?.gemm(dx, dw, dxs, dws, dout, m_, n_, k_dim, work, work_len, s);
            return plan.?.run(dx, dw, dxs, dws, dout, work, work_len, s.handle);
        }
    }.call;
    try launch(backend, cutlass_lib, if (lt_plan) |*p| p else null, stream, d_x.ptr, d_w.ptr, d_xs.ptr, d_ws.ptr, d_out.ptr, exec_m, n, d_workspace.ptr, workspace_bytes);
    try stream.synchronize();
    var start = try cuda.Event.init(driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(driver, true);
    defer end.deinit();
    var times: [5]f64 = undefined;
    for (&times) |*ms| {
        try start.record(stream);
        for (0..20) |_| try launch(backend, cutlass_lib, if (lt_plan) |*p| p else null, stream, d_x.ptr, d_w.ptr, d_xs.ptr, d_ws.ptr, d_out.ptr, exec_m, n, d_workspace.ptr, workspace_bytes);
        try end.record(stream);
        try end.synchronize();
        ms.* = (try cuda.Event.elapsedMs(start, end)) / 20;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    const elapsed = times[times.len / 2];
    const backend_name = if (backend == .cutlass) "CUTLASS" else "cuBLASLt";
    var best_backend = backend_name;
    var best_tflops: ?f64 = null;
    const output = try gpa.alloc(u16, m * n);
    errdefer gpa.free(output);
    const padded_output = try gpa.alloc(u16, exec_m * n);
    defer gpa.free(padded_output);
    try d_out.download(0, std.mem.sliceAsBytes(padded_output));
    @memcpy(output, padded_output[0 .. m * n]);
    try launch(backend, cutlass_lib, if (lt_plan) |*p| p else null, stream, d_x.ptr, d_w.ptr, d_xs.ptr, d_ws.ptr, d_repeat.ptr, exec_m, n, d_workspace.ptr, workspace_bytes);
    try stream.synchronize();
    const repeated = try gpa.alloc(u16, m * n);
    defer gpa.free(repeated);
    try d_repeat.download(0, std.mem.sliceAsBytes(padded_output));
    @memcpy(repeated, padded_output[0 .. m * n]);
    const repeat_equal = std.mem.eql(u16, output, repeated);

    var mma_equal = true;
    if (backend == .cutlass) {
        const x_fragment = try gpa.alloc(u8, m * k_dim);
        defer gpa.free(x_fragment);
        const w_fragment = try gpa.alloc(u8, ((n + 63) / 64 * 64) * k_dim);
        defer gpa.free(w_fragment);
        for (0..m) |row| {
            for (0..k_dim) |col| x_fragment[row * k_dim + fp8.activationFragmentIndex(row, col)] = xq[row * k_dim + col];
        }
        @memset(w_fragment, 0);
        for (0..n) |row| {
            for (0..k_dim) |col| w_fragment[fp8.weightFragmentIndex(row, col, k_dim)] = w[row * k_dim + col];
        }
        var d_mma_x = try cuda.DeviceBuffer.fromHost(driver, x_fragment);
        defer d_mma_x.free();
        var d_mma_w = try cuda.DeviceBuffer.fromHost(driver, w_fragment);
        defer d_mma_w.free();
        var d_mma_y = try cuda.DeviceBuffer.alloc(driver, m * n * @sizeOf(u16));
        defer d_mma_y.free();
        try mma.?.run(stream, d_mma_x.ptr, d_xs.ptr, d_mma_w.ptr, d_ws.ptr, d_mma_y.ptr, m, n, k_dim, false);
        try stream.synchronize();
        const mma_output = try gpa.alloc(u16, m * n);
        defer gpa.free(mma_output);
        try d_mma_y.download(0, std.mem.sliceAsBytes(mma_output));
        var mma_start = try cuda.Event.init(driver, true);
        defer mma_start.deinit();
        var mma_end = try cuda.Event.init(driver, true);
        defer mma_end.deinit();
        for (0..3) |_| try mma.?.run(stream, d_mma_x.ptr, d_xs.ptr, d_mma_w.ptr, d_ws.ptr, d_mma_y.ptr, m, n, k_dim, false);
        try stream.synchronize();
        var mma_times: [5]f64 = undefined;
        for (&mma_times) |*ms| {
            try mma_start.record(stream);
            for (0..20) |_| try mma.?.run(stream, d_mma_x.ptr, d_xs.ptr, d_mma_w.ptr, d_ws.ptr, d_mma_y.ptr, m, n, k_dim, false);
            try mma_end.record(stream);
            try mma_end.synchronize();
            ms.* = (try cuda.Event.elapsedMs(mma_start, mma_end)) / 20;
        }
        std.mem.sort(f64, &mma_times, {}, std.sort.asc(f64));
        const mma_ms = mma_times[mma_times.len / 2];
        const cutlass_ms = elapsed;
        best_backend = if (cutlass_ms <= mma_ms) "CUTLASS" else "MMA";
        const cutlass_tflops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * k_dim / (cutlass_ms / 1000) / 1e12;
        const mma_tflops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * k_dim / (mma_ms / 1000) / 1e12;
        const cutlass_gbps = @as(f64, @floatFromInt(n * k_dim)) / (cutlass_ms / 1000) / 1e9;
        const mma_gbps = @as(f64, @floatFromInt(n * k_dim)) / (mma_ms / 1000) / 1e9;
        std.debug.print("CUTLASS vs MMA M={d} N={d}: cutlass_ms={d:.4} mma_ms={d:.4} speedup={d:.3} cutlass={d:.2} MMA={d:.2} TFLOPS", .{ m, n, cutlass_ms, mma_ms, cutlass_ms / mma_ms, cutlass_tflops, mma_tflops });
        best_tflops = if (cutlass_ms <= mma_ms) cutlass_tflops else mma_tflops;
        if (m <= 64) std.debug.print(" weight_GB/s CUTLASS={d:.1} MMA={d:.1}", .{ cutlass_gbps, mma_gbps });
        std.debug.print("\n", .{});
        var max_diff: f32 = 0;
        for (output, mma_output) |cutlass_value, mma_value| {
            if (cutlass_value != mma_value) mma_equal = false;
            const cutlass_f: f32 = @bitCast(@as(u32, cutlass_value) << 16);
            const mma_f: f32 = @bitCast(@as(u32, mma_value) << 16);
            max_diff = @max(max_diff, @abs(cutlass_f - mma_f));
        }
        std.debug.print("{s} CUTLASS vs mma bf16 bit-equality: full output, max abs diff={e:.6}\n", .{ if (mma_equal) "PASS" else "FAIL", max_diff });
    }

    var max_abs: f64 = 0;
    var max_ref: f64 = 0;
    for (0..64) |sample| {
        const row = (sample * 67) % m;
        const col = (sample * 131) % n;
        var expected: f32 = 0;
        for (0..groups) |g| {
            var partial: f32 = 0;
            for (0..128) |i| partial += @as(f32, @floatCast(fp8.e4m3(xq[row * k_dim + g * 128 + i]))) * @as(f32, @floatCast(fp8.e4m3(w[col * k_dim + g * 128 + i])));
            expected = @mulAdd(f32, partial, xs[row * groups + g] * ws[(col / 128) * groups + g], expected);
        }
        const actual: f32 = @bitCast(@as(u32, output[row * n + col]) << 16);
        max_abs = @max(max_abs, @abs(@as(f64, actual) - expected));
        max_ref = @max(max_ref, @abs(@as(f64, expected)));
    }
    const normalized = max_abs / @max(max_ref, 1e-30);
    const pass = repeat_equal and numericGate(normalized) and mma_equal;
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * k_dim;
    const tflops = flops / (elapsed / 1000) / 1e12;
    std.debug.print("{s} M={d} N={d} K={d} {d:.2} TFLOPS err={e:.4} gate={d:.4}\n", .{ backend_name, m, n, k_dim, tflops, normalized, bf16_gate });
    std.debug.print("{s} {s} M={d} (exec {d}) N={d} K={d}: median5x20={d:.4} ms, {d:.2} TFLOPS, abs={e:.4}, rel={e:.4}, repeat={s}\n", .{ if (pass) "PASS" else "FAIL", if (backend == .cutlass) "cutlass" else "cublaslt", m, exec_m, n, k_dim, elapsed, tflops, max_abs, normalized, if (repeat_equal) "bit-identical" else "DIFFERENT" });
    return .{ .output = output, .tflops = best_tflops orelse tflops, .err = normalized, .passed = pass, .best_backend = best_backend };
}

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    const mode = args.next() orelse return error.ModeRequired;
    try runMode(init, mode);
}

/// Spark parity/throughput gate for the model's 6144-wide FP8 projections.
/// The test uses fixed row tiles so batch and prompt-chunk sizes do not alter
/// either backend's quantization or accumulation contract.
pub fn runPrefill(gpu: anytype) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var library = try cutlass.Library.open();
    defer library.close();
    var mma = try fp8.Fp8Mma.load(gpu.d);
    defer mma.unload();
    std.debug.print("glm53-prefill CUTLASS dynamic shared memory: {d} bytes (opted in by init)\n", .{library.dynamicSmemBytes()});

    var failures: usize = 0;
    for ([_]usize{ 2048, 6144, 8192 }) |n| {
        const result = try runCase(gpu.gpa, gpu.d, stream, .cutlass, null, &library, &mma, 2048, n) orelse {
            failures += 1;
            continue;
        };
        defer gpu.gpa.free(result.output);
        if (!result.passed) failures += 1;
        std.debug.print("glm53-prefill projection N={d} selected={s} TFLOPS={d:.2} MMA parity={s}\n", .{ n, result.best_backend, result.tflops, if (result.passed) "PASS" else "FAIL" });
    }
    if (failures != 0) return error.PrefillParityFailed;
}

pub fn runMode(init: std.process.Init, mode: []const u8) !void {
    const backend: Backend = if (std.mem.eql(u8, mode, "g")) .cutlass else if (std.mem.eql(u8, mode, "h")) .cublaslt else return error.InvalidMode;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var cutlass_lib: ?cutlass.Library = null;
    if (backend == .cutlass) {
        cutlass_lib = cutlass.Library.open() catch |e| {
            if (e == error.LibraryUnavailable) std.debug.print("CUTLASS library unavailable; default zig-out/kernels/libtf_glm53_cutlass.so is relative to the repository root (or set TF_GLM53_CUTLASS_LIB)\n", .{});
            return e;
        };
    }
    defer if (cutlass_lib) |*library| library.close();
    if (cutlass_lib) |*library| std.debug.print("CUTLASS dynamic shared memory: {d} bytes (opted in by init)\n", .{library.dynamicSmemBytes()});
    var lt_lib: ?cuda.cublaslt.Library = if (backend == .cublaslt) try cuda.cublaslt.Library.open() else null;
    defer if (lt_lib) |*library| library.close();
    var mma: ?fp8.Fp8Mma = if (backend == .cutlass) try fp8.Fp8Mma.load(&driver) else null;
    defer if (mma) |*kernel| kernel.unload();
    const cutlass_ptr = if (cutlass_lib) |*p| p else null;
    const lt_ptr = if (lt_lib) |*p| p else null;
    var first_row_1: ?[]u16 = null;
    defer if (first_row_1) |p| init.gpa.free(p);
    var first_row_4096: ?[]u16 = null;
    defer if (first_row_4096) |p| init.gpa.free(p);
    var first_64_rows: ?[]u16 = null;
    defer if (first_64_rows) |p| init.gpa.free(p);
    var first_4096_rows: ?[]u16 = null;
    defer if (first_4096_rows) |p| init.gpa.free(p);
    var summary: [4]SummaryShape = @splat(.{ .m = 0, .n = 0, .backend = "UNSUPPORTED", .tflops = 0 });
    var supported: usize = 0;
    for ([_]struct { m: usize, n: usize }{ .{ .m = 4096, .n = 6144 }, .{ .m = 4096, .n = 2048 }, .{ .m = 64, .n = 6144 }, .{ .m = 1, .n = 6144 } }) |shape| {
        const result = try runCase(init.gpa, &driver, stream, backend, lt_ptr, cutlass_ptr, if (mma) |*p| p else null, shape.m, shape.n) orelse continue;
        defer init.gpa.free(result.output);
        supported += 1;
        summary[supported - 1] = .{ .m = shape.m, .n = shape.n, .backend = result.best_backend, .tflops = result.tflops };
        if (shape.m == 1) first_row_1 = try init.gpa.dupe(u16, result.output[0..shape.n]);
        if (shape.m == 4096 and shape.n == 6144) {
            first_row_4096 = try init.gpa.dupe(u16, result.output[0..shape.n]);
            first_4096_rows = try init.gpa.dupe(u16, result.output[0 .. 64 * shape.n]);
        }
        if (shape.m == 64 and shape.n == 6144) first_64_rows = try init.gpa.dupe(u16, result.output);
    }
    if (first_row_1 != null and first_row_4096 != null) {
        const equal = std.mem.eql(u16, first_row_1.?, first_row_4096.?);
        std.debug.print("{s} row invariance M=1 vs M=4096 first row\n", .{if (equal) "PASS" else "FAIL"});
    }
    if (first_64_rows != null and first_4096_rows != null) {
        const equal = std.mem.eql(u16, first_64_rows.?, first_4096_rows.?);
        std.debug.print("{s} row invariance M=64 vs M=4096 rows 0-63\n", .{if (equal) "PASS" else "FAIL"});
    }
    var formatted: [1024]u8 = undefined;
    const summary_text = try formatSummary(&formatted, summary[0..supported], "not measured", "see MMA probe", "not measured", "see MMA probe");
    std.debug.print("{s}", .{summary_text});
    std.debug.print("cuBLASLt 128x128 block-scaling support: {s}\n", .{if (backend == .cublaslt) (if (supported > 0) "SUPPORTED" else "UNSUPPORTED") else "not probed in g mode"});
}

test {
    std.testing.refAllDecls(@This());
}
