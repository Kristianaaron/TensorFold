// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! DFlash row chunking: full-linear and four row batches produce identical q4 storage.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");

pub fn run(gpu: anytype) !void {
    _ = glm.dflash_load.CHUNK; // Keep the actual loader module in this test's build graph.
    const rows: usize = 500;
    const k: usize = 1024;
    const batch_rows: usize = 128;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var qm = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    defer qm.unload();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer pm.unload();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer gm.unload();
    const qk = try glm.quant.Kernels.load(qm);
    const pk = try glm.q4.Kernels.load(gm, pm, gpu.ctx);

    const host = try gpu.gpa.alloc(u16, rows * k);
    defer gpu.gpa.free(host);
    for (host, 0..) |*v, i| v.* = @truncate((i * 37 + i / k * 13) & 0xffff);
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    defer input.free();

    const raw_words_n = rows * k / 2;
    const raw_scale_n = rows * k / 32;
    var whole_words = try cuda.DeviceBuffer.alloc(gpu.d, raw_words_n + raw_scale_n * 2 + 1024);
    defer whole_words.free();
    const whole_scales = whole_words.ptr + std.mem.alignForward(u64, raw_words_n, 256);
    const whole_biases = whole_scales + std.mem.alignForward(u64, raw_scale_n, 256);
    var whole = try allocPacked(gpu.d, rows, k);
    defer whole.deinit();
    try glm.quant.q4(&qk, stream, input.ptr, rows, k, whole_words.ptr, whole_scales, whole_biases);
    try glm.q4.pack(&pk, stream, whole_words.ptr, whole_scales, whole_biases, whole.q);
    try stream.synchronize();

    const chunk_raw_words_n = batch_rows * k / 2;
    const chunk_raw_scale_n = batch_rows * k / 32;
    var chunk_words = try cuda.DeviceBuffer.alloc(gpu.d, chunk_raw_words_n + chunk_raw_scale_n * 2 + 1024);
    defer chunk_words.free();
    var chunk_input = try cuda.DeviceBuffer.alloc(gpu.d, batch_rows * k * 2);
    defer chunk_input.free();
    var chunked = try allocPacked(gpu.d, rows, k);
    defer chunked.deinit();
    var chunk_packed = try allocPacked(gpu.d, batch_rows, k);
    defer chunk_packed.deinit();
    try chunked.words.fill8(0xA5, null);
    try chunked.scales.fill8(0xA5, null);
    try chunked.biases.fill8(0xA5, null);
    try std.testing.expectError(error.BadChunkSize, glm.dflash_load.quantizeLinearChunked(
        host[0 .. 96 * k], rows, k, 0, 96, stream, &qk, &pk, &chunk_input, &chunk_words, chunk_packed.q, chunked.q,
    ));
    var first: usize = 0;
    while (first < rows) {
        const count = @min(batch_rows, rows - first);
        try glm.dflash_load.quantizeLinearChunked(
            host[first * k ..][0 .. count * k],
            rows,
            k,
            first,
            batch_rows,
            stream,
            &qk,
            &pk,
            &chunk_input,
            &chunk_words,
            .{ .w = chunk_packed.q.w, .s = chunk_packed.q.s, .b = chunk_packed.q.b, .n = count, .k = k, .npad = glm.q4.npadOf(count) },
            chunked.q,
        );
        first += count;
    }
    try equal(gpu, whole.words, chunked.words);
    try equal(gpu, whole.scales, chunked.scales);
    try equal(gpu, whole.biases, chunked.biases);
    try fp32Out(gpu);
    std.debug.print("PASS glm-dflash-chunked-load q4 output: {d} rows in {d}-row chunks equals whole linear\n", .{rows, batch_rows});
    try rowsIndependent(gpu);
}

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn rowsIndependent(gpu: anytype) !void {
    const a = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer gm.unload();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer pm.unload();
    const pk = try glm.q4.Kernels.load(gm, pm, gpu.ctx);
    const shapes = [_]struct { n: usize, k: usize, seed: usize }{
        .{ .n = 6144, .k = 30720, .seed = 17 },
        .{ .n = 8192, .k = 6144, .seed = 29 },
    };
    for (shapes) |shape| {
        const qbytes = glm.q4.bytes(shape.n, shape.k);
        const weight_host = try a.alloc(u8, qbytes[0]);
        defer a.free(weight_host);
        for (weight_host, 0..) |*v, i| v.* = @truncate((i * 37 + shape.seed * 101) & 0xff);
        const scale_host = try a.alloc(u16, qbytes[1] / @sizeOf(u16));
        defer a.free(scale_host);
        @memset(scale_host, bf16(0.125));
        const bias_host = try a.alloc(u16, qbytes[2] / @sizeOf(u16));
        defer a.free(bias_host);
        @memset(bias_host, bf16(0.01));
        var weight = try cuda.DeviceBuffer.fromHost(gpu.d, weight_host);
        defer weight.free();
        var scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scale_host));
        defer scales.free();
        var biases = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bias_host));
        defer biases.free();
        const q: glm.q4.Q4 = .{ .w = weight.ptr, .s = scales.ptr, .b = biases.ptr, .n = shape.n, .k = shape.k, .npad = glm.q4.npadOf(shape.n) };

        const x_host = try a.alloc(u16, 8 * shape.k);
        defer a.free(x_host);
        for (x_host, 0..) |*v, i| {
            const value = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16.0;
            v.* = bf16(value);
        }
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x_host));
        defer x.free();
        const xs_host = try a.alloc(f32, 8 * (shape.k / 64));
        defer a.free(xs_host);
        for (xs_host, 0..) |*v, i| v.* = 0.25 + @as(f32, @floatFromInt(i % 13)) / 32.0;
        var xs = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs_host));
        defer xs.free();
        var wide = try cuda.DeviceBuffer.alloc(gpu.d, 8 * shape.n * @sizeOf(u16));
        defer wide.free();
        var short = try cuda.DeviceBuffer.alloc(gpu.d, 3 * shape.n * @sizeOf(u16));
        defer short.free();
        try glm.q4.matmul(&pk, stream, x.ptr, shape.k, xs.ptr, q, wide.ptr, 8);
        try glm.q4.matmul(&pk, stream, x.ptr, shape.k, xs.ptr, q, short.ptr, 3);
        try stream.synchronize();
        const wide_host = try a.alloc(u16, 3 * shape.n);
        defer a.free(wide_host);
        const short_host = try a.alloc(u16, 3 * shape.n);
        defer a.free(short_host);
        try wide.download(0, std.mem.sliceAsBytes(wide_host));
        try short.download(0, std.mem.sliceAsBytes(short_host));
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(wide_host), std.mem.sliceAsBytes(short_host))) return error.TestFailed;
    }
    std.debug.print("PASS glm-dflash rows independent: 8-row Q4 outputs match 3-row prefixes for fc and kv shapes\n", .{});
}

fn fp32Out(gpu: anytype) !void {
    const n: usize = 6144;
    const k: usize = 6144;
    const max_rows: usize = 8;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var qm = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    defer qm.unload();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer pm.unload();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer gm.unload();
    var dm = try cuda.Module.load(gpu.d, cuda.kernels.glm_dflash);
    defer dm.unload();
    const quant_kernels = try glm.quant.Kernels.load(qm);
    var q4_kernels = try glm.q4.Kernels.load(gm, pm, gpu.ctx);
    try q4_kernels.loadF32(gm);
    const df_kernels = try glm.dflash.Kernels.load(dm, .{});

    const host_input = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(host_input);
    var state: u32 = 0x6d2b79f5;
    for (host_input) |*value| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const centered: i32 = @as(i32, @intCast(state & 0xffff)) - 32768;
        value.* = f32ToBf16(@as(f32, @floatFromInt(centered)) / 32768.0);
    }
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_input));
    defer input.free();
    const raw_words_n = n * k / 2;
    const raw_scale_n = n * k / 32;
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, raw_words_n + raw_scale_n * 2 + 1024);
    defer raw.free();
    const scales = raw.ptr + std.mem.alignForward(u64, raw_words_n, 256);
    const biases = scales + std.mem.alignForward(u64, raw_scale_n, 256);
    var quantized = try allocPacked(gpu.d, n, k);
    defer quantized.deinit();
    try glm.quant.q4(&quant_kernels, stream, input.ptr, n, k, raw.ptr, scales, biases);
    try glm.q4.pack(&q4_kernels, stream, raw.ptr, scales, biases, quantized.q);

    var sums = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * (k / 64) * 4);
    defer sums.free();
    var bf16_output = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 2);
    defer bf16_output.free();
    var f32_output = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 4);
    defer f32_output.free();
    const rows_to_check = [_]usize{ 1, 3, 8 };
    for (rows_to_check) |rows| {
        var args: cuda.Args = .{};
        args.add(input.ptr);
        args.add(@as(i32, @intCast(k)));
        args.add(@as(i32, @intCast(k)));
        args.add(sums.ptr);
        try cuda.launch.launch(df_kernels.gsum, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &args);
        try glm.q4.matmul(&q4_kernels, stream, input.ptr, k, sums.ptr, quantized.q, bf16_output.ptr, rows);
        try glm.q4.matmulF32(&q4_kernels, stream, input.ptr, k, sums.ptr, quantized.q, f32_output.ptr, rows);
        try stream.synchronize();
        const wide = try gpu.gpa.alloc(f32, rows * n);
        defer gpu.gpa.free(wide);
        const narrow = try gpu.gpa.alloc(u16, rows * n);
        defer gpu.gpa.free(narrow);
        try f32_output.download(0, std.mem.sliceAsBytes(wide));
        try bf16_output.download(0, std.mem.sliceAsBytes(narrow));
        for (wide, narrow) |value, actual| try std.testing.expectEqual(actual, f32ToBf16(value));
    }
}

fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

const Packed = struct {
    words: cuda.DeviceBuffer,
    scales: cuda.DeviceBuffer,
    biases: cuda.DeviceBuffer,
    q: glm.q4.Q4,

    fn deinit(p: *Packed) void {
        p.words.free();
        p.scales.free();
        p.biases.free();
    }
};

fn allocPacked(d: *const cuda.Driver, n: usize, k: usize) !Packed {
    const sizes = glm.q4.bytes(n, k);
    var w = try cuda.DeviceBuffer.alloc(d, sizes[0]);
    errdefer w.free();
    var s = try cuda.DeviceBuffer.alloc(d, sizes[1]);
    errdefer s.free();
    var b = try cuda.DeviceBuffer.alloc(d, sizes[2]);
    errdefer b.free();
    return .{
        .words = w,
        .scales = s,
        .biases = b,
        .q = .{ .w = w.ptr, .s = s.ptr, .b = b.ptr, .n = n, .k = k, .npad = glm.q4.npadOf(n) },
    };
}

fn equal(gpu: anytype, left: cuda.DeviceBuffer, right: cuda.DeviceBuffer) !void {
    const a = try gpu.gpa.alloc(u8, left.len);
    defer gpu.gpa.free(a);
    const b = try gpu.gpa.alloc(u8, right.len);
    defer gpu.gpa.free(b);
    try left.download(0, a);
    try right.download(0, b);
    if (!std.mem.eql(u8, a, b)) {
        const first = std.mem.indexOfDiff(u8, a, b).?;
        std.debug.print("FAIL dflash chunked q4: first difference at byte {d}\n", .{first});
        return error.TestFailed;
    }
}
