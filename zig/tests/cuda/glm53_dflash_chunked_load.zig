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
    std.debug.print("PASS glm-dflash-chunked-load q4 output: {d} rows in {d}-row chunks equals whole linear\n", .{rows, batch_rows});
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
