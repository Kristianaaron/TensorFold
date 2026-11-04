// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GPU parity checks for unpacking and chunked quantization of the packed target lm_head.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const drafts = @import("drafts");

const n = 448;
const k = 6144;
const npad = 512;

pub fn run(gpu: anytype) !void {
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer glue_mod.unload();
    var quant_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    defer quant_mod.unload();
    const unpack = try glue_mod.function("tf_glm53_unpack_bf16_rows");
    const quant = try glm.quant.Kernels.load(quant_mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();

    const src = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(src);
    var rng = std.Random.DefaultPrng.init(0x53d_f1a5);
    for (src) |*v| {
        const f = rng.random().float(f32) * 2 - 1;
        const bits: u32 = @bitCast(f);
        v.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }
    const packed_host = try gpu.gpa.alloc(u16, npad * k);
    defer gpu.gpa.free(packed_host);
    cuda.qmmf.packBf16(packed_host, src, n, k);
    var packed_w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_host));
    defer packed_w.free();

    var full = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
    defer full.free();
    try drafts.unpackRows(unpack, stream, packed_w.ptr, full.ptr, k, 0, n);
    var partial = try cuda.DeviceBuffer.alloc(gpu.d, (n - 192) * k * 2);
    defer partial.free();
    try drafts.unpackRows(unpack, stream, packed_w.ptr, partial.ptr, k, 192, n - 192);
    try stream.synchronize();
    const got_full = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(got_full);
    try full.download(0, std.mem.sliceAsBytes(got_full));
    try expect(std.mem.eql(u8, std.mem.sliceAsBytes(src), std.mem.sliceAsBytes(got_full)), "unpack bit-exact rows [0, 448)", .{});
    const got_partial = try gpu.gpa.alloc(u16, (n - 192) * k);
    defer gpu.gpa.free(got_partial);
    try partial.download(0, std.mem.sliceAsBytes(got_partial));
    try expect(std.mem.eql(u8, std.mem.sliceAsBytes(src[192 * k ..]), std.mem.sliceAsBytes(got_partial)), "unpack bit-exact rows [192, 448)", .{});

    const words_bytes = n * k / 2;
    const meta_bytes = n * k / 32;
    var whole_words = try cuda.DeviceBuffer.alloc(gpu.d, words_bytes);
    defer whole_words.free();
    var whole_scales = try cuda.DeviceBuffer.alloc(gpu.d, meta_bytes);
    defer whole_scales.free();
    var whole_biases = try cuda.DeviceBuffer.alloc(gpu.d, meta_bytes);
    defer whole_biases.free();
    var chunk_words = try cuda.DeviceBuffer.alloc(gpu.d, words_bytes);
    defer chunk_words.free();
    var chunk_scales = try cuda.DeviceBuffer.alloc(gpu.d, meta_bytes);
    defer chunk_scales.free();
    var chunk_biases = try cuda.DeviceBuffer.alloc(gpu.d, meta_bytes);
    defer chunk_biases.free();
    var row_major = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(src));
    defer row_major.free();
    var tmp = try cuda.DeviceBuffer.alloc(gpu.d, 128 * k * 2);
    defer tmp.free();
    try glm.quant.q4(&quant, stream, row_major.ptr, n, k, whole_words.ptr, whole_scales.ptr, whole_biases.ptr);
    try drafts.quantizePacked(unpack, &quant, stream, packed_w.ptr, n, k, 128, tmp.ptr, chunk_words.ptr, chunk_scales.ptr, chunk_biases.ptr);
    try stream.synchronize();
    try expect(try equalBuffers(gpu, whole_words, chunk_words), "chunked q4 words == whole q4", .{});
    try expect(try equalBuffers(gpu, whole_scales, chunk_scales), "chunked q4 scales == whole q4", .{});
    try expect(try equalBuffers(gpu, whole_biases, chunk_biases), "chunked q4 biases == whole q4", .{});
}

fn equalBuffers(gpu: anytype, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer) !bool {
    const aa = try gpu.gpa.alloc(u8, a.len);
    defer gpu.gpa.free(aa);
    const bb = try gpu.gpa.alloc(u8, b.len);
    defer gpu.gpa.free(bb);
    try a.download(0, aa);
    try b.download(0, bb);
    return std.mem.eql(u8, aa, bb);
}

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (!ok) {
        std.debug.print("FAIL " ++ fmt ++ "\n", args);
        return error.TestFailed;
    }
}

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa }) catch |e| {
        std.debug.print("FAIL glm53-draft-head: {t}\n", .{e});
        return 1;
    };
    std.debug.print("PASS glm53-draft-head\n", .{});
    return 0;
}
