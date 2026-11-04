// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GPU parity and row-invariance checks for opt-in GLM-5.3 prefill absorb/expand tensor cores.
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("mla.zig");

test "GLM-5.3 prompt MLA tensor-core declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

const Case = struct { absorb: []u8, expand: []u8, absorb_big: []u8, expand_big: []u8, scalar_absorb: []u8, scalar_expand: []u8 };

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    const kernels = try mla.Mla.load(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const codes = try gpu.gpa.alloc(u8, mla.kvb_rows_per_head * mla.latent_dim);
    defer gpu.gpa.free(codes);
    const scales = try gpu.gpa.alloc(f32, 16);
    defer gpu.gpa.free(scales);
    var prng = std.Random.DefaultPrng.init(0x53_4d_4c_41);
    const random = prng.random();
    for (codes) |*v| {
        const e: u8 = random.intRangeAtMost(u8, 4, 9);
        const m: u8 = random.intRangeAtMost(u8, 0, 6);
        v.* = (if (random.boolean()) @as(u8, 0x80) else 0) | (e << 3) | m;
    }
    for (scales) |*v| v.* = 0.03 + random.float(f32) * 0.09;
    var d_codes = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer d_codes.free();
    var d_scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer d_scales.free();

    const row_cases = [_]usize{ 1, 17, 512, 4096 };
    for (row_cases) |rows| {
        const c = try runCase(gpu, stream, &kernels, d_codes.ptr, d_scales.ptr, rows, random);
        defer gpu.gpa.free(c.absorb);
        defer gpu.gpa.free(c.expand);
        defer gpu.gpa.free(c.absorb_big);
        defer gpu.gpa.free(c.expand_big);
        defer gpu.gpa.free(c.scalar_absorb);
        defer gpu.gpa.free(c.scalar_expand);
        try compare(gpu, c.absorb, c.scalar_absorb, rows, "absorb");
        try compare(gpu, c.expand, c.scalar_expand, rows, "expand_v");
        try compareExact(c.absorb, c.absorb_big, rows, mla.latent_dim, "absorb");
        try compareExact(c.expand, c.expand_big, rows, mla.value_dim, "expand_v");
        std.debug.print("RESULT glm53-prompt-mla rows={d}: absorb and expand_v parity checked\n", .{rows});
    }

    try runDeployedShape(gpu, stream, &kernels, random, 4096);
    try runDeployedShape(gpu, stream, &kernels, random, 4095);
    try runDeployedShape(gpu, stream, &kernels, random, 100);

    try invariance(gpu, stream, &kernels, d_codes.ptr, d_scales.ptr, random, false, .tile16);
    try invariance(gpu, stream, &kernels, d_codes.ptr, d_scales.ptr, random, true, .tile16);
    try invariance(gpu, stream, &kernels, d_codes.ptr, d_scales.ptr, random, false, .tile64);
    try invariance(gpu, stream, &kernels, d_codes.ptr, d_scales.ptr, random, true, .tile64);
}

fn runDeployedShape(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, random: std.Random, rows: usize) !void {
    const local_heads = 32;
    const kvb_rows = local_heads * mla.kvb_rows_per_head;
    const code_data = try gpu.gpa.alloc(u8, kvb_rows * mla.latent_dim);
    defer gpu.gpa.free(code_data);
    const scale_data = try gpu.gpa.alloc(f32, (kvb_rows / 128) * 4);
    defer gpu.gpa.free(scale_data);
    for (code_data) |*v| {
        const e: u8 = random.intRangeAtMost(u8, 4, 9);
        const m: u8 = random.intRangeAtMost(u8, 0, 6);
        v.* = (if (random.boolean()) @as(u8, 0x80) else 0) | (e << 3) | m;
    }
    for (scale_data) |*v| v.* = 0.03 + random.float(f32) * 0.09;
    var d_codes = try cuda.DeviceBuffer.fromHost(gpu.d, code_data);
    defer d_codes.free();
    var d_scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scale_data));
    defer d_scales.free();

    const q = try gpu.gpa.alloc(u16, rows * local_heads * 256);
    defer gpu.gpa.free(q);
    const ol = try gpu.gpa.alloc(u16, rows * local_heads * mla.latent_dim);
    defer gpu.gpa.free(ol);
    for (q) |*v| v.* = bf16Bits((random.float(f32) - 0.5) * 0.5);
    for (ol) |*v| v.* = bf16Bits((random.float(f32) - 0.5) * 0.5);
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dq.free();
    var dol = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ol));
    defer dol.free();
    var da = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.latent_dim * 2);
    defer da.free();
    var das = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.latent_dim * 2);
    defer das.free();
    var dv = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.value_dim * 2);
    defer dv.free();
    var dvs = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.value_dim * 2);
    defer dvs.free();
    var dab = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.latent_dim * 2);
    defer dab.free();
    var dvb = try cuda.DeviceBuffer.alloc(gpu.d, rows * local_heads * mla.value_dim * 2);
    defer dvb.free();

    try kernels.absorbPrompt(stream, dq.ptr, d_codes.ptr, d_scales.ptr, da.ptr, rows, local_heads, 0, .tile16);
    try kernels.absorb(stream, dq.ptr, d_codes.ptr, d_scales.ptr, das.ptr, rows, local_heads, 0, false);
    try kernels.expandVPrompt(stream, dol.ptr, d_codes.ptr, d_scales.ptr, dv.ptr, rows, local_heads, 0, .tile16);
    try kernels.absorbPrompt(stream, dq.ptr, d_codes.ptr, d_scales.ptr, dab.ptr, rows, local_heads, 0, .tile64);
    try kernels.expandVPrompt(stream, dol.ptr, d_codes.ptr, d_scales.ptr, dvb.ptr, rows, local_heads, 0, .tile64);
    try kernels.expandV(stream, dol.ptr, d_codes.ptr, d_scales.ptr, dvs.ptr, rows, local_heads, 0, false);
    try stream.synchronize();
    const absorb = try download(gpu, da);
    defer gpu.gpa.free(absorb);
    const scalar_absorb = try download(gpu, das);
    defer gpu.gpa.free(scalar_absorb);
    const expand = try download(gpu, dv);
    defer gpu.gpa.free(expand);
    const scalar_expand = try download(gpu, dvs);
    defer gpu.gpa.free(scalar_expand);
    try compare(gpu, absorb, scalar_absorb, rows, "absorb");
    try compare(gpu, expand, scalar_expand, rows, "expand_v");
    const absorb_big = try download(gpu, dab);
    defer gpu.gpa.free(absorb_big);
    const expand_big = try download(gpu, dvb);
    defer gpu.gpa.free(expand_big);
    try compareExact(absorb, absorb_big, rows * local_heads, mla.latent_dim, "deployed absorb");
    try compareExact(expand, expand_big, rows * local_heads, mla.value_dim, "deployed expand_v");
    std.debug.print("RESULT glm53-prompt-mla deployed shape rows={d} local_heads={d} kvb_rows={d} scale_blocks={d}: parity checked\n", .{ rows, local_heads, kvb_rows, scale_data.len / 4 });
    if (rows == 4096) {
        const absorb_1 = try benchPrompt(gpu, stream, kernels, dq.ptr, d_codes.ptr, d_scales.ptr, da.ptr, rows, local_heads, false, .tile16);
        const absorb_2 = try benchPrompt(gpu, stream, kernels, dq.ptr, d_codes.ptr, d_scales.ptr, dab.ptr, rows, local_heads, false, .tile64);
        const expand_1 = try benchPrompt(gpu, stream, kernels, dol.ptr, d_codes.ptr, d_scales.ptr, dv.ptr, rows, local_heads, true, .tile16);
        const expand_2 = try benchPrompt(gpu, stream, kernels, dol.ptr, d_codes.ptr, d_scales.ptr, dvb.ptr, rows, local_heads, true, .tile64);
        std.debug.print("BENCH glm53-prompt-mla 32 heads x 4096 rows: absorb =1 {d:.2} us =2 {d:.2} us {d:.2}x | expand_v =1 {d:.2} us =2 {d:.2} us {d:.2}x\n", .{ absorb_1, absorb_2, absorb_1 / absorb_2, expand_1, expand_2, expand_1 / expand_2 });
    }
}

fn benchPrompt(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, input: u64, codes: u64, scales: u64, output: u64, rows: usize, local_heads: usize, expand: bool, mode: mla.PromptMlaMode) !f32 {
    const reps = 10;
    for (0..2) |_| {
        if (expand) try kernels.expandVPrompt(stream, input, codes, scales, output, rows, local_heads, 0, mode) else try kernels.absorbPrompt(stream, input, codes, scales, output, rows, local_heads, 0, mode);
    }
    try stream.synchronize();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    try start.record(stream);
    for (0..reps) |_| {
        if (expand) try kernels.expandVPrompt(stream, input, codes, scales, output, rows, local_heads, 0, mode) else try kernels.absorbPrompt(stream, input, codes, scales, output, rows, local_heads, 0, mode);
    }
    try stop.record(stream);
    try stream.synchronize();
    return (try start.elapsedMs(stop)) * 100.0;
}

fn runCase(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, codes: u64, scales: u64, rows: usize, random: std.Random) !Case {
    const qwidth = 256;
    const inwidth = mla.latent_dim;
    const q = try gpu.gpa.alloc(u16, rows * qwidth);
    defer gpu.gpa.free(q);
    const ol = try gpu.gpa.alloc(u16, rows * inwidth);
    defer gpu.gpa.free(ol);
    for (q) |*v| v.* = bf16Bits((random.float(f32) - 0.5) * 0.5);
    for (ol) |*v| v.* = bf16Bits((random.float(f32) - 0.5) * 0.5);
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dq.free();
    var dol = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ol));
    defer dol.free();
    var da = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.latent_dim * 2);
    defer da.free();
    var das = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.latent_dim * 2);
    defer das.free();
    var dv = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.value_dim * 2);
    defer dv.free();
    var dvs = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.value_dim * 2);
    defer dvs.free();
    var dab = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.latent_dim * 2);
    defer dab.free();
    var dvb = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.value_dim * 2);
    defer dvb.free();
    try kernels.absorbPrompt(stream, dq.ptr, codes, scales, da.ptr, rows, 1, 0, .tile16);
    try kernels.absorb(stream, dq.ptr, codes, scales, das.ptr, rows, 1, 0, false);
    try kernels.expandVPrompt(stream, dol.ptr, codes, scales, dv.ptr, rows, 1, 0, .tile16);
    try kernels.absorbPrompt(stream, dq.ptr, codes, scales, dab.ptr, rows, 1, 0, .tile64);
    try kernels.expandVPrompt(stream, dol.ptr, codes, scales, dvb.ptr, rows, 1, 0, .tile64);
    try kernels.expandV(stream, dol.ptr, codes, scales, dvs.ptr, rows, 1, 0, false);
    try stream.synchronize();
    return .{ .absorb = try download(gpu, da), .expand = try download(gpu, dv), .absorb_big = try download(gpu, dab), .expand_big = try download(gpu, dvb), .scalar_absorb = try download(gpu, das), .scalar_expand = try download(gpu, dvs) };
}

fn compareExact(tile16: []const u8, tile64: []const u8, rows: usize, width: usize, label: []const u8) !void {
    if (tile16.len != tile64.len or tile16.len != rows * width * 2) return error.BadShape;
    if (!std.mem.eql(u8, tile16, tile64)) {
        const a = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, tile16));
        const b = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, tile64));
        for (a, b, 0..) |x, y, i| if (x != y) {
            std.debug.print("FAIL glm53-prompt-mla {s} rows={d}: first differing bf16 at {d}: {x:0>4} vs {x:0>4}\n", .{ label, rows, i, x, y });
            return error.BitMismatch;
        };
    }
    std.debug.print("PASS glm53-prompt-mla {s} rows={d}: tile64 bit-identical to tile16\n", .{ label, rows });
}

fn compare(gpu: anytype, got: []const u8, ref: []const u8, rows: usize, label: []const u8) !void {
    const ga = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got));
    const ra = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, ref));
    if (ga.len != ra.len) return error.BadShape;
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var max_small_abs: f32 = 0;
    for (ga, ra) |g, r| {
        const gv = bf16Value(g);
        const rv = bf16Value(r);
        const abs = @abs(gv - rv);
        max_abs = @max(max_abs, abs);
        if (@abs(rv) >= 0.05) {
            max_rel = @max(max_rel, abs / @abs(rv));
        } else {
            max_small_abs = @max(max_small_abs, abs);
        }
    }
    std.debug.print("RESULT glm53-prompt-mla {s} rows={d}: max_abs={e:.3} max_rel(|ref|>=0.05)={e:.3}\n", .{ label, rows, max_abs, max_rel });
    // FP32 MMA changes the accumulation tree versus the scalar serial FMA; BF16 output adds one final rounding.
    if (max_small_abs > 0.01 or max_rel > 0.01) {
        for (ga, ra, 0..) |g, r, idx| {
            const gv = bf16Value(g);
            const rv = bf16Value(r);
            const bad = if (@abs(rv) >= 0.05) @abs(gv - rv) / @abs(rv) > 0.01 else @abs(gv - rv) > 0.01;
            if (bad) {
                std.debug.print("FAIL glm53-prompt-mla {s} rows={d}: first bad element {d} (row {d}, col {d}): got {e:.4} want {e:.4}; max_abs {e:.3} max_rel {e:.3}\n", .{ label, rows, idx, idx / (ga.len / rows), idx % (ga.len / rows), gv, rv, max_abs, max_rel });
                break;
            }
        }
        std.debug.print("FAIL glm53-prompt-mla {s} rows={d}: tolerance relative 1e-2 above 0.05, absolute 1e-2 below\n", .{ label, rows });
        return error.TestFailed;
    }
    _ = gpu;
    std.debug.print("PASS glm53-prompt-mla {s} rows={d}: FP32 accumulation order tolerance relative 1e-2 above 0.05, absolute 1e-2 below\n", .{ label, rows });
}

fn invariance(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, codes: u64, scales: u64, random: std.Random, expand: bool, mode: mla.PromptMlaMode) !void {
    const width: usize = if (expand) mla.latent_dim else 256;
    const out_width: usize = if (expand) mla.value_dim else mla.latent_dim;
    const source = try gpu.gpa.alloc(u16, width);
    defer gpu.gpa.free(source);
    for (source) |*v| v.* = bf16Bits((random.float(f32) - 0.5) * 0.5);
    const alone = try runInvariantBatch(gpu, stream, kernels, codes, scales, source, 1, width, out_width, expand, 0, mode);
    defer gpu.gpa.free(alone);
    const locations = [_][2]usize{ .{ 17, 9 }, .{ 512, 255 } };
    for (locations) |spec| {
        const batch = try runInvariantBatch(gpu, stream, kernels, codes, scales, source, spec[0], width, out_width, expand, spec[1], mode);
        defer gpu.gpa.free(batch);
        const start = spec[1] * out_width * 2;
        if (!std.mem.eql(u8, alone, batch[start .. start + alone.len])) {
            std.debug.print("FAIL glm53-prompt-mla row invariance op={s} rows={d} row={d}\n", .{ if (expand) "expand_v" else "absorb", spec[0], spec[1] });
            return error.RowInvariant;
        }
    }
    std.debug.print("PASS glm53-prompt-mla row invariance op={s}: alone, offset 9 in 17, offset 255 in 512 bit-exact\n", .{if (expand) "expand_v" else "absorb"});
}

fn runInvariantBatch(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, codes: u64, scales: u64, source: []const u16, rows: usize, width: usize, out_width: usize, expand: bool, target: usize, mode: mla.PromptMlaMode) ![]u8 {
    const input = try gpu.gpa.alloc(u16, rows * width);
    defer gpu.gpa.free(input);
    @memset(input, 0);
    @memcpy(input[target * width ..][0..width], source);
    var di = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(input));
    defer di.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, rows * out_width * 2);
    defer dout.free();
    if (expand) try kernels.expandVPrompt(stream, di.ptr, codes, scales, dout.ptr, rows, 1, 0, mode) else try kernels.absorbPrompt(stream, di.ptr, codes, scales, dout.ptr, rows, 1, 0, mode);
    try stream.synchronize();
    return download(gpu, dout);
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn bf16Bits(value: f32) u16 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn bf16Value(value: u16) f32 {
    return @as(f32, @bitCast(@as(u32, value) << 16));
}
