// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 W8A8 prefill projection (kernels/cuda/glm_dsa/w8a8.cu).
//! TF_GLM53_PREFILL_W8A8=1 runs o_proj. =2 reuses these two kernels for every other FP8B prefill
//! projection `serves` accepts. Activations: bf16 rows -> e4m3 with one fp32 scale per (row, 128-K block), amax / 448, row-local.
//! Weights: the checkpoint's FP8B projection exactly as qmmf reads it (cuda/fp8.zig packCodes + packBlockScales),
//! with no repack. GEMM: mma.sync m16n8k32 e4m3, fp32 accumulators, one FP32 promotion per 64-K weight tile.
//! Provenance: fp8_mma.cu (adapted from ashhart/TensorFold@d1018b31 ck_lane.cu, Apache-2.0) and qmmf.cu
//! (adapted from bertholomus/TensorFold@ddfad35, Apache-2.0); see the kernel file header.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const qmmf = @import("qmmf.zig");
const fp8 = @import("fp8.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// K of one activation scale.
pub const block_k: usize = 128;
/// Rows and columns of one GEMM block tile (w8a8.cu BM, BN).
pub const tile_m: usize = 128;
pub const tile_n: usize = 128;
pub const threads: u32 = 256;
/// 4 stages x (8 KiB activations + 8 KiB weights): w8a8.cu SMEM.
pub const shared_bytes: u32 = 4 * (128 * 64 + 128 * 64);
/// 128-K blocks one quantizer CTA covers (one warp each).
pub const quant_blocks: usize = 8;
/// Largest K of a level-2 projection (hidden). o_proj's K is 8192 and keeps the attention-partial scratch.
pub const level2_k: usize = 6144;

pub const quant_symbol: [:0]const u8 = "tf_glm53_w8a8_quant";
pub const gemm_symbol: [:0]const u8 = "tf_glm53_w8a8_gemm";

/// Where the e4m3 rows [m][k] and the scales [k/128][m] sit in one scratch span.
pub const Scratch = struct { xq: u64, xs: u64 };

fn codeBytes(m: usize, k: usize) usize {
    return std.mem.alignForward(usize, m * k, 256);
}

/// Scratch a call of m rows and k inputs needs: e4m3 codes, then fp32 scales.
pub fn scratchBytes(m: usize, k: usize) usize {
    return codeBytes(m, k) + (k / block_k) * m * 4;
}

pub fn scratchAt(base: u64, m: usize, k: usize) Scratch {
    return .{ .xq = base, .xs = base + codeBytes(m, k) };
}

/// The projections this kernel reads: FP8B, whole 128-K blocks, an even output count, 128-row padding.
pub fn supports(w: qmmf.Weight) bool {
    return w.mode == .fp8b and w.k > 0 and w.k % block_k == 0 and w.n > 0 and w.n % 2 == 0 and w.npad % tile_n == 0 and w.npad >= w.n;
}

/// Level 2 projections. The GEMM arguments are already runtime M, N, K and the fp32/bf16 flag, so no new
/// entry point is required. K above `level2_k` (o_proj) stays on the attention-partial scratch.
pub fn serves(w: qmmf.Weight) bool {
    return supports(w) and @as(usize, @intCast(w.k)) <= level2_k;
}

/// Outputs below this stay on the release lane: measured on GB10 (glm53-w8a8-all, 512 rows), N 32..576 ran
/// 0.77-0.99x of release and N 1024..12288 ran 1.29-1.86x.
pub const level2_min_n: usize = 1024;

/// The level-2 projections the engine routes to W8A8: `serves` minus the narrow outputs that measured slower.
pub fn servesFast(w: qmmf.Weight) bool {
    return serves(w) and @as(usize, @intCast(w.n)) >= level2_min_n;
}

/// GEMM blocks: row tiles x column tiles, one dimension (w8a8.cu takes bands of 8 row tiles).
pub fn gemmBlocks(m: usize, n: usize) usize {
    return ((m + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n);
}

/// The two kernels of one call; a value type so Forward's kernel table can hold it (the module stays with Kernels).
pub const Fns = struct {
    quant: Function,
    gemm: Function,

    /// x bf16 [m][ldx] -> xq e4m3 [m][k] (fragment K order), xs fp32 [k/128][m].
    pub fn quantize(f: Fns, s: Stream, x: u64, ldx: usize, m: usize, k: usize, xq: u64, xs: u64) !void {
        if (k == 0 or k % block_k != 0 or ldx < k or ldx % 2 != 0 or x % 4 != 0 or m > 65535) return error.W8a8Shape;
        if (m == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(ldx)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(k)));
        a.add(xq);
        a.add(xs);
        const blocks = (k / block_k + quant_blocks - 1) / quant_blocks;
        try launch_.launch(f.quant, .{ .grid = .{ .x = @intCast(blocks), .y = @intCast(m), .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    /// y [m][w.n] (`out` fp32 or bf16, rows w.n apart) from quantized rows and the FP8B weight.
    pub fn gemmRows(f: Fns, s: Stream, xq: u64, xs: u64, w: qmmf.Weight, y: u64, m: usize, out: qmmf.Out) !void {
        if (!supports(w)) return error.W8a8Weight;
        if (m == 0) return;
        var a: launch_.Args = .{};
        a.add(xq);
        a.add(xs);
        a.add(w.codes);
        a.add(w.scales);
        a.add(y);
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(w.n)));
        a.add(@as(i32, @intCast(w.k)));
        a.add(@as(i32, if (out == .f32) 1 else 0));
        try launch_.launch(f.gemm, .{ .grid = .{ .x = @intCast(gemmBlocks(m, w.n)), .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = shared_bytes }, s, &a);
    }

    /// Quantize then multiply: the same inputs and output as qmmf's matmulOut, scratch in place of its partials.
    pub fn run(f: Fns, s: Stream, x: u64, ldx: usize, m: usize, w: qmmf.Weight, y: u64, out: qmmf.Out, scratch: u64, scratch_len: u64) !void {
        if (!supports(w)) return error.W8a8Weight;
        const k: usize = w.k;
        if (scratch_len < scratchBytes(m, k)) return error.W8a8Scratch;
        const at = scratchAt(scratch, m, k);
        try f.quantize(s, x, ldx, m, k, at.xq, at.xs);
        try f.gemmRows(s, at.xq, at.xs, w, y, m, out);
    }
};

/// The W8A8 image's module and its kernels.
pub const Kernels = struct {
    mod: Module,
    fns: Fns,

    /// Load from a fatbin image (kernels.glm53_w8a8, or a bring-up fatbin file's bytes).
    pub fn load(d: *const Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, image);
        errdefer mod.unload();
        return fromModule(mod);
    }

    /// Takes ownership of `mod` on success.
    pub fn fromModule(mod: Module) !Kernels {
        const gemm = try mod.function(gemm_symbol);
        try gemm.allowDynamicShared(shared_bytes);
        return .{ .mod = mod, .fns = .{ .quant = try mod.function(quant_symbol), .gemm = gemm } };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

/// Byte offset inside a row of xq of input k: each 16-K group stores K 2q, 2q+1, 2q+8, 2q+9 at bytes 4q..4q+3
/// (fp8_mma.cu tf_glm53_xoff), the K order fp8.zig's packer gives FP8B B lane q.
pub fn activationOffset(k: usize) usize {
    const local = k % 16;
    return (k / 16) * 16 + 4 * ((local % 8) / 2) + (local % 2) + 2 * (local / 8);
}

/// e4m3 (fn) value of a code; 0x7f / 0xff are NaN.
pub fn e4m3Value(code: u8) f32 {
    const sign: f32 = if (code & 0x80 != 0) -1 else 1;
    const exponent: i32 = (code >> 3) & 0xF;
    const mantissa: f32 = @floatFromInt(code & 7);
    if (exponent == 15 and code & 7 == 7) return std.math.nan(f32);
    if (exponent == 0) return sign * std.math.scalbn(mantissa, -9);
    return sign * std.math.scalbn(8 + mantissa, exponent - 10);
}

/// cvt.rn.satfinite.e4m3 of one fp32 value: nearest, ties to the even code, magnitudes past 448 to 448, sign kept.
pub fn e4m3Nearest(x: f32) u8 {
    if (std.math.isNan(x)) return 0x7f;
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a: f64 = @abs(@as(f64, x));
    if (a >= 448) return sign | 0x7e;
    // Largest code at or below a; codes 0x00..0x7e increase with value.
    var lo: u8 = 0;
    var hi: u8 = 0x7e;
    while (lo < hi) {
        const mid: u8 = lo + (hi - lo + 1) / 2;
        if (@as(f64, e4m3Value(mid)) <= a) lo = mid else hi = mid - 1;
    }
    const below: f64 = e4m3Value(lo);
    const above: f64 = e4m3Value(lo + 1);
    const down = a - below;
    const up = above - a;
    const code: u8 = if (down < up) lo else if (up < down) lo + 1 else if (lo % 2 == 0) lo else lo + 1;
    return sign | code;
}

/// The quantizer's numerics for one row of bf16 values (as f32): codes [k] in xq order, scales [k/128].
pub fn quantizeRow(x: []const f32, codes: []u8, scales: []f32) error{W8a8Shape}!void {
    const k = x.len;
    if (k == 0 or k % block_k != 0 or codes.len != k or scales.len != k / block_k) return error.W8a8Shape;
    for (0..k / block_k) |b| {
        var amax: f32 = 0;
        for (x[b * block_k ..][0..block_k]) |v| amax = @max(amax, @abs(v));
        const scale: f32 = if (amax > 0) amax / 448.0 else 1.0;
        scales[b] = scale;
        for (0..block_k) |i| {
            const kk = b * block_k + i;
            codes[activationOffset(kk)] = e4m3Nearest(x[kk] / scale);
        }
    }
}

test "e4m3: every finite code round trips, ties go to the even code, saturation keeps the sign" {
    for (0..256) |c| {
        const code: u8 = @intCast(c);
        if (code & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(code, e4m3Nearest(e4m3Value(code)));
    }
    try std.testing.expectEqual(@as(u8, 0x38), e4m3Nearest(1.0625)); // between 1 (0x38) and 1.125 (0x39)
    try std.testing.expectEqual(@as(u8, 0x3a), e4m3Nearest(1.1875)); // between 0x39 and 0x3a
    try std.testing.expectEqual(@as(u8, 0x00), e4m3Nearest(std.math.scalbn(@as(f32, 1), -10)));
    try std.testing.expectEqual(@as(u8, 0x02), e4m3Nearest(3 * std.math.scalbn(@as(f32, 1), -10)));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3Nearest(1000));
    try std.testing.expectEqual(@as(u8, 0xfe), e4m3Nearest(-std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 0x80), e4m3Nearest(-1e-30));
    try std.testing.expectEqual(@as(f32, 448), e4m3Value(0x7e));
    try std.testing.expectEqual(std.math.scalbn(@as(f32, 1), -9), e4m3Value(0x01));
}

test "activation K order is the FP8B B lane order: one fragment register pairs the same K on both sides" {
    // Every (row, input) of a 64x64 FP8B tile lands at (row group a, lane, k32 half h, byte b). The MMA reads byte
    // b of lane l as logical K 4 (l % 4) + b % 4 + 16 (b / 4) of half h; the activation row must hold the same input there.
    var seen: [64]bool = @splat(false);
    for (0..64) |row| for (0..64) |col| {
        const idx = fp8.packedCodeIndex(row, col, 64);
        const b = idx % 8;
        const h = (idx / 8) % 2;
        const lane = (idx / 16) % 32;
        const a = (idx / 512) % 8;
        try std.testing.expectEqual(row / 8, a);
        try std.testing.expectEqual(row % 8, lane / 4);
        try std.testing.expectEqual(h * 32 + 16 * (b / 4) + 4 * (lane % 4) + b % 4, activationOffset(col));
        if (row == 0) {
            try std.testing.expect(!seen[activationOffset(col)]);
            seen[activationOffset(col)] = true;
        }
    };
    try std.testing.expectEqual(@as(usize, 128 + 4 * 3 + 1 + 2), activationOffset(128 + 15));
}

test "host quantizer: per-block amax / 448, codes in xq order, zero blocks scale one" {
    var x: [256]f32 = @splat(0);
    x[3] = -2.5;
    x[17] = 0.25;
    var codes: [256]u8 = undefined;
    var scales: [2]f32 = undefined;
    try quantizeRow(&x, &codes, &scales);
    try std.testing.expectEqual(@as(f32, 2.5 / 448.0), scales[0]);
    try std.testing.expectEqual(@as(f32, 1), scales[1]);
    try std.testing.expectEqual(@as(u8, 0xfe), codes[activationOffset(3)]);
    try std.testing.expectEqual(e4m3Nearest(0.25 / (2.5 / 448.0)), codes[activationOffset(17)]);
    try std.testing.expectEqual(@as(u8, 0), codes[activationOffset(200)]);
    try std.testing.expectError(error.W8a8Shape, quantizeRow(x[0..100], codes[0..100], scales[0..1]));
}

test "launch geometry, scratch and the weights the kernel takes" {
    try std.testing.expectEqual(@as(u32, 65536), shared_bytes);
    try std.testing.expectEqual(@as(usize, 192), gemmBlocks(512, 6144));
    try std.testing.expectEqual(@as(usize, 48), gemmBlocks(1, 6144));
    try std.testing.expectEqual(@as(usize, 768), gemmBlocks(2048, 6144));
    try std.testing.expectEqual(@as(usize, 512 * 8192 + 64 * 512 * 4), scratchBytes(512, 8192));
    try std.testing.expectEqual(@as(u64, 0x1000 + 256), scratchAt(0x1000, 1, 128).xs);
    try std.testing.expect(supports(fp8.blockWeight(0x1000, 0x2000, 6144, 8192)));
    try std.testing.expect(!supports(fp8.weight(0x1000, 0x2000, 6144, 8192)));
    try std.testing.expect(!supports(fp8.blockWeight(0x1000, 0x2000, 6144, 8192 - 64)));
    try std.testing.expect(!serves(fp8.blockWeight(0x1000, 0x2000, 6144, 8192)));
    try std.testing.expect(serves(fp8.blockWeight(0x1000, 0x2000, 2048, 6144)));
    try std.testing.expect(serves(fp8.blockWeight(0x1000, 0x2000, 32, 6144)));
    try std.testing.expect(!servesFast(fp8.blockWeight(0x1000, 0x2000, 576, 6144)));
    try std.testing.expect(servesFast(fp8.blockWeight(0x1000, 0x2000, 1024, 6144)));
    try std.testing.expect(serves(fp8.blockWeight(0x1000, 0x2000, 6144, 1024)));
    try std.testing.expect(!serves(fp8.blockWeight(0x1000, 0x2000, 2048, 6144 + 128)));
    try std.testing.expectEqual(@as(usize, 3_244_032), scratchBytes(512, level2_k));
}
