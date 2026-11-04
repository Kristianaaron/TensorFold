// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! Adapted from bertholomus/TensorFold@ddfad35: glm5_next/cuda/qmm.py (Apache-2.0).
//! Changes: keep the native block scales and provide host mapping for qmmf's FP8B 64x64 tiles.
//! Block-FP8 checkpoints (128 x 128 e4m3 blocks, fp32 scale_inv) repacked for qmmf.zig as linear.py does.
const std = @import("std");
const qmmf = @import("qmmf.zig");

pub const block = 128; // a checkpoint scale's rows and inputs
pub const group = qmmf.group; // inputs a kernel stage reads, one column scale each

/// Rows padded to whole 128-row blocks, as the kernel's 64-column tiles read them.
pub fn padded(n: usize) usize {
    return (n + block - 1) / block * block;
}

/// The k of byte `byte` of `lane` in a k32 step (fragment_index's kin).
fn kin(lane: usize, byte: usize) usize {
    const p = 16 * (byte / 4) + 4 * (lane % 4) + byte % 4;
    return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}

/// e4m3 codes [n, k] in the kernel's order [npad/64][k/64][8][32][2][8], rows past n zero (linear._fragment_order).
pub fn packCodes(out: []u8, codes: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    std.debug.assert(out.len == npad * k and codes.len == n * k and k % group == 0);
    var i: usize = 0;
    for (0..npad / 64) |t| for (0..k / group) |g| for (0..8) |a| for (0..32) |lane| for (0..2) |h| for (0..8) |b| {
        const row = t * 64 + a * 8 + lane / 4;
        const col = g * group + h * 32 + kin(lane, b);
        out[i] = if (row < n) codes[row * k + col] else 0;
        i += 1;
    };
}

/// Block scales [ceil(n/128), ceil(k/128)] as kernel tiles [npad/64][k/64][64] of fp32, rows past n one (from_rows).
pub fn packScales(out: []f32, scale_inv: []const f32, n: usize, k: usize) void {
    packScalesAt(out, scale_inv, n, k, 0, 0, n, k);
}

/// packScales of rows [row0, row0+n) and inputs [k0, k0+k) of a rows x cols matrix, from the whole's block scales.
pub fn packScalesAt(out: []f32, scale_inv: []const f32, rows: usize, cols: usize, row0: usize, k0: usize, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / group;
    const kb = (cols + block - 1) / block;
    std.debug.assert(out.len == npad * kg and scale_inv.len == ((rows + block - 1) / block) * kb);
    std.debug.assert(row0 + n <= rows and k0 + k <= cols and k0 % group == 0 and k % group == 0);
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) scale_inv[((row0 + row) / block) * kb + (k0 + g * group) / block] else 1.0;
    };
}

/// The lane matmul's view of a packed block-FP8 projection.
pub fn weight(codes: u64, scales: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp8g, .codes = codes, .scales = scales, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

/// The block-scaled GLM view uses the same code order with scale_inv expanded to 64x64 tile lookups.
pub fn blockWeight(codes: u64, scales: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp8b, .codes = codes, .scales = scales, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

/// Pack native 128x128 scale lookups for qmmf's 64x64 code tiles. Values are repeated, never quantized.
/// Expanding to 64x64 lookups preserves a split beginning at row 64 of a source scale block, such as GLM kv_a.
/// Expand one contiguous source's scales. Stacking separate sources requires the caller to concatenate scale rows first.
pub fn packBlockScales(out: []f32, scale_inv: []const f32, rows: usize, cols: usize, row0: usize, k0: usize, n: usize, k: usize) error{InvalidShape}!void {
    const ntile = padded(n) / group;
    const ktile = k / group;
    const full_n = (rows + block - 1) / block;
    const full_k = (cols + block - 1) / block;
    if (rows == 0 or cols == 0 or n == 0 or k == 0 or k % group != 0 or
        out.len != ntile * ktile or scale_inv.len != full_n * full_k or
        row0 > rows or n > rows - row0 or k0 > cols or k > cols - k0 or
        row0 % group != 0 or k0 % group != 0) return error.InvalidShape;
    for (0..ntile) |r| for (0..ktile) |c| {
        out[r * ktile + c] = if (r * group < n)
            scale_inv[((row0 + r * group) / block) * full_k + (k0 + c * group) / block]
        else
            1.0;
    };
}

/// The Python GLM path rounds each dequantized e4m3 weight to bf16 before its IEEE fp32 dot.
pub fn fp8bWeight(code: u8, scale_inv: f32) f32 {
    const sign: f64 = if (code & 0x80 != 0) -1 else 1;
    const exponent: i32 = (code >> 3) & 0xF;
    const mantissa: f64 = @floatFromInt(code & 7);
    const value: f64 = if (exponent == 0)
        sign * mantissa / 8.0 * std.math.pow(f64, 2, -6)
    else
        sign * (1 + mantissa / 8.0) * std.math.pow(f64, 2, @floatFromInt(exponent - 7));
    const dequant = @as(f32, @floatCast(value)) * scale_inv;
    return roundBf16(dequant);
}

/// Round an fp32 weight to bf16, then return its fp32 value. Kept explicit for host references.
pub fn roundBf16(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    const rounded = bits +% 0x7fff +% ((bits >> 16) & 1);
    return @as(f32, @bitCast(rounded & 0xffff0000));
}

/// FP8B code byte for one element in the packed [64-row][64-column] fragment layout.
pub fn packedCodeIndex(row: usize, col: usize, k: usize) usize {
    const t = row / 64;
    const g = col / group;
    const a = (row % 64) / 8;
    const rr = row % 8;
    const within = col % group;
    const h = within / 32;
    const q = within % 32;
    for (0..4) |low| for (0..8) |b| {
        if (kin(low, b) == q) {
            const lane = rr * 4 + low;
            return (((((t * (k / group) + g) * 8 + a) * 32 + lane) * 2 + h) * 8 + b);
        }
    };
    unreachable;
}

/// Dequantize packed FP8B tiles to qmmf's packed BF16 layout. Each element applies its expanded
/// block scale in fp32, rounds once to BF16, and is then stored in the BF16 fragment order.
pub fn dequantizeBf16Packed(out: []u16, codes: []const u8, scales: []const f32, n: usize, k: usize) error{BadFp8BShape}!void {
    const npad = padded(n);
    const expected = npad * k;
    if (n == 0 or k == 0 or k % group != 0 or out.len != expected or codes.len != expected or
        scales.len != (npad / group) * (k / group)) return error.BadFp8BShape;
    const kg = k / group;
    for (0..npad / group) |t| for (0..kg) |g| for (0..8) |jj| for (0..4) |kt| for (0..32) |lane| for (0..4) |e| {
        const row = t * 64 + jj * 8 + lane / 4;
        const col = g * group + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1);
        const dst = (((((t * kg + g) * 8 + jj) * 4 + kt) * 32 + lane) * 4 + e);
        const value: f32 = if (row < n) fp8bWeight(codes[packedCodeIndex(row, col, k)], scales[t * kg + g]) else 0;
        out[dst] = @truncate(@as(u32, @bitCast(roundBf16(value))) >> 16);
    };
}

/// Row-major BF16 weights for cuBLASLt. This has the same fp8 -> fp32 scale -> BF16 round point as the packed path.
pub fn dequantizeBf16RowMajor(out: []u16, codes: []const u8, scales: []const f32, n: usize, k: usize) error{BadFp8BShape}!void {
    const npad = padded(n);
    if (n == 0 or k == 0 or k % group != 0 or out.len != n * k or codes.len != npad * k or
        scales.len != (npad / group) * (k / group)) return error.BadFp8BShape;
    const kg = k / group;
    for (0..n) |row| for (0..k) |col| {
        const tile = row / group * kg + col / group;
        const value = fp8bWeight(codes[packedCodeIndex(row, col, k)], scales[tile]);
        out[row * k + col] = @truncate(@as(u32, @bitCast(value)) >> 16);
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "FP8B scratch uses block-scaled BF16 round-to-nearest values in qmmf layout" {
    const n = 128;
    const k = 64;
    var codes: [n * k]u8 = @splat(0);
    // All elements are E4M3 1.0; the expanded per-64 tile scale is 0.5.
    for (0..n) |r| {
        for (0..k) |c| codes[packedCodeIndex(r, c, k)] = 0x38;
    }
    const scales: [(n / group) * (k / group)]f32 = @splat(0.5);
    var out: [n * k]u16 = undefined;
    try dequantizeBf16Packed(&out, &codes, &scales, n, k);
    for (out) |v| try std.testing.expectEqual(@as(u16, 0x3f00), v);
}

test "FP8B row-major dequant preserves code and native scale mapping" {
    const n = 130;
    const k = 256;
    const npad = padded(n);
    const codes = try std.testing.allocator.alloc(u8, npad * k);
    defer std.testing.allocator.free(codes);
    const scales = try std.testing.allocator.alloc(f32, (npad / group) * (k / group));
    defer std.testing.allocator.free(scales);
    for (codes, 0..) |*v, i| v.* = @truncate((i *% 37 + 0x28) & 0x7e);
    for (scales, 0..) |*v, i| v.* = if (i % 2 == 0) 0.5 else 1.25;
    const rowmajor = try std.testing.allocator.alloc(u16, n * k);
    defer std.testing.allocator.free(rowmajor);
    try dequantizeBf16RowMajor(rowmajor, codes, scales, n, k);
    for (0..n) |row| for (0..k) |col| {
        const tile = (row / group) * (k / group) + col / group;
        const expected = fp8bWeight(codes[packedCodeIndex(row, col, k)], scales[tile]);
        try std.testing.expectEqual(@as(u16, @truncate(@as(u32, @bitCast(expected)) >> 16)), rowmajor[row * k + col]);
    };
}

test "the code repack is a permutation that puts each byte where fragment_index says" {
    const gpa = std.testing.allocator;
    const n = 130;
    const k = 128;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    const out = try gpa.alloc(u8, padded(n) * k);
    defer gpa.free(out);
    packCodes(out, codes, n, k);
    const idx = ((((0 * 2 + 1) * 8 + 3) * 32 + 9) * 2 + 1) * 8 + 5; // (t 0, g 1, a 3, lane 9, h 1, byte 5): row 26
    try std.testing.expectEqual(codes[26 * k + 64 + 32 + kin(9, 5)], out[idx]);
    var zeros: usize = 0;
    for (out) |b| zeros += @intFromBool(b == 0);
    try std.testing.expect(zeros >= (padded(n) - n) * k);
    var seen: [32]bool = @splat(false); // kin covers a k32 step once over a lane quad's 4 x 8 bytes
    for (0..4) |lane| for (0..8) |b| {
        try std.testing.expect(!seen[kin(lane, b)]);
        seen[kin(lane, b)] = true;
    };
}

test "scale tiles: each 64-input group takes its 128 x 128 block's scale, padded rows one" {
    const n = 130;
    const k = 256;
    var inv: [2 * 2]f32 = .{ 1, 2, 3, 4 };
    var out: [256 * 4]f32 = undefined;
    packScales(&out, &inv, n, k);
    try std.testing.expectEqual(@as(f32, 2), out[(0 * 4 + 3) * 64 + 5]); // row 5, group 3 -> block (0, 1)
    try std.testing.expectEqual(@as(f32, 3), out[(2 * 4 + 0) * 64 + 1]); // row 129 -> block (1, 0)
    try std.testing.expectEqual(@as(f32, 1), out[(2 * 4 + 1) * 64 + 2]); // row 130: padding
}

/// packScalesAt of a share equals the share's range of the whole matrix's packScales, its padded rows one.
fn expectShare(rows: usize, cols: usize, row0: usize, k0: usize, n: usize, k: usize) !void {
    const gpa = std.testing.allocator;
    const inv = try gpa.alloc(f32, ((rows + block - 1) / block) * ((cols + block - 1) / block));
    defer gpa.free(inv);
    for (inv, 0..) |*s, i| s.* = @floatFromInt(i + 2);
    const whole = try gpa.alloc(f32, padded(rows) * (cols / group));
    defer gpa.free(whole);
    packScales(whole, inv, rows, cols);
    const share = try gpa.alloc(f32, padded(n) * (k / group));
    defer gpa.free(share);
    packScalesAt(share, inv, rows, cols, row0, k0, n, k);
    for (0..padded(n)) |r| for (0..k / group) |g| {
        const at = (r / 64 * (k / group) + g) * 64 + r % 64;
        const row = row0 + r;
        const want = if (r < n) whole[(row / 64 * (cols / group) + k0 / group + g) * 64 + row % 64] else 1.0;
        try std.testing.expectEqual(want, share[at]);
    };
}

test "a rank's scale tiles are the whole matrix's from its origins, inside a block too (TP=3 of 2,048)" {
    for ([_][2]usize{ .{ 0, 704 }, .{ 704, 704 }, .{ 1408, 640 } }) |s| {
        try expectShare(2048, 4096, s[0], 0, s[1], 4096); // the shared gate/up's output rows
        try expectShare(4096, 2048, 0, s[0], 4096, s[1]); // the shared down's inputs
    }
}
