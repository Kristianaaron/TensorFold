// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 X3E4 routed expert prefill (revised K7, X3E4 / MXFP8 path).
//! This path changes expert activation numerics and remains opt-in until its Spark KLD gate passes.
//! GEMM interface decision: X3E4 will use the WP3c MMA path extended with expert-array
//! dispatch and per-32 UE8M0 activation scales. WP3c's MMA interface also takes 128x128
//! weight scales; X3E4's trellis-rebuilt E4M3 weights currently have unit scale for each such
//! block. The existing `tf_glm53_fp8_mma_mx` has this activation granularity but is scalar and
//! must be extended for grouped experts. CUTLASS 87c grouped-groupwise needs 1x128 activation
//! groups plus per-expert grouped weights, so it cannot consume X3E4's current scales.
//! X3E4 does not implement a GEMM. Still missing are the grouped/expert-array call and the
//! H_N/svh -> SwiGLU -> suh_d/H -> down -> H_D/svh_d chain; current helpers only stage tiles
//! and run isolated epilogues.
const std = @import("std");
const cuda = @import("cuda");
const fp8_mma = @import("fp8_mma");

pub const group_k: usize = 32;
pub const item_rows: usize = 128;
pub const hadamard_width: usize = 128;
pub const supported_k2 = [_]u8{ 2, 4, 6, 8, 10, 12, 14, 16 };

/// MCG EXL3's 16x16 trellis tile contains 256 codes in 4 * (2 * bits) u32 words.
/// The tensor stores [K/16,N/16,16*bits] i16, byte-identical to this word stream.
pub fn trellisWords(bits: usize) usize {
    return 8 * bits;
}

pub fn tileGrid(tiles_n: usize, tiles_k: usize) [2]usize {
    return .{ tiles_n, tiles_k };
}

/// Pack 256 integer codes into the safetensors EXL3 tile's MSB-first u32 stream.
pub fn packTrellisCodes(bits: usize, codes: []const u8, words: []u32) error{Shape}!void {
    if (bits < 1 or bits > 8 or codes.len != 256 or words.len != trellisWords(bits)) return error.Shape;
    @memset(words, 0);
    const max_code = (@as(usize, 1) << @intCast(bits)) - 1;
    for (codes, 0..) |code, i| {
        if (code > max_code) return error.Shape;
        for (0..bits) |b| {
            const bit = (code >> @intCast(bits - b - 1)) & 1;
            const pos = i * bits + b;
            words[pos / 32] |= @as(u32, bit) << @intCast(31 - pos % 32);
        }
    }
}

fn streamEnd(index: usize, bits: usize) usize {
    const k2 = bits * 2;
    return if (k2 & 1 != 0) ((index + 1) * k2 - ((index + 1) & 1)) / 2 else (index + 1) * (k2 / 2);
}

/// Read one 16-bit EXL3 codebook state from the circular trellis bit stream.
pub fn trellisCode(words: []const u32, bits: usize, index: usize) error{Shape}!u16 {
    if (bits < 1 or bits > 8 or words.len != trellisWords(bits) or index >= 256) return error.Shape;
    const end = streamEnd(index, bits);
    const ring = words.len * 32;
    const start = end + ring - 16;
    var code: u16 = 0;
    for (0..bits) |b| {
        const pos = start + b;
        const word = words[(pos / 32) % words.len];
        code = (code << 1) | @as(u16, @truncate((word >> @intCast(31 - (pos % 32))) & 1));
    }
    // Continue the window to 16 bits; EXL3's MCG codebook consumes this state, not the packed integer.
    for (bits..16) |b| {
        const pos = start + b;
        const word = words[(pos / 32) % words.len];
        code = (code << 1) | @as(u16, @truncate((word >> @intCast(31 - (pos % 32))) & 1));
    }
    return code;
}

/// Decode one K16,N16 trellis tile in logical [N][K] order, applying the exact WP4 MCG-E4M3 oracle.
pub fn decodeTrellisTile(words: []const u32, bits: usize, out: []u8) error{Shape}!void {
    if (out.len != 16 * 16 or words.len != trellisWords(bits)) return error.Shape;
    for (0..256) |i| {
        const code = try trellisCode(words, bits, i);
        const lane = i / 8;
        const j = i % 8;
        const row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
        const col = lane / 4 + 8 * (j >> 2);
        out[col * 16 + row] = fp8_mma.toE4m3(@floatCast(decodeMgcE4m3(code)));
    }
}

/// Packed byte location used by WP3's FP8G weight fragments [npad/64][k/64][8][32][2][8].
pub const weightFragmentIndex = fp8_mma.weightFragmentIndex;

pub const WidthKey = struct { gate_up_bits: u8, down_bits: u8 };
pub const WidthBucket = struct { key: WidthKey, first: usize, count: usize };
pub const WidthPlan = struct {
    allocator: std.mem.Allocator,
    buckets: []WidthBucket,
    expert_ids: []usize,

    pub fn deinit(p: *WidthPlan) void {
        p.allocator.free(p.buckets);
        p.allocator.free(p.expert_ids);
    }
};

/// Stable expert-id order, grouped by gate/up and down widths without atomics.
pub fn widthBuckets(allocator: std.mem.Allocator, keys: []const WidthKey) !WidthPlan {
    const ids = try allocator.alloc(usize, keys.len);
    errdefer allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = i;
    std.mem.sort(usize, ids, keys, struct {
        fn less(widths: []const WidthKey, a: usize, b: usize) bool {
            const x = widths[a];
            const y = widths[b];
            if (x.gate_up_bits != y.gate_up_bits) return x.gate_up_bits < y.gate_up_bits;
            if (x.down_bits != y.down_bits) return x.down_bits < y.down_bits;
            return a < b;
        }
    }.less);
    var result: std.ArrayList(WidthBucket) = .empty;
    errdefer result.deinit(allocator);
    for (ids, 0..) |id, i| {
        const key = keys[id];
        if (i == 0 or key.gate_up_bits != keys[ids[i - 1]].gate_up_bits or key.down_bits != keys[ids[i - 1]].down_bits) {
            try result.append(allocator, .{ .key = key, .first = i, .count = 1 });
        } else result.items[result.items.len - 1].count += 1;
    }
    return .{ .allocator = allocator, .buckets = try result.toOwnedSlice(allocator), .expert_ids = ids };
}

/// Sum routed slots in ascending slot order into WP8's fp32 [row][hidden] partial layout.
pub fn routedPartial(slot_values: []const f32, weights: []const f32, rows: usize, slots: usize, slot_stride: usize, hidden: usize, out: []f32) error{Shape}!void {
    if (slots == 0 or slot_stride < slots or slot_values.len != rows * slot_stride * hidden or weights.len != rows * slot_stride or out.len != rows * hidden) return error.Shape;
    for (0..rows) |r| for (0..hidden) |h| {
        var acc: f32 = 0;
        for (0..slots) |s| acc = @mulAdd(f32, slot_values[(r * slot_stride + s) * hidden + h], weights[r * slot_stride + s], acc);
        out[r * hidden + h] = acc;
    };
}

/// WP4 routed contract: bf16(bf16(silu(gate)) * up), using expf and IEEE division; GLM-5.3 has no limit clamp.
/// The intermediate BF16 SiLU matches WP8 mlp.zig:49-53 and WP4 exl3_experts.cu:311.
pub fn swigluBf16(gate: f32, up: f32) f32 {
    const g = bf16Round(gate);
    const u = bf16Round(up);
    const silu = bf16Round(g / (1 + @exp(-g)));
    return bf16Round(silu * u);
}

pub fn bf16Round(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    const rounded = bits +% 0x7fff + ((bits >> 16) & 1);
    return @bitCast(rounded & 0xffff0000);
}

/// Host model for the no-activation-scale weight fragment staging layout.
pub fn stageWeightTile(decoded: []const u8, row0: usize, k0: usize, k_width: usize, packed_weights: []u8) error{Shape}!void {
    if (decoded.len != 16 * 16 or k_width == 0 or k_width % 64 != 0 or packed_weights.len != (k_width / 64) * ((row0 + 16 + 63) / 64) * 64 * 64) return error.Shape;
    for (0..16) |n| for (0..16) |k| {
        const at = weightFragmentIndex(row0 + n, k0 + k, k_width);
        if (at >= packed_weights.len) return error.Shape;
        packed_weights[at] = decoded[n * 16 + k];
    };
}

/// Environment gate; the WP4 path remains the default for every other value.
pub fn enabledByEnvironment(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "x3e4") else false;
}

pub const PrefillExperts = enum { wp4, x3e4 };

pub fn prefillExperts(value: ?[]const u8) error{BadPrefillExperts}!PrefillExperts {
    if (value == null or std.mem.eql(u8, value.?, "wp4")) return .wp4;
    if (enabledByEnvironment(value)) return .x3e4;
    return error.BadPrefillExperts;
}

/// Trellis and fragment helpers used by the CUDA kernel and host oracle.
pub const X3e4 = struct {
    mod: cuda.Module,
    tile: cuda.Function,
    mma: cuda.Function,
    swiglu: cuda.Function,
    route_sum: cuda.Function,

    pub fn load(d: *const cuda.Driver) !X3e4 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm53_x3e4);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .tile = try mod.function("tf_glm53_x3e4_tile"),
            .mma = try mod.function("tf_glm53_x3e4_fragment_mma"),
            .swiglu = try mod.function("tf_glm53_x3e4_swiglu"),
            .route_sum = try mod.function("tf_glm53_x3e4_route_sum"),
        };
    }

    pub fn unload(x: *X3e4) void {
        x.mod.unload();
    }

    /// Runs one row-major 16x32 by 32x8 e4m3 fragment tile. The output is fp32.
    pub fn fragmentMma(x: *const X3e4, s: cuda.Stream, a: u64, b: u64, out: u64) !void {
        var args: cuda.Args = .{};
        args.add(a);
        args.add(b);
        args.add(out);
        try cuda.launch.launch(x.mma, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &args);
    }

    /// Decode one 16x16 EXL3 trellis tile into WP3's global FP8 weight order.
    /// Decode every trellis tile of one [N,K] expert matrix with a single launch.
    pub fn decodeMatrix(x: *const X3e4, s: cuda.Stream, trellis: u64, bits: usize, tiles_n: usize, tiles_k: usize, k_width: usize, packed_weights: u64, scales: u64) !void {
        if (bits < 1 or bits > 8 or tiles_n == 0 or tiles_k == 0 or k_width == 0 or k_width % 64 != 0 or tiles_n % 4 != 0 or tiles_k % 4 != 0) return error.Shape;
        var args: cuda.Args = .{};
        args.add(trellis);
        args.add(@as(i32, @intCast(bits)));
        args.add(@as(i32, @intCast(tiles_n)));
        args.add(@as(i32, @intCast(tiles_k)));
        args.add(@as(i32, @intCast(k_width)));
        args.add(packed_weights);
        args.add(scales);
        const grid = tileGrid(tiles_n, tiles_k);
        try cuda.launch.launch(x.tile, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = 1 }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn swigluRows(x: *const X3e4, s: cuda.Stream, gate: u64, up: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var args: cuda.Args = .{};
        args.add(gate);
        args.add(up);
        args.add(out);
        args.add(@as(i32, @intCast(count)));
        try cuda.launch.launch(x.swiglu, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn routedPartial(x: *const X3e4, s: cuda.Stream, slots_ptr: u64, weights: u64, out: u64, rows: usize, slot_count: usize, slot_stride: usize, hidden: usize) !void {
        if (rows == 0 or hidden == 0) return;
        var args: cuda.Args = .{};
        args.add(slots_ptr);
        args.add(weights);
        args.add(out);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(slot_count)));
        args.add(@as(i32, @intCast(slot_stride)));
        args.add(@as(i32, @intCast(hidden)));
        const count = rows * hidden;
        try cuda.launch.launch(x.route_sum, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
    }
};

/// Row-local per-32 UE8M0 quantization. `permuted` is the X3E4 K-order consumed by the e4m3 MMA.
pub fn quantizeMxfp8(input: []const f32, rows: usize, k: usize, permuted: []u8, scales: []u8) error{Shape}!void {
    if (k == 0 or k % group_k != 0 or input.len != rows * k or permuted.len != input.len or scales.len != rows * (k / group_k)) return error.Shape;
    for (0..rows) |r| for (0..k / group_k) |g| {
        const base = r * k + g * group_k;
        var amax: f32 = 0;
        for (input[base..][0..group_k]) |v| amax = @max(amax, @abs(v));
        const raw_exp: i32 = if (amax == 0) 0 else @intFromFloat(@ceil(std.math.log2(@as(f64, amax / 448.0))));
        const exp = @max(-127, @min(127, raw_exp));
        const scale = std.math.scalbn(@as(f32, 1), exp);
        scales[r * (k / group_k) + g] = @intCast(exp + 127);
        for (0..group_k) |i| {
            const dst = fp8_mma.activationFragmentIndex(0, i);
            permuted[base + dst] = fp8_mma.toE4m3(@floatCast(input[base + i] / scale));
        }
    };
}

/// Inverse K permutation: return the logical K consumed at the given physical FP8 fragment offset.
pub fn permutedK(block_k: usize, t: usize, j: usize) usize {
    const physical = t * 4 + j;
    for (0..group_k) |k| if (fp8_mma.activationFragmentIndex(0, k) == physical) return block_k + k;
    unreachable;
}

/// Compatibility wrapper for WP3's single E4M3 encoder.
pub fn e4m3Encode(x: f32) u8 {
    return fp8_mma.toE4m3(@floatCast(x));
}

pub fn e4m3Decode(code: u8) f32 {
    if (code & 0x7f == 0x7f) return std.math.nan(f32);
    const exp: i32 = (code >> 3) & 15;
    const mant: f32 = @floatFromInt(code & 7);
    const value = if (exp == 0) mant * 0x1p-9 else (1 + mant / 8) * std.math.scalbn(@as(f32, 1), exp - 7);
    return if (code & 0x80 != 0) -value else value;
}

/// Host model for the X3E4 layout. Every MMA operand byte has a stable (row, k) or (k, column) owner.
pub const FragmentLayout = struct {
    /// A is row-major 16x32, four 32-bit registers per lane.
    pub fn aIndex(lane: usize, register: usize, byte: usize) usize {
        const i = register * 4 + byte;
        const row = (lane >> 2) + @as(usize, if (i < 4 or (i >= 8 and i < 12)) 0 else 8);
        const k = (lane & 3) * 4 + (i & 3) + @as(usize, if (i >= 8) 16 else 0);
        return row * 32 + k;
    }

    /// B is row-major 32x8, two 32-bit registers per lane.
    pub fn bIndex(lane: usize, register: usize, byte: usize) usize {
        const i = register * 4 + byte;
        const k = (lane & 3) * 4 + (i & 3) + @as(usize, if (i >= 4) 16 else 0);
        return k * 8 + (lane >> 2);
    }
};

/// Scale and dot in the same fixed K32 order as the fragment MMA path.
pub fn fragmentReference(a: []const u8, b: []const u8, out: []f32) error{Shape}!void {
    if (a.len != 16 * 32 or b.len != 32 * 8 or out.len != 16 * 8) return error.Shape;
    for (0..16) |m| for (0..8) |n| {
        var sum: f32 = 0;
        for (0..32) |k| sum = @mulAdd(f32, e4m3Decode(a[m * 32 + k]), e4m3Decode(b[k * 8 + n]), sum);
        out[m * 8 + n] = sum;
    };
}

/// Normalized Hadamard-128 in the same ascending butterfly schedule as exllamav3's fwht128.
pub fn hadamard128(values: *[128]f32) void {
    var stride: usize = 1;
    while (stride < 128) : (stride *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += stride * 2) for (0..stride) |i| {
            const a = values[base + i];
            const b = values[base + stride + i];
            values[base + i] = a + b;
            values[base + stride + i] = a - b;
        };
    }
    for (values) |*v| v.* *= 0.08838834764831845;
}

/// Converts bf16 inputs represented as fp32, applies suh and Had128, then MXFP8 quantizes.
pub fn rotateQuant(input: []const f32, suh: []const f32, rows: usize, k: usize, xq: []u8, xs: []u8) error{Shape}!void {
    if (k == 0 or k % hadamard_width != 0 or input.len != rows * k or suh.len != k or xq.len != input.len or xs.len != rows * (k / group_k)) return error.Shape;
    for (0..rows) |r| for (0..k / hadamard_width) |block| {
        var vals: [hadamard_width]f32 = undefined;
        for (0..hadamard_width) |i| vals[i] = input[r * k + block * hadamard_width + i] * suh[block * hadamard_width + i];
        hadamard128(&vals);
        try quantizeMxfp8(
            &vals,
            1,
            hadamard_width,
            xq[r * k + block * hadamard_width ..][0..hadamard_width],
            xs[r * (k / group_k) + block * (hadamard_width / group_k) ..][0 .. hadamard_width / group_k],
        );
    };
}

/// MCG-e4m3 lookup for one EXL3 code. The state-to-half arithmetic follows WP4's decode2 path.
pub fn decodeMgcE4m3(code: u16) f32 {
    const x = @as(u32, code) *% 0xCBAC1FED;
    const bits = (x & 0x8FFF8FFF) ^ 0x3B603B60;
    const lo: f16 = @bitCast(@as(u16, @truncate(bits)));
    const hi: f16 = @bitCast(@as(u16, @truncate(bits >> 16)));
    const v0: f16 = lo + hi;
    return @as(f32, @floatCast(@as(f16, @floatCast(e4m3Decode(e4m3Encode(@as(f32, @floatCast(v0))))))));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(X3e4);
    std.testing.refAllDecls(WidthPlan);
    std.testing.refAllDecls(FragmentLayout);
}

test "X3E4 MXFP8 activations use the WP3 K32 bijection and exact shared quantizer" {
    for ([_]usize{ 1, 2, 3 }) |rows| {
        var input: [rows * 64]f32 = undefined;
        for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast((i * 37 + 11) % 127)) - 63)) / 17;
        var got: [rows * 64]u8 = undefined;
        var got_scales: [rows * 2]u8 = undefined;
        var oracle: [rows * 64]u8 = undefined;
        var oracle_scales: [rows * 2]u8 = undefined;
        try quantizeMxfp8(&input, rows, 64, &got, &got_scales);
        try fp8_mma.quantizeMxfp8(&input, rows, 64, &oracle, &oracle_scales);
        try std.testing.expectEqualSlices(u8, &oracle, &got);
        try std.testing.expectEqualSlices(u8, &oracle_scales, &got_scales);
    }
    var seen = @as([32]bool, @splat(false));
    for (0..32) |k| {
        const physical = fp8_mma.activationFragmentIndex(0, k);
        try std.testing.expect(physical < 32 and !seen[physical]);
        seen[physical] = true;
    }
    for (seen) |was_seen| try std.testing.expect(was_seen);
}

test "X3E4 fragment layout is bijective and agrees with host MMA" {
    var seen_a = @as([512]bool, @splat(false));
    for (0..32) |lane| for (0..4) |reg| for (0..4) |byte| {
        const index = FragmentLayout.aIndex(lane, reg, byte);
        try std.testing.expect(index < seen_a.len and !seen_a[index]);
        seen_a[index] = true;
    };
    var seen_b = @as([256]bool, @splat(false));
    for (0..32) |lane| for (0..2) |reg| for (0..4) |byte| {
        const index = FragmentLayout.bIndex(lane, reg, byte);
        try std.testing.expect(index < seen_b.len and !seen_b[index]);
        seen_b[index] = true;
    };
    const a = @as([512]u8, @splat(0x38));
    const b = @as([256]u8, @splat(0x38));
    var out: [16 * 8]f32 = undefined;
    try fragmentReference(&a, &b, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 32), v);
}

test "X3E4 codebook E4M3 conversion covers every finite code" {
    for (0..256) |raw| {
        const code: u8 = @intCast(raw);
        if (code & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(code, e4m3Encode(e4m3Decode(code)));
    }
    for (0..65536) |raw| {
        const x = decodeMgcE4m3(@intCast(raw));
        try std.testing.expect(std.math.isFinite(x));
        try std.testing.expectEqual(x, @as(f32, @floatCast(@as(f16, @floatCast(x)))));
    }
}
