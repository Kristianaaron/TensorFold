// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP5 host checks for the X3E4 quantizer, mixed-width dispatch contract and expert quality metric.
const std = @import("std");
const x3e4 = @import("x3e4");
const cuda = @import("cuda");
const fp8_mma = @import("fp8_mma");
const exl3_format = @import("exl3_format");

fn bf16Round(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    const rounded = bits +% 0x7fff + ((bits >> 16) & 1);
    return @bitCast(rounded & 0xffff0000);
}

fn mxLinearRow(input: []const u8, scales: []const u8, row: usize, weights: []const u8, col: usize) f32 {
    var acc: f32 = 0;
    for (0..4) |g| {
        var partial: f32 = 0;
        for (0..32) |i| {
            const k = g * 32 + i;
            partial += x3e4.e4m3Decode(input[row * 128 + fp8_mma.activationFragmentIndex(0, k)]) *
                x3e4.e4m3Decode(weights[col * 128 + k]);
        }
        const xs = std.math.scalbn(@as(f32, 1), @as(i32, scales[row * 4 + g]) - 127);
        acc += partial * xs;
    }
    return acc;
}

fn fp16LinearRow(input: []const f32, weights: []const u8, col: usize) f32 {
    var acc: f32 = 0;
    for (0..128) |k| {
        const activation: f32 = @floatCast(@as(f16, @floatCast(bf16Round(input[k]))));
        acc = @mulAdd(f32, activation, x3e4.e4m3Decode(weights[col * 128 + k]), acc);
    }
    return acc;
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(x3e4);
    std.testing.refAllDecls(x3e4.X3e4);
    std.testing.refAllDecls(x3e4.FragmentLayout);
    std.testing.refAllDecls(x3e4.WidthPlan);
}

fn syntheticPackedExpertWeights(bits: usize, seed: usize, out: []u8) !void {
    if (out.len != 128 * 128) return error.BadFixture;
    var codes: [256]u8 = undefined;
    var words: [64]u32 = undefined;
    var tile: [256]u8 = undefined;
    for (0..8) |kt| for (0..8) |nt| {
        var state: u32 = @intCast(seed + kt * 101 + nt * 997 + bits * 7919);
        for (&codes) |*code| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            code.* = @intCast(state & ((@as(usize, 1) << @intCast(bits)) - 1));
        }
        try x3e4.packTrellisCodes(bits, &codes, words[0..x3e4.trellisWords(bits)]);
        try x3e4.decodeTrellisTile(words[0..x3e4.trellisWords(bits)], bits, &tile);
        for (0..16) |n| for (0..16) |k| {
            out[(nt * 16 + n) * 128 + kt * 16 + k] = tile[n * 16 + k];
        };
    };
}

test "X3E4 quantizer bytes match WP3 for a full K32 permutation" {
    var input: [64]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast((i * 29 + 7) % 101)) - 50)) / 11;
    var got: [64]u8 = undefined;
    var got_scales: [2]u8 = undefined;
    var oracle: [64]u8 = undefined;
    var oracle_scales: [2]u8 = undefined;
    try x3e4.quantizeMxfp8(&input, 1, 64, &got, &got_scales);
    try fp8_mma.quantizeMxfp8(&input, 1, 64, &oracle, &oracle_scales);
    try std.testing.expectEqualSlices(u8, &oracle, &got);
    try std.testing.expectEqualSlices(u8, &oracle_scales, &got_scales);
    var seen = @as([32]bool, @splat(false));
    for (0..32) |k| {
        const at = fp8_mma.activationFragmentIndex(0, k);
        try std.testing.expect(at < 32 and !seen[at]);
        seen[at] = true;
    }
    for (seen) |v| try std.testing.expect(v);
}

test "X3E4 trellis states, placement and MCG values match core EXL3 oracle at widths 1 through 8" {
    for (1..9) |bits| {
        const words = try std.testing.allocator.alloc(u32, x3e4.trellisWords(bits));
        defer std.testing.allocator.free(words);
        var rng: u32 = @intCast(0x9e3779b9 ^ (bits * 0x45d9f3b));
        for (words) |*word| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            word.* = rng;
        }
        const oracle_words = try std.testing.allocator.alloc(i16, words.len * 2);
        defer std.testing.allocator.free(oracle_words);
        for (words, 0..) |word, i| {
            oracle_words[2 * i] = @bitCast(@as(u16, @truncate(word)));
            oracle_words[2 * i + 1] = @bitCast(@as(u16, @truncate(word >> 16)));
        }
        const width: exl3_format.Width = .{ .halves = @intCast(2 * bits) };
        var oracle_states: [256]u32 = undefined;
        var placements: [256]exl3_format.Placement = undefined;
        exl3_format.tileStates(width, oracle_words, &oracle_states);
        exl3_format.tilePlacement(&placements);
        var decoded: [256]u8 = undefined;
        try x3e4.decodeTrellisTile(words, bits, &decoded);
        for (0..256) |p| {
            const got = try x3e4.trellisCode(words, bits, p);
            try std.testing.expectEqual(oracle_states[p], @as(u32, got));
            const pos = placements[p];
            const value: f32 = @floatCast(exl3_format.codebookValue(.mcg, @intCast(oracle_states[p])));
            const expected = fp8_mma.toE4m3(@floatCast(cuda.exl3.roundE4m3(value)));
            try std.testing.expectEqual(expected, decoded[@as(usize, pos.col) * 16 + pos.row]);
        }
    }
}

test "mixed matrix widths produce a deterministic stable dispatch plan" {
    const keys = [_]x3e4.WidthKey{
        .{ .gate_up_bits = 4, .down_bits = 2 }, .{ .gate_up_bits = 2, .down_bits = 4 },
        .{ .gate_up_bits = 4, .down_bits = 2 }, .{ .gate_up_bits = 2, .down_bits = 6 },
        .{ .gate_up_bits = 2, .down_bits = 4 },
    };
    var plan = try x3e4.widthBuckets(std.testing.allocator, &keys);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 3), plan.buckets.len);
    try std.testing.expectEqual(x3e4.WidthKey{ .gate_up_bits = 2, .down_bits = 4 }, plan.buckets[0].key);
    try std.testing.expectEqual(@as(usize, 2), plan.buckets[0].count);
    try std.testing.expectEqual(x3e4.WidthKey{ .gate_up_bits = 4, .down_bits = 2 }, plan.buckets[2].key);
    try std.testing.expectEqual(@as(usize, 1), plan.expert_ids[0]);
    try std.testing.expectEqual(@as(usize, 4), plan.expert_ids[1]);
    try std.testing.expect(x3e4.enabledByEnvironment("x3e4"));
    try std.testing.expect(!x3e4.enabledByEnvironment(null));
    try std.testing.expectEqual(x3e4.PrefillExperts.wp4, try x3e4.prefillExperts(null));
    try std.testing.expectEqual(x3e4.PrefillExperts.x3e4, try x3e4.prefillExperts("x3e4"));
}

test "one X3E4 decode launch grids all tiles of an expert matrix" {
    const grid = x3e4.tileGrid(6144 / 16, 1024 / 16);
    try std.testing.expectEqual(@as(usize, 384), grid[0]);
    try std.testing.expectEqual(@as(usize, 64), grid[1]);
    try std.testing.expectEqual(@as(usize, 24576), grid[0] * grid[1]);
}

test "decoded EXL3 tile stages through WP3 FP8G weight fragment indices" {
    var tile: [16 * 16]u8 = undefined;
    for (&tile, 0..) |*v, i| v.* = @intCast(i);
    var staged: [64 * 64]u8 = @splat(0);
    try x3e4.stageWeightTile(&tile, 0, 0, 64, &staged);
    for (0..16) |row| for (0..16) |k| {
        const at = x3e4.weightFragmentIndex(row, k, 64);
        try std.testing.expectEqual(tile[row * 16 + k], staged[at]);
    };
}

test "routed sum matches WP4 fmaf slot order including the ninth supplied slot" {
    // WP4 exl3_experts.cu:368-374 reduces a.slots in ascending order with fmaf.
    // WP8 moe.zig:10-13 allocates 9 slots (8 routed and slot 8 shared); the caller
    // supplies the desired slot_count, and this helper mirrors WP4 for all supplied slots.
    const rows = 2;
    const hidden = 2;
    const slot_stride = 9;
    var values: [rows * slot_stride * hidden]f32 = undefined;
    var weights: [rows * slot_stride]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 23)) - 11)) * 0.125;
    for (&weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 7)) - 3)) * 0.25;
    values[8 * hidden] = 0.375;
    weights[8] = 1;
    var whole: [rows * hidden]f32 = undefined;
    try x3e4.routedPartial(&values, &weights, rows, 9, slot_stride, hidden, &whole);
    var expected: [rows * hidden]f32 = undefined;
    for (0..rows) |r| for (0..hidden) |h| {
        var acc: f32 = 0;
        for (0..9) |s| acc = @mulAdd(f32, weights[r * slot_stride + s], values[(r * slot_stride + s) * hidden + h], acc);
        expected[r * hidden + h] = acc;
    };
    try std.testing.expectEqualSlices(f32, &expected, &whole);
    var split: [rows * hidden]f32 = undefined;
    for (0..rows) |row| try x3e4.routedPartial(values[row * slot_stride * hidden ..][0 .. slot_stride * hidden], weights[row * slot_stride ..][0..slot_stride], 1, 9, slot_stride, hidden, split[row * hidden ..][0..hidden]);
    try std.testing.expectEqualSlices(f32, &whole, &split);
}

test "X3E4 SwiGLU epilogue observes BF16 gate up and output roundings" {
    const gate: f32 = 0.318726;
    const up: f32 = -1.2739;
    const g = bf16Round(gate);
    const u = bf16Round(up);
    const expected = bf16Round(bf16Round(g / (1 + @exp(-g))) * u);
    try std.testing.expectEqual(expected, x3e4.swigluBf16(gate, up));
    const old_contract = bf16Round((g / (1 + @exp(-g))) * u);
    try std.testing.expect(expected != old_contract);
}

test "X3E4 row quantization is bitwise invariant to prompt chunk splits" {
    const rows = 7;
    const k = 128;
    var input: [rows * k]f32 = undefined;
    var suh: [k]f32 = undefined;
    for (&suh, 0..) |*scale, i| scale.* = if (i % 2 == 0) 0.75 else -1.25;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 37)) - 18)) / 13;
    var whole_q: [rows * k]u8 = undefined;
    var whole_s: [rows * (k / 32)]u8 = undefined;
    try x3e4.rotateQuant(&input, &suh, rows, k, &whole_q, &whole_s);

    var split_q: [rows * k]u8 = undefined;
    var split_s: [rows * (k / 32)]u8 = undefined;
    for (0..rows) |r| try x3e4.rotateQuant(input[r * k ..][0..k], &suh, 1, k, split_q[r * k ..][0..k], split_s[r * 4 ..][0..4]);
    try std.testing.expectEqualSlices(u8, &whole_q, &split_q);
    try std.testing.expectEqualSlices(u8, &whole_s, &split_s);
}

test "X3E4 gate projection NMSE stays within twice expected MXFP8 activation quantization error" {
    const rows = 1024;
    const k = 128;
    const input = try std.testing.allocator.alloc(f32, rows * k);
    defer std.testing.allocator.free(input);
    const xq = try std.testing.allocator.alloc(u8, rows * k);
    defer std.testing.allocator.free(xq);
    const xs = try std.testing.allocator.alloc(u8, rows * 4);
    defer std.testing.allocator.free(xs);
    const weights = try std.testing.allocator.alloc(u8, 128 * k);
    defer std.testing.allocator.free(weights);
    try syntheticPackedExpertWeights(2, 0x531, weights);

    // CLT Gaussian (12 uniform draws) with bf16-rounded inputs and one 6-sigma
    // outlier channel per row, matching the routed projection's activation range.
    var rng: u32 = 0x6d2b79f5;
    for (input, 0..) |*v, i| {
        var sum: f32 = 0;
        for (0..12) |_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            sum += @as(f32, @floatFromInt(rng & 0xffff)) / 65535 - 0.5;
        }
        const row = i / k;
        const col = i % k;
        const with_outlier = sum + if (col == (row * 17 + 13) % k) @as(f32, 6.0) else 0.0;
        v.* = bf16Round(with_outlier);
    }
    try x3e4.quantizeMxfp8(input, rows, k, xq, xs);

    var nmse_num: f64 = 0;
    var nmse_den: f64 = 0;
    var dot: f64 = 0;
    var q_norm: f64 = 0;
    var ref_norm: f64 = 0;
    for (0..rows) |r| for (0..128) |n| {
        const actual = mxLinearRow(xq, xs, r, weights, n);
        const reference = fp16LinearRow(input[r * k ..][0..k], weights, n);
        const delta = @as(f64, actual) - reference;
        nmse_num += delta * delta;
        nmse_den += @as(f64, reference) * reference;
        dot += @as(f64, actual) * reference;
        q_norm += @as(f64, actual) * actual;
        ref_norm += @as(f64, reference) * reference;
    };
    const nmse = nmse_num / @max(nmse_den, 1e-20);
    const cosine = dot / @sqrt(@max(q_norm * ref_norm, 1e-30));
    try std.testing.expect(std.math.isFinite(nmse));
    try std.testing.expect(cosine > 0.999);
    // E4M3's relative rounding step is about 1/8, giving an activation-only
    // normalized MSE near (1/16)^2 / 3 ~= 1/768 ~= 0.0013 under a uniform
    // within-bin error model. The 0.002 gate is about twice the typical ~0.001
    // MXFP8/Gaussian estimate (and within 1.5x of 1/768); the reviewed H1 bug's
    // projection NMSE (~0.0058) therefore fails with this realistic fixture.
    try std.testing.expect(nmse < 0.002);
}

test "X3E4 trellis codebook decode matches WP4 host oracle for every 16-bit state" {
    for (0..65536) |raw| {
        const code: u16 = @intCast(raw);
        const product = @as(u32, code) *% 0xCBAC1FED;
        const bits = (product & 0x8FFF8FFF) ^ 0x3B603B60;
        const low: f16 = @bitCast(@as(u16, @truncate(bits)));
        const high: f16 = @bitCast(@as(u16, @truncate(bits >> 16)));
        const half_sum: f16 = low + high;
        const oracle = cuda.exl3.roundE4m3(@as(f32, @floatCast(half_sum)));
        try std.testing.expectEqual(oracle, x3e4.decodeMgcE4m3(code));
    }
}
