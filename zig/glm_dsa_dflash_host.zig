// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 DFlash2 fixture, memory and tap layout checks.
const std = @import("std");
const drafts = @import("drafts");
const taps = @import("taps");
const cuda = @import("cuda");
const glm = @import("glm");
const abi = cuda.abi;

const Recorded = struct { f: usize, args: [5]u64 };
var record_driver: cuda.Driver = undefined;
var recorded: [20]Recorded = undefined;
var recorded_len: usize = 0;

fn recordLaunch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
    const count: usize = 5;
    var r: Recorded = .{ .f = @intFromPtr(f.?), .args = @splat(0) };
    for (0..count) |i| {
        const p = params.?[i].?;
        r.args[i] = if ((@intFromPtr(f.?) == 1 and i >= 2) or (@intFromPtr(f.?) == 2 and i == 1)) @bitCast(@as(i64, @as(*const i64, @ptrCast(@alignCast(p))).*)) else @as(*const u64, @ptrCast(@alignCast(p))).*;
    }
    recorded[recorded_len] = r;
    recorded_len += 1;
    return abi.success;
}

test "glm dsa dflash rejects the Flash drafter shape" {
    const flash = "{\"architectures\":[\"DFlash2DraftModel\"],\"model_type\":\"qwen3\",\"hidden_size\":4096,\"dflash_config\":{\"block_size\":8}}";
    try std.testing.expectError(error.DrafterConfigUnexpected, drafts.parse(std.testing.allocator, flash));
}

test "glm dsa dflash module declarations analyze" {
    std.testing.refAllDecls(drafts);
    std.testing.refAllDecls(taps);
}

test "glm dsa dflash target head and loader call typecheck without a GPU" {
    var host: drafts.Host = undefined;
    const fixture: drafts.Geometry = .{};
    var instantiate = false;
    @atomicStore(bool, &instantiate, false, .monotonic);
    if (instantiate) _ = try drafts.Drafter.load(std.testing.allocator, undefined, &host, ".", 0, 2, 4, fixture, false);
}

test "glm dsa dflash: packed index inverts packBf16" {
    const a = std.testing.allocator;
    for ([_][2]usize{ .{ 70, 128 }, .{ 448, 6144 } }) |shape| {
        const n = shape[0];
        const k = shape[1];
        const w = try a.alloc(u16, n * k);
        defer a.free(w);
        for (w, 0..) |*v, i| v.* = @truncate(i * 7 + 3);
        const npad = (n + 127) / 128 * 128;
        const packed_w = try a.alloc(u16, npad * k);
        defer a.free(packed_w);
        cuda.qmmf.packBf16(packed_w, w, n, k);
        const seen = try a.alloc(bool, npad * k);
        defer a.free(seen);
        @memset(seen, false);
        for (0..n) |r| for (0..k) |c| {
            const ix = drafts.packedIndex(r, c, k);
            try std.testing.expect(ix < npad * k);
            try std.testing.expect(!seen[ix]);
            seen[ix] = true;
            try std.testing.expectEqual(w[r * k + c], packed_w[ix]);
        };
    }
}

test "glm dsa dflash: head quantizer alternates unpack and q4 per 8192-row chunk" {
    record_driver = undefined;
    record_driver.api.cuLaunchKernel = recordLaunch;
    recorded_len = 0;
    const unpack: cuda.Function = .{ .d = &record_driver, .handle = @ptrFromInt(1), .name = "tf_glm53_unpack_bf16_rows" };
    const q4_fn: cuda.Function = .{ .d = &record_driver, .handle = @ptrFromInt(2), .name = "tf_glm_quantize4" };
    const quant: glm.quant.Kernels = .{ .amax = q4_fn, .nvfp4 = q4_fn, .q4 = q4_fn };
    try drafts.quantizePacked(unpack, &quant, .{ .d = &record_driver, .handle = null }, 0x1000, 77440, 6144, 8192, 0x2000, 0x3000, 0x4000, 0x5000);
    try std.testing.expectEqual(@as(usize, 20), recorded_len);
    for (0..10) |chunk| {
        const unpack_rec = recorded[chunk * 2];
        const q4_rec = recorded[chunk * 2 + 1];
        const r0 = chunk * 8192;
        const rows: usize = @min(@as(usize, 8192), 77440 - r0); // explicit: @min narrows to the comptime bound's type
        try std.testing.expectEqual(@as(usize, 1), unpack_rec.f);
        try std.testing.expectEqual(@as(u64, r0), unpack_rec.args[3]);
        try std.testing.expectEqual(@as(u64, rows), unpack_rec.args[4]);
        try std.testing.expectEqual(@as(usize, 2), q4_rec.f);
        try std.testing.expectEqual(@as(u64, rows * 6144 / 64), q4_rec.args[1]);
        try std.testing.expectEqual(@as(u64, 0x3000 + r0 * 6144 / 2), q4_rec.args[2]);
    }
    try std.testing.expectEqual(@as(u64, 3712), recorded[18].args[4]);
}
