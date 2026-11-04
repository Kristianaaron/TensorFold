// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 EXL3 MCG-E4M3 trellis tile decode parity for widths (half-bits) 2, 3, 4, 6 and 8: K1, K1.5, K2, K3, K4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

test "GLM-5.3 x3e4 declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    // Use a diagnostic instantiated beside GLM's production grouped kernel. The
    // DSV4.1 decoder has different template parameters (K2, bits), so it is not
    // a valid MCG-E4M3 oracle target.
    var module = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    inline for (.{ 2, 3, 4, 6, 8 }) |width| try decodeCase(gpu, module, stream, width);
}

fn decodeCase(gpu: anytype, module: cuda.Module, stream: cuda.Stream, comptime width: usize) !void {
    const words = width * 8;
    const trellis = try gpu.gpa.alloc(i16, words);
    defer gpu.gpa.free(trellis);
    var seed: u32 = 0x53e400 + width;
    for (trellis) |*word| {
        seed = seed *% 1664525 +% 1013904223;
        word.* = @bitCast(@as(u16, @truncate(seed >> 8)));
    }
    const symbol = comptime std.fmt.comptimePrint("tf_exl3_mcg_e4m3_tile_{d}", .{width});
    const kernel = try module.function(symbol);
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(trellis));
    defer dt.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, 16 * 16 * 2);
    defer dout.free();
    var args: cuda.Args = .{};
    inline for (.{ dt.ptr, dout.ptr }) |value| args.add(value);
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 } }, stream, &args);
    try stream.synchronize();
    const got_bytes = try download(gpu, dout);
    defer gpu.gpa.free(got_bytes);
    const got = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got_bytes));

    var raw: [256]f16 = undefined;
    core.exl3_format.unpack(.{ .halves = @intCast(width) }, .mcg, trellis, 1, 1, &raw);
    for (raw, 0..) |value, i| {
        const expected: f16 = @floatCast(cuda.exl3.roundE4m3(@floatCast(value)));
        const want: u16 = @bitCast(expected);
        if (got[i] != want) {
            std.debug.print("glm53-x3e4 width={d} mismatch tile=(0,0) element=({d},{d}) gpu=0x{x:0>4} host=0x{x:0>4}\n", .{ width, i / 16, i % 16, got[i], want });
            return error.TileMismatch;
        }
    }
    std.debug.print("PASS glm53-x3e4: MCG-E4M3 tile width={d}, 256 values match host oracle\n", .{width});
}

test "GLM-5.3 x3e4 host oracle covers all requested tile widths" {
    std.testing.refAllDecls(@This());
    for ([_]usize{ 2, 3, 4, 6, 8 }) |width| {
        var trellis: [64]i16 = @splat(0x4a35);
        var decoded: [256]f16 = undefined;
        core.exl3_format.unpack(.{ .halves = @intCast(width) }, .mcg, trellis[0 .. width * 8], 1, 1, &decoded);
        for (decoded) |value| try std.testing.expect(std.math.isFinite(cuda.exl3.roundE4m3(@floatCast(value))));
    }
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}
