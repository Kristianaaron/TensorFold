// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 rank-1 residual ablation kernel parity checks.
const std = @import("std");
const cuda = @import("cuda");
const ablate = @import("ablate");

const width = ablate.width;

test "GLM-5.3 ablation declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer module.unload();
    const kernel = try module.function("tf_glm53_ablate_6144");
    const residual = try module.function("tf_glm53_residual_rmsnorm_6144");
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var rng = std.Random.DefaultPrng.init(0x530a_b1e);
    for ([_]usize{ 1, 8, 37 }) |rows| for ([_]ablate.Kind{ .f32_parts, .bf16_parts }) |kind| {
        try parity(gpu, stream, kernel, rows, kind, rng.random());
    };
    try zeroScaleNorm(gpu, stream, kernel, residual, rng.random());
}

fn parity(gpu: anytype, stream: cuda.Stream, kernel: cuda.Function, rows: usize, kind: ablate.Kind, random: std.Random) !void {
    const a = gpu.gpa;
    const n = rows * width;
    const psize = n * 2 * @as(usize, if (kind == .f32_parts) 4 else 2);
    const x = try a.alloc(u16, n);
    defer a.free(x);
    const r = try a.alloc(f32, width);
    defer a.free(r);
    const parts = try a.alloc(u8, psize);
    defer a.free(parts);
    for (x) |*v| {
        const bits: u32 = @bitCast(random.float(f32) * 2 - 1);
        v.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }
    for (r) |*v| v.* = (random.float(f32) * 2 - 1) / @sqrt(@as(f32, @floatFromInt(width)));
    if (kind == .f32_parts) {
        const p = std.mem.bytesAsSlice(f32, parts);
        for (p) |*v| v.* = (random.float(f32) * 2 - 1) * 0.05;
    } else {
        const p = std.mem.bytesAsSlice(u16, parts);
        for (p) |*v| {
            const f = (random.float(f32) * 2 - 1) * 0.05;
            const bits: u32 = @bitCast(f);
            v.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
        }
    }
    const expected = try a.dupe(u16, x);
    defer a.free(expected);
    ablate.ablateRef(expected, parts, kind, r, 0.75, rows);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x)); defer dx.free();
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, parts); defer dp.free();
    var dr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(r)); defer dr.free();
    var da = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&[_]f32{0.75})); defer da.free();
    try launch(kernel, stream, dx.ptr, dp.ptr, kind, .{ .r = dr.ptr, .a = da.ptr }, rows, 0, 0, 0);
    try stream.synchronize();
    const got = try a.alloc(u16, n); defer a.free(got);
    try dx.download(0, std.mem.sliceAsBytes(got));
    try expect(std.mem.eql(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(got)), "ablation kind={s} rows={d}", .{ @tagName(kind), rows });
    std.debug.print("PASS glm53 ablation kernel == ablateRef: kind={s} rows={d}\n", .{ @tagName(kind), rows });
}

fn launch(kernel: cuda.Function, stream: cuda.Stream, x: u64, parts: u64, kind: ablate.Kind, v: ablate.Vec, rows: usize, norm_w: u64, normed: u64, eps: f32) !void {
    var args: cuda.Args = .{};
    inline for (.{ x, parts, @as(c_int, @intCast(@intFromEnum(kind))), v.r, v.a, norm_w, normed, @as(c_int, @intCast(rows)), eps }) |arg| args.add(arg);
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
}

fn zeroScaleNorm(gpu: anytype, stream: cuda.Stream, kernel: cuda.Function, residual: cuda.Function, random: std.Random) !void {
    const rows: usize = 8;
    const a = gpu.gpa;
    const x = try a.alloc(u16, rows * width); defer a.free(x);
    const parts = try a.alloc(f32, rows * width * 2); defer a.free(parts);
    const rw = try a.alloc(u16, width); defer a.free(rw);
    const r = try a.alloc(f32, width); defer a.free(r);
    for (x) |*v| { const bits: u32 = @bitCast(random.float(f32) * 2 - 1); v.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16); }
    for (parts) |*v| v.* = (random.float(f32) * 2 - 1) * 0.01;
    for (rw, 0..) |*v, i| {
        const bits: u32 = @bitCast(0.5 + @as(f32, @floatFromInt(i % 7)) * 0.25);
        v.* = @truncate(bits >> 16);
    }
    @memset(r, 0);
    var dx0 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x)); defer dx0.free();
    var dx1 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x)); defer dx1.free();
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(parts)); defer dp.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rw)); defer dw.free();
    var dr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(r)); defer dr.free();
    var da = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&[_]f32{0})); defer da.free();
    var no = try cuda.DeviceBuffer.alloc(gpu.d, rows * width * 2); defer no.free();
    var ng = try cuda.DeviceBuffer.alloc(gpu.d, rows * width * 2); defer ng.free();
    var ns = try cuda.DeviceBuffer.alloc(gpu.d, rows * width * 2); defer ns.free();
    var args: cuda.Args = .{};
    inline for (.{
        dx0.ptr, dp.ptr, dw.ptr, no.ptr, @as(c_int, rows), @as(f32, 1e-6),
    }) |arg| args.add(arg);
    try cuda.launch.launch(residual, .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    // The ablation kernel runs after the residual add (it sees the x that already holds the block output), so the
    // ablated path is the plain fused kernel on dx1 followed by the ablation launch with the norm fused in.
    var args1: cuda.Args = .{};
    inline for (.{
        dx1.ptr, dp.ptr, dw.ptr, ns.ptr, @as(c_int, rows), @as(f32, 1e-6),
    }) |arg| args1.add(arg);
    try cuda.launch.launch(residual, .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args1);
    try launch(kernel, stream, dx1.ptr, dp.ptr, .f32_parts, .{ .r = dr.ptr, .a = da.ptr }, rows, dw.ptr, ng.ptr, 1e-6);
    try stream.synchronize();
    const x0 = try download(gpu, dx0); defer a.free(x0);
    const x1 = try download(gpu, dx1); defer a.free(x1);
    const n0 = try download(gpu, no); defer a.free(n0);
    const n1 = try download(gpu, ng); defer a.free(n1);
    const xs = std.mem.eql(u8, x0, x1);
    const ns_eq = std.mem.eql(u8, n0, n1);
    if (!xs or !ns_eq) {
        const xd = std.mem.indexOfDiff(u8, x0, x1);
        const nd = std.mem.indexOfDiff(u8, n0, n1);
        std.debug.print("  residual equal={}, first differing byte {?d}; normed equal={}, first differing byte {?d}\n", .{ xs, xd, ns_eq, nd });
    }
    try expect(xs and ns_eq, "zero-scale ablation preserves fused residual and RMSNorm bits", .{});
    std.debug.print("PASS glm53 zero-scale ablation preserves fused residual and RMSNorm bits\n", .{});
}

fn download(gpu: anytype, b: cuda.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(out);
    try b.download(0, out);
    return out;
}
fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (!ok) { std.debug.print("FAIL " ++ fmt ++ "\n", args); return error.TestFailed; }
}

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open(); defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0); defer ctx.deinit();
    run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa }) catch |e| {
        std.debug.print("FAIL glm53-ablate: {t}\n", .{e});
        return 1;
    };
    return 0;
}
