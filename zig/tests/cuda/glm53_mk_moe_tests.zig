// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-moe: one-task graph parity for existing GLM MoE glue tasks.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;

const LowRankReduce = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, partials: u64, mid: u64, total: i32, splits: i32 };
const MoeResidualAdd = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, xout: u64, parts: u64, count: i32 };
const RouterTopk = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, logits: u64, bias: u64, picks: u64, weights: u64, scale: f32, experts: i32, top_k: i32, norm: i32 };
const RouterPart = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, w: u64, parts: u64, rows: i32, stride: i64, hidden: i64, experts: i64, ks: i32 };
const SliceSum = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, part: u64, out: u64, total: i64, ks: i32, f32: i32 };
const Swiglu = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, gu: u64, out: u64, rows: i32, width: i32 };
const LowrankPartial = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, a: u64, partials: u64, rows: i32, hidden: i32, rank: i32, x_stride: i32, a_stride: i32, splits: i32 };
const AddPartial = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, add: u64, acc: u64, count: i32 };
const LowrankCombine = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, routed: u64, shared: u64, b: u64, mid: u64, out: u64, rows: i32, hidden: i32, rank: i32, b_stride: i32 };
const ResidualNorm = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, parts: u64, w: u64, normed: u64, rows: i32, eps: f32 };
const LowrankFinish = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, b: u64, mid: u64, acc: u64, rows: i32, hidden: i32, rank: i32, b_stride: i32 };
const EmbedPick = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, gathered: u64, owners: u64, out: u64, rows: i32, hidden: i32, world: i32 };
const Exl3RotIn = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, x_stride: i32, x_dtype: i32, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, k: i32, slots: i32, experts: i32 };
const Exl3DownCombine = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, p: i32, d: i32, sk: i32, experts: i32, slots: i32 };

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(LowRankReduce);
    std.testing.refAllDecls(MoeResidualAdd);
    std.testing.refAllDecls(RouterTopk);
    std.testing.refAllDecls(RouterPart);
    std.testing.refAllDecls(SliceSum);
    std.testing.refAllDecls(Swiglu);
    std.testing.refAllDecls(LowrankPartial);
    std.testing.refAllDecls(AddPartial);
    std.testing.refAllDecls(LowrankCombine);
    std.testing.refAllDecls(ResidualNorm);
    std.testing.refAllDecls(LowrankFinish);
    std.testing.refAllDecls(EmbedPick);
    std.testing.refAllDecls(Exl3RotIn);
    std.testing.refAllDecls(Exl3DownCombine);
    if (@sizeOf(LowRankReduce) != 40 or @offsetOf(LowRankReduce, "partials") != 16) @compileError("MK lowrank reduce ABI drift");
    if (@sizeOf(MoeResidualAdd) != 48 or @offsetOf(MoeResidualAdd, "x") != 16) @compileError("MK residual add ABI drift");
    if (@sizeOf(RouterTopk) != 64 or @offsetOf(RouterTopk, "logits") != 16) @compileError("MK router topk ABI drift");
    if (@sizeOf(RouterPart) != 80 or @offsetOf(RouterPart, "stride") != 48) @compileError("MK router part ABI drift");
    if (@sizeOf(SliceSum) != 48 or @offsetOf(SliceSum, "part") != 16) @compileError("MK slice sum ABI drift");
    if (@sizeOf(Swiglu) != 40 or @sizeOf(LowrankPartial) != 64 or @sizeOf(AddPartial) != 40 or @sizeOf(LowrankCombine) != 72 or @sizeOf(ResidualNorm) != 56 or @sizeOf(LowrankFinish) != 56 or @sizeOf(EmbedPick) != 56) @compileError("MK utility payload ABI drift");
    if (@sizeOf(Exl3RotIn) != 88 or @offsetOf(Exl3RotIn, "x") != 16) @compileError("EXL3 rot-in payload ABI drift");
    if (@sizeOf(Exl3DownCombine) != 88 or @offsetOf(Exl3DownCombine, "z") != 16) @compileError("EXL3 down combine payload ABI drift");
}

fn oneTask(allocator: std.mem.Allocator, rows: usize, kind: graph.Kind) !graph.Graph {
    const tasks = try allocator.alloc(graph.Task, 1);
    errdefer allocator.free(tasks);
    tasks[0] = .{ .kind = kind, .layer = 3, .row = 0 };
    const counters = try allocator.alloc(u32, 1);
    errdefer allocator.free(counters);
    counters[0] = 0;
    const successors = try allocator.alloc(u32, 0);
    return .{ .allocator = allocator, .params = .{ .rows = rows, .layers = 4, .dcp = 1 }, .tasks = tasks, .initial_counters = counters, .successors = successors };
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn dependencyChain(gpu: anytype, scheduler: cuda.Module, stream: cuda.Stream, task_kind: graph.Kind, op: mk.Op, blocks: u32, payload: []const u8, rows: usize) !void {
    const tasks = try gpu.gpa.alloc(graph.Task, 3);
    defer gpu.gpa.free(tasks);
    for (tasks, 0..) |*task, i| task.* = .{ .kind = task_kind, .layer = @intCast(i), .row = 0, .dependency_count = if (i == 0) 0 else 1, .successor_offset = if (i < 2) @intCast(i) else 2, .successor_count = if (i < 2) 1 else 0 };
    var counters = [_]u32{ 0, 1, 1 };
    var successors = [_]u32{ 1, 2 };
    const g: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 3, .dcp = 1 }, .tasks = tasks, .initial_counters = &counters, .successors = &successors };
    const ops = [_]mk.Op{ op, op, op };
    const block_counts = [_]u32{ blocks, blocks, blocks };
    const payloads = [_][]const u8{ payload, payload, payload };
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &block_counts, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
}

fn runReduce(gpu: anytype, scheduler: cuda.Module, wrappers: cuda.Module, stream: cuda.Stream, rows: usize, op: mk.Op) !void {
    const total = rows * 32;
    const splits = 8;
    const host = try gpu.gpa.alloc(f32, total * splits);
    defer gpu.gpa.free(host);
    for (host, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 19) % 127)) * 0.03125;
    const zeros = try gpu.gpa.alloc(f32, total);
    defer gpu.gpa.free(zeros);
    @memset(zeros, 0);
    var partials = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    defer partials.free();
    var reference = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(zeros));
    defer reference.free();
    var actual = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(zeros));
    defer actual.free();
    const params = LowRankReduce{ .gx = @intCast((total + 255) / 256), .gy = 1, .gz = 1, .threads = 256, .partials = partials.ptr, .mid = reference.ptr, .total = @intCast(total), .splits = splits };
    const wrapper = try wrappers.function("tf_glm53_lowrank_reduce");
    var args: cuda.Args = .{};
    inline for (.{ params.partials, params.mid, params.total, params.splits }) |v| args.add(v);
    try cuda.launch.launch(wrapper, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const expected = try download(gpu, reference);
    defer gpu.gpa.free(expected);

    const graph_params = LowRankReduce{ .gx = @intCast((total + 255) / 256), .gy = 1, .gz = 1, .threads = 256, .partials = partials.ptr, .mid = actual.ptr, .total = @intCast(total), .splits = splits };
    var g = try oneTask(gpu.gpa, rows, .moe_combine);
    defer g.deinit();
    const ops = [_]mk.Op{op};
    const blocks = [_]u32{@intCast((total + 255) / 256)};
    const payloads = [_][]const u8{std.mem.asBytes(&graph_params)};
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &blocks, &payloads, 10_000_000);
    defer uploaded.deinit();
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes)));
    const got = try download(gpu, actual);
    defer gpu.gpa.free(got);
    try std.testing.expectEqualSlices(u8, expected, got);
    try dependencyChain(gpu, scheduler, stream, .moe_combine, op, @intCast((total + 255) / 256), std.mem.asBytes(&graph_params), rows);
    try stream.synchronize();
    const chained = try download(gpu, actual);
    defer gpu.gpa.free(chained);
    try std.testing.expectEqualSlices(u8, expected, chained);
    std.debug.print("PASS glm53-mk-moe {s} rows={d}\n", .{ @tagName(op), rows });
}

fn runResidualAdd(gpu: anytype, scheduler: cuda.Module, wrappers: cuda.Module, stream: cuda.Stream, rows: usize, op: mk.Op) !void {
    const count = rows * 6144;
    const xh = try gpu.gpa.alloc(u16, count);
    defer gpu.gpa.free(xh);
    const ph = try gpu.gpa.alloc(u16, count * 2);
    defer gpu.gpa.free(ph);
    for (xh, 0..) |*v, i| v.* = @truncate(0x3f00 + (i % 127));
    for (ph, 0..) |*v, i| v.* = @truncate(0x3e80 + (i % 63));
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var parts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ph));
    defer parts.free();
    var reference = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer reference.free();
    var actual = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer actual.free();
    const params = MoeResidualAdd{ .gx = @intCast((count + 255) / 256), .gy = 1, .gz = 1, .threads = 256, .x = x.ptr, .xout = reference.ptr, .parts = parts.ptr, .count = @intCast(count) };
    const wrapper = try wrappers.function("tf_glm53_moe_residual_add");
    var args: cuda.Args = .{};
    inline for (.{ params.x, params.xout, params.parts, params.count }) |v| args.add(v);
    try cuda.launch.launch(wrapper, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const expected = try download(gpu, reference);
    defer gpu.gpa.free(expected);
    const graph_params = MoeResidualAdd{ .gx = @intCast((count + 255) / 256), .gy = 1, .gz = 1, .threads = 256, .x = x.ptr, .xout = actual.ptr, .parts = parts.ptr, .count = @intCast(count) };
    var g = try oneTask(gpu.gpa, rows, .moe_exchange_residual);
    defer g.deinit();
    const ops = [_]mk.Op{op};
    const blocks = [_]u32{@intCast((count + 255) / 256)};
    const payloads = [_][]const u8{std.mem.asBytes(&graph_params)};
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &blocks, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
    const got = try download(gpu, actual);
    defer gpu.gpa.free(got);
    try std.testing.expectEqualSlices(u8, expected, got);
    std.debug.print("PASS glm53-mk-moe {s} rows={d}\n", .{ @tagName(op), rows });
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0 or cuda.kernels.glm53_glue.len == 0)
        return error.KernelsNotEmbedded;
    var scheduler = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer scheduler.unload();
    var wrappers = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    defer wrappers.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    inline for (.{ 1, 4 }) |rows| {
        try runReduce(gpu, scheduler, wrappers, stream, rows, .lowrank_reduce);
        try runReduce(gpu, scheduler, wrappers, stream, rows, .moe_lowrank_reduce);
        try runResidualAdd(gpu, scheduler, wrappers, stream, rows, .moe_residual_add);
        try runResidualAdd(gpu, scheduler, wrappers, stream, rows, .moe_residual_add_group);
    }
}
