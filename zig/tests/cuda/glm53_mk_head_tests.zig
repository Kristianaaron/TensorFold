// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-head: head wrapper parity for the MK head dispatch package.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;
const sampler = @import("sampler.zig");

const Top1Params = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, vocab: u32, offset: u32, values: u64, ids: u64 };
const NormParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, xs: i64, w: u64, out: u64, os: i64, sums: u64, eps: f32, do_sums: i32 };
const CandFillParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, values: u64, ids: u64, count: u32 };
const CandPartParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, vocab: u32, offset: u32, k: u32, scratch: u64 };
const CandMergeParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, scratch: u64, vocab: u32, k: u32, values: u64, ids: u64 };
const LseParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, vocab: u32, partial: u64 };
const GateGatherParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, logits: u64, ids: u64, vocab: u32, offset: u32, values: u64, probes: u32, rows: u32 };
const GateMassParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, probe_logits: u64, lse: u64, logp: u64, mass_outside: u64, probes: u32, rows: u32 };

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

fn download(gpu: anytype, b: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(bytes);
    try b.download(0, bytes);
    return bytes;
}

fn checkTop1(gpu: anytype, h: sampler.Head, scheduler: cuda.Module, stream: cuda.Stream, rows: usize) !void {
    const vocab: usize = 257;
    const values = try gpu.gpa.alloc(f32, rows * vocab);
    defer gpu.gpa.free(values);
    for (values, 0..) |*v, i| {
        const row = i / vocab;
        const col = i % vocab;
        v.* = @as(f32, @floatFromInt((col * 17 + row * 29) % 101)) / 7.0;
    }
    values[vocab - 1] = values[0]; // stable tie selects the lower global id
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(values));
    defer x.free();
    var out_values = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(rows * 2 * @sizeOf(f32)));
    defer out_values.free();
    var out_ids = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(rows * 2 * @sizeOf(u32)));
    defer out_ids.free();
    try h.top1(stream, x.ptr, vocab, 77000, out_values.ptr, out_ids.ptr, rows);
    try stream.synchronize();
    const got_v = try download(gpu, out_values);
    defer gpu.gpa.free(got_v);
    const got_i = try download(gpu, out_ids);
    defer gpu.gpa.free(got_i);
    const actual_values = std.mem.bytesAsSlice(f32, got_v);
    const actual_ids = std.mem.bytesAsSlice(u32, got_i);
    for (0..rows) |r| {
        const expected = sampler.top1Ref(values[r * vocab ..][0..vocab], 77000);
        try expect(@as(u32, @bitCast(actual_values[r])) == @as(u32, @bitCast(expected.value)), "top1 value row {d}", .{r});
        try expect(actual_ids[r] == expected.id, "top1 id row {d}", .{r});
    }

    const graph_values = try gpu.gpa.alloc(f32, rows);
    defer gpu.gpa.free(graph_values);
    const graph_ids = try gpu.gpa.alloc(u32, rows);
    defer gpu.gpa.free(graph_ids);
    var graph_v = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(graph_values));
    defer graph_v.free();
    var graph_i = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(graph_ids));
    defer graph_i.free();
    const p: Top1Params = .{ .gx = @intCast(rows), .gy = 1, .gz = 1, .threads = 256, .x = x.ptr, .vocab = @intCast(vocab), .offset = 77000, .values = graph_v.ptr, .ids = graph_i.ptr };
    const tasks = try gpu.gpa.alloc(graph.Task, 1);
    defer gpu.gpa.free(tasks);
    tasks[0] = .{ .kind = .head_logits, .layer = 0, .row = 0 };
    const counters = try gpu.gpa.alloc(u32, 1);
    defer gpu.gpa.free(counters);
    counters[0] = 0;
    const successors = try gpu.gpa.alloc(u32, 0);
    defer gpu.gpa.free(successors);
    const g: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 1, .dcp = 1 }, .tasks = tasks, .initial_counters = counters, .successors = successors };
    const ops = [_]mk.Op{.head_top1_f32};
    const blocks = [_]u32{@intCast(rows)};
    const payloads = [_][]const u8{std.mem.asBytes(&p)};
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &blocks, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
    var chain_tasks = [_]graph.Task{
        .{ .kind = .head_logits, .layer = 0, .row = 0, .successor_offset = 0, .successor_count = 1 },
        .{ .kind = .head_logits, .layer = 1, .row = 0, .dependency_count = 1, .successor_offset = 1, .successor_count = 1 },
        .{ .kind = .head_logits, .layer = 2, .row = 0, .dependency_count = 1, .successor_offset = 2, .successor_count = 0 },
    };
    var chain_counters = [_]u32{ 0, 1, 1 };
    var chain_successors = [_]u32{ 1, 2 };
    const chain: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 3, .dcp = 1 }, .tasks = &chain_tasks, .initial_counters = &chain_counters, .successors = &chain_successors };
    const chain_ops = [_]mk.Op{ .head_top1_f32, .head_top1_f32, .head_top1_f32 };
    const chain_blocks = [_]u32{ @intCast(rows), @intCast(rows), @intCast(rows) };
    const chain_payloads = [_][]const u8{ std.mem.asBytes(&p), std.mem.asBytes(&p), std.mem.asBytes(&p) };
    var chain_uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, chain, &chain_ops, &chain_blocks, &chain_payloads, 10_000_000);
    defer chain_uploaded.deinit();
    try std.testing.expectEqual(mk.ErrorCode.ok, try chain_uploaded.run(scheduler, stream, plan));
    const want_v = try download(gpu, out_values);
    defer gpu.gpa.free(want_v);
    const want_i = try download(gpu, out_ids);
    defer gpu.gpa.free(want_i);
    const got_graph_v = try download(gpu, graph_v);
    defer gpu.gpa.free(got_graph_v);
    const got_graph_i = try download(gpu, graph_i);
    defer gpu.gpa.free(got_graph_i);
    try std.testing.expectEqualSlices(u8, want_v[0 .. rows * @sizeOf(f32)], got_graph_v);
    try std.testing.expectEqualSlices(u8, want_i[0 .. rows * @sizeOf(u32)], got_graph_i);
    std.debug.print("PASS glm53-mk-head rows={d}\n", .{rows});
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_head.len == 0) return error.KernelsNotEmbedded;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_head);
    defer module.unload();
    var scheduler = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer scheduler.unload();
    const head = try sampler.Head.resolve(module);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    try checkTop1(gpu, head, scheduler, stream, 1);
    try checkTop1(gpu, head, scheduler, stream, 4);
}

test "head candidate reference keeps row and tie ordering for 1 and 4 rows" {
    const values = [_]f32{ 4, -1, 4, 0, 3, 2, 2, 1 };
    const expected = try sampler.topCandidatesRadixRef(std.testing.allocator, &values, 900, 4);
    defer std.testing.allocator.free(expected);
    try std.testing.expectEqual(@as(usize, 4), expected.len);
    try std.testing.expectEqual(@as(u32, 900), expected[0].id);
    try std.testing.expectEqual(@as(u32, 902), expected[1].id);
    try std.testing.expectEqual(@as(u32, 904), expected[2].id);
    try std.testing.expectEqual(@as(u32, 905), expected[3].id);
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Top1Params);
    std.testing.refAllDecls(NormParams);
    std.testing.refAllDecls(CandFillParams);
    std.testing.refAllDecls(CandPartParams);
    std.testing.refAllDecls(CandMergeParams);
    std.testing.refAllDecls(LseParams);
    std.testing.refAllDecls(GateGatherParams);
    std.testing.refAllDecls(GateMassParams);
    std.testing.refAllDecls(sampler.Head);
    if (@sizeOf(Top1Params) != 48 or @sizeOf(CandFillParams) != 40 or @sizeOf(CandPartParams) != 48 or @sizeOf(CandMergeParams) != 48 or @sizeOf(LseParams) != 40 or @sizeOf(GateGatherParams) != 56 or @sizeOf(GateMassParams) != 56) @compileError("MK head payload ABI drift");
    if (@sizeOf(NormParams) != 72 or @offsetOf(NormParams, "eps") != 64) @compileError("MK head norm payload ABI drift");
}
