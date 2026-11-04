// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk: runtime image and task ABI smoke checks (GPU graph parity is the next gate).
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;

const AddWrapperParams = extern struct { add: u64, acc: u64, count: i32 };
const AddParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, add: u64, acc: u64, count: i32 };

fn syntheticGraph(allocator: std.mem.Allocator, bad_final_counter: bool) !graph.Graph {
    const n = 200;
    const tasks = try allocator.alloc(graph.Task, n);
    errdefer allocator.free(tasks);
    const counters = try allocator.alloc(u32, n);
    errdefer allocator.free(counters);
    const successors = try allocator.alloc(u32, n);
    errdefer allocator.free(successors);
    var successor_counts = try allocator.alloc(u16, n);
    defer allocator.free(successor_counts);
    var incoming = try allocator.alloc(u16, n);
    defer allocator.free(incoming);
    @memset(successor_counts, 0);
    @memset(incoming, 0);
    for (1..199) |i| {
        const parent = (i - 1) / 2;
        successor_counts[parent] += 1;
        incoming[i] = 1;
    }
    successor_counts[98] += 1;
    successor_counts[99] += 1;
    incoming[199] = if (bad_final_counter) 3 else 2;
    var offset: usize = 0;
    for (tasks, 0..) |*task, i| {
        task.* = .{
            .kind = .moe_combine, // any model kind: the op below selects the dispatch
            .layer = 0,
            .row = @intCast(i % 16),
            .dependency_count = incoming[i],
            .successor_offset = @intCast(offset),
            .successor_count = successor_counts[i],
        };
        counters[i] = incoming[i];
        offset += successor_counts[i];
    }
    var cursor = try allocator.alloc(u16, n);
    defer allocator.free(cursor);
    @memset(cursor, 0);
    for (1..199) |i| {
        const parent = (i - 1) / 2;
        successors[tasks[parent].successor_offset + cursor[parent]] = @intCast(i);
        cursor[parent] += 1;
    }
    successors[tasks[98].successor_offset + cursor[98]] = 199;
    cursor[98] += 1;
    successors[tasks[99].successor_offset + cursor[99]] = 199;
    return .{ .allocator = allocator, .params = .{ .rows = 1, .layers = 1, .dcp = 1 }, .tasks = tasks, .initial_counters = counters, .successors = successors };
}

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0) return error.KernelsNotEmbedded;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer module.unload();
    _ = try module.function("tf_glm53_mk_reset");
    _ = try module.function("tf_glm53_mk_run");
    const reference = try module.function("tf_glm53_mk_add_partial_ref");
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    const n: usize = 200;
    const input_host = try gpu.gpa.alloc(f32, n);
    defer gpu.gpa.free(input_host);
    const output_host = try gpu.gpa.alloc(f32, n);
    defer gpu.gpa.free(output_host);
    for (input_host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) / 17.0;
    @memset(output_host, 0);
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(input_host));
    defer input.free();
    var output = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(output_host));
    defer output.free();
    const ops = try gpu.gpa.alloc(mk.Op, n);
    defer gpu.gpa.free(ops);
    const blocks = try gpu.gpa.alloc(u32, n);
    defer gpu.gpa.free(blocks);
    const payload_storage = try gpu.gpa.alloc(AddParams, n);
    defer gpu.gpa.free(payload_storage);
    const payloads = try gpu.gpa.alloc([]const u8, n);
    defer gpu.gpa.free(payloads);
    for (0..n) |i| {
        ops[i] = .add_partial;
        blocks[i] = 1;
        payload_storage[i] = .{ .gx = 1, .gy = 1, .gz = 1, .threads = 256, .add = input.ptr + i * @sizeOf(f32), .acc = output.ptr + i * @sizeOf(f32), .count = 1 };
        payloads[i] = std.mem.asBytes(&payload_storage[i]);
    }
    var g = try syntheticGraph(gpu.gpa, false);
    defer g.deinit();
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, ops, blocks, payloads, 10_000_000);
    defer uploaded.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const ref_cfg: cuda.Config = .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
    for (payload_storage) |params| {
        var args: cuda.Args = .{};
        args.add(AddWrapperParams{ .add = params.add, .acc = params.acc, .count = params.count });
        try cuda.launch.launch(reference, ref_cfg, stream, &args);
    }
    try stream.synchronize();
    const reference_bytes = try download(gpu, output);
    defer gpu.gpa.free(reference_bytes);
    try output.upload(0, std.mem.sliceAsBytes(output_host));
    const code = try uploaded.run(module, stream, plan);
    try expect(code == .ok, "megakernel status {s}", .{@tagName(code)});
    const got = try download(gpu, output);
    defer gpu.gpa.free(got);
    try expect(std.mem.eql(u8, got, reference_bytes), "200-task wrapper parity", .{});
    try output.upload(0, std.mem.sliceAsBytes(output_host));
    const replay = try uploaded.run(module, stream, plan);
    try expect(replay == .ok, "replay status {s}", .{@tagName(replay)});
    const replay_bytes = try download(gpu, output);
    defer gpu.gpa.free(replay_bytes);
    try expect(std.mem.eql(u8, replay_bytes, reference_bytes), "replayed graph parity", .{});

    var broken = try syntheticGraph(gpu.gpa, true);
    defer broken.deinit();
    var deadlock = try mk.Uploaded.init(gpu.gpa, gpu.d, broken, ops, blocks, payloads, 50_000);
    defer deadlock.deinit();
    const deadlock_code = try deadlock.run(module, stream, plan);
    try expect(deadlock_code == .timeout, "deadlock status {s}", .{@tagName(deadlock_code)});
    std.debug.print("PASS 200-task chain/fan-out/fan-in parity, replay reset and dependency timeout; grid={d}, smem={d} B\n", .{ plan.persistent_blocks, plan.dynamic_smem });
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(AddWrapperParams);
    std.testing.refAllDecls(AddParams);
    if (@sizeOf(AddParams) != 40 or @offsetOf(AddParams, "add") != 16) @compileError("MK add payload ABI drift");
}
