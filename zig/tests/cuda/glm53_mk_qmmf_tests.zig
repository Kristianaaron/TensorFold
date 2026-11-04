// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-qmmf: clustered qmmf versus persistent partial/reduce tasks.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const bind_qmmf = @import("bind_qmmf");
const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(bind_qmmf);
}

const Shape = struct { name: []const u8, projection: mk.graph_api.Kind, n: usize, k: usize, bf16: bool = false };
const shapes = [_]Shape{
    .{ .name = "q_a", .projection = .q_a, .n = 2048, .k = 6144 },
    .{ .name = "q_b", .projection = .q_b, .n = 8192, .k = 2048 },
    .{ .name = "kv_a", .projection = .kv_projection, .n = 576, .k = 6144 },
    .{ .name = "o_proj", .projection = .o_proj, .n = 6144, .k = 8192 },
    .{ .name = "dense_up_gate", .projection = .dense_mlp, .n = 12288, .k = 6144 },
    .{ .name = "dense_down", .projection = .dense_mlp, .n = 6144, .k = 6144 },
    .{ .name = "head_slice", .projection = .head_logits, .n = 77440, .k = 6144, .bf16 = true },
};

fn runShape(gpu: anytype, scheduler: cuda.Module, lane: qmmf.Lane, stream: cuda.Stream, shape: Shape, rows: usize) !void {
    const npad = fp8.padded(shape.n);
    const count = shape.n * shape.k;
    const packed_codes = try gpu.gpa.alloc(u8, npad * shape.k * (if (shape.bf16) @as(usize, 2) else 1));
    defer gpu.gpa.free(packed_codes);
    var scales_device: ?cuda.DeviceBuffer = null;
    defer if (scales_device) |*buffer| buffer.free();
    var scale_ptr: u64 = 0;
    const mode: qmmf.Mode = if (shape.bf16) .bf16 else .fp8b;
    if (shape.bf16) {
        const raw = try gpu.gpa.alloc(u16, count);
        defer gpu.gpa.free(raw);
        for (raw, 0..) |*v, i| v.* = @truncate(0x3c00 + (i * 11) % 73);
        qmmf.packBf16(@as([]u16, @alignCast(std.mem.bytesAsSlice(u16, packed_codes))), raw, shape.n, shape.k);
    } else {
        const raw = try gpu.gpa.alloc(u8, count);
        defer gpu.gpa.free(raw);
        const codebook = [_]u8{ 0x38, 0x30, 0xb8, 0x40 };
        for (raw, 0..) |*code, i| code.* = codebook[(i * 17 + 3) % 4];
        fp8.packCodes(packed_codes, raw, shape.n, shape.k);
        const scale_count = ((shape.n + 127) / 128) * ((shape.k + 127) / 128);
        const scales_host = try gpu.gpa.alloc(f32, scale_count);
        defer gpu.gpa.free(scales_host);
        for (scales_host, 0..) |*scale, i| scale.* = if (i % 3 == 0) 0.5 else if (i % 3 == 1) 1 else 2;
        const packed_scales = try gpu.gpa.alloc(f32, npad / fp8.group * (shape.k / fp8.group)); // one scale per group x group tile
        defer gpu.gpa.free(packed_scales);
        try fp8.packBlockScales(packed_scales, scales_host, shape.n, shape.k, 0, 0, shape.n, shape.k);
        scales_device = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales));
        scale_ptr = scales_device.?.ptr;
    }

    const x_host = try gpu.gpa.alloc(u16, rows * shape.k);
    defer gpu.gpa.free(x_host);
    for (x_host, 0..) |*v, i| v.* = @truncate(0x3c00 + (i * 13) % 97);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x_host));
    defer x.free();
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
    defer w.free();
    const out_bytes = rows * shape.n * 2;
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(out_bytes));
    defer reference.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(out_bytes));
    defer actual.free();
    const split = try bind_qmmf.splitK(shape.n, shape.k);
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(@max(4, split * rows * shape.n * 4)));
    defer scratch.free();
    const weight: qmmf.Weight = .{ .mode = mode, .codes = w.ptr, .scales = scale_ptr, .n = @intCast(shape.n), .k = @intCast(shape.k), .npad = @intCast(npad) };
    try lane.matmul(stream, x.ptr, shape.k, rows, weight, reference.ptr, scratch.ptr);

    const bound = try bind_qmmf.bind(.{
        .rows = rows,
        .n = shape.n,
        .k = shape.k,
        .npad = npad,
        .ldx = shape.k,
        .mode = @as(bind_qmmf.Mode, @enumFromInt(@intFromEnum(mode))), // the bound subset (fp8b, bf16) shares qmmf's values
        .f32 = false,
    }, x.ptr, w.ptr, scale_ptr, actual.ptr, scratch.ptr);
    const pair = bind_qmmf.taskPayloads(bound);
    var tasks = [_]mk.graph_api.Task{
        .{ .kind = .qmmf_partial, .layer = 0, .row = 0, .aux = @backingInt(shape.projection), .successor_count = 1 },
        .{ .kind = .qmmf_reduce, .layer = 0, .row = 0, .aux = @backingInt(shape.projection), .dependency_count = 1, .successor_offset = 0 },
    };
    var counters = [_]u32{ 0, 1 };
    var successors = [_]u32{1};
    const graph: mk.graph_api.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 1, .dcp = 1 }, .tasks = &tasks, .initial_counters = &counters, .successors = &successors };
    const ops = [_]mk.Op{ .qmmf_partial_task, .qmmf_reduce_task };
    const blocks = [_]u32{ bound.partial_blocks, bound.reduce_blocks };
    const payloads = [_][]const u8{ std.mem.asBytes(&pair.partial), std.mem.asBytes(&pair.reduce) };
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, graph, &ops, &blocks, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, bound.dynamic_smem);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
    try stream.synchronize();
    const expected = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(expected);
    try reference.download(0, expected);
    const got = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(got);
    try actual.download(0, got);
    var max_abs: f32 = 0;
    var first: ?usize = null;
    for (0..rows * shape.n) |i| {
        const a: u16 = std.mem.readInt(u16, expected[i * 2 ..][0..2], .little);
        const b: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
        const av: f32 = @bitCast(@as(u32, a) << 16);
        const bv: f32 = @bitCast(@as(u32, b) << 16);
        max_abs = @max(max_abs, @abs(av - bv));
        if (a != b and first == null) first = i;
    }
    if (first) |index| {
        std.debug.print("FAIL glm53-mk-qmmf {s} rows={d} max_abs={d} first_diff={d}\n", .{ shape.name, rows, max_abs, index });
        return error.QmmfMismatch;
    }
    std.debug.print("PASS glm53-mk-qmmf {s} rows={d} max_abs={d} first_diff=none\n", .{ shape.name, rows, max_abs });
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0) return error.KernelsNotEmbedded;
    var scheduler = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer scheduler.unload();
    var lane = try qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.attribute(.compute_capability_major)));
    defer lane.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    for (shapes) |shape| inline for (.{ 1, 4 }) |rows| try runShape(gpu, scheduler, lane, stream, shape, rows);
}
