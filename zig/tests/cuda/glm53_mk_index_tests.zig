// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-index: index wrappers versus one-task persistent graphs.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;

const QueryParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, input: u64, cos: u64, sin: u64, output: u64 };
const ScoresParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    qi: u64,
    iw: u64,
    iw_stride: i32,
    iw_col: i32,
    keys: u64,
    positions: u64,
    scores: u64,
    nkeys: i32,
    key_stride: i32,
    rank: i32,
    world: i32,
};
const KeyParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    proj: u64,
    weight: u64,
    bias: u64,
    cos: u64,
    sin: u64,
    positions: u64,
    output: u64,
    stride: i32,
    col: i32,
    rank: i32,
    world: i32,
    eps: f32,
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(QueryParams);
    std.testing.refAllDecls(ScoresParams);
    std.testing.refAllDecls(KeyParams);
    if (@sizeOf(QueryParams) != 48 or @offsetOf(QueryParams, "input") != 16) @compileError("index query payload ABI drift");
    if (@sizeOf(ScoresParams) != 80 or @offsetOf(ScoresParams, "scores") != 56) @compileError("index scores payload ABI drift");
    if (@sizeOf(KeyParams) != 96 or @offsetOf(KeyParams, "eps") != 88) @compileError("index key payload ABI drift");
}

fn uploadGraph(gpu: anytype, scheduler: cuda.Module, stream: cuda.Stream, kind: graph.Kind, op: mk.Op, blocks: u32, payload: []const u8, rows: usize) !void {
    const tasks = try gpu.gpa.alloc(graph.Task, 1);
    defer gpu.gpa.free(tasks);
    tasks[0] = .{ .kind = kind, .layer = 0, .row = 0 };
    const counters = try gpu.gpa.alloc(u32, 1);
    defer gpu.gpa.free(counters);
    counters[0] = 0;
    const successors = try gpu.gpa.alloc(u32, 0);
    defer gpu.gpa.free(successors);
    const g: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 1, .dcp = 1 }, .tasks = tasks, .initial_counters = counters, .successors = successors };
    const ops = [_]mk.Op{op};
    const block_counts = [_]u32{blocks};
    const payloads = [_][]const u8{payload};
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &block_counts, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
}

fn uploadChain(gpu: anytype, scheduler: cuda.Module, stream: cuda.Stream, kind: graph.Kind, op: mk.Op, blocks: u32, payload: []const u8, rows: usize) !void {
    const tasks = try gpu.gpa.alloc(graph.Task, 3);
    defer gpu.gpa.free(tasks);
    for (tasks, 0..) |*task, i| task.* = .{ .kind = kind, .layer = @intCast(i), .row = 0, .dependency_count = if (i == 0) 0 else 1, .successor_offset = if (i < 2) @intCast(i) else 2, .successor_count = if (i < 2) 1 else 0 };
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

fn download(gpu: anytype, b: cuda.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(out);
    try b.download(0, out);
    return out;
}

fn queryParity(gpu: anytype, wrappers: cuda.Module, scheduler: cuda.Module, stream: cuda.Stream, rows: usize) !void {
    const count = rows * 32 * 128;
    const input = try gpu.gpa.alloc(u16, count);
    defer gpu.gpa.free(input);
    for (input, 0..) |*v, i| v.* = @truncate(0x3f00 + (i * 13) % 127);
    const cos = try gpu.gpa.alloc(f32, rows * 32);
    defer gpu.gpa.free(cos);
    const sin = try gpu.gpa.alloc(f32, rows * 32);
    defer gpu.gpa.free(sin);
    for (cos, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) / 23);
    for (sin, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) / 23);
    var di = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(input));
    defer di.free();
    var dc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cos));
    defer dc.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sin));
    defer ds.free();
    var expected = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
    defer expected.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, count * 2);
    defer actual.free();
    var args: cuda.Args = .{};
    args.add(di.ptr);
    args.add(dc.ptr);
    args.add(ds.ptr);
    args.add(expected.ptr);
    try cuda.launch.launch(try wrappers.function("tf_glm53_index_q"), .{ .grid = .{ .x = 32, .y = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &args);
    const p: QueryParams = .{ .gx = 32, .gy = @intCast(rows), .gz = 1, .threads = 128, .input = di.ptr, .cos = dc.ptr, .sin = ds.ptr, .output = actual.ptr };
    try uploadGraph(gpu, scheduler, stream, .index_key_write, .index_query, @intCast(32 * rows), std.mem.asBytes(&p), rows);
    try uploadChain(gpu, scheduler, stream, .index_key_write, .index_query, @intCast(32 * rows), std.mem.asBytes(&p), rows);
    try stream.synchronize();
    const a = try download(gpu, expected);
    defer gpu.gpa.free(a);
    const b = try download(gpu, actual);
    defer gpu.gpa.free(b);
    try std.testing.expectEqualSlices(u8, a, b);
    std.debug.print("PASS glm53-mk-index query rows={d}\n", .{rows});
}

fn scoreParity(gpu: anytype, wrappers: cuda.Module, scheduler: cuda.Module, stream: cuda.Stream, rows: usize) !void {
    const nkeys: usize = 35;
    const qh = try gpu.gpa.alloc(u16, rows * 32 * 128);
    defer gpu.gpa.free(qh);
    const wh = try gpu.gpa.alloc(u16, rows * 32);
    defer gpu.gpa.free(wh);
    const kh = try gpu.gpa.alloc(u16, nkeys * 128);
    defer gpu.gpa.free(kh);
    for (qh, 0..) |*v, i| v.* = @truncate(0x3e80 + (i * 5) % 127);
    for (wh, 0..) |*v, i| v.* = @truncate(0x3f00 + (i * 11) % 31);
    for (kh, 0..) |*v, i| v.* = @truncate(0x3e00 + (i * 7) % 95);
    const posh = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(posh);
    @memset(posh, @intCast(nkeys - 1));
    var q = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qh));
    defer q.free();
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wh));
    defer w.free();
    var k = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(kh));
    defer k.free();
    var pos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(posh));
    defer pos.free();
    var expected = try cuda.DeviceBuffer.alloc(gpu.d, rows * nkeys * 4);
    defer expected.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, rows * nkeys * 4);
    defer actual.free();
    var args: cuda.Args = .{};
    args.add(q.ptr);
    args.add(w.ptr);
    args.add(@as(i32, 32));
    args.add(@as(i32, 0));
    args.add(k.ptr);
    args.add(pos.ptr);
    args.add(expected.ptr);
    args.add(@as(i32, @intCast(nkeys)));
    args.add(@as(i32, @intCast(nkeys)));
    args.add(@as(i32, 0));
    args.add(@as(i32, 1));
    const gx: u32 = @intCast((nkeys + 15) / 16);
    try cuda.launch.launch(try wrappers.function("tf_glm53_index_scores"), .{ .grid = .{ .x = gx, .y = @intCast(rows) }, .block = .{ .x = 256 }, .shared = 11_392 }, stream, &args);
    const p: ScoresParams = .{ .gx = gx, .gy = @intCast(rows), .gz = 1, .threads = 256, .qi = q.ptr, .iw = w.ptr, .iw_stride = 32, .iw_col = 0, .keys = k.ptr, .positions = pos.ptr, .scores = actual.ptr, .nkeys = @intCast(nkeys), .key_stride = @intCast(nkeys), .rank = 0, .world = 1 };
    try uploadGraph(gpu, scheduler, stream, .index_scores, .index_scores, gx * @as(u32, @intCast(rows)), std.mem.asBytes(&p), rows);
    try uploadChain(gpu, scheduler, stream, .index_scores, .index_scores, gx * @as(u32, @intCast(rows)), std.mem.asBytes(&p), rows);
    try stream.synchronize();
    const a = try download(gpu, expected);
    defer gpu.gpa.free(a);
    const b = try download(gpu, actual);
    defer gpu.gpa.free(b);
    try std.testing.expectEqualSlices(u8, a, b);
    std.debug.print("PASS glm53-mk-index scores rows={d}\n", .{rows});
}

fn keyParity(gpu: anytype, wrappers: cuda.Module, scheduler: cuda.Module, stream: cuda.Stream, rows: usize) !void {
    const stride: usize = 640;
    const projh = try gpu.gpa.alloc(u16, rows * stride);
    defer gpu.gpa.free(projh);
    const wh = try gpu.gpa.alloc(u16, 128);
    defer gpu.gpa.free(wh);
    const bh = try gpu.gpa.alloc(u16, 128);
    defer gpu.gpa.free(bh);
    const ch = try gpu.gpa.alloc(f32, rows * 32);
    defer gpu.gpa.free(ch);
    const sh = try gpu.gpa.alloc(f32, rows * 32);
    defer gpu.gpa.free(sh);
    const posh = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(posh);
    for (projh, 0..) |*v, i| v.* = @truncate(0x3e80 + (i * 3) % 127);
    for (wh, 0..) |*v, i| v.* = @truncate(0x3f00 + (i * 5) % 31);
    for (bh, 0..) |*v, i| v.* = @truncate(0x3d00 + (i * 7) % 63);
    for (ch, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) / 19);
    for (sh, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) / 19);
    for (posh, 0..) |*v, i| v.* = @intCast(i);
    var proj = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(projh));
    defer proj.free();
    var weight = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wh));
    defer weight.free();
    var bias = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bh));
    defer bias.free();
    var cos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ch));
    defer cos.free();
    var sin = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sh));
    defer sin.free();
    var positions = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(posh));
    defer positions.free();
    var expected = try cuda.DeviceBuffer.alloc(gpu.d, rows * 128 * 2);
    defer expected.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, rows * 128 * 2);
    defer actual.free();
    const eps: f32 = 1e-6;
    var args: cuda.Args = .{};
    args.add(proj.ptr);
    args.add(weight.ptr);
    args.add(bias.ptr);
    args.add(cos.ptr);
    args.add(sin.ptr);
    args.add(positions.ptr);
    args.add(expected.ptr);
    args.add(@as(i32, @intCast(stride)));
    args.add(@as(i32, 0));
    args.add(@as(i32, 0));
    args.add(@as(i32, 1));
    args.add(eps);
    try cuda.launch.launch(try wrappers.function("tf_glm53_index_key"), .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &args);
    const p: KeyParams = .{ .gx = @intCast(rows), .gy = 1, .gz = 1, .threads = 128, .proj = proj.ptr, .weight = weight.ptr, .bias = bias.ptr, .cos = cos.ptr, .sin = sin.ptr, .positions = positions.ptr, .output = actual.ptr, .stride = @intCast(stride), .col = 0, .rank = 0, .world = 1, .eps = eps };
    try uploadGraph(gpu, scheduler, stream, .index_key_write, .index_key, @intCast(rows), std.mem.asBytes(&p), rows);
    try uploadChain(gpu, scheduler, stream, .index_key_write, .index_key, @intCast(rows), std.mem.asBytes(&p), rows);
    try stream.synchronize();
    const a = try download(gpu, expected);
    defer gpu.gpa.free(a);
    const b = try download(gpu, actual);
    defer gpu.gpa.free(b);
    try std.testing.expectEqualSlices(u8, a, b);
    std.debug.print("PASS glm53-mk-index key rows={d}\n", .{rows});
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_index.len == 0 or cuda.kernels.glm53_megakernel.len == 0) return error.KernelsNotEmbedded;
    var wrappers = try cuda.Module.load(gpu.d, cuda.kernels.glm53_index);
    defer wrappers.unload();
    var scheduler = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer scheduler.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    inline for (.{ 1, 4 }) |rows| {
        try queryParity(gpu, wrappers, scheduler, stream, rows);
        try keyParity(gpu, wrappers, scheduler, stream, rows);
        try scoreParity(gpu, wrappers, scheduler, stream, rows);
    }
}
