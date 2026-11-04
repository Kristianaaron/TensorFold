// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-attn: QPE pack wrapper versus one-task persistent graph.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;

const Params = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    q: u64,
    cos: u64,
    sin: u64,
    qpe: u64,
    rows: i32,
    local_heads: i32,
};
const InputNormParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, x: u64, xs: i64, w: u64, out: u64, os: i64, sums: u64, eps: f32, do_sums: i32 };
const AttendParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, qa: u64, qpe: u64, latent: u64, rope: u64, stage_latent: u64, stage_rope: u64, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial: u64, rows: i32, heads: i32, rank: i32, world: i32, prefill: i32 };
const KvWriteParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, proj: u64, norm: u64, cos: u64, sin: u64, positions: u64, latent: u64, rope: u64, stage_latent: u64, stage_rope: u64, rows: i32, stride: i32, rank: i32, world: i32 };
const AbsorbParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, q: u64, kvb_codes: u64, kvb_scales: u64, qa: u64, rows: i32, heads: i32, head0: i32 };
const MlaMergeParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, local: u64, peer: u64, out: u64, lse: u64, rows: i32, heads: i32, rank: i32 };
const ExpandVParams = extern struct { gx: u32, gy: u32, gz: u32, threads: u32, ol: u64, kvb_codes: u64, kvb_scales: u64, out: u64, rows: i32, heads: i32, head0: i32 };

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Params);
    std.testing.refAllDecls(InputNormParams);
    std.testing.refAllDecls(AttendParams);
    std.testing.refAllDecls(KvWriteParams);
    std.testing.refAllDecls(AbsorbParams);
    std.testing.refAllDecls(MlaMergeParams);
    std.testing.refAllDecls(ExpandVParams);
    if (@sizeOf(Params) != 56 or @offsetOf(Params, "rows") != 48) @compileError("attention QPE payload ABI drift");
    if (@sizeOf(InputNormParams) != 72 or @offsetOf(InputNormParams, "eps") != 64) @compileError("attention norm ABI drift");
    if (@sizeOf(AttendParams) != 144 or @offsetOf(AttendParams, "rows") != 120) @compileError("attention attend ABI drift");
    if (@sizeOf(KvWriteParams) != 104 or @offsetOf(KvWriteParams, "rows") != 88) @compileError("attention KV write ABI drift");
    if (@sizeOf(AbsorbParams) != 64 or @offsetOf(AbsorbParams, "rows") != 48) @compileError("attention absorb ABI drift");
    if (@sizeOf(MlaMergeParams) != 64 or @offsetOf(MlaMergeParams, "rows") != 48) @compileError("attention merge ABI drift");
    if (@sizeOf(ExpandVParams) != 64 or @offsetOf(ExpandVParams, "rows") != 48) @compileError("attention expand ABI drift");
}

fn download(gpu: anytype, b: cuda.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(out);
    try b.download(0, out);
    return out;
}

fn graphOne(gpu: anytype, scheduler: cuda.Module, stream: cuda.Stream, p: Params, rows: usize, heads: usize) !void {
    const tasks = try gpu.gpa.alloc(graph.Task, 1);
    defer gpu.gpa.free(tasks);
    tasks[0] = .{ .kind = .qpe_pack, .layer = 0, .row = 0 };
    const counters = try gpu.gpa.alloc(u32, 1);
    defer gpu.gpa.free(counters);
    counters[0] = 0;
    const succ = try gpu.gpa.alloc(u32, 0);
    defer gpu.gpa.free(succ);
    const g: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 1, .dcp = 1 }, .tasks = tasks, .initial_counters = counters, .successors = succ };
    const blocks = [_]u32{@intCast(rows * heads)};
    const ops = [_]mk.Op{.attn_qpe_pack};
    const payloads = [_][]const u8{std.mem.asBytes(&p)};
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &blocks, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
}

fn graphChain(gpu: anytype, scheduler: cuda.Module, stream: cuda.Stream, p: Params, rows: usize, heads: usize) !void {
    const tasks = try gpu.gpa.alloc(graph.Task, 3);
    defer gpu.gpa.free(tasks);
    for (tasks, 0..) |*task, i| task.* = .{ .kind = .qpe_pack, .layer = @intCast(i), .row = 0, .dependency_count = if (i == 0) 0 else 1, .successor_offset = if (i < 2) @intCast(i) else 2, .successor_count = if (i < 2) 1 else 0 };
    var counters = [_]u32{ 0, 1, 1 };
    var successors = [_]u32{ 1, 2 };
    const g: graph.Graph = .{ .allocator = gpu.gpa, .params = .{ .rows = rows, .layers = 3, .dcp = 1 }, .tasks = tasks, .initial_counters = &counters, .successors = &successors };
    const ops = [_]mk.Op{ .attn_qpe_pack, .attn_qpe_pack, .attn_qpe_pack };
    const block_counts = [_]u32{ @intCast(rows * heads), @intCast(rows * heads), @intCast(rows * heads) };
    const payloads = [_][]const u8{ std.mem.asBytes(&p), std.mem.asBytes(&p), std.mem.asBytes(&p) };
    var uploaded = try mk.Uploaded.init(gpu.gpa, gpu.d, g, &ops, &block_counts, &payloads, 10_000_000);
    defer uploaded.deinit();
    const plan = try mk.Plan.init(@intCast(try gpu.ctx.attribute(.multiprocessor_count)), 1, mk.task_smem_bytes);
    try std.testing.expectEqual(mk.ErrorCode.ok, try uploaded.run(scheduler, stream, plan));
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0 or cuda.kernels.glm53_mla.len == 0) return error.KernelsNotEmbedded;
    var scheduler = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
    defer scheduler.unload();
    var wrappers = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer wrappers.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const wrapper = try wrappers.function("tf_glm53_qpe_pack");
    inline for (.{ 1, 4 }) |rows| {
        const heads = 32;
        const n = rows * heads * 256;
        const qh = try gpu.gpa.alloc(u16, n);
        defer gpu.gpa.free(qh);
        for (qh, 0..) |*v, i| v.* = @truncate(0x3f00 + (i * 7) % 127);
        const ch = try gpu.gpa.alloc(f32, rows * 32);
        defer gpu.gpa.free(ch);
        const sh = try gpu.gpa.alloc(f32, rows * 32);
        defer gpu.gpa.free(sh);
        for (ch, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) / 17);
        for (sh, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) / 17);
        var q = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qh));
        defer q.free();
        var c = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ch));
        defer c.free();
        var s = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sh));
        defer s.free();
        var expected = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(rows * heads * 128 * 2));
        defer expected.free();
        var actual = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(rows * heads * 128 * 2));
        defer actual.free();
        var args: cuda.Args = .{};
        args.add(q.ptr);
        args.add(c.ptr);
        args.add(s.ptr);
        args.add(expected.ptr);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, heads));
        try cuda.launch.launch(wrapper, .{ .grid = .{ .x = @intCast(rows), .y = heads }, .block = .{ .x = 32 } }, stream, &args);
        const p: Params = .{ .gx = @intCast(rows), .gy = heads, .gz = 1, .threads = 32, .q = q.ptr, .cos = c.ptr, .sin = s.ptr, .qpe = actual.ptr, .rows = @intCast(rows), .local_heads = heads };
        try graphOne(gpu, scheduler, stream, p, rows, heads);
        try graphChain(gpu, scheduler, stream, p, rows, heads);
        try stream.synchronize();
        const expected_bytes = try download(gpu, expected);
        defer gpu.gpa.free(expected_bytes);
        const actual_bytes = try download(gpu, actual);
        defer gpu.gpa.free(actual_bytes);
        try std.testing.expectEqualSlices(u8, expected_bytes, actual_bytes);
        std.debug.print("PASS glm53-mk-attn rows={d}\n", .{rows});
    }
}
