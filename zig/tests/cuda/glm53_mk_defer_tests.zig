// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-cuda-test glm53-mk-defer: eight real GLM-5.3 decode kernels (rmsnorm_6144, qpe_pack, attend, swiglu, low-rank
//! partial/reduce/finish, MoE residual add) on seeded data, run normally and through the MK8 deferred-launch sink (eager,
//! cached plan, inside a CUDA graph capture, and a cold capture that falls back): every output bit-identical.
//! The low-rank finish is part of the deferred sequence, so the eight-kernel decode slice is one segment.
//! Rows 1, 2, 4 and 8 (4 is the speculative-decode verify shape). Each case also reports, as numbers and not as
//! pass/fail thresholds: per-kernel normal launch times (events between launches), the per-task breakdown of the
//! persistent run (globaltimer trace: first block start to last block end, mean block time, scheduling gap), and
//! the step time of the MK8 launch shapes (one CTA per SM with the 48 KiB arena vs the occupancy-sized grid).
//! tf-cuda-test glm53-mk-policy (runPolicy) times every launch shape on the same slice and reports the launch-shape
//! policy's choice on this GPU against the measured best.
const std = @import("std");
const cuda = @import("cuda");
const mk_defer = @import("mk_defer");
const mk = mk_defer.launch_api;

const hidden = 6144;
const heads = 32;
const latent_dim = 512;
const rope_dim = 64;
const cache_slots = 2048;
const top_k = 2048;
const rank_local = 32;
const splits = 8;
const swiglu_width = hidden / 2;
const partial_row = heads * (latent_dim * 2 + 4);

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Inputs);
    std.testing.refAllDecls(Set);
}

const task_names = [_][]const u8{ "rmsnorm_6144", "qpe_pack", "attend", "swiglu", "lowrank_partial", "lowrank_reduce", "lowrank_finish", "moe_residual_add" };
const task_count = task_names.len;

/// Each kernel's launch as step() issues it: blocks (grid volume) and threads per block.
fn launchShapes(rows: usize) [task_count][2]u32 {
    const r: u32 = @intCast(rows);
    return .{
        .{ r, 256 },                             .{ r * heads, 32 },                             .{ r * heads, 256 },
        .{ blocksOf(rows * swiglu_width), 256 }, .{ blocksOf(rows * rank_local * splits), 256 }, .{ blocksOf(rows * rank_local), 256 },
        .{ blocksOf(rows * hidden), 256 },       .{ blocksOf(rows * hidden), 256 },
    };
}

const Kernels = struct {
    rmsnorm: cuda.Function,
    qpe: cuda.Function,
    attend: cuda.Function,
    swiglu: cuda.Function,
    partial: cuda.Function,
    reduce: cuda.Function,
    finish: cuda.Function,
    residual: cuda.Function,
};

fn go(f: cuda.Function, grid: cuda.Dim3, threads: u32, stream: cuda.Stream, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, stream, &a);
}

fn blocksOf(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// Read-only inputs shared by every run.
const Inputs = struct {
    rows: usize,
    bufs: [13]cuda.DeviceBuffer,
    x0: []u16,
    // indices into bufs
    const w = 0;
    const q = 1;
    const cos = 2;
    const sin = 3;
    const qa = 4;
    const latent = 5;
    const rope = 6;
    const positions = 7;
    const slots = 8;
    const counts = 9;
    const dense_len = 10;
    const a = 11;
    const b_parts = 12;

    fn ptr(in: *const Inputs, i: usize) u64 {
        return in.bufs[i].ptr;
    }
    fn deinit(in: *Inputs, gpa: std.mem.Allocator) void {
        for (&in.bufs) |*b| b.free();
        gpa.free(in.x0);
    }
};

fn fillBf16(r: std.Random, out: []u16) void {
    for (out) |*v| v.* = @truncate(@as(u32, @bitCast(r.float(f32) * 2 - 1)) >> 16);
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(d, bytes);
}

fn inputs(gpu: anytype, rows: usize, seed: u64) !Inputs {
    const gpa = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var in: Inputs = .{ .rows = rows, .bufs = undefined, .x0 = try gpa.alloc(u16, rows * hidden) };
    errdefer gpa.free(in.x0);
    fillBf16(r, in.x0);
    var made: usize = 0;
    errdefer for (in.bufs[0..made]) |*b| b.free();
    const sizes_bf16 = [_]struct { usize, usize }{
        .{ Inputs.w, hidden },                     .{ Inputs.q, rows * heads * 256 },
        .{ Inputs.qa, rows * heads * latent_dim }, .{ Inputs.latent, cache_slots * latent_dim },
        .{ Inputs.rope, cache_slots * rope_dim },  .{ Inputs.b_parts, 2 * rows * hidden },
    };
    var order: [13]?cuda.DeviceBuffer = @splat(null);
    for (sizes_bf16) |e| {
        const host = try gpa.alloc(u16, e[1]);
        defer gpa.free(host);
        fillBf16(r, host);
        order[e[0]] = try upload(gpu.d, std.mem.sliceAsBytes(host));
    }
    const cs = try gpa.alloc(f32, rows * rope_dim / 2);
    defer gpa.free(cs);
    for (cs, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) / 7);
    order[Inputs.cos] = try upload(gpu.d, std.mem.sliceAsBytes(cs));
    for (cs, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) / 7);
    order[Inputs.sin] = try upload(gpu.d, std.mem.sliceAsBytes(cs));
    const pos = try gpa.alloc(i64, rows);
    defer gpa.free(pos);
    const dense = try gpa.alloc(i32, rows);
    defer gpa.free(dense);
    for (pos, dense, 0..) |*p, *dl, i| {
        p.* = @intCast(i * 37 + 300); // dense rows: rank 0 reads slots 0..p/2 of its own plane
        dl.* = @intCast(p.* + 1);
    }
    order[Inputs.positions] = try upload(gpu.d, std.mem.sliceAsBytes(pos));
    order[Inputs.dense_len] = try upload(gpu.d, std.mem.sliceAsBytes(dense));
    const slots = try gpa.alloc(i32, rows * top_k);
    defer gpa.free(slots);
    @memset(slots, -1);
    order[Inputs.slots] = try upload(gpu.d, std.mem.sliceAsBytes(slots));
    const counts = try gpa.alloc(i32, rows);
    defer gpa.free(counts);
    @memset(counts, 0);
    order[Inputs.counts] = try upload(gpu.d, std.mem.sliceAsBytes(counts));
    const a = try gpa.alloc(f32, hidden * rank_local);
    defer gpa.free(a);
    for (a) |*v| v.* = (r.float(f32) * 2 - 1) * 0.05;
    order[Inputs.a] = try upload(gpu.d, std.mem.sliceAsBytes(a));
    for (order, 0..) |o, i| {
        in.bufs[i] = o orelse return error.Invalid;
        made = i + 1;
    }
    return in;
}

/// One run's outputs (and its in-place residual, low-rank and finish buffers), freshly initialised.
const Set = struct {
    bufs: [8]cuda.DeviceBuffer,
    const x = 0; // residual, updated in place by the MoE residual add
    const xn = 1;
    const qpe = 2;
    const partial = 3;
    const act = 4;
    const mid = 5;
    const acc = 6;
    const b = 7; // low-rank B for the finish
    const names = [_][]const u8{ "x", "xn", "qpe", "partial", "act", "mid", "acc", "b" };

    fn ptr(s: *const Set, i: usize) u64 {
        return s.bufs[i].ptr;
    }
    fn deinit(s: *Set) void {
        for (&s.bufs) |*b_| b_.free();
    }
};

fn set(gpu: anytype, in: *const Inputs) !Set {
    const gpa = gpu.gpa;
    const rows = in.rows;
    var s: Set = .{ .bufs = undefined };
    var made: usize = 0;
    errdefer for (s.bufs[0..made]) |*b_| b_.free();
    s.bufs[Set.x] = try upload(gpu.d, std.mem.sliceAsBytes(in.x0));
    made += 1;
    const zero_sizes = [_]usize{ rows * hidden * 2, rows * heads * rope_dim * 2, rows * partial_row, rows * swiglu_width * 2, splits * rows * rank_local * 4, rows * hidden * 4 };
    for (zero_sizes) |n| {
        const z = try gpa.alloc(u8, n);
        defer gpa.free(z);
        @memset(z, 0);
        s.bufs[made] = try upload(gpu.d, z);
        made += 1;
    }
    const b = try gpa.alloc(f32, rank_local * hidden);
    defer gpa.free(b);
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 23)) - 11)) * 0.0078125;
    s.bufs[Set.b] = try upload(gpu.d, std.mem.sliceAsBytes(b));
    return s;
}

/// The synthetic decode slice, launched exactly as Forward's wrappers launch these kernels.
fn step(k: *const Kernels, in: *const Inputs, s: *const Set, stream: cuda.Stream) !void {
    return stepMarked(k, in, s, stream, null);
}

/// step() with an event recorded before each kernel and after the last (normal launches only: an event record
/// flushes a deferred segment).
fn stepMarked(k: *const Kernels, in: *const Inputs, s: *const Set, stream: cuda.Stream, marks: ?*const [task_count + 1]cuda.Event) !void {
    const rows = in.rows;
    const r32: i32 = @intCast(rows);
    var at: usize = 0;
    const mark = struct {
        fn f(m: ?*const [task_count + 1]cuda.Event, i: *usize, st: cuda.Stream) !void {
            if (m) |e| try e[i.*].record(st);
            i.* += 1;
        }
    }.f;
    // glue.rmsnorm (6144): grid rows, 8 warps, no sums
    try mark(marks, &at, stream);
    try go(k.rmsnorm, .{ .x = @intCast(rows) }, 256, stream, .{ s.ptr(Set.x), @as(i64, hidden), in.ptr(Inputs.w), s.ptr(Set.xn), @as(i64, hidden), @as(u64, 0), @as(f32, 1e-5), @as(c_int, 0) });
    try mark(marks, &at, stream);
    // mla.packQpe
    try go(k.qpe, .{ .x = @intCast(rows), .y = heads }, 32, stream, .{ in.ptr(Inputs.q), in.ptr(Inputs.cos), in.ptr(Inputs.sin), s.ptr(Set.qpe), r32, @as(i32, heads) });
    try mark(marks, &at, stream);
    // mla.attend, decode (DCP2 rank 0, dense rows, no stage)
    try go(k.attend, .{ .x = @intCast(rows), .y = heads, .z = 1 }, 256, stream, .{
        in.ptr(Inputs.qa),     s.ptr(Set.qpe),           in.ptr(Inputs.latent), in.ptr(Inputs.rope),  @as(u64, 0),
        @as(u64, 0),           in.ptr(Inputs.positions), in.ptr(Inputs.slots),  in.ptr(Inputs.slots), in.ptr(Inputs.counts),
        in.ptr(Inputs.counts), in.ptr(Inputs.dense_len), s.ptr(Set.partial),    r32,                  @as(i32, heads),
        @as(i32, 0),           @as(i32, 2),              @as(i32, 0),
    });
    try mark(marks, &at, stream);
    // mlp.swigluPlain over the normed rows as [gate | up]
    try go(k.swiglu, .{ .x = blocksOf(rows * swiglu_width) }, 256, stream, .{ s.ptr(Set.xn), s.ptr(Set.act), @as(c_int, r32), @as(c_int, swiglu_width) });
    try mark(marks, &at, stream);
    // moe.block low-rank partial and in-place reduce
    try go(k.partial, .{ .x = blocksOf(rows * rank_local * splits) }, 256, stream, .{ s.ptr(Set.xn), in.ptr(Inputs.a), s.ptr(Set.mid), @as(c_int, r32), @as(c_int, hidden), @as(c_int, rank_local), @as(c_int, hidden), @as(c_int, rank_local), @as(c_int, splits) });
    try mark(marks, &at, stream);
    try go(k.reduce, .{ .x = blocksOf(rows * rank_local) }, 256, stream, .{ s.ptr(Set.mid), s.ptr(Set.mid), @as(c_int, @intCast(rows * rank_local)), @as(c_int, splits) });
    try mark(marks, &at, stream);
    // moe.lowRankAdd's finish
    try go(k.finish, .{ .x = blocksOf(rows * hidden) }, 256, stream, .{ s.ptr(Set.b), s.ptr(Set.mid), s.ptr(Set.acc), @as(c_int, r32), @as(c_int, hidden), @as(c_int, rank_local), @as(c_int, hidden) });
    try mark(marks, &at, stream);
    // moe.residualAddBf16, in place
    try go(k.residual, .{ .x = blocksOf(rows * hidden) }, 256, stream, .{ s.ptr(Set.x), s.ptr(Set.x), in.ptr(Inputs.b_parts), @as(c_int, @intCast(rows * hidden)) });
    try mark(marks, &at, stream);
}

fn same(gpu: anytype, want: *const Set, got: *const Set, what: []const u8, rows: usize) !void {
    for (want.bufs, got.bufs, Set.names) |wb, gb, name| {
        const a = try gpu.gpa.alloc(u8, wb.len);
        defer gpu.gpa.free(a);
        const b = try gpu.gpa.alloc(u8, gb.len);
        defer gpu.gpa.free(b);
        try wb.download(0, a);
        try gb.download(0, b);
        if (!std.mem.eql(u8, a, b)) {
            const at = std.mem.indexOfDiff(u8, a, b).?;
            std.debug.print("FAIL glm53-mk-defer rows={d} {s}: {s} differs at byte {d}\n", .{ rows, what, name, at });
            return error.TestFailed;
        }
    }
}

fn timed(gpu: anytype, stream: cuda.Stream, comptime body: anytype, ctx: anytype, reps: usize) !f32 {
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    try body(ctx, stream);
    try start.record(stream);
    for (0..reps) |_| try body(ctx, stream);
    try stop.record(stream);
    try stop.synchronize();
    return try cuda.Event.elapsedMs(start, stop) * 1000 / @as(f32, @floatFromInt(reps));
}

const Ctx = struct { k: *const Kernels, in: *const Inputs, s: *const Set };

fn stepCtx(c: Ctx, stream: cuda.Stream) !void {
    try step(c.k, c.in, c.s, stream);
    try stream.flushDeferred();
}

fn rowsCase(gpu: anytype, k: *const Kernels, stream: cuda.Stream, rows: usize) !void {
    var in = try inputs(gpu, rows, 0x6d6b38 + rows);
    defer in.deinit(gpu.gpa);
    var want = try set(gpu, &in);
    defer want.deinit();
    try step(k, &in, &want, stream);
    try stream.synchronize();

    // The chain scheduler with the SM map, pinned (glm53-mk-policy covers the per-device choice).
    const chain_sm: mk_defer.policy.Policy = .{ .override = .chain_sm };
    const x = try mk_defer.Executor.createWith(gpu.gpa, gpu.d, chain_sm);
    defer x.destroy();
    const ds = stream.withDeferred(x.sink());
    // 1. eager: plans uploaded on first use
    var eager = try set(gpu, &in);
    defer eager.deinit();
    try step(k, &in, &eager, ds);
    try ds.flushDeferred();
    try stream.synchronize();
    if (try x.gpu.check() != .ok) return error.TestFailed;
    try same(gpu, &want, &eager, "eager", rows);
    // 2. cached plans, new buffers: payload bytes refreshed
    var hit = try set(gpu, &in);
    defer hit.deinit();
    try step(k, &in, &hit, ds);
    try ds.flushDeferred();
    try stream.synchronize();
    try same(gpu, &want, &hit, "cached plan", rows);
    // 3. inside a CUDA graph capture (plans cached: store, reset and run nodes only)
    var captured = try set(gpu, &in);
    defer captured.deinit();
    try cuda.graph.beginCapture(stream, .thread_local);
    step(k, &in, &captured, ds) catch |err| {
        if (cuda.graph.endCapture(stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    try ds.flushDeferred();
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    if (try x.gpu.check() != .ok) return error.TestFailed;
    try same(gpu, &want, &captured, "graph replay", rows);
    const st = x.deferral.stats;
    if (st.fallback_segments != 0 or st.plans != 1 or st.segments != 3 or st.tasks != 24) {
        std.debug.print("FAIL glm53-mk-defer rows={d} stats {any}\n", .{ rows, st });
        return error.TestFailed;
    }
    // 4. a cold sink inside a capture: no plan to upload, so the recorded launches go in unchanged
    const cold = try mk_defer.Executor.createWith(gpu.gpa, gpu.d, chain_sm);
    defer cold.destroy();
    const cs = stream.withDeferred(cold.sink());
    var fallback = try set(gpu, &in);
    defer fallback.deinit();
    try cuda.graph.beginCapture(stream, .thread_local);
    step(k, &in, &fallback, cs) catch |err| {
        if (cuda.graph.endCapture(stream)) |g2| {
            var gg = g2;
            gg.deinit();
        } else |_| {}
        return err;
    };
    try cs.flushDeferred();
    var g2 = try cuda.graph.endCapture(stream);
    defer g2.deinit();
    var exec2 = try g2.instantiate();
    defer exec2.deinit();
    try exec2.launchOn(stream);
    try stream.synchronize();
    try same(gpu, &want, &fallback, "capture fallback", rows);
    if (cold.deferral.stats.fallback_segments != 1 or cold.deferral.stats.plans != 0) return error.TestFailed;

    // Timings (outputs of these repeats are not checked: the residual and finish update in place).
    // `want` stays the reference: the normal-launch repeats run on their own buffers.
    const reps = 50;
    var scratch = try set(gpu, &in);
    defer scratch.deinit();
    const normal_us = try timed(gpu, stream, stepCtx, Ctx{ .k = k, .in = &in, .s = &scratch }, reps);
    const deferred_us = try timed(gpu, ds, stepCtx, Ctx{ .k = k, .in = &in, .s = &hit }, reps);
    var replay_start = try cuda.Event.init(gpu.d, true);
    defer replay_start.deinit();
    var replay_stop = try cuda.Event.init(gpu.d, true);
    defer replay_stop.deinit();
    try replay_start.record(stream);
    for (0..reps) |_| try exec.launchOn(stream);
    try replay_stop.record(stream);
    try replay_stop.synchronize();
    const graph_us = try cuda.Event.elapsedMs(replay_start, replay_stop) * 1000 / reps;
    const all = x.deferral.stats;
    const plan = x.gpu.uploaded.items[0].plan;
    const fell_back = try x.gpu.smMapFellBack();

    // Launch shapes: the MK8 shape before the fix (the dependency queue, 48 KiB arena, one CTA per SM), the queue at
    // full occupancy, and the chain scheduler with both worker mappings.
    const shapes = [_]struct { []const u8, mk_defer.LaunchShape }{
        .{ "queue, 48K arena, 1/SM (before)", .queue },
        .{ "queue, occupancy", .queue_occ },
        .{ "chain, blockIdx map", .chain_block },
        .{ "chain, SM map (default)", .chain_sm },
    };
    var shape_us: [shapes.len][2]f32 = undefined;
    for (shapes, 0..) |sh, i| shape_us[i] = try shapeTimes(gpu, k, &in, &want, stream, sh[1], reps);

    // Per-task breakdown: normal launches (events between kernels) vs the persistent run's trace.
    const normal = try normalBreakdown(gpu, k, &in, &scratch, stream, 20);
    const legacy = try traceBreakdown(gpu, k, &in, &want, stream, .queue, 20);
    const fixed = try traceBreakdown(gpu, k, &in, &want, stream, .chain_sm, 20);
    const launches = launchShapes(rows);
    const kfns = [_]cuda.Function{ k.rmsnorm, k.qpe, k.attend, k.swiglu, k.partial, k.reduce, k.finish, k.residual };
    std.debug.print("RESULT glm53-mk-defer rows={d} per task (us): kernel blocks x threads [regs, resident/SM] | normal | queue 1/SM 48K: span block gap | chain SM map: span block gap\n", .{rows});
    for (task_names, 0..) |name, i| {
        std.debug.print("RESULT glm53-mk-defer rows={d}   {s:<16} {d:>4} x {d:<3} [{d:>3}, {d:>2}] | {d:>8.1} | {d:>8.1} {d:>8.1} {d:>6.1} | {d:>8.1} {d:>8.1} {d:>6.1}\n", .{
            rows,                                                 name,                                     launches[i][0], launches[i][1],
            @as(u32, @intCast(try kfns[i].attribute(.num_regs))), try kfns[i].occupancy(launches[i][1], 0), normal[i],      legacy.span[i],
            legacy.block[i],                                      legacy.gap[i],                            fixed.span[i],  fixed.block[i],
            fixed.gap[i],
        });
    }
    var normal_sum: f32 = 0;
    for (normal) |v| normal_sum += v;
    std.debug.print("RESULT glm53-mk-defer rows={d}   sum of kernels {d:.1} us | persistent run, queue 1/SM 48K {d:.1} us ({d} CTAs) | chain SM map {d:.1} us ({d} CTAs)\n", .{ rows, normal_sum, legacy.total, legacy.ctas, fixed.total, fixed.ctas });
    for (shapes, shape_us) |sh, t| std.debug.print("RESULT glm53-mk-defer rows={d} shape {s:<32} deferred {d:>8.1} us, deferred graph {d:>8.1} us\n", .{ rows, sh[0], t[0], t[1] });
    std.debug.print("PASS glm53-mk-defer rows={d}: bit-identical (eager, cached plan, graph replay, capture fallback, every launch shape, traced runs); per step 1 segment, 8 deferred tasks; plans={d} hits={d} segments={d} tasks={d}; chain grid {d} CTAs (SM map {s}), arena {d} B; normal {d:.1} us, deferred {d:.1} us, deferred graph {d:.1} us (queue 1/SM 48K arena: {d:.1} / {d:.1} us)\n", .{ rows, all.plans, all.hits, all.segments, all.tasks, plan.persistent_blocks, if (fell_back) "fell back to blockIdx" else "held", plan.dynamic_smem, normal_us, deferred_us, graph_us, shape_us[0][0], shape_us[0][1] });
}

/// Mean per-kernel time of the normal launches, from events recorded between them.
fn normalBreakdown(gpu: anytype, k: *const Kernels, in: *const Inputs, s: *const Set, stream: cuda.Stream, reps: usize) ![task_count]f32 {
    var events: [task_count + 1]cuda.Event = undefined;
    var made: usize = 0;
    defer for (events[0..made]) |*e| e.deinit();
    for (&events) |*e| {
        e.* = try cuda.Event.init(gpu.d, true);
        made += 1;
    }
    try step(k, in, s, stream);
    var sum: [task_count]f32 = @splat(0);
    for (0..reps) |_| {
        try stepMarked(k, in, s, stream, &events);
        try events[task_count].synchronize();
        for (&sum, 0..) |*v, i| v.* += try cuda.Event.elapsedMs(events[i], events[i + 1]) * 1000;
    }
    for (&sum) |*v| v.* /= @floatFromInt(reps);
    return sum;
}

const Breakdown = struct {
    /// First block start to last block end, per task (us).
    span: [task_count]f64 = @splat(0),
    /// Mean time of one block of the task (us).
    block: [task_count]f64 = @splat(0),
    /// Previous task's last block end (the first worker's entry for task 0) to this task's first block start (us).
    gap: [task_count]f64 = @splat(0),
    /// First worker entry to last worker exit (us).
    total: f64 = 0,
    ctas: u32 = 0,
};

/// The persistent run's per-task trace (mk_runtime.cuh tf_glm53_mk_state.trace), averaged over `reps` steps.
/// Every traced step's outputs are checked too: the trace stamps must not change a bit.
fn traceBreakdown(gpu: anytype, k: *const Kernels, in: *const Inputs, want: *const Set, stream: cuda.Stream, shape: mk_defer.LaunchShape, reps: usize) !Breakdown {
    const x = try mk_defer.Executor.createWith(gpu.gpa, gpu.d, .{ .override = shape });
    defer x.destroy();
    const words = mk.traceLen(task_count);
    var buf = try cuda.DeviceBuffer.alloc(gpu.d, words * 8);
    defer buf.free();
    const host = try gpu.gpa.alloc(u64, words);
    defer gpu.gpa.free(host);
    x.gpu.trace = buf.ptr;
    const ds = stream.withDeferred(x.sink());
    const blocks = launchShapes(in.rows);
    var out: Breakdown = .{};
    for (0..reps + 1) |rep| {
        var s = try set(gpu, in);
        defer s.deinit();
        mk.traceInit(host);
        try stream.synchronize();
        try buf.upload(0, std.mem.sliceAsBytes(host));
        try step(k, in, &s, ds);
        try ds.flushDeferred();
        try stream.synchronize();
        if (try x.gpu.check() != .ok) return error.TestFailed;
        try same(gpu, want, &s, "traced run", in.rows);
        if (rep == 0) continue; // the first run uploads the plan
        try buf.download(0, std.mem.sliceAsBytes(host));
        const w = mk.trace_words;
        const entry = host[task_count * w];
        const exit = host[task_count * w + 1];
        out.total += @as(f64, @floatFromInt(exit - entry)) / 1000;
        for (0..task_count) |i| {
            const first = host[i * w];
            const last = host[i * w + 1];
            const before = if (i == 0) entry else host[(i - 1) * w + 1];
            out.span[i] += @as(f64, @floatFromInt(last - first)) / 1000;
            out.block[i] += @as(f64, @floatFromInt(host[i * w + 2])) / 1000 / @as(f64, @floatFromInt(blocks[i][0]));
            out.gap[i] += (@as(f64, @floatFromInt(first)) - @as(f64, @floatFromInt(before))) / 1000;
        }
    }
    const n: f64 = @floatFromInt(reps);
    out.total /= n;
    for (0..task_count) |i| {
        out.span[i] /= n;
        out.block[i] /= n;
        out.gap[i] /= n;
    }
    out.ctas = x.gpu.uploaded.items[0].plan.persistent_blocks;
    return out;
}

/// One launch shape: eager deferred and graph-replay step times, after checking both against the normal launches.
fn shapeTimes(gpu: anytype, k: *const Kernels, in: *const Inputs, want: *const Set, stream: cuda.Stream, shape: mk_defer.LaunchShape, reps: usize) ![2]f32 {
    return policyTimes(gpu, k, in, want, stream, .{ .override = shape }, reps, null);
}

/// What a policy chose for the slice's one plan key, and where the choice came from.
const Picked = struct { shape: mk_defer.LaunchShape, source: mk_defer.policy.Source, eager_windows: usize };

/// An executor under policy `p`: eager windows (fresh buffers each, every one checked bit for bit) until no key
/// is calibrating, then a graph capture and its replay (checked), then the eager and graph-replay step times.
fn policyTimes(gpu: anytype, k: *const Kernels, in: *const Inputs, want: *const Set, stream: cuda.Stream, p: mk_defer.policy.Policy, reps: usize, picked: ?*Picked) ![2]f32 {
    const x = try mk_defer.Executor.createWith(gpu.gpa, gpu.d, p);
    defer x.destroy();
    const ds = stream.withDeferred(x.sink());
    var eager = try set(gpu, in);
    defer eager.deinit();
    const label = if (p.override) |o| o.name() else "policy";
    var windows: usize = 0;
    while (true) {
        var w = try set(gpu, in);
        defer w.deinit();
        try step(k, in, &w, ds);
        try ds.flushDeferred();
        try stream.synchronize();
        if (try x.gpu.check() != .ok) return error.TestFailed;
        windows += 1;
        same(gpu, want, &w, "launch shape eager", in.rows) catch |err| {
            std.debug.print("FAIL glm53-mk-defer launch shape {s}, eager window {d}\n", .{ label, windows });
            return err;
        };
        if (!x.deferral.calibrating()) break;
        if (windows > mk_defer.max_calibration_windows) return error.TestFailed;
    }
    if (x.deferral.planList().len != 1) return error.TestFailed;
    if (picked) |out| out.* = .{ .shape = x.deferral.planList()[0].shape, .source = x.deferral.planList()[0].source, .eager_windows = windows };
    var captured = try set(gpu, in);
    defer captured.deinit();
    try cuda.graph.beginCapture(stream, .thread_local);
    step(k, in, &captured, ds) catch |err| {
        if (cuda.graph.endCapture(stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    try ds.flushDeferred();
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    if (try x.gpu.check() != .ok) return error.TestFailed;
    same(gpu, want, &captured, "launch shape graph replay", in.rows) catch |err| {
        std.debug.print("FAIL glm53-mk-defer launch shape {s}, graph replay\n", .{label});
        return err;
    };
    const eager_us = try timed(gpu, ds, stepCtx, Ctx{ .k = k, .in = in, .s = &eager }, reps);
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    try start.record(stream);
    for (0..reps) |_| try exec.launchOn(stream);
    try stop.record(stream);
    try stop.synchronize();
    return .{ eager_us, try cuda.Event.elapsedMs(start, stop) * 1000 / @as(f32, @floatFromInt(reps)) };
}

const Loaded = struct {
    glue: cuda.Module,
    glue53: cuda.Module,
    mla: cuda.Module,
    k: Kernels,

    fn deinit(l: *Loaded) void {
        l.mla.unload();
        l.glue53.unload();
        l.glue.unload();
    }
};

fn load(gpu: anytype) !Loaded {
    if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0 or cuda.kernels.glm53_glue.len == 0 or
        cuda.kernels.glm53_mla.len == 0 or cuda.kernels.glm_glue.len == 0) return error.KernelsNotEmbedded;
    var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    errdefer glue.unload();
    var glue53 = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
    errdefer glue53.unload();
    var mla = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    errdefer mla.unload();
    return .{ .glue = glue, .glue53 = glue53, .mla = mla, .k = .{
        .rmsnorm = try glue.function("tf_glm_rmsnorm_6144"),
        .qpe = try mla.function("tf_glm53_qpe_pack"),
        .attend = try mla.function("tf_glm53_attend"),
        .swiglu = try glue53.function("tf_glm53_swiglu_plain"),
        .partial = try glue53.function("tf_glm53_lowrank_partial"),
        .reduce = try glue53.function("tf_glm53_lowrank_reduce"),
        .finish = try glue53.function("tf_glm53_lowrank_finish"),
        .residual = try glue53.function("tf_glm53_moe_residual_add"),
    } };
}

pub fn run(gpu: anytype) !void {
    var l = try load(gpu);
    defer l.deinit();
    const k = l.k;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    {
        var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_megakernel);
        defer module.unload();
        for ([_][:0]const u8{ "tf_glm53_mk_run", "tf_glm53_mk_run_chain" }) |name| {
            const f = try module.function(name);
            std.debug.print("RESULT glm53-mk-defer interpreter {s}: {d} registers, {d} B static shared, resident CTAs/SM {d} (0 B arena) {d} (48 KiB arena)\n", .{
                name, @as(u32, @intCast(try f.attribute(.num_regs))), @as(u32, @intCast(try f.attribute(.shared_size_bytes))), try f.occupancy(mk.block_threads, 0),
                blk: {
                    const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
                    if (mk.task_smem_bytes + static > 48 * 1024) try f.allowDynamicShared(mk.task_smem_bytes);
                    break :blk try f.occupancy(mk.block_threads, mk.task_smem_bytes);
                },
            });
        }
    }
    inline for (.{ 1, 2, 4, 8 }) |rows| try rowsCase(gpu, &k, stream, rows);
}

/// tf-cuda-test glm53-mk-policy: per rows 1/2/4/8, every launch shape's step time (graph replay; eager in brackets),
/// the policy's choice on this GPU (TF_GLM53_MK_SHAPE / TF_GLM53_MK_CALIBRATE, the device table, the default),
/// whether that choice is the measured best, and what a forced calibration picks. Every shape, the policy's run and
/// every calibration window are checked bit for bit against the normal launches.
pub fn runPolicy(gpu: anytype) !void {
    var l = try load(gpu);
    defer l.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const env = try mk_defer.policy.Policy.fromEnv();
    const device = blk: {
        const x = try mk_defer.Executor.createWith(gpu.gpa, gpu.d, .{});
        defer x.destroy();
        break :blk x.gpu.device;
    };
    var note: []const u8 = "none (default shape)";
    for (env.table) |e| if (e.sms == device.sms and e.cc == device.cc) {
        note = e.note;
    };
    std.debug.print("RESULT glm53-mk-policy device {d} SMs, sm_{d}; table entry: {s}; TF_GLM53_MK_SHAPE={s} TF_GLM53_MK_CALIBRATE={s} ({d} windows per shape)\n", .{
        device.sms, device.cc, note, if (env.override) |o| o.name() else "(unset)", @tagName(env.calibrate), env.windows,
    });
    var summary: [4]Verdict = undefined;
    inline for (.{ 1, 2, 4, 8 }, 0..) |rows, i| summary[i] = try policyCase(gpu, &l.k, stream, rows, env, device);
    std.debug.print("RESULT glm53-mk-policy summary, graph replay us: rows | plain queue queue-occ chain-block chain-sm | choice (source) | measured best | calibration pick | choice is best\n", .{});
    for (summary, [_]usize{ 1, 2, 4, 8 }) |v, rows| {
        std.debug.print("RESULT glm53-mk-policy summary rows={d} | {d:>8.1} {d:>8.1} {d:>8.1} {d:>8.1} {d:>8.1} | {s} ({s}) | {s} | {s} | {s} (+{d:.1}%)\n", .{
            rows,            v.graph[0],         v.graph[1],    v.graph[2],          v.graph[3],                              v.graph[4],
            v.choice.name(), @tagName(v.source), v.best.name(), v.calibrated.name(), if (v.choice == v.best) "yes" else "no", v.gap_pct,
        });
    }
    std.debug.print("PASS glm53-mk-policy: every launch shape, the policy's choice and every calibration window bit-identical to the normal launches, rows 1/2/4/8\n", .{});
}

const Verdict = struct {
    graph: [mk_defer.policy.shape_count]f32,
    choice: mk_defer.LaunchShape,
    source: mk_defer.policy.Source,
    best: mk_defer.LaunchShape,
    calibrated: mk_defer.LaunchShape,
    /// The choice's time over the best's, in percent (0 when the choice is the best).
    gap_pct: f32,
};

fn policyCase(gpu: anytype, k: *const Kernels, stream: cuda.Stream, rows: usize, env: mk_defer.policy.Policy, device: mk_defer.policy.Device) !Verdict {
    const pol = mk_defer.policy;
    var in = try inputs(gpu, rows, 0x6d6b38 + rows);
    defer in.deinit(gpu.gpa);
    var want = try set(gpu, &in);
    defer want.deinit();
    try step(k, &in, &want, stream);
    try stream.synchronize();
    const reps = 50;
    var scratch = try set(gpu, &in);
    defer scratch.deinit();
    const normal_us = try timed(gpu, stream, stepCtx, Ctx{ .k = k, .in = &in, .s = &scratch }, reps);
    var t: [pol.shape_count][2]f32 = undefined;
    for (pol.all, 0..) |sh, i| t[i] = try policyTimes(gpu, k, &in, &want, stream, .{ .override = sh }, reps, null);
    var best: usize = 0;
    for (t, 0..) |v, i| if (v[1] < t[best][1]) {
        best = i;
    };
    // The policy as the engine runs it (environment, table, default), then a forced calibration.
    var chosen: Picked = undefined;
    const chosen_us = try policyTimes(gpu, k, &in, &want, stream, env, reps, &chosen);
    var cal: Picked = undefined;
    const cal_us = try policyTimes(gpu, k, &in, &want, stream, .{ .calibrate = .force, .windows = env.windows }, reps, &cal);
    const table = pol.lookup(env.table, device, @intCast(rows));
    for (pol.all, t, 0..) |sh, v, i| std.debug.print("RESULT glm53-mk-policy rows={d} shape {s:<11} graph {d:>8.1} us (eager {d:>8.1} us){s}{s}\n", .{
        rows, sh.name(), v[1], v[0], if (i == best) "  <- measured best" else "", if (sh == chosen.shape) "  <- policy choice" else "",
    });
    const c = @backingInt(chosen.shape);
    const gap = (t[c][1] / t[best][1] - 1) * 100;
    std.debug.print("RESULT glm53-mk-policy rows={d} normal launches (eager) {d:.1} us; table entry {s}; policy choice {s} ({s}) graph {d:.1} us; measured best {s} {d:.1} us; choice is best: {s} (+{d:.1}%); calibration picks {s} after {d} eager windows (graph {d:.1} us)\n", .{
        rows,                           normal_us, if (table) |x| x.name() else "none", chosen.shape.name(), @tagName(chosen.source), chosen_us[1], pol.all[best].name(), t[best][1],
        if (c == best) "yes" else "no", gap,       cal.shape.name(),                    cal.eager_windows,   cal_us[1],
    });
    var graph: [pol.shape_count]f32 = undefined;
    for (&graph, t) |*g, v| g.* = v[1];
    return .{ .graph = graph, .choice = chosen.shape, .source = chosen.source, .best = pol.all[best], .calibrated = cal.shape, .gap_pct = gap };
}
