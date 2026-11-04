// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! Two ranks' RoCE all-gather (Group.openRoce) against NCCL's byte for byte, graph replays, time a call (NCCL_COMM_ID).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const roce = cuda.roce;

const cap = 512 << 10;
/// Shards a rank: tiny, a few decode rows, the cap, and past it or not a multiple of 16 (those stay on NCCL).
const sizes = [_]usize{ 16, 48, 4096, 8192, 32768, 65536, 262144, cap, cap + 16, 100 };
const seeds = [_]u64{ 1, 2, 3 };
const graph_gathers = 4;
const graph_replays = 64;
const reps = 2000;

/// Rank `rank`'s shard for a case: bytes that differ by rank, size and seed.
fn shard(out: []u8, seed: u64, rank: usize) void {
    var rng = std.Random.DefaultPrng.init(seed *% 1000003 +% rank *% 7919 +% out.len);
    rng.random().bytes(out);
}

pub fn run(gpu: Gpu, rank: usize) !void {
    const a = gpu.gpa;
    try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "roce-gather: NCCL_COMM_ID names rank 0's address", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.tp_group.Group.open(rank, 2);
    defer g.close();
    try g.openRoce(a, gpu.io, gpu.d, s, cap, false);
    try check.expect(g.fast != null, "roce-gather: the RoCE path did not come up on both ranks (see the warning above)", .{});

    const most = sizes[std.mem.indexOfMax(usize, &sizes)];
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, most);
    defer mine.free();
    var by_group = try cuda.DeviceBuffer.alloc(gpu.d, 2 * most);
    defer by_group.free();
    var by_nccl = try cuda.DeviceBuffer.alloc(gpu.d, 2 * most);
    defer by_nccl.free();
    const host = try a.alloc(u8, most);
    defer a.free(host);
    const peer = try a.alloc(u8, most);
    defer a.free(peer);
    const got = try a.alloc(u8, 2 * most);
    defer a.free(got);
    const want = try a.alloc(u8, 2 * most);
    defer a.free(want);

    var by_roce: usize = 0;
    for (sizes) |n| for (seeds) |seed| {
        shard(host[0..n], seed, rank);
        shard(peer[0..n], seed, 1 - rank);
        try mine.upload(0, host[0..n]);
        try g.allGather(s, mine.ptr, by_group.ptr, n);
        try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, by_nccl.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
        try s.synchronize();
        try g.check();
        @memcpy(want[rank * n ..][0..n], host[0..n]);
        @memcpy(want[(1 - rank) * n ..][0..n], peer[0..n]);
        try by_group.download(0, got[0 .. 2 * n]);
        try check.sameBytes("group all-gather against both ranks' shards", got[0 .. 2 * n], want[0 .. 2 * n]);
        try by_nccl.download(0, got[0 .. 2 * n]);
        try check.sameBytes("NCCL all-gather against both ranks' shards", got[0 .. 2 * n], want[0 .. 2 * n]);
        if (g.fast.?.fits(n)) by_roce += 1;
    };
    check.pass("roce-gather rank {d}: shards of {any} bytes (seeds 1-3) equal NCCL's all-gather; {d} of {d} cases by RoCE", .{ rank, sizes, by_roce, sizes.len * seeds.len });

    var s2 = try cuda.Stream.init(gpu.d, true);
    defer s2.deinit();
    var joined = try cuda.Event.init(gpu.d, false);
    defer joined.deinit();
    const before = g.routeCounts();
    var fitted: u64 = 0;
    for (sizes) |n| {
        if (!g.fast.?.fits(n)) continue;
        fitted += 1;
        shard(host[0..n], 991, rank);
        shard(peer[0..n], 991, 1 - rank);
        try mine.upload(0, host[0..n]);
        try g.allGather(s, mine.ptr, by_group.ptr, n);
        try g.allGather(s2, mine.ptr, by_nccl.ptr, n);
        try joined.record(s2);
        try s.wait(joined);
        try s.synchronize();
        try g.check();
        @memcpy(want[rank * n ..][0..n], host[0..n]);
        @memcpy(want[(1 - rank) * n ..][0..n], peer[0..n]);
        try by_group.download(0, got[0 .. 2 * n]);
        try check.sameBytes("main-stream RoCE gather", got[0 .. 2 * n], want[0 .. 2 * n]);
        try by_nccl.download(0, got[0 .. 2 * n]);
        try check.sameBytes("second-stream NCCL gather", got[0 .. 2 * n], want[0 .. 2 * n]);
    }
    const after = g.routeCounts();
    try check.expect(after[0] == before[0] + fitted and after[2] == before[2] + fitted, "roce-gather: expected one RoCE and one other-stream NCCL route per fitting size", .{});
    check.pass("roce-gather rank {d}: main-stream RoCE and second-stream NCCL agree", .{rank});

    try replays(gpu, &g, s, rank);
    try timing(gpu, &g, s, rank);
}

/// Prepared Spark comparison/equality target for the GLM hidden-width collectives (not run on Mac).
pub fn runAllReduce(gpu: Gpu, rank: usize) !void {
    try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "roce-allreduce: NCCL_COMM_ID names rank 0's address", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.tp_group.Group.open(rank, 2);
    defer g.close();
    try g.openRoce(gpu.gpa, gpu.io, gpu.d, s, cap, false);
    try check.expect(g.fast != null, "roce-allreduce: RoCE did not come up on both ranks", .{});
    const f = g.fast.?;
    for (1..9) |rows| for (0..2) |dtype| for (0..4) |input_case| {
        try equalityCase(gpu, &g, f, s, rank, rows, dtype, input_case);
    };
    try allReduceReplay(gpu, &g, f, s, rank);
    try benchmarkAllReduce(gpu, &g, f, s, rank);
    try poisonGate(gpu, &g, f, s, rank);
}

fn poisonGate(gpu: Gpu, g: *cuda.tp_group.Group, f: *roce.Gather, s: cuda.Stream, rank: usize) !void {
    const count = 16;
    var input = try cuda.DeviceBuffer.alloc(gpu.d, count * 4);
    defer input.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, count * 4);
    defer output.free();
    var token = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer token.free();
    var status = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer status.free();
    const zeros: [64]u8 = @splat(0);
    try input.upload(0, &zeros);
    try token.upload(0, &.{ 42, 0, 0, 0 });
    try status.upload(0, &.{ 0, 0, 0, 0 });
    const started = std.Io.Clock.awake.now(gpu.io);
    // One-sided failure is the serving case: the peer can enter another op before
    // the sticky poison reaches its checked synchronization.
    if (rank == 0) f.stopProxyForTest();
    try g.allReduce(s, input.ptr, output.ptr, count, .f32, .sum);
    try g.allReduce(s, input.ptr, output.ptr, count, .f32, .sum);
    try g.gateResult(s, token.ptr, status.ptr);
    const sync_result = g.syncChecked(s);
    if (sync_result) |_| {
        return error.ExpectedPoisonFailure;
    } else |err| {
        try check.expect(err == error.ProxyFailed, "poisoned step returned unexpected error {s}", .{@errorName(err)});
    }
    var got_token: [4]u8 = undefined;
    var got_status: [4]u8 = undefined;
    var got_output: [4]u8 = undefined;
    try token.download(0, &got_token);
    try status.download(0, &got_status);
    try output.download(0, &got_output);
    try check.expect(std.mem.readInt(u32, &got_token, .little) == 0xffff_ffff, "poison gate rank {d} allowed token {x}", .{ rank, std.mem.readInt(u32, &got_token, .little) });
    try check.expect(std.mem.readInt(u32, &got_status, .little) == 1, "poison gate rank {d} status was not failed", .{rank});
    try check.expect(std.mem.readInt(u32, &got_output, .little) == 0xffff_ffff, "poisoned fp32 reduction rank {d} was not the 0xffffffff canary", .{rank});
    const elapsed_ms = started.durationTo(std.Io.Clock.awake.now(gpu.io)).toMilliseconds();
    try check.expect(elapsed_ms < 10_000, "rank-0-only proxy failure took {d} ms instead of the configured short deadline", .{elapsed_ms});
    check.pass("roce-allreduce rank {d}: rank-0-only stopped proxy poisons two collectives in {d} ms, yielding NaN output and invalid token status", .{ rank, elapsed_ms });
}

fn inputBits(index: usize, rank: usize, dtype: usize, input_case: usize, seed: u64) u32 {
    var rng = std.Random.DefaultPrng.init(seed ^ (@as(u64, rank) *% 0x9e3779b97f4a7c15) ^ @as(u64, index));
    const raw = rng.random().int(u32);
    const special = index % 32;
    if (dtype == 1) {
        if (input_case == 0 and special < 4) return switch (special) {
            0 => if (rank == 0) 0x3f80 else 0x3b80,
            1 => if (rank == 0) 0x3f81 else 0x3b80,
            2 => if (rank == 0) 0x3f80 else 0xbf7f,
            else => if (rank == 0) 0x8000 else 0,
        };
        if (input_case == 3) return switch (special) {
            0 => if (rank == 0) 0x7fc1 else 0x7fe1,
            1 => if (rank == 0) 0x7f80 else 0xff80,
            2 => if (rank == 0) 0xff80 else 0x7f80,
            3 => if (rank == 0) 0x8000 else 0,
            else => @truncate((raw & 0x007f) | 0x3f00),
        };
        const bits: u16 = switch (input_case) {
            0 => @truncate((raw & 0x007f) | (0x3f00 +% (raw & 0xff))),
            1 => @truncate((raw & 0x007f) | 0x0001),
            2 => @truncate((raw & 0x007f) | (if (special % 2 == 0) @as(u32, 0x7e00) else 0x0100)),
            else => unreachable,
        };
        return @as(u32, bits);
    }
    if (input_case == 0 and special < 4) return switch (special) {
        0 => if (rank == 0) 0x3f800000 else 0x33800000,
        1 => if (rank == 0) 0x3f800001 else 0x33800000,
        2 => if (rank == 0) 0x3f800000 else 0xbf7fffff,
        else => if (rank == 0) 0x80000000 else 0,
    };
    if (input_case == 3) return switch (special) {
        0 => if (rank == 0) 0x7fc00001 else 0x7fc12345,
        1 => if (rank == 0) 0x7f800000 else 0xff800000,
        2 => if (rank == 0) 0xff800000 else 0x7f800000,
        3 => if (rank == 0) 0x80000000 else 0,
        else => (raw & 0x007fffff) | 0x3f000000,
    };
    return switch (input_case) {
        0 => (raw & 0x807fffff) | (0x3f000000 +% (raw & 0x00ff0000)),
        1 => (raw & 0x807fffff) | 0x00000001,
        2 => if (special % 2 == 0) 0x7f7fffff else 0x00800001,
        else => unreachable,
    };
}

fn nanOrBitsEqual(dtype: usize, got: []const u8, want: []const u8) !void {
    if (dtype == 0) {
        for (0..got.len / 4) |i| {
            const a = std.mem.readInt(u32, got[i * 4 ..][0..4], .little);
            const b = std.mem.readInt(u32, want[i * 4 ..][0..4], .little);
            const an = (a & 0x7f800000) == 0x7f800000 and (a & 0x007fffff) != 0;
            const bn = (b & 0x7f800000) == 0x7f800000 and (b & 0x007fffff) != 0;
            if (an or bn) try check.expect(an and bn, "NaN handling differs at element {d}", .{i}) else try check.expect(a == b, "fp32 bits differ at element {d}: {x} != {x}", .{ i, a, b });
        }
    } else {
        for (0..got.len / 2) |i| {
            const a = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
            const b = std.mem.readInt(u16, want[i * 2 ..][0..2], .little);
            const an = (a & 0x7f80) == 0x7f80 and (a & 0x007f) != 0;
            const bn = (b & 0x7f80) == 0x7f80 and (b & 0x007f) != 0;
            if (an or bn) try check.expect(an and bn, "bf16 NaN handling differs at element {d}", .{i}) else try check.expect(a == b, "bf16 bits differ at element {d}: {x} != {x}", .{ i, a, b });
        }
    }
}

fn replayValue(index: usize, sender_rank: usize, replay: usize) f32 {
    const shift: i32 = @intCast((index + replay * 17 + sender_rank * 31) % 1009);
    return (@as(f32, @floatFromInt(shift - 504)) + 0.37109375) / 1024.0;
}

fn equalityCase(gpu: Gpu, g: *cuda.tp_group.Group, f: *roce.Gather, s: cuda.Stream, rank: usize, rows: usize, dtype: usize, input_case: usize) !void {
    const count = rows * 6144;
    const width: usize = if (dtype == 0) 4 else 2;
    const bytes = count * width;
    var input = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer input.free();
    var fast_out = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer fast_out.free();
    var nccl_out = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer nccl_out.free();
    const host = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(host);
    const fast = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(fast);
    const nccl = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(nccl);
    for (0..count) |i| {
        const bits = inputBits(i, rank, dtype, input_case, 0x524f434500000000 +% rows *% 17 +% input_case);
        if (dtype == 0) {
            std.mem.writeInt(u32, @ptrCast(host.ptr + i * width), bits, .little);
        } else {
            std.mem.writeInt(u16, @ptrCast(host.ptr + i * width), @truncate(bits), .little);
        }
    }
    try input.upload(0, host);
    const kind: u32 = @intCast(dtype);
    const nccl_dtype: cuda.nccl.DataType = if (dtype == 0) .f32 else .bf16;
    try f.allReduce(s.handle, input.ptr, fast_out.ptr, count, kind);
    try g.lib.check(g.lib.api.ncclAllReduce(input.ptr, nccl_out.ptr, count, nccl_dtype, .sum, g.comm, s.handle), "ncclAllReduce(equality)");
    try g.syncChecked(s);
    try fast_out.download(0, fast);
    try nccl_out.download(0, nccl);
    try nanOrBitsEqual(dtype, fast, nccl);
    // Compare the rank-local RoCE outputs directly; NaN payloads are canonicalized by the comparison rule.
    const both = try gpu.gpa.alloc(u8, 2 * bytes);
    defer gpu.gpa.free(both);
    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, 2 * bytes);
    defer gathered.free();
    try g.lib.check(g.lib.api.ncclAllGather(fast_out.ptr, gathered.ptr, bytes, .u8, g.comm, s.handle), "ncclAllGather(RoCE output copies)");
    try g.syncChecked(s);
    try gathered.download(0, both);
    try nanOrBitsEqual(dtype, both[0..bytes], both[bytes..]);
    check.pass("roce-allreduce rank {d}: R={d} dtype={s} case={d} matches NCCL (NaN == NaN by bits)", .{ rank, rows, if (dtype == 0) "fp32" else "bf16", input_case });
}

fn allReduceReplay(gpu: Gpu, g: *cuda.tp_group.Group, f: *roce.Gather, s: cuda.Stream, rank: usize) !void {
    const count = 6144;
    const bytes = count * @sizeOf(f32);
    const ops = 156;
    var inputs = try cuda.DeviceBuffer.alloc(gpu.d, ops * bytes);
    defer inputs.free();
    var outputs = try cuda.DeviceBuffer.alloc(gpu.d, ops * bytes);
    defer outputs.free();
    const mine = try gpu.gpa.alloc(u8, ops * bytes);
    defer gpu.gpa.free(mine);
    const got = try gpu.gpa.alloc(u8, ops * bytes);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(u8, ops * bytes);
    defer gpu.gpa.free(want);
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..ops) |op| try f.allReduce(s.handle, inputs.ptr + op * bytes, outputs.ptr + op * bytes, count, 0);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    for (0..32) |replay| {
        for (0..ops) |op| for (0..count) |i| {
            const idx = op * count + i;
            std.mem.writeInt(u32, mine[idx * 4 ..][0..4], @bitCast(replayValue(idx, rank, replay)), .little);
            const rank0 = replayValue(idx, 0, replay);
            const rank1 = replayValue(idx, 1, replay);
            std.mem.writeInt(u32, want[idx * 4 ..][0..4], @bitCast(rank0 + rank1), .little);
        };
        try inputs.upload(0, mine);
        try exec.launchOn(s);
        try g.syncChecked(s);
        try outputs.download(0, got);
        try check.sameBytes("replayed 156-op RoCE graph with fresh per-op inputs", got, want);
    }
    // The production in-place form must retain the same semantics.
    var inplace = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer inplace.free();
    var nccl_inplace = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer nccl_inplace.free();
    const inplace_got = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(inplace_got);
    const inplace_want = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(inplace_want);
    try inplace.upload(0, mine[0..bytes]);
    try nccl_inplace.upload(0, mine[0..bytes]);
    try f.allReduce(s.handle, inplace.ptr, inplace.ptr, count, 0);
    try g.lib.check(g.lib.api.ncclAllReduce(nccl_inplace.ptr, nccl_inplace.ptr, count, .f32, .sum, g.comm, s.handle), "ncclAllReduce(in-place reference)");
    try g.syncChecked(s);
    try inplace.download(0, inplace_got);
    try nccl_inplace.download(0, inplace_want);
    try check.sameBytes("in-place RoCE all-reduce", inplace_got, inplace_want);
    check.pass("roce-allreduce rank {d}: captured {d} distinct reductions and replayed 32 changing inputs; in-place passed", .{ rank, ops });
}

fn benchmarkAllReduce(gpu: Gpu, g: *cuda.tp_group.Group, f: *roce.Gather, s: cuda.Stream, rank: usize) !void {
    const ops = 156;
    const graph_samples = 1000;
    const eager_samples = 100;
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    for (1..9) |rows| for (0..2) |dtype| {
        const count = rows * 6144;
        const width: usize = if (dtype == 0) 4 else 2;
        const bytes = count * width;
        var data = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
        defer data.free();
        const kind: u32 = @intCast(dtype);
        const ndt: cuda.nccl.DataType = if (dtype == 0) .f32 else .bf16;
        for (0..2) |backend| {
            try cuda.graph.beginCapture(s, .thread_local);
            for (0..ops) |_| {
                if (backend == 0) {
                    try f.allReduce(s.handle, data.ptr, data.ptr, count, kind);
                } else {
                    try g.lib.check(g.lib.api.ncclAllReduce(data.ptr, data.ptr, count, ndt, .sum, g.comm, s.handle), "ncclAllReduce(capture)");
                }
            }
            var graph = try cuda.graph.endCapture(s);
            defer graph.deinit();
            var exec = try graph.instantiate();
            defer exec.deinit();
            for (0..50) |_| try exec.launchOn(s);
            try g.syncChecked(s);
            const samples = try gpu.gpa.alloc(f64, graph_samples);
            defer gpu.gpa.free(samples);
            for (samples) |*sample| {
                try start.record(s);
                try exec.launchOn(s);
                try stop.record(s);
                try stop.synchronize();
                sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0 / ops;
            }
            const p50_graph = check.median(samples);
            const p99_graph = samples[@as(usize, @intFromFloat(@ceil(0.99 * @as(f64, graph_samples)))) - 1];
            const eager = try gpu.gpa.alloc(f64, eager_samples);
            defer gpu.gpa.free(eager);
            for (eager) |*sample| {
                const before = check.now(gpu.io);
                if (backend == 0) try f.allReduce(s.handle, data.ptr, data.ptr, count, kind) else try g.lib.check(g.lib.api.ncclAllReduce(data.ptr, data.ptr, count, ndt, .sum, g.comm, s.handle), "ncclAllReduce(eager)");
                try g.syncChecked(s);
                sample.* = @as(f64, @floatFromInt(check.now(gpu.io) - before)) / 1000.0;
            }
            const p50_eager = check.median(eager);
            const p99_eager = eager[@as(usize, @intFromFloat(@ceil(0.99 * @as(f64, eager_samples)))) - 1];
            check.pass("roce-allreduce rank {d}: R={d} dtype={s} backend={s} graph156 p50={d:.2} p99={d:.2} us/op; eager p50={d:.2} p99={d:.2} us", .{ rank, rows, if (dtype == 0) "fp32" else "bf16", if (backend == 0) "roce" else "nccl", p50_graph, p99_graph, p50_eager, p99_eager });
        }
        var gather_out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * bytes);
        defer gather_out.free();
        for (0..2) |backend| {
            try cuda.graph.beginCapture(s, .thread_local);
            for (0..ops) |_| {
                if (backend == 0) {
                    try f.gather(s.handle, data.ptr, gather_out.ptr, bytes);
                } else {
                    try g.lib.check(g.lib.api.ncclAllGather(data.ptr, gather_out.ptr, bytes, .u8, g.comm, s.handle), "ncclAllGather(capture)");
                }
            }
            var gather_graph = try cuda.graph.endCapture(s);
            defer gather_graph.deinit();
            var gather_exec = try gather_graph.instantiate();
            defer gather_exec.deinit();
            for (0..50) |_| try gather_exec.launchOn(s);
            try g.syncChecked(s);
            const gather_samples = try gpu.gpa.alloc(f64, graph_samples);
            defer gpu.gpa.free(gather_samples);
            for (gather_samples) |*sample| {
                try start.record(s);
                try gather_exec.launchOn(s);
                try stop.record(s);
                try stop.synchronize();
                sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0 / ops;
            }
            const gather_p50 = check.median(gather_samples);
            const gather_p99 = gather_samples[@as(usize, @intFromFloat(@ceil(0.99 * @as(f64, graph_samples)))) - 1];
            check.pass("roce-allreduce rank {d}: R={d} dtype={s} backend={s} gather-only graph156 p50={d:.2} p99={d:.2} us/op", .{ rank, rows, if (dtype == 0) "fp32" else "bf16", if (backend == 0) "roce" else "nccl", gather_p50, gather_p99 });
        }
    };
}

/// A graph of `graph_gathers` RoCE gathers replayed with new shards each time: each replay's rows are its own.
fn replays(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize) !void {
    const a = gpu.gpa;
    const n = 8192;
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * n);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * 2 * n);
    defer all.free();
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..graph_gathers) |i| try g.allGather(s, mine.ptr + i * n, all.ptr + i * 2 * n, n);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    const host = try a.alloc(u8, graph_gathers * n);
    defer a.free(host);
    const peer = try a.alloc(u8, graph_gathers * n);
    defer a.free(peer);
    const got = try a.alloc(u8, graph_gathers * 2 * n);
    defer a.free(got);
    const want = try a.alloc(u8, graph_gathers * 2 * n);
    defer a.free(want);
    for (0..graph_replays) |r| {
        for (0..graph_gathers) |i| {
            shard(host[i * n ..][0..n], 100 + r * graph_gathers + i, rank);
            shard(peer[i * n ..][0..n], 100 + r * graph_gathers + i, 1 - rank);
            @memcpy(want[i * 2 * n + rank * n ..][0..n], host[i * n ..][0..n]);
            @memcpy(want[i * 2 * n + (1 - rank) * n ..][0..n], peer[i * n ..][0..n]);
        }
        try mine.upload(0, host);
        try exec.launchOn(s);
        try s.synchronize();
        try g.check();
        try all.download(0, got);
        try check.sameBytes("a replayed graph's gathers", got, want);
    }
    check.pass("roce-gather rank {d}: a captured graph of {d} gathers of {d} bytes replayed {d} times, each replay's rows its own", .{ rank, graph_gathers, n, graph_replays });
}

/// Median microseconds a call of `reps` back-to-back gathers, RoCE against NCCL, at decode-sized shards.
fn timing(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize) !void {
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, 65536);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 65536);
    defer all.free();
    for ([_]usize{ 4096, 8192, 32768, 65536 }) |n| {
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            for (0..2) |round| { // the first round warms
                try t0.record(s);
                for (0..reps) |_| {
                    if (which == 0) try g.allGather(s, mine.ptr, all.ptr, n) else try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, all.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
                }
                try t1.record(s);
                try t1.synchronize();
                if (round == 1) us[which] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) * 1000.0 / reps;
            }
        }
        try g.check();
        check.pass("roce-gather rank {d}: {d} bytes a rank, {d} calls: RoCE {d:.1} us a call, NCCL {d:.1} us", .{ rank, n, reps, us[0], us[1] });
    }
}
