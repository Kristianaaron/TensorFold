// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 decode through keyed CUDA graphs vs eager on one GPU: a synthetic GLM-shaped rank (real kernel widths:
//! hidden 6144, 32 local heads, MLA 512 + 64, indexer 32 x 128; two dense layers, the first with a full indexer; a
//! 1,024-row vocabulary shard) over the real Forward and Engine, rank 0 of TP2 with a loopback collective (the peer's
//! rows are copies of this rank's, by device copies a stream capture records). Prefill, then decode steps whose
//! positions move inside one graph key and across a key: graphs on must give the eager run's tokens, logits and head
//! rows bit for bit at every step. The collective is the one part not exercised: NCCL's own first-use behaviour in a
//! capture is what engine.decode_graphs (warm = 1) and tf-glm53-bringup's peer warm-up keep out of the graphs.
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("forward");
const engine = @import("engine");
const state = @import("state");
const weights = @import("weights");
const ablate = @import("ablate");

const hidden = forward.hidden;
const vocab_local = 1024;
const context = 4096;

/// The peer is this rank: every all-gather slot and every received block is a device copy of this rank's own bytes.
const Loopback = struct {
    d: *const cuda.Driver,

    fn collective(l: *Loopback) forward.Collective {
        return .{ .ctx = l, .all_gather_fn = gather, .send_recv_fn = sendRecv };
    }

    fn copy(l: *const Loopback, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0 or dst == src) return;
        try l.d.check(l.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "loopback copy");
    }

    fn gather(ctx: *anyopaque, s: cuda.Stream, mine: u64, all: u64, bytes: usize, _: forward.Phase) anyerror!void {
        const l: *Loopback = @ptrCast(@alignCast(ctx));
        for (0..forward.world) |q| try l.copy(s, all + q * bytes, mine, bytes);
    }

    fn sendRecv(ctx: *anyopaque, s: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, _: usize, _: forward.Phase) anyerror!void {
        const l: *Loopback = @ptrCast(@alignCast(ctx));
        try l.copy(s, recv, send, @min(send_bytes, recv_bytes));
    }
};

/// Device tensors of the synthetic rank, filled once from a fixed seed (finite values only).
const Synth = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    rng: std.Random.DefaultPrng = .init(0x61a5_3e11),

    fn deinit(s: *Synth) void {
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(s.gpa);
    }

    fn put(s: *Synth, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(s.d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        try s.bufs.append(s.gpa, b);
        return b.ptr;
    }

    /// bf16 values uniform in [center - spread, center + spread].
    fn bf16s(s: *Synth, n: usize, center: f32, spread: f32) !u64 {
        const host = try s.gpa.alloc(u16, n);
        defer s.gpa.free(host);
        const r = s.rng.random();
        for (host) |*v| v.* = weights.toBf16(center + (r.float(f32) * 2 - 1) * spread);
        return s.put(std.mem.sliceAsBytes(host));
    }

    fn f32s(s: *Synth, n: usize, value: f32) !u64 {
        const host = try s.gpa.alloc(f32, n);
        defer s.gpa.free(host);
        @memset(host, value);
        return s.put(std.mem.sliceAsBytes(host));
    }

    fn f16s(s: *Synth, n: usize) !u64 {
        const host = try s.gpa.alloc(f16, n);
        defer s.gpa.free(host);
        const random = s.rng.random();
        for (host) |*v| v.* = @floatCast(random.float(f32) * 2 - 1);
        return s.put(std.mem.sliceAsBytes(host));
    }

    fn routedProjection(s: *Synth, k: usize, n: usize) !struct { ptrs: u64, k2: u64, suh: u64, svh: u64 } {
        var buffers: [8]u64 = undefined;
        var widths: [8]i32 = undefined;
        for (0..8) |i| {
            const k2: usize = if (i % 2 == 0) 2 else 4;
            widths[i] = @intCast(k2);
            const bytes = (k / 16) * (n / 16) * 16 * k2;
            const host = try s.gpa.alloc(u8, bytes);
            defer s.gpa.free(host);
            s.rng.random().bytes(host);
            buffers[i] = try s.put(host);
        }
        const ptr_table = try s.gpa.alloc(i64, 256);
        defer s.gpa.free(ptr_table);
        const width_table = try s.gpa.alloc(i32, 256);
        defer s.gpa.free(width_table);
        for (0..256) |e| {
            ptr_table[e] = @intCast(buffers[e % 8]);
            width_table[e] = widths[e % 8];
        }
        return .{
            .ptrs = try s.put(std.mem.sliceAsBytes(ptr_table)),
            .k2 = try s.put(std.mem.sliceAsBytes(width_table)),
            .suh = try s.f16s(256 * k),
            .svh = try s.f16s(256 * n),
        };
    }

    fn routed(s: *Synth) !@FieldType(weights.Moe, "routed") {
        const gate = try s.routedProjection(hidden, 1024);
        const up = try s.routedProjection(hidden, 1024);
        const down = try s.routedProjection(1024, hidden);
        return .{
            .gate = gate.ptrs, .up = up.ptrs, .down = down.ptrs,
            .gate_k2 = gate.k2, .up_k2 = up.k2, .down_k2 = down.k2,
            .suh_g = gate.suh, .suh_u = up.suh, .svh_g = gate.svh, .svh_u = up.svh,
            .suh_d = down.suh, .svh_d = down.svh,
            .codebook = .mcg_e4m3, .uniform_k2 = 0, .widths_ok = true, .dims = hidden, .width = 1024, .experts = 256,
        };
    }

    /// e4m3 codes of magnitude 1/8 .. 15/8 (exponent fields 4..7: never NaN), random signs.
    fn e4m3s(s: *Synth, n: usize) !u64 {
        const host = try s.gpa.alloc(u8, n);
        defer s.gpa.free(host);
        s.rng.random().bytes(host);
        for (host) |*b| b.* = (b.* & 0x87) | ((4 + ((b.* >> 3) & 3)) << 3);
        return s.put(host);
    }

    /// A block-FP8 lane weight [n][k] in qmmf's fp8b layout: npad*k codes, npad*(k/64) fp32 tile scales.
    fn fp8b(s: *Synth, n: usize, k: usize) !cuda.qmmf.Weight {
        const npad = cuda.fp8.padded(n);
        const codes = try s.e4m3s(npad * k);
        const scales = try s.f32s(npad * (k / cuda.fp8.group), 1.0 / 128.0);
        return cuda.fp8.blockWeight(codes, scales, n, k);
    }

    /// A bf16 lane weight [n][k] (npad rows of 128).
    fn bf16w(s: *Synth, n: usize, k: usize, spread: f32) !cuda.qmmf.Weight {
        const npad = (n + 127) / 128 * 128;
        return .{ .mode = .bf16, .codes = try s.bf16s(npad * k, 0, spread), .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
    }

    fn layer(s: *Synth, i: usize, full: bool) !weights.DeviceLayer {
        const heads = state.local_heads;
        var a: weights.Attention = .{
            .q_a = try s.fp8b(state.q_lora, hidden),
            .q_b = try s.fp8b(heads * state.q_head, state.q_lora),
            .kv_a = try s.fp8b(state.kv_a_width, hidden),
            .kv_b = .{ .codes = try s.e4m3s(heads * 448 * 512), .scales = try s.f32s(heads * 448 / 128 * 4, 1.0 / 32.0), .rows = heads * 448, .cols = 512 },
            .o_proj = try s.fp8b(hidden, heads * 256),
            .q_a_norm = try s.bf16s(state.q_lora, 1, 0.1),
            .kv_a_norm = try s.bf16s(512, 1, 0.1),
            .local_heads = heads,
            .head0 = 0,
        };
        if (full) a.indexer = .{
            .wq_b = try s.fp8b(32 * 128, state.q_lora),
            .wk = try s.fp8b(128, hidden),
            .weights_proj = try s.bf16w(32, hidden, 0.02),
            .k_norm = try s.bf16s(128, 1, 0.1),
            .k_bias = try s.bf16s(128, 0, 0.05),
        };
        return .{
            .index = i,
            .in_norm = try s.bf16s(hidden, 1, 0.1),
            .post_norm = try s.bf16s(hidden, 1, 0.1),
            .attn = a,
            .ffn = .{ .dense = .{ .gate_up = try s.fp8b(2 * 6144, hidden), .down = try s.fp8b(hidden, 6144) } },
        };
    }

    fn moeLayer(s: *Synth, i: usize) !weights.DeviceLayer {
        var l = try s.layer(i, i == 0);
        l.ffn = .{ .moe = .{
            .router = try s.bf16s(256 * hidden, 0, 0.02),
            .bias = try s.f32s(256, 0),
            .routed = try s.routed(),
            .shared = .{ .gate_up = try s.fp8b(2 * 1024, hidden), .down = try s.fp8b(hidden, 1024) },
            .lowrank = null,
        } };
        return l;
    }
};

/// Every step's drawn token, the rank-local fp32 logits row and the gathered head rows.
const Trace = struct {
    tokens: std.ArrayList(u32) = .empty,
    logits: std.ArrayList(u8) = .empty,
    heads: std.ArrayList(u8) = .empty,
    kv: std.ArrayList(u8) = .empty,
    captured: u64 = 0,
    replayed: u64 = 0,

    fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        t.tokens.deinit(gpa);
        t.logits.deinit(gpa);
        t.heads.deinit(gpa);
        t.kv.deinit(gpa);
    }
};

const Scenario = struct { name: []const u8, prompt: usize, steps: usize, rank: usize = 0 };

fn record(t: *Trace, gpa: std.mem.Allocator, f: forward.Forward, st: *const state.State, row: []u8, e: *const engine.Engine, tok: u32) !void {
    try f.stream.synchronize();
    try f.stream.d.check(f.stream.d.api.cuMemcpyDtoH_v2(row.ptr, st.ws.logits, row.len), "logits row");
    try t.logits.appendSlice(gpa, row);
    try t.heads.appendSlice(gpa, e.head_host[0 .. e.head_per_rank * forward.world]);
    try t.tokens.append(gpa, tok);
}

/// The first decode step where two runs differ in drawn token, fp32 logits row or gathered head rows.
fn firstDiff(a: *const Trace, b: *const Trace, sc: Scenario, label: []const u8) ?usize {
    const row = vocab_local * 4;
    const head = a.heads.items.len / sc.steps;
    var first: ?usize = null;
    if (a.tokens.items[0] != b.tokens.items[0]) std.debug.print("  {s}: prefill token differs\n", .{label});
    for (0..sc.steps) |i| {
        const same_token = a.tokens.items[i + 1] == b.tokens.items[i + 1];
        const same_logits = std.mem.eql(u8, a.logits.items[i * row ..][0..row], b.logits.items[i * row ..][0..row]);
        const same_head = std.mem.eql(u8, a.heads.items[i * head ..][0..head], b.heads.items[i * head ..][0..head]);
        const same_kv = std.mem.eql(u8, a.kv.items, b.kv.items);
        if (same_token and same_logits and same_head and same_kv) continue;
        if (first == null) first = i;
        std.debug.print("  {s}: decode {d} (pos {d}): token {s}, logits {s}, head rows {s}\n", .{ label, i, sc.prompt + i, if (same_token) "same" else "DIFFERS", if (same_logits) "same" else "DIFFER", if (same_head) "same" else "DIFFER" });
    }
    return first;
}

/// The State after one prefill (KV, rope and index planes, workspace) and its head record: every decode run starts
/// from these bytes, so the comparison is of the decode path alone (the restore keeps the decode comparison independent of the prefill; prefill repeatability
/// and resume == fresh are checked on their own below).
const Snapshot = struct {
    kv: cuda.DeviceBuffer,
    ws: cuda.DeviceBuffer,
    taken: bool = false,
    token: u32 = 0,

    fn init(d: *const cuda.Driver, st: *const state.State) !Snapshot {
        var kv = try cuda.DeviceBuffer.alloc(d, st.kv_arena.len);
        errdefer kv.free();
        return .{ .kv = kv, .ws = try cuda.DeviceBuffer.alloc(d, st.workspace.len) };
    }

    fn deinit(s: *Snapshot) void {
        s.kv.free();
        s.ws.free();
    }

    /// Stream-ordered: the rig's stream is non-blocking, so a NULL-stream copy or memset would race its work.
    fn copy(d: *const cuda.Driver, dst: cuda.DeviceBuffer, src: cuda.DeviceBuffer, s: cuda.Stream) !void {
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst.ptr, src.ptr, src.len, s.handle), "snapshot copy");
        try s.synchronize();
    }
};

fn traceRun(gpa: std.mem.Allocator, f: forward.Forward, st: *state.State, sc: Scenario, graphs: bool, snap: *Snapshot, abl: ?*const ablate.Ablation) !Trace {
    const d = f.stream.d;
    var e: engine.Engine = .{ .gpa = gpa, .f = f, .auto_graphs = graphs };
    defer e.deinit();
    e.abl = abl;
    if (abl != null) try e.setAblate(true);
    var t: Trace = .{};
    errdefer t.deinit(gpa);
    const row = try gpa.alloc(u8, vocab_local * 4);
    defer gpa.free(row);
    var tok: u32 = undefined;
    if (!snap.taken) {
        try st.kv_arena.fill8(0, f.stream.handle);
        try st.workspace.fill8(0, f.stream.handle);
        if (st.stage_plane.len != 0) try st.stage_plane.fill8(0, f.stream.handle);
        const ids = try gpa.alloc(u32, sc.prompt);
        defer gpa.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast(f.w.vocab.lo + (i * 37 + 11) % vocab_local);
        tok = try e.prefill(ids);
        try f.stream.synchronize();
        try Snapshot.copy(d, snap.kv, st.kv_arena, f.stream);
        try Snapshot.copy(d, snap.ws, st.workspace, f.stream);
        snap.token = tok;
        snap.taken = true;
    } else {
        try f.stream.synchronize();
        try Snapshot.copy(d, st.kv_arena, snap.kv, f.stream);
        try Snapshot.copy(d, st.workspace, snap.ws, f.stream);
        tok = snap.token;
        e.pos = sc.prompt;
        e.frontier = sc.prompt;
    }
    try t.tokens.append(gpa, tok);
    for (0..sc.steps) |_| {
        var out: [1]u32 = undefined;
        try e.verify(&.{tok}, &out);
        try record(&t, gpa, f, st, row, &e, out[0]);
        try e.keep(1);
        tok = out[0];
    }
    if (e.graphs) |cache| {
        t.captured = cache.stats.captured;
        t.replayed = cache.stats.replayed;
    }
    try f.stream.synchronize();
    const kv = try gpa.alloc(u8, st.kv_arena.len);
    errdefer gpa.free(kv);
    try st.kv_arena.download(0, kv);
    try t.kv.appendSlice(gpa, kv);
    gpa.free(kv);
    return t;
}

pub const Gpu = struct { d: *const cuda.Driver, ctx: *const cuda.Context, gpa: std.mem.Allocator };

pub fn run(gpu: Gpu) !void {
    const d = gpu.d;
    const gpa = gpu.gpa;
    var lane = try cuda.qmmf.Lane.load(d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var ex = try cuda.exl3_experts.Experts.load(d);
    defer ex.unload();
    var mods: [5]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ cuda.kernels.glm_glue, cuda.kernels.glm53_glue, cuda.kernels.glm53_mla, cuda.kernels.glm53_index, cuda.kernels.glm53_head }) |image| {
        mods[loaded] = try cuda.Module.load(d, image);
        loaded += 1;
    }
    var k = try forward.Kernels.load(.{ .glm_glue = mods[0], .glm53_glue = mods[1], .glm53_mla = mods[2], .glm53_index = mods[3], .glm53_head = mods[4] }, &lane, &ex, 4);
    k.ablate = try mods[1].function("tf_glm53_ablate_6144");

    var synth: Synth = .{ .d = d, .gpa = gpa };
    defer synth.deinit();
    var layers = [_]weights.DeviceLayer{ try synth.moeLayer(0), try synth.layer(1, false) };
    var model: weights.ModelWeights = .{
        .allocator = gpa,
        .rank = 0,
        .world = 2,
        .layers = &layers,
        .embed = try synth.bf16s(vocab_local * hidden, 0, 1),
        .vocab = .{ .lo = 0, .hi = vocab_local },
        .norm = try synth.bf16s(hidden, 1, 0.1),
        .head = try synth.bf16w(vocab_local, hidden, 0.02),
    };
    const g: state.Geometry = .{ .layers = 2, .full_index_layers = 1, .graphs_bytes = 0, .vocab_local = vocab_local, .decode_rows = 16, .head_rows = 16, .lane_clusters = lane.clusters, .decode_side = true, .piece_pipe = true, .expert_rows = 1024 };
    const adm = try state.admitRank(g, context, .{}, 1 << 50, state.reserveFor(1 << 50, false, 0));
    var st = try state.State.init(d, g, context, 0, 2, &adm, 1 << 50);
    defer st.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var loop: Loopback = .{ .d = d };

    var ablation = try makeAblation(gpa, d, 0);
    defer ablation.deinit();

    // One graph key (rows 1, bucket 2048) for every step; then a run whose steps cross 2047 -> 2048 (a new key with
    // sparse rows and the indexer's top-k) while later steps replay it; each as TP rank 0 and rank 1. The graphs-on
    // run goes first, as in a served process: its first decode step is the capture (nothing decode-only ran before).
    const scenarios = [_]Scenario{
        .{ .name = "rank 0, 35-row prompt, 6 decode steps", .prompt = 35, .steps = 6 },
        .{ .name = "rank 1, 35-row prompt, 6 decode steps", .prompt = 35, .steps = 6, .rank = 1 },
        .{ .name = "rank 0, 2040-row prompt, 14 decode steps across 2048", .prompt = 2040, .steps = 14 },
        .{ .name = "rank 1, 2040-row prompt, 14 decode steps across 2048", .prompt = 2040, .steps = 14, .rank = 1 },
    };
    var failed = false;
    try prefillRepeatability(gpa, &k, &model, &st, stream, &loop);
    try windowParity(gpa, &k, &model, &st, stream, &loop);
    var snap = try Snapshot.init(d, &st);
    defer snap.deinit();
    for (scenarios) |sc| {
        model.vocab = .{ .lo = sc.rank * vocab_local, .hi = (sc.rank + 1) * vocab_local };
        st.rank = sc.rank;
        const f: forward.Forward = .{ .gpa = gpa, .k = &k, .w = &model, .st = &st, .stream = stream, .coll = loop.collective(), .rank = sc.rank, .decode_side = false };
        snap.taken = false;
        var graphed = try traceRun(gpa, f, &st, sc, true, &snap, null);
        defer graphed.deinit(gpa);
        var eager = try traceRun(gpa, f, &st, sc, false, &snap, null);
        defer eager.deinit(gpa);
        var again = try traceRun(gpa, f, &st, sc, false, &snap, null);
        defer again.deinit(gpa);
        std.debug.print("RESULT {s}: eager tokens {any}\n", .{ sc.name, eager.tokens.items });
        std.debug.print("RESULT {s}: graph tokens {any} ({d} captured, {d} replayed)\n", .{ sc.name, graphed.tokens.items, graphed.captured, graphed.replayed });
        const repeat = firstDiff(&eager, &again, sc, "eager vs eager");
        const parity = firstDiff(&eager, &graphed, sc, "eager vs graphs");
        if (repeat != null) {
            std.debug.print("FAIL glm53 eager decode is not repeatable: {s} (first differing decode step {?d})\n", .{ sc.name, repeat });
            failed = true;
        }
        if (parity == null and graphed.captured >= 1 and graphed.replayed >= 1) {
            std.debug.print("PASS glm53 graph replay == eager: {s}\n", .{sc.name});
        } else {
            std.debug.print("FAIL glm53 graph replay != eager: {s} (first differing decode step {?d})\n", .{ sc.name, parity });
            failed = true;
        }
        var side_snap = try Snapshot.init(d, &st);
        defer side_snap.deinit();
        var side_f = f;
        side_f.decode_side = true;
        var side_eager = try traceRun(gpa, side_f, &st, sc, false, &side_snap, null);
        defer side_eager.deinit(gpa);
        var side_graph = try traceRun(gpa, side_f, &st, sc, true, &side_snap, null);
        defer side_graph.deinit(gpa);
        const side_eager_diff = firstDiff(&eager, &side_eager, sc, "side on vs off eager");
        const side_graph_diff = firstDiff(&side_eager, &side_graph, sc, "side eager vs replayed graph");
        if (side_eager_diff == null) {
            std.debug.print("PASS glm53 side eager parity: {s}\n", .{sc.name});
        } else {
            std.debug.print("FAIL glm53 side eager parity: {s}\n", .{sc.name});
            failed = true;
        }
        if (side_graph_diff == null and side_graph.captured >= 1 and side_graph.replayed >= 1) {
            std.debug.print("PASS glm53 side graph replay parity: {s}\n", .{sc.name});
        } else {
            std.debug.print("FAIL glm53 side graph replay parity: {s}\n", .{sc.name});
            failed = true;
        }
    }
    // Ablated decode windows participate in graph fingerprints. Zero scale must preserve
    // token, logits, head rows, and KV bytes; nonzero runs must differ and repeat exactly.
    model.vocab = .{ .lo = 0, .hi = vocab_local };
    st.rank = 0;
    const sc: Scenario = .{ .name = "ablation graph parity", .prompt = 35, .steps = 4 }; // warm = 1: step 0 eager, step 1 captures, steps 2-3 replay
    const f: forward.Forward = .{ .gpa = gpa, .k = &k, .w = &model, .st = &st, .stream = stream, .coll = loop.collective(), .rank = 0 };
    var off_snap = try Snapshot.init(d, &st); defer off_snap.deinit();
    var zero_snap = try Snapshot.init(d, &st); defer zero_snap.deinit();
    var off = try traceRun(gpa, f, &st, sc, false, &off_snap, null); defer off.deinit(gpa);
    var zero = try traceRun(gpa, f, &st, sc, true, &zero_snap, &ablation); defer zero.deinit(gpa);
    if (firstDiff(&off, &zero, sc, "ablation zero vs off") != null) {
        std.debug.print("FAIL glm53 ablation zero-scale differs from off\n", .{});
        return error.TestFailed;
    }
    if (zero.captured == 0 or zero.replayed == 0) {
        std.debug.print("FAIL glm53 ablation zero-scale run did not exercise graphs: {d} captured, {d} replayed\n", .{ zero.captured, zero.replayed });
        return error.TestFailed;
    }
    try ablationScale(gpa, &ablation, 0.25);
    var active_snap = try Snapshot.init(d, &st); defer active_snap.deinit();
    var active = try traceRun(gpa, f, &st, sc, true, &active_snap, &ablation); defer active.deinit(gpa);
    var repeat_snap = try Snapshot.init(d, &st); defer repeat_snap.deinit();
    var repeat = try traceRun(gpa, f, &st, sc, true, &repeat_snap, &ablation); defer repeat.deinit(gpa);
    if (firstDiff(&off, &active, sc, "ablation nonzero vs off") == null) {
        std.debug.print("FAIL glm53 ablation with a nonzero scale did not change the output\n", .{});
        return error.TestFailed;
    }
    if (firstDiff(&active, &repeat, sc, "ablation nonzero repeat") != null) {
        std.debug.print("FAIL glm53 ablation nonzero run is not repeatable\n", .{});
        return error.TestFailed;
    }
    std.debug.print("PASS glm53 ablation graph replay: zero-scale parity and repeatable nonzero output\n", .{});
    try expertRowsParity(gpa, d, &k, &synth, stream, &loop, lane.clusters);
    try chunkRowsParity(gpa, d, &k, &synth, stream, &loop, lane.clusters);
    if (failed) return error.TestFailed;
}

fn expertRowsParity(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const forward.Kernels, synth: *Synth, stream: cuda.Stream, loop: *Loopback, clusters: bool) !void {
    const context_rows = 8192;
    var layers = [_]weights.DeviceLayer{ try synth.layer(0, true), try synth.moeLayer(1), try synth.layer(2, false) };
    const model: weights.ModelWeights = .{
        .allocator = gpa, .rank = 0, .world = 2, .layers = &layers,
        .embed = try synth.bf16s(vocab_local * hidden, 0, 1), .vocab = .{ .lo = 0, .hi = vocab_local },
        .norm = try synth.bf16s(hidden, 1, 0.1), .head = try synth.bf16w(vocab_local, hidden, 0.02),
    };
    const prompts = [_]usize{ 600, 2040, 2600, 4500 };
    const row_bytes = vocab_local * 4;
    var failed = false;
    for (prompts) |prompt_len| {
        const ids = try gpa.alloc(u32, prompt_len);
        defer gpa.free(ids);
        for (ids, 0..) |*token, i| token.* = @intCast((i * 37 + 11) % vocab_local);
        var ref_planes: ?[]u8 = null;
        var ref_logits: ?[]u8 = null;
        var ref_token: u32 = 0;
        for ([_]usize{ 128, 512, 1024, 2048 }) |er| {
            const geometry: state.Geometry = .{ .layers = 3, .full_index_layers = 1, .graphs_bytes = 0, .vocab_local = vocab_local, .lane_clusters = clusters, .expert_rows = er };
            const admission = try state.admitRank(geometry, context_rows, .{}, 1 << 50, state.reserveFor(1 << 50, false, 0));
            var st = try state.State.init(d, geometry, context_rows, 0, 2, &admission, 1 << 50);
            defer st.deinit();
            const f: forward.Forward = .{ .gpa = gpa, .k = k, .w = &model, .st = &st, .stream = stream, .coll = loop.collective(), .rank = 0 };
            const result = try freshPrefill(gpa, f, &st, stream, &.{ids});
            defer gpa.free(result.planes);
            const logits = try gpa.alloc(u8, row_bytes);
            defer gpa.free(logits);
            try d.check(d.api.cuMemcpyDtoH_v2(logits.ptr, st.ws.logits, logits.len), "expert_rows parity logits");
            if (er == 128) {
                ref_planes = try gpa.dupe(u8, result.planes);
                ref_logits = try gpa.dupe(u8, logits);
                ref_token = result.token;
                std.debug.print("PASS glm53 expert_rows 128 == 128 (prompt {d})\n", .{prompt_len});
            } else {
                const same = std.mem.eql(u8, ref_planes.?, result.planes) and std.mem.eql(u8, ref_logits.?, logits) and result.token == ref_token;
                std.debug.print("{s} glm53 expert_rows {d} == 128 (prompt {d})\n", .{ if (same) "PASS" else "FAIL", er, prompt_len });
                failed = failed or !same;
            }
            if (prompt_len == 2600 and er == 128) {
                // runRouted reuses picks8 for each <= expert_rows slice; inspect the final slice, which is valid
                // device memory and still exercises the 2600-row prompt's router on its tail rows.
                const batch_rows = @min(prompt_len, er);
                const picks_bytes = batch_rows * state.routed_slots * @sizeOf(u32);
                const host = try gpa.alloc(u32, batch_rows * state.routed_slots);
                defer gpa.free(host);
                try d.check(d.api.cuMemcpyDtoH_v2(host.ptr, st.ws.picks8, picks_bytes), "expert_rows router picks");
                var counts: [state.routed_experts]usize = @splat(0);
                for (host) |expert| if (expert < counts.len) { counts[expert] += 1; };
                var minimum: usize = std.math.maxInt(usize);
                var maximum: usize = 0;
                var min_id: usize = 0;
                var max_id: usize = 0;
                for (counts, 0..) |count, expert| {
                    if (count < minimum) { minimum = count; min_id = expert; }
                    if (count > maximum) { maximum = count; max_id = expert; }
                }
                std.debug.print("ROUTER prompt 2600 members max expert {d}={d} min expert {d}={d}\n", .{ max_id, maximum, min_id, minimum });
            }
        }
        gpa.free(ref_planes.?);
        gpa.free(ref_logits.?);
    }
    var split_reference: ?struct { planes: []u8, token: u32 } = null;
    for ([_]usize{ 128, 1024, 2048 }) |er| {
        const split = try splitPrefillParity(gpa, d, k, &model, stream, loop, clusters, er, context_rows);
        defer gpa.free(split.planes);
        if (er == 128) {
            split_reference = .{ .planes = try gpa.dupe(u8, split.planes), .token = split.token };
        } else {
            const same = std.mem.eql(u8, split_reference.?.planes, split.planes) and split_reference.?.token == split.token;
            std.debug.print("{s} glm53 expert_rows {d} == 128 (prefills 300 + 2300, second starts at 300)\n", .{ if (same) "PASS" else "FAIL", er });
            failed = failed or !same;
        }
    }
    gpa.free(split_reference.?.planes);
    if (failed) return error.ExpertRowsParityFailed;
}

fn splitPrefillParity(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const forward.Kernels, model: *const weights.ModelWeights, stream: cuda.Stream, loop: *Loopback, clusters: bool, expert_rows: usize, context_rows: usize) !struct { planes: []u8, token: u32 } {
    const geometry: state.Geometry = .{ .layers = 3, .full_index_layers = 1, .graphs_bytes = 0, .vocab_local = vocab_local, .lane_clusters = clusters, .expert_rows = expert_rows };
    const admission = try state.admitRank(geometry, context_rows, .{}, 1 << 50, state.reserveFor(1 << 50, false, 0));
    var st = try state.State.init(d, geometry, context_rows, 0, 2, &admission, 1 << 50);
    defer st.deinit();
    try st.kv_arena.fill8(0, stream.handle);
    try st.workspace.fill8(0, stream.handle);
    if (st.stage_plane.len != 0) try st.stage_plane.fill8(0, stream.handle);
    const f: forward.Forward = .{ .gpa = gpa, .k = k, .w = model, .st = &st, .stream = stream, .coll = loop.collective(), .rank = 0 };
    var e: engine.Engine = .{ .gpa = gpa, .f = f, .auto_graphs = false };
    defer e.deinit();
    const ids = try gpa.alloc(u32, 2600);
    defer gpa.free(ids);
    for (ids, 0..) |*token, i| token.* = @intCast((i * 37 + 11) % vocab_local);
    _ = try e.prefill(ids[0..300]);
    const token = try e.prefill(ids[300..]);
    try stream.synchronize();
    const planes = try gpa.alloc(u8, st.kv_arena.len);
    try d.check(d.api.cuMemcpyDtoH_v2(planes.ptr, st.kv_arena.ptr, planes.len), "expert_rows split prefill planes");
    return .{ .planes = planes, .token = token };
}

fn chunkRowsParity(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const forward.Kernels, synth: *Synth, stream: cuda.Stream, loop: *Loopback, clusters: bool) !void {
    const context_rows = 8192;
    var layers = [_]weights.DeviceLayer{ try synth.layer(0, true), try synth.layer(1, false) };
    const model: weights.ModelWeights = .{
        .allocator = gpa, .rank = 0, .world = 2, .layers = &layers,
        .embed = try synth.bf16s(vocab_local * hidden, 0, 1), .vocab = .{ .lo = 0, .hi = vocab_local },
        .norm = try synth.bf16s(hidden, 1, 0.1), .head = try synth.bf16w(vocab_local, hidden, 0.02),
    };
    const prompts = [_]usize{ 2600, 4500 };
    var failed = false;
    for (prompts) |prompt_len| {
        const ids = try gpa.alloc(u32, prompt_len);
        defer gpa.free(ids);
        for (ids, 0..) |*token, i| token.* = @intCast((i * 37 + 11) % vocab_local);
        var ref_planes: ?[]u8 = null;
        var ref_logits: ?[]u8 = null;
        var ref_token: u32 = 0;
        for ([_]usize{ 4096, 2048, 8192 }) |chunk_rows| { // 1024 is refused: not byte-equal (2026-10-29)
            const result = try runChunkRowsPrefill(gpa, d, k, &model, stream, loop, clusters, ids, chunk_rows, context_rows);
            defer gpa.free(result.planes);
            defer gpa.free(result.logits);
            const same = if (chunk_rows == 4096) true else std.mem.eql(u8, ref_planes.?, result.planes) and std.mem.eql(u8, ref_logits.?, result.logits) and result.token == ref_token;
            if (chunk_rows == 4096) {
                ref_planes = try gpa.dupe(u8, result.planes);
                ref_logits = try gpa.dupe(u8, result.logits);
                ref_token = result.token;
            }
            std.debug.print("{s} glm53 chunk_rows {d} == 4096 (prompt {d})\n", .{ if (same) "PASS" else "FAIL", chunk_rows, prompt_len });
            failed = failed or !same;
        }
        gpa.free(ref_planes.?);
        gpa.free(ref_logits.?);
    }
    if (failed) return error.ChunkRowsParityFailed;
}

const ChunkRowsPrefill = struct { planes: []u8, logits: []u8, token: u32 };

fn runChunkRowsPrefill(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const forward.Kernels, model: *const weights.ModelWeights, stream: cuda.Stream, loop: *Loopback, clusters: bool, ids: []const u32, chunk_rows: usize, context_rows: usize) !ChunkRowsPrefill {
    const geometry: state.Geometry = .{ .layers = 2, .full_index_layers = 1, .graphs_bytes = 0, .vocab_local = vocab_local, .lane_clusters = clusters, .chunk_rows = chunk_rows };
    const admission = try state.admitRank(geometry, context_rows, .{}, 1 << 50, state.reserveFor(1 << 50, false, 0));
    var st = try state.State.init(d, geometry, context_rows, 0, 2, &admission, 1 << 50);
    defer st.deinit();
    const f: forward.Forward = .{ .gpa = gpa, .k = k, .w = model, .st = &st, .stream = stream, .coll = loop.collective(), .rank = 0, .chunk_rows = chunk_rows };
    const prefill = try freshPrefill(gpa, f, &st, stream, &.{ids});
    errdefer gpa.free(prefill.planes);
    const logits = try gpa.alloc(u8, vocab_local * 4);
    errdefer gpa.free(logits);
    try d.check(d.api.cuMemcpyDtoH_v2(logits.ptr, st.ws.logits, logits.len), "chunk_rows parity logits");
    return .{ .planes = prefill.planes, .logits = logits, .token = prefill.token };
}

/// Compare serial one-row decode with multi-row verify windows from identical prefills, including all KV/latent/rope/index bytes.
fn windowParity(gpa: std.mem.Allocator, k: *const forward.Kernels, model: *weights.ModelWeights, st: *state.State, stream: cuda.Stream, loop: *Loopback) !void {
    var failed = false;
    for ([_]usize{ 35, 2040 }) |prompt_len| {
        const prompt = try gpa.alloc(u32, prompt_len);
        defer gpa.free(prompt);
        for (prompt, 0..) |*x, i| x.* = @intCast((i * 37 + 11) % vocab_local);
        model.vocab = .{ .lo = 0, .hi = vocab_local };
        st.rank = 0;
        const f: forward.Forward = .{ .gpa = gpa, .k = k, .w = model, .st = st, .stream = stream, .coll = loop.collective(), .rank = 0, .decode_side = false };
        const baseline_prefill = try freshPrefill(gpa, f, st, stream, &.{prompt});
        defer gpa.free(baseline_prefill.planes);
        var baseline: engine.Engine = .{ .gpa = gpa, .f = f, .auto_graphs = false };
        defer baseline.deinit();
        try seedParityEngine(&baseline, prompt);
        // 48 is divisible by all supported copy-draft widths (including 12), so every
        // iteration below is a complete verify window and the logits destination fits.
        var inputs: [48]u32 = undefined;
        var expected: [48]u32 = undefined;
        const logits_row_bytes = vocab_local * 4;
        const reference_logits = try gpa.alloc(u8, inputs.len * logits_row_bytes);
        defer gpa.free(reference_logits);
        inputs[0] = baseline_prefill.token;
        for (0..inputs.len) |i| {
            try baseline.verify(inputs[i .. i + 1], expected[i .. i + 1]);
            try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(reference_logits.ptr + i * logits_row_bytes, st.ws.logits, logits_row_bytes), "serial side parity logits");
            try baseline.keep(1);
            if (i + 1 < inputs.len) inputs[i + 1] = expected[i];
        }
        try stream.synchronize();
        const reference = try gpa.alloc(u8, st.kv_arena.len);
        defer gpa.free(reference);
        try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(reference.ptr, st.kv_arena.ptr, reference.len), "serial parity KV download");
        for ([_]usize{ 1, 2, 4, 8, 12, 16 }) |width| {
            var side_f = f;
            side_f.decode_side = true;
            const fresh = try freshPrefill(gpa, f, st, stream, &.{prompt});
            gpa.free(fresh.planes);
            var candidate: engine.Engine = .{ .gpa = gpa, .f = side_f, .auto_graphs = true };
            defer candidate.deinit();
            try seedParityEngine(&candidate, prompt);
            // Match the 48-row input/reference window: the 12- and 16-row
            // replays write through rows 32..47 on the final iterations.
            var actual: [48]u32 = undefined;
            const actual_logits = try gpa.alloc(u8, actual.len * logits_row_bytes);
            defer gpa.free(actual_logits);
            var start: usize = 0;
            while (start < inputs.len) : (start += width) {
                try candidate.verify(inputs[start..][0..width], actual[start..][0..width]);
                try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(actual_logits.ptr + start * logits_row_bytes, st.ws.logits, width * logits_row_bytes), "window side parity logits");
                try candidate.keep(width);
            }
            try stream.synchronize();
            const planes = try gpa.alloc(u8, st.kv_arena.len);
            try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(planes.ptr, st.kv_arena.ptr, planes.len), "window parity KV download");
            const same_tokens = std.mem.eql(u32, &expected, &actual);
            const same_logits = std.mem.eql(u8, reference_logits, actual_logits);
            const same_planes = std.mem.eql(u8, reference, planes);
            var captured: u64 = 0;
            var replayed: u64 = 0;
            if (candidate.graphs) |cache| {
                captured = cache.stats.captured;
                replayed = cache.stats.replayed;
            }
            std.debug.print("{s} glm53 side parity w={d} prompt={d}: tokens {s}, logits {s}, planes {s}\n", .{ if (same_tokens and same_logits and same_planes) "PASS" else "FAIL", width, prompt_len, if (same_tokens) "equal" else "DIFFER", if (same_logits) "equal" else "DIFFER", if (same_planes) "equal" else "DIFFER" });
            if (captured > 0 and replayed > 0) {
                std.debug.print("PASS glm53 side graph replay parity w={d} prompt={d} ({d} captured, {d} replayed)\n", .{ width, prompt_len, captured, replayed });
            } else {
                std.debug.print("FAIL glm53 side graph replay coverage w={d} prompt={d} ({d} captured, {d} replayed)\n", .{ width, prompt_len, captured, replayed });
                failed = true;
            }
            if (!same_tokens or !same_logits or !same_planes) { planeDiff(st, reference, planes, prompt_len + inputs.len); failed = true; }
            gpa.free(planes);
        }
    }
    if (failed) return error.WindowParityFailed;
}

fn seedParityEngine(e: *engine.Engine, prompt: []const u32) !void {
    e.pos = prompt.len;
    e.frontier = prompt.len;
    try e.history.appendSlice(e.gpa, prompt);
}

fn makeAblation(gpa: std.mem.Allocator, d: *const cuda.Driver, scale: f32) !ablate.Ablation {
    const layers: usize = 2;
    const rbytes = layers * 2 * hidden * 4;
    const align_a = std.mem.alignForward(usize, rbytes, 256);
    var buf = try cuda.DeviceBuffer.alloc(d, align_a + layers * 2 * 4);
    errdefer buf.free();
    const host_r = try gpa.alloc(f32, layers * 2 * hidden);
    defer gpa.free(host_r);
    @memset(host_r, 0);
    for (0..layers * 2) |v| host_r[v * hidden + 17] = 1;
    var host_a: [layers * 2]f32 = @splat(scale);
    try buf.upload(0, std.mem.sliceAsBytes(host_r));
    try buf.upload(align_a, std.mem.sliceAsBytes(&host_a));
    var present: [78][2]bool = undefined;
    @memset(&present, .{ false, false });
    present[0] = .{ true, true };
    present[1] = .{ true, true };
    return .{ .layers = layers, .buf = buf, .present = present, .digest = @as([32]u8, @splat(0)) };
}

fn ablationScale(gpa: std.mem.Allocator, a: *ablate.Ablation, scale: f32) !void {
    const align_a = std.mem.alignForward(usize, a.layers * 2 * hidden * 4, 256);
    const values = try gpa.alloc(f32, a.layers * 2);
    defer gpa.free(values);
    @memset(values, scale);
    try a.buf.upload(align_a, std.mem.sliceAsBytes(values));
}

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa }) catch |e| {
        std.debug.print("FAIL glm53-graph-replay: {t}\n", .{e});
        return 1;
    };
    return 0;
}

fn freshPrefill(gpa: std.mem.Allocator, f: forward.Forward, st: *state.State, stream: cuda.Stream, parts: []const []const u32) !struct { planes: []u8, token: u32 } {
    const d = stream.d;
    try st.kv_arena.fill8(0, stream.handle);
    try st.workspace.fill8(0, stream.handle);
    if (st.stage_plane.len != 0) try st.stage_plane.fill8(0, stream.handle);
    var e: engine.Engine = .{ .gpa = gpa, .f = f, .auto_graphs = false };
    defer e.deinit();
    var tok: u32 = 0;
    for (parts) |part| tok = try e.prefill(part);
    try stream.synchronize();
    const planes = try gpa.alloc(u8, st.kv_arena.len);
    try d.check(d.api.cuMemcpyDtoH_v2(planes.ptr, st.kv_arena.ptr, planes.len), "kv download");
    return .{ .planes = planes, .token = tok };
}

/// The same prompt prefilled twice from zeroed planes (rank 0): KV/rope/index planes and the drawn token repeat bit for
/// bit. (Found the fused post-attention norm writing xn at the piece's row offset: past xn into the gathered partials
/// the same launch reads, for every piece after the first; and the rig's NULL-stream memsets racing its stream.)
fn prefillRepeatability(gpa: std.mem.Allocator, k: *const forward.Kernels, model: *weights.ModelWeights, st: *state.State, stream: cuda.Stream, loop: *Loopback) !void {
    model.vocab = .{ .lo = 0, .hi = vocab_local };
    st.rank = 0;
    const f_off: forward.Forward = .{ .gpa = gpa, .k = k, .w = model, .st = st, .stream = stream, .coll = loop.collective(), .rank = 0, .piece_pipe = false };
    const f_on: forward.Forward = .{ .gpa = gpa, .k = k, .w = model, .st = st, .stream = stream, .coll = loop.collective(), .rank = 0, .piece_pipe = true };
    for ([_]usize{ 35, 600, 2040, 2600 }) |prompt| {
        const ids = try gpa.alloc(u32, prompt);
        defer gpa.free(ids);
        for (ids, 0..) |*x, j| x.* = @intCast((j * 37 + 11) % vocab_local);
        const a = try freshPrefill(gpa, f_off, st, stream, &.{ids});
        defer gpa.free(a.planes);
        const b = try freshPrefill(gpa, f_on, st, stream, &.{ids});
        defer gpa.free(b.planes);
        const same = std.mem.eql(u8, a.planes, b.planes) and a.token == b.token;
        std.debug.print("{s} glm53 prefill pipe off/on prompt={d}: planes {s}, token {d} vs {d}\n", .{ if (same) "PASS" else "FAIL", prompt, if (std.mem.eql(u8, a.planes, b.planes)) "equal" else "DIFFER", a.token, b.token });
        if (!same) {
            planeDiff(st, a.planes, b.planes, prompt);
            return error.TestFailed;
        }
    }
}

// Resume == fresh needs a real DCP2 peer: this rig's loopback peer is the rank itself, so a resumed prefill's history
// fetch of the peer's (odd) rows returns this rank's own rows. It runs on the Sparks: tf-glm53-bringup resume, with
// the split at a multiple of the piece rows (512); a split inside a piece changes the projections' row counts, so
// their tiling and summation order, by design.

fn planeDiff(st: *const state.State, a: []const u8, b: []const u8, prompt: usize) void {
    const planes = [_]struct { []const u8, u64, usize, usize }{
        .{ "latent", st.latentAt(0) - st.kv_arena.ptr, st.layers, 512 * 2 },
        .{ "rope", st.ropeAt(0) - st.kv_arena.ptr, st.layers, 64 * 2 },
        .{ "index", st.indexAt(0) - st.kv_arena.ptr, st.full_index_layers, 128 * 2 },
    };
    for (planes) |pl| for (0..pl[2]) |layer| {
        var first: ?usize = null;
        var count: usize = 0;
        for (0..(prompt + 1) / 2 + 1) |slot| {
            const at = pl[1] + (layer * st.slots + slot) * pl[3];
            if (at + pl[3] > a.len) break;
            if (!std.mem.eql(u8, a[at..][0..pl[3]], b[at..][0..pl[3]])) {
                if (first == null) first = slot;
                count += 1;
            }
        }
        if (first) |fs| std.debug.print("  {s} layer {d}: {d} slots differ, first slot {d} (position {d})\n", .{ pl[0], layer, count, fs, 2 * fs });
    };
}
