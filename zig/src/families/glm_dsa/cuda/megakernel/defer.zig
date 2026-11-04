// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! MK8 deferred launch for the GLM-5.3 decode step. While a stream carries this sink, a launch of a PROVEN kernel
//! (the dispatch table below) is recorded instead of launched: its packed CUDA parameters behind a 16-byte
//! {gx, gy, gz, threads} header become one task of the pending segment. Anything else on the stream (another
//! kernel, a launch with cluster/PDL/cooperative attributes, an event, a copy, a collective, an explicit flush)
//! flushes first: the segment's tasks run as ONE persistent megakernel launch on the same stream, task i+1
//! depending on task i, so the result is the kernels' own (same code, same arguments, same order).
//!
//! Plans are cached by (op sequence, grid shapes); a hit refreshes only the payload bytes (pointers can change).
//! CUDA graph capture: a flush with a cached plan issues kernel launches only (payload store, reset, persistent
//! run; the payload bytes travel as the store kernel's parameter, so the graph node owns them), with no host sync
//! and no allocation once the pending arrays have grown in an eager warm-up step. A flush that misses the cache
//! inside a capture cannot upload a plan (no cuMemAlloc or sync copy there), so it launches the recorded kernels
//! one by one, unchanged: still bit-identical, just not fused. Engine.verify runs a key's first window eagerly
//! when TF_GLM53_MK=1 so its capture finds every plan cached.
//!
//! Launch shape: each plan key runs with one shape chosen by policy.zig when the key is first seen eagerly (plain
//! launches, the ready queue at one CTA per SM or at occupancy, the chain with either worker mapping): an override, the
//! per-device table, calibration over eager windows, or the default. A capture never chooses or calibrates.
//! Original to TensorFold (MK8); scheduler ABI in mk_runtime.cuh, proofs in mk_dispatch_*.cuh.
const std = @import("std");
const cuda = @import("cuda");
const mk = @import("mk_launch");
const graph = mk.graph_api;
/// launch.zig (trace layout, Plan) for the GPU tests that only import this module.
pub const launch_api = mk;
/// The launch-shape policy (per-device table, calibration, override).
pub const policy = @import("policy.zig");
pub const LaunchShape = policy.Shape;
/// Upper bound on the eager windows a calibration needs (every shape, the most windows per shape, one to settle).
pub const max_calibration_windows: usize = policy.shape_count * policy.max_windows + 1;

/// The 16-byte header every ROUND 4 payload starts with (tf_glm53_mk_hdr in mk_runtime.cuh).
pub const Header = extern struct { gx: u32, gy: u32, gz: u32, threads: u32 };

/// Mirrors of each proven kernel's __global__ parameter list, in signature order: the CUDA parameter ABI places
/// every parameter at its natural alignment, which is exactly these extern structs' layout. The C++ side checks
/// the same mirrors against decltype(&kernel) and the adapter payload offsets (mk_dispatch_util/attn.cuh).
pub const sig = struct {
    /// tf_glm_rmsnorm_6144 (glm_glue_norm.cuh RMSNORM(6144, ...)).
    pub const Rmsnorm6144 = extern struct { x: u64, x_stride: i64, w: u64, out: u64, o_stride: i64, xs: u64, eps: f32, sums: i32 };
    /// tf_glm53_qpe_pack (glm_dsa/mla.cu).
    pub const QpePack = extern struct { q: u64, cos: u64, sin: u64, qpe: u64, rows: i32, local_heads: i32 };
    /// tf_glm53_attend (glm_dsa/mla.cu).
    pub const Attend = extern struct {
        qa: u64,
        qpe: u64,
        latent: u64,
        rope: u64,
        stage_latent: u64,
        stage_rope: u64,
        positions: u64,
        global_tokens: u64,
        local_slots: u64,
        global_counts: u64,
        local_counts: u64,
        dense_len: u64,
        partial: u64,
        rows: i32,
        local_heads: i32,
        rank: i32,
        world: i32,
        prefill: i32,
    };
    pub const KvWrite = extern struct { proj: u64, norm: u64, cos: u64, sin: u64, positions: u64, latent: u64, rope: u64, stage_latent: u64, stage_rope: u64, rows: i32, stride: i32, rank: i32, world: i32 };
    pub const Absorb = extern struct { q: u64, kvb_codes: u64, kvb_scales: u64, qa: u64, rows: i32, local_heads: i32, head0: i32 };
    pub const MlaMerge = extern struct { partial_local: u64, partial_peer: u64, out: u64, lse_out: u64, rows: i32, local_heads: i32, rank: i32 };
    pub const ExpandV = extern struct { ol: u64, kvb_codes: u64, kvb_scales: u64, out: u64, rows: i32, local_heads: i32, head0: i32 };
    pub const LowrankCombine = extern struct { routed: u64, shared: u64, b: u64, mid: u64, out: u64, rows: i32, hidden: i32, rank: i32, b_stride: i32 };
    pub const ResidualRmsnorm = extern struct { x: u64, parts: u64, w: u64, normed: u64, rows: i32, eps: f32 };
    pub const LowrankFinish = extern struct { b: u64, mid: u64, acc: u64, rows: i32, hidden: i32, rank: i32, b_stride: i32 };
    pub const EmbedPick = extern struct { gathered: u64, owners: u64, out: u64, rows: i32, hidden: i32, world: i32 };
    pub const RouterPart = extern struct { x: u64, w: u64, parts: u64, rows: i64, x_stride: i64, d: i64, ne: i64, ks: i32 };
    pub const RouterTopk = extern struct { logits: u64, bias: u64, pick: u64, wts: u64, scale: f32, ne: i32, top_k: i32, norm: i32 };
    pub const SliceSum = extern struct { part: u64, out: u64, total: i64, ks: i32, f32: i32 };
    pub const IndexQuery = extern struct { input: u64, cos: u64, sin: u64, output: u64 };
    pub const IndexKey = extern struct { proj: u64, weight: u64, bias: u64, cos: u64, sin: u64, positions: u64, output: u64, stride: i32, col: i32, rank: i32, world: i32, eps: f32 };
    pub const IndexScores = extern struct { qi: u64, iw: u64, iw_stride: i32, iw_col: i32, keys: u64, positions: u64, scores: u64, nkeys: i32, key_stride: i32, rank: i32, world: i32 };
    pub const HeadTop1 = extern struct { x: u64, vocab: u32, offset: u32, values: u64, ids: u64 };
    pub const HeadFill = extern struct { values: u64, ids: u64, count: u32 };
    pub const HeadPart = extern struct { x: u64, vocab: u32, offset: u32, k: u32, scratch: u64 };
    pub const HeadMerge = extern struct { scratch: u64, vocab: u32, k: u32, values: u64, ids: u64 };
    pub const HeadLse = extern struct { x: u64, vocab: u32, partial: u64 };
    pub const HeadGather = extern struct { logits: u64, ids: u64, vocab: u32, offset: u32, values: u64, probes: u32, rows: u32 };
    pub const HeadMass = extern struct { probe_logits: u64, lse: u64, logp: u64, mass_outside: u64, probes: u32, rows: u32 };
    /// tf_glm53_swiglu_plain (glm_dsa/glue.cu).
    pub const SwigluPlain = extern struct { gu: u64, out: u64, rows: i32, width: i32 };
    /// tf_glm53_lowrank_partial (glm_dsa/glue.cu).
    pub const LowrankPartial = extern struct { x: u64, a: u64, partials: u64, rows: i32, hidden: i32, rank: i32, x_stride: i32, a_stride: i32, splits: i32 };
    /// tf_glm53_lowrank_reduce (glm_dsa/glue.cu).
    pub const LowrankReduce = extern struct { partials: u64, mid: u64, total: i32, splits: i32 };
    /// tf_glm53_moe_residual_add (glm_dsa/glue.cu).
    pub const MoeResidualAdd = extern struct { x: u64, xout: u64, parts: u64, count: i32 };
};

/// A task payload as the scheduler reads it: the header, then the parameters at 16 + their ABI offsets.
pub fn Payload(comptime Sig: type) type {
    return extern struct { h: Header, p: Sig };
}

/// One dispatchable kernel: loader symbol, scheduler op, and its parameter ABI (count, offsets, packed length).
pub const Entry = struct {
    symbol: []const u8,
    op: mk.Op,
    param_count: usize,
    /// End of the last parameter: what cuda.Args.used is for this signature (no tail padding).
    param_bytes: usize,
    /// @sizeOf(Payload(Sig)): the header, the parameters, the struct's tail padding.
    payload_bytes: usize,
    offsets: []const u16,
};

fn entry(comptime symbol: []const u8, comptime op: mk.Op, comptime Sig: type) Entry {
    const info = @typeInfo(Sig).@"struct";
    const names = info.field_names;
    const offsets = comptime blk: {
        var out: [names.len]u16 = undefined;
        for (names, 0..) |name, i| out[i] = @offsetOf(Sig, name);
        break :blk out;
    };
    const last = names[names.len - 1];
    const P = Payload(Sig);
    comptime {
        if (@offsetOf(P, "p") != @sizeOf(Header) or @sizeOf(P) != @sizeOf(Header) + @sizeOf(Sig))
            @compileError("MK8 payload header must keep the parameters at 16 + their ABI offset: " ++ symbol);
    }
    const final = offsets;
    return .{
        .symbol = symbol,
        .op = op,
        .param_count = names.len,
        .param_bytes = @offsetOf(Sig, last) + @sizeOf(info.field_types[names.len - 1]),
        .payload_bytes = @sizeOf(P),
        .offsets = &final,
    };
}

/// The dispatch table: symbol -> op. Only kernels whose adapter payload is proven to be header + signature ABI
/// (C++ static_asserts in mk_dispatch_util.cuh / mk_dispatch_attn.cuh) are listed; every other kernel is a boundary.
pub const table = [_]Entry{
    entry("tf_glm_rmsnorm_6144", .attn_input_norm, sig.Rmsnorm6144),
    entry("tf_glm53_qpe_pack", .attn_qpe_pack, sig.QpePack),
    entry("tf_glm53_attend", .attn_attend, sig.Attend),
    entry("tf_glm53_kv_write", .attn_kv_write, sig.KvWrite),
    entry("tf_glm53_absorb", .attn_absorb, sig.Absorb),
    entry("tf_glm53_mla_merge", .attn_mla_merge, sig.MlaMerge),
    entry("tf_glm53_expand_v", .attn_expand_v, sig.ExpandV),
    entry("tf_glm53_lowrank_combine_bf16", .lowrank_combine_bf16, sig.LowrankCombine),
    entry("tf_glm53_residual_rmsnorm_6144", .residual_rmsnorm_6144, sig.ResidualRmsnorm),
    entry("tf_glm53_lowrank_finish", .lowrank_finish, sig.LowrankFinish),
    entry("tf_glm53_embed_pick", .embed_pick, sig.EmbedPick),
    entry("tf_glm_router_part", .moe_router_part, sig.RouterPart),
    entry("tf_glm_router_tile", .moe_router_part, sig.RouterPart),
    entry("tf_glm_topk", .moe_router_topk, sig.RouterTopk),
    entry("tf_glm_slice_sum", .moe_router_slice_sum, sig.SliceSum),
    entry("tf_glm53_index_q", .index_query, sig.IndexQuery),
    entry("tf_glm53_index_key", .index_key, sig.IndexKey),
    entry("tf_glm53_index_scores", .index_scores, sig.IndexScores),
    entry("tf_glm53_top1_f32", .head_top1_f32, sig.HeadTop1),
    entry("tf_glm53_cand_fill_f32", .head_candidate_fill_f32, sig.HeadFill),
    entry("tf_glm53_cand_part_f32", .head_candidate_part_f32, sig.HeadPart),
    entry("tf_glm53_cand_merge_f32", .head_candidate_merge_f32, sig.HeadMerge),
    entry("tf_glm53_lse_f32", .head_lse_f32, sig.HeadLse),
    entry("tf_glm53_gate_gather_f32", .head_gate_gather_f32, sig.HeadGather),
    entry("tf_glm53_gate_mass_f32", .head_gate_mass_f32, sig.HeadMass),
    entry("tf_glm53_swiglu_plain", .swiglu, sig.SwigluPlain),
    entry("tf_glm53_lowrank_partial", .lowrank_partial, sig.LowrankPartial),
    entry("tf_glm53_lowrank_reduce", .lowrank_reduce, sig.LowrankReduce),
    entry("tf_glm53_moe_residual_add", .moe_residual_add, sig.MoeResidualAdd),
};

pub fn lookup(symbol: []const u8) ?*const Entry {
    for (&table) |*e| if (std.mem.eql(u8, e.symbol, symbol)) return e;
    return null;
}

/// What a plan is keyed by, per task: the op and the launch shape.
pub const Shape = extern struct { op: u32, threads: u32, shared: u32, gx: u32, gy: u32, gz: u32 };

/// One recorded task, as the fallback path needs it to issue the original launch.
pub const Recorded = struct {
    entry: *const Entry,
    function: cuda.abi.Function,
    grid: cuda.Dim3,
    block: cuda.Dim3,
    shared: u32,
    /// The task's payload in the segment image: header + parameters.
    payload: []const u8,
};

/// What a backend needs to upload a new plan: a linear chain in recording order.
pub const PlanSpec = struct {
    tasks: []graph.Task,
    initial_counters: []u32,
    successors: []u32,
    ops: []const mk.Op,
    blocks: []const u32,
    payloads: []const []const u8,
    /// The largest dynamic shared memory any recorded launch asked for: the plan's task arena.
    arena_bytes: u32,
};

/// The device side of the executor. GpuBackend is the real one; host tests record the calls.
pub const Backend = struct {
    ctx: *anyopaque,
    /// A new plan for one fused launch shape (never .plain; never called inside a stream capture).
    upload_fn: *const fn (ctx: *anyopaque, spec: *const PlanSpec, shape: LaunchShape) anyerror!*anyopaque,
    /// Payload refresh then one persistent launch on `stream`: launches only, no host sync.
    run_fn: *const fn (ctx: *anyopaque, plan: *anyopaque, image: []const u8, stream: cuda.Stream) anyerror!void,
    /// The original launch of one recorded task (the .plain shape, and a capture that misses the plan cache).
    launch_fn: *const fn (ctx: *anyopaque, task: *const Recorded, stream: cuda.Stream) anyerror!void,
    capturing_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream) anyerror!bool,
    free_fn: *const fn (ctx: *anyopaque, plan: *anyopaque) void,
    /// Calibration timers (eager windows only): a start/stop pair of timing events.
    timer_new_fn: *const fn (ctx: *anyopaque) anyerror!*anyopaque,
    timer_mark_fn: *const fn (ctx: *anyopaque, timer: *anyopaque, stream: cuda.Stream, stop: bool) anyerror!void,
    /// Microseconds between the two marks (waits for the stop mark: a host sync, never inside a capture).
    timer_read_fn: *const fn (ctx: *anyopaque, timer: *anyopaque) anyerror!f32,
    timer_free_fn: *const fn (ctx: *anyopaque, timer: *anyopaque) void,
};

pub const Stats = struct {
    segments: u64 = 0,
    tasks: u64 = 0,
    /// Plan keys (cache entries).
    plans: u64 = 0,
    /// Fused plans uploaded (a calibrating key uploads one per fused shape).
    uploads: u64 = 0,
    hits: u64 = 0,
    fallback_segments: u64 = 0,
    fallback_tasks: u64 = 0,
    /// Segments a .plain plan ran as the recorded launches.
    plain_segments: u64 = 0,
    /// Timed calibration windows, and keys whose shape calibration chose.
    calibration_windows: u64 = 0,
    calibrated: u64 = 0,
};

/// One plan key's calibration in progress: a plan per fused shape, a timer, the window not yet read back.
const Calibrating = struct {
    c: policy.Calibration,
    handles: [policy.shape_count]?*anyopaque = @splat(null),
    timer: ?*anyopaque = null,
    pending: ?LaunchShape = null,
    /// What the key runs with if a capture arrives before any window was measured.
    fallback: LaunchShape,
    fallback_source: policy.Source,
};

/// A plan cache entry: the key (segment launch shapes and rows bucket) and the launch shape chosen for it.
pub const Plan = struct {
    hash: u64,
    bucket: u8,
    shapes: []Shape,
    shape: LaunchShape,
    source: policy.Source,
    /// The fused plan for `shape`; null for .plain (and while calibrating).
    handle: ?*anyopaque,
    cal: ?*Calibrating = null,
};

const no_bucket: u8 = 0xff;

/// The per-stream sink: the pending segment, the plan cache, and the backend that runs them.
pub const Deferral = struct {
    allocator: std.mem.Allocator,
    backend: Backend,
    /// How a new plan key picks its launch shape (Executor.create reads it from the environment).
    policy: policy.Policy = .{},
    /// The table key (Executor.create fills it from the device).
    device: policy.Device = .{},
    /// The caller's row count, for a segment whose kernels do not show it (0 = unknown).
    rows: u32 = 0,
    vtable: cuda.Stream.Deferred = undefined,
    shapes: std.ArrayList(Shape) = .empty,
    entries: std.ArrayList(*const Entry) = .empty,
    functions: std.ArrayList(cuda.abi.Function) = .empty,
    launches: std.ArrayList([2]cuda.Dim3) = .empty,
    /// The segment's payloads at layoutPayloads offsets (16-byte aligned, recording order).
    image: std.ArrayList(u8) = .empty,
    offsets: std.ArrayList(usize) = .empty,
    // Chain and spec scratch, reused (no allocation once warm).
    tasks: std.ArrayList(graph.Task) = .empty,
    counters: std.ArrayList(u32) = .empty,
    successors: std.ArrayList(u32) = .empty,
    ops: std.ArrayList(mk.Op) = .empty,
    blocks: std.ArrayList(u32) = .empty,
    payloads: std.ArrayList([]const u8) = .empty,
    /// Pointers stay valid as the cache grows (a calibrating flush holds one across backend calls).
    plans: std.ArrayList(*Plan) = .empty,
    /// Plans and timers a calibration dropped inside a capture, freed at the next eager flush (no cuMemFree while
    /// capturing).
    retired_plans: std.ArrayList(*anyopaque) = .empty,
    retired_timers: std.ArrayList(*anyopaque) = .empty,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator, backend: Backend) Deferral {
        return .{ .allocator = allocator, .backend = backend };
    }

    pub fn deinit(s: *Deferral) void {
        const a = s.allocator;
        for (s.plans.items) |p| {
            if (p.handle) |h| s.backend.free_fn(s.backend.ctx, h);
            if (p.cal) |cal| s.dropCalibration(cal);
            a.free(p.shapes);
            a.destroy(p);
        }
        s.freeRetired();
        inline for (.{ "shapes", "entries", "functions", "launches", "image", "offsets", "tasks", "counters", "successors", "ops", "blocks", "payloads", "plans", "retired_plans", "retired_timers" }) |name| @field(s, name).deinit(a);
        s.* = undefined;
    }

    /// The sink to hang on a stream (cuda.Stream.withDeferred). `s` must not move while a stream holds it.
    pub fn sink(s: *Deferral) *const cuda.Stream.Deferred {
        s.vtable = .{ .ctx = s, .record_fn = recordRaw, .flush_fn = flushRaw };
        return &s.vtable;
    }

    pub fn pending(s: *const Deferral) usize {
        return s.shapes.items.len;
    }

    /// True while some plan key is still being calibrated (Engine.verify keeps its warm-up eager until then).
    pub fn calibrating(s: *const Deferral) bool {
        for (s.plans.items) |p| if (p.cal != null) return true;
        return false;
    }

    /// The cache entries, in creation order (tests and reports).
    pub fn planList(s: *const Deferral) []const *Plan {
        return s.plans.items;
    }

    fn recordRaw(ctx: *anyopaque, launch: *const cuda.Stream.DeferredLaunch) cuda.Error!bool {
        const s: *Deferral = @ptrCast(@alignCast(ctx));
        return s.record(launch) catch |err| {
            std.log.err("GLM-5.3 MK8 cannot defer {s}: {s}", .{ launch.name, @errorName(err) });
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.Invalid,
            };
        };
    }

    fn flushRaw(ctx: *anyopaque, stream: cuda.Stream) cuda.Error!void {
        const s: *Deferral = @ptrCast(@alignCast(ctx));
        s.flush(stream) catch |err| {
            std.log.err("GLM-5.3 MK8 segment flush failed: {s}", .{@errorName(err)});
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.Invalid,
            };
        };
    }

    /// True when the launch became a pending task; false makes the launch helper flush and launch it normally.
    pub fn record(s: *Deferral, launch: *const cuda.Stream.DeferredLaunch) !bool {
        const e = lookup(launch.name) orelse return false;
        // The adapters model a block as (threads, 1, 1): a 2-D or 3-D block stays a boundary.
        if (launch.block.y != 1 or launch.block.z != 1) return false;
        const threads: u32 = launch.block.x * launch.block.y * launch.block.z;
        if (!mk.isDispatchable(threads, launch.shared, false, false)) return false;
        // The call site's packed arguments must be the proven signature, byte for byte.
        if (launch.param_count != e.param_count or launch.params.len != e.param_bytes) return error.MkSignatureMismatch;
        if (launch.offsets.len != 0 and !std.mem.eql(u16, launch.offsets, e.offsets)) return error.MkSignatureMismatch;
        const g = launch.grid;
        const work = @as(u64, g.x) * g.y * g.z;
        if (work == 0 or work > std.math.maxInt(u32)) return false;
        const a = s.allocator;
        const at = std.mem.alignForward(usize, s.image.items.len, 16);
        try s.image.appendNTimes(a, 0, at - s.image.items.len);
        const header: Header = .{ .gx = g.x, .gy = g.y, .gz = g.z, .threads = threads };
        try s.image.appendSlice(a, std.mem.asBytes(&header));
        try s.image.appendSlice(a, launch.params);
        try s.image.appendNTimes(a, 0, e.payload_bytes - @sizeOf(Header) - launch.params.len);
        try s.offsets.append(a, at);
        try s.shapes.append(a, .{ .op = @backingInt(e.op), .threads = threads, .shared = launch.shared, .gx = g.x, .gy = g.y, .gz = g.z });
        try s.entries.append(a, e);
        try s.functions.append(a, launch.function);
        try s.launches.append(a, .{ g, launch.block });
        return true;
    }

    fn payloadAt(s: *const Deferral, i: usize) []const u8 {
        return s.image.items[s.offsets.items[i]..][0..s.entries.items[i].payload_bytes];
    }

    fn clear(s: *Deferral) void {
        inline for (.{ "shapes", "entries", "functions", "launches", "image", "offsets" }) |name| @field(s, name).clearRetainingCapacity();
    }

    fn find(s: *const Deferral, hash: u64, b: u8) ?*Plan {
        for (s.plans.items) |p| if (p.hash == hash and p.bucket == b and std.mem.eql(u8, std.mem.sliceAsBytes(p.shapes), std.mem.sliceAsBytes(s.shapes.items))) return p;
        return null;
    }

    /// The pending segment's row count: the grid x of a kernel launched one block row per token (input norm, qpe
    /// pack, attend, residual norm), else the caller's hint.
    pub fn segmentRows(s: *const Deferral) u32 {
        var rows: u32 = 0;
        for (s.shapes.items) |sh| {
            const op: mk.Op = @fromBackingInt(@intCast(sh.op));
            switch (op) {
                .attn_input_norm, .attn_qpe_pack, .attn_attend, .residual_rmsnorm_6144 => rows = @max(rows, sh.gx),
                else => {},
            }
        }
        return if (rows != 0) rows else s.rows;
    }

    /// Run the pending segment on `stream` (the sink is stripped so the backend's own launches go straight through).
    pub fn flush(s: *Deferral, stream: cuda.Stream) !void {
        const n = s.shapes.items.len;
        if (n == 0) return;
        defer s.clear();
        var plain = stream;
        plain.deferred = null;
        const capturing = try s.backend.capturing_fn(s.backend.ctx, plain);
        if (!capturing) s.freeRetired();
        const rows = s.segmentRows();
        const b = policy.bucket(rows) orelse no_bucket;
        var h = std.hash.Wyhash.init(0x4d4b38);
        h.update(std.mem.sliceAsBytes(s.shapes.items));
        h.update(&.{b});
        const hash = h.final();
        const cached = s.find(hash, b);
        if (cached != null) s.stats.hits += 1;
        const plan = cached orelse blk: {
            // A capture cannot upload a plan or choose a shape: the recorded kernels go in unchanged.
            if (capturing) {
                try s.launchRecorded(plain);
                s.stats.fallback_segments += 1;
                s.stats.fallback_tasks += n;
                return;
            }
            break :blk try s.newPlan(hash, b, rows);
        };
        try s.runPlan(plan, plain, capturing);
        s.stats.segments += 1;
        s.stats.tasks += n;
    }

    fn launchRecorded(s: *Deferral, stream: cuda.Stream) !void {
        for (0..s.shapes.items.len) |i| {
            const task: Recorded = .{ .entry = s.entries.items[i], .function = s.functions.items[i], .grid = s.launches.items[i][0], .block = s.launches.items[i][1], .shared = s.shapes.items[i].shared, .payload = s.payloadAt(i) };
            try s.backend.launch_fn(s.backend.ctx, &task, stream);
        }
    }

    fn uploadShape(s: *Deferral, shape: LaunchShape) !*anyopaque {
        var arena: u32 = 0;
        for (s.shapes.items) |sh| arena = @max(arena, sh.shared);
        const spec: PlanSpec = .{ .tasks = s.tasks.items, .initial_counters = s.counters.items, .successors = s.successors.items, .ops = s.ops.items, .blocks = s.blocks.items, .payloads = s.payloads.items, .arena_bytes = arena };
        const handle = try s.backend.upload_fn(s.backend.ctx, &spec, shape);
        s.stats.uploads += 1;
        return handle;
    }

    /// A cache entry for the pending segment (eager only): the policy's shape, or a calibration with one plan per
    /// fused shape uploaded up front (so a capture that ends the calibration early finds its winner's plan).
    fn newPlan(s: *Deferral, hash: u64, b: u8, rows: u32) !*Plan {
        const a = s.allocator;
        try s.plans.ensureUnusedCapacity(a, 1);
        const p = try a.create(Plan);
        errdefer a.destroy(p);
        const shapes = try a.dupe(Shape, s.shapes.items);
        errdefer a.free(shapes);
        try s.buildChain();
        switch (s.policy.decide(s.device, rows)) {
            .fixed => |f| {
                const handle: ?*anyopaque = if (f.shape == .plain) null else try s.uploadShape(f.shape);
                p.* = .{ .hash = hash, .bucket = b, .shapes = shapes, .shape = f.shape, .source = f.source, .handle = handle };
            },
            .calibrate => {
                const cal = try a.create(Calibrating);
                const fallback = s.policy.fallback(s.device, rows);
                cal.* = .{ .c = policy.Calibration.init(s.policy.windows), .fallback = fallback, .fallback_source = if (policy.lookup(s.policy.table, s.device, rows) != null) .table else .default };
                errdefer s.dropCalibration(cal);
                for (policy.all) |shape| if (shape != .plain) {
                    cal.handles[@backingInt(shape)] = try s.uploadShape(shape);
                };
                cal.timer = try s.backend.timer_new_fn(s.backend.ctx);
                p.* = .{ .hash = hash, .bucket = b, .shapes = shapes, .shape = fallback, .source = .calibration, .handle = null, .cal = cal };
            },
        }
        s.plans.appendAssumeCapacity(p);
        s.stats.plans += 1;
        return p;
    }

    fn runPlan(s: *Deferral, p: *Plan, stream: cuda.Stream, capturing: bool) !void {
        if (p.cal) |cal| {
            if (capturing) {
                // The choice is fixed before the capture: the best window measured so far, else the fallback.
                if (cal.c.pick(cal.fallback, policy.keep_margin)) |best| s.settle(p, best, .calibration, true) else s.settle(p, cal.fallback, cal.fallback_source, true);
            } else {
                if (cal.pending) |shape| {
                    cal.c.record(shape, try s.backend.timer_read_fn(s.backend.ctx, cal.timer.?));
                    cal.pending = null;
                    s.stats.calibration_windows += 1;
                }
                if (cal.c.next()) |shape| {
                    try s.backend.timer_mark_fn(s.backend.ctx, cal.timer.?, stream, false);
                    try s.runShape(shape, cal.handles[@backingInt(shape)], stream);
                    try s.backend.timer_mark_fn(s.backend.ctx, cal.timer.?, stream, true);
                    cal.pending = shape;
                    return;
                }
                s.settle(p, cal.c.pick(cal.fallback, policy.keep_margin) orelse cal.fallback, .calibration, false);
            }
        }
        try s.runShape(p.shape, p.handle, stream);
    }

    /// End a key's calibration with `shape`: keep its plan, drop the others and the timer.
    fn settle(s: *Deferral, p: *Plan, shape: LaunchShape, source: policy.Source, capturing: bool) void {
        const cal = p.cal.?;
        p.shape = shape;
        p.source = source;
        p.handle = cal.handles[@backingInt(shape)];
        cal.handles[@backingInt(shape)] = null;
        p.cal = null;
        if (source == .calibration) s.stats.calibrated += 1;
        if (capturing) {
            for (cal.handles) |h| if (h) |x| s.retired_plans.append(s.allocator, x) catch s.backend.free_fn(s.backend.ctx, x);
            if (cal.timer) |t| s.retired_timers.append(s.allocator, t) catch s.backend.timer_free_fn(s.backend.ctx, t);
            cal.handles = @splat(null);
            cal.timer = null;
        }
        s.dropCalibration(cal);
    }

    fn dropCalibration(s: *Deferral, cal: *Calibrating) void {
        for (cal.handles) |h| if (h) |x| s.backend.free_fn(s.backend.ctx, x);
        if (cal.timer) |t| s.backend.timer_free_fn(s.backend.ctx, t);
        s.allocator.destroy(cal);
    }

    fn freeRetired(s: *Deferral) void {
        for (s.retired_plans.items) |x| s.backend.free_fn(s.backend.ctx, x);
        for (s.retired_timers.items) |t| s.backend.timer_free_fn(s.backend.ctx, t);
        s.retired_plans.clearRetainingCapacity();
        s.retired_timers.clearRetainingCapacity();
    }

    fn runShape(s: *Deferral, shape: LaunchShape, handle: ?*anyopaque, stream: cuda.Stream) !void {
        if (shape == .plain) {
            try s.launchRecorded(stream);
            s.stats.plain_segments += 1;
            return;
        }
        try s.backend.run_fn(s.backend.ctx, handle orelse return error.MkPlanMissing, s.image.items, stream);
    }

    /// Task i+1 depends on task i: the persistent run keeps the recording (stream) order.
    fn buildChain(s: *Deferral) !void {
        const a = s.allocator;
        const n = s.shapes.items.len;
        inline for (.{ "tasks", "counters", "successors", "ops", "blocks", "payloads" }) |name| @field(s, name).clearRetainingCapacity();
        for (0..n) |i| {
            const last = i + 1 == n;
            // `kind` is informational for the scheduler; it dispatches on the op stored in the record.
            try s.tasks.append(a, .{ .kind = .input_norm, .layer = 0, .row = 0, .aux = @intCast(@min(i, std.math.maxInt(u16))), .dependency_count = @intFromBool(i > 0), .successor_offset = @intCast(@min(i, n - 1)), .successor_count = @intFromBool(!last) });
            try s.counters.append(a, @intFromBool(i > 0));
            if (!last) try s.successors.append(a, @intCast(i + 1));
            try s.ops.append(a, s.entries.items[i].op);
            const sh = s.shapes.items[i];
            try s.blocks.append(a, sh.gx * sh.gy * sh.gz);
            try s.payloads.append(a, s.payloadAt(i));
        }
    }
};

/// mk_runtime's store kernel parameter (megakernel.cu tf_glm53_mk_store_chunk).
pub const StoreChunk = extern struct {
    dst: u64,
    len: u32,
    reserved: u32 = 0,
    bytes: [store_bytes]u8,

    pub const store_bytes = 2032;
};

comptime {
    std.debug.assert(@sizeOf(StoreChunk) == 2048 and @offsetOf(StoreChunk, "bytes") == 16);
    std.debug.assert(@sizeOf(StoreChunk) <= cuda.Args.max_bytes);
}

/// How the CUDA backend runs a fused plan (one per fused LaunchShape, Config.of).
pub const Config = struct {
    /// .chain: tf_glm53_mk_run_chain (block b of each task on worker b % workers, no ready queue);
    /// .queue: tf_glm53_mk_run (the dependency ready queue over every op, as the MK2-MK7 runtime tests launch it).
    interpreter: enum { chain, queue } = .chain,
    /// Chain worker mapping (launch.ChainMap): .sm pairs block b with b + SMs on one SM, .block uses blockIdx.x.
    map: mk.ChainMap = .sm,
    /// .plan: the task arena is the plan's largest recorded dynamic shared memory; .full: task_smem_bytes always.
    arena: enum { plan, full } = .plan,
    /// Persistent CTAs per SM; 0 = as many as can be resident (cuOccupancyMaxActiveBlocksPerMultiprocessor).
    blocks_per_sm: u32 = 0,

    /// The MK8 launch shape before the rows=4 fix: the ready-queue interpreter, a 48 KiB arena, one CTA per SM.
    pub const one_per_sm: Config = .{ .interpreter = .queue, .arena = .full, .blocks_per_sm = 1 };

    pub fn chained(c: Config) bool {
        return c.interpreter == .chain;
    }

    /// The backend configuration of a fused shape; null for .plain (no persistent launch).
    pub fn of(shape: LaunchShape) ?Config {
        return switch (shape) {
            .plain => null,
            .queue => one_per_sm,
            .queue_occ => .{ .interpreter = .queue },
            .chain_block => .{ .map = .block },
            .chain_sm => .{},
        };
    }
};

/// One uploaded plan and the persistent launch shape chosen for it.
pub const Prepared = struct {
    u: mk.Uploaded,
    plan: mk.Plan,
    config: Config,
};

/// The persistent grid for a plan: `per_sm` CTAs on each of `sms` SMs, never more than the plan's widest task has
/// blocks (a CTA beyond that only polls the ready queue).
pub fn persistentBlocks(sms: u32, per_sm: u32, blocks: []const u32) u32 {
    var widest: u32 = 1;
    for (blocks) |n| widest = @max(widest, n);
    return @max(1, @min(sms * @max(per_sm, 1), widest));
}

/// The CUDA backend: the persistent scheduler image with both interpreters, one launch.Uploaded per fused plan.
pub const GpuBackend = struct {
    allocator: std.mem.Allocator,
    d: *const cuda.Driver,
    module: cuda.Module,
    chain: mk.Functions,
    queue: mk.Functions,
    store: cuda.Function,
    sms: u32,
    /// The policy table key: SM count and compute capability.
    device: policy.Device,
    timeout: u32 = 10_000_000,
    /// Test-only: device address of a launch.traceLen(task_count) u64 buffer, or 0 (the production value).
    trace: u64 = 0,
    uploaded: std.ArrayList(*Prepared) = .empty,
    /// The chain scheduler's per-SM scratch (launch.sm_words_len words), zeroed by every chain reset.
    sm_words: cuda.DeviceBuffer,

    /// Loads the embedded glm53_megakernel image and resolves both interpreters.
    pub fn init(allocator: std.mem.Allocator, d: *const cuda.Driver) !GpuBackend {
        if (!cuda.kernels.available or cuda.kernels.glm53_megakernel.len == 0) return error.KernelsNotEmbedded;
        var module = try cuda.Module.load(d, cuda.kernels.glm53_megakernel);
        errdefer module.unload();
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var sms: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        var major: c_int = 0;
        var minor: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev), "cuDeviceGetAttribute");
        try d.check(d.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        // Opt in once for the largest arena; a plan's launch then asks for its own (smaller or equal) arena.
        const widest = try mk.Plan.init(@intCast(@max(sms, 1)), 1, mk.task_smem_bytes);
        const chain = try mk.Functions.resolveNamed(module, widest, "tf_glm53_mk_run_chain");
        const queue = try mk.Functions.resolveNamed(module, widest, "tf_glm53_mk_run");
        const store = try module.function("tf_glm53_mk_store");
        const words = try cuda.DeviceBuffer.alloc(d, mk.sm_words_len * 4);
        const n: u32 = @intCast(@max(sms, 1));
        return .{ .allocator = allocator, .d = d, .module = module, .chain = chain, .queue = queue, .store = store, .sms = n, .device = .{ .sms = n, .cc = @intCast(10 * major + minor) }, .sm_words = words };
    }

    pub fn deinit(b: *GpuBackend) void {
        for (b.uploaded.items) |p| {
            p.u.deinit();
            b.allocator.destroy(p);
        }
        b.uploaded.deinit(b.allocator);
        b.sm_words.free();
        b.module.unload();
        b.* = undefined;
    }

    pub fn backend(b: *GpuBackend) Backend {
        return .{ .ctx = b, .upload_fn = upload, .run_fn = run, .launch_fn = launchOne, .capturing_fn = capturing, .free_fn = free, .timer_new_fn = timerNew, .timer_mark_fn = timerMark, .timer_read_fn = timerRead, .timer_free_fn = timerFree };
    }

    fn fnsOf(b: *const GpuBackend, config: Config) mk.Functions {
        return if (config.chained()) b.chain else b.queue;
    }

    /// The launch shape for a plan: its arena, then as many CTAs per SM as that arena and the interpreter's
    /// registers allow (or the configured count).
    pub fn planFor(b: *const GpuBackend, config: Config, arena_bytes: u32, blocks: []const u32) !mk.Plan {
        const arena: u32 = switch (config.arena) {
            .plan => @intCast(std.mem.alignForward(usize, arena_bytes, 128)),
            .full => mk.task_smem_bytes,
        };
        if (arena > mk.task_smem_bytes) return error.MkArenaTooLarge;
        const resident = @max(1, try b.fnsOf(config).run.occupancy(mk.block_threads, arena));
        const per_sm = if (config.blocks_per_sm != 0) @min(config.blocks_per_sm, resident) else resident;
        var plan = try mk.Plan.init(b.sms, per_sm, arena);
        // The queue grid stops at the widest task (more CTAs only poll); the chain's SM mapping needs every slot.
        if (!config.chained()) plan.persistent_blocks = persistentBlocks(b.sms, per_sm, blocks);
        return plan;
    }

    fn upload(ctx: *anyopaque, spec: *const PlanSpec, shape: LaunchShape) anyerror!*anyopaque {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        const config = Config.of(shape) orelse return error.MkPlainHasNoPlan;
        const g: graph.Graph = .{ .allocator = b.allocator, .params = .{ .rows = 1, .layers = 1, .dcp = 1 }, .tasks = spec.tasks, .initial_counters = spec.initial_counters, .successors = spec.successors };
        const plan = try b.planFor(config, spec.arena_bytes, spec.blocks);
        try b.uploaded.ensureUnusedCapacity(b.allocator, 1);
        const p = try b.allocator.create(Prepared);
        errdefer b.allocator.destroy(p);
        p.* = .{ .u = try mk.Uploaded.init(b.allocator, b.d, g, spec.ops, spec.blocks, spec.payloads, b.timeout), .plan = plan, .config = config };
        b.uploaded.appendAssumeCapacity(p);
        return p;
    }

    fn free(ctx: *anyopaque, plan: *anyopaque) void {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        const p: *Prepared = @ptrCast(@alignCast(plan));
        for (b.uploaded.items, 0..) |item, i| if (item == p) {
            _ = b.uploaded.orderedRemove(i);
            break;
        };
        p.u.deinit();
        b.allocator.destroy(p);
    }

    /// Payload arena refresh (the bytes are the store kernel's parameter), then reset + persistent run.
    fn run(ctx: *anyopaque, plan: *anyopaque, image: []const u8, stream: cuda.Stream) anyerror!void {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        const p: *Prepared = @ptrCast(@alignCast(plan));
        const u = &p.u;
        if (image.len > u.payload_bytes) return error.MkPayloadOverflow;
        var at: usize = 0;
        while (at < image.len) : (at += StoreChunk.store_bytes) {
            const n = @min(StoreChunk.store_bytes, image.len - at);
            var chunk: StoreChunk = .{ .dst = u.payload_base + at, .len = @intCast(n), .bytes = undefined };
            @memcpy(chunk.bytes[0..n], image[at..][0..n]);
            @memset(chunk.bytes[n..], 0);
            var args: cuda.Args = .{};
            args.add(chunk);
            try cuda.launch.launch(b.store, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &args);
        }
        var state = u.state;
        state.trace = b.trace;
        if (p.config.chained()) {
            state.sm_words = b.sm_words.ptr;
            state.chain_sms = b.sms;
            state.chain_slots = p.plan.persistent_blocks / b.sms;
            state.chain_map = p.config.map;
        }
        try mk.launchWith(b.fnsOf(p.config), stream, state, state.task_count, p.plan);
    }

    /// The recorded launch as the call site issued it: same function, grid, block, shared bytes and parameters.
    fn launchOne(ctx: *anyopaque, task: *const Recorded, stream: cuda.Stream) anyerror!void {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        var ptrs: [cuda.Args.max_args]?*anyopaque = undefined;
        const params = task.payload[@sizeOf(Header)..];
        for (task.entry.offsets, 0..) |off, i| ptrs[i] = @ptrCast(@constCast(&params[off]));
        const g = task.grid;
        const k = task.block;
        try b.d.check(b.d.api.cuLaunchKernel(task.function, g.x, g.y, g.z, k.x, k.y, k.z, task.shared, stream.handle, &ptrs, null), "cuLaunchKernel (MK8 recorded launch)");
    }

    fn capturing(_: *anyopaque, stream: cuda.Stream) anyerror!bool {
        return (try cuda.graph.captureStatus(stream)) != .none;
    }

    const Timer = struct { start: cuda.Event, stop: cuda.Event };

    fn timerNew(ctx: *anyopaque) anyerror!*anyopaque {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        const t = try b.allocator.create(Timer);
        errdefer b.allocator.destroy(t);
        t.start = try cuda.Event.init(b.d, true);
        errdefer t.start.deinit();
        t.stop = try cuda.Event.init(b.d, true);
        return t;
    }

    fn timerMark(_: *anyopaque, timer: *anyopaque, stream: cuda.Stream, stop: bool) anyerror!void {
        const t: *Timer = @ptrCast(@alignCast(timer));
        try (if (stop) t.stop else t.start).record(stream);
    }

    fn timerRead(_: *anyopaque, timer: *anyopaque) anyerror!f32 {
        const t: *Timer = @ptrCast(@alignCast(timer));
        try t.stop.synchronize();
        return try cuda.Event.elapsedMs(t.start, t.stop) * 1000;
    }

    fn timerFree(ctx: *anyopaque, timer: *anyopaque) void {
        const b: *GpuBackend = @ptrCast(@alignCast(ctx));
        const t: *Timer = @ptrCast(@alignCast(timer));
        t.start.deinit();
        t.stop.deinit();
        b.allocator.destroy(t);
    }

    /// True when the last chain run's SM mapping did not hold and its CTAs used blockIdx.x (a host sync: tests and
    /// debug only, never inside a capture).
    pub fn smMapFellBack(b: *const GpuBackend) !bool {
        var flag: u32 = 0;
        try b.sm_words.download((mk.sm_words_len - 1) * 4, std.mem.asBytes(&flag));
        return flag != 0;
    }

    /// Every plan's last error word (host syncs: tests and debug only, never inside a capture).
    pub fn check(b: *const GpuBackend) !mk.ErrorCode {
        for (b.uploaded.items) |p| {
            const code = try p.u.errorCode();
            if (code != .ok) return code;
        }
        return .ok;
    }
};

/// The sink Forward's decode step hangs on its stream: Deferral over the CUDA backend, heap-pinned.
pub const Executor = struct {
    gpu: GpuBackend,
    deferral: Deferral,

    /// The policy from TF_GLM53_MK_SHAPE / TF_GLM53_MK_CALIBRATE and the built-in device table.
    pub fn create(allocator: std.mem.Allocator, d: *const cuda.Driver) !*Executor {
        return createWith(allocator, d, try policy.Policy.fromEnv());
    }

    /// An explicit policy (tests: `.{ .override = shape }` pins one launch shape).
    pub fn createWith(allocator: std.mem.Allocator, d: *const cuda.Driver, p: policy.Policy) !*Executor {
        const x = try allocator.create(Executor);
        errdefer allocator.destroy(x);
        x.gpu = try GpuBackend.init(allocator, d);
        x.deferral = Deferral.init(allocator, x.gpu.backend());
        x.deferral.policy = p;
        x.deferral.device = x.gpu.device;
        return x;
    }

    pub fn destroy(x: *Executor) void {
        const allocator = x.deferral.allocator;
        x.deferral.deinit();
        x.gpu.deinit();
        allocator.destroy(x);
    }

    pub fn sink(x: *Executor) *const cuda.Stream.Deferred {
        return x.deferral.sink();
    }

    /// The fused plan a cache entry runs with (null for .plain), for tests that read its grid.
    pub fn prepared(_: *const Executor, p: *const Plan) ?*const Prepared {
        const h = p.handle orelse return null;
        return @ptrCast(@alignCast(h));
    }
};

const testing = std.testing;

/// Host test backend: logs every call, keeps the last spec and image.
const FakeBackend = struct {
    log: std.ArrayList(u8) = .empty,
    uploads: usize = 0,
    runs: usize = 0,
    launches: usize = 0,
    frees: usize = 0,
    timer_frees: usize = 0,
    capture: bool = false,
    image: std.ArrayList(u8) = .empty,
    counters: std.ArrayList(u32) = .empty,
    successors: std.ArrayList(u32) = .empty,
    blocks: std.ArrayList(u32) = .empty,
    ops: std.ArrayList(mk.Op) = .empty,
    launched: std.ArrayList(u8) = .empty,
    arena: u32 = 0,
    handles: [16]u8 = undefined,
    /// The fused shape each handle was uploaded for, and the order of uploads.
    handle_shapes: [16]LaunchShape = undefined,
    uploaded: std.ArrayList(LaunchShape) = .empty,
    /// Scripted calibration: microseconds per shape; the timer reports the shape run between its marks.
    times: [policy.shape_count]f32 = @splat(1),
    timer: u8 = 0,
    marked: ?LaunchShape = null,
    timed: bool = false,

    fn deinit(f: *FakeBackend) void {
        inline for (.{ "log", "image", "counters", "successors", "blocks", "ops", "launched", "uploaded" }) |name| @field(f, name).deinit(testing.allocator);
    }
    fn backend(f: *FakeBackend) Backend {
        return .{ .ctx = f, .upload_fn = upload, .run_fn = run, .launch_fn = launchOne, .capturing_fn = capturing, .free_fn = free, .timer_new_fn = timerNew, .timer_mark_fn = timerMark, .timer_read_fn = timerRead, .timer_free_fn = timerFree };
    }
    fn upload(ctx: *anyopaque, spec: *const PlanSpec, shape: LaunchShape) anyerror!*anyopaque {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        const a = testing.allocator;
        if (shape == .plain) return error.MkPlainHasNoPlan;
        try f.log.append(a, 'U');
        f.counters.clearRetainingCapacity();
        try f.counters.appendSlice(a, spec.initial_counters);
        f.successors.clearRetainingCapacity();
        try f.successors.appendSlice(a, spec.successors);
        f.blocks.clearRetainingCapacity();
        try f.blocks.appendSlice(a, spec.blocks);
        f.ops.clearRetainingCapacity();
        try f.ops.appendSlice(a, spec.ops);
        for (spec.tasks, 0..) |t, i| {
            if (t.dependency_count != @intFromBool(i > 0)) return error.NotAChain;
            if (t.successor_count != @intFromBool(i + 1 < spec.tasks.len)) return error.NotAChain;
            if (t.successor_count == 1 and spec.successors[t.successor_offset] != i + 1) return error.NotAChain;
        }
        f.arena = spec.arena_bytes;
        try f.uploaded.append(a, shape);
        f.uploads += 1;
        const slot = f.uploads % f.handles.len;
        f.handle_shapes[slot] = shape;
        return &f.handles[slot];
    }
    fn run(ctx: *anyopaque, plan: *anyopaque, image: []const u8, stream: cuda.Stream) anyerror!void {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        if (stream.deferred != null) return error.SinkNotStripped;
        try f.log.append(testing.allocator, 'R');
        f.image.clearRetainingCapacity();
        try f.image.appendSlice(testing.allocator, image);
        const slot = @intFromPtr(plan) - @intFromPtr(&f.handles[0]);
        if (f.timed) f.marked = f.handle_shapes[slot];
        f.runs += 1;
    }
    fn launchOne(ctx: *anyopaque, task: *const Recorded, _: cuda.Stream) anyerror!void {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        try f.log.append(testing.allocator, 'L');
        try f.launched.appendSlice(testing.allocator, task.payload[@sizeOf(Header)..][0..task.entry.param_bytes]);
        if (f.timed) f.marked = .plain;
        f.launches += 1;
    }
    fn capturing(ctx: *anyopaque, _: cuda.Stream) anyerror!bool {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        return f.capture;
    }
    fn free(ctx: *anyopaque, _: *anyopaque) void {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        f.frees += 1;
    }
    fn timerNew(ctx: *anyopaque) anyerror!*anyopaque {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        return &f.timer;
    }
    fn timerMark(ctx: *anyopaque, _: *anyopaque, _: cuda.Stream, stop: bool) anyerror!void {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        if (f.capture) return error.MarkInsideCapture;
        f.timed = !stop;
        if (!stop) f.marked = null;
    }
    fn timerRead(ctx: *anyopaque, _: *anyopaque) anyerror!f32 {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        if (f.capture) return error.ReadInsideCapture;
        try f.log.append(testing.allocator, 'T');
        return f.times[@backingInt(f.marked orelse return error.NothingTimed)];
    }
    fn timerFree(ctx: *anyopaque, _: *anyopaque) void {
        const f: *FakeBackend = @ptrCast(@alignCast(ctx));
        f.timer_frees += 1;
    }
};

fn fakeFunction(name: []const u8) cuda.Function {
    return .{ .d = undefined, .handle = null, .name = name };
}

/// Forward's rmsnorm call (glue.rmsnorm, 6144): grid rows, 8 warps.
fn rmsnormArgs(x: u64, out: u64) cuda.Args {
    var a: cuda.Args = .{};
    inline for (.{ x, @as(i64, 6144), @as(u64, 0x7000), out, @as(i64, 6144), @as(u64, 0), @as(f32, 1e-5), @as(c_int, 0) }) |v| a.add(v);
    return a;
}

/// Forward's packQpe call: grid (rows, heads), 32 threads.
fn qpeArgs(q: u64, qpe: u64, rows: i32) cuda.Args {
    var a: cuda.Args = .{};
    inline for (.{ q, @as(u64, 0x100), @as(u64, 0x200), qpe, rows, @as(i32, 32) }) |v| a.add(v);
    return a;
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(sig);
    inline for (.{
        sig.Rmsnorm6144, sig.QpePack,    sig.Attend,     sig.SwigluPlain, sig.LowrankPartial, sig.LowrankReduce,   sig.MoeResidualAdd,
        sig.KvWrite,     sig.Absorb,     sig.MlaMerge,   sig.ExpandV,     sig.LowrankCombine, sig.ResidualRmsnorm, sig.LowrankFinish,
        sig.EmbedPick,   sig.RouterPart, sig.RouterTopk, sig.SliceSum,    sig.IndexQuery,     sig.IndexKey,        sig.IndexScores,
        sig.HeadTop1,    sig.HeadFill,   sig.HeadPart,   sig.HeadMerge,   sig.HeadLse,        sig.HeadGather,      sig.HeadMass,
    }) |T| std.testing.refAllDecls(T);
    std.testing.refAllDecls(Deferral);
    std.testing.refAllDecls(GpuBackend);
    std.testing.refAllDecls(Executor);
    std.testing.refAllDecls(StoreChunk);
    _ = policy;
}

test "dispatch table: packed parameter bytes and payload sizes are the signature ABI behind a 16-byte header" {
    const want = [_]struct { []const u8, usize, usize, usize }{
        .{ "tf_glm_rmsnorm_6144", 8, 56, 72 },
        .{ "tf_glm53_qpe_pack", 6, 40, 56 },
        .{ "tf_glm53_attend", 18, 124, 144 },
        .{ "tf_glm53_kv_write", 13, 88, 104 },
        .{ "tf_glm53_absorb", 7, 44, 64 },
        .{ "tf_glm53_mla_merge", 7, 44, 64 },
        .{ "tf_glm53_expand_v", 7, 44, 64 },
        .{ "tf_glm53_lowrank_combine_bf16", 9, 56, 72 },
        .{ "tf_glm53_residual_rmsnorm_6144", 6, 40, 56 },
        .{ "tf_glm53_lowrank_finish", 7, 40, 56 },
        .{ "tf_glm53_embed_pick", 6, 36, 56 },
        .{ "tf_glm_router_part", 8, 60, 80 },
        .{ "tf_glm_router_tile", 8, 60, 80 },
        .{ "tf_glm_topk", 8, 48, 64 },
        .{ "tf_glm_slice_sum", 5, 32, 48 },
        .{ "tf_glm53_index_q", 4, 32, 48 },
        .{ "tf_glm53_index_key", 12, 76, 96 },
        .{ "tf_glm53_index_scores", 11, 64, 80 },
        .{ "tf_glm53_top1_f32", 5, 32, 48 },
        .{ "tf_glm53_cand_fill_f32", 3, 20, 40 },
        .{ "tf_glm53_cand_part_f32", 5, 32, 48 },
        .{ "tf_glm53_cand_merge_f32", 5, 32, 48 },
        .{ "tf_glm53_lse_f32", 3, 24, 40 },
        .{ "tf_glm53_gate_gather_f32", 7, 40, 56 },
        .{ "tf_glm53_gate_mass_f32", 6, 40, 56 },
        .{ "tf_glm53_swiglu_plain", 4, 24, 40 },
        .{ "tf_glm53_lowrank_partial", 9, 48, 64 },
        .{ "tf_glm53_lowrank_reduce", 4, 24, 40 },
        .{ "tf_glm53_moe_residual_add", 4, 28, 48 },
    };
    try testing.expectEqual(want.len, table.len);
    for (want) |w| {
        const e = lookup(w[0]).?;
        try testing.expectEqual(w[1], e.param_count);
        try testing.expectEqual(w[2], e.param_bytes);
        try testing.expectEqual(w[3], e.payload_bytes);
    }
    // The existing adapter mirrors in launch.zig carry the same byte sizes.
    try testing.expectEqual(@sizeOf(mk.AttnInputNormParams), lookup("tf_glm_rmsnorm_6144").?.payload_bytes);
    try testing.expectEqual(@sizeOf(mk.AttnQpePackParams), lookup("tf_glm53_qpe_pack").?.payload_bytes);
    try testing.expectEqual(@sizeOf(mk.AttnAttendParams), lookup("tf_glm53_attend").?.payload_bytes);
    // Forward's own argument packing lands on the mirror's offsets.
    const a = rmsnormArgs(1, 2);
    try testing.expectEqual(lookup("tf_glm_rmsnorm_6144").?.param_bytes, a.used);
    try testing.expectEqualSlices(u16, lookup("tf_glm_rmsnorm_6144").?.offsets, a.offsets[0..a.count]);
    try testing.expect(lookup("tf_glm53_topk") == null and lookup("tf_glm_rmsnorm_2048") == null);
}

test "dispatchable launches are recorded; other kernels, 2-D blocks, big shared memory and attributes are boundaries" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    var norm = rmsnormArgs(0x1000, 0x2000);
    const row: cuda.Config = .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 } };
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), row, stream, &norm));
    try testing.expectEqual(@as(usize, 1), s.pending());
    // Not in the table: the segment flushes, the caller launches it.
    var other: cuda.Args = .{};
    other.add(@as(u64, 5));
    try testing.expect(!try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_topk"), row, stream, &other));
    try testing.expectEqual(@as(usize, 0), s.pending());
    try testing.expectEqualStrings("UR", fake.log.items);
    // A table kernel launched with a 2-D block or more task shared memory than the arena is a boundary.
    try testing.expect(!try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 2 }, .block = .{ .x = 128, .y = 2 } }, stream, &norm));
    try testing.expect(!try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 }, .shared = mk.task_smem_bytes + 16 }, stream, &norm));
    try testing.expect(!try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 }, .pdl = true }, stream, &norm));
    try testing.expectEqual(@as(usize, 0), s.pending());
    try testing.expectEqualStrings("UR", fake.log.items); // empty flushes do nothing
    // A table symbol with a different argument list is a binding bug, not a boundary.
    // (Through the launch helper this surfaces as cuda error.Invalid with the reason logged.)
    try testing.expectError(error.MkSignatureMismatch, s.record(&.{ .name = "tf_glm_rmsnorm_6144", .grid = .{ .x = 2 }, .block = .{ .x = 256 }, .shared = 0, .params = norm.storage[0..8], .param_count = 1 }));
    try testing.expectError(error.MkSignatureMismatch, s.record(&.{ .name = "tf_glm53_qpe_pack", .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .shared = 0, .params = norm.storage[0..40], .param_count = 6, .offsets = &.{ 0, 8, 16, 24, 32, 40 } }));
}

test "flush runs the segment first, in recording order, as a linear chain; payload = header + recorded bytes" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    var norm = rmsnormArgs(0x1000, 0x2000);
    var qpe = qpeArgs(0x3000, 0x4000, 4);
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, stream, &norm));
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_qpe_pack"), .{ .grid = .{ .x = 4, .y = 32 }, .block = .{ .x = 32 } }, stream, &qpe));
    try testing.expectEqual(@as(usize, 0), fake.runs);
    // An event, copy or collective calls flushDeferred before it reaches the driver.
    try stream.flushDeferred();
    try testing.expectEqualStrings("UR", fake.log.items);
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, fake.counters.items);
    try testing.expectEqualSlices(u32, &.{1}, fake.successors.items);
    try testing.expectEqualSlices(u32, &.{ 4, 4 * 32 }, fake.blocks.items);
    try testing.expectEqualSlices(mk.Op, &.{ .attn_input_norm, .attn_qpe_pack }, fake.ops.items);
    // Task 0 at 0 (72 B), task 1 at the next 16-byte boundary (80); each = header, then exactly the packed args.
    const img = fake.image.items;
    try testing.expectEqual(@as(usize, 80 + 56), img.len);
    const h0 = std.mem.bytesToValue(Header, img[0..16]);
    try testing.expectEqual(Header{ .gx = 4, .gy = 1, .gz = 1, .threads = 256 }, h0);
    try testing.expectEqualSlices(u8, norm.storage[0..norm.used], img[16..][0..norm.used]);
    const h1 = std.mem.bytesToValue(Header, img[80..96]);
    try testing.expectEqual(Header{ .gx = 4, .gy = 32, .gz = 1, .threads = 32 }, h1);
    try testing.expectEqualSlices(u8, qpe.storage[0..qpe.used], img[96..][0..qpe.used]);
    const p1 = std.mem.bytesToValue(Payload(sig.QpePack), img[80..][0..56]);
    try testing.expectEqual(@as(u64, 0x4000), p1.p.qpe);
    try testing.expectEqual(@as(i32, 32), p1.p.local_heads);
    try testing.expectEqual(@as(u64, 1), s.stats.segments);
    try testing.expectEqual(@as(u64, 2), s.stats.tasks);
}

test "plan cache: same ops and grids reuse the plan with refreshed payload bytes; a new grid uploads a new plan" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    const cfg: cuda.Config = .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } };
    for ([_]u64{ 0x1000, 0x9000 }) |x| {
        var norm = rmsnormArgs(x, x + 0x100);
        try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), cfg, stream, &norm));
        try stream.flushDeferred();
        try testing.expectEqual(x, std.mem.bytesToValue(Payload(sig.Rmsnorm6144), fake.image.items[0..72]).p.x);
    }
    try testing.expectEqual(@as(usize, 1), fake.uploads);
    try testing.expectEqual(@as(usize, 2), fake.runs);
    try testing.expectEqual(@as(u64, 1), s.stats.hits);
    var norm = rmsnormArgs(0x1000, 0x2000);
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 3 }, .block = .{ .x = 256 } }, stream, &norm));
    try stream.flushDeferred();
    try testing.expectEqual(@as(usize, 2), fake.uploads);
    try testing.expectEqual(@as(u64, 2), s.stats.plans);
    try testing.expectEqualStrings("URRUR", fake.log.items);
}

test "a capture that misses the plan cache launches the recorded kernels unchanged; a cached plan runs fused" {
    var fake: FakeBackend = .{ .capture = true };
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    var norm = rmsnormArgs(0x1000, 0x2000);
    var qpe = qpeArgs(0x3000, 0x4000, 1);
    const seq = struct {
        fn record(st: cuda.Stream, n: *cuda.Args, q: *cuda.Args) !void {
            try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, st, n));
            try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_qpe_pack"), .{ .grid = .{ .x = 1, .y = 32 }, .block = .{ .x = 32 } }, st, q));
            try st.flushDeferred();
        }
    };
    try seq.record(stream, &norm, &qpe);
    try testing.expectEqualStrings("LL", fake.log.items);
    try testing.expectEqualSlices(u8, norm.storage[0..norm.used], fake.launched.items[0..norm.used]);
    try testing.expectEqualSlices(u8, qpe.storage[0..qpe.used], fake.launched.items[norm.used..]);
    fake.capture = false; // the eager warm-up uploads the plan
    try seq.record(stream, &norm, &qpe);
    fake.capture = true; // the capture now hits it
    try seq.record(stream, &norm, &qpe);
    try testing.expectEqualStrings("LLURR", fake.log.items);
    try testing.expectEqual(@as(u64, 1), s.stats.fallback_segments);
    try testing.expectEqual(@as(u64, 2), s.stats.segments);
}

test "launch shape: the arena is the plan's largest recorded dynamic shared memory; the grid never exceeds the widest task" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    var norm = rmsnormArgs(0x1000, 0x2000);
    var qpe = qpeArgs(0x3000, 0x4000, 4);
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 }, .shared = 1024 }, stream, &norm));
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_qpe_pack"), .{ .grid = .{ .x = 4, .y = 32 }, .block = .{ .x = 32 } }, stream, &qpe));
    try stream.flushDeferred();
    try testing.expectEqual(@as(u32, 1024), fake.arena);
    // A different arena is a different launch shape, so it is a separate plan.
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, stream, &norm));
    try stream.flushDeferred();
    try testing.expectEqual(@as(u32, 0), fake.arena);
    try testing.expectEqual(@as(usize, 2), fake.uploads);
    try testing.expectEqual(@as(u32, 128), persistentBlocks(70, 4, &.{ 1, 128, 24 }));
    try testing.expectEqual(@as(u32, 70), persistentBlocks(70, 1, &.{ 4, 128 }));
    try testing.expectEqual(@as(u32, 4), persistentBlocks(48, 2, &.{4}));
    try testing.expectEqual(@as(u32, 1), persistentBlocks(48, 0, &.{}));
}

/// One rmsnorm + qpe segment at `rows` (both launched one block row per token), flushed.
fn normQpe(stream: cuda.Stream, rows: u32, x: u64) !void {
    var norm = rmsnormArgs(x, x + 0x100);
    var qpe = qpeArgs(0x3000, 0x4000, @intCast(rows));
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &norm));
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_qpe_pack"), .{ .grid = .{ .x = rows, .y = 32 }, .block = .{ .x = 32 } }, stream, &qpe));
    try stream.flushDeferred();
}

test "policy: the GB10 table runs rows=2 as plain launches and rows=1 fused; an override pins one shape" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    s.device = .{ .sms = 48, .cc = 121 };
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    try normQpe(stream, 2, 0x1000);
    try testing.expectEqualStrings("LL", fake.log.items); // no upload: the recorded launches, unchanged
    try testing.expectEqual(@as(usize, 0), s.pending());
    const p2 = s.planList()[0];
    try testing.expectEqual(LaunchShape.plain, p2.shape);
    try testing.expectEqual(policy.Source.table, p2.source);
    try testing.expectEqual(@as(?*anyopaque, null), p2.handle);
    // Cached .plain inside a capture: still the recorded launches (graph nodes), no fallback.
    fake.capture = true;
    try normQpe(stream, 2, 0x9000);
    fake.capture = false;
    try testing.expectEqualStrings("LLLL", fake.log.items);
    try testing.expectEqual(@as(u64, 0), s.stats.fallback_segments);
    try testing.expectEqual(@as(u64, 2), s.stats.plain_segments);
    try testing.expectEqual(@as(u64, 2), s.stats.segments);
    try testing.expectEqual(@as(u64, 1), s.stats.hits);
    // rows=1: chain-sm, rows=4: chain-block, rows=8: queue-occ.
    for ([_]struct { u32, LaunchShape }{ .{ 1, .chain_sm }, .{ 4, .chain_block }, .{ 8, .queue_occ } }) |c| {
        try normQpe(stream, c[0], 0x1000);
        const p = s.planList()[s.planList().len - 1];
        try testing.expectEqual(c[1], p.shape);
        try testing.expectEqual(c[1], fake.uploaded.items[fake.uploaded.items.len - 1]);
    }
    try testing.expectEqualStrings("LLLLURURUR", fake.log.items);
    // The override wins over the table.
    var o = Deferral.init(testing.allocator, fake.backend());
    defer o.deinit();
    o.device = s.device;
    o.policy = .{ .override = .queue };
    const os: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = o.sink() };
    try normQpe(os, 2, 0x1000);
    try testing.expectEqual(LaunchShape.queue, o.planList()[0].shape);
    try testing.expectEqual(policy.Source.override, o.planList()[0].source);
    try testing.expectEqual(LaunchShape.queue, fake.uploaded.items[fake.uploaded.items.len - 1]);
}

test "rows: from the segment's per-token kernels, else the caller's hint; the bucket is part of the plan key" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    s.device = .{ .sms = 48, .cc = 121 };
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    // A swiglu-only segment does not show its rows.
    var act: cuda.Args = .{};
    inline for (.{ @as(u64, 0x1000), @as(u64, 0x2000), @as(c_int, 2), @as(c_int, 3072) }) |v| act.add(v);
    const cfg: cuda.Config = .{ .grid = .{ .x = 24 }, .block = .{ .x = 256 } };
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_swiglu_plain"), cfg, stream, &act));
    try testing.expectEqual(@as(u32, 0), s.segmentRows());
    s.rows = 2;
    try testing.expectEqual(@as(u32, 2), s.segmentRows());
    try stream.flushDeferred();
    try testing.expectEqual(LaunchShape.plain, s.planList()[0].shape);
    // Same launches under another row count: a separate key with its own choice.
    s.rows = 8;
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_swiglu_plain"), cfg, stream, &act));
    try stream.flushDeferred();
    try testing.expectEqual(@as(usize, 2), s.planList().len);
    try testing.expectEqual(LaunchShape.queue_occ, s.planList()[1].shape);
    // Unknown rows: no table bucket, the default shape.
    s.rows = 0;
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm53_swiglu_plain"), cfg, stream, &act));
    try stream.flushDeferred();
    try testing.expectEqual(policy.default_shape, s.planList()[2].shape);
    try testing.expectEqual(policy.Source.default, s.planList()[2].source);
    // The kernels' own rows win over the hint.
    s.rows = 8;
    var norm = rmsnormArgs(1, 2);
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &norm));
    try testing.expectEqual(@as(u32, 1), s.segmentRows());
    try stream.flushDeferred();
}

test "calibration: each shape timed over eager windows, the fastest cached for the key; same bytes every window" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    // plain 10, queue 20, queue-occ 30, chain-block 5, chain-sm 7 (us)
    fake.times = .{ 10, 20, 30, 5, 7 };
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    s.device = .{ .sms = 84, .cc = 89 }; // not in the table
    s.policy = .{ .calibrate = .on, .windows = 2 };
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    try normQpe(stream, 1, 0x1000);
    // One plan per fused shape, uploaded up front; window 1 runs plain.
    try testing.expectEqualSlices(LaunchShape, &.{ .queue, .queue_occ, .chain_block, .chain_sm }, fake.uploaded.items);
    try testing.expectEqualStrings("UUUULL", fake.log.items);
    try testing.expect(s.calibrating());
    const want = try testing.allocator.dupe(u8, fake.launched.items);
    defer testing.allocator.free(want);
    var windows: usize = 1;
    while (s.calibrating()) : (windows += 1) {
        try normQpe(stream, 1, 0x1000);
        // Every fused window refreshes the same payload bytes the plain launches passed.
        if (fake.log.items[fake.log.items.len - 1] == 'R') try testing.expectEqualSlices(u8, want[0..56], fake.image.items[16..][0..56]);
        try testing.expect(windows < 20);
    }
    // 2 windows x 5 shapes, then the flush that reads the last window and settles.
    try testing.expectEqual(@as(usize, 11), windows);
    try testing.expectEqual(@as(u64, 10), s.stats.calibration_windows);
    try testing.expectEqual(@as(u64, 1), s.stats.calibrated);
    const p = s.planList()[0];
    try testing.expectEqual(LaunchShape.chain_block, p.shape);
    try testing.expectEqual(policy.Source.calibration, p.source);
    try testing.expectEqual(@as(usize, 3), fake.frees); // the other fused plans
    try testing.expectEqual(@as(usize, 1), fake.timer_frees);
    try testing.expectEqual(@as(usize, 4), fake.uploads);
    // Cached: the next window runs chain-block with no timing, also inside a capture.
    const log_len = fake.log.items.len;
    fake.capture = true;
    try normQpe(stream, 1, 0x1000);
    fake.capture = false;
    try testing.expectEqualStrings("R", fake.log.items[log_len..]);
    try testing.expectEqual(@as(u64, 0), s.stats.fallback_segments);
    // A key that the table knows is not calibrated unless forced.
    var t = Deferral.init(testing.allocator, fake.backend());
    defer t.deinit();
    t.device = .{ .sms = 48, .cc = 121 };
    t.policy = .{ .calibrate = .on };
    const ts: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = t.sink() };
    try normQpe(ts, 2, 0x1000);
    try testing.expect(!t.calibrating());
    t.policy = .{ .calibrate = .force };
    try normQpe(ts, 4, 0x1000);
    try testing.expect(t.calibrating());
}

test "calibration never runs inside a capture: the choice is fixed first, dropped plans freed after the capture" {
    var fake: FakeBackend = .{};
    defer fake.deinit();
    fake.times = .{ 10, 8, 30, 5, 7 };
    var s = Deferral.init(testing.allocator, fake.backend());
    defer s.deinit();
    s.policy = .{ .calibrate = .on };
    const stream: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = s.sink() };
    // Windows 1-3 run plain, queue, queue-occ; windows 1 and 2 are read back by the next eager flushes.
    for (0..3) |_| try normQpe(stream, 1, 0x1000);
    try testing.expectEqual(@as(u64, 2), s.stats.calibration_windows);
    // The capture settles on the best measured so far (queue, 8 us) without reading a timer or freeing a plan.
    fake.capture = true;
    const log_len = fake.log.items.len;
    try normQpe(stream, 1, 0x1000);
    try testing.expectEqualStrings("R", fake.log.items[log_len..]);
    try testing.expect(!s.calibrating());
    try testing.expectEqual(LaunchShape.queue, s.planList()[0].shape);
    try testing.expectEqual(@as(usize, 0), fake.frees);
    try testing.expectEqual(@as(usize, 0), fake.timer_frees);
    // A capture that meets a key with no measured window yet runs the table/default shape.
    var cold = rmsnormArgs(1, 2);
    try testing.expect(try cuda.launch.deferOrFlush(fakeFunction("tf_glm_rmsnorm_6144"), .{ .grid = .{ .x = 3 }, .block = .{ .x = 256 } }, stream, &cold));
    try stream.flushDeferred(); // a cold key in a capture: the recorded launches, no plan
    try testing.expectEqual(@as(u64, 1), s.stats.fallback_segments);
    fake.capture = false;
    // The next eager flush frees what the capture dropped (3 plans and the timer).
    try normQpe(stream, 1, 0x1000);
    try testing.expectEqual(@as(usize, 3), fake.frees);
    try testing.expectEqual(@as(usize, 1), fake.timer_frees);
    // A fresh calibrating key whose first window is pending when a capture arrives: nothing measured yet.
    var c = Deferral.init(testing.allocator, fake.backend());
    defer c.deinit();
    c.device = .{ .sms = 48, .cc = 121 };
    c.policy = .{ .calibrate = .force };
    const cs: cuda.Stream = .{ .d = undefined, .handle = null, .deferred = c.sink() };
    try normQpe(cs, 4, 0x1000);
    fake.capture = true;
    try normQpe(cs, 4, 0x1000);
    fake.capture = false;
    try testing.expectEqual(LaunchShape.chain_block, c.planList()[0].shape); // the GB10 table's rows=4 entry
    try testing.expectEqual(policy.Source.table, c.planList()[0].source);
}
