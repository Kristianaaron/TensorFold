// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host ABI, upload plan, and CUDA graph launch for the GLM-5.3 persistent scheduler.
const std = @import("std");
const cuda = @import("cuda");
pub const graph_api = @import("graph.zig");
const graph = graph_api;

/// Largest dynamic task allocation currently admitted by the MK task set (qmmf: 49,152 B).
pub const task_smem_bytes: u32 = 49_152;
pub const block_threads: u32 = 256;
pub const sm121_optin_smem_bytes: u32 = 101376;
pub const scheduler_static_smem_bytes: u32 = 8;
pub const ErrorCode = enum(u32) { ok = 0, timeout = 1, bad_task = 2, unsupported = 3 };
pub const Op = enum(u8) {
    swiglu = 0,
    lowrank_partial = 1,
    lowrank_reduce = 2,
    add_partial = 3,
    lowrank_combine_bf16 = 4,
    residual_rmsnorm_6144 = 5,
    moe_residual_add = 6,
    lowrank_finish = 7,
    embed_pick = 8,
    // Group ids mirror tf_glm53_mk_op in mk_runtime.cuh.
    attn_input_norm = 9,
    attn_qpe_pack = 10,
    attn_attend = 11,
    moe_post_norm = 12,
    moe_router_part = 13,
    moe_router_topk = 14,
    moe_lowrank_partial = 15,
    moe_lowrank_reduce = 16,
    moe_lowrank_finish = 17,
    moe_residual_add_group = 18,
    moe_swiglu = 19,
    head_final_norm = 20,
    head_top1_f32 = 21,
    head_candidate_fill_f32 = 22,
    head_candidate_part_f32 = 23,
    head_candidate_merge_f32 = 24,
    head_lse_f32 = 25,
    head_gate_gather_f32 = 26,
    head_gate_mass_f32 = 27,
    head_embed_pick = 28,
    index_query = 29,
    index_key = 30,
    index_scores = 31,
    attn_kv_write = 32,
    attn_absorb = 33,
    attn_mla_merge = 34,
    attn_expand_v = 35,
    moe_router_slice_sum = 36,
    attn_q_a_norm = 37,
    exl3_rot_in = 38,
    exl3_down_combine = 39,
    qmmf_partial_task = 40,
    qmmf_reduce_task = 41,
};

comptime {
    std.debug.assert(@backingInt(Op.swiglu) == 0);
    std.debug.assert(@backingInt(Op.embed_pick) == 8);
    std.debug.assert(@backingInt(Op.attn_input_norm) == 9);
    std.debug.assert(@backingInt(Op.moe_residual_add_group) == 18);
    std.debug.assert(@backingInt(Op.head_embed_pick) == 28);
    std.debug.assert(@backingInt(Op.index_query) == 29);
    std.debug.assert(@backingInt(Op.index_key) == 30);
    std.debug.assert(@backingInt(Op.index_scores) == 31);
    std.debug.assert(@backingInt(Op.attn_kv_write) == 32);
    std.debug.assert(@backingInt(Op.attn_absorb) == 33);
    std.debug.assert(@backingInt(Op.attn_mla_merge) == 34);
    std.debug.assert(@backingInt(Op.attn_expand_v) == 35);
    std.debug.assert(@backingInt(Op.moe_router_slice_sum) == 36);
    std.debug.assert(@backingInt(Op.attn_q_a_norm) == 37);
    std.debug.assert(@backingInt(Op.exl3_rot_in) == 38);
    std.debug.assert(@backingInt(Op.exl3_down_combine) == 39);
    std.debug.assert(@backingInt(Op.qmmf_partial_task) == 40);
    std.debug.assert(@backingInt(Op.qmmf_reduce_task) == 41);
}

pub const attn_base: u8 = @backingInt(Op.attn_input_norm);
pub const moe_base: u8 = @backingInt(Op.moe_post_norm);
pub const head_base: u8 = @backingInt(Op.head_final_norm);
pub const index_base: u8 = @backingInt(Op.index_query);

/// ABI mirrors of mk_dispatch_attn.cuh. CUDA static_asserts pin these byte sizes and offsets.
pub const AttnInputNormParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    x: u64,
    xs: i64,
    weight: u64,
    out: u64,
    os: i64,
    sums: u64,
    eps: f32,
    do_sums: i32,
};
pub const AttnQANormParams = AttnInputNormParams;
pub const AttnQpePackParams = extern struct {
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
pub const AttnAttendParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
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
pub const AttnKvWriteParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    proj: u64,
    norm: u64,
    cos: u64,
    sin: u64,
    positions: u64,
    latent: u64,
    rope: u64,
    stage_latent: u64,
    stage_rope: u64,
    rows: i32,
    stride: i32,
    rank: i32,
    world: i32,
};
pub const AttnAbsorbParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    q: u64,
    kvb_codes: u64,
    kvb_scales: u64,
    qa: u64,
    rows: i32,
    local_heads: i32,
    head0: i32,
};
pub const AttnMlaMergeParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    local: u64,
    peer: u64,
    out: u64,
    lse_out: u64,
    rows: i32,
    local_heads: i32,
    rank: i32,
};
pub const AttnExpandVParams = extern struct {
    gx: u32,
    gy: u32,
    gz: u32,
    threads: u32,
    ol: u64,
    kvb_codes: u64,
    kvb_scales: u64,
    out: u64,
    rows: i32,
    local_heads: i32,
    head0: i32,
};
comptime {
    std.debug.assert(@sizeOf(AttnInputNormParams) == 72 and @offsetOf(AttnInputNormParams, "eps") == 64);
    std.debug.assert(@sizeOf(AttnQpePackParams) == 56 and @offsetOf(AttnQpePackParams, "rows") == 48);
    std.debug.assert(@sizeOf(AttnAttendParams) == 144 and @offsetOf(AttnAttendParams, "rows") == 120);
    std.debug.assert(@sizeOf(AttnKvWriteParams) == 104 and @offsetOf(AttnKvWriteParams, "rows") == 88);
    std.debug.assert(@sizeOf(AttnAbsorbParams) == 64 and @offsetOf(AttnAbsorbParams, "rows") == 48);
    std.debug.assert(@sizeOf(AttnMlaMergeParams) == 64 and @offsetOf(AttnMlaMergeParams, "rows") == 48);
    std.debug.assert(@sizeOf(AttnExpandVParams) == 64 and @offsetOf(AttnExpandVParams, "rows") == 48);
}

pub const DispatchKind = enum {
    residual_rmsnorm_6144,
    rmsnorm_6144,
    rmsnorm_2048,
    kv_write,
    topk,
    index_key,
    index_scores_oracle,
    index_scores,
    merge,
    top1_f32,
    cand_part_f32,
    cand_merge_f32,
    lse_f32,
    router_tile,
    swiglu_64,
    exl3_grouped,
    exl3_group,
    exl3_down_combine,
    exl3_rot_in,
    exl3_linear,
    absorb,
    mla_merge,
    expand_v,
    router_slice_sum,
};

pub const DispatchSpec = struct { kind: DispatchKind, threads: u16, static_smem: u32 };

/// These are the task entries admitted by the group dispatchers. Shared arrays
/// of 4 KiB or more are carved from task_smem_bytes and do not count as static.
pub const dispatched_specs = [_]DispatchSpec{
    .{ .kind = .residual_rmsnorm_6144, .threads = 256, .static_smem = 32 }, // utility op 5
    .{ .kind = .rmsnorm_6144, .threads = 256, .static_smem = 32 }, // attention input norm
    .{ .kind = .residual_rmsnorm_6144, .threads = 256, .static_smem = 32 }, // MoE post norm
    .{ .kind = .router_tile, .threads = 128, .static_smem = 0 }, // tile buffers come from the task arena
    .{ .kind = .rmsnorm_6144, .threads = 256, .static_smem = 32 }, // head final norm
    .{ .kind = .top1_f32, .threads = 256, .static_smem = 2048 },
    .{ .kind = .cand_part_f32, .threads = 256, .static_smem = 0 },
    .{ .kind = .cand_merge_f32, .threads = 256, .static_smem = 0 },
    .{ .kind = .lse_f32, .threads = 256, .static_smem = 1024 },
    .{ .kind = .index_key, .threads = 128, .static_smem = 1024 },
    .{ .kind = .index_scores, .threads = 256, .static_smem = 0 },
    .{ .kind = .kv_write, .threads = 128, .static_smem = 16 },
    .{ .kind = .absorb, .threads = 128, .static_smem = 0 },
    .{ .kind = .mla_merge, .threads = 128, .static_smem = 0 },
    .{ .kind = .expand_v, .threads = 128, .static_smem = 0 },
    .{ .kind = .router_slice_sum, .threads = 256, .static_smem = 0 },
    .{ .kind = .exl3_down_combine, .threads = 256, .static_smem = 0 },
    .{ .kind = .rmsnorm_2048, .threads = 256, .static_smem = 16 },
    .{ .kind = .exl3_rot_in, .threads = 128, .static_smem = 0 },
    // EXL3 grouped (32,832 B) is omitted: adding it with down_combine and the
    // existing head/index/router set would exceed the 101,376 B arena budget.
    // qpe_pack/attend/candidate_fill/gate gather/mass/embed/absorb/merge/expand have no static arrays.
};

pub fn dispatchedStaticSmem() u32 {
    var total: u32 = task_smem_bytes + scheduler_static_smem_bytes;
    for (dispatched_specs) |spec| total += spec.static_smem;
    return total;
}

pub fn isDispatchable(threads: u32, dynamic_smem: u32, uses_cluster: bool, uses_grid_sync: bool) bool {
    return threads > 0 and threads <= block_threads and threads % 32 == 0 and
        dynamic_smem <= task_smem_bytes and !uses_cluster and !uses_grid_sync;
}

/// CUDA's explicit padding keeps this record byte-compatible with graph.Task (16 bytes).
pub const TaskRecord = extern struct {
    successor_offset: u32,
    layer: u16,
    aux: u16,
    dependency_count: u16,
    successor_count: u16,
    kind: u8,
    row: u8,
    reserved: u16 = 0,

    pub fn fromTask(task: graph.Task, op: Op) TaskRecord {
        return .{
            .kind = @backingInt(task.kind),
            .layer = task.layer,
            .row = task.row,
            .aux = task.aux,
            .dependency_count = task.dependency_count,
            .successor_offset = task.successor_offset,
            .successor_count = task.successor_count,
            .reserved = @backingInt(op),
        };
    }
};

pub const RuntimeState = extern struct {
    tasks: u64,
    counters: u64,
    initial_counters: u64,
    successors: u64,
    params: u64,
    blocks: u64,
    next_block: u64,
    finished_blocks: u64,
    active: u64,
    ready: u64,
    ready_head: u64,
    ready_tail: u64,
    ready_lock: u64,
    completed: u64,
    error_word: u64,
    task_count: u32,
    edge_count: u32,
    ready_capacity: u32,
    timeout_cycles: u32,
    /// Test-only per-task timing buffer (mk_runtime.cuh tf_glm53_mk_state.trace); 0 in production.
    trace: u64 = 0,
    /// Chain scheduler only (tf_glm53_mk_run_chain): sm_words_len words the reset kernel zeroes, the SM count and
    /// CTAs per SM the grid was sized for, and the worker mapping.
    sm_words: u64 = 0,
    chain_sms: u32 = 0,
    chain_slots: u32 = 0,
    chain_map: ChainMap = .block,
    reserved: u32 = 0,
};

/// mk_runtime.cuh tf_glm53_mk_map: how a chain-scheduler CTA picks its worker index.
pub const ChainMap = enum(u32) {
    /// worker = blockIdx.x
    block = 0,
    /// worker = (the CTA's slot on its SM) * chain_sms + dense SM index: block b and b + chain_sms share an SM.
    /// Falls back to blockIdx.x for the whole run when any SM holds more than chain_slots of the grid's CTAs.
    sm = 1,
};

/// TF_GLM53_MK_SM_WORDS: the chain scheduler's per-SM scratch.
pub const sm_words_len: u32 = 2051;

comptime {
    std.debug.assert(@sizeOf(RuntimeState) == 168 and @offsetOf(RuntimeState, "trace") == 136);
    std.debug.assert(@offsetOf(RuntimeState, "sm_words") == 144 and @offsetOf(RuntimeState, "chain_map") == 160);
}

/// Words per task in a trace buffer: {first block start, last block end, summed block ns, ready time} (ns).
pub const trace_words: usize = 4;

/// Trace buffer length in u64 words for `task_count` tasks (one more record holds the run's first entry and last exit).
pub fn traceLen(task_count: usize) usize {
    return (task_count + 1) * trace_words;
}

/// The initial trace image: starts at max (atomicMin), everything else 0.
pub fn traceInit(out: []u64) void {
    for (out, 0..) |*w, i| w.* = if (i % trace_words == 0) std.math.maxInt(u64) else 0;
}

pub const Plan = struct {
    persistent_blocks: u32,
    dynamic_smem: u32,

    pub fn init(sm_count: u32, blocks_per_sm: u32, max_task_smem: u32) error{Invalid}!Plan {
        if (sm_count == 0 or blocks_per_sm == 0 or
            dispatchedStaticSmem() - task_smem_bytes + max_task_smem > sm121_optin_smem_bytes) return error.Invalid;
        return .{ .persistent_blocks = sm_count * blocks_per_sm, .dynamic_smem = max_task_smem };
    }
};

pub fn encodeTask(task: graph.Task, op: Op) TaskRecord {
    return TaskRecord.fromTask(task, op);
}

/// Serial host reference for the work-block queue and dependency release protocol.
pub fn simulate(allocator: std.mem.Allocator, g: graph.Graph, blocks: []const u32, timeout: usize) ![]u32 {
    if (blocks.len != g.tasks.len or g.initial_counters.len != g.tasks.len) return error.Invalid;
    const counters = try allocator.dupe(u32, g.initial_counters);
    defer allocator.free(counters);
    const finished = try allocator.alloc(u32, g.tasks.len);
    defer allocator.free(finished);
    @memset(finished, 0);
    var ready: std.ArrayList(u32) = .empty;
    defer ready.deinit(allocator);
    var order: std.ArrayList(u32) = .empty;
    errdefer order.deinit(allocator);
    for (blocks, 0..) |count, i| {
        if (count == 0) return error.Invalid;
        if (counters[i] == 0) for (0..count) |_| try ready.append(allocator, @intCast(i));
    }
    var cursor: usize = 0;
    var idle: usize = 0;
    var completed: usize = 0;
    while (completed < g.tasks.len) {
        if (cursor == ready.items.len) {
            if (idle >= timeout) return error.Timeout;
            idle += 1;
            continue;
        }
        idle = 0;
        const id: usize = ready.items[cursor];
        cursor += 1;
        if (id >= g.tasks.len or finished[id] >= blocks[id]) return error.BadTask;
        try order.append(allocator, @intCast(id));
        finished[id] += 1;
        if (finished[id] != blocks[id]) continue;
        completed += 1;
        const task = g.tasks[id];
        const start: usize = task.successor_offset;
        const end = start + task.successor_count;
        if (end > g.successors.len) return error.BadTask;
        for (g.successors[start..end]) |raw_successor| {
            const successor: usize = raw_successor;
            if (successor >= g.tasks.len or counters[successor] == 0) return error.BadTask;
            counters[successor] -= 1;
            if (counters[successor] == 0) for (0..blocks[successor]) |_| try ready.append(allocator, raw_successor);
        }
    }
    return order.toOwnedSlice(allocator);
}

/// The payload arena layout Uploaded uses: each task's bytes at the next 16-byte boundary, in task order.
/// Returns the arena length; `offsets[i]` receives task i's offset.
pub fn layoutPayloads(lengths: []const usize, offsets: []usize) !usize {
    if (offsets.len != lengths.len) return error.Invalid;
    var end: usize = 0;
    for (lengths, offsets) |len, *offset| {
        end = std.mem.alignForward(usize, end, 16);
        offset.* = end;
        end = try std.math.add(usize, end, len);
    }
    return end;
}

pub const Uploaded = struct {
    allocator: std.mem.Allocator,
    d: *const cuda.Driver,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    state: RuntimeState,
    error_buffer: cuda.DeviceBuffer,
    /// Device address and length of the payload arena (layoutPayloads), for per-step payload refreshes.
    payload_base: u64 = 0,
    payload_bytes: usize = 0,

    pub fn deinit(self: *Uploaded) void {
        for (self.buffers.items) |*buffer| buffer.free();
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }

    fn copy(self: *Uploaded, bytes: []const u8) !cuda.DeviceBuffer {
        var buffer = try cuda.DeviceBuffer.fromHost(self.d, bytes);
        errdefer buffer.free();
        try self.buffers.append(self.allocator, buffer);
        return buffer;
    }

    /// Upload fixed-shape DAG data and typed task payload bytes. Payloads are 16-byte aligned in one arena.
    pub fn init(allocator: std.mem.Allocator, d: *const cuda.Driver, g: graph.Graph, ops: []const Op, blocks: []const u32, payloads: []const []const u8, timeout: u32) !Uploaded {
        if (ops.len != g.tasks.len or blocks.len != g.tasks.len or payloads.len != g.tasks.len) return error.Invalid;
        var out: Uploaded = .{ .allocator = allocator, .d = d, .state = undefined, .error_buffer = undefined };
        errdefer out.deinit();
        var records = try allocator.alloc(TaskRecord, g.tasks.len);
        defer allocator.free(records);
        var params_host = try allocator.alloc(u64, g.tasks.len);
        defer allocator.free(params_host);
        const payload_offsets = try allocator.alloc(usize, g.tasks.len);
        defer allocator.free(payload_offsets);
        const payload_lengths = try allocator.alloc(usize, g.tasks.len);
        defer allocator.free(payload_lengths);
        for (payloads, payload_lengths) |bytes, *len| len.* = bytes.len;
        var payload_len = try layoutPayloads(payload_lengths, payload_offsets);
        // An empty arena still gets one aligned word, so the payload buffer has a device address.
        if (payload_len == 0) payload_len = 16;
        var payload_arena = try allocator.alloc(u8, payload_len);
        defer allocator.free(payload_arena);
        @memset(payload_arena, 0);
        for (g.tasks, 0..) |task, i| {
            records[i] = encodeTask(task, ops[i]);
            @memcpy(payload_arena[payload_offsets[i]..][0..payloads[i].len], payloads[i]);
        }
        var ready_capacity: usize = 0;
        for (blocks) |count| {
            if (count == 0) return error.Invalid;
            ready_capacity = try std.math.add(usize, ready_capacity, count);
        }
        if (ready_capacity > std.math.maxInt(u32)) return error.Invalid;
        const zero_words = try allocator.alloc(u32, g.tasks.len);
        defer allocator.free(zero_words);
        @memset(zero_words, 0);
        const ready_words = try allocator.alloc(u32, ready_capacity);
        defer allocator.free(ready_words);
        @memset(ready_words, 0);
        const task_buf = try out.copy(std.mem.sliceAsBytes(records));
        const counter_buf = try out.copy(std.mem.sliceAsBytes(g.initial_counters));
        const counters_zero_buf = try out.copy(std.mem.sliceAsBytes(zero_words));
        const successor_buf = try out.copy(std.mem.sliceAsBytes(g.successors));
        const blocks_buf = try out.copy(std.mem.sliceAsBytes(blocks));
        const payload_buf = try out.copy(payload_arena);
        for (payload_offsets, 0..) |offset, i| params_host[i] = payload_buf.ptr + offset;
        const params_buf = try out.copy(std.mem.sliceAsBytes(params_host));
        const next_buf = try out.copy(std.mem.sliceAsBytes(zero_words));
        const finished_buf = try out.copy(std.mem.sliceAsBytes(zero_words));
        const active_buf = try out.copy(std.mem.sliceAsBytes(zero_words));
        const ready_buf = try out.copy(std.mem.sliceAsBytes(ready_words));
        const scalar_buf = try out.copy(std.mem.sliceAsBytes(&[_]u32{0}));
        const error_buf = try out.copy(std.mem.sliceAsBytes(&[_]u32{0}));
        const state = RuntimeState{
            .tasks = task_buf.ptr,
            .counters = counters_zero_buf.ptr,
            .initial_counters = counter_buf.ptr,
            .successors = successor_buf.ptr,
            .params = params_buf.ptr,
            .blocks = blocks_buf.ptr,
            .next_block = next_buf.ptr,
            .finished_blocks = finished_buf.ptr,
            .active = active_buf.ptr,
            .ready = ready_buf.ptr,
            .ready_head = scalar_buf.ptr,
            .ready_tail = scalar_buf.ptr + 4,
            .ready_lock = scalar_buf.ptr + 8,
            .completed = scalar_buf.ptr + 12,
            .error_word = error_buf.ptr,
            .task_count = @intCast(g.tasks.len),
            .edge_count = @intCast(g.successors.len),
            .ready_capacity = @intCast(ready_capacity),
            .timeout_cycles = timeout,
        };
        out.state = state;
        out.error_buffer = error_buf;
        out.payload_base = payload_buf.ptr;
        out.payload_bytes = payload_len;
        return out;
    }

    /// The run's error word (a host sync; never call it inside a stream capture).
    pub fn errorCode(self: *const Uploaded) !ErrorCode {
        var code: u32 = 0;
        try self.error_buffer.download(0, std.mem.asBytes(&code));
        return std.enums.fromInt(ErrorCode, code) orelse error.Invalid;
    }

    pub fn run(self: *Uploaded, module: cuda.Module, stream: cuda.Stream, plan: Plan) !ErrorCode {
        const state = self.state;
        try launchStep(module, stream, state, state.task_count, plan);
        try stream.synchronize();
        var code: u32 = 0;
        try self.error_buffer.download(0, std.mem.asBytes(&code));
        return std.enums.fromInt(ErrorCode, code) orelse error.Invalid;
    }
};

/// The scheduler's two kernels, resolved once with the dynamic shared-memory opt-in done, so a launch inside a
/// stream capture issues kernel launches only.
pub const Functions = struct {
    reset: cuda.Function,
    run: cuda.Function,

    pub fn resolve(module: cuda.Module, plan: Plan) !Functions {
        return resolveNamed(module, plan, "tf_glm53_mk_run");
    }

    /// `run_symbol` picks the interpreter: tf_glm53_mk_run (every op) or tf_glm53_mk_run_deferred (the MK8 table's).
    pub fn resolveNamed(module: cuda.Module, plan: Plan, run_symbol: [:0]const u8) !Functions {
        const reset = try module.function("tf_glm53_mk_reset");
        const persistent = try module.function(run_symbol);
        // The 48 KiB default covers static + dynamic shared memory together: opt in whenever their sum exceeds it
        // (sm_120: a 48 KiB arena plus the scheduler's static state failed with CUDA_ERROR_INVALID_VALUE).
        const static_smem: u32 = @intCast(try persistent.attribute(.shared_size_bytes));
        if (plan.dynamic_smem + static_smem > 48 * 1024) try persistent.allowDynamicShared(plan.dynamic_smem);
        return .{ .reset = reset, .run = persistent };
    }
};

pub fn launchStep(module: cuda.Module, stream: cuda.Stream, state: RuntimeState, task_count: u32, plan: Plan) !void {
    try launchWith(try Functions.resolve(module, plan), stream, state, task_count, plan);
}

/// Reset then the persistent launch on `stream`: two kernel launches, no host sync (graph-capturable).
pub fn launchWith(fns: Functions, stream: cuda.Stream, state: RuntimeState, task_count: u32, plan: Plan) !void {
    const reset = fns.reset;
    const persistent = fns.run;
    var reset_args: cuda.Args = .{};
    inline for (.{
        state.tasks,           state.counters,   state.initial_counters, state.blocks,     state.next_block,
        state.finished_blocks, state.active,     state.ready,            state.ready_head, state.ready_tail,
        state.completed,       state.error_word, state.ready_lock,       task_count,       state.ready_capacity,
        state.sm_words,
    }) |arg| reset_args.add(arg);
    const reset_threads: u32 = if (state.sm_words != 0) @max(task_count, sm_words_len) else task_count;
    const reset_cfg: cuda.Config = .{
        .grid = .{ .x = @max(1, (reset_threads + 255) / 256), .y = 1, .z = 1 },
        .block = .{ .x = 256, .y = 1, .z = 1 },
    };
    try cuda.launch.launch(reset, reset_cfg, stream, &reset_args);
    var run_args: cuda.Args = .{};
    run_args.add(state);
    const run_cfg: cuda.Config = .{
        .grid = .{ .x = plan.persistent_blocks, .y = 1, .z = 1 },
        .block = .{ .x = block_threads, .y = 1, .z = 1 },
        .shared = plan.dynamic_smem,
    };
    try cuda.launch.launch(persistent, run_cfg, stream, &run_args);
}

test "task records round trip at the graph byte offsets" {
    try std.testing.expectEqual(@sizeOf(graph.Task), @sizeOf(TaskRecord));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(TaskRecord));
    try std.testing.expectEqual(@offsetOf(graph.Task, "layer"), @offsetOf(TaskRecord, "layer"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "aux"), @offsetOf(TaskRecord, "aux"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "successor_offset"), @offsetOf(TaskRecord, "successor_offset"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "kind"), @offsetOf(TaskRecord, "kind"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "row"), @offsetOf(TaskRecord, "row"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "dependency_count"), @offsetOf(TaskRecord, "dependency_count"));
    try std.testing.expectEqual(@offsetOf(graph.Task, "successor_count"), @offsetOf(TaskRecord, "successor_count"));
    const source: graph.Task = .{ .kind = .lowrank, .layer = 77, .row = 15, .aux = 0x1234, .dependency_count = 2, .successor_offset = 0x12345678, .successor_count = 9 };
    const encoded = encodeTask(source, .lowrank_partial);
    try std.testing.expectEqual(@backingInt(source.kind), encoded.kind);
    try std.testing.expectEqual(@as(u16, 77), encoded.layer);
    try std.testing.expectEqual(@as(u16, 0x1234), encoded.aux);
    try std.testing.expectEqual(@as(u32, 0x12345678), encoded.successor_offset);
    try std.testing.expectEqual(@as(u16, 9), encoded.successor_count);
    try std.testing.expectEqual(@as(u16, 1), encoded.reserved);
    try std.testing.expectEqual(@as(u8, 4), @backingInt(graph.Kind.kv_write));
    try std.testing.expectEqual(@as(u8, 27), @backingInt(graph.Kind.qpe_pack));
    try std.testing.expectEqual(@as(u8, 28), @backingInt(graph.Kind.index_key_write));
    try std.testing.expectEqual(@as(u8, 29), @backingInt(graph.Kind.expand_v));
}

test "serial scheduler reference covers fan-out, fan-in and task work blocks" {
    var g = try graph.build(std.testing.allocator, .{ .rows = 2, .layers = 1, .dcp = 1 });
    defer g.deinit();
    const blocks = try std.testing.allocator.alloc(u32, g.tasks.len);
    defer std.testing.allocator.free(blocks);
    @memset(blocks, 1);
    blocks[0] = 3;
    const order = try simulate(std.testing.allocator, g, blocks, 100);
    defer std.testing.allocator.free(order);
    var expected: usize = g.tasks.len;
    expected += 2;
    try std.testing.expectEqual(expected, order.len);
    for (g.tasks, 0..) |task, from| {
        var last_from: ?usize = null;
        var first_to: ?usize = null;
        for (order, 0..) |id, at| {
            if (id == from) last_from = at;
            for (g.successors[task.successor_offset .. task.successor_offset + task.successor_count]) |to| {
                if (id == to and first_to == null) first_to = at;
            }
        }
        if (task.successor_count == 0) continue;
        try std.testing.expect(last_from.? < first_to.?);
    }

    const broken_counters = try std.testing.allocator.dupe(u32, g.initial_counters);
    defer std.testing.allocator.free(broken_counters);
    broken_counters[1] += 1;
    var broken = g;
    broken.initial_counters = broken_counters;
    try std.testing.expectError(error.Timeout, simulate(std.testing.allocator, broken, blocks, 5));
}

test "dispatch operation ids, task block contract, and shared-memory budget" {
    try std.testing.expectEqual(@as(u8, 8), @backingInt(Op.embed_pick));
    try std.testing.expectEqual(@as(u8, 9), attn_base);
    try std.testing.expectEqual(@as(u8, 12), moe_base);
    try std.testing.expectEqual(@as(u8, 20), head_base);
    try std.testing.expectEqual(@as(u8, 31), @backingInt(Op.index_scores));
    try std.testing.expectEqual(@as(u8, 36), @backingInt(Op.moe_router_slice_sum));
    try std.testing.expect(isDispatchable(32, 0, false, false));
    try std.testing.expect(isDispatchable(128, task_smem_bytes, false, false));
    try std.testing.expect(!isDispatchable(1, 0, false, false));
    try std.testing.expect(!isDispatchable(288, 0, false, false));
    try std.testing.expect(!isDispatchable(128, task_smem_bytes + 1, false, false));
    try std.testing.expect(!isDispatchable(128, 0, true, false));
    try std.testing.expect(!isDispatchable(128, 0, false, true));
    try std.testing.expectEqual(@as(u32, 53416), dispatchedStaticSmem());
    try std.testing.expect(dispatchedStaticSmem() <= sm121_optin_smem_bytes);
    for (dispatched_specs) |spec| {
        try std.testing.expect(isDispatchable(spec.threads, task_smem_bytes, false, false));
    }
    std.debug.print("GLM53-MK shared memory total: {d} B / {d} B (task arena {d} B, scheduler {d} B)\n", .{
        dispatchedStaticSmem(), sm121_optin_smem_bytes, task_smem_bytes, scheduler_static_smem_bytes,
    });
}

test "payload arena layout: 16-byte aligned offsets in task order" {
    var offsets: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 16 + 48 + 0 + 9), try layoutPayloads(&.{ 1, 40, 0, 9 }, &offsets));
    try std.testing.expectEqualSlices(usize, &.{ 0, 16, 64, 64 }, &offsets);
    try std.testing.expectError(error.Invalid, layoutPayloads(&.{1}, offsets[0..2]));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Functions);
    std.testing.refAllDecls(Uploaded);
    std.testing.refAllDecls(TaskRecord);
    std.testing.refAllDecls(RuntimeState);
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(Op);
    std.testing.refAllDecls(ErrorCode);
    std.testing.refAllDecls(DispatchKind);
    std.testing.refAllDecls(DispatchSpec);
    std.testing.refAllDecls(AttnInputNormParams);
    std.testing.refAllDecls(AttnQANormParams);
    std.testing.refAllDecls(AttnQpePackParams);
    std.testing.refAllDecls(AttnAttendParams);
    std.testing.refAllDecls(AttnKvWriteParams);
    std.testing.refAllDecls(AttnAbsorbParams);
    std.testing.refAllDecls(AttnMlaMergeParams);
    std.testing.refAllDecls(AttnExpandVParams);
}
