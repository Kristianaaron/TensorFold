// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 CUDA forward through the REAL launch wrappers on a recording host driver: every cuLaunchKernel(Ex) and
//! 2D copy is recorded (kernel name, grid, arguments), every collective call (phase, bytes) and every row-input
//! upload (bytes). Device memory is a bump allocator behind cuda.memory's hook (no bytes behind it, no GPU).
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("forward");
const engine = @import("engine");
const state = @import("state");
const weights = @import("weights");
const sampler = @import("sampler");

const gpa = std.testing.allocator;
const abi = cuda.abi;
const Params = ?[*]?*anyopaque;

/// Kernel names by fake function handle (handle = index + 1).
const names = [_][]const u8{
    "tf_glm_rmsnorm_6144",      "tf_glm_rmsnorm_2048",           "tf_glm_residual_add",       "tf_glm_embed_b16",               "tf_glm_router_part",
    "tf_glm_router_tile",       "tf_glm_topk",                   "tf_glm_topk_compact",        "tf_glm_slice_sum",               "tf_glm53_lowrank_partial", "tf_glm53_lowrank_reduce", "tf_glm53_lowrank_decode",
    "tf_glm53_lowrank_finish",  "tf_glm53_embed_pick",           "tf_glm53_add_partial",      "tf_glm53_swiglu_plain",          "tf_glm53_kv_write",
    "tf_glm53_qpe_pack",        "tf_glm53_absorb",               "tf_glm53_attend",           "tf_glm53_mla_merge",             "tf_glm53_expand_v", "tf_glm53_absorb_decode", "tf_glm53_expand_v_decode", "tf_glm53_absorb_decode_v2", "tf_glm53_absorb_decode_v3", "tf_glm53_expand_v_decode_v2", "tf_glm53_expand_v_decode_v2_16",
    "tf_glm53_index_key",       "tf_glm53_index_q",              "tf_glm53_index_scores",     "tf_glm53_index_scores_oracle",   "tf_glm53_topk",
    "tf_glm53_merge",           "tf_glm53_merge2",               "tf_glm53_top1_f32",          "tf_glm53_cand_part_f32",         "tf_glm53_cand_part_fast_f32",   "tf_glm53_cand_merge_f32",        "tf_glm53_lse_f32",
    "tf_glm53_gate_gather_f32", "tf_glm53_gate_mass_f32",        "qmmf_bf16",                 "qmmf_f32",                       "qmmf_reduce",
    "exl3_group",               "exl3_rot_in",                   "exl3_grouped",              "exl3_gateup",                    "exl3_down_combine",
    "tf_exl3_prompt_rotate",    "tf_exl3_prompt_gu",              "tf_exl3_prompt_down",       "tf_exl3_prompt_gu_e4m3",            "tf_exl3_prompt_down_e4m3",
    "group_plan_small",         "group_plan_rank",                "group_plan_offsets",        "group_plan_scatter",
    "tf_glm53_cand_fill_f32",   "tf_glm53_lowrank_combine_bf16", "tf_glm53_moe_residual_add", "tf_glm53_residual_rmsnorm_6144", "tf_glm53_attend_mma", "tf_glm53_attend_split_merge", "tf_glm53_index_scores_mma_4", "tf_glm53_index_scores_mma_8", "tf_glm53_index_scores_mma_16", "tf_glm53_index_scores_qs_s2", "tf_glm53_index_scores_qs_s3", "tf_glm53_index_scores_qs2_k64_s2", "tf_glm53_index_scores_qs2_k32_s2", "tf_glm53_index_scores_qs2_k32_s3", "tf_glm53_ablate_6144", "tf_glm53_unpack_bf16_rows", "df_tap_glm", "tf_glm53_absorb_prompt_mma", "tf_glm53_expand_v_prompt_mma", "tf_glm53_smla_p",
    "tf_glm53_cand_fill_f32",   "tf_glm53_lowrank_combine_bf16", "tf_glm53_moe_residual_add", "tf_glm53_residual_rmsnorm_6144", "tf_glm53_attend_mma", "tf_glm53_attend_split_merge", "tf_glm53_index_scores_mma_4", "tf_glm53_index_scores_mma_8", "tf_glm53_index_scores_mma_16", "tf_glm53_index_scores_qs_s2", "tf_glm53_index_scores_qs_s3", "tf_glm53_index_scores_qs2_k64_s2", "tf_glm53_index_scores_qs2_k32_s2", "tf_glm53_index_scores_qs2_k32_s3", "tf_glm53_ablate_6144", "tf_glm53_unpack_bf16_rows", "df_tap_glm", "tf_glm53_absorb_prompt_mma", "tf_glm53_expand_v_prompt_mma", "tf_glm53_smla_p", "tf_glm53_smla_p_fp8",
    "tf_glm53_absorb_prompt_mma_big", "tf_glm53_expand_v_prompt_mma_big", "tf_glm53_absorb_prompt_p2", "tf_glm53_expand_v_prompt_p2", "tf_glm53_expand_v_prompt_p2n", "tf_glm53_lowrank_combine_bf16_decode",
    "tf_exl3_prompt_gu_s",      "tf_exl3_prompt_gu_s_e4m3",      "tf_exl3_prompt_down_s",     "tf_exl3_prompt_down_s_e4m3",
    "tf_exl3_prompt_wup",       "tf_exl3_prompt_wup_e4m3",       "tf_exl3_prompt_wgate",      "tf_exl3_prompt_wgate_e4m3",
    "tf_exl3_prompt_wdown",     "tf_exl3_prompt_wdown_e4m3",
    "tf_glm53_pmx_gu_mcg",       "tf_glm53_pmx_gu_e4m3",
    "tf_glm53_w8a8_quant",      "tf_glm53_w8a8_gemm",
    "tf_glm53_smla_union",      "tf_glm53_smla_p3", "tf_qmmf_group_fp8b_16c", "tf_qmmf_group_fp8b_32c", "tf_qmmf_group_fp8b_64c", "tf_qmmf_group_fp8b_64f",
    "tf_glm53_attend_verify",
    "tf_glm53_p5a_gu_mcg", "tf_glm53_p5a_gu_e4m3", "tf_glm53_p5a_down_mcg", "tf_glm53_p5a_down_e4m3",
    "tf_glm53_mask_f32", "tf_glm53_cand2_fill_f32", "tf_glm53_cand2_hist_f32", "tf_glm53_cand2_select_f32",
    "tf_glm53_moe_residual_rmsnorm_6144", "tf_glm53_absorb_qpe_decode_v3",
    "l2pf_segments", "l2pf_paced",
    "tf_glm53_rmsnorm_2048_halves",
};

fn id(comptime name: []const u8) usize {
    @setEvalBranchQuota(100_000);
    inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i + 1;
    @compileError("unknown kernel " ++ name);
}

fn candFastFromEnv() bool {
    const value = std.c.getenv("TF_GLM53_CAND_FAST") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

fn candV2FromEnv() bool {
    const value = std.c.getenv("TF_GLM53_CAND_V2") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

var driver: cuda.Driver = undefined;

fn func(comptime name: []const u8) cuda.Function {
    return .{ .d = &driver, .handle = @ptrFromInt(id(name)) };
}

const max_args = 24;
const Launch = struct { name: usize, grid: [3]u32, block: [3]u32, shared: u32, nargs: usize, args: [max_args]u64, group_outs: [3]u64 = @splat(0) };
const Gather = struct { phase: forward.Phase, bytes: usize, mine: u64, all: u64, send_recv: bool, recv_bytes: usize = 0, ctx: usize = 0, stream: usize = 0 };
/// One ordered step on some stream: a kernel, a collective, an event record or an event wait.
const Step = struct { kind: enum { launch, collective, record, wait }, what: usize, stream: usize };
const Upload = struct { dst: u64, bytes: []u8 };
const CopyRecord = struct { src: u64, dst: u64, src_pitch: usize, dst_pitch: usize, width: usize, height: usize, stream: usize };
const EventStamp = struct { handle: usize, ms: f64 };

/// Argument layouts of the kernels the contract checks read (p pointer, i i32, l i64, u u32, f f32).
fn layout(name: usize) ?[]const u8 {
    const table = .{
        .{ "l2pf_paced", "piupp" }, // rows ptr, rows, chunk, ns a piece, start
        .{ "l2pf_segments", "piu" },
        .{ "tf_glm53_lowrank_partial", "pppiiiiii" },
        .{ "tf_glm53_lowrank_decode", "pppiiiiii" },
        .{ "tf_glm53_lowrank_reduce", "ppii" },
        .{ "tf_glm53_lowrank_combine_bf16", "pppppiiii" },
        .{ "tf_glm53_lowrank_combine_bf16_decode", "pppppiiii" },
        .{ "tf_glm53_kv_write", "pppppppppiiii" },
        .{ "tf_glm53_qpe_pack", "ppppii" },
        .{ "tf_glm53_absorb", "ppppiii" },
        .{ "tf_glm53_absorb_decode", "ppppiii" },
        .{ "tf_glm53_absorb_decode_v2", "ppppiii" },
        .{ "tf_glm53_absorb_decode_v3", "ppppiii" },
        .{ "tf_glm53_attend", "pppppppppppppiiiii" },
        .{ "tf_glm53_attend_mma", "pppppppppppppiiiiipi" },
        .{ "tf_glm53_smla_p", "pppppppppppppiiiii" },
        .{ "tf_glm53_smla_p_fp8", "pppppppppppppiiiii" },
        .{ "tf_glm53_smla_union", "ppppiii" },
        .{ "tf_glm53_smla_p3", "pppppppppppppiiiiip" },
        .{ "tf_glm53_attend_verify", "ppppppppiiii" },
        .{ "tf_glm53_attend_split_merge", "ppiii" },
        .{ "tf_glm53_mla_merge", "ppppiii" },
        .{ "tf_glm53_expand_v", "ppppiii" },

        .{ "tf_glm53_absorb_prompt_mma", "ppppiii" },
        .{ "tf_glm53_expand_v_prompt_mma", "ppppiii" },
        .{ "tf_glm53_absorb_prompt_mma_big", "ppppiii" },
        .{ "tf_glm53_expand_v_prompt_mma_big", "ppppiii" },
        .{ "tf_glm53_absorb_prompt_p2", "ppppiii" },
        .{ "tf_glm53_expand_v_prompt_p2", "ppppiii" },
        .{ "tf_glm53_expand_v_prompt_p2n", "ppppiii" },
        .{ "tf_glm53_expand_v_decode", "ppppiii" },
        .{ "tf_glm53_expand_v_decode_v2", "ppppiii" },
        .{ "tf_glm53_expand_v_decode_v2_16", "ppppiii" },

        .{ "tf_glm53_index_key", "pppppppiiiif" },
        .{ "tf_glm53_index_q", "pppp" },
        .{ "tf_glm53_index_scores", "ppiipppiiii" },
        .{ "tf_glm53_index_scores_qs_s2", "ppiipppiiiiiii" },
        .{ "tf_glm53_index_scores_qs_s3", "ppiipppiiiiiii" },
        .{ "tf_glm53_index_scores_qs2_k64_s2", "ppiipppiiiiiii" },
        .{ "tf_glm53_index_scores_qs2_k32_s2", "ppiipppiiiiiii" },
        .{ "tf_glm53_index_scores_qs2_k32_s3", "ppiipppiiiiiii" },
        .{ "tf_glm53_merge", "ppppppiiii" },
        .{ "tf_glm53_merge2", "ppppppiiiii" },
        .{ "tf_glm_rmsnorm_6144", "plpplpfi" },
        .{ "tf_glm53_residual_rmsnorm_6144", "ppppif" },
        .{ "tf_glm53_moe_residual_rmsnorm_6144", "ppppif" },
        .{ "tf_glm53_absorb_qpe_decode_v3", "ppppiiippp" },
        .{ "tf_glm_rmsnorm_2048", "plpplpfi" },
        .{ "tf_glm53_rmsnorm_2048_halves", "pplpplf" },
        .{ "tf_glm_residual_add", "ppplli" },
        .{ "tf_glm_embed_b16", "pppli" },
        .{ "tf_glm53_embed_pick", "pppiii" },
        .{ "tf_glm53_unpack_bf16_rows", "pplll" },
        .{ "qmmf_bf16", "pppfppiiiiiii" },
        .{ "qmmf_f32", "pppfppiiiiiii" },
        .{ "tf_glm53_top1_f32", "puupp" },
        .{ "tf_glm53_mask_f32", "puupu" },
        .{ "tf_glm53_cand2_fill_f32", "ppupu" },
        .{ "tf_glm53_cand2_hist_f32", "puup" },
        .{ "tf_glm53_cand2_select_f32", "puuuppp" },
        .{ "tf_glm53_lse_f32", "pup" },
        .{ "tf_glm53_ablate_6144", "ppippppif" },
        .{ "df_tap_glm", "ppiiii" },
        .{ "exl3_down_combine", "ppppppiiiii" },
        .{ "tf_glm53_w8a8_quant", "pliipp" },
        .{ "tf_glm53_w8a8_gemm", "pppppiiii" },
    };
    inline for (table) |t| if (name == id(t[0])) return t[1];
    return null;
}

const Rec = struct {
    var launches: std.ArrayList(Launch) = .empty;
    var gathers: std.ArrayList(Gather) = .empty;
    var uploads: std.ArrayList(Upload) = .empty;
    var copies: usize = 0;
    var copy_log: std.ArrayList(CopyRecord) = .empty;
    var gate_calls: usize = 0;
    var gate_token: u64 = 0;
    var gate_status: u64 = 0;
    var order: std.ArrayList(Step) = .empty;
    var event_stamps: std.ArrayList(EventStamp) = .empty;
    var event_clock_ms: f64 = 0;
    var use_event_clock = false;
    var next_event: usize = 0x700000;
    var next_stream: usize = 0x51de;

    fn eventElapsed(ms: *f32, start: abi.Event, end: abi.Event) callconv(.c) abi.Result {
        if (use_event_clock) {
            var start_ms: ?f64 = null;
            var end_ms: ?f64 = null;
            for (event_stamps.items) |stamp| {
                if (stamp.handle == @intFromPtr(start)) start_ms = stamp.ms;
                if (stamp.handle == @intFromPtr(end)) end_ms = stamp.ms;
            }
            ms.* = @floatCast((end_ms orelse 0) - (start_ms orelse 0));
        } else {
            ms.* = 1.25;
        }
        return abi.success;
    }

    fn step(kind: @FieldType(Step, "kind"), what: usize, stream: ?*anyopaque) void {
        order.append(gpa, .{ .kind = kind, .what = what, .stream = @intFromPtr(stream) }) catch @panic("recording");
    }

    fn eventRecord(e: abi.Event, s: abi.Stream) callconv(.c) abi.Result {
        event_stamps.append(gpa, .{ .handle = @intFromPtr(e), .ms = event_clock_ms }) catch @panic("recording");
        event_clock_ms += 1.25;
        step(.record, @intFromPtr(e), s);
        return abi.success;
    }

    fn streamWait(s: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) abi.Result {
        step(.wait, @intFromPtr(e), s);
        return abi.success;
    }

    fn streamCreate(out: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(next_stream);
        next_stream += 1;
        return abi.success;
    }

    fn streamDestroy(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }

    fn streamSynchronize(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }

    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        next_event += 1;
        out.* = @ptrFromInt(next_event);
        return abi.success;
    }

    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return abi.success;
    }

    fn reset() void {
        marks.clearRetainingCapacity();
        order.clearRetainingCapacity();
        event_stamps.clearRetainingCapacity();
        event_clock_ms = 0;
        use_event_clock = false;
        launches.clearRetainingCapacity();
        gathers.clearRetainingCapacity();
        for (uploads.items) |u| gpa.free(u.bytes);
        uploads.clearRetainingCapacity();
        copies = 0;
        copy_log.clearRetainingCapacity();
        gate_calls = 0;
        gate_token = 0;
        gate_status = 0;
    }

    fn free() void {
        reset();
        launches.deinit(gpa);
        gathers.deinit(gpa);
        uploads.deinit(gpa);
        copy_log.deinit(gpa);
        order.deinit(gpa);
        event_stamps.deinit(gpa);
        order = .empty;
        marks.deinit(gpa);
        marks = .empty;
        launches = .empty;
        gathers = .empty;
        uploads = .empty;
        copy_log = .empty;
        event_stamps = .empty;
    }

    fn record(f: abi.Function, grid: [3]u32, block: [3]u32, shared: u32, params: ?[*]?*anyopaque) void {
        const name = @intFromPtr(f.?);
        var l: Launch = .{ .name = name, .grid = grid, .block = block, .shared = shared, .nargs = 0, .args = @splat(0) };
        if (layout(name)) |kinds| {
            l.nargs = kinds.len;
            for (kinds, 0..) |k, i| {
                const p = params.?[i].?;
                l.args[i] = switch (k) {
                    'p' => @as(*const u64, @ptrCast(@alignCast(p))).*,
                    'l' => @bitCast(@as(*const i64, @ptrCast(@alignCast(p))).*),
                    'i' => @bitCast(@as(i64, @as(*const i32, @ptrCast(@alignCast(p))).*)),
                    'u' => @as(*const u32, @ptrCast(@alignCast(p))).*,
                    'f' => @as(*const u32, @ptrCast(@alignCast(p))).*,
                    else => unreachable,
                };
            }
        }
        if (name == id("tf_qmmf_group_fp8b_16c") or name == id("tf_qmmf_group_fp8b_32c") or name == id("tf_qmmf_group_fp8b_64c") or name == id("tf_qmmf_group_fp8b_64f")) {
            const group_args: cuda.qmmf.GroupArgsAbi = @as(*const cuda.qmmf.GroupArgsAbi, @ptrCast(@alignCast(params.?[0].?))).*;
            const group_count: usize = @intCast(group_args.count);
            for (0..@min(group_count, l.group_outs.len)) |i| {
                l.group_outs[i] = group_args.seg[i].out;
            }
        }
        launches.append(gpa, l) catch @panic("recording");
    }

    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: abi.Stream, params: Params, _: Params) callconv(.c) abi.Result {
        record(f, .{ gx, gy, gz }, .{ bx, by, bz }, shared, params);
        step(.launch, @intFromPtr(f.?), s);
        return abi.success;
    }

    fn launchEx(cfg: *const abi.LaunchConfig, f: abi.Function, params: Params, _: Params) callconv(.c) abi.Result {
        record(f, .{ cfg.grid_x, cfg.grid_y, cfg.grid_z }, .{ cfg.block_x, cfg.block_y, cfg.block_z }, cfg.shared_bytes, params);
        step(.launch, @intFromPtr(f.?), cfg.stream);
        return abi.success;
    }

    fn copy2d(copy: *const abi.Memcpy2D, stream: abi.Stream) callconv(.c) abi.Result {
        copies += 1;
        copy_log.append(gpa, .{ .src = copy.src, .dst = copy.dst, .src_pitch = copy.src_pitch, .dst_pitch = copy.dst_pitch, .width = copy.width, .height = copy.height, .stream = @intFromPtr(stream) }) catch @panic("recording");
        return abi.success;
    }

    fn memset32(_: abi.DevicePtr, _: c_uint, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }

    fn gather(ctx: *anyopaque, s: cuda.Stream, mine: u64, all: u64, bytes: usize, phase: forward.Phase) anyerror!void {
        try gathers.append(gpa, .{ .phase = phase, .bytes = bytes, .mine = mine, .all = all, .send_recv = false, .ctx = @intFromPtr(ctx), .stream = if (s.handle) |h| @intFromPtr(h) else 0 });
        step(.collective, @backingInt(phase), s.handle);
    }

    fn sendRecv(ctx: *anyopaque, s: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, _: usize, phase: forward.Phase) anyerror!void {
        try gathers.append(gpa, .{ .phase = phase, .bytes = send_bytes, .mine = send, .all = recv, .send_recv = true, .recv_bytes = recv_bytes, .ctx = @intFromPtr(ctx), .stream = if (s.handle) |h| @intFromPtr(h) else 0 });
        step(.collective, @backingInt(phase), s.handle);
    }

    fn oneShot(ctx: *anyopaque, s: cuda.Stream, mine: u64, all: u64, bytes: usize) anyerror!void {
        // Decode's only one-shot gather in these fixtures is the attention partial (PARTIAL_ROCE; COLL_FUSE is off).
        try gathers.append(gpa, .{ .phase = .attention_partial, .bytes = bytes, .mine = mine, .all = all, .send_recv = false, .ctx = @intFromPtr(ctx), .stream = if (s.handle) |h| @intFromPtr(h) else 0 });
        step(.collective, @backingInt(forward.Phase.attention_partial), s.handle);
    }

    fn gate(_: *anyopaque, _: cuda.Stream, token: u64, status: u64) anyerror!void {
        gate_calls += 1;
        gate_token = token;
        gate_status = status;
    }

    var positions_dst: u64 = 0;
    var marks: std.ArrayList(usize) = .empty;

    fn upload(_: ?*anyopaque, _: cuda.Stream, dst: u64, bytes: []const u8) anyerror!void {
        try uploads.append(gpa, .{ .dst = dst, .bytes = try gpa.dupe(u8, bytes) });
        if (dst == positions_dst) try marks.append(gpa, launches.items.len); // a window (chunk) starts
    }
};

/// Device addresses from a bump counter; no memory behind them (nothing executes).
const Bump = struct {
    var next: u64 = 0x10_0000_0000;
    fn alloc(_: *anyopaque, len: usize) ?abi.DevicePtr {
        const p = next;
        next += std.mem.alignForward(u64, @max(len, 1), 4096);
        return p;
    }
    fn free(_: *anyopaque, _: abi.DevicePtr) bool {
        return true;
    }
};

fn installDriver() void {
    driver.api.cuLaunchKernel = Rec.launch;
    driver.api.cuLaunchKernelEx = Rec.launchEx;
    driver.api.cuMemcpy2DAsync_v2 = Rec.copy2d;
    driver.api.cuMemsetD32Async = Rec.memset32;
    driver.api.cuEventRecord = Rec.eventRecord;
    driver.api.cuEventCreate = Rec.eventCreate;
    driver.api.cuEventDestroy_v2 = Rec.eventDestroy;
    driver.api.cuEventElapsedTime = Rec.eventElapsed;
    driver.api.cuStreamWaitEvent = Rec.streamWait;
    driver.api.cuStreamCreate = Rec.streamCreate;
    driver.api.cuStreamDestroy_v2 = Rec.streamDestroy;
    driver.api.cuStreamSynchronize = Rec.streamSynchronize;
}

fn fakeLane() cuda.qmmf.Lane {
    var l: cuda.qmmf.Lane = undefined;
    for (&l.fns) |*set| set.* = @splat(func("qmmf_bf16"));
    for (&l.f32_fns) |*set| set.* = @splat(func("qmmf_f32"));
    l.reduce = @splat(func("qmmf_reduce"));
    l.group_fns = @splat(null);
    l.clusters = true; // sm_90+: split-K slices meet in a cluster (cuLaunchKernelEx), no reduce launch
    return l;
}

fn groupLane(fx: *Fixture) void {
    fx.lane.group_fns = .{ func("tf_qmmf_group_fp8b_16c"), func("tf_qmmf_group_fp8b_32c"), func("tf_qmmf_group_fp8b_64c"), func("tf_qmmf_group_fp8b_64f") };
}

fn fakeExperts() cuda.exl3_experts.Experts {
    var e: cuda.exl3_experts.Experts = undefined;
    e.mod = .{ .d = &driver, .handle = @ptrFromInt(0x4000) };
    e.router_mod = .{ .d = &driver, .handle = @ptrFromInt(0x5000) };
    e.router = .{ .plan_small = func("group_plan_small"), .plan_rank = func("group_plan_rank"), .plan_offsets = func("group_plan_offsets"), .plan_scatter = func("group_plan_scatter") };
    e.group = func("exl3_group");
    e.rot_in = func("exl3_rot_in");
    e.grouped = @splat(func("exl3_grouped"));
    e.gateup = func("exl3_gateup");
    e.down_combine = func("exl3_down_combine");
    e.prompt = .{ .rotate = func("tf_exl3_prompt_rotate"), .gateup = func("tf_exl3_prompt_gu"), .down = func("tf_exl3_prompt_down"), .gateup_e4m3 = func("tf_exl3_prompt_gu_e4m3"), .down_e4m3 = func("tf_exl3_prompt_down_e4m3"), .wide = .{ .gu_small = .{ func("tf_exl3_prompt_gu_s"), func("tf_exl3_prompt_gu_s_e4m3") }, .down_small = .{ func("tf_exl3_prompt_down_s"), func("tf_exl3_prompt_down_s_e4m3") }, .up = .{ func("tf_exl3_prompt_wup"), func("tf_exl3_prompt_wup_e4m3") }, .gate = .{ func("tf_exl3_prompt_wgate"), func("tf_exl3_prompt_wgate_e4m3") }, .down = .{ func("tf_exl3_prompt_wdown"), func("tf_exl3_prompt_wdown_e4m3") } } };
    e.pmx = .{ .mod = .{ .d = &driver, .handle = @ptrFromInt(0x6000) }, .plan_mod = .{ .d = &driver, .handle = @ptrFromInt(0x5000) }, .router = e.router, .gu = .{ func("tf_glm53_pmx_gu_mcg"), func("tf_glm53_pmx_gu_e4m3") }, .dn = .{ undefined, undefined }, .dequant = @splat(undefined) };
    e.p5a = .{ .mod = .{ .d = &driver, .handle = @ptrFromInt(0x7000) }, .gu = .{ func("tf_glm53_p5a_gu_mcg"), func("tf_glm53_p5a_gu_e4m3") }, .down = .{ func("tf_glm53_p5a_down_mcg"), func("tf_glm53_p5a_down_e4m3") } };
    e.plan_rotate = func("group_plan_small");
    e.plan_finish = func("group_plan_scatter");
    return e;
}

fn fakeKernels(lane: *const cuda.qmmf.Lane, ex: *const cuda.exl3_experts.Experts) forward.Kernels {
    var k: forward.Kernels = undefined;
    inline for (@typeInfo(forward.Glue).@"struct".field_names) |n| @field(k.glue, n) = func("tf_glm_" ++ n);
    k.glue53 = .{ .lowrank = .{ .partial = func("tf_glm53_lowrank_partial"), .reduce = func("tf_glm53_lowrank_reduce"), .finish = func("tf_glm53_lowrank_finish"), .decode = func("tf_glm53_lowrank_decode") }, .embed_pick = func("tf_glm53_embed_pick"), .lowrank_combine_bf16 = func("tf_glm53_lowrank_combine_bf16"), .lowrank_combine_bf16_decode = func("tf_glm53_lowrank_combine_bf16_decode"), .residual_add_bf16 = func("tf_glm53_moe_residual_add") };
    k.swiglu_plain = func("tf_glm53_swiglu_plain");
    k.residual_rmsnorm_6144 = func("tf_glm53_residual_rmsnorm_6144");
    k.moe_residual_rmsnorm_6144 = null;
    k.rmsnorm_2048_halves = null;

    k.mla = .{ .prep_fn = func("tf_glm53_kv_write"), .qpe_fn = func("tf_glm53_qpe_pack"), .absorb_fn = func("tf_glm53_absorb"), .absorb_decode_fn = func("tf_glm53_absorb_decode"), .absorb_decode_v2_fn = func("tf_glm53_absorb_decode_v2"), .absorb_decode_v3_fn = func("tf_glm53_absorb_decode_v3"), .attend_fn = func("tf_glm53_attend"), .attend_mma_fn = func("tf_glm53_attend_mma"), .attend_split_merge_fn = func("tf_glm53_attend_split_merge"), .attend_kernel = .scalar, .merge_fn = func("tf_glm53_mla_merge"), .expand_fn = func("tf_glm53_expand_v"), .expand_decode_fn = func("tf_glm53_expand_v_decode"), .expand_decode_v2_fn = func("tf_glm53_expand_v_decode_v2"), .expand_decode_v2_16_fn = func("tf_glm53_expand_v_decode_v2_16"), .absorb_prompt_fn = func("tf_glm53_absorb_prompt_mma"), .expand_prompt_fn = func("tf_glm53_expand_v_prompt_mma"), .smla_p_fn = func("tf_glm53_smla_p"), .smla_fp8_fn = func("tf_glm53_smla_p_fp8"), .absorb_prompt_big_fn = func("tf_glm53_absorb_prompt_mma_big"), .expand_prompt_big_fn = func("tf_glm53_expand_v_prompt_mma_big"), .absorb_prompt_p2_fn = func("tf_glm53_absorb_prompt_p2"), .expand_prompt_p2_fn = func("tf_glm53_expand_v_prompt_p2"), .expand_prompt_p2n_fn = func("tf_glm53_expand_v_prompt_p2n") }; // the recorded contract below is the scalar attend's; the mma path is covered on GPU (glm53-mla, graph replay)

    k.index = .{ .key = func("tf_glm53_index_key"), .query = func("tf_glm53_index_q"), .score = func("tf_glm53_index_scores"), .score_mma_4 = func("tf_glm53_index_scores_mma_4"), .score_mma_8 = func("tf_glm53_index_scores_mma_8"), .score_mma_16 = func("tf_glm53_index_scores_mma_16"), .score_qs_2 = func("tf_glm53_index_scores_qs_s2"), .score_qs_3 = func("tf_glm53_index_scores_qs_s3"), .score_qs2_k64_s2 = func("tf_glm53_index_scores_qs2_k64_s2"), .score_qs2_k32_s2 = func("tf_glm53_index_scores_qs2_k32_s2"), .score_qs2_k32_s3 = func("tf_glm53_index_scores_qs2_k32_s3"), .score_kernel = .scalar, .score_oracle = func("tf_glm53_index_scores_oracle"), .select = func("tf_glm53_topk"), .merge_fn = func("tf_glm53_merge"), .merge2_fn = func("tf_glm53_merge2") };
    k.head = .{ .top1_fn = func("tf_glm53_top1_f32"), .mask_fn = func("tf_glm53_mask_f32"), .candidates_fill_fn = func("tf_glm53_cand_fill_f32"), .candidates_part_fn = func("tf_glm53_cand_part_f32"), .candidates_part_fast_fn = func("tf_glm53_cand_part_fast_f32"), .candidates_v2_fill_fn = func("tf_glm53_cand2_fill_f32"), .candidates_v2_hist_fn = func("tf_glm53_cand2_hist_f32"), .candidates_v2_select_fn = func("tf_glm53_cand2_select_f32"), .candidates_merge_fn = func("tf_glm53_cand_merge_f32"), .lse_fn = func("tf_glm53_lse_f32"), .gate_gather_fn = func("tf_glm53_gate_gather_f32"), .gate_mass_fn = func("tf_glm53_gate_mass_f32"), .fast = candFastFromEnv(), .v2 = candV2FromEnv() };
    k.lane = lane;
    k.experts = ex;
    k.w8a8 = null;
    return k;
}

fn fakeW8a8() cuda.w8a8.Fns {
    return .{ .quant = func("tf_glm53_w8a8_quant"), .gemm = func("tf_glm53_w8a8_gemm") };
}

fn w8(n: usize, k: usize, base: u64) cuda.qmmf.Weight {
    return cuda.fp8.blockWeight(base, base + 0x100, n, k);
}

fn attnOf(i: u64, ix: bool) weights.Attention {
    const b = 0x9000_0000 + i * 0x100_0000;
    return .{ .q_a = w8(2048, 6144, b + 0x1000), .q_b = w8(8192, 2048, b + 0x2000), .kv_a = w8(576, 6144, b + 0x3000), .kv_b = .{ .codes = b + 0x4000, .scales = b + 0x4100, .rows = 32 * 448, .cols = 512 }, .o_proj = w8(6144, 8192, b + 0x5000), .q_a_norm = b + 0x6000, .kv_a_norm = b + 0x6100, .local_heads = 32, .head0 = 0, .indexer = if (ix) .{ .wq_b = w8(4096, 2048, b + 0x7000), .wk = w8(128, 6144, b + 0x8000), .weights_proj = .{ .mode = .bf16, .codes = b + 0x9000, .scales = 0, .n = 32, .k = 6144, .npad = 128 }, .k_norm = b + 0xa000, .k_bias = b + 0xa100 } else null };
}

/// A GLM-5.3-shaped fake layer (all weight addresses in [0x8000_0000, 0xb000_0000)).
fn glmLayer(i: usize, dense: bool, full: bool) weights.DeviceLayer {
    const loaded: @FieldType(weights.Moe, "routed") = .{ .gate = 0xa100_0000, .up = 0xa100_1000, .down = 0xa100_2000, .gate_k2 = 0xa100_3000, .up_k2 = 0xa100_4000, .down_k2 = 0xa100_5000, .suh_g = 0xa100_6000, .suh_u = 0xa100_7000, .svh_g = 0xa100_8000, .svh_u = 0xa100_9000, .suh_d = 0xa100_a000, .svh_d = 0xa100_b000, .codebook = .mcg_e4m3, .uniform_k2 = 0, .widths_ok = true, .pmx_ok = true, .dims = 6144, .width = 1024, .experts = 256 };
    const n: u64 = @intCast(i);
    const ffn: weights.Ffn = if (dense) .{ .dense = .{ .gate_up = w8(12288, 6144, 0x8100_0000), .down = w8(6144, 6144, 0x8200_0000) } } else .{ .moe = .{ .router = 0x8300_0000, .bias = 0x8300_1000, .routed = loaded, .shared = .{ .gate_up = w8(2048, 6144, 0x8400_0000), .down = w8(6144, 1024, 0x8500_0000) }, .lowrank = .{ .a = 0x8600_0000, .b = 0x8700_0000, .rank_local = 32, .rank_full = 64, .span = .{ .lo = 0, .hi = 32 } } } };
    return .{ .index = i, .in_norm = 0x8000_0000 + n * 0x2000, .post_norm = 0x8000_1000 + n * 0x2000, .attn = attnOf(n, full), .ffn = ffn };
}

fn layers() [2]weights.DeviceLayer {
    return .{ glmLayer(0, true, true), glmLayer(1, false, false) };
}

const Fixture = struct {
    lane: cuda.qmmf.Lane,
    ex: cuda.exl3_experts.Experts,
    k: forward.Kernels,
    layer_store: [8]weights.DeviceLayer,
    model: weights.ModelWeights,
    st: state.State,

    fn init(fx: *Fixture, g: state.Geometry, context: usize) !void {
        installDriver();
        var dummy: u8 = 0;
        cuda.memory.setHook(.{ .ctx = &dummy, .alloc = Bump.alloc, .free = Bump.free });
        fx.lane = fakeLane();
        fx.ex = fakeExperts();
        fx.k = fakeKernels(&fx.lane, &fx.ex);
        const two = layers();
        @memcpy(fx.layer_store[0..2], &two);
        fx.model = .{ .allocator = gpa, .rank = 0, .world = 2, .layers = fx.layer_store[0..2], .embed = 0x8800_0000, .vocab = .{ .lo = 0, .hi = 77440 }, .norm = 0x8900_0000, .head = .{ .mode = .bf16, .codes = 0x8a00_0000, .scales = 0, .n = 77440, .k = 6144, .npad = 77440 } };
        const adm = try state.admitRank(g, context, .{}, 1 << 50, state.reserveFor(1 << 50, false, 0));
        Rec.next_stream = 0x51de;
        Rec.next_event = 0x700000;
        fx.st = try state.State.init(&driver, g, context, 0, 2, &adm, 1 << 50);
        Rec.positions_dst = fx.st.ws.positions;
        Rec.reset();
    }

    fn deinit(fx: *Fixture) void {
        fx.st.deinit();
        cuda.memory.setHook(null);
        Rec.free();
    }

    fn fwd(fx: *const Fixture, rank: usize) forward.Forward {
        return .{ .gpa = gpa, .k = &fx.k, .w = &fx.model, .st = &fx.st, .stream = .{ .d = &driver, .handle = null }, .coll = .{ .ctx = @ptrCast(@constCast(fx)), .all_gather_fn = Rec.gather, .send_recv_fn = Rec.sendRecv, .one_shot_gather_fn = Rec.oneShot }, .upload = .{ .upload_fn = Rec.upload }, .rank = rank, .decode_side = fx.st.side != null, .prompt_experts = false, .prompt_experts_wide = false, .smla_p = false, .shard_qkv_a = false, .smla_fp8 = false, .lowrank_decode = false };
    }
};

fn geometry() state.Geometry {
    // Pieces of 1024, 128 score rows, 256 expert rows: the launch-order expectations below are written for them
    // (the bit-neutral production defaults are 512 / 64 / 128; see the "defaults" test).
    return .{ .layers = 2, .full_index_layers = 1, .chunk_rows = 4096, .piece_rows = 1024, .score_rows = 128, .expert_rows = 256, .decode_rows = 8, .graphs_bytes = 0, .head_rows = 8, .candidate_k = 64 };
}

fn prefetchGeometry() state.Geometry {
    var g = geometry();
    g.prefetch = true; // two stage pairs for the side stream
    return g;
}

fn commPipeGeometry() state.Geometry {
    var g = geometry();
    g.expert_rows = 2048;
    g.piece_pipe = true;
    g.moe_pipe = true;
    g.index_pipe = true;
    return g;
}

fn seq(comptime list: anytype) [list.len]usize {
    var out: [list.len]usize = undefined;
    inline for (list, 0..) |n, i| out[i] = id(n);
    return out;
}

const attn_head = .{ "tf_glm_rmsnorm_6144", "qmmf_bf16", "tf_glm_rmsnorm_2048", "qmmf_bf16", "qmmf_bf16", "tf_glm53_kv_write" };
// Keys for every row; the query side (weights_proj, wq_b, index_q) and the selection only when a row is sparse.
const index_part = .{ "qmmf_bf16", "tf_glm53_index_key", "qmmf_bf16", "qmmf_bf16", "tf_glm53_index_q", "tf_glm53_index_scores", "tf_glm53_topk", "tf_glm53_merge" };
const score_block = .{ "tf_glm53_index_scores", "tf_glm53_topk", "tf_glm53_merge" };
const dense_ffn = .{ "qmmf_bf16", "tf_glm53_swiglu_plain", "qmmf_f32", "tf_glm_residual_add" };
const moe_ffn = .{ "tf_glm_router_tile", "tf_glm_slice_sum", "tf_glm_topk", "exl3_group", "exl3_rot_in", "exl3_grouped", "exl3_gateup", "exl3_grouped", "exl3_down_combine", "qmmf_bf16", "tf_glm53_swiglu_plain", "qmmf_f32", "tf_glm53_lowrank_partial", "tf_glm53_lowrank_reduce", "tf_glm53_lowrank_combine_bf16", "tf_glm53_moe_residual_add" };
const moe_ffn_compact = .{ "tf_glm_router_tile", "tf_glm_slice_sum", "tf_glm_topk_compact", "exl3_group", "exl3_rot_in", "exl3_grouped", "exl3_gateup", "exl3_grouped", "exl3_down_combine", "qmmf_bf16", "tf_glm53_swiglu_plain", "qmmf_f32", "tf_glm53_lowrank_partial", "tf_glm53_lowrank_reduce", "tf_glm53_lowrank_combine_bf16", "tf_glm53_moe_residual_add" };
const experts_run = .{ "exl3_group", "exl3_rot_in", "exl3_grouped", "exl3_gateup", "exl3_grouped", "exl3_down_combine" };
const moe_route = .{ "tf_glm_router_tile", "tf_glm_slice_sum", "tf_glm_topk" };
const moe_rest = .{ "qmmf_bf16", "tf_glm53_swiglu_plain", "qmmf_f32", "tf_glm53_lowrank_partial", "tf_glm53_lowrank_reduce", "tf_glm53_lowrank_combine_bf16", "tf_glm53_moe_residual_add" };
const decode_mla = .{ "tf_glm53_qpe_pack", "tf_glm53_absorb", "tf_glm53_attend", "tf_glm53_mla_merge", "tf_glm53_expand_v", "qmmf_f32", "tf_glm53_residual_rmsnorm_6144" };
const prefill_mla = .{ "tf_glm53_qpe_pack", "tf_glm53_absorb", "tf_glm53_attend", "tf_glm53_expand_v", "qmmf_f32", "tf_glm53_residual_rmsnorm_6144" };
const embed_part = .{ "tf_glm_embed_b16", "tf_glm53_embed_pick" };
const head_part = .{ "tf_glm_rmsnorm_6144", "qmmf_f32", "tf_glm53_top1_f32", "tf_glm53_lse_f32" };

fn names_of() ![]usize {
    const out = try gpa.alloc(usize, Rec.launches.items.len);
    for (Rec.launches.items, out) |l, *o| o.* = l.name;
    return out;
}

fn expectSeq(want: []const usize) !void {
    const got = try names_of();
    defer gpa.free(got);
    if (!std.mem.eql(usize, want, got)) {
        for (got, 0..) |g, i| std.debug.print("{d}: {s}{s}\n", .{ i, names[g - 1], if (i < want.len and want[i] != g) "   <- want " else "" });
        return error.LaunchOrder;
    }
}

fn concat(comptime parts: anytype) []const usize {
    comptime var total: usize = 0;
    inline for (parts) |p| total += p.len;
    const S = struct {
        const v = blk: {
            var out: [total]usize = undefined;
            var at: usize = 0;
            for (parts) |p| {
                for (p) |n| {
                    out[at] = id(n);
                    at += 1;
                }
            }
            break :blk out;
        };
    };
    return &S.v;
}

test "decode window: launch order per layer through the packed DCP2 merge wrappers" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(0);
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    try expectSeq(concat(.{ embed_part, attn_head, index_part, decode_mla, dense_ffn, attn_head, decode_mla, moe_ffn, head_part }));
    try std.testing.expectEqual(@as(usize, 2), Rec.copies); // route picks and weights compacted [R][9] -> [R][8]
}

test "decode head mask launch precedes top1 and leaves the unmasked head sequence intact" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.mask = .{ .bits = 0x1234_0000, .words = 4840 };
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    const masked_head = .{ "tf_glm_rmsnorm_6144", "qmmf_f32", "tf_glm53_mask_f32", "tf_glm53_top1_f32", "tf_glm53_lse_f32" };
    try expectSeq(concat(.{ embed_part, attn_head, index_part, decode_mla, dense_ffn, attn_head, decode_mla, moe_ffn, masked_head }));
    var found = false;
    for (Rec.launches.items) |launch| if (launch.name == id("tf_glm53_mask_f32")) {
        found = true;
        try std.testing.expectEqual(fx.st.ws.logits, launch.args[0]);
        try std.testing.expectEqual(@as(u64, 77440), launch.args[1]);
        try std.testing.expectEqual(@as(u64, 0), launch.args[2]);
        try std.testing.expectEqual(@as(u64, 0x1234_0000), launch.args[3]);
        try std.testing.expectEqual(@as(u64, 4840), launch.args[4]);
    };
    try std.testing.expect(found);
}

test "pre fuse groups the decode q_a, kv_a and indexer wk projections" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    groupLane(&fx);
    var f = fx.fwd(0);
    f.pre_fuse = .on;
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    const grouped_attn = .{ "tf_glm_rmsnorm_6144", "tf_qmmf_group_fp8b_16c", "tf_glm_rmsnorm_2048", "qmmf_bf16", "tf_glm53_kv_write" };
    const grouped_index = .{ "tf_glm53_index_key", "qmmf_bf16", "qmmf_bf16", "tf_glm53_index_q", "tf_glm53_index_scores", "tf_glm53_topk", "tf_glm53_merge" };
    try expectSeq(concat(.{ embed_part, grouped_attn, grouped_index, decode_mla, dense_ffn, grouped_attn, decode_mla, moe_ffn, head_part }));
    var grids: [2][3]u32 = undefined;
    var found: usize = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("tf_qmmf_group_fp8b_16c")) {
        grids[found] = launch.grid;
        try std.testing.expectEqual(@as(u32, 128), launch.block[0]);
        try std.testing.expectEqual(@as(u32, 24576), launch.shared);
        found += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), found);
    try std.testing.expectEqual([3]u32{ 43, 1, 8 }, grids[0]);
    try std.testing.expectEqual([3]u32{ 41, 1, 8 }, grids[1]);
}

test "pre fuse with the side stream: grouped launch stays on main before each fork" {
    var fx: Fixture = undefined;
    var g = geometry();
    g.decode_side = true;
    try fx.init(g, 16384);
    defer fx.deinit();
    groupLane(&fx);
    var f = fx.fwd(0);
    f.pre_fuse = .on;
    const args: forward.Forward.Args = .{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } };

    f.decode_side = false;
    try f.whole(args);
    const baseline = try names_of();
    defer gpa.free(baseline);
    Rec.reset();

    f.decode_side = true;
    try f.whole(args);
    const side_names = try names_of();
    defer gpa.free(side_names);
    const base_sorted = try gpa.dupe(usize, baseline);
    defer gpa.free(base_sorted);
    std.mem.sort(usize, base_sorted, {}, std.sort.asc(usize));
    std.mem.sort(usize, side_names, {}, std.sort.asc(usize));
    try std.testing.expectEqualSlices(usize, base_sorted, side_names);

    const side_stream_handle: usize = 0x51de;
    const fork_event = @intFromPtr(fx.st.fork.?.handle.?);
    var groups: usize = 0;
    var forks: usize = 0;
    var side_kv: usize = 0;
    var side_keys: usize = 0;
    for (Rec.order.items) |step| switch (step.kind) {
        .launch => {
            if (step.what == id("tf_qmmf_group_fp8b_16c") or step.what == id("tf_qmmf_group_fp8b_32c") or step.what == id("tf_qmmf_group_fp8b_64c") or step.what == id("tf_qmmf_group_fp8b_64f")) {
                try std.testing.expectEqual(@as(usize, 0), step.stream);
                try std.testing.expectEqual(groups, forks);
                groups += 1;
            }
            if (step.what == id("tf_glm53_kv_write") or step.what == id("tf_glm53_index_key")) {
                try std.testing.expectEqual(side_stream_handle, step.stream);
                side_kv += @intFromBool(step.what == id("tf_glm53_kv_write"));
                side_keys += @intFromBool(step.what == id("tf_glm53_index_key"));
            }
        },
        .record => if (step.what == fork_event) { forks += 1; },
        .collective, .wait => {},
    };
    try std.testing.expectEqual(@as(usize, 2), groups);
    try std.testing.expectEqual(groups, forks);
    try std.testing.expectEqual(@as(usize, 2), side_kv);
    try std.testing.expectEqual(@as(usize, 1), side_keys);
}

test "prep fuse uses grouped prefill plans and leaves decode selection independent" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    groupLane(&fx);
    var f = fx.fwd(1);
    var pos: [600]usize = undefined;
    var ids: [600]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| { p.* = 4096 + i; t.* = @intCast(i); }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var qmmf_off: usize = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("qmmf_bf16")) { qmmf_off += 1; };
    Rec.reset();
    f.prep_fuse = .on;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var qmmf_on: usize = 0;
    var grouped_count: usize = 0;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("qmmf_bf16")) qmmf_on += 1;
        if (launch.name == id("tf_qmmf_group_fp8b_64f")) {
            try std.testing.expectEqual(@as(u32, 1), launch.grid[2]);
            try std.testing.expectEqual(@as(u32, 49152), launch.shared);
            grouped_count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), grouped_count);
    try std.testing.expectEqual(qmmf_off - 5, qmmf_on);
    Rec.reset();
    f.prep_fuse = .split;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    grouped_count = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("tf_qmmf_group_fp8b_64c")) {
        try std.testing.expectEqual(@as(u32, 8), launch.grid[2]);
        grouped_count += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), grouped_count);
}

test "decode MLA override changes only absorb and expand_v launches" {
    var fx: Fixture = undefined;
    var g = geometry();
    g.decode_mla = true;
    try fx.init(g, 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    const args: forward.Forward.Args = .{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } };
    try std.testing.expectEqual(forward.DecodeMlaMode.off, forward.decodeMlaModeSelected("0"));
    try std.testing.expectEqual(forward.DecodeMlaMode.v1, forward.decodeMlaModeSelected("1"));
    try std.testing.expectEqual(forward.DecodeMlaMode.v2, forward.decodeMlaModeSelected("2"));
    try std.testing.expectEqual(forward.DecodeMlaMode.v3, forward.decodeMlaModeSelected("3"));
    f.decode_mla = false;
    try f.whole(args);
    const baseline = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(baseline);
    Rec.reset();
    f.decode_mla = null;
    try f.whole(args);
    try std.testing.expectEqual(baseline.len, Rec.launches.items.len);
    for (baseline, Rec.launches.items) |before, after| {
        try std.testing.expectEqual(before.nargs, after.nargs);
        try std.testing.expectEqualSlices(u64, &before.args, &after.args);
        try std.testing.expectEqual(before.block, after.block);
        try std.testing.expectEqual(before.shared, after.shared);
        if (before.name == id("tf_glm53_absorb")) {
            try std.testing.expectEqual(id("tf_glm53_absorb_decode"), after.name);
            try std.testing.expectEqual(before.grid, after.grid);
        } else if (before.name == id("tf_glm53_expand_v")) {
            try std.testing.expectEqual(id("tf_glm53_expand_v_decode"), after.name);
            try std.testing.expect(after.grid[2] > before.grid[2]);
        } else {
            try std.testing.expectEqual(before.name, after.name);
            try std.testing.expectEqual(before.grid, after.grid);
        }
    }
    Rec.reset();
    f.decode_mla_v2 = true;
    try f.whole(args);
    var saw_absorb_v1 = false;
    var saw_absorb_v2 = false;
    var saw_expand_v2 = false;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_absorb_decode")) {
            saw_absorb_v1 = true;
            try std.testing.expectEqual([3]u32{ 1, 32, 2 }, launch.grid);
            try std.testing.expectEqual(@as(u32, 128), launch.block[0]);
        } else if (launch.name == id("tf_glm53_absorb_decode_v2")) {
            saw_absorb_v2 = true;
        } else if (launch.name == id("tf_glm53_expand_v_decode_v2")) {
            saw_expand_v2 = true;
            try std.testing.expectEqual([3]u32{ 1, 32, 2 }, launch.grid);
            try std.testing.expectEqual(@as(u32, 128), launch.block[0]);
        }
    }
    try std.testing.expect(saw_absorb_v1 and !saw_absorb_v2 and saw_expand_v2);

    Rec.reset();
    f.decode_mla_v2 = true;
    f.decode_mla_v3 = true;
    try f.whole(args);
    var saw_absorb_v3 = false;
    saw_expand_v2 = false;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_absorb_decode_v3")) {
            saw_absorb_v3 = true;
            try std.testing.expectEqual([3]u32{ 8, 32, 1 }, launch.grid);
            try std.testing.expectEqual(@as(u32, 128), launch.block[0]);
        } else if (launch.name == id("tf_glm53_expand_v_decode_v2")) {
            saw_expand_v2 = true;
        } else if (launch.name == id("tf_glm53_absorb_decode")) {
            return error.UnexpectedV1Absorb;
        }
    }
    try std.testing.expect(saw_absorb_v3 and saw_expand_v2);
}

test "qkv-a sharding adds one packed gather and four assemble copies per attention piece" {
    try std.testing.expect(!forward.shardQkvASwitch());
    var g = geometry();
    g.shard_qkv_a = true;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    const ws = &fx.st.ws;
    try std.testing.expectEqual(@as(u64, g.piece_rows * 1024 * 2), ws.sizes.sh_q);
    try std.testing.expectEqual(@as(u64, g.piece_rows * 320 * 2), ws.sizes.sh_kv);
    try std.testing.expectEqual(@as(u64, g.piece_rows * 2688), ws.sizes.sh_mine);
    try std.testing.expectEqual(@as(u64, 2 * g.piece_rows * 2688), ws.sizes.sh_all);
    const off_ws = state.Workspace.carve(0x2000_0000, geometry(), 16384);
    try std.testing.expectEqual(@as(u64, 0), off_ws.sizes.sh_q);
    try std.testing.expectEqual(@as(u64, 0), off_ws.sizes.sh_kv);
    try std.testing.expectEqual(@as(u64, 0), off_ws.sizes.sh_mine);
    try std.testing.expectEqual(@as(u64, 0), off_ws.sizes.sh_all);
    var f = fx.fwd(0);
    f.shard_qkv_a = true;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    var qkv_gathers: usize = 0;
    for (Rec.gathers.items) |gather| if (gather.phase == .shard_qkv_a) {
        qkv_gathers += 1;
        try std.testing.expectEqual(@as(usize, 2688), gather.bytes);
    };
    try std.testing.expectEqual(@as(usize, 2), qkv_gathers);
    // 2 baseline copies (as with the switch off) + per layer (2 in this fixture): 2 pack + 4 assemble copies.
    try std.testing.expectEqual(@as(usize, 2 + 2 * 6), Rec.copies);
    Rec.reset();
    f.shard_qkv_a = false;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    for (Rec.gathers.items) |gather| try std.testing.expect(gather.phase != .shard_qkv_a);
    try std.testing.expectEqual(@as(usize, 2), Rec.copies);
}

test "QA_SHARD from the environment: one q_a gather per layer, halves norm, no copies, plan bytes match" {
    const on = forward.qaShardSwitch();
    var g = geometry();
    g.qa_shard = on;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    try std.testing.expectEqual(@as(u64, @intCast(if (on) g.piece_rows * 2048 else 0)), fx.st.ws.sizes.sh_q);
    fx.k.rmsnorm_2048_halves = func("tf_glm53_rmsnorm_2048_halves");
    if (on) {
        for (fx.layer_store[0..2]) |*l| {
            l.attn.q_a.n = 1024;
            l.attn.q_a.npad = 1024;
        }
    }
    var f = fx.fwd(1);
    f.qa_shard = on;
    const positions = [_]usize{ 100, 101, 102, 103 };
    try f.whole(.{ .positions = &positions, .tokens = &.{ 7, 8, 9, 10 } });
    var q_gathers: usize = 0;
    for (Rec.gathers.items) |gather| {
        if (gather.phase == .shard_qkv_a) {
            q_gathers += 1;
            try std.testing.expectEqual(@as(usize, 4 * 2048), gather.bytes);
            try std.testing.expectEqual(fx.st.ws.sh_q, gather.mine);
            try std.testing.expectEqual(fx.st.ws.q_a, gather.all);
        }
    }
    try std.testing.expectEqual(@as(usize, if (on) 2 else 0), q_gathers);
    var halves: usize = 0;
    var full: usize = 0;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_rmsnorm_2048_halves")) {
            try std.testing.expectEqual(fx.st.ws.q_a, launch.args[0]);
            try std.testing.expectEqual(fx.st.ws.q_a + 4 * 2048, launch.args[1]);
            try std.testing.expectEqual(fx.st.ws.q_resid, launch.args[4]);
            halves += 1;
        }
        full += @intFromBool(launch.name == id("tf_glm_rmsnorm_2048"));
    }
    try std.testing.expectEqual(@as(usize, if (on) 2 else 0), halves);
    try std.testing.expectEqual(@as(usize, if (on) 0 else 2), full);
    try std.testing.expectEqual(@as(usize, 2), Rec.copies);
    var plan_layers: [2]forward.Layer = undefined;
    try expectPlan(try f.stepPlan(.decode, &positions, 4, &plan_layers));
}

test "QA_SHARD grouped pre-fuse writes its q_a half to sh_q" {
    const on = forward.qaShardSwitch();
    var g = geometry();
    g.qa_shard = on;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.rmsnorm_2048_halves = func("tf_glm53_rmsnorm_2048_halves");
    if (on) {
        for (fx.layer_store[0..2]) |*l| {
            l.attn.q_a.n = 1024;
            l.attn.q_a.npad = 1024;
        }
    }
    groupLane(&fx);
    var f = fx.fwd(0);
    f.qa_shard = on;
    f.pre_fuse = .on;
    const positions = [_]usize{ 100, 101, 102, 103 };
    try f.whole(.{ .positions = &positions, .tokens = &.{ 7, 8, 9, 10 } });
    var grouped: usize = 0;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_qmmf_group_fp8b_16c") or launch.name == id("tf_qmmf_group_fp8b_32c") or launch.name == id("tf_qmmf_group_fp8b_64c") or launch.name == id("tf_qmmf_group_fp8b_64f")) {
            try std.testing.expectEqual(if (on) fx.st.ws.sh_q else fx.st.ws.q_a, launch.group_outs[0]);
            grouped += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), grouped);
    var q_gathers: usize = 0;
    for (Rec.gathers.items) |gather| {
        if (gather.phase == .shard_qkv_a) q_gathers += 1;
    }
    try std.testing.expectEqual(@as(usize, if (on) 2 else 0), q_gathers);
}

test "QA_SHARD prefill gathers each layer's three-row q_a half" {
    const on = forward.qaShardSwitch();
    var g = geometry();
    g.qa_shard = on;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.rmsnorm_2048_halves = func("tf_glm53_rmsnorm_2048_halves");
    if (on) {
        for (fx.layer_store[0..2]) |*l| {
            l.attn.q_a.n = 1024;
            l.attn.q_a.npad = 1024;
        }
    }
    var f = fx.fwd(0);
    f.qa_shard = on;
    const positions = [_]usize{ 100, 101, 102 };
    try f.pieced(.{ .positions = &positions, .tokens = &.{ 7, 8, 9 } });
    var q_gathers: usize = 0;
    for (Rec.gathers.items) |gather| {
        if (gather.phase == .shard_qkv_a) {
            q_gathers += 1;
            try std.testing.expectEqual(@as(usize, 3 * 2048), gather.bytes);
        }
    }
    try std.testing.expectEqual(@as(usize, if (on) 2 else 0), q_gathers);
    var plan_layers: [2]forward.Layer = undefined;
    // Prefill runs the head on its last row only: head_rows 1, as the other prefill plan test.
    try expectPlan(try f.stepPlan(.prefill, &positions, 1, &plan_layers));
}

test "decode side stream splits key and shared expert launches and rejoins before main consumers" {
    var fx: Fixture = undefined;
    var g = geometry();
    g.decode_side = true;
    try fx.init(g, 16384);
    defer fx.deinit();
    try std.testing.expectEqual(false, forward.decodeSideSelected(null));
    try std.testing.expectEqual(false, forward.decodeSideSelected("0"));
    try std.testing.expectEqual(true, forward.decodeSideSelected("1"));
    var f = fx.fwd(0);
    f.decode_side = false;
    const args: forward.Forward.Args = .{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } };
    try f.whole(args);
    const baseline = try names_of();
    defer gpa.free(baseline);
    try std.testing.expectEqualSlices(usize, concat(.{ embed_part, attn_head, index_part, decode_mla, dense_ffn, attn_head, decode_mla, moe_ffn, head_part }), baseline);
    for (Rec.order.items) |step| try std.testing.expect(step.stream == 0);

    Rec.reset();
    f.decode_side = true;
    try f.whole(args);
    // Side on launches exactly the kernels of side off, only on two streams (review of 2b9a090e: the low-rank
    // partial/reduce were missing): equal as multisets.
    {
        const side_names = try names_of();
        defer gpa.free(side_names);
        const base_sorted = try gpa.dupe(usize, baseline);
        defer gpa.free(base_sorted);
        std.mem.sort(usize, base_sorted, {}, std.sort.asc(usize));
        std.mem.sort(usize, side_names, {}, std.sort.asc(usize));
        try std.testing.expectEqualSlices(usize, base_sorted, side_names);
    }
    const side_stream_handle: usize = 0x51de;
    var main_count: usize = 0;
    var side_count: usize = 0;
    var side_keys: usize = 0;
    var side_kv: usize = 0;
    var side_shared: usize = 0;
    var side_collectives: usize = 0;
    var launch_index: usize = 0;
    for (Rec.order.items) |step| switch (step.kind) {
        .launch => {
            const launch = Rec.launches.items[launch_index];
            launch_index += 1;
            if (step.stream == side_stream_handle) {
                side_count += 1;
                side_keys += @intFromBool(step.what == id("tf_glm53_index_key"));
                side_kv += @intFromBool(step.what == id("tf_glm53_kv_write"));
                side_shared += @intFromBool(step.what == id("qmmf_f32") and launch.args[4] == fx.st.ws.shared_out);
            } else {
                try std.testing.expectEqual(@as(usize, 0), step.stream);
                main_count += 1;
            }
        },
        .collective => {
            try std.testing.expect(step.stream != side_stream_handle);
            side_collectives += @intFromBool(step.stream == side_stream_handle);
        },
        .record, .wait => {},
    };
    try std.testing.expect(main_count > 0 and side_count > 0);
    try std.testing.expectEqual(@as(usize, 1), side_keys);
    // Both fixture layers write KV once; the one indexer key launch belongs only to the full-indexer layer.
    try std.testing.expectEqual(@as(usize, 2), side_kv);
    try std.testing.expectEqual(@as(usize, 1), side_shared);
    try std.testing.expectEqual(@as(usize, 0), side_collectives);
    // The side writes its own attention scratch and cache planes; the main q, sparse query, route and expert writes
    // occupy separate carved intervals. The shared output and MLP scratch do not overlap main route/expert scratch.
    const ws = &fx.st.ws;
    const side_ranges = [_][2]u64{
        .{ ws.kv, ws.sizes.kv }, .{ ws.ik, ws.sizes.ik }, .{ ws.shared_out, ws.sizes.shared_out },
        .{ ws.mlp_gate_up, ws.sizes.mlp_gate_up }, .{ ws.mlp_act, ws.sizes.mlp_act },
        .{ ws.lane_part_side, ws.sizes.lane_part_side },
        .{ fx.st.latentAt(0), 2 * 512 * 2 }, .{ fx.st.ropeAt(0), 2 * 64 * 2 }, .{ fx.st.indexAt(0), 2 * 128 * 2 },
    };
    const main_attention_ranges = [_][2]u64{
        .{ ws.q_a, ws.sizes.q_a }, .{ ws.q_resid, ws.sizes.q_resid }, .{ ws.q, ws.sizes.q },
        .{ ws.iw, ws.sizes.iw }, .{ ws.iq, ws.sizes.iq }, .{ ws.qi, ws.sizes.qi }, .{ ws.lane_part, ws.sizes.lane_part },
    };
    const main_moe_ranges = [_][2]u64{
        .{ ws.router_logits, ws.sizes.router_logits }, .{ ws.router_part, ws.sizes.router_part },
        .{ ws.picks, ws.sizes.picks }, .{ ws.wts, ws.sizes.wts }, .{ ws.routed_out, ws.sizes.routed_out },
        .{ ws.experts, ws.sizes.experts }, .{ ws.lowrank_mid, ws.sizes.lowrank_mid }, .{ ws.gathered, ws.sizes.gathered },
    };
    for ([_][2]u64{ side_ranges[0], side_ranges[1], side_ranges[5], side_ranges[6], side_ranges[7], side_ranges[8] }) |side_range| for (main_attention_ranges) |main_range| {
        try std.testing.expect(side_range[0] + side_range[1] <= main_range[0] or main_range[0] + main_range[1] <= side_range[0]);
    };
    for (side_ranges[2..6]) |side_range| for (main_moe_ranges) |main_range| {
        try std.testing.expect(side_range[0] + side_range[1] <= main_range[0] or main_range[0] + main_range[1] <= side_range[0]);
    };
    try std.testing.expect(side_count < Rec.launches.items.len);
}

test "route compact switch dispatches the compact top-k kernel and skips route copies" {
    var fx: Fixture = undefined;
    var g = geometry();
    g.decode_side = true;
    try fx.init(g, 16384);
    defer fx.deinit();
    try std.testing.expect(!forward.routeCompactSelected(null));
    try std.testing.expect(!forward.routeCompactSelected("0"));
    try std.testing.expect(forward.routeCompactSelected("1"));
    var f = fx.fwd(0);
    f.decode_side = true;
    f.route_compact = forward.routeCompactSelected("1");
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    try expectSeq(concat(.{ embed_part, attn_head, index_part, decode_mla, dense_ffn, attn_head, decode_mla, moe_ffn_compact, head_part }));
    try std.testing.expectEqual(@as(usize, 0), Rec.copies);
    var compact_launches: usize = 0;
    var legacy_launches: usize = 0;
    for (Rec.launches.items) |launch| {
        compact_launches += @intFromBool(launch.name == id("tf_glm_topk_compact"));
        legacy_launches += @intFromBool(launch.name == id("tf_glm_topk"));
    }
    try std.testing.expectEqual(@as(usize, 1), compact_launches);
    try std.testing.expectEqual(@as(usize, 0), legacy_launches);
}

test "L2PF forks at sites f and n of each decode layer and joins before the head" {
    var fx: Fixture = undefined;
    var g = geometry();
    g.decode_side = true;
    try fx.init(g, 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.decode_side = true;
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    const before = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(before);
    const gathers_before = try gpa.dupe(Gather, Rec.gathers.items);
    defer gpa.free(gathers_before);
    const copies_before = Rec.copies;
    const copy_log_before = try gpa.dupe(CopyRecord, Rec.copy_log.items);
    defer gpa.free(copy_log_before);

    Rec.reset();
    var table_rows: std.ArrayList([2]i64) = .empty;
    defer table_rows.deinit(gpa);
    const layers_slice = fx.model.layers;
    var regions: std.ArrayList(forward.l2pf_mod.Region) = .empty;
    defer regions.deinit(gpa);
    const source_ptrs = [_]u64{ 0x8100_0100, 0x8100_0000, 0x9100_1100, 0x9100_1000, 0x9100_3100, 0x9100_3000, 0x9100_2100, 0x9100_2000, 0x8300_0000 };
    for (source_ptrs) |ptr| try regions.append(gpa, .{ .ptr = ptr, .len = 16 << 20 });
    const spans = try gpa.alloc(forward.l2pf_mod.Span, layers_slice.len * 2);
    try forward.l2pf_mod.table(gpa, regions.items, layers_slice, 12 << 20, &table_rows, spans);
    const side = try cuda.Stream.init(&driver, true);
    const fork_ev = try cuda.Event.init(&driver, false);
    const join_ev = try cuda.Event.init(&driver, false);
    const row_buffer = try cuda.DeviceBuffer.alloc(&driver, table_rows.items.len * 16);
    const pieces = try gpa.alloc(u64, spans.len);
    @memset(pieces, 0);
    for (spans, pieces) |span, *count| {
        const first: usize = span.at;
        for (table_rows.items[first .. first + @as(usize, span.rows)]) |row| count.* += (@as(u64, @intCast(row[1])) + 32767) / 32768;
    }
    var pf: forward.l2pf_mod.L2pf = .{ .gpa = gpa, .o = .{ .on = true }, .side = side, .fork_ev = fork_ev, .join_ev = join_ev, .mods = undefined, .mods_loaded = 0, .segments = func("l2pf_segments"), .paced = func("l2pf_paced"), .rows = row_buffer, .spans = spans, .pieces = pieces };
    defer pf.deinit();
    f.l2pf = &pf;
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });

    var pf_count: usize = 0;
    var launch_index: usize = 0;
    var record_positions: std.ArrayList(usize) = .empty;
    defer record_positions.deinit(gpa);
    var attn_collectives: std.ArrayList(usize) = .empty;
    defer attn_collectives.deinit(gpa);
    var moe_collectives: std.ArrayList(usize) = .empty;
    defer moe_collectives.deinit(gpa);
    for (Rec.order.items, 0..) |step, order_index| switch (step.kind) {
        .launch => {
            if (step.stream == @intFromPtr(side.handle)) {
                const launch = Rec.launches.items[launch_index];
                try std.testing.expectEqual(id("l2pf_paced"), launch.name);
                pf_count += 1;
                try std.testing.expect(order_index >= 2);
                const record = Rec.order.items[order_index - 2];
                const wait = Rec.order.items[order_index - 1];
                try std.testing.expectEqual(@as(@FieldType(Step, "kind"), .record), record.kind);
                try std.testing.expectEqual(@as(usize, 0), record.stream);
                try std.testing.expectEqual(@as(@FieldType(Step, "kind"), .wait), wait.kind);
                try std.testing.expectEqual(@intFromPtr(side.handle), wait.stream);
                try std.testing.expectEqual(record.what, wait.what);
            }
            launch_index += 1;
        },
        .record => if (step.what == @intFromPtr(fork_ev.handle)) try record_positions.append(gpa, order_index),
        .collective => {
            if (step.what == @backingInt(forward.Phase.attention_output)) try attn_collectives.append(gpa, order_index);
            if (step.what == @backingInt(forward.Phase.moe_partial)) try moe_collectives.append(gpa, order_index);
        },
        .wait => {},
    };
    try std.testing.expectEqual(@as(usize, 3), pf_count);
    const expected_spans = [_]usize{ 1, 0, 3 };
    var expected_pf: usize = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("l2pf_paced")) {
        try std.testing.expectEqual(row_buffer.ptr + @as(u64, pf.spans[expected_spans[expected_pf]].at) * 16, launch.args[0]);
        expected_pf += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), record_positions.items.len);
    try std.testing.expectEqual(@as(usize, 2), attn_collectives.items.len);
    try std.testing.expectEqual(@as(usize, 2), moe_collectives.items.len);
    try std.testing.expect(record_positions.items[0] < attn_collectives.items[0]);
    try std.testing.expect(record_positions.items[1] < moe_collectives.items[0]);
    try std.testing.expect(record_positions.items[2] < attn_collectives.items[1]);
    const join_wait = for (Rec.order.items, 0..) |step, i| {
        if (step.kind == .wait and step.what == @intFromPtr(join_ev.handle) and step.stream == 0) break i;
    } else return error.MissingL2pfJoin;
    const first_head = for (Rec.order.items, 0..) |step, i| {
        if (step.kind == .launch and step.what == id("tf_glm53_top1_f32")) break i;
    } else return error.MissingHeadLaunch;
    try std.testing.expect(join_wait < first_head);
    var kept: std.ArrayList(Launch) = .empty;
    defer kept.deinit(gpa);
    for (Rec.launches.items) |launch| if (launch.name != id("l2pf_paced") and launch.name != id("l2pf_segments")) {
        try kept.append(gpa, launch);
    };
    try std.testing.expectEqualSlices(Launch, before, kept.items);
    try std.testing.expectEqualSlices(Gather, gathers_before, Rec.gathers.items);
    try std.testing.expectEqual(copies_before, Rec.copies);
    try std.testing.expectEqualSlices(CopyRecord, copy_log_before, Rec.copy_log.items);

    Rec.reset();
    pf.o.pace_gbps = 0;
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    var segments_count: usize = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("l2pf_segments")) {
        const wanted = @min(pf.o.ctas, @max(@as(u32, 1), @as(u32, @intCast((pf.pieces[expected_spans[segments_count]] + 127) / 128))));
        try std.testing.expectEqual(wanted, launch.grid[0]);
        segments_count += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), segments_count);

    Rec.reset();
    try f.pieced(.{ .positions = &.{ 100, 101 }, .tokens = &.{ 7, 8 } });
    for (Rec.launches.items) |launch| try std.testing.expect(launch.name != id("l2pf_paced") and launch.name != id("l2pf_segments"));
    for (Rec.order.items) |step| try std.testing.expect(step.kind != .record and step.kind != .wait);
}

test "dflash taps: decode window adds df_tap_glm after the tapped layer only" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const base = fx.fwd(0);
    try base.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    const baseline = try names_of();
    defer gpa.free(baseline);
    Rec.reset();
    var tapped = base;
    tapped.taps = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{1}, .from = 100, .cap_rows = 32 };
    try tapped.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    const got = try names_of();
    defer gpa.free(got);
    const tap_id = id("df_tap_glm");
    const tail: usize = head_part.len;
    var tap_at: ?usize = null;
    for (got, 0..) |launch, i| if (launch == tap_id) {
        try std.testing.expect(tap_at == null);
        tap_at = i;
    };
    const at = tap_at orelse return error.MissingTap;
    try std.testing.expectEqual(baseline.len + 1, got.len);
    try std.testing.expectEqualSlices(usize, baseline[0 .. baseline.len - tail], got[0..at]);
    try std.testing.expectEqualSlices(usize, baseline[baseline.len - tail ..], got[at + 1 ..]);
    const launch = Rec.launches.items[at];
    try std.testing.expectEqual(@as(u32, 2), launch.grid[0]);
    try std.testing.expectEqual(fx.st.ws.x, launch.args[0]);
    try std.testing.expectEqual(@as(u64, 0x9b00_0000), launch.args[1]);
}

const IngestProbe = struct {
    forks: usize = 0,
    joins: usize = 0,
    fork_at: usize = 0,
    join_at: usize = 0,
    positions: u64 = 0,
    rows: usize = 0,

    fn fork(ctx: *anyopaque, _: cuda.Stream, positions: u64, rows: usize) anyerror!void {
        const probe: *IngestProbe = @ptrCast(@alignCast(ctx));
        probe.forks += 1;
        probe.fork_at = Rec.launches.items.len;
        probe.positions = positions;
        probe.rows = rows;
    }

    fn join(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
        const probe: *IngestProbe = @ptrCast(@alignCast(ctx));
        probe.joins += 1;
        probe.join_at = Rec.launches.items.len;
    }
};

test "dflash taps: TF_GLM53_DRAFT_INGEST_SIDE forks after the last tap layer and joins after the head" {
    const on = forward.draftIngestSideSwitch();
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const positions = &.{ 100, 3000 };
    const tokens = &.{ 7, 77441 };
    var f = fx.fwd(0);
    f.taps = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{ 0, 1 }, .from = 100, .cap_rows = 32, .ingest = null };
    try f.whole(.{ .positions = positions, .tokens = tokens });
    const expected = try names_of();
    defer gpa.free(expected);

    Rec.reset();
    var probe: IngestProbe = .{};
    f.taps = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{ 0, 1 }, .from = 100, .cap_rows = 32, .ingest = if (on) .{ .ctx = &probe, .fork_fn = IngestProbe.fork, .join_fn = IngestProbe.join } else null };
    try f.whole(.{ .positions = positions, .tokens = tokens });
    const actual = try names_of();
    defer gpa.free(actual);
    try std.testing.expectEqualSlices(usize, expected, actual);
    if (on) {
        var tap_count: usize = 0;
        var second_tap_at: ?usize = null;
        for (Rec.launches.items, 0..) |launch, i| if (launch.name == id("df_tap_glm")) {
            if (tap_count == 1) second_tap_at = i;
            tap_count += 1;
        };
        try std.testing.expectEqual(@as(usize, 2), tap_count);
        try std.testing.expectEqual(@as(usize, 1), probe.forks);
        try std.testing.expectEqual(@as(usize, 1), probe.joins);
        try std.testing.expectEqual((second_tap_at orelse return error.MissingTap) + 1, probe.fork_at);
        try std.testing.expectEqual(Rec.launches.items.len, probe.join_at);
        try std.testing.expectEqual(fx.st.ws.positions, probe.positions);
        try std.testing.expectEqual(@as(usize, 2), probe.rows);
    } else {
        try std.testing.expectEqual(@as(usize, 0), probe.forks);
        try std.testing.expectEqual(@as(usize, 0), probe.joins);
    }

    Rec.reset();
    probe = .{};
    try f.pieced(.{ .positions = &.{ 100, 101, 102 }, .tokens = &.{ 7, 8, 9 } });
    try std.testing.expectEqual(@as(usize, 0), probe.forks);
    try std.testing.expectEqual(@as(usize, 0), probe.joins);
}

test "dflash taps: prefill taps only the last window rows" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 8192);
    defer fx.deinit();
    const count = 3000;
    const positions = try gpa.alloc(usize, count);
    defer gpa.free(positions);
    const tokens = try gpa.alloc(u32, count);
    defer gpa.free(tokens);
    for (positions, 0..) |*p, i| p.* = i;
    @memset(tokens, 7);
    var f = fx.fwd(0);
    f.taps = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{0}, .from = 952, .cap_rows = 2048 };
    try f.pieced(.{ .positions = positions, .tokens = tokens });
    var count_taps: usize = 0;
    for (Rec.launches.items) |launch| if (launch.name == id("df_tap_glm")) {
        count_taps += 1;
        try std.testing.expectEqual(@as(u32, 2048), launch.grid[0]);
        try std.testing.expectEqual(fx.st.ws.x + 952 * 12288, launch.args[0]);
    };
    try std.testing.expectEqual(@as(usize, 1), count_taps);
}

test "runtime ablation launches only at active sites immediately after each residual reduce" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var ab: @import("ablate").Ablation = .{
        .layers = 2,
        .buf = undefined,
        .present = blk: {
            var sites: [78][2]bool = @splat(.{ false, false });
            sites[1] = .{ true, true };
            break :blk sites;
        },
        .digest = @splat(0),
    };
    ab.buf.ptr = 0xa000_0000;
    fx.k.ablate = func("tf_glm53_ablate_6144");
    var f = fx.fwd(0);
    f.ablate = &ab;
    try f.whole(.{ .positions = &.{ 100, 3000 }, .tokens = &.{ 7, 77441 } });
    var count: usize = 0;
    for (Rec.launches.items, 0..) |launch, i| {
        if (launch.name != id("tf_glm53_ablate_6144")) continue;
        count += 1;
        try std.testing.expect(i > 0);
        const prior = Rec.launches.items[i - 1].name;
        try std.testing.expect(prior == id("tf_glm53_residual_rmsnorm_6144") or prior == id("tf_glm53_moe_residual_add"));
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "head completion gates the captured token and status words" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.coll.gate_result_fn = Rec.gate;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    try std.testing.expectEqual(@as(usize, 1), Rec.gate_calls);
    try std.testing.expectEqual(f.st.ws.roce_token, Rec.gate_token);
    try std.testing.expectEqual(f.st.ws.roce_status, Rec.gate_status);
}

test "prefill chunk: gather-on-prefill launch order (no merge, no filter) and stage fills before kv write" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(1);
    var pos: [600]usize = undefined;
    var ids: [600]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = 4096 + i;
        t.* = @intCast(i);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    // 600 rows: one piece, 5 score blocks of 128, 3 expert calls of 256.
    try expectSeq(concat(.{ embed_part, attn_head, index_part, score_block, score_block, score_block, score_block, prefill_mla, dense_ffn, attn_head, prefill_mla, moe_route, experts_run, experts_run, experts_run, moe_rest, head_part }));
    // Two stage fetches a layer (latent, rope), each of ceil(4096/2) peer slots, into the layer's stage buffer.
    var fetches: usize = 0;
    for (Rec.gathers.items) |g| if (g.phase == .stage_fetch) {
        try std.testing.expect(g.send_recv);
        const layer = fetches / 2;
        const latent = fetches % 2 == 0;
        try std.testing.expectEqual(if (latent) fx.st.latentAt(layer) else fx.st.ropeAt(layer), g.mine);
        try std.testing.expectEqual(if (latent) state.stageLatent(&fx.st.stage, layer) else state.stageRope(&fx.st.stage, layer), g.all);
        try std.testing.expectEqual(@as(usize, 2048) * (if (latent) @as(usize, 1024) else 128), g.bytes);
        try std.testing.expectEqual(g.bytes, g.recv_bytes);
        fetches += 1;
    };
    try std.testing.expectEqual(@as(usize, 4), fetches);
}

test "prompt experts switch launches prompt GEMMs in prefill and leaves decode on the grouped path" {
    try std.testing.expect(!forward.promptExpertsSelected(null));
    try std.testing.expect(!forward.promptExpertsSelected("0"));
    try std.testing.expect(forward.promptExpertsSelected("1"));
    try std.testing.expect(!forward.promptExpertsSelected("true"));
    var g = geometry();
    g.layers = 2;
    g.full_index_layers = 0;
    g.prompt_experts = true;
    const pos = [_]usize{0};
    const ids = [_]u32{0};
    Rec.reset();
    var fx: Fixture = undefined;
    try fx.init(g, 4096);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(0, true, false);
    fx.layer_store[1] = glmLayer(1, false, false);
    fx.model.layers = fx.layer_store[0..2];
    var f = fx.fwd(0);
    f.prompt_experts = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var prompt_gateup = false;
    var prompt_down = false;
    var decode_group = false;
    for (Rec.launches.items) |launch| {
        prompt_gateup = prompt_gateup or launch.name == id("tf_exl3_prompt_gu_e4m3");
        prompt_down = prompt_down or launch.name == id("tf_exl3_prompt_down_e4m3");
        decode_group = decode_group or launch.name == id("exl3_group");
    }
    try std.testing.expect(prompt_gateup and prompt_down and !decode_group);

    Rec.reset();
    var e: engine.Engine = .{ .gpa = gpa, .f = f, .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer e.deinit();
    var out: [1]u32 = undefined;
    try e.verify(&ids, &out);
    prompt_gateup = false;
    prompt_down = false;
    decode_group = false;
    for (Rec.launches.items) |launch| {
        prompt_gateup = prompt_gateup or launch.name == id("tf_exl3_prompt_gu_e4m3");
        prompt_down = prompt_down or launch.name == id("tf_exl3_prompt_down_e4m3");
        decode_group = decode_group or launch.name == id("exl3_group");
    }
    try std.testing.expect(!prompt_gateup and !prompt_down and decode_group);
}

test "prompt experts =2 swaps in only the 128-member plan's gate/up and down launches; =1 and off are unchanged" {
    const P = forward.PromptExperts;
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected(null));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected("0"));
    try std.testing.expectEqual(P.tiles, forward.promptExpertsModeSelected("1"));
    try std.testing.expectEqual(P.wide, forward.promptExpertsModeSelected("2"));
    try std.testing.expectEqual(P.pmx, forward.promptExpertsModeSelected("3"));
    try std.testing.expectEqual(P.p5a, forward.promptExpertsModeSelected("10"));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected("11"));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected("9"));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected("010"));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected("true"));
    try std.testing.expectEqual(P.off, forward.promptExpertsModeSelected(""));
    try std.testing.expect(forward.promptExpertsSelected("2"));
    var g = geometry();
    g.layers = 2;
    g.full_index_layers = 0;
    g.prompt_experts = true;
    // 64 rows: 512 pairs, so the 64-pair and 128-pair plans size different item grids (264 vs 260).
    var pos: [64]usize = undefined;
    var ids: [64]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    Rec.reset();
    var fx: Fixture = undefined;
    try fx.init(g, 4096);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(0, true, false);
    fx.layer_store[1] = glmLayer(1, false, false);
    fx.model.layers = fx.layer_store[0..2];
    var f = fx.fwd(0);
    f.prompt_experts = true;
    f.prompt_experts_wide = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const tiles = try names_of();
    defer gpa.free(tiles);
    var tiles_gu: ?Launch = null;
    var tiles_down: ?Launch = null;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_exl3_prompt_gu_e4m3")) tiles_gu = launch;
        if (launch.name == id("tf_exl3_prompt_down_e4m3")) tiles_down = launch;
        try std.testing.expect(launch.name != id("tf_exl3_prompt_wup_e4m3") and launch.name != id("tf_exl3_prompt_gu_s_e4m3") and launch.name != id("tf_exl3_prompt_wdown_e4m3"));
    }
    // =1 as before: 256 threads, static shared only, (width / 128, items) and (dims / 256, items) at tile 64.
    const pairs = 64 * state.routed_slots;
    const items64: u32 = @intCast(cuda.grouped.maxItems(pairs, 256, cuda.exl3_experts.prompt.tile));
    const items128: u32 = @intCast(cuda.grouped.maxItems(pairs, 256, cuda.exl3_experts.prompt.wide_tile));
    try std.testing.expect(items64 != items128);
    try std.testing.expectEqual([3]u32{ 8, items64, 1 }, tiles_gu.?.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, tiles_gu.?.block);
    try std.testing.expectEqual(@as(u32, 0), tiles_gu.?.shared);
    try std.testing.expectEqual([3]u32{ 24, items64, 1 }, tiles_down.?.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, tiles_down.?.block);

    Rec.reset();
    f.prompt_experts_wide = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    // =2: =1's sequence with gate/up -> (=1 body for <= 64-member items, wide up, wide gate) and
    // down -> (=1 body for <= 64-member items, wide down); every other launch unchanged.
    var wide_want: std.ArrayList(usize) = .empty;
    defer wide_want.deinit(gpa);
    var swapped: usize = 0;
    for (tiles) |kernel| {
        if (kernel == id("tf_exl3_prompt_gu_e4m3")) {
            try wide_want.appendSlice(gpa, &.{ id("tf_exl3_prompt_gu_s_e4m3"), id("tf_exl3_prompt_wup_e4m3"), id("tf_exl3_prompt_wgate_e4m3") });
            swapped += 1;
        } else if (kernel == id("tf_exl3_prompt_down_e4m3")) {
            try wide_want.appendSlice(gpa, &.{ id("tf_exl3_prompt_down_s_e4m3"), id("tf_exl3_prompt_wdown_e4m3") });
            swapped += 1;
        } else try wide_want.append(gpa, kernel);
    }
    try std.testing.expectEqual(@as(usize, 2), swapped); // one MoE layer: one gate/up and one down
    try expectSeq(wide_want.items);
    const wide_smem = cuda.exl3_experts.prompt.wide_smem;
    for (Rec.launches.items) |launch| {
        const want_launch: ?struct { grid: [3]u32, threads: u32, shared: u32 } =
            if (launch.name == id("tf_exl3_prompt_gu_s_e4m3")) .{ .grid = .{ 8, items128, 1 }, .threads = 256, .shared = 0 }
            else if (launch.name == id("tf_exl3_prompt_wup_e4m3") or launch.name == id("tf_exl3_prompt_wgate_e4m3")) .{ .grid = .{ 8, items128, 1 }, .threads = 256, .shared = wide_smem }
            else if (launch.name == id("tf_exl3_prompt_down_s_e4m3")) .{ .grid = .{ 24, items128, 1 }, .threads = 256, .shared = 0 }
            else if (launch.name == id("tf_exl3_prompt_wdown_e4m3")) .{ .grid = .{ 48, items128, 1 }, .threads = 256, .shared = wide_smem }
            else null;
        if (want_launch) |wl| {
            try std.testing.expectEqual(wl.grid, launch.grid);
            try std.testing.expectEqual([3]u32{ wl.threads, 1, 1 }, launch.block);
            try std.testing.expectEqual(wl.shared, launch.shared);
        }
    }

    // =3: same 128-member plan, small-item =1 gate/up, fused PMX gate/up, then the =2 down sequence.
    Rec.reset();
    f.prompt_experts_pmx = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var pmx_want: std.ArrayList(usize) = .empty;
    defer pmx_want.deinit(gpa);
    for (tiles) |kernel| {
        if (kernel == id("tf_exl3_prompt_gu_e4m3")) {
            try pmx_want.appendSlice(gpa, &.{ id("tf_exl3_prompt_gu_s_e4m3"), id("tf_glm53_pmx_gu_e4m3") });
        } else if (kernel == id("tf_exl3_prompt_down_e4m3")) {
            try pmx_want.appendSlice(gpa, &.{ id("tf_exl3_prompt_down_s_e4m3"), id("tf_exl3_prompt_wdown_e4m3") });
        } else try pmx_want.append(gpa, kernel);
    }
    try expectSeq(pmx_want.items);
    var pmx_seen = false;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_pmx_gu_e4m3")) {
            pmx_seen = true;
            try std.testing.expectEqual([3]u32{ 8, items128, 1 }, launch.grid);
            try std.testing.expectEqual([3]u32{ 512, 1, 1 }, launch.block);
            try std.testing.expectEqual(@as(u32, @intCast(cuda.exl3_experts.pmx.gu_smem)), launch.shared);
        }
    }
    try std.testing.expect(pmx_seen);

    Rec.reset();
    f.prompt_experts_p5a = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var p5a_want: std.ArrayList(usize) = .empty;
    defer p5a_want.deinit(gpa);
    for (pmx_want.items) |kernel| {
        if (kernel == id("tf_exl3_prompt_gu_s_e4m3")) {
            try p5a_want.append(gpa, id("tf_glm53_p5a_gu_e4m3"));
        } else if (kernel == id("tf_exl3_prompt_down_s_e4m3")) {
            try p5a_want.append(gpa, id("tf_glm53_p5a_down_e4m3"));
        } else try p5a_want.append(gpa, kernel);
    }
    try expectSeq(p5a_want.items);
    for (Rec.launches.items) |launch| {
        const want_launch: ?struct { grid: [3]u32, threads: u32 } =
            if (launch.name == id("tf_glm53_p5a_gu_e4m3")) .{ .grid = .{ 8, items128, 1 }, .threads = 256 }
            else if (launch.name == id("tf_glm53_p5a_down_e4m3")) .{ .grid = .{ 24, items128, 1 }, .threads = 256 }
            else null;
        if (want_launch) |wl| {
            try std.testing.expectEqual(wl.grid, launch.grid);
            try std.testing.expectEqual([3]u32{ wl.threads, 1, 1 }, launch.block);
            try std.testing.expectEqual(@as(u32, 0), launch.shared);
        }
    }
    f.prompt_experts_p5a = false;

    // Off: the grouped decode kernels, no prompt GEMM of any width; the mode flags alone select nothing.
    Rec.reset();
    f.prompt_experts = false;
    f.prompt_experts_pmx = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var any_prompt = false;
    var decode_group = false;
    for (Rec.launches.items) |launch| {
        for ([_]usize{ id("tf_exl3_prompt_gu_e4m3"), id("tf_exl3_prompt_down_e4m3"), id("tf_exl3_prompt_gu_s_e4m3"), id("tf_exl3_prompt_down_s_e4m3"), id("tf_exl3_prompt_wup_e4m3"), id("tf_exl3_prompt_wgate_e4m3"), id("tf_exl3_prompt_wdown_e4m3"), id("tf_glm53_pmx_gu_e4m3"), id("tf_glm53_p5a_gu_e4m3"), id("tf_glm53_p5a_down_e4m3") }) |k| {
            any_prompt = any_prompt or launch.name == k;
        }
        decode_group = decode_group or launch.name == id("exl3_group");
    }
    try std.testing.expect(!any_prompt and decode_group);
}

test "prompt experts: flags from promptExpertsFlags launch each mode's kernels (=10 launches p5a)" {
    var g = geometry();
    g.layers = 2;
    g.full_index_layers = 0;
    g.prompt_experts = true;
    var pos: [64]usize = undefined;
    var ids: [64]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    var fx: Fixture = undefined;
    try fx.init(g, 4096);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(0, true, false);
    fx.layer_store[1] = glmLayer(1, false, false);
    fx.model.layers = fx.layer_store[0..2];

    var pmx_seq: ?[]usize = null;
    var p5a_seq: ?[]usize = null;
    defer if (pmx_seq) |pmx_list| gpa.free(pmx_list);
    defer if (p5a_seq) |p5a_list| gpa.free(p5a_list);
    const modes = [_]forward.PromptExperts{ .tiles, .wide, .pmx, .p5a };
    for (modes) |mode| {
        const flags = forward.promptExpertsFlags(mode);
        var f = fx.fwd(0);
        f.prompt_experts = flags.on;
        f.prompt_experts_wide = flags.wide;
        f.prompt_experts_pmx = flags.pmx;
        f.prompt_experts_p5a = flags.p5a;
        try f.validateSetup();
        Rec.reset();
        try f.pieced(.{ .positions = &pos, .tokens = &ids });
        const launch_seq = try names_of();
        var keep_launch_seq = false;
        errdefer if (!keep_launch_seq) gpa.free(launch_seq);
        var seen_tiles_gu = false;
        var seen_wide_up = false;
        var seen_small_gu = false;
        var seen_small_down = false;
        var seen_pmx = false;
        var seen_p5a_gu = false;
        var seen_p5a_down = false;
        var seen_wide_down = false;
        for (launch_seq) |kernel| {
            seen_tiles_gu = seen_tiles_gu or kernel == id("tf_exl3_prompt_gu_e4m3");
            seen_wide_up = seen_wide_up or kernel == id("tf_exl3_prompt_wup_e4m3");
            seen_small_gu = seen_small_gu or kernel == id("tf_exl3_prompt_gu_s_e4m3");
            seen_small_down = seen_small_down or kernel == id("tf_exl3_prompt_down_s_e4m3");
            seen_pmx = seen_pmx or kernel == id("tf_glm53_pmx_gu_e4m3");
            seen_p5a_gu = seen_p5a_gu or kernel == id("tf_glm53_p5a_gu_e4m3");
            seen_p5a_down = seen_p5a_down or kernel == id("tf_glm53_p5a_down_e4m3");
            seen_wide_down = seen_wide_down or kernel == id("tf_exl3_prompt_wdown_e4m3");
        }
        switch (mode) {
            .tiles => try std.testing.expect(seen_tiles_gu),
            .wide => try std.testing.expect(seen_wide_up),
            .pmx => {
                try std.testing.expect(seen_small_gu and seen_pmx);
                try std.testing.expect(!seen_p5a_gu and !seen_p5a_down);
                pmx_seq = launch_seq;
                keep_launch_seq = true;
            },
            .p5a => {
                try std.testing.expect(seen_p5a_gu and seen_p5a_down and seen_pmx and seen_wide_down);
                try std.testing.expect(!seen_small_gu and !seen_small_down and !seen_tiles_gu);
                p5a_seq = launch_seq;
                keep_launch_seq = true;
            },
            .off => unreachable,
        }
        if (!keep_launch_seq) gpa.free(launch_seq);
    }
    var expected: std.ArrayList(usize) = .empty;
    defer expected.deinit(gpa);
    for (pmx_seq.?) |kernel| {
        if (kernel == id("tf_exl3_prompt_gu_s_e4m3")) {
            try expected.appendSlice(gpa, &.{ id("tf_glm53_p5a_gu_e4m3") });
        } else if (kernel == id("tf_exl3_prompt_down_s_e4m3")) {
            try expected.appendSlice(gpa, &.{ id("tf_glm53_p5a_down_e4m3") });
        } else try expected.append(gpa, kernel);
    }
    try std.testing.expectEqualSlices(usize, expected.items, p5a_seq.?);

    var invalid = fx.fwd(0);
    invalid.prompt_experts = true;
    invalid.prompt_experts_wide = true;
    invalid.prompt_experts_pmx = false;
    invalid.prompt_experts_p5a = true;
    try std.testing.expectError(error.PromptExpertsP5aNeedsPmx, invalid.validateSetup());
}

test "prompt MLA opt-in changes only prefill absorb and expand launches" {
    try std.testing.expectEqual(forward.PromptMlaMode.off, forward.promptMlaSelected(null));
    try std.testing.expectEqual(forward.PromptMlaMode.off, forward.promptMlaSelected("0"));
    try std.testing.expectEqual(forward.PromptMlaMode.tile16, forward.promptMlaSelected("1"));
    try std.testing.expectEqual(forward.PromptMlaMode.tile64, forward.promptMlaSelected("2"));
    try std.testing.expectEqual(forward.PromptMlaMode.p2, forward.promptMlaSelected("3"));
    try std.testing.expectEqual(forward.PromptMlaMode.p2n, forward.promptMlaSelected("4"));
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(1);
    f.prompt_mla = .off;
    const pos: [4]usize = .{ 4096, 4097, 4098, 4099 };
    const ids: [4]u32 = .{ 7, 8, 9, 10 };
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const baseline = try names_of();
    defer gpa.free(baseline);
    const prompt_want = try gpa.dupe(usize, baseline);
    defer gpa.free(prompt_want);
    for (prompt_want) |*kernel| {
        if (kernel.* == id("tf_glm53_absorb")) kernel.* = id("tf_glm53_absorb_prompt_mma");
        if (kernel.* == id("tf_glm53_expand_v")) kernel.* = id("tf_glm53_expand_v_prompt_mma");
    }

    Rec.reset();
    f.prompt_mla = .tile16;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSeq(prompt_want);

    const prompt_big_want = try gpa.dupe(usize, baseline);
    defer gpa.free(prompt_big_want);
    for (prompt_big_want) |*kernel| {
        if (kernel.* == id("tf_glm53_absorb")) kernel.* = id("tf_glm53_absorb_prompt_mma_big");
        if (kernel.* == id("tf_glm53_expand_v")) kernel.* = id("tf_glm53_expand_v_prompt_mma_big");
    }
    Rec.reset();
    f.prompt_mla = .tile64;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSeq(prompt_big_want);

    const prompt_p2_want = try gpa.dupe(usize, baseline);
    defer gpa.free(prompt_p2_want);
    for (prompt_p2_want) |*kernel| {
        if (kernel.* == id("tf_glm53_absorb")) kernel.* = id("tf_glm53_absorb_prompt_p2");
        if (kernel.* == id("tf_glm53_expand_v")) kernel.* = id("tf_glm53_expand_v_prompt_p2");
    }
    Rec.reset();
    f.prompt_mla = .p2;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSeq(prompt_p2_want);

    const prompt_p2n_want = try gpa.dupe(usize, baseline);
    defer gpa.free(prompt_p2n_want);
    for (prompt_p2n_want) |*kernel| {
        if (kernel.* == id("tf_glm53_expand_v")) kernel.* = id("tf_glm53_expand_v_prompt_p2n");
        if (kernel.* == id("tf_glm53_absorb")) kernel.* = id("tf_glm53_absorb_prompt_p2");
    }
    Rec.reset();
    f.prompt_mla = .p2n;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSeq(prompt_p2n_want);

    Rec.reset();
    f.prompt_mla = .off;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_baseline = try names_of();
    defer gpa.free(decode_baseline);
    Rec.reset();
    f.prompt_mla = .tile64;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_prompt = try names_of();
    defer gpa.free(decode_prompt);
    try std.testing.expectEqualSlices(usize, decode_baseline, decode_prompt);
}

test "SMLA-P switch replaces only the prefill attend launch and leaves decode unchanged" {
    try std.testing.expect(!forward.smlaPSelected(null));
    try std.testing.expect(!forward.smlaPSelected("0"));
    try std.testing.expect(forward.smlaPSelected("1"));
    try std.testing.expect(!forward.smlaPSelected("true"));
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(1);
    // Five prefill rows; SMLA-P blocks them in pairs (3 blocks, the last one half empty).
    const pos: [5]usize = .{ 4096, 4097, 4098, 4099, 4100 };
    const ids: [5]u32 = .{ 7, 8, 9, 10, 11 };
    f.smla_p = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const baseline = try names_of();
    defer gpa.free(baseline);
    const baseline_launches = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(baseline_launches);
    const want = try gpa.dupe(usize, baseline);
    defer gpa.free(want);
    var attends: usize = 0;
    for (want) |*kernel| {
        if (kernel.* == id("tf_glm53_attend")) {
            kernel.* = id("tf_glm53_smla_p");
            attends += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), attends); // one prefill attend a layer

    Rec.reset();
    f.smla_p = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSeq(want);
    // Same arguments as the prefill attend it replaces (prefill = 1); 2-row blocks of 512 threads, 99 KB or less.
    var checked: usize = 0;
    for (Rec.launches.items, baseline_launches) |got, base| {
        if (got.name != id("tf_glm53_smla_p")) continue;
        try std.testing.expectEqual(id("tf_glm53_attend"), base.name);
        try std.testing.expectEqual([3]u32{ 3, 1, 1 }, got.grid);
        try std.testing.expectEqual([3]u32{ 512, 1, 1 }, got.block);
        try std.testing.expect(got.shared > 48 * 1024 and got.shared <= 99 * 1024);
        try std.testing.expectEqual(base.nargs, got.nargs);
        try std.testing.expectEqualSlices(u64, base.args[0..base.nargs], got.args[0..got.nargs]);
        try std.testing.expectEqual(@as(u64, 1), got.args[17]);
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), checked);

    // Decode: identical launch list (names, grids, arguments) with the switch on or off.
    Rec.reset();
    f.smla_p = false;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(decode_off);
    Rec.reset();
    f.smla_p = true;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_on = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(decode_on);
    try std.testing.expectEqual(decode_off.len, decode_on.len);
    for (decode_off, decode_on) |a, b| {
        try std.testing.expect(a.name != id("tf_glm53_smla_p") and b.name != id("tf_glm53_smla_p"));
        try std.testing.expectEqual(a.name, b.name);
        try std.testing.expectEqual(a.grid, b.grid);
        try std.testing.expectEqual(a.block, b.block);
        try std.testing.expectEqual(a.shared, b.shared);
        try std.testing.expectEqualSlices(u64, a.args[0..a.nargs], b.args[0..b.nargs]);
    }
}

fn expectSameLaunch(want: Launch, got: Launch) !void {
    try std.testing.expectEqual(want.name, got.name);
    try std.testing.expectEqual(want.grid, got.grid);
    try std.testing.expectEqual(want.block, got.block);
    try std.testing.expectEqual(want.shared, got.shared);
    try std.testing.expectEqualSlices(u64, want.args[0..want.nargs], got.args[0..got.nargs]);
}

fn expectSameLaunches(want: []const Launch, got: []const Launch) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |a, b| try expectSameLaunch(a, b);
}

fn w8a8Launch(b: Launch, level2: bool) bool {
    if (b.name != id("qmmf_bf16") and b.name != id("qmmf_f32")) return false;
    const cols = b.args[7];
    const k_dim = b.args[8];
    const f32_out = b.name == id("qmmf_f32");
    if (f32_out and cols == 6144 and k_dim == 8192) return true;
    if (!level2) return false;
    if (cols < cuda.w8a8.level2_min_n) return false; // narrow outputs stay on release (w8a8.servesFast)
    const pairs = [_][3]u64{
        .{ 2048, 6144, 0 },
        .{ 8192, 2048, 0 },
        .{ 576, 6144, 0 },
        .{ 128, 6144, 0 },
        .{ 4096, 2048, 0 },
        .{ 12288, 6144, 0 },
        .{ 6144, 6144, 1 },
        .{ 6144, 1024, 1 },
    };
    const out_flag: u64 = @as(u64, if (f32_out) 1 else 0);
    for (pairs) |pair| {
        if (pair[0] == cols and pair[1] == k_dim and pair[2] == out_flag) return true;
    }
    return false;
}

fn expectW8a8Pair(b: Launch, quant: Launch, gemm: Launch, rows: usize, part: u64, scratch: u64) !void {
    const cols_us: usize = @intCast(b.args[7]);
    const k_us: usize = @intCast(b.args[8]);
    const base: u64 = if (b.args[8] == 8192) part else scratch;
    const at = cuda.w8a8.scratchAt(base, rows, k_us);
    const grid_x: u32 = @intCast((k_us / cuda.w8a8.block_k + cuda.w8a8.quant_blocks - 1) / cuda.w8a8.quant_blocks);
    try std.testing.expectEqual(id("tf_glm53_w8a8_quant"), quant.name);
    try std.testing.expectEqual(b.args[0], quant.args[0]);
    try std.testing.expectEqual(b.args[8], quant.args[1]);
    try std.testing.expectEqual(@as(u64, rows), quant.args[2]);
    try std.testing.expectEqual(b.args[8], quant.args[3]);
    try std.testing.expectEqual(at.xq, quant.args[4]);
    try std.testing.expectEqual(at.xs, quant.args[5]);
    try std.testing.expectEqual([3]u32{ grid_x, @intCast(rows), 1 }, quant.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, quant.block);
    try std.testing.expectEqual(@as(u32, 0), quant.shared);
    try std.testing.expectEqual(id("tf_glm53_w8a8_gemm"), gemm.name);
    try std.testing.expectEqual(at.xq, gemm.args[0]);
    try std.testing.expectEqual(at.xs, gemm.args[1]);
    try std.testing.expectEqual(b.args[1], gemm.args[2]);
    try std.testing.expectEqual(b.args[2], gemm.args[3]);
    try std.testing.expectEqual(b.args[4], gemm.args[4]);
    try std.testing.expectEqual(@as(u64, rows), gemm.args[5]);
    try std.testing.expectEqual(b.args[7], gemm.args[6]);
    try std.testing.expectEqual(b.args[8], gemm.args[7]);
    try std.testing.expectEqual(@as(u64, if (b.name == id("qmmf_f32")) 1 else 0), gemm.args[8]);
    try std.testing.expectEqual([3]u32{ @intCast(cuda.w8a8.gemmBlocks(rows, cols_us)), 1, 1 }, gemm.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, gemm.block);
    try std.testing.expectEqual(cuda.w8a8.shared_bytes, gemm.shared);
}

fn expectPrefillW8a8Launches(baseline: []const Launch, got: []const Launch, rows: usize, part: u64, scratch: u64, level2: bool) !void {
    var gi: usize = 0;
    var swapped: usize = 0;
    for (baseline) |b| {
        if (w8a8Launch(b, level2)) {
            try std.testing.expect(gi + 1 < got.len);
            try expectW8a8Pair(b, got[gi], got[gi + 1], rows, part, scratch);
            gi += 2;
            swapped += 1;
        } else {
            try std.testing.expect(gi < got.len);
            try expectSameLaunch(b, got[gi]);
            gi += 1;
        }
    }
    try std.testing.expectEqual(got.len, gi);
    if (level2) try std.testing.expect(swapped > 2 and swapped < 14) else try std.testing.expectEqual(@as(usize, 2), swapped);
}

test "TF_GLM53_PREFILL_W8A8 replaces only the prefill o_proj launch; off and decode launch exactly as before" {
    try std.testing.expect(!forward.prefillW8a8Selected(null));
    try std.testing.expect(!forward.prefillW8a8Selected("0"));
    try std.testing.expect(forward.prefillW8a8Selected("1"));
    try std.testing.expect(!forward.prefillW8a8Selected("true"));
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(1);
    const pos: [5]usize = .{ 4096, 4097, 4098, 4099, 4100 };
    const ids: [5]u32 = .{ 7, 8, 9, 10, 11 };
    const rows: usize = pos.len;

    // Today's launches: no W8A8 kernels, switch off.
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const baseline = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(baseline);

    // Off with the W8A8 kernels loaded: the same launch list, argument for argument.
    fx.k.w8a8 = fakeW8a8();
    Rec.reset();
    f.prefill_w8a8 = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectSameLaunches(baseline, Rec.launches.items);

    // On: each prefill o_proj (the qmmf_f32 right after expand_v) becomes quantize + GEMM; nothing else moves.
    Rec.reset();
    f.prefill_w8a8 = true;
    try f.validateSetup();
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const got = Rec.launches.items;
    const at = cuda.w8a8.scratchAt(fx.st.ws.part, rows, 8192);
    var gi: usize = 0;
    var replaced: usize = 0;
    for (baseline, 0..) |b, bi| {
        const o_proj = b.name == id("qmmf_f32") and bi > 0 and baseline[bi - 1].name == id("tf_glm53_expand_v");
        if (!o_proj) {
            try std.testing.expect(gi < got.len);
            try expectSameLaunch(b, got[gi]);
            gi += 1;
            continue;
        }
        try std.testing.expect(gi + 1 < got.len);
        const q = got[gi];
        const g = got[gi + 1];
        gi += 2;
        replaced += 1;
        // qmmf_f32 arguments: x, codes, scales, scale, y, part, m, n, k, sk, npad, ldx, band.
        try std.testing.expectEqual(id("tf_glm53_w8a8_quant"), q.name);
        try std.testing.expectEqual(b.args[0], q.args[0]); // x: ws.vo
        try std.testing.expectEqual(@as(u64, 8192), q.args[1]); // ldx: 32 heads x 256
        try std.testing.expectEqual(@as(u64, rows), q.args[2]);
        try std.testing.expectEqual(@as(u64, 8192), q.args[3]);
        try std.testing.expectEqual(at.xq, q.args[4]); // scratch: ws.part, dead after expand_v
        try std.testing.expectEqual(at.xs, q.args[5]);
        try std.testing.expectEqual([3]u32{ 8, @intCast(rows), 1 }, q.grid); // 64 128-K blocks, 8 a CTA; one row a grid row
        try std.testing.expectEqual([3]u32{ 256, 1, 1 }, q.block);
        try std.testing.expectEqual(id("tf_glm53_w8a8_gemm"), g.name);
        try std.testing.expectEqual(at.xq, g.args[0]);
        try std.testing.expectEqual(at.xs, g.args[1]);
        try std.testing.expectEqual(b.args[1], g.args[2]); // the same FP8B codes and expanded scales
        try std.testing.expectEqual(b.args[2], g.args[3]);
        try std.testing.expectEqual(b.args[4], g.args[4]); // the same fp32 rank partial slot
        try std.testing.expectEqual(@as(u64, rows), g.args[5]);
        try std.testing.expectEqual(@as(u64, 6144), g.args[6]);
        try std.testing.expectEqual(@as(u64, 8192), g.args[7]);
        try std.testing.expectEqual(@as(u64, 1), g.args[8]); // fp32 out
        try std.testing.expectEqual([3]u32{ @intCast(cuda.w8a8.gemmBlocks(rows, 6144)), 1, 1 }, g.grid);
        try std.testing.expectEqual([3]u32{ 256, 1, 1 }, g.block);
        try std.testing.expectEqual(cuda.w8a8.shared_bytes, g.shared);
    }
    try std.testing.expectEqual(got.len, gi);
    try std.testing.expectEqual(@as(usize, 2), replaced); // one prefill o_proj a layer

    // Decode: the same launches with the switch on or off.
    Rec.reset();
    f.prefill_w8a8 = false;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(decode_off);
    Rec.reset();
    f.prefill_w8a8 = true;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    try expectSameLaunches(decode_off, Rec.launches.items);
    for (Rec.launches.items) |l| try std.testing.expect(l.name != id("tf_glm53_w8a8_quant") and l.name != id("tf_glm53_w8a8_gemm"));

    // Setup refuses the switch without its kernels.
    fx.k.w8a8 = null;
    try std.testing.expectError(error.MissingW8a8Kernels, f.validateSetup());
}

test "TF_GLM53_PREFILL_W8A8=2 adds the other served prefill projections; 0 and 1 stay on today's launches" {
    try std.testing.expectEqual(@as(u8, 0), forward.prefillW8a8LevelSelected(null));
    try std.testing.expectEqual(@as(u8, 0), forward.prefillW8a8LevelSelected("0"));
    try std.testing.expectEqual(@as(u8, 0), forward.prefillW8a8LevelSelected("true"));
    try std.testing.expectEqual(@as(u8, 0), forward.prefillW8a8LevelSelected("3"));
    try std.testing.expectEqual(@as(u8, 1), forward.prefillW8a8LevelSelected("1"));
    try std.testing.expectEqual(@as(u8, 2), forward.prefillW8a8LevelSelected("2"));
    try std.testing.expect(forward.prefillW8a8Selected("1"));
    try std.testing.expect(forward.prefillW8a8Selected("2"));
    try std.testing.expect(!forward.prefillW8a8Selected("true"));
    try std.testing.expect(!forward.prefillW8a8AllSelected("1"));
    try std.testing.expect(forward.prefillW8a8AllSelected("2"));

    var g = geometry();
    g.prefill_w8a8_all = true;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    try std.testing.expectEqual(@as(usize, 1024), fx.st.ws.piece_rows);
    try std.testing.expectEqual(@as(u64, 6_488_064), fx.st.ws.sizes.w8a8_scratch);
    var f = fx.fwd(1);
    const pos: [5]usize = .{ 4096, 4097, 4098, 4099, 4100 };
    const ids: [5]u32 = .{ 7, 8, 9, 10, 11 };
    const rows: usize = pos.len;
    fx.k.w8a8 = fakeW8a8();

    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const baseline = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(baseline);

    Rec.reset();
    f.prefill_w8a8 = true;
    f.prefill_w8a8_all = false;
    try f.validateSetup();
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectPrefillW8a8Launches(baseline, Rec.launches.items, rows, fx.st.ws.part, fx.st.ws.w8a8_scratch, false);

    Rec.reset();
    f.prefill_w8a8_all = true;
    try f.validateSetup();
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try expectPrefillW8a8Launches(baseline, Rec.launches.items, rows, fx.st.ws.part, fx.st.ws.w8a8_scratch, true);

    Rec.reset();
    f.prefill_w8a8 = false;
    f.prefill_w8a8_all = false;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    const decode_off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(decode_off);
    Rec.reset();
    f.prefill_w8a8 = true;
    f.prefill_w8a8_all = true;
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    try expectSameLaunches(decode_off, Rec.launches.items);
    for (Rec.launches.items) |launch| try std.testing.expect(launch.name != id("tf_glm53_w8a8_quant") and launch.name != id("tf_glm53_w8a8_gemm"));

    f.prefill_w8a8 = false;
    f.prefill_w8a8_all = true;
    try std.testing.expectError(error.W8a8Level, f.validateSetup());
    f.prefill_w8a8 = true;
    fx.k.w8a8 = null;
    try std.testing.expectError(error.MissingW8a8Kernels, f.validateSetup());
    fx.k.w8a8 = fakeW8a8();
    fx.st.ws.sizes.w8a8_scratch = 0;
    try std.testing.expectError(error.W8a8Scratch, f.validateSetup());
}

test "SMLA-P refuses scalar attend during setup" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(1);
    f.smla_p = true;
    try std.testing.expectError(error.SmlaPWithScalarAttend, f.validateSetup());
}

test "SMLA-FP8 off leaves the SMLA-P launches unchanged; on swaps only the SMLA-P kernel" {
    try std.testing.expect(!forward.smlaFp8Selected(null));
    try std.testing.expect(!forward.smlaFp8Selected("0"));
    try std.testing.expect(forward.smlaFp8Selected("1"));
    try std.testing.expect(!forward.smlaFp8Selected("true"));
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(1);
    const pos: [5]usize = .{ 4096, 4097, 4098, 4099, 4100 };
    const ids: [5]u32 = .{ 7, 8, 9, 10, 11 };

    // Off: SMLA-P with smla_fp8 = false is the SMLA-P launch list (names, grids, shared memory, arguments) exactly.
    f.smla_p = true;
    f.smla_fp8 = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const smla_off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(smla_off);
    var smla_launches: usize = 0;
    for (smla_off) |l| {
        try std.testing.expect(l.name != id("tf_glm53_smla_p_fp8"));
        smla_launches += @intFromBool(l.name == id("tf_glm53_smla_p"));
    }
    try std.testing.expectEqual(@as(usize, 2), smla_launches); // one prefill attend a layer

    // On: every tf_glm53_smla_p launch becomes tf_glm53_smla_p_fp8 with the same grid, block and arguments; the shared
    // memory is the FP8 layout (<= 99 KB); every other launch is unchanged.
    Rec.reset();
    f.smla_fp8 = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const smla_on = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(smla_on);
    try std.testing.expectEqual(smla_off.len, smla_on.len);
    var swapped: usize = 0;
    for (smla_off, smla_on) |a, b| {
        try std.testing.expectEqual(a.grid, b.grid);
        try std.testing.expectEqual(a.block, b.block);
        try std.testing.expectEqual(a.nargs, b.nargs);
        try std.testing.expectEqualSlices(u64, a.args[0..a.nargs], b.args[0..b.nargs]);
        if (a.name == id("tf_glm53_smla_p")) {
            try std.testing.expectEqual(id("tf_glm53_smla_p_fp8"), b.name);
            try std.testing.expectEqual(@as(u32, 101184), b.shared);
            try std.testing.expect(b.shared <= 99 * 1024);
            swapped += 1;
        } else {
            try std.testing.expectEqual(a.name, b.name);
            try std.testing.expectEqual(a.shared, b.shared);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), swapped);

    // Decode never launches either SMLA kernel, switch on or off.
    Rec.reset();
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    for (Rec.launches.items) |l| try std.testing.expect(l.name != id("tf_glm53_smla_p") and l.name != id("tf_glm53_smla_p_fp8"));
}

test "SMLA-P3 off launches exactly SMLA-P; on replaces each SMLA-P attend by the union pre-kernel + P3" {
    try std.testing.expect(!forward.smlaP3Selected(null));
    try std.testing.expect(!forward.smlaP3Selected("0"));
    try std.testing.expect(forward.smlaP3Selected("1"));
    try std.testing.expect(!forward.smlaP3Selected("true"));
    var g = geometry();
    g.smla_p3 = true; // carves ws.smla_union; the other regions do not move (state.zig SMLA-P3 carve test)
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.mla.smla_union_fn = func("tf_glm53_smla_union");
    fx.k.mla.smla_p3_fn = func("tf_glm53_smla_p3");
    var f = fx.fwd(1);
    // Five prefill rows: 3 row pairs (the last one half empty), odd on purpose.
    const pos: [5]usize = .{ 4096, 4097, 4098, 4099, 4100 };
    const ids: [5]u32 = .{ 7, 8, 9, 10, 11 };
    // mla.smlaP3TableBytes(piece rows): [ceil(rows / 2) pairs][128 steps][176 B].
    try std.testing.expectEqual(@as(u64, (fx.st.ws.piece_rows + 1) / 2 * 128 * 176), fx.st.ws.sizes.smla_union);

    // Off: the SMLA-P launch list; no P3 kernel.
    f.smla_p = true;
    f.smla_p3 = false;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(off);
    for (off) |l| try std.testing.expect(l.name != id("tf_glm53_smla_union") and l.name != id("tf_glm53_smla_p3"));

    // On: every tf_glm53_smla_p launch becomes tf_glm53_smla_union (positions, tokens, counts, table, rows, rank,
    // world) then tf_glm53_smla_p3 (the same grid, block and arguments + the table, P3 shared memory); the rest is equal.
    Rec.reset();
    f.smla_p3 = true;
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const on = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(on);
    var j: usize = 0;
    var replaced: usize = 0;
    for (off) |a| {
        if (a.name != id("tf_glm53_smla_p")) {
            try expectSameLaunch(a, on[j]);
            j += 1;
            continue;
        }
        const u = on[j];
        const p = on[j + 1];
        try std.testing.expectEqual(id("tf_glm53_smla_union"), u.name);
        const rows: usize = @intCast(a.args[13]);
        try std.testing.expectEqual([3]u32{ @intCast(((rows + 1) / 2 + 3) / 4), 1, 1 }, u.grid); // 4 row pairs a block
        try std.testing.expectEqual([3]u32{ 128, 1, 1 }, u.block);
        try std.testing.expectEqual(@as(u32, 0), u.shared);
        try std.testing.expectEqualSlices(u64, &[_]u64{ a.args[6], a.args[7], a.args[9], fx.st.ws.smla_union, a.args[13], a.args[15], a.args[16] }, u.args[0..u.nargs]);
        try std.testing.expectEqual(id("tf_glm53_smla_p3"), p.name);
        try std.testing.expectEqual(a.grid, p.grid);
        try std.testing.expectEqual(a.block, p.block);
        try std.testing.expectEqual(@as(u32, 96704), p.shared);
        try std.testing.expect(p.shared <= 99 * 1024);
        try std.testing.expectEqual(a.nargs + 1, p.nargs);
        try std.testing.expectEqualSlices(u64, a.args[0..a.nargs], p.args[0..a.nargs]);
        try std.testing.expectEqual(fx.st.ws.smla_union, p.args[a.nargs]);
        j += 2;
        replaced += 1;
    }
    try std.testing.expectEqual(on.len, j);
    try std.testing.expectEqual(@as(usize, 2), replaced); // one prefill attend a layer

    // Decode never launches the P3 kernels.
    Rec.reset();
    try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
    for (Rec.launches.items) |l| try std.testing.expect(l.name != id("tf_glm53_smla_union") and l.name != id("tf_glm53_smla_p3"));
}

test "TF_GLM53_DECODE_UNION parsing" {
    try std.testing.expect(!forward.decodeUnionSelected(null));
    try std.testing.expect(!forward.decodeUnionSelected("0"));
    try std.testing.expect(!forward.decodeUnionSelected("true"));
    try std.testing.expect(!forward.decodeUnionSelected("2"));
    try std.testing.expect(forward.decodeUnionSelected("1"));
}

test "OWN_ATTEND and Q_ONE_GATHER switches accept only exact opt-in" {
    for ([_]?[]const u8{ null, "0", "true", "2" }) |value| {
        try std.testing.expect(!forward.ownAttendSelected(value));
        try std.testing.expect(!forward.qOneGatherSelected(value));
    }
    try std.testing.expect(forward.ownAttendSelected("1"));
    try std.testing.expect(forward.qOneGatherSelected("1"));
}

test "own attend and one query gather dispatch two head groups with matching collectives" {
    const cases = [_]struct { own: bool, one: bool }{ .{ .own = true, .one = false }, .{ .own = false, .one = true }, .{ .own = true, .one = true } };
    for (cases) |case| {
        var g = geometry();
        g.decode_side = case.own;
        g.own_attend = case.own;
        g.q_one_gather = case.one;
        g.decode_mla = case.one;
        g.decode_mla_v2 = case.one;
        g.decode_mla_v3 = case.one;
        var fx: Fixture = undefined;
        try fx.init(g, 16384);
        defer fx.deinit();
        fx.k.mla.attend_kernel = .mma;
        fx.k.mla.attend_verify_fn = func("tf_glm53_attend_verify");
        var f = fx.fwd(0);
        f.own_attend = case.own;
        f.q_one_gather = case.one;
        f.decode_union = case.own;
        try f.whole(.{ .positions = &.{ 100, 101 }, .tokens = &.{ 7, 8 } });
        var query_gathers: usize = 0;
        for (Rec.gathers.items) |gather| if (gather.phase == .absorbed_query) {
            const qa_bytes = 2 * 32 * 512 * 2;
            const qpe_bytes = 2 * 32 * 64 * 2;
            query_gathers += 1;
            if (case.one) {
                try std.testing.expectEqual(qa_bytes + qpe_bytes, gather.bytes);
            } else {
                try std.testing.expect(gather.bytes == qa_bytes or gather.bytes == qpe_bytes);
            }
        };
        try std.testing.expectEqual(if (case.one) @as(usize, 2) else @as(usize, 4), query_gathers);
        var verify_launches: usize = 0;
        for (Rec.launches.items) |launch| {
            if (launch.name == id("tf_glm53_attend_verify")) verify_launches += 1;
        }
        try std.testing.expectEqual(@as(usize, 4), verify_launches);
        if (case.own) {
            var side_attends: usize = 0;
            var fork_records: usize = 0;
            var fork_waits: usize = 0;
            var done_records: usize = 0;
            var done_waits: usize = 0;
            const fork_handle = @intFromPtr(fx.st.attn_fork.?.handle.?);
            const done_handle = @intFromPtr(fx.st.attn_done.?.handle.?);
            for (Rec.order.items) |event| {
                if (event.kind == .launch and event.what == id("tf_glm53_attend_verify") and event.stream == 0x51de) side_attends += 1;
                if (event.what == fork_handle and event.kind == .record) fork_records += 1;
                if (event.what == fork_handle and event.kind == .wait) fork_waits += 1;
                if (event.what == done_handle and event.kind == .record) done_records += 1;
                if (event.what == done_handle and event.kind == .wait) done_waits += 1;
            }
            try std.testing.expectEqual(@as(usize, 2), side_attends);
            try std.testing.expectEqual(@as(usize, 2), fork_records);
            try std.testing.expectEqual(@as(usize, 2), fork_waits);
            try std.testing.expectEqual(@as(usize, 2), done_records);
            try std.testing.expectEqual(@as(usize, 2), done_waits);
        }
    }
}

test "decode union off launches exactly attend_mma; on swaps decode attend in 1..8 rows" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    fx.k.mla.attend_kernel = .mma;
    fx.k.mla.attend_verify_fn = func("tf_glm53_attend_verify");
    for (0..2) |rank| {
        var f = fx.fwd(rank);
        Rec.reset();
        try f.whole(.{ .positions = &.{ 100, 101, 102, 103 }, .tokens = &.{ 7, 8, 9, 10 } });
        const off = try gpa.dupe(Launch, Rec.launches.items);
        defer gpa.free(off);
        Rec.reset();
        f.decode_union = true;
        try f.whole(.{ .positions = &.{ 100, 101, 102, 103 }, .tokens = &.{ 7, 8, 9, 10 } });
        var j: usize = 0;
        var replaced: usize = 0;
        var oi: usize = 0;
        while (oi < off.len) : (oi += 1) {
            const a = off[oi];
            const v = Rec.launches.items[j];
            if (a.name != id("tf_glm53_attend_mma")) {
                try expectSameLaunch(a, v);
                j += 1;
                continue;
            }
            try std.testing.expectEqual(id("tf_glm53_attend_verify"), v.name);
            try std.testing.expectEqual([3]u32{ 4, 2, 16 }, v.grid);
            try std.testing.expectEqual([3]u32{ 256, 1, 1 }, v.block);
            try std.testing.expectEqual(@as(u32, 81536), v.shared);
            try std.testing.expect(v.shared <= 99 * 1024);
            const mapped = [_]u64{ a.args[0], a.args[1], a.args[2], a.args[3], a.args[6], a.args[8], a.args[10], a.args[18], a.args[13], a.args[14], a.args[15], a.args[16] };
            try std.testing.expectEqualSlices(u64, &mapped, v.args[0..v.nargs]);
            try expectSameLaunch(off[oi + 1], Rec.launches.items[j + 1]);
            j += 2;
            oi += 1;
            replaced += 1;
        }
        try std.testing.expectEqual(Rec.launches.items.len, j);
        try std.testing.expectEqual(@as(usize, 2), replaced);
        Rec.reset();
        f.decode_union = false;
        try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
        const off_one = try gpa.dupe(Launch, Rec.launches.items);
        defer gpa.free(off_one);
        Rec.reset();
        f.decode_union = true;
        try f.whole(.{ .positions = &.{100}, .tokens = &.{7} });
        j = 0;
        oi = 0;
        while (oi < off_one.len) : (oi += 1) {
            const a = off_one[oi];
            const v = Rec.launches.items[j];
            if (a.name != id("tf_glm53_attend_mma")) {
                try expectSameLaunch(a, v);
                j += 1;
                continue;
            }
            try std.testing.expectEqual(id("tf_glm53_attend_verify"), v.name);
            try std.testing.expectEqual([3]u32{ 1, 2, 16 }, v.grid);
            try std.testing.expectEqual([3]u32{ 256, 1, 1 }, v.block);
            try std.testing.expectEqual(@as(u32, 81536), v.shared);
            const mapped = [_]u64{ a.args[0], a.args[1], a.args[2], a.args[3], a.args[6], a.args[8], a.args[10], a.args[18], a.args[13], a.args[14], a.args[15], a.args[16] };
            try std.testing.expectEqualSlices(u64, &mapped, v.args[0..v.nargs]);
            try expectSameLaunch(off_one[oi + 1], Rec.launches.items[j + 1]);
            j += 2;
            oi += 1;
        }
        try std.testing.expectEqual(Rec.launches.items.len, j);
        Rec.reset();
        try f.pieced(.{ .positions = &.{ 4096, 4097, 4098, 4099, 4100 }, .tokens = &.{ 7, 8, 9, 10, 11 } });
        const prefill_on = try gpa.dupe(Launch, Rec.launches.items);
        defer gpa.free(prefill_on);
        Rec.reset();
        f.decode_union = false;
        try f.pieced(.{ .positions = &.{ 4096, 4097, 4098, 4099, 4100 }, .tokens = &.{ 7, 8, 9, 10, 11 } });
        try expectSameLaunches(prefill_on, Rec.launches.items);
    }
}

test "decode union setup refuses scalar attend and a missing kernel" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.decode_union = true;
    fx.k.mla.attend_kernel = .scalar;
    try std.testing.expectError(error.DecodeUnionWithScalarAttend, f.validateSetup());
    f.decode_union = false;
    try f.validateSetup();
    f.q_one_gather = true;
    try std.testing.expectError(error.QOneGatherNeedsDecodeMla3, f.validateSetup());
    f.decode_mla_v3 = true;
    fx.k.mla.attend_verify_fn = null;
    try std.testing.expectError(error.MissingQOneGatherKernel, f.validateSetup());
    fx.k.mla.attend_verify_fn = func("tf_glm53_attend_verify");
    try f.validateSetup();
    f.q_one_gather = false;
    f.decode_mla_v3 = null;
    fx.k.mla.attend_kernel = .mma;
    f.decode_union = true;
    fx.k.mla.attend_verify_fn = null;
    try std.testing.expectError(error.MissingDecodeUnionKernel, f.validateSetup());
    fx.k.mla.attend_verify_fn = func("tf_glm53_attend_verify");
    try f.validateSetup();
    const split_size = fx.st.ws.sizes.attend_split;
    fx.st.ws.sizes.attend_split = 0;
    try std.testing.expectError(error.DecodeUnionWorkspace, f.validateSetup());
    fx.st.ws.sizes.attend_split = split_size;
    f.decode_union = false;
    try f.validateSetup();
}

test "SMLA-P3 setup refuses a missing SMLA-P, SMLA-FP8 and missing kernels" {
    var g = geometry();
    g.smla_p3 = true;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.mla.attend_kernel = .mma; // SMLA-P itself refuses the scalar attend the fixture records
    fx.k.mla.smla_union_fn = func("tf_glm53_smla_union");
    fx.k.mla.smla_p3_fn = func("tf_glm53_smla_p3");
    var f = fx.fwd(1);
    f.smla_p3 = true;
    f.smla_p = false;
    try std.testing.expectError(error.SmlaP3WithoutSmlaP, f.validateSetup());
    f.smla_p = true;
    f.smla_fp8 = true;
    try std.testing.expectError(error.SmlaP3WithSmlaFp8, f.validateSetup());
    f.smla_fp8 = false;
    try f.validateSetup();
    fx.k.mla.smla_p3_fn = null;
    try std.testing.expectError(error.MissingSmlaP3Kernels, f.validateSetup());
    fx.k.mla.smla_p3_fn = func("tf_glm53_smla_p3");
    fx.k.mla.smla_union_fn = null;
    try std.testing.expectError(error.MissingSmlaP3Kernels, f.validateSetup());
    fx.k.mla.smla_union_fn = func("tf_glm53_smla_union");
    try f.validateSetup();
}

test "SMLA-P3 setup refuses a workspace carved without the union table" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384); // smla_p3 off in the geometry: no table
    defer fx.deinit();
    fx.k.mla.attend_kernel = .mma;
    fx.k.mla.smla_union_fn = func("tf_glm53_smla_union");
    fx.k.mla.smla_p3_fn = func("tf_glm53_smla_p3");
    var f = fx.fwd(1);
    f.smla_p = true;
    f.smla_p3 = true;
    try std.testing.expectEqual(@as(u64, 0), fx.st.ws.sizes.smla_union);
    try std.testing.expectError(error.SmlaP3Table, f.validateSetup());
    f.smla_p3 = false;
    try f.validateSetup();
}

test "SMLA-FP8 without SMLA-P is refused during setup" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    fx.k.mla.attend_kernel = .mma; // SMLA-P itself refuses the scalar attend the fixture records
    var f = fx.fwd(1);
    f.smla_p = false;
    f.smla_fp8 = true;
    try std.testing.expectError(error.SmlaFp8WithoutSmlaP, f.validateSetup());
    f.smla_p = true;
    try f.validateSetup();
    f.smla_fp8 = false;
    try f.validateSetup();
    f.smla_p = false;
    try f.validateSetup();
}

test "deferred expert prefill groups routed calls and leaves decode launches unchanged" {
    var g = geometry();
    g.layers = 2;
    g.full_index_layers = 0;
    g.piece_rows = 512;
    g.expert_rows = 1024;
    var pos: [1536]usize = undefined;
    var ids: [1536]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| { p.* = i; t.* = @intCast(i); }
    for ([_]usize{ 1024, 128 }) |er| {
        Rec.reset();
        g.expert_rows = er;
        var fx: Fixture = undefined;
        try fx.init(g, 4096);
        defer fx.deinit();
        fx.layer_store[0] = glmLayer(0, false, false);
        fx.layer_store[1] = glmLayer(1, false, false);
        fx.model.layers = fx.layer_store[0..2];
        try fx.fwd(0).pieced(.{ .positions = &pos, .tokens = &ids });
        var routed_launches: usize = 0;
        for (Rec.launches.items) |launch| if (launch.name == id("exl3_group")) { routed_launches += 1; };
        const per_layer = if (er == 1024) @as(usize, 2) else 12;
        try std.testing.expectEqual(per_layer * 2, routed_launches);
        // The MoE residual add runs once per 512-row piece (3 per layer): routed calls before the 3rd add of a layer
        // belong to that layer.
        var layer_calls: [2]usize = @splat(0);
        var adds: usize = 0;
        for (Rec.launches.items) |launch| {
            if (launch.name == id("exl3_group")) layer_calls[@min(adds / 3, 1)] += 1;
            if (launch.name == id("tf_glm53_moe_residual_add")) adds += 1;
        }
        try std.testing.expectEqual(@as(usize, 6), adds);
        try std.testing.expectEqual(per_layer, layer_calls[0]);
        try std.testing.expectEqual(per_layer, layer_calls[1]);
    }

    var decode_launches: [2][]Launch = undefined;
    for ([_]usize{ 1024, 128 }, 0..) |er, idx| {
        Rec.reset();
        g.expert_rows = er;
        var fx: Fixture = undefined;
        try fx.init(g, 4096);
        defer fx.deinit();
        fx.layer_store[0] = glmLayer(0, false, false);
        fx.layer_store[1] = glmLayer(1, false, false);
        fx.model.layers = fx.layer_store[0..2];
        const one_id = [_]u32{0};
        var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
        defer e.deinit();
        var out: [1]u32 = undefined;
        try e.verify(&one_id, &out);
        decode_launches[idx] = try gpa.dupe(Launch, Rec.launches.items);
    }
    defer gpa.free(decode_launches[0]);
    defer gpa.free(decode_launches[1]);
    try std.testing.expectEqual(decode_launches[0].len, decode_launches[1].len);
    for (decode_launches[0], decode_launches[1]) |a, b| {
        try std.testing.expectEqual(a.name, b.name);
        try std.testing.expectEqual(a.grid, b.grid);
        try std.testing.expectEqual(a.block, b.block);
        try std.testing.expectEqual(a.nargs, b.nargs);
        if (layout(a.name)) |kinds| for (kinds, 0..) |kind, arg| {
            if (kind != 'p') try std.testing.expectEqual(a.args[arg], b.args[arg]);
        };
    }
}

fn inRegion(ws: *const state.Workspace, comptime region: []const u8, ptr: u64, need: u64) !void {
    const s = @field(ws, region);
    const z = @field(ws.sizes, region);
    if (ptr < s or ptr + need > s + z) {
        std.debug.print("buffer contract: {s} [{x}, +{d}) outside [{x}, +{d})\n", .{ region, ptr, need, s, z });
        return error.BufferContract;
    }
}

/// A pointer and extent inside some workspace region (qmmf outputs land in several).
fn inAnyRegion(ws: *const state.Workspace, ptr: u64, need: u64) !void {
    inline for (@typeInfo(state.Workspace.Sizes).@"struct".field_names) |n| {
        const s = @field(ws, n);
        const size = @field(ws.sizes, n);
        if (comptime @TypeOf(s) == [2]u64) {
            inline for (0..2) |half| {
                if (ptr >= s[half] and ptr + need <= s[half] + size[half]) return;
            }
        } else if (ptr >= s and ptr + need <= s + size) return;
    }
    std.debug.print("buffer contract: [{x}, +{d}) in no workspace region\n", .{ ptr, need });
    return error.BufferContract;
}

/// Every recorded launch's buffers hold what its kernel contract reads or writes (rows, heads, widths from its
/// own scalar arguments), planes are this layer's, and decode never passes a stage plane.
fn checkContracts(fx: *const Fixture, mode: forward.Mode, max_position: usize) !void {
    const ws = &fx.st.ws;
    var kv_writes: usize = 0;
    var last_layer: usize = 0;
    var checked: usize = 0;
    for (Rec.launches.items) |l| {
        const a = l.args;
        const layer = last_layer;
        switch (l.name) {
            id("tf_glm53_kv_write") => {
                kv_writes += 1;
                const rows = a[9];
                const this: usize = @intCast((a[5] - fx.st.latent.ptr) / (fx.st.slots * 1024));
                try std.testing.expect(this < fx.st.layers);
                last_layer = this;
                try inRegion(ws, "kv", a[0], rows * a[10] * 2);
                try std.testing.expectEqual(@as(u64, 576), a[10]);
                try inRegion(ws, "cos", a[2], rows * 32 * 4);
                try inRegion(ws, "positions", a[4], rows * 8);
                try std.testing.expectEqual(fx.st.latentAt(this), a[5]);
                try std.testing.expectEqual(fx.st.ropeAt(this), a[6]);
                try std.testing.expect(max_position / 2 < fx.st.slots);
                if (mode == .decode) try std.testing.expectEqual(@as(u64, 0), a[7] | a[8]) else {
                    try std.testing.expectEqual(state.stageLatent(&fx.st.stage, this), a[7]);
                    try std.testing.expectEqual(state.stageRope(&fx.st.stage, this), a[8]);
                    try std.testing.expect(max_position / 2 < fx.st.stage.rows);
                }
            },
            id("tf_glm53_absorb") => {
                const rows = a[4];
                const lh = a[5];
                try inRegion(ws, "q", a[0], rows * lh * 256 * 2);
                try inRegion(ws, "qa", a[3], rows * lh * 512 * 2);
                try std.testing.expect(a[6] + lh <= 32); // local kv_b rows: 32 heads x 448
            },
            id("tf_glm53_qpe_pack") => try inRegion(ws, "qpe", a[3], a[4] * a[5] * 64 * 2),
            id("tf_glm53_attend") => {
                const rows = a[13];
                const lh = a[14];
                try std.testing.expectEqual(@as(u64, if (mode == .decode) 64 else 32), lh);
                try std.testing.expectEqual(@as(u64, @intFromBool(mode == .prefill)), a[17]);
                try inRegion(ws, "qa", a[0], rows * lh * 512 * 2);
                try inRegion(ws, "qpe", a[1], rows * lh * 64 * 2);
                try std.testing.expectEqual(fx.st.latentAt(layer), a[2]);
                try inRegion(ws, "global_tokens", a[7], rows * 2048 * 4);
                if (mode == .decode) try inRegion(ws, "local_slots", a[8], rows * 2048 * 4); // prefill never reads it
                try inRegion(ws, "dense_len", a[11], rows * 4);
                try inRegion(ws, "part", a[12], rows * lh * 512 * 2);
                if (mode == .decode) try std.testing.expect(a[4] == 0 and a[5] == 0) else try std.testing.expect(a[4] != 0 and a[5] != 0);
            },
            id("tf_glm53_mla_merge") => {
                const rows = a[4];
                const lh = a[5];
                try inRegion(ws, "part", a[0], rows * lh * 1024);
                try inRegion(ws, "part", a[1], rows * lh * 1024);
                try inRegion(ws, "ol", a[2], rows * lh * 1024);
                try inRegion(ws, "lse_out", a[3], rows * lh * 4);
            },
            id("tf_glm53_expand_v") => {
                const rows = a[4];
                const lh = a[5];
                if (mode == .decode) try inRegion(ws, "ol", a[0], rows * lh * 1024) else try inRegion(ws, "part", a[0], rows * lh * 1024);
                try inRegion(ws, "vo", a[3], rows * lh * 256 * 2);
            },
            id("tf_glm53_index_scores") => {
                const rows = l.grid[1];
                try inRegion(ws, "qi", a[0], rows * 4096 * 2);
                try inRegion(ws, "iw", a[1], rows * a[2] * 2);
                try std.testing.expect(a[4] >= fx.st.indexAt(0) and a[4] < fx.st.indexAt(fx.st.full_index_layers) and (a[4] - fx.st.indexAt(0)) % (fx.st.slots * 256) == 0);
                try std.testing.expect(a[7] <= fx.st.slots and a[7] <= a[8]);
                try inRegion(ws, "scores", a[6], rows * a[8] * 4);
            },
            id("tf_glm53_merge") => {
                const rows = a[8];
                const k = a[9];
                try inRegion(ws, "gathered_candidates", a[0], 2 * rows * k * 8);
                try inRegion(ws, "global_tokens", a[1], rows * k * 4);
                try inRegion(ws, "global_counts", a[2], rows * 4);
                try inRegion(ws, "local_slots", a[3], rows * k * 4);
                try inRegion(ws, "local_counts", a[4], rows * 4);
            },
            id("tf_glm_residual_add") => try inRegion(ws, "gathered", a[2], 2 * a[3] * 4),
            id("tf_glm53_embed_pick") => try inRegion(ws, "gathered", a[0], 2 * a[3] * a[4] * 2),
            id("qmmf_bf16"), id("qmmf_f32") => {
                const m = a[6];
                const out: u64 = if (l.name == id("qmmf_f32")) 4 else 2;
                try inAnyRegion(ws, a[4], m * a[7] * out);
                try inAnyRegion(ws, a[0], m * (if (m == 1) a[8] else a[11]) * 2);
            },
            id("tf_glm53_top1_f32") => {
                try inRegion(ws, "logits", a[0], l.grid[0] * a[1] * 4);
                try inRegion(ws, "head_gather", a[3], l.grid[0] * 4);
            },
            id("exl3_down_combine") => try inRegion(ws, "gathered", a[5], l.grid[0] * 6144 * 4),
            else => continue,
        }
        checked += 1;
    }
    try std.testing.expect(checked > 20);
}

/// Bytes and calls by phase from the recorded collective calls.
fn recorded(phase: forward.Phase) [2]usize {
    var out: [2]usize = .{ 0, 0 };
    for (Rec.gathers.items) |g| if (g.phase == phase) {
        out[0] += g.bytes;
        out[1] += 1;
    };
    return out;
}

fn expectPlan(p: forward.ExchangePlan) !void {
    inline for (@typeInfo(forward.Phase).@"enum".field_names) |n| {
        const phase = @field(forward.Phase, n);
        const got = recorded(phase);
        if (got[0] != p.of(phase).bytes or got[1] != p.of(phase).calls) {
            std.debug.print("phase {s}: recorded {d} B in {d} calls, plan {d} B in {d}\n", .{ n, got[0], got[1], p.of(phase).bytes, p.of(phase).calls });
            return error.ExchangePlanMismatch;
        }
    }
}

test "decode: every buffer has the kernel contract's size and exchange bytes equal the plan" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(0);
    const pos = [_]usize{ 2045, 2046, 2047, 2048, 2049, 2050, 9000, 9001 };
    try f.whole(.{ .positions = &pos, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } });
    try checkContracts(&fx, .decode, 9001);
    var ls: [2]forward.Layer = undefined;
    try expectPlan(try f.stepPlan(.decode, &pos, pos.len, &ls));
    // DCP2 decode: 36,864 B per row of absorbed query, 32,896 B per peer partial/LSE block.
    const expected_absorbed_query = [2]usize{ 2 * 8 * 36864, 4 };
    const actual_absorbed_query = recorded(.absorbed_query);
    try std.testing.expectEqualSlices(usize, &expected_absorbed_query, &actual_absorbed_query);
    const expected_attention_partial = [2]usize{ 2 * 8 * 32896, 2 };
    const actual_attention_partial = recorded(.attention_partial);
    try std.testing.expectEqualSlices(usize, &expected_attention_partial, &actual_attention_partial);
    for (Rec.gathers.items) |g| if (g.phase == .attention_partial) {
        try std.testing.expect(g.send_recv);
        try std.testing.expectEqual(@as(usize, 32896 * 8), g.bytes);
        try std.testing.expectEqual(g.bytes, g.recv_bytes);
    };
}

test "partial RoCE host dispatch follows TF_GLM53_PARTIAL_ROCE for attention partials" {
    const enabled = forward.partialRoceSwitch();
    var g = geometry();
    g.partial_roce = enabled;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.partial_roce = enabled;
    const positions = [_]usize{ 100, 101, 102, 103, 104, 105, 106, 107 };
    const tokens = [_]u32{ 7, 8, 9, 10, 11, 12, 13, 14 };
    for (1..9) |rows| {
        Rec.reset();
        try f.whole(.{ .positions = positions[0..rows], .tokens = tokens[0..rows] });

        var partial_calls: usize = 0;
        for (Rec.gathers.items) |exchange| if (exchange.phase == .attention_partial) {
            partial_calls += 1;
            try std.testing.expectEqual(!enabled, exchange.send_recv);
            try std.testing.expectEqual(rows * 32896, exchange.bytes);
            if (!enabled) try std.testing.expectEqual(exchange.bytes, exchange.recv_bytes);
        };
        try std.testing.expectEqual(@as(usize, 2), partial_calls);

        var merge_after_exchange = false;
        for (Rec.order.items, 0..) |step, i| if (step.kind == .collective and step.what == @backingInt(forward.Phase.attention_partial)) {
            if (i + 1 < Rec.order.items.len) {
                const next = Rec.order.items[i + 1];
                merge_after_exchange = next.kind == .launch and next.what == id("tf_glm53_mla_merge");
            }
        };
        try std.testing.expect(merge_after_exchange);
    }
}

test "decode levers from the environment launch the serving kernels (B0)" {
    const route_compact = forward.routeCompactSelected(if (std.c.getenv("TF_GLM53_ROUTE_COMPACT")) |v| std.mem.span(v) else null);
    const own = forward.ownAttendSwitch();
    const one = forward.qOneGatherSwitch();
    const partial = forward.partialRoceSwitch();
    var g = geometry();
    g.decode_side = true;
    g.route_compact = route_compact;
    g.own_attend = own;
    g.q_one_gather = one;
    g.partial_roce = partial;
    g.decode_mla = one;
    g.decode_mla_v2 = one;
    g.decode_mla_v3 = one;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.mla.attend_kernel = .mma;
    fx.k.mla.attend_verify_fn = func("tf_glm53_attend_verify");
    var f = fx.fwd(0);
    f.decode_side = true;
    f.decode_union = own or one;
    f.route_compact = route_compact;
    f.own_attend = own;
    f.q_one_gather = one;
    f.partial_roce = partial;
    try f.whole(.{ .positions = &.{ 100, 101, 102, 103 }, .tokens = &.{ 7, 8, 9, 10 } });
    var compact: usize = 0;
    var legacy: usize = 0;
    var verify: usize = 0;
    var side_verify: usize = 0;
    for (Rec.order.items) |rec_step| {
        if (rec_step.kind == .launch) {
            compact += @intFromBool(rec_step.what == id("tf_glm_topk_compact"));
            legacy += @intFromBool(rec_step.what == id("tf_glm_topk"));
            verify += @intFromBool(rec_step.what == id("tf_glm53_attend_verify"));
            side_verify += @intFromBool(rec_step.what == id("tf_glm53_attend_verify") and rec_step.stream == 0x51de);
        }
    }
    try std.testing.expectEqual(@as(usize, @intFromBool(route_compact)), compact);
    try std.testing.expectEqual(@as(usize, @intFromBool(!route_compact)), legacy);
    try std.testing.expectEqual(@as(usize, if (own or one) 4 else 0), verify);
    try std.testing.expectEqual(@as(usize, if (own) 2 else 0), side_verify);
    var query_gathers: usize = 0;
    var partials: usize = 0;
    for (Rec.gathers.items) |exchange| {
        if (exchange.phase == .absorbed_query) query_gathers += 1;
        if (exchange.phase == .attention_partial) {
            partials += 1;
            try std.testing.expectEqual(!partial, exchange.send_recv);
        }
    }
    try std.testing.expectEqual(@as(usize, if (one) 2 else 4), query_gathers);
    try std.testing.expectEqual(@as(usize, 2), partials);
}

test "prefill in 1024-row pieces: buffers fit, exchanges equal the plan per piece" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(1);
    var pos: [4096]usize = undefined;
    var ids: [4096]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = 4096 + i;
        t.* = @intCast((i * 7919) % 154880);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try checkContracts(&fx, .prefill, 8191);
    var ls: [2]forward.Layer = undefined;
    const p = try f.stepPlan(.prefill, &pos, 1, &ls);
    try std.testing.expect(p.two_batch_overlap and p.gather_on_prefill);
    try expectPlan(p);
    try std.testing.expectEqual(@as(usize, 2 * 4), recorded(.moe_partial)[1]); // 2 layers x 4 pieces
    try std.testing.expectEqual(@as(usize, 4), recorded(.embed)[1]);
}

fn expectAttnPipeMoeRing(moe_ring: u8) !void {
    var fx: Fixture = undefined;
    var ring_geometry = commPipeGeometry();
    ring_geometry.moe_ring = moe_ring;
    try fx.init(ring_geometry, 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.piece_pipe = true;
    f.moe_pipe = true;
    f.index_pipe = true;
    f.tbo_enabled = false;
    var positions: [4096]usize = undefined;
    var tokens: [4096]u32 = undefined;
    for (&positions, &tokens, 0..) |*position, *token, i| {
        position.* = 4096 + i;
        token.* = @intCast((i * 7919) % 154880);
    }

    try f.pieced(.{ .positions = &positions, .tokens = &tokens });
    const launches_off = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(launches_off);
    const gathers_off = try gpa.dupe(Gather, Rec.gathers.items);
    defer gpa.free(gathers_off);
    Rec.reset();

    f.pipe_coll = pipeColl();
    try f.pieced(.{ .positions = &positions, .tokens = &tokens });
    try std.testing.expectEqual(launches_off.len, Rec.launches.items.len);
    for (launches_off, Rec.launches.items) |want, got| try std.testing.expect(std.meta.eql(want, got));
    try std.testing.expectEqual(gathers_off.len, Rec.gathers.items.len);
    const main_ctx = @intFromPtr(f.coll.ctx);
    const pipe_ctx = @intFromPtr(&pipe_comm);
    const comm_stream = @intFromPtr(f.st.tbo_stream.?.handle.?);
    var comm_stream_gathers: usize = 0;
    for (gathers_off, Rec.gathers.items) |off, on| {
        try std.testing.expect(off.phase == on.phase and off.bytes == on.bytes and off.mine == on.mine and off.all == on.all and off.send_recv == on.send_recv and off.recv_bytes == on.recv_bytes and off.stream == on.stream);
        if (on.stream == comm_stream) {
            comm_stream_gathers += 1;
            try std.testing.expectEqual(main_ctx, off.ctx);
            try std.testing.expectEqual(pipe_ctx, on.ctx);
        } else {
            try std.testing.expectEqual(main_ctx, off.ctx);
            try std.testing.expectEqual(main_ctx, on.ctx);
        }
    }
    try std.testing.expect(comm_stream_gathers > 0);

    // Decode does not use the ordered comm stream, so recordings remain identical with the pipe communicator set.
    Rec.reset();
    f.pipe_coll = null;
    try f.whole(.{ .positions = &.{ 8192, 8193 }, .tokens = &.{ 7, 77441 } });
    const decode_launches = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(decode_launches);
    const decode_gathers = try gpa.dupe(Gather, Rec.gathers.items);
    defer gpa.free(decode_gathers);
    Rec.reset();
    f.pipe_coll = pipeColl();
    try f.whole(.{ .positions = &.{ 8192, 8193 }, .tokens = &.{ 7, 77441 } });
    try std.testing.expectEqual(decode_launches.len, Rec.launches.items.len);
    for (decode_launches, Rec.launches.items) |want, got| try std.testing.expect(std.meta.eql(want, got));
    try std.testing.expectEqual(decode_gathers.len, Rec.gathers.items.len);
    for (decode_gathers, Rec.gathers.items) |want, got| try std.testing.expect(std.meta.eql(want, got));
}

test "ATTN_PIPE moves only comm-stream gathers to the pipe communicator for ring sizes two and eight" {
    try expectAttnPipeMoeRing(2);
    try expectAttnPipeMoeRing(8);
}

test "ATTN_PIPE setup refuses missing comm stream, shared communicators and TBO" {
    var plain: Fixture = undefined;
    try plain.init(geometry(), 16384);
    defer plain.deinit();
    var missing = plain.fwd(0);
    missing.pipe_coll = pipeColl();
    missing.tbo_enabled = false;
    try std.testing.expectError(error.AttnPipeWithoutCommStream, missing.validateSetup());

    var piped: Fixture = undefined;
    try piped.init(commPipeGeometry(), 16384);
    defer piped.deinit();
    var same_main = piped.fwd(0);
    same_main.piece_pipe = true;
    same_main.tbo_enabled = false;
    same_main.pipe_coll = same_main.coll;
    try std.testing.expectError(error.SharedCommunicator, same_main.validateSetup());

    const pf: forward.Prefetch = .{ .side = .{ .d = &driver, .handle = @ptrFromInt(side_handle) }, .coll = sideColl(), .ready = ev(0xe0), .fetched = .{ ev(0xe1), ev(0xe2) }, .attended = .{ ev(0xe3), ev(0xe4) } };
    var same_prefetch = piped.fwd(0);
    same_prefetch.piece_pipe = true;
    same_prefetch.tbo_enabled = false;
    same_prefetch.prefetch = &pf;
    same_prefetch.pipe_coll = pf.coll;
    try std.testing.expectError(error.SharedCommunicator, same_prefetch.validateSetup());

    var tbo_geometry = geometry();
    tbo_geometry.tbo = true;
    var tbo: Fixture = undefined;
    try tbo.init(tbo_geometry, 16384);
    defer tbo.deinit();
    var with_tbo = tbo.fwd(0);
    with_tbo.pipe_coll = pipeColl();
    with_tbo.tbo_enabled = true;
    try std.testing.expectError(error.AttnPipeWithTbo, with_tbo.validateSetup());
}

fn ev(h: usize) cuda.Event {
    return .{ .d = &driver, .handle = @ptrFromInt(h) };
}

const side_handle: usize = 0x5151;

/// The side stream's own communicator (review M3), recorded under a distinct context.
var side_comm: u8 = 0;
fn sideColl() forward.Collective {
    return .{ .ctx = &side_comm, .all_gather_fn = Rec.gather, .send_recv_fn = Rec.sendRecv };
}

var pipe_comm: u8 = 0;
fn pipeColl() forward.Collective {
    return .{ .ctx = &pipe_comm, .all_gather_fn = Rec.gather, .send_recv_fn = Rec.sendRecv };
}

test "prefetch stream ordering: fetch waits on attend two layers back, attend waits on fetch, collective completes before the event" {
    var fx: Fixture = undefined;
    try fx.init(prefetchGeometry(), 16384);
    defer fx.deinit();
    const pf: forward.Prefetch = .{ .side = .{ .d = &driver, .handle = @ptrFromInt(side_handle) }, .coll = sideColl(), .ready = ev(0xe0), .fetched = .{ ev(0xe1), ev(0xe2) }, .attended = .{ ev(0xe3), ev(0xe4) } };
    var f = fx.fwd(0);
    f.prefetch = &pf;
    var pos: [8192]usize = undefined;
    var ids: [8192]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const per_layer = 4; // 4096-row chunks in 1024-row pieces: 4 attends a layer
    var attends: usize = 0; // attend launches so far (flat across chunks)
    var layers_done: usize = 0; // layers whose last piece attended
    var fetches: usize = 0; // prefetchLayer calls so far
    var fetched_recs: usize = 0;
    var side_collectives: usize = 0;
    var last_attended: [2]?usize = .{ null, null };
    var last_main: ?Step = null;
    for (Rec.order.items) |st| {
        const on_side = st.stream == side_handle;
        switch (st.kind) {
            .wait => if (on_side and (st.what == 0xe3 or st.what == 0xe4)) {
                const b = st.what - 0xe3;
                try std.testing.expectEqual(fetches % 2, b);
                // Rule 3: at most one layer ahead. Rule 1: buffer b's last reader is attend(fetch - 2).
                try std.testing.expect(layers_done + 1 >= fetches);
                if (fetches >= 2) try std.testing.expectEqual(@as(?usize, fetches - 2), last_attended[b]);
                fetches += 1;
                side_collectives = 0;
            } else if (!on_side) {
                last_main = st;
            },
            .collective => if (on_side) {
                try std.testing.expectEqual(@backingInt(forward.Phase.stage_fetch), st.what);
                side_collectives += 1;
            } else try std.testing.expect(st.what != @backingInt(forward.Phase.stage_fetch)),
            .record => if (on_side and (st.what == 0xe1 or st.what == 0xe2)) {
                // Rule 4: the event follows the fetch's grouped send/recv on the same stream (chunk 0: nothing to fetch).
                try std.testing.expect(side_collectives == 2 or fx_chunk0(fetched_recs));
                try std.testing.expectEqual(fetched_recs % 2, st.what - 0xe1);
                fetched_recs += 1;
            } else if (!on_side and (st.what == 0xe3 or st.what == 0xe4)) {
                try std.testing.expectEqual(@as(usize, 0), attends % per_layer); // after the layer's last piece
                last_attended[st.what - 0xe3] = layers_done;
                layers_done += 1;
                last_main = st;
            } else if (!on_side) {
                last_main = st;
            },
            .launch => if (!on_side) {
                if (st.what == id("tf_glm53_attend")) {
                    // Rule 2: the step right before a layer's first attend on the compute stream waits on
                    // fetched[L % 2], and that event's record for fetch(L) was already issued.
                    const layer_seq = attends / per_layer;
                    if (attends % per_layer == 0) {
                        const lm = last_main orelse return error.MissingWait;
                        try std.testing.expect(lm.kind == .wait and lm.what == 0xe1 + layer_seq % 2);
                    }
                    try std.testing.expect(fetched_recs > layer_seq);
                    attends += 1;
                }
                last_main = st;
            } else return error.KernelOnSideStream,
        }
    }
    try std.testing.expectEqual(@as(usize, 4 * per_layer), attends);
    try std.testing.expectEqual(@as(usize, 4), fetches);
    // Every stage fetch went through the side communicator, every main collective through the main one.
    for (Rec.gathers.items) |g| try std.testing.expectEqual(g.phase == .stage_fetch, g.ctx == @intFromPtr(&side_comm));
}

/// The first chunk (chunk_start 0) has no peer rows to fetch: its two prefetches record without a collective.
fn fx_chunk0(fetched_recs: usize) bool {
    return fetched_recs < 2;
}

fn uploadsTo(dst: u64) []const Upload {
    var n: usize = 0;
    var store: [64]Upload = undefined;
    for (Rec.uploads.items) |u| if (u.dst == dst) {
        store[n] = u;
        n += 1;
    };
    const S = struct {
        var out: [64]Upload = undefined;
    };
    @memcpy(S.out[0..n], store[0..n]);
    return S.out[0..n];
}

test "a 9k prompt runs as 3 chunks at absolute positions, with stage fetches sized by chunk start" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(1);
    var pos: [9000]usize = undefined;
    var ids: [9000]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    const ups = uploadsTo(fx.st.ws.positions);
    try std.testing.expectEqual(@as(usize, 3), ups.len);
    const starts = [_]usize{ 0, 4096, 8192 };
    const ends = [_]usize{ 4096, 8192, 9000 };
    for (ups, starts, ends) |u, lo, hi| {
        const got = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, u.bytes));
        try std.testing.expectEqual(hi - lo, got.len);
        for (got, lo..) |v, p| try std.testing.expectEqual(@as(i64, @intCast(p)), v);
    }
    // Rank 1 sends its own rows below chunk_start (odd positions): 2048 then 4096 slots, two layers each.
    var sent: usize = 0;
    for (Rec.gathers.items) |g| if (g.phase == .stage_fetch) {
        sent += g.bytes;
    };
    try std.testing.expectEqual(@as(usize, 2 * (2048 + 4096) * 1152), sent);
    // The head ran once, on the prompt's last row.
    var heads: usize = 0;
    for (Rec.launches.items) |l| if (l.name == id("tf_glm53_top1_f32")) {
        try std.testing.expectEqual(@as(u32, 1), l.grid[0]);
        heads += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), heads);
}

test "odd chunk starts count the fetch per owner, not per world" {
    try std.testing.expectEqual(@as(usize, 3), forward.stageFetchSlots(5, 0)); // positions 0, 2, 4
    try std.testing.expectEqual(@as(usize, 2), forward.stageFetchSlots(5, 1)); // positions 1, 3
    try std.testing.expectEqual(@as(usize, 0), forward.stageFetchSlots(1, 1));
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(1);
    try f.pieced(.{ .positions = &.{ 4097, 4098, 4099 }, .tokens = &.{ 1, 2, 3 } });
    for (Rec.gathers.items) |g| if (g.phase == .stage_fetch) {
        const row: usize = if (g.bytes % 1024 == 0 and g.bytes >= 1024 and g.bytes / 1024 == 2048) 1024 else 128;
        try std.testing.expectEqual(@as(usize, 2048) * row, g.bytes); // rank 1 owns 2048 positions below 4097
        try std.testing.expectEqual(@as(usize, 2049) * row, g.recv_bytes); // rank 0 owns 2049
    };
}

const Snapshot = struct {
    launches: []Launch,
    marks: []usize,
    rows: [6][]u8, // concatenated row inputs by field: positions, dense_len, cos, sin, ids, owners

    fn take(fx: *const Fixture) !Snapshot {
        const ws = &fx.st.ws;
        const fields = [_]u64{ ws.positions, ws.dense_len, ws.cos, ws.sin, ws.ids, ws.owners };
        var s: Snapshot = .{ .launches = try gpa.dupe(Launch, Rec.launches.items), .marks = try gpa.dupe(usize, Rec.marks.items), .rows = undefined };
        for (fields, &s.rows) |dst, *out| {
            var list: std.ArrayList(u8) = .empty;
            for (Rec.uploads.items) |u| if (u.dst == dst) try list.appendSlice(gpa, u.bytes);
            out.* = try list.toOwnedSlice(gpa);
        }
        Rec.reset();
        return s;
    }

    fn deinit(s: *Snapshot) void {
        gpa.free(s.launches);
        gpa.free(s.marks);
        for (s.rows) |r| gpa.free(r);
    }

    /// Launches of the `k`-th window (chunk): from its row-input upload to the next one's (or the end).
    fn chunk(s: *const Snapshot, k: usize) []const Launch {
        const lo = s.marks[k];
        const hi = if (k + 1 < s.marks.len) s.marks[k + 1] else s.launches.len;
        return s.launches[lo..hi];
    }
};

fn sameLaunches(a: []const Launch, b: []const Launch) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try std.testing.expectEqual(x.name, y.name);
        try std.testing.expectEqual(x.grid, y.grid);
        try std.testing.expectEqualSlices(u64, x.args[0..x.nargs], y.args[0..y.nargs]);
    }
}

test "chunk invariance: a resumed prompt launches what a fresh one does; per-row inputs follow absolute positions" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    const f = fx.fwd(0);
    var pos: [6000]usize = undefined;
    var ids: [6000]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast((i * 31) % 154880);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var fresh = try Snapshot.take(&fx);
    defer fresh.deinit();
    try f.pieced(.{ .positions = pos[0..4096], .tokens = ids[0..4096] });
    try f.pieced(.{ .positions = pos[4096..], .tokens = ids[4096..] });
    var resumed = try Snapshot.take(&fx);
    defer resumed.deinit();
    // The resumed first call adds a head on row 4095 (4 launches); everything else is identical, argument for argument.
    const c0 = fresh.chunk(0);
    const r0 = resumed.chunk(0);
    try sameLaunches(c0, r0[0 .. r0.len - 4]);
    try sameLaunches(fresh.chunk(1), resumed.chunk(1));
    for (fresh.rows, resumed.rows) |a, b| try std.testing.expectEqualSlices(u8, a, b);

    // One 3000-row chunk vs three chunks of 1024: identical per-row inputs and the same planes and scalars.
    var g = fx.fwd(0);
    try g.pieced(.{ .positions = pos[0..3000], .tokens = ids[0..3000] });
    var one = try Snapshot.take(&fx);
    defer one.deinit();
    g.chunk_rows = 1024;
    try g.pieced(.{ .positions = pos[0..3000], .tokens = ids[0..3000] });
    var three = try Snapshot.take(&fx);
    defer three.deinit();
    for (one.rows, three.rows) |a, b| try std.testing.expectEqualSlices(u8, a, b);
    // Pieces cut at absolute multiples of 1024 inside the single chunk, so its kv writes are the three chunks'
    // kv writes argument for argument (planes, stage pair, row-input offsets, rows, stride, rank, world).
    var kv_one: std.ArrayList(Launch) = .empty;
    defer kv_one.deinit(gpa);
    var kv_three: std.ArrayList(Launch) = .empty;
    defer kv_three.deinit(gpa);
    // Layer-major in one window, chunk-major across three: compare each layer's kv writes in order.
    for (0..2) |layer| {
        for (one.launches) |l| if (l.name == id("tf_glm53_kv_write") and l.args[5] == fx.st.latentAt(layer)) try kv_one.append(gpa, l);
        for (three.launches) |l| if (l.name == id("tf_glm53_kv_write") and l.args[5] == fx.st.latentAt(layer)) try kv_three.append(gpa, l);
    }
    try std.testing.expectEqual(@as(usize, 2 * 3), kv_one.items.len);
    // Row-input offsets differ only by the window base (one window vs three): compare all but those pointers.
    for (kv_one.items, kv_three.items) |x, y| try std.testing.expectEqualSlices(u64, &.{ x.args[5], x.args[6], x.args[7], x.args[8], x.args[9], x.args[10], x.args[11], x.args[12] }, &.{ y.args[5], y.args[6], y.args[7], y.args[8], y.args[9], y.args[10], y.args[11], y.args[12] });
}

test "a prefill past the context, or a prefetch over aliased stage pairs, fails closed before any launch" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 8192);
    defer fx.deinit();
    var f = fx.fwd(0);
    var pos: [600]usize = undefined;
    var ids: [600]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = 8000 + i;
        t.* = 1;
    }
    try std.testing.expectError(error.ContextFull, f.pieced(.{ .positions = &pos, .tokens = &ids }));
    try std.testing.expectEqual(@as(usize, 0), Rec.launches.items.len);
    const pf: forward.Prefetch = .{ .side = .{ .d = &driver, .handle = @ptrFromInt(side_handle) }, .coll = sideColl(), .ready = ev(0xe0), .fetched = .{ ev(0xe1), ev(0xe2) }, .attended = .{ ev(0xe3), ev(0xe4) } };
    f.prefetch = &pf;
    fx.st.stage.buffers[1] = fx.st.stage.buffers[0];
    try std.testing.expectError(error.StageBuffersAliased, f.pieced(.{ .positions = pos[0..100], .tokens = ids[0..100] }));
    for (Rec.gathers.items) |g| try std.testing.expect(g.phase != .stage_fetch);
}

test "candidate heads fill WP10's sentinels before each 64-row part/merge tile" {
    var g = geometry();
    g.head_rows = 64;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    f.head = .{ .candidates = 64 };
    f.gate_mode = true;
    var pos: [100]usize = undefined;
    var ids: [100]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = 1;
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    var heads: std.ArrayList(usize) = .empty;
    defer heads.deinit(gpa);
    const part_name = if (candFastFromEnv()) id("tf_glm53_cand_part_fast_f32") else id("tf_glm53_cand_part_f32");
    const v2 = candV2FromEnv();
    for (Rec.launches.items) |l| {
        if (v2) {
            if (l.name == id("tf_glm53_cand2_fill_f32") or l.name == id("tf_glm53_cand2_hist_f32") or l.name == id("tf_glm53_cand2_select_f32")) try heads.append(gpa, l.name);
        } else if (l.name == id("tf_glm53_cand_fill_f32") or l.name == part_name or l.name == id("tf_glm53_cand_merge_f32")) try heads.append(gpa, l.name);
    }
    // 100 gate rows in head blocks of 64: two candidate-kernel tiles.
    const tile = if (v2) [_]usize{ id("tf_glm53_cand2_fill_f32"), id("tf_glm53_cand2_hist_f32"), id("tf_glm53_cand2_select_f32") } else [_]usize{ id("tf_glm53_cand_fill_f32"), part_name, id("tf_glm53_cand_merge_f32") };
    const want = tile ++ tile;
    try std.testing.expectEqualSlices(usize, &want, heads.items);
    try std.testing.expectEqual(@as(usize, 2), recorded(.head)[1]);
}

test "CAND_V2 from the environment: fill2, hist and select per tile where the scratch holds 16 KiB a row" {
    const on = candV2FromEnv();
    for ([_]usize{ 64, 32 }) |kc| {
        var g = geometry();
        g.head_rows = 64;
        g.candidate_k = kc;
        var fx: Fixture = undefined;
        try fx.init(g, 16384);
        defer fx.deinit();
        var f = fx.fwd(0);
        f.head = .{ .candidates = kc };
        f.gate_mode = true;
        var pos: [100]usize = undefined;
        var ids: [100]u32 = undefined;
        for (&pos, &ids, 0..) |*p, *t, i| {
            p.* = i;
            t.* = 1;
        }
        try f.pieced(.{ .positions = &pos, .tokens = &ids });
        const v2_used = on and kc == 64;
        var selects: usize = 0;
        var merges: usize = 0;
        for (Rec.launches.items) |l| {
            if (l.name == id("tf_glm53_cand2_select_f32")) {
                try std.testing.expectEqual(@as(u32, 1024), l.block[0]);
                try std.testing.expectEqual(@as(u32, 35_880), l.shared);
                try std.testing.expect(l.grid[0] == 64 or l.grid[0] == 36);
                selects += 1;
            }
            if (l.name == id("tf_glm53_cand2_hist_f32")) try std.testing.expectEqual(@as(u32, 32), l.grid[1]);
            merges += @intFromBool(l.name == id("tf_glm53_cand_merge_f32"));
        }
        try std.testing.expectEqual(@as(usize, if (v2_used) 2 else 0), selects);
        try std.testing.expectEqual(@as(usize, if (v2_used) 0 else 2), merges);
    }
}

const FakeHead = struct {
    var rows: [2][8]struct { v: f32, id: u32 } = undefined;
    fn download(_: ?*anyopaque, _: cuda.Stream, _: u64, dst: []u8) anyerror!void {
        const per = dst.len / 2;
        const n = per / 16;
        if (n > 8) { // a logits read (replay parity): a fixed pattern stands in for the device rows
            for (dst, 0..) |*b, i| b.* = @truncate(i *% 131);
            return;
        }
        for (0..2) |q| for (0..n) |r| {
            std.mem.writeInt(u32, dst[q * per + r * 4 ..][0..4], @bitCast(rows[q][r].v), .little);
            std.mem.writeInt(u32, dst[q * per + n * 4 + r * 4 ..][0..4], rows[q][r].id, .little);
        };
    }
};

const HookEvent = struct { kind: enum { restart, prompt, verify, draft }, a: usize, b: usize = 0, c: u32 = 0 };

const HookProbe = struct {
    restarts: usize = 0,
    restart_from: usize = 0,
    prompts: usize = 0,
    prompt_start: usize = 0,
    prompt_rows: usize = 0,
    verifies: usize = 0,
    verify_rows: usize = 0,
    verify_pos: usize = 0,
    drafts: usize = 0,
    draft_pos: usize = 0,
    draft_rows: usize = 0,
    draft_anchor: u32 = 0,
    events: [16]HookEvent = undefined,
    event_count: usize = 0,

    fn record(p: *HookProbe, event: HookEvent) void {
        p.events[p.event_count] = event;
        p.event_count += 1;
    }

    fn restart(ctx: *anyopaque, from: usize) anyerror!void {
        const p: *HookProbe = @ptrCast(@alignCast(ctx));
        p.restarts += 1;
        p.restart_from = from;
        p.record(.{ .kind = .restart, .a = from });
    }
    fn rewind(ctx: *anyopaque, _: usize, from: usize) anyerror!void {
        return restart(ctx, from);
    }
    fn prompt(ctx: *anyopaque, start: usize, rows: usize) anyerror!void {
        const p: *HookProbe = @ptrCast(@alignCast(ctx));
        p.prompts += 1;
        p.prompt_start = start;
        p.prompt_rows = rows;
        p.record(.{ .kind = .prompt, .a = start, .b = rows });
    }
    fn verify(ctx: *anyopaque, rows: usize, pos0: usize) anyerror!void {
        const p: *HookProbe = @ptrCast(@alignCast(ctx));
        p.verifies += 1;
        p.verify_rows = rows;
        p.verify_pos = pos0;
        p.record(.{ .kind = .verify, .a = rows, .b = pos0 });
    }
    fn draft(ctx: *anyopaque, pos: usize, n: usize, anchor: u32, _: ?[]u32, _: ?[]f64, _: u64, _: f64) anyerror!void {
        const p: *HookProbe = @ptrCast(@alignCast(ctx));
        p.drafts += 1;
        p.draft_pos = pos;
        p.draft_rows = n;
        p.draft_anchor = anchor;
        p.record(.{ .kind = .draft, .a = pos, .b = n, .c = anchor });
    }
};

test "engine drafter hooks: restart, ingest, verify and draft bookkeeping" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 8192);
    defer fx.deinit();
    var probe: HookProbe = .{};
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false, .defer_draft_fill = true };
    defer e.deinit();
    e.drafter = .{
        .ctx = @ptrCast(&probe),
        .window = 2048,
        .taps_prefill = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{0}, .cap_rows = 2048 },
        .taps_step = .{ .f = func("df_tap_glm"), .out = 0x9b00_1000, .layers = &.{0}, .cap_rows = 8 },
        .restart_fn = HookProbe.restart,
        .rewind_fn = HookProbe.rewind,
        .prompt_fn = HookProbe.prompt,
        .verify_fn = HookProbe.verify,
        .draft_fn = HookProbe.draft,
    };
    _ = try e.prefill(&.{ 5, 6, 7 });
    try std.testing.expectEqual(@as(usize, 0), probe.restarts);
    try std.testing.expectEqual(@as(usize, 0), probe.prompts);
    FakeHead.rows[0][0] = .{ .v = 1, .id = 10 };
    FakeHead.rows[1][0] = .{ .v = 0, .id = 20 };
    FakeHead.rows[0][1] = .{ .v = 1, .id = 11 };
    FakeHead.rows[1][1] = .{ .v = 0, .id = 21 };
    FakeHead.rows[0][2] = .{ .v = 1, .id = 12 };
    FakeHead.rows[1][2] = .{ .v = 0, .id = 22 };
    var out: [3]u32 = undefined;
    try e.verify(&.{ 1, 2, 3 }, &out);
    try std.testing.expectEqual(@as(usize, 1), probe.prompts);
    try std.testing.expectEqual(@as(usize, 0), probe.prompt_start);
    try std.testing.expectEqual(@as(usize, 3), probe.prompt_rows);
    try std.testing.expectEqual(@as(usize, 1), probe.verifies);
    try std.testing.expectEqual(@as(usize, 3), probe.verify_rows);
    try std.testing.expectEqual(@as(usize, 3), probe.verify_pos);
    try e.keep(2);
    try e.draftBlock(5, 2, 9, null, null);
    try std.testing.expectEqual(@as(usize, 1), probe.drafts);
    try std.testing.expectEqual(@as(usize, 5), probe.draft_pos);
    try std.testing.expectEqual(@as(usize, 2), probe.draft_rows);
    try std.testing.expectEqual(@as(u32, 9), probe.draft_anchor);
    try std.testing.expectEqual(@as(usize, 5), e.tapped);
    try e.rollback(3);
    _ = try e.prefill(&.{ 8, 9 });
    try std.testing.expectEqual(@as(usize, 0), probe.restarts);
    try std.testing.expectEqual(@as(usize, 1), probe.prompts);
    try e.draftBlock(5, 1, 9, null, null);
    try std.testing.expectEqual(@as(usize, 2), probe.restarts);
    try std.testing.expectEqual(@as(usize, 5), probe.restart_from);
    try std.testing.expectEqual(@as(usize, 2), probe.prompts);
    try std.testing.expectEqual(@as(usize, 3), probe.prompt_start);
    try std.testing.expectEqual(@as(usize, 2), probe.prompt_rows);
    try std.testing.expectEqualSlices(HookEvent, &.{
        .{ .kind = .prompt, .a = 0, .b = 3 },
        .{ .kind = .verify, .a = 3, .b = 3 },
        .{ .kind = .draft, .a = 5, .b = 2, .c = 9 },
        .{ .kind = .restart, .a = 3 },
        .{ .kind = .prompt, .a = 3, .b = 2 },
        .{ .kind = .restart, .a = 5 },
        .{ .kind = .draft, .a = 5, .b = 1, .c = 9 },
    }, probe.events[0..probe.event_count]);
}

test "last-layer row pruning writes every KV row and attends only the head row" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 8192);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(1, true, true);
    fx.model.layers = fx.layer_store[0..1];
    fx.model.config.layers = 2;
    var f = fx.fwd(0);
    f.row_prune = true;
    try f.pieced(.{ .positions = &.{ 0, 1, 2 }, .tokens = &.{ 5, 6, 7 } });
    var kv_rows: usize = 0;
    var attend_rows: usize = 0;
    var ffn_rows: usize = 0;
    // SwiGLU's grid uses this rank's 6,144-wide slice, so convert blocks back to rows with that width.
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_kv_write")) kv_rows += @intCast(launch.args[9]);
        if (launch.name == id("tf_glm53_attend")) attend_rows += @intCast(launch.args[13]);
        if (launch.name == id("tf_glm53_swiglu_plain")) ffn_rows += @intCast(launch.grid[0] * 256 / 6144);
    }
    try std.testing.expectEqual(@as(usize, 3), kv_rows);
    try std.testing.expectEqual(@as(usize, 1), attend_rows);
    try std.testing.expectEqual(@as(usize, 1), ffn_rows);

    Rec.reset();
    f.taps = .{ .f = func("df_tap_glm"), .out = 0x9b00_0000, .layers = &.{1}, .from = 1, .cap_rows = 3 };
    try f.pieced(.{ .positions = &.{ 0, 1, 2 }, .tokens = &.{ 5, 6, 7 } });
    kv_rows = 0;
    attend_rows = 0;
    ffn_rows = 0;
    var tap_rows: usize = 0;
    for (Rec.launches.items) |launch| {
        if (launch.name == id("tf_glm53_kv_write")) kv_rows += @intCast(launch.args[9]);
        if (launch.name == id("tf_glm53_attend")) attend_rows += @intCast(launch.args[13]);
        if (launch.name == id("tf_glm53_swiglu_plain")) ffn_rows += @intCast(launch.grid[0] * 256 / 6144);
        if (launch.name == id("df_tap_glm")) tap_rows += @intCast(launch.grid[0]);
    }
    try std.testing.expectEqual(@as(usize, 3), kv_rows);
    try std.testing.expectEqual(@as(usize, 2), attend_rows);
    try std.testing.expectEqual(@as(usize, 2), ffn_rows);
    try std.testing.expectEqual(@as(usize, 2), tap_rows);
}

test "last-layer row pruning preserves the host fixture's first token" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 8192);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(1, true, true);
    fx.model.layers = fx.layer_store[0..1];
    fx.model.config.layers = 2;
    FakeHead.rows[0][0] = .{ .v = 4, .id = 10 };
    FakeHead.rows[1][0] = .{ .v = 3, .id = 20 };
    var eager: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer eager.deinit();
    const baseline = try eager.prefill(&.{ 5, 6, 7 });
    Rec.reset();
    var pruned_forward = fx.fwd(0);
    pruned_forward.row_prune = true;
    var pruned: engine.Engine = .{ .gpa = gpa, .f = pruned_forward, .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer pruned.deinit();
    const candidate = try pruned.prefill(&.{ 5, 6, 7 });
    try std.testing.expectEqual(baseline, candidate);
}

test "engine: verify draws greedy tokens across ranks with ties to the lower id; keep and rollback move the position" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer e.deinit();
    FakeHead.rows[0][0] = .{ .v = 2.0, .id = 11 };
    FakeHead.rows[1][0] = .{ .v = 2.0, .id = 77440 + 5 };
    FakeHead.rows[0][1] = .{ .v = 1.0, .id = 3 };
    FakeHead.rows[1][1] = .{ .v = 1.5, .id = 77440 + 9 };
    FakeHead.rows[0][2] = .{ .v = std.math.nan(f32), .id = 4 };
    FakeHead.rows[1][2] = .{ .v = -1.0, .id = 77440 + 1 };
    const first = try e.prefill(&.{ 5, 6, 7 });
    try std.testing.expectEqual(@as(u32, 11), first); // prefill's head row: rank 0 and rank 1 tie, lower id wins
    try std.testing.expectEqual(@as(usize, 3), e.pos);
    var out: [3]u32 = undefined;
    try e.verify(&.{ 1, 2, 3 }, &out);
    try std.testing.expectEqualSlices(u32, &.{ 11, 77440 + 9, 77440 + 1 }, &out);
    try std.testing.expectEqual(@as(usize, 3), e.pos);
    try e.keep(2);
    try std.testing.expectEqual(@as(usize, 5), e.pos);
    try e.rollback(4);
    try std.testing.expectEqual(@as(usize, 4), e.pos);
    try std.testing.expect(e.keys.contains(try engine.decodeKey(3, 5)));
    try std.testing.expect(e.keys.contains(try engine.prefillKey(3)));
}

test "engine: TF_GLM53_LAZY_DRAW stops drawing at the first rejection" {
    const lazy = if (std.c.getenv("TF_GLM53_LAZY_DRAW")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false, .lazy_draw = lazy };
    defer e.deinit();
    FakeHead.rows[0][0] = .{ .v = 2.0, .id = 11 };
    FakeHead.rows[1][0] = .{ .v = 2.0, .id = 77440 + 5 };
    FakeHead.rows[0][1] = .{ .v = 1.0, .id = 3 };
    FakeHead.rows[1][1] = .{ .v = 1.5, .id = 77440 + 9 };
    FakeHead.rows[0][2] = .{ .v = std.math.nan(f32), .id = 4 };
    FakeHead.rows[1][2] = .{ .v = -1.0, .id = 77440 + 1 };
    _ = try e.prefill(&.{ 5, 6, 7 });
    Rec.reset();
    var out: [3]u32 = undefined;
    try e.verify(&.{ 1, 11, 3 }, &out);
    const expected: []const u32 = if (lazy) &.{ 11, 77440 + 9, engine.lazy_undrawn } else &.{ 11, 77440 + 9, 77440 + 1 };
    try std.testing.expectEqualSlices(u32, expected, &out);
    const launch_count = Rec.launches.items.len;
    const launches = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(launches);
    e.full_draw = true;
    try e.verify(&.{ 1, 11, 3 }, &out);
    try std.testing.expectEqualSlices(u32, &.{ 11, 77440 + 9, 77440 + 1 }, &out);
    try std.testing.expect(!e.full_draw);
    try std.testing.expectEqual(launch_count * 2, Rec.launches.items.len);
    try std.testing.expectEqualSlices(Launch, launches, Rec.launches.items[launch_count..]);
}

const FakeGraphs = struct {
    var captured: usize = 0;
    var replayed: usize = 0;
    var exec: u8 = 0;
    fn capture(_: *anyopaque, stream: cuda.Stream, body: cuda.graph_cache.Body) anyerror!*anyopaque {
        try body.run(body.ctx, stream); // stands in for stream capture: the body's launches are recorded once
        captured += 1;
        return &exec;
    }
    fn launch(_: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        replayed += 1;
    }
    fn free(_: *anyopaque, _: *anyopaque) void {}
};

test "the side communicator's open path is analysed" {
    std.testing.refAllDecls(forward.SideGroup);
}

test "decode windows go through keyed CUDA graphs: one capture per (rows, position bucket), then replays" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var dummy: u8 = 0;
    var cache = cuda.graph_cache.Cache(engine.GraphKey).init(gpa, .{ .ctx = &dummy, .capture = FakeGraphs.capture, .launch = FakeGraphs.launch, .free = FakeGraphs.free }, .{});
    defer cache.deinit();
    FakeGraphs.captured = 0;
    FakeGraphs.replayed = 0;
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .graphs = &cache, .pos = 3000 };
    defer e.deinit();
    for (0..2) |r| FakeHead.rows[r][0] = .{ .v = 1, .id = @intCast(r) };
    var out: [1]u32 = undefined;
    try e.verify(&.{9}, &out);
    const launches = Rec.launches.items.len;
    try e.keep(1);
    try e.verify(&.{9}, &out); // same rows, same bucket (4096): replayed, no new launches recorded
    try std.testing.expectEqual(launches, Rec.launches.items.len);
    if (cache.settings.on) {
        try std.testing.expectEqual(@as(usize, 1), FakeGraphs.captured);
        try std.testing.expect(FakeGraphs.replayed >= 1);
    }
    e.pos = 4096; // position 4096 enters the next bucket: a new key
    try e.verify(&.{9}, &out);
    try std.testing.expect(Rec.launches.items.len > launches);
    try std.testing.expectEqual(@as(usize, 2), e.keys.count());
}

test "decode graphs warm each key eagerly before capturing it: the first window's send/recv never runs in a capture" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var dummy: u8 = 0;
    // The engine's own decode settings (warm = 1) over the recording engine.
    var cache = cuda.graph_cache.Cache(engine.GraphKey).init(gpa, .{ .ctx = &dummy, .capture = FakeGraphs.capture, .launch = FakeGraphs.launch, .free = FakeGraphs.free }, engine.decode_graphs);
    defer cache.deinit();
    if (!cache.settings.on) return; // TF_GRAPH_CACHE=0 in this environment: nothing to check
    FakeGraphs.captured = 0;
    FakeGraphs.replayed = 0;
    // A short prompt's first decode window: positions 35.., one key (rows 1, bucket 2048) for every step.
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .graphs = &cache, .pos = 35, .frontier = 35 };
    defer e.deinit();
    for (0..2) |r| FakeHead.rows[r][0] = .{ .v = 1, .id = @intCast(r) };
    var out: [1]u32 = undefined;
    try e.verify(&.{9}, &out);
    // Decode 0 ran eagerly, outside any capture, and it holds the communicator's first point-to-point exchange.
    try std.testing.expectEqual(cuda.graph_cache.Outcome.eager, e.last_outcome);
    try std.testing.expectEqual(@as(usize, 0), FakeGraphs.captured);
    var p2p: usize = 0;
    for (Rec.gathers.items) |g| p2p += @intFromBool(g.send_recv);
    try std.testing.expect(p2p > 0);
    const eager_launches = Rec.launches.items.len;
    try e.keep(1);
    try e.verify(&.{9}, &out); // decode 1: the same key, now captured (the recording engine runs the body once)
    try std.testing.expectEqual(cuda.graph_cache.Outcome.captured, e.last_outcome);
    try std.testing.expectEqual(@as(usize, 1), FakeGraphs.captured);
    try std.testing.expectEqual(2 * eager_launches, Rec.launches.items.len); // the capture records decode 0's launches
    try e.keep(1);
    for (0..3) |_| { // decode 2..4 replay: nothing new is launched or captured
        try e.verify(&.{9}, &out);
        try std.testing.expectEqual(cuda.graph_cache.Outcome.replayed, e.last_outcome);
        try e.keep(1);
    }
    try std.testing.expectEqual(2 * eager_launches, Rec.launches.items.len);
    try std.testing.expectEqual(@as(usize, 1), FakeGraphs.captured);
    try std.testing.expectEqual(@as(usize, 4), FakeGraphs.replayed); // the capture's own launch + 3 replays
    try std.testing.expectEqual(@as(u64, 1), cache.stats.warmed);
}

/// Output arguments by kernel (what a launch writes). Everything else a kernel takes is read-only for it.
fn outputs(name: usize) []const usize {
    const table = .{
        .{ "tf_glm53_kv_write", &[_]usize{ 5, 6, 7, 8 } }, .{ "tf_glm53_qpe_pack", &[_]usize{3} },       .{ "tf_glm53_absorb", &[_]usize{3} },
        .{ "tf_glm53_attend", &[_]usize{12} },             .{ "tf_glm53_attend_mma", &[_]usize{ 12, 18 } },  .{ "tf_glm53_smla_p", &[_]usize{12} },  .{ "tf_glm53_smla_p_fp8", &[_]usize{12} },  .{ "tf_glm53_attend_split_merge", &[_]usize{1} },
        .{ "tf_glm53_attend_verify", &[_]usize{7} },
        .{ "tf_glm53_mla_merge", &[_]usize{ 2, 3 } }, .{ "tf_glm53_expand_v", &[_]usize{3} },
        .{ "tf_glm53_index_key", &[_]usize{6} },           .{ "tf_glm53_index_q", &[_]usize{3} },        .{ "tf_glm53_index_scores", &[_]usize{6} },
        .{ "tf_glm53_merge", &[_]usize{ 1, 2, 3, 4 } },    .{ "tf_glm_rmsnorm_6144", &[_]usize{3} },     .{ "tf_glm_rmsnorm_2048", &[_]usize{3} },
        .{ "tf_glm_residual_add", &[_]usize{1} },          .{ "tf_glm_embed_b16", &[_]usize{2} },        .{ "tf_glm53_embed_pick", &[_]usize{2} },
        .{ "qmmf_bf16", &[_]usize{ 4, 5 } },               .{ "qmmf_f32", &[_]usize{ 4, 5 } },           .{ "tf_glm53_top1_f32", &[_]usize{ 3, 4 } }, .{ "tf_glm53_mask_f32", &[_]usize{0} },
        .{ "tf_glm53_lse_f32", &[_]usize{2} },             .{ "exl3_down_combine", &[_]usize{ 3, 5 } },
    };
    inline for (table) |t| if (name == id(t[0])) return t[1];
    return &.{};
}

fn inside(p: u64, buf: cuda.DeviceBuffer) bool {
    return p >= buf.ptr and p < buf.ptr + buf.len;
}

test "a captured decode graph never writes into weights, constants or row inputs (only scratch and caches)" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var dummy: u8 = 0;
    var cache = cuda.graph_cache.Cache(engine.GraphKey).init(gpa, .{ .ctx = &dummy, .capture = FakeGraphs.capture, .launch = FakeGraphs.launch, .free = FakeGraphs.free }, .{});
    defer cache.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .graphs = &cache, .pos = 3000 };
    defer e.deinit();
    for (0..2) |r| for (0..8) |j| {
        FakeHead.rows[r][j] = .{ .v = 1, .id = @intCast(j) };
    };
    var out: [4]u32 = undefined;
    try e.verify(&.{ 1, 2, 3, 4 }, &out);
    const ws = &fx.st.ws;
    const inputs = [_][2]u64{ .{ ws.positions, ws.sizes.positions }, .{ ws.dense_len, ws.sizes.dense_len }, .{ ws.cos, ws.sizes.cos }, .{ ws.sin, ws.sizes.sin }, .{ ws.ids, ws.sizes.ids }, .{ ws.owners, ws.sizes.owners } };
    var writes: usize = 0;
    for (Rec.launches.items) |l| for (outputs(l.name)) |i| {
        const p = l.args[i];
        if (p == 0) continue; // absent stage planes (decode) and unused split-K partials
        // Weights and norms of the fixture live in [0x8000_0000, 0xb000_0000); State buffers come from the hook.
        try std.testing.expect(p < 0x8000_0000 or p >= 0xb000_0000);
        for (inputs) |in| try std.testing.expect(p < in[0] or p >= in[0] + in[1]);
        try std.testing.expect(inside(p, fx.st.workspace) or inside(p, fx.st.latent) or inside(p, fx.st.rope) or inside(p, fx.st.index_keys) or inside(p, fx.st.stage_plane));
        writes += 1;
    };
    try std.testing.expect(writes > 40);
}

test "replay parity hook: live and graph-replayed logits compare bit for bit" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var dummy: u8 = 0;
    var cache = cuda.graph_cache.Cache(engine.GraphKey).init(gpa, .{ .ctx = &dummy, .capture = FakeGraphs.capture, .launch = FakeGraphs.launch, .free = FakeGraphs.free }, .{});
    defer cache.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .graphs = &cache, .pos = 100 };
    defer e.deinit();
    try e.replayParity(&.{ 1, 2 });
    try std.testing.expectEqual(@as(usize, 100), e.pos);
    // A replay whose results differ is caught.
    const Drift = struct {
        var calls: usize = 0;
        fn download(ctx: ?*anyopaque, s: cuda.Stream, src: u64, dst: []u8) anyerror!void {
            try FakeHead.download(ctx, s, src, dst);
            calls += 1;
            if (calls > 2 and dst.len > 0) dst[0] ^= 1;
        }
    };
    e.download = .{ .download_fn = Drift.download };
    try std.testing.expectError(error.GraphReplayMismatch, e.replayParity(&.{ 1, 2 }));
}

test "continuity: prompt, decoded reply, next turn re-prefilled from the frontier == a fresh prefill of the transcript" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    for (0..2) |r| for (0..8) |j| {
        FakeHead.rows[r][j] = .{ .v = 1, .id = @intCast(j) };
    };
    var transcript: [4096 + 5 + 10]u32 = undefined;
    for (&transcript, 0..) |*t, i| t.* = @intCast((i * 131) % 154880);
    const prompt = transcript[0..4096];
    const reply = transcript[4096..4101];
    const turn = transcript[4101..];
    // Fresh: the whole transcript in one prefill (chunks [0,4096) and [4096,4111)).
    var fresh_e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer fresh_e.deinit();
    _ = try fresh_e.prefill(&transcript);
    var fresh = try Snapshot.take(&fx);
    defer fresh.deinit();
    // Resume: prompt, a decoded reply (verify + keep), then the next turn.
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer e.deinit();
    _ = try e.prefill(prompt);
    var out: [5]u32 = undefined;
    try e.verify(reply, &out);
    try e.keep(5);
    try std.testing.expectEqual(@as(usize, 4096), e.frontier); // decode rows are never kept as prefill KV
    try std.testing.expectEqual(@as(usize, 4101), e.pos);
    try std.testing.expectEqualSlices(u32, reply, e.decodedSinceFrontier());
    try std.testing.expectError(error.DecodeRowsNotReprefilled, e.prefill(turn));
    Rec.reset();
    _ = try e.nextTurn(turn);
    var resumed = try Snapshot.take(&fx);
    defer resumed.deinit();
    try std.testing.expectEqual(@as(usize, 4111), e.frontier);
    try std.testing.expectEqualSlices(u32, &transcript, e.history.items);
    // The re-prefill window is the fresh run's second chunk, launch for launch and argument for argument:
    // the reply rows' KV (and indexer keys) come from the prefill path, as in a fresh prefill.
    try std.testing.expectEqual(@as(usize, 1), resumed.marks.len);
    try sameLaunches(fresh.chunk(1), resumed.chunk(0));
    for (fresh.rows, resumed.rows) |a, b| try std.testing.expectEqualSlices(u8, a[a.len - b.len ..], b);
    // The warm-pass entry point drops decode rows without re-prefilling them.
    try e.verify(&.{ 1, 2 }, out[0..2]);
    try e.keep(2);
    e.rewindToFrontier();
    try std.testing.expectEqual(e.frontier, e.pos);
    try std.testing.expectEqual(@as(usize, 4111), e.history.items.len);
}

test "the stage prefetch refuses to share the main communicator, and a single stage pair" {
    var fx: Fixture = undefined;
    try fx.init(prefetchGeometry(), 16384);
    defer fx.deinit();
    var f = fx.fwd(0);
    const pf: forward.Prefetch = .{ .side = .{ .d = &driver, .handle = @ptrFromInt(side_handle) }, .coll = f.coll, .ready = ev(0xe0), .fetched = .{ ev(0xe1), ev(0xe2) }, .attended = .{ ev(0xe3), ev(0xe4) } };
    f.prefetch = &pf;
    var pos: [10]usize = undefined;
    for (&pos, 0..) |*p, i| p.* = 4096 + i;
    try std.testing.expectError(error.SharedCommunicator, f.pieced(.{ .positions = &pos, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } }));
    try std.testing.expectEqual(@as(usize, 0), Rec.launches.items.len);
    // A State carved with one stage pair (no prefetch stream) cannot serve a prefetch.
    var one: Fixture = undefined;
    try one.init(geometry(), 16384);
    defer one.deinit();
    var g = one.fwd(0);
    const pf2: forward.Prefetch = .{ .side = .{ .d = &driver, .handle = @ptrFromInt(side_handle) }, .coll = sideColl(), .ready = ev(0xe0), .fetched = .{ ev(0xe1), ev(0xe2) }, .attended = .{ ev(0xe3), ev(0xe4) } };
    g.prefetch = &pf2;
    try std.testing.expectError(error.StageBuffersAliased, g.pieced(.{ .positions = &pos, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } }));
}

test "plane bounds: a layer subset that does not start at 0 maps to planes 0..n-1; too few planes fail closed" {
    var g = geometry();
    g.layers = 1;
    g.full_index_layers = 0;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.layer_store[0] = glmLayer(3, false, false); // the one-layer smoke on layer 3
    fx.model.layers = fx.layer_store[0..1];
    const f = fx.fwd(0);
    try f.whole(.{ .positions = &.{ 100, 101 }, .tokens = &.{ 1, 2 } });
    for (Rec.launches.items) |l| if (l.name == id("tf_glm53_kv_write")) try std.testing.expectEqual(fx.st.latentAt(0), l.args[5]);
    // Two loaded layers against one plane, or a full indexer without an index plane: refused before any launch.
    Rec.reset();
    fx.layer_store[1] = glmLayer(4, false, false);
    fx.model.layers = fx.layer_store[0..2];
    try std.testing.expectError(error.LayerOutOfPlanes, f.whole(.{ .positions = &.{ 100, 101 }, .tokens = &.{ 1, 2 } }));
    fx.layer_store[0] = glmLayer(6, false, true);
    fx.model.layers = fx.layer_store[0..1];
    try std.testing.expectError(error.IndexPlaneOutOfRange, f.whole(.{ .positions = &.{ 100, 101 }, .tokens = &.{ 1, 2 } }));
    try std.testing.expectEqual(@as(usize, 0), Rec.launches.items.len);
}

test "the bit-neutral production defaults: 512-row pieces, 64 score rows, 128 expert rows, one stage pair, clusters" {
    const d: state.Geometry = .{};
    try std.testing.expectEqual(@as(usize, 512), d.piece_rows);
    try std.testing.expectEqual(@as(usize, 64), d.score_rows);
    try std.testing.expectEqual(@as(usize, 128), d.expert_rows);
    try std.testing.expect(!d.prefetch and d.lane_clusters);
    // A cluster-carved workspace refuses a lane without clusters (it would need split-K partials).
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    fx.lane.clusters = false;
    try std.testing.expectError(error.LanePartNeedsClusters, fx.fwd(0).whole(.{ .positions = &.{5}, .tokens = &.{1} }));
}

test "greedy draw never returns WP10's absent sentinel" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(0), .download = .{ .download_fn = FakeHead.download }, .pos = 10, .frontier = 10, .auto_graphs = false };
    defer e.deinit();
    FakeHead.rows[0][0] = .{ .v = -std.math.inf(f32), .id = 0xFFFFFFFF };
    FakeHead.rows[1][0] = .{ .v = -3, .id = 77440 + 2 };
    var out: [1]u32 = undefined;
    try e.verify(&.{7}, &out);
    try std.testing.expectEqual(@as(u32, 77440 + 2), out[0]);
    FakeHead.rows[1][0] = .{ .v = -std.math.inf(f32), .id = 0xFFFFFFFF };
    try std.testing.expectError(error.NoFiniteLogit, e.verify(&.{7}, &out));
}

test "bring-up: TP2 + DCP2 over layers 0-7 (dense 0-2, MoE 3-7, full indexers 0, 1, 2, 6), decode and prefill" {
    var g = geometry();
    g.layers = 8;
    g.full_index_layers = 4;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    for (0..8) |i| fx.layer_store[i] = glmLayer(i, i < 3, i < 3 or i == 6);
    fx.model.layers = fx.layer_store[0..8];
    fx.model.config.layers = 78; // the full model's config; the load selected layers 0-7 (weights.Select.bringUp)
    var store: [8]usize = undefined;
    const sel = @import("weights").Select.bringUp(&store, 8);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, sel.layers.?);
    const f = fx.fwd(1);
    // Decode across the dense limit: every layer writes KV; the four full layers select, the others reuse.
    try f.whole(.{ .positions = &.{ 2047, 2048, 2049 }, .tokens = &.{ 1, 2, 3 } });
    var kv: usize = 0;
    var keys: usize = 0;
    var merges: usize = 0;
    for (Rec.launches.items) |l| {
        if (l.name == id("tf_glm53_kv_write")) kv += 1;
        if (l.name == id("tf_glm53_index_key")) keys += 1;
        if (l.name == id("tf_glm53_merge")) merges += 1;
    }
    try std.testing.expectEqual(@as(usize, 8), kv);
    try std.testing.expectEqual(@as(usize, 4), keys);
    try std.testing.expectEqual(@as(usize, 4), merges);
    var ls: [8]forward.Layer = undefined;
    try expectPlan(try f.stepPlan(.decode, &.{ 2047, 2048, 2049 }, 3, &ls));
    try checkContracts(&fx, .decode, 2049);
    // A 5k prompt: 2 chunks, stage fetches on chunk 2, every exchange as planned for the last chunk.
    Rec.reset();
    var pos: [5000]usize = undefined;
    var ids: [5000]u32 = undefined;
    for (&pos, &ids, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    try f.pieced(.{ .positions = &pos, .tokens = &ids });
    try std.testing.expectEqual(@as(usize, 2), Rec.marks.items.len);
    var fetched: usize = 0;
    for (Rec.gathers.items) |gt| if (gt.phase == .stage_fetch) {
        fetched += 1;
    };
    try std.testing.expectEqual(@as(usize, 2 * 8), fetched); // chunk 2 only: latent + rope for each of 8 layers
}

test "ADD_NORM_FUSE from the environment: decode layers before the last also write the next input norm" {
    const on = forward.addNormFuseSwitch();
    var g = geometry();
    g.layers = 8;
    g.full_index_layers = 4;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    for (0..8) |i| fx.layer_store[i] = glmLayer(i, i < 3, i < 3 or i == 6);
    fx.model.layers = fx.layer_store[0..8];
    fx.k.moe_residual_rmsnorm_6144 = func("tf_glm53_moe_residual_rmsnorm_6144");
    var f = fx.fwd(1);
    f.add_norm_fuse = on;
    try f.whole(.{ .positions = &.{ 100, 101, 102, 103 }, .tokens = &.{ 1, 2, 3, 4 } });
    var moe_fused: usize = 0;
    var moe_add: usize = 0;
    var dense_add: usize = 0;
    var post: usize = 0;
    var in_norm: usize = 0;
    var dense_fused: usize = 0;
    for (Rec.launches.items) |l| {
        if (l.name == id("tf_glm53_moe_residual_rmsnorm_6144")) {
            try std.testing.expectEqual(fx.layer_store[4 + moe_fused].in_norm, l.args[2]);
            try std.testing.expectEqual(fx.st.ws.xn, l.args[3]);
            moe_fused += 1;
        }
        if (l.name == id("tf_glm53_moe_residual_add")) {
            moe_add += 1;
        }
        if (l.name == id("tf_glm_residual_add")) {
            dense_add += 1;
        }
        if (l.name == id("tf_glm53_residual_rmsnorm_6144")) {
            post += 1;
            for (fx.layer_store[1..4]) |next| {
                dense_fused += @intFromBool(l.args[2] == next.in_norm);
            }
        }
        if (l.name == id("tf_glm_rmsnorm_6144")) {
            in_norm += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, if (on) 4 else 0), moe_fused);
    try std.testing.expectEqual(@as(usize, if (on) 1 else 5), moe_add);
    try std.testing.expectEqual(@as(usize, if (on) 0 else 3), dense_add);
    try std.testing.expectEqual(@as(usize, if (on) 11 else 8), post);
    try std.testing.expectEqual(@as(usize, if (on) 3 else 0), dense_fused);
    try std.testing.expectEqual(@as(usize, if (on) 2 else 9), in_norm);
    for (Rec.order.items, 0..) |st, i| if (st.kind == .launch and st.what == id("tf_glm53_moe_residual_rmsnorm_6144")) {
        try std.testing.expect(i > 0 and Rec.order.items[i - 1].kind == .collective and Rec.order.items[i - 1].what == @backingInt(forward.Phase.moe_partial));
    };
    Rec.reset();
    try f.pieced(.{ .positions = &.{ 0, 1, 2, 3 }, .tokens = &.{ 1, 2, 3, 4 } });
    for (Rec.launches.items) |l| try std.testing.expect(l.name != id("tf_glm53_moe_residual_rmsnorm_6144"));
}

test "QPE_ABSORB_FUSE from the environment swaps q_pe pack + absorb v3 for one launch" {
    const on = forward.qpeAbsorbFuseSwitch();
    var g = geometry();
    g.decode_mla = true;
    g.decode_mla_v2 = true;
    g.decode_mla_v3 = true;
    var fx: Fixture = undefined;
    try fx.init(g, 16384);
    defer fx.deinit();
    fx.k.mla.absorb_qpe_v3_fn = func("tf_glm53_absorb_qpe_decode_v3");
    var f = fx.fwd(0);
    f.qpe_absorb_fuse = on;
    try f.whole(.{ .positions = &.{ 100, 101, 102, 103 }, .tokens = &.{ 7, 8, 9, 10 } });
    var pack: usize = 0;
    var v3: usize = 0;
    var fused: usize = 0;
    for (Rec.launches.items) |l| {
        pack += @intFromBool(l.name == id("tf_glm53_qpe_pack"));
        v3 += @intFromBool(l.name == id("tf_glm53_absorb_decode_v3"));
        if (l.name == id("tf_glm53_absorb_qpe_decode_v3")) {
            fused += 1;
            try std.testing.expectEqual([3]u32{ 8, 32, 1 }, l.grid);
            try std.testing.expectEqual(@as(u32, 128), l.block[0]);
            try std.testing.expectEqual(@as(u64, 4), l.args[4]);
        }
    }
    try std.testing.expectEqual(@as(usize, if (on) 0 else 2), pack);
    try std.testing.expectEqual(@as(usize, if (on) 0 else 2), v3);
    try std.testing.expectEqual(@as(usize, if (on) 2 else 0), fused);
}

fn eventRecordIndex(handle: usize, start: usize) ?usize {
    for (Rec.order.items[start..], start..) |item, i| if (item.kind == .record and item.what == handle) return i;
    return null;
}

/// Some recorded begin/end pair of `stage` at `layer` (stages may hold several pairs a layer) encloses `kernel`.
fn expectProfilePair(prof: *const forward.Profile, layer: usize, stage: forward.Stage, kernel: usize) !void {
    const base = forward.stagePairBase(layer, stage);
    for (0..forward.stagePairs(stage)) |slot| {
        const pair = base + slot;
        const begin_h = prof.events[pair * 2].handle orelse continue;
        const end_h = prof.events[pair * 2 + 1].handle orelse continue;
        const lo = eventRecordIndex(@intFromPtr(begin_h), 0) orelse continue;
        const hi = eventRecordIndex(@intFromPtr(end_h), lo + 1) orelse continue;
        for (Rec.order.items[lo + 1 .. hi]) |item| if (item.kind == .launch and item.what == kernel) return;
    }
    return error.ProfilePairMissing;
}

test "decode profile records stage event pairs around expected launches and leaves null launches unchanged" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 4096);
    defer fx.deinit();
    var prof = try forward.Profile.init(gpa, &driver, 3);
    defer prof.deinit();
    var f = fx.fwd(0);
    f.prof = &prof;
    try f.whole(.{ .positions = &.{10}, .tokens = &.{7} });

    try expectProfilePair(&prof, 0, .attn_pre, id("tf_glm_rmsnorm_6144"));
    try expectProfilePair(&prof, 0, .attn_post, id("qmmf_f32"));
    try expectProfilePair(&prof, 1, .attn_pre, id("tf_glm_rmsnorm_6144"));
    try expectProfilePair(&prof, 1, .attn_post, id("qmmf_f32"));
    try expectProfilePair(&prof, 1, .route, id("tf_glm_router_tile"));
    try expectProfilePair(&prof, 1, .routed, id("exl3_group"));
    try expectProfilePair(&prof, 1, .shared, id("qmmf_bf16"));
    try expectProfilePair(&prof, 1, .lowrank, id("tf_glm53_lowrank_partial"));
    try expectProfilePair(&prof, 0, .dense_mlp, id("tf_glm53_swiglu_plain"));
    try expectProfilePair(&prof, 2, .head, id("tf_glm53_top1_f32"));
    try prof.collect();
    for (prof.counts, prof.totals_ms) |count, total| if (count > 0) try std.testing.expectEqual(@as(f64, @floatFromInt(count)) * 1.25, total);

    const profiled_launches = try gpa.dupe(Launch, Rec.launches.items);
    defer gpa.free(profiled_launches);
    Rec.reset();
    f.prof = null;
    try f.whole(.{ .positions = &.{10}, .tokens = &.{7} });
    try std.testing.expectEqualSlices(Launch, profiled_launches, Rec.launches.items);
    for (Rec.order.items) |item| try std.testing.expect(item.kind != .record);
}

test "prefill profile records stage event pairs only when enabled" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 4096);
    defer fx.deinit();
    var prof = try forward.Profile.init(gpa, &driver, 3);
    defer prof.deinit();
    var f = fx.fwd(0);
    f.prof = &prof;
    f.prof_prefill = true;
    try f.pieced(.{ .positions = &.{10}, .tokens = &.{7} });

    try expectProfilePair(&prof, 0, .attn_pre, id("tf_glm_rmsnorm_6144"));
    try expectProfilePair(&prof, 1, .routed, id("exl3_group"));
    try std.testing.expect(prof.counts[@intFromEnum(forward.Stage.attn_pre)] > 0);
    try std.testing.expect(prof.counts[@intFromEnum(forward.Stage.routed)] > 0);

    Rec.reset();
    prof.reset();
    f.prof_prefill = false;
    try f.pieced(.{ .positions = &.{10}, .tokens = &.{7} });
    for (Rec.order.items) |item| try std.testing.expect(item.kind != .record);
}

test "attn_comm profile measures its wait span across intervening stream work" {
    Rec.reset();
    defer Rec.free(); // the recorder lists are globals: free them with this test's allocator
    Rec.use_event_clock = true;
    var prof = try forward.Profile.init(gpa, &driver, 1);
    defer prof.deinit();
    const main: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(0x10) };
    const other: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(0x20) };

    try prof.begin(0, .attn_comm, main);
    Rec.event_clock_ms = 30;
    try prof.begin(0, .shared, other);
    Rec.event_clock_ms = 40;
    try prof.end(0, .shared, other);
    Rec.event_clock_ms = 3;
    try prof.end(0, .attn_comm, main);
    try prof.collect();

    try std.testing.expectEqual(@as(f64, 3), prof.totals_ms[@intFromEnum(forward.Stage.attn_comm)]);
    try std.testing.expect(prof.totals_ms[@intFromEnum(forward.Stage.attn_comm)] < 40);
}

fn checkGroupedProfilePieces(piece_count: usize) !void {
    Rec.reset();
    defer Rec.free(); // the recorder lists are globals: free them with this test's allocator
    Rec.use_event_clock = true;
    var prof = try forward.Profile.init(gpa, &driver, 1);
    defer prof.deinit();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for (0..piece_count) |piece| {
        const start = @as(f64, @floatFromInt(piece)) * 10;
        Rec.event_clock_ms = start;
        try prof.begin(0, .attn_pre, stream);
        // Indexer scoring has 16 slots; overflow forces collection while attn_pre remains open.
        for (0..18) |_| {
            try prof.begin(0, .attn_pre_indexer_scoring, stream);
            Rec.event_clock_ms += 1;
            try prof.end(0, .attn_pre_indexer_scoring, stream);
        }
        Rec.event_clock_ms = start + 5;
        try prof.end(0, .attn_pre, stream);
    }
    try prof.collect();
    const parent_ms = prof.totals_ms[@intFromEnum(forward.Stage.attn_pre)];
    try std.testing.expectEqual(@as(f64, @floatFromInt(piece_count)) * 5, parent_ms);
    try std.testing.expect(parent_ms <= @as(f64, @floatFromInt(piece_count)) * 10);
    try std.testing.expectEqual(@as(u64, @intCast(piece_count)), prof.counts[@intFromEnum(forward.Stage.attn_pre)]);
    try std.testing.expectEqual(@as(u64, @intCast(piece_count * 18)), prof.counts[@intFromEnum(forward.Stage.attn_pre_indexer_scoring)]);
}

test "grouped profile ordering survives substage overflow across two pieces" {
    try checkGroupedProfilePieces(2);
}

test "grouped profile ordering survives substage overflow across four pieces" {
    try checkGroupedProfilePieces(4);
}

test "prefill profiling setup refuses TBO" {
    try std.testing.expectError(error.ProfilePrefillWithTbo, forward.validateProfilePrefillTbo(true, true));
}

test "prefill profiling setup allows TBO when profiling is off" {
    try forward.validateProfilePrefillTbo(false, true);
    try forward.validateProfilePrefillTbo(true, false);
}

test "low-rank decode selector changes decode launches while prefill stays on the legacy kernels" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 4096);
    defer fx.deinit();
    for ([_]bool{ true, false }) |enabled| {
        var f = fx.fwd(0);
        f.lowrank_decode = enabled;
        try f.whole(.{ .positions = &.{10}, .tokens = &.{7} });
        var seen: [3]usize = undefined;
        var n: usize = 0;
        for (Rec.launches.items) |l| {
            if (l.name == id("tf_glm53_lowrank_partial") or l.name == id("tf_glm53_lowrank_decode") or l.name == id("tf_glm53_lowrank_reduce") or l.name == id("tf_glm53_lowrank_combine_bf16") or l.name == id("tf_glm53_lowrank_combine_bf16_decode")) {
                if (n < seen.len) seen[n] = l.name;
                n += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, 3), n);
        if (enabled) {
            try std.testing.expectEqualSlices(usize, &.{ id("tf_glm53_lowrank_decode"), id("tf_glm53_lowrank_reduce"), id("tf_glm53_lowrank_combine_bf16_decode") }, seen[0..n]);
        } else {
            try std.testing.expectEqualSlices(usize, &.{ id("tf_glm53_lowrank_partial"), id("tf_glm53_lowrank_reduce"), id("tf_glm53_lowrank_combine_bf16") }, seen[0..n]);
        }
        Rec.reset();
    }
    var f = fx.fwd(0);
    f.lowrank_decode = true;
    Rec.reset();
    var positions: [64]usize = undefined;
    var tokens: [64]u32 = undefined;
    for (&positions, &tokens, 0..) |*p, *t, i| {
        p.* = i;
        t.* = @intCast(i);
    }
    try f.pieced(.{ .positions = &positions, .tokens = &tokens });
    var saw_legacy_partial = false;
    var saw_legacy_reduce = false;
    var saw_legacy_combine = false;
    var partial_launch: ?Launch = null;
    var reduce_launch: ?Launch = null;
    var combine_launch: ?Launch = null;
    for (Rec.launches.items) |l| {
        try std.testing.expect(l.name != id("tf_glm53_lowrank_decode"));
        try std.testing.expect(l.name != id("tf_glm53_lowrank_combine_bf16_decode"));
        if (l.name == id("tf_glm53_lowrank_partial")) {
            saw_legacy_partial = true;
            partial_launch = l;
        }
        if (l.name == id("tf_glm53_lowrank_reduce")) {
            saw_legacy_reduce = true;
            reduce_launch = l;
        }
        if (l.name == id("tf_glm53_lowrank_combine_bf16")) {
            saw_legacy_combine = true;
            combine_launch = l;
        }
    }
    try std.testing.expect(saw_legacy_partial and saw_legacy_reduce and saw_legacy_combine);
    // A 64-row prefill piece stays on the established 8-split, 256-thread path.
    // The fixture's TP-local rank is 32, so the legacy launch grids are 64, 8, and 1536.
    const partial = partial_launch.?;
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, partial.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, partial.block);
    try std.testing.expectEqual(@as(u64, 64), partial.args[3]);
    try std.testing.expectEqual(@as(u64, 8), partial.args[8]);
    const reduce = reduce_launch.?;
    try std.testing.expectEqual([3]u32{ 8, 1, 1 }, reduce.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, reduce.block);
    try std.testing.expectEqual(@as(u64, 2048), reduce.args[2]);
    try std.testing.expectEqual(@as(u64, 8), reduce.args[3]);
    const combine = combine_launch.?;
    try std.testing.expectEqual([3]u32{ 1536, 1, 1 }, combine.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, combine.block);
    try std.testing.expectEqual(@as(u64, 64), combine.args[5]);
}

test "continuity with an unaligned frontier (4100): per-position row inputs equal a fresh run; rows 4096-4099 come from the fetch" {
    var fx: Fixture = undefined;
    try fx.init(geometry(), 16384);
    defer fx.deinit();
    for (0..2) |r| for (0..8) |j| {
        FakeHead.rows[r][j] = .{ .v = 1, .id = @intCast(j) };
    };
    var transcript: [4100 + 5 + 10]u32 = undefined;
    for (&transcript, 0..) |*t, i| t.* = @intCast((i * 131) % 154880);
    var fresh_e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(1), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer fresh_e.deinit();
    _ = try fresh_e.prefill(&transcript);
    var fresh = try Snapshot.take(&fx);
    defer fresh.deinit();
    var e: engine.Engine = .{ .gpa = gpa, .f = fx.fwd(1), .download = .{ .download_fn = FakeHead.download }, .auto_graphs = false };
    defer e.deinit();
    _ = try e.prefill(transcript[0..4100]);
    var out: [5]u32 = undefined;
    try e.verify(transcript[4100..4105], &out);
    try e.keep(5);
    Rec.reset();
    _ = try e.nextTurn(transcript[4105..]);
    // The resume window is [4100, 4115): rank 1 sends its 2050 own rows below 4100 and receives rank 0's 2050,
    // which include the peer rows of positions 4096-4099 that the fresh run's chunk [4096, 4115) wrote in-chunk.
    var sent: usize = 0;
    var recv: usize = 0;
    for (Rec.gathers.items) |g| if (g.phase == .stage_fetch and g.bytes > 512 * 1024) {
        sent = g.bytes;
        recv = g.recv_bytes;
    };
    try std.testing.expectEqual(@as(usize, 2050 * 1024), sent);
    try std.testing.expectEqual(@as(usize, 2050 * 1024), recv);
    var resumed = try Snapshot.take(&fx);
    defer resumed.deinit();
    try std.testing.expectEqual(@as(usize, 1), resumed.marks.len);
    // Row inputs by absolute position: the resumed rows 4100..4114 equal the fresh run's last 15 rows.
    const sizes = [_]usize{ 8, 4, 32 * 4, 32 * 4, 4, 4 }; // bytes a row: positions, dense_len, cos, sin, ids, owners
    for (fresh.rows, resumed.rows, sizes) |a, b, row| try std.testing.expectEqualSlices(u8, a[a.len - 15 * row ..], b);
    try std.testing.expectEqual(@as(usize, 4115), e.frontier);
}
