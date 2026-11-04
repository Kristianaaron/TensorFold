// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host ABI bindings for the dispatchable GLM-5.3 MLA attention tasks.
//! Buffer ownership and argument order follow forward.zig:740-820 and mla.zig:346-371.
const std = @import("std");

pub const Kind = enum { input_norm, qpe_pack, attend };

pub const Plan = struct {
    grid: [3]u32,
    block: u32,
    dynamic_smem: u32 = 0,

    pub fn blocks(p: Plan) !u32 {
        return std.math.mul(u32, try std.math.mul(u32, p.grid[0], p.grid[1]), p.grid[2]);
    }
};

/// Pointer values are CUDA device addresses, matching cuda.Args and the wrapper launch ABI.
pub const QpePackParams = extern struct {
    q: u64,
    cos: u64,
    sin: u64,
    qpe: u64,
    rows: i32,
    local_heads: i32,
};

pub const InputNormParams = extern struct {
    x: u64,
    x_stride: i64,
    weight: u64,
    out: u64,
    out_stride: i64,
    eps: f32,
};

/// Exact argument layout of tf_glm53_attend and tf_glm53_attend_task.
pub const AttendParams = extern struct {
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

/// This is the attention portion of a forward window after row offsets and DCP
/// pointer selection have been applied, just before the corresponding wrapper
/// launch in forward.zig:791-810.
pub const Window = struct {
    norm_x: u64,
    norm_weight: u64,
    norm_out: u64,
    norm_x_stride: i64,
    norm_out_stride: i64,
    norm_eps: f32,
    q: u64,
    cos: u64,
    sin: u64,
    qpe: u64,
    qa: u64,
    latent: u64,
    rope: u64,
    stage_latent: u64 = 0,
    stage_rope: u64 = 0,
    positions: u64,
    global_tokens: u64,
    local_slots: u64,
    global_counts: u64,
    local_counts: u64,
    dense_len: u64,
    partial: u64,
    rows: u32,
    local_heads: u32,
    rank: u32,
    world: u32,
    prefill: bool,
};

pub fn plan(kind: Kind, rows: u32, local_heads: u32) !Plan {
    if (rows == 0 or local_heads == 0) return error.EmptyWindow;
    return switch (kind) {
        .input_norm => .{ .grid = .{ rows, 1, 1 }, .block = 256 },
        .qpe_pack => .{ .grid = .{ rows, local_heads, 1 }, .block = 32 },
        .attend => .{ .grid = .{ rows, local_heads, 1 }, .block = 256 },
    };
}

pub fn bindInputNorm(w: Window) InputNormParams {
    return .{
        .x = w.norm_x,
        .x_stride = w.norm_x_stride,
        .weight = w.norm_weight,
        .out = w.norm_out,
        .out_stride = w.norm_out_stride,
        .eps = w.norm_eps,
    };
}

pub fn bindQpePack(w: Window) QpePackParams {
    return .{
        .q = w.q,
        .cos = w.cos,
        .sin = w.sin,
        .qpe = w.qpe,
        .rows = @intCast(w.rows),
        .local_heads = @intCast(w.local_heads),
    };
}

pub fn bindAttend(w: Window) AttendParams {
    return .{
        .qa = w.qa,
        .qpe = w.qpe,
        .latent = w.latent,
        .rope = w.rope,
        .stage_latent = w.stage_latent,
        .stage_rope = w.stage_rope,
        .positions = w.positions,
        .global_tokens = w.global_tokens,
        .local_slots = w.local_slots,
        .global_counts = w.global_counts,
        .local_counts = w.local_counts,
        .dense_len = w.dense_len,
        .partial = w.partial,
        .rows = @intCast(w.rows),
        .local_heads = @intCast(w.local_heads),
        .rank = @intCast(w.rank),
        .world = @intCast(w.world),
        .prefill = @intFromBool(w.prefill),
    };
}

test "attention binding declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(InputNormParams);
    std.testing.refAllDecls(QpePackParams);
    std.testing.refAllDecls(AttendParams);
    std.testing.refAllDecls(Window);
    std.testing.refAllDecls(Kind);
}
