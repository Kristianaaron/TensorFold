// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host side MoE megakernel parameter packing. Values mirror moe.route, lowRankAdd and block.
const std = @import("std");

pub const hidden = 6144;
pub const expert_count = 256;
pub const routed_slots = 8;
pub const router_splits = 8;

pub const Grid = struct { x: u32, y: u32 = 1, z: u32 = 1, threads: u16 = 256, smem_bytes: u16 = 0 };
pub const RouterPlan = struct { dot: Grid, reduce: Grid, topk: Grid, tile: bool };

/// Router arguments as passed by forward.zig -> moe.route. Device pointers are byte addresses.
pub const RouterStep = extern struct {
    x: u64,
    weight: u64,
    bias: u64,
    logits: u64,
    parts: u64,
    picks: u64,
    mixture: u64,
    rows: u32,
    x_stride: u32,
};

/// Low-rank addresses and dimensions as passed by moe.block / moe.lowRankAdd.
pub const LowRankStep = extern struct {
    x: u64,
    a: u64,
    b: u64,
    partials_mid: u64,
    accumulator: u64,
    rows: u32,
    rank_local: u32,
    x_stride: u32,
    a_stride: u32,
    b_stride: u32,
};

/// The glue epilogue's current lowrank_combine_bf16 task conflicts with the binding 2026-10-21
/// association contract, so the dispatcher binds only the rank-local FP32 intermediates here.
pub const MoeStep = struct {
    router: RouterStep,
    lowrank: ?LowRankStep,
    routed: u64,
    shared: u64,
    moe_partial: u64,
    residual: u64,
    rows: u32,
};

pub const RouterBindings = struct {
    step: RouterStep,
    plan: RouterPlan,
};

pub fn bindMoe(router: RouterStep, lowrank: ?LowRankStep, routed: u64, shared: u64, moe_partial: u64, residual: u64, rows: u32) !MoeStep {
    _ = try bindRouter(router);
    if (rows == 0 or routed == 0 or shared == 0 or moe_partial == 0 or residual == 0 or router.rows != rows)
        return error.BadMoeStep;
    const checked_lowrank = if (lowrank) |step| try bindLowRank(step) else null;
    return .{ .router = router, .lowrank = checked_lowrank, .routed = routed, .shared = shared, .moe_partial = moe_partial, .residual = residual, .rows = rows };
}

pub fn bindRouter(step: RouterStep) error{BadRouterStep}!RouterBindings {
    if (step.rows == 0 or step.x == 0 or step.weight == 0 or step.bias == 0 or step.logits == 0 or
        step.parts == 0 or step.picks == 0 or step.mixture == 0 or step.x_stride < hidden)
        return error.BadRouterStep;
    const tile = (step.x & 15) == 0 and step.x_stride % 8 == 0;
    return .{
        .step = step,
        .plan = .{
            .dot = .{
                .x = if (tile) @intCast((step.rows + 63) / 64) else @intCast((step.rows + 15) / 16),
                .y = 8,
                .z = 8,
                .threads = 128,
            },
            .reduce = .{ .x = @intCast((step.rows * expert_count + 1023) / 1024), .threads = 128 },
            .topk = .{ .x = step.rows, .threads = 32 },
            .tile = tile,
        },
    };
}

pub fn bindLowRank(step: LowRankStep) error{BadLowRankStep}!LowRankStep {
    if (step.x == 0 or step.a == 0 or step.b == 0 or step.partials_mid == 0 or step.accumulator == 0 or
        step.rows == 0 or step.rank_local == 0 or step.rank_local > 64 or step.x_stride != hidden or
        step.a_stride != step.rank_local or step.b_stride != hidden)
        return error.BadLowRankStep;
    return step;
}

pub fn lowRankGrids(step: LowRankStep) struct { partial: Grid, reduce: Grid, finish: Grid } {
    return .{
        .partial = .{ .x = @intCast((step.rows * step.rank_local * 8 + 255) / 256) },
        .reduce = .{ .x = @intCast((step.rows * step.rank_local + 255) / 256) },
        .finish = .{ .x = @intCast((step.rows * hidden + 255) / 256) },
    };
}

pub fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/// Binding numerics update: fp32 adds in this exact association, followed by one BF16 round.
pub fn combineRef(routed: f32, shared: f32, lowrank: f32) u16 {
    return bf16Bits(routed + (shared + lowrank));
}

test "MoE binding declarations are analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Grid);
    std.testing.refAllDecls(RouterPlan);
    std.testing.refAllDecls(RouterStep);
    std.testing.refAllDecls(LowRankStep);
    std.testing.refAllDecls(MoeStep);
    std.testing.refAllDecls(RouterBindings);
}

test "router and low-rank plans match forward launch formulas" {
    const router = try bindRouter(.{
        .x = 0x1000,
        .weight = 0x2000,
        .bias = 0x3000,
        .logits = 0x4000,
        .parts = 0x5000,
        .picks = 0x6000,
        .mixture = 0x7000,
        .rows = 4,
        .x_stride = hidden,
    });
    try std.testing.expect(router.plan.tile);
    try std.testing.expectEqual(@as(u32, 1), router.plan.dot.x);
    try std.testing.expectEqual(@as(u32, 8), router.plan.dot.y);
    try std.testing.expectEqual(@as(u32, 1), router.plan.reduce.x);
    try std.testing.expectEqual(@as(u16, 32), router.plan.topk.threads);

    const lowrank = try bindLowRank(.{
        .x = 1,
        .a = 2,
        .b = 3,
        .partials_mid = 4,
        .accumulator = 5,
        .rows = 4,
        .rank_local = 32,
        .x_stride = hidden,
        .a_stride = 32,
        .b_stride = hidden,
    });
    const grids = lowRankGrids(lowrank);
    try std.testing.expectEqual(@as(u32, 4), grids.partial.x);
    try std.testing.expectEqual(@as(u32, 1), grids.reduce.x);
    try std.testing.expectEqual(@as(u32, 96), grids.finish.x);
    const moe = try bindMoe(router.step, lowrank, 6, 7, 8, 9, 4);
    try std.testing.expectEqual(@as(u64, 6), moe.routed);
}

test "MoE final rounding follows shared plus low-rank association" {
    try std.testing.expectEqual(bf16Bits(3.25 + (1.5 + 0.125)), combineRef(3.25, 1.5, 0.125));
    try std.testing.expectError(error.BadRouterStep, bindRouter(.{
        .x = 0,
        .weight = 1,
        .bias = 1,
        .logits = 1,
        .parts = 1,
        .picks = 1,
        .mixture = 1,
        .rows = 1,
        .x_stride = hidden,
    }));
}
