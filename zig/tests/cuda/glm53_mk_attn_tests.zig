// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host checks for GLM-5.3 megakernel attention payloads and wrapper-equivalent grids.
const std = @import("std");
const bind = @import("bind_attn");

fn fixture(prefill: bool) bind.Window {
    return .{
        .norm_x = 0x0800,
        .norm_weight = 0x0900,
        .norm_out = 0x0a00,
        .norm_x_stride = 6144,
        .norm_out_stride = 6144,
        .norm_eps = 1e-5,
        .q = 0x1000,
        .cos = 0x2000,
        .sin = 0x3000,
        .qpe = 0x4000,
        .qa = 0x5000,
        .latent = 0x6000,
        .rope = 0x7000,
        .stage_latent = if (prefill) 0x8000 else 0,
        .stage_rope = if (prefill) 0x9000 else 0,
        .positions = 0xa000,
        .global_tokens = 0xb000,
        .local_slots = 0xc000,
        .global_counts = 0xd000,
        .local_counts = 0xe000,
        .dense_len = 0xf000,
        .partial = 0x11000,
        .rows = 4,
        .local_heads = if (prefill) 32 else 64,
        .rank = 1,
        .world = 2,
        .prefill = prefill,
    };
}

test "dispatchable attention task grids equal wrapper launch grids" {
    const norm_plan = try bind.plan(.input_norm, 4, 64);
    try std.testing.expectEqualSlices(u32, &.{ 4, 1, 1 }, &norm_plan.grid);
    try std.testing.expectEqual(@as(u32, 256), norm_plan.block);
    try std.testing.expectEqual(@as(u32, 4), try norm_plan.blocks());

    const qpe_plan = try bind.plan(.qpe_pack, 4, 32);
    try std.testing.expectEqualSlices(u32, &.{ 4, 32, 1 }, &qpe_plan.grid);
    try std.testing.expectEqual(@as(u32, 32), qpe_plan.block);
    try std.testing.expectEqual(@as(u32, 128), try qpe_plan.blocks());

    const attend_plan = try bind.plan(.attend, 4, 64);
    try std.testing.expectEqualSlices(u32, &.{ 4, 64, 1 }, &attend_plan.grid);
    try std.testing.expectEqual(@as(u32, 256), attend_plan.block);
    try std.testing.expectEqual(@as(u32, 256), try attend_plan.blocks());
    try std.testing.expectError(error.EmptyWindow, bind.plan(.attend, 0, 32));
}

test "input norm payload matches the 6144-wide forward norm call" {
    const p = bind.bindInputNorm(fixture(false));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(bind.InputNormParams));
    try std.testing.expectEqual(@as(u64, 0x0800), p.x);
    try std.testing.expectEqual(@as(i64, 6144), p.x_stride);
    try std.testing.expectEqual(@as(u64, 0x0900), p.weight);
    try std.testing.expectEqual(@as(u64, 0x0a00), p.out);
    try std.testing.expectEqual(@as(i64, 6144), p.out_stride);
    try std.testing.expectEqual(@as(f32, 1e-5), p.eps);
}

test "qpe pack payload preserves the forward wrapper argument order" {
    const p = bind.bindQpePack(fixture(false));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(bind.QpePackParams));
    try std.testing.expectEqual(@as(u64, 0x1000), p.q);
    try std.testing.expectEqual(@as(u64, 0x2000), p.cos);
    try std.testing.expectEqual(@as(u64, 0x3000), p.sin);
    try std.testing.expectEqual(@as(u64, 0x4000), p.qpe);
    try std.testing.expectEqual(@as(i32, 4), p.rows);
    try std.testing.expectEqual(@as(i32, 64), p.local_heads);
}

test "attend payload binds decode and prefill DCP buffers exactly" {
    const decode = bind.bindAttend(fixture(false));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(bind.AttendParams));
    try std.testing.expectEqual(@as(u64, 0x5000), decode.qa);
    try std.testing.expectEqual(@as(u64, 0x4000), decode.qpe);
    try std.testing.expectEqual(@as(u64, 0x6000), decode.latent);
    try std.testing.expectEqual(@as(u64, 0), decode.stage_latent);
    try std.testing.expectEqual(@as(u64, 0), decode.stage_rope);
    try std.testing.expectEqual(@as(u64, 0xa000), decode.positions);
    try std.testing.expectEqual(@as(u64, 0xb000), decode.global_tokens);
    try std.testing.expectEqual(@as(u64, 0xc000), decode.local_slots);
    try std.testing.expectEqual(@as(u64, 0xd000), decode.global_counts);
    try std.testing.expectEqual(@as(u64, 0xe000), decode.local_counts);
    try std.testing.expectEqual(@as(u64, 0xf000), decode.dense_len);
    try std.testing.expectEqual(@as(u64, 0x11000), decode.partial);
    try std.testing.expectEqual(@as(i32, 4), decode.rows);
    try std.testing.expectEqual(@as(i32, 64), decode.local_heads);
    try std.testing.expectEqual(@as(i32, 1), decode.rank);
    try std.testing.expectEqual(@as(i32, 2), decode.world);
    try std.testing.expectEqual(@as(i32, 0), decode.prefill);

    const prefill = bind.bindAttend(fixture(true));
    try std.testing.expectEqual(@as(u64, 0x8000), prefill.stage_latent);
    try std.testing.expectEqual(@as(u64, 0x9000), prefill.stage_rope);
    try std.testing.expectEqual(@as(i32, 32), prefill.local_heads);
    try std.testing.expectEqual(@as(i32, 1), prefill.prefill);
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(bind.Plan);
    std.testing.refAllDecls(bind.InputNormParams);
    std.testing.refAllDecls(bind.QpePackParams);
    std.testing.refAllDecls(bind.AttendParams);
    std.testing.refAllDecls(bind.Window);
    std.testing.refAllDecls(bind.Kind);
}
