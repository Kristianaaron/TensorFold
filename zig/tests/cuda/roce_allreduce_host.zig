// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host protocol model for the optional GLM-5.3 two-rank RoCE all-reduce.
const std = @import("std");
const cuda = @import("cuda");
const roce = cuda.roce;
const tp_group = cuda.tp_group;

test "GLM hidden-width RoCE byte plan covers decode rows and all layer calls" {
    var sum: usize = 0;
    for (1..9) |rows| {
        const bytes = rows * 6144 * @sizeOf(f32);
        try std.testing.expectEqual(rows * 24 * 1024, bytes);
        try std.testing.expect(bytes <= 512 * 1024);
        sum += bytes * 2 * 78;
    }
    try std.testing.expectEqual(@as(usize, 2 * 78 * 24 * 1024 * 36), sum);
}

test "sequence failures always use a nonzero status at sequence zero" {
    try std.testing.expectEqual(@as(u32, 1), roce.errorFlag(0));
    try std.testing.expectEqual(@as(u32, 1), roce.errorFlag(23));
    try std.testing.expect(roce.completionMatches(0, 4096, 0, 4096));
    try std.testing.expect(!roce.completionMatches(0, 4096, 0, 2048));
    try std.testing.expect(!roce.completionMatches(11, 4096, 10, 4096));
}

test "bf16 reference rounds exact half-ulp ties both directions" {
    try std.testing.expectEqual(@as(u16, 0x3f80), roce.f32ToBf16(1.00390625));
    try std.testing.expectEqual(@as(u16, 0x3f82), roce.f32ToBf16(1.01171875));
    try std.testing.expectEqual(@as(u16, 0x0000), roce.f32ToBf16(0x1p-134));
}

test "reducer and token gate host references match the poison contract" {
    try std.testing.expectEqual(@as(u32, 0xffff_ffff), @as(u32, @bitCast(roce.reduceF32(1.0, 2.0, true))));
    try std.testing.expectEqual(@as(u16, 0xffff), roce.reduceBf16(0x3f80, 0x3f80, true));
    try std.testing.expectEqual(@as(u16, 0x4000), roce.reduceBf16(0x3f80, 0x3f80, false));
    try std.testing.expectEqual(roce.TokenStatus{ .token = 0xffff_ffff, .status = 1 }, roce.gateToken(7, true));
    try std.testing.expectEqual(roce.TokenStatus{ .token = 7, .status = 0 }, roce.gateToken(7, false));
}

test "RoCE timeout keeps the legacy gather window and bounds the GLM opt-in" {
    try std.testing.expectEqual(@as(u32, 120_000), try roce.timeoutMs(null, false));
    try std.testing.expectEqual(@as(u32, 2_000), try roce.timeoutMs(null, true));
    try std.testing.expectEqual(@as(u32, 250), try roce.timeoutMs("250", false));
    try std.testing.expectError(error.InvalidRoceTimeout, roce.timeoutMs("0", false));
    try std.testing.expectError(error.InvalidRoceTimeout, roce.timeoutMs("bad", true));
    try std.testing.expectEqual(@as(u32, 0xffff_ffff), roce.poisonBits());
    var poison_state = [_]u32{ 1, 1, 9 };
    roce.resetPoisonState(&poison_state);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, &poison_state);
}

test "proxy sequence model advances consecutive calls and alternates slots across wrap" {
    var last = [_]u32{ 0, 0 };
    var previous: u32 = 0;
    for (1..130) |n| {
        const seq: u32 = @intCast(n);
        const state = roce.sequenceState(previous, seq).?;
        try std.testing.expectEqual(@as(u32, 1), state.pending);
        try std.testing.expectEqual(seq, state.next);
        const slot = seq & 1;
        try std.testing.expect(last[slot] == 0 or last[slot] + 2 == seq);
        last[slot] = seq;
        previous = seq;
    }
    try std.testing.expectEqual(@as(u32, 128), last[0]);
    try std.testing.expectEqual(@as(u32, 129), last[1]);
    try std.testing.expectEqual(@as(u32, 0), roce.sequenceState(0xffff_ffff, 0).?.next);
}

test "RoCE selector sends only agreed fp32 and bf16 sums within the cap" {
    const cap = 512 * 1024;
    try std.testing.expect(!roce.Gather.dispatchRoce(false, 2, 0, true, 6144, cap));
    try std.testing.expect(!roce.Gather.dispatchRoce(true, 1, 0, true, 6144, cap));
    try std.testing.expect(!roce.Gather.dispatchRoce(true, 2, 2, true, 6144, cap));
    try std.testing.expect(!roce.Gather.dispatchRoce(true, 2, 1, false, 6144, cap));
    try std.testing.expect(!roce.Gather.dispatchRoce(true, 2, 0, true, cap, cap));
    for (1..9) |rows| {
        try std.testing.expect(roce.Gather.dispatchRoce(true, 2, 0, true, rows * 6144, cap));
        try std.testing.expect(roce.Gather.dispatchRoce(true, 2, 1, true, rows * 6144, cap));
    }
}

test "the collective selector defaults to NCCL" {
    try std.testing.expectEqual(tp_group.GlmCollective.nccl, try tp_group.glmCollective(null));
    try std.testing.expectEqual(tp_group.GlmCollective.nccl, try tp_group.glmCollective("nccl"));
    try std.testing.expectEqual(tp_group.GlmCollective.roce, try tp_group.glmCollective("roce"));
    try std.testing.expectError(error.InvalidCollective, tp_group.glmCollective("RoCE"));
    try std.testing.expect(tp_group.Group.agreesGlmCollective(true, true));
    try std.testing.expect(!tp_group.Group.agreesGlmCollective(false, true));
    try std.testing.expect(tp_group.agreesRoceSettings(.{ .glm = true, .gather = true }, .{ .glm = true, .gather = true }));
    try std.testing.expect(!tp_group.agreesRoceSettings(.{ .glm = true, .gather = true }, .{ .glm = true, .gather = false }));
    try std.testing.expect(!tp_group.agreesRoceSettings(.{ .glm = true, .gather = true, .aligned = false }, .{ .glm = true, .gather = true }));
    try std.testing.expect(!tp_group.roceSettingsAllowed(.{ .glm = true, .gather = false }));
    try std.testing.expect(!tp_group.roceSettingsAllowed(.{ .glm = false, .gather = true, .aligned = false }));
    try std.testing.expect(tp_group.roceSettingsAllowed(.{ .glm = false, .gather = false }));
}

test {
    std.testing.refAllDecls(roce);
    std.testing.refAllDecls(roce.Gather);
    std.testing.refAllDecls(tp_group);
    std.testing.refAllDecls(tp_group.Group);
}
