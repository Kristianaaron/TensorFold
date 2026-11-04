// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Tensor-parallel spans for glm_moe_dsa CUDA.
const std = @import("std");
const Config = @import("config").Config;

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    heads: Span,
    moe: Span,
    shared: Span,
    dense: Span,
    vocab: Span,

    pub fn of(c: *const Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        return .{
            .rank = rank,
            .world = world,
            .heads = try spanChannels(c.heads, rank, world),
            .moe = try span128Channels(c.moe_inter, rank, world),
            .shared = try span128Channels(c.moe_inter, rank, world),
            .dense = try span128Channels(c.dense_inter, rank, world),
            .vocab = try spanChannels(c.vocab, rank, world),
        };
    }
};

/// Split a channel axis evenly and return global channel coordinates.
pub fn spanChannels(total_channels: usize, rank: usize, world: usize) !Span {
    if (world == 0 or world > 256 or total_channels % world != 0) return error.BadSplit;
    const width = total_channels / world;
    return .{ .lo = rank * width, .hi = (rank + 1) * width };
}

/// Split an axis on 128-channel quantization units, returning channel coordinates.
pub fn span128Channels(total_channels: usize, rank: usize, world: usize) !Span {
    if (world == 0 or total_channels % 128 != 0 or (total_channels / 128) % world != 0) return error.BadSplit;
    return spanChannels(total_channels, rank, world);
}

test "TP2 owns 32 heads and halves of expert, dense and vocabulary axes" {
    const c = Config{};
    const r0 = try Share.of(&c, 0, 2);
    const r1 = try Share.of(&c, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 32 }, r0.heads);
    try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, r1.heads);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 1024 }, r0.moe);
    try std.testing.expectEqual(Span{ .lo = 1024, .hi = 2048 }, r1.shared);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 6144 }, r0.dense);
    try std.testing.expectEqual(Span{ .lo = 77440, .hi = 154880 }, r1.vocab);
    try std.testing.expectError(error.BadRank, Share.of(&c, 2, 2));
}
