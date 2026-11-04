// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host binding for the index task ABI. Buffer addresses and dimensions follow Forward's index call sites.
const std = @import("std");

pub const IndexShare = struct {
    /// For each layer, the full indexer layer supplying its token list, from Config.indexFrom.
    index_from: []const u16,
    /// The per-layer outputs allocated by the forward workspace.
    token_lists: []const TokenListBuffers,
};

pub const TokenListBuffers = struct {
    positions: u64,
    counts: u64,
    local_slots: u64,
    local_counts: u64,
    dense_len: u64,
};

pub const Buffers = struct {
    key_projection: u64,
    key_norm: u64,
    key_bias: u64,
    query_projection: u64,
    weights_projection: u64,
    query_rotated: u64,
    key_cache: u64,
    cos: u64,
    sin: u64,
    positions: u64,
    scores: u64,
};

pub const Shape = struct {
    rows: usize,
    nkeys: usize,
    key_stride: usize = 128,
    key_col: usize = 0,
    iw_stride: usize = 32,
    iw_col: usize = 0,
    score_stride: usize,
    rank: usize,
    world: usize,
    eps: f32 = 1e-6,
};

pub const Bound = struct {
    layer: u16,
    index_from: u16,
    runs_indexer: bool,
    buffers: Buffers,
    shape: Shape,
    tokens: TokenListBuffers,

    /// Boundary: key LayerNorm uses CTA barriers with a 128-thread wrapper.
    pub fn keyGrid(self: Bound) Grid {
        return .{ .x = @intCast(self.shape.rows), .y = 1, .z = 1, .blocks = @intCast(self.shape.rows), .threads = 128, .dynamic_smem = 0 };
    }

    /// One index query CTA per row/head (32 x rows), block=128, smem=0.
    pub fn queryGrid(self: Bound) Grid {
        return .{ .x = 32, .y = @intCast(self.shape.rows), .z = 1, .blocks = 32, .threads = 128, .dynamic_smem = 0 };
    }

    /// One scoring CTA per row/16-key tile, block=256, 11,392 B dynamic score scratch.
    pub fn scoreGrid(self: Bound) Grid {
        const tiles = (self.shape.nkeys + 15) / 16;
        return .{ .x = @intCast(tiles), .y = @intCast(self.shape.rows), .z = 1, .blocks = @intCast(tiles), .threads = 256, .dynamic_smem = 11_392 };
    }

    /// Local radix top-k is a cluster boundary task (512 threads, 67,584 B dynamic smem).
    pub fn topkGrid(self: Bound) Grid {
        const blocks = @min(8, (self.shape.nkeys + 16_895) / 16_896);
        return .{ .x = @intCast(blocks), .y = 1, .z = 1, .blocks = @intCast(blocks), .threads = 512, .dynamic_smem = 67_584 };
    }

    /// Boundary: DCP merge/filter/remap requires 512 threads and CTA barriers.
    pub fn mergeGrid(self: Bound) Grid {
        return .{ .x = @intCast(self.shape.rows), .y = 1, .z = 1, .blocks = @intCast(self.shape.rows), .threads = 512, .dynamic_smem = 32_768 };
    }

    /// One persistent block executes the intentional no-op index-share dependency marker.
    pub fn shareGrid(_: Bound) Grid {
        return .{ .x = 1, .y = 1, .z = 1, .blocks = 1, .threads = 256, .dynamic_smem = 0 };
    }
};

pub const Grid = struct { x: u32, y: u32, z: u32, blocks: u32, threads: u32, dynamic_smem: u32 };

/// Byte layout of MkStepIndex in mk_dispatch_index.cuh (pointers followed by int32 fields).
pub const DeviceParams = extern struct {
    query_projection: u64,
    qi: u64,
    iw: u64,
    keys: u64,
    cos: u64,
    sin: u64,
    positions: u64,
    scores: u64,
    q_rotated: u64,
    rows: i32,
    nkeys: i32,
    iw_stride: i32,
    iw_col: i32,
    score_stride: i32,
    rank: i32,
    world: i32,
    threads: i32,
};

comptime {
    std.debug.assert(@sizeOf(DeviceParams) == 104);
}

pub fn deviceParams(bound: Bound, threads: u32) DeviceParams {
    return .{
        .query_projection = bound.buffers.query_projection,
        .qi = bound.buffers.query_rotated,
        .iw = bound.buffers.weights_projection,
        .keys = bound.buffers.key_cache,
        .cos = bound.buffers.cos,
        .sin = bound.buffers.sin,
        .positions = bound.buffers.positions,
        .scores = bound.buffers.scores,
        .q_rotated = bound.buffers.query_rotated,
        .rows = @intCast(bound.shape.rows),
        .nkeys = @intCast(bound.shape.nkeys),
        .iw_stride = @intCast(bound.shape.iw_stride),
        .iw_col = @intCast(bound.shape.iw_col),
        .score_stride = @intCast(bound.shape.score_stride),
        .rank = @intCast(bound.shape.rank),
        .world = @intCast(bound.shape.world),
        .threads = @intCast(threads),
    };
}

pub fn queryParams(bound: Bound) DeviceParams {
    return deviceParams(bound, bound.queryGrid().threads);
}

pub fn scoreParams(bound: Bound) DeviceParams {
    return deviceParams(bound, bound.scoreGrid().threads);
}

/// Bind exactly the values passed by forward.zig's index key/query/scores path.
/// `share.index_from` must be built from the parsed config, never a model-specific pattern.
pub fn bind(layer: usize, share: IndexShare, buffers: Buffers, shape: Shape) !Bound {
    if (layer >= share.index_from.len or layer > std.math.maxInt(u16)) return error.IndexShareShape;
    if (shape.rows == 0 or shape.nkeys == 0 or shape.world == 0 or shape.rank >= shape.world) return error.IndexShape;
    if (shape.key_stride < shape.key_col + 128 or shape.iw_stride < shape.iw_col + 32 or shape.score_stride < shape.nkeys) return error.IndexShape;
    const source = share.index_from[layer];
    if (source > layer or source >= share.index_from.len or source >= share.token_lists.len) return error.IndexShareMap;
    return .{
        .layer = @intCast(layer),
        .index_from = source,
        .runs_indexer = source == layer,
        .buffers = buffers,
        .shape = shape,
        .tokens = share.token_lists[source],
    };
}

/// Forward's row-batched pointers: qi advances by 32*128 BF16, iw by 32 BF16,
/// and positions by one i64 per row. Score storage starts at the row-batch base.
pub fn rowOffsets(base: Buffers, first_row: usize) !Buffers {
    const qi_bytes = try std.math.mul(usize, first_row, 32 * 128 * 2);
    const iw_bytes = try std.math.mul(usize, first_row, 32 * 2);
    const pos_bytes = try std.math.mul(usize, first_row, 8);
    return .{
        .key_projection = try add(base.key_projection, try std.math.mul(usize, first_row, 128 * 2)),
        .key_norm = base.key_norm,
        .key_bias = base.key_bias,
        .query_projection = try add(base.query_projection, qi_bytes),
        .weights_projection = try add(base.weights_projection, iw_bytes),
        .query_rotated = try add(base.query_rotated, qi_bytes),
        .key_cache = base.key_cache,
        .cos = try add(base.cos, try std.math.mul(usize, first_row, 32 * 4)),
        .sin = try add(base.sin, try std.math.mul(usize, first_row, 32 * 4)),
        .positions = try add(base.positions, pos_bytes),
        .scores = base.scores,
    };
}

fn add(ptr: u64, offset: usize) !u64 {
    return std.math.add(u64, ptr, @intCast(offset));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(IndexShare);
    std.testing.refAllDecls(TokenListBuffers);
    std.testing.refAllDecls(Buffers);
    std.testing.refAllDecls(Shape);
    std.testing.refAllDecls(Bound);
    std.testing.refAllDecls(Grid);
    std.testing.refAllDecls(DeviceParams);
}

test "index binder follows config sharing map and forward batch offsets" {
    const buffers: Buffers = .{
        .key_projection = 0x1000,
        .key_norm = 0x2000,
        .key_bias = 0x3000,
        .query_projection = 0x4000,
        .weights_projection = 0x5000,
        .query_rotated = 0x6000,
        .key_cache = 0x7000,
        .cos = 0x8000,
        .sin = 0x9000,
        .positions = 0xa000,
        .scores = 0xb000,
    };
    const shape: Shape = .{ .rows = 4, .nkeys = 65_537, .score_stride = 65_537, .rank = 1, .world = 2 };
    const map = [_]u16{ 0, 0, 2, 2, 4 };
    const lists = [_]TokenListBuffers{
        .{ .positions = 100, .counts = 101, .local_slots = 102, .local_counts = 103, .dense_len = 104 },
        .{ .positions = 200, .counts = 201, .local_slots = 202, .local_counts = 203, .dense_len = 204 },
        .{ .positions = 300, .counts = 301, .local_slots = 302, .local_counts = 303, .dense_len = 304 },
        .{ .positions = 400, .counts = 401, .local_slots = 402, .local_counts = 403, .dense_len = 404 },
        .{ .positions = 500, .counts = 501, .local_slots = 502, .local_counts = 503, .dense_len = 504 },
    };
    const sharing: IndexShare = .{ .index_from = &map, .token_lists = &lists };
    const shared = try bind(3, sharing, buffers, shape);
    try std.testing.expect(!shared.runs_indexer);
    try std.testing.expectEqual(@as(u16, 2), shared.index_from);
    try std.testing.expectEqual(@as(u64, 300), shared.tokens.positions);
    const full = try bind(4, sharing, buffers, shape);
    try std.testing.expect(full.runs_indexer);
    try std.testing.expectEqual(@as(u64, 500), full.tokens.positions);
    const offset = try rowOffsets(buffers, 2);
    try std.testing.expectEqual(@as(u64, 0x4000 + 2 * 32 * 128 * 2), offset.query_projection);
    try std.testing.expectEqual(@as(u64, 0x5000 + 2 * 32 * 2), offset.weights_projection);
    try std.testing.expectEqual(@as(u64, 0xa000 + 16), offset.positions);
    try std.testing.expectEqual(@as(u32, 32), shared.queryGrid().x);
    try std.testing.expectEqual(@as(u32, 32), shared.queryGrid().blocks);
    try std.testing.expectEqual(@as(u32, 4), shared.keyGrid().blocks);
    try std.testing.expectEqual(@as(u32, 4097), shared.scoreGrid().x);
    try std.testing.expectEqual(@as(u32, 4097), shared.scoreGrid().blocks);
    try std.testing.expectEqual(@as(u32, 11_392), shared.scoreGrid().dynamic_smem);
    try std.testing.expectEqual(@as(u32, 4), shared.topkGrid().x);
    try std.testing.expectEqual(@as(u32, 67_584), shared.topkGrid().dynamic_smem);
    try std.testing.expectEqual(@as(u32, 4), shared.mergeGrid().blocks);
    try std.testing.expectEqual(@as(u32, 32_768), shared.mergeGrid().dynamic_smem);
    try std.testing.expectEqual(@as(u32, 1), shared.shareGrid().blocks);
    const device = queryParams(shared);
    try std.testing.expectEqual(@as(u64, 0x4000), device.query_projection);
    try std.testing.expectEqual(@as(u64, 0x6000), device.qi);
    try std.testing.expectEqual(@as(i32, 128), device.threads);
    try std.testing.expectEqual(@as(i32, 256), scoreParams(shared).threads);
}
