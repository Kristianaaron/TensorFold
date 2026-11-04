// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Shared host-side buffer contracts for the GLM-5.3 CUDA packages.
const std = @import("std");
const memory = @import("cuda_memory");

/// Scalar encoding stored in a device buffer.
pub const DType = enum { bf16, f32, u8, i32, i64 };
/// Underlying CUDA allocation owner.
pub const Buffer = memory.DeviceBuffer;

/// Device allocation paired with its scalar encoding.
pub fn TypedBuffer(comptime encoding: DType) type {
    return struct {
        /// Compile-time scalar encoding for this buffer type.
        pub const dtype = encoding;
        /// Owned or borrowed CUDA allocation.
        buffer: Buffer,
    };
}

/// Float32 device allocation.
pub const F32Buffer = TypedBuffer(.f32);
/// BFloat16 device allocation.
pub const Bf16Buffer = TypedBuffer(.bf16);
/// Byte device allocation, including FP8 codes.
pub const U8Buffer = TypedBuffer(.u8);
/// Signed 32-bit device allocation.
pub const I32Buffer = TypedBuffer(.i32);
/// Signed 64-bit device allocation for global sequence positions.
pub const I64Buffer = TypedBuffer(.i64);

/// One row-major cache plane stored at a single dtype.
pub const Plane = struct {
    /// Device allocation for the plane elements.
    data: Buffer,
    /// Element encoding for `data`.
    dtype: DType,
    /// Number of cache rows allocated.
    rows: usize,
    /// Number of elements in each row.
    cols: usize,
    /// Element stride between row starts.
    stride: usize,
};

/// KV cache allocations and the DCP placement contract for this rank.
pub const Planes = struct {
    /// Latent cache rows, bf16 or FP8 according to the selected KV format.
    latent: Plane,
    /// RoPE cache rows, bf16 or FP8 according to the selected KV format.
    rope: Plane,
    /// Optional DSA index-key cache plane.
    index: ?Plane = null,
    /// Optional fp32 scale per row for e4m3 latent and rope planes.
    kv8_scale: ?F32Buffer = null,
    /// Number of slots in the DCP cycle (1 replicated, 2 interleaved).
    dcp_slots: usize,
    /// Rank owning this cache allocation.
    rank: usize,
    /// Number of tensor-parallel ranks sharing the cache.
    world: usize,
    /// Global position capacity of each plane.
    capacity: usize,
};

/// Latent and RoPE planes paired for one prefill stage buffer.
pub const StagePlanePair = struct {
    latent: Plane,
    rope: Plane,
};

/// Two peer-cache stage buffers shared across all model layers; layer L uses L % 2.
/// The current MLA stage kernels read BF16 rows: 512 latent + 64 RoPE elements = 1,152 bytes/row.
pub const StagePlanes = struct {
    buffers: [2]StagePlanePair,
    rows: usize,
    capacity: usize,
    world: usize,

    /// Absolute-slot extent for a context-capacity allocation.
    pub fn rowsForCapacity(context_capacity: usize, world: usize) !usize {
        if (world == 0) return error.InvalidStageWorld;
        return std.math.divCeil(usize, context_capacity, world);
    }

    /// Stage row bytes for the current MLA kernel's BF16 reads.
    pub fn bytesPerRow(dtype: DType) !usize {
        if (dtype != .bf16) return error.UnsupportedStageDType;
        return (512 + 64) * @sizeOf(u16);
    }

    /// Select the reusable pair assigned to a layer in the two-buffer pipeline.
    pub fn forLayer(self: *const StagePlanes, layer: usize) *const StagePlanePair {
        return &self.buffers[layer % self.buffers.len];
    }

    /// Verify both shared pairs match the exact BF16 layout read by mla.cu.
    pub fn validate(self: *const StagePlanes) !void {
        if (self.world == 0 or self.rows != try rowsForCapacity(self.capacity, self.world)) return error.InvalidStageExtent;
        for (self.buffers) |pair| {
            if (pair.latent.rows != self.rows or pair.rope.rows != self.rows) return error.InvalidStageExtent;
            if (pair.latent.dtype != .bf16 or pair.rope.dtype != .bf16) return error.UnsupportedStageDType;
            if (pair.latent.cols != 512 or pair.latent.stride != 512 or pair.rope.cols != 64 or pair.rope.stride != 64) return error.InvalidStageLayout;
        }
    }
};

/// A layer's device allocation and element count.
pub const LayerBuffer = struct {
    /// Layer index represented by this allocation.
    layer: usize,
    /// Device allocation for the layer data.
    data: Buffer,
    /// Number of elements in the allocation.
    elements: usize,
};

/// A row-major batch of token vectors.
pub const TokenRows = struct {
    /// Device allocation containing the token vectors.
    data: Buffer,
    /// Number of token rows.
    rows: usize,
    /// Elements per token row.
    width: usize,
    /// Element encoding for the token rows.
    dtype: DType,
};

/// An fp32 row-major partial produced by a rank-local block.
pub const Partial = struct {
    /// Device allocation containing partial values.
    data: F32Buffer,
    /// Number of token rows.
    rows: usize,
    /// Elements per token row.
    width: usize,
};

/// Global token positions selected for each query row plus rank-local cache slots.
pub const TokenList = struct {
    /// Global sequence positions in ascending order; -1 pads unused entries.
    positions: I32Buffer,
    /// Counts of valid global positions for each query row.
    counts: I32Buffer,
    /// Rank-local cache slot for each valid position, after DCP ownership mapping.
    local_slots: I32Buffer,
    /// Counts of valid rank-local slots for each query row (this rank's share after DCP ownership).
    local_counts: I32Buffer,
    /// Per-row dense key count: global position + 1 for rows below the 2048 dense threshold, 0 for sparse rows.
    dense_len: I32Buffer,
    /// Number of query rows represented by the list.
    rows: usize,
    /// Maximum selected positions per query row.
    capacity: usize = 2048,
};

/// MLA query, absorbed query, and distributed attention output buffers.
pub const MlaBuffers = struct {
    /// Query rows with latent and RoPE components.
    query: TokenRows,
    /// Query rows after applying the kv_b absorb weights.
    absorbed: TokenRows,
    /// Local packed q_pe values in [rows][local_heads][64] layout; DCP gather exposes [group][rows][32][64].
    q_pe: TokenRows,
    /// Rank-local bf16 normalized attention partials in [group][rows][32][512] layout.
    partial: Bf16Buffer,
    /// Rank-local fp32 log-sum-exp values in [group][rows][32] layout.
    lse: F32Buffer,
};
/// Short name used by kernel packages.
pub const Mla = MlaBuffers;

/// DSA indexer query, scores, head weights, and selected positions.
pub const IndexBuffers = struct {
    /// Indexer query vectors after RoPE, for this layer's heads.
    query: TokenRows,
    /// Int64 global sequence position for each query row.
    query_positions: I64Buffer,
    /// Indexer scores before top-k selection.
    scores: F32Buffer,
    /// Per-head weights projected from the input rows.
    iw: Bf16Buffer,
    /// Element stride between rows of the fused projection's per-head weights.
    iw_stride: usize,
    /// Element offset of the head weights within each fused projection row.
    iw_col: usize,
    /// Local top-k candidate scores, shaped [rows][selected.capacity].
    candidate_scores: F32Buffer,
    /// Local top-k candidate global positions, shaped [rows][selected.capacity].
    candidate_positions: I32Buffer,
    /// All-gathered candidate scores, shaped [world][rows][selected.capacity].
    gathered_scores: F32Buffer,
    /// All-gathered candidate positions, shaped [world][rows][selected.capacity].
    gathered_positions: I32Buffer,
    /// Selected global token positions and local cache slots.
    selected: TokenList,
};
/// Short name used by kernel packages.
pub const Index = IndexBuffers;

/// MoE routing, mixture weights, and low-rank correction buffers.
pub const MoeBuffers = struct {
    /// fp32 router logits for each row and routed expert.
    router_logits: F32Buffer,
    /// Selected routed expert ids, with the shared slot represented separately.
    choices: I32Buffer,
    /// fp32 mixture weights for routed and shared slots.
    weights: F32Buffer,
    /// Rank-local fp32 output partial.
    partial: Partial,
    /// FP32 resident copy of the BF16-on-disk mlp.lowrank_a factor, rank-local R columns.
    lowrank_a: F32Buffer,
    /// FP32 resident copy of the BF16-on-disk mlp.lowrank_b factor, rank-local R rows.
    lowrank_b: F32Buffer,
    /// Global low-rank correction width R before splitting across ranks.
    lowrank_rank: usize,
    /// Global R-column span owned by this rank.
    lowrank_r: Span,
};
/// Short name used by kernel packages.
pub const Moe = MoeBuffers;

/// Dense MLP intermediate and output buffers.
pub const MlpBuffers = struct {
    /// Gate and up projection rows.
    gate_up: TokenRows,
    /// SwiGLU activation rows.
    activation: TokenRows,
    /// Rank-local fp32 output partial.
    partial: Partial,
};
/// Short name used by kernel packages.
pub const Mlp = MlpBuffers;

/// Tensor-parallel interval in global element or channel coordinates.
pub const Span = struct {
    /// Inclusive first global coordinate.
    lo: usize,
    /// Exclusive last global coordinate.
    hi: usize,
};

/// Forward-pass scratch buffers.
pub const ForwardBuffers = struct {
    /// Residual stream rows.
    residual: TokenRows,
    /// RMS-normalized rows.
    normalized: TokenRows,
    /// MLA attention work buffers.
    attention: MlaBuffers,
    /// MoE block work buffers.
    moe: MoeBuffers,
    /// Dense MLP work buffers.
    mlp: MlpBuffers,
    /// Two prefill peer-cache planes shared across layers; ignored for decode.
    stage: StagePlanes,
};
/// Short name used by kernel packages.
pub const Forward = ForwardBuffers;

test {
    std.testing.refAllDecls(@This());
}

/// Persistent per-layer cache and index state.
pub const StateBuffers = struct {
    /// Cache planes for every model layer.
    planes: []Planes,
    /// Optional indexer buffers, absent on shared-indexer layers.
    indices: []?IndexBuffers,
};
/// Short name used by engine packages.
pub const State = StateBuffers;

/// Final vocabulary logits and sampling candidates.
pub const HeadBuffers = struct {
    /// Device allocation containing logits.
    logits: F32Buffer,
    /// Device allocation containing local top candidates.
    candidates: F32Buffer,
    /// Number of input token rows.
    rows: usize,
    /// Vocabulary width represented by this rank.
    vocab: usize,
};
/// Short name used by engine packages.
pub const Head = HeadBuffers;

/// DFlash2 draft model buffers.
pub const DrafterBuffers = struct {
    /// Target-model hidden tap rows consumed by the drafter.
    taps: TokenRows,
    /// Draft token ring buffer.
    ring: I32Buffer,
    /// Draft vocabulary head allocation.
    head: Buffer,
};
/// Short name used by engine packages.
pub const Drafter = DrafterBuffers;

test "typed buffer encodings and token-list defaults are stable" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(TokenList);
    try std.testing.expectEqual(DType.f32, F32Buffer.dtype);
    try std.testing.expectEqual(DType.bf16, Bf16Buffer.dtype);
    try std.testing.expectEqual(DType.u8, U8Buffer.dtype);
    try std.testing.expectEqual(DType.i32, I32Buffer.dtype);
    try std.testing.expectEqual(DType.i64, I64Buffer.dtype);
    const tokens: TokenList = .{ .positions = undefined, .counts = undefined, .local_slots = undefined, .local_counts = undefined, .dense_len = undefined, .rows = 0 };
    try std.testing.expectEqual(@as(usize, 2048), tokens.capacity);
}

test "contract declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plane);
    std.testing.refAllDecls(Planes);
    std.testing.refAllDecls(StagePlanePair);
    std.testing.refAllDecls(StagePlanes);
    std.testing.refAllDecls(LayerBuffer);
    std.testing.refAllDecls(TokenRows);
    std.testing.refAllDecls(Partial);
    std.testing.refAllDecls(TokenList);
    std.testing.refAllDecls(MlaBuffers);
    std.testing.refAllDecls(IndexBuffers);
    std.testing.refAllDecls(MoeBuffers);
    std.testing.refAllDecls(MlpBuffers);
    std.testing.refAllDecls(ForwardBuffers);
    std.testing.refAllDecls(StateBuffers);
    std.testing.refAllDecls(HeadBuffers);
    std.testing.refAllDecls(DrafterBuffers);
}

test "prefill stage planes are two BF16 peer buffers shared by layer parity" {
    try std.testing.expectEqual(@as(usize, 131_072), try StagePlanes.rowsForCapacity(262_144, 2));
    try std.testing.expectEqual(@as(usize, 1_152), try StagePlanes.bytesPerRow(.bf16));
    try std.testing.expectError(error.UnsupportedStageDType, StagePlanes.bytesPerRow(.u8));
    var stages: StagePlanes = undefined;
    try std.testing.expectEqual(@intFromPtr(&stages.buffers[0]), @intFromPtr(stages.forLayer(0)));
    try std.testing.expectEqual(@intFromPtr(&stages.buffers[1]), @intFromPtr(stages.forLayer(1)));
    try std.testing.expectEqual(@intFromPtr(&stages.buffers[0]), @intFromPtr(stages.forLayer(2)));
}

test "stage plane validation enforces the shared capacity and MLA BF16 row layout" {
    const latent = Plane{ .data = .{ .d = undefined, .ptr = 1, .len = 512 * 131_072 * 2 }, .dtype = .bf16, .rows = 131_072, .cols = 512, .stride = 512 };
    const rope = Plane{ .data = .{ .d = undefined, .ptr = 2, .len = 64 * 131_072 * 2 }, .dtype = .bf16, .rows = 131_072, .cols = 64, .stride = 64 };
    const good: StagePlanePair = .{ .latent = latent, .rope = rope };
    const stages: StagePlanes = .{ .buffers = .{ good, good }, .rows = 131_072, .capacity = 262_144, .world = 2 };
    try stages.validate();

    var bad = stages;
    bad.buffers[1].rope.stride = 65;
    try std.testing.expectError(error.InvalidStageLayout, bad.validate());
    bad = stages;
    bad.rows -= 1;
    try std.testing.expectError(error.InvalidStageExtent, bad.validate());
}
