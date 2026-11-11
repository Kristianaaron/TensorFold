// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Config-derived geometry shared by the GLM-5.3 DFlash loader and taps.
pub const Geometry = struct {
    pub const Kind = enum { dflash2, dspark };
    kind: Kind = .dflash2,
    conv: bool = true,
    markov_top_k: usize = 64,
    confidence: bool = false,
    sample_from_anchor: bool = false,
    hidden: usize = 6144,
    layers: usize = 6,
    heads: usize = 64,
    kv_heads: usize = 8,
    head_dim: usize = 128,
    intermediate: usize = 12288,
    vocab: usize = 154880,
    block: usize = 8,
    window: usize = 2048,
    mask: u32 = 154856,
    selector_rank: usize = 256,
    selector_top_k: usize = 16,
    target_layers: []const usize = &.{ 5, 19, 33, 47, 61, 75 },
    conv_group_size: usize = 16,
    conv_kernel_size: usize = 2,
    rope_theta: f32 = 1e6,
    rms_norm_eps: f32 = 1e-5,
    /// TF_GLM53_DRAFT_DOWN4: each rank's down K-half in 4 bits (fp32-out tf_qmm_group).
    down4: bool = false,

    pub fn kvWidth(g: Geometry) usize {
        return g.kv_heads * g.head_dim;
    }
    pub fn queryWidth(g: Geometry) usize {
        return g.heads * g.head_dim;
    }
    pub fn qkvWidth(g: Geometry) usize {
        return g.queryWidth() + 2 * g.kvWidth();
    }
    pub fn convWidth(g: Geometry) usize {
        return 2 * g.conv_kernel_size * g.hidden / g.conv_group_size;
    }
    pub fn tapRowBytes(g: Geometry) usize {
        return g.target_layers.len * g.hidden * 2;
    }
    pub fn ringBytes(g: Geometry) u64 {
        return @intCast(g.layers * 2 * g.window * g.kvWidth() * 2);
    }
};
