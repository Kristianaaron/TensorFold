// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Launch limits for GLM-5.3 CUDA kernels; keep these beside the owning family.
const std = @import("std");

/// The exact top-k CTA image tile at 512 threads and 33 entries per thread.
pub const topk_threads = 512;
pub const topk_entries_per_thread = 33;
pub const topk_dynamic_smem = topk_threads * topk_entries_per_thread * @sizeOf(u32);

/// Indexer key-position capacity is 33 entries/thread across eight CTAs.
pub const max_local_keys = topk_threads * topk_entries_per_thread * 8;

test "GLM-5.3 index launch contract is fully analyzed" {
    std.testing.refAllDecls(@This());
    try std.testing.expectEqual(@as(usize, 67584), topk_dynamic_smem);
    try std.testing.expectEqual(@as(usize, 135168), max_local_keys);
}
