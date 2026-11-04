// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! `tf-glm53-bench prefill`: arithmetic estimate from the port's per-rank FLOP/rate contract.
const std = @import("std");

const Case = struct { context: usize, mla_gflop: f64, index_gflop: f64, topk_us: f64 };
const cases = [_]Case{
    .{ .context = 4096, .mla_gflop = 7.30, .index_gflop = 0.07, .topk_us = 0.5 },
    .{ .context = 32768, .mla_gflop = 10.78, .index_gflop = 1.40, .topk_us = 1.0 },
    .{ .context = 131072, .mla_gflop = 11.04, .index_gflop = 5.64, .topk_us = 6.0 },
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 or !std.mem.eql(u8, args[1], "prefill")) {
        std.debug.print("usage: tf-glm53-bench prefill\n", .{});
        return 2;
    }
    report();
    return 0;
}

/// Uses prefill FLOP and rate estimates, updated for the BF16 cuBLASLt backbone path.
pub fn report() void {
    std.debug.print("GLM-5.3 prefill estimate, TP2 per rank (FLOPs / sustained TFLOPS; not a Spark measurement)\n", .{});
    std.debug.print("context op_class us_per_token\n", .{});
    for (cases) |c| {
        const projections = 14.35 / 88.0 * 1e3;
        const routed = 22.65 / 125.0 * 1e3;
        const shared_dense = 3.51 / 88.0 * 1e3;
        const router = 0.24 / 20.0 * 1e3;
        const mla = c.mla_gflop / 55.0 * 1e3;
        const indexer = c.index_gflop / 70.0 * 1e3;
        std.debug.print("{d} projections BF16 GEMM       {d:.1}\n", .{ c.context, projections });
        std.debug.print("{d} routed expert GEMM          {d:.1}\n", .{ c.context, routed });
        std.debug.print("{d} shared+dense BF16 GEMM       {d:.1}\n", .{ c.context, shared_dense });
        std.debug.print("{d} router                        {d:.1}\n", .{ c.context, router });
        std.debug.print("{d} sparse MLA BF16 core         {d:.1}\n", .{ c.context, mla });
        std.debug.print("{d} indexer BF16 scores          {d:.1}\n", .{ c.context, indexer });
        const compute = projections + routed + shared_dense + router + mla + indexer;
        const fixed = compute + 35 + 11 + c.topk_us;
        const total = fixed * 1.04;
        std.debug.print("{d} total incl 35 us glue, 11 us exchange, top-k, 4% tails: {d:.1} us -> {d:.0} tok/s\n", .{ c.context, total, 1e6 / total });
    }
}

test "prefill estimate covers the requested prompt lengths and finite throughput" {
    try std.testing.expectEqual(@as(usize, 4096), cases[0].context);
    try std.testing.expectEqual(@as(usize, 32768), cases[1].context);
    try std.testing.expectEqual(@as(usize, 131072), cases[2].context);
    const total = (14.35 / 88.0 + 22.65 / 125.0 + 3.51 / 88.0 + 0.24 / 20.0 + 10.78 / 55.0 + 1.4 / 70.0) * 1e3;
    try std.testing.expect(total > 600 and total < 800);
}

test {
    std.testing.refAllDecls(@This());
}
