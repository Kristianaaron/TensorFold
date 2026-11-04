// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! The loader owns each tail allocation until all books, scratch and tables are ready.
const std = @import("std");
const df = @import("dflash.zig");
pub const Sizes = struct {
    book: usize,
    ctx: usize,
    cand: usize,
    hidden: usize,
    gh: usize,
    kind: df.Geometry.Kind,
    confidence: bool,
    block_size: usize,
};

pub fn acquire(g: std.mem.Allocator, dr: anytype, ctx: anytype, comptime Ops: type, n: Sizes) !void {
    // Each owner's errdefer must sit at function scope: one inside an if/else block expires at the block's end.
    if (n.kind == .dspark) try acquireDspark(g, dr, ctx, Ops, n) else try acquireBooks(g, dr, ctx, Ops, n);
    errdefer releaseHead(g, dr, n.kind);
    try Ops.scratch(ctx, dr);
    errdefer Ops.freeScratch(ctx, dr);
    dr.h_cpos = try g.alloc(i64, n.ctx);
    errdefer g.free(dr.h_cpos);
    dr.h_crings = try g.alloc(u64, n.ctx);
    errdefer g.free(dr.h_crings);
    dr.cand = try g.alloc(i32, n.cand);
    errdefer g.free(dr.cand);
    dr.gh = try g.alloc(f32, n.gh);
    errdefer g.free(dr.gh);
    if (n.confidence) dr.hrows = try g.alloc(u16, 16 * n.block_size * n.hidden);
}

fn acquireBooks(g: std.mem.Allocator, dr: anytype, ctx: anytype, comptime Ops: type, n: Sizes) !void {
    dr.pred = try g.alloc(u16, n.book);
    errdefer g.free(dr.pred);
    try Ops.readBook(ctx, 0, dr.pred);
    dr.succ = try g.alloc(u16, n.book);
    errdefer g.free(dr.succ);
    try Ops.readBook(ctx, 1, dr.succ);
}

fn acquireDspark(g: std.mem.Allocator, dr: anytype, ctx: anytype, comptime Ops: type, n: Sizes) !void {
    dr.markov_w1 = try g.alloc(u16, n.book);
    errdefer g.free(dr.markov_w1);
    dr.markov_w2 = try g.alloc(u16, n.book);
    errdefer g.free(dr.markov_w2);
    dr.conf_w = try g.alloc(u16, 6400);
    errdefer g.free(dr.conf_w);
    try Ops.readDspark(ctx, dr);
}

fn releaseHead(g: std.mem.Allocator, dr: anytype, kind: df.Geometry.Kind) void {
    if (kind == .dspark) {
        g.free(dr.markov_w1);
        g.free(dr.markov_w2);
        g.free(dr.conf_w);
    } else {
        g.free(dr.pred);
        g.free(dr.succ);
    }
}
