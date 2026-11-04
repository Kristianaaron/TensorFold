// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host-built, fixed-shape task DAG for the GLM-5.3 persistent decode interpreter (MK1).
const std = @import("std");

pub const Kind = enum(u8) {
    input_norm,
    q_a,
    q_norm,
    q_b,
    kv_write,
    index_query,
    index_scores,
    index_topk,
    index_exchange,
    absorb,
    query_exchange,
    attend,
    partial_exchange,
    partial_merge,
    o_proj,
    tp_exchange,
    residual_post_norm,
    router_topk,
    expert_gather_decode_gemv,
    shared_expert,
    lowrank,
    moe_combine,
    moe_exchange_residual,
    dense_mlp,
    final_norm,
    head_logits,
    head_candidates,
    // Split points needed by the converted CUDA task functions. Keep these at the end:
    // Kind's byte value is part of the persistent task record ABI.
    qpe_pack,
    index_key_write,
    expand_v,
    // MK4 host planner nodes: ordinary qmmf launches cut persistent segments.
    kv_projection,
    index_projection,
    // MK6 qmmf projection split tasks. `aux` stores the original projection Kind.
    qmmf_partial,
    qmmf_reduce,
};

pub const Task = struct {
    kind: Kind,
    layer: u16,
    row: u8,
    aux: u16 = 0,
    dependency_count: u16 = 0,
    successor_offset: u32 = 0,
    successor_count: u16 = 0,
};

pub const Params = struct { rows: usize, layers: usize = 78, dcp: usize = 2, rank: usize = 0 };

pub const Graph = struct {
    allocator: std.mem.Allocator,
    params: Params,
    tasks: []Task,
    initial_counters: []u32,
    successors: []u32,

    pub fn deinit(g: *Graph) void {
        g.allocator.free(g.tasks);
        g.allocator.free(g.initial_counters);
        g.allocator.free(g.successors);
        g.* = undefined;
    }

    /// Binary device layout: Task records followed by one aligned u32 counter per task,
    /// then a u32 successor CSR array. CUDA graph replay resets counters from initial_counters.
    pub fn deviceBytes(g: Graph) !usize {
        const task_bytes = try std.math.mul(usize, g.tasks.len, @sizeOf(Task));
        const counter_bytes = try std.math.mul(usize, g.initial_counters.len, @sizeOf(u32));
        const offset = std.mem.alignForward(usize, task_bytes, @alignOf(u32));
        const edge_bytes = try std.math.mul(usize, g.successors.len, @sizeOf(u32));
        return try std.math.add(usize, try std.math.add(usize, offset, counter_bytes), edge_bytes);
    }

    /// Stable review dump. Task indices are deterministic; each task lists its ready successors.
    pub fn dump(g: Graph, writer: anytype) !void {
        try writer.print("GLM53-MK1 rows={d} layers={d} dcp={d} rank={d} tasks={d} edges={d} bytes={d}\n", .{ g.params.rows, g.params.layers, g.params.dcp, g.params.rank, g.tasks.len, g.successors.len, try g.deviceBytes() });
        for (g.tasks, 0..) |task, i| {
            try writer.print("{d:0>5} L{d:0>2} r{d:0>2} {s} deps={d} ->", .{ i, task.layer, task.row, @tagName(task.kind), task.dependency_count });
            const start: usize = task.successor_offset;
            for (g.successors[start .. start + task.successor_count]) |to| try writer.print(" {d}", .{to});
            try writer.writeByte('\n');
        }
    }
};

const Builder = struct {
    allocator: std.mem.Allocator,
    tasks: std.ArrayList(Task) = .empty,
    edges: std.ArrayList([2]u32) = .empty,

    fn add(b: *Builder, kind: Kind, layer: usize, row: usize, deps: []const u32, aux: u16) !u32 {
        const id: u32 = @intCast(b.tasks.items.len);
        const task: Task = .{ .kind = kind, .layer = @intCast(layer), .row = @intCast(row), .aux = aux, .dependency_count = @intCast(deps.len) };
        try b.tasks.append(b.allocator, task);
        for (deps) |from| try b.edges.append(b.allocator, .{ from, id });
        return id;
    }

    fn addQmmf(b: *Builder, projection: Kind, layer: usize, row: usize, deps: []const u32) !u32 {
        const shape: u16 = @intCast(@backingInt(projection));
        const partial = try b.add(.qmmf_partial, layer, row, deps, shape);
        return b.add(.qmmf_reduce, layer, row, &.{partial}, shape);
    }

    fn finish(b: *Builder, p: Params) !Graph {
        const tasks = try b.allocator.dupe(Task, b.tasks.items);
        errdefer b.allocator.free(tasks);
        const counters = try b.allocator.alloc(u32, tasks.len);
        errdefer b.allocator.free(counters);
        const successors = try b.allocator.alloc(u32, b.edges.items.len);
        errdefer b.allocator.free(successors);
        var counts = try b.allocator.alloc(u16, tasks.len);
        defer b.allocator.free(counts);
        @memset(counts, 0);
        for (b.edges.items) |edge| counts[edge[0]] += 1;
        var offset: usize = 0;
        for (tasks, 0..) |*task, i| {
            task.successor_offset = @intCast(offset);
            task.successor_count = counts[i];
            counters[i] = task.dependency_count;
            offset += counts[i];
        }
        @memset(counts, 0);
        for (b.edges.items) |edge| {
            const from: usize = edge[0];
            successors[tasks[from].successor_offset + counts[from]] = edge[1];
            counts[from] += 1;
        }
        return .{ .allocator = b.allocator, .params = p, .tasks = tasks, .initial_counters = counters, .successors = successors };
    }
};

/// Build an insertion-ordered DAG. Routers fill a fixed R*8 expert-slot table at runtime;
/// expert tasks compact/deduplicate those ids on device without changing graph storage shape.
pub fn build(allocator: std.mem.Allocator, p: Params) !Graph {
    if (p.rows == 0 or p.rows > 16 or p.layers == 0 or p.layers > 78 or (p.dcp != 1 and p.dcp != 2) or p.rank >= p.dcp) return error.BadGraphParams;
    var b: Builder = .{ .allocator = allocator };
    defer b.tasks.deinit(allocator);
    defer b.edges.deinit(allocator);
    var previous_layer: ?[]u32 = null;
    defer if (previous_layer) |v| allocator.free(v);
    for (0..p.layers) |layer| {
        const row_tail = try allocator.alloc(u32, p.rows);
        errdefer allocator.free(row_tail);
        for (0..p.rows) |row| {
            const prior: []const u32 = if (previous_layer) |v| &.{v[row]} else &.{};
            const n = try b.add(.input_norm, layer, row, prior, 0);
            const qa = try b.addQmmf(.q_a, layer, row, &.{n});
            const qn = try b.add(.q_norm, layer, row, &.{qa}, 0);
            const qb = try b.addQmmf(.q_b, layer, row, &.{qn});
            const kv_proj = try b.addQmmf(.kv_projection, layer, row, &.{n});
            const kv = try b.add(.kv_write, layer, row, &.{kv_proj}, 0);
            var ikey = kv;
            var index_proj: ?u32 = null;
            if (layer >= 3) {
                index_proj = try b.addQmmf(.index_projection, layer, row, &.{ n, qn });
                ikey = try b.add(.index_key_write, layer, row, &.{index_proj.?}, 0);
            }
            const qpe = try b.add(.qpe_pack, layer, row, &.{qb}, 0);
            var index_tail: ?u32 = null;
            if (layer >= 3) {
                const iq = try b.add(.index_query, layer, row, &.{ index_proj.?, qn }, 0);
                const is = try b.add(.index_scores, layer, row, &.{ iq, ikey }, 0);
                var it = try b.add(.index_topk, layer, row, &.{is}, 0);
                if (p.dcp == 2) it = try b.add(.index_exchange, layer, row, &.{it}, 0);
                index_tail = it;
            }
            const absorb = try b.add(.absorb, layer, row, &.{ qb, kv }, 0);
            const query_tail = if (p.dcp == 2) try b.add(.query_exchange, layer, row, &.{ absorb, qpe }, 0) else null;
            const attend_deps = if (query_tail) |qt|
                (if (index_tail) |it| &.{ qt, it } else &.{qt})
            else if (index_tail) |it|
                &.{ absorb, qpe, it }
            else
                &.{ absorb, qpe };
            const attend = try b.add(.attend, layer, row, attend_deps, 0);
            var merged = attend;
            if (p.dcp == 2) {
                const partial = try b.add(.partial_exchange, layer, row, &.{attend}, 0);
                merged = try b.add(.partial_merge, layer, row, &.{ attend, partial }, 0);
            }
            const expanded = try b.add(.expand_v, layer, row, &.{merged}, 0);
            const proj = try b.addQmmf(.o_proj, layer, row, &.{expanded});
            var residual = proj;
            if (p.dcp == 2) residual = try b.add(.tp_exchange, layer, row, &.{proj}, 0);
            const post = try b.add(.residual_post_norm, layer, row, &.{residual}, 0);
            if (layer < 3) {
                row_tail[row] = try b.addQmmf(.dense_mlp, layer, row, &.{post});
            } else {
                const route = try b.add(.router_topk, layer, row, &.{post}, 0);
                const exp = try b.add(.expert_gather_decode_gemv, layer, row, &.{route}, 0);
                const shared = try b.addQmmf(.shared_expert, layer, row, &.{post});
                const lowrank = try b.add(.lowrank, layer, row, &.{post}, 0);
                const combine = try b.add(.moe_combine, layer, row, &.{ exp, shared, lowrank }, 0);
                row_tail[row] = if (p.dcp == 2) try b.add(.moe_exchange_residual, layer, row, &.{combine}, 0) else combine;
            }
        }
        if (previous_layer) |old| allocator.free(old);
        previous_layer = row_tail;
    }
    const head = try allocator.alloc(u32, p.rows);
    defer allocator.free(head);
    for (0..p.rows) |row| {
        const norm = try b.add(.final_norm, p.layers, row, &.{previous_layer.?[row]}, 0);
        const logits = try b.addQmmf(.head_logits, p.layers, row, &.{norm});
        head[row] = try b.add(.head_candidates, p.layers, row, &.{logits}, 0);
    }
    return b.finish(p);
}

test "graph API declarations are analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Task);
    std.testing.refAllDecls(Params);
    std.testing.refAllDecls(Graph);
    std.testing.refAllDecls(Builder);
}

test "GLM53 graph has deterministic acyclic dependencies and fixed layout" {
    const p: Params = .{ .rows = 2, .layers = 4, .dcp = 2, .rank = 1 };
    var g = try build(std.testing.allocator, p);
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 208), g.tasks.len);
    try std.testing.expectEqual(@as(usize, 242), g.successors.len);
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Task));
    try std.testing.expectEqual(g.tasks.len, g.initial_counters.len);
    for (g.tasks, 0..) |task, i| {
        try std.testing.expect(task.successor_offset + task.successor_count <= g.successors.len);
        for (g.successors[task.successor_offset .. task.successor_offset + task.successor_count]) |to| try std.testing.expect(to > i);
    }
    for (g.tasks, 0..) |task, i| {
        var incoming: usize = 0;
        for (g.tasks) |producer| {
            for (g.successors[producer.successor_offset .. producer.successor_offset + producer.successor_count]) |to| {
                if (to == i) incoming += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, task.dependency_count), incoming);
        if (incoming == 0) try std.testing.expectEqual(@as(u32, 0), g.initial_counters[i]);
    }
    var again = try build(std.testing.allocator, p);
    defer again.deinit();
    try std.testing.expectEqualSlices(Task, g.tasks, again.tasks);
    try std.testing.expectEqualSlices(u32, g.successors, again.successors);
    try std.testing.expectEqual(@as(usize, 5128), try g.deviceBytes());
    var dump: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer dump.deinit();
    try g.dump(&dump.writer);
    try std.testing.expect(std.mem.startsWith(u8, dump.written(), "GLM53-MK1 rows=2 layers=4 dcp=2 rank=1"));
}

test "each qmmf projection is a payload-ready partial then reduce pair" {
    var g = try build(std.testing.allocator, .{ .rows = 1, .layers = 2, .dcp = 1 });
    defer g.deinit();
    var pairs: usize = 0;
    for (g.tasks, 0..) |task, i| {
        if (task.kind != .qmmf_partial) continue;
        const projection = std.enums.fromInt(Kind, task.aux) orelse return error.BadProjectionTag;
        try std.testing.expect(projection != .qmmf_partial and projection != .qmmf_reduce);
        const reduce_id = i + 1;
        try std.testing.expect(reduce_id < g.tasks.len);
        const reduce = g.tasks[reduce_id];
        try std.testing.expectEqual(Kind.qmmf_reduce, reduce.kind);
        try std.testing.expectEqual(task.aux, reduce.aux);
        try std.testing.expect(hasSuccessor(g, i, reduce_id));
        pairs += 1;
    }
    try std.testing.expectEqual(@as(usize, 11), pairs);
}

test "attention graph splits converted task functions at their data dependencies" {
    var g = try build(std.testing.allocator, .{ .rows = 1, .layers = 4, .dcp = 2 });
    defer g.deinit();
    var qpe: ?usize = null;
    var absorb: ?usize = null;
    var query_exchange: ?usize = null;
    var index_key: ?usize = null;
    var index_scores: ?usize = null;
    var merge: ?usize = null;
    var expand: ?usize = null;
    var o_proj: ?usize = null;
    var o_proj_partial: ?usize = null;
    for (g.tasks, 0..) |task, i| {
        if (task.layer != 3) continue;
        switch (task.kind) {
            .qpe_pack => qpe = i,
            .absorb => absorb = i,
            .query_exchange => query_exchange = i,
            .index_key_write => index_key = i,
            .index_scores => index_scores = i,
            .partial_merge => merge = i,
            .expand_v => expand = i,
            .qmmf_partial => if (task.aux == @backingInt(Kind.o_proj)) {
                o_proj_partial = i;
            },
            .qmmf_reduce => if (task.aux == @backingInt(Kind.o_proj)) {
                o_proj = i;
            },
            else => {},
        }
    }
    const qpe_id = qpe.?;
    const absorb_id = absorb.?;
    const query_id = query_exchange.?;
    const index_key_id = index_key.?;
    const scores_id = index_scores.?;
    const merge_id = merge.?;
    const expand_id = expand.?;
    const proj_id = o_proj.?;
    try std.testing.expect(hasSuccessor(g, qpe_id, query_id));
    try std.testing.expect(hasSuccessor(g, absorb_id, query_id));
    try std.testing.expect(hasSuccessor(g, index_key_id, scores_id));
    try std.testing.expect(hasSuccessor(g, merge_id, expand_id));
    try std.testing.expect(hasSuccessor(g, expand_id, o_proj_partial.?));
    try std.testing.expect(hasSuccessor(g, o_proj_partial.?, proj_id));
}

fn hasSuccessor(g: Graph, from: usize, to: usize) bool {
    const task = g.tasks[from];
    return std.mem.indexOfScalar(u32, g.successors[task.successor_offset .. task.successor_offset + task.successor_count], @intCast(to)) != null;
}

test "graph accepts TP1 and rejects invalid dimensions" {
    var g = try build(std.testing.allocator, .{ .rows = 1, .layers = 1, .dcp = 1, .rank = 0 });
    defer g.deinit();
    try std.testing.expectError(error.BadGraphParams, build(std.testing.allocator, .{ .rows = 17 }));
    try std.testing.expectError(error.BadGraphParams, build(std.testing.allocator, .{ .rows = 1, .dcp = 2, .rank = 2 }));
}
