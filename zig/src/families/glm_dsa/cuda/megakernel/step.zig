// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host step planner: cut a decode DAG into persistent launches and ordered boundary launches.
//! Forward tensor allocation and numerical kernels remain owned by forward.zig and their binders.
const std = @import("std");
const graph_api = @import("graph.zig");
const bind_qmmf = @import("bind_qmmf.zig");

pub const Boundary = enum(u8) {
    /// Retained in the review vocabulary; MK6 qmmf projections run as persistent task pairs.
    qmmf_projection,
    index_topk_cluster,
    dcp_exchange,
};

pub const SegmentKind = enum(u8) { persistent, boundary };

/// A contiguous topological slice. Boundary slices contain exactly one ordinary launch task.
pub const Segment = struct {
    kind: SegmentKind,
    first_task: u32,
    task_count: u32,
    boundary: ?Boundary = null,
};

/// Payload references are produced by the operation-specific bind_*.zig adapters. `bytes` is
/// copied by launch.Uploaded, which owns the aligned device payload arena for graph replay.
pub const Payload = struct {
    bytes: []const u8,

    /// Serialize one ABI record returned by a bind_*.zig adapter. The value must outlive upload.
    pub fn from(value: anytype) Payload {
        return .{ .bytes = std.mem.asBytes(value) };
    }
};

/// Encode a binder plan in the same order as its graph partial -> reduce pair.
/// The plan must outlive upload because Payload holds borrowed byte slices.
pub fn qmmfPayloadPair(plan: *const bind_qmmf.Plan) [2]Payload {
    return .{ Payload.from(&plan.partial), Payload.from(&plan.reduce) };
}

pub const Step = struct {
    allocator: std.mem.Allocator,
    graph: graph_api.Graph,
    segments: []Segment,
    payloads: []Payload,

    pub fn deinit(s: *Step) void {
        s.allocator.free(s.segments);
        s.allocator.free(s.payloads);
        s.graph.deinit();
        s.* = undefined;
    }

    pub fn boundaryCount(s: Step) usize {
        var count: usize = 0;
        for (s.segments) |segment| if (segment.kind == .boundary) {
            count += 1;
        };
        return count;
    }

    pub fn taskCount(s: Step) usize {
        return s.graph.tasks.len;
    }

    pub fn dump(s: Step, writer: anytype) !void {
        try writer.print("GLM53-MK step rows={d} layers={d} tasks={d} segments={d} boundaries={d}\n", .{
            s.graph.params.rows,
            s.graph.params.layers,
            s.graph.tasks.len,
            s.segments.len,
            s.boundaryCount(),
        });
        for (s.segments, 0..) |segment, index| {
            if (segment.kind == .boundary) {
                const task = s.graph.tasks[segment.first_task];
                try writer.print("{d:0>3} boundary {s} L{d:0>2} r{d:0>2} {s}\n", .{
                    index,
                    @tagName(segment.boundary.?),
                    task.layer,
                    task.row,
                    @tagName(task.kind),
                });
            } else {
                try writer.print("{d:0>3} persistent tasks={d} first={d}\n", .{ index, segment.task_count, segment.first_task });
            }
        }
    }
};

/// Build one decode step. Payloads are indexed by the resulting graph task id and must come from
/// the corresponding bind_attn/bind_index/bind_moe/bind_head adapter before upload.
pub fn build(allocator: std.mem.Allocator, params: graph_api.Params, payloads: []const Payload) !Step {
    var graph = try graph_api.build(allocator, params);
    errdefer graph.deinit();
    if (payloads.len != graph.tasks.len) return error.PayloadCountMismatch;
    const segments = try allocator.alloc(Segment, graph.tasks.len);
    errdefer allocator.free(segments);
    var nsegments: usize = 0;
    var first: usize = 0;
    while (first < graph.tasks.len) {
        const boundary = boundaryFor(graph.tasks[first].kind, params.dcp);
        if (boundary) |kind| {
            segments[nsegments] = .{ .kind = .boundary, .first_task = @intCast(first), .task_count = 1, .boundary = kind };
            nsegments += 1;
            first += 1;
            continue;
        }
        var end = first + 1;
        while (end < graph.tasks.len and boundaryFor(graph.tasks[end].kind, params.dcp) == null) : (end += 1) {}
        segments[nsegments] = .{ .kind = .persistent, .first_task = @intCast(first), .task_count = @intCast(end - first) };
        nsegments += 1;
        first = end;
    }
    const owned_segments = try allocator.dupe(Segment, segments[0..nsegments]);
    errdefer allocator.free(owned_segments);
    allocator.free(segments);
    const owned_payloads = try allocator.dupe(Payload, payloads);
    return .{ .allocator = allocator, .graph = graph, .segments = owned_segments, .payloads = owned_payloads };
}

/// Operations that still require an ordinary launch between persistent task segments.
/// Qmmf projections use task payloads from bind_qmmf and are not boundaries in MK6.
pub fn boundaryFor(kind: graph_api.Kind, dcp: usize) ?Boundary {
    return switch (kind) {
        .qmmf_partial, .qmmf_reduce => null,
        .index_topk => .index_topk_cluster,
        .index_exchange, .query_exchange, .partial_exchange, .tp_exchange, .moe_exchange_residual => if (dcp > 1) .dcp_exchange else null,
        else => null,
    };
}

test "step planner declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Boundary);
    std.testing.refAllDecls(SegmentKind);
    std.testing.refAllDecls(Segment);
    std.testing.refAllDecls(Payload);
    std.testing.refAllDecls(bind_qmmf.TaskPayloads);
    std.testing.refAllDecls(Step);
}

test "two dense TP2 layers have no qmmf boundary segments" {
    var graph = try graph_api.build(std.testing.allocator, .{ .rows = 1, .layers = 2, .dcp = 2 });
    defer graph.deinit();
    const payloads = try std.testing.allocator.alloc(Payload, graph.tasks.len);
    defer std.testing.allocator.free(payloads);
    @memset(payloads, .{ .bytes = &.{} });
    var step = try build(std.testing.allocator, graph.params, payloads);
    defer step.deinit();

    try std.testing.expectEqual(@as(usize, 48), step.taskCount());
    try std.testing.expectEqual(@as(usize, 6), step.boundaryCount());
    try std.testing.expectEqual(@as(usize, 13), step.segments.len);
    var assigned = try std.testing.allocator.alloc(bool, step.graph.tasks.len);
    defer std.testing.allocator.free(assigned);
    @memset(assigned, false);
    for (step.segments) |segment| {
        try std.testing.expect(segment.task_count > 0);
        try std.testing.expect(segment.boundary != .qmmf_projection);
        for (segment.first_task..segment.first_task + segment.task_count) |id| {
            try std.testing.expect(!assigned[id]);
            assigned[id] = true;
            if (segment.kind == .boundary) try std.testing.expectEqual(@as(u32, 1), segment.task_count);
        }
    }
    for (assigned) |visited| try std.testing.expect(visited);
    for (step.graph.tasks, 0..) |task, id| {
        var incoming: usize = 0;
        for (step.graph.tasks) |producer| {
            for (step.graph.successors[producer.successor_offset .. producer.successor_offset + producer.successor_count]) |to| {
                if (to == id) incoming += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, task.dependency_count), incoming);
    }
}

test "four dense TP2 rows retain row-local exchanges without qmmf boundaries" {
    var graph = try graph_api.build(std.testing.allocator, .{ .rows = 4, .layers = 2, .dcp = 2 });
    defer graph.deinit();
    const payloads = try std.testing.allocator.alloc(Payload, graph.tasks.len);
    defer std.testing.allocator.free(payloads);
    @memset(payloads, .{ .bytes = &.{} });
    var step = try build(std.testing.allocator, graph.params, payloads);
    defer step.deinit();
    try std.testing.expectEqual(@as(usize, 192), step.taskCount());
    try std.testing.expectEqual(@as(usize, 24), step.boundaryCount());
    try std.testing.expectEqual(@as(usize, 49), step.segments.len);
    for (step.segments) |segment| try std.testing.expect(segment.boundary != .qmmf_projection);
}

test "payload serialization and step summary are analyzed" {
    const Record = extern struct { address: u64, rows: u32 };
    const record: Record = .{ .address = 0x1234, .rows = 4 };
    const payload = Payload.from(&record);
    try std.testing.expectEqual(@sizeOf(Record), payload.bytes.len);

    var graph = try graph_api.build(std.testing.allocator, .{ .rows = 1, .layers = 2, .dcp = 1 });
    defer graph.deinit();
    const payloads = try std.testing.allocator.alloc(Payload, graph.tasks.len);
    defer std.testing.allocator.free(payloads);
    @memset(payloads, .{ .bytes = &.{} });
    var step = try build(std.testing.allocator, graph.params, payloads);
    defer step.deinit();
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try step.dump(&output.writer);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "boundary qmmf_projection") == null);
}

test "qmmf binder records flow into matching graph task payloads" {
    const plan = try bind_qmmf.bind(.{ .rows = 4, .n = 8192, .k = 2048, .npad = 8192, .ldx = 2048, .mode = .fp8b, .f32 = false }, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000);
    const payloads = qmmfPayloadPair(&plan);
    try std.testing.expectEqual(@sizeOf(bind_qmmf.PartialParams), payloads[0].bytes.len);
    try std.testing.expectEqual(@sizeOf(bind_qmmf.ReduceParams), payloads[1].bytes.len);
    const partial = std.mem.bytesAsValue(bind_qmmf.PartialParams, payloads[0].bytes);
    const reduce = std.mem.bytesAsValue(bind_qmmf.ReduceParams, payloads[1].bytes);
    try std.testing.expectEqual(@as(u64, 0x5000), partial.partial);
    try std.testing.expectEqual(partial.partial, reduce.partial);
    try std.testing.expectEqual(@as(u64, 0x4000), reduce.out);
    try std.testing.expectEqual(partial.sk, reduce.sk);
}
