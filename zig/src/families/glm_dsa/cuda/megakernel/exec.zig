// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host task payload ownership and segmented execution for GLM-5.3 decode.
//! Payload values are supplied by the forward call-site binders; boundary callbacks
//! are where Forward performs its existing collectives and cluster-only launches.
const std = @import("std");
const graph_api = @import("graph.zig");
const step_api = @import("step.zig");
const bind_qmmf = @import("bind_qmmf.zig");

pub const BoundaryRunner = struct {
    ctx: *anyopaque,
    run_fn: *const fn (ctx: *anyopaque, task: graph_api.Task, payload: step_api.Payload) anyerror!void,

    pub fn run(r: BoundaryRunner, task: graph_api.Task, payload: step_api.Payload) !void {
        try r.run_fn(r.ctx, task, payload);
    }
};

pub const PersistentRunner = struct {
    ctx: *anyopaque,
    run_fn: *const fn (ctx: *anyopaque, step: *const step_api.Step, segment: step_api.Segment) anyerror!void,

    pub fn run(r: PersistentRunner, step: *const step_api.Step, segment: step_api.Segment) !void {
        try r.run_fn(r.ctx, step, segment);
    }
};

/// Owns serialized binder output until launch.Uploaded has copied it to the device.
/// Task ids and their payloads follow graph.build's stable insertion order.
pub const PayloadBuilder = struct {
    allocator: std.mem.Allocator,
    graph: graph_api.Graph,
    payloads: []step_api.Payload,
    assigned: []bool,
    offsets: []usize,
    lengths: []usize,
    arena: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, params: graph_api.Params) !PayloadBuilder {
        var graph = try graph_api.build(allocator, params);
        errdefer graph.deinit();
        const payloads = try allocator.alloc(step_api.Payload, graph.tasks.len);
        errdefer allocator.free(payloads);
        @memset(payloads, .{ .bytes = &.{} });
        const assigned = try allocator.alloc(bool, graph.tasks.len);
        errdefer allocator.free(assigned);
        @memset(assigned, false);
        const offsets = try allocator.alloc(usize, graph.tasks.len);
        errdefer allocator.free(offsets);
        const lengths = try allocator.alloc(usize, graph.tasks.len);
        errdefer allocator.free(lengths);
        @memset(offsets, 0);
        @memset(lengths, 0);
        return .{ .allocator = allocator, .graph = graph, .payloads = payloads, .assigned = assigned, .offsets = offsets, .lengths = lengths };
    }

    pub fn deinit(b: *PayloadBuilder) void {
        b.arena.deinit(b.allocator);
        b.allocator.free(b.assigned);
        b.allocator.free(b.offsets);
        b.allocator.free(b.lengths);
        b.allocator.free(b.payloads);
        b.graph.deinit();
        b.* = undefined;
    }

    /// Copy one binder record into stable storage. Empty payloads are allowed for explicit
    /// no-op boundary records, but each graph task must be assigned exactly once.
    pub fn bind(b: *PayloadBuilder, task_id: usize, value: anytype) !void {
        return b.bindBytes(task_id, std.mem.asBytes(value));
    }

    pub fn bindBytes(b: *PayloadBuilder, task_id: usize, bytes: []const u8) !void {
        if (task_id >= b.graph.tasks.len) return error.TaskOutOfRange;
        if (b.assigned[task_id]) return error.TaskAlreadyBound;
        const aligned = std.mem.alignForward(usize, b.arena.items.len, 16);
        try b.arena.appendNTimes(b.allocator, 0, aligned - b.arena.items.len);
        const start = b.arena.items.len;
        try b.arena.appendSlice(b.allocator, bytes);
        b.offsets[task_id] = start;
        b.lengths[task_id] = bytes.len;
        b.assigned[task_id] = true;
    }

    /// Bind the adjacent qmmf partial/reduce pair emitted by graph.Builder.addQmmf.
    pub fn bindQmmfPair(b: *PayloadBuilder, partial_task: usize, plan: bind_qmmf.Plan) !void {
        if (partial_task + 1 >= b.graph.tasks.len) return error.TaskOutOfRange;
        const pair = step_api.qmmfPayloadPair(&plan);
        if (b.graph.tasks[partial_task].kind != .qmmf_partial or b.graph.tasks[partial_task + 1].kind != .qmmf_reduce)
            return error.NotQmmfPair;
        try b.bindBytes(partial_task, pair[0].bytes);
        try b.bindBytes(partial_task + 1, pair[1].bytes);
    }

    pub fn finish(b: *PayloadBuilder) !step_api.Step {
        for (b.assigned) |bound| if (!bound) return error.UnboundTask;
        for (b.payloads, b.offsets, b.lengths, 0..) |*payload, offset, len, id| {
            _ = id;
            payload.bytes = b.arena.items[offset..][0..len];
        }
        return step_api.build(b.allocator, b.graph.params, b.payloads);
    }
};

/// Run the planner's topological segments in order. Forward owns the boundary implementation:
/// DCP query/partial/index exchanges use its Collective, and index top-k uses its cluster wrapper.
pub fn execute(s: *const step_api.Step, persistent: PersistentRunner, boundary: BoundaryRunner) !void {
    for (s.segments) |segment| {
        if (segment.kind == .persistent) {
            try persistent.run(s, segment);
        } else {
            const id: usize = segment.first_task;
            try boundary.run(s.graph.tasks[id], s.payloads[id]);
        }
    }
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(BoundaryRunner);
    std.testing.refAllDecls(PersistentRunner);
    std.testing.refAllDecls(PayloadBuilder);
}

test "payload builder owns aligned binder bytes and requires complete assignment" {
    var builder = try PayloadBuilder.init(std.testing.allocator, .{ .rows = 1, .layers = 1, .dcp = 1 });
    defer builder.deinit();
    const qmmf = try bind_qmmf.bind(.{ .rows = 1, .n = 2048, .k = 6144, .npad = 2048, .ldx = 6144, .mode = .fp8b, .f32 = false }, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000);
    var partial: ?usize = null;
    for (builder.graph.tasks, 0..) |task, id| if (task.kind == .qmmf_partial) {
        partial = id;
        break;
    };
    try std.testing.expect(partial != null);
    try builder.bindQmmfPair(partial.?, qmmf);
    try std.testing.expectEqual(@sizeOf(bind_qmmf.PartialParams), builder.lengths[partial.?]);
    try std.testing.expectEqual(@sizeOf(bind_qmmf.ReduceParams), builder.lengths[partial.? + 1]);
    const marker: u8 = 0x5a;
    for (builder.assigned, 0..) |bound, id| if (!bound) try builder.bind(id, &marker);
    var built = try builder.finish();
    defer built.deinit();
    const partial_record = std.mem.bytesAsValue(bind_qmmf.PartialParams, built.payloads[partial.?].bytes);
    try std.testing.expectEqual(@as(u64, 0x5000), partial_record.partial);
    try std.testing.expectEqual(@as(usize, @sizeOf(bind_qmmf.ReduceParams)), built.payloads[partial.? + 1].bytes.len);
}

test "execution invokes persistent and boundary runners in planned order" {
    const graph = @import("graph.zig");
    var g = try graph.build(std.testing.allocator, .{ .rows = 1, .layers = 2, .dcp = 2 });
    defer g.deinit();
    const payloads = try std.testing.allocator.alloc(step_api.Payload, g.tasks.len);
    defer std.testing.allocator.free(payloads);
    @memset(payloads, .{ .bytes = &.{} });
    var planned = try step_api.build(std.testing.allocator, g.params, payloads);
    defer planned.deinit();
    const Counts = struct { persistent: usize = 0, boundary: usize = 0 };
    const Runners = struct {
        fn persistent(ctx: *anyopaque, _: *const step_api.Step, _: step_api.Segment) anyerror!void {
            const counts: *Counts = @ptrCast(@alignCast(ctx));
            counts.persistent += 1;
        }
        fn boundary(ctx: *anyopaque, _: graph_api.Task, _: step_api.Payload) anyerror!void {
            const counts: *Counts = @ptrCast(@alignCast(ctx));
            counts.boundary += 1;
        }
    };
    var counts: Counts = .{};
    try execute(&planned, .{ .ctx = &counts, .run_fn = Runners.persistent }, .{ .ctx = &counts, .run_fn = Runners.boundary });
    try std.testing.expectEqual(@as(usize, planned.boundaryCount()), counts.boundary);
    try std.testing.expectEqual(planned.segments.len - planned.boundaryCount(), counts.persistent);
}
