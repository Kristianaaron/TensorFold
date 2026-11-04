// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Fixed-shape-plan cuBLASLt BF16 prefill projections, selected independently of chunk row count.
const std = @import("std");
const cuda = @import("cuda");

const PlanKey = struct { n: u64, k: u64, out: u64 };
const workspace_bytes = 64 << 20;
const plan_rows = 2048;

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: cuda.cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged(PlanKey, cuda.cublaslt.Linear) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        var g: Gemms = .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
        errdefer g.deinit();
        // Build-time contract: the plan row count is always 2048; every GLM shape is selected before prefill starts.
        const shapes = [_]struct { n: u64, k: u64, out: cuda.cublaslt.DataType }{
            .{ .n = 2048, .k = 6144, .out = .bf16 }, .{ .n = 8192, .k = 2048, .out = .bf16 },
            .{ .n = 576, .k = 6144, .out = .bf16 },  .{ .n = 6144, .k = 8192, .out = .f32 },
            .{ .n = 4096, .k = 2048, .out = .bf16 }, .{ .n = 128, .k = 6144, .out = .bf16 },
            .{ .n = 32, .k = 6144, .out = .bf16 },   .{ .n = 2048, .k = 6144, .out = .bf16 },
            .{ .n = 6144, .k = 1024, .out = .f32 },  .{ .n = 12288, .k = 6144, .out = .bf16 },
            .{ .n = 6144, .k = 6144, .out = .f32 },
        };
        for (shapes) |shape| _ = try g.plan(shape.n, shape.k, shape.out);
        return g;
    }

    pub fn deinit(g: *Gemms) void {
        var plans = g.plans.valueIterator();
        while (plans.next()) |linear| linear.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
        g.* = undefined;
    }

    /// x [M,K], row-major BF16 w [N,K], output BF16 or FP32. Plan geometry is fixed at M=2048;
    /// runRows only updates the cuBLASLt matrix extents and reuses the N/K plan for every chunk split.
    pub fn run(g: *Gemms, stream: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, dtype: cuda.cublaslt.DataType) !void {
        if (rows == 0 or rows > 8192 or n == 0 or k == 0) return error.BadPrefillGemmShape;
        const linear = try g.plan(n, k, dtype);
        try linear.runRows(rows, x, w, out, g.workspace.ptr, workspace_bytes, stream.handle);
    }

    fn plan(g: *Gemms, n: u64, k: u64, dtype: cuda.cublaslt.DataType) !*cuda.cublaslt.Linear {
        const key: PlanKey = .{ .n = n, .k = k, .out = @intCast(@backingInt(dtype)) };
        const slot = try g.plans.getOrPut(g.gpa, key);
        if (!slot.found_existing) slot.value_ptr.* = cuda.cublaslt.Linear.init(&g.lt, plan_rows, n, k, dtype, workspace_bytes) catch |err| {
            _ = g.plans.remove(key);
            return err;
        };
        return slot.value_ptr;
    }
};

test "shape plan key deliberately excludes M and distinguishes output type" {
    const a: PlanKey = .{ .n = 6144, .k = 6144, .out = @intCast(@backingInt(cuda.cublaslt.DataType.bf16)) };
    const b: PlanKey = .{ .n = 6144, .k = 6144, .out = @intCast(@backingInt(cuda.cublaslt.DataType.bf16)) };
    const c: PlanKey = .{ .n = 6144, .k = 6144, .out = @intCast(@backingInt(cuda.cublaslt.DataType.f32)) };
    try std.testing.expect(std.meta.eql(a, b));
    try std.testing.expect(!std.meta.eql(a, c));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Gemms);
}
