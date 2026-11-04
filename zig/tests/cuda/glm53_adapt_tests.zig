// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! CUDA VMM double-map smoke test used as the first Spark gate for adaptive expert rows.
const std = @import("std");
const cuda = @import("cuda");

test "adaptive expert VMM declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    runDoubleMap(gpu) catch |err| {
        std.debug.print("FAIL glm53-adapt VMM double mapping: {s}\n", .{@errorName(err)});
        return err;
    };
    // Routed parity (adaptive vs static 2048/4096 expert rows) runs end to end in the Spark lane's bring-up A/B.
}

fn runDoubleMap(gpu: anytype) !void {
    const gran: usize = 2 << 20;
    var vmm = try cuda.arena.Vmm.open(gpu.d, gpu.ctx.device, 6 * gran);
    defer vmm.close();
    const base = vmm.va;
    const borrowed = base + 3 * gran;
    defer vmm.unmapRange(base, 3 * gran);
    defer vmm.unmapRange(borrowed, 3 * gran);
    for (0..3) |i| {
        const h = try vmm.createHandle(gran);
        defer vmm.releaseHandle(h);
        const offset: u64 = @intCast(i * gran);
        try vmm.mapHandle(base + offset, gran, h);
        try vmm.mapHandle(borrowed + offset, gran, h);
    }
    try vmm.grantAccess(base, 3 * gran);
    try vmm.grantAccess(borrowed, 3 * gran);
    const n = 3 * gran;
    const pattern = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(pattern);
    const readback = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(readback);
    @memset(pattern, 0x5a);
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(borrowed, pattern.ptr, n), "VMM pattern into borrowed view");
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(readback.ptr, base, n), "VMM read through KV view");
    try std.testing.expectEqualSlices(u8, pattern, readback);
    @memset(pattern, 0x3c);
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(base, pattern.ptr, n), "VMM pattern into KV view");
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(readback.ptr, borrowed, n), "VMM read through expert view");
    try std.testing.expectEqualSlices(u8, pattern, readback);
    try gpu.d.check(gpu.d.api.cuMemsetD8_v2(base, 0xa5, n), "VMM memset through KV view");
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(readback.ptr, borrowed, n), "VMM read through expert view");
    @memset(pattern, 0xa5);
    try std.testing.expectEqualSlices(u8, pattern, readback);
    std.debug.print("PASS glm53-adapt VMM double mapping bytes={d} granularity={d}; writes and full-view memset are coherent\n", .{ n, gran });
}
