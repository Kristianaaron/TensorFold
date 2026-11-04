// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Measure GB10 system-memory overhead for committed CUDA allocations backed by cuMemAlloc and CUDA VMM.
const std = @import("std");
const cuda = @import("cuda");
const uma = @import("uma");

const gib_bytes: usize = 1 << 30;
const Usage = enum { alloc, vmm, both };

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5 or !std.mem.eql(u8, args[1], "--gib") or !std.mem.eql(u8, args[3], "--mode")) {
        std.debug.print("usage: tf-uma-alloc-probe --gib N --mode alloc|vmm|both\n", .{});
        return 2;
    }
    const gib = std.fmt.parseInt(usize, args[2], 10) catch return 2;
    if (gib == 0 or gib > std.math.maxInt(usize) / gib_bytes) return 2;
    const usage: Usage = if (std.mem.eql(u8, args[4], "alloc")) .alloc else if (std.mem.eql(u8, args[4], "vmm")) .vmm else if (std.mem.eql(u8, args[4], "both")) .both else return 2;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const bytes: usize = gib * gib_bytes;
    switch (usage) {
        .alloc => try measure(&driver, &ctx, init.io, bytes, .alloc),
        .vmm => try measure(&driver, &ctx, init.io, bytes, .vmm),
        .both => {
            try measure(&driver, &ctx, init.io, bytes, .alloc);
            try settleMemAvailable(std.heap.page_allocator, init.io);
            try measure(&driver, &ctx, init.io, bytes, .vmm);
        },
    }
    return 0;
}

fn settleMemAvailable(gpa: std.mem.Allocator, io: std.Io) !void {
    const stable_delta_bytes: u64 = 64 * 1024 * 1024;
    var previous = try uma.readMeminfo(gpa, io);
    var stable_samples: usize = 0;
    for (0..20) |_| {
        try std.Io.sleep(io, .fromMilliseconds(250), .awake);
        const current = try uma.readMeminfo(gpa, io);
        const delta = if (current.available >= previous.available) current.available - previous.available else previous.available - current.available;
        if (delta <= stable_delta_bytes) {
            stable_samples += 1;
            if (stable_samples == 3) return;
        } else {
            stable_samples = 0;
        }
        previous = current;
    }
}

fn measure(driver: *const cuda.Driver, ctx: *const cuda.Context, io: std.Io, bytes: usize, usage: Usage) !void {
    const before = try uma.readMeminfoDetail(std.heap.page_allocator, io);
    const size_gib = bytes / gib_bytes;
    if (usage == .alloc) {
        var buffer = try cuda.DeviceBuffer.alloc(driver, bytes);
        defer buffer.free();
        try driver.check(driver.api.cuMemsetD8_v2(buffer.ptr, 0, bytes), "cuMemsetD8");
        try ctx.synchronize();
        const after = try uma.readMeminfoDetail(std.heap.page_allocator, io);
        report(usage, size_gib, before, after);
    } else {
        var vmm = try cuda.arena.Vmm.open(driver, ctx.device, bytes);
        defer vmm.close();
        const be = vmm.backend();
        const chunk_bytes: usize = 1 << 30;
        var mapped: std.ArrayList([2]u64) = .empty;
        defer {
            for (mapped.items) |chunk| be.release(be.ctx, chunk[0], @intCast(chunk[1]));
            mapped.deinit(std.heap.page_allocator);
        }
        var left = bytes;
        while (left > 0) {
            const len: usize = @min(chunk_bytes, left);
            const address = be.chunk(be.ctx, len) orelse return error.VmmChunkFailed;
            try mapped.append(std.heap.page_allocator, .{ address, len });
            try driver.check(driver.api.cuMemsetD8_v2(address, 0, len), "cuMemsetD8 VMM");
            left -= len;
        }
        try ctx.synchronize();
        const after = try uma.readMeminfoDetail(std.heap.page_allocator, io);
        report(usage, size_gib, before, after);
    }
}

fn report(usage: Usage, size_gib: usize, before: uma.MemDetail, after: uma.MemDetail) void {
    const available_drop = @as(f64, @floatFromInt(before.fields[1])) - @as(f64, @floatFromInt(after.fields[1]));
    const drop_gib = available_drop / @as(f64, @floatFromInt(gib_bytes));
    std.debug.print("{s} {d} GiB: MemAvailable drop {d:.3} GiB, overhead (D - N) {d:.3} GiB = {d:.3} %\n", .{ @tagName(usage), size_gib, drop_gib, drop_gib - @as(f64, @floatFromInt(size_gib)), (drop_gib - @as(f64, @floatFromInt(size_gib))) / @as(f64, @floatFromInt(size_gib)) * 100.0 });
    for (uma.mem_detail_keys, 0..) |key, i| {
        const delta = @as(i128, before.fields[i]) - @as(i128, after.fields[i]);
        if (delta == 0) continue;
        if (i == 17) {
            std.debug.print("  {s}: {d} pages\n", .{ key, delta });
            continue;
        }
        std.debug.print("  {s}: {d:.3} GiB\n", .{ key, @as(f64, @floatFromInt(delta)) / @as(f64, @floatFromInt(gib_bytes)) });
    }
}

test {
    std.testing.refAllDecls(@This());
}
