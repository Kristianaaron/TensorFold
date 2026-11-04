// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// TRACE:
// L1 load-gap emission: zig/src/families/glm_dsa/cuda/native.zig:675, LoadSteps.markExpected.
//    The production weight comparison follows ModelWeights.loadConfig at native.zig:1005.
// L2 weight-switch parsing: weights.zig injects its sentinel-safe processEnvLookup into this std-only planner.
// L3 weight device allocation: zig/src/families/glm_dsa/cuda/weights.zig:1081 (VMM) and :1096 (legacy),
//    ModelWeights.loadConfig allocates one buffer per non-empty weight class; legacy reaches
//    cuMemAlloc_v2 at zig/src/cuda/memory.zig:146. Tensors are carved from class arenas.
// Per-rank options are constructed at native.zig:1005 in the same openInner path with p.rank passed
//    to loadConfig; both ranks use that same construction path.
// CUDA VMM driver bindings exist in zig/src/cuda/arena.zig:197-269; the VMM path splits/maps at 330-365.
// The VMM path maps all non-empty class arenas as one contiguous span. The loader allocates one
// buffer per class; tensor placement and upload order stay within those class arenas.

const std = @import("std");

pub const Mode = enum { legacy, vmm };
pub const Plan = struct {
    mode: Mode,
    offsets: []u64,
    table_bytes: u64,
    mapped_bytes: u64,
    granularity: u64,
};

/// Parse the opt-in VMM switch. The caller injects the environment lookup for host tests.
pub const Lookup = *const fn (name: [:0]const u8) ?[]const u8;

pub fn weightsVmmFromLookup(lookup: Lookup) bool {
    const value = lookup("TF_GLM53_WEIGHTS_VMM") orelse return false;
    return std.mem.eql(u8, value, "1") or std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "on");
}

fn roundUp(value: u64, multiple: u64) error{InvalidGranularity, Overflow}!u64 {
    if (multiple == 0) return error.InvalidGranularity;
    const rem = value % multiple;
    if (rem == 0) return value;
    return std.math.add(u64, value, multiple - rem) catch error.Overflow;
}

/// Build a stable contiguous VMM layout or model the per-allocation legacy rounding.
pub fn build(alloc: std.mem.Allocator, lookup: Lookup, sizes: []const u64, align_bytes: u64, granularity: u64) !Plan {
    if (granularity == 0) return error.InvalidGranularity;
    const offsets = try alloc.alloc(u64, sizes.len);
    errdefer alloc.free(offsets);

    var table_bytes: u64 = 0;
    var mapped_bytes: u64 = 0;
    if (!weightsVmmFromLookup(lookup)) {
        for (sizes, 0..) |size, i| {
            offsets[i] = 0;
            table_bytes = try std.math.add(u64, table_bytes, size);
            mapped_bytes = try std.math.add(u64, mapped_bytes, try roundUp(size, granularity));
        }
        return .{ .mode = .legacy, .offsets = offsets, .table_bytes = table_bytes, .mapped_bytes = mapped_bytes, .granularity = granularity };
    }

    const tensor_alignment = @max(align_bytes, 256);
    var end: u64 = 0;
    for (sizes, 0..) |size, i| {
        const offset = try roundUp(end, tensor_alignment);
        offsets[i] = offset;
        end = try std.math.add(u64, offset, size);
        table_bytes = try std.math.add(u64, table_bytes, size);
    }
    mapped_bytes = try roundUp(end, granularity);
    return .{ .mode = .vmm, .offsets = offsets, .table_bytes = table_bytes, .mapped_bytes = mapped_bytes, .granularity = granularity };
}

var test_values: ?*std.StringHashMap([]const u8) = null;
fn fakeLookup(name: [:0]const u8) ?[]const u8 {
    const values = test_values orelse return null;
    return values.get(name);
}

fn sizesForTest(alloc: std.mem.Allocator) ![]u64 {
    const sizes = try alloc.alloc(u64, 2000);
    for (sizes, 0..) |*size, i| {
        size.* = switch (i % 4) {
            0 => 1,
            1 => 4097,
            2 => (3 << 20) + 7,
            else => 512 << 20,
        };
    }
    return sizes;
}

test "VMM weight plan is tight, aligned and deterministic for both ranks" {
    const alloc = std.testing.allocator;
    var values = std.StringHashMap([]const u8).init(alloc);
    defer values.deinit();
    try values.put("TF_GLM53_WEIGHTS_VMM", "1");
    test_values = &values;
    defer test_values = null;
    const sizes = try sizesForTest(alloc);
    defer alloc.free(sizes);
    const first = try build(alloc, fakeLookup, sizes, 256, 2 << 20);
    defer alloc.free(first.offsets);
    const second = try build(alloc, fakeLookup, sizes, 256, 2 << 20);
    defer alloc.free(second.offsets);
    try std.testing.expectEqual(Mode.vmm, first.mode);
    try std.testing.expect(first.table_bytes <= first.mapped_bytes);
    try std.testing.expect(first.mapped_bytes - first.table_bytes < first.granularity + @as(u64, @intCast(sizes.len)) * 256);
    const last = first.offsets.len - 1;
    const aligned_end = first.offsets[last] + sizes[last];
    try std.testing.expect(first.mapped_bytes >= aligned_end);
    try std.testing.expect(first.mapped_bytes - aligned_end < first.granularity);
    try std.testing.expectEqual(try roundUp(first.offsets[sizes.len - 1] + sizes[sizes.len - 1], first.granularity), first.mapped_bytes);
    for (first.offsets, 0..) |offset, i| {
        try std.testing.expectEqual(@as(u64, 0), offset % 256);
        if (i != 0) try std.testing.expect(offset >= first.offsets[i - 1] + sizes[i - 1]);
    }
    try std.testing.expectEqual(first.mode, second.mode);
    try std.testing.expectEqual(first.table_bytes, second.table_bytes);
    try std.testing.expectEqual(first.mapped_bytes, second.mapped_bytes);
    try std.testing.expectEqual(first.granularity, second.granularity);
    try std.testing.expectEqualSlices(u64, first.offsets, second.offsets);
}

test "unset, zero and empty weight VMM values select legacy allocation" {
    const alloc = std.testing.allocator;
    var values = std.StringHashMap([]const u8).init(alloc);
    defer values.deinit();
    test_values = &values;
    defer test_values = null;
    const sizes = [_]u64{ 1, 4097, (3 << 20) + 7, 512 << 20 };
    const unset = try build(alloc, fakeLookup, &sizes, 256, 2 << 20);
    defer alloc.free(unset.offsets);
    try std.testing.expectEqual(Mode.legacy, unset.mode);
    try values.put("TF_GLM53_WEIGHTS_VMM", "0");
    const zero = try build(alloc, fakeLookup, &sizes, 256, 2 << 20);
    defer alloc.free(zero.offsets);
    try std.testing.expectEqual(Mode.legacy, zero.mode);
    try values.put("TF_GLM53_WEIGHTS_VMM", "");
    const empty = try build(alloc, fakeLookup, &sizes, 256, 2 << 20);
    defer alloc.free(empty.offsets);
    try std.testing.expectEqual(Mode.legacy, empty.mode);
    try std.testing.expectEqual(@as(u64, 2 << 20) + (2 << 20) + (4 << 20) + (512 << 20), unset.mapped_bytes);
}

test "VMM mapped span is strictly smaller than the legacy per-allocation model" {
    const alloc = std.testing.allocator;
    var values = std.StringHashMap([]const u8).init(alloc);
    defer values.deinit();
    test_values = &values;
    defer test_values = null;
    const sizes = [_]u64{ 1, 4097, (3 << 20) + 7, 512 << 20, 1, 4097, (3 << 20) + 7, 512 << 20 };
    const legacy = try build(alloc, fakeLookup, &sizes, 256, 2 << 20);
    defer alloc.free(legacy.offsets);
    try values.put("TF_GLM53_WEIGHTS_VMM", "on");
    const vmm = try build(alloc, fakeLookup, &sizes, 256, 2 << 20);
    defer alloc.free(vmm.offsets);
    try std.testing.expectEqual(Mode.legacy, legacy.mode);
    try std.testing.expectEqual(Mode.vmm, vmm.mode);
    try std.testing.expect(vmm.mapped_bytes < legacy.mapped_bytes);

    const many_sizes = try sizesForTest(alloc);
    defer alloc.free(many_sizes);
    try values.put("TF_GLM53_WEIGHTS_VMM", "0");
    const many_legacy = try build(alloc, fakeLookup, many_sizes, 256, 2 << 20);
    defer alloc.free(many_legacy.offsets);
    try values.put("TF_GLM53_WEIGHTS_VMM", "1");
    const many_vmm = try build(alloc, fakeLookup, many_sizes, 256, 2 << 20);
    defer alloc.free(many_vmm.offsets);
    try std.testing.expect(many_vmm.mapped_bytes < many_legacy.mapped_bytes);
}

test "weight VMM switch accepts only the specified affirmative spellings" {
    const alloc = std.testing.allocator;
    var values = std.StringHashMap([]const u8).init(alloc);
    defer values.deinit();
    test_values = &values;
    defer test_values = null;
    try values.put("TF_GLM53_WEIGHTS_VMM", "true");
    try std.testing.expect(weightsVmmFromLookup(fakeLookup));
    try values.put("TF_GLM53_WEIGHTS_VMM", "on");
    try std.testing.expect(weightsVmmFromLookup(fakeLookup));
    try values.put("TF_GLM53_WEIGHTS_VMM", "TRUE");
    try std.testing.expect(!weightsVmmFromLookup(fakeLookup));
}
