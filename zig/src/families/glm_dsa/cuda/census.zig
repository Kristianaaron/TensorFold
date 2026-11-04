// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Count checkpoint tensor payload bytes by storage class from a HF index and safetensors headers.
const std = @import("std");

pub const Census = struct {
    routed: u64 = 0,
    backbone_fp8: u64 = 0,
    bf16_f32: u64 = 0,

    pub fn total(c: Census) u64 {
        return c.routed + c.backbone_fp8 + c.bf16_f32;
    }
};

pub fn readModelDir(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Census {
    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const index_bytes = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(64 * 1024 * 1024));
    defer gpa.free(index_bytes);
    const index = try std.json.parseFromSlice(std.json.Value, gpa, index_bytes, .{});
    defer index.deinit();
    const weight_map = index.value.object.get("weight_map") orelse return error.BadIndex;
    if (weight_map != .object) return error.BadIndex;

    var shards = std.StringHashMap(void).init(gpa);
    defer shards.deinit();
    var iter = weight_map.object.iterator();
    while (iter.next()) |entry| {
        if (entry.value_ptr.* != .string) return error.BadIndex;
        try shards.put(entry.value_ptr.string, {});
    }

    var out: Census = .{};
    var seen = std.StringHashMapUnmanaged(void){};
    defer {
        var seen_keys = seen.keyIterator();
        while (seen_keys.next()) |key| gpa.free(key.*);
        seen.deinit(gpa);
    }
    var shard_iter = shards.keyIterator();
    while (shard_iter.next()) |shard_name| {
        const shard_path = try std.fs.path.join(gpa, &.{ model_dir, shard_name.* });
        defer gpa.free(shard_path);
        const header_bytes = try readHeader(gpa, io, shard_path);
        defer gpa.free(header_bytes);
        const header = try std.json.parseFromSlice(std.json.Value, gpa, header_bytes, .{});
        defer header.deinit();
        var tensor_iter = header.value.object.iterator();
        while (tensor_iter.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            const expected_shard = weight_map.object.get(entry.key_ptr.*) orelse return error.UnindexedTensor;
            if (expected_shard != .string or !std.mem.eql(u8, expected_shard.string, shard_name.*)) return error.WrongShard;
            if (seen.contains(entry.key_ptr.*)) return error.DuplicateTensor;
            const stable_name = try gpa.dupe(u8, entry.key_ptr.*);
            seen.put(gpa, stable_name, {}) catch |err| {
                gpa.free(stable_name);
                return err;
            };
            if (entry.value_ptr.* != .object) return error.BadSafetensors;
            const tensor = entry.value_ptr.object;
            const dtype_value = tensor.get("dtype") orelse return error.BadSafetensors;
            const offsets = tensor.get("data_offsets") orelse return error.BadSafetensors;
            if (dtype_value != .string or offsets != .array or offsets.array.items.len != 2) return error.BadSafetensors;
            const begin = try jsonU64(offsets.array.items[0]);
            const end = try jsonU64(offsets.array.items[1]);
            if (end < begin) return error.BadSafetensors;
            const bytes = end - begin;
            if (isRouted(entry.key_ptr.*)) {
                out.routed = try std.math.add(u64, out.routed, bytes);
            } else if (isFp8(dtype_value.string) or isFp8Scale(entry.key_ptr.*)) {
                out.backbone_fp8 = try std.math.add(u64, out.backbone_fp8, bytes);
            } else if (std.mem.eql(u8, dtype_value.string, "BF16") or std.mem.eql(u8, dtype_value.string, "F32")) {
                out.bf16_f32 = try std.math.add(u64, out.bf16_f32, bytes);
            } else return error.UnsupportedDtype;
        }
    }

    // The index must not claim tensors absent from shard headers.
    var index_iter = weight_map.object.iterator();
    while (index_iter.next()) |entry| if (!seen.contains(entry.key_ptr.*)) return error.MissingTensor;
    return out;
}

fn readHeader(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var len_bytes: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &len_bytes, 0) != len_bytes.len) return error.BadSafetensors;
    const header_len = std.mem.readInt(u64, &len_bytes, .little);
    if (header_len > 64 * 1024 * 1024) return error.BadSafetensors;
    const header = try gpa.alloc(u8, @intCast(header_len));
    errdefer gpa.free(header);
    if (try file.readPositionalAll(io, header, 8) != header.len) return error.BadSafetensors;
    return header;
}

fn jsonU64(value: std.json.Value) !u64 {
    return switch (value) {
        .integer => |i| if (i >= 0) @intCast(i) else error.BadSafetensors,
        else => error.BadSafetensors,
    };
}

fn isRouted(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".experts.") != null or std.mem.indexOf(u8, name, ".experts_") != null;
}

fn isFp8(dtype: []const u8) bool {
    return std.mem.startsWith(u8, dtype, "F8_") or std.mem.startsWith(u8, dtype, "E4M3");
}

fn isFp8Scale(name: []const u8) bool {
    return std.mem.indexOf(u8, name, "scale_inv") != null;
}

pub fn write(out: anytype, c: Census) !void {
    try out.print("routed: {d:.2} GiB ({d} bytes)\nFP8 backbone: {d:.2} GiB ({d} bytes)\nBF16+F32: {d:.2} GiB ({d} bytes)\ntotal: {d:.2} GiB ({d} bytes)\n", .{
        gib(c.routed), c.routed, gib(c.backbone_fp8), c.backbone_fp8, gib(c.bf16_f32), c.bf16_f32, gib(c.total()), c.total(),
    });
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}
