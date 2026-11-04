// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 dump tensors and the Python-compatible safetensors wire format (K9).
//! Dump row fields follow bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/forward.py (Apache-2.0).
//! Changes: Zig safetensors encoding, GLM dump keys, and sparse gate rows.
const std = @import("std");

pub const Tensor = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const usize,
    bytes: []const u8,
};

fn escaped(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    return try std.json.Stringify.valueAlloc(allocator, s, .{});
}

/// Build one safetensors file. Names must be unique; shape and dtype byte counts are checked.
pub fn encode(allocator: std.mem.Allocator, tensors: []const Tensor) ![]u8 {
    return encodeWithMetadata(allocator, tensors, null, null);
}

/// Encode a safetensors file with optional Python-compatible `__metadata__` string entries.
pub fn encodeWithMetadata(allocator: std.mem.Allocator, tensors: []const Tensor, metadata_key: ?[]const u8, metadata_value: ?[]const u8) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
    try header.append(allocator, '{');
    if (metadata_key) |mk| {
        const mv = metadata_value orelse return error.InvalidMetadata;
        try header.appendSlice(allocator, "\"__metadata__\":{");
        const ek = try escaped(allocator, mk);
        defer allocator.free(ek);
        const ev = try escaped(allocator, mv);
        defer allocator.free(ev);
        try header.appendSlice(allocator, ek);
        try header.append(allocator, ':');
        try header.appendSlice(allocator, ev);
        try header.append(allocator, '}');
        if (tensors.len != 0) try header.append(allocator, ',');
    }
    for (tensors, 0..) |tensor, i| {
        const dtype_size = dtypeSize(tensor.dtype) orelse return error.UnsupportedDType;
        var count: usize = 1;
        for (tensor.shape) |dim| count = try std.math.mul(usize, count, dim);
        if (try std.math.mul(usize, count, dtype_size) != tensor.bytes.len) return error.TensorByteSize;
        for (tensors[0..i]) |prev| if (std.mem.eql(u8, tensor.name, prev.name)) return error.DuplicateTensor;
        const encoded_key = try escaped(allocator, tensor.name);
        defer allocator.free(encoded_key);
        try header.appendSlice(allocator, encoded_key);
        try header.print(allocator, ":{{\"dtype\":\"{s}\",\"shape\":[", .{tensor.dtype});
        for (tensor.shape, 0..) |dim, j| {
            if (j != 0) try header.append(allocator, ',');
            try header.print(allocator, "{d}", .{dim});
        }
        try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ data.items.len, data.items.len + tensor.bytes.len });
        if (i + 1 != tensors.len) try header.append(allocator, ',');
        try data.appendSlice(allocator, tensor.bytes);
    }
    try header.append(allocator, '}');
    const padding = (8 - (header.items.len % 8)) % 8;
    try header.appendNTimes(allocator, ' ', padding);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.items.len, .little);
    var result: std.ArrayList(u8) = .empty;
    defer result.deinit(allocator);
    try result.appendSlice(allocator, &prefix);
    try result.appendSlice(allocator, header.items);
    try result.appendSlice(allocator, data.items);
    return try result.toOwnedSlice(allocator);
}

fn dtypeSize(dtype: []const u8) ?usize {
    if (std.mem.eql(u8, dtype, "BOOL") or std.mem.eql(u8, dtype, "U8") or std.mem.eql(u8, dtype, "I8") or std.mem.eql(u8, dtype, "F8_E4M3")) return 1;
    if (std.mem.eql(u8, dtype, "U16") or std.mem.eql(u8, dtype, "I16") or std.mem.eql(u8, dtype, "F16") or std.mem.eql(u8, dtype, "BF16")) return 2;
    if (std.mem.eql(u8, dtype, "U32") or std.mem.eql(u8, dtype, "I32") or std.mem.eql(u8, dtype, "F32")) return 4;
    if (std.mem.eql(u8, dtype, "U64") or std.mem.eql(u8, dtype, "I64") or std.mem.eql(u8, dtype, "F64")) return 8;
    return null;
}

/// The shared Python dump key. `final_logits` always uses synthetic layer 99.
pub fn key(allocator: std.mem.Allocator, kind: []const u8, layer: usize, call: usize) ![]u8 {
    const n = if (std.mem.eql(u8, kind, "final_logits")) 99 else layer;
    return try std.fmt.allocPrint(allocator, "{s}.L{d:0>2}.{d}", .{ kind, n, call });
}

/// Gate mode's tensor set: probe ids, logq, full-vocabulary lse, omitted tail mass, argmax, window and position.
pub const GateRow = struct {
    ids: [64]i32,
    logp: [64]f32,
    lse: f32,
    mass_outside: f32,
    argmax: i32,
    window_id: i32,
    pos: i64,
};

/// Flatten gate rows into the Python tool's safetensors names, shapes and dtypes.
pub fn gateTensors(allocator: std.mem.Allocator, rows: []const GateRow) ![]Tensor {
    const ids = try allocator.alloc(i32, rows.len * 64);
    defer allocator.free(ids);
    const logp = try allocator.alloc(f32, rows.len * 64);
    defer allocator.free(logp);
    const lse = try allocator.alloc(f32, rows.len);
    defer allocator.free(lse);
    const mass = try allocator.alloc(f32, rows.len);
    defer allocator.free(mass);
    const argmax = try allocator.alloc(i32, rows.len);
    defer allocator.free(argmax);
    const window_id = try allocator.alloc(i32, rows.len);
    defer allocator.free(window_id);
    const pos = try allocator.alloc(i64, rows.len);
    defer allocator.free(pos);
    for (rows, 0..) |row, i| {
        @memcpy(ids[i * 64 ..][0..64], &row.ids);
        @memcpy(logp[i * 64 ..][0..64], &row.logp);
        lse[i] = row.lse;
        mass[i] = row.mass_outside;
        argmax[i] = row.argmax;
        window_id[i] = row.window_id;
        pos[i] = row.pos;
    }
    const result = try allocator.alloc(Tensor, 7);
    result[0] = .{ .name = "ids", .dtype = "I32", .shape = try allocator.dupe(usize, &.{ rows.len, 64 }), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(ids)) };
    result[1] = .{ .name = "logp", .dtype = "F32", .shape = try allocator.dupe(usize, &.{ rows.len, 64 }), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(logp)) };
    result[2] = .{ .name = "lse", .dtype = "F32", .shape = try allocator.dupe(usize, &.{rows.len}), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(lse)) };
    result[3] = .{ .name = "mass_outside", .dtype = "F32", .shape = try allocator.dupe(usize, &.{rows.len}), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(mass)) };
    result[4] = .{ .name = "argmax", .dtype = "I32", .shape = try allocator.dupe(usize, &.{rows.len}), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(argmax)) };
    result[5] = .{ .name = "window_id", .dtype = "I32", .shape = try allocator.dupe(usize, &.{rows.len}), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(window_id)) };
    result[6] = .{ .name = "pos", .dtype = "I64", .shape = try allocator.dupe(usize, &.{rows.len}), .bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(pos)) };
    return result;
}

/// Release every allocation returned by gateTensors when the caller uses a non-arena allocator.
pub fn freeGateTensors(allocator: std.mem.Allocator, tensors: []Tensor) void {
    for (tensors) |tensor| {
        allocator.free(tensor.shape);
        allocator.free(tensor.bytes);
    }
    allocator.free(tensors);
}

test "dump recursively analyzes exported declarations" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Tensor);
    std.testing.refAllDecls(GateRow);
}
