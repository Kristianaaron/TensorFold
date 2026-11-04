// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP9 host integration: a synthetic two-layer GLM-5.3 checkpoint (one dense layer with a full indexer, one MoE layer
//! with low-rank and a shared indexer) written here as safetensors + index, loaded through the real loader into
//! host memory behind the repo's CUDA memory hook; every tensor's bytes, rank slice and dtype are checked.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights");
const state = @import("state");
const forward = @import("forward");
const engine = @import("engine");
const glm_config = @import("glm_config");
const weights_uma = @import("uma");
const manifest_mod = weights.manifest;

const fp8 = cuda.fp8;
const gpa = std.testing.allocator;

test "WP9 modules expose the intended engine components" {
    std.testing.refAllDecls(weights);
    std.testing.refAllDecls(state);
    std.testing.refAllDecls(forward);
    std.testing.refAllDecls(engine);
}

test "manifest plans routed widths from width filtered packs and loads their per matrix K2" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa) };
    defer b.arena.deinit();
    const c = tinyConfig();
    try buildCheckpoint(&b, c);
    var i: usize = 0;
    while (i < b.entries[1].items.len) {
        if (std.mem.indexOf(u8, b.entries[1].items[i].name, ".mlp.experts.") != null) {
            _ = b.entries[1].orderedRemove(i);
        } else i += 1;
    }
    try b.write(tmp.dir, io);
    try tmp.dir.createDirPath(io, "packs/L01");
    for (1..4) |k| {
        // Pack tensors store the complete name. Keep the fixture's payloads distinct
        // across widths so the load assertion can identify the selected pack.
        var pb: Builder = .{ .arena = .init(gpa), .seed = @intCast(k * 100), .prefix = "model." };
        defer pb.arena.deinit();
        for (0..c.experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
            const down = std.mem.eql(u8, proj, "down_proj");
            const rows = if (down) c.moe_inter else c.hidden;
            const cols = if (down) c.hidden else c.moe_inter;
            const base = try std.fmt.allocPrint(pb.a(), "model.layers.1.mlp.experts.{d}.{s}.", .{ e, proj });
            _ = try pb.add(0, try cat(&pb, base, "trellis"), "I16", &.{ rows / 16, cols / 16, 16 * k }, 2);
            _ = try pb.add(0, try cat(&pb, base, "suh"), "F16", &.{rows}, 2);
            _ = try pb.add(0, try cat(&pb, base, "svh"), "F16", &.{cols}, 2);
            const marker = try pb.add(0, try cat(&pb, base, "mcg_e4m3"), "I32", &.{1}, 4);
            std.mem.writeInt(u32, marker[0..4], 0xCBAC1FED, .little);
        };
        const pack_path = try std.fmt.allocPrint(pb.a(), "packs/L01/K{d}.e0-2.safetensors", .{k});
        try pb.writePack(tmp.dir, io, pack_path);
    }
    var manifest_json: std.ArrayList(u8) = .empty;
    defer manifest_json.deinit(gpa);
    try manifest_json.appendSlice(gpa, "{\"max_context\":131072");
    for (0..c.experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        try manifest_json.print(gpa, ",\"1.{d}.{s}\":{d}", .{ e, proj, 1 + (e + pi) % 3 });
    };
    try manifest_json.append(gpa, '}');
    var m = try manifest_mod.Manifest.parse(gpa, "mixed", manifest_json.items);
    defer m.deinit();
    var dir_buf: [96]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const select: weights.Select = .{ .manifest = &m };
    var missing = try manifest_mod.Manifest.parse(gpa, "missing", "{\"max_context\":131072}");
    defer missing.deinit();
    try std.testing.expectError(error.ManifestIncomplete, weights.planRank(gpa, io, dir, c, 0, 2, .{ .manifest = &missing }));
    try m.widths.put(gpa, manifest_mod.key(1, 0, .gate_proj), 4);
    try std.testing.expectError(error.ManifestMatrixMissing, weights.planRank(gpa, io, dir, c, 0, 2, select));
    try m.widths.put(gpa, manifest_mod.key(1, 0, .gate_proj), 1);
    var plan = try weights.planRank(gpa, io, dir, c, 0, 2, select);
    defer plan.deinit(gpa);
    try std.testing.expect(plan.weights.routed > 0);
    HostDevice.install();
    defer cuda.memory.setHook(null);
    const adm = try state.admitRank(tiny_geometry, 16, plan.weights, 1 << 30, state.reserveFor(1 << 30, false, 0));
    var loaded = try weights.ModelWeights.loadConfig(dir, c, 0, 2, budget(io, &adm), select);
    defer loaded.deinit();
    const moe = loaded.layers[1].ffn.moe;
    const gate = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, HostDevice.bytesAt(moe.routed.gate_k2, 8)));
    const up = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, HostDevice.bytesAt(moe.routed.up_k2, 8)));
    const down = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, HostDevice.bytesAt(moe.routed.down_k2, 8)));
    for (0..c.experts) |e| {
        try std.testing.expectEqual(@as(i32, 2 * m.width(1, e, .gate_proj).?), gate[e]);
        try std.testing.expectEqual(@as(i32, 2 * m.width(1, e, .up_proj).?), up[e]);
        try std.testing.expectEqual(@as(i32, 2 * m.width(1, e, .down_proj).?), down[e]);
    }
}

/// Host "device" memory: the hook hands out real host allocations, so uploaded bytes can be read back.
const HostDevice = struct {
    var blocks: [512][]align(256) u8 = undefined;
    var count: usize = 0;
    var live: usize = 0;
    var allocs: usize = 0;

    fn alloc(_: *anyopaque, len: usize) ?cuda.abi.DevicePtr {
        if (count == blocks.len) return null;
        const b = gpa.alignedAlloc(u8, .fromByteUnits(256), len) catch return null;
        blocks[count] = b;
        count += 1;
        live += 1;
        allocs += 1;
        return @intFromPtr(b.ptr);
    }

    fn free(_: *anyopaque, p: cuda.abi.DevicePtr) bool {
        for (blocks[0..count]) |*b| if (b.len != 0 and @intFromPtr(b.ptr) == p) {
            gpa.free(b.*);
            b.* = b.*[0..0];
            live -= 1;
            return true;
        };
        return false;
    }

    fn devAlloc(_: ?*anyopaque, d: *const cuda.Driver, len: usize) anyerror!cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(d, len);
    }

    fn devWrite(_: ?*anyopaque, buf: cuda.DeviceBuffer, offset: usize, bytes: []const u8) anyerror!void {
        if (offset + bytes.len > buf.len) return error.WriteOutOfBounds;
        @memcpy(@as([*]u8, @ptrFromInt(buf.ptr))[offset..][0..bytes.len], bytes);
    }

    const device: weights.Device = .{ .alloc_fn = devAlloc, .write_fn = devWrite };

    fn bytesAt(p: u64, n: usize) []const u8 {
        return @as([*]const u8, @ptrFromInt(p))[0..n];
    }

    fn install() void {
        count = 0;
        live = 0;
        allocs = 0;
        var dummy: u8 = 0;
        _ = &dummy;
        cuda.memory.setHook(.{ .ctx = @ptrCast(&blocks), .alloc = alloc, .free = free });
    }
};

/// One tensor of the synthetic checkpoint: dtype as safetensors names it, shape, little-endian bytes.
const Entry = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []u8 };

const Builder = struct {
    arena: std.heap.ArenaAllocator,
    entries: [2]std.ArrayList(Entry) = .{ .empty, .empty },
    seed: u32 = 1,
    /// Name form of the language-model tensors ("model." or a merged conversion's "model.language_model.").
    prefix: []const u8 = "model.",

    fn a(b: *Builder) std.mem.Allocator {
        return b.arena.allocator();
    }

    fn add(b: *Builder, shard: usize, name: []const u8, dtype: []const u8, shape: []const usize, elem: usize) ![]u8 {
        var n: usize = elem;
        for (shape) |d| n *= d;
        const bytes = try b.a().alloc(u8, n);
        // Distinct, deterministic bytes; e4m3 codes avoid NaN (0x7f/0xff) so host references stay finite.
        for (bytes, 0..) |*v, i| {
            var x: u8 = @truncate((i *% 131) +% b.seed *% 17);
            if (std.mem.eql(u8, dtype, "F8_E4M3") and (x & 0x7f) == 0x7f) x ^= 1;
            v.* = x;
        }
        b.seed +%= 1;
        try b.entries[shard].append(b.a(), .{ .name = try b.a().dupe(u8, name), .dtype = dtype, .shape = try b.a().dupe(usize, shape), .bytes = bytes });
        return bytes;
    }

    /// bf16 values (small, distinct) rather than raw bytes, so f32 conversions are meaningful.
    fn bf16s(b: *Builder, shard: usize, name: []const u8, shape: []const usize) ![]u8 {
        const bytes = try b.add(shard, name, "BF16", shape, 2);
        for (0..bytes.len / 2) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + b.seed) % 61)) - 30)) / 16.0;
            std.mem.writeInt(u16, bytes[i * 2 ..][0..2], weights.toBf16(v), .little);
        }
        return bytes;
    }

    fn f32s(b: *Builder, shard: usize, name: []const u8, shape: []const usize) ![]u8 {
        const bytes = try b.add(shard, name, "F32", shape, 4);
        for (0..bytes.len / 4) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatFromInt(i + 1)) * 0.03125), .little);
        return bytes;
    }

    /// An FP8 linear `prefix`weight [rows][cols] with weight_scale_inv [ceil(rows/128)][ceil(cols/128)].
    fn fp8Linear(b: *Builder, shard: usize, prefix: []const u8, rows: usize, cols: usize) !void {
        var nb: [256]u8 = undefined;
        _ = try b.add(shard, try std.fmt.bufPrint(&nb, "{s}weight", .{prefix}), "F8_E4M3", &.{ rows, cols }, 1);
        _ = try b.f32s(shard, try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{prefix}), &.{ (rows + 127) / 128, (cols + 127) / 128 });
    }

    fn write(b: *Builder, dir: std.Io.Dir, io: std.Io) !void {
        const files = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
        var index: std.ArrayList(u8) = .empty;
        try index.appendSlice(b.a(), "{\"metadata\":{},\"weight_map\":{");
        var first = true;
        for (b.entries, files) |list, fname| {
            var header: std.ArrayList(u8) = .empty;
            try header.append(b.a(), '{');
            var offset: usize = 0;
            for (list.items, 0..) |e, i| {
                if (i != 0) try header.append(b.a(), ',');
                try header.print(b.a(), "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ e.name, e.dtype });
                for (e.shape, 0..) |d, j| try header.print(b.a(), "{s}{d}", .{ if (j == 0) "" else ",", d });
                try header.print(b.a(), "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.bytes.len });
                offset += e.bytes.len;
                try index.print(b.a(), "{s}\"{s}\":\"{s}\"", .{ if (first) "" else ",", e.name, fname });
                first = false;
            }
            try header.append(b.a(), '}');
            var image: std.ArrayList(u8) = .empty;
            var len: [8]u8 = undefined;
            std.mem.writeInt(u64, &len, header.items.len, .little);
            try image.appendSlice(b.a(), &len);
            try image.appendSlice(b.a(), header.items);
            for (list.items) |e| try image.appendSlice(b.a(), e.bytes);
            try dir.writeFile(io, .{ .sub_path = fname, .data = image.items });
        }
        try index.appendSlice(b.a(), "}}");
        try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    }

    fn writePack(b: *Builder, dir: std.Io.Dir, io: std.Io, path: []const u8) !void {
        var header: std.ArrayList(u8) = .empty;
        try header.append(b.a(), '{');
        var offset: usize = 0;
        for (b.entries[0].items, 0..) |e, i| {
            if (i != 0) try header.append(b.a(), ',');
            try header.print(b.a(), "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ e.name, e.dtype });
            for (e.shape, 0..) |d, j| try header.print(b.a(), "{s}{d}", .{ if (j == 0) "" else ",", d });
            try header.print(b.a(), "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.bytes.len });
            offset += e.bytes.len;
        }
        try header.append(b.a(), '}');
        var image: std.ArrayList(u8) = .empty;
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.items.len, .little);
        try image.appendSlice(b.a(), &len);
        try image.appendSlice(b.a(), header.items);
        for (b.entries[0].items) |e| try image.appendSlice(b.a(), e.bytes);
        try dir.writeFile(io, .{ .sub_path = path, .data = image.items });
    }

    fn find(b: *const Builder, name: []const u8) []const u8 {
        for (b.entries) |list| for (list.items) |e| if (std.mem.eql(u8, e.name, name)) return e.bytes;
        @panic("synthetic tensor missing");
    }
};

/// Tiny glm_moe_dsa geometry (the loader's override; config.zig's parse refuses non-GLM-5.3 shapes).
fn tinyConfig() glm_config.Config {
    var c: glm_config.Config = .{ .hidden = 256, .layers = 2, .vocab = 256, .dense_layers = 1, .dense_inter = 256, .experts = 2, .moe_inter = 256, .heads = 4, .q_lora = 128, .routed_codebook = .mcg_e4m3 };
    c.full.set(0); // layer 0 runs its own indexer; layer 1 shares layer 0's selection
    return c;
}

const k2_of = [2][3]usize{ .{ 2, 4, 6 }, .{ 4, 2, 2 } }; // expert x (gate, up, down) half-bit widths

fn buildCheckpoint(b: *Builder, c: glm_config.Config) !void {
    const h = c.hidden;
    var nb: [256]u8 = undefined;
    for (0..2) |i| {
        const s = i; // layer i in shard i, globals in shard 0
        const p = try std.fmt.bufPrint(&nb, "{s}layers.{d}.", .{ b.prefix, i });
        var pb: [64]u8 = undefined;
        const pre = try b.a().dupe(u8, p);
        _ = &pb;
        _ = try b.bf16s(s, try cat(b, pre, "input_layernorm.weight"), &.{h});
        _ = try b.bf16s(s, try cat(b, pre, "post_attention_layernorm.weight"), &.{h});
        try b.fp8Linear(s, try cat(b, pre, "self_attn.q_a_proj."), c.q_lora, h);
        _ = try b.bf16s(s, try cat(b, pre, "self_attn.q_a_layernorm.weight"), &.{c.q_lora});
        try b.fp8Linear(s, try cat(b, pre, "self_attn.q_b_proj."), c.heads * (c.nope + c.rope), c.q_lora);
        try b.fp8Linear(s, try cat(b, pre, "self_attn.kv_a_proj_with_mqa."), c.kv_lora + c.rope, h);
        _ = try b.bf16s(s, try cat(b, pre, "self_attn.kv_a_layernorm.weight"), &.{c.kv_lora});
        try b.fp8Linear(s, try cat(b, pre, "self_attn.kv_b_proj."), c.heads * (c.nope + c.v_dim), c.kv_lora);
        try b.fp8Linear(s, try cat(b, pre, "self_attn.o_proj."), h, c.heads * c.v_dim);
        if (i == 0) {
            try b.fp8Linear(s, try cat(b, pre, "self_attn.indexer.wq_b."), c.i_heads * c.i_dim, c.q_lora);
            try b.fp8Linear(s, try cat(b, pre, "self_attn.indexer.wk."), c.i_dim, h);
            _ = try b.bf16s(s, try cat(b, pre, "self_attn.indexer.weights_proj.weight"), &.{ c.i_heads, h });
            _ = try b.bf16s(s, try cat(b, pre, "self_attn.indexer.k_norm.weight"), &.{c.i_dim});
            _ = try b.bf16s(s, try cat(b, pre, "self_attn.indexer.k_norm.bias"), &.{c.i_dim});
            try b.fp8Linear(s, try cat(b, pre, "mlp.gate_proj."), c.dense_inter, h);
            try b.fp8Linear(s, try cat(b, pre, "mlp.up_proj."), c.dense_inter, h);
            try b.fp8Linear(s, try cat(b, pre, "mlp.down_proj."), h, c.dense_inter);
            continue;
        }
        _ = try b.bf16s(s, try cat(b, pre, "mlp.gate.weight"), &.{ c.experts, h });
        _ = try b.f32s(s, try cat(b, pre, "mlp.gate.e_score_correction_bias"), &.{c.experts});
        for (0..c.experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
            const ep = try std.fmt.allocPrint(b.a(), "{s}mlp.experts.{d}.{s}.", .{ pre, e, proj });
            const k: usize = if (pi == 2) c.moe_inter else h;
            const n: usize = if (pi == 2) h else c.moe_inter;
            _ = try b.add(s, try cat(b, ep, "trellis"), "I16", &.{ k / 16, n / 16, k2_of[e][pi] * 8 }, 2);
            _ = try b.add(s, try cat(b, ep, "suh"), "F16", &.{k}, 2);
            _ = try b.add(s, try cat(b, ep, "svh"), "F16", &.{n}, 2);
            const marker = try b.add(s, try cat(b, ep, "mcg_e4m3"), "I32", &.{1}, 4);
            std.mem.writeInt(u32, marker[0..4], 0xCBAC1FED, .little);
        };
        try b.fp8Linear(s, try cat(b, pre, "mlp.shared_experts.gate_proj."), c.moe_inter, h);
        try b.fp8Linear(s, try cat(b, pre, "mlp.shared_experts.up_proj."), c.moe_inter, h);
        try b.fp8Linear(s, try cat(b, pre, "mlp.shared_experts.down_proj."), h, c.moe_inter);
        _ = try b.bf16s(s, try cat(b, pre, "mlp.lowrank_a"), &.{ h, 64 });
        _ = try b.bf16s(s, try cat(b, pre, "mlp.lowrank_b"), &.{ 64, h });
    }
    _ = try b.bf16s(0, try cat(b, b.prefix, "embed_tokens.weight"), &.{ c.vocab, h });
    _ = try b.bf16s(0, try cat(b, b.prefix, "norm.weight"), &.{h});
    _ = try b.bf16s(1, "lm_head.weight", &.{ c.vocab, h });
}

fn cat(b: *Builder, x: []const u8, y: []const u8) ![]const u8 {
    return std.mem.concat(b.a(), u8, &.{ x, y });
}

const Span = weights.Span;

fn scaleFloats(bytes: []const u8) ![]f32 {
    const out = try gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = std.mem.bytesToValue(f32, bytes[i * 4 ..][0..4]);
    return out;
}

/// The packed bytes the loader must produce for rows `parts` (each a row span of one FP8 source) and columns `cols`.
fn expectFp8(b: *const Builder, prefixes: []const []const u8, rows: []const Span, full_rows: usize, full_cols: usize, cols: Span, w: cuda.qmmf.Weight) !void {
    var n: usize = 0;
    for (rows) |r| n += r.len();
    const k = cols.len();
    try std.testing.expectEqual(cuda.qmmf.Mode.fp8b, w.mode);
    try std.testing.expectEqual(@as(u32, @intCast(n)), w.n);
    try std.testing.expectEqual(@as(u32, @intCast(k)), w.k);
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const kb = (full_cols + 127) / 128;
    var scale: std.ArrayList(f32) = .empty;
    defer scale.deinit(gpa);
    var at: usize = 0;
    var nb: [256]u8 = undefined;
    for (prefixes, rows) |p, r| {
        const src = b.find(try std.fmt.bufPrint(&nb, "{s}weight", .{p}));
        for (r.lo..r.hi) |row| {
            @memcpy(codes[at..][0..k], src[row * full_cols + cols.lo ..][0..k]);
            at += k;
        }
        const s = try scaleFloats(b.find(try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{p})));
        defer gpa.free(s);
        if (prefixes.len == 1) try scale.appendSlice(gpa, s) else try scale.appendSlice(gpa, s[r.lo / 128 * kb .. r.hi / 128 * kb]);
    }
    const want = try gpa.alloc(u8, fp8.padded(n) * k);
    defer gpa.free(want);
    fp8.packCodes(want, codes, n, k);
    try std.testing.expectEqualSlices(u8, want, HostDevice.bytesAt(w.codes, want.len));
    const want_s = try gpa.alloc(f32, fp8.padded(n) / 64 * (k / 64));
    defer gpa.free(want_s);
    try fp8.packBlockScales(want_s, scale.items, if (prefixes.len == 1) full_rows else n, full_cols, if (prefixes.len == 1) rows[0].lo else 0, cols.lo, n, k);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_s), HostDevice.bytesAt(w.scales, want_s.len * 4));
}

fn expectBf16Linear(b: *const Builder, name: []const u8, rows: Span, full_cols: usize, w: cuda.qmmf.Weight) !void {
    const n = rows.len();
    try std.testing.expectEqual(cuda.qmmf.Mode.bf16, w.mode);
    const src = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, b.find(name)));
    const words = try gpa.alloc(u16, n * full_cols);
    defer gpa.free(words);
    for (0..n) |r| for (0..full_cols) |c| {
        words[r * full_cols + c] = src[(rows.lo + r) * full_cols + c];
    };
    const npad = (n + 127) / 128 * 128;
    const want = try gpa.alloc(u16, npad * full_cols);
    defer gpa.free(want);
    cuda.qmmf.packBf16(want, words, n, full_cols);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), HostDevice.bytesAt(w.codes, want.len * 2));
}

fn expectDtype(m: *const weights.ModelWeights, name: []const u8, dtype: weights.DType) !void {
    const t = m.find(name) orelse return error.TensorNotLoaded;
    try std.testing.expectEqual(dtype, t.dtype);
}

/// A tiny State geometry for the synthetic model's admission rows.
const tiny_geometry: state.Geometry = .{ .layers = 2, .full_index_layers = 1, .chunk_rows = 64, .piece_rows = 32, .index_slots = 16, .decode_rows = 2, .graphs_bytes = 0, .expert_rows = 16, .head_rows = 2, .vocab_local = 128 };

/// Admission as production makes it: the exact header plan of what this rank loads, then admitRank (once).
fn admitted(io: std.Io, dir: []const u8, c: glm_config.Config, rank: usize, select: weights.Select) !state.Admission {
    var plan = try weights.planRank(gpa, io, dir, c, rank, 2, select);
    defer plan.deinit(gpa);
    return state.admitRank(tiny_geometry, 16, plan.weights, 1 << 30, state.reserveFor(1 << 30, false, 0));
}

fn budget(io: std.Io, adm: *const state.Admission) weights.Budget {
    return .{ .allocator = gpa, .io = io, .driver = undefined, .admission = adm, .device = HostDevice.device };
}

fn writeSynthetic(b: *Builder, io: std.Io, tmp: std.testing.TmpDir, c: glm_config.Config, path: *[96]u8) ![]const u8 {
    try buildCheckpoint(b, c);
    try b.write(tmp.dir, io);
    return std.fmt.bufPrint(path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

test "synthetic two-layer checkpoint loads through the real loader: bytes, rank slices and dtypes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa) };
    defer b.arena.deinit();
    const c = tinyConfig();
    var path: [96]u8 = undefined;
    const dir = try writeSynthetic(&b, io, tmp, c, &path);
    HostDevice.install();
    defer cuda.memory.setHook(null);
    const rank = 1;
    const adm = try admitted(io, dir, c, rank, .{});
    var m = try weights.ModelWeights.loadConfig(dir, c, rank, 2, budget(io, &adm), .{});
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 2), m.layers.len);
    const h = c.hidden;
    const heads: Span = .{ .lo = 2, .hi = 4 };
    for (m.layers, 0..) |l, i| {
        const pre = try std.fmt.allocPrint(gpa, "model.layers.{d}.self_attn.", .{i});
        defer gpa.free(pre);
        var nb: [4][256]u8 = undefined;
        const at = l.attn;
        try std.testing.expectEqual(@as(usize, 2), at.local_heads);
        try std.testing.expectEqual(@as(usize, 2), at.head0);
        try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}q_a_proj.", .{pre})}, &.{.{ .lo = 0, .hi = c.q_lora }}, c.q_lora, h, .{ .lo = 0, .hi = h }, at.q_a);
        try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}q_b_proj.", .{pre})}, &.{.{ .lo = heads.lo * 256, .hi = heads.hi * 256 }}, c.heads * 256, c.q_lora, .{ .lo = 0, .hi = c.q_lora }, at.q_b);
        try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}kv_a_proj_with_mqa.", .{pre})}, &.{.{ .lo = 0, .hi = 576 }}, 576, h, .{ .lo = 0, .hi = h }, at.kv_a);
        try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}o_proj.", .{pre})}, &.{.{ .lo = 0, .hi = h }}, h, c.heads * 256, .{ .lo = heads.lo * 256, .hi = heads.hi * 256 }, at.o_proj);
        // kv_b: one copy, checkpoint row-major rows of this rank's heads and their scale rows.
        const kvb = b.find(try std.fmt.bufPrint(&nb[0], "{s}kv_b_proj.weight", .{pre}));
        try std.testing.expectEqualSlices(u8, kvb[heads.lo * 448 * 512 .. heads.hi * 448 * 512], HostDevice.bytesAt(at.kv_b.codes, 896 * 512));
        const kvs = b.find(try std.fmt.bufPrint(&nb[0], "{s}kv_b_proj.weight_scale_inv", .{pre}));
        try std.testing.expectEqualSlices(u8, kvs[7 * 4 * 4 .. 14 * 4 * 4], HostDevice.bytesAt(at.kv_b.scales, 7 * 4 * 4));
        try std.testing.expectEqual(@as(usize, 896), at.kv_b.rows);
        try std.testing.expectEqualSlices(u8, b.find(try std.fmt.bufPrint(&nb[0], "{s}kv_a_layernorm.weight", .{pre})), HostDevice.bytesAt(at.kv_a_norm, 1024));
        var count: usize = 0;
        for (m.tensors.items) |t| if (std.mem.indexOf(u8, t.name, "kv_b_proj") != null and std.mem.startsWith(u8, t.name, pre[6..])) {
            count += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), count);
        try expectDtype(&m, try std.fmt.bufPrint(&nb[1], "{s}q_a_proj.weight", .{pre[6..]}), .e4m3);
        try expectDtype(&m, try std.fmt.bufPrint(&nb[1], "{s}q_a_layernorm.weight", .{pre[6..]}), .bf16);
        if (i == 0) {
            const ix = at.indexer orelse return error.MissingIndexer;
            try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}indexer.wq_b.", .{pre})}, &.{.{ .lo = 0, .hi = 4096 }}, 4096, c.q_lora, .{ .lo = 0, .hi = c.q_lora }, ix.wq_b);
            try expectFp8(&b, &.{try std.fmt.bufPrint(&nb[0], "{s}indexer.wk.", .{pre})}, &.{.{ .lo = 0, .hi = 128 }}, 128, h, .{ .lo = 0, .hi = h }, ix.wk);
            try expectBf16Linear(&b, try std.fmt.bufPrint(&nb[0], "{s}indexer.weights_proj.weight", .{pre}), .{ .lo = 0, .hi = 32 }, h, ix.weights_proj);
            try expectDtype(&m, "layers.0.self_attn.indexer.weights_proj.weight", .bf16);
            const d = l.ffn.dense;
            const span: Span = .{ .lo = 128, .hi = 256 };
            try expectFp8(&b, &.{ "model.layers.0.mlp.gate_proj.", "model.layers.0.mlp.up_proj." }, &.{ span, span }, 256, h, .{ .lo = 0, .hi = h }, d.gate_up);
            try expectFp8(&b, &.{"model.layers.0.mlp.down_proj."}, &.{.{ .lo = 0, .hi = h }}, h, 256, span, d.down);
        } else {
            try std.testing.expect(at.indexer == null);
            const mo = l.ffn.moe;
            const span: Span = .{ .lo = 128, .hi = 256 };
            try expectFp8(&b, &.{ "model.layers.1.mlp.shared_experts.gate_proj.", "model.layers.1.mlp.shared_experts.up_proj." }, &.{ span, span }, 256, h, .{ .lo = 0, .hi = h }, mo.shared.gate_up);
            try expectFp8(&b, &.{"model.layers.1.mlp.shared_experts.down_proj."}, &.{.{ .lo = 0, .hi = h }}, h, 256, span, mo.shared.down);
            try std.testing.expectEqualSlices(u8, b.find("model.layers.1.mlp.gate.weight"), HostDevice.bytesAt(mo.router, 2 * h * 2));
            try std.testing.expectEqualSlices(u8, b.find("model.layers.1.mlp.gate.e_score_correction_bias"), HostDevice.bytesAt(mo.bias, 8));
            // Low-rank: bf16 on disk -> fp32 resident; A keeps columns R/2.., B keeps rows R/2...
            const lr = mo.lowrank orelse return error.MissingLowRank;
            try std.testing.expectEqual(@as(usize, 32), lr.rank_local);
            try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, lr.span);
            const a_src = b.find("model.layers.1.mlp.lowrank_a");
            const a_dev = HostDevice.bytesAt(lr.a, h * 32 * 4);
            for (0..h) |r| for (0..32) |j| {
                const bits = std.mem.readInt(u16, a_src[(r * 64 + 32 + j) * 2 ..][0..2], .little);
                try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, a_dev[(r * 32 + j) * 4 ..][0..4], .little));
            };
            const b_src = b.find("model.layers.1.mlp.lowrank_b");
            const b_dev = HostDevice.bytesAt(lr.b, 32 * h * 4);
            for (0..32 * h) |e| {
                const bits = std.mem.readInt(u16, b_src[(32 * h + e) * 2 ..][0..2], .little);
                try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, b_dev[e * 4 ..][0..4], .little));
            }
            try expectDtype(&m, "layers.1.mlp.lowrank_a", .f32);
            // Routed: WP4's loader, this rank's 128-wide halves, per-matrix K2 and the mcg_e4m3 codebook.
            try std.testing.expectEqual(@as(usize, 128), mo.routed.width);
            try std.testing.expectEqual(cuda.exl3.Codebook.mcg_e4m3, mo.routed.codebook);
            for ([_]u64{ mo.routed.gate_k2, mo.routed.up_k2, mo.routed.down_k2 }, 0..) |ptr, pi| {
                const ks = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, HostDevice.bytesAt(ptr, 8)));
                for (0..2) |e| try std.testing.expectEqual(@as(i32, @intCast(k2_of[e][pi])), ks[e]);
            }
        }
    }
    try std.testing.expectEqual(Span{ .lo = 128, .hi = 256 }, m.vocab);
    try std.testing.expectEqualSlices(u8, b.find("model.embed_tokens.weight")[128 * h * 2 ..], HostDevice.bytesAt(m.embed, 128 * h * 2));
    try expectBf16Linear(&b, "lm_head.weight", .{ .lo = 128, .hi = 256 }, h, m.head.?);
    try expectDtype(&m, "lm_head.weight", .bf16);
    // The logical classes cover every loaded byte exactly once; the arenas hold them in one allocation per class.
    var all: u64 = 0;
    for (m.tensors.items) |t| all += t.data.len + if (t.scales) |sc| sc.len else 0;
    const logical = m.logicalClasses();
    try std.testing.expectEqual(all, logical.total());
    try std.testing.expect(logical.routed > 0 and logical.backbone_fp8 > 0 and logical.embed_head > 0 and logical.lowrank == 2 * 256 * 32 * 4);
    try std.testing.expectEqual(@as(usize, 5), m.allocationCount());
    try std.testing.expect(m.classes().total() >= all and m.classes().total() % (2 << 20) == 0);
    // Every tensor sits inside its class arena at 256-byte placement.
    for (m.tensors.items) |t| {
        try std.testing.expect(t.data.ptr % 256 == 0 and t.data.borrowed);
        var inside = false;
        for (m.arenas) |a| inside = inside or (t.data.ptr >= a.ptr and t.data.ptr + t.data.len <= a.ptr + a.len);
        try std.testing.expect(inside);
    }
}

test "loader fails closed: admission mismatch before any allocation, codebook mismatch, single-layer select" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa) };
    defer b.arena.deinit();
    const c = tinyConfig();
    var path: [96]u8 = undefined;
    const dir = try writeSynthetic(&b, io, tmp, c, &path);
    HostDevice.install();
    defer cuda.memory.setHook(null);
    // An admission made for another selection is refused before anything is uploaded.
    const layer1 = try admitted(io, dir, c, 0, .{ .layers = &.{1}, .globals = false });
    try std.testing.expectError(error.AdmissionMismatch, weights.ModelWeights.loadConfig(dir, c, 0, 2, budget(io, &layer1), .{}));
    try std.testing.expectEqual(@as(usize, 0), HostDevice.allocs);
    // A budget that does not fit fails at admission, the single point, before any load.
    var plan = try weights.planRank(gpa, io, dir, c, 0, 2, .{});
    defer plan.deinit(gpa);
    try std.testing.expectError(error.MemoryBudget, state.admitRank(tiny_geometry, 16, plan.weights, 1 << 20, state.reserveFor(1 << 20, false, 0)));
    // mcg_e4m3 is accepted only when the config names it for the routed packs.
    var mcg = c;
    mcg.routed_codebook = .mcg;
    const all0 = try admitted(io, dir, c, 0, .{});
    try std.testing.expectError(error.BadExpertCodebook, weights.ModelWeights.loadConfig(dir, mcg, 0, 2, budget(io, &all0), .{}));
    try std.testing.expectEqual(@as(usize, 0), HostDevice.live);
    // One layer alone (no globals), and the rank-0 slices.
    var one = try weights.ModelWeights.loadConfig(dir, c, 0, 2, budget(io, &layer1), .{ .layers = &.{1}, .globals = false });
    defer one.deinit();
    try std.testing.expectEqual(@as(usize, 1), one.layers.len);
    try std.testing.expectEqual(@as(usize, 0), one.layers[0].attn.head0);
    try std.testing.expectEqual(weights.Span{ .lo = 0, .hi = 32 }, one.layers[0].ffn.moe.lowrank.?.span);
    // Rank 0 keeps kv_b rows [0, 896) (heads 0-1) and their scale rows (review L7: both ranks' cuts).
    const kvb = b.find("model.layers.1.self_attn.kv_b_proj.weight");
    try std.testing.expectEqualSlices(u8, kvb[0 .. 896 * 512], HostDevice.bytesAt(one.layers[0].attn.kv_b.codes, 896 * 512));
}

test "exact header plan: per-class bytes equal the loaded bytes; unexpected tensors fail a full load closed" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa) };
    defer b.arena.deinit();
    const c = tinyConfig();
    try buildCheckpoint(&b, c);
    _ = try b.bf16s(1, "model.layers.1.self_attn.indexer.k_norm.weight", &.{128}); // an indexer part on a shared layer
    try b.write(tmp.dir, io);
    var path: [96]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    HostDevice.install();
    defer cuda.memory.setHook(null);
    var plan = try weights.planRank(gpa, io, dir, c, 1, 2, .{});
    defer plan.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), plan.unused.len);
    try std.testing.expectEqualStrings("model.layers.1.self_attn.indexer.k_norm.weight", plan.unused[0]);
    const adm = try state.admitRank(tiny_geometry, 16, plan.weights, 1 << 30, state.reserveFor(1 << 30, false, 0));
    try std.testing.expectError(error.UnusedCheckpointTensors, weights.ModelWeights.loadConfig(dir, c, 1, 2, budget(io, &adm), .{}));
    try std.testing.expectEqual(@as(usize, 0), HostDevice.allocs);
    // A layer selection ignores what it does not select, and its plan equals its loaded bytes class by class.
    var sel_plan = try weights.planRank(gpa, io, dir, c, 1, 2, .{ .layers = &.{ 0, 1 } });
    defer sel_plan.deinit(gpa);
    const sel = try state.admitRank(tiny_geometry, 16, sel_plan.weights, 1 << 30, state.reserveFor(1 << 30, false, 0));
    var m = try weights.ModelWeights.loadConfig(dir, c, 1, 2, budget(io, &sel), .{ .layers = &.{ 0, 1 } });
    defer m.deinit();
    const got = m.classes();
    inline for (.{ "routed", "backbone_fp8", "embed_head", "router_norms", "lowrank" }) |f| try std.testing.expectEqual(@field(sel_plan.weights, f), @field(got, f));
    const logical = m.logicalClasses();
    inline for (.{ "routed", "backbone_fp8", "embed_head", "router_norms", "lowrank" }) |f| try std.testing.expectEqual(@field(sel_plan.logical, f), @field(logical, f));
    try std.testing.expect(got.routed > 0 and logical.lowrank == 256 * 32 * 4 * 2);
}

test "the merged-conversion name form model.language_model.* loads the same bytes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa), .prefix = "model.language_model." };
    defer b.arena.deinit();
    const c = tinyConfig();
    var path: [96]u8 = undefined;
    const dir = try writeSynthetic(&b, io, tmp, c, &path);
    HostDevice.install();
    defer cuda.memory.setHook(null);
    const adm = try admitted(io, dir, c, 1, .{});
    var m = try weights.ModelWeights.loadConfig(dir, c, 1, 2, budget(io, &adm), .{});
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 2), m.layers.len);
    try std.testing.expectEqualSlices(u8, b.find("model.language_model.embed_tokens.weight")[128 * 256 * 2 ..], HostDevice.bytesAt(m.embed, 128 * 256 * 2));
}

test "O_DIRECT reads through four aligned staging slots upload the same device bytes as the mmap" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b: Builder = .{ .arena = .init(gpa) };
    defer b.arena.deinit();
    const c = tinyConfig();
    var path: [96]u8 = undefined;
    const dir = try writeSynthetic(&b, io, tmp, c, &path);
    HostDevice.install();
    defer cuda.memory.setHook(null);
    const adm = try admitted(io, dir, c, 1, .{});
    var counting_mapped: Counting = .{ .child = gpa };
    var mapped_budget = budget(io, &adm);
    mapped_budget.allocator = counting_mapped.allocator();
    var mapped = try weights.ModelWeights.loadConfig(dir, c, 1, 2, mapped_budget, .{});
    defer mapped.deinit();
    var staging = try weights_uma.Staging.initHost(gpa, 8192); // small slots: every tensor crosses several
    defer staging.deinit();
    var counting_direct: Counting = .{ .child = gpa };
    var direct_budget = budget(io, &adm);
    direct_budget.allocator = counting_direct.allocator();
    direct_budget.odirect = true;
    direct_budget.staging = &staging;
    var direct = try weights.ModelWeights.loadConfig(dir, c, 1, 2, direct_budget, .{});
    defer direct.deinit();
    try std.testing.expectEqual(mapped.tensors.items.len, direct.tensors.items.len);
    for (mapped.tensors.items, direct.tensors.items, 0..) |x, y, i| {
        try std.testing.expectEqualStrings(x.name, y.name);
        try std.testing.expectEqual(x.data.len, y.data.len);
        if (std.mem.endsWith(u8, x.name, "_ptrs")) {
            // WP4's per-expert trellis pointers are device addresses: equal relative to their trellis upload.
            const xp = std.mem.bytesAsSlice(u64, @as([]align(1) const u8, HostDevice.bytesAt(x.data.ptr, x.data.len)));
            const yp = std.mem.bytesAsSlice(u64, @as([]align(1) const u8, HostDevice.bytesAt(y.data.ptr, y.data.len)));
            for (xp, yp) |a, bb| try std.testing.expectEqual(a - mapped.tensors.items[i - 1].data.ptr, bb - direct.tensors.items[i - 1].data.ptr);
            continue;
        }
        try std.testing.expectEqualSlices(u8, HostDevice.bytesAt(x.data.ptr, x.data.len), HostDevice.bytesAt(y.data.ptr, y.data.len));
        if (x.scales) |xs| try std.testing.expectEqualSlices(u8, HostDevice.bytesAt(xs.ptr, xs.len), HostDevice.bytesAt(y.scales.?.ptr, xs.len));
    }
    // Host transient peak (loader allocations only; the resident tensor list is excluded by subtracting it).
    // TF_GLM53_PRINT_TABLES=1 prints it; a passing test stays quiet otherwise (review N8).
    if (std.c.getenv("TF_GLM53_PRINT_TABLES") != null) std.debug.print("synthetic load host peak: mmap {d} B, O_DIRECT {d} B (largest tensor {d} B)\n", .{ counting_mapped.peak, counting_direct.peak, 256 * 256 * 2 });
    try std.testing.expect(counting_direct.peak < 4 << 20);
}

/// Counts live and peak bytes through an allocator (the loader's host transients).
const Counting = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(c: *Counting) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawAlloc(len, a, ra) orelse return null;
        c.live += len;
        c.peak = @max(c.peak, c.live);
        return p;
    }
    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        if (!c.child.rawResize(m, a, n, ra)) return false;
        c.live = c.live - m.len + n;
        c.peak = @max(c.peak, c.live);
        return true;
    }
    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawRemap(m, a, n, ra) orelse return null;
        c.live = c.live - m.len + n;
        c.peak = @max(c.peak, c.live);
        return p;
    }
    fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        c.child.rawFree(m, a, ra);
        c.live -= m.len;
    }
};
