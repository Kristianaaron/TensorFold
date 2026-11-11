// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! DFlash allocation boundaries and the production loader's partial owner, without a GPU.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const registry = @import("core").contract.registry;
const t = std.testing;

test {
    _ = @import("src/families/glm/dflash_select.zig");
}

test "glm dflash: gather and read fit for every supported rank count" {
    for (1..17) |world| for ([_]usize{ 1, 8, glm.dflash.MAX_M }) |rows| {
        const n = glm.dflash_candidates.Layout.init(world, rows, glm.dflash.CW);
        const gathered = world * rows * glm.dflash.CW;
        try t.expect(gathered <= n.device);
        try t.expect(n.read <= n.host);
        try t.expectEqual(gathered, n.read);
    };
}

const L = glm.dflash_load;

test "glm dflash: staging holds every chunk planRows makes, for every k" {
    for ([_]usize{ L.CHUNK, 7_864_320, 3 * 1024 * 1024 + 4096 }) |limit| {
        const staging = L.stagingBytes(limit);
        var k: usize = 64;
        while (k * 2 * L.PACK_ROWS <= limit) : (k += 64) {
            const chunks = try L.planRows(t.allocator, 3 * limit / (2 * k) + 77, k * 2, limit);
            defer t.allocator.free(chunks);
            for (chunks) |chunk| {
                try t.expect(L.rawBytes(chunk.rows, k) <= staging.raw); // quantizeLinearChunked's raw check
                try t.expect(2 * glm.q4.bytes(chunk.rows, k)[1] <= staging.qscale); // the chunk's packed s then b
            }
        }
    }
    // The GLM-5.3 o_proj chunk that the old k = hidden sizing rejected.
    const old_raw = (L.CHUNK / 2 / 6144) * 6144 / 2 + 2 * ((L.CHUNK / 2 / 6144) * 6144) / 32 + 1024;
    try t.expect(L.rawBytes(4096, 8192) > old_raw);
}

const Resources = struct {
    pred: []u16 = &.{},
    succ: []u16 = &.{},
    markov_w1: []u16 = &.{},
    markov_w2: []u16 = &.{},
    conf_w: []u16 = &.{},
    conf_b: f32 = 0,
    hrows: []u16 = &.{},
    scratch: []u8 = &.{},
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{},
    gh: []f32 = &.{},
};
const Ops = struct {
    g: std.mem.Allocator,
    fail_read: ?usize = null,
    pub fn readBook(o: *Ops, i: usize, dst: []u16) !void {
        if (o.fail_read == i) return error.BadBook;
        @memset(dst, 7);
    }
    pub fn readDspark(_: *Ops, _: *Resources) !void {}
    pub fn scratch(o: *Ops, r: *Resources) !void {
        r.scratch = try o.g.alloc(u8, 64);
    }
    pub fn freeScratch(o: *Ops, r: *Resources) void {
        o.g.free(r.scratch);
    }
};
fn release(g: std.mem.Allocator, r: Resources) void {
    g.free(r.pred);
    g.free(r.succ);
    g.free(r.markov_w1);
    g.free(r.markov_w2);
    g.free(r.conf_w);
    g.free(r.hrows);
    g.free(r.scratch);
    g.free(r.h_cpos);
    g.free(r.h_crings);
    g.free(r.cand);
    g.free(r.gh);
}
fn attempt(fail_index: usize, fail_read: ?usize) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index });
    const g = f.allocator();
    var r: Resources = .{};
    var o: Ops = .{ .g = g, .fail_read = fail_read };
    if (glm.dflash_load_owner.acquire(g, &r, &o, Ops, .{
        .book = 16,
        .ctx = 8,
        .cand = 32,
        .hidden = 2,
        .gh = 16,
        .kind = .dflash2,
        .confidence = false,
        .block_size = 8,
    })) |_| {
        try t.expect(fail_index >= 7 and fail_read == null);
        try t.expect(f.allocated_bytes > f.freed_bytes);
        release(g, r);
    } else |err| {
        try t.expect(err == error.OutOfMemory or err == error.BadBook);
    }
    try t.expectEqual(f.allocated_bytes, f.freed_bytes);
}

test "glm dflash: every allocation failure releases earlier tail resources" {
    for (0..8) |i| try attempt(i, null);
}

test "glm dflash: either book read failure releases its allocated storage" {
    try attempt(99, 0);
    try attempt(99, 1);
}

test "glm dflash: every loaded symbol has a measured launch record" {
    inline for (@typeInfo(glm.dflash.Kernels).@"struct".field_names) |name| {
        // `add` (DSpark's no-conv residual) has only the GLM symbol; its record is checked in the resolve test.
        if (comptime std.mem.eql(u8, name, "add")) continue;
        const symbol = "df_" ++ name;
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            try t.expectEqualStrings("glm_dflash", r.launch.image);
            for (r.usage) |u| try t.expect(u.regs > 0);
        };
        try t.expect(found);
    }
}

test "glm dflash: module declarations analyze" {
    t.refAllDecls(glm.dflash);
    t.refAllDecls(glm.dflash_block);
    t.refAllDecls(glm.dflash_load);
    t.refAllDecls(glm.dflash_select);
    t.refAllDecls(glm.cuda_taps);
}

test "glm dflash: Flash and GLM geometry has distinct runtime block dimensions" {
    const flash: glm.dflash.Geometry = .{};
    const dsa: glm.dflash.Geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1 };
    try t.expectEqual(@as(usize, 6144), dsa.hidden);
    try t.expectEqual(@as(usize, 6), dsa.layers);
    try t.expectEqual(@as(usize, 64), dsa.heads);
    try t.expectEqual(@as(usize, 8), dsa.kv_heads);
    try t.expectEqual(@as(usize, 128), dsa.head_dim);
    try t.expectEqual(@as(usize, 12288), dsa.ff);
    try t.expectEqual(@as(usize, 6144), flash.qkvWidth());
    try t.expectEqual(@as(usize, 10240), dsa.qkvWidth());
    try t.expectEqual(@as(usize, 4), flash.heads / flash.kv_heads);
    try t.expectEqual(@as(usize, 8), dsa.heads / dsa.kv_heads);
    try t.expectEqual(@as(usize, 40_960), flash.tapRowBytes());
    try t.expectEqual(@as(usize, 73_728), dsa.tapRowBytes());
    try t.expectEqual(@as(usize, 1024), flash.convWidth());
    try t.expectEqual(@as(usize, 1536), dsa.convWidth());
    const flash_plan = try glm.dflash_block.Plan.init(flash, 2);
    const dsa_plan = try glm.dflash_block.Plan.init(dsa, 2);
    try t.expectError(error.DrafterGeometryUnsupported, glm.dflash.kernelShape(.{ .heads = 48, .kv_heads = 8 }));
    try t.expectError(error.DrafterGeometryUnsupported, glm.dflash_block.Plan.init(.{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 4 }, 2));
    try t.expectEqual(@as(usize, 16), flash_plan.rows);
    try t.expectEqual(@as(usize, 16), dsa_plan.rows);
    try t.expectEqual(@as(usize, 4), flash_plan.query_heads_per_kv);
    try t.expectEqual(@as(usize, 8), dsa_plan.query_heads_per_kv);
    try t.expectEqual(@as(usize, 10240), dsa_plan.qkv_width);
    try t.expectEqual(@as(u64, 41_943_040), flash.ringBytes());
    try t.expectEqual(@as(u64, 50_331_648), dsa.ringBytes());
    const flash_shape = try glm.dflash.kernelShape(flash);
    const dsa_shape = try glm.dflash.kernelShape(dsa);
    try t.expectEqual(@as(usize, 2), glm.dflash.instantiated_shapes.len);
    try t.expectEqual(@as(usize, 4), glm.dflash.instantiated_shapes[0].heads_per_kv);
    try t.expectEqual(@as(usize, 8), glm.dflash.instantiated_shapes[1].heads_per_kv);
    try t.expectEqual(@as(usize, 4), flash_shape.heads_per_kv);
    try t.expectEqual(@as(usize, 8), dsa_shape.heads_per_kv);
    try t.expectEqual(@as(usize, 2048), flash_shape.window);
    try t.expectEqual(@as(usize, 8), flash_shape.kv_heads);
    try t.expectEqual(@as(u32, 256), flash_shape.attention_threads);
    try t.expectEqual(@as(u32, 64), flash_shape.context_threads);
    try t.expectEqual(@as(usize, 16), flash_shape.attention_chunks);
    try t.expectEqual(flash_shape.attention_threads, dsa_shape.attention_threads);
    try t.expectEqual(flash_shape.context_threads, dsa_shape.context_threads);
    try t.expectEqual(flash_shape.attention_chunks, dsa_shape.attention_chunks);
    try t.expectEqual(flash_shape.kv_heads, dsa_shape.kv_heads);
    try t.expectEqual(flash_shape.block_rows, dsa_shape.block_rows);
    try t.expectEqual(@as(?i32, 5), glm.cuda_taps.Taps.slotOf(dsa.taps, 75));
    try t.expectEqual(@as(?i32, null), glm.cuda_taps.Taps.slotOf(dsa.taps, 14));
    try t.expectError(error.DrafterConfigUnexpected, glm.dflash_load.checkConfigBytes(t.allocator, "{\"architectures\":[\"DFlash2DraftModel\"],\"model_type\":\"qwen3\",\"hidden_size\":4096,\"dflash_config\":{}}", dsa));
}

test "glm dflash: Flash config needs only fields accepted by the base loader" {
    const flash: glm.dflash.Geometry = .{};
    const bytes = "{\"hidden_size\":4096,\"num_hidden_layers\":5,\"num_attention_heads\":32,\"num_key_value_heads\":8,\"intermediate_size\":12288,\"vocab_size\":154880,\"dflash_config\":{\"block_size\":8,\"mask_token_id\":154856,\"selector_top_k\":16,\"selector_rank\":256,\"conv_kernel_size\":2,\"conv_group_size\":16,\"target_layer_ids\":[5,14,24,33,42]}}";
    try glm.dflash_load.checkConfigBytes(t.allocator, bytes, flash);
}

test "glm dflash: geometry selects plain GLM taps and averaged Flash taps" {
    var flash: glm.dflash.Drafter = undefined;
    flash.geometry = .{};
    flash.k.tap = undefined;
    var glm53: glm.dflash.Drafter = undefined;
    glm53.geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1 };
    glm53.k.tap = undefined;
    try t.expectEqual(@as(usize, 4), flash.taps(false).streams);
    try t.expectEqual(@as(usize, 1), glm53.taps(false).streams);
    try t.expect(flash.taps(false).flash_abi);
    try t.expect(!glm53.taps(false).flash_abi);
}

test "glm dflash: convolution host oracle covers current and prior block rows" {
    const g: glm.dflash.Geometry = .{ .hidden = 2, .layers = 1, .heads = 1, .kv_heads = 1, .head_dim = 2, .ff = 2, .taps = &.{0}, .window = 2, .block_size = 2, .conv_group = 1, .conv_kernel = 2 };
    const x = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080 };
    const dyn = [_]u16{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const base = [_]u16{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 };
    var out = [_]u16{ 0, 0, 0, 0 };
    try glm.dflash.convReference(g, &x, &dyn, &base, &out, 2, 0, false);
    try t.expectEqualSlices(u16, &.{ 0x3f80, 0x4000, 0x4080, 0x40c0 }, &out);
}

test "glm dflash: synthetic drafter safetensors header is parsed without changing bytes" {
    const metadata = "{\"fc.weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]},\"norm.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[16,24]}}";
    const payload = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };
    var synthetic: std.ArrayList(u8) = .empty;
    defer synthetic.deinit(t.allocator);
    var header_len: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len, metadata.len, .little);
    try synthetic.appendSlice(t.allocator, &header_len);
    try synthetic.appendSlice(t.allocator, metadata);
    try synthetic.appendSlice(t.allocator, &payload);
    const data_offset: u64 = 8 + metadata.len;
    var header = try glm.dflash_load.parseSafetensorsHeader(t.allocator, synthetic.items[8 .. 8 + metadata.len], data_offset);
    defer header.deinit();
    try t.expectEqual(@as(usize, 2), header.entries.len);
    try t.expectEqualStrings("fc.weight", header.entries[0].name);
    try t.expectEqual(data_offset, header.entries[0].off);
    try t.expectEqual(@as(u64, 16), header.entries[0].n);
    try t.expectEqualStrings("norm.weight", header.entries[1].name);
    try t.expectEqual(data_offset + 16, header.entries[1].off);
    try t.expectEqualSlices(u8, &payload, synthetic.items[8 + metadata.len ..]);
    try t.expectError(error.DrafterNotBf16, glm.dflash_load.parseSafetensorsHeader(t.allocator, "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,2]}}", 0));
}

test "glm dflash: every launch record resolves to an exported Flash or GLM symbol" {
    t.refAllDecls(registry);
    const symbols = [_][]const u8{
        "df_tap",     "df_block_qkv",     "df_ctx_kv",     "df_conv",          "df_swiglu",   "df_attn_part",     "df_attn_merge",       "df_sum2",           "df_top16_part", "df_top16_merge",    "df_gsum",
        "df_tap_glm", "df_block_qkv_glm", "df_ctx_kv_glm", "df_ctx_kv_h8_glm", "df_conv_glm", "df_attn_part_glm", "df_attn_part_h8_glm", "df_attn_merge_glm", "df_sum2_glm",   "df_top16_part_glm", "df_top16_merge_glm", "df_add_glm", "df_ctx_kv_stage_glm", "df_ring_commit_glm",
    };
    var checked: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "glm_dflash")) continue;
        checked += 1;
        try t.expectEqualStrings(record.launch.name, record.launch.symbol);
        var found = false;
        for (symbols) |symbol| found = found or std.mem.eql(u8, symbol, record.launch.symbol);
        try t.expect(found);
    }
    try t.expectEqual(@as(usize, 25), checked);
    const flash: glm.dflash.Geometry = .{};
    const dsa: glm.dflash.Geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1 };
    for (@typeInfo(glm.dflash.Kernels).@"struct".field_names) |field| {
        const fs = try glm.dflash.Kernels.symbolFor(field, flash);
        const ds = try glm.dflash.Kernels.symbolFor(field, dsa);
        var flash_found = false;
        var dsa_found = false;
        for (symbols) |symbol| {
            flash_found = flash_found or std.mem.eql(u8, symbol, fs);
            dsa_found = dsa_found or std.mem.eql(u8, symbol, ds);
        }
        try t.expect(flash_found);
        try t.expect(dsa_found);
    }
}

test "glm dflash: draft embed hook path typechecks against a vocab-sharded mock" {
    const Head = struct { draft: glm.q4.Q4 };
    const Hook = struct {
        calls: usize = 0,
        pub fn embedDraft(self: *@This(), ids: u64, out: u64, rows: usize, width: usize) !void {
            _ = ids;
            _ = out;
            try t.expectEqual(@as(usize, 2), rows);
            try t.expectEqual(@as(usize, 6144), width);
            self.calls += 1;
        }
    };
    const Engine = struct {
        d: *const cuda.Driver,
        stream: cuda.Stream,
        spans: []glm.shard.Span,
        share: glm.shard.Share,
        group: ?*const cuda.tp_group.Group,
        lib: *const cuda.nccl.Library,
        comm: cuda.nccl.Comm,
        m: struct { glue: glm.glue.Kernels, mm: glm.mm.Kernels, q4: glm.q4.Kernels },
        w: struct { embed: u64, mtp: ?Head },
        calls: usize = 0,
        pub fn embedDraft(self: *@This(), ids: u64, out: u64, rows: usize, width: usize) !void {
            _ = ids;
            _ = out;
            _ = rows;
            _ = width;
            self.calls += 1;
        }
        pub fn embedDraftUpload(self: *@This(), rows: usize) !void {
            _ = rows;
            self.calls += 1;
        }
        pub fn embedDraftDevice(self: *@This(), out: u64, rows: usize, width: usize) !void {
            _ = out;
            _ = rows;
            _ = width;
            self.calls += 1;
        }
    };
    var dr: glm.dflash.Drafter = undefined;
    dr.geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1, .down4 = true };
    dr.ids = 154856;
    dr.x = 1;
    var layer_weights: [1]glm.dflash.LayerW = undefined;
    layer_weights[0].down4 = .{ .w = 0, .s = 0, .b = 0, .n = 6144, .k = 6144, .npad = 6144 };
    dr.layers = &layer_weights;
    var hook: Hook = .{};
    var instantiate = false;
    @atomicStore(bool, &instantiate, false, .monotonic);
    if (instantiate) try glm.dflash_block.embedTarget(&dr, &hook, 2, 6144);

    var engine: Engine = undefined;
    if (instantiate) try glm.dflash_block.pass(&dr, &engine, 1, false);
    if (instantiate) try glm.dflash_block.passDevice(&dr, &engine, 1, false);
    if (instantiate) try glm.dflash_block.ingestPass(&dr, &engine, 1);
}

test "glm dflash: device embed fails closed without the device hook" {
    const Engine = struct {
        stream: cuda.Stream,
        m: struct { glue: glm.glue.Kernels, mm: glm.mm.Kernels },
        w: struct { embed: u64 },
    };
    var dr: glm.dflash.Drafter = undefined;
    dr.geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1 };
    dr.ids = 154856;
    dr.x = 1;
    var engine: Engine = undefined;
    try t.expectError(error.DraftGraphNeedsShardHook, glm.dflash_block.embedTargetDevice(&dr, &engine, 2, 6144));
}

test "glm dflash: vocab-sharded target fails closed without the embed hook" {
    const Engine = struct {
        stream: cuda.Stream,
        m: struct { glue: glm.glue.Kernels, mm: glm.mm.Kernels },
        w: struct { embed: u64 },
    };
    var dr: glm.dflash.Drafter = undefined;
    dr.geometry = .{ .hidden = 6144, .layers = 6, .heads = 64, .kv_heads = 8, .head_dim = 128, .ff = 12288, .rope_theta = 1e6, .taps = &.{ 5, 19, 33, 47, 61, 75 }, .window = 2048, .tap_streams = 1 };
    dr.ids = 154856;
    dr.x = 1;
    var engine: Engine = undefined;
    try t.expectError(error.DrafterEmbedNeedsShardHook, glm.dflash_block.embedTarget(&dr, &engine, 2, 6144));
}

test "glm dflash: Flash defaults keep the base kernel ABI" {
    try t.expect((glm.dflash.Geometry{}).flashKernelAbi());
    try t.expect(!(glm.dflash.Geometry{ .hidden = 6144 }).flashKernelAbi());
    try t.expect(!(glm.dflash.Geometry{ .tap_streams = 1 }).flashKernelAbi());
}
