// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");
const config = @import("config");
const census = @import("census");

test "synthetic GLM-5.3 index and safetensors headers are classified by payload bytes" {
    const got = try census.readModelDir(std.testing.allocator, std.testing.io, "zig/tests/fixtures/glm53/census");
    try std.testing.expectEqual(@as(u64, 8), got.routed);
    try std.testing.expectEqual(@as(u64, 12), got.backbone_fp8);
    try std.testing.expectEqual(@as(u64, 16), got.bf16_f32);
    try std.testing.expectEqual(@as(u64, 36), got.total());
}

test "real GLM-5.3 config: indexer map is read from indexer_types and MTP remains visible to Metal" {
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/glm53/config.json"));
    try std.testing.expectEqual(@as(u32, 1), c.configured_mtp);
    try std.testing.expectEqual(@as(u32, 1), c.mtp);
    try std.testing.expect(c.full.isSet(c.layers));
    try std.testing.expectEqual(@as(u32, 1), c.n_shared_experts);
    try std.testing.expect(c.norm_topk_prob);
    try std.testing.expectEqual(@as(u32, 1), c.dcp);
    try std.testing.expectEqual(config.KvFormat.bf16, c.kv);
    try std.testing.expectEqual(config.RoutedCodebook.mcg, c.routed_codebook);
    try std.testing.expectEqual(config.BackboneFormat.fp8_block, c.backbone);
    var actual: [21]u32 = undefined;
    var n: usize = 0;
    for (0..c.layers) |i| if (c.full.isSet(i)) {
        actual[n] = @intCast(i);
        n += 1;
    };
    try std.testing.expectEqual(@as(usize, actual.len), n);
    try std.testing.expectEqual([21]u32{ 0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74 }, actual);
    try std.testing.expectEqual(@as(u32, 1), config.ServingOptions.defaults(.metal).dcp);
    try std.testing.expectEqual(@as(u32, 2), config.ServingOptions.defaults(.cuda).dcp);
    var cuda_config = c;
    try cuda_config.applyServingOptions(config.ServingOptions.defaults(.cuda));
    try std.testing.expectEqual(@as(u32, 2), cuda_config.dcp);
    try config.Config.admitCuda(&cuda_config);
}

test "BF16 parent config without quantization metadata parses and is refused only by CUDA admission" {
    const gpa = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures/glm53/config.json"), .{});
    defer parsed.deinit();
    _ = parsed.value.object.orderedRemove("quantization_config");
    const json = try std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
    defer gpa.free(json);

    var c = try config.parse(gpa, json);
    try std.testing.expectEqual(config.BackboneFormat.bf16, c.backbone);
    try std.testing.expectError(error.UnsupportedBackbone, config.Config.admitCuda(&c));
}

test {
    _ = @import("share");
    _ = @import("types");
}
