// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP10 dump writer host contract; details live beside the implementation.
const dump = @import("dump");
const std = @import("std");
const st = @import("safetensors");
test "GLM-5.3 dump writer host contracts" {
    std.testing.refAllDecls(dump);
    std.testing.refAllDecls(dump.Tensor);
    std.testing.refAllDecls(dump.GateRow);
    const a = std.testing.allocator;
    const name = try dump.key(a, "mlp_in", 7, 2);
    defer a.free(name);
    try std.testing.expectEqualStrings("mlp_in.L07.2", name);
    const final_name = try dump.key(a, "final_logits", 3, 0);
    defer a.free(final_name);
    try std.testing.expectEqualStrings("final_logits.L99.0", final_name);

    const ids = [_]i32{ 3, 9, 12, 1, 4, 7 };
    const logp = [_]f32{ -0.125, -1.5, -2.25, -0.5 };
    const tensors = [_]dump.Tensor{
        .{ .name = "ids", .dtype = "I32", .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&ids) },
        .{ .name = "logq", .dtype = "F32", .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&logp) },
    };
    const encoded = try dump.encode(a, &tensors);
    defer a.free(encoded);
    const fixture = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/glm53/wp10/python_golden.safetensors", a, .limited(1 << 20));
    defer a.free(fixture);
    const hlen: usize = @intCast(std.mem.readInt(u64, encoded[0..8], .little));
    const f_hlen: usize = @intCast(std.mem.readInt(u64, fixture[0..8], .little));
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const parsed = try st.parseHeader(arena.allocator(), encoded[8 .. 8 + hlen], encoded.len - 8 - hlen);
    const golden = try st.parseHeader(arena.allocator(), fixture[8 .. 8 + f_hlen], fixture.len - 8 - f_hlen);
    try std.testing.expectEqual(parsed.count(), golden.count());
    for ([_][]const u8{ "ids", "logq" }) |key| {
        const got = parsed.get(key).?;
        const want = golden.get(key).?;
        try std.testing.expectEqual(got.dtype, want.dtype);
        try std.testing.expectEqualSlices(usize, got.shape[0..got.rank], want.shape[0..want.rank]);
        try std.testing.expectEqualSlices(u8, encoded[8 + hlen + got.begin .. 8 + hlen + got.end], fixture[8 + f_hlen + want.begin .. 8 + f_hlen + want.end]);
    }

    const row: dump.GateRow = .{ .ids = @splat(5), .logp = @splat(-1), .lse = 3, .mass_outside = 0.25, .argmax = 7, .window_id = 2, .pos = 4096 };
    const gate = try dump.gateTensors(a, &.{row});
    defer dump.freeGateTensors(a, gate);
    try std.testing.expectEqual(@as(usize, 7), gate.len);
    try std.testing.expectEqualStrings("mass_outside", gate[3].name);
    try std.testing.expectEqualSlices(usize, &.{ 1, 64 }, gate[0].shape);
    try std.testing.expectEqual(@as(usize, 8), gate[6].bytes.len);
    const with_meta = try dump.encodeWithMetadata(a, &tensors, "tag", "parity");
    defer a.free(with_meta);
    const with_meta_len: usize = @intCast(std.mem.readInt(u64, with_meta[0..8], .little));
    try std.testing.expect(std.mem.indexOf(u8, with_meta[8 .. 8 + with_meta_len], "\"__metadata__\":{\"tag\":\"parity\"}") != null);
}
