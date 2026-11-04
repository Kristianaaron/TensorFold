// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP10 fp32 candidate and gate reductions host contract.
const sampler = @import("sampler");
const std = @import("std");
test "GLM-5.3 fp32 gate and sampling host contracts" {
    std.testing.refAllDecls(sampler);
    std.testing.refAllDecls(sampler.Head);
    std.testing.refAllDecls(sampler.Candidate);
    std.testing.refAllDecls(sampler.Lse);
    std.testing.refAllDecls(sampler.Sampling);
    const a = std.testing.allocator;
    const r0 = [_]sampler.Candidate{ .{ .value = 3, .id = 9 }, .{ .value = 1, .id = 0 } };
    const r1 = [_]sampler.Candidate{ .{ .value = 3, .id = 2 }, .{ .value = 2, .id = 6 } };
    const candidates = try sampler.mergeCandidates(a, &.{ &r0, &r1 }, 3);
    defer a.free(candidates);
    try std.testing.expectEqualSlices(sampler.Candidate, &.{ .{ .value = 3, .id = 2 }, .{ .value = 3, .id = 9 }, .{ .value = 2, .id = 6 } }, candidates);
    const head_values = [_]f32{ 2, 3, 3, std.math.nan(f32), -std.math.inf(f32) };
    try std.testing.expectEqual(sampler.Candidate{ .value = 3, .id = 10 }, sampler.top1Ref(&head_values, 9));
    const local = try sampler.topCandidatesRef(a, &head_values, 9, 3);
    defer a.free(local);
    try std.testing.expectEqualSlices(sampler.Candidate, &.{ .{ .value = 3, .id = 10 }, .{ .value = 3, .id = 11 }, .{ .value = 2, .id = 9 } }, local);

    var radix_values: [4096]f32 = @splat(-100);
    for (0..16) |lane_zero_rank| radix_values[lane_zero_rank * 256] = 100 - @as(f32, @floatFromInt(lane_zero_rank));
    for (80..160) |i| radix_values[i] = 50; // many tied values; lower global id must win
    for (0..4096) |i| if (i % 13 == 0) {
        radix_values[i] = @as(f32, @floatFromInt(i % 37)) - 10;
    };
    for ([_]usize{ 1, 8, 9, 64, 2048 }) |k| {
        const expected = try sampler.topCandidatesRef(a, &radix_values, 700, k);
        defer a.free(expected);
        const got = try sampler.topCandidatesRadixRef(a, &radix_values, 700, k);
        defer a.free(got);
        try std.testing.expectEqualSlices(sampler.Candidate, expected, got);
    }

    const rank0 = [_]f32{ 1000, 999, -100 };
    const rank1 = [_]f32{ 998, 997 };
    const parts = [_]sampler.Lse{ sampler.partialLse(&rank0), sampler.partialLse(&rank1) };
    const all = [_]f32{ 1000, 999, -100, 998, 997 };
    const whole = sampler.partialLse(&all);
    try std.testing.expectApproxEqAbs(whole.max + @log(whole.sum), sampler.mergeLse(&parts), 1e-5);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, 0.34082920607897454))), @as(u64, @bitCast(sampler.keyedUniform(5, 17, 4))));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, 0.7317128056764994))), @as(u64, @bitCast(sampler.keyedUniform(5, 17, 0))));
    try std.testing.expect(sampler.keyedUniform(5, 17, 4) != sampler.keyedUniform(5, 18, 4));
    const draw_cs = [_]sampler.Candidate{ .{ .value = 2, .id = 4 }, .{ .value = 1, .id = 8 } };
    const rule: sampler.Sampling = .{ .temperature = 0.01, .top_k = 2 };
    try std.testing.expectEqual(@as(u32, 4), try sampler.drawCandidates(a, &draw_cs, 8, 10, rule));
    try std.testing.expectEqual(try sampler.drawCandidates(a, &draw_cs, 8, 10, rule), try sampler.drawCandidates(a, &draw_cs, 8, 10, rule));

    const python_case0 = [_]sampler.Candidate{ .{ .value = 1.25, .id = 0 }, .{ .value = 1.25, .id = 2 }, .{ .value = -0.5, .id = 4 }, .{ .value = 2, .id = 8 } };
    try std.testing.expectEqual(@as(u32, 0), try sampler.drawCandidates(a, &python_case0, 5, 17, .{ .temperature = 0.7 }));
    const python_case1 = [_]sampler.Candidate{ .{ .value = 2.5, .id = 17 }, .{ .value = 1, .id = 3 }, .{ .value = 2.5, .id = 99 }, .{ .value = -1, .id = 42 }, .{ .value = 0.25, .id = 11 }, .{ .value = 2, .id = 7 }, .{ .value = 0.5, .id = 100 }, .{ .value = 1.5, .id = 5 } };
    try std.testing.expectEqual(@as(u32, 5), try sampler.drawCandidates(a, &python_case1, 123456789, 987, .{ .temperature = 1.3, .top_k = 6, .top_p = 0.82, .min_log = @log(0.04) }));
    const python_case2 = [_]sampler.Candidate{ .{ .value = 1.2, .id = 0 }, .{ .value = -0.4, .id = 1 }, .{ .value = 2.3, .id = 2 }, .{ .value = 0.8, .id = 3 }, .{ .value = 1.9, .id = 4 }, .{ .value = -1.1, .id = 5 }, .{ .value = 0.2, .id = 6 }, .{ .value = 1.5, .id = 7 }, .{ .value = 0.4, .id = 8 }, .{ .value = 2, .id = 9 }, .{ .value = -0.7, .id = 10 }, .{ .value = 1, .id = 11 } };
    try std.testing.expectEqual(@as(u32, 2), try sampler.drawNucleusCandidates(a, &python_case2, 12, 77, 2500, .{ .temperature = 0.85, .top_p = 0.91 }));
    try std.testing.expectError(error.IncompleteNucleusCandidates, sampler.drawNucleusCandidates(a, &python_case0, 100, 1, 2, .{ .temperature = 1, .top_p = 0.95 }));
    try std.testing.expectEqual(sampler.RowWindow{ .start = 4, .count = 1 }, sampler.headWindow(5, false));
    try std.testing.expectEqual(sampler.RowWindow{ .start = 0, .count = 5 }, sampler.headWindow(5, true));
    try std.testing.expectEqual(sampler.RowWindow{ .start = 4, .count = 1 }, sampler.layer77Window(5, false, true));
    try std.testing.expectEqual(sampler.RowWindow{ .start = 0, .count = 5 }, sampler.layer77Window(5, true, true));

    const shards = [_][]const f32{ &.{ 1, 2, 3 }, &.{ 4, 5 } };
    var probe_ids: [64]u32 = @splat(0);
    @memcpy(probe_ids[0..5], &[_]u32{ 0, 1, 2, 3, 4 });
    const gate = try sampler.gateRow(a, &shards, &.{ 0, 3 }, &probe_ids);
    defer a.free(gate.logp);
    try std.testing.expectEqual(@as(u32, 4), gate.argmax);
    try std.testing.expect(gate.mass_outside >= 0);
}
