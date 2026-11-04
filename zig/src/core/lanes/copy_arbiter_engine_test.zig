// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! End-to-end parity and pricing for per-round copy arbitration.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const ZNgram = @import("zngram.zig").ZNgram;

const prompt = [_]u32{ 11, 12, 13, 11, 12, 13, 11, 12, 13, 11, 12, 13 };
const continuation = [_]u32{ 21, 22, 23 };
const costs = [_]Cost{
    .{ .width = 1, .ms = 5.8 }, .{ .width = 2, .ms = 6.6 }, .{ .width = 3, .ms = 7.4 }, .{ .width = 4, .ms = 8.2 },
    .{ .width = 5, .ms = 9.0 }, .{ .width = 6, .ms = 9.8 }, .{ .width = 7, .ms = 10.6 }, .{ .width = 8, .ms = 11.4 },
};

const Result = struct { tokens: []u32, suffix_rounds: u64 };

fn run(arbitrate: bool, drafts: bool, cold_copies: bool) !Result {
    const gpa = std.testing.allocator;
    var cfg = try Config.init(gpa, .{
        .exact_width = 8,
        .first_copy_rows = 8,
        .copy_match = 6,
        .copy_drafts = true,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 7,
        .draft_streams = true,
        .gpu_tokens = false,
        .batch_rows = 8,
        .max_streams = 1,
        .verify_widths = true,
        .draft_probabilities = true,
        .copy_arbitrate = arbitrate,
        .copy_costs = &costs,
    }, 8, 7);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 0, .probe_prompt = prompt.len, .pattern = &continuation };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var zngram = try ZNgram.init(gpa, .{ .min_match = 6, .max_draft = 7 });
    defer zngram.deinit();
    if (cold_copies) {
        zngram.landed = 0.01;
        zngram.missed = 10;
    }
    var stream = try Stream.init(gpa, .{ .id = "copy-arbiter-test", .prompt = &prompt, .max_new = 24, .eos = &.{96}, .drafts = drafts, .proposer = zngram.proposer() });
    defer stream.deinit(gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .tokens = try gpa.dupe(u32, stream.emitted()), .suffix_rounds = zngram.stats.suffix_rounds };
}

test "copy arbitration keeps every token" {
    const serial = try run(false, false, false);
    defer std.testing.allocator.free(serial.tokens);
    const off = try run(false, true, false);
    defer std.testing.allocator.free(off.tokens);
    const on = try run(true, true, false);
    defer std.testing.allocator.free(on.tokens);
    const cold = try run(true, true, true);
    defer std.testing.allocator.free(cold.tokens);
    try std.testing.expectEqualSlices(u32, serial.tokens, off.tokens);
    try std.testing.expectEqualSlices(u32, off.tokens, on.tokens);
    try std.testing.expectEqualSlices(u32, off.tokens, cold.tokens);
    try std.testing.expect(off.suffix_rounds > 0);
}

test "a copy that rarely lands loses rounds to the drafter" {
    const off = try run(false, true, false);
    defer std.testing.allocator.free(off.tokens);
    const cold = try run(true, true, true);
    defer std.testing.allocator.free(cold.tokens);
    try std.testing.expect(cold.suffix_rounds < off.suffix_rounds);
}

test "copy arbitration is off by default and requires copy drafts" {
    const gpa = std.testing.allocator;
    var defaults = try Config.init(gpa, .{ .exact_width = 8 }, 8, 7);
    defer defaults.deinit(gpa);
    try std.testing.expect(!defaults.copy_arbitrate and !defaults.k.copy_arbitrate and defaults.copy_costs.empty());
    const gb10 = [_]Cost{
        .{ .width = 1, .ms = 88 }, .{ .width = 2, .ms = 104.1 }, .{ .width = 3, .ms = 120.2 }, .{ .width = 4, .ms = 130.3 },
        .{ .width = 5, .ms = 145.1 }, .{ .width = 6, .ms = 159.8 }, .{ .width = 7, .ms = 175.8 }, .{ .width = 8, .ms = 191.8 },
        .{ .width = 9, .ms = 212.4 }, .{ .width = 10, .ms = 233.1 }, .{ .width = 11, .ms = 253.7 }, .{ .width = 12, .ms = 274.3 },
        .{ .width = 13, .ms = 288.8 }, .{ .width = 14, .ms = 303.2 }, .{ .width = 15, .ms = 317.7 }, .{ .width = 16, .ms = 332.2 },
    };
    var enabled = try Config.init(gpa, .{ .exact_width = 8, .copy_arbitrate = true, .copy_drafts = true, .copy_costs = &gb10 }, 8, 7);
    defer enabled.deinit(gpa);
    try std.testing.expect(enabled.copy_arbitrate and enabled.k.copy_arbitrate);
    try std.testing.expectEqual(@as(?f64, 191.8), enabled.copy_costs.get(8));
    var disabled = try Config.init(gpa, .{ .exact_width = 8, .copy_arbitrate = true, .copy_drafts = false, .copy_costs = &gb10 }, 8, 7);
    defer disabled.deinit(gpa);
    try std.testing.expect(!disabled.copy_arbitrate and !disabled.k.copy_arbitrate);
}
