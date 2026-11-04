// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! End-to-end parity for suffix lookup and the suffix automaton proposer.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const ZNgram = @import("zngram.zig").ZNgram;

const prompt = [_]u32{ 11, 12, 13, 11, 12, 13, 11, 12, 13, 11, 12, 13 };
const continuation = [_]u32{ 21, 22, 23 };

const Result = struct { tokens: []u32, accepted: u64, rounds: u64 };

fn run(comptime UseZngram: bool) !Result {
    const gpa = std.testing.allocator;
    var cfg = try Config.init(gpa, .{ .exact_width = 8, .first_copy_rows = 8, .copy_match = 6, .copy_drafts = true, .gpu_tokens = false, .batch_rows = 8, .max_streams = 1 }, 8, 7);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 0, .probe_prompt = prompt.len, .pattern = &continuation };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var lookup: SuffixLookup = undefined;
    var zngram: ZNgram = undefined;
    const proposer = if (UseZngram) blk: {
        zngram = try ZNgram.init(gpa, .{ .min_match = 6, .max_draft = 7 });
        break :blk zngram.proposer();
    } else blk: {
        lookup = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 6 });
        break :blk lookup.proposer();
    };
    defer if (UseZngram) zngram.deinit() else lookup.deinit();
    var stream = try Stream.init(gpa, .{ .id = "zngram-test", .prompt = &prompt, .max_new = 24, .eos = &.{96}, .drafts = true, .proposer = proposer });
    defer stream.deinit(gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .tokens = try gpa.dupe(u32, stream.emitted()), .accepted = stream.accepted, .rounds = target.rounds };
}

test "zngram and suffix lookup produce equal target tokens and land drafts" {
    const lookup = try run(false);
    defer std.testing.allocator.free(lookup.tokens);
    const zngram = try run(true);
    defer std.testing.allocator.free(zngram.tokens);
    try std.testing.expectEqualSlices(u32, lookup.tokens, zngram.tokens);
    try std.testing.expect(zngram.accepted > 0);
    try std.testing.expect(zngram.rounds < zngram.tokens.len);
}
