// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Adapted from upstream TensorFold #1381: drafter chances enable grant-following draft depths.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Config = @import("config.zig").Config;

test "granted depth follows the last verify grant and its short or full slack" {
    const empty: Table = .{};
    var rule: depth.Rule = .{ .gpa = std.testing.allocator, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    const Copy = struct { fn no(_: *anyopaque, _: *const depth.Who) bool { return false; } };
    var probe_byte: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &probe_byte, .copy = Copy.no };
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(std.testing.allocator);
    var who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .granted = 3 };
    try std.testing.expectEqual(@as(i64, 4), try rule.headDepth(&who, null, probe));
    who.grant_short = true;
    try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&who, null, probe));
}

test "lane chances stay off by default and are enabled only with draft probabilities" {
    const gpa = std.testing.allocator;
    var plain = try Config.init(gpa, .{ .exact_width = 8 }, 8, 7);
    defer plain.deinit(gpa);
    try std.testing.expect(!plain.granted_depth);
    var enabled = try Config.init(gpa, .{ .exact_width = 8, .draft_probabilities = true, .granted_depth = true }, 8, 7);
    defer enabled.deinit(gpa);
    try std.testing.expect(enabled.granted_depth and enabled.node_probabilities);
}
