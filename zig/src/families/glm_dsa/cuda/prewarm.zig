// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");

pub const Prompt = struct { name: []const u8, ids: []const u32 };
pub const Report = struct { warmed: u32 = 0, skipped: u32 = 0, rows: u64 = 0 };

pub fn deinit(gpa: std.mem.Allocator, prompts: []Prompt) void {
    for (prompts) |p| { gpa.free(p.name); gpa.free(p.ids); }
    gpa.free(prompts);
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) ![]Prompt {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always }) catch return error.PrewarmBadJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.PrewarmBadJson;
    const list = parsed.value.object.get("prompts") orelse return error.PrewarmBadJson;
    if (list != .array) return error.PrewarmBadJson;
    if (list.array.items.len > 16) return error.PrewarmTooMany;
    var out = try gpa.alloc(Prompt, list.array.items.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |p| { gpa.free(p.name); gpa.free(p.ids); }
        gpa.free(out);
    }
    for (list.array.items) |item| {
        if (item != .object) return error.PrewarmBadJson;
        const name_v = item.object.get("name") orelse return error.PrewarmBadJson;
        const ids_v = item.object.get("ids") orelse return error.PrewarmBadJson;
        if (name_v != .string or ids_v != .array) return error.PrewarmBadJson;
        for (ids_v.array.items) |v| {
            if (v != .integer or v.integer < 0) return error.PrewarmBadJson;
            if (v.integer > std.math.maxInt(u32)) return error.PrewarmBadId;
        }
        const name = try gpa.dupe(u8, name_v.string);
        errdefer gpa.free(name);
        const ids = try gpa.alloc(u32, ids_v.array.items.len);
        errdefer gpa.free(ids);
        for (ids_v.array.items, 0..) |v, i| {
            ids[i] = @intCast(v.integer);
        }
        out[made] = .{ .name = name, .ids = ids };
        made += 1;
    }
    return out;
}

pub fn run(adapter: anytype, prompts: []const Prompt, tier: anytype, vocab: u32) !Report {
    if (!adapter.resident) return error.PrewarmNeedsResident;
    for (prompts) |p| for (p.ids) |id| { if (id >= vocab) return error.PrewarmBadId; };
    var report: Report = .{};
    if (@TypeOf(tier) == void) {
        var warmed = false;
        for (prompts) |p| {
            if (p.ids.len < 512) { report.skipped += 1; std.log.info("prewarm {s}: skipped (fewer than 512 tokens)", .{p.name}); continue; }
            if (warmed) { report.skipped += 1; std.log.info("prewarm {s}: skipped (needs the radix park tier)", .{p.name}); continue; }
            try adapter.warm(p.ids);
            warmed = true;
            report.warmed += 1;
            report.rows += p.ids.len / 512 * 512;
        }
    } else {
        for (prompts) |p| {
            if (p.ids.len < 512) { report.skipped += 1; continue; }
            try tier.park(adapter, p.name, p.ids);
            report.warmed += 1;
            report.rows += p.ids.len / 512 * 512;
        }
        for (prompts) |p| if (p.ids.len >= 512) { try adapter.warm(p.ids); break; };
    }
    return report;
}

test "prewarm parses, validates, skips short prompts and warms one resident prompt" {
    const gpa = std.testing.allocator;
    const prompts = try parse(gpa, "{\"prompts\":[{\"name\":\"system\",\"ids\":[1,2]},{\"name\":\"tools\",\"ids\":[3]}]}");
    defer deinit(gpa, prompts);
    const Fake = struct { resident: bool = true, calls: u32 = 0, fn warm(self: *@This(), ids: []const u32) !void { _ = ids; self.calls += 1; } };
    var fake: Fake = .{};
    const report = try run(&fake, prompts, {}, 10);
    try std.testing.expectEqual(@as(u32, 0), report.warmed);
    try std.testing.expectEqual(@as(u32, 2), report.skipped);
    try std.testing.expectEqual(@as(u32, 0), fake.calls);
}

test "prewarm parses valid ids and rejects malformed JSON and unrepresentable ids" {
    const prompts = try parse(std.testing.allocator, "{\"prompts\":[{\"name\":\"agent-sys\",\"ids\":[0,17,4294967295]}]}");
    defer deinit(std.testing.allocator, prompts);
    try std.testing.expectEqualStrings("agent-sys", prompts[0].name);
    try std.testing.expectEqualSlices(u32, &.{ 0, 17, std.math.maxInt(u32) }, prompts[0].ids);
    try std.testing.expectError(error.PrewarmBadJson, parse(std.testing.allocator, "not json"));
    try std.testing.expectError(error.PrewarmBadId, parse(std.testing.allocator, "{\"prompts\":[{\"name\":\"x\",\"ids\":[4294967296]}]}"));
}

test "prewarm void tier warms only the first eligible prompt" {
    const Fake = struct { resident: bool = true, warmed: usize = 0, fn warm(self: *@This(), ids: []const u32) !void { self.warmed = ids.len; } };
    const ids: [512]u32 = @splat(1);
    const prompts = [_]Prompt{ .{ .name = "short", .ids = ids[0..100] }, .{ .name = "first", .ids = &ids }, .{ .name = "second", .ids = &ids } };
    var fake: Fake = .{};
    const report = try run(&fake, &prompts, {}, 2);
    try std.testing.expectEqual(@as(u32, 1), report.warmed);
    try std.testing.expectEqual(@as(u32, 2), report.skipped);
    try std.testing.expectEqual(@as(u64, 512), report.rows);
    try std.testing.expectEqual(@as(usize, 512), fake.warmed);
}

test "prewarm tier parks all eligible prompts and leaves the first resident" {
    const Fake = struct {
        resident: bool = true,
        first: []const u32 = &.{},
        calls: u32 = 0,
        fn warm(self: *@This(), ids: []const u32) !void { self.first = ids; }
    };
    const Tier = struct {
        calls: *u32,
        fn park(t: @This(), _: anytype, _: []const u8, _: []const u32) !void { t.calls.* += 1; }
    };
    const ids: [512]u32 = @splat(1);
    const prompts = [_]Prompt{ .{ .name = "a", .ids = &ids }, .{ .name = "b", .ids = &ids } };
    var fake: Fake = .{};
    var parks: u32 = 0;
    const report = try run(&fake, &prompts, Tier{ .calls = &parks }, 2);
    try std.testing.expectEqual(@as(u32, 2), report.warmed);
    try std.testing.expectEqual(@as(u32, 2), parks);
    try std.testing.expectEqualSlices(u32, &ids, fake.first);
}

test "prewarm rejects too many entries and bad ids" {
    try std.testing.expectError(error.PrewarmTooMany, parse(std.testing.allocator, "{\"prompts\":[{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]},{\"name\":\"x\",\"ids\":[]}]}") );
    const p = [_]Prompt{.{ .name = "bad", .ids = &.{ 1, 10 } }};
    const Fake = struct {
        resident: bool = true,
        fn warm(_: *@This(), _: []const u32) !void {}
    };
    var fake: Fake = .{};
    try std.testing.expectError(error.PrewarmBadId, run(&fake, &p, {}, 10));
}
