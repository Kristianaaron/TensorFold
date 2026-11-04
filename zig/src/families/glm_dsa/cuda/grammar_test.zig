// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Ported test grammar scaffold from ashhart/TensorFold zig/src/core/lanes/engine_test.zig, Apache-2.0; GLM masks run through the real lane loop.
const std = @import("std");
const lanes = @import("lanes");
const adapter_mod = @import("backend.zig");
const follow = @import("glm_follow");
const be = lanes.backend;
const Stream = lanes.Stream;
const gpa = std.testing.allocator;

// Ported from ashhart/TensorFold zig/src/core/lanes/engine_test.zig Classes, Apache-2.0.
const Classes = struct {
    length: u32 = 32,
    after: ?u32 = null,
    at: u32 = 0,
    done: bool = false,

    fn rules(c: *Classes) lanes.grammar.Rules {
        return .{ .ptr = c, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .drafts = draftsFn, .free = freeFn } };
    }
    fn cast(ptr: *anyopaque) *Classes { return @ptrCast(@alignCast(ptr)); }
    fn allows(c: *const Classes, token: u32) bool {
        if (c.done) return false;
        if (c.at == c.length) return token == 96;
        return token != 96 and token % 3 == c.at % 3;
    }
    fn acceptFn(ptr: *anyopaque, token: u32) anyerror!bool {
        const c = cast(ptr);
        if (!c.allows(token)) return false;
        if (c.at == c.length) c.done = true else c.at += 1;
        return true;
    }
    fn rollbackFn(ptr: *anyopaque, count: usize) anyerror!void {
        const c = cast(ptr);
        for (0..count) |_| if (c.done) { c.done = false; } else { c.at -= 1; };
    }
    fn terminatedFn(ptr: *anyopaque) bool { return cast(ptr).done; }
    fn fillFn(ptr: *anyopaque, bits: []u32) anyerror!void {
        const c = cast(ptr);
        @memset(bits, 0);
        for (0..97) |i| {
            if (c.allows(@intCast(i))) bits[i / 32] |= @as(u32, 1) << @intCast(i % 32);
        }
    }
    fn draftsFn(_: *anyopaque, _: []u32) anyerror!usize { return 0; }
    fn freeFn(_: *anyopaque) void {}
};

const Frames = struct {
    masks: usize = 0,
    fn send(ptr: *anyopaque, header: follow.Header, _: []const u32) anyerror!void {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        if (header.kind == @backingInt(follow.Kind.masks)) f.masks += 1;
    }
};

const FakeModel = struct {
    pos: usize = 0,
    head_rows: usize = 1,
    seed: u64 = 0,
    rule: @import("glm_sampler").Sampling = .{ .temperature = 0 },
    prompt_len: usize = 0,
    history: [256]u32 = @splat(0),
    history_n: usize = 0,
    window: [follow.max_decode_rows]u32 = @splat(0),
    masks: [follow.max_decode_rows * 4]u32 = @splat(0),
    mask_rows: usize = 0,
    mask_words: usize = 0,
    mask_calls: usize = 0,
    head_token: u32 = 0,
    offered: usize = 0,
    trimmed: usize = 0,
    rejected: usize = 0,
    violations: usize = 0,
    draft_index: usize = 0,
    last_offered: usize = 0,
    after: ?u32 = null,
    drafter: Drafter = .good,

    const Drafter = enum { good, old_off_by_one, forbidden, allowed_wrong };

    fn target(e: *const FakeModel, position: usize) u32 {
        const start = e.prompt_len - 1;
        const index = position - start;
        if (e.after != null) {
            if (index < 3) return @intCast(4 + index);
            if (index == 3) return 91;
            if (index == 36) return 96;
            return @intCast(3 + (index - 4) % 3);
        }
        if (index == 32) return 96;
        return @intCast(3 + index % 3);
    }
    fn allowed(e: *const FakeModel, row: usize, token: u32) bool {
        if (e.mask_rows == 0 or row >= e.mask_rows) return true;
        return (e.masks[row * e.mask_words + @as(usize, @intCast(token / 32))] >> @intCast(token % 32)) & 1 != 0;
    }
    fn choose(e: *const FakeModel, row: usize, position: usize) u32 {
        const target_token = e.target(position);
        var best: u32 = 0;
        var best_score: i32 = std.math.minInt(i32);
        for (0..97) |i| {
            const token: u32 = @intCast(i);
            if (!e.allowed(row, token)) continue;
            const score: i32 = if (token == target_token) 100 else @intCast((i * 13 + position) % 50);
            if (score > best_score) { best_score = score; best = token; }
        }
        return best;
    }
    pub fn setMasks(e: *FakeModel, bits: []const u32, rows: usize) !void {
        e.mask_calls += 1;
        if (bits.len % rows != 0) return error.BadMask;
        e.mask_rows = rows;
        e.mask_words = bits.len / rows;
        @memcpy(e.masks[0..bits.len], bits);
    }
    pub fn clearMasks(e: *FakeModel) void { e.mask_rows = 0; }
    pub fn prefill(e: *FakeModel, ids: []const u32) !u32 {
        e.prompt_len = ids.len;
        @memcpy(e.history[0..ids.len], ids);
        e.history_n = ids.len;
        e.pos += ids.len;
        e.head_token = e.choose(0, e.pos - 1);
        e.clearMasks();
        return e.head_token;
    }
    pub fn draw(e: *FakeModel, _: usize, _: usize, _: u64, out: []u32) !void { out[0] = e.head_token; }
    pub fn verify(e: *FakeModel, ids: []const u32, out: []u32) !void {
        if (e.last_offered > ids.len - 1) e.trimmed += e.last_offered - (ids.len - 1);
        @memcpy(e.window[0..ids.len], ids);
        for (ids, 0..) |token, row| {
            if (row > 0 and e.mask_rows > row - 1 and !e.allowed(row - 1, token)) e.violations += 1;
            out[row] = e.choose(row, e.pos + row);
            if (row > 0 and token != out[row - 1]) e.rejected += 1;
        }
        e.clearMasks();
    }
    pub fn keep(e: *FakeModel, rows: usize) !void {
        @memcpy(e.history[e.history_n ..][0..rows], e.window[0..rows]);
        e.history_n += rows;
        e.pos += rows;
    }
    pub fn rollback(e: *FakeModel, to: usize) !void { e.pos = to; e.history_n = to; }
    pub fn draftBlock(e: *FakeModel, pos: usize, _: usize, _: u32, out: ?[]u32, _: ?[]f64) !void {
        const drafts = out orelse return error.NoDraftOutput;
        e.last_offered = drafts.len;
        e.offered += drafts.len;
        for (drafts, 0..) |*token, i| {
            const place = e.draft_index + i + 1;
            // `pos + i` is where this proposed token lands; the logits that predict it
            // belong to the preceding row, just as in verify's row-to-token mapping.
            // This position-only target remains unchanged by rejected draft prefixes.
            const base = e.target(pos + i - 1);
            token.* = switch (e.drafter) {
                .good => if (place % 4 == 0) 96 else if (place % 7 == 0) base + 3 else base,
                .old_off_by_one => e.target(pos + i),
                .forbidden => 97,
                .allowed_wrong => if (base == 96) 96 else base + 3,
            };
        }
        e.draft_index += drafts.len;
    }
};

const Outcome = struct { tokens: []u32, model: FakeModel, adapter_matched: u64, frames: Frames };

fn run(after: ?u32, drafted: bool, with_grammar: bool, drafter: FakeModel.Drafter) !Outcome {
    var model: FakeModel = .{ .after = after, .drafter = drafter };
    var frames: Frames = .{};
    const B = adapter_mod.Adapter(FakeModel);
    var adapter = B.init(gpa, &model);
    adapter.drafting = drafted;
    adapter.send_fn = Frames.send;
    adapter.link_ctx = &frames;
    var cfg = try lanes.Config.init(gpa, adapter.facts(), 8, 7);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, adapter.backend(), clock.clock());
    defer engine.deinit();
    var proposer = try lanes.SuffixLookup.init(gpa, .{ .min_match = 1000 });
    defer proposer.deinit();
    var classes: Classes = .{ .after = after };
    var constraint: ?*lanes.grammar.Constraint = null;
    if (with_grammar) {
        constraint = try gpa.create(lanes.grammar.Constraint);
        constraint.?.* = .init(classes.rules(), 4, after);
    }
    defer if (constraint) |c| c.deinit(gpa);
    const prompt = [_]u32{ 1, 2, 3 };
    var stream = try Stream.init(gpa, .{ .id = "grammar", .prompt = &prompt, .max_new = 64, .eos = &.{96}, .drafts = drafted, .proposer = proposer.proposer(), .grammar = constraint });
    defer stream.deinit(gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .tokens = try gpa.dupe(u32, stream.emitted()), .model = model, .adapter_matched = adapter.matched_drafts, .frames = frames };
}

fn expectReplyEqual(mode: []const u8, after: ?u32, plain: []const u32, drafted: []const u32) !void {
    if (!std.mem.eql(u32, plain, drafted)) {
        var diverge = @min(plain.len, drafted.len);
        for (plain[0..diverge], drafted[0..diverge], 0..) |left, right, i| {
            if (left != right) {
                diverge = i;
                break;
            }
        }
        const first: usize = diverge -| 4;
        const plain_end: usize = @min(plain.len, diverge + 5);
        const drafted_end: usize = @min(drafted.len, diverge + 5);
        std.debug.print(
            "GLM grammar drafted reply mismatch: mode={s} after={any} first diverging reply index={d}; plain[{d}..{d}]={any}; drafted[{d}..{d}]={any}\n",
            .{ mode, after, diverge, first, plain_end, plain[first..plain_end], first, drafted_end, drafted[first..drafted_end] },
        );
    }
    try std.testing.expectEqualSlices(u32, plain, drafted);
}

test "GLM grammar masks keep drafted replies exact through lane constrain" {
    inline for (.{ @as(?u32, null), @as(?u32, 91) }) |after| {
        const plain = try run(after, false, true, .good);
        defer gpa.free(plain.tokens);
        var drafted = try run(after, true, true, .good);
        defer gpa.free(drafted.tokens);
        try expectReplyEqual("good", after, plain.tokens, drafted.tokens);
        try std.testing.expect(drafted.tokens.len > 0 and drafted.tokens[drafted.tokens.len - 1] == 96);
        const grammar_start: usize = if (after == null) 0 else 4;
        if (after != null) try std.testing.expectEqual(@as(u32, 91), drafted.tokens[3]);
        for (drafted.tokens[grammar_start..], 0..) |token, i| {
            if (token == 96) break;
            try std.testing.expectEqual(@as(u32, @intCast(i % 3)), token % 3);
        }
        try std.testing.expectEqual(@as(usize, 0), drafted.model.violations);
        try std.testing.expect(drafted.model.offered > 0);
        // Efficiency is REPORTED, not asserted (exactness is asserted above for every drafter mode). 2026-11-03: the good fake
        // drafter had 0 accepted drafts under the grammar: follow-up (does masked verify ever accept drafts?), tracked in the ledger.
        std.debug.print("GLM grammar drafting efficiency: after={any} offered={d} matched={d} trimmed={d} rejected={d}\n", .{ after, drafted.model.offered, drafted.adapter_matched, drafted.model.trimmed, drafted.model.rejected });

        inline for (.{ FakeModel.Drafter.old_off_by_one, .forbidden, .allowed_wrong }) |draft_mode| {
            const adversarial = try run(after, true, true, draft_mode);
            defer gpa.free(adversarial.tokens);
            try expectReplyEqual(@tagName(draft_mode), after, plain.tokens, adversarial.tokens);
            try std.testing.expect(adversarial.model.offered > 0);
            // Lane constrain trims grammar-violating drafts before verify (2026-11-03 gate: offered 231, trimmed 224, rejected 0),
            // so wrong drafts show up as trimmed or rejected; the output stays exact either way.
            if (draft_mode == .forbidden) {
                try std.testing.expect(adversarial.model.violations > 0); // forbidden drafts reach the window, then get trimmed
                try std.testing.expect(adversarial.model.trimmed + adversarial.model.rejected > 0);
            } else if (draft_mode == .allowed_wrong) {
                try std.testing.expectEqual(@as(usize, 0), adversarial.model.violations);
                try std.testing.expect(adversarial.model.trimmed + adversarial.model.rejected > 0);
            }
        }
    }
    const plain = try run(null, false, false, .good);
    defer gpa.free(plain.tokens);
    try std.testing.expectEqual(@as(usize, 0), plain.model.mask_calls);
    try std.testing.expectEqual(@as(usize, 0), plain.frames.masks);
}
