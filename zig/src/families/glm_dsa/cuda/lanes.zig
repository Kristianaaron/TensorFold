// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31: zig/src/families/glm/cuda_lanes.zig (Apache-2.0). Changes: tracks prefill-only DSA cache reuse and suffix continuation.
//! Lane state for the GLM-5.3 DSA engine. Cache writes are position addressed and rollback resets position.

const std = @import("std");
const pages = @import("pages.zig");

pub const prompt_chunk_rows: u32 = 4096;
pub const verify_rows: u32 = 8;

pub const Sequence = struct {
    position: u64 = 0,
    prefill_end: u64 = 0,
    slot: u32,

    /// Reuse only the exact matching prefill prefix, then recompute the suffix in fixed chunks.
    pub fn startFrom(s: *Sequence, prefix: ?*const pages.Entry, hash: [32]u8, prompt_tokens: u32) u32 {
        s.position = 0;
        s.prefill_end = 0;
        if (prefix) |entry| {
            const cached = pages.PageBook.reusableRows(entry, hash, prompt_tokens);
            s.position = cached;
            s.prefill_end = cached;
            return cached;
        }
        return 0;
    }

    pub fn prefillChunk(s: *Sequence, start: u64, end: u64) !void {
        if (start != s.position or end < start or end - start > prompt_chunk_rows) return error.BadPrefillChunk;
        s.position = end;
        s.prefill_end = end;
    }

    /// Decode may write cache rows, but the prompt cache's reusable boundary stays at prefill_end.
    pub fn decodeWrite(s: *Sequence, rows: u32) void {
        s.position += rows;
    }

    /// Keep rolls the position back after a verify; rejected rows are overwritten on the next pass.
    pub fn keep(s: *Sequence, rows: u32) !void {
        if (rows > s.position) return error.KeepPastPosition;
        s.position = rows;
    }

    /// Continue a resumed conversation through WP9's continuity boundary. `nextTurn` rewinds
    /// decode-written rows to its prefill frontier, then prefill-writes the reply and new turn.
    pub fn nextTurn(s: *Sequence, engine: anytype, turn: []const u32) !u32 {
        if (s.position < s.prefill_end) return error.BadPrefillFrontier;
        const first = try engine.nextTurn(turn);
        s.position = engine.pos;
        s.prefill_end = engine.frontier;
        return first;
    }

    /// A background warm pass drops decode KV before writing its reply and the next user prefix.
    pub fn warmFromFrontier(engine: anytype, turn: []const u32) !u32 {
        engine.rewindToFrontier();
        return engine.nextTurn(turn);
    }
};

test "resume reuses only prefill rows and recomputes the suffix" {
    var seq: Sequence = .{ .slot = 1 };
    const entry = pages.Entry{
        .fingerprint = .{ .hash = @splat(9), .tokens = 6000, .prefill_rows = 6000 },
        .page_ids = @constCast(&[_]u32{1}),
        .prefill_end = 6000,
        .touched = 0,
        .bytes = 1,
    };
    const cached = seq.startFrom(&entry, @splat(9), 7000);
    try std.testing.expectEqual(@as(u32, 6000), cached);
    try seq.prefillChunk(6000, 7000);
    seq.decodeWrite(4);
    try std.testing.expectEqual(@as(u64, 7000), seq.prefill_end);
    try seq.keep(7000);
    try std.testing.expectEqual(@as(u64, 7000), seq.position);
}

test "prefill chunks are bounded and contiguous" {
    var seq: Sequence = .{ .slot = 0 };
    try seq.prefillChunk(0, prompt_chunk_rows);
    try std.testing.expectError(error.BadPrefillChunk, seq.prefillChunk(0, 10));
    try std.testing.expectError(error.BadPrefillChunk, seq.prefillChunk(prompt_chunk_rows, prompt_chunk_rows + prompt_chunk_rows + 1));
}

test "resume delegates the suffix to Engine.nextTurn at the cached frontier" {
    const FakeEngine = struct {
        frontier: usize = 6000,
        pos: usize = 6004,
        decoded: [4]u32 = .{ 91, 92, 93, 94 },
        rows_seen: usize = 0,
        calls: usize = 0,
        pub fn nextTurn(e: *@This(), turn: []const u32) !u32 {
            e.calls += 1;
            e.rows_seen = e.decoded.len + turn.len;
            e.pos = e.frontier + e.rows_seen;
            e.frontier = e.pos;
            return 17;
        }
        pub fn rewindToFrontier(e: *@This()) void {
            e.pos = e.frontier;
        }
    };
    var engine: FakeEngine = .{};
    var seq: Sequence = .{ .slot = 0, .position = 6004, .prefill_end = 6000 };
    try std.testing.expectEqual(@as(u32, 17), try seq.nextTurn(&engine, &.{ 7, 8, 9 }));
    try std.testing.expectEqual(@as(usize, 1), engine.calls);
    try std.testing.expectEqual(@as(usize, 7), engine.rows_seen);
    try std.testing.expectEqual(@as(u64, 6007), seq.position);
    try std.testing.expectEqual(@as(u32, 17), try Sequence.warmFromFrontier(&engine, &.{10}));
    try std.testing.expectEqual(@as(usize, 2), engine.calls);
    try std.testing.expectEqual(@as(usize, 5), engine.rows_seen);
}

test "lane public declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Sequence);
}
