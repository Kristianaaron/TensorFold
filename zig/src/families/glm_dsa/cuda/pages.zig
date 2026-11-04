// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31: zig/src/sessions/pool.zig and zig/src/families/deepseek_v41/kv/sess.zig (Apache-2.0). Changes: DCP2 RAM-only latent, rope, and index prompt pages.
//! Rank-local RAM prompt pages for GLM-5.3 DSA. Decode-written rows are never reusable.

const std = @import("std");

pub const radix = @import("radix.zig");

pub const page_tokens: u32 = 256;
pub const dcp_world: u32 = 2;
pub const latent_row_bytes: u32 = 1152; // 512 latent + 64 RoPE, bf16
pub const index_row_bytes: u32 = 256; // 32 x 128 bf16
pub const index_layers: u32 = 21;
pub const model_layers: u32 = 78;

/// SHA-256 over little-endian token ids; the hash changes when any token in the prefix changes.
pub fn prefixHash(tokens: []const u32) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var encoded: [4]u8 = undefined;
    for (tokens) |token| {
        std.mem.writeInt(u32, &encoded, token, .little);
        hash.update(&encoded);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

pub const Plane = enum(u8) { latent_rope, index };

/// Position p belongs to exactly one DCP rank. The slot is local to that rank's cache.
pub fn slotOf(position: u64, rank: u32, world: u32) ?u64 {
    if (world == 0 or rank >= world or position % world != rank) return null;
    return position / world;
}

pub fn ownerOf(position: u64, world: u32) u32 {
    return @intCast(position % world);
}

/// Number of rank-local token rows through `end` (exclusive).
pub fn localRows(end: u64, rank: u32, world: u32) u64 {
    if (world == 0 or rank >= world) return 0;
    if (end <= rank) return 0;
    return (end - 1 - rank) / world + 1;
}

pub const Fingerprint = struct {
    hash: [32]u8,
    tokens: u32,
    prefill_rows: u32,
    protocol: u16 = 1,
};

pub const Entry = struct {
    fingerprint: Fingerprint,
    page_ids: []u32,
    /// Resume only through this position. Decode writes are excluded by `keepPrefill`.
    prefill_end: u64,
    touched: u64,
    bytes: u64,
};

/// Find the longest cached token prefix. Hashes are computed only at page boundaries so a resumed
/// transcript can reuse complete pages and prefill its remaining suffix through Engine.nextTurn.
pub fn longestPrefix(book: *PageBook, tokens: []const u32) ?struct { entry: *Entry, rows: u32 } {
    var end = tokens.len - tokens.len % page_tokens;
    while (end != 0) : (end -= page_tokens) {
        const hash = prefixHash(tokens[0..end]);
        if (book.lookup(hash)) |entry| {
            const rows = PageBook.reusableRows(entry, hash, @intCast(end));
            if (rows == end) return .{ .entry = entry, .rows = rows };
        }
    }
    return null;
}

pub const PageBook = struct {
    gpa: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    capacity_bytes: u64,
    used_bytes: u64 = 0,
    next_touch: u64 = 1,

    pub fn init(gpa: std.mem.Allocator, capacity_bytes: u64) PageBook {
        return .{ .gpa = gpa, .capacity_bytes = capacity_bytes };
    }

    pub fn deinit(book: *PageBook) void {
        for (book.entries.items) |entry| book.gpa.free(entry.page_ids);
        book.entries.deinit(book.gpa);
        book.* = undefined;
    }

    pub fn lookup(book: *PageBook, hash: [32]u8) ?*Entry {
        for (book.entries.items) |*entry| if (std.mem.eql(u8, &entry.fingerprint.hash, &hash)) {
            entry.touched = book.tick();
            return entry;
        };
        return null;
    }

    /// Insert page ids, evicting least-recently-used entries until the byte budget fits.
    pub fn insert(book: *PageBook, fp: Fingerprint, page_ids: []const u32, bytes: u64, prefill_end: u64) !void {
        if (bytes > book.capacity_bytes) return error.PageBudgetExceeded;
        if (book.lookup(fp.hash)) |old| {
            const index = (@intFromPtr(old) - @intFromPtr(book.entries.items.ptr)) / @sizeOf(Entry);
            book.removeAt(index);
        }
        const owned = try book.gpa.dupe(u32, page_ids);
        errdefer book.gpa.free(owned);
        while (book.used_bytes + bytes > book.capacity_bytes) book.evictOldest();
        try book.entries.append(book.gpa, .{ .fingerprint = fp, .page_ids = owned, .prefill_end = prefill_end, .touched = book.tick(), .bytes = bytes });
        book.used_bytes += bytes;
    }

    /// A prompt prefix is reusable only if every row was written by prefill and the prefix hash matches.
    pub fn reusableRows(entry: *const Entry, requested_hash: [32]u8, requested_tokens: u32) u32 {
        if (!std.mem.eql(u8, &entry.fingerprint.hash, &requested_hash)) return 0;
        return @intCast(@min(@min(@min(entry.prefill_end, entry.fingerprint.prefill_rows), requested_tokens), entry.fingerprint.tokens));
    }

    fn tick(book: *PageBook) u64 {
        const t = book.next_touch;
        book.next_touch +%= 1;
        return t;
    }

    fn removeAt(book: *PageBook, index: usize) void {
        const old = book.entries.orderedRemove(index);
        book.used_bytes -= old.bytes;
        book.gpa.free(old.page_ids);
    }

    fn evictOldest(book: *PageBook) void {
        if (book.entries.items.len == 0) return;
        var oldest: usize = 0;
        for (book.entries.items[1..], 1..) |entry, i| if (entry.touched < book.entries.items[oldest].touched) {
            oldest = i;
        };
        book.removeAt(oldest);
    }
};

/// Interface to WP9's eventual capacity/budget calculation. `availableBytes` must already
/// include model weights, workspace, drafter, and the safety floor on the smallest rank.
pub const Budget = struct {
    ctx: *anyopaque,
    availableBytes: *const fn (*anyopaque) u64,
};

pub const Admission = struct {
    context_tokens: u64,
    request_tokens: u64,
    page_bytes: u64,
    cache_bytes: u64,
};

/// Derive the RAM cache ceiling from WP9's exact rank admission. Only bytes left after its
/// weights, state, graph runtime, and reserve are available to prompt pages.
pub fn cacheBudgetFromRank(admission: anytype) u64 {
    const admitted = admission.breakdown.total();
    return admission.mem_available -| admission.reserve.bytes -| admitted;
}

pub fn admit(budget: Budget, context_tokens: u64, max_new: u64, page_cache_bytes: u64) !Admission {
    const request_tokens = std.math.add(u64, context_tokens, max_new) catch return error.RequestTooLarge;
    const rounded = std.math.add(u64, request_tokens, page_tokens - 1) catch return error.RequestTooLarge;
    const pages = rounded / page_tokens;
    // Each DCP rank owns half the positions; each rank stores 78 latent/rope planes and 21 index planes.
    const page_bytes = @as(u64, page_tokens / dcp_world) * (model_layers * latent_row_bytes + index_layers * index_row_bytes);
    const cache_bytes = std.math.mul(u64, pages, page_bytes) catch return error.RequestTooLarge;
    const total_bytes = std.math.add(u64, cache_bytes, page_cache_bytes) catch return error.RequestTooLarge;
    if (total_bytes > budget.availableBytes(budget.ctx)) return error.RequestTooLarge;
    return .{ .context_tokens = context_tokens, .request_tokens = request_tokens, .page_bytes = page_bytes, .cache_bytes = total_bytes };
}

test "page book helpers and public nested declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(radix);
    std.testing.refAllDecls(PageBook);
    std.testing.refAllDecls(Fingerprint);
    std.testing.refAllDecls(Entry);
    std.testing.refAllDecls(Budget);
    std.testing.refAllDecls(Admission);
}

test "DCP slots map logical positions to rank-local rows" {
    try std.testing.expectEqual(@as(u32, 0), ownerOf(0, 2));
    try std.testing.expectEqual(@as(u32, 1), ownerOf(5, 2));
    try std.testing.expectEqual(@as(?u64, 3), slotOf(6, 0, 2));
    try std.testing.expectEqual(@as(?u64, null), slotOf(6, 1, 2));
    try std.testing.expectEqual(@as(u64, 4), localRows(8, 0, 2));
    try std.testing.expectEqual(@as(u64, 4), localRows(8, 1, 2));
}

test "prefix hash uses a stable token-id encoding" {
    try std.testing.expectEqualSlices(u8, &prefixHash(&.{ 1, 2, 3 }), &prefixHash(&.{ 1, 2, 3 }));
    try std.testing.expect(!std.mem.eql(u8, &prefixHash(&.{ 1, 2, 3 }), &prefixHash(&.{ 1, 2, 4 })));
}

test "page book insert lookup and LRU eviction" {
    var book = PageBook.init(std.testing.allocator, 2 * 256 * (latent_row_bytes + index_layers * index_row_bytes));
    defer book.deinit();
    const p1: Fingerprint = .{ .hash = @splat(1), .tokens = 256, .prefill_rows = 256 };
    const p2: Fingerprint = .{ .hash = @splat(2), .tokens = 256, .prefill_rows = 256 };
    const p3: Fingerprint = .{ .hash = @splat(3), .tokens = 256, .prefill_rows = 256 };
    const bytes: u64 = 256 * (latent_row_bytes + index_layers * index_row_bytes);
    try book.insert(p1, &.{4}, bytes, 256);
    try book.insert(p2, &.{6}, bytes, 256);
    try std.testing.expect(book.lookup(p1.hash) != null);
    try book.insert(p3, &.{8}, bytes, 256);
    try std.testing.expect(book.lookup(p1.hash) != null);
    try std.testing.expect(book.lookup(p2.hash) == null);
    try std.testing.expectEqual(@as(u32, 128), PageBook.reusableRows(book.lookup(p3.hash).?, p3.hash, 128));
}

test "longest page prefix lookup chooses the furthest complete cached prefix" {
    var book = PageBook.init(std.testing.allocator, 1 << 20);
    defer book.deinit();
    const short_tokens = std.mem.zeroes([page_tokens]u32);
    const long_tokens = std.mem.zeroes([2 * page_tokens]u32);
    try book.insert(.{ .hash = prefixHash(&short_tokens), .tokens = page_tokens, .prefill_rows = page_tokens }, &.{1}, 1024, page_tokens);
    try book.insert(.{ .hash = prefixHash(&long_tokens), .tokens = 2 * page_tokens, .prefill_rows = 2 * page_tokens }, &.{ 1, 2 }, 2048, 2 * page_tokens);
    const full = std.mem.zeroes([2 * page_tokens + 1]u32);
    const found = longestPrefix(&book, &full).?;
    try std.testing.expectEqual(@as(u32, 2 * page_tokens), found.rows);
    try std.testing.expectEqual(@as(u32, 2 * page_tokens), found.entry.prefill_end);
}

test "admission rejects context plus generation beyond budget" {
    const Ctx = struct { bytes: u64 };
    const read = struct {
        fn f(p: *anyopaque) u64 {
            return @as(*Ctx, @ptrCast(@alignCast(p))).bytes;
        }
    }.f;
    var ctx = Ctx{ .bytes = 1 };
    try std.testing.expectError(error.RequestTooLarge, admit(.{ .ctx = &ctx, .availableBytes = read }, 1, 1, 0));
    ctx.bytes = 1 << 40;
    try std.testing.expect((try admit(.{ .ctx = &ctx, .availableBytes = read }, 1, 1, 0)).cache_bytes > 0);
}

test "page-cache budget uses WP9 rank admission remainder" {
    const FakeAdmission = struct {
        mem_available: u64,
        reserve: struct { bytes: u64 },
        breakdown: struct {
            bytes: u64,
            fn total(b: @This()) u64 {
                return b.bytes;
            }
        },
    };
    const a: FakeAdmission = .{ .mem_available = 1000, .reserve = .{ .bytes = 100 }, .breakdown = .{ .bytes = 700 } };
    try std.testing.expectEqual(@as(u64, 200), cacheBudgetFromRank(a));
}
