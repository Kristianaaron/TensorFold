// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// GLM-5.3 prompt cache radix index (TensorFold fork); replaces pages.zig's whole-prefix rehashing with chained page keys.
//! Host-side chained page keys and tiered cache index for GLM-5.3 prompt pages.

const std = @import("std");

pub const page_tokens: u32 = 256;
pub const Key = [32]u8;
pub const Tier = enum(u8) { ram, nvme };
pub const Loc = struct { tier: Tier, id: u64 };
pub const Page = struct {
    key: Key,
    parent: ?Key,
    index: u32,
    loc: Loc,
    tokens: [page_tokens]u32,
    bytes: u64,
    refs: u32 = 0,
    touched: u64 = 0,
};

/// Hash the model and rank agreement fingerprint into the radix root.
pub fn rootKey(fingerprint: []const u8) Key {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(fingerprint);
    var digest: Key = undefined;
    hash.final(&digest);
    return digest;
}

/// Hash a page after its parent key, using stable little-endian token ids.
pub fn nextKey(prev: Key, page: []const u32) Key {
    std.debug.assert(page.len == page_tokens);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&prev);
    var encoded: [4]u8 = undefined;
    for (page) |token| {
        std.mem.writeInt(u32, &encoded, token, .little);
        hash.update(&encoded);
    }
    var digest: Key = undefined;
    hash.final(&digest);
    return digest;
}

/// Return chained keys for complete pages; an incomplete tail has no key.
pub fn keys(gpa: std.mem.Allocator, root: Key, tokens: []const u32) ![]Key {
    const count = tokens.len / page_tokens;
    const result = try gpa.alloc(Key, count);
    errdefer gpa.free(result);
    var parent = root;
    for (0..count) |i| {
        const start = i * page_tokens;
        parent = nextKey(parent, tokens[start .. start + page_tokens]);
        result[i] = parent;
    }
    return result;
}

/// In-memory index of resident RAM and NVMe pages, with leaf-only LRU eviction.
pub const Index = struct {
    pub const Hit = struct { pages: u32, tokens: u32, last: ?Key };
    pub const Stats = struct { pages: [2]u32, bytes: [2]u64 };
    pub const EvictFn = *const fn (ctx: ?*anyopaque, loc: Loc) void;

    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(Key, Page) = .empty,
    budget: [2]u64,
    used: [2]u64 = .{ 0, 0 },
    clock: u64 = 0,
    on_evict: ?EvictFn = null,
    ctx: ?*anyopaque = null,

    pub fn init(gpa: std.mem.Allocator, ram_budget: u64, nvme_budget: u64) Index {
        return .{ .gpa = gpa, .budget = .{ ram_budget, nvme_budget } };
    }

    pub fn deinit(ix: *Index) void {
        ix.map.deinit(ix.gpa);
        ix.* = undefined;
    }

    /// Find the longest chain whose resident token payloads match the prompt.
    pub fn longest(ix: *Index, root: Key, tokens: []const u32) !Hit {
        var parent: ?Key = null;
        var current = root;
        var pages: u32 = 0;
        const count = tokens.len / page_tokens;
        for (0..count) |i| {
            const start = i * page_tokens;
            const slice = tokens[start .. start + page_tokens];
            current = nextKey(current, slice);
            const page = ix.map.getPtr(current) orelse break;
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(&page.tokens), std.mem.sliceAsBytes(slice))) break;
            ix.clock +%= 1;
            page.touched = ix.clock;
            pages += 1;
            parent = current;
        }
        return .{ .pages = pages, .tokens = pages * page_tokens, .last = parent };
    }

    /// Insert a page, evicting eligible leaves until its tier has room.
    pub fn insert(ix: *Index, key: Key, parent: ?Key, index: u32, loc: Loc, page_tokens_slice: []const u32, bytes: u64) !void {
        std.debug.assert(page_tokens_slice.len == page_tokens);
        const tier_index = tierIndex(loc.tier);
        if (bytes > ix.budget[tier_index]) return error.PageBudgetExceeded;
        if (ix.map.fetchRemove(key)) |old| ix.removePage(old.value);
        while (ix.used[tier_index] > ix.budget[tier_index] - bytes) {
            if (!ix.evictOne(loc.tier)) return error.PageBudgetExceeded;
        }
        var token_array: [page_tokens]u32 = undefined;
        @memcpy(&token_array, page_tokens_slice);
        ix.clock +%= 1;
        try ix.map.put(ix.gpa, key, .{ .key = key, .parent = parent, .index = index, .loc = loc, .tokens = token_array, .bytes = bytes, .touched = ix.clock });
        ix.used[tier_index] += bytes;
    }

    /// Evict the oldest unpinned leaf in one tier and notify its storage owner.
    pub fn evictOne(ix: *Index, tier: Tier) bool {
        var candidate: ?Key = null;
        var oldest: u64 = std.math.maxInt(u64);
        var it = ix.map.iterator();
        while (it.next()) |entry| {
            const page = entry.value_ptr;
            if (page.loc.tier != tier or page.refs != 0 or page.touched >= oldest) continue;
            var has_child = false;
            var children = ix.map.valueIterator();
            while (children.next()) |other| if (other.parent) |p| {
                if (std.mem.eql(u8, &p, entry.key_ptr)) { has_child = true; break; }
            };
            if (!has_child) { oldest = page.touched; candidate = entry.key_ptr.*; }
        }
        const key = candidate orelse return false;
        const removed = ix.map.fetchRemove(key).?;
        const loc = removed.value.loc;
        ix.removePage(removed.value);
        if (ix.on_evict) |callback| callback(ix.ctx, loc);
        return true;
    }

    /// Pin all present keys while a session uses or restores their pages.
    pub fn pin(ix: *Index, page_keys: []const Key) void {
        for (page_keys) |key| {
            if (ix.map.getPtr(key)) |page| page.refs +%= 1;
        }
    }

    /// Release pins acquired for a resident session or restore.
    pub fn unpin(ix: *Index, page_keys: []const Key) void {
        for (page_keys) |key| if (ix.map.getPtr(key)) |page| {
            std.debug.assert(page.refs != 0);
            page.refs -= 1;
        };
    }

    /// Count cached pages and bytes by tier.
    pub fn stats(ix: *const Index) Stats {
        var result = Stats{ .pages = .{ 0, 0 }, .bytes = .{ 0, 0 } };
        var it = ix.map.valueIterator();
        while (it.next()) |page| {
            const t = tierIndex(page.loc.tier);
            result.pages[t] += 1;
            result.bytes[t] += page.bytes;
        }
        return result;
    }

    fn removePage(ix: *Index, page: Page) void {
        ix.used[tierIndex(page.loc.tier)] -= page.bytes;
    }
};

fn tierIndex(tier: Tier) usize { return @intFromEnum(tier); }

test "chained keys change from the first differing page onward" {
    const allocator = std.testing.allocator;
    var tokens = @as([600]u32, @splat(0));
    const root = rootKey("fingerprint");
    const original = try keys(allocator, root, &tokens);
    defer allocator.free(original);
    try std.testing.expectEqual(@as(usize, 2), original.len);
    try std.testing.expectEqualSlices(u8, &original[1], &nextKey(original[0], tokens[256..512]));
    tokens[0] = 1;
    const first_changed = try keys(allocator, root, &tokens);
    defer allocator.free(first_changed);
    try std.testing.expect(!std.mem.eql(u8, &original[0], &first_changed[0]));
    try std.testing.expect(!std.mem.eql(u8, &original[1], &first_changed[1]));
    tokens[0] = 0;
    tokens[256] = 1;
    const second_changed = try keys(allocator, root, &tokens);
    defer allocator.free(second_changed);
    try std.testing.expectEqualSlices(u8, &original[0], &second_changed[0]);
    try std.testing.expect(!std.mem.eql(u8, &original[1], &second_changed[1]));
}

fn testPage(ix: *Index, key: Key, parent: ?Key, index: u32, tokens: []const u32, bytes: u64, tier: Tier) !void {
    try ix.insert(key, parent, index, .{ .tier = tier, .id = index }, tokens, bytes);
}

test "longest returns shared complete prompt pages" {
    const allocator = std.testing.allocator;
    const root = rootKey("test");
    var a = @as([768]u32, @splat(1));
    var b = a;
    b[300] = 8;
    var c = a;
    c[0] = 9;
    const ak = try keys(allocator, root, &a); defer allocator.free(ak);
    var ix = Index.init(allocator, 3 * 1024, 0); defer ix.deinit();
    for (0..3) |i| try testPage(&ix, ak[i], if (i == 0) root else ak[i - 1], @intCast(i), a[i * 256 ..][0..256], 1024, .ram);
    const full = try ix.longest(root, &a); try std.testing.expectEqual(@as(u32, 3), full.pages);
    const short = try ix.longest(root, a[0..700]); try std.testing.expectEqual(@as(u32, 2), short.pages);
    const shared = try ix.longest(root, &b); try std.testing.expectEqual(@as(u32, 1), shared.pages);
    const miss = try ix.longest(root, &c); try std.testing.expectEqual(@as(u32, 0), miss.pages);
}

test "token corruption stops longest prefix" {
    const allocator = std.testing.allocator;
    const root = rootKey("corrupt");
    const a = @as([512]u32, @splat(4));
    const ak = try keys(allocator, root, &a); defer allocator.free(ak);
    var ix = Index.init(allocator, 4096, 0); defer ix.deinit();
    try testPage(&ix, ak[0], root, 0, a[0..256], 1024, .ram);
    var altered = a[256..512].*;
    altered[0] = 5;
    try testPage(&ix, ak[1], ak[0], 1, &altered, 1024, .ram);
    const hit = try ix.longest(root, &a);
    try std.testing.expectEqual(@as(u32, 1), hit.pages);
}

test "leaf LRU eviction respects pins and tiers" {
    const allocator = std.testing.allocator;
    const root = rootKey("evict");
    const a = @as([768]u32, @splat(1));
    const ak = try keys(allocator, root, &a); defer allocator.free(ak);
    const b = @as([256]u32, @splat(2));
    const bk = try keys(allocator, root, &b); defer allocator.free(bk);
    const Ctx = struct { evicted: ?Loc = null };
    var ctx = Ctx{};
    const callback = struct { fn call(raw: ?*anyopaque, loc: Loc) void { @as(*Ctx, @ptrCast(@alignCast(raw.?))).evicted = loc; } }.call;
    var ix = Index.init(allocator, 3 * 1024, 1024); defer ix.deinit();
    ix.on_evict = callback; ix.ctx = &ctx;
    for (0..3) |i| try testPage(&ix, ak[i], if (i == 0) root else ak[i - 1], @intCast(i), a[i * 256 ..][0..256], 1024, .ram);
    _ = try ix.longest(root, &a);
    try testPage(&ix, bk[0], root, 0, &b, 1024, .ram);
    try std.testing.expectEqual(@as(u64, 2), ctx.evicted.?.id);
    ix.pin(&.{ak[0]});
    // A0 is pinned; A1 (a leaf once A2 went, touched before B was inserted) is the oldest unpinned leaf.
    ctx.evicted = null;
    try testPage(&ix, rootKey("ram-page"), null, 9, &b, 1024, .ram);
    try std.testing.expectEqual(@as(u64, 1), ctx.evicted.?.id);
    try testPage(&ix, rootKey("nvme-page"), null, 10, &b, 1024, .nvme);
    try std.testing.expectEqual(@as(u32, 1), ix.stats().pages[1]);
    ix.unpin(&.{ak[0]});
}

test "replacement accounting and per-tier budgets" {
    const allocator = std.testing.allocator;
    const root = rootKey("replace");
    const tokens = @as([256]u32, @splat(3));
    const key = nextKey(root, &tokens);
    var ix = Index.init(allocator, 4096, 2048); defer ix.deinit();
    try testPage(&ix, key, root, 0, &tokens, 512, .ram);
    try testPage(&ix, key, root, 0, &tokens, 700, .ram);
    try std.testing.expectEqual(@as(u64, 700), ix.used[0]);
    const other = nextKey(rootKey("other"), &tokens);
    try testPage(&ix, other, null, 0, &tokens, 1500, .nvme);
    const stats = ix.stats();
    try std.testing.expectEqual(@as(u64, 700), stats.bytes[0]);
    try std.testing.expectEqual(@as(u64, 1500), stats.bytes[1]);
    try std.testing.expectError(error.PageBudgetExceeded, testPage(&ix, rootKey("too-big"), null, 0, &tokens, 2049, .nvme));
}
