// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GB10 unified-memory hygiene for the GLM-5.3 loader and state (WP9/WP13): the page cache shares the DRAM that
//! CUDA allocates from, so budgets come from /proc/meminfo (not cuMemGetInfo), weight reads can bypass the page
//! cache (O_DIRECT through four 4 MiB aligned staging slots), shard pages are dropped after unmapping, and pinned
//! host buffers are mlocked (registration does not pin on an integrated GPU). Stock system only: no sudo, sysctl,
//! drop_caches or limit changes; mlock fails soft inside the existing RLIMIT_MEMLOCK.
//! Design after NVIDIA flare-local-inference (Apache-2.0): helios runtime_package payload.rs (O_DIRECT staging),
//! archive.rs / host_memory.py (posix_fadvise DONTNEED after unmap), runtime_package mod.rs (mlock before
//! registration), memory.rs (MemAvailable budgets). No code copied; Zig written for this engine.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

pub const gib: u64 = 1 << 30;
pub const stage_slots = 4;
pub const stage_bytes: usize = 4 << 20;
/// O_DIRECT offset, length and buffer alignment.
pub const direct_align: usize = 4096;

const libc = struct {
    extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;
    extern "c" fn munlock(addr: *const anyopaque, len: usize) c_int;
};

pub const MemInfo = struct { total: u64, available: u64, free: u64 };

pub const mem_detail_keys = [_][]const u8{
    "MemFree", "MemAvailable", "Buffers", "Cached", "AnonPages", "Shmem", "Mlocked", "Unevictable",
    "Slab", "SReclaimable", "SUnreclaim", "KernelStack", "PageTables", "SecPageTables", "VmallocUsed",
    "Percpu", "AnonHugePages", "HugePages_Total", "CmaFree",
};

pub const MemDetail = struct { fields: [mem_detail_keys.len]u64 };

var graph_room_threshold: std.atomic.Value(u64) = .init(0);

/// Configure the no-argument graph-cache room callback for the process's single GLM engine.
pub fn configureGraphRoom(guard_floor: u64, half_margin: u64, per_graph: u64) !void {
    const threshold = try std.math.add(u64, guard_floor, half_margin);
    graph_room_threshold.store(try std.math.add(u64, threshold, per_graph), .release);
}

pub fn graphRoom() bool {
    const info = readMeminfoFast() orelse return false;
    return info.available >= graph_room_threshold.load(.acquire);
}

pub const GraphStats = struct {
    held: usize = 0,
    captured: u64 = 0,
    capture_mib: f64 = 0,
};

pub const MemTrace = struct {
    /// The admission floor (state.floorFromEnv), set by the caller: uma does not import state.zig.
    floor: u64 = 0,
    on: bool = false,
    rank: usize = 0,
    min_avail: u64 = std.math.maxInt(u64),
    min_free: u64 = std.math.maxInt(u64),

    pub fn fromEnv(rank: usize) !MemTrace {
        return fromEnvWith(rank, envValue);
    }

    /// Injectable environment lookup keeps host tests independent of process environment state.
    pub fn fromEnvWith(rank: usize, get: *const fn ([]const u8) ?[]const u8) !MemTrace {
        const value = get("TF_GLM53_MEM_TRACE") orelse return .{ .rank = rank };
        if (std.mem.eql(u8, value, "1")) return .{ .on = true, .rank = rank };
        if (std.mem.eql(u8, value, "0")) return .{ .rank = rank };
        return error.BadMemTrace;
    }

    fn envValue(name: []const u8) ?[]const u8 {
        _ = name;
        const value = std.c.getenv("TF_GLM53_MEM_TRACE") orelse return null;
        return std.mem.span(value);
    }

    /// Per-chunk memory trace after MiaAI-Lab GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patch 0124 (Apache-2.0); independent Zig.
    pub fn point(t: *MemTrace, gpa: std.mem.Allocator, io: std.Io, stream: cuda.Stream, what: []const u8, graphs: GraphStats) !void {
        if (!t.on) return;
        try stream.synchronize();
        const info = try readMeminfo(gpa, io);
        t.min_avail = @min(t.min_avail, info.available);
        t.min_free = @min(t.min_free, info.free);
        const floor = t.floor;
        var buf: [512]u8 = undefined;
        const line = try formatLine(&buf, t.rank, what, info.available, t.min_avail, info.free, graphs.held, graphs.captured, graphs.capture_mib, floor);
        std.log.info("{s}", .{line});
    }
};

fn gibFloat(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gib));
}

pub fn formatLine(buf: []u8, rank: usize, what: []const u8, avail: u64, min_avail: u64, free: u64, held: usize, captured: u64, capture_mib: f64, floor: u64) ![]const u8 {
    return std.fmt.bufPrint(buf, "glm53 mem rank {d} {s}: MemAvailable {d:.2} GiB (min {d:.2}), MemFree {d:.2} GiB, graphs held {d} captured {d} capture_mib {d:.2}, floor {d:.2} GiB", .{ rank, what, gibFloat(avail), gibFloat(min_avail), gibFloat(free), held, captured, capture_mib, gibFloat(floor) });
}

test "memory trace formatLine golden startup sample" {
    var buf: [256]u8 = undefined;
    const line = try formatLine(&buf, 0, "state", 8 * gib, 7 * gib, 2 * gib, 0, 0, 0, 4 * gib);
    try std.testing.expectEqualStrings("glm53 mem rank 0 state: MemAvailable 8.00 GiB (min 7.00), MemFree 2.00 GiB, graphs held 0 captured 0 capture_mib 0.00, floor 4.00 GiB", line);
}

test "memory trace formatLine golden graph capture sample" {
    var buf: [256]u8 = undefined;
    const line = try formatLine(&buf, 1, "capture rows 8 bucket 64", 3 * gib + gib / 4, 3 * gib, gib / 2, 12, 15, 256.5, 4 * gib);
    try std.testing.expectEqualStrings("glm53 mem rank 1 capture rows 8 bucket 64: MemAvailable 3.25 GiB (min 3.00), MemFree 0.50 GiB, graphs held 12 captured 15 capture_mib 256.50, floor 4.00 GiB", line);
}

test "memory trace environment accepts one" {
    const One = struct { pub fn get(_: []const u8) ?[]const u8 { return "1"; } };
    const trace = try MemTrace.fromEnvWith(1, One.get);
    try std.testing.expect(trace.on);
    try std.testing.expectEqual(@as(usize, 1), trace.rank);
}

test "memory trace environment accepts zero" {
    const Zero = struct { pub fn get(_: []const u8) ?[]const u8 { return "0"; } };
    const trace = try MemTrace.fromEnvWith(1, Zero.get);
    try std.testing.expect(!trace.on);
    try std.testing.expectEqual(@as(usize, 1), trace.rank);
}

test "memory trace environment defaults off when unset" {
    const Empty = struct { pub fn get(_: []const u8) ?[]const u8 { return null; } };
    const trace = try MemTrace.fromEnvWith(1, Empty.get);
    try std.testing.expect(!trace.on);
}

test "memory trace environment rejects other values" {
    const Other = struct { pub fn get(_: []const u8) ?[]const u8 { return "yes"; } };
    try std.testing.expectError(error.BadMemTrace, MemTrace.fromEnvWith(1, Other.get));
}

/// MemTotal, MemAvailable and MemFree in bytes, or null when any is missing, repeated or malformed.
pub fn parseMeminfo(text: []const u8) ?MemInfo {
    var out: [3]?u64 = .{ null, null, null };
    const keys = [_][]const u8{ "MemTotal", "MemAvailable", "MemFree" };
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        const key = words.next() orelse continue;
        for (keys, &out) |k, *slot| if (std.mem.eql(u8, key, k)) {
            const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
            if (!std.mem.eql(u8, words.next() orelse return null, "kB") or slot.* != null) return null;
            slot.* = std.math.mul(u64, kib, 1024) catch return null;
        };
    }
    const info: MemInfo = .{ .total = out[0] orelse return null, .available = out[1] orelse return null, .free = out[2] orelse return null };
    return if (info.total > 0 and info.available <= info.total and info.free <= info.total) info else null;
}

/// Parse the selected /proc/meminfo counters in bytes. Missing counters are zero; malformed or duplicate
/// counters are rejected so a partial read cannot masquerade as a valid delta.
pub fn parseMeminfoDetail(text: []const u8) ?MemDetail {
    var out: MemDetail = .{ .fields = @splat(0) };
    var seen: [mem_detail_keys.len]bool = @splat(false);
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        const key = words.next() orelse continue;
        for (mem_detail_keys, 0..) |name, i| {
            if (!std.mem.eql(u8, key, name)) continue;
            const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
            if (seen[i]) return null;
            if (i == 17) {
                if (words.next() != null) return null; // HugePages_Total is a count, not a kB value.
                out.fields[i] = kib;
            } else {
                if (!std.mem.eql(u8, words.next() orelse return null, "kB")) return null;
                out.fields[i] = std.math.mul(u64, kib, 1024) catch return null;
            }
            seen[i] = true;
        }
    }
    return out;
}

pub fn readMeminfoDetail(gpa: std.mem.Allocator, io: std.Io) !MemDetail {
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return error.NoMemInfo;
    defer file.close(io);
    const text = readStreamingAlloc(&file, gpa, io, 1 << 20) catch return error.NoMemInfo;
    defer gpa.free(text);
    return parseMeminfoDetail(text) orelse error.NoMemInfo;
}

test "detailed meminfo parser reads SecPageTables and converts kB to bytes" {
    const parsed = parseMeminfoDetail("MemFree: 10 kB\nSecPageTables: 23 kB\n") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 10 * 1024), parsed.fields[0]);
    try std.testing.expectEqual(@as(u64, 23 * 1024), parsed.fields[13]);
    try std.testing.expectEqual(@as(u64, 0), parsed.fields[14]);
}

test "detailed meminfo parser defaults a missing SecPageTables counter to zero" {
    const parsed = parseMeminfoDetail("PageTables: 7 kB\n") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 7 * 1024), parsed.fields[12]);
    try std.testing.expectEqual(@as(u64, 0), parsed.fields[13]);
}

pub fn readMeminfo(gpa: std.mem.Allocator, io: std.Io) !MemInfo {
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return error.NoMemInfo;
    defer file.close(io);
    const text = readStreamingAlloc(&file, gpa, io, 1 << 20) catch return error.NoMemInfo;
    defer gpa.free(text);
    return parseMeminfo(text) orelse error.NoMemInfo;
}

/// Current MemAvailable for cuda.graph_cache's no-argument capture meter. The callback runs around graph
/// capture, so use a bounded stack buffer and direct procfs reads instead of allocating through std.Io.
pub fn memAvailableGib() ?f64 {
    const info = readMeminfoFast() orelse return null;
    return @as(f64, @floatFromInt(info.available)) / @as(f64, @floatFromInt(gib));
}

pub fn memAvailableBytes() ?u64 {
    return (readMeminfoFast() orelse return null).available;
}

fn readMeminfoFast() ?MemInfo {
    if (comptime builtin.os.tag != .linux) return null;
    const fd = open("/proc/meminfo", 0);
    if (fd < 0) return null;
    defer _ = close(fd);
    var buf: [4096]u8 = undefined;
    const n = read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return parseMeminfo(buf[0..@intCast(n)]);
}

/// Read to EOF without relying on stat size; procfs and sysfs files commonly report zero bytes of size.
pub fn readStreamingAlloc(file: *std.Io.File, gpa: std.mem.Allocator, io: std.Io, limit: usize) ![]u8 {
    var scratch: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &scratch);
    return reader.interface.allocRemaining(gpa, .limited(limit));
}

// The 2 GiB + 5% headroom is state.umaHeadroomBytes, applied only with TF_GLM53_UMA_HEADROOM=1 (it replaces the
// TF_GLM53_MEM_FLOOR_GIB floor, never adds to it); admission reads MemAvailable from readMeminfo.

pub fn odirectFromEnv() bool {
    const v = std.c.getenv("TF_GLM53_ODIRECT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// mlock `bytes` (page aligned); false when the limit or the OS refuses (fail soft: the caller keeps going).
pub fn lockSoft(bytes: []align(direct_align) u8) bool {
    if (bytes.len == 0) return false;
    return libc.mlock(bytes.ptr, bytes.len) == 0;
}

pub fn unlockSoft(bytes: []align(direct_align) u8) void {
    if (bytes.len != 0) _ = libc.munlock(bytes.ptr, bytes.len);
}

/// A cuMemHostAlloc buffer that is also mlocked: registration alone does not pin on GB10, and a swapped page would
/// fault every GPU access. munlock before the free.
pub const PinnedHost = struct {
    host: cuda.HostBuffer,
    locked: bool,

    pub fn alloc(d: *const cuda.Driver, len: usize) !PinnedHost {
        var h = try cuda.HostBuffer.alloc(d, len);
        errdefer h.free();
        const span: []align(direct_align) u8 = @alignCast(h.slice(u8));
        return .{ .host = h, .locked = lockSoft(span) };
    }

    pub fn bytes(p: *const PinnedHost) []align(direct_align) u8 {
        return @alignCast(p.host.slice(u8));
    }

    pub fn free(p: *PinnedHost) void {
        if (p.locked) unlockSoft(p.bytes());
        p.host.free();
        p.* = undefined;
    }
};

/// Four aligned 4 MiB slots for O_DIRECT reads: pinned + mlocked on a GPU host, plain aligned memory in host tests.
pub const Staging = struct {
    slots: [stage_slots][]align(direct_align) u8,
    pinned: ?[stage_slots]PinnedHost = null,
    gpa: ?std.mem.Allocator = null,

    pub fn initPinned(d: *const cuda.Driver) !Staging {
        var p: [stage_slots]PinnedHost = undefined;
        var made: usize = 0;
        errdefer for (p[0..made]) |*h| h.free();
        for (&p) |*h| {
            h.* = try PinnedHost.alloc(d, stage_bytes);
            made += 1;
        }
        var s: Staging = .{ .slots = undefined, .pinned = p };
        for (&s.slots, &s.pinned.?) |*slot, *h| slot.* = h.bytes();
        return s;
    }

    pub fn initHost(gpa: std.mem.Allocator, slot_bytes: usize) !Staging {
        var s: Staging = .{ .slots = undefined, .gpa = gpa };
        var made: usize = 0;
        errdefer for (s.slots[0..made]) |b| gpa.free(b);
        for (&s.slots) |*slot| {
            slot.* = try gpa.alignedAlloc(u8, .fromByteUnits(direct_align), slot_bytes);
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Staging) void {
        if (s.pinned) |*p| for (p) |*h| h.free();
        if (s.gpa) |g| for (s.slots) |b| g.free(b);
        s.* = undefined;
    }
};

/// A shard opened for page-cache-free reads: O_DIRECT on Linux, F_NOCACHE on macOS (host tests).
pub const DirectFile = struct {
    fd: c_int,

    pub fn open(path: [:0]const u8) !DirectFile {
        var flags: std.c.O = .{ .ACCMODE = .RDONLY };
        if (comptime @hasField(std.c.O, "DIRECT")) flags.DIRECT = true;
        const fd = std.c.open(path.ptr, flags);
        if (fd < 0) return error.DirectOpen;
        if (comptime builtin.os.tag.isDarwin()) _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
        return .{ .fd = fd };
    }

    pub fn close(f: *DirectFile) void {
        _ = std.c.close(f.fd);
        f.* = undefined;
    }

    /// File bytes [offset, offset + dst.len) into `dst`: aligned reads of whole 4 KiB blocks through the staging
    /// slots in turn, a short final block at end of file accepted, the requested span copied out.
    pub fn read(f: DirectFile, staging: *const Staging, offset: u64, dst: []u8) !void {
        if (dst.len == 0) return;
        const end = offset + dst.len;
        var pos = offset / direct_align * direct_align;
        var slot: usize = 0;
        while (pos < end) : (slot = (slot + 1) % stage_slots) {
            const buf = staging.slots[slot];
            const want = @min(buf.len, std.mem.alignForward(u64, end, direct_align) - pos);
            const got = std.c.pread(f.fd, buf.ptr, @intCast(want), @intCast(pos));
            if (got <= 0) return error.DirectRead;
            const n: u64 = @intCast(got);
            const lo = @max(pos, offset);
            const hi = @min(pos + n, end);
            if (hi <= lo) return error.DirectRead;
            @memcpy(dst[lo - offset .. hi - offset], buf[lo - pos .. hi - pos]);
            if (n < want and hi < end) return error.DirectRead; // short read before the span ended
            pos += n;
        }
    }
};

fn regularFile(fd: c_int) !bool {
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var st: linux.Statx = undefined;
        const at_empty_path: u32 = 0x1000;
        if (linux.errno(linux.statx(fd, "", at_empty_path, .{ .TYPE = true }, &st)) != .SUCCESS) return error.DropStat;
        return (st.mode & linux.S.IFMT) == linux.S.IFREG;
    }
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.DropStat;
    return (st.mode & std.c.S.IFMT) == std.c.S.IFREG;
}

/// Drop a weight file's clean pages after it is unmapped (posix_fadvise DONTNEED; a live mapping keeps them).
/// Regular files only. Returns whether the advice was given (false off Linux).
pub fn dropPageCache(path: [:0]const u8) !bool {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.DropOpen;
    defer _ = std.c.close(fd);
    if (!try regularFile(fd)) return error.NotRegularFile;
    if (comptime builtin.os.tag != .linux) return false;
    return std.os.linux.fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.DONTNEED) == 0;
}

test "meminfo parse: MemTotal, MemAvailable and MemFree in bytes, malformed input refused" {
    const info = parseMeminfo("MemTotal: 125829120 kB\nMemFree: 8388608 kB\nMemAvailable: 121110528 kB\n").?;
    try std.testing.expectEqual(@as(u64, 120 * gib), info.total);
    try std.testing.expectEqual(@as(u64, 8 * gib), info.free);
    try std.testing.expectEqual(@as(?MemInfo, null), parseMeminfo("MemTotal: 1 kB\nMemAvailable: 1 kB\n"));
    try std.testing.expectEqual(@as(?MemInfo, null), parseMeminfo("MemTotal: 4 kB\nMemFree: 1 kB\nMemFree: 1 kB\nMemAvailable: 1 kB\n"));
}

test "streaming file read consumes all bytes without using the file's stat size" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const expected = "first line\nsecond line\nthird line\n";
    try tmp.dir.writeFile(io, .{ .sub_path = "meminfo", .data = expected });
    var file = try tmp.dir.openFile(io, "meminfo", .{});
    defer file.close(io);
    const got = try readStreamingAlloc(&file, gpa, io, 1024);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, expected, got);
}

test "direct reads return the exact bytes of unaligned spans and a short final block" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const len = 3 * 8192 + 777; // the last block is short
    const data = try gpa.alloc(u8, len);
    defer gpa.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
    try tmp.dir.writeFile(io, .{ .sub_path = "shard.bin", .data = data });
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, ".zig-cache/tmp/{s}/shard.bin", .{tmp.sub_path}, 0);
    var staging = try Staging.initHost(gpa, 8192); // small slots so spans cross several of them
    defer staging.deinit();
    var f = try DirectFile.open(path);
    defer f.close();
    const spans = [_][2]usize{ .{ 0, len }, .{ 1, 10 }, .{ 4095, 2 }, .{ 5000, 20000 }, .{ len - 13, 13 }, .{ 8192, 8192 } };
    for (spans) |s| {
        const out = try gpa.alloc(u8, s[1]);
        defer gpa.free(out);
        try f.read(&staging, s[0], out);
        try std.testing.expectEqualSlices(u8, data[s[0] .. s[0] + s[1]], out);
    }
    var past: [100]u8 = undefined;
    try std.testing.expectError(error.DirectRead, f.read(&staging, len - 4, &past));
    try std.testing.expectError(error.NotRegularFile, dropPageCache(try std.fmt.bufPrintSentinel(&path_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0)));
    _ = try dropPageCache(try std.fmt.bufPrintSentinel(&path_buf, ".zig-cache/tmp/{s}/shard.bin", .{tmp.sub_path}, 0));
}

test "mlock fails soft and pairs with munlock" {
    const gpa = std.testing.allocator;
    const b = try gpa.alignedAlloc(u8, .fromByteUnits(direct_align), direct_align);
    defer gpa.free(b);
    if (lockSoft(b)) unlockSoft(b);
    try std.testing.expect(!lockSoft(b[0..0]));
}

test "uma declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(PinnedHost);
    std.testing.refAllDecls(Staging);
    std.testing.refAllDecls(DirectFile);
}
