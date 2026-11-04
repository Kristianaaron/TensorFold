// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Adapted from ashhart/TensorFold@d1018b31: zig/src/sessions/format.zig and disk.zig (Apache-2.0).
//! Changes: family-specific saved-prefix KV layout; DCP2 planes are stored losslessly by position; the body is
//! streamed in fixed chunks (each with its own SHA-256) so a multi-GB prefix never sits whole in host memory.
//!
//! File layout (all offsets and lengths 4 KiB aligned, so the file can be read and written with O_DIRECT):
//!   [0, header_bytes)            Header, then `digest_count` SHA-256 digests (one per body chunk), zero padded
//!   [header_bytes, body_offset)  delta token ids [base_at, at), little-endian u32, zero padded
//!   [body_offset, end)           v2 body: padded ring, then delta KV planes in arbitrary-order chunks
//! The header is written last (after every chunk), and the file is published by rename, so a reader never sees a
//! partly written file under the final name.
const std = @import("std");
const builtin = @import("builtin");

const Sha256 = std.crypto.hash.sha2.Sha256;
pub const format_version: u32 = 2;
/// One body chunk: the size of one uma.Staging slot (cuda/uma.zig stage_bytes), so a pinned slot holds a chunk.
pub const chunk_bytes: usize = 4 << 20;
const write_behind_chunks: usize = 16;
const write_behind_bytes: usize = write_behind_chunks * chunk_bytes;
pub const alignment: usize = 4096;
/// A 6 GB rank body (a 126,976-token prefix) is ~1,450 chunks; 64k chunks bound the digest table at 2 MiB.
pub const max_chunks: usize = 1 << 16;
pub const magic = "TFKVGLM2".*;

pub const Digest = [32]u8;

/// Aligned cuts must preserve complete expert groups as well as the 512-row prefill boundary.
pub fn cutAlignment(expert_rows: usize) !usize {
    if (expert_rows == 0) return error.BadExpertRows;
    var a: usize = 512;
    var b = expert_rows;
    while (b != 0) : ({ const rem = a % b; a = b; b = rem; }) {}
    return 512 / a * expert_rows;
}

pub fn roundCutDown(cut: usize, expert_rows: usize) !usize {
    const a = try cutAlignment(expert_rows);
    return cut / a * a;
}

/// Incremental Wyhash key for a full prefix, updated in 512-id pieces.
pub fn streamedKey(ids: []const u32) !u64 {
    if (ids.len % 512 != 0) return error.BadAlignment;
    var state = std.hash.Wyhash.init(0x1e47);
    var at: usize = 0;
    while (at < ids.len) : (at += 512) state.update(std.mem.sliceAsBytes(ids[at..][0..512]));
    return state.final();
}

pub const Header = extern struct {
    magic_bytes: [8]u8 = magic,
    version: u32 = format_version,
    header_bytes: u32 = 0,
    header_sha: Digest = @splat(0),
    identity: Digest,
    key: u64,
    base_at: u64,
    parent_key: u64,
    seq: u64,
    rank: u32,
    world: u32,
    at: u64,
    ids_len: u64,
    body_len: u64,
    /// The DFlash2 ring's `from` at the cut; 0 without a drafter.
    ring_from: i64 = 0,
    ring_span: u64 = 0,
    digest_count: u32,
    _reserved: u32 = 0,
};

pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, alignment);
}

pub fn keyOf(ids: []const u32) u64 {
    return keyOfBytes(std.mem.sliceAsBytes(ids));
}

/// Match core/prompt_imprint.zig's Imprint.keyOf exactly (Wyhash seed 0x1e47 over the ids' bytes).
pub fn keyOfBytes(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0x1e47, bytes);
}

pub fn identityOf(settings_digest: Digest, fatbin_hash: Digest, exe_hash: Digest, rank: u32, world: u32, alignment_rows: u32) Digest {
    var h = Sha256.init(.{});
    h.update("tfkv-identity-v2\x00");
    h.update(&settings_digest);
    h.update(&fatbin_hash);
    h.update(&exe_hash);
    var word: [4]u8 = undefined;
    for ([_]u32{ format_version, rank, world, alignment_rows }) |v| {
        std.mem.writeInt(u32, &word, v, .little);
        h.update(&word);
    }
    var out: Digest = undefined;
    h.final(&out);
    return out;
}

pub fn digestCount(body_len: usize) usize {
    return (body_len + chunk_bytes - 1) / chunk_bytes;
}

pub fn headerBytes(digest_count: usize) usize {
    return aligned(@sizeOf(Header) + digest_count * @sizeOf(Digest));
}

pub fn fileBytes(body_len: usize, ids_len: usize) u64 {
    return @intCast(headerBytes(digestCount(body_len)) + aligned(4 * ids_len) + aligned(body_len));
}

/// `root/<identity hex>/rank<r>/<key hex>.tfkv`.
pub fn pathFor(gpa: std.mem.Allocator, root: []const u8, identity: Digest, rank: u32, key: u64) ![:0]u8 {
    const hex = std.fmt.bytesToHex(identity, .lower);
    return std.fmt.allocPrintSentinel(gpa, "{s}/{s}/rank{d}/{x:0>16}.tfkv", .{ root, &hex, rank, key }, 0);
}

pub fn pathInto(buf: []u8, root: []const u8, identity: Digest, rank: u32, key: u64) ![:0]u8 {
    const hex = std.fmt.bytesToHex(identity, .lower);
    return std.mem.printSentinel(buf, "{s}/{s}/rank{d}/{x:0>16}.tfkv", .{ root, &hex, rank, key }, 0);
}

const StatCount = if (builtin.os.tag == .linux) u64 else u32;
const Stat = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: StatCount,
    bfree: StatCount,
    bavail: StatCount,
    files: StatCount,
    ffree: StatCount,
    favail: StatCount,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [if (builtin.os.tag == .linux) 24 else 0]u8,
};
extern "c" fn statvfs(path: [*:0]const u8, result: *Stat) c_int;

pub fn freeBytes(dir: [:0]const u8) ?u64 {
    if (builtin.os.tag == .windows) return null;
    var stat: Stat = undefined;
    if (statvfs(dir.ptr, &stat) != 0) return null;
    return std.math.mul(u64, stat.frsize, stat.bavail) catch null;
}

/// `path`'s directories (root/<identity>/rank<r>), made as needed.
pub fn makeParents(path: []const u8) !void {
    const dir = std.fs.path.dirname(path) orelse return;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var at: usize = 0;
    while (at < dir.len) {
        const next = std.mem.indexOfScalarPos(u8, dir, at + 1, '/') orelse dir.len;
        if (next >= buf.len) return error.NameTooLong;
        @memcpy(buf[0..next], dir[0..next]);
        buf[next] = 0;
        if (next > 0 and std.c.mkdir(@ptrCast(&buf), 0o700) != 0 and std.c._errno().* != @intFromEnum(std.c.E.EXIST)) return error.DiskMkdir;
        at = next;
    }
}

/// The header block (Header + digests, zero padded) with its checksum filled in.
fn headerBlock(gpa: std.mem.Allocator, h_in: Header, digests: []const Digest) ![]align(alignment) u8 {
    const hb = headerBytes(digests.len);
    const block = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), hb);
    @memset(block, 0);
    var h = h_in;
    h.header_bytes = @intCast(hb);
    h.digest_count = @intCast(digests.len);
    h.header_sha = @splat(0);
    @memcpy(block[0..@sizeOf(Header)], std.mem.asBytes(&h));
    @memcpy(block[@sizeOf(Header)..][0 .. digests.len * @sizeOf(Digest)], std.mem.sliceAsBytes(digests));
    var sha: Digest = undefined;
    Sha256.hash(block[0 .. @sizeOf(Header) + digests.len * @sizeOf(Digest)], &sha, .{});
    h.header_sha = sha;
    @memcpy(block[0..@sizeOf(Header)], std.mem.asBytes(&h));
    return block;
}

/// fsync `path`'s directory, so the rename that published the file survives a crash.
fn syncParent(path: []const u8) !void {
    const dir = std.fs.path.dirname(path) orelse ".";
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (dir.len >= buf.len) return error.NameTooLong;
    @memcpy(buf[0..dir.len], dir);
    buf[dir.len] = 0;
    const fd = std.c.open(@ptrCast(&buf), .{ .ACCMODE = .RDONLY, .DIRECTORY = true });
    if (fd < 0) return error.DiskOpen;
    defer _ = std.c.close(fd);
    if (std.c.fsync(fd) != 0) return error.DiskSync;
}

fn setDirect(flags: *std.c.O, enabled: bool) bool {
    if (comptime @hasField(std.c.O, "DIRECT")) {
        flags.DIRECT = enabled;
        return true;
    }
    return !enabled;
}

fn pwriteAll(fd: c_int, bytes: []const u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pwrite(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n <= 0) return error.DiskWrite;
        done += @intCast(n);
    }
}

fn preadAll(fd: c_int, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const n = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
        if (n <= 0) return error.DiskRead;
        done += @intCast(n);
    }
}

fn prefixCacheEnabled() bool {
    const raw = std.c.getenv("TF_GLM53_PREFIX_CACHE") orelse return false;
    return raw[0] == '1' and raw[1] == 0;
}

var fadvise_error_logged: std.atomic.Value(bool) = .init(false);
var write_behind_error_logged: std.atomic.Value(bool) = .init(false);

const SyncFileRangeFn = *const fn (ctx: ?*anyopaque, fd: c_int, offset: u64, len: usize) void;

fn syncFileRangeLinux(_: ?*anyopaque, fd: c_int, offset: u64, len: usize) void {
    if (comptime builtin.os.tag != .linux) return;
    const file_fd: usize = @intCast(fd);
    const range_offset: usize = @intCast(offset);
    const range_len: usize = len;
    const sync_result = if (comptime builtin.cpu.arch == .aarch64)
        std.os.linux.syscall4(@enumFromInt(84), file_fd, range_offset, range_len, 7)
    else if (comptime builtin.cpu.arch == .x86_64)
        std.os.linux.syscall4(@enumFromInt(277), file_fd, range_offset, range_len, 7)
    else
        return;
    if (std.os.linux.errno(sync_result) != .SUCCESS and !write_behind_error_logged.swap(true, .monotonic)) {
        const err = std.os.linux.errno(sync_result);
        std.log.warn("prefix cache: sync_file_range failed: {s}", .{@tagName(err)});
    }
}

fn writeBehindCompletedWindow(written: []const bool, body_len: usize, body_offset: u64, i: usize, fd: c_int, ctx: ?*anyopaque, sync_fn: SyncFileRangeFn) void {
    const chunk_count = digestCount(body_len);
    const first = i / write_behind_chunks * write_behind_chunks;
    const end = @min(first + write_behind_chunks, chunk_count);
    if (end == 0 or end > written.len) return;
    for (written[first..end]) |ok| if (!ok) return;
    const first_offset = first * chunk_bytes;
    const range_len = @min(end * chunk_bytes, body_len) - first_offset;
    sync_fn(ctx, fd, body_offset + @as(u64, @intCast(first_offset)), range_len);
}

/// Advise Linux to discard file pages. This is best-effort: advice errors are logged once and ignored.
fn adviseDontNeed(fd: c_int) void {
    if (comptime builtin.os.tag != .linux) return;
    const result = std.os.linux.fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
    if (result != 0 and !fadvise_error_logged.swap(true, .monotonic)) {
        const err = std.os.linux.errno(result);
        std.log.warn("prefix cache: fadvise DONTNEED failed: {s}", .{@tagName(err)});
    }
}

/// Flush and evict a bounded range while streaming a save. The body uses O_DIRECT, but
/// keeping this range discipline also covers filesystems that still account write pages.
fn writeBehind(fd: c_int, offset: u64, len: usize) void {
    if (!prefixCacheEnabled()) return;
    if (comptime builtin.os.tag != .linux) return;
    syncFileRangeLinux(null, fd, offset, len);
    const file_fd: usize = @intCast(fd);
    const range_offset: usize = @intCast(offset);
    const range_len: usize = len;
    const advise_result = if (comptime builtin.cpu.arch == .aarch64)
        std.os.linux.syscall4(@enumFromInt(223), file_fd, range_offset, range_len, 4)
    else if (comptime builtin.cpu.arch == .x86_64)
        std.os.linux.syscall4(@enumFromInt(221), file_fd, range_offset, range_len, 4)
    else
        return;
    if (std.os.linux.errno(advise_result) != .SUCCESS and !write_behind_error_logged.swap(true, .monotonic)) {
        const err = std.os.linux.errno(advise_result);
        std.log.warn("prefix cache: write fadvise DONTNEED failed: {s}", .{@tagName(err)});
    }
}

fn writeBehindLinux(_: ?*anyopaque, fd: c_int, offset: u64, len: usize) void {
    writeBehind(fd, offset, len);
}

/// Clean pages of `path` out of the page cache (unified memory: cached pages take the GPU's). Linux only.
pub fn dropPageCache(path: [:0]const u8) bool {
    if (comptime @import("builtin").os.tag != .linux) return false;
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    return std.os.linux.fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.DONTNEED) == 0;
}

/// Streams one rank's saved prefix to `<path>.tmp` and publishes it by rename in `finish`.
pub const Writer = struct {
    gpa: std.mem.Allocator,
    path: [:0]const u8,
    tmp: [:0]u8,
    fd: c_int,
    direct: bool,
    header: Header,
    digests: []Digest,
    body_offset: u64,
    written: []bool,
    sealed: bool = false,
    sync_fn: SyncFileRangeFn = &writeBehindLinux,
    sync_ctx: ?*anyopaque = null,

    /// `body_len` is fixed up front (it sizes the digest table, and so the body's offset).
    pub fn begin(gpa: std.mem.Allocator, path: [:0]const u8, identity: Digest, rank: u32, world: u32, ids_full: []const u32, base_at: usize, at: usize, parent_key: u64, seq: u64, body_len: usize, ring_from: i64, ring_span: usize) !Writer {
        if (world == 0 or rank >= world or at <= base_at or at > ids_full.len or body_len == 0 or ring_span > body_len or ring_span % chunk_bytes != 0) return error.BadHeader;
        const nd = digestCount(body_len);
        if (nd > max_chunks) return error.BodyTooLarge;
        try makeParents(path);
        const tmp = try std.fmt.allocPrintSentinel(gpa, "{s}.tmp", .{path}, 0);
        errdefer gpa.free(tmp);
        var flags: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
        if (!setDirect(&flags, true)) return error.ODirectUnavailable;
        const fd = std.c.open(tmp.ptr, flags, @as(std.c.mode_t, 0o600));
        const direct = true;
        if (fd < 0) {
            return error.DiskOpen;
        }
        errdefer {
            _ = std.c.close(fd);
            _ = std.c.unlink(tmp.ptr);
        }
        const digests = try gpa.alloc(Digest, nd);
        errdefer gpa.free(digests);
        const written = try gpa.alloc(bool, nd);
        errdefer gpa.free(written);
        @memset(written, false);
        const hb = headerBytes(nd);
        const ids = ids_full[base_at..at];
        const ids_bytes = std.mem.sliceAsBytes(ids);
        const ids_span = aligned(ids_bytes.len);
        var blk: [64 * 1024]u8 align(alignment) = undefined;
        var off: usize = 0;
        while (off < ids_bytes.len) {
            const len: usize = @min(blk.len, ids_bytes.len - off);
            const padded = aligned(len);
            @memset(blk[0..padded], 0);
            @memcpy(blk[0..len], ids_bytes[off..][0..len]);
            try pwriteAll(fd, blk[0..padded], hb + off);
            off += len;
        }
        return .{
            .gpa = gpa,
            .path = path,
            .tmp = tmp,
            .fd = fd,
            .direct = direct,
            .header = .{ .identity = identity, .key = keyOf(ids_full[0..at]), .base_at = base_at, .parent_key = parent_key, .seq = seq, .rank = rank, .world = world, .at = at, .ids_len = ids.len, .body_len = body_len, .ring_from = ring_from, .ring_span = ring_span, .digest_count = @intCast(nd) },
            .digests = digests,
            .written = written,
            .body_offset = hb + ids_span,
        };
    }

    /// The next chunk: `buf[0..n]` is body data (n == chunk_bytes except for the last chunk), `buf` is 4 KiB aligned
    /// and holds at least aligned(n) bytes; the tail up to aligned(n) is zeroed here before the write.
    pub fn writeChunkAt(w: *Writer, i: usize, buf: []align(alignment) u8, n: usize) !void {
        if (w.sealed) return error.WriterSealed;
        if (i >= w.digests.len) return error.TooManyChunks;
        if (w.written[i]) return error.DuplicateChunk;
        const last = i + 1 == w.digests.len;
        const want = if (last) w.header.body_len - i * chunk_bytes else chunk_bytes;
        if (n != want or buf.len < aligned(n)) return error.BadChunk;
        Sha256.hash(buf[0..n], &w.digests[i], .{});
        @memset(buf[n..aligned(n)], 0);
        try pwriteAll(w.fd, buf[0..aligned(n)], w.body_offset + i * chunk_bytes);
        w.written[i] = true;
        if (prefixCacheEnabled() or w.sync_fn != &writeBehindLinux) writeBehindCompletedWindow(w.written, @intCast(w.header.body_len), w.body_offset, i, w.fd, w.sync_ctx, w.sync_fn);
    }

    pub fn writeChunk(w: *Writer, buf: []align(alignment) u8, n: usize) !void {
        var i: usize = 0;
        while (i < w.written.len and w.written[i]) : (i += 1) {}
        try w.writeChunkAt(i, buf, n);
    }

    /// Write the header last and fsync the temporary file; publish remains a separate rename step.
    pub fn seal(w: *Writer) !void {
        if (w.fd < 0) return error.WriterClosed;
        for (w.written) |ok| if (!ok) return error.MissingChunks;
        const block = try headerBlock(w.gpa, w.header, w.digests);
        defer w.gpa.free(block);
        try pwriteAll(w.fd, block, 0);
        if (prefixCacheEnabled()) {
            if (comptime builtin.os.tag == .linux) {
                if (std.os.linux.errno(std.os.linux.fdatasync(w.fd)) != .SUCCESS) return error.DiskSync;
            } else if (std.c.fsync(w.fd) != 0) return error.DiskSync;
        } else if (std.c.fsync(w.fd) != 0) return error.DiskSync;
        w.sealed = true;
    }

    pub fn publish(w: *Writer) !void {
        defer w.release();
        if (w.fd < 0) return error.WriterClosed;
        if (!w.sealed) return error.WriterNotSealed;
        if (prefixCacheEnabled()) adviseDontNeed(w.fd);
        if (std.c.close(w.fd) != 0) { w.fd = -1; return error.DiskClose; }
        w.fd = -1;
        if (std.c.rename(w.tmp.ptr, w.path.ptr) != 0) {
            _ = std.c.unlink(w.tmp.ptr);
            return error.DiskRename;
        }
        try syncParent(w.path);
    }

    pub fn finish(w: *Writer) !void {
        w.seal() catch |err| {
            w.abort();
            return err;
        };
        try w.publish();
    }

    /// Give up: the temporary file is removed and the final name is untouched.
    pub fn abort(w: *Writer) void {
        if (w.fd >= 0) _ = std.c.close(w.fd);
        w.fd = -1;
        _ = std.c.unlink(w.tmp.ptr);
        w.release();
    }

    fn release(w: *Writer) void {
        if (w.fd >= 0) {
            _ = std.c.close(w.fd);
            w.fd = -1;
        }
        _ = std.c.unlink(w.tmp.ptr);
        w.gpa.free(w.digests);
        w.gpa.free(w.written);
        w.gpa.free(w.tmp);
        w.digests = &.{};
        w.tmp = undefined;
    }
};

/// Opens and validates one rank's saved prefix before any device state changes, then hands out verified chunks.
pub const Reader = struct {
    gpa: std.mem.Allocator,
    path: [:0]const u8,
    fd: c_int,
    direct: bool,
    header: Header,
    digests: []Digest,
    body_offset: u64,

    /// Validate the fixed v2 header with one O_DIRECT sector read; digest tables, token ids, and body are untouched.
    pub fn headerOnly(gpa: std.mem.Allocator, path: [:0]const u8, identity: Digest) !Header {
        var flags: std.c.O = .{ .ACCMODE = .RDONLY };
        if (!setDirect(&flags, true)) return error.ODirectUnavailable;
        const fd = std.c.open(path.ptr, flags);
        if (fd < 0) return error.DiskOpen;
        defer _ = std.c.close(fd);
        const first = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), alignment);
        defer gpa.free(first);
        try preadAll(fd, first, 0);
        var h: Header = undefined;
        @memcpy(std.mem.asBytes(&h), first[0..@sizeOf(Header)]);
        if (!std.mem.eql(u8, &h.magic_bytes, &magic) or h.version != format_version) return error.BadHeader;
        if (!std.mem.eql(u8, &h.identity, &identity)) return error.IdentityMismatch;
        if (h.digest_count > max_chunks or h.header_bytes != headerBytes(h.digest_count) or h.header_bytes % alignment != 0) return error.BadHeader;
        if (h.at == 0 or h.base_at >= h.at or h.ids_len != h.at - h.base_at or h.body_len == 0 or
            h.digest_count != digestCount(std.math.cast(usize, h.body_len) orelse return error.BadHeader) or
            h.ring_span > h.body_len or h.ring_span % chunk_bytes != 0) return error.BadHeader;
        return h;
    }

    /// Everything a reader checks up front: magic, version, header checksum, identity, rank/world, the prefix length,
    /// the body length the caller expects, and the token ids themselves (byte equal; the key is checked too).
    pub fn open(gpa: std.mem.Allocator, path: [:0]const u8, identity: Digest, rank: u32, world: u32, ids_full: []const u32, base_at: usize, at: usize, body_len: usize) !Reader {
        var flags: std.c.O = .{ .ACCMODE = .RDONLY };
        if (!setDirect(&flags, true)) return error.ODirectUnavailable;
        const fd = std.c.open(path.ptr, flags);
        const direct = true;
        if (fd < 0) {
            return error.DiskOpen;
        }
        errdefer _ = std.c.close(fd);
        const first = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), alignment);
        defer gpa.free(first);
        try preadAll(fd, first, 0);
        var h: Header = undefined;
        @memcpy(std.mem.asBytes(&h), first[0..@sizeOf(Header)]);
        if (!std.mem.eql(u8, &h.magic_bytes, &magic) or h.version != format_version) return error.BadHeader;
        if (h.digest_count > max_chunks or h.header_bytes != headerBytes(h.digest_count)) return error.BadHeader;
        const block = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), h.header_bytes);
        defer gpa.free(block);
        try preadAll(fd, block, 0);
        var check_h = h;
        check_h.header_sha = @splat(0);
        var hash = Sha256.init(.{});
        hash.update(std.mem.asBytes(&check_h));
        hash.update(block[@sizeOf(Header)..][0 .. @as(usize, h.digest_count) * @sizeOf(Digest)]);
        var sha: Digest = undefined;
        hash.final(&sha);
        if (!std.mem.eql(u8, &sha, &h.header_sha)) return error.HeaderChecksum;
        if (!std.mem.eql(u8, &h.identity, &identity)) return error.IdentityMismatch;
        if (base_at >= at or at > ids_full.len) return error.BadHeader;
        if (h.rank != rank or h.world != world or h.at != at or h.base_at != base_at or h.ids_len != at - base_at or h.body_len != body_len) return error.BadHeader;
        if (h.ring_span > h.body_len or h.ring_span % chunk_bytes != 0) return error.BadHeader;
        if (h.digest_count != digestCount(body_len)) return error.BadHeader;
        const ids_bytes = std.mem.sliceAsBytes(ids_full[base_at..at]);
        var blk: [64 * 1024]u8 align(alignment) = undefined;
        var off: usize = 0;
        while (off < ids_bytes.len) {
            const len: usize = @min(blk.len, ids_bytes.len - off);
            try preadAll(fd, blk[0..aligned(len)], h.header_bytes + off);
            if (!std.mem.eql(u8, blk[0..len], ids_bytes[off..][0..len])) return error.TokenMismatch;
            off += len;
        }
        if (keyOf(ids_full[0..at]) != h.key) return error.TokenMismatch;
        const digests = try gpa.alloc(Digest, h.digest_count);
        errdefer gpa.free(digests);
        @memcpy(std.mem.sliceAsBytes(digests), block[@sizeOf(Header)..][0 .. digests.len * @sizeOf(Digest)]);
        return .{ .gpa = gpa, .path = path, .fd = fd, .direct = direct, .header = h, .digests = digests, .body_offset = h.header_bytes + aligned(ids_bytes.len) };
    }

    /// Chunk `i` into `buf` (4 KiB aligned, at least chunk_bytes long), verified against its digest; the data bytes.
    pub fn readChunk(r: *const Reader, i: usize, buf: []align(alignment) u8) ![]const u8 {
        return r.readChunkAt(i, buf);
    }

    pub fn readChunkAt(r: *const Reader, i: usize, buf: []align(alignment) u8) ![]const u8 {
        if (i >= r.digests.len or buf.len < chunk_bytes) return error.BadChunk;
        const n: usize = if (i + 1 == r.digests.len) @intCast(r.header.body_len - i * chunk_bytes) else chunk_bytes;
        try preadAll(r.fd, buf[0..aligned(n)], r.body_offset + i * chunk_bytes);
        if (prefixCacheEnabled()) {
            if (comptime builtin.os.tag == .linux) {
                const result = std.os.linux.fadvise(r.fd, @intCast(r.body_offset + i * chunk_bytes), @intCast(aligned(n)), std.os.linux.POSIX_FADV.DONTNEED);
                if (result != 0 and !fadvise_error_logged.swap(true, .monotonic)) {
                    const err = std.os.linux.errno(result);
                    std.log.warn("prefix cache: restore fadvise DONTNEED failed: {s}", .{@tagName(err)});
                }
            }
        }
        var sha: Digest = undefined;
        Sha256.hash(buf[0..n], &sha, .{});
        if (!std.mem.eql(u8, &sha, &r.digests[i])) return error.ChunkChecksum;
        return buf[0..n];
    }

    pub fn close(r: *Reader) void {
        if (prefixCacheEnabled()) adviseDontNeed(r.fd);
        _ = std.c.close(r.fd);
        if (!r.direct) _ = dropPageCache(r.path);
        r.gpa.free(r.digests);
        r.* = undefined;
    }
};

/// DCP2 position p belongs to rank p%2 at local slot p/2. Saved cuts are multiples of 512, so each rank holds
/// at/2 contiguous rows [0, at/2) of every plane.
pub fn localRows(rank: usize, at: usize) !usize {
    if (rank > 1) return error.BadRank;
    return if (at <= rank) 0 else (at - rank + 1) / 2;
}

pub fn localDeltaRows(rank: usize, base_at: usize, at: usize) !usize {
    if (base_at > at or (at - base_at) % 2 != 0) return error.BadAlignment;
    return try localRows(rank, at - base_at);
}

pub fn owner(position: usize) usize {
    return position % 2;
}

pub fn localSlot(position: usize) usize {
    return position / 2;
}

fn writeBody(gpa: std.mem.Allocator, w: *Writer, body: []const u8) !void {
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(buf);
    var at: usize = 0;
    while (at < body.len) {
        const n: usize = @min(chunk_bytes, body.len - at);
        @memcpy(buf[0..n], body[at..][0..n]);
        try w.writeChunk(buf, n);
        at += n;
    }
}

fn readBody(gpa: std.mem.Allocator, r: *const Reader, out: []u8) !void {
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(buf);
    var at: usize = 0;
    for (0..r.digests.len) |i| {
        const data = try r.readChunk(i, buf);
        @memcpy(out[at..][0..data.len], data);
        at += data.len;
    }
    if (at != out.len) return error.Truncated;
}

fn testPath(gpa: std.mem.Allocator, tmp: std.testing.TmpDir, name: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name }, 0);
}

const WriteBehindRanges = struct {
    offsets: [2]u64 = @splat(0),
    lengths: [2]usize = @splat(0),
    count: usize = 0,
};

fn recordWriteBehindRange(ctx: ?*anyopaque, _: c_int, offset: u64, len: usize) void {
    const ranges: *WriteBehindRanges = @ptrCast(@alignCast(ctx.?));
    ranges.offsets[ranges.count] = offset;
    ranges.lengths[ranges.count] = len;
    ranges.count += 1;
}

test "128 MiB writer body issues one sync range per 64 MiB window" {
    const body_len: usize = 128 << 20;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try testPath(gpa, tmp, "write-behind.tfkv");
    defer gpa.free(path);
    const ids = [_]u32{ 1, 2 };
    var writer = try Writer.begin(gpa, path, @splat(1), 0, 2, &ids, 0, ids.len, 0, 1, body_len, 0, 0);
    var ranges: WriteBehindRanges = .{};
    writer.sync_fn = recordWriteBehindRange;
    writer.sync_ctx = &ranges;
    const chunk = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(chunk);
    @memset(chunk, 0x5a);
    for (0..digestCount(body_len)) |i| {
        const n = @min(chunk_bytes, body_len - i * chunk_bytes);
        try writer.writeChunk(chunk, n);
    }
    const body_offset = writer.body_offset;
    try writer.finish();
    try std.testing.expectEqual(@as(usize, 2), ranges.count);
    try std.testing.expectEqual(body_offset, ranges.offsets[0]);
    try std.testing.expectEqual(body_offset + @as(u64, write_behind_bytes), ranges.offsets[1]);
    try std.testing.expectEqual(write_behind_bytes, ranges.lengths[0]);
    try std.testing.expectEqual(write_behind_bytes, ranges.lengths[1]);
}

test "saved prefix streams through chunks and round trips with every header check" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try testPath(gpa, tmp, "kv/id/rank1/a.tfkv");
    defer gpa.free(path);
    const ids = [_]u32{ 3, 5, 8, 13, 21 };
    const identity = identityOf(@splat(1), @splat(2), @splat(3), 1, 2, 512);
    const body = try gpa.alloc(u8, 2 * chunk_bytes + 4099);
    defer gpa.free(body);
    for (body, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    var w = try Writer.begin(gpa, path, identity, 1, 2, &ids, 0, ids.len, 0, 1, body.len, 42, 0);
    try writeBody(gpa, &w, body);
    try w.finish();
    var r = try Reader.open(gpa, path, identity, 1, 2, &ids, 0, ids.len, body.len);
    defer r.close();
    try std.testing.expectEqual(@as(i64, 42), r.header.ring_from);
    const back = try gpa.alloc(u8, body.len);
    defer gpa.free(back);
    try readBody(gpa, &r, back);
    try std.testing.expectEqualSlices(u8, body, back);
    try std.testing.expectError(error.BadHeader, Reader.open(gpa, path, identity, 0, 2, &ids, 0, ids.len, body.len));
    try std.testing.expectError(error.BadHeader, Reader.open(gpa, path, identity, 1, 2, &ids, 0, ids.len, body.len - 1));
    try std.testing.expectError(error.IdentityMismatch, Reader.open(gpa, path, identityOf(@splat(9), @splat(2), @splat(3), 1, 2, 512), 1, 2, &ids, 0, ids.len, body.len));
    try std.testing.expectError(error.BadHeader, Reader.open(gpa, path, identity, 1, 2, ids[0..4], 0, 4, body.len));
    try std.testing.expectError(error.TokenMismatch, Reader.open(gpa, path, identity, 1, 2, &.{ 3, 5, 8, 13, 22 }, 0, 5, body.len));
    const missing = try testPath(gpa, tmp, "kv/id/rank1/missing.tfkv");
    defer gpa.free(missing);
    try std.testing.expectError(error.DiskOpen, Reader.open(gpa, missing, identity, 1, 2, &ids, 0, ids.len, body.len));
}

test "a flipped body byte fails only its chunk; a flipped header byte fails the open" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try testPath(gpa, tmp, "b.tfkv");
    defer gpa.free(path);
    const ids = [_]u32{ 1, 2 };
    const identity = identityOf(@splat(3), @splat(4), @splat(5), 0, 2, 512);
    const body = try gpa.alloc(u8, chunk_bytes + 17);
    defer gpa.free(body);
    @memset(body, 0x5a);
    var w = try Writer.begin(gpa, path, identity, 0, 2, &ids, 0, ids.len, 0, 1, body.len, 0, 0);
    try writeBody(gpa, &w, body);
    try w.finish();
    const body_offset = headerBytes(2) + aligned(ids.len * 4);
    {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDWR });
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        const one = [1]u8{0x5b};
        try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &one, 1, @intCast(body_offset + chunk_bytes + 3)));
    }
    var r = try Reader.open(gpa, path, identity, 0, 2, &ids, 0, ids.len, body.len);
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(buf);
    _ = try r.readChunk(0, buf);
    try std.testing.expectError(error.ChunkChecksum, r.readChunk(1, buf));
    r.close();
    {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDWR });
        defer _ = std.c.close(fd);
        const one = [1]u8{0xff};
        try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &one, 1, @offsetOf(Header, "key")));
    }
    try std.testing.expectError(error.HeaderChecksum, Reader.open(gpa, path, identity, 0, 2, &ids, 0, ids.len, body.len));
}

test "an aborted or unfinished writer never publishes the final name" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try testPath(gpa, tmp, "c.tfkv");
    defer gpa.free(path);
    const ids = [_]u32{7};
    const identity = identityOf(@splat(5), @splat(6), @splat(7), 0, 1, 512);
    var w = try Writer.begin(gpa, path, identity, 0, 1, &ids, 0, ids.len, 0, 1, chunk_bytes + 1, 0, 0);
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(buf);
    @memset(buf, 1);
    try w.writeChunk(buf, chunk_bytes);
    try std.testing.expectError(error.BadChunk, w.writeChunk(buf, 2)); // the last chunk is exactly 1 byte
    w.abort();
    try std.testing.expectError(error.DiskOpen, Reader.open(gpa, path, identity, 0, 1, &ids, 0, ids.len, chunk_bytes + 1));
    var w2 = try Writer.begin(gpa, path, identity, 0, 1, &ids, 0, ids.len, 0, 1, chunk_bytes + 1, 0, 0);
    try w2.writeChunk(buf, chunk_bytes);
    try std.testing.expectError(error.MissingChunks, w2.finish());
    try std.testing.expectError(error.DiskOpen, Reader.open(gpa, path, identity, 0, 1, &ids, 0, ids.len, chunk_bytes + 1));
}

test "DCP2 row counts map 512-aligned cuts to contiguous local rows on both ranks" {
    for ([_]usize{ 512, 2048, 2560, 29696, 126976 }) |at| {
        try std.testing.expectEqual(at / 2, try localRows(0, at));
        try std.testing.expectEqual(at / 2, try localRows(1, at));
        for (0..at) |p| {
            const rank = owner(p);
            try std.testing.expectEqual(p % 2, rank);
            try std.testing.expectEqual(p / 2, localSlot(p));
            try std.testing.expect(localSlot(p) < try localRows(rank, at));
        }
    }
    try std.testing.expectError(error.BadRank, localRows(2, 512));
}

test "key and id derivation bind ordered tokens, settings, fatbin, rank and world" {
    const ids_a = [_]u32{ 1, 2, 3 };
    const ids_b = [_]u32{ 1, 3, 2 };
    try std.testing.expect(keyOf(&ids_a) != keyOf(&ids_b));
    try std.testing.expectEqual(keyOf(&ids_a), keyOfBytes(std.mem.sliceAsBytes(&ids_a)));
    const settings = @as(Digest, @splat(0x11));
    const fatbin = @as(Digest, @splat(0x22));
    const exe = @as(Digest, @splat(0x33));
    const id0 = identityOf(settings, fatbin, exe, 0, 2, 1024);
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(settings, fatbin, exe, 1, 2, 1024)));
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(settings, fatbin, exe, 0, 1, 1024)));
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(@splat(0x12), fatbin, exe, 0, 2, 1024)));
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(settings, @splat(0x23), exe, 0, 2, 1024)));
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(settings, fatbin, @splat(0x34), 0, 2, 1024)));
    try std.testing.expect(!std.mem.eql(u8, &id0, &identityOf(settings, fatbin, exe, 0, 2, 2048)));
    const path = try pathFor(std.testing.allocator, "/nvme/kv", id0, 1, keyOf(&ids_a));
    defer std.testing.allocator.free(path);
    try std.testing.expect(std.mem.startsWith(u8, path, "/nvme/kv/"));
    try std.testing.expect(std.mem.indexOf(u8, path, "/rank1/") != null);
    try std.testing.expect(std.mem.endsWith(u8, path, ".tfkv"));
}

test "cut rounding follows expert groups and streamed keys match full keys" {
    for ([_]usize{ 1024, 2048, 4096 }) |rows| {
        const a = try cutAlignment(rows);
        try std.testing.expectEqual(rows, a);
        try std.testing.expectEqual(@as(usize, 2 * rows), try roundCutDown(2 * rows + 511, rows));
        try std.testing.expectEqual(@as(usize, 0), try roundCutDown(rows - 1, rows));
    }
    const ids = try std.testing.allocator.alloc(u32, 2048);
    defer std.testing.allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i * 17 + 3);
    for ([_]usize{ 512, 1024, 1536, 2048 }) |at| try std.testing.expectEqual(keyOf(ids[0..at]), try streamedKey(ids[0..at]));
}

test "v2 stores delta ids and accepts body chunks in arbitrary order" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try testPath(gpa, tmp, "delta.tfkv");
    defer gpa.free(path);
    const ids = try gpa.alloc(u32, 1024);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i * 13 + 1);
    const body = try gpa.alloc(u8, 2 * chunk_bytes + 17);
    defer gpa.free(body);
    for (body, 0..) |*b, i| b.* = @truncate(i *% 7 +% 19);
    const identity = identityOf(@splat(1), @splat(2), @splat(3), 1, 2, 1024);
    var w = try Writer.begin(gpa, path, identity, 1, 2, ids, 512, 1024, keyOf(ids[0..512]), 9, body.len, 77, chunk_bytes);
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), chunk_bytes);
    defer gpa.free(buf);
    for ([_]usize{ 2, 0, 1 }) |i| {
        const start = i * chunk_bytes;
        const n: usize = @min(chunk_bytes, body.len - start);
        @memcpy(buf[0..n], body[start..][0..n]);
        try w.writeChunkAt(i, buf, n);
    }
    try w.seal();
    // A sealed .tmp must remain invisible under the final name.
    try std.testing.expectError(error.DiskOpen, Reader.open(gpa, path, identity, 1, 2, ids, 512, 1024, body.len));
    try w.publish();
    var r = try Reader.open(gpa, path, identity, 1, 2, ids, 512, 1024, body.len);
    defer r.close();
    try std.testing.expectEqual(@as(u64, 512), r.header.base_at);
    try std.testing.expectEqual(@as(u64, 512), r.header.ids_len);
    try std.testing.expectEqual(@as(u64, chunk_bytes), r.header.ring_span);
    var out_at: usize = 0;
    for (0..r.digests.len) |i| {
        const data = try r.readChunkAt(i, buf);
        try std.testing.expectEqualSlices(u8, body[out_at..][0..data.len], data);
        out_at += data.len;
    }
}
