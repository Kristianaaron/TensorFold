// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! A model folder's safetensors as one name space (every backend's index): shard files, tensors mapped, leftovers refused.

const std = @import("std");
const Io = std.Io;
const st = @import("safetensors.zig");

pub const Tensor = st.Tensor;
pub const DType = st.DType;

pub const PackFile = struct { path: [:0]u8, layer: u32, width: u32 };
pub const PackSelect = struct { ctx: *anyopaque, keep: *const fn (ctx: *anyopaque, layer: u32, width: u32, tensor: []const u8) bool };

/// The checkpoint's shard paths in name order: model.safetensors.index.json's weight map, else every model*.safetensors.
pub fn shardFiles(gpa: std.mem.Allocator, io: Io, dir: []const u8) ![][:0]u8 {
    return shardFilesBut(gpa, io, dir, false);
}

fn shardFilesBut(gpa: std.mem.Allocator, io: Io, dir: []const u8, skip_packs: bool) ![][:0]u8 {
    var names: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    if (Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26))) |text| {
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(gpa);
        var it = parsed.value.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| {
            if (skip_packs and std.mem.startsWith(u8, e.value_ptr.string, "packs/")) continue;
            const file = e.value_ptr.string;
            if ((try seen.getOrPut(gpa, file)).found_existing) continue;
            try names.ensureUnusedCapacity(gpa, 1);
            names.appendAssumeCapacity(try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, file }, 0));
        }
    } else |_| {
        var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.startsWith(u8, e.name, "model") or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            try names.ensureUnusedCapacity(gpa, 1);
            names.appendAssumeCapacity(try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, e.name }, 0));
        }
    }
    if (names.items.len == 0) return error.NoSafetensors;
    std.mem.sort([:0]u8, names.items, {}, struct {
        fn lt(_: void, a: [:0]u8, b: [:0]u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    return names.toOwnedSlice(gpa);
}

pub fn packFiles(gpa: std.mem.Allocator, io: Io, dir: []const u8) ![]PackFile {
    const root = try std.fs.path.join(gpa, &.{ dir, "packs" });
    defer gpa.free(root);
    var out: std.ArrayList(PackFile) = .empty;
    errdefer { for (out.items) |p| gpa.free(p.path); out.deinit(gpa); }
    var pd = Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch return error.NoPacks;
    defer pd.close(io);
    var layers = pd.iterate();
    while (try layers.next(io)) |le| {
        if (le.kind != .directory) continue;
        const layer = decimalFolder(le.name) orelse continue;
        const sub = try std.fs.path.join(gpa, &.{ root, le.name });
        defer gpa.free(sub);
        var ld = try Io.Dir.cwd().openDir(io, sub, .{ .iterate = true });
        defer ld.close(io);
        var files = ld.iterate();
        while (try files.next(io)) |fe| {
            if (fe.kind != .file and fe.kind != .sym_link) continue;
            const width = packWidth(fe.name) orelse continue;
            try out.append(gpa, .{ .path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}/{s}", .{ root, le.name, fe.name }, 0), .layer = layer, .width = width });
        }
    }
    if (out.items.len == 0) return error.NoPacks;
    std.mem.sort(PackFile, out.items, {}, struct { fn lt(_: void, a: PackFile, b: PackFile) bool { return std.mem.order(u8, a.path, b.path) == .lt; } }.lt);
    return out.toOwnedSlice(gpa);
}

pub fn freePackFiles(gpa: std.mem.Allocator, list: []PackFile) void { for (list) |p| gpa.free(p.path); gpa.free(list); }
fn decimalFolder(name: []const u8) ?u32 { if (name.len < 2 or name[0] != 'L') return null; return std.fmt.parseInt(u32, name[1..], 10) catch null; }
fn packWidth(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "K") or !std.mem.endsWith(u8, name, ".safetensors")) return null;
    const end = name.len - ".safetensors".len;
    const dot = std.mem.indexOfScalarPos(u8, name, 1, '.') orelse return null;
    if (dot >= end) return null;
    const suffix = name[dot..end];
    if (!std.mem.eql(u8, suffix, ".esel")) {
        if (!std.mem.startsWith(u8, suffix, ".e")) return null;
        const dash = std.mem.indexOfScalarPos(u8, suffix, 2, '-') orelse return null;
        if (std.mem.indexOfScalarPos(u8, suffix, dash + 1, '-') != null) return null;
        _ = std.fmt.parseInt(u32, suffix[2..dash], 10) catch return null;
        _ = std.fmt.parseInt(u32, suffix[dash + 1 ..], 10) catch return null;
    }
    return std.fmt.parseInt(u32, name[1..dot], 10) catch null;
}

pub fn freeShardFiles(gpa: std.mem.Allocator, files: [][:0]u8) void {
    for (files) |f| gpa.free(f);
    gpa.free(files);
}

pub const Checkpoint = struct {
    gpa: std.mem.Allocator,
    io: Io,
    files: std.ArrayList(st.File) = .empty,
    used: std.StringHashMapUnmanaged(void) = .empty,

    /// Prepended to the names get, has and skip take ("language_model." for Bolt's LM beside its vision tower).
    scope: []const u8 = "",

    fn scoped(self: *const Checkpoint, buf: *[320]u8, name: []const u8) ![]const u8 {
        if (self.scope.len == 0) return name;
        return std.fmt.bufPrint(buf, "{s}{s}", .{ self.scope, name });
    }

    /// Every shard of `dir`, mapped.
    pub fn openModel(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Checkpoint {
        return openModelPrefix(gpa, io, dir, null);
    }
    pub fn openModelPrefix(gpa: std.mem.Allocator, io: Io, dir: []const u8, prefix: ?[]const u8) !Checkpoint {
        try @import("save_guard").check(io, dir);
        const paths = try shardFiles(gpa, io, dir);
        defer freeShardFiles(gpa, paths);
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        errdefer ck.close();
        try ck.files.ensureTotalCapacityPrecise(gpa, paths.len);
        for (paths) |p| ck.files.appendAssumeCapacity(try st.File.openPrefix(gpa, io, p, prefix));
        return ck;
    }

    pub fn openModelPacks(gpa: std.mem.Allocator, io: Io, dir: []const u8, select: PackSelect) !Checkpoint {
        try @import("save_guard").check(io, dir);
        const shards: [][:0]u8 = shardFilesBut(gpa, io, dir, true) catch |err| switch (err) { error.NoSafetensors => @constCast(&.{}), else => return err };
        defer freeShardFiles(gpa, shards);
        const packs = try packFiles(gpa, io, dir);
        defer freePackFiles(gpa, packs);
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        errdefer ck.close();
        var names: std.StringHashMapUnmanaged(void) = .empty;
        defer names.deinit(gpa);
        for (shards) |p| try ck.files.append(gpa, try st.File.open(gpa, io, p));
        for (packs) |p| {
            const ctx = FilterCtx{ .select = select, .layer = p.layer, .width = p.width };
            var f = try st.File.openFiltered(gpa, io, p.path, .{ .ctx = @constCast(&ctx), .keep = filterPack });
            if (f.names.count() == 0) { f.close(io); continue; }
            try ck.files.append(gpa, f);
        }
        for (ck.files.items) |f| for (f.names.keys()) |name| {
            if ((try names.getOrPut(gpa, name)).found_existing) return error.DuplicateTensor;
        };
        return ck;
    }

    const FilterCtx = struct { select: PackSelect, layer: u32, width: u32 };
    fn filterPack(raw: *anyopaque, tensor: []const u8) bool { const c: *const FilterCtx = @ptrCast(@alignCast(raw)); return c.select.keep(c.select.ctx, c.layer, c.width, tensor); }

    /// One more file (an MTP head beside the model's files).
    pub fn add(self: *Checkpoint, dir: []const u8, name: []const u8) !void {
        const path = try std.fs.path.join(self.gpa, &.{ dir, name });
        defer self.gpa.free(path);
        try self.files.ensureUnusedCapacity(self.gpa, 1);
        self.files.appendAssumeCapacity(try st.File.open(self.gpa, self.io, path));
    }

    pub fn close(self: *Checkpoint) void {
        for (self.files.items) |*f| f.close(self.io);
        self.files.deinit(self.gpa);
        self.used.deinit(self.gpa);
        self.* = undefined;
    }

    /// The tensor named `name`, marked as used; a missing name is an error naming it.
    pub fn get(self: *Checkpoint, short: []const u8) !Tensor {
        var buf: [320]u8 = undefined;
        const name = try self.scoped(&buf, short);
        for (self.files.items) |*f| if (f.get(name)) |t| {
            try self.used.put(self.gpa, f.names.getKey(name).?, {});
            return t;
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// `get` with the dtype and shape checked.
    pub fn expect(self: *Checkpoint, name: []const u8, dtype: DType, shape: []const usize) !Tensor {
        const t = try self.get(name);
        if (!t.is(dtype, shape)) {
            std.log.err("{s}: {t} {any}, expected {t} {any}", .{ name, t.dtype, t.shape[0..t.rank], dtype, shape });
            return error.UnexpectedTensor;
        }
        return t;
    }

    /// Whether any file holds `name` (it is not marked as used).
    pub fn has(self: *const Checkpoint, short: []const u8) bool {
        var buf: [320]u8 = undefined;
        const name = self.scoped(&buf, short) catch return false;
        for (self.files.items) |*f| if (f.get(name) != null) return true;
        return false;
    }

    /// Marks every name starting with `prefix` as used: a part the loader leaves out on purpose.
    pub fn skip(self: *Checkpoint, short: []const u8) !void {
        var buf: [320]u8 = undefined;
        const prefix = try self.scoped(&buf, short);
        for (self.files.items) |*f| for (f.names.keys()) |k| {
            if (std.mem.startsWith(u8, k, prefix)) try self.used.put(self.gpa, k, {});
        };
    }

    /// Every file's pages out of the page cache (safetensors.File.release).
    pub fn release(self: *const Checkpoint) void {
        for (self.files.items) |*f| f.release();
    }

    /// Names no `get` took (the loader refuses a checkpoint with leftovers).
    pub fn unused(self: *const Checkpoint) usize {
        var n: usize = 0;
        for (self.files.items) |*f| for (f.names.keys()) |k| {
            if (self.used.contains(k)) continue;
            if (n < 5) std.log.err("unused checkpoint tensor {s}", .{k});
            n += 1;
        };
        return n;
    }
};

/// A safetensors image of one U8 tensor of `bytes` bytes named `name`.
fn testImage(a: std.mem.Allocator, name: []const u8, bytes: usize) ![]u8 {
    const header = try std.fmt.allocPrint(a, "{{\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ name, bytes, bytes });
    defer a.free(header);
    const out = try a.alloc(u8, 8 + header.len + bytes);
    std.mem.writeInt(u64, out[0..8], header.len, .little);
    @memcpy(out[8..][0..header.len], header);
    @memset(out[8 + header.len ..], 0);
    return out;
}

/// Fails each allocation of `run` in turn, and every resize; bytes must balance, and `run` must pass when none fails.
fn expectBalanced(comptime run: anytype, io: Io, dir: []const u8) !void {
    var i: usize = 0;
    while (true) : (i += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = i, .resize_fail_index = 0 });
        run(failing.allocator(), io, dir) catch |err| if (!failing.has_induced_failure) return err;
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) return;
    }
}

fn listShards(gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
    freeShardFiles(gpa, try shardFiles(gpa, io, dir));
}

fn openShards(gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    ck.close();
}

fn addShard(gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
    var ck: Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    try ck.add(dir, "model-00001-of-00002.safetensors");
}

const PackTestSelect = struct {
    width: u32,
    all: bool = false,
    fn keep(raw: *anyopaque, _: u32, width: u32, name: []const u8) bool {
        const s: *const PackTestSelect = @ptrCast(@alignCast(raw));
        return std.mem.eql(u8, name, "same") and (s.all or width == s.width);
    }
};

test "pack store filters widths and rejects kept duplicate names" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "packs/L01");
    for ([_][]const u8{ "K1.e0-1.safetensors", "K2.e0-1.safetensors" }) |file_name| {
        const image = try testImage(gpa, "same", 1);
        defer gpa.free(image);
        var path: [96]u8 = undefined;
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path, "packs/L01/{s}", .{file_name}), .data = image });
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"ignored\":\"packs/L01/K1.e0-1.safetensors\"}}" });
    var dir_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const ordinary = try shardFiles(gpa, io, dir);
    defer freeShardFiles(gpa, ordinary);
    try std.testing.expectEqual(@as(usize, 1), ordinary.len);
    try std.testing.expect(std.mem.endsWith(u8, ordinary[0], "packs/L01/K1.e0-1.safetensors"));
    var wanted: PackTestSelect = .{ .width = 2 };
    var ck = try Checkpoint.openModelPacks(gpa, io, dir, .{ .ctx = &wanted, .keep = PackTestSelect.keep });
    try std.testing.expectEqual(@as(usize, 1), ck.files.items.len);
    try std.testing.expect(ck.has("same"));
    ck.close();
    wanted.all = true;
    try std.testing.expectError(error.DuplicateTensor, Checkpoint.openModelPacks(gpa, io, dir, .{ .ctx = &wanted, .keep = PackTestSelect.keep }));
    wanted.all = false;
    wanted.width = 3;
    ck = try Checkpoint.openModelPacks(gpa, io, dir, .{ .ctx = &wanted, .keep = PackTestSelect.keep });
    try std.testing.expectEqual(@as(usize, 0), ck.files.items.len);
    ck.close();
}

test "a failed allocation while a checkpoint opens leaves nothing behind" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var walk = std.testing.tmpDir(.{});
    defer walk.cleanup();
    var indexed = std.testing.tmpDir(.{});
    defer indexed.cleanup();
    const shards = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    for (shards, 1..) |name, i| {
        const image = try testImage(gpa, name[0..11], i * 4);
        defer gpa.free(image);
        try walk.dir.writeFile(io, .{ .sub_path = name, .data = image });
    }
    try walk.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "" });
    try walk.dir.createDirPath(io, "model-dir.safetensors");
    try indexed.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}
    });
    var paths: [2][64]u8 = undefined;
    const walk_dir = try std.fmt.bufPrint(&paths[0], ".zig-cache/tmp/{s}", .{walk.sub_path});
    const indexed_dir = try std.fmt.bufPrint(&paths[1], ".zig-cache/tmp/{s}", .{indexed.sub_path});

    for ([_][]const u8{ walk_dir, indexed_dir }) |dir| {
        const listed = try shardFiles(gpa, io, dir);
        defer freeShardFiles(gpa, listed);
        try std.testing.expectEqual(shards.len, listed.len);
        for (shards, listed) |name, path| try std.testing.expectEqualStrings(name, std.fs.path.basename(path));
    }
    try expectBalanced(listShards, io, indexed_dir);
    try expectBalanced(listShards, io, walk_dir);
    try expectBalanced(openShards, io, walk_dir);
    try expectBalanced(addShard, io, walk_dir);
}

test "the loader reads the folder's own shards: hidden folders (a download's .cache, --slide's snapshot) pass by" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const shards = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    const index = "{\"weight_map\":{\"a\":\"" ++ shards[0] ++ "\",\"b\":\"" ++ shards[1] ++ "\"}}";
    for ([_][]const u8{ "", ".tensorfold-original/", ".cache/huggingface/" }) |at| {
        if (at.len > 0) try tmp.dir.createDirPath(io, at);
        for (shards, [_][]const u8{ "a", "b" }) |name, tensor| {
            const image = try testImage(gpa, tensor, 4);
            defer gpa.free(image);
            var sub: [96]u8 = undefined;
            const sub_path = try std.fmt.bufPrint(&sub, "{s}{s}", .{ at, name });
            try tmp.dir.writeFile(io, .{ .sub_path = sub_path, .data = image });
        }
        var sub: [96]u8 = undefined;
        const sub_path = try std.fmt.bufPrint(&sub, "{s}model.safetensors.index.json", .{at});
        try tmp.dir.writeFile(io, .{ .sub_path = sub_path, .data = index });
    }
    var buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    for (0..2) |pass| {
        if (pass == 1) try tmp.dir.deleteFile(io, "model.safetensors.index.json"); // no index: model*.safetensors
        const listed = try shardFiles(gpa, io, dir);
        defer freeShardFiles(gpa, listed);
        try std.testing.expectEqual(shards.len, listed.len);
        for (shards, listed) |name, p| try std.testing.expectEqualStrings(name, p[dir.len + 1 ..]);
        var ck = try Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        for (ck.files.items) |f| try std.testing.expect(std.mem.indexOf(u8, f.path[dir.len..], "/.") == null);
        _ = try ck.get("a");
        _ = try ck.get("b");
        try std.testing.expectEqual(@as(usize, 0), ck.unused());
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const linked = try @import("shard_edit.zig").linked(arena.allocator(), io, dir);
    try std.testing.expect(linked == null); // --slide's startup check
}

test "has finds a name without taking it; skip marks a prefix used" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "mtp.layers.0.w", "backbone.x" }, 0..) |name, i| {
        const image = try testImage(gpa, name, 4);
        defer gpa.free(image);
        var file: [32]u8 = undefined;
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "model-{d}.safetensors", .{i}), .data = image });
    }
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    try std.testing.expect(ck.has("backbone.x") and !ck.has("backbone.y"));
    try ck.skip("mtp.");
    _ = try ck.get("backbone.x");
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "after release the mapping stays readable: a later read faults the same bytes back in" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try testImage(gpa, "x", 4096);
    defer gpa.free(image);
    for (image[image.len - 4096 ..], 0..) |*b, i| b.* = @truncate(i * 7 + 1);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const t = try ck.get("x");
    try std.testing.expectEqualSlices(u8, image[image.len - 4096 ..], t.bytes);
    ck.release();
    try std.testing.expectEqualSlices(u8, image[image.len - 4096 ..], t.bytes);
}

test "release keeps a map that fell back to reads (no mmap): its heap copy is the only one" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{ .disable_memory_mapping = true });
    defer threaded.deinit();
    const io = threaded.io();
    const n = 1 << 20; // past the allocator's small buckets, so the copy has pages of its own
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try testImage(gpa, "x", n);
    defer gpa.free(image);
    for (image[image.len - n ..], 0..) |*b, i| b.* = @truncate(i * 7 + 1);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    try std.testing.expect(ck.files.items[0].map.section == null);
    const t = try ck.get("x");
    ck.release();
    try std.testing.expectEqualSlices(u8, image[image.len - n ..], t.bytes);
}

test "Save recovery: an unresolved partial save refuses actual checkpoint reopening" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try testImage(a, "probe", 32);
    defer a.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    try tmp.dir.writeFile(io, .{ .sub_path = ".tensorfold-save-pending", .data = "unfinished" });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    if (Checkpoint.openModel(a, io, dir)) |opened| {
        var ck = opened;
        ck.close();
        return error.UncertainCheckpointReopened;
    } else |err| try std.testing.expectEqual(error.CheckpointSaveUncertain, err);
}
