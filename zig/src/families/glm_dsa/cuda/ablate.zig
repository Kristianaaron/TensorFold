// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Runtime GLM-5.3 residual direction ablation vectors.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const width = 6144;
pub const Site = enum(u1) { attn, ffn };
pub const Kind = enum(u32) { f32_parts = 0, bf16_parts = 1 };
pub const Vec = struct { r: u64, a: u64 };

pub const Ablation = struct {
    layers: usize,
    buf: cuda.DeviceBuffer,
    present: [78][2]bool,
    digest: [32]u8,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, layers: usize) !Ablation {
        if (layers > 78) return error.AblationBadTensor;
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        const rs = try gpa.alloc([width]f32, layers * 2);
        defer gpa.free(rs);
        const as = try gpa.alloc(f32, layers * 2);
        defer gpa.free(as);
        @memset(rs, @splat(0)); @memset(as, 0);
        var have_r: [78][2]bool = undefined;
        var have_a: [78][2]bool = undefined;
        @memset(&have_r, .{ false, false });
        @memset(&have_a, .{ false, false });
        var it = file.names.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            if (!std.mem.startsWith(u8, name, "ablate.L")) return error.AblationUnknownTensor;
            const rest = name[8..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.AblationUnknownTensor;
            const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return error.AblationUnknownTensor;
            if (layer >= layers or layer >= 78) return error.AblationUnknownTensor;
            const tail = rest[dot + 1 ..];
            const site_dot = std.mem.indexOfScalar(u8, tail, '.') orelse return error.AblationUnknownTensor;
            const si: usize = if (std.mem.eql(u8, tail[0..site_dot], "attn")) 0 else if (std.mem.eql(u8, tail[0..site_dot], "ffn")) 1 else return error.AblationUnknownTensor;
            const suffix = tail[site_dot + 1 ..];
            const t = file.get(name).?;
            if (std.mem.eql(u8, suffix, "r")) {
                if (have_r[layer][si]) return error.AblationUnknownTensor;
                if (t.rank != 1 or t.shape[0] != width or (t.dtype != .f32 and t.dtype != .bf16)) return error.AblationBadTensor;
                for (0..width) |i| {
                    const v: f32 = if (t.dtype == .f32) @bitCast(std.mem.readInt(u32, t.bytes[i*4..][0..4], .little)) else @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[i*2..][0..2], .little)) << 16));
                    if (!std.math.isFinite(v)) return error.AblationNotFinite;
                    rs[layer * 2 + si][i] = v;
                }
                have_r[layer][si] = true;
            } else if (std.mem.eql(u8, suffix, "a")) {
                if (have_a[layer][si]) return error.AblationUnknownTensor;
                if (t.dtype != .f32 or t.rank != 1 or t.shape[0] != 1) return error.AblationBadTensor;
                as[layer * 2 + si] = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
                if (!std.math.isFinite(as[layer * 2 + si])) return error.AblationNotFinite;
                have_a[layer][si] = true;
            } else return error.AblationUnknownTensor;
        }
        var present: [78][2]bool = undefined;
        @memset(&present, .{ false, false });
        for (0..layers) |l| for (0..2) |s| {
            if (have_r[l][s] != have_a[l][s]) return error.AblationIncomplete;
            present[l][s] = have_r[l][s];
        };
        const rbytes = layers * 2 * width * 4;
        const abytes = layers * 2 * 4;
        const align_a = std.mem.alignForward(usize, rbytes, 256);
        var buf = try cuda.DeviceBuffer.alloc(d, align_a + abytes);
        errdefer buf.free();
        // Host integration tests install a CUDA memory hook backed by ordinary host
        // allocations. Keep those uploads on the host as well; real device buffers
        // still take the normal staged CUDA copy path.
        if (buf.owner != null) {
            @memcpy(@as([*]u8, @ptrFromInt(buf.ptr))[0..rbytes], std.mem.sliceAsBytes(rs));
            @memcpy(@as([*]u8, @ptrFromInt(buf.ptr + align_a))[0..abytes], std.mem.sliceAsBytes(as));
        } else {
            try buf.upload(0, std.mem.sliceAsBytes(rs));
            try buf.upload(align_a, std.mem.sliceAsBytes(as));
        }
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(file.map.memory, &digest, .{});
        return .{ .layers = layers, .buf = buf, .present = present, .digest = digest };
    }

    pub fn deinit(a: *Ablation) void { a.buf.free(); }
    pub fn vec(a: *const Ablation, layer: usize, site: Site) ?Vec {
        const s = @intFromEnum(site);
        if (layer >= a.layers or !a.present[layer][s]) return null;
        const align_a = std.mem.alignForward(usize, a.layers * 2 * width * 4, 256);
        return .{ .r = a.buf.ptr + (layer * 2 + s) * width * 4, .a = a.buf.ptr + align_a + (layer * 2 + s) * 4 };
    }
    pub fn launch(_: *const Ablation, k: cuda.Function, stream: cuda.Stream, x: u64, parts: u64, kind: Kind, v: Vec, rows: usize, norm_w: u64, normed: u64, eps: f32) !void {
        if (rows == 0) return;
        var args: cuda.Args = .{};
        inline for (.{ x, parts, @as(c_int, @intCast(@intFromEnum(kind))), v.r, v.a, norm_w, normed, @as(c_int, @intCast(rows)), eps }) |varg| args.add(varg);
        try cuda.launch.launch(k, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

fn bf16(v: f32) u16 { const bits: u32 = @bitCast(v); return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16); }
fn fromBf16(v: u16) f32 { return @bitCast(@as(u32, v) << 16); }
pub fn ablateRef(x: []u16, parts: []const u8, kind: Kind, r: []const f32, a: f32, rows: usize) void {
    for (0..rows) |row| {
        var acc: [256]f32 = @splat(0);
        for (0..256) |t| for (0..24) |u| {
            const j = t + 256 * u;
            const ix = row * width + j;
            const d: f32 = if (kind == .f32_parts) blk: {
                const p0: f32 = @bitCast(std.mem.readInt(u32, parts[ix*4..][0..4], .little));
                const p1: f32 = @bitCast(std.mem.readInt(u32, parts[(rows*width+ix)*4..][0..4], .little));
                break :blk fromBf16(bf16(p0 + p1));
            } else blk: {
                const p0 = std.mem.readInt(u16, parts[ix*2..][0..2], .little);
                const p1 = std.mem.readInt(u16, parts[(rows*width+ix)*2..][0..2], .little);
                break :blk fromBf16(bf16(fromBf16(p0) + fromBf16(p1)));
            };
            acc[t] = @mulAdd(f32, r[j], d, acc[t]);
        };
        var warp: [8]f32 = undefined;
        for (0..8) |w| {
            var lanes: [32]f32 = undefined;
            for (0..32) |lane| lanes[lane] = acc[w*32+lane];
            var offset: usize = 16;
            while (offset > 0) : (offset /= 2) { const old = lanes; for (0..32) |lane| lanes[lane] = old[lane] + old[lane ^ offset]; }
            warp[w] = lanes[0];
        }
        var dot = warp[0]; for (warp[1..]) |v| dot += v;
        const scale = a * dot;
        for (0..width) |j| {
            const ix = row * width + j;
            const value = fromBf16(x[ix]);
            x[ix] = bf16(value - scale * r[j]);
        }
    }
}

test "ablateRef zero scale preserves residual and one hot removes rounded delta" {
    const x = try std.testing.allocator.alloc(u16, width);
    defer std.testing.allocator.free(x);
    @memset(x, bf16(1.25));
    const original = try std.testing.allocator.dupe(u16, x);
    defer std.testing.allocator.free(original);
    const parts = try std.testing.allocator.alloc(u8, width * 2 * 4);
    defer std.testing.allocator.free(parts);
    @memset(parts, 0);
    const index = 17;
    std.mem.writeInt(u32, parts[index * 4 ..][0..4], @bitCast(@as(f32, 0.375)), .little);
    std.mem.writeInt(u32, parts[(width + index) * 4 ..][0..4], @bitCast(@as(f32, 0.125)), .little);
    var r: [width]f32 = @splat(0);
    r[index] = 1;
    ablateRef(x, parts, .f32_parts, &r, 0, 1);
    try std.testing.expectEqualSlices(u16, original, x);
    ablateRef(x, parts, .f32_parts, &r, 1, 1);
    try std.testing.expectEqual(bf16(fromBf16(original[index]) - fromBf16(bf16(0.5))), x[index]);
}

test "ablateRef uses fixed thread and butterfly reduction order" {
    const x = try std.testing.allocator.alloc(u16, width);
    defer std.testing.allocator.free(x);
    @memset(x, 0);
    const parts = try std.testing.allocator.alloc(u8, width * 2 * 4);
    defer std.testing.allocator.free(parts);
    @memset(parts, 0);
    const r = try std.testing.allocator.alloc(f32, width);
    defer std.testing.allocator.free(r);
    @memset(r, 0);
    // Contributions land in separate warps and make sequential host accumulation differ.
    for ([_]usize{ 0, 32, 64, 96, 128, 160, 192, 224 }) |j| {
        const v: f32 = if (j == 32) 1.0e20 else if (j == 64) -1.0e20 else 3.0;
        r[j] = 1;
        std.mem.writeInt(u32, parts[j * 4 ..][0..4], @bitCast(v), .little);
    }
    ablateRef(x, parts, .f32_parts, r, 1, 1);
    // Warp order gives (((3 + 1e20) - 1e20) + 3 ...), which rounds to 15;
    // a regrouped reduction can retain a different small-term total.
    try std.testing.expectEqual(bf16(-15), x[0]);
}

test "safetensors loader accepts valid vectors and rejects malformed layer tensors" {
    const shard_edit = core.shard_edit;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "vectors.safetensors" });
    defer a.free(path);
    const rb = try a.alloc(f32, width);
    defer a.free(rb);
    @memset(rb, 0);
    rb[3] = 1;
    const scale = [_]f32{0.25};
    // Match the other host integration tests: DeviceBuffer allocation is backed by host
    // memory, so the loader can be exercised without a CUDA context or copy calls.
    const TestDriver = struct {
        var blocks: [4][]align(256) u8 = undefined;
        var count: usize = 0;
        fn alloc(_: *anyopaque, n: usize) ?cuda.abi.DevicePtr {
            if (count == blocks.len) return null;
            const b = a.alignedAlloc(u8, .fromByteUnits(256), n) catch return null;
            blocks[count] = b;
            count += 1;
            return @intFromPtr(b.ptr);
        }
        fn free(_: *anyopaque, p: cuda.abi.DevicePtr) bool {
            for (blocks[0..count]) |*b| if (b.len != 0 and @intFromPtr(b.ptr) == p) {
                a.free(b.*);
                b.* = b.*[0..0];
                return true;
            };
            return false;
        }
    };
    TestDriver.count = 0;
    cuda.memory.setHook(.{ .ctx = @ptrCast(&TestDriver.blocks), .alloc = TestDriver.alloc, .free = TestDriver.free });
    defer cuda.memory.setHook(null);
    var driver: @import("cuda").Driver = undefined;

    try shard_edit.create(a, path, &.{}, &.{
        .{ .name = "ablate.L0.attn.r", .dtype = .f32, .shape = &.{width}, .bytes = std.mem.sliceAsBytes(rb) },
        .{ .name = "ablate.L0.attn.a", .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&scale) },
    });
    var loaded = try Ablation.load(a, io, &driver, path, 1);
    defer loaded.deinit();
    try std.testing.expect(loaded.buf.owner != null);
    try std.testing.expect(loaded.present[0][0]);
    try std.testing.expect(!loaded.present[0][1]);
    try std.testing.expect(loaded.vec(0, .attn) != null);
    try std.testing.expect(loaded.vec(0, .ffn) == null);
    const zero_digest: [32]u8 = @splat(0);
    try std.testing.expect(!std.mem.eql(u8, &loaded.digest, &zero_digest));

    try shard_edit.create(a, path, &.{}, &.{
        .{ .name = "ablate.L0.attn.r", .dtype = .f32, .shape = &.{width - 1}, .bytes = std.mem.sliceAsBytes(rb[0 .. width - 1]) },
        .{ .name = "ablate.L0.attn.a", .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&scale) },
    });
    try std.testing.expectError(error.AblationBadTensor, Ablation.load(a, io, &driver, path, 1));

    try shard_edit.create(a, path, &.{}, &.{
        .{ .name = "ablate.L1.attn.r", .dtype = .f32, .shape = &.{width}, .bytes = std.mem.sliceAsBytes(rb) },
        .{ .name = "ablate.L1.attn.a", .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&scale) },
    });
    try std.testing.expectError(error.AblationUnknownTensor, Ablation.load(a, io, &driver, path, 1));

    try shard_edit.create(a, path, &.{}, &.{
        .{ .name = "ablate.L0.attn.r", .dtype = .f32, .shape = &.{width}, .bytes = std.mem.sliceAsBytes(rb) },
    });
    try std.testing.expectError(error.AblationIncomplete, Ablation.load(a, io, &driver, path, 1));

    const half = try a.alloc(u16, width);
    defer a.free(half);
    @memset(half, 0);
    try shard_edit.create(a, path, &.{}, &.{
        .{ .name = "ablate.L0.attn.r", .dtype = .f16, .shape = &.{width}, .bytes = std.mem.sliceAsBytes(half) },
        .{ .name = "ablate.L0.attn.a", .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&scale) },
    });
    try std.testing.expectError(error.AblationBadTensor, Ablation.load(a, io, &driver, path, 1));
}
