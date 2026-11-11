// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@2bcc2e2db8bd311d3f0a72e0bd5bab614de34d67 (Apache-2.0), zig/src/families/glm/cuda_l2pf.zig by Bizuayeu (PR #1580 / #2162).
//! TF_GLM53_L2PF: a decode window's next weights read into L2 on a side stream (DeepSeek V4.1's l2pf and l2pace kernels).

const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");

pub const Options = struct {
    on: bool = false,
    mb: u32 = 12,
    ctas: u32 = 2,
    pace_gbps: u32 = 150,

    pub fn want(o: Options) bool {
        return o.on;
    }

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Options {
        const raw = get("TF_GLM53_L2PF") orelse return .{};
        var o: Options = .{};
        if (std.mem.eql(u8, raw, "0")) return o;
        if (!std.mem.eql(u8, raw, "1")) {
            problem.* = "TF_GLM53_L2PF: 0 or 1";
            return error.BadSetting;
        }
        o.on = true;
        o.mb = try readInt(get, problem, "TF_GLM53_L2PF_MB", 1, std.math.maxInt(u32), o.mb);
        o.ctas = try readInt(get, problem, "TF_GLM53_L2PF_CTAS", 1, std.math.maxInt(u32), o.ctas);
        o.pace_gbps = try readInt(get, problem, "TF_GLM53_L2PF_PACE_GBPS", 0, std.math.maxInt(u32), o.pace_gbps);
        return o;
    }

    fn readInt(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8, name: [:0]const u8, min: u32, max: u32, default: u32) !u32 {
        const raw = get(name) orelse return default;
        const value = std.fmt.parseInt(u32, raw, 10) catch {
            problem.* = if (std.mem.eql(u8, name, "TF_GLM53_L2PF_MB")) "TF_GLM53_L2PF_MB: integer >= 1" else if (std.mem.eql(u8, name, "TF_GLM53_L2PF_CTAS")) "TF_GLM53_L2PF_CTAS: integer >= 1" else "TF_GLM53_L2PF_PACE_GBPS: integer >= 0";
            return error.BadSetting;
        };
        if (value < min or value > max) {
            problem.* = if (std.mem.eql(u8, name, "TF_GLM53_L2PF_MB")) "TF_GLM53_L2PF_MB: integer >= 1" else if (std.mem.eql(u8, name, "TF_GLM53_L2PF_CTAS")) "TF_GLM53_L2PF_CTAS: integer >= 1" else "TF_GLM53_L2PF_PACE_GBPS: integer >= 0";
            return error.BadSetting;
        }
        return value;
    }
};

pub const Site = enum(u1) { n, f };
pub const Span = struct { at: u32 = 0, rows: u32 = 0, bytes: u64 = 0 };
pub const Region = struct { ptr: u64, len: u64 };

fn locate(regions: []const Region, p: u64) ?Region {
    for (regions) |r| if (r.ptr <= p and p < r.ptr + r.len) return .{ .ptr = p, .len = r.ptr + r.len - p };
    return null;
}

pub fn take(sizes: []const u64, budget: u64, out: []u64) void {
    @memset(out, 0);
    var left = budget / 16 * 16;
    for (sizes, out) |size, *bytes| {
        if (left < 16) break;
        bytes.* = @min(left, size) / 16 * 16;
        left -= bytes.*;
    }
}

const Sources = struct {
    ptr: [8]u64 = @splat(0),
    bytes: [8]u64 = @splat(0),
    len: usize = 0,

    fn add(s: *Sources, ptr: u64, bytes: u64) void {
        if (ptr == 0 or s.len == s.ptr.len) return;
        s.ptr[s.len] = ptr;
        s.bytes[s.len] = bytes;
        s.len += 1;
    }

    fn addWeight(s: *Sources, w: cuda.qmmf.Weight) void {
        if (w.scales != 0) s.add(w.scales, @as(u64, w.npad) * @as(u64, w.k) / cuda.qmmf.group * 4);
        if (w.codes != 0) s.add(w.codes, @as(u64, w.npad) * @as(u64, w.k));
    }
};

fn sources(layers: []const weights.DeviceLayer, j: usize, site: Site) Sources {
    var s: Sources = .{};
    if (j >= layers.len) return s;
    switch (site) {
        .f => switch (layers[j].ffn) {
            .moe => |m| s.add(m.router, std.math.maxInt(u64)),
            .dense => |m| s.addWeight(m.gate_up),
        },
        .n => {
            if (j + 1 >= layers.len) return s;
            const a = layers[j + 1].attn;
            s.addWeight(a.q_a);
            s.addWeight(a.kv_a);
            if (a.indexer) |ix| s.addWeight(ix.wk);
            s.addWeight(a.q_b);
        },
    }
    return s;
}

pub fn table(gpa: std.mem.Allocator, regions: []const Region, layers: []const weights.DeviceLayer, budget: u64, rows: *std.ArrayList([2]i64), spans: []Span) !void {
    @memset(spans, .{});
    for (layers, 0..) |_, j| inline for (.{ Site.n, Site.f }) |site| {
        const ix = j * 2 + @intFromEnum(site);
        const src = sources(layers, j, site);
        var sizes: [8]u64 = @splat(0);
        var found: [8]Region = undefined;
        var starts: [8]u64 = @splat(0);
        for (0..src.len) |i| {
            found[i] = locate(regions, src.ptr[i]) orelse {
                // warn, not err: the refusal is the error return (a logged err fails the zig test of this path).
                std.log.warn("TF_GLM53_L2PF: layer {d} site {t} reads 0x{x}, which no weight tensor holds", .{ j, site, src.ptr[i] });
                return error.L2pfSourceNotATensor;
            };
            starts[i] = std.mem.alignForward(u64, found[i].ptr, 16);
            const skipped = starts[i] - found[i].ptr;
            sizes[i] = @min(src.bytes[i], found[i].len -| skipped);
        }
        var chosen: [8]u64 = @splat(0);
        take(sizes[0..src.len], budget, chosen[0..src.len]);
        spans[ix].at = @intCast(rows.items.len);
        for (0..src.len) |i| {
            if (chosen[i] == 0) continue;
            try rows.append(gpa, .{ @intCast(starts[i]), @intCast(chosen[i]) });
            spans[ix].rows += 1;
            spans[ix].bytes += chosen[i];
        }
    };
    if (layers.len != 0 and spans[@intFromEnum(Site.f)].rows == 0) return error.L2pfNoFirstSite;
}

pub fn regionsOf(gpa: std.mem.Allocator, m: *const weights.ModelWeights) ![]Region {
    var out: std.ArrayList(Region) = .empty;
    errdefer out.deinit(gpa);
    for (m.tensors.items) |t| {
        try out.append(gpa, .{ .ptr = t.data.ptr, .len = @intCast(t.data.len) });
        if (t.scales) |scale| try out.append(gpa, .{ .ptr = scale.ptr, .len = @intCast(scale.len) });
    }
    return out.toOwnedSlice(gpa);
}

pub const L2pf = struct {
    gpa: std.mem.Allocator,
    o: Options,
    side: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    mods: [2]cuda.Module = undefined,
    mods_loaded: usize = 0,
    segments: cuda.Function,
    paced: cuda.Function,
    rows: cuda.DeviceBuffer,
    spans: []Span,
    pieces: []u64,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, regions: []const Region, layers: []const weights.DeviceLayer, o: Options) !L2pf {
        var host_rows: std.ArrayList([2]i64) = .empty;
        defer host_rows.deinit(gpa);
        const spans = try gpa.alloc(Span, layers.len * 2);
        errdefer gpa.free(spans);
        try table(gpa, regions, layers, @as(u64, o.mb) << 20, &host_rows, spans);
        if (host_rows.items.len == 0) return error.L2pfNoRows;
        const pieces = try gpa.alloc(u64, spans.len);
        errdefer gpa.free(pieces);
        @memset(pieces, 0);
        for (spans, pieces) |span, *count| {
            const first: usize = span.at;
            const end = first + @as(usize, span.rows);
            for (host_rows.items[first..end]) |row| {
                const bytes: u64 = @intCast(row[1]);
                count.* += (bytes + 32767) / 32768;
            }
        }
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork_ev = try cuda.Event.init(d, false);
        errdefer fork_ev.deinit();
        var join_ev = try cuda.Event.init(d, false);
        errdefer join_ev.deinit();
        var l2pf_mod = try cuda.Module.load(d, cuda.kernels.l2pf);
        errdefer l2pf_mod.unload();
        var pace_mod = try cuda.Module.load(d, cuda.kernels.l2pace);
        errdefer pace_mod.unload();
        const segments = try l2pf_mod.function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij");
        const paced = try pace_mod.function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        var device_rows = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host_rows.items));
        errdefer device_rows.free();
        std.log.info("TF_GLM53_L2PF=1: {d} rows, {d} MiB a decode window, {d} CTAs a site, {d} GB/s (0: unpaced); launches _ZN12dsv41_l2pace12paced_kernelEPKlijmm or _ZN6tfl2pf15segments_kernelILb1EEEvPKlij on its own stream", .{ host_rows.items.len, o.mb, o.ctas, o.pace_gbps });
        return .{ .gpa = gpa, .o = o, .side = side, .fork_ev = fork_ev, .join_ev = join_ev, .mods = .{ l2pf_mod, pace_mod }, .mods_loaded = 2, .segments = segments, .paced = paced, .rows = device_rows, .spans = spans, .pieces = pieces };
    }

    pub fn deinit(p: *L2pf) void {
        p.side.synchronize() catch {};
        p.rows.free();
        for (p.mods[0..p.mods_loaded]) |*m| m.unload();
        p.join_ev.deinit();
        p.fork_ev.deinit();
        p.side.deinit();
        p.gpa.free(p.spans);
        p.gpa.free(p.pieces);
    }

    pub fn fork(p: *const L2pf, main: cuda.Stream, j: usize, site: Site) !void {
        const ix = j * 2 + @intFromEnum(site);
        if (ix >= p.spans.len or p.spans[ix].rows == 0) return;
        try p.fork_ev.record(main);
        try p.side.wait(p.fork_ev);
        const span = p.spans[ix];
        const pieces = p.pieces[ix];
        var args: cuda.Args = .{};
        args.add(p.rows.ptr + @as(u64, span.at) * 16);
        args.add(@as(c_int, @intCast(span.rows)));
        args.add(@as(u32, 32768));
        if (p.o.pace_gbps > 0) {
            args.add(@as(u64, 32768) / @as(u64, p.o.pace_gbps));
            args.add(@as(u64, 0));
            try cuda.launch.launch(p.paced, .{ .grid = .{ .x = @intCast(p.o.ctas) }, .block = .{ .x = 32 } }, p.side, &args);
        } else {
            const grid = @min(p.o.ctas, @max(@as(u32, 1), @as(u32, @intCast((pieces + 127) / 128))));
            try cuda.launch.launch(p.segments, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 256 } }, p.side, &args);
        }
    }

    pub fn join(p: *const L2pf, main: cuda.Stream) !void {
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
    }
};

test "l2pf.take spends aligned budgets in source order" {
    const sizes = [_]u64{ 48, 96, 4096 };
    var out: [3]u64 = undefined;
    take(&sizes, 128, &out);
    try std.testing.expectEqualSlices(u64, &.{ 48, 80, 0 }, &out);
}

var option_values: struct { on: ?[]const u8 = null, mb: ?[]const u8 = null, ctas: ?[]const u8 = null, pace: ?[]const u8 = null } = .{};

fn optionGet(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM53_L2PF")) return option_values.on;
    if (std.mem.eql(u8, name, "TF_GLM53_L2PF_MB")) return option_values.mb;
    if (std.mem.eql(u8, name, "TF_GLM53_L2PF_CTAS")) return option_values.ctas;
    if (std.mem.eql(u8, name, "TF_GLM53_L2PF_PACE_GBPS")) return option_values.pace;
    return null;
}

test "l2pf options are default off, parse knobs only when on, and name bad settings" {
    option_values = .{};
    var problem: []const u8 = "";
    try std.testing.expectEqual(Options{}, try Options.read(optionGet, &problem));
    option_values.on = "0";
    option_values.mb = "bad";
    option_values.ctas = "bad";
    option_values.pace = "bad";
    try std.testing.expectEqual(Options{}, try Options.read(optionGet, &problem));
    option_values.on = "1";
    option_values.mb = null;
    option_values.ctas = null;
    option_values.pace = null;
    try std.testing.expectEqual(Options{ .on = true }, try Options.read(optionGet, &problem));
    option_values.mb = "4";
    option_values.ctas = "3";
    option_values.pace = "0";
    try std.testing.expectEqual(Options{ .on = true, .mb = 4, .ctas = 3, .pace_gbps = 0 }, try Options.read(optionGet, &problem));
    for ([_][]const u8{ "true", "2", "bad" }) |bad| {
        option_values.on = bad;
        try std.testing.expectError(error.BadSetting, Options.read(optionGet, &problem));
        try std.testing.expectEqualStrings("TF_GLM53_L2PF: 0 or 1", problem);
    }
    option_values.on = "1";
    for ([_]struct { field: u8, value: []const u8, message: []const u8 }{
        .{ .field = 0, .value = "0", .message = "TF_GLM53_L2PF_MB: integer >= 1" },
        .{ .field = 0, .value = "bad", .message = "TF_GLM53_L2PF_MB: integer >= 1" },
        .{ .field = 1, .value = "0", .message = "TF_GLM53_L2PF_CTAS: integer >= 1" },
        .{ .field = 1, .value = "bad", .message = "TF_GLM53_L2PF_CTAS: integer >= 1" },
        .{ .field = 2, .value = "-1", .message = "TF_GLM53_L2PF_PACE_GBPS: integer >= 0" },
        .{ .field = 2, .value = "bad", .message = "TF_GLM53_L2PF_PACE_GBPS: integer >= 0" },
    }) |bad| {
        option_values.mb = "4";
        option_values.ctas = "3";
        option_values.pace = "0";
        switch (bad.field) {
            0 => option_values.mb = bad.value,
            1 => option_values.ctas = bad.value,
            else => option_values.pace = bad.value,
        }
        try std.testing.expectError(error.BadSetting, Options.read(optionGet, &problem));
        try std.testing.expectEqualStrings(bad.message, problem);
    }
    option_values = .{};
}

fn testWeight(codes: u64, scales: u64) cuda.qmmf.Weight {
    return .{ .mode = .fp8b, .codes = codes, .scales = scales, .n = 128, .k = 128, .npad = 128 };
}

test "l2pf table bounds sources to tensors and each site's budget" {
    const dense: weights.DeviceLayer = .{
        .index = 0,
        .in_norm = 0,
        .post_norm = 0,
        .attn = .{ .q_a = testWeight(0x3008, 0x3000), .q_b = testWeight(0x9008, 0x9000), .kv_a = testWeight(0x5008, 0x5000), .kv_b = .{ .codes = 0, .scales = 0, .rows = 0, .cols = 0 }, .o_proj = testWeight(0xb000, 0xa000), .q_a_norm = 0, .kv_a_norm = 0, .local_heads = 1, .head0 = 0 },
        .ffn = .{ .dense = .{ .gate_up = testWeight(0x2000, 0x1000), .down = testWeight(0xe000, 0xd000) } },
    };
    const moe: weights.DeviceLayer = .{
        .index = 1,
        .in_norm = 0,
        .post_norm = 0,
        .attn = .{ .q_a = testWeight(0x11008, 0x11000), .q_b = testWeight(0x15008, 0x15000), .kv_a = testWeight(0x13008, 0x13000), .kv_b = .{ .codes = 0, .scales = 0, .rows = 0, .cols = 0 }, .o_proj = testWeight(0x17000, 0x16000), .q_a_norm = 0, .kv_a_norm = 0, .indexer = .{ .wq_b = testWeight(0x1d000, 0x1c000), .wk = testWeight(0x19008, 0x19000), .weights_proj = testWeight(0x1f000, 0x1e000), .k_norm = 0, .k_bias = 0 }, .local_heads = 1, .head0 = 0 },
        .ffn = .{ .moe = .{ .router = 0x18000, .bias = 0, .routed = undefined, .shared = .{ .gate_up = testWeight(0, 0), .down = testWeight(0, 0) } } },
    };
    const layers = [_]weights.DeviceLayer{ dense, moe };
    const regions = [_]Region{
        .{ .ptr = 0x1000, .len = 4096 }, .{ .ptr = 0x2000, .len = 4096 },
        .{ .ptr = 0x3000, .len = 4096 }, .{ .ptr = 0x5000, .len = 4096 }, .{ .ptr = 0x7000, .len = 4096 }, .{ .ptr = 0x9000, .len = 4096 },
        .{ .ptr = 0xa000, .len = 4096 }, .{ .ptr = 0xb000, .len = 4096 }, .{ .ptr = 0xc000, .len = 4096 }, .{ .ptr = 0xd000, .len = 4096 }, .{ .ptr = 0xe000, .len = 4096 }, .{ .ptr = 0xf000, .len = 4096 },
        .{ .ptr = 0x11000, .len = 4096 }, .{ .ptr = 0x13000, .len = 4096 }, .{ .ptr = 0x15000, .len = 4096 }, .{ .ptr = 0x16000, .len = 4096 }, .{ .ptr = 0x17000, .len = 4096 }, .{ .ptr = 0x18000, .len = 4096 },
        .{ .ptr = 0x19000, .len = 4096 }, .{ .ptr = 0x1c000, .len = 4096 }, .{ .ptr = 0x1d000, .len = 4096 }, .{ .ptr = 0x1e000, .len = 4096 }, .{ .ptr = 0x1f000, .len = 4096 },
    };
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(std.testing.allocator);
    var spans: [4]Span = undefined;
    try table(std.testing.allocator, &regions, &layers, 1 << 20, &rows, &spans);
    try std.testing.expectEqual(@as(u32, 8), spans[@intFromEnum(Site.f)].at);
    try std.testing.expectEqual(@as(u32, 8), spans[0].rows);
    try std.testing.expectEqual(@as(u32, 0), spans[2].rows);
    var found_interior = false;
    for (rows.items) |row| if (@as(u64, @intCast(row[0])) == 0x11010 and @as(u64, @intCast(row[1])) == 4080) { // layer 1's q_a codes, read at layer 0's site n
        found_interior = true;
    };
    try std.testing.expect(found_interior);

    rows.clearRetainingCapacity();
    spans = undefined;
    try table(std.testing.allocator, &regions, &layers, 64, &rows, &spans);
    try std.testing.expectEqual(@as(u32, 1), spans[@intFromEnum(Site.f)].rows);
    try std.testing.expectEqual(@as(i64, 64), rows.items[0][1]);
    try std.testing.expectError(error.L2pfSourceNotATensor, table(std.testing.allocator, &.{}, &layers, 64, &rows, &spans));
}
