// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 draft-depth recalibration helpers for the lane adapter (rank 0): TF_GLM53_DRAFT_RECAL (per-width round costs
//! and per-slot logistic weights for the DSpark confidence), TF_GLM53_DRAFT_SWEEP (interleaved width sweep) and
//! TF_GLM53_ROUND_TRACE (one JSON line per draft, verify and release).
//! Depth policy after MiaAI-Lab GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patches 0114/0116/0119/0126 (Apache-2.0,
//! Copyright 2026 Mia's AI Lab): the confidence survival product times measured per-width round costs picks the verified
//! prefix; costs are measured on real text and kept nondecreasing. Independent Zig; no patch code is copied.
//! The confidence head is Red Hat's DSpark (RedHatAI/GLM-5.3-speculator.dspark, GLM-5.3 license; vllm-project/speculators, Apache-2.0).
//! The weights are fitted offline by tools/glm53/recal_fit.py (Platt-style per-slot logistic).
//! These switches change only how many held drafts a round verifies, never a token: verify stays exact.
const std = @import("std");
const lanes = @import("lanes");

pub const Cost = lanes.config.Cost;
pub const schema = "tf.glm53.draft-recal/v1";
pub const trace_schema = "tf.glm53.round-trace/v1";
pub const widths: usize = 16;
pub const slots: usize = 7;
/// Per-slot logistic weights: [mode][slot][feature]; mode 0 greedy (temperature 0), 1 sampled; features z, margin, logp, 1.
pub const ConfSets = [2][slots][4]f64;

pub const Recal = struct {
    costs: ?[widths]Cost = null,
    conf: ?ConfSets = null,
    sha: [32]u8,
};

/// What the trace's first line records about the serving switches.
pub const Open = struct {
    draft_chances: bool = false,
    live_costs: bool = false,
    lane_chances: bool = false,
    copyd: bool = false,
    copy_rows: u32 = 1,
    verify_max: usize = 8,
    conf_temp: f64 = 1.0,
    dspark: bool = false,
    recal: ?[32]u8 = null,
    sweep: bool = false,
};

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => v.float,
        .integer => @floatFromInt(v.integer),
        else => error.BadDraftRecal,
    };
}

/// A recal file: {"schema": "tf.glm53.draft-recal/v1", "costs": [[1, ms], ... [16, ms]], "conf": {"features":
/// ["z","margin","logp","1"], "greedy": [7 rows of 4], "sampled": [7 rows of 4]}}; costs or conf may be left out (not
/// both); other keys are ignored. Any other shape is refused.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Recal {
    var out: Recal = .{ .sha = undefined };
    std.crypto.hash.sha2.Sha256.hash(bytes, &out.sha, .{});
    const json = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return error.BadDraftRecal;
    defer json.deinit();
    if (json.value != .object) return error.BadDraftRecal;
    const o = json.value.object;
    const s = o.get("schema") orelse return error.BadDraftRecal;
    if (s != .string or !std.mem.eql(u8, s.string, schema)) return error.BadDraftRecal;
    if (o.get("costs")) |cv| out.costs = try parseCosts(cv);
    if (o.get("conf")) |fv| out.conf = try parseConf(fv);
    if (out.costs == null and out.conf == null) return error.BadDraftRecal;
    return out;
}

/// Widths 1..16 in order; ms finite, 10 < ms < 2000, nondecreasing (Mia 0116).
fn parseCosts(v: std.json.Value) ![widths]Cost {
    if (v != .array or v.array.items.len != widths) return error.BadDraftRecal;
    var out: [widths]Cost = undefined;
    for (v.array.items, 0..) |row, i| {
        if (row != .array or row.array.items.len != 2) return error.BadDraftRecal;
        const width = try number(row.array.items[0]);
        const ms = try number(row.array.items[1]);
        if (width != @as(f64, @floatFromInt(i + 1))) return error.BadDraftRecal;
        if (!std.math.isFinite(ms) or ms <= 10.0 or ms >= 2000.0) return error.BadDraftRecal;
        if (i > 0 and ms < out[i - 1].ms) return error.BadDraftRecal;
        out[i] = .{ .width = @intCast(i + 1), .ms = ms };
    }
    return out;
}

fn parseConf(v: std.json.Value) !ConfSets {
    if (v != .object) return error.BadDraftRecal;
    const o = v.object;
    const f = o.get("features") orelse return error.BadDraftRecal;
    const names = [_][]const u8{ "z", "margin", "logp", "1" };
    if (f != .array or f.array.items.len != names.len) return error.BadDraftRecal;
    for (f.array.items, names) |item, name| {
        if (item != .string or !std.mem.eql(u8, item.string, name)) return error.BadDraftRecal;
    }
    var out: ConfSets = undefined;
    const modes = [_][]const u8{ "greedy", "sampled" };
    for (modes, 0..) |mode, m| {
        const rows = o.get(mode) orelse return error.BadDraftRecal;
        if (rows != .array or rows.array.items.len != slots) return error.BadDraftRecal;
        for (rows.array.items, 0..) |row, j| {
            if (row != .array or row.array.items.len != 4) return error.BadDraftRecal;
            for (row.array.items, 0..) |x, k| {
                const weight = try number(x);
                if (!std.math.isFinite(weight) or @abs(weight) > 100.0) return error.BadDraftRecal;
                out[m][j][k] = weight;
            }
        }
    }
    return out;
}

/// The calibrated conditional chance of one slot. Weights {1, 0, 0, 0} give sigmoid(z): the selector's own chance, bit
/// for bit (same formula, same f64 operands).
pub fn calibrate(w: [4]f64, z: f64, margin: f64, logp: f64) f64 {
    const x = w[0] * z + w[1] * margin + w[2] * logp + w[3];
    return 1.0 / (1.0 + @exp(-x));
}

/// The settings digest with a recal file and/or the sweep mixed in; unchanged when both are off.
pub fn digest(base: [32]u8, recal_sha: ?[32]u8, sweep: bool) [32]u8 {
    if (recal_sha == null and !sweep) return base;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(&base);
    if (recal_sha) |s| {
        h.update("TF_GLM53_DRAFT_RECAL");
        h.update(&s);
    }
    if (sweep) h.update("TF_GLM53_DRAFT_SWEEP=1");
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// TF_GLM53_DRAFT_SWEEP: unset, empty or "0" off; "1" on; anything else refused.
pub fn parseSweep(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftSweep;
}

/// Monotonic ns (the same clock as cluster/barrier.zig).
pub fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds);
}

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, count: usize, stream: *anyopaque) usize;
extern "c" fn fflush(stream: *anyopaque) c_int;
extern "c" fn fclose(stream: *anyopaque) c_int;

/// Whether `path` can be appended to (the load checks once; a trace that cannot be written fails the load).
pub fn writable(path: [*:0]const u8) bool {
    const file = fopen(path, "a") orelse return false;
    _ = fclose(file);
    return true;
}

/// Append one line (best effort after the load check, as TF_GLM_DSPARK_TRACE does).
pub fn append(path: [*:0]const u8, bytes: []const u8) void {
    const file = fopen(path, "a") orelse return;
    defer _ = fclose(file);
    if (fwrite(bytes.ptr, 1, bytes.len, file) != bytes.len) return;
    _ = fflush(file);
}

fn writeU32s(w: *std.Io.Writer, xs: []const u32) !void {
    try w.writeAll("[");
    for (xs, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", x });
    try w.writeAll("]");
}

fn writeF64s(w: *std.Io.Writer, xs: []const f64) !void {
    try w.writeAll("[");
    for (xs, 0..) |x, i| try w.print("{s}{d:.6}", .{ if (i == 0) "" else ",", x });
    try w.writeAll("]");
}

pub fn formatOpen(w: *std.Io.Writer, o: Open) !void {
    const hex: [64]u8 = if (o.recal) |s| std.fmt.bytesToHex(s, .lower) else @splat('0');
    try w.print("{{\"ev\":\"open\",\"schema\":\"{s}\",\"draft_chances\":{d},\"live_costs\":{d},\"lane_chances\":{d},\"copyd\":{d},\"copy_rows\":{d},\"verify_max\":{d},\"conf_temp\":{d:.6},\"dspark\":{d},\"recal\":\"{s}\",\"sweep\":{d}}}\n", .{ trace_schema, @intFromBool(o.draft_chances), @intFromBool(o.live_costs), @intFromBool(o.lane_chances), @intFromBool(o.copyd), o.copy_rows, o.verify_max, o.conf_temp, @intFromBool(o.dspark), &hex, @intFromBool(o.sweep) });
}

/// `feats` is the adapter's 32-slot buffer: [8,16) z, [16,24) margin, [24,32) logp (zeros when the drafter wrote none).
pub fn formatDraft(w: *std.Io.Writer, req: u64, pos: usize, anchor: u32, n: usize, depth: usize, temperature: f64, held: []const u32, p: []const f64, feats: []const f64) !void {
    try w.print("{{\"ev\":\"draft\",\"req\":{d},\"pos\":{d},\"anchor\":{d},\"n\":{d},\"depth\":{d},\"t\":{d:.6},\"drafts\":", .{ req, pos, anchor, n, depth, temperature });
    try writeU32s(w, held);
    try w.writeAll(",\"p\":");
    try writeF64s(w, p);
    try w.writeAll(",\"z\":");
    try writeF64s(w, feats[8..16]);
    try w.writeAll(",\"m\":");
    try writeF64s(w, feats[16..24]);
    try w.writeAll(",\"lp\":");
    try writeF64s(w, feats[24..32]);
    try w.writeAll("}\n");
}

pub fn formatVerify(w: *std.Io.Writer, req: u64, start: usize, rows: usize, held: usize, copy: usize, t_ns: u64, v_ns: u64, ids: []const u32, sampled: []const u32) !void {
    try w.print("{{\"ev\":\"verify\",\"req\":{d},\"start\":{d},\"rows\":{d},\"held\":{d},\"copy\":{d},\"t_ns\":{d},\"v_ns\":{d},\"ids\":", .{ req, start, rows, held, copy, t_ns, v_ns });
    try writeU32s(w, ids);
    try w.writeAll(",\"sampled\":");
    try writeU32s(w, sampled);
    try w.writeAll("}\n");
}

pub fn formatRelease(w: *std.Io.Writer, req: u64, pos: usize, t_ns: u64) !void {
    try w.print("{{\"ev\":\"release\",\"req\":{d},\"pos\":{d},\"t_ns\":{d}}}\n", .{ req, pos, t_ns });
}

pub fn openLine(path: [*:0]const u8, o: Open) void {
    var line: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer line.deinit();
    formatOpen(&line.writer, o) catch return;
    append(path, line.written());
}

pub fn draftLine(path: [*:0]const u8, req: u64, pos: usize, anchor: u32, n: usize, depth: usize, temperature: f64, held: []const u32, p: []const f64, feats: []const f64) void {
    var line: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer line.deinit();
    formatDraft(&line.writer, req, pos, anchor, n, depth, temperature, held, p, feats) catch return;
    append(path, line.written());
}

pub fn verifyLine(path: [*:0]const u8, req: u64, start: usize, rows: usize, held: usize, copy: usize, t_ns: u64, v_ns: u64, ids: []const u32, sampled: []const u32) void {
    var line: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer line.deinit();
    formatVerify(&line.writer, req, start, rows, held, copy, t_ns, v_ns, ids, sampled) catch return;
    append(path, line.written());
}

pub fn releaseLine(path: [*:0]const u8, req: u64, pos: usize, t_ns: u64) void {
    var line: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer line.deinit();
    formatRelease(&line.writer, req, pos, t_ns) catch return;
    append(path, line.written());
}

const costs16 = "[[1,80],[2,95],[3,110],[4,122],[5,135],[6,148],[7,161],[8,175],[9,193],[10,212],[11,230],[12,249],[13,262],[14,275],[15,289],[16,302]]";
const ident7 = "[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]";
const sampled7 = "[[0.5,0.1,0.2,-0.25],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]";
const conf_ok = "{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":" ++ ident7 ++ ",\"sampled\":" ++ sampled7 ++ "}";
const head = "{\"schema\":\"tf.glm53.draft-recal/v1\",\"note\":\"ignored\"";

test "glm53 recal: parses costs and conf, sha of the bytes" {
    const bytes = head ++ ",\"costs\":" ++ costs16 ++ ",\"conf\":" ++ conf_ok ++ "}";
    const parsed = try parse(std.testing.allocator, bytes);
    try std.testing.expectEqual(Cost{ .width = 1, .ms = 80 }, parsed.costs.?[0]);
    try std.testing.expectEqual(@as(u32, 16), parsed.costs.?[15].width);
    try std.testing.expectEqual(@as(f64, 302), parsed.costs.?[15].ms);
    try std.testing.expectEqual([_]f64{ 0.5, 0.1, 0.2, -0.25 }, parsed.conf.?[1][0]);
    try std.testing.expectEqual([_]f64{ 1, 0, 0, 0 }, parsed.conf.?[0][3]);
    var expected_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &expected_sha, .{});
    try std.testing.expectEqual(expected_sha, parsed.sha);
    const only_costs = try parse(std.testing.allocator, head ++ ",\"costs\":" ++ costs16 ++ "}");
    try std.testing.expect(only_costs.conf == null);
    const only_conf = try parse(std.testing.allocator, head ++ ",\"conf\":" ++ conf_ok ++ "}");
    try std.testing.expect(only_conf.costs == null);
}

test "glm53 recal: refuses bad files" {
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, "{\"schema\":\"tf.glm53.draft-recal/v0\",\"costs\":" ++ costs16 ++ "}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ "}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, "[1,2]"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, "not json"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"costs\":[[1,80],[2,95],[3,110],[4,122],[5,135],[6,148],[7,161],[8,175],[9,193],[10,212],[11,230],[12,249],[13,262],[14,275],[15,289]]}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"costs\":[[2,95],[1,80],[3,110],[4,122],[5,135],[6,148],[7,161],[8,175],[9,193],[10,212],[11,230],[12,249],[13,262],[14,275],[15,289],[16,302]]}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"costs\":[[1,80],[2,95],[3,110],[4,122],[5,135],[6,148],[7,161],[8,150],[9,193],[10,212],[11,230],[12,249],[13,262],[14,275],[15,289],[16,302]]}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"costs\":[[1,5],[2,95],[3,110],[4,122],[5,135],[6,148],[7,161],[8,175],[9,193],[10,212],[11,230],[12,249],[13,262],[14,275],[15,289],[16,302]]}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],\"sampled\":" ++ sampled7 ++ "}}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":[[1,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],\"sampled\":" ++ sampled7 ++ "}}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":" ++ ident7 ++ ",\"sampled\":[[101,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]}}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":" ++ ident7 ++ ",\"sampled\":[[1e400,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]}}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"margin\",\"z\",\"logp\",\"1\"],\"greedy\":" ++ ident7 ++ ",\"sampled\":" ++ sampled7 ++ "}}"));
    try std.testing.expectError(error.BadDraftRecal, parse(std.testing.allocator, head ++ ",\"conf\":{\"features\":[\"z\",\"margin\",\"logp\",\"1\"],\"greedy\":" ++ ident7 ++ "}}"));
}

test "glm53 recal: identity weights equal the selector chance bit for bit" {
    const zs = [_]f64{ -3.5, -0.25, 0.0, 0.7, 4.0 };
    for (zs) |z| try std.testing.expectEqual(1.0 / (1.0 + @exp(-z)), calibrate(.{ 1, 0, 0, 0 }, z, 2.0, -1.0));
    try std.testing.expectEqual(@as(f64, 0.5), calibrate(.{ 0, 0, 0, 0 }, 3.0, 1.0, -1.0));
}

test "glm53 recal: digest unchanged off, changed by recal or sweep" {
    var base: [32]u8 = @splat(7);
    try std.testing.expectEqual(base, digest(base, null, false));
    const s: [32]u8 = @splat(9);
    const recal_only = digest(base, s, false);
    const sweep_only = digest(base, null, true);
    const both = digest(base, s, true);
    try std.testing.expect(!std.mem.eql(u8, &base, &recal_only));
    try std.testing.expect(!std.mem.eql(u8, &base, &sweep_only));
    try std.testing.expect(!std.mem.eql(u8, &base, &both));
    try std.testing.expect(!std.mem.eql(u8, &recal_only, &sweep_only));
    try std.testing.expect(!std.mem.eql(u8, &recal_only, &both));
    try std.testing.expect(!std.mem.eql(u8, &sweep_only, &both));
    try std.testing.expectEqual(recal_only, digest(base, s, false));
}

test "glm53 recal: sweep switch" {
    try std.testing.expect(!(try parseSweep(null)));
    try std.testing.expect(!(try parseSweep("")));
    try std.testing.expect(!(try parseSweep("0")));
    try std.testing.expect(try parseSweep("1"));
    try std.testing.expectError(error.BadDraftSweep, parseSweep("true"));
    try std.testing.expectError(error.BadDraftSweep, parseSweep("2"));
}

test "glm53 recal: trace lines" {
    {
        var line: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer line.deinit();
        try formatVerify(&line.writer, 3, 812, 4, 3, 0, 1000, 250, &.{ 9, 100, 101, 102 }, &.{ 100, 101, 7, 8 });
        try std.testing.expectEqualStrings("{\"ev\":\"verify\",\"req\":3,\"start\":812,\"rows\":4,\"held\":3,\"copy\":0,\"t_ns\":1000,\"v_ns\":250,\"ids\":[9,100,101,102],\"sampled\":[100,101,7,8]}\n", line.written());
    }
    {
        var line: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer line.deinit();
        try formatRelease(&line.writer, 3, 2049, 77);
        try std.testing.expectEqualStrings("{\"ev\":\"release\",\"req\":3,\"pos\":2049,\"t_ns\":77}\n", line.written());
    }
    {
        var feats: [32]f64 = @splat(0);
        @memset(feats[8..16], 1.5);
        @memset(feats[16..24], 0.25);
        @memset(feats[24..32], -0.5);
        const p = [_]f64{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
        const held = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
        var line: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer line.deinit();
        try formatDraft(&line.writer, 0, 17, 42, 0, 7, 1.0, &held, &p, &feats);
        const out = line.written();
        try std.testing.expect(std.mem.startsWith(u8, out, "{\"ev\":\"draft\",\"req\":0,\"pos\":17,\"anchor\":42,\"n\":0,\"depth\":7,\"t\":1.000000,\"drafts\":[1,2,3,4,5,6,7],\"p\":[0.500000,"));
        try std.testing.expect(std.mem.indexOf(u8, out, ",\"z\":[1.500000,") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, ",\"m\":[0.250000,") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, ",\"lp\":[-0.500000,") != null);
        try std.testing.expect(std.mem.endsWith(u8, out, "-0.500000]}\n"));
        try std.testing.expectEqual(@as(usize, 8), std.mem.count(u8, out, "1.500000"));
    }
    {
        var line: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer line.deinit();
        try formatOpen(&line.writer, .{});
        const out = line.written();
        const prefix = "{\"ev\":\"open\",\"schema\":\"tf.glm53.round-trace/v1\",\"draft_chances\":0,\"live_costs\":0,\"lane_chances\":0,\"copyd\":0,\"copy_rows\":1,\"verify_max\":8,\"conf_temp\":1.000000,\"dspark\":0,\"recal\":\"";
        try std.testing.expect(std.mem.startsWith(u8, out, prefix));
        try std.testing.expect(std.mem.endsWith(u8, out, "\",\"sweep\":0}\n"));
        for (out[prefix.len .. prefix.len + 64]) |c| try std.testing.expectEqual(@as(u8, '0'), c);
        var hex_line: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer hex_line.deinit();
        try formatOpen(&hex_line.writer, .{ .recal = @splat(0xab) });
        const hex_out = hex_line.written();
        for (hex_out[prefix.len .. prefix.len + 64], 0..) |c, i| try std.testing.expectEqual(if (i % 2 == 0) @as(u8, 'a') else @as(u8, 'b'), c);
    }
}
