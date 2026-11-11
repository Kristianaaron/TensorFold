// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Bench-only (qmmf2): schedule variants of the FP8B dense-linear kernel (qmmf2.cu), the release body with other
//! template arguments. Checks whole-buffer byte equality against the release lane, then times each variant.
const std = @import("std");
const cuda = @import("cuda");
const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;

const Shape = struct { n: usize, k: usize, out: qmmf.Out };
/// Split-K > 1 shapes (the fused entries at 256 rows and up), then one split-K 1 shape (the plain entries).
const shapes = [_]Shape{
    .{ .n = 8192, .k = 2048, .out = .bf16 },
    .{ .n = 2048, .k = 6144, .out = .bf16 },
    .{ .n = 6144, .k = 1024, .out = .f32 },
    .{ .n = 2624, .k = 6144, .out = .bf16 },
    .{ .n = 12288, .k = 2048, .out = .bf16 },
    .{ .n = 12288, .k = 2048, .out = .f32 },
    .{ .n = 6144, .k = 8192, .out = .f32 }, // prefill o_proj on a rank (32 local heads x 256): about 180 us/token of the trace
};
const fused_row_cases = [_]usize{ 256, 257, 320, 512 };
const plain_row_cases = [_]usize{ 33, 100, 257, 512 };
/// qmmf2.cu: threads a block, row tile, dynamic shared bytes, CTAs an SM the launch bounds ask for.
const Variant = struct { label: []const u8, block: u32, bm: u32, shared: u32, ctas: u32 };
const variants = [_]Variant{
    .{ .label = "s2", .block = 128, .bm = 64, .shared = 24576, .ctas = 4 },
    .{ .label = "w2", .block = 256, .bm = 64, .shared = 49152, .ctas = 2 },
    .{ .label = "w2s2", .block = 256, .bm = 64, .shared = 24576, .ctas = 4 },
    .{ .label = "b128", .block = 256, .bm = 128, .shared = 40960, .ctas = 2 },
};
/// Entries of one variant: [fused][f32] and the grouped entry.
const Fns = struct { entry: [2][2]cuda.Function, grouped: cuda.Function };
const bench_m = 512;
const warmups = 2;
const timed_launches = 10;
const kill_below = 1.10;

fn bandOf(m: usize, bm: usize, k: usize) usize {
    const rows_t = (m + bm - 1) / bm;
    return @max(1, @min(rows_t, (12 << 20) / (bm * k * 2)));
}

/// One variant launch of a single weight: the fused entry when the release plan has K slices, else the plain entry.
fn launchVariant(fns: Fns, v: Variant, stream: cuda.Stream, x: u64, w: qmmf.Weight, y: u64, m: usize, out: qmmf.Out) !void {
    const p = qmmf.planForMode(m, w.n, w.k, true, .fp8b);
    if (p.sk > 1 and !p.fused) return error.Qmmf2NeedsFusedRows;
    const f = fns.entry[@intFromBool(p.fused)][@intFromBool(out == .f32)];
    var args: cuda.Args = .{};
    args.add(x);
    args.add(w.codes);
    args.add(w.scales);
    args.add(w.scale);
    args.add(y);
    args.add(@as(u64, 0));
    args.add(@as(i32, @intCast(m)));
    args.add(@as(i32, @intCast(w.n)));
    args.add(@as(i32, @intCast(w.k)));
    args.add(@as(i32, @intCast(p.sk)));
    args.add(@as(i32, @intCast(w.npad)));
    args.add(@as(i32, @intCast(w.k)));
    args.add(@as(i32, @intCast(bandOf(m, v.bm, w.k))));
    const rows_t = (m + v.bm - 1) / v.bm;
    try cuda.launch.launch(f, .{
        .grid = .{ .x = @intCast(rows_t * ((w.n + 63) / 64)), .y = 1, .z = 1 },
        .block = .{ .x = @intCast(v.block), .y = 1, .z = 1 },
        .shared = v.shared,
    }, stream, &args);
}

/// One grouped variant launch: each weight's blocks in grid.x order, as Lane.matmulGroup lays them out for its tile.
fn launchGroup(fns: Fns, v: Variant, stream: cuda.Stream, x: u64, m: usize, outs: []const qmmf.GroupOut, options: qmmf.GroupOptions) !void {
    const k: usize = outs[0].w.k;
    var a: qmmf.GroupArgsAbi = .{ .x = x, .prefetch = options.prefetch, .prefetch_bytes = if (options.prefetch == 0) 0 else @intCast(options.prefetch_bytes), .m = @intCast(m), .k = @intCast(k), .ldx = @intCast(k), .band = @intCast(bandOf(m, v.bm, k)), .count = @intCast(outs.len), .seg = @splat(.{}) };
    var total: usize = 0;
    for (outs, 0..) |o, i| {
        const p = qmmf.planForMode(m, o.w.n, k, true, .fp8b);
        if (!p.fused) return error.Qmmf2NeedsFusedRows;
        const blocks = ((m + v.bm - 1) / v.bm) * ((o.w.n + 63) / 64);
        a.seg[i] = .{ .w = o.w.codes, .bs = o.w.scales, .out = o.y, .scale = o.w.scale, .n = @intCast(o.w.n), .npad = @intCast(o.w.npad), .sk = @intCast(p.sk), .blocks = @intCast(blocks) };
        total += blocks;
    }
    var args: cuda.Args = .{};
    args.add(a);
    try cuda.launch.launch(fns.grouped, .{
        .grid = .{ .x = @intCast(total), .y = 1, .z = 1 },
        .block = .{ .x = @intCast(v.block), .y = 1, .z = 1 },
        .shared = v.shared,
    }, stream, &args);
}

const WeightData = struct { codes: cuda.DeviceBuffer, scales: cuda.DeviceBuffer, weight: qmmf.Weight };

fn fillWeight(gpu: anytype, random: std.Random, n: usize, k: usize) !WeightData {
    const npad = fp8.padded(n);
    const raw = try gpu.gpa.alloc(u8, n * k);
    defer gpu.gpa.free(raw);
    for (raw) |*v| {
        v.* = random.int(u8);
        while (v.* == 0x7f or v.* == 0xff) v.* = random.int(u8);
    }
    const packed_codes = try gpu.gpa.alloc(u8, npad * k);
    defer gpu.gpa.free(packed_codes);
    fp8.packCodes(packed_codes, raw, n, k);
    const packed_scales = try gpu.gpa.alloc(f32, npad / 64 * (k / 64));
    defer gpu.gpa.free(packed_scales);
    for (packed_scales) |*v| v.* = @as(f32, 1) / @as(f32, @floatFromInt(@as(u32, 1) << random.intRangeAtMost(u5, 2, 12)));
    var codes = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
    errdefer codes.free();
    const scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_scales));
    return .{ .codes = codes, .scales = scales, .weight = .{ .mode = .fp8b, .codes = codes.ptr, .scales = scales.ptr, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } };
}

/// Finite bf16 inputs: random sign and mantissa, exponents 2^-15 .. 2^1.
fn fillRows(gpu: anytype, random: std.Random, m: usize, k: usize) !cuda.DeviceBuffer {
    const xh = try gpu.gpa.alloc(u16, m * k);
    defer gpu.gpa.free(xh);
    for (xh) |*v| {
        const sign: u16 = @as(u16, random.int(u1)) << 15;
        const exponent: u16 = random.intRangeAtMost(u16, 112, 128) << 7;
        v.* = sign | exponent | @as(u16, random.int(u7));
    }
    return cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
}

fn expectSame(expected: []const u8, got: []const u8, label: []const u8, n: usize, k: usize, m: usize) !void {
    if (std.mem.eql(u8, expected, got)) return;
    const at = std.mem.indexOfDiff(u8, expected, got).?;
    std.debug.print("FAIL qmmf2 {s} N={d} K={d} M={d}: first differing byte {d} of {d}\n", .{ label, n, k, m, at, expected.len });
    return error.Qmmf2Mismatch;
}

fn expectVarSame(expected: []const u8, got: []const u8, label: []const u8, n: usize, k: usize, m: usize) !void {
    if (std.mem.eql(u8, expected, got)) return;
    const at = std.mem.indexOfDiff(u8, expected, got).?;
    std.debug.print("FAIL qmmf-var {s} N={d} K={d} M={d}: first differing byte {d} of {d}\n", .{ label, n, k, m, at, expected.len });
    return error.QmmfVarMismatch;
}

fn variantOutParity(gpu: anytype, lane: qmmf.Lane, vf: *const qmmf.VariantFns, stream: cuda.Stream, wd: WeightData, shape: Shape, m: usize, variant: qmmf.Variant, label: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(0x514d4d46564152);
    var x = try fillRows(gpu, prng.random(), m, shape.k);
    defer x.free();
    const out_bytes = m * shape.n * @as(usize, if (shape.out == .bf16) 2 else 4);
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer reference.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer actual.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, lane.partBytes(m, wd.weight)));
    defer part.free();
    const expected = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(expected);
    const got = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(got);
    try reference.fill8(0x7f, stream.handle);
    try lane.matmulOut(stream, x.ptr, shape.k, m, wd.weight, reference.ptr, shape.out, part.ptr);
    try stream.synchronize();
    try reference.download(0, expected);
    try actual.fill8(0x7f, stream.handle);
    try qmmf.matmulOutVariant(lane, vf, variant, stream, x.ptr, shape.k, m, wd.weight, actual.ptr, shape.out, part.ptr);
    try stream.synchronize();
    try actual.download(0, got);
    try expectVarSame(expected, got, label, shape.n, shape.k, m);
}

fn groupVariantParity(gpu: anytype, lane: qmmf.Lane, vf: *const qmmf.VariantFns, stream: cuda.Stream, weights: []const qmmf.Weight, m: usize) !void {
    var prng = std.Random.DefaultPrng.init(0x514d4d46564752);
    var x = try fillRows(gpu, prng.random(), m, 6144);
    defer x.free();
    var refs: [qmmf.max_group]cuda.DeviceBuffer = undefined;
    var outputs: [qmmf.max_group]cuda.DeviceBuffer = undefined;
    var ref_outs: [qmmf.max_group]qmmf.GroupOut = undefined;
    var var_outs: [qmmf.max_group]qmmf.GroupOut = undefined;
    var made: usize = 0;
    defer {
        for (0..made) |i| {
            refs[i].free();
            outputs[i].free();
        }
    }
    for (weights, 0..) |w, i| {
        refs[i] = try cuda.DeviceBuffer.alloc(gpu.d, m * w.n * 2);
        outputs[i] = cuda.DeviceBuffer.alloc(gpu.d, m * w.n * 2) catch |e| {
            refs[i].free();
            return e;
        };
        made = i + 1;
        ref_outs[i] = .{ .w = w, .y = refs[i].ptr };
        var_outs[i] = .{ .w = w, .y = outputs[i].ptr };
        try refs[i].fill8(0x7f, stream.handle);
        try outputs[i].fill8(0x7f, stream.handle);
    }
    try lane.matmulGroup(stream, x.ptr, 6144, m, ref_outs[0..weights.len], .{});
    try stream.synchronize();
    try qmmf.matmulGroupVariant(lane, vf, stream, x.ptr, 6144, m, var_outs[0..weights.len], .{});
    try stream.synchronize();
    for (weights, 0..) |w, i| {
        const expected = try gpu.gpa.alloc(u8, refs[i].len);
        defer gpu.gpa.free(expected);
        const got = try gpu.gpa.alloc(u8, outputs[i].len);
        defer gpu.gpa.free(got);
        try refs[i].download(0, expected);
        try outputs[i].download(0, got);
        var label_buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "group M={d} segment={d}", .{ m, i });
        try expectVarSame(expected, got, label, w.n, 6144, m);
    }
}

/// Whole output buffers (0x7f pre-filled) of every variant equal the release lane's, for one shape and row count.
fn plainParity(gpu: anytype, random: std.Random, lane: qmmf.Lane, stream: cuda.Stream, fns: [variants.len]Fns, wd: WeightData, shape: Shape, m: usize, repeat: bool) !void {
    var x = try fillRows(gpu, random, m, shape.k);
    defer x.free();
    const out_bytes = m * shape.n * @as(usize, if (shape.out == .bf16) 2 else 4);
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer reference.free();
    var actual = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer actual.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, lane.partBytes(m, wd.weight)));
    defer part.free();
    const expected = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(expected);
    const got = try gpu.gpa.alloc(u8, out_bytes);
    defer gpu.gpa.free(got);
    try reference.fill8(0x7f, stream.handle);
    try lane.matmulOut(stream, x.ptr, shape.k, m, wd.weight, reference.ptr, shape.out, part.ptr);
    try stream.synchronize();
    try reference.download(0, expected);
    const runs: usize = if (repeat) 2 else 1;
    for (variants, fns) |v, f| {
        for (0..runs) |_| {
            try actual.fill8(0x7f, stream.handle);
            try launchVariant(f, v, stream, x.ptr, wd.weight, actual.ptr, m, shape.out);
            try stream.synchronize();
            try actual.download(0, got);
            try expectSame(expected, got, v.label, shape.n, shape.k, m);
        }
    }
}

/// Prints where one grouped output differs from its reference: byte counts, the first element, the 64 x 64 output blocks.
fn groupDiag(expected: []const u8, got: []const u8, what: []const u8, seg: usize, n: usize, m: usize, prefetch_on: bool) bool {
    if (std.mem.eql(u8, expected, got)) return true;
    var count: usize = 0;
    var first: usize = 0;
    var last: usize = 0;
    var unwritten: usize = 0;
    var touched: [8 * 64]bool = @splat(false);
    for (expected, got, 0..) |e, g, i| {
        if (e == g) continue;
        if (count == 0) first = i;
        last = i;
        count += 1;
        if (g == 0x7f) unwritten += 1;
        const row = i / (n * 2);
        const col = (i % (n * 2)) / 2;
        touched[@as(usize, @min(row / 64, 7)) * 64 + @as(usize, @min(col / 64, 63))] = true;
    }
    var nblocks: usize = 0;
    for (touched) |t| {
        if (t) nblocks += 1;
    }
    const at = first & ~@as(usize, 1);
    const exp_half = @as(u16, expected[at]) | (@as(u16, expected[at + 1]) << 8);
    const got_half = @as(u16, got[at]) | (@as(u16, got[at + 1]) << 8);
    const all_blocks = ((m + 63) / 64) * ((n + 63) / 64);
    std.debug.print("DIAG qmmf2 group {s} seg={d} N={d} M={d} prefetch={s}: {d} of {d} bytes differ, first {d} last {d}, {d} of them still 0x7f in got; first element expected 0x{x:0>4} got 0x{x:0>4}; {d} of {d} 64x64 blocks touched (row tile, col tile):", .{ what, seg, n, m, @as([]const u8, if (prefetch_on) "on" else "off"), count, expected.len, first, last, unwritten, exp_half, got_half, nblocks, all_blocks });
    var shown: usize = 0;
    for (touched, 0..) |t, i| {
        if (!t or shown >= 16) continue;
        std.debug.print(" ({d},{d})", .{ i / 64, i % 64 });
        shown += 1;
    }
    std.debug.print("\n", .{});
    return false;
}

/// Downloads every segment's reference and output and reports each difference; true when all are byte-equal.
fn groupCompare(gpu: anytype, refs: []const cuda.DeviceBuffer, outputs: []const cuda.DeviceBuffer, weights: []const qmmf.Weight, what: []const u8, m: usize, prefetch_on: bool) !bool {
    var same = true;
    for (weights, 0..) |w, i| {
        const expected = try gpu.gpa.alloc(u8, refs[i].len);
        defer gpu.gpa.free(expected);
        const got = try gpu.gpa.alloc(u8, outputs[i].len);
        defer gpu.gpa.free(got);
        try refs[i].download(0, expected);
        try outputs[i].download(0, got);
        if (!groupDiag(expected, got, what, i, w.n, m, prefetch_on)) same = false;
    }
    return same;
}

/// Diagnostic grouped parity (non-fatal): the release group twice, the release single-weight path per segment, then every
/// variant's group entry, each against the first release group run. Returns true when everything is byte-equal.
fn groupedParity(gpu: anytype, random: std.Random, lane: qmmf.Lane, stream: cuda.Stream, fns: [variants.len]Fns, weights: []const qmmf.Weight, m: usize, prefetch: u64, prefetch_bytes: usize) !bool {
    var x = try fillRows(gpu, random, m, 6144);
    defer x.free();
    var refs: [qmmf.max_group]cuda.DeviceBuffer = undefined;
    var outputs: [qmmf.max_group]cuda.DeviceBuffer = undefined;
    var ref_outs: [qmmf.max_group]qmmf.GroupOut = undefined;
    var var_outs: [qmmf.max_group]qmmf.GroupOut = undefined;
    var made: usize = 0;
    defer {
        for (0..made) |i| {
            refs[i].free();
            outputs[i].free();
        }
    }
    var part_bytes: usize = 4;
    for (weights, 0..) |w, i| {
        refs[i] = try cuda.DeviceBuffer.alloc(gpu.d, m * w.n * 2);
        outputs[i] = cuda.DeviceBuffer.alloc(gpu.d, m * w.n * 2) catch |e| {
            refs[i].free();
            return e;
        };
        made = i + 1;
        ref_outs[i] = .{ .w = w, .y = refs[i].ptr };
        var_outs[i] = .{ .w = w, .y = outputs[i].ptr };
        try refs[i].fill8(0x7f, stream.handle);
        part_bytes = @max(part_bytes, lane.partBytes(m, w));
    }
    var part = try cuda.DeviceBuffer.alloc(gpu.d, part_bytes);
    defer part.free();
    const count = weights.len;
    const prefetch_on = prefetch != 0;
    const options: qmmf.GroupOptions = .{ .prefetch = prefetch, .prefetch_bytes = prefetch_bytes };
    try lane.matmulGroup(stream, x.ptr, 6144, m, ref_outs[0..count], options);
    try stream.synchronize();
    var same = true;

    for (0..count) |i| try outputs[i].fill8(0x7f, stream.handle);
    try lane.matmulGroup(stream, x.ptr, 6144, m, var_outs[0..count], options);
    try stream.synchronize();
    if (!try groupCompare(gpu, refs[0..count], outputs[0..count], weights, "release-group-repeat", m, prefetch_on)) same = false;

    for (0..count) |i| try outputs[i].fill8(0x7f, stream.handle);
    for (weights, 0..) |w, i| try lane.matmulOut(stream, x.ptr, 6144, m, w, outputs[i].ptr, .bf16, part.ptr);
    try stream.synchronize();
    if (!try groupCompare(gpu, refs[0..count], outputs[0..count], weights, "release-single", m, prefetch_on)) same = false;

    for (variants, fns) |v, f| {
        for (0..count) |i| try outputs[i].fill8(0x7f, stream.handle);
        try launchGroup(f, v, stream, x.ptr, m, var_outs[0..count], options);
        try stream.synchronize();
        if (!try groupCompare(gpu, refs[0..count], outputs[0..count], weights, v.label, m, prefetch_on)) same = false;
    }
    return same;
}

/// Mean microseconds a launch between two stream events.
const Clock = struct {
    first: cuda.Event,
    last: cuda.Event,
    stream: cuda.Stream,

    fn begin(c: Clock) !void {
        try c.first.record(c.stream);
    }
    fn finish(c: Clock) !f64 {
        try c.last.record(c.stream);
        try c.stream.synchronize();
        const ms: f64 = try c.first.elapsedMs(c.last);
        return ms * 1000.0 / @as(f64, @floatFromInt(timed_launches));
    }
};

/// Prints the timing columns and returns the best variant's speedup over the release kernel.
fn printTimes(times: [variants.len + 1]f64, flops: f64) f64 {
    std.debug.print(" | ref {d:.1} us ({d:.1} TFLOP/s)", .{ times[0], flops / times[0] / 1e6 });
    var best: usize = 0;
    for (variants, 0..) |v, i| {
        std.debug.print(" | {s} {d:.1} ({d:.1})", .{ v.label, times[i + 1], flops / times[i + 1] / 1e6 });
        if (times[i + 1] < times[best + 1]) best = i;
    }
    const speedup = times[0] / times[best + 1];
    std.debug.print(" | best {s} {d:.2}x\n", .{ variants[best].label, speedup });
    return speedup;
}

fn benchPlain(gpu: anytype, random: std.Random, lane: qmmf.Lane, clock: Clock, fns: [variants.len]Fns, wd: WeightData, shape: Shape, m: usize) !f64 {
    const stream = clock.stream;
    var x = try fillRows(gpu, random, m, shape.k);
    defer x.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, m * shape.n * 4);
    defer output.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, lane.partBytes(m, wd.weight)));
    defer part.free();
    var times: [variants.len + 1]f64 = undefined;
    for (0..warmups) |_| try lane.matmulOut(stream, x.ptr, shape.k, m, wd.weight, output.ptr, shape.out, part.ptr);
    try clock.begin();
    for (0..timed_launches) |_| try lane.matmulOut(stream, x.ptr, shape.k, m, wd.weight, output.ptr, shape.out, part.ptr);
    times[0] = try clock.finish();
    for (variants, fns, 0..) |v, f, i| {
        for (0..warmups) |_| try launchVariant(f, v, stream, x.ptr, wd.weight, output.ptr, m, shape.out);
        try clock.begin();
        for (0..timed_launches) |_| try launchVariant(f, v, stream, x.ptr, wd.weight, output.ptr, m, shape.out);
        times[i + 1] = try clock.finish();
    }
    const sk = qmmf.b16SplitK(shape.n, shape.k);
    std.debug.print("BENCH qmmf2 M={d} N={d} K={d} sk={d} out={s}", .{ m, shape.n, shape.k, sk, @tagName(shape.out) });
    return printTimes(times, 2.0 * @as(f64, @floatFromInt(m * shape.n * shape.k)));
}

fn benchGrouped(gpu: anytype, random: std.Random, lane: qmmf.Lane, clock: Clock, fns: [variants.len]Fns, weights: []const qmmf.Weight, m: usize) !void {
    const stream = clock.stream;
    var x = try fillRows(gpu, random, m, 6144);
    defer x.free();
    var outputs: [qmmf.max_group]cuda.DeviceBuffer = undefined;
    var outs: [qmmf.max_group]qmmf.GroupOut = undefined;
    var made: usize = 0;
    defer {
        for (0..made) |i| outputs[i].free();
    }
    var n: usize = 0;
    for (weights, 0..) |w, i| {
        outputs[i] = try cuda.DeviceBuffer.alloc(gpu.d, m * w.n * 2);
        made = i + 1;
        outs[i] = .{ .w = w, .y = outputs[i].ptr };
        n += w.n;
    }
    const options: qmmf.GroupOptions = .{};
    var times: [variants.len + 1]f64 = undefined;
    for (0..warmups) |_| try lane.matmulGroup(stream, x.ptr, 6144, m, outs[0..weights.len], options);
    try clock.begin();
    for (0..timed_launches) |_| try lane.matmulGroup(stream, x.ptr, 6144, m, outs[0..weights.len], options);
    times[0] = try clock.finish();
    for (variants, fns, 0..) |v, f, i| {
        for (0..warmups) |_| try launchGroup(f, v, stream, x.ptr, m, outs[0..weights.len], options);
        try clock.begin();
        for (0..timed_launches) |_| try launchGroup(f, v, stream, x.ptr, m, outs[0..weights.len], options);
        times[i + 1] = try clock.finish();
    }
    std.debug.print("BENCH qmmf2 M={d} N={d} K=6144 segments={d} out=group", .{ m, n, weights.len });
    _ = printTimes(times, 2.0 * @as(f64, @floatFromInt(m * n * 6144)));
}

fn resource(label: []const u8, entry: []const u8, f: cuda.Function, block: u32, shared: u32) !void {
    const regs = try f.attribute(.num_regs);
    const local = try f.attribute(.local_size_bytes);
    const ctas = try f.occupancy(block, shared);
    std.debug.print("INFO qmmf2 resource {s}-{s}: regs {d} shared {d} B local {d} B {d} CTAs/SM ({d} threads)\n", .{ label, entry, regs, shared, local, ctas, block });
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.qmmf2.len == 0) return error.KernelsNotEmbedded;
    var lane = try qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    if (!lane.clusters) return error.Qmmf2NeedsClusters;
    try lane.loadGroup();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.qmmf2);
    defer module.unload();
    var variant_fns = try qmmf.VariantFns.load(gpu.d);
    defer variant_fns.deinit();
    var fns: [variants.len]Fns = undefined;
    inline for (variants, 0..) |v, i| {
        const names = [2][2][:0]const u8{
            .{ std.fmt.comptimePrint("tf_qmmf2_{s}_p_b16", .{v.label}), std.fmt.comptimePrint("tf_qmmf2_{s}_p_f32", .{v.label}) },
            .{ std.fmt.comptimePrint("tf_qmmf2_{s}_f_b16", .{v.label}), std.fmt.comptimePrint("tf_qmmf2_{s}_f_f32", .{v.label}) },
        };
        for (names, 0..) |pair, fi| {
            for (pair, 0..) |name, oi| {
                const f = try module.function(name);
                try f.allowDynamicShared(v.shared);
                fns[i].entry[fi][oi] = f;
            }
        }
        fns[i].grouped = try module.function(std.fmt.comptimePrint("tf_qmmf2_group_{s}", .{v.label}));
        try fns[i].grouped.allowDynamicShared(v.shared);
    }
    const release_shared = qmmf.sharedBytes(.fp8b, 64);
    try resource("ref", "fused-b16", lane.fns[5][6], 128, release_shared);
    try resource("ref", "plain-b16", lane.fns[5][4], 128, release_shared);
    try resource("ref", "group", lane.group_fns[3].?, 128, release_shared);
    for (variants, fns) |v, f| {
        try resource(v.label, "fused-b16", f.entry[1][0], v.block, v.shared);
        try resource(v.label, "fused-f32", f.entry[1][1], v.block, v.shared);
        try resource(v.label, "plain-b16", f.entry[0][0], v.block, v.shared);
        try resource(v.label, "group", f.grouped, v.block, v.shared);
    }

    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var first_event = try cuda.Event.init(gpu.d, true);
    defer first_event.deinit();
    var last_event = try cuda.Event.init(gpu.d, true);
    defer last_event.deinit();
    const clock: Clock = .{ .first = first_event, .last = last_event, .stream = stream };
    var prng = std.Random.DefaultPrng.init(0x514d4d4632);
    const random = prng.random();

    var gate: [2]f64 = .{ 0, 0 };
    const qmmf_var_rows = [_]usize{ 1, 32, 100, 255, 256, 257, 512, 1024 };
    for (shapes, 0..) |shape, si| {
        var wd = try fillWeight(gpu, random, shape.n, shape.k);
        defer {
            wd.codes.free();
            wd.scales.free();
        }
        const sk = qmmf.b16SplitK(shape.n, shape.k);
        const rows = if (sk > 1) fused_row_cases else plain_row_cases;
        for (rows) |m| try plainParity(gpu, random, lane, stream, fns, wd, shape, m, si == 0 and m == 512);
        if (si == 0 or si == 1 or si == 2 or si == 6) {
            for (qmmf_var_rows) |m| try variantOutParity(gpu, lane, &variant_fns, stream, wd, shape, m, .b128, "table b128");
        }
        std.debug.print("RESULT qmmf2 parity N={d} K={d} out={s} sk={d}: s2, w2, w2s2, b128 byte-equal to release at M={any}\n", .{ shape.n, shape.k, @tagName(shape.out), sk, rows });
        const speedup = try benchPlain(gpu, random, lane, clock, fns, wd, shape, bench_m);
        if (si < gate.len) gate[si] = speedup;
    }
    std.debug.print("PASS qmmf2 plain parity: 7 shapes, whole buffers byte-equal after a 0x7f fill; repeat-run equal at 8192x2048 M=512\n", .{});

    var seg_a = try fillWeight(gpu, random, 2048, 6144);
    defer {
        seg_a.codes.free();
        seg_a.scales.free();
    }
    var seg_b = try fillWeight(gpu, random, 576, 6144);
    defer {
        seg_b.codes.free();
        seg_b.scales.free();
    }
    var seg_c = try fillWeight(gpu, random, 128, 6144);
    defer {
        seg_c.codes.free();
        seg_c.scales.free();
    }
    const qmmf_var_group_weights = [_]qmmf.Weight{ seg_a.weight, seg_b.weight, seg_c.weight };
    for ([_]usize{ 257, 512, 1024 }) |m| try groupVariantParity(gpu, lane, &variant_fns, stream, &qmmf_var_group_weights, m);
    var fallback = try fillWeight(gpu, random, 12288, 6144);
    defer {
        fallback.codes.free();
        fallback.scales.free();
    }
    try variantOutParity(gpu, lane, &variant_fns, stream, fallback, .{ .n = 12288, .k = 6144, .out = .bf16 }, 512, .release, "release fallback");
    std.debug.print("PASS qmmf-var table shapes 8192x2048/bf16, 6144x8192/f32, 2048x6144/bf16, 6144x1024/f32 at M=1,32,100,255,256,257,512,1024; group 2048+576+128 K=6144 at M=257,512,1024; release fallback 12288x6144/bf16 M=512\n", .{});
    const group_weights = [_]qmmf.Weight{ seg_a.weight, seg_b.weight };
    var prefetch_buffer = try cuda.DeviceBuffer.alloc(gpu.d, 16 * 1024 * 1024);
    defer prefetch_buffer.free();
    const counts = [_]usize{ 1, 2 };
    const group_rows = [_]usize{ 512, 257 };
    var group_same = true;
    for (counts) |count| {
        for (group_rows) |m| {
            if (!try groupedParity(gpu, random, lane, stream, fns, group_weights[0..count], m, 0, 0)) group_same = false;
            if (!try groupedParity(gpu, random, lane, stream, fns, group_weights[0..count], m, prefetch_buffer.ptr, prefetch_buffer.len)) group_same = false;
        }
    }
    if (group_same) {
        std.debug.print("PASS qmmf2 grouped parity: N=2048 and N=2048+576, K=6144, M=512,257, prefetch off and on\n", .{});
    } else {
        std.debug.print("FAIL qmmf2 grouped parity: see the DIAG lines (bench continues)\n", .{});
    }
    for (counts) |count| try benchGrouped(gpu, random, lane, clock, fns, group_weights[0..count], bench_m);
    const keep = gate[0] >= kill_below or gate[1] >= kill_below;
    std.debug.print("BENCH qmmf2 gate: 8192x2048 {d:.2}x, 2048x6144 {d:.2}x at M=512, kill below {d:.2}x on both: {s}\n", .{ gate[0], gate[1], @as(f64, kill_below), if (keep) "KEEP" else "KILL" });
    if (!group_same) return error.Qmmf2Mismatch;
    std.debug.print("PASS qmmf2\n", .{});
}
