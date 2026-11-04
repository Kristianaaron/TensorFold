// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GPU parity and row-invariance checks for SMLA-P (tf_glm53_smla_p, TF_GLM53_SMLA_P=1): the query-blocked prefill
//! attend against tf_glm53_attend_mma (prefill, gather-on-prefill) on the same planes and selections. SMLA-FP8
//! (tf_glm53_smla_p_fp8, TF_GLM53_SMLA_FP8=1): the e4m3 score GEMM against bf16 SMLA-P (the lossy delta: output NMSE,
//! LSE, attention KL and top-mass overlap on sampled rows from a host emulation of the e4m3 scores), row invariance
//! bit-exact, and a bench of both. SMLA-P3 (tf_glm53_smla_union + tf_glm53_smla_p3, TF_GLM53_SMLA_P3=1, command
//! glm53-smla-p3): byte-equal to SMLA-P on dense, sparse and mixed pieces of odd row counts on both ranks, row
//! invariance against the SMLA-P batch, and a bench.
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("mla.zig");
const types = @import("types");

test "GLM-5.3 SMLA-P declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

const token_cols = 2048;
const cache_rows = 4096; // plane slots: positions below 8192 at world 2
const world = 2;
const o_row_bytes = 32 * mla.latent_dim * 2;
const lse_row_bytes = 32 * 4;
const partial_row_bytes = o_row_bytes + lse_row_bytes;
/// Output tolerance against attend_mma, per row-head: 2 bf16 ulps of the row-head's largest |output| + 1e-3. SMLA-P
/// reassociates fp32 sums only (4 K-quarter score partials vs 2 halves; a tree softmax sum vs a serial one), which can
/// flip a rare bf16 rounding of P or of the output. LSE: 1e-4 absolute; -inf (empty rows) exactly.
const lse_tol = 1e-4;

const Shape = struct { rows: usize, rank: usize, start: usize };
const Kernel = enum { mma, smla, fp8, p3 };

fn label(kernel: Kernel) []const u8 {
    return switch (kernel) {
        .mma => "attend_mma",
        .smla => "smla_p",
        .fp8 => "smla_fp8",
        .p3 => "smla_p3",
    };
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    var kernels = try mla.Mla.load(module);
    kernels.attend_kernel = .mma;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    // Positions are consecutive prompt positions from `start`: 1 sparse row (7500) or 1 dense row (1000); 7 rows
    // across the dense/sparse boundary (2044..2050); 64 sparse rows deep in the context (7000..7063); 512 rows across
    // the boundary (1948..2459: 100 dense, 412 sparse). Sparse rows: 2048 of every key <= position, adjacent rows
    // sharing about 85-95%; every 11th row from 4 empty, from 7 short (about 1,500 keys), from 9 with invalid entries.
    const shapes = [_]Shape{
        .{ .rows = 1, .rank = 0, .start = 7500 }, .{ .rows = 1, .rank = 1, .start = 1000 },
        .{ .rows = 7, .rank = 0, .start = 2044 }, .{ .rows = 7, .rank = 1, .start = 2044 },
        .{ .rows = 64, .rank = 0, .start = 7000 }, .{ .rows = 64, .rank = 1, .start = 7000 },
        .{ .rows = 512, .rank = 0, .start = 1948 }, .{ .rows = 512, .rank = 1, .start = 1948 },
    };
    for (shapes) |shape| try runShape(gpu, stream, &kernels, shape);
}

const Inputs = struct {
    qa: u64,
    qpe: u64,
    latent: u64,
    rope: u64,
    stage: types.StagePlanePair,
    positions: u64,
    tokens: u64,
    slots: u64,
    global_counts: u64,
    local_counts: u64,
    dense_len: u64,
    /// SMLA-P3 only: the union table (mla.smlaP3TableBytes(rows) bytes or more).
    table: u64 = 0,
    table_bytes: u64 = 0,

    /// Rows [first, first + n) of the inputs, output [n][32][512] BF16 + [n][32] FP32 LSE at `out`.
    fn launch(inputs: Inputs, kernels: *const mla.Mla, stream: cuda.Stream, kernel: Kernel, first: usize, n: usize, out: u64, rank: usize) !void {
        const qa = inputs.qa + first * 32 * mla.latent_dim * 2;
        const qpe = inputs.qpe + first * 32 * mla.rope_dim * 2;
        const pos = inputs.positions + first * 8;
        const tok = inputs.tokens + first * token_cols * 4;
        const sl = inputs.slots + first * token_cols * 4;
        const gc = inputs.global_counts + first * 4;
        const lc = inputs.local_counts + first * 4;
        const dl = inputs.dense_len + first * 4;
        switch (kernel) {
            .mma => try kernels.attend(stream, qa, qpe, inputs.latent, inputs.rope, inputs.stage, pos, tok, sl, gc, lc, dl, out, n, 32, rank, world, true, 0),
            .smla => try kernels.attendSmlaP(stream, qa, qpe, inputs.latent, inputs.rope, inputs.stage, pos, tok, sl, gc, lc, dl, out, n, 32, rank, world),
            .fp8 => try kernels.attendSmlaPFp8(stream, qa, qpe, inputs.latent, inputs.rope, inputs.stage, pos, tok, sl, gc, lc, dl, out, n, 32, rank, world),
            .p3 => try kernels.attendSmlaP3(stream, qa, qpe, inputs.latent, inputs.rope, inputs.stage, pos, tok, sl, gc, lc, dl, out, n, 32, rank, world, inputs.table, inputs.table_bytes),
        }
    }
};

const Cand = struct { score: f32, pos: i32 };

fn candBefore(_: void, a: Cand, b: Cand) bool {
    return a.score > b.score or (a.score == b.score and a.pos < b.pos);
}

fn noise(p: u64, r: u64, seed: u64) f32 {
    var x = (p *% 0x9E3779B97F4A7C15) ^ (r *% 0xC2B2AE3D27D4EB4F) ^ seed;
    x ^= x >> 31;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 29;
    return @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
}

fn runShape(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, shape: Shape) !void {
    const rows = shape.rows;
    const seed: u64 = 0x5317_a000 + rows * 4 + shape.rank;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const q = try gpu.gpa.alloc(u16, rows * 32 * mla.latent_dim);
    defer gpu.gpa.free(q);
    const qp = try gpu.gpa.alloc(u16, rows * 32 * mla.rope_dim);
    defer gpu.gpa.free(qp);
    const latent = try gpu.gpa.alloc(u16, cache_rows * mla.latent_dim);
    defer gpu.gpa.free(latent);
    const rope = try gpu.gpa.alloc(u16, cache_rows * mla.rope_dim);
    defer gpu.gpa.free(rope);
    const stage_latent = try gpu.gpa.alloc(u16, cache_rows * mla.latent_dim);
    defer gpu.gpa.free(stage_latent);
    const stage_rope = try gpu.gpa.alloc(u16, cache_rows * mla.rope_dim);
    defer gpu.gpa.free(stage_rope);
    // Q in [-4, 4] and keys in [-1, 1] (exact bf16 grids): scores spread a few units, so the online max moves.
    fillBf16(q, random, 16);
    fillBf16(qp, random, 16);
    fillBf16(latent, random, 64);
    fillBf16(rope, random, 64);
    fillBf16(stage_latent, random, 64);
    fillBf16(stage_rope, random, 64);

    const positions = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(positions);
    const tokens = try gpu.gpa.alloc(i32, rows * token_cols);
    defer gpu.gpa.free(tokens);
    const slots = try gpu.gpa.alloc(i32, rows * token_cols);
    defer gpu.gpa.free(slots);
    const global_counts = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(global_counts);
    const local_counts = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(local_counts);
    const dense_len = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(dense_len);
    @memset(tokens, -1);
    @memset(slots, -1);
    @memset(local_counts, 0);

    const max_pos = shape.start + rows;
    const base = try gpu.gpa.alloc(f32, max_pos);
    defer gpu.gpa.free(base);
    for (base) |*b| b.* = random.float(f32);
    const cands = try gpu.gpa.alloc(Cand, max_pos);
    defer gpu.gpa.free(cands);
    var visible_keys: usize = 0;
    for (0..rows) |r| {
        const pos = shape.start + r;
        positions[r] = @intCast(pos);
        dense_len[r] = if (pos < 2048) @intCast(pos + 1) else 0;
        if (pos < 2048) {
            global_counts[r] = @intCast(pos + 1);
            visible_keys += pos + 1;
            continue;
        }
        // Top-k of a shared base score plus per-row noise (about 400 keys inside the noise band), a 32-key local
        // window and 4 sink keys always in: adjacent rows share most of their lists, as DSA top-k does.
        const eps: f32 = 400.0 / @as(f32, @floatFromInt(pos + 1));
        for (0..pos + 1) |p| {
            var score = base[p] + eps * noise(p, r + shape.start, seed);
            if (p + 32 > pos or p < 4) score += 2;
            cands[p] = .{ .score = score, .pos = @intCast(p) };
        }
        const kind = r % 11;
        const k: usize = if (kind == 7) 1500 + r % 97 else token_cols;
        std.mem.sort(Cand, cands[0 .. pos + 1], {}, candBefore);
        const row_tokens = tokens[r * token_cols ..][0..token_cols];
        for (0..k) |j| row_tokens[j] = cands[j].pos;
        std.mem.sort(i32, row_tokens[0..k], {}, std.sort.asc(i32));
        if (kind == 9) { // invalid entries inside the count: keys past the position, then -1
            for (k - 32..k - 8) |j| row_tokens[j] = @intCast(pos + 1 + (j - (k - 32)));
            for (k - 8..k) |j| row_tokens[j] = -1;
        }
        global_counts[r] = if (kind == 4) 0 else @intCast(k);
        if (kind != 4) {
            const last: i64 = @intCast(pos);
            for (row_tokens[0..k]) |t| {
                visible_keys += @intFromBool(t >= 0 and t <= last);
            }
        }
    }
    const overlap = adjacentOverlap(positions, tokens, global_counts);
    const loads = keyLoads(positions, tokens, global_counts);

    var dqa = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dqa.free();
    var dqpe = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qp));
    defer dqpe.free();
    var dlatent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(latent));
    defer dlatent.free();
    var drope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rope));
    defer drope.free();
    var dstage_latent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(stage_latent));
    defer dstage_latent.free();
    var dstage_rope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(stage_rope));
    defer dstage_rope.free();
    var dpositions = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
    defer dpositions.free();
    var dtokens = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(tokens));
    defer dtokens.free();
    var dslots = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots));
    defer dslots.free();
    var dglobal_counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(global_counts));
    defer dglobal_counts.free();
    var dlocal_counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(local_counts));
    defer dlocal_counts.free();
    var ddense_len = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(dense_len));
    defer ddense_len.free();
    const inputs: Inputs = .{
        .qa = dqa.ptr,
        .qpe = dqpe.ptr,
        .latent = dlatent.ptr,
        .rope = drope.ptr,
        .stage = .{
            .latent = .{ .data = dstage_latent, .dtype = .bf16, .rows = cache_rows, .cols = mla.latent_dim, .stride = mla.latent_dim },
            .rope = .{ .data = dstage_rope, .dtype = .bf16, .rows = cache_rows, .cols = mla.rope_dim, .stride = mla.rope_dim },
        },
        .positions = dpositions.ptr,
        .tokens = dtokens.ptr,
        .slots = dslots.ptr,
        .global_counts = dglobal_counts.ptr,
        .local_counts = dlocal_counts.ptr,
        .dense_len = ddense_len.ptr,
    };

    var dref = try cuda.DeviceBuffer.alloc(gpu.d, rows * partial_row_bytes);
    defer dref.free();
    var dgot = try cuda.DeviceBuffer.alloc(gpu.d, rows * partial_row_bytes);
    defer dgot.free();
    var dfp8 = try cuda.DeviceBuffer.alloc(gpu.d, rows * partial_row_bytes);
    defer dfp8.free();
    try inputs.launch(kernels, stream, .mma, 0, rows, dref.ptr, shape.rank);
    try inputs.launch(kernels, stream, .smla, 0, rows, dgot.ptr, shape.rank);
    try inputs.launch(kernels, stream, .fp8, 0, rows, dfp8.ptr, shape.rank);
    try stream.synchronize();
    const ref = try download(gpu, dref);
    defer gpu.gpa.free(ref);
    const got = try download(gpu, dgot);
    defer gpu.gpa.free(got);
    const fp8 = try download(gpu, dfp8);
    defer gpu.gpa.free(fp8);
    try compareParity(ref, got, shape, overlap, loads);
    try compareFp8(got, fp8, shape);
    const host: Host = .{ .q = q, .qp = qp, .latent = latent, .rope = rope, .stage_latent = stage_latent, .stage_rope = stage_rope, .positions = positions, .tokens = tokens, .counts = global_counts };
    try emulateFp8(gpu.gpa, host, got, fp8, shape);
    try invariance(gpu, stream, kernels, inputs, got, shape, .smla);
    try invariance(gpu, stream, kernels, inputs, fp8, shape, .fp8);
    if (rows == 512 and shape.rank == 0) try bench(gpu, stream, kernels, inputs, .{ dref.ptr, dgot.ptr, dfp8.ptr }, shape, visible_keys);
}

/// Mean and minimum |A n B| / |A| over adjacent full sparse rows (both 2048 keys, no special kind).
fn adjacentOverlap(positions: []const i64, tokens: []const i32, counts: []const i32) struct { mean: f64, min: f64, pairs: usize } {
    var total: f64 = 0;
    var lowest: f64 = 1;
    var pairs: usize = 0;
    var r: usize = 1;
    while (r < positions.len) : (r += 1) {
        if (positions[r - 1] < 2048 or counts[r - 1] != token_cols or counts[r] != token_cols) continue;
        if ((r - 1) % 11 == 9 or r % 11 == 9) continue;
        const a = tokens[(r - 1) * token_cols ..][0..token_cols];
        const b = tokens[r * token_cols ..][0..token_cols];
        var i: usize = 0;
        var j: usize = 0;
        var shared: usize = 0;
        while (i < a.len and j < b.len) {
            if (a[i] == b[j]) {
                shared += 1;
                i += 1;
                j += 1;
            } else if (a[i] < b[j]) {
                i += 1;
            } else {
                j += 1;
            }
        }
        const f = @as(f64, @floatFromInt(shared)) / token_cols;
        total += f;
        lowest = @min(lowest, f);
        pairs += 1;
    }
    return .{ .mean = if (pairs == 0) 1 else total / @as(f64, @floatFromInt(pairs)), .min = lowest, .pairs = pairs };
}

/// Key rows loaded: attend_mma loads every list entry of every row; SMLA-P loads each step's union of the pair's
/// valid tile-t keys once (mla.cu: rows 2i and 2i + 1 in lockstep over their own 16-entry tiles).
fn keyLoads(positions: []const i64, tokens: []const i32, counts: []const i32) struct { smla: usize, mma: usize } {
    var smla: usize = 0;
    var mma: usize = 0;
    const n = positions.len;
    var b0: usize = 0;
    while (b0 < n) : (b0 += mla.smla_p_rows) {
        var steps: usize = 0;
        for (b0..@min(n, b0 + mla.smla_p_rows)) |r| {
            const len = rowLen(positions[r], counts[r]);
            mma += len;
            steps = @max(steps, (len + 15) / 16);
        }
        for (0..steps) |t| {
            var seen: [mla.smla_p_rows * 16]i64 = @splat(-1);
            var distinct: usize = 0;
            for (b0..@min(n, b0 + mla.smla_p_rows)) |r| {
                const len = rowLen(positions[r], counts[r]);
                for (@min(len, t * 16)..@min(len, t * 16 + 16)) |j| {
                    const key = rowKey(positions[r], tokens[r * token_cols ..][0..token_cols], j);
                    if (key < 0) continue;
                    const dup = std.mem.indexOfScalar(i64, seen[0..distinct], key) != null;
                    if (!dup) {
                        seen[distinct] = key;
                        distinct += 1;
                    }
                }
            }
            smla += distinct;
        }
    }
    return .{ .smla = smla, .mma = mma };
}

fn rowLen(pos: i64, count: i32) usize {
    return if (pos < 2048) @intCast(pos + 1) else @intCast(count);
}

fn rowKey(pos: i64, list: []const i32, j: usize) i64 {
    const key: i64 = if (pos < 2048) @intCast(j) else list[j];
    return if (key < 0 or key > pos) -1 else key;
}

fn compareParity(ref: []const u8, got: []const u8, shape: Shape, overlap: anytype, loads: anytype) !void {
    const rows = shape.rows;
    const ro = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, ref[0 .. rows * o_row_bytes]));
    const go = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got[0 .. rows * o_row_bytes]));
    var max_abs: f64 = 0;
    var max_rel: f64 = 0;
    var worst: f64 = 0;
    var equal: usize = 0;
    for (0..rows * 32) |rh| {
        var scale: f64 = 0;
        for (0..mla.latent_dim) |d| scale = @max(scale, @abs(@as(f64, bf16Value(ro[rh * mla.latent_dim + d]))));
        const tol = 2.0 * scale / 128.0 + 1e-3;
        for (0..mla.latent_dim) |d| {
            const i = rh * mla.latent_dim + d;
            const rv: f64 = bf16Value(ro[i]);
            const gv: f64 = bf16Value(go[i]);
            const diff = @abs(rv - gv);
            equal += @intFromBool(ro[i] == go[i]);
            max_abs = @max(max_abs, diff);
            if (@abs(rv) >= 0.01) max_rel = @max(max_rel, diff / @abs(rv));
            worst = @max(worst, diff / tol);
            if (!(diff <= tol)) {
                std.debug.print("FAIL glm53-smla-p rows={d} rank={d}: row {d} head {d} dim {d}: attend_mma {d} smla_p {d} tol {d}\n", .{ rows, shape.rank, rh / 32, rh % 32, d, rv, gv, tol });
                return error.TestFailed;
            }
        }
    }
    const rl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, ref[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    const gl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    var lse_abs: f64 = 0;
    var empty: usize = 0;
    for (rl, gl, 0..) |a, b, i| {
        if (std.math.isInf(a) or std.math.isInf(b) or std.math.isNan(b)) {
            if (a != b) {
                std.debug.print("FAIL glm53-smla-p rows={d} rank={d}: LSE row {d} head {d}: attend_mma {d} smla_p {d}\n", .{ rows, shape.rank, i / 32, i % 32, a, b });
                return error.TestFailed;
            }
            empty += 1;
            continue;
        }
        lse_abs = @max(lse_abs, @abs(@as(f64, a) - b));
    }
    const equal_pct = 100.0 * @as(f64, @floatFromInt(equal)) / @as(f64, @floatFromInt(ro.len));
    const load_ratio = @as(f64, @floatFromInt(loads.smla)) / @as(f64, @floatFromInt(@max(loads.mma, 1)));
    std.debug.print("RESULT glm53-smla-p rows={d} rank={d}: out max_abs={e:.3} max_rel(|ref|>=0.01)={e:.3} worst={d:.3} of tol, bit-equal {d:.2}%; lse max_abs={e:.3}, empty row-heads {d}; adjacent overlap mean {d:.3} min {d:.3} ({d} pairs); key-row loads smla_p {d} vs attend_mma {d} ({d:.3})\n", .{ rows, shape.rank, max_abs, max_rel, worst, equal_pct, lse_abs, empty, overlap.mean, overlap.min, overlap.pairs, loads.smla, loads.mma, load_ratio });
    if (lse_abs > lse_tol) {
        std.debug.print("FAIL glm53-smla-p rows={d} rank={d}: LSE max abs {e:.3} > {e:.1}\n", .{ rows, shape.rank, lse_abs, lse_tol });
        return error.TestFailed;
    }
    std.debug.print("PASS glm53-smla-p parity rows={d} rank={d}: outputs within 2 bf16 ulps of the row-head max + 1e-3, LSE within 1e-4, -inf rows exact\n", .{ rows, shape.rank });
}

/// Bit-exact: each checked row alone (a block of one), and every row in a shifted pairing (rows 1.. so that row r's
/// block-mate changes), against the full batch of the same kernel (SMLA-P or SMLA-FP8).
fn invariance(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, inputs: Inputs, batch: []const u8, shape: Shape, kernel: Kernel) !void {
    const rows = shape.rows;
    var alone = try cuda.DeviceBuffer.alloc(gpu.d, partial_row_bytes);
    defer alone.free();
    const sample = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100, 101, 255, 256, 257, 510, 511 };
    var checked: usize = 0;
    for (0..rows) |r| {
        if (rows > 64 and std.mem.indexOfScalar(usize, &sample, r) == null) continue;
        try inputs.launch(kernels, stream, kernel, r, 1, alone.ptr, shape.rank);
        try stream.synchronize();
        const one = try download(gpu, alone);
        defer gpu.gpa.free(one);
        try expectRowEqual(one, 1, 0, batch, rows, r, shape, kernel, "alone");
        checked += 1;
    }
    if (rows >= 2) {
        var shifted = try cuda.DeviceBuffer.alloc(gpu.d, (rows - 1) * partial_row_bytes);
        defer shifted.free();
        try inputs.launch(kernels, stream, kernel, 1, rows - 1, shifted.ptr, shape.rank);
        try stream.synchronize();
        const moved = try download(gpu, shifted);
        defer gpu.gpa.free(moved);
        for (1..rows) |r| try expectRowEqual(moved, rows - 1, r - 1, batch, rows, r, shape, kernel, "shifted pairing");
    }
    std.debug.print("PASS glm53-smla-p row invariance ({s}) rows={d} rank={d}: {d} rows alone and {d} rows in a shifted pairing bit-exact vs the batch\n", .{ label(kernel), rows, shape.rank, checked, if (rows >= 2) rows - 1 else 0 });
}

fn expectRowEqual(a: []const u8, a_rows: usize, a_row: usize, b: []const u8, b_rows: usize, b_row: usize, shape: Shape, kernel: Kernel, how: []const u8) !void {
    const ao = a[a_row * o_row_bytes ..][0..o_row_bytes];
    const bo = b[b_row * o_row_bytes ..][0..o_row_bytes];
    const al = a[a_rows * o_row_bytes + a_row * lse_row_bytes ..][0..lse_row_bytes];
    const bl = b[b_rows * o_row_bytes + b_row * lse_row_bytes ..][0..lse_row_bytes];
    if (std.mem.eql(u8, ao, bo) and std.mem.eql(u8, al, bl)) return;
    const ah = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, ao));
    const bh = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, bo));
    var first: ?usize = null;
    var max_abs: f64 = 0;
    for (ah, bh, 0..) |x, y, i| {
        if (x != y) {
            if (first == null) first = i;
            max_abs = @max(max_abs, @abs(@as(f64, bf16Value(x)) - bf16Value(y)));
        }
    }
    std.debug.print("FAIL glm53-smla-p row invariance ({s}, {s}) rows={d} rank={d} row={d}: first differing output {any}, max abs {e:.3}, LSE equal {}\n", .{ label(kernel), how, shape.rows, shape.rank, b_row, first, max_abs, std.mem.eql(u8, al, bl) });
    return error.RowInvariant;
}

/// BENCH: attend_mma vs SMLA-P vs SMLA-FP8 on the 512-row shape, 2 warm-ups then 10 timed launches each.
fn bench(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, inputs: Inputs, outs: [3]u64, shape: Shape, visible_keys: usize) !void {
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    var us: [3]f64 = undefined;
    for ([_]Kernel{ .mma, .smla, .fp8 }, outs, 0..) |kernel, out, idx| {
        for (0..12) |i| {
            if (i == 2) try e0.record(stream);
            try inputs.launch(kernels, stream, kernel, 0, shape.rows, out, shape.rank);
        }
        try e1.record(stream);
        try stream.synchronize();
        us[idx] = @as(f64, try e0.elapsedMs(e1)) * 1000.0 / 10.0;
    }
    const flops = @as(f64, @floatFromInt(visible_keys)) * 32.0 * (576.0 + 512.0) * 2.0;
    std.debug.print("BENCH glm53-smla-p rows={d} (dense+sparse, {d} visible keys): attend_mma {d:.1} us ({d:.2} TFLOPS), smla_p {d:.1} us ({d:.2} TFLOPS), speedup {d:.2}x\n", .{ shape.rows, visible_keys, us[0], flops / us[0] * 1e-6, us[1], flops / us[1] * 1e-6, us[0] / us[1] });
    // Same FLOP count (576-dim scores + 512-dim P.V); 53% of it (the scores) runs on e4m3 tensor cores in smla_fp8.
    std.debug.print("BENCH glm53-smla-fp8 rows={d} (dense+sparse, {d} visible keys): smla_p {d:.1} us ({d:.2} TFLOPS), smla_fp8 {d:.1} us ({d:.2} TFLOPS), speedup {d:.2}x\n", .{ shape.rows, visible_keys, us[1], flops / us[1] * 1e-6, us[2], flops / us[2] * 1e-6, us[1] / us[2] });
}

/// SMLA-FP8 vs bf16 SMLA-P over every row-head: output NMSE (sum of squared differences / sum of squared bf16 outputs)
/// and max abs, LSE max and mean abs; empty row-heads exact in both (zeros, -inf). The bounds only catch broken
/// layouts (uniform random data gives an NMSE of a few 1e-3); the quality decision is the KLD gate's.
const fp8_nmse_max: f64 = 0.03;
const fp8_lse_max: f64 = 0.25;

fn compareFp8(ref: []const u8, got: []const u8, shape: Shape) !void {
    const rows = shape.rows;
    const ro = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, ref[0 .. rows * o_row_bytes]));
    const go = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got[0 .. rows * o_row_bytes]));
    const rl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, ref[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    const gl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var max_abs: f64 = 0;
    var lse_max: f64 = 0;
    var lse_sum: f64 = 0;
    var live: usize = 0;
    var empty: usize = 0;
    for (0..rows * 32) |rh| {
        const a = rl[rh];
        const b = gl[rh];
        if (std.math.isInf(a) or std.math.isInf(b) or std.math.isNan(b)) {
            var zeros = a == b;
            for (0..mla.latent_dim) |d| zeros = zeros and ro[rh * mla.latent_dim + d] == 0 and go[rh * mla.latent_dim + d] == 0;
            if (!zeros) {
                std.debug.print("FAIL glm53-smla-fp8 rows={d} rank={d}: empty row {d} head {d}: smla_p LSE {d} smla_fp8 LSE {d}, outputs zero {}\n", .{ rows, shape.rank, rh / 32, rh % 32, a, b, zeros });
                return error.TestFailed;
            }
            empty += 1;
            continue;
        }
        const dl = @abs(@as(f64, a) - b);
        lse_max = @max(lse_max, dl);
        lse_sum += dl;
        live += 1;
        for (0..mla.latent_dim) |d| {
            const i = rh * mla.latent_dim + d;
            const rv: f64 = bf16Value(ro[i]);
            const gv: f64 = bf16Value(go[i]);
            err2 += (gv - rv) * (gv - rv);
            ref2 += rv * rv;
            max_abs = @max(max_abs, @abs(gv - rv));
        }
    }
    const nmse = if (ref2 > 0) err2 / ref2 else 0;
    const lse_mean = if (live > 0) lse_sum / @as(f64, @floatFromInt(live)) else 0;
    std.debug.print("RESULT glm53-smla-fp8 rows={d} rank={d}: vs smla_p out nmse={e:.3} max_abs={e:.3}; lse max_abs={e:.3} mean_abs={e:.3}; live row-heads {d}, empty {d}\n", .{ rows, shape.rank, nmse, max_abs, lse_max, lse_mean, live, empty });
    if (!(nmse <= fp8_nmse_max) or !(lse_max <= fp8_lse_max)) {
        std.debug.print("FAIL glm53-smla-fp8 rows={d} rank={d}: out nmse {e:.3} (bound {e:.1}) or lse max abs {e:.3} (bound {d}) out of the sanity bounds\n", .{ rows, shape.rank, nmse, fp8_nmse_max, lse_max, fp8_lse_max });
        return error.TestFailed;
    }
    std.debug.print("PASS glm53-smla-fp8 vs smla_p rows={d} rank={d}: nmse <= {e:.1}, lse max abs <= {d}, empty rows exact\n", .{ rows, shape.rank, fp8_nmse_max, fp8_lse_max });
}

const Host = struct {
    q: []const u16,
    qp: []const u16,
    latent: []const u16,
    rope: []const u16,
    stage_latent: []const u16,
    stage_rope: []const u16,
    positions: []const i64,
    tokens: []const i32,
    counts: []const i32,
};

/// Sampled rows of the host emulation (dense, sparse, each special kind r % 11: 4 empty, 7 short, 9 invalid entries).
const emu_rows = [_]usize{ 0, 1, 4, 7, 9, 63, 99, 100, 103, 106, 108, 255, 511 };
/// Kernel LSE vs the f64 emulation of the same score math (e4m3 codes, power-of-two scales): fp32 accumulation order
/// and any reduced internal precision of the e4m3 tensor-core sums. A wrong fragment layout or scale misses by units.
const emu_lse_tol: f64 = 5e-3;
const emu_mass: f64 = 0.9;
const qk_dim = mla.latent_dim + mla.rope_dim;

/// 2^k as f64, k in the normal range.
fn pow2(k: i32) f64 {
    return @bitCast(@as(u64, @intCast(k + 1023)) << 52);
}

/// floor(log2 a) for a normal positive f64.
fn exponentOf(a: f64) i32 {
    const bits: u64 = @bitCast(a);
    return @as(i32, @intCast((bits >> 52) & 0x7ff)) - 1023;
}

/// The kernel's scale rule, written independently: the smallest 2^k with amax / 2^k <= 448; 1 for amax 0.
fn pow2Scale(amax: f64) f64 {
    if (!(amax > 0)) return 1;
    var k: i32 = exponentOf(amax) - 9;
    while (amax / pow2(k) > 448) k += 1;
    return pow2(@max(k, -126));
}

fn roundHalfEven(v: f64) f64 {
    const fl = @floor(v);
    const frac = v - fl;
    if (frac > 0.5) return fl + 1;
    if (frac < 0.5) return fl;
    return if (@mod(fl, 2) == 0) fl else fl + 1;
}

/// e4m3 round-to-nearest-even (normal step 2^(e - 3), subnormal step 2^-9), saturating at 448, decoded to f64.
fn e4m3(x: f64) f64 {
    const a = @abs(x);
    if (a == 0) return 0;
    const step = if (a < 0x1p-6) 0x1p-9 else pow2(exponentOf(a) - 3);
    return std.math.copysign(@min(roundHalfEven(a / step) * step, 448.0), x);
}

fn amaxOf(v: []const f64) f64 {
    var m: f64 = 0;
    for (v) |x| m = @max(m, @abs(x));
    return m;
}

const ByScore = struct {
    s: []const f64,
    fn before(ctx: ByScore, a: u32, b: u32) bool {
        return ctx.s[a] > ctx.s[b] or (ctx.s[a] == ctx.s[b] and a < b);
    }
};

fn logSumExp(s: []const f64) f64 {
    var m: f64 = -std.math.inf(f64);
    for (s) |x| m = @max(m, x);
    var sum: f64 = 0;
    for (s) |x| sum += @exp(x - m);
    return m + @log(sum);
}

/// KLD-gate inputs on sampled rows (all 32 heads): f64 scores of the bf16 inputs (reference) and of the e4m3 codes
/// times their scales (what tf_glm53_smla_p_fp8 computes). Checks the kernels' LSE against both (layout and scale
/// bugs), then reports the attention KL(p_bf16 || p_fp8), the top-1 key agreement and the overlap of the keys holding
/// 90% of the reference attention mass with the same number of top keys under the FP8 scores.
fn emulateFp8(gpa: std.mem.Allocator, h: Host, smla: []const u8, fp8: []const u8, shape: Shape) !void {
    const rows = shape.rows;
    const sl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, smla[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    const fl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, fp8[rows * o_row_bytes ..][0 .. rows * lse_row_bytes]));
    const kref = try gpa.alloc(f64, token_cols * qk_dim);
    defer gpa.free(kref);
    const kcode = try gpa.alloc(f64, token_cols * qk_dim);
    defer gpa.free(kcode);
    const kscale = try gpa.alloc(f64, token_cols);
    defer gpa.free(kscale);
    const s_ref = try gpa.alloc(f64, token_cols);
    defer gpa.free(s_ref);
    const s_fp8 = try gpa.alloc(f64, token_cols);
    defer gpa.free(s_fp8);
    const by_ref = try gpa.alloc(u32, token_cols);
    defer gpa.free(by_ref);
    const by_fp8 = try gpa.alloc(u32, token_cols);
    defer gpa.free(by_fp8);
    const in_top = try gpa.alloc(bool, token_cols);
    defer gpa.free(in_top);
    var qv: [qk_dim]f64 = undefined;
    var qc: [qk_dim]f64 = undefined;
    var lse_fp8_err: f64 = 0;
    var lse_bf16_err: f64 = 0;
    var kl_sum: f64 = 0;
    var kl_max: f64 = 0;
    var overlap_sum: f64 = 0;
    var overlap_min: f64 = 1;
    var top1: usize = 0;
    var heads: usize = 0;
    var empty: usize = 0;
    for (emu_rows) |r| {
        if (r >= rows) continue;
        const pos = h.positions[r];
        const len = rowLen(pos, h.counts[r]);
        var nk: usize = 0;
        for (0..len) |j| {
            const key = rowKey(pos, h.tokens[r * token_cols ..][0..token_cols], j);
            if (key < 0) continue;
            const owner = @mod(key, world) == @as(i64, @intCast(shape.rank));
            const slot: usize = @intCast(@divTrunc(key, world));
            const lat = if (owner) h.latent else h.stage_latent;
            const rp = if (owner) h.rope else h.stage_rope;
            const kv = kref[nk * qk_dim ..][0..qk_dim];
            for (0..qk_dim) |d| kv[d] = if (d < mla.latent_dim) bf16Value(lat[slot * mla.latent_dim + d]) else bf16Value(rp[slot * mla.rope_dim + d - mla.latent_dim]);
            const sk = pow2Scale(amaxOf(kv));
            for (kv, kcode[nk * qk_dim ..][0..qk_dim]) |x, *c| c.* = e4m3(x / sk);
            kscale[nk] = sk;
            nk += 1;
        }
        for (0..32) |head| {
            const rh = r * 32 + head;
            if (nk == 0) {
                if (!std.math.isInf(fl[rh]) or !std.math.isInf(sl[rh])) {
                    std.debug.print("FAIL glm53-smla-fp8 emulation rows={d} rank={d}: row {d} has no valid key but LSE smla_p {d} smla_fp8 {d}\n", .{ rows, shape.rank, r, sl[rh], fl[rh] });
                    return error.TestFailed;
                }
                empty += 1;
                continue;
            }
            for (0..qk_dim) |d| qv[d] = if (d < mla.latent_dim) bf16Value(h.q[rh * mla.latent_dim + d]) else bf16Value(h.qp[rh * mla.rope_dim + d - mla.latent_dim]);
            const sq = pow2Scale(amaxOf(&qv));
            for (qv, &qc) |x, *c| c.* = e4m3(x / sq);
            for (0..nk) |i| {
                const kv = kref[i * qk_dim ..][0..qk_dim];
                const kc = kcode[i * qk_dim ..][0..qk_dim];
                var dot: f64 = 0;
                var dot8: f64 = 0;
                for (0..qk_dim) |d| {
                    dot += qv[d] * kv[d];
                    dot8 += qc[d] * kc[d];
                }
                s_ref[i] = dot * 0.0625;
                s_fp8[i] = dot8 * sq * kscale[i] * 0.0625;
            }
            const lse_ref = logSumExp(s_ref[0..nk]);
            const lse8 = logSumExp(s_fp8[0..nk]);
            lse_fp8_err = @max(lse_fp8_err, @abs(@as(f64, fl[rh]) - lse8));
            lse_bf16_err = @max(lse_bf16_err, @abs(@as(f64, sl[rh]) - lse_ref));
            var kl: f64 = 0;
            for (0..nk) |i| {
                const lp = s_ref[i] - lse_ref;
                kl += @exp(lp) * (lp - (s_fp8[i] - lse8));
            }
            kl_sum += kl;
            kl_max = @max(kl_max, kl);
            for (0..nk) |i| {
                by_ref[i] = @intCast(i);
                by_fp8[i] = @intCast(i);
            }
            std.mem.sort(u32, by_ref[0..nk], ByScore{ .s = s_ref[0..nk] }, ByScore.before);
            std.mem.sort(u32, by_fp8[0..nk], ByScore{ .s = s_fp8[0..nk] }, ByScore.before);
            top1 += @intFromBool(by_ref[0] == by_fp8[0]);
            @memset(in_top[0..nk], false);
            var mass: f64 = 0;
            var top: usize = 0;
            while (top < nk and mass < emu_mass) : (top += 1) {
                mass += @exp(s_ref[by_ref[top]] - lse_ref);
                in_top[by_ref[top]] = true;
            }
            var shared: usize = 0;
            for (by_fp8[0..top]) |i| shared += @intFromBool(in_top[i]);
            const ov = @as(f64, @floatFromInt(shared)) / @as(f64, @floatFromInt(top));
            overlap_sum += ov;
            overlap_min = @min(overlap_min, ov);
            heads += 1;
        }
    }
    const n: f64 = @floatFromInt(@max(heads, 1));
    std.debug.print("RESULT glm53-smla-fp8 emulation rows={d} rank={d}: {d} sampled row-heads ({d} empty); kernel LSE vs emulation: smla_fp8 max_abs={e:.3}, smla_p max_abs={e:.3}; attention KL(bf16||fp8) mean={e:.3} max={e:.3}; top-1 key agreement {d:.2}%; top-{d:.0}%-mass key overlap mean={d:.4} min={d:.4}\n", .{ rows, shape.rank, heads, empty, lse_fp8_err, lse_bf16_err, kl_sum / n, kl_max, 100.0 * @as(f64, @floatFromInt(top1)) / n, emu_mass * 100.0, overlap_sum / n, overlap_min });
    if (!(lse_fp8_err <= emu_lse_tol) or !(lse_bf16_err <= emu_lse_tol)) {
        std.debug.print("FAIL glm53-smla-fp8 emulation rows={d} rank={d}: kernel LSE differs from its emulation by more than {e:.1} (smla_fp8 {e:.3}, smla_p {e:.3})\n", .{ rows, shape.rank, emu_lse_tol, lse_fp8_err, lse_bf16_err });
        return error.TestFailed;
    }
    std.debug.print("PASS glm53-smla-fp8 emulation rows={d} rank={d}: kernel LSE within {e:.1} of the e4m3 emulation\n", .{ rows, shape.rank, emu_lse_tol });
}

/// Uniform samples on the bf16-exact grid k / div, k in [-64, 64].
fn fillBf16(values: []u16, random: std.Random, div: f32) void {
    for (values) |*v| {
        const k: i32 = random.intRangeAtMost(i32, -64, 64);
        v.* = bf16Bits(@as(f32, @floatFromInt(k)) / div);
    }
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn bf16Bits(value: f32) u16 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn bf16Value(value: u16) f32 {
    return @as(f32, @bitCast(@as(u32, value) << 16));
}

// SMLA-P3 (glm53-smla-p3): tf_glm53_smla_union + tf_glm53_smla_p3 against tf_glm53_smla_p, byte for byte.

const P3Case = struct { shape: Shape, bench: bool = false };

pub fn runP3(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    var kernels = try mla.Mla.load(module);
    kernels.attend_kernel = .mma;
    try kernels.loadSmlaP3(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    // Production dims: 32 local heads a rank, 576-dim scores (512 latent + 64 rope), 512-dim values; both ranks (owner
    // vs stage plane per key). Row counts are odd, or not multiples of the 2-row block, the 4-pair union block or the
    // 16-key tile. Sparse rows: top-2048 lists (every 11th row from 4 empty, from 7 short, from 9 with invalid entries).
    const cases = [_]P3Case{
        .{ .shape = .{ .rows = 1, .rank = 0, .start = 0 } }, // one dense row with one key
        .{ .shape = .{ .rows = 1, .rank = 1, .start = 7500 } }, // one sparse row
        .{ .shape = .{ .rows = 7, .rank = 0, .start = 2044 } }, // 4 dense + 3 sparse
        .{ .shape = .{ .rows = 7, .rank = 1, .start = 2044 } },
        .{ .shape = .{ .rows = 63, .rank = 1, .start = 0 } }, // dense from position 0: partial first tiles, half last pair
        .{ .shape = .{ .rows = 64, .rank = 0, .start = 7000 } },
        .{ .shape = .{ .rows = 64, .rank = 1, .start = 7000 } },
        .{ .shape = .{ .rows = 255, .rank = 1, .start = 6000 } }, // sparse, 128 pairs, the last one half empty
        .{ .shape = .{ .rows = 513, .rank = 0, .start = 1536 } }, // 512 dense (1,537..2,048 keys) + 1 sparse row
        .{ .shape = .{ .rows = 512, .rank = 0, .start = 1948 }, .bench = true }, // 100 dense + 412 sparse
        .{ .shape = .{ .rows = 512, .rank = 1, .start = 1948 } },
        .{ .shape = .{ .rows = 512, .rank = 1, .start = 1536 }, .bench = true }, // all dense
        .{ .shape = .{ .rows = 511, .rank = 0, .start = 7600 }, .bench = true }, // all sparse top-2048, odd
    };
    for (cases) |c| try runShapeP3(gpu, stream, &kernels, c);
}

fn runShapeP3(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, case: P3Case) !void {
    const shape = case.shape;
    const rows = shape.rows;
    const seed: u64 = 0x5317_b300 + rows * 4 + shape.rank + shape.start * 4096;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const q = try gpu.gpa.alloc(u16, rows * 32 * mla.latent_dim);
    defer gpu.gpa.free(q);
    const qp = try gpu.gpa.alloc(u16, rows * 32 * mla.rope_dim);
    defer gpu.gpa.free(qp);
    const latent = try gpu.gpa.alloc(u16, cache_rows * mla.latent_dim);
    defer gpu.gpa.free(latent);
    const rope = try gpu.gpa.alloc(u16, cache_rows * mla.rope_dim);
    defer gpu.gpa.free(rope);
    const stage_latent = try gpu.gpa.alloc(u16, cache_rows * mla.latent_dim);
    defer gpu.gpa.free(stage_latent);
    const stage_rope = try gpu.gpa.alloc(u16, cache_rows * mla.rope_dim);
    defer gpu.gpa.free(stage_rope);
    fillBf16(q, random, 16);
    fillBf16(qp, random, 16);
    fillBf16(latent, random, 64);
    fillBf16(rope, random, 64);
    fillBf16(stage_latent, random, 64);
    fillBf16(stage_rope, random, 64);

    const positions = try gpu.gpa.alloc(i64, rows);
    defer gpu.gpa.free(positions);
    const tokens = try gpu.gpa.alloc(i32, rows * token_cols);
    defer gpu.gpa.free(tokens);
    const slots = try gpu.gpa.alloc(i32, rows * token_cols);
    defer gpu.gpa.free(slots);
    const global_counts = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(global_counts);
    const local_counts = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(local_counts);
    const dense_len = try gpu.gpa.alloc(i32, rows);
    defer gpu.gpa.free(dense_len);
    @memset(tokens, -1);
    @memset(slots, -1);
    @memset(local_counts, 0);

    // The same selection model as runShape: top-k of a shared base score plus per-row noise, a 32-key local window and
    // 4 sink keys, so adjacent rows share most of their lists (the union path dedupes real overlaps).
    const max_pos = shape.start + rows;
    const base = try gpu.gpa.alloc(f32, max_pos);
    defer gpu.gpa.free(base);
    for (base) |*b| b.* = random.float(f32);
    const cands = try gpu.gpa.alloc(Cand, max_pos);
    defer gpu.gpa.free(cands);
    var visible_keys: usize = 0;
    for (0..rows) |r| {
        const pos = shape.start + r;
        positions[r] = @intCast(pos);
        dense_len[r] = if (pos < 2048) @intCast(pos + 1) else 0;
        if (pos < 2048) {
            global_counts[r] = @intCast(pos + 1);
            visible_keys += pos + 1;
            continue;
        }
        const eps: f32 = 400.0 / @as(f32, @floatFromInt(pos + 1));
        for (0..pos + 1) |p| {
            var score = base[p] + eps * noise(p, r + shape.start, seed);
            if (p + 32 > pos or p < 4) score += 2;
            cands[p] = .{ .score = score, .pos = @intCast(p) };
        }
        const kind = r % 11;
        const k: usize = if (kind == 7) 1500 + r % 97 else token_cols;
        std.mem.sort(Cand, cands[0 .. pos + 1], {}, candBefore);
        const row_tokens = tokens[r * token_cols ..][0..token_cols];
        for (0..k) |j| row_tokens[j] = cands[j].pos;
        std.mem.sort(i32, row_tokens[0..k], {}, std.sort.asc(i32));
        if (kind == 9) { // invalid entries inside the count: keys past the position, then -1
            for (k - 32..k - 8) |j| row_tokens[j] = @intCast(pos + 1 + (j - (k - 32)));
            for (k - 8..k) |j| row_tokens[j] = -1;
        }
        global_counts[r] = if (kind == 4) 0 else @intCast(k);
        if (kind != 4) {
            const last: i64 = @intCast(pos);
            for (row_tokens[0..k]) |t| visible_keys += @intFromBool(t >= 0 and t <= last);
        }
    }

    var dqa = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dqa.free();
    var dqpe = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qp));
    defer dqpe.free();
    var dlatent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(latent));
    defer dlatent.free();
    var drope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rope));
    defer drope.free();
    var dstage_latent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(stage_latent));
    defer dstage_latent.free();
    var dstage_rope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(stage_rope));
    defer dstage_rope.free();
    var dpositions = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
    defer dpositions.free();
    var dtokens = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(tokens));
    defer dtokens.free();
    var dslots = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots));
    defer dslots.free();
    var dglobal_counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(global_counts));
    defer dglobal_counts.free();
    var dlocal_counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(local_counts));
    defer dlocal_counts.free();
    var ddense_len = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(dense_len));
    defer ddense_len.free();
    var dtable = try cuda.DeviceBuffer.alloc(gpu.d, @intCast(mla.smlaP3TableBytes(rows)));
    defer dtable.free();
    // A record the attend reads but the union kernel did not write reads as count 0 / union row 0 (a wrong output the
    // memcmp catches, not a fault).
    try dtable.fill8(0, stream.handle);
    const inputs: Inputs = .{
        .qa = dqa.ptr,
        .qpe = dqpe.ptr,
        .latent = dlatent.ptr,
        .rope = drope.ptr,
        .stage = .{
            .latent = .{ .data = dstage_latent, .dtype = .bf16, .rows = cache_rows, .cols = mla.latent_dim, .stride = mla.latent_dim },
            .rope = .{ .data = dstage_rope, .dtype = .bf16, .rows = cache_rows, .cols = mla.rope_dim, .stride = mla.rope_dim },
        },
        .positions = dpositions.ptr,
        .tokens = dtokens.ptr,
        .slots = dslots.ptr,
        .global_counts = dglobal_counts.ptr,
        .local_counts = dlocal_counts.ptr,
        .dense_len = ddense_len.ptr,
        .table = dtable.ptr,
        .table_bytes = dtable.len,
    };

    // Distinct fill bytes so an output position P3 fails to write cannot match SMLA-P's by accident.
    var dref = try cuda.DeviceBuffer.alloc(gpu.d, rows * partial_row_bytes);
    defer dref.free();
    var dgot = try cuda.DeviceBuffer.alloc(gpu.d, rows * partial_row_bytes);
    defer dgot.free();
    try dref.fill8(0x11, stream.handle);
    try dgot.fill8(0x77, stream.handle);
    try inputs.launch(kernels, stream, .smla, 0, rows, dref.ptr, shape.rank);
    try inputs.launch(kernels, stream, .p3, 0, rows, dgot.ptr, shape.rank);
    try stream.synchronize();
    const ref = try download(gpu, dref);
    defer gpu.gpa.free(ref);
    const got = try download(gpu, dgot);
    defer gpu.gpa.free(got);
    try expectBytesP3(ref, got, shape);
    // Row invariance of P3 against the SMLA-P batch: rows alone and every row in a shifted pairing, byte-equal.
    try invariance(gpu, stream, kernels, inputs, ref, shape, .p3);
    // Re-run over the table P3 just used (stale records from the previous launch): still byte-equal.
    try dgot.fill8(0x77, stream.handle);
    try inputs.launch(kernels, stream, .p3, 0, rows, dgot.ptr, shape.rank);
    try stream.synchronize();
    const again = try download(gpu, dgot);
    defer gpu.gpa.free(again);
    try expectBytesP3(ref, again, shape);
    if (case.bench) try benchP3(gpu, stream, kernels, inputs, .{ dref.ptr, dgot.ptr }, shape, visible_keys);
}

/// The whole [rows][32][512] BF16 output plane and [rows][32] FP32 LSE plane, byte for byte.
fn expectBytesP3(ref: []const u8, got: []const u8, shape: Shape) !void {
    const rows = shape.rows;
    if (std.mem.eql(u8, ref, got)) {
        std.debug.print("PASS glm53-smla-p3 rows={d} rank={d} start={d}: output and LSE planes byte-equal to smla_p ({d} bytes)\n", .{ rows, shape.rank, shape.start, ref.len });
        return;
    }
    const first = std.mem.indexOfDiff(u8, ref, got).?;
    var differing: usize = 0;
    for (ref, got) |a, b| differing += @intFromBool(a != b);
    if (first < rows * o_row_bytes) {
        const e = first / 2;
        const rv = bf16Value(std.mem.readInt(u16, ref[e * 2 ..][0..2], .little));
        const gv = bf16Value(std.mem.readInt(u16, got[e * 2 ..][0..2], .little));
        std.debug.print("FAIL glm53-smla-p3 rows={d} rank={d} start={d}: output differs first at row {d} head {d} dim {d} (smla_p {d} smla_p3 {d}); {d} bytes differ\n", .{ rows, shape.rank, shape.start, e / (32 * mla.latent_dim), (e / mla.latent_dim) % 32, e % mla.latent_dim, rv, gv, differing });
    } else {
        const i = (first - rows * o_row_bytes) / 4;
        std.debug.print("FAIL glm53-smla-p3 rows={d} rank={d} start={d}: LSE differs first at row {d} head {d}; {d} bytes differ\n", .{ rows, shape.rank, shape.start, i / 32, i % 32, differing });
    }
    return error.TestFailed;
}

/// BENCH: SMLA-P vs SMLA-P3 (union pre-kernel + attend) vs the union pre-kernel alone, 2 warm-ups then 10 timed
/// launches each.
fn benchP3(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, inputs: Inputs, outs: [2]u64, shape: Shape, visible_keys: usize) !void {
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    var us: [2]f64 = undefined;
    for ([_]Kernel{ .smla, .p3 }, outs, 0..) |kernel, out, idx| {
        for (0..12) |i| {
            if (i == 2) try e0.record(stream);
            try inputs.launch(kernels, stream, kernel, 0, shape.rows, out, shape.rank);
        }
        try e1.record(stream);
        try stream.synchronize();
        us[idx] = @as(f64, try e0.elapsedMs(e1)) * 1000.0 / 10.0;
    }
    // The union pre-kernel alone (same launch as inside attendSmlaP3).
    const u = mla.smlaUnionPlan(shape.rows);
    for (0..12) |i| {
        if (i == 2) try e0.record(stream);
        var a: cuda.Args = .{};
        inline for (.{ inputs.positions, inputs.tokens, inputs.global_counts, inputs.table, @as(i32, @intCast(shape.rows)), @as(i32, @intCast(shape.rank)), @as(i32, world) }) |v| a.add(v);
        try cuda.launch.launch(kernels.smla_union_fn.?, .{ .grid = .{ .x = u.grid[0] }, .block = .{ .x = u.block } }, stream, &a);
    }
    try e1.record(stream);
    try stream.synchronize();
    const union_us = @as(f64, try e0.elapsedMs(e1)) * 1000.0 / 10.0;
    const flops = @as(f64, @floatFromInt(visible_keys)) * 32.0 * (576.0 + 512.0) * 2.0;
    std.debug.print("BENCH glm53-smla-p3 rows={d} start={d} ({d} visible keys): smla_p {d:.1} us ({d:.2} TFLOPS), smla_p3 {d:.1} us ({d:.2} TFLOPS; union {d:.1} us + attend {d:.1} us), speedup {d:.2}x\n", .{ shape.rows, shape.start, visible_keys, us[0], flops / us[0] * 1e-6, us[1], flops / us[1] * 1e-6, union_us, us[1] - union_us, us[0] / us[1] });
}
