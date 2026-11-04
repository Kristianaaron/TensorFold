// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 MLA GPU parity checks against the CUDA-order host references (WP6).
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("mla.zig");
const types = @import("types");

test "GLM-5.3 MLA declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    const kernels = try mla.Mla.load(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try qpeCase(gpu, stream, &kernels);
    try kvWriteCase(gpu, stream, &kernels);
    try absorbExpandCases(gpu, stream, &kernels);
    try decodeAbsorbExpandCases(gpu, stream, &kernels);
    try mergeCase(gpu, stream, &kernels);
    try attendCases(gpu, stream, &kernels);
}

/// GPU parity gate for the A3 decode kernel; exercises every legal decode window width and both rank-local head groups.
pub fn runDecodeUnion(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    var kernels = try mla.Mla.load(module);
    kernels.attend_kernel = .mma;
    try kernels.loadAttendVerify(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const latent_host = try gpu.gpa.alloc(u16, 8192 * mla.latent_dim);
    defer gpu.gpa.free(latent_host);
    const rope_host = try gpu.gpa.alloc(u16, 8192 * mla.rope_dim);
    defer gpu.gpa.free(rope_host);
    var seed: u32 = 0x53a3_2026;
    fillBf16(latent_host, &seed);
    fillBf16(rope_host, &seed);
    var latent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(latent_host));
    defer latent.free();
    var rope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rope_host));
    defer rope.free();
    var bench_mma: [16]f32 = @splat(0);
    var bench_verify: [16]f32 = @splat(0);
    const case_names = [_][]const u8{ "dense-mid", "dense-edge", "dense-tiny", "sparse-overlapping", "sparse-disjoint", "full-local", "ragged-invalid", "empty-rows" };
    for ([_]usize{ 32, 64 }) |heads| for (0..case_names.len) |case_id| for (1..mla.verify_max_rows + 1) |rows| {
        const qa_host = try gpu.gpa.alloc(u16, rows * heads * mla.latent_dim);
        defer gpu.gpa.free(qa_host);
        const qp_host = try gpu.gpa.alloc(u16, rows * heads * mla.rope_dim);
        defer gpu.gpa.free(qp_host);
        fillBf16(qa_host, &seed);
        fillBf16(qp_host, &seed);
        if (heads == 64) {
            const qcopy = try gpu.gpa.dupe(u16, qa_host); defer gpu.gpa.free(qcopy);
            const pcopy = try gpu.gpa.dupe(u16, qp_host); defer gpu.gpa.free(pcopy);
            for (0..2) |group| for (0..rows) |r| for (0..32) |h| {
                const src = (r * heads + group * 32 + h);
                const dst = (group * rows + r) * 32 + h;
                @memcpy(qa_host[dst * mla.latent_dim ..][0 .. mla.latent_dim], qcopy[src * mla.latent_dim ..][0 .. mla.latent_dim]);
                @memcpy(qp_host[dst * mla.rope_dim ..][0 .. mla.rope_dim], pcopy[src * mla.rope_dim ..][0 .. mla.rope_dim]);
            };
        }
        const positions_host = try gpu.gpa.alloc(i64, rows);
        defer gpu.gpa.free(positions_host);
        const slots_host = try gpu.gpa.alloc(i32, rows * 2048);
        defer gpu.gpa.free(slots_host);
        const counts_host = try gpu.gpa.alloc(i32, rows);
        defer gpu.gpa.free(counts_host);
        const tokens_host = try gpu.gpa.alloc(i32, rows * 2048);
        defer gpu.gpa.free(tokens_host);
        const global_counts_host = try gpu.gpa.alloc(i32, rows);
        defer gpu.gpa.free(global_counts_host);
        const dense_host = try gpu.gpa.alloc(i32, rows);
        defer gpu.gpa.free(dense_host);
        const ragged = [_]i32{ 1, 15, 17, 63, 65, 640, 1000, 1025, 2, 31, 129, 511, 1023, 1500, 2047, 2048 };
        for (0..rows) |r| {
            positions_host[r] = switch (case_id) {
                0 => 1000 + @as(i64, @intCast(r)),
                1 => 2040 + @as(i64, @intCast(r)),
                2 => @intCast(r),
                3, 7 => 7000 + @as(i64, @intCast(r)),
                4 => 17000 + @as(i64, @intCast(r)),
                5 => 9000 + @as(i64, @intCast(r)),
                else => 7000 + @as(i64, @intCast(r)),
            };
            counts_host[r] = switch (case_id) {
                3, 7 => 1024,
                4 => 1024,
                5 => 2048,
                6 => ragged[r],
                else => 0,
            };
            if (case_id == 7 and (r == 1 or r == 5)) counts_host[r] = 0;
            global_counts_host[r] = 0;
            dense_host[r] = if (positions_host[r] < 2048) @intCast(positions_host[r] + 1) else 0;
            @memset(slots_host[r * 2048 ..][0..2048], -1);
            @memset(tokens_host[r * 2048 ..][0..2048], -1);
            if (case_id >= 3) for (0..@intCast(counts_host[r])) |j| {
                var slot: i32 = switch (case_id) {
                    3, 7 => @intCast(j + (j / 10) * 16),
                    4 => @intCast((8 * j + r) % 8192),
                    5 => @intCast(r + j),
                    6 => if (j % 29 == 0) @intCast(@as(usize, @intCast(positions_host[r])) / 2 + 1) else @intCast(j),
                    else => @intCast(j),
                };
                if (case_id == 6 and j % 13 == 0) slot = -1;
                slots_host[r * 2048 + j] = slot;
                tokens_host[r * 2048 + j] = if (slot >= 0) slot * 2 else -1;
            };
        }
        var qa = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qa_host)); defer qa.free();
        var qp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qp_host)); defer qp.free();
        var positions = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions_host)); defer positions.free();
        var slots = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots_host)); defer slots.free();
        var counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(counts_host)); defer counts.free();
        var tokens = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(tokens_host)); defer tokens.free();
        var global_counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(global_counts_host)); defer global_counts.free();
        var dense = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(dense_host)); defer dense.free();
        const partial_bytes = (heads / 32) * rows * (32 * mla.latent_dim * 2 + 32 * 4);
        var pa = try cuda.DeviceBuffer.alloc(gpu.d, partial_bytes); defer pa.free();
        var pb = try cuda.DeviceBuffer.alloc(gpu.d, partial_bytes); defer pb.free();
        const ws_bytes = mla.splitWorkspaceBytes(rows, heads);
        var wa = try cuda.DeviceBuffer.alloc(gpu.d, ws_bytes); defer wa.free();
        var wb = try cuda.DeviceBuffer.alloc(gpu.d, ws_bytes); defer wb.free();
        for (0..2) |rank| {
            try wb.fill8(0xff, stream.handle);
            try kernels.attend(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, null, positions.ptr, tokens.ptr, slots.ptr, global_counts.ptr, counts.ptr, dense.ptr, pa.ptr, rows, heads, rank, 2, false, wa.ptr);
            try kernels.attendVerify(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, positions.ptr, slots.ptr, counts.ptr, pb.ptr, rows, heads, rank, 2, wb.ptr);
            try stream.synchronize();
            const a = try download(gpu, pa); defer gpu.gpa.free(a);
            const b = try download(gpu, pb); defer gpu.gpa.free(b);
            if (!std.mem.eql(u8, a, b)) {
                const group_bytes = rows * (32 * mla.latent_dim * 2 + 32 * 4);
                for (0..a.len) |byte| if (a[byte] != b[byte]) {
                    const group = byte / group_bytes;
                    const within = byte % group_bytes;
                    const o_bytes = rows * 32 * mla.latent_dim * 2;
                    const row_head = if (within < o_bytes) within / (mla.latent_dim * 2) else (within - o_bytes) / 4;
                    const row = row_head / 32;
                    const head = row_head % 32;
                    const dim = if (within < o_bytes) (within / 2) % mla.latent_dim else mla.latent_dim;
                    std.debug.print("FAIL glm53-decode-union: case={s} rank={d} heads={d} rows={d}: first differing (group={d}, row={d}, head={d}, dim={d})\n", .{ case_names[case_id], rank, heads, rows, group, row, head, dim });
                    break;
                };
                return error.TestFailed;
            }
            const wa_host = try download(gpu, wa); defer gpu.gpa.free(wa_host);
            const wb_host = try download(gpu, wb); defer gpu.gpa.free(wb_host);
            const wa_words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, wa_host));
            const wb_words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, wb_host));
            const record = 32 * 514;
            for (0..heads / 32) |group| for (0..rows) |r| for (0..32) |split| {
                const pos: usize = @intCast(positions_host[r]);
                const dense_row = pos < 2048;
                const local_keys = if (dense_row) (if (pos < rank) 0 else (pos - rank) / 2 + 1) else @as(usize, @intCast(counts_host[r]));
                const rec = ((group * rows + r) * 32 + split) * record;
                if (mla.splitBusy(local_keys, split)) {
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(wa_words[rec..][0..record]), std.mem.sliceAsBytes(wb_words[rec..][0..record]))) {
                        std.debug.print("FAIL glm53-decode-union: workspace record group={d} row={d} split={d}\n", .{ group, r, split });
                        return error.TestFailed;
                    }
                } else for (0..32) |h| {
                    if (wb_words[rec + h * 514 + 512] != 0xff800000 or wb_words[rec + h * 514 + 513] != 0) {
                        std.debug.print("FAIL glm53-decode-union: empty header group={d} row={d} split={d} head={d}\n", .{ group, r, split, h });
                        return error.TestFailed;
                    }
                }
            };
            for (0..rows) |r| for (0..heads / 32) |g| {
                const row_qa = try gpu.gpa.alloc(u16, 32 * mla.latent_dim); defer gpu.gpa.free(row_qa);
                const row_qp = try gpu.gpa.alloc(u16, 32 * mla.rope_dim); defer gpu.gpa.free(row_qp);
                for (0..32) |h| {
                    const src = ((g * rows + r) * 32 + h) * mla.latent_dim;
                    @memcpy(row_qa[h * mla.latent_dim ..][0..mla.latent_dim], qa_host[src..][0..mla.latent_dim]);
                    const qpsrc = ((g * rows + r) * 32 + h) * mla.rope_dim;
                    @memcpy(row_qp[h * mla.rope_dim ..][0..mla.rope_dim], qp_host[qpsrc..][0..mla.rope_dim]);
                }
                var dqa = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(row_qa)); defer dqa.free();
                var dqp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(row_qp)); defer dqp.free();
                var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions_host[r .. r + 1])); defer dpos.free();
                var dslots = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots_host[r * 2048 ..][0..2048])); defer dslots.free();
                var dcount = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(counts_host[r .. r + 1])); defer dcount.free();
                var alone = try cuda.DeviceBuffer.alloc(gpu.d, 32 * (mla.latent_dim * 2 + 4)); defer alone.free();
                var alone_ws = try cuda.DeviceBuffer.alloc(gpu.d, mla.splitWorkspaceBytes(1, 32)); defer alone_ws.free();
                try alone_ws.fill8(0xff, stream.handle);
                try kernels.attendVerify(stream, dqa.ptr, dqp.ptr, latent.ptr, rope.ptr, dpos.ptr, dslots.ptr, dcount.ptr, alone.ptr, 1, 32, rank, 2, alone_ws.ptr);
                try stream.synchronize();
                const alone_host = try download(gpu, alone); defer gpu.gpa.free(alone_host);
                const row_o_bytes = 32 * mla.latent_dim * 2;
                const group_bytes = rows * row_o_bytes + rows * 32 * 4;
                const batch_group = g * group_bytes;
                const batch_o = batch_group + r * row_o_bytes;
                if (!std.mem.eql(u8, a[batch_o..][0..row_o_bytes], alone_host[0..row_o_bytes])) {
                    std.debug.print("FAIL glm53-decode-union: case={s} rank={d} heads={d}: row-alone differs (group={d}, row={d}, head=0, dim=0)\n", .{ case_names[case_id], rank, heads, g, r });
                    return error.TestFailed;
                }
                const batch_lse = batch_group + rows * row_o_bytes + r * 32 * 4;
                if (!std.mem.eql(u8, a[batch_lse..][0 .. 32 * 4], alone_host[row_o_bytes..][0 .. 32 * 4])) {
                    std.debug.print("FAIL glm53-decode-union: case={s} rank={d} heads={d}: row-alone LSE differs (group={d}, row={d})\n", .{ case_names[case_id], rank, heads, g, r });
                    return error.TestFailed;
                }
            };
            if (case_id == 3 and heads == 64 and rank == 0) {
                var start = try cuda.Event.init(gpu.d, true); defer start.deinit();
                var stop = try cuda.Event.init(gpu.d, true); defer stop.deinit();
                for (0..3) |_| try kernels.attend(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, null, positions.ptr, tokens.ptr, slots.ptr, global_counts.ptr, counts.ptr, dense.ptr, pa.ptr, rows, heads, rank, 2, false, wa.ptr);
                try start.record(stream);
                for (0..20) |_| try kernels.attend(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, null, positions.ptr, tokens.ptr, slots.ptr, global_counts.ptr, counts.ptr, dense.ptr, pa.ptr, rows, heads, rank, 2, false, wa.ptr);
                try stop.record(stream); try stream.synchronize();
                bench_mma[rows - 1] = (try start.elapsedMs(stop)) * 50;
                for (0..3) |_| try kernels.attendVerify(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, positions.ptr, slots.ptr, counts.ptr, pb.ptr, rows, heads, rank, 2, wb.ptr);
                try start.record(stream);
                for (0..20) |_| try kernels.attendVerify(stream, qa.ptr, qp.ptr, latent.ptr, rope.ptr, positions.ptr, slots.ptr, counts.ptr, pb.ptr, rows, heads, rank, 2, wb.ptr);
                try stop.record(stream); try stream.synchronize();
                bench_verify[rows - 1] = (try start.elapsedMs(stop)) * 50;
                std.debug.print("RESULT glm53-decode-union rows={d} attend_mma_us={d:.3} verify_us={d:.3}\n", .{ rows, bench_mma[rows - 1], bench_verify[rows - 1] });
            }
        if (rows == 8) pass("glm53-decode-union: case={s} rank={d} heads={d} rows 1..8: verify == attend_mma bytes (batch, alone, workspace)", .{ case_names[case_id], rank, heads });
        if (rows == 16) pass("glm53-decode-union: case={s} rank={d} heads={d} rows 9..16: verify == attend_mma bytes (batch, alone, workspace)", .{ case_names[case_id], rank, heads });
        }
    };
    std.debug.print("RESULT glm53-decode-union per-extra-row-slope attend_mma_us={d:.3} verify_us={d:.3}\n", .{ (bench_mma[7] - bench_mma[0]) / 7, (bench_verify[7] - bench_verify[0]) / 7 });
    std.debug.print("RESULT glm53-decode-union per-extra-row-slope-9-16 attend_mma_us={d:.3} verify_us={d:.3}\n", .{ (bench_mma[15] - bench_mma[8]) / 7, (bench_verify[15] - bench_verify[8]) / 7 });
}

fn kvWriteCase(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    const rows = 3;
    const stride = mla.latent_dim + mla.rope_dim;
    const proj = try gpu.gpa.alloc(u16, rows * stride);
    defer gpu.gpa.free(proj);
    const projf = try gpu.gpa.alloc(f32, proj.len);
    defer gpu.gpa.free(projf);
    const norm = try gpu.gpa.alloc(u16, mla.latent_dim);
    defer gpu.gpa.free(norm);
    const normf = try gpu.gpa.alloc(f32, norm.len);
    defer gpu.gpa.free(normf);
    const cos = try gpu.gpa.alloc(f32, rows * 32);
    defer gpu.gpa.free(cos);
    const sin = try gpu.gpa.alloc(f32, cos.len);
    defer gpu.gpa.free(sin);
    for (proj, 0..) |*v, i| {
        v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 64);
        projf[i] = bf16Value(v.*);
    }
    for (norm, 0..) |*v, i| {
        v.* = bf16Bits(1 + @as(f32, @floatFromInt(i % 3)) / 128);
        normf[i] = bf16Value(v.*);
    }
    for (cos, 0..) |*v, i| {
        const phase = @as(f32, @floatFromInt(i % 32)) / 23;
        v.* = @cos(phase);
        sin[i] = @sin(phase);
    }
    const positions: [rows]i64 = .{ 0, 1, 2 };
    var latent = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.latent_dim * 2);
    defer latent.free();
    var rope = try cuda.DeviceBuffer.alloc(gpu.d, rows * mla.rope_dim * 2);
    defer rope.free();
    var dproj = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(proj));
    defer dproj.free();
    var dnorm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(norm));
    defer dnorm.free();
    var dcos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cos));
    defer dcos.free();
    var dsin = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sin));
    defer dsin.free();
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&positions));
    defer dpos.free();
    try kernels.prep(stream, dproj.ptr, dnorm.ptr, dcos.ptr, dsin.ptr, dpos.ptr, latent.ptr, rope.ptr, null, rows, stride, 0, 1, false);
    try stream.synchronize();
    const got_latent = try download(gpu, latent);
    defer gpu.gpa.free(got_latent);
    const got_rope = try download(gpu, rope);
    defer gpu.gpa.free(got_rope);
    const want_latent = try gpu.gpa.alloc(f32, rows * mla.latent_dim);
    defer gpu.gpa.free(want_latent);
    const want_rope = try gpu.gpa.alloc(f32, rows * mla.rope_dim);
    defer gpu.gpa.free(want_rope);
    var pos: [rows]usize = .{ 0, 1, 2 };
    try mla.kvWriteRef(projf, rows, stride, normf, &pos, cos, sin, want_latent, want_rope);
    try expectBf16(got_latent, want_latent, "kv_write latent");
    try expectBf16(got_rope, want_rope, "kv_write rope");
    pass("glm53-mla: kv cache write normalization and RoPE match BF16 host oracle", .{});
}

fn absorbExpandCases(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    const rows = 1;
    const kvb_rows = mla.kvb_rows_per_head;
    const codes = try gpu.gpa.alloc(u8, kvb_rows * mla.latent_dim);
    defer gpu.gpa.free(codes);
    var prng = std.Random.DefaultPrng.init(0xdec1_a004);
    const random = prng.random();
    for (codes) |*v| {
        v.* = random.int(u8);
        if ((v.* & 0x7f) == 0x7f) v.* = (v.* & 0x80) | 0x7e;
    }
    const scales = try gpu.gpa.alloc(f32, ((kvb_rows + 127) / 128) * 4);
    defer gpu.gpa.free(scales);
    for (scales) |*v| v.* = 0.01 + random.float(f32) * 0.2;
    const q = try gpu.gpa.alloc(u16, mla.nope_dim + mla.rope_dim);
    defer gpu.gpa.free(q);
    const qf = try gpu.gpa.alloc(f32, q.len);
    defer gpu.gpa.free(qf);
    for (q, 0..) |*v, i| {
        const sign: u16 = if (random.boolean()) 0x8000 else 0;
        const exponent: u16 = 96 + random.uintLessThan(u16, 64);
        v.* = sign | (exponent << 7) | random.uintLessThan(u16, 128);
        qf[i] = bf16Value(v.*);
    }
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dq.free();
    var dcodes = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dcodes.free();
    var dscales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer dscales.free();
    var qa = try cuda.DeviceBuffer.alloc(gpu.d, mla.latent_dim * 2);
    defer qa.free();
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, qa.ptr, rows, 1, 0, false);
    try stream.synchronize();
    const got_qa = try download(gpu, qa);
    defer gpu.gpa.free(got_qa);
    var want_qa: [mla.latent_dim]f32 = undefined;
    try mla.absorbRef(qf, rows, 1, codes, scales, mla.latent_dim, &want_qa);
    try expectBf16(got_qa, &want_qa, "absorb");

    var ol: [mla.latent_dim]u16 = undefined;
    var olf: [mla.latent_dim]f32 = undefined;
    for (&ol, 0..) |*v, i| {
        const sign: u16 = if (random.boolean()) 0x8000 else 0;
        const exponent: u16 = 96 + random.uintLessThan(u16, 64);
        v.* = sign | (exponent << 7) | random.uintLessThan(u16, 128);
        olf[i] = bf16Value(v.*);
    }
    var dol = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ol));
    defer dol.free();
    var expanded = try cuda.DeviceBuffer.alloc(gpu.d, mla.value_dim * 2);
    defer expanded.free();
    try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, expanded.ptr, 1, 1, 0, false);
    try stream.synchronize();
    const got_v = try download(gpu, expanded);
    defer gpu.gpa.free(got_v);
    var want_v: [mla.value_dim]f32 = undefined;
    try mla.expandVRef(gpu.gpa, &olf, codes, scales, 0, &want_v);
    try expectBf16(got_v, &want_v, "expand_v");
    pass("glm53-mla: absorb and expand_v match host FP8/BF16 references on random full-mantissa inputs", .{});
}

fn decodeAbsorbExpandCases(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    const rows = 8;
    const local_heads = 32;
    const global_heads = 64;
    const kvb_count = mla.kvb_rows_per_head * mla.latent_dim;
    const codes = try gpu.gpa.alloc(u8, kvb_count * global_heads);
    defer gpu.gpa.free(codes);
    var prng = std.Random.DefaultPrng.init(0xdec1_a004);
    const random = prng.random();
    for (codes) |*v| {
        v.* = random.int(u8);
        if ((v.* & 0x7f) == 0x7f) v.* = (v.* & 0x80) | 0x7e;
    }
    const scale_count = ((mla.kvb_rows_per_head * global_heads + 127) / 128) * 4;
    const scales = try gpu.gpa.alloc(f32, scale_count);
    defer gpu.gpa.free(scales);
    for (scales) |*v| v.* = 0.01 + random.float(f32) * 0.2;
    const q_count = rows * local_heads * 256;
    const q = try gpu.gpa.alloc(u16, q_count);
    defer gpu.gpa.free(q);
    const ol = try gpu.gpa.alloc(u16, rows * local_heads * mla.latent_dim);
    defer gpu.gpa.free(ol);
    for (q) |*v| {
        const sign: u16 = if (random.boolean()) 0x8000 else 0;
        const exponent: u16 = 96 + random.uintLessThan(u16, 64);
        v.* = sign | (exponent << 7) | random.uintLessThan(u16, 128);
    }
    for (ol) |*v| {
        const sign: u16 = if (random.boolean()) 0x8000 else 0;
        const exponent: u16 = 96 + random.uintLessThan(u16, 64);
        v.* = sign | (exponent << 7) | random.uintLessThan(u16, 128);
    }

    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dq.free();
    var dol = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ol));
    defer dol.free();
    var dcodes = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dcodes.free();
    var dscales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer dscales.free();
    const qa_bytes = rows * local_heads * mla.latent_dim * 2;
    const vo_bytes = rows * local_heads * mla.value_dim * 2;
    var scalar_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer scalar_qa.free();
    var decode_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer decode_qa.free();
    var repeat_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer repeat_qa.free();
    var scalar_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer scalar_v.free();
    var decode_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer decode_v.free();
    var repeat_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer repeat_v.free();
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, scalar_qa.ptr, rows, local_heads, 0, false);
    try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, scalar_v.ptr, rows, local_heads, 0, false);
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, decode_qa.ptr, rows, local_heads, 0, true);
    try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, decode_v.ptr, rows, local_heads, 0, true);
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, repeat_qa.ptr, rows, local_heads, 0, true);
    try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, repeat_v.ptr, rows, local_heads, 0, true);
    try stream.synchronize();
    const want_qa = try download(gpu, scalar_qa);
    defer gpu.gpa.free(want_qa);
    const got_qa = try download(gpu, decode_qa);
    defer gpu.gpa.free(got_qa);
    const again_qa = try download(gpu, repeat_qa);
    defer gpu.gpa.free(again_qa);
    const want_v = try download(gpu, scalar_v);
    defer gpu.gpa.free(want_v);
    const got_v = try download(gpu, decode_v);
    defer gpu.gpa.free(got_v);
    const again_v = try download(gpu, repeat_v);
    defer gpu.gpa.free(again_v);
    try std.testing.expectEqualSlices(u8, want_qa, got_qa);
    try std.testing.expectEqualSlices(u8, want_v, got_v);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got_qa), std.mem.sliceAsBytes(again_qa));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got_v), std.mem.sliceAsBytes(again_v));

    var graph_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer graph_qa.free();
    var graph_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer graph_v.free();
    var direct_v2_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer direct_v2_qa.free();
    var direct_v3_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer direct_v3_qa.free();
    var direct_v2_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer direct_v2_v.free();
    var graph_v2_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer graph_v2_qa.free();
    var graph_v3_qa = try cuda.DeviceBuffer.alloc(gpu.d, qa_bytes);
    defer graph_v3_qa.free();
    var graph_v2_v = try cuda.DeviceBuffer.alloc(gpu.d, vo_bytes);
    defer graph_v2_v.free();
    try kernels.absorbV2(stream, dq.ptr, dcodes.ptr, dscales.ptr, direct_v2_qa.ptr, rows, local_heads, 0);
    try kernels.absorbV3(stream, dq.ptr, dcodes.ptr, dscales.ptr, direct_v3_qa.ptr, rows, local_heads, 0);
    try kernels.expandVV2(stream, dol.ptr, dcodes.ptr, dscales.ptr, direct_v2_v.ptr, rows, local_heads, 0);
    try stream.synchronize();
    try cuda.graph.beginCapture(stream, .thread_local);
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, graph_qa.ptr, rows, local_heads, 0, true);
    try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, graph_v.ptr, rows, local_heads, 0, true);
    try kernels.absorbV2(stream, dq.ptr, dcodes.ptr, dscales.ptr, graph_v2_qa.ptr, rows, local_heads, 0);
    try kernels.absorbV3(stream, dq.ptr, dcodes.ptr, dscales.ptr, graph_v3_qa.ptr, rows, local_heads, 0);
    try kernels.expandVV2(stream, dol.ptr, dcodes.ptr, dscales.ptr, graph_v2_v.ptr, rows, local_heads, 0);
    var captured = try cuda.graph.endCapture(stream);
    defer captured.deinit();
    var graph_exec = try captured.instantiate();
    defer graph_exec.deinit();
    try graph_exec.launchOn(stream);
    try stream.synchronize();
    const graph_got_qa = try download(gpu, graph_qa);
    defer gpu.gpa.free(graph_got_qa);
    const graph_got_v = try download(gpu, graph_v);
    defer gpu.gpa.free(graph_got_v);
    const direct_v2_got_qa = try download(gpu, direct_v2_qa);
    defer gpu.gpa.free(direct_v2_got_qa);
    const direct_v2_got_v = try download(gpu, direct_v2_v);
    defer gpu.gpa.free(direct_v2_got_v);
    const graph_v2_got_qa = try download(gpu, graph_v2_qa);
    defer gpu.gpa.free(graph_v2_got_qa);
    const direct_v3_got_qa = try download(gpu, direct_v3_qa);
    defer gpu.gpa.free(direct_v3_got_qa);
    const graph_v3_got_qa = try download(gpu, graph_v3_qa);
    defer gpu.gpa.free(graph_v3_got_qa);
    const graph_v2_got_v = try download(gpu, graph_v2_v);
    defer gpu.gpa.free(graph_v2_got_v);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got_qa), std.mem.sliceAsBytes(graph_got_qa));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got_v), std.mem.sliceAsBytes(graph_got_v));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_v2_got_qa), std.mem.sliceAsBytes(graph_v2_got_qa));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got_qa), std.mem.sliceAsBytes(direct_v3_got_qa));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_v3_got_qa), std.mem.sliceAsBytes(graph_v3_got_qa));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_v2_got_v), std.mem.sliceAsBytes(graph_v2_got_v));

    for ([_]usize{ 1, 2, 3, 5, 8 }) |n| {
        var out_qa = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.latent_dim * 2);
        defer out_qa.free();
        var out_v = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.value_dim * 2);
        defer out_v.free();
        var scalar_rows_qa = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.latent_dim * 2);
        defer scalar_rows_qa.free();
        var scalar_rows_v = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.value_dim * 2);
        defer scalar_rows_v.free();
        var v2_rows_qa = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.latent_dim * 2);
        defer v2_rows_qa.free();
        var v3_rows_qa = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.latent_dim * 2);
        defer v3_rows_qa.free();
        var v2_rows_v = try cuda.DeviceBuffer.alloc(gpu.d, n * local_heads * mla.value_dim * 2);
        defer v2_rows_v.free();
        try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, out_qa.ptr, n, local_heads, 0, true);
        try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, out_v.ptr, n, local_heads, 0, true);
        try kernels.absorbV2(stream, dq.ptr, dcodes.ptr, dscales.ptr, v2_rows_qa.ptr, n, local_heads, 0);
        try kernels.absorbV3(stream, dq.ptr, dcodes.ptr, dscales.ptr, v3_rows_qa.ptr, n, local_heads, 0);
        try kernels.expandVV2(stream, dol.ptr, dcodes.ptr, dscales.ptr, v2_rows_v.ptr, n, local_heads, 0);
        try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, scalar_rows_qa.ptr, n, local_heads, 0, false);
        try kernels.expandV(stream, dol.ptr, dcodes.ptr, dscales.ptr, scalar_rows_v.ptr, n, local_heads, 0, false);
        try stream.synchronize();
        const batch_qa = try download(gpu, out_qa);
        defer gpu.gpa.free(batch_qa);
        const batch_v = try download(gpu, out_v);
        defer gpu.gpa.free(batch_v);
        const scalar_qa_rows = try download(gpu, scalar_rows_qa);
        defer gpu.gpa.free(scalar_qa_rows);
        const scalar_v_rows = try download(gpu, scalar_rows_v);
        defer gpu.gpa.free(scalar_v_rows);
        try std.testing.expectEqualSlices(u8, scalar_qa_rows, batch_qa);
        try std.testing.expectEqualSlices(u8, scalar_v_rows, batch_v);
        const batch_v2_qa = try download(gpu, v2_rows_qa);
        defer gpu.gpa.free(batch_v2_qa);
        const batch_v3_qa = try download(gpu, v3_rows_qa);
        defer gpu.gpa.free(batch_v3_qa);
        const batch_v2_v = try download(gpu, v2_rows_v);
        defer gpu.gpa.free(batch_v2_v);
        try std.testing.expectEqualSlices(u8, scalar_qa_rows, batch_v2_qa);
        try expectDecodeAbsorbBytes(batch_qa, batch_v3_qa, n, local_heads, "v1/head0=0");
        try expectDecodeAbsorbBytes(scalar_qa_rows, batch_v3_qa, n, local_heads, "head0=0");
        try std.testing.expectEqualSlices(u8, scalar_v_rows, batch_v2_v);
        try std.testing.expectEqualSlices(u8, got_qa[0 .. n * local_heads * mla.latent_dim * @sizeOf(u16)], batch_qa);
        try std.testing.expectEqualSlices(u8, got_v[0 .. n * local_heads * mla.value_dim * @sizeOf(u16)], batch_v);
    }
    // Exercise the odd absolute head offset, where kvb() crosses three 128-row scale blocks.
    var head1_v1 = try cuda.DeviceBuffer.alloc(gpu.d, 3 * local_heads * mla.latent_dim * 2);
    defer head1_v1.free();
    var head1_v3 = try cuda.DeviceBuffer.alloc(gpu.d, 3 * local_heads * mla.latent_dim * 2);
    defer head1_v3.free();
    try kernels.absorb(stream, dq.ptr, dcodes.ptr, dscales.ptr, head1_v1.ptr, 3, local_heads, 1, true);
    try kernels.absorbV3(stream, dq.ptr, dcodes.ptr, dscales.ptr, head1_v3.ptr, 3, local_heads, 1);
    try stream.synchronize();
    const head1_v1_bytes = try download(gpu, head1_v1);
    defer gpu.gpa.free(head1_v1_bytes);
    const head1_v3_bytes = try download(gpu, head1_v3);
    defer gpu.gpa.free(head1_v3_bytes);
    try expectDecodeAbsorbBytes(head1_v1_bytes, head1_v3_bytes, 3, local_heads, "head0=1");
    pass("glm53-mla: decode absorb v1/v2/v3 and expand_v v1/v2 are bit-identical to scalar at rows 1/2/3/5/8, 32 heads", .{});
    for ([_]usize{ 1, 8 }) |n| {
        const absorb_us = try decodeMlaBench(gpu, stream, kernels, dq.ptr, dcodes.ptr, dscales.ptr, 0, n, true);
        const expand_us = try decodeMlaBench(gpu, stream, kernels, dol.ptr, dcodes.ptr, dscales.ptr, 0, n, false);
        std.debug.print("BENCH decode_mla rows={d} absorb scalar={d:.2}us v1={d:.2}us v2={d:.2}us v3={d:.2}us | expand_v scalar={d:.2}us v1={d:.2}us v2={d:.2}us\n", .{ n, absorb_us[0], absorb_us[1], absorb_us[2], absorb_us[3], expand_us[0], expand_us[1], expand_us[2] });
    }
}

fn expectDecodeAbsorbBytes(want: []const u8, got: []const u8, rows: usize, local_heads: usize, label: []const u8) !void {
    if (want.len != got.len) return error.DecodeMlaOutputSize;
    for (0..want.len / 2) |i| {
        const expected = std.mem.readInt(u16, want[i * 2 ..][0..2], .little);
        const actual = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
        if (expected != actual) {
            const c = i % mla.latent_dim;
            const h = (i / mla.latent_dim) % local_heads;
            const r = i / (mla.latent_dim * local_heads);
            if (r >= rows) return error.DecodeMlaOutputShape;
            std.debug.print("FAIL decode_mla absorb_v3 {s}: first differing (r={d}, h={d}, c={d}) v1=0x{x:0>4} v3=0x{x:0>4}\n", .{ label, r, h, c, expected, actual });
            return error.DecodeMlaMismatch;
        }
    }
}

fn decodeMlaBench(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, input: u64, codes: u64, scales: u64, head0: usize, rows: usize, absorb: bool) ![4]f32 {
    const width: usize = if (absorb) mla.latent_dim else mla.value_dim;
    var output = try cuda.DeviceBuffer.alloc(gpu.d, rows * 32 * width * 2);
    defer output.free();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    var times: [4]f32 = .{ 0, 0, 0, 0 };
    const variants: usize = if (absorb) 4 else 3;
    for (0..variants) |variant| {
        for (0..24) |rep| {
            if (rep == 3) try start.record(stream);
            if (absorb) {
                if (variant == 0) {
                    try kernels.absorb(stream, input, codes, scales, output.ptr, rows, 32, head0, false);
                } else if (variant == 1) {
                    try kernels.absorb(stream, input, codes, scales, output.ptr, rows, 32, head0, true);
                } else if (variant == 2) {
                    try kernels.absorbV2(stream, input, codes, scales, output.ptr, rows, 32, head0);
                } else {
                    try kernels.absorbV3(stream, input, codes, scales, output.ptr, rows, 32, head0);
                }
            } else {
                if (variant == 0) {
                    try kernels.expandV(stream, input, codes, scales, output.ptr, rows, 32, head0, false);
                } else if (variant == 1) {
                    try kernels.expandV(stream, input, codes, scales, output.ptr, rows, 32, head0, true);
                } else {
                    try kernels.expandVV2(stream, input, codes, scales, output.ptr, rows, 32, head0);
                }
            }
        }
        try stop.record(stream);
        try stream.synchronize();
        times[variant] = (try start.elapsedMs(stop)) * 1000 / 21;
    }
    return times;
}

fn qpeCase(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    const rows = 3;
    const heads = 1;
    const q = try gpu.gpa.alloc(u16, rows * heads * 256);
    defer gpu.gpa.free(q);
    const cos = try gpu.gpa.alloc(f32, rows * mla.rope_dim / 2);
    defer gpu.gpa.free(cos);
    const sin = try gpu.gpa.alloc(f32, cos.len);
    defer gpu.gpa.free(sin);
    const want = try gpu.gpa.alloc(u16, rows * heads * mla.rope_dim);
    defer gpu.gpa.free(want);
    for (q, 0..) |*v, i| v.* = bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16);
    for (cos, 0..) |*v, i| {
        const phase = @as(f32, @floatFromInt(i % 32)) / 17;
        v.* = @cos(phase);
        sin[i] = @sin(phase);
    }
    for (0..rows) |r| for (0..mla.rope_dim / 2) |i| {
        const base = r * 256 + mla.nope_dim + 2 * i;
        const pair = mla.ropePair(bf16Value(q[base]), bf16Value(q[base + 1]), cos[r * 32 + i], sin[r * 32 + i]);
        want[r * 64 + 2 * i] = bf16Bits(mla.bf16(pair[0]));
        want[r * 64 + 2 * i + 1] = bf16Bits(mla.bf16(pair[1]));
    };
    var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dq.free();
    var dc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cos));
    defer dc.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sin));
    defer ds.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * 64 * 2);
    defer out.free();
    try kernels.packQpe(stream, dq.ptr, dc.ptr, ds.ptr, out.ptr, rows, heads);
    try stream.synchronize();
    const got = try download(gpu, out);
    defer gpu.gpa.free(got);
    try expect(std.mem.eql(u8, got, std.mem.sliceAsBytes(want)), "q_pe pack BF16 output", .{});
    for (0..rows) |r| {
        var one = try cuda.DeviceBuffer.alloc(gpu.d, 64 * 2);
        defer one.free();
        try kernels.packQpe(stream, dq.ptr + r * 256 * 2, dc.ptr + r * 32 * 4, ds.ptr + r * 32 * 4, one.ptr, 1, heads);
        try stream.synchronize();
        const alone = try download(gpu, one);
        defer gpu.gpa.free(alone);
        try expect(std.mem.eql(u8, alone, got[r * 128 ..][0..128]), "q_pe row invariance row={d}", .{r});
    }
    pass("glm53-mla: q_pe RoPE pack matches BF16 host oracle and row-invariance", .{});
}

fn mergeCase(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    const width = mla.latent_dim;
    const Partial = extern struct { o: [width]u16, lse: f32 };
    var local: Partial = undefined;
    var peer: Partial = undefined;
    var expected: [width]f32 = undefined;
    var local_values: [width]f32 = undefined;
    var peer_values: [width]f32 = undefined;
    for (0..width) |i| {
        local_values[i] = mla.bf16(@as(f32, @floatFromInt(i % 29)) / 8);
        peer_values[i] = mla.bf16(@as(f32, @floatFromInt(i % 13)) / 4);
        local.o[i] = @truncate(@as(u32, @bitCast(local_values[i])) >> 16);
        peer.o[i] = @truncate(@as(u32, @bitCast(peer_values[i])) >> 16);
    }
    local.lse = -std.math.inf(f32);
    peer.lse = 1.5;
    const want_lse = try mla.mergePeerRef(0, -std.math.inf(f32), 1.5, &local_values, &peer_values, &expected);
    var dl = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&local));
    defer dl.free();
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&peer));
    defer dp.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, width * 2);
    defer dout.free();
    var dlse = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer dlse.free();
    try kernels.merge(stream, dl.ptr, dp.ptr, dout.ptr, dlse.ptr, 1, 1, 0);
    try stream.synchronize();
    const got = try download(gpu, dout);
    defer gpu.gpa.free(got);
    const got_lse = try download(gpu, dlse);
    defer gpu.gpa.free(got_lse);
    const want_bf16 = try gpu.gpa.alloc(u16, width);
    defer gpu.gpa.free(want_bf16);
    for (expected, want_bf16) |v, *out| out.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    try expect(std.mem.eql(u8, got, std.mem.sliceAsBytes(want_bf16)), "DCP merge BF16 output", .{});
    const actual_lse = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, got_lse))[0];
    try expect(@abs(actual_lse - want_lse) < 2e-6, "DCP merge LSE", .{});
    pass("glm53-mla: local/peer contiguous partial merge handles -inf empty LSE", .{});
}

fn attendCases(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla) !void {
    var scalar = kernels.*;
    scalar.attend_kernel = .scalar;
    var tc = kernels.*;
    tc.attend_kernel = .mma;
    const Shape = struct { rows: usize, prefill: bool, full: bool };
    for ([_]Shape{ .{ .rows = 1, .prefill = false, .full = false }, .{ .rows = 3, .prefill = false, .full = false }, .{ .rows = 8, .prefill = false, .full = false }, .{ .rows = 64, .prefill = true, .full = false }, .{ .rows = 1, .prefill = false, .full = true }, .{ .rows = 8, .prefill = false, .full = true }, .{ .rows = 64, .prefill = true, .full = true } }) |c| {
        const a = try attendCase(gpu, stream, &scalar, c.rows, c.prefill, c.full, false);
        defer gpu.gpa.free(a);
        const b = try attendCase(gpu, stream, &tc, c.rows, c.prefill, c.full, false);
        defer gpu.gpa.free(b);
        try compareAttend(a, b, c.rows, c.prefill, c.full);
        if (!c.prefill) { // decode key splits (the forward's decode path) against the scalar kernel
            const z = try attendCase(gpu, stream, &tc, c.rows, c.prefill, c.full, true);
            defer gpu.gpa.free(z);
            try compareAttend(a, z, c.rows, c.prefill, c.full);
        }
    }
}

/// The tensor-core attend against the scalar one: BF16 outputs within 2 bf16 ulps of the row-head max + 1e-3, LSE 1e-3.
fn compareAttend(scalar: []const u8, tc: []const u8, rows: usize, prefill: bool, full: bool) !void {
    const o_bytes = rows * 32 * mla.latent_dim * 2;
    const so = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, scalar[0..o_bytes]));
    const to = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, tc[0..o_bytes]));
    var worst: f64 = 0;
    for (0..rows * 32) |rh| {
        var scale: f64 = 0;
        for (0..mla.latent_dim) |d| scale = @max(scale, @abs(@as(f64, bf16Value(so[rh * mla.latent_dim + d]))));
        const tol = 2.0 * scale / 128.0 + 1e-3;
        for (0..mla.latent_dim) |d| {
            const diff = @abs(@as(f64, bf16Value(so[rh * mla.latent_dim + d])) - bf16Value(to[rh * mla.latent_dim + d]));
            worst = @max(worst, diff / tol);
            if (!(diff <= tol)) {
                std.debug.print("FAIL glm53-mla mma vs scalar rows={d} prefill={} full={}: row-head {d} dim {d} scalar {d} mma {d} tol {d}\n", .{ rows, prefill, full, rh, d, bf16Value(so[rh * mla.latent_dim + d]), bf16Value(to[rh * mla.latent_dim + d]), tol });
                return error.TestFailed;
            }
        }
    }
    const sl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, scalar[o_bytes..][0 .. rows * 32 * 4]));
    const tl = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, tc[o_bytes..][0 .. rows * 32 * 4]));
    var lse_worst: f64 = 0;
    for (sl, tl) |a, b| {
        if (std.math.isInf(a) or std.math.isInf(b)) {
            if (a != b) return error.TestFailed;
            continue;
        }
        lse_worst = @max(lse_worst, @abs(@as(f64, a) - b));
    }
    if (lse_worst > 1e-3) {
        std.debug.print("FAIL glm53-mla mma vs scalar rows={d} prefill={} full={}: LSE max abs {d}\n", .{ rows, prefill, full, lse_worst });
        return error.TestFailed;
    }
    pass("glm53-mla: mma attend vs scalar rows={d} prefill={} keys={s}: worst {d:.3} of tolerance, LSE max abs {e:.2}", .{ rows, prefill, if (full) "2048" else "few", worst, lse_worst });
}

fn attendCase(gpu: anytype, stream: cuda.Stream, kernels: *const mla.Mla, rows: usize, prefill: bool, full: bool, split: bool) ![]u8 {
    const cache_rows = 2048;
    const token_cols = 2048;
    const head_row = mla.latent_dim * 2 + 4;
    const partial_bytes = rows * 32 * head_row;
    var seed: u32 = @intCast(0x53a100 + rows + @as(usize, @intFromBool(prefill)));
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
    fillBf16(q, &seed);
    fillBf16(qp, &seed);
    fillBf16(latent, &seed);
    fillBf16(rope, &seed);
    fillBf16(stage_latent, &seed);
    fillBf16(stage_rope, &seed);
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
    for (0..rows) |r| {
        // full: dense rows near 2047 (prefill) and sparse rows over 2048 selected keys (every slot of this rank's plane)
        const pos: usize = if (full) (if (prefill and r < rows / 2) 2047 - r else 2048 + 2 * r + 1000) else if (prefill and r >= rows / 2) 2048 + r - rows / 2 else r + 16;
        positions[r] = @intCast(pos);
        dense_len[r] = if (pos < 2048) @intCast(pos + 1) else 0;
        if (full and pos >= 2048) {
            // 2048 ascending keys below pos: every other position (prefill reads both planes, decode this rank's slots)
            var n: usize = 0;
            var key: usize = pos % 2;
            while (n < token_cols and key <= pos and key / 2 < cache_rows) : (key += 2) {
                tokens[r * token_cols + n] = @intCast(key);
                slots[r * token_cols + n] = @intCast(key / 2);
                n += 1;
            }
            global_counts[r] = @intCast(n);
            local_counts[r] = @intCast(n);
            continue;
        }
        global_counts[r] = if (prefill and pos >= 2048) 3 else 0;
        local_counts[r] = if (pos >= 2048) 3 else 0;
        if (pos >= 2048) for ([_]usize{ 0, pos / 2, pos }, 0..) |key, j| {
            tokens[r * token_cols + j] = @intCast(key);
            if (key % 2 == 0) slots[r * token_cols + j] = @intCast(key / 2);
        };
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
    var dpartial = try cuda.DeviceBuffer.alloc(gpu.d, partial_bytes);
    defer dpartial.free();
    var dsplit = try cuda.DeviceBuffer.alloc(gpu.d, @max(mla.splitWorkspaceBytes(rows, 32), 4));
    defer dsplit.free();
    var dsplit_one = try cuda.DeviceBuffer.alloc(gpu.d, @max(mla.splitWorkspaceBytes(1, 32), 4));
    defer dsplit_one.free();
    const split_ws: u64 = if (split) dsplit.ptr else 0;
    const split_alone: u64 = if (split) dsplit_one.ptr else 0;
    const stage: ?types.StagePlanePair = if (prefill) .{
        .latent = .{ .data = dstage_latent, .dtype = .bf16, .rows = cache_rows, .cols = mla.latent_dim, .stride = mla.latent_dim },
        .rope = .{ .data = dstage_rope, .dtype = .bf16, .rows = cache_rows, .cols = mla.rope_dim, .stride = mla.rope_dim },
    } else null;
    try kernels.attend(stream, dqa.ptr, dqpe.ptr, dlatent.ptr, drope.ptr, stage, dpositions.ptr, dtokens.ptr, dslots.ptr, dglobal_counts.ptr, dlocal_counts.ptr, ddense_len.ptr, dpartial.ptr, rows, 32, 0, 2, prefill, split_ws);
    try stream.synchronize();
    const batch = try download(gpu, dpartial);
    errdefer gpu.gpa.free(batch);
    for (0..rows) |r| {
        var alone = try cuda.DeviceBuffer.alloc(gpu.d, 32 * head_row);
        defer alone.free();
        const token_offset = r * token_cols * @sizeOf(i32);
        const head_offset = r * 32 * mla.latent_dim * @sizeOf(u16);
        try kernels.attend(stream, dqa.ptr + head_offset, dqpe.ptr + head_offset / mla.latent_dim * mla.rope_dim, dlatent.ptr, drope.ptr, stage, dpositions.ptr + r * @sizeOf(i64), dtokens.ptr + token_offset, dslots.ptr + token_offset, dglobal_counts.ptr + r * @sizeOf(i32), dlocal_counts.ptr + r * @sizeOf(i32), ddense_len.ptr + r * @sizeOf(i32), alone.ptr, 1, 32, 0, 2, prefill, split_alone);
        try stream.synchronize();
        const one = try download(gpu, alone);
        defer gpu.gpa.free(one);
        // K4 stores the full [rows][heads][512] BF16 plane followed by the
        // [rows][heads] FP32 LSE plane. Compare both planes using their own
        // row strides; a per-head interleaved row stride is not the ABI.
        const batch_o_row = r * 32 * mla.latent_dim * @sizeOf(u16);
        const batch_lse_base = rows * 32 * mla.latent_dim * @sizeOf(u16) + r * 32 * @sizeOf(f32);
        const one_o_bytes = 32 * mla.latent_dim * @sizeOf(u16);
        const one_lse_base = one_o_bytes;
        const one_o_equal = std.mem.eql(u8, one[0..one_o_bytes], batch[batch_o_row..][0..one_o_bytes]);
        const one_lse_equal = std.mem.eql(u8, one[one_lse_base..][0 .. 32 * @sizeOf(f32)], batch[batch_lse_base..][0 .. 32 * @sizeOf(f32)]);
        if (!one_o_equal or !one_lse_equal) {
            var first: ?usize = null;
            var max_abs: f64 = 0;
            const alone_o = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, one[0..one_o_bytes]));
            const batch_o = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, batch[batch_o_row..][0..one_o_bytes]));
            for (alone_o, batch_o, 0..) |a, b, i| {
                if (a != b) {
                    if (first == null) first = i;
                    max_abs = @max(max_abs, @abs(@as(f64, bf16Value(a)) - bf16Value(b)));
                }
            }
            const alone_lse = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, one[one_lse_base..][0 .. 32 * @sizeOf(f32)]));
            const batch_lse = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, batch[batch_lse_base..][0 .. 32 * @sizeOf(f32)]));
            var first_lse: ?usize = null;
            var max_abs_lse: f64 = 0;
            for (alone_lse, batch_lse, 0..) |a, b, i| {
                if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) {
                    if (first_lse == null) first_lse = i;
                    max_abs_lse = @max(max_abs_lse, @abs(@as(f64, a) - b));
                }
            }
            std.debug.print("MLA attend row mismatch row={d} first_o={any} max_abs_o={d} first_lse={any} max_abs_lse={d}\n", .{ r, first, max_abs, first_lse, max_abs_lse });
        }
        try expect(one_o_equal and one_lse_equal, "MLA attend row-alone vs batch rows={d} row={d} prefill={}", .{ rows, r, prefill });
    }
    pass("glm53-mla: attend ({t}{s}) row-invariant rows={d}, dense/sparse gather prefill={} keys={s}", .{ kernels.attend_kernel, if (split) ", key splits" else "", rows, prefill, if (full) "2048" else "few" });
    return batch;
}

fn fillBf16(values: []u16, seed: *u32) void {
    for (values) |*value| {
        seed.* = seed.* *% 1664525 +% 1013904223;
        const sample: i32 = @as(i32, @intCast((seed.* >> 16) % 129)) - 64;
        value.* = bf16Bits(@as(f32, @floatFromInt(sample)) / 512);
    }
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(bytes);
    try buffer.download(0, bytes);
    return bytes;
}

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) error{TestFailed}!void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}

fn expectBf16(got: []const u8, expected: []const f32, label: []const u8) !void {
    if (got.len != expected.len * 2) return error.BadShape;
    const actual = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got));
    for (actual, expected, 0..) |a, e, i| {
        try expect(a == bf16Bits(mla.bf16(e)), "{s} BF16 item={d}", .{ label, i });
    }
}

fn bf16Bits(value: f32) u16 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @truncate(bits >> 16);
}

fn bf16Value(value: u16) f32 {
    return @as(f32, @bitCast(@as(u32, value) << 16));
}

test "DCP merge host reference ignores empty -inf rank and uses rank order" {
    const local = [_]f32{ 2, -1, 0.25 };
    const peer = [_]f32{ -0.5, 4, 3 };
    var out: [3]f32 = undefined;
    const lse = try mla.mergePeerRef(0, -std.math.inf(f32), 3, &local, &peer, &out);
    try std.testing.expectEqual(@as(f32, 3), lse);
    for (out, peer) |a, b| try std.testing.expectEqual(mla.bf16(b), a);
}

/// `tf-cuda-test glm53-mla-bench`: scalar vs tensor-core attend, 32 heads, 2048 selected keys a row (sparse decode on
/// this rank's slots, and gather-on-prefill), 3 warm-ups then 20 timed launches.
pub fn bench(gpu: anytype) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_mla);
    defer module.unload();
    const base = try mla.Mla.load(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const cache_rows = 4096;
    const max_rows = 256;
    var seed: u32 = 0x5be4c4;
    const q = try gpu.gpa.alloc(u16, max_rows * 32 * mla.latent_dim);
    defer gpu.gpa.free(q);
    const qp = try gpu.gpa.alloc(u16, max_rows * 32 * mla.rope_dim);
    defer gpu.gpa.free(qp);
    const latent = try gpu.gpa.alloc(u16, cache_rows * mla.latent_dim);
    defer gpu.gpa.free(latent);
    const rope = try gpu.gpa.alloc(u16, cache_rows * mla.rope_dim);
    defer gpu.gpa.free(rope);
    fillBf16(q, &seed);
    fillBf16(qp, &seed);
    fillBf16(latent, &seed);
    fillBf16(rope, &seed);
    const positions = try gpu.gpa.alloc(i64, max_rows);
    defer gpu.gpa.free(positions);
    const ids = try gpu.gpa.alloc(i32, max_rows * 2048);
    defer gpu.gpa.free(ids);
    const counts = try gpu.gpa.alloc(i32, max_rows);
    defer gpu.gpa.free(counts);
    for (0..max_rows) |r| {
        positions[r] = @intCast(8000 + r);
        counts[r] = 2048;
        for (0..2048) |j| ids[r * 2048 + j] = @intCast((j * 3 + r) % 4000 / 2 * 2); // even positions: slots of rank 0
    }
    // decode reads slots (position / 2), prefill reads global positions through both planes (one plane here: rank 0 owns evens)
    const slots = try gpu.gpa.alloc(i32, max_rows * 2048);
    defer gpu.gpa.free(slots);
    for (ids, slots) |p, *sl| sl.* = @divTrunc(p, 2);
    var dqa = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
    defer dqa.free();
    var dqpe = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qp));
    defer dqpe.free();
    var dlatent = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(latent));
    defer dlatent.free();
    var drope = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rope));
    defer drope.free();
    var dpos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(positions));
    defer dpos.free();
    var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
    defer dids.free();
    var dslots = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots));
    defer dslots.free();
    var dcounts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(counts));
    defer dcounts.free();
    var dpartial = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * 32 * (mla.latent_dim * 2 + 4));
    defer dpartial.free();
    var dsplit = try cuda.DeviceBuffer.alloc(gpu.d, mla.splitWorkspaceBytes(8, 32));
    defer dsplit.free();
    const stage: types.StagePlanePair = .{
        .latent = .{ .data = dlatent, .dtype = .bf16, .rows = cache_rows, .cols = mla.latent_dim, .stride = mla.latent_dim },
        .rope = .{ .data = drope, .dtype = .bf16, .rows = cache_rows, .cols = mla.rope_dim, .stride = mla.rope_dim },
    };
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    const Case = struct { rows: usize, prefill: bool };
    for ([_]Case{ .{ .rows = 1, .prefill = false }, .{ .rows = 8, .prefill = false }, .{ .rows = 256, .prefill = true } }) |c| {
        for ([_]struct { mla.AttendKernel, bool }{ .{ .scalar, false }, .{ .mma, false }, .{ .mma, true } }) |variant| {
            const kind = variant[0];
            const split = variant[1];
            if (split and c.prefill) continue;
            var k = base;
            k.attend_kernel = kind;
            const st: ?types.StagePlanePair = if (c.prefill) stage else null;
            for (0..23) |i| {
                if (i == 3) try e0.record(stream);
                try k.attend(stream, dqa.ptr, dqpe.ptr, dlatent.ptr, drope.ptr, st, dpos.ptr, dids.ptr, dslots.ptr, dcounts.ptr, dcounts.ptr, dcounts.ptr, dpartial.ptr, c.rows, 32, 0, 2, c.prefill, if (split) dsplit.ptr else 0);
            }
            try e1.record(stream);
            try stream.synchronize();
            const us = (try e0.elapsedMs(e1)) * 1000.0 / 20.0;
            std.debug.print("BENCH attend {t}{s} rows={d} {s} keys=2048 heads=32: {d:.1} us/launch\n", .{ kind, if (split) " (key splits)" else "", c.rows, if (c.prefill) "prefill" else "decode", us });
        }
    }
}
