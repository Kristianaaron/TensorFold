// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! `glm53-gsum`: the rank-major gather + residual consumers against the one-block sum gather + `_sum` consumers.
//! Attention sums stay fp32 (one block of exact IEEE adds); MoE sums are one bf16 block.
//! A single GPU stands in for the peer: n_hca = 0 skips the peer wait, and the recv slot is pre-filled with the
//! shard the peer would have written. Both receive slots hold that shard because the epoch starts at zero and the
//! kernel's first sequence is 1 (slot 1); the next call uses slot 0.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const hidden = 6144;
const eps: f32 = 1e-6;
const rank = 0;
const timeout_ns: u64 = 1_000_000_000;
const max_blocks = 8;

const Kernels = struct {
    gather: cuda.Function,
    sum_f32: cuda.Function,
    sum_bf16: cuda.Function,
    attn: cuda.Function,
    attn_sum: cuda.Function,
    attn_reg: cuda.Function,
    attn_sum_reg: cuda.Function,
    moe: cuda.Function,
    moe_sum: cuda.Function,
};

const Loaded = struct {
    roce: cuda.Module,
    glue: cuda.Module,
    k: Kernels,

    fn init(gpu: Gpu) !Loaded {
        var roce = try cuda.Module.load(gpu.d, cuda.kernels.roce_gather);
        errdefer roce.unload();
        var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm53_glue);
        errdefer glue.unload();
        return .{ .roce = roce, .glue = glue, .k = .{
            .gather = try roce.function("roce_gather"),
            .sum_f32 = try roce.function("roce_gather_sum_f32"),
            .sum_bf16 = try roce.function("roce_gather_sum_bf16"),
            .attn = try glue.function("tf_glm53_residual_rmsnorm_6144"),
            .attn_sum = try glue.function("tf_glm53_residual_rmsnorm_6144_sum"),
            .attn_reg = try glue.function("tf_glm53_residual_rmsnorm_6144_reg"),
            .attn_sum_reg = try glue.function("tf_glm53_residual_rmsnorm_6144_sum_reg"),
            .moe = try glue.function("tf_glm53_moe_residual_add"),
            .moe_sum = try glue.function("tf_glm53_moe_residual_add_sum"),
        } };
    }

    fn deinit(l: *Loaded) void {
        l.roce.unload();
        l.glue.unload();
    }
};

fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn bf16FromBits(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn gridFor(packs: usize) u32 {
    const need = @max(@as(usize, 1), (packs + 1023) / 1024);
    const pow = std.math.ceilPowerOfTwo(usize, need) catch max_blocks;
    return @intCast(@min(pow, max_blocks));
}

fn launchGather(f: cuda.Function, stream: cuda.Stream, in: u64, out: u64, packs: usize, recv: u64, flags: u64, send: u64, ctrl: u64, slot_bytes: usize, counters: u64) !void {
    const grid = gridFor(packs);
    const klass: i32 = @intCast(std.math.log2_int(u32, grid));
    var a: cuda.Args = .{};
    inline for (.{
        in, out, @as(i32, @intCast(packs)), @as(i32, rank), recv, flags, send, ctrl,
        @as(i64, @intCast(slot_bytes)), counters, klass, timeout_ns, @as(i32, 0),
    }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = 512 } }, stream, &a);
}

fn launchAttn(f: cuda.Function, stream: cuda.Stream, x: u64, parts: u64, w: u64, normed: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, parts, w, normed, @as(c_int, @intCast(rows)), eps }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

fn launchMoe(f: cuda.Function, stream: cuda.Stream, x: u64, xout: u64, parts: u64, count: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, parts, @as(c_int, @intCast(count)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}

/// One local and one peer shard. Every thirteenth index is a bf16 half-ulp: the fp32 block must keep the exact add
/// (a bf16-rounded block would lose it), while the bf16 block has two representable values whose fp32 sum is that tie.
fn fillShards(a32: []u8, b32: []u8, a16: []u8, b16: []u8, count: usize) void {
    var rng = std.Random.DefaultPrng.init(0x51ed_2701);
    const random = rng.random();
    for (0..count) |i| {
        var va = (random.float(f32) - 0.5) * 8.0;
        var vb = (random.float(f32) - 0.5) * 8.0;
        var ha = bf16Bits(va);
        var hb = bf16Bits(vb);
        switch (i % 13) {
            0 => {
                va = 1.00390625;
                vb = 0;
                ha = 0x3f80;
                hb = 0x3b80;
            },
            1 => {
                va = 1.01171875;
                vb = 0;
                ha = 0x3f81;
                hb = 0x3b80;
            },
            2 => {
                va = -1.00390625;
                vb = 0;
                ha = 0xbf80;
                hb = 0xbb80;
            },
            else => {},
        }
        std.mem.writeInt(u32, a32[i * 4 ..][0..4], @bitCast(va), .little);
        std.mem.writeInt(u32, b32[i * 4 ..][0..4], @bitCast(vb), .little);
        std.mem.writeInt(u16, a16[i * 2 ..][0..2], ha, .little);
        std.mem.writeInt(u16, b16[i * 2 ..][0..2], hb, .little);
    }
}

const Rig = struct {
    gpu: Gpu,
    k: Kernels,
    stream: cuda.Stream,
    rows: usize,
    count: usize,
    packs32: usize,
    packs16: usize,
    a32: []u8,
    b32: []u8,
    a16: []u8,
    b16: []u8,
    f32_a: cuda.DeviceBuffer,
    f32_rank: cuda.DeviceBuffer,
    f32_sum: cuda.DeviceBuffer,
    f32_recv: cuda.DeviceBuffer,
    f32_send: cuda.DeviceBuffer,
    bf16_a: cuda.DeviceBuffer,
    bf16_rank: cuda.DeviceBuffer,
    bf16_sum: cuda.DeviceBuffer,
    bf16_recv: cuda.DeviceBuffer,
    bf16_send: cuda.DeviceBuffer,
    x_old: cuda.DeviceBuffer,
    x_new: cuda.DeviceBuffer,
    x_reg: cuda.DeviceBuffer,
    x_sum_reg: cuda.DeviceBuffer,
    normed_old: cuda.DeviceBuffer,
    normed_new: cuda.DeviceBuffer,
    normed_reg: cuda.DeviceBuffer,
    normed_sum_reg: cuda.DeviceBuffer,
    xm_old: cuda.DeviceBuffer,
    xm_new: cuda.DeviceBuffer,
    w: cuda.DeviceBuffer,
    ctrl: cuda.DeviceBuffer,
    counters: cuda.DeviceBuffer,
    flags: cuda.DeviceBuffer,

    fn init(gpu: Gpu, k: Kernels, rows: usize) !Rig {
        var stream = try cuda.Stream.init(gpu.d, false);
        errdefer stream.deinit();
        const count = rows * hidden;
        const f32_bytes = count * 4;
        const bf16_bytes = count * 2;
        const a32 = try gpu.gpa.alloc(u8, f32_bytes);
        errdefer gpu.gpa.free(a32);
        const b32 = try gpu.gpa.alloc(u8, f32_bytes);
        errdefer gpu.gpa.free(b32);
        const a16 = try gpu.gpa.alloc(u8, bf16_bytes);
        errdefer gpu.gpa.free(a16);
        const b16 = try gpu.gpa.alloc(u8, bf16_bytes);
        errdefer gpu.gpa.free(b16);
        fillShards(a32, b32, a16, b16, count);
        const x = try gpu.gpa.alloc(u8, bf16_bytes);
        defer gpu.gpa.free(x);
        const w = try gpu.gpa.alloc(u8, hidden * 2);
        defer gpu.gpa.free(w);
        const zeros = try gpu.gpa.alloc(u8, 64 * 4);
        defer gpu.gpa.free(zeros);
        @memset(zeros, 0);
        var wrng = std.Random.DefaultPrng.init(0xa5a5_1234);
        const wr = wrng.random();
        for (0..count) |i| std.mem.writeInt(u16, x[i * 2 ..][0..2], bf16Bits((wr.float(f32) - 0.5) * 4.0), .little);
        for (0..hidden) |i| std.mem.writeInt(u16, w[i * 2 ..][0..2], bf16Bits(0.5 + wr.float(f32)), .little);

        const r: Rig = .{
            .gpu = gpu,
            .k = k,
            .stream = stream,
            .rows = rows,
            .count = count,
            .packs32 = count / 4,
            .packs16 = count / 8,
            .a32 = a32,
            .b32 = b32,
            .a16 = a16,
            .b16 = b16,
            .f32_a = try cuda.DeviceBuffer.alloc(gpu.d, f32_bytes),
            .f32_rank = try cuda.DeviceBuffer.alloc(gpu.d, 2 * f32_bytes),
            .f32_sum = try cuda.DeviceBuffer.alloc(gpu.d, f32_bytes),
            .f32_recv = try cuda.DeviceBuffer.alloc(gpu.d, 2 * f32_bytes),
            .f32_send = try cuda.DeviceBuffer.alloc(gpu.d, 2 * f32_bytes),
            .bf16_a = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .bf16_rank = try cuda.DeviceBuffer.alloc(gpu.d, 2 * bf16_bytes),
            .bf16_sum = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .bf16_recv = try cuda.DeviceBuffer.alloc(gpu.d, 2 * bf16_bytes),
            .bf16_send = try cuda.DeviceBuffer.alloc(gpu.d, 2 * bf16_bytes),
            .x_old = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .x_new = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .x_reg = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .x_sum_reg = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .normed_old = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .normed_new = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .normed_reg = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .normed_sum_reg = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .xm_old = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .xm_new = try cuda.DeviceBuffer.alloc(gpu.d, bf16_bytes),
            .w = try cuda.DeviceBuffer.alloc(gpu.d, hidden * 2),
            .ctrl = try cuda.DeviceBuffer.alloc(gpu.d, 64 * 4),
            .counters = try cuda.DeviceBuffer.alloc(gpu.d, 64 * 4),
            .flags = try cuda.DeviceBuffer.alloc(gpu.d, 256),
        };
        try r.f32_a.upload(0, a32);
        try r.bf16_a.upload(0, a16);
        try r.x_old.upload(0, x);
        try r.x_new.upload(0, x);
        try r.x_reg.upload(0, x);
        try r.x_sum_reg.upload(0, x);
        try r.xm_old.upload(0, x);
        try r.xm_new.upload(0, x);
        try r.w.upload(0, w);
        try r.ctrl.upload(0, zeros);
        try r.counters.upload(0, zeros);
        try r.flags.upload(0, zeros);
        // Both epoch slots hold the peer shard.
        try r.f32_recv.upload(0, b32);
        try r.f32_recv.upload(f32_bytes, b32);
        try r.bf16_recv.upload(0, b16);
        try r.bf16_recv.upload(bf16_bytes, b16);
        return r;
    }

    fn deinit(r: *Rig) void {
        r.f32_a.free();
        r.f32_rank.free();
        r.f32_sum.free();
        r.f32_recv.free();
        r.f32_send.free();
        r.bf16_a.free();
        r.bf16_rank.free();
        r.bf16_sum.free();
        r.bf16_recv.free();
        r.bf16_send.free();
        r.x_old.free();
        r.x_new.free();
        r.x_reg.free();
        r.x_sum_reg.free();
        r.normed_old.free();
        r.normed_new.free();
        r.normed_reg.free();
        r.normed_sum_reg.free();
        r.xm_old.free();
        r.xm_new.free();
        r.w.free();
        r.ctrl.free();
        r.counters.free();
        r.flags.free();
        r.gpu.gpa.free(r.a32);
        r.gpu.gpa.free(r.b32);
        r.gpu.gpa.free(r.a16);
        r.gpu.gpa.free(r.b16);
        r.stream.deinit();
    }

    fn oldPair(r: *Rig) !void {
        const s = r.stream;
        try launchGather(r.k.gather, s, r.f32_a.ptr, r.f32_rank.ptr, r.packs32, r.f32_recv.ptr, r.flags.ptr, r.f32_send.ptr, r.ctrl.ptr, r.count * 4, r.counters.ptr);
        try launchAttn(r.k.attn, s, r.x_old.ptr, r.f32_rank.ptr, r.w.ptr, r.normed_old.ptr, r.rows);
        try launchGather(r.k.gather, s, r.bf16_a.ptr, r.bf16_rank.ptr, r.packs16, r.bf16_recv.ptr, r.flags.ptr, r.bf16_send.ptr, r.ctrl.ptr, r.count * 2, r.counters.ptr);
        try launchMoe(r.k.moe, s, r.xm_old.ptr, r.xm_old.ptr, r.bf16_rank.ptr, r.count);
    }

    fn newPair(r: *Rig) !void {
        const s = r.stream;
        try launchGather(r.k.sum_f32, s, r.f32_a.ptr, r.f32_sum.ptr, r.packs32, r.f32_recv.ptr, r.flags.ptr, r.f32_send.ptr, r.ctrl.ptr, r.count * 4, r.counters.ptr);
        try launchAttn(r.k.attn_sum, s, r.x_new.ptr, r.f32_sum.ptr, r.w.ptr, r.normed_new.ptr, r.rows);
        try launchGather(r.k.sum_bf16, s, r.bf16_a.ptr, r.bf16_sum.ptr, r.packs16, r.bf16_recv.ptr, r.flags.ptr, r.bf16_send.ptr, r.ctrl.ptr, r.count * 2, r.counters.ptr);
        try launchMoe(r.k.moe_sum, s, r.xm_new.ptr, r.xm_new.ptr, r.bf16_sum.ptr, r.count);
    }

    fn regPair(r: *Rig) !void {
        try launchAttn(r.k.attn_reg, r.stream, r.x_reg.ptr, r.f32_rank.ptr, r.w.ptr, r.normed_reg.ptr, r.rows);
    }

    fn sumRegPair(r: *Rig) !void {
        try launchAttn(r.k.attn_sum_reg, r.stream, r.x_sum_reg.ptr, r.f32_sum.ptr, r.w.ptr, r.normed_sum_reg.ptr, r.rows);
    }

    fn compare(r: *Rig) !void {
        try r.stream.synchronize();
        const got_x_old = try check.download(r.gpu, r.x_old);
        defer r.gpu.gpa.free(got_x_old);
        const got_x_new = try check.download(r.gpu, r.x_new);
        defer r.gpu.gpa.free(got_x_new);
        try check.sameBytes("glm53-gsum attention x", got_x_new, got_x_old);
        const got_x_reg = try check.download(r.gpu, r.x_reg);
        defer r.gpu.gpa.free(got_x_reg);
        try check.sameBytes("glm53-rmsreg attention x", got_x_reg, got_x_old);
        const got_x_sum_reg = try check.download(r.gpu, r.x_sum_reg);
        defer r.gpu.gpa.free(got_x_sum_reg);
        try check.sameBytes("glm53-rmsreg sum attention x", got_x_sum_reg, got_x_new);
        const got_normed_old = try check.download(r.gpu, r.normed_old);
        defer r.gpu.gpa.free(got_normed_old);
        const got_normed_new = try check.download(r.gpu, r.normed_new);
        defer r.gpu.gpa.free(got_normed_new);
        try check.sameBytes("glm53-gsum attention normed", got_normed_new, got_normed_old);
        const got_normed_reg = try check.download(r.gpu, r.normed_reg);
        defer r.gpu.gpa.free(got_normed_reg);
        try check.sameBytes("glm53-rmsreg attention normed", got_normed_reg, got_normed_old);
        const got_normed_sum_reg = try check.download(r.gpu, r.normed_sum_reg);
        defer r.gpu.gpa.free(got_normed_sum_reg);
        try check.sameBytes("glm53-rmsreg sum attention normed", got_normed_sum_reg, got_normed_new);
        const got_xm_old = try check.download(r.gpu, r.xm_old);
        defer r.gpu.gpa.free(got_xm_old);
        const got_xm_new = try check.download(r.gpu, r.xm_new);
        defer r.gpu.gpa.free(got_xm_new);
        try check.sameBytes("glm53-gsum moe xout", got_xm_new, got_xm_old);

        const got_f32_sum = try check.download(r.gpu, r.f32_sum);
        defer r.gpu.gpa.free(got_f32_sum);
        const got_bf16_sum = try check.download(r.gpu, r.bf16_sum);
        defer r.gpu.gpa.free(got_bf16_sum);
        const want_f32 = try r.gpu.gpa.alloc(u8, r.count * 4);
        defer r.gpu.gpa.free(want_f32);
        const want_bf16 = try r.gpu.gpa.alloc(u8, r.count * 2);
        defer r.gpu.gpa.free(want_bf16);
        for (0..r.count) |i| {
            const fa: f32 = @bitCast(std.mem.readInt(u32, r.a32[i * 4 ..][0..4], .little));
            const fb: f32 = @bitCast(std.mem.readInt(u32, r.b32[i * 4 ..][0..4], .little));
            std.mem.writeInt(u32, want_f32[i * 4 ..][0..4], @bitCast(fa + fb), .little);
            const ha = std.mem.readInt(u16, r.a16[i * 2 ..][0..2], .little);
            const hb = std.mem.readInt(u16, r.b16[i * 2 ..][0..2], .little);
            std.mem.writeInt(u16, want_bf16[i * 2 ..][0..2], bf16Bits(bf16FromBits(ha) + bf16FromBits(hb)), .little);
        }
        try check.sameBytes("glm53-gsum fp32 sum block", got_f32_sum, want_f32);
        try check.sameBytes("glm53-gsum bf16 sum block", got_bf16_sum, want_bf16);
    }
};

pub fn run(gpu: Gpu) !void {
    var images = try Loaded.init(gpu);
    defer images.deinit();
    const rows = 4;
    var r = try Rig.init(gpu, images.k, rows);
    defer r.deinit();
    try r.oldPair();
    try r.newPair();
    try r.regPair();
    try r.sumRegPair();
    try r.compare();
    check.pass("glm53-gsum rows={d}: rank-major and one-block sum agree byte for byte on x, normed and xout (fp32 attention block, bf16 ties included)", .{rows});
}

pub fn bench(gpu: Gpu, rows: usize) !void {
    var images = try Loaded.init(gpu);
    defer images.deinit();
    var r = try Rig.init(gpu, images.k, rows);
    defer r.deinit();
    try r.oldPair();
    try r.newPair();
    try r.regPair();
    try r.sumRegPair();
    try r.stream.synchronize();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(gpu.d, true);
    defer stop.deinit();
    const samples = 20;
    const old_us = try gpu.gpa.alloc(f64, samples);
    defer gpu.gpa.free(old_us);
    const sum_us = try gpu.gpa.alloc(f64, samples);
    defer gpu.gpa.free(sum_us);
    const rms_reg_us = try gpu.gpa.alloc(f64, samples);
    defer gpu.gpa.free(rms_reg_us);
    const sum_reg_us = try gpu.gpa.alloc(f64, samples);
    defer gpu.gpa.free(sum_reg_us);
    for (old_us) |*sample| {
        try start.record(r.stream);
        try launchAttn(r.k.attn, r.stream, r.x_old.ptr, r.f32_rank.ptr, r.w.ptr, r.normed_old.ptr, r.rows);
        try stop.record(r.stream);
        try stop.synchronize();
        sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0;
    }
    for (sum_us) |*sample| {
        try start.record(r.stream);
        try launchAttn(r.k.attn_sum, r.stream, r.x_new.ptr, r.f32_sum.ptr, r.w.ptr, r.normed_new.ptr, r.rows);
        try stop.record(r.stream);
        try stop.synchronize();
        sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0;
    }
    for (rms_reg_us) |*sample| {
        try start.record(r.stream);
        try r.regPair();
        try stop.record(r.stream);
        try stop.synchronize();
        sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0;
    }
    for (sum_reg_us) |*sample| {
        try start.record(r.stream);
        try r.sumRegPair();
        try stop.record(r.stream);
        try stop.synchronize();
        sample.* = @as(f64, try cuda.Event.elapsedMs(start, stop)) * 1000.0;
    }
    std.debug.print("BENCH glm53-rmsreg rows={d} | rms {d:.1} | rms_reg {d:.1} | sum {d:.1} | sum_reg {d:.1} us\n", .{ rows, check.median(old_us), check.median(rms_reg_us), check.median(sum_us), check.median(sum_reg_us) });
}
