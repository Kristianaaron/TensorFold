// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host argument packing for GLM-5.3 head tasks. Values follow forward.zig's
//! final head and embed call sites; fp8b/bf16 qmmf projection stays a boundary.
const std = @import("std");

pub const FinalNorm = extern struct {
    input: u64,
    weight: u64,
    output: u64,
    rows: u32,
    eps: f32,
};

pub const Top1 = extern struct { logits: u64, values: u64, ids: u64, vocab: u32, offset: u32 };
pub const CandidateFill = extern struct { values: u64, ids: u64, count: u32 };
pub const CandidatePart = extern struct { logits: u64, scratch: u64, vocab: u32, offset: u32, k: u32 };
pub const CandidateMerge = extern struct { scratch: u64, values: u64, ids: u64, vocab: u32, k: u32 };
pub const Lse = extern struct { logits: u64, partial: u64, vocab: u32 };
pub const GateGather = extern struct { logits: u64, ids: u64, output: u64, vocab: u32, offset: u32, probes: u32, rows: u32 };
pub const GateMass = extern struct { probe_logits: u64, lse: u64, logp: u64, mass_outside: u64, probes: u32, rows: u32 };
pub const EmbedPick = extern struct { gathered: u64, owners: u64, output: u64, rows: u32, hidden: u32 = 6144, world: u32 = 2 };

pub const Kind = enum { final_norm, top1, candidate_fill, candidate_part, candidate_merge, lse, gate_gather, gate_mass, embed_pick, lm_head_projection };
pub const GridPlan = struct { gx: u32, gy: u32 = 1, gz: u32 = 1, threads: u32, smem_bytes: u32 };

/// Wrapper-equivalent grid and static shared-memory requirements. LM-head qmmf
/// remains a boundary: its task uses staging barriers and non-256 block shapes.
pub fn plan(kind: Kind, rows: usize, vocab: usize, k: usize, probes: usize) !GridPlan {
    if (rows == 0) return error.EmptyWork;
    if (kind != .embed_pick and vocab == 0) return error.EmptyVocab;
    if ((kind == .candidate_fill or kind == .candidate_part or kind == .candidate_merge) and (k == 0 or k > 2048)) return error.InvalidCandidateCount;
    return switch (kind) {
        .final_norm => .{ .gx = @intCast(rows), .threads = 256, .smem_bytes = 32 },
        .top1 => .{ .gx = @intCast(rows), .threads = 256, .smem_bytes = 2048 },
        .candidate_fill => .{ .gx = @intCast(try std.math.divCeil(usize, try std.math.mul(usize, rows, k), 256)), .threads = 256, .smem_bytes = 0 },
        .candidate_part => .{ .gx = @intCast(rows), .gy = 32, .threads = 256, .smem_bytes = 19484 },
        .candidate_merge => .{ .gx = @intCast(rows), .gy = 32, .threads = 256, .smem_bytes = 16384 },
        .lse => .{ .gx = @intCast(rows), .threads = 256, .smem_bytes = 1024 },
        .gate_gather => .{ .gx = @intCast(try std.math.divCeil(usize, try std.math.mul(usize, rows, probes), 256)), .threads = 256, .smem_bytes = 0 },
        .gate_mass => .{ .gx = @intCast(rows), .threads = 1, .smem_bytes = 0 },
        .embed_pick => .{ .gx = @intCast(try std.math.divCeil(usize, try std.math.mul(usize, rows, 6144), 256)), .threads = 256, .smem_bytes = 0 },
        .lm_head_projection => error.BoundaryTask,
    };
}

/// The host-side inputs at Forward.headRows (`forward.zig:903-911`) and
/// Forward.embed (`forward.zig:705-713`). Addresses are CUDA device pointers.
pub const Step = struct {
    x: u64,
    xn: u64,
    norm: u64,
    logits: u64,
    candidate_values: u64,
    candidate_ids: u64,
    candidate_scratch: u64,
    gathered: u64,
    owners: u64,
    vocab: u32,
    vocab_offset: u32,
    rows: u32,
    first_row: u32 = 0,
    eps: f32 = 1e-5,
};

/// Bind the same subspans selected in `Forward.headRows` (`forward.zig:903-912`).
/// `first_row` addresses residual input; output row 0 is this head tile's rank-major `mine` slot.
/// `candidate_k == null` selects greedy's value/id layout; otherwise it selects the k-candidate layout.
pub fn fromForward(f: anytype, first_row: usize, rows: usize, candidate_k: ?u32) Step {
    const ws = f.st.ws;
    const candidates = candidate_k orelse 1;
    const row_bytes = if (candidate_k) |k| @as(usize, k) * 8 else 16;
    const mine = ws.head_gather + @as(u64, @intCast(f.rank * rows * row_bytes));
    return .{
        .x = ws.x,
        .xn = ws.xn,
        .norm = f.w.norm,
        .logits = ws.logits,
        .candidate_values = mine,
        .candidate_ids = mine + @as(u64, @intCast(rows * @as(usize, candidates) * 4)),
        .candidate_scratch = ws.cand_scratch,
        .gathered = ws.gathered,
        .owners = ws.owners,
        .vocab = @intCast(f.w.vocab.hi - f.w.vocab.lo),
        .vocab_offset = @intCast(f.w.vocab.lo),
        .rows = @intCast(rows),
        .first_row = @intCast(first_row),
        .eps = f.w.config.eps,
    };
}

pub fn finalNorm(s: Step) FinalNorm {
    return .{
        .input = s.x + @as(u64, s.first_row) * 6144 * 2,
        .weight = s.norm,
        .output = s.xn,
        .rows = s.rows,
        .eps = s.eps,
    };
}

pub fn top1(s: Step) Top1 {
    return .{ .logits = s.logits, .values = s.candidate_values, .ids = s.candidate_ids, .vocab = s.vocab, .offset = s.vocab_offset };
}

pub fn candidateFill(s: Step, k: u32) CandidateFill {
    return .{ .values = s.candidate_values, .ids = s.candidate_ids, .count = s.rows * k };
}

pub fn candidatePart(s: Step, k: u32) CandidatePart {
    return .{ .logits = s.logits, .scratch = s.candidate_scratch, .vocab = s.vocab, .offset = s.vocab_offset, .k = k };
}

pub fn candidateMerge(s: Step, k: u32) CandidateMerge {
    return .{ .scratch = s.candidate_scratch, .values = s.candidate_values, .ids = s.candidate_ids, .vocab = s.vocab, .k = k };
}

pub fn lse(s: Step, partial: u64) Lse {
    return .{ .logits = s.logits, .partial = partial, .vocab = s.vocab };
}

pub fn gateGather(s: Step, ids: u64, output: u64, probes: u32) GateGather {
    return .{ .logits = s.logits, .ids = ids, .output = output, .vocab = s.vocab, .offset = s.vocab_offset, .probes = probes, .rows = s.rows };
}

pub fn gateMass(probe_logits: u64, lse_ptr: u64, logp: u64, mass_outside: u64, probes: u32, rows: u32) GateMass {
    return .{ .probe_logits = probe_logits, .lse = lse_ptr, .logp = logp, .mass_outside = mass_outside, .probes = probes, .rows = rows };
}

pub fn embedPick(gathered: u64, owners: u64, output: u64, rows: u32) EmbedPick {
    return .{ .gathered = gathered, .owners = owners, .output = output, .rows = rows };
}

/// `Forward.embed` (`forward.zig:705-713`): caller passes this rank's gathered
/// segment, then we bind the corresponding owner and residual row slices.
pub fn embedPickFromForward(f: anytype, rank_gathered: u64, first_row: usize, rows: usize) EmbedPick {
    return .{
        .gathered = rank_gathered,
        .owners = f.st.ws.owners + @as(u64, @intCast(first_row * @sizeOf(i32))),
        .output = f.st.ws.x + @as(u64, @intCast(first_row * 6144 * @sizeOf(u16))),
        .rows = @intCast(rows),
    };
}

test "head task argument packers preserve forward buffer spans" {
    const s: Step = .{
        .x = 0x1000,
        .xn = 0x8000,
        .norm = 0x9000,
        .logits = 0xa000,
        .candidate_values = 0xb000,
        .candidate_ids = 0xc000,
        .candidate_scratch = 0xd000,
        .gathered = 0xe000,
        .owners = 0xf000,
        .vocab = 77312,
        .vocab_offset = 0,
        .rows = 4,
        .first_row = 2,
    };
    const norm = finalNorm(s);
    try std.testing.expectEqual(@as(u64, 0x1000 + 2 * 6144 * 2), norm.input);
    try std.testing.expectEqual(s.norm, norm.weight);
    try std.testing.expectEqual(s.xn, norm.output);
    try std.testing.expectEqual(@as(u32, 4), norm.rows);
    try std.testing.expectEqual(s.logits, top1(s).logits);
    try std.testing.expectEqual(@as(u32, 8), candidateFill(s, 2).count);
    try std.testing.expectEqual(s.candidate_scratch, candidatePart(s, 64).scratch);
    try std.testing.expectEqual(s.candidate_scratch, candidateMerge(s, 64).scratch);
    try std.testing.expectEqual(@as(u32, 77312), lse(s, 0x11000).vocab);
    try std.testing.expectEqual(@as(u32, 4), gateGather(s, 0x12000, 0x13000, 128).rows);
    try std.testing.expectEqual(@as(u32, 2), embedPick(s.gathered, s.owners, s.x, 2).world);
    const parts = try plan(.candidate_part, 4, 77312, 64, 0);
    try std.testing.expectEqual(@as(u32, 4), parts.gx);
    try std.testing.expectEqual(@as(u32, 32), parts.gy);
    try std.testing.expectEqual(@as(u32, 19484), parts.smem_bytes);
    try std.testing.expectError(error.BoundaryTask, plan(.lm_head_projection, 4, 77312, 0, 0));
}

test "forward fields bind head and next-step embedding slices" {
    const Mock = struct {
        st: struct { ws: struct {
            x: u64,
            xn: u64,
            logits: u64,
            cand_scratch: u64,
            gathered: u64,
            owners: u64,
            head_gather: u64,
        } },
        w: struct { norm: u64, vocab: struct { lo: usize, hi: usize }, config: struct { eps: f32 } },
        rank: usize,
    };
    const f: Mock = .{
        .st = .{ .ws = .{ .x = 0x1000, .xn = 0x2000, .logits = 0x3000, .cand_scratch = 0x6000, .gathered = 0x7000, .owners = 0x8000, .head_gather = 0xb000 } },
        .w = .{ .norm = 0x9000, .vocab = .{ .lo = 77440, .hi = 154880 }, .config = .{ .eps = 1e-5 } },
        .rank = 1,
    };
    const step = fromForward(&f, 2, 4, null);
    try std.testing.expectEqual(@as(u32, 77440), step.vocab);
    try std.testing.expectEqual(@as(u32, 77440), step.vocab_offset);
    try std.testing.expectEqual(@as(u64, 0x9000), finalNorm(step).weight);
    try std.testing.expectEqual(@as(u64, 0x3000), top1(step).logits);
    try std.testing.expectEqual(@as(u64, 0xb000 + 4 * 16), step.candidate_values);
    try std.testing.expectEqual(@as(u64, 0xb000 + 4 * 16 + 4 * 4), step.candidate_ids);
    const candidate_step = fromForward(&f, 2, 4, 64);
    try std.testing.expectEqual(@as(u64, 0xb000 + 4 * 64 * 8), candidate_step.candidate_values);
    try std.testing.expectEqual(@as(u64, 0xb000 + 4 * 64 * 8 + 4 * 64 * 4), candidate_step.candidate_ids);
    const embed = embedPickFromForward(&f, 0xa000, 2, 4);
    try std.testing.expectEqual(@as(u64, 0xa000), embed.gathered);
    try std.testing.expectEqual(@as(u64, 0x8000 + 8), embed.owners);
    try std.testing.expectEqual(@as(u64, 0x1000 + 2 * 6144 * 2), embed.output);
}

test {
    std.testing.refAllDecls(@This());
    inline for (.{
        FinalNorm,  Top1,     CandidateFill, CandidatePart, CandidateMerge, Lse,
        GateGather, GateMass, EmbedPick,     Step,
    }) |T| std.testing.refAllDecls(T);
}
