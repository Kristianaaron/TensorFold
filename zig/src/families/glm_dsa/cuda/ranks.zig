// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31: zig/src/families/glm/cuda_ranks.zig (Apache-2.0). Changes: fingerprints DCP, KV format, routed codebook, and override digest.
//! Cross-rank agreement for GLM-5.3 CUDA settings that change cache layout or arithmetic.

const std = @import("std");

pub const Settings = struct {
    dcp: u8 = 2,
    kv_format: KvFormat = .bf16,
    routed_codebook: u8 = 3,
    drafter: bool = false,
    prompt_rows: u32 = 2048,
    prefill_chunk: u32 = 4096,
    overrides: [32]u8 = @splat(0),
    config: [32]u8 = @splat(0),
    checkpoint: [32]u8 = @splat(0),
};

pub const KvFormat = enum(u8) { bf16, fp8 };

pub const Fingerprint = extern struct {
    protocol: u32 = 1,
    world: u32,
    rank: u32,
    dcp: u32,
    kv_format: u32,
    routed_codebook: u32,
    drafter: u32,
    prompt_rows: u32,
    prefill_chunk: u32,
    overrides: [32]u8,
    config: [32]u8,
    checkpoint: [32]u8,

    pub fn of(world: u32, rank: u32, s: Settings) Fingerprint {
        return .{ .world = world, .rank = rank, .dcp = s.dcp, .kv_format = @backingInt(s.kv_format), .routed_codebook = s.routed_codebook, .drafter = @intFromBool(s.drafter), .prompt_rows = s.prompt_rows, .prefill_chunk = s.prefill_chunk, .overrides = s.overrides, .config = s.config, .checkpoint = s.checkpoint };
    }
};

pub const Mismatch = struct { field: Field, rank: usize };
pub const Field = enum { protocol, rank, world, dcp, kv_format, routed_codebook, drafter, prompt_rows, prefill_chunk, overrides, config, checkpoint };

/// Rank zero supplies the reference tuple; every rank must have identical numerical and layout settings.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    if (all.len == 0) return null;
    for (all, 0..) |f, rank| {
        if (f.protocol != all[0].protocol) return .{ .field = .protocol, .rank = rank };
        if (f.rank != rank) return .{ .field = .rank, .rank = rank };
        if (f.world != all[0].world) return .{ .field = .world, .rank = rank };
        if (f.dcp != all[0].dcp) return .{ .field = .dcp, .rank = rank };
        if (f.kv_format != all[0].kv_format) return .{ .field = .kv_format, .rank = rank };
        if (f.routed_codebook != all[0].routed_codebook) return .{ .field = .routed_codebook, .rank = rank };
        if (f.drafter != all[0].drafter) return .{ .field = .drafter, .rank = rank };
        if (f.prompt_rows != all[0].prompt_rows) return .{ .field = .prompt_rows, .rank = rank };
        if (f.prefill_chunk != all[0].prefill_chunk) return .{ .field = .prefill_chunk, .rank = rank };
        if (!std.mem.eql(u8, &f.overrides, &all[0].overrides)) return .{ .field = .overrides, .rank = rank };
        if (!std.mem.eql(u8, &f.config, &all[0].config)) return .{ .field = .config, .rank = rank };
        if (!std.mem.eql(u8, &f.checkpoint, &all[0].checkpoint)) return .{ .field = .checkpoint, .rank = rank };
    }
    return null;
}

test "rank fingerprint agrees on DCP, cache, codebook, and override digest" {
    const a = Fingerprint.of(2, 0, .{ .overrides = @splat(7) });
    var b = Fingerprint.of(2, 1, .{ .overrides = @splat(7) });
    var both = [_]Fingerprint{ a, b };
    try std.testing.expect(mismatch(&both) == null);
    b.kv_format = @backingInt(KvFormat.fp8);
    both[1] = b;
    try std.testing.expectEqual(Field.kv_format, mismatch(&both).?.field);
}

test "rank fingerprints reject drafter mode mismatch and analyze public declarations" {
    const a = Fingerprint.of(2, 0, .{ .drafter = true });
    const b = Fingerprint.of(2, 1, .{ .drafter = false });
    try std.testing.expectEqual(Field.drafter, mismatch(&.{ a, b }).?.field);
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Settings);
    std.testing.refAllDecls(Fingerprint);
}
