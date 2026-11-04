// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 routed prompt GEMMs against decode per-slot outputs and row-window invariance.
const std = @import("std");
const cuda = @import("cuda");
const quant_ref = @import("quant_ref_tests.zig");
const experts_f64 = @import("exl3_experts_f64_tests.zig");
const Rig = quant_ref.RigT;

test "GLM-5.3 prompt experts declarations are analyzed" {
    std.testing.refAllDecls(@This());
}

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.exl3_experts.len == 0) return error.KernelsNotEmbedded;
    var exp = try cuda.exl3_experts.Experts.load(gpu.d);
    defer exp.unload();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x530e_128);
    var rig: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer rig.s.deinit();
    defer {
        for (rig.bufs.items) |*buf| buf.free();
        rig.bufs.deinit(gpu.gpa);
    }
    try experts_f64.promptMixed(&rig, &exp);
}
