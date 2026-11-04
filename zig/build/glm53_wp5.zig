// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP5's standalone host references for GLM-5.3's opt-in X3E4 expert path.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const host = b.graph.host;
    const cuda = @import("cuda.zig").runtime(b, host, .debug, &.{});
    const x3e4 = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/x3e4.zig"),
        .target = host,
        .optimize = .Debug,
        .link_libc = true,
    });
    x3e4.addImport("cuda", cuda);
    const fp8_mma = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/fp8_mma.zig"),
        .target = host,
        .optimize = .Debug,
        .link_libc = true,
    });
    fp8_mma.addImport("cuda", cuda);
    x3e4.addImport("fp8_mma", fp8_mma);
    const exl3_format = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exl3_format.zig"),
        .target = host,
        .optimize = .Debug,
        .link_libc = true,
    });
    const tests = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm53_x3e4_host.zig"),
        .target = host,
        .optimize = .Debug,
        .link_libc = true,
    });
    tests.addImport("x3e4", x3e4);
    tests.addImport("cuda", cuda);
    tests.addImport("fp8_mma", fp8_mma);
    tests.addImport("exl3_format", exl3_format);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    const step = b.step("test-glm53-wp5-x3e4-host", "GLM-5.3 X3E4 quantization, trellis and fragment references without a GPU");
    step.dependOn(&run.step);
}
