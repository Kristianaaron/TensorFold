// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! MK1 persistent megakernel task graph host checks.
const std = @import("std");

pub fn steps(b: *std.Build, all_tests: *std.Build.Step) void {
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/graph.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    const step = b.step("test-glm53-mk1-host", "GLM-5.3 MK1 persistent task graph and host layout checks");
    step.dependOn(&run.step);
    all_tests.dependOn(step);
}
