// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP10 host tests for safetensors dumps, fp32 sampling and gate partials.
const std = @import("std");
const cuda_build = @import("cuda.zig");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = cuda_build.runtime(b, host, .Debug, &.{});
    const dump = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/dump.zig"), .target = host, .optimize = .Debug });
    const safetensors = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = host, .optimize = .Debug });
    dump.addImport("safetensors", safetensors);
    const sampler = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/sampler.zig"), .target = host, .optimize = .Debug });
    sampler.addImport("cuda", cuda);
    const dump_test = b.createModule(.{ .root_source_file = b.path("zig/tests/glm53_cuda_dump.zig"), .target = host, .optimize = .Debug });
    dump_test.addImport("dump", dump);
    dump_test.addImport("safetensors", safetensors);
    const gate_test = b.createModule(.{ .root_source_file = b.path("zig/tests/glm53_cuda_gate.zig"), .target = host, .optimize = .Debug });
    gate_test.addImport("sampler", sampler);
    const run_dump = b.addRunArtifact(b.addTest(.{ .root_module = dump_test }));
    const run_gate = b.addRunArtifact(b.addTest(.{ .root_module = gate_test }));
    const run_sampler = b.addRunArtifact(b.addTest(.{ .root_module = sampler }));
    run_dump.setCwd(b.path("."));
    const step = b.step("test-glm53-wp10-host", "GLM-5.3 safetensors dump, candidate sampling and gate host references");
    step.dependOn(&run_dump.step);
    step.dependOn(&run_gate.step);
    step.dependOn(&run_sampler.step);
    test_step.dependOn(step);
}
