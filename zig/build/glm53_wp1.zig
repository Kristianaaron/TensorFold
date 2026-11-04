// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP1 host checks and the GLM-5.3 checkpoint census executable.
const std = @import("std");

pub fn steps(b: *std.Build, all_tests: *std.Build.Step) void {
    const host = b.graph.host;
    const config = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/config.zig"), .target = host, .optimize = .Debug });
    const share = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/share.zig"), .target = host, .optimize = .Debug });
    share.addImport("config", config);
    const cuda_memory = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/memory.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    const types = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/types.zig"), .target = host, .optimize = .Debug });
    types.addImport("cuda_memory", cuda_memory);
    const census = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/census.zig"), .target = host, .optimize = .Debug });
    const weight_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/weight_plan.zig"), .target = host, .optimize = .Debug });
    const test_root = b.createModule(.{
        .root_source_file = b.path("zig/tests/glm53_census.zig"),
        .target = host,
        .optimize = .Debug,
    });
    test_root.addImport("config", config);
    test_root.addImport("share", share);
    test_root.addImport("types", types);
    test_root.addImport("census", census);
    const test_step = b.step("test-glm53-cuda-host", "GLM-5.3 config, TP shares, buffer contracts and synthetic checkpoint census on the host");
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = test_root }));
    tests.setCwd(b.path("."));
    test_step.dependOn(&tests.step);
    // Each imported module is also a test root so Zig collects its own test blocks.
    addModuleTests(b, test_step, config);
    addModuleTests(b, test_step, share);
    addModuleTests(b, test_step, types);
    addModuleTests(b, test_step, census);
    addModuleTests(b, test_step, weight_plan);
    b.step("test-glm53-wp1-host", "GLM-5.3 full WP1 host contracts").dependOn(test_step);
    all_tests.dependOn(test_step);

    const tool = b.createModule(.{
        .root_source_file = b.path("zig/tools/glm53_census.zig"),
        .target = host,
        .optimize = .ReleaseFast,
    });
    tool.addImport("census", census);
    const census_exe = b.addExecutable(.{ .name = "tf-glm53-census", .root_module = tool });
    const census_install = b.addInstallArtifact(census_exe, .{});
    b.getInstallStep().dependOn(&census_install.step);
    b.step("tf-glm53-census", "Build the GLM-5.3 safetensors byte census tool").dependOn(&census_install.step);
}

fn addModuleTests(b: *std.Build, parent: *std.Build.Step, module: *std.Build.Module) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    run.setCwd(b.path("."));
    parent.dependOn(&run.step);
}
