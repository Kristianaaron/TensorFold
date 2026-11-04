// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! MK3 runtime host checks. Keep graph.zig and launch.zig as independent test roots.
const std = @import("std");

pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const host = b.graph.host;
    const graph_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/graph.zig"),
        .target = host,
        .optimize = .Debug,
    });
    const graph_test = b.addRunArtifact(b.addTest(.{ .root_module = graph_module }));
    const qmmf_bind_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/bind_qmmf.zig"), .target = host, .optimize = .Debug });
    const step_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/step.zig"), .target = host, .optimize = .Debug });
    step_module.addImport("graph.zig", graph_module);
    step_module.addImport("bind_qmmf.zig", qmmf_bind_module);
    const step_test = b.addRunArtifact(b.addTest(.{ .root_module = step_module }));
    const qmmf_bind_test = b.addRunArtifact(b.addTest(.{ .root_module = qmmf_bind_module }));
    const exec_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/exec.zig"), .target = host, .optimize = .Debug });
    exec_module.addImport("graph.zig", graph_module);
    exec_module.addImport("step.zig", step_module);
    exec_module.addImport("bind_qmmf.zig", qmmf_bind_module);
    const exec_test = b.addRunArtifact(b.addTest(.{ .root_module = exec_module }));
    const cuda = @import("cuda.zig").runtime(b, host, .Debug, &.{});
    const launch_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/launch.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    launch_module.addImport("cuda", cuda);
    const launch_test = b.addRunArtifact(b.addTest(.{ .root_module = launch_module }));
    // MK8 deferred launch: the sink, plan cache and CUDA backend over launch.zig (imported by module name).
    const defer_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/defer.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    defer_module.addImport("cuda", cuda);
    defer_module.addImport("mk_launch", launch_module);
    const defer_test = b.addRunArtifact(b.addTest(.{ .root_module = defer_module }));
    const bind_head_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/bind_head.zig"), .target = host, .optimize = .Debug });
    const bind_head_test = b.addRunArtifact(b.addTest(.{ .root_module = bind_head_module }));
    const head_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_head_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    head_test_module.addImport("cuda", cuda);
    head_test_module.addImport("mk_launch", launch_module);
    const sampler_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/sampler.zig"), .target = host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    head_test_module.addImport("sampler.zig", sampler_module);
    const head_test = b.addRunArtifact(b.addTest(.{ .root_module = head_test_module }));
    const step = b.step("test-glm53-mk-host", "GLM-5.3 persistent runtime ABI, upload planning and graph tests without a GPU");
    const attn = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_attn_tests.zig"), .target = host, .optimize = .Debug });
    attn.addImport("bind_attn", b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/bind_attn.zig"), .target = host, .optimize = .Debug }));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = attn })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/bind_index.zig"), .target = host, .optimize = .Debug }) })).step);
    step.dependOn(&graph_test.step);
    step.dependOn(&step_test.step);
    step.dependOn(&exec_test.step);
    step.dependOn(&qmmf_bind_test.step);
    step.dependOn(&launch_test.step);
    step.dependOn(&defer_test.step);
    const defer_gpu_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_defer_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    defer_gpu_module.addImport("cuda", cuda);
    defer_gpu_module.addImport("mk_defer", defer_module);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = defer_gpu_module })).step);
    step.dependOn(&bind_head_test.step);
    step.dependOn(&head_test.step);
    const gpu_attn_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_attn_gpu_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    gpu_attn_test_module.addImport("cuda", cuda);
    gpu_attn_test_module.addImport("mk_launch", launch_module);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gpu_attn_test_module })).step);
    const index_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_index_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    index_test_module.addImport("cuda", cuda);
    index_test_module.addImport("mk_launch", launch_module);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = index_test_module })).step);
    const qmmf_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_qmmf_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    qmmf_test_module.addImport("cuda", cuda);
    qmmf_test_module.addImport("mk_launch", launch_module);
    qmmf_test_module.addImport("bind_qmmf", b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/megakernel/bind_qmmf.zig"), .target = host, .optimize = .Debug }));
    qmmf_test_module.addImport("check.zig", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/check.zig"), .target = host, .optimize = .Debug, .imports = &.{.{ .name = "cuda", .module = cuda }} }));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qmmf_test_module })).step);
    const moe_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_moe_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    moe_test_module.addImport("cuda", cuda);
    moe_test_module.addImport("mk_launch", launch_module);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = moe_test_module })).step);
    const runtime_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mk_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    runtime_test_module.addImport("cuda", cuda);
    runtime_test_module.addImport("mk_launch", launch_module);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = runtime_test_module })).step);
    all.dependOn(step);
}
