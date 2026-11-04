// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const admission = b.createModule(.{ .root_source_file = b.path("zig/glm_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "engine_api", .module = api } } });
    const admission_run = b.addRunArtifact(b.addTest(.{ .root_module = admission, .filters = &.{"GLM CUDA admission:"} }));
    b.step("test-glm-admission-host", "Real GLM admission capabilities through the lane worker without GPU work").dependOn(&admission_run.step);
    step.dependOn(&admission_run.step);
    b.step("test-glm-prompt-settings-host", "Actual opt-in settings and rank agreement on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prompt K experiment"} })).step);
    const mm_record = b.createModule(.{ .root_source_file = b.path("zig/test_support/glm_mm_cuda.zig"), .target = host, .optimize = .debug });
    const mm_actual = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_mm.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = mm_record }} });
    const mm_policy = b.createModule(.{ .root_source_file = b.path("zig/glm_mm_policy_host.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = mm_record }, .{ .name = "mm", .module = mm_actual } } });
    b.step("test-glm-mm-policy-host", "Actual prompt K policy dispatch without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mm_policy })).step);
    const drafter_host = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core }, .{ .name = "cuda", .module = cuda } } });
    b.step("test-glm-dflash-host", "DFlash rank buffers, partial loader ownership and pending records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = drafter_host, .filters = &.{"glm dflash:"} })).step);
    const ab = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_dflash_ab.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "check", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/check.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = cuda }} }) } } });
    const ab_obj = b.addObject(.{ .name = "glm53-dflash-ab-host", .root_module = ab });
    b.step("compile-glm53-dflash-ab-host", "Compile the Spark DFlash kernel parity harness on the host without CUDA hardware").dependOn(&ab_obj.step);
    const ab_tests = b.addRunArtifact(b.addTest(.{ .root_module = ab }));
    const ab_test_step = b.step("test-glm53-dflash-ab-host", "Run the GLM DFlash h8 attention FP64 oracle and host references");
    ab_test_step.dependOn(&ab_tests.step);
    step.dependOn(ab_test_step);
    const dsa_geometry = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/dflash_geometry.zig"), .target = host, .optimize = .debug });
    const dsa_drafts = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/drafts.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "geometry", .module = dsa_geometry }, .{ .name = "glm", .module = glm } } });
    const dsa_taps = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/taps.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "geometry", .module = dsa_geometry } } });
    const dsa_dflash = b.createModule(.{ .root_source_file = b.path("zig/glm_dsa_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "drafts", .module = dsa_drafts }, .{ .name = "taps", .module = dsa_taps }, .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    const dsa_dflash_step = b.step("test-glm-dsa-dflash-host", "GLM-5.3 DFlash geometry, taps and per-rank memory plans on the CPU");
    dsa_dflash_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dsa_dflash })).step);
    dsa_dflash_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dsa_drafts })).step);
    const packed_host = b.createModule(.{ .root_source_file = b.path("zig/glm_packed_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-packed-host", "Packed GLM launch records, cap parsing and native loader on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = packed_host, .filters = &.{"glm packed:"} })).step);
    const ring_host = b.createModule(.{ .root_source_file = b.path("zig/glm_ring_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-glm-ring-host", "Actual GLM MTP rollback pool through recording CUDA copies").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ring_host, .filters = &.{"glm ring:"} })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    const prompt_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes } } });
    const prompt_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_host, .filters = &.{"exl3 prompt:"} }));
    prompt_run.has_side_effects = true;
    b.step("test-exl3-prompt-host", "Actual common prompt dispatch and scratch with recording CUDA calls").dependOn(&prompt_run.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, comptime makeGlm: anytype) void {
    const ab = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_dflash_ab.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "check", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/check.zig"), .target = target, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = cuda }} }) } } });
    const ab_obj = b.addObject(.{ .name = "glm53-dflash-ab-linux", .root_module = ab });
    const ab_exe = b.addExecutable(.{ .name = "tf-glm53-dflash-ab", .root_module = ab });
    const ab_install = b.addInstallArtifact(ab_exe, .{});
    const ab_linux = b.step("compile-glm53-dflash-ab-linux", "Compile the runnable Spark DFlash kernel parity harness for the Linux CUDA target");
    ab_linux.dependOn(&ab_obj.step);
    ab_linux.dependOn(&ab_install.step);
    const glm_open = b.addObject(.{ .name = "glm-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const prompt_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_mm_fixed_tests.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }, .{ .name = "cuda", .module = cuda } } });
    const prompt_gate_compile = b.addObject(.{ .name = "glm-prompt-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = prompt_gate }} }) });
    _ = prompt_gate_compile.getEmittedBin();
    b.step("compile-glm-prompt-gate-host", "Compile the actual device prompt comparison gate with inert images").dependOn(&prompt_gate_compile.step);
}
