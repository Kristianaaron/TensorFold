// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP3 host math checks and standalone GB10 MMA microbenchmark target.
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = @import("cuda.zig").runtime(b, host, .debug, &.{});
    const ref = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/fp8_mma.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    ref.addImport("cuda", cuda);
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = ref }));
    const step = b.step("test-glm53-fp8-mma-host", "GLM-5.3 W8A8 activation quantization and scaled GEMM references");
    step.dependOn(&tests.step);
    test_step.dependOn(step);

    const cutlass = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/cutlass.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    cutlass.addImport("cuda", cuda);
    const probe = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_fp8_backend_probe.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    probe.addImport("cuda", cuda);
    probe.addImport("glm53_cutlass", cutlass);
    probe.addImport("glm53_fp8", ref);

    const bench = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mma_peak.zig"), .target = host, .optimize = .ReleaseFast, .link_libc = true });
    bench.addImport("cuda", cuda);
    bench.addImport("fp8_mma", ref);
    bench.addImport("fp8_backend_probe", probe);
    const bench_compile = b.addObject(.{ .name = "glm53_mma_peak_host_compile", .root_module = bench });
    const bench_tests = b.addRunArtifact(b.addTest(.{ .root_module = bench }));
    const compile_bench = b.step("compile-glm53-mma-peak-host", "Type-check and fully analyze the GLM-5.3 MMA benchmark without CUDA hardware");
    compile_bench.dependOn(&bench_compile.step);
    compile_bench.dependOn(&bench_tests.step);

    const cutlass_script = b.addWriteFiles().add("build-glm53-cutlass.sh", "#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")/../../zig/kernels/cuda/host\"\nnvcc -shared -Xcompiler -fPIC -cudart static -std=c++17 --expt-relaxed-constexpr -O3 -I../third_party/cutlass/include -I../third_party/cutlass/tools/util/include -gencode=arch=compute_121a,code=sm_121a -o libtf_glm53_cutlass.so glm53_cutlass.cu\nmkdir -p ../../../../zig-out/kernels\nmv libtf_glm53_cutlass.so ../../../../zig-out/kernels/libtf_glm53_cutlass.so\n");
    const cutlass_step = b.step("build-glm53-cutlass-script", "Install the exact Spark nvcc command for the GLM-5.3 CUTLASS host library");
    const cutlass_install = b.addInstallFile(cutlass_script, "bin/build-glm53-cutlass.sh");
    const cutlass_chmod = b.addSystemCommand(&.{ "chmod", "+x", "zig-out/bin/build-glm53-cutlass.sh" });
    cutlass_chmod.step.dependOn(&cutlass_install.step);
    cutlass_step.dependOn(&cutlass_chmod.step);

    const backends = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/fp8_backend.zig"), .target = host, .optimize = .Debug });
    const backend_tests = b.addRunArtifact(b.addTest(.{ .root_module = backends }));
    const cutlass_tests = b.addRunArtifact(b.addTest(.{ .root_module = cutlass }));
    const probe_tests = b.addRunArtifact(b.addTest(.{ .root_module = probe }));
    const cublas_test_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_cublaslt_host.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    cublas_test_module.addImport("cuda", cuda);
    const cublas_tests = b.addRunArtifact(b.addTest(.{ .root_module = cublas_test_module }));
    const probe_compile = b.addObject(.{ .name = "glm53_fp8_backend_probe_host", .root_module = probe });
    const probe_step = b.step("compile-glm53-fp8-backend-probe-host", "Fully analyze the CUTLASS/cuBLASLt probe without CUDA hardware");
    probe_step.dependOn(&probe_compile.step);
    const linux = b.resolveTargetQuery(.{ .cpu_arch = host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const cuda_linux = @import("cuda.zig").runtime(b, linux, .Debug, &.{});
    const cutlass_linux = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/cutlass.zig"), .target = linux, .optimize = .Debug, .link_libc = true });
    cutlass_linux.addImport("cuda", cuda_linux);
    const fp8_linux = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/fp8_mma.zig"), .target = linux, .optimize = .Debug, .link_libc = true });
    fp8_linux.addImport("cuda", cuda_linux);
    const probe_linux = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_fp8_backend_probe.zig"), .target = linux, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda_linux }, .{ .name = "glm53_cutlass", .module = cutlass_linux }, .{ .name = "glm53_fp8", .module = fp8_linux } } });
    const probe_exe = b.addExecutable(.{ .name = "glm53_fp8_backend_probe", .root_module = probe_linux });
    const probe_install = b.addInstallArtifact(probe_exe, .{});
    b.step("glm53-fp8-backend-probe", "Build the Linux GLM-5.3 CUTLASS/cuBLASLt Spark probe (run with g or h)").dependOn(&probe_install.step);
    const backend_step = b.step("test-glm53-fp8-backends-host", "GLM-5.3 FP8 backend selection and CUTLASS loader analysis on host");
    backend_step.dependOn(&backend_tests.step);
    backend_step.dependOn(&cutlass_tests.step);
    backend_step.dependOn(&cublas_tests.step);
    backend_step.dependOn(&probe_tests.step);
    backend_step.dependOn(probe_step);
    b.step("test-glm53-prefill-host", "GLM-5.3 FP8 prefill backend reference, fixed row tiles and launch-shape analysis").dependOn(&probe_tests.step);
}
