// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
const metal_build = @import("zig/build/metal_targets.zig");
const std = @import("std");
const glm_expiry_build = @import("zig/build/glm_expiry.zig");
const builtin = @import("builtin");
const cuda_build = @import("zig/build/cuda.zig");
const dist_build = @import("zig/build/dist.zig");
const qwen27_build = @import("zig/build/qwen27.zig");
const core_checks = @import("zig/build/core_checks.zig");
const glm_checks = @import("zig/build/glm_checks.zig");
const merge_build = @import("zig/build/merge.zig");
const flashnext_checks = @import("zig/build/flashnext_checks.zig");
const hip_build = @import("zig/build/hip.zig");
const xpu_build = @import("zig/build/xpu.zig");
const grammar_build = @import("zig/build/grammar.zig");
const counters_build = @import("zig/build/metal_counters.zig");
const exl3_build = @import("zig/build/exl3.zig");
const exlr8_build = @import("zig/build/exlr8.zig");
const server_build = @import("zig/build/server.zig");
const golden_build = @import("zig/build/golden.zig");
const contract_build = @import("zig/build/kernel_contract.zig");
const video_build = @import("zig/build/video.zig");
const sys_build = @import("zig/build/sys.zig");

/// Add the WP9 module graph to WP12's native entry point without changing WP9-owned build helpers.
fn glm53NativeImports(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, lanes: *std.Build.Module, native: *std.Build.Module) void {
    const file = struct {
        fn module(bld: *std.Build, t: std.Build.ResolvedTarget, path: []const u8) *std.Build.Module {
            return bld.createModule(.{ .root_source_file = bld.path(path), .target = t, .optimize = .Debug, .link_libc = true });
        }
    }.module;
    const dir = "zig/src/families/glm_dsa/cuda/";
    const cfg = file(b, target, "zig/src/families/glm_dsa/config.zig");
    const tokenizer = file(b, target, "zig/src/core/tokenizer/tokenizer.zig");
    const living = file(b, target, "zig/src/living/living.zig");
    const core = file(b, target, "zig/src/core/root.zig");
    core.addImport("gpu_memory", @import("zig/build/gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", living);
    const types = file(b, target, dir ++ "types.zig");
    const glm_follow = file(b, target, dir ++ "follow.zig");
    glm_follow.addImport("cuda", cuda);
    const weights = file(b, target, dir ++ "weights.zig");
    const state = file(b, target, dir ++ "state.zig");
    const forward = file(b, target, dir ++ "forward.zig");
    const taps = file(b, target, dir ++ "taps.zig");
    taps.addImport("cuda", cuda);
    forward.addImport("taps.zig", taps);
    const prefill_gemm = file(b, target, dir ++ "prefill_gemm.zig");
    const ablate = file(b, target, dir ++ "ablate.zig");
    ablate.addImport("cuda", cuda);
    ablate.addImport("core", core);
    const engine = file(b, target, dir ++ "engine.zig");
    const sampler = file(b, target, dir ++ "sampler.zig");
    const mla = file(b, target, dir ++ "mla.zig");
    const index = file(b, target, dir ++ "index.zig");
    const moe = file(b, target, dir ++ "moe.zig");
    const mlp = file(b, target, dir ++ "mlp.zig");
    const experts = file(b, target, dir ++ "experts.zig");
    const share = file(b, target, dir ++ "share.zig");
    const uma = file(b, target, dir ++ "uma.zig");
    types.addImport("cuda_memory", cuda);
    mla.addImport("types", types);
    mla.addCSourceFile(.{ .file = b.path("zig/kernels/cuda/glm_dsa/mla_index_host.c"), .flags = &.{} });
    state.addImport("types", types);
    forward.addImport("types", types);
    forward.addImport("prefill_gemm", prefill_gemm);
    inline for (.{ weights, state, forward, engine, sampler, mla, index, moe, mlp, experts, uma, prefill_gemm }) |m| m.addImport("cuda", cuda);
    share.addImport("config", cfg);
    index.addImport("glm_config", cfg);
    moe.addImport("mlp.zig", mlp);
    experts.addImport("core", core);
    state.addImport("mla.zig", mla);
    state.addImport("index.zig", index);
    state.addImport("sampler.zig", sampler);
    weights.addImport("core", core);
    weights.addImport("glm_config", cfg);
    weights.addImport("share.zig", share);
    weights.addImport("experts.zig", experts);
    weights.addImport("mlp.zig", mlp);
    weights.addImport("state.zig", state);
    weights.addImport("uma.zig", uma);
    inline for (.{ .{ "index.zig", index }, .{ "mla.zig", mla }, .{ "moe.zig", moe }, .{ "mlp.zig", mlp }, .{ "experts.zig", experts }, .{ "sampler.zig", sampler }, .{ "weights.zig", weights }, .{ "state.zig", state }, .{ "glm_config", cfg }, .{ "ablate", ablate } }) |imp| forward.addImport(imp[0], imp[1]);
    engine.addImport("forward.zig", forward);
    engine.addImport("ablate", ablate);
    engine.addImport("sampler.zig", sampler);
    engine.addImport("state.zig", state);
    engine.addImport("uma.zig", uma);
    engine.addImport("weights.zig", weights);
    // MK8 deferred launch (TF_GLM53_MK=1), as in zig/build/cuda.zig.
    const mk_launch = file(b, target, dir ++ "megakernel/launch.zig");
    mk_launch.addImport("cuda", cuda);
    const mk_defer = file(b, target, dir ++ "megakernel/defer.zig");
    mk_defer.addImport("cuda", cuda);
    mk_defer.addImport("mk_launch", mk_launch);
    engine.addImport("mk_defer", mk_defer);
    inline for (.{ .{ "engine", engine }, .{ "forward", forward }, .{ "ablate", ablate }, .{ "weights", weights }, .{ "state", state }, .{ "uma", uma }, .{ "glm_config", cfg }, .{ "glm_sampler", sampler }, .{ "glm_follow", glm_follow }, .{ "core", core }, .{ "cuda", cuda }, .{ "lanes", lanes } }) |imp| native.addImport(imp[0], imp[1]);
}

comptime {
    const required = std.mem.trim(u8, @embedFile(".zig-version"), "\r\n");
    if (!std.mem.eql(u8, builtin.zig_version_string, required))
        @compileError("TensorFold's Zig engine requires Zig " ++ required);
}

const metal_lists = @import("zig/build/metal_source_lists.zig");
const kernels = metal_lists.kernels;
const generated = metal_lists.generated;
const kimi_dir = metal_lists.kimi_dir;
const kimi_kernels = metal_lists.kimi_kernels;
const nemotron_packed = metal_lists.nemotron_packed;

/// MLX's language version for mx.fast.metal_kernel on this macOS: 4.1 from 27, 4.0 on 26, 3.2 on 15, else 3.1.
const mlxMetalStd = metal_build.mlxMetalStd;

/// xcrun metal -c each source to .air with `flags`, then link them into `name`.metallib.
const metallib = metal_build.metallib;

/// The Kimi metallib: each source compiled with kimi_dir on the include path, the shared header a dependency.
const kimiMetallib = metal_build.kimiMetallib;

/// A module whose `bytes` is the metallib at `lib`, embedded at compile time.
const embedded = metal_build.embedded;

/// The page pool's own tests, and its checks under engine_api's paged store and lane host (kv_pool a module of its...
fn poolChecks(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const pool = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool.zig"), .target = target, .optimize = .Debug });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const checks = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool_checks.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "kv_pool", .module = pool }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes } } });
    const step = b.step("test-kv-pool", "The KV page pool: layout, leases, injected failures, the paged store and lane-host admission");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pool })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = checks })).step);
    test_step.dependOn(step);
}

pub fn build(b: *std.Build) void {
    @import("zig/build/lane_halt_host.zig").steps(b);
    @import("zig/build/dsv41_shared.zig").steps(b);
    @import("zig/build/dsv41_cuda.zig").steps(b);
    @import("zig/build/dsv41_state.zig").steps(b);
    @import("zig/build/dsv41_draft.zig").steps(b);
    const target = b.standardTargetOptions(.{});
    @import("zig/build/copy_source.zig").steps(b);
    const release_version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH");
    if (release_version) |v| if (!dist_build.validVersion(v)) @panic("-Dversion must be exactly MAJOR.MINOR.PATCH");
    const build_options = b.addOptions();
    // Keep the existing manifest version for developer builds; releases require an explicit override.
    const manifest = @import("build.zig.zon");
    build_options.addOption([]const u8, "version", release_version orelse manifest.version);
    const slide_test_hooks = b.option(bool, "slide-test-hooks", "Living Weights test hooks, such as TF_SLIDE_FORGET_FAIL (never in a release)") orelse false;
    if (slide_test_hooks and release_version != null) @panic("-Dslide-test-hooks is for test builds: a -Dversion release never carries it");
    build_options.addOption(bool, "slide_test_hooks", slide_test_hooks);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    // Nemotron's draft vocabulary (zig/src/families/nemotron/draft_ids.txt), for every backend
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const test_step = b.step("test", "Unit tests (macOS includes available Metal device checks)");
    @import("zig/build/glm53_wp1.zig").steps(b, test_step);
    @import("zig/build/glm53_mk.zig").steps(b, test_step);
    @import("zig/build/glm53_wp9.zig").steps(b);
    @import("zig/build/glm53_wp3.zig").steps(b, test_step);
    @import("zig/build/glm53_wp5.zig").steps(b);
    @import("zig/build/glm53_wp10.zig").steps(b, test_step);
    @import("zig/build/glm53_mk1.zig").steps(b, test_step);
    @import("zig/build/kernel_symbols.zig").tests(b, test_step);
    if (b.graph.host.result.os.tag == .macos) {
        const step5_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const empty = b.addWriteFiles().add("step5-compile-only.metallib", "");
        step5_mods.sources.addImport("nemotron_packed_metallib", embedded(b, empty, "nemotron_packed.metallib"));
        @import("zig/build/step5_host.zig").steps(b, step5_mods.metal, step5_mods.engine, step5_mods.lanes);
    }
    @import("zig/build/h3_host.zig").tests(b, target, test_step);
    @import("zig/build/digest.zig").steps(b, test_step, draft_ids, build_options);
    @import("zig/build/flashnext_budget.zig").steps(b, test_step);
    @import("zig/build/flashnext_batch_host.zig").tests(b, test_step);
    if (b.graph.host.result.os.tag == .macos) {
        const budget_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const budget_empty = b.addWriteFiles().add("budget-compile-only.metallib", "");
        budget_mods.sources.addImport("nemotron_packed_metallib", embedded(b, budget_empty, "nemotron_packed.metallib"));
        @import("zig/build/flashnext_budget.zig").compileHost(b, budget_mods.metal, budget_mods.engine, budget_mods.lanes);
    }
    // every tracked path checks out on Windows too
    const names_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/portable_names.zig"), .target = target }) }));
    names_run.setCwd(b.path("."));
    test_step.dependOn(&names_run.step);
    // the XPU server's stream admission, a pure function of the card's memory
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu_admit.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/safetensors.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/limits.zig"), .target = target }) })).step);
    const cost_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/cost_rule_cases.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const cost_run = b.addRunArtifact(cost_cases);
    b.step("test-cost-rule", "CPU scripted window and interleaved head timing").dependOn(&cost_run.step);
    test_step.dependOn(&cost_run.step);
    merge_build.targets(b, target, optimize); // tensorfold merge's tests and tf-merge-check, host only on every OS
    switch (target.result.os.tag) {
        .macos => metalTargets(b, target, optimize, draft_ids, build_options, test_step),
        .linux, .windows => cuda_build.targets(b, target, optimize, draft_ids, build_options),
        else => {},
    }
    const gemma4_host_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host });
    gemma4_host_sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("gemma4-host-empty.metallib", "") });
    const gemma4_contract_host = @import("zig/build/gemma4_host.zig").tests(b, gemma4_host_sources);
    if (b.graph.host.result.os.tag == .macos) {
        const gemma4_host_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        gemma4_host_mods.sources.addImport("nemotron_packed_metallib", embedded(b, b.addWriteFiles().add("gemma4-compile-empty.metallib", ""), "nemotron_packed.metallib"));
        @import("zig/build/gemma4_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes, gemma4_host_mods.sources);
        @import("zig/build/qwen27_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes);
    }
    @import("zig/build/qwen27_host.zig").tests(b);
    const xgrammar = grammar_build.option(b);
    dist_build.targets(b, draft_ids, build_options, release_version, xgrammar);
    @import("zig/build/sessions.zig").add(b, test_step);
    @import("zig/build/tp.zig").add(b, target, optimize, test_step);
    @import("zig/build/gradient_host.zig").add(b);
    @import("zig/build/living_tools_host.zig").steps(b);
    cuda_build.hostTests(b, draft_ids, build_options, test_step, target.result.os.tag != .macos);
    const glm53_cuda_host = cuda_build.runtime(b, b.graph.host, .Debug, &.{});
    const glm53_config_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/config.zig"), .target = b.graph.host, .optimize = .Debug });
    const glm53_index_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/index.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{ .{ .name = "cuda", .module = glm53_cuda_host }, .{ .name = "glm_config", .module = glm53_config_host } },
    }) }));
    const glm53_index_step = b.step("test-glm53-index-host", "GLM-5.3 DSA indexer references and CUDA launch plans without a GPU");
    glm53_index_step.dependOn(&glm53_index_host.step);
    const glm53_index_parity_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_index_tests.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const glm53_index_api = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/index.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = glm53_cuda_host }, .{ .name = "glm_config", .module = glm53_config_host } } });
    glm53_index_parity_module.addImport("cuda", glm53_cuda_host);
    glm53_index_parity_module.addImport("index.zig", glm53_index_api);
    glm53_index_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm53_index_parity_module })).step);
    test_step.dependOn(glm53_index_step);
    const glm53_mla_test = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/mla.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }},
    }) });
    const glm53_mla_types = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/types.zig"), .target = b.graph.host, .optimize = .Debug, .imports = &.{.{ .name = "cuda_memory", .module = glm53_cuda_host }} });
    glm53_mla_test.root_module.addImport("types", glm53_mla_types);
    glm53_mla_test.root_module.addCSourceFile(.{ .file = b.path("zig/kernels/cuda/glm_dsa/mla_index_host.c"), .flags = &.{} });
    const glm53_mla_host = b.addRunArtifact(glm53_mla_test);
    const glm53_mla_step = b.step("test-glm53-mla-host", "GLM-5.3 MLA references and CUDA launch plans without a GPU");
    glm53_mla_step.dependOn(&glm53_mla_host.step);
    const glm53_mla_parity_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_mla_tests.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const glm53_mla_api = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/mla.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }} });
    const glm53_mla_parity_types = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/types.zig"), .target = b.graph.host, .optimize = .Debug, .imports = &.{.{ .name = "cuda_memory", .module = glm53_cuda_host }} });
    glm53_mla_api.addImport("types", glm53_mla_parity_types);
    glm53_mla_parity_module.addImport("cuda", glm53_cuda_host);
    glm53_mla_parity_module.addImport("mla.zig", glm53_mla_api);
    glm53_mla_parity_module.addImport("types", glm53_mla_parity_types);
    glm53_mla_parity_module.addCSourceFile(.{ .file = b.path("zig/kernels/cuda/glm_dsa/mla_index_host.c"), .flags = &.{} });
    glm53_mla_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm53_mla_parity_module })).step);
    test_step.dependOn(glm53_mla_step);
    const glm53_moe_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/moe.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }},
    }) }));
    const glm53_mlp_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/mlp.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }},
    }) }));
    const glm53_moe_step = b.step("test-glm53-moe-host", "GLM-5.3 MoE, low-rank and dense MLP references without a GPU");
    glm53_moe_step.dependOn(&glm53_moe_host.step);
    glm53_moe_step.dependOn(&glm53_mlp_host.step);
    const glm53_moe_parity_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_moe_tests.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const glm53_mlp_parity_api = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/mlp.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }} });
    const glm53_moe_parity_api = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/moe.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    glm53_moe_parity_api.addImport("cuda", glm53_cuda_host);
    glm53_moe_parity_api.addImport("mlp.zig", glm53_mlp_parity_api);
    glm53_moe_parity_module.addImport("cuda", glm53_cuda_host);
    glm53_moe_parity_module.addImport("moe.zig", glm53_moe_parity_api);
    glm53_moe_parity_module.addImport("mlp.zig", glm53_mlp_parity_api);
    glm53_moe_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm53_moe_parity_module })).step);
    test_step.dependOn(glm53_moe_step);
    const glm53_fuse_host_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_fuse_tests.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }} });
    const glm53_fuse_host = b.addRunArtifact(b.addTest(.{ .root_module = glm53_fuse_host_module }));
    const glm53_fuse_step = b.step("test-glm53-fuse-host", "GLM-5.3 fusion references and declaration analysis without a GPU");
    glm53_fuse_step.dependOn(&glm53_fuse_host.step);
    test_step.dependOn(glm53_fuse_step);
    const glm53_pages_step = b.step("test-glm53-pages-host", "GLM-5.3 DSA pages, cache continuity, follower protocol and rank agreement on the CPU");
    inline for (.{ "pages", "radix", "lanes", "follow", "ranks" }) |part| {
        const module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/" ++ part ++ ".zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
        if (comptime std.mem.eql(u8, part, "follow")) module.addImport("cuda", glm53_cuda_host);
        glm53_pages_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module })).step);
    }
    const glm53_kernel_options = b.addOptions();
    glm53_kernel_options.addOption(bool, "with_kernels", false);
    const glm53_kernel_images = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/kernels.zig"), .target = b.graph.host, .optimize = .Debug });
    glm53_kernel_images.addOptions("kernel_options", glm53_kernel_options);
    glm53_pages_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm53_kernel_images })).step);
    const glm53_backend_test = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_dsa/cuda/backend.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }},
    }) });
    glm53_backend_test.root_module.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true }));
    glm53_backend_test.root_module.addImport("glm_sampler", b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/sampler.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }} }));
    glm53_backend_test.root_module.addImport("glm_follow", b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/follow.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = glm53_cuda_host }} }));
    glm53_pages_step.dependOn(&b.addRunArtifact(glm53_backend_test).step);
    const glm53_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const glm53_native = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/native.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = glm53_cuda_host }, .{ .name = "lanes", .module = glm53_lanes } } });
    glm53NativeImports(b, b.graph.host, glm53_cuda_host, glm53_lanes, glm53_native);
    const glm53_native_test = b.addRunArtifact(b.addTest(.{ .root_module = glm53_native }));
    glm53_pages_step.dependOn(&glm53_native_test.step);
    const glm53_linux = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu });
    const glm53_linux_cuda = cuda_build.runtime(b, glm53_linux, .Debug, &.{});
    const glm53_linux_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = glm53_linux, .optimize = .Debug, .link_libc = true });
    const glm53_linux_native = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_dsa/cuda/native.zig"), .target = glm53_linux, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = glm53_linux_cuda }, .{ .name = "lanes", .module = glm53_linux_lanes } } });
    glm53NativeImports(b, glm53_linux, glm53_linux_cuda, glm53_linux_lanes, glm53_linux_native);
    const glm53_open_check = b.addObject(.{ .name = "glm53-native-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm53_open_compile.zig"), .target = glm53_linux, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "glm53", .module = glm53_linux_native }} }) });
    _ = glm53_open_check.getEmittedBin();
    const glm53_open_step = b.step("compile-glm53-native-open", "Compile and host-smoke the GLM-5.3 DSA native open surface");
    glm53_open_step.dependOn(&glm53_open_check.step);
    glm53_open_step.dependOn(&glm53_native_test.step);
    test_step.dependOn(glm53_pages_step);
    xpu_build.hostTests(b, test_step);
    if (target.result.os.tag == .linux or target.result.os.tag == .windows) {
        // the engine seam's tests (lane_host.zig and its siblings), which the macOS branch runs from its own api module
        const lanes_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
        const api_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_mod }} });
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api_mod })).step);
    }
    poolChecks(b, b.graph.host, test_step);
    hip_build.steps(b, target, optimize, build_options, test_step);
    exlr8_build.hostTests(b, target, test_step);
    xpu_build.steps(b, target, optimize, build_options);
    video_build.steps(b, target);
    const lanes_cpu = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const disk_cpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_disk_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const disk_run = b.addRunArtifact(disk_cpu);
    test_step.dependOn(&disk_run.step);
    b.step("test-learn-recovery", "CPU learned-state recovery and admission tests").dependOn(&disk_run.step);
    @import("zig/build/qwen27_required.zig").tests(b, lanes_cpu);
    glm_expiry_build.steps(b, test_step, lanes_cpu);
    const warning = b.addExecutable(.{ .name = "learned-warning-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/learned_warning_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const warning_run = b.addRunArtifact(warning);
    test_step.dependOn(&warning_run.step);
    b.step("test-learn-warning", "CPU production-log refusal, backoff and recovery test").dependOn(&warning_run.step);
    contract_build.hostTests(b, gemma4_contract_host);
    sys_build.wire(b);
    grammar_build.steps(b, target, xgrammar, lanes_cpu, lanes_cpu.import_table.get("sys") orelse @panic("sys_build.wire did not wire lanes_cpu"));
    @import("zig/build/checkpoint_config.zig").wire(b);
    @import("zig/build/save_guard.zig").wire(b);
}

/// The native Metal server build is owned by the shared Metal target module.
const nativeServer = metal_build.nativeServer;

/// macOS: the metallib, the Metal runtime's test programs and the engine over Metal.
const metalTargets = metal_build.metalTargets;

/// The server and distribution use the same embedded-source Metal engine.
const metalEngineModules = metal_build.metalEngineModules;

pub const distMetalServer = metal_build.distMetalServer;

test {
    _ = dist_build;
}
