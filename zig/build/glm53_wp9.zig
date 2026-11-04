// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! WP9 host checks for GLM-5.3: loader, state, forward binding and engine, one module per file.
const std = @import("std");
const cuda_build = @import("cuda.zig");

const Mods = struct {
    cuda: *std.Build.Module,
    core: *std.Build.Module,
    config: *std.Build.Module,
    weights: *std.Build.Module,
    state: *std.Build.Module,
    forward: *std.Build.Module,
    taps: *std.Build.Module,
    engine: *std.Build.Module,
    sampler: *std.Build.Module,
    mla: *std.Build.Module,
    index: *std.Build.Module,
    moe: *std.Build.Module,
    mlp: *std.Build.Module,
    experts: *std.Build.Module,
    share: *std.Build.Module,
    uma: *std.Build.Module,
    types: *std.Build.Module,
    prefill_gemm: *std.Build.Module,
    ablate: *std.Build.Module,
    kvsave: *std.Build.Module,
};

var opt: std.builtin.OptimizeMode = .Debug;

fn file(b: *std.Build, host: std.Build.ResolvedTarget, path: []const u8) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(path), .target = host, .optimize = opt, .link_libc = true });
}

fn modules(b: *std.Build) Mods {
    const host = b.graph.host;
    opt = .Debug;
    const cuda = cuda_build.runtime(b, host, .Debug, &.{});
    // The core module as cuda.zig's family() builds it (checkpoint, exl3_load, safetensors).
    const tokenizer = file(b, host, "zig/src/core/tokenizer/tokenizer.zig");
    const living = file(b, host, "zig/src/living/living.zig");
    const core = file(b, host, "zig/src/core/root.zig");
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, host));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", living);
    return modulesFor(b, host, cuda, core);
}

/// The WP9 module set over a given runtime and core (the host tests' own, or cuda.zig's with kernel images).
fn modulesFor(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module) Mods {
    const dir = "zig/src/families/glm_dsa/cuda/";
    const config = file(b, host, "zig/src/families/glm_dsa/config.zig");
    const m: Mods = .{
        .cuda = cuda,
        .core = core,
        .config = config,
        .weights = file(b, host, dir ++ "weights.zig"),
        .state = file(b, host, dir ++ "state.zig"),
        .forward = file(b, host, dir ++ "forward.zig"),
        .taps = file(b, host, dir ++ "taps.zig"),
        .engine = file(b, host, dir ++ "engine.zig"),
        .sampler = file(b, host, dir ++ "sampler.zig"),
        .mla = file(b, host, dir ++ "mla.zig"),
        .index = file(b, host, dir ++ "index.zig"),
        .moe = file(b, host, dir ++ "moe.zig"),
        .mlp = file(b, host, dir ++ "mlp.zig"),
        .experts = file(b, host, dir ++ "experts.zig"),
        .share = file(b, host, dir ++ "share.zig"),
        .uma = file(b, host, dir ++ "uma.zig"),
        .types = file(b, host, dir ++ "types.zig"),
        .prefill_gemm = file(b, host, dir ++ "prefill_gemm.zig"),
        .ablate = file(b, host, dir ++ "ablate.zig"),
        .kvsave = file(b, host, dir ++ "kvsave.zig"),
    };
    // Round 7: mla.zig and the shared StagePlanes contract import types.zig (as build.zig wires test-glm53-mla-host).
    m.types.addImport("cuda_memory", cuda);
    m.ablate.addImport("cuda", cuda);
    m.ablate.addImport("core", core);
    m.mla.addImport("types", m.types);
    m.mla.addCSourceFile(.{ .file = b.path("zig/kernels/cuda/glm_dsa/mla_index_host.c"), .flags = &.{} });
    m.state.addImport("types", m.types);
    m.forward.addImport("types", m.types);
    m.taps.addImport("cuda", cuda);
    m.forward.addImport("taps.zig", m.taps);
    m.forward.addImport("prefill_gemm", m.prefill_gemm);
    for ([_]*std.Build.Module{ m.weights, m.state, m.forward, m.engine, m.sampler, m.mla, m.index, m.moe, m.mlp, m.experts, m.uma, m.prefill_gemm }) |mod| mod.addImport("cuda", cuda);
    m.share.addImport("config", config);
    m.index.addImport("glm_config", config);
    m.moe.addImport("mlp.zig", m.mlp);
    m.experts.addImport("core", core);
    m.state.addImport("mla.zig", m.mla);
    m.state.addImport("index.zig", m.index);
    m.state.addImport("sampler.zig", m.sampler);
    m.weights.addImport("core", core);
    m.weights.addImport("glm_config", config);
    m.weights.addImport("share.zig", m.share);
    m.weights.addImport("experts.zig", m.experts);
    m.weights.addImport("mlp.zig", m.mlp);
    m.weights.addImport("state.zig", m.state);
    m.weights.addImport("uma.zig", m.uma);
    inline for (.{ .{ "index.zig", m.index }, .{ "mla.zig", m.mla }, .{ "moe.zig", m.moe }, .{ "mlp.zig", m.mlp }, .{ "experts.zig", m.experts }, .{ "sampler.zig", m.sampler }, .{ "weights.zig", m.weights }, .{ "state.zig", m.state }, .{ "glm_config", config }, .{ "ablate", m.ablate } }) |imp| m.forward.addImport(imp[0], imp[1]);
    m.engine.addImport("forward.zig", m.forward);
    m.engine.addImport("ablate", m.ablate);
    m.engine.addImport("sampler.zig", m.sampler);
    m.engine.addImport("state.zig", m.state);
    m.engine.addImport("uma.zig", m.uma);
    m.engine.addImport("weights.zig", m.weights);
    // MK8 deferred launch (TF_GLM53_MK=1), as cuda.zig's glm53Module wires it.
    const mk_launch = file(b, host, dir ++ "megakernel/launch.zig");
    mk_launch.addImport("cuda", cuda);
    const mk_defer = file(b, host, dir ++ "megakernel/defer.zig");
    mk_defer.addImport("cuda", cuda);
    mk_defer.addImport("mk_launch", mk_launch);
    m.engine.addImport("mk_defer", mk_defer);
    return m;
}

fn root(b: *std.Build, m: Mods, path: []const u8) *std.Build.Module {
    const r = file(b, b.graph.host, path);
    inline for (.{ "cuda", "core", "weights", "state", "forward", "engine", "sampler", "ablate" }) |name| r.addImport(name, @field(m, name));
    r.addImport("glm_config", m.config);
    r.addImport("uma", m.uma);
    r.addImport("prefill_gemm", m.prefill_gemm);
    r.addImport("kvsave", m.kvsave);
    r.addImport("dump", file(b, b.graph.host, "zig/src/families/glm_dsa/cuda/dump.zig"));
    // The bring-up tool imports prefix_snap (prefix cache Phase B); host test roots built from it need the module too.
    const psnap = file(b, b.graph.host, "zig/src/families/glm_dsa/cuda/prefix_snap.zig");
    const pcache = file(b, b.graph.host, "zig/src/families/glm_dsa/cuda/prefix_cache.zig");
    pcache.addImport("cuda", m.cuda);
    pcache.addImport("uma", m.uma);
    pcache.addImport("kvsave", m.kvsave);
    psnap.addImport("cuda", m.cuda);
    psnap.addImport("uma", m.uma);
    psnap.addImport("kvsave", m.kvsave);
    psnap.addImport("engine", m.engine);
    psnap.addImport("prefix_cache.zig", pcache);
    r.addImport("prefix_snap", psnap);
    return r;
}

/// tf-glm53-bringup (review r2 N1) over cuda.zig's runtime (with kernel images when built with -Dnvcc or
/// -Dfatbins). `zig build glm53-bringup` installs it; `zig build compile-glm53-bringup -Dtarget=aarch64-linux-gnu`
/// is the compile check from a Mac (no images).
pub fn bringup(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, glm: *std.Build.Module) void {
    opt = optimize;
    defer opt = .Debug;
    const m = modulesFor(b, target, cuda, core);
    const r = file(b, target, "zig/tools/glm53_bringup.zig");
    inline for (.{ "cuda", "weights", "state", "forward", "engine", "uma", "prefill_gemm", "kvsave" }) |name| r.addImport(name, @field(m, name));
    const psnap = file(b, target, "zig/src/families/glm_dsa/cuda/prefix_snap.zig");
    const pcache = file(b, target, "zig/src/families/glm_dsa/cuda/prefix_cache.zig");
    pcache.addImport("cuda", cuda);
    pcache.addImport("uma", m.uma);
    pcache.addImport("kvsave", m.kvsave);
    psnap.addImport("cuda", cuda);
    psnap.addImport("uma", m.uma);
    psnap.addImport("kvsave", m.kvsave);
    psnap.addImport("engine", m.engine);
    psnap.addImport("prefix_cache.zig", pcache);
    r.addImport("prefix_snap", psnap);
    r.addImport("core", core);
    r.addImport("sampler", m.sampler);
    r.addImport("dump", file(b, target, "zig/src/families/glm_dsa/cuda/dump.zig"));
    const drafts = draftsModule(b, target, cuda, glm);
    r.addImport("draft_admit", draftAdmit(b, target, cuda, glm, drafts));
    r.addImport("drafts", drafts);
    r.addImport("glm", glm);
    const exe = b.addExecutable(.{ .name = "tf-glm53-bringup", .root_module = r });
    b.step("glm53-bringup", "GLM-5.3 WP9 GPU bring-up driver (plan, load, run, resume, replay at TP2 + DCP2)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    b.step("compile-glm53-bringup", "Compile tf-glm53-bringup without installing or running it").dependOn(&exe.step);
    // tf-glm53-graph-replay: one GPU, a synthetic GLM-shaped rank on a loopback collective; keyed decode graphs on
    // vs off must agree bit for bit (installed with the other test runners).
    const g = file(b, target, "zig/tests/cuda/glm53_graph_replay_tests.zig");
    inline for (.{ "cuda", "weights", "state", "forward", "engine" }) |name| g.addImport(name, @field(m, name));
    g.addImport("ablate", m.ablate);
    const graph_exe = b.addExecutable(.{ .name = "tf-glm53-graph-replay", .root_module = g });
    b.installArtifact(graph_exe);
    b.step("compile-glm53-graph-replay", "Compile tf-glm53-graph-replay without installing or running it").dependOn(&graph_exe.step);
    const ablate_test = file(b, target, "zig/tests/cuda/glm53_ablate_tests.zig");
    ablate_test.addImport("cuda", cuda);
    ablate_test.addImport("ablate", m.ablate);
    const ablate_exe = b.addExecutable(.{ .name = "tf-glm53-ablate-tests", .root_module = ablate_test });
    b.installArtifact(ablate_exe);
    b.step("compile-glm53-ablate-tests", "Compile tf-glm53-ablate-tests without running it").dependOn(&ablate_exe.step);
    const draft_head_test = file(b, target, "zig/tests/cuda/glm53_draft_head_tests.zig");
    draft_head_test.addImport("cuda", cuda);
    draft_head_test.addImport("glm", glm);
    draft_head_test.addImport("drafts", draftsModule(b, target, cuda, glm));
    const draft_head_exe = b.addExecutable(.{ .name = "tf-glm53-draft-head-tests", .root_module = draft_head_test });
    b.installArtifact(draft_head_exe);
    b.step("compile-glm53-draft-head-tests", "Compile tf-glm53-draft-head-tests without running it").dependOn(&draft_head_exe.step);
    const bench_module = file(b, target, "zig/tools/glm53_bench.zig");
    const bench_exe = b.addExecutable(.{ .name = "tf-glm53-bench", .root_module = bench_module });
    b.step("glm53-bench", "GLM-5.3 prefill operation-class throughput estimate").dependOn(&b.addInstallArtifact(bench_exe, .{}).step);
}

pub fn steps(b: *std.Build) void {
    const m = modules(b);
    const probe_module = file(b, b.graph.host, "zig/src/tools/uma_alloc_probe.zig");
    probe_module.addImport("cuda", m.cuda);
    probe_module.addImport("uma", m.uma);
    const probe_exe = b.addExecutable(.{ .name = "tf-uma-alloc-probe", .root_module = probe_module });
    const probe_install = b.addInstallArtifact(probe_exe, .{});
    b.step("uma-alloc-probe", "Build the CUDA alloc versus VMM UMA overhead probe").dependOn(&probe_install.step);

    const engine_run = b.addRunArtifact(b.addTest(.{ .root_module = root(b, m, "zig/tests/glm53_cuda_engine_host.zig") }));
    const step = b.step("test-glm53-engine-host", "GLM-5.3 WP9 synthetic checkpoint load, memory plans, state and graph keys");
    step.dependOn(&engine_run.step);
    // Each WP9 module is also a test root so Zig collects its own test blocks.
    for ([_]*std.Build.Module{ m.weights, m.state, m.forward, m.engine, m.uma, m.prefill_gemm, m.ablate, m.kvsave }) |module| {
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module })).step);
    }
    const prewarm = file(b, b.graph.host, "zig/src/families/glm_dsa/cuda/prewarm.zig");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = prewarm })).step);
    const bench_module = file(b, b.graph.host, "zig/tools/glm53_bench.zig");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = bench_module })).step);
    const fp8_ref = file(b, b.graph.host, "zig/tests/glm53_prefill_fp8_host.zig");
    fp8_ref.addImport("cuda", m.cuda);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fp8_ref })).step);
    const recording_run = b.addRunArtifact(b.addTest(.{ .root_module = root(b, m, "zig/tests/glm53_cuda_forward_host.zig") }));
    const recording_step = b.step("test-glm53-forward-host", "GLM-5.3 CUDA forward through the real launch wrappers on a recording host driver");
    recording_step.dependOn(&recording_run.step);
    // Bring-up (design §7 L1 at TP2 + DCP2, review M2): layers 0-7 through the real wrappers on the recording driver.
    const bringup_run = b.addRunArtifact(b.addTest(.{ .root_module = root(b, m, "zig/tests/glm53_cuda_forward_host.zig"), .filters = &.{"bring-up"} }));
    const bringup_step = b.step("test-glm53-bringup-host", "GLM-5.3 TP2 bring-up over layers 0-7 (layer-subset forward, plans and contracts)");
    bringup_step.dependOn(&bringup_run.step);
}

/// The drafter admission module (zig/tools/glm53_draft_admit.zig) over Flash's glm family, which carries the DFlash2
/// loader and its kernels; `drafts` (GLM-5.3's DFlash2 geometry and memory plan) over the same glm and cuda.
fn draftAdmit(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, glm: *std.Build.Module, drafts: *std.Build.Module) *std.Build.Module {
    const m = file(b, target, "zig/tools/glm53_draft_admit.zig");
    m.addImport("cuda", cuda);
    m.addImport("glm", glm);
    m.addImport("drafts", drafts);
    return m;
}

fn draftsModule(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, glm: *std.Build.Module) *std.Build.Module {
    const geometry = file(b, target, "zig/src/families/glm_dsa/cuda/dflash_geometry.zig");
    const drafts = file(b, target, "zig/src/families/glm_dsa/cuda/drafts.zig");
    drafts.addImport("cuda", cuda);
    drafts.addImport("geometry", geometry);
    drafts.addImport("glm", glm);
    return drafts;
}

/// The bring-up driver's host tests need Flash's glm family (for the drafter admission), which only the CUDA host
/// build makes: cuda.zig's hostTests calls this with its host cuda, core and glm modules.
pub fn driverHost(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, glm: *std.Build.Module) void {
    const m = modulesFor(b, b.graph.host, cuda, core);
    const r = root(b, m, "zig/tools/glm53_bringup.zig");
    const drafts = draftsModule(b, b.graph.host, cuda, glm);
    const admit = draftAdmit(b, b.graph.host, cuda, glm, drafts);
    r.addImport("draft_admit", admit);
    r.addImport("drafts", drafts);
    r.addImport("glm", glm);
    const step = b.top_level_steps.get("test-glm53-bringup-host") orelse return;
    step.step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = r })).step);
    step.step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = admit })).step);
}
