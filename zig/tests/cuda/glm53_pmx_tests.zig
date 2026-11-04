// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! P1 parity checks for the GLM-5.3 width-specialised EXL3 prompt kernels.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const quant_ref = @import("quant_ref_tests.zig");
const experts_f64 = @import("exl3_experts_f64_tests.zig");
const exl3e = cuda.exl3_experts;
const Rig = quant_ref.RigT;
const SLOTS = experts_f64.SLOTS;

test "GLM-5.3 PMX declarations are analyzed" { std.testing.refAllDecls(@This()); }

pub fn run(gpu: anytype) !void {
    if (!cuda.kernels.available or cuda.kernels.glm53_pmx.len == 0) return error.KernelsNotEmbedded;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm53_pmx);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    inline for ([_]usize{ 4, 6 }) |k2| {
        try decodeCase(gpu, module, stream, k2, false);
        try decodeCase(gpu, module, stream, k2, true);
    }
    // The router lives in the generic experts image; the EXL3 input rotation in the exl3_experts image.
    var router_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer router_mod.unload();
    var experts_mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
    defer experts_mod.unload();
    const router = try cuda.grouped.Router.resolve(router_mod);
    const rot_in = try experts_mod.function("tf_exl3_experts_rot_in");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x5301);
    var rig: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (rig.bufs.items) |*b| b.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const width_cases = [_]struct { name: []const u8, g: u32, u: u32 }{
        .{ .name = "all-4", .g = 4, .u = 4 },
        .{ .name = "all-6", .g = 6, .u = 6 },
        .{ .name = "gate4-up6", .g = 4, .u = 6 },
        .{ .name = "gate6-up4", .g = 6, .u = 4 },
    };
    const codebooks = [_]exl3e.Codebook{ .mcg, .mcg_e4m3 };
    inline for (width_cases) |width_case| {
        inline for (codebooks) |cb| {
            try guCase(&rig, module, experts_mod, router, rot_in, width_case.name, width_case.g, width_case.u, cb);
        }
    }
}

fn decodeCase(gpu: anytype, module: cuda.Module, stream: cuda.Stream, comptime k2: usize, comptime e4m3: bool) !void {
    const K = 512;
    const N = 256;
    const width = core.exl3_format.Width{ .halves = @intCast(k2) };
    const nwords = (K / 16) * (N / 16) * width.tileWords();
    const trellis = try gpu.gpa.alloc(i16, nwords);
    defer gpu.gpa.free(trellis);
    var seed: u32 = 0x530000 + @as(u32, @intCast(k2 * 17 + @intFromBool(e4m3)));
    for (trellis) |*word| { seed = seed *% 1664525 +% 1013904223; word.* = @bitCast(@as(u16, @truncate(seed >> 8))); }
    const symbol = comptime std.fmt.comptimePrint("tf_glm53_pmx_dequant_{d}_{s}", .{ k2, if (e4m3) "e4m3" else "mcg" });
    const kernel = try module.function(symbol);
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(trellis));
    defer dt.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, K * N * 2);
    defer dout.free();
    var args: cuda.Args = .{};
    args.add(dt.ptr); args.add(dout.ptr); args.add(@as(i32, N));
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = K / 16, .y = N / 16, .z = 1 }, .block = .{ .x = 32 } }, stream, &args);
    try stream.synchronize();
    const got_bytes = try download(gpu, dout);
    defer gpu.gpa.free(got_bytes);
    const got = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, got_bytes));
    const oracle_cb: core.exl3_format.Codebook = .mcg;
    const expected = try gpu.gpa.alloc(f16, K * N);
    defer gpu.gpa.free(expected);
    core.exl3_format.unpack(width, oracle_cb, trellis, K / 16, N / 16, expected);
    for (expected, 0..) |*value, i| {
        if (e4m3) value.* = @floatCast(cuda.exl3.roundE4m3(@floatCast(value.*)));
        const want: u16 = @bitCast(value.*);
        if (got[i] != want) {
            std.debug.print("FAIL glm53.pmx.decode k2={d} {s}: first mismatch at {d}, got 0x{x:0>4}, want 0x{x:0>4}\n", .{ k2, if (e4m3) "e4m3" else "mcg", i, got[i], want });
            return error.TestFailed;
        }
    }
    std.debug.print("PASS glm53.pmx.decode k2={d} {s}\n", .{ k2, if (e4m3) "e4m3" else "mcg" });
}

fn guCase(r: *Rig, pmx_mod: cuda.Module, experts_mod: cuda.Module, router: cuda.grouped.Router, rot_in: cuda.Function, name: []const u8, comptime gh: u32, comptime uh: u32, comptime cb: exl3e.Codebook) !void {
    const E = experts_f64.E;
    const all_g: [E]u32 = @splat(gh);
    const all_u: [E]u32 = @splat(uh);
    const all_d: [E]u32 = @splat(4);
    const bits: [E]u32 = @splat(gh / 2);
    const built = try experts_f64.buildLayerWith(r, &bits, &all_g, &all_u, &all_d, cb);
    const layer = built.layer;
    const symbol = if (cb == .mcg) "tf_glm53_pmx_gu_mcg" else "tf_glm53_pmx_gu_e4m3";
    const gu = try pmx_mod.function(symbol);
    try gu.allowDynamicShared(exl3e.pmx.gu_smem);
    const small_symbol = if (cb == .mcg) "tf_exl3_prompt_gu_s" else "tf_exl3_prompt_gu_s_e4m3";
    const gu_small = try experts_mod.function(small_symbol);
    const cbname = if (cb == .mcg) "mcg" else "mcg_e4m3";
    const plan_mem = try r.zeros(try exl3e.pmx.bytes(layer, 300, SLOTS));
    const plan = try exl3e.pmx.scratch(plan_mem, layer, 300, SLOTS);
    for ([_]usize{ 1, 17, 128, 300 }) |rows| {
        try router.route(r.s, built.dpick, rows * SLOTS, E, exl3e.pmx.bm, .{ .members = plan.members, .items = plan.items, .counts = plan.counts, .rank = plan.rank, .hist = plan.hist });
        r.s.synchronize() catch |err| { std.debug.print("FAIL glm53.pmx.gu {s}/{s} rows={d}: router fault\n", .{ name, cbname, rows }); return err; };
        var rot: cuda.Args = .{};
        rot.add(built.dx); rot.add(@as(i32, experts_f64.D)); rot.add(@as(i32, @intFromEnum(exl3e.DType.bf16)));
        rot.add(built.dpick); rot.add(layer.suh_g); rot.add(layer.suh_u); rot.add(plan.xg); rot.add(plan.xu);
        rot.add(@as(i32, experts_f64.D)); rot.add(@as(i32, SLOTS)); rot.add(@as(i32, E));
        try cuda.launch.launch(rot_in, .{ .grid = .{ .x = @intCast(rows * SLOTS), .y = 4, .z = 2 }, .block = .{ .x = 32 } }, r.s, &rot);
        r.s.synchronize() catch |err| { std.debug.print("FAIL glm53.pmx.gu {s}/{s} rows={d}: rot_in fault\n", .{ name, cbname, rows }); return err; };
        const items = cuda.grouped.maxItems(rows * SLOTS, E, exl3e.prompt.wide_tile);
        var small_args: cuda.Args = .{};
        inline for (.{
            plan.xg, plan.xu, layer.gate, layer.up, plan.items, plan.counts, plan.members,
            layer.svh_g, layer.svh_u, layer.suh_d, plan.xd,
            @as(i32, @intCast(layer.dims)), @as(i32, @intCast(layer.width)), @as(i32, 1),
            layer.limit, layer.act_mode, layer.gate_k2, layer.up_k2,
        }) |v| small_args.add(v);
        try cuda.launch.launch(gu_small, .{ .grid = .{ .x = @intCast(layer.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, r.s, &small_args);
        var args: cuda.Args = .{};
        args.add(exl3e.pmx.PmxArgs{
            .x0 = plan.xg, .x1 = plan.xu, .tp0 = layer.gate, .tp1 = layer.up,
            .k2_0 = layer.gate_k2, .k2_1 = layer.up_k2, .items = plan.items, .counts = plan.counts, .members = plan.members,
            .sv0 = layer.svh_g, .sv1 = layer.svh_u, .sd = layer.suh_d, .xd = plan.xd, .y = plan.y,
            .K = @intCast(layer.dims), .N = @intCast(layer.width), .E = @intCast(E), .limit = layer.limit, .act_mode = layer.act_mode,
        });
        try cuda.launch.launch(gu, .{ .grid = .{ .x = @intCast(layer.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @intCast(exl3e.pmx.gu_smem) }, r.s, &args);
        r.s.synchronize() catch |err| { std.debug.print("FAIL glm53.pmx.gu {s}/{s} rows={d}: gu fault\n", .{ name, cbname, rows }); return err; };
        var row_arena = std.heap.ArenaAllocator.init(r.a);
        defer row_arena.deinit();
        const old_allocator = r.a;
        r.a = row_arena.allocator();
        const ref = try referencePmx(r, &built, rows, gh, uh);
        r.a = old_allocator;
        const raw = try download(r.gpu, .{ .d = r.gpu.d, .ptr = plan.xd, .len = rows * SLOTS * experts_f64.I * 2 });
        defer r.gpu.gpa.free(raw);
        const got = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, raw));
        var largest: f64 = 0;
        var at: usize = 0;
        for (ref.xd, ref.xderr, 0..) |want, tol, i| {
            const actual: f64 = @floatCast(@as(f16, @bitCast(got[i])));
            const ratio = @abs(actual - want) / tol;
            if (!std.math.isFinite(ratio)) {
                std.debug.print("FAIL glm53.pmx.gu {s}/{s} rows={d}: non-finite ratio at element {d}, actual={d}, want={d}, tol={d}\n", .{ name, cbname, rows, i, actual, want, tol });
                return error.TestFailed;
            }
            if (ratio > largest) { largest = ratio; at = i; }
        }
        std.debug.print("RESULT glm53.pmx.gu {s}/{s} rows={d}: largest difference {d:.3} of its tolerance (element {d})\n", .{ name, cbname, rows, largest, at });
        if (largest > 1.0) {
            std.debug.print("FAIL glm53.pmx.gu {s}/{s} rows={d}\n", .{ name, cbname, rows });
            return error.TestFailed;
        }
        std.debug.print("PASS glm53.pmx.gu {s}/{s} rows={d}\n", .{ name, cbname, rows });
    }
}

/// f64 Experts reference through xd, using xu for up and PMX's per-projection widths/codebook rounding.
fn referencePmx(r: *Rig, b: *const experts_f64.Built, rows_n: usize, gh: u32, uh: u32) !struct { xd: []f64, xderr: []f64 } {
    const E = experts_f64.E;
    const D = experts_f64.D;
    const I = experts_f64.I;
    const p = rows_n * SLOTS;
    const fmt = core.exl3_format;
    const bf = quant_ref.helpers.bfOf;
    const f16v = struct { fn f(v: u16) f64 { return @floatCast(@as(f16, @bitCast(v))); } }.f;
    const bf16r = struct { fn f(v: f64) f64 { const u: u32 = @bitCast(@as(f32, @floatCast(v))); return bf(@intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16)); } }.f;
    const xsg = try r.a.alloc(f64, p * D);
    const xsu = try r.a.alloc(f64, p * D);
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        for (0..D) |i| {
            const x = bf(b.x[(s0 / SLOTS) * D + i]);
            xsg[s0 * D + i] = x * f16v(b.suh_g[ex * D + i]);
            xsu[s0 * D + i] = x * f16v(b.suh_u[ex * D + i]);
        }
    }
    fmt.rotate(xsg, p, D, 1); fmt.rotate(xsu, p, D, 1);
    const xg = try r.a.alloc(f16, p * D); const xu = try r.a.alloc(f16, p * D);
    const xge = try r.a.alloc(f64, p * D); const xue = try r.a.alloc(f64, p * D);
    for (0..p * D) |i| { xg[i] = @floatCast(xsg[i]); xu[i] = @floatCast(xsu[i]); xge[i] = @abs(xsg[i]); xue[i] = @abs(xsu[i]); }
    absRotateCols(xge, p, D); absRotateCols(xue, p, D);
    for (xge) |*v| v.* *= 1.0 / 2048.0 + 8.0 / 16777216.0;
    for (xue) |*v| v.* *= 1.0 / 2048.0 + 8.0 / 16777216.0;
    const zg = try r.a.alloc(f64, I * p); const zu = try r.a.alloc(f64, I * p);
    const sg = try r.a.alloc(f64, I * p); const su = try r.a.alloc(f64, I * p);
    const eg = try r.a.alloc(f64, I * p); const eu = try r.a.alloc(f64, I * p);
    @memset(zg, 0); @memset(zu, 0); @memset(sg, 0); @memset(su, 0); @memset(eg, 0); @memset(eu, 0);
    const wq = try r.a.alloc(f16, D * I);
    for (0..E) |ex| {
        try decodeProjection(b, ex, true, gh, cbFromLayer(b), wq);
        gemmPmx(xg, wq, xge, zg, sg, eg, b.pick, ex, D, I, p);
        try decodeProjection(b, ex, false, uh, cbFromLayer(b), wq);
        gemmPmx(xu, wq, xue, zu, su, eu, b.pick, ex, D, I, p);
    }
    for (eg, sg) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    for (eu, su) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    fmt.rotate(zg, I, p, 0); fmt.rotate(zu, I, p, 0);
    absRotate(eg, I, p); absRotate(eu, I, p);
    const zge = try r.a.alloc(f64, I * p); const zue = try r.a.alloc(f64, I * p);
    for (zge, zg) |*o, v| o.* = @abs(v); for (zue, zu) |*o, v| o.* = @abs(v);
    absRotate(zge, I, p); absRotate(zue, I, p);
    for (zge) |*v| v.* *= 8.0 / 16777216.0; for (zue) |*v| v.* *= 8.0 / 16777216.0;
    for (eg, zge) |*e, v| e.* += v; for (eu, zue) |*e, v| e.* += v;
    const v = try r.a.alloc(f64, I * p); const verr = try r.a.alloc(f64, I * p);
    for (0..I) |n| for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        const vg = zg[n * p + s0] * f16v(b.svh_g[ex * I + n]);
        const gg = @min(bf16r(vg), @as(f64, 10));
        const vu = zu[n * p + s0] * f16v(b.svh_u[ex * I + n]);
        const uu = @min(@max(bf16r(vu), -@as(f64, 10)), @as(f64, 10));
        const inner = bf16r(gg / (1.0 + std.math.exp(-gg)));
        const act = bf16r(inner * uu);
        v[n * p + s0] = act * f16v(b.suh_d[ex * I + n]);
        const egk = eg[n * p + s0] * @abs(f16v(b.svh_g[ex * I + n])) + (@abs(vg) + @abs(gg)) / 256;
        const euk = eu[n * p + s0] * @abs(f16v(b.svh_u[ex * I + n])) + (@abs(vu) + @abs(uu)) / 256;
        const einner = egk + (@abs(gg) + 1) / 524288 + @abs(inner) / 256;
        verr[n * p + s0] = (@abs(uu) * einner + @abs(inner) * euk + @abs(act) / 256) * @abs(f16v(b.suh_d[ex * I + n])) + @abs(v[n * p + s0]) / 16777216;
    };
    fmt.rotate(v, I, p, 0); absRotate(verr, I, p);
    const vrot = try r.a.alloc(f64, I * p);
    for (vrot, v) |*o, value| o.* = @abs(value);
    absRotate(vrot, I, p); for (vrot) |*o| o.* *= 8.0 / 16777216.0;
    const xd = try r.a.alloc(f64, p * I); const xderr = try r.a.alloc(f64, p * I);
    for (0..p) |s0| for (0..I) |ki| {
        xd[s0 * I + ki] = @floatCast(@as(f16, @floatCast(v[ki * p + s0])));
        xderr[s0 * I + ki] = verr[ki * p + s0] + vrot[ki * p + s0] + @abs(xd[s0 * I + ki]) / 2048;
    };
    return .{ .xd = xd, .xderr = xderr };
}

fn cbFromLayer(b: *const experts_f64.Built) exl3e.Codebook { return b.layer.codebook; }

fn decodeProjection(b: *const experts_f64.Built, ex: usize, gate: bool, k2: u32, cb: exl3e.Codebook, out: []f16) !void {
    const trellis = if (gate) b.gate[ex] else b.up[ex];
    const base: core.exl3_format.Codebook = switch (cb) {
        .inst3 => .inst3,
        .mcg, .mcg_e4m3 => .mcg,
        .mul1 => .mul1,
    };
    core.exl3_format.unpack(.{ .halves = @intCast(k2) }, base, trellis, experts_f64.D / 16, experts_f64.I / 16, out);
    if (cb == .mcg_e4m3) {
        for (out) |*v| {
            v.* = @floatCast(cuda.exl3.roundE4m3(@floatCast(v.*)));
        }
    }
}

fn gemmPmx(x: []const f16, w: []const f16, xe: []const f64, z: []f64, s1: []f64, e: []f64, picks: []const i32, ex: usize, k: usize, n: usize, p: usize) void {
    for (0..p) |s0| {
        if (picks[s0] != @as(i32, @intCast(ex))) continue;
        for (0..k) |ki| {
            const t: f64 = @floatCast(x[s0 * k + ki]); const ta = @abs(t); const xek = xe[s0 * k + ki];
            for (0..n) |col| { const wv: f64 = @floatCast(w[ki * n + col]); z[col * p + s0] += t * wv; s1[col * p + s0] += ta * @abs(wv); e[col * p + s0] += @abs(wv) * xek; }
        }
    }
}

fn absRotate(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128));
    var row: usize = 0;
    while (row < nrows) : (row += 128) {
        for (0..cols) |c| { var sum: f64 = 0; for (0..128) |n| sum += @abs(x[(row + n) * cols + c]); sum *= inv; for (0..128) |n| x[(row + n) * cols + c] = sum; }
    }
}

fn absRotateCols(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128));
    for (0..nrows) |row| {
        var col: usize = 0;
        while (col < cols) : (col += 128) { var sum: f64 = 0; for (0..128) |n| sum += @abs(x[row * cols + col + n]); sum *= inv; for (0..128) |n| x[row * cols + col + n] = sum; }
    }
}

fn download(gpu: anytype, buffer: cuda.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, buffer.len);
    errdefer gpu.gpa.free(out);
    try buffer.download(0, out);
    return out;
}
