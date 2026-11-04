// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! The EXL3 routed experts' float64 reference: the mirrored chain and each output's worst-case error bound.
const std = @import("std");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const cuda = @import("cuda");
const exl3fmt = @import("core").exl3_format;
const exl3e = cuda.exl3_experts;
const exl3t = @import("exl3_tests.zig");
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;
const Run = exl3t.Run;

pub const E = 9; // experts of a synthetic layer
pub const D = 512; // model width
pub const I = 256; // expert width
pub const SLOTS = 8;
const LIMIT: f32 = 10.0; // the SwiGLU clamp
const CAP = 300; // baseline rows of x, pick and wts the builders make
const rows = [_]usize{ 1, 3, 8, 16, 17, 64, 128 }; // the f64 checks' row counts (the reference builds 128 rows)

const f64_ids = [_][]const u8{ "exl3.experts.f64.mul1.4", "exl3.experts.f64.mul1.5", "exl3.experts.f64.mul1.6" };

/// One synthetic layer: the device tensors, the host trellises the reference unpacks, and the seeded routing.
pub const Built = struct {
    layer: exl3e.Layer,
    bits: []const u32,
    rows: usize,
    gh: []const u32,
    uh: []const u32,
    dh: []const u32,
    gate: [E][]i16,
    up: [E][]i16,
    down: [E][]i16,
    suh_g: []u16, // fp16 [E, D]
    suh_u: []u16,
    svh_g: []u16, // fp16 [E, I]
    svh_u: []u16,
    suh_d: []u16, // fp16 [E, I]
    svh_d: []u16, // fp16 [E, D]
    dx: u64,
    dpick: u64,
    dwts: u64,
    x: []u16, // bf16 [rows, D]
    pick: []i32, // [rows, SLOTS]
    wts: []f32, // [rows, SLOTS]
};

/// The f64 id checks: every width alone, then the mixed layer folded into the .6 id (the contract's ids).
pub fn checks(r: *Rig, exp: *const exl3e.Experts, state: *Run) void {
    for (f64_ids, 0..) |id, i| {
        var ok = true;
        f64Check(r, exp, &allBits(@intCast(i + 4))) catch |e| {
            state.fail(e, id);
            ok = false;
        };
        if (ok) check.pass("{s}", .{id});
    }
    {
        var ok = true;
        f64Check(r, exp, &.{ 4, 5, 6, 4, 5, 6, 4, 5, 6 }) catch |e| {
            state.fail(e, f64_ids[2]);
            ok = false;
        };
        if (ok) check.pass("{s}", .{f64_ids[2]});
    }
}

fn allBits(bits: u32) [E]u32 {
    return .{ bits, bits, bits, bits, bits, bits, bits, bits, bits };
}

pub fn widthOf(bits: u32) exl3fmt.Width {
    return .{ .halves = @intCast(2 * bits) };
}

fn halvesOf(halves: u32) exl3fmt.Width {
    return .{ .halves = @intCast(halves) };
}

fn unpackProjection(b: *const Built, bits2: u32, trellis: []const i16, k_tiles: usize, n_tiles: usize, out: []f16) void {
    const codebook: exl3fmt.Codebook = switch (b.layer.codebook) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
        .mcg_e4m3 => .mcg,
    };
    exl3fmt.unpack(halvesOf(bits2), codebook, trellis, k_tiles, n_tiles, out);
    if (b.layer.codebook == .mcg_e4m3) {
        for (out) |*v| v.* = @floatCast(cuda.exl3.roundE4m3(@floatCast(v.*)));
    }
}

fn f16v(v: u16) f64 {
    return @floatCast(@as(f16, @bitCast(v)));
}

/// bf16 round-to-nearest-even of an f64 (the kernel rounds its fp32 value; the double round is within the tolerance).
fn bf16r(v: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return bf(@intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16));
}

pub fn buildLayer(r: *Rig, bits: []const u32) !Built {
    var halves: [E]u32 = undefined;
    for (&halves, bits) |*h, v| h.* = 2 * v;
    return buildLayerWith(r, bits, &halves, &halves, &halves, .mul1);
}

/// `buildLayer` with each projection's half-bit width per expert and the codebook (the prompt path's mixed cases);
/// `bits` is only what the f64 checks print.
pub fn buildLayerWith(r: *Rig, bits: []const u32, gh: []const u32, uh: []const u32, dh: []const u32, codebook: exl3e.Codebook) !Built {
    return buildLayerWithRows(r, bits, gh, uh, dh, codebook, CAP);
}

pub fn buildLayerWithRows(r: *Rig, bits: []const u32, gh: []const u32, uh: []const u32, dh: []const u32, codebook: exl3e.Codebook, rows_n: usize) !Built {
    var b: Built = .{
        .layer = undefined,
        .bits = bits,
        .rows = rows_n,
        .gh = gh,
        .uh = uh,
        .dh = dh,
        .gate = undefined,
        .up = undefined,
        .down = undefined,
        .dx = 0,
        .dpick = 0,
        .dwts = 0,
        .suh_g = undefined,
        .suh_u = undefined,
        .svh_g = undefined,
        .svh_u = undefined,
        .suh_d = undefined,
        .svh_d = undefined,
        .x = undefined,
        .pick = undefined,
        .wts = undefined,
    };
    var gate_p: [E]u64 = undefined;
    var up_p: [E]u64 = undefined;
    var down_p: [E]u64 = undefined;
    var gk2: [E]i32 = undefined;
    var uk2: [E]i32 = undefined;
    var dk2: [E]i32 = undefined;
    const suh_g = try r.a.alloc(u16, E * D);
    const suh_u = try r.a.alloc(u16, E * D);
    const svh_g = try r.a.alloc(u16, E * I);
    const svh_u = try r.a.alloc(u16, E * I);
    const suh_d = try r.a.alloc(u16, E * I);
    const svh_d = try r.a.alloc(u16, E * D);
    for (0..E) |e| {
        b.gate[e] = try randI16(r, (D / 16) * (I / 16) * halvesOf(gh[e]).tileWords());
        b.up[e] = try randI16(r, (D / 16) * (I / 16) * halvesOf(uh[e]).tileWords());
        b.down[e] = try randI16(r, (I / 16) * (D / 16) * halvesOf(dh[e]).tileWords());
        gate_p[e] = try r.dev(std.mem.sliceAsBytes(b.gate[e]));
        up_p[e] = try r.dev(std.mem.sliceAsBytes(b.up[e]));
        down_p[e] = try r.dev(std.mem.sliceAsBytes(b.down[e]));
        gk2[e] = @intCast(gh[e]);
        uk2[e] = @intCast(uh[e]);
        dk2[e] = @intCast(dh[e]);
        randScale(r, suh_g[e * D ..][0..D]);
        randScale(r, suh_u[e * D ..][0..D]);
        randScale(r, svh_g[e * I ..][0..I]);
        randScale(r, svh_u[e * I ..][0..I]);
        randScale(r, suh_d[e * I ..][0..I]);
        randScale(r, svh_d[e * D ..][0..D]);
    }
    b.suh_g = suh_g;
    b.suh_u = suh_u;
    b.svh_g = svh_g;
    b.svh_u = svh_u;
    b.suh_d = suh_d;
    b.svh_d = svh_d;
    b.layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p[0..])),
        .up = try r.dev(std.mem.sliceAsBytes(up_p[0..])),
        .down = try r.dev(std.mem.sliceAsBytes(down_p[0..])),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2[0..])),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2[0..])),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2[0..])),
        .suh_g = try r.dev(std.mem.sliceAsBytes(suh_g)),
        .suh_u = try r.dev(std.mem.sliceAsBytes(suh_u)),
        .svh_g = try r.dev(std.mem.sliceAsBytes(svh_g)),
        .svh_u = try r.dev(std.mem.sliceAsBytes(svh_u)),
        .suh_d = try r.dev(std.mem.sliceAsBytes(suh_d)),
        .svh_d = try r.dev(std.mem.sliceAsBytes(svh_d)),
        .experts = E,
        .dims = D,
        .width = I,
        .codebook = codebook,
        .widths_ok = codebook != .mul1,
        .limit = LIMIT,
        .act_mode = 0,
    };
    b.x = try r.bfs(rows_n * D, 0.5); // half-range x, with the small scales, keeps the SwiGLU inputs moderate
    b.pick = try picks(r, rows_n);
    b.wts = try wtsOf(r, rows_n);
    b.dx = try r.dev(std.mem.sliceAsBytes(b.x));
    b.dpick = try r.dev(std.mem.sliceAsBytes(b.pick));
    b.dwts = try r.dev(std.mem.sliceAsBytes(b.wts));
    return b;
}

fn randI16(r: *Rig, n: usize) ![]i16 {
    const out = try r.a.alloc(i16, n);
    for (out) |*v| v.* = r.rng.int(i16);
    return out;
}

/// Small scales (with the half-range x): the |t||w| sums and the SwiGLU inputs stay moderate for the bound.
fn randScale(r: *Rig, out: []u16) void {
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 0.25 - 0.125)));
}

/// Distinct experts a row: a partial Fisher-Yates over the layer's experts, seeded.
fn picks(r: *Rig, cap: usize) ![]i32 {
    const out = try r.a.alloc(i32, cap * SLOTS);
    for (0..cap) |row| {
        var idx: [E]u32 = undefined;
        for (0..E, 0..) |_, i| idx[i] = @intCast(i);
        var n: usize = E;
        for (0..SLOTS) |k| {
            const j = r.rng.uintLessThan(usize, n);
            out[row * SLOTS + k] = @intCast(idx[j]);
            idx[j] = idx[n - 1];
            n -= 1;
        }
    }
    return out;
}

fn wtsOf(r: *Rig, cap: usize) ![]f32 {
    const out = try r.a.alloc(f32, cap * SLOTS);
    for (out) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    return out;
}

/// One routed call over `rows` rows starting at row 0 of the built routing; the scratch and out are fresh canaries.
pub fn runRows(r: *Rig, exp: *const exl3e.Experts, b: *const Built, rows_n: usize) !u64 {
    return (try runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, rows_n)).out;
}

pub fn runSlice(r: *Rig, exp: *const exl3e.Experts, layer: exl3e.Layer, x: u64, pick: u64, wts: u64, rows_n: usize) !struct { out: u64, y: u64 } {
    const sb = try r.zeros(try exl3e.scratchBytes(layer, rows_n, SLOTS));
    const sc = try exl3e.scratch(sb, layer, rows_n, SLOTS);
    const out = try r.zeros(rows_n * D * 4);
    try exp.run(r.s, layer, x, D, .bf16, pick, wts, rows_n, out, sc);
    return .{ .out = out, .y = sc.y };
}

fn f64Check(r: *Rig, exp: *const exl3e.Experts, bits: []const u32) !void {
    const b = try buildLayer(r, bits);
    const want = try reference(r, &b, 128);
    for (rows) |m| try closeOut(r, b.bits, m, (try runRows(r, exp, &b, m)), want.want, want.tol);
}

/// fp32 outputs within the f64 reference's tolerance, with the exact FAIL line.
fn closeOut(r: *Rig, bits: []const u32, m: usize, ptr: u64, want: []const f64, tol: []const f64) !void {
    const raw = try r.back(ptr, m * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want[0 .. m * D], tol[0 .. m * D], 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.experts.f64.mul1.{d} m={d}: largest difference {d:.3} of its tolerance (element {d})\n", .{ bits[0], m, most, at });
    if (most > 1) {
        std.debug.print("FAIL exl3.experts.f64.mul1.{d}: largest difference {d:.3} of its tolerance\n", .{ bits[0], most });
        return error.TestFailed;
    }
}

/// A 300-row chunk against the f64 reference (the mixed-width layer).
pub fn prompt(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [E]u32 = .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    const b = try buildLayer(r, &bits);
    const want = try reference(r, &b, 300);
    const res = try runRows(r, exp, &b, 300);
    const raw = try r.back(res, 300 * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want.want, want.tol, 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.prompt.experts.f64: largest difference {d:.3} of its tolerance (element {d})\n", .{ most, at });
    if (most > 1) return error.TestFailed;
}

/// The f64 reference: the kernel's chain mirrored in f64, each output's error its per-element worst case.
pub fn reference(r: *Rig, b: *const Built, rows_n: usize) !struct { want: []f64, tol: []f64 } {
    const p = rows_n * SLOTS;
    // xg: fp16((x * suh_g) @ H) per slot; |xg| and its error <= V, the block's sum of |x * suh| over sqrt(128)
    const xs = try r.a.alloc(f64, p * D);
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        for (0..D) |i| xs[s0 * D + i] = bf(b.x[(s0 / SLOTS) * D + i]) * f16v(b.suh_g[ex * D + i]);
    }
    exl3fmt.rotate(xs, p, D, 1);
    const xg = try r.a.alloc(f16, p * D);
    for (xg, xs) |*o, v| o.* = @floatCast(v);
    const xgerr = try r.a.alloc(f64, p * D);
    for (xgerr, xs) |*o, v| o.* = @abs(v);
    absRotateCols(xgerr, p, D);
    for (xgerr) |*v| v.* *= 1.0 / 2048.0 + 8.0 / 16777216.0; // the fp16 rounding (2^-11) plus the fp32 rotation's (8 * 2^-24), both of V

    const wq = try r.a.alloc(f16, @max(D, I) * @max(D, I));
    const zg = try r.a.alloc(f64, I * p); // [n][p]
    const zu = try r.a.alloc(f64, I * p);
    const s1g = try r.a.alloc(f64, I * p);
    const s1u = try r.a.alloc(f64, I * p);
    const eg = try r.a.alloc(f64, I * p); // the GEMM's propagated input error, before the output rotation
    const eu = try r.a.alloc(f64, I * p);
    @memset(zg, 0);
    @memset(zu, 0);
    @memset(s1g, 0);
    @memset(s1u, 0);
    @memset(eg, 0);
    @memset(eu, 0);
    for (0..E) |ex| {
        unpackProjection(b, b.gh[ex], b.gate[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zg[s0..], s1g[s0..], eg[s0..], xgerr[s0 * D ..][0..D], D, I, p);
        }
        unpackProjection(b, b.uh[ex], b.up[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zu[s0..], s1u[s0..], eu[s0..], xgerr[s0 * D ..][0..D], D, I, p);
        }
    }
    // the fp16 rounding (2^-11) and the fp32 accumulation (2^-17 of 128 roundings), both on the |t||w| sum
    for (eg, s1g) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    for (eu, s1u) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    exl3fmt.rotate(zg, I, p, 0);
    exl3fmt.rotate(zu, I, p, 0);
    // the output Hadamard: worst case is the block's sum of |errors| over sqrt(128), plus the fp32 rotation's own
    absRotate(eg, I, p);
    absRotate(eu, I, p);
    const zgerr = try r.a.alloc(f64, I * p);
    for (zgerr, zg) |*o, v| o.* = @abs(v);
    absRotate(zgerr, I, p);
    for (zgerr) |*v| v.* *= 8.0 / 16777216.0;
    const zuerr = try r.a.alloc(f64, I * p);
    for (zuerr, zu) |*o, v| o.* = @abs(v);
    absRotate(zuerr, I, p);
    for (zuerr) |*v| v.* *= 8.0 / 16777216.0;
    for (eg, zgerr) |*e, v| e.* += v;
    for (eu, zuerr) |*e, v| e.* += v;

    // the SwiGLU (bf16 roundings, f64 exp): g's error is the rotated GEMM error through svh plus a bf16 double rounding
    const v = try r.a.alloc(f64, I * p);
    const verr = try r.a.alloc(f64, I * p);
    for (0..I) |n| {
        for (0..p) |s0| {
            const ex: usize = @intCast(b.pick[s0]);
            const vg = zg[n * p + s0] * f16v(b.svh_g[ex * I + n]);
            const gg = @min(bf16r(vg), @as(f64, LIMIT));
            const vu = zu[n * p + s0] * f16v(b.svh_u[ex * I + n]);
            const uu = @min(@max(bf16r(vu), -@as(f64, LIMIT)), @as(f64, LIMIT));
            const inner = bf16r(gg / (1.0 + std.math.exp(-gg)));
            const act = bf16r(inner * uu);
            v[n * p + s0] = act * f16v(b.suh_d[ex * I + n]);
            const egk = eg[n * p + s0] * @abs(f16v(b.svh_g[ex * I + n])) + (@abs(vg) + @abs(gg)) / 256;
            const euk = eu[n * p + s0] * @abs(f16v(b.svh_u[ex * I + n])) + (@abs(vu) + @abs(uu)) / 256;
            const einner = egk + (@abs(gg) + 1) / 524288 + @abs(inner) / 256;
            verr[n * p + s0] = (@abs(uu) * einner + @abs(inner) * euk + @abs(act) / 256) * @abs(f16v(b.suh_d[ex * I + n])) +
                @abs(v[n * p + s0]) / 16777216; // act's: |silu'| (<= 1) g's, |silu(g)| (<= |g|) u's, expf and roundings
        }
    }
    // the down input: the fp32 rotation's roundings (of the block's sum of |v|) and the fp16 store (2^-11)
    exl3fmt.rotate(v, I, p, 0);
    absRotate(verr, I, p);
    const vrot = try r.a.alloc(f64, I * p);
    for (vrot, v) |*o, val| o.* = @abs(val);
    absRotate(vrot, I, p);
    for (vrot) |*o| o.* *= 8.0 / 16777216.0;
    const xd = try r.a.alloc(f16, p * I);
    const xderr = try r.a.alloc(f64, p * I);
    for (0..p) |s0| {
        for (0..I) |ki| {
            xd[s0 * I + ki] = @floatCast(v[ki * p + s0]);
            xderr[s0 * I + ki] = verr[ki * p + s0] + vrot[ki * p + s0] + @abs(xd[s0 * I + ki]) / 2048;
        }
    }

    // down: the GEMM, its |t| and |w| sums, and the propagated input error
    const zd = try r.a.alloc(f64, D * p); // [d][p]
    const s1d = try r.a.alloc(f64, D * p);
    const de = try r.a.alloc(f64, D * p);
    @memset(zd, 0);
    @memset(s1d, 0);
    @memset(de, 0);
    for (0..E) |ex| {
        unpackProjection(b, b.dh[ex], b.down[ex], I / 16, D / 16, wq[0 .. I * D]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            for (0..I) |ki| {
                const t: f64 = @floatCast(xd[s0 * I + ki]);
                const ta = @abs(t);
                const row = wq[ki * D ..][0..D];
                const xek = xderr[s0 * I + ki];
                for (0..D) |d| {
                    const wv: f64 = @floatCast(row[d]);
                    zd[d * p + s0] += t * wv;
                    s1d[d * p + s0] += ta * @abs(wv);
                    de[d * p + s0] += @abs(wv) * xek;
                }
            }
        }
    }
    // the fp16 xd rounding (2^-11 of the |t||w| sum) and the fp32 accumulation (2^-17), then the output Hadamard's mix
    for (de, s1d) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    exl3fmt.rotate(zd, D, p, 0);
    absRotate(de, D, p);
    const zderr = try r.a.alloc(f64, D * p);
    for (zderr, zd) |*o, val| o.* = @abs(val);
    absRotate(zderr, D, p);
    for (zderr) |*o| o.* *= 8.0 / 16777216.0;
    for (de, zderr) |*e, val| e.* += val;

    // the per-slot outputs and the weighted combine: the fp32 fma chain (8 roundings of the running sum) and the store
    const want = try r.a.alloc(f64, rows_n * D);
    const tol = try r.a.alloc(f64, rows_n * D);
    for (0..rows_n) |row| {
        for (0..D) |d| {
            var acc: f32 = 0;
            var err: f64 = 0;
            var swy: f64 = 0;
            for (0..SLOTS) |k| {
                const s0 = row * SLOTS + k;
                const ex: usize = @intCast(b.pick[s0]);
                const y: f32 = @floatCast(zd[d * p + s0] * f16v(b.svh_d[ex * D + d]));
                acc = @mulAdd(f32, b.wts[row * SLOTS + k], y, acc);
                const yerr = de[d * p + s0] * @abs(f16v(b.svh_d[ex * D + d])) + @abs(y) / 16777216;
                err += @abs(b.wts[row * SLOTS + k]) * yerr;
                swy += @abs(b.wts[row * SLOTS + k] * y);
            }
            want[row * D + d] = acc;
            tol[row * D + d] = err + swy / 2097152 + @abs(want[row * D + d]) / 16777216 + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

/// One output column's GEMM over k: the sums into [n][p], their |t| sums, the propagated input error.
fn gemmK(xrow: []const f16, wq: []const f16, z: []f64, s1: []f64, e: []f64, xerr: []const f64, k: usize, n: usize, p: usize) void {
    for (0..k) |ki| {
        const t: f64 = @floatCast(xrow[ki]);
        const ta = @abs(t);
        const wrow = wq[ki * n ..][0..n];
        const xek = xerr[ki];
        for (0..n) |col| {
            const wv: f64 = @floatCast(wrow[col]);
            z[col * p] += t * wv;
            s1[col * p] += ta * @abs(wv);
            e[col * p] += @abs(wv) * xek;
        }
    }
}

/// Each 128-block along the row axis replaced by the sum of its |entries| over sqrt(128) (the worst case).
fn absRotate(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    var r: usize = 0;
    while (r < nrows) : (r += 128) {
        for (0..cols) |c| {
            var s: f64 = 0;
            for (0..128) |n| s += @abs(x[(r + n) * cols + c]);
            s *= inv;
            for (0..128) |n| x[(r + n) * cols + c] = s;
        }
    }
}

test "the ids are the contract's" {
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.4", f64_ids[0]);
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.6", f64_ids[2]);
}

/// The column-axis twin: each 128-run along the columns replaced by the sum of its |entries| over sqrt(128).
fn absRotateCols(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    for (0..nrows) |row| {
        var c: usize = 0;
        while (c < cols) : (c += 128) {
            var s: f64 = 0;
            for (0..128) |n| s += @abs(x[row * cols + c + n]);
            s *= inv;
            for (0..128) |n| x[row * cols + c + n] = s;
        }
    }
}

/// The tensor-core prompt path (exl3_prompt.run: mixed widths, mcg and mcg_e4m3) against the decode path (runSlots) on
/// the same rows, slot by slot; and the prompt path in two calls == one call, bit for bit.
pub fn promptMixed(r: *Rig, exp: *const exl3e.Experts) !void {
    var experts_mod = try cuda.Module.load(r.gpu.d, cuda.kernels.experts);
    defer experts_mod.unload();
    const router = try cuda.grouped.Router.resolve(experts_mod);
    const all4: [E]u32 = @splat(4);
    const all6: [E]u32 = @splat(6);
    const all8: [E]u32 = @splat(8);
    const mixed: [E]u32 = .{ 2, 4, 6, 8, 2, 4, 6, 8, 2 };
    const gate4_up6: [E]u32 = @splat(4);
    const gate4_up6_u: [E]u32 = @splat(6);
    const down4: [E]u32 = @splat(4);
    const Case = struct { name: []const u8, g: []const u32, u: []const u32, d: []const u32, cb: exl3e.Codebook };
    const cases = [_]Case{
        .{ .name = "mcg all-4", .g = &all4, .u = &all4, .d = &all4, .cb = .mcg },
        .{ .name = "mcg all-6", .g = &all6, .u = &all6, .d = &all6, .cb = .mcg },
        .{ .name = "mcg all-8", .g = &all8, .u = &all8, .d = &all8, .cb = .mcg },
        .{ .name = "mcg mixed 2-4-6-8", .g = &mixed, .u = &mixed, .d = &mixed, .cb = .mcg },
        .{ .name = "mcg gate4-up6", .g = &gate4_up6, .u = &gate4_up6_u, .d = &down4, .cb = .mcg },
        .{ .name = "mcg_e4m3 all-4", .g = &all4, .u = &all4, .d = &all4, .cb = .mcg_e4m3 },
        .{ .name = "mcg_e4m3 all-6", .g = &all6, .u = &all6, .d = &all6, .cb = .mcg_e4m3 },
        .{ .name = "mcg_e4m3 all-8", .g = &all8, .u = &all8, .d = &all8, .cb = .mcg_e4m3 },
        .{ .name = "mcg_e4m3 mixed 2-4-6-8", .g = &mixed, .u = &mixed, .d = &mixed, .cb = .mcg_e4m3 },
        .{ .name = "mcg_e4m3 gate4-up6", .g = &gate4_up6, .u = &gate4_up6_u, .d = &down4, .cb = .mcg_e4m3 },
    };
    for (cases) |c| {
        var bits: [E]u32 = undefined;
        for (&bits, c.g) |*o, v| o.* = v;
        const b = try buildLayerWithRows(r, &bits, c.g, c.u, c.d, c.cb, 2048);
        if (!exl3e.prompt.eligible(b.layer)) return error.PromptNotEligible;
        var row_baseline: ?[]u8 = null;
        for ([_]usize{ 128, 512, 2048, 1, 17, 64, 300 }) |n| {
            const whole = try promptRun(r, exp, router, b.layer, b.dx, b.dpick, n);
            const over = try promptRunOverlay(r, exp, router, b.layer, b.dx, b.dpick, n);
            const decode_run = try runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, n);
            const dec = decode_run.y;
            const got = try r.back(whole, n * SLOTS * D * 4);
            const over_got = try r.back(over, n * SLOTS * D * 4);
            if (!std.mem.eql(u8, got, over_got)) return error.PromptOverlayMismatch;
            const want = try r.back(dec, n * SLOTS * D * 4);
            if (n == 128) {
                row_baseline = try r.a.dupe(u8, got);
            } else if (n > 128 and !std.mem.eql(u8, row_baseline.?, got[0..row_baseline.?.len])) {
                std.debug.print("FAIL glm53-prompt-experts {s} rows={d}: first 128 rows changed with batch size\n", .{ c.name, n });
                return error.PromptRowInvariance;
            }
            var max_abs: f64 = 0;
            var max_rel: f64 = 0;
            for (0..n * SLOTS) |ps| {
                var scale: f64 = 0;
                for (0..D) |i| scale = @max(scale, @abs(@as(f64, std.mem.bytesToValue(f32, want[4 * (ps * D + i) ..][0..4]))));
                for (0..D) |i| {
                    const pv: f64 = std.mem.bytesToValue(f32, got[4 * (ps * D + i) ..][0..4]);
                    const dv: f64 = std.mem.bytesToValue(f32, want[4 * (ps * D + i) ..][0..4]);
                    const diff = @abs(pv - dv);
                    max_abs = @max(max_abs, diff);
                    if (scale > 0) max_rel = @max(max_rel, diff / scale);
                    if (!std.math.isFinite(pv) or diff > 5e-3 * scale + 1e-6) {
                        std.debug.print("FAIL exl3-prompt-mixed {s} rows={d}: pair-slot {d} col {d} prompt {e} decode {e} (row-slot max {e})\n", .{ c.name, n, ps, i, pv, dv, scale });
                        return error.TestFailed;
                    }
                }
            }
            const prompt_out = try r.zeros(n * D * 4);
            const invalid_ids = try r.zeros(n * SLOTS * 4);
            try exp.combinePrompt(r.s, b.layer, whole, b.dwts, n, SLOTS, prompt_out, invalid_ids);
            try r.s.synchronize();
            const combined = try r.back(prompt_out, n * D * 4);
            const decode_combined = try r.back(decode_run.out, n * D * 4);
            var max_combined: f64 = 0;
            for (0..n * D) |i| {
                const pv: f64 = std.mem.bytesToValue(f32, combined[4 * i ..][0..4]);
                const dv: f64 = std.mem.bytesToValue(f32, decode_combined[4 * i ..][0..4]);
                max_combined = @max(max_combined, @abs(pv - dv));
                if (!std.math.isFinite(pv) or @abs(pv - dv) > 8e-3 * @abs(dv) + 2e-5) return error.PromptCombineMismatch;
            }
            var f64_ratio: f64 = 0;
            var row0: usize = 0;
            while (row0 + 128 <= n) : (row0 += 128) {
                f64_ratio = @max(f64_ratio, try referenceWindow(r, &b, combined, row0));
            }
            std.debug.print("RESULT glm53-prompt-experts {s} rows={d}: max_abs {e:.3}, f64_tolerance_ratio {d:.4}\n", .{ c.name, n, max_combined, f64_ratio });
            if (f64_ratio > 1) return error.PromptF64Mismatch;
            if (n > 128) { // two calls: rows 0..127 then 128..n-1, each writing its own rows
                const first = try promptRun(r, exp, router, b.layer, b.dx, b.dpick, 128);
                const second = try promptRun(r, exp, router, b.layer, b.dx + 128 * D * 2, b.dpick + 128 * SLOTS * 4, n - 128);
                const a = try r.back(first, 128 * SLOTS * D * 4);
                const z = try r.back(second, (n - 128) * SLOTS * D * 4);
                if (!std.mem.eql(u8, a, got[0..a.len]) or !std.mem.eql(u8, z, got[a.len..][0..z.len])) {
                    std.debug.print("FAIL exl3-prompt-mixed {s} rows={d}: two calls differ from one call\n", .{ c.name, n });
                    return error.TestFailed;
                }
            }
            std.debug.print("PASS exl3-prompt-mixed {s} rows={d}: slots max_abs {e:.3} max_rel {e:.3}; weighted max_abs {e:.3} vs decode{s}\n", .{ c.name, n, max_abs, max_rel, max_combined, if (n > 128) "; two calls == one" else "" });
        }
    }
}

fn referenceWindow(r: *Rig, b: *const Built, output: []const u8, row0: usize) !f64 {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    var ref_rig = r.*;
    ref_rig.a = arena.allocator();
    var ref_b = b.*;
    ref_b.x = b.x[row0 * D ..];
    ref_b.pick = b.pick[row0 * SLOTS ..];
    ref_b.wts = b.wts[row0 * SLOTS ..];
    const reference_rows = try reference(&ref_rig, &ref_b, 128);
    var ratio: f64 = 0;
    for (0..128 * D) |i| {
        const got: f64 = std.mem.bytesToValue(f32, output[4 * (row0 * D + i) ..][0..4]);
        ratio = @max(ratio, @abs(got - reference_rows.want[i]) / reference_rows.tol[i]);
    }
    return ratio;
}

/// One prompt-path call over `rows` rows: y [rows][SLOTS][D] fp32 (fresh scratch and plan).
fn promptRun(r: *Rig, exp: *const exl3e.Experts, router: cuda.grouped.Router, layer: exl3e.Layer, x: u64, pick: u64, rows_n: usize) !u64 {
    return promptRunLayout(r, exp, router, layer, x, pick, rows_n, false);
}

fn promptRunOverlay(r: *Rig, exp: *const exl3e.Experts, router: cuda.grouped.Router, layer: exl3e.Layer, x: u64, pick: u64, rows_n: usize) !u64 {
    return promptRunLayout(r, exp, router, layer, x, pick, rows_n, true);
}

fn promptRunLayout(r: *Rig, exp: *const exl3e.Experts, router: cuda.grouped.Router, layer: exl3e.Layer, x: u64, pick: u64, rows_n: usize, overlay: bool) !u64 {
    const P = rows_n * SLOTS;
    const tile = exl3e.prompt.tile;
    const wide = P > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E, tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) P else 1) * 4),
        .hist = try r.zeros((if (wide) (P + 1023) / 1024 * E else 1) * 4),
    };
    const base = try r.zeros(try exl3e.prompt.bytes(layer, rows_n, SLOTS, overlay));
    const sc = try exl3e.prompt.scratch(base, layer, rows_n, SLOTS, overlay);
    try exp.prompt.runTile(exp, router, r.s, layer, x, D, pick, rows_n, sc, plan, tile);
    try r.s.synchronize();
    return sc.y;
}

/// `tf-cuda-test exl3-prompt-bench`: one GLM-5.3-shaped routed layer (256 experts, D 6144, rank width 1024, 8 slots,
/// mcg_e4m3, mixed widths: mostly K2 with some K1 and K3; random trellis bits), 4096 prompt rows:
/// the tensor-core prompt path against the decode path in the row chunks GLM-5.3's runPrompt used before (<= 1024).
pub fn promptBench(r: *Rig, exp: *const exl3e.Experts) !void {
    const BE = 256;
    const BD = 6144;
    const BI = 1024;
    const rows_n: usize = 4096;
    var experts_mod = try cuda.Module.load(r.gpu.d, cuda.kernels.experts);
    defer experts_mod.unload();
    const router = try cuda.grouped.Router.resolve(experts_mod);
    var ptrs: [3][BE]u64 = undefined;
    var k2s: [3][BE]i32 = undefined;
    for (0..BE) |e| for (0..3) |m| {
        const h: u32 = if (e % 16 == 0) 2 else if (e % 9 == 0) 6 else 4; // a fixed mixed-width pattern
        k2s[m][e] = @intCast(h);
        const words = (BD / 16) * (BI / 16) * halvesOf(h).tileWords();
        const t = try randI16(r, words);
        ptrs[m][e] = try r.dev(std.mem.sliceAsBytes(t));
        r.a.free(t);
    };
    const sv = try r.a.alloc(u16, BE * BD);
    randScale(r, sv);
    const s_d = try r.dev(std.mem.sliceAsBytes(sv));
    const s_i = try r.dev(std.mem.sliceAsBytes(sv[0 .. BE * BI]));
    var layer: exl3e.Layer = .{ .gate = try r.dev(std.mem.sliceAsBytes(ptrs[0][0..])), .up = try r.dev(std.mem.sliceAsBytes(ptrs[1][0..])), .down = try r.dev(std.mem.sliceAsBytes(ptrs[2][0..])), .gate_k2 = try r.dev(std.mem.sliceAsBytes(k2s[0][0..])), .up_k2 = try r.dev(std.mem.sliceAsBytes(k2s[1][0..])), .down_k2 = try r.dev(std.mem.sliceAsBytes(k2s[2][0..])), .suh_g = s_d, .suh_u = s_d, .svh_g = s_i, .svh_u = s_i, .suh_d = s_i, .svh_d = s_d, .experts = BE, .dims = BD, .width = BI, .codebook = .mcg_e4m3, .widths_ok = true, .limit = LIMIT, .act_mode = 0 };
    const x = try r.bfs(rows_n * BD, 0.5);
    const pk = try r.a.alloc(i32, rows_n * SLOTS);
    for (0..rows_n) |row| for (0..SLOTS) |k| {
        pk[row * SLOTS + k] = @intCast((row * 37 + k * 97 + (row / 13) * 5) % BE); // spread, distinct within a row
    };
    for (0..rows_n) |row| {
        for (0..SLOTS) |k| for (0..k) |j| if (pk[row * SLOTS + k] == pk[row * SLOTS + j]) {
            pk[row * SLOTS + k] = @intCast((@as(usize, @intCast(pk[row * SLOTS + k])) + 1 + k) % BE);
        };
    }
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dpk = try r.dev(std.mem.sliceAsBytes(pk));
    const wts = try wtsOf(r, rows_n);
    const dw = try r.dev(std.mem.sliceAsBytes(wts));
    // prompt path: one call for all rows (fresh plan and scratch outside the timing)
    const P = rows_n * SLOTS;
    const tile = exl3e.prompt.tile;
    const plan: cuda.grouped.Plan = .{ .members = try r.zeros(P * 4), .items = try r.zeros(cuda.grouped.maxItems(P, BE, tile) * 12), .counts = try r.zeros(8), .rank = try r.zeros(P * 4), .hist = try r.zeros((P + 1023) / 1024 * BE * 4) };
    const psc = try exl3e.prompt.scratch(try r.zeros(try exl3e.prompt.bytes(layer, rows_n, SLOTS, false)), layer, rows_n, SLOTS, false);
    var ev0 = try cuda.Event.init(r.gpu.d, true);
    defer ev0.deinit();
    var ev1 = try cuda.Event.init(r.gpu.d, true);
    defer ev1.deinit();
    for (0..2) |_| try exp.prompt.run(exp, router, r.s, layer, dx, BD, dpk, rows_n, psc, plan);
    try ev0.record(r.s);
    const reps = 5;
    for (0..reps) |_| try exp.prompt.run(exp, router, r.s, layer, dx, BD, dpk, rows_n, psc, plan);
    try ev1.record(r.s);
    try r.s.synchronize();
    const prompt_ms = (try ev0.elapsedMs(ev1)) / reps;
    // decode path in 1024-row chunks (exp.run: routed + combine), as the fallback ran
    const chunk: usize = 1024;
    const dsc = try exl3e.scratch(try r.zeros(try exl3e.scratchBytes(layer, chunk, SLOTS)), layer, chunk, SLOTS);
    const out = try r.zeros(chunk * BD * 4);
    layer.widths_ok = true;
    try exp.run(r.s, layer, dx, BD, .bf16, dpk, dw, chunk, out, dsc);
    try ev0.record(r.s);
    var at: usize = 0;
    while (at < rows_n) : (at += chunk) try exp.run(r.s, layer, dx + at * BD * 2, BD, .bf16, dpk + at * SLOTS * 4, dw + at * SLOTS * 4, chunk, out, dsc);
    try ev1.record(r.s);
    try r.s.synchronize();
    const decode_ms = try ev0.elapsedMs(ev1);
    const flop = 2.0 * @as(f64, @floatFromInt(P)) * (2.0 * BD * BI + BI * BD);
    std.debug.print("BENCH exl3-prompt GLM-5.3 layer (256 x 6144 x 1024 rank half, mcg_e4m3 mixed K1/K2/K3), {d} rows x 8 slots: prompt path {d:.2} ms ({d:.1} TFLOP/s) | decode path (1024-row chunks) {d:.2} ms ({d:.1} TFLOP/s) | {d:.2}x\n", .{ rows_n, prompt_ms, flop / prompt_ms / 1e9, decode_ms, flop / decode_ms / 1e9, decode_ms / prompt_ms });
}
