// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-glm53-bringup's drafter admission: the DFlash2 drafter made resident beside the GLM-5.3 target (weights at
//! 4 bits as Flash's loader packs them, a draft head of the exact size, scratch, one ring per stream) to measure
//! MemAvailable at the serving context. Memory only: the drafter is never run; when the target lm_head is present,
//! its qmmf BF16 rows are unpacked and quantized in device memory to make the draft head.
//! Before any allocation the planned peak (resident bytes + the loader's transients) is checked against the floor:
//! a rank that would cross it reports FAIL from the measured base and the plan, and allocates nothing.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const drafts = @import("drafts");

const gib = 1024.0 * 1024.0 * 1024.0;

fn g(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

/// Offsets of a Q4 head in one buffer, as drafts.packTargetHead lays it out (words, scales, biases at 256 B).
pub const HeadLayout = struct { total: u64, soff: u64, boff: u64 };

pub fn headLayout(n: usize, k: usize) HeadLayout {
    const sizes = glm.q4.bytes(n, k);
    const soff = std.mem.alignForward(u64, sizes[0], 256);
    const boff = soff + std.mem.alignForward(u64, sizes[1], 256);
    return .{ .total = boff + sizes[2], .soff = soff, .boff = boff };
}

/// The loader's device transients while it reads (dflash_load.loadWithHead: the largest linear as stored, and its
/// 4-bit staging), freed when it returns.
pub fn loaderTransientBytes(geometry: drafts.Geometry, bits: u8) u64 {
    const rt = drafts.runtimeGeometry(geometry);
    const d: u64 = @intCast(rt.hidden);
    const ff: u64 = @intCast(rt.ff);
    const qkv: u64 = @intCast(rt.qkvWidth());
    const taps: u64 = @intCast(geometry.target_layers.len);
    const max_linear = @max(2 * ff * d, @max(qkv * d, taps * d * d));
    const raw: u64 = if (bits == 4) max_linear / 2 + 2 * max_linear / 32 + 1024 else 256;
    return max_linear * 2 + raw;
}

pub const Verdict = struct { available: u64, pass: bool, allocated: bool };

/// `available` is this rank's MemAvailable in steady state before the drafter; `meminfo` reads MemAvailable now.
pub fn admit(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, stream: cuda.Stream, dir: []const u8, rank: usize, bits: u8, streams: usize, available: u64, floor: u64, meminfo: *const fn (std.mem.Allocator, std.Io) ?u64, head: ?cuda.qmmf.Weight) !Verdict {
    const share_scratch = drafts.shareScratchEnabled();
    if (bits != 4) return error.DrafterBitsUnsupported; // the loader packs 4 bits or keeps bf16 (bits 0), no 8-bit path
    const geometry = try drafts.checkConfig(gpa, io, dir);
    const plan = try drafts.memoryPlan(geometry, bits, 2, streams);
    const transient = if (head != null) @max(loaderTransientBytes(geometry, bits), drafts.packerTransientBytes(geometry, 2)) else loaderTransientBytes(geometry, bits);
    std.debug.print("DRAFTER plan ({d}-bit, {d} stream(s)): weights {d:.3} draft_head {d:.3} scratch {d:.3} rings {d:.3} device {d:.3} GiB; host {d:.3} GiB; loader transient {d:.3} GiB\n", .{ bits, streams, g(plan.weights), g(plan.draft_head), g(plan.scratch), g(plan.rings), g(plan.deviceBytes()), g(plan.hostBytes()), g(transient) });
    const peak = plan.deviceBytes() + transient;
    if (available < floor + peak) {
        const projected = available -| plan.deviceBytes();
        std.debug.print("DRAFTER not allocated: steady MemAvailable {d:.3} GiB - planned peak {d:.3} GiB would cross the {d:.2} GiB floor\n", .{ g(available), g(peak), g(floor) });
        return .{ .available = projected, .pass = false, .allocated = false };
    }
    var quant_mod = try cuda.Module.load(d, cuda.kernels.glm_quant);
    defer quant_mod.unload();
    var group_mod = try cuda.Module.load(d, cuda.kernels.qmm_group);
    defer group_mod.unload();
    var pack_mod = try cuda.Module.load(d, cuda.kernels.affine4_pack);
    defer pack_mod.unload();
    var mm_mod = try cuda.Module.load(d, cuda.kernels.glm_mm);
    defer mm_mod.unload();
    var glm53_mod: ?cuda.Module = null;
    if (head != null) glm53_mod = try cuda.Module.load(d, cuda.kernels.glm53_glue);
    defer if (glm53_mod) |*m| m.unload();
    const Mods = struct { quant: glm.quant.Kernels, q4: glm.q4.Kernels, mm: glm.mm.Kernels };
    const Adapter = struct { d: *const cuda.Driver, stream: cuda.Stream, m: Mods, share: struct { rank: usize }, unpack: cuda.Function, w: struct { head: ?cuda.qmmf.Weight = null, mtp: ?struct { draft: glm.q4.Q4 } = null } };
    var adapter: Adapter = .{ .d = d, .stream = stream, .m = .{ .quant = try glm.quant.Kernels.load(quant_mod), .q4 = try glm.q4.Kernels.load(group_mod, pack_mod, ctx), .mm = try glm.mm.Kernels.load(mm_mod) }, .share = .{ .rank = rank }, .unpack = if (glm53_mod) |m| try m.function("tf_glm53_unpack_bf16_rows") else undefined, .w = .{ .head = head } };
    if (head != null) std.debug.print("DRAFTER head: packed from the target lm_head (Q4, in memory)\n", .{});
    if (head != null) {
        const dr = try drafts.Drafter.load(gpa, io, &adapter, dir, rank, 2, bits, geometry, share_scratch);
        defer dr.deinit();
        // Rings are made per stream slot on first use: stand-ins of the planned size.
        var rings = try cuda.DeviceBuffer.alloc(d, plan.rings);
        defer rings.free();
        try stream.synchronize();
        const now = meminfo(gpa, io) orelse return error.NoMemInfo;
        std.debug.print("DRAFTER resident: planned {d:.3} GiB, MemAvailable drop {d:.3} GiB (MemAvailable now {d:.3} GiB)\n", .{ g(plan.deviceBytes()), g(available -| now), g(now) });
        return .{ .available = now, .pass = now >= floor, .allocated = true };
    }
    const n = geometry.vocab / 2;
    const k = geometry.hidden;
    const layout = headLayout(n, k);
    var head_storage = try cuda.DeviceBuffer.alloc(d, layout.total);
    var own_head = true;
    defer if (own_head) head_storage.free();
    try d.check(d.api.cuMemsetD8_v2(head_storage.ptr, 0, layout.total), "cuMemsetD8");
    const zero_head: glm.q4.Q4 = .{ .w = head_storage.ptr, .s = head_storage.ptr + layout.soff, .b = head_storage.ptr + layout.boff, .n = n, .k = k, .npad = glm.q4.npadOf(n) };
    std.debug.print("DRAFTER head: placeholder bytes (memory only; not a usable draft head)\n", .{});
    const dr = try glm.dflash_load.loadWithHead(gpa, io, &adapter, dir, rank, 2, bits, drafts.runtimeGeometry(geometry), zero_head, share_scratch);
    dr.head_storage = head_storage;
    own_head = false;
    defer dr.deinit();
    // Rings are made per stream slot on first use: stand-ins of the planned size.
    var rings = try cuda.DeviceBuffer.alloc(d, plan.rings);
    defer rings.free();
    try stream.synchronize();
    const now = meminfo(gpa, io) orelse return error.NoMemInfo;
    std.debug.print("DRAFTER resident: planned {d:.3} GiB, MemAvailable drop {d:.3} GiB (MemAvailable now {d:.3} GiB)\n", .{ g(plan.deviceBytes()), g(available -| now), g(now) });
    return .{ .available = now, .pass = now >= floor, .allocated = true };
}

test "draft head layout mirrors packTargetHead's offsets" {
    const n: usize = 154880 / 2;
    const k: usize = 6144;
    const sizes = glm.q4.bytes(n, k);
    const l = headLayout(n, k);
    try std.testing.expectEqual(std.mem.alignForward(u64, sizes[0], 256), l.soff);
    try std.testing.expectEqual(l.soff + std.mem.alignForward(u64, sizes[1], 256), l.boff);
    try std.testing.expectEqual(l.boff + sizes[2], l.total);
    try std.testing.expect(l.total >= sizes[0] + sizes[1] + sizes[2]);
}
