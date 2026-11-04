// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! A device sub-allocator under DeviceBuffer.alloc: buffers carved from a few large chunks, not one cuMemAlloc each.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

pub const Error = error{ DriverUnavailable, MissingSymbol, Unsupported, CudaFailed, OutOfMemory };

/// Where chunks come from: `chunk` maps `len` bytes (a multiple of `granularity`), `release` gives a chunk back.
pub const Backend = struct {
    ctx: *anyopaque,
    granularity: usize,
    chunk: *const fn (ctx: *anyopaque, len: usize) ?u64,
    release: *const fn (ctx: *anyopaque, base: u64, len: usize) void,
    name: []const u8,
};

pub const Options = struct {
    small_max: usize = 4 << 20, // requests up to this many bytes share chunks
    small_chunk: usize = 64 << 20, // a shared chunk's bytes, rounded up to the granularity
    alignment: usize = 512, // every range starts at a multiple of this
};

pub const Stats = struct { small: u64 = 0, large: u64 = 0, reused: u64 = 0, fallback: u64 = 0, chunk_bytes: u64 = 0 };

/// Small requests bump 512-byte aligned through shared chunks and are reused by size; large ones get their own chunk.
pub const Arena = struct {
    gpa: std.mem.Allocator,
    be: Backend,
    o: Options,
    mutex: std.atomic.Mutex = .unlocked,
    small: std.ArrayList([2]u64) = .empty, // shared chunks (base, len)
    cursor: u64 = 0, // the current shared chunk's next free byte
    limit: u64 = 0,
    free_small: std.AutoHashMapUnmanaged(u64, std.ArrayList(u64)) = .empty, // freed small ranges by rounded size
    live: std.AutoHashMapUnmanaged(u64, u64) = .empty, // live small ranges: address to rounded size
    large: std.AutoHashMapUnmanaged(u64, u64) = .empty, // large chunks: address to mapped size
    dead: [2]u32 = .{ 0, 0 }, // removals from `live` and `large` since each was last rehashed
    stats: Stats = .{},
    previous: ?memory.Hook = null,

    pub fn init(gpa: std.mem.Allocator, be: Backend, o: Options) Arena {
        return .{ .gpa = gpa, .be = be, .o = o };
    }

    pub fn deinit(a: *Arena) void {
        a.uninstall();
        var it = a.large.iterator();
        while (it.next()) |e| a.be.release(a.be.ctx, e.key_ptr.*, e.value_ptr.*);
        for (a.small.items) |c| a.be.release(a.be.ctx, c[0], c[1]);
        var f = a.free_small.valueIterator();
        while (f.next()) |l| l.deinit(a.gpa);
        a.free_small.deinit(a.gpa);
        a.live.deinit(a.gpa);
        a.large.deinit(a.gpa);
        a.small.deinit(a.gpa);
        a.* = undefined;
    }

    fn lock(a: *Arena) void {
        while (!a.mutex.tryLock()) std.Thread.yield() catch {};
    }

    fn roundUp(x: usize, m: usize) usize {
        return std.mem.alignForward(usize, x, m);
    }

    /// An address for `len` bytes, or null (the caller then allocates as before).
    pub fn alloc(a: *Arena, len: usize) ?u64 {
        if (len == 0) return null;
        a.lock();
        defer a.mutex.unlock();
        return a.allocLocked(len) catch null;
    }

    fn allocLocked(a: *Arena, len: usize) !?u64 {
        if (len > a.o.small_max) {
            const n = roundUp(len, a.be.granularity);
            const p = a.be.chunk(a.be.ctx, n) orelse {
                a.fellBack(n);
                return null;
            };
            errdefer a.be.release(a.be.ctx, p, n);
            try a.large.put(a.gpa, p, n);
            a.stats.large += 1;
            a.stats.chunk_bytes += n;
            return p;
        }
        const n: u64 = roundUp(len, a.o.alignment);
        if (a.free_small.getPtr(n)) |l| if (l.items.len > 0) {
            const p = l.items[l.items.len - 1];
            try a.live.put(a.gpa, p, n);
            _ = l.pop();
            a.stats.reused += 1;
            return p;
        };
        if (a.limit - a.cursor < n) {
            const cl = roundUp(@max(a.o.small_chunk, n), a.be.granularity);
            const base = a.be.chunk(a.be.ctx, cl) orelse {
                a.fellBack(cl);
                return null;
            };
            errdefer a.be.release(a.be.ctx, base, cl);
            try a.small.append(a.gpa, .{ base, cl });
            a.cursor = base;
            a.limit = base + cl;
            a.stats.chunk_bytes += cl;
        }
        const p = a.cursor;
        try a.live.put(a.gpa, p, n);
        a.cursor += n;
        a.stats.small += 1;
        return p;
    }

    /// A chunk the backend did not give: the request goes to cuMemAlloc (the first logged: a dead arena shows).
    fn fellBack(a: *Arena, len: usize) void {
        if (!quiet and a.stats.fallback == 0) std.log.scoped(.cuda).warn("device arena ({s}): a {d} MiB chunk failed; requests go to cuMemAlloc", .{ a.be.name, len >> 20 });
        a.stats.fallback += 1;
    }

    /// True when `p` was the arena's (and is now free); false: not ours, the caller frees it as before.
    pub fn free(a: *Arena, p: u64) bool {
        a.lock();
        defer a.mutex.unlock();
        if (a.large.fetchRemove(p)) |e| {
            removed(&a.large, &a.dead[1]);
            a.be.release(a.be.ctx, p, e.value);
            return true;
        }
        const e = a.live.fetchRemove(p) orelse return false;
        removed(&a.live, &a.dead[0]);
        const g = a.free_small.getOrPut(a.gpa, e.value) catch return true; // leaked range, still ours
        if (!g.found_existing) g.value_ptr.* = .empty;
        g.value_ptr.append(a.gpa, p) catch {};
        return true;
    }

    /// std's tombstones never clear and keys fill 80 %: rehash at a tenth, so a tenth of the map always stays free.
    fn removed(map: *std.AutoHashMapUnmanaged(u64, u64), dead: *u32) void {
        dead.* += 1;
        if (@as(u64, dead.*) * 10 < map.capacity()) return;
        map.rehash(std.hash_map.AutoContext(u64){});
        dead.* = 0;
    }

    // -- memory.zig's hook ----------------------------------------------------------------------------------------

    fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        return a.alloc(len);
    }

    fn hookFree(ctx: *anyopaque, p: u64) bool {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        return a.free(p);
    }

    /// Every later DeviceBuffer.alloc comes from the arena, which must outlive its buffers; only addresses change.
    pub fn uninstall(a: *Arena) void {
        if (memory.currentHook()) |hook| if (hook.ctx == @as(*anyopaque, @ptrCast(a))) memory.setHook(a.previous);
    }

    pub fn install(a: *Arena) void {
        if (memory.currentHook()) |hook| if (hook.ctx == @as(*anyopaque, @ptrCast(a))) return;
        a.previous = memory.currentHook();
        memory.setHook(.{ .ctx = a, .alloc = hookAlloc, .free = hookFree });
    }
};

// -- backends ---------------------------------------------------------------------------------------------------------

/// One cuMemAlloc a chunk (the driver's own 2 MiB-aligned allocations).
pub const Plain = struct {
    d: *const Driver,

    pub fn backend(p: *Plain) Backend {
        return .{ .ctx = p, .granularity = 2 << 20, .chunk = chunk, .release = release, .name = "plain" };
    }

    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const p: *Plain = @ptrCast(@alignCast(ctx));
        var q: abi.DevicePtr = 0;
        if (p.d.api.cuMemAlloc_v2(&q, len) != abi.success) return null;
        return q;
    }

    fn release(ctx: *anyopaque, base: u64, _: usize) void {
        const p: *Plain = @ptrCast(@alignCast(ctx));
        _ = p.d.api.cuMemFree_v2(base);
    }
};

/// CUDA's virtual memory management: one reserved virtual range, chunks of the recommended granularity mapped in order.
pub const Vmm = struct {
    lib: std.DynLib,
    device: abi.Device,
    prop: Prop,
    granularity: usize,
    va: u64 = 0,
    va_len: usize = 0,
    next: u64 = 0,
    api: Api,
    retain_handles: bool = false,
    close_unmap_all: bool = false,
    handles: [256]u64 = @splat(0),
    handle_count: usize = 0,
    /// the first chunk that failed: the driver call and its CUresult (null: none failed)
    failed: ?struct { call: []const u8, code: c_int } = null,

    const Api = struct {
        reserve: *const fn (*u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        addressFree: *const fn (u64, usize) callconv(.c) abi.Result,
        create: *const fn (*u64, usize, *const Prop, c_ulonglong) callconv(.c) abi.Result,
        release: *const fn (u64) callconv(.c) abi.Result,
        map: *const fn (u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        unmap: *const fn (u64, usize) callconv(.c) abi.Result,
        setAccess: *const fn (u64, usize, *const AccessDesc, usize) callconv(.c) abi.Result,
        granularity: *const fn (*usize, *const Prop, c_int) callconv(.c) abi.Result,
    };

    pub const Prop = extern struct { // CUmemAllocationProp
        type: c_int = 1, // CU_MEM_ALLOCATION_TYPE_PINNED
        handle_types: c_int = 0,
        loc_type: c_int = 1, // CU_MEM_LOCATION_TYPE_DEVICE
        loc_id: c_int = 0,
        win32: ?*anyopaque = null,
        compression: u8 = 0,
        rdma: u8 = 0,
        usage: u16 = 0,
        reserved: [4]u8 = .{ 0, 0, 0, 0 },
    };
    const AccessDesc = extern struct { loc_type: c_int = 1, loc_id: c_int, flags: c_int = 3 }; // CUmemAccessDesc, read-write

    comptime {
        std.debug.assert(@sizeOf(Prop) == 32 and @offsetOf(Prop, "compression") == 24 and @sizeOf(AccessDesc) == 12);
    }

    /// `va_bytes` of virtual addresses reserved (cheap: no memory behind them until a chunk maps).
    pub fn open(d: *const Driver, device: abi.Device, va_bytes: usize) Error!Vmm {
        return openWithGranularity(d, device, va_bytes, 1, true, true);
    }

    /// Weight arenas use the requested minimum allocation granularity without a recommended-size floor.
    pub fn openMinimum(d: *const Driver, device: abi.Device, va_bytes: usize) Error!Vmm {
        return openWithGranularity(d, device, va_bytes, 0, false, true);
    }

    /// openMinimum with CUmemAllocationProp.gpuDirectRDMACapable left 0 (the TF_GLM53_WEIGHT_VMM_RDMA=0 A/B for the weights).
    pub fn openMinimumNoRdma(d: *const Driver, device: abi.Device, va_bytes: usize) Error!Vmm {
        return openWithGranularity(d, device, va_bytes, 0, false, false);
    }

    fn openWithGranularity(d: *const Driver, device: abi.Device, va_bytes: usize, query: c_int, apply_floor: bool, allow_rdma: bool) Error!Vmm {
        var vmm_ok: c_int = 0;
        const support_rc = d.api.cuDeviceGetAttribute(&vmm_ok, .virtual_memory_management_supported, device);
        if (support_rc != abi.success) {
            std.log.err("CUDA VMM step cuDeviceGetAttribute failed with CUresult {d}", .{support_rc});
            return error.Unsupported;
        }
        if (vmm_ok == 0) return error.Unsupported; // VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const names = .{ .{ "reserve", "cuMemAddressReserve" }, .{ "addressFree", "cuMemAddressFree" }, .{ "create", "cuMemCreate" }, .{ "release", "cuMemRelease" }, .{ "map", "cuMemMap" }, .{ "unmap", "cuMemUnmap" }, .{ "setAccess", "cuMemSetAccess" }, .{ "granularity", "cuMemGetAllocationGranularity" } };
        inline for (names) |n| @field(api, n[0]) = lib.lookup(@FieldType(Api, n[0]), n[1]) orelse return error.MissingSymbol;
        var prop: Prop = .{ .loc_id = device };
        var rdma: c_int = 0; // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED: keep the memory registrable as cuMemAlloc's is
        if (d.api.cuDeviceGetAttribute(&rdma, .gpu_direct_rdma_with_cuda_vmm_supported, device) == abi.success and rdma != 0 and allow_rdma) prop.rdma = 1;
        var g: usize = 0;
        const granularity_rc = api.granularity(&g, &prop, query);
        if (granularity_rc != abi.success) {
            std.log.err("CUDA VMM step cuMemGetAllocationGranularity failed with CUresult {d}", .{granularity_rc});
            return error.Unsupported;
        }
        if (g == 0) return error.Unsupported;
        if (apply_floor) g = @max(g, 2 << 20);
        const len = std.mem.alignForward(usize, va_bytes, g);
        var va: u64 = 0;
        const reserve_rc = api.reserve(&va, len, g, 0, 0);
        if (reserve_rc != abi.success) {
            std.log.err("CUDA VMM step cuMemAddressReserve failed with CUresult {d}", .{reserve_rc});
            return error.CudaFailed;
        }
        return .{ .lib = lib, .device = device, .prop = prop, .granularity = g, .va = va, .va_len = len, .next = va, .api = api };
    }

    pub fn close(v: *Vmm) void {
        if (v.close_unmap_all and v.next > v.va) {
            const rc = v.api.unmap(v.va, @intCast(v.next - v.va));
            if (rc != abi.success) std.log.err("CUDA VMM step cuMemUnmap failed with CUresult {d}", .{rc});
        }
        for (v.handles[0..v.handle_count]) |handle| {
            const rc = v.api.release(handle);
            if (rc != abi.success) std.log.err("CUDA VMM step cuMemRelease failed with CUresult {d}", .{rc});
        }
        const free_rc = v.api.addressFree(v.va, v.va_len);
        if (free_rc != abi.success) std.log.err("CUDA VMM step cuMemAddressFree failed with CUresult {d}", .{free_rc});
        v.lib.close();
    }

    pub fn backend(v: *Vmm) Backend {
        return .{ .ctx = v, .granularity = v.granularity, .chunk = chunk, .release = release, .name = "vmm" };
    }

    /// Create one physical allocation handle. The caller owns the handle until it is mapped and released.
    pub fn createHandle(v: *Vmm, len: usize) Error!u64 {
        if (len == 0 or len % v.granularity != 0) return error.CudaFailed;
        var h: u64 = 0;
        const rc = v.api.create(&h, len, &v.prop, 0);
        if (rc != abi.success) return error.CudaFailed;
        return h;
    }

    /// Map offset zero of `handle` at an already-reserved virtual address.
    pub fn mapHandle(v: *Vmm, at: u64, len: usize, handle: u64) Error!void {
        if (len == 0 or len % v.granularity != 0) return error.CudaFailed;
        if (v.api.map(at, len, 0, handle, 0) != abi.success) return error.CudaFailed;
    }

    pub fn grantAccess(v: *Vmm, at: u64, len: usize) Error!void {
        if (len == 0 or len % v.granularity != 0) return error.CudaFailed;
        const desc: AccessDesc = .{ .loc_id = v.device };
        if (v.api.setAccess(at, len, &desc, 1) != abi.success) return error.CudaFailed;
    }

    pub fn releaseHandle(v: *Vmm, handle: u64) void {
        _ = v.api.release(handle);
    }

    pub fn unmapRange(v: *Vmm, at: u64, len: usize) void {
        if (len != 0) _ = v.api.unmap(at, len);
    }

    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const v: *Vmm = @ptrCast(@alignCast(ctx));
        if (v.va + v.va_len - v.next < len) return v.fail("the reserved range", 0);
        const at = v.next;
        const max_chunk = if (v.retain_handles) ((@as(usize, 4) << 30) / v.granularity) * v.granularity else @max(len, 1);
        if (max_chunk == 0) return v.fail("4 GiB chunk limit", 0);
        var offset: usize = 0;
        while (offset < len) {
            const chunk_len = @min(max_chunk, len - offset);
            if (v.retain_handles and v.handle_count == v.handles.len) {
                if (offset != 0) _ = v.api.unmap(at, offset);
                return v.fail("VMM handle tracking capacity", 0);
            }
            var h: u64 = 0;
            const rc = v.api.create(&h, chunk_len, &v.prop, 0);
            if (rc != abi.success) {
                if (offset != 0) _ = v.api.unmap(at, offset);
                return v.fail("cuMemCreate", rc);
            }
            const rm = v.api.map(at + offset, chunk_len, 0, h, 0);
            if (rm != abi.success) {
                _ = v.api.release(h);
                if (offset != 0) _ = v.api.unmap(at, offset);
                return v.fail("cuMemMap", rm);
            }
            if (v.retain_handles) {
                v.handles[v.handle_count] = h;
                v.handle_count += 1;
            } else {
                _ = v.api.release(h);
            }
            offset += chunk_len;
        }
        const desc: AccessDesc = .{ .loc_id = v.device };
        const ra = v.api.setAccess(at, len, &desc, 1);
        if (ra != abi.success) {
            _ = v.api.unmap(at, len);
            return v.fail("cuMemSetAccess", ra);
        }
        v.next += len;
        return at;
    }

    fn fail(v: *Vmm, call: []const u8, code: c_int) ?u64 {
        if (v.failed == null) v.failed = .{ .call = call, .code = code };
        return null;
    }

    fn release(ctx: *anyopaque, base: u64, len: usize) void {
        const v: *Vmm = @ptrCast(@alignCast(ctx));
        _ = v.api.unmap(base, len); // the virtual range is not reused: the reservation is sized for the process
    }
};

// -- tests (a fake backend: addresses only) ---------------------------------------------------------------------------

const Fake = struct {
    next: u64 = 1 << 40,
    chunks: u32 = 0,
    released: u32 = 0,
    fail: bool = false,

    fn backend(f: *Fake, g: usize) Backend {
        return .{ .ctx = f, .granularity = g, .chunk = chunk, .release = release, .name = "fake" };
    }
    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail) return null;
        const p = f.next;
        f.next += len;
        f.chunks += 1;
        return p;
    }
    fn release(ctx: *anyopaque, _: u64, _: usize) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }
};

test "arena: small requests pack 512-aligned into shared chunks, are reused by size, large ones get their own" {
    const t = std.testing;
    var f: Fake = .{};
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{ .small_max = 1 << 20, .small_chunk = 4 << 20 });
    defer a.deinit();
    const p1 = a.alloc(20480).?; // a norm weight
    const p2 = a.alloc(983040).?; // an mHC fn16
    const p3 = a.alloc(100).?;
    try t.expectEqual(p1 + 20480, p2);
    try t.expectEqual(p2 + 983040, p3);
    for ([_]u64{ p1, p2, p3 }) |p| try t.expectEqual(@as(u64, 0), p % 512);
    try t.expectEqual(@as(u32, 1), f.chunks);
    // a freed range comes back for the same rounded size
    try t.expect(a.free(p3));
    try t.expectEqual(p3, a.alloc(17).?);
    try t.expectEqual(@as(u64, 1), a.stats.reused);
    // three more 1 MiB ranges fit the 4 MiB chunk, the fourth opens a new shared chunk
    for (0..3) |_| _ = a.alloc(1 << 20).?;
    try t.expectEqual(@as(u32, 1), f.chunks);
    _ = a.alloc(1 << 20).?;
    try t.expectEqual(@as(u32, 2), f.chunks);
    // large: its own chunk, rounded to the granularity, released when freed
    const big = a.alloc(5 << 20).?;
    try t.expectEqual(@as(u32, 3), f.chunks);
    try t.expectEqual(@as(u64, 6 << 20), a.large.get(big).?);
    try t.expect(a.free(big));
    try t.expectEqual(@as(u32, 1), f.released);
    // not ours
    try t.expect(!a.free(12345));
    // a backend that cannot map: null, the caller allocates as before
    f.fail = true;
    try t.expectEqual(@as(?u64, null), a.alloc(64 << 20));
    try t.expectEqual(@as(u64, 1), a.stats.fallback);
}

test "arena: under DeviceBuffer.alloc through memory.setHook" {
    const t = std.testing;
    var f: Fake = .{};
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{});
    defer a.deinit();
    a.install();
    defer memory.setHook(null);
    const h = memory.currentHook().?;
    const p = h.alloc(h.ctx, 4096).?;
    try t.expect(h.free(h.ctx, p));
    try t.expect(!h.free(h.ctx, 7));
}

/// Slots of `m` that hold neither a key nor a tombstone.
fn freeSlots(m: *const std.AutoHashMapUnmanaged(u64, u64)) usize {
    const meta = m.metadata orelse return 0;
    var n: usize = 0;
    for (meta[0..m.capacity()]) |x| n += @intFromBool(x.isFree());
    return n;
}

test "arena: a long run of allocations and frees leaves both address maps free slots" {
    const t = std.testing;
    var f: Fake = .{};
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{ .small_max = 1 << 20, .small_chunk = 4 << 20 });
    defer a.deinit();
    var held: std.ArrayList(u64) = .empty;
    defer held.deinit(t.allocator);
    var rng = std.Random.DefaultPrng.init(3);
    for (0..20000) |_| {
        // a serving process's buffers come and go (prompt workspaces, grown buffers): small and large alike
        if (held.items.len > 0 and rng.random().boolean()) {
            const k = rng.random().uintLessThan(usize, held.items.len);
            try t.expect(a.free(held.swapRemove(k)));
        } else {
            const big = rng.random().uintLessThan(u32, 4) == 0;
            try held.append(t.allocator, a.alloc(if (big) 3 << 20 else 512 * (1 + rng.random().uintLessThan(u64, 64))).?);
        }
        for ([_]*const std.AutoHashMapUnmanaged(u64, u64){ &a.live, &a.large }) |m| {
            if (m.capacity() == 0) continue;
            try t.expect(freeSlots(m) * 10 >= m.capacity());
        }
    }
}

test "arena: a backend that maps nothing sends every request to cuMemAlloc and counts each fallback" {
    const t = std.testing;
    var f: Fake = .{ .fail = true };
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{ .small_max = 1 << 20, .small_chunk = 4 << 20 });
    defer a.deinit();
    try t.expectEqual(@as(?u64, null), a.alloc(8 << 20)); // a large request's own chunk
    try t.expectEqual(@as(?u64, null), a.alloc(4096)); // a small one's shared chunk
    try t.expectEqual(@as(u64, 2), a.stats.fallback);
    try t.expectEqual(@as(u64, 0), a.stats.chunk_bytes);
    try t.expect(!a.free(0x1000)); // nothing was the arena's: the caller frees as before
    f.fail = false; // the backend recovers: the next request maps again
    try t.expect(a.alloc(4096) != null);
    try t.expectEqual(@as(u64, 2), a.stats.fallback);
}

test "arena: VMM keeps the first failed driver call and its code, later failures leave it" {
    const t = std.testing;
    const Api = struct {
        var create_rc: abi.Result = 2; // CUDA_ERROR_OUT_OF_MEMORY
        var unmaps: u32 = 0;
        fn create(h: *u64, _: usize, _: *const Vmm.Prop, _: c_ulonglong) callconv(.c) abi.Result {
            h.* = 7;
            return create_rc;
        }
        fn ok1(_: u64) callconv(.c) abi.Result {
            return abi.success;
        }
        fn map(_: u64, _: usize, _: usize, _: u64, _: c_ulonglong) callconv(.c) abi.Result {
            return abi.success;
        }
        fn unmap(_: u64, _: usize) callconv(.c) abi.Result {
            unmaps += 1;
            return abi.success;
        }
        fn setAccess(_: u64, _: usize, _: *const Vmm.AccessDesc, _: usize) callconv(.c) abi.Result {
            return 1; // CUDA_ERROR_INVALID_VALUE
        }
    };
    var v: Vmm = .{ .lib = undefined, .device = 0, .prop = .{}, .granularity = 2 << 20, .va = 1 << 40, .va_len = 4 << 20, .next = 1 << 40, .api = undefined };
    v.api.create = Api.create;
    v.api.release = Api.ok1;
    v.api.map = Api.map;
    v.api.unmap = Api.unmap;
    v.api.setAccess = Api.setAccess;
    const be = v.backend();
    try t.expectEqual(@as(?u64, null), be.chunk(be.ctx, 2 << 20));
    try t.expectEqualStrings("cuMemCreate", v.failed.?.call);
    try t.expectEqual(@as(c_int, 2), v.failed.?.code);
    Api.create_rc = abi.success; // now access fails: the mapping is undone, the first failure stays
    try t.expectEqual(@as(?u64, null), be.chunk(be.ctx, 2 << 20));
    try t.expectEqual(@as(u32, 1), Api.unmaps);
    try t.expectEqualStrings("cuMemCreate", v.failed.?.call);
    try t.expectEqual(@as(u64, 1 << 40), v.next); // nothing was mapped
    v.failed = null; // past the reserved range: no driver call at all
    try t.expectEqual(@as(?u64, null), be.chunk(be.ctx, 8 << 20));
    try t.expectEqualStrings("the reserved range", v.failed.?.call);
}

test "VMM helpers record create, double map, and access arguments in order" {
    const Api = struct {
        var n: usize = 0;
        var calls: [3]u8 = .{ 0, 0, 0 };
        var at: [3]u64 = .{ 0, 0, 0 };
        var len: [3]usize = .{ 0, 0, 0 };
        fn create(h: *u64, bytes: usize, _: *const Vmm.Prop, _: c_ulonglong) callconv(.c) abi.Result {
            calls[n] = 1; len[n] = bytes; n += 1; h.* = 99; return abi.success;
        }
        fn map(address: u64, bytes: usize, offset: usize, handle: u64, flags: c_ulonglong) callconv(.c) abi.Result {
            _ = offset; _ = flags; _ = handle; calls[n] = 2; at[n] = address; len[n] = bytes; n += 1; return abi.success;
        }
        fn access(address: u64, bytes: usize, desc: *const Vmm.AccessDesc, count: usize) callconv(.c) abi.Result {
            _ = desc; _ = count; calls[n] = 3; at[n] = address; len[n] = bytes; n += 1; return abi.success;
        }
        fn ok(_: u64) callconv(.c) abi.Result { return abi.success; }
    };
    Api.n = 0;
    var v: Vmm = .{ .lib = undefined, .device = 7, .prop = .{ .loc_id = 7 }, .granularity = 2 << 20, .va = 1 << 40, .va_len = 8 << 20, .next = 1 << 40, .api = undefined };
    v.api.create = Api.create;
    v.api.map = Api.map;
    v.api.setAccess = Api.access;
    v.api.release = Api.ok;
    v.api.unmap = struct { fn f(_: u64, _: usize) callconv(.c) abi.Result { return abi.success; } }.f;
    const h = try v.createHandle(2 << 20);
    try v.mapHandle(1 << 40, 2 << 20, h);
    try v.grantAccess(1 << 40, 2 << 20);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, Api.calls[0..3]);
    try std.testing.expectEqual(@as(usize, 2 << 20), Api.len[0]);
    try std.testing.expectEqual(@as(u64, 1 << 40), Api.at[1]);
    try std.testing.expectEqual(@as(usize, 2 << 20), Api.len[2]);
}

test "Failure boundaries: a free range survives failure inserting its live address" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var f: Fake = .{};
    var a = Arena.init(fail.allocator(), f.backend(2 << 20), .{});
    defer a.deinit();
    const first = a.alloc(4096).?;
    try std.testing.expect(a.free(first));
    a.live.clearAndFree(fail.allocator());
    fail.fail_index = fail.alloc_index;
    try std.testing.expectEqual(@as(?u64, null), a.alloc(4096));
    fail.fail_index = std.math.maxInt(usize);
    const retried = a.alloc(4096).?;
    try std.testing.expectEqual(first, retried);
    try std.testing.expect(a.free(retried));
}
