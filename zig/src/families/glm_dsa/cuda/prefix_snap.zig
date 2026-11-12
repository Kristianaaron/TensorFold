// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Chunked save/restore after ashhart/TensorFold sessions/format.zig and disk.zig (Apache-2.0);
//! index/sweep after MiaAI-Lab patch 0132 (Apache-2.0, Copyright 2026 Mia's AI Lab).
const std = @import("std");
const cuda = @import("cuda");
const uma = @import("uma");
const kvsave = @import("kvsave");
const engine = @import("engine");
pub const prefix_cache = @import("prefix_cache.zig");

pub const max_jobs = 32;
pub const max_chain = 64;
pub const max_planes = 256;
pub const max_segments = max_planes + 2;
pub const path_bytes = 512;
pub const writer_slot0 = 8;
pub const reader_slots = 8;

pub const Plane = struct { base: u64, row_bytes: usize };
pub const Segment = struct { dev: u64, buf_off: usize, len: usize };
pub const RingView = struct { ptr: u64, len: usize, from: i64 };
pub const RingHooks = struct { ctx: *anyopaque, view_fn: *const fn (ctx: *anyopaque) anyerror!RingView };

pub const Layout = struct {
    planes: [max_planes]Plane = undefined,
    count: usize = 0,
    row_sum: usize = 0,

    pub fn add(l: *Layout, base: u64, row_bytes: usize) !void {
        if (l.count == max_planes or row_bytes == 0) return error.TooManyPlanes;
        l.planes[l.count] = .{ .base = base, .row_bytes = row_bytes };
        l.count += 1;
        l.row_sum += row_bytes;
    }

    pub fn bodyLen(l: *const Layout, ring_span: usize, lo: usize, hi: usize) usize {
        return ring_span + (hi - lo) * l.row_sum;
    }

    pub fn chunkSegments(l: *const Layout, ring: ?RingView, ring_span: usize, lo: usize, hi: usize, keep_lo: usize, chunk: usize, n: usize, out: []Segment) !usize {
        const begin = chunk * kvsave.chunk_bytes;
        const end = begin + n;
        var used: usize = 0;
        if (ring) |view| {
            const rb = @max(begin, 0);
            const re = @min(end, ring_span);
            if (rb < re and rb < view.len) {
                const actual_end = @min(re, view.len);
                try pushSegment(out, &used, .{ .dev = view.ptr + rb, .buf_off = rb - begin, .len = actual_end - rb });
            }
        }
        var section = ring_span;
        for (l.planes[0..l.count]) |plane| {
            const plane_len = (hi - lo) * plane.row_bytes;
            const pb = section;
            const pe = section + plane_len;
            const ob = @max(begin, pb);
            const oe = @min(end, pe);
            if (ob < oe) {
                const row_skip = (keep_lo -| lo) * plane.row_bytes;
                const from = @max(ob, pb + row_skip);
                if (from < oe) try pushSegment(out, &used, .{ .dev = plane.base + lo * plane.row_bytes + (from - pb), .buf_off = from - begin, .len = oe - from });
            }
            section = pe;
        }
        return used;
    }

    fn pushSegment(out: []Segment, used: *usize, sg: Segment) !void {
        if (used.* == out.len) return error.TooManySegments;
        out[used.*] = sg;
        used.* += 1;
    }
};

pub fn layoutFromState(st: anytype, latent_row: usize, rope_row: usize, index_row: usize) !Layout {
    var layout: Layout = .{};
    for (0..st.layers) |layer| {
        try layout.add(st.latentAt(layer), latent_row);
        try layout.add(st.ropeAt(layer), rope_row);
    }
    for (0..st.full_index_layers) |ordinal| try layout.add(st.indexAt(ordinal), index_row);
    return layout;
}

pub const Device = struct {
    ctx: *anyopaque,
    bind_thread_fn: *const fn (ctx: *anyopaque) anyerror!void,
    sync_main_fn: *const fn (ctx: *anyopaque) anyerror!void,
    mark_cut_fn: *const fn (ctx: *anyopaque, job: usize) anyerror!void,
    wait_cut_fn: *const fn (ctx: *anyopaque, job: usize) anyerror!void,
    save_copy_fn: *const fn (ctx: *anyopaque, dst: [*]u8, src: u64, n: usize) anyerror!void,
    save_record_fn: *const fn (ctx: *anyopaque, slot: usize) anyerror!void,
    save_wait_fn: *const fn (ctx: *anyopaque, slot: usize) anyerror!void,
    ring_copy_fn: *const fn (ctx: *anyopaque, dst: [*]u8, src: u64, n: usize) anyerror!void,
    load_copy_fn: *const fn (ctx: *anyopaque, dst: u64, src: [*]const u8, n: usize) anyerror!void,
    load_record_fn: *const fn (ctx: *anyopaque, slot: usize) anyerror!void,
    load_wait_fn: *const fn (ctx: *anyopaque, slot: usize) anyerror!void,
    load_finish_fn: *const fn (ctx: *anyopaque) anyerror!void,
};

pub const CudaDevice = struct {
    d: *const cuda.Driver,
    cu: *const cuda.Context,
    main: cuda.Stream,
    save: cuda.Stream,
    copy: cuda.Stream,
    cut_events: [max_jobs]cuda.Event,
    save_events: [2]cuda.Event,
    load_events: [reader_slots]cuda.Event,

    pub fn init(d: *const cuda.Driver, cu: *const cuda.Context, main: cuda.Stream) !CudaDevice {
        var result: CudaDevice = .{ .d = d, .cu = cu, .main = main, .save = try cuda.Stream.initPriority(d, .default), .copy = undefined, .cut_events = undefined, .save_events = undefined, .load_events = undefined };
        errdefer result.save.deinit();
        result.copy = try cuda.Stream.init(d, true);
        errdefer result.copy.deinit();
        var cuts_made: usize = 0;
        errdefer for (result.cut_events[0..cuts_made]) |*event| event.deinit();
        for (&result.cut_events) |*event| { event.* = try cuda.Event.init(d, false); cuts_made += 1; }
        var saves_made: usize = 0;
        errdefer for (result.save_events[0..saves_made]) |*event| event.deinit();
        for (&result.save_events) |*event| { event.* = try cuda.Event.init(d, false); saves_made += 1; }
        var loads_made: usize = 0;
        errdefer for (result.load_events[0..loads_made]) |*event| event.deinit();
        for (&result.load_events) |*event| { event.* = try cuda.Event.init(d, false); loads_made += 1; }
        return result;
    }

    pub fn deinit(c: *CudaDevice) void {
        for (&c.load_events) |*event| event.deinit();
        for (&c.save_events) |*event| event.deinit();
        for (&c.cut_events) |*event| event.deinit();
        c.copy.deinit();
        c.save.deinit();
        c.* = undefined;
    }

    pub fn device(c: *CudaDevice) Device {
        return .{ .ctx = c, .bind_thread_fn = bind, .sync_main_fn = syncMain, .mark_cut_fn = markCut, .wait_cut_fn = waitCut, .save_copy_fn = saveCopy, .save_record_fn = saveRecord, .save_wait_fn = saveWait, .ring_copy_fn = ringCopy, .load_copy_fn = loadCopy, .load_record_fn = loadRecord, .load_wait_fn = loadWait, .load_finish_fn = loadFinish };
    }

    fn self(ctx: *anyopaque) *CudaDevice { return @ptrCast(@alignCast(ctx)); }
    fn bind(ctx: *anyopaque) anyerror!void { try self(ctx).cu.makeCurrent(); }
    fn syncMain(ctx: *anyopaque) anyerror!void { try self(ctx).main.synchronize(); }
    fn markCut(ctx: *anyopaque, job: usize) anyerror!void { try self(ctx).cut_events[job].record(self(ctx).main); }
    fn waitCut(ctx: *anyopaque, job: usize) anyerror!void { const c = self(ctx); try c.save.wait(c.cut_events[job]); }
    fn saveCopy(ctx: *anyopaque, dst: [*]u8, src: u64, n: usize) anyerror!void { const c = self(ctx); try c.d.check(c.d.api.cuMemcpyDtoHAsync_v2(dst, src, n, c.save.handle), "prefix cache save copy"); }
    fn saveRecord(ctx: *anyopaque, slot: usize) anyerror!void { const c = self(ctx); try c.save_events[slot].record(c.save); }
    fn saveWait(ctx: *anyopaque, slot: usize) anyerror!void { try self(ctx).save_events[slot].synchronize(); }
    fn ringCopy(ctx: *anyopaque, dst: [*]u8, src: u64, n: usize) anyerror!void { const c = self(ctx); try c.d.check(c.d.api.cuMemcpyDtoHAsync_v2(dst, src, n, c.copy.handle), "prefix cache ring copy"); try c.copy.synchronize(); }
    fn loadCopy(ctx: *anyopaque, dst: u64, src: [*]const u8, n: usize) anyerror!void { const c = self(ctx); try c.d.check(c.d.api.cuMemcpyHtoDAsync_v2(dst, src, n, c.copy.handle), "prefix cache load copy"); }
    fn loadRecord(ctx: *anyopaque, slot: usize) anyerror!void { const c = self(ctx); try c.load_events[slot].record(c.copy); }
    fn loadWait(ctx: *anyopaque, slot: usize) anyerror!void { try self(ctx).load_events[slot].synchronize(); }
    fn loadFinish(ctx: *anyopaque) anyerror!void { try self(ctx).copy.synchronize(); }
};

pub const HostDevice = struct {
    copies: usize = 0,
    pub fn device(h: *HostDevice) Device { return .{ .ctx = h, .bind_thread_fn = noop, .sync_main_fn = noop, .mark_cut_fn = eventNoop, .wait_cut_fn = eventNoop, .save_copy_fn = copyDtoh, .save_record_fn = eventNoop, .save_wait_fn = eventNoop, .ring_copy_fn = copyDtoh, .load_copy_fn = copyHtod, .load_record_fn = eventNoop, .load_wait_fn = eventNoop, .load_finish_fn = noop }; }
    fn self(ctx: *anyopaque) *HostDevice { return @ptrCast(@alignCast(ctx)); }
    fn noop(_: *anyopaque) anyerror!void {}
    fn eventNoop(_: *anyopaque, _: usize) anyerror!void {}
    fn copyDtoh(ctx: *anyopaque, dst: [*]u8, src: u64, n: usize) anyerror!void { const h = self(ctx); @memcpy(dst[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]); h.copies += 1; }
    fn copyHtod(ctx: *anyopaque, dst: u64, src: [*]const u8, n: usize) anyerror!void { const h = self(ctx); @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], src[0..n]); h.copies += 1; }
};

pub fn jobMetaBytes(chunks: usize) usize {
    return 4 * kvsave.alignment + path_bytes + 8 + chunks * (@sizeOf(kvsave.Digest) + 1) + kvsave.headerBytes(chunks);
}

/// Reader.open retains its digest table and sector/header scratch in the fixed allocator.
pub fn loadMetaBytes(ring_span: usize, job_max_chunks: usize) usize {
    const total = max_chain * (ring_span / kvsave.chunk_bytes + 1) + job_max_chunks;
    return max_chain * 6 * kvsave.alignment + total * 2 * @sizeOf(kvsave.Digest);
}

pub const Fault = enum { none, begin, ring, body, seal, publish };
const JobState = enum(u8) { free, queued, done };
const Job = struct {
    state: JobState = .free,
    ok: bool = false,
    has_writer: bool = false,
    writer: kvsave.Writer = undefined,
    fba: std.heap.FixedBufferAllocator = undefined,
    path_buf: [path_bytes]u8 = undefined,
    path: [:0]const u8 = "",
    key: u64 = 0,
    parent_key: u64 = 0,
    base_at: usize = 0,
    at: usize = 0,
    seq: u64 = 0,
    bytes: u64 = 0,
    lo: usize = 0,
    hi: usize = 0,
    body_len: usize = 0,
};

pub fn Snap(comptime E: type) type {
    return struct {
        gpa: std.mem.Allocator,
        io: std.Io,
        runtime: *prefix_cache.Runtime,
        engine_ptr: *E,
        device: Device,
        ring: ?RingHooks,
        layout: Layout,
        slots: [10][]align(kvsave.alignment) u8,
        ring_span: usize,
        job_max_chunks: usize,
        reader_workers: usize,
        jobs: [max_jobs]Job = @splat(.{}),
        job_meta: []align(kvsave.alignment) u8,
        job_meta_bytes: usize,
        pending: [max_jobs]u8 = @splat(0),
        pending_len: usize = 0,
        queue: [max_jobs]u8 = @splat(0),
        q_head: usize = 0,
        q_len: usize = 0,
        mutex: std.Io.Mutex = .init,
        work_cond: std.Io.Condition = .init,
        done_cond: std.Io.Condition = .init,
        stop: bool = false,
        thread: ?std.Thread = null,
        bind_failed: bool = false,
        disabled: bool = false,
        fault: Fault = .none,
        free_override: ?u64 = null,
        cache_dir_buf: [path_bytes]u8 = undefined,
        cache_dir: [:0]const u8 = "",
        saves: u64 = 0,
        skips: u64 = 0,
        chain: [max_chain]prefix_cache.Node = undefined,
        chain_len: usize = 0,
        readers: [max_chain]kvsave.Reader = undefined,
        reader_open: [max_chain]bool = @splat(false),
        load_paths: [max_chain][path_bytes]u8 = undefined,
        chunk_base: [max_chain + 1]usize = undefined,
        node_failed: [max_chain]std.atomic.Value(bool) = undefined,
        next_task: std.atomic.Value(usize) = .init(0),
        load_keep_lo: usize = 0,
        load_leaf: usize = 0,
        chain_len_loaded: usize = 0,
        load_ring: ?RingView = null,
        load_meta: []align(kvsave.alignment) u8 = &.{},
        load_fba: std.heap.FixedBufferAllocator = undefined,
        restores: u64 = 0,
        fallbacks: u64 = 0,
        worker_threads: [3]?std.Thread = @splat(null),

        const Self = @This();

        pub fn init(s: *Self, gpa: std.mem.Allocator, io: std.Io, runtime: *prefix_cache.Runtime, engine_ptr: *E, device: Device, ring: ?RingHooks, layout: Layout, slots: [10][]align(kvsave.alignment) u8, capacity: usize, reader_workers: usize) !void {
            if (runtime.root.len > path_bytes - 112) return error.PrefixCacheDirTooLong;
            const ring_span = if (ring) |hooks| std.mem.alignForward(usize, (try hooks.view_fn(hooks.ctx)).len, kvsave.chunk_bytes) else 0;
            const local_cap = (capacity + 1) / 2;
            const chunks = kvsave.digestCount(layout.bodyLen(ring_span, 0, local_cap));
            const workers: usize = @max(@as(usize, 1), @min(reader_workers, @as(usize, 4)));
            const meta_bytes = jobMetaBytes(chunks);
            const meta = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), max_jobs * meta_bytes);
            errdefer gpa.free(meta);
            const load_meta = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), loadMetaBytes(ring_span, chunks));
            errdefer gpa.free(load_meta);
            const hex = std.fmt.bytesToHex(runtime.identity, .lower);
            const dir = try std.mem.printSentinel(&s.cache_dir_buf, "{s}/{s}/rank{d}", .{ runtime.root, &hex, runtime.rank }, 0);
            s.* = .{ .gpa = gpa, .io = io, .runtime = runtime, .engine_ptr = engine_ptr, .device = device, .ring = ring, .layout = layout, .slots = slots, .ring_span = ring_span, .job_max_chunks = chunks, .reader_workers = workers, .job_meta = meta, .job_meta_bytes = meta_bytes, .cache_dir = dir, .load_meta = load_meta, .load_fba = .init(load_meta) };
            std.log.info("prefix cache: staging 40 MiB pinned + {d} MiB job metadata", .{@as(usize, @intCast((max_jobs * meta_bytes + (1 << 20) - 1) >> 20))});
            s.thread = try std.Thread.spawn(.{}, writerMain, .{s});
        }

        pub fn deinit(s: *Self) void {
            s.mutex.lockUncancelable(s.io);
            s.stop = true;
            s.work_cond.broadcast(s.io);
            s.mutex.unlock(s.io);
            if (s.thread) |thread| thread.join();
            for (s.jobs[0..]) |*job| if (job.has_writer) job.writer.abort();
            s.closeReaders();
            s.gpa.free(s.job_meta);
            s.gpa.free(s.load_meta);
            s.* = undefined;
        }

        pub fn ops(s: *Self) engine.SnapOps { return .{ .ctx = s, .save_fn = saveThunk, .load_fn = loadThunk, .commit_fn = commitThunk }; }

        pub fn save(s: *Self, cut: engine.SnapCut) !void {
            const align_rows = prefix_cache.cutGrain(s.runtime.options.expert_rows, s.runtime.options.engine_piece_rows) catch return error.SnapInvariant;
            var slot: ?usize = null;
            for (s.jobs, 0..) |job, idx| {
                if (job.state == .free) { slot = idx; break; }
            }
            var key_ok = !s.disabled and cut.at == s.engine_ptr.pos and s.engine_ptr.pos == s.engine_ptr.frontier and cut.at % align_rows == 0 and cut.parent_at < cut.at and cut.at <= s.engine_ptr.history.items.len and kvsave.keyOf(s.engine_ptr.history.items[0..cut.at]) == cut.key and s.runtime.index.get(cut.key) == null and cut.seq == s.runtime.index.next_seq and s.pending_len < max_jobs and slot != null;
            if (cut.parent_key == 0) key_ok = key_ok and cut.parent_at == 0 else if (s.runtime.index.get(cut.parent_key)) |parent| key_ok = key_ok and parent.at == cut.parent_at else key_ok = false;
            const free_bytes = s.free_override orelse kvsave.freeBytes(s.cache_dir) orelse 0;
            const free_mib: u32 = @intCast(@min(free_bytes >> 20, @as(u64, std.math.maxInt(u32))));
            var words = [2]u32{ free_mib, @intFromBool(key_ok) };
            try s.engine_ptr.agreeWords(&words);
            if (words[1] == 0) { s.skips += 1; std.log.warn("prefix cache rank {d}: save skipped c={d} reason=key", .{ s.runtime.rank, cut.at }); return; }
            const lo = kvsave.localRows(@intCast(s.runtime.rank), cut.parent_at) catch return error.SnapInvariant;
            const hi = kvsave.localRows(@intCast(s.runtime.rank), cut.at) catch return error.SnapInvariant;
            const body_len = s.layout.bodyLen(s.ring_span, lo, hi);
            const bytes = kvsave.fileBytes(body_len, cut.at - cut.parent_at);
            if (!s.runtime.room(bytes, @as(u64, words[0]) << 20)) { s.skips += 1; std.log.warn("prefix cache rank {d}: save skipped c={d} reason=budget", .{ s.runtime.rank, cut.at }); return; }
            const job_index = slot.?;
            const node: prefix_cache.Node = .{ .key = cut.key, .at = cut.at, .alignment = align_rows, .base_at = cut.parent_at, .parent_key = cut.parent_key, .seq = cut.seq, .bytes = bytes, .pending = true };
            s.runtime.index.insertPending(node) catch |err| {
                if (err == error.IndexFull) { s.skips += 1; return; }
                return error.SnapInvariant;
            };
            s.runtime.used_bytes +|= bytes;
            s.runtime.chain_key = cut.key;
            s.runtime.chain_at = cut.at;
            const job = &s.jobs[job_index];
            job.* = .{};
            job.key = cut.key; job.parent_key = cut.parent_key; job.base_at = cut.parent_at; job.at = cut.at; job.seq = cut.seq; job.bytes = bytes; job.lo = lo; job.hi = hi; job.body_len = body_len;
            const meta_start = job_index * s.job_meta_bytes;
            job.fba = std.heap.FixedBufferAllocator.init(s.job_meta[meta_start..][0..s.job_meta_bytes]);
            const root = s.runtime.root;
            const identity = s.runtime.identity;
            const rank = s.runtime.rank;
            job.path = kvsave.pathInto(&job.path_buf, root, identity, rank, cut.key) catch |err| {
                std.log.warn("prefix cache rank {d}: path formatting failed: {s}", .{ rank, @errorName(err) });
                job.ok = false;
                queueJob(s, job_index);
                s.runtime.pending_count = s.pending_len;
                s.runtime.pending_max_at = @max(s.runtime.pending_max_at, cut.at);
                return;
            };
            job.ok = true;
            if (s.fault == .begin) { std.log.warn("prefix cache rank {d}: writer begin failed: {s}", .{ rank, "InjectedFault" }); job.ok = false; }
            if (job.ok) {
                const view = if (s.ring) |hooks| hooks.view_fn(hooks.ctx) catch |err| blk: {
                    std.log.warn("prefix cache rank {d}: ring view failed: {s}", .{ rank, @errorName(err) });
                    job.ok = false;
                    break :blk RingView{ .ptr = 0, .len = 0, .from = 0 };
                } else RingView{ .ptr = 0, .len = 0, .from = 0 };
                s.engine_ptr.drainDraftFill() catch |err| { std.log.warn("prefix cache rank {d}: draft drain failed: {s}", .{ rank, @errorName(err) }); job.ok = false; };
                s.device.sync_main_fn(s.device.ctx) catch |err| { std.log.warn("prefix cache rank {d}: main stream sync failed: {s}", .{ rank, @errorName(err) }); job.ok = false; };
                if (job.ok) {
                    job.writer = kvsave.Writer.begin(job.fba.allocator(), job.path, identity, rank, 2, s.engine_ptr.history.items, cut.parent_at, cut.at, cut.parent_key, cut.seq, body_len, view.from, s.ring_span) catch |err| blk: {
                        std.log.warn("prefix cache rank {d}: writer begin failed: {s}", .{ rank, @errorName(err) });
                        job.ok = false;
                        break :blk undefined;
                    };
                    job.has_writer = job.ok;
                }
                if (job.ok and s.fault == .ring) { std.log.warn("prefix cache rank {d}: ring capture failed: {s}", .{ rank, "InjectedFault" }); job.ok = false; }
                if (job.ok) {
                    for (0..s.ring_span / kvsave.chunk_bytes) |chunk| {
                        const off = chunk * kvsave.chunk_bytes;
                        const real: usize = if (off < view.len) @min(kvsave.chunk_bytes, view.len - off) else 0;
                        const stage = s.slots[0];
                        if (real > 0) s.device.ring_copy_fn(s.device.ctx, stage.ptr, view.ptr + off, real) catch |err| { std.log.warn("prefix cache rank {d}: ring copy failed: {s}", .{ rank, @errorName(err) }); job.ok = false; break; };
                        @memset(stage[real..kvsave.chunk_bytes], 0);
                        job.writer.writeChunkAt(chunk, stage, kvsave.chunk_bytes) catch |err| { std.log.warn("prefix cache rank {d}: ring write failed: {s}", .{ rank, @errorName(err) }); job.ok = false; break; };
                    }
                }
                if (job.ok) s.device.mark_cut_fn(s.device.ctx, job_index) catch |err| { std.log.warn("prefix cache rank {d}: cut event failed: {s}", .{ rank, @errorName(err) }); job.ok = false; };
            }
            queueJob(s, job_index);
            s.runtime.pending_count = s.pending_len;
            s.runtime.pending_max_at = @max(s.runtime.pending_max_at, cut.at);
            s.saves += 1;
            std.log.info("prefix cache rank {d}: save c={d} kind={d} base={d} bytes={d}", .{ rank, cut.at, cut.kind, cut.parent_at, bytes });
        }

        fn queueJob(s: *Self, job_index: usize) void {
            s.pending[s.pending_len] = @intCast(job_index);
            s.pending_len += 1;
            s.mutex.lockUncancelable(s.io);
            s.jobs[job_index].state = .queued;
            const tail = (s.q_head + s.q_len) % max_jobs;
            s.queue[tail] = @intCast(job_index);
            s.q_len += 1;
            s.work_cond.signal(s.io);
            s.mutex.unlock(s.io);
        }

        fn writerMain(s: *Self) void {
            s.device.bind_thread_fn(s.device.ctx) catch |err| { s.bind_failed = true; std.log.warn("prefix cache rank {d}: writer context bind failed: {s}", .{ s.runtime.rank, @errorName(err) }); };
            while (true) {
                s.mutex.lockUncancelable(s.io);
                while (s.q_len == 0 and !s.stop) s.work_cond.waitUncancelable(s.io, &s.mutex);
                if (s.stop and s.q_len == 0) { s.mutex.unlock(s.io); return; }
                const job_index = s.queue[s.q_head];
                s.q_head = (s.q_head + 1) % max_jobs;
                s.q_len -= 1;
                s.runtime.writer_idle.store(false, .release);
                s.mutex.unlock(s.io);
                var job_ok = s.jobs[job_index].ok and !s.bind_failed;
                if (job_ok) writeBody(s, job_index) catch |err| {
                    std.log.warn("prefix cache rank {d}: body write failed: {s}", .{ s.runtime.rank, @errorName(err) });
                    job_ok = false;
                };
                s.mutex.lockUncancelable(s.io);
                s.jobs[job_index].ok = job_ok;
                s.jobs[job_index].state = .done;
                if (s.q_len == 0) s.runtime.writer_idle.store(true, .release);
                s.done_cond.broadcast(s.io);
                s.mutex.unlock(s.io);
            }
        }

        fn writeBody(s: *Self, job_index: usize) !void {
            const job = &s.jobs[job_index];
            if (s.fault == .body) return error.InjectedFault;
            try s.device.wait_cut_fn(s.device.ctx, job_index);
            const first = s.ring_span / kvsave.chunk_bytes;
            const count = kvsave.digestCount(job.body_len);
            var previous: ?struct { chunk: usize, slot: usize, len: usize } = null;
            var chunk = first;
            while (chunk < count) : (chunk += 1) {
                const slot: usize = (chunk - first) % 2;
                const buf = s.slots[writer_slot0 + slot];
                const len: usize = if (chunk + 1 == count) job.body_len - chunk * kvsave.chunk_bytes else kvsave.chunk_bytes;
                var segments: [max_segments]Segment = undefined;
                const used = try s.layout.chunkSegments(null, s.ring_span, job.lo, job.hi, job.lo, chunk, len, &segments);
                for (segments[0..used]) |segment| try s.device.save_copy_fn(s.device.ctx, buf[segment.buf_off..].ptr, segment.dev, segment.len);
                try s.device.save_record_fn(s.device.ctx, slot);
                if (previous) |prev| { try s.device.save_wait_fn(s.device.ctx, prev.slot); try job.writer.writeChunkAt(prev.chunk, s.slots[writer_slot0 + prev.slot], prev.len); }
                previous = .{ .chunk = chunk, .slot = slot, .len = len };
            }
            if (previous) |prev| { try s.device.save_wait_fn(s.device.ctx, prev.slot); try job.writer.writeChunkAt(prev.chunk, s.slots[writer_slot0 + prev.slot], prev.len); }
            if (s.fault == .seal) return error.InjectedFault;
            try job.writer.seal();
        }

        pub fn commit(s: *Self, n: usize) !void {
            if (n == 0) return;
            const join_t0 = std.Io.Clock.awake.now(s.io).toNanoseconds();
            s.mutex.lockUncancelable(s.io);
            while (hasIncomplete(s)) s.done_cond.waitUncancelable(s.io, &s.mutex);
            s.mutex.unlock(s.io);
            const join_ms = @divTrunc(std.Io.Clock.awake.now(s.io).toNanoseconds() - join_t0, 1_000_000);
            const match = s.pending_len == n and n <= max_jobs;
            var sealed: [64]u32 = @splat(0);
            if (match) {
                for (0..n) |idx| sealed[idx] = @intFromBool(s.jobs[s.pending[idx]].ok);
            }
            try s.engine_ptr.agreeWords(&sealed);
            var published: [64]u32 = @splat(0);
            var local_published: [64]u32 = @splat(0);
            for (0..s.pending_len) |idx| {
                const ji = s.pending[idx];
                const job = &s.jobs[ji];
                var parent_ok = job.parent_key == 0;
                if (job.parent_key != 0) {
                    if (s.runtime.index.get(job.parent_key)) |parent| parent_ok = !parent.pending;
                    for (0..idx) |prior| if (s.jobs[s.pending[prior]].key == job.parent_key and published[prior] == 1) { parent_ok = true; };
                }
                if (match and sealed[idx] == 1 and parent_ok and s.fault != .publish) {
                    if (job.writer.publish()) |_| {
                        published[idx] = 1;
                        local_published[idx] = 1;
                    } else |err| {
                        std.log.warn("prefix cache rank {d}: publish failed: {s}", .{ s.runtime.rank, @errorName(err) });
                        _ = std.c.unlink(job.path.ptr);
                    }
                } else {
                    if (s.fault == .publish and match and sealed[idx] == 1 and parent_ok) std.log.warn("prefix cache rank {d}: publish failed: {s}", .{ s.runtime.rank, "InjectedFault" });
                    if (job.has_writer) job.writer.abort();
                }
                job.has_writer = false;
            }
            try s.engine_ptr.agreeWords(&published);
            var keep: [max_jobs]bool = @splat(false);
            for (0..s.pending_len) |idx| keep[idx] = match and published[idx] == 1;
            for (0..s.pending_len) |idx| {
                if (keep[idx] and s.jobs[s.pending[idx]].parent_key != 0) {
                    for (0..idx) |prior| {
                        if (!keep[prior] and s.jobs[s.pending[prior]].key == s.jobs[s.pending[idx]].parent_key) keep[idx] = false;
                    }
                }
            }
            for (0..s.pending_len) |idx| if (keep[idx]) s.runtime.index.commitPending(s.jobs[s.pending[idx]].key);
            var rev = s.pending_len;
            while (rev > 0) { rev -= 1; if (!keep[rev]) {
                const ji = s.pending[rev];
                const job = &s.jobs[ji];
                if (local_published[rev] == 1 or published[rev] == 1) _ = std.c.unlink(job.path.ptr);
                s.runtime.index.dropPending(job.key);
                s.runtime.used_bytes -|= job.bytes;
            } }
            for (0..s.pending_len) |idx| { const ji = s.pending[idx]; s.jobs[ji].state = .free; s.jobs[ji].fba.reset(); }
            const kept_count = countTrue(keep[0..s.pending_len]);
            std.log.info("prefix cache rank {d}: commit n={d} kept={d} dropped={d} join_ms={d}", .{ s.runtime.rank, n, kept_count, s.pending_len - kept_count, join_ms });
            s.pending_len = 0; s.runtime.pending_count = 0; s.runtime.pending_max_at = 0;
        }

        fn hasIncomplete(s: *Self) bool { for (s.pending[0..s.pending_len]) |idx| if (s.jobs[idx].state != .done) return true; return false; }
        fn countTrue(items: []const bool) usize { var count: usize = 0; for (items) |value| count += @intFromBool(value); return count; }
        pub fn load(s: *Self, r: usize, ids: []const u32, leaf_key: u64) !usize {
            const align_rows = prefix_cache.cutGrain(s.runtime.options.expert_rows, s.runtime.options.engine_piece_rows) catch 0;
            s.closeReaders();
            s.load_fba.reset();
            s.chain_len = 0;
            s.reader_open = @splat(false);
            var words: [68]u32 = @splat(0);
            var local_ok = !s.disabled and align_rows != 0 and ids.len % align_rows == 0 and ids.len > r and r % 512 == 0 and s.engine_ptr.pos == r and s.engine_ptr.frontier == r and kvsave.keyOf(ids) == leaf_key;
            if (local_ok) {
                s.chain_len = s.runtime.index.chainAbove(leaf_key, r, &s.chain) catch |err| blk: {
                    std.log.warn("prefix cache rank {d}: chain lookup failed: {s}", .{ s.runtime.rank, @errorName(err) });
                    break :blk 0;
                };
                if (s.chain_len == 0) local_ok = false;
            }
            if (local_ok and s.chain[s.chain_len - 1].at != ids.len) {
                s.chain_len = 0;
                local_ok = false;
            }
            if (local_ok) for (0..s.chain_len) |idx| {
                const node = s.chain[idx];
                const lo = kvsave.localRows(@intCast(s.runtime.rank), node.base_at) catch {
                    std.log.warn("prefix cache rank {d}: invalid base row for {x}", .{ s.runtime.rank, node.key });
                    break;
                };
                const hi = kvsave.localRows(@intCast(s.runtime.rank), node.at) catch {
                    std.log.warn("prefix cache rank {d}: invalid end row for {x}", .{ s.runtime.rank, node.key });
                    break;
                };
                const path = kvsave.pathInto(&s.load_paths[idx], s.runtime.root, s.runtime.identity, s.runtime.rank, node.key) catch |err| {
                    std.log.warn("prefix cache rank {d}: path formatting failed: {s}", .{ s.runtime.rank, @errorName(err) });
                    break;
                };
                s.readers[idx] = kvsave.Reader.open(s.load_fba.allocator(), path, s.runtime.identity, s.runtime.rank, 2, ids, node.base_at, node.at, s.layout.bodyLen(s.ring_span, lo, hi)) catch |err| {
                    std.log.warn("prefix cache rank {d}: header verify failed key={x}: {s}", .{ s.runtime.rank, node.key, @errorName(err) });
                    break;
                };
                s.reader_open[idx] = true;
                if (s.readers[idx].header.ring_span != s.ring_span or s.readers[idx].header.parent_key != node.parent_key) {
                    std.log.warn("prefix cache rank {d}: incompatible header key={x}", .{ s.runtime.rank, node.key });
                    break;
                }
                words[idx] = 1;
            };
            words[64] = @truncate(s.chain_len);
            words[65] = ~words[64];
            words[66] = @truncate(ids.len / 512);
            words[67] = ~words[66];
            try s.engine_ptr.agreeWords(&words);
            if (words[64] != ~words[65] or words[66] != ~words[67]) {
                s.disabled = true;
                std.log.warn("prefix cache disabled: rank indexes diverged", .{});
                s.closeReaders();
                s.fallbacks += 1;
                return r;
            }
            const agreed_len: usize = words[64];
            var verified: usize = 0;
            while (verified < agreed_len and words[verified] == 1) : (verified += 1) {}
            if (verified < agreed_len and verified < s.chain_len) _ = s.runtime.dropSubtree(s.chain[verified].key);
            if (verified == 0) {
                s.closeReaders();
                s.fallbacks += 1;
                std.log.info("prefix cache rank {d}: fallback reason=verify", .{s.runtime.rank});
                return r;
            }
            const restore_at = s.chain[verified - 1].at;
            s.load_leaf = verified - 1;
            s.load_keep_lo = kvsave.localRows(@intCast(s.runtime.rank), r) catch return error.SnapInvariant;
            var pre_ok = restore_at % 512 == 0 and restore_at % align_rows == 0;
            if (comptime @hasField(E, "drafter")) {
                if (s.engine_ptr.drafter != null) {
                    pre_ok = pre_ok and s.ring != null and s.readers[s.load_leaf].header.ring_span > 0;
                    if (s.engine_ptr.drafter.?.ring_adopt_fn == null) pre_ok = false;
                } else pre_ok = pre_ok and s.ring == null;
            }
            for (0..verified) |idx| s.node_failed[idx].store(!pre_ok, .monotonic);
            s.chunk_base[0] = 0;
            for (0..verified) |idx| s.chunk_base[idx + 1] = s.chunk_base[idx] + s.readers[idx].digests.len;
            s.chain_len_loaded = verified;
            s.load_ring = null;
            if (pre_ok and s.ring != null) s.load_ring = s.ring.?.view_fn(s.ring.?.ctx) catch |err| blk: {
                std.log.warn("prefix cache rank {d}: ring view failed: {s}", .{ s.runtime.rank, @errorName(err) });
                for (0..verified) |idx| s.node_failed[idx].store(true, .monotonic);
                break :blk null;
            };
            s.device.sync_main_fn(s.device.ctx) catch |err| {
                std.log.warn("prefix cache rank {d}: main stream sync failed: {s}", .{ s.runtime.rank, @errorName(err) });
                for (0..verified) |idx| s.node_failed[idx].store(true, .monotonic);
            };
            if (pre_ok) {
                s.next_task.store(0, .monotonic);
                const thread_count = s.reader_workers - 1;
                for (0..thread_count) |idx| {
                    s.worker_threads[idx] = std.Thread.spawn(.{}, loadWorker, .{ s, idx + 1 }) catch |err| {
                        std.log.warn("prefix cache rank {d}: load worker spawn failed: {s}", .{ s.runtime.rank, @errorName(err) });
                        for (0..verified) |node_idx| s.node_failed[node_idx].store(true, .monotonic);
                        break;
                    };
                }
                loadWorker(s, 0);
                for (&s.worker_threads) |*worker| if (worker.*) |thread| { thread.join(); worker.* = null; };
            }
            s.device.load_finish_fn(s.device.ctx) catch |err| {
                std.log.warn("prefix cache rank {d}: load stream finish failed: {s}", .{ s.runtime.rank, @errorName(err) });
                for (0..verified) |idx| s.node_failed[idx].store(true, .monotonic);
            };
            var body: [64]u32 = @splat(0);
            for (0..verified) |idx| body[idx] = @intFromBool(!s.node_failed[idx].load(.monotonic));
            try s.engine_ptr.agreeWords(&body);
            const ring_from = s.readers[s.load_leaf].header.ring_from;
            s.closeReaders();
            var body_ok: usize = 0;
            while (body_ok < verified and body[body_ok] == 1) : (body_ok += 1) {}
            if (body_ok == verified) {
                s.engine_ptr.restoreToDevice(restore_at, ids, if (s.ring != null) ring_from else null) catch return error.SnapInvariant;
                s.runtime.index.touchChain(s.chain[s.load_leaf].key);
                s.restores += 1;
                std.log.info("prefix cache rank {d}: restore at={d} from={d} files={d}", .{ s.runtime.rank, restore_at, r, verified });
                return restore_at;
            }
            s.engine_ptr.restoreFailed(r) catch return error.SnapInvariant;
            if (body_ok < verified) _ = s.runtime.dropSubtree(s.chain[body_ok].key);
            s.fallbacks += 1;
            std.log.warn("prefix cache rank {d}: fallback reason=body node={d}", .{ s.runtime.rank, body_ok });
            return r;
        }

        fn closeReaders(s: *Self) void {
            for (&s.reader_open, 0..) |*opened, idx| if (opened.*) {
                s.readers[idx].close();
                opened.* = false;
            };
        }

        fn loadWorker(s: *Self, worker: usize) void {
            s.device.bind_thread_fn(s.device.ctx) catch |err| {
                std.log.warn("prefix cache rank {d}: load worker bind failed: {s}", .{ s.runtime.rank, @errorName(err) });
                for (0..s.chain_len_loaded) |idx| s.node_failed[idx].store(true, .monotonic);
                return;
            };
            var uses: usize = 0;
            const total = s.chunk_base[s.chain_len_loaded];
            while (true) {
                const task = s.next_task.fetchAdd(1, .monotonic);
                if (task >= total) break;
                var idx: usize = 0;
                while (idx + 1 < s.chain_len_loaded and task >= s.chunk_base[idx + 1]) : (idx += 1) {}
                if (s.node_failed[idx].load(.monotonic)) continue;
                const chunk_idx = task - s.chunk_base[idx];
                const node = s.chain[idx];
                const lo = kvsave.localRows(@intCast(s.runtime.rank), node.base_at) catch { s.node_failed[idx].store(true, .monotonic); continue; };
                const hi = kvsave.localRows(@intCast(s.runtime.rank), node.at) catch { s.node_failed[idx].store(true, .monotonic); continue; };
                const body_len = s.layout.bodyLen(s.ring_span, lo, hi);
                const chunk_len: usize = if (chunk_idx + 1 == s.readers[idx].digests.len) body_len - chunk_idx * kvsave.chunk_bytes else kvsave.chunk_bytes;
                var segs: [max_segments]Segment = undefined;
                const ring_view = if (idx == s.load_leaf and s.ring != null) s.load_ring else null;
                const seg_count = s.layout.chunkSegments(ring_view, s.ring_span, lo, hi, s.load_keep_lo, chunk_idx, chunk_len, &segs) catch {
                    s.node_failed[idx].store(true, .monotonic);
                    continue;
                };
                if (seg_count == 0) continue;
                const slot = worker * 2 + uses % 2;
                uses += 1;
                s.device.load_wait_fn(s.device.ctx, slot) catch { s.node_failed[idx].store(true, .monotonic); continue; };
                const data = s.readers[idx].readChunkAt(chunk_idx, s.slots[slot]) catch |err| {
                    std.log.warn("prefix cache rank {d}: chunk verify failed node={d} chunk={d}: {s}", .{ s.runtime.rank, idx, chunk_idx, @errorName(err) });
                    s.node_failed[idx].store(true, .monotonic);
                    continue;
                };
                var copied = true;
                for (segs[0..seg_count]) |segment| s.device.load_copy_fn(s.device.ctx, segment.dev, data[segment.buf_off..].ptr, segment.len) catch {
                    s.node_failed[idx].store(true, .monotonic);
                    copied = false;
                    break;
                };
                if (!copied) continue;
                s.device.load_record_fn(s.device.ctx, slot) catch { s.node_failed[idx].store(true, .monotonic); continue; };
            }
        }
        fn saveThunk(ctx: *anyopaque, cut: engine.SnapCut) anyerror!void { const s: *Self = @ptrCast(@alignCast(ctx)); return s.save(cut); }
        fn loadThunk(ctx: *anyopaque, r: usize, ids: []const u32, leaf_key: u64) anyerror!usize { const s: *Self = @ptrCast(@alignCast(ctx)); return s.load(r, ids, leaf_key); }
        fn commitThunk(ctx: *anyopaque, n: usize) anyerror!void { const s: *Self = @ptrCast(@alignCast(ctx)); return s.commit(n); }
    };
}

test "layout chunkSegments covers every body byte exactly once and honours keep_lo" {
    var layout: Layout = .{};
    try layout.add(0x1000, 2);
    try layout.add(0x2000, 4);
    const lo: usize = 2;
    const hi: usize = 6;
    const body = layout.bodyLen(0, lo, hi);
    var segs: [max_segments]Segment = undefined;
    const used = try layout.chunkSegments(null, 0, lo, hi, 4, 0, body, &segs);
    try std.testing.expectEqual(@as(usize, 2), used);
    try std.testing.expectEqual(Segment{ .dev = 0x1000 + 8, .buf_off = 4, .len = 4 }, segs[0]);
    try std.testing.expectEqual(Segment{ .dev = 0x2000 + 16, .buf_off = 16, .len = 8 }, segs[1]);
    try std.testing.expectEqual(body - 12, segs[0].len + segs[1].len);
    const ring_view = RingView{ .ptr = 0x3000, .len = kvsave.chunk_bytes - 16, .from = 7 };
    const ring_used = try layout.chunkSegments(ring_view, kvsave.chunk_bytes, lo, hi, lo, 0, kvsave.chunk_bytes, &segs);
    try std.testing.expectEqual(@as(usize, 1), ring_used);
    try std.testing.expectEqual(@as(usize, kvsave.chunk_bytes - 16), segs[0].len);
}

test "host device copies host addressed planes" {
    const source: [8]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var target: [8]u8 = @splat(0);
    var host: HostDevice = .{};
    const device = host.device();
    try device.save_copy_fn(device.ctx, &target, @intFromPtr(&source), source.len);
    try std.testing.expectEqualSlices(u8, &source, &target);
    try std.testing.expectEqual(@as(usize, 1), host.copies);
}

test "save then load restores the saved plane and ring bytes" {
    const FakeEngine = struct {
        history: std.ArrayList(u32) = .empty,
        pos: usize = 0,
        frontier: usize = 0,
        restored_from: ?i64 = null,
        fn agreeWords(_: *@This(), _: []u32) !void {}
        fn drainDraftFill(_: *@This()) !void {}
        fn restoreToDevice(self: *@This(), at: usize, tokens: []const u32, ring_from: ?i64) !void {
            self.history.clearRetainingCapacity();
            try self.history.appendSlice(std.testing.allocator, tokens[0..at]);
            self.pos = at;
            self.frontier = at;
            self.restored_from = ring_from;
        }
        fn restoreFailed(self: *@This(), at: usize) !void {
            self.history.shrinkRetainingCapacity(at);
            self.pos = at;
            self.frontier = at;
        }
    };
    const FakeRing = struct {
        bytes: []u8,
        fn view(ctx: *anyopaque) anyerror!RingView {
            const ring: *@This() = @ptrCast(@alignCast(ctx));
            return .{ .ptr = @intFromPtr(ring.bytes.ptr), .len = ring.bytes.len, .from = 19 };
        }
    };
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/cache", .{tmp.sub_path});
    defer gpa.free(root);
    const identity = kvsave.identityOf(@splat(1), @splat(2), @splat(3), 0, 2, 512);
    var runtime = try prefix_cache.Runtime.init(gpa, root, identity, 0, .{ .expert_rows = 512 }, 1 << 30, 0, null);
    defer runtime.deinit();
    var engine_value: FakeEngine = .{};
    defer engine_value.history.deinit(gpa);
    var ids: [1024]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 5);
    try engine_value.history.appendSlice(gpa, &ids);
    engine_value.pos = ids.len;
    engine_value.frontier = ids.len;
    var plane: [512 * 16]u8 = undefined;
    for (&plane, 0..) |*byte, idx| byte.* = @truncate(idx * 3 + 11);
    var ring_bytes: [64]u8 = undefined;
    for (&ring_bytes, 0..) |*byte, idx| byte.* = @truncate(idx + 90);
    var fake_ring: FakeRing = .{ .bytes = &ring_bytes };
    const ring_hooks: RingHooks = .{ .ctx = &fake_ring, .view_fn = FakeRing.view };
    var layout: Layout = .{};
    try layout.add(@intFromPtr(&plane), 16);
    var slots: [10][]align(kvsave.alignment) u8 = undefined;
    for (&slots) |*slot| slot.* = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), kvsave.chunk_bytes);
    defer for (slots) |slot| gpa.free(slot);
    var host_device: HostDevice = .{};
    const snap_type = Snap(FakeEngine);
    var snap: snap_type = undefined;
    try snap.init(gpa, io, &runtime, &engine_value, host_device.device(), ring_hooks, layout, slots, ids.len, 4);
    snap.free_override = 100 << 30;
    defer snap.deinit();
    const key = kvsave.keyOf(&ids);
    try snap.save(.{ .at = ids.len, .kind = 0, .key = key, .parent_key = 0, .parent_at = 0, .seq = 1 });
    try snap.commit(1);
    const node = runtime.index.get(key) orelse return error.MissingSavedNode;
    const ring_span = kvsave.chunk_bytes;
    const body_len = ring_span + plane.len;
    try std.testing.expectEqual(kvsave.fileBytes(body_len, ids.len), node.bytes);
    const file_path = try runtime.path(key);
    defer gpa.free(file_path);
    var reader = try kvsave.Reader.open(gpa, file_path, identity, 0, 2, &ids, 0, ids.len, body_len);
    defer reader.close();
    const chunk = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), kvsave.chunk_bytes);
    defer gpa.free(chunk);
    const ring_part = try reader.readChunkAt(0, chunk);
    try std.testing.expectEqualSlices(u8, &ring_bytes, ring_part[0..ring_bytes.len]);
    for (ring_part[ring_bytes.len..]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
    const plane_part = try reader.readChunkAt(1, chunk);
    try std.testing.expectEqualSlices(u8, &plane, plane_part);
    const saved_plane = plane;
    const saved_ring = ring_bytes;
    @memset(&plane, 0xa5);
    @memset(&ring_bytes, 0x5a);
    engine_value.pos = 0;
    engine_value.frontier = 0;
    const restored = try snap.load(0, &ids, key);
    try std.testing.expectEqual(@as(usize, ids.len), restored);
    try std.testing.expectEqualSlices(u8, &saved_plane, &plane);
    try std.testing.expectEqualSlices(u8, &saved_ring, &ring_bytes);
    try std.testing.expectEqual(@as(usize, ids.len), engine_value.pos);
    try std.testing.expectEqualSlices(u32, &ids, engine_value.history.items);
    try std.testing.expectEqual(@as(?i64, 19), engine_value.restored_from);
}
