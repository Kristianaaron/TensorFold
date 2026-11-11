// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! tf-glm53-bringup: the WP9 GPU driver for Spark bring-up (review r2 N1). One rank per process, TP2 + DCP2.
//!   plan   <ckpt> <rank>                 header-only: per-class bytes (rounded vs logical), unused tensors,
//!                                        the admission table at --context; no GPU
//!   load   <ckpt> <rank>                 admit, load weights, re-check MemAvailable (N4), allocate State; print
//!                                        MemAvailable/MemFree after each stage and the drop against the plan;
//!                                        --checksum hashes every device tensor (run with TF_GLM53_ODIRECT=0 and =1)
//!   run    <ckpt> <rank> --ids F         prefill F's ids, then --decode N greedy steps; dumps the rope table
//!                                        and each step's rank-local fp32 logits rows into --out DIR
//!   bench  <ckpt> <rank> --ids F         per point L: cold prefill of L ids, --decode-steps greedy decodes, then an
//!                                        --append-row prefill on top; timings to stdout and --out/bench.jsonl
//!   resume <ckpt> <rank> --ids F         fresh prefill of F vs prefill(F[0..--split]) + decode --reply rows +
//!                                        nextTurn(rest): last-row logits and KV/index planes bit-equal
//!   replay <ckpt> <rank> --ids F         Engine.replayParity at --positions (default 2047,4095,32767), rows through verify max
//!   gate   <ckpt> <rank> --probe P (--ids F [--window-id W] | --cases C) --out O
//!                                        C has one "window_id ids_path" pair per line
//! Options: --layers N (bring-up subset 0..N-1, design §7 L1), --chunk-rows N (2048 or 4096, default 4096),
//! --context C (default 8192), --idfile PATH (NCCL
//! unique id: rank 0 writes it, rank 1 waits for it), --prefetch (side-stream stage fetch on its own communicator),
//! --fatbins DIR (glm53_*.fatbin, default zig-out/fatbin), --out DIR, --checksum.
//! The runbook wraps both rank commands in `timeout -s TERM -k 60 <limit>` so peer loss during startup is bounded.
//! Env: TF_GLM53_MEM_FLOOR_GIB, TF_GLM53_UMA_HEADROOM, TF_GLM53_ODIRECT, TF_GLM53_DECODE_UNION, TF_GLM53_EXPERT_ROWS_ADAPT,
//! TF_GLM53_EXPERT_ROWS_ADAPT_MAXPOS; NCCL_COMM_ID must be unset (both
//! communicators come from unique ids, review N7); NCCL_IB_HCA / NCCL_SOCKET_IFNAME as for the other TP tools.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const weights = @import("weights");
const state = @import("state");
const prefill_gemm = @import("prefill_gemm");
const forward = @import("forward");
const engine = @import("engine");
const uma = @import("uma");
const sampler = @import("sampler");

pub fn decodeUnionSetup(selected: bool, megakernel: bool) !bool {
    if (selected and megakernel) return error.DecodeUnionWithMegakernel;
    return selected;
}

pub fn roceRequestedFrom(tf_roce: ?[]const u8, collective: ?[]const u8) bool {
    return (if (tf_roce) |v| !std.mem.eql(u8, v, "0") else false) or (if (collective) |v| std.mem.eql(u8, v, "roce") else false);
}

fn roceRequested() bool {
    const tf = std.c.getenv("TF_ROCE");
    const coll = std.c.getenv("TF_GLM53_COLLECTIVE");
    return roceRequestedFrom(if (tf) |v| std.mem.span(v) else null, if (coll) |v| std.mem.span(v) else null);
}

/// B0: TF_GLM53_ROUTE_COMPACT, OWN_ATTEND, Q_ONE_GATHER and PARTIAL_ROCE, refused as native.zig openInner refuses
/// them (native.zig:944, :992-993, :1012). DECODE_UNION16 rows are read only when OWN_ATTEND or Q_ONE_GATHER is on,
/// so with all four off the bench keeps its old 8-row union.
pub const DecodeLevers = struct { route_compact: bool = false, own_attend: bool = false, q_one_gather: bool = false, partial_roce: bool = false, decode_union_rows: usize = forward.decode_mla_rows };

pub fn decodeLevers(route_compact: bool, own_attend: bool, q_one_gather: bool, partial_roce: bool, coll_fuse_requested: bool, decode_mla_mode: forward.DecodeMlaMode, decode_union: bool, union16: bool, verify_max: usize, roce_wanted: bool) !DecodeLevers {
    if (q_one_gather and decode_mla_mode != .v3) return error.QOneGatherNeedsDecodeMla3;
    if ((own_attend or q_one_gather) and !roce_wanted) return error.AttendGatherSwitchNeedsRoce;
    if (q_one_gather and coll_fuse_requested) return error.QOneGatherWithCollFuse;
    const union_rows = if (own_attend or q_one_gather) try forward.decodeUnionRows(decode_union, union16, verify_max) else forward.decode_mla_rows;
    if (q_one_gather and (!decode_union or union_rows < verify_max)) return error.QOneGatherNeedsDecodeUnionAllRows;
    return .{ .route_compact = route_compact, .own_attend = own_attend, .q_one_gather = q_one_gather, .partial_roce = partial_roce, .decode_union_rows = union_rows };
}

/// The bits native.zig agrees across ranks (native.zig:1181): bit 0 COLL_FUSE (never set by the bench), bit 1
/// OWN_ATTEND, bit 2 Q_ONE_GATHER, bit 3 PARTIAL_ROCE.
pub fn decodeSwitchBits(l: DecodeLevers) u8 {
    return (@as(u8, @intFromBool(l.own_attend)) << 1) | (@as(u8, @intFromBool(l.q_one_gather)) << 2) | (@as(u8, @intFromBool(l.partial_roce)) << 3);
}
const dump = @import("dump");
const draft_admit = @import("draft_admit");
const drafts = @import("drafts");
const glm = @import("glm");
const kvsave = @import("kvsave");
const prefix_snap = @import("prefix_snap");
const manifest_mod = weights.manifest;

fn parseExpertRowsAdapt(raw: ?[]const u8) !usize {
    const v = raw orelse return 0;
    if (std.mem.eql(u8, v, "0")) return 0;
    if (std.mem.eql(u8, v, "2048")) return 2048;
    if (std.mem.eql(u8, v, "4096")) return 4096;
    if (std.mem.eql(u8, v, "8192")) return 8192;
    return error.BadExpertRowsAdapt;
}

fn parseAdaptMaxPos(raw: ?[]const u8) !usize {
    const v = raw orelse return 0;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadExpertRowsAdaptMaxPos;
    if (n == 0) return error.BadExpertRowsAdaptMaxPos;
    return n;
}

fn validateExpertRowsAdapt(g: state.Geometry) !void {
    if (g.expert_rows_adapt == 0) return;
    if (g.expert_rows_adapt <= g.expert_rows) return error.ExpertRowsAdaptNotLarger;
    if (g.expert_rows_adapt > g.chunk_rows) return error.ExpertRowsAdaptExceedsChunk;
    if (!state.validExpertRows(.{ .expert_rows = g.expert_rows_adapt, .chunk_rows = g.chunk_rows, .piece_rows = g.piece_rows }) or !state.groupedExperts(g.expert_rows_adapt, g.piece_rows)) return error.ExpertRowsAdaptNotGrouped;
}

fn parseL2Depth(raw: ?[]const u8) !i32 {
    const depth = if (raw) |value| std.fmt.parseInt(i32, value, 10) catch return error.BadL2Depth else 4;
    if (depth < 0 or depth > 8) return error.BadL2Depth;
    return depth;
}

fn parseChunkRows(raw: ?[]const u8) !usize {
    const rows = if (raw) |value| std.fmt.parseInt(usize, value, 10) catch return error.Usage else state.default_chunk_rows;
    if (rows == 2048 or rows == 4096 or rows == 8192) return rows; // as native.parseChunkRows
    return error.Usage;
}

fn verifyMaxFromEnv(get: anytype) !usize {
    const raw = get("TF_GLM53_VERIFY_MAX") orelse return 8;
    const rows = std.fmt.parseInt(usize, raw, 10) catch return error.BadVerifyMax;
    if (rows != 8 and rows != 16) return error.BadVerifyMax;
    return rows;
}

fn setVerifyGeometryRows(geometry: *state.Geometry, verify_max: usize) void {
    geometry.decode_rows = verify_max;
    geometry.head_rows = verify_max;
}

const posix = std.posix;
var stop_requested: std.atomic.Value(bool) = .init(false);
var watchdog_done: std.atomic.Value(bool) = .init(false);

fn onSignal(_: posix.SIG) callconv(.c) void {
    stop_requested.store(true, .release);
}

fn installSignals() void {
    const act: posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.INT, &act, null);
    posix.sigaction(.TERM, &act, null);
}

fn checkStop() error{Interrupted}!void {
    if (stop_requested.load(.acquire)) return error.Interrupted;
}

fn signalRequested() bool {
    return stop_requested.load(.acquire);
}

fn commFailed(lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm) bool {
    var async_result: cuda.nccl.Result = 0;
    const api_result = lib.api.ncclCommGetAsyncError(comm, &async_result);
    return api_result != 0 or (async_result != 0 and async_result != cuda.nccl.in_progress);
}

fn stopWatchdog(g: *Gpu) void {
    while (!watchdog_done.load(.acquire)) {
        if (signalRequested()) {
            g.abortCommunicators();
        } else {
            var failed = false;
            if (g.group_live.load(.acquire)) failed = commFailed(&g.group.lib, g.group.comm);
            if (g.side_live.load(.acquire)) if (g.side) |side| {
                failed = commFailed(side.lib, side.comm) or failed;
            };
            if (failed) {
                std.debug.print("NCCL watchdog: communicator error or peer loss; aborting both communicators\n", .{});
                stop_requested.store(true, .release);
                g.abortCommunicators();
            }
        }
        std.Io.sleep(g.io, .fromMilliseconds(500), .awake) catch {};
    }
}

const preflight_available = 110 * 1024 * 1024 * 1024;
const preflight_free = 100 * 1024 * 1024 * 1024;

fn preflightOk(info: uma.MemInfo) bool {
    return info.available >= preflight_available and info.free >= preflight_free;
}

fn preflight(gpa: std.mem.Allocator, io: std.Io) !bool {
    const info = try uma.readMeminfo(gpa, io);
    std.debug.print("PREFLIGHT /proc/meminfo: MemAvailable {d:.3} GiB, MemFree {d:.3} GiB (required >= 110 GiB / >= 100 GiB)\n", .{ state.gib(info.available), state.gib(info.free) });
    if (!preflightOk(info)) {
        std.debug.print("PREFLIGHT refused: stop other GPU jobs; if MemFree is low because of page cache, drop the page cache and retry.\n", .{});
        return false;
    }
    return true;
}

const Opts = struct {
    cmd: []const u8,
    ckpt: []const u8,
    rank: usize,
    layers: ?usize = null,
    expert_rows: usize = 128,
    chunk_rows: usize = state.default_chunk_rows,
    index_mma_rows: usize = 4,
    index_qs: u8 = 0,
    qmmf_var: u8 = 0,
    gather_ovl: u8 = 0,
    moe_ep: bool = false,
    expert_rows_values: [6]usize = .{ 128, 512, 1024, 2048, 4096, 8192 },
    expert_rows_count: usize = 0,
    context: usize = 8192,
    context_set: bool = false,
    idfile: ?[]const u8 = null,
    id_wait_s: u64 = 600,
    ids: ?[]const u8 = null,
    side_ids: ?[]const u8 = null,
    cases: ?[]const u8 = null,
    points: []const u8 = "2048,8192,32768,131072,261000",
    decode_steps: usize = 64,
    rows: usize = 1,
    head_k: ?usize = null,
    append: usize = 850,
    warmup: usize = 2048,
    decode: usize = 8,
    prefetch: bool = false,
    out: ?[]const u8 = null,
    fatbins: []const u8 = "zig-out/fatbin",
    split: usize = 4100,
    reply: usize = 5,
    positions: []const u8 = "2047,4095,32767",
    probe: ?[]const u8 = null,
    window_id: i32 = 0,
    checksum: bool = false,
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 4,
    kv_dir: ?[]const u8 = null,
    cuts: ?[]const u8 = null,
    drafter_streams: usize = 1,
    steady: bool = false,
    manifest: ?[]const u8 = null,
    reuse: bool = false,
    fine_grain: usize = 0,
    fine_long: bool = false,
};

fn parse(args: []const [:0]const u8) !Opts {
    if (args.len < 2) return error.Usage;
    const diskprobe_mode = std.mem.eql(u8, args[1], "diskprobe");
    if (!diskprobe_mode and args.len < 4) return error.Usage;
    var o: Opts = if (diskprobe_mode) .{ .cmd = args[1], .ckpt = "", .rank = 0 } else .{ .cmd = args[1], .ckpt = args[2], .rank = try std.fmt.parseInt(usize, args[3], 10) };
    var chunk_rows_set = false;
    var i: usize = if (diskprobe_mode) 2 else 4;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const next = struct {
            fn f(list: []const [:0]const u8, at: *usize) ![]const u8 {
                at.* += 1;
                if (at.* >= list.len) return error.Usage;
                return list[at.*];
            }
        }.f;
        if (std.mem.eql(u8, a, "--reuse")) o.reuse = true else if (std.mem.eql(u8, a, "--fine-grain")) {
            o.fine_grain = try std.fmt.parseInt(usize, try next(args, &i), 10);
            if (o.fine_grain != 0 and o.fine_grain != 1 and o.fine_grain != 2 and o.fine_grain != 4 and o.fine_grain != 8 and o.fine_grain != 16 and o.fine_grain != 32 and o.fine_grain != 64 and o.fine_grain != 128 and o.fine_grain != 256) return error.Usage;
        } else if (std.mem.eql(u8, a, "--fine-long")) o.fine_long = true else if (std.mem.eql(u8, a, "--chunk-rows")) {
            o.chunk_rows = parseChunkRows(try next(args, &i)) catch return error.Usage;
            chunk_rows_set = true;
        } else if (std.mem.eql(u8, a, "--expert-rows")) {
            const value = try next(args, &i);
            var it = std.mem.tokenizeScalar(u8, value, ',');
            while (it.next()) |part| {
                if (o.expert_rows_count == o.expert_rows_values.len) return error.Usage;
                const er = try std.fmt.parseInt(usize, part, 10);
                if (!((er >= 16 and er <= 512 and er % 16 == 0) or er == 1024 or er == 1536 or er == 2048 or er == 4096 or er == 8192)) return error.Usage;
                o.expert_rows_values[o.expert_rows_count] = er;
                o.expert_rows_count += 1;
            }
            if (o.expert_rows_count == 0) return error.Usage;
            o.expert_rows = o.expert_rows_values[0];
        } else if (std.mem.eql(u8, a, "--kv-dir")) o.kv_dir = try next(args, &i) else if (std.mem.eql(u8, a, "--cuts")) o.cuts = try next(args, &i) else if (std.mem.eql(u8, a, "--layers")) o.layers = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--context")) {
            o.context = try std.fmt.parseInt(usize, try next(args, &i), 10);
            o.context_set = true;
        } else if (std.mem.eql(u8, a, "--manifest")) o.manifest = try next(args, &i) else if (std.mem.eql(u8, a, "--idfile")) o.idfile = try next(args, &i) else if (std.mem.eql(u8, a, "--id-wait-s")) o.id_wait_s = try std.fmt.parseInt(u64, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--decode")) o.decode = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--prefetch")) o.prefetch = true else if (std.mem.eql(u8, a, "--out")) o.out = try next(args, &i) else if (std.mem.eql(u8, a, "--fatbins")) o.fatbins = try next(args, &i) else if (std.mem.eql(u8, a, "--split")) o.split = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--reply")) o.reply = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--positions")) o.positions = try next(args, &i) else if (std.mem.eql(u8, a, "--ids")) o.ids = try next(args, &i) else if (std.mem.eql(u8, a, "--side-ids")) o.side_ids = try next(args, &i) else if (std.mem.eql(u8, a, "--cases")) o.cases = try next(args, &i) else if (std.mem.eql(u8, a, "--probe")) o.probe = try next(args, &i) else if (std.mem.eql(u8, a, "--window-id")) o.window_id = try std.fmt.parseInt(i32, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--checksum")) o.checksum = true else if (std.mem.eql(u8, a, "--steady")) o.steady = true else if (std.mem.eql(u8, a, "--drafter")) o.drafter = try next(args, &i) else if (std.mem.eql(u8, a, "--drafter-bits")) { o.drafter_bits = try std.fmt.parseInt(u8, try next(args, &i), 10); if (o.drafter_bits != 4 and o.drafter_bits != 8) return error.Usage; } else if (std.mem.eql(u8, a, "--drafter-streams")) { o.drafter_streams = try std.fmt.parseInt(usize, try next(args, &i), 10); if (o.drafter_streams == 0 or o.drafter_streams > 16) return error.Usage; } else if (std.mem.eql(u8, a, "--points")) o.points = try next(args, &i) else if (std.mem.eql(u8, a, "--decode-steps")) o.decode_steps = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--head-k")) o.head_k = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--append")) o.append = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--warmup")) o.warmup = try std.fmt.parseInt(usize, try next(args, &i), 10) else if (std.mem.eql(u8, a, "--rows")) {
            o.rows = try std.fmt.parseInt(usize, try next(args, &i), 10);
            if (o.rows != 1 and o.rows != 2 and o.rows != 4 and o.rows != 8 and o.rows != 16) return error.Usage;
        } else return error.Usage;
    }
    if (!chunk_rows_set) {
        const raw = std.c.getenv("TF_GLM53_CHUNK_ROWS");
        o.chunk_rows = try parseChunkRows(if (raw) |value| std.mem.span(value) else null);
    }
    if (o.fine_long and (!o.context_set or o.context < 23300)) return error.Usage;
    if (o.fine_long and o.fine_grain == 0) return error.Usage;
    const index_rows_raw = std.c.getenv("TF_GLM53_INDEX_ROWS");
    o.index_mma_rows = forward.index.parseScoreRows(if (index_rows_raw) |value| std.mem.span(value) else null) catch return error.Usage;
    const index_qs_raw = std.c.getenv("TF_GLM53_INDEX_QS");
    o.index_qs = forward.index.parseIndexQs(if (index_qs_raw) |value| std.mem.span(value) else null) catch return error.Usage;
    o.qmmf_var = @intFromBool(try forward.qmmfVarSwitch());
    const gather_ovl_raw = std.c.getenv("TF_GLM53_GATHER_OVL");
    o.gather_ovl = forward.parseGatherOvl(if (gather_ovl_raw) |value| std.mem.span(value) else null) catch return error.BadGatherOvl;
    const moe_ep_raw = std.c.getenv("TF_GLM53_MOE_EP");
    o.moe_ep = if (moe_ep_raw) |value| blk: {
        const setting = std.mem.span(value);
        if (std.mem.eql(u8, setting, "0")) break :blk false;
        if (std.mem.eql(u8, setting, "1")) break :blk true;
        return error.BadMoeEp;
    } else false;
    if (o.rank >= 2) return error.Usage;
    if (diskprobe_mode and o.kv_dir == null) return error.Usage;
    if ((std.mem.eql(u8, o.cmd, "reuse") or o.reuse) and !o.context_set) o.context = 126850;
    if (o.drafter != null) o.steady = true;
    if (std.mem.eql(u8, o.cmd, "load") and o.steady and o.ids == null) return error.Usage;
    if (std.mem.eql(u8, o.cmd, "saved") and (o.ids == null or o.kv_dir == null or o.drafter == null or o.drafter_bits != 4)) return error.Usage;
    if (std.mem.eql(u8, o.cmd, "cachegate")) {
        if (o.ids == null or o.kv_dir == null or o.drafter == null or o.drafter_bits != 4) return error.Usage;
        const alignment = try kvsave.cutAlignment(o.expert_rows);
        if (o.cuts) |raw_cuts| {
            const parsed_cuts = try parseCuts(std.heap.page_allocator, raw_cuts);
            defer std.heap.page_allocator.free(parsed_cuts);
            var previous: usize = 0;
            for (parsed_cuts) |cut| {
                if (cut % alignment != 0 or cut <= previous) return error.Usage;
                previous = cut;
            }
            if (parsed_cuts.len != 2) return error.Usage;
        }
    }
    if (std.mem.eql(u8, o.cmd, "gate") and (o.probe == null or o.out == null or ((o.ids == null) == (o.cases == null)))) return error.Usage;
    if (std.mem.eql(u8, o.cmd, "bench") and o.ids == null) return error.Usage;
    if (o.head_k) |k| if (!std.mem.eql(u8, o.cmd, "bench") or k == 0 or k > 2048) return error.Usage;
    if (o.rows != 1 and !std.mem.eql(u8, o.cmd, "bench")) return error.Usage;
    if (std.mem.eql(u8, o.cmd, "replay")) {
        const max_position = try positionsMaximum(o.positions);
        const needed = std.math.add(usize, max_position, 9) catch return error.Usage;
        if (!o.context_set) o.context = needed;
        if (o.context < needed) return error.Usage;
    }
    return o;
}

fn positionsMaximum(text: []const u8) !usize {
    var it = std.mem.tokenizeScalar(u8, text, ',');
    var max_position: ?usize = null;
    while (it.next()) |tok| {
        const position = try std.fmt.parseInt(usize, tok, 10);
        max_position = @max(max_position orelse position, position);
    }
    return max_position orelse error.BadPositions;
}

fn realtimeNs() !i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.REALTIME, &ts) != 0 or ts.sec < 0 or ts.nsec < 0) return error.ClockUnavailable;
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn monotonicNs() !i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0 or ts.sec < 0 or ts.nsec < 0) return error.ClockUnavailable;
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

const FsStat = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [24]u8,
};
extern "c" fn statvfs(path: [*:0]const u8, result: *FsStat) c_int;

fn diskFree(path: [:0]const u8) !u64 {
    var st: FsStat = undefined;
    if (statvfs(path.ptr, &st) != 0) return error.DiskStat;
    return std.math.mul(u64, st.frsize, st.bavail) catch return error.DiskStat;
}

const DiskReadJob = struct {
    gpa: std.mem.Allocator,
    fd: c_int,
    offset: u64,
    bytes: usize,
    chunk: usize,
    elapsed_ns: i128 = 0,
    digest: kvsave.Digest = @splat(0),
    failed: bool = false,
};

fn diskReadWorker(job: *DiskReadJob) void {
    const buf = job.gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), job.chunk) catch {
        job.failed = true;
        return;
    };
    defer job.gpa.free(buf);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    const t0 = monotonicNs() catch {
        job.failed = true;
        return;
    };
    var done: usize = 0;
    while (done < job.bytes) {
        const n = @min(job.chunk, job.bytes - done);
        const got = std.c.pread(job.fd, buf.ptr, n, @intCast(job.offset + done));
        if (got != @as(isize, @intCast(n))) {
            job.failed = true;
            return;
        }
        for (buf[0..n]) |byte| if (byte != 0) {
            job.failed = true;
            return;
        };
        hash.update(buf[0..n]);
        done += n;
    }
    job.elapsed_ns = (monotonicNs() catch 0) - t0;
    hash.final(&job.digest);
}

fn diskprobe(gpa: std.mem.Allocator, dir: []const u8) !bool {
    const dpath = try std.fmt.allocPrintSentinel(gpa, "{s}", .{dir}, 0);
    defer gpa.free(dpath);
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/.tfkv-diskprobe.tmp", .{dir}, 0);
    defer gpa.free(path);
    const free_before = try diskFree(dpath);
    std.debug.print("DISKPROBE free before {d:.2} GiB\n", .{state.gib(free_before)});
    if (free_before < (4 << 30) + 50_000_000_000) return error.DiskSpaceBelowFloor;
    var flags_in: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
    if (comptime @hasField(std.c.O, "DIRECT")) flags_in.DIRECT = true else return error.ODirectUnavailable;
    const wfd = std.c.open(path.ptr, flags_in, @as(std.c.mode_t, 0o600));
    if (wfd < 0) return error.ODirectOpen;
    defer _ = std.c.close(wfd);
    defer _ = std.c.unlink(path.ptr);
    const file_bytes: usize = 4 << 30;
    const wbuf = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), 4 << 20);
    defer gpa.free(wbuf);
    @memset(wbuf, 0);
    const write_start = try monotonicNs();
    var at: usize = 0;
    while (at < file_bytes) : (at += wbuf.len) {
        if (std.c.pwrite(wfd, wbuf.ptr, wbuf.len, @intCast(at)) != @as(isize, @intCast(wbuf.len))) return error.DiskWrite;
    }
    if (std.c.fsync(wfd) != 0) return error.DiskSync;
    const write_seconds = @as(f64, @floatFromInt(try monotonicNs() - write_start)) / std.time.ns_per_s;
    std.debug.print("DISKPROBE write 4 GiB: {d:.2} GB/s\n", .{4.294967296 / write_seconds});

    var rflags: std.c.O = .{ .ACCMODE = .RDONLY };
    if (comptime @hasField(std.c.O, "DIRECT")) rflags.DIRECT = true else return error.ODirectUnavailable;
    const rfd = std.c.open(path.ptr, rflags);
    if (rfd < 0) return error.ODirectOpen;
    defer _ = std.c.close(rfd);
    for ([_]usize{ 1 << 20, 4 << 20, 16 << 20 }) |chunk| {
        for ([_]usize{ 1, 2, 4, 8 }) |thread_count| {
            var jobs: [8]DiskReadJob = undefined;
            var threads: [7]std.Thread = undefined;
            const per = (file_bytes / thread_count / chunk) * chunk;
            for (0..thread_count) |i| {
                const bytes = if (i + 1 == thread_count) file_bytes - per * i else per;
                jobs[i] = .{ .gpa = gpa, .fd = rfd, .offset = per * i, .bytes = bytes, .chunk = chunk };
                if (i + 1 < thread_count) threads[i] = try std.Thread.spawn(.{}, diskReadWorker, .{&jobs[i]});
            }
            diskReadWorker(&jobs[thread_count - 1]);
            for (0..thread_count - 1) |i| threads[i].join();
            var elapsed: i128 = 0;
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            for (jobs[0..thread_count]) |job| {
                if (job.failed) return error.DiskRead;
                elapsed = @max(elapsed, job.elapsed_ns);
                hash.update(&job.digest);
            }
            var digest: kvsave.Digest = undefined;
            hash.final(&digest);
            const seconds = @as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s;
            std.debug.print("DISKPROBE read chunk={d} MiB threads={d}: {d:.2} GB/s sha256={x}\n", .{ chunk >> 20, thread_count, 4.294967296 / seconds, digest[0..8].* });
        }
    }
    _ = kvsave.dropPageCache(path);
    _ = std.c.unlink(path.ptr);
    std.debug.print("DISKPROBE free after {d:.2} GiB\n", .{state.gib(try diskFree(dpath))});
    return true;
}

fn idFileFresh(mtime_ns: i128, process_start_ns: i128) bool {
    return process_start_ns > 0 and mtime_ns > process_start_ns;
}

fn readyNonceMatches(ready: []const u8, nonce: []const u8) bool {
    return std.mem.eql(u8, ready, nonce);
}

/// Fixed on-disk NCCL handoff layout, serialized explicitly to avoid ABI padding:
/// magic[8], version u16 LE, nonce length u16 LE, nonce[16], NCCL UniqueId[128].
const IdFile = struct {
    const magic = "TFGLM53\x00";
    const version: u16 = 1;
    const nonce_len: u16 = 16;
    const nonce_bytes: usize = 16;
    const magic_start = 0;
    const version_start = magic_start + magic.len;
    const nonce_len_start = version_start + 2;
    const nonce_start = nonce_len_start + 2;
    const id_start = nonce_start + nonce_bytes;
    const size = id_start + @sizeOf(cuda.nccl.UniqueId);

    fn encode(id: cuda.nccl.UniqueId, nonce: [16]u8) [size]u8 {
        var bytes: [size]u8 = undefined;
        @memcpy(bytes[magic_start..version_start], magic);
        std.mem.writeInt(u16, bytes[version_start..][0..2], version, .little);
        std.mem.writeInt(u16, bytes[nonce_len_start..][0..2], nonce_len, .little);
        @memcpy(bytes[nonce_start..id_start], &nonce);
        @memcpy(bytes[id_start..size], &id.internal);
        return bytes;
    }

    fn decode(bytes: []const u8) !struct { cuda.nccl.UniqueId, [16]u8 } {
        if (bytes.len != size or !std.mem.eql(u8, bytes[magic_start..version_start], magic)) return error.BadIdFile;
        if (std.mem.readInt(u16, bytes[version_start..][0..2], .little) != version or
            std.mem.readInt(u16, bytes[nonce_len_start..][0..2], .little) != nonce_len) return error.BadIdFile;
        var id: cuda.nccl.UniqueId = undefined;
        var nonce: [16]u8 = undefined;
        @memcpy(&nonce, bytes[nonce_start..id_start]);
        @memcpy(&id.internal, bytes[id_start..size]);
        return .{ id, nonce };
    }
};

fn gateSidecar(allocator: std.mem.Allocator) ![]u8 {
    return std.json.Stringify.valueAlloc(allocator, .{ .fp32_moe_sum = true }, .{});
}

/// Ids from a JSON array or whitespace/comma-separated integers.
fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, "[], \t\r\n");
    while (it.next()) |tok| try out.append(gpa, try std.fmt.parseInt(u32, tok, 10));
    return out.toOwnedSlice(gpa);
}

fn mem(gpa: std.mem.Allocator, io: std.Io, label: []const u8) ?uma.MemInfo {
    const info = uma.readMeminfo(gpa, io) catch {
        std.debug.print("MEM {s}: /proc/meminfo unreadable\n", .{label});
        return null;
    };
    std.debug.print("MEM {s}: MemAvailable {d:.3} GiB, MemFree {d:.3} GiB, MemTotal {d:.3} GiB\n", .{ label, state.gib(info.available), state.gib(info.free), state.gib(info.total) });
    return info;
}

const Setup = struct {
    config: weights.Config,
    store: [128]usize = undefined,
    select: weights.Select = .{},
    manifest: ?*manifest_mod.Manifest = null,
    geometry: state.Geometry = .{},
    plan: weights.Plan,
    /// TF_GLM53_SMLA_P, read once here and passed into every Forward (prefill attend kernel).
    smla_p: bool = false,
    /// TF_GLM53_SMLA_FP8, read once here and passed into every Forward; setup refuses it without TF_GLM53_SMLA_P=1.
    smla_fp8: bool = false,
    /// TF_GLM53_ATTN_PIPE, captured once here for communicator setup and every Forward.
    attn_pipe: bool = false,
    /// TF_GLM53_DECODE_UNION=1 (A3), read once here: decode windows of 1..8 rows attend with tf_glm53_attend_verify.
    decode_union: bool = false,
    /// TF_GLM53_VERIFY_MAX (8 or 16), read once here: decode/head rows of the workspace (16 for copy rounds).
    verify_max: usize = 8,
    /// B0: decode union rows as native.zig computes them (DecodeLevers); 8 unless OWN_ATTEND or Q_ONE_GATHER is on.
    decode_union_rows: usize = forward.decode_mla_rows,
    decode_mla16: bool = false,
    /// Profiling and TBO compatibility, resolved once during setup.
    profile_prefill: bool = false,
    profile_decode_sub: bool = false,
    l2pf_options: forward.l2pf_mod.Options = .{},

    fn init(gpa: std.mem.Allocator, io: std.Io, o: Opts) !Setup {
        const moe_ep = o.moe_ep;
        if (std.c.getenv("TF_GLM53_DRAFT_GRAPH")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "1")) std.log.warn("TF_GLM53_DRAFT_GRAPH=1 is inactive in tf-glm53-bringup: it runs no draft rounds", .{});
        }
        if (std.c.getenv("TF_GLM53_ONE_FRAME")) |value| {
            if (std.mem.eql(u8, std.mem.span(value), "1")) std.log.warn("TF_GLM53_ONE_FRAME=1 is inactive in tf-glm53-bringup: it has no follower frames", .{});
        }
        if (std.c.getenv("TF_GLM53_LAZY_DRAW")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "1")) std.log.warn("TF_GLM53_LAZY_DRAW=1 is inactive in tf-glm53-bringup: its windows keep every row", .{});
        }
        const decode_mla_mode = forward.decodeMlaModeSwitch();
        const decode_mla16 = forward.decodeMla16Switch();
        const cand_fast = if (std.c.getenv("TF_GLM53_CAND_FAST")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
        if (cand_fast and forward.megakernelSwitch()) return error.CandFastNeedsNoMk;
        const add_norm_fuse = forward.addNormFuseSwitch();
        if (add_norm_fuse and forward.megakernelSwitch()) return error.AddNormFuseNeedsNoMk;
        if (add_norm_fuse) std.log.info("TF_GLM53_ADD_NORM_FUSE=1: decode layers before the last end in tf_glm53_moe_residual_rmsnorm_6144 (MoE) or tf_glm53_residual_rmsnorm_6144 (dense), which also write the next layer's input norm; that layer launches no tf_glm_rmsnorm_6144 (inactive with an ablation file)", .{});
        const qpe_absorb_fuse = forward.qpeAbsorbFuseSwitch();
        if (qpe_absorb_fuse and forward.megakernelSwitch()) return error.QpeAbsorbFuseNeedsNoMk;
        if (qpe_absorb_fuse and decode_mla_mode != .v3) return error.QpeAbsorbFuseNeedsDecodeMla3;
        if (qpe_absorb_fuse) std.log.info("TF_GLM53_QPE_ABSORB_FUSE=1: decode MLA v3 (rows 1..8) launches tf_glm53_absorb_qpe_decode_v3 in place of tf_glm53_qpe_pack + tf_glm53_absorb_decode_v3", .{});
        const cand_v2 = if (std.c.getenv("TF_GLM53_CAND_V2")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
        if (cand_v2 and forward.megakernelSwitch()) return error.CandV2NeedsNoMk;
        var l2pf_problem: []const u8 = "";
        const l2pf_options = forward.l2pf_mod.Options.read(struct { fn get(name: [:0]const u8) ?[]const u8 {
            const value = std.c.getenv(name) orelse return null;
            return std.mem.span(value);
        } }.get, &l2pf_problem) catch |e| {
            std.log.err("{s}", .{l2pf_problem});
            return e;
        };
        if (l2pf_options.on and forward.megakernelSwitch()) return error.L2pfWithMegakernel;
        if (forward.megakernelSwitch() and decode_mla_mode != .off) {
            std.log.err("TF_GLM53_MK=1 and TF_GLM53_DECODE_MLA are incompatible", .{});
            return error.IncompatibleGlm53Modes;
        }
        const c = try weights.readConfig(gpa, io, o.ckpt);
        var s: Setup = .{ .config = c, .plan = undefined };
        s.verify_max = try verifyMaxFromEnv(struct { fn get(name: []const u8) ?[]const u8 {
            if (!std.mem.eql(u8, name, "TF_GLM53_VERIFY_MAX")) return null;
            const value = std.c.getenv("TF_GLM53_VERIFY_MAX");
            return if (value) |v| std.mem.span(v) else null;
        } }.get);
        s.decode_mla16 = decode_mla16;
        try forward.validateDecodeMla16(decode_mla16, decode_mla_mode, s.verify_max);
        if (o.layers) |n| {
            if (n == 0 or n > c.layers or n > s.store.len) return error.Usage;
            s.select = weights.Select.bringUp(&s.store, n);
        }
        if (o.manifest) |name| {
            const m = try gpa.create(manifest_mod.Manifest);
            errdefer gpa.destroy(m);
            m.* = try manifest_mod.Manifest.load(gpa, io, o.ckpt, name);
            s.manifest = m;
            s.select.manifest = m;
        }
        s.select.moe_ep = moe_ep;
        s.select.rank = o.rank;
        const n: usize = o.layers orelse c.layers;
        var full: usize = 0;
        for (0..n) |l| full += @intFromBool(c.full.isSet(l));

        const prompt_mla = forward.promptMlaSwitch();
        const smla_p = forward.smlaPSwitch();
        const smla_fp8 = forward.smlaFp8Switch();
        const smla_p3 = forward.smlaP3Switch();
        const moe_pipe = forward.moePipeSwitch();
        const moe_ring = try forward.moeRingSwitch();
        if (!moe_pipe and moe_ring != 2) return error.MoeRingWithoutMoePipe;
        if (moe_ring != 2) std.log.info("TF_GLM53_MOE_RING={d}: MoE gather slots", .{moe_ring});
        const piece_pipe = forward.piecePipeSwitch();
        const index_merge = forward.indexMergeSwitch();
        const index_pipe = forward.indexPipeSwitch();
        const attn_pipe = forward.attnPipeSwitch();
        s.attn_pipe = attn_pipe;
        s.decode_union = try decodeUnionSetup(forward.decodeUnionSwitch(), forward.megakernelSwitch());
        if (s.decode_union) std.log.info("TF_GLM53_DECODE_UNION=1: decode attend (rows 1..8) tf_glm53_attend_verify + tf_glm53_attend_split_merge (byte-identical to attend_mma)", .{});
        const route_compact_env = if (std.c.getenv("TF_GLM53_ROUTE_COMPACT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const coll_fuse_env = if (std.c.getenv("TF_GLM53_COLL_FUSE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const levers = try decodeLevers(route_compact_env, forward.ownAttendSwitch(), forward.qOneGatherSwitch(), forward.partialRoceSwitch(), coll_fuse_env, decode_mla_mode, s.decode_union, forward.decodeUnion16Switch(), s.verify_max, cuda.roce.wanted());
        if (levers.route_compact) std.log.info("TF_GLM53_ROUTE_COMPACT=1: side decode routing launches tf_glm_router_tile or tf_glm_router_part, tf_glm_slice_sum, and tf_glm_topk_compact", .{});
        if (levers.own_attend) std.log.info("TF_GLM53_OWN_ATTEND=1: decode attend tf_glm53_attend_verify + tf_glm53_attend_split_merge on side and main streams", .{});
        if (levers.q_one_gather) std.log.info("TF_GLM53_Q_ONE_GATHER=1: decode all-gather packed QA/QPE once, then tf_glm53_attend_verify + tf_glm53_attend_split_merge per head group", .{});
        if (levers.partial_roce) std.log.info("TF_GLM53_PARTIAL_ROCE=1: attention partials use RoCE oneShotGather before tf_glm53_mla_merge; absorbed queries remain NCCL", .{});
        if (index_pipe and !piece_pipe and !moe_pipe) return error.IndexPipeRequiresCommStream;
        const tbo_enabled = forward.tboSwitch();
        const ws_trim = forward.wsTrimSwitch();
        const moe_post = forward.moePostSwitch();
        const comm_prio = forward.commPrioSwitch();
        if (comm_prio and !(o.prefetch or piece_pipe or moe_pipe or tbo_enabled or index_pipe)) return error.CommPrioWithoutCommStream;
        if (attn_pipe and !piece_pipe and !moe_pipe and !index_pipe) return error.AttnPipeWithoutCommStream;
        if (attn_pipe and tbo_enabled) return error.AttnPipeWithTbo;
        std.log.info("TF_GLM53_ATTN_PIPE={d}; comm-stream gathers on their own NCCL communicator", .{@intFromBool(attn_pipe)});
        s.profile_prefill = forward.profilePrefillSwitch();
        s.profile_decode_sub = forward.profileDecodeSubSwitch();
        try forward.validateProfilePrefillTbo(s.profile_prefill, tbo_enabled);
        if (moe_pipe and tbo_enabled) return error.MoePipeWithTbo;
        if (moe_ep and (if (std.c.getenv("TF_GLM53_ROW_PRUNE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false)) return error.MoeEpWithRowPrune;
        if (moe_pipe and (if (std.c.getenv("TF_GLM53_ROW_PRUNE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false)) return error.MoePipeWithRowPrune;
        if (moe_pipe and !state.groupedExperts(o.expert_rows, 512)) {
            std.log.err("TF_GLM53_MOE_PIPE=1 requires grouped prefill experts (expert_rows must exceed piece_rows=512)", .{});
            return error.MoePipeRequiresGroupedExperts;
        }
        std.log.info("TF_GLM53_MOE_PIPE={d}; grouped prefill MoE only (incompatible with TBO and ROW_PRUNE)", .{@intFromBool(moe_pipe)});
        if (smla_fp8 and !smla_p) return error.SmlaFp8WithoutSmlaP;
        if (smla_p3 and !smla_p) return error.SmlaP3WithoutSmlaP;
        if (smla_p3 and smla_fp8) return error.SmlaP3WithSmlaFp8;
        const prompt_experts = forward.promptExpertsMode();
        if (moe_ep and !moe_post) return error.MoeEpRequiresMoePost;
        if (moe_ep and prompt_experts == .off) return error.MoeEpRequiresPromptExperts;
        if (moe_ep and !state.groupedExperts(o.expert_rows, 512)) return error.MoeEpRequiresGroupedExperts;
        if (moe_ep) std.log.info("TF_GLM53_MOE_EP=1: two-rank expert-parallel routed experts (grouped prefill; decode on the generic routed path)", .{});
        const pe_flags = forward.promptExpertsFlags(prompt_experts);
        const adapt_rows = try parseExpertRowsAdapt(if (std.c.getenv("TF_GLM53_EXPERT_ROWS_ADAPT")) |v| std.mem.span(v) else null);
        const adapt_maxpos = try parseAdaptMaxPos(if (std.c.getenv("TF_GLM53_EXPERT_ROWS_ADAPT_MAXPOS")) |v| std.mem.span(v) else null);
        const scratch_overlay = if (std.c.getenv("TF_GLM53_SCRATCH_OVERLAY")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (ws_trim and scratch_overlay) return error.WsTrimWithScratchOverlay;
        if (ws_trim and prompt_experts == .off) return error.WsTrimNeedsPromptExperts;
        if (moe_post and (!moe_pipe or prompt_experts == .off or tbo_enabled)) return error.MoePostNeedsPromptMoePipe;
        if (o.expert_rows > o.chunk_rows) {
            std.log.err("expert_rows {d} exceeds chunk_rows {d}; set TF_GLM53_CHUNK_ROWS >= expert_rows on both ranks", .{ o.expert_rows, o.chunk_rows });
            return error.ExpertRowsExceedChunkRows;
        }
        s.geometry = .{ .layers = n, .full_index_layers = full, .vocab_local = c.vocab / 2, .candidate_k = o.head_k orelse 64, .cand_fast = cand_fast, .prefetch = o.prefetch, .tbo = tbo_enabled, .piece_pipe = piece_pipe, .moe_pipe = moe_pipe, .moe_ring = moe_ring, .ws_trim = ws_trim, .moe_post = moe_post, .moe_ep = moe_ep, .comm_prio = comm_prio, .index_merge = index_merge, .index_pipe = index_pipe, .prompt_mla = prompt_mla, .smla_p = smla_p, .smla_fp8 = smla_fp8, .smla_p3 = smla_p3, .shard_qkv_a = forward.shardQkvASwitch(), .prefill_bf16 = try state.prefillGemmFromEnv() == .bf16, .scratch_overlay = scratch_overlay, .decode_side = forward.decodeSideSwitch(), .decode_mla = decode_mla_mode != .off, .decode_mla_v2 = decode_mla_mode == .v2 or decode_mla_mode == .v3, .decode_mla_v3 = decode_mla_mode == .v3, .prompt_experts = pe_flags.on, .prompt_experts_wide = pe_flags.wide, .prompt_experts_pmx = pe_flags.pmx, .expert_rows = o.expert_rows, .chunk_rows = o.chunk_rows, .index_mma_rows = o.index_mma_rows, .index_qs = o.index_qs, .qmmf_var = o.qmmf_var, .gather_ovl = o.gather_ovl, .expert_rows_adapt = adapt_rows, .expert_rows_adapt_maxpos = adapt_maxpos, .prompt_experts_p5a = pe_flags.p5a, .l2pf = l2pf_options.on };
        s.l2pf_options = l2pf_options;
        s.geometry.qa_shard = forward.qaShardSwitch();
        if (s.geometry.qa_shard and s.geometry.shard_qkv_a) return error.QaShardWithShardQkvA;
        if (s.geometry.qa_shard and forward.megakernelSwitch()) return error.QaShardWithMegakernel;
        if (s.geometry.qa_shard and tbo_enabled) return error.QaShardWithTbo;
        if (s.geometry.qa_shard and s.geometry.prefill_bf16) return error.QaShardWithPrefillBf16;
        std.log.info("TF_GLM53_CAND_FAST={d}: launches tf_glm53_cand_fill_f32, {s}, then tf_glm53_cand_merge_f32", .{ @intFromBool(cand_fast), if (cand_fast) "tf_glm53_cand_part_fast_f32" else "tf_glm53_cand_part_f32" });
        s.geometry.route_compact = levers.route_compact;
        s.geometry.own_attend = levers.own_attend;
        s.geometry.q_one_gather = levers.q_one_gather;
        s.geometry.partial_roce = levers.partial_roce;
        s.decode_union_rows = levers.decode_union_rows;
        if (forward.draftIngestSideSwitch()) std.log.warn("TF_GLM53_DRAFT_INGEST_SIDE=1 is inactive in tf-glm53-bringup: it runs no draft rounds; gates keep the draft-time ingest", .{});
        s.geometry.add_norm_fuse = add_norm_fuse;
        s.geometry.qpe_absorb_fuse = qpe_absorb_fuse;
        s.geometry.cand_v2 = cand_v2;
        std.log.info("TF_GLM53_CAND_V2={d}: {s}", .{ @intFromBool(s.geometry.cand_v2), if (s.geometry.cand_v2) "launches tf_glm53_cand2_fill_f32, tf_glm53_cand2_hist_f32, then tf_glm53_cand2_select_f32 where the scratch holds 16 KiB a row (k >= 64); smaller tiles keep the CAND_FAST kernels" else "off" });
        if (drafts.draftDown4Switch()) std.log.info("TF_GLM53_DRAFT_DOWN4=1: drafter admission and gate loads use the 4-bit down (no draft rounds run here)", .{});
        s.geometry.park_rows = try state.sideRowsFromEnv(512, o.context);
        if (s.geometry.park_rows != 0) {
            const resident = std.c.getenv("TF_GLM53_RESIDENT") orelse return error.SideRowsNeedResident;
            if (!std.mem.eql(u8, std.mem.span(resident), "1")) return error.SideRowsNeedResident;
        }
        if (prompt_experts != .off) std.log.info("rank={d} TF_GLM53_PROMPT_EXPERTS dispatch: mode={s} wide={} pmx={} p5a={}", .{ o.rank, @tagName(prompt_experts), s.geometry.prompt_experts_wide, s.geometry.prompt_experts_pmx, s.geometry.prompt_experts_p5a });
        try validateExpertRowsAdapt(s.geometry);
        if (adapt_rows != 0 and (prompt_experts == .off or tbo_enabled)) return error.ExpertRowsAdaptNeedsPromptExperts;
        setVerifyGeometryRows(&s.geometry, s.verify_max);
        try state.validateMoePipeGeometry(s.geometry);
        std.log.info("TF_GLM53_PIECE_PIPE={d}; grouped prefill only (non-grouped prefill and ROW_PRUNE use the old schedule)", .{@intFromBool(s.geometry.piece_pipe)});
        const prefill_w8a8 = forward.prefillW8a8Switch();
        s.geometry.prefill_w8a8 = prefill_w8a8;
        s.geometry.prefill_w8a8_all = forward.prefillW8a8AllSwitch();
        if (s.geometry.qmmf_var != 0) std.log.info("TF_GLM53_QMMF_VAR=1: selected prefill FP8B shapes use same-bit QMMF schedule variants", .{});
        const pre_fuse = forward.preFuseSwitch();
        const prep_fuse = forward.prepFuseSwitch();
        if ((pre_fuse != .off or prep_fuse != .off) and s.geometry.shard_qkv_a) return error.PreFuseWithShardQkvA;
        if (s.geometry.qa_shard and prep_fuse == .split) return error.QaShardWithPrepFuseSplit;
        if (s.geometry.qa_shard) std.log.info("TF_GLM53_QA_SHARD=1: q_a rows {d}..{d} on this rank; per layer one shard_qkv_a all-gather of rows x 2048 B, then tf_glm53_rmsnorm_2048_halves (bit-identical to tf_glm_rmsnorm_2048); kv_a stays whole", .{ o.rank * 1024, o.rank * 1024 + 1024 });
        if (pre_fuse != .off and forward.megakernelSwitch()) return error.PreFuseWithMegakernel;
        std.log.info("TF_GLM53_PRE_FUSE={s} TF_GLM53_PREP_FUSE={s}: grouped q_a | kv_a | indexer wk projections (bit-identical; 0 = off)", .{ switch (pre_fuse) { .off => "0", .on => "1", .prefetch => "2" }, switch (prep_fuse) { .off => "0", .on => "1", .split => "2" } });
        s.geometry.pre_fuse = pre_fuse;
        s.geometry.prep_fuse = prep_fuse;
        var geometry_without_overlay = s.geometry;
        geometry_without_overlay.scratch_overlay = false;
        std.log.info("TF_GLM53_SCRATCH_OVERLAY={s}: workspace total {d} bytes (without overlay {d} bytes)", .{ if (scratch_overlay) "on" else "off", state.workspaceLayout(s.geometry, o.context).bytes, state.workspaceLayout(geometry_without_overlay, o.context).bytes });
        s.select.shard_qkv_a = s.geometry.shard_qkv_a; // load only this rank's q_a / kv_a rows, as native.zig does
        s.select.shard_q_a = s.geometry.qa_shard;
        s.plan = try weights.planRank(gpa, io, o.ckpt, c, o.rank, 2, s.select);
        s.smla_p = s.geometry.smla_p;
        s.smla_fp8 = s.geometry.smla_fp8;
        if (s.smla_p) if (std.c.getenv("TF_GLM53_ATTEND")) |value| {
            if (std.mem.eql(u8, std.mem.span(value), "scalar")) return error.SmlaPWithScalarAttend;
        };
        return s;
    }

    fn deinit(s: *Setup, gpa: std.mem.Allocator) void {
        s.plan.deinit(gpa);
        if (s.manifest) |m| { m.deinit(); gpa.destroy(m); }
    }

    fn report(s: *const Setup) void {
        const w = s.plan.weights;
        const l = s.plan.logical;
        std.debug.print("PLAN weights per class (GiB, rounded arena | logical):\n", .{});
        inline for (.{ "routed", "backbone_fp8", "embed_head", "router_norms", "lowrank" }) |f| std.debug.print("  {s:<14} {d:>9.4} | {d:>9.4}\n", .{ f, state.gib(@field(w, f)), state.gib(@field(l, f)) });
        std.debug.print("  {s:<14} {d:>9.4} | {d:>9.4}\nPLAN unused tensors: {d}\n", .{ "total", state.gib(w.total()), state.gib(l.total()), s.plan.unused.len });
        const absorb_kernel = if (!s.geometry.decode_mla) "tf_glm53_absorb" else if (s.geometry.decode_mla_v3) "tf_glm53_absorb_decode_v3" else "tf_glm53_absorb_decode";
        const expand_kernel = if (!s.geometry.decode_mla) "tf_glm53_expand_v" else if (s.geometry.decode_mla_v2) "tf_glm53_expand_v_decode_v2" else "tf_glm53_expand_v_decode";
        std.debug.print("PLAN decode MLA kernels: absorb={s} expand_v={s}\n", .{ absorb_kernel, expand_kernel });
        for (s.plan.unused[0..@min(s.plan.unused.len, 64)]) |name| std.debug.print("  unused {s}\n", .{name});
        const attend = if (s.geometry.smla_p3) "tf_glm53_smla_union + tf_glm53_smla_p3 (TF_GLM53_SMLA_P=1 TF_GLM53_SMLA_P3=1, bit-identical to SMLA-P)" else if (s.smla_fp8) "tf_glm53_smla_p_fp8 (TF_GLM53_SMLA_P=1 TF_GLM53_SMLA_FP8=1, e4m3 scores)" else if (s.smla_p) "tf_glm53_smla_p (TF_GLM53_SMLA_P=1)" else if (std.c.getenv("TF_GLM53_ATTEND")) |value|
            if (std.mem.eql(u8, std.mem.span(value), "scalar")) "tf_glm53_attend (scalar)" else "tf_glm53_attend_mma"
        else "tf_glm53_attend_mma";
        std.debug.print("PLAN prefill attend: {s}\n", .{attend});
        if (s.geometry.prefill_w8a8_all) {
            std.debug.print("PLAN prefill W8A8: {s}\n", .{"tf_glm53_w8a8_quant + tf_glm53_w8a8_gemm (TF_GLM53_PREFILL_W8A8=2, lossy, KLD-gated)"});
        } else {
            const o_proj = if (s.geometry.prefill_w8a8) "tf_glm53_w8a8_quant + tf_glm53_w8a8_gemm (TF_GLM53_PREFILL_W8A8=1, lossy, KLD-gated)" else "qmmf (default)";
            std.debug.print("PLAN prefill o_proj: {s}\n", .{o_proj});
        }
        std.debug.print("PLAN decode attend: {s}\n", .{if (s.decode_union) "tf_glm53_attend_verify + tf_glm53_attend_split_merge (TF_GLM53_DECODE_UNION=1)" else "tf_glm53_attend_mma"});
    }
};

fn admit(gpa: std.mem.Allocator, io: std.Io, s: *const Setup, o: Opts) !state.Admission {
    const info = mem(gpa, io, "before admission") orelse return error.NoMemInfo;
    const reserve = try state.reserveFromEnv(info.total);
    const adm = state.admitRank(s.geometry, o.context, s.plan.weights, info.available, reserve) catch |e| {
        state.breakdown(s.geometry, o.context, s.plan.weights).report(o.context);
        return e;
    };
    adm.report();
    return adm;
}

fn wyhash(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer) !u64 {
    var h = std.hash.Wyhash.init(0);
    const block = 64 << 20;
    const host = try gpa.alloc(u8, @min(block, buf.len));
    defer gpa.free(host);
    var at: usize = 0;
    while (at < buf.len) : (at += block) {
        const n = @min(block, buf.len - at);
        try buf.download(at, host[0..n]);
        h.update(host[0..n]);
    }
    return h.final();
}

fn writeAtomic(io: std.Io, gpa: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const temp = try std.fmt.allocPrint(gpa, "{s}.tmp.{d}", .{ path, std.c.getpid() });
    defer gpa.free(temp);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = temp, .data = bytes });
    try std.Io.Dir.cwd().rename(temp, std.Io.Dir.cwd(), path, io);
}

fn waitReady(io: std.Io, gpa: std.mem.Allocator, path: []const u8, nonce: []const u8, timeout_s: u64) !void {
    const ready_path = try std.fmt.allocPrint(gpa, "{s}.ready", .{path});
    defer gpa.free(ready_path);
    const deadline = try monotonicNs() + @as(i128, timeout_s) * std.time.ns_per_s;
    var bytes: [64]u8 = undefined;
    while (try monotonicNs() < deadline) {
        try checkStop();
        const got = std.Io.Dir.cwd().readFile(io, ready_path, &bytes) catch |err| {
            if (err != error.FileNotFound) return err;
            try std.Io.sleep(io, .fromMilliseconds(200), .awake);
            continue;
        };
        if (readyNonceMatches(got, nonce)) return;
        return error.BadReadyNonce;
    }
    return error.IdWaitTimeout;
}

fn waitAccepted(io: std.Io, gpa: std.mem.Allocator, path: []const u8, nonce: []const u8, timeout_s: u64) !void {
    const accepted_path = try std.fmt.allocPrint(gpa, "{s}.accepted", .{path});
    defer gpa.free(accepted_path);
    const deadline = try monotonicNs() + @as(i128, timeout_s) * std.time.ns_per_s;
    var bytes: [64]u8 = undefined;
    while (try monotonicNs() < deadline) {
        try checkStop();
        const got = std.Io.Dir.cwd().readFile(io, accepted_path, &bytes) catch |err| {
            if (err != error.FileNotFound) return err;
            try std.Io.sleep(io, .fromMilliseconds(200), .awake);
            continue;
        };
        if (readyNonceMatches(got, nonce)) return;
        return error.BadReadyNonce;
    }
    return error.IdWaitTimeout;
}

fn waitId(io: std.Io, gpa: std.mem.Allocator, path: []const u8, process_start_ns: i128, timeout_s: u64) !struct { cuda.nccl.UniqueId, [16]u8 } {
    const deadline = try monotonicNs() + @as(i128, timeout_s) * std.time.ns_per_s;
    var bytes: [320]u8 = undefined;
    while (try monotonicNs() < deadline) {
        try checkStop();
        const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| {
            if (err != error.FileNotFound) return err;
            try std.Io.sleep(io, .fromMilliseconds(200), .awake);
            continue;
        };
        if (!idFileFresh(stat.mtime.nanoseconds, process_start_ns)) {
            std.debug.print("NCCL id handoff: ignoring stale file (mtime {d}, process start {d})\n", .{ stat.mtime.nanoseconds, process_start_ns });
            try std.Io.sleep(io, .fromMilliseconds(200), .awake);
            continue;
        }
        const got = std.Io.Dir.cwd().readFile(io, path, &bytes) catch |err| {
            if (err == error.FileNotFound) continue;
            return err;
        };
        const handoff = try IdFile.decode(got);
        const result = handoff[0];
        const nonce = handoff[1];
        const nonce_hex = std.fmt.bytesToHex(nonce, .lower);
        const ready_path = try std.fmt.allocPrint(gpa, "{s}.ready", .{path});
        defer gpa.free(ready_path);
        try writeAtomic(io, gpa, ready_path, &nonce_hex);
        try waitAccepted(io, gpa, path, &nonce_hex, timeout_s);
        return .{ result, nonce };
    }
    return error.IdWaitTimeout;
}

/// The main communicator uses a fresh rank-0 id, mtime check and nonce echo before either rank enters NCCL init.
fn mainGroup(gpa: std.mem.Allocator, io: std.Io, path: []const u8, rank: usize, process_start_ns: i128, timeout_s: u64) !cuda.tp_group.Group {
    if (std.c.getenv("NCCL_COMM_ID") != null) return error.UnsetNcclCommId;
    var lib = try cuda.nccl.Library.open();
    errdefer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var nonce: [16]u8 = undefined;
        const now = try realtimeNs();
        var h = std.hash.Wyhash.init(@truncate(@as(u128, @bitCast(now))));
        const pid: u32 = @intCast(std.c.getpid());
        h.update(std.mem.asBytes(&pid));
        const nonce_a = h.final();
        std.mem.writeInt(u64, &nonce[0..8].*, nonce_a, .little);
        std.mem.writeInt(u64, &nonce[8..16].*, nonce_a ^ 0xa985a0d2fbd94a2b, .little);
        const nonce_hex = std.fmt.bytesToHex(nonce, .lower);
        const payload = IdFile.encode(id, nonce);
        try writeAtomic(io, gpa, path, &payload);
        try waitReady(io, gpa, path, &nonce_hex, timeout_s);
        const accepted_path = try std.fmt.allocPrint(gpa, "{s}.accepted", .{path});
        defer gpa.free(accepted_path);
        try writeAtomic(io, gpa, accepted_path, &nonce_hex);
    } else {
        const handoff = try waitId(io, gpa, path, process_start_ns, timeout_s);
        id = handoff[0];
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, @intCast(rank)), "ncclCommInitRank");
    return .{ .lib = lib, .comm = comm, .rank = rank, .world = 2 };
}

fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8) !cuda.Module {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.fatbin", .{ dir, name });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return cuda.Module.load(d, bytes);
}

/// Everything one rank runs: weights, State, kernels, transport, Forward and Engine.
const Gpu = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    setup: Setup,
    adm: state.Admission,
    w: weights.ModelWeights,
    st: state.State,
    prefill_gemms: ?prefill_gemm.Gemms = null,
    lane: cuda.qmmf.Lane = undefined,
    ex: cuda.exl3_experts.Experts = undefined,
    mods: [5]cuda.Module = undefined,
    k: forward.Kernels = undefined,
    /// TF_GLM53_PREFILL_W8A8=1 or =2: glm53_w8a8.fatbin from the fatbins directory.
    w8a8: cuda.w8a8.Kernels = undefined,
    qmmf_var: cuda.qmmf.VariantFns = undefined,
    l2pf: ?forward.l2pf_mod.L2pf = null,
    w8a8_live: bool = false,
    qmmf_var_live: bool = false,
    group: cuda.tp_group.Group = undefined,
    side: ?forward.SideGroup = null,
    pipe: ?forward.SideGroup = null,
    side_stream: ?cuda.Stream = null,
    events: [5]cuda.Event = undefined,
    pf: forward.Prefetch = undefined,
    stream: cuda.Stream = undefined,
    with_kernels: bool = false,
    weights_live: bool = false,
    state_live: bool = false,
    group_live: std.atomic.Value(bool) = .init(false),
    side_live: std.atomic.Value(bool) = .init(false),
    pipe_live: std.atomic.Value(bool) = .init(false),
    side_aborted: std.atomic.Value(bool) = .init(false),
    pipe_aborted: std.atomic.Value(bool) = .init(false),
    stream_live: bool = false,
    side_stream_live: bool = false,
    event_count: usize = 0,
    module_count: usize = 0,
    setup_live: bool = false,

    fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: cuda.abi.Device, o: Opts) !*Gpu {
        const g = try gpa.create(Gpu);
        g.* = .{ .gpa = gpa, .io = io, .d = d, .setup = undefined, .adm = undefined, .w = undefined, .st = undefined };
        errdefer if (g.setup_live) g.deinit(true) else gpa.destroy(g);
        g.setup = try Setup.init(gpa, io, o);
        g.setup_live = true;
        try checkStop();
        g.setup.report();
        g.adm = try admit(gpa, io, &g.setup, o);
        try checkStop();
        const before = mem(gpa, io, "before weights") orelse return error.NoMemInfo;
        g.w = try weights.ModelWeights.loadConfig(o.ckpt, g.setup.config, o.rank, 2, .{ .allocator = gpa, .io = io, .driver = d, .admission = &g.adm }, g.setup.select);
        g.weights_live = true;
        try checkStop();
        const after = mem(gpa, io, "after weights (unmapped, fadvised)") orelse return error.NoMemInfo;
        const cls = g.w.classes();
        std.debug.print("LOAD weights: {d} allocations, {d:.4} GiB allocated (logical {d:.4}); MemAvailable dropped {d:.4} GiB (overhead {d:.4} GiB)\n", .{ g.w.allocationCount(), state.gib(cls.total()), state.gib(g.w.logicalClasses().total()), state.gib(before.available -| after.available), state.gib((before.available -| after.available) -| cls.total()) });
        // N4: State is allocated only if the fresh MemAvailable still holds it above the reserve.
        if (g.setup.geometry.expert_rows_adapt != 0) {
            var probe = try cuda.arena.Vmm.open(d, device, 2 << 20);
            const gran: u64 = @intCast(probe.granularity);
            probe.close();
            const ap = try state.planAdapt(gpa, g.setup.geometry, o.context, gran);
            defer gpa.free(ap.tails);
            var vmm = try cuda.arena.Vmm.open(d, device, @intCast(ap.kv_bytes + ap.pool));
            var vmm_owned = true;
            defer if (vmm_owned) vmm.close();
            std.log.info("expert rows adapt: {d} below position {d} (KV tails {d:.2} GiB)", .{ g.setup.geometry.expert_rows_adapt, ap.limit_position, state.gib(ap.pool) });
            vmm_owned = false;
            g.st = try state.State.initWith(d, device, vmm, g.setup.geometry, o.context, o.rank, 2, &g.adm, after.available);
        } else {
            std.log.info("glm53 expert rows adapt: off", .{});
            g.st = try state.State.init(d, g.setup.geometry, o.context, o.rank, 2, &g.adm, after.available);
        }
        g.state_live = true;
        try checkStop();
        _ = mem(gpa, io, "after State");
        return g;
    }

    fn kernels(g: *Gpu, ctx: *const cuda.Context, o: Opts, process_start_ns: i128) !void {
        try checkStop();
        g.lane = try cuda.qmmf.Lane.load(g.d, (try ctx.capability()) / 10);
        if (g.setup.geometry.pre_fuse != .off or g.setup.geometry.prep_fuse != .off) try g.lane.loadGroup();
        g.ex = try cuda.exl3_experts.Experts.loadWith(g.d, if (std.c.getenv("TF_GLM53_MOE_NOALLOC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false);
        const rot_w4_raw = std.c.getenv("TF_GLM53_ROT_W4");
        const rot_w4 = if (rot_w4_raw) |rot_value| blk: {
            const rot_setting = std.mem.span(rot_value);
            if (std.mem.eql(u8, rot_setting, "0")) break :blk false;
            if (std.mem.eql(u8, rot_setting, "1")) break :blk true;
            return error.BadRotW4;
        } else false;
        g.ex.rot_w4 = rot_w4;
        if (rot_w4) std.log.info("TF_GLM53_ROT_W4=1: prompt routed rotation as one CTA a pair", .{});
        if (g.setup.geometry.prompt_experts_pmx) g.ex.pmx = try cuda.exl3_experts.pmx.Pmx.load(g.d);
        if (std.c.getenv("TF_GLM53_GU_MB")) |gu_value| {
            const gu_setting = std.mem.span(gu_value);
            if (std.mem.eql(u8, gu_setting, "1")) {
                if (g.ex.pmx) |*px| try px.useGuMb() else return error.GuMbNeedsPmx;
                std.log.info("TF_GLM53_GU_MB=1: prompt routed gate/up through a 5-slot mbarrier ring", .{});
            } else if (!std.mem.eql(u8, gu_setting, "0")) return error.BadGuMb;
        }
        if (g.setup.geometry.prompt_experts_p5a) g.ex.p5a = try cuda.exl3_experts.p5a.P5a.load(g.d);
        if (std.c.getenv("TF_GLM53_GU_W8S")) |w8s_value| {
            const w8s_setting = std.mem.span(w8s_value);
            if (std.mem.eql(u8, w8s_setting, "2")) {
                if (!g.setup.geometry.prompt_experts_p5a) return error.GuW8sNeedsP5a;
                g.ex.w8s = true;
                g.ex.w8s_all = true;
                std.log.info("TF_GLM53_GU_W8S=2: prompt routed gate/up of all items on e4m3 activations (changes output bits)", .{});
            } else if (std.mem.eql(u8, w8s_setting, "1")) {
                if (!g.setup.geometry.prompt_experts_p5a) return error.GuW8sNeedsP5a;
                g.ex.w8s = true;
                std.log.info("TF_GLM53_GU_W8S=1: prompt routed gate/up of large items on e4m3 activations (changes output bits)", .{});
            } else if (!std.mem.eql(u8, w8s_setting, "0")) return error.BadGuW8s;
        }
        if (std.c.getenv("TF_GLM53_DOWN_W8S")) |dw_value| {
            const dw_setting = std.mem.span(dw_value);
            if (std.mem.eql(u8, dw_setting, "1")) {
                if (!g.ex.w8s) return error.DownW8sNeedsGuW8s;
                g.ex.down_w8s = true;
                std.log.info("TF_GLM53_DOWN_W8S=1: prompt routed down of large items on e4m3 activations (changes output bits)", .{});
            } else if (!std.mem.eql(u8, dw_setting, "0")) return error.BadDownW8s;
        }
        g.mods[0] = try cuda.Module.load(g.d, cuda.kernels.glm_glue);
        g.module_count = 1;
        for ([_][]const u8{ "glm53_glue", "glm53_mla", "glm53_index", "glm53_head" }, 1..) |name, i| {
            try checkStop();
            g.mods[i] = try loadImage(g.gpa, g.io, g.d, o.fatbins, name);
            g.module_count += 1;
        }
        g.k = try forward.Kernels.load(.{ .glm_glue = g.mods[0], .glm53_glue = g.mods[1], .glm53_mla = g.mods[2], .glm53_index = g.mods[3], .glm53_head = g.mods[4] }, &g.lane, &g.ex, g.setup.geometry.index_mma_rows);
        g.k.head.fast = g.setup.geometry.cand_fast;
        g.k.head.v2 = g.setup.geometry.cand_v2;
        if (g.setup.geometry.smla_p3) try g.k.mla.loadSmlaP3(g.mods[2]);
        if (g.setup.decode_union or g.setup.geometry.own_attend or g.setup.geometry.q_one_gather) try g.k.mla.loadAttendVerify(g.mods[2]);
        if (g.setup.geometry.add_norm_fuse) try g.k.loadAddNormFuse(g.mods[1]);
        if (g.setup.geometry.qpe_absorb_fuse) try g.k.mla.loadAbsorbQpeV3(g.mods[2]);
        if (g.setup.geometry.qa_shard) try g.k.loadQaShard(g.mods[1]);
        if (g.setup.geometry.prefill_w8a8) {
            var w8a8_mod = try loadImage(g.gpa, g.io, g.d, o.fatbins, "glm53_w8a8");
            errdefer w8a8_mod.unload();
            g.w8a8 = try cuda.w8a8.Kernels.fromModule(w8a8_mod);
            if (std.c.getenv("TF_GLM53_W8_M64")) |m64_value| {
                const m64_setting = std.mem.span(m64_value);
                if (std.mem.eql(u8, m64_setting, "1")) {
                    try g.w8a8.useM64();
                    std.log.info("TF_GLM53_W8_M64=1: prefill W8A8 GEMM on 64-row tiles, two CTAs an SM", .{});
                } else if (!std.mem.eql(u8, m64_setting, "0")) return error.BadW8M64;
            }
            g.w8a8_live = true;
            g.k.w8a8 = g.w8a8.fns;
        }
        if (g.setup.geometry.qmmf_var != 0) {
            g.qmmf_var = try cuda.qmmf.VariantFns.load(g.d);
            g.qmmf_var_live = true;
            g.k.qmmf_var = &g.qmmf_var;
        }
        if (g.st.dequant_stream != null) g.prefill_gemms = try prefill_gemm.Gemms.init(g.gpa, g.d);
        g.stream = try cuda.Stream.init(g.d, true);
        g.stream_live = true;
        if (g.setup.l2pf_options.on) {
            const regions = try forward.l2pf_mod.regionsOf(g.gpa, &g.w);
            defer g.gpa.free(regions);
            g.l2pf = try forward.l2pf_mod.L2pf.init(g.gpa, g.d, regions, g.w.layers, g.setup.l2pf_options);
        }
        g.group = try mainGroup(g.gpa, g.io, o.idfile orelse return error.NeedIdFile, o.rank, process_start_ns, o.id_wait_s);
        g.group_live.store(true, .release);
        try g.warmPeer(o);
        const SwitchBits = extern struct { bits: u8 };
        var switch_bits: [2]SwitchBits = undefined;
        try g.group.gather(SwitchBits, g.d, g.stream, .{ .bits = decodeSwitchBits(.{ .route_compact = g.setup.geometry.route_compact, .own_attend = g.setup.geometry.own_attend, .q_one_gather = g.setup.geometry.q_one_gather, .partial_roce = g.setup.geometry.partial_roce }) }, &switch_bits);
        if (switch_bits[0].bits != switch_bits[1].bits) return error.PrefillCommunicationSettingsDisagree;
        const levers_on = g.setup.geometry.own_attend or g.setup.geometry.q_one_gather or g.setup.geometry.partial_roce;
        // B0: native.zig:1176-1177 opens RoCE on every run (gathers when roce.wanted()) and forces it for PARTIAL_ROCE.
        if (roceRequested() or levers_on) try g.group.openRoce(g.gpa, g.io, g.d, g.stream, 4 << 20, g.setup.geometry.partial_roce);
        if (g.setup.geometry.partial_roce and g.group.fast == null) return error.RoceUnavailable;
        if (levers_on and g.group.fast != null) std.log.info("RoCE gathers bound to the main stream; other streams and setup handshakes use NCCL", .{});
        if (g.group.fast != null) if (g.st.adapt) |ad| {
            if (!state.adaptRoceAligned(ad.ws)) return error.ExpertRowsAdaptRoceMisaligned;
            std.log.info("glm53 expert rows adapt with RoCE: adapt workspace 16-byte aligned (RoCE registers only its pinned region)", .{});
        };
        if (o.prefetch) {
            g.side = try forward.SideGroup.open(&g.group, g.d, g.stream);
            g.side_live.store(true, .release);
            g.side_stream = if (g.setup.geometry.comm_prio) try cuda.Stream.initPriority(g.d, .highest) else try cuda.Stream.init(g.d, true);
            g.side_stream_live = true;
            for (&g.events) |*e| {
                e.* = try cuda.Event.init(g.d, false);
                g.event_count += 1;
            }
            g.pf = .{ .side = g.side_stream.?, .coll = g.side.?.collective(), .ready = g.events[0], .fetched = .{ g.events[1], g.events[2] }, .attended = .{ g.events[3], g.events[4] } };
        }
        if (g.setup.attn_pipe) {
            g.pipe = try forward.SideGroup.open(&g.group, g.d, g.stream);
            g.pipe_live.store(true, .release);
        }
        g.with_kernels = true;
    }

    /// The peer's send/recv connected now, eagerly, through the forward's collective (as glm/cuda_native.zig
    /// warmPeers does for Flash): NCCL connects a peer on its first send/recv, and the first one of a short prompt is
    /// decode's DCP2 partial exchange. Both directions at once, from and into the State workspace (no allocation).
    fn warmPeer(g: *Gpu, o: Opts) !void {
        const ws = &g.st.ws;
        const bytes = @min(@as(usize, 4 << 20), ws.sizes.part / 2);
        const f = g.forwardFor(o);
        try f.coll.sendRecv(g.stream, ws.part, bytes, ws.part + bytes, bytes, 1 - o.rank, .attention_partial);
        try g.group.syncChecked(g.stream);
        std.debug.print("NCCL peer {d} connected ({d} B each way)\n", .{ 1 - o.rank, bytes });
    }

    fn abortCommunicators(g: *Gpu) void {
        if (g.side_live.load(.acquire) and g.side_aborted.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) if (g.side) |side| {
            _ = side.lib.api.ncclCommAbort(side.comm);
        };
        if (g.pipe_live.load(.acquire) and g.pipe_aborted.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) if (g.pipe) |pipe| {
            _ = pipe.lib.api.ncclCommAbort(pipe.comm);
        };
        if (g.group_live.load(.acquire)) g.group.abort();
    }

    fn deinit(g: *Gpu, aborting: bool) void {
        if (aborting) g.abortCommunicators();
        if (g.stream_live) g.stream.synchronize() catch {};
        if (g.side_stream_live) if (g.side_stream) |s| s.synchronize() catch {};
        if (g.l2pf) |*pf| pf.deinit();
        if (g.side_live.load(.acquire)) {
            if (!aborting) if (g.side) |*side| side.close();
            g.side_live.store(false, .release);
        }
        if (g.pipe_live.load(.acquire)) {
            if (!aborting) if (g.pipe) |*pipe| pipe.close();
            g.pipe_live.store(false, .release);
        }
        if (g.group_live.load(.acquire)) {
            if (g.group.fast != null) {
                const counts = g.group.routeCounts();
                std.debug.print("RoCE routes: roce {d}, nccl (size) {d}, nccl (other stream) {d}\n", .{ counts[0], counts[1], counts[2] });
            }
            g.group.close();
            g.group_live.store(false, .release);
        }
        if (g.state_live) g.st.deinit();
        if (g.prefill_gemms) |*gemms| gemms.deinit();
        if (g.weights_live) g.w.deinit();
        if (g.setup_live) g.setup.deinit(g.gpa);
        for (g.events[0..g.event_count]) |*e| e.deinit();
        if (g.side_stream_live) if (g.side_stream) |*s| s.deinit();
        if (g.stream_live) g.stream.deinit();
        if (g.w8a8_live) g.w8a8.unload();
        if (g.qmmf_var_live) g.qmmf_var.deinit();
        var i = g.module_count;
        while (i > 0) {
            i -= 1;
            g.mods[i].unload();
        }
        _ = mem(g.gpa, g.io, "after teardown");
        g.gpa.destroy(g);
    }

    fn forwardFor(g: *Gpu, o: Opts) forward.Forward {
        return .{ .gpa = g.gpa, .k = &g.k, .w = &g.w, .st = &g.st, .stream = g.stream, .decode_mla16 = g.setup.decode_mla16, .coll = forward.Collective.group(&g.group), .prompt_experts_p5a = g.setup.geometry.prompt_experts_p5a, .pre_fuse = g.setup.geometry.pre_fuse, .prep_fuse = g.setup.geometry.prep_fuse, .ws_trim = g.setup.geometry.ws_trim, .moe_post = g.setup.geometry.moe_post, .moe_ep = g.setup.geometry.moe_ep, .comm_prio = g.setup.geometry.comm_prio, .pipe_coll = if (g.pipe) |*p| p.collective() else null, .rank = o.rank, .chunk_rows = g.setup.geometry.chunk_rows, .index_mma_rows = g.setup.geometry.index_mma_rows, .index_qs = g.setup.geometry.index_qs, .prompt_mla = g.setup.geometry.prompt_mla, .smla_p = g.setup.geometry.smla_p, .smla_fp8 = g.setup.geometry.smla_fp8, .smla_p3 = g.setup.geometry.smla_p3, .decode_union = g.setup.decode_union, .route_compact = g.setup.geometry.route_compact, .own_attend = g.setup.geometry.own_attend, .q_one_gather = g.setup.geometry.q_one_gather, .partial_roce = g.setup.geometry.partial_roce, .decode_union_rows = g.setup.decode_union_rows, .shard_qkv_a = g.setup.geometry.shard_qkv_a, .qa_shard = g.setup.geometry.qa_shard, .prefill_w8a8 = g.setup.geometry.prefill_w8a8, .prefill_w8a8_all = g.setup.geometry.prefill_w8a8_all, .qmmf_var = g.setup.geometry.qmmf_var != 0, .gather_ovl = g.setup.geometry.gather_ovl, .decode_side = g.setup.geometry.decode_side, .prompt_experts = g.setup.geometry.prompt_experts, .prompt_experts_wide = g.setup.geometry.prompt_experts_wide, .prompt_experts_pmx = g.setup.geometry.prompt_experts_pmx, .prefetch = if (g.setup.geometry.prefetch) &g.pf else null, .l2pf = if (g.l2pf) |*pf| pf else null, .prefetch_enabled = g.setup.geometry.prefetch, .tbo_enabled = g.setup.geometry.tbo, .piece_pipe = g.setup.geometry.piece_pipe, .moe_pipe = g.setup.geometry.moe_pipe, .index_merge = g.setup.geometry.index_merge, .index_pipe = g.setup.geometry.index_pipe, .add_norm_fuse = g.setup.geometry.add_norm_fuse, .qpe_absorb_fuse = g.setup.geometry.qpe_absorb_fuse, .prefill_gemms = if (g.prefill_gemms) |*gemms| gemms else null };
    }

    fn downloadFor(g: *const Gpu) engine.Download {
        return .{ .group = if (g.group.fast != null) &g.group else null };
    }

    /// Last head rows' rank-local fp32 logits (rows x vocab_local).
    fn logits(g: *Gpu, rows: usize, out: []u8) !void {
        try g.stream.synchronize();
        const ws = &g.st.ws;
        if (out.len != rows * g.w.vocab.len() * 4) return error.BadLogitsBuffer;
        try g.d.check(g.d.api.cuMemcpyDtoH_v2(out.ptr, ws.logits, out.len), "logits");
    }

    /// Hash of the KV, rope and index rows of every plane for positions below `n` (this rank's slots: rank r holds
    /// positions p < n with p % 2 == r, so rank 1 holds floor(n/2) rows and rank 0 ceil(n/2)).
    fn planesHash(g: *Gpu, n: usize) !u64 {
        try g.stream.synchronize();
        var h = std.hash.Wyhash.init(0);
        const slots = try kvsave.localRows(g.group.rank, n);
        for (0..g.st.layers) |l| {
            inline for (.{ .{ 1024, "latentAt" }, .{ 128, "ropeAt" } }) |pl| {
                const ptr = @field(state.State, pl[1])(&g.st, l);
                const buf: cuda.DeviceBuffer = .{ .d = g.d, .ptr = ptr, .len = slots * pl[0], .borrowed = true };
                h.update(std.mem.asBytes(&(try wyhash(g.gpa, buf))));
            }
        }
        for (0..g.st.full_index_layers) |l| {
            const buf: cuda.DeviceBuffer = .{ .d = g.d, .ptr = g.st.indexAt(l), .len = slots * 256, .borrowed = true };
            h.update(std.mem.asBytes(&(try wyhash(g.gpa, buf))));
        }
        return h.final();
    }
};

fn writeOut(io: std.Io, gpa: std.mem.Allocator, o: Opts, name: []const u8, bytes: []const u8) !void {
    const dir = o.out orelse return;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.rank{d}", .{ dir, name, o.rank });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    std.debug.print("DUMP {s} ({d} B)\n", .{ path, bytes.len });
}

const ProbeRow = struct { ids: [64]u32, window_id: i32, pos: usize };
const GateCapture = struct {
    g: *Gpu,
    rows: []const ProbeRow,
    window_id: i32,
    base_pos: usize = 0,
    output: std.ArrayList(dump.GateRow) = .empty,

    fn sink(ctx: *anyopaque, stream: cuda.Stream, first: usize, rows: usize, _: u64, _: usize) anyerror!void {
        const self: *GateCapture = @ptrCast(@alignCast(ctx));
        const g = self.g;
        const vocab = g.w.vocab.len();
        const per_rank = try std.math.mul(usize, rows * vocab, @sizeOf(f32));
        var gathered = try cuda.DeviceBuffer.alloc(g.d, per_rank * 2);
        defer gathered.free();
        try g.group.allGather(stream, g.st.ws.logits, gathered.ptr, per_rank);
        try stream.synchronize();
        const values = try g.gpa.alloc(f32, rows * vocab * 2);
        defer g.gpa.free(values);
        try gathered.download(0, std.mem.sliceAsBytes(values));
        const rank_values = try g.gpa.alloc([]const f32, 2);
        defer g.gpa.free(rank_values);
        const offsets = [_]usize{ 0, vocab };
        for (0..rows) |r| {
            const position = self.base_pos + first + r;
            const probe = for (self.rows) |candidate| {
                if (candidate.window_id == self.window_id and candidate.pos == position) break candidate;
            } else continue;
            for (0..2) |rank| rank_values[rank] = values[rank * rows * vocab + r * vocab ..][0..vocab];
            const result = try sampler.gateRow(g.gpa, rank_values, &offsets, &probe.ids);
            defer g.gpa.free(result.logp);
            var row: dump.GateRow = .{
                .ids = undefined,
                .logp = undefined,
                .lse = result.lse,
                .mass_outside = result.mass_outside,
                .argmax = @intCast(result.argmax),
                .window_id = probe.window_id,
                .pos = @intCast(probe.pos),
            };
            for (0..64) |j| {
                row.ids[j] = @intCast(probe.ids[j]);
                row.logp[j] = result.logp[j];
            }
            try self.output.append(g.gpa, row);
        }
    }
};

fn probeRows(gpa: std.mem.Allocator, io: std.Io, path: []const u8, window_id: i32) ![]ProbeRow {
    var file = try core.safetensors.File.open(gpa, io, path);
    defer file.close(io);
    const ids = file.get("ids") orelse return error.ProbeMissingIds;
    const windows = file.get("window_id") orelse return error.ProbeMissingWindowIds;
    const positions = file.get("pos") orelse return error.ProbeMissingPositions;
    if (ids.dtype != .i32 or ids.rank != 2 or ids.dim(1) != 64 or windows.dtype != .i32 or windows.rank != 1 or windows.dim(0) != ids.dim(0) or windows.bytes.len != ids.dim(0) * 4 or positions.rank != 1 or positions.dim(0) != ids.dim(0)) return error.BadProbeShape;
    if ((positions.dtype != .i32 and positions.dtype != .i64) or positions.bytes.len != ids.dim(0) * positions.dtype.size()) return error.BadProbeShape;
    var out: std.ArrayList(ProbeRow) = .empty;
    errdefer out.deinit(gpa);
    for (0..ids.dim(0)) |i| {
        const win = std.mem.readInt(i32, windows.bytes[i * 4 ..][0..4], .little);
        if (win != window_id) continue;
        const position: usize = if (positions.dtype == .i32) blk: {
            const p = std.mem.readInt(i32, positions.bytes[i * 4 ..][0..4], .little);
            if (p < 0) return error.BadProbePosition;
            break :blk @intCast(p);
        } else blk: {
            const p = std.mem.readInt(i64, positions.bytes[i * 8 ..][0..8], .little);
            if (p < 0) return error.BadProbePosition;
            break :blk @intCast(p);
        };
        for (out.items) |prior| if (prior.pos == position) return error.DuplicateProbePosition;
        var row: ProbeRow = .{ .ids = undefined, .window_id = win, .pos = position };
        for (0..64) |j| {
            const id = std.mem.readInt(i32, ids.bytes[(i * 64 + j) * 4 ..][0..4], .little);
            if (id < 0) return error.BadProbeId;
            const token_id: u32 = @intCast(id);
            for (row.ids[0..j]) |prior| if (prior == token_id) return error.DuplicateProbeId;
            row.ids[j] = token_id;
        }
        try out.append(gpa, row);
    }
    if (out.items.len == 0) return error.EmptyProbeWindow;
    return out.toOwnedSlice(gpa);
}

const Case = struct { window_id: i32, ids_path: []const u8 };

fn parseCases(gpa: std.mem.Allocator, text: []const u8) ![]Case {
    var out: std.ArrayList(Case) = .empty;
    errdefer out.deinit(gpa);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;
        var fields = std.mem.tokenizeAny(u8, trimmed, " \t");
        const win_text = fields.next() orelse return error.BadCases;
        const path = fields.next() orelse return error.BadCases;
        if (fields.next() != null) return error.BadCases;
        const win = std.fmt.parseInt(i32, win_text, 10) catch return error.BadCases;
        for (out.items) |prior| if (prior.window_id == win) return error.BadCases;
        try out.append(gpa, .{ .window_id = win, .ids_path = path });
    }
    if (out.items.len == 0) return error.BadCases;
    return out.toOwnedSlice(gpa);
}

fn parsePoints(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| {
        if (part.len == 0) return error.BadPoints;
        const value = std.fmt.parseInt(usize, part, 10) catch return error.BadPoints;
        if (value == 0 or (out.items.len > 0 and value <= out.items[out.items.len - 1])) return error.BadPoints;
        try out.append(gpa, value);
    }
    if (out.items.len == 0) return error.BadPoints;
    return out.toOwnedSlice(gpa);
}

fn percentile(sorted: []const f64, q: f64) f64 {
    if (sorted.len == 0) return 0;
    const rank = @as(usize, @intFromFloat(@ceil(q * @as(f64, @floatFromInt(sorted.len)))));
    return sorted[@min(sorted.len - 1, if (rank == 0) 0 else rank - 1)];
}

fn cycledIds(gpa: std.mem.Allocator, src: []const u32, n: usize) ![]u32 {
    if (src.len == 0) return error.EmptyPrompt;
    const ids = try gpa.alloc(u32, n);
    for (ids, 0..) |*id, i| id.* = src[i % src.len];
    return ids;
}

fn gate(g: *Gpu, o: Opts) !bool {
    var case_text: ?[]u8 = null;
    defer if (case_text) |text| g.gpa.free(text);
    var parsed_cases: ?[]Case = null;
    defer if (parsed_cases) |cases| g.gpa.free(cases);
    const cases: []const Case = if (o.cases) |path| blk: {
        case_text = try std.Io.Dir.cwd().readFileAlloc(g.io, path, g.gpa, .limited(1 << 24));
        parsed_cases = try parseCases(g.gpa, case_text.?);
        break :blk parsed_cases.?;
    } else &.{.{ .window_id = o.window_id, .ids_path = o.ids.? }};

    var capture: GateCapture = .{ .g = g, .rows = &.{}, .window_id = 0 };
    defer capture.output.deinit(g.gpa);
    var f = g.forwardFor(o);
    f.gate_mode = true;
    f.sink = .{ .ctx = @ptrCast(&capture), .sink_fn = GateCapture.sink };
    for (cases) |case| {
        const started = try monotonicNs();
        const rows = try probeRows(g.gpa, g.io, o.probe.?, case.window_id);
        defer g.gpa.free(rows);
        const ids = try readIds(g.gpa, g.io, case.ids_path);
        defer g.gpa.free(ids);
        if (ids.len > o.context) return error.CaseTooLong;
        for (rows) |row| if (row.pos >= ids.len) return error.ProbePositionPastPrompt;
        const before = capture.output.items.len;
        capture.rows = rows;
        capture.window_id = case.window_id;
        capture.base_pos = 0;
        const positions = try g.gpa.alloc(usize, ids.len);
        defer g.gpa.free(positions);
        for (positions, 0..) |*position, i| position.* = i;
        var at: usize = 0;
        while (at < ids.len) {
            try checkStop();
            const chunk = try forward.chunkRows(positions, f.chunk_rows, at);
            capture.base_pos = positions[chunk.start];
            try f.pieced(.{ .positions = positions[chunk.start..chunk.end], .tokens = ids[chunk.start..chunk.end] });
            at = chunk.end;
        }
        if (capture.output.items.len - before != rows.len) return error.GateCaptureIncomplete;
        const elapsed = try monotonicNs() - started;
        std.debug.print("GATE case window {d}: {d} prompt rows, {d} probe rows, {d} ms\n", .{ case.window_id, ids.len, rows.len, @divTrunc(elapsed, std.time.ns_per_ms) });
    }
    try checkStop();
    if (o.rank != 0) return true;
    const tensors = try dump.gateTensors(g.gpa, capture.output.items);
    defer dump.freeGateTensors(g.gpa, tensors);
    const bytes = try dump.encode(g.gpa, tensors);
    defer g.gpa.free(bytes);
    try writeAtomic(g.io, g.gpa, o.out.?, bytes);
    const sidecar = try gateSidecar(g.gpa);
    defer g.gpa.free(sidecar);
    const sidecar_path = try std.fmt.allocPrint(g.gpa, "{s}.json", .{o.out.?});
    defer g.gpa.free(sidecar_path);
    try writeAtomic(g.io, g.gpa, sidecar_path, sidecar);
    std.debug.print("GATE dump {s}: {d} rows, fp32_moe_sum=true\n", .{ o.out.?, capture.output.items.len });
    return true;
}

fn memAvailable(gpa: std.mem.Allocator, io: std.Io) ?u64 {
    const info = uma.readMeminfo(gpa, io) catch return null;
    return info.available;
}

/// `load --steady [--drafter DIR]`: the serving steady state (kernels, NCCL connected, a 2048-row prefill and 4
/// decodes so the decode graphs are captured), its MemAvailable, then the drafter made resident beside it; one
/// ADMISSION line per rank against the reserve floor. A measurement: the exit code is 0 for PASS and FAIL alike.
fn steadyLoad(g: *Gpu, ctx: *const cuda.Context, o: Opts) !void {
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    var e: engine.Engine = .{ .gpa = g.gpa, .f = g.forwardFor(o), .download = g.downloadFor() };
    defer e.deinit();
    var tok = try prefillChecked(&e, ids[0..@min(ids.len, 2048)]);
    for (0..4) |_| {
        try checkStop();
        var out: [1]u32 = undefined;
        try e.verify(&.{tok}, &out);
        try e.keep(1);
        tok = out[0];
    }
    const steady = mem(g.gpa, g.io, "steady (kernels, NCCL, graphs)") orelse return error.NoMemInfo;
    const floor = (try state.reserveFromEnv(steady.total)).bytes;
    var available = steady.available;
    var pass = available >= floor;
    if (o.drafter) |dir| {
        const v = try draft_admit.admit(g.gpa, g.io, g.d, ctx, g.stream, dir, o.rank, o.drafter_bits, o.drafter_streams, steady.available, floor, memAvailable, e.f.w.head);
        available = v.available;
        pass = v.pass;
    }
    std.debug.print("ADMISSION context {d}{s}: MemAvailable {d:.3} GiB vs floor {d:.2} GiB -> {s}\n", .{ o.context, if (o.drafter != null) " with drafter" else "", state.gib(available), state.gib(floor), if (pass) "PASS" else "FAIL" });
}

fn prefillChecked(e: *engine.Engine, ids: []const u32) !u32 {
    if (ids.len == 0) return error.EmptyPrompt;
    var at: usize = 0;
    var token: u32 = 0;
    while (at < ids.len) {
        try checkStop();
        // The engine splits a prompt by its own chunk_rows; hand it whole chunks so --chunk-rows 8192 runs 8192-row chunks.
        const end = @min(ids.len, at + @max(@as(usize, forward.prefill_chunk_rows), e.f.chunk_rows));
        token = try e.prefill(ids[at..end]);
        at = end;
    }
    return token;
}

fn run(g: *Gpu, o: Opts) !void {
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    // First L1 check (review M6): the rope table rows Zig uploads, for the first positions, against Python's.
    var rope: [16 * 2 * forward.rope_half]f32 = undefined;
    for (0..16) |p| forward.ropeRow(p, rope[p * 2 * forward.rope_half ..][0..forward.rope_half], rope[(p * 2 + 1) * forward.rope_half ..][0..forward.rope_half]);
    try writeOut(g.io, g.gpa, o, "rope_cos_sin_16.f32", std.mem.sliceAsBytes(&rope));
    var e: engine.Engine = .{ .gpa = g.gpa, .io = g.io, .f = g.forwardFor(o), .download = g.downloadFor(), .mem = try uma.MemTrace.fromEnv(o.rank) };
    e.mem.floor = try state.floorFromEnv();
    defer e.deinit();
    // TF_GLM53_POISON=B (0-255): fill the KV arena, workspace and stage plane with byte B before the prefill, so a
    // kernel that reads unwritten memory shows up as output that depends on B (diagnosis; unset = today's run).
    if (std.c.getenv("TF_GLM53_POISON")) |v| {
        const byte = try std.fmt.parseInt(u8, std.mem.span(v), 10);
        try g.st.kv_arena.fill8(byte, e.f.stream.handle);
        try g.st.workspace.fill8(byte, e.f.stream.handle);
        if (g.st.stage_plane.len != 0) try g.st.stage_plane.fill8(byte, e.f.stream.handle);
        try e.f.stream.synchronize();
        std.debug.print("RUN state poisoned with byte {d}\n", .{byte});
    }
    const row = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(row);
    var all: std.ArrayList(u8) = .empty;
    defer all.deinit(g.gpa);
    var tok = try prefillChecked(&e, ids);
    try checkStop();
    try g.logits(1, row);
    try all.appendSlice(g.gpa, row);
    std.debug.print("RUN prefill {d} rows -> token {d}\n", .{ ids.len, tok });
    for (0..o.decode) |step| {
        try checkStop();
        var out: [1]u32 = undefined;
        try e.verify(&.{tok}, &out);
        try g.logits(1, row);
        try all.appendSlice(g.gpa, row);
        try e.keep(1);
        std.debug.print("RUN decode {d} pos {d} -> token {d} ({t})\n", .{ step, e.pos - 1, out[0], e.last_outcome });
        tok = out[0];
    }
    try checkStop();
    // Rank-local fp32 logits rows [steps][vocab_local] for the gate (the other rank has the other half).
    try writeOut(g.io, g.gpa, o, "final_logits.f32", all.items);
}

fn hasProfileSamples(profile: *const forward.Profile) bool {
    for (profile.counts) |count| if (count != 0) return true;
    return false;
}

fn bench(g: *Gpu, o: Opts) !void {
    if (o.rows > g.setup.verify_max) return error.RowsPastVerifyMax;
    const profile_prefill = g.setup.profile_prefill;
    if (profile_prefill) {
        const graph_cache = std.c.getenv("TF_GRAPH_CACHE") orelse return error.PrefillProfileNeedsGraphCacheOff;
        if (!std.mem.eql(u8, std.mem.span(graph_cache), "0")) return error.PrefillProfileNeedsGraphCacheOff;
    }
    const after_load = mem(g.gpa, g.io, "bench after load");
    if (o.rank == 0) if (after_load) |info| std.debug.print("BENCH MemAvailable after load {d:.2} GiB\n", .{state.gib(info.available)});
    const src = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(src);
    const points = try parsePoints(g.gpa, o.points);
    defer g.gpa.free(points);
    const need = try std.math.add(usize, try std.math.add(usize, points[points.len - 1], o.append), try std.math.add(usize, try std.math.mul(usize, o.decode_steps, o.rows), 1));
    if (need > o.context) return error.ContextTooSmall;
    const ids = try cycledIds(g.gpa, src, need);
    defer g.gpa.free(ids);
    if (o.rank == 0) std.debug.print("BENCH ids {d} from {d} source tokens{s}\n", .{ need, src.len, if (src.len < need) " (cycled)" else "" });
    if (o.rows != 1 and o.rank == 0) std.debug.print("BENCH rows={d}\n", .{o.rows});
    const expert_rows_values = if (o.expert_rows_count == 0) o.expert_rows_values[0..1] else o.expert_rows_values[0..o.expert_rows_count];
    if (g.setup.geometry.expert_rows_adapt != 0 and expert_rows_values.len > 1) return error.OneExpertRowsPerRun;
    // One value per process on the Sparks: re-admitting a new State after the weights are resident double-counts them on
    // UMA (MemAvailable already excludes them), and a one-rank refusal would leave the other rank in a collective.
    if (expert_rows_values.len > 1) return error.OneExpertRowsPerRun;
    var first = true;
    for (expert_rows_values) |expert_rows| {
    if (g.setup.geometry.expert_rows != expert_rows) {
        g.st.deinit();
        g.state_live = false;
        g.setup.geometry.expert_rows = expert_rows;
        g.adm = try admit(g.gpa, g.io, &g.setup, o);
        const info = mem(g.gpa, g.io, "bench state") orelse return error.NoMemInfo;
        g.st = try state.State.init(g.d, g.setup.geometry, o.context, o.rank, 2, &g.adm, info.available);
        g.state_live = true;
    }
    var e: engine.Engine = .{ .gpa = g.gpa, .io = g.io, .f = g.forwardFor(o), .download = g.downloadFor(), .mem = try uma.MemTrace.fromEnv(o.rank) };
    if (o.head_k) |k| try e.setSamplingRule(.{ .temperature = 1, .top_k = k });
    e.mem.floor = try state.floorFromEnv();
    defer e.deinit();
    if (profile_prefill) {
        e.prof = try forward.Profile.init(e.gpa, e.f.stream.d, e.f.w.config.layers + 1);
        e.f.prof = &e.prof.?;
        e.f.prof_prefill = false;
    }
    const stream = e.f.stream;
    if (o.warmup > 0) {
        if (o.warmup > need) return error.ContextTooSmall;
        var warm_tok = try prefillChecked(&e, ids[0..o.warmup]);
        for (0..2) |_| {
            var out: [1]u32 = undefined;
            try e.verify(&.{warm_tok}, &out);
            try e.keep(1);
            warm_tok = out[0];
        }
        try e.rollback(0);
        if (e.prof) |*prof| prof.reset();
    }
    if (profile_prefill) {
        e.f.prof_prefill = true;
        e.f.prof_decode_sub = g.setup.profile_decode_sub;
    }
    for (points) |point| {
        try checkStop();
        try e.rollback(0);
        try g.st.kv_arena.fill8(0, stream.handle);
        try g.st.workspace.fill8(0, stream.handle);
        if (g.st.stage_plane.len != 0) try g.st.stage_plane.fill8(0, stream.handle);
        try stream.synchronize(); // the memsets are not part of the prefill time
        if (profile_prefill) e.prof.?.reset();
        var t0 = try monotonicNs();
        var tok = try prefillChecked(&e, ids[0..point]);
        try stream.synchronize();
        const prefill_ms = @as(f64, @floatFromInt(try monotonicNs() - t0)) / std.time.ns_per_ms;
        if (profile_prefill) {
            const prof = &e.prof.?;
            if (o.rank == 0) {
                var profile_stage_ms: f64 = 0;
                for (std.enums.values(forward.Stage)) |stage| if (!forward.isSubStage(stage)) {
                    profile_stage_ms += prof.totals_ms[@intFromEnum(stage)];
                };
                inline for (std.enums.values(forward.Stage)) |stage| {
                    const i = @intFromEnum(stage);
                    const total = prof.totals_ms[i];
                    const per_token_us = total / @as(f64, @floatFromInt(point)) * 1000;
                    if (forward.isSubStage(stage)) {
                        const parent_stage: forward.Stage = if (stage == .attn_pre_indexer_gather) .attn_comm else .attn_pre;
                        const parent = prof.totals_ms[@intFromEnum(parent_stage)];
                        const share = if (parent == 0) 0 else total * 100 / parent;
                        std.debug.print("PROFILE_PREFILL sub={s} total_ms={d:.3} per_token_us={d:.3} share_of_parent={d:.1}\n", .{ forward.Profile.name(stage), total, per_token_us, share });
                    } else {
                        const share = if (profile_stage_ms == 0) 0 else total * 100 / profile_stage_ms;
                        std.debug.print("PROFILE_PREFILL stage={s} total_ms={d:.3} per_token_us={d:.3} share={d:.1}\n", .{ forward.Profile.name(stage), total, per_token_us, share });
                    }
                }
                std.debug.print("PROFILE_PREFILL sum_stage_ms={d:.3} measured_ms={d:.3}\n", .{ profile_stage_ms, prefill_ms });
            }
            prof.reset();
        }
        var samples = try g.gpa.alloc(f64, o.decode_steps);
        defer g.gpa.free(samples);
        var outcomes = try g.gpa.alloc(cuda.graph_cache.Outcome, o.decode_steps);
        defer g.gpa.free(outcomes);
        var eager: usize = 0;
        var captured: usize = 0;
        var replayed: usize = 0;
        for (0..o.decode_steps) |s| {
            try checkStop();
            t0 = try monotonicNs();
            var out: [16]u32 = undefined;
            if (o.rows == 1) {
                try e.verify(&.{tok}, out[0..1]);
                try e.keep(1);
            } else {
                var win: [16]u32 = undefined;
                win[0] = tok;
                for (1..o.rows) |r| win[r] = ids[point + s * o.rows + r];
                try e.verify(win[0..o.rows], out[0..o.rows]);
                try e.keep(o.rows);
            }
            samples[s] = @as(f64, @floatFromInt(try monotonicNs() - t0)) / std.time.ns_per_ms;
            outcomes[s] = e.last_outcome;
            switch (e.last_outcome) { .eager => eager += 1, .captured => captured += 1, .replayed => replayed += 1 }
            tok = out[o.rows - 1];
        }
        const replay_samples = try g.gpa.alloc(f64, if (replayed > 0) replayed else samples.len);
        defer g.gpa.free(replay_samples);
        var n: usize = 0;
        // Steady state is the replayed steps (a key's warm-up and capture steps can fall anywhere: keys follow the
        // position bucket); without any replay, every step.
        for (samples, outcomes) |sample, outcome| if (replayed == 0 or outcome == .replayed) {
            replay_samples[n] = sample;
            n += 1;
        };
        std.mem.sort(f64, replay_samples, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (replay_samples) |sample| sum += sample;
        const mean = if (n == 0) 0 else sum / @as(f64, @floatFromInt(n));
        e.rewindToFrontier();
        var append_ms: f64 = 0;
        if (o.append > 0) {
            t0 = try monotonicNs();
            _ = try prefillChecked(&e, ids[point .. point + o.append]);
            try stream.synchronize();
            append_ms = @as(f64, @floatFromInt(try monotonicNs() - t0)) / std.time.ns_per_ms;
        }
        const memory = mem(g.gpa, g.io, "bench point");
        const available = if (memory) |m| state.gib(m.available) else 0;
        const p50 = percentile(replay_samples[0..n], 0.5);
        const p90 = percentile(replay_samples[0..n], 0.9);
        if (o.rank == 0) {
            if (e.prof) |*prof| {
                if (hasProfileSamples(prof)) {
                    var profile_step_ms: f64 = 0;
                    for (std.enums.values(forward.Stage)) |stage| if (!forward.isSubStage(stage)) {
                        profile_step_ms += prof.totals_ms[@intFromEnum(stage)] / @as(f64, @floatFromInt(o.decode_steps));
                    };
                    inline for (std.enums.values(forward.Stage)) |stage| {
                        const i = @intFromEnum(stage);
                        const per_step = prof.totals_ms[i] / @as(f64, @floatFromInt(o.decode_steps));
                        if (!forward.isSubStage(stage)) {
                            const share = if (profile_step_ms == 0) 0 else per_step * 100 / profile_step_ms;
                            std.debug.print("PROFILE stage={s} total_ms={d:.3} per_step_ms={d:.3} share={d:.1}\n", .{ forward.Profile.name(stage), prof.totals_ms[i], per_step, share });
                        } else if (e.f.prof_decode_sub and prof.totals_ms[i] > 0) {
                            std.debug.print("PROFILE sub={s} total_ms={d:.3} per_step_ms={d:.3}\n", .{ forward.Profile.name(stage), prof.totals_ms[i], per_step });
                        }
                    }
                    std.debug.print("PROFILE sum_stage_ms={d:.3} measured_step_ms={d:.3}\n", .{ profile_step_ms, mean });
                    prof.reset();
                }
            }
            std.debug.print("BENCH point L={d} expert_rows={d} cold-prefill {d:.1} ms ({d:.1} tok/s) | decode {d} steps mean {d:.2} ms p50 {d:.2} p90 {d:.2} ({d:.2} tok/s; eager {d} captured {d} replayed {d}) | append {d} rows {d:.1} ms | MemAvailable {d:.2} GiB\n", .{ point, expert_rows, prefill_ms, @as(f64, @floatFromInt(point)) * 1000 / prefill_ms, o.decode_steps, mean, p50, p90, if (mean == 0) 0 else 1000 / mean, eager, captured, replayed, o.append, append_ms, available });
            if (o.out) |dir| {
                const path = try std.fmt.allocPrint(g.gpa, "{s}/bench.jsonl", .{dir});
                defer g.gpa.free(path);
                if (first) try std.Io.Dir.cwd().writeFile(g.io, .{ .sub_path = path, .data = "" });
                const line = try std.fmt.allocPrint(g.gpa, "{{\"point\":{d},\"expert_rows\":{d},\"prefill_ms\":{d},\"prefill_tok_s\":{d},\"decode_steps\":{d},\"decode_mean_ms\":{d},\"decode_p50_ms\":{d},\"decode_p90_ms\":{d},\"decode_tok_s\":{d},\"eager\":{d},\"captured\":{d},\"replayed\":{d},\"append_rows\":{d},\"append_ms\":{d},\"mem_available_gib\":{d}}}\n", .{ point, expert_rows, prefill_ms, @as(f64, @floatFromInt(point)) * 1000 / prefill_ms, o.decode_steps, mean, p50, p90, if (mean == 0) 0 else 1000 / mean, eager, captured, replayed, o.append, append_ms, available });
                defer g.gpa.free(line);
                var file = try std.Io.Dir.cwd().openFile(g.io, path, .{ .mode = .write_only });
                defer file.close(g.io);
                const stat = try file.stat(g.io);
                try file.writePositionalAll(g.io, line, stat.size);
            }
        }
        first = false;
    }
    }
}

/// `saved` v2: aligned expert-group cuts are the pass/fail set. Unaligned requested cuts are informational and are
/// reported as valid but not bit-exact; the format rounds every save cut down by A=lcm(512,expert_rows).
/// For each aligned cut m and suffix s,
/// a fresh prefill of F[0..m+s] gives the reference planes hash and last-row logits; then F[0..m] is prefilled, this
/// rank's rows [0, m/2) of every latent/rope/indexer plane and the DFlash2 ring are saved to --kv-dir, the planes, the
/// stage plane and the ring are overwritten with garbage, the saved file is restored and F[m..m+s] prefilled. PASS when
/// the hash and logits equal the fresh run and the ring after restore equals the ring at the save. Fault cases: one
/// rank's file corrupted, or missing -> agree() makes both ranks fall back to a fresh prefill with identical output.
const SavedGate = struct {
    g: *Gpu,
    o: Opts,
    e: *engine.Engine,
    host: *drafts.Host,
    staging: ?*uma.Staging,
    flag: cuda.DeviceBuffer,
    identity: kvsave.Digest,
    ring: []u8,
    ring_back: []u8,
    ring_check: []u8,
    sections: std.ArrayList(Section) = .empty,

    const Section = struct { dev: u64 = 0, host: ?[]u8 = null, len: usize };
    const default_cuts = [_]usize{ 1024, 2048, 2560, 29696, 126976 };
    const suffixes = [_]usize{ 1, 511, 3000 };

    /// Both ranks always call this after each saved-file step (also after a local failure): min over the ranks.
    /// The flag words move by async copies on g.stream, ordered with the all-reduce on the same stream.
    fn agreeCall(ctx: *anyopaque, words: []u32) anyerror!void {
        const s: *SavedGate = @ptrCast(@alignCast(ctx));
        if (words.len > 128) return error.AgreementTooLarge;
        const g = s.g;
        const bytes = std.mem.sliceAsBytes(words);
        try g.d.check(g.d.api.cuMemcpyHtoDAsync_v2(s.flag.ptr, bytes.ptr, bytes.len, g.stream.handle), "saved agree in");
        try g.group.allReduce(g.stream, s.flag.ptr, s.flag.ptr + bytes.len, words.len, .u32, .min);
        try g.d.check(g.d.api.cuMemcpyDtoHAsync_v2(bytes.ptr, s.flag.ptr + bytes.len, bytes.len, g.stream.handle), "saved agree out");
        try g.stream.synchronize();
    }

    /// The v2 body starts with the padded ring, then delta planes. The current gate writes root deltas (base_at=0).
    fn plan(s: *SavedGate, m: usize, ring_host: []u8, base_at: usize) !usize {
        const g = s.g;
        const rows = try kvsave.localDeltaRows(s.o.rank, base_at, m);
        const base_rows = try kvsave.localRows(s.o.rank, base_at);
        s.sections.clearRetainingCapacity();
        var total: usize = 0;
        if (base_at == 0) {
            try s.sections.append(g.gpa, .{ .host = ring_host, .len = ring_host.len });
            total += ring_host.len;
        }
        for (0..g.st.layers) |l| {
            try s.sections.append(g.gpa, .{ .dev = g.st.latentAt(l) + base_rows * 1024, .len = rows * 1024 });
            try s.sections.append(g.gpa, .{ .dev = g.st.ropeAt(l) + base_rows * 128, .len = rows * 128 });
            total += rows * (1024 + 128);
        }
        for (0..g.st.full_index_layers) |l| {
            try s.sections.append(g.gpa, .{ .dev = g.st.indexAt(l) + base_rows * 256, .len = rows * 256 });
            total += rows * 256;
        }
        return total;
    }

    fn save(s: *SavedGate, path: [:0]const u8, ids_full: []const u32, at: usize, base_at: usize, parent_key: u64, seq: u64, from: i64) !void {
        const g = s.g;
        const ring_host = if (base_at == 0) s.ring else s.ring[0..0];
        const body_len = try s.plan(at, ring_host, base_at);
        const ring_span = if (base_at == 0) s.ring.len else 0;
        var w = try kvsave.Writer.begin(g.gpa, path, s.identity, @intCast(s.o.rank), 2, ids_full, base_at, at, parent_key, seq, body_len, from, ring_span);
        s.pump(&w) catch |err| {
            w.abort();
            return err;
        };
        try w.finish();
    }

    fn pump(s: *SavedGate, w: *kvsave.Writer) !void {
        const g = s.g;
        const buf = s.staging.?.slots[0];
        var fill: usize = 0;
        try g.stream.synchronize();
        for (s.sections.items) |sec| {
            var off: usize = 0;
            while (off < sec.len) {
                const n = @min(sec.len - off, kvsave.chunk_bytes - fill);
                if (sec.host) |h| @memcpy(buf[fill..][0..n], h[off..][0..n]) else try g.d.check(g.d.api.cuMemcpyDtoH_v2(buf[fill..].ptr, sec.dev + off, n), "saved plane read");
                fill += n;
                off += n;
                if (fill == kvsave.chunk_bytes) {
                    try w.writeChunk(buf, fill);
                    fill = 0;
                }
            }
        }
        if (fill > 0) try w.writeChunk(buf, fill);
    }

    /// Restore the planes (straight into the State arenas through one pinned staging slot) and the ring bytes into
    /// `ring_back`; returns the ring's `from`. A failure part way leaves the planes partly written: the caller agrees
    /// with the peer and falls back to a fresh prefill from position 0.
    fn restore(s: *SavedGate, path: [:0]const u8, ids_full: []const u32, at: usize, base_at: usize) !i64 {
        const g = s.g;
        const ring_host = if (base_at == 0) s.ring_back else s.ring_back[0..0];
        const body_len = try s.plan(at, ring_host, base_at);
        var r = try kvsave.Reader.open(g.gpa, path, s.identity, @intCast(s.o.rank), 2, ids_full, base_at, at, body_len);
        defer r.close();
        const buf = s.staging.?.slots[0];
        var sec: usize = 0;
        var sec_off: usize = 0;
        try g.stream.synchronize();
        for (0..r.digests.len) |i| {
            const data = try r.readChunk(i, buf);
            var off: usize = 0;
            while (off < data.len) {
                if (sec >= s.sections.items.len) return error.SavedBodyTooLong;
                const cur = s.sections.items[sec];
                const n = @min(cur.len - sec_off, data.len - off);
                if (cur.host) |h| @memcpy(h[sec_off..][0..n], data[off..][0..n]) else try g.d.check(g.d.api.cuMemcpyHtoD_v2(cur.dev + sec_off, data[off..].ptr, n), "saved plane write");
                off += n;
                sec_off += n;
                if (sec_off == cur.len) {
                    sec += 1;
                    sec_off = 0;
                }
            }
        }
        if (sec != s.sections.items.len) return error.SavedBodyTooShort;
        return r.header.ring_from;
    }

    /// Planes, stage plane and ring overwritten, engine bookkeeping reset: what a fresh process would hold.
    fn clobber(s: *SavedGate) !void {
        const g = s.g;
        try s.e.rollback(0);
        for ([_]cuda.DeviceBuffer{ g.st.latent, g.st.rope, g.st.index_keys, g.st.stage_plane }) |b| try b.fill8(0xa5, g.stream.handle);
        try g.stream.synchronize();
        @memset(s.ring_check, 0x5a);
        try drafts.Host.adoptHook(s.host, s.ring_check[0..@intCast(s.host.dr.?.geometry.ringBytes())], 0);
    }

    /// A rank-local step whose failure must not leave the peer alone in a collective: the error is logged, and the
    /// caller hands the result to agree() before the next collective.
    fn localOk(s: *SavedGate, what: []const u8, result: anyerror!void) bool {
        result catch |err| {
            std.debug.print("SAVED rank {d}: {s} failed: {s}\n", .{ s.o.rank, what, @errorName(err) });
            return false;
        };
        return true;
    }

    /// Both ranks clobber (agreed), then prefill F[0..n] from position 0; the reference planes hash and logits.
    fn fresh(s: *SavedGate, ids: []const u32, n: usize, logits_out: []u8) !u64 {
        if (!try s.e.agree(s.localOk("clobber", s.clobber()))) return error.SavedClobberFailed;
        _ = try prefillChecked(s.e, ids[0..n]);
        try s.g.logits(1, logits_out);
        return s.g.planesHash(n);
    }

    /// The workspace and planes after an unrelated 512-token prefill, so a resume cannot lean on rows or scratch
    /// that a run over the same tokens left behind.
    fn scramble(s: *SavedGate) !void {
        if (!try s.e.agree(s.localOk("clobber", s.clobber()))) return error.SavedClobberFailed;
        var other: [512]u32 = undefined;
        for (&other, 0..) |*id, i| id.* = @intCast((i * 29 + 7) % 32000 + 1);
        _ = try prefillChecked(s.e, &other);
    }

    fn restoreAll(s: *SavedGate, path: [:0]const u8, ids: []const u32, m: usize) !void {
        try s.clobber();
        const from = try s.restore(path, ids, m, 0);
        const ring_bytes: usize = @intCast(s.host.dr.?.geometry.ringBytes());
        try s.e.restoreTo(m, ids, s.ring_back[0..ring_bytes], from);
    }

    fn restoreChainAll(s: *SavedGate, base_path: [:0]const u8, child_path: [:0]const u8, ids: []const u32, base_at: usize, at: usize) !void {
        try s.clobber();
        const from = try s.restore(base_path, ids, base_at, 0);
        _ = try s.restore(child_path, ids, at, base_at);
        const ring_bytes: usize = @intCast(s.host.dr.?.geometry.ringBytes());
        try s.e.restoreTo(at, ids, s.ring_back[0..ring_bytes], from);
    }

    fn saveChainChild(s: *SavedGate, path: [:0]const u8, ids: []const u32, base_at: usize, at: usize) !void {
        _ = try prefillChecked(s.e, ids[base_at..at]);
        const parent_key = kvsave.keyOf(ids[0..base_at]);
        try s.save(path, ids, at, base_at, parent_key, 2, 0);
    }

    /// Restore + suffix prefill, or (when any rank failed any step) a fresh prefill of the whole n rows on both ranks.
    fn resumed(s: *SavedGate, path: [:0]const u8, ids: []const u32, m: usize, n: usize, saved_from: i64, logits_out: []u8, restored: *bool, ring_ok: *bool) !u64 {
        try s.scramble();
        restored.* = try s.e.agree(s.localOk("restore", s.restoreAll(path, ids, m)));
        if (!restored.*) {
            ring_ok.* = true;
            return s.fresh(ids, n, logits_out);
        }
        const ring_bytes: usize = @intCast(s.host.dr.?.geometry.ringBytes());
        const ring_local = if (s.e.saveDrafterState(s.ring_check[0..ring_bytes])) |now_from|
            now_from == saved_from and std.mem.eql(u8, s.ring_check[0..ring_bytes], s.ring[0..ring_bytes])
        else |err| blk: {
            std.debug.print("SAVED rank {d}: ring export failed: {s}\n", .{ s.o.rank, @errorName(err) });
            break :blk false;
        };
        ring_ok.* = try s.e.agree(ring_local);
        _ = try prefillChecked(s.e, ids[m..n]);
        try s.g.logits(1, logits_out);
        return s.g.planesHash(n);
    }

    fn resumedChain(s: *SavedGate, base_path: [:0]const u8, child_path: [:0]const u8, ids: []const u32, base_at: usize, at: usize, n: usize, saved_from: i64, logits_out: []u8, restored: *bool, ring_ok: *bool) !u64 {
        try s.scramble();
        restored.* = try s.e.agree(s.localOk("restore base+delta", s.restoreChainAll(base_path, child_path, ids, base_at, at)));
        if (!restored.*) {
            ring_ok.* = true;
            return s.fresh(ids, n, logits_out);
        }
        const ring_bytes: usize = @intCast(s.host.dr.?.geometry.ringBytes());
        const ring_local = if (s.e.saveDrafterState(s.ring_check[0..ring_bytes])) |now_from|
            now_from == saved_from and std.mem.eql(u8, s.ring_check[0..ring_bytes], s.ring[0..ring_bytes])
        else |err| blk: {
            std.debug.print("SAVED rank {d}: chain ring export failed: {s}\n", .{ s.o.rank, @errorName(err) });
            break :blk false;
        };
        ring_ok.* = try s.e.agree(ring_local);
        _ = try prefillChecked(s.e, ids[at..n]);
        try s.g.logits(1, logits_out);
        return s.g.planesHash(n);
    }

    /// Prefill F[0..m] and save it on both ranks (agreed); the ring's `from` at the cut.
    fn saveCut(s: *SavedGate, path: [:0]const u8, ids: []const u32, m: usize, from_out: *i64) !bool {
        if (!try s.e.agree(s.localOk("clobber", s.clobber()))) return error.SavedClobberFailed;
        _ = try prefillChecked(s.e, ids[0..m]);
        var from: i64 = 0;
        const ring_bytes: usize = @intCast(s.host.dr.?.geometry.ringBytes());
        @memset(s.ring[ring_bytes..], 0);
        const exported = if (s.e.saveDrafterState(s.ring[0..ring_bytes])) |f| blk: {
            from = f;
            break :blk true;
        } else |err| blk: {
            std.debug.print("SAVED rank {d}: ring export failed: {s}\n", .{ s.o.rank, @errorName(err) });
            break :blk false;
        };
        const saved = exported and s.localOk("save", s.save(path, ids, m, 0, 0, 1, from));
        from_out.* = from;
        return s.e.agree(saved);
    }
};

fn settingsIdentity(gpa: std.mem.Allocator, o: Opts) !kvsave.Digest {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("glm53-bringup-saved\x00");
    h.update(o.ckpt);
    var word: [8]u8 = undefined;
    for ([_]u64{ o.context, o.chunk_rows, o.expert_rows }) |v| {
        std.mem.writeInt(u64, &word, v, .little);
        h.update(&word);
    }
    // Every numerics switch the process was started with (TF_GLM53_*, TF_GLM_*, TF_ROCE*), sorted by name.
    var vars: std.ArrayList([]const u8) = .empty;
    defer vars.deinit(gpa);
    var i: usize = 0;
    while (std.c.environ[i]) |entry| : (i += 1) {
        const kv = std.mem.span(entry);
        if (std.mem.startsWith(u8, kv, "TF_GLM53_") or std.mem.startsWith(u8, kv, "TF_GLM_") or std.mem.startsWith(u8, kv, "TF_ROCE")) try vars.append(gpa, kv);
    }
    std.mem.sort([]const u8, vars.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    for (vars.items) |kv| {
        h.update(kv);
        h.update("\x00");
    }
    var out: kvsave.Digest = undefined;
    h.final(&out);
    return out;
}

fn fatbinIdentity() kvsave.Digest {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    inline for (comptime std.meta.declarations(cuda.kernels)) |decl| {
        if (comptime @TypeOf(@field(cuda.kernels, decl)) == []const u8) {
            const image: []const u8 = @field(cuda.kernels, decl);
            h.update(decl);
            h.update(&.{0});
            h.update(image);
        }
    }
    var out: kvsave.Digest = undefined;
    h.final(&out);
    return out;
}

fn executableIdentity() !kvsave.Digest {
    const fd = std.c.open("/proc/self/exe", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.ExecutableOpen;
    defer _ = std.c.close(fd);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 20]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, buf[0..].ptr, buf.len);
        if (n < 0) return error.ExecutableRead;
        if (n == 0) break;
        h.update(buf[0..@intCast(n)]);
    }
    var out: kvsave.Digest = undefined;
    h.final(&out);
    return out;
}

fn parseCuts(gpa: std.mem.Allocator, text: ?[]const u8) ![]usize {
    const raw = text orelse return gpa.dupe(usize, &SavedGate.default_cuts);
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const v = std.fmt.parseInt(usize, part, 10) catch return error.Usage;
        if (v == 0 or v % 512 != 0) return error.Usage;
        try out.append(gpa, v);
    }
    if (out.items.len == 0) return error.Usage;
    return out.toOwnedSlice(gpa);
}

/// Flip one byte of this file's body (rank-local; no collective).
fn corruptSaved(path: [:0]const u8, body_offset: u64) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDWR });
    if (fd < 0) return error.DiskOpen;
    defer _ = std.c.close(fd);
    var one: [1]u8 = undefined;
    if (std.c.pread(fd, &one, 1, @intCast(body_offset + 4096)) != 1) return error.DiskRead;
    one[0] ^= 0x01;
    if (std.c.pwrite(fd, &one, 1, @intCast(body_offset + 4096)) != 1) return error.DiskWrite;
    if (std.c.fsync(fd) != 0) return error.DiskSync;
    _ = kvsave.dropPageCache(path);
}

fn gateEngine(g: *Gpu, ctx: *const cuda.Context, o: Opts, e: *engine.Engine, host: *drafts.Host) !void {
    e.* = .{ .gpa = g.gpa, .io = g.io, .f = g.forwardFor(o), .mem = try uma.MemTrace.fromEnv(o.rank) };
    e.mem.floor = try state.floorFromEnv();
    try e.history.ensureTotalCapacity(g.gpa, o.context);
    try drafts.Host.init(host, g.gpa, ctx, g.stream, &g.group, g.mods[0], g.mods[1], g.w.embed, g.w.head, .{ .lo = g.w.vocab.lo, .hi = g.w.vocab.hi }, o.rank);
    host.down4 = drafts.draftDown4Switch();
    errdefer host.deinit();
    if (drafts.shareScratchEnabled()) return error.SavedGateNeedsPrivateDrafterScratch;
    try host.load(g.io, o.drafter orelse return error.NeedDrafter, o.drafter_bits, false);
    const dr = host.dr.?;
    e.drafter = .{
        .ctx = host,
        .window = dr.geometry.window,
        .taps_prefill = .{ .f = dr.k.tap, .out = dr.taps_prefill, .hidden = 6144, .layers = dr.geometry.taps, .cap_rows = dr.geometry.window },
        .taps_step = .{ .f = dr.k.tap, .out = dr.taps_step, .hidden = 6144, .layers = dr.geometry.taps, .cap_rows = glm.dflash.STEP_ROWS },
        .restart_fn = drafts.Host.restartHook,
        .rewind_fn = drafts.Host.rewindHook,
        .prompt_fn = drafts.Host.promptHook,
        .verify_fn = drafts.Host.verifyHook,
        .draft_fn = drafts.Host.draftHook,
        .state_fn = drafts.Host.stateHook,
        .adopt_fn = drafts.Host.adoptHook,
        .ring_adopt_fn = drafts.Host.ringAdopt,
    };
}

fn savedCheck(g: *Gpu, ctx: *const cuda.Context, o: Opts) !bool {
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    const cuts = try parseCuts(g.gpa, o.cuts);
    defer g.gpa.free(cuts);
    if (g.st.ws.piece_rows != 512) return error.UnexpectedPieceRows;
    var e: engine.Engine = undefined;
    defer e.deinit();
    var host: drafts.Host = undefined;
    try gateEngine(g, ctx, o, &e, &host);
    defer host.deinit();
    const dr = host.dr.?;
    var staging = try uma.Staging.initPinned(g.d);
    defer staging.deinit();
    var flag = try cuda.DeviceBuffer.alloc(g.d, 1024);
    defer flag.free();
    const ring_len: usize = @intCast(dr.geometry.ringBytes());
    const ring_span = std.mem.alignForward(usize, ring_len, kvsave.chunk_bytes); // padded to whole 4 MiB chunks
    const rings = try g.gpa.alloc(u8, 3 * ring_span);
    defer g.gpa.free(rings);
    var s: SavedGate = .{ .g = g, .o = o, .e = &e, .host = &host, .staging = &staging, .flag = flag, .identity = kvsave.identityOf(try settingsIdentity(g.gpa, o), fatbinIdentity(), try executableIdentity(), @intCast(o.rank), 2, @intCast(try kvsave.cutAlignment(o.expert_rows))), .ring = rings[0..ring_span], .ring_back = rings[ring_span..][0..ring_span], .ring_check = rings[2 * ring_span ..][0..ring_span] };
    defer s.sections.deinit(g.gpa);
    e.agreement = .{ .ctx = &s, .call_fn = SavedGate.agreeCall };
    const after = mem(g.gpa, g.io, "saved: drafter + staging") orelse return error.NoMemInfo;
    const floor = (try state.reserveFromEnv(after.total)).bytes;
    if (!try e.agree(after.available >= floor)) {
        std.debug.print("SAVED FAIL: MemAvailable below the {d:.2} GiB floor on a rank after the drafter + staging\n", .{state.gib(floor)});
        return false;
    }
    const a = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(a);
    const b = try g.gpa.alloc(u8, a.len);
    defer g.gpa.free(b);
    var ok = true;
    var faults_done = false;
    const alignment = try kvsave.cutAlignment(o.expert_rows);
    for (cuts) |m| {
        try checkStop();
        if (m % alignment != 0) {
            const rounded = try kvsave.roundCutDown(m, o.expert_rows);
            std.debug.print("INFO saved cut={d}: valid, not bit-exact (A={d}; save rounds to {d}); excluded from pass/fail\n", .{ m, alignment, rounded });
            continue;
        }
        if (m + SavedGate.suffixes[0] > ids.len or m + SavedGate.suffixes[0] > o.context) {
            std.debug.print("SAVED skip cut {d}: ids {d}, context {d}\n", .{ m, ids.len, o.context });
            continue;
        }
        const path = try kvsave.pathFor(g.gpa, o.kv_dir.?, s.identity, @intCast(o.rank), kvsave.keyOf(ids[0..m]));
        defer g.gpa.free(path);
        defer _ = std.c.unlink(path.ptr);
        var from: i64 = 0;
        const t0 = try monotonicNs();
        if (!try s.saveCut(path, ids, m, &from)) {
            std.debug.print("FAIL saved cut={d}: save failed on a rank\n", .{m});
            ok = false;
            continue;
        }
        const save_ms = @as(f64, @floatFromInt(try monotonicNs() - t0)) / std.time.ns_per_ms;
        std.debug.print("SAVED cut={d} rank {d}: {s} (prefill + save {d:.0} ms)\n", .{ m, o.rank, path, save_ms });
        // Exercise a base plus one delta child. The child contains only rows [m, child_at) and names the base key.
        const child_at = m + alignment;
        if (child_at + SavedGate.suffixes[0] <= ids.len and child_at + SavedGate.suffixes[0] <= o.context) {
            const child_path = try kvsave.pathFor(g.gpa, o.kv_dir.?, s.identity, @intCast(o.rank), kvsave.keyOf(ids[0..child_at]));
            defer g.gpa.free(child_path);
            defer _ = std.c.unlink(child_path.ptr);
            const chain_saved = s.localOk("save delta child", s.saveChainChild(child_path, ids, m, child_at));
            if (!try e.agree(chain_saved)) {
                std.debug.print("FAIL saved chain base={d} child={d}: save failed on a rank\n", .{ m, child_at });
                ok = false;
            } else {
                const chain_n = child_at + SavedGate.suffixes[0];
                const chain_fresh = try s.fresh(ids, chain_n, a);
                var chain_restored = false;
                var chain_ring_ok = false;
                const chain_hash = try s.resumedChain(path, child_path, ids, m, child_at, chain_n, from, b, &chain_restored, &chain_ring_ok);
                const chain_same = try e.agree(chain_hash == chain_fresh and std.mem.eql(u8, a, b));
                const chain_pass = chain_restored and chain_ring_ok and chain_same;
                std.debug.print("{s} saved chain base={d} child={d}: restored {}, ring {s}, planes+logits {s}\n", .{ if (chain_pass) "PASS" else "FAIL", m, child_at, chain_restored, if (chain_ring_ok) "equal" else "DIFFER", if (chain_same) "equal" else "DIFFER" });
                ok = chain_pass and ok;
            }
        }
        for (SavedGate.suffixes) |suffix| {
            try checkStop();
            const n = m + suffix;
            if (n > ids.len or n > o.context) continue;
            const fresh_hash = try s.fresh(ids, n, a);
            var restored = false;
            var ring_ok = false;
            const t1 = try monotonicNs();
            const hash = try s.resumed(path, ids, m, n, from, b, &restored, &ring_ok);
            const ms = @as(f64, @floatFromInt(try monotonicNs() - t1)) / std.time.ns_per_ms;
            const same = try e.agree(hash == fresh_hash and std.mem.eql(u8, a, b));
            const pass = restored and ring_ok and same;
            std.debug.print("{s} saved cut={d} suffix={d}: restored {}, ring {s}, planes+logits {s} (scramble + restore + suffix {d:.0} ms)\n", .{ if (pass) "PASS" else "FAIL", m, suffix, restored, if (ring_ok) "equal" else "DIFFER", if (same) "equal" else "DIFFER", ms });
            ok = pass and ok;
        }
        if (!faults_done) {
            // Fault cases on the first cut, each from a fresh save: one rank's file corrupted or missing, the other
            // rank's file good. Both ranks must agree to fall back and produce the fresh output.
            faults_done = true;
            const n = m + SavedGate.suffixes[0];
            const fresh_hash = try s.fresh(ids, n, a);
            const body_len = try s.plan(m, s.ring, 0);
            const body_offset = kvsave.headerBytes(kvsave.digestCount(body_len)) + kvsave.aligned(m * 4);
            const Fault = struct { label: []const u8, rank: usize, corrupt: bool };
            const faults = [_]Fault{
                .{ .label = "rank 1 file corrupt", .rank = 1, .corrupt = true },
                .{ .label = "rank 0 file corrupt", .rank = 0, .corrupt = true },
                .{ .label = "rank 0 file missing", .rank = 0, .corrupt = false },
                .{ .label = "rank 1 file missing", .rank = 1, .corrupt = false },
            };
            for (faults) |fault| {
                try checkStop();
                if (!try s.saveCut(path, ids, m, &from)) {
                    std.debug.print("FAIL saved fault ({s}): re-save failed on a rank\n", .{fault.label});
                    ok = false;
                    continue;
                }
                if (o.rank == fault.rank) {
                    if (fault.corrupt) {
                        corruptSaved(path, body_offset) catch |err| {
                            std.debug.print("SAVED rank {d}: corrupt failed: {s}\n", .{ o.rank, @errorName(err) });
                        };
                    } else {
                        _ = std.c.unlink(path.ptr);
                    }
                }
                var f_restored = true;
                var f_ring = false;
                const f_hash = try s.resumed(path, ids, m, n, from, b, &f_restored, &f_ring);
                const f_same = try e.agree(f_hash == fresh_hash and std.mem.eql(u8, a, b));
                const f_pass = !f_restored and f_same;
                std.debug.print("{s} saved fault ({s}) cut={d} suffix={d}: {s}\n", .{ if (f_pass) "PASS" else "FAIL", fault.label, m, SavedGate.suffixes[0], if (f_pass) "fallback ok" else if (f_restored) "restored a bad file" else "fallback output DIFFERS" });
                ok = f_pass and ok;
            }
        }
    }
    std.debug.print("{s} saved: all cases\n", .{if (ok) "PASS" else "FAIL"});
    return ok;
}

fn cacheGate(g: *Gpu, ctx: *const cuda.Context, o: Opts) !bool {
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    const alignment = try kvsave.cutAlignment(o.expert_rows);
    if (g.st.ws.piece_rows != 512 or alignment < 1024) {
        std.debug.print("FAIL cachegate: run with --expert-rows 1024 (served fit config)\n", .{});
        return false;
    }
    const n: usize = @min(ids.len, o.context -| 1);
    var cut_pair: [2]usize = undefined;
    if (o.cuts) |raw| {
        const parsed = try parseCuts(g.gpa, raw);
        defer g.gpa.free(parsed);
        if (parsed.len != 2 or parsed[0] >= parsed[1] or parsed[1] >= n or parsed[0] % alignment != 0 or parsed[1] % alignment != 0) return error.Usage;
        cut_pair = .{ parsed[0], parsed[1] };
    } else {
        cut_pair = .{ (n / 3) / alignment * alignment, ((2 * n) / 3) / alignment * alignment };
    }
    const c1 = cut_pair[0];
    const c2 = cut_pair[1];
    if (c1 < 2 * alignment or c1 >= c2 or c2 >= n) return error.Usage;

    var e: engine.Engine = undefined;
    defer e.deinit();
    var host: drafts.Host = undefined;
    try gateEngine(g, ctx, o, &e, &host);
    defer host.deinit();
    var flag = try cuda.DeviceBuffer.alloc(g.d, 1024);
    defer flag.free();
    const ring_bytes: usize = @intCast(host.dr.?.geometry.ringBytes());
    const ring = try g.gpa.alloc(u8, ring_bytes * 2);
    defer g.gpa.free(ring);
    var agree_gate: SavedGate = .{ .g = g, .o = o, .e = &e, .host = &host, .staging = null, .flag = flag, .identity = undefined, .ring = ring[0..ring_bytes], .ring_back = ring[ring_bytes..], .ring_check = ring[ring_bytes..] };
    e.agreement = .{ .ctx = &agree_gate, .call_fn = SavedGate.agreeCall };
    const sys = [_]u32{@intCast(c1)};
    const settings = try settingsIdentity(g.gpa, o);
    const identity = kvsave.identityOf(settings, fatbinIdentity(), try executableIdentity(), @intCast(o.rank), 2, @intCast(alignment));
    agree_gate.identity = identity;
    const hex = std.fmt.bytesToHex(identity, .lower);
    const cache_path = try std.fmt.allocPrintSentinel(g.gpa, "{s}/{s}/rank{d}", .{ o.kv_dir.?, &hex, o.rank }, 0);
    defer g.gpa.free(cache_path);
    var setup_ok = true;
    std.Io.Dir.cwd().deleteTree(g.io, cache_path) catch |err| if (err != error.FileNotFound) { setup_ok = false; };
    std.Io.Dir.cwd().createDirPath(g.io, cache_path) catch { setup_ok = false; };
    if (!try e.agree(setup_ok)) return error.CacheGateSetupFailed;

    var pool = try prefix_snap.prefix_cache.Pool.init(g.d, ctx, 4);
    var pool_live = true;
    defer if (pool_live) pool.deinit();
    var runtime = try g.gpa.create(prefix_snap.prefix_cache.Runtime);
    errdefer g.gpa.destroy(runtime);
    runtime.* = try prefix_snap.prefix_cache.Runtime.init(g.gpa, o.kv_dir.?, identity, @intCast(o.rank), .{ .expert_rows = o.expert_rows, .step = alignment, .span = std.mem.alignForward(usize, n + 1, alignment) }, 64 << 30, 0, pool);
    defer { runtime.deinit(); g.gpa.destroy(runtime); }
    pool_live = false;
    pool = undefined;
    var dev = try prefix_snap.CudaDevice.init(g.d, ctx, g.stream);
    defer dev.deinit();
    const ring_hooks: prefix_snap.RingHooks = .{ .ctx = &host, .view_fn = gateRingView };
    var slots: [10][]align(kvsave.alignment) u8 = undefined;
    for (&slots, 0..) |*slot, slot_index| slot.* = runtime.pool.?.bytes(slot_index);
    const layout = try prefix_snap.layoutFromState(&g.st, state.latent_row_bytes, state.rope_row_bytes, state.index_row_bytes);
    var snap = try gpaCreateSnap(g, g.io, runtime, &e, dev.device(), ring_hooks, layout, slots, o.context);
    defer { snap.deinit(); g.gpa.destroy(snap); }
    const logits_ref = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(logits_ref);
    const logits_got = try g.gpa.alloc(u8, logits_ref.len);
    defer g.gpa.free(logits_got);
    const logits_ref2 = try g.gpa.alloc(u8, logits_ref.len);
    defer g.gpa.free(logits_ref2);
    const ring_at_c2 = try g.gpa.alloc(u8, ring_bytes);
    defer g.gpa.free(ring_at_c2);
    const info = mem(g.gpa, g.io, "cachegate: drafter + staging + prefix cache") orelse return error.NoMemInfo;
    const reserve = (try state.reserveFromEnv(info.total)).bytes;
    if (!try e.agree(info.available >= reserve)) {
        std.debug.print("FAIL cachegate: MemAvailable below the {d:.2} GiB floor after setup\n", .{state.gib(reserve)});
        return false;
    }
    var ring_from_c2: i64 = 0;
    var failures: usize = 0;
    const ref = try cacheGateCold(&agree_gate, ids[0..n], logits_ref);
    std.debug.print("PASS cachegate reference: cold output captured for {d} tokens\n", .{n});

    try checkStop();
    const planned = try runtime.plan(ids[0..n], 0, &sys, c2);
    if (planned.len != 2 or planned[0].at != c1 or planned[0].kind != .system or planned[1].at != c2 or planned[1].kind != .history) {
        std.debug.print("FAIL cachegate planner: expected [{d} system, {d} history]\n", .{ c1, c2 });
        return false;
    }
    try cacheGateClobber(&agree_gate);
    const save_start = try monotonicNs();
    try e.prefillRows(ids[0..c1]);
    const ring_save1_start = try monotonicNs();
    try snap.save(try cacheGateCut(runtime, c1));
    const ring_save1_ms = @as(f64, @floatFromInt(try monotonicNs() - ring_save1_start)) / std.time.ns_per_ms;
    try e.prefillRows(ids[c1..c2]);
    const ring_save2_start = try monotonicNs();
    try snap.save(try cacheGateCut(runtime, c2));
    const ring_save2_ms = @as(f64, @floatFromInt(try monotonicNs() - ring_save2_start)) / std.time.ns_per_ms;
    ring_from_c2 = try e.saveDrafterState(ring_at_c2);
    const final_tok = try prefillChecked(&e, ids[c2..n]);
    try g.logits(1, logits_got);
    const final_hash = try g.planesHash(n);
    var save_same = try cacheGateSame(&agree_gate, final_tok, final_hash, logits_got, ref, logits_ref);
    try snap.commit(2);
    const key1 = kvsave.keyOf(ids[0..c1]);
    const key2 = kvsave.keyOf(ids[0..c2]);
    const node1 = runtime.index.get(key1);
    const node2 = runtime.index.get(key2);
    save_same = save_same and node1 != null and node2 != null and !node1.?.pending and !node2.?.pending;
    if (node1) |_| save_same = save_same and cacheGateFileExists(runtime, key1);
    if (node2) |_| save_same = save_same and cacheGateFileExists(runtime, key2);
    save_same = try e.agree(save_same);
    const save_ms = @as(f64, @floatFromInt(try monotonicNs() - save_start)) / std.time.ns_per_ms;
    std.debug.print("{s} cachegate save: split invariance, committed files; total {d:.1} ms saves={d} bytes={d} ring_ms=[{d:.1},{d:.1}]\n", .{ if (save_same) "PASS" else "FAIL", save_ms, snap.saves, runtime.used_bytes, ring_save1_ms, ring_save2_ms });
    if (!save_same) failures += 1;

    try checkStop();
    try cacheGateClobber(&agree_gate);
    const hit2 = runtime.lookup(ids[0..n], 0);
    var load_ok = hit2 != null and hit2.?.key == key2;
    const restore_start = try monotonicNs();
    const got_at = try snap.load(0, ids[0..c2], key2);
    const restore_ms = @as(f64, @floatFromInt(try monotonicNs() - restore_start)) / std.time.ns_per_ms;
    load_ok = load_ok and got_at == c2;
    const ring_from_got = try e.saveDrafterState(agree_gate.ring_check[0..ring_bytes]);
    load_ok = load_ok and ring_from_got == ring_from_c2 and std.mem.eql(u8, ring_at_c2, agree_gate.ring_check[0..ring_bytes]);
    const tok2 = try prefillChecked(&e, ids[c2..n]);
    try g.logits(1, logits_got);
    const hash2 = try g.planesHash(n);
    load_ok = try cacheGateSame(&agree_gate, tok2, hash2, logits_got, ref, logits_ref) and load_ok;
    load_ok = try e.agree(load_ok);
    const restore_gbps = if (restore_ms > 0) @as(f64, @floatFromInt(runtime.used_bytes)) / (restore_ms * 1_000_000.0) else 0;
    std.debug.print("{s} cachegate fresh leaf load: {d:.1} ms {d:.2} GB/s\n", .{ if (load_ok) "PASS" else "FAIL", restore_ms, restore_gbps });
    if (!load_ok) failures += 1;

    try checkStop();
    const tail_len: usize = @min(n - c1, 3000);
    const prompt2 = try g.gpa.alloc(u32, c1 + tail_len);
    defer g.gpa.free(prompt2);
    @memcpy(prompt2[0..c1], ids[0..c1]);
    for (prompt2[c1..], 0..) |*token, idx| token.* = @intCast((idx * 29 + 7) % 32000 + 1);
    const ref2 = try cacheGateCold(&agree_gate, prompt2, logits_ref2);
    try cacheGateClobber(&agree_gate);
    const hit1 = runtime.lookup(prompt2, 0);
    var shared_ok = hit1 != null and hit1.?.key == key1;
    const shared_at = try snap.load(0, prompt2[0..c1], key1);
    shared_ok = shared_ok and shared_at == c1;
    const tok3 = try prefillChecked(&e, prompt2[c1..]);
    try g.logits(1, logits_got);
    const hash3 = try g.planesHash(prompt2.len);
    shared_ok = try cacheGateSame(&agree_gate, tok3, hash3, logits_got, ref2, logits_ref2) and shared_ok;
    shared_ok = try e.agree(shared_ok);
    std.debug.print("{s} cachegate shared system prefix: C1 hit and suffix == cold\n", .{if (shared_ok) "PASS" else "FAIL"});
    if (!shared_ok) failures += 1;

    try checkStop();
    try cacheGateClobber(&agree_gate);
    try e.prefillRows(ids[0..alignment]);
    var resident_ok = try snap.load(alignment, ids[0..c2], key2) == c2;
    const tok4 = try prefillChecked(&e, ids[c2..n]);
    try g.logits(1, logits_got);
    const hash4 = try g.planesHash(n);
    resident_ok = try cacheGateSame(&agree_gate, tok4, hash4, logits_got, ref, logits_ref) and resident_ok;
    resident_ok = try e.agree(resident_ok);
    std.debug.print("{s} cachegate resident rows below restore: r={d}\n", .{if (resident_ok) "PASS" else "FAIL", alignment});
    if (!resident_ok) failures += 1;

    try checkStop();
    const body_len = layout.bodyLen(snap.ring_span, try kvsave.localRows(@intCast(o.rank), c1), try kvsave.localRows(@intCast(o.rank), c2));
    const body_offset: u64 = @intCast(kvsave.headerBytes(kvsave.digestCount(body_len)) + kvsave.aligned(4 * (c2 - c1)));
    const corrupt_path = try runtime.path(key2);
    defer g.gpa.free(corrupt_path);
    var corrupt_ok = true;
    if (o.rank == 1) corruptSaved(corrupt_path, body_offset) catch { corrupt_ok = false; };
    if (!try e.agree(corrupt_ok)) return error.CacheGateCorruptFailed;
    try cacheGateClobber(&agree_gate);
    const failed_at = try snap.load(0, ids[0..c2], key2);
    var fallback_ok = failed_at == 0;
    const tok5 = try prefillChecked(&e, ids[0..n]);
    try g.logits(1, logits_got);
    const hash5 = try g.planesHash(n);
    fallback_ok = try cacheGateSame(&agree_gate, tok5, hash5, logits_got, ref, logits_ref) and fallback_ok;
    fallback_ok = fallback_ok and runtime.index.get(key2) == null and runtime.index.get(key1) != null and !cacheGateFileExists(runtime, key2) and cacheGateFileExists(runtime, key1);
    fallback_ok = try e.agree(fallback_ok);
    std.debug.print("{s} cachegate fault fallback: corrupt C2 removed, C1 retained, cold output\n", .{if (fallback_ok) "PASS" else "FAIL"});
    if (!fallback_ok) failures += 1;
    if (failures == 0) std.debug.print("PASS cachegate: all cases\n", .{}) else std.debug.print("FAIL cachegate: {d} case(s) failed\n", .{failures});
    return failures == 0;
}

fn gpaCreateSnap(g: *Gpu, io: std.Io, runtime: *prefix_snap.prefix_cache.Runtime, e: *engine.Engine, device: prefix_snap.Device, ring: prefix_snap.RingHooks, layout: prefix_snap.Layout, slots: [10][]align(kvsave.alignment) u8, capacity: usize) !*prefix_snap.Snap(engine.Engine) {
    const snap = try g.gpa.create(prefix_snap.Snap(engine.Engine));
    errdefer g.gpa.destroy(snap);
    try snap.init(g.gpa, io, runtime, e, device, ring, layout, slots, capacity, 4);
    return snap;
}

const CacheGateResult = struct { tok: u32, hash: u64 };

fn gateRingView(ctx: *anyopaque) anyerror!prefix_snap.RingView {
    const view = try drafts.Host.ringView(ctx);
    return .{ .ptr = view.ptr, .len = view.len, .from = view.from };
}

fn cacheGateClobber(s: *SavedGate) !void {
    const local = blk: {
        s.clobber() catch |err| {
            std.debug.print("CACHEGATE rank {d}: clobber failed: {s}\n", .{ s.o.rank, @errorName(err) });
            break :blk false;
        };
        break :blk true;
    };
    if (!try s.e.agree(local)) return error.CacheGateClobberFailed;
}

fn cacheGateCold(s: *SavedGate, ids: []const u32, logits_out: []u8) !CacheGateResult {
    try cacheGateClobber(s);
    const tok = try prefillChecked(s.e, ids);
    try s.g.logits(1, logits_out);
    return .{ .tok = tok, .hash = try s.g.planesHash(ids.len) };
}

fn cacheGateSame(s: *SavedGate, tok: u32, hash: u64, logits: []const u8, expected: CacheGateResult, expected_logits: []const u8) !bool {
    const local = tok == expected.tok and hash == expected.hash and std.mem.eql(u8, logits, expected_logits);
    return s.e.agree(local);
}

fn cacheGateCut(runtime: *prefix_snap.prefix_cache.Runtime, at: usize) !engine.SnapCut {
    const header = try runtime.saveHeader(at);
    return .{ .at = at, .kind = 0, .key = header.key, .parent_key = header.parent_key, .parent_at = @intCast(header.base_at), .seq = header.seq };
}

fn cacheGateFileExists(runtime: *prefix_snap.prefix_cache.Runtime, key: u64) bool {
    const path = runtime.path(key) catch return false;
    defer runtime.gpa.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

fn resumeCheck(g: *Gpu, o: Opts) !bool {
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    if (o.split + o.reply >= ids.len) return error.Usage;
    var e: engine.Engine = .{ .gpa = g.gpa, .io = g.io, .f = g.forwardFor(o), .download = g.downloadFor(), .mem = try uma.MemTrace.fromEnv(o.rank) };
    e.mem.floor = try state.floorFromEnv();
    defer e.deinit();
    const a = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(a);
    const b = try g.gpa.alloc(u8, a.len);
    defer g.gpa.free(b);
    _ = try prefillChecked(&e, ids);
    try checkStop();
    try g.logits(1, a);
    const fresh = try g.planesHash(ids.len);
    try e.rollback(0);
    _ = try prefillChecked(&e, ids[0..o.split]);
    for (0..o.reply) |r| {
        try checkStop();
        var out: [1]u32 = undefined;
        try e.verify(ids[o.split + r ..][0..1], &out);
        try e.keep(1);
    }
    _ = try e.nextTurn(ids[o.split + o.reply ..]);
    try checkStop();
    try g.logits(1, b);
    const resumed = try g.planesHash(ids.len);
    const ok = std.mem.eql(u8, a, b) and fresh == resumed;
    std.debug.print("{s} resume == fresh split={d}: logits {s}, planes {x} vs {x}\n", .{ if (ok) "PASS" else "FAIL", o.split, if (std.mem.eql(u8, a, b)) "equal" else "DIFFER", fresh, resumed });
    return ok;
}

fn fineReuseCase(g: *Gpu, e: *engine.Engine, grain: usize, a_len: usize, shared: usize) !bool {
    try checkStop();
    const b_len = @max(a_len + 300, shared + 1);
    const a = try g.gpa.alloc(u32, a_len);
    defer g.gpa.free(a);
    for (a, 0..) |*id, i| id.* = @intCast((i * 17 + 3) % 32000 + 1);
    const b = try g.gpa.alloc(u32, b_len);
    defer g.gpa.free(b);
    @memcpy(b[0..a_len], a);
    for (b[shared..a_len], shared..) |*id, i| id.* = @intCast((i * 23 + 97) % 32000 + 1);
    b[shared] = a[shared] ^ 0xffff;
    for (b[a_len..], a_len..) |*id, i| id.* = @intCast((i * 31 + 211) % 32000 + 1);
    try e.rollback(0);
    _ = try prefillChecked(e, a);
    try e.settle();
    _ = try e.prefillResident(b);
    const reused = e.last_reused;
    const expected: usize = @min(a_len, shared / grain * grain);
    std.debug.assert(reused == expected);
    const resumed_hash = try g.planesHash(b.len);
    const resumed_logits = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(resumed_logits);
    try g.logits(1, resumed_logits);
    try e.rollback(0);
    _ = try prefillChecked(e, b);
    const fresh_hash = try g.planesHash(b.len);
    const fresh_logits = try g.gpa.alloc(u8, resumed_logits.len);
    defer g.gpa.free(fresh_logits);
    try g.logits(1, fresh_logits);
    const same = resumed_hash == fresh_hash and std.mem.eql(u8, resumed_logits, fresh_logits);
    std.debug.print("{s} glm53 fine reuse grain={d} a={d} shared={d} reused={d}\n", .{ if (same) "PASS" else "FAIL", grain, a_len, shared, reused });
    return same;
}

fn parkCheck(g: *Gpu, o: Opts) !bool {
    if (g.st.park_slots == 0) return error.NeedSideRows;
    const ids = try readIds(g.gpa, g.io, o.ids orelse return error.NeedIds);
    defer g.gpa.free(ids);
    const side = try readIds(g.gpa, g.io, o.side_ids orelse return error.NeedSideIds);
    defer g.gpa.free(side);
    if (o.split + o.reply >= ids.len) return error.Usage;
    const rows = 2 * g.st.park_slots;
    const predicted_f = o.split;
    const predicted_lo = engine.Engine.reusableRows(ids[0..predicted_f], predicted_f, side, 512);
    if (predicted_f < predicted_lo + 512) {
        std.debug.print("FAIL park: frontier {d}, reusable {d}, insufficient park range\n", .{ predicted_f, predicted_lo });
        return false;
    }
    const predicted_hi: usize = @min(predicted_f, predicted_lo + rows);
    const predicted_write_high = side.len + o.decode;
    const adapt_limit = if (g.st.adapt) |ad| ad.limit_position else 0;
    const adapt_floor: ?usize = if (adapt_limit != 0 and predicted_lo < adapt_limit) adapt_limit else null;
    const predicted_frontier = engine.side_park.effectiveFrontier(predicted_f, predicted_lo, predicted_hi, predicted_write_high, adapt_floor);
    var e: engine.Engine = .{ .gpa = g.gpa, .io = g.io, .f = g.forwardFor(o), .download = g.downloadFor(), .mem = try uma.MemTrace.fromEnv(o.rank) };
    e.mem.floor = try state.floorFromEnv();
    e.park = .{ .rows = rows, .adapt_limit = if (g.st.adapt) |ad| ad.limit_position else 0 };
    defer e.deinit();
    const logits_a = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(logits_a);
    const logits_b = try g.gpa.alloc(u8, logits_a.len);
    defer g.gpa.free(logits_b);

    try e.rollback(0);
    _ = try prefillChecked(&e, ids[0..o.split]);
    for (0..o.reply) |i| {
        var out: [1]u32 = undefined;
        try e.verify(ids[o.split + i ..][0..1], &out);
        try e.keep(1);
    }
    try e.settle();
    const f = e.frontier;
    const hash_a_frontier = try g.planesHash(predicted_frontier);
    _ = try e.prefillResident(ids);
    try g.logits(1, logits_a);
    const planes_a = try g.planesHash(ids.len);

    try e.rollback(0);
    _ = try prefillChecked(&e, ids[0..o.split]);
    for (0..o.reply) |i| {
        var out: [1]u32 = undefined;
        try e.verify(ids[o.split + i ..][0..1], &out);
        try e.keep(1);
    }
    try e.settle();
    const lo = e.reusable(side);
    if (f < lo + 512) {
        std.debug.print("FAIL park: frontier {d}, reusable {d}, insufficient park range\n", .{ f, lo });
        return false;
    }
    try e.parkSave(lo, @min(f, lo + rows));
    _ = try e.prefillResident(side);
    for (0..o.decode) |_| {
        try checkStop();
        var token: [1]u32 = undefined;
        try e.draw(g.st.ws.head_rows - 1, 1, e.pos - 1, &token);
        var out: [1]u32 = undefined;
        try e.verify(&token, &out);
        try e.keep(1);
    }
    try e.settle();
    const actual_adapt_floor: ?usize = if (e.park.?.adapt_hit) e.park.?.adapt_limit else null;
    const expected = engine.side_park.effectiveFrontier(f, lo, @min(f, lo + rows), e.park.?.write_high, actual_adapt_floor);
    const restored = try e.parkRestore();
    const hash_b_frontier = try g.planesHash(restored);
    _ = try e.prefillResident(ids);
    try g.logits(1, logits_b);
    const planes_b = try g.planesHash(ids.len);
    const logits_equal = std.mem.eql(u8, logits_a, logits_b);
    const ok = restored == expected and expected == predicted_frontier and hash_b_frontier == hash_a_frontier and (restored != f or (logits_equal and planes_a == planes_b));
    std.debug.print("{s} park: frontier {d} -> {d}, rows hash {x} vs {x}, logits {s}, planes {x} vs {x}\n", .{ if (ok) "PASS" else "FAIL", f, restored, hash_a_frontier, hash_b_frontier, if (logits_equal) "equal" else "DIFFER", planes_a, planes_b });
    return ok;
}

fn reuseCheck(g: *Gpu, o: Opts) !bool {
    const piece: usize = g.st.ws.piece_rows;
    if (piece != 512) return error.UnexpectedPieceRows;
    var e: engine.Engine = .{ .gpa = g.gpa, .f = g.forwardFor(o), .download = g.downloadFor() };
    defer e.deinit();
    var ok = true;
    const prefixes = [_]usize{ 511, 512, 700, 1536, 2560 };
    for (prefixes) |prefix| {
        try checkStop();
        const a = try g.gpa.alloc(u32, 3000);
        defer g.gpa.free(a);
        for (a, 0..) |*id, i| id.* = @intCast((i * 17 + 3) % 32000 + 1);
        const b = try g.gpa.alloc(u32, 3300);
        defer g.gpa.free(b);
        @memcpy(b[0..a.len], a);
        for (b[prefix..a.len], prefix..) |*id, i| id.* = @intCast((i * 23 + 97) % 32000 + 1);
        for (b[a.len..], a.len..) |*id, i| id.* = @intCast((i * 31 + 211) % 32000 + 1);
        try e.rollback(0);
        _ = try prefillChecked(&e, a);
        try e.settle();
        _ = try e.prefillResident(b);
        const resumed_hash = try g.planesHash(b.len);
        const resumed_logits = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
        defer g.gpa.free(resumed_logits);
        try g.logits(1, resumed_logits);
        try e.rollback(0);
        _ = try prefillChecked(&e, b);
        const fresh_hash = try g.planesHash(b.len);
        const fresh_logits = try g.gpa.alloc(u8, resumed_logits.len);
        defer g.gpa.free(fresh_logits);
        try g.logits(1, fresh_logits);
        const same = resumed_hash == fresh_hash and std.mem.eql(u8, resumed_logits, fresh_logits);
        std.debug.print("{s} glm53 reuse == fresh shared={d}, reused={d}\n", .{ if (same) "PASS" else "FAIL", prefix, e.last_reused });
        ok = same and ok;
    }

    if (o.fine_grain != 0) {
        e.reuse_grain = o.fine_grain;
        const cases = [_]struct { a_len: usize, shared: usize }{
            .{ .a_len = 3000, .shared = 2561 },
            .{ .a_len = 3000, .shared = 2999 },
            .{ .a_len = 2100, .shared = 2050 },
            .{ .a_len = 2100, .shared = 2000 },
            .{ .a_len = 4100, .shared = 4099 },
            .{ .a_len = 5000, .shared = 4609 },
        };
        for (cases) |item| ok = try fineReuseCase(g, &e, o.fine_grain, item.a_len, item.shared) and ok;
        if (o.fine_long) ok = try fineReuseCase(g, &e, o.fine_grain, 23046, 23000) and ok;
    }

    const append_ids = try g.gpa.alloc(u32, 5560);
    defer g.gpa.free(append_ids);
    for (append_ids, 0..) |*id, i| id.* = @intCast((i * 13 + 53) % 32000 + 1);
    try e.rollback(0);
    _ = try prefillChecked(&e, append_ids[0..2560]);
    try e.settle();
    try e.appendBegin(append_ids[0..2560]);
    const chunks = [_]usize{ 300, 900, 1400, 400 };
    var at: usize = 2560;
    for (chunks) |n| {
        try e.appendTokens(append_ids[at .. at + n]);
        at += n;
    }
    _ = try e.appendFinish();
    const append_hash = try g.planesHash(append_ids.len);
    const append_logits = try g.gpa.alloc(u8, g.w.vocab.len() * 4);
    defer g.gpa.free(append_logits);
    try g.logits(1, append_logits);
    try e.rollback(0);
    _ = try prefillChecked(&e, append_ids);
    const fresh_hash = try g.planesHash(append_ids.len);
    const fresh_logits = try g.gpa.alloc(u8, append_logits.len);
    defer g.gpa.free(fresh_logits);
    try g.logits(1, fresh_logits);
    const append_ok = append_hash == fresh_hash and std.mem.eql(u8, append_logits, fresh_logits);
    std.debug.print("{s} glm53 append chunks 300,900,1400,400 == fresh\n", .{if (append_ok) "PASS" else "FAIL"});
    ok = append_ok and ok;

    const resident = try g.gpa.alloc(u32, 126000);
    defer g.gpa.free(resident);
    for (resident, 0..) |*id, i| id.* = @intCast((i * 17 + 3) % 32000 + 1);
    const joined = try g.gpa.alloc(u32, 126850);
    defer g.gpa.free(joined);
    @memcpy(joined[0..resident.len], resident);
    for (joined[resident.len..], resident.len..) |*id, i| id.* = @intCast((i * 19 + 101) % 32000 + 1);
    try e.rollback(0);
    _ = try prefillChecked(&e, resident);
    try e.settle();
    const start = try monotonicNs();
    _ = try e.prefillResident(joined);
    try g.stream.synchronize();
    const elapsed = @as(f64, @floatFromInt(try monotonicNs() - start)) / std.time.ns_per_ms;
    if (o.rank == 0) std.debug.print("TTFT resident=126000 delta=850: {d:.2} ms\n", .{elapsed});
    return ok;
}

fn replay(g: *Gpu, o: Opts) !bool {
    var ce: cuda.graph_cache.CudaEngine = .{ .d = g.d };
    var cache = cuda.graph_cache.Cache(engine.GraphKey).init(g.gpa, ce.engine(), .{});
    defer cache.deinit();
    var e: engine.Engine = .{ .gpa = g.gpa, .f = g.forwardFor(o), .download = g.downloadFor(), .graphs = &cache };
    defer e.deinit();
    try e.reserveSteadyState();
    const max_position = try positionsMaximum(o.positions);
    const prefill_ids = try g.gpa.alloc(u32, max_position);
    defer g.gpa.free(prefill_ids);
    for (prefill_ids, 0..) |*id, i| id.* = @intCast(i % 8 + 1);
    if (prefill_ids.len == 0) return error.BadPositions;
    _ = try prefillChecked(&e, prefill_ids);
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var ok = true;
    var it = std.mem.tokenizeScalar(u8, o.positions, ',');
    while (it.next()) |tok| {
        try checkStop();
        const p = try std.fmt.parseInt(usize, tok, 10);
        const widths: []const usize = if (g.setup.verify_max == 16) &.{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 } else &.{ 1, 2, 3, 4, 5, 6, 7, 8 };
        for (widths) |rows| {
            try checkStop();
            e.pos = p;
            e.frontier = p;
            e.history.clearRetainingCapacity();
            try e.history.appendNTimes(g.gpa, 0, p);
            if (rows > g.st.ws.head_rows) continue;
            e.replayParity(ids[0..rows]) catch |err| {
                std.debug.print("FAIL replay at {d}, {d} rows: {t}\n", .{ p, rows, err });
                ok = false;
                continue;
            };
            std.debug.print("PASS replay at {d}, {d} rows (bucket {d})\n", .{ p, rows, forward.decodeKeyBucket(p + rows - 1) });
        }
    }
    return ok;
}

const usage =
    \\usage: tf-glm53-bringup plan|load|run|resume|reuse|park|replay|gate <ckpt> <rank> [--layers N] [--chunk-rows N] [--context C] [--idfile P]
    \\       [--ids F] [--decode N] [--prefetch] [--out DIR] [--fatbins DIR] [--split P] [--reply R]
    \\       reuse [--fine-grain N] [--fine-long --context 23300+]
    \\       park <ckpt> <rank> --ids F --side-ids PATH [--split P] [--reply R] [--decode N]
    \\       gate <ckpt> <rank> --probe P (--ids F [--window-id W] | --cases C) --out O
    \\       saved <ckpt> <rank> --ids F --kv-dir DIR --drafter DIR [--cuts 512,2048,...] (saved prefix KV: restore == fresh)
    \\       cachegate <ckpt> <rank> --ids F --kv-dir DIR --drafter DIR --expert-rows 1024 [--cuts C1,C2]  (prefix cache Runtime+Snap: save, fresh load, hit, fallback == cold)
    \\       diskprobe --kv-dir DIR (4 GiB O_DIRECT write/read; 1/4/16 MiB x 1/2/4/8 readers)
    \\       bench <ckpt> <rank> --ids F --points L,... --decode-steps N --append N --warmup N [--head-k K]
    \\       [--positions a,b,c] [--checksum]
    \\       load ... [--steady --ids F] [--drafter DIR [--drafter-streams N]]  (steady state + drafter admission)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const o = parse(args) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    if (std.mem.eql(u8, o.cmd, "diskprobe")) return if (try diskprobe(gpa, o.kv_dir.?)) 0 else 1;
    cuda.exl3_experts.l2_depth = parseL2Depth(if (std.c.getenv("TF_EXL3X_L2D")) |v| std.mem.span(v) else null) catch {
        std.debug.print("TF_EXL3X_L2D must be an integer from 0 through 8\n", .{});
        return 2;
    };
    if (std.mem.eql(u8, o.cmd, "plan")) {
        var s = try Setup.init(gpa, io, o);
        defer s.deinit(gpa);
        s.report();
        _ = admit(gpa, io, &s, o) catch |e| {
            std.debug.print("ADMIT refused: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    installSignals();
    const process_start_ns = realtimeNs() catch {
        std.debug.print("NCCL id handoff refused: cannot read process start clock\n", .{});
        return 1;
    };
    if (!(preflight(gpa, io) catch |e| {
        std.debug.print("PREFLIGHT refused: cannot read /proc/meminfo ({t})\n", .{e});
        return 1;
    })) return 1;
    try checkStop();
    if (o.rank == 0) if (o.idfile) |path| {
        const ready_path = try std.fmt.allocPrint(gpa, "{s}.ready", .{path});
        defer gpa.free(ready_path);
        std.Io.Dir.cwd().deleteFile(io, path) catch |e| if (e != error.FileNotFound) return e;
        std.Io.Dir.cwd().deleteFile(io, ready_path) catch |e| if (e != error.FileNotFound) return e;
        const accepted_path = try std.fmt.allocPrint(gpa, "{s}.accepted", .{path});
        defer gpa.free(accepted_path);
        std.Io.Dir.cwd().deleteFile(io, accepted_path) catch |e| if (e != error.FileNotFound) return e;
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    _ = mem(gpa, io, "after CUDA context");
    const g = try Gpu.load(gpa, io, &driver, ctx.device, o);
    var completed = false;
    defer g.deinit(!completed or signalRequested());
    try checkStop();
    if (std.mem.eql(u8, o.cmd, "load")) {
        if (o.checksum) {
            var total = std.hash.Wyhash.init(0);
            for (g.w.tensors.items) |t| {
                total.update(std.mem.asBytes(&(try wyhash(gpa, t.data))));
                if (t.scales) |sc| total.update(std.mem.asBytes(&(try wyhash(gpa, sc))));
            }
            std.debug.print("CHECKSUM weights {x} over {d} tensors (TF_GLM53_ODIRECT={s})\n", .{ total.final(), g.w.tensors.items.len, if (uma.odirectFromEnv()) "1" else "0" });
        }
        if (!o.steady) {
            completed = true;
            return 0;
        }
    }
    var watcher: ?std.Thread = null;
    watchdog_done.store(false, .release);
    watcher = try std.Thread.spawn(.{}, stopWatchdog, .{g});
    defer {
        watchdog_done.store(true, .release);
        if (watcher) |thread| thread.join();
    }
    try g.kernels(&ctx, o, process_start_ns);
    try g.forwardFor(o).validateSetup(); // after kernels(): it reads g.k (W8A8, SMLA-P kernels)
    try checkStop();
    if (std.mem.eql(u8, o.cmd, "load")) {
        try steadyLoad(g, &ctx, o);
        try checkStop();
        completed = true;
        return 0;
    }
    if (std.mem.eql(u8, o.cmd, "run")) {
        try run(g, o);
        try checkStop();
        completed = true;
        return 0;
    }
    if (std.mem.eql(u8, o.cmd, "bench")) {
        try bench(g, o);
        try checkStop();
        completed = true;
        return 0;
    }
    if (std.mem.eql(u8, o.cmd, "gate")) {
        const ok = try gate(g, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "resume")) {
        const ok = if (o.reuse) try reuseCheck(g, o) else try resumeCheck(g, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "reuse")) {
        const ok = try reuseCheck(g, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "park")) {
        const ok = try parkCheck(g, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "saved")) {
        const ok = try savedCheck(g, &ctx, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "cachegate")) {
        const ok = try cacheGate(g, &ctx, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, o.cmd, "replay")) {
        const ok = try replay(g, o);
        try checkStop();
        completed = ok;
        return if (ok) 0 else 1;
    }
    std.debug.print("{s}", .{usage});
    return 2;
}

test "verify max defaults to 8, accepts 16, and sets decode and head geometry" {
    const Empty = struct { fn get(_: []const u8) ?[]const u8 { return null; } };
    const Wide = struct { fn get(_: []const u8) ?[]const u8 { return "16"; } };
    try std.testing.expectEqual(@as(usize, 8), try verifyMaxFromEnv(Empty.get));
    try std.testing.expectEqual(@as(usize, 16), try verifyMaxFromEnv(Wide.get));
    var geometry: state.Geometry = .{};
    setVerifyGeometryRows(&geometry, try verifyMaxFromEnv(Wide.get));
    try std.testing.expectEqual(@as(usize, 16), geometry.decode_rows);
    try std.testing.expectEqual(@as(usize, 16), geometry.head_rows);
    setVerifyGeometryRows(&geometry, 8);
    try std.testing.expectEqual(@as(usize, 8), geometry.decode_rows);
    try std.testing.expectEqual(@as(usize, 8), geometry.head_rows);
    const Bad = struct { fn get(_: []const u8) ?[]const u8 { return "12"; } };
    try std.testing.expectError(error.BadVerifyMax, verifyMaxFromEnv(Bad.get));
}

test "gate case list parsing" {
    const cases = try parseCases(std.testing.allocator, "# c\n0 a.txt\n\n3\tb/c.txt\n");
    defer std.testing.allocator.free(cases);
    try std.testing.expectEqual(@as(usize, 2), cases.len);
    try std.testing.expectEqual(@as(i32, 0), cases[0].window_id);
    try std.testing.expectEqualStrings("a.txt", cases[0].ids_path);
    try std.testing.expectEqual(@as(i32, 3), cases[1].window_id);
    try std.testing.expectEqualStrings("b/c.txt", cases[1].ids_path);
    inline for (.{ "0 a b\n", "x a\n", "1 a\n1 b\n", "# only\n" }) |bad| try std.testing.expectError(error.BadCases, parseCases(std.testing.allocator, bad));
}

test "saved arguments require ids, a kv dir and a 4-bit drafter; cuts must be 512-aligned" {
    const good = [_][:0]const u8{ "tool", "saved", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--drafter", "d" };
    const o = try parse(&good);
    try std.testing.expectEqualStrings("/kv", o.kv_dir.?);
    const no_dir = [_][:0]const u8{ "tool", "saved", "ck", "0", "--ids", "f", "--drafter", "d" };
    try std.testing.expectError(error.Usage, parse(&no_dir));
    const no_drafter = [_][:0]const u8{ "tool", "saved", "ck", "0", "--ids", "f", "--kv-dir", "/kv" };
    try std.testing.expectError(error.Usage, parse(&no_drafter));
    const cuts = try parseCuts(std.testing.allocator, "512,2560");
    defer std.testing.allocator.free(cuts);
    try std.testing.expectEqualSlices(usize, &.{ 512, 2560 }, cuts);
    try std.testing.expectError(error.Usage, parseCuts(std.testing.allocator, "700"));
    const default = try parseCuts(std.testing.allocator, null);
    defer std.testing.allocator.free(default);
    try std.testing.expectEqualSlices(usize, &SavedGate.default_cuts, default);
}

test "cachegate arguments require ids, a kv dir and a 4-bit drafter; cuts must be aligned and increasing" {
    const good = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--drafter", "d", "--expert-rows", "1024", "--cuts", "2048,4096" };
    const parsed = try parse(&good);
    try std.testing.expectEqualStrings("/kv", parsed.kv_dir.?);
    const no_ids = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--kv-dir", "/kv", "--drafter", "d", "--expert-rows", "1024" };
    try std.testing.expectError(error.Usage, parse(&no_ids));
    const no_dir = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--drafter", "d", "--expert-rows", "1024" };
    try std.testing.expectError(error.Usage, parse(&no_dir));
    const no_drafter = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--expert-rows", "1024" };
    try std.testing.expectError(error.Usage, parse(&no_drafter));
    const bad_drafter = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--drafter", "d", "--drafter-bits", "8", "--expert-rows", "1024" };
    try std.testing.expectError(error.Usage, parse(&bad_drafter));
    const unaligned = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--drafter", "d", "--expert-rows", "1024", "--cuts", "2048,4608" };
    try std.testing.expectError(error.Usage, parse(&unaligned));
    const decreasing = [_][:0]const u8{ "tool", "cachegate", "ck", "0", "--ids", "f", "--kv-dir", "/kv", "--drafter", "d", "--expert-rows", "1024", "--cuts", "4096,2048" };
    try std.testing.expectError(error.Usage, parse(&decreasing));
}

test "gate argument source validation" {
    const good = [_][:0]const u8{ "tool", "gate", "ck", "0", "--probe", "p", "--cases", "c", "--out", "o" };
    _ = try parse(&good);
    const both = [_][:0]const u8{ "tool", "gate", "ck", "0", "--probe", "p", "--ids", "f", "--cases", "c", "--out", "o" };
    try std.testing.expectError(error.Usage, parse(&both));
    const neither = [_][:0]const u8{ "tool", "gate", "ck", "0", "--probe", "p", "--out", "o" };
    try std.testing.expectError(error.Usage, parse(&neither));
}

test "bench helpers and argument validation" {
    const points = try parsePoints(std.testing.allocator, "2048,8192");
    defer std.testing.allocator.free(points);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 8192 }, points);
    inline for (.{ "", "0", "8192,2048", "a" }) |bad| try std.testing.expectError(error.BadPoints, parsePoints(std.testing.allocator, bad));
    const sorted = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expectEqual(@as(f64, 5), percentile(&sorted, 0.5));
    try std.testing.expectEqual(@as(f64, 9), percentile(&sorted, 0.9));
    try std.testing.expectEqual(@as(f64, 10), percentile(&sorted, 1));
    try std.testing.expectEqual(@as(f64, 3), percentile(&.{3}, 0.5));
    const ids = try cycledIds(std.testing.allocator, &.{ 1, 2, 3 }, 7);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 1, 2, 3, 1 }, ids);
    try std.testing.expectError(error.EmptyPrompt, cycledIds(std.testing.allocator, &.{}, 2));
    const good = [_][:0]const u8{ "tool", "bench", "ck", "1", "--ids", "f", "--points", "2048,4096", "--decode-steps", "8", "--append", "0", "--warmup", "0", "--head-k", "256" };
    try std.testing.expectEqual(@as(?usize, 256), (try parse(&good)).head_k);
    const good_rows = [_][:0]const u8{ "tool", "bench", "ck", "1", "--ids", "f", "--rows", "8" };
    try std.testing.expectEqual(@as(usize, 8), (try parse(&good_rows)).rows);
    const bad_rows = [_][:0]const u8{ "tool", "bench", "ck", "1", "--ids", "f", "--rows", "3" };
    try std.testing.expectError(error.Usage, parse(&bad_rows));
    const run_rows = [_][:0]const u8{ "tool", "run", "ck", "1", "--ids", "f", "--rows", "4" };
    try std.testing.expectError(error.Usage, parse(&run_rows));
    const bad_head_k = [_][:0]const u8{ "tool", "bench", "ck", "1", "--ids", "f", "--head-k", "2049" };
    try std.testing.expectError(error.Usage, parse(&bad_head_k));
    const head_k_outside_bench = [_][:0]const u8{ "tool", "run", "ck", "1", "--ids", "f", "--head-k", "256" };
    try std.testing.expectError(error.Usage, parse(&head_k_outside_bench));
    const missing = [_][:0]const u8{ "tool", "bench", "ck", "1" };
    try std.testing.expectError(error.Usage, parse(&missing));
    const experts = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "128,256,512,1024" };
    const er = try parse(&experts);
    try std.testing.expectEqualSlices(usize, &.{ 128, 256, 512, 1024 }, er.expert_rows_values[0..er.expert_rows_count]);
    const phase1 = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "1536" };
    const phase1_option = try parse(&phase1);
    try std.testing.expectEqual(@as(usize, 1536), phase1_option.expert_rows);
    const phase1_max = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "2048" };
    const phase1_max_option = try parse(&phase1_max);
    try std.testing.expectEqual(@as(usize, 2048), phase1_max_option.expert_rows);
    const er4k_args = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "4096" };
    try std.testing.expectEqual(@as(usize, 4096), (try parse(&er4k_args)).expert_rows);
    const er8k_args = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "8192" };
    try std.testing.expectEqual(@as(usize, 8192), (try parse(&er8k_args)).expert_rows);
    const unsupported_expert_rows = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "768" };
    try std.testing.expectError(error.Usage, parse(&unsupported_expert_rows));
    const single = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--expert-rows", "512" };
    const one = try parse(&single);
    try std.testing.expectEqual(@as(usize, 512), one.expert_rows);
    try std.testing.expectEqual(@as(usize, 1), one.expert_rows_count);
    const chunk_rows = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--chunk-rows", "2048" };
    const chunk_option = try parse(&chunk_rows);
    try std.testing.expectEqual(@as(usize, 2048), chunk_option.chunk_rows);
    const chunk_rows_8k = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--chunk-rows", "8192" };
    try std.testing.expectEqual(@as(usize, 8192), (try parse(&chunk_rows_8k)).chunk_rows);
    const bad_chunk_rows = [_][:0]const u8{ "tool", "bench", "ckpt", "0", "--ids", "ids.txt", "--chunk-rows", "1000" };
    try std.testing.expectError(error.Usage, parse(&bad_chunk_rows));
}

test "fine reuse bring-up arguments validate grain and long context" {
    const fine = [_][:0]const u8{ "tool", "reuse", "ck", "0", "--fine-grain", "64" };
    const fine_opts = try parse(&fine);
    try std.testing.expectEqual(@as(usize, 64), fine_opts.fine_grain);
    const bad_grain = [_][:0]const u8{ "tool", "reuse", "ck", "0", "--fine-grain", "512" };
    try std.testing.expectError(error.Usage, parse(&bad_grain));
    const fine_long = [_][:0]const u8{ "tool", "reuse", "ck", "0", "--fine-grain", "1", "--fine-long", "--context", "23300" };
    const long_opts = try parse(&fine_long);
    try std.testing.expect(long_opts.fine_long and long_opts.context == 23300);
    const missing_context = [_][:0]const u8{ "tool", "reuse", "ck", "0", "--fine-long" };
    try std.testing.expectError(error.Usage, parse(&missing_context));
    const small_context = [_][:0]const u8{ "tool", "reuse", "ck", "0", "--fine-long", "--context", "23299" };
    try std.testing.expectError(error.Usage, parse(&small_context));
}

test "bring-up host contracts and full declaration analysis" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Opts);
    std.testing.refAllDecls(Setup);
    std.testing.refAllDecls(Gpu);
    const good: uma.MemInfo = .{ .total = 128 * 1024 * 1024 * 1024, .available = preflight_available, .free = preflight_free };
    try std.testing.expect(preflightOk(good));
    var low_available = good;
    low_available.available -= 1;
    try std.testing.expect(!preflightOk(low_available));
    var low_free = good;
    low_free.free -= 1;
    try std.testing.expect(!preflightOk(low_free));
    try std.testing.expectEqual(@as(usize, 32767), try positionsMaximum("2047,4095,32767"));
    try std.testing.expect(!idFileFresh(1000, 1000));
    try std.testing.expect(!idFileFresh(999, 1000));
    try std.testing.expect(!idFileFresh(1001, 0));
    try std.testing.expect(readyNonceMatches("abc123", "abc123"));
    try std.testing.expect(!readyNonceMatches("oldnonce", "abc123"));

    var writer_id: cuda.nccl.UniqueId = undefined;
    for (&writer_id.internal, 0..) |*byte, i| byte.* = @truncate(i *% 17 +% 3);
    const writer_nonce: [16]u8 = "glm53-rank0-nonce"[0..16].*;
    const rank0_file = IdFile.encode(writer_id, writer_nonce);
    const rank1_file = try IdFile.decode(&rank0_file);
    try std.testing.expectEqual(writer_id, rank1_file[0]);
    try std.testing.expectEqual(writer_nonce, rank1_file[1]);
    try std.testing.expectEqual(@as(usize, 156), IdFile.size);
    var bad_id_file = rank0_file;
    bad_id_file[IdFile.version_start] = 2;
    try std.testing.expectError(error.BadIdFile, IdFile.decode(&bad_id_file));

    stop_requested.store(false, .release);
    onSignal(.INT);
    try std.testing.expect(signalRequested());
    stop_requested.store(false, .release);

    const replay_args = [_][:0]const u8{ "tf-glm53-bringup", "replay", "ckpt", "0", "--positions", "2047,4095,32767" };
    const replay_opts = try parse(&replay_args);
    try std.testing.expectEqual(@as(usize, 32776), replay_opts.context);
    const too_small_args = [_][:0]const u8{ "tf-glm53-bringup", "replay", "ckpt", "0", "--positions", "32767", "--context", "32775" };
    try std.testing.expectError(error.Usage, parse(&too_small_args));

    var gate_row: dump.GateRow = .{ .ids = @splat(7), .logp = @splat(-1), .lse = 3, .mass_outside = 0.5, .argmax = 7, .window_id = 4, .pos = 9 };
    gate_row.ids[63] = 63;
    const tensors = try dump.gateTensors(std.testing.allocator, &.{gate_row});
    defer dump.freeGateTensors(std.testing.allocator, tensors);
    const encoded = try dump.encode(std.testing.allocator, tensors);
    defer std.testing.allocator.free(encoded);
    const header_len = std.mem.readInt(u64, encoded[0..8], .little);
    var header_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer header_arena.deinit();
    const entries = try core.safetensors.parseHeader(header_arena.allocator(), encoded[8..][0..@intCast(header_len)], encoded.len - 8 - @as(usize, @intCast(header_len)));
    const schema = .{
        .{ "ids", core.safetensors.DType.i32, 2 },
        .{ "logp", core.safetensors.DType.f32, 2 },
        .{ "lse", core.safetensors.DType.f32, 1 },
        .{ "mass_outside", core.safetensors.DType.f32, 1 },
        .{ "argmax", core.safetensors.DType.i32, 1 },
        .{ "window_id", core.safetensors.DType.i32, 1 },
        .{ "pos", core.safetensors.DType.i64, 1 },
    };
    inline for (schema) |field| {
        const entry = entries.get(field[0]).?;
        try std.testing.expectEqual(field[1], entry.dtype);
        try std.testing.expectEqual(field[2], entry.rank);
        try std.testing.expectEqual(@as(usize, 1), entry.dim(0));
        if (field[2] == 2) try std.testing.expectEqual(@as(usize, 64), entry.dim(1));
    }
    const sidecar = try gateSidecar(std.testing.allocator);
    defer std.testing.allocator.free(sidecar);
    const meta = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, sidecar, .{});
    defer meta.deinit();
    try std.testing.expectEqual(true, meta.value.object.get("fp32_moe_sum").?.bool);
}

test "load steady and drafter argument validation" {
    const steady = [_][:0]const u8{ "tool", "load", "ck", "0", "--context", "262144", "--steady", "--ids", "f" };
    try std.testing.expect((try parse(&steady)).steady);
    const drafter = [_][:0]const u8{ "tool", "load", "ck", "1", "--drafter", "/m", "--drafter-bits", "4", "--drafter-streams", "4", "--ids", "f" };
    const o = try parse(&drafter);
    try std.testing.expect(o.steady and o.drafter_streams == 4 and o.drafter_bits == 4);
    const bad_bits = [_][:0]const u8{ "tool", "load", "ck", "0", "--drafter", "/m", "--drafter-bits", "3", "--ids", "f" };
    try std.testing.expectError(error.Usage, parse(&bad_bits));
    const bad_streams = [_][:0]const u8{ "tool", "load", "ck", "0", "--drafter", "/m", "--drafter-streams", "0", "--ids", "f" };
    try std.testing.expectError(error.Usage, parse(&bad_streams));
    const no_ids = [_][:0]const u8{ "tool", "load", "ck", "0", "--steady" };
    try std.testing.expectError(error.Usage, parse(&no_ids));
}

test "bring-up RoCE opt-in environment contract" {
    try std.testing.expect(!roceRequestedFrom(null, null));
    try std.testing.expect(!roceRequestedFrom("0", null));
    try std.testing.expect(roceRequestedFrom("1", null));
    try std.testing.expect(roceRequestedFrom(null, "roce"));
    try std.testing.expect(!roceRequestedFrom(null, "nccl"));
}

test "bring-up decode union switch contract" {
    try std.testing.expectEqual(false, try decodeUnionSetup(false, false));
    try std.testing.expectEqual(true, try decodeUnionSetup(true, false));
    try std.testing.expectEqual(false, try decodeUnionSetup(false, true));
    try std.testing.expectError(error.DecodeUnionWithMegakernel, decodeUnionSetup(true, true));
    try std.testing.expect(!forward.decodeUnionSelected(null));
    try std.testing.expect(forward.decodeUnionSelected("1"));
    try std.testing.expect(!forward.decodeUnionSelected("0"));
    try std.testing.expect(!forward.decodeUnionSelected("true"));
}

test "bring-up decode levers mirror native refusals and rank bits (B0)" {
    const off = try decodeLevers(false, false, false, false, false, .off, false, false, 8, true);
    try std.testing.expect(!off.route_compact and !off.own_attend and !off.q_one_gather and !off.partial_roce);
    try std.testing.expectEqual(@as(usize, 8), off.decode_union_rows);
    try std.testing.expectEqual(@as(u8, 0), decodeSwitchBits(off));
    // DECODE_UNION16 is not read with the attend switches off (rows stay 8, no refusal), as before B0.
    try std.testing.expectEqual(@as(usize, 8), (try decodeLevers(false, false, false, false, false, .off, false, true, 16, true)).decode_union_rows);
    try std.testing.expectError(error.QOneGatherNeedsDecodeMla3, decodeLevers(false, false, true, false, false, .v2, true, false, 8, true));
    try std.testing.expectError(error.AttendGatherSwitchNeedsRoce, decodeLevers(false, true, false, false, false, .off, true, false, 8, false));
    try std.testing.expectError(error.AttendGatherSwitchNeedsRoce, decodeLevers(false, false, true, false, false, .v3, true, false, 8, false));
    try std.testing.expectError(error.QOneGatherWithCollFuse, decodeLevers(false, false, true, false, true, .v3, true, false, 8, true));
    try std.testing.expectError(error.QOneGatherNeedsDecodeUnionAllRows, decodeLevers(false, false, true, false, false, .v3, false, false, 8, true));
    try std.testing.expectError(error.QOneGatherNeedsDecodeUnionAllRows, decodeLevers(false, false, true, false, false, .v3, true, false, 16, true));
    try std.testing.expectError(error.DecodeUnion16WithoutDecodeUnion, decodeLevers(false, true, false, false, false, .off, false, true, 16, true));
    const wide = try decodeLevers(false, false, true, false, false, .v3, true, true, 16, true);
    try std.testing.expectEqual(@as(usize, 16), wide.decode_union_rows);
    const all = try decodeLevers(true, true, false, true, false, .off, true, false, 8, true);
    try std.testing.expect(all.route_compact and all.own_attend and all.partial_roce);
    try std.testing.expectEqual(@as(u8, 0b1010), decodeSwitchBits(all));
    try std.testing.expectEqual(@as(u8, 0b0100), decodeSwitchBits(wide));
}
