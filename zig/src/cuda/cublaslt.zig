// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! cuBLASLt opened at run time: descriptors, layouts, heuristics and matmul, with fp32 compute only (no TF32 or fast modes).

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const MatmulDesc = ?*opaque {};
pub const Layout = ?*opaque {};
pub const Preference = ?*opaque {};

pub const Algo = extern struct { data: [8]u64 };

pub const Heuristic = extern struct {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14, e4m3 = 28 };
pub const compute_32f: c_int = 68;
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const Order = enum(c_int) { col = 0, row = 1 };
pub const Fp8TnLayouts = struct { transa: Op, transb: Op, order: Order, scale_a: c_int, scale_b: c_int };

pub fn fp8TnLayouts() Fp8TnLayouts {
    return .{ .transa = .t, .transb = .n, .order = .col, .scale_a = scale_blk128x128_32f, .scale_b = scale_vec128_32f };
}

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
pub const desc_epilogue: c_int = 7;
pub const desc_bias_pointer: c_int = 8;
pub const layout_order: c_int = 1;
pub const layout_rows: c_int = 2;
pub const layout_cols: c_int = 3;
pub const pref_max_workspace_bytes: c_int = 1;
// CUDA 13.0 cublasLt.h: scale-pointer attributes and FP8 block-scale modes.
pub const desc_a_scale_pointer: c_int = 17;
pub const desc_b_scale_pointer: c_int = 18;
pub const desc_a_scale_mode: c_int = 31;
pub const desc_b_scale_mode: c_int = 32;
pub const scale_vec128_32f: c_int = 4;
pub const scale_blk128x128_32f: c_int = 5;

/// What D gets after the product: nothing, or a bias over its columns added in f32 before D's one rounding.
pub const Epilogue = enum(u32) { none = 1, bias = 4 };

pub const Error = platform.optional_libraries.Failure || error{ LibraryUnavailable, MissingSymbol, CublasFailed, NoAlgorithm, Invalid };

const S = Status;
const CPtr = ?*const anyopaque;

/// Device pointers are declared as u64: the same register class as `void*` on every LP64 target.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Preference) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Preference) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Preference, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, Preference, c_int, [*]Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmulAlgoCheck: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, *const Algo, *Heuristic) callconv(.c) S,
    cublasLtMatmulAlgoConfigSetAttribute: *const fn (*Algo, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, MatmulDesc, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, ?*const Algo, abi.DevicePtr, usize, abi.Stream) callconv(.c) S,
};

pub const algo_config_splitk_num: c_int = 2;
pub const algo_config_reduction_scheme: c_int = 3;
pub const pref_reduction_scheme_mask: c_int = 3;
pub const reduction_none: c_int = 0;

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    /// The CUDA 13 soname first, then the unversioned link name; on Windows the toolkit's DLL.
    pub fn open() Error!Library {
        const names: []const []const u8 = if (@import("builtin").os.tag == .windows) &.{"cublasLt64_13.dll"} else &.{ "libcublasLt.so.13", "libcublasLt.so" };
        for (names) |path| {
            return openCandidate(path) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                error.MissingSymbol => return platform.optional_libraries.unavailable(.cublaslt),
                else => return e,
            };
        }
        return platform.optional_libraries.unavailable(.cublaslt);
    }

    pub fn openPath(path: []const u8) Error!Library {
        return openCandidate(path) catch |e| switch (e) {
            error.LibraryUnavailable, error.MissingSymbol => platform.optional_libraries.unavailable(.cublaslt),
            else => e,
        };
    }

    fn openCandidate(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        const n = self.api.cublasLtGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (n) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }
};

/// One row-major bf16 product D[m,n] = X[m,k] . W[n,k]^T (a torch Linear), fp32 accumulate, bf16 or fp32 out.
pub const Linear = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,
    epilogue: Epilogue, // fused into `algo`: `initEpilogue` asks, the algorithm decides
    plan_m: u64,
    layout_m: u64,

    /// Picks the heuristic's first algorithm for this shape once; `workspace_limit` bounds what it may ask for.
    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize) Error!Linear {
        return initEpilogue(lt, m, n, k, out, workspace_limit, .none);
    }

    /// `init`'s plan, its algorithm also running `epilogue` when it supports it (`epilogue` says if); see `runBias`.
    pub fn initEpilogue(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize, epilogue: Epilogue) Error!Linear {
        var self: Linear = undefined;
        self.lt = lt;
        self.plan_m = m;
        self.layout_m = m;
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(self.handle);
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = lt.api.cublasLtMatmulDescDestroy(self.desc);
        const ta: c_int = @backingInt(Op.t);
        const tb: c_int = @backingInt(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // column-major view: D^T[n,m] = W[n,k] (stored k x n, transposed) . X^T[k,m]
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, .bf16, k, n, @intCast(k)), "layout W");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.a);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, .bf16, k, m, @intCast(k)), "layout X");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.b);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.c);
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate");
        errdefer _ = lt.api.cublasLtMatmulPreferenceDestroy(self.pref);
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        // A short, deterministic candidate list is queried with the caller's fixed plan shape. Higher-level
        // shape-stable paths can reuse the selected algorithm for other row counts via runRows.
        var found: [4]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, found.len, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1) return error.NoAlgorithm;
        var chosen: ?Heuristic = null;
        for (found[0..@min(found.len, @as(usize, @intCast(count)))]) |candidate| if (candidate.state == 0 and candidate.workspace_size <= workspace_limit) {
            chosen = candidate;
            break;
        };
        const selected = chosen orelse return error.NoAlgorithm;
        self.algo = selected.algo;
        self.workspace_need = selected.workspace_size;
        self.epilogue = .none;
        if (epilogue != .none) try self.fuse(epilogue, workspace_limit);
        return self;
    }

    pub fn deinit(self: *Linear) void {
        const api = self.lt.api;
        _ = api.cublasLtMatmulPreferenceDestroy(self.pref);
        _ = api.cublasLtMatrixLayoutDestroy(self.c);
        _ = api.cublasLtMatrixLayoutDestroy(self.b);
        _ = api.cublasLtMatrixLayoutDestroy(self.a);
        _ = api.cublasLtMatmulDescDestroy(self.desc);
        _ = api.cublasLtDestroy(self.handle);
        self.* = undefined;
    }

    /// D = X . W^T on `stream`; the workspace must hold `workspace_need` bytes.
    pub fn run(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (workspace_len < self.workspace_need) return error.Invalid;
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }

    /// Reuse this plan's N/K-selected algorithm with another row count. The algorithm is selected once at init;
    /// only the activation/output layout extents change, so chunk rows do not select a different kernel algorithm.
    pub fn runRows(self: *Linear, rows: u64, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (rows == 0 or rows > 8192) return error.Invalid;
        if (rows != self.layout_m) {
            const cols = rows;
            try self.lt.check(self.lt.api.cublasLtMatrixLayoutSetAttribute(self.b, layout_cols, &cols, @sizeOf(u64)), "set activation rows");
            try self.lt.check(self.lt.api.cublasLtMatrixLayoutSetAttribute(self.c, layout_cols, &cols, @sizeOf(u64)), "set output rows");
            self.layout_m = rows;
        }
        try self.run(x, w, d, workspace, workspace_len, stream);
    }

    /// D = X . W^T + bias (bias [n], D's type) on a plan that fused `.bias`, one rounding to D.
    pub fn runBias(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, bias: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (self.epilogue != .bias) return error.Invalid;
        try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_bias_pointer, &bias, @sizeOf(abi.DevicePtr)), "set bias");
        try self.run(x, w, d, workspace, workspace_len, stream);
    }

    /// Fuses `epilogue` into the chosen algorithm when it supports it within `workspace_limit`; else leaves it unfused.
    fn fuse(self: *Linear, epilogue: Epilogue, workspace_limit: usize) Error!void {
        const api = self.lt.api;
        const e: u32 = @backingInt(epilogue);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &e, @sizeOf(u32)), "set epilogue");
        var checked: Heuristic = undefined;
        if (api.cublasLtMatmulAlgoCheck(self.handle, self.desc, self.a, self.b, self.c, self.c, &self.algo, &checked) == 0 and checked.state == 0 and checked.workspace_size <= workspace_limit) {
            self.epilogue = epilogue;
            self.workspace_need = @max(self.workspace_need, checked.workspace_size);
            return;
        }
        const none: u32 = @backingInt(Epilogue.none);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &none, @sizeOf(u32)), "unset epilogue");
    }
};

/// FP8 block-scaled GEMM X[M,K] * W[N,K]^T with a vec128 activation scale and
/// BLK128x128 weight scales. The selected algorithm is fixed at init and forbids split-K.
pub const Fp8BlockLinear = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,
    m: u64,
    n: u64,
    k: u64,

    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, workspace_limit: usize) Error!Fp8BlockLinear {
        if (m == 0 or n == 0 or k == 0 or m % 4 != 0 or n % 4 != 0 or k % 128 != 0) return error.Invalid;
        var self: Fp8BlockLinear = .{ .lt = lt, .handle = null, .desc = null, .a = null, .b = null, .c = null, .pref = null, .algo = undefined, .workspace_need = 0, .m = m, .n = n, .k = k };
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate FP8");
        errdefer _ = lt.api.cublasLtDestroy(self.handle);
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate FP8");
        errdefer _ = lt.api.cublasLtMatmulDescDestroy(self.desc);
        // Canonical CUDA FP8 TN mapping: row-major W[N,K] bytes are column-major
        // A[K,N], row-major X[M,K] bytes are column-major B[K,M], and column-
        // major D[N,M] bytes are TensorFold's row-major output [M,N].
        const mapping = fp8TnLayouts();
        const ta: c_int = @backingInt(mapping.transa);
        const tb: c_int = @backingInt(mapping.transb);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set FP8 transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set FP8 transb");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_a_scale_mode, &mapping.scale_a, @sizeOf(c_int)), "set FP8 A scale mode");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_b_scale_mode, &mapping.scale_b, @sizeOf(c_int)), "set FP8 B scale mode");
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, .e4m3, k, n, @intCast(k)), "layout FP8 W as A[K,N]");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.a);
        try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(self.a, layout_order, &mapping.order, @sizeOf(c_int)), "order FP8 A");
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, .e4m3, k, m, @intCast(k)), "layout FP8 X as B[K,M]");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.b);
        try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(self.b, layout_order, &mapping.order, @sizeOf(c_int)), "order FP8 B");
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, .bf16, n, m, @intCast(n)), "layout FP8 D as column-major [N,M]");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.c);
        try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(self.c, layout_order, &mapping.order, @sizeOf(c_int)), "order FP8 D");
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate FP8");
        errdefer _ = lt.api.cublasLtMatmulPreferenceDestroy(self.pref);
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set FP8 workspace");
        const allowed_reductions: u32 = @intCast(reduction_none);
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_reduction_scheme_mask, &allowed_reductions, @sizeOf(u32)), "disable FP8 reduction");
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic FP8 TN column-major");
        std.debug.print("cuBLASLt version={d}, FP8 TN column-major heuristic count={d}, status={d}\n", .{ lt.api.cublasLtGetVersion(), count, if (count > 0) found[0].state else -1 });
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        self.algo = found[0].algo;
        const one: c_int = 1;
        try lt.check(lt.api.cublasLtMatmulAlgoConfigSetAttribute(&self.algo, algo_config_splitk_num, &one, @sizeOf(c_int)), "disable FP8 split-K");
        try lt.check(lt.api.cublasLtMatmulAlgoConfigSetAttribute(&self.algo, algo_config_reduction_scheme, &reduction_none, @sizeOf(c_int)), "disable FP8 reduction scheme");
        var checked: Heuristic = undefined;
        try lt.check(lt.api.cublasLtMatmulAlgoCheck(self.handle, self.desc, self.a, self.b, self.c, self.c, &self.algo, &checked), "check FP8 algorithm");
        if (checked.state != 0 or checked.workspace_size > workspace_limit) return error.NoAlgorithm;
        self.workspace_need = checked.workspace_size;
        return self;
    }

    pub fn deinit(self: *Fp8BlockLinear) void {
        const api = self.lt.api;
        _ = api.cublasLtMatmulPreferenceDestroy(self.pref);
        _ = api.cublasLtMatrixLayoutDestroy(self.c);
        _ = api.cublasLtMatrixLayoutDestroy(self.b);
        _ = api.cublasLtMatrixLayoutDestroy(self.a);
        _ = api.cublasLtMatmulDescDestroy(self.desc);
        _ = api.cublasLtDestroy(self.handle);
        self.* = undefined;
    }

    pub fn run(self: *const Fp8BlockLinear, x: abi.DevicePtr, w: abi.DevicePtr, sx: abi.DevicePtr, sw: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (workspace_len < self.workspace_need) return error.Invalid;
        try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_a_scale_pointer, &sw, @sizeOf(abi.DevicePtr)), "set FP8 A=weight scales");
        try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_b_scale_pointer, &sx, @sizeOf(abi.DevicePtr)), "set FP8 B=activation scales");
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLt FP8 TN column-major matmul");
    }
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Linear);
    std.testing.refAllDecls(Fp8BlockLinear);
}

test "cuBLASLt FP8 operand map is TN column-major" {
    const mapping = fp8TnLayouts();
    try std.testing.expectEqual(Op.t, mapping.transa);
    try std.testing.expectEqual(Op.n, mapping.transb);
    try std.testing.expectEqual(Order.col, mapping.order);
    try std.testing.expectEqual(scale_blk128x128_32f, mapping.scale_a);
    try std.testing.expectEqual(scale_vec128_32f, mapping.scale_b);
}

comptime {
    std.debug.assert(@sizeOf(Algo) == 64 and @sizeOf(Heuristic) == 96);
}
