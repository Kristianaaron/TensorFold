// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Runtime loader for the optional GLM-5.3 CUTLASS host launcher.
const std = @import("std");
const cuda = @import("cuda");

pub const Error = error{ LibraryUnavailable, MissingSymbol, Unsupported, InvalidShape, WorkspaceTooSmall };
pub const InitFn = *const fn () callconv(.c) c_int;
pub const WorkspaceFn = *const fn (c_int, c_int, c_int) callconv(.c) usize;
pub const DynamicSmemFn = *const fn () callconv(.c) usize;
pub const GemmFn = *const fn (cuda.abi.DevicePtr, cuda.abi.DevicePtr, cuda.abi.DevicePtr, cuda.abi.DevicePtr, cuda.abi.DevicePtr, c_int, c_int, c_int, cuda.abi.DevicePtr, usize, cuda.abi.Stream) callconv(.c) c_int;

pub const Api = struct {
    tf_glm53_cutlass_init: InitFn,
    tf_glm53_cutlass_workspace: WorkspaceFn,
    tf_glm53_cutlass_dynamic_smem_bytes: DynamicSmemFn,
    tf_glm53_cutlass_gemm: GemmFn,
};

/// The loaded launcher owns no CUDA context and uses the caller's primary-context pointers and stream.
pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    pub fn open() Error!Library {
        const env = std.c.getenv("TF_GLM53_CUTLASS_LIB");
        const path = if (env) |p| std.mem.span(p) else "zig-out/kernels/libtf_glm53_cutlass.so";
        return openPath(path);
    }

    pub fn openPath(path: []const u8) Error!Library {
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        const status = api.tf_glm53_cutlass_init();
        if (status != 0) return error.Unsupported;
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn dynamicSmemBytes(self: *const Library) usize {
        return self.api.tf_glm53_cutlass_dynamic_smem_bytes();
    }

    pub fn workspace(self: *const Library, m: usize, n: usize, k: usize) Error!usize {
        if (m == 0 or n == 0 or k == 0 or m > std.math.maxInt(c_int) or n > std.math.maxInt(c_int) or k > std.math.maxInt(c_int)) return error.InvalidShape;
        const bytes = self.api.tf_glm53_cutlass_workspace(@intCast(m), @intCast(n), @intCast(k));
        if (bytes == std.math.maxInt(usize)) return error.Unsupported;
        return bytes;
    }

    pub fn gemm(self: *const Library, a: cuda.abi.DevicePtr, b: cuda.abi.DevicePtr, sfa: cuda.abi.DevicePtr, sfb: cuda.abi.DevicePtr, d: cuda.abi.DevicePtr, m: usize, n: usize, k: usize, workspace_ptr: cuda.abi.DevicePtr, workspace_bytes: usize, stream: cuda.Stream) Error!void {
        if (m == 0 or n == 0 or k == 0 or m > std.math.maxInt(c_int) or n > std.math.maxInt(c_int) or k > std.math.maxInt(c_int)) return error.InvalidShape;
        const status = self.api.tf_glm53_cutlass_gemm(a, b, sfa, sfb, d, @intCast(m), @intCast(n), @intCast(k), workspace_ptr, workspace_bytes, stream.handle);
        if (status != 0) {
            std.log.err("CUTLASS GEMM returned status {d} for M={d} N={d} K={d}", .{ status, m, n, k });
            return error.Unsupported;
        }
    }
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Library);
}
