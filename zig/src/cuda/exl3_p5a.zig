// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 TF_GLM53_PROMPT_EXPERTS=10: the glm53_p5a small-item (<= 64-member) prompt expert kernels.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
pub const threads: u32 = 256;
pub const small_max: usize = 64;
pub const smem_static: u32 = 43_008;
pub const P5a = struct {
    mod: module.Module,
    gu: [2]module.Function,
    down: [2]module.Function,
    pub fn load(d: *const driver.Driver) !P5a {
        if (!kernels.available or kernels.glm53_p5a.len == 0) return error.BuiltWithoutKernels;
        var p: P5a = .{ .mod = try module.Module.load(d, kernels.glm53_p5a), .gu = undefined, .down = undefined };
        errdefer p.mod.unload();
        p.gu = .{ try p.mod.function("tf_glm53_p5a_gu_mcg"), try p.mod.function("tf_glm53_p5a_gu_e4m3") };
        p.down = .{ try p.mod.function("tf_glm53_p5a_down_mcg"), try p.mod.function("tf_glm53_p5a_down_e4m3") };
        return p;
    }
    pub fn unload(p: *P5a) void { p.mod.unload(); }
};
test "p5a static resources fit two CTAs per SM" {
    try std.testing.expect(2 * (smem_static + 64 * 4 + 64 + 1024) <= 102_400);
    try std.testing.expectEqual(@as(usize, 64), small_max);
    try std.testing.expectEqual(@as(u32, 256), threads);
}
