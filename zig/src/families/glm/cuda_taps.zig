// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! DFlash2's taps on the target's forward: after each tapped layer, one configured row into its tap slot.
const cuda = @import("cuda");

/// The target layers DFlash2 reads (incoai/GLM-5.3-Flash-DFlash2's target_layer_ids), in tap-slot order.
pub const layers = [_]usize{ 5, 14, 24, 33, 42 };

/// Where a forward's taps go: rows [R][n_taps * hidden] bf16 at `out`, through glm_dflash's df_tap.
pub const Taps = struct {
    f: cuda.Function,
    out: u64,
    hidden: usize = 4096,
    n_taps: usize = layers.len,
    layers: []const usize = &layers,
    streams: usize = 4,
    flash_abi: bool = true,

    /// After layer `j` of an `R`-row pass on residual `x`: Flash averages four streams; GLM copies plain hidden rows.
    pub fn at(t: Taps, s: cuda.Stream, j: usize, x: u64, R: usize) !void {
        const slot = slotOf(t.layers, j) orelse return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(t.out);
        a.add(slot);
        a.add(@as(i32, @intFromBool(t.streams == 1))); // mode 1 copies plain rows; mode 0 averages Flash's four streams
        if (!t.flash_abi) {
            a.add(@as(i32, @intCast(t.hidden)));
            a.add(@as(i32, @intCast(t.n_taps)));
        }
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(R), .y = @intCast(t.hidden / 1024) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn slotOf(target_layers: []const usize, layer: usize) ?i32 {
        for (target_layers, 0..) |target, slot| if (target == layer) return @intCast(slot);
        return null;
    }

    test "glm dflash taps map geometry-provided target layers" {
        const targets = [_]usize{ 5, 19, 33, 47, 61, 75 };
        try @import("std").testing.expectEqual(@as(?i32, 0), slotOf(&targets, 5));
        try @import("std").testing.expectEqual(@as(?i32, 5), slotOf(&targets, 75));
        try @import("std").testing.expectEqual(@as(?i32, null), slotOf(&targets, 14));
    }
};

test {
    @import("std").testing.refAllDecls(@This());
    @import("std").testing.refAllDecls(Taps);
}
