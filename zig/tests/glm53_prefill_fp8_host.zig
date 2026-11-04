// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Host reference coverage for the row-major BF16 scratch used by GLM-53 prefill.
const std = @import("std");
const cuda = @import("cuda");
const fp8 = cuda.fp8;

test "FP8B row-major scratch matches the native packed code and scale lookup" {
    const n = 130;
    const k = 256;
    const npad = fp8.padded(n);
    const codes = try std.testing.allocator.alloc(u8, npad * k);
    defer std.testing.allocator.free(codes);
    const scales = try std.testing.allocator.alloc(f32, npad / fp8.group * (k / fp8.group));
    defer std.testing.allocator.free(scales);
    for (codes, 0..) |*code, i| code.* = @truncate((i *% 37 + 0x28) & 0x7e);
    for (scales, 0..) |*scale, i| scale.* = if (i % 2 == 0) 0.5 else 1.25;
    const rowmajor = try std.testing.allocator.alloc(u16, n * k);
    defer std.testing.allocator.free(rowmajor);
    try fp8.dequantizeBf16RowMajor(rowmajor, codes, scales, n, k);
    for (0..n) |row| for (0..k) |col| {
        const tile = (row / fp8.group) * (k / fp8.group) + col / fp8.group;
        const expected = fp8.fp8bWeight(codes[fp8.packedCodeIndex(row, col, k)], scales[tile]);
        try std.testing.expectEqual(@as(u16, @truncate(@as(u32, @bitCast(expected)) >> 16)), rowmajor[row * k + col]);
    };
}

test {
    std.testing.refAllDecls(@This());
}
