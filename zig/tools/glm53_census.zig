// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
const std = @import("std");
const census = @import("census");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.Usage;
    const result = try census.readModelDir(init.gpa, init.io, args[1]);
    var buffer: [1024]u8 = undefined;
    var output = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    defer output.interface.flush() catch {};
    try census.write(&output.interface, result);
    return 0;
}
