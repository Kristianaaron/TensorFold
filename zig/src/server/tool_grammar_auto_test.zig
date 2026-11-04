// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Ported from ashhart/TensorFold c766e8d0e and 42d478a9f, Apache-2.0.
//! Auto tool requests hold strict arguments to schema and preserve free-form replies.
const std = @import("std");
const json = @import("json");
const grammar = @import("tool_grammar.zig");
const parse = @import("tool_parse.zig");
const Value = json.Value;

pub const offered =
    \\[{"type":"function","function":{"name":"lookup","strict":true,"parameters":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer"}},"required":["q","n"],"additionalProperties":false}}},{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}}}]
;

fn value(a: std.mem.Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

fn schemaOf(call: Value) Value {
    for (call.get("elements").?.array) |part| if (part.typeIs("json_schema")) return part.get("json_schema").?;
    unreachable;
}

test "auto tools: a strict tool asks for a grammar, non-strict tools only with the knob, and false is not strict" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    try std.testing.expect(grammar.autoWanted(tools, false));
    try std.testing.expect(!grammar.autoWanted(tools[1..], false));
    try std.testing.expect(grammar.autoWanted(tools[1..], true));
    try std.testing.expect(!grammar.autoWanted(&.{}, true));
    const off = try value(a, "[{\"name\":\"f\",\"strict\":false},{\"name\":\"g\",\"strict\":\"true\"}]");
    try std.testing.expect(!grammar.autoWanted(off.array, false));
}

test "auto tools: the other styles open one call tag at their shared opener and keep each tool's name after it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    const Case = struct { style: grammar.Style, begin: []const u8, end: []const u8 };
    for ([_]Case{
        .{ .style = .xml, .begin = "<tool_call>", .end = "</tool_call>" },
        .{ .style = .json, .begin = "<tool_call>", .end = "</tool_call>" },
        .{ .style = .gemma, .begin = "<|tool_call>", .end = "<tool_call|>" },
        .{ .style = .glm, .begin = "<tool_call>", .end = "</tool_call>" },
    }) |c| {
        const parallel = c.style != .glm;
        const text = try grammar.optional(a, c.style, tools, false, parallel);
        const format = (try value(a, text)).get("format").?;
        try std.testing.expectEqualStrings("triggered_tags", format.strField("type").?);
        const tag = format.get("tags").?.array[0];
        try std.testing.expectEqualStrings(c.begin, tag.strField("begin").?);
        try std.testing.expectEqualStrings(c.end, tag.strField("end").?);
        const bodies = tag.get("content").?.get("elements").?.array;
        try std.testing.expectEqual(@as(usize, 2), bodies.len);
        const strict_schema = schemaOf(bodies[0]);
        if (c.style == .glm) {
            try std.testing.expectEqualStrings("lookup", bodies[0].get("elements").?.array[0].strField("value").?);
            try std.testing.expectEqualStrings("glm_xml", bodies[0].get("elements").?.array[1].strField("style").?);
            try std.testing.expectEqualStrings("q", strict_schema.get("required").?.array[0].string);
            try std.testing.expectEqualStrings("n", strict_schema.get("required").?.array[1].string);
            try std.testing.expect(!strict_schema.get("additionalProperties").?.bool);
            const free_schema = schemaOf(bodies[1]);
            try std.testing.expectEqualStrings("object", free_schema.strField("type").?);
            try std.testing.expect(free_schema.get("required") == null);
            try std.testing.expect(free_schema.get("properties") == null);
            try std.testing.expect(free_schema.get("additionalProperties").?.bool);
            try std.testing.expectEqualStrings("<tool_call>", format.get("triggers").?.array[0].string);
            try std.testing.expect(format.get("stop_after_first").?.bool);
            try std.testing.expect(std.mem.indexOf(u8, text, "\"value\":\"<tool_call>") == null);
            try std.testing.expectEqualStrings(text, try grammar.optional(a, .glm, tools, false, false));
        } else {
            try std.testing.expect(strict_schema.typeIs("object"));
        }
    }
}

test "auto tools: GLM-5.3's rendered probe picks glm" {
    // Copied from zai-org GLM-5.3 chat_template.jinja, tool-call rendering lines 156-157.
    try std.testing.expectEqual(grammar.Style.glm, grammar.style("<|assistant|><think></think><tool_call>tfprobe_fn<arg_key>tfprobe_k</arg_key><arg_value>v</arg_value></tool_call>").?);
}

test "auto tools: a required GLM form stays a sequence and a rendered reply parses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try value(a, offered)).array;
    const required = try value(a, try grammar.build(a, .glm, tools, false));
    try std.testing.expectEqualStrings("sequence", required.get("format").?.strField("type").?);
    try std.testing.expect(std.mem.indexOf(u8, try grammar.build(a, .glm, tools, false), "<tool_call>now") != null);
    const reply = "Let me check.<tool_call>lookup<arg_key>q</arg_key><arg_value>hi</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call>";
    const result = try parse.parse(a, reply, tools, null);
    try std.testing.expect(std.mem.indexOf(u8, result.content, "Let me check.") != null);
    try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
    try std.testing.expectEqualStrings("lookup", result.calls.?[0].get("function").?.strField("name").?);
}

test "auto tools: the switch parses only 1" {
    try std.testing.expect(grammar.parseToolGrammar("1"));
    try std.testing.expect(grammar.parseToolGrammar(" 1 "));
    try std.testing.expect(!grammar.parseToolGrammar("0"));
    try std.testing.expect(!grammar.parseToolGrammar("true"));
    try std.testing.expect(!grammar.parseToolGrammar(""));
    try std.testing.expect(!grammar.parseToolGrammar(null));
}
