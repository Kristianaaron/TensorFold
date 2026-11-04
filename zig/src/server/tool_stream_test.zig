const std = @import("std");
const json = @import("json");
const tool_params = @import("tool_params.zig");
const stream = @import("tool_stream.zig");
const Streamer = stream.Streamer;
const Policy = stream.Policy;
const Value = json.Value;
const Allocator = std.mem.Allocator;

test "a streamed call its end token left open completes as its closing markup completes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n";
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    const ended = try streamed(a, tools, &.{ open, try std.mem.concat(a, u8, &.{ open, try tool_parse.closeCall(a, open, tools) }) });
    const marked = try streamed(a, tools, &.{ open, open ++ "</function>\n</tool_call>" });
    // the arguments sent before the end, closed by closedJson, are the ones the client ends with
    try std.testing.expectEqualStrings((try tool_params.closedJson(a, sent)).?, try sentArguments(a, ended));
    try std.testing.expectEqual(marked.len, ended.len);
    for (marked[1..], ended[1..]) |x, y| try std.testing.expectEqualStrings(try json.stringify(a, x, .{}), try json.stringify(a, y, .{}));
}

test "a streamed call cut inside a value stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars and more";
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, open, tools));
    // its opening delta went out and cannot be taken back; what it sent does not close under closedJson
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    try std.testing.expectEqualStrings("{\"q\":\"fast c", sent);
    try std.testing.expect((try tool_params.closedJson(a, sent)) == null);
}

test "a one-call reply drops an unoffered call and still hides an unclosed one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const single: Policy = .{ .single = true };
    const text = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    // what attachCalls makes of the reply: the parse, then the policy's content
    const r = try tool_parse.parse(a, text, tools, single.maxCalls());
    try std.testing.expect(r.calls == null);
    try std.testing.expectEqualStrings("Launching.", r.content);
    try std.testing.expectEqualStrings("Launching.", try single.parsedContent(a, r.content));
    try std.testing.expectEqualStrings("Launching. ", try single.parsedContent(a, "Launching. <tool_call>{\"name\""));
    // a stream that sent the prose before the block ends with the block
    try std.testing.expectEqualStrings(text["Launching. ".len..], single.kept(text, "\nLaunching. ").?);
    try std.testing.expect(single.kept("Launching.", "Launching. ") == null);
    const parallel: Policy = .{};
    try std.testing.expect(parallel.kept(text, "Launching. ") == null);
}

const tool_parse = @import("tool_parse.zig");

/// The deltas one streamer sends while the reply grows through ``texts``.
fn streamed(a: Allocator, tools: []const Value, texts: []const []const u8) Allocator.Error![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (texts) |t| try s.feed(t, &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn sentArguments(a: Allocator, deltas: []const Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

fn sentArgumentsAt(a: Allocator, deltas: []const Value, index: i64) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |delta| for (delta.get("tool_calls").?.array) |one| {
        if (one.get("index").?.int64().? != index) continue;
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

fn expectXmlParity(a: Allocator, schema: []const u8, value: []const u8) !void {
    const tool_text = try std.mem.concat(a, u8, &.{
        "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"$defs\":{\"Limit\":{\"type\":\"integer\"},\"Nested\":{\"type\":\"object\",\"properties\":{\"limit\":{\"$ref\":\"#/$defs/Limit\"}}},\"A\":{\"$ref\":\"#/$defs/B\"},\"B\":{\"$ref\":\"#/$defs/A\"}},\"properties\":{\"n\":",
        schema,
        "}}}}]",
    });
    const tools = (try json.parse(a, tool_text)).ok.array;
    const text = try std.fmt.allocPrint(a, "<tool_call><function=lookup><parameter=n>\n{s}\n</parameter></function></tool_call>", .{value});
    const parsed = try tool_parse.parse(a, text, tools, null);
    const expected = parsed.calls.?[0].get("function").?.get("arguments").?.string;
    const actual = try sentArguments(a, try streamed(a, tools, &.{text}));
    const expected_value = (try json.parse(a, expected)).ok;
    const actual_value = (try json.parse(a, actual)).ok;
    try std.testing.expectEqualStrings(try json.stringify(a, expected_value, .{}), try json.stringify(a, actual_value, .{}));
}

test "effective schemas keep streamed and whole arguments in parity" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectXmlParity(a, "{\"$ref\":\"#/$defs/Limit\"}", "5");
    try expectXmlParity(a, "{\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"null\"}]}", "null");
    try expectXmlParity(a, "{\"oneOf\":[{\"type\":\"integer\"},{\"type\":\"null\"}]}", "5");
    try expectXmlParity(a, "{\"type\":[\"integer\",\"null\"]}", "5");
    try expectXmlParity(a, "{\"$ref\":\"#/$defs/Nested\"}", "{\"limit\":5}");
    try expectXmlParity(a, "{\"$ref\":\"#/$defs/A\"}", "5");
    try expectXmlParity(a, "{\"$ref\":\"#/$defs/Missing\"}", "5");
    try expectXmlParity(a, "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}", "hello");
}

fn expectGlmParity(a: Allocator, tools: []const Value, text: []const u8) !void {
    const parsed = try tool_parse.parse(a, text, tools, null);
    const expected = parsed.calls orelse return error.TestExpectedCalls;
    const whole = try streamed(a, tools, &.{text});
    try std.testing.expectEqual(expected.len, countNamedCalls(whole));
    for (expected, 0..) |call, idx| {
        const expected_args = call.get("function").?.get("arguments").?.string;
        const actual = try sentArgumentsAt(a, whole, @intCast(idx));
        const want_value = (try json.parse(a, expected_args)).ok;
        const got_value = (try json.parse(a, actual)).ok;
        try std.testing.expectEqualStrings(try json.stringify(a, want_value, .{}), try json.stringify(a, got_value, .{}));
    }

    var s = try Streamer.init(a, tools);
    var pieces: std.ArrayList(Value) = .empty;
    var at: usize = 0;
    while (at < text.len) {
        var end = at + 1;
        while (end < text.len and text[end] & 0xc0 == 0x80) end += 1;
        try s.feed(text[0..end], &pieces);
        at = end;
    }
    try std.testing.expectEqual(expected.len, countNamedCalls(pieces.items));
    for (expected, 0..) |call, idx| {
        const expected_args = call.get("function").?.get("arguments").?.string;
        const actual = try sentArgumentsAt(a, pieces.items, @intCast(idx));
        const want_value = (try json.parse(a, expected_args)).ok;
        const got_value = (try json.parse(a, actual)).ok;
        try std.testing.expectEqualStrings(try json.stringify(a, want_value, .{}), try json.stringify(a, got_value, .{}));
    }
}

fn countNamedCalls(deltas: []const Value) usize {
    var count: usize = 0;
    for (deltas) |delta| for (delta.get("tool_calls").?.array) |one| {
        if (one.get("id") != null) count += 1;
    };
    return count;
}

test "GLM arguments stream incrementally before a long value closes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const value = try a.alloc(u8, 5000);
    @memset(value, 'x');
    const prefix = "<tool_call>lookup<arg_key>q</arg_key><arg_value>";
    const before_close = try std.mem.concat(a, u8, &.{ prefix, value });
    const complete = try std.mem.concat(a, u8, &.{ before_close, "</arg_value></tool_call>" });
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    var first_name = false;
    var name_at: usize = 0;
    for (0..(before_close.len + 49) / 50) |step| {
        const end = @min(before_close.len, (step + 1) * 50);
        try s.feed(before_close[0..end], &out);
        for (out.items) |delta| for (delta.get("tool_calls").?.array) |one| {
            const function = one.get("function").?;
            if (!first_name and function.get("name") != null) {
                first_name = true;
                name_at = end;
            }
        };
    }
    try std.testing.expect(first_name);
    try std.testing.expect(name_at < before_close.len);
    var argument_deltas: usize = 0;
    for (out.items) |delta| for (delta.get("tool_calls").?.array) |one| {
        if (one.get("function").?.get("arguments").?.string.len > 0) argument_deltas += 1;
    };
    try std.testing.expect(argument_deltas >= 40);
    try s.feed(complete, &out);
}

test "GLM streamed arguments match whole parsing across values and feed boundaries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"q\":{\"type\":\"string\"},\"other\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"ok\":{\"type\":\"boolean\"},\"items\":{\"type\":\"array\"}}}}}]")).ok.array;
    const cases = [_][]const u8{
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>one</arg_value><arg_key>other</arg_key><arg_value>two</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>n</arg_key><arg_value>17</arg_value><arg_key>ok</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>\n  \"quoted\" \\\\ slash\nsecond line  \n</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>  lead\nline\n tail  </arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>café 🚀</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>< and </arg and </arg_key></arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value></arg_value></tool_call>",
        "<tool_call>lookup</tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>one</arg_value></tool_call><tool_call>lookup<arg_key>q</arg_key><arg_value>two</arg_value></tool_call>",
        "A little prose. <tool_call>lookup<arg_key>q</arg_key><arg_value>inside</arg_value></tool_call>",
    };
    for (cases) |text| try expectGlmParity(a, tools, text);
}

test "untyped GLM JSON containers stream with whole-parser parity" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"payload\":{\"description\":\"arbitrary request data\"}}}}}]")).ok.array;
    const text = "<tool_call>lookup<arg_key>payload</arg_key><arg_value>{\"id\": 7, \"tags\": [\"alpha\", \"beta\"]}</arg_value></tool_call>";
    try expectGlmParity(a, tools, text);
}

test "unrecognized and legacy call envelopes retain their existing stream paths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    for ([_][]const u8{
        "<tool_call>missing<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "<tool_call><function=lookup",
        "<tool_call>lookup q</arg_key><arg_value>x</arg_value></tool_call>",
    }) |text| {
        const deltas = try streamed(a, tools, &.{text});
        try std.testing.expectEqual(@as(usize, 0), deltas.len);
    }
    const xml = try streamed(a, tools, &.{"<tool_call><function=lookup><parameter=q>x</parameter></function></tool_call>"});
    try std.testing.expectEqualStrings("{\"q\":\"x\"}", try sentArguments(a, xml));
}

test "a GLM call cut inside a value closes to the sent arguments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const open = "<tool_call>lookup<arg_key>q</arg_key><arg_value>unfinished value";
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    try std.testing.expect(std.mem.startsWith(u8, sent, "{\"q\":\"unfi"));
    const close = try tool_parse.closeCall(a, open, tools);
    try std.testing.expect(close.len > 0);
    const completed = try streamed(a, tools, &.{ open, try std.mem.concat(a, u8, &.{ open, close }) });
    try std.testing.expectEqualStrings("{\"q\":\"unfinished value\"}", try sentArguments(a, completed));
}

test "DSML containment survives every byte boundary and preserves adjacent call order" {
    const prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">";
    const suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
    const legacy = "<tool_call><function=lookup></function></tool_call>";
    for ([_][]const u8{ legacy, "<tool_call><function=lookup></function>" }) |literal| {
        var outer_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer outer_arena.deinit();
        const outer_a = outer_arena.allocator();
        const text = try std.mem.concat(outer_a, u8, &.{ legacy, prefix, literal, suffix, legacy });
        const expected = try std.mem.concat(outer_a, u8, &.{ "{}{\"q\":", try json.quote(outer_a, literal, .{}), "}{}" });
        for (0..text.len + 1) |boundary| {
            var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
            var s = try Streamer.init(a, tools);
            var out: std.ArrayList(Value) = .empty;
            try s.feed(text[0..boundary], &out);
            try s.feed(text, &out);
            try s.feed(text, &out);
            var calls: i64 = 0;
            for (out.items) |delta| for (delta.get("tool_calls").?.array) |call| {
                if (call.get("id") == null) continue;
                try std.testing.expectEqual(calls, call.get("index").?.int64().?);
                try std.testing.expectEqualStrings("lookup", call.get("function").?.strField("name").?);
                calls += 1;
            };
            try std.testing.expectEqual(@as(i64, 3), calls);
            try std.testing.expectEqualStrings(expected, try sentArguments(a, out.items));
        }
    }
}

test "unfinished DSML retains containment through byte feeds and final closeCall" {
    const prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">";
    const literals = [_][]const u8{
        "<tool_call><function=lookup></function></tool_call>",
        "<tool_call><function=lookup></function>",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{}",
    };
    for (literals) |literal| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
        const text = try std.mem.concat(a, u8, &.{ prefix, literal });
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        for (0..text.len + 1) |n| {
            try s.feed(text[0..n], &out);
            try std.testing.expectEqual(@as(usize, 0), out.items.len);
        }
        const tail = try tool_parse.closeCall(a, text, tools);
        try std.testing.expectEqualStrings("", tail);
        try s.feed(try std.mem.concat(a, u8, &.{ text, tail }), &out);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
        try std.testing.expect(!s.streamed);
    }
}

test "malformed closed DSML masks its nested XML and resumes the next legitimate call" {
    const malformed = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
        "<｜DSML｜parameter name=\"q\" string=\"true\">" ++
        "<tool_call><function=lookup></function></tool_call>" ++
        "</｜DSML｜parameter></｜DSML｜tool_calls>";
    const legacy = "<tool_call><function=lookup></function></tool_call>";
    const text = malformed ++ legacy;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (0..malformed.len + 1) |n| {
        try s.feed(text[0..n], &out);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
    }
    for (malformed.len + 1..text.len + 1) |n| try s.feed(text[0..n], &out);
    var calls: usize = 0;
    for (out.items) |delta| for (delta.get("tool_calls").?.array) |call| {
        if (call.get("id") != null) calls += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), calls);
    try std.testing.expectEqualStrings("{}", try sentArguments(a, out.items));
}
