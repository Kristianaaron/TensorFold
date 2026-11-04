#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
"""Single-stream decode speed benchmark grouped by output content type."""
import argparse
import http.client
import json
import statistics
import time
import urllib.error
import urllib.request

PROMPTS = {
    "prose": [
        "Write a thoughtful 450-token essay on why public libraries matter in the digital age. Use clear paragraphs and concrete examples.",
        "Explain in about 500 tokens how urban trees improve quality of life, including tradeoffs and practical maintenance considerations.",
        "Write a 450-token travel narrative about a rainy day in a small coastal town, with vivid but restrained details.",
    ],
    "code_fresh": [
        "Write a complete Python 3.11 module of roughly 450 tokens that implements a small in-memory task queue with priorities, validation, and a few usage examples. Include docstrings.",
        "Create a self-contained Python module, about 500 tokens, for parsing a simple .env file. Handle comments, quoted values, malformed lines, and include a usage example.",
        "Write a complete Python module of roughly 450 tokens implementing a sliding-window rate limiter with a clean API, validation, and a short example.",
    ],
    "code_edit": [
        "Rename `calculate_total` to `calculate_invoice_total`, add an optional `discount=0` parameter, and apply it after tax. Return the entire edited file, with no explanation.\n```python\nfrom dataclasses import dataclass\nfrom decimal import Decimal\n\n@dataclass\nclass Item:\n    name: str\n    price: Decimal\n    quantity: int = 1\n\n@dataclass\nclass Invoice:\n    items: list[Item]\n    tax_rate: Decimal = Decimal('0.0')\n\ndef validate_item(item: Item) -> None:\n    if not item.name.strip():\n        raise ValueError('item name is required')\n    if item.price < 0:\n        raise ValueError('price cannot be negative')\n    if item.quantity < 1:\n        raise ValueError('quantity must be positive')\n\ndef calculate_total(invoice: Invoice) -> Decimal:\n    if invoice.tax_rate < 0:\n        raise ValueError('tax rate cannot be negative')\n    subtotal = Decimal('0')\n    for item in invoice.items:\n        validate_item(item)\n        subtotal += item.price * item.quantity\n    tax = (subtotal * invoice.tax_rate).quantize(Decimal('0.01'))\n    return (subtotal + tax).quantize(Decimal('0.01'))\n\ndef format_invoice(invoice: Invoice) -> str:\n    lines = ['Invoice']\n    for item in invoice.items:\n        line_total = item.price * item.quantity\n        lines.append(f'{item.name}: {item.quantity} x {item.price} = {line_total}')\n    lines.append(f'Total: {calculate_total(invoice)}')\n    return '\\n'.join(lines)\n\ndef sample_invoice() -> Invoice:\n    return Invoice([Item('Notebook', Decimal('4.50'), 2), Item('Pen', Decimal('1.25'))], Decimal('0.08'))\n\nif __name__ == '__main__':\n    print(format_invoice(sample_invoice()))\n```",
        "Rename `calculate_total` to `calculate_invoice_total`, add optional `discount=0`, subtract it after tax, and return the whole file only.\n```python\nfrom dataclasses import dataclass\nfrom decimal import Decimal\n\n@dataclass\nclass Item:\n    name: str\n    price: Decimal\n    quantity: int = 1\n\n@dataclass\nclass Invoice:\n    items: list[Item]\n    tax_rate: Decimal = Decimal('0.0')\n\ndef validate_item(item: Item) -> None:\n    if not item.name.strip():\n        raise ValueError('item name is required')\n    if item.price < 0 or item.quantity < 1:\n        raise ValueError('invalid item amount')\n\ndef calculate_total(invoice: Invoice) -> Decimal:\n    if invoice.tax_rate < 0:\n        raise ValueError('tax rate cannot be negative')\n    subtotal = Decimal('0')\n    for item in invoice.items:\n        validate_item(item)\n        subtotal += item.price * item.quantity\n    tax = (subtotal * invoice.tax_rate).quantize(Decimal('0.01'))\n    return (subtotal + tax).quantize(Decimal('0.01'))\n\ndef format_invoice(invoice: Invoice) -> str:\n    rows = ['Invoice']\n    for item in invoice.items:\n        rows.append(f'{item.name}: {item.quantity} x {item.price}')\n    rows.append(f'Total: {calculate_total(invoice)}')\n    return '\\n'.join(rows)\n\ndef sample_invoice() -> Invoice:\n    return Invoice([Item('Book', Decimal('12.00')), Item('Tea', Decimal('3.50'), 2)], Decimal('0.05'))\n\nif __name__ == '__main__':\n    print(format_invoice(sample_invoice()))\n```",
        "Rename `calculate_total` to `calculate_invoice_total`, add optional `discount=0`, apply after tax, and output the entire file without commentary.\n```python\nfrom dataclasses import dataclass\nfrom decimal import Decimal\n\n@dataclass\nclass Item:\n    name: str\n    price: Decimal\n    quantity: int = 1\n\n@dataclass\nclass Invoice:\n    items: list[Item]\n    tax_rate: Decimal = Decimal('0.0')\n\ndef validate_item(item: Item) -> None:\n    if not item.name.strip():\n        raise ValueError('item name is required')\n    if item.price < 0:\n        raise ValueError('price cannot be negative')\n    if item.quantity <= 0:\n        raise ValueError('quantity must be positive')\n\ndef calculate_total(invoice: Invoice) -> Decimal:\n    subtotal = sum((item.price * item.quantity for item in invoice.items), Decimal('0'))\n    if invoice.tax_rate < 0:\n        raise ValueError('tax rate cannot be negative')\n    tax = (subtotal * invoice.tax_rate).quantize(Decimal('0.01'))\n    return (subtotal + tax).quantize(Decimal('0.01'))\n\ndef format_invoice(invoice: Invoice) -> str:\n    details = [f'{item.name}: {item.quantity} x {item.price}' for item in invoice.items]\n    details.append(f'Total: {calculate_total(invoice)}')\n    return '\\n'.join(details)\n\ndef sample_invoice() -> Invoice:\n    return Invoice([Item('Coffee', Decimal('5.25'), 2), Item('Mug', Decimal('8.00'))], Decimal('0.07'))\n\nif __name__ == '__main__':\n    print(format_invoice(sample_invoice()))\n```",
    ],
    "structured": [
        "Return only a valid JSON array of exactly 40 objects, each with fixed keys `id` (integer 1-40), `name` (string), `category` (string), `active` (boolean), and `score` (number). Use varied plausible sample records; no markdown.",
        "Produce only a JSON array containing 40 objects. Every object must have exactly these keys: id (integer), name (string), category (string), active (boolean), score (number). IDs run from 1 through 40. No prose.",
        "Generate valid JSON only: an array of 40 distinct sample objects with keys id, name, category, active, score. Use IDs 1..40, correct JSON types, varied values, and no extra keys.",
    ],
    "tool_json": [
        "Prepare a three-day trip plan for a user. You must use exactly three tool calls: search_flights, find_hotel, and create_itinerary. Choose reasonable illustrative parameters and make the calls in that order.",
        "A user wants a weekend visit to Kyoto. Make exactly three tool calls, in order: search_flights, find_hotel, create_itinerary. Use sensible parameters; do not answer in prose.",
        "Plan a short business trip to Berlin. Issue exactly three tool calls in order: search_flights, find_hotel, create_itinerary, with useful arguments. No prose response.",
    ],
}

TOOLS = [
    {"type": "function", "function": {"name": "search_flights", "description": "Search available flights", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["destination", "date"]}}},
    {"type": "function", "function": {"name": "find_hotel", "description": "Find a hotel", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "nights": {"type": "integer"}}, "required": ["city", "nights"]}}},
    {"type": "function", "function": {"name": "create_itinerary", "description": "Create an itinerary", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "days": {"type": "integer"}}, "required": ["destination", "days"]}}},
]


def stream_request(url, payload, thinking=False):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    token_count = 0
    thinking_runs = 0
    tool_chunks = 0
    tool_calls = 0
    usage = None
    extra = {}
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") if thinking else None
                calls = delta.get("tool_calls") or []
                if reasoning:
                    thinking_runs += 1
                # Tool calls stream as delta.tool_calls, not content; both count as output for TTFT.
                if content or reasoning or calls:
                    if first is None:
                        first = now
                    token_count += 1  # chunks, not tokens: a speculative round can stream several tokens at once
                if calls:
                    tool_chunks += 1
                    tool_calls += len(calls)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for key, value in chunk.items():
                if key not in ("id", "object", "created", "model", "choices", "usage"):
                    extra[key] = value
    end = time.perf_counter()
    ttft = first - start if first is not None else None
    elapsed = end - first if first is not None else None
    # Decode speed from the server's token count (usage), since one chunk can hold several accepted draft tokens.
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if tokens is None:
        raise RuntimeError("server returned no usage.completion_tokens; decode tok/s would count chunks, not tokens")
    streamed = not (tool_calls > 1 and tool_chunks == 1) and elapsed is not None and elapsed >= 0.05
    speed = (max(0, tokens - 1) / elapsed) if streamed else None
    return {"ttft_s": ttft, "decode_s": elapsed, "content_chunks": token_count, "thinking_runs": thinking_runs, "stream_status": "streamed" if streamed else "not streamed", "tok_s": speed, "usage": usage, "server_fields": extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--thinking", action="store_true", help="enable model thinking and include reasoning deltas in TTFT")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", choices=sorted(PROMPTS), action="append")
    parser.add_argument("--json", metavar="OUT")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    selected = args.only or list(PROMPTS)
    results = []
    for set_name in selected:
        for repeat in range(1, args.repeat + 1):
            for prompt_index, prompt in enumerate(PROMPTS[set_name], 1):
                payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "stream": True, "stream_options": {"include_usage": True}, "temperature": args.temperature, "max_tokens": 700, "chat_template_kwargs": {"enable_thinking": args.thinking}}
                if set_name == "tool_json":
                    payload["tools"] = TOOLS
                    payload["tool_choice"] = "auto"
                record = {"set": set_name, "repeat": repeat, "prompt": prompt_index}
                try:
                    record.update(stream_request(args.base_url.rstrip("/") + "/chat/completions", payload, args.thinking))
                    record["error"] = None
                    print(f"{set_name} repeat={repeat} prompt={prompt_index}: {record['tok_s'] if record['tok_s'] is not None else 'n/a'} tok/s TTFT={record['ttft_s']}")
                except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                    record.update({"ttft_s": None, "decode_s": None, "content_chunks": None, "tok_s": None, "usage": None, "server_fields": {}, "error": str(exc)})
                    print(f"{set_name} repeat={repeat} prompt={prompt_index}: ERROR {exc}")
                results.append(record)
    print("\nSet summary (tok/s; TTFT seconds)")
    for set_name in selected:
        rows = [r for r in results if r["set"] == set_name and r["tok_s"] is not None]
        if rows:
            speeds = [r["tok_s"] for r in rows]
            ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
            print(f"{set_name:12} mean={statistics.mean(speeds):.2f} min={min(speeds):.2f} max={max(speeds):.2f} mean_ttft={statistics.mean(ttfts):.3f}" if ttfts else f"{set_name:12} mean={statistics.mean(speeds):.2f} min={min(speeds):.2f} max={max(speeds):.2f} mean_ttft=n/a")
        else:
            print(f"{set_name:12} no successful requests")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output:
            json.dump({"model": args.model, "base_url": args.base_url, "temperature": args.temperature, "repeat": args.repeat, "results": results}, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
