# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
# Draft-depth recalibration for the GLM-5.3 TensorFold engine.
# Depth policy after MiaAI-Lab GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold patches 0114/0116/0119/0126 (Apache-2.0,
# Copyright 2026 Mia's AI Lab): confidence survival times measured per-width round costs, costs from real text, kept
# nondecreasing. The confidence head is Red Hat's DSpark (RedHatAI/GLM-5.3-speculator.dspark, GLM-5.3 license; vllm-project/speculators,
# Apache-2.0). The allocator replica follows zig/src/core/lanes/allocate.zig (port of ashhart/TensorFold allocate.py).
# Per-slot logistic recalibration is Platt scaling. Independent implementation; no source copied.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

GB10 = [88.0, 104.1, 120.2, 130.3, 145.1, 159.8, 175.8, 191.8, 212.4, 233.1, 253.7, 274.3, 288.8, 303.2, 317.7, 332.2]
SCHEMA = "tf.glm53.draft-recal/v1"
TRACE_SCHEMA = "tf.glm53.round-trace/v1"
FEATURES = ["z", "margin", "logp", "1"]
SLOTS = 7
_FIT_MIN_ROWS = 200


@dataclass
class Draft:
    pos: int
    anchor: int
    n: int
    depth: int
    t: float
    drafts: list[int]
    p: list[float]
    z: list[float]
    m: list[float]
    lp: list[float]


@dataclass
class Verify:
    start: int
    rows: int
    held: int
    copy: int
    t_ns: int
    v_ns: int
    ids: list[int]
    sampled: list[int]


@dataclass
class Request:
    key: tuple[int, int, int]
    open: dict
    mode: str
    drafts: list[Draft] = field(default_factory=list)
    verifies: list[Verify] = field(default_factory=list)
    released: bool = False


@dataclass
class Block:
    mode: str
    key: tuple[int, int, int]
    pos: int
    L: int
    n: int
    p: list[float]
    z: list[float]
    m: list[float]
    lp: list[float]


def _bad(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _require(obj: dict, keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        _bad(f"missing required key(s) {missing} in {where}")


def _int(obj: dict, key: str, where: str) -> int:
    try:
        value = obj[key]
        converted = int(value)
        if isinstance(value, float) and value != converted:
            raise ValueError
        return converted
    except (TypeError, ValueError, OverflowError):
        _bad(f"{where}: {key} must be an integer")


def _float(obj: dict, key: str, where: str) -> float:
    try:
        value = float(obj[key])
    except (TypeError, ValueError, OverflowError):
        _bad(f"{where}: {key} must be numeric")
    if not math.isfinite(value):
        _bad(f"{where}: {key} must be finite")
    return value


def _as_int(value):
    converted = int(value)
    if isinstance(value, float) and value != converted:
        raise ValueError("non-integral value")
    return converted


def _array(obj: dict, key: str, length: int, where: str, cast: Callable = float) -> list:
    value = obj[key]
    if not isinstance(value, list) or len(value) < length:
        _bad(f"{where}: {key} must be a list with at least {length} items")
    try:
        result = [cast(x) for x in value]
    except (TypeError, ValueError, OverflowError):
        _bad(f"{where}: invalid values in {key}")
    if cast is float and not all(math.isfinite(x) for x in result):
        _bad(f"{where}: non-finite values in {key}")
    return result


def load(paths) -> list[Request]:
    requests: list[Request] = []
    skipped = 0
    for file_index, path in enumerate(paths):
        sessions: list[Request] = []
        current: Request | None = None
        session_no = -1
        try:
            raw_lines = Path(path).read_text().splitlines()
        except OSError as exc:
            _bad(f"cannot read trace {path}: {exc}")
        for line_no, line in enumerate(raw_lines, 1):
            if not line.strip():
                _bad(f"{path}:{line_no}: blank line is not valid JSON")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_no == len(raw_lines):
                    warnings.warn(f"{path}:{line_no}: skipping truncated final trace line", RuntimeWarning)
                    break
                _bad(f"{path}:{line_no}: invalid JSON: {exc}")
            if not isinstance(obj, dict) or "ev" not in obj:
                _bad(f"{path}:{line_no}: event must be a JSON object with ev")
            ev = obj["ev"]
            where = f"{path}:{line_no}"
            if ev == "open":
                _require(obj, ("schema", "draft_chances", "live_costs", "lane_chances", "copyd", "copy_rows", "verify_max", "conf_temp", "dspark", "recal", "sweep"), where)
                if obj["schema"] != TRACE_SCHEMA:
                    _bad(f"{where}: unsupported trace schema {obj['schema']!r}")
                current = None
                session_no += 1
                sessions.append(None)  # session boundary marker
            elif ev in ("draft", "verify", "release"):
                if session_no < 0:
                    _bad(f"{where}: {ev} appears before open")
                if ev == "draft":
                    _require(obj, ("req", "pos", "anchor", "n", "depth", "t", "drafts", "p", "z", "m", "lp"), where)
                    if current is None:
                        current = Request((file_index, session_no, _int(obj, "req", where)), {}, "greedy")
                        sessions.append(current)
                    if _int(obj, "req", where) != current.key[2]:
                        _bad(f"{where}: draft request number does not match current request")
                    d = Draft(_int(obj, "pos", where), _int(obj, "anchor", where), _int(obj, "n", where), _int(obj, "depth", where), _float(obj, "t", where),
                              _array(obj, "drafts", 7, where, _as_int), _array(obj, "p", 7, where),
                              _array(obj, "z", 8, where), _array(obj, "m", 8, where), _array(obj, "lp", 8, where))
                    if d.depth < 0 or d.n < 0 or not math.isfinite(d.t):
                        _bad(f"{where}: invalid draft depth, n, or temperature")
                    current.drafts.append(d)
                elif ev == "verify":
                    _require(obj, ("req", "start", "rows", "held", "copy", "t_ns", "v_ns", "ids", "sampled"), where)
                    if current is None:
                        current = Request((file_index, session_no, _int(obj, "req", where)), {}, "greedy")
                        sessions.append(current)
                    if _int(obj, "req", where) != current.key[2]:
                        _bad(f"{where}: verify request number does not match current request")
                    rows = _int(obj, "rows", where)
                    if rows < 1:
                        _bad(f"{where}: verify rows must be positive")
                    current.verifies.append(Verify(_int(obj, "start", where), rows, _int(obj, "held", where), _int(obj, "copy", where),
                                                    _int(obj, "t_ns", where), _int(obj, "v_ns", where),
                                                    _array(obj, "ids", rows, where, _as_int), _array(obj, "sampled", rows, where, _as_int)))
                else:
                    _require(obj, ("req", "pos", "t_ns"), where)
                    if current is None:
                        current = Request((file_index, session_no, _int(obj, "req", where)), {}, "greedy")
                        sessions.append(current)
                    if _int(obj, "req", where) != current.key[2]:
                        _bad(f"{where}: release request number does not match current request")
                    current.released = True
                    # The session open data is assigned below from its boundary record.
                    current = None
            else:
                # Unknown event names are not trace records and can desynchronize requests.
                _bad(f"{where}: unknown event {ev!r}")
        # Session markers are replaced by open metadata when parsing; reconstruct from file order below.
        # Keep session opens for request calibration checks.
        opens = []
        for line_no, line in enumerate(raw_lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(record, dict) and record.get("ev") == "open":
                opens.append(record)
        session_open = 0
        last_open = None
        for item in sessions:
            if item is None:
                last_open = opens[session_open]
                session_open += 1
            else:
                item.open = last_open or {}
                if item.verifies:
                    requests.append(item)
                else:
                    skipped += 1
    load.last_skipped = skipped
    for req in requests:
        temps = {d.t for d in req.drafts}
        if len(temps) > 1:
            _bad(f"mixed temperatures within request {req.key}")
        req.mode = "sampled" if temps and max(temps) > 0 else "greedy"
    return requests


def _leading(v: Verify) -> int:
    count = 0
    for i in range(min(v.rows - 1, len(v.sampled) - 1)):
        if v.ids[i + 1] != v.sampled[i]:
            break
        count += 1
    return 1 + count


def committed(req: Request) -> tuple[dict[int, int], list[int], dict]:
    seq: dict[int, int] = {}
    kept: list[int] = []
    stats = {"accept_rule": 0, "pending": 0, "overwrite": 0}
    bad = None
    for r, verify in enumerate(req.verifies):
        count = req.verifies[r + 1].start - verify.start if r + 1 < len(req.verifies) else _leading(verify)
        if not 1 <= count <= verify.rows:
            _bad(f"request {req.key} round {r}: kept={count} outside 1..{verify.rows}")
        kept.append(count)
        if verify.held > 0 and r + 1 < len(req.verifies) and count != _leading(verify):
            stats["accept_rule"] += 1; bad = bad or (r, "accept_rule")
        if r + 1 < len(req.verifies) and req.verifies[r + 1].ids[0] != verify.sampled[count - 1]:
            stats["pending"] += 1; bad = bad or (r, "pending")
        for i in range(count):
            pos, token = verify.start + i, verify.ids[i]
            if pos in seq and seq[pos] != token:
                stats["overwrite"] += 1; bad = bad or (r, "overwrite")
            seq[pos] = token
    if req.verifies:
        last = req.verifies[-1]
        pos, token = last.start + kept[-1], last.sampled[kept[-1] - 1]
        if pos in seq and seq[pos] != token:
            stats["overwrite"] += 1; bad = bad or (len(kept) - 1, "overwrite")
        seq[pos] = token
    if bad:
        _bad(f"request {req.key} round {bad[0]}: {bad[1]} consistency check failed (stats={stats})")
    return seq, kept, stats


def blocks(req: Request, seq: dict[int, int]) -> tuple[list[Block], dict]:
    result: list[Block] = []
    stats = {"anchor_mismatch": 0, "censored": 0}
    for draft in req.drafts:
        if seq.get(draft.pos) != draft.anchor:
            stats["anchor_mismatch"] += 1
            continue
        n = min(SLOTS, draft.depth)
        if n <= 0:
            continue
        L = 0
        censored = False
        for j in range(n):
            actual = seq.get(draft.pos + 1 + j)
            if actual is None:
                censored = True
                break
            if draft.drafts[j] != actual:
                break
            L += 1
        if censored:
            stats["censored"] += 1
            continue
        result.append(Block(req.mode, req.key, draft.pos, L, n, draft.p[:n], draft.z[:n], draft.m[:n], draft.lp[:n]))
    return result, stats


def survival(cond: Iterable[float]) -> list[float]:
    out = []
    product = 1.0
    for value in cond:
        product *= float(value)
        out.append(product)
    return out


def allocate(chain, table, max_rows=12) -> int:
    rows, expected = 1, 1.0
    best, best_k = expected / table[0], 0
    for k, s in enumerate(chain, 1):
        if rows >= max_rows:
            break
        rows += 1; expected += s
        now = expected / table[rows - 1]
        if now > best:
            best, best_k = now, k
    return best_k


def periods(requests) -> dict:
    result = {"greedy": {}, "sampled": {}}
    copies = {"greedy": [], "sampled": []}
    for req in requests:
        vs = req.verifies
        for i, v in enumerate(vs):
            if v.copy > 0:
                copies[req.mode].append((v, req.key))
            if i == 0 or i > len(vs) - 2 or v.copy != 0:
                continue
            ms = (vs[i + 1].t_ns - v.t_ns) / 1e6
            if ms <= 0:
                continue
            result[req.mode].setdefault(v.rows, []).append((ms, v.start))
    result["copy_rounds"] = copies
    return result


def cost_table(per, min_samples=30) -> tuple[list[float], dict]:
    modes = ("greedy", "sampled")
    sampled_counts = {w: len(per.get("sampled", {}).get(w, [])) for w in range(1, 9)}
    source = "sampled" if all(n >= min_samples for n in sampled_counts.values()) else "pooled"
    values: dict[int, float] = {}
    for w in range(1, 9):
        samples = per.get("sampled", {}).get(w, []) if source == "sampled" else [x for mode in modes for x in per.get(mode, {}).get(w, [])]
        if len(samples) >= min_samples:
            values[w] = float(statistics.median(x[0] for x in samples))
    if not values:
        _bad("no width 1..8 has enough periods to measure costs")
    raw = {}
    for w in range(1, 17):
        if w in values:
            raw[w] = values[w]
            continue
        below = [x for x in values if x < w]
        above = [x for x in values if x > w]
        if below and above:
            lo, hi = max(below), min(above)
            raw[w] = values[lo] + (values[hi] - values[lo]) * (w - lo) / (hi - lo)
        elif below:
            lo = max(below); raw[w] = values[lo] * GB10[w - 1] / GB10[lo - 1]
        else:
            hi = min(above); raw[w] = values[hi] * GB10[w - 1] / GB10[hi - 1]
    table = []
    running = -math.inf
    for w in range(1, 17):
        running = max(running, raw[w])
        table.append(round(running, 1))
    if any(not 10 < x < 2000 for x in table):
        _bad("computed cost table has a value outside (10, 2000) ms")
    info = {mode: {} for mode in modes}
    for mode in modes:
        for w in range(1, 17):
            vals = [x[0] for x in per.get(mode, {}).get(w, [])]
            info[mode][str(w)] = {"n": len(vals), "median": statistics.median(vals) if vals else None,
                                  "p10": float(np.percentile(vals, 10)) if vals else None,
                                  "p90": float(np.percentile(vals, 90)) if vals else None}
    pooled = [(ms, rows, ctx) for mode in modes for rows, pairs in per.get(mode, {}).items() if isinstance(rows, int) for ms, ctx in pairs]
    if pooled:
        mat = np.asarray([[1.0, rows, ctx / 1000, rows * ctx / 1000] for ms, rows, ctx in pooled])
        yy = np.asarray([ms for ms, _, _ in pooled])
        coeff = np.linalg.lstsq(mat, yy, rcond=None)[0].tolist()
    else:
        coeff = [None] * 4
    info.update({"source": source, "measured_widths": sorted(values), "fit_ms": dict(zip(("a", "b", "c", "d"), coeff))})
    return table, info


def _sigmoid(x):
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, l2=1e-3, iters=100) -> np.ndarray:
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1], dtype=float); w[0] = 1.0
    penalty = np.eye(X.shape[1]); penalty[-1, -1] = 0.0
    for _ in range(iters):
        mu = np.clip(_sigmoid(X @ w), 1e-9, 1 - 1e-9)
        g = X.T @ (mu - y) + l2 * penalty @ w
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + l2 * penalty + 1e-12 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def ece(pred, y, bins=10):
    pred = np.asarray(pred); y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (pred >= edges[i]) & ((pred <= edges[i + 1]) if i == bins - 1 else (pred < edges[i + 1]))
        if np.any(mask):
            total += np.mean(mask) * abs(float(np.mean(y[mask]) - np.mean(pred[mask])))
    return float(total)


def _features(block, j, kind="feat"):
    return [block.z[j], block.m[j], block.lp[j], 1.0] if kind == "feat" else [block.z[j], 1.0]


def fit_mode(blocks_fit, kind) -> tuple[list[list[float]], list[dict]]:
    if kind not in ("feat", "platt"):
        raise ValueError("kind must be feat or platt")
    weights, stats = [], []
    for j in range(SLOTS):
        selected = [b for b in blocks_fit if b.L >= j and b.n > j]
        y = np.asarray([float(b.L > j) for b in selected])
        X = np.asarray([_features(b, j, kind) for b in selected], dtype=float).reshape((-1, 4 if kind == "feat" else 2))
        positives = int(np.sum(y)); negatives = len(y) - positives
        status = "fit"
        if len(y) < _FIT_MIN_ROWS or positives < 20 or negatives < 20:
            w = np.array([1.0, 0.0, 0.0, 0.0]); status = "identity"
        else:
            learned = fit_logistic(X, y)
            if not np.all(np.isfinite(learned)) or np.any(np.abs(learned) > 100):
                w = np.array([1.0, 0.0, 0.0, 0.0]); status = "identity"
            elif kind == "feat":
                w = learned
            else:
                w = np.array([learned[0], 0.0, 0.0, learned[1]])
        p_asis = np.asarray([b.p[j] for b in selected])
        p_cal = _sigmoid(np.asarray([b.z[j] * w[0] + b.m[j] * w[1] + b.lp[j] * w[2] + w[3] for b in selected]))
        stats.append({"n": len(y), "pos": positives, "status": status, "w": [float(x) for x in w],
                      "ece_asis": ece(p_asis, y), "ece_cal": ece(p_cal, y)})
        weights.append([float(x) for x in w])
    return weights, stats


def cond(block: Block, w) -> list[float]:
    return [float(_sigmoid(w[j][0] * block.z[j] + w[j][1] * block.m[j] + w[j][2] * block.lp[j] + w[j][3])) for j in range(block.n)]


def simulate(blocks, choose, table) -> dict:
    tokens = 0; ms = 0.0
    for block in blocks:
        k = int(choose(block))
        if not 0 <= k <= block.n:
            raise ValueError(f"chooser returned {k} for n={block.n}")
        tokens += min(k, block.L) + 1
        ms += table[k]
    rounds = len(blocks)
    return {"blocks": rounds, "tokens": tokens, "ms": ms, "tokps": 1000 * tokens / ms if ms else 0.0,
            "tok_per_round": tokens / rounds if rounds else 0.0, "ms_per_round": ms / rounds if rounds else 0.0}


def asis(table):
    return lambda b: allocate(survival(b.p), table)


def asis_seed(block):
    return allocate(survival(block.p), GB10)


def cal(w, table):
    return lambda b: allocate(survival(cond(b, w)), table)


def fixed(k):
    return lambda b: min(k, b.n)


def oracle(block):
    return min(block.L, block.n)


def req_report(req: Request, seq: dict[int, int], kept: list[int]) -> dict:
    vs = req.verifies
    rounds = len(vs)
    decode_ms = (vs[-1].t_ns + vs[-1].v_ns - vs[0].t_ns) / 1e6 if rounds else 0.0
    tokens = len(seq)
    drafted = accepted = 0
    for v, k in zip(vs, kept):
        if v.held > 0:
            drafted += max(0, v.rows - 1); accepted += max(0, k - 1)
    sha = hashlib.sha256(np.asarray([seq[q] for q in sorted(seq)], dtype="<u4").tobytes()).hexdigest()
    return {"mode": req.mode, "key": list(req.key), "tokens": tokens, "rounds": rounds, "decode_ms": decode_ms,
            "tokps": 1000 * tokens / decode_ms if decode_ms else 0.0, "tok_per_round": tokens / rounds if rounds else 0.0,
            "ms_per_round": decode_ms / rounds if rounds else 0.0, "drafted": drafted, "accepted": accepted,
            "accept_rate": accepted / drafted if drafted else 0.0, "sha256": sha, "ctx_end": max(seq) if seq else None}


def _prepare(requests):
    built = []
    all_blocks = []
    for req in requests:
        seq, kept, cstats = committed(req)
        blks, bstats = blocks(req, seq)
        built.append((req, seq, kept, cstats, bstats))
        all_blocks.extend(blks)
    return built, all_blocks


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _aggregate(rows):
    total_tokens = sum(x["tokens"] for x in rows); total_ms = sum(x["decode_ms"] for x in rows)
    rounds = sum(x["rounds"] for x in rows); drafted = sum(x["drafted"] for x in rows); accepted = sum(x["accepted"] for x in rows)
    return {"requests": len(rows), "tokps": 1000 * total_tokens / total_ms if total_ms else 0.0,
            "tok_per_round": total_tokens / rounds if rounds else 0.0, "ms_per_round": total_ms / rounds if rounds else 0.0,
            "accept_rate": accepted / drafted if drafted else 0.0}


def _print_json(path, data):
    if path:
        Path(path).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def _report(args):
    reqs = load(args.trace); rows = []
    for req in reqs:
        seq, kept, _ = committed(req); rows.append(req_report(req, seq, kept))
    labels = (args.classes or "").split(",") if args.classes else []
    for i, row in enumerate(rows):
        row["class"] = labels[i] if i < len(labels) else "?"
    shown = rows[args.skip:]
    print("mode class key tokens rounds tok/s tok/round ms/round accept")
    for row in shown:
        print(f"{row['mode']} {row['class']} {row['key']} {row['tokens']} {row['rounds']} {row['tokps']:.3f} {row['tok_per_round']:.3f} {row['ms_per_round']:.3f} {row['accept_rate']:.4f}")
    grouped = {}
    for row in shown:
        grouped.setdefault((row["mode"], row["class"]), []).append(row)
    aggregates = {f"{mode}:{cls}": _aggregate(group) for (mode, cls), group in grouped.items()}
    print("AGGREGATES")
    for key, value in aggregates.items():
        print(f"{key} requests={value['requests']} tok/s={value['tokps']:.3f} tok/round={value['tok_per_round']:.3f} "
              f"ms/round={value['ms_per_round']:.3f} accept={value['accept_rate']:.4f}")
    data = {"requests": shown, "aggregates": aggregates}
    _print_json(args.json, data)
    return 0


def _compare(args):
    ra = load(args.a); rb = load(args.b); ra = ra[args.skip:]; rb = rb[args.skip:]
    if len(ra) != len(rb):
        _bad(f"trace request counts differ after skip: {len(ra)} vs {len(rb)}")
    labels = (args.classes or "").split(",") if args.classes else []
    pairs = []; mismatch = False
    for i, (a, b) in enumerate(zip(ra, rb)):
        sa, ka, _ = committed(a); sb, kb, _ = committed(b)
        aa, bb = req_report(a, sa, ka), req_report(b, sb, kb)
        equal = aa["sha256"] == bb["sha256"]; mismatch |= not equal
        label_index = i + args.skip
        pairs.append({"class": labels[label_index] if label_index < len(labels) else "?", "a": aa, "b": bb, "sha_equal": equal})
    print("MISMATCHES")
    for x in pairs:
        if not x["sha_equal"]: print(f"{x['class']} A={x['a']['key']} B={x['b']['key']} sha differs")
    print("PAIRS")
    for x in pairs:
        if x["sha_equal"]: print(f"{x['class']} A={x['a']['key']} B={x['b']['key']} sha equal")
    aggregates = {}
    modes = sorted({p[s]["mode"] for p in pairs for s in ("a", "b")})
    for mode in modes:
        for cls in sorted({p["class"] for p in pairs} | {"overall"}):
            group = [p for p in pairs if p["a"]["mode"] == mode and (cls == "overall" or p["class"] == cls)]
            if not group: continue
            av = _aggregate([p["a"] for p in group]); bv = _aggregate([p["b"] for p in group])
            aggregates[f"{mode}:{cls}"] = {"a": av, "b": bv, "delta_pct": (bv["tokps"] / av["tokps"] - 1) * 100 if av["tokps"] else 0.0}
    print("AGGREGATES")
    for key, value in aggregates.items():
        av, bv = value["a"], value["b"]
        print(f"{key} A={av['tokps']:.3f} tok/s B={bv['tokps']:.3f} tok/s delta={value['delta_pct']:.3f}% "
              f"tok/round={av['tok_per_round']:.3f}/{bv['tok_per_round']:.3f} "
              f"ms/round={av['ms_per_round']:.3f}/{bv['ms_per_round']:.3f}")
    data = {"pairs": pairs, "aggregates": aggregates}
    _print_json(args.json, data)
    return 4 if mismatch else 0


def _fit(args):
    global _FIT_MIN_ROWS
    _FIT_MIN_ROWS = args.min_rows
    all_paths = args.trace + (args.held_trace or [])
    all_reqs = load(all_paths)
    fit_limit = len(args.trace)
    fit_reqs = [r for r in all_reqs if r.key[0] < fit_limit]
    held_reqs = [r for r in all_reqs if r.key[0] >= fit_limit]
    prepared, all_blocks = _prepare(all_reqs)
    rounds_by_mode = {mode: sum(len(req.verifies) for req in all_reqs if req.mode == mode) for mode in ("greedy", "sampled")}
    blocks_by_mode = {mode: sum(1 for block in all_blocks if block.mode == mode) for mode in ("greedy", "sampled")}
    print(f"loaded {len(all_reqs)} requests; skipped no-verify requests: {load.last_skipped}; "
          f"rounds={rounds_by_mode}; blocks={blocks_by_mode}")
    for req in all_reqs:
        if not args.no_conf and (req.open.get("draft_chances") != 1 or req.open.get("dspark") != 1 or req.open.get("conf_temp") != 1.0):
            _bad(f"request {req.key}: requires draft_chances=1, dspark=1, conf_temp=1.0")
    cost_info = {}
    if args.no_costs:
        table = list(GB10)
    else:
        table, cost_info = cost_table(periods(all_reqs), args.min_samples)
    if args.held_trace:
        fit_keys = {r.key for r in fit_reqs}; held_keys = {r.key for r in held_reqs}
    else:
        fit_keys, held_keys = set(), set()
        counters = {"greedy": 0, "sampled": 0}
        for req in all_reqs:
            mode = req.mode
            (fit_keys if counters[mode] % 2 == 0 else held_keys).add(req.key)
            counters[mode] += 1
    by_mode = {mode: {"fit": [], "held": []} for mode in ("greedy", "sampled")}
    for req, seq, _, _, _ in prepared:
        dest = "fit" if req.key in fit_keys else "held"
        blks, _ = blocks(req, seq)
        by_mode[req.mode][dest].extend(blks)
    conf = {"features": FEATURES}
    fit_info = {}; predicted = {}; identity_modes = 0
    for mode in ("greedy", "sampled"):
        bfit, bheld = by_mode[mode]["fit"], by_mode[mode]["held"]
        if args.no_conf:
            wf = wp = [[1.0, 0.0, 0.0, 0.0] for _ in range(SLOTS)]; sf = sp = []
        else:
            wf, sf = fit_mode(bfit, "feat"); wp, sp = fit_mode(bfit, "platt")
        chosen_kind = "platt"
        if bheld:
            sim_asis = simulate(bheld, asis(table), table)
            sim_feat = simulate(bheld, cal(wf, table), table)
            sim_platt = simulate(bheld, cal(wp, table), table)
            if sim_feat["tokps"] > sim_platt["tokps"]: chosen_kind = "feat"
            sim_chosen = sim_feat if chosen_kind == "feat" else sim_platt
            gain = sim_chosen["tokps"] / sim_asis["tokps"] - 1 if sim_asis["tokps"] else 0.0
        else:
            sim_asis = simulate(bheld, asis(table), table); sim_feat = simulate(bheld, cal(wf, table), table); sim_platt = simulate(bheld, cal(wp, table), table)
            chosen_kind = "platt"; sim_chosen = sim_platt; gain = 0.0
        if gain < args.gate:
            chosen_kind = "identity (gain below gate)"; gain = max(gain, -1.0); sim_chosen = sim_asis; identity_modes += 1
            winning_weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(SLOTS)]
        elif not bfit:
            chosen_kind = "identity (no fit blocks)"; winning_weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(SLOTS)]; identity_modes += 1
        else:
            winning_weights = wf if chosen_kind == "feat" else wp
        if not args.no_conf: conf[mode] = winning_weights
        fit_info[mode] = {"kind": chosen_kind, "slots": sf if chosen_kind == "feat" else sp}
        predicted[mode] = {"fit_blocks": len(bfit), "held_blocks": len(bheld), "asis": sim_asis,
                           "asis_seed": simulate(bheld, asis_seed, table), "cal": sim_chosen,
                           "oracle": simulate(bheld, oracle, table), "fixed": {str(k): simulate(bheld, fixed(k), table) for k in range(8)}, "gain": gain}
    if args.no_costs and (identity_modes == 2 or args.no_conf):
        print("no usable calibration output; exiting 3")
        return 3
    out = {"schema": SCHEMA}
    if not args.no_costs: out["costs"] = [[i + 1, table[i]] for i in range(16)]
    if not args.no_conf: out["conf"] = conf
    out["costs_info"] = cost_info
    out["fit"] = fit_info
    out["predicted"] = predicted
    out["inputs"] = [{"path": str(p), "sha256": _sha(p), "requests": len([r for r in all_reqs if r.key[0] == i]),
                       "rounds": sum(len(r.verifies) for r in all_reqs if r.key[0] == i)} for i, p in enumerate(all_paths)]
    out["created_utc"] = datetime.now(timezone.utc).isoformat()
    out["tool"] = "tools/glm53/recal_fit.py v1"
    Path(args.out).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.out}; greedy={fit_info['greedy']['kind']}, sampled={fit_info['sampled']['kind']}")
    _print_json(args.json, out)
    return 0


def _predict(args):
    reqs = load(args.trace); _, blks = _prepare(reqs)
    if args.recal:
        data = json.loads(Path(args.recal).read_text())
        table = [float(x[1]) for x in data.get("costs", [])] if data.get("costs") else list(GB10)
        weights = data.get("conf") or {"greedy": [[1, 0, 0, 0]] * 7, "sampled": [[1, 0, 0, 0]] * 7}
    else:
        table = list(GB10); weights = {"greedy": [[1, 0, 0, 0]] * 7, "sampled": [[1, 0, 0, 0]] * 7}
    results = {}
    for mode in ("greedy", "sampled"):
        selected = [b for b in blks if b.mode == mode]
        asis_result = simulate(selected, asis(table), table)
        cal_result = simulate(selected, cal(weights[mode], table), table)
        results[mode] = {"asis": asis_result, "cal": cal_result,
                         "cal_over_asis": cal_result["tokps"] / asis_result["tokps"] if asis_result["tokps"] else 0.0}
    heads = [(ms, req, i) for req in reqs for i, v in enumerate(req.verifies)
                                 if v.held > 0 and i in range(1, len(req.verifies) - 1) and v.copy == 0
                                 for ms in [(req.verifies[i + 1].t_ns - v.t_ns) / 1e6] if ms > 0]
    # Head throughput uses kept tokens paired to each eligible period.
    measured_tokens = 0; measured_ms = 0.0
    kept_by = {id(req): committed(req)[1] for req in reqs}
    for ms, req, i in heads:
        measured_tokens += kept_by[id(req)][i]; measured_ms += ms
    measured = 1000 * measured_tokens / measured_ms if measured_ms else 0.0
    data = {"predicted": results, "measured_head_tokps": measured}
    print(f"measured head tok/s={measured:.3f}")
    for mode, r in results.items(): print(f"{mode}: sim_asis/measured={r['asis']['tokps']/measured if measured else 0:.4f} cal/asis={r['cal_over_asis']:.4f}")
    _print_json(args.json, data)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit GLM-5.3 draft-depth recalibration from round traces")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_p = sub.add_parser("fit"); fit_p.add_argument("--trace", nargs="+", required=True); fit_p.add_argument("--held-trace", nargs="+")
    fit_p.add_argument("--out", required=True); fit_p.add_argument("--split", choices=("alternate",), default="alternate")
    fit_p.add_argument("--min-samples", type=int, default=30); fit_p.add_argument("--min-rows", type=int, default=200)
    fit_p.add_argument("--gate", type=float, default=0.005); fit_p.add_argument("--no-costs", action="store_true"); fit_p.add_argument("--no-conf", action="store_true"); fit_p.add_argument("--json")
    report_p = sub.add_parser("report"); report_p.add_argument("--trace", nargs="+", required=True); report_p.add_argument("--classes"); report_p.add_argument("--skip", type=int, default=0); report_p.add_argument("--json")
    compare_p = sub.add_parser("compare"); compare_p.add_argument("--a", nargs="+", required=True); compare_p.add_argument("--b", nargs="+", required=True); compare_p.add_argument("--classes"); compare_p.add_argument("--skip", type=int, default=0); compare_p.add_argument("--json")
    predict_p = sub.add_parser("predict"); predict_p.add_argument("--trace", nargs="+", required=True); predict_p.add_argument("--recal"); predict_p.add_argument("--json")
    args = parser.parse_args(argv)
    if args.command == "fit": return _fit(args)
    if args.command == "report": return _report(args)
    if args.command == "compare": return _compare(args)
    return _predict(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
