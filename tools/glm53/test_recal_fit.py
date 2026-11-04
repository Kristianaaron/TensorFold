# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import recal_fit as rf


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def synth(path, requests):
    lines = [{"ev": "open", "schema": rf.TRACE_SCHEMA, "draft_chances": 1, "live_costs": 1,
              "lane_chances": 0, "copyd": 0, "copy_rows": 8, "verify_max": 16,
              "conf_temp": 1.0, "dspark": 1, "recal": "0" * 64, "sweep": 1}]
    for req_id, (target, t, k_of_round, match_fn, z_fn) in enumerate(requests):
        P = 0
        clock = 0
        round_no = 0
        while P + 9 < len(target):
            z = [float(z_fn(P, j)) for j in range(8)]
            drafts = [target[P + 1 + j] if match_fn(P, j) else target[P + 1 + j] + 1000 for j in range(7)]
            p = [sigmoid(x) for x in z[:7]]
            lines.append({"ev": "draft", "req": req_id, "pos": P, "anchor": target[P], "n": 7,
                          "depth": 7, "t": t, "drafts": drafts, "p": p, "z": z,
                          "m": [1.0] * 8, "lp": [-0.5] * 8})
            k = int(k_of_round(round_no))
            rows = k + 1
            ids = [target[P]] + drafts[:k]
            sampled = [target[P + 1 + i] for i in range(rows)]
            kept = 1
            while kept <= k and ids[kept] == sampled[kept - 1]:
                kept += 1
            lines.append({"ev": "verify", "req": req_id, "start": P, "rows": rows,
                          "held": 1, "copy": 0, "t_ns": clock, "v_ns": 1000,
                          "ids": ids, "sampled": sampled})
            clock += int((80 + 15 * k) * 1e6)
            P += kept
            round_no += 1
        z = [float(z_fn(P, j)) for j in range(8)]
        drafts = [target[min(P + 1 + j, len(target) - 1)] if match_fn(P, j)
                  else target[min(P + 1 + j, len(target) - 1)] + 1000 for j in range(7)]
        p = [sigmoid(x) for x in z[:7]]
        lines.append({"ev": "draft", "req": req_id, "pos": P, "anchor": target[P], "n": 7,
                      "depth": 7, "t": t, "drafts": drafts, "p": p, "z": z,
                      "m": [1.0] * 8, "lp": [-0.5] * 8})
        lines.append({"ev": "release", "req": req_id, "pos": P, "t_ns": clock})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def rand_match(seed, offset=0.0):
    rng = random.Random(seed)
    vals = {}

    def z_fn(P, j):
        key = (P, j)
        if key not in vals:
            vals[key] = rng.uniform(-3, 3)
        return vals[key]

    match_rng = random.Random(seed + 10000)

    def match_fn(P, j):
        z = z_fn(P, j)
        return match_rng.random() < sigmoid(z + offset)

    return match_fn, z_fn


def test_committed_and_labels(tmp_path):
    target = list(range(10, 400))
    path = tmp_path / "trace.jsonl"
    synth(path, [(target, 0.0, lambda r: r % 8, lambda P, j: j < 3, lambda P, j: 0.0)])
    req = rf.load([path])[0]
    seq, kept, stats = rf.committed(req)
    assert list(seq.values()) == target[:len(seq)]
    blks, bstats = rf.blocks(req, seq)
    assert all(b.L == 3 for b in blks)
    assert bstats["censored"] > 0
    assert stats == {"accept_rule": 0, "pending": 0, "overwrite": 0}


def test_accept_rule_violation_exits(tmp_path):
    target = list(range(10, 400)); path = tmp_path / "trace.jsonl"
    synth(path, [(target, 0.0, lambda r: 7, lambda P, j: j < 3, lambda P, j: 0.0)])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    first = next(x for x in lines if x["ev"] == "verify")
    first["ids"][1] += 100
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    req = rf.load([path])[0]
    with pytest.raises(SystemExit) as exc:
        rf.committed(req)
    assert exc.value.code == 2


def test_allocate_vector():
    cond = [0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert np.allclose(rf.survival(cond), [0.9, 0.72, 0.36, 0.18, 0.09, 0.045, 0.0225])
    assert rf.allocate(rf.survival(cond), rf.GB10) == 3
    assert rf.allocate([1] * 7, rf.GB10) == 7
    assert rf.allocate([0] * 7, rf.GB10) == 0


def test_irls_recovers():
    rng = np.random.default_rng(1)
    n = 20000
    X = np.column_stack((rng.normal(0, 1.5, n), rng.uniform(0, 3, n), rng.uniform(-3, 0, n), np.ones(n)))
    true = np.array([0.8, 0.3, 0.5, -0.2])
    prob = 1 / (1 + np.exp(-(X @ true)))
    y = rng.binomial(1, prob)
    assert np.max(np.abs(rf.fit_logistic(X, y) - true)) < 0.1


def test_cost_table():
    rng = random.Random(3)
    per = {"greedy": {}, "sampled": {}}
    for w in range(1, 9):
        n = 3 if w == 5 else 40
        per["sampled"][w] = [(80 + 15 * (w - 1) + rng.uniform(-1, 1), i) for i in range(n)]
    table, info = rf.cost_table(per)
    for w in (1, 2, 3, 4, 6, 7, 8):
        assert abs(table[w - 1] - (80 + 15 * (w - 1))) < 0.6
    assert abs(table[4] - (table[3] + table[5]) / 2) <= 0.1
    for w in range(9, 17):
        ratio = rf.GB10[w - 1] / rf.GB10[7]
        assert abs(table[w - 1] - table[7] * ratio) <= 0.05 * ratio + 0.05 + 1e-9
    assert all(a <= b for a, b in zip(table, table[1:]))
    assert all(10 < x < 2000 for x in table)


def engine_parse_ok(data):
    assert data["schema"] == rf.SCHEMA
    assert data["conf"]["features"] == rf.FEATURES
    for row in data["costs"]:
        assert len(row) == 2 and 1 <= row[0] <= 16 and 10 < row[1] < 2000
    assert [row[0] for row in data["costs"]] == list(range(1, 17))
    costs = [row[1] for row in data["costs"]]
    assert costs == sorted(costs)
    for mode in ("greedy", "sampled"):
        assert len(data["conf"][mode]) == 7
        for w in data["conf"][mode]:
            assert len(w) == 4 and all(math.isfinite(x) and abs(x) <= 100 for x in w)


def test_fit_end_to_end(tmp_path):
    targets = [list(range(10, 1510)) for _ in range(8)]
    greedy = []; sampled = []
    for i, target in enumerate(targets):
        gm, gz = rand_match(20 + i, 0.0); sm, sz = rand_match(50 + i, -1.0)
        greedy.append((target, 0.0, lambda r: r % 8, gm, gz))
        sampled.append((target, 1.0, lambda r: r % 8, sm, sz))
    ga, sa = tmp_path / "greedy.jsonl", tmp_path / "sampled.jsonl"
    synth(ga, greedy); synth(sa, sampled)
    out = tmp_path / "recal.json"
    assert rf.main(["fit", "--trace", str(ga), str(sa), "--out", str(out)]) == 0
    data = json.loads(out.read_text()); engine_parse_ok(data)
    sampled_fit = data["fit"]["sampled"]
    slot = sampled_fit["slots"][0]
    if slot["status"] != "identity":
        assert -1.4 <= slot["w"][3] <= -0.6
    assert data["predicted"]["sampled"]["gain"] > 0
    assert data["fit"]["greedy"]["kind"].startswith("identity") or abs(data["predicted"]["greedy"]["gain"]) < 0.02


def test_gate_exit_3(tmp_path):
    target = list(range(10, 1510)); match, z = rand_match(8, 0.0)
    trace = tmp_path / "greedy.jsonl"
    synth(trace, [(target, 0.0, lambda r: r % 8, match, z) for _ in range(8)])
    out = tmp_path / "no-costs.json"
    assert rf.main(["fit", "--trace", str(trace), "--out", str(out), "--no-costs"]) == 3
    assert not out.exists()


def test_compare_sha(tmp_path):
    target = list(range(10, 400))
    match_a, z_a = rand_match(17)
    match_b, z_b = rand_match(17)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    synth(a, [(target, 0.0, lambda r: r % 8, match_a, z_a)])
    synth(b, [(target, 0.0, lambda r: r % 8, match_b, z_b)])
    report = tmp_path / "comparison.json"
    assert rf.main(["compare", "--a", str(a), "--b", str(b), "--json", str(report)]) == 0
    comparison = json.loads(report.read_text())
    assert all(x["delta_pct"] == 0 for x in comparison["aggregates"].values())
    lines = [json.loads(x) for x in b.read_text().splitlines()]
    verify = next(x for x in lines if x["ev"] == "verify")
    verify["sampled"][0] += 1
    # Keep the altered token coherent at the next fed anchor so sequence checks pass.
    verifies = [x for x in lines if x["ev"] == "verify"]
    if len(verifies) > 1:
        verifies[1]["ids"][0] = verify["sampled"][0]
    b.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert rf.main(["compare", "--a", str(a), "--b", str(b)]) == 4
