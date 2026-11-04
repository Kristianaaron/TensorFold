# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
import argparse
import json
import sys
import os
from collections import defaultdict

TARGETS = {
    "prose": 40,
    "code_fresh": 100,
    "code_edit": 100,
    "structured": 100,
    "tool_json": 100,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSON report files")
    parser.add_argument("--label", nargs="+", default=None,
                        help="Labels for each file (same order as files)")
    args = parser.parse_args()

    if args.label is None:
        args.label = [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    elif len(args.label) != len(args.files):
        print("Error: --label count must match file count", file=sys.stderr)
        sys.exit(1)

    # Per-file data: {label: {set: [results]}}
    file_data = {}
    for fpath, label in zip(args.files, args.label):
        with open(fpath) as f:
            data = json.load(f)
        base = data["base_url"]
        model = data["model"]
        results = data["results"]
        fdata = {}
        for r in results:
            s = r["set"]
            if s not in fdata:
                fdata[s] = []
            fdata[s].append(r)
        file_data[label] = fdata

    # Collect all (label, set) pairs
    all_pairs = []
    for label, fdata in file_data.items():
        for s in fdata:
            all_pairs.append((label, s))

    # Compute per (label, set) stats
    rows = []
    for label, s in all_pairs:
        results = file_data[label][s]
        ok = [r for r in results if r.get("error") is None]
        err = [r for r in results if r.get("error") is not None]

        # per-set ok counts
        n_ok = len(ok)
        n_err = len(err)

        # per-set token/s stats (every ok result; usage matters only for the weighted overall line)
        tok_s_vals = []
        ttft_vals = []
        for r in ok:
            if r.get("tok_s") is not None:
                tok_s_vals.append(r["tok_s"])
            if r.get("ttft_s") is not None:
                ttft_vals.append(r["ttft_s"])

        if tok_s_vals:
            mean_tok_s = sum(tok_s_vals) / len(tok_s_vals)
            min_tok_s = min(tok_s_vals)
            max_tok_s = max(tok_s_vals)
        else:
            mean_tok_s = "-"
            min_tok_s = "-"
            max_tok_s = "-"

        # TTFT mean: use all ok results' ttft_s values
        if ttft_vals:
            mean_ttft = sum(ttft_vals) / len(ttft_vals)
        else:
            mean_ttft = "-"

        # target check
        # determine target from set name
        target = TARGETS.get(s, 100)
        # meets target if mean_tok_s is a number and >= target
        if isinstance(mean_tok_s, float) and mean_tok_s >= target:
            meets = "yes"
        else:
            meets = "no (%.1f of target)" % (mean_tok_s / target if isinstance(mean_tok_s, float) else 0)

        rows.append({
            "label": label,
            "set": s,
            "n_ok": n_ok,
            "n_err": n_err,
            "mean_tok_s": mean_tok_s,
            "min_tok_s": min_tok_s,
            "max_tok_s": max_tok_s,
            "mean_ttft": mean_ttft,
            "meets": meets,
        })

    # Print markdown table
    header = "| label | set | n ok | n errors | mean tok/s | min | max | mean TTFT s | target tok/s | meets target? |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        ttft = r.get("mean_ttft")
        ttft_str = str(ttft) if ttft is not None else "-"
        print("| %s | %s | %d | %d | %s | %s | %s | %s | %d | %s |" %
              (r["label"], r["set"], r["n_ok"], r["n_err"],
               r["mean_tok_s"], r["min_tok_s"], r["max_tok_s"],
               ttft_str, TARGETS.get(r["set"], 100), r["meets"]))

    # Weighted overall mean tok/s across all ok results (weighted by completion_tokens)
    total_weight = 0
    weighted_sum = 0
    for label, fdata in file_data.items():
        for s, results in fdata.items():
            for r in results:
                if r.get("error") is not None:
                    continue
                u = r.get("usage")
                if u is not None and "completion_tokens" in u and u["completion_tokens"] is not None:
                    weighted_sum += r["tok_s"] * u["completion_tokens"]
                    total_weight += u["completion_tokens"]

    if total_weight > 0:
        overall_mean = weighted_sum / total_weight
    else:
        overall_mean = 0.0

    print("\nOverall mean tok/s (weighted by completion_tokens): %.2f" % overall_mean)


if __name__ == "__main__":
    main()
