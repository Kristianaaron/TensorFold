#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
# Purpose: fit DSpark draft confidence temperature from trace data.
# Input: JSONL pair records {"conf": float, "accepted": 0|1} or round records
# {"conf": [float, ...], "accepted": leading accepted draft count}.
# Output: calibration metrics, optional per-slot round metrics, and a switch value.
# This script fits T for TF_GLM53_DRAFT_CONF_TEMP.

import argparse
import json
import math
import random
import sys


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def logit(probability):
    bounded = min(max(probability, 1e-7), 1.0 - 1e-7)
    return math.log(bounded / (1.0 - bounded))


def log_sigmoid(value):
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def parse_records(path):
    pairs = []
    slots = []
    has_round = False
    try:
        source = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise ValueError("cannot open input: " + str(exc)) from exc
    with source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
                if not isinstance(item, dict) or "conf" not in item or "accepted" not in item:
                    raise ValueError("expected object with conf and accepted")
                confidences = item["conf"]
                accepted = item["accepted"]
                if isinstance(confidences, list):
                    has_round = True
                    if isinstance(accepted, bool) or not isinstance(accepted, int):
                        raise ValueError("round accepted must be an integer")
                    if accepted < 0 or accepted > len(confidences):
                        raise ValueError("round accepted is outside the confidence array")
                    parsed_confidences = []
                    for confidence in confidences:
                        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                            raise ValueError("conf values must be numbers")
                        confidence = float(confidence)
                        if not math.isfinite(confidence):
                            raise ValueError("conf values must be finite")
                        parsed_confidences.append(confidence)
                    while len(slots) < len(confidences):
                        slots.append([])
                    for slot, confidence in enumerate(parsed_confidences):
                        if slot > accepted:
                            break
                        outcome = 1 if slot < accepted else 0
                        pair = (logit(confidence), outcome, confidence, slot)
                        pairs.append(pair)
                        slots[slot].append(pair)
                else:
                    if isinstance(confidences, bool) or not isinstance(confidences, (int, float)):
                        raise ValueError("pair conf must be a number or a list")
                    if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted not in (0, 1):
                        raise ValueError("pair accepted must be 0 or 1")
                    confidence = float(confidences)
                    if not math.isfinite(confidence):
                        raise ValueError("conf must be finite")
                    pairs.append((logit(confidence), int(accepted), confidence, None))
            except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("line " + str(line_number) + ": " + str(exc)) from exc
    return pairs, slots, has_round


def nll(pairs, temperature):
    total = 0.0
    for z_value, accepted, _confidence, _slot in pairs:
        scaled = z_value / temperature
        total -= log_sigmoid(scaled if accepted else -scaled)
    return total / len(pairs)


def ece(pairs, temperature, bins):
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    accepted_sums = [0] * bins
    for z_value, accepted, _confidence, _slot in pairs:
        probability = sigmoid(z_value / temperature)
        bin_index = min(int(probability * bins), bins - 1)
        counts[bin_index] += 1
        confidence_sums[bin_index] += probability
        accepted_sums[bin_index] += accepted
    total = len(pairs)
    return sum(
        (count / total) * abs(confidence_sums[index] / count - accepted_sums[index] / count)
        for index, count in enumerate(counts)
        if count
    )


def fit(pairs, grid_lo, grid_hi, grid_step):
    count = round((grid_hi - grid_lo) / grid_step)
    temperatures = [grid_lo + index * grid_step for index in range(count + 1)]
    return min(temperatures, key=lambda temperature: (nll(pairs, temperature), abs(temperature - 1.0)))


def self_test():
    rng = random.Random(0)
    pairs = []
    for _ in range(20000):
        z_value = rng.uniform(-4.0, 4.0)
        accepted = int(rng.random() < sigmoid(z_value / 2.0))
        confidence = sigmoid(z_value)
        pairs.append((logit(confidence), accepted, confidence, None))
    best_temperature = fit(pairs, 0.5, 3.0, 0.01)
    before = ece(pairs, 1.0, 15)
    after = ece(pairs, best_temperature, 15)
    if abs(best_temperature - 2.0) <= 0.1 and after < before:
        print("self-test ok")
        return 0
    print("self-test failed: T={:.3f}, ECE before={:.6f}, after={:.6f}".format(best_temperature, before, after))
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit DSpark confidence temperature from trace JSONL.")
    parser.add_argument("data", nargs="?", help="JSONL pair or round records")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--grid-lo", type=float, default=0.5)
    parser.add_argument("--grid-hi", type=float, default=3.0)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.data is None:
        parser.error("data.jsonl is required unless --self-test is used")
    if args.bins < 1 or not all(math.isfinite(value) for value in (args.grid_lo, args.grid_hi, args.grid_step)):
        parser.error("bins must be positive and grid values must be finite")
    if args.grid_lo <= 0.0 or args.grid_hi < args.grid_lo or args.grid_step <= 0.0:
        parser.error("grid requires 0 < lo <= hi and step > 0")
    try:
        pairs, slots, has_round = parse_records(args.data)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not pairs:
        print("no usable pairs", file=sys.stderr)
        return 2
    best_temperature = fit(pairs, args.grid_lo, args.grid_hi, args.grid_step)
    before_nll = nll(pairs, 1.0)
    after_nll = nll(pairs, best_temperature)
    before_ece = ece(pairs, 1.0, args.bins)
    after_ece = ece(pairs, best_temperature, args.bins)
    print("pairs: {}".format(len(pairs)))
    print("base acceptance rate: {:.6f}".format(sum(pair[1] for pair in pairs) / len(pairs)))
    print("T=1: NLL {:.6f}, ECE {:.6f}".format(before_nll, before_ece))
    print("best T: {:.3f}".format(best_temperature))
    print("best: NLL {:.6f}, ECE {:.6f}".format(after_nll, after_ece))
    if has_round:
        print("slot\tn\tmean conf\taccept rate\tbefore\tafter")
        for slot, slot_pairs in enumerate(slots):
            if not slot_pairs:
                continue
            print("{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                slot,
                len(slot_pairs),
                sum(pair[2] for pair in slot_pairs) / len(slot_pairs),
                sum(pair[1] for pair in slot_pairs) / len(slot_pairs),
                sum(sigmoid(pair[0]) for pair in slot_pairs) / len(slot_pairs),
                sum(sigmoid(pair[0] / best_temperature) for pair in slot_pairs) / len(slot_pairs),
            ))
    if best_temperature == args.grid_lo or best_temperature == args.grid_hi:
        print("warning: best T is at the grid edge")
    print("TF_GLM53_DRAFT_CONF_TEMP={:.2f}".format(best_temperature))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
