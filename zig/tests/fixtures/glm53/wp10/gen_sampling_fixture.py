#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
"""Generate the checked-in WP10 sampler vectors from bertholomus/TensorFold exact_sampling.py."""
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.environ["TENSORFOLD_PY_SRC"])  # src/ of bertholomus/TensorFold at ddfad35
from tensorfold.engine.exact_sampling import Sampling, choose_rows, uniform_rows

cases = [
    {"seed": 5, "position": 17, "ids": [0, 2, 4, 8], "values": [1.25, 1.25, -0.5, 2.0], "top_k": 0, "top_p": 1.0, "min_p": 0.0, "temperature": 0.7},
    {"seed": 123456789, "position": 987, "ids": [17, 3, 99, 42, 11, 7, 100, 5], "values": [2.5, 1.0, 2.5, -1.0, 0.25, 2.0, 0.5, 1.5], "top_k": 6, "top_p": 0.82, "min_p": 0.04, "temperature": 1.3},
    {"seed": 77, "position": 2500, "ids": list(range(12)), "values": [1.2, -0.4, 2.3, 0.8, 1.9, -1.1, 0.2, 1.5, 0.4, 2.0, -0.7, 1.0], "top_k": 0, "top_p": 0.91, "min_p": 0.0, "temperature": 0.85},
]
for case in cases:
    ids = np.asarray([case["ids"]], dtype=np.int64)
    vals = np.asarray([case["values"]], dtype=np.float32)
    s = Sampling(seed=case["seed"], temperature=case["temperature"], top_k=case["top_k"], top_p=case["top_p"], min_p=case["min_p"])
    case["uniforms"] = uniform_rows(s.seed, np.asarray([case["position"]]), ids)[0].tolist()
    case["chosen"] = choose_rows(vals, ids, [case["position"]], s)[0]
print(json.dumps({"source": "bertholomus/TensorFold exact_sampling.py @ ddfad35", "cases": cases}, indent=2, sort_keys=True))
