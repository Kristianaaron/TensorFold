#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
"""Check virtual block-coordinate conversion and CUDA task entry instrumentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "zig/kernels/cuda"
VIRTUAL_HEADER = CUDA / "tf_vblock.cuh"

# MK2 sources named by the task plus the in-scope headers that read coordinates.
STEP2_FILES = (
    "exl3_experts.cu",
    "exl3_linear.cu",
    "exl3_candidate.cu",
    "exl3_block/x3ld_epi.cu",
    "exl3_block/exl3_experts.cu",
    "exl3_block/experts_grouped.cuh",
    "exl3_block/x3ld.cu",
    "exl3_block/x3gm.cu",
    "exl3_block/x3gm3.cu",
    "exl3_block/x3pf.cu",
    "exl3_block/x3gm2.cuh",
    "exl3_prompt.cuh",
    "glm_dsa/fp8_mma.cu",
    "glm_dsa/glue.cu",
    "glm_dsa/head.cu",
    "glm_dsa/index.cu",
    "glm_dsa/mla.cu",
    "glm_dsa/rope.cuh",
    "glm_dsa/tasks.cuh",
    "glm_glue.cu",
    "glm_glue_attn.cuh",
    "glm_glue_bmm.cuh",
    "glm_glue_hc.cuh",
    "glm_glue_index.cuh",
    "glm_glue_latent.cuh",
    "glm_glue_moe.cuh",
    "glm_glue_norm.cuh",
    "glm_glue_router.cuh",
    "glm_glue_rows.cuh",
    "qmmf.cu",
)

GLOBAL_FILES = tuple(
    f
    for f in STEP2_FILES
    if f.endswith((".cu", ".cuh"))
)
GLOBAL_FILES = tuple(dict.fromkeys(GLOBAL_FILES + (
    "exl3_linear.cu",
    "exl3_candidate.cu",
    "exl3_block/x3ld_epi.cu",
    "exl3_block/exl3_experts.cu",
    "exl3_block/experts_grouped.cuh",
    "exl3_block/x3ld.cu",
    "exl3_block/x3gm.cu",
    "exl3_block/x3gm3.cu",
    "exl3_block/x3pf.cu",
)))
COORDINATE = re.compile(r"\b(?:blockIdx|gridDim)\b")
GLOBAL = re.compile(r"__global__\b")
ENTRY = re.compile(r"(?:[ \t]*\\\r?\n[ \t]*)*\s*TF_VBLOCK_ENTER\s*\(")


def check_coordinates() -> int:
    errors = 0
    for rel in STEP2_FILES:
        path = CUDA / rel
        if not path.is_file():
            print(f"FAIL missing step-2 file: {path.relative_to(ROOT)}")
            errors += 1
            continue
        if path == VIRTUAL_HEADER:
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            # cudaLaunchConfig_t's host-side field is named gridDim; it is not
            # the device builtin and must remain intact for ordinary launches.
            device_line = re.sub(r"\b\w+\.gridDim\b", "", line)
            if COORDINATE.search(device_line):
                print(f"FAIL raw block coordinate: {path.relative_to(ROOT)}:{line_no}: {line.strip()}")
                errors += 1
    return errors


def check_entries() -> tuple[int, int]:
    errors = 0
    count = 0
    for rel in GLOBAL_FILES:
        path = CUDA / rel
        if not path.is_file():
            continue
        text = path.read_text()
        for match in GLOBAL.finditer(text):
            brace = text.find("{", match.end())
            semicolon = text.find(";", match.end())
            if brace < 0 or (semicolon >= 0 and semicolon < brace):
                continue  # declaration or explicit template instantiation
            count += 1
            if not ENTRY.match(text[brace + 1 :]):
                line_no = text.count("\n", 0, match.start()) + 1
                print(f"FAIL missing first-statement entry: {path.relative_to(ROOT)}:{line_no}")
                errors += 1
    return errors, count


def check_task_intrinsics() -> int:
    """Task bodies and their in-scope helpers must use virtual dimensions/barriers."""
    errors = 0
    for rel in STEP2_FILES:
        path = CUDA / rel
        if not path.is_file():
            continue
        source = path.read_text()
        # Wrapper bodies retain ordinary whole-CTA barriers; inspect the
        # remaining device helpers and task bodies where virtual dimensions apply.
        spans: list[tuple[int, int]] = []
        for match in GLOBAL.finditer(source):
            brace = source.find("{", match.end())
            semicolon = source.find(";", match.end())
            if brace < 0 or (semicolon >= 0 and semicolon < brace):
                continue
            depth = 1
            end = brace + 1
            while end < len(source) and depth:
                if source[end] == "{":
                    depth += 1
                elif source[end] == "}":
                    depth -= 1
                end += 1
            spans.append((brace + 1, end - 1))
        chars = list(source)
        for start, end in spans:
            for i in range(start, end):
                if chars[i] != "\n":
                    chars[i] = " "
        for line_no, line in enumerate("".join(chars).splitlines(), 1):
            # A host-side cudaLaunchConfig_t field is spelled `.blockDim` too,
            # but is not the CUDA device builtin.
            device_line = re.sub(r"\.blockDim\b", "", line)
            if re.search(r"\bblockDim\b", device_line):
                print(f"FAIL raw blockDim in task source: {path.relative_to(ROOT)}:{line_no}: {line.strip()}")
                errors += 1
            if re.search(r"\b__syncthreads\s*\(", line):
                print(f"FAIL raw __syncthreads in task source: {path.relative_to(ROOT)}:{line_no}: {line.strip()}")
                errors += 1
    return errors


def main() -> int:
    errors = check_coordinates()
    entry_errors, entry_count = check_entries()
    errors += entry_errors
    errors += check_task_intrinsics()
    if errors:
        print(f"FAIL task ABI: {errors} issue(s); checked {entry_count} global definitions")
        return 1
    print(
        "PASS task ABI: no raw device coordinates, blockDim, or __syncthreads in task sources "
        "(host cudaLaunchConfig_t fields retained); "
        f"{entry_count} global definitions enter virtual block context"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
