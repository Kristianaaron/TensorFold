# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
"""GLM Python qmm fixtures for `tf-cuda-test fp8b-lane <fixture>` (Apache-2.0 reference API)."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

ROWS = (*range(1, 17), 17, 33, 64, 256, 300)


def case(out: Path, n: int, k: int, row0: int, col0: int, full_n: int, full_k: int) -> None:
    from tensorfold.families.glm5_next.cuda.fp8 import make_f8
    from tensorfold.families.glm5_next.cuda.qmm import matmul

    g = torch.Generator().manual_seed(n * 31 + k + row0 * 7 + col0)
    codes = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    scale_inv = torch.rand(((-(-full_n // 128)), (-(-full_k // 128))), generator=g) * 0.01 + 1e-3
    q = make_f8(codes.cuda().view(torch.float8_e4m3fn), scale_inv.cuda().float(),
                torch.arange(row0, row0 + n, device="cuda"), col0)
    arrays = {"codes": codes, "scale_inv": scale_inv.float()}
    for m in ROWS:
        x = (torch.randn((m, k), generator=g) * 0.5).to(torch.bfloat16)
        arrays[f"x{m}"] = x
        arrays[f"y{m}"] = matmul(x.cuda(), q).cpu()
        arrays[f"y32_{m}"] = matmul(x.cuda(), q, f32=True).cpu()
    torch.cuda.synchronize()
    save(out, {"n": n, "k": k, "rows": ",".join(map(str, ROWS)), "row0": row0, "col0": col0,
              "full_n": full_n, "full_k": full_k, "math": "bf16(e4m3 * scale_inv); ieee fp32 accumulate"}, arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=576)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--row0", type=int, default=576)
    ap.add_argument("--col0", type=int, default=0)
    ap.add_argument("--full-n", type=int, default=1152)
    ap.add_argument("--full-k", type=int, default=6144)
    a = ap.parse_args()
    case(Path(a.out), a.n, a.k, a.row0, a.col0, a.full_n, a.full_k)


if __name__ == "__main__":
    main()
