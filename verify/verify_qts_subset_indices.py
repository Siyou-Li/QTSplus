#!/usr/bin/env python3
"""
Verify that QTS outputs are a subset of input vision embeddings (no reencoding)
and that returned indices are accurate.

Checks performed per batch element b:
  - Z_b equals Xv[b, idx_b] exactly (within tolerance) when reencode=False and mode!='train'
  - idx_b equals the ascending-sorted top-k indices of r[b] where k=n[b]
  - idx_b is strictly increasing, in-range, and has no duplicates

This script uses synthetic inputs; the property is data-independent.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch


def add_src_to_path():
    # Ensure repo root is on sys.path so we can import src.*
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../QTSplus
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify QTS subset + index correctness (no reencode)")
    p.add_argument("--mode", choices=["infer", "train"], default="infer", help="QTS forward mode to test")
    p.add_argument("--batch", type=int, default=1, help="Batch size B")
    p.add_argument("--m", type=int, default=512, help="Number of vision tokens M")
    p.add_argument("--d", type=int, default=1024, help="Embedding dim D")
    p.add_argument("--l", type=int, default=64, help="Text tokens L (for scoring)")
    p.add_argument("--heads", type=int, default=8, help="QTS attention heads")
    p.add_argument("--nmax", type=int, default=2560, help="QTS nmax budget upper bound")
    p.add_argument("--rho_min", type=float, default=0.05, help="QTS min compression ratio")
    p.add_argument("--rho_max", type=float, default=0.5, help="QTS max compression ratio")
    p.add_argument("--tau_s", type=float, default=0.1, help="QTS threshold temperature")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    p.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for equality check")
    p.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for equality check")
    # Qwen weights are the default init; no toggle
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    add_src_to_path()

    # Import after path fix
    from src.model.qts_plus_tokenizer import (
        QTSplusTokenizer, QTSplusTokenizerConfig,
    )

    # Construct QTS with reencode disabled
    cfg = QTSplusTokenizerConfig(
        embedding_dim=args.d,
        n_heads=args.heads,
        tau_s=args.tau_s,
        nmax=args.nmax,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        block_dropout=0.0,
        reencode=False,                 # IMPORTANT: disable reencoding
        scoring_layers=1,
        reencode_layers=1,
        project_text_if_needed=False,
    )

    qts = QTSplusTokenizer(cfg).to(args.device)
    qts.eval()

    B, M, D, L = args.batch, args.m, args.d, args.l
    Xv = torch.randn(B, M, D, device=args.device)
    Qt = torch.randn(B, L, D, device=args.device)

    # Run selected mode
    mode = args.mode
    # Use a tiny tolerance by default in train mode (due to numeric ops)
    rtol = args.rtol if (args.rtol > 0 or mode == "infer") else 1e-6
    atol = args.atol if (args.atol > 0 or mode == "infer") else 1e-7
    
    with torch.no_grad():
        out = qts(Xv, Qt, mode=mode)

    Z_list: List[torch.Tensor] = out["Z"]
    idx_list: List[torch.Tensor] = out["indices"]
    r = out["r"]  # [B, M]
    n = out["n"]  # [B]

    assert len(Z_list) == B and len(idx_list) == B, "Output list lengths must equal batch size"

    ok = True
    messages: List[str] = []

    for b in range(B):
        zb = Z_list[b]                 # [k_b, D]
        idx = idx_list[b].to(torch.long)  # [k_b]
        kb = idx.numel()
        # Shape checks
        if kb != int(n[b].item()):
            ok = False
            messages.append(f"[b={b}] n mismatch: idx={kb} vs n[b]={int(n[b].item())}")

        # Index invariants
        if kb > 0:
            if not torch.all(idx[1:] > idx[:-1]):  # strictly increasing order (sorted)
                ok = False
                messages.append(f"[b={b}] indices not strictly increasing: {idx.tolist()}")
            if idx.min().item() < 0 or idx.max().item() >= M:
                ok = False
                messages.append(f"[b={b}] indices out of bounds [0,{M-1}]: {idx.tolist()}")
            if torch.unique(idx).numel() != kb:
                ok = False
                messages.append(f"[b={b}] duplicate indices present: {idx.tolist()}")

        # Subset/value check
        if kb > 0:
            gathered = Xv[b, idx]
            if not torch.allclose(zb, gathered, rtol=rtol, atol=atol):
                ok = False
                max_abs = (zb - gathered).abs().max().item()
                messages.append(
                    f"[b={b}] Z != Xv[idx]; max_abs_diff={max_abs:.3e}, k={kb}, rtol={rtol}, atol={atol}"
                )

        # Top-k correctness: idx should equal sorted top-k of r[b]
        if kb > 0:
            topk = torch.topk(r[b], k=kb, dim=0).indices
            topk_sorted, _ = torch.sort(topk)
            if not torch.equal(idx.cpu(), topk_sorted.cpu()):
                ok = False
                messages.append(
                    f"[b={b}] indices != sorted(topk(r)): got={idx.tolist()}, expected={topk_sorted.tolist()}"
                )

    if ok:
        tag = "infer" if mode == "infer" else "train"
        print(f"QTS subset + index verification PASSED (reencode disabled, {tag} mode).")
        return 0
    else:
        print("QTS subset + index verification FAILED:")
        for m in messages:
            print(" - ", m)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
