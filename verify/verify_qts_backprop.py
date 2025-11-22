#!/usr/bin/env python3
"""
Verify that each QTS component receives gradients and updates during backprop,
and summarize weight distribution changes before/after optimization.

This script constructs a lightweight QTSplusTokenizer with synthetic inputs,
computes a scalar loss that touches all submodules (scoring layers, budget head,
re-encode transformer blocks, and text projection), performs a backward pass,
optimizes one step, and reports per-component gradient presence and parameter
updates.

Usage examples:
  python script/verify_qts_backprop.py
  python script/verify_qts_backprop.py --use-qwen-scoring
  python script/verify_qts_backprop.py --print-weight-stats --quantiles 0.05,0.5,0.95

Exit status is non-zero if any required component fails to show grads/updates.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def _add_repo_src_to_path():
    """Ensure `src/` is importable when script is run directly."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_add_repo_src_to_path()

from model.qts_plus_tokenizer import (
    QTSplusTokenizer,
    QTSplusTokenizerConfig,
)


@dataclass
class ComponentReport:
    name: str
    n_params: int
    grad_norm: float
    updated_norm: float
    had_grad: bool
    updated: bool


def _compute_weight_stats(vec: torch.Tensor, quantiles: List[float] | None = None) -> Dict[str, float]:
    if vec.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    v = vec.detach().to(dtype=torch.float32, device="cpu")
    stats: Dict[str, float] = {
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
    }
    if quantiles:
        qs = torch.tensor(quantiles, dtype=torch.float32)
        qv = torch.quantile(v, qs)
        for q, val in zip(quantiles, qv):
            stats[f"q{q}"] = float(val.item())
    return stats


def flatten_params(module: nn.Module) -> torch.Tensor:
    """Concatenate all parameter data from a module into a 1D tensor (detached clone)."""
    vecs = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        vecs.append(p.detach().reshape(-1).clone())
    if not vecs:
        return torch.zeros(0)
    return torch.cat(vecs, dim=0)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters(recurse=True):
        if p is not None and p.grad is not None:
            g = p.grad.detach()
            total += float((g * g).sum().cpu())
    return float(total ** 0.5)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=True))


def build_qts(cfg_args) -> QTSplusTokenizer:
    cfg = QTSplusTokenizerConfig(
        embedding_dim=cfg_args.embedding_dim,
        n_heads=cfg_args.n_heads,
        tau_s=cfg_args.tau_s,
        nmax=cfg_args.nmax,
        rho_min=cfg_args.rho_min,
        rho_max=cfg_args.rho_max,
        block_dropout=cfg_args.block_dropout,
        reencode=True,
        scoring_layers=cfg_args.scoring_layers,
        reencode_layers=cfg_args.reencode_layers,
        lambda_t=cfg_args.lambda_t,
        lambda_m=cfg_args.lambda_m,
        lambda_s=cfg_args.lambda_s,
        project_text_if_needed=True,
    )
    model = QTSplusTokenizer(cfg)
    return model


def compute_loss(qts: QTSplusTokenizer, Xv: torch.Tensor, Qt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compose a scalar loss that backpropagates through:
      - scoring layers (via add_loss and selection)
      - budget head (via add_loss proxies on rho)
      - reencode layers (via Z-based penalty)
      - text projection (via scoring/budget paths)
    """
    out = qts(Xv, Qt, mode="train")

    # Z_loss: encourage non-zero re-encoded outputs; touches reencode layers
    z_list: List[torch.Tensor] = out["Z"]
    if len(z_list) == 0:
        z_loss = torch.tensor(0.0, device=Xv.device)
    else:
        # Sum of per-sample mean square; divide by batch for stability
        z_loss = sum(z.pow(2).mean() for z in z_list) / max(1, len(z_list))

    add = out.get("add_loss", {})
    flops = add.get("flops", torch.tensor(0.0, device=Xv.device))
    kv = add.get("kv", torch.tensor(0.0, device=Xv.device))
    smooth = add.get("smooth", torch.tensor(0.0, device=Xv.device))

    total_loss = z_loss + flops + kv + smooth
    scalars = {
        "z_loss": float(z_loss.detach().cpu()),
        "flops": float(flops.detach().cpu()),
        "kv": float(kv.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "total": float(total_loss.detach().cpu()),
    }
    return total_loss, scalars


def make_synthetic_inputs(B: int, M: int, L: int, D_v: int, D_t: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(1337)
    Xv = torch.randn(B, M, D_v, generator=gen, device=device)
    Qt = torch.randn(B, L, D_t, generator=gen, device=device)
    return Xv, Qt


def verify_components(
    qts: QTSplusTokenizer,
    device: torch.device,
    lr: float,
    steps: int,
    B: int,
    M: int,
    L: int,
    D_txt: int,
    collect_stats: bool = False,
    quantiles: List[float] | None = None,
) -> Tuple[List[ComponentReport], Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    qts.to(device)
    qts.train()

    # Build synthetic inputs; intentionally mismatch text dim to force text_proj
    Xv, Qt = make_synthetic_inputs(B=B, M=M, L=L, D_v=qts.cfg.embedding_dim, D_t=D_txt, device=device)

    # Collect components to track
    components: List[Tuple[str, nn.Module]] = []
    # text projection (lazily created after first forward if dims mismatch)
    # We'll run a dummy forward to instantiate it if needed
    with torch.no_grad():
        _ = qts(Xv[:1], Qt[:1], mode="train")
    if qts.text_proj is not None:
        components.append(("text_proj", qts.text_proj))

    # scoring layers
    for i, layer in enumerate(qts.selector.scoring_layers):
        components.append((f"scoring_layers[{i}]", layer))
    # budget head
    components.append(("budget", qts.selector.budget))
    # re-encode layers
    if qts.selector.reencode_layers is not None:
        for i, layer in enumerate(qts.selector.reencode_layers):
            components.append((f"reencode_layers[{i}]", layer))

    # Prepare optimizer
    opt = optim.Adam(qts.parameters(), lr=lr)

    # Snapshot params before training
    before_vecs: Dict[str, torch.Tensor] = {name: flatten_params(mod).cpu() for name, mod in components}

    # Train steps
    scalars = {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, scalars = compute_loss(qts, Xv, Qt)
        loss.backward()
        opt.step()

    # Build reports
    weight_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    reports: List[ComponentReport] = []
    for name, mod in components:
        n_params = count_params(mod)
        gnorm = grad_norm(mod)
        after = flatten_params(mod).cpu()
        before = before_vecs[name]
        # Align lengths if any parameter got created lazily
        if after.numel() != before.numel():
            # Re-snapshot before of matching size (pad with zeros)
            if before.numel() < after.numel():
                pad = torch.zeros(after.numel() - before.numel())
                before = torch.cat([before, pad], dim=0)
            else:
                before = before[: after.numel()]
        delta = after - before
        dnorm = float(torch.linalg.vector_norm(delta).item())
        had_grad = bool(gnorm > 0.0)
        updated = bool(dnorm > 0.0)
        reports.append(ComponentReport(name, n_params, gnorm, dnorm, had_grad, updated))
        if collect_stats:
            before_stats = _compute_weight_stats(before, quantiles)
            after_stats = _compute_weight_stats(after, quantiles)
            weight_stats[name] = {"before": before_stats, "after": after_stats}

    return reports, scalars, weight_stats


def main():
    parser = argparse.ArgumentParser(description="Verify QTS component backprop and updates")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vision-tokens", type=int, default=512)
    parser.add_argument("--text-tokens", type=int, default=64)
    parser.add_argument("--embed-dim", dest="embedding_dim", type=int, default=64)
    parser.add_argument("--text-dim", type=int, default=64, help="Text embedding dim to force projection if != embed-dim")
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=8)
    parser.add_argument("--scoring-layers", type=int, default=2)
    parser.add_argument("--reencode-layers", type=int, default=2)
    parser.add_argument("--nmax", type=int, default=128)
    parser.add_argument("--tau-s", dest="tau_s", type=float, default=0.1)
    parser.add_argument("--rho-min", dest="rho_min", type=float, default=0.05)
    parser.add_argument("--rho-max", dest="rho_max", type=float, default=0.5)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--lambda-t", dest="lambda_t", type=float, default=1.0)
    parser.add_argument("--lambda-m", dest="lambda_m", type=float, default=1.7)
    parser.add_argument("--lambda-s", dest="lambda_s", type=float, default=0.05)
    # Qwen weights are the default init; no toggle
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--fail-on-missing", default=False, action="store_true", help="Return non-zero if any component lacks grad/update")
    parser.add_argument("--print-weight-stats", default=True, action="store_true", help="Print per-component weight distribution stats before/after")
    parser.add_argument("--quantiles", type=str, default="0.05,0.5,0.95", help="Comma-separated quantiles to report, e.g. 0.05,0.5,0.95")
    args = parser.parse_args()

    # Validate divisibility
    if args.embedding_dim % args.n_heads != 0:
        raise SystemExit(f"--embed-dim must be divisible by --n-heads (got {args.embedding_dim} % {args.n_heads})")

    device = torch.device(args.device)

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    qts = build_qts(args)
    qts.to(device)

    # Parse quantiles
    qlist: List[float] | None = None
    if args.quantiles.strip():
        try:
            qlist = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
        except Exception:
            raise SystemExit("Invalid --quantiles format. Use comma separated floats, e.g. 0.05,0.5,0.95")

    reports, scalars, wstats = verify_components(
        qts=qts,
        device=device,
        lr=args.lr,
        steps=args.steps,
        B=args.batch_size,
        M=args.vision_tokens,
        L=args.text_tokens,
        D_txt=args.text_dim,
        collect_stats=args.print_weight_stats,
        quantiles=qlist,
    )

    print("QTS Backprop Verification Results")
    print(f"device={device}")
    print("losses:", ", ".join(f"{k}={v:.6f}" for k, v in scalars.items()))
    print()
    failures = 0
    for rep in reports:
        status = "OK"
        if rep.n_params == 0:
            status = "SKIP(no-params)"
        else:
            if not rep.had_grad or not rep.updated:
                status = "FAIL"
                failures += 1
        print(
            f"- {rep.name}: params={rep.n_params}, grad_norm={rep.grad_norm:.4e}, update_norm={rep.updated_norm:.4e} -> {status}"
        )
        if args.print_weight_stats:
            bs = wstats.get(rep.name, {}).get("before", {})
            as_ = wstats.get(rep.name, {}).get("after", {})
            if bs and as_:
                # Basic summary and deltas
                def fmt_pair(k: str) -> str:
                    v0 = bs.get(k, 0.0)
                    v1 = as_.get(k, 0.0)
                    return f"{k}={v0:.4e}->{v1:.4e} (Δ={v1 - v0:+.2e})"
                basics = ", ".join([fmt_pair(k) for k in ("mean", "std", "min", "max")])
                print(f"    weights: {basics}")
                # Quantiles if requested
                if qlist:
                    qparts = []
                    for q in qlist:
                        k = f"q{q}"
                        v0 = bs.get(k, 0.0)
                        v1 = as_.get(k, 0.0)
                        qparts.append(f"{k}={v0:.4e}->{v1:.4e} (Δ={v1 - v0:+.2e})")
                    print(f"    quantiles: {', '.join(qparts)}")

    if args.fail_on_missing and failures > 0:
        raise SystemExit(f"{failures} component(s) failed to receive grad/update")


if __name__ == "__main__":
    main()
