# -*- encoding: utf-8 -*-
# @File        :   tokenizer.py
# @Time        :   2025/03/16 20:45:07
# @Author      :   Siyou
# @Description :
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qts_plus import QTSplus


@dataclass
class QTSplusTokenizerConfig:

    embedding_dim: int

    # QTS+
    n_heads: int = 8
    num_kv_heads: Optional[int] = None
    tau_s: float = 0.1
    nmax: int = 2560
    rho_min: float = 0.05
    rho_max: float = 0.5
    block_dropout: float = 0.0
    reencode: bool = True
    scoring_layers: int = 1
    reencode_layers: int = 1

    lambda_t: float = 1.0
    lambda_m: float = 1.7
    lambda_s: float = 0.05

    # Misc
    project_text_if_needed: bool = False

    # Qwen weights are always used for initialization; no random-init toggles


class QTSplusTokenizer(nn.Module):
    """
    End-to-end *QTSplusTok* tokenizer.

    Pipeline:
      X_v --(QTS+)--> X′
    """
    def __init__(self, cfg: QTSplusTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.selector = QTSplus(
            d_model=cfg.embedding_dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.num_kv_heads or cfg.n_heads,
            tau_s=cfg.tau_s,
            nmax=cfg.nmax,
            rho_min=cfg.rho_min,
            rho_max=cfg.rho_max,
            block_dropout=cfg.block_dropout,
            use_reencode=cfg.reencode,
            n_scoring_layers=cfg.scoring_layers,
            n_reencode_layers=cfg.reencode_layers,
        )

        # If text embeddings come in a different dimensionality, learn a light projection.
        self.text_proj: Optional[nn.Linear] = None

        self.rho_sum = 0
        self.rho_count = 0

    def forward(
        self,
        X_v: torch.Tensor,        # [B, M, D]
        Q_t: torch.Tensor,      # [B, L, D_txt]
        mode: str = "train",      # 'train' | 'infer'
    ) -> Dict[str, Any]:
        assert mode in ("train", "infer")
        B, M, D = X_v.shape
        D_txt = Q_t.shape[-1]

        # --- Project text if needed ---
        if D_txt != D:
            if self.cfg.project_text_if_needed:
                if self.text_proj is None:
                    self.text_proj = nn.Linear(D_txt, D, bias=False)
                    # Ensure the projection layer uses the same dtype as input
                    self.text_proj = self.text_proj.to(device=Q_t.device, dtype=Q_t.dtype)
                Q_proj = self.text_proj(Q_t)
            else:
                raise ValueError(f"QTS+ expects text dim {D}, got {D_txt}. Set project_text_if_needed=True.")
        else:
            Q_proj = Q_t

        sel = self.selector(X_v, Q_proj, mode=mode)  # returns dict per qts_plus.py
        Z_list: List[torch.Tensor] = sel["Z"]          # list of [T_b, D] tensors per sample
        n_vec: torch.Tensor = sel["n"]                 # [B]
        rho: torch.Tensor = sel["rho"]                 # [B]
        r: torch.Tensor = sel["r"]                     # [B, M]

        # Compute Eq. (1) compute proxies (per-batch averages for convenience)
        # flops ~ (ρM)^2 / n_max^2 ; kv ~ (ρM) / n_max
        M_tensor = torch.tensor(float(M), device=X_v.device)
        flops_proxy = ((rho * M_tensor) ** 2) / float(self.cfg.nmax ** 2)
        kv_proxy = (rho * M_tensor) / float(self.cfg.nmax)
        self.rho_sum += rho.sum().item()
        self.rho_count += B
        rho_loss = (rho - self.rho_sum / self.rho_count) ** 2

        return {
            "indices": sel["indices"],  # kept indices per sample (list[LongTensor])
            "Z": Z_list,
            "rho": rho,
            "r": r,
            "n": n_vec,
            "add_loss": {
                "flops": flops_proxy.mean() * self.cfg.lambda_t,
                "kv": kv_proxy.mean() * self.cfg.lambda_m,
                "smooth": rho_loss.mean() * self.cfg.lambda_s,
            },
        }

if __name__ == "__main__":
    cfg = QTSplusTokenizerConfig(
        embedding_dim=1024, n_heads=8, tau_s=0.1, nmax=512, rho_min=0.05, rho_max=0.5
    )

    qts = QTSplusTokenizer(cfg)

    # X_v: [B, M, D] vision latents (abs. pos kept upstream)
    X_v = torch.randn(1, 4096, 1024)
    # Q_t: [B, L, D] text/query embeddings (will be projected if D differs)
    Q_t = torch.randn(1, 77, 1024)
    out = qts(X_v, Q_t, mode='train')

    for k, v in out.items():
        if k != "indices":
            print(f"{k}: {v}")
        else:
            print(f"indices: {[x.shape for x in v]}")
