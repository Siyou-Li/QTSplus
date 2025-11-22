# -*- encoding: utf-8 -*-
# @File        :   qts_plus.py
# @Time        :   2025/08/27 03:12:40
# @Author      :   Siyou
# @Description :

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F



# Small utilities
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # [B, T, D]
        return self.net(x)


class TinyTransformerBlock(nn.Module):
    """
    Lightweight re-encoding block used after pruning.
    Single block with RMSNorms, MHA, FFN.
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff or (4 * d_model), dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        h = self.norm1(x)
        # self-attention on pruned tokens; support key_padding_mask for padded positions
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

class ScoringCrossAttentionLayer(nn.Module):
    """
    Cross-attention block: pre-norm Q and KV, MHA(Q, K=V), then FFN on Q path.
    Returns updated Q and optional attention weights.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, d_ff: Optional[int] = None):
        super().__init__()
        self.q_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff or (4 * d_model), dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,                # [B, L, D]
        kv: torch.Tensor,               # [B, M, D]
        kv_key_padding_mask: Optional[torch.Tensor] = None,  # [B, M]
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hq = self.q_norm(q)
        hkv = self.kv_norm(kv)
        out, attn = self.mha(
            hq, hkv, hkv, 
            key_padding_mask=kv_key_padding_mask, 
            need_weights=need_weights, 
            average_attn_weights=False
            )
        q = q + out
        h = self.ffn_norm(q)
        q = q + self.ffn(h)
        return q, attn

class Qwen2_5_ScoringCrossAttentionLayer(nn.Module):
    """
    Cross-attention block compatible with Qwen2.5 attention parameterization.
    - Separate q/k/v projections (with optional multi-query kv heads)
    - Optional reuse of weights from Qwen2.5-VL-3B-Instruct-LM layers
    - Pre/post RMSNorm + simple FFN on the query path

    Notes:
    - Rotary embeddings are intentionally not applied here since kv come from
      vision features with their own positional scheme; we only reuse the
      projection weights and output projection.
    - If num_key_value_heads < num_heads, k/v heads are repeated to match num_heads
      as in Qwen2.5 attention.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        d_ff: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        use_qwen_rms: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "hidden size must be divisible by num_heads"
        self.hidden_size = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = dropout

        # Norms
        if use_qwen_rms:
            # Local minimal RMSNorm following Qwen2RMSNorm behavior
            class _Qwen2RMSNorm(nn.Module):
                def __init__(self, hidden_size: int, eps: float = 1e-6):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(hidden_size))
                    self.eps = eps

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    dtype = x.dtype
                    x = x.to(torch.float32)
                    var = x.pow(2).mean(dim=-1, keepdim=True)
                    x = x * torch.rsqrt(var + self.eps)
                    return (self.weight * x).to(dtype)

            self.q_norm = _Qwen2RMSNorm(d_model, eps=rms_norm_eps)
            self.kv_norm = _Qwen2RMSNorm(d_model, eps=rms_norm_eps)
            self.ffn_norm = _Qwen2RMSNorm(d_model, eps=rms_norm_eps)
        else:
            self.q_norm = RMSNorm(d_model, eps=rms_norm_eps)
            self.kv_norm = RMSNorm(d_model, eps=rms_norm_eps)
            self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)

        # Qwen-style projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # FFN on query path
        self.ffn = FeedForward(d_model, d_ff or (4 * d_model), dropout=dropout)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: [B, H_kv, T, Dh] -> [B, H, T, Dh]
        b, h_kv, t, dh = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(b, h_kv, n_rep, t, dh)
        return x.reshape(b, h_kv * n_rep, t, dh)

    def forward(
        self,
        q: torch.Tensor,                # [B, L, D]
        kv: torch.Tensor,               # [B, M, D]
        kv_key_padding_mask: Optional[torch.Tensor] = None,  # [B, M]
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = q.shape
        _, M, _ = kv.shape

        # Pre norms
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)

        # Projections
        q_states = self.q_proj(qn)      # [B, L, H*Dh]
        k_states = self.k_proj(kvn)     # [B, M, Hkv*Dh]
        v_states = self.v_proj(kvn)     # [B, M, Hkv*Dh]

        # Reshape to heads
        q_states = q_states.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, L, Dh]
        k_states = k_states.view(B, M, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, M, Dh]
        v_states = v_states.view(B, M, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, M, Dh]

        # Repeat kv if necessary
        if self.num_key_value_groups > 1:
            k_states = self._repeat_kv(k_states, self.num_key_value_groups)
            v_states = self._repeat_kv(v_states, self.num_key_value_groups)

        # Attention weights [B, H, L, M]
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if kv_key_padding_mask is not None:
            # Convert mask [B, M] -> broadcast [B, 1, 1, M]
            mask = kv_key_padding_mask[:, None, None, :].to(dtype=attn_weights.dtype)
            attn_weights = attn_weights.masked_fill(mask > 0.5, float('-inf'))

        # Softmax in float32 for stability
        attn_dtype = attn_weights.dtype
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Aggregate values -> [B, H, L, Dh]
        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)

        # Final projection and residual + FFN
        out = self.o_proj(attn_output)
        q = q + out
        q = q + self.ffn(self.ffn_norm(q))

        return q, (attn_weights if need_weights else None)

    def init_from_qwen_attn(self, qwen_attn: nn.Module, qwen_input_ln: Optional[nn.Module] = None):
        """
        Initialize projections from a Qwen2.5 attention module.
        Expects attributes: q_proj, k_proj, v_proj, o_proj.
        Optionally copy RMSNorm weight from the corresponding input_layernorm.
        """
        def _copy_param(dst: torch.nn.Parameter, src: torch.nn.Parameter, name: str):
            if dst.shape != src.shape:
                print(f"[QTS+] Skip copy for {name}: shape mismatch {tuple(src.shape)} -> {tuple(dst.shape)}")
                return False
            dst.copy_(src)
            return True

        # Copy weights without tracking gradients to avoid in-place leaf autograd errors
        with torch.no_grad():
            ok = True
            ok &= _copy_param(self.q_proj.weight, qwen_attn.q_proj.weight, "q_proj.weight")
            if self.q_proj.bias is not None and getattr(qwen_attn.q_proj, 'bias', None) is not None:
                ok &= _copy_param(self.q_proj.bias, qwen_attn.q_proj.bias, "q_proj.bias")

            ok &= _copy_param(self.k_proj.weight, qwen_attn.k_proj.weight, "k_proj.weight")
            if self.k_proj.bias is not None and getattr(qwen_attn.k_proj, 'bias', None) is not None:
                ok &= _copy_param(self.k_proj.bias, qwen_attn.k_proj.bias, "k_proj.bias")

            ok &= _copy_param(self.v_proj.weight, qwen_attn.v_proj.weight, "v_proj.weight")
            if self.v_proj.bias is not None and getattr(qwen_attn.v_proj, 'bias', None) is not None:
                ok &= _copy_param(self.v_proj.bias, qwen_attn.v_proj.bias, "v_proj.bias")

            ok &= _copy_param(self.o_proj.weight, qwen_attn.o_proj.weight, "o_proj.weight")

            # Copy RMSNorm weights if compatible
            if qwen_input_ln is not None and hasattr(qwen_input_ln, 'weight'):
                if hasattr(self.q_norm, 'weight'):
                    _copy_param(self.q_norm.weight, qwen_input_ln.weight, "q_norm.weight")
                if hasattr(self.kv_norm, 'weight'):
                    _copy_param(self.kv_norm.weight, qwen_input_ln.weight, "kv_norm.weight")

        print("Qwen2.5 attention weights initialized (best-effort shape check).")


class Qwen2_5_SelfReencodeLayer(nn.Module):
    """
    Thin wrapper that reuses Qwen2_5_ScoringCrossAttentionLayer as a self-attention
    re-encoding block (q == kv). This avoids duplicating attention code while enabling
    initialization from Qwen2.5 LM layers.

    Forward signature matches TinyTransformerBlock(x[, key_padding_mask]).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        d_ff: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        use_qwen_rms: bool = True,
    ):
        super().__init__()
        self.core = Qwen2_5_ScoringCrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads or num_heads,
            dropout=dropout,
            d_ff=d_ff,
            rms_norm_eps=rms_norm_eps,
            use_qwen_rms=use_qwen_rms,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.core(x, x, kv_key_padding_mask=key_padding_mask, need_weights=False)
        return y

    def init_from_qwen_attn(self, qwen_attn: nn.Module, qwen_input_ln: Optional[nn.Module] = None):
        self.core.init_from_qwen_attn(qwen_attn, qwen_input_ln)

# QTS+ 
class BudgetHead(nn.Module):
    """
    ρ = ρ_min + (ρ_max - ρ_min) * σ( MLP([sq, log M, max r, H(p)]) )
    where sq is the mean query embedding.
    """
    def __init__(self, d_model: int, hidden: int = 256, rho_min: float = 0.05, rho_max: float = 0.5):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, sq: torch.Tensor, logM: torch.Tensor, r_max: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        sq: [B, D] (mean of query embeddings)
        logM, r_max, H: [B]
        returns ρ in [rho_min, rho_max], shape [B]
        """
        B, D = sq.shape
        x = torch.cat([sq, logM.view(B, 1), r_max.view(B, 1), H.view(B, 1)], dim=1)
        logits = self.mlp(x).squeeze(1)
        rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(logits)
        return rho

class QTSplus(nn.Module):
    """
    Query-Aware Token Selector with Adaptive Budget.
    - Cross-attention scoring: r in [0,1]^M via max over text & heads.
    - Predict ρ from query & video stats.
    - Train mode: differentiable threshold gate with bisection.
    - Infer mode: hard Top-n selection.
    - Then one tiny re-encoding transformer block.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        tau_s: float = 0.1,
        nmax: int = 2560,
        rho_min: float = 0.05,
        rho_max: float = 0.5,
        block_dropout: float = 0.0,
        use_reencode: bool = True,
        n_scoring_layers: int = 1,
        n_reencode_layers: int = 1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.tau_s = tau_s
        self.nmax = nmax
        self.use_reencode = use_reencode
        self.n_scoring_layers = max(int(n_scoring_layers), 1)
        self.n_reencode_layers = max(int(n_reencode_layers), 1)

        # linear projections for cross-attn scoring
        # self.Wk = nn.Linear(d_model, d_model, bias=False)
        # self.Wq = nn.Linear(d_model, d_model, bias=False)

        # scoring layers: Qwen2.5-compatible by default
        n_heads_eff = self.n_heads
        n_kv_heads_eff = int(n_kv_heads) if (n_kv_heads is not None and int(n_kv_heads) > 0) else self.n_heads
        self.scoring_layers = nn.ModuleList([
            Qwen2_5_ScoringCrossAttentionLayer(
                d_model,
                num_heads=n_heads_eff,
                num_key_value_heads=n_kv_heads_eff,
                dropout=0.0,
                rms_norm_eps=1e-6,
                use_qwen_rms=True,
            ) for _ in range(self.n_scoring_layers)
        ])
        
        self.budget = BudgetHead(d_model, rho_min=rho_min, rho_max=rho_max)

        # re-encode layers: Qwen-style self-attention by default when enabled
        if use_reencode:
            self.reencode_layers = nn.ModuleList([
                Qwen2_5_SelfReencodeLayer(
                    d_model,
                    num_heads=self.n_heads,
                    num_key_value_heads=n_kv_heads_eff,
                    dropout=block_dropout,
                    rms_norm_eps=1e-6,
                    use_qwen_rms=True,
                ) for _ in range(self.n_reencode_layers)
            ])
        else:
            self.reencode_layers = None

    @staticmethod
    def _entropy_from_r(r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # r: [B, M] relevance in [0,1]; form normalized p then H(p)
        p = r / (r.sum(dim=1, keepdim=True) + eps)  # [B, M]
        H = -(p * (p + eps).log()).sum(dim=1)       # [B]
        return H.clamp_min(0.0), p

    def _find_threshold(self, r: torch.Tensor, rho: torch.Tensor, tau_s: float, iters: int = 10) -> torch.Tensor:
        """
        Bisection per-batch-element for t s.t. sum σ((r - t)/τ) = ρ M
        r: [B, M], rho: [B]
        returns t: [B]
        """
        B, M = r.shape
        t_low = r.min(dim=1).values - 6.0 * tau_s
        t_high = r.max(dim=1).values + 6.0 * tau_s
        for _ in range(iters):
            t = 0.5 * (t_low + t_high)
            s = torch.sigmoid((r - t.unsqueeze(1)) / tau_s).sum(dim=1) - (rho * M)
            go_low = s > 0   # if too many kept, increase threshold lower bound
            t_low = torch.where(go_low, t, t_low)
            t_high = torch.where(go_low, t_high, t)
        return 0.5 * (t_low + t_high)
    
    def _find_threshold_differentiable(self, r, rho, tau_s, iters=6, eps=1e-6):
        # r: [B, M], rho: [B]
        t = r.median(dim=1, keepdim=True).values  # good starting point
        M = r.size(1)
        for _ in range(iters):
            s = torch.sigmoid((r - t) / tau_s)             # [B, M]
            g = s.sum(dim=1, keepdim=True) - (rho*M).view(-1,1)
            gp = -(s * (1 - s) / tau_s).sum(dim=1, keepdim=True)  # d/dt
            t = t - g / (gp + eps)
        return t.squeeze(1)

    def _cross_attention_scores(self, Xv: torch.Tensor, Qt: torch.Tensor) -> torch.Tensor:
        """
        Xv: [B, M, D] visual tokens (after codebook, with abs pos encoding kept upstream)
        Qt: [B, L, D] text tokens
        returns r: [B, M] in [0,1]
        """
        q = Qt
        attn_weights: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.scoring_layers):
            need_w = (i == len(self.scoring_layers) - 1)
            q, w = layer(q, Xv, kv_key_padding_mask=None, need_weights=need_w)
            if need_w:
                attn_weights = w  # [B, h, L, M]
        # r via max over text (L) and heads (h)
        assert attn_weights is not None
        r = attn_weights.amax(dim=2).amax(dim=1)  # [B, M]
        return r

        # B, M, D = Xv.shape
        # _, L, _ = Qt.shape
        # # Fallback: manual scaled dot-product using explicit Wq/Wk
        # K = self.Wk(Xv)  # [B, M, D]
        # U = self.Wq(Qt)  # [B, L, D]

        # # reshape to heads
        # K = K.view(B, M, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, M, dh]
        # U = U.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, L, dh]

        # # attention: softmax over visual positions (M)
        # # scores: [B, h, L, M]
        # scores = torch.matmul(U, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # A = F.softmax(scores, dim=-1)

        # # max-pool over text (L) and heads (h): r in [0,1]^M
        # r = A.amax(dim=2).amax(dim=1)  # [B, M]
        # return r

    def forward(
        self,
        Xv: torch.Tensor,  # [B, M, D]
        Qt: torch.Tensor,  # [B, L, D]
        mode: str = "train",
    ) -> Dict[str, Any]:
        B, M, D = Xv.shape
        assert D == self.d_model

        # 1) Cross-attention scoring
        r = self._cross_attention_scores(Xv, Qt)                  # [B, M] in [0,1]

        # 2) Adaptive budget prediction
        H, p = self._entropy_from_r(r)
        sq = Qt.mean(dim=1)                                       # [B, D]
        logM = torch.full((B,), float(math.log(max(M, 1))), device=Xv.device, dtype=Xv.dtype)
        r_max = r.max(dim=1).values
        rho = self.budget(sq, logM, r_max, H)                     # [B], clamp in head

        # fixed rho for debugging
        # rho = torch.full_like(rho, 0.5)

        n_pred = torch.clamp((rho * M).ceil().long(), min=1)      # at least 1
        n = torch.minimum(n_pred, torch.full_like(n_pred, self.nmax))

        # 3) Train-time differentiable gate / Inference hard Top-n
        if mode == "train":
            # Differentiable threshold with Newton-style refinement (keeps budget expectation)
            t = self._find_threshold_differentiable(r, rho, self.tau_s, iters=10)  # [B]

            # Replace TopK + manual straight-through with Gumbel-Softmax (binary keep/drop)
            # logits_keep ~ (r - t); logits_drop ~ 0; temperature = tau_s
            logits = torch.stack([r - t.unsqueeze(1), torch.zeros_like(r)], dim=-1)  # [B, M, 2]
            y = F.gumbel_softmax(logits, tau=self.tau_s, hard=True, dim=-1)         # one-hot along 2
            s_keep = y[..., 0]                                                      # [B, M] in {0,1}, grad via GS

            # Ensure at least one token per sample (rare edge if GS picks all drop)
            with torch.no_grad():
                none_kept = (s_keep.sum(dim=1) < 0.5)
                if none_kept.any():
                    for b in torch.nonzero(none_kept, as_tuple=False).view(-1):
                        j = torch.argmax(r[b])
                        s_keep[b].zero_()
                        s_keep[b, j] = 1.0

            Z = s_keep.unsqueeze(-1) * Xv  # [B, M, D]

            # Gather kept tokens per sample in original order
            kept_list = []
            kept_idx_list = []
            for b in range(B):
                kb = (s_keep[b] > 0.5).nonzero(as_tuple=False).squeeze(1)
                kb, _ = torch.sort(kb)
                kept_list.append(Z[b, kb])
                kept_idx_list.append(kb)

            if self.use_reencode:
                # Pad/tile to max kept for batched re-encoding
                max_keep = int(max([len(x) for x in kept_list]))
                Zb = []
                for b in range(B):
                    x = kept_list[b]
                    if x.size(0) < max_keep:
                        # Repeat last kept token to pad; guaranteed at least one by fallback above
                        pad = x[-1:].repeat(max_keep - x.size(0), 1)
                        x = torch.cat([x, pad], dim=0)
                    Zb.append(x.unsqueeze(0))
                Zb = torch.cat(Zb, dim=0)  # [B, max_keep, D]

                # Debug: skip previous step and re-encode all visual tokens
                # Zb = Xv

                # Apply each re-encode block sequentially
                for layer in self.reencode_layers:
                    Zb = layer(Zb)
                # Slice back to each sample's true kept count
                Z_out = []
                for b in range(B):
                    Z_out.append(Zb[b, : kept_idx_list[b].numel()])
            else:
                # Skip re-encoding; directly return kept features
                Z_out = kept_list
            # ragged output, collate as list for flexibility
            return {
                "Z": Z_out,               # list of [n[b], D]
                "indices": kept_idx_list, # list of [n[b]]
                "rho": rho,               # [B]
                "r": r,                   # [B, M]
                "p": p,                   # [B, M]            
                "n": n,                   # [B]
            }
        else:
            # inference: hard Top-n, but preserve original temporal order
            kept_idx_list = []
            Z_out = []
            for b in range(B):
                kb = torch.topk(r[b], k=int(n[b].item()), dim=0).indices
                kb, _ = torch.sort(kb)  # keep ascending to preserve original positions
                kept_idx_list.append(kb)
                Z_out.append(Xv[b, kb])
            if self.use_reencode:
                # optional single re-encoding applied per batch via padding
                max_keep = int(max([z.size(0) for z in Z_out]))
                Zb = []
                for z in Z_out:
                    if z.size(0) < max_keep:
                        pad = z[-1:].repeat(max_keep - z.size(0), 1)
                        z = torch.cat([z, pad], dim=0)
                    Zb.append(z.unsqueeze(0))
                Zb = torch.cat(Zb, dim=0)  # [B, max_keep, D]
                # apply each re-encode block sequentially
                for layer in self.reencode_layers:
                    Zb = layer(Zb)
                Z_final = []
                for b in range(B):
                    Z_final.append(Zb[b, : kept_idx_list[b].numel()])
            else:
                # Skip re-encoding; return selected tokens directly
                Z_final = Z_out
            return {
                "Z": Z_final,
                "indices": kept_idx_list,
                "rho": rho,
                "r": r,
                "n": n,
            }

    # --- Utilities to initialize Qwen-scoring directly from a loaded LM ---
    def init_qwen_scoring_from_lm_model(self, lm_model: nn.Module, layer_indices: list, rms_norm_eps: Optional[float] = None):
        """
        Initialize scoring layers from a loaded Qwen2.5 text model instance, avoiding the need
        for explicit qts_plus_qwen_* configuration beyond the layer indices.

        lm_model: a Qwen2_5_VLTextModel-compatible module with attributes:
          - config.hidden_size, config.num_attention_heads, config.num_key_value_heads
          - layers: iterable of decoder layers with .self_attn and .input_layernorm
        layer_indices: list[int] mapping QTS+ scoring layers to LM layers to copy from.
        """
        text_cfg = getattr(lm_model, 'config', None)
        hidden_size = getattr(text_cfg, 'hidden_size', self.d_model)
        num_heads = getattr(text_cfg, 'num_attention_heads', self.n_heads)
        num_kv_heads = getattr(text_cfg, 'num_key_value_heads', num_heads)
        if rms_norm_eps is None:
            rms_norm_eps = getattr(text_cfg, 'rms_norm_eps', 1e-6)

        # Rebuild if d_model differs or head counts differ and are compatible
        want_heads = int(num_heads)
        can_use_lm_heads = (self.d_model % want_heads) == 0
        cur_kv_heads = None
        if hasattr(self, 'scoring_layers') and len(self.scoring_layers) > 0:
            cur_kv_heads = getattr(self.scoring_layers[0], 'num_key_value_heads', None)
        rebuild = (
            (hidden_size != self.d_model)
            or ((want_heads != self.n_heads) and can_use_lm_heads)
            or (cur_kv_heads is None or int(cur_kv_heads) != int(num_kv_heads))
        )
        if rebuild:
            # Only adopt LM head count if compatible with our d_model; else keep current heads
            self.n_heads = want_heads if can_use_lm_heads else self.n_heads
            self.d_head = self.d_model // self.n_heads
            self.scoring_layers = nn.ModuleList([
                Qwen2_5_ScoringCrossAttentionLayer(
                    self.d_model,
                    num_heads=self.n_heads,
                    num_key_value_heads=int(num_kv_heads),
                    dropout=0.0,
                    rms_norm_eps=rms_norm_eps,
                    use_qwen_rms=True,
                ) for _ in range(self.n_scoring_layers)
            ])

        # Collect LM layers and copy
        lm_layers = list(getattr(lm_model, 'layers', []))
        if not lm_layers and hasattr(lm_model, 'model') and hasattr(lm_model.model, 'layers'):
            lm_layers = list(lm_model.model.layers)
        if not lm_layers:
            return  # can't proceed

        for i, layer in enumerate(self.scoring_layers):
            idx = int(layer_indices[i]) if i < len(layer_indices) else int(layer_indices[-1])
            idx = max(0, min(idx, len(lm_layers) - 1))
            q_layer = lm_layers[idx]
            if hasattr(q_layer, 'self_attn'):
                layer.init_from_qwen_attn(q_layer.self_attn, getattr(q_layer, 'input_layernorm', None))
        print("Qwen scoring layers initialized from LM model (where shapes matched).")

    def init_qwen_reencode_from_lm_model(self, lm_model: nn.Module, layer_indices: list, rms_norm_eps: Optional[float] = None):
        """
        Initialize re-encoding self-attention layers from a loaded Qwen2.5 text model.
        This rebuilds reencode layers to Qwen-style if necessary and copies projections
        from the specified LM layer indices.
        """
        if not self.use_reencode:
            return

        text_cfg = getattr(lm_model, 'config', None)
        hidden_size = getattr(text_cfg, 'hidden_size', self.d_model)
        num_heads = getattr(text_cfg, 'num_attention_heads', self.n_heads)
        num_kv_heads = getattr(text_cfg, 'num_key_value_heads', num_heads)
        if rms_norm_eps is None:
            rms_norm_eps = getattr(text_cfg, 'rms_norm_eps', 1e-6)

        # Rebuild reencode layers as Qwen-style self-attn if needed
        can_use_lm_heads = (self.d_model % int(num_heads)) == 0
        # Detect current kv heads from the first reencode layer (wrapper -> core)
        cur_kv_heads = None
        if hasattr(self, 'reencode_layers') and self.reencode_layers is not None and len(self.reencode_layers) > 0:
            core0 = getattr(self.reencode_layers[0], 'core', self.reencode_layers[0])
            cur_kv_heads = getattr(core0, 'num_key_value_heads', None)
        rebuild = (
            (hidden_size != self.d_model)
            or ((int(num_heads) != self.n_heads) and can_use_lm_heads)
            or (cur_kv_heads is None or int(cur_kv_heads) != int(num_kv_heads))
        )
        if rebuild:
            if can_use_lm_heads:
                self.n_heads = int(num_heads)
                self.d_head = self.d_model // self.n_heads
            self.reencode_layers = nn.ModuleList([
                Qwen2_5_SelfReencodeLayer(
                    self.d_model,
                    num_heads=self.n_heads,
                    num_key_value_heads=int(num_kv_heads),
                    dropout=0.0,
                    rms_norm_eps=rms_norm_eps,
                    use_qwen_rms=True,
                ) for _ in range(self.n_reencode_layers)
            ])

        # Collect LM layers and copy
        lm_layers = list(getattr(lm_model, 'layers', []))
        if not lm_layers and hasattr(lm_model, 'model') and hasattr(lm_model.model, 'layers'):
            lm_layers = list(lm_model.model.layers)
        if not lm_layers:
            return

        for i, layer in enumerate(self.reencode_layers):
            idx = int(layer_indices[i]) if i < len(layer_indices) else int(layer_indices[-1])
            idx = max(0, min(idx, len(lm_layers) - 1))
            q_layer = lm_layers[idx]
            if hasattr(q_layer, 'self_attn') and hasattr(layer, 'init_from_qwen_attn'):
                layer.init_from_qwen_attn(q_layer.self_attn, getattr(q_layer, 'input_layernorm', None))
        print("Qwen re-encode layers initialized from LM model (where shapes matched).")
