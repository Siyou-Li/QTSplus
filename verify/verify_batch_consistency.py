"""
Verify per-sample output consistency across different batchings.

This script checks that, for the same set of samples and the same model, the
outputs (logits and greedy predictions) are identical whether each sample is
run alone (batch size = 1) or together in a larger batch (batch size = B>1).

It uses the model's multimodal integration (vision + text) in inference mode,
so results should not depend on batch size. Any discrepancy indicates a
batched handling bug.

Example:
  python -m verify.verify_batch_consistency
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
from contextlib import contextmanager

from src.dataset.sharegptvideo_choice_dataset import ShareGPTVideoChoiceDataset
from src.model.language_model import QTSplusQwen2_5_VLTextForCausalLM
from src.model.vision_encoder import Qwen2_5_VLVisionProcessor


def _device_from_arg(arg: str) -> torch.device:
    arg = (arg or "auto").lower()
    if arg == "cpu":
        return torch.device("cpu")
    if arg.startswith("cuda") or arg == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_from_arg(arg: str | None):
    if arg is None:
        return torch.float32
    a = arg.lower()
    if a in ("bf16", "bfloat16"):
        return torch.bfloat16
    if a in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


_GLOBAL_SDP_MATH_ONLY = False

def _set_sdp(math: bool | None = None, mem_efficient: bool | None = None, flash: bool | None = None):
    if not torch.cuda.is_available():
        return
    try:
        if math is not None:
            torch.backends.cuda.enable_math_sdp(bool(math))
        if mem_efficient is not None:
            torch.backends.cuda.enable_mem_efficient_sdp(bool(mem_efficient))
        if flash is not None:
            torch.backends.cuda.enable_flash_sdp(bool(flash))
    except Exception:
        pass

@contextmanager
def _sdp_mode(math: bool | None = None, mem_efficient: bool | None = None, flash: bool | None = None):
    # Save current
    cur_math = getattr(torch.backends.cuda, "sdp_kernel", None)
    # Fallback if attribute not present
    try:
        prev_math = torch.backends.cuda.sdp_kernel.is_math
        prev_mem = torch.backends.cuda.sdp_kernel.is_mem_efficient
        prev_flash = torch.backends.cuda.sdp_kernel.is_flash
    except Exception:
        prev_math = prev_mem = prev_flash = None
    # Set desired
    _set_sdp(math=math, mem_efficient=mem_efficient, flash=flash)
    try:
        yield
    finally:
        # Restore previous if known
        if prev_math is not None:
            _set_sdp(math=prev_math, mem_efficient=prev_mem, flash=prev_flash)

@torch.no_grad()
def run_prepare_infer(model: QTSplusQwen2_5_VLTextForCausalLM, batch: Dict[str, Any]):
    """Use the model's multimodal prep in inference mode to get embeds + mask.

    We temporarily prefer efficient/flash SDPA during vision+QTS+ prep to avoid OOM
    when the CLI is run with --sdp-math-only for deterministic LM forward.
    """
    if _GLOBAL_SDP_MATH_ONLY and torch.cuda.is_available():
        # Use memory-efficient path for vision encoding to reduce VRAM
        with _sdp_mode(math=False, mem_efficient=True, flash=True):
            return model.prepare_inputs_for_multimodal(
                vision_input=batch.get("vision_input", None),
                input_ids=batch["input_ids"],
                position_ids=None,
                attention_mask=batch["attention_mask"],
                past_key_values=None,
                labels=None,
                question_input_ids=batch.get("question_input_ids", None),
                video_token_id=None,
                mode="infer",
            )
    # Default path
    return model.prepare_inputs_for_multimodal(
        vision_input=batch.get("vision_input", None),
        input_ids=batch["input_ids"],
        position_ids=None,
        attention_mask=batch["attention_mask"],
        past_key_values=None,
        labels=None,
        question_input_ids=batch.get("question_input_ids", None),
        video_token_id=None,
        mode="infer",
    )


@torch.no_grad()
def _debug_qts_indices(model: QTSplusQwen2_5_VLTextForCausalLM, batch: Dict[str, Any]):
    """Return kept indices from QTS+ for each sample (infer mode), if available.

    Only works when `vision_input` is a list of dicts and `question_input_ids` is a list of 1D tensors.
    """
    vision_input = batch.get("vision_input", None)
    qi = batch.get("question_input_ids", None)
    if not isinstance(vision_input, list) or qi is None:
        return None
    mdl = model.get_model()
    vt = mdl.get_vision_tower()
    qts = mdl.get_qts_plus_tower()
    te_layer = mdl.get_input_embeddings()
    kept = []
    # Use efficient kernels for vision if global math-only is requested
    if _GLOBAL_SDP_MATH_ONLY and torch.cuda.is_available():
        ctx = _sdp_mode(math=False, mem_efficient=True, flash=True)
    else:
        @contextmanager
        def _null_ctx():
            yield
        ctx = _null_ctx()
    with ctx:
        for b, vi in enumerate(vision_input):
            vf = vt.get_video_features(
                vi["pixel_values_videos"].to(vt.device),
                vi["video_grid_thw"].to(vt.device),
            )
            if vf.dim() == 2:
                vf = vf.unsqueeze(0)
            qb = qi[b]
            if qb.dtype is not torch.long:
                qb = qb.long()
            te = te_layer(qb.unsqueeze(0).to(te_layer.weight.device))
            vf = vf.to(device=te.device, dtype=te.dtype)
            out = qts(vf, te, mode="infer")
            kept.append(out["indices"][0].detach().cpu())
    return kept


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if k == "vision_input" and isinstance(v, dict):
            out[k] = {sk: (sv.to(device) if torch.is_tensor(sv) else sv) for sk, sv in v.items()}
        elif k == "question_input_ids" and isinstance(v, list):
            # Move list of 1D tensors to device
            out[k] = [t.to(device) if torch.is_tensor(t) else t for t in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def as_batched_listlike(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Batch size 1 collate that mirrors the multi-sample list path.

    - Keep `vision_input` as a 1-element list of dicts
    - Keep `question_input_ids` as a 1-element list of 1D tensors
    - Unsqueeze tensor fields like input_ids/attention_mask/labels to [1, S]
    This ensures single-sample goes through the same code path as multi-sample.
    """
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if k == "vision_input" and isinstance(v, dict):
            out[k] = [v]
        elif k == "question_input_ids" and torch.is_tensor(v):
            out[k] = [v]
        elif torch.is_tensor(v):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def collate_as_list(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate into a batch while keeping variable-length fields as lists.

    - Stack input_ids, attention_mask (fixed length from dataset)
    - Keep question_input_ids as list of 1D tensors
    - Keep vision_input as list of dicts
    """
    keys = batch_list[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        v0 = batch_list[0][k]
        if k in ("vision_input", "question_input_ids"):
            out[k] = [b[k] for b in batch_list]
        elif torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
        else:
            out[k] = [b[k] for b in batch_list]
    return out


def main():
    # Global determinism knobs to minimize backend-dependent differences
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    try:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Warn only to avoid hard failures if an op lacks a deterministic algorithm
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Verify batch-size invariant outputs")
    p.add_argument("--pretrain-lm-model", type=str, default="pretrained_models/Qwen2.5-VL-3B-Instruct-LM")
    p.add_argument("--pretrain-vision-model", type=str, default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision/model.safetensors")
    p.add_argument("--vision-processor", type=str, default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision")
    p.add_argument("--dataset-base-path", type=str, default="datasets/ShareGPTVideoChoice/train_300k_480p")
    p.add_argument("--dataset-jsonl-path", type=str, default="datasets/ShareGPTVideoChoice/prediction_correct_eval.jsonl")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp32")
    p.add_argument("--sdp-math-only", action="store_true", help="Force PyTorch SDPA math backend for determinism.")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--debug-embeds", action="store_true", help="Also compare integrated inputs_embeds and masks.")
    p.add_argument("--debug-qts", action="store_true", help="Also compare QTS+ kept indices.")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    dtype = _dtype_from_arg(args.dtype)

    # Load processor + model
    processor = Qwen2_5_VLVisionProcessor.from_pretrained(args.vision_processor)
    tok = processor.tokenizer
    tok.pad_token = "<|endoftext|>"
    tok.eos_token = "<|im_end|>"
    tok.bos_token = "<|endoftext|>"
    tok.padding_side = "right"

    model = QTSplusQwen2_5_VLTextForCausalLM.from_pretrained(args.pretrain_lm_model)
    try:
        model.to(dtype=dtype)
    except Exception:
        pass
    model = model.to(device)
    model.eval()

    # Ensure vision tower and QTS+ are initialized similarly to training
    from types import SimpleNamespace
    cfg = model.config
    margs = SimpleNamespace(
        vision_tower="qwen2_5_vl_vision",
        pretrain_vision_model=args.pretrain_vision_model,
        enable_qts_plus=True,
        lm_embed_size=getattr(cfg, "hidden_size", 2048),
        vision_embed_size=getattr(cfg, "vision_embed_size", 2048),
        project_text_if_needed=False,
        qts_plus_n_heads=getattr(cfg, "num_attention_heads", 8),
        qts_plus_tau_s=0.85,
        qts_plus_nmax=256,
        qts_plus_rho_min=0.05,
        qts_plus_rho_max=0.5,
        qts_plus_block_dropout=0.0,
        qts_plus_reencode=True,
        qts_plus_scoring_layers=1,
        qts_plus_reencode_layers=1,
        freeze_vision_model=True,
        lambda_t=0.1,
        lambda_m=0.17,
        lambda_s=0.05
    )
    model.get_model().initialize_vision_modules(model_args=margs)
    # Prefer SDPA implementation to avoid backend-dependent kernels
    try:
        if hasattr(model, "config"):
            model.config._attn_implementation = "sdpa"
    except Exception:
        pass
    # Record global preference for LM forward; we’ll bracket calls to avoid OOM in vision
    global _GLOBAL_SDP_MATH_ONLY
    _GLOBAL_SDP_MATH_ONLY = bool(args.sdp_math_only)
    # Move any newly created submodules (e.g., qts_plus, vision_tower) to target device and set eval()
    model = model.to(device)
    model.eval()

    # Load dataset and sample examples with available vision
    ds = ShareGPTVideoChoiceDataset(
        base_path=args.dataset_base_path,
        jsonl_path=args.dataset_jsonl_path,
        processor=processor,
        max_length=128,
        local_rank=0,
        train=False,
    )

    samples: List[Dict[str, Any]] = []
    i = 0
    while len(samples) < min(args.num_samples, args.batch_size) and i < len(ds):
        s = ds[i]
        i += 1
        if s is None:
            continue
        # Require vision_input to exist for this check
        if "vision_input" not in s:
            continue
        samples.append(s)

    if len(samples) < 1:
        raise RuntimeError("No valid samples with vision_input found.")

    # Build singles and batch
    batched = collate_as_list(samples)
    singles = [as_batched_listlike(s) for s in samples]

    # Move to device
    batched = to_device(batched, device)
    singles = [to_device(s, device) for s in singles]

    # Prepare inputs (infer mode) to get inputs_embeds + attention_mask
    _, _, attn_b, _, emb_b, _, *_ = run_prepare_infer(model, batched)
    if _GLOBAL_SDP_MATH_ONLY and torch.cuda.is_available():
        with _sdp_mode(math=True, mem_efficient=False, flash=False):
            logits_b = model(
                inputs_embeds=emb_b,
                attention_mask=attn_b,
                labels=None,
                return_dict=True,
            ).logits.float()
    else:
        logits_b = model(
            inputs_embeds=emb_b,
            attention_mask=attn_b,
            labels=None,
            return_dict=True,
        ).logits.float()

    max_diff = 0.0
    all_ok = True
    for idx, s in enumerate(singles):
        _, _, attn_s, _, emb_s, _, *_ = run_prepare_infer(model, s)
        if _GLOBAL_SDP_MATH_ONLY and torch.cuda.is_available():
            with _sdp_mode(math=True, mem_efficient=False, flash=False):
                logits_s = model(
                    inputs_embeds=emb_s,
                    attention_mask=attn_s,
                    labels=None,
                    return_dict=True,
                ).logits.float()
        else:
            logits_s = model(
                inputs_embeds=emb_s,
                attention_mask=attn_s,
                labels=None,
                return_dict=True,
            ).logits.float()

        Ls = int(attn_s[0].sum().item())
        Lb = int(attn_b[idx].sum().item())
        if Ls != Lb:
            print(f"[FAIL][{idx}] length mismatch: single {Ls} vs batched {Lb}")
            all_ok = False
            continue

        # Optionally compare integrated embeddings and masks up to valid length
        if args.debug_embeds:
            ea = emb_s[0, :Ls].float()
            eb = emb_b[idx, :Ls].float()
            ma = attn_s[0, :Ls].float()
            mb = attn_b[idx, :Ls].float()
            ediff = (ea - eb).abs().max().item()
            mdiff = (ma - mb).abs().max().item()
            print(f"[emb][{idx}] L={Ls} max_abs_diff={ediff:.3e} mask_diff={mdiff:.3e}")

        # Compare logits up to valid length
        a = logits_s[0, :Ls]
        b = logits_b[idx, :Ls]
        diff = (a - b).abs().max().item()
        max_diff = max(max_diff, diff)
        equal = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
        print(f"[cmp][{idx}] L={Ls} max_abs_diff={diff:.3e} allclose={equal}")
        if not equal:
            all_ok = False

        # Also compare greedy predictions
        pred_s = a.argmax(dim=-1)
        pred_b = b.argmax(dim=-1)
        if not torch.equal(pred_s, pred_b):
            print(f"[FAIL][{idx}] greedy predictions differ")
            all_ok = False

        if args.debug_qts:
            ks_list = _debug_qts_indices(model, s)
            kb_list = _debug_qts_indices(model, batched)
            if ks_list is not None and kb_list is not None:
                ks = ks_list[0].tolist()
                kb = kb_list[idx].tolist()
                print(f"[qts][{idx}] kept_single={len(ks)} kept_batch={len(kb)} first10_s={ks[:10]} first10_b={kb[:10]}")

    status = "PASS" if all_ok else "FAIL"
    print(f"[summary] status={status} max_abs_diff={max_diff:.3e} tol=({args.atol},{args.rtol})")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
