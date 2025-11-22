"""
Verify training I/O correctness for QTS+/Qwen2.5-VL pipeline.

This script performs end-to-end checks on a few samples from the
ShareGPTVideoChoiceDataset and a loaded model. It validates:

- Batch keys and tensor shapes/dtypes
- Attention mask values are {0,1}
- Label masking semantics (-100 on prompt and pad positions)
- Input/label alignment (shifted) and basic exact-match accuracy
- Decoded prompt/answer snippets for human inspection
- Model output/logits shapes and NaN/Inf checks
- Presence and sanity of QTS+ auxiliary losses

Usage examples:

  python script/verify_training_io.py \
    --train-base-path datasets/ShareGPTVideo_train/data/train_300k \
    --train-jsonl-path datasets/ShareGPTVideoChoice/choice_eval.jsonl \
    --pretrain-lm-model pretrained_models/Qwen2.5-VL-3B-Instruct-LM \
    --vision-processor pretrained_models/Qwen2.5-VL-3B-Instruct-Vision \
    --vision-tower qwen2_5_vl_vision \
    --lm-model-type qwen2_5_vl_causal_lm \
    --device cuda \
    --num-samples 2

Notes:
- Run from repo root so that `src` package is resolvable.
- This script uses a tiny batch size and few samples to minimize memory.
- If you only want dataset-level checks (without loading the model), pass
  `--skip-model`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

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


def load_processor_and_model(
    pretrain_lm_model: str,
    vision_processor_path: str,
    device: torch.device,
    lm_model_type: str = "qwen2_5_vl_causal_lm",
) -> Tuple[Qwen2_5_VLVisionProcessor, QTSplusQwen2_5_VLTextForCausalLM]:
    # Vision processor (provides tokenizer too)
    processor = Qwen2_5_VLVisionProcessor.from_pretrained(vision_processor_path)
    tokenizer = processor.tokenizer

    # Normalize special tokens to mirror training setup
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.padding_side = "right"

    # Load model (QTSplus wrapper) and move to device
    if "qwen2_5_vl_causal_lm" not in lm_model_type:
        raise ValueError(f"Unsupported lm_model_type: {lm_model_type}")
    model = QTSplusQwen2_5_VLTextForCausalLM.from_pretrained(pretrain_lm_model)

    # Align PAD/BOS/EOS config to tokenizer
    try:
        pad_id = getattr(tokenizer, "pad_token_id", 151643)
        bos_id = getattr(tokenizer, "bos_token_id", 151643)
        eos_id = getattr(tokenizer, "eos_token_id", 151645)
        if pad_id is not None:
            model.config.pad_token_id = pad_id
        if bos_id is not None:
            model.config.bos_token_id = bos_id
        if eos_id is not None:
            model.config.eos_token_id = eos_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            if pad_id is not None:
                model.generation_config.pad_token_id = pad_id
            if bos_id is not None:
                model.generation_config.bos_token_id = bos_id
            if eos_id is not None:
                model.generation_config.eos_token_id = eos_id
    except Exception:
        pass

    model = model.to(device)
    model.eval()
    return processor, model


def as_batched(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dataset sample dict to a batch of size 1.

    Keeps `vision_input` as-is (dict) and unsqueezes tensor fields.
    """
    batched: Dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batched[k] = v.unsqueeze(0)
        else:
            batched[k] = v
    return batched


def check_tensor(name: str, t: torch.Tensor, shape2: Optional[Tuple[int, ...]] = None) -> List[str]:
    errs: List[str] = []
    if not torch.is_tensor(t):
        errs.append(f"{name} is not a tensor")
        return errs
    if not torch.isfinite(t).all():
        errs.append(f"{name} contains NaN/Inf")
    if shape2 is not None and tuple(t.shape) != tuple(shape2):
        errs.append(f"{name} shape {tuple(t.shape)} != expected {tuple(shape2)}")
    return errs


def verify_inputs(batch: Dict[str, Any], tokenizer) -> List[str]:
    errs: List[str] = []
    required = ["input_ids", "attention_mask", "labels"]
    for k in required:
        if k not in batch:
            errs.append(f"missing key: {k}")
    if errs:
        return errs

    input_ids: torch.Tensor = batch["input_ids"]
    attn: torch.Tensor = batch["attention_mask"]
    labels: torch.Tensor = batch["labels"]

    B, L = input_ids.shape
    errs += check_tensor("input_ids", input_ids)
    errs += check_tensor("attention_mask", attn, (B, L))
    errs += check_tensor("labels", labels, (B, L))

    # attention_mask must be {0,1}
    uniq = torch.unique(attn)
    valid_mask = all(int(x.item()) in (0, 1) for x in uniq)
    if not valid_mask:
        errs.append(f"attention_mask has values outside {{0,1}}: {uniq.tolist()}")

    # labels should be -100 for ignore or valid token ids
    pad_id = getattr(tokenizer, "pad_token_id", None)
    lab_non_ignore = labels[labels != -100]
    if lab_non_ignore.numel() > 0:
        if lab_non_ignore.min() < 0:
            errs.append("labels contain negative values other than -100")
    if pad_id is not None:
        # No pad tokens should be active in labels
        pad_in_labels = (labels == pad_id) & (labels != -100)
        if pad_in_labels.any():
            errs.append("pad token id found in labels without -100 mask")

    # Consistency: positions with attn==0 must have labels==-100
    attn0_labels = labels[attn == 0]
    if attn0_labels.numel() > 0 and not torch.all(attn0_labels == -100):
        errs.append("labels at attention_mask==0 must be -100")

    # There should be at least one supervised label token
    supervised = (labels != -100) & (attn == 1)
    if not supervised.any():
        errs.append("no supervised label tokens (labels != -100) present")

    return errs


@torch.no_grad()
def verify_forward(
    model: QTSplusQwen2_5_VLTextForCausalLM,
    batch: Dict[str, Any],
    tokenizer,
    device: torch.device,
) -> Tuple[List[str], Dict[str, Any]]:
    errs: List[str] = []
    # Move tensors to device
    inputs: Dict[str, Any] = {}
    for k, v in batch.items():
        if k == "vision_input" and isinstance(v, dict):
            inputs[k] = {sk: (sv.to(device) if torch.is_tensor(sv) else sv) for sk, sv in v.items()}
        elif torch.is_tensor(v):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v

    outputs, qts_loss = model(**inputs)

    # ModelOutput can be tuple or dict-like; handle logits extraction
    logits = None
    if isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    elif isinstance(outputs, (list, tuple)):
        logits = outputs[0]
    else:
        # Fallback to attribute
        logits = getattr(outputs, "logits", None)

    if logits is None:
        errs.append("model did not return logits in outputs")
        return errs, {"outputs": outputs, "qts_loss": qts_loss}

    # Logits shape and finiteness
    B, L, V = logits.shape
    errs += check_tensor("logits", logits)

    # NaN/Inf in qts losses
    for k in ("flops_loss", "kv_loss", "smooth_loss"):
        val = qts_loss.get(k, None) if isinstance(qts_loss, dict) else None
        if val is None:
            errs.append(f"missing qts_loss['{k}']")
        else:
            t = torch.as_tensor(val)
            if not torch.isfinite(t).all():
                errs.append(f"qts_loss['{k}'] contains NaN/Inf: {t}")

    # Basic token-level accuracy over supervised positions (shifted)
    labels = inputs.get("labels")
    attn = inputs.get("attention_mask")
    if labels is not None:
        pred_ids = torch.argmax(logits, dim=-1)
        labels_shift = labels[:, 1:]
        preds_shift = pred_ids[:, :-1]
        mask = (labels_shift != -100)
        if mask.any():
            correct = (preds_shift[mask] == labels_shift[mask]).sum().item()
            total = int(mask.sum().item())
            acc = correct / max(1, total)
        else:
            acc = float("nan")
            total = 0

        # Decode small snippets for first example
        b0 = 0
        tok = tokenizer
        # Prompt length derived from labels mask on first sample
        valid0 = (attn[b0] == 1)
        q_mask0 = (labels[b0] == -100) & valid0
        q_len0 = int(q_mask0.sum().item())
        prompt_ids = inputs["input_ids"][b0, :q_len0].detach().cpu()
        gt_ids = inputs["input_ids"][b0, (labels[b0] != -100) & valid0].detach().cpu()
        pred_ids_seq = pred_ids[b0].detach().cpu()
        # Align predicted answer tokens to GT span
        start = q_len0
        end = int(valid0.sum().item())
        pred_ans_ids = pred_ids_seq[start:end]
        # Decode for quick inspection
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        gt_text = tok.decode(gt_ids, skip_special_tokens=True)
        pred_text = tok.decode(pred_ans_ids, skip_special_tokens=True)

        info = {
            "token_accuracy": acc,
            "token_total": total,
            "prompt_text": prompt_text,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "logits_shape": (B, L, V),
        }
    else:
        info = {"logits_shape": (B, L, V)}

    return errs, {"outputs": outputs, "qts_loss": qts_loss, **info}


def run_checks(
    base_path: str,
    jsonl_path: str,
    processor: Qwen2_5_VLVisionProcessor,
    model: Optional[QTSplusQwen2_5_VLTextForCausalLM],
    device: torch.device,
    num_samples: int,
    max_length: int,
) -> int:
    # Dataset expects jsonl_path relative to base_path. Make it relative if possible.
    rel_jsonl = jsonl_path

    ds = ShareGPTVideoChoiceDataset(
        base_path=base_path,
        jsonl_path=rel_jsonl,
        processor=processor,
        max_length=max_length,
        local_rank=0,
        train=True,
    )

    print(f"[verify] dataset size = {len(ds)}")
    err_count = 0

    # Iterate a few samples deterministically from the start
    tested = 0
    for idx in range(len(ds)):
        if tested >= num_samples:
            break
        sample = ds[idx]
        if sample is None:
            continue
        tested += 1
        print("=" * 16 + f" sample {idx} " + "=" * 16)

        batch = as_batched(sample)

        # Input validations
        in_errs = verify_inputs(batch, processor.tokenizer)
        if in_errs:
            for e in in_errs:
                print(f"[ERR][inputs] {e}")
            err_count += len(in_errs)

        # Human-inspect decoded sequences
        tok = processor.tokenizer
        ids0 = batch["input_ids"][0]
        attn0 = batch["attention_mask"][0]
        labels0 = batch["labels"][0]
        full_text = tok.decode(ids0[attn0 == 1])
        label_ids = ids0[(labels0 != -100) & (attn0 == 1)]
        label_text = tok.decode(label_ids)
        print(f"[inspect] prompt+answer tokens (attn==1):\n{full_text}")
        print(f"[inspect] supervised label text:\n{label_text}")

        # Vision input presence and shapes
        vin = batch.get("vision_input", None)
        if isinstance(vin, dict):
            if "pixel_values_videos" in vin and "video_grid_thw" in vin:
                pv = vin["pixel_values_videos"]
                g = vin["video_grid_thw"]
                print(f"[vision] pixel_values_videos: shape={tuple(pv.shape)} dtype={pv.dtype}")
                print(f"[vision] video_grid_thw: shape={tuple(g.shape)} dtype={g.dtype}")
            elif "pixel_values" in vin and "image_grid_thw" in vin:
                pv = vin["pixel_values"]
                g = vin["image_grid_thw"]
                print(f"[vision] pixel_values: shape={tuple(pv.shape)} dtype={pv.dtype}")
                print(f"[vision] image_grid_thw: shape={tuple(g.shape)} dtype={g.dtype}")
            else:
                print("[vision] unexpected vision_input keys: ", list(vin.keys()))
        else:
            print("[vision] vision_input missing or not a dict (ok for text-only)")

        if model is None:
            continue

        # Forward pass validations
        f_errs, info = verify_forward(model, batch, processor.tokenizer, device)
        if f_errs:
            for e in f_errs:
                print(f"[ERR][forward] {e}")
            err_count += len(f_errs)

        # Summaries
        logits_shape = info.get("logits_shape")
        token_acc = info.get("token_accuracy", None)
        t_total = info.get("token_total", None)
        if logits_shape is not None:
            print(f"[ok] logits shape: {logits_shape}")
        if token_acc is not None:
            print(f"[ok] token acc (shifted, supervised): {token_acc:.4f} over {t_total} tokens")
        if "prompt_text" in info:
            print(f"[pred] prompt: {info['prompt_text'][:300]}")
            print(f"[pred]  pred : {info['pred_text'][:300]}")
            print(f"[pred]  gt   : {info['gt_text'][:300]}")

    print("=" * 32)
    if err_count == 0:
        print("[verify] All checks passed on sampled data.")
    else:
        print(f"[verify] Completed with {err_count} error(s). See logs above.")
    return err_count


def main():
    p = argparse.ArgumentParser(description="Verify model inputs/outputs/labels during training")
    p.add_argument("--train-base-path", default="/data/siyou/QTSplus/datasets/ShareGPTVideo_train/data/train_300k_480p", help="Base path to training media root")
    p.add_argument("--train-jsonl-path", default="/data/siyou/QTSplus/datasets/ShareGPTVideoChoice/choice_eval.jsonl", help="Relative or absolute JSONL path under base")
    p.add_argument("--pretrain-lm-model", default="pretrained_models/Qwen2.5-VL-3B-Instruct-LM")
    p.add_argument("--vision-processor", default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision")
    p.add_argument("--vision-tower", default="qwen2_5_vl_vision")
    p.add_argument("--lm-model-type", default="qwen2_5_vl_causal_lm")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-model", action="store_true", help="Only run dataset/input checks (no model forward)")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    print(f"[env] using device: {device}")

    # Load processor and optionally model
    processor = Qwen2_5_VLVisionProcessor.from_pretrained(args.vision_processor)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.padding_side = "right"

    model = None
    if not args.skip_model:
        print("[load] loading model (this can take a while)...")
        model = QTSplusQwen2_5_VLTextForCausalLM.from_pretrained(args.pretrain_lm_model).to(device)
        model.eval()
        # Align special ids to tokenizer
        try:
            pad_id = getattr(tokenizer, "pad_token_id", 151643)
            bos_id = getattr(tokenizer, "bos_token_id", 151643)
            eos_id = getattr(tokenizer, "eos_token_id", 151645)
            if pad_id is not None:
                model.config.pad_token_id = pad_id
            if bos_id is not None:
                model.config.bos_token_id = bos_id
            if eos_id is not None:
                model.config.eos_token_id = eos_id
        except Exception:
            print("[warn] failed to align model special token ids to tokenizer")

    # Run
    err_count = run_checks(
        base_path=args.train_base_path,
        jsonl_path=args.train_jsonl_path,
        processor=processor,
        model=model,
        device=device,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )

    # Exit non-zero on errors for CI usage
    sys.exit(1 if err_count > 0 else 0)


if __name__ == "__main__":
    main()
