#!/usr/bin/env python3
"""
Evaluate Qwen2.5-VL-3B-Instruct on ShareGPTVideoChoice (choice_eval.jsonl).

This script mirrors the data loading and inference used in example.ipynb:
- Loads the full model with transformers' Qwen2_5_VLForConditionalGeneration
- Uses AutoProcessor.apply_chat_template and process_vision_info to prepare inputs
- Computes next-token probabilities over options [A, B, C, D]

Outputs
- JSONL with per-sample predictions and probabilities
- Metrics JSON with accuracy and binary (one-vs-rest) metrics
- A ROC/PR visualization PNG (drawn with PIL, no matplotlib dependency)

Usage
  python script/eval_sharegpt_video_choice.py \
      --model pretrained_models/Qwen2.5-VL-3B-Instruct \
      --dataset datasets/ShareGPTVideoChoice/choice_eval.jsonl \
      --media-base datasets/ShareGPTVideo_train/data/train_300k_480p \
      --out-dir eval/sharegpt_choice_eval

Optional
  --max-samples N     Limit evaluation to the first N items
  --dtype auto|bf16|fp16|fp32  Compute dtype (auto: bf16 if CUDA else fp32)
  --max-frames N      Max frames to sample per video directory (default: 48)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Tuple
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration

from src.utils.qwen_vision_process import process_vision_info
from tqdm import tqdm

def pick_device_and_dtype(dtype_choice: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dtype_choice == "auto":
        if device.type == "cuda:0":
            return device, torch.bfloat16
        return device, torch.float32
    return device, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_choice]


def build_question_text(question: str, options: dict, instruction: str) -> str:
    parts = [question]
    for k in sorted(options.keys()):
        parts.append(f"\n{k}. {options[k]}")
    parts.append(f"\n{instruction}")
    return "".join(parts)


def list_video_frames(vision_path: Path, max_frames: int = 48) -> List[str]:
    """Return a subsampled, sorted list of file:// frame paths from a directory.
    Supports .jpg/.jpeg/.png. If fewer than max_frames, returns all frames.
    """
    files = [
        f for f in sorted(os.listdir(vision_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(files) == 0:
        return []
    if len(files) > max_frames:
        step = int(np.ceil(len(files) / float(max_frames)))
        files = files[::step]
        # If still slightly over due to rounding, trim
        files = files[:max_frames]
    return ["file://" + str(vision_path / f) for f in files]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def compute_confusion_matrix(y_true_idx: List[int], y_pred_idx: List[int], n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[t, p] += 1
    return cm


def compute_binary_pair_metrics(y_true_pair: np.ndarray, y_pred_pair: np.ndarray) -> dict:
    """Compute binary precision/recall/F1/accuracy over option-level pairs.

    y_true_pair: shape [N_pairs], 1 for correct option, 0 otherwise per-question/option pair
    y_pred_pair: same shape, 1 for predicted option, 0 otherwise
    """
    tp = int(((y_true_pair == 1) & (y_pred_pair == 1)).sum())
    fp = int(((y_true_pair == 0) & (y_pred_pair == 1)).sum())
    fn = int(((y_true_pair == 1) & (y_pred_pair == 0)).sum())
    tn = int(((y_true_pair == 0) & (y_pred_pair == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR) for binary labels with scores.
    Returns fpr, tpr arrays starting at (0,0) and ending at (1,1).
    """
    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    P = float(np.sum(y_true_sorted == 1))
    N = float(np.sum(y_true_sorted == 0))
    if P == 0 or N == 0:
        # Degenerate: return diagonal
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / P
    fpr = fps / N

    # Prepend (0,0) and ensure end at (1,1)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision-recall curve for binary labels with scores.
    Returns precision, recall arrays with recall increasing.
    """
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    P = float(np.sum(y_true_sorted == 1))
    if P == 0:
        return np.array([1.0, 1.0]), np.array([0.0, 1.0])

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    # For each threshold at rank k (1..n):
    precision = tp / (tp + fp)
    recall = tp / P

    # Prepend starting point (precision=1.0 at recall=0.0)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def draw_plots_roc_pr(
    out_path: Path,
    roc_fpr: np.ndarray,
    roc_tpr: np.ndarray,
    roc_auc: float,
    pr_prec: np.ndarray,
    pr_rec: np.ndarray,
    pr_auc: float,
) -> None:
    """Draw ROC (left) and PR (right) curves using PIL (no matplotlib)."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1000, 500
    margin = 70
    mid_gap = 40
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a nicer default font; fallback to basic
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    # Panels
    left = (margin, margin, W // 2 - mid_gap, H - margin)
    right = (W // 2 + mid_gap, margin, W - margin, H - margin)

    def draw_axes(rect, title: str, xlab: str, ylab: str, baseline: str = None):
        x0, y0, x1, y1 = rect
        draw.rectangle(rect, outline=(0, 0, 0), width=2)
        # grid lines
        for t in [0.25, 0.5, 0.75]:
            # vertical
            xv = x0 + t * (x1 - x0)
            draw.line((xv, y0, xv, y1), fill=(220, 220, 220))
            draw.text((xv - 10, y1 + 5), f"{t:.2f}", fill=(0, 0, 0), font=font_small)
            # horizontal
            yh = y1 - t * (y1 - y0)
            draw.line((x0, yh, x1, yh), fill=(220, 220, 220))
            draw.text((x0 - 35, yh - 7), f"{t:.2f}", fill=(0, 0, 0), font=font_small)
        # axes labels
        draw.text((x0 + (x1 - x0) / 2 - 30, y1 + 25), xlab, fill=(0, 0, 0), font=font)
        draw.text((x0 - 55, y0 - 30), ylab, fill=(0, 0, 0), font=font)
        # title
        draw.text((x0 + 10, y0 - 30), title, fill=(0, 0, 0), font=font_title)

    def plot_curve(rect, xs, ys, color=(40, 120, 220)):
        x0, y0, x1, y1 = rect
        pts = []
        for x, y in zip(xs, ys):
            px = x0 + float(x) * (x1 - x0)
            py = y1 - float(y) * (y1 - y0)
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)

    # ROC
    draw_axes(left, "ROC Curve", "FPR", "TPR")
    # diagonal baseline
    x0, y0, x1, y1 = left
    draw.line((x0, y1, x1, y0), fill=(180, 180, 180), width=2)
    plot_curve(left, roc_fpr, roc_tpr, color=(40, 120, 220))
    draw.text((x0 + 10, y0 - 10), f"AUC = {roc_auc:.4f}", fill=(0, 0, 0), font=font)

    # PR
    draw_axes(right, "PR Curve", "Recall", "Precision")
    plot_curve(right, pr_rec, pr_prec, color=(220, 120, 40))
    draw.text((right[0] + 10, right[1] - 10), f"AUC = {pr_auc:.4f}", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_confusion_matrix(cm: np.ndarray, out_path: Path, labels: List[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    n = cm.shape[0]
    cell = 80
    margin = 120
    W = margin + n * cell + 40
    H = margin + n * cell + 40
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    max_v = cm.max() if cm.size > 0 else 1
    # axes labels
    draw.text((margin + (n * cell) / 2 - 50, 20), "Predicted", fill=(0, 0, 0), font=font)
    draw.text((20, margin + (n * cell) / 2 - 10), "True", fill=(0, 0, 0), font=font)

    # cells
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            intensity = int(255 - (0 if max_v == 0 else (v / max_v) * 200))
            color = (intensity, 200, 255)
            x0 = margin + j * cell
            y0 = margin + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), fill=color)
            draw.text((x0 + cell / 2 - 10, y0 + cell / 2 - 8), str(int(v)), fill=(0, 0, 0), font=font)

    # tick labels
    for idx, lab in enumerate(labels):
        # x-axis (pred)
        draw.text((margin + idx * cell + cell / 2 - 5, margin + n * cell + 10), lab, fill=(0, 0, 0), font=font_small)
        # y-axis (true)
        draw.text((margin - 25, margin + idx * cell + cell / 2 - 8), lab, fill=(0, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL-3B-Instruct on ShareGPTVideoChoice JSONL")
    p.add_argument("--model", default="")
    p.add_argument("--dataset", default="")
    p.add_argument("--media-base", default="", help="Base directory that contains subfolders named by vision_id with frames")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--max-frames", type=int, default=48)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    roc_pr_path = out_dir / "roc_pr.png"
    cm_path = out_dir / "confusion_matrix.png"

    # Load processor and model (as in example.ipynb)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True, 
        local_files_only=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    device, dtype = pick_device_and_dtype(args.dtype)
    model.to(device=device, dtype=dtype).eval()

    # Precompute option token IDs
    options = ["A", "B", "C", "D"]
    option_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in options]
    if any([i is None or int(i) < 0 for i in option_ids]):
        raise RuntimeError(f"Could not resolve token IDs for options {options}: {option_ids}")

    total = 0
    correct = 0
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    # Option-level pairs
    y_true_pair: List[int] = []
    y_score_pair: List[float] = []

    media_base = Path(args.media_base)
    if not media_base.exists():
        raise FileNotFoundError(f"Media base not found: {media_base}")

    # Open dataset and iterate
    with open(args.dataset, "r", encoding="utf-8") as f_in, open(preds_path, "w", encoding="utf-8") as f_out:
        for line_idx, line in tqdm(enumerate(f_in, 1)):
            if args.max_samples is not None and total >= args.max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = rec.get("question")
            ans = rec.get("answer")
            opts = rec.get("options")
            vid = rec.get("vision_id")
            if not q or not ans or not opts or not vid:
                continue

            vision_path = media_base / vid
            if not vision_path.exists():
                # Skip items with missing media
                continue

            # Build user message with video frames + text (as in dataset/verify scripts)
            q_text = build_question_text(
                q, opts,
                "Please provide your options (only A or B or C or D) directly. Do not give any other responses."
            )

            if vision_path.is_dir():
                frames = list_video_frames(vision_path, max_frames=args.max_frames)
                if not frames:
                    continue
                content = [{"type": "video", "video": frames}, {"type": "text", "text": q_text}]
            else:
                # Single file (video or image)
                ext = vision_path.suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                    content = [{"type": "image", "image": str(vision_path)}, {"type": "text", "text": q_text}]
                else:
                    content = [{"type": "video", "video": str(vision_path), "fps": 1.0, "max_pixels": 360 * 420},
                               {"type": "text", "text": q_text}]

            messages = [{"role": "user", "content": content}]

            # Prepare model inputs following example.ipynb and verify script
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            # Pack vision tensors under a single key to match the model's
            # forward signature (expects `vision_input` dict), and add
            # `question_input_ids` for QTS+ re-encode.
            model_inputs = {
                "input_ids": inputs.get("input_ids"),
                "attention_mask": inputs.get("attention_mask"),
            }

            # Construct question_input_ids from the raw question only
            q_ids = processor.tokenizer(
                q, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]  # shape [1, Lq]
            model_inputs["question_input_ids"] = q_ids

            # Build `vision_input` expected by QTSplusQwen2_5_VLTextForCausalLM
            if "pixel_values_videos" in inputs and "video_grid_thw" in inputs:
                vision_input = {
                    "pixel_values_videos": inputs["pixel_values_videos"],
                    "video_grid_thw": inputs["video_grid_thw"],
                }
                model_inputs["vision_input"] = vision_input
            elif "pixel_values" in inputs and "image_grid_thw" in inputs:
                vision_input = {
                    "pixel_values": inputs["pixel_values"],
                    "image_grid_thw": inputs["image_grid_thw"],
                }
                model_inputs["vision_input"] = vision_input

            # Move to device/dtype
            for k, v in list(model_inputs.items()):
                if isinstance(v, torch.Tensor):
                    if v.dtype.is_floating_point:
                        model_inputs[k] = v.to(device=device, dtype=dtype)
                    else:
                        model_inputs[k] = v.to(device=device)
                elif isinstance(v, dict):
                    # Move nested tensors inside `vision_input` if present
                    for vk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            if vv.dtype.is_floating_point:
                                v[vk] = vv.to(device=device, dtype=dtype)
                            else:
                                v[vk] = vv.to(device=device)

            with torch.no_grad():
                out = model(**model_inputs)
                logits = out.logits  # [1, L, V]
                last = logits[0, -1, :]  # next-token distribution
                sel = last[option_ids].float().cpu().numpy()
                probs = softmax(sel)
                pred_idx = int(np.argmax(probs))

            gold_idx = options.index(ans.strip()) if ans.strip() in options else None
            if gold_idx is None:
                continue

            total += 1
            if pred_idx == gold_idx:
                correct += 1
            y_true_idx.append(gold_idx)
            y_pred_idx.append(pred_idx)

            # Accumulate pairwise scores/labels for binary curves
            for i in range(len(options)):
                y_score_pair.append(float(probs[i]))
                y_true_pair.append(1 if i == gold_idx else 0)

            # Write prediction record
            out_rec = {
                "vision_id": vid,
                "question": q,
                "answer": ans,
                "pred": options[pred_idx],
                "probs": {options[i]: float(probs[i]) for i in range(len(options))},
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    if total == 0:
        raise RuntimeError("No valid samples were processed. Check media-base and dataset vision_id paths.")

    accuracy = correct / total

    # Binary metrics over option-level pairs
    y_true_pair_arr = np.array(y_true_pair, dtype=np.int32)
    # Predicted positive only for the chosen option per question (1-of-4)
    # Build y_pred_pair by repeating per-sample one-hot predictions
    y_pred_pair_list: List[int] = []
    for p_idx in y_pred_idx:
        y_pred_pair_list.extend([1 if i == p_idx else 0 for i in range(4)])
    y_pred_pair_arr = np.array(y_pred_pair_list, dtype=np.int32)
    bin_metrics = compute_binary_pair_metrics(y_true_pair_arr, y_pred_pair_arr)

    # ROC / PR curves and AUCs
    y_score_pair_arr = np.array(y_score_pair, dtype=np.float32)
    fpr, tpr = roc_curve(y_true_pair_arr, y_score_pair_arr)
    roc_auc = auc_trapezoid(fpr, tpr)
    prec, rec = precision_recall_curve(y_true_pair_arr, y_score_pair_arr)
    pr_auc = auc_trapezoid(rec, prec)

    # Confusion matrix (multi-class)
    cm = compute_confusion_matrix(y_true_idx, y_pred_idx, n_classes=4)

    # Persist metrics
    metrics = {
        "num_samples": total,
        "accuracy": accuracy,
        "binary_pair_metrics": bin_metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "labels": ["A", "B", "C", "D"],
        "confusion_matrix": cm.tolist(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Visualizations
    draw_plots_roc_pr(roc_pr_path, fpr, tpr, roc_auc, prec, rec, pr_auc)
    draw_confusion_matrix(cm, cm_path, labels=["A", "B", "C", "D"]) 

    print(f"Done. Samples: {total}  Acc: {accuracy:.4f}")
    print(f"- Predictions: {preds_path}")
    print(f"- Metrics:     {metrics_path}")
    print(f"- ROC/PR:      {roc_pr_path}")
    print(f"- Confusion:   {cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
