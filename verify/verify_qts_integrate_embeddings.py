"""
Verify inputs, outputs, and labels for qts_integrate_embeddings during training.

This script reproduces the model's multimodal preparation steps up to the
integration point, calls qts_integrate_embeddings directly, and checks:

- Shapes and dtypes of inputs and outputs
- Correct expansion/replacement around the <|video_pad|> token(s)
- Attention mask update (only {0,1}) and shape consistency
- Label masking of inserted vision tokens (-100)
- Equality of text embedding segments before/after the insertion

Run from the repo root. Example:

  python script/verify_qts_integrate_embeddings.py \
    --train-base-path datasets/ShareGPTVideo_train/data/train_300k \
    --train-jsonl-path datasets/ShareGPTVideoChoice/choice_eval.jsonl \
    --pretrain-lm-model pretrained_models/Qwen2.5-VL-3B-Instruct-LM \
    --vision-processor pretrained_models/Qwen2.5-VL-3B-Instruct-Vision \
    --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

import torch

from src.dataset.sharegptvideo_choice_dataset import ShareGPTVideoChoiceDataset
from src.model.language_model import QTSplusQwen2_5_VLTextForCausalLM
from src.model.vision_encoder import Qwen2_5_VLVisionProcessor
from src.model.qts_plus_arch import qts_integrate_embeddings


def _fmt_shape(x: torch.Tensor | None) -> str:
    if x is None:
        return "None"
    try:
        return str(tuple(x.shape))
    except Exception:
        return "?"

def _fmt_list(xs, max_n: int = 6) -> str:
    xs = list(xs)
    if len(xs) <= max_n:
        return str(xs)
    return str(xs[:max_n] + ["..."])


def _device_from_arg(arg: str) -> torch.device:
    arg = (arg or "auto").lower()
    if arg == "cpu":
        return torch.device("cpu")
    if arg.startswith("cuda") or arg == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def verify_one(
    model: QTSplusQwen2_5_VLTextForCausalLM,
    processor: Qwen2_5_VLVisionProcessor,
    sample: Dict[str, Any],
    device: torch.device,
    atol: float,
    rtol: float,
) -> int:
    errs = 0

    # Resolve video token id from processor/tokenizer to avoid mismatch
    tok = processor.tokenizer
    video_tok_str = getattr(processor, "video_token", "<|video_pad|>")
    try:
        video_token_id = int(tok.convert_tokens_to_ids(video_tok_str))
    except Exception:
        video_token_id = int(getattr(getattr(model, 'config', object()), 'video_token_id', 151656))

    # Prepare inputs
    input_ids = sample["input_ids"].unsqueeze(0).to(device)  # [1, S]
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)  # [1, S]
    labels = sample["labels"].unsqueeze(0).to(device)  # [1, S]
    question_input_ids = sample.get("question_input_ids", None)
    if question_input_ids is not None:
        question_input_ids = question_input_ids.unsqueeze(0).to(device)
    vision_input = sample.get("vision_input", None)

    if not isinstance(vision_input, dict) or "pixel_values_videos" not in vision_input:
        print("[skip] Sample is not a video-based item with pixel_values_videos; skipping.")
        return 0

    # Resolve model pieces
    model = model.to(device)
    model.eval()
    vt = model.get_vision_tower()
    qts = model.get_qts_plus_tower()
    embed_tokens = model.get_model().get_input_embeddings()
    hidden_size = embed_tokens.embedding_dim

    # Compute raw text embeddings (full sequence) for later equality checks
    full_text_embeds = embed_tokens(input_ids.long())  # [1, S, D]

    # Derive vision features using the same path as prepare_inputs_for_multimodal
    if vt is None:
        print("[ERR] Vision tower is not initialized on the model.")
        return 1
    pv = vision_input["pixel_values_videos"].to(vt.device)
    gthw = vision_input["video_grid_thw"].to(vt.device)
    vision_features = vt.get_video_features(pv, gthw)  # [1, T, Dv] or [T, Dv]
    if vision_features.ndim == 2:
        vision_features = vision_features.unsqueeze(0)  # [1, T, Dv]

    # Text embeddings for the question (no answer) are used by QTS+ selector
    if question_input_ids is None:
        print("[ERR] question_input_ids missing in dataset sample; cannot verify without leakage.")
        return 1
    text_q_embeds = embed_tokens(question_input_ids.long())  # [1, Sq, D]
    vision_features = vision_features.to(dtype=text_q_embeds.dtype)

    # Run QTS+ to select/re-encode features
    qts_out = qts(vision_features, text_q_embeds, mode="train")
    Z = qts_out["Z"][0]  # [T_sel, D]
    t_sel = int(Z.shape[0])

    # Locate placeholder(s) in the original ids
    vid_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    s = int(input_ids.shape[1])

    # Concise INPUT summary
    attn1 = int(attention_mask.sum().item())
    sup = int(((labels != -100) & (attention_mask == 1)).sum().item())
    print(
        f"[in] ids={_fmt_shape(input_ids)} attn1={attn1} labels_sup={sup} vid_tok={video_token_id} pos={_fmt_list(vid_pos.tolist())}"
    )

    # Call integration
    inputs_embeds, attn_out, labels_out = qts_integrate_embeddings(
        vision_features=Z,  # [T_sel, D]
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        video_token_id=video_token_id,
        text_model_embed_layer=embed_tokens,
        video_grid_thw=gthw,
    )

    # Basic shape checks
    assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1 and inputs_embeds.shape[2] == hidden_size
    assert attn_out.shape[0] == 1 and attn_out.shape[1] == inputs_embeds.shape[1]
    assert labels_out is None or (labels_out.shape[0] == 1 and labels_out.shape[1] == inputs_embeds.shape[1])

    # Mask validity
    uniq = torch.unique(attn_out)
    if not all(int(x.item()) in (0, 1) for x in uniq):
        print(f"[ERR] attention_mask has non-binary values: {uniq.tolist()}")
        errs += 1

    # Path-dependent checks
    if vid_pos.numel() == 1:
        insert_idx = int(vid_pos.item())
        expected_len = s - 1 + t_sel
        final_len = int(inputs_embeds.shape[1])
        if final_len != expected_len:
            print(f"[ERR] final length {final_len} != expected {expected_len} (single-token expansion)")
            errs += 1

        # Pre and post consistency of text embeddings
        pre_ref = full_text_embeds[:, :insert_idx, :]
        post_ref = full_text_embeds[:, insert_idx + 1 :, :]
        pre_out = inputs_embeds[:, :insert_idx, :]
        post_out = inputs_embeds[:, insert_idx + t_sel :, :]
        pre_eq = torch.allclose(pre_ref, pre_out, rtol=rtol, atol=atol)
        post_eq = torch.allclose(post_ref, post_out, rtol=rtol, atol=atol)
        if not pre_eq:
            print("[ERR] pre-insertion text embeddings differ from reference")
            errs += 1
        if not post_eq:
            print("[ERR] post-insertion text embeddings differ from reference")
            errs += 1

        # Inserted window should align to selected vision features Z
        ins_out = inputs_embeds[:, insert_idx : insert_idx + t_sel, :]
        ins_eq = torch.allclose(ins_out.squeeze(0), Z.to(ins_out.dtype), rtol=rtol, atol=atol)
        if not ins_eq:
            print("[ERR] inserted vision feature slice does not match Z")
            errs += 1

        # Labels on inserted region must be -100
        if labels_out is not None:
            ins_labels = labels_out[:, insert_idx : insert_idx + t_sel]
            labels_ok = torch.all(ins_labels == -100).item()
            if not labels_ok:
                print("[ERR] inserted region labels must be -100")
                errs += 1
        
        # Attention mask on inserted region must be 1
        ins_mask = attn_out[:, insert_idx : insert_idx + t_sel]
        mask_ok = torch.all(ins_mask == 1).item()
        if not mask_ok:
            print("[ERR] inserted region attention_mask must be 1")
            errs += 1

        # Concise OUTPUT summary
        delta = final_len - s
        print(
            f"[out] embeds={_fmt_shape(inputs_embeds)} attn={_fmt_shape(attn_out)} labels={_fmt_shape(labels_out)} "
            f"insert=[{insert_idx}:{insert_idx + t_sel}) Δlen={delta} pre=={pre_eq} post=={post_eq} ins==Z:{ins_eq} labels_ins=-100:{labels_ok} mask_ins=1:{mask_ok}"
        )
    else:
        # Multi-token template fallback
        m = int(vid_pos.numel())
        n = t_sel
        if n > m:
            print("[ERR] selected features exceed available placeholders (N > M)")
            errs += 1
        # Replace up to n placeholders in-place and mask labels at those spots
        # Check that logits-length equals original length (if n == m) or shorter if placeholders dropped
        final_len = int(inputs_embeds.shape[1])
        if n < m and final_len != s - (m - n):
            print(f"[ERR] final length {final_len} inconsistent with dropping {m-n} placeholders from {s}")
            errs += 1
        if labels_out is not None:
            kept_vid_pos = vid_pos[:n]
            lab_positions = labels_out[0, kept_vid_pos]
            labels_ok = torch.all(lab_positions == -100).item()
            if not labels_ok:
                print("[ERR] labels at replaced placeholder positions must be -100")
                errs += 1

        # Concise OUTPUT summary for multi-token case
        delta = final_len - s
        print(
            f"[out] embeds={_fmt_shape(inputs_embeds)} attn={_fmt_shape(attn_out)} labels={_fmt_shape(labels_out)} "
            f"placeholders={m} selected={n} Δlen={delta} labels_vidpos=-100:{labels_ok if labels_out is not None else 'n/a'}"
        )

    # Sanity: finiteness
    for name, t in ("inputs_embeds", inputs_embeds), ("attention_mask", attn_out), ("labels", labels_out if labels_out is not None else torch.tensor([0])):
        if not torch.isfinite(torch.as_tensor(t)).all():
            print(f"[ERR] {name} contains NaN/Inf")
            errs += 1

    if errs == 0:
        print("[verify] qts_integrate_embeddings checks passed.")
    return errs


def main():
    p = argparse.ArgumentParser(description="Verify qts_integrate_embeddings I/O correctness")
    p.add_argument("--train-base-path", default="datasets/ShareGPTVideo_train/data/train_300k_480p", help="Base path to training media root")
    p.add_argument("--train-jsonl-path", default="datasets/ShareGPTVideoChoice/choice_eval.jsonl", help="Relative or absolute JSONL path under base")
    p.add_argument("--pretrain-lm-model", default="pretrained_models/Qwen2.5-VL-3B-Instruct-LM")
    p.add_argument("--vision-processor", default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vision-tower", default="qwen2_5_vl_vision")
    p.add_argument("--pretrain-vision-model", default=None, help="Path to vision weights (.safetensors or .bin)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    args = p.parse_args()

    device = _device_from_arg(args.device)

    processor = Qwen2_5_VLVisionProcessor.from_pretrained(args.vision_processor)
    tok = processor.tokenizer
    tok.pad_token = "<|endoftext|>"
    tok.eos_token = "<|im_end|>"
    tok.bos_token = "<|endoftext|>"
    tok.padding_side = "right"

    # Make jsonl relative to base if applicable
    rel_jsonl = args.train_jsonl_path
    try:
        if os.path.isabs(args.train_jsonl_path) and os.path.commonpath([args.train_base_path, args.train_jsonl_path]) == args.train_base_path:
            rel_jsonl = os.path.relpath(args.train_jsonl_path, start=args.train_base_path)
    except Exception:
        pass

    ds = ShareGPTVideoChoiceDataset(
        base_path=args.train_base_path,
        jsonl_path=rel_jsonl,
        processor=processor,
        max_length=args.max_length,
        local_rank=0,
        train=False,
    )
    if len(ds) == 0:
        print("Dataset is empty or filtered out.")
        sys.exit(1)

    print(f"[env] device={device}  samples={len(ds)}")
    print("[load] model ...")
    model = QTSplusQwen2_5_VLTextForCausalLM.from_pretrained(args.pretrain_lm_model).to(device)
    model.eval()

    # Ensure vision + QTS+ modules are initialized so get_vision_tower() is available
    try:
        margs = SimpleNamespace(
            vision_tower=args.vision_tower,
            enable_qts_plus=True,
            lm_embed_size=int(getattr(model.config, "hidden_size", 2048)),
            vision_embed_size=int(getattr(model.config, "vision_embed_size", 2048)),
            project_text_if_needed=False,
            qts_plus_n_heads=8,
            qts_plus_tau_s=0.1,
            qts_plus_nmax=2560,
            qts_plus_rho_min=0.05,
            qts_plus_rho_max=0.5,
            qts_plus_block_dropout=0.0,
            qts_plus_reencode=True,
            qts_plus_scoring_layers=1,
            qts_plus_reencode_layers=1,
            freeze_vision_model=False,
            pretrain_vision_model=args.pretrain_vision_model,
            # Loss weights (not used directly here but expected downstream)
            lambda_t=1.0,
            lambda_m=1.7,
            lambda_s=0.05,
            # Unused in this script
            tune_mm_mlp_adapter=False,
        )
        model.get_model().initialize_vision_modules(margs)
    except Exception as e:
        print(f"[warn] initialize_vision_modules failed: {e}")

    # Pick the first valid video sample
    sample = None
    for i in range(min(128, len(ds))):
        s = ds[i]
        if s is not None and isinstance(s.get("vision_input", None), dict) and "pixel_values_videos" in s["vision_input"]:
            sample = s
            break
    if sample is None:
        print("No suitable video sample found in the provided dataset.")
        sys.exit(1)

    errs = verify_one(model, processor, sample, device, atol=args.atol, rtol=args.rtol)
    sys.exit(1 if errs > 0 else 0)


if __name__ == "__main__":
    main()
