#!/usr/bin/env python3
"""
Verify that generation WITHOUT a QTS layer matches Qwen2.5-VL-Instruct.

This script compares two paths on the same input (image or video):
  1) Baseline full model: transformers' Qwen2.5-VL-3B-Instruct
  2) Split pipeline without QTS: Qwen2.5-VL vision encoder + Qwen2.5-VL text LM,
     with vision features directly integrated into the text embeddings by replacing
     the placeholder tokens (no QTS selection or re-encode involved).

If both generations are identical, the script exits 0 and prints a success line.
On mismatch, it prints both outputs and exits with a non-zero code.

Default model paths use local directories under `pretrained_models/` and the
default input is `examples/example.mov` with a simple prompt.

Usage examples:
  - Video default:
      python script/verify_no_qts_equals_qwen.py

  - Specific video:
      python script/verify_no_qts_equals_qwen.py --video examples/example.mov --prompt "Describe this video."

  - Image:
      python script/verify_no_qts_equals_qwen.py --image examples/example.jpeg --prompt "Describe this image."

  - Adjust generation tokens and dtype:
      python script/verify_no_qts_equals_qwen.py --max-new-tokens 64 --dtype bf16

Note: This verifies the "no QTS layer" path by using the plain split
vision+text pipeline, not the QTS+ wrapper with enable_qts_plus=False.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable regardless of working dir
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoConfig
from safetensors.torch import load_file as safe_load_file
import json

# Local model components (no QTS)
from src.model.vision_encoder import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
from src.model.language_model import (
    Qwen2_5_VLTextForCausalLM,
)

from src.utils.qwen_vision_process import process_vision_info
from src.model.qts_plus_arch import qts_integrate_embeddings
from src.utils.integrate_embeddings import prepare_multimodal_inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify no-QTS generation equals Qwen2.5-VL-Instruct")
    p.add_argument("--full-model", default="pretrained_models/Qwen2.5-VL-7B-Instruct",
                   help="Path to full Qwen2.5-VL-Instruct (processor + full model)")
    p.add_argument("--vision-model", default="pretrained_models/Qwen2.5-VL-7B-Instruct-Vision",
                   help="Path to Qwen2.5-VL vision encoder")
    p.add_argument("--lm-model", default="pretrained_models/Qwen2.5-VL-7B-Instruct-LM",
                   help="Path to Qwen2.5-VL text LM")

    media = p.add_mutually_exclusive_group()
    media.add_argument("--video", type=str, default="examples/example.mov", help="Path to a video file")
    media.add_argument("--image", type=str, default=None, help="Path to an image file")

    p.add_argument("--prompt", type=str, default="Describe this video.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16",
                   help="Computation dtype to use (auto picks bf16 if cuda else fp32)")

    return p.parse_args()


def pick_device_and_dtype(dtype_choice: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dtype_choice == "auto":
        if device.type == "cuda":
            return device, torch.bfloat16
        return device, torch.float32
    return device, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_choice]


@torch.no_grad()
def build_inputs_embeds_with_qwen_integration(
    text_model,
    tokenizer,
    text: str,
    vision_features: torch.Tensor,
    placeholder_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate vision features by expanding a single placeholder into all features.

    Uses qts_integrate_embeddings (generic helper) to splice features into
    the token embeddings and adjust the attention mask accordingly.
    """
    tokenized = tokenizer([text], return_tensors="pt")
    dev = next(text_model.parameters()).device
    input_ids = tokenized["input_ids"].to(dev)
    attention_mask = tokenized.get("attention_mask").to(dev)

    embeds, attn, _labels = qts_integrate_embeddings(
        vision_features=vision_features,  # expects [T, D]
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        video_token_id=placeholder_token_id,
        text_model_embed_layer=text_model.get_input_embeddings(),
    )
    return embeds, attn

@torch.no_grad()
def build_inputs_embeds_with_placeholder_expansion(
    text_model,
    tokenizer,
    expanded_text: str,
    vision_features: torch.Tensor,
    placeholder_token_id: int,
):
    """Integrate features by expanding placeholders and replacing only those positions.

    Keeps any surrounding special tokens (e.g., <|vision_start|>, <|vision_end|>) intact.
    """
    dev = next(text_model.parameters()).device
    text_inputs = tokenizer(text=[expanded_text], padding=True, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(dev)
    attention_mask = text_inputs["attention_mask"].to(dev)
    embeds = text_model.get_input_embeddings()(input_ids)

    # Locate placeholder positions and ensure counts match features
    pos = (input_ids[0] == placeholder_token_id).nonzero(as_tuple=False).flatten()
    T = vision_features.shape[0]
    if int(pos.numel()) != T:
        raise ValueError(f"Expanded placeholders ({int(pos.numel())}) != features ({T})")

    vf = vision_features.to(embeds.device, embeds.dtype)
    for i in range(T):
        embeds[0, int(pos[i].item())] = vf[i]
    return embeds, attention_mask


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device, dtype = pick_device_and_dtype(args.dtype)

    # 1) Processor for chat templating, vision IO, and decoding
    processor = AutoProcessor.from_pretrained(args.full_model)

    # 2) Build a single-message conversation with media + text
    if args.image:
        media_path = Path(args.image)
        if not media_path.exists():
            print(f"[ERROR] Image not found: {media_path}", file=sys.stderr)
            return 2
        content = [{"type": "image", "image": str(media_path)}, {"type": "text", "text": args.prompt}]
    else:
        media_path = Path(args.video)
        if not media_path.exists():
            print(f"[ERROR] Video not found: {media_path}", file=sys.stderr)
            return 2
        content = [{"type": "video", "video": str(media_path), "max_pixels": 360 * 420, "fps": 1.0}, {"type": "text", "text": args.prompt}]

    messages = [{"role": "user", "content": content}]

    # 3) Prepare model inputs
    #    text string (with special placeholder tokens) and vision tensors
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

    # 4) Baseline full model generation (transformers /huggingface/)
    # Some local checkpoints may have extended token sets; allow size auto-adjust.
    # Inspect checkpoint embedding size to align config.vocab_size
    cfg = AutoConfig.from_pretrained(args.full_model)
    index_path = Path(args.full_model) / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            key = "model.language_model.embed_tokens.weight"
            shard = weight_map.get(key)
            if shard:
                shard_path = Path(args.full_model) / shard
                state = safe_load_file(str(shard_path))
                ckpt_w = state.get(key)
                if ckpt_w is not None and ckpt_w.ndim == 2:
                    ckpt_vocab = ckpt_w.shape[0]
                    if getattr(cfg, "vocab_size", None) != ckpt_vocab:
                        cfg.vocab_size = ckpt_vocab
        except Exception:
            # Fallback to config value
            pass
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.full_model, config=cfg
    )
    full_model.to(device=device, dtype=dtype).eval()
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                inputs[k] = v.to(device=device, dtype=dtype)
            else:
                inputs[k] = v.to(device=device)

    with torch.no_grad():
        out_full = full_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Trim the prompt tokens from full-model outputs before decoding
    trimmed_full = [o[len(inputs["input_ids"][0]):] for o in out_full]
    trimmed_full_cpu = [t.cpu() for t in trimmed_full]
    text_full = processor.batch_decode(
        trimmed_full_cpu,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 5) Split pipeline: vision encoder + text LM (no QTS)
    vision_model = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(args.vision_model)
    vision_model.to(device=device, dtype=dtype).eval()
    text_model = Qwen2_5_VLTextForCausalLM.from_pretrained(args.lm_model)
    text_model.to(device=device, dtype=dtype).eval()

    if "pixel_values_videos" in inputs:
        feats = vision_model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        feats_TD = feats[0] if feats.ndim == 3 else feats
        # Expand the single video token into T placeholders to match features
        T = feats_TD.shape[0]
        expanded_text = chat_text.replace(processor.video_token, processor.video_token * T, 1)
        text_inputs = processor.tokenizer(text=[expanded_text], padding=True, return_tensors="pt").to(text_model.device)
        mm_inputs = prepare_multimodal_inputs(
            vision_model_outputs=feats_TD.to(text_model.device, dtype=text_model.dtype),
            tokenizer_outputs=text_inputs,
            tokenizer=processor.tokenizer,
            text_embed_layer=text_model.get_input_embeddings(),
            video_grid_thw=inputs.get("video_grid_thw"),
        )
        embeds, attn_mask = mm_inputs["inputs_embeds"], mm_inputs["attention_mask"]
    else:
        feats = vision_model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        feats_TD = feats[0] if feats.ndim == 3 else feats
        # Expand the single image token into T placeholders to match features
        T = feats_TD.shape[0]
        expanded_text = chat_text.replace(processor.image_token, processor.image_token * T, 1)
        embeds, attn_mask = build_inputs_embeds_with_placeholder_expansion(
            text_model,
            processor.tokenizer,
            expanded_text,
            feats_TD,
            placeholder_token_id=processor.image_token_id,
        )
    embeds = embeds.to(device=device, dtype=dtype)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        out_split = text_model.generate(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # For inputs_embeds, returned sequences contain only new tokens.
    text_split = processor.batch_decode(
        out_split.cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Compare both text and token IDs for robustness
    same_text = (text_full == text_split)
    full_ids_trimmed = trimmed_full[0].to(out_split.device)
    same_ids = torch.equal(full_ids_trimmed, out_split[0])

    if same_text and same_ids:
        print("[OK] No-QTS split pipeline matches Qwen2.5-VL-Instruct output.")
        print(f"Output: {text_split}")
        return 0

    print("[MISMATCH] Outputs differ.")
    
    print("- Full model:")
    print(text_full)
    print("- Split (no QTS):")
    print(text_split)
    return 1


if __name__ == "__main__":
    sys.exit(main())
