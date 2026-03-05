import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dtype(dtype_str: Optional[str]):
    import torch

    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if s in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    return None


def _iter_unique_shards(index_json: dict) -> Iterable[str]:
    weight_map = index_json.get("weight_map") or {}
    seen = set()
    for _, shard in weight_map.items():
        if shard not in seen:
            seen.add(shard)
            yield shard


def _build_qwen2_config_from_llava_json(cfg: Dict[str, Any]):
    """
    Build a `transformers.Qwen2Config` from an LLaVA(-Video) config.json.

    The LLaVA checkpoint stores Qwen2 LM hyperparameters at the top-level
    alongside multimodal fields. We transfer only fields that exist on Qwen2Config.
    """
    from transformers import Qwen2Config

    qcfg = Qwen2Config()
    for k, v in cfg.items():
        if hasattr(qcfg, k):
            try:
                setattr(qcfg, k, v)
            except Exception:
                # Best-effort: ignore incompatible fields
                pass

    # Ensure vocab/pad/eos/bos ids are consistent when present in the source config
    for k in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id"):
        if k in cfg and hasattr(qcfg, k):
            try:
                setattr(qcfg, k, cfg[k])
            except Exception:
                pass

    # Transformers>=4.57 validates `layer_types` length against `num_hidden_layers` during init,
    # but attribute mutation above doesn't keep them in sync. Prefer dropping `layer_types`
    # to match the original LLaVA config (which doesn't define it).
    layer_types = getattr(qcfg, "layer_types", None)
    num_hidden_layers = getattr(qcfg, "num_hidden_layers", None)
    if isinstance(layer_types, list) and isinstance(num_hidden_layers, int) and len(layer_types) != num_hidden_layers:
        try:
            setattr(qcfg, "layer_types", None)
        except Exception:
            pass
    return qcfg


def _infer_siglip_vision_tower_name(cfg: Dict[str, Any]) -> Optional[str]:
    name = cfg.get("mm_vision_tower", None)
    if isinstance(name, str) and len(name) > 0:
        return name
    return None


def _load_siglip_vision_config(
    cfg: Dict[str, Any],
    *,
    fallback_hidden_size: int,
    fallback_image_size: int,
) -> Tuple[dict, dict]:
    """
    Return a `(siglip_cfg_dict, extra_cfg_dict)` tuple.

    We try to fetch the official SigLIP config from HF via `from_pretrained` so
    that num_heads/num_layers/intermediate_size match the original model.
    If that fails (offline env), we fall back to a shape-consistent config.
    """
    siglip_cfg_dict: dict = {}
    extra: dict = {}

    name = _infer_siglip_vision_tower_name(cfg)
    if name is not None:
        try:
            from transformers import SiglipVisionConfig

            siglip_cfg = SiglipVisionConfig.from_pretrained(name)
            siglip_cfg_dict = siglip_cfg.to_dict()
        except Exception:
            siglip_cfg_dict = {}

    if not siglip_cfg_dict:
        # Minimal fallback that matches this checkpoint (so400m-p14-384).
        # hidden_size comes from LLaVA config `mm_hidden_size`.
        siglip_cfg_dict = {
            "hidden_size": int(fallback_hidden_size),
            "intermediate_size": 4304,
            "num_hidden_layers": 26,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": int(fallback_image_size),
            "patch_size": 14,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
        }

    # Carry over a few LLaVA-specific feature extraction knobs for the vision-head config.
    for k in (
        "mm_vision_select_layer",
        "mm_vision_select_feature",
        "mm_patch_merge_type",
        "mm_newline_position",
        "image_aspect_ratio",
        "image_grid_pinpoints",
    ):
        if k in cfg:
            extra[k] = cfg[k]
    return siglip_cfg_dict, extra


def split_weights(model_path: str, vision_out: str, lm_out: str, *, max_shard_size: str = "5GB") -> None:
    """
    Split `pretrained_models/LLaVA-Video-7B-Qwen2` into:
      - `*-LM`: Qwen2 causal LM weights (no vision tower / projector)
      - `*-Vision`: SigLIP vision tower + mm_projector (+ image_newline) for feature extraction
    """
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, Qwen2ForCausalLM

    from src.model.vision_encoder.llava_siglip_vision import LlavaSiglipVisionConfig, LlavaSiglipVisionTower

    model_path = str(model_path)
    cfg_path = os.path.join(model_path, "config.json")
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json under {model_path}")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {model_path}")

    cfg = _load_json(cfg_path)
    index = _load_json(idx_path)

    dtype = _resolve_dtype(cfg.get("torch_dtype"))
    if dtype is None:
        dtype = torch.bfloat16

    Path(vision_out).mkdir(parents=True, exist_ok=True)
    Path(lm_out).mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Build and load LM model
    # -----------------------
    lm_config = _build_qwen2_config_from_llava_json(cfg)
    lm_model = Qwen2ForCausalLM(lm_config)
    lm_model.to(dtype=dtype)
    lm_model.eval()

    # --------------------------
    # Build and load vision tower
    # --------------------------
    mm_hidden = int(cfg.get("mm_hidden_size", 0) or 0)
    llm_hidden = int(cfg.get("hidden_size", 0) or 0)
    if mm_hidden <= 0 or llm_hidden <= 0:
        raise ValueError(f"Invalid mm_hidden_size/hidden_size in config.json: {mm_hidden=}, {llm_hidden=}")

    # Prefer the processor's target size when present (usually 384 for SigLIP).
    fallback_image_size = 384
    try:
        pp = _load_json(os.path.join(model_path, "preprocessor_config.json"))
        size = pp.get("size", None)
        if isinstance(size, dict) and "height" in size:
            fallback_image_size = int(size["height"])
    except Exception:
        pass

    siglip_cfg_dict, extra_cfg = _load_siglip_vision_config(
        cfg,
        fallback_hidden_size=mm_hidden,
        fallback_image_size=fallback_image_size,
    )
    vision_config = LlavaSiglipVisionConfig(
        vision_config=siglip_cfg_dict,
        llm_hidden_size=llm_hidden,
        **extra_cfg,
    )
    vision_tower = LlavaSiglipVisionTower(vision_config)
    vision_tower.to(dtype=dtype)
    vision_tower.eval()

    # Load weights from shards
    for shard_name in _iter_unique_shards(index):
        shard_path = os.path.join(model_path, shard_name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing shard file referenced by index: {shard_path}")

        # LM: keep everything except vision_tower/mm_projector/image_newline tensors.
        # (The checkpoint uses top-level keys: model.*, lm_head.*)
        import torch
        from safetensors.torch import safe_open

        part_lm: Dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.vision_tower.") or k.startswith("model.mm_projector.") or k == "model.image_newline":
                    continue
                if not (k.startswith("model.") or k.startswith("lm_head.")):
                    continue
                part_lm[k] = f.get_tensor(k)
        if part_lm:
            lm_model.load_state_dict(part_lm, strict=False)

        # Vision: strip leading `model.` and also collapse `model.vision_tower.` -> `vision_tower.`
        # so it matches `LlavaSiglipVisionTower` state_dict keys.
        # We load only vision + projector (+ newline) tensors.
        part: Dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if not (
                    k.startswith("model.vision_tower.")
                    or k.startswith("model.mm_projector.")
                    or k == "model.image_newline"
                ):
                    continue
                out_k = k
                if out_k.startswith("model.vision_tower."):
                    out_k = out_k[len("model.vision_tower.") :]
                elif out_k.startswith("model."):
                    out_k = out_k[len("model.") :]
                part[out_k] = f.get_tensor(k)
        if part:
            vision_tower.load_state_dict(part, strict=False)

    # Save LM + tokenizer
    lm_model.save_pretrained(lm_out, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(lm_out)

    # Save vision tower + image processor
    vision_tower.save_pretrained(vision_out, safe_serialization=True, max_shard_size=max_shard_size)
    try:
        img_proc = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        img_proc.save_pretrained(vision_out)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Split LLaVA-Video-7B-Qwen2 into LM and Vision components")
    parser.add_argument("--model_path", required=True, help="Path to original LLaVA-Video-7B-Qwen2 checkpoint dir")
    parser.add_argument("--vision_out", default=None, help="Output directory for the vision head")
    parser.add_argument("--lm_out", default=None, help="Output directory for the language model")
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Max shard size for saving (passed to HuggingFace `save_pretrained`).",
    )
    args = parser.parse_args()

    if args.vision_out is None:
        args.vision_out = args.model_path.rstrip("/").rstrip("\\") + "-Vision"
    if args.lm_out is None:
        args.lm_out = args.model_path.rstrip("/").rstrip("\\") + "-LM"

    split_weights(args.model_path, args.vision_out, args.lm_out, max_shard_size=args.max_shard_size)


if __name__ == "__main__":
    main()
