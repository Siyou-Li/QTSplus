import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional


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


def _load_and_apply_safetensors_prefix(
    model,
    shard_path: str,
    *,
    keep_prefixes: Iterable[str],
    strip_prefix: Optional[str] = None,
) -> None:
    """
    Load tensors from a shard and apply them to ``model`` via ``load_state_dict``.

    Parameters
    ----------
    model:
        A torch nn.Module.
    shard_path:
        Path to a `.safetensors` shard.
    keep_prefixes:
        Key prefixes to keep from the shard.
    strip_prefix:
        If set, strip this prefix from keys before loading (used for language_model.*).
    """
    import torch
    from safetensors.torch import safe_open

    keep_prefixes = tuple(keep_prefixes)
    part: Dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith(keep_prefixes):
                continue
            out_k = k
            if strip_prefix is not None and out_k.startswith(strip_prefix):
                out_k = out_k[len(strip_prefix) :]
            part[out_k] = f.get_tensor(k)

    if part:
        model.load_state_dict(part, strict=False)


def split_weights(model_path: str, vision_out: str, lm_out: str, *, max_shard_size: str = "5GB") -> None:
    """
    Split InternVL2.5 checkpoints into:
    - ``*-Vision``: vision encoder + InternVL projection (mlp1) for feature extraction
    - ``*-LM``: underlying causal LM weights (InternLM2/Qwen2/Llama depending on the checkpoint)

    This implementation avoids importing InternVL's original full chat model and instead
    loads weights directly from the sharded `.safetensors` files.
    """
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer

    from src.model.internvl.internlm2.configuration_internlm2 import InternLM2Config
    from src.model.internvl.internlm2.modeling_internlm2 import InternLM2ForCausalLM
    from src.model.vision_encoder.internvl_vision import InternVL2_5VisionConfig, InternVL2_5VisionTower

    model_path = str(model_path)
    cfg_path = os.path.join(model_path, "config.json")
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json under {model_path}")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {model_path}")

    cfg = _load_json(cfg_path)
    index = _load_json(idx_path)

    llm_cfg = cfg.get("llm_config") or {}
    if (llm_cfg.get("model_type") or "").lower() not in {"internlm2"}:
        raise ValueError(
            f"Only InternVL2.5 checkpoints with llm_config.model_type='internlm2' are supported by this script. "
            f"Got: {llm_cfg.get('model_type')}"
        )

    dtype = _resolve_dtype(cfg.get("torch_dtype") or llm_cfg.get("torch_dtype"))
    if dtype is None:
        dtype = torch.bfloat16

    Path(vision_out).mkdir(parents=True, exist_ok=True)
    Path(lm_out).mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Build and load LM model
    # -----------------------
    lm_config = InternLM2Config(**llm_cfg)
    lm_model = InternLM2ForCausalLM(lm_config)
    lm_model.to(dtype=dtype)
    lm_model.eval()

    # --------------------------
    # Build and load vision tower
    # --------------------------
    vision_cfg = InternVL2_5VisionConfig.from_internvl_chat_config(cfg)
    vision_tower = InternVL2_5VisionTower(vision_cfg)
    vision_tower.to(dtype=dtype)
    vision_tower.eval()

    # Load weights from shards (4 files for 8B)
    for shard_name in _iter_unique_shards(index):
        shard_path = os.path.join(model_path, shard_name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing shard file referenced by index: {shard_path}")

        _load_and_apply_safetensors_prefix(
            lm_model,
            shard_path,
            keep_prefixes=("language_model.",),
            strip_prefix="language_model.",
        )
        _load_and_apply_safetensors_prefix(
            vision_tower,
            shard_path,
            keep_prefixes=("vision_model.", "mlp1."),
            strip_prefix=None,
        )

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
        # Best-effort: config-only setups can copy preprocessor_config.json manually.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Split InternVL2.5 weights into LM and Vision components")
    parser.add_argument("--model_path", required=True, help="Path to original InternVL2.5 checkpoint directory")
    parser.add_argument("--vision_out", default=None, help="Output directory for vision tower model")
    parser.add_argument("--lm_out", default=None, help="Output directory for language model")
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

