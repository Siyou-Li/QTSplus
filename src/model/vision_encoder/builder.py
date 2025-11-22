import json
import os
import torch
from typing import Any, Dict, Optional
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel as Qwen2_5_VisionTransformerPretrainedModelBase

class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModelBase):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
    
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        # Return the output from the base implementation.
        # Without this return, callers receive None and downstream code fails.
        return super().forward(hidden_states, grid_thw, **kwargs)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.dtype)
        video_embeds = self.forward(pixel_values_videos, grid_thw=video_grid_thw)
        # split_sizes = (video_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        # video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds
    
def _try_load_vision_config_from_path(path: str) -> Optional[Dict[str, Any]]:
    """Best-effort load of Qwen2.5-VL vision `config.json`.

    Accepts either a directory containing `config.json` or a file path to a
    weights file. In the latter case, attempts to locate a sibling
    `config.json` in the same directory.
    """
    if not path:
        return None

    cfg_path = None
    if os.path.isdir(path):
        candidate = os.path.join(path, "config.json")
        if os.path.isfile(candidate):
            cfg_path = candidate
    else:
        # If a file is given (e.g., .../model.safetensors), look next to it
        base_dir = os.path.dirname(path)
        candidate = os.path.join(base_dir, "config.json")
        if os.path.isfile(candidate):
            cfg_path = candidate

    if cfg_path is None:
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower != "qwen2_5_vl_vision":
        raise ValueError(f"Unknown vision tower type: {vision_tower}")

    # Attempt to infer correct dimensions from the provided pretrained path
    pretrained_path = getattr(vision_tower_cfg, 'pretrain_vision_model', None)
    cfg_json = _try_load_vision_config_from_path(pretrained_path) if pretrained_path else None

    if cfg_json is not None:
        # Map json fields to Qwen2_5_VLVisionConfig kwargs (use json defaults when available)
        config = Qwen2_5_VLVisionConfig(
            hidden_size=cfg_json.get("hidden_size", 1280),
            out_hidden_size=cfg_json.get("out_hidden_size", cfg_json.get("hidden_size", 1280)),
            depth=cfg_json.get("depth", 32),
            intermediate_size=cfg_json.get("intermediate_size", 3420),
            num_heads=cfg_json.get("num_heads", 16),
            fullatt_block_indexes=cfg_json.get("fullatt_block_indexes", [7, 15, 23, 31]),
            in_channels=cfg_json.get("in_channels", cfg_json.get("in_chans", 3)),
            patch_size=cfg_json.get("patch_size", cfg_json.get("spatial_patch_size", 14)),
            spatial_merge_size=cfg_json.get("spatial_merge_size", 2),
            temporal_patch_size=cfg_json.get("temporal_patch_size", 2),
            tokens_per_second=cfg_json.get("tokens_per_second", 2),
            window_size=cfg_json.get("window_size", 112),
            initializer_range=cfg_json.get("initializer_range", 0.02),
        )
    else:
        # Fallback to a safe default (3B) when no config file is available
        # This keeps backwards-compatibility but different-scale checkpoints
        # should always provide a config.json alongside the weights.
        config = Qwen2_5_VLVisionConfig(
            hidden_size=1280,
            out_hidden_size=2048,
            depth=32,
            intermediate_size=3420,
            num_heads=16,
            fullatt_block_indexes=[7, 15, 23, 31],
        )

    return Qwen2_5_VisionTransformerPretrainedModel(config)
