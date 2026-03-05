from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers import SiglipVisionConfig, SiglipVisionModel
except Exception:  # pragma: no cover
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig  # type: ignore
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel  # type: ignore


class LlavaSiglipVisionConfig(PretrainedConfig):
    """
    Vision-head-only config for LLaVA(-Video) Qwen2 checkpoints.

    Stores:
      - SigLIP vision config
      - LLaVA-NeXT style feature extraction knobs
      - mm_projector output dim (LM hidden size)
    """

    model_type = "llava_siglip_vision"
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Union[SiglipVisionConfig, Dict[str, Any]]] = None,
        llm_hidden_size: Optional[int] = None,
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = "patch",
        mm_patch_merge_type: str = "spatial_unpad",
        mm_newline_position: str = "grid",
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["SiglipVisionModel"]}
        self.vision_config = (
            vision_config
            if isinstance(vision_config, SiglipVisionConfig)
            else SiglipVisionConfig(**vision_config)
        )

        self.mm_vision_select_layer = int(mm_vision_select_layer)
        self.mm_vision_select_feature = str(mm_vision_select_feature)
        self.mm_patch_merge_type = str(mm_patch_merge_type)
        self.mm_newline_position = str(mm_newline_position)
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints

        self.hidden_size = int(self.vision_config.hidden_size)
        self.mm_hidden_size = int(self.hidden_size)

        self.out_hidden_size = int(llm_hidden_size) if llm_hidden_size is not None else int(self.hidden_size)
        self.llm_hidden_size = int(self.out_hidden_size)

        self.architectures = ["LlavaSiglipVisionTower"]

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LlavaSiglipVisionTower(PreTrainedModel):
    """
    Vision head used by QTSplus for LLaVA-Video-7B-Qwen2.

    Pipeline (LLaVA-NeXT style):
      1) SigLIP vision tower forward (optionally selecting an intermediate layer)
      2) mm_projector (mlp2x_gelu)
      3) optional `image_newline` insertion on a patch grid
    """

    config_class = LlavaSiglipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SiglipVisionModel", "SiglipEncoderLayer"]

    def __init__(self, config: LlavaSiglipVisionConfig) -> None:
        super().__init__(config)

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.select_layer = int(config.mm_vision_select_layer)
        self.select_feature = str(config.mm_vision_select_feature)
        self.patch_merge_type = str(config.mm_patch_merge_type)
        self.newline_position = str(config.mm_newline_position)

        vit_hidden = int(config.vision_config.hidden_size)
        llm_hidden = int(config.out_hidden_size)

        self.mm_projector = nn.Sequential(
            nn.Linear(vit_hidden, llm_hidden, bias=True),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden, bias=True),
        )
        # LLaVA-NeXT uses a learned newline embedding in LM hidden space.
        self.image_newline = nn.Parameter(torch.zeros(llm_hidden))

        self.post_init()

    def _select_vision_hidden(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # SigLIP has no CLS token; outputs are patch tokens only: [B, N, D]
        if self.select_layer == -1:
            out = self.vision_tower(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
            return out.last_hidden_state
        out = self.vision_tower(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        return out.hidden_states[self.select_layer]

    def _grid_newline(self, feats: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Insert `image_newline` after each spatial row: [H*W, D] -> [H*(W+1), D].
        """
        if feats.ndim != 3:
            return feats
        b, n, d = feats.shape
        patch = int(getattr(self.config.vision_config, "patch_size", 14))
        h_in = int(pixel_values.shape[-2])
        w_in = int(pixel_values.shape[-1])
        h = max(1, (h_in - patch) // patch + 1)
        w = max(1, (w_in - patch) // patch + 1)
        if h * w != n:
            side = int(n**0.5)
            if side * side == n:
                h = w = side
            else:
                h, w = 1, n
        feats = feats.reshape(b, h, w, d)
        nl = self.image_newline.to(device=feats.device, dtype=feats.dtype).view(1, 1, 1, d)
        nl = nl.expand(b, h, 1, d)
        feats = torch.cat([feats, nl], dim=2)
        return feats.reshape(b, h * (w + 1), d)

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Match the vision tower dtype (bf16/fp16) for Flash-Attn compatibility and speed.
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dtype != self.dtype:
            pixel_values = pixel_values.to(dtype=self.dtype)
        feats = self._select_vision_hidden(pixel_values)
        if self.select_feature not in {"patch"}:
            raise ValueError(f"Unsupported mm_vision_select_feature={self.select_feature}; expected 'patch'.")

        feats = self.mm_projector(feats)
        if self.newline_position == "grid":
            feats = self._grid_newline(feats, pixel_values)
        return feats

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.extract_feature(pixel_values)

    def forward(self, pixel_values: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.get_image_features(pixel_values)


__all__ = [
    "LlavaSiglipVisionConfig",
    "LlavaSiglipVisionTower",
]
