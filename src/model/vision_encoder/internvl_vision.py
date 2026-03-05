from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from src.model.internvl.intern_vit.configuration_intern_vit import InternVisionConfig
from src.model.internvl.intern_vit.modeling_intern_vit import InternVisionModel


class InternVL2_5VisionConfig(PretrainedConfig):
    """
    Vision-tower-only config for InternVL2.5.

    This config is intentionally minimal: it stores the InternVisionConfig and the
    few extra knobs needed to reproduce InternVL's feature extraction pipeline
    (pixel shuffle + `mlp1` projection into the LM hidden space).
    """

    model_type = "internvl2_5_vision"
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Union[InternVisionConfig, Dict[str, Any]]] = None,
        llm_hidden_size: Optional[int] = None,
        select_layer: int = -1,
        force_image_size: Optional[int] = None,
        downsample_ratio: float = 0.5,
        ps_version: str = "v2",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["InternVisionModel"]}
        self.vision_config = (
            vision_config if isinstance(vision_config, InternVisionConfig) else InternVisionConfig(**vision_config)
        )

        self.select_layer = int(select_layer)
        self.force_image_size = int(force_image_size) if force_image_size is not None else None
        self.downsample_ratio = float(downsample_ratio)
        self.ps_version = str(ps_version)

        self.hidden_size = int(self.vision_config.hidden_size)
        # Expose output dim for QTSplus mm_projector sizing logic.
        self.out_hidden_size = int(llm_hidden_size) if llm_hidden_size is not None else int(self.hidden_size)
        self.llm_hidden_size = int(self.out_hidden_size)

        # Keep a minimal architectures hint so configs survive save/load cycles.
        self.architectures = ["InternVL2_5VisionTower"]

    @classmethod
    def from_internvl_chat_config(cls, cfg: Dict[str, Any]) -> "InternVL2_5VisionConfig":
        """
        Build a vision-tower config from a full InternVL chat `config.json` dict.
        """
        model_type = (cfg.get("model_type") or "").lower()
        if model_type == cls.model_type:
            return cls(**cfg)

        if model_type != "internvl_chat":
            raise ValueError(f"Unsupported source config model_type={cfg.get('model_type')}")

        vision_cfg = cfg.get("vision_config") or {}
        llm_cfg = cfg.get("llm_config") or {}
        llm_hidden = llm_cfg.get("hidden_size")
        if not isinstance(llm_hidden, int):
            raise ValueError("Expected `llm_config.hidden_size` in InternVL chat config.")

        return cls(
            vision_config=vision_cfg,
            llm_hidden_size=int(llm_hidden),
            select_layer=int(cfg.get("select_layer", -1)),
            force_image_size=cfg.get("force_image_size", None),
            downsample_ratio=float(cfg.get("downsample_ratio", 0.5)),
            ps_version=str(cfg.get("ps_version", "v2")),
        )

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class InternVL2_5VisionTower(PreTrainedModel):
    """
    Vision tower used by QTSplus with InternVL2.5 checkpoints.

    It wraps InternVL's InternVisionModel and applies:
    1) dropping CLS token
    2) pixel shuffle downsampling
    3) `mlp1` projection into LM hidden space
    """

    config_class = InternVL2_5VisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["InternVisionModel", "InternVisionEncoderLayer"]

    def __init__(self, config: InternVL2_5VisionConfig):
        super().__init__(config)

        vision_cfg = config.vision_config
        if config.force_image_size is not None:
            vision_cfg = copy.deepcopy(vision_cfg)
            vision_cfg.image_size = int(config.force_image_size)

        self.vision_model = InternVisionModel(vision_cfg)
        self.select_layer = int(config.select_layer)
        self.downsample_ratio = float(config.downsample_ratio)
        self.ps_version = str(config.ps_version)

        vit_hidden_size = int(vision_cfg.hidden_size)
        llm_hidden_size = int(config.out_hidden_size)
        mlp_in = vit_hidden_size * int(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.post_init()

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.select_layer == -1:
            vit_out = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_out = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]

        vit_out = vit_out[:, 1:, :]  # drop CLS

        h = w = int(vit_out.shape[1] ** 0.5)
        vit_out = vit_out.reshape(vit_out.shape[0], h, w, -1)
        vit_out = self.pixel_shuffle(vit_out, scale_factor=self.downsample_ratio)
        vit_out = vit_out.reshape(vit_out.shape[0], -1, vit_out.shape[-1])
        vit_out = self.mlp1(vit_out)
        return vit_out

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.extract_feature(pixel_values)

    def forward(self, pixel_values: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.get_image_features(pixel_values)


__all__ = [
    "InternVL2_5VisionConfig",
    "InternVL2_5VisionTower",
]

