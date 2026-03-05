# -*- encoding: utf-8 -*-
# QTSplus wrapper for Qwen2 Causal LM (used by LLaVA-Video-7B-Qwen2-LM)

from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
except Exception:  # pragma: no cover
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # type: ignore
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model  # type: ignore

from ..qts_plus_arch import QTSplusMetaForCausalLM, QTSplusMetaModel


class QTSplusQwen2_CausalLM_Config(Qwen2Config):
    model_type = "qts_plus_qwen2_causal_lm"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # Allow loading from a plain Qwen2 checkpoint config.
        if str(config_dict.get("model_type") or "").lower() == "qwen2":
            config_dict["model_type"] = cls.model_type

        # Transformers>=4.57 validates `layer_types` length against `num_hidden_layers`.
        # Some converted checkpoints (e.g. from LLaVA configs) may carry a stale default
        # `layer_types` list that doesn't match the real layer count.
        layer_types = config_dict.get("layer_types", None)
        num_hidden_layers = config_dict.get("num_hidden_layers", None)
        if isinstance(layer_types, list):
            if isinstance(num_hidden_layers, int):
                if len(layer_types) != num_hidden_layers:
                    warnings.warn(
                        f"Config mismatch: num_hidden_layers={num_hidden_layers} but layer_types has "
                        f"{len(layer_types)} entries. Auto-fixing to enable loading.",
                        UserWarning,
                    )
                    if len(layer_types) > num_hidden_layers:
                        config_dict["layer_types"] = layer_types[:num_hidden_layers]
                    else:
                        pad = layer_types[-1] if layer_types else "full_attention"
                        config_dict["layer_types"] = layer_types + [pad] * (num_hidden_layers - len(layer_types))
            else:
                # If layer_types is explicitly provided but num_hidden_layers is missing, infer it.
                config_dict["num_hidden_layers"] = len(layer_types)
        return cls.from_dict(config_dict, **kwargs)


class QTSplusQwen2_Model(QTSplusMetaModel, Qwen2Model):
    config_class = QTSplusQwen2_CausalLM_Config

    def __init__(self, config: Qwen2Config):
        super().__init__(config)


class QTSplusQwen2_ForCausalLM(QTSplusMetaForCausalLM, Qwen2ForCausalLM):
    config_class = QTSplusQwen2_CausalLM_Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        # Configure attention backend before modules are built.
        try:
            cfg_attn = getattr(config, "attn_implementation", None)
            if not torch.cuda.is_available():
                # Avoid Flash-Attn on CPU-only runs.
                if cfg_attn is None or str(cfg_attn) in {"auto", "flash_attention_2"}:
                    setattr(config, "attn_implementation", "eager")
                    setattr(config, "_attn_implementation", "eager")
                elif hasattr(config, "_attn_implementation"):
                    setattr(config, "_attn_implementation", str(cfg_attn))
            else:
                if (cfg_attn is None or str(cfg_attn) == "auto") and is_flash_attn_available():
                    setattr(config, "attn_implementation", "flash_attention_2")
                    setattr(config, "_attn_implementation", "flash_attention_2")
                elif hasattr(config, "_attn_implementation") and cfg_attn is not None:
                    setattr(config, "_attn_implementation", str(cfg_attn))
        except Exception:
            pass

        # Skip Qwen2ForCausalLM.__init__ to avoid constructing a second backbone.
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = QTSplusQwen2_Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        vision_input: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        **kwargs: Any,
    ):
        # HF Trainer (>=4.56) may pass this for loss normalization; Qwen2 forward doesn't accept it.
        kwargs.pop("num_items_in_batch", None)

        if inputs_embeds is not None:
            input_ids = None

        if inputs_embeds is None:
            (
                vision_input,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                flops_loss,
                kv_loss,
                smooth_loss,
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                question_input_ids,
                video_token_id=video_token_id,
                image_token_id=image_token_id,
                mode="train" if self.training else "infer",
            )
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_model().get_input_embeddings()(input_ids)

        input_ids = None

        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        add_loss = {
            "flops_loss": flops_loss if vision_input is not None else 0.0,
            "kv_loss": kv_loss if vision_input is not None else 0.0,
            "smooth_loss": smooth_loss if vision_input is not None else 0.0,
        }

        if labels is None and not self.training:
            return outputs

        return (outputs, add_loss)

    @torch.no_grad()
    def generate(
        self,
        vision_input: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        question_input_ids: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if vision_input is not None:
            (
                vision_input,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                *_unused_losses,
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                question_input_ids,
                video_token_id=video_token_id,
                image_token_id=image_token_id,
                mode="infer",
            )
        else:
            inputs_embeds = self.get_model().get_input_embeddings()(input_ids)

        kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        kwargs.pop("input_ids", None)

        if "use_cache" not in kwargs:
            kwargs["use_cache"] = True
        output_ids = super().generate(inputs_embeds=inputs_embeds, **kwargs)
        if input_ids is not None:
            input_ids = input_ids.to(output_ids.device)
            output_ids = torch.cat([input_ids, output_ids], dim=1)
        return output_ids


AutoConfig.register("qts_plus_qwen2_causal_lm", QTSplusQwen2_CausalLM_Config)
AutoModelForCausalLM.register(QTSplusQwen2_CausalLM_Config, QTSplusQwen2_ForCausalLM)
