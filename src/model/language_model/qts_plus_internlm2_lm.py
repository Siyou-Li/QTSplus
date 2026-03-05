# -*- encoding: utf-8 -*-
# QTSplus wrapper for InternLM2 Causal LM (used by InternVL2.5-8B)

from __future__ import annotations

import os
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available

from ..internvl.internlm2.configuration_internlm2 import InternLM2Config
from ..internvl.internlm2.modeling_internlm2 import InternLM2ForCausalLM, InternLM2Model
from ..qts_plus_arch import QTSplusMetaForCausalLM, QTSplusMetaModel

def _hf_generate_fallback(model: "QTSplusInternLM2_ForCausalLM", **kwargs):
    """Call into HF generation even when `super().generate` is unavailable."""
    try:
        return super(QTSplusInternLM2_ForCausalLM, model).generate(**kwargs)
    except AttributeError as e:
        msg = str(e)
        if "generate" not in msg:
            raise
        try:
            from transformers.generation.utils import GenerationMixin  # type: ignore
        except Exception:  # pragma: no cover
            from transformers.generation_utils import GenerationMixin  # type: ignore
        return GenerationMixin.generate(model, **kwargs)


class QTSplusInternLM2_CausalLM_Config(InternLM2Config):
    model_type = "qts_plus_internlm2_causal_lm"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if str(config_dict.get("model_type") or "").lower() == "internlm2":
            config_dict["model_type"] = cls.model_type
        return cls.from_dict(config_dict, **kwargs)


class QTSplusInternLM2_Model(QTSplusMetaModel, InternLM2Model):
    config_class = QTSplusInternLM2_CausalLM_Config

    def __init__(self, config: InternLM2Config):
        super().__init__(config)


class QTSplusInternLM2_ForCausalLM(QTSplusMetaForCausalLM, InternLM2ForCausalLM):
    config_class = QTSplusInternLM2_CausalLM_Config
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config: InternLM2Config):
        # Configure attention backend before modules are built.
        try:
            cfg_attn = getattr(config, "attn_implementation", None)
            if (cfg_attn is None or str(cfg_attn) == "auto") and is_flash_attn_available():
                setattr(config, "attn_implementation", "flash_attention_2")
        except Exception:
            pass

        # Skip InternLM2ForCausalLM.__init__ to avoid constructing a second backbone.
        super(InternLM2ForCausalLM, self).__init__(config)
        self.model = QTSplusInternLM2_Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        # InternLM2ForCausalLM doesn't accept cache_position; ignore for compatibility with newer HF.
        _ = cache_position
        # HF Trainer (>=4.56) may pass this for loss normalization; InternLM2 forward doesn't accept it.
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
        # `generate()` should run in eval mode to avoid returning the training-only
        # tuple `(outputs, add_loss)` from `forward()` when `self.training == True`.
        was_training = self.training
        if was_training:
            self.eval()
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
        try:
            output_ids = _hf_generate_fallback(self, inputs_embeds=inputs_embeds, **kwargs)
        finally:
            if was_training:
                self.train()
        if input_ids is not None:
            input_ids = input_ids.to(output_ids.device)
            output_ids = torch.cat([input_ids, output_ids], dim=1)
        return output_ids


AutoConfig.register("qts_plus_internlm2_causal_lm", QTSplusInternLM2_CausalLM_Config)
AutoModelForCausalLM.register(QTSplusInternLM2_CausalLM_Config, QTSplusInternLM2_ForCausalLM)
