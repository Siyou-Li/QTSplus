# -*- encoding: utf-8 -*-
# QTSplusTokenizer wrapper for Qwen2.5-VL Text Causal LM

from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLTextModel,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from ..qts_plus_arch import QTSplusMetaModel, QTSplusMetaForCausalLM
from .qwen2_5_vl_text_for_causallm import Qwen2_5_VLTextForCausalLM

class QTSplusQwen2_5_VL_CausalLM_Config(Qwen2_5_VLTextConfig):
    model_type = "qts_plus_qwen2_5_vl_causal_lm"


class QTSplusQwen2_5_VLModel(QTSplusMetaModel, Qwen2_5_VLTextModel):
    config_class = QTSplusQwen2_5_VL_CausalLM_Config

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super(QTSplusQwen2_5_VLModel, self).__init__(config)


class QTSplusQwen2_5_VLTextForCausalLM(QTSplusMetaForCausalLM, Qwen2_5_VLTextForCausalLM):
    config_class = QTSplusQwen2_5_VL_CausalLM_Config

    def __init__(self, config):
        # Ensure memory-efficient attention is configured BEFORE submodules are built
        # so decoder layers pick the correct attention class.
        try:
            cfg_attn = getattr(config, "attn_implementation", None)
            if (cfg_attn is None or str(cfg_attn) == "auto") and is_flash_attn_available():
                setattr(config, "attn_implementation", "flash_attention_2")
                setattr(config, "_attn_implementation", "flash_attention_2")
        except Exception:
            # Keep defaults if feature probing fails
            pass

        super(Qwen2_5_VLTextForCausalLM, self).__init__(config)
        self.model = QTSplusQwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        vision_input: Optional[torch.FloatTensor] = None,
        input_ids: torch.LongTensor = None,
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
        video_token_id: Optional[int] = None,
    ):

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
                smooth_loss
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                question_input_ids,
                video_token_id,
                # Ensure QTS+ runs in the correct mode
                mode="train" if self.training else "infer",
            )
            if inputs_embeds is None:
                inputs_embeds = self.get_model().embed_tokens(input_ids)

        input_ids = None

        try:
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
            )
        except ValueError as error:
            raise ValueError(
                f"{error} (input_ids is None: {input_ids is None}, inputs_embeds is None: {inputs_embeds is None})"
            ) from error
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
        video_token_id: Optional[int] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # Ensure an attention mask exists when integrating visual features
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
                video_token_id,
                # Generation must use inference (hard Top-n) behavior
                mode="infer",
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        kwargs.pop("input_ids", None)

        # Ensure caching is enabled to bound memory during decoding
        if "use_cache" not in kwargs:
            kwargs["use_cache"] = True
        output_ids = super().generate(inputs_embeds=inputs_embeds, **kwargs)
        # When using inputs_embeds, HF's generate does not include the original
        # prompt tokens in the returned sequences. Many downstream utilities
        # (including our demo) trim the output using the original input length.
        # To keep compatibility, prepend the original input_ids so trimming works.
        if input_ids is not None:
            input_ids = input_ids.to(output_ids.device)
            output_ids = torch.cat([input_ids, output_ids], dim=1)
        return output_ids


AutoConfig.register("qts_plus_qwen2_5_vl_causal_lm", QTSplusQwen2_5_VL_CausalLM_Config)
AutoModelForCausalLM.register(QTSplusQwen2_5_VL_CausalLM_Config, QTSplusQwen2_5_VLTextForCausalLM)
