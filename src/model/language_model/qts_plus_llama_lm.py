# -*- encoding: utf-8 -*-
# QTSplus wrapper for Llama 3.1 Causal LM

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available

try:
    # Preferred import path in recent HF versions
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
except Exception:  # pragma: no cover
    # Fallback for older versions
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from ..qts_plus_arch import QTSplusMetaForCausalLM, QTSplusMetaModel


class QTSplusLlama3_CausalLM_Config(LlamaConfig):
    model_type = "qts_plus_llama_3_causal_lm"


class QTSplusLlama3_Model(QTSplusMetaModel, LlamaModel):
    config_class = QTSplusLlama3_CausalLM_Config

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class QTSplusLlama3_ForCausalLM(QTSplusMetaForCausalLM, LlamaForCausalLM):
    config_class = QTSplusLlama3_CausalLM_Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        # Ensure memory-efficient attention is configured BEFORE submodules are built
        # so decoder layers pick the correct attention class.
        try:
            cfg_attn = getattr(config, "attn_implementation", None)
            if (cfg_attn is None or str(cfg_attn) == "auto") and is_flash_attn_available():
                setattr(config, "attn_implementation", "flash_attention_2")
                setattr(config, "_attn_implementation", "flash_attention_2")
        except Exception:
            pass

        # Skip LlamaForCausalLM.__init__ to avoid constructing a second backbone.
        super(LlamaForCausalLM, self).__init__(config)
        self.model = QTSplusLlama3_Model(config)
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
        video_token_id: Optional[int] = None,
        **kwargs: Any,
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
                smooth_loss,
            ) = self.prepare_inputs_for_multimodal(
                vision_input,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                question_input_ids,
                video_token_id,
                mode="train" if self.training else "infer",
            )
            if inputs_embeds is None and input_ids is not None:
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
                **kwargs,
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
                mode="infer",
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

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


AutoConfig.register("qts_plus_llama_3_causal_lm", QTSplusLlama3_CausalLM_Config)
AutoModelForCausalLM.register(QTSplusLlama3_CausalLM_Config, QTSplusLlama3_ForCausalLM)

