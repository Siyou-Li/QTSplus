# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


try:  # Optional dependency
    from timm.models.layers import DropPath as _DropPath  # type: ignore

    DropPath = _DropPath
except Exception:  # pragma: no cover
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = random_tensor.floor()
            return x.div(keep_prob) * random_tensor


try:
    from flash_attn.bert_padding import pad_input, unpad_input  # type: ignore
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func  # type: ignore

    has_flash_attn = True
except Exception:  # pragma: no cover
    pad_input, unpad_input, flash_attn_varlen_qkvpacked_func = None, None, None
    has_flash_attn = False


class FlashAttention(nn.Module):
    """Scaled dot-product attention implemented with FlashAttention2."""

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv,
        key_padding_mask=None,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False,
    ):
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seqlen,
                    step=seqlen,
                    dtype=torch.int32,
                    device=qkv.device,
                )
                output = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm  # type: ignore

    InternRMSNorm = FusedRMSNorm  # noqa
    logger.info("Discovered apex.normalization.FusedRMSNorm - using it instead of InternRMSNorm")
except Exception:  # pragma: no cover
    pass


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        H = patch_embeds.shape[-2]
        W = patch_embeds.shape[-1]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # shape = [*, grid ** 2, width]

        class_embeds = self.class_embedding.expand(batch_size, -1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        pos_embeds = self.position_embedding
        if H != self.image_size // self.patch_size or W != self.image_size // self.patch_size:
            pos_embeds = torch.cat(
                [pos_embeds[:, :1, :], self._get_pos_embed(pos_embeds[:, 1:, :], H, W)],
                dim=1,
            )

        embeddings = embeddings + pos_embeds
        return embeddings


class InternSelfAttention(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = config.qkv_bias

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=self.qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.head_dim)
            self.k_norm = InternRMSNorm(self.head_dim)

        if config.use_flash_attn and has_flash_attn:
            self.inner_attn = FlashAttention(softmax_scale=None, attention_dropout=config.attention_dropout)
        else:
            self.inner_attn = None

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.qk_normalization:
            q, k, v = qkv.unbind(dim=2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            qkv = torch.stack([q, k, v], dim=2)

        if self.inner_attn is not None and x.is_cuda:
            attn_output, _ = self.inner_attn(qkv=qkv, key_padding_mask=attn_mask, need_weights=False)
            attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_output = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.norm1 = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.attn = InternSelfAttention(config)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = InternMLP(config)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        hidden_states = hidden_states + self.drop_path(self.attn(self.norm1(hidden_states), attn_mask=attn_mask))
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class InternVisionEncoder(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config, drop_path_rate=dpr[i]) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attn_mask)
            else:
                hidden_states = layer(hidden_states, attn_mask=attn_mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class InternVisionModel(PreTrainedModel):
    config_class = InternVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)
        self.post_layernorm = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embeddings = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=embeddings,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.pooler(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=None,
        )


__all__ = [
    "InternVisionConfig",
    "InternVisionModel",
    "has_flash_attn",
]

