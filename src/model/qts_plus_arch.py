# -*- encoding: utf-8 -*-
# @File        :   QTSplus_arch.py
# @Time        :   2025/04/15 16:28:48
# @Author      :   Siyou
# @Description :

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn

from .vision_encoder.builder import build_vision_tower
from .qts_plus_tokenizer.builder import build_qts_plus_tower

def qts_integrate_embeddings(
    vision_features: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    text_model_embed_layer: Optional[nn.Embedding] = None,
    kept_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Integrates vision features from Qwen2_5_VisionTransformerPretrainedModel with text embeddings
    for input to Qwen2_5_VLTextForCausalLM.

    This function replicates the embedding integration logic from Qwen2_5_VLForConditionalGeneration
    to enable separate processing of vision and text components.

    Args:
        vision_features (torch.Tensor): Output from Qwen2_5_VisionTransformerPretrainedModel
            Shape: [total_vision_patches, hidden_size]
        input_ids (torch.Tensor): Tokenized text input ids from Qwen2Tokenizer
            Shape: [batch_size, sequence_length]
        attention_mask (torch.Tensor): Attention mask for text tokens
            Shape: [batch_size, sequence_length]
        labels (torch.Tensor, optional): Training labels tensor
            Shape: [batch_size, sequence_length]
        image_token_id (int, optional): Token ID for image placeholder tokens
        video_token_id (int, optional): Token ID for video placeholder tokens
        image_grid_thw (torch.Tensor, optional): Image grid dimensions [num_images, 3]
        video_grid_thw (torch.Tensor, optional): Video grid dimensions [num_videos, 3]
        text_model_embed_layer (nn.Embedding, optional): Text embedding layer from the text model

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - final_embeddings: Integrated embeddings ready for VLTextForCausalLM
              Shape: [batch_size, final_sequence_length, hidden_size]
            - final_attention_mask: Updated attention mask for integrated sequence
              Shape: [batch_size, final_sequence_length]
            - final_labels: Updated labels tensor (if provided)
              Shape: [batch_size, final_sequence_length]
    """
    
    if text_model_embed_layer is None:
        raise ValueError("text_model_embed_tokens is required for text embedding integration")
    # Guard against accidental autocasting of token indices
    if input_ids.dtype is not torch.long:
        input_ids = input_ids.long()
    
    # Get text embeddings using the same method as the full model
    inputs_embeds = text_model_embed_layer(input_ids)
    
    # Replicate the masked_scatter approach from Qwen2_5_VLForConditionalGeneration
    if vision_features.shape[0] <= 0:
        raise ValueError("vision_features must contain at least one feature vector")
    # Check for video tokens and replace them with vision features
    if video_token_id is None:
        raise ValueError("video_token_id must be provided for video feature integration")

    # batch-size == 1 variant (current pipeline constructs single-sample batches)
    B, S = input_ids.shape
    assert B == 1, "Sequence-trimming currently assumes batch_size == 1."

    # Locate all video placeholder positions in the text sequence
    vid_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()

    # Fast-path for the new dataset behavior: a single <|video_pad|> token
    # Expand that single placeholder into the full sequence of selected
    # vision features and adjust attention_mask and labels accordingly.
    # This preserves all QTS+ selected tokens even if the template only
    # inserts one video token.
    n_feats = int(vision_features.shape[0])
    if vid_pos.numel() == 1 and n_feats >= 1:
        insert_idx = int(vid_pos.item())

        # Ensure dtype/device alignment
        vision_features = vision_features.to(inputs_embeds.device, inputs_embeds.dtype)

        # Slice original tensors around the single video token position
        pre_embeds  = inputs_embeds[:, :insert_idx, :]
        post_embeds = inputs_embeds[:, insert_idx + 1 :, :]

        # Build new inputs_embeds by splicing in all visual features
        feats_embeds = vision_features.unsqueeze(0)  # [1, T, D]
        inputs_embeds = torch.cat([pre_embeds, feats_embeds, post_embeds], dim=1)

        # Update attention_mask to account for inserted features
        feats_mask = torch.ones((1, n_feats), dtype=attention_mask.dtype, device=attention_mask.device)
        pre_mask   = attention_mask[:, :insert_idx]
        post_mask  = attention_mask[:, insert_idx + 1 :]
        attention_mask = torch.cat([pre_mask, feats_mask, post_mask], dim=1)

        # Update labels: ignore loss on inserted visual tokens
        if labels is not None:
            feat_labels = torch.full((1, n_feats), -100, dtype=labels.dtype, device=labels.device)
            pre_labels  = labels[:, :insert_idx]
            post_labels = labels[:, insert_idx + 1 :]
            labels = torch.cat([pre_labels, feat_labels, post_labels], dim=1)

        # Return early; remaining code handles multi-token templates
        final_attention_mask = attention_mask.to(inputs_embeds.device)
        return inputs_embeds, final_attention_mask, labels
    else:
        # General fallback: handle multi-video-token templates by mapping
        # features onto existing placeholders, trimming extras on either side.
        M = int(vid_pos.numel())
        if M == 0:
            raise ValueError("No video placeholder tokens found in input_ids for provided vision_features")

        # Align dtypes/devices
        vision_features = vision_features.to(inputs_embeds.device, inputs_embeds.dtype)

        N = n_feats  # number of visual features
        # If we have more features than placeholders, keep the first M features
        if N > M:
            raise NotImplementedError("Number of vision features exceeds video placeholders; please use single <|video_pad|> token templates.")

        # If we have fewer features than placeholders, drop the extra placeholders
        if N < M:
            drop_pos = vid_pos[N:]
            if drop_pos.numel() > 0:
                keep_seq = torch.ones(S, dtype=torch.bool, device=input_ids.device)
                keep_seq[drop_pos] = False
                input_ids = input_ids[:, keep_seq]
                attention_mask = attention_mask[:, keep_seq]
                inputs_embeds = inputs_embeds[:, keep_seq, :]
                if labels is not None:
                    labels = labels[:, keep_seq]
                # Recompute video positions after dropping
                vid_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
                M = int(vid_pos.numel())
                # By construction, M == N now

        # Replace each remaining video token with its corresponding visual feature
        for i in range(N):
            pos = int(vid_pos[i].item())
            inputs_embeds[0, pos, :] = vision_features[i, :]

        # Ignore loss on visual tokens
        if labels is not None and N > 0:
            labels = labels.clone()
            labels[0, vid_pos[:N]] = -100

        final_attention_mask = attention_mask.to(inputs_embeds.device)
        return inputs_embeds, final_attention_mask, labels

class QTSplusMetaModel:
    def __init__(self, config):
        super(QTSplusMetaModel, self).__init__(config)

        self.config = config

        if hasattr(config, "vision_tower"):
            # Defer QTS+ build until we know the vision out_hidden_size to set a correct embedding dim
            self.vision_tower = build_vision_tower(config)
            # QTS+ will be (re)built in initialize_vision_modules after we sync embed size
            # Best-effort populate vision_embed_size early so remote loading can
            # instantiate QTS+ before weights are loaded by from_pretrained.
            try:
                vt = getattr(self, "vision_tower", None)
                out_hidden = getattr(getattr(vt, "config", None), "out_hidden_size", None)
                if isinstance(out_hidden, int) and out_hidden > 0:
                    self.config.vision_embed_size = out_hidden
            except Exception:
                pass

        # Build QTS+ tower early if enabled so that its parameters exist during
        # from_pretrained weight loading. This prevents "unused weights" warnings
        # for keys like model.qts_plus.selector.* when loading checkpoints.
        if getattr(self.config, "enable_qts_plus", False) and getattr(self, "qts_plus", None) is None:
            try:
                self.qts_plus = build_qts_plus_tower(self.config)
            except Exception:
                # If dimensions are not yet fully known, leave construction to
                # initialize_vision_modules which will be called in training code.
                # In pure inference runs, vision_embed_size is typically known
                # from the config and this will succeed.
                pass

    def get_qts_plus_tower(self):
        qts_plus = getattr(self, 'qts_plus', None)
        return qts_plus
    
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        # vision config
        self.config.vision_tower = model_args.vision_tower
        # Pass through pretrained path so the builder can infer correct sizes
        self.config.pretrain_vision_model = getattr(model_args, 'pretrain_vision_model', None)

        # qts_plus config
        self.config.enable_qts_plus = model_args.enable_qts_plus
        self.config.embedding_dim = model_args.lm_embed_size
        # Set a tentative vision embed size, will sync with actual tower config below
        self.config.vision_embed_size = model_args.vision_embed_size
        self.config.project_text_if_needed = model_args.project_text_if_needed
        self.config.qts_plus_n_heads = model_args.qts_plus_n_heads
        self.config.qts_plus_tau_s = model_args.qts_plus_tau_s
        self.config.qts_plus_nmax = model_args.qts_plus_nmax
        self.config.qts_plus_rho_min = model_args.qts_plus_rho_min
        self.config.qts_plus_rho_max = model_args.qts_plus_rho_max
        self.config.qts_plus_block_dropout = model_args.qts_plus_block_dropout
        # optional re-encode switch
        self.config.qts_plus_reencode = model_args.qts_plus_reencode
        # QTS+ layer counts
        self.config.qts_plus_scoring_layers = model_args.qts_plus_scoring_layers
        self.config.qts_plus_reencode_layers = model_args.qts_plus_reencode_layers
        self.config.lambda_t = model_args.lambda_t
        self.config.lambda_m = model_args.lambda_m
        self.config.lambda_s = model_args.lambda_s

        # Use LM head count directly for QTS+; builder will validate divisibility
        lm_heads = getattr(self.config, 'num_attention_heads', None)
        if isinstance(lm_heads, int) and lm_heads > 0:
            self.config.qts_plus_n_heads = lm_heads
            
        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)

        # Sync the effective vision embedding size from the actual tower config (if available)
        vt = self.get_vision_tower()
        out_hidden = getattr(getattr(vt, 'config', None), 'out_hidden_size', None)
        if isinstance(out_hidden, int) and out_hidden > 0:
            self.config.vision_embed_size = out_hidden

        # qts_plus tower — build only after vision embed size and scoring_layers are finalized
        if self.get_qts_plus_tower() is None and model_args.enable_qts_plus:
            self.qts_plus = build_qts_plus_tower(self.config)

        if model_args.pretrain_vision_model is not None:
            if self.config.vision_tower == 'qwen2_5_vl_vision':
                # Load from directory or file robustly
                from safetensors.torch import load_file
                v_path = model_args.pretrain_vision_model
                weights = None
                if os.path.isdir(v_path):
                    # Prefer a single-file safetensors if present
                    candidate = os.path.join(v_path, 'model.safetensors')
                    if os.path.isfile(candidate):
                        weights = load_file(candidate)
                    else:
                        # Fallback to PyTorch bin
                        candidate = os.path.join(v_path, 'pytorch_model.bin')
                        if os.path.isfile(candidate):
                            weights = torch.load(candidate, map_location='cpu')
                else:
                    if v_path.endswith('.safetensors'):
                        weights = load_file(v_path)
                    elif v_path.endswith('.bin'):
                        weights = torch.load(v_path, map_location='cpu')

                if weights is None:
                    raise FileNotFoundError(f"No vision weights found under {v_path}. Expected model.safetensors or pytorch_model.bin.")

                self.vision_tower.load_state_dict(weights, strict=False)
            else:
                raise ValueError("Not support this model")
                
        # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
        self.vision_tower.requires_grad_(not model_args.freeze_vision_model)

        # Always initialize from the loaded LM by default
        n_layers = int(getattr(self.config, 'num_hidden_layers', 0))
        default_sc_idx = max(0, n_layers - 1)
        sc_count = max(1, int(self.config.qts_plus_scoring_layers))
        sc_init_indices = [default_sc_idx for _ in range(sc_count)]
        qts = self.get_qts_plus_tower()
        qts.selector.init_qwen_scoring_from_lm_model(self, sc_init_indices)

        if getattr(self.config, 'qts_plus_reencode', True):
            re_count = max(1, int(self.config.qts_plus_reencode_layers))
            re_init_indices = [min(i, max(0, n_layers - 1)) for i in range(re_count)]
            qts.selector.init_qwen_reencode_from_lm_model(self, re_init_indices)


class QTSplusMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_qts_plus_tower(self):
        return self.get_model().get_qts_plus_tower()

    def encode_visions(self, vision):
        vision_features = self.get_model().get_vision_tower()(vision)
        return vision_features

    def prepare_inputs_for_multimodal(
        self,
        vision_input,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        question_input_ids: Optional[torch.Tensor] = None,
        video_token_id: Optional[int] = None,
        mode: str = 'train',
    ):
        vision_tower = self.get_vision_tower()
        qts_plus_tower = self.get_qts_plus_tower()
        text_embed_layer = self.get_model().get_input_embeddings()
        if vision_tower is None or vision_input is None or input_ids.shape[1] == 1:
            # Return default values with zero losses for text-only mode
            flops_loss = torch.tensor(0.0, device=input_ids.device)
            kv_loss = torch.tensor(0.0, device=input_ids.device)
            smooth_loss = torch.tensor(0.0, device=input_ids.device)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, flops_loss, kv_loss, smooth_loss
        else:
            if self.config.enable_qts_plus:
                if self.config.vision_tower == 'qwen2_5_vl_vision':
                    # Handle collated list case from DataCollator
                    if isinstance(vision_input, list):
                        if len(vision_input) == 0:
                            # Return default values with zero losses for empty vision list
                            flops_loss = torch.tensor(0.0, device=input_ids.device)
                            kv_loss = torch.tensor(0.0, device=input_ids.device)
                            smooth_loss = torch.tensor(0.0, device=input_ids.device)
                            return input_ids, position_ids, attention_mask, past_key_values, None, labels, flops_loss, kv_loss, smooth_loss
                        vision_input = vision_input[0]

                    vision_features = vision_tower.get_video_features(
                        vision_input["pixel_values_videos"].to(vision_tower.device),
                        vision_input["video_grid_thw"].to(vision_tower.device)
                    )
                    video_grid_thw = vision_input["video_grid_thw"]
                    if vision_features.ndim == 2:
                        vision_features = vision_features.unsqueeze(0)

                    # Use question_input_ids from dataset to prevent data leakage
                    # If not provided, fall back to input_ids (for inference or backward compatibility)
                    if question_input_ids is None:
                        assert False, "question_input_ids must be provided in training to avoid data leakage"

                    # Get text embeddings only for the question part
                    if question_input_ids is not None and question_input_ids.dtype is not torch.long:
                        question_input_ids = question_input_ids.long()
                    text_embeddings = text_embed_layer(question_input_ids)
                    vision_features = vision_features.to(dtype=text_embeddings.dtype)

                    qts_plus_out = qts_plus_tower(vision_features, text_embeddings, mode=mode)
                    vision_features = qts_plus_out["Z"]
                    # Track selection size for benchmarking (batch == 1)
                    try:
                        if isinstance(vision_features, list) and len(vision_features) > 0:
                            _z_count = int(vision_features[0].shape[0])
                        elif isinstance(vision_features, torch.Tensor):
                            _z_count = int(vision_features.shape[1]) if vision_features.ndim == 3 else int(vision_features.shape[0])
                        else:
                            _z_count = 0
                    except Exception:
                        _z_count = 0
                    flops_loss = qts_plus_out["add_loss"]["flops"]
                    kv_loss = qts_plus_out["add_loss"]["kv"]
                    smooth_loss = qts_plus_out["add_loss"]["smooth"]
                    
                    # Resolve video token id if not provided
                    if video_token_id is None:
                        video_token_id = getattr(self.config, "video_token_id", None)
                        if video_token_id is None:
                            # Default to Qwen2.5-VL video token id
                            video_token_id = 151656
                    
                    inputs_embeds, attention_mask, labels = qts_integrate_embeddings(
                        vision_features=vision_features[0],
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        video_token_id=video_token_id,
                        text_model_embed_layer=text_embed_layer,
                        video_grid_thw=video_grid_thw,
                    )
                    # Persist debug metrics for downstream benchmarking
                    try:
                        _placeholders = int((input_ids == video_token_id).sum().item()) if video_token_id is not None else None
                    except Exception:
                        _placeholders = None
                    try:
                        _prefill_len = int(inputs_embeds.shape[1]) if hasattr(inputs_embeds, 'shape') else None
                    except Exception:
                        _prefill_len = None
                    try:
                        _dtype = text_embeddings.dtype
                        _hidden = int(text_embeddings.shape[-1]) if hasattr(text_embeddings, 'shape') else int(getattr(self.config, 'hidden_size', 0))
                    except Exception:
                        _dtype = None
                        _hidden = int(getattr(self.config, 'hidden_size', 0))
                    try:
                        self._qtsplus_benchmark = {
                            'z_count': _z_count,
                            'placeholders': _placeholders,
                            'prefill_len': _prefill_len,
                            'dtype': str(_dtype) if _dtype is not None else None,
                            'hidden_size': _hidden,
                            'num_layers': int(getattr(self.config, 'num_hidden_layers', 0)),
                        }
                    except Exception:
                        pass
            else:
                raise ValueError("Not support this model")
        return vision_input, position_ids, attention_mask, past_key_values, inputs_embeds, labels, flops_loss, kv_loss, smooth_loss

    def vision_features_count_qtsplus(
        self,
        pixel_values_videos: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        question_input_ids: Optional[torch.Tensor],
    ) -> int:
        """
        Count the number of video embeddings selected by QTS+ (tokens fed into the LM)
        for a single-sample batch.

        This mirrors the inference path: vision -> QTS+ selector -> Z, and returns
        len(Z[0]) when Z is a list of [T_b, D] or the appropriate dimension when
        it is a tensor.
        """
        try:
            if pixel_values_videos is None or video_grid_thw is None or question_input_ids is None:
                return 0

            vision_tower = self.get_vision_tower()
            qts_tower = self.get_qts_plus_tower()
            text_embed = self.get_model().get_input_embeddings()

            if vision_tower is None or qts_tower is None or text_embed is None:
                return 0

            # Ensure proper dtypes/devices
            if question_input_ids.dtype is not torch.long:
                question_input_ids = question_input_ids.long()

            # Compute vision features using the vision encoder
            try:
                vt_device = next(vision_tower.parameters()).device
            except StopIteration:
                vt_device = text_embed.weight.device
            vf = vision_tower.get_video_features(
                pixel_values_videos.to(vt_device),
                video_grid_thw.to(vt_device),
            )
            if isinstance(vf, list):
                # Pack to tensor if needed (single video assumed in this benchmark)
                if len(vf) > 0:
                    vf = vf[0]
                if vf.ndim == 2:
                    vf = vf.unsqueeze(0)
            if isinstance(vf, torch.Tensor) and vf.ndim == 2:
                vf = vf.unsqueeze(0)

            # Map to text embedding device/dtype
            te = text_embed(question_input_ids.to(text_embed.weight.device))
            if isinstance(vf, torch.Tensor):
                vf = vf.to(device=te.device, dtype=te.dtype)
            try:
                qts_tower.to(device=te.device, dtype=te.dtype)
            except Exception:
                qts_tower.to(device=te.device)

            # Run QTS+ selector in inference mode
            with torch.inference_mode():
                qpo = qts_tower(vf, te, mode='infer')
            Z = qpo.get("Z")

            if isinstance(Z, list) and len(Z) > 0:
                return int(Z[0].shape[0])
            if isinstance(Z, torch.Tensor):
                # Accept both [B, T, D] and [T, D]
                if Z.ndim == 3:
                    return int(Z.shape[1])
                return int(Z.shape[0])
            return 0
        except Exception:
            return 0

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
