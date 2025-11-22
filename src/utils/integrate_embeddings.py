# -*- encoding: utf-8 -*-
# @File        :   integrate_embeddings.py
# @Time        :   2025/07/13
# @Author      :   Siyou
# @Description :   Utility to integrate vision and text embeddings for Qwen2.5-VL model pipeline

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


def integrate_embeddings(
    vision_features: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    text_model_embed_layer: Optional[nn.Embedding] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        tokenizer (PreTrainedTokenizer): The Qwen2 tokenizer instance
        image_token_id (int, optional): Token ID for image placeholder tokens
        video_token_id (int, optional): Token ID for video placeholder tokens
        image_grid_thw (torch.Tensor, optional): Image grid dimensions [num_images, 3]
        video_grid_thw (torch.Tensor, optional): Video grid dimensions [num_videos, 3]
        text_model_embed_tokens (nn.Embedding, optional): Text embedding layer from the text model
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - final_embeddings: Integrated embeddings ready for VLTextForCausalLM
              Shape: [batch_size, final_sequence_length, hidden_size]
            - final_attention_mask: Updated attention mask for integrated sequence
              Shape: [batch_size, final_sequence_length]
    """
    device = vision_features.device
    dtype = vision_features.dtype
    
    if text_model_embed_layer is None:
        raise ValueError("text_model_embed_tokens is required for text embedding integration")
    
    # Get proper token IDs from tokenizer config
    if hasattr(tokenizer, 'image_token_id'):
        image_token_id = tokenizer.image_token_id
    elif image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
    if hasattr(tokenizer, 'video_token_id'):
        video_token_id = tokenizer.video_token_id
    elif video_token_id is None:
        video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    
    batch_size, seq_len = input_ids.shape
    
    # Get text embeddings using the same method as the full model
    inputs_embeds = text_model_embed_layer(input_ids)
    # Replicate the masked_scatter approach from Qwen2_5_VLForConditionalGeneration
    if video_grid_thw is not None and vision_features.shape[0] > 0:
        # Check for video tokens and replace them with vision features
        n_video_tokens = (input_ids == video_token_id).sum().item()
        n_video_features = vision_features.shape[0]
        
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        
        # Create mask for video tokens
        mask = input_ids == video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)
        
        # Apply vision features to video token positions using masked_scatter
        vision_features = vision_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, vision_features)
    
    # Handle image tokens similarly if present
    if image_grid_thw is not None:
        n_image_tokens = (input_ids == image_token_id).sum().item()
        if n_image_tokens > 0:
            # For images, we would need separate image features
            # This is a placeholder for when image features are also provided
            pass
    
    # Update attention mask to match the integrated embeddings
    final_attention_mask = attention_mask.to(inputs_embeds.device)
    
    return inputs_embeds, final_attention_mask


def prepare_multimodal_inputs(
    vision_model_outputs: torch.Tensor,
    tokenizer_outputs: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    text_embed_layer: nn.Embedding,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    High-level wrapper function to prepare multimodal inputs for VLTextForCausalLM.
    
    Args:
        vision_model_outputs (torch.Tensor): Vision features from Qwen2_5_VisionTransformerPretrainedModel
        tokenizer_outputs (Dict[str, torch.Tensor]): Outputs from Qwen2Tokenizer containing:
            - input_ids: Token IDs
            - attention_mask: Attention mask
        tokenizer (PreTrainedTokenizer): Qwen2 tokenizer instance
        text_embed_layer (nn.Embedding): Text embedding layer from VLTextForCausalLM
        image_grid_thw (torch.Tensor, optional): Image grid dimensions
        video_grid_thw (torch.Tensor, optional): Video grid dimensions
        
    Returns:
        Dict[str, torch.Tensor]: Processed inputs ready for VLTextForCausalLM containing:
            - inputs_embeds: Integrated embeddings
            - attention_mask: Updated attention mask
    """
    input_ids = tokenizer_outputs["input_ids"]
    attention_mask = tokenizer_outputs["attention_mask"]
    
    inputs_embeds, final_attention_mask = integrate_embeddings(
        vision_features=vision_model_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        text_model_embed_layer=text_embed_layer,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": final_attention_mask,
    }