#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File        :   qwen2_5_vl_vision_causal_lm_concat_pipleline.py
# @Time        :   2025/07/13
# @Author      :   Siyou
# @Description :   Test script demonstrating the integration of Qwen2.5-VL vision and text components
#                  using the separate pipeline approach to match Qwen2_5_VLForConditionalGeneration results

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from PIL import Image
import requests
from io import BytesIO

from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from src.model.vision_encoder import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionProcessor,
)
from src.model.language_model import Qwen2_5_VLTextForCausalLM
from src.utils.integrate_embeddings import prepare_multimodal_inputs, integrate_embeddings
from src.utils.qwen_vision_process import process_vision_info

FULL_MODEL = "pretrained_models/Qwen2.5-VL-3B-Instruct"
VISION_MODEL = "pretrained_models/Qwen2.5-VL-3B-Instruct-Vision"
LM_MODEL = "pretrained_models/Qwen2.5-VL-3B-Instruct-LM"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_models(
        full_model_path: str = FULL_MODEL,
        vision_model_path: str = VISION_MODEL,
        text_model_path: str = LM_MODEL
    ):
    """Load all required models and tokenizer"""
    print("Loading models...")
    
    # Load the full model for comparison
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        full_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Load vision encoder
    vision_model = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(
        vision_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Load text model 
    text_model = Qwen2_5_VLTextForCausalLM.from_pretrained(
        text_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    processor = AutoProcessor.from_pretrained(full_model_path)
    vision_processor = Qwen2_5_VLVisionProcessor.from_pretrained(vision_model_path)
    
    # Set models to eval mode
    full_model.eval()
    vision_model.eval()
    text_model.eval()
    
    print("Models loaded successfully!")
    return full_model, vision_model, text_model, tokenizer, processor, vision_processor

def prepare_messages(text_prompt: str, video_path: str) -> List[Dict]:
    """Prepare message format for chat template"""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,
                    "max_pixels": 360 * 420,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]


def test_full_model_inference(model, processor, messages, video_path):
    """Test inference using the full Qwen2_5_VLForConditionalGeneration model"""
    print("\\n=== Testing Full Model (Qwen2_5_VLForConditionalGeneration) ===")
    
    # Process vision info to load the video
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    # Prepare inputs with loaded video tensors
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs
    ).to(model.device)
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Available keys: {list(inputs.keys())}")
    if 'pixel_values' in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        print(f"Image grid shape: {inputs['image_grid_thw'].shape}")
    if 'video_grid_thw' in inputs:
        print(f"Video grid shape: {inputs['video_grid_thw'].shape}")
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Full model response: {response}")
    return response


def test_separate_pipeline(vision_model, text_model, messages, video_path, processor):
    """Test inference using separate vision and text models with integration utility"""
    print("\\n=== Testing Separate Pipeline (Vision + Text Integration) ===")
    
    # Step 1: Load video using qwen vision utilities
    video_info = {
        "video": video_path,
        "fps": 1.0,
        "max_pixels": 360 * 420,
    }
    
    # Step 2: Process the loaded video tensor
    vision_inputs = processor(
        [{"type": "video", "video": video_path, "fps": 1.0, "max_pixels": 360 * 420}]
    )
    video_tensor = vision_inputs["pixel_values_videos"]
    video_grid_thw = vision_inputs["video_grid_thw"]
    
    print(f"Vision input shape: {video_tensor.shape}")
    
    # Step 2: Extract vision features
    with torch.no_grad():
        vision_features = vision_model.get_video_features(
            vision_inputs["pixel_values_videos"].to(vision_model.device),
            vision_inputs["video_grid_thw"].to(vision_model.device)
        )
    
    print(f"Vision features shape: {vision_features.shape}")
    
    # Step 3: Process text input with proper video token expansion
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Manually implement video token expansion logic
    # Calculate the number of video tokens needed
    merge_length = processor.video_processor.merge_size ** 2  # spatial_merge_size^2 = 4
    num_video_tokens = video_grid_thw[0].prod().item() // merge_length
    
    print(f"Video grid: {video_grid_thw[0]}, merge_length: {merge_length}, num_video_tokens: {num_video_tokens}")
    
    # Replace single video token with multiple video tokens
    video_token = processor.video_token  # "<|video_pad|>"
    expanded_video_tokens = video_token * num_video_tokens
    text_with_expanded_tokens = text_prompt.replace(video_token, expanded_video_tokens, 1)
    
    # Tokenize the expanded text
    text_inputs = processor.tokenizer(
        text=[text_with_expanded_tokens],
        padding=True,
        return_tensors="pt"
    ).to(text_model.device)
    
    print(f"Text input IDs shape (with expanded video tokens): {text_inputs['input_ids'].shape}")
    
    # Step 4: Integrate vision and text embeddings
    integrated_inputs = prepare_multimodal_inputs(
        vision_model_outputs=vision_features,
        tokenizer_outputs=text_inputs,
        tokenizer=processor.tokenizer,
        text_embed_layer=text_model.get_input_embeddings(),
        video_grid_thw=video_grid_thw
    )
    
    print(f"Integrated embeddings shape: {integrated_inputs['inputs_embeds'].shape}")
    print(f"Final attention mask shape: {integrated_inputs['attention_mask'].shape}")
    
    # Step 5: Generate response using text model
    with torch.no_grad():
        print(f"Generating with input embeds shape: {integrated_inputs['inputs_embeds'].shape}")
        print(f"Attention mask shape: {integrated_inputs['attention_mask'].shape}")
        print(f"First few attention mask values: {integrated_inputs['attention_mask'][0, :10]}")
        
        generated_ids = text_model.generate(
            inputs_embeds=integrated_inputs["inputs_embeds"],
            attention_mask=integrated_inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    print(f"Generated IDs shape: {generated_ids.shape}")
    
    # When using inputs_embeds, the generated_ids only contains the new tokens
    # No need to skip input length as with input_ids
    print(f"Generated tokens: {generated_ids.shape}")
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Separate pipeline response: {response}")
    return response


def compare_outputs(full_response: str, separate_response: str):
    """Compare outputs from both approaches"""
    print("\\n=== Comparison Results ===")
    print(f"Full model response: '{full_response}'")
    print(f"Separate pipeline response: '{separate_response}'")
    
    # Simple similarity check
    if full_response.strip() == separate_response.strip():
        print("✅ Responses are identical!")
    else:
        print("⚠️  Responses differ")
        
        # Calculate approximate similarity
        full_tokens = set(full_response.lower().split())
        separate_tokens = set(separate_response.lower().split())
        
        if full_tokens and separate_tokens:
            intersection = full_tokens.intersection(separate_tokens)
            union = full_tokens.union(separate_tokens)
            similarity = len(intersection) / len(union) if union else 0
            print(f"Token overlap similarity: {similarity:.2%}")


def main():
    import glob
    """Main test function"""
    print("🚀 Starting Qwen2.5-VL Vision-Text Integration Pipeline Test")
    
    # Configuration
    text_prompt = "Describe this video."
    #video_path = "examples/example2.mp4"
    VIDEO_PATH= "examples/example_images"
    video_path = sorted(glob.glob(f"{VIDEO_PATH}/*.jpeg"))
    try:
        # Load models
        full_model, vision_model, text_model, tokenizer, processor, vision_processor = load_models()
        
        # Prepare messages
        messages = prepare_messages(text_prompt, video_path)
        
        # Test separate pipeline
        separate_response = test_separate_pipeline(
            vision_model, text_model, messages, video_path, vision_processor
        )

        # Test full model
        full_response = test_full_model_inference(
            full_model, processor, messages, video_path
        )
        
        
        # Compare results
        compare_outputs(full_response, separate_response)
        
        print("\\n✅ Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()