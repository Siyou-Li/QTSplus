"""
Gradio app for running QTSplus on Hugging Face Spaces.

This follows the inference example in README.md and uses the
`AlpachinoNLP/QTSplus-3B` Hugging Face model.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, List, Tuple
os.system('pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu -U')
os.system('pip install transformers==4.57.1 av qwen-vl-utils sentencepiece bitsandbytes -U')
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional, List, Tuple

# Ensure project root (which contains `src/`) is on PYTHONPATH so we can
# reuse the local vision processing utilities instead of relying on
# external `qwen_vl_utils`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_vl_utils import process_vision_info


DEFAULT_MODEL_ID = os.environ.get("QTSPLUS_MODEL_ID", "AlpachinoNLP/QTSplus-3B")
DEFAULT_QUESTION = "What is happening in the video?"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float16

_MODEL: Optional[AutoModelForCausalLM] = None
_PROCESSOR: Optional[AutoProcessor] = None


def load_model_and_processor() -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Lazy-load the QTSplus model and processor."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    model_id = DEFAULT_MODEL_ID

    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    if DEVICE.type == "cuda":
        # Let Transformers place layers automatically on the available GPU.
        model_kwargs["device_map"] = "auto"

    _MODEL = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    _MODEL.eval()
    if DEVICE.type != "cuda":
        _MODEL.to(DEVICE)

    _PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Ensure generation uses explicit pad/eos ids.
    tok = getattr(_PROCESSOR, "tokenizer", None)
    if tok is not None:
        if getattr(_MODEL.config, "pad_token_id", None) is None:
            _MODEL.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
        if getattr(_MODEL.config, "eos_token_id", None) is None:
            _MODEL.config.eos_token_id = tok.eos_token_id
        if hasattr(_MODEL, "generation_config") and _MODEL.generation_config is not None:
            _MODEL.generation_config.pad_token_id = _MODEL.config.pad_token_id
            _MODEL.generation_config.eos_token_id = _MODEL.config.eos_token_id

    return _MODEL, _PROCESSOR


# Preload model and processor at import time (for faster first inference).
load_model_and_processor()


def build_messages(video: Optional[str], prompt: str) -> List[dict]:
    """Build chat-style messages for a single video + question."""
    if not video:
        raise ValueError("Please upload a video before running the model.")

    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": prompt or DEFAULT_QUESTION},
            ],
        }
    ]


def qtsplus_generate(video_path: Optional[str], question: str, max_new_tokens: int = 256) -> str:
    """Run QTSplus on the given video and question."""
    if not video_path:
        return "Please upload a video first."

    model, processor = load_model_and_processor()

    messages = build_messages(video_path, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **(video_kwargs or {}),
    )

    # Move inputs to the correct device and dtype.
    if DEVICE.type == "cuda":
        inputs = inputs.to(dtype=DTYPE, device=DEVICE)
    else:
        inputs = inputs.to(device=DEVICE)

    # Extract vision tensors for QTSplus-specific `vision_input` argument.
    pixel_values_videos = inputs.pop("pixel_values_videos", None)
    video_grid_thw = inputs.pop("video_grid_thw", None)
    if "second_per_grid_ts" in inputs:
        inputs.pop("second_per_grid_ts")

    vision_input = None
    if pixel_values_videos is not None and video_grid_thw is not None:
        vision_input = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    # Build question_input_ids from the raw textual question.
    tok = getattr(processor, "tokenizer", None)
    question_ids = None
    if tok is not None and question:
        question_ids = tok(
            question,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            vision_input=vision_input,
            input_ids=inputs.input_ids,
            question_input_ids=question_ids if question_ids is not None else inputs.input_ids,
            max_new_tokens=int(max_new_tokens),
        )

    # Remove the prompt tokens from the generated sequence.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Fallback: if trimming yields empty text, decode full sequences instead.
    if not output_text or not output_text[0].strip():
        output_text = [
            processor.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

    return output_text[0] if output_text else ""


with gr.Blocks() as demo:
    gr.Markdown("# QTSplus-3B Video QA Demo")

    with gr.Row():
        video = gr.Video(label="Video")
        with gr.Column():
            question_box = gr.Textbox(
                label="Question",
                lines=3,
                value=DEFAULT_QUESTION,
            )
            max_tokens = gr.Slider(
                minimum=16,
                maximum=512,
                step=16,
                value=256,
                label="Max new tokens",
            )
            run_button = gr.Button("Run")

    output_box = gr.Textbox(label="Model answer", lines=6)

    run_button.click(
        fn=qtsplus_generate,
        inputs=[video, question_box, max_tokens],
        outputs=output_box,
    )


if __name__ == "__main__":
    demo.queue().launch()
