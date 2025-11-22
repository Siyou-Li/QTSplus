import argparse
from ast import parse
import glob
import sys
import os
import torch
from safetensors.torch import save_file
import shutil


# Ensure project root (contains src/) is on PYTHONPATH when running the demo directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from src.utils.qwen_vision_process import process_vision_info


def build_messages(video: str | None, images_dir: str | None, prompt: str) -> list[list[dict]]:
    msgs_list: list[list[dict]] = []
    if video:
        msgs_list.append([
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video, "max_pixels": 360 * 420, "fps": 1.0},
                    {"type": "text", "text": prompt or "Describe this video."},
                ],
            }
        ])
    if images_dir:
        image_list = sorted(glob.glob(os.path.join(images_dir, "*.jpeg")))
        if not image_list:
            image_list = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        if image_list:
            msgs_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": image_list},
                        {"type": "text", "text": prompt or "What is in these images?"},
                    ],
                }
            ])
    return msgs_list

def convert_bin_to_safetensors(model_dir: str):
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(model_dir, "model.safetensors")

    print(f"Loading checkpoint from {bin_path}...")

    # Load the checkpoint
    checkpoint = torch.load(bin_path, map_location="cpu", weights_only=False)

    print(f"Checkpoint keys: {len(checkpoint.keys())}")
    print(f"Sample keys: {list(checkpoint.keys())[:5]}")

    # Save as safetensors
    print(f"Saving as safetensors to {safetensors_path}...")
    save_file(checkpoint, safetensors_path)

    # Remove the old .bin file
    print(f"Removing old .bin file...")
    os.remove(bin_path)

    print("✅ Conversion complete!")
    print(f"✅ Model saved as: {safetensors_path}")

def main():
    parser = argparse.ArgumentParser(description="QTS+ demo")
    parser.add_argument("--model", type=str, default="", help="Path to HF model folder")
    parser.add_argument("--video", type=str, default="", help="Path to a video file")
    parser.add_argument("--images_dir", type=str, default="", help="Directory containing frames/images")
    parser.add_argument("--prompt", type=str, default=None, help="User question/prompt")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device like cuda:0 or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length")
    args = parser.parse_args()

    # convert .bin checkpoint to safetensors if needed
    if os.path.isfile(os.path.join(args.model, "pytorch_model.bin")):
        convert_bin_to_safetensors(args.model)
    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    ).to(dtype=torch.float16, device=device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )

    # Ensure generation uses an explicit pad/eos to avoid warnings
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
        if getattr(model.config, "eos_token_id", None) is None:
            model.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id
            model.generation_config.eos_token_id = model.config.eos_token_id

    # Defaults: use bundled examples if no input specified
    if not args.video and not args.images_dir:
        default_video = os.path.join(PROJECT_ROOT, "examples", "example.mov")
        if os.path.exists(default_video):
            args.video = default_video
        default_images = os.path.join(PROJECT_ROOT, "examples", "example_images")
        if os.path.isdir(default_images):
            args.images_dir = default_images

    messages_list = build_messages(args.video, args.images_dir, args.prompt or "")
    if not messages_list:
        print("No valid input provided. Use --video or --images_dir.")
        return

    for messages in messages_list:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(dtype=torch.float16, device=device)

        # Extract and format the vision input for QTS+ model
        pixel_values_videos = inputs.pop('pixel_values_videos', None)
        video_grid_thw = inputs.pop('video_grid_thw', None)
        inputs.pop('second_per_grid_ts', None)  # Remove unused parameter

        # Format vision input as expected by QTS+ model
        vision_input = None
        if pixel_values_videos is not None and video_grid_thw is not None:
            vision_input = {
                'pixel_values_videos': pixel_values_videos,
                'video_grid_thw': video_grid_thw
            }
        print("="*40)
        # Build question_input_ids from the textual question only (avoid including system/vision tokens)
        question_ids = None
        qt = None
        if tok is not None:
            question_texts = []
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for c in msg["content"]:
                        if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                            question_texts.append(c["text"])
            if question_texts:
                qt = "\n".join(question_texts)
                enc = tok(qt, add_special_tokens=False, return_tensors="pt")
                question_ids = enc.input_ids.to(device)
        if qt:
            print("Question:", qt)

        # Inference
        generated_ids = model.generate(
            vision_input=vision_input,
            input_ids=inputs.input_ids,
            question_input_ids=question_ids if question_ids is not None else inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # Fallback: if trimming logic yields empty text (common when using inputs_embeds),
        # decode the full sequences instead.
        output_text = [
            txt if (txt is not None and txt.strip() != "") else processor.decode(ids, skip_special_tokens=True)
            for txt, ids in zip(output_text, generated_ids)
        ]
        print(output_text[0])
        print("="*40)

if __name__ == "__main__":
    main()
