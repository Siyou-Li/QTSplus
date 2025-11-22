import argparse
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from src.model.language_model import Qwen2_5_VLTextForCausalLM


def split_weights(model_path: str, vision_out: str, lm_out: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    Path(vision_out).mkdir(parents=True, exist_ok=True)
    Path(lm_out).mkdir(parents=True, exist_ok=True)

    # Save vision model
    vision_model = model.model.visual
    vision_model.save_pretrained(vision_out)
    processor.save_pretrained(vision_out)
    model.config.vision_config.save_pretrained(vision_out)

    # Save language model as CausalLM
    text_config = model.config.text_config
    lm_model = Qwen2_5_VLTextForCausalLM(text_config)
    lm_model.model.load_state_dict(model.model.language_model.state_dict())
    lm_model.lm_head.load_state_dict(model.lm_head.state_dict())
    lm_model.save_pretrained(lm_out)
    tokenizer.save_pretrained(lm_out)


def main():
    parser = argparse.ArgumentParser(description="Split Qwen2.5-VL weights")
    parser.add_argument("--model_path", help="Path to original Qwen2.5-VL checkpoint")
    parser.add_argument("--vision_out", default=None, help="Output directory for vision model")
    parser.add_argument("--lm_out", default=None, help="Output directory for language model")
    args = parser.parse_args()

    if args.vision_out is None:
        args.vision_out = args.model_path + "-Vision"
    if args.lm_out is None:
        args.lm_out = args.model_path + "-LM"
    split_weights(args.model_path, args.vision_out, args.lm_out)


if __name__ == "__main__":
    main()

