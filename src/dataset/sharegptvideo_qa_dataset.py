"""
# -*- encoding: utf-8 -*-
# @File        :   vision_dataset_qwen2_5_vl.py
# @Time        :   2025/06/06 00:35:17
# @Author      :   Siyou

Dataset utilities for Qwen2.5-VL vision + text training (QTSplusTokenizer).

dataset structure:
{
    "vision_id": "v_--1DO2V4K74-Scene-002",
    "question": "What does the absence of visible safety ropes suggest about the climbing style?",
    "prediction": "The absence of visible safety ropes suggests that the climber is using a lead climbing technique. In lead climbing, the climber ascends the rock face without a rope, relying on their own strength and the security of the holds they use. This style requires more skill and control from the climber, as they must manage their own weight and the placement of each hold."
}
"""

import os
import json
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset

from src.utils.qwen_vision_process import process_vision_info


class ShareGPTVideoQADataset(Dataset):
    def __init__(
        self,
        base_path: str,
        jsonl_path: str,
        processor: str,
        max_length: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
        local_rank: int = 0,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.annotations = self._load_jsonl(os.path.join(base_path, jsonl_path), base_path)

        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Token strings used in chat template expansion
        self.image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        self.video_token = getattr(self.processor, "video_token", "<|video_pad|>")

        self.max_length = max_length
        self.system_prompt = system_prompt
        self.local_rank = local_rank
        self.train = train

    def __len__(self) -> int:
        return len(self.annotations)

    @staticmethod
    def _load_jsonl(path: str, base_path: str = None) -> List[Dict[str, Any]]:
        """
        Load JSONL file and optionally filter entries based on video folder existence.

        Args:
            path: Path to the JSONL file
            base_path: Base path for video folders (optional, for existence checking)

        Returns:
            List of annotation dictionaries with existing video folders
        """
        out: List[Dict[str, Any]] = []
        skipped_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    annotation = json.loads(line)
                    # Check "question", "answer" and "options" existence
                    if ("question" not in annotation) or ("prediction" not in annotation):
                        skipped_count += 1
                        continue
                    # Check "question", "answer" and "options" is NoneType
                    if annotation["question"] is None or annotation["prediction"] is None:
                        skipped_count += 1
                        continue
                    # Check "question" "answer" and "options" is non-empty
                    if len(annotation["question"].strip()) == 0 or len(annotation["prediction"].strip()) == 0:
                        skipped_count += 1
                        continue

                    # Check if video folder exists when base_path is provided
                    if base_path is not None and 'vision_id' in annotation:
                        vision_path = os.path.join(base_path, annotation['vision_id'])

                        # Check if the video folder/file exists
                        if not os.path.exists(vision_path):
                            skipped_count += 1
                            continue

                        # For directories, also check if they contain image files
                        if os.path.isdir(vision_path):
                            image_files = [f for f in os.listdir(vision_path)
                                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                            if not image_files:
                                skipped_count += 1
                                continue

                    out.append(annotation)

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue

        # Log filtering results
        if base_path is not None and skipped_count > 0:
            print(f"[Dataset] Loaded {len(out)} valid annotations, skipped {skipped_count} "
                  f"entries with missing/empty video folders")

        return out

    def expand_video_tokens(self, s: str, video_grid_thw: list, merge_len: int) -> str:
        text = s
        idx = 0
        while self.video_token in text and idx < len(video_grid_thw):
            #num_tokens = int(video_grid_thw[idx].prod().item() // merge_len)
            num_tokens = 1
            text = text.replace(self.video_token, "<|placeholder|>" * num_tokens, 1)
            idx += 1
        return text.replace("<|placeholder|>", self.video_token)

    def _build_messages(self, q_text: str, vision_path: str, data_type: str) -> List[Dict[str, Any]]:
        # Normalize to Qwen-style content blocks
        if data_type == "video":
            if os.path.isdir(vision_path):
                # Directory of frames
                frames = [
                    "file://" + os.path.join(vision_path, f)
                    for f in sorted(os.listdir(vision_path))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if len(frames) >= 50:
                    frames = frames[:: len(frames) // 50]  # Sample down to max 50 frames
                content = [{"type": "video", "video": frames}, {"type": "text", "text": q_text}]
            else:
                # Single video file
                content = [
                    {"type": "video", "video": vision_path, "fps": 1.0, "max_pixels": 360 * 420},
                    {"type": "text", "text": q_text},
                ]
        elif data_type == "image":
            content = [{"type": "image", "image": vision_path}, {"type": "text", "text": q_text}]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        vision_path = os.path.join(self.base_path, ann["vision_id"])
        if not os.path.exists(vision_path):
            if self.local_rank == 0 and self.train:
                print(f"Vision file not found: {vision_path}")
            return None

        # Build messages and strings with Qwen chat template
        raw_q = ann["question"]
        question_input_ids = self.tokenizer(raw_q, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        data_type = ann.get("data_type", "video")
        messages = self._build_messages(raw_q, vision_path, data_type)

        question = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
            )
        qa_messages = messages + [{"role": "assistant", "content": ann["prediction"]}]
        question_and_answer = self.processor.apply_chat_template(
            qa_messages, tokenize=False, add_generation_prompt=False
        )

        # Prepare vision inputs using the Qwen2.5-VL vision processor (for grid meta)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Convert vision info into model-ready tensors via the vision processor
        # Build a minimal list for the processor call (mirrors test pipeline)
        vision_info_list: List[Dict[str, Any]] = []
        if image_inputs is not None:
            for img in image_inputs:
                vision_info_list.append({"type": "image", "image": img})
        if video_inputs is not None:
            for v in video_inputs:
                vision_info_list.append({"type": "video", "video": v})

        vp_out = {}
        if len(vision_info_list) > 0:
            vp_out = self.processor(vision_info_list)

        # Expand <|video_pad|> tokens according to grid so token count matches vision features
        if "video_grid_thw" in vp_out:
            video_grid_thw = vp_out["video_grid_thw"]
            # Determine merge length from the processor
            merge_len = None
            if hasattr(self.processor, "video_processor") and hasattr(self.processor.video_processor, "merge_size"):
                merge_len = self.processor.video_processor.merge_size ** 2
            elif hasattr(self.processor, "video_processor") and hasattr(self.processor.video_processor, "merge_size"):
                merge_len = self.processor.video_processor.merge_size ** 2
            if merge_len is None:
                raise ValueError("Cannot resolve video processor merge_size for token expansion.")

            question = self.expand_video_tokens(question, video_grid_thw, merge_len)
            question_and_answer = self.expand_video_tokens(question_and_answer, video_grid_thw, merge_len)

        # Tokenize after expansion
        q_tensor = self.tokenizer(
            question,
            add_special_tokens=True,
            padding=False,
            return_tensors="pt",
        )
        qa_tensor = self.tokenizer(
            question_and_answer,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = qa_tensor["input_ids"][0]
        attention_mask = qa_tensor["attention_mask"][0]
        question_len = torch.sum(q_tensor["attention_mask"][0])
        labels = input_ids.clone()
        labels[:question_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.eos_token_id] = -100
        labels[labels == 198] = -100  # also mask out vision token id in labels
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_input_ids": question_input_ids,
            # "raw_question_ids": qa_tensor["input_ids"][0],
        }
        # Pack vision tensors under a single key to match model forward signature
        if "pixel_values_videos" in vp_out and "video_grid_thw" in vp_out:
            payload["vision_input"] = {
                "pixel_values_videos": vp_out["pixel_values_videos"],
                "video_grid_thw": vp_out["video_grid_thw"],
            }
        elif "pixel_values" in vp_out and "image_grid_thw" in vp_out:
            payload["vision_input"] = {
                "pixel_values": vp_out["pixel_values"],
                "image_grid_thw": vp_out["image_grid_thw"],
            }

        return payload