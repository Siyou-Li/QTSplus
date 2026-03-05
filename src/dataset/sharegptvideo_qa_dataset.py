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
import random
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset

from src.utils.qwen_vision_process import extract_vision_info


def _is_image_file(name: str) -> bool:
    return str(name).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def _sample_indices(num_frames: int, vlen: int, sample: str = "uniform") -> List[int]:
    if vlen <= 0:
        return []
    num_frames = int(num_frames)
    if num_frames <= 0:
        return []

    acc_samples = min(num_frames, int(vlen))
    if acc_samples <= 0:
        return []

    sample = (sample or "uniform").lower()
    if sample in {"random"}:
        sample = "rand"

    if sample == "uniform":
        if acc_samples == 1:
            out = [max(0, (vlen - 1) // 2)]
        else:
            denom = acc_samples - 1
            last = vlen - 1
            out = [int((i * last + (denom // 2)) // denom) for i in range(acc_samples)]
        if len(out) < num_frames:
            out = out + [out[-1]] * (num_frames - len(out))
        return out

    if sample != "rand":
        raise ValueError(f"video_sampling must be one of {{uniform, rand}}, got: {sample}")

    intervals = [int(round(i * vlen / acc_samples)) for i in range(acc_samples + 1)]
    ranges = [(intervals[i], max(intervals[i + 1] - 1, intervals[i])) for i in range(acc_samples)]

    out: List[int] = []
    for start, end in ranges:
        if start >= end:
            out.append(start)
        else:
            out.append(random.randint(start, end))

    if len(out) < num_frames:
        out = out + [out[-1]] * (num_frames - len(out))
    return out


class ShareGPTVideoQADataset(Dataset):
    def __init__(
        self,
        base_path: str,
        jsonl_path: str,
        processor: str,
        tokenizer=None,
        prompt_style: str = "qwen2_5_vl",
        max_length: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
        local_rank: int = 0,
        train: bool = True,
        video_max_frames: int = 50,
        video_min_frames: int = 50,
        video_sampling: str = "uniform",
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.annotations = self._load_jsonl(os.path.join(base_path, jsonl_path), base_path)

        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.prompt_style = prompt_style

        # Token strings used in chat template expansion
        if str(prompt_style).lower().startswith("llava"):
            # LLaVA-style prompts use the single `<image>` placeholder token.
            self.image_token = "<image>"
            self.video_token = "<image>"
        else:
            self.image_token = "<|image_pad|>"
            self.video_token = "<|video_pad|>"

        self.max_length = max_length
        self.system_prompt = system_prompt
        self.local_rank = local_rank
        self.train = train
        self.video_max_frames = int(video_max_frames)
        self.video_min_frames = int(video_min_frames)
        self.video_sampling = str(video_sampling or "uniform")

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
                files = [f for f in os.listdir(vision_path) if _is_image_file(f)]
                files = sorted(files)
                vlen = len(files)

                max_f = max(int(self.video_max_frames), 1)
                min_f = max(int(self.video_min_frames), 1)
                if min_f > max_f:
                    min_f = max_f

                if self.train:
                    t_num = random.randint(min(min_f, vlen), min(max_f, vlen)) if vlen > 0 else 0
                    sampling = self.video_sampling
                else:
                    t_num = min(max_f, vlen)
                    sampling = "uniform"

                indices = _sample_indices(t_num, vlen, sample=sampling)
                frames = ["file://" + os.path.join(vision_path, files[i]) for i in indices] if indices else []
                content = [
                    {
                        "type": "video",
                        "video": frames,
                        "max_frames": int(t_num) if t_num else 0,
                        "min_frames": int(min_f),
                        "sampling": str(sampling or "uniform").lower(),
                    },
                    {"type": "text", "text": q_text},
                ]
            else:
                # Single video file
                max_f = max(int(self.video_max_frames), 1)
                min_f = max(int(self.video_min_frames), 1)
                if min_f > max_f:
                    min_f = max_f
                t_num = random.randint(min_f, max_f) if self.train else max_f
                content = [
                    {
                        "type": "video",
                        "video": vision_path,
                        "fps": 1.0,
                        "max_pixels": 360 * 420,
                        "max_frames": t_num,
                        "min_frames": min_f,
                        "sampling": str(self.video_sampling or "uniform").lower(),
                    },
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
        vision_messages = self._build_messages(raw_q, vision_path, data_type)

        if self.prompt_style == "qwen2_5_vl":
            question = self.processor.apply_chat_template(
                vision_messages, tokenize=False, add_generation_prompt=True
            )
            qa_messages = vision_messages + [{"role": "assistant", "content": ann["prediction"]}]
            question_and_answer = self.processor.apply_chat_template(
                qa_messages, tokenize=False, add_generation_prompt=False
            )
            add_special_tokens = True
        else:
            user_text = f"{self.video_token}\n{raw_q}" if data_type == "video" else f"{self.image_token}\n{raw_q}"
            text_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ]
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not support apply_chat_template; please update transformers.")
            question = self.tokenizer.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True
            )
            qa_messages = text_messages + [{"role": "assistant", "content": ann["prediction"]}]
            question_and_answer = self.tokenizer.apply_chat_template(
                qa_messages, tokenize=False, add_generation_prompt=False
            )
            add_special_tokens = False

        # Extract raw vision entries (paths/URLs/frames) and let the processor
        # handle decoding. Avoid pre-decoding then re-processing, which can
        # yield tensors in the "video" field and break `fetch_video()`.
        vision_info_list = extract_vision_info(vision_messages)
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
            add_special_tokens=add_special_tokens,
            padding=False,
            return_tensors="pt",
        )
        qa_tensor = self.tokenizer(
            question_and_answer,
            add_special_tokens=add_special_tokens,
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
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        if self.tokenizer.eos_token_id is not None:
            labels[labels == self.tokenizer.eos_token_id] = -100
        # Mask the placeholder vision token id in labels (if present).
        try:
            v_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
            labels[labels == v_id] = -100
        except Exception:
            pass
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_input_ids": question_input_ids,
            # "raw_question_ids": qa_tensor["input_ids"][0],
        }
        # Pack vision tensors under a single key to match model forward signature
        if "pixel_values_videos" in vp_out:
            payload["vision_input"] = {"pixel_values_videos": vp_out["pixel_values_videos"]}
            if "video_grid_thw" in vp_out:
                payload["vision_input"]["video_grid_thw"] = vp_out["video_grid_thw"]
        elif "pixel_values" in vp_out:
            payload["vision_input"] = {"pixel_values": vp_out["pixel_values"]}
            if "image_grid_thw" in vp_out:
                payload["vision_input"]["image_grid_thw"] = vp_out["image_grid_thw"]

        return payload
if __name__ == '__main__':

    from src.model.vision_encoder import Qwen2_5_VLVisionProcessor
    from config import config
    from torch.utils.data import Dataset, DataLoader

    base_path = "/home/ubuntu/mnt/train_300k_480p"
    jsonl_path = "/home/ubuntu/data/siyou/QTSplus/datasets/ShareGPTVideoChoice/3b/qa/prediction_correct_train.jsonl"
    processor = Qwen2_5_VLVisionProcessor.from_pretrained("/data/siyou/QTSplus/pretrained_models/Qwen2.5-VL-3B-Instruct-Vision")
    dataset = ShareGPTVideoQADataset(
        base_path = base_path,
        jsonl_path = jsonl_path,
        processor = processor,
        max_length = 512
    )
    print(f"[*] Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        if batch is None:
            continue
        rets = batch
        print("[*] Vision input shape:", rets["vision_input"]["pixel_values_videos"].shape)
        print("[*] Input ids shape:", rets["input_ids"].shape)
        print("--" * 20)
        # print label token sequence
        # print("[*] Full token sequence:\n", processor.tokenizer.decode(rets["input_ids"][0]))
        # print("--" * 20)
        
        label_token = []
        for i in range(len(rets["labels"][0])):
            if rets["labels"][0][i] != -100:
                label_token.append(rets["labels"][0][i])
        print("[*] Answer token ids:", label_token)
        print("[*] Label token sequence:", processor.tokenizer.decode(label_token, skip_special_tokens=False))
        print("--" * 20)
        # print unmasked token sequence
        unmasked_token = rets["input_ids"][0] * rets["attention_mask"][0]
        unmasked_token = unmasked_token[unmasked_token != 0]
        print("[*] Unmasked token sequence:\n", processor.tokenizer.decode(unmasked_token))
        print("--" * 20)
        # vision_token_in_input = processor.tokenizer.decode(rets["input_ids"][0][rets["vision_token_index"]])
        # print("[*] Vision token string in input:", vision_token_in_input, " Correct!" if vision_token_in_input == dataset.vision_token else " Wrong!")
        print("[*] Question token ids:\n", processor.tokenizer.decode(rets["question_input_ids"][0]))
        break
