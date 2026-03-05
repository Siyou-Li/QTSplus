"""
InternVL2.5 (image) VQA-style dataset for QTSplus training.

This mirrors the JSONL schema used by `ShareGPTVideoQADataset` but formats the prompt
using InternVL's image tokens:
  - `<img> ... </img>` wrapper
  - `<IMG_CONTEXT>` placeholders (one per vision token)
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ShareGPTVideoQAInternVLDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        jsonl_path: str,
        image_processor,
        tokenizer,
        vision_config_path: str,
        max_length: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
        local_rank: int = 0,
        train: bool = True,
        video_max_frames: int = 4,
        video_min_frames: int = 4,
        video_sampling: str = "rand",
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.annotations = self._load_jsonl(os.path.join(base_path, jsonl_path), base_path)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.local_rank = local_rank
        self.train = train
        self.video_max_frames = int(video_max_frames)
        self.video_min_frames = int(video_min_frames)
        self.video_sampling = str(video_sampling)

        # InternVL image tokens
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_context_token = "<IMG_CONTEXT>"

        self.num_image_token = self._infer_num_image_tokens(vision_config_path)

    @staticmethod
    def _load_jsonl(path: str, base_path: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    ann = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if ("question" not in ann) or ("prediction" not in ann):
                    skipped += 1
                    continue
                if ann["question"] is None or ann["prediction"] is None:
                    skipped += 1
                    continue
                if len(str(ann["question"]).strip()) == 0 or len(str(ann["prediction"]).strip()) == 0:
                    skipped += 1
                    continue

                if base_path is not None and "vision_id" in ann:
                    vision_path = os.path.join(base_path, ann["vision_id"])
                    if not os.path.exists(vision_path):
                        skipped += 1
                        continue

                out.append(ann)

        if base_path is not None and skipped > 0:
            print(f"[Dataset/InternVL] Loaded {len(out)} valid annotations, skipped {skipped} missing/invalid entries")
        return out

    @staticmethod
    def _infer_num_image_tokens(vision_model_path: str) -> int:
        cfg_path = os.path.join(vision_model_path, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing InternVL vision config.json at: {cfg_path}")
        cfg = _load_json(cfg_path)

        # Accept either InternVL chat config or the split InternVL2_5VisionConfig.
        vision_cfg = cfg.get("vision_config") or {}
        force_image_size = cfg.get("force_image_size", None)
        image_size = force_image_size if isinstance(force_image_size, int) else vision_cfg.get("image_size", 448)
        patch_size = vision_cfg.get("patch_size", 14)
        downsample_ratio = cfg.get("downsample_ratio", 0.5)

        return int((int(image_size) // int(patch_size)) ** 2 * (float(downsample_ratio) ** 2))

    def __len__(self) -> int:
        return len(self.annotations)

    def _build_image_tokens(self, num_images: int = 1) -> str:
        num_images = max(int(num_images), 1)
        one = self.img_start_token + (self.img_context_token * int(self.num_image_token)) + self.img_end_token
        return one * num_images

    @staticmethod
    def _is_image_file(fn: str) -> bool:
        return fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))

    @staticmethod
    def _extract_frame_sort_key(filename: str) -> tuple[int, int]:
        # Try common patterns like: c01_0001.jpeg, frame_12.jpg, 000123.png
        stem = os.path.splitext(os.path.basename(filename))[0]
        # Prefer the QTS frame-folder convention: cXX_YYYY
        m = re.search(r"^c(\d+)[-_](\d+)$", stem, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass

        nums = re.findall(r"\d+", stem)
        try:
            if len(nums) >= 2:
                return int(nums[-2]), int(nums[-1])
            if len(nums) == 1:
                return 0, int(nums[0])
        except Exception:
            pass
        return 0, -1

    def _sorted_frame_files(self, folder: str) -> List[str]:
        files = [f for f in os.listdir(folder) if self._is_image_file(f)]
        return sorted(files, key=lambda x: (self._extract_frame_sort_key(x), x))

    @staticmethod
    def _sample_indices(num_frames: int, vlen: int, sample: str = "rand") -> List[int]:
        if vlen <= 0:
            return []
        acc_samples = min(int(num_frames), int(vlen))
        if acc_samples <= 0:
            return []

        sample = (sample or "rand").lower()

        # Uniform sampling across the full video length (deterministic).
        # Similar to `np.linspace(0, vlen - 1, acc_samples)` with integer rounding.
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

        # Split into `acc_samples` intervals and sample one index per interval (InternVL-style).
        # Use integer arithmetic to avoid numpy.
        intervals = [int(round(i * vlen / acc_samples)) for i in range(acc_samples + 1)]
        ranges = [(intervals[i], max(intervals[i + 1] - 1, intervals[i])) for i in range(acc_samples)]

        out: List[int] = []
        for start, end in ranges:
            if sample == "middle":
                out.append((start + end) // 2)
            else:  # default: rand
                if start >= end:
                    out.append(start)
                else:
                    out.append(random.randint(start, end))

        # Pad with last frame to fixed length if needed.
        if len(out) < num_frames:
            out = out + [out[-1]] * (num_frames - len(out))
        return out

    def _load_video_frames(self, vision_path: str) -> List[Image.Image]:
        files = self._sorted_frame_files(vision_path)
        if not files:
            return []
        vlen = len(files)

        max_f = max(int(self.video_max_frames), 1)
        min_f = max(int(self.video_min_frames), 1)
        if min_f > max_f:
            min_f = max_f

        # Match InternVL behavior: randomize the number of frames during training.
        if self.train:
            t_num = random.randint(min(min_f, vlen), min(max_f, vlen))
            indices = self._sample_indices(t_num, vlen, sample=self.video_sampling)
        else:
            t_num = min(max_f, vlen)
            indices = self._sample_indices(t_num, vlen, sample="middle")

        frames: List[Image.Image] = []
        for i in indices:
            i = max(0, min(int(i), vlen - 1))
            fp = os.path.join(vision_path, files[i])
            frames.append(Image.open(fp).convert("RGB"))
        return frames

    def _load_vision(self, vision_path: str, data_type: str) -> List[Image.Image]:
        # For QTS frame folders, treat directories as videos (frames).
        if os.path.isdir(vision_path):
            frames = self._load_video_frames(vision_path)
            if frames:
                return frames
            # Fallback: try first frame if sampling failed.
            for fn in self._sorted_frame_files(vision_path):
                return [Image.open(os.path.join(vision_path, fn)).convert("RGB")]
            return []
        # Non-directory: treat as single image.
        return [Image.open(vision_path).convert("RGB")]

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        ann = self.annotations[idx]
        vision_path = os.path.join(self.base_path, ann["vision_id"])
        if not os.path.exists(vision_path):
            if self.local_rank == 0 and self.train:
                print(f"Vision file not found: {vision_path}")
            return None

        raw_q = str(ann["question"])
        answer = str(ann["prediction"])

        # Query tokens for QTS+ (text-only question to avoid leakage)
        question_input_ids = self.tokenizer(raw_q, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        data_type = str(ann.get("data_type", "video") or "video").lower()
        images = self._load_vision(vision_path, data_type=data_type)
        if not images:
            if self.local_rank == 0 and self.train:
                print(f"Empty vision frames: {vision_path}")
            return None

        img_tokens = self._build_image_tokens(num_images=len(images))
        user_content = f"{img_tokens}\n{raw_q}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template; please use a tokenizer with chat_template.")

        question = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        qa_messages = messages + [{"role": "assistant", "content": answer}]
        question_and_answer = self.tokenizer.apply_chat_template(qa_messages, tokenize=False, add_generation_prompt=False)

        q_tensor = self.tokenizer(
            question,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        qa_tensor = self.tokenizer(
            question_and_answer,
            add_special_tokens=False,
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

        # Mask vision placeholder tokens in labels
        try:
            ctx_id = self.tokenizer.convert_tokens_to_ids(self.img_context_token)
            labels[labels == ctx_id] = -100
        except Exception:
            pass

        # Vision preprocessing
        vp = self.image_processor(images=images, return_tensors="pt")
        pixel_values = vp["pixel_values"]  # [T, 3, H, W] (or [1, 3, H, W])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_input_ids": question_input_ids,
            "vision_input": pixel_values,
        }
