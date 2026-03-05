"""
# -*- encoding: utf-8 -*-
# @File        :   llava_video_178k_qa_dataset.py
# @Time        :   2026/01/24
# @Author      :   Siyou

Dataset loader for LLaVA-Video-178K style JSONL, reusing the same processing logic as
`src/dataset/sharegptvideo_qa_dataset.py` (Qwen2.5-VL vision + text training).

Expected JSONL line structure (example):
{
  "id": "Zi4HnhNcrEY",
  "conversations": [
    {"from": "human", "value": "<video>\\nOffer a detailed interpretation of the video's message and imagery."},
    {"from": "gpt", "value": "..." }
  ],
  "video": "liwei_youtube_videos/videos/youtube_video_2024/ytb_Zi4HnhNcrEY.mp4"
}

This dataset normalizes each entry into the ShareGPTVideoQA schema:
{
  "vision_id": <video path>,
  "question": <human prompt without <video>/<image> tokens>,
  "prediction": <assistant answer>,
  "data_type": "video"
}
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from src.dataset.sharegptvideo_qa_dataset import ShareGPTVideoQADataset


_MEDIA_TOKEN_RE = re.compile(r"</?(video|image)>", re.IGNORECASE)


class LlavaVideo178KQADataset(ShareGPTVideoQADataset):
    """
    A thin adapter over ShareGPTVideoQADataset that parses the LLaVA-Video-178K JSONL format.
    """

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
        # Support jsonl paths relative to CWD (repo root), even when `base_path` points to a
        # separate media root. If `jsonl_path` exists as provided, make it absolute so the
        # parent class's `os.path.join(base_path, jsonl_path)` resolves correctly.
        if jsonl_path and not os.path.isabs(jsonl_path) and os.path.exists(jsonl_path):
            jsonl_path = os.path.abspath(jsonl_path)
        super().__init__(
            base_path=base_path,
            jsonl_path=jsonl_path,
            processor=processor,
            tokenizer=tokenizer,
            prompt_style=prompt_style,
            max_length=max_length,
            system_prompt=system_prompt,
            local_rank=local_rank,
            train=train,
            video_max_frames=video_max_frames,
            video_min_frames=video_min_frames,
            video_sampling=video_sampling,
        )

    @staticmethod
    def _extract_qa(conversations: list[dict]) -> tuple[Optional[str], Optional[str]]:
        q: Optional[str] = None
        a: Optional[str] = None
        for turn in conversations:
            role = turn.get("from", None)
            text = turn.get("value", None)
            if not isinstance(text, str):
                continue
            if role == "human" and q is None:
                q = text
            elif role == "gpt" and a is None:
                a = text
            if q is not None and a is not None:
                break
        return q, a

    @staticmethod
    def _clean_question(text: str) -> str:
        # Strip media placeholders like "<video>" and "<image>" used by LLaVA-style prompts.
        text = _MEDIA_TOKEN_RE.sub("", text)
        return text.strip()

    @staticmethod
    def _resolve_vision_id(video_rel: str, data_source: Optional[str], base_path: Optional[str]) -> str:
        """
        Resolve the relative media path inside LLaVA-Video-178K.

        The provided JSONL typically stores:
          - data_source: e.g. "0_30_s_youtube_v0_1"
          - video: e.g. "liwei_youtube_videos/videos/.../xxx.mp4"

        In our repo layout, the actual file lives under:
          <base_path>/<data_source>/<video>

        However, some callers may already pass `base_path` pointing at the data_source folder,
        in which case `<base_path>/<video>` already exists.
        """
        if not isinstance(video_rel, str):
            return video_rel

        # If the media exists directly under base_path, keep as-is.
        if base_path is not None and os.path.exists(os.path.join(base_path, video_rel)):
            return video_rel

        # Otherwise, prefix with data_source when available.
        if isinstance(data_source, str) and len(data_source) > 0:
            candidate = os.path.join(data_source, video_rel)
            if base_path is None or os.path.exists(os.path.join(base_path, candidate)):
                return candidate

        return video_rel

    @staticmethod
    def _load_jsonl(path: str, base_path: str = None) -> List[Dict[str, Any]]:
        """
        Load LLaVA-Video-178K JSONL and filter out invalid entries (and optionally missing media).
        """
        out: List[Dict[str, Any]] = []
        skipped_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        skipped_count += 1
                        continue

                    video_rel = raw.get("video", None)
                    data_source = raw.get("data_source", None)
                    conversations = raw.get("conversations", None)
                    if not isinstance(video_rel, str) or not isinstance(conversations, list) or len(conversations) == 0:
                        skipped_count += 1
                        continue

                    vision_id = LlavaVideo178KQADataset._resolve_vision_id(video_rel, data_source, base_path)

                    q_raw, a_raw = LlavaVideo178KQADataset._extract_qa(conversations)
                    if q_raw is None or a_raw is None:
                        skipped_count += 1
                        continue

                    question = LlavaVideo178KQADataset._clean_question(q_raw)
                    prediction = a_raw.strip()
                    if len(question) == 0 or len(prediction) == 0:
                        skipped_count += 1
                        continue

                    ann: Dict[str, Any] = {
                        "vision_id": vision_id,
                        "question": question,
                        "prediction": prediction,
                        "data_type": "video",
                    }
                    if "id" in raw:
                        ann["id"] = raw["id"]
                    if isinstance(data_source, str) and len(data_source) > 0:
                        ann["data_source"] = data_source

                    # Check if video path exists when base_path is provided
                    if base_path is not None:
                        vision_path = os.path.join(base_path, ann["vision_id"])
                        if not os.path.exists(vision_path):
                            skipped_count += 1
                            continue
                        if os.path.isdir(vision_path):
                            image_files = [
                                fn
                                for fn in os.listdir(vision_path)
                                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                            ]
                            if not image_files:
                                skipped_count += 1
                                continue

                    out.append(ann)

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue

        if base_path is not None and skipped_count > 0:
            print(
                f"[Dataset] Loaded {len(out)} valid annotations, skipped {skipped_count} "
                f"entries with missing/empty video folders"
            )

        return out
