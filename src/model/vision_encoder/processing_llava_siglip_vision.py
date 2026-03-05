from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor


def _is_image_file(fn: str) -> bool:
    return fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))


def _strip_file_prefix(path: str) -> str:
    return path[7:] if isinstance(path, str) and path.startswith("file://") else path


class LlavaSiglipVisionProcessor:
    """
    Lightweight vision processor for LLaVA(-Video) SigLIP towers.

    It consumes the same `vision_info_list` structure produced by
    `src.utils.qwen_vision_process.extract_vision_info(...)` so existing datasets
    can be reused. Unlike Qwen2.5-VL's processor, it does not produce grid_thw.
    """

    def __init__(
        self,
        image_processor,
        *,
        max_video_frames: int = 50,
        default_video_fps: float = 1.0,
    ) -> None:
        self.image_processor = image_processor
        self.max_video_frames = int(max_video_frames)
        self.default_video_fps = float(default_video_fps)

    @classmethod
    def from_pretrained(cls, model_path: str, *, trust_remote_code: bool = True, **kwargs):
        img_proc = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return cls(img_proc, **kwargs)

    def _load_image(self, path: str) -> Image.Image:
        path = _strip_file_prefix(path)
        return Image.open(path).convert("RGB")

    def _load_video_frames_from_folder(self, folder: str, *, max_frames: int) -> List[Image.Image]:
        files = [f for f in os.listdir(folder) if _is_image_file(f)]
        if not files:
            return []
        files = sorted(files)
        vlen = len(files)
        if vlen > max_frames:
            sample = getattr(self, "video_sampling", "uniform")
            if sample in {"random"}:
                sample = "rand"
            sample = str(sample or "uniform").lower()
            if sample == "rand":
                acc_samples = min(int(max_frames), vlen)
                intervals = [int(round(i * vlen / acc_samples)) for i in range(acc_samples + 1)]
                ranges = [(intervals[i], max(intervals[i + 1] - 1, intervals[i])) for i in range(acc_samples)]
                idx = []
                for start, end in ranges:
                    if start >= end:
                        idx.append(start)
                    else:
                        idx.append(random.randint(start, end))
                files = [files[i] for i in idx][:max_frames]
            else:
                step = max(1, vlen // max_frames)
                files = files[::step][:max_frames]
        return [Image.open(os.path.join(folder, f)).convert("RGB") for f in files]

    def _load_video_frames_from_file(self, video_path: str, *, fps: float, max_frames: int) -> List[torch.Tensor]:
        # Use torchvision's FFmpeg-backed reader when available.
        import torchvision
        from torchvision import io

        video_path = _strip_file_prefix(video_path)
        video, _audio, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
        total_frames = int(video.shape[0])
        src_fps = float(info.get("video_fps", 0.0) or 0.0)
        if total_frames <= 0:
            return []

        # Fall back to uniform sampling when fps metadata is missing.
        if src_fps <= 0.0 or fps <= 0.0:
            n = min(total_frames, max_frames)
        else:
            duration = total_frames / src_fps
            n = int(round(duration * fps))
            n = max(1, min(n, max_frames, total_frames))

        sample = getattr(self, "video_sampling", "uniform")
        if sample in {"random"}:
            sample = "rand"
        sample = str(sample or "uniform").lower()
        if sample == "rand" and n > 1 and total_frames > 1:
            # Interval-based random sampling for temporal coverage (sorted by construction).
            acc_samples = min(int(n), int(total_frames))
            intervals = [int(round(i * total_frames / acc_samples)) for i in range(acc_samples + 1)]
            ranges = [(intervals[i], max(intervals[i + 1] - 1, intervals[i])) for i in range(acc_samples)]
            idx_list: List[int] = []
            for start, end in ranges:
                if start >= end:
                    idx_list.append(start)
                else:
                    idx_list.append(random.randint(start, end))
            idx = torch.tensor(idx_list, dtype=torch.long)
        else:
            idx = torch.linspace(0, total_frames - 1, n).round().long()
        frames = video.index_select(0, idx).contiguous()
        # Return per-frame uint8 tensors (C, H, W) to feed into SiglipImageProcessor.
        if frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)
        return [frames[i] for i in range(frames.shape[0])]

    def _decode_video(self, ele: Dict[str, Any]) -> List[Union[Image.Image, torch.Tensor]]:
        video = ele.get("video", None)
        if video is None:
            return []

        max_frames = int(ele.get("max_frames", self.max_video_frames))
        max_frames = max(1, max_frames)
        fps = float(ele.get("fps", self.default_video_fps))
        self.video_sampling = str(ele.get("sampling", getattr(self, "video_sampling", "uniform")) or "uniform").lower()

        if isinstance(video, (list, tuple)):
            frames: List[Image.Image] = []
            for p in video:
                if not isinstance(p, str):
                    continue
                frames.append(self._load_image(p))
                if len(frames) >= max_frames:
                    break
            return frames

        if isinstance(video, str):
            video_path = _strip_file_prefix(video)
            if os.path.isdir(video_path):
                return self._load_video_frames_from_folder(video_path, max_frames=max_frames)
            return self._load_video_frames_from_file(video_path, fps=fps, max_frames=max_frames)

        return []

    def __call__(self, vision_info_list: Sequence[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        images: List[Union[Image.Image, torch.Tensor]] = []
        is_video = False

        for ele in vision_info_list:
            if not isinstance(ele, dict):
                continue
            if "video" in ele or ele.get("type", None) == "video":
                frames = self._decode_video(ele)
                images.extend(frames)
                is_video = True
            elif "image" in ele or "image_url" in ele or ele.get("type", None) in {"image", "image_url"}:
                path = ele.get("image", ele.get("image_url"))
                if isinstance(path, str):
                    images.append(self._load_image(path))

        if not images:
            return {}

        out = self.image_processor(images=images, return_tensors="pt")
        pixel_values = out["pixel_values"]
        return {
            "pixel_values": pixel_values,
            "is_video": is_video,
        }


__all__ = ["LlavaSiglipVisionProcessor"]
