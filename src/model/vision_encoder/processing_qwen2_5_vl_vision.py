# -*- encoding: utf-8 -*-
# @File        :   qwen_vl_utils.py
# @Time        :   2025/06/25 15:43:42
# @Author      :   Siyou
# @Description :

from typing import List, Optional, Union, Dict, Any

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from src.utils.qwen_vision_process import (
    fetch_image,
    fetch_video,
    process_vision_info,
)

class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: Union[List[float], float]


class Qwen2_5_VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2_5_VLImagesKwargs
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }


class Qwen2_5_VLVisionProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2.5-VL processor which wraps a Qwen2.5-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2_5_VLVisionProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5_VLVisionProcessor.__call__`] and [`~Qwen2_5_VLVisionProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen2_5_VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]

    def __call__(
        self,
        vision_inputs: Dict[str, Any],
        *,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Process vision inputs or formatted messages.

        Parameters
        ----------
        vision_inputs:
            Either a single path/dict, a list of them, or a conversation
            message in the format used by :func:`process_vision_info`.
        return_tensors:
            Tensor format of the returned values. Defaults to ``"pt"``.
        """
         ## Read images or videos
        images = []
        videos = []
        video_sample_fps_list = []
        for vision_info in vision_inputs:
            if "image" in vision_info or "image_url" in vision_info:
                images.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
                video_sample_fps_list.append(video_sample_fps)
                videos.append(video_input)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        # Don't pass fps to the video processor to avoid loading too many frames
        # which may lead to excessive memory usage. Instead, use the sampled
        # fps only for computing ``second_per_grid_ts``.
        video_kwargs = {"fps": video_sample_fps_list} if len(video_sample_fps_list) > 0 else {}

        out: Dict[str, Any] = {}
        if images is not None and self.image_processor is not None:
            out.update(self.image_processor(images=images, return_tensors=return_tensors))
        if videos is not None and self.video_processor is not None:
            vp_inputs = self.video_processor(videos=videos, return_tensors=return_tensors, **(video_kwargs or {}))
            video_grid_thw = vp_inputs["video_grid_thw"]
            fps = video_sample_fps_list if len(video_sample_fps_list) > 0 else 2.0
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / f for f in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            vp_inputs.update({"second_per_grid_ts": second_per_grid_ts})
            out.update(vp_inputs)
        out["video_kwargs"] = video_kwargs
        return out
    
__all__ = ["Qwen2_5_VLVisionProcessor"]
