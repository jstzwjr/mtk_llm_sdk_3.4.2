# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Define Qwen2VL model class."""

import json
import math
import os
from typing import Dict, List, Optional, Union

import mtk_quantization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from PIL import Image
from torch.nn import LayerNorm
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType

from ...utils import logger, utils
from ..activations import FastGelu, QuickGelu
from ..modeling_base import BaseVisionEncoderChunk


# FIXME: Put these function here or move to utils.image_utils?
def make_batched_images(images) -> List[List[ImageInput]]:
    """Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    if isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    if is_valid_image(images):
        return [images]

    logger.error(f'Could not make batched images from {images}', err=ValueError)
    return None


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    """Make batched videos from the input.

    Args:
        videos : Union[list, tuple, np.ndarray, Image.Image]
            The input videos, which can be a list or tuple of images, a list or tuple of videos, or a single video.

    Returns:
        List[VideoInput]: A list of batched videos.

    Raises:
        ValueError: If the input videos cannot be batched.
    """
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    if isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        if len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    logger.error(f'Could not make batched video from {videos}', err=ValueError)
    return None


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Do any resolution smart resize.

    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        logger.error(f'height:{height} or width:{width} must be larger than factor:{factor}', err=ValueError)
    if max(height, width) / min(height, width) > 200:
        logger.error(
            f'absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}',
            err=ValueError,
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in
            the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']

    def __init__(
        self,
        do_resize: bool = True,
        resample=PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        """Initialize the Qwen2-VL Image Processor.

        Args:
            do_resize (bool, optional): Whether to resize the image's (height, width) dimensions. Defaults to True.
            resample (PILImageResampling, optional): Resampling filter to use when resizing the image. Defaults to
                Resampling.BICUBIC.
            do_rescale (bool, optional): Whether to rescale the image by the specified scale `rescale_factor`.
                Defaults to True.
            rescale_factor (int or float, optional): Scale factor to use if rescaling the image. Defaults to 1/255.
            do_normalize (bool, optional): Whether to normalize the image. Defaults to True.
            image_mean (float or List[float], optional): Mean to use if normalizing the image.
                Defaults to [0.48145466, 0.4578275, 0.40821073].
            image_std (float or List[float], optional): Standard deviation to use if normalizing the image.
                Defaults to [0.26862954, 0.26130258, 0.27577711].
            do_convert_rgb (bool, optional): Whether to convert the image to RGB. Defaults to True.
            min_pixels (int, optional): The min pixels of the image to resize the image. Defaults to 56 * 56.
            max_pixels (int, optional): The max pixels of the image to resize the image. Defaults to 28 * 28 * 1280.
            patch_size (int, optional): The spacial patch size of the vision encoder. Defaults to 14.
            temporal_patch_size (int, optional): The temporal patch size of the vision encoder. Defaults to 2.
            merge_size (int, optional): The merge size of the vision encoder to llm encoder. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {'min_pixels': min_pixels, 'max_pixels': max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255.
                If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`List[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of
                channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to
                the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or
                    `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning(
                'It looks like you are trying to rescale already rescaled images. If the input'
                ' images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.'
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """Qwen2-VL image preprocessing.

        Args:
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        videos (`VideoInput`):
            Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
            passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio.
        resample (`int`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            The type of tensors to return. Can be one of:
            - Unset: Return a list of `np.ndarray`.
            - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
            - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
            - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        **kwargs: Other kwargs.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            logger.error(
                'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, '
                'torch.Tensor, tf.Tensor or jax.ndarray.',
                err=ValueError,
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {'pixel_values': pixel_values, 'image_grid_thw': vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {'pixel_values_videos': pixel_values, 'video_grid_thw': vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to the vision tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        freqs (torch.Tensor): The frequency tensor.

    Returns:
        torch.Tensor: The tensor with applied rotary positional embedding.
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def precompute_qwen2vl_vision_rot_emb(grid_thw, config):
    """Precompute rotary positional embeddings for Qwen2-VL vision model.

    Args:
        grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).
        config (object): The configuration object containing model parameters.

    Returns:
        torch.Tensor: The precomputed rotary positional embeddings.
    """
    pos_ids = []
    spatial_merge_size = config.spatial_merge_size
    head_dim = config.embed_dim // config.num_heads
    rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
    for t, h, w in grid_thw:
        t = t.to(torch.int32).item()
        h = h.to(torch.int32).item()
        w = w.to(torch.int32).item()
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max().item()
    rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    rotary_pos_emb_cos = rotary_pos_emb.cos().unsqueeze(1)
    rotary_pos_emb_sin = rotary_pos_emb.sin().unsqueeze(1)
    return torch.cat([rotary_pos_emb_cos, rotary_pos_emb_sin], dim=1)


def precompute_qwen2vl_vision_attn_mask(seq_length, cu_seqlens):
    """Precompute attention mask for Qwen2-VL vision model.

    Args:
        seq_length (int): The sequence length.
        cu_seqlens (torch.Tensor): The cumulative sequence lengths.

    Returns:
        torch.Tensor: The precomputed attention mask.
    """
    attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    return attention_mask


class PatchMerger(nn.Module):
    """Patch Merger class for merging patches in vision models.

    Attributes:
        hidden_size (int): The hidden size for the MLP.
        ln_q (LayerNorm): The layer normalization layer.
        mlp (nn.Sequential): The MLP for merging patches.

    Methods:
        __init__(dim, context_dim, spatial_merge_size): Initialize the Patch Merger.
        forward(x): Forward pass for merging patches.
    """

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        """Initialize the Patch Merger.

        Args:
            dim (int): The dimension of the output.
            context_dim (int): The context dimension.
            spatial_merge_size (int, optional): The spatial merge size. Defaults to 2.
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for merging patches.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after merging patches.
        """
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


# For replace conv3d with conv2d
class Conv2dInplaceConv3d(torch.nn.Module):
    """Conv2dInplaceConv3d class."""

    def __init__(self, conv3d):
        """Only stride=1 and bias=False are supported."""
        super().__init__()
        inc, outc, ksize = conv3d.in_channels, conv3d.out_channels, conv3d.kernel_size
        self.conv2d1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, bias=False)
        self.conv2d2 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, bias=False)

        with torch.no_grad():
            self.conv2d1.weight.data = conv3d.weight.data[:, :, 0, :, :]
            self.conv2d2.weight.data = conv3d.weight.data[:, :, 1, :, :]
        self.conv2d1.to(conv3d.weight.data.device)
        self.conv2d2.to(conv3d.weight.data.device)

    def __getattr__(self, attr):
        """Attribute gettter."""
        conv2d1 = self._modules['conv2d1']
        conv2d2 = self._modules['conv2d2']
        if attr == 'conv2d1':
            return conv2d1
        if attr == 'conv2d2':
            return conv2d2
        # Note that since dtype is only obtained here in other places, conv2d1 can be returned.
        return getattr(conv2d1, attr)

    def forward(self, x: torch.Tensor):
        """Forward function."""
        return (self.conv2d1(x[:, 0::2, :, :]) + self.conv2d2(x[:, 1::2, :, :])).unsqueeze(2)


class VisionRotaryEmbedding(nn.Module):
    """Vision Rotary Embedding class for applying rotary positional embeddings in vision models.

    Attributes:
        inv_freq (torch.Tensor): The inverse frequency tensor.

    Methods:
        __init__(dim, theta): Initialize the Vision Rotary Embedding.
        forward(seqlen): Forward pass to compute rotary positional embeddings.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Initialize the Vision Rotary Embedding.

        Args:
            dim (int): The dimension of the embedding.
            theta (float, optional): The base frequency. Defaults to 10000.0.
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen) -> torch.Tensor:
        """Forward pass to compute rotary positional embeddings.

        Args:
            seqlen (int): The sequence length.

        Returns:
            torch.Tensor: The computed rotary positional embeddings.
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class PatchEmbed(nn.Module):
    """Patch Embedding class for converting images to patch embeddings in vision models.

    Attributes:
        patch_size (int): The size of each patch.
        temporal_patch_size (int): The temporal size of each patch.
        in_channels (int): The number of input channels.
        embed_dim (int): The embedding dimension.
        proj (nn.Conv3d): The 3D convolutional layer for projecting patches to embeddings.

    Methods:
        __init__(patch_size, temporal_patch_size, in_channels, embed_dim): Initialize the Patch Embed.
        forward(hidden_states): Forward pass to convert images to patch embeddings.
    """

    def __init__(
        self,
        config,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """Initialize the Patch Embed.

        Args:
            config (Config): The configuration object.
            patch_size (int, optional): The size of each patch. Defaults to 14.
            temporal_patch_size (int, optional): The temporal size of each patch. Defaults to 2.
            in_channels (int, optional): The number of input channels. Defaults to 3.
            embed_dim (int, optional): The embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        if self.config.use_conv2d_patch_embed:
            self.proj = Conv2dInplaceConv3d(
                nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
            )
        else:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass to convert images to patch embeddings.

        Args:
            hidden_states (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with patch embeddings.
        """
        target_dtype = self.proj.weight.dtype
        if self.config.use_conv2d_patch_embed:
            hidden_states = hidden_states.view(
                -1, self.in_channels * self.temporal_patch_size, self.patch_size, self.patch_size
            )
        else:
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class VisionMlp(nn.Module):
    """Vision MLP class for applying MLP layers in vision models.

    Attributes:
        fc1 (nn.Linear): The first linear layer.
        act (nn.Module): The activation function.
        fc2 (nn.Linear): The second linear layer.

    Methods:
        __init__(dim, hidden_dim, hidden_act): Initialize the Vision MLP.
        forward(x): Forward pass for the MLP.
    """

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        """Initialize the Vision MLP.

        Args:
            dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            hidden_act (str): The activation function.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        if hidden_act == 'gelu_pytorch_tanh':
            self.act = FastGelu()
        else:
            self.act = QuickGelu()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        """Forward pass for the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the MLP.
        """
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    """Vision Attention class for applying attention mechanisms in vision models.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        qkv (nn.Linear): The linear layer for query, key, and value projections.
        proj (nn.Linear): The linear layer for the output projection.
        q_mul1 (mtk_quantization.pytorch.functional.Mul): The first multiplication function for query.
        q_mul2 (mtk_quantization.pytorch.functional.Mul): The second multiplication function for query.
        q_add (mtk_quantization.pytorch.functional.Add): The addition function for query.
        q_cat (mtk_quantization.pytorch.functional.Cat): The concatenation function for query.
        k_mul1 (mtk_quantization.pytorch.functional.Mul): The first multiplication function for key.
        k_mul2 (mtk_quantization.pytorch.functional.Mul): The second multiplication function for key.
        k_add (mtk_quantization.pytorch.functional.Add): The addition function for key.
        k_cat (mtk_quantization.pytorch.functional.Cat): The concatenation function for key.

    Methods:
        __init__(dim, num_heads): Initialize the Vision Attention.
        apply_rotary_pos_emb_mtk(q, k, cos, sin): Apply rotary positional embedding using MTK functions.
        forward(hidden_states, attention_mask, pos_emb): Forward pass for the attention mechanism.
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        """Initialize the Vision Attention.

        Args:
            dim (int): The dimension of the input.
            num_heads (int, optional): The number of attention heads. Defaults to 16.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Rotary embedding modules
        self.q_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.q_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.q_add = mtk_quantization.pytorch.functional.Add()
        self.q_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        self.k_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.k_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.k_add = mtk_quantization.pytorch.functional.Add()
        self.k_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding using MTK functions.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            cos (torch.Tensor): The cosine tensor.
            sin (torch.Tensor): The sine tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with applied rotary positional embedding.
        """
        # q1, q2 = torch.split(q, self.head_dim//2, dim=-1)
        q1 = q[..., : torch.div(q.shape[-1], 2, rounding_mode='floor')]
        q2 = q[..., torch.div(q.shape[-1], 2, rounding_mode='floor') :]
        q_rotated = self.q_cat((-q2, q1))
        # k1, k2 = torch.split(k, self.head_dim//2, dim=-1)
        k1 = k[..., : torch.div(k.shape[-1], 2, rounding_mode='floor')]
        k2 = k[..., torch.div(k.shape[-1], 2, rounding_mode='floor') :]
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (torch.Tensor): The attention mask.
            pos_emb (torch.Tensor, optional): The positional embedding. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying attention.
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        # Official
        # q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        # k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # MTK
        cos, sin = torch.split(pos_emb, 1, dim=1)
        cos = cos.repeat(1, 1, 2).float()
        sin = sin.repeat(1, 1, 2).float()
        q, k = self.apply_rotary_pos_emb_mtk(q, k, cos, sin)

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #    attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen2VLVisionBlock(nn.Module):
    """Qwen2-VL Vision Block class for applying vision blocks in the Qwen2-VL model.

    Attributes:
        norm1 (LayerNorm): The first layer normalization layer.
        norm2 (LayerNorm): The second layer normalization layer.
        attn (VisionAttention): The attention layer.
        mlp (VisionMlp): The MLP layer.

    Methods:
        __init__(config): Initialize the Qwen2-VL Vision Block.
        forward(hidden_states, attn_mask, rotary_pos_emb): Forward pass for the vision block.
    """

    def __init__(self, config) -> None:
        """Initialize the Qwen2-VL Vision Block.

        Args:
            config (object): The configuration object containing model parameters.
        """
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, attn_mask, rotary_pos_emb) -> torch.Tensor:
        """Forward pass for the vision block.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): The attention mask.
            rotary_pos_emb (torch.Tensor): The rotary positional embedding.

        Returns:
            torch.Tensor: The output tensor after applying the vision block.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attn_mask, rotary_pos_emb)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class CumulativeSeqenceLens(nn.Module):
    """Cumulative Sequence Lens class for computing cumulative sequence lengths.

    Methods:
        __init__(): Initialize the Cumulative Sequence Lens.
        forward(grid_thw): Forward pass to compute cumulative sequence lengths.
    """

    def __init__(self):
        """Initialize the Cumulative Sequence Lens."""
        super().__init__()

    def forward(self, grid_thw):
        """Forward pass to compute cumulative sequence lengths.

        Args:
            grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).

        Returns:
            torch.Tensor: The cumulative sequence lengths.
        """
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen2VLVisionModel(BaseVisionEncoderChunk):
    """Qwen2-VL Vision Model class for the Qwen2-VL vision model.

    Attributes:
        spatial_merge_size (int): The spatial merge size.
        patch_embed (PatchEmbed): The patch embedding layer.
        rotary_pos_emb (VisionRotaryEmbedding): The rotary positional embedding layer.
        blocks (nn.ModuleList): The list of vision blocks.
        merger (PatchMerger): The patch merger layer.
        jit_trace (bool): Whether to use JIT tracing.
        main_device (str): The main device for the model.
        cu_seqlens (CumulativeSeqenceLens): The cumulative sequence lens.

    Methods:
        __init__(config, jit_trace): Initialize the Qwen2-VL Vision Model.
        get_dtype(): Get the data type of the model.
        get_device(): Get the device of the model.
        rot_pos_emb(grid_thw): Compute rotary positional embeddings.
        forward(hidden_states, attn_mask, rotary_pos_emb): Forward pass for the vision model.
    """

    def __init__(
        self,
        config,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        parallel_lora=False,
        distribute_layers=True,
        **kwargs,
    ) -> None:
        """Initialize the Qwen2-VL Vision Model.

        Args:
            config (object): The configuration object containing model parameters.
            lora (LoRA): The lora object.
            num_layers (int): The number of encoder layers for the current chunk.
            first_layer_idx (int): The index of the first encoder layer in the current chunk.
            chunk_idx (int): The chunk index of the current chunk.
            dtype (torch.dtype): The default dtype to use for this chunk.
            jit_trace (bool): Flag to determine if model is to be run as part of JIT tracing or not.
            parallel_lora (bool): Flag to determine if parallel lora is used in the current chunk.
            distribute_layers (bool): Flag to determine if encoder layers should be evenly distributed among available
                GPUs or not.
            kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            config,
            lora,
            num_layers,
            first_layer_idx,
            chunk_idx,
            dtype,
            jit_trace,
            parallel_lora,
            distribute_layers,
            **kwargs,
        )

        self.spatial_merge_size = config.spatial_merge_size

        if self.chunk_idx == 0:
            self.patch_embed = PatchEmbed(
                config,
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.embed_dim,
            )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.layers = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(self.num_layers)])

        self.jit_trace = jit_trace
        self.main_device = 'cuda:0'
        # FIXME: This attribute needs to be taken care of during deployment,
        # One solution is to convert a separate graph to do the torch.repeat_interleave and cumsum operation
        # and offload this graph to CPU.
        self.cu_seqlens = CumulativeSeqenceLens()

        self._vision_attention_mask = 1
        self._vision_rot_emb = 1

        self.image_grid_thw = None

    def get_dtype(self) -> torch.dtype:
        """Get the data type of the model.

        Returns:
            torch.dtype: The data type of the model.
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        """Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return self.blocks[0].mlp.fc2.weight.device

    @property
    def attention_mask(self):
        """Get Qwen2-VL ViT vision attention mask."""
        return self._vision_attention_mask

    @attention_mask.setter
    def attention_mask(self, attn_mask):
        self._vision_attention_mask = attn_mask

    @property
    def rot_emb(self):
        """Get Qwen2-VL ViT vision rotary embedding."""
        return self._vision_rot_emb

    @rot_emb.setter
    def rot_emb(self, r):
        self._vision_rot_emb = r

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {}
        if self.chunk_idx == 0:
            if self.config.use_conv2d_patch_embed:
                # Need to handle conv3d weight slicing later
                state_dict_mapping = {
                    'patch_embed_conv2d1_weight': {'patch_embed.proj.conv2d1.weight': 'visual.patch_embed.proj.weight'},
                    'patch_embed_conv2d2_weight': {'patch_embed.proj.conv2d2.weight': 'visual.patch_embed.proj.weight'},
                }
            else:
                state_dict_mapping = {
                    'patch_embed_weight': {'patch_embed.proj.weight': 'visual.patch_embed.proj.weight'}
                }
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}.attn.proj_bias': {
                        f'layers.{inner_layer_idx}.attn.proj.bias': f'visual.blocks.{outer_layer_idx}.attn.proj.bias'
                    },
                    f'{outer_layer_idx}.attn.proj_weight': {
                        f'layers.{inner_layer_idx}.attn.proj.weight': f'visual.blocks.{outer_layer_idx}.attn.proj.weight'  # noqa: E501
                    },
                    f'{outer_layer_idx}.attn.qkv_bias': {
                        f'layers.{inner_layer_idx}.attn.qkv.bias': f'visual.blocks.{outer_layer_idx}.attn.qkv.bias'
                    },
                    f'{outer_layer_idx}.attn.qkv_weight': {
                        f'layers.{inner_layer_idx}.attn.qkv.weight': f'visual.blocks.{outer_layer_idx}.attn.qkv.weight'
                    },
                    f'{outer_layer_idx}.mlp.fc1_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc1.bias': f'visual.blocks.{outer_layer_idx}.mlp.fc1.bias'
                    },
                    f'{outer_layer_idx}.mlp.fc1_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc1.weight': f'visual.blocks.{outer_layer_idx}.mlp.fc1.weight'
                    },
                    f'{outer_layer_idx}.mlp.fc2_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc2.bias': f'visual.blocks.{outer_layer_idx}.mlp.fc2.bias'
                    },
                    f'{outer_layer_idx}.mlp.fc2_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc2.weight': f'visual.blocks.{outer_layer_idx}.mlp.fc2.weight'
                    },
                    f'{outer_layer_idx}.norm1_bias': {
                        f'layers.{inner_layer_idx}.norm1.bias': f'visual.blocks.{outer_layer_idx}.norm1.bias'
                    },
                    f'{outer_layer_idx}.norm1_weight': {
                        f'layers.{inner_layer_idx}.norm1.weight': f'visual.blocks.{outer_layer_idx}.norm1.weight'
                    },
                    f'{outer_layer_idx}.norm2_bias': {
                        f'layers.{inner_layer_idx}.norm2.bias': f'visual.blocks.{outer_layer_idx}.norm2.bias'
                    },
                    f'{outer_layer_idx}.norm2_weight': {
                        f'layers.{inner_layer_idx}.norm2.weight': f'visual.blocks.{outer_layer_idx}.norm2.weight'
                    },
                }
            )
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())

        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter Qwen2VLVisionModel load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

        # state_dict = self._pop_text_model_weights(state_dict)

        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        # Handle use_conv2d_patch_embed
        if self.chunk_idx == 0 and self.config.use_conv2d_patch_embed:
            conv3dweight = state_dict.pop('visual.patch_embed.proj.weight')
            conv2d1 = conv3dweight[:, :, 0, :, :].contiguous()
            conv2d2 = conv3dweight[:, :, 1, :, :].contiguous()
            weights_to_load.update(
                {
                    'patch_embed.proj.conv2d1.weight': conv2d1.to(torch.float32),
                    'patch_embed.proj.conv2d2.weight': conv2d2.to(torch.float32),
                }
            )
            self.state_dict_mapping.pop('patch_embed_conv2d1_weight')
            self.state_dict_mapping.pop('patch_embed_conv2d2_weight')

        for internal_key, mapping_dict in self.state_dict_mapping.items():
            logger.debug(f'internal_key={internal_key}, mapping_dict={mapping_dict}')
            found = False
            # Ensure that weight exist in state
            # mapping_dict values should be a dict with exactly length 1
            if not isinstance(mapping_dict, dict):
                logger.error(f'Expected dict for mapping_dict but got {type(mapping_dict)}', err=TypeError)
            if len(mapping_dict) != 1:
                logger.error(f'Expected exactly 1 key-value pair in mapping_dict but got {len(mapping_dict)}')
            model_key = next(iter(mapping_dict))
            external_key = mapping_dict[model_key]

            dtype = self.dtype if internal_key.split('_')[-2] != 'norm' else torch.float32

            # Check if key with all found prefixes directly matches the state_dict key
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(
                        f'Found {internal_key} weight using prefix, state_dict key={key_to_test}, dtype={dtype}'
                    )
                    weights_to_load.update({model_key: state_dict.pop(key_to_test).to(dtype)})
                    found = True
                    state_dict_keys.remove(key_to_test)
                    break

            if not found:
                for k in state_dict_keys:
                    if k.endswith(external_key):
                        logger.debug(
                            f'Found {internal_key} weight using iteration, state_dict key={k}, dtype={dtype}. '
                            f'Adding prefix: {k[: -len(external_key)]}'
                        )
                        self.prefixes.append(k[: -len(external_key)])
                        weights_to_load.update({model_key: state_dict.pop(k).to(dtype)})
                        found = True
                        state_dict_keys.remove(k)
                        break

            if not found and internal_key.endswith('_bias'):
                # Bias key not found, default to zeros
                # For shape, check corresponding weight shape in both state_dict and weights_to_load
                weight_internal_key = internal_key.replace('_bias', '_weight')
                weight_model_key = next(iter(self.state_dict_mapping[weight_internal_key]))
                shape = None
                if weight_model_key in weights_to_load:
                    logger.debug(
                        f'Init bias {internal_key} to zeros using shape={shape} found from {weight_internal_key} '
                        f'weight shape and dtype={dtype}. Found from weights_to_load.'
                    )
                    weight_shape = weights_to_load[weight_model_key].shape
                    shape = weight_shape[0]
                else:
                    # Could be that the corresponding weight has not been loaded into weights_to_load
                    weight_external_key = external_key.replace('_bias', '_weight')
                    for pre in self.prefixes:
                        key_to_test = pre + weight_external_key
                        if key_to_test in state_dict_keys:
                            logger.debug(
                                f'Init bias {internal_key} to zeros using shape={shape} found from '
                                f'{weight_internal_key} weight shape and dtype={dtype}. Found from state_dict.'
                            )
                            weight_shape = state_dict[key_to_test].shape
                            shape = weight_shape[0]
                            break
                if shape is None:
                    logger.error(
                        'Unable to find weight in both weights_to_load and state_dict associated with bias: '
                        f'{internal_key}'
                    )
                weights_to_load.update({model_key: torch.zeros(shape, dtype=dtype)})
                continue

            if not found:
                logger.debug(f'Cannot find {internal_key} weight')
                missing_keys.append((internal_key, external_key))

        if len(missing_keys) > 0:
            for internal_key, external_key in missing_keys:
                logger.warning(f'Unable to find {internal_key} weight in state_dict. Expected subkey: {external_key}')
            logger.info(f'state dict keys for reference: {state_dict_keys}')
            logger.error('Please modify your state_dict keys according to the expected subkeys.', err=KeyError)

        if len(self.prefixes) > 2:
            logger.warning(
                f'More than 1 prefix found (found {self.prefixes[1:]}). '
                'This is unexpected and will likely cause errors during weight loading.'
            )

        num_gpu = torch.cuda.device_count()
        if num_gpu == 0 or self.jit_trace:
            self.device_list = ['cpu' for _ in range(self.num_layers)]
        else:
            if self.distribute_layers:
                master_gpu_ids = sorted(
                    list(range(num_gpu)) * (self.config.num_hidden_layers // num_gpu)
                    + (
                        list(range(num_gpu))[: self.config.num_hidden_layers % num_gpu]
                        if self.config.num_hidden_layers % num_gpu != 0
                        else []
                    )
                )
            else:
                master_gpu_ids = [os.getenv('LOCAL_RANK', 0)] * self.config.num_hidden_layers
            self.device_list = [f'cuda:{x}' for x in master_gpu_ids][state_dict_start_idx:state_dict_end_idx]

        if weights_to_load.keys() != self.state_dict().keys():
            weights_to_load_only_keys = [x for x in weights_to_load if x not in self.state_dict()]
            model_only_keys = [x for x in self.state_dict() if x not in weights_to_load and 'lora' not in x]
            if self.parallel_lora:
                model_only_keys = [
                    x for x in model_only_keys if '_weight_quantizer' not in x and '_act_quantizer' not in x
                ]
            if model_only_keys != [] or weights_to_load_only_keys != []:
                logger.error(
                    f"model state dict keys don't match with state_dict to load into model.\n"
                    f'Model only keys:{model_only_keys}\nstate_dict only keys:{weights_to_load_only_keys}'
                )

        if quant_config is not None:
            logger.info(f'Quantizing chunk {self.chunk_idx} using quant config: {quant_config}')
            quant_config_chunk = int(quant_config.rsplit('_', 1)[-1].split('.json')[0])
            if quant_config_chunk != self.chunk_idx:
                logger.error(
                    f'chunk_idx={self.chunk_idx} but quant config used {quant_config} is for chunk {quant_config_chunk}'
                )
            quantize_handler = mtk_quantization.pytorch.QuantizeHandler()
            self = quantize_handler.prepare(self, quant_config)
            self._quantizer_dict = quantize_handler._quantizer_dict  # noqa: SLF001
            if not self.parallel_lora:
                with open(quant_config) as f:
                    data = f.read()
                quant_config_dict = json.loads(data)
                weight_targets = quant_config_dict['quantizer_targets']['constant_weights']
                for wgt in weight_targets:
                    weights_to_load = self._add_quantizer_weights(
                        state_dict,
                        weights_to_load,
                        wgt,
                        'weight',
                        prefix=self.prefixes[-1],
                    )
                act_targets = quant_config_dict['quantizer_targets']['activations']
                for act in act_targets:
                    weights_to_load = self._add_quantizer_weights(
                        state_dict,
                        weights_to_load,
                        act,
                        'activation',
                        prefix=self.prefixes[-1],
                    )

        self.load_state_dict(weights_to_load, strict=False)

        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        if self.first_layer_idx == 0:
            self.patch_embed.to(self.device_list[0])
            # self.pre_layernorm.to(self.device_list[0])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict

    def rot_pos_emb(self, grid_thw):
        """Compute rotary positional embeddings.

        Args:
            grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).

        Returns:
            torch.Tensor: The computed rotary positional embeddings.
        """
        pos_ids = []
        for t, h, w in grid_thw:
            t = t.to(torch.int32).item()
            h = h.to(torch.int32).item()
            w = w.to(torch.int32).item()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def load_vision_weight(self, state_dict, prefix=''):
        """Load qwen2vl vision model weights.

        Args:
            state_dict (dict): The model state_dict.
            prefix (str): The prefix for the state dictionary keys. Defaults to ''.
        """
        temp_state_dict = {}
        mapping_keys_list = []
        # ViT part
        for layer_idx in range(self.vision_config.depth):
            temp_state_dict = {
                **temp_state_dict,
                f'blocks.{layer_idx}.attn.proj.bias': state_dict.pop(f'visual.blocks.{layer_idx}.attn.proj.bias'),
                f'blocks.{layer_idx}.attn.proj.weight': state_dict.pop(f'visual.blocks.{layer_idx}.attn.proj.weight'),
                f'blocks.{layer_idx}.attn.qkv.bias': state_dict.pop(f'visual.blocks.{layer_idx}.attn.qkv.bias'),
                f'blocks.{layer_idx}.attn.qkv.weight': state_dict.pop(f'visual.blocks.{layer_idx}.attn.qkv.weight'),
                f'blocks.{layer_idx}.mlp.fc1.bias': state_dict.pop(f'visual.blocks.{layer_idx}.mlp.fc1.bias'),
                f'blocks.{layer_idx}.mlp.fc1.weight': state_dict.pop(f'visual.blocks.{layer_idx}.mlp.fc1.weight'),
                f'blocks.{layer_idx}.mlp.fc2.bias': state_dict.pop(f'visual.blocks.{layer_idx}.mlp.fc2.bias'),
                f'blocks.{layer_idx}.mlp.fc2.weight': state_dict.pop(f'visual.blocks.{layer_idx}.mlp.fc2.weight'),
                f'blocks.{layer_idx}.norm1.bias': state_dict.pop(f'visual.blocks.{layer_idx}.norm1.bias'),
                f'blocks.{layer_idx}.norm1.weight': state_dict.pop(f'visual.blocks.{layer_idx}.norm1.weight'),
                f'blocks.{layer_idx}.norm2.bias': state_dict.pop(f'visual.blocks.{layer_idx}.norm2.bias'),
                f'blocks.{layer_idx}.norm2.weight': state_dict.pop(f'visual.blocks.{layer_idx}.norm2.weight'),
            }
            mapping_keys_list = [
                *mapping_keys_list,
                f'visual.blocks.{layer_idx}.attn.proj.bias',
                f'visual.blocks.{layer_idx}.attn.proj.weight',
                f'visual.blocks.{layer_idx}.attn.qkv.bias',
                f'visual.blocks.{layer_idx}.attn.qkv.weight',
                f'visual.blocks.{layer_idx}.mlp.fc1.bias',
                f'visual.blocks.{layer_idx}.mlp.fc1.weight',
                f'visual.blocks.{layer_idx}.mlp.fc2.bias',
                f'visual.blocks.{layer_idx}.mlp.fc2.weight',
                f'visual.blocks.{layer_idx}.norm1.bias',
                f'visual.blocks.{layer_idx}.norm1.weight',
                f'visual.blocks.{layer_idx}.norm2.bias',
                f'visual.blocks.{layer_idx}.norm2.weight',
            ]
            if torch.cuda.device_count() == 0 or self.jit_trace:
                self.vision_device_list.append('cpu')
            else:
                device_id = layer_idx // (
                    self.vision_config.depth // torch.cuda.device_count()
                    + (self.vision_config.depth % torch.cuda.device_count() != 0)
                )
                self.vision_device_list.append(f'cuda:{device_id}')

        # Merger (Projector) part
        temp_state_dict = {
            **temp_state_dict,
            'merger.ln_q.bias': state_dict.pop('visual.merger.ln_q.bias'),
            'merger.ln_q.weight': state_dict.pop('visual.merger.ln_q.weight'),
            'merger.mlp.0.bias': state_dict.pop('visual.merger.mlp.0.bias'),
            'merger.mlp.0.weight': state_dict.pop('visual.merger.mlp.0.weight'),
            'merger.mlp.2.bias': state_dict.pop('visual.merger.mlp.2.bias'),
            'merger.mlp.2.weight': state_dict.pop('visual.merger.mlp.2.weight'),
        }
        mapping_keys_list = [
            *mapping_keys_list,
            'visual.merger.ln_q.bias',
            'visual.merger.ln_q.weight',
            'visual.merger.mlp.0.bias',
            'visual.merger.mlp.0.weight',
            'visual.merger.mlp.2.bias',
            'visual.merger.mlp.2.weight',
        ]

        # Patch embedding part

        temp_state_dict = {
            **temp_state_dict,
            'patch_embed.proj.weight': state_dict.pop('visual.patch_embed.proj.weight'),
        }
        mapping_keys_list = [*mapping_keys_list, 'visual.patch_embed.proj.weight']

        try:
            self.load_state_dict(temp_state_dict, stricg=True)
            self.eval()
            return True, self
        except RuntimeError:
            return False, mapping_keys_list

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for the vision model.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): The attention mask.
            rotary_pos_emb (torch.Tensor): The rotary positional embedding.
            kwargs: Other kwargs

        Returns:
            torch.Tensor: The output tensor after applying the vision model.
        """
        attn_mask = kwargs.pop('qwen2_vl_attn_mask', None)
        rotary_pos_emb = kwargs.pop('qwen2_vl_rotary_pos_emb', None)
        if attn_mask is None:
            attn_mask = self._vision_attention_mask
            logger.debug('Vision attention mask is not passed as argument, using the pre-set attribute instead.')
        if rotary_pos_emb is None:
            rotary_pos_emb = self._vision_rot_emb
            logger.debug('Vision rotary embedding is not passed as argument, using the pre-set attribute instead.')
        if attn_mask is None:
            logger.error(
                'qwen2_vl_attn_mask must be either passed or pre-set when forwarding dynamic shape Qwen2-VL ViT.',
                err=ValueError,
            )
        if rotary_pos_emb is None:
            logger.error(
                'qwen2_vl_rotary_pos_emb must be passed or pre-set when forwarding dynamic shape Qwen2-VL ViT.',
                err=ValueError,
            )

        hidden_states = hidden_states.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.patch_embed(hidden_states)

        attn_mask = attn_mask.to(device=self.device_list[0])
        rotary_pos_emb = rotary_pos_emb.to(self.device_list[0])
        hidden_states = hidden_states.to(self.device_list[0])

        for idx, blk in enumerate(self.layers):
            hidden_states = blk(
                hidden_states.to(self.device_list[idx]),
                attn_mask=attn_mask.to(self.device_list[idx]),
                rotary_pos_emb=rotary_pos_emb.to(self.device_list[idx]),
            )

        return hidden_states

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        self._calculate_ptq_fixed_shape_batch()
        self._calulate_ptq_attnmask_rotemb()
        fixed_batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        if self.chunk_idx == 0:
            return torch.randn(
                fixed_batch_size,
                self.config.patch_size
                * self.config.patch_size
                * self.config.in_channels
                * self.config.temporal_patch_size,
                device='cpu',
                dtype=torch.float32,
            )
        feature_size = int(self.config.embed_dim)
        return torch.randn(fixed_batch_size, feature_size, device='cpu', dtype=torch.float32)

    def _calculate_ptq_fixed_shape_batch(self):
        from ..preprocessors.configuration_qwen2vl_vision import Qwen2VLPreprocessorConfig
        from ..preprocessors.preprocessor_qwen2vl_vision import Qwen2VLImageProcessor

        logger.debug('Enter Qwen2-VL _calculate_ptq_fixed_shape_batch.')
        preprocessor_config = Qwen2VLPreprocessorConfig(**self.config.preprocessor_config)
        processor = Qwen2VLImageProcessor(**preprocessor_config.get())
        image_hw = self.config.image_resolution
        if image_hw is None:
            logger.error('image_resolution in config must be set when PTQing Qwen2-VL ViT.', err=ValueError)
        image = np.random.rand(image_hw[0], image_hw[1], 3)
        image = Image.fromarray(image.astype('uint8'))
        return_dict, _ = processor.preprocess([image])
        self.image_grid_thw = torch.tensor(return_dict['image_grid_thw'])
        logger.debug(f'Qwen2-VL PTQ image_grid_thw: {self.image_grid_thw}')

    def _calulate_ptq_attnmask_rotemb(self):
        from ..hooks.qwen2_vl_pre_encoder import CumulativeSeqenceLens

        logger.debug('Enter Qwen2-VL _calulate_ptq_attnmask_rotemb.')
        cuseqlen = CumulativeSeqenceLens()
        cu_seqlens = cuseqlen(self.image_grid_thw)
        batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        self._vision_attention_mask = precompute_qwen2vl_vision_attn_mask(seq_length=batch_size, cu_seqlens=cu_seqlens)
        self._vision_rot_emb = precompute_qwen2vl_vision_rot_emb(self.image_grid_thw, self.config)

    def get_ptq_inputs(self, args, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            kwargs: Additional keyword arguments.

        Returns:
            tuple: Tuple containing input shapes, input value ranges, calibration data generator,
            and evaluation data generator.
        """
        fixed_batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        flatten_patch_size = (
            self.config.patch_size * self.config.patch_size * self.config.in_channels * self.config.temporal_patch_size
        )
        if self.chunk_idx == 0:
            input_shapes = [[fixed_batch_size, flatten_patch_size]]
        else:
            feature_size = int(self.config.embed_dim)
            input_shapes = [[fixed_batch_size, feature_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [np.random.rand(fixed_batch_size, flatten_patch_size).astype(np.float32)]
                    else:
                        yield [np.random.rand(fixed_batch_size, feature_size).astype(np.float32)]
        else:

            def calib_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.calibration_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [np.random.rand(fixed_batch_size, flatten_patch_size).astype(np.float32)]
                    else:
                        yield [np.random.rand(fixed_batch_size, feature_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    # FIXME: (Andy): Remove this method after verification
    def forward_fixed_shape(self, hidden_states):
        """Temp. Remove this method after verification."""
        hidden_states = hidden_states.to(self.patch_embed.proj.weight.device)
        if not self.jit_trace:
            hidden_states = self.patch_embed(hidden_states)

        hidden_states = hidden_states.to(self.blocks[0].attn.qkv.weight.device)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, attn_mask=self._vision_attention_mask, rotary_pos_emb=self._vision_rot_emb
            )

        return self.merger(hidden_states)
