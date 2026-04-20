# Copyright 2022 The HuggingFace Inc. team. All rights reserved.  # noqa: CPY001
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
"""Preprocessor classes."""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import (
    center_crop,
    convert_to_rgb,
    get_resize_output_image_size,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)

from ...utils import const, image_utils, logger


class CLIPImageProcessor(BaseImageProcessor):
    """Define CLIP image preprocessor."""

    model_input_names = ['pixel_values']

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample=image_utils.PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the CLIPImageProcessor.

        Args:
            do_resize (bool, optional): Whether to resize the image. Defaults to True.
            size (Optional[Dict[str, int]], optional): The size to resize the image to. Defaults to None.
            resample (image_utils.PILImageResampling, optional): The resampling method to use. Defaults to BICUBIC.
            do_center_crop (bool, optional): Whether to center crop the image. Defaults to True.
            crop_size (Optional[Dict[str, int]], optional): The size to crop the image to. Defaults to None.
            do_rescale (bool, optional): Whether to rescale the image. Defaults to True.
            rescale_factor (Union[int, float], optional): The factor to rescale the image by. Defaults to 1/255.
            do_normalize (bool, optional): Whether to normalize the image. Defaults to True.
            image_mean (Optional[Union[float, List[float]]], optional): The mean to use for normalization.
                Defaults to None.
            image_std (Optional[Union[float, List[float]]], optional): The standard deviation to use for normalization.
                Defaults to None.
            do_convert_rgb (bool, optional): Whether to convert the image to RGB. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        size = size
        crop_size = crop_size
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else const.OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else const.OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample=image_utils.PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, image_utils.ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize an image.

        The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample: (`image_utils.PILImageResampling`, *optional*,
                defaults to `image_utils.PILImageResampling.BICUBIC`) Resampling filter to use when resiizing
                the image.
            data_format (`str` or `image_utils.ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            **kwargs: Other kwargs.
        """
        if 'shortest_edge' not in size:
            logger.error(
                f'The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}', err=ValueError
            )
        output_size = get_resize_output_image_size(image, size=size['shortest_edge'], default_to_square=False)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, image_utils.ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Center crop an image.

        If the image is too small to be cropped to the size given, it will be padded (so the
        returned result will always be of size `size`).

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image in the form of a dictionary with keys `height` and `width`.
            data_format (`str` or `image_utils.ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            **kwargs: Other kwargs.
        """
        if 'height' not in size or 'width' not in size:
            logger.error(
                f'The `size` parameter must contain the keys (height, width). Got {size.keys()}', err=ValueError
            )
        return center_crop(image, size=(size['height'], size['width']), data_format=data_format, **kwargs)

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, image_utils.ChannelDimension]] = None,
        **kwargs,
    ):
        """Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `image_utils.ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            **kwargs: Other kwargs.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, image_utils.ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean.
            std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `image_utils.ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            **kwargs: Other kwargs.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: image_utils.ImageInput,
        do_resize: Optional[bool] = None,
        size=None,
        resample=None,
        do_center_crop: Optional[bool] = None,
        crop_size=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[image_utils.ChannelDimension] = image_utils.ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """Preprocess an image or batch of images.

        Args:
            images (`image_utils.ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum
                `image_utils.PILImageResampling`. Only has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
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
            return_tensors (`str` or `image_utils.TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `image_utils.TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `image_utils.TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `image_utils.TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `image_utils.TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format: (`image_utils.ChannelDimension` or `str`, *optional*, defaults to
                `image_utils.ChannelDimension.FIRST`)
                The channel dimension format for the output image. Can be one of:
                - `image_utils.ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `image_utils.ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: defaults to the channel dimension format of the input image.
            **kwargs: Other kwargs.
        """
        logger.debug('Enter CLIP preprocessor forward')
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        if isinstance(size, int):
            size = {'shortest_edge': size}
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        if isinstance(crop_size, int):
            crop_size = {'height': crop_size, 'width': crop_size}
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = image_utils.make_list_of_images(images)

        if not image_utils.valid_images(images):
            raise ValueError(
                'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, '
                'torch.Tensor, tf.Tensor or jax.ndarray.'
            )

        if do_resize and size is None:
            logger.error('Size must be specified if do_resize is True.', err=ValueError)

        if do_center_crop and crop_size is None:
            logger.error('Crop size must be specified if do_center_crop is True.', err=ValueError)

        if do_rescale and rescale_factor is None:
            logger.error('Rescale factor must be specified if do_rescale is True.', err=ValueError)

        if do_normalize and (image_mean is None or image_std is None):
            logger.error('Image mean and std must be specified if do_normalize is True.', err=ValueError)

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            logger.debug('[CLIP Preprocessor] Convert RGB')
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [image_utils.to_numpy_array(image) for image in images]

        if do_resize:
            logger.debug('[CLIP Preprocessor] Resize')
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_center_crop:
            logger.debug('[CLIP Preprocessor] Center crop')
            images = [self.center_crop(image=image, size=crop_size) for image in images]

        if do_rescale:
            logger.debug('[CLIP Preprocessor] Rescale')
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            logger.debug('[CLIP Preprocessor] Normalize')
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        logger.debug(f'[CLIP Preprocessor] To data_format={data_format}')
        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {'input_features': images}
        return BatchFeature(data=data), kwargs
