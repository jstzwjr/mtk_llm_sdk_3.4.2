# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# ==================================================================================================
"""Define vision-related helper functions for mtk_llm_sdk."""

import ast
import math
from enum import Enum
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms
from packaging import version
from transformers.image_processing_utils import BatchFeature

from . import logger

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse('9.1.0'):
    PILImageResampling = PIL.Image.Resampling
else:
    PILImageResampling = PIL.Image

ImageInput = Union[
    'PIL.Image.Image',
    np.ndarray,
    'torch.Tensor',
    List['PIL.Image.Image'],
    List[np.ndarray],
    List['torch.Tensor'],
]


class ExplicitEnum(str, Enum):
    """Enum with more explicit error message for missing values."""

    @classmethod
    def _missing_(cls, value):
        logger.error(
            f'{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}',
            err=ValueError,
        )


class ChannelDimension(ExplicitEnum):
    """Enum for specifying the channel dimension in image data."""

    FIRST = 'channels_first'
    LAST = 'channels_last'


class TensorType(ExplicitEnum):
    """Enum for specifying the type of tensor."""

    PYTORCH = 'pt'
    TENSORFLOW = 'tf'
    NUMPY = 'np'
    JAX = 'jax'


def NHWCWrapper(func):  # noqa: N802
    """Decorator to convert inputs from NHWC to NCHW format and outputs back to NHWC format.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function.
    """

    def wrapper(*inputs):
        # NHWC -> NCHW
        inputs_nchw = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, (np.ndarray, torch.Tensor)) and inp.ndim == 4:
                if isinstance(inp, np.ndarray):
                    inputs_nchw.append(inputs[i].transpose(0, 3, 1, 2))
                elif isinstance(inp, torch.Tensor):
                    inputs_nchw.append(inputs[i].permute(0, 3, 1, 2))
                continue
            inputs_nchw.append(inputs[i])
        outputs = func(*inputs_nchw)

        if not isinstance(outputs, (list, tuple)):
            if isinstance(outputs, (np.ndarray, torch.Tensor)) and outputs.ndim == 4:
                if isinstance(outputs, np.ndarray):
                    return outputs.transpose(0, 2, 3, 1)
                return outputs.permute(0, 2, 3, 1)
            return outputs
        # NCHW -> NHWC
        outputs_nhwc = []
        for i, oup in enumerate(outputs):
            if isinstance(oup, (np.ndarray, torch.Tensor)) and oup.ndim == 4:
                if isinstance(oup, np.ndarray):
                    outputs_nhwc.append(outputs[i].transpose(0, 2, 3, 1))
                elif isinstance(oup, torch.Tensor):
                    outputs_nhwc.append(outputs[i].permute(0, 2, 3, 1))
                continue
            outputs_nhwc.append(outputs[i])

        return outputs_nhwc

    return wrapper


def _is_batched(img):
    """Checks if the input is a batch of images.

    Args:
        img (list or tuple): The input to check.

    Returns:
        bool: True if the input is a batch of images, False otherwise.
    """
    if isinstance(img, (list, tuple)):
        return _is_valid_image(img[0])
    return False


def _is_valid_image(img):
    """Checks if the input is a valid image.

    Args:
        img (PIL.Image.Image, np.ndarray, torch.Tensor): The input to check.

    Returns:
        bool: True if the input is a valid image, False otherwise.
    """
    return isinstance(img, (PIL.Image.Image, np.ndarray, torch.Tensor))


def calc_hd_transform_size(width, height, hd_num=16):
    """Calculate image size of Phi3-V HD transform."""
    transposed = False
    if width < height:
        width, height = height, width
        transposed = True

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1

    new_width = int(scale * 336)
    new_height = int(new_width / ratio)

    padded_width, padded_height = calc_padded_size(new_width, new_height)

    if transposed:
        padded_width, padded_height = padded_height, padded_width

    return padded_width, padded_height


def calc_padded_size(width, height, padding_unit=336):
    """Calculate padding size of Phi3-V image preprocessing."""
    target_height = int(np.ceil(height / padding_unit) * padding_unit)
    top_padding = int((target_height - height) / 2)
    bottom_padding = target_height - height - top_padding
    left_padding = 0
    right_padding = 0
    padded_width = width + left_padding + right_padding
    padded_height = height + top_padding + bottom_padding

    return padded_width, padded_height


def divide_to_patches(image, patch_size):
    """Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def expand2square(pil_img, background_color):
    """Expands a PIL image to a square by adding padding.

    Args:
        pil_img (PIL.Image.Image): The input image.
        background_color (tuple or int): The background color for padding.

    Returns:
        PIL.Image.Image: The squared image with padding.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = PIL.Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def HD_transform(img, hd_num=16):  # noqa: N802
    """Phi3-V HD transform."""
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(PIL.Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(PIL.Image.TRANSPOSE)

    return img


def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    """Converts the input into a list of images.

    Args:
        images (PIL.Image.Image, np.ndarray, torch.Tensor): The input images.
        expected_ndims (int, optional): The expected number of dimensions. Default is 3.

    Returns:
        list: A list of images.

    Raises:
        ValueError: If the input image shape or type is invalid.
    """
    if _is_batched(images):
        return images

    # Either the input is a single image, in which case we create a list of length 1
    if isinstance(images, PIL.Image.Image):
        # PIL images are never batched
        return [images]

    if _is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == expected_ndims:
            # Single image
            images = [images]
        else:
            logger.error(
                f'Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, '
                f'but got {images.ndim} dimensions.',
                err=ValueError,
            )
        return images
    logger.error(
        'Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or '
        f'jax.ndarray, but got {type(images)}.',
        err=ValueError,
    )
    raise


def padding_336(b):
    """Pad Phi3-V input image to 336."""
    _, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0

    return transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255]
    )


def pad_to_max_num_crops_tensor(images, max_crops=5):
    """For Phi3-V model.

    images: B x 3 x H x W, B<=max_crops
    """
    b, _, h, w = images.shape
    if b < max_crops:
        pad = torch.zeros(max_crops - b, 3, h, w, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images


def process_anyres_image(image, processor, grid_pinpoints, processor_cfg):
    """Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        processor_cfg: The image processor config.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    possible_resolutions = grid_pinpoints if type(grid_pinpoints) is list else ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor_cfg['crop_size'])

    image_original_resize = image.resize((processor_cfg['size'], processor_cfg['size']))

    image_patches = [image_original_resize, *patches]
    image_patches = [
        processor.preprocess(image_patch, return_tensors='pt', **processor_cfg)['pixel_values'][0]
        for image_patch in image_patches
    ]
    data = {'pixel_values': torch.stack(image_patches, dim=0)}
    return BatchFeature(data=data, tensor_type='pt')


def process_anyres_image_no_processor(image, processor, grid_pinpoints, processor_cfg):
    """Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        processor_cfg: The image processor config.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    possible_resolutions = grid_pinpoints if type(grid_pinpoints) is list else ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor_cfg['crop_size'])

    image_original_resize = image.resize((processor_cfg['size'], processor_cfg['size']))

    return [image_original_resize, *patches]


def resize_and_pad_image(image, target_resolution):
    """Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = PIL.Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def select_best_resolution(original_size, possible_resolutions):
    """Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format
        [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def to_numpy_array(img) -> np.ndarray:
    """Converts an image to a NumPy array.

    Args:
        img (PIL.Image.Image, np.ndarray, torch.Tensor): The input image.

    Returns:
        np.ndarray: The image as a NumPy array.

    Raises:
        ValueError: If the input image type is invalid.
    """
    if not _is_valid_image(img):
        logger.error(f'Invalid image type: {type(img)}', err=ValueError)

    if isinstance(img, PIL.Image.Image):
        return np.array(img)
    if isinstance(img, np.ndarray):
        return img
    return img.detach().cpu().numpy()


def valid_images(imgs):
    """Checks if the input is a valid image or a list/tuple of valid images.

    Args:
        imgs (PIL.Image.Image, np.ndarray, torch.Tensor, list, tuple): The input image(s).

    Returns:
        bool: True if the input is a valid image or a list/tuple of valid images, False otherwise.
    """
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # If not a list of tuple, we have been given a single image or batched tensor of images
    elif not _is_valid_image(imgs):
        return False
    return True
