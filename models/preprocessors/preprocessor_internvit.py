# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""InternViT Preprocessor class."""

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_utils import BatchFeature

from ...utils import logger


class InternViTPreprocessor:
    """Class for InternViTPreprocessor.

    This class provides methods for preprocessing images for the InternViT model.

    Attributes:
        input_size (int): The target size for resizing the image.
        max_num (int): The maximum number of dynamic patches.
        min_num (int): The minimum number of dynamic patches.

    Methods:
        internvl2_build_transform: Builds a transformation pipeline for image preprocessing.
        internvl2_find_closest_aspect_ratio: Finds the closest aspect ratio from a list of target ratios.
        internvl2_dynamic_preprocess: Dynamically preprocesses an image by resizing and splitting it into blocks.
        preprocess: Preprocesses an image file.
    """

    def __init__(self, force_image_size, max_dynamic_patch, min_dynamic_patch, use_thumbnail, **kwargs):
        """Initialize the InternViTPreprocessor.

        Args:
            force_image_size: The processed image size.
            max_dynamic_patch: Maximum number of image patch.
            min_dynamic_patch: Minimum number of image patch.
            use_thumbnail: Whether to use thumbnail (global view) image or not.
            kwargs: Other kwargs.
        """
        self.input_size = force_image_size
        self.max_num = max_dynamic_patch
        self.min_num = min_dynamic_patch
        self.use_thumbnail = use_thumbnail

    def internvl2_build_transform(self, input_size):
        """Builds a transformation pipeline for image preprocessing.

        Args:
            input_size (int): The target size for resizing the image.

        Returns:
            torchvision.transforms.Compose: The composed transformation pipeline.
        """
        from ...utils.const import IMAGENET_MEAN, IMAGENET_STD

        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def internvl2_find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Finds the closest aspect ratio from a list of target ratios.

        Args:
            aspect_ratio (float): The aspect ratio of the original image.
            target_ratios (list): A list of target aspect ratios.
            width (int): The width of the original image.
            height (int): The height of the original image.
            image_size (int): The target image size.

        Returns:
            tuple: The closest aspect ratio as a tuple (width_ratio, height_ratio).
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def internvl2_dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamically preprocesses an image by resizing and splitting it into blocks.

        Args:
            image (PIL.Image.Image): The input image.
            min_num (int, optional): The minimum number of blocks. Default is 1.
            max_num (int, optional): The maximum number of blocks. Default is 12.
            image_size (int, optional): The target size for resizing the image. Default is 448.
            use_thumbnail (bool, optional): Whether to use a thumbnail image. Default is False.

        Returns:
            list: A list of processed image blocks.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.internvl2_find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def preprocess(self, image, input_size=None, max_num=None, min_num=None, **kwargs):
        """Preprocesses an image file.

        Args:
            image (PIL.Image): The input image.
            input_size (int, optional): The target size for resizing the image. Default is 448.
            max_num (int, optional): The maximum number of blocks. Default is None and will use value from attributes.
            min_num (int, optional): The maximum number of blocks. Default is None and will use value from attributes.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor: A tensor containing the preprocessed image blocks.
        """
        logger.debug('Enter CLIP preprocessor forward')
        if input_size is None:
            input_size = self.input_size
            logger.debug(f'InternVL2 input_size is not provided, use default {self.input_size}')
        if max_num is None:
            max_num = self.max_num
            logger.debug(f'InternVL2 max_num is not provided, use default {self.max_num}')
        if min_num is None:
            min_num = self.min_num
            logger.debug(f'InternVL2 max_num is not provided, use default {self.min_num}')

        image = image.convert('RGB')

        logger.debug('Build InternVL2 transform.')
        transform = self.internvl2_build_transform(input_size=input_size)
        logger.debug('Do InternVL2 dynamic preprocess.')
        images = self.internvl2_dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=self.use_thumbnail, max_num=max_num, min_num=min_num
        )
        pixel_values = [transform(image) for image in images]
        data = {'input_features': torch.stack(pixel_values).permute(0, 2, 3, 1)}

        kwargs.update({'internvl2_num_images': data['input_features'].shape[0], 'internvl2_max_patch': max_num})

        return BatchFeature(data=data), kwargs
