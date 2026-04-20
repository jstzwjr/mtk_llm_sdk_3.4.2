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
"""InternVLNavitRope Preprocessor class."""

import torch
import torchvision.transforms as T  # noqa: N812
from qwen_vl_utils.vision_process import fetch_image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

from ...utils import const, logger


class InternVLNavitRopePreprocessor(BaseImageProcessor):
    """Class for InternVLNavitRopePreprocessor.

    This class provides methods for preprocessing images for the InternVL-navit-rope model.
    """

    def __init__(self, image_resolution=None, patch_size=None, image_mean=None, image_std=None, **kwargs) -> None:
        """Initialize InternVLNavitRopePreprocessor."""
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        if self.patch_size is None:
            logger.error('patch_size must be set when using InternVLNavitRopePreprocessor')
        self.image_mean = image_mean if image_mean is not None else const.OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else const.OPENAI_CLIP_STD
        self.max_pixels = kwargs.get('max_pixels', 1280 * 28 * 28)
        self.min_pixels = kwargs.get('min_pixels', 4 * 28 * 28)

    def get_flated_pixel_values(self, pixel_values):
        """Get flatten pixel values."""
        flated_pixel_values = []
        image_grid_hw = []
        for pv in pixel_values:
            c, h, w = pv.shape
            assert c == 3 and h % self.patch_size == 0 and w % self.patch_size == 0, f'{c}, {w}, {h}, {self.patch_size}'
            image_grid_hw.append((h // self.patch_size, w // self.patch_size))
            fpv = pv.reshape(
                c, h // (2 * self.patch_size), 2, self.patch_size, w // (2 * self.patch_size), 2, self.patch_size
            )
            flated_pixel_values.append(
                fpv.permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, c * self.patch_size * self.patch_size)
            )
        flated_pixel_values = torch.cat(flated_pixel_values, dim=0)  # (Len_img, C, H, W)
        image_grid_hw = torch.tensor(image_grid_hw, device=flated_pixel_values.device)  # (N_img, 2)
        return flated_pixel_values, image_grid_hw

    def preprocess(self, images, **kwargs):
        """Do the actual preprocess."""
        base = self.patch_size * 2  # Hard-coded
        transformations = [T.ToTensor()]
        if self.image_resolution is not None:
            transformations.append(T.Resize(self.image_resolution))
        transformations.append(T.Normalize(mean=self.image_mean, std=self.image_std))
        transform = T.Compose(transformations)

        pixel_values = []
        image_tokens = []

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            images = [
                fetch_image({'image': item, 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels})
                for item in images
            ]
            for image in images:
                image = transform(image)
                pixel_values.append(image)
                image_tokens.append(image.shape[1] * image.shape[2] // (base * base))

        image_inputs = {}
        if image_tokens:
            pixel_values, image_grid_hw = self.get_flated_pixel_values(pixel_values)
            image_inputs = {'input_features': pixel_values, 'image_grid_hw': image_grid_hw}
        kwargs.update({'AndesVL_image_tokens': image_tokens, 'Andes_image_grid_hw': image_grid_hw})

        return BatchFeature(data={**image_inputs}), kwargs
