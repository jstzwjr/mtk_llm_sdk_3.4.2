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
"""Qwen2 VL Preprocessor Configuration."""

from transformers.image_utils import PILImageResampling

from ...utils import logger
from ..configuration_base import BasePreprocessorConfig


class Qwen2VLPreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the Qwen2 VL Preprocessor model.

    This class is used to store the configuration of a Qwen2 VL Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the Qwen2VLPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'qwen2_vl'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'qwen2_vl')
        if self.model_type != 'qwen2_vl':
            logger.error(f'Expected model_type to be qwen2_vl but got {self.model_type} instead')

        self.do_resize = self.kwargs.get('do_resize', True)
        self.resample = self.kwargs.get('resample', PILImageResampling.BICUBIC)
        self.do_rescale = self.kwargs.get('do_rescale', True)
        self.rescale_factor = self.kwargs.get('rescale_factor', 1 / 255)
        self.do_normalize = self.kwargs.get('do_normalize', True)
        self.image_mean = self.kwargs.get('image_mean', None)
        self.image_std = self.kwargs.get('image_std', None)
        self.do_convert_rgb = self.kwargs.get('do_convert_rgb', True)
        self.min_pixels = self.kwargs.get('min_pixels', 56 * 56)
        self.max_pixels = self.kwargs.get('max_pixels', 28 * 28 * 1280)
        self.patch_size = self.kwargs.get('patch_size', 14)
        self.temporal_patch_size = self.kwargs.get('temporal_patch_size', 2)
        self.merge_size = self.kwargs.get('merge_size', 2)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'do_resize:           {self.do_resize}')
        logger.info(f'resample:            {self.resample}')
        logger.info(f'do_rescale:          {self.do_rescale}')
        logger.info(f'rescale_factor:      {self.rescale_factor}')
        logger.info(f'do_normalize:        {self.do_normalize}')
        logger.info(f'image_mean:          {self.image_mean}')
        logger.info(f'image_std:           {self.image_std}')
        logger.info(f'do_convert_rgb:      {self.do_convert_rgb}')
        logger.info(f'min_pixels:          {self.min_pixels}')
        logger.info(f'max_pixels:          {self.max_pixels}')
        logger.info(f'patch_size:          {self.patch_size}')
        logger.info(f'temporal_patch_size: {self.temporal_patch_size}')
        logger.info(f'merge_size:          {self.merge_size}')
