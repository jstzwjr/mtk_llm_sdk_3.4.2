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
"""CLIP Preprocessor Configuration."""

from ...utils import logger
from ..configuration_base import BasePreprocessorConfig


class CLIPPreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the CLIP Preprocessor model.

    This class is used to store the configuration of a CLIP Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the CLIPPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'clip'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'clip')
        if self.model_type != 'clip':
            logger.error(f'Expected model_type to be clip but got {self.model_type} instead')

        self.do_center_crop = self.kwargs.get('do_center_crop', False)
        self.do_normalize = self.kwargs.get('do_normalize', False)
        self.do_resize = self.kwargs.get('do_resize', False)
        self.image_mean = self.kwargs.get('image_mean', None)
        self.image_std = self.kwargs.get('image_std', None)
        self.resample = self.kwargs.get('resample', None)
        self.crop_size = self.kwargs.get('crop_size', None)
        self.size = self.kwargs.get('size', None)
        self.data_format = self.kwargs.get('data_format', None)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'Center crop:   {self.do_center_crop}')
        logger.info(f'Normalize:     {self.do_normalize}')
        logger.info(f'Resize:        {self.do_resize}')
        logger.info(f'Crop size:     {self.crop_size}')
        logger.info(f'Image mean:    {self.image_mean}')
        logger.info(f'Image std:     {self.image_std}')
        logger.info(f'Resample mode: {self.resample}')
        logger.info(f'Resize size:   {self.size}')
        logger.info(f'Data format:   {self.data_format}')
