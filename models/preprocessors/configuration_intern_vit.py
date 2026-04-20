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
"""Intern VIT Preprocessor Configuration."""

from ...utils import logger
from ..configuration_base import BasePreprocessorConfig


class InternViTPreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the Intern VIT Preprocessor model.

    This class is used to store the configuration of a Intern VIT Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the InternViTPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'intern_vit_6b'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.get('model_type', 'intern_vit_6b')
        self.force_image_size = self.kwargs.get('force_image_size', 448)
        self.max_dynamic_patch = self.kwargs.get('max_dynamic_patch', 12)
        self.min_dynamic_patch = self.kwargs.get('min_dynamic_patch', 1)
        self.use_thumbnail = self.kwargs.get('use_thumbnail', True)
        if self.model_type != 'intern_vit_6b':
            logger.error(f'Expected model_type to be intern_vit_6b but got {self.model_type} instead')

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'Force image size:   {self.force_image_size}')
        logger.info(f'Max dynamic patch:  {self.max_dynamic_patch}')
        logger.info(f'Min dynamic patch:  {self.min_dynamic_patch}')
        logger.info(f'Use thumbnail:      {self.use_thumbnail}')
