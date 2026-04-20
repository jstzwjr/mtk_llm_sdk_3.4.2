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


class InternVLNavitRopePreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the InternVLNavitRope Preprocessor model.

    This class is used to store the configuration of a InternVLNavitRope Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the InternVLNavitRopePreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'clip'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'intern_vit_6b_navit_rope')
        if self.model_type != 'intern_vit_6b_navit_rope':
            logger.error(f'Expected model_type to be intern_vit_6b_navit_rope but got {self.model_type} instead')
        self.min_pixels = self.kwargs.get('min_pixels', 4 * 28 * 28)
        self.max_pixels = self.kwargs.get('max_pixels', 1280 * 28 * 28)

        self.patch_size = self.kwargs.get('patch_size', 14)
        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'patch size:    {self.patch_size}')
        logger.info(f'min_pixels:    {self.min_pixels}')
        logger.info(f'max_pixels:    {self.max_pixels}')
