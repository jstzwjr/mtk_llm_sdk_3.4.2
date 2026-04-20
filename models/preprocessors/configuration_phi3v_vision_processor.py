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


class Phi3VPreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the Phi3V Preprocessor model.

    This class is used to store the configuration of a Phi3V Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the Phi3VPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'clip'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'phi3v')
        if self.model_type != 'phi3v':
            logger.error(f'Expected model_type to be phi3v but got {self.model_type} instead')

        self.num_crops = self.kwargs.get('num_crops', 16)
        self.image_mean = self.kwargs.get('image_mean', None)
        self.image_std = self.kwargs.get('image_std', None)
        self.do_convert_rgb = self.kwargs.get('do_convert_rgb', True)
        self.use_hd_transform = self.kwargs.get('use_hd_transform', True)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'Number of crop:   {self.num_crops}')
        logger.info(f'Image mean:       {self.image_mean}')
        logger.info(f'Image std:        {self.image_std}')
        logger.info(f'Do convert RGB:   {self.do_convert_rgb}')
        logger.info(f'Use HD Transform: {self.use_hd_transform}')
