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
"""Siglip Preprocessor Configuration."""

from ...utils import logger
from .configuration_clip import CLIPPreprocessorConfig


class SiglipPreprocessorConfig(CLIPPreprocessorConfig):
    """Configuration class for the Siglip Preprocessor model.

    This class is used to store the configuration of a Siglip Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the SiglipPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'siglip'.
            KeyError: If required configuration parameters are missing.
        """
        model_type = kwargs.pop('model_type', 'siglip')
        if model_type != 'siglip':
            logger.error(f'Expected model_type to be siglip but got {model_type} instead')

        verbose = kwargs.pop('verbose', True)
        kwargs['model_type'] = 'clip'
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.model_type = model_type
        self.rescale_factor = self.kwargs.pop('rescale_factor', 1 / 255)

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        super().print_config()
        logger.info(f'rescale_factor:    {self.rescale_factor}')
