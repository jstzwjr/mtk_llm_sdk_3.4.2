# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""InternVL2 vision projector Configuration."""

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class InternVL2ProjectorConfig(BaseProjectorConfig):
    """Configuration class for the InternVL2 vision projector model.

    This class is used to store the configuration of an InternVL2 vision projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the InternVL2ProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'internvl2')
        if self.model_type != 'internvl2':
            logger.error(f'Expected model_type to be internvl2 but got {self.model_type} instead', err=RuntimeError)

        self.encoder_hidden_size = self.kwargs.get('encoder_hidden_size', None)
        if self.encoder_hidden_size is None:
            logger.error('encoder_hidden_size is required but missing from projector config', err=KeyError)

        self.hidden_size = self.kwargs.get('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from projector config', err=KeyError)

        self.downsample_ratio = self.kwargs.pop('downsample_ratio', None)
        if self.downsample_ratio is None:
            logger.error('downsample_ratio is required but missing from projector config', err=KeyError)

        # For PTQ
        self.image_size = self.kwargs.pop('image_size', None)
        if self.image_size is None:
            logger.warning(
                'Image size is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )

        self.patch_size = self.kwargs.pop('patch_size', None)
        if self.patch_size is None:
            logger.warning(
                'Patch size is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} vision projector config:')
        logger.info(f'Hidden size:      {self.hidden_size}')
        logger.info(f'Downsample ratio: {self.downsample_ratio}')
