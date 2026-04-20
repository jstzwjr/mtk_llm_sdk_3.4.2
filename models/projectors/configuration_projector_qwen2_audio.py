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
"""Qwen2Audio projector Configuration."""

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class Qwen2AudioProjectorConfig(BaseProjectorConfig):
    """Configuration class for the Qwen2Audio projector model.

    This class is used to store the configuration of an Qwen2Audio projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the Qwen2AudioProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'qwen2_audio_encoder')
        if self.model_type != 'qwen2_audio_encoder':
            logger.error(
                f'Expected model_type to be qwen2_audio_encoder but got {self.model_type} instead', err=RuntimeError
            )

        self.d_model = self.kwargs.get('d_model', None)
        if self.d_model is None:
            logger.error('d_model is required but missing from projector config', err=KeyError)

        self.hidden_size = self.kwargs.get('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from projector config', err=KeyError)
        self.max_source_positions = self.kwargs.get('max_source_positions', None)
        if self.max_source_positions is None:
            logger.error('max_source_positions is required but missing from projector config', err=KeyError)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} vision projector config:')
        logger.info(f'Hidden size:           {self.hidden_size}')
        logger.info(f'D_model:               {self.d_model}')
        logger.info(f'Max source position:   {self.max_source_positions}')
