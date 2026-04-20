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
from .configuration_projector_internvl2 import InternVL2ProjectorConfig


class AndesVLProjectorConfig(InternVL2ProjectorConfig):
    """Configuration class for the AndesVL vision projector model.

    This class is used to store the configuration of an AndesVL vision projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the InternVL2ProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        model_type = kwargs.pop('model_type', 'andesvl')
        if model_type not in ['andesvl']:
            logger.error(f'Expected model_type to be andesvl but got {self.model_type} instead', err=RuntimeError)
        verbose = kwargs.pop('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)
        self.model_type = model_type

        # For PTQ
        self.image_resolution = self.kwargs.pop('image_resolution', None)
        if self.image_resolution is None:
            logger.warning(
                'image_resolution is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )
        self.rope_theta = self.kwargs.pop('rope_theta', None)
        if self.rope_theta is None:
            logger.warning(
                'rope_theta is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )
        self.num_attention_heads = self.kwargs.pop('num_attention_heads', None)
        if self.num_attention_heads is None:
            logger.warning(
                'num_attention_heads is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )
        self.preprocessor_config = self.kwargs.pop('preprocessor_config', None)
        if self.preprocessor_config is None:
            logger.warning(
                'preprocessor_config is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        super().print_config()
        logger.info(f'image_resolution:        {self.image_resolution}')
        logger.info(f'rope_theta:              {self.rope_theta}')
        logger.info(f'num_attention_heads:     {self.num_attention_heads}')
