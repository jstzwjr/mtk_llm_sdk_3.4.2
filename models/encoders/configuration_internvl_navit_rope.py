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
"""InternViT Navit Rope configuration."""

from ...utils import logger
from .configuration_intern_vit import InternViTConfig


class InternViTNavitRopeConfig(InternViTConfig):
    """Configuration class for the InternViTNavitRope model.

    This class is used to store the configuration of an InternViTNavitRope model.
    It is used to instantiate an InternViTNavitRope model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the InternViTNavitRope model.
    """

    def __init__(self, **kwargs):
        """Initializes the InternViTConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'intern_vit_6b'.
        """
        model_type = kwargs.get('model_type')
        if model_type != 'intern_vit_6b_navit_rope':
            logger.error(
                f'Expected model_type to be intern_vit_6b_navit_rope but got {model_type} instead', err=RuntimeError
            )

        verbose = kwargs.get('verbose', True)

        kwargs['model_type'] = 'intern_vit_6b'  # trick InternViTConfig to accept and prepare this config
        kwargs['verbose'] = False  # trick InternViTConfig to not logger.info it's config first

        super().__init__(**kwargs)
        self.model_type = model_type

        self.rope_theta = self.kwargs.pop('rope_theta', 10000.0)
        self.norm_type = self.kwargs.pop('norm_type', 'rms_norm')
        self.initializer_factor = self.kwargs.pop('initializer_factor', 1.0)

        # PTQ
        self.image_resolution = self.kwargs.pop('image_resolution', None)
        if self.image_resolution is None:
            logger.warning(
                'image_resolution is not given, make sure the task is not PTQ, otherwise error would be encountered.'
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
        logger.info(f'rope_theta:              {self.rope_theta}')
        logger.info(f'norm_type:               {self.norm_type}')
        logger.info(f'initializer_factor:      {self.initializer_factor}')
        logger.info(f'image_resolution:        {self.image_resolution}')
