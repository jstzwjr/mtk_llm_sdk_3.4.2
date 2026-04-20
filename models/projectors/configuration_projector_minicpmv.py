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
"""MinicpmV projector Configuration."""

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class MinicpmVProjectorConfig(BaseProjectorConfig):
    """Configuration class for the MinicpmV projector model.

    This class is used to store the configuration of an MinicpmV projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the MinicpmVProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        model_type = kwargs.pop('model_type', 'minicpmv')
        if model_type != 'minicpmv':
            logger.error(f'Expected model_type to be minicpmv but got {self.model_type} instead', err=RuntimeError)

        self.model_type = model_type

        self.grid_size = self.kwargs.pop('grid_size', None)
        if self.grid_size is None:
            logger.warning(
                'grid_size is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )

        self.embed_dim = self.kwargs.pop('embed_dim', None)
        if self.embed_dim is None:
            logger.warning(
                'embed_dim is not given, make sure the task is not PTQ, otherwise error would be encountered.'
            )

        self.kv_dim = self.kwargs.pop('kv_dim', None)
        if self.kv_dim is None:
            logger.warning('kv_dim is not given, make sure the task is not PTQ, otherwise error would be encountered.')

        self.use_resampler_embed = self.kwargs.pop('use_resampler_embed', False)

        self.patch_size = self.kwargs.pop('patch_size', None)
        if self.patch_size is None:
            raise KeyError('patch_size is required but missing from config.json')

        self.ptq_image_batch = self.kwargs.pop('ptq_image_batch', None)
        if self.ptq_image_batch is None:
            raise KeyError('ptq_image_batch is required but missing from config.json')

        self.ptq_image_width = self.kwargs.pop('ptq_image_width', None)
        if self.ptq_image_width is None:
            raise KeyError('ptq_image_width is required but missing from config.json')

        self.ptq_image_height = self.kwargs.pop('ptq_image_height', None)
        if self.ptq_image_height is None:
            raise KeyError('ptq_image_height is required but missing from config.json')

        self.num_heads = self.embed_dim // 128

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        super().print_config()
        logger.info(f'Grid size:               {self.grid_size}')
        logger.info(f'Embed dim:               {self.embed_dim}')
        logger.info(f'Num heads:               {self.num_heads}')
        logger.info(f'KV dim:                  {self.kv_dim}')
        logger.info(f'Use resampler embed:     {self.use_resampler_embed}')
        logger.info(f'Patch size:              {self.patch_size}')
        logger.info(f'PTQ image batch:         {self.ptq_image_batch}')
        logger.info(f'PTQ image width:         {self.ptq_image_width}')
        logger.info(f'PTQ image height:        {self.ptq_image_height}')
