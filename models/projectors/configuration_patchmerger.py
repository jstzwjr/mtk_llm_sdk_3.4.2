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
"""Qwen2-VL vision projector (Patch Merger) Configuration."""

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class PatchMergerProjectorConfig(BaseProjectorConfig):
    """Configuration class for the Qwen2-VL vision projector model (Patch Merger).

    This class is used to store the configuration of a Qwen2-VL patch merger model.
    """

    def __init__(self, **kwargs):
        """Initializes the PatchMergerProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = self.kwargs.pop('model_type', 'qwen2_vl')
        if self.model_type not in ['qwen2_vl', 'patch_merger', 'qwen2_5_vl']:
            logger.error(
                f'Expected model_type to be qwen2_vl or patch_merger but got {self.model_type} instead',
                err=RuntimeError,
            )
        self.dim = self.kwargs.pop('dim', None)
        if self.dim is None:
            logger.error('dim must be in PAtch Merger config but got None.', err=ValueError)

        self.embed_dim = self.kwargs.pop('embed_dim', None)
        if self.embed_dim is None:
            logger.error('embed_dim must be in Patch Merger config but got None.', err=ValueError)

        self.spatial_merge_size = self.kwargs.pop('spatial_merge_size', 2)
        self.hidden_size = self.embed_dim * (self.spatial_merge_size**2)

        # For PTQ
        self.image_resolution = self.kwargs.pop('image_resolution', None)
        self.preprocessor_config = self.kwargs.pop('preprocessor_config', None)
        self.num_heads = self.kwargs.pop('num_heads', 16)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} vision projector config:')
        logger.info(f'Dim:                {self.dim}')
        logger.info(f'Embed dim:          {self.embed_dim}')
        logger.info(f'Spatial merge size: {self.spatial_merge_size}')
        logger.info(f'Hidden size:        {self.hidden_size}')
