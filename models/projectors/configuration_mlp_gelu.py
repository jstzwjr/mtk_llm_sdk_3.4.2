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
"""MLP-Gelu vision projector Configuration."""

import re

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class MLPGeluProjectorConfig(BaseProjectorConfig):
    """Configuration class for the MLP-Gelu vision projector model.

    This class is used to store the configuration of a MLP-Gelu vision projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the MLPGeluProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'mlp_gelu')
        if self.model_type != 'mlp_gelu':
            logger.error(f'Expected model_type to be mlp_gelu but got {self.model_type} instead', err=RuntimeError)

        self.projector_type = self.kwargs.pop('projector_type', 'mlp2x_gelu')
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        self.mlp_depth = int(mlp_gelu_match.group(1))

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from projector config', err=KeyError)

        self.projection_dim = self.kwargs.pop('projection_dim', None)
        if self.projection_dim is None:
            logger.error('projection_dim is required but missing from projector config', err=KeyError)

        self.image_size = self.kwargs.pop('image_size', None)
        if self.image_size is None:
            logger.error('image_size is required but missing from config.json', err=KeyError)

        self.patch_size = self.kwargs.pop('patch_size', None)
        if self.patch_size is None:
            logger.error('patch_size is required but missing from config.json', err=KeyError)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} vision projector config:')
        logger.info(f'Projector type: {self.projector_type}')
        logger.info(f'MLP depth:      {self.mlp_depth}')
        logger.info(f'Hidden size:    {self.hidden_size}')
        logger.info(f'Projection dim: {self.projection_dim}')
        logger.info(f'Image size:     {self.image_size}')
        logger.info(f'Patch size:     {self.patch_size}')
