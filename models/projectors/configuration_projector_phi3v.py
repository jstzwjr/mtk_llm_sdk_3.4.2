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
"""Phi3-V vision projector Configuration."""

from ...utils import logger
from ..configuration_base import BaseProjectorConfig


class Phi3VProjectorConfig(BaseProjectorConfig):
    """Configuration class for the Phi3-V vision projector model.

    This class is used to store the configuration of an Phi3-V vision projector model.
    """

    def __init__(self, **kwargs):
        """Initializes the Phi3VProjectorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'phi3v')
        if self.model_type != 'phi3v':
            logger.error(f'Expected model_type to be phi3v but got {self.model_type} instead', err=RuntimeError)

        self.projector_cls = self.kwargs.pop('projector_cls', 'linear')
        self.use_hd_transform = self.kwargs.pop('use_hd_transform', None)
        if self.use_hd_transform is None:
            logger.error('use_hd_transform is required but missing in config.json.', err=ValueError)

        self.img_dim_out = self.kwargs.pop('img_dim_out', None)
        if self.img_dim_out is None:
            logger.error('img_dim_out is required but missing in config.json.', err=ValueError)

        self.hidden_size = self.kwargs.pop('n_embd', self.kwargs.pop('hidden_size', None))
        if self.hidden_size is None:
            logger.error('n_embd or hidden_size is required but missing in config.json.', err=ValueError)

        # For PTQ
        self.img_sizes = self.kwargs.pop('fixed_img_size', None)
        if self.img_sizes is None:
            logger.error('img_sizes is required but missing in config.json.')

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} vision projector config:')
        logger.info(f'Projector cls:      {self.projector_cls}')
        logger.info(f'UsSe HD Transform:  {self.use_hd_transform}')
        logger.info(f'Img dim out:        {self.img_dim_out}')
        logger.info(f'Hidden size:        {self.hidden_size}')
        logger.info(f'Fixed image size:   {self.img_sizes}')
