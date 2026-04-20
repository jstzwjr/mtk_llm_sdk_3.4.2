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
"""MinicpmVNavitSigLIP Preprocessor Configuration."""

from ...utils import logger
from .configuration_siglip import SiglipPreprocessorConfig


class MinicpmVNavitSigLIPPreprocessorConfig(SiglipPreprocessorConfig):
    """Configuration class for the MinicpmVNavitSigLIP Preprocessor model.

    This class is used to store the configuration of a MinicpmVNavitSigLIP Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the MinicpmVNavitSigLIPPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'minicpmv_navit_siglip'.
            KeyError: If required configuration parameters are missing.
        """
        model_type = kwargs.pop('model_type', 'minicpmv_navit_siglip')
        if model_type != 'minicpmv_navit_siglip':
            logger.error(f'Expected model_type to be minicpmv_navit_siglip but got {model_type} instead')

        verbose = kwargs.pop('verbose', True)
        kwargs['model_type'] = 'siglip'
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.model_type = model_type
        self.do_reshape_by_patch = self.kwargs.get('do_reshape_by_patch', False)
        self.do_slice_mode = self.kwargs.get('do_slice_mode', False)
        self.patch_size = self.kwargs.get('patch_size', False)
        self.max_slice_nums = self.kwargs.get('max_slice_nums', None)
        self.scale_resolution = self.kwargs.get('scale_resolution', None)

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        super().print_config()
        logger.info(f'do_reshape_by_patch:    {self.do_reshape_by_patch}')
        logger.info(f'do_slice_mode:          {self.do_slice_mode}')
        logger.info(f'patch_size:             {self.patch_size}')
        logger.info(f'max_slice_nums:         {self.max_slice_nums}')
        logger.info(f'scale_resolution:       {self.scale_resolution}')
