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
"""MinicpmV NavitSigLIP configuration."""

from ...utils import logger
from .configuration_siglip import SigLIPConfig


class MinicpmVNavitSigLIPConfig(SigLIPConfig):
    """Configuration class for the MinicpmVNavitSigLIPConfig model.

    This class is used to store the configuration of a MinicpmVNavitSigLIPConfig model. It is used to instantiate the
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the SigLIP model.

    Attributes:
        model_type (str): The type of the model. Should be 'minicpmv_navit_siglip'.
        mlp_gelu (str): The type of GELU activation function used in the MLP part.

    Methods:
        print_config(): Prints the configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initializes the MinicpmVNavitSigLIPConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'minicpmv_navit_siglip'.
        """
        model_type = kwargs.get('model_type', 'minicpmv_navit_siglip')
        if model_type != 'minicpmv_navit_siglip':
            logger.error(f'Expected model_type to be siglip but got {model_type} instead', err=RuntimeError)

        verbose = kwargs.get('verbose', True)

        kwargs['model_type'] = 'siglip'  # trick SigLIPConfig to accept and prepare this config
        kwargs['verbose'] = False  # trick SigLIPConfig to not print it's config first

        super().__init__(**kwargs)

        self.model_type = model_type  # Override model_type back to navitsiglip

        self.ptq_image_batch = self.kwargs.pop('ptq_image_batch', None)
        if self.ptq_image_batch is None:
            raise KeyError('ptq_image_batch is required but missing from config.json')

        self.ptq_image_width = self.kwargs.pop('ptq_image_width', None)
        if self.ptq_image_width is None:
            raise KeyError('ptq_image_width is required but missing from config.json')

        self.ptq_image_height = self.kwargs.pop('ptq_image_height', None)
        if self.ptq_image_height is None:
            raise KeyError('ptq_image_height is required but missing from config.json')

        self.image_text = self.kwargs.pop('image_text', None)
        if self.image_text is None:
            raise KeyError('image_text is required but missing from config.json')

        self.image_token = self.kwargs.pop('image_token', None)
        if self.image_token is None:
            raise KeyError('image_token is required but missing from config.json')

        self.do_reshape_by_patch = self.kwargs.pop('do_reshape_by_patch', None)
        if self.do_reshape_by_patch is None:
            raise KeyError('do_reshape_by_patch is required but missing from config.json')

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        super().print_config()
        logger.info(f'{self.model_type} config:')
        logger.info(f'PTQ image batch:     {self.ptq_image_batch}')
        logger.info(f'PTQ image width:     {self.ptq_image_width}')
        logger.info(f'PTQ image height:    {self.ptq_image_height}')
        logger.info(f'Image text:          {self.image_text}')
        logger.info(f'Image token:         {self.image_token}')
        logger.info(f'Do reshape by patch: {self.do_reshape_by_patch}')
