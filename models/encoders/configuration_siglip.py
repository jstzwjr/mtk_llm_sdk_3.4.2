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
"""SigLIP Configuration."""

from ...utils import logger
from .configuration_clip import CLIPConfig


class SigLIPConfig(CLIPConfig):
    """Configuration class for the SigLIP model.

    This class is used to store the configuration of a SigLIP model. It is used to instantiate a SigLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the SigLIP model.

    Attributes:
        model_type (str): The type of the model. Should be 'siglip'.
        mlp_gelu (str): The type of GELU activation function used in the MLP part.

    Methods:
        print_config(): Prints the configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initializes the SigLIPConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'siglip'.
        """
        model_type = kwargs.get('model_type', 'siglip')
        if model_type != 'siglip':
            logger.error(f'Expected model_type to be siglip but got {model_type} instead', err=RuntimeError)

        verbose = kwargs.get('verbose', True)

        kwargs['model_type'] = 'clip'  # trick CLIPConfig to accept and prepare this config
        kwargs['verbose'] = False  # trick CLIPConfig to not logger.info it's config first

        super().__init__(**kwargs)

        self.model_type = model_type  # Override model_type back to mobilevlm

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

        self.mlp_gelu = kwargs.pop(
            'mlp_gelu', 'quick_gelu'
        )  # TinyLLaVA SigLIP uses "gelu_pytorch_tanh" for gelu in MLP part
