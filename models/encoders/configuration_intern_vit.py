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
"""InternViT configuration."""

from ...utils import logger
from .configuration_clip import CLIPConfig


class InternViTConfig(CLIPConfig):
    """Configuration class for the InternViT model.

    This class is used to store the configuration of an InternViT model. It is used to instantiate an InternViT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the InternViT model.

    Attributes:
        model_type (str): The type of the model. Should be 'intern_vit_6b'.
        qk_normalization (bool): Whether to use QK normalization.
        qkv_bias (bool): Whether to use QKV bias.
        mlp_gelu (str): The activation function for the MLP.

    Methods:
        print_config(): Prints the configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initializes the InternViTConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'intern_vit_6b'.
        """
        model_type = kwargs.get('model_type')  # FIXME: Do we follow HF config or manully change this into `intern_vit`?
        if model_type != 'intern_vit_6b':
            logger.error(f'Expected model_type to be intern_vit_6b but got {model_type} instead', err=RuntimeError)

        verbose = kwargs.get('verbose', True)

        kwargs['model_type'] = 'clip'  # trick CLIPConfig to accept and prepare this config
        kwargs['verbose'] = False  # trick CLIPConfig to not logger.info it's config first

        super().__init__(**kwargs)
        self.model_type = model_type  # Override model_type back to intern_vit_6b
        self.qk_normalization = self.kwargs.pop('qk_normalization', False)
        self.qkv_bias = self.kwargs.pop('qkv_bias', True)
        self.projector_type = self.kwargs.pop('projector_type', 'internvl')
        self.downsample_ratio = self.kwargs.pop('downsample_ratio', 0.5)

        self.select_layer = self.kwargs.pop('select_layer', -1)
        if self.select_layer >= self.num_hidden_layers:
            logger.error('vision_select_layer must either be negative or less than num_hidden_layers', err=ValueError)
        if self.select_layer < 0:
            if self.select_layer < -self.num_hidden_layers:
                logger.error('vision_select_layer cannot be less than -num_hidden_layers', err=ValueError)
            self.select_layer += self.num_hidden_layers + 1

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

        self.mlp_gelu = self.kwargs.pop('encoder_hidden_act', 'gelu')
        if self.mlp_gelu == 'gelu':
            self.mlp_gelu = 'gelu_pytorch_tanh'

    def print_config(self):
        """Prints the configuration parameters."""
        super().print_config()
        logger.info(f'qk_normalization:        {self.qk_normalization}')
        logger.info(f'qkv_bias:                {self.qkv_bias}')
        logger.info(f'projector type:          {self.projector_type}')
        logger.info(f'downsample ratio:        {self.downsample_ratio}')
