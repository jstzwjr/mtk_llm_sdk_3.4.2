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
"""CLIP ViT Configuration."""

from ...utils import logger
from ..configuration_base import BaseVisionEncoderChunkConfig


class CLIPConfig(BaseVisionEncoderChunkConfig):
    """Configuration class for the CLIP model.

    This class is used to store the configuration of a CLIP model. It is used to instantiate a CLIP model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the CLIP model.

    Attributes:
        model_type (str): The type of the model. Should be 'clip'.
        hidden_size (int): The size of the hidden layers.
        intermediate_size (int): The size of the intermediate layers.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        image_size (int): The size of the input images.
        patch_size (int): The size of the patches.
        num_channels (int): The number of channels in the input images.
        norm_eps (float): The epsilon value for layer normalization.
        mask_value (float): The value used for masking.

    Methods:
        print_config(): Prints the configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initializes the CLIPConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'clip'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'clip')
        if self.model_type != 'clip':
            logger.error(f'Expected model_type to be clip but got {self.model_type} instead', err=RuntimeError)

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from config.json', err=KeyError)

        self.intermediate_size = self.kwargs.pop('intermediate_size', None)
        if self.intermediate_size is None:
            logger.error('intermediate_size is required but missing from config.json', err=KeyError)

        self.num_hidden_layers = self.kwargs.pop('num_hidden_layers', None)
        if self.num_hidden_layers is None:
            logger.error('num_hidden_layers is required but missing from config.json', err=KeyError)

        self.num_attention_heads = self.kwargs.pop('num_attention_heads', None)
        if self.num_attention_heads is None:
            logger.error('num_attention_heads is required but missing from config.json', err=KeyError)

        self.image_size = self.kwargs.pop('image_size', None)
        if self.image_size is None:
            logger.error('image_size is required but missing from config.json', err=KeyError)

        self.patch_size = self.kwargs.pop('patch_size', None)
        if self.patch_size is None:
            logger.error('patch_size is required but missing from config.json', err=KeyError)

        self.num_channels = self.kwargs.pop('num_channels', 3)

        self.norm_eps = self.kwargs.pop('layer_norm_eps', self.kwargs.pop('norm_eps', 1e-5))
        self.mask_value = self.kwargs.pop('mask_value', self.mask_value)

        self.select_layer = self.kwargs.pop('vision_select_layer', -1)
        if self.select_layer >= self.num_hidden_layers:
            logger.error('vision_select_layer must either be negative or less than num_hidden_layers', err=ValueError)
        if self.select_layer < 0:
            if self.select_layer < -self.num_hidden_layers:
                logger.error('vision_select_layer cannot be less than -num_hidden_layers', err=ValueError)
            self.select_layer += self.num_hidden_layers + 1

        self.fc_names = {
            'attn': {
                'name': 'self_attn',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'out_proj',
            },
            'mlp': {'name': 'mlp', 'fc1': 'fc1', 'fc2': 'fc2'},
        }
        self.norm_names = {
            'pre': 'pre_layrnorm',  # Intentional typo
            'post': 'post_layernorm',
            'layernorm1': 'layer_norm1',
            'layernorm2': 'layer_norm2',
        }

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} config:')
        logger.info(f'Hidden size:          {self.hidden_size}')
        logger.info(f'Intermediate size:    {self.intermediate_size}')
        logger.info(f'Num layers:           {self.num_hidden_layers}')
        logger.info(f'Num attention heads:  {self.num_attention_heads}')
        logger.info(f'Image size:           {self.image_size}')
        logger.info(f'Patch size:           {self.patch_size}')
        logger.info(f'Norm epsilon:         {self.norm_eps}')
        logger.info(f'Num channels:         {self.num_channels}')
        logger.info(f'Select layer:         {self.select_layer}')
