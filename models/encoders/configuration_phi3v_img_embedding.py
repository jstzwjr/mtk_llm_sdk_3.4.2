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
"""Phi3-V ViT Configuration."""

from ...utils import logger
from ..configuration_base import BaseVisionEncoderChunkConfig


class Phi3VImgEmbeddingConfig(BaseVisionEncoderChunkConfig):
    """This config extend BaseVisionEncoderChunkConfig to contain necessary config attributes for phi3-v."""

    def __init__(self, **kwargs):
        """Initializes the Phi3VImgEmbeddingConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'phi3_vision_emb'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'phi3_vision_emb')
        if self.model_type != 'phi3_vision_emb':
            logger.error(
                f'Expected model_type to be phi3_vision_emb but got {self.model_type} instead', err=RuntimeError
            )

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from config.json', err=KeyError)

        self.img_processor = self.kwargs.pop('img_processor', None)
        if self.img_processor is None:
            logger.error('img_processor is required but missing from config.json', err=KeyError)

        self.vocab_size = self.kwargs.pop('vocab_size', None)
        if self.vocab_size is None:
            logger.error('vocab_size is required but missing from config.json', err=ValueError)

        self.use_hd_transform = self.kwargs.pop('use_hd_transform', True)
        self.with_learnable_separator = self.kwargs.pop('with_learnable_separator', True)
        self.hd_transform_order = self.kwargs.pop('hd_transform_order', 'glb_sub')

        self.num_hidden_layers = self.kwargs.pop('num_hidden_layers', 24)  # Is fixed in modeling

        if isinstance(self.img_processor, dict):
            self.select_layer = self.img_processor.get('layer_idx', -2)
            self.type_feature = self.img_processor.get('type_feature', 'patch')
        else:
            self.select_layer = -2
            self.type_feature = 'patch'
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

        # For PTQ
        self.fixed_img_sizes = self.kwargs.pop('fixed_img_size', None)
        if self.fixed_img_sizes is None:
            logger.error('fixed_img_size is required but missing in config.json.', err=ValueError)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} config:')
        logger.info(f'Hidden size:              {self.hidden_size}')
        logger.info(f'Img processor:            {self.img_processor}')
        logger.info(f'LLM vocab size:           {self.vocab_size}')
        logger.info(f'Use HD Transform:         {self.use_hd_transform}')
        logger.info(f'With learnable separator: {self.with_learnable_separator}')
        logger.info(f'HD Transform order:       {self.hd_transform_order}')
        logger.info(f'Select layer:             {self.select_layer}')
        logger.info(f'Type feature:             {self.type_feature}')
        logger.info(f'Fixed image size:         {self.fixed_img_sizes}')
