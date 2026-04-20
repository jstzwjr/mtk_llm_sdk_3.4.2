# Copyright (C) 2025 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# ==================================================================================================
"""Define AndesVL pre-encoder operations."""

from ...utils import logger
from ..configuration_hook import HookConfig
from ..encoders.modeling_internvl_navit_rope import (
    precompute_cuseqlen,
    precompute_vision_rotary_embedding,
)
from ..modeling_hook_base import BaseHook
from .numpy_to_torch import NumpyToTorch


class AndesVLPreEncoderConfig(HookConfig):
    """Configuration class for AndesVLPreEncoderHook.

    This class extends the HookConfig to include additional configurations specific to AndesVLPreEncoderHook.

    Attributes:
        spatial_merge_size (int): The spatial merge size.
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of heads.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the Qwen2VLPreEncoderConfig.

        Args:
            kwargs (dict, optional): Additional keyword arguments.

        Raises:
            ValueError: If spatial_merge_size, embed_dim, or num_heads are not provided.
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing in config.', err=ValueError)
        self.num_attention_heads = self.kwargs.pop('num_attention_heads', None)
        if self.num_attention_heads is None:
            logger.error('num_attention_heads is requried but missing in config.', err=ValueError)
        self.rope_theta = self.kwargs.pop('rope_theta', None)
        if self.rope_theta is None:
            logger.error('rope_theta is required but missing in config.', err=ValueError)

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'hidden_size:           {self.hidden_size}')
        logger.info(f'num_attention_heads:   {self.num_attention_heads}')
        logger.info(f'rope_theta:            {self.rope_theta}')


class AndesVLPreEncoderHook(BaseHook):
    """Define AndesVLPreEncoderHook class."""

    def __init__(self, config: HookConfig, **kwargs):
        """Initialize AndesVLPreEncoderHook."""
        super().__init__(config, **kwargs)
        self.numpy_to_torch = NumpyToTorch(config=HookConfig(name='numpy_to_torch', type='preencoder'))

    def forward(self, image, **kwargs):
        """Forward pass.

        Args:
            image: The image to be passed to encoder. It will just directly be passed through.
            kwargs (dict, optional): Additional keyword arguments.
        """
        grid_hw = kwargs.get('Andes_image_grid_hw')
        if grid_hw is None:
            logger.error('Andes_image_grid_hw must be passed when forwarding AndesVLPreEncoderHook')
        pipeline_type = kwargs.get('pipeline_type')
        if pipeline_type is None:
            logger.error('pipeline_type must be passed when forwarding AndesVLPreEncoderHook.')
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        logger.debug(f'seq_length: {image["input_features"].shape[0]}')
        cu_seqlens = precompute_cuseqlen(grid_hw, seq_length=image['input_features'].shape[0])
        rotary_pos_emb = precompute_vision_rotary_embedding(head_dim, self.config.rope_theta, grid_hw)

        kwargs.update(
            {'internvl_navit_rope_cu_seqlens': cu_seqlens, 'internvl_navit_rope_rotary_pos_emb': rotary_pos_emb}
        )
        if pipeline_type == 'float':
            # numpy to torch
            self.numpy_to_torch(image)

        return image, kwargs
