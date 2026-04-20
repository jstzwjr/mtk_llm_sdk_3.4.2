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
"""Qwen2.5-VL ViT Configuration."""

from ...utils import logger
from ..configuration_base import BaseVisionEncoderChunkConfig


class Qwen2_5VLVisionConfig(BaseVisionEncoderChunkConfig):  # noqa: N801
    """Configuration class for the Qwen2.5-VL Vision Transformer (ViT) model.

    This class is used to store the configuration of a Qwen2.5-VL ViT model. It is used to instantiate a Qwen2.5-VL ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Qwen2.5-VL ViT model.

    Attributes:
        model_type (str): The type of the model. Should be 'qwen2_5_vl_vision'.
        depth (int): The depth of the model.
        embed_dim (int): The embedding dimension.
        intermediate_size (int) : The output size of MLP gate_proj & up_proj
        hidden_size (int): The hidden size of the model.
        hidden_act (str): The activation function.
        mlp_ratio (int): The ratio of the MLP.
        num_heads (int): The number of attention heads.
        in_channels (int): The number of input channels.
        patch_size (int): The size of the patches.
        spatial_merge_size (int): The size of the spatial merge.
        temporal_patch_size (int): The size of the temporal patches.

    Methods:
        print_config(): Prints the configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initializes the Qwen2_5VLVisionConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'qwen2_5_vl_vision'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'qwen2_5_vl_vision')
        if self.model_type != 'qwen2_5_vl_vision':
            raise RuntimeError(f'Expected model_type to be qwen2_5_vl_vision but got {self.model_type} instead')

        self.depth = self.kwargs.pop('num_hidden_layers', 32)
        self.num_hidden_layers = self.depth
        self.embed_dim = self.kwargs.pop('embed_dim', 1280)
        self.intermediate_size = self.kwargs.pop('intermediate_size', 3420)
        self.hidden_size = self.kwargs.pop('hidden_size', 3584)
        self.hidden_act = self.kwargs.pop('hidden_act', 'silu')
        if self.hidden_act != 'silu':
            logger.error(
                f'Only `silu` act function is supported for {self.model_type} MLP but {self.hidden_act} was stated.',
                err=ValueError,
            )
        self.mlp_ratio = self.kwargs.pop('mlp_ratio', 4)
        self.num_heads = self.kwargs.pop('num_heads', 16)
        self.num_attention_heads = self.num_heads
        self.in_channels = self.kwargs.pop('in_chans', 3)
        self.patch_size = self.kwargs.pop('patch_size', 14)
        self.spatial_merge_size = self.kwargs.pop('spatial_merge_size', 2)
        self.temporal_patch_size = self.kwargs.pop('temporal_patch_size', 2)
        self.use_conv2d_patch_embed = self.kwargs.pop('use_conv2d_patch_embed', False)

        # Qwen2.5-VL attributes
        self.tokens_per_second = self.kwargs.pop('tokens_per_second', 4)
        self.window_size = self.kwargs.pop('window_size', 112)
        self.out_hidden_size = self.kwargs.pop('out_hidden_size', 3584)
        self.fullatt_block_indexes = self.kwargs.pop('fullatt_block_indexes', [7, 15, 23, 31])
        self.exclude_first_gather = self.kwargs.pop('exclude_first_gather', False)
        self.mask_value = self.kwargs.pop('mask_value', -10000)

        # For fixed shape PTQ
        self.image_resolution = self.kwargs.pop('image_resolution', None)
        self.preprocessor_config = self.kwargs.pop('preprocessor_config', None)

        self.projector_type = kwargs.pop('projector_type', 'patchmerger')
        if self.projector_type != 'patchmerger':
            raise ValueError(f"The projector for Qwen2.5-VL ViT must be 'patchmerger', got {self.projector_type}.")

        self.fc_names = {
            'attn': {
                'name': 'attn',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'proj',
            },
            'mlp': {'name': 'mlp', 'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'},
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
        logger.info(f'depth:                 {self.depth}')
        logger.info(f'embed_dim:             {self.embed_dim}')
        logger.info(f'intermediate_size:     {self.intermediate_size}')
        logger.info(f'hidden_size:           {self.hidden_size}')
        logger.info(f'hidden_act:            {self.hidden_act}')
        logger.info(f'mlp_ratio:             {self.mlp_ratio}')
        logger.info(f'num_heads:             {self.num_heads}')
        logger.info(f'in_channels:           {self.in_channels}')
        logger.info(f'patch_size:            {self.patch_size}')
        logger.info(f'spatial_merge_size:    {self.spatial_merge_size}')
        logger.info(f'temporal_patch_size:   {self.temporal_patch_size}')
        logger.info(f'use_conv2d_patch_embed:{self.use_conv2d_patch_embed}')
        # Qwen2.5-VL
        logger.info(f'tokens_per_second:     {self.tokens_per_second}')
        logger.info(f'window_size:           {self.window_size}')
        logger.info(f'out_hidden_size:       {self.out_hidden_size}')
        logger.info(f'fullatt_block_indexes: {self.fullatt_block_indexes}')
        logger.info(f'exclude_first_gather:  {self.exclude_first_gather}')
        logger.info(f'mask_value:            {self.mask_value}')
