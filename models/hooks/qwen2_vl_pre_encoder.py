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
"""Define Qwen2-VL pre-encoder hook to precompute vision attention mask and vision rotary embedding."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...utils import logger
from ..configuration_hook import HookConfig
from ..encoders.modeling_qwen2vl_vision import (
    precompute_qwen2vl_vision_attn_mask,
    precompute_qwen2vl_vision_rot_emb,
)
from ..modeling_hook_base import BaseHook
from .numpy_to_torch import NumpyToTorch


class CumulativeSeqenceLens(nn.Module):
    """Cumulative Sequence Lens class for computing cumulative sequence lengths.

    Methods:
        __init__(): Initialize the Cumulative Sequence Lens.
        forward(grid_thw): Forward pass to compute cumulative sequence lengths.
    """

    def __init__(self):
        """Initialize the Cumulative Sequence Lens."""
        super().__init__()

    def forward(self, grid_thw):
        """Forward pass to compute cumulative sequence lengths.

        Args:
            grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).

        Returns:
            torch.Tensor: The cumulative sequence lengths.
        """
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen2VLPreEncoderConfig(HookConfig):
    """Configuration class for Qwen2VLPreEncoder.

    This class extends the HookConfig to include additional configurations specific to Qwen2VLPreEncoder.

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

        self.spatial_merge_size = kwargs.pop('spatial_merge_size', None)
        if self.spatial_merge_size is None:
            logger.error('Must provide spatial_merge_size in Qwen2VLPReEncoderConfig.')
        self.embed_dim = kwargs.pop('embed_dim', None)
        if self.embed_dim is None:
            logger.error('Must provide embed_dim in Qwen2VLPReEncoderConfig.')
        self.num_heads = kwargs.pop('num_heads', None)
        if self.num_heads is None:
            logger.error('Must provide num_heads in Qwen2VLPReEncoderConfig.')

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'spatial_merge_size:   {self.spatial_merge_size}')
        logger.info(f'embed_dim:            {self.embed_dim}')
        logger.info(f'num_heads:            {self.num_heads}')


class Qwen2VLPreEncoder(BaseHook):
    """Qwen2VLPreEncoder class that precomputes necessary input for Qwen2-VL ViT.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Qwen2VLPreEncoder.
        forward(inputs): Forward pass.
    """

    def __init__(self, config: Qwen2VLPreEncoderConfig):
        """Initialize the NumpyToTorch.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)
        self.cu_seqlens = CumulativeSeqenceLens()
        self.numpy_to_torch = NumpyToTorch(config=HookConfig(name='numpy_to_torch', type='preencoder'))

    def forward(self, image, **kwargs):
        """Forward pass.

        Args:
            image: The image to be passed to encoder. It will just directly be passed through.
            kwargs (dict, optional): Additional keyword arguments.
        """
        image_grid_thw = kwargs.get('image_grid_thw')

        if image_grid_thw is None:
            logger.error('image_grid_thw must be passed when forwarding Qwen2VLPreEncoder.')
        if isinstance(image_grid_thw, np.ndarray):
            image_grid_thw = torch.from_numpy(image_grid_thw)

        seq_length = kwargs.get('qwen2vl_img_seq_length')
        if seq_length is None:
            logger.error('qwen2vl_img_seq_length must be passed when forwarding Qwen2VLPreEncoder.')

        pipeline_type = kwargs.get('pipeline_type')
        if pipeline_type is None:
            logger.error('pipeline_type must be passed when forwarding Qwen2VLPreEncoder.')

        cu_seqlens = self.cu_seqlens(image_grid_thw)
        vision_attn_mask = precompute_qwen2vl_vision_attn_mask(seq_length=seq_length, cu_seqlens=cu_seqlens)
        vision_rot_emb = precompute_qwen2vl_vision_rot_emb(image_grid_thw, self.config)

        kwargs.update({'qwen2vl_vision_attn_mask': vision_attn_mask, 'qwen2vl_vision_rot_emb': vision_rot_emb})

        if pipeline_type == 'float':
            # numpy to torch
            image, kwargs = self.numpy_to_torch(image, **kwargs)

        return image, kwargs
