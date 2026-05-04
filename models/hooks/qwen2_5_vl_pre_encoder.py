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
"""Define Qwen2.5-VL pre-encoder hook to precompute vision attention mask and vision rotary embedding."""

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from ...utils import logger
from ..configuration_hook import HookConfig
from ..encoders.modeling_qwen2_5vl_vision import (
    get_window_index,
    precompute_qwen2_5vl_vision_attn_mask,
    precompute_qwen2_5vl_vision_rot_emb,
)
from ..modeling_hook_base import BaseHook
from .numpy_to_torch import NumpyToTorch
from .qwen2_vl_pre_encoder import CumulativeSeqenceLens


class Qwen2_5VLPreEncoderConfig(HookConfig):  # noqa: N801
    """Configuration class for Qwen2_5VLPreEncoderConfig.

    This class extends the HookConfig to include additional configurations specific to Qwen2_5VLPreEncoderConfig.

    Attributes:
        spatial_merge_size (int): The spatial merge size.
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of heads.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the Qwen2_5VLPreEncoderConfig.

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
            logger.error('Must provide spatial_merge_size in Qwen2_5VLPreEncoderConfig.', err=ValueError)
        self.embed_dim = kwargs.pop('embed_dim', None)
        if self.embed_dim is None:
            logger.error('Must provide embed_dim in Qwen2_5VLPreEncoderConfig.', err=ValueError)
        self.num_heads = kwargs.pop('num_heads', None)
        if self.num_heads is None:
            logger.error('Must provide num_heads in Qwen2_5VLPreEncoderConfig.', err=ValueError)

        # Qwen2.5-VL
        self.window_size = kwargs.pop('window_size', None)
        if self.window_size is None:
            logger.error('Must provide window_size in Qwen2_5VLPreEncoderConfig', err=ValueError)
        self.patch_size = kwargs.pop('patch_size', None)
        if self.patch_size is None:
            logger.error('Must provide patch_size in Qwen2_5VLPreEncoderConfig', err=ValueError)
        self.fullatt_block_indexes = self.kwargs.pop('fullatt_block_indexes', [7, 15, 23, 31])
        self.exclude_first_gather = self.kwargs.pop('exclude_first_gather', False)
        self.mask_value = self.kwargs.pop('mask_value', -10000)

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
        # Qwen2.5-VL
        logger.info(f'window_size:          {self.window_size}')
        logger.info(f'patch_size:           {self.patch_size}')
        logger.info(f'fullatt_block_indexes:{self.fullatt_block_indexes}')
        logger.info(f'exclude_first_gather: {self.exclude_first_gather}')
        logger.info(f'mask_value:           {self.mask_value}')


class Qwen2_5VLPreEncoder(BaseHook):  # noqa: N801
    """Qwen2VLPreEncoder class that precomputes necessary input for Qwen2.5-VL ViT.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Qwen2_5VLPreEncoder.
        forward(inputs): Forward pass.
    """

    def __init__(self, config: Qwen2_5VLPreEncoderConfig):
        """Initialize the Qwen2_5VLPreEncoder.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)
        self.numpy_to_torch = NumpyToTorch(config=HookConfig(name='numpy_to_torch', type='preencoder'))
        self.cumulative_sequencelens = CumulativeSeqenceLens()
        self.spatial_merge_unit = self.config.spatial_merge_size * self.config.spatial_merge_size

    def forward(self, image, **kwargs):
        """Forward pass.

        Args:
            image: The image to be passed to encoder. It will just directly be passed through.
            kwargs (dict, optional): Additional keyword arguments.
        """
        image_grid_thw = kwargs.get('image_grid_thw')

        if image_grid_thw is None:
            logger.error('image_grid_thw must be passed when forwarding Qwen2_5VLPreEncoder.')
        if isinstance(image_grid_thw, np.ndarray):
            image_grid_thw = torch.from_numpy(image_grid_thw)

        seq_length = kwargs.get('qwen2vl_img_seq_length')
        if seq_length is None:
            logger.error('qwen2vl_img_seq_length must be passed when forwarding Qwen2_5VLPreEncoder.')

        pipeline_type = kwargs.get('pipeline_type')
        if pipeline_type is None:
            logger.error('pipeline_type must be passed when forwarding Qwen2_5VLPreEncoder.')

        # Calculate window attention cu_seqlens
        # Skip window attention for models without window (e.g. Qwen3-VL, window_size=0)
        if self.config.window_size == 0:
            window_index = None
            cu_window_seqlens = None
        else:
            window_index, cu_window_seqlens = get_window_index(image_grid_thw, self.config)
        # Calculate normal attention cu_seqlens
        cu_seqlens = self.cumulative_sequencelens(image_grid_thw)

        if cu_window_seqlens is not None:
            cu_window_seqlens = torch.tensor(cu_window_seqlens, device=image_grid_thw.device, dtype=image_grid_thw.dtype)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
            vision_attn_mask_window = precompute_qwen2_5vl_vision_attn_mask(
                seq_length=seq_length, cu_seqlens=cu_window_seqlens, mask_value=self.config.mask_value
            )
        else:
            vision_attn_mask_window = None

        vision_attn_mask_normal = precompute_qwen2_5vl_vision_attn_mask(
            seq_length=seq_length, cu_seqlens=cu_seqlens, mask_value=self.config.mask_value
        )

        vision_rot_emb = precompute_qwen2_5vl_vision_rot_emb(image_grid_thw, self.config)

        kwargs.update(
            {
                'qwen2_5vl_vision_attn_mask': vision_attn_mask_normal,
                'qwen2_5vl_vision_attn_mask_window': vision_attn_mask_window,
                'qwen2_5vl_vision_rot_emb': vision_rot_emb,
                'qwen2_5vl_vision_window_index': window_index,
                'qwen2_5vl_vision_fullatt_block_indexes': self.config.fullatt_block_indexes,
            }
        )

        # Need to perform gather operation if the tflite does not contains gather during inference quantized
        if pipeline_type == 'quantized' and self.config.exclude_first_gather:
            logger.debug('Inference quantized of Qwen2.5-VL, prepare input gather before tflite inference.')
            batch_feature = isinstance(image, BatchFeature)
            logger.debug(f'Is batch_feature: {batch_feature}')
            arr = image['input_features'] if batch_feature else image
            arr = torch.tensor(arr)
            seq_len = arr.size()[0]
            arr = arr.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)

            # GATHER OP implementation (skip if no window attention, e.g. Qwen3-VL)
            if window_index is not None:
                arr = torch.index_select(arr, 0, window_index)
            arr = arr.reshape(seq_len, -1).detach().cpu().numpy()
            if batch_feature:
                image['input_features'] = arr

        if pipeline_type == 'float':
            # numpy to torch
            image, kwargs = self.numpy_to_torch(image, **kwargs)

        return image, kwargs
