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
"""Define internvl2 get_embeds class."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class InternVL2GetEmbeds(BaseGetEmbedsHook):
    """Class for InternVL2GetEmbeds.

    This class extends the BaseGetEmbedsHook to include functionalities specific to InternVL2.

    Attributes:
        config (object): The configuration object.
        kwargs (dict, optional): Additional keyword arguments.

    Methods:
        forward: Forward pass to get embeddings.
    """

    def __init__(self, config, **kwargs):
        """Initialize the InternVL2GetEmbeds.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config=config, **kwargs)
        logger.debug(f'image_token_ids={self.mm_token_ids}')

    def forward(self, tokens=None, multimodal_embeds=None, custom_embeds=None, **kwargs):
        """Forward pass to get embeddings.

        Args:
            tokens (torch.Tensor, optional): Input tokens. Defaults to None.
            multimodal_embeds (torch.Tensor, optional): Multimodal embeddings. Defaults to None.
            custom_embeds (Any, optional): Custom embeddings. Defaults to None.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or tuple: The embeddings.
        """
        logger.debug('Enter InternVL2GetEmbeds forward')
        self.mm_token_ids = kwargs.get('internvl2_image_token_id')
        logger.debug(f'InternVL2 mm_token_ids: {self.mm_token_ids}')
        if self.mm_token_ids is None:
            logger.error('Must pass image_token_ids when getting InternVL2 embeddings', err=ValueError)
        if custom_embeds is not None:
            return custom_embeds, kwargs
        if multimodal_embeds is None:
            logger.error('multimodal_embeds should not be None for llava get_embeds hook.')
        pipeline_type = kwargs.get('pipeline_type')
        logger.debug(f'pipeline_type: {pipeline_type}.')

        if pipeline_type == 'quantized':
            max_patch = kwargs.get('internvl2_max_patch')
            original_batch_size = kwargs.get('internvl2_original_batch_size')
            if max_patch is None:
                logger.error('max_patch must be passed when inference quantized InternVL2.')
            if original_batch_size is None:
                logger.error('original_batch_size must be passed when inference quantized InternVL2.')

            batch_pad = (max_patch + 1) - original_batch_size
            if original_batch_size < (max_patch + 1):
                multimodal_embeds[0] = multimodal_embeds[0][0 : (max_patch + 1) - batch_pad]
            logger.debug(f'Unpad multimodal embedding to {multimodal_embeds[0].shape}.')

        text_embeds = self.text_embedding_layer(torch.tensor(tokens).to(self.text_embedding_layer.weight.device))
        b, n, c = text_embeds.shape
        text_embeds = text_embeds.reshape(b * n, c)
        image_embeds = multimodal_embeds[0]
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds).to(
                device=self.text_embedding_layer.weight.device, dtype=text_embeds.dtype
            )
        tokens = torch.tensor(tokens).reshape(b * n)
        selected = self.get_mm_mask(tokens)
        text_embeds[selected] = image_embeds.reshape(-1, c).to(dtype=text_embeds.dtype, device=text_embeds.device)
        text_embeds = text_embeds.reshape(b, n, c)

        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return text_embeds.to(self.dtype), kwargs
        # For numpy embedes, do not cast to self.dtype because `text_embeds` might be int, and self.dtype is float
        text_embeds = text_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {text_embeds.dtype}')

        return text_embeds, kwargs
