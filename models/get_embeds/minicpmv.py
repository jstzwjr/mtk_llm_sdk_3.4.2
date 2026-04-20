# Copyright (C) 2024 MediaTek Inc. All rights reserved.
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
"""Define MinicpmV get_embeds class."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class MinicpmVGetEmbeds(BaseGetEmbedsHook):
    """MinicpmVGetEmbeds class that gets embeddings for minicpmv model.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the MinicpmVGetEmbeds.
        forward(inputs): Forward pass that gets the input embeddings for minicpmv model.
    """

    def __init__(self, config, **kwargs):
        """Initialize the MinicpmVGetEmbeds.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config=config, **kwargs)
        logger.debug(f'image_token_ids={self.mm_token_ids}')

    def forward(self, tokens=None, multimodal_embeds=None, custom_embeds=None, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            tokens: The input tokens to be embedded.
            multimodal_embeds: The encoder outputs.
            custom_embeds: The custom embeds to override this function with.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The token embeddings.
        """
        logger.debug('Enter MinicpmVGetEmbeds forward')
        if custom_embeds is not None:
            return custom_embeds, kwargs
        if multimodal_embeds is None:
            logger.error('multimodal_embeds should not be None for MinicpmVGetEmbeds get_embeds hook.')

        text_embeds = self.text_embedding_layer(torch.tensor(tokens).to(self.text_embedding_layer.weight.device))
        logger.debug(f'text_embeds: {text_embeds.shape}')

        b, n, c = text_embeds.shape
        text_embeds = text_embeds.reshape(b * n, c)
        image_embeds = multimodal_embeds[0]
        tokens = torch.tensor(tokens).reshape(b * n)

        selected = self.get_mm_mask(tokens)
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds).to(
                device=self.text_embedding_layer.weight.device, dtype=text_embeds.dtype
            )

        text_embeds = text_embeds.cpu()
        image_embeds = image_embeds.cpu()

        text_embeds[selected] = image_embeds.reshape(-1, c)
        text_embeds = text_embeds.reshape(b, n, c).to(self.text_embedding_layer.weight.device)

        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return text_embeds.to(self.dtype), kwargs
        # For numpy embeds, do not cast to self.dtype because `combined_embeds` might be int, and self.dtype is float
        text_embeds = text_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {text_embeds.dtype}')
        return text_embeds, kwargs
