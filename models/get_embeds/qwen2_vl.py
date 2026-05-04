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
"""Define qwen2vl get_embeds class."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class Qwen2VLGetEmbeds(BaseGetEmbedsHook):
    """Class for Qwen2VLGetEmbeds.

    This class extends the BaseGetEmbedsHook to include functionalities specific to Qwen2-VL.

    Attributes:
        config (object): The configuration object.
        kwargs (dict, optional): Additional keyword arguments.

    Methods:
        forward: Forward pass to get embeddings.
    """

    def __init__(self, config, **kwargs):
        """Initialize the Qwen2VLGetEmbeds.

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
        logger.debug('Enter Qwen2VLGetEmbeds forward')
        if custom_embeds is not None:
            return custom_embeds, kwargs
        if multimodal_embeds is None:
            logger.error('multimodal_embeds should not be None for qwen2-vl get_embeds hook.')
        text_tokens = tokens
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
        text_embeds[selected] = image_embeds.reshape(-1, c).to(dtype=text_embeds.dtype)
        text_embeds = text_embeds.reshape(b, n, c)
        kwargs.update({'text_tokens': text_tokens})

        pipeline_type = kwargs.get('pipeline_type')
        if pipeline_type is None:
            logger.error('pipeline_type must be passed when forwarding Qwen2VLGetEmbeds.')
        if pipeline_type == 'quantized':
            return text_embeds.detach().cpu().numpy(), kwargs

        return text_embeds, kwargs
