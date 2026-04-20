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
"""Define llava get_embeds class."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class LlavaGetEmbeds(BaseGetEmbedsHook):
    """LlavaGetEmbeds class that gets embeddings for llava model.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the LlavaGetEmbeds.
        forward(inputs): Forward pass that gets the input embeddings for llava model.
    """

    def __init__(self, config, **kwargs):
        """Initialize the LlavaGetEmbeds.

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
        logger.debug('Enter LlavaGetEmbeds forward')
        if custom_embeds is not None:
            return custom_embeds, kwargs
        if multimodal_embeds is None:
            logger.error('multimodal_embeds should not be None for llava get_embeds hook.')

        image_token_mask = self.get_mm_mask(tokens[0])
        image_indices = [i for i, x in enumerate(image_token_mask) if x]
        num_images = len(image_indices)
        if num_images == 0:
            logger.error('There is no image token in this prompt to generate multimodal embedding.')
        elif num_images > 1:
            logger.error('There are more than 1 image tokens in this prompt, which is not supported.')

        if num_images != len(multimodal_embeds):
            logger.error(
                f'Counted {num_images} image tokens in prompt tokens but got {len(multimodal_embeds)} image embeddings.'
            )

        tokens_to_embed = np.delete(tokens[0], image_indices)[None, :]

        text_embeds = self.text_embedding_layer(
            torch.tensor(tokens_to_embed).to(self.text_embedding_layer.weight.device)
        )
        image_embeds = multimodal_embeds[0]
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds).to(self.text_embedding_layer.weight.device)

        image_token_index = image_indices[0]
        logger.debug(f'image_token_index={image_token_index}')

        combined_embeds = torch.cat(
            [text_embeds[:, :image_token_index, :], image_embeds, text_embeds[:, image_token_index:, :]], dim=1
        )

        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return combined_embeds.to(self.dtype), kwargs
        # For numpy embeds, do not cast to self.dtype because `combined_embeds` might be int, and self.dtype is float
        combined_embeds = combined_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {combined_embeds.dtype}')
        return combined_embeds, kwargs
