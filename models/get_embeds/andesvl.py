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
"""Define AndesVL get_embeds class."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class AndesVLGetEmbeds(BaseGetEmbedsHook):
    """Class for AndesVLGetEmbeds.

    This class extends the BaseGetEmbedsHook to include functionalities specific to AndesVL.

    Attributes:
        config (object): The configuration object.
        kwargs (dict, optional): Additional keyword arguments.

    Methods:
        forward: Forward pass to get embeddings.
    """

    def __init__(self, config, **kwargs):
        """Initialize the AndesVLGetEmbeds.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config=config, **kwargs)
        logger.debug(f'image_token_ids={self.mm_token_ids}')

    def forward(self, tokens=None, multimodal_embeds=None, custom_embeds=None, **kwargs):
        """Forward pass to get embeddings."""
        image_token_id = kwargs.get('AndesVL_image_token_id')
        if image_token_id is None:
            logger.error('Must pass AndesVL_image_token_id when forwarding AndesVLGetEmbeds.', err=RuntimeError)
        self.mm_token_ids.append(image_token_id)
        logger.debug(f'mm_token_ids: {self.mm_token_ids}')
        tokens = torch.tensor(tokens).to(self.text_embedding_layer.weight.device)
        text_embeds = self.text_embedding_layer(tokens)
        n_image_tokens = (tokens == image_token_id).sum().item()
        image_embeds = multimodal_embeds[0]
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            logger.error(
                f'Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}',
                err=ValueError,
            )

        image_mask = (tokens == image_token_id).unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds).to(text_embeds.device, text_embeds.dtype)
        text_embeds = text_embeds.masked_scatter(image_mask, image_embeds)

        return text_embeds, kwargs
