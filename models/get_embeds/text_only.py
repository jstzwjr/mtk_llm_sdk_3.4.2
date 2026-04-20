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
"""Define text-only get_embeds class."""

import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class TextOnlyGetEmbeds(BaseGetEmbedsHook):
    """TextOnlyGetEmbeds class that gets text-only token embeddings.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the TextOnlyGetEmbeds.
        forward(inputs): Forward pass that embeds the tokens.
    """

    def __init__(self, config, **kwargs):
        """Initialize the TextOnlyGetEmbeds.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config=config, **kwargs)

    def forward(self, tokens=None, multimodal_embeds=None, custom_embeds=None, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            tokens: The input tokens to be embedded.
            multimodal_embeds: multimodal embeddings. Expect to be None.
            custom_embeds: The custom embeds to override this function with.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The token embeddings.
        """
        logger.debug('Enter TextOnlyGetEmbeds forward')

        if custom_embeds is not None:
            if isinstance(self.dtype, torch.dtype):
                logger.debug(f'Return embeds dtype: {self.dtype}')
                return torch.tensor(custom_embeds).to(self.dtype), kwargs
            return custom_embeds, kwargs

        combined_embeds = self.text_embedding_layer(torch.tensor(tokens).to(self.text_embedding_layer.weight.device))
        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return combined_embeds.to(self.dtype), kwargs
        # For numpy embeds, do not cast to self.dtype because `combined_embeds` might be int, and self.dtype is float
        combined_embeds = combined_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {combined_embeds.dtype}')
        return combined_embeds, kwargs
