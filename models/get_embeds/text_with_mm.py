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
"""Define get_embeds class for text with mm."""

import numpy as np
import torch

from ...utils import logger
from ..modeling_hook_base import BaseGetEmbedsHook


class TextWithMMGetEmbeds(BaseGetEmbedsHook):
    """TextWithMMGetEmbeds class that gets text-only token embeddings.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the TextWithMMGetEmbeds.
        forward(inputs): Forward pass that embeds the tokens.
    """

    def __init__(self, config, **kwargs):
        """Initialize the TextWithMMGetEmbeds.

        Args:
            config (HookConfig): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config=config, **kwargs)
        self.mm_tokens = [config.audio_token_ids, config.image_token_ids]

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
        logger.debug('Enter TextWithMMGetEmbeds forward')
        if multimodal_embeds is None:
            logger.error('Text_with_MM requires multimodal_embeds as input!')

        boolean_array = np.isin(tokens, self.mm_tokens)
        mm_indices = np.where(boolean_array)[0] + 1
        assert len(multimodal_embeds) == len(mm_indices)

        for i in range(len(mm_indices)):
            np.delete(tokens, mm_indices[i], axis=-1)

        if custom_embeds is not None:
            return custom_embeds, kwargs
        tokens = [np.delete(tokens, mm_indices)]
        text_embeds = self.text_embedding_layer(torch.tensor(tokens).to(self.text_embedding_layer.weight.device))
        split_input = np.split(text_embeds, mm_indices, axis=1)

        for i in range(len(multimodal_embeds)):
            if 'audio_output_lengths' in kwargs:
                if isinstance(multimodal_embeds[i], torch.Tensor):
                    embed = torch.split(multimodal_embeds[i], int(kwargs['audio_output_lengths'][i]), dim=1)[0]
                else:
                    embed = torch.tensor(
                        np.split(multimodal_embeds[i], [int(kwargs['audio_output_lengths'][i])], axis=1)[0]
                    ).to(self.text_embedding_layer.weight.device)
            else:
                embed = multimodal_embeds[i]
            split_input.insert(2 * i + 1, embed)

        combined_embeds = torch.cat(split_input, axis=1)

        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return combined_embeds.to(self.dtype), kwargs

        combined_embeds = combined_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {combined_embeds.dtype}')
        return combined_embeds, kwargs
