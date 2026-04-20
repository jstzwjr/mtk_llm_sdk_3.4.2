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
"""Define qwen2.5vl get_embeds class."""

import torch

from ...utils import logger
from .qwen2_vl import Qwen2VLGetEmbeds


class Qwen2_5VLGetEmbeds(Qwen2VLGetEmbeds):  # noqa: N801
    """Class for Qwen2_5VLGetEmbeds.

    This class extends the Qwen2VLGetEmbeds to include functionalities specific to Qwen2.5-VL.

    Attributes:
        config (object): The configuration object.
        kwargs (dict, optional): Additional keyword arguments.

    Methods:
        forward: Forward pass to get embeddings.
    """

    def __init__(self, config, **kwargs):
        """Initialize the Qwen2_5VLGetEmbeds.

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
        logger.debug('Enter Qwen2_5VLGetEmbeds forward')
        if custom_embeds is not None:
            return custom_embeds, kwargs
        if multimodal_embeds is None:
            logger.error('multimodal_embeds should not be None for qwen2.5-vl get_embeds hook.')

        # Different from models.get_embeds.qwen2_vl.Qwen2VLGetEmbeds
        # Argsort the encoder output
        window_index = kwargs.get('qwen2_5vl_vision_window_index')
        if window_index is None:
            logger.error(
                'qwen2_5vl_vision_window_index must be passed when forwarding Qwen2_5VLGetEmbeds', err=ValueError
            )
        reverse_indices = torch.argsort(window_index)
        logger.debug(f'reverse_indices: {reverse_indices}')
        for i in range(len(multimodal_embeds)):
            logger.debug(f'{i} - {multimodal_embeds[i].shape}')
            multimodal_embeds[i] = multimodal_embeds[i][reverse_indices, :]
            logger.debug(f'{i} - {multimodal_embeds[i].shape}')

        return super().forward(
            tokens=tokens, multimodal_embeds=multimodal_embeds, custom_embeds=custom_embeds, **kwargs
        )
