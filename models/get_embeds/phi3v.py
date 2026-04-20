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
from ..configuration_hook import GetEmbedsHooksConfig
from ..modeling_hook_base import BaseGetEmbedsHook


class Phi3VGetEmbedsConfig(GetEmbedsHooksConfig):
    """Phi3VGetEmbedsConfig class for handling hook configuration settings.

    Attributes:
        audio_token_ids: The audio token ids.
        image_token_ids: The image token ids.

    Methods:
        print_config(): Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs.pop('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.intrinsic_num_img_tokens = self.kwargs.pop('num_img_tokens', 144)
        self.vocab_size = self.kwargs.pop('vocab_size', 32064)


class Phi3VGetEmbeds(BaseGetEmbedsHook):
    """Class for Phi3VGetEmbeds.

    This class extends the BaseGetEmbedsHook to include functionalities specific to Phi3-V.

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
        hd_transform = kwargs.get('phi3v_hd_transform')
        if hd_transform is None:
            logger.error('Must pass phi3v_hd_transform when forwarding Phi3VGetEmbeds.', err=ValueError)
        selected_g_values = kwargs.get('phi3v_selected_g_values')
        if not hd_transform and selected_g_values is None:
            logger.error(
                'Must pass phi3v_selected_g_values when phi3v_hd_transform is False when forwarding Phi3VGetEmbeds.',
                err=ValueError,
            )
        phi3v_image_batch_features = kwargs.get('phi3v_image_batch_features')
        if phi3v_image_batch_features is None:
            logger.warning('Must pass phi3v_image_batch_features when forwarding Phi3vPreProjector.')
            num_img_tokens = None
        else:
            num_img_tokens = phi3v_image_batch_features.get('num_img_tokens', None)
        if num_img_tokens is None:
            logger.error(
                'phi3v_image_batch_features must contains phi3v_num_img_tokens when forwarding Phi3VGetEmbeds.',
                err=ValueError,
            )
        intrinsic_num_img_tokens = self.config.intrinsic_num_img_tokens
        vocab_size = self.config.vocab_size

        tokens = torch.tensor(tokens)
        MAX_INPUT_ID = int(1e9)  # noqa: N806
        with torch.no_grad():
            positions = torch.nonzero((tokens < 0) & (tokens > -MAX_INPUT_ID), as_tuple=False)

        tokens.clamp_min_(0).clamp_max_(vocab_size)
        text_embeds = self.text_embedding_layer(tokens.to(self.text_embedding_layer.weight.device))

        pipeline_type = kwargs.get('pipeline_type')
        logger.debug(f'pipeline_type: {pipeline_type}.')
        if pipeline_type == 'quantized':
            original_batch_size = kwargs.get('phi3v_original_batch_size')
            if original_batch_size is None:
                logger.error('original_batch_size must be passed when inference quantized Phi3-V.')
            max_patch = 16
            batch_pad = (max_patch + 1) - original_batch_size
            if original_batch_size < (max_patch + 1):
                multimodal_embeds[0] = multimodal_embeds[0][0 : (max_patch + 1) - batch_pad]
                logger.debug(f'Unpad multimodal embedding to {multimodal_embeds[0].shape}.')
        image_embeds = multimodal_embeds[0]
        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds).to(self.text_embedding_layer.weight.device)

        if hd_transform:
            idx = 0
            for i, cnt in enumerate(num_img_tokens):
                text_embeds[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                    image_embeds[i].to(text_embeds.dtype).to(text_embeds.device)
                )
                idx += cnt
        else:
            idx = 0
            if len(selected_g_values) * intrinsic_num_img_tokens != len(image_embeds):
                logger.error(
                    f'len(selected_g_values) * intrinsic_num_img_tokens = '
                    f'{len(selected_g_values) * intrinsic_num_img_tokens},'
                    f' len(image_embeds) = {len(image_embeds)}',
                    err=ValueError,
                )
            for i, _g in enumerate(selected_g_values):
                cnt = intrinsic_num_img_tokens
                text_embeds[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                    image_embeds[i * cnt : (i + 1) * cnt].to(text_embeds.dtype).to(text_embeds.device)
                )
                idx += cnt

        if isinstance(self.dtype, torch.dtype):
            logger.debug(f'Return embeds dtype: {self.dtype}')
            return text_embeds.to(self.dtype), kwargs
        # For numpy embedes, do not cast to self.dtype because `text_embeds` might be int, and self.dtype is float
        text_embeds = text_embeds.cpu().numpy()
        logger.debug(f'Return embeds dtype: {text_embeds.dtype}')

        return text_embeds, kwargs
