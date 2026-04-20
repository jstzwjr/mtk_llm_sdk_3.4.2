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
"""Define AndesVL format_text class."""

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class AndesVLFormatTextConfig(HookConfig):
    """AndesVLFormatTextConfig class that extend HookConfig."""

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.image_token_id = self.kwargs.pop('image_token_id', 151654)
        self.image_token = self.kwargs.pop('image_token', '<|vision_pad|>')
        self.video_token = self.kwargs.pop('video_token', '<|vision_pad|>')
        self.vision_start_token = self.kwargs.pop('vision_start_token', '<img>')
        self.vision_end_token = self.kwargs.pop('vision_end_token', '</img>')
        self.vision_start_token_id = self.kwargs.pop('vision_start_token_id', 151665)
        self.vision_end_token_id = self.kwargs.pop('vision_end_token_id', 151666)


class AndesVLFormatText(BaseHook):
    """AndesVLFormatText class that formats text prompts for AndesVL models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the AndesVLFormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config: AndesVLFormatTextConfig, **kwargs):
        """Initialize the AndesVLFormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_token = self.config.image_token

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The formatted InternVL2 prompt.
        """
        logger.debug('Enter AndesVLFormatText.')
        logger.debug(f'Text before formatted: {text}')
        image_tokens = kwargs.get('AndesVL_image_tokens')
        if image_tokens is None:
            logger.error('Must pass AndesVL_image_tokens when forwarding AndesVLFormatText.', err=RuntimeError)
        text_to_log = text
        text = self.image_token + text
        index = 0
        placeholder_count = 0
        if not isinstance(text, list):
            text = [text]
        for i in range(len(text)):
            while self.image_token in text[i]:
                text[i] = text[i].replace(self.image_token, '<|placeholder|>' * (image_tokens[index]), 1)
                placeholder_count += image_tokens[index]
                index += 1
            text[i] = text[i].replace('<|placeholder|>', self.image_token)
        text_to_log = self.image_token + f' * {placeholder_count}' + text_to_log
        logger.debug(f'Text after formatted: {text_to_log}')
        kwargs.update({'AndesVL_image_token_id': self.config.image_token_id})
        return text[0], kwargs
