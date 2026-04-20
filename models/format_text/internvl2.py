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
"""Define InternVL2 format_text class."""

from ...utils import const, logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class InternVL2FormatTextConfig(HookConfig):
    """InternVL2FormatTextConfig class that extend HookConfig."""

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.image_token = self.kwargs.pop('image_token', const.INTERNVL2_IMG_CONTEXT_TOKEN)
        self.image_start_token = self.kwargs.pop('image_start_token', const.INTERNVL2_IMG_START_TOKEN)
        self.image_end_token = self.kwargs.pop('image_end_token', const.INTERNVL2_IMG_END_TOKEN)
        self.num_image_token = self.kwargs.pop('num_image_token', None)
        if self.num_image_token is None:
            logger.error('num_image_token is required but missing in InternVL2FormatTextConfig.')


class InternVL2FormatText(BaseHook):
    """InternVL2FormatText class that formats text prompts for InternVL2 models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the InternVL2FormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config: InternVL2FormatTextConfig, **kwargs):
        """Initialize the InternVL2FormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_token = self.config.image_token
        self.image_start_token = self.config.image_start_token
        self.image_end_token = self.config.image_end_token
        self.num_image_token = self.config.num_image_token

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The formatted InternVL2 prompt.
        """
        num_image_token = self.num_image_token
        num_images = kwargs.get('internvl2_num_images', 1)
        num_patches_list = kwargs.get('num_patches_list')
        if num_image_token is None:
            logger.error('Must pass num_image_token when formatting InternVL2 prompt.', err=ValueError)
        if num_patches_list is None:
            num_patches_list = [num_images]
        logger.debug('Enter LlavaFormatText forward')
        logger.debug(f'Text before format: {text}')

        text = const.DEFAULT_IMAGE_TOKEN + '\n' + text
        text_to_log = text
        for num_patches in num_patches_list:
            image_tokens = (
                self.image_start_token + self.image_token * num_image_token * num_patches + self.image_end_token
            )
            text = text.replace('<image>', image_tokens, 1)
        image_token_to_log = (
            ' * '
            + self.image_start_token
            + self.image_token
            + str(int(num_image_token * num_patches))
            + self.image_end_token
        )
        text_to_log = text_to_log.replace('<image>', image_token_to_log, 1)
        logger.debug(f'Text after format: {text_to_log}')
        kwargs.update({'internvl2_image_token': self.image_token, 'prompt_to_log': text_to_log})

        return text, kwargs
