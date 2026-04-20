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
"""Define Llava format_text class."""

from ...utils import const, logger
from ..modeling_hook_base import BaseHook


class LlavaFormatText(BaseHook):
    """LlavaFormatText class that formats text prompts for Llava models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the LlavaFormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config, **kwargs):
        """Initialize the LlavaFormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_token = kwargs.pop('image_token', const.DEFAULT_IMAGE_TOKEN)

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The formatted LLaVA prompt.
        """
        logger.debug('Enter LlavaFormatText forward')
        logger.debug(f'Text before format: {text}')
        formatted_text = f'{self.image_token}\n{text}'
        logger.debug(f'Text after format: {formatted_text}')
        return formatted_text, kwargs
