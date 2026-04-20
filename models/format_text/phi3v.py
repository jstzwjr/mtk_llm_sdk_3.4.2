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

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class Phi3VFormatText(BaseHook):
    """Phi3VFormatText class that formats text prompts for Phi3V models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Phi3VFormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config: HookConfig, **kwargs):
        """Initialize the Phi3VFormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The formatted InternVL2 prompt.
        """
        logger.debug('Enter Phi3VFormatText forward')
        logger.debug(f'Text before format: {text}')
        count = kwargs.get('phi3v_img_count', 1)
        text = f'<|image_{count}|>' + '\n' + text
        kwargs.update({'phi3v_img_token_idx': -count})
        logger.debug(f'Text after format: {text}')
        return text, kwargs
