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
"""Define base hook config class."""

from ..utils import logger
from ..utils.const import DEFAULT_AUDIO_TOKEN_ID, DEFAULT_IMAGE_TOKEN_ID, SUPPORTED_HOOKS
from .configuration_base import BaseConfig


class HookConfig(BaseConfig):
    """HookConfig class for handling hook configuration settings.

    Attributes:
        type (str): The type of the hook.
        name (str): The name of the hook.
        custom_path (str): The custom path for the hook.

    Methods:
        print_config(): Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.type = self.kwargs.pop('type')
        self.name = self.kwargs.pop('name', None)
        self.custom_path = self.kwargs.pop('custom_path', None)
        self.replace_existing = self.kwargs.pop('replace_existing', False)

        if self.replace_existing and self.type not in ['pre_tokenizer_hook', 'pre_getembed_hook']:
            logger.error(
                'Only hook of types `pre_tokenizer_hook` and `pre_getembed_hook` can be used for `replace_existing`, '
                f'but got {self.type}.',
                err=ValueError,
            )

        if self.name in SUPPORTED_HOOKS and self.custom_path is not None:
            logger.error(
                f'Received a pre-defined hook name ({self.name}) as well as a '
                f'custom hook path ({self.custom_path}). Please only use either, not both.'
            )

        if self.name is None and self.custom_path is None:
            self.name = 'passthrough'

        if self.kwargs.pop('verbose', True) and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')


class GetEmbedsHooksConfig(HookConfig):
    """GetEmbedsHooksConfig class for handling hook configuration settings.

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
        verbose = kwargs.pop('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)
        self.audio_token_ids = self.kwargs.pop('audio_token_ids', [DEFAULT_AUDIO_TOKEN_ID])
        if not isinstance(self.audio_token_ids, list):
            self.audio_token_ids = [self.audio_token_ids]
        self.image_token_ids = self.kwargs.pop('image_token_ids', [DEFAULT_IMAGE_TOKEN_ID])
        if not isinstance(self.image_token_ids, list):
            self.image_token_ids = [self.image_token_ids]

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        super().print_config()
        logger.info(f'audio_token_ids: {self.audio_token_ids}')
        logger.info(f'image_token_ids: {self.image_token_ids}')
