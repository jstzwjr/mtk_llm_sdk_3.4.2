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
"""Define BiTA pre-getembed hook to adds draft tokens when conditions are met."""

import numpy as np

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class BiTAPreGetEmbedConfig(HookConfig):
    """Configuration class for BitaPreGetEmbed.

    This class extends the HookConfig to include additional configurations specific to BitaPreGetEmbed.

    Attributes:
        use_bita (bool): Whether to use BITA.
        task (str): The current task.
        bita_draft_length (int): The length of BITA draft tokens.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the BiTAPreGetEmbedConfig.

        Args:
            kwargs (dict, optional): Additional keyword arguments.
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.bita_draft_length = kwargs.pop('bita_inference_draft_length', None)
        if self.bita_draft_length is None:
            logger.error('[BiTAPreGetEmbedConfig]: Must provide bita_inference_draft_length in BitaPreGetEmbedConfig.')

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'bita_draft_length:  {self.bita_draft_length}')


class BiTAPreGetEmbed(BaseHook):
    """BitaPreGetEmbedHook class that adds draft tokens when conditions are met.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the BitaPreGetEmbedHook.
        forward(inputs, **kwargs): Forward pass.
    """

    def __init__(self, config: BiTAPreGetEmbedConfig):
        """Initialize the BitaPreGetEmbedHook.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: The input tokens.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            tuple: The processed inputs and kwargs.
        """
        add_bita_draft = kwargs.get('add_bita_draft', False)
        vocab_size = kwargs.get('vocab_size')

        if add_bita_draft:
            draft_token = np.array([[vocab_size + i for i in range(self.config.bita_draft_length)]])
            logger.debug(f'Adding {self.config.bita_draft_length} draft tokens to inputs')
            inputs = np.concatenate([inputs, draft_token], axis=1)

        return inputs, kwargs
