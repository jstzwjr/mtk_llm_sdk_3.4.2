# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Whisper Preprocessor Configuration."""

from ...utils import logger
from ..configuration_base import BasePreprocessorConfig


class WhisperPreprocessorConfig(BasePreprocessorConfig):
    """Configuration class for the Whisper Preprocessor model.

    This class is used to store the configuration of a Whisper Preprocessor model.
    """

    def __init__(self, **kwargs):
        """Initializes the WhisperPreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            RuntimeError: If the model type is not 'whisper'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'whisper')
        if self.model_type != 'whisper':
            logger.error(f'Expected model_type to be whisper but got {self.model_type} instead')

        self.feature_size = self.kwargs.get('feature_size', 80)
        self.sampling_rate = self.kwargs.get('sampling_rate', 16000)
        self.return_attention_mask = self.kwargs.get('return_attention_mask', False)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} preprocessor config:')
        logger.info(f'Feature Size:    {self.feature_size}')
        logger.info(f'Sampling Rate:   {self.sampling_rate}')
        logger.info(f'Audio Attn Mask: {self.return_attention_mask}')
