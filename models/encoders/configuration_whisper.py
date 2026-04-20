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
"""Configuration of Whisper."""

from ...utils import logger
from ..configuration_base import BaseAudioEncoderConfig


class WhisperEncoderConfig(BaseAudioEncoderConfig):
    """Configuration class for the Whisper model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the Whisper model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'whisper'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
    """

    def __init__(self, **kwargs):
        """Initializes the WhisperConfig class.

        Args:
            response_handler (optional): A handler for responses, default is None.
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'whisper'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'whisper')
        if self.model_type != 'whisper':
            logger.error(f'Expected model_type to be whisper but got {self.model_type} instead')

        self.intermediate_size = self.kwargs.pop('encoder_ffn_dim', None)
        if self.intermediate_size is None:
            logger.error('intermediate_size is required but missing from config.json', err=KeyError)

        self.num_hidden_layers = self.kwargs.pop('encoder_layers', None)
        if self.num_hidden_layers is None:
            logger.error('num_hidden_layers is required but missing from config.json', err=KeyError)

        self.num_attention_heads = self.kwargs.pop('encoder_attention_heads', None)
        if self.num_attention_heads is None:
            logger.error('num_attention_heads is required but missing from config.json', err=KeyError)

        self.hidden_size = self.kwargs.pop('d_model', None)
        if self.hidden_size is None:
            logger.error('d_model is required but missing from config.json', err=KeyError)

        self.decoder_num_layers = self.kwargs.pop('decoder_layers', None)
        if self.decoder_num_layers is None:
            logger.error('decoder_layers is required but missing from config.json', err=KeyError)

        self.num_mel_bins = self.kwargs.pop('num_mel_bins', None)
        if self.num_mel_bins is None:
            logger.error('num_mel_bins is required but missing from config.json', err=KeyError)

        self.max_source_positions = self.kwargs.pop('max_source_positions', None)
        if self.max_source_positions is None:
            logger.error('max_source_positions is required but missing from config.json', err=KeyError)

        self.fc_names = {
            'attn': {
                'name': 'self_attn',
                'qkv': 'qkv_proj',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'out_proj',
            },
            'mlp': {'name': None, 'up': 'fc1', 'down': 'fc2'},
            'cross_attn': {
                'name': 'encoder_attn',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'out_proj',
            },
            'tail': {'name': 'lm_head'},
        }
        self.norm = self.kwargs.pop('norm', 'LayerNorm')
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': 'self_attn_layer_norm',
            'post_attn': 'final_layer_norm',
            'final': 'norm',
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }
        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration parameters."""
        logger.info(f'{self.model_type} encoder config:')
        logger.info(f'Encoder ffn size:        {self.intermediate_size}')
        logger.info(f'Encoder layers:          {self.num_hidden_layers}')
        logger.info(f'Encoder Attention head:  {self.num_attention_heads}')
        logger.info(f'Encoder d_model:         {self.hidden_size}')
        logger.info(f'Decoder layers:          {self.decoder_num_layers}')
        logger.info(f'Num mel bins:            {self.num_mel_bins}')
        logger.info(f'Max source position:     {self.max_source_positions}')
