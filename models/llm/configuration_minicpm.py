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
"""Configuration of MiniCPM."""

from ...utils import logger
from .configuration_common import CommonConfig


class MinicpmConfig(CommonConfig):
    """Configuration class for the MiniCPM model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the MiniCPM model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'minicpm'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
        llama_format (bool): A flag indicating whether the model uses the LLaMA format.
        scale_emb (float): The scale of the embeddings.
        dim_model_base (int): The base dimension of the model.
        scale_depth (float): The scale of the depth.
    """

    def __init__(self, **kwargs):
        """Initializes the MinicpmConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'minicpm'.
            KeyError: If required configuration parameters are missing.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'minicpm')
        if self.model_type != 'minicpm':
            logger.error(f'Expected model_type to be minicpm but got {self.model_type} instead')

        self.fc_names = {
            'attn': {
                'name': 'self_attn',
                'qkv': 'qkv_proj',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'o_proj',
            },
            'mlp': {'name': 'mlp', 'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj', 'gateup': 'gate_up_proj'},
            'tail': {'name': 'lm_head'},
        }
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': 'input_layernorm',
            'post_attn': 'post_attention_layernorm',
            'final': 'norm',
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }

        self.llama_format = self.kwargs.pop('llama_format', False)

        self.scale_emb = self.kwargs.pop('scale_emb', None)
        if self.scale_emb is None:
            logger.error('scale_emb is required but missing from config.json', err=KeyError)

        self.dim_model_base = self.kwargs.pop('dim_model_base', None)
        if self.dim_model_base is None and not self.llama_format:
            logger.error('dim_model_base is required for non-llama format but missing from config.json', err=KeyError)

        self.scale_depth = self.kwargs.pop('scale_depth', None)
        if self.scale_depth is None and not self.llama_format:
            logger.error('scale_depth is required for non-llama format but missing from config.json', err=KeyError)

        # Replace rope_type with type for MiniCPM4
        if self.rope_scaling is not None:
            self.rope_scaling['type'] = (
                self.rope_scaling['rope_type'] if 'type' not in self.rope_scaling else self.rope_scaling['type']
            )
            self.original_max_position_embeddings = (
                self.rope_scaling['original_max_position_embeddings']
                if self.original_max_position_embeddings is None
                else self.original_max_position_embeddings
            )

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration of the MiniCPM model."""
        super().print_config()
        logger.info(f'Llama format:        {self.llama_format}')
        logger.info(f'Embedding scale:     {self.scale_emb}')
        if not self.llama_format:
            logger.info(f'Scale depth:         {self.scale_depth}')
            logger.info(f'Dim model base:      {self.dim_model_base}')
