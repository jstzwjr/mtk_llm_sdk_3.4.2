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
"""Configuration of Qwen."""

from ...utils import logger
from .configuration_common import CommonConfig


class QwenConfig(CommonConfig):
    """Configuration class for the Qwen model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the Qwen model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be a variant of 'qwen'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
        embedding_key (str): The key for the embedding.
    """

    def __init__(self, **kwargs):
        """Initializes the QwenConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not a variant of 'qwen'.
        """
        embedding_key = kwargs.get(
            'embedding_key', 'wte.weight' if kwargs['model_type'] == 'qwen' else 'embed_tokens.weight'
        )

        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', None)
        self.sdpa_layers = self.kwargs.pop('sdpa_layers', [])
        self.sdpa_layers_dummy_scale = self.kwargs.pop('sdpa_layers_dummy_scale', 0.5)
        if self.model_type not in ['qwen', 'qwen1.5', 'qwen2', 'qwen3']:
            logger.error(f'Expected model_type to be a qwen variant but got {self.model_type} instead')

        self.use_split_mask = self.kwargs.pop('use_split_mask', True)

        if self.model_type == 'qwen':
            self.fc_names = {
                'attn': {'name': 'attn', 'qkv': 'c_attn', 'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'c_proj'},
                'mlp': {'name': 'mlp', 'gate': 'w1', 'up': 'w2', 'down': 'c_proj', 'gateup': 'w12'},
                'tail': {'name': 'lm_head'},
            }
            self.norm_names = {
                'stable_embedding': 'embed_layer_norm',
                'input': 'ln_1',
                'post_attn': 'ln_2',
                'final': 'ln_f',
                'query': 'query_layernorm',
                'key': 'key_layernorm',
            }
        else:
            q_norm_name = 'q_norm' if self.model_type == 'qwen3' else 'query_layernorm'
            k_norm_name = 'k_norm' if self.model_type == 'qwen3' else 'key_layernorm'
            self.fc_names = {
                'attn': {
                    'name': 'self_attn',
                    'qkv': 'qkv_proj',
                    'q': 'q_proj',
                    'k': 'k_proj',
                    'v': 'v_proj',
                    'o': 'o_proj',
                },
                'mlp': {
                    'name': 'mlp',
                    'gate': 'gate_proj',
                    'up': 'up_proj',
                    'down': 'down_proj',
                    'gateup': 'gate_up_proj',
                },
                'tail': {'name': 'lm_head'},
            }
            self.norm_names = {
                'stable_embedding': 'embed_layer_norm',
                'input': 'input_layernorm',
                'post_attn': 'post_attention_layernorm',
                'final': 'norm',
                'query': q_norm_name,
                'key': k_norm_name,
            }

        self.embedding_key = embedding_key

        # Flag to differentiate between text-only Qwen2 and Qwen2VL
        self.is_vl = self.kwargs.pop('is_vl', False)

        if self.model_type == 'qwen':
            if self.is_vl:
                logger.error('`is_vl` is only supported by qwen2 model.')
            if self.rope_scaling is not None:
                logger.error('`rope_scaling` is only supported by qwen2 model.')

        if self.kwargs.pop('verbose', True):
            self.print_config()
            logger.info(f'SDPA layers:             {self.sdpa_layers}')
            logger.info(f'SDPA layers dummy scale: {self.sdpa_layers_dummy_scale}')
            self.print_unused_kwargs()
