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
"""Configuration of InternLM2."""

from ...utils import logger
from .configuration_common import CommonConfig


class InternLM2Config(CommonConfig):
    """Configuration class for the InternLM2 model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the InternLM2 model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'internlm2'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
        embedding_key (str): The key for the embedding.
    """

    def __init__(self, **kwargs):
        """Initializes the InternLM2Config class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'internlm2'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'internlm2')
        if self.model_type != 'internlm2':
            logger.error(f'Expected model_type to be internlm2 but got {self.model_type} instead')

        self.fc_names = {
            'attn': {'name': 'attention', 'qkv': 'wqkv', 'o': 'wo'},
            'mlp': {'name': 'feed_forward', 'gate': 'w1', 'up': 'w3', 'down': 'w2', 'gateup': 'w13'},
            'tail': {'name': 'output'},
        }
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': 'attention_norm',
            'post_attn': 'ffn_norm',
            'final': 'norm',
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }

        self.embedding_key = 'tok_embeddings'
        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()
