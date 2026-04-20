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
"""Configuration of HunYuan."""

from ...utils import logger
from .configuration_common import CommonConfig


class HunYuanConfig(CommonConfig):
    """Configuration class for the HunYuan model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the HunYuan model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'hunyuan'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
    """

    def __init__(self, **kwargs):
        """Initializes the HunYuanConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'hunyuan'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'hunyuan')
        if self.model_type != 'hunyuan':
            logger.error(f'Expected model_type to be hunyuan but got {self.model_type} instead')

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

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()
