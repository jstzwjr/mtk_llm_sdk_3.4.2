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
"""Define Configuration od Baichuan model."""

from ...utils import logger
from .configuration_common import CommonConfig


class BaichuanConfig(CommonConfig):
    """Configuration class for the Baichuan model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the Baichuan model, including handling response, model type, and specific
    parameters related to attention mechanisms.

    Attributes:
        model_type (str): The type of the model, expected to be 'baichuan'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
        sparse_attn (bool): A flag indicating whether sparse attention is used.
        sparse_attn_num_head (int): The number of heads used in sparse attention.
    """

    def __init__(self, **kwargs):
        """Initializes the BaichuanConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'baichuan'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'baichuan')
        if self.model_type != 'baichuan':
            logger.error(f'Expected model_type to be baichuan but got {self.model_type} instead')

        self.fc_names = {
            'attn': {'name': 'self_attn', 'qkv': 'W_pack', 'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'o_proj'},
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

        self.sparse_attn = self.kwargs.pop('sparse_attn', False)
        self.sparse_attn_num_head = self.kwargs.pop('sparse_attn_num_head', 4)

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Prints the configuration settings.

        This method overrides the print_config method from the CommonConfig class
        to include additional settings specific to the Baichuan model.
        """
        super().print_config()
        logger.info(f'Sparse attention:       {self.sparse_attn}')
        if self.sparse_attn:
            logger.info(f'Sparse attention heads: {self.sparse_attn_num_head}')
