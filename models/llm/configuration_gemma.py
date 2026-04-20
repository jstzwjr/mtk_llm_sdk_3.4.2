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
"""Define configuration of Gemma, Gemma2 and Gemma3."""

from ...utils import logger
from .configuration_common import CommonConfig


class GemmaConfig(CommonConfig):
    """Configuration class for the Gemma model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the Gemma model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'gemma'.
        tie_word_embeddings (bool): A flag indicating whether to tie word embeddings.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
    """

    def __init__(self, **kwargs):
        """Initializes the GemmaConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'gemma'.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type')
        if self.model_type not in ['gemma', 'gemma2', 'gemma3']:
            logger.error(
                f'Expected model_type to be gemma, gemma2 or gemma3 but got {self.model_type} instead', err=TypeError
            )
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
            'pre_mlp': 'pre_feedforward_layernorm',
            'post_mlp': 'post_feedforward_layernorm',
            'final': 'norm',
            'query': 'q_norm',
            'key': 'k_norm',
        }

        if self.model_type == 'gemma2':
            self.attn_logit_softcapping = kwargs.pop('attn_logit_softcapping', 50.0)
            self.final_logit_softcapping = kwargs.pop('final_logit_softcapping', 30.0)
            self.query_pre_attn_scalar = kwargs.pop('query_pre_attn_scalar', 256)
        elif self.model_type == 'gemma3':
            self.sliding_window_attention_size = kwargs.pop('sliding_window_attention_size', 1024)
            self.global_local_attention_pattern = kwargs.pop('global_local_attention_pattern', None)
            self.rope_local_base_freq = kwargs.pop('rope_local_base_freq', 10000)
            self.query_pre_attn_scalar = kwargs.pop('query_pre_attn_scalar', 256)
            self.use_vision = kwargs.pop('use_vision', False)
            self.use_qk_norm = kwargs.pop('use_qk_norm', True)
            self.use_res_clamp = kwargs.pop('use_res_clamp', True)
            if self.use_split_mask:
                logger.error('Gemma3 does not support split mask.')

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """logger.info the configuration of Gemma model."""
        super().print_config()
        if self.model_type == 'gemma3':
            logger.info(f'sliding_window_attention_sizes:        {self.sliding_window_attention_size}')
            logger.info(f'global_local_attention_pattern:        {self.global_local_attention_pattern}')
            logger.info(f'rope_local_base_freq:        {self.rope_local_base_freq}')
            logger.info(f'use_vision:                            {self.use_vision}')
            logger.info(f'use_res_clamp:                            {self.use_res_clamp}')
