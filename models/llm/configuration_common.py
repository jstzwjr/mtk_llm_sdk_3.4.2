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
"""Define configuration of common decoder only LLM."""

from ...utils import logger
from ..configuration_base import BaseLLMConfig


class CommonConfig(BaseLLMConfig):
    """Configuration class for common decoder-only LLMs.

    This class inherits from BaseLLMConfig and is used to set up the configuration
    for common decoder-only language models, including handling response, model parameters,
    and specific settings related to attention mechanisms and embeddings.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The size of the hidden layers.
        intermediate_size (int): The size of the intermediate layers.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        head_dim (float): The dimension of each attention head.
        num_key_value_heads (int): The number of key-value heads.
        norm (str): The type of normalization used ('RMSNorm' or 'LayerNorm').
        max_position_embeddings (int): The maximum number of position embeddings.
        ntk_scaling_factor (float): The scaling factor for NTK.
        rotary_emb_base (float): The base value for rotary embeddings.
        norm_eps (float): The epsilon value for normalization.
        bos_token_id (int): The ID of the beginning-of-sequence token.
        eos_token_id (int or list): The ID(s) of the end-of-sequence token.
        pad_token_id (int): The ID of the padding token.
        unk_token_id (int): The ID of the unknown token.
        ring_buffer (bool): A flag indicating whether to use a ring buffer.
        sliding_window_attention_size (int): The size of the sliding window for attention.
        use_stable_embedding (bool): A flag indicating whether to use stable embeddings.
        tie_word_embeddings (bool): A flag indicating whether to tie word embeddings.
        use_qk_norm (bool): A flag indicating whether to use QK normalization.
        tokenizer (str): The tokenizer used.
        embedding_key (str): The key for the embedding.
        mask_value (float): The value used for masking.
        mask_scaling_factors (list): List of scaling factors for per layer mask value.
        early_exit_index (int): The index for early exit.
        early_exit_num_layers (int): The number of layers for early exit.
    """

    def __init__(self, **kwargs):
        """Initializes the CommonConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            KeyError: If required configuration parameters are missing.
            RuntimeError: If certain conditions on configuration parameters are not met.
            ValueError: If invalid values are provided for certain parameters.
        """
        super().__init__(**kwargs)

        self.vocab_size = self.kwargs.pop('vocab_size', None)
        if self.vocab_size is None:
            logger.error('vocab_size is required but missing from config.json', err=KeyError)

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from config.json', err=KeyError)

        self.intermediate_size = self.kwargs.pop('intermediate_size', None)
        if self.intermediate_size is None:
            logger.error('intermediate_size is required but missing from config.json', err=KeyError)

        self.num_hidden_layers = self.kwargs.pop('num_hidden_layers', None)
        if self.num_hidden_layers is None:
            logger.error('num_hidden_layers is required but missing from config.json', err=KeyError)

        self.num_attention_heads = self.kwargs.pop('num_attention_heads', None)
        if self.num_attention_heads is None:
            logger.error('num_attention_heads is required but missing from config.json', err=KeyError)

        self.head_dim = self.kwargs.pop('head_dim', self.hidden_size // self.num_attention_heads)

        self.num_key_value_heads = self.kwargs.pop('num_key_value_heads', self.num_attention_heads)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            logger.error(
                f'num_attention_heads ({self.num_attention_heads}) must be exactly '
                f'divisible by num_key_value_heads ({self.num_key_value_heads})'
            )

        self.norm = self.kwargs.pop('norm', 'RMSNorm')
        if self.norm not in ['RMSNorm', 'LayerNorm']:
            logger.error('norm must be one of: RMSNorm (default) or LayerNorm', err=ValueError)

        self.max_position_embeddings = self.kwargs.pop('max_position_embeddings', None)
        if self.max_position_embeddings is None:
            logger.error('max_position_embeddings is required but is missing from config.json', err=KeyError)
        self.original_max_position_embeddings = kwargs.pop('original_max_position_embeddings', None)

        self.ntk_scaling_factor = self.kwargs.pop('ntk_scaling_factor', 1.0)

        self.rotary_emb_base = self.kwargs.pop('rotary_emb_base', self.kwargs.pop('rope_theta', self.rotary_emb_base))
        self.rope_scaling = self.kwargs.pop('rope_scaling', None)
        self.norm_eps = self.kwargs.pop('rms_norm_eps', self.kwargs.pop('norm_eps', 1e-6))

        self.bos_token_id = self.kwargs.pop('bos_token_id', 0)
        self.eos_token_id = self.kwargs.pop('eos_token_id', 1)
        self.pad_token_id = self.kwargs.pop('pad_token_id', 2)
        self.unk_token_id = self.kwargs.pop('unk_token_id', 0)

        self.ring_buffer = self.kwargs.pop('ring_buffer', True)
        self.sliding_window_attention_size = self.kwargs.pop('sliding_window_attention_size', 0)

        self.use_stable_embedding = self.kwargs.pop('use_stable_embedding', False)
        self.tie_word_embeddings = self.kwargs.pop('tie_word_embeddings', False)
        self.use_qk_norm = self.kwargs.pop('use_qk_norm', False)

        self.tokenizer = self.kwargs.pop('tokenizer', self.tokenizer)
        self.embedding_key = self.kwargs.pop('embedding_key', 'embed_tokens.weight')
        self.mask_value = self.kwargs.pop('mask_value', self.mask_value)
        self.mask_scaling_factors = self.kwargs.pop('mask_scaling_factors', [1] * self.num_hidden_layers)

        self.early_exit_index = self.kwargs.pop('early_exit_index', None)
        self.early_exit_num_layers = self.kwargs.pop('early_exit_num_layers', None)
        if (self.early_exit_index is None and self.early_exit_num_layers is not None) or (
            self.early_exit_index is not None and self.early_exit_num_layers is None
        ):
            logger.error(
                'Either both or neither early_exit_index and early_exit_num_layers need to be specified in config.json.'
            )

        self.sdpa_layers = []

        self.overture_dict = self.kwargs.pop('overture', None)
        self.infini_attention = self.kwargs.pop('infini_attention', False)
        self.infini_sink_size = self.kwargs.pop('infini_sink_size', 256)
        self.infini_window_size = self.kwargs.pop('infini_window_size', 256)
        self.infini_update = self.kwargs.pop('infini_update', 'delta')
        self.infini_kv_norm = self.kwargs.pop('infini_kv_norm', False)
        self.infini_segment_size = self.kwargs.pop('infini_segment_size', 2048)
        self.infini_use_combined_mlp = self.kwargs.pop('infini_use_combined_mlp', False)
        self.infini_use_alternative_calc = self.kwargs.pop('infini_use_alternative_calc', False)
        self.infini_max_gen_length = self.kwargs.pop('infini_max_gen_length', 0)
        self.bita = self.kwargs.pop('bita', False)
        self.bita_config = self.kwargs.pop('bita_config', False)

        self.use_split_mask = self.kwargs.get('use_split_mask', False)
        self.lm_head_pad_size = self.kwargs.pop('lm_head_pad_size', 0)

        self.cache_evict = self.kwargs.pop('cache_evict', {'method': ''})
        if self.cache_evict['method'] not in ['LocalSnapKV', 'GlobalSnapKV', '']:
            logger.error(
                'cacheEvict must be one of: ["LocalSnapKV", "GlobalSnapKV", ""]',
                err=ValueError,
            )

        # Extra IO
        self.extra_input = self.kwargs.pop('extra_input', {})
        self.extra_output = self.kwargs.pop('extra_output', {})
        # Init extra IO
        self.init_extra_input_ouput()

    def init_extra_input_ouput(self):
        """Initializes extra IO."""
        self.extra_input['sink_rope'] = self.extra_input.get('sink_rope', False)
        for name in self.extra_input:
            if name not in ['sink_rope']:
                logger.error(f'Only support extra_input: ["sink_rope"]. But get {name}.', err=ValueError)

        self.extra_output['attn_logits'] = self.extra_output.get('attn_logits', False)
        self.extra_output['attn_weights'] = self.extra_output.get('attn_weights', False)
        for name in self.extra_output:
            if name not in ['attn_logits', 'attn_weights']:
                logger.error(
                    f'Only support extra_output: ["attn_logits", "attn_weights"]. But get {name}.', err=ValueError
                )

        if self.cache_evict.get('method', '') == 'LocalSnapKV':
            self.extra_input['sink_rope'] = True
            self.extra_output['attn_logits'] = False
            self.extra_output['attn_weights'] = True
            self.cache_evict['in_graph'] = self.cache_evict.get(
                'in_graph', False
            )  # TODO: set default to True when localSnapKV in-graph is ready.

        elif self.cache_evict.get('method', '') == 'GlobalSnapKV':
            self.extra_input['sink_rope'] = False
            self.extra_output['attn_logits'] = False
            self.extra_output['attn_weights'] = True
            self.cache_evict['in_graph'] = self.cache_evict.get('in_graph', True)
            self.cache_evict['obs_window'] = self.cache_evict.get('obs_window', 32)

    def print_config(self):
        """Prints the configuration settings.

        This method prints the configuration settings, including model parameters,
        attention settings, and other relevant configurations.
        """
        logger.info(f'{self.model_type} config:')
        logger.info(f'Hidden size:                          {self.hidden_size}')
        logger.info(f'Intermediate size:                    {self.intermediate_size}')
        logger.info(f'Num layers:                           {self.num_hidden_layers}')
        logger.info(f'Num attention heads:                  {self.num_attention_heads}')
        logger.info(f'Head dim:                             {self.head_dim}')
        logger.info(f'Num KV heads:                         {self.num_key_value_heads}')
        logger.info(f'Max pos emb:                          {self.max_position_embeddings}')
        logger.info(f'Rope theta:                           {self.rotary_emb_base}')
        if self.ntk_scaling_factor != 1.0:
            logger.info(f'NTK scaling factor:                   {self.ntk_scaling_factor}')
        if self.rope_scaling is not None:
            logger.info(f'Original max pos emb                  {self.original_max_position_embeddings}')
            logger.info(f'Rope scaling config:                  {self.rope_scaling}')
        logger.info(f'Norm type:                            {self.norm}')
        logger.info(f'Norm epsilon:                         {self.norm_eps}')
        logger.info(f'BOS token id:                         {self.bos_token_id}')
        logger.info(f'EOS token id:                         {self.eos_token_id}')
        logger.info(f'PAD token id:                         {self.pad_token_id}')
        logger.info(f'UNK token id:                         {self.unk_token_id}')
        logger.info(f'Vocab size:                           {self.vocab_size}')
        logger.info(f'Use stable embedding:                 {self.use_stable_embedding}')
        logger.info(f'Tie word embeddings:                  {self.tie_word_embeddings}')
        logger.info(f'Use QK norm:                          {self.use_qk_norm}')
        logger.info(f'Ring buffer:                          {self.ring_buffer}')
        logger.info(f'Use BiTA:                             {self.bita}')
        if self.mask_value != -100.0:
            logger.info(f'Mask value:                           {self.mask_value}')
        if self.mask_scaling_factors is not None:
            logger.info(f'Mask Scaling Factors:                 {self.mask_scaling_factors}')
        if self.sliding_window_attention_size > 0:
            logger.info(f'SWA window size:                      {self.sliding_window_attention_size}')
        if self.tokenizer != 'default':
            logger.info(f'Tokenizer:                            {self.tokenizer}')
        if self.early_exit_index is not None:
            logger.info(f'Early exit index:                     {self.early_exit_index}')
            logger.info(f'Early exit layers:                    {self.early_exit_num_layers}')
        if self.infini_attention:
            logger.info('Infini Attention:                      True')
            logger.info(f'Infini segment size:                  {self.infini_segment_size}')
            logger.info(f'Infini sink size:                     {self.infini_sink_size}')
            logger.info(f'Infini window size:                   {self.infini_window_size}')
            logger.info(f'Infini update:                        {self.infini_update}')
            logger.info(f'Infini KV norm:                       {self.infini_kv_norm}')
            logger.info(f'Infini Combined MLP:                  {self.infini_use_combined_mlp}')
            logger.info(f'Infini Use alternative calculation:   {self.infini_use_alternative_calc}')
            logger.info(f'Infini Max Generation Length:         {self.infini_max_gen_length}')
        if self.use_split_mask:
            logger.info('Use split mask:                       True')
        if self.bita:
            logger.info(f'BiTA config:                            {self.bita_config}')
        if self.cache_evict and self.cache_evict['method'] != '':
            for key, value in self.cache_evict.items():
                logger.info(f'Cache Evict {key}:            {value}')
        for key, value in self.extra_input.items():
            logger.info(f'Extra Input:                          {key}: {value}')
        for key, value in self.extra_output.items():
            logger.info(f'Extra Output:                         {key}: {value}')
