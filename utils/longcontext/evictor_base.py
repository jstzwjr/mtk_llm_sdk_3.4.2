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
"""Evictor Bass Class."""

from abc import ABC, abstractmethod


class EvictorBase(ABC):
    """Base class of Evictor."""

    def __init__(self, config, cache_size, llm_layers_per_chunk) -> None:
        """Base class of Evictor.

        Attributes:
            head_dim (int): Dimension of each attention head.
            num_heads (int): Number of attention heads.
            num_key_value_heads (int): Number of key-value heads.
            num_key_value_groups (int): Number of key-value groups.
            num_hidden_layers (int): Number of hidden layers.
            llm_layers_per_chunk (list): Layers per chunk in the model.
            cache_size (int): Size of the cache.
            num_token (int): Number of tokens.
            max_capacity_prompt (int): Maximum capacity of the prompt.
            prompt_length (int): Length of the prompt.
            sink_rope (bool): Sink rope flag.
            attn_logits (None or list): Attention logits.
            attn_weights (None or list): Attention weights.

        Methods:
            update_prompt_length(self, length): Update current input prompt length.
            trigger(self, seq_length, **kwargs): Return whether Evictor needs to evict KV.
            update_num_token(self, num_token): Update the number of tokens.
            update_attn(self, attn_score): Update current attention-related scores.
            compress(self, cache, last_token_idx): Evict and update cache.
        """
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.llm_layers_per_chunk = llm_layers_per_chunk.copy()
        self.cache_size = cache_size
        self._num_token = 128
        self.max_capacity_prompt = 4096
        self._prompt_length = 0
        self.sink_rope = False
        self.attn_logits = None
        self.attn_weights = None

    def update_prompt_length(self, length: int, overture_size: int):
        """Update current input prompt length.

        Args:
            length (int): input prompt length (number of tokens)
            overture_size (int): overture length (number of overture tokens)
        """
        self._prompt_length = length
        self._overture_size = overture_size

    @abstractmethod
    def trigger(self, seq_length) -> bool:
        """Return whether Evictor need to evict KV.

        Args:
            seq_length: Current sequence length
        """

    def update_num_token(self, num_token):
        """Update num_token of current forward. For example, LocalSnapKV may need this information."""
        self._num_token = num_token

    def update_attn(self, **kwargs):
        """Update current attn related scores.

        Args:
            kwargs: Other inputs for diff evictor
        """
        self.attn_logits = kwargs.get('attn_logits')
        self.attn_weights = kwargs.get('attn_weights')

    def cache_insert(self, **kwargs):
        """Insert current pass KV cache into cache object.

        Args:
            kwargs: Other inputs for diff evictor
        """
        cache = kwargs.get('cache_evictor')
        cache.insert(kwargs.get('curr_k'), kwargs.get('curr_v'))

    @abstractmethod
    def compress(self, cache, last_token_idx, **kwargs):
        """Evict and update cache.

        Args:
            cache: KV cache
            last_token_idx: Valid last token
            kwargs: Other inputs for diff evictor
        """
