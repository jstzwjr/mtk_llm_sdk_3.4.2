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
"""GlobalSnapKV Evictor Class."""

import numpy as np
import torch

from ..utils import logger
from .evictor_base import EvictorBase


class GlobalSnapKV(EvictorBase):
    """GlobalSnapKV Evictor."""

    def __init__(
        self,
        config,
        cache_size=2048,
        llm_layers_per_chunk=None,
        max_capacity_prompt=5,
        pooling='maxpool',
        window_size=32,
        kernel_size=7,
        in_graph=True,
    ) -> None:
        """GlobalSnapKV Evictor.

        Attributes:
            window_size (int): Size of the window.
            pooling (str): Pooling method.
            kernel_size (int): Size of the kernel.
            max_capacity_prompt (int): Maximum capacity of the prompt.
            gqa_support (bool): Whether GQA support is enabled.
            sorted_indices (bool): Whether indices are sorted.
            in_graph (bool): Whether in graph mode.

        Methods:
            update_prompt_length(self, length): Update current input prompt length.
            trigger(self, seq_length): Triggered after reading full prompt and seq_length exceed max capacity.
            compress(self, cache, last_token_idx): Evict and update cache.
            _top_k(self, attn_weights): Get top-k attention weights.
            _update_cache(self, layer_idx, cache, indices, last_token_idx): Get compressed key and value states.
        """
        super().__init__(config, cache_size, llm_layers_per_chunk)
        logger.debug('### Initiating GlobalSnapKV ###')
        self.window_size = window_size
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.max_capacity_prompt = max_capacity_prompt
        self.gqa_support = True
        self.sorted_indices = False
        self.in_graph = in_graph
        self._overture_size = 0

    def update_prompt_length(self, length, overture_size):
        """Update current input prompt length.

        Args:
            length (int): Input prompt length (number of tokens).
            overture_size (int): overture length (number of overture tokens)
        """
        self._prompt_length = length
        self._overture_size = overture_size
        if self._prompt_length + self._overture_size > self.cache_size:
            raise logger.error(
                f'The prompt length({self._prompt_length}) + overture length({self._overture_size}) ',
                f'is greater than cache size({self.cache_size}), '
                'which is not a feasible scenario in GlobalSnapKV. Please enlarge cache size.',
                err=RuntimeError,
            )

    def trigger(self, seq_length):
        """Triggered after reading full prompt and seq_length exceed max capacity.

        Args:
            seq_length (int): Current sequence length.
        """
        return bool(seq_length == self._prompt_length and seq_length + self._overture_size > self.max_capacity_prompt)

    def compress(self, cache, last_token_idx, **kwargs):
        """Evict and update cache.

        Args:
            cache: KV cache.
            last_token_idx (int): Valid last token.
            kwargs: Other inputs for diff evictor.
        """
        curr_k, curr_v = [], []

        last_token_idx += self._overture_size
        for layer_idx in range(self.num_hidden_layers):
            indices = self._top_k(self.attn_weights[layer_idx][..., -last_token_idx:])
            k_cache, v_cache = self._update_cache(layer_idx, cache, indices, last_token_idx)
            curr_k.append(k_cache)
            curr_v.append(v_cache)
        cache.offset = 0
        cache.insert(curr_k, curr_v)

    def _top_k(self, attn_weights):
        """Get top-k attention indices.

        Args:
            attn_weights: Attention weights.
        """
        if isinstance(attn_weights, np.ndarray):
            attn_weights = torch.from_numpy(attn_weights)

        if not self.in_graph:
            attn_weights_mean = attn_weights[:, :, -self.window_size :, : -self.window_size].mean(dim=-2)

            if self.gqa_support:
                attn_weights_mean = attn_weights_mean.view(
                    attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1]
                )
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            attn_weights_mean = attn_weights[..., : -self.window_size]

        if self.pooling == 'avgpool':
            attn_cache = torch.nn.functional.avg_pool1d(
                attn_weights_mean, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )
        elif self.pooling == 'maxpool':
            attn_cache = torch.nn.functional.max_pool1d(
                attn_weights_mean, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )
        elif self.pooling == 'nopool':
            attn_cache = attn_weights_mean
        else:
            raise logger.error('Pooling method not supported', err=ValueError)

        topk_num = self.max_capacity_prompt - self.window_size
        indices = attn_cache.topk(topk_num, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        return indices.cpu()

    def _update_cache(self, layer_idx, cache, indices, last_token_idx):
        """Update cache with compressed key and value states.

        Args:
            layer_idx (int): Index of the layer.
            cache: KV cache.
            indices: Indices of the top-k attention weights.
            last_token_idx (int): Valid last token.
        """
        origin_key_states, origin_value_states = cache.get(layer=layer_idx)

        if isinstance(origin_key_states, np.ndarray):
            origin_key_states = torch.from_numpy(origin_key_states)
            origin_value_states = torch.from_numpy(origin_value_states)

        k_past_compress = origin_key_states[:, :, -last_token_idx : -self.window_size, :].gather(
            dim=2, index=indices.to(origin_key_states.device)
        )
        v_past_compress = origin_value_states[:, :, -last_token_idx : -self.window_size, :].gather(
            dim=2, index=indices.to(origin_value_states.device)
        )

        # According to Qwen2.5-3b exp result, evict overture could get better quality
        k_cur = origin_key_states[:, :, -self.window_size :, :]
        v_cur = origin_value_states[:, :, -self.window_size :, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        compress_length = key_states.shape[2]
        assert compress_length == self.max_capacity_prompt

        if isinstance(origin_key_states, np.ndarray) and torch.is_tensor(key_states):
            key_states = key_states.numpy()
            value_states = value_states.numpy()

        return key_states, value_states
