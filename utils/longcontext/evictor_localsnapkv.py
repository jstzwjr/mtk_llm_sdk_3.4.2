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
"""LocalSnapKV Evictor Class."""

import numpy as np
import torch
import torch.nn.functional as f
from scipy.special import softmax

from ...utils import const
from ..utils import logger
from .evictor_base import EvictorBase


class LocalSnapKV(EvictorBase):
    """LocalSnapKV Evictor."""

    def __init__(
        self,
        config,
        cache_size=2048,
        llm_layers_per_chunk=None,
        window_size=128,
        kernel_size=7,
    ) -> None:
        """LocalSnapKV Evictor.

        Attributes:
            cache_size (int): Size of the cache.
            window_size (int): Size of the window.
            kernel_size (int): Size of the kernel.
            h2o_ori_attn_values (np.ndarray): Array of original attention values for H2O, initialized to zeros.
            h2o_norm_attn_values (np.ndarray): Array of normalized attention values for H2O, initialized to zeros.
            cumulate_freq (np.ndarray): Array of cumulative frequency values, initialized to zeros.
            attn_sink_size (int): Size of the attention sink. Default is 4.

        Methods:
            update_attn(self, attn_score, offset, cache): Get attn output and do dimension reduction.
            _accumulate_attn_index(self, offset): Accumulate attn indices based on H2O paper.
            _evict_attn_index(self, indices): Evict attn indices.
            trigger(self, cur_length): Triggered when cur_length exceed cache size.
            compress(self, cache, last_token_idx, local_curr_k, local_curr_v): Evict and update cache.
            _remove_least_k_avgpool(self, mean_attn_scores, preserved_size): Remove least-k attention weights.
            _update_cache(self, layer_idx, cache, indices, last_token_idx): Get compressed key and value states.
        """
        super().__init__(config, cache_size, llm_layers_per_chunk)
        logger.debug('### Initiating LocalSnapKV ###')
        self.cache_size = cache_size
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.h2o_ori_attn_values = np.zeros(self.cache_size + self.window_size, dtype=np.float32)
        self.h2o_norm_attn_values = np.zeros(self.cache_size + self.window_size, dtype=np.float32)
        self.cumulate_freq = np.zeros(self.cache_size + self.window_size, dtype=np.float32)
        self.is_ingraph = config.cache_evict.get('in_graph', False)
        self.attn_sink_size = 4

    def update_attn(self, **kwargs):
        """Update current attn related scores.

        Args:
            kwargs: Other inputs for diff evictor.
            attn_logits (np.ndarray, optional): Attention logits from self attention map, provided via kwargs.
            attn_weights (np.ndarray, optional): Attention weights from self attention map, provided via kwargs.
            seq_length (int, optional): Current sequence length index, provided via kwargs.
            cache (object, optional): Class cache (cache_utils), provided via kwargs.
        """
        offset = kwargs.get('seq_length')
        cache = kwargs.get('cache_evictor')

        # Check if offset is provided and is an integer
        if offset is None or not isinstance(offset, int):
            logger.error(
                'offset must be passed when update_attn in localsnapkv. It should be an integer.', err=ValueError
            )

        # Check if cache is provided and has the required attribute
        if cache is None or not hasattr(cache, 'is_full'):
            logger.error(
                "cache must be passed when update_attn in localsnapkv. It should be with 'is_full' attr.",
                err=ValueError,
            )

        # Keep offset as cache_size + num_token when cache eviction (LocalSnapKV) starts
        if cache.is_full:
            offset = self.cache_size + self._num_token - self._overture_size

        # TODO: self.attn_logits implementation for DX5 FLA
        self.attn_logits = kwargs.get('attn_logits')
        self.attn_weights = kwargs.get('attn_weights')

        if isinstance(self.attn_weights, torch.Tensor) and self.attn_weights.device.type.startswith('cuda'):
            self.attn_weights = self.attn_weights.to('cpu')
        elif isinstance(self.attn_weights, list):
            self.attn_weights = [
                w.to('cpu').numpy() if isinstance(w, torch.Tensor) and w.device.type == 'cuda' else w
                for w in self.attn_weights
            ]
        if self.is_ingraph:
            self.compress_attn_weights = np.squeeze(np.amax(np.concatenate(self.attn_weights, axis=0), axis=0))
        else:
            self.compress_attn_weights = np.amax(np.amax(np.concatenate(self.attn_weights, axis=0), axis=0), axis=2)
        self._accumulate_attn_index(offset)

    def _accumulate_attn_index(self, offset):
        """Update attention index by weighted cumulation (H2O-style).

        Args:
            offset (int): Current cache offset index
        """
        if offset <= self._num_token:
            if self.is_ingraph:
                self.h2o_ori_attn_values[:offset] += self.compress_attn_weights[:offset]
            else:
                self.h2o_ori_attn_values[:offset] += np.mean(self.compress_attn_weights, axis=0)[:offset]
            self.h2o_norm_attn_values[:offset] += softmax(self.h2o_ori_attn_values[:offset])
        else:
            if self.is_ingraph:
                self.h2o_ori_attn_values[offset - self._num_token : offset] += self.compress_attn_weights
            else:
                self.h2o_ori_attn_values[offset - self._num_token : offset] += np.mean(
                    self.compress_attn_weights, axis=0
                )
        self.cumulate_freq[:offset] += 1

    def _evict_attn_index(self, indices):
        """Update attention index according to the preserved indices after eviction.

        Args:
            indices (int): The preserved indices after eviction
        """
        self.h2o_ori_attn_values[: len(indices)] = self.h2o_ori_attn_values[indices]
        self.h2o_norm_attn_values[: len(indices)] = self.h2o_norm_attn_values[indices]
        self.cumulate_freq[: len(indices)] = self.cumulate_freq[indices]

        self.h2o_ori_attn_values[len(indices) :] = 0
        self.h2o_norm_attn_values[len(indices) :] = 0
        self.cumulate_freq[len(indices) :] = 0

    def cache_insert(self, **kwargs):
        """Insert current pass KV cache into cache object.

        Args:
            kwargs: Other inputs for diff evictor
            cache (object, optional): cache: KV cache, provided via kwargs.
            num_token (int, optioanl): Number of current inference pass token, provided via kwargs.
            curr_k (list, optional): Current pass inference K cache (num_token), provided via kwargs.
            curr_v (list, optioanl): Current pass inference V cache (num_token), provided via kwargs.
        """
        cache = kwargs.get('cache_evictor')
        curr_k = kwargs.get('curr_k')
        curr_v = kwargs.get('curr_v')

        # Check if cache is provided and has the required attributes
        if cache is None or not hasattr(cache, 'offset') or not hasattr(cache, 'is_full'):
            logger.error(
                'cache must be passed when cache_insert in localsnapkv. It should be with offset and is_full attr.',
                err=ValueError,
            )

        # Check if curr_k and curr_v are provided and are lists
        if curr_k is None or not isinstance(curr_k, list):
            logger.error('curr_k must be passed when cache_insert in localsnapkv. It should be a list.', err=ValueError)
        if curr_v is None or not isinstance(curr_v, list):
            logger.error('curr_v must be passed when cache_insert in localsnapkv. It should be a list.', err=ValueError)

        if cache.offset + self._num_token > self.cache_size or cache.is_full:
            self.curr_k = curr_k
            self.curr_v = curr_v
        else:
            cache.insert(curr_k, curr_v)

    def trigger(self, cur_length):
        """Triggered when cur_length exceed cache size.

        Args:
            cur_length (int): Current sequence length.
        """
        return bool(cur_length > (self.cache_size - self._overture_size))

    def compress(self, cache, last_token_idx):
        """Evict and update cache.

        Args:
            cache: KV cache.
            last_token_idx (int): Valid last token.
        """
        # Keep offset as cache_size + num_token when cache eviction (LocalSnapKV) starts
        if cache.is_full:
            last_token_idx = self.cache_size + self._num_token - self._overture_size

        mean_attn_scores = np.divide(self.h2o_norm_attn_values[:last_token_idx], self.cumulate_freq[:last_token_idx])
        mean_attn_scores[: self.attn_sink_size] = const.EVICTION_INDEX_MAX_SCORE
        mean_attn_scores[-self.window_size :] = const.EVICTION_INDEX_MAX_SCORE

        curr_k, curr_v = [], []
        for layer_idx in range(self.num_hidden_layers):
            indices, _ = torch.sort(
                self._remove_least_k_avgpool(mean_attn_scores, last_token_idx - self.cache_size + self._overture_size),
                descending=False,
            )
            k_cache, v_cache = self._update_cache(
                layer_idx, cache, indices, last_token_idx, self.curr_k[layer_idx], self.curr_v[layer_idx]
            )
            curr_k.append(k_cache)
            curr_v.append(v_cache)
        self._evict_attn_index(indices)
        cache.offset = 0
        cache.insert(curr_k, curr_v)

    def _remove_least_k_avgpool(self, mean_attn_scores, removed_size):
        """Calculate average pooling according to LocalSnapKV alogrithm and then return preserved indices.

        Args:
            mean_attn_scores (np.ndarray): attention scores after divided with cummulative freq.
            removed_size (int): Size of the removed index
        """
        mean_avg_pool_attn_arr = f.avg_pool1d(
            torch.from_numpy(mean_attn_scores).view(1, 1, -1),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        sorted_scores = np.squeeze(np.argsort(mean_avg_pool_attn_arr))
        return sorted_scores[removed_size:].cpu()

    def _update_cache(self, layer_idx, cache, indices, last_token_idx, local_curr_k, local_curr_v):
        """Update cache with compressed key and value states.

        Args:
            layer_idx (int): Index of the layer.
            cache: KV cache.
            indices: Indices of the top-k attention weights.
            last_token_idx (int): Valid last token.
            local_curr_k (np.ndarray): current pass inference K cache with the size of num_token, per layer.
            local_curr_v (np.ndarray): current pass inference V cache with the size of num_token, per layer.
        """
        origin_key_states, origin_value_states = cache.get(layer=layer_idx)

        if isinstance(origin_key_states, np.ndarray):
            origin_key_states = torch.from_numpy(origin_key_states)
            origin_value_states = torch.from_numpy(origin_value_states)

        if isinstance(local_curr_k, np.ndarray):
            local_curr_k = torch.from_numpy(local_curr_k)
            local_curr_v = torch.from_numpy(local_curr_v)

        if cache.offset % self._num_token != 0:
            rb_offset = cache.cache_size - cache.offset
            if self._overture_size > 0:
                ot_key = origin_key_states[:, :, rb_offset : rb_offset + self._overture_size, :]
                ot_value = origin_value_states[:, :, rb_offset : rb_offset + self._overture_size, :]
            origin_key_states = origin_key_states[:, :, rb_offset + self._overture_size : cache.cache_size, :]
            origin_value_states = origin_value_states[:, :, rb_offset + self._overture_size : cache.cache_size, :]

        else:
            if self._overture_size > 0:
                ot_key = origin_key_states[:, :, : self._overture_size, :]
                ot_value = origin_value_states[:, :, : self._overture_size, :]
            origin_key_states = origin_key_states[:, :, self._overture_size :, :]
            origin_value_states = origin_value_states[:, :, self._overture_size :, :]

        origin_key_states = torch.cat((origin_key_states, local_curr_k), dim=2)
        origin_value_states = torch.cat((origin_value_states, local_curr_v), dim=2)

        indices = indices.view(1, 1, -1, 1)
        indices = indices.repeat(origin_key_states.shape[0], origin_key_states.shape[1], 1, origin_key_states.shape[3])
        key_states = origin_key_states[:, :, :, :].gather(dim=2, index=indices.to(origin_key_states.device))
        value_states = origin_value_states[:, :, :, :].gather(dim=2, index=indices.to(origin_value_states.device))

        if self._overture_size > 0:
            key_states = torch.cat([ot_key, key_states], dim=2)
            value_states = torch.cat([ot_value, value_states], dim=2)

        compress_length = key_states.shape[2]
        assert compress_length == self.cache_size

        if not (isinstance(key_states, torch.Tensor) and key_states.device.type.startswith('cuda')):
            key_states = key_states.numpy()
            value_states = value_states.numpy()

        return key_states, value_states
