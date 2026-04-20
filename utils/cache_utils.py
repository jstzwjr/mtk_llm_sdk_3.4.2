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
"""Utility functions for cache management."""

import mtk_quantization
import numpy as np
import torch
import torch.nn as nn

from . import logger


class InfiniUpdate(nn.Module):
    """Infini attention memory update class."""

    def __init__(self, config, device, jit_trace=False):
        """Initialize the Infini attention updater."""
        super().__init__()
        self.config = config
        self.device = device
        self.num_layers = self.config.num_hidden_layers
        self.jit_trace = jit_trace

        self.mem_update_add = []
        self.mem_update_bmm = []
        self.z_update_add = []
        self.cache_pad_mul = []
        if self.config.infini_update == 'delta':
            self.mem_update_sub = []
            self.mem_update_div = []
            self.sigmak_mem_bmm = []
            self.sigmak_z_bmm = []

        for _ in range(self.num_layers):
            self.mem_update_add.append(mtk_quantization.pytorch.functional.Add())
            self.mem_update_bmm.append(mtk_quantization.pytorch.functional.Matmul())
            if self.config.infini_update == 'delta':
                self.mem_update_sub.append(mtk_quantization.pytorch.functional.Sub())
                self.mem_update_div.append(mtk_quantization.pytorch.functional.Div())
                self.sigmak_mem_bmm.append(mtk_quantization.pytorch.functional.Matmul())
                self.sigmak_z_bmm.append(mtk_quantization.pytorch.functional.Matmul())
            self.z_update_add.append(mtk_quantization.pytorch.functional.Add())
            self.cache_pad_mul.append(mtk_quantization.pytorch.functional.Mul())

    def forward(self, *cache):
        """Update infini memory using current KV caches."""
        updated_cache = []
        assert len(cache) == self.num_layers * 4, f'cache length: {len(cache)}, Expected: {self.num_layers * 4}'
        for i in range(self.num_layers):
            past_key = cache[i * 4].to(self.device)
            past_value = cache[i * 4 + 1].to(self.device)
            mem = cache[i * 4 + 2].to(self.device)
            z = cache[i * 4 + 3].to(self.device)

            sigma_k = nn.functional.relu(past_key) + 1.0

            # Apply mem update
            if self.config.infini_update == 'linear':
                mem = self.mem_update_add[i](mem, self.mem_update_bmm[i](sigma_k.transpose(-2, -1), past_value))
                mem_l2norm = torch.norm(mem, p=2, dim=(-1, -2), keepdim=True)
                mem = mem / (mem_l2norm + self.config.norm_eps)
            elif self.config.infini_update == 'delta':
                mem = self.mem_update_add[i](
                    mem,
                    self.mem_update_bmm[i](
                        sigma_k.transpose(-2, -1),
                        self.mem_update_sub[i](
                            past_value,
                            self.mem_update_div[i](
                                self.sigmak_mem_bmm[i](sigma_k, mem), self.sigmak_z_bmm[i](sigma_k, z)
                            ),
                        ),
                    ),
                )
            # Apply normalization term update
            z = self.z_update_add[i](z, sigma_k.sum(dim=-2, keepdim=True))
            z_l2norm = torch.norm(z, p=2, dim=(-1, -2), keepdim=True)
            z = z / (z_l2norm + self.config.norm_eps)

            updated_cache.append(mem)
            updated_cache.append(z)

        return (*updated_cache,)

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        head_dim = self.config.head_dim

        example_inputs = []
        for i in range(self.config.num_hidden_layers):
            if self.config.sparse_attn and i not in [0, self.config.num_hidden_layers - 1]:
                num_head = self.config.sparse_attn_num_head
            else:
                num_head = self.config.num_key_value_heads
            example_inputs.extend(
                [
                    torch.randn(1, num_head, 128, head_dim, device='cpu', dtype=torch.float32),
                    torch.randn(1, num_head, 128, head_dim, device='cpu', dtype=torch.float32),
                    torch.randn(1, num_head, head_dim, head_dim, device='cpu', dtype=torch.float32),
                    torch.randn(1, num_head, 1, head_dim, device='cpu', dtype=torch.float32),
                ]
            )

        return example_inputs

    def get_ptq_inputs(
        self,
    ):
        """Gets inputs for post-training quantization (PTQ).

        This method generates inputs for PTQ.

        Args:
            args (Namespace): Arguments for PTQ.

        Returns:
            Tuple containing input shapes, input value ranges, calibration data generator,
                and evaluation data generator.
        """
        head_dim = self.config.head_dim

        input_shapes = []
        for i in range(self.config.num_hidden_layers):
            if self.config.sparse_attn and i not in [0, self.config.num_hidden_layers - 1]:
                num_head = self.config.sparse_attn_num_head
            else:
                num_head = self.config.num_key_value_heads
            input_shapes.extend(
                [
                    [None, num_head, None, head_dim],
                    [None, num_head, None, head_dim],
                    [None, num_head, head_dim, head_dim],
                    [None, num_head, 1, head_dim],
                ]
            )

        input_value_ranges = [None for _ in range(len(input_shapes))]

        def calib_data_gen():
            for _ in range(10):
                return_data = []
                for i in range(self.config.num_hidden_layers):
                    if self.config.sparse_attn and i not in [0, self.config.num_hidden_layers - 1]:
                        num_head = self.config.sparse_attn_num_head
                    else:
                        num_head = self.config.num_key_value_heads
                    return_data.extend(
                        [
                            np.random.rand(1, num_head, 128, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, 128, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, head_dim, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, head_dim, 1).astype(np.float32),
                        ]
                    )
                yield return_data

        def eval_data_gen():
            for _ in range(10):
                return_data = []
                for i in range(self.config.num_hidden_layers):
                    if self.config.sparse_attn and i not in [0, self.config.num_hidden_layers - 1]:
                        num_head = self.config.sparse_attn_num_head
                    else:
                        num_head = self.config.num_key_value_heads
                    return_data.extend(
                        [
                            np.random.rand(1, num_head, 128, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, 128, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, head_dim, head_dim).astype(np.float32),
                            np.random.rand(1, num_head, head_dim, 1).astype(np.float32),
                        ]
                    )
                yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen


class Memory:
    """Class to manage infini attention memory."""

    def __init__(self, config, dtype, prompt_cache_size, gen_cache_size, is_local, device=None):
        """Initialize memory class."""
        self.config = config
        self.dtype = dtype
        self.is_first_segment = True
        self.cur_total_processed_length = 0
        self.cur_segment_processed_length = 0
        self.is_local = is_local
        self.prompt_cache_size = prompt_cache_size
        self.gen_cache_size = gen_cache_size
        self.cur_cache_size = self.prompt_cache_size
        self.head_dim = int(getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads))
        self.segment_size = self.config.infini_segment_size
        self.attention_sink_size = self.config.infini_sink_size
        self.infini_window_size = self.config.infini_window_size
        self.infini_update_mode = self.config.infini_update
        self.infini_max_gen_length = self.config.infini_max_gen_length
        self.update_during_gen = not self.infini_max_gen_length > 0
        self.cache_contain_window = False
        self.is_prompt_full_segment_mode = False
        self.is_prompt_mode = True

        self.num_layers = (
            config.early_exit_index + config.early_exit_num_layers
            if config.early_exit_index is not None
            else config.num_hidden_layers
        )

        self.device = device
        if isinstance(self.dtype, torch.dtype) and self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.attention_sink_size > 0:
            self.attention_sink = Cache(
                config, cache_size=self.attention_sink_size, dtype=dtype, mode='static', device=device, overture=None
            )

        self.updater = InfiniUpdate(config, device=self.device)
        self.mem = []
        self.z = []
        self.is_memory_updated = False
        self.reset_memory()

    def reset_memory(self):
        """Reset memory elements to zeros."""
        self.is_first_segment = True
        self.cur_total_processed_length = 0
        self.cur_segment_processed_length = 0

        num_head = self.config.num_key_value_heads

        for _i in range(self.num_layers):
            if isinstance(self.dtype, torch.dtype):
                curr_layer_mem = torch.zeros(
                    (1, num_head, self.head_dim, self.head_dim),
                    dtype=self.dtype,
                ).to(self.device)
            else:
                curr_layer_mem = np.zeros(
                    (1, num_head, self.head_dim, self.head_dim),
                    dtype=self.dtype,
                )

            if isinstance(self.dtype, torch.dtype):
                curr_layer_z = torch.zeros(
                    (1, num_head, 1, self.head_dim),
                    dtype=self.dtype,
                ).to(self.device)
            else:
                curr_layer_z = np.zeros(
                    (1, num_head, 1, self.head_dim),
                    dtype=self.dtype,
                )

            self.mem.append(curr_layer_mem)
            self.z.append(curr_layer_z)

    def get(self, layer=None, layers=None, kvs=None):
        """Insert memory elements into KV caches for specified layers."""
        if layer is not None and layers is not None:
            logger.error(
                """`layer` and `layers` cannot be specified together.
                Please use `layer` for single layer and `layers` for multi layers."""
            )
            raise
        target_layers = [layer] if layer is not None else layers

        if not self.is_first_segment:
            # prepend attention sink to all kvs
            kvs = self._prepend_attention_sink(target_layers, kvs)

        return self._insert_memory_elements_to_cache(target_layers, kvs)

    def _prepend_attention_sink(self, layers, kvs):
        if self.attention_sink_size == 0:
            return kvs
        # Note: the attention sink will need to be placed before the cur segment kv
        insert_start_position = max(
            0, self.cur_cache_size - self.cur_segment_processed_length - self.attention_sink_size
        )
        insert_end_position = insert_start_position + self.attention_sink_size
        for i in range(len(layers)):
            attention_sink = self.attention_sink.get(layer=layers[i])

            kvs[2 * i][:, :, insert_start_position:insert_end_position, :] = attention_sink[0]
            kvs[2 * i + 1][:, :, insert_start_position:insert_end_position, :] = attention_sink[1]

        return kvs

    def _insert_memory_elements_to_cache(self, layers, kvs, is_updating=False):
        new_kvs = []
        for i in range(len(layers)):
            if self.is_memory_updated or is_updating:
                new_kvs.extend(kvs[2 * i : 2 * (i + 1)])
                new_kvs.append(self.mem[layers[i]])
                new_kvs.append(self.z[layers[i]])
            else:
                new_kvs.extend(kvs[2 * i : 2 * (i + 1)])
                if isinstance(self.dtype, torch.dtype):
                    z_ones = torch.ones(
                        (1, self.config.num_key_value_heads, 1, self.head_dim),
                        dtype=self.dtype,
                    ).to(self.device)
                else:
                    z_ones = np.ones(
                        (1, self.config.num_key_value_heads, 1, self.head_dim),
                        dtype=self.dtype,
                    )
                new_kvs.append(self.mem[layers[i]])
                # use this to prevent z from being zero
                new_kvs.append(z_ones)

        return new_kvs

    def insert_and_update(self, cur_kvs, num_token):
        """Update the memory and attention sink if required using the new KV caches."""
        self.cur_total_processed_length += num_token
        self.cur_segment_processed_length += num_token

        expected_max_cur_segment_process_length = self.segment_size
        if self.is_first_segment:
            expected_max_cur_segment_process_length += self.attention_sink_size
        if not self.is_prompt_full_segment_mode and not self.cache_contain_window:
            expected_max_cur_segment_process_length += self.infini_window_size

        if self.is_prompt_mode or self.update_during_gen:
            assert self.cur_segment_processed_length <= expected_max_cur_segment_process_length, (
                f'cur segment length is greater than expected, {num_token}, {self.cur_segment_processed_length}'
            )

        if self.attention_sink_size > 0 and not self.attention_sink.is_full:
            # need to add attention sinks
            sink_len = self.attention_sink.remaining_size_to_full

            assert len(cur_kvs) // 2 == self.num_layers
            start_position = self.cur_cache_size - num_token
            end_position = start_position + sink_len
            insert_k = [cur_kvs[2 * i][:, :, start_position:end_position, :] for i in range(self.num_layers)]
            insert_v = [cur_kvs[2 * i + 1][:, :, start_position:end_position, :] for i in range(self.num_layers)]

            self.attention_sink.insert(insert_k, insert_v)

        if self.cur_segment_processed_length == expected_max_cur_segment_process_length:
            logger.debug('Updating memory.')
            logger.debug(f'First segment: {self.is_first_segment}, segment size: {self.cur_segment_processed_length}')

            if not self.is_prompt_full_segment_mode:
                # the first segment after full segment mode will contain the window
                self.cache_contain_window = True

                if self.is_prompt_mode:
                    logger.error('During prompt mode processing of leftover, memory update should not happen')

                if self.update_during_gen:
                    # slice out the window part
                    logger.warning(
                        'Memory is updated during gen mode. Note that window tokens will not be recalculated.'
                    )
                    slice_offset = self.infini_window_size
                    if slice_offset > 0:
                        cur_kvs = [ele[:, :, :-slice_offset, :] for ele in cur_kvs]

                else:
                    logger.debug(
                        'Cache will have enough space to hold infini_max_gen_length'
                        'Under gen mode, memory is not compressed and updated. '
                    )
                    return False
            else:
                # slice out the unnecessary part, here the unnecessary part is at the front of cache
                slice_offset = self.infini_window_size
                if slice_offset > 0:
                    cur_kvs = [ele[:, :, slice_offset:, :] for ele in cur_kvs]

            self.is_first_segment = False
            self.cur_segment_processed_length = 0

            kvs_with_mem = self._insert_memory_elements_to_cache(
                layers=list(range(self.num_layers)), kvs=cur_kvs, is_updating=True
            )

            if not isinstance(self.dtype, torch.dtype):
                kvs_with_mem = [torch.tensor(ele) for ele in kvs_with_mem]

            updated_mem = self.updater(*kvs_with_mem)

            if not isinstance(self.dtype, torch.dtype):
                updated_mem = [ele.detach().cpu().numpy() for ele in updated_mem]

            # handle updated mem
            self.mem = []
            self.z = []
            for i in range(len(updated_mem) // 2):
                self.mem.append(updated_mem[2 * i])
                self.z.append(updated_mem[2 * i + 1])

            logger.debug('Memory Updated')
            self.is_memory_updated = True
            return True
        logger.debug('Memory not updated')
        return False

    def get_valid_cache_size(self, overture_size):
        """Get the valid cache size for generation, taking into acccount of attention sink."""
        expected_seg_cache_size = self.cur_segment_processed_length

        if not self.is_first_segment:
            expected_seg_cache_size += self.attention_sink_size

        if not self.is_prompt_full_segment_mode and self.cache_contain_window:
            expected_seg_cache_size += self.infini_window_size

        if self.is_first_segment:
            return min(expected_seg_cache_size + overture_size, self.cur_cache_size)
        # no need to bother overture for 2nd segment onwards because it is already gone
        return min(expected_seg_cache_size, self.cur_cache_size)

    def get_start_position(self, overture_size):
        """Get the valid cache size for generation, taking into acccount of attention sink."""
        # Consider only local for now
        self.is_local = True
        if self.is_local:
            return self.get_valid_cache_size(overture_size)
        logger.error('global position mode for memory not supported')
        raise

    def set_prompt_full_segment_mode(self, status):
        """Indicate if model is processing full segment prompts."""
        self.is_prompt_full_segment_mode = status

    def set_prompt_mode(self, status):
        """Indicate if model is in prompt mode or gen mode."""
        self.is_prompt_mode = status
        self.cur_cache_size = self.prompt_cache_size if self.is_prompt_mode else self.gen_cache_size


class Cache:
    """Class to manage/manipulate LLM cache."""

    def __init__(self, config, cache_size, dtype, mode='static', device=None, overture=None, bita_prefix=None):
        """Instantiate Cache object.

        Args:
            config (object): The configuration object containing model parameters.
            cache_size (int): The size of the cache.
            mode (str): Mode of cache shape. One of: static, dynamic.
            dtype (torch.dtype or numpy.dtype): The data type of the cache (e.g., np.float32, torch.float32).
            device (str): Device to put the cache on. Only used for torch dtypes.
            overture (str or numpy.ndarray): path to model config or overture npy or Overture array.
            bita_prefix (dict): BiTA prefix weights containing mask_tokens and prefix_encoder.
        """
        logger.debug(f'[Cache] Initialize Cache. cache_size={cache_size}, dtype={dtype}, mode={mode}.')
        self.config = config
        self.cache_size = cache_size
        self.orig_cache_size = cache_size  # Keep track of the original cache size in case it needs to be reverted
        self.dtype = dtype
        self.mode = mode
        if self.mode not in ('static', 'dynamic'):
            logger.error(f'Cache mode must be `static` or `dynamic`, but got {self.mode}.', err=ValueError)
        self.device = device
        if isinstance(self.dtype, torch.dtype) and self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_layers = (
            config.early_exit_index + config.early_exit_num_layers
            if config.early_exit_index is not None
            else config.num_hidden_layers
        )

        self.head_dim = int(getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads))
        logger.debug(f'[Cache] head_dim={self.head_dim}, num_layers={self.num_layers}, device={self.device}')
        self.keys = []
        self.values = []
        self.offset = 0  # For ring buffer

        # BiTA prefix handling
        self.bita_prefix = bita_prefix
        self.use_bita = self.bita_prefix is not None
        if self.use_bita:
            logger.debug(f'[Cache] Using BiTA prefix with shape: {self.bita_prefix.shape}')
            self.bita_prefix_length = self.bita_prefix.shape[4]

        self.overture = overture
        if self.overture is not None:
            if self.overture.shape[0] != 2 * self.num_layers:
                logger.error(
                    f'[Cache] Expected {2 * self.num_layers} number of layers in overture cache but got '
                    f'{self.overture.shape[0]}'
                )
            if isinstance(self.overture.dtype, torch.dtype) and isinstance(self.dtype, np.dtype):
                self.overture = self.overture.cpu().numpy().astype(self.dtype)
            if isinstance(self.overture.dtype, np.dtype) and isinstance(self.dtype, torch.dtype):
                self.overture = torch.from_numpy(self.overture).to(self.dtype).to(self.device)
        self.overture_size = self.overture.shape[2] if self.overture is not None else 0
        if mode == 'static' and self.overture_size > self.cache_size:
            logger.error(
                f'[Cache] Overture size ({self.overture_size}) cannot be more than cache size ({self.cache_size}) if '
                '`mode`=static'
            )

        self.reset()

    def _init_curr_layer_past_kv(self, num_head):
        if isinstance(self.dtype, torch.dtype):
            curr_layer_past_keys = torch.zeros(
                (1, num_head, self.cache_size, self.head_dim),
                dtype=self.dtype,
            ).to(self.device)
            curr_layer_past_values = torch.zeros(
                (1, num_head, self.cache_size, self.head_dim),
                dtype=self.dtype,
            ).to(self.device)
        else:
            curr_layer_past_keys = np.zeros(
                (1, num_head, self.cache_size, self.head_dim),
                dtype=self.dtype,
            )
            curr_layer_past_values = np.zeros(
                (1, num_head, self.cache_size, self.head_dim),
                dtype=self.dtype,
            )

        return curr_layer_past_keys, curr_layer_past_values

    def reset(self):
        """Resets all key caches and value caches to zeros, or overture cache if overture is enabled."""
        logger.debug('[Cache] Reset cache.')
        if self.use_bita:
            self.cache_size = self.orig_cache_size + self.bita_prefix_length

        if self.mode == 'dynamic' and self.overture_size > 0:
            self.cache_size = self.overture_size
            logger.debug(f'[Cache] Forcefully change cache_size to {self.overture_size} since loading overture cache.')

        self.is_cache_full = False
        self.keys = []
        self.values = []
        self.offset = 0
        for i in range(self.num_layers):
            if self.config.sparse_attn and i not in [0, self.num_layers - 1]:
                num_head = self.config.sparse_attn_num_head
            else:
                num_head = self.config.num_key_value_heads

            curr_layer_past_keys, curr_layer_past_values = self._init_curr_layer_past_kv(num_head)
            if self.overture_size > 0:
                curr_layer_past_keys[:, :, : self.overture_size, :] = self.overture[2 * i]
                curr_layer_past_values[:, :, : self.overture_size, :] = self.overture[2 * i + 1]

            # Add BiTA prefix
            if self.use_bita and self.bita_prefix_length > 0:
                # Extract the prefix for the current layer
                layer_prefix_k = self.bita_prefix[i, 0]  # Key prefix for layer i
                layer_prefix_v = self.bita_prefix[i, 1]  # Value prefix for layer i
                # Add prefix
                curr_layer_past_keys[:, :, -self.bita_prefix_length :, :] = layer_prefix_k
                curr_layer_past_values[:, :, -self.bita_prefix_length :, :] = layer_prefix_v

            self.keys.append(curr_layer_past_keys)
            self.values.append(curr_layer_past_values)

        if self.mode != 'dynamic' and self.overture_size > 0:
            self._increment_offset(self.overture_size)

    def get(self, ret='all', layer=None, layer_from=0, layer_to=None):
        """Gets K and/or V cache(s), based on arguments. Defaults to entire K/V cache.

        Args:
            ret (str): Type of cache to get. One of: key, value, all. Defaults to all.
            layer (int, optional): Layer index to get the cache(s) of. Cannot be used together with either
                `layer_from` or `layer_to`.
            layer_from (int, optional): Layer index of the first layer to get the cache(s) of. Defaults to 0.
                Cannot be used together with `layer`.
            layer_to (int, optional): Layer index of the last layer to get the cache(s) of.
                Defaults to num_hidden_layers. Cannot be used together with `layer`.

        Returns:
            list: A list of key/value caches in the order [K0, V0, K1, V1, K2, V2, ..., K(L-1), V(L-1)].
                Only returns key or value caches if `ret` is `key` or `value`.
                Only returns some layer caches if layer_from or layer_to are not default.
        """
        if ret not in ('all', 'key', 'keys', 'k', 'value', 'values', 'v'):
            logger.error('[Cache] ret must be one of: key, value, all', err=ValueError)

        if layer is not None and (layer_from != 0 or layer_to is not None):
            logger.error('[Cache] `layer` cannot be used together with either `layer_from` or `layer_to`.')

        layer_from = layer_from if layer is None else layer
        if layer_to is None:
            layer_to = self.num_layers - 1 if layer is None else layer

        logger.debug(f'[Cache] Get {ret} cache, layer idx from {layer_from} to {layer_to}, offset={self.offset}.')

        # Don't need to handle offset for mode=dynamic,
        # as offset is never incremented in dynamic mode so get() will always return the full cache.

        if ret in ['key', 'keys', 'k']:
            if isinstance(self.dtype, torch.dtype):
                return [
                    torch.cat((k[:, :, self.offset :, :], k[:, :, : self.offset, :]), dim=2)
                    for k in self.keys[layer_from : layer_to + 1]
                ]
            return [
                np.concatenate((k[:, :, self.offset :, :], k[:, :, : self.offset, :]), axis=2)
                for k in self.keys[layer_from : layer_to + 1]
            ]
        if ret in ['value', 'values', 'v']:
            if isinstance(self.dtype, torch.dtype):
                return [
                    torch.cat((v[:, :, self.offset :, :], v[:, :, : self.offset, :]), dim=2)
                    for v in self.values[layer_from : layer_to + 1]
                ]
            return [
                np.concatenate((v[:, :, self.offset :, :], v[:, :, : self.offset, :]), axis=2)
                for v in self.values[layer_from : layer_to + 1]
            ]
        if isinstance(self.dtype, torch.dtype):
            return [
                torch.cat((c[:, :, self.offset :, :], c[:, :, : self.offset, :]), dim=2)
                for kv in list(zip(self.keys, self.values))[layer_from : layer_to + 1]
                for c in kv
            ]
        return [
            np.concatenate((c[:, :, self.offset :, :], c[:, :, : self.offset, :]), axis=2)
            for kv in list(zip(self.keys, self.values))[layer_from : layer_to + 1]
            for c in kv
        ]

    def insert(self, k, v):
        """Inserts new full K/V cache.

        Returns:
            list: A list of key/value caches in the order [K0, V0, K1, V1, K2, V2, ..., K(L-1), V(L-1)].
        """
        logger.debug('[Cache] Insert cache')
        if len(k) != self.num_layers:
            logger.error(f'[Cache] Expect {self.num_layers} key caches but got {len(k)} layers')
        if len(v) != self.num_layers:
            logger.error(f'[Cache] Expect {self.num_layers} value caches but got {len(v)} layers')

        num_token = k[0].shape[2]
        assert v[0].shape[2] == num_token
        logger.debug(f'[Cache] Length of cache to insert: {num_token}')

        if isinstance(self.dtype, torch.dtype):
            k = [x.to(self.device) for x in k]
            v = [x.to(self.device) for x in v]

        if self.mode == 'dynamic':
            if isinstance(self.dtype, torch.dtype):
                self.keys = [torch.cat(prev_curr_keys, dim=2) for prev_curr_keys in zip(self.keys, k)]
                self.values = [torch.cat(prev_curr_values, dim=2) for prev_curr_values in zip(self.values, v)]
            else:
                self.keys = [np.concatenate(prev_curr_keys, axis=2) for prev_curr_keys in zip(self.keys, k)]
                self.values = [np.concatenate(prev_curr_values, axis=2) for prev_curr_values in zip(self.values, v)]
            self.cache_size += num_token
        else:
            overflow = max(0, self.offset + num_token - self.cache_size)
            for i in range(self.num_layers):
                if num_token >= self.cache_size:
                    self.keys[i] = k[i][:, :, -self.cache_size :, :]
                    self.values[i] = v[i][:, :, -self.cache_size :, :]
                elif overflow:
                    self.keys[i][:, :, self.offset :, :] = k[i][:, :, :-overflow, :]
                    self.keys[i][:, :, :overflow, :] = k[i][:, :, -overflow:, :]
                    self.values[i][:, :, self.offset :, :] = v[i][:, :, :-overflow, :]
                    self.values[i][:, :, :overflow, :] = v[i][:, :, -overflow:, :]
                else:
                    self.keys[i][:, :, self.offset : self.offset + num_token, :] = k[i]
                    self.values[i][:, :, self.offset : self.offset + num_token, :] = v[i]

            self._increment_offset(num_token)

    def _increment_offset(self, num_token):
        if self.mode == 'dynamic':
            logger.error(
                '[Cache] Dynamic shape cache does not use offset. Do not call _increment_offset() when `mode`=dynamic'
            )

        if self.offset + num_token >= self.cache_size:
            self.is_cache_full = True

        if self.offset + num_token > self.cache_size:
            logger.debug('[Cache] Dropping the oldest cache since cache is full. May degrade accuracy.')

        if num_token >= self.cache_size:
            logger.debug('[Cache] Reset offset to zero due to num_token >= cache_size.')
            self.offset = 0
            return

        logger.debug(f'[Cache] Increment offset from {self.offset} to {(self.offset + num_token) % self.cache_size}')
        self.offset = (self.offset + num_token) % self.cache_size

    @property
    def is_full(self):
        """Check if cache is full or not. It will always return full after overflow happens."""
        return self.is_cache_full

    @property
    def remaining_size_to_full(self):
        """Check remaining empty space in cache."""
        if self.mode == 'dynamic':
            return -1
        return self.cache_size - self.offset
