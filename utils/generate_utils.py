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
"""Define inference-related helper functions for mtk_llm_sdk."""

import math
import os

import numpy as np
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from . import logger


def get_sample_logits_warper(temperature=None, top_p=None, top_k=None) -> LogitsProcessorList:
    """Creates a list of logits processors for sampling.

    Args:
        temperature (Optional[float]): The temperature for sampling. If None, no temperature warping is applied.
        top_p (Optional[float]): The cumulative probability for nucleus sampling. If None, no top-p warping is applied.
        top_k (Optional[int]): The number of highest probability tokens to keep for top-k sampling.
            If None, no top-k warping is applied.

    Returns:
        LogitsProcessorList: A list of logits processors.
    """
    logger.debug('Enter get_sample_logits_warper')
    warpers = LogitsProcessorList()

    if temperature is not None:
        logger.debug(f'Add TemperatureLogitsWarper, temperature={temperature}')
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None:
        logger.debug(f'Add TopKLogitsWarper, top_k={top_k}')
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None:
        logger.debug(f'Add TopPLogitsWarper, top_p={top_p}')
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    return warpers


def generate_mask(
    cache_size,
    valid_cache,
    input_length,
    valid_input,
    batch_size=1,
    mask_value=-100.0,
    medusa_mask=None,
    sliding_window=False,
    sliding_window_size=None,
    dtype=np.float32,
    bita_prefix_length=0,
    bita_draft_length=0,
):
    """Generates a mask for attention mechanisms, supporting both sliding window and normal attention.

    Args:
        cache_size (int): The size of the cache.
        valid_cache (int): The valid cache size.
        input_length (int): The length of the input.
        valid_input (int): The valid input length.
        batch_size (int, optional): The batch size. Defaults to 1.
        mask_value (float, optional): The value to use for masking. Defaults to -100.0.
        medusa_mask (Optional[np.ndarray], optional): The Medusa mask. Defaults to None.
        sliding_window (bool, optional): Whether to use sliding window attention. Defaults to False.
        sliding_window_size (int, optional): Sliding window size, required when sliding_window is True.
        dtype (np.dtype, optional): The data type of the mask. Defaults to np.float32.
        bita_prefix_length (int, optional): Length of the prefix for bidirectional attention. Defaults to 0.
        bita_draft_length (int, optional): Length of the draft tokens that can attend. Defaults to 0.

    Returns:
        np.ndarray: The generated mask.

    Raises:
        AssertionError: If `valid_cache` is greater than `cache_size`.
        AssertionError: If `valid_input` is greater than `input_length`.
    """
    logger.debug(
        f'Enter generate_mask. cache_size={cache_size}, valid_cache={valid_cache}, input_length={input_length}, '
        f'valid_input={valid_input}, batch_size={batch_size}, mask_value={mask_value}, medusa_mask={medusa_mask}, '
        f'sliding_window={sliding_window}, dtype={dtype}.'
    )
    assert valid_cache <= cache_size, 'valid_cache must be less than or equal to cache_size'

    if sliding_window:  # SWA
        assert sliding_window_size is not None
        # Full mask: (input_length, cache_size + input_length)
        full_mask_length = cache_size + input_length
        # 1. generate invalid input portion
        invalid_input_part = np.full(
            (batch_size, 1, input_length - valid_input, full_mask_length), mask_value, dtype=np.float32
        )

        # 2. valid input + invalid cache part
        invalid_cache_part = np.full(
            (batch_size, 1, valid_input, cache_size - valid_cache), mask_value, dtype=np.float32
        )

        # 3. valid input + invalid input part
        valid_input_vs_invalid_input_part = np.full(
            (batch_size, 1, valid_input, input_length - valid_input), mask_value, dtype=np.float32
        )

        # 4. valid input + valid_cache, valid_input part
        valid_part = np.full((batch_size, 1, valid_input, valid_cache + valid_input), mask_value, dtype=np.float32)

        # swap to zeros when within sliding window size
        for i in range(valid_part.shape[2]):
            end_position = valid_cache + i
            start_position = max(0, end_position - sliding_window_size + 1)
            valid_part[:, :, i, start_position : end_position + 1] = 0

        # combine everything together
        final_mask = np.concatenate([invalid_cache_part, valid_part, valid_input_vs_invalid_input_part], axis=-1)
        final_mask = np.concatenate([final_mask, invalid_input_part], axis=-2)
        combined_mask = final_mask

    else:  # Normal attention
        assert valid_input <= input_length, 'valid_input must be less than or equal to input_length'
        # Cache mask portion
        valid = np.zeros((1, 1, 1, valid_cache + input_length), dtype=np.float32)
        cache_mask = np.full((1, 1, 1, cache_size - valid_cache), mask_value, dtype=np.float32)
        cache_mask = np.concatenate((cache_mask, valid), axis=-1)
        cache_mask_final_shape = np.broadcast_to(cache_mask, (batch_size, 1, input_length, cache_size + input_length))

        # Attention mask portion
        mask_cond = np.arange(valid_input)
        triangle = mask_cond >= (mask_cond + 1).reshape(valid_input, 1)
        small_attention_mask = triangle.astype(np.float32) * mask_value
        attention_mask = np.pad(
            small_attention_mask, (0, input_length - valid_input), 'constant', constant_values=mask_value
        )
        attention_mask_with_cache = np.concatenate(
            [np.zeros((input_length, cache_size), dtype=np.float32), attention_mask], axis=-1
        )
        attention_mask_final_shape = np.broadcast_to(
            attention_mask_with_cache[None, None, :, :],
            (batch_size, 1, input_length, cache_size + input_length),
        )

        combined_mask = attention_mask_final_shape + cache_mask_final_shape
        combined_mask[
            :, :, -bita_draft_length:, -valid_cache - input_length - bita_prefix_length : -valid_cache - input_length
        ] = 0

    if medusa_mask is not None:
        medusa_len = medusa_mask.size(-1)
        combined_mask[:, :, -medusa_len:, -medusa_len:][medusa_mask == 0] = mask_value

    if isinstance(dtype, torch.dtype):
        return torch.from_numpy(combined_mask.copy()).to(dtype)
    return combined_mask.copy().astype(dtype)


def get_master_pos_emb(config, dtype, **kwargs):
    """Generates the master positional embeddings.

    Args:
        config (object): The configuration object containing model parameters.
        dtype (type): The data type of the embeddings.
        kwargs: Other kwargs.

    Returns:
        np.ndarray or torch.Tensor: The master positional embeddings.
    """
    logger.debug(
        f'Enter get_master_pos_emb. rotary_emb_base={config.rotary_emb_base}, embedding_dims={config.hidden_size}, '
        f'max_position_embeddings={config.max_position_embeddings}, dtype={dtype}'
    )

    state_dict = kwargs.pop('state_dict', None)

    if config.model_type == 'whisper_decoder':
        weight_dir = config.weight_dir
        if weight_dir is None and state_dict is None:
            logger.error('Expect either weight_dir or state_dict, but got neither')
            raise

        expected_path = os.path.join(weight_dir, 'embedding_pos_fp16.bin')
        if os.path.exists(expected_path):
            embedding_weight = np.fromfile(expected_path, dtype=np.float16).reshape(
                1, config.max_position_embeddings, config.hidden_size
            )
        else:
            embedding_weight = state_dict['model.decoder.embed_positions.weight'].unsqueeze(0).numpy()
            embedding_weight.astype(np.float16).tofile(expected_path)

        if isinstance(dtype, torch.dtype):
            return torch.tensor(embedding_weight).to(dtype)
        return embedding_weight.astype(dtype)

    max_timescale = config.rotary_emb_base
    min_timescale = 1
    embedding_dims = config.hidden_size
    position = np.arange(config.max_position_embeddings)[None, :]

    num_timescales = embedding_dims // 2
    log_timescale_increment = np.log(float(max_timescale) / float(min_timescale)) / np.maximum(num_timescales - 1, 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment)
    scaled_time = position[:, :, None].astype(np.float32) * inv_timescales[None, None, :]
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=2)
    signal = np.pad(signal, [[0, 0], [0, 0], [0, embedding_dims % 2]])

    if isinstance(dtype, torch.dtype):
        return torch.from_numpy(signal).to(dtype)
    return signal.astype(dtype)


def get_master_rot_emb(config, dtype, **kwargs):
    """Generates the master rotary embeddings.

    Args:
        config (object): The configuration object containing model parameters.
        dtype (type): The data type of the embeddings.
        kwargs: Other kwargs.

    Returns:
        np.ndarray or torch.Tensor: The master rotary embeddings.

    Raises:
        AssertionError: If the config is not an instance of PipelineConfig.
        NotImplementedError: If the rope scaling type is not supported.
    """
    logger.debug('Enter get_master_rot_emb.')
    from ..models.configuration_pipeline import PipelineConfig

    assert isinstance(config, PipelineConfig)
    partial_rotary_factor = getattr(config.l, 'partial_rotary_factor', 1)
    rot_dim = (
        int(int(config.l.hidden_size / config.l.num_attention_heads) * partial_rotary_factor)
        if config.l.model_type not in ['gecko2', 'gemma2', 'gemma3', 'qwen2', 'qwen3']
        else config.l.head_dim
    )

    logger.debug(
        f'rotary_emb_base={config.l.rotary_emb_base}, '
        f'partial_rotary_factor: {partial_rotary_factor}, '
        f'rot_dim={rot_dim}, '
        f'max_position_embeddings={config.l.max_position_embeddings}, '
        f'ntk_scaling_factor={config.l.ntk_scaling_factor}, model_type={config.l.model_type}, dtype={dtype}, '
        f'rope_scaling={getattr(config.l, "rope_scaling", None)}'
    )

    length = int(config.l.max_position_embeddings * config.l.ntk_scaling_factor)
    base = config.l.rotary_emb_base

    if config.l.ntk_scaling_factor != 1.0:
        base = (base * config.l.ntk_scaling_factor) ** (rot_dim / (rot_dim - 2))
    else:
        base = base

    if config.l.model_type in ['gecko', 'gecko2']:
        seq = np.arange(length).reshape([length, 1])
        frac = np.arange(rot_dim // 2) * 2 / float(rot_dim)
        sinusoid = seq / np.power(base, frac)
        sinusoid = sinusoid.reshape((1, length, rot_dim // 2))
        master_cos = np.cos(sinusoid)  # (1,len,rot_dim/2)
        master_sin = np.sin(sinusoid)  # (1,len,rot_dim/2)

        rot_emb = np.stack((master_cos, master_sin), axis=1)
    elif getattr(config.l, 'rope_scaling', None) is not None:
        if config.l.rope_scaling['type'] == 'longrope':
            short_factor = config.l.rope_scaling['short_factor']
            long_factor = config.l.rope_scaling['long_factor']
            original_max_position_embeddings = config.l.original_max_position_embeddings
            length = original_max_position_embeddings

            ext_factors_long = torch.tensor(long_factor, dtype=torch.float32)
            ext_factors = torch.tensor(short_factor, dtype=torch.float32)

            inv_freq_shape = torch.arange(0, rot_dim, 2, dtype=torch.int64).float() / rot_dim
            inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
            inv_freq = inv_freq.unsqueeze(1)
            t = torch.arange(length, dtype=torch.float32)
            t = t.unsqueeze(0)

            inv_freq_long = 1.0 / (ext_factors_long * base**inv_freq_shape)
            inv_freq_long = inv_freq_long.unsqueeze(1)
            t_long = torch.arange(config.l.max_position_embeddings - length, dtype=torch.float32)
            t_long = t_long.unsqueeze(0)

            # Short factor
            freqs = (inv_freq.float() @ t.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = config.l.max_position_embeddings / original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
            master_cos = emb.cos() * scaling_factor
            master_sin = emb.sin() * scaling_factor
            master_cos = master_cos[None, None, :, :]
            master_sin = master_sin[None, None, :, :]

            # Long factor
            freqs_long = (inv_freq_long.float() @ t_long.float()).transpose(0, 1)
            emb_long = torch.cat((freqs_long, freqs_long), dim=-1)

            scale = config.l.max_position_embeddings / original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
            master_cos_long = emb_long.cos() * scaling_factor
            master_sin_long = emb_long.sin() * scaling_factor
            master_cos_long = master_cos_long[None, None, :, :]
            master_sin_long = master_sin_long[None, None, :, :]

            master_cos = torch.cat((master_cos, master_cos_long), dim=2)
            master_sin = torch.cat((master_sin, master_sin_long), dim=2)
            rot_emb = torch.cat((master_cos, master_sin), dim=1).numpy()
        elif config.l.rope_scaling['type'] == 'mrope':
            # position_ids = config.e.mrope_position_ids
            # rope_delta = config.e.mrope_delta
            position_ids = kwargs.get('qwen2_vl_position_ids')
            rope_delta = kwargs.get('qwen2_vl_mrope_delta')
            if position_ids is None:
                logger.error('Must pass position_ids when using Qwen2-VL mrope.', err=ValueError)
            if rope_delta is None:
                logger.error('Must pass rope_delta when using Qwen2-VL mrope.', err=ValueError)

            mrope_section = config.l.rope_scaling['mrope_section'] * 2
            cache_position = torch.arange(position_ids.shape[2], config.l.max_position_embeddings, dtype=torch.float32)
            delta = cache_position + rope_delta
            delta = delta.view(1, -1).expand(1, -1)
            delta = delta.unsqueeze(0).expand(3, -1, -1)
            position_ids = torch.cat([position_ids, delta], dim=-1)

            inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))  # (rot_dim/2)
            inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, positions)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            master_cos = emb.cos()
            master_sin = emb.sin()

            cos = torch.cat(
                [m[i % 3] for i, m in enumerate(master_cos.split(mrope_section, dim=-1))], dim=-1
            ).unsqueeze(1)
            sin = torch.cat(
                [m[i % 3] for i, m in enumerate(master_sin.split(mrope_section, dim=-1))], dim=-1
            ).unsqueeze(1)
            rot_emb = torch.cat([cos, sin], dim=1).numpy()

        elif config.l.rope_scaling['type'] == 'yarn':
            scaling_factor = config.l.rope_scaling['factor']
            extrapolation_factor = config.l.rope_scaling.get('extrapolation_factor', 1)
            beta_fast = config.l.rope_scaling.get('beta_fast', 32)
            beta_slow = config.l.rope_scaling.get('beta_slow', 1)
            original_max_position_embeddings = config.l.original_max_position_embeddings
            if original_max_position_embeddings is None:
                logger.error(
                    'Using Yarn for rope scaling but original max_position_embeddings not given in config.json.'
                )
                raise

            def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
                return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

            def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
                low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
                high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
                return max(low, 0), min(high, dim - 1)  # Clamp values just in case

            def _yarn_linear_ramp_mask(min_val, max_val, dim):
                if min_val == max_val:
                    max_val += 0.001  # Prevent singularity

                linear_func = (np.arange(dim, dtype=np.float32) - min_val) / (max_val - min_val)
                return np.clip(linear_func, 0, 1)

            # yarn has 3 parts: interpolation, extrapolation and linear ramp
            inv_freq_extrapolation = 1.0 / (
                base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim)
            )  # (rot_dim/2)
            inv_freq_interpolation = inv_freq_extrapolation / scaling_factor

            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, rot_dim, base, original_max_position_embeddings
            )
            inv_freq_mask = (
                1 - _yarn_linear_ramp_mask(low, high, rot_dim // 2).astype(np.float32)
            ) * extrapolation_factor
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

            t = np.arange(length, dtype=np.float32)  # (len)
            freqs = np.einsum('i,j->ij', t, inv_freq)  # (len, rot_dim/2)
            emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
            master_cos = np.cos(emb)[None, None, :, :]  # (1,1,len,rot_dim)
            master_sin = np.sin(emb)[None, None, :, :]  # (1,1,len,rot_dim)

            rot_emb = np.concatenate((master_cos, master_sin), axis=1)

            # check if rot_emb range exceeds [-1, 1]
            if np.max(np.abs(rot_emb)) > 1:
                logger.error(
                    'Rotary embedding range should not exceed [-1, 1]. '
                    f'Largest absolute value: {np.max(np.abs(rot_emb))}.'
                )

        elif config.l.rope_scaling['type'] == 'linear':
            inv_freq = 1.0 / (base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))  # (rot_dim/2)
            inv_freq /= config.l.rope_scaling['factor']
            t = np.arange(length, dtype=np.float32)  # (len)
            freqs = np.einsum('i,j->ij', t, inv_freq)  # (len, rot_dim/2)
            emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
            master_cos = np.cos(emb)[None, None, :, :]  # (1,1,len,rot_dim)
            master_sin = np.sin(emb)[None, None, :, :]  # (1,1,len,rot_dim)
            rot_emb = np.concatenate((master_cos, master_sin), axis=1)

        else:
            logger.error(
                f'Rope scaling only supports longrope and mrope, but got {config.l.rope_scaling["type"]}',
                err=NotImplementedError,
            )
    else:
        inv_freq = 1.0 / (base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))  # (rot_dim/2)
        t = np.arange(length, dtype=np.float32)  # (len)
        freqs = np.einsum('i,j->ij', t, inv_freq)  # (len, rot_dim/2)
        emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
        master_cos = np.cos(emb)[None, None, :, :]  # (1,1,len,rot_dim)
        master_sin = np.sin(emb)[None, None, :, :]  # (1,1,len,rot_dim)

        rot_emb = np.concatenate((master_cos, master_sin), axis=1)

    if isinstance(dtype, torch.dtype):
        if isinstance(rot_emb, torch.Tensor):
            return rot_emb.to(dtype)
        return torch.from_numpy(rot_emb).to(dtype)
    if isinstance(rot_emb, torch.Tensor):
        return rot_emb.detach().cpu().numpy().astype(dtype)
    return rot_emb.astype(dtype)


def get_firelite_pos_emb(config, dtype, length, **kwargs):
    """Generates the fire positional embeddings used by gecko2."""
    # fire lite is only used for gecko2
    assert config.model_type == 'gecko2', (
        f'Fire lite is only supported for gecko2, current model is {config.model_type}.'
    )

    state_dict = kwargs.pop('state_dict', None)

    weight_dir = config.weight_dir
    if weight_dir is None and state_dict is None:
        logger.error('Expect either weight_dir or state_dict, but got neither')
        raise

    expected_log_scale_path = os.path.join(weight_dir, 'embedding_fire_pe_log_params.bin')
    expected_normalizer_factors_path = os.path.join(weight_dir, 'embedding_fire_pe_normalizer_params.bin')

    if os.path.exists(expected_log_scale_path) and os.path.exists(expected_normalizer_factors_path):
        # calculate number of layers
        is_ee = config.early_exit_index is not None
        num_of_layers = config.early_exit_index + config.early_exit_num_layers if is_ee else config.num_hidden_layers
        num_fire_pe = len([i for i in config.fire_pe_index if i < num_of_layers])
        log_scale_factors = np.fromfile(expected_log_scale_path, dtype=np.float32).reshape(num_fire_pe, -1)
        normalizer_threshold_factors = np.fromfile(expected_normalizer_factors_path, dtype=np.float32).reshape(
            num_fire_pe, -1
        )

    else:
        log_scale_factors = []
        ee_log_scale_factors = []
        normalizer_threshold_factors = []
        ee_normalizer_threshold_factors = []

        for key in state_dict:
            if 'fire' in key:
                if 'log_scale_factor' in key:
                    if 'ee' in key:
                        ee_log_scale_factors.append(key)
                    else:
                        log_scale_factors.append(key)
                if 'normalizer_threshold_factor' in key:
                    if 'ee' in key:
                        ee_normalizer_threshold_factors.append(key)
                    else:
                        normalizer_threshold_factors.append(key)

        assert len(log_scale_factors) == len(normalizer_threshold_factors)
        assert len(ee_log_scale_factors) == len(ee_normalizer_threshold_factors)

        # ensure the order is correct
        log_scale_factors = sorted(log_scale_factors, key=lambda f: int(f.split('.')[1]))
        ee_log_scale_factors = sorted(ee_log_scale_factors, key=lambda f: int(f.split('.')[2]))
        normalizer_threshold_factors = sorted(normalizer_threshold_factors, key=lambda f: int(f.split('.')[1]))
        ee_normalizer_threshold_factors = sorted(ee_normalizer_threshold_factors, key=lambda f: int(f.split('.')[2]))

        log_scale_factors.extend(ee_log_scale_factors)
        normalizer_threshold_factors.extend(ee_normalizer_threshold_factors)

        log_scale_factors = [state_dict[key].cpu().to(torch.float32).numpy() for key in log_scale_factors]
        normalizer_threshold_factors = [
            state_dict[key].cpu().to(torch.float32).numpy() for key in normalizer_threshold_factors
        ]

        log_scale_factors = np.stack(log_scale_factors)  # (NG, LD)
        normalizer_threshold_factors = np.stack(normalizer_threshold_factors)

        log_scale_factors.astype(np.float32).tofile(expected_log_scale_path)
        normalizer_threshold_factors.astype(np.float32).tofile(expected_normalizer_factors_path)

    normalizer_threshold = getattr(config, 'fire_normalizer_threshold', 8192)  # nano2v2 default is 8192

    # NG: number of global layers, LD: log scale dimension (1 for n2v2), L: context window
    def _position_pre_process(relative_position, log_scale_factors):
        x = relative_position.reshape(1, relative_position.shape[0], -1, 1) * log_scale_factors.reshape(
            log_scale_factors.shape[0], 1, 1, -1
        )  # (L, L) * (NG, LD) -> (NG, L, L, LD)
        x = np.log(np.abs(x) + 1)
        sign = np.sign(relative_position)  # (L, L)
        x *= sign[None, :, :, None]  # (NG, L, L, LD)
        return x

    def _position_normalizer(query_pos, log_scale_factors, normalizer_threshold_factors, normalizer_threshold):
        threshold = normalizer_threshold
        threshold *= np.abs(normalizer_threshold_factors)  # (1, ) * (NG, 1) -> (NG, 1)
        query_pos = np.clip(query_pos, threshold[:, :, None], None)  # (L, 1) clip by (NG, 1) to (NG, L, 1)
        x = np.expand_dims(query_pos, -1) * log_scale_factors.reshape(
            log_scale_factors.shape[0], 1, 1, -1
        )  # (NG, L, 1, 1) * (NG, 1, 1, LD) -> (NG, L, 1, LD)
        return np.log(np.absolute(x) + 1)

    q_segment_pos = np.arange(length)
    expanded_q_segment_pos = np.expand_dims(q_segment_pos, -1)
    k_segment_pos = np.arange(length)

    relative_position = np.expand_dims(k_segment_pos, -2) - expanded_q_segment_pos
    relative_position = _position_pre_process(relative_position, log_scale_factors)  # (NG, L, L, LD)
    processed_query = _position_normalizer(
        expanded_q_segment_pos,
        log_scale_factors,
        normalizer_threshold_factors,
        normalizer_threshold,  # (NG, L, 1, LD)
    )
    relative_position = relative_position / (processed_query + 1)  # (NG, L, L, LD)

    relative_bias = np.expand_dims(relative_position, -1)  # (NG, L, L, LD, 1)
    relative_bias = np.transpose(relative_bias, (0, 3, 4, 1, 2))  # (NG, LD, 1, L, L)

    # dump theoretical minmax for each fire layer
    theoretical_minmax = []
    theoretical_minmax_path = os.path.join(weight_dir, f'embedding_fire_pe_minmax_{length}.bin')
    theoretical_min = np.min(relative_bias, axis=tuple(range(1, relative_bias.ndim)))
    theoretical_max = np.max(relative_bias, axis=tuple(range(1, relative_bias.ndim)))

    theoretical_minmax = np.stack([theoretical_min, theoretical_max], axis=1)  # (NG, LD)
    theoretical_minmax.astype(np.float32).tofile(theoretical_minmax_path)

    # if LD is dim 1, squeeze it
    if relative_bias.shape[1] == 1:
        relative_bias = np.squeeze(relative_bias, axis=1)

    def convert_dtype(data):
        """Convert dtype of data."""
        if isinstance(dtype, torch.dtype):
            if isinstance(data, torch.Tensor):
                return data.to(dtype)
            return torch.from_numpy(data).to(dtype)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(dtype)
        return data.astype(dtype)

    # reshape relative position and query pos
    relative_position = np.transpose(relative_position, (0, 3, 1, 2))
    processed_query = np.transpose(processed_query, (0, 3, 1, 2))

    return convert_dtype(relative_position), convert_dtype(processed_query)
