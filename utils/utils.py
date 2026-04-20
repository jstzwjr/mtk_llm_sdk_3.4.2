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
"""Define common helper functions for mtk_llm_sdk."""

import argparse
import gc
import importlib
import json
import os
import shutil
import traceback
import types
from contextlib import contextmanager

import numpy as np
import torch
from safetensors.torch import load
from sentencepiece import SentencePieceProcessor

from .. import __version__
from ..tokenizers.tokenization_milm import MiTokenizer
from . import logger


def get_compatible_lora_config(lora_config_):
    """Return lora config compatible to 3.0."""
    if 'llm' in lora_config_:
        # Post-3.0
        if 'r' not in lora_config_['llm']:
            logger.error('`r` (rank) must be one of the keys of LLM lora config', err=KeyError)
        if 'target_modules' not in lora_config_['llm']:
            logger.error('`target_modules` must be one of the keys of LLM lora config', err=KeyError)

    if 'encoder' in lora_config_:
        logger.error('Adding dynamic LoRA inputs to encoder is currently not supported', err=NotImplementedError)
        # Post-3.0
        if 'r' not in lora_config_['encoder']:
            logger.error('`r` (rank) must be one of the keys of encoder lora config', err=KeyError)
        if 'target_modules' not in lora_config_['encoder']:
            logger.error('`target_modules` must be one of the keys of encoder lora config', err=KeyError)

    if 'llm' not in lora_config_ and 'encoder' not in lora_config_:
        # Pre-3.0
        if 'r' not in lora_config_:
            logger.error('`r` (rank) must be one of the keys of lora config', err=KeyError)
        if 'target_modules' not in lora_config_:
            logger.error('`target_modules` must be one of the keys of lora config', err=KeyError)
        rotate = lora_config_.pop('rotate', False)
        lora_config = {
            'llm': lora_config_,
            'rotate': rotate,
        }
    else:
        lora_config = lora_config_

    return lora_config


def create_lora_bin_for_cmdline(
    quantized_model_path, chunk_idx, pipeline, lora_precision, subgraph=None, encoder=False
):
    """Dumps the LoRA binary file for command line usage.

    Args:
        quantized_model_path (str): The path to the reference quantized model. Needed to quantize the LoRA bins.
        chunk_idx (int): The chunk index of the provided quantized model.
        pipeline (Object): The Pipeline object.
        lora_precision (float): The expected LoRA precision.
        subgraph (optional): The NIR subgraph.
        encoder (bool, optional): Flag to dump encoder lora bin instead of LLM. Defaults to False.
    """
    import struct

    from mtk_converter.python.utils import tensor_utils

    def get_type_converter(type_name):
        def get_integer_limits(dtype):
            assert np.issubdtype(dtype, np.integer)
            dtype_info = np.iinfo(dtype)  # Use np.finfo to get floating limits
            return dtype_info.min, dtype_info.max

        def get_caster(dtype):
            return lambda x: x.astype(dtype)

        def get_quantizer(dtype):
            return lambda x, qscale: np.clip(np.round(x / qscale), *get_integer_limits(dtype)).astype(dtype)

        type_converter_map = {
            'int8': get_quantizer(np.int8),
            'int16': get_quantizer(np.int16),
            'fp16': get_caster(np.float16),
            'fp32': get_caster(np.float32),
        }
        return type_converter_map[type_name.lower()]

    def get_header(version, num_lora_inputs):
        return struct.pack('<II', version, num_lora_inputs)

    def get_nbytes(nbytes):
        return struct.pack(f'<{len(nbytes)}I', *nbytes)

    quantized_model_dir = get_dirpath(quantized_model_path)
    lora_handler = pipeline.lora_handler

    if lora_precision == 'float':
        lora_precision = 'fp16'

    num_lora_configs = len(lora_handler.lora_config_paths)

    if subgraph is None:
        quantized_model_info = get_quantized_model_info_json(quantized_model_path, chunk_idx)

    for lora_id in range(num_lora_configs):
        lora_cfg = lora_handler.lora_config_paths[lora_id]
        # lora_inputs should have been prescaled during lora_handler load_state_dict
        lora_inputs = (
            lora_handler.encoder_lora_inputs[chunk_idx] if encoder else lora_handler.llm_lora_inputs[chunk_idx]
        )
        if len(lora_inputs) == 1:
            if lora_id == 0:
                # task = ptq, extra lora_id nested list
                lora_inputs = lora_inputs[lora_id]
            else:
                logger.error(f'Detected multiple ({num_lora_configs}) LoRA configs but only got 1 set of LoRA inputs.')
        else:
            if len(lora_inputs) == num_lora_configs:
                # Multiple LoRA configs
                # FIXME: This check is dangerous and will fail if there's 2 lora configs and only 1 target FC
                lora_inputs = lora_inputs[lora_id]
            else:
                # Task != ptq, no nested lora_id list
                pass

        if lora_inputs is None:
            continue

        lora_name = get_dirpath(lora_cfg, full=False)
        if lora_handler.rotated and not encoder:
            # For rotated lora_cfg model_name/lora_A/rotated_hadamard_0/adapter_config.json
            # => make lora_name 'lora_A/rotated_hadamard_0'
            # However, QALFT does not have the same naming as PTQ with lora
            # so the lora_name should be reconstructed directly from rotation settings
            lora_name = f'{lora_name}/rotate_{pipeline.config.rotate_mode}_{pipeline.config.rotate_seed}'
        output_dir = os.path.join(quantized_model_dir, lora_name)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'lora_chunk_{chunk_idx}.bin')
        if os.path.exists(output_path):
            logger.warning(f'Skip create lora bin as exist: {output_path}')
            continue

        lora_weights_to_merge = []
        for i in range(len(lora_inputs)):
            idx = len(lora_inputs) - i
            type_converter = get_type_converter(lora_precision)
            lora_input = lora_inputs[i]
            # currently qalft only uses this
            if lora_precision == 'fp16':
                lora_weights_to_merge.append(type_converter(lora_input))
            else:
                if subgraph is not None:
                    # scale is a list
                    # This scale should already be for the pre-scaled lora
                    # FIXME: if we let qalft dump int bins, this will be an issue
                    scale = tensor_utils.get_linear_quant_param(subgraph.tensor_map[subgraph.inputs[-idx]]).scale_vals
                else:
                    scale = [quantized_model_info['input_scales'][-idx]]
                lora_weights_to_merge.append(type_converter(lora_input, scale))

        lora_weight_nbytes = [klora_weights.nbytes for klora_weights in lora_weights_to_merge]
        merged_lora_weights_chunk = np.concatenate(lora_weights_to_merge, axis=None)

        num_lora_inputs_per_chunk = len(lora_weights_to_merge)
        logger.debug(f'Writing {num_lora_inputs_per_chunk} LoRA weights')
        header = get_header(1, num_lora_inputs_per_chunk)
        nbytes = get_nbytes(lora_weight_nbytes)
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(nbytes)
            f.write(merged_lora_weights_chunk.tobytes())

        logger.info(f'Writing {lora_name} LoRA to {output_path}')
        logger.debug(f'sizeof(header): {len(header)}')
        logger.debug(f'sizeof(weight): {merged_lora_weights_chunk.nbytes}')
        logger.debug(f'LoRA input sizes: {lora_weight_nbytes}')


def cleanup_pipeline(pipeline):
    """Cleanup the pipeline.

    Args:
        pipeline (Object): The Pipeline object.
    """
    for attr in list(vars(pipeline).keys()):
        cur = getattr(pipeline, attr)
        if isinstance(cur, torch.nn.Module):
            setattr(pipeline, attr, cur.cpu())
        elif isinstance(cur, torch.Tensor):
            setattr(pipeline, attr, cur.detach().cpu())
        delattr(pipeline, attr)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def _get_embedding_state_dict(config, weight_dir=None, state_dict=None):
    if weight_dir is None and state_dict is None:
        logger.error('Expect either weight_dir or state_dict, but got neither')
    if weight_dir is not None and state_dict is not None:
        logger.error('Expect only either weight_dir or state_dict, not both')

    try_last = False
    checkpoint_filename = None
    if weight_dir is not None:
        expected_path = os.path.join(weight_dir, 'embedder_model.bin')
        if os.path.exists(expected_path):
            state_dict = torch.load(expected_path)
        else:
            checkpoint_files = [
                os.path.join(weight_dir, f)
                for f in os.listdir(weight_dir)
                if (
                    (f.startswith('pytorch_model') and f.endswith('.bin'))
                    or (f.startswith('model') and f.endswith('.safetensors'))
                )
            ]

            for f in checkpoint_files:
                if (
                    'pytorch_model.bin' in f
                    or 'pytorch_model-00001-of' in f
                    or 'model.safetensors' in f
                    or 'model-00001-of' in f
                ):
                    checkpoint_filename = f
                    break
            if checkpoint_filename is None:
                logger.error(
                    f'Unable to find the first checkpoint file in {weight_dir}! '
                    'This folder must have either the file pytorch_model.bin or '
                    'pytorch_model-00001-of-XXXXX.bin or '
                    'model.safetensors or '
                    'model-00001-of-XXXXX.safetensors.',
                    err=FileNotFoundError,
                )

            if checkpoint_filename.endswith('.bin'):
                state_dict = torch.load(checkpoint_filename, map_location='cpu')
            elif checkpoint_filename.endswith('.safetensors'):
                state_dict = load_file(checkpoint_filename)
            try_last = True

    state_dict_keys = list(state_dict.keys())

    expected_embedding_prefix = config.embedding_prefix + '.'

    embedder_sd = {}
    embed_prefix = None
    for key in state_dict_keys:
        if expected_embedding_prefix in key:
            embed_prefix = expected_embedding_prefix
            break
    if embed_prefix is None:
        if try_last:
            if checkpoint_filename == 'pytorch_model.bin' or checkpoint_filename == 'model.safetensors':
                logger.debug(f'state_dict keys: {state_dict_keys}')
                logger.error(
                    f'Cannot find embedding layer weight inside {checkpoint_filename}. '
                    f'Please ensure embedding layer weight key contains {expected_embedding_prefix}',
                    err=KeyError,
                )
            checkpoint_filename = checkpoint_filename.replace('00001', checkpoint_filename.split('-')[-1].split('.')[0])
            if checkpoint_filename.endswith('.bin'):
                state_dict = torch.load(checkpoint_filename, map_location='cpu')
            elif checkpoint_filename.endswith('.safetensors'):
                state_dict = load_file(checkpoint_filename)
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                if expected_embedding_prefix in key:
                    embed_prefix = key
                    break
            if embed_prefix is None:
                logger.debug(f'state_dict keys: {state_dict_keys}')
                logger.error(
                    f'Cannot find embedding layer weight inside {checkpoint_filename}. '
                    f'Please ensure embedding layer weight key contains {expected_embedding_prefix}',
                    err=KeyError,
                )
        else:
            logger.debug(f'state_dict keys: {state_dict_keys}')
            logger.error(
                f'Cannot find embedding layer weight inside state dict. '
                f'Please ensure embedding layer weight key contains {expected_embedding_prefix}',
                err=KeyError,
            )
    for key in state_dict_keys:
        if embed_prefix in key:
            embed_key = key.replace(embed_prefix, '')
            embedder_sd[embed_key] = state_dict.pop(key)
    return embedder_sd


def get_embedding_weight(config, weight_dir=None, state_dict=None, dummy_weights=False):
    """Get embedding weight from config and state_dict."""
    if weight_dir is None and state_dict is None:
        logger.error('Expect either weight_dir or state_dict, but got neither')
    if weight_dir is not None and state_dict is not None:
        logger.error('Expect only either weight_dir or state_dict, not both')

    try_last = False
    checkpoint_filename = None
    if weight_dir is not None:
        expected_path = os.path.join(weight_dir, 'embedding_fp16.bin')
        if os.path.exists(expected_path) and not dummy_weights:
            embedding_weight = np.fromfile(expected_path, dtype=np.float16)
            return torch.from_numpy(
                embedding_weight.reshape(
                    -1 if getattr(config, 'bita', False) else config.vocab_size,
                    config.mlp_hidden_size if config.model_type == 'milm' else config.hidden_size,
                )
            ).to(torch.float32)
        checkpoint_files = [
            os.path.join(weight_dir, f)
            for f in os.listdir(weight_dir)
            if (
                (f.startswith('pytorch_model') and f.endswith('.bin'))
                or (f.startswith('model') and f.endswith('.safetensors'))
            )
        ]

        for f in checkpoint_files:
            if (
                'pytorch_model.bin' in f
                or 'pytorch_model-00001-of' in f
                or 'model.safetensors' in f
                or 'model-00001-of' in f
            ):
                checkpoint_filename = f
                break
        if checkpoint_filename is None:
            if dummy_weights:
                logger.warning(
                    f'Unable to find the first checkpoint file in {weight_dir}! '
                    'This folder must have either the file pytorch_model.bin or '
                    'pytorch_model-00001-of-XXXXX.bin or '
                    'model.safetensors or '
                    'model-00001-of-XXXXX.safetensors if performing real PTQ else dummy weights will be loaded.',
                )

                return torch.randn(
                    config.vocab_size, config.mlp_hidden_size if config.model_type == 'milm' else config.hidden_size
                )

            logger.error(
                f'Unable to find the first checkpoint file in {weight_dir}! '
                'This folder must have either the file pytorch_model.bin or '
                'pytorch_model-00001-of-XXXXX.bin or '
                'model.safetensors or '
                'model-00001-of-XXXXX.safetensors.',
                err=FileNotFoundError,
            )

        if checkpoint_filename.endswith('.bin') and 'training_args' not in checkpoint_filename:
            state_dict = torch.load(checkpoint_filename, map_location='cpu')
        elif checkpoint_filename.endswith('.safetensors'):
            state_dict = load_file(checkpoint_filename)
        try_last = True

    state_dict_keys = list(state_dict.keys())

    expected_embedding_subkey = config.embedding_key

    embed_key = None
    for key in state_dict_keys:
        if expected_embedding_subkey in key:
            embed_key = key
            break
    if embed_key is None:
        if try_last:
            if checkpoint_filename == 'pytorch_model.bin' or checkpoint_filename == 'model.safetensors':
                logger.debug(f'state_dict keys: {state_dict_keys}')
                logger.error(
                    f'Cannot find embedding layer weight inside {checkpoint_filename}. '
                    f'Please ensure embedding layer weight key contains {expected_embedding_subkey}',
                    err=KeyError,
                )
            checkpoint_filename = checkpoint_filename.replace('00001', checkpoint_filename.split('-')[-1].split('.')[0])
            if checkpoint_filename.endswith('.bin') and 'training_args' not in checkpoint_filename:
                state_dict = torch.load(checkpoint_filename, map_location='cpu')
            elif checkpoint_filename.endswith('.safetensors'):
                state_dict = load_file(checkpoint_filename)
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                if expected_embedding_subkey in key:
                    embed_key = key
                    break
            if embed_key is None:
                logger.debug(f'state_dict keys: {state_dict_keys}')
                logger.error(
                    f'Cannot find embedding layer weight inside {checkpoint_filename}. '
                    f'Please ensure embedding layer weight key contains {expected_embedding_subkey}',
                    err=KeyError,
                )
        else:
            logger.debug(f'state_dict keys: {state_dict_keys}')
            logger.error(
                f'Cannot find embedding layer weight inside state dict. '
                f'Please ensure embedding layer weight key contains {expected_embedding_subkey}',
                err=KeyError,
            )

    if hasattr(config, 'bita') and config.bita is True:
        return load_bita_embeddings(config, state_dict, embed_key)

    return state_dict[embed_key]


def load_bita_embeddings(config, state_dict, embed_key):
    """Load and merge BiTA embedding weights with the model's embeddings."""
    if not config.bita:
        return state_dict[embed_key]

    try:
        # Load and parse BiTA config
        with open(config.bita_config) as f:
            bita_config = json.load(f)

        # Extract required config values
        weight_path = bita_config.get('weight_path')
        embedding_key = bita_config.get('embedding_key_name')

        if not weight_path or not embedding_key:
            logger.error('Missing required keys in BiTA config', err=ValueError)

        # Load BiTA weights and merge embeddings
        bita_weights = torch.load(weight_path)
        bita_embedding = bita_weights[embedding_key].to(state_dict[embed_key].device)

        return torch.cat((state_dict[embed_key], bita_embedding))

    except FileNotFoundError:
        logger.error(f'BiTA config or weights file not found: {config.bita_config}', err=FileNotFoundError)
    except json.JSONDecodeError:
        logger.error(f'Invalid JSON format in BiTA config: {config.bita_config}', err=ValueError)
    except KeyError as e:
        logger.error(f'Missing key in BiTA weights: {e!s}', err=KeyError)
    except Exception as e:
        logger.error(f'Failed to load BiTA embeddings: {e!s}', err=RuntimeError)


def quantize(tensor, scale=None, max_=None, bitwidth=16, zp=0, signed=True):
    """Quantizes an input tensor based on the given scale and zero point.

    Args:
        tensor (torch.Tensor or np.ndarray): The tensor to be quantized.
        scale (float, optional): The scale to use to quantize the tensor. Either max_ or scale must be given.
        max_ (float, optional): The maximum float value this tensor can reach. Either max_ or scale must be given.
        bitwidth (int, optional): The bitwidth to quantize the tensor to. Defaults to 16.
        zp (int, optional): The zero point to use to quantize the tensor. Defaults to 0.
        signed (bool, optional): Quantize the tensor using a signed or unsigned integer data type.

    Returns:
        torch.Tensor or np.ndarray: The quantized tensor.
    """
    if scale is None and max_ is None:
        # Assume FP16
        assert bitwidth == 16
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float16)
        if isinstance(tensor, torch.Tensor):
            return tensor.to(torch.float16)
        logger.error(f'Unexpected tensor type: {type(tensor)}', err=TypeError)
    if signed:
        qmin = -(2 ** (bitwidth - 1))
        qmax = 2 ** (bitwidth - 1) - 1
        if scale is None:
            scale = max_ / qmax
    else:
        qmin = 0
        qmax = 2**bitwidth - 1
        if scale is None:
            scale = max_ / qmax

    if isinstance(tensor, np.ndarray):
        bitwidth_dtype_map = {
            4: np.int8 if signed else np.uint8,
            8: np.int8 if signed else np.uint8,
            16: np.int16 if signed else np.uint16,
        }
        out_dtype = bitwidth_dtype_map[bitwidth]
        return np.clip(np.round(tensor / scale) - zp, qmin, qmax).astype(out_dtype)
    if isinstance(tensor, torch.Tensor):
        if bitwidth == 16 and not signed:
            logger.error('torch does not support uint16 dtype')
        bitwidth_dtype_map = {
            4: torch.int8 if signed else torch.uint8,
            8: torch.int8 if signed else torch.uint8,
            16: torch.int16,
        }
        out_dtype = bitwidth_dtype_map[bitwidth]
        return torch.clip(torch.round(tensor / scale) - zp, qmin, qmax).to(out_dtype)
    logger.error(f'Unexpected tensor type: {type(tensor)}', err=TypeError)
    raise


def dequantize(tensor, scale, zp=0):
    """Dequantizes an input tensor based on the given scale and zero point.

    Args:
        tensor (torch.Tensor or np.ndarray): The tensor to be dequantized.
        scale (float, optional): The scale to use to dequantize the tensor.
        zp (int, optional): The zero point to use to dequantize the tensor. Defaults to 0.

    Returns:
        torch.Tensor or np.ndarray: The dequantized tensor.
    """
    return (tensor + zp) * scale


def _dump_pos_emb(pipeline, quant=True, bitwidth=None, quantized_model_folder=None):
    """Dumps the positions embedding lookup table (LUT) for command line usage.

    Args:
        pipeline (BasePipeline): The pipeline object.
        bitwidth (int, optional): The bitwidth for quantization if quant is True. Default is 16.
        quant (bool, optional): Whether to quantize the positions embedding. Default is True.
        quantized_model_folder (str, optional): The folder containing the quantized model files. Default is None.

    Raises:
        RuntimeError: If the conditions for generating the positions embedding LUT are not met.
    """
    if pipeline.config.l.model_type != 'whisper_decoder':
        return

    if quant:
        if bitwidth is None:
            embeds_precision = pipeline.precision_config.embeds_precision
            embeds_precision_name = pipeline.precision_config.get_precision_name(embeds_precision)[1]
            embeds_precision_bitwidth = pipeline.precision_config.get_bitwidth(embeds_precision)[1]
            postfix = embeds_precision_name
        else:
            postfix = f'int{bitwidth}'
        output_path = os.path.join(quantized_model_folder, f'embedding_pos_{postfix}.bin')
    else:
        if quantized_model_folder is None:
            output_path = os.path.join(pipeline.llm_weight_dir, 'embedding_pos_fp16.bin')
        else:
            output_path = os.path.join(quantized_model_folder, 'embedding_pos_fp16.bin')

    if pipeline.state_dict['llm'] is None:
        embedding = np.fromfile(
            os.path.join(pipeline.llm_weight_dir, 'embedding_pos_fp16.bin'), dtype=np.float16
        ).astype(np.float32)
    else:
        embedding = pipeline.state_dict['llm']['model.decoder.embed_positions.weight'].to(torch.float32).cpu().numpy()

    if quant:
        embedding_absmax = np.amax(np.abs(embedding))
        bitwidth = embeds_precision_bitwidth if bitwidth is None else bitwidth
        embedding = quantize(embedding, max_=embedding_absmax, bitwidth=bitwidth)
    else:
        embedding = embedding.astype(np.float16)

    os.makedirs(get_dirpath(output_path), exist_ok=True)
    embedding.tofile(output_path)
    print(
        f'{"Quantized INT" + embeds_precision_name.split("int")[1] if quant else "Float16"} '
        f'cmdline LUT position embedding bin exported to `{output_path}`.'
    )


def dump_embedding_lut_for_cmdline(
    pipeline,
    embedding=None,
    bitwidth=None,
    quant=True,
    quantized_model_folder=None,
    dummy_weights=False,
    embedding_scale_zp=None,
):
    """Dumps the embedding lookup table (LUT) for command line usage.

    Args:
        pipeline (BasePipeline): The pipeline object.
        embedding (np.ndarray, optional): The embedding array to be dumped. If not provided, will acquire from the
            pipeline. Default is None.
        bitwidth (int, optional): The bitwidth for quantization if quant is True. If not provided, will acquire from
            pipeline precision config. Default is None.
        quant (bool, optional): Whether to quantize the embedding. Default is True.
        quantized_model_folder (str, optional): The folder containing the quantized model files. Default is None.
        dummy_weights (bool, optional): Whether to use dummy embedding weights. Default is False.
        embedding_scale_zp (tuple, optional): Whether to use specified scale and zp to quantize embedding.

    Raises:
        RuntimeError: If the conditions for generating the embedding LUT are not met.
    """
    if quant and quantized_model_folder is None:
        logger.error('`quantized_model_folder` cannot be None when `quant`=True.')

    _dump_pos_emb(pipeline, quant=quant, bitwidth=bitwidth, quantized_model_folder=quantized_model_folder)

    if quant:
        if bitwidth is None:
            embeds_precision = pipeline.precision_config.embeds_precision
            embeds_precision_name = pipeline.precision_config.get_precision_name(embeds_precision)[1]
            embeds_precision_bitwidth = pipeline.precision_config.get_bitwidth(embeds_precision)[1]
            postfix = embeds_precision_name
        else:
            postfix = f'int{bitwidth}'
        output_path = os.path.join(quantized_model_folder, f'embedding_{postfix}.bin')
    else:
        if quantized_model_folder is None:
            output_path = os.path.join(pipeline.llm_weight_dir, 'embedding_fp16.bin')
        else:
            output_path = os.path.join(quantized_model_folder, 'embedding_fp16.bin')

    if os.path.exists(output_path):
        return

    if embedding is None:
        if pipeline.state_dict['llm']:
            embedding = get_embedding_weight(
                pipeline.config.l, state_dict=pipeline.state_dict['llm'], dummy_weights=dummy_weights
            )
        else:
            embedding = get_embedding_weight(
                pipeline.config.l, weight_dir=pipeline.llm_weight_dir, dummy_weights=dummy_weights
            )

        embedding = embedding.to(torch.float32).cpu().numpy()

    if quant:
        bitwidth = embeds_precision_bitwidth if bitwidth is None else bitwidth
        if embedding_scale_zp is not None:
            embedding = quantize(embedding, scale=embedding_scale_zp[0], zp=embedding_scale_zp[1], bitwidth=bitwidth)
        else:
            embedding_absmax = np.amax(np.abs(embedding))
            embedding = quantize(embedding, max_=embedding_absmax, bitwidth=bitwidth)
    else:
        embedding = embedding.astype(np.float16)

    os.makedirs(get_dirpath(output_path), exist_ok=True)
    embedding.tofile(output_path)
    logger.info(
        f'{"Quantized INT" + str(bitwidth) if quant else "Float16"} '
        f'cmdline LUT embedding bin exported to `{output_path}`.'
    )


def dump_embedding_tflite_for_cmdline(pipeline, quant=False, quantized_model_folder=None, dummy_weights=False):
    """Dumps the embedding tflite cmdline usage.

    Args:
        pipeline (BasePipeline): The pipeline object.
        quant (bool, optional): Whether to quantize the embedding. Default is True.
        quantized_model_folder (str, optional): The folder containing the quantized model files. Default is None.
        dummy_weights (bool, optional): Whether to use dummy embedding weights. Default is False.

    Raises:
        RuntimeError: If the conditions for generating the embedding LUT are not met.
    """
    import mtk_converter

    if quant and quantized_model_folder is None:
        logger.error('`quantized_model_folder` cannot be None when `quant`=True.')

    embeds_precision = pipeline.precision_config.embeds_precision
    embeds_precision_name = pipeline.precision_config.get_precision_name(embeds_precision)[1]
    embeds_precision_bitwidth = pipeline.precision_config.get_bitwidth(embeds_precision)[1]

    if quant:
        output_path = os.path.join(quantized_model_folder, f'embedding_{embeds_precision_name}.tflite')
    else:
        if quantized_model_folder is None:
            output_path = os.path.join(pipeline.llm_weight_dir, 'embedder', 'embedding_fp16.tflite')
        else:
            output_path = os.path.join(quantized_model_folder, 'embedding_fp16.tflite')

    if os.path.exists(output_path):
        return

    os.makedirs(get_dirpath(output_path), exist_ok=True)

    if pipeline.state_dict['llm'] is None:
        embedder_model = get_embedding_layer(
            pipeline.config.l, weight_dir=pipeline.llm_weight_dir, dummy_weights=dummy_weights
        )
    else:
        embedder_model = get_embedding_layer(
            pipeline.config.l, state_dict=pipeline.state_dict['llm'], dummy_weights=dummy_weights
        )

    input_ids = torch.randint(3, pipeline.config.l.vocab_size, (1, 10)).to('cpu')

    logger.info('JIT tracing embedding layer')
    trace = torch.jit.trace(embedder_model, example_inputs=input_ids)

    input_shapes = [[None, None]]
    input_types = ['int32']

    converter = mtk_converter.PyTorchConverter(trace, input_shapes=input_shapes, input_types=input_types)

    if quant:
        converter.default_quantization_bitwidth = embeds_precision_bitwidth
        converter.use_symmetric_quantization = True
        converter.quantize = True
        converter.allow_missing_quantization_ranges = True
    else:
        converter.quantize = False

    converter.convert_to_tflite(
        output_file=output_path, custom_description=f'Post Training Quantized by mtk_llm_sdk v{__version__}'
    )
    logger.info(
        f'{"Quantized INT" + embeds_precision_name.split("int")[1] if quant else "Float16"} '
        f'cmdline LUT embedding bin exported to `{output_path}`.'
    )


def dump_infini_update_tflite_for_cmdline(pipeline, quant=False, quantized_model_folder=None):
    """Dumps the infini update tflite for cmdline usage.

    Args:
        pipeline (BasePipeline): The pipeline object.
        quant (bool, optional): Whether to quantize the embedding. Default is True.
        quantized_model_folder (str, optional): The folder containing the quantized model files. Default is None.

    Raises:
        RuntimeError: If the conditions for generating the embedding LUT are not met.
    """
    import mtk_converter

    from .cache_utils import InfiniUpdate

    if quant and quantized_model_folder is None:
        logger.error('`quantized_model_folder` cannot be None when `quant`=True.')

    if quant:
        output_path = os.path.join(
            quantized_model_folder, f'infini_update_{pipeline.precision_config.infini_update_precision}.tflite'
        )
    else:
        if quantized_model_folder is None:
            output_path = os.path.join(pipeline.llm_weight_dir, 'infini_update', 'infini_update_fp.tflite')
        else:
            output_path = os.path.join(quantized_model_folder, 'infini_update_fp.tflite')

    if os.path.exists(output_path):
        return

    os.makedirs(get_dirpath(output_path), exist_ok=True)

    infini_update_model = InfiniUpdate(pipeline.config.l, device='cpu')

    example_inputs = infini_update_model.get_jit_trace_inputs()

    logger.info('JIT tracing infini update layer')
    trace = torch.jit.trace(infini_update_model, example_inputs=example_inputs)

    # FIXME: currently support only float for infini update
    input_shapes, _, _ = infini_update_model.get_ptq_inputs()[:3]

    converter = mtk_converter.PyTorchConverter(trace, input_shapes=input_shapes, experimental_debug_tensor_names=True)

    if quant:
        converter.default_quantization_bitwidth = int(pipeline.precision_config.infini_update_precision.split('int')[1])
        converter.use_symmetric_quantization = True
        converter.quantize = True
        converter.allow_missing_quantization_ranges = True
    else:
        converter.quantize = False

    converter.convert_to_tflite(
        output_file=output_path, custom_description=f'Post Training Quantized by mtk_llm_sdk v{__version__}'
    )
    logger.info(
        f'{"Quantized INT" + pipeline.precision_config.infini_update_precision.split("int")[1] if quant else "Float"} '
        f'cmdline LUT embedding bin exported to `{output_path}`.'
    )


def enforce_add_bos_mode(add_bos, prompt_tokens, bos_token_id=None):
    """Cross checks the input prompt tokens with the Pipeline tokenizer's add_bos setting.

    Inserts/removes the BOS token if necessary.

    Args:
        add_bos (bool): Whether add_bos is needed.
        prompt_tokens (np.ndarray): The input prompt tokens.
        bos_token_id (int, optional): The bos token id.

    Returns:
        np.ndarray: The prompt tokens with bos token appropriately inserted or removed.
    """
    if len(prompt_tokens.shape) == 1:
        prompt_tokens = prompt_tokens[None, :]

    if add_bos and bos_token_id is None:
        logger.error('Add bos is set but bos token id is None!', err=ValueError)

    if add_bos and prompt_tokens[0][0] != bos_token_id:
        logger.warning('Force prepend BOS token')
        return np.pad(prompt_tokens, ((0, 0), (1, 0)), constant_values=bos_token_id)

    if add_bos and (prompt_tokens[0][:2] == bos_token_id).all():
        # Assume duplicate bos can only be found in 1st two tokens across all batches
        logger.warning('Found duplicated BOS token. Removing duplicate.')
        return prompt_tokens[:, 1:]

    if not add_bos and prompt_tokens[0][0] == bos_token_id:
        logger.warning('Force remove BOS token')
        return prompt_tokens[:, 1:]

    return prompt_tokens


def evenly_distribute(total, each):
    """Evenly distributes a total amount into a specified number of parts.

    Args:
        total (int): The total amount to distribute.
        each (int): The number of parts to distribute into.

    Returns:
        list: A list of integers representing the distributed amounts.
    """
    return [(total // each) + (i < (total % each)) for i in range(each)]


def extract_emb_minmax(weight_dir):
    """Extracts the minimum and maximum values from the embedding file.

    Args:
        weight_dir (str): The directory containing the weight files.

    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
    emb = np.fromfile(os.path.join(weight_dir, 'embedding_fp16.bin'), dtype=np.float16)
    absmax = np.amax(np.abs(emb)).item()
    return -absmax, absmax


def load_file(path):
    """Wrapper around safetensors.torch.load.

    Use load instead of load_file based on:
    https://github.com/huggingface/safetensors/issues/369#issuecomment-2760415741
    https://github.com/huggingface/safetensors/issues/200#issuecomment-1478326999

    Args:
        path (str): Path to checkpoint/state dict file

    Returns:
        dict: Dictionary that contains name as key, value as `torch.Tensor` on cpu
    """
    with open(path, 'rb') as f:
        return load(f.read())


def get_converter_calibration_methods():
    """Gets all valid operators supported by mtk_converter.

    Returns:
        A list of all valid operators supported by mtk_converter.
    """
    import inspect

    from mtk_converter.python.utils.defs import CalibrationMethod

    attributes = inspect.getmembers(CalibrationMethod, lambda a: not (inspect.isroutine(a)))
    return [a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]


def get_converter_ops():
    """Gets all valid operators supported by mtk_converter.

    Returns:
        A list of all valid operators supported by mtk_converter.
    """
    import inspect

    from mtk_converter.python.utils.defs import Op

    attributes = inspect.getmembers(Op, lambda a: not (inspect.isroutine(a)))
    return [a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]


def get_converter_precisions():
    """Gets all valid precisions supported by mtk_converter.

    Returns:
        A list of all valid precision strings supported by mtk_converter.
    """
    import inspect

    from mtk_converter.python.utils.defs import PrecisionSetting

    attributes = inspect.getmembers(PrecisionSetting, lambda a: not (inspect.isroutine(a)))
    return [a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]


def get_converter_version(include_minor=False):
    """Gets the version of the converter.

    Args:
        include_minor (bool, optional): Whether to include the minor version. Default is False.

    Returns:
        int or tuple: The major version or a tuple of the major and minor versions.
    """
    import mtk_converter

    major, minor = mtk_converter.__version__.split('.')[:2]
    if include_minor:
        return int(major), int(minor)
    return int(major)


def get_dirpath(file_path, full=True):
    """Gets the directory name from a file path.

    Args:
        file_path (str): The file path.
        full (bool, optional): Flag to return the full parent directory path or just the name of the parent folder.

    Returns:
        str: The directory name.
    """
    if full:
        return os.path.dirname(file_path)
    return os.path.dirname(file_path).split('/')[-1]


def get_embedding_bin(config, quantized_model_folder):
    """Gets the embedding binary file from the quantized model folder.

    Args:
        config (object): The configuration object.
        quantized_model_folder (str): The folder containing the quantized model files.

    Returns:
        torch.nn.Embedding: The embedding layer.

    Raises:
        FileNotFoundError: If the embedding binary file is not found.
    """
    embedding_paths = [
        os.path.join(quantized_model_folder, f)
        for f in os.listdir(quantized_model_folder)
        if (f.startswith('embedding_') and f.endswith('.bin'))
    ]

    if config.model_type == 'whisper_decoder':
        embedding_paths = [s for s in embedding_paths if 'embedding_pos_' not in s]

    if len(embedding_paths) != 1:
        logger.error(f'Expect exactly one Embedding bin in `{quantized_model_folder}`.', err=FileNotFoundError)

    embedding_path = embedding_paths[0]
    embedding_dtype_str = os.path.basename(embedding_path).split('embedding_')[1].split('.bin')[0]
    embedding_dtype, embedding_torch_dtype = {
        'int8': (np.int8, torch.int8),
        'int16': (np.int16, torch.int16),
        # cast fp16 to fp32 to match quantized model input dtype
        'fp16': (np.float16, torch.float32),
    }[embedding_dtype_str]
    logger.info(f'Loading embedding weights from `{embedding_path}`.')
    embedding_weight = np.fromfile(embedding_path, dtype=embedding_dtype).astype(np.float16)
    embedding_weight = torch.from_numpy(
        embedding_weight.reshape(
            config.vocab_size,
            config.mlp_hidden_size if config.model_type == 'milm' else config.hidden_size,
        )
    )
    model = torch.nn.Embedding.from_pretrained(embedding_weight, padding_idx=-1)

    def forward_alt(self, inp):
        out = torch.nn.Embedding.forward(model, inp)
        return out.to(dtype=embedding_torch_dtype)

    model.forward = types.MethodType(forward_alt, model)
    return model


@torch.no_grad()
def get_embedding_layer(config, weight_dir=None, state_dict=None, dummy_weights=False):
    """Gets the embedding layer from the configuration and weight files.

    Args:
        config (object): The configuration object.
        weight_dir (str, optional): The directory containing the weight files. Default is None.
        state_dict (dict, optional): The state dictionary. Default is None.
        dummy_weights (bool, optional): Whether to load dummy embedding weights. Default is False.

    Returns:
        torch.nn.Embedding: The embedding layer.
    """
    if config.model_type == 'gecko2':
        from ..models.embedders.modeling_x import Gecko2Embedder

        text_embedding_layer = Gecko2Embedder(config)
        embedder_state_dict = _get_embedding_state_dict(config, weight_dir, state_dict)
        text_embedding_layer.load_state_dict(embedder_state_dict)  # temporary bypass audio and mm
        return text_embedding_layer

    embedding_weight = get_embedding_weight(config, weight_dir, state_dict, dummy_weights=dummy_weights).to(
        torch.float32
    )

    return torch.nn.Embedding.from_pretrained(embedding_weight, padding_idx=-1)


def get_exp_name(config_path, lora_path=None):
    """Gets the experiment name from the configuration path.

    Args:
        config_path (str): The path to the model config json file.
        lora_path (str, optional): The path to the lora adapter config json file.

    Returns:
        str: The experiment name.
    """
    weight_dir = get_dirpath(config_path)
    weight_name = os.path.basename(weight_dir)
    config_name = os.path.basename(config_path).split('.json')[0].replace('config', '')
    if config_name == '':
        exp_name = f'{weight_name}'
    else:
        if config_name.startswith('_'):
            config_name = config_name[1:]
        exp_name = f'{weight_name}_{config_name}'

    if lora_path is not None:
        exp_name += f'_{get_dirpath(lora_path, full=False)}'
    return exp_name


def get_hook_config(config_dict, hook_type, verbose=False):
    """Gets the hook configuration.

    Args:
        config_dict (dict): The configuration dictionary.
        hook_type (str): The type of the hook.
        verbose (bool, optional): Whether to print verbose output. Default is False.

    Returns:
        HookConfig: The hook configuration instance.
    """
    from ..models.configuration_hook import HookConfig

    if config_dict is None:
        config_dict = {}
    if config_dict.get('name', None) == 'qwen2vl_preencoder':
        from ..models.hooks.qwen2_vl_pre_encoder import Qwen2VLPreEncoderConfig

        return Qwen2VLPreEncoderConfig(**config_dict, type=hook_type, verbose=verbose)
    if config_dict.get('name', None) == 'qwen2vl_prellm':
        from ..models.hooks.qwen2_vl_pre_llm import Qwen2VLPreLLMConfig

        return Qwen2VLPreLLMConfig(**config_dict, type=hook_type, verbose=verbose)
    if config_dict.get('name', None) == 'internvl2_pixel_shuffle':
        from ..models.hooks.pixel_shuffle import InternVL2PixelShuffleConfig

        return InternVL2PixelShuffleConfig(**config_dict, type=hook_type, verbose=verbose)
    if config_dict.get('name', None) == 'phi3v_preprojector':
        from ..models.hooks.phi3v_preprojector import Phi3vPreprojectorConfig

        return Phi3vPreprojectorConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'get_embeds':
        if config_dict.get('name', None) == 'phi3v':
            from ..models.get_embeds.phi3v import Phi3VGetEmbedsConfig

            return Phi3VGetEmbedsConfig(**config_dict, type=hook_type, verbose=verbose)
        from ..models.configuration_hook import GetEmbedsHooksConfig

        return GetEmbedsHooksConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'format_text' and config_dict.get('name', None) == 'internvl2':
        from ..models.format_text.internvl2 import InternVL2FormatTextConfig

        return InternVL2FormatTextConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'format_text' and config_dict.get('name', None) == 'andesvl':
        from ..models.format_text.andesvl import AndesVLFormatTextConfig

        return AndesVLFormatTextConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'format_text' and config_dict.get('name', None) == 'minicpmv':
        from ..models.format_text.minicpmv import MinicpmVFormatTextConfig

        return MinicpmVFormatTextConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'tokenizer_func_hook' and config_dict.get('name', None) == 'phi3v':
        from ..models.tokenizer_func.phi3v import Phi3VFormatTextConfig

        return Phi3VFormatTextConfig(**config_dict, type=hook_type, verbose=verbose)
    if config_dict.get('name', None) == 'phi4o_img_preprojector':
        from ..models.hooks.phi4o_img_pre_projector import Phi4OImgPreprojectorConfig

        return Phi4OImgPreprojectorConfig(**config_dict, type=hook_type, verbose=verbose)
    if hook_type == 'tokenizer_func_hook' and config_dict.get('name', None) == 'phi4o':
        from ..models.tokenizer_func.phi4o import Phi4OTokenizerFuncConfig

        return Phi4OTokenizerFuncConfig(**config_dict, type=hook_type, verbose=verbose)

    if config_dict.get('name', None) == 'andesvl_preencoder':
        from ..models.hooks.andesvl_preencoder import AndesVLPreEncoderConfig

        return AndesVLPreEncoderConfig(**config_dict, type=hook_type, verbose=verbose)

    if config_dict.get('name', None) == 'qwen2_5vl_preencoder':
        from ..models.hooks.qwen2_5_vl_pre_encoder import Qwen2_5VLPreEncoderConfig

        return Qwen2_5VLPreEncoderConfig(**config_dict, type=hook_type, verbose=verbose)

    if config_dict.get('name', None) == 'bita_pre_getembed':
        from ..models.hooks.bita_pre_getembed import BiTAPreGetEmbedConfig

        return BiTAPreGetEmbedConfig(**config_dict, type=hook_type, verbose=verbose)

    return HookConfig(**config_dict, type=hook_type, verbose=verbose)


def get_multimodal_inputs_from_jsonl_line(line):
    """Extracts multi-modal input paths from a JSONL line.

    Args:
        line (dict): A dictionary representing a line from a JSONL file.

    Returns:
        list: A list of multi-modal input paths (images or audio).
    """
    mm_inputs = []
    for k, v in line.items():
        if 'image' in k or 'audio' in k:
            if isinstance(v, str):
                mm_inputs = [v]
            elif isinstance(v, list):
                mm_inputs = v
            else:
                logger.error(
                    f'Unsupported input for {k}. Only support string of single {k} path or list of {k}', err=ValueError
                )
    if len(mm_inputs) > 0:
        logger.debug(f'Found {len(mm_inputs)} multimodal inputs in prompt line.')
    return mm_inputs


def get_normalized_config(config_dict, module, return_class=False, verbose=True):
    """Gets the normalized configuration class or instance based on the model type.

    Args:
        config_dict (dict): The configuration dictionary.
        module (str): The type of module to resolve the configuration class for.
        return_class (bool, optional): Whether to return the configuration class instead of an instance.
            Default is False.
        verbose (bool, optional): Whether to print verbose output. Default is True.

    Returns:
        object: The configuration class or instance.

    Raises:
        KeyError: If `model_type` is not in the configuration dictionary.
        NotImplementedError: If the `model_type` is not supported.
    """
    logger.debug(f'Enter get_normalized_config, config_dict={config_dict}, module={module}')
    if config_dict is None:
        return None

    from .const import (
        PIPELINE_CORE_MODULES,
        SUPPORTED_ENCODERS,
        SUPPORTED_LLMS,
        SUPPORTED_MODELS,
        SUPPORTED_PREPROCESSORS,
        SUPPORTED_PROJECTORS,
    )

    if 'model_type' not in config_dict:
        logger.error("`model_type` must be given for each component's config", err=KeyError)
    if module not in PIPELINE_CORE_MODULES:
        logger.error(f'Got {module} for `module`, but must be one of: {PIPELINE_CORE_MODULES}')

    model_type = config_dict['model_type']
    if model_type not in SUPPORTED_MODELS:
        logger.error(f'Unsupported model_type: {model_type}', err=NotImplementedError)

    import_path = None
    class_name = None

    # Preprocessor and encoders have overlapping model_type
    if module == 'preprocessor':
        if model_type not in SUPPORTED_PREPROCESSORS:
            logger.error(f'Unsupported preprossor: {model_type}')
        if model_type == 'clip':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_clip'
            class_name = 'CLIPPreprocessorConfig'
        elif model_type == 'siglip':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_siglip'
            class_name = 'SiglipPreprocessorConfig'
        elif model_type == 'qwen2_vl':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_qwen2vl_vision'
            class_name = 'Qwen2VLPreprocessorConfig'
        elif model_type == 'minicpmv_navit_siglip':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_minicpmv_navit_siglip'
            class_name = 'MinicpmVNavitSigLIPPreprocessorConfig'
        elif model_type == 'intern_vit_6b':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_intern_vit'
            class_name = 'InternViTPreprocessorConfig'
        elif model_type == 'whisper':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_whisper'
            class_name = 'WhisperPreprocessorConfig'
        elif model_type == 'phi3v':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_phi3v_vision_processor'
            class_name = 'Phi3VPreprocessorConfig'
        elif model_type == 'gecko2_vision':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_x2_vision'
            class_name = 'Gecko2VisionPreprocessorConfig'
        elif model_type == 'phi4o_img':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_phi4o'
            class_name = 'Phi4OmniImgPreprocessorConfig'
        elif model_type == 'intern_vit_6b_navit_rope':
            import_path = 'mtk_llm_sdk.models.preprocessors.configuration_internvl_navit_rope'
            class_name = 'InternVLNavitRopePreprocessorConfig'
    elif module == 'encoder':
        if model_type not in SUPPORTED_ENCODERS:
            logger.error(f'Unsupported encoder: {model_type}')
        if model_type == 'clip':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_clip'
            class_name = 'CLIPConfig'
        elif model_type == 'siglip':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_siglip'
            class_name = 'SigLIPConfig'
        elif model_type == 'intern_vit_6b':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_intern_vit'
            class_name = 'InternViTConfig'
        elif model_type == 'qwen2_vl_vision':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_qwen2vl_vision'
            class_name = 'Qwen2VLVisionConfig'
        elif model_type == 'minicpmv_navit_siglip':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_minicpmv_navit_siglip'
            class_name = 'MinicpmVNavitSigLIPConfig'
        elif model_type == 'whisper':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_whisper'
            class_name = 'WhisperEncoderConfig'
        elif model_type == 'phi3_vision_emb':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_phi3v_img_embedding'
            class_name = 'Phi3VImgEmbeddingConfig'
        elif model_type == 'gecko2_vision':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_x2_vision'
            class_name = 'Gecko2VisionConfig'
        elif model_type == 'phi4o_navit_siglip':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_phi4o_img_encoder'
            class_name = 'Phi4ONavitSigLIPConfig'
        elif model_type == 'intern_vit_6b_navit_rope':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_internvl_navit_rope'
            class_name = 'InternViTNavitRopeConfig'
        elif model_type == 'qwen2_5_vl_vision':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_qwen2_5vl_vision'
            class_name = 'Qwen2_5VLVisionConfig'
        elif model_type == 'qwen2_audio_encoder':
            import_path = 'mtk_llm_sdk.models.encoders.configuration_qwen2_audio'
            class_name = 'Qwen2AudioEncoderConfig'
    elif module == 'projector':
        if model_type not in SUPPORTED_PROJECTORS:
            logger.error(f'Unsupported projector: {model_type}')
        if model_type == 'mlp_gelu':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_mlp_gelu'
            class_name = 'MLPGeluProjectorConfig'
        elif model_type == 'mlp_downsample':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_mlp_downsample'
            class_name = 'MLPDownsampleProjectorConfig'
        elif model_type == 'ldpnetv2':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_ldpnetv2'
            class_name = 'LDPNetV2ProjectorConfig'
        elif model_type == 'internvl2':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_internvl2'
            class_name = 'InternVL2ProjectorConfig'
        elif model_type in ['qwen2_vl', 'patch_merger', 'qwen2_5_vl']:
            import_path = 'mtk_llm_sdk.models.projectors.configuration_patchmerger'
            class_name = 'PatchMergerProjectorConfig'
        elif model_type == 'phi3v':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_phi3v'
            class_name = 'Phi3VProjectorConfig'
        elif model_type == 'gecko2_vision':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_x2_vision'
            class_name = 'Gecko2VisionProjectorConfig'
        elif model_type == 'phi4o':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_phi3v'
            class_name = 'Phi4OImgProjectorConfig'
        elif model_type == 'andesvl':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_andesvl'
            class_name = 'AndesVLProjectorConfig'
        elif model_type == 'qwen2_audio_encoder':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_qwen2_audio'
            class_name = 'Qwen2AudioProjectorConfig'
        elif model_type == 'minicpmv':
            import_path = 'mtk_llm_sdk.models.projectors.configuration_projector_minicpmv'
            class_name = 'MinicpmVProjectorConfig'
    elif module == 'llm':
        if model_type not in SUPPORTED_LLMS:
            logger.error(f'Unsupported LLM: {model_type}')
        if model_type == 'llama':
            import_path = 'mtk_llm_sdk.models.llm.configuration_llama'
            class_name = 'LlamaConfig'
        elif model_type == 'baichuan':
            import_path = 'mtk_llm_sdk.models.llm.configuration_baichuan'
            class_name = 'BaichuanConfig'
        elif model_type in ['qwen', 'qwen1.5', 'qwen2', 'qwen3']:
            import_path = 'mtk_llm_sdk.models.llm.configuration_qwen'
            class_name = 'QwenConfig'
        elif model_type == 'milm':
            import_path = 'mtk_llm_sdk.models.llm.configuration_milm'
            class_name = 'MiLMConfig'
        elif model_type == 'phi3':
            import_path = 'mtk_llm_sdk.models.llm.configuration_phi3'
            class_name = 'Phi3Config'
        elif model_type == 'hunyuan':
            import_path = 'mtk_llm_sdk.models.llm.configuration_hunyuan'
            class_name = 'HunYuanConfig'
        elif model_type == 'internlm2':
            import_path = 'mtk_llm_sdk.models.llm.configuration_internlm2'
            class_name = 'InternLM2Config'
        elif model_type == 'minicpm':
            import_path = 'mtk_llm_sdk.models.llm.configuration_minicpm'
            class_name = 'MinicpmConfig'
        elif model_type in ['gemma', 'gemma2', 'gemma3']:
            import_path = 'mtk_llm_sdk.models.llm.configuration_gemma'
            class_name = 'GemmaConfig'
        elif model_type in ['gecko', 'gecko2']:
            import_path = 'mtk_llm_sdk.models.llm.configuration_x'
            class_name = 'GeckoConfig'
        elif model_type == 'whisper_decoder':
            import_path = 'mtk_llm_sdk.models.llm.configuration_whisper'
            class_name = 'WhisperDecoderConfig'
        elif model_type == 'phi4':
            import_path = 'mtk_llm_sdk.models.llm.configuration_phi3'
            class_name = 'Phi4Config'
    elif module == 'tail':
        if model_type != 'medusa':
            logger.error(f'Unsupported tail: {model_type}')
        import_path = 'mtk_llm_sdk.models.llm.configuration_medusa'
        class_name = 'MedusaConfig'

    if import_path is None or class_name is None:
        logger.error(f'Unknown `model_type`: {model_type}', err=KeyError)

    config_class = getattr(importlib.import_module(import_path), class_name)

    if return_class:
        return config_class
    return config_class(**config_dict, verbose=verbose)


def get_op_export_spec(np_version, file_format):
    """Gets the op export spec for the respective quantized model exporter based on Neuropilot compatibility version.

    Args:
        np_version (int): The Neuropilot version to export the quantized model to.
        file_format (str): The output file format of the quantized model, must be either `tflite` or `mlir`.

    Returns:
        list: A sorted list of file paths.
    """
    from mtk_converter.python.converters.mlir import defs as mlir_defs
    from mtk_converter.python.converters.tflite import defs as tflite_defs

    cvt_version = get_converter_version()
    if np_version is None:
        np_version = cvt_version
    if np_version > cvt_version:
        logger.error(f'np_version ({np_version}) cannot be more than mtk_converter version ({cvt_version})')

    if np_version == cvt_version:
        return tflite_defs.OpExportSpec.BUILTIN_FIRST if file_format == 'tflite' else mlir_defs.ExportSpec.DEFAULT
    if np_version == 7:
        return tflite_defs.OpExportSpec.NPSDK_V7 if file_format == 'tflite' else mlir_defs.ExportSpec.NPSDK_V7
    if np_version == 8:
        return tflite_defs.OpExportSpec.NPSDK_V8 if file_format == 'tflite' else mlir_defs.ExportSpec.NPSDK_V8
    raise


def get_quantized_model_info_json(quantized_model_info_path, chunk_idx=None):
    """Get the quantized model info json from the quantized model path.

    Args:
        quantized_model_info_path (str): Path to either the quantized model file or the quantized model info file.
        chunk_idx (int, optional): Current chunk index.

    Returns:
        dict: The quantized model info.
    """
    _, ext = os.path.splitext(quantized_model_info_path)
    if ext in ('.tflite', '.mlir'):
        if chunk_idx is None:
            if ext == '.tflite':
                chunk_idx = quantized_model_info_path.replace('.tflite', '').rsplit('_', 1)[-1]
            else:
                chunk_idx = quantized_model_info_path.replace('.mlir', '').rsplit('_', 1)[-1]
        quantized_model_info_path = quantized_model_info_path.replace(ext, '.json').replace(
            f'_{chunk_idx}', f'_info_{chunk_idx}'
        )

    if not os.path.exists(quantized_model_info_path):
        logger.error(f'Quantized model info path: {quantized_model_info_path} not found', err=FileNotFoundError)

    with open(quantized_model_info_path) as f:
        quantized_model_info = json.load(f)
        logger.debug(f'[get_quantized_model_info_json] Quantized model info:\n{quantized_model_info}\n')
        return quantized_model_info


def get_sorted_path_list(folder, ext=None, sep='_'):
    """Gets a sorted list of file paths in a folder based on a specified extension and separator.

    Args:
        folder (str): The folder containing the files.
        ext (str, optional): The file extension to filter by. Default is None (do not filter by extension).
        sep (str, optional): The separator used in the file names. Default is '_'.

    Returns:
        list: A sorted list of file paths.
    """
    if ext is None:
        ext = '.'

    if isinstance(ext, (list, tuple)):
        files = []
        for e in ext:
            curr_files = [x for x in os.listdir(folder) if x.endswith(e)] if e != '.' else os.listdir(folder)

            if len(curr_files) > 0:
                found_ext = e
            files.append(curr_files)
        files = [x for x in files if len(x) > 0]
        if len(files) == 0:
            return []
        if len(files) > 1:
            logger.error(
                f'Expected exactly 1 extension match in folder {folder}, but got {len(files)} extension matches',
                err=RuntimeError,
            )
        if sep is not None:
            sorted_list = sorted(files[0], key=lambda f: int(f.rsplit(sep, 1)[1].split(found_ext)[0]))
        else:
            sorted_list = sorted(files[0])
    else:
        files = [x for x in os.listdir(folder) if x.endswith(ext)] if ext != '.' else os.listdir(folder)
        if sep is not None:
            sorted_list = sorted(files, key=lambda f: int(f.rsplit(sep, 1)[1].split(ext)[0]))
        else:
            sorted_list = sorted(files)
    return [os.path.join(folder, x) for x in sorted_list]


def get_tokenizer(config, llm_weight_dir, add_bos=True):
    """Gets the tokenizer based on the provided configuration and weight directory.

    Args:
        config (object): The configuration object.
        llm_weight_dir (str): The directory containing the LLM weights.
        add_bos (bool, optional): Whether to add a BOS (beginning of sentence) token. Default is True.

    Returns:
        object: The tokenizer instance.

    Raises:
        NotImplementedError: If no default tokenizer is implemented for the specified model type.
    """
    if config.tokenizer == 'default':
        if config.model_type in ['llama', 'phi3', 'minicpm']:
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_llama'
            tokenizer_class_name = 'LlamaTokenizer'
        elif config.model_type == 'baichuan':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_baichuan'
            tokenizer_class_name = 'BaichuanTokenizer'
        elif config.model_type == 'qwen':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_qwen'
            tokenizer_class_name = 'QWenTokenizer'
        elif config.model_type in ['qwen1.5', 'qwen2', 'qwen3']:
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_qwen2_fast'
            tokenizer_class_name = 'Qwen2TokenizerFast'
        elif config.model_type in ['gemma', 'gemma2', 'gemma3']:
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gemma_fast'
            tokenizer_class_name = 'GemmaTokenizerFast'
        elif config.model_type == 'milm':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_milm'
            tokenizer_class_name = 'MiTokenizer'
        elif config.model_type in ['gecko', 'gecko2']:
            import_path = 'sentencepiece'
            tokenizer_class_name = 'SentencePieceProcessor'
        elif config.model_type == 'hunyuan':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_hunyuan'
            tokenizer_class_name = 'HunYuanTokenizer'
        elif config.model_type == 'internlm2':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_internlm2'
            tokenizer_class_name = 'InternLM2Tokenizer'
        elif config.model_type == 'whisper_decoder':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_whisper'
            tokenizer_class_name = 'WhisperTokenizer'
        elif config.model_type == 'phi4':  # FIXME: To be confirmed
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gpt2'
            tokenizer_class_name = 'GPT2Tokenizer'

        if config.model_type in ['gecko', 'gecko2']:
            tokenizer_class = getattr(importlib.import_module(import_path), tokenizer_class_name)
        else:
            tokenizer_class = getattr(importlib.import_module(import_path), tokenizer_class_name)
    else:
        if config.tokenizer == 'baichuan':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_baichuan'
            tokenizer_class_name = 'BaichuanTokenizer'
        elif config.tokenizer == 'gpt2':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gpt2'
            tokenizer_class_name = 'GPT2Tokenizer'
        elif config.tokenizer == 'gpt2_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gpt2_fast'
            tokenizer_class_name = 'GPT2TokenizerFast'
        elif config.tokenizer == 'qwen':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_qwen'
            tokenizer_class_name = 'QWenTokenizer'
        elif config.tokenizer == 'qwen2':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_qwen2'
            tokenizer_class_name = 'Qwen2Tokenizer'
        elif config.tokenizer == 'qwen2_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_qwen2_fast'
            tokenizer_class_name = 'Qwen2TokenizerFast'
        elif config.tokenizer == 'gemma':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gemma'
            tokenizer_class_name = 'GemmaTokenizer'
        elif config.tokenizer == 'gemma_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_gemma_fast'
            tokenizer_class_name = 'GemmaTokenizerFast'
        elif config.tokenizer == 'llama':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_llama'
            tokenizer_class_name = 'LlamaTokenizer'
        elif config.tokenizer == 'pretrained_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_utils_fast'
            tokenizer_class_name = 'PreTrainedTokenizerFast'
        elif config.tokenizer == 'llama_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_llama_fast'
            tokenizer_class_name = 'LlamaTokenizerFast'
        elif config.tokenizer == 'sentencepiece':
            import_path = 'sentencepiece'
            tokenizer_class_name = 'SentencePieceProcessor'
        elif config.tokenizer == 'hunyuan':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_hunyuan'
            tokenizer_class_name = 'HunYuanTokenizer'
        elif config.tokenizer == 'internlm2':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_internlm2'
            tokenizer_class_name = 'InternLM2Tokenizer'
        elif config.tokenizer == 'internlm2_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_internlm2_fast'
            tokenizer_class_name = 'InternLM2TokenizerFast'
        elif config.tokenizer == 'milm':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_milm'
            tokenizer_class_name = 'MiTokenizer'
        elif config.tokenizer == 'whisper_decoder':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_whisper'
            tokenizer_class_name = 'WhisperTokenizer'
        elif config.tokenizer == 'whisper_fast':
            import_path = 'mtk_llm_sdk.tokenizers.tokenization_whisper_fast'
            tokenizer_class_name = 'WhisperFastTokenizer'
        if config.tokenizer == 'sentencepiece':
            tokenizer_class = getattr(importlib.import_module(import_path), tokenizer_class_name)
        else:
            tokenizer_class = getattr(importlib.import_module(import_path), tokenizer_class_name)

    if tokenizer_class is SentencePieceProcessor:
        tokenizer = tokenizer_class(model_file=os.path.join(llm_weight_dir, 'tokenizer.model'), add_bos=add_bos)
    elif tokenizer_class is MiTokenizer:
        tokenizer = tokenizer_class(vocab_file=os.path.join(llm_weight_dir, 'tokenizer.model'), add_bos=add_bos)
    else:
        tokenizer = tokenizer_class.from_pretrained(llm_weight_dir)
        tokenizer.add_bos_token = add_bos
    return tokenizer


def get_torch_dtype(dtype):
    """Converts non torch dtype to torch dtype if needed and returns it.

    Args:
        dtype(numpy dtype or str or torch dtype): The datatype to convert to torch dtype if needed.

    Returns:
        torch.dtype: The torch dtype.
    """
    if dtype in [np.float16, 'float16']:
        return torch.float16
    if dtype in [np.float32, 'float']:
        return torch.float32
    return dtype


def jit_trace(layer, example_inputs, output_folder):
    """Traces the given layer using JIT and saves the traced model to the output folder.

    Args:
        layer (torch.nn.Module): The layer to trace.
        example_inputs (tuple): The example inputs for tracing.
        output_folder (str): The folder to save the traced model.

    Returns:
        str: The path to the saved traced model.
    """
    trace = torch.jit.trace(layer, example_inputs=example_inputs)
    temp_trace_filename = os.path.join(output_folder, 'tmp_ptq_chunked.pt')
    temp_idx = 0
    while True:
        if os.path.exists(temp_trace_filename):
            temp_idx += 1
            temp_trace_filename = os.path.join(output_folder, f'tmp_ptq_chunked_{temp_idx}.pt')
        else:
            break
    trace.save(temp_trace_filename)

    return temp_trace_filename


def make_quantized_model_info_dumpable(quantized_model_info):
    """Converts the numpy dtypes in the quantized model info dictionary to string format for serialization.

    Args:
        quantized_model_info (dict): The quantized model info dictionary.

    Returns:
        dict: The modified quantized model info dictionary with numpy dtypes replaced with strings.

    Raises:
        TypeError: If `quantized_model_info` is not a dictionary.
        RuntimeError: If an unexpected numpy dtype is encountered.
    """

    def _numpy_dtype_to_str(dtype):
        if dtype == np.float32:
            return 'float32'
        if dtype == np.int32:
            return 'int32'
        if dtype == np.float16:
            return 'float16'
        if dtype == np.int16:
            return 'int16'
        if dtype == np.int8:
            return 'int8'
        logger.error('Unexpected numpy dtype:', dtype)
        raise

    if not isinstance(quantized_model_info, dict):
        logger.error(f'Expected quantized_model_info to be dict, but got {type(quantized_model_info)}', err=TypeError)

    new_dict = {k: v for k, v in quantized_model_info.items() if 'dtypes' not in k}
    new_dict['input_dtypes'] = [_numpy_dtype_to_str(x) for x in quantized_model_info['input_dtypes']]
    new_dict['output_dtypes'] = [_numpy_dtype_to_str(x) for x in quantized_model_info['output_dtypes']]

    return new_dict


def pad_lm_head_to_any_n(lm_head, llm_config, base_num):
    """Pads the language model head to a multiple of base_num.

    Args:
        lm_head (torch.nn.Linear): The language model head.
        llm_config (Object): The llm model config.
        base_num (int): The base number to multiply.

    Returns:
        torch.nn.Linear: The padded language model head.
        pad_size: The pad size.
    """
    vocab_size = llm_config.vocab_size
    pad_vocab_size = vocab_size + (base_num - (vocab_size % base_num)) % base_num
    pad_size = pad_vocab_size - vocab_size

    if llm_config.lm_head_pad_size == pad_size:
        # if the padding is calculated before, the tail will be padded already
        if lm_head.out_features != pad_vocab_size:
            logger.error(
                'Tail is not initialized properly with size = vocab_size + pad_size. '
                'Please consult llm_sdk maintainers.'
            )
        return lm_head, pad_size

    has_bias = lm_head.bias is not None
    padded_lm_head = torch.nn.Linear(
        lm_head.in_features, pad_vocab_size, bias=has_bias, dtype=lm_head.weight.dtype, device=lm_head.weight.device
    )

    # need to slice lm_head weight and bias because the lm_head vocab size may be different due to padding
    padded_lm_head.weight.data[:vocab_size] = lm_head.weight.data[:vocab_size]
    if has_bias:
        padded_lm_head.bias.data[:vocab_size] = lm_head.bias.data[:vocab_size]
    return padded_lm_head, pad_size


def update_config_file_lm_head_pad(config_path, pad_size):
    """Write lm head pad size to config file."""
    with open(config_path) as f:
        config = json.load(f)

    if 'llm' in config:
        # v3.x config
        config['llm']['lm_head_pad_size'] = pad_size
    else:
        # legacy config
        config['lm_head_pad_size'] = pad_size

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def recursive_remove_if_exist(folder, recreate=False):
    """Recursively removes a folder if it exists and optionally recreates it.

    Args:
        folder (str): The folder to remove.
        recreate (bool, optional): Whether to recreate the folder. Default is False.
    """
    if os.path.exists(folder):
        logger.warning(f'Deleting existing folder {folder}')
        shutil.rmtree(folder)
    if recreate:
        os.makedirs(folder)


def resolve_preprocessor_class(config):
    """Resolves the preprocessor class based on the configuration.

    Args:
        config (object): The configuration object.

    Returns:
        class: The preprocessor class.
    """
    logger.debug('Enter resolve_preprocessor_class')
    if config is None:
        logger.debug('No preprocessor')
        return None

    if config.model_type == 'clip':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_clip'
        preprocessor_class_name = 'CLIPImageProcessor'
    elif config.model_type == 'siglip':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_siglip'
        preprocessor_class_name = 'SigLIPImageProcessor'
    elif config.model_type == 'qwen2_vl_vision':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_qwen2vl_vision'
        preprocessor_class_name = 'Qwen2VLImageProcessor'
    elif config.model_type == 'phi3_vision_emb':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_phi3v_vision_processor'
        preprocessor_class_name = 'Phi3VImageProcessor'
    elif config.model_type == 'minicpmv_navit_siglip':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_minicpmv_navit_siglip'
        preprocessor_class_name = 'MinicpmVNavitSigLIPImageProcessor'
    elif config.model_type == 'intern_vit_6b':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_internvit'
        preprocessor_class_name = 'InternViTPreprocessor'
    elif config.model_type == 'qwen2_vl':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_qwen2vl_vision'
        preprocessor_class_name = 'Qwen2VLImageProcessor'
    elif config.model_type == 'whisper':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_whisper'
        preprocessor_class_name = 'WhisperAudioProcessor'
    elif config.model_type == 'phi3v':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_phi3v_vision_processor'
        preprocessor_class_name = 'Phi3VImageProcessor'
    elif config.model_type == 'gecko2_vision':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_x2_vision'
        preprocessor_class_name = 'Gecko2VisionImageProcessor'
    elif config.model_type == 'phi4o_img':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_phi4o'
        preprocessor_class_name = 'Phi4OImageProcessor'
    elif config.model_type == 'intern_vit_6b_navit_rope':
        import_path = 'mtk_llm_sdk.models.preprocessors.preprocessor_internvl_navit_rope'
        preprocessor_class_name = 'InternVLNavitRopePreprocessor'
    else:
        logger.error(f'Unknown preprocessor `model_type`: {config.model_type}')

    logger.debug(f'Preprocessor import_path={import_path}, preprocessor_class_name={preprocessor_class_name}')
    return getattr(importlib.import_module(import_path), preprocessor_class_name)


def resolve_llm_class(config):
    """Resolves the LLM (Language Model) class based on the configuration.

    Args:
        config (object): The configuration object.

    Returns:
        class: The LLM class.
    """
    logger.debug('Enter resolve_llm_class')
    if config.model_type == 'llama':
        import_path = 'mtk_llm_sdk.models.llm.modeling_llama'
        chunk_class_name = 'LlamaModelChunk'
    elif config.model_type == 'baichuan':
        import_path = 'mtk_llm_sdk.models.llm.modeling_baichuan'
        chunk_class_name = 'BaichuanModelChunk'
    elif config.model_type == 'qwen':
        import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
        chunk_class_name = 'QwenModelChunk'
    elif config.model_type in ['qwen1.5', 'qwen2']:
        import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
        chunk_class_name = 'Qwen2ModelChunk'
    elif config.model_type == 'qwen3':
        print('Using Qwen3ModelChunk')
        import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
        chunk_class_name = 'Qwen3ModelChunk'
    elif config.model_type == 'milm':
        import_path = 'mtk_llm_sdk.models.llm.modeling_milm'
        chunk_class_name = 'MiLMModelChunk'
    elif config.model_type == 'gecko':
        import_path = 'mtk_llm_sdk.models.llm.modeling_x'
        chunk_class_name = 'GeckoModelChunk'
    elif config.model_type == 'gecko2':
        import_path = 'mtk_llm_sdk.models.llm.modeling_x'
        chunk_class_name = 'Gecko2ModelChunk'
    elif config.model_type == 'phi3':
        import_path = 'mtk_llm_sdk.models.llm.modeling_phi3'
        chunk_class_name = 'Phi3ModelChunk'
    elif config.model_type == 'gemma':
        import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
        chunk_class_name = 'GemmaModelChunk'
    elif config.model_type == 'gemma2':
        import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
        chunk_class_name = 'Gemma2ModelChunk'
    elif config.model_type == 'gemma3':
        import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
        chunk_class_name = 'Gemma3ModelChunk'
    elif config.model_type == 'minicpm':
        import_path = 'mtk_llm_sdk.models.llm.modeling_minicpm'
        chunk_class_name = 'MinicpmModelChunk'
    elif config.model_type == 'hunyuan':
        import_path = 'mtk_llm_sdk.models.llm.modeling_hunyuan'
        chunk_class_name = 'HunYuanModelChunk'
    elif config.model_type == 'internlm2':
        import_path = 'mtk_llm_sdk.models.llm.modeling_internlm2'
        chunk_class_name = 'InternLM2ModelChunk'
    elif config.model_type == 'whisper_decoder':
        import_path = 'mtk_llm_sdk.models.llm.modeling_whisper'
        chunk_class_name = 'WhisperDecoderModelChunk'
    elif config.model_type in 'phi4':
        import_path = 'mtk_llm_sdk.models.llm.modeling_phi3'
        chunk_class_name = 'Phi4ModelChunk'
    else:
        logger.error(f'Unknown LLM `model_type`: {config.model_type}')

    logger.debug(f'LLM import_path={import_path}, chunk_class_name={chunk_class_name}')
    return getattr(importlib.import_module(import_path), chunk_class_name)


def resolve_encoder_class(config):
    """Resolves the encoder class based on the configuration.

    Args:
        config (object): The configuration object.

    Returns:
        class: The encoder class.
    """
    logger.debug('Enter resolve_encoder_class')
    if config is None:
        logger.debug('No encoder')
        return None

    if config.model_type == 'clip':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_clip'
        encoder_class_name = 'CLIPVisionEncoderChunk'
    elif config.model_type == 'siglip':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_siglip'
        encoder_class_name = 'SigLIPVisionTransformer'
    elif config.model_type == 'intern_vit_6b':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_intern_vit'
        encoder_class_name = 'InternVisionModel'
    elif config.model_type == 'minicpmv_navit_siglip':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_minicpmv_navit_siglip'
        encoder_class_name = 'MinicpmVNavitSigLIPVisionTransformer'
    elif config.model_type == 'qwen2_vl_vision':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_qwen2vl_vision'
        encoder_class_name = 'Qwen2VLVisionModel'
    elif config.model_type == 'phi3_vision_emb':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_phi3v_img_embedding'
        encoder_class_name = 'Phi3VImageEmbeddingChunk'
    elif config.model_type == 'whisper':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_whisper'
        encoder_class_name = 'WhisperEncoderChunk'
    elif config.model_type == 'gecko2_vision':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_x2_vision'
        encoder_class_name = 'Gecko2VisionEncoderChunk'
    elif config.model_type == 'phi4o_navit_siglip':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_phi4o_img_encoder'
        encoder_class_name = 'Phi4OImgEncoder'
    elif config.model_type == 'intern_vit_6b_navit_rope':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_internvl_navit_rope'
        encoder_class_name = 'InternVLNavitRopeModel'
    elif config.model_type == 'qwen2_5_vl_vision':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_qwen2_5vl_vision'
        encoder_class_name = 'Qwen2_5VLVisionModel'
    elif config.model_type == 'qwen2_audio_encoder':
        import_path = 'mtk_llm_sdk.models.encoders.modeling_qwen2_audio'
        encoder_class_name = 'Qwen2AudioEncoderChunk'
    else:
        logger.error(f'Unknown encoder `model_type`: {config.model_type}')

    logger.debug(f'Encoder import_path={import_path}, encoder_class_name={encoder_class_name}')
    return getattr(importlib.import_module(import_path), encoder_class_name)


def resolve_projector_class(config):
    """Resolves the projector class based on the configuration.

    Args:
        config (object): The configuration object.

    Returns:
        class: The projector class.
    """
    logger.debug('Enter resolve_projector_class')
    if config is None:
        logger.debug('No projector')
        return None

    if config.model_type == 'mlp_gelu':
        import_path = 'mtk_llm_sdk.models.projectors.projector_mlp_gelu'
        projector_class_name = 'MLPGeluProjector'
    elif config.model_type == 'mlp_downsample':
        import_path = 'mtk_llm_sdk.models.projectors.projector_mlp_downsample'
        projector_class_name = 'MLPDownsampleProjector'
    elif config.model_type == 'ldpnetv2':
        import_path = 'mtk_llm_sdk.models.projectors.projector_ldpnetv2'
        projector_class_name = 'LDPNetV2Projector'
    elif config.model_type == 'internvl2':
        import_path = 'mtk_llm_sdk.models.projectors.projector_internvl2'
        projector_class_name = 'InternVL2Projector'
    elif config.model_type in ['qwen2_vl', 'patch_merger']:
        import_path = 'mtk_llm_sdk.models.projectors.projector_patchmerger'
        projector_class_name = 'PatchMergerProjector'
    elif config.model_type in ['phi3v']:
        import_path = 'mtk_llm_sdk.models.projectors.projector_phi3v'
        projector_class_name = 'Phi3VProjector'
    elif config.model_type in ['gecko2_vision']:
        import_path = 'mtk_llm_sdk.models.projectors.projector_x2_vision'
        projector_class_name = 'Gecko2VisionProjector'
    elif config.model_type == 'phi4o':
        import_path = 'mtk_llm_sdk.models.projectors.projector_phi3v'
        projector_class_name = 'Phi4OProjector'
    elif config.model_type == 'andesvl':
        import_path = 'mtk_llm_sdk.models.projectors.projector_andesvl'
        projector_class_name = 'InternVLNavitRopeProjector'
    elif config.model_type == 'qwen2_5_vl':
        import_path = 'mtk_llm_sdk.models.projectors.projector_qwen2_5_vl_patchmerger'
        projector_class_name = 'Qwen2_5VLPatchMergerProjector'
    elif config.model_type == 'qwen2_audio_encoder':
        import_path = 'mtk_llm_sdk.models.projectors.projector_qwen2_audio'
        projector_class_name = 'Qwen2AudioProjector'
    elif config.model_type == 'minicpmv':
        import_path = 'mtk_llm_sdk.models.projectors.projector_minicpmv'
        projector_class_name = 'MinicpmVProjector'
    else:
        logger.error(f'Unknown projector `model_type`: {config.model_type}')

    logger.debug(f'Projector import_path={import_path}, projector_class_name={projector_class_name}')
    return getattr(importlib.import_module(import_path), projector_class_name)


def resolve_tail_class(tail_config, llm_config):
    """Resolves the tail class based on the provided tail and LLM configurations.

    Args:
        tail_config (object): The configuration for the tail.
        llm_config (object): The configuration for the LLM.

    Returns:
        tuple: A tuple containing the tail configuration, tail class, and decoder class.

    Raises:
        RuntimeError: If the tail_config model type is not supported.
    """
    logger.debug('Enter resolve_tail_class')
    decoder_class = None
    if tail_config is None:
        logger.debug('Tail type is normal tail.')
        if llm_config.model_type == 'llama':
            import_path = 'mtk_llm_sdk.models.llm.modeling_llama'
            tail_class_name = 'LlamaTail'
        elif llm_config.model_type == 'baichuan':
            import_path = 'mtk_llm_sdk.models.llm.modeling_baichuan'
            tail_class_name = 'BaichuanTail'
        elif llm_config.model_type == 'qwen':
            import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
            tail_class_name = 'QwenTail'
        elif llm_config.model_type in ['qwen1.5', 'qwen2', 'qwen3']:
            import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
            tail_class_name = 'Qwen2Tail'
        elif llm_config.model_type == 'milm':
            import_path = 'mtk_llm_sdk.models.llm.modeling_milm'
            tail_class_name = 'MiLMTail'
        elif llm_config.model_type in ['gecko', 'gecko2']:
            import_path = 'mtk_llm_sdk.models.llm.modeling_x'
            tail_class_name = 'GeckoTail'
        elif llm_config.model_type in ['phi3', 'phi4']:
            import_path = 'mtk_llm_sdk.models.llm.modeling_phi3'
            tail_class_name = 'Phi3Tail'
        elif llm_config.model_type == 'gemma':
            import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
            tail_class_name = 'GemmaTail'
        elif llm_config.model_type == 'gemma2':
            import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
            tail_class_name = 'Gemma2Tail'
        elif llm_config.model_type == 'gemma3':
            import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
            tail_class_name = 'Gemma3Tail'
        elif llm_config.model_type == 'minicpm':
            import_path = 'mtk_llm_sdk.models.llm.modeling_minicpm'
            tail_class_name = 'MinicpmTail'
        elif llm_config.model_type == 'hunyuan':
            import_path = 'mtk_llm_sdk.models.llm.modeling_hunyuan'
            tail_class_name = 'HunYuanTail'
        elif llm_config.model_type == 'internlm2':
            import_path = 'mtk_llm_sdk.models.llm.modeling_internlm2'
            tail_class_name = 'InternLM2Tail'
        elif llm_config.model_type == 'whisper_decoder':
            import_path = 'mtk_llm_sdk.models.llm.modeling_whisper'
            tail_class_name = 'WhisperDecoderTail'
        else:
            logger.error(f'Unknown LLM `model_type`: {llm_config.model_type}')
    else:
        from .const import SUPPORTED_LLMS

        if tail_config.model_type == 'medusa':
            logger.debug('Tail type is Medusa tail.')
            import_path = 'mtk_llm_sdk.models.llm.modeling_medusa'
            tail_class_name = 'MedusaTail'
        else:
            if tail_config.model_type not in SUPPORTED_LLMS:
                logger.error(
                    f'Unsupported EAGLE model_type: {tail_config.model_type}. List of '
                    f'supported model_types: {SUPPORTED_LLMS}'
                )
            logger.debug('Tail type is EAGLE tail.')
            import_path = 'mtk_llm_sdk.models.llm.modeling_eagle'
            tail_class_name = 'EagleTail'
            if tail_config.model_type == 'llama':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_llama'
                decoder_class_name = 'LlamaDecoderLayer'
            elif tail_config.model_type == 'baichuan':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_baichuan'
                decoder_class_name = 'BaichuanDecoderLayer'
            elif tail_config.model_type in ['qwen', 'qwen1.5', 'qwen2', 'qwen3']:
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_qwen'
                decoder_class_name = 'QwenDecoderLayer'
            elif tail_config.model_type == 'milm':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_milm'
                decoder_class_name = 'MiLMDecoderLayer'
            elif tail_config.model_type == 'gecko':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_x'
                decoder_class_name = 'GeckoDecoderLayer'
            elif tail_config.model_type == 'gecko2':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_x'
                decoder_class_name = 'Gecko2DecoderLayer'
            elif tail_config.model_type == 'phi3':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_phi3'
                decoder_class_name = 'Phi3DecoderLayer'
            elif tail_config.model_type == 'gemma':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
                decoder_class_name = 'GemmaDecoderLayer'
            elif tail_config.model_type == 'gemma2':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
                decoder_class_name = 'Gemma2DecoderLayer'
            elif tail_config.model_type == 'gemma3':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_gemma'
                decoder_class_name = 'Gemma3DecoderLayer'
            elif tail_config.model_type == 'minicpm':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_minicpm'
                decoder_class_name = 'MinicpmDecoderLayer'
            elif tail_config.model_type == 'hunyuan':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_hunyuan'
                decoder_class_name = 'HunYuanDecoderLayer'
            elif tail_config.model_type == 'internlm2':
                decoder_import_path = 'mtk_llm_sdk.models.llm.modeling_internlm2'
                decoder_class_name = 'InternLM2DecoderLayer'
            else:
                logger.error(f'Unknown EAGLE tail `model_type`: {tail_config.model_type}')

            decoder_class = getattr(importlib.import_module(decoder_import_path), decoder_class_name)

        if tail_config.model_type != 'medusa':
            tail_config.model_type = 'eagle'

    logger.debug(f'Tail import_path={import_path}, tail_class_name={tail_class_name}')
    tail_class = getattr(importlib.import_module(import_path), tail_class_name)

    return tail_config, tail_class, decoder_class


def resolve_hook_class(hook_name, special=None):
    """Resolves the hook class based on the hook name.

    Args:
        hook_name (str): The name of the hook.
        special (str): Imply whether the hook is special.
            Only `format_text`, `tokenizer_func`, `get_embeds` can be special. Defaults to None.

    Returns:
        class: The hook class.

    Raises:
        NotImplementedError: If the hook name is not supported.
    """
    logger.debug(f'Enter resolve_hook_class, special={special}')
    import_path = None
    hook_class_name = None
    if special == 'format_text':
        if hook_name == 'passthrough':
            import_path = 'mtk_llm_sdk.models.hooks.passthrough'
            hook_class_name = 'Passthrough'
        elif hook_name == 'llava':
            import_path = 'mtk_llm_sdk.models.format_text.llava'
            hook_class_name = 'LlavaFormatText'
        elif hook_name == 'internvl2':
            import_path = 'mtk_llm_sdk.models.format_text.internvl2'
            hook_class_name = 'InternVL2FormatText'
        elif hook_name == 'qwen2_vl':
            import_path = 'mtk_llm_sdk.models.format_text.qwen2_vl'
            hook_class_name = 'Qwen2VLFormatText'
        elif hook_name == 'phi3v':
            import_path = 'mtk_llm_sdk.models.format_text.phi3v'
            hook_class_name = 'Phi3VFormatText'
        elif hook_name == 'phi4o':
            import_path = 'mtk_llm_sdk.models.format_text.phi4o'
            hook_class_name = 'Phi4OFormatText'
        elif hook_name == 'andesvl':
            import_path = 'mtk_llm_sdk.models.format_text.andesvl'
            hook_class_name = 'AndesVLFormatText'
        elif hook_name == 'minicpmv':
            import_path = 'mtk_llm_sdk.models.format_text.minicpmv'
            hook_class_name = 'MinicpmVFormatText'
    elif special == 'tokenizer_func':
        if hook_name == 'default':
            import_path = 'mtk_llm_sdk.models.tokenizer_func.default'
            hook_class_name = 'DefaultTokenizerFunc'
        elif hook_name == 'phi3v':
            import_path = 'mtk_llm_sdk.models.tokenizer_func.phi3v'
            hook_class_name = 'Phi3VTokenizerFunc'
        elif hook_name == 'phi4o':
            import_path = 'mtk_llm_sdk.models.tokenizer_func.phi4o'
            hook_class_name = 'Phi4OTokenizerFunc'
    elif special == 'get_embeds':
        if hook_name == 'text_only':
            import_path = 'mtk_llm_sdk.models.get_embeds.text_only'
            hook_class_name = 'TextOnlyGetEmbeds'
        elif hook_name == 'gecko2':
            import_path = 'mtk_llm_sdk.models.get_embeds.gecko2'
            hook_class_name = 'Gecko2GetEmbeds'
        elif hook_name == 'llava':
            import_path = 'mtk_llm_sdk.models.get_embeds.llava'
            hook_class_name = 'LlavaGetEmbeds'
        elif hook_name == 'internvl2':
            import_path = 'mtk_llm_sdk.models.get_embeds.internvl2'
            hook_class_name = 'InternVL2GetEmbeds'
        elif hook_name == 'qwen2_vl':
            import_path = 'mtk_llm_sdk.models.get_embeds.qwen2_vl'
            hook_class_name = 'Qwen2VLGetEmbeds'
        elif hook_name == 'phi3v':
            import_path = 'mtk_llm_sdk.models.get_embeds.phi3v'
            hook_class_name = 'Phi3VGetEmbeds'
        elif hook_name == 'andesvl':
            import_path = 'mtk_llm_sdk.models.get_embeds.andesvl'
            hook_class_name = 'AndesVLGetEmbeds'
        elif hook_name == 'qwen2_5_vl':
            import_path = 'mtk_llm_sdk.models.get_embeds.qwen2_5_vl'
            hook_class_name = 'Qwen2_5VLGetEmbeds'
        elif hook_name == 'text_with_mm':
            import_path = 'mtk_llm_sdk.models.get_embeds.text_with_mm'
            hook_class_name = 'TextWithMMGetEmbeds'
        elif hook_name == 'minicpmv':
            import_path = 'mtk_llm_sdk.models.get_embeds.minicpmv'
            hook_class_name = 'MinicpmVGetEmbeds'
    elif special == 'generation_modifiers':
        if hook_name == 'passthrough':
            import_path = 'mtk_llm_sdk.models.generation_modifiers.passthrough'
            hook_class_name = 'Passthrough'
        elif hook_name == 'whisper':
            import_path = 'mtk_llm_sdk.models.generation_modifiers.whisper'
            hook_class_name = 'WhisperLogitsProcessor'
    else:
        if special is not None:
            logger.error(
                f'Unknown `special` option: {special}. '
                'Supported: format_text, tokenizer_func, get_embeds, generation_modifiers.'
            )
        if hook_name == 'passthrough':
            import_path = 'mtk_llm_sdk.models.hooks.passthrough'
            hook_class_name = 'Passthrough'
        elif hook_name == 'patch_select':  # LLaVA1.5
            import_path = 'mtk_llm_sdk.models.hooks.patch_select'
            hook_class_name = 'PatchSelect'
        elif hook_name == 'numpy_to_torch':  # LLaVA1.5
            import_path = 'mtk_llm_sdk.models.hooks.numpy_to_torch'
            hook_class_name = 'NumpyToTorch'
        elif hook_name == 'torch_to_numpy':  # LLaVA1.5
            import_path = 'mtk_llm_sdk.models.hooks.torch_to_numpy'
            hook_class_name = 'TorchToNumpy'
        elif hook_name == 'pad_internvl2':  # InternVL2
            import_path = 'mtk_llm_sdk.models.hooks.pad_internvl2'
            hook_class_name = 'PadInternVL2'
        elif hook_name == 'internvl2_pixel_shuffle':  # InternVL2
            import_path = 'mtk_llm_sdk.models.hooks.pixel_shuffle'
            hook_class_name = 'InternVL2PixelShuffle'
        elif hook_name == 'qwen2vl_preencoder':  # Qwen2-VL
            import_path = 'mtk_llm_sdk.models.hooks.qwen2_vl_pre_encoder'
            hook_class_name = 'Qwen2VLPreEncoder'
        elif hook_name == 'qwen2vl_prellm':  # Qwen2-VL
            import_path = 'mtk_llm_sdk.models.hooks.qwen2_vl_pre_llm'
            hook_class_name = 'Qwen2VLPreLLM'
        elif hook_name == 'phi3v_preprojector':  # Phi3-V
            import_path = 'mtk_llm_sdk.models.hooks.phi3v_preprojector'
            hook_class_name = 'Phi3vPreProjector'
        elif hook_name == 'pad_phi3v':  # Phi3-V
            import_path = 'mtk_llm_sdk.models.hooks.pad_phi3v'
            hook_class_name = 'PadPhi3V'
        elif hook_name == 'phi4o_img_preprojector':  # Phi4-Omni
            import_path = 'mtk_llm_sdk.models.hooks.phi4o_img_pre_projector'
            hook_class_name = 'Phi4OImgPreprojector'
        elif hook_name == 'andesvl_preprojector':  # AndesVL
            import_path = 'mtk_llm_sdk.models.hooks.andesvl_preprojector'
            hook_class_name = 'AndesVLPreprojector'
        elif hook_name == 'andesvl_preencoder':  # AndesVL
            import_path = 'mtk_llm_sdk.models.hooks.andesvl_preencoder'
            hook_class_name = 'AndesVLPreEncoderHook'
        elif hook_name == 'qwen2_5vl_preencoder':  # Qwen2.5-VL
            import_path = 'mtk_llm_sdk.models.hooks.qwen2_5_vl_pre_encoder'
            hook_class_name = 'Qwen2_5VLPreEncoder'
        elif hook_name == 'bita_pre_getembed':  # BiTA
            import_path = 'mtk_llm_sdk.models.hooks.bita_pre_getembed'
            hook_class_name = 'BiTAPreGetEmbed'

    if import_path is None or hook_class_name is None:
        logger.error(f'Unsupported hook: {hook_name}', err=NotImplementedError)

    logger.debug(f'Hook import_path={import_path}, hook_class_name={hook_class_name}')
    return getattr(importlib.import_module(import_path), hook_class_name)


def resolve_evictor_class(evict_method):
    """Resolves the Evictor class based on the evict_method.

    Args:
        evict_method (str): The name of evictor.

    Returns:
        class: The Evictor class.
    """
    logger.debug('Enter resolve_evictor_class')
    if evict_method == 'GlobalSnapKV':
        import_path = 'mtk_llm_sdk.utils.longcontext.evictor_globalsnapkv'
        chunk_class_name = 'GlobalSnapKV'
    elif evict_method == 'LocalSnapKV':
        import_path = 'mtk_llm_sdk.utils.longcontext.evictor_localsnapkv'
        chunk_class_name = 'LocalSnapKV'
    else:
        logger.debug('No Evictor Specified')
        return None

    logger.debug(f'EvictorClass={evict_method}')
    return getattr(importlib.import_module(import_path), chunk_class_name)


@contextmanager
def temp_file(f):
    """Context manager for temporary file or directory handling.

    Args:
        f (str): The file or directory path.

    Raises:
        TypeError: If `f` is neither a file nor a directory.

    Yields:
        None

    Ensures:
        The file or directory is removed after the context is exited, even if an exception occurs.
    """
    if not os.path.isfile(f) and not os.path.isdir(f):
        logger.error(f'Expected {f} to be file or folder, but is neither.', err=TypeError)
    try:
        yield
    except Exception:
        if os.path.exists(f):
            if os.path.isdir(f):
                logger.debug(f'Remove directory: {f}')
                shutil.rmtree(f)
            else:
                logger.debug(f'Remove file: {f}')
                os.remove(f)
        logger.error(traceback.format_exc())
    finally:
        if os.path.exists(f):
            if os.path.isdir(f):
                logger.debug(f'Remove directory: {f}')
                shutil.rmtree(f)
            else:
                logger.debug(f'Remove file: {f}')
                os.remove(f)


def tokenized_text_to_array(text):
    """Converts tokenized text to a numpy array.

    Args:
        text (str): The tokenized text.

    Returns:
        np.ndarray: The numpy array representation of the tokenized text.
    """
    logger.debug('Enter tokenized_text_to_array')
    if isinstance(text, np.ndarray):
        logger.debug('Input is already array, do nothing.')
        return text
    if ',' in text:
        arr = np.array([[int(x) for x in text.strip().split(',')]], dtype=np.int32)
    else:
        arr = np.array([[int(x) for x in text.split(' ')]], dtype=np.int32)

    logger.debug(f'{text} -> {arr}')
    return arr


class PrintFilepathAndExit(argparse.Action):
    """Argparse action to print the absolute file path and exit.

    Args:
        option_strings (list): The option strings.
        dest (str): The destination.
        **kwargs: Additional keyword arguments.

    Attributes:
        file (str): The file path to print.
    """

    def __init__(self, option_strings, dest, **kwargs):
        """Initializes the PrintFilepathAndExit class.

        Args:
            option_strings (list): The option strings.
            dest (str): The destination.
            **kwargs: Additional keyword arguments.
        """
        self.file = kwargs.pop('file')
        return super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        """Prints the absolute file path and exits.

        Args:
            parser (argparse.ArgumentParser): The argument parser.
            namespace (argparse.Namespace): The namespace.
            values (str): The values.
            option_string (str): The option string.
            **kwargs: Additional keyword arguments.
        """
        print(os.path.abspath(self.file))
        exit()
