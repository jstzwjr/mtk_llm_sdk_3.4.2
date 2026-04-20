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
"""Define QAT- or QALFT-related helper functions for mtk_llm_sdk."""

import copy
import math
import re

import numpy as np
import torch
from mtk_converter.python.proto import nrpmodel_pb2 as nrp_pb2
from mtk_converter.python.utils import (
    defs,
    linear_quant_param_utils,
    operator_utils,
    subgraph_utils,
    tensor_utils,
    type_utils,
)
from torch import nn

from . import logger, quantized_model_utils

FQ_NAMES = ['weight', 'bias', 'linear', 'add', 'sub', 'mul', 'div', 'cat', 'matmul']


def extract_quant_params(quantized_model_path, quantized_model_info, config, chunk_idx):
    """Extract Weight and Activation Scale and Min Max."""
    subgraph = quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
    model_type = config.l.model_type
    num_decoder_layers = config.l.num_hidden_layers
    precision_config = quantized_model_info['precision_config']
    is_fp_respath = (
        precision_config['respath_precision']
        == precision_config['embeds_precision']
        == precision_config['logits_precision']
        == 'FP'
    )
    logger.debug(f'[Extract Quant Params] FP Respath: {is_fp_respath}')

    chunk_quant_param_details = {}
    tensor_names = subgraph.tensor_map.keys()
    input_names = subgraph.inputs
    extracted = set()

    for tensor_name in tensor_names:
        tensor = subgraph.tensor_map[tensor_name]
        if any(key in tensor_name for key in FQ_NAMES) and not any(
            key in tensor_name for key in ['requantize', 'Constant', 'gelu']
        ):
            original_tensor_name = tensor_name
            if 'weight' in tensor_name or 'bias' in tensor_name:
                tensor_name = tensor_name.replace('/', '.')
            else:
                if tensor_name not in input_names:
                    tmp = tensor_name.rsplit('/', 1)[0]
                    if re.fullmatch(r'(\w+/)(\d+/)self_attn', tmp) is not None or ('norm' in tmp and 'mul' not in tmp):
                        logger.debug(f'Skipping: {tensor_name}')
                        continue
                    # remove flattened and dequantized
                    if '_flattened' in tensor_name or '_dequantized' in tensor_name:
                        logger.debug(f'Skipping: {tensor_name}')
                        continue

                    tensor_name = tmp
                    tensor_name = tensor_name.replace('/', '.')
                    if (
                        model_type in ['gecko', 'gemma2']
                        and chunk_idx == num_decoder_layers
                        and tensor_name == 'tanh_div'
                    ):
                        tensor_name = 'lm_head'
            tensor_name = tensor_name.replace('.0.', f'.{chunk_idx}.')

            try:
                quant_param = tensor_utils.get_linear_quant_param(tensor)
            except Exception:
                if type_utils.is_quantizable_type(tensor.type):
                    logger.debug(
                        f'Skipping this tensor because it is quantizable but do not have qparam: {original_tensor_name}'
                    )
                else:
                    if 'weight' in tensor_name or 'bias' in tensor_name:
                        logger.error('Weight and bias are not supported in FP for QALFT.')
                    else:
                        if tensor_name in extracted:
                            logger.error(f'Encountered duplicate tensor: {tensor_name}, {original_tensor_name}')
                        chunk_quant_param_details[tensor_name] = {}
                        extracted.add(tensor_name)
                        logger.debug(f'Saving fp activations: {tensor_name}, {original_tensor_name}')
                continue
            # tensor_name = re.sub(r'(\w+)\.(\d+)(\.\w+)', r'\1[\2]\3', tensor_name)
            # replace x.0.y with x[0].y
            # layers/0/self_attn/mul2/mul -> Appears when u define any op using qat module
            # layers/0/self_attn/mul -> python division (don't store since it won't be recognised by qat tool)
            # .../v_proj/linear -> standard act quant after fc
            # div/div
            if tensor_name not in extracted:
                linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
                bitwidth = type_utils.get_quant_bitwidth(tensor.type)
                chunk_quant_param_details[tensor_name] = {}
                if 'weight' in tensor_name:
                    np_scale = torch.tensor(quant_param.scale_vals, dtype=torch.float32)
                    np_zp = torch.tensor(quant_param.zero_point_vals, dtype=torch.float32)
                else:
                    np_scale = quant_param.scale_vals[0]
                    np_zp = quant_param.zero_point_vals[0]
                np_min = np.array(quant_param.min_vals, dtype=np.float32)
                np_max = np.array(quant_param.max_vals, dtype=np.float32)
                chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
                chunk_quant_param_details[tensor_name]['scale'] = np_scale
                chunk_quant_param_details[tensor_name]['zp'] = np_zp
                chunk_quant_param_details[tensor_name]['min'] = np_min
                chunk_quant_param_details[tensor_name]['max'] = np_max
                extracted.add(tensor_name)
                if 'weight' in tensor_name or 'bias' in tensor_name:
                    np_fp32_weights = linear_quant_param_utils.dequantize_data(
                        quant_param, tensor_utils.get_numpy_data(tensor)
                    )
                    chunk_quant_param_details[tensor_name]['weight'] = np_fp32_weights

    lora_inputs_count = quantized_model_info.get('num_lora_inputs', 0)
    # Settle inputs separately
    for i in range(len(input_names) - lora_inputs_count):
        if i == 0:
            tensor_name = f'stubs.{i}' if chunk_idx == 0 else f'layers.{chunk_idx - 1}.add2'
        elif i == 1:
            if chunk_idx > 0:
                continue
            tensor_name = f'stubs.{i}'
        elif i == (len(input_names) - lora_inputs_count - 1) and quantized_model_info['model_config'].get(
            'use_split_mask', False
        ):
            if chunk_idx == 0:
                tensor_name = 'split_mask_stub'
            else:
                continue
        else:
            if 'gecko' in model_type and chunk_idx == 0:
                tensor_name = f'stubs.{i}'
            else:
                if (i == 2 or i == 3) and chunk_idx > 0:
                    continue
                tensor_name = f'stubs.{i + 2 * quantized_model_info["layer_ids"][0]}'
        if input_names[i] not in extracted:
            tensor = subgraph.tensor_map[input_names[i]]
            try:
                quant_param = tensor_utils.get_linear_quant_param(tensor)
            except Exception:
                logger.debug(f'Input {input_names[0]} is not quantizable')
                chunk_quant_param_details[tensor_name] = {}
                extracted.add(input_names[i])
                continue
            linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
            bitwidth = type_utils.get_quant_bitwidth(tensor.type)
            chunk_quant_param_details[tensor_name] = {}
            np_min = np.array(quant_param.min_vals, dtype=np.float32)
            np_max = np.array(quant_param.max_vals, dtype=np.float32)
            np_scale = quant_param.scale_vals[0]
            np_zp = quant_param.zero_point_vals[0]
            chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
            chunk_quant_param_details[tensor_name]['scale'] = np_scale
            chunk_quant_param_details[tensor_name]['zp'] = np_zp
            chunk_quant_param_details[tensor_name]['min'] = np_min
            chunk_quant_param_details[tensor_name]['max'] = np_max
            extracted.add(input_names[i])

    if chunk_idx < num_decoder_layers:
        if chunk_idx == 0 and config.l.use_stable_embedding:
            # Handle stable embedding layernorm
            embed_layernorm_name = 'to.3'
            if embed_layernorm_name in tensor_names and embed_layernorm_name not in extracted:
                tensor = subgraph.tensor_map[embed_layernorm_name]
                quant_param = tensor_utils.get_linear_quant_param(tensor)
                linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
                bitwidth = type_utils.get_quant_bitwidth(tensor.type)
                tensor_name = 'embed_layer_norm'
                chunk_quant_param_details[tensor_name] = {}
                np_min = np.array(quant_param.min_vals, dtype=np.float32)
                np_max = np.array(quant_param.max_vals, dtype=np.float32)
                np_scale = quant_param.scale_vals[0]
                np_zp = quant_param.zero_point_vals[0]
                chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
                chunk_quant_param_details[tensor_name]['scale'] = np_scale
                chunk_quant_param_details[tensor_name]['zp'] = np_zp
                chunk_quant_param_details[tensor_name]['min'] = np_min
                chunk_quant_param_details[tensor_name]['max'] = np_max
                extracted.add(embed_layernorm_name)

        layers_to_1_tensor_names = [
            name for name in tensor_names if re.fullmatch(r'layers/(\d+)/to.1', name) is not None
        ]
        if 'gecko' in model_type:
            # Note: layers.0.add2 (pytorch) maps to layers/1/to.1 (quantized_model)
            # Note: add (pytorch) maps to layers/0/to.1
            for quantized_model_name in layers_to_1_tensor_names:
                tensor = subgraph.tensor_map[quantized_model_name]
                quant_param = tensor_utils.get_linear_quant_param(tensor)
                linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
                bitwidth = type_utils.get_quant_bitwidth(tensor.type)
                layer_num = int(quantized_model_name.rsplit('/', 2)[1])
                # if > 1 chunk, and chunk_idx > 0,
                # every layers/x/to.1 is layers/x-1/add2 up to the num_layers-1 layer
                # if > 1 chunk and chunk_idx == 0, layers/0/to.1 is add
                # and layers/x/to.1 is layers/x-1/add2 up to the num_layers-1 layer
                if chunk_idx == 0 and layer_num == 0:
                    tensor_name = 'add'
                else:
                    tensor_name = (
                        quantized_model_name.replace(f'layers/{layer_num!s}/', f'layers/{layer_num - 1!s}/')
                        .replace('to.1', 'add2')
                        .replace('/', '.')
                    )
                chunk_quant_param_details[tensor_name] = {}
                np_min = np.array(quant_param.min_vals, dtype=np.float32)
                np_max = np.array(quant_param.max_vals, dtype=np.float32)
                chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
                chunk_quant_param_details[tensor_name]['min'] = np_min
                chunk_quant_param_details[tensor_name]['max'] = np_max
        else:
            # qconfig layers.x.input_norm.mul maps to quantized_model layers/x/to.1
            for quantized_model_name in layers_to_1_tensor_names:
                if quantized_model_name not in extracted:
                    tensor = subgraph.tensor_map[quantized_model_name]
                    quant_param = tensor_utils.get_linear_quant_param(tensor)
                    linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
                    bitwidth = type_utils.get_quant_bitwidth(tensor.type)
                    prefix = quantized_model_name.rsplit('/', 1)[0].replace('/', '.')
                    tensor_name = prefix + '.input_norm.mul'
                    tensor_name = tensor_name.replace('.0.', f'.{chunk_idx}.')
                    chunk_quant_param_details[tensor_name] = {}
                    np_min = np.array(quant_param.min_vals, dtype=np.float32)
                    np_max = np.array(quant_param.max_vals, dtype=np.float32)
                    np_scale = quant_param.scale_vals[0]
                    np_zp = quant_param.zero_point_vals[0]
                    chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
                    chunk_quant_param_details[tensor_name]['scale'] = np_scale
                    chunk_quant_param_details[tensor_name]['zp'] = np_zp
                    chunk_quant_param_details[tensor_name]['min'] = np_min
                    chunk_quant_param_details[tensor_name]['max'] = np_max
                    extracted.add(quantized_model_name)

        assert len(subgraph.outputs) == 3
        key_cache_output = subgraph.outputs[1]
        if key_cache_output.endswith('_requantize'):
            key_cache_output = key_cache_output[:-11]
        if config.l.ring_buffer and key_cache_output not in extracted:
            # get all output to.* on k side
            # FQ Model does not care about cache dtype
            tensor = subgraph.tensor_map[key_cache_output]
            cache_precision = tensor.type
            quant_param = tensor_utils.get_linear_quant_param(tensor)
            linear_quant_param_utils.deduce_minmax_from_quant_param(quant_param, tensor.type)
            bitwidth = type_utils.get_quant_bitwidth(tensor.type)
            if 'gecko' in model_type:
                tensor_name = f'layers.{chunk_idx}.self_attn.k_cat'
            else:
                tensor_name = f'layers.{chunk_idx}.self_attn.k_add'
            chunk_quant_param_details[tensor_name] = {}
            np_min = np.array(quant_param.min_vals, dtype=np.float32)
            np_max = np.array(quant_param.max_vals, dtype=np.float32)
            np_scale = quant_param.scale_vals[0]
            if cache_precision == nrp_pb2.DT_INT8:
                np_scale = np_scale / 256.0
            np_zp = quant_param.zero_point_vals[0]
            chunk_quant_param_details[tensor_name]['bitwidth'] = bitwidth
            chunk_quant_param_details[tensor_name]['scale'] = np_scale
            chunk_quant_param_details[tensor_name]['zp'] = np_zp
            chunk_quant_param_details[tensor_name]['min'] = np_min
            chunk_quant_param_details[tensor_name]['max'] = np_max
            extracted.add(key_cache_output)

    return chunk_quant_param_details


def map_quant_params(
    fq_model,
    extracted_q_param_dict: dict,
    buffer_scale=1.0,
    target_with_fakequant=None,
):
    """Map quantization parameters and de-quantized weights extracted from pre-quantized model to a FQ model.

    Args:
        fq_model (FakeQuantModel): The FQ model to insert quant params into.
        extracted_q_param_dict (dict): A dictionary mapping names of weight/activation to their corresponding
            quantization parameters.
        buffer_scale (float): Scale factor applied to quantization buffers.
        target_with_fakequant (list): List of strings indicating which targets are expected to have fake quantization.
            Default is ['weight', 'activation'].
    """
    if target_with_fakequant is None:
        target_with_fakequant = ['weight', 'activation']

    fq_model_quantizer_keys = {k for k in fq_model._quantizer_dict if 'lora' not in k}  # noqa: SLF001
    fq_model_lora_quantizer_keys = {k for k in fq_model._quantizer_dict if 'lora' in k}  # noqa: SLF001

    # Insert qparams for lora activations into extracted_q_param_dict
    for key in fq_model_lora_quantizer_keys:
        name_in_extracted_dict = (
            key.split('_lora')[0].replace('_proj', '') + '_proj'
        )  # Only qkv lora has _proj in pytorch name
        if name_in_extracted_dict not in extracted_q_param_dict:
            logger.error(
                f'Expected {name_in_extracted_dict} to be in extracted quant param dict but does not exist',
                err=KeyError,
            )
        extracted_q_param_dict[key] = copy.deepcopy(extracted_q_param_dict[name_in_extracted_dict])
        # Apply 20% buffer to lora add activation minmaxes
        if key.endswith('lora_add'):
            if len(extracted_q_param_dict[key]) == 0:
                # FC proj output is FP
                continue
            bitwidth = extracted_q_param_dict[key]['bitwidth']
            extracted_q_param_dict[key]['min'][0] = (
                extracted_q_param_dict[key]['min'][0] - extracted_q_param_dict[key]['zp']
            ) * buffer_scale + extracted_q_param_dict[key]['zp']
            extracted_q_param_dict[key]['max'][0] = (
                extracted_q_param_dict[key]['max'][0] - extracted_q_param_dict[key]['zp']
            ) * buffer_scale + extracted_q_param_dict[key]['zp']
            qmax = 2 ** (bitwidth - 1) - 1
            qmin = -(2 ** (bitwidth - 1))
            extracted_q_param_dict[key]['scale'] = (
                extracted_q_param_dict[key]['max'][0] - extracted_q_param_dict[key]['min'][0]
            ) / (qmax - qmin)

    # We overwrite the quantizer bitwidth/symmetric setting based on the actual tensor's quant param
    #
    # Note that symmetric setting is set to `False` directly for simplicity cause we don't need to
    # update the minmax value during training.
    visited = set()
    visited_quantizer = set()
    for name, module in fq_model.named_modules():
        if name not in visited and name in extracted_q_param_dict:
            if 'activation' in target_with_fakequant:
                # handle case where activation is FP
                if extracted_q_param_dict[name] == {}:
                    logger.debug(f'Found tensor {name} without quant params. Removing FQ op.')
                    del module._act_quantizer  # noqa: SLF001
                    visited_quantizer.add(name)
                else:
                    module._act_quantizer.bitwidth = extracted_q_param_dict[name]['bitwidth']  # noqa: SLF001
                    module._act_quantizer.symmetric = False  # noqa: SLF001
                    module._act_quantizer.set_minmax(  # noqa: SLF001
                        extracted_q_param_dict[name]['min'][0], extracted_q_param_dict[name]['max'][0]
                    )
                    module._act_quantizer._scale_val = extracted_q_param_dict[name]['scale']  # noqa: SLF001
                    module._act_quantizer._zero_point_val = extracted_q_param_dict[name]['zp']  # noqa: SLF001
                    module._act_quantizer.freeze()  # noqa: SLF001
                    visited_quantizer.add(name)
            visited.add(name)  # act quant
            weight_name = name + '.weight'  # if this exists then we are dealing with FC layer
            if weight_name in extracted_q_param_dict:
                with torch.no_grad():
                    module.weight.copy_(
                        torch.tensor(extracted_q_param_dict[weight_name]['weight'], dtype=module.weight.dtype)
                    )
                if 'weight' in target_with_fakequant:
                    module._weight_quantizer.bitwidth = extracted_q_param_dict[name]['bitwidth']  # noqa: SLF001
                    module._weight_quantizer.symmetric = False  # noqa: SLF001
                    module._weight_quantizer.set_minmax(  # noqa: SLF001
                        extracted_q_param_dict[weight_name]['min'], extracted_q_param_dict[weight_name]['max']
                    )
                    module._weight_quantizer._scale_vals.copy_(  # noqa: SLF001
                        extracted_q_param_dict[weight_name]['scale']
                    )
                    module._weight_quantizer._zero_point_vals.copy_(  # noqa: SLF001
                        extracted_q_param_dict[weight_name]['zp']
                    )
                    module._weight_quantizer.freeze()  # noqa: SLF001
                    visited_quantizer.add(weight_name)
                visited.add(weight_name)
            bias_name = name + '.bias'
            if bias_name in extracted_q_param_dict:
                with torch.no_grad():
                    module.bias.copy_(
                        torch.tensor(extracted_q_param_dict[bias_name]['weight'], dtype=module.bias.dtype)
                    )
                visited.add(bias_name)
    if len(visited) != len(extracted_q_param_dict.keys()):
        logger.error(
            'Quant Params not mapped yet from extracted_q_param_dict: '
            f'{set(extracted_q_param_dict.keys()).symmetric_difference(visited)}'
        )
    if len(visited_quantizer) != len(fq_model_quantizer_keys) + len(fq_model_lora_quantizer_keys):
        logger.error(
            'Quant Params not mapped yet from quantizer_dict:\n'
            f'{fq_model_quantizer_keys.union(fq_model_lora_quantizer_keys).symmetric_difference(visited_quantizer)}'
        )


def _remove_existing_lora_graph(subgraph, target_module):
    lora_add_name = f'{target_module.split("_proj")[0]}_lora_add'
    found = False

    target_removed_tensor_name = None
    replacement_tensor_name = None
    consumer_ops = None
    for op in subgraph.operators:
        if op.type not in [defs.Op.Add]:
            continue
        for output_tensor in op.outputs:
            if lora_add_name in output_tensor:
                logger.info(f'Removing old lora for {target_module}')
                found = True
                target_removed_tensor_name = output_tensor

                consumer_ops = subgraph_utils.get_consumer_ops(subgraph, target_removed_tensor_name)
                # find replacement tensor from the add input
                for input_tensor in op.inputs:
                    if 'lora' not in input_tensor:
                        replacement_tensor_name = input_tensor
                        break
                break

        if found:
            logger.info(f'removed tensor: {target_removed_tensor_name}, replacement tensor: {replacement_tensor_name}')
            for consumer in consumer_ops:
                consumer.inputs[:] = [
                    x if x != target_removed_tensor_name else replacement_tensor_name for x in consumer.inputs
                ]
            break

    # propagate quant params starting from the replacement tensor
    replacement_tensor = subgraph.tensor_map[replacement_tensor_name]
    visited = set()
    quantized_model_utils._dfs_propagate_quant_param(  # noqa: SLF001
        subgraph,
        replacement_tensor_name,
        replacement_tensor.quant_param.linear.scale_vals[0],
        replacement_tensor,
        visited,
    )


def undo_i16_v_proj_i8_cache_fc_changes(current_type, target_type, is_quantized, fc_output_tensor):
    """Undo specific optimizations applied by int16 v_proj and int8 cache when doing LoRA hotplug.

    Args:
        current_type: The current data type of the FC output tensor.
        target_type: The desired data type to convert the FC output tensor back to.
        is_quantized: Flag indicating whether the model is currently quantized.
        fc_output_tensor: The output tensor of FC layer.

    Returns:
        The quant params of fc_output_tensor or None.
    """
    if current_type != target_type and is_quantized:
        # keep old tensor scale
        org_scale = fc_output_tensor.quant_param.linear.scale_vals[0]
        org_zp = fc_output_tensor.quant_param.linear.zero_point_vals[0]

        # save original quant param
        modify_precision_tensor_quant_param = nrp_pb2.QuantParamProto()
        modify_precision_tensor_quant_param.CopyFrom(fc_output_tensor.quant_param)

        if current_type == nrp_pb2.DT_INT8 and target_type == nrp_pb2.DT_INT16:
            new_scale = org_scale / 256.0
        else:
            return None

        # build new quant param for FC output
        new_quant_param = nrp_pb2.LinearQuantParamProto()
        new_quant_param.scale_vals.append(new_scale)
        new_quant_param.zero_point_vals.append(org_zp)
        new_quant_param.dimensions.extend([])
        nrp_type = nrp_pb2.DT_INT16
        linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, nrp_type)

        fc_output_tensor.type = target_type
        fc_output_tensor.quant_param.linear.CopyFrom(new_quant_param)

        return modify_precision_tensor_quant_param

    return None


def add_dynamic_lora_to_base_quantized_model(
    subgraph,
    quantized_model_info,
    pipeline,
    file_format_separator,
    lora_start_layer_idx,
    lora_end_layer_idx,
    buffer_scale=1.0,
):
    """Adds dynamic LoRA (Low-Rank Adaptation) to a base quantized model.

    Args:
        subgraph (Subgraph): The Subgraph proto object.
        quantized_model_info (dict): The quantized model info dictionary.
        pipeline (FloatPipeline): The FloatPipeline class.
        file_format_separator (str): The separator used in the quantized model.
        lora_start_layer_idx (int): The first layer index in which lora will be inserted.
        lora_end_layer_idx (int): The last layer index in which lora will be inserted.
        buffer_scale (float, optional): The scale factor to apply to LoRA add activation min/max values. Default is 1.0.

    Returns:
        dict: The modified quantized_model_info dictionary with LoRA information added.
    """
    from .precision_config import PTQPrecisionConfig

    target_modules = pipeline.lora_handler.global_llm_target_modules
    if not (lora_end_layer_idx > lora_start_layer_idx):
        logger.error('lora_end_layer_idx must be greater than lora_start_layer_idx', err=ValueError)

    precision_config = quantized_model_info['precision_config']
    lora_precision = precision_config['lora_precision']
    # type utils requires float32 instead of float or FP but quantized model info uses FP
    target_lora_precision = PTQPrecisionConfig.get_precision_name(lora_precision)[1]
    target_lora_bitwith = PTQPrecisionConfig.get_bitwidth(lora_precision)[1]
    if target_lora_precision == 'float':
        target_lora_precision = 'float32'
    target_lora_nrp_dtype = type_utils.get_nrp_type(target_lora_precision)
    target_lora_np_dtype = quantized_model_utils._make_dtype_numpy(target_lora_precision)  # noqa: SLF001

    if target_lora_precision != 'float32':
        target_lora_symmetric_setting = PTQPrecisionConfig.get_symmetric_setting(lora_precision)[1]
        final_lora_qparam = pipeline.lora_handler.get_widest_lora_ranges()
        # (scale, zp)
        final_lora_qparam = {
            k: deduce_scale_zp_from_minmax(v['min'], v['max'], target_lora_bitwith, target_lora_symmetric_setting)
            for k, v in final_lora_qparam.items()
        }
        logger.debug(f'Final Lora scale and zp: {final_lora_qparam}')

    is_fp_respath = (
        precision_config['respath_precision']
        == precision_config['embeds_precision']
        == precision_config['logits_precision']
        == 'FP'
    )
    logger.debug(f'FP respath: {is_fp_respath}')

    # remove the old lora graph part if current model has lora
    if quantized_model_info['num_lora_inputs'] != 0:
        logger.info('Current quantized model has lora, removing old lora...')

        for target_module in target_modules:
            _remove_existing_lora_graph(subgraph, target_module)

        subgraph.inputs[:] = subgraph.inputs[: quantized_model_info['num_non_lora_inputs']]
        subgraph_utils.remove_unused_ops_and_tensors(subgraph)
        subgraph_utils.ensure_topological_op_order(subgraph)
        quantized_model_info['num_lora_inputs'] = 0
        quantized_model_info['input_scales'] = quantized_model_info['input_scales'][
            : quantized_model_info['num_non_lora_inputs']
        ]
        quantized_model_info['input_zero_points'] = quantized_model_info['input_zero_points'][
            : quantized_model_info['num_non_lora_inputs']
        ]
        quantized_model_info['input_dtypes'] = quantized_model_info['input_dtypes'][
            : quantized_model_info['num_non_lora_inputs']
        ]

    # enforce lora order
    lora_ops_mapping = {}
    all_operators = subgraph.operators
    for op in all_operators:
        if op.type != defs.Op.FullyConnected:
            continue

        name = op.inputs[1].split(file_format_separator)[-2]
        if name not in target_modules:
            continue

        lora_ops_mapping[name] = op

    lora_ops = [(lora_target_module, lora_ops_mapping[lora_target_module]) for lora_target_module in target_modules]

    for name, op in lora_ops:
        logger.debug(f'[Add dynamic lora] Current Tensor: {name}')

        # get fc target dtype from quantized info
        fc_simplified_name = name.split('_proj')[0]
        # FIXME: only works for per layer PTQ
        layer_idx = quantized_model_info['layer_ids'][0]
        fc_target_output_dtype = PTQPrecisionConfig.get_precision_name(
            quantized_model_info['precision_config']['llm_precision'][str(layer_idx)][fc_simplified_name]
        )[1]
        fc_target_output_nrp_dtype = type_utils.get_nrp_type(fc_target_output_dtype)

        fc_input = op.inputs[0]
        fc_weight_shape = subgraph.tensor_map[op.inputs[1]].shape
        fc_in_dim = fc_weight_shape.dims[1].dim_value
        fc_out_dim = fc_weight_shape.dims[0].dim_value
        fc_output = op.outputs[0]
        fc_output_tensor = subgraph.tensor_map[fc_output]
        fc_output_type = fc_output_tensor.type

        is_quantized = fc_output_type != type_utils.get_nrp_type('float32')
        logger.debug(f'[Add dynamic lora] Is Quantized: {is_quantized}')

        modify_precision_tensor_quant_param = undo_i16_v_proj_i8_cache_fc_changes(
            fc_output_type, fc_target_output_nrp_dtype, is_quantized, fc_output_tensor
        )

        # Check if FC input is output of quantize OP, if yes, directly take the FP input to quantize
        producer = subgraph_utils.get_producer_ops(subgraph, fc_input)[0]

        insert_dequant = False
        # This whole block is based on the assumption of fp lora
        if target_lora_precision == 'float32':
            if producer.type == defs.Op.Quantize:
                quantize_input = producer.inputs[0]
                if subgraph.tensor_map[quantize_input].type == nrp_pb2.DT_FLOAT32:
                    lora_a_input = quantize_input
                    insert_dequant = False
                else:
                    insert_dequant = True
            else:
                insert_dequant = True
        else:
            lora_a_input = fc_input

        # Add dequant OP before lora A BMM if needed
        if insert_dequant:
            dequant_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_A_input'])
            new_tensor = subgraph.tensor_map[dequant_tensor_name]
            new_tensor.type = nrp_pb2.DT_FLOAT32
            new_tensor.shape.CopyFrom(subgraph.tensor_map[fc_input].shape)
            lora_a_input = dequant_tensor_name

            dequant_op = subgraph.operators.add()
            dequant_op.type = defs.Op.Dequantize
            dequant_op.inputs.append(fc_input)
            dequant_op.outputs.append(lora_a_input)

        # Add lora A weight input
        lora_a_weight_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_A_weight'])
        new_tensor = subgraph.tensor_map[lora_a_weight_tensor_name]
        new_tensor.type = target_lora_nrp_dtype
        lora_a_weight_scale = None
        lora_a_weight_zp = None
        if target_lora_nrp_dtype != nrp_pb2.DT_FLOAT32:
            final_lora_qparam_name = f'0_{name.split("_", 1)[0]}_A'
            new_quant_param = nrp_pb2.LinearQuantParamProto()
            lora_a_weight_scale = final_lora_qparam[final_lora_qparam_name][0]
            lora_a_weight_zp = final_lora_qparam[final_lora_qparam_name][1]
            new_quant_param.scale_vals.append(lora_a_weight_scale)
            new_quant_param.zero_point_vals.append(lora_a_weight_zp)
            new_quant_param.dimensions.extend([])
            linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, target_lora_nrp_dtype)
            new_tensor.quant_param.type = nrp_pb2.QPT_LINEAR
            new_tensor.quant_param.linear.CopyFrom(new_quant_param)
        tensor_utils.set_shape(new_tensor, [1, None, fc_in_dim])

        # Add lora A BMM output tensor
        lora_a_output = file_format_separator.join(['layers', '0', name, 'lora_A', 'matmul'])
        new_tensor = subgraph.tensor_map[lora_a_output]
        # follow fc output precision only for non float hotplug case
        if target_lora_precision != 'float32':
            new_tensor.type = fc_output_type
            tensor_utils.copy_type_and_quant_param(new_tensor, fc_output_tensor)
        else:
            new_tensor.type = nrp_pb2.DT_FLOAT32
        tensor_utils.set_shape(new_tensor, [None, None, None])

        # Add lora A BMM OP
        lora_a_bmm_op = subgraph.operators.add()
        lora_a_bmm_op.type = defs.Op.MatMul
        operator_utils.set_bool_attr(lora_a_bmm_op, defs.Attr.TransposeA, False)
        operator_utils.set_bool_attr(lora_a_bmm_op, defs.Attr.TransposeB, True)
        lora_a_bmm_op.inputs.append(lora_a_input)
        lora_a_bmm_op.inputs.append(lora_a_weight_tensor_name)
        lora_a_bmm_op.outputs.append(lora_a_output)

        # Add lora B weight input
        lora_b_weight_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_B_weight'])
        new_tensor = subgraph.tensor_map[lora_b_weight_tensor_name]
        new_tensor.type = target_lora_nrp_dtype
        lora_b_weight_scale = None
        lora_b_weight_zp = None
        if target_lora_nrp_dtype != nrp_pb2.DT_FLOAT32:
            final_lora_qparam_name = f'0_{name.split("_", 1)[0]}_B'
            new_quant_param = nrp_pb2.LinearQuantParamProto()
            lora_b_weight_scale = final_lora_qparam[final_lora_qparam_name][0]
            lora_b_weight_zp = final_lora_qparam[final_lora_qparam_name][1]
            new_quant_param.scale_vals.append(lora_b_weight_scale)
            new_quant_param.zero_point_vals.append(lora_b_weight_zp)
            new_quant_param.dimensions.extend([])
            linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, target_lora_nrp_dtype)
            new_tensor.quant_param.type = nrp_pb2.QPT_LINEAR
            new_tensor.quant_param.linear.CopyFrom(new_quant_param)
        tensor_utils.set_shape(new_tensor, [1, fc_out_dim, None])

        # Add lora B BMM output tensor
        lora_b_output = file_format_separator.join(['layers', '0', name, 'lora_B', 'matmul'])
        new_tensor = subgraph.tensor_map[lora_b_output]
        tensor_utils.copy_type_and_quant_param(new_tensor, fc_output_tensor)
        tensor_utils.set_shape(new_tensor, [None, None, fc_out_dim])

        # Add lora B BMM OP
        lora_b_bmm_op = subgraph.operators.add()
        lora_b_bmm_op.type = defs.Op.MatMul
        operator_utils.set_bool_attr(lora_b_bmm_op, defs.Attr.TransposeA, False)
        operator_utils.set_bool_attr(lora_b_bmm_op, defs.Attr.TransposeB, True)
        lora_b_bmm_op.inputs.append(lora_a_output)
        lora_b_bmm_op.inputs.append(lora_b_weight_tensor_name)
        lora_b_bmm_op.outputs.append(lora_b_output)

        # Add add OP output tensor, use buffered output scale if necessary
        lora_add_out_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_add', 'add'])
        new_tensor = subgraph.tensor_map[lora_add_out_tensor_name]
        new_tensor.type = fc_output_type
        new_tensor.shape.CopyFrom(subgraph.tensor_map[lora_b_output].shape)

        if is_quantized:
            # Base model o_proj and down_proj would not have scale in fp respath
            if buffer_scale > 1.0:
                orig_scale = tensor_utils.get_linear_quant_param(fc_output_tensor).scale_vals[0]
                new_quant_param = nrp_pb2.LinearQuantParamProto()
                new_quant_param.scale_vals.append(orig_scale * buffer_scale)
                new_quant_param.zero_point_vals.append(0)
                new_quant_param.dimensions.extend([])
                linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, fc_output_tensor.type)
                tensor_utils.apply_linear_quant_param(new_tensor, fc_output_tensor.type, new_quant_param)

                # Add add OP output requant tensor, use original output scale
                lora_add_requant_out_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_add_requant'])
                new_tensor = subgraph.tensor_map[lora_add_requant_out_tensor_name]
                tensor_utils.copy_type_and_quant_param(new_tensor, fc_output_tensor)
                new_tensor.shape.CopyFrom(subgraph.tensor_map[lora_add_out_tensor_name].shape)
            else:
                tensor_utils.copy_type_and_quant_param(new_tensor, fc_output_tensor)

            if modify_precision_tensor_quant_param is not None:
                # create new tensor
                modify_precision_tensor_name = file_format_separator.join(['layers', '0', name, 'lora_add_quantize'])
                new_modify_precision_tensor = subgraph.tensor_map[modify_precision_tensor_name]
                # copy shape from add tensor
                new_modify_precision_tensor.shape.CopyFrom(new_tensor.shape)
                new_modify_precision_tensor.type = nrp_pb2.DT_INT8
                new_modify_precision_tensor.quant_param.CopyFrom(modify_precision_tensor_quant_param)

        # Reroute consumer inputs of FC output before connecting the add OP & requant OP
        consumers = subgraph_utils.get_consumer_ops(subgraph, fc_output)
        for consumer in consumers:
            if buffer_scale > 1.0 and is_quantized:
                consumer.inputs[:] = [
                    x if x != fc_output else lora_add_requant_out_tensor_name for x in consumer.inputs
                ]
            else:
                if modify_precision_tensor_quant_param is not None:
                    consumer.inputs[:] = [
                        x if x != fc_output else modify_precision_tensor_name for x in consumer.inputs
                    ]
                else:
                    consumer.inputs[:] = [x if x != fc_output else lora_add_out_tensor_name for x in consumer.inputs]

        add_op = subgraph.operators.add()
        add_op.type = defs.Op.Add
        operator_utils.set_str_attr(add_op, defs.Attr.FusedActivation, defs.FusedActivation.kNone)
        add_op.inputs.append(fc_output)
        add_op.inputs.append(lora_b_output)
        add_op.outputs.append(lora_add_out_tensor_name)

        if buffer_scale > 1.0 and is_quantized:
            # Add requant OP to requant back to original minmax
            requant_op = subgraph.operators.add()
            requant_op.type = defs.Op.Requantize
            requant_op.inputs.append(lora_add_out_tensor_name)
            requant_op.outputs.append(lora_add_requant_out_tensor_name)

        if modify_precision_tensor_quant_param is not None:
            quant_op = subgraph.operators.add()
            quant_op.type = defs.Op.Quantize
            quant_op.inputs.append(lora_add_out_tensor_name)
            quant_op.outputs.append(modify_precision_tensor_name)

        # Add lora inputs to subgraph inputs
        subgraph.inputs.append(lora_a_weight_tensor_name)
        subgraph.inputs.append(lora_b_weight_tensor_name)
        quantized_model_info['num_lora_inputs'] += 2
        quantized_model_info['input_scales'] += [lora_a_weight_scale, lora_b_weight_scale]
        quantized_model_info['input_zero_points'] += [lora_a_weight_zp, lora_b_weight_zp]
        quantized_model_info['input_dtypes'] += [target_lora_np_dtype, target_lora_np_dtype]

    quantized_model_info['lora_rank'] = pipeline.lora_handler.global_llm_max_rank
    quantized_model_info['lora_start_layer_idx'] = lora_start_layer_idx
    quantized_model_info['lora_end_layer_idx'] = lora_end_layer_idx
    quantized_model_info['precision_config']['has_lora'] = True
    quantized_model_info['precision_config']['lora_precision'] = lora_precision

    return quantized_model_info


def init_lora_A(module):  # noqa: N802
    """Initialize LoRA down projections with kaiming uniform distribution.

    Args:
        module (nn.Linear): The torch module to initialize.
    """
    if not isinstance(module, nn.Linear):
        logger.error(f'Expect torch.nn.Linear module for `init_lora_A` but got {type(module)} instead.')
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))


def init_lora_B(module):  # noqa: N802
    """Initialize LoRA up projections with zeros.

    Args:
        module (nn.Linear): The torch module to initialize.
    """
    if not isinstance(module, nn.Linear):
        logger.error(f'Expect torch.nn.Linear module for `init_lora_B` but got {type(module)} instead.')
    nn.init.zeros_(module.weight)


def fake_quant(x, bits, axis):
    """Applies fake quant to x.

    Modified from Google's implementation. Currently used for inference float testing.
    """
    scale_epsilon: float = 1e-8

    min_value = -1 * 2 ** (bits - 1)
    max_value = 2 ** (bits - 1) - 1

    bound = torch.amax(torch.abs(x), dim=axis, keepdim=True)
    scale_bound = max_value
    scale = bound / scale_bound
    scale = scale + scale_epsilon

    x_quant = torch.divide(x, scale)
    # Round to integer only for the forward pass.
    x_quant = torch.round(x_quant)
    x_quant = torch.clip(x_quant, min_value, max_value)
    # Dequantize
    return torch.multiply(x_quant, scale)


def get_quantized_value_range(bitwidth):
    """Get the quantized value range based on the given bitwidth.

    Adapted from mtk_converter.

    The value range is deduced based on the signed quantized data type to fit converter interface.

    Args:
        bitwidth: An int value. The bitwidth to deduce the quantized value range.
    """
    qmin = -1 * (1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    return qmin, qmax


def deduce_scale_zp_from_minmax(min_val, max_val, bitwidth, is_symmetric):
    """Deduce the quant scale and zero_point of a tensor given its min, max and bitwidth.

    Args:
        min_val (float): The min value of the tensor.
        max_val (float): The max value of the tensor.
        bitwidth (int): The bitwidth to quantize the tensor to.
        is_symmetric (bool): Whether to use symmetric or asymmetric quantization.

    Returns:
        scale (float): The quantization scale.
    """
    min_val = torch.minimum(torch.tensor(0.0), min_val)
    max_val = torch.maximum(torch.tensor(0.0), max_val)

    qmin, qmax = get_quantized_value_range(bitwidth)

    if is_symmetric:
        scale_from_min = min_val / qmin
        scale_from_max = max_val / qmax
        scale = torch.maximum(scale_from_min, scale_from_max).item()
        zp = 0
    else:
        scale = ((max_val - min_val) / (qmax - qmin)).item()
        zp = qmin if scale == 0.0 else torch.round(qmin - min_val / scale).item()

    return scale, zp
