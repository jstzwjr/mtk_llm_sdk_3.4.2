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
"""Define quantized model related helper functions for mtk_llm_sdk."""

import json
import os
import types

import numpy as np
from mtk_converter.python.converters.mlir import importer as mlir_importer
from mtk_converter.python.converters.tflite import importer as tflite_importer
from mtk_converter.python.proto import nrpmodel_pb2 as nrp_pb2
from mtk_converter.python.utils import defs, linear_quant_param_utils, subgraph_utils, tensor_utils, type_utils

from .. import __version__
from . import const, logger, utils


def _append_dequantize_op(subgraph, output_idx, name):
    logger.debug(f'Appending dequantize OP at output index {output_idx}')
    org_name = subgraph.outputs[output_idx]
    org_tensor = subgraph.tensor_map[org_name]

    nrp_type = type_utils.get_nrp_type('float32')

    new_tensor = subgraph.tensor_map[name]
    new_tensor.type = nrp_type
    new_tensor.shape.CopyFrom(org_tensor.shape)
    requant_op = subgraph.operators.add()
    requant_op.type = defs.Op.Dequantize
    requant_op.inputs.append(org_name)
    requant_op.outputs.append(name)
    subgraph.outputs[output_idx] = name


def _append_requantize_op(subgraph, output_idx, name, dtype, scale, zero_point):
    logger.debug(f'Appending requantize OP at output index {output_idx} to dtype {dtype}')
    org_name = subgraph.outputs[output_idx]
    org_tensor = subgraph.tensor_map[org_name]

    if dtype == 'float':
        dtype = 'float32'

    new_tensor = subgraph.tensor_map[name]
    nrp_type = type_utils.get_nrp_type(dtype)
    new_quant_param = nrp_pb2.LinearQuantParamProto()
    new_quant_param.scale_vals.append(scale)
    new_quant_param.zero_point_vals.append(zero_point)
    new_quant_param.dimensions.extend([])
    linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, nrp_type)

    new_tensor = subgraph.tensor_map[name]
    new_tensor.type = nrp_type
    new_tensor.shape.CopyFrom(org_tensor.shape)
    new_tensor.quant_param.type = nrp_pb2.QPT_LINEAR
    new_tensor.quant_param.linear.CopyFrom(new_quant_param)
    requant_op = subgraph.operators.add()
    requant_op.type = defs.Op.Requantize
    requant_op.inputs.append(org_name)
    requant_op.outputs.append(name)
    subgraph.outputs[output_idx] = name


def _export_nrp_to_mlir(nrpmod, output_dest_path, np_compat=None):
    """Exports mtk_converter internal representation quantized model into mlir format.

    Args:
        nrpmod: The mtk_converter internal representation model.
        output_dest_path: The output path to the mlir format quantized model.
        np_compat (int, optional): The specific neuropilot version op export spec to use during export. Defaults to
            the currently installed mtk_converter version.
    """
    logger.debug('Exporting to MLIR')
    assert output_dest_path.endswith('.mlir')
    from mtk_converter.python.converters.mlir import exporter as mlir_exporter

    cvt_version = utils.get_converter_version()
    if np_compat is None:
        np_compat = cvt_version
    logger.debug(f'Using NP{np_compat} compatibility OP spec.')
    mlir_op_export_spec = utils.get_op_export_spec(np_compat, 'mlir')

    mlir_buffer = mlir_exporter.export(nrpmod, True, mlir_op_export_spec)
    with open(output_dest_path, 'wb') as mlir_file:
        mlir_file.write(mlir_buffer)


def _export_nrp_to_tflite(nrpmod, output_dest_path, np_compat=None):
    """Exports mtk_converter internal representation quantized model into tflite format.

    Args:
        nrpmod: The mtk_converter internal representation model.
        output_dest_path: The output path to the tflite format quantized model.
        np_compat (int, optional): The specific neuropilot version op export spec to use during export. Defaults to
            the currently installed mtk_converter version.
    """
    logger.debug('Exporting to TFLite')
    assert output_dest_path.endswith('.tflite')
    from mtk_converter.python.converters.tflite import exporter as tflite_exporter

    cvt_version, cvt_minor = utils.get_converter_version(include_minor=True)
    if np_compat is None:
        np_compat = cvt_version
    logger.debug(f'Using NP{np_compat} compatibility OP spec.')
    tflite_op_export_spec = utils.get_op_export_spec(np_compat, 'tflite')

    if (cvt_version == 8 and cvt_minor >= 5) or cvt_version >= 9:
        tflite_buffer = tflite_exporter.export(
            nrpmod,
            tflite_op_export_spec,
            custom_description=f'Post Training Quantized by mtk_llm_sdk v{__version__}',
        )
    with open(output_dest_path, 'wb') as tflite_file:
        tflite_file.write(tflite_buffer)


def _get_dtype(subgraph, index, mode):
    if mode not in ['input', 'output']:
        logger.error('mode must be one of: input, output')
    if mode == 'input':
        if index >= len(subgraph.inputs):
            logger.error(f'index is {index} but quantized model only has {len(subgraph.inputs)} inputs')
        name = subgraph.inputs[index]
    elif mode == 'output':
        if index >= len(subgraph.outputs):
            logger.error(f'index is {index} but quantized model only has {len(subgraph.outputs)} outputs')
        name = subgraph.outputs[index]

    dtype = subgraph.tensor_map[name].type
    return type_utils.get_numpy_type(dtype)


def _dfs_propagate_quant_param(subgraph, source_name, ref_tensor_scale, ref_tensor, visited: set):
    consumers = subgraph_utils.get_consumer_ops(subgraph, source_name)
    for consumer in consumers:
        if consumer.type not in (
            defs.Op.MatMul,
            defs.Op.Mul,
            defs.Op.Add,
            defs.Op.Shape,
            defs.Op.Quantize,
            defs.Op.Requantize,
            defs.Op.Dequantize,
            defs.Op.Neg,
        ):
            for output in consumer.outputs:
                op_output = subgraph.tensor_map[output]
                if output not in visited and tensor_utils.is_quantized(op_output):
                    visited.add(output)
                    try:
                        op_out_scale = op_output.quant_param.linear.scale_vals[0]

                        # TODO: Remove Redundant Requantize

                        if op_out_scale != ref_tensor_scale:
                            tensor_utils.copy_type_and_quant_param(op_output, ref_tensor)
                        curr_tensor = output
                        _dfs_propagate_quant_param(subgraph, curr_tensor, ref_tensor_scale, ref_tensor, visited)

                    except Exception as e:
                        print(f'Error for {output}:\n{e}')
                        exit()


def _intra_chunk_qparam_alignment(subgraph, config, quantized_model_info):
    logger.debug('Performing intra-chunk qparam alignment')
    num_non_cache_inputs = quantized_model_info['num_non_cache_inputs']
    num_other_masks_inputs = quantized_model_info['num_other_masks_inputs']

    if quantized_model_info['model_type'] == 'whisper_decoder':
        num_non_cache_inputs -= 2  # remove cross attn here

    cache_start_idx = num_non_cache_inputs - num_other_masks_inputs
    cache_outputs = subgraph.outputs[1:3] if config.extra_output['attn_weights'] else subgraph.outputs[1:]

    if (
        quantized_model_info['num_lora_inputs'] > 0
        or quantized_model_info['model_type'] == 'whisper_decoder'
        or quantized_model_info['infini_attention']
        or quantized_model_info['use_split_mask']
    ):
        cache_inputs = subgraph.inputs[cache_start_idx : cache_start_idx + len(cache_outputs)]
    else:
        cache_inputs = subgraph.inputs[cache_start_idx:]

    if len(cache_inputs) != len(cache_outputs):
        logger.error(
            f'Number of cache inputs ({len(cache_inputs)}) '
            f'does not match number of cache outputs ({len(cache_outputs)}).'
        )

    cur_cache_input_index = cache_start_idx
    cur_cache_output_index = 1
    cache_input_output_mapping = dict(zip(cache_inputs, cache_outputs))
    for inp_key, out_key in cache_input_output_mapping.items():
        inp_tensor = subgraph.tensor_map[inp_key]
        out_tensor = subgraph.tensor_map[out_key]
        out_scale = out_tensor.quant_param.linear.scale_vals[0]
        out_zp = out_tensor.quant_param.linear.zero_point_vals[0]
        tensor_utils.copy_type_and_quant_param(inp_tensor, out_tensor)

        assert tensor_utils.have_same_type_and_quant_param([subgraph.tensor_map[inp_key], subgraph.tensor_map[out_key]])

        # TODO: Improve this part to be more general
        if config.extra_input['sink_rope']:
            visited = set()
            _dfs_propagate_quant_param(subgraph, inp_key, out_scale, out_tensor, visited)
        else:
            curr_tensor = inp_key
            i = 0
            while True:
                consumer = subgraph_utils.get_consumer_ops(subgraph, curr_tensor)[i]
                if consumer.type == defs.Op.MatMul:
                    break
                if consumer.type == defs.Op.Shape:
                    i += 1  # Skip shape OP
                else:
                    for input_ in consumer.inputs:
                        op_input = subgraph.tensor_map[input_]
                        try:
                            op_inp_scale = op_input.quant_param.linear.scale_vals[0]
                            op_inp_zp = op_input.quant_param.linear.zero_point_vals[0]
                            if op_inp_scale != out_scale or op_inp_zp != out_zp:
                                tensor_utils.copy_type_and_quant_param(op_input, out_tensor)
                            break
                        except IndexError:
                            continue
                    for output in consumer.outputs:
                        op_output = subgraph.tensor_map[output]
                        try:
                            op_out_scale = op_output.quant_param.linear.scale_vals[0]
                            op_out_zp = op_output.quant_param.linear.zero_point_vals[0]
                            if op_out_scale != out_scale or op_out_zp != out_zp:
                                tensor_utils.copy_type_and_quant_param(op_output, out_tensor)
                            curr_tensor = output
                            i = 0
                            break
                        except IndexError:
                            continue

        quantized_model_info['input_scales'][cur_cache_input_index] = out_scale
        quantized_model_info['input_zero_points'][cur_cache_input_index] = out_zp
        quantized_model_info['input_dtypes'][cur_cache_input_index] = quantized_model_info['output_dtypes'][
            cur_cache_output_index
        ]
        cur_cache_input_index += 1
        cur_cache_output_index += 1

    if config.model_type == 'eagle':
        org_tensor = subgraph.tensor_map[subgraph.inputs[1]]
        org_scale = org_tensor.quant_param.linear.scale_vals[0]
        org_zp = org_tensor.quant_param.linear.zero_point_vals[0]
        dtype = type_utils.TYPE_NAME_MAPPINGS[org_tensor.type]
        _append_requantize_op(subgraph, 0, 'hidden_states_requantize', dtype, org_scale, org_zp)


def _make_dtype_numpy(dtype):
    if dtype in ['float', 'float32']:
        return np.float32
    if dtype == 'int32':
        return np.int32
    if dtype == 'float16':
        return np.float16
    if dtype == 'int16':
        return np.int16
    if dtype == 'int8':
        return np.int8
    logger.error(f'Unexpected dtype: {dtype}')
    raise


def _make_quantized_model_info_dtype_numpy(quantized_model_info):
    for i in range(len(quantized_model_info['output_dtypes'])):
        dtype = quantized_model_info['output_dtypes'][i]
        quantized_model_info['output_dtypes'][i] = _make_dtype_numpy(dtype)
    for i in range(len(quantized_model_info['input_dtypes'])):
        dtype = quantized_model_info['input_dtypes'][i]
        quantized_model_info['input_dtypes'][i] = _make_dtype_numpy(dtype)
    return quantized_model_info


def _prepend_quantize_op(subgraph, input_idx, name):
    logger.debug(f'Prepending quantize OP at input index {input_idx}')
    org_name = subgraph.inputs[input_idx]
    org_tensor = subgraph.tensor_map[org_name]

    nrp_type = type_utils.get_nrp_type('float32')

    new_tensor = subgraph.tensor_map[name]
    new_tensor.type = nrp_type
    new_tensor.shape.CopyFrom(org_tensor.shape)
    quant_op = subgraph.operators.add()
    quant_op.type = defs.Op.Quantize
    quant_op.inputs.append(name)
    quant_op.outputs.append(org_name)
    subgraph.inputs[input_idx] = name


def _remove_rotemb_quantize(subgraph):
    operators = subgraph.operators
    for op in operators:
        if op.type != defs.Op.Neg:
            continue
        ref_tensor = subgraph.tensor_map[op.inputs[0]]
        tensor_utils.copy_type_and_quant_param(subgraph.tensor_map[op.outputs[0]], ref_tensor)

        split_op = subgraph_utils.get_producer_ops(subgraph, op.inputs[0])
        if len(split_op) != 1:
            logger.error('Found non rot emb split ops!')
        split_op = split_op[0]
        if split_op.type != defs.Op.Split:
            logger.error(f'Expected op to be of type "Split" but found op of type "{split_op.type}" instead')

        split_output = [x for x in split_op.outputs if x != op.inputs[0]]
        if len(split_output) != 1:
            logger.error(f'Expected only one other split output but found: {split_output}')
        split_output = split_output[0]

        quantize_op = subgraph_utils.get_consumer_ops(subgraph, split_output)
        if len(quantize_op) != 1:
            logger.error('Non rot emb split encountered')
        quantize_op = quantize_op[0]
        if quantize_op.type not in (defs.Op.Quantize, defs.Op.Requantize):
            logger.error('Non rot emb split encountered')

        concat_op = subgraph_utils.get_consumer_ops(subgraph, op.outputs[0])
        if len(concat_op) != 1:
            logger.error('Rot emb branch has more than one consumer')
        concat_op = concat_op[0]
        if concat_op.type != defs.Op.Concat:
            logger.error('Rot emb does not have concat')

        concat_op.inputs[:] = [x if x == op.outputs[0] else split_output for x in concat_op.inputs]

        tensor_utils.copy_type_and_quant_param(subgraph.tensor_map[concat_op.outputs[0]], ref_tensor)

        subgraph.operators.remove(quantize_op)


def change_embed_dtype(subgraph, target_dtype, quantized_model_info):
    """Changes the given subgraph's embedding data type to the specified target data type.

    Args:
        subgraph: The internal representation subgraph.
        target_dtype: The target data type. Current supported: int8, int16, float32.
        quantized_model_info: The quantized model info dictionary.
    """
    embed_dtype = quantized_model_info['input_dtypes'][0]
    if embed_dtype == _make_dtype_numpy(target_dtype):
        return quantized_model_info

    if embed_dtype == np.float32:
        logger.error('Unable to quantize float embedding inputs/outputs outside of PTQ flow.')

    org_name = subgraph.inputs[0]
    name = org_name + '_requantize'

    org_tensor = subgraph.tensor_map[org_name]
    org_scale = org_tensor.quant_param.linear.scale_vals[0]
    org_zp = org_tensor.quant_param.linear.zero_point_vals[0]

    org_absmax = const.DTYPE_ABSMAX_MAP[embed_dtype]
    target_absmax = const.DTYPE_ABSMAX_MAP[_make_dtype_numpy(target_dtype)]

    new_scale = org_scale * org_absmax / target_absmax

    logger.debug(f'Appending requantize OP at input embedding to dtype {target_dtype}')

    nrp_type = type_utils.get_nrp_type(target_dtype)
    new_quant_param = nrp_pb2.LinearQuantParamProto()
    new_quant_param.scale_vals.append(new_scale)
    new_quant_param.zero_point_vals.append(org_zp)
    new_quant_param.dimensions.extend([])
    linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, nrp_type)

    new_tensor = subgraph.tensor_map[name]
    new_tensor.type = nrp_type
    new_tensor.shape.CopyFrom(org_tensor.shape)
    new_tensor.quant_param.type = nrp_pb2.QPT_LINEAR
    new_tensor.quant_param.linear.CopyFrom(new_quant_param)

    requant_op = subgraph.operators.add()
    requant_op.type = defs.Op.Requantize
    requant_op.inputs.append(name)
    requant_op.outputs.append(org_name)
    subgraph.inputs[0] = name

    quantized_model_info['input_scales'][0] = new_scale
    quantized_model_info['input_dtypes'][0] = _make_dtype_numpy(target_dtype)

    return quantized_model_info


def _sanitize_quantized_model_info(subgraph, quantized_model_info, mode):
    if mode == 'input':
        subgraph_tensors = subgraph.inputs
        logger.debug(f'[sanitize_quantized_model_info] Subgraph inputs:\n{subgraph_tensors}')
    elif mode == 'output':
        subgraph_tensors = subgraph.outputs
        logger.debug(f'[sanitize_quantized_model_info] Subgraph outputs:\n{subgraph_tensors}')
    else:
        logger.error(f'Invalid mode: {mode}')

    # check length of infos
    def _verify_length(cur_info, info_name, tensors):
        default_values = {'scales': None, 'zps': None, 'dtypes': np.float32}
        if len(cur_info) < len(tensors):
            logger.warning(
                f'There are missing info in {info_name}, appending default values {default_values[info_name]} to it...'
            )
            cur_info.extend([default_values[info_name] for _ in range(len(tensors) - len(cur_info))])
        elif len(cur_info) > len(tensors):
            logger.warning(f'There are more info values than expected in {info_name}, truncating...')
            cur_info = cur_info[: len(tensors)]
        return cur_info

    tensor_map = subgraph.tensor_map
    info_scales = _verify_length(quantized_model_info[f'{mode}_scales'], 'scales', subgraph_tensors)
    info_zps = _verify_length(quantized_model_info[f'{mode}_zero_points'], 'zps', subgraph_tensors)
    info_dtypes = _verify_length(quantized_model_info[f'{mode}_dtypes'], 'dtypes', subgraph_tensors)
    for idx, ten in enumerate(subgraph_tensors):
        logger.debug(f'\nChecking {mode} tensor: {ten}')
        tensor = tensor_map[ten]
        np_dtype = _get_dtype(subgraph, idx, mode)
        logger.debug(f'[sanitize_quantized_model_info] Quantized model dtype: {np_dtype}')

        info_dtype = info_dtypes[idx]
        info_scale = info_scales[idx]
        info_zp = info_zps[idx]
        logger.debug(f'[sanitize_quantized_model_info] Model info dtype: {info_dtype}')
        logger.debug(f'[sanitize_quantized_model_info] Model scale: {info_scale}')
        logger.debug(f'[sanitize_quantized_model_info] Model zp: {info_zp}')

        # FIXME: Will fail if act precision < 8
        if np_dtype != np.dtype(info_dtype):
            logger.warning(
                f'{mode} {ten} (index {idx}) has mismatched dtype with quantized model ({info_dtype} vs {np_dtype}). '
                'Updating model info.'
            )
            quantized_model_info[f'{mode}_dtypes'][idx] = np_dtype

        if np_dtype not in (np.float16, np.float32):
            qparams = tensor_utils.get_linear_quant_param(tensor)
            scale = qparams.scale_vals[0]
            zp = qparams.zero_point_vals[0]
            logger.debug(f'[sanitize_quantized_model_info] Quantized model scale: {scale}')
            logger.debug(f'[sanitize_quantized_model_info] Quantized model zp: {zp}')

            if info_scale != scale:
                logger.warning(
                    f'{mode} {ten} (index {idx}) has mismatched scale with quantized model ({info_scale} vs {scale}). '
                    'Updating model info.'
                )
                quantized_model_info[f'{mode}_scales'][idx] = scale
            if info_zp != zp:
                logger.warning(
                    f'{mode} {ten} (index {idx}) has mismatched zero point with quantized model ({info_zp} vs {zp}). '
                    'Updating model info.'
                )
                quantized_model_info[f'{mode}_zero_points'][idx] = zp

        else:
            if info_scale is not None:
                logger.warning(
                    f'{mode} {ten} (index {idx}) has mismatched scale with quantized model ({info_scale} vs None). '
                    'Updating model info.'
                )
                quantized_model_info[f'{mode}_scales'][idx] = None
            if info_zp is not None:
                logger.warning(
                    f'{mode} {ten} (index {idx}) has mismatched scale with quantized model ({info_zp} vs None). '
                    'Updating model info.'
                )
                quantized_model_info[f'{mode}_zero_points'][idx] = None


def change_cache_dtype(subgraph, config, chunk_idx, target_dtype, quantized_model_info):
    """Changes the given subgraph's cache data type to the specified target data type.

    Args:
        subgraph: The internal representation subgraph.
        config (BaseConfig): The model config.
        chunk_idx (int): The index of the current chunk.
        target_dtype: The target data type. Current supported: int8, int16, float32.
        quantized_model_info: The quantized model info dictionary.
    """
    logger.debug('Enter change_cache_dtype')

    if target_dtype == 'float':
        target_dtype = 'float32'  # mtk_converter's get_nrp_type does not accept 'float'
    target_np_dtype = _make_dtype_numpy(target_dtype)

    # FIXME: make more generic using start index
    num_non_cache_inputs = quantized_model_info['num_non_cache_inputs'] - quantized_model_info['num_other_masks_inputs']
    if quantized_model_info['model_type'] == 'whisper_decoder':
        num_non_cache_inputs -= 2  # remove cross attn here

    for output_idx in range(1, 3):
        output_cache_dtype = quantized_model_info['output_dtypes'][output_idx]
        is_output_cache_ok = False
        if output_cache_dtype == target_np_dtype:
            input_idx = output_idx + num_non_cache_inputs - 1
            input_cache_dtype = quantized_model_info['input_dtypes'][input_idx]
            if input_cache_dtype != target_np_dtype:
                is_output_cache_ok = True
            else:
                continue

        if output_cache_dtype == np.float32:
            logger.error('Unable to quantize float cache inputs/outputs outside of PTQ flow.', err=ValueError)
        elif target_np_dtype == np.float32:
            mode = 'dequantize'
        else:
            mode = 'requantize'

        org_name = subgraph.outputs[output_idx]

        # Append output (re/de)quant
        name = org_name + f'_{mode}'
        org_tensor = subgraph.tensor_map[org_name]
        target_nrp_type = type_utils.get_nrp_type(target_dtype)
        if mode == 'dequantize':
            _append_dequantize_op(subgraph, output_idx, name)
            quantized_model_info['output_scales'][output_idx] = None
            quantized_model_info['output_zero_points'][output_idx] = None
            quantized_model_info['output_dtypes'][output_idx] = np.float32
        else:
            # remove duplicate tensors
            if name in subgraph.tensor_map and config.extra_input['sink_rope']:
                consumers = subgraph_utils.get_consumer_ops(subgraph, name)
                assert len(consumers) == 1, f'{name} Consumers:\n{consumers}\n'
                assert consumers[0].type == 'StridedSlice', f'Consumer type: {consumers[0].type}'
                existing_requant_op = subgraph_utils.get_producer_ops(subgraph, name)[0]
                # create new tensor with new name
                new_name = name + '_slice'
                new_tensor = subgraph.tensor_map[new_name]
                org_req_tensor = subgraph.tensor_map[name]
                tensor_utils.copy_type_and_quant_param(new_tensor, org_req_tensor)
                new_tensor.shape.CopyFrom(org_req_tensor.shape)
                new_op = subgraph.operators.add()
                new_op.type = defs.Op.Requantize
                new_op.inputs.append(org_name)
                new_op.outputs.append(new_name)

                # remove previous op and tensor with same name
                subgraph.operators.remove(existing_requant_op)
                del subgraph.tensor_map[name]

                # reroute consumer input
                consumers[0].inputs[:] = [new_name] + consumers[0].inputs[1:]

            if is_output_cache_ok:
                # for the case where output cache is already correct dtype
                new_quant_param = org_tensor.quant_param.linear
                new_scale = org_tensor.quant_param.linear.scale_vals[0]  # technically org_scale
                org_zp = org_tensor.quant_param.linear.zero_point_vals[0]
            else:
                org_scale = org_tensor.quant_param.linear.scale_vals[0]
                org_zp = org_tensor.quant_param.linear.zero_point_vals[0]
                if output_cache_dtype == np.int16:
                    if target_np_dtype == np.int8:
                        new_scale = org_scale * 256.0
                    else:
                        logger.error(
                            f'Unknown combination of cache ({output_cache_dtype}) and target '
                            f'({target_np_dtype}) dtypes.',
                            err=ValueError,
                        )
                elif output_cache_dtype == np.int8:
                    if target_np_dtype == np.int16:
                        new_scale = org_scale / 256.0
                    else:
                        logger.error(
                            f'Unknown combination of cache ({output_cache_dtype}) and target '
                            f'({target_np_dtype}) dtypes.',
                            err=ValueError,
                        )
                else:
                    logger.error(
                        f'Unknown combination of cache ({output_cache_dtype}) and target ({target_np_dtype}) dtypes.',
                        err=ValueError,
                    )
                _append_requantize_op(subgraph, output_idx, name, target_dtype, new_scale, org_zp)
                quantized_model_info['output_scales'][output_idx] = new_scale
                quantized_model_info['output_dtypes'][output_idx] = target_np_dtype

                new_quant_param = nrp_pb2.LinearQuantParamProto()
                new_quant_param.scale_vals.append(new_scale)
                new_quant_param.zero_point_vals.append(org_zp)
                new_quant_param.dimensions.extend([])
                linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, target_nrp_type)

        input_idx = output_idx + num_non_cache_inputs - 1
        inp_name = subgraph.inputs[input_idx]
        if mode == 'dequantize':
            name = inp_name + '_quantize'
            quantized_model_info['input_scales'][input_idx] = None
            quantized_model_info['input_zero_points'][input_idx] = None
            quantized_model_info['input_dtypes'][input_idx] = np.float32
            _prepend_quantize_op(subgraph, input_idx, name)
        else:
            # Force input dtype until BMM (for GQA) and separate bmm else insert requant
            inp_tensor = subgraph.tensor_map[inp_name]

            if config.extra_input['sink_rope'] and ('past_keys' in inp_name or inp_name == 'argument_7.1'):
                requant_name = inp_name.replace(inp_name, f'past_keys_{chunk_idx}_requantize')
                requant_tensor = subgraph.tensor_map[requant_name]
                requant_tensor.shape.CopyFrom(inp_tensor.shape)
                # copy original int16 quant params
                tensor_utils.copy_type_and_quant_param(requant_tensor, inp_tensor)

            inp_tensor.type = target_nrp_type
            inp_tensor.quant_param.linear.CopyFrom(new_quant_param)

            terminate = False
            while True:
                consumers = subgraph_utils.get_consumer_ops(subgraph, inp_name)

                if config.extra_input['sink_rope'] and ('past_keys' in inp_name or inp_name == 'argument_7.1'):
                    for consumer in consumers:
                        # re route consumer input name
                        new_inputs = []
                        for inp in consumer.inputs:
                            if inp == inp_name:
                                new_inputs.append(requant_name)
                            else:
                                new_inputs.append(inp)
                        consumer.inputs[:] = new_inputs
                    # finally reroute past_keys
                    requant_op = subgraph.operators.add()
                    requant_op.type = defs.Op.Requantize
                    requant_op.inputs.append(inp_name)
                    requant_op.outputs.append(requant_name)
                    terminate = True
                else:
                    for consumer in consumers:
                        if consumer.type not in [defs.Op.MatMul, defs.Op.Tile, defs.Op.Reshape]:
                            continue
                        if consumer.type == defs.Op.MatMul:
                            terminate = True
                            break
                        inp_name = consumer.outputs[0]
                        tensor = subgraph.tensor_map[inp_name]
                        tensor.type = target_nrp_type
                        tensor.quant_param.linear.CopyFrom(new_quant_param)

                if terminate:
                    break

            quantized_model_info['input_scales'][input_idx] = new_scale
            quantized_model_info['input_dtypes'][input_idx] = target_np_dtype
    if 'precision_config' in quantized_model_info:
        quantized_model_info['precision_config']['cache_precision'] = target_dtype
    return quantized_model_info


def change_lora_dtype(subgraph, precision_config, quantized_model_info):
    """Changes the given subgraph's lora data type to the specified target data type.

    Args:
        subgraph: The internal representation subgraph.
        precision_config: The precision config.
        quantized_model_info: The quantized model info dictionary.
    """
    logger.debug('Enter change_lora_dtype')
    if quantized_model_info['num_lora_inputs'] == 0:
        return quantized_model_info

    target_dtype = precision_config.get_precision_name(precision_config.lora_precision)[1]
    target_np_dtype = _make_dtype_numpy(target_dtype)

    if target_dtype == 'float':
        target_dtype = 'float32'  # mtk_converter's get_nrp_type does not accept 'float'

    new_nrp_type = type_utils.get_nrp_type(target_dtype)

    # Check if requant is needed
    start_idx = quantized_model_info['num_non_lora_inputs']

    is_fp_respath = (
        precision_config.respath_precision
        == precision_config.embeds_precision
        == precision_config.logits_precision
        == 'FP'
    )
    logger.info(f'[change_lora_dtype] is fp respath: {is_fp_respath}')

    for input_idx in range(start_idx, len(subgraph.inputs)):
        lora_dtype = quantized_model_info['input_dtypes'][input_idx]
        if lora_dtype == target_np_dtype:
            continue

        if lora_dtype == np.float32:
            logger.error('Unable to quantize float lora inputs outside of PTQ flow.', err=ValueError)
        elif target_np_dtype == np.float32:
            mode = 'dequantize'
        else:
            mode = 'requantize'

        org_name = subgraph.inputs[input_idx]
        logger.debug(f'[change_lora_dtype] Org Name: {org_name}')
        org_tensor = subgraph.tensor_map[org_name]
        bmm_op = subgraph_utils.get_consumer_ops(subgraph, org_name)[0]
        bmm_op_out_name = bmm_op.outputs[0]
        logger.debug(f'[change_lora_dtype] BMM op out name: {bmm_op_out_name}')
        is_o_down_lora_b_input = any(n in bmm_op_out_name for n in ['o_lora_B', 'down_lora_B'])
        logger.debug(f'[change_lora_dtype] is lora b for O or down: {is_o_down_lora_b_input}')
        if mode == 'dequantize':
            name = org_name + f'_{mode}'
            new_tensor = subgraph.tensor_map[name]
            new_tensor.type = new_nrp_type
            new_tensor.shape.CopyFrom(org_tensor.shape)
            if not is_fp_respath or (is_fp_respath and not is_o_down_lora_b_input):
                logger.debug('[change_lora_dtype] Adding requant op')
                requant_op = subgraph.operators.add()
                requant_op.type = defs.Op.Quantize
                requant_op.inputs.append(name)
                requant_op.outputs.append(org_name)
            else:
                logger.debug(f'[change_lora_dtype] BMM Op old inputs: {bmm_op.inputs}')
                # change lora_B bmm input and remove original lora B input tensor for o/down_proj
                new_inputs = []
                for inp in bmm_op.inputs:
                    if inp == org_name:
                        new_inputs.append(name)
                    else:
                        new_inputs.append(inp)
                bmm_op.inputs[:] = new_inputs
                del subgraph.tensor_map[org_name]
                logger.debug(f'[change_lora_dtype] BMM Op new inputs: {bmm_op.inputs}')
            subgraph.inputs[input_idx] = name
            quantized_model_info['input_scales'][input_idx] = None
            quantized_model_info['input_zero_points'][input_idx] = None
            quantized_model_info['input_dtypes'][input_idx] = np.float32
        else:
            org_scale = org_tensor.quant_param.linear.scale_vals[0]
            org_zp = org_tensor.quant_param.linear.zero_point_vals[0]
            if lora_dtype == np.int16:
                if target_np_dtype == np.int8:
                    new_scale = org_scale * 256.0
                else:
                    logger.error(
                        f'Unknown combination of lora ({lora_dtype}) and target ({target_np_dtype}) dtypes.',
                        err=ValueError,
                    )
            elif lora_dtype == np.int8:
                if target_np_dtype == np.int16:
                    new_scale = org_scale / 256.0
                else:
                    logger.error(
                        f'Unknown combination of lora ({lora_dtype}) and target ({target_np_dtype}) dtypes.',
                        err=ValueError,
                    )
            else:
                logger.error(
                    f'Unknown combination of lora ({lora_dtype}) and target ({target_np_dtype}) dtypes.', err=ValueError
                )
            new_quant_param = nrp_pb2.LinearQuantParamProto()
            new_quant_param.scale_vals.append(new_scale)
            new_quant_param.zero_point_vals.append(org_zp)
            new_quant_param.dimensions.extend([])
            linear_quant_param_utils.deduce_minmax_from_quant_param(new_quant_param, new_nrp_type)
            org_tensor.type = new_nrp_type
            org_tensor.quant_param.linear.CopyFrom(new_quant_param)
            quantized_model_info['input_scales'][input_idx] = new_scale
            quantized_model_info['input_dtypes'][input_idx] = target_np_dtype
    if 'precision_config' in quantized_model_info:
        quantized_model_info['precision_config']['lora_precision'] = target_dtype
    return quantized_model_info


def export_quantized_model(nrpmod, output_dest_path, np_compat=None):
    """Exports mtk_converter internal representation quantized model into tflite or mlir format.

    Args:
        nrpmod: The mtk_converter internal representation model.
        output_dest_path: The output path to the quantized model. Must end with .tflite or .mlir.
        np_compat (int, optional): The specific neuropilot version op export spec to use during export. Defaults to
            the currently installed mtk_converter version.
    """
    if output_dest_path.endswith('.tflite'):
        _export_nrp_to_tflite(nrpmod, output_dest_path, np_compat)
    elif output_dest_path.endswith('.mlir'):
        _export_nrp_to_mlir(nrpmod, output_dest_path, np_compat)
    else:
        logger.error(f"Expect `output_dest_path` to end with either '.tflite' or '.mlir', but got: {output_dest_path}")


def export_quantized_model_info(quantized_model_info, config, precision_config, output_dest_path):
    """Exports the quantized model metadata information dictionary into a json file.

    Args:
        quantized_model_info (dict): The quantized model metadata dict to export to json.
        config (BaseConfig): The model config.
        precision_config (BasePrecisionConfig): The PrecisionConfig object.
        output_dest_path (str): The output path to the quantized model information.
    """
    logger.debug('Exporting quantized model info json.')
    from ..models.configuration_base import BaseConfig
    from .precision_config import PTQPrecisionConfig

    def _to_serializalbe(config):
        if config is None or isinstance(config, (bool, int, float, str)):
            return config
        if isinstance(config, (list, tuple, set)):
            return [_to_serializalbe(item) for item in config]
        if isinstance(config, dict):
            return {key: _to_serializalbe(value) for key, value in config.items()}
        if hasattr(config, '__dict__'):
            return {
                key: _to_serializalbe(value)
                for key, value in config.__dict__.items()
                if type(key) is not types.MethodType
            }
        return str(config)

    if isinstance(config, BaseConfig):
        config_dict = _to_serializalbe(config)
        logger.debug(f'config_dict: {config_dict}')
        quantized_model_info['model_config'] = config_dict
    else:
        assert isinstance(config, dict)
        quantized_model_info['model_config'] = config

    if isinstance(precision_config, PTQPrecisionConfig):
        config_attrs = [
            x
            for x in dir(precision_config)
            if (not x.startswith('_') and not x.startswith('all_valid') and x not in ['config', 'lora_handler'])
        ]
        config_dict = {
            k: getattr(precision_config, k)
            for k in config_attrs
            if type(getattr(precision_config, k)) is not types.MethodType
        }
        config_dict['llm_unique_precisions'] = list(config_dict['llm_unique_precisions'])
        config_dict = {k: v for k, v in config_dict.items() if type(v) is not set}
        quantized_model_info['precision_config'] = config_dict
    else:
        assert isinstance(precision_config, dict)
        quantized_model_info['precision_config'] = precision_config
    dumpable_quantized_model_info = utils.make_quantized_model_info_dumpable(quantized_model_info)

    with open(output_dest_path, 'w') as f:
        f.write(json.dumps(dumpable_quantized_model_info, indent=4))


def extract_embedder_quantized_model_info(quantized_model_path_or_subgraph, pipeline=None):
    """Extracts information from a Embedder TFLite or MLIR file or subgraph.

    Currently used by Gecko 2 only.

    Args:
        quantized_model_path_or_subgraph (str or nrp_pb2.SubgraphProto): The path to the Embedder quantized model file
            or the subgraph.
        pipeline (Pipeline): The Pipeline object.

    Returns:
        dict: A dictionary containing extracted information.

    Raises:
        ValueError: If the file extension is unknown.
        TypeError: If the input is not a valid path or subgraph.
        AssertionError: If the cache tensor shape is invalid.
    """
    logger.debug('Enter extract_embedder_quantized_model_info')
    if isinstance(quantized_model_path_or_subgraph, str):
        if quantized_model_path_or_subgraph.endswith('.tflite'):
            quantized_model_info_path = quantized_model_path_or_subgraph.replace('.tflite', '_info.json')
        elif quantized_model_path_or_subgraph.endswith('.mlir'):
            quantized_model_info_path = quantized_model_path_or_subgraph.replace('.mlir', '_info.json')
        else:
            logger.error(
                f'Unknown extension: {quantized_model_path_or_subgraph}. Expected quantized model. (tflite/mlir)',
                err=ValueError,
            )
        if os.path.exists(quantized_model_info_path):
            with open(quantized_model_info_path) as f:
                quantized_model_info = json.load(f)
            return _make_quantized_model_info_dtype_numpy(quantized_model_info)
        subgraph = get_subgraph_from_quantized_model(quantized_model_path_or_subgraph)
    else:
        if not isinstance(quantized_model_path_or_subgraph, nrp_pb2.SubgraphProto):
            logger.error(
                f'Expected quantized_model_path_or_subgraph to be quantized model path or nrp subgraph, '
                f'but got type {type(quantized_model_path_or_subgraph)}',
                err=TypeError,
            )
        subgraph = quantized_model_path_or_subgraph

    num_inputs = len(subgraph.inputs)
    num_outputs = len(subgraph.outputs)

    model_type = pipeline.config.l.model_type
    embedding_type = pipeline.precision_config.get_precision_name(pipeline.precision_config.embeds_precision)[1]

    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    batch_size = input_tensor_shape[0]

    num_token = None  # input_tensor_shape[1]

    input_types = []
    for i in range(num_inputs):
        input_types.append(_get_dtype(subgraph, i, 'input'))

    output_types = []
    for i in range(num_outputs):
        output_types.append(_get_dtype(subgraph, i, 'output'))

    quantized_model_info = {
        'batch_size': batch_size,
        't': num_token,
        'model_type': model_type,
        'embedding_dtype': embedding_type,
        'input_dtypes': input_types,
        'output_dtypes': output_types,
    }
    logger.debug(f'embedder quantized_model_info={quantized_model_info}')

    return quantized_model_info


def extract_infini_update_quantized_model_info(quantized_model_path_or_subgraph, pipeline=None):
    """Extracts information from a Infini Update TFLite or MLIR file or subgraph.

    Used by infini attention only.

    Args:
        quantized_model_path_or_subgraph (str or nrp_pb2.SubgraphProto): The path to the Embedder quantized model file
            or the subgraph.
        pipeline (Pipeline): The Pipeline object.

    Returns:
        dict: A dictionary containing extracted information.

    Raises:
        ValueError: If the file extension is unknown.
        TypeError: If the input is not a valid path or subgraph.
        AssertionError: If the cache tensor shape is invalid.
    """
    logger.debug('Enter extract_infini_update_quantized_model_info')
    if isinstance(quantized_model_path_or_subgraph, str):
        if quantized_model_path_or_subgraph.endswith('.tflite'):
            quantized_model_info_path = quantized_model_path_or_subgraph.replace('.tflite', '_info.json')
        elif quantized_model_path_or_subgraph.endswith('.mlir'):
            quantized_model_info_path = quantized_model_path_or_subgraph.replace('.mlir', '_info.json')
        else:
            logger.error(
                f'Unknown extension: {quantized_model_path_or_subgraph}. Expected quantized model. (tflite/mlir)',
                err=ValueError,
            )
        if os.path.exists(quantized_model_info_path):
            with open(quantized_model_info_path) as f:
                quantized_model_info = json.load(f)
            return _make_quantized_model_info_dtype_numpy(quantized_model_info)
        subgraph = get_subgraph_from_quantized_model(quantized_model_path_or_subgraph)
    else:
        if not isinstance(quantized_model_path_or_subgraph, nrp_pb2.SubgraphProto):
            logger.error(
                f'Expected quantized_model_path_or_subgraph to be quantized model path or nrp subgraph, '
                f'but got type {type(quantized_model_path_or_subgraph)}',
                err=TypeError,
            )
        subgraph = quantized_model_path_or_subgraph

    num_inputs = len(subgraph.inputs)
    num_outputs = len(subgraph.outputs)

    model_type = pipeline.config.l.model_type
    infini_update_type = pipeline.precision_config.infini_update_precision

    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    batch_size = input_tensor_shape[0]

    num_token = None  # input_tensor_shape[1]

    input_types = []
    for i in range(num_inputs):
        input_types.append(_get_dtype(subgraph, i, 'input'))

    output_types = []
    for i in range(num_outputs):
        output_types.append(_get_dtype(subgraph, i, 'output'))

    quantized_model_info = {
        'batch_size': batch_size,
        't': num_token,
        'model_type': model_type,
        'infini_update_dtype': infini_update_type,
        'input_dtypes': input_types,
        'output_dtypes': output_types,
    }
    logger.debug(f'infini update quantized_model_info={quantized_model_info}')

    return quantized_model_info


def extract_encoder_quantized_model_info(quantized_model_path_or_subgraph, pipeline=None, chunk_idx=None):
    """Extracts information from an encoder TFLite or MLIR file or subgraph.

    Args:
        quantized_model_path_or_subgraph (str or nrp_pb2.SubgraphProto): The path to the encoder quantized model file
            or the subgraph.
        pipeline (Pipeline): The Pipeline object.
        chunk_idx (int): The index of the current chunk.

    Returns:
        dict: A dictionary containing extracted information.

    Raises:
        ValueError: If the file extension is unknown.
        TypeError: If the input is not a valid path or subgraph.
    """
    logger.debug('Enter extract_encoder_quantized_model_info')
    if isinstance(quantized_model_path_or_subgraph, str):
        if quantized_model_path_or_subgraph.endswith('.tflite'):
            chunk_id = quantized_model_path_or_subgraph.replace('.tflite', '').rsplit('_', 1)[-1]
            quantized_model_info_path = quantized_model_path_or_subgraph.replace(
                f'{chunk_id}.tflite', f'info_{chunk_id}.json'
            )
        elif quantized_model_path_or_subgraph.endswith('.mlir'):
            chunk_id = quantized_model_path_or_subgraph.replace('.mlir', '').rsplit('_', 1)[-1]
            quantized_model_info_path = quantized_model_path_or_subgraph.replace(
                f'{chunk_id}.mlir', f'info_{chunk_id}.json'
            )
        else:
            logger.error(
                f'Unknown extension: {quantized_model_path_or_subgraph}. Expected quantized model. (tflite/mlir)',
                err=ValueError,
            )
        if os.path.exists(quantized_model_info_path):
            with open(quantized_model_info_path) as f:
                quantized_model_info = json.load(f)
            return _make_quantized_model_info_dtype_numpy(quantized_model_info)
        subgraph = get_subgraph_from_quantized_model(quantized_model_path_or_subgraph)
    else:
        if not isinstance(quantized_model_path_or_subgraph, nrp_pb2.SubgraphProto):
            logger.error(
                f'Expected quantized_model_path_or_subgraph to be quantized model path or nrp subgraph, '
                f'but got type {type(quantized_model_path_or_subgraph)}',
                err=TypeError,
            )
        subgraph = quantized_model_path_or_subgraph

    if pipeline is None or chunk_idx is None:
        logger.error('`pipeline` and `chunk_idx` cannot be None when quantized model info json does not already exist.')

    num_inputs = len(subgraph.inputs)
    num_outputs = len(subgraph.outputs)

    projector = chunk_idx == pipeline.num_encoder_layers and pipeline.has_projector()
    model_type = pipeline.config.p2.model_type if projector else pipeline.config.e.model_type
    num_layers = 0 if projector else pipeline.encoder_layers_per_chunk[chunk_idx]
    layer_ids = [] if projector else pipeline.encoder_layer_ids[chunk_idx]
    num_lora_inputs = 0 if projector else len(pipeline.lora_handler.encoder_lora_inputs[chunk_idx][0])
    num_non_lora_inputs = num_inputs - num_lora_inputs
    lora_start_layer_idx = pipeline.lora_handler.global_llm_start_idx
    lora_end_layer_idx = pipeline.lora_handler.global_llm_end_idx

    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    output_tensor = subgraph.tensor_map[subgraph.outputs[0]]
    output_tensor_shape = tensor_utils.get_shape(output_tensor).as_list()
    batch_size = input_tensor_shape[0]
    height = None
    width = None
    num_channels = None
    feature_size_in = None
    feature_size_out = None
    hidden_size = None
    if 0 in layer_ids:
        if len(input_tensor_shape) == 4:  # Typical encoder cases
            height = input_tensor_shape[2]
            width = input_tensor_shape[3]
            num_channels = input_tensor_shape[1]
            feature_size_out = output_tensor_shape[1]
            hidden_size = output_tensor_shape[2]
        elif len(input_tensor_shape) == 2:  # Qwen2-VL ViT case
            num_channels = input_tensor_shape[1]
            feature_size_out = output_tensor_shape[0]
            hidden_size = output_tensor_shape[1]
    else:
        if len(input_tensor_shape) == 3:  # Typical encoder cases
            feature_size_in = input_tensor_shape[1]
            feature_size_out = output_tensor_shape[1]
            hidden_size = input_tensor_shape[2]
        elif len(input_tensor_shape) == 2:  # Qwen2-VL ViT case
            feature_size_in = input_tensor_shape[0]
            feature_size_out = output_tensor_shape[0]
            hidden_size = input_tensor_shape[1]

    input_scales = []
    input_zero_points = []
    input_types = []
    for i in range(num_inputs):
        input_scales.append(extract_quantized_model_qparams(subgraph, i, 'input'))
        input_zero_points.append(extract_quantized_model_qparams(subgraph, i, 'input', 'zp'))
        input_types.append(_get_dtype(subgraph, i, 'input'))

    output_scales = []
    output_zero_points = []
    output_types = []
    for i in range(num_outputs):
        output_scales.append(extract_quantized_model_qparams(subgraph, i, 'output'))
        output_zero_points.append(extract_quantized_model_qparams(subgraph, i, 'output', 'zp'))
        output_types.append(_get_dtype(subgraph, i, 'output'))

    # lora details
    if num_lora_inputs > 0:
        if num_lora_inputs % 2 != 0:
            logger.error(f'Expected an even number of LoRA inputs but detected {num_lora_inputs}')
        lora_a = subgraph.inputs[num_non_lora_inputs]
        lora_b = subgraph.inputs[num_non_lora_inputs + 1]
        lora_a_rank = tensor_utils.get_shape(subgraph.tensor_map[lora_a]).as_list()[1]
        lora_b_rank = tensor_utils.get_shape(subgraph.tensor_map[lora_b]).as_list()[2]
        if lora_a_rank != lora_b_rank:
            logger.error(
                f'Lora ranks mismatch! Found lora A tensor with rank {lora_a_rank} and lora B tensor with rank '
                f'{lora_b_rank}'
            )
    else:
        lora_a_rank = None

    quantized_model_info = {
        'batch_size': batch_size,
        'h': height,
        'w': width,
        'num_channels': num_channels,
        'fi': feature_size_in,
        'fo': feature_size_out,
        'model_type': model_type,
        'projector': projector,
        'num_layers': num_layers,
        'layer_ids': layer_ids,
        'hidden_size': hidden_size,
        'lora_rank': lora_a_rank,
        'lora_start_layer_idx': lora_start_layer_idx,
        'lora_end_layer_idx': lora_end_layer_idx,
        'num_lora_inputs': num_lora_inputs,
        'num_non_lora_inputs': num_non_lora_inputs,
        'input_scales': input_scales,
        'output_scales': output_scales,
        'input_zero_points': input_zero_points,
        'output_zero_points': output_zero_points,
        'input_dtypes': input_types,
        'output_dtypes': output_types,
    }
    logger.debug(f'quantized_model_info={quantized_model_info}')

    return quantized_model_info


def extract_llm_quantized_model_info(quantized_model_path_or_subgraph, pipeline=None, chunk_idx=None):
    """Extracts information from a LLM TFLite or MLIR file or subgraph.

    Args:
        quantized_model_path_or_subgraph (str or nrp_pb2.SubgraphProto): The path to the LLM quantized model file or the
            subgraph.
        pipeline (Pipeline): The Pipeline object.
        chunk_idx (int): The index of the current chunk.

    Returns:
        dict: A dictionary containing extracted information.

    Raises:
        ValueError: If the file extension is unknown.
        TypeError: If the input is not a valid path or subgraph.
        AssertionError: If the cache tensor shape is invalid.
    """
    logger.debug('Enter extract_llm_quantized_model_info')
    if isinstance(quantized_model_path_or_subgraph, str):
        if quantized_model_path_or_subgraph.endswith('.tflite'):
            chunk_id = quantized_model_path_or_subgraph.replace('.tflite', '').rsplit('_', 1)[-1]
            quantized_model_info_path = quantized_model_path_or_subgraph.replace(
                f'{chunk_id}.tflite', f'info_{chunk_id}.json'
            )
        elif quantized_model_path_or_subgraph.endswith('.mlir'):
            chunk_id = quantized_model_path_or_subgraph.replace('.mlir', '').rsplit('_', 1)[-1]
            quantized_model_info_path = quantized_model_path_or_subgraph.replace(
                f'{chunk_id}.mlir', f'info_{chunk_id}.json'
            )
        else:
            logger.error(
                f'Unknown extension: {quantized_model_path_or_subgraph}. Expected quantized model. (tflite/mlir)',
                err=ValueError,
            )
        if os.path.exists(quantized_model_info_path):
            with open(quantized_model_info_path) as f:
                quantized_model_info = json.load(f)
            return _make_quantized_model_info_dtype_numpy(quantized_model_info)
        subgraph = get_subgraph_from_quantized_model(quantized_model_path_or_subgraph)
    else:
        if not isinstance(quantized_model_path_or_subgraph, nrp_pb2.SubgraphProto):
            logger.error(
                f'Expected quantized_model_path_or_subgraph to be quantized model path or nrp subgraph, '
                f'but got type {type(quantized_model_path_or_subgraph)}',
                err=TypeError,
            )
        subgraph = quantized_model_path_or_subgraph

    if pipeline is None or chunk_idx is None:
        logger.error('`pipeline` and `chunk_idx` cannot be None when quantized model info json does not already exist.')

    num_inputs = len(subgraph.inputs)
    num_outputs = len(subgraph.outputs)

    infini_attention = getattr(pipeline.config.l, 'infini_attention', False)
    use_split_mask = getattr(pipeline.config.l, 'use_split_mask', False)
    tail = chunk_idx == pipeline.num_decoder_layers
    tail_type = ('tail' if not pipeline.has_custom_tail() else pipeline.config.t.model_type) if tail else None
    model_type = pipeline.config.t.model_type if tail and pipeline.has_custom_tail() else pipeline.config.l.model_type
    num_layers = 0 if tail else pipeline.llm_layers_per_chunk[chunk_idx]
    layer_ids = [] if tail else pipeline.llm_layer_ids[chunk_idx]
    num_lora_inputs = 0 if tail else len(pipeline.lora_handler.llm_lora_inputs[chunk_idx][0])
    num_non_lora_inputs = num_inputs - num_lora_inputs
    lora_start_layer_idx = pipeline.lora_handler.global_llm_start_idx
    lora_end_layer_idx = pipeline.lora_handler.global_llm_end_idx
    num_cache_inputs = 0 if (tail and model_type != 'eagle') else 2 + 2 * int(infini_attention)
    num_other_masks_inputs = 0 if tail else int(infini_attention) + int(use_split_mask)
    num_non_cache_inputs = num_non_lora_inputs - num_cache_inputs

    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    batch_size = input_tensor_shape[0]
    hidden_size = input_tensor_shape[2]

    num_token = None  # input_tensor_shape[1]
    cache_size = None
    head_dim = int(
        getattr(pipeline.config.l, 'head_dim', pipeline.config.l.hidden_size // pipeline.config.l.num_attention_heads)
    )

    # FIXME: Will fail for sparse attention heads (baichuan3)
    num_attention_heads = [pipeline.config.l.num_attention_heads for _ in range(num_layers)]

    num_key_value_heads = [pipeline.config.l.num_key_value_heads for _ in range(num_layers)]

    input_scales = []
    input_zero_points = []
    input_types = []
    for i in range(num_inputs):
        input_scales.append(extract_quantized_model_qparams(subgraph, i, 'input'))
        input_zero_points.append(extract_quantized_model_qparams(subgraph, i, 'input', 'zp'))
        input_types.append(_get_dtype(subgraph, i, 'input'))

    output_scales = []
    output_zero_points = []
    output_types = []
    for i in range(num_outputs):
        output_scales.append(extract_quantized_model_qparams(subgraph, i, 'output'))
        output_zero_points.append(extract_quantized_model_qparams(subgraph, i, 'output', 'zp'))
        output_types.append(_get_dtype(subgraph, i, 'output'))

    # lora details
    if num_lora_inputs > 0:
        if num_lora_inputs % 2 != 0:
            logger.error(f'Expected an even number of LoRA inputs but detected {num_lora_inputs}')
        lora_a = subgraph.inputs[num_non_lora_inputs]
        lora_b = subgraph.inputs[num_non_lora_inputs + 1]
        lora_a_rank = tensor_utils.get_shape(subgraph.tensor_map[lora_a]).as_list()[1]
        lora_b_rank = tensor_utils.get_shape(subgraph.tensor_map[lora_b]).as_list()[2]
        if lora_a_rank != lora_b_rank:
            logger.error(
                f'Lora ranks mismatch! Found lora A tensor with rank {lora_a_rank} and lora B tensor with rank '
                f'{lora_b_rank}'
            )
    else:
        lora_a_rank = None

    quantized_model_info = {
        'batch_size': batch_size,
        't': num_token,
        'c': cache_size,
        'model_type': model_type,
        'tail': tail_type,
        'num_layers': num_layers,
        'layer_ids': layer_ids,
        'hidden_size': hidden_size,
        'head_dim': head_dim,
        'num_attention_heads': num_attention_heads,
        'num_key_value_heads': num_key_value_heads,
        'infini_attention': infini_attention,
        'use_split_mask': use_split_mask,
        'lora_rank': lora_a_rank,
        'lora_start_layer_idx': lora_start_layer_idx,
        'lora_end_layer_idx': lora_end_layer_idx,
        'num_non_cache_inputs': num_non_cache_inputs,
        'num_other_masks_inputs': num_other_masks_inputs,
        'num_cache_inputs': num_cache_inputs,
        'num_non_lora_inputs': num_non_lora_inputs,
        'num_lora_inputs': num_lora_inputs,
        'input_scales': input_scales,
        'output_scales': output_scales,
        'input_zero_points': input_zero_points,
        'output_zero_points': output_zero_points,
        'input_dtypes': input_types,
        'output_dtypes': output_types,
    }
    logger.debug(f'quantized_model_info={quantized_model_info}')

    return quantized_model_info


def extract_quantized_model_qparams(quantized_model_path_or_subgraph, index, mode, param='scale'):
    """Extracts quantization parameters from a TFLite or MLIR file or subgraph.

    Args:
        quantized_model_path_or_subgraph (str or nrp_pb2.SubgraphProto): The path to the quantized model file or the
            subgraph.
        index (int): The index of the input or output tensor.
        mode (str): The mode, either 'input' or 'output'.
        param (str, optional): The parameter to extract, either 'scale' or 'minmax'. Default is 'scale'.

    Returns:
        float or tuple: The extracted quantization parameter.

    Raises:
        TypeError: If the input is not a valid path or subgraph.
        ValueError: If the param or mode is invalid.
        IndexError: If the index is out of range.
    """
    logger.debug(f'Enter extract_quantized_model_qparams to get {mode} index {index} {param}')
    if isinstance(quantized_model_path_or_subgraph, str):
        subgraph = get_subgraph_from_quantized_model(quantized_model_path_or_subgraph)
    else:
        if not isinstance(quantized_model_path_or_subgraph, nrp_pb2.SubgraphProto):
            logger.error(
                f'Expected quantized_model_path_or_subgraph to be quantized model path or nrp subgraph, '
                f'but got type {type(quantized_model_path_or_subgraph)}',
                err=TypeError,
            )
        subgraph = quantized_model_path_or_subgraph

    if param not in ['scale', 'zp', 'minmax']:
        logger.error('param must be one of: scale, zp, minmax', err=ValueError)
    if mode not in ['input', 'output']:
        logger.error('mode must be one of: input, output', err=ValueError)

    if mode == 'input':
        if index >= len(subgraph.inputs):
            logger.error(f'index is {index} but quantized model only has {len(subgraph.inputs)} inputs', err=IndexError)
        name = subgraph.inputs[index]
    elif mode == 'output':
        if index >= len(subgraph.outputs):
            logger.error(
                f'index is {index} but quantized model only has {len(subgraph.outputs)} outputs', err=IndexError
            )
        name = subgraph.outputs[index]

    if param == 'scale':
        if type_utils.get_numpy_type(subgraph.tensor_map[name].type) in [np.int32, np.float32]:
            return None
        return subgraph.tensor_map[name].quant_param.linear.scale_vals[0]
    if param == 'zp':
        if type_utils.get_numpy_type(subgraph.tensor_map[name].type) in [np.int32, np.float32]:
            return None
        return subgraph.tensor_map[name].quant_param.linear.zero_point_vals[0]
    if param == 'minmax':
        if type_utils.get_numpy_type(subgraph.tensor_map[name].type) in [np.int32, np.float32]:
            return (None, None)
        min_val = subgraph.tensor_map[name].quant_param.linear.min_vals[0]
        max_val = subgraph.tensor_map[name].quant_param.linear.max_vals[0]
        return (min_val, max_val)
    return None


def finalize_embedder_quantized_model(quantized_model_path, config, pipeline):
    """Finalizes the Embedder TFLite or MLIR model.

    By performing various operations such as quantization parameter alignment,
    inserting requantization operations, and exporting the model.

    Args:
        quantized_model_path (str): The path to the LLM TFLite or MLIR file.
        config (object): The configuration object.
        pipeline (object): The pipeline object.

    Raises:
        AssertionError: If LoRA inputs are not provided when required.
    """
    ext = quantized_model_path.rsplit('.', 1)[-1]
    logger.info('Finalizing Embedder quantized model')

    subgraph, _ = get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=True)
    quantized_model_info = extract_embedder_quantized_model_info(subgraph, pipeline=pipeline)

    export_quantized_model_info(
        quantized_model_info,
        config,
        pipeline.precision_config,
        quantized_model_path.replace(f'.{ext}', '_info.json'),
    )


def finalize_infini_update_quantized_model(quantized_model_path, config, pipeline):
    """Finalizes the Infini TFLite or MLIR model.

    By performing various operations such as quantization parameter alignment,
    inserting requantization operations, and exporting the model.

    Args:
        quantized_model_path (str): The path to the LLM TFLite or MLIR file.
        config (object): The configuration object.
        pipeline (object): The pipeline object.

    """
    ext = quantized_model_path.rsplit('.', 1)[-1]
    logger.info('Finalizing Infini Update quantized model')

    subgraph, _ = get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=True)
    quantized_model_info = extract_infini_update_quantized_model_info(subgraph, pipeline=pipeline)

    export_quantized_model_info(
        quantized_model_info,
        config,
        pipeline.precision_config,
        quantized_model_path.replace(f'.{ext}', '_info.json'),
    )


def finalize_encoder_quantized_model(
    quantized_model_path, config, pipeline, default_precision, chunk_idx=None, np_compat=None
):
    """Finalizes the encoder TFLite or MLIR model.

    By performing various operations such as quantization parameter alignment,
    inserting requantization operations, and exporting the model.

    Args:
        quantized_model_path (str): The path to the encoder TFLite or MLIR file.
        config (BaseEncoderConfig): The Encoder configuration object.
        pipeline (Pipeline): The pipeline object.
        default_precision (str): The default precision field of the mtk_converter precision hint config.
        chunk_idx (int, optional): The chunk index. Default is None.
        np_compat (int, optional): Whether to enable backward-compatibility mode for a specific neuropilot version.

    Raises:
        AssertionError: If LoRA inputs are not provided when required.
    """
    ext = quantized_model_path.rsplit('.', 1)[-1]
    logger.info('Finalizing encoder quantized model')

    subgraph, nrpmod = get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=True)
    quantized_model_info = extract_encoder_quantized_model_info(subgraph, pipeline=pipeline, chunk_idx=chunk_idx)
    precision_config = pipeline.precision_config
    if quantized_model_info['model_type'] == 'whisper' and (
        pipeline.config.e.num_hidden_layers - 1 in quantized_model_info['layer_ids']
    ):
        _append_dequantize_op(subgraph, 0, 'encoder_output_dequantize')
    if quantized_model_info['num_lora_inputs'] > 0:
        lora_precision = precision_config.get_precision_name(precision_config.lora_precision)[1]
        change_lora_dtype(subgraph, precision_config, quantized_model_info)
        if chunk_idx is None:
            logger.error('`chunk_idx` should not be None for `change_cache_dtype`', err=ValueError)
        utils.create_lora_bin_for_cmdline(
            quantized_model_path,
            chunk_idx,
            pipeline,
            lora_precision,
            subgraph=subgraph,
            encoder=True,
        )

    subgraph_utils.remove_unused_ops_and_tensors(subgraph)
    subgraph_utils.ensure_topological_op_order(subgraph)

    export_quantized_model(nrpmod, quantized_model_path, np_compat)
    export_quantized_model_info(
        quantized_model_info,
        config,
        pipeline.precision_config,
        quantized_model_path.replace(f'{chunk_idx}.{ext}', f'info_{chunk_idx}.json'),
    )


def finalize_projector_quantized_model(
    quantized_model_path, config, pipeline, default_precision, chunk_idx=None, np_compat=None, decoder_model_info=None
):
    """Finalizes the encoder TFLite or MLIR model.

    By performing various operations such as quantization parameter alignment,
    inserting requantization operations, and exporting the model.

    Args:
        quantized_model_path (str): The path to the encoder TFLite or MLIR file.
        config (BaseEncoderConfig): The Encoder configuration object.
        pipeline (Pipeline): The pipeline object.
        default_precision (str): The default precision field of the mtk_converter precision hint config.
        chunk_idx (int, optional): The chunk index. Default is None.
        np_compat (int, optional): Whether to enable backward-compatibility mode for a specific neuropilot version.
        decoder_model_info (dict, optional): The decoder model info for alignment. Default is None.

    Raises:
        AssertionError: If LoRA inputs are not provided when required.
    """
    ext = quantized_model_path.rsplit('.', 1)[-1]
    logger.info('Finalizing projector quantized model')

    subgraph, nrpmod = get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=True)
    quantized_model_info = extract_encoder_quantized_model_info(subgraph, pipeline=pipeline, chunk_idx=chunk_idx)
    precision_config = pipeline.precision_config

    logits_dtype = quantized_model_info['output_dtypes'][0]
    # align the datatype and quantized scale of the encoder output with the decoder input
    if decoder_model_info is not None:
        decoder_dtype = decoder_model_info['input_dtypes'][0]
        embedding_scale = decoder_model_info['input_scales'][0]
        embedding_zp = decoder_model_info['input_zero_points'][0]
        if decoder_dtype == 'float32':
            if logits_dtype != _make_dtype_numpy('float'):
                logger.debug('[Uniform Quant] Appending dequantize OP to the output of encoder.')
                _append_dequantize_op(subgraph, 0, 'encoder_output_dequantize')
                quantized_model_info['output_dtypes'][0] = np.float32
                quantized_model_info['output_scales'][0] = None
                quantized_model_info['output_zero_points'][0] = None
                if len(subgraph.outputs) > 1:
                    for i in range(1, len(subgraph.outputs)):
                        _append_dequantize_op(subgraph, i, f'encoder_output_{i}_dequantize')
                        quantized_model_info['output_dtypes'][i] = np.float32
                        quantized_model_info['output_scales'][i] = None
                        quantized_model_info['output_zero_points'][i] = None
        else:
            logger.debug('[Uniform Quant] Appending requantize OP to the output of encoder.')
            dtype = type_utils.TYPE_NAME_MAPPINGS[type_utils.get_nrp_type(decoder_dtype)]
            _append_requantize_op(subgraph, 0, 'encoder_output_quantize', dtype, embedding_scale, embedding_zp)
            quantized_model_info['output_dtypes'][0] = _make_dtype_numpy(decoder_dtype)
            quantized_model_info['output_scales'][0] = embedding_scale
            quantized_model_info['output_zero_points'][0] = embedding_zp

    subgraph_utils.remove_unused_ops_and_tensors(subgraph)
    subgraph_utils.ensure_topological_op_order(subgraph)

    export_quantized_model(nrpmod, quantized_model_path, np_compat)
    export_quantized_model_info(
        quantized_model_info,
        config,
        precision_config,
        quantized_model_path.replace(f'{chunk_idx}.{ext}', f'info_{chunk_idx}.json'),
    )


def finalize_llm_quantized_model(
    quantized_model_path, config, pipeline, default_precision, chunk_idx=None, np_compat=None
):
    """Finalizes the LLM TFLite or MLIR model.

    By performing various operations such as quantization parameter alignment,
    inserting requantization operations, and exporting the model.

    Args:
        quantized_model_path (str): The path to the LLM TFLite or MLIR file.
        config (object): The configuration object.
        pipeline (object): The pipeline object.
        default_precision (str): The default precision field of the mtk_converter precision hint config.
        chunk_idx (int, optional): The chunk index. Default is None.
        np_compat (int, optional): Whether to enable backward-compatibility mode for a specific neuropilot version.

    Raises:
        AssertionError: If LoRA inputs are not provided when required.
    """
    ext = quantized_model_path.rsplit('.', 1)[-1]
    logger.info('Finalizing LLM quantized model')

    subgraph, nrpmod = get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=True)
    quantized_model_info = extract_llm_quantized_model_info(subgraph, pipeline=pipeline, chunk_idx=chunk_idx)
    precision_config = pipeline.precision_config

    if chunk_idx == 0:
        embed_precision = precision_config.embeds_precision
        quantized_model_info = change_embed_dtype(
            subgraph, precision_config.get_precision_name(embed_precision)[1], quantized_model_info
        )

    if quantized_model_info['tail'] is None:
        cache_dtype = quantized_model_info['output_dtypes'][-1]
        if cache_dtype != np.float32:
            _intra_chunk_qparam_alignment(subgraph, config, quantized_model_info)
            cache_precision = precision_config.cache_precision
            # let change cache dtype run all the time as a form of sanity checking
            if chunk_idx is None:
                logger.error('`chunk_idx` should not be None for `change_cache_dtype`', err=ValueError)
            quantized_model_info = change_cache_dtype(
                subgraph,
                config,
                chunk_idx,
                precision_config.get_precision_name(cache_precision)[1],
                quantized_model_info,
            )
        if quantized_model_info['num_lora_inputs'] > 0:
            if chunk_idx is None:
                logger.error('`chunk_idx` should not be None for `change_lora_dtype`', err=ValueError)
            lora_precision = precision_config.get_precision_name(precision_config.lora_precision)[1]
            quantized_model_info = change_lora_dtype(subgraph, precision_config, quantized_model_info)

            utils.create_lora_bin_for_cmdline(
                quantized_model_path,
                chunk_idx,
                pipeline,
                lora_precision,
                subgraph=subgraph,
            )
        # NOTE: Remove this in future?
        if pipeline.use_single_bmm_attention:
            _remove_rotemb_quantize(subgraph)
    else:
        logits_dtype = quantized_model_info['output_dtypes'][0]  # <class np dtype>
        logits_precision_name = precision_config.get_precision_name(precision_config.logits_precision)[1]
        # FIXME: Currently cannot handle logits precision < int16
        if logits_dtype != _make_dtype_numpy(logits_precision_name):
            if np.dtype(logits_dtype) == np.float32:
                logger.warning(
                    'Found float logits dtype. Setting logits_precision to float as quant params cannot be determined '
                    'from float output'
                )
                pipeline.precision_config.logits_precision = 'FP'
            else:
                scale = extract_quantized_model_qparams(subgraph, 0, 'output', 'scale')
                zp = extract_quantized_model_qparams(subgraph, 0, 'output', 'zp')
                _append_requantize_op(subgraph, 0, 'logits_requantize', logits_precision_name, scale, zp)
    subgraph_utils.remove_unused_ops_and_tensors(subgraph)
    subgraph_utils.ensure_topological_op_order(subgraph)

    export_quantized_model(nrpmod, quantized_model_path, np_compat)
    export_quantized_model_info(
        quantized_model_info,
        config,
        pipeline.precision_config,
        quantized_model_path.replace(f'{chunk_idx}.{ext}', f'info_{chunk_idx}.json'),
    )


def get_subgraph_from_quantized_model(quantized_model_path, return_nrpmodel=False):
    """Gets the subgraph from a TFLite or MLIR file.

    Args:
        quantized_model_path (str): The path to the TFLite or MLIR file.
        return_nrpmodel (bool, optional): Whether to return the NRP model. Default is False.

    Returns:
        object: The subgraph or a tuple of the subgraph and the NRP model.

    Raises:
        FileNotFoundError: If the TFLite or MLIR file is not found.
    """
    logger.debug('Enter get_subgraph_from_quantized_model')
    is_tflite = quantized_model_path.endswith('.tflite')
    if not os.path.exists(quantized_model_path):
        logger.error(f'{"TFLite" if is_tflite else "MLIR"} not found: {quantized_model_path}.', err=FileNotFoundError)

    with open(quantized_model_path, 'rb') as quantized_model_file:
        quantized_model = quantized_model_file.read()

    if is_tflite:
        model = tflite_importer.import_from_model_buffer(quantized_model)
    else:
        model = mlir_importer.import_from_model_buffer(quantized_model)
    subgraph = model.subgraphs[0]
    if return_nrpmodel:
        return subgraph, model
    return subgraph


def has_separate_tails(quantized_model_folder):
    """Checks if the quantized models in the folder have separate tails.

    Args:
        quantized_model_folder (str): The folder containing the quantized models.

    Returns:
        tuple: A tuple containing a boolean indicating if separate tails exist and the type of the last layer.

    Raises:
        FileNotFoundError: If the quantized model folder does not exist.
        TypeError: If the quantized model folder is not a directory.
    """
    logger.debug('Enter has_separate_tails')
    if not os.path.exists(quantized_model_folder):
        logger.error(f'{quantized_model_folder} does not exist.', err=FileNotFoundError)
    if not os.path.isdir(quantized_model_folder):
        logger.error(f'Expected dir for `quantized_model_folder` but got {quantized_model_folder}', err=TypeError)

    all_files = utils.get_sorted_path_list(quantized_model_folder, ['.tflite', '.mlir'])
    if len(all_files) == 1:
        return False, None

    last_chunk = all_files[-1]
    last_chunk_tail_type = extract_llm_quantized_model_info(last_chunk)

    if len(all_files) == 2:
        if last_chunk_tail_type == 'tail':
            return True, None
        return False, None
    second_last_chunk = all_files[-2]
    second_last_tail_type = extract_llm_quantized_model_info(second_last_chunk)

    if second_last_tail_type == 'tail':
        return True, last_chunk_tail_type
    if last_chunk_tail_type == 'tail':
        return True, None
    return False, None


def load_quantized_models(folder):
    """Loads the TFLite or MLIR models from the specified folder.

    Args:
        folder (str): The folder containing the TFLite or MLIR models.

    Returns:
        list: A list of loaded models.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        TypeError: If the specified folder is not a directory.
    """
    # TODO: enable parallel loading of tflite models
    logger.debug('Enter load_quantized_models')
    from mtk_converter import MlirExecutor, TFLiteExecutor

    if not os.path.exists(folder):
        logger.error(f'{folder} does not exist.', err=FileNotFoundError)
    if not os.path.isdir(folder):
        logger.error(f'Expected directory for folder but got: {folder}', err=TypeError)

    if len(os.listdir(folder)) == 2:
        # Single tflite/mlir, does not end with chunk idx
        quantized_model_files = [os.path.join(folder, x) for x in os.listdir(folder) if not x.endswith('.json')]
    else:
        quantized_model_files = utils.get_sorted_path_list(folder, ext=['.tflite', '.mlir'])

    models = []
    for f in quantized_model_files:
        logger.info(f'Loading {f}')
        if f.endswith('.tflite'):
            models.append(TFLiteExecutor(f, simulate_fp16=True))
        else:
            assert f.endswith('.mlir')
            models.append(MlirExecutor(f, simulate_fp16=True))
    return models


def merge_quantized_model_infos(quantized_model_infos):
    """Merges per-layer quantized model info dicts into one chunked quantized model info dictionary.

    Args:
        quantized_model_infos (list): A list of quantized model info dictionaries to merge.

    Returns:
        dict: A dict containing the merged quantized model info of the model chunk.
    """
    if not isinstance(quantized_model_infos, list):
        logger.error(f'Expected list but got {type(quantized_model_infos)}', err=TypeError)

    if 't' in quantized_model_infos[0] and 'c' in quantized_model_infos[0]:
        encoder = False
    elif 'h' in quantized_model_infos[0] and 'w' in quantized_model_infos[0]:
        encoder = True
    else:
        logger.error(f'Unrecognized quantized model info: {quantized_model_infos}.')

    if len(quantized_model_infos) == 1:
        return quantized_model_infos[0]

    chunk_quantized_model_info = quantized_model_infos[0]
    num_lora_inputs = quantized_model_infos[0]['num_lora_inputs']

    cache_evict = chunk_quantized_model_info['model_config'].get('cache_evict', {'method': ''})
    if isinstance(cache_evict, str):
        is_cache_evict = cache_evict in ['LocalSnapKV', 'GlobalSnapKV']
    else:
        is_cache_evict = cache_evict['method'] in ['LocalSnapKV', 'GlobalSnapKV']

    if num_lora_inputs > 0:
        lora_input_scales = quantized_model_infos[0]['input_scales'][-num_lora_inputs:]
        lora_input_zps = quantized_model_infos[0]['input_zero_points'][-num_lora_inputs:]
        lora_input_dtypes = quantized_model_infos[0]['input_dtypes'][-num_lora_inputs:]
        chunk_quantized_model_info['input_scales'] = chunk_quantized_model_info['input_scales'][:-num_lora_inputs]
        chunk_quantized_model_info['input_zero_points'] = chunk_quantized_model_info['input_zero_points'][
            :-num_lora_inputs
        ]
        chunk_quantized_model_info['input_dtypes'] = chunk_quantized_model_info['input_dtypes'][:-num_lora_inputs]
    else:
        lora_input_scales = []
        lora_input_zps = []
        lora_input_dtypes = []

    # FIXME: Temporary, handle split mask, should fix during input/output refactor
    if not encoder:
        use_split_mask = quantized_model_infos[0]['model_config']['use_split_mask']
        if use_split_mask:
            # since split mask is reordered during shape fixing, thus need to reorder quantized info
            # Note: lora scales are popped before, so split mask will be the last one
            split_mask_scale = chunk_quantized_model_info['input_scales'].pop(-1)
            split_mask_zp = chunk_quantized_model_info['input_zero_points'].pop(-1)
            split_mask_dtype = chunk_quantized_model_info['input_dtypes'].pop(-1)

            # FIXME: only works for common (llama-based) models
            chunk_quantized_model_info['input_scales'].insert(2, split_mask_scale)
            chunk_quantized_model_info['input_zero_points'].insert(2, split_mask_zp)
            chunk_quantized_model_info['input_dtypes'].insert(2, split_mask_dtype)

    for i in range(1, len(quantized_model_infos)):
        chunk_quantized_model_info['num_layers'] += quantized_model_infos[i]['num_layers']
        chunk_quantized_model_info['layer_ids'] += quantized_model_infos[i]['layer_ids']
        num_lora_inputs = quantized_model_infos[i]['num_lora_inputs']
        chunk_quantized_model_info['num_lora_inputs'] += num_lora_inputs
        if not encoder:
            num_non_cache_inputs = quantized_model_infos[i]['num_non_cache_inputs']
            num_cache_inputs = quantized_model_infos[i]['num_cache_inputs']
            num_other_masks_inputs = quantized_model_infos[i].get('num_other_masks_inputs', 0)

            # FIXME: currently other masks are behind cache, so need to handle like this
            cache_inputs_start_idx = num_non_cache_inputs - num_other_masks_inputs
            cache_inputs_end_idx = cache_inputs_start_idx + num_cache_inputs
            if quantized_model_infos[i]['model_type'] == 'whisper_decoder':
                cache_inputs_start_idx -= 2

            chunk_quantized_model_info['num_attention_heads'] += quantized_model_infos[i]['num_attention_heads']
            chunk_quantized_model_info['num_key_value_heads'] += quantized_model_infos[i]['num_key_value_heads']
            chunk_quantized_model_info['num_cache_inputs'] += quantized_model_infos[i]['num_cache_inputs']
            chunk_quantized_model_info['num_non_lora_inputs'] += quantized_model_infos[i]['num_cache_inputs']
            chunk_quantized_model_info['input_scales'] += quantized_model_infos[i]['input_scales'][
                cache_inputs_start_idx:cache_inputs_end_idx
            ]
            chunk_quantized_model_info['input_zero_points'] += quantized_model_infos[i]['input_zero_points'][
                cache_inputs_start_idx:cache_inputs_end_idx
            ]
            chunk_quantized_model_info['input_dtypes'] += quantized_model_infos[i]['input_dtypes'][
                cache_inputs_start_idx:cache_inputs_end_idx
            ]
            if quantized_model_infos[i]['tail'] is not None:
                # tail layer -> replace output info of first decoder layer in current chunk with tail output
                # current assumption: tail would always have one output only
                chunk_quantized_model_info['output_scales'][0] = quantized_model_infos[i]['output_scales'][0]
                chunk_quantized_model_info['output_zero_points'][0] = quantized_model_infos[i]['output_zero_points'][0]
                chunk_quantized_model_info['output_dtypes'][0] = quantized_model_infos[i]['output_dtypes'][0]
            else:
                if i == len(quantized_model_infos) - 1:
                    # account for multi chunk case -> replace first output info of first decoder layer in current chunk
                    # with first output info of last decoder layer in current chunk
                    chunk_quantized_model_info['output_scales'][0] = quantized_model_infos[i]['output_scales'][0]
                    chunk_quantized_model_info['output_zero_points'][0] = quantized_model_infos[i][
                        'output_zero_points'
                    ][0]
                    chunk_quantized_model_info['output_dtypes'][0] = quantized_model_infos[i]['output_dtypes'][0]
                # cache outputs layers
                if is_cache_evict:
                    # need to prevent interleaving of attn weights with cache outputs
                    # attn weights need to be behind cache
                    insert_pos = i * num_cache_inputs + 1
                    non_hidden_output_scales = quantized_model_infos[i]['output_scales'][1:]
                    non_hidden_output_zero_points = quantized_model_infos[i]['output_zero_points'][1:]
                    non_hidden_output_dtypes = quantized_model_infos[i]['output_dtypes'][1:]
                    for c_idx in range(num_cache_inputs):
                        chunk_quantized_model_info['output_scales'].insert(
                            insert_pos + c_idx, non_hidden_output_scales[c_idx]
                        )
                        chunk_quantized_model_info['output_zero_points'].insert(
                            insert_pos + c_idx, non_hidden_output_zero_points[c_idx]
                        )
                        chunk_quantized_model_info['output_dtypes'].insert(
                            insert_pos + c_idx, non_hidden_output_dtypes[c_idx]
                        )
                    chunk_quantized_model_info['output_scales'].append(non_hidden_output_scales[num_cache_inputs])
                    chunk_quantized_model_info['output_zero_points'].append(
                        non_hidden_output_zero_points[num_cache_inputs]
                    )
                    chunk_quantized_model_info['output_dtypes'].append(non_hidden_output_dtypes[num_cache_inputs])
                else:
                    chunk_quantized_model_info['output_scales'] += quantized_model_infos[i]['output_scales'][1:]
                    chunk_quantized_model_info['output_zero_points'] += quantized_model_infos[i]['output_zero_points'][
                        1:
                    ]
                    chunk_quantized_model_info['output_dtypes'] += quantized_model_infos[i]['output_dtypes'][1:]

        else:
            if chunk_quantized_model_info['model_type'] == 'whisper' and i == len(quantized_model_infos) - 1:
                chunk_quantized_model_info['output_scales'] = quantized_model_infos[i]['output_scales']
                chunk_quantized_model_info['output_zero_points'] = quantized_model_infos[i]['output_zero_points']
                chunk_quantized_model_info['output_dtypes'] = quantized_model_infos[i]['output_dtypes']
        if num_lora_inputs > 0:
            lora_input_scales.extend(quantized_model_infos[i]['input_scales'][-num_lora_inputs:])
            lora_input_zps.extend(quantized_model_infos[i]['input_zero_points'][-num_lora_inputs:])
            lora_input_dtypes.extend(quantized_model_infos[i]['input_dtypes'][-num_lora_inputs:])
    if len(lora_input_scales) > 0:
        chunk_quantized_model_info['input_scales'] += lora_input_scales
        chunk_quantized_model_info['input_zero_points'] += lora_input_zps
        chunk_quantized_model_info['input_dtypes'] += lora_input_dtypes
    return chunk_quantized_model_info


def sanitize_quantized_model_info(quantized_model_path, quantized_model_info):
    """Check that the quantized model i/o info tallies with the exported shape fixed model.

    Updates the quantized model info if any mismatches found.

    NOTE: Expand checks to other parts of quantized model info in future if necessary.

    Args:
    quantized_model_path (str): Path to exported shape fixed model.
    quantized_model_info (dict): The corresponding quantized model info for that shape fixed model.
    """
    subgraph = get_subgraph_from_quantized_model(quantized_model_path)
    _sanitize_quantized_model_info(subgraph, quantized_model_info, mode='input')
    _sanitize_quantized_model_info(subgraph, quantized_model_info, mode='output')
