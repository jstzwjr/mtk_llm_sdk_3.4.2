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
"""Script to Generate fixed-shape quantized model from a dynamic-shape quantized model generated from PTQ step."""

import argparse
import contextlib
import gc
import itertools
import os
import re
import shutil
import sys
from collections import OrderedDict
from copy import deepcopy

import mtk_converter
import numpy as np
from mtk_converter.python.tools.builder import ModelBuilder
from mtk_converter.python.utils import tensor_utils, type_utils
from tqdm import tqdm

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.lora_handler import LoRAHandler
from .models.pipeline import FloatPipeline
from .utils import const, logger, overture_utils, qat_utils, quantized_model_utils, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_fix_llm_shape'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert dynamic shape quantized models to fixed shape quantized models.', allow_abbrev=False
    )
    parser.add_argument(
        'quantized_model_folder', type=str, help='[Required] Input dynamic shape quantized model folder.'
    )
    parser.add_argument(
        'shapes',
        nargs='+',
        help='[Required] Expected input shapes to reconfigure quantized models to. Space separated list of '
        'shapes in the format: vl OR we OR xtyc OR xtyczr where v, w, x, y, and z are integers. '
        'v=embedder token size, w=encoder batch size, x=llm token size, y=llm cache size, z=lora rank. '
        '(e.g. 2l OR 17e OR 128t1024c OR 128t1024c16r). '
        'Rank cannot be omitted for models with LoRA inputs. '
        'Alternatively, every shape needs to be of the format: vl OR we OR xt OR yc OR zr for permutation mode.',
    )
    parser.add_argument(
        '-n',
        '--num_chunks_or_layers_per_chunk',
        type=int,
        default=[1],
        nargs='+',
        help='int or list of ints. If int, number of chunks the LLM should have, with evenly '
        'distributed number of decoder layers per chunk. If list of ints, number of decoder layers for each chunk, '
        'total sum needs to add up to total number of expected decoder layers. Default = 1. '
        'Encoder does not support chunking.',
    )
    parser.add_argument(
        '-l',
        '--lora_config',
        type=str,
        default=None,
        nargs='*',
        help='LoRA adapter config json file. Only specify when adding dynamic lora inputs to a non-dynamic lora '
        'input base model. Only specify multiple lora configs when using hotplug with int lora precision.',
    )
    parser.add_argument(
        '--separate_tail',
        action='store_true',
        help='flag to keep final LLM norm layer and lm_head FC as its own chunk.',
    )
    parser.add_argument(
        '--buffer_scale',
        type=float,
        default=1.0,
        help='Scaling factor for activation minmax as a buffer after LoRA Add OP. Defaults to 1.0 (no buffer). '
        'Only specify when adding dynamic lora inputs to a non-dynamic lora input base model.',
    )
    parser.add_argument(
        '-o',
        '--output_folder',
        type=str,
        default=None,
        help='User-specified output folder to save quantized model to. '
        'Will automatically save quantized model to ./quantized_models folder using name of weights '
        'folder and other parameters set for this PTQ script by default.',
    )
    parser.add_argument(
        '--np_compat',
        type=int,
        default=None,
        choices=const.NP_COMPAT_VERIONS,
        help='Export quantized model using specific backward-compatible neuropilot version OP export spec. '
        'Defaults to using the same version as installed `mtk_converter` package '
        '(NP7 converter will default to NP7 export spec). '
        'Cannot specify a neuropilot version that is newer than installed `mtk_converter` version.',
    )
    parser.add_argument(
        '--use_single_bmm_attention',
        action='store_true',
        help='Whether to use single bmm attention graph. ',
    )
    parser.add_argument(
        '--bita_decode_token',
        type=int,
        default=None,
        nargs='+',
        help='Number of bita decode token',
    )
    parser.add_argument('--encoder_only', action='store_true', help='Flag to do shape fixer on encoder only.')
    parser.add_argument('--per_op', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument(
        '--force_overwrite',
        action='store_true',
        help='Force overwrite of fixed shape quantized model folder if it already exists. Use with caution.',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    parser.add_argument(
        '--clear_attn_outputs',
        action='store_true',
        help='Forcely remove additional outputs which only relates to cache evict graph.',
    )
    parser.add_argument(
        '--no_pad_cache',
        action='store_false',
        dest='pad_cache',
        default=True,
        help='Disable padding precomputed cache (e.g., overture) to the given cache size.',
    )
    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        tuple: A tuple containing the list of quantized model files, permutation mode, custom tail type (if any), and
            quantized model information dicts.

    Raises:
        RuntimeError: If any of the argument checks fail.
        ValueError: If the sum of chunks does not equal the expected number of decoder layers.
    """
    sc.check_exist(args.quantized_model_folder, 'quantized model directory')
    sc.check_isdir(args.quantized_model_folder, 'quantized model directory')
    perm, has_rank = sc.check_shapes(args.shapes, encoder_only=args.encoder_only)
    quantized_model_list = {
        'embedder': [],
        'encoder': [],
        'llm': [],
    }
    embedding_folder = os.path.join(args.quantized_model_folder, 'embedder')
    encoder_folder = os.path.join(args.quantized_model_folder, 'encoder')
    llm_folder = os.path.join(args.quantized_model_folder, 'llm')
    if os.path.exists(embedding_folder):
        quantized_model_list['embedder'] = utils.get_sorted_path_list(embedding_folder, ['.tflite', '.mlir'], sep='int')

    if os.path.exists(encoder_folder):
        quantized_model_list['encoder'] = utils.get_sorted_path_list(encoder_folder, ['.tflite', '.mlir'])
    else:
        if args.encoder_only:
            logger.error('Must have dynamic shape encoder tflite folder when using encoder_only.')

    if args.encoder_only:
        if args.num_chunks_or_layers_per_chunk != [1]:
            logger.error('Chunking is not supported for encoder.')
        custom_tail_type = None
    else:
        quantized_model_list['llm'] = utils.get_sorted_path_list(llm_folder, ['.tflite', '.mlir'])

        sc.check_num_chunks(quantized_model_list['llm'], args.num_chunks_or_layers_per_chunk, args.separate_tail)
        custom_tail_type = quantized_model_utils.has_separate_tails(llm_folder)[1]

        if custom_tail_type is not None and not args.separate_tail:
            logger.error('`separate_tail` is required for models with medusa/EAGLE tails')
        num_decoder_layers = len(quantized_model_list['llm']) - 1

        if len(args.num_chunks_or_layers_per_chunk) == 1:
            if args.per_op:
                if args.num_chunks_or_layers_per_chunk[0] != num_decoder_layers:
                    logger.error(
                        '`num_chunks_or_layers_per_chunk` must be equal to number of decoder layers '
                        f'({num_decoder_layers}) for per-op output quantized model generation',
                        err=ValueError,
                    )
            else:
                if args.per_op:
                    logger.error(
                        '`num_chunks_or_layers_per_chunk` must be equal to number of decoder layers '
                        f'({num_decoder_layers}) for per-op output quantized model generation',
                        err=ValueError,
                    )
                sc.check_between_inclusive(args.num_chunks_or_layers_per_chunk[0], 1, num_decoder_layers, 'num_chunks')
        else:
            if sum(args.num_chunks_or_layers_per_chunk) != num_decoder_layers:
                logger.error(
                    f'sum of `num_chunks_or_layers_per_chunk` ({sum(args.num_chunks_or_layers_per_chunk)}) '
                    f'does not equal to expected number of decoder layers ({num_decoder_layers})',
                    err=ValueError,
                )

    quantized_model_infos = {
        'embedder': [],
        'encoder': [],
        'llm': [],
    }
    for f in quantized_model_list['embedder']:
        sc.check_ext(f, ['.tflite', '.mlir'])
        quantized_model_info = quantized_model_utils.extract_embedder_quantized_model_info(f)
        quantized_model_infos['embedder'].append(quantized_model_info)
    encoder_base_has_lora = False
    for f in quantized_model_list['encoder']:
        sc.check_ext(f, ['.tflite', '.mlir'])
        quantized_model_info = quantized_model_utils.extract_encoder_quantized_model_info(f)
        encoder_base_has_lora = encoder_base_has_lora or quantized_model_info['num_lora_inputs'] > 0
        quantized_model_infos['encoder'].append(quantized_model_info)

    llm_base_has_lora = False
    for f in quantized_model_list['llm']:
        sc.check_ext(f, ['.tflite', '.mlir'])
        sc.check_dynamic_shape(f)
        quantized_model_info = quantized_model_utils.extract_llm_quantized_model_info(f)
        llm_base_has_lora = llm_base_has_lora or quantized_model_info['num_lora_inputs'] > 0
        quantized_model_infos['llm'].append(quantized_model_info)

    if args.lora_config is not None:
        model_config_path = os.path.join(args.quantized_model_folder, 'config.json')
        model_config = PipelineConfig(model_config_path, verbose=False)
        lora_handler = LoRAHandler(args.lora_config, model_config)
        encoder_has_new_lora = lora_handler.has_encoder_lora()
        llm_has_new_lora = lora_handler.has_llm_lora()
        if encoder_base_has_lora and encoder_has_new_lora:
            logger.error(
                'Cannot add encoder dynamic lora inputs to an encoder model that already has dynamic lora inputs.'
            )
        if llm_base_has_lora and llm_has_new_lora:
            logger.warning(
                'The LLM model already has dynamic lora inputs. They will be replaced using the specified lora_config.'
            )

        # check lora_precision. Currently we only check llm since encoder lora not implemented yet
        for q_model_json in quantized_model_infos['llm']:
            lora_precision = q_model_json['precision_config']['lora_precision']
            if lora_precision is None:
                logger.warning('Found None Lora Precision in quantized model info. This will hotplug fp lora.')

        sc.check_between_inclusive(args.buffer_scale, 1.0, 10.0, 'buffer_scale')

        if len(quantized_model_infos['encoder']) > 0 and (
            quantized_model_infos['encoder'][0]['model_config']['fc_names']['attn'].get('qkv', 'qkv_proj')
            in lora_handler.global_encoder_target_modules
            or quantized_model_infos['encoder'][0]['model_config']['fc_names']['mlp'].get('gateup', 'gateup_proj')
            in lora_handler.global_encoder_target_modules
        ):
            logger.error('Combined FC target_modules not allowed. Please split them into individual FC target modules.')

        if (
            quantized_model_infos['llm'][0]['model_config']['fc_names']['attn'].get('qkv', 'qkv_proj')
            in lora_handler.global_llm_target_modules
            or quantized_model_infos['llm'][0]['model_config']['fc_names']['mlp'].get('gateup', 'gateup_proj')
            in lora_handler.global_llm_target_modules
        ):
            logger.error('Combined FC target_modules not allowed. Please split them into individual FC target modules.')

        if (llm_has_new_lora or encoder_has_new_lora) and not has_rank:
            logger.error('lora rank (r) needs to be provided for model with LoRA inputs.')
        if not (llm_has_new_lora or encoder_has_new_lora) and has_rank:
            logger.error('lora rank (r) provided for model without LoRA inputs.')
    else:
        if args.buffer_scale != 1.0:
            logger.error(
                'Cannot specify buffer_scale when not adding dynamic lora inputs to a non-dynamic lora input base model'
            )

    if args.output_folder is not None and os.path.exists(args.output_folder) and not args.force_overwrite:
        logger.error(
            f'Output folder {args.output_folder} already exists. '
            'Please manually delete the existing folder  or use --force_overwrite if intending to overwrite.',
            err=FileExistsError,
        )

    cvt_ver = utils.get_converter_version(include_minor=True)
    if args.np_compat is not None and args.np_compat > cvt_ver[0]:
        logger.error(
            f'`np_compat` ({args.np_compat}) cannot be greater than current mtk_converter version ({cvt_ver[0]})'
        )

    if cvt_ver[0] < 9 or (cvt_ver[0] == 9 and cvt_ver[1] < 2):
        logger.error(f'Found converter version {cvt_ver[0]}.{cvt_ver[1]} but >= 9.2.0 is required')

    if args.pad_cache:
        logger.warning(
            'Please note that padding Overture to the full cache size will be deprecated in the future. '
            'Padding will no longer be the default behavior. Currently, you can use the `--no_pad_cache` '
            'option to disable padding.'
        )

    return quantized_model_list, perm, custom_tail_type, quantized_model_infos


def print_args(args, partial_output_folder_without_shape):
    """Prints the arguments for verification.

    Args:
        args (argparse.Namespace): The parsed arguments.
        partial_output_folder_without_shape (str): The outermost output folder.
    """
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Quantized model folder: {args.quantized_model_folder}')
    llm_shapes = {x for x in args.shapes if 't' in x and 'c' in x}
    embedder_shapes = {x for x in args.shapes if 'l' in x}
    enc_shapes = args.shapes - llm_shapes - embedder_shapes
    if len(embedder_shapes) > 0:
        logger.info(f'Embedder shapes:                             {embedder_shapes}')
    if len(enc_shapes) > 0:
        logger.info(f'Encoder shapes:                              {enc_shapes}')
    if len(llm_shapes) > 0:
        logger.info(f'LLM shapes:                                  {llm_shapes}')
    logger.info(f'Output folder:                               {partial_output_folder_without_shape}')
    if len(args.num_chunks_or_layers_per_chunk) == 1:
        logger.info(f'Number of LLM chunks:                        {args.num_chunks_or_layers_per_chunk[0]}')
    else:
        logger.info(f'Number of LLM chunks:                        {len(args.num_chunks_or_layers_per_chunk)}')
    if args.lora_config is not None:
        logger.info(f'Add dynamic lora inputs:                     {args.lora_config}')
    logger.info(f'Use Single BMM Attention Graph:              {args.use_single_bmm_attention}')
    logger.info(f'Clear Attention Outputs:                     {args.clear_attn_outputs}')
    logger.info(f'Force Overwrite existing fixed shape models: {args.force_overwrite}')
    logger.info(f'mtk_llm_sdk version:                        {__version__}')
    logger.info(f'mtk_converter version:                      {mtk_converter.__version__}')


def export_lora_weight_bins(
    fixed_shape_model_dir,
    per_layer_dir,
    quantized_model_info,
    lora_shapes,
    chunk_idx,
):
    """Export LoRA weight binaries into fixed shape model directory.

    Args:
        fixed_shape_model_dir (str): The path to the chunked and fixed shape quantized model directory.
        per_layer_dir (str): The path to the per-layer dynamic shape quantized model directory.
        quantized_model_info (dict): The quantized model information dictionary.
        lora_shapes (nested list): The shapes of the LoRA weights.
        chunk_idx (int): The chunk index.

    Methods:
        get_header(version, lora_weights_sizes):
            Generates the header for the merged LoRA binary.

        read_header(data):
            Reads the header from a LoRA binary.

        export_lora_bins(lora_bin_paths, lora_precision, lora_shapes, lora_rank):
            Merges multiple LoRA binaries into a single binary.

        export_split_lora_bins(lora_bin_paths, lora_precision, lora_shapes, lora_rank):
            Splits the lora to per input and exports LoRA binaries into fixed shape model directory.

        export_merged_lora_bins(lora_bin_paths, output_merged_bin_path, lora_precision, lora_shapes, lora_rank):
            Merges all LoRA binary into one bin and exports them to the output_merged_bin_path.
    """
    import struct

    lora_bin_version = 1

    def get_header(version, lora_weights_sizes):
        num_lora_inputs = len(lora_weights_sizes)

        header_format = '<II'
        top_header_data = struct.pack(header_format, version, num_lora_inputs)

        sizes_section_format = f'<{num_lora_inputs}I'
        sizes_data = struct.pack(sizes_section_format, *lora_weights_sizes)

        return top_header_data + sizes_data

    def read_header(data):
        read_offset = 0

        # Read top header section
        header_format = '<II'
        version, num_lora_inputs = struct.unpack_from(header_format, data, read_offset)
        read_offset += struct.calcsize(header_format)

        # Read sizes section
        sizes_section_format = f'<{num_lora_inputs}I'
        lora_weights_sizes = struct.unpack_from(sizes_section_format, data, read_offset)
        read_offset += struct.calcsize(sizes_section_format)

        return read_offset, version, lora_weights_sizes

    def export_lora_bins(lora_bin_paths, lora_precision, lora_shapes, lora_rank, remove_header=False, split=False):
        lora_weights_sizes_to_export = []
        lora_weights_data_to_export = []

        if len(lora_bin_paths) != len(lora_shapes):
            logger.error(
                f'length of lora_bin_paths ({len(lora_bin_paths)}) is different from lora_shapes ({len(lora_shapes)})'
            )

        if not split:
            pbar = tqdm(total=len(lora_bin_paths))
        for layer_lora_bin_path, layer_lora_shapes in zip(lora_bin_paths, lora_shapes):
            lora_bin_data = None
            with open(layer_lora_bin_path, 'rb') as file:
                lora_bin_data = file.read()
            weights_start_offset, version, lora_weights_sizes = read_header(lora_bin_data)

            # Ensure bin version is supported
            if version != lora_bin_version:
                logger.error(f'Unsupported lora bin version: {version}')

            if len(lora_weights_sizes) != len(layer_lora_shapes):
                logger.error(
                    f'length of lora_weights_sizes ({len(lora_weights_sizes)}) is '
                    f'different from layer_lora_shapes ({len(layer_lora_shapes)})'
                )

            # Lora weights size checking
            expected_total_size = sum(lora_weights_sizes)
            actual_total_size = len(lora_bin_data) - weights_start_offset
            if expected_total_size != actual_total_size:
                logger.error(
                    f'Expected to read total {expected_total_size} bytes of lora weights but only '
                    f'{actual_total_size} bytes is available.'
                )

            # Merge lora weights
            # NOTE: Or simply replace below code section with this one-liner:
            #     `lora_weights_data_to_export.append(lora_bin_data[weights_start_offset:])`
            # but will lose the flexibility to select lora weights + sizes PER INPUT.
            lora_weights_data = lora_bin_data[weights_start_offset:]
            read_offset = 0
            post_pad_lora_weights_sizes = []
            for lora_weights_size, lora_shape in zip(lora_weights_sizes, layer_lora_shapes):
                if len(lora_shape) == 3:
                    assert lora_shape[0] == 1, lora_shape
                    lora_shape = [lora_shape[1], lora_shape[2]]
                else:
                    lora_shape = [lora_shape[0], lora_shape[1]]
                if lora_shape[0] == lora_rank:
                    lora_non_rank_idx = 1
                else:
                    assert lora_shape[1] == lora_rank, lora_shape
                    lora_non_rank_idx = 0
                lora_non_rank_shape = lora_shape[lora_non_rank_idx]
                end_offset = read_offset + lora_weights_size
                current_lora_weights = lora_weights_data[read_offset:end_offset]
                lora_np = np.frombuffer(current_lora_weights, dtype=lora_precision)
                if lora_non_rank_idx == 0:
                    lora_np = lora_np.reshape(lora_non_rank_shape, -1)
                else:
                    lora_np = lora_np.reshape(-1, lora_non_rank_shape)

                pad_len = lora_rank - lora_np.shape[int(1 - lora_non_rank_idx)]
                if pad_len < 0:
                    logger.error(
                        f'lora rank ({lora_rank}) to fix shape cannot be smaller than'
                        f' lora rank of lora weights ({lora_np.shape[int(1 - lora_non_rank_idx)]})',
                        err=ValueError,
                    )
                if pad_len > 0:
                    # Require padding
                    if lora_non_rank_idx == 0:
                        lora_np = np.pad(lora_np, ((0, 0), (0, pad_len)))
                    else:
                        lora_np = np.pad(lora_np, ((0, pad_len), (0, 0)))
                    post_pad_lora_weights_sizes.append(lora_np.nbytes)
                    current_lora_weights = lora_np.tobytes()
                lora_weights_data_to_export.append(current_lora_weights)
                read_offset = end_offset

            # Merge lora weights sizes
            if len(post_pad_lora_weights_sizes) == 0:
                lora_weights_sizes_to_export.extend(lora_weights_sizes)
            else:
                lora_weights_sizes_to_export.extend(post_pad_lora_weights_sizes)
            assert read_offset == actual_total_size, f'read_offset={read_offset}, actual_total_size={actual_total_size}'
            assert len(lora_weights_sizes_to_export) == len(lora_weights_data_to_export)  # Each item is PER INPUT
            if not split:
                pbar.update(1)
        if not split:
            pbar.close()

        if split:
            final_data = []
            for i in range(len(lora_weights_data_to_export)):
                header_data = get_header(lora_bin_version, [lora_weights_sizes_to_export[i]])
                final_data.extend(
                    [header_data + lora_weights_data_to_export[i]]
                    if not remove_header
                    else [lora_weights_data_to_export[i]]
                )
        else:
            header_data_merged = get_header(lora_bin_version, lora_weights_sizes_to_export)
            lora_weights_data_merged = b''.join(lora_weights_data_to_export)
            final_data = (
                [header_data_merged + lora_weights_data_merged] if not remove_header else [lora_weights_data_merged]
            )

        return final_data

    def export_merged_lora_bins(
        lora_bin_paths, output_merged_bin_path, lora_precision, lora_shapes, lora_rank, remove_header=False
    ):
        logger.debug(f'LoRA bin files to merge: {lora_bin_paths}')
        logger.debug(f'Number of chunks with lora: {len(lora_shapes)}')
        # Read and merge the lora bins
        logger.info('Exporting merged lora bins.')
        merged_data = export_lora_bins(lora_bin_paths, lora_precision, lora_shapes, lora_rank, remove_header)

        # Export merged bin
        with open(output_merged_bin_path, 'wb') as outfile:
            outfile.write(merged_data[0])

    def export_split_lora_bins(lora_bin_paths, output_dir, lora_precision, lora_shapes, lora_rank, remove_header):
        # loop through all bins one by one
        logger.debug(f'LoRA bin files to export: {lora_bin_paths}')
        logger.debug(f'Number of chunks with lora: {len(lora_shapes)}')

        if len(lora_bin_paths) != len(lora_shapes):
            logger.error(
                f'length of lora_bin_paths ({len(lora_bin_paths)}) is different from lora_shapes ({len(lora_shapes)})'
            )

        logger.info('Exporting split lora bins.')
        pbar = tqdm(total=len(lora_bin_paths))
        for lora_bin_path, lora_shape in zip(lora_bin_paths, lora_shapes):
            split_data = export_lora_bins(
                [lora_bin_path], lora_precision, [lora_shape], lora_rank, remove_header, split=True
            )

            for lora_input in split_data:
                output_path = os.path.join(output_dir, f'lora_chunk_{len(os.listdir(output_dir))}.bin')
                # Export split bin
                with open(output_path, 'wb') as outfile:
                    outfile.write(lora_input)
            pbar.update(1)
        pbar.close()

    per_layer_lora_dirs = [
        os.path.join(per_layer_dir, x)
        for x in os.listdir(per_layer_dir)
        if os.path.isdir(os.path.join(per_layer_dir, x))
    ]

    config = PipelineConfig(os.path.join(os.path.dirname(per_layer_dir), 'config.json'), verbose=False)
    if config.rotate:
        # PTQ with Lora case, per-layer rotated Lora bins are saved in lora/rotate_hadamard_0/
        per_layer_lora_dirs = [
            os.path.join(d, f'rotate_{config.rotate_mode}_{config.rotate_seed}') for d in per_layer_lora_dirs
        ]

    if len(per_layer_lora_dirs) == 0:
        logger.warning(
            f'Skipping merging lora cmdline bins as no per-layer lora bin folders detected in {per_layer_dir}'
        )
        return
    logger.debug(f'Found {len(per_layer_lora_dirs)} LoRA directories to merge LoRA bins for: {per_layer_lora_dirs}')

    # FIXME: will fail for non sym lora
    lora_precision = quantized_model_info['precision_config']['lora_precision']
    lora_precision = quantized_model_info['precision_config']['config_precision_mapping'][lora_precision]
    if lora_precision == 'float':
        lora_precision = 'float16'
    lora_precision = quantized_model_utils._make_dtype_numpy(lora_precision)  # noqa: SLF001

    for lora_dir in per_layer_lora_dirs:
        lora_name = '/'.join(lora_dir.split('/')[-2 if config.rotate else -1 :])
        logger.debug(f'Curr LoRA: {lora_name}')
        output_dir = os.path.join(fixed_shape_model_dir, lora_name)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'lora_chunk_{chunk_idx}.bin')
        if os.path.exists(output_path):
            logger.debug(f'Skip create merged lora bin as exist: {output_path}')
            continue

        bin_list = []
        skip = False
        lora_start_layer_idx = quantized_model_info.get('lora_start_layer_idx', None)
        lora_end_layer_idx = quantized_model_info.get('lora_end_layer_idx', None)
        for idx in quantized_model_info['layer_ids']:
            if lora_start_layer_idx is not None and idx < lora_start_layer_idx:
                continue
            if lora_end_layer_idx is not None and idx > lora_end_layer_idx:
                continue
            per_layer_bin_path = os.path.join(lora_dir, f'lora_chunk_{idx}.bin')
            if not os.path.exists(per_layer_bin_path):
                logger.warning(
                    f'Skipping merging cmdline bins as per-layer cmdline bin does not exist: {per_layer_bin_path}'
                )
                skip = True
                break
            bin_list.append(per_layer_bin_path)
        if not skip:
            if (
                quantized_model_info['model_config']['model_type'] == 'gecko2'
                or quantized_model_info['model_config']['model_type'] == 'gecko'
            ):
                export_split_lora_bins(
                    bin_list,
                    output_dir,
                    lora_precision,
                    lora_shapes,
                    quantized_model_info['lora_rank'],
                    remove_header=True,
                )
            else:
                export_merged_lora_bins(
                    bin_list, output_path, lora_precision, lora_shapes, quantized_model_info['lora_rank']
                )


def permutate_shapes(orig_shapes):
    """Permutates the given shapes to generate all possible combinations of tokens, caches, and ranks.

    Args:
        orig_shapes (list): The original shapes.

    Returns:
        list: A list of permutated shapes.
    """
    logger.debug(f'Pre-permutated shapes: {orig_shapes}')
    encoder_batches = []
    batches = []
    tokens = []
    caches = []
    ranks = []
    for shape in orig_shapes:
        if shape.count('b') == 1:
            batches.append(shape)
        elif shape.count('t') == 1:
            tokens.append(shape)
        elif shape.count('c') == 1:
            caches.append(shape)
        elif shape.count('r') == 1:
            ranks.append(shape)
        elif shape.count('e') == 1:
            encoder_batches.append(shape)
    if len(ranks) == 0:
        ranks = ['']
    shapes = []
    for r in itertools.product(batches, ranks):
        shapes.append(r[0] + r[1])
    for r in itertools.product(tokens, caches, ranks):
        shapes.append(r[0] + r[1] + r[2])
    shapes = set(shapes + encoder_batches)
    logger.debug(f'Permutated shapes: {shapes}')
    return shapes


def shape_fix_embedder(
    args,
    exp_name,
    partial_output_folder_without_shape,
    output_dest_path,
    embedder_model_list,
    embedder_quantized_model_infos,
):
    """This function contains flow for shape fixing embedder.

    Args:
        args (Namespace): custom argument passing from another python script.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        output_dest_path (str): The path to the chunked dynamic shape quantized embedder models.
        embedder_model_list (list): The list of chunked dynamic shape quantized embedder models.
        embedder_quantized_model_infos (list): The list of quantized model infos for the embedder model.
    """
    for i, f in enumerate(embedder_model_list):
        subgraph, model = quantized_model_utils.get_subgraph_from_quantized_model(f, return_nrpmodel=True)
        embedder_quantized_model_info = embedder_quantized_model_infos[i]
        logger.debug(f'[shape_fix_embedder] Embedder, quantized_model_info={embedder_quantized_model_info}')
        quantized_model_description = model.description
        logger.debug(f'[shape_fix_embedder] description={quantized_model_description}')

        output_name_remap = OrderedDict()
        output_name_remap[subgraph.outputs[0]] = 'embeddings'
        output_name_remap[subgraph.outputs[1]] = 'per_layer_embeddings'
        logger.debug(f'[shape_fix_embedder] output_name_remap={output_name_remap}')

        for out_idx, (src_name, dst_name) in enumerate(output_name_remap.items()):
            if src_name != dst_name:
                logger.debug(f'[shape_fix_embedder] Rename output tensor {src_name} to {dst_name}')
                subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
                del subgraph.tensor_map[src_name]
                subgraph.outputs[out_idx] = dst_name

        for op in subgraph.operators:
            for out_idx, op_out in enumerate(op.outputs):
                if op_out in output_name_remap:
                    op.outputs[out_idx] = output_name_remap[op_out]

        if i == 0:
            builder = ModelBuilder()
            builder.import_subgraph(model)
        else:
            logger.error('[shape_fix_embedder] Only single embedder models supported.', err=ValueError)

    input_names = ['token_ids']
    output_names = ['embeddings', 'per_layer_embeddings']

    logger.debug(f'[shape_fix_embedder] Finalized Input Names:\n{input_names}')
    logger.debug(f'[shape_fix_embedder] Finalized Output Names:\n{output_names}\n')

    out_nrpmodel = builder.export(input_names, output_names)
    subgraph = out_nrpmodel.subgraphs[0]
    quantized_model_utils.export_quantized_model(out_nrpmodel, output_dest_path, np_compat=args.np_compat)

    fix_embedder_shape(
        output_dest_path,
        exp_name,
        partial_output_folder_without_shape,
        args.shapes,
        embedder_quantized_model_info,
        ptq_description=quantized_model_description,
        file_format=output_dest_path.rsplit('.', 1)[1],
        np_compat=args.np_compat,
    )
    logger.info('[shape_fix_embedder] Done fixing embedder quantized model.')


def fix_embedder_shape(
    quantized_model_path,
    exp_name,
    partial_output_folder_without_shape,
    shapes,
    quantized_model_info,
    ptq_description='',
    file_format='tflite',
    np_compat=None,
):
    """Fixes the shape of the constructed embedder TFLite or MLIR model.

    Currently only used by Gecko2.

    Args:
        quantized_model_path (str): The path to the chunked dynamic shape quantized model.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        shapes (set): The set of shapes to fix.
        quantized_model_info (dict): The quantized model info dictionary containing all the dynamic quantized model
            information.
        ptq_description (str, optional): The custom tflite description from PTQ step. Default is ''.
        file_format (str, optional): The file format of the quantized model. Either `tflite` or `mlir`.
            Default is `tflite`.
        np_compat (int, optional): Whether to use backward neuropilot compatibility mode when exporting.

    Raises:
        FileNotFoundError: If the encoder TFLite or MLIR model is not found.
    """
    if not os.path.exists(quantized_model_path):
        logger.error(
            f'[fix_embedder_shape] {file_format} to reconfigure shape not found: {quantized_model_path}.',
            err=FileNotFoundError,
        )

    subgraph = quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    logger.debug(f'[fix_embedder_shape] Embedder input shape: {input_tensor_shape}')
    if None in input_tensor_shape:
        shapes = {x for x in shapes if 'l' in x}
        logger.debug(f'[fix_embedder_shape] Embedder shapes to fix: {shapes}')
        logger.info(f'[fix_embedder_shape] Fixing {len(shapes)} shapes for embedder:')

        for shape in tqdm(shapes):
            logger.debug(f'[fix_embedder_shape] curr shape={shape}')
            output_folder = os.path.join(partial_output_folder_without_shape, 'embedder', f'embedder_{shape}')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f'{exp_name}_{shape}.{file_format}')
            batch_size = 1  # hardcode for now
            num_token = int(re.findall(r'\d+l', shape)[0][:-1])

            editor = (
                mtk_converter.TFLiteEditor(quantized_model_path)
                if file_format == 'tflite'
                else mtk_converter.MlirEditor(quantized_model_path)
            )

            target_shapes = [[batch_size, num_token]]  # Input Token IDs

            logger.debug(f'[fix_embedder_shape] target_shapes={target_shapes}')

            editor.reconfigure_input_shapes(target_shapes)

            op_export_spec = utils.get_op_export_spec(np_compat, file_format)

            if file_format == 'tflite':
                if ptq_description != '':
                    custom_description = ptq_description.rsplit('\n', 1)[1]
                    custom_description = custom_description + f'\nShape fixed by mtk_llm_sdk v{__version__}'
                else:
                    custom_description = 'Post Training Quantized by unknown version of mtk_llm_sdk\n'
                    f'Shape fixed by mtk_llm_sdk v{__version__}'
                try:
                    editor.export(
                        output_path, tflite_op_export_spec=op_export_spec, custom_description=custom_description
                    )
                except TypeError:
                    editor.export(output_path, tflite_op_export_spec=op_export_spec)
            else:
                editor.export(output_path, export_spec=op_export_spec)

            # Generate fixed shape quantized model info json
            quantized_model_info['batch_size'] = batch_size
            new_quantized_model_info_path = output_path.replace(f'.{file_format}', '_info.json')
            quantized_model_utils.export_quantized_model_info(
                quantized_model_info,
                quantized_model_info['model_config'],
                quantized_model_info['precision_config'],
                new_quantized_model_info_path,
            )
    else:
        # Embedder is already fixed shape, just copy to fixed-shape output directory
        output_folder = os.path.join(partial_output_folder_without_shape, 'embedder', 'embedder')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{exp_name}.{file_format}')
        shutil.copyfile(quantized_model_path, output_path)
        new_quantized_model_info_path = output_path.replace(f'.{file_format}', '_info.json')
        quantized_model_utils.export_quantized_model_info(
            quantized_model_info,
            quantized_model_info['model_config'],
            quantized_model_info['precision_config'],
            new_quantized_model_info_path,
        )


def shape_fix_encoder(
    args,
    exp_name,
    partial_output_folder_without_shape,
    output_dest_path,
    encoder_model_list,
    encoder_quantized_model_infos,
    file_format_separator,
):
    """This function contains flow for shape fixing encoder.

    Args:
        args (Namespace): custom argument passing from another python script.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        output_dest_path (str): The path to the chunked dynamic shape quantized encoder models.
        encoder_model_list (list): The list of chunked dynamic shape quantized encoder models.
        encoder_quantized_model_infos (list): The list of quantized model infos for the encoder model.
        file_format_separator (str): The separator used in the quantized model.

    Returns:
        enc_chunked_model_info (dict): A dict containing the merged quantized model info of the encoder model chunk.

    """
    prev_output_scale = None
    prev_output_zp = None
    prev_output_dtype = None
    quantized_model_description = None
    lora_inp_shapes = []
    lora_inp_names = []
    extra_inp_names = set()
    # Patch F: 跨 chunk 累积"暴露成 final model output 的 secondary outputs"
    final_extra_output_names = []
    deepstack_chunk_indices = []   # 记录哪些 encoder chunk 贡献了 secondary（projector 引用其名字）
    pbar = tqdm(total=len(encoder_model_list))
    for i, f in enumerate(encoder_model_list):
        subgraph, model = quantized_model_utils.get_subgraph_from_quantized_model(f, return_nrpmodel=True)
        quantized_model_info = encoder_quantized_model_infos[i]
        logger.debug(f'[shape_fix_encoder] layer index {i}, quantized_model_info={quantized_model_info}')

        # Inter-chunk quant param alignment
        if prev_output_scale is not None:
            assert prev_output_zp is not None
            if quantized_model_info['input_dtypes'][0] != prev_output_dtype:
                logger.error(
                    f'[shape_fix_encoder] Layer {i} input dtype ({quantized_model_info["input_dtypes"][0]}) '
                    f'does not match layer {i - 1} output dtype ({prev_output_dtype})'
                )
            logger.debug(
                f'[shape_fix_encoder] Aligning hidden states quant params of layer {i} input to layer {i - 1} output.'
            )
            inp_tensor = subgraph.tensor_map[subgraph.inputs[0]]
            inp_tensor.quant_param.linear.scale_vals[0] = prev_output_scale
            inp_tensor.quant_param.linear.zero_point_vals[0] = prev_output_zp
            final_qmin, final_qmax, _ = type_utils.get_quant_value_range(
                type_utils.get_nrp_type(quantized_model_info['input_dtypes'][0])
            )
            final_min_vals = utils.dequantize(final_qmin, prev_output_scale, prev_output_zp)
            final_max_vals = utils.dequantize(final_qmax, prev_output_scale, prev_output_zp)
            inp_tensor.quant_param.linear.min_vals[0] = final_min_vals
            inp_tensor.quant_param.linear.max_vals[0] = final_max_vals
            quantized_model_info['input_scales'][0] = prev_output_scale
            quantized_model_info['input_zero_points'][0] = prev_output_zp

        prev_output_scale = quantized_model_info['output_scales'][0]
        prev_output_zp = quantized_model_info['output_zero_points'][0]
        prev_output_dtype = quantized_model_info['output_dtypes'][0]

        if quantized_model_description is None:
            quantized_model_description = model.description
            logger.debug(f'[shape_fix_encoder] description={quantized_model_description}')
        else:
            if model.description != quantized_model_description:
                logger.warning(
                    "[shape_fix_encoder] current layer's description does not match previous layers' descriptions."
                )

        num_non_lora_inputs = quantized_model_info['num_non_lora_inputs']
        is_projector_chunk = quantized_model_info.get('projector', False)
        # rename all non-lora inputs
        input_name_remap = OrderedDict()
        input_name_remap[subgraph.inputs[0]] = 'encoder_input'
        if num_non_lora_inputs > 1:
            if is_projector_chunk:
                # Patch F6: projector inputs[1..] 起 neutral 名字（避免与 encoder chunks 已暴露的
                # secondary outputs 同名导致 builder.import_subgraph 报 conflict）；
                # 真正的连接通过 name_mapping_dict（在 import 阶段做映射）实现。
                for _k_in in range(1, num_non_lora_inputs):
                    input_name_remap[subgraph.inputs[_k_in]] = f'projector_input_{_k_in}'
            else:
                input_name_remap[subgraph.inputs[1]] = 'audio_attn_mask'
                extra_inp_names.add('audio_attn_mask')

        # rename all lora inputs
        for j in range(quantized_model_info['num_lora_inputs']):
            if subgraph.inputs[num_non_lora_inputs + j].startswith('argument'):
                input_name_remap[subgraph.inputs[num_non_lora_inputs + j]] = f'lora_input_{i}_{j}'
            else:
                initial_str = file_format_separator.join(['layers', '0']) + file_format_separator
                replacement_str = file_format_separator.join(['layers', str(i)]) + file_format_separator
                input_name_remap[subgraph.inputs[num_non_lora_inputs + j]] = subgraph.inputs[
                    num_non_lora_inputs + j
                ].replace(initial_str, replacement_str)
        logger.debug(f'[shape_fix_encoder] input_name_remap={input_name_remap}')

        output_name_remap = OrderedDict()

        out_name_list = ['encoder_output']
        if quantized_model_info['model_type'] == 'whisper' and (
            quantized_model_info['model_config']['num_hidden_layers'] - 1 in quantized_model_info['layer_ids']
        ):
            output_name_remap[subgraph.outputs[0]] = out_name_list[0]

            for i in range(1, len(subgraph.outputs)):
                out_name = f'layer_{(i - 1) // 2}_cross_k' if i % 2 != 0 else f'layer_{(i - 2) // 2}_cross_v'
                output_name_remap[subgraph.outputs[i]] = out_name
                out_name_list.append(out_name)
        else:
            output_name_remap[subgraph.outputs[0]] = (
                out_name_list[0]
                if quantized_model_info['projector'] or i == len(encoder_model_list) - 1
                else f'hidden_states_{i}'
            )
            # Patch F: 任何 chunk 的 secondary outputs 都用唯一 chunk-indexed 名命名，
            # 跨 chunk 累积到 final_extra_output_names；最终作为合并 encoder model 的额外 outputs。
            for _j_out in range(1, len(subgraph.outputs)):
                _ds_name = f'extra_output_chunk{i}_idx{_j_out}'
                output_name_remap[subgraph.outputs[_j_out]] = _ds_name
                final_extra_output_names.append(_ds_name)
                if not is_projector_chunk and _j_out == 1:
                    deepstack_chunk_indices.append(i)

        logger.debug(f'[shape_fix_encoder] output_name_remap={output_name_remap}')

        for inp_idx, (src_name, dst_name) in enumerate(input_name_remap.items()):
            if src_name != dst_name:
                logger.debug(f'[shape_fix_encoder] Rename input tensor {src_name} to {dst_name}')
                subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
                del subgraph.tensor_map[src_name]
                subgraph.inputs[inp_idx] = dst_name

        for out_idx, (src_name, dst_name) in enumerate(output_name_remap.items()):
            if src_name != dst_name:
                logger.debug(f'[shape_fix_encoder] Rename output tensor {src_name} to {dst_name}')
                subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
                del subgraph.tensor_map[src_name]
                subgraph.outputs[out_idx] = dst_name

        remainder_name_remap = {}
        for k in set(subgraph.tensor_map):
            if k not in input_name_remap.values() and k not in output_name_remap.values():
                suffix = k[9:] if k.startswith('layers/0/') else k
                new_name = f'layers/{i}/' + suffix
                if k != new_name:
                    logger.debug(f'[shape_fix_encoder] Rename tensor {k} to {new_name}')
                    subgraph.tensor_map[new_name].CopyFrom(subgraph.tensor_map[k])
                    del subgraph.tensor_map[k]
                remainder_name_remap[k] = new_name
        logger.debug(f'[shape_fix_encoder] remainder_name_remap={remainder_name_remap}')

        for op in subgraph.operators:
            for inp_idx, op_inp in enumerate(op.inputs):
                if op_inp in input_name_remap:
                    op.inputs[inp_idx] = input_name_remap[op_inp]
                elif op_inp in output_name_remap:
                    op.inputs[inp_idx] = output_name_remap[op_inp]
                elif op_inp in remainder_name_remap:
                    op.inputs[inp_idx] = remainder_name_remap[op_inp]
                else:
                    logger.error(f'[shape_fix_encoder] {op_inp} is not in any of the name remaps')

            for out_idx, op_out in enumerate(op.outputs):
                if op_out in output_name_remap:
                    op.outputs[out_idx] = output_name_remap[op_out]
                elif op_out in input_name_remap:
                    op.outputs[out_idx] = input_name_remap[op_out]
                elif op_out in remainder_name_remap:
                    op.outputs[out_idx] = remainder_name_remap[op_out]
                else:
                    logger.error(f'[shape_fix_encoder] {op_out} is not in any of the name remaps')

        if i == 0:
            if quantized_model_info['projector']:
                logger.error('[shape_fix_encoder] First encoder chunk is not expected to be a projector layer.')
            main_subgraph_inputs = subgraph.inputs[:]
            prev_subgraph_output = subgraph.outputs[0]
            builder = ModelBuilder()
            builder.import_subgraph(model)
        else:
            name_mapping_dict = {}
            name_mapping_dict[subgraph.inputs[0]] = prev_subgraph_output
            if is_projector_chunk:
                # Patch F6: projector inputs[1..]（已 rename 为 'projector_input_<k>'）显式映射到
                # 前面 encoder chunks 暴露的 secondary outputs 名字，让 builder 在 import 时替换为
                # 已存在的 tensor，从而正确连接 deepstack 通路。
                for _k_in in range(1, num_non_lora_inputs):
                    if _k_in - 1 < len(deepstack_chunk_indices):
                        _src_chunk = deepstack_chunk_indices[_k_in - 1]
                        name_mapping_dict[f'projector_input_{_k_in}'] = f'extra_output_chunk{_src_chunk}_idx1'
            elif num_non_lora_inputs > 1:
                name_mapping_dict[subgraph.inputs[1]] = main_subgraph_inputs[1]
            logger.debug(f'[shape_fix_encoder] name_mapping_dict={name_mapping_dict}')

            builder.import_subgraph(model, input_name_mappings=name_mapping_dict)
            prev_subgraph_output = subgraph.outputs[0]

        layer_lora_inp_shapes = []
        for lora_inp in subgraph.inputs[num_non_lora_inputs:]:
            lora_input_tensor = subgraph.tensor_map[lora_inp]
            input_tensor_shape = tensor_utils.get_shape(lora_input_tensor).as_list()
            layer_lora_inp_shapes.append(input_tensor_shape)
            lora_inp_names.append(lora_inp)
        logger.debug(f'[shape_fix_encoder] layer_lora_inp_shapes={layer_lora_inp_shapes}')
        if len(layer_lora_inp_shapes) > 0:
            lora_inp_shapes.append(layer_lora_inp_shapes)

        pbar.update(1)
    pbar.close()

    input_names = ['encoder_input', *extra_inp_names, *lora_inp_names]
    # Patch F: 把跨 chunk 累积的 secondary outputs 拼到最终 output_names 末尾
    # （旧模型没有 secondary outputs 时 final_extra_output_names 为空，行为同前）
    output_names = list(out_name_list) + final_extra_output_names

    logger.debug(f'[shape_fix_encoder] Finalized Input Names:\n{input_names}')
    logger.debug(f'[shape_fix_encoder] Finalized Output Names:\n{output_names}\n')

    out_nrpmodel = builder.export(input_names, output_names)
    subgraph = out_nrpmodel.subgraphs[0]
    quantized_model_utils.export_quantized_model(out_nrpmodel, output_dest_path, np_compat=args.np_compat)

    enc_chunked_model_info = quantized_model_utils.merge_quantized_model_infos(encoder_quantized_model_infos)

    fix_encoder_shape(
        output_dest_path,
        exp_name,
        partial_output_folder_without_shape,
        os.path.join(args.quantized_model_folder, 'encoder'),
        args.shapes,
        enc_chunked_model_info,
        lora_inp_shapes,
        ptq_description=quantized_model_description,
        file_format=output_dest_path.rsplit('.', 1)[1],
        np_compat=args.np_compat,
    )
    logger.info('[shape_fix_encoder] Done fixing encoder quantized model.')

    return enc_chunked_model_info


def fix_encoder_shape(
    quantized_model_path,
    exp_name,
    partial_output_folder_without_shape,
    per_layer_dir,
    shapes,
    quantized_model_info,
    lora_shapes=None,
    ptq_description='',
    file_format='tflite',
    np_compat=None,
):
    """Fixes the shape of the constructed encoder TFLite or MLIR model.

    Args:
        quantized_model_path (str): The path to the chunked dynamic shape quantized model.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        per_layer_dir (str): The path to the per-layer dynamic shape quantized model directory.
        shapes (set): The set of shapes to fix.
        quantized_model_info (dict): The quantized model info dictionary containing all the dynamic quantized model
            information.
        lora_shapes (list, optional): The shapes of the LoRA inputs. Default is None.
        ptq_description (str, optional): The custom tflite description from PTQ step. Default is ''.
        file_format (str, optional): The file format of the quantized model. Either `tflite` or `mlir`.
            Default is `tflite`.
        np_compat (int, optional): Whether to use backward neuropilot compatibility mode when exporting.

    Raises:
        FileNotFoundError: If the encoder TFLite or MLIR model is not found.
    """
    if lora_shapes is None:
        lora_shapes = []
    if not os.path.exists(quantized_model_path):
        logger.error(
            f'[fix_encoder_shape] {file_format} to reconfigure shape not found: {quantized_model_path}.',
            err=FileNotFoundError,
        )

    subgraph = quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
    input_tensor_shape = tensor_utils.get_shape(input_tensor).as_list()
    logger.debug(f'[fix_encoder_shape] Encoder input shape: {input_tensor_shape}')
    if None in input_tensor_shape:
        # At least one encoder dim is dynamic
        shapes = {x for x in shapes if 'e' in x}
        logger.debug(f'[fix_encoder_shape] Encoder shapes to fix: {shapes}')
        logger.info(f'[fix_encoder_shape] Fixing {len(shapes)} shapes for encoder:')
        if None in input_tensor_shape[1:]:
            logger.error(
                '[fix_encoder_shape] Only expected first dim of encoder to be dynamic, '
                f'but got input shape: {input_tensor_shape}',
                err=ValueError,
            )

        for shape in tqdm(shapes):
            logger.debug(f'[fix_encoder_shape] curr shape={shape}')
            output_folder = os.path.join(partial_output_folder_without_shape, 'encoder', f'encoder_{shape}')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f'{exp_name}_{shape}_0.{file_format}')
            batch_size = int(re.findall(r'\d+e', shape)[0][:-1])

            editor = (
                mtk_converter.TFLiteEditor(quantized_model_path)
                if file_format == 'tflite'
                else mtk_converter.MlirEditor(quantized_model_path)
            )
            if len(lora_shapes) > 0:
                lora_shapes_unpacked = [x for xx in lora_shapes for x in xx]  # flatten lora shapes
                lora_rank = int(re.findall(r'\d+r', shape)[0][:-1])
                for idd in range(len(lora_shapes_unpacked)):
                    lora_shapes_unpacked[idd][0] = 1
                    for i, dim in enumerate(lora_shapes_unpacked[idd]):
                        if dim is None:
                            lora_shapes_unpacked[idd][i] = lora_rank
            else:
                lora_shapes_unpacked = []

            target_shapes = [[batch_size, *input_tensor_shape[1:]]]
            target_shapes = target_shapes + lora_shapes_unpacked  # Lora inputs

            logger.debug(f'[fix_encoder_shape] target_shapes={target_shapes}')

            editor.reconfigure_input_shapes(target_shapes)

            op_export_spec = utils.get_op_export_spec(np_compat, file_format)

            if file_format == 'tflite':
                if ptq_description != '':
                    custom_description = ptq_description.rsplit('\n', 1)[1]
                    custom_description = custom_description + f'\nShape fixed by mtk_llm_sdk v{__version__}'
                else:
                    custom_description = 'Post Training Quantized by unknown version of mtk_llm_sdk\n'
                    f'Shape fixed by mtk_llm_sdk v{__version__}'
                try:
                    editor.export(
                        output_path, tflite_op_export_spec=op_export_spec, custom_description=custom_description
                    )
                except TypeError:
                    editor.export(output_path, tflite_op_export_spec=op_export_spec)
            else:
                editor.export(output_path, export_spec=op_export_spec)

            # Generate fixed shape quantized model info json
            quantized_model_info['batch_size'] = batch_size
            if len(lora_shapes) > 0:
                quantized_model_info['lora_rank'] = lora_rank
            new_quantized_model_info_path = output_path.replace(f'_0.{file_format}', '_info_0.json')
            quantized_model_utils.export_quantized_model_info(
                quantized_model_info,
                quantized_model_info['model_config'],
                quantized_model_info['precision_config'],
                new_quantized_model_info_path,
            )

    else:
        # Encoder is already fixed shape, just copy to fixed-shape output directory
        output_folder = os.path.join(partial_output_folder_without_shape, 'encoder', 'encoder')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{exp_name}_0.{file_format}')
        shutil.copyfile(quantized_model_path, output_path)
        new_quantized_model_info_path = output_path.replace(f'_0.{file_format}', '_info_0.json')
        quantized_model_utils.export_quantized_model_info(
            quantized_model_info,
            quantized_model_info['model_config'],
            quantized_model_info['precision_config'],
            new_quantized_model_info_path,
        )

        logger.info(f'Encoder already fixed shape. Found in: {output_folder}')

    # Merge cmdline lora input bin files
    if len(lora_shapes) > 0:
        logger.info('[fix_encoder_shape] Exporting encoder lora weight bins for cmdline')
        export_lora_weight_bins(
            output_folder,
            per_layer_dir,
            quantized_model_info,
            lora_shapes,
            0,
        )
        logger.info('[fix_encoder_shape] Done exporting encoder lora weight bins for cmdline')


def fix_infini_update_shape(
    llm_intermediate_quantized_model_path,
    partial_output_folder_without_shape,
    per_layer_dir,
    ptq_description,
    file_format,
    np_compat,
    cache_shapes,
    num_token,
):
    """Fixes the shape of the infini update TFLite or MLIR model.

    Currently only used by Infini Transformer.

    Args:
        llm_intermediate_quantized_model_path (str): The path to the chunked dynamic shape quantized model.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        per_layer_dir (str): The path to the per-layer dynamic shape quantized model directory.
        ptq_description (str): The custom tflite description from PTQ step. Default is ''.
        file_format (str): The file format of the quantized model. Either `tflite` or `mlir`.
            Default is `tflite`.
        np_compat (int): Whether to use backward neuropilot compatibility mode when exporting.
        cache_shapes (int): The target cache shapes when running llm shape fixing.
        num_token (int): The num_token for current llm to know prompt or gen mode.
    """
    logger.info(f'{llm_intermediate_quantized_model_path}, {per_layer_dir}, {partial_output_folder_without_shape}')

    infini_update_dir = os.path.join(per_layer_dir, '../', 'infini_update')
    if not os.path.exists(infini_update_dir):
        logger.error(
            f'[fix_infini_update_shape] Infini update quantized model directory {infini_update_dir} does not exist.'
        )
        return

    infini_update_quantized_path = utils.get_sorted_path_list(infini_update_dir, ['.tflite', '.mlir'], sep=None)
    if len(infini_update_quantized_path) > 1 or len(infini_update_quantized_path) == 0:
        logger.error(
            f'[fix_infini_update_shape] Expected 1 infini update quantized model, get {infini_update_quantized_path}'
        )
        return

    infini_update_quantized_path = infini_update_quantized_path[0]

    infini_update_quantized_model_info = quantized_model_utils.extract_infini_update_quantized_model_info(
        infini_update_quantized_path
    )

    # check if current shape is done
    cache_size = cache_shapes[0][2]
    adjusted_cache_size = cache_size - infini_update_quantized_model_info['model_config']['infini_window_size']
    if num_token == 1:
        adjusted_cache_size -= infini_update_quantized_model_info['model_config'].get('infini_max_gen_length', 0)
    logger.info(f'[fix_infini_update_shape] cache size: {adjusted_cache_size}')
    output_folder = os.path.join(partial_output_folder_without_shape, 'infini_update', f'{adjusted_cache_size}c')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder,
        f'{os.path.basename(infini_update_quantized_path).split(".")[0]}_{adjusted_cache_size}c.{file_format}',
    )

    if os.path.exists(output_path):
        logger.info(
            f'[fix_infini_update_shape] Infini update model with cache size {adjusted_cache_size} already exists. '
            'Skipping.'
        )
        return

    # do preprocessing
    subgraph, model = quantized_model_utils.get_subgraph_from_quantized_model(
        infini_update_quantized_path, return_nrpmodel=True
    )

    logger.debug(f'[fix_infini_update_shape] Infini update, quantized_model_info={infini_update_quantized_model_info}')
    quantized_model_description = model.description
    logger.debug(f'[fix_infini_update_shape] description={quantized_model_description}')

    input_name_remap = OrderedDict()
    for i in range(infini_update_quantized_model_info['model_config']['num_hidden_layers']):
        input_name_remap[subgraph.inputs[4 * i]] = f'past_key_{i}'
        input_name_remap[subgraph.inputs[4 * i + 1]] = f'past_value_{i}'
        input_name_remap[subgraph.inputs[4 * i + 2]] = f'past_mem_{i}'
        input_name_remap[subgraph.inputs[4 * i + 3]] = f'past_z_{i}'
    logger.debug(f'[fix_infini_update_shape] input_name_remap={input_name_remap}')

    for inp_idx, (src_name, dst_name) in enumerate(input_name_remap.items()):
        if src_name != dst_name:
            logger.debug(f'[fix_infini_update_shape] Rename input tensor {src_name} to {dst_name}')
            subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
            del subgraph.tensor_map[src_name]
            subgraph.inputs[inp_idx] = dst_name

    output_name_remap = OrderedDict()
    for i in range(infini_update_quantized_model_info['model_config']['num_hidden_layers']):
        output_name_remap[subgraph.outputs[2 * i]] = f'curr_mem_{i}'
        output_name_remap[subgraph.outputs[2 * i + 1]] = f'curr_z_{i}'
    logger.debug(f'[fix_infini_update_shape] output_name_remap={output_name_remap}')

    for out_idx, (src_name, dst_name) in enumerate(output_name_remap.items()):
        if src_name != dst_name:
            logger.debug(f'[fix_infini_update_shape] Rename output tensor {src_name} to {dst_name}')
            subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
            del subgraph.tensor_map[src_name]
            subgraph.outputs[out_idx] = dst_name

    for op in subgraph.operators:
        for inp_idx, op_inp in enumerate(op.inputs):
            if op_inp in input_name_remap:
                op.inputs[inp_idx] = input_name_remap[op_inp]
        for out_idx, op_out in enumerate(op.outputs):
            if op_out in output_name_remap:
                op.outputs[out_idx] = output_name_remap[op_out]

    builder = ModelBuilder()
    builder.import_subgraph(model)

    output_dest_path = os.path.join(
        os.path.dirname(llm_intermediate_quantized_model_path), f'infini_update.{file_format}'
    )
    input_names = list(input_name_remap.values())
    output_names = list(output_name_remap.values())

    logger.debug(f'[fix_infini_update_shape] Finalized Input Names:\n{input_names}')
    logger.debug(f'[fix_infini_update_shape] Finalized Output Names:\n{output_names}\n')

    out_nrpmodel = builder.export(input_names, output_names)
    quantized_model_utils.export_quantized_model(out_nrpmodel, output_dest_path, np_compat=np_compat)

    subgraph = quantized_model_utils.get_subgraph_from_quantized_model(output_dest_path)

    editor = (
        mtk_converter.TFLiteEditor(output_dest_path)
        if file_format == 'tflite'
        else mtk_converter.MlirEditor(output_dest_path)
    )

    # change shapes to not include infini window
    target_shapes = []
    for i in range(len(cache_shapes)):
        if i % 4 == 0 or i % 4 == 1:
            target_shapes.append(cache_shapes[i].copy())
            target_shapes[-1][2] = adjusted_cache_size
        else:
            target_shapes.append(cache_shapes[i])

    logger.debug(f'[fix_infini_update_shape] target_shapes={target_shapes}')

    editor.reconfigure_input_shapes(target_shapes)

    op_export_spec = utils.get_op_export_spec(np_compat, file_format)

    if file_format == 'tflite':
        if ptq_description != '':
            custom_description = ptq_description.rsplit('\n', 1)[1]
            custom_description = custom_description + f'\nShape fixed by mtk_llm_sdk v{__version__}'
        else:
            custom_description = 'Post Training Quantized by unknown version of mtk_llm_sdk\n'
            f'Shape fixed by mtk_llm_sdk v{__version__}'
        try:
            editor.export(output_path, tflite_op_export_spec=op_export_spec, custom_description=custom_description)
        except TypeError:
            editor.export(output_path, tflite_op_export_spec=op_export_spec)
    else:
        editor.export(output_path, export_spec=op_export_spec)

    # Generate fixed shape quantized model info json
    infini_update_quantized_model_info['batch_size'] = cache_shapes[0][0]
    new_quantized_model_info_path = output_path.replace(f'.{file_format}', '_info.json')
    quantized_model_utils.export_quantized_model_info(
        infini_update_quantized_model_info,
        infini_update_quantized_model_info['model_config'],
        infini_update_quantized_model_info['precision_config'],
        new_quantized_model_info_path,
    )


def fix_llm_shape(
    quantized_model_path,
    exp_name,
    partial_output_folder_without_shape,
    per_layer_dir,
    shapes,
    quantized_model_info,
    chunk_idx,
    lora_shapes=None,
    ptq_description='',
    file_format='tflite',
    np_compat=None,
    use_single_bmm_attention=False,
    per_op=False,
    sink_rope=False,
    cache_evict_attributes=None,
    output_names=None,
    clear_attn_outputs=False,
    bita_decode_token=None,
):
    """Fixes the shape of the constructed LLM TFLite or MLIR model.

    Args:
        quantized_model_path (str): The path to the chunked dynamic shape quantized model.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        per_layer_dir (str): The path to the per-layer dynamic shape quantized model directory.
        shapes (set): The set of shapes to fix.
        quantized_model_info (dict): The quantized model info dictionary containing all the dynamic quantized model
            information.
        chunk_idx (int): The index of the current chunk to shape fix.
        lora_shapes (list, optional): The shapes of the LoRA inputs. Default is None.
        ptq_description (str, optional): The custom tflite description from PTQ step. Default is ''.
        file_format (str, optional): The file format of the quantized model. Either `tflite` or `mlir`.
            Default is `tflite`.
        np_compat (int, optional): Whether to use backward neuropilot compatibility mode when exporting.
        use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        per_op (bool, optional): Whether to extract per-op output quantized model. Default is False.
        sink_rope (bool, optional): Whether to split Rotary Embedding QK. If True, cache eviction
            will be used. Defaults to False.
        cache_evict_attributes (dict, optional): The cache eviction related attributes. Defaults to {'method': ''}.
        output_names (list, optional): Customized output names to export. Defaults to None.
        clear_attn_outputs (bool, optional): Whether to clear the GlobalSnapKV attn weight outputs. Defaults to False.
        bita_decode_token (list, optional): List of token positions that should use BiTA decoding. Defaults to None.

    Raises:
        FileNotFoundError: If the LLM TFLite or MLIR model is not found.
    """
    if cache_evict_attributes is None:
        cache_evict_attributes = {}
    if lora_shapes is None:
        lora_shapes = []
    if not os.path.exists(quantized_model_path):
        logger.error(
            f'[fix_llm_shape] {file_format} to reconfigure shape not found: {quantized_model_path}.',
            err=FileNotFoundError,
        )

    shapes = {x for x in shapes if 't' in x and 'c' in x}
    logger.debug(f'[fix_llm_shape] LLM shapes to fix: {shapes}')
    logger.info(f'[fix_llm_shape] Fixing {len(shapes)} shapes for chunk {chunk_idx}:')

    for shape in tqdm(shapes):
        logger.debug(f'[fix_llm_shape] curr shape={shape}')
        output_folder = os.path.join(partial_output_folder_without_shape, 'llm', f'llm_{shape}')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{exp_name}_{shape}_{chunk_idx}.{file_format}')
        batch_size = 1  # use token_size for llm batch
        num_token = int(re.findall(r'\d+t', shape)[0][:-1])
        modified_quantized_model_info = None
        if (
            (
                output_names is not None
                and cache_evict_attributes['method'] == 'GlobalSnapKV'
                and num_token < int(cache_evict_attributes.get('obs_window', 0))
            )
            or clear_attn_outputs
        ):  # Only remove attn_weights and attn_logits from the GlobalSnapKV 1t graph or when forced to
            final_output_names = []
            indices_to_remove = []  # remove relevant parts in quantized model info
            for idx, name in enumerate(output_names):
                if not ('attn_weights_' in name or 'attn_logits_' in name):
                    final_output_names.append(name)
                else:
                    indices_to_remove.append(idx)
        else:
            # assume that currently no other graph config would need to change their final outputs
            final_output_names = None
            indices_to_remove = None
        logger.debug(f'[fix_llm_shape] Final Output Names: {final_output_names}')
        logger.debug(f'[fix_llm_shape] Output Indices to Remove: {indices_to_remove}')

        editor = (
            mtk_converter.TFLiteEditor(quantized_model_path)
            if file_format == 'tflite'
            else mtk_converter.MlirEditor(quantized_model_path)
        )
        if len(lora_shapes) > 0:
            lora_shapes_unpacked = [x for xx in lora_shapes for x in xx]  # flatten lora shapes
            lora_rank = int(re.findall(r'\d+r', shape)[0][:-1])
            for idd in range(len(lora_shapes_unpacked)):
                lora_shapes_unpacked[idd][0] = 1
                for i, dim in enumerate(lora_shapes_unpacked[idd]):
                    if dim is None:
                        lora_shapes_unpacked[idd][i] = lora_rank
        else:
            lora_shapes_unpacked = []

        assert len(quantized_model_info['num_attention_heads']) == quantized_model_info['num_layers']
        cache_size = int(re.findall(r'\d+c', shape)[0][:-1])
        if quantized_model_info['model_config'].get('infini_attention', False) and num_token == 1:
            cache_size += quantized_model_info['model_config'].get('infini_max_gen_length', 0)

        if quantized_model_info['model_config'].get('use_split_mask', False):
            normal_mask = [batch_size, 1, num_token, num_token]
        else:
            normal_mask = [batch_size, 1, num_token, num_token + cache_size]

        target_shapes = [
            [batch_size, num_token, quantized_model_info['hidden_size']],  # Input embeds
            normal_mask,
        ]
        if quantized_model_info['model_config'].get('use_split_mask', False):
            # split mask
            if bita_decode_token is not None and num_token in bita_decode_token:
                target_shapes.append(
                    [
                        batch_size,
                        1,
                        num_token,
                        cache_size,
                    ]
                )
                quantized_model_info['bita_decode_token'] = num_token
            else:
                target_shapes.append(
                    [
                        batch_size,
                        1,
                        1,
                        cache_size,
                    ]
                )

        if (
            quantized_model_info['model_type'] == 'whisper_decoder' or quantized_model_info['model_type'] == 'gecko'
        ) and 0 in quantized_model_info['layer_ids']:
            target_shapes = [
                *target_shapes,
                [batch_size, num_token, quantized_model_info['hidden_size']],  # Pos emb
            ]
        if 'whisper_decoder' in quantized_model_info['model_type']:
            for i in range(quantized_model_info['num_layers']):
                target_shapes = [
                    *target_shapes,
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        cache_size,
                        quantized_model_info['head_dim'],
                    ],
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        cache_size,
                        quantized_model_info['head_dim'],
                    ],
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        quantized_model_info['model_config']['max_source_positions'],
                        quantized_model_info['head_dim'],
                    ],  # cross_key
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        quantized_model_info['model_config']['max_source_positions'],
                        quantized_model_info['head_dim'],
                    ],  # cross_value
                ]
        else:
            if 'gecko' in quantized_model_info['model_type']:  # both gecko and gecko2 are using same rot_dim
                rot_dim = int(quantized_model_info['head_dim'] / 2)
            elif 'phi4' in quantized_model_info['model_type']:  # phi4 uses partial rotary embedding
                rot_dim = int(
                    quantized_model_info['head_dim']
                    * quantized_model_info['model_config'].get('partial_rotary_factor', 1.0)
                )
            else:
                rot_dim = quantized_model_info['head_dim']

            # FIXME: cache eviction + single_bmm_attention
            if sink_rope:
                target_shapes = [
                    *target_shapes,
                    [batch_size, 1, num_token, rot_dim]  # q cos
                    if not use_single_bmm_attention
                    else [batch_size, num_token, 1, rot_dim],
                    [batch_size, 1, num_token, rot_dim]  # q sin
                    if not use_single_bmm_attention
                    else [batch_size, num_token, 1, rot_dim],
                    [batch_size, 1, num_token + cache_size, rot_dim]  # k cos
                    if not use_single_bmm_attention
                    else [batch_size, num_token + cache_size, 1, rot_dim],
                    [batch_size, 1, num_token + cache_size, rot_dim]  # k sin
                    if not use_single_bmm_attention
                    else [batch_size, num_token + cache_size, 1, rot_dim],
                ]
            else:
                target_shapes = [
                    *target_shapes,
                    [batch_size, 1, num_token, rot_dim]
                    if not use_single_bmm_attention
                    else [batch_size, num_token, 1, rot_dim],  # Cos
                    [batch_size, 1, num_token, rot_dim]
                    if not use_single_bmm_attention
                    else [batch_size, num_token, 1, rot_dim],  # Sin
                ]
            # gecko2 per layer embed and fire pe
            if quantized_model_info['model_type'] == 'gecko2':
                # per layer embed
                cos_sin = target_shapes[-2:]
                target_shapes = target_shapes[:-2]
                is_ee = quantized_model_info['model_config']['early_exit_index'] is not None
                per_layer_emb_num_layers = (
                    quantized_model_info['model_config']['early_exit_index']
                    if is_ee
                    else quantized_model_info['num_layers']
                )
                target_shapes.append(
                    [
                        batch_size,
                        num_token,
                        per_layer_emb_num_layers,
                        quantized_model_info['model_config']['d_per_layer_embedding'],
                    ]
                )
                # fire pe
                num_layers = (
                    quantized_model_info['model_config']['early_exit_index']
                    + quantized_model_info['model_config']['early_exit_num_layers']
                    if is_ee
                    else quantized_model_info['num_layers']
                )
                total_fire_pe_count = num_layers // len(
                    quantized_model_info['model_config']['global_local_attention_pattern']
                )

                if set(quantized_model_info['layer_ids']).isdisjoint(
                    set(quantized_model_info['model_config']['fire_pe_index'])
                ):
                    target_shapes = target_shapes + cos_sin
                elif set(quantized_model_info['layer_ids']).issubset(
                    set(quantized_model_info['model_config']['fire_pe_index'])
                ):
                    target_shapes.extend(
                        [
                            [total_fire_pe_count, 1, num_token, num_token + cache_size],
                            [total_fire_pe_count, 1, num_token, 1],
                        ]
                    )
                else:
                    target_shapes.extend(
                        [
                            [total_fire_pe_count, 1, num_token, num_token + cache_size],
                            [total_fire_pe_count, 1, num_token, 1],
                        ]
                    )
                    target_shapes = target_shapes + cos_sin
                    target_shapes.insert(2, target_shapes[1].copy())

            # cache
            cache_shapes = []
            for i in range(quantized_model_info['num_layers']):
                cache_shapes.append(
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        cache_size,
                        quantized_model_info['head_dim'],
                    ]
                )  # key
                cache_shapes.append(
                    [
                        batch_size,
                        quantized_model_info['num_key_value_heads'][i],
                        cache_size,
                        quantized_model_info['head_dim'],
                    ]
                )  # value

                # infini mem and z
                if quantized_model_info['model_config'].get('infini_attention', False):
                    # mem
                    cache_shapes.append(
                        [
                            batch_size,
                            quantized_model_info['num_key_value_heads'][i],
                            quantized_model_info['head_dim'],
                            quantized_model_info['head_dim'],
                        ]
                    )
                    # z
                    cache_shapes.append(
                        [
                            batch_size,
                            quantized_model_info['num_key_value_heads'][i],
                            1,
                            quantized_model_info['head_dim'],
                        ]
                    )
            other_mask_shapes = []
            if quantized_model_info['model_config'].get('infini_attention', False):
                # infini mask
                other_mask_shapes.append(
                    [
                        batch_size,
                        quantized_model_info['model_config']['num_key_value_heads'],
                        num_token,
                        quantized_model_info['head_dim'],
                    ]
                )

            target_shapes = target_shapes + cache_shapes + other_mask_shapes
            # Patch F8: deepstack ds_padded_<i> inputs 同形于 inputs_embeds，按实际 model 中
            # 名为 'ds_padded_*' 的 input 个数加 shape，避免 reconfigure_input_shapes 报 mismatch。
            try:
                _sg = quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
                _ds_count = sum(1 for _inp in _sg.inputs if _inp.startswith('ds_padded'))
            except Exception:
                _ds_count = 0
            if _ds_count > 0:
                ds_padded_shape = [batch_size, num_token, quantized_model_info['hidden_size']]
                target_shapes = target_shapes + [list(ds_padded_shape) for _ in range(_ds_count)]
            target_shapes = target_shapes + lora_shapes_unpacked  # Lora inputs

        logger.debug(f'[fix_llm_shape] target_shapes={target_shapes}')

        editor.reconfigure_input_shapes(target_shapes)

        op_export_spec = utils.get_op_export_spec(np_compat, file_format)

        if file_format == 'tflite':
            if ptq_description != '':
                custom_description = ptq_description.rsplit('\n', 1)[1]
                custom_description = custom_description + f'\nShape fixed by mtk_llm_sdk v{__version__}'
            else:
                custom_description = 'Post Training Quantized by unknown version of mtk_llm_sdk\n'
                f'Shape fixed by mtk_llm_sdk v{__version__}'
            try:
                editor.export(
                    output_path,
                    tflite_op_export_spec=op_export_spec,
                    output_names=final_output_names,
                    custom_description=custom_description,
                )
            except TypeError:
                editor.export(output_path, tflite_op_export_spec=op_export_spec, output_names=final_output_names)
        else:
            editor.export(output_path, export_spec=op_export_spec, output_names=final_output_names)

        logger.debug(f'[fix_llm_shape] Exported LLM Chunk {chunk_idx} to {output_path}')

        if per_op:
            extract_per_op_quantized_model(output_path)

        # Generate fixed shape quantized model info json
        quantized_model_info['batch_size'] = batch_size
        quantized_model_info['t'] = num_token
        quantized_model_info['c'] = cache_size
        if len(lora_shapes) > 0:
            quantized_model_info['lora_rank'] = lora_rank
        new_quantized_model_info_path = output_path.replace(f'{chunk_idx}.{file_format}', f'info_{chunk_idx}.json')

        if final_output_names is not None:
            output_scales = quantized_model_info['output_scales']
            output_zp = quantized_model_info['output_zero_points']
            output_dtypes = quantized_model_info['output_dtypes']
            modified_quantized_model_info = deepcopy(quantized_model_info)
            modified_quantized_model_info['output_scales'] = [
                output_scales[idx] for idx in range(len(output_scales)) if idx not in indices_to_remove
            ]
            modified_quantized_model_info['output_zero_points'] = [
                output_zp[idx] for idx in range(len(output_zp)) if idx not in indices_to_remove
            ]
            modified_quantized_model_info['output_dtypes'] = [
                output_dtypes[idx] for idx in range(len(output_dtypes)) if idx not in indices_to_remove
            ]
            logger.debug(f'\n[fix_llm_shape] Modified Quantized Model Info:\n{modified_quantized_model_info}\n')

            # sanitize quantized model info json
            quantized_model_utils.sanitize_quantized_model_info(output_path, modified_quantized_model_info)

            quantized_model_utils.export_quantized_model_info(
                modified_quantized_model_info,
                modified_quantized_model_info['model_config'],
                modified_quantized_model_info['precision_config'],
                new_quantized_model_info_path,
            )

        else:
            # sanitize quantized model info json
            quantized_model_utils.sanitize_quantized_model_info(output_path, quantized_model_info)

            quantized_model_utils.export_quantized_model_info(
                quantized_model_info,
                quantized_model_info['model_config'],
                quantized_model_info['precision_config'],
                new_quantized_model_info_path,
            )

        # Merge cmdline lora input bin files
        if len(lora_shapes) > 0:
            logger.info('[fix_llm_shape] Exporting LLM lora weight bins for cmdline')
            export_lora_weight_bins(
                output_folder,
                per_layer_dir,
                quantized_model_info if modified_quantized_model_info is None else modified_quantized_model_info,
                lora_shapes,
                chunk_idx,
            )
            logger.info('[fix_llm_shape] Done exporting LLM lora weight bins for cmdline')

        if quantized_model_info['model_config'].get('infini_attention', False):
            fix_infini_update_shape(
                quantized_model_path,
                partial_output_folder_without_shape,
                per_layer_dir,
                ptq_description,
                file_format,
                np_compat,
                cache_shapes,
                num_token,
            )


def fix_tail_shape(
    quantized_model_path,
    exp_name,
    partial_output_folder_without_shape,
    shapes,
    quantized_model_info,
    chunk_idx,
    ptq_description='',
    file_format='tflite',
    np_compat=None,
    per_op=False,
):
    """Fixes the shape of the tail layer in the TFLite or MLIR model.

    Args:
        quantized_model_path (str): The path to the TFLite or MLIR model.
        exp_name (str): The experiment name.
        partial_output_folder_without_shape (str): The partial output folder path without shape.
        shapes (set): The set of shapes to fix.
        quantized_model_info (dict): The quantized model info dictionary containing all the dynamic quantized model
            information.
        chunk_idx (int): The index of the current chunk to shape fix.
        ptq_description (str, optional): The custom quantized model description from PTQ step. Default is ''.
        file_format (str, optional): The file format of the quantized model. Either `tflite` or `mlir`.
            Default is `tflite`.
        np_compat (int, optional): Whether to use backward neuropilot compatibility mode when exporting.
        per_op (bool, optional): Whether to extract per-op output quantized model. Default is False.

    Raises:
        FileNotFoundError: If the TFLite or MLIR model is not found.
    """
    if not os.path.exists(quantized_model_path):
        logger.error(
            f'[fix_tail_shape] {file_format} to reconfigure shape not found: {quantized_model_path}.',
            err=FileNotFoundError,
        )

    shapes = {x for x in shapes if 't' in x and 'c' in x}
    logger.debug(f'[fix_tail_shape] Tail shapes to fix: {shapes}')

    for shape in tqdm(shapes):
        output_folder = os.path.join(partial_output_folder_without_shape, 'llm', f'llm_{shape}')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{exp_name}_{shape}_{chunk_idx}.{file_format}')
        batch_size = 1
        num_token = int(re.findall(r'\d+t', shape)[0][:-1])

        editor = (
            mtk_converter.TFLiteEditor(quantized_model_path)
            if file_format == 'tflite'
            else mtk_converter.MlirEditor(quantized_model_path)
        )

        if quantized_model_info['tail'] == 'tail':
            target_shapes = [[batch_size, num_token, quantized_model_info['hidden_size']]]
        else:
            assert quantized_model_info['tail'] == 'eagle'
            cache_size = int(re.findall(r'\d+c', shape)[0][:-1])
            target_shapes = [
                [batch_size, num_token, quantized_model_info['hidden_size']],  # Input embeds
                [batch_size, num_token, quantized_model_info['hidden_size']],  # Hidden states
                [batch_size, 1, num_token, num_token + cache_size],  # Mask
            ]
            target_shapes = [
                *target_shapes,
                [batch_size, 2, num_token, quantized_model_info['head_dim']],  # Rot emb
            ]
            target_shapes = target_shapes + [
                [
                    batch_size,
                    quantized_model_info['num_key_value_heads'][i],
                    cache_size,
                    quantized_model_info['head_dim'],
                ]
                for i in range(quantized_model_info['num_layers'])
            ]  # Key cache
            target_shapes = target_shapes + [
                [
                    batch_size,
                    quantized_model_info['num_key_value_heads'][i],
                    cache_size,
                    quantized_model_info['head_dim'],
                ]
                for i in range(quantized_model_info['num_layers'])
            ]  # Value cache

        editor.reconfigure_input_shapes(target_shapes)

        op_export_spec = utils.get_op_export_spec(np_compat, file_format)

        if file_format == 'tflite':
            if ptq_description != '':
                custom_description = ptq_description.rsplit('\n', 1)[1]
                custom_description = custom_description + f'\nShape fixed by mtk_llm_sdk v{__version__}'
            else:
                custom_description = 'Post Training Quantized by unknown version of mtk_llm_sdk\n'
                f'Shape fixed by mtk_llm_sdk v{__version__}'
            try:
                editor.export(output_path, tflite_op_export_spec=op_export_spec, custom_description=custom_description)
            except TypeError:
                editor.export(output_path, tflite_op_export_spec=op_export_spec)
        else:
            editor.export(output_path, export_spec=op_export_spec)

        logger.debug(f'Exported Separate Tail to {output_path}')

        if per_op:
            extract_per_op_quantized_model(output_path)

        # Generate fixed shape quantized model info json
        quantized_model_info['batch_size'] = batch_size
        quantized_model_info['t'] = num_token
        new_quantized_model_info_path = output_path.replace(f'{chunk_idx}.{file_format}', f'info_{chunk_idx}.json')
        quantized_model_utils.export_quantized_model_info(
            quantized_model_info,
            quantized_model_info['model_config'],
            quantized_model_info['precision_config'],
            new_quantized_model_info_path,
        )


def extract_per_op_quantized_model(quantized_model_path):
    """Extracts per-operation output quantized model from the given reference quantized model.

    Args:
        quantized_model_path (str): The path to the reference quantized model.
    """
    subgraph, model = quantized_model_utils.get_subgraph_from_quantized_model(
        quantized_model_path, return_nrpmodel=True
    )

    builder = ModelBuilder()
    builder.import_subgraph(model)

    input_names = subgraph.inputs
    all_act_names = subgraph.outputs
    for op in subgraph.operators:
        for op_out in op.outputs:
            if op_out not in all_act_names:
                all_act_names.append(op_out)

    out_nrpmodel = builder.export(input_names, all_act_names)
    subgraph = out_nrpmodel.subgraphs[0]

    quantized_model_utils.export_quantized_model(out_nrpmodel, quantized_model_path)


@memory_peak_profile
def main(args=None):
    """Main function.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    quantized_model_list, perm, custom_tail_type, quantized_model_infos = args_sanity_checks(args)
    if perm:
        args.shapes = permutate_shapes(args.shapes)
    if not isinstance(args.shapes, set):
        args.shapes = set(args.shapes)
    exp_name = os.path.basename(args.quantized_model_folder.rstrip('/'))
    if args.lora_config is not None:
        # just take the first lora config name
        lora_name = utils.get_dirpath(args.lora_config[0], full=False)
        exp_name += f'_{lora_name}'
        if len(args.lora_config) > 1:
            exp_name += '_combined_lora_range'
    if args.output_folder is None:
        args.output_folder = 'quantized_models'
        partial_output_folder_without_shape = os.path.join(args.output_folder, f'{exp_name}')
    else:
        partial_output_folder_without_shape = args.output_folder

    # FIXME: if infini attention, need to modify cache size, wondering if there is any better way to do this
    if not args.encoder_only and quantized_model_infos['llm'][0]['model_config'].get('infini_attention', False):
        new_shapes = set()
        for shape in args.shapes:
            if 'c' in shape:
                cache_size = int(re.findall(r'\d+c', shape)[0][:-1])
                real_cache_size = (
                    cache_size
                    + int(quantized_model_infos['llm'][0]['model_config']['infini_sink_size'])
                    + int(quantized_model_infos['llm'][0]['model_config']['infini_window_size'])
                )
                new_shape = re.sub(r'\d+c', f'{real_cache_size!s}c', shape)
                new_shapes.add(new_shape)
            else:
                new_shapes.add(shape)
        args.shapes = new_shapes

    # Additional sanity checks
    enc_shapes = {x for x in args.shapes if 'e' in x}
    for shape in tqdm(enc_shapes):
        output_folder = os.path.join(partial_output_folder_without_shape, 'encoder', f'encoder_{shape}')
        if os.path.exists(output_folder) and not args.force_overwrite:
            logger.error(
                f'Output folder for shape {shape} already exists: {output_folder}. Please manually delete it '
                'or use --force_overwrite if you wish to overwrite the model inside.',
                err=FileExistsError,
            )
    llm_shapes = {x for x in args.shapes if 't' in x and 'c' in x}
    for shape in tqdm(llm_shapes):
        output_folder = os.path.join(partial_output_folder_without_shape, 'llm', f'llm_{shape}')
        if os.path.exists(output_folder) and not args.force_overwrite:
            logger.error(
                f'Output folder for shape {shape} already exists: {output_folder}. Please manually delete it '
                'or use --force_overwrite if you wish to overwrite the model inside.',
                err=FileExistsError,
            )
    print_args(args, partial_output_folder_without_shape)

    intermediate_folder = os.path.join('temp', exp_name)  # Used to store dynamic shape chunked quantized models
    utils.recursive_remove_if_exist(intermediate_folder, recreate=True)

    if not args.encoder_only:
        cache_evict_attributes = {'method': ''}
        if (
            'extra_input' not in quantized_model_infos['llm'][0]['model_config']
            or 'extra_output' not in quantized_model_infos['llm'][0]['model_config']
        ):
            # For backward compatibility
            sink_rope = False
            get_attn_logits = False
            get_attn_weights = False
        else:
            sink_rope = quantized_model_infos['llm'][0]['model_config']['extra_input'].get('sink_rope', False)
            get_attn_logits = quantized_model_infos['llm'][0]['model_config']['extra_output'].get('attn_logits', False)
            get_attn_weights = quantized_model_infos['llm'][0]['model_config']['extra_output'].get(
                'attn_weights', False
            )
            if quantized_model_infos['llm'][0]['model_config'].get('cache_evict'):
                cache_evict = quantized_model_infos['llm'][0]['model_config']['cache_evict']

                # Compatable with old tflite with 'cache_evict' string
                if isinstance(cache_evict, str):
                    cache_evict_attributes['method'] = cache_evict
                    cache_evict_attributes['obs_window'] = quantized_model_infos['llm'][0]['model_config'].get(
                        'obs_window', 0
                    )
                else:
                    cache_evict_attributes = cache_evict

        if args.lora_config is not None:
            model_config_path = os.path.join(args.quantized_model_folder, 'config.json')
            pipeline = FloatPipeline(model_config_path, args.lora_config, task='hotplug')

    with utils.temp_file(intermediate_folder):
        if args.encoder_only:
            file_format = 'tflite' if quantized_model_list['encoder'][0].endswith('.tflite') else 'mlir'
        else:
            file_format = 'tflite' if quantized_model_list['llm'][0].endswith('.tflite') else 'mlir'

        file_format_separator = '/' if file_format == 'tflite' else '.'

        if len(quantized_model_list['encoder']) > 0:
            output_dest_path = os.path.join(intermediate_folder, f'encoder.{file_format}')
            logger.debug('Has encoder')
            logger.info('Constructing Encoder:')
            enc_chunked_model_info = shape_fix_encoder(
                args,
                exp_name,
                partial_output_folder_without_shape,
                output_dest_path,
                quantized_model_list['encoder'],
                quantized_model_infos['encoder'],
                file_format_separator,
            )

        if args.encoder_only:
            for shape in args.shapes:
                output_folder = os.path.join(partial_output_folder_without_shape, 'encoder', f'encoder_{shape}')
                logger.info(f'Encoder {shape} -> {output_folder}')

            # Copy config.json to fixed-shaped quantized model folder
            shutil.copyfile(
                os.path.join(args.quantized_model_folder, 'config.json'),
                os.path.join(partial_output_folder_without_shape, 'config.json'),
            )

            with contextlib.suppress(OSError):
                os.rmdir('temp')

            return

        if len(quantized_model_list['embedder']) > 0:
            output_dest_path = os.path.join(intermediate_folder, f'embedder.{file_format}')
            shape_fix_embedder(
                args,
                exp_name,
                partial_output_folder_without_shape,
                output_dest_path,
                quantized_model_list['embedder'],
                quantized_model_infos['embedder'],
            )

        num_decoder_layers = len(quantized_model_list['llm']) - 1 - int(custom_tail_type is not None)
        if len(args.num_chunks_or_layers_per_chunk) == 1:
            num_layers_per_chunk = utils.evenly_distribute(num_decoder_layers, args.num_chunks_or_layers_per_chunk[0])
        else:
            num_layers_per_chunk = args.num_chunks_or_layers_per_chunk

        if args.separate_tail:
            num_layers_per_chunk = [*num_layers_per_chunk, 0]
            if custom_tail_type is not None:
                if custom_tail_type == 'eagle':
                    num_layers_per_chunk = [*num_layers_per_chunk, 1]
                elif custom_tail_type == 'medusa':
                    num_layers_per_chunk = [*num_layers_per_chunk, 0]
                else:
                    logger.error(f'Expected custom tail to be either eagle or medusa, but got {custom_tail_type}')

        logger.info(
            'Number of decoder layers per chunk '
            f'{"(0 means only lm_head/medusa tail)" if 0 in num_layers_per_chunk else ""}: {num_layers_per_chunk}'
        )

        curr_chunk_idx = 0
        inner_layer_idx = 0  # tracks layer idx for current chunk
        lora_inp_shapes = []
        lora_inp_names = []
        num_key_heads = []
        num_value_heads = []
        curr_chunk_quantized_model_infos = []
        gecko = None
        whisper = None
        prev_output_scale = None
        prev_output_zp = None
        prev_output_dtype = None
        quantized_model_description = None

        logger.info('Constructing LLM chunk 0:')
        pbar = tqdm(total=num_layers_per_chunk[curr_chunk_idx] + int(len(num_layers_per_chunk) == 1))

        for layer_idx, dyn_file_path in enumerate(quantized_model_list['llm']):
            subgraph, model = quantized_model_utils.get_subgraph_from_quantized_model(
                dyn_file_path, return_nrpmodel=True
            )
            quantized_model_info = quantized_model_infos['llm'][layer_idx]
            curr_chunk_quantized_model_infos.append(quantized_model_info)
            logger.debug(f'layer index {layer_idx}, quantized_model_info={quantized_model_info}')

            # add restriction for gecko2 so that it can only have one chunk
            if quantized_model_info['model_type'] == 'gecko2' and len(args.num_chunks_or_layers_per_chunk) != 1:
                logger.error('Currently gecko2 only supports 1 chunk (-n 1)')

            # Inter-chunk quant param alignment
            if prev_output_scale is not None:
                assert prev_output_zp is not None
                if quantized_model_info['input_dtypes'][0] != prev_output_dtype:
                    logger.error(
                        f'Layer {layer_idx} input dtype ({quantized_model_info["input_dtypes"][0]}) does not match '
                        f'layer {layer_idx - 1} output dtype ({prev_output_dtype})'
                    )
                logger.debug(
                    f'Aligning hidden states quant params of layer {layer_idx} input ({subgraph.inputs[0]}) '
                    f'to layer {layer_idx - 1} output.'
                )
                inp_tensor = subgraph.tensor_map[subgraph.inputs[0]]
                logger.debug(f'Scale change: {inp_tensor.quant_param.linear.scale_vals[0]} -> {prev_output_scale}')
                inp_tensor.quant_param.linear.scale_vals[0] = prev_output_scale
                logger.debug(f'ZP change: {inp_tensor.quant_param.linear.zero_point_vals[0]} -> {prev_output_zp}')
                inp_tensor.quant_param.linear.zero_point_vals[0] = prev_output_zp
                final_qmin, final_qmax, _ = type_utils.get_quant_value_range(
                    type_utils.get_nrp_type(quantized_model_info['input_dtypes'][0])
                )
                final_min_vals = utils.dequantize(final_qmin, prev_output_scale, prev_output_zp)
                final_max_vals = utils.dequantize(final_qmax, prev_output_scale, prev_output_zp)
                inp_tensor.quant_param.linear.min_vals[0] = final_min_vals
                inp_tensor.quant_param.linear.max_vals[0] = final_max_vals
                quantized_model_info['input_scales'][0] = prev_output_scale
                quantized_model_info['input_zero_points'][0] = prev_output_zp

            # don't need to get q param info when you are at last layer
            if layer_idx < len(quantized_model_list['llm']) - 1:
                prev_output_scale = quantized_model_info['output_scales'][0]
                prev_output_zp = quantized_model_info['output_zero_points'][0]
                prev_output_dtype = quantized_model_info['output_dtypes'][0]
            # At this point separate tail q params have been set

            if (
                quantized_model_info['model_type'] == 'whisper_decoder'
                and layer_idx != (len(quantized_model_list['llm']) - 1)
                and quantized_model_info['precision_config']['encoder_precision'] not in ['dynamic_quant', 'FP']
            ):
                # Do cross attention alignment here
                idx = 4 if layer_idx == 0 else 3
                for k in range(1, 3):
                    cross_tensor = subgraph.tensor_map[subgraph.inputs[idx + k]]
                    k_cross_scale = enc_chunked_model_info['output_scales'][layer_idx * 2 + k]
                    k_cross_zp = enc_chunked_model_info['output_zero_points'][layer_idx * 2 + k]

                    cross_tensor.quant_param.linear.scale_vals[0] = k_cross_scale
                    cross_tensor.quant_param.linear.zero_point_vals[0] = k_cross_zp
                    kfinal_qmin, kfinal_qmax, _ = type_utils.get_quant_value_range(
                        type_utils.get_nrp_type(enc_chunked_model_info['output_dtypes'][layer_idx * 2 + k])
                    )
                    kfinal_min_vals = utils.dequantize(kfinal_qmin, k_cross_scale, k_cross_zp)
                    kfinal_max_vals = utils.dequantize(kfinal_qmax, k_cross_scale, k_cross_zp)
                    cross_tensor.quant_param.linear.min_vals[0] = kfinal_min_vals
                    cross_tensor.quant_param.linear.max_vals[0] = kfinal_max_vals

            if args.lora_config is not None:
                lora_start_layer_idx = pipeline.lora_handler.global_llm_start_idx
                lora_end_layer_idx = pipeline.lora_handler.global_llm_end_idx
                if layer_idx >= lora_start_layer_idx and layer_idx <= lora_end_layer_idx:
                    logger.debug('Adding dynamic LoRA inputs to quantized model.')
                    quantized_model_info = qat_utils.add_dynamic_lora_to_base_quantized_model(
                        subgraph,
                        quantized_model_info,
                        pipeline,
                        file_format_separator=file_format_separator,
                        lora_start_layer_idx=lora_start_layer_idx,
                        lora_end_layer_idx=lora_end_layer_idx,
                        buffer_scale=args.buffer_scale,
                    )

            if quantized_model_description is None:
                quantized_model_description = model.description
                logger.debug(f'description={quantized_model_description}')
            else:
                if model.description != quantized_model_description:
                    logger.warning("current layer's description does not match previous layers' descriptions.")

            gecko = gecko or quantized_model_info['model_type'] == 'gecko'
            whisper = whisper or 'whisper_decoder' in quantized_model_info['model_type']
            curr_chunk_num_layer = num_layers_per_chunk[curr_chunk_idx]
            num_non_lora_inputs = quantized_model_info['num_non_lora_inputs']
            # FIXME: dont use this to determine start idx of num cache
            # Patch F: 减去 num_extra_inputs（如 deepstack ds_padded）以保持 cache 索引正确；
            # 默认 0，向下兼容旧模型 info JSON。
            num_non_cache_inputs = quantized_model_info['num_non_cache_inputs'] - quantized_model_info.get(
                'num_other_masks_inputs', 0
            ) - quantized_model_info.get('num_extra_inputs', 0)

            # rename all non-lora inputs
            input_name_remap = OrderedDict()
            if quantized_model_info['tail'] == 'eagle':
                for j in range(num_non_lora_inputs):
                    if j == 0:
                        input_name_remap[subgraph.inputs[j]] = 'input_embeds'
                    elif j == 1:
                        input_name_remap[subgraph.inputs[j]] = 'hidden_states'
                    elif j == 2:
                        input_name_remap[subgraph.inputs[j]] = 'mask'
                    elif j == 3:
                        input_name_remap[subgraph.inputs[j]] = 'cos'
                    elif j == 4:
                        input_name_remap[subgraph.inputs[j]] = 'sin'
                    elif j == 5:
                        input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                    elif j == 6:
                        input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                    else:
                        logger.error('Too many inputs for eagle tail!')
            else:
                # only separate tail itself does not need remap
                if not args.separate_tail or (args.separate_tail and quantized_model_info['tail'] is None):
                    for j in range(num_non_lora_inputs):
                        if j == 0:
                            input_name_remap[subgraph.inputs[j]] = 'input_embeds'
                        elif j == 1 and quantized_model_info['model_type'] != 'gecko2':
                            input_name_remap[subgraph.inputs[j]] = 'mask'
                        else:
                            if quantized_model_info['model_type'] == 'gecko' and 0 in quantized_model_info['layer_ids']:
                                if j == 2:
                                    input_name_remap[subgraph.inputs[j]] = 'pos_emb'
                                elif j == 3:
                                    input_name_remap[subgraph.inputs[j]] = 'cos'
                                elif j == 4:
                                    input_name_remap[subgraph.inputs[j]] = 'sin'
                                elif j == 5:
                                    input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                                elif j == 6:
                                    input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                                else:
                                    logger.error('Too many non-lora inputs!')
                            elif 'whisper_decoder' in quantized_model_info['model_type']:
                                if 0 in quantized_model_info['layer_ids']:
                                    if j == 2:
                                        input_name_remap[subgraph.inputs[j]] = 'pos_emb'
                                    elif j == 3:
                                        input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                                    elif j == 4:
                                        input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                                    elif j == 5:
                                        input_name_remap[subgraph.inputs[j]] = f'cross_keys_{inner_layer_idx}'
                                    elif j == 6:
                                        input_name_remap[subgraph.inputs[j]] = f'cross_values_{inner_layer_idx}'
                                    else:
                                        logger.error('Too many non-lora inputs!')
                                else:
                                    if j == 2:
                                        input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                                    elif j == 3:
                                        input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                                    elif j == 4:
                                        input_name_remap[subgraph.inputs[j]] = f'cross_keys_{inner_layer_idx}'
                                    elif j == 5:
                                        input_name_remap[subgraph.inputs[j]] = f'cross_values_{inner_layer_idx}'
                            elif quantized_model_info['model_type'] == 'gecko2':
                                if set(quantized_model_info['layer_ids']).isdisjoint(
                                    set(quantized_model_info['model_config']['fire_pe_index'])
                                ):
                                    input_name_remap_list = [
                                        'swa_mask',
                                        'per_layer_emb',
                                        'cos',
                                        'sin',
                                        f'past_keys_{inner_layer_idx}',
                                        f'past_values_{inner_layer_idx}',
                                    ]
                                else:
                                    input_name_remap_list = [
                                        'global_mask',
                                        'per_layer_emb',
                                        'fire_pe_relative_pos',
                                        'fire_pe_query_pos',
                                        f'past_keys_{inner_layer_idx}',
                                        f'past_values_{inner_layer_idx}',
                                    ]
                                # Note: this can be written like this because ptq is per layer
                                is_ee = quantized_model_info['model_config']['early_exit_index'] is not None
                                if (
                                    is_ee
                                    and quantized_model_info['layer_ids'][0]
                                    >= quantized_model_info['model_config']['early_exit_index']
                                ):
                                    # remove per layer embedding
                                    input_name_remap_list.pop(1)
                                if j <= len(input_name_remap_list) and j >= 1:
                                    input_name_remap[subgraph.inputs[j]] = input_name_remap_list[j - 1]
                                else:
                                    logger.error('Too many non-lora inputs!')
                            else:
                                if sink_rope:
                                    if j == 2:
                                        input_name_remap[subgraph.inputs[j]] = 'q_cos'
                                    elif j == 3:
                                        input_name_remap[subgraph.inputs[j]] = 'q_sin'
                                    elif j == 4:
                                        input_name_remap[subgraph.inputs[j]] = 'k_cos'
                                    elif j == 5:
                                        input_name_remap[subgraph.inputs[j]] = 'k_sin'
                                    elif j == 6:
                                        input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                                    elif j == 7:
                                        input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                                    else:
                                        if quantized_model_info['model_config'].get('infini_attention', False):
                                            if j == 8:
                                                input_name_remap[subgraph.inputs[j]] = f'mem_{inner_layer_idx}'
                                            elif j == 9:
                                                input_name_remap[subgraph.inputs[j]] = f'z_{inner_layer_idx}'
                                            elif j == 10:
                                                input_name_remap[subgraph.inputs[j]] = 'infini_mask'
                                            else:
                                                logger.error('Too many non-lora inputs!')

                                        # FIXME: does not support infini alongside split mask for now
                                        elif quantized_model_info['model_config'].get('use_split_mask', False):
                                            if j == 8:
                                                input_name_remap[subgraph.inputs[j]] = 'split_mask'
                                            else:
                                                logger.error('Too many non-lora inputs!')
                                        else:
                                            logger.error('Too many non-lora inputs!')
                                else:
                                    if j == 2:
                                        input_name_remap[subgraph.inputs[j]] = 'cos'
                                    elif j == 3:
                                        input_name_remap[subgraph.inputs[j]] = 'sin'
                                    elif j == 4:
                                        input_name_remap[subgraph.inputs[j]] = f'past_keys_{inner_layer_idx}'
                                    elif j == 5:
                                        input_name_remap[subgraph.inputs[j]] = f'past_values_{inner_layer_idx}'
                                    else:
                                        if quantized_model_info['model_config'].get('infini_attention', False):
                                            if j == 6:
                                                input_name_remap[subgraph.inputs[j]] = f'mem_{inner_layer_idx}'
                                            elif j == 7:
                                                input_name_remap[subgraph.inputs[j]] = f'z_{inner_layer_idx}'
                                            elif j == 8:
                                                input_name_remap[subgraph.inputs[j]] = 'infini_mask'
                                            else:
                                                logger.error('Too many non-lora inputs!')

                                        # FIXME: does not support infini alongside split mask for now
                                        elif quantized_model_info['model_config'].get('use_split_mask', False):
                                            if j == 6:
                                                input_name_remap[subgraph.inputs[j]] = 'split_mask'
                                            elif j == 7 and quantized_model_info.get('num_extra_inputs', 0) >= 1:
                                                # Patch F: ds_padded 用 chunk-specific 命名，让各 chunk 各自暴露成独立 model input
                                                input_name_remap[subgraph.inputs[j]] = f'ds_padded_{inner_layer_idx}'
                                            else:
                                                logger.error('Too many non-lora inputs!')
                                        else:
                                            logger.error('Too many non-lora inputs!')

            # only separate tail itself does not need this
            if not args.separate_tail or (args.separate_tail and quantized_model_info['tail'] is None):
                # rename all lora inputs
                for j in range(quantized_model_info['num_lora_inputs']):
                    if subgraph.inputs[num_non_lora_inputs + j].startswith('argument'):
                        input_name_remap[subgraph.inputs[num_non_lora_inputs + j]] = f'lora_input_{inner_layer_idx}_{j}'
                    else:
                        initial_str = file_format_separator.join(['layers', '0']) + file_format_separator
                        replacement_str = (
                            file_format_separator.join(['layers', str(inner_layer_idx)]) + file_format_separator
                        )
                        input_name_remap[subgraph.inputs[num_non_lora_inputs + j]] = subgraph.inputs[
                            num_non_lora_inputs + j
                        ].replace(initial_str, replacement_str)
                logger.debug(f'input_name_remap={input_name_remap}')

                output_name_remap = OrderedDict()
                output_name_remap[subgraph.outputs[0]] = f'hidden_states_{inner_layer_idx}'
                if quantized_model_info['tail'] is None:
                    output_name_remap[subgraph.outputs[1]] = f'curr_keys_{inner_layer_idx}'
                    output_name_remap[subgraph.outputs[2]] = f'curr_values_{inner_layer_idx}'

                    if len(subgraph.outputs) > 3:  # Exists other outputs
                        outs = subgraph.outputs[3:]
                        if get_attn_logits:
                            key = outs.pop(0)
                            output_name_remap[key] = f'attn_logits_{inner_layer_idx}'
                        if get_attn_weights:
                            key = outs.pop(0)
                            output_name_remap[key] = f'attn_weights_{inner_layer_idx}'

                elif quantized_model_info['tail'] == 'eagle':
                    output_name_remap[subgraph.outputs[1]] = 'curr_keys'
                    output_name_remap[subgraph.outputs[2]] = 'curr_values'

                logger.debug(f'output_name_remap={output_name_remap}')

                for inp_idx, (src_name, dst_name) in enumerate(input_name_remap.items()):
                    if src_name != dst_name:
                        logger.debug(f'Rename input tensor {src_name} to {dst_name}')
                        subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
                        del subgraph.tensor_map[src_name]
                        subgraph.inputs[inp_idx] = dst_name

                for out_idx, (src_name, dst_name) in enumerate(output_name_remap.items()):
                    if src_name != dst_name:
                        logger.debug(f'Rename output tensor {src_name} to {dst_name}')
                        subgraph.tensor_map[dst_name].CopyFrom(subgraph.tensor_map[src_name])
                        del subgraph.tensor_map[src_name]
                        subgraph.outputs[out_idx] = dst_name

                remainder_name_remap = {}
                for k in set(subgraph.tensor_map):
                    if k not in input_name_remap.values() and k not in output_name_remap.values():
                        initial_str = file_format_separator.join(['layers', '0']) + file_format_separator
                        replacement_str = (
                            file_format_separator.join(['layers', str(inner_layer_idx)]) + file_format_separator
                        )
                        suffix = k[9:] if k.startswith(initial_str) else k
                        new_name = replacement_str + suffix
                        if k != new_name:
                            logger.debug(f'Rename tensor {k} to {new_name}')
                            subgraph.tensor_map[new_name].CopyFrom(subgraph.tensor_map[k])
                            del subgraph.tensor_map[k]
                        remainder_name_remap[k] = new_name
                logger.debug(f'remainder_name_remap={remainder_name_remap}')

                for op in subgraph.operators:
                    for inp_idx, op_inp in enumerate(op.inputs):
                        if op_inp in input_name_remap:
                            op.inputs[inp_idx] = input_name_remap[op_inp]
                        elif op_inp in output_name_remap:
                            op.inputs[inp_idx] = output_name_remap[op_inp]
                        elif op_inp in remainder_name_remap:
                            op.inputs[inp_idx] = remainder_name_remap[op_inp]
                        else:
                            logger.error(f'{op_inp} is not in any of the name remaps')

                    for out_idx, op_out in enumerate(op.outputs):
                        if op_out in output_name_remap:
                            op.outputs[out_idx] = output_name_remap[op_out]
                        elif op_out in input_name_remap:
                            op.outputs[out_idx] = input_name_remap[op_out]
                        elif op_out in remainder_name_remap:
                            op.outputs[out_idx] = remainder_name_remap[op_out]
                        else:
                            logger.error(f'{op_out} is not in any of the name remaps')

            logger.debug(f'inner_layer_idx={inner_layer_idx}')
            if inner_layer_idx == 0:
                if quantized_model_info['tail'] is None:
                    main_subgraph_inputs = subgraph.inputs[:]
                    prev_subgraph_output = subgraph.outputs[0]

                    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
                    input_tensor_shape = tensor_utils.get_shape(input_tensor)
                    if quantized_model_info['model_type'] == 'whisper_decoder':
                        cache_tensor = subgraph.tensor_map[subgraph.inputs[3]]
                    else:
                        cache_tensor = subgraph.tensor_map[subgraph.inputs[num_non_cache_inputs]]
                    cache_tensor_shape = tensor_utils.get_shape(cache_tensor)
                    num_key_heads.append(int(cache_tensor_shape[1]))
                    num_value_heads.append(int(cache_tensor_shape[1]))
                elif quantized_model_info['tail'] == 'tail':
                    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
                    input_tensor_shape = tensor_utils.get_shape(input_tensor)
                elif quantized_model_info['tail'] == 'eagle':
                    input_tensor = subgraph.tensor_map[subgraph.inputs[0]]
                    input_tensor_shape = tensor_utils.get_shape(input_tensor)
                    cache_tensor = subgraph.tensor_map[subgraph.inputs[num_non_cache_inputs]]
                    cache_tensor_shape = tensor_utils.get_shape(cache_tensor)
                if quantized_model_info['tail'] != 'tail':
                    builder = ModelBuilder()
                    builder.import_subgraph(model)
            else:
                name_mapping_dict = {}
                if quantized_model_info['tail'] is None:
                    if quantized_model_info['model_type'] == 'whisper_decoder':
                        cache_tensor = subgraph.tensor_map[subgraph.inputs[1]]
                        cache_tensor_shape = tensor_utils.get_shape(cache_tensor)
                        num_key_heads.append(int(cache_tensor_shape[1]))
                        num_value_heads.append(int(cache_tensor_shape[1]))
                        name_mapping_dict[subgraph.inputs[1]] = main_subgraph_inputs[1]
                    else:
                        cache_tensor = subgraph.tensor_map[subgraph.inputs[num_non_cache_inputs]]
                        cache_tensor_shape = tensor_utils.get_shape(cache_tensor)
                        num_key_heads.append(int(cache_tensor_shape[1]))
                        num_value_heads.append(int(cache_tensor_shape[1]))

                        if quantized_model_info['model_type'] == 'gecko2':
                            if not set(quantized_model_info['layer_ids']).isdisjoint(
                                set(quantized_model_info['model_config']['fire_pe_index'])
                            ):
                                # Note: only works when number of chunks == 1
                                if quantized_model_info['layer_ids'][0] != 3:
                                    name_mapping_dict[subgraph.inputs[1]] = 'global_mask'
                                    name_mapping_dict['fire_pe_relative_pos'] = 'fire_pe_relative_pos'
                                    name_mapping_dict['fire_pe_query_pos'] = 'fire_pe_query_pos'
                            else:
                                name_mapping_dict[subgraph.inputs[1]] = 'swa_mask'
                        else:
                            name_mapping_dict[subgraph.inputs[1]] = main_subgraph_inputs[1]

                        if quantized_model_info['model_type'] == 'gecko2':
                            # handle per layer embed
                            if 'per_layer_emb' in subgraph.inputs:
                                name_mapping_dict['per_layer_emb'] = 'per_layer_emb'
                            if 'cos' in subgraph.inputs:
                                name_mapping_dict['cos'] = 'cos'
                            if 'sin' in subgraph.inputs:
                                name_mapping_dict['sin'] = 'sin'
                        else:
                            for j in range(2, num_non_cache_inputs):
                                # only need to account for pos_emb in gecko chunk 0
                                if (
                                    'gecko' in quantized_model_info['model_type']
                                    and 0 not in quantized_model_info['layer_ids']
                                    and 'pos_emb' in main_subgraph_inputs
                                ):
                                    name_mapping_dict[subgraph.inputs[j]] = main_subgraph_inputs[j + 1]
                                else:
                                    name_mapping_dict[subgraph.inputs[j]] = main_subgraph_inputs[j]

                        if quantized_model_info['model_config'].get('infini_attention', False):
                            # infini mask
                            name_mapping_dict[subgraph.inputs[num_non_lora_inputs - 1]] = 'infini_mask'
                        elif quantized_model_info['model_config'].get('use_split_mask', False):
                            # FIXME: support infini and split mask together
                            # split mask
                            # Patch F: split_mask sits before any extra inputs (e.g., ds_padded)
                            _sm_extra = quantized_model_info.get('num_extra_inputs', 0)
                            name_mapping_dict[subgraph.inputs[num_non_lora_inputs - 1 - _sm_extra]] = 'split_mask'

                name_mapping_dict[subgraph.inputs[0]] = prev_subgraph_output
                logger.debug(f'name_mapping_dict={name_mapping_dict}')

                builder.import_subgraph(model, input_name_mappings=name_mapping_dict)
                prev_subgraph_output = subgraph.outputs[0]

            layer_lora_inp_shapes = []
            for lora_inp in subgraph.inputs[num_non_lora_inputs:]:
                lora_input_tensor = subgraph.tensor_map[lora_inp]
                input_tensor_shape = tensor_utils.get_shape(lora_input_tensor).as_list()
                layer_lora_inp_shapes.append(input_tensor_shape)
                lora_inp_names.append(lora_inp)
            logger.debug(f'layer_lora_inp_shapes={layer_lora_inp_shapes}')
            if len(layer_lora_inp_shapes) > 0:
                lora_inp_shapes.append(layer_lora_inp_shapes)

            pbar.update(1)
            inner_layer_idx += 1
            if (
                inner_layer_idx == curr_chunk_num_layer
                and curr_chunk_idx == len(num_layers_per_chunk) - 1
                and not args.separate_tail
            ):
                # Include tail
                continue

            if inner_layer_idx >= curr_chunk_num_layer:
                if (curr_chunk_num_layer == 0 and quantized_model_info['tail'] == 'tail') or (
                    curr_chunk_num_layer == 1 and quantized_model_info['tail'] == 'eagle'
                ):
                    # Export tail only + fix shape
                    pbar.close()

                    output_dest_path = os.path.join(intermediate_folder, f'chunk_{curr_chunk_idx}.{file_format}')
                    quantized_model_utils.export_quantized_model(model, output_dest_path, np_compat=args.np_compat)

                    fix_tail_shape(
                        output_dest_path,
                        exp_name,
                        partial_output_folder_without_shape,
                        args.shapes,
                        quantized_model_info,
                        curr_chunk_idx,
                        ptq_description=quantized_model_description,
                        file_format=file_format,
                        np_compat=args.np_compat,
                        per_op=args.per_op,
                    )
                    inner_layer_idx = 0
                    curr_chunk_idx += 1
                    gc.collect()
                    if curr_chunk_idx < len(num_layers_per_chunk):
                        logger.info(f'\nConstructing chunk {curr_chunk_idx}:')
                        pbar = tqdm(total=max(1, num_layers_per_chunk[curr_chunk_idx]))
                else:
                    pbar.close()

                    output_dest_path = os.path.join(intermediate_folder, f'chunk_{curr_chunk_idx}.{file_format}')
                    if sink_rope:
                        input_names = ['input_embeds', 'mask', 'q_cos', 'q_sin', 'k_cos', 'k_sin']
                    else:
                        input_names = ['input_embeds', 'mask', 'pos_emb', 'cos', 'sin']
                    if whisper:
                        input_names.remove('cos')
                        input_names.remove('sin')
                    logger.debug(f'Current Chunk: {curr_chunk_idx}')
                    if not ((gecko or whisper) and curr_chunk_idx == 0) and not (sink_rope):
                        input_names.remove('pos_emb')

                    # add per layer embed and fire pe if current model is gecko2
                    if quantized_model_infos['llm'][0]['model_type'] == 'gecko2':
                        cos_sin = input_names[-2:]
                        input_names = input_names[:-2]
                        input_names.append('per_layer_emb')

                        curr_chunk_layer_ids = [
                            sum(num_layers_per_chunk[:curr_chunk_idx]) + idx for idx in range(curr_chunk_num_layer)
                        ]

                        if set(curr_chunk_layer_ids).isdisjoint(
                            set(quantized_model_infos['llm'][0]['model_config']['fire_pe_index'])
                        ):
                            input_names = input_names + cos_sin
                            input_names[1] = 'swa_mask'
                        elif set(curr_chunk_layer_ids).issubset(
                            set(quantized_model_infos['llm'][0]['model_config']['fire_pe_index'])
                        ):
                            input_names.append('fire_pe_relative_pos')
                            input_names.append('fire_pe_query_pos')
                            input_names[1] = 'global_mask'
                        else:
                            input_names.append('fire_pe_relative_pos')
                            input_names.append('fire_pe_query_pos')
                            input_names = input_names + cos_sin
                            input_names[1] = 'swa_mask'
                            input_names.insert(2, 'global_mask')

                    cache_inputs = []
                    for idx in range(curr_chunk_num_layer):
                        cache_inputs.extend(
                            [
                                f'past_keys_{idx}',
                                f'past_values_{idx}',
                            ]
                        )
                        if quantized_model_info['model_config'].get('infini_attention', False):
                            cache_inputs.extend(
                                [
                                    f'mem_{idx}',
                                    f'z_{idx}',
                                ]
                            )
                        if whisper:
                            cache_inputs.extend(
                                [
                                    f'cross_keys_{idx}',
                                    f'cross_values_{idx}',
                                ]
                            )
                    other_mask_input = []
                    if quantized_model_info['model_config'].get('infini_attention', False):
                        other_mask_input.append('infini_mask')
                    if quantized_model_info['model_config'].get('use_split_mask', False):
                        # FIXME: Fix the hardoded position in shape fixer rehaul
                        # find mask
                        try:
                            mask_index = input_names.index('mask')
                            input_names.insert(mask_index + 1, 'split_mask')
                        except ValueError:
                            logger.error('Split mask is currently unsupported for this model.')
                    # Patch F7: deepstack ds_padded_<inner_idx> 在合并 LLM 模型里也是 model-level inputs，
                    # 必须出现在 input_names 才能让 builder.export 不报 "Missing some of the subgraph inputs"。
                    ds_padded_inputs = []
                    for _inner_idx in range(curr_chunk_num_layer):
                        # 仅当该层在 PTQ 时被标了 num_extra_inputs（即 chunks 0..num_deepstack_inject-1）才有 ds_padded
                        try:
                            _layer_info = curr_chunk_quantized_model_infos[_inner_idx]
                        except (NameError, IndexError):
                            _layer_info = None
                        _n_extra = (_layer_info.get('num_extra_inputs', 0) if _layer_info else 0)
                        for _e in range(_n_extra):
                            ds_padded_inputs.append(f'ds_padded_{_inner_idx}' if _e == 0 else f'ds_padded_{_inner_idx}_{_e}')

                    input_names = input_names + cache_inputs + other_mask_input + ds_padded_inputs + lora_inp_names
                    cache_outputs = []
                    for idx in range(curr_chunk_num_layer):
                        cache_outputs.extend(
                            [
                                f'curr_keys_{idx}',
                                f'curr_values_{idx}',
                            ]
                        )
                    output_names = [
                        f'hidden_states_{curr_chunk_num_layer - int(quantized_model_info["tail"] is None)}',
                        *cache_outputs,
                    ]

                    if get_attn_logits:
                        output_names += [f'attn_logits_{x}' for x in range(curr_chunk_num_layer)]
                    if get_attn_weights:
                        output_names += [f'attn_weights_{x}' for x in range(curr_chunk_num_layer)]

                    logger.debug(f'Finalized Input Names:\n{input_names}')
                    logger.debug(f'Finalized Output Names:\n{output_names}\n')

                    out_nrpmodel = builder.export(input_names, output_names)  # sets IO names for model
                    subgraph = out_nrpmodel.subgraphs[0]
                    quantized_model_utils.export_quantized_model(
                        out_nrpmodel, output_dest_path, np_compat=args.np_compat
                    )

                    chunked_model_info = quantized_model_utils.merge_quantized_model_infos(
                        curr_chunk_quantized_model_infos
                    )

                    fix_llm_shape(
                        output_dest_path,
                        exp_name,
                        partial_output_folder_without_shape,
                        os.path.join(args.quantized_model_folder, 'llm'),
                        args.shapes,
                        chunked_model_info,
                        curr_chunk_idx,
                        lora_inp_shapes,
                        ptq_description=quantized_model_description,
                        file_format=file_format,
                        np_compat=args.np_compat,
                        use_single_bmm_attention=args.use_single_bmm_attention,
                        per_op=args.per_op,
                        sink_rope=sink_rope,
                        cache_evict_attributes=cache_evict_attributes,
                        output_names=output_names
                        if (get_attn_logits or get_attn_weights or args.clear_attn_outputs)
                        else None,
                        clear_attn_outputs=args.clear_attn_outputs,
                        bita_decode_token=args.bita_decode_token,
                    )

                    inner_layer_idx = 0
                    curr_chunk_idx += 1
                    lora_inp_shapes = []
                    lora_inp_names = []
                    num_key_heads = []
                    num_value_heads = []
                    curr_chunk_quantized_model_infos = []
                    gc.collect()

                    if curr_chunk_idx < len(num_layers_per_chunk):
                        logger.info(f'\nConstructing chunk {curr_chunk_idx}:')
                        if curr_chunk_idx == len(num_layers_per_chunk) - 1:
                            pbar = tqdm(total=max(1, num_layers_per_chunk[curr_chunk_idx] + 1))
                        else:
                            pbar = tqdm(total=max(1, num_layers_per_chunk[curr_chunk_idx]))

    standalone_embedder_path = os.path.join(args.quantized_model_folder, 'embedder')
    llm_folder = os.path.join(args.quantized_model_folder, 'llm')
    if not os.path.exists(standalone_embedder_path):
        embedding_paths = [
            os.path.join(llm_folder, f)
            for f in os.listdir(llm_folder)
            if (f.startswith('embedding_') and f.endswith('.bin'))
        ]
        if len(embedding_paths) != 1 and not whisper:
            logger.error(f'Expect exactly one Embedding bin in `{llm_folder}`.', err=FileNotFoundError)
    else:
        embedding_paths = [
            os.path.join(standalone_embedder_path, f)
            for f in os.listdir(standalone_embedder_path)
            if (f.startswith('embedding_') and f.endswith('.tflite'))
        ]
        if len(embedding_paths) != 1 and not whisper:
            logger.error(f'Expect exactly one Embedding tflite in `{standalone_embedder_path}`.', err=FileNotFoundError)

    embedding_path = embedding_paths[0]
    if whisper:
        embedding_path2 = embedding_paths[1]
    for shape in args.shapes:
        if 't' in shape and 'c' in shape:
            output_folder = os.path.join(partial_output_folder_without_shape, 'llm', f'llm_{shape}')
            output_embedding_folder = output_folder

            # if there is standalone embedding (tflite), create another folder to store
            if os.path.exists(standalone_embedder_path):
                output_embedding_folder = os.path.join(output_embedding_folder, 'original_embedder')
                os.makedirs(output_embedding_folder)
                # if there is standalone embedding (tflite), copy the info json
                shutil.copy2(embedding_path.replace('.tflite', '_info.json'), output_embedding_folder)

            output_embedding_path = os.path.join(output_embedding_folder, os.path.basename(embedding_path))
            # Copy embedding bin to each fixed-shaped quantized model folder
            shutil.copy2(embedding_path, output_embedding_path)
            overture_utils.dump_overture(llm_folder, output_folder, num_layers_per_chunk, args.pad_cache)
            if whisper:
                output_embedding_path2 = os.path.join(output_folder, os.path.basename(embedding_path2))
                # Copy embedding bin to each fixed-shaped quantized model folder
                shutil.copy2(embedding_path2, output_embedding_path2)
        elif 'l' in shape:
            continue
        else:
            output_folder = os.path.join(partial_output_folder_without_shape, 'encoder', f'encoder_{shape}')
        logger.info(f'{shape} -> {output_folder}')

    # Copy config.json to fixed-shaped quantized model folder
    shutil.copyfile(
        os.path.join(args.quantized_model_folder, 'config.json'),
        os.path.join(partial_output_folder_without_shape, 'config.json'),
    )

    with contextlib.suppress(OSError):
        os.rmdir('temp')


if __name__ == '__main__':
    main()
