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
"""Script to export float LoRA weights to cmdline-compatible chunked LoRA bins."""

import argparse
import json
import os
import sys

import torch

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.lora_handler import LoRAHandler
from .models.pipeline import FloatPipeline, QuantizedPipeline
from .utils import logger, quantized_model_utils, rotate, utils
from .utils import sanity_checks as sc

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_export_llm_lora_bins'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Export PyTorch LoRA weights to bin files with metadata inserted for cmdline inference.'
    )
    parser.add_argument(
        'quantized_model_folder',
        type=str,
        help='[Required] Fixed shape quantized model folder. Either prompt or generative model will do.',
    )
    parser.add_argument('lora_config', type=str, default=None, nargs='+', help='LoRA adapter config json files.')
    parser.add_argument(
        '--use_json',
        action='store_true',
        help='Whether to extract the scales using the quantized model info json instead of the quantized model itself. '
        'Defaults to False.',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        list: A list of quantized model information dictionaries.

    Raises:
        RuntimeError: If any of the argument checks fail.
    """
    config = PipelineConfig(
        os.path.join(args.quantized_model_folder, 'config.json'),
        verbose=False,
    )
    for lora_config in args.lora_config:
        sc.check_exist(lora_config, 'Lora config file')
        sc.check_ext(lora_config, '.json', 'Lora config file')
        sc.check_lora_config(lora_config, config)

    sc.check_exist(args.quantized_model_folder, 'quantized model directory')
    sc.check_isdir(args.quantized_model_folder, 'quantized model directory')
    llm_folder = os.path.join(args.quantized_model_folder, 'llm')
    fixed_shape_model_folders = [os.path.join(llm_folder, x) for x in os.listdir(llm_folder)]
    if len(fixed_shape_model_folders) == 0:
        logger.error(f'No quantized model folders detected in: {llm_folder}', err=FileNotFoundError)

    quantized_model_infos = []
    if args.use_json:
        for fixed_shape_model_folder in fixed_shape_model_folders:
            quantized_model_paths = utils.get_sorted_path_list(fixed_shape_model_folder, ext='.json')
            if quantized_model_paths == []:
                logger.error(f'No quantized model info json files found in {fixed_shape_model_folder}')
        for quantized_model_path in quantized_model_paths:
            with open(quantized_model_path) as f:
                quantized_model_infos.append(json.load(f))
    else:
        quantized_model_paths = utils.get_sorted_path_list(fixed_shape_model_folders[0], ext=['.tflite', '.mlir'])
        if len(quantized_model_paths) == 0:
            logger.error(f'No quantized models found in {fixed_shape_model_folders[0]}', err=FileNotFoundError)
        for quantized_model_path in quantized_model_paths:
            quantized_model_infos.append(quantized_model_utils.extract_llm_quantized_model_info(quantized_model_path))

    if sum(x['num_lora_inputs'] for x in quantized_model_infos) == 0:
        logger.error('Quantized model does not have lora inputs.')

    if quantized_model_infos[0]['t'] is None or quantized_model_infos[0]['c'] is None:
        logger.error('Dynamic shape quantized model not supported. Please run mtk_fix_llm_shape command first.')

    for lora_conf in args.lora_config:
        with open(lora_conf) as f:
            lora_config = json.load(f)
        if not config.rotate and lora_config.get('rotate', False):
            logger.error(f'{lora_conf} Lora is rotated, expect TFLite/MLIR model to be rotated as well.')


def main(args=None):
    """Main function to process quantized models with LoRA inputs.

    This function parses the arguments, performs sanity checks, updates the configuration with LoRA inputs,
    and creates LoRA binary files for the specified quantized models.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))
    logger.debug(f'mtk_llm_sdk version: {__version__}')

    args_sanity_checks(args)

    config = PipelineConfig(
        os.path.join(args.quantized_model_folder, 'config.json'),
        verbose=False,
    )
    lora_handler = LoRAHandler(args.lora_config, config)

    if config.rotate and not lora_handler.rotated:
        # When quantized model has been rotated in base-only PTQ, need to rotate LoRA as well if not come from QALFT
        logger.info('Detected a rotated quantized model, but unrotated LoRA.')
        logger.info(f'Rotating given LoRA: {args.lora_config}')
        pipeline = FloatPipeline(
            config,
            args.lora_config,
            task='export_lora',
            dtype=torch.float32,
            debug=args.debug,
            quiet=True,
        )
        lora_config_path = rotate.rotate_and_save_lora(pipeline)
        del pipeline
    else:
        lora_config_path = args.lora_config

    llm_folder = os.path.join(args.quantized_model_folder, 'llm')
    fixed_shape_model_folders = [os.path.join(llm_folder, x) for x in os.listdir(llm_folder)]

    logger.info(f'Detected {len(fixed_shape_model_folders)} fixed shape quantized model folders to export to:')
    logger.info(f'{os.listdir(llm_folder)}')

    for fixed_shape_model_folder in fixed_shape_model_folders:
        logger.info(f'Current fixed shape model: {fixed_shape_model_folder}')
        gen_models = [x for x in os.listdir(fixed_shape_model_folder) if x.startswith('llm_1t')]
        if len(gen_models) > 0:
            quantized_model_folders = {
                'encoder': None,
                'prompt': None,
                'generative': fixed_shape_model_folder,
            }
        else:
            quantized_model_folders = {
                'encoder': None,
                'prompt': fixed_shape_model_folder,
                'generative': None,
            }
        pipeline = QuantizedPipeline(
            config,
            lora_config_path,
            task='export_lora',
            quantized_model_folders=quantized_model_folders,
            debug=args.debug,
            quiet=True,
            use_json=args.use_json,
        )

        for chunk_idx, quantized_model_path in enumerate(
            utils.get_sorted_path_list(fixed_shape_model_folder, ext='.json' if args.use_json else ['.tflite', '.mlir'])
        ):
            if len(gen_models) > 0 and pipeline.llm_gen_quantized_model_infos[chunk_idx]['num_lora_inputs'] == 0:
                continue
            if len(gen_models) == 0 and pipeline.llm_prompt_quantized_model_infos[chunk_idx]['num_lora_inputs'] == 0:
                continue
            lora_precision = pipeline.precision_config.get_precision_name(pipeline.precision_config.lora_precision)[1]
            logger.info(
                f'Creating {lora_precision} lora bins for Chunk {chunk_idx}, target quantized model path: '
                f'{quantized_model_path}'
            )
            utils.create_lora_bin_for_cmdline(
                quantized_model_path,
                chunk_idx,
                pipeline,
                lora_precision,
                subgraph=quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
                if not args.use_json
                else None,
            )


if __name__ == '__main__':
    main()
