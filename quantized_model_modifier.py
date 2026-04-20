#
# Copyright (C) 2024 MediaTek Inc. All rights reserved.
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
"""Script to perform some modifications of quantized models."""

import argparse
import os
import shutil
import sys

from mtk_converter.python.utils import subgraph_utils
from tqdm import tqdm

from . import __version__
from .utils import const, logger, quantized_model_utils, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_modify_quantized_llm'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert dynamic shape quantized models to one or more fixed shape quantized models.',
        allow_abbrev=False,
    )
    parser.add_argument(
        'quantized_model_folder',
        type=str,
        help='[Required] Input quantized model folder to be modified. Supports both dynamic shape and fixed '
        'shape quantized models.',
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='User-specified output folder to save modified quantized model to.',
    )
    parser.add_argument(
        '-c',
        '--cache_dtype',
        type=str,
        choices=['int8', 'int16', 'float'],
        default=None,
        help='New cache dtype. Cannot be the same as current quantized model. Original cache dtype cannot be float.',
    )
    parser.add_argument(
        '-l',
        '--lora_dtype',
        type=str,
        choices=['int8', 'int16', 'float'],
        default=None,
        help='New lora dtype. Cannot be the same as current quantized model. Original lora dtype cannot be float.',
    )
    parser.add_argument(
        '--np_compat',
        type=int,
        default=None,
        choices=const.NP_COMPAT_VERIONS,
        help='Export quantized model using specific neuropilot version OP export spec. '
        'Defaults to using the same version as installed mtk_converter '
        '(NP7 converter will default to NP7 export spec). '
        'Cannot specify a neuropilot version that is newer than installed mtk_converter version.',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    return parser


def args_sanity_checks(args):
    """Prints the arguments for verification.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    sc.check_exist(args.quantized_model_folder, 'quantized model directory')
    sc.check_isdir(args.quantized_model_folder, 'quantized model directory')

    if args.cache_dtype is None and args.lora_dtype is None:
        logger.error('Nothing to modify. Please specify either `cache_dtype` or `lora_dtype`.')

    if args.np_compat is not None:
        cvt_version = utils.get_converter_version()
        if args.np_compat > cvt_version:
            logger.error(
                f'`np_compat` ({args.np_compat}) cannot be greater than current mtk_converter version ({cvt_version})'
            )

    if os.path.exists(args.output_folder):
        logger.error(
            f'Output folder {args.output_folder} already exists. Please manually delete it first if you want to '
            'overwrite it.',
            err=FileExistsError,
        )


def print_args(args):
    """Prints the arguments for verification.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Quantized model folder:                {args.quantized_model_folder}')
    if args.cache_dtype is not None:
        logger.info(f'Change cache inputs/outputs dtype to:  {args.cache_dtype}')
    if args.lora_dtype is not None:
        logger.info(f'Change LoRA inputs dtype to:           {args.lora_dtype}')
    logger.info(f'mtk_llm_sdk version:                  {__version__}')


def change_cache_dtype(subgraph, target_dtype, quantized_model_info):
    """Changes the cache dtype of the subgraph.

    Args:
        subgraph (object): The subgraph object.
        target_dtype (str): The target dtype ('int8', 'int16', 'float').
        quantized_model_info (dict): The quantized model information dict.
    """
    if target_dtype is None:
        return quantized_model_info
    return quantized_model_utils.change_cache_dtype(subgraph, target_dtype, quantized_model_info)


def change_lora_dtype(subgraph, target_dtype, quantized_model_info):
    """Changes the lora dtype of the subgraph.

    Args:
        subgraph (object): The subgraph object.
        target_dtype (str): The target dtype ('int8', 'int16', 'float').
        quantized_model_info (dict): The quantized model information dict.
    """
    if target_dtype is None:
        return quantized_model_info
    return quantized_model_utils.change_lora_dtype(subgraph, target_dtype, quantized_model_info)


@memory_peak_profile
def main(args=None):
    """Main function to modify and export quantized model models.

    This function parses the arguments, performs sanity checks, modifies the cache dtype of the models if specified,
    and exports the modified models to the specified directory.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    args_sanity_checks(args)
    print_args(args)

    llm_quantized_model_list = utils.get_sorted_path_list(args.quantized_model_folder, ['.tflite', '.mlir'])
    if len(llm_quantized_model_list) == 0:
        logger.error(
            f'Did not find any files ending with .tflite or .mlir in {args.quantized_model_folder}.',
            err=FileNotFoundError,
        )

    nrpmodels = []
    subgraphs = []
    quantized_model_infos = []

    logger.info('Loading quantized models...')
    for f in llm_quantized_model_list:
        sc.check_ext(f, ['.tflite', '.mlir'])
        quantized_model_info = quantized_model_utils.extract_llm_quantized_model_info(f)
        if args.cache_dtype is not None:
            cache_dtype = quantized_model_info['output_dtypes'][-1]
            if cache_dtype == quantized_model_utils._make_dtype_numpy(args.cache_dtype):  # noqa: SLF001
                logger.error(f'Cache inputs are already {args.cache_dtype} dtype. Nothing to modify.', err=ValueError)
        if args.lora_dtype is not None:
            if quantized_model_info['num_lora_inputs'] == 0:
                logger.error('Quantized model provided does not have LoRA inputs.')
            start_idx = quantized_model_info['num_non_lora_inputs']
            lora_dtype = quantized_model_info['input_dtypes'][start_idx]
            if lora_dtype == quantized_model_utils._make_dtype_numpy(args.lora_dtype):  # noqa: SLF001
                logger.error(f'Lora inputs are already {args.lora_dtype} dtype. Nothing to modify.', err=ValueError)

        quantized_model_infos.append(quantized_model_info)
        subgraph, nrpmodel = quantized_model_utils.get_subgraph_from_quantized_model(f, return_nrpmodel=True)
        nrpmodels.append(nrpmodel)
        subgraphs.append(subgraph)

    ext = llm_quantized_model_list[0].rsplit('.', 1)[-1]

    logger.info('Modifying quantized models...')
    assert len(subgraphs) == len(quantized_model_infos)
    if args.cache_dtype is not None:
        logger.info(f'Changing cache inputs/outputs dtype to {args.cache_dtype}...')
        for subgraph, quantized_model_info in tqdm(zip(subgraphs, quantized_model_infos), total=len(subgraphs)):
            quantized_model_info = change_cache_dtype(subgraph, args.cache_dtype, quantized_model_info)

    if args.lora_dtype is not None:
        logger.info(f'Changing LoRA inputs dtype to {args.lora_dtype}...')
        for subgraph, quantized_model_info in tqdm(zip(subgraphs, quantized_model_infos), total=len(subgraphs)):
            quantized_model_info = change_lora_dtype(subgraph, args.lora_dtype, quantized_model_info)

    logger.info('Exporting modified models...')
    os.makedirs(args.output_folder)
    for i, nrpmodel in tqdm(enumerate(nrpmodels), total=len(nrpmodels)):
        subgraph_utils.remove_unused_ops_and_tensors(subgraphs[i])
        subgraph_utils.ensure_topological_op_order(subgraphs[i])

        orig_filename = os.path.basename(llm_quantized_model_list[i])
        new_quantized_model_path = os.path.join(args.output_folder, orig_filename)

        quantized_model_utils.export_quantized_model(nrpmodel, new_quantized_model_path, np_compat=args.np_compat)
        quantized_model_utils.export_quantized_model_info(
            quantized_model_infos[i],
            quantized_model_infos[i]['model_config'],
            quantized_model_infos[i]['precision_config'],
            new_quantized_model_path.replace(f'.{ext}', '_info.json'),
        )

    logger.info(f'Modified quantized model exported to: {new_quantized_model_path}')

    embedding_paths = [
        os.path.join(args.quantized_model_folder, f)
        for f in os.listdir(args.quantized_model_folder)
        if (f.startswith('embedding_') and f.endswith('.bin'))
    ]
    if len(embedding_paths) != 1:
        logger.error(f'Expect exactly one embedding bin in `{llm_quantized_model_list}`.', err=FileNotFoundError)
    embedding_path = embedding_paths[0]
    # Copy embedding bin to new quantized model folder
    shutil.copy2(embedding_path, args.output_folder)


if __name__ == '__main__':
    main()
