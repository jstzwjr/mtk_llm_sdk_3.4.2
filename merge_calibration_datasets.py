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
"""Script to merge multiple calibration datasets for LLM PTQ into 1 dataset."""

import argparse
import os
import shutil
import sys

from tqdm import tqdm

from . import __version__
from .utils import logger, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_merge_calib_datasets'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Merge multiple calibration datasets into one.', allow_abbrev=False)
    parser.add_argument('name', type=str, help='[Required] Output dataset name')
    parser.add_argument(
        'dataset1', nargs=1, metavar='datasets', help='[Required] path(s) to calibration datasets to merge.'
    )
    parser.add_argument('dataset2', nargs='+', metavar='datasets', help=argparse.SUPPRESS)
    parser.add_argument(
        '-d',
        '--delete_original',
        action='store_true',
        help='Boolean flag to delete source calibration datasets after merging to save disk space. CANNOT BE UNDONE.',
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

    Raises:
        FileExistsError: If the output dataset path already exists.
        RuntimeError: If the datasets do not have the same number of layers.
    """
    output_path = os.path.join('calibration_datasets', args.name)
    if os.path.exists(output_path):
        logger.error(
            f'dataset name {args.name} already exists. Please manually delete the existing one '
            'if you wish to overwrite it.',
            err=FileExistsError,
        )

    num_layers = None
    for i, ds in enumerate([os.path.join(x, 'llm') for x in args.datasets]):
        sc.check_exist(ds, f'Dataset {i}')
        sc.check_isdir(ds, f'Dataset {i}')
        curr_num_layers = len(os.listdir(ds))
        if num_layers is not None and curr_num_layers != num_layers:
            logger.error(
                'All LLM datasets must have the same number of chunks, but found '
                f'{num_layers} and {curr_num_layers} chunks.'
            )
        num_layers = curr_num_layers


def print_args(args):
    """Prints the arguments for verification.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    output_path = os.path.join('calibration_datasets', args.name)
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Output dataset path:  {output_path}')
    for i, ds in enumerate(args.datasets):
        logger.info(f'Dataset {i + 1}:            {ds}')
    logger.info(f'mtk_llm_sdk version: {__version__}')


@memory_peak_profile
def main(args=None):
    """Main function to merge calibration datasets.

    This function parses the arguments, performs sanity checks, and merges the specified calibration datasets
    into a single dataset. It saves the merged dataset to the specified output directory.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))
    args.datasets = args.dataset1 + args.dataset2

    args_sanity_checks(args)
    print_args(args)

    num_layers = len(os.listdir(os.path.join(args.datasets[0], 'llm')))
    output_dir = os.path.join('calibration_datasets', args.name)
    has_encoder = False
    for i in range(len(args.datasets)):
        has_encoder = has_encoder or os.path.exists(os.path.join(args.datasets[i], 'encoder'))
    os.makedirs(output_dir)

    total_encoder_batches = 0
    total_llm_batches = 0
    for ds in args.datasets:
        if has_encoder:
            total_encoder_batches += len([x for x in os.listdir(os.path.join(ds, 'encoder')) if x.endswith('.npz')])
        total_llm_batches += len([x for x in os.listdir(os.path.join(ds, 'llm', 'chunk_0')) if x.endswith('.npz')])

    if has_encoder:
        num_layers_enc = len(os.listdir(os.path.join(args.datasets[0], 'encoder')))
        for i in range(num_layers_enc):
            os.makedirs(os.path.join(output_dir, 'encoder', f'chunk_{i}'))
    for i in range(num_layers):
        os.makedirs(os.path.join(output_dir, 'llm', f'chunk_{i}'))
    new_encoder_batch_idx = 0 if not has_encoder else [0] * num_layers_enc
    new_batch_idx = [0] * num_layers
    pbar = tqdm(total=total_encoder_batches + total_llm_batches * num_layers)
    for ds in args.datasets:
        if has_encoder:
            logger.info(f'Merging {ds} (Encoder)')
            for layer_idx in range(num_layers):
                curr_ds_chunk_batches = utils.get_sorted_path_list(
                    os.path.join(ds, 'encoder', f'chunk_{layer_idx}'), '.npz', sep='-'
                )
                if layer_idx != num_layers - 1 or num_layers == 1:
                    lora_mapping_file = os.path.join(ds, 'encoder', 'lora_mapper.txt')
                    if os.path.exists(lora_mapping_file):
                        with open(lora_mapping_file) as f:
                            lora_maps = f.readlines()
                        lora_maps = [x.rstrip('\n') for x in lora_maps if x != '']
                    else:
                        lora_maps = ['None' for _ in range(len(curr_ds_chunk_batches))]
                    for i, batch in enumerate(curr_ds_chunk_batches):
                        dst = os.path.join(
                            'calibration_datasets',
                            args.name,
                            'encoder',
                            f'chunk_{layer_idx}',
                            'batch-{:04d}.npz'.format(new_encoder_batch_idx[layer_idx]),
                        )
                        shutil.copyfile(batch, dst)
                        with open(
                            os.path.join(output_dir, 'encoder', f'chunk_{layer_idx}', 'lora_mapper.txt'), 'w'
                        ) as f:
                            f.write(lora_maps[i] + '\n')
                        new_encoder_batch_idx[layer_idx] += 1
                        pbar.update(1)

        logger.info(f'Merging {ds} (LLM)')
        for layer_idx in range(num_layers):
            curr_ds_chunk_batches = utils.get_sorted_path_list(
                os.path.join(ds, 'llm', f'chunk_{layer_idx}'), '.npz', sep='-'
            )
            if layer_idx != num_layers - 1 or num_layers == 1:
                lora_mapping_file = os.path.join(ds, 'llm', f'chunk_{layer_idx}', 'lora_mapper.txt')
                if os.path.exists(lora_mapping_file):
                    with open(lora_mapping_file) as f:
                        lora_maps = f.readlines()
                    lora_maps = [x.rstrip('\n') for x in lora_maps if x != '']
                else:
                    lora_maps = ['None' for _ in range(len(curr_ds_chunk_batches))]

                for i, batch in enumerate(curr_ds_chunk_batches):
                    dst = os.path.join(
                        'calibration_datasets',
                        args.name,
                        'llm',
                        f'chunk_{layer_idx}',
                        'batch-{:04d}.npz'.format(new_batch_idx[layer_idx]),
                    )
                    shutil.copyfile(batch, dst)
                    with open(os.path.join(output_dir, 'llm', f'chunk_{layer_idx}', 'lora_mapper.txt'), 'a') as f:
                        f.write(lora_maps[i] + '\n')
                    new_batch_idx[layer_idx] += 1
                    pbar.update(1)
            else:  # Tail don't need lora mapping
                for batch in curr_ds_chunk_batches:
                    dst = os.path.join(
                        'calibration_datasets',
                        args.name,
                        'llm',
                        f'chunk_{layer_idx}',
                        'batch-{:04d}.npz'.format(new_batch_idx[layer_idx]),
                    )
                    shutil.copyfile(batch, dst)
                    new_batch_idx[layer_idx] += 1
                    pbar.update(1)
    pbar.close()

    logger.info(f'Datasets merged to calibration_datasets/{args.name}')

    if args.delete_original:
        for ds in args.datasets:
            logger.info(f'Deleting {ds} ...')
            utils.recursive_remove_if_exist(ds)
        logger.info('done')


if __name__ == '__main__':
    main()
