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
"""Script to perform Quantization-Aware LoRA FineTuning."""

import argparse
import collections
import copy
import json
import os
import pathlib
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mtk_quantization
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import __version__

try:
    import deepspeed
    from deepspeed import comm as dist

    deepspeed_installed = True
except ModuleNotFoundError:
    deepspeed_installed = False
    deepspeed_launched = False

if deepspeed_installed:
    ds_env_vars = [
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'CROSS_RANK',
        'CROSS_SIZE',
        'LOCAL_SIZE',
        'RANK',
        'LOCAL_RANK',
    ]
    deepspeed_launched = True
    for env_var in ds_env_vars:
        if env_var not in os.environ:
            deepspeed_launched = False
            break

from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import FloatPipeline
from .utils import cache_utils, const, generate_utils, logger, qat_utils, quantized_model_utils, rotate, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.preformatter import Preformatter

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_qalft_llm'

LABEL_PAD_TOKEN_ID = -100
PADDING_SIDE = 'right'
PRECISION = {'float32': torch.float32, 'float16': torch.float16}


@dataclass
class DataCollator:
    """Data collator that dynamically pads the inputs and prepares the necessary tensors for the model.

    Attributes:
        pipeline (FloatPipeline): The Pipeline object.
        padding (Optional[str]): The padding strategy to use. Default is 'longest'.
        max_length (Optional[int]): The maximum length of the sequences.
        pad_to_multiple_of (Optional[int]): If set, will pad the sequence to a multiple of the provided value.
        label_pad_token_id (int): The token ID to use for padding the labels. Default is -100.
        return_tensors (str): The type of tensors to return. Default is 'pt'.

    Methods:
        __call__(self, features, return_tensors=None):
            Pads the input features and prepares the necessary tensors for the model.
    """

    pipeline: FloatPipeline
    padding: Optional[str] = 'longest'
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'
    dtype: torch.dtype = torch.float32

    def __call__(self, features, return_tensors=None):
        """Pads the input features and prepares the necessary tensors for the model.

        Args:
            features (list): A list of input features.
            return_tensors (Optional[str]): The type of tensors to return. Default is None.

        Returns:
            dict: A dictionary containing the padded input features and additional tensors required by the model.
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature['labels'] for feature in features] if 'labels' in features[0] else None
        mm_inputs = [feature['mm_inputs'] for feature in features] if 'mm_inputs' in features[0] else None
        kwargs = [feature['kwargs'] for feature in features] if 'kwargs' in features[0] else None
        # Keep original features for potential multimodal scenario

        features = [{k: v for k, v in feature.items() if k == 'input_ids'} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        if not hasattr(self.pipeline.tokenizer, 'deprecation_warnings'):
            batch = self.pipeline.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            # Save the state of the warning, then disable it
            warning_state = self.pipeline.tokenizer.deprecation_warnings.get('Asking-to-pad-a-fast-tokenizer', False)
            self.pipeline.tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True

            try:
                batch = self.pipeline.tokenizer.pad(
                    features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )
            finally:
                # Restore the state of the warning.
                self.pipeline.tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = warning_state

        if labels is not None:
            max_label_length = max(len(label) for label in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.pipeline.tokenizer.padding_side
            batch['labels'] = torch.tensor(
                [
                    label + [LABEL_PAD_TOKEN_ID] * (max_label_length - len(label))
                    if padding_side == 'right'
                    else [LABEL_PAD_TOKEN_ID] * (max_label_length - len(label)) + label
                    for label in labels
                ],
                dtype=torch.int64,
            )

        master_rot_emb = generate_utils.get_master_rot_emb(self.pipeline.config, self.pipeline.dtype)
        padding = batch['attention_mask']
        max_length = padding.shape[-1]
        mask = []
        split_mask = []
        cos = []
        sin = []
        cache = [[] for _ in range(2 * self.pipeline.config.l.num_hidden_layers)]

        # for deepspeed, put in cpu first, deepspeed will help to send to cuda
        target_device = self.pipeline.main_device if not deepspeed_launched else 'cpu'

        for b in padding:
            num_valid = sum(b).item()

            one_batch_cache = cache_utils.Cache(
                self.pipeline.config.l,
                1,
                self.pipeline.dtype,
                mode='dynamic',
                device=target_device,
                overture=self.pipeline.overture,
            )
            overture_size = one_batch_cache.overture_size
            for i in range(self.pipeline.config.l.num_hidden_layers):
                cache[2 * i].append(one_batch_cache.get(ret='k', layer=i)[0])
                cache[2 * i + 1].append(one_batch_cache.get(ret='v', layer=i)[0])

            num_invalid = max_length - num_valid
            one_batch_mask = generate_utils.generate_mask(
                one_batch_cache.cache_size,
                overture_size,
                num_valid + num_invalid,
                num_valid,
                mask_value=self.pipeline.config.l.mask_value,
                dtype=self.pipeline.dtype,
            )
            one_batch_rot_emb = master_rot_emb[:, :, overture_size : overture_size + num_valid, :]
            if num_invalid > 0:
                one_batch_rot_emb = torch.cat(
                    [
                        one_batch_rot_emb,
                        torch.zeros(1, 2, num_invalid, one_batch_rot_emb.shape[-1], dtype=one_batch_rot_emb.dtype),
                    ],
                    dim=2,
                )

            if self.pipeline.config.l.use_split_mask:
                mask.append(one_batch_mask[:, :, :, -max_length:])
                split_mask.append(one_batch_mask[:, :, :1, :-max_length])
            else:
                mask.append(one_batch_mask)
            cos.append(one_batch_rot_emb[:, :1, :, :])
            sin.append(one_batch_rot_emb[:, 1:, :, :])

        batch['mask'] = torch.cat(mask, dim=0).to(self.dtype)
        if self.pipeline.config.l.use_split_mask:
            batch['split_mask'] = torch.cat(split_mask, dim=0).to(self.dtype)
        batch['cos'] = torch.cat(cos, dim=0).to(self.dtype)
        batch['sin'] = torch.cat(sin, dim=0).to(self.dtype)
        for i in range(2 * self.pipeline.config.l.num_hidden_layers):
            cache[i] = torch.cat(cache[i], dim=0).to(self.dtype)
        batch['cache'] = cache
        batch['mm_inputs'] = mm_inputs
        batch['mm_kwargs'] = kwargs

        # Add multimodal related inputs if any
        """ batch['img_path'] = [feature['images'] for feature in features_copy] if 'images' in features_copy[0] else None

        batch['mm_embeddings'] = (
            [
                torch.load(feature['mm_embeddings'], map_location=target_device).get('mm_embeddings', None)
                for feature in features_copy
            ]
            if 'mm_embeddings' in features_copy[0]
            else None
        )

        batch['mm_kwargs'] = (
            [
                torch.load(feature['mm_embeddings'], map_location=target_device).get('mm_kwargs', None)
                for feature in features_copy
            ]
            if 'mm_embeddings' in features_copy[0]
            else None
        )

        batch['num_image_tokens'] = (
            [feature['num_img_tokens'] for feature in features_copy] if 'num_img_tokens' in features_copy[0] else None
        ) """  # noqa: E501

        return batch


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    def _get_common_parser(parser):
        parser.add_argument(
            'config',
            type=str,
            help='[Required] Model config json file. Model config must be in same directory as all tokenizer files.',
        )
        parser.add_argument(
            'quantized_model_folder',
            type=str,
            help='[Required] Dynamic shape quantized model folder to extract quantization parameters from.',
        )
        parser.add_argument(
            'dataset',
            type=str,
            help='[Required] Path to jsonl file of prompts (text only) to train the LoRA adapters on.',
        )
        parser.add_argument(
            'lora_name',
            type=str,
            help='[Required] LoRA name. Will be used to name output lora weights folder. Will '
            'load existing lora weights instead if already exists.',
        )
        parser.add_argument(
            '-r',
            '--lora_rank',
            type=int,
            default=None,
            help='lora rank. This argument is ignored if existing lora is found.',
        )
        parser.add_argument(
            '-m',
            '--lora_target_modules',
            type=str,
            default=[],
            nargs='*',
            help='lora target modules. This argument is ignored if existing lora is found.',
        )
        parser.add_argument(
            '-a',
            '--lora_alpha',
            type=int,
            default=None,
            help='lora alpha, defaults to lora_rank. This argument is ignored if existing lora is found.',
        )
        parser.add_argument('-d', '--lora_dropout', type=float, default=0.0, help='lora dropout, defaults to 0.')
        parser.add_argument(
            '--lora_start_layer_idx',
            type=int,
            default=0,
            help='Layer index of first layer to attach LoRA adapters to. Defaults to 0. This argument is ignored if '
            'existing lora is found.',
        )
        parser.add_argument(
            '--lora_end_layer_idx',
            type=int,
            default=-1,
            help='Layer index of last layer to attach LoRA adapters to. Defaults to -1 (last layer). This argument is '
            'ignored if existing lora is found.',
        )
        parser.add_argument(
            '-lr', '--learning_rate', type=float, default=3e-4, help='lora finetune learning rate, defaults to 0.0003.'
        )
        parser.add_argument(
            '-e', '--num_epoch', type=int, default=10, help='Number of epochs to finetune for. Defaults to 10.'
        )
        parser.add_argument('-b', '--batch_size', type=int, default=1, help='Training batch size. Defaults to 1.')
        parser.add_argument(
            '-c',
            '--epochs_per_checkpoint',
            type=int,
            default=0,
            help='Number of epochs before saving a checkpoint. 0 means only save final (best loss) lora weights. '
            'Defaults to 0.',
        )
        parser.add_argument(
            '-p',
            '--preformatter',
            type=str,
            default=None,
            help='Preformatter json file path to wrap instructions with for instruction-tuned models. '
            'Defaults to None.',
        )
        parser.add_argument(
            '--response_only_loss',
            action='store_true',
            help='Only calculate loss on response instead of prompt+response.',
        )
        parser.add_argument(
            '--buffer_scale',
            type=float,
            default=1.0,
            help='Scaling factor for activation minmax as a buffer after LoRA Add OP. Defaults to 1.0 (no buffer).',
        )
        parser.add_argument(
            '--use_main_gpu',
            action='store_true',
            help='Force all decoder layers onto the first GPU instead of evenly distributing the model across all '
            'available GPUs.',
        )
        parser.add_argument(
            '-st',
            '--safetensors',
            action='store_true',
            help='Save lora weights as .safetensors instead of PyTorch .bin',
        )
        parser.add_argument(
            '--force_overwrite',
            action='store_true',
            help='Force overwrite of existing quantized_model_folder, lora will always be trained from scratch and '
            'overwrite any existing lora of the same name. Use with caution.',
        )
        parser.add_argument(
            '--training_precision',
            type=str,
            choices=['float16', 'float32'],
            default='float32',
            help='Precision to use for training LoRA. Defaults to float32.',
        )
        parser.add_argument(
            '--bos_mode',
            type=str,
            default='see',
            choices=['see', 'skip'],
            help='How BOS token should be handled.Defaults to `see`.',
        )
        parser.add_argument(
            '--use_single_bmm_attention',
            action='store_true',
            help='Whether to use single bmm attention graph. ',
        )
        parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
        parser.add_argument(
            '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
        )
        if deepspeed_launched:
            # Deepspeed Arguments
            parser.add_argument(
                '--train_micro_batch_size_per_gpu',
                type=int,
                default=1,
                help='[DeepSpeed] Training micro batch size for each GPU. Defaults to 1.',
            )
            parser = deepspeed.add_config_arguments(parser)
        return parser

    def _get_converter_parser(subparser):
        parser = subparser.add_parser('converter', help='Run QALFT to train LoRA weights using mtk_converter backend.')
        return _get_common_parser(parser)

    def _get_mlkits_parser(subparser):
        parser = subparser.add_parser('mlkits', help='Run QALFT to train LoRA weights using mlkits backend.')
        parser.add_argument(
            '--workspace', type=pathlib.Path, help='Specifies the workspace directory for the output files.'
        )
        parser.add_argument(
            '--reduce-lora',
            action='store_true',
            help='This method merges the LoRA weights into the base model after fine-tuning MLKits optimized graph. '
            'The weights of the base model will be modified, so please use this option carefully when working with '
            'multiple LoRAs.',
        )
        parser.add_argument(
            '--opt-config',
            type=pathlib.Path,
            default=None,
            help='Specifies the path to a YAML file that contains optimization configurations. Note that the '
            'configurations should align with the graph existing in the workspace.',
        )
        parser.add_argument(
            '-i8c',
            '--int8_cache',
            action='store_true',
            help='Flag to force cache inputs/outputs to be int8 data type. Note that this setting should align with '
            'the graph existing in the workspace.',
        )

        # MLIR/TFLite Conversion.
        parser.add_argument(
            '-f',
            '--format',
            type=str,
            default='tflite',
            choices=['tflite', 'mlir'],
            help='File format to save output model as. Defaults to `tflite`. `mlir` is only supported on NP9 versions '
            'of mtk_converter.',
        )
        parser.add_argument(
            '--prompt-token-size',
            type=int,
            help='Specifies the token size for the prompt mode in the MLIR/TFLite model.',
        )
        parser.add_argument(
            '--gen-token-size',
            type=int,
            default=None,
            help='Specifies the token size for the generation mode in the TFLite model.',
        )
        parser.add_argument(
            '--cache-size', type=int, help='Specifies the cache size for the generation mode in the MLIR/TFLite model.'
        )
        parser.add_argument(
            '--num-chunks', type=int, default=1, help='Specifies the number of chunks for the MLIR/TFLite model.'
        )
        return _get_common_parser(parser)

    parser = argparse.ArgumentParser(
        description='Run Quantization-Aware LoRA FineTuning (QALFT) to train LoRA weights using a pre-quantized base model with chosen backend.',  # noqa: E501
        allow_abbrev=False,
    )
    if deepspeed_launched:
        parser.add_argument('--local_rank', type=int, default=-1, help=argparse.SUPPRESS)
    subparser = parser.add_subparsers(dest='backend')
    _get_converter_parser(subparser)
    _get_mlkits_parser(subparser)

    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        tuple: A tuple containing a boolean indicating if existing LoRA weights are used and the LoRA configuration.

    Raises:
        RuntimeError: If any of the argument checks fail.
        ValueError: If the LoRA end layer index is not greater than the start layer index.
        KeyError: If an invalid LoRA module target is specified.
    """

    def _common_checks(args):
        sc.check_exist(args.config, 'Config file')
        sc.check_ext(args.config, '.json', 'Config file')
        config = PipelineConfig(args.config, verbose=False)

        sc.check_weights_exist(config.l.weight_dir)
        if config.e and config.e.weight_dir:
            sc.check_weights_exist(config.e.weight_dir)
        if config.t and config.t.weight_dir:
            sc.check_weights_exist(config.t.weight_dir)

        sc.check_supported_tokenizer(config.l)

        input_mode = sc.check_input_jsonl(args.dataset)
        if input_mode != 'text':
            logger.error("Only jsonl files with both 'text' and 'label' keys are allowed for QALFT.")

        sc.check_tokenizer_exist(config.l.weight_dir)
        if args.preformatter is not None:
            sc.check_exist(args.preformatter, 'Preformatter json file')
            sc.check_ext(args.preformatter, '.json', 'preformatter')

        output_dir = os.path.join(config.l.weight_dir, args.lora_name)
        existing_lora = os.path.exists(output_dir) and not args.force_overwrite

        sc.check_exist(args.quantized_model_folder, 'quantized model directory')
        sc.check_isdir(args.quantized_model_folder, 'quantized model directory')

        accepted_modules = [v for k, v in config.l.fc_names['attn'].items() if k not in ['name', 'qkv']] + [
            v for k, v in config.l.fc_names['mlp'].items() if k not in ['name', 'gateup']
        ]
        for target in args.lora_target_modules:
            if target not in accepted_modules:
                logger.error(
                    f'{target} is not a valid lora module target. Accepted targets: {accepted_modules}', err=KeyError
                )

        # taken from check_lora_config sanity_check
        curr_idx = 0
        for target in args.lora_target_modules:
            j = accepted_modules.index(target)
            if j < curr_idx:
                logger.error(
                    f'QALFT lora_target_modules is ordered incorrectly: {args.lora_target_modules}.'
                    f'\nPlease arrange the lora target modules to be in this order: {accepted_modules}'
                )
            curr_idx = j

        sc.check_positive_int(args.num_epoch, 'num_epoch')
        sc.check_positive_int(args.batch_size, 'batch_size')

        sc.check_between_inclusive(args.lora_start_layer_idx, 0, config.l.num_hidden_layers - 1, 'lora_start_layer_idx')
        sc.check_between_inclusive(args.lora_end_layer_idx, -1, config.l.num_hidden_layers - 1, 'lora_end_layer_idx')
        if args.lora_end_layer_idx > -1 and not (args.lora_end_layer_idx >= args.lora_start_layer_idx):
            logger.error(
                'lora_end_layer_idx must be greater or equal to lora_start_layer_idx or -1 (last layer)',
                err=ValueError,
            )

        sc.check_between_inclusive(args.buffer_scale, 1.0, 10.0, 'buffer_scale')

        # Two cases:
        # 1. Pretrained lora
        # 2. Finetuned lora that is to be further finetuned
        if existing_lora:
            logger.info('Found existing lora')
            lora_config_path = os.path.join(output_dir, 'adapter_config.json')
            with open(lora_config_path) as f:
                lora_config = json.load(f)

            config_lora_dropout = lora_config.get('lora_dropout', 0.0)
            if args.lora_dropout != config_lora_dropout:
                logger.warning(
                    f'`--lora_dropout` arg provided but found dropout {config_lora_dropout} in lora config. '
                    'Overwriting with `--lora_dropout`.'
                )
                lora_config['lora_dropout'] = args.lora_dropout
            if lora_config['lora_dropout'] == 0.0:
                logger.info('No dropout level specified for lora training. Defaulting to 0.0')

            # The following args below should follow existing ones in the config if available
            if lora_config.get('r', None) is None:
                logger.error('Expected existing lora config to have lora rank ("r") but found None!', err=KeyError)
            if lora_config.get('lora_alpha', None) is None:
                logger.error('Expected existing lora config to have ("lora_alpha") but found None!', err=KeyError)
            if lora_config.get('target_modules', None) is None:
                logger.error(
                    'Expected existing lora config to have lora target modules ("target_modules") but found None!',
                    err=KeyError,
                )
            if lora_config.get('lora_start_layer_idx', None) is None:
                logger.error(
                    'Expected existing lora config to have "lora_start_layer_idx" but found None!', err=KeyError
                )
            if lora_config.get('lora_end_layer_idx', None) is None:
                logger.error('Expected existing lora config to have "lora_end_layer_idx" but found None!', err=KeyError)
            if not config.rotate and lora_config.get('rotate', False):
                logger.error('Expect LoRA to be unrotated.')
            if config.rotate and lora_config.get('rotate', False):
                if config.rotate_seed != lora_config.get('rotate_seed', 0):
                    logger.error(f'Expect rotated LoRA to have rotate_seed={config.rotate_seed}.')
                if config.rotate_mode != lora_config.get('rotate_mode', 'hadamard'):
                    logger.error(f'Expect rotated LoRA to have rotate_mode={config.rotate_mode}.')

        else:
            logger.info('Training new lora from scratch')
            required_args = []
            if args.lora_rank is None:
                required_args.append('lora_rank')
            if len(args.lora_target_modules) == 0:
                required_args.append('lora_target_modules')
            if len(required_args) > 0:
                logger.error(f'Required arguments not present when training lora from scratch: {required_args}')
            lora_config = {
                'r': args.lora_rank,
                'lora_alpha': args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
                'target_modules': args.lora_target_modules,
                'lora_dropout': args.lora_dropout,
                'lora_start_layer_idx': args.lora_start_layer_idx,
                'lora_end_layer_idx': config.l.num_hidden_layers - 1
                if args.lora_end_layer_idx == -1
                else args.lora_end_layer_idx,
            }
        return config, lora_config, existing_lora

    def _converter_checks(args, config):
        quantized_model_list = utils.get_sorted_path_list(
            os.path.join(args.quantized_model_folder, 'llm'),
            ['.tflite', '.mlir'],
        )
        if len(quantized_model_list) != config.l.num_hidden_layers + 1:
            logger.error(
                f'Expected {config.l.num_hidden_layers + 1} TFLites/MLIRs in `quantized_model_folder` but only got '
                f'{len(quantized_model_list)} files.'
            )
        quantized_model_infos = []
        for f in quantized_model_list:
            sc.check_dynamic_shape(f)
            quantized_model_info = quantized_model_utils.extract_llm_quantized_model_info(f)
            quantized_model_infos.append(quantized_model_info)

        # only converter requires QuantizerStub during QALFT
        if not hasattr(mtk_quantization.pytorch.functional, 'QuantizerStub'):
            logger.error('Quantization-aware lora finetune requires mtk_quantization >= 8.0.0')

        return quantized_model_list, quantized_model_infos

    def _mlkits_checks(args):
        quantized_model_list = list(pathlib.Path(args.quantized_model_folder).glob('*.pickle'))
        if len(quantized_model_list) != 1:
            logger.error(
                f'Expected 1 optimized pickle in `quantized_model_folder` but got {len(quantized_model_list)} files.'
            )

        if args.opt_config is None:
            logger.error('Optimization configuration file is required for MLKits backend.')

        if (args.prompt_token_size is not None or args.gen_token_size is not None) and args.cache_size is None:
            logger.error(
                'The `--prompt-token-size` or `--gen-token-size` is given, --cache-size must be '
                'specified too. Please check the input arguments.'
            )

        if args.training_precision == 'float16':
            logger.error('Training precision float16 is not supported for MLKits backend yet.')

        return quantized_model_list

    config, lora_config, existing_lora = _common_checks(args)
    quantized_model_infos = None
    if args.backend == const.CONVERTER:
        quantized_model_list, quantized_model_infos = _converter_checks(args, config)
    else:
        quantized_model_list = _mlkits_checks(args)

    return quantized_model_list, quantized_model_infos, lora_config, existing_lora


def print_args(args, lora_config):
    """Prints the arguments for verification.

    Args:
        args (argparse.Namespace): The parsed arguments.
        lora_config (dict): The LoRA configuration.
    """

    def print_common_args(args, lora_config):
        # logger.info the arguments for the main process.
        # rank=0 in DDP; rank=-1 indicates a non-distributed environment.
        logger.info('Please check if all arguments are correct:')
        logger.info(f'Config file:                             {args.config}')
        logger.info(f'Quantized base model folder:             {args.quantized_model_folder}')
        logger.info(f'Use Single BMM Attention Graph:          {args.use_single_bmm_attention}')
        logger.info(f'Training dataset:                        {args.dataset}')
        logger.info(f'Training precision:                      {args.training_precision}')
        logger.info(f'LoRA name:                               {args.lora_name}')
        logger.info(f'Preformatter json:                       {args.preformatter}')
        logger.info(f'BOS mode:                                {args.bos_mode}')
        logger.info(f'Target modules:                          {lora_config["target_modules"]}')
        logger.info(f'LoRA rank:                               {lora_config["r"]}')
        logger.info(f'LoRA alpha:                              {lora_config["lora_alpha"]}')
        logger.info(f'LoRA dropout:                            {lora_config["lora_dropout"]}')
        logger.info(f'LoRA start layer index:                  {lora_config["lora_start_layer_idx"]}')
        logger.info(f'LoRA end layer index:                    {lora_config["lora_end_layer_idx"]}')
        logger.info(f'Batch size:                              {args.batch_size}')
        if deepspeed_installed:
            logger.info(f'Deepspeed:                               {args.deepspeed}')
            if args.deepspeed:
                logger.info(f'Train micro batch size per gpu:          {args.train_micro_batch_size_per_gpu}')
        logger.info(f'Learning rate:                           {args.learning_rate}')
        logger.info(f'Num epoch:                               {args.num_epoch}')
        if args.epochs_per_checkpoint > 0:
            logger.info(f'Num epochs per checkpoint:               {args.epochs_per_checkpoint}')
        logger.info(
            f'Compute loss on:                         {"response" if args.response_only_loss else "prompt+response"}'
        )
        logger.info(f'Force overwrite:                         {args.force_overwrite}')
        logger.info(f'mtk_llm_sdk version:                    {__version__}')

    def print_converter_args():
        logger.debug(f'mtk_quantization version:               {mtk_quantization.__version__}')

    def print_mlkits_args():
        logger.info(f'Workspace:                               {args.workspace}')
        logger.info(f'Output format:                           {args.format}')
        logger.info(f'Reduce LoRA:                             {args.reduce_lora}')
        logger.info(f'Optimization config:                     {args.opt_config}')
        logger.info(f'Prompt token size:                       {args.prompt_token_size}')
        logger.info(f'Gen token size:                          {args.gen_token_size}')
        logger.info(f'Cache size:                              {args.cache_size}')
        if args.int8_cache:
            logger.info('Force int8 cache I/O:                     True')
        logger.info(f'Number of MLIR/TFLite chunks:            {args.num_chunks}')
        logger.info(f'mlkits version:                         {getattr(mlkits, "__version__", "dev")}')

    if is_main_process():
        print_common_args(args, lora_config)
        if args.backend == const.CONVERTER:
            print_converter_args()
        else:
            import mlkits

            print_mlkits_args()


def print_main(msg, level='info'):
    """Prints messages if the current process is the main process.

    Args:
        msg: The message to print.
        level: The log level to use.
    """
    if is_main_process():
        if level == 'info':
            logger.info(msg)
        elif level == 'warning':
            logger.warning(msg)
        elif level == 'error':
            logger.error(msg)
        else:
            logger.error(f'Unexpected value for `level`: {level}')


def gen_quant_config(
    model,
    weight_bit,
    weight_sym,
    act_bit,
    act_sym,
    example_inputs,
    output_path,
    per_channel_weight=True,
    weight_quantizer='LastValueQuantizer',
    activation_quantizer='EMAQuantizer',
):
    """Generates a quantization configuration for the model.

    Args:
        model (torch.nn.Module): The model to be quantized.
        weight_bit (int): The bitwidth for weight quantization.
        weight_sym (bool): Whether to use symmetric quantization for weights.
        act_bit (int): The bitwidth for activation quantization.
        act_sym (bool): Whether to use symmetric quantization for activations.
        example_inputs (torch.Tensor): Example inputs for the model.
        output_path (str): The path to save the generated configuration.
        per_channel_weight (bool, optional): Whether to use per-channel quantization for weights. Default is True.
        weight_quantizer (str, optional): The type of quantizer to use for weights. Default is 'LastValueQuantizer'.
        activation_quantizer (str, optional): The type of quantizer to use for activations. Default is 'EMAQuantizer'.
    """
    try:
        config_generator = mtk_quantization.pytorch.MixedPrecisionConfigGenerator(model)
    except Exception:
        config_generator = mtk_quantization.pytorch.ConfigGenerator(model)
    config_generator.weights_bitwidth = weight_bit
    config_generator.activations_bitwidth = act_bit
    config_generator.use_weights_symmetric_quantization = weight_sym
    config_generator.use_activations_symmetric_quantization = act_sym
    config_generator.use_params_symmetric_quantization = True
    config_generator.use_per_output_channel_quantization = per_channel_weight
    config_generator.weights_quantizer_type = weight_quantizer
    config_generator.activations_quantizer_type = activation_quantizer
    config_generator.ignore_union_quantizer_targets = True

    config_generator.export_config(output_path, example_inputs)


def gen_deepspeed_config(args):
    """Generates a DeepSpeed configuration dictionary.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        dict: A dictionary containing the DeepSpeed configuration.
    """
    return {
        'train_batch_size': args.batch_size,
        'train_micro_batch_size_per_gpu': args.train_micro_batch_size_per_gpu,
        'steps_per_print': 500,
        'fp16': {
            'enabled': args.training_precision == 'float16',
            'loss_scale': 0,  # 0 means dynamic loss scaling. Non-zero values mean static fixed loss scaling.
            'initial_scale_power': 16,  # initial dynamic loss scale value, 2^16.
        },
    }


def train(
    pipeline,
    fq_model,
    train_dataset,
    embedding_layer,
    args,
    tokenizer,
    output_dir,
    out_lora_config,
    save_reduced_lora=None,
):
    """Trains the model on the given dataset.

    Args:
        pipeline (FloatPipeline): The FloatPipeline object.
        fq_model (torch.nn.Module): The fake-quantized model to train.
        train_dataset (torch.utils.data.Dataset): The dataset to train the model on.
        embedding_layer (torch.nn.Module): The embedding layer of the model.
        args (argparse.Namespace): The parsed arguments.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenization.
        output_dir (str): The directory to save the trained model.
        out_lora_config (str): The path to save Lora config.
        save_reduced_lora (callable, optional): The function to save reduced Lora. Default is None.
            If not provided, save Lora with default behavior.

    Methods:
        _training_step(model, inputs, labels):
            Performs a single training step.

        _ds_training_step(deepspeed_engine, config, inputs, labels):
            Performs a single training step using DeepSpeed.

        _create_optimizer(opt_model, learning_rate):
            Creates an optimizer for the model.

        _create_scheduler(num_training_steps, optimizer):
            Creates a learning rate scheduler.

        _get_model_param_count(model, trainable_only=False):
            Returns the number of parameters in the model.

        _prepare_inputs(inputs, embedding_layer):
            Prepares the inputs for the model.
    """

    def _training_step(model, inputs, labels):
        """Performs a single training step.

        Args:
            model (torch.nn.Module): The model to be trained.
            inputs (tuple): The input data for the model.
            labels (torch.Tensor): The labels for the input data.

        Returns:
            torch.Tensor: The detached loss tensor.
        """
        from torch.nn import CrossEntropyLoss

        model.train()

        if pipeline.config.l.use_split_mask:
            logits = model(inputs[0], inputs[1], inputs[2], inputs[3], *inputs[4], inputs[5])[0]
        else:
            logits = model(inputs[0], inputs[1], inputs[2], inputs[3], *inputs[4])[0]

        shift_logits = logits[:, :-1, : model.config.vocab_size].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss()  # finds mean and reduces by default
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        loss.backward()

        del inputs

        return loss.detach()

    def _ds_training_step(deepspeed_engine, config, inputs, labels):
        """Performs a single training step using DeepSpeed.

        Args:
            deepspeed_engine (deepspeed.DeepSpeedEngine): The DeepSpeed engine.
            config (Config): The model configuration.
            inputs (tuple): The input data for the model.
            labels (torch.Tensor): The labels for the input data.

        Returns:
            torch.Tensor: The detached loss tensor.
        """
        from torch.nn import CrossEntropyLoss

        deepspeed_engine.train()

        if pipeline.config.l.use_split_mask:
            inputs = (
                [x.to(deepspeed_engine.device) for x in inputs[:4]]
                + [[x.to(deepspeed_engine.device) for x in inputs[4]]]
                + [inputs[5].to(deepspeed_engine.device)]
            )
        else:
            inputs = [x.to(deepspeed_engine.device) for x in inputs[:4]] + [
                [x.to(deepspeed_engine.device) for x in inputs[4]]
            ]
        deepspeed_engine.zero_grad()

        if pipeline.config.l.use_split_mask:
            logits = model(inputs[0], inputs[1], inputs[2], inputs[3], *inputs[4], inputs[5])[0]
        else:
            logits = model(inputs[0], inputs[1], inputs[2], inputs[3], *inputs[4])[0]

        shift_logits = logits[:, :-1, : config.vocab_size].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        deepspeed_engine.backward(loss)
        deepspeed_engine.step()

        return loss.detach()

    def _create_optimizer(opt_model, learning_rate):
        """Creates an optimizer for the model.

        Args:
        opt_model (torch.nn.Module): The model to be optimized.
        learning_rate (float): The learning rate for the optimizer.

        Returns:
        torch.optim.Optimizer: The created optimizer.
        """
        from torch.optim import AdamW

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in opt_model.named_parameters() if p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        optimizer_kwargs = {'lr': learning_rate}
        adam_kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        }
        optimizer_kwargs.update(adam_kwargs)

        return AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def _create_scheduler(num_training_steps, optimizer):
        """Creates a learning rate scheduler.

        Args:
        num_training_steps (int): The number of training steps.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.

        Returns:
        torch.optim.lr_scheduler.LambdaLR: The created learning rate scheduler.
        """
        from torch.optim.lr_scheduler import LambdaLR

        def _get_linear_schedule_with_warmup_lr_lambda(
            current_step: int, *, num_warmup_steps: int, num_training_steps: int
        ):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, -1)

    def _get_model_param_count(model, trainable_only=False):
        """Returns the number of parameters in the model.

        Args:
        model (torch.nn.Module): The model.
        trainable_only (bool, optional): Whether to count only trainable parameters. Default is False.

        Returns:
        int: The number of parameters in the model.
        """
        return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

    def _prepare_inputs(inputs, embedding_layer, pipeline):
        labels = inputs.pop('labels', inputs['input_ids'])
        input_ids = inputs['input_ids']
        padding = inputs.pop('attention_mask')
        # ignore padding token
        labels = torch.where(labels == tokenizer.pad_token_id, LABEL_PAD_TOKEN_ID, labels)

        # multi-modal embeddings
        mm_embeddings = inputs.get('mm_embeddings', None)
        if mm_embeddings is not None:  # Multimodal case
            mm_kwargs = inputs['mm_kwargs']
            pipeline.text_embedding_layer = embedding_layer
            print_main(f'Number of multimodal batch to prepare: {len(mm_embeddings)}')
            print_main(f'Number of input_ids to prepare: {len(input_ids)}')
            assert len(input_ids) == len(mm_embeddings) == len(mm_kwargs), 'All batch size must equal'
            input_embeds = []
            for i in range(len(input_ids)):
                # assert mm_embeddings[i].shape[1] == mm_num_image_tokens[i], (
                #    'num_image_tokens must equal to mm_embedding length.'
                # )
                print_main(f'input_ids {input_ids[i].shape}')
                print_main(f'mm_embeddings {mm_embeddings[i].shape}')
                input_embed, _kwargs = pipeline.get_embeds.forward(
                    input_ids[i].unsqueeze(0), [mm_embeddings[i]], **mm_kwargs[i]
                )
                input_embeds.append(input_embed)
            input_embeds = torch.cat(input_embeds, dim=0)
            print_main(f'Concated multimodal embedding batch size: {input_embeds.shape}')

        else:  # Text only case
            input_embeds = embedding_layer(input_ids)
        input_embeds = (input_embeds * padding.unsqueeze(2).to(input_embeds.device)).to(
            PRECISION[args.training_precision]
        )
        input_list = [input_embeds, inputs.pop('mask'), inputs.pop('cos'), inputs.pop('sin'), inputs.pop('cache')]
        if pipeline.config.l.use_split_mask:
            input_list.append(inputs.pop('split_mask'))
        del inputs
        return input_list, labels

    def _prepare_multimodal_inputs(inputs, embedding_layer, pipeline):
        labels = inputs.pop('labels', inputs['input_ids'])
        input_ids = inputs['input_ids']
        padding = inputs.pop('attention_mask')
        mm_kwargs = inputs.pop('mm_kwargs', [])
        # ignore padding token
        labels = torch.where(labels == tokenizer.pad_token_id, LABEL_PAD_TOKEN_ID, labels)
        multimodal_inputs = inputs.pop('mm_inputs', [])
        pipeline.get_embeds.text_embedding_layer = embedding_layer
        input_embeds = encode_multimodal_input(input_ids, multimodal_inputs, mm_kwargs, pipeline)
        input_embeds = (
            (input_embeds * padding.unsqueeze(2)).to(input_embeds.device).to(PRECISION[args.training_precision])
            # .squeeze(0)
        )
        input_list = [
            input_embeds.detach().cpu(),
            inputs.pop('mask'),
            inputs.pop('cos'),
            inputs.pop('sin'),
            inputs.pop('cache'),
        ]
        del inputs
        return input_list, labels

    def encode_multimodal_input(input_ids, multimodal_inputs, mm_kwargs, pipeline):
        llm_embeds = []
        if not multimodal_inputs:
            multimodal_inputs = len(input_ids) * []
        if not mm_kwargs:
            mm_kwargs = len(input_ids) * {}
        assert len(input_ids) == len(multimodal_inputs) == len(mm_kwargs), 'All batch size must equal'
        for i in range(len(input_ids)):
            one_input_ids = input_ids[i]
            mm_inputs = multimodal_inputs[i]
            kwargs = mm_kwargs[i]
            kwargs['pipeline_type'] = 'float'
            logger.debug(f'mm_kwargs: {kwargs}')
            multimodal_embeds, _, kwargs = pipeline.process_multimodal_input(mm_inputs, **kwargs)
            one_input_ids, kwargs = pipeline.forward_pre_getembed_hook(one_input_ids, **kwargs)
            input_embeds, kwargs = pipeline.get_embeds.forward(one_input_ids.unsqueeze(0), multimodal_embeds, **kwargs)
            llm_input, kwargs = pipeline.forward_pre_llm_hook(input_embeds, **kwargs)
            llm_embeds.append(llm_input)
        return torch.cat(llm_embeds, dim=0)

    print_main('***** Running training *****')
    num_examples = len(train_dataset)
    num_update_steps_per_epoch = int(num_examples // args.batch_size)
    num_train_epochs = args.num_epoch
    max_steps = int(args.num_epoch * num_update_steps_per_epoch)
    optimizer = _create_optimizer(fq_model, args.learning_rate)
    lr_scheduler = _create_scheduler(max_steps, optimizer)

    if args.deepspeed:
        accumulation_steps = int(args.batch_size // (args.train_micro_batch_size_per_gpu * dist.get_world_size()))
        print_main(f'Num GPU = {dist.get_world_size()}')
        print_main(f'Micro Batch Size per GPU = {args.train_micro_batch_size_per_gpu}')
        print_main(f'Gradient accumulation steps = {accumulation_steps}')
    print_main(f'Batch Size = {args.batch_size}')
    print_main(f'Num examples = {num_examples}')
    print_main(f'Num Epochs = {num_train_epochs}')
    print_main(f'Total optimization steps = {max_steps}')
    print_main(f'Number of trainable parameters = {_get_model_param_count(fq_model, trainable_only=True)}')

    if args.deepspeed:
        barrier()
        config = pipeline.config.l
        parameters = filter(lambda p: p.requires_grad, fq_model.parameters())

        model, optimizer, train_dataloader, _ = deepspeed.initialize(
            model=fq_model,
            args=args,
            model_parameters=parameters,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            training_data=train_dataset,
            collate_fn=DataCollator(pipeline),
            config=gen_deepspeed_config(args),
        )
    else:
        model = fq_model
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=DataCollator(pipeline, dtype=PRECISION[args.training_precision]),
        )

    model.zero_grad()

    best_state_dict = None
    best_loss = None
    for epoch in range(num_train_epochs):
        epoch_iterator = train_dataloader

        print_main(f'Epoch {epoch + 1}')
        epoch_loss = []
        pbar = tqdm(total=int(num_examples // args.batch_size) + int(num_examples % args.batch_size != 0))
        for inputs in epoch_iterator:
            if pipeline.has_encoder():
                inputs, labels = _prepare_multimodal_inputs(inputs, embedding_layer, pipeline)
            else:
                inputs, labels = _prepare_inputs(inputs, embedding_layer, pipeline)

            if args.deepspeed:
                loss = _ds_training_step(model, config, inputs, labels)
            else:
                loss = _training_step(model, inputs, labels)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
            epoch_loss.append(loss)
            pbar.update(1)
        pbar.close()
        epoch_loss = torch.stack(epoch_loss)
        loss_sum = torch.nansum(epoch_loss)
        if best_loss is None or loss_sum < best_loss:
            best_loss = loss_sum
            new_best = True
        else:
            new_best = False
        print_main(
            f'Epoch {epoch + 1} loss: {loss_sum} (sum), {torch.nanmean(epoch_loss)} '
            f'(mean){", new best" if new_best else ""}'
        )
        if args.deepspeed:
            # Deepspeed.runtime.engine call module_state_dict() to get model's parameters among different ranks.
            # Use dist.barrier() to ensure state_dict collects the parameters from all ranks.
            barrier()
            state_dict = model.module_state_dict()
            barrier()
        else:
            state_dict = model.state_dict()

        if new_best:
            best_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            best_state_dict = collections.OrderedDict(best_state_dict)
        if (
            args.epochs_per_checkpoint > 0
            and (epoch + 1) % args.epochs_per_checkpoint == 0
            and (epoch + 1) < num_train_epochs
        ):
            # Save intermediate checkpoint (state dict)
            save_directory = os.path.join(output_dir, f'epoch_{epoch + 1}')
            # Execute in main process only.
            if is_main_process():
                os.makedirs(save_directory, exist_ok=True)
                if save_reduced_lora is not None:
                    save_reduced_lora(model, save_directory)
                else:
                    out_lora_config_path = os.path.join(save_directory, 'adapter_config.json')
                    with open(out_lora_config_path, 'w') as f:
                        f.write(json.dumps(out_lora_config, indent=4))
                    save_lora(save_directory, state_dict, safe_serialization=args.safetensors)
                print_main(f'Checkpoint for epoch {epoch + 1} saved to {save_directory}')
    print_main('Training completed.')
    barrier()
    return best_state_dict


def save_lora(save_directory, state_dict, safe_serialization=False):
    """Saves the LoRA weights to the specified directory.

    Args:
    save_directory (str): The directory to save the LoRA weights.
    state_dict (dict): The state dictionary containing the model weights.
    safe_serialization (bool, optional): Whether to use safe serialization. Default is False.

    Raises:
    ValueError: If the provided path is a file instead of a directory.
    """
    if os.path.isfile(save_directory):
        logger.error(f'Provided path ({save_directory}) should be a directory, not a file', err=ValueError)

    os.makedirs(save_directory, exist_ok=True)

    to_save = {}
    for k, v in state_dict.items():
        if 'lora' in k and k.endswith('.weight'):
            subkey = k.split('.')[-2]
            corrected_subkey = f'{subkey.split("_lora")[0] + "_proj"}.{"lora" + subkey.split("_lora")[1]}'
            to_save[k.replace(subkey, corrected_subkey)] = v

    assert len(to_save) > 0

    if safe_serialization:
        from safetensors.torch import save_file as safe_save_file
        from safetensors.torch import storage_ptr, storage_size

        # Safetensors does not allow tensor aliasing.
        # We're going to remove aliases before saving
        ptrs = collections.defaultdict(list)
        for name, tensor in to_save.items():
            # Sometimes in the state_dict we have non-tensor objects.
            # e.g. in bitsandbytes we have some `str` objects in the state_dict
            if isinstance(tensor, torch.Tensor):
                ptrs[tensor.device, storage_ptr(tensor), storage_size(tensor)].append(name)
            else:
                # In the non-tensor case, fall back to the pointer of the object itself
                ptrs[id(tensor)].append(name)

        # These are all the pointers of shared tensors.
        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

        for _, names in shared_ptrs.items():
            # Here we just clone the shared tensors to avoid tensor aliasing which is
            # not supported in safetensors.
            for shared_tensor_name in names[1:]:
                to_save[shared_tensor_name] = to_save[shared_tensor_name].clone()
        safe_save_file(
            to_save,
            os.path.join(save_directory, 'adapter_model.safetensors'),
            metadata={'format': 'pt'},
        )
    else:
        torch.save(to_save, os.path.join(save_directory, 'adapter_model.bin'))


@contextmanager
def execute_main_process_first():
    """Let the main process (rank=0 or -1) to execute first.

    Allowing main process to create cache files first, speed up subsequent processes to directly read
    from the cache. If deepspeed is False, it will just execute the scripts in block.
    """
    if not is_main_process():
        barrier()
    yield  # break here to execute the latter code, then return here
    if is_main_process():
        barrier()


def is_main_process():
    """Checks if the current process is the main process.

    Returns:
    bool: True if the current process is the main process, False otherwise.
    """
    return int(os.getenv('LOCAL_RANK', -1)) in [-1, 0]


def barrier():
    """Synchronizes all processes in a distributed training setup.

    This function uses a barrier to ensure that all processes reach this point before continuing.
    """
    if deepspeed_launched:
        dist.barrier()


def _append_mm_token_strings(text, num_mm_tokens, mm_token_str):
    if num_mm_tokens is None:
        logger.error('Must contain num_img_tokens in data when qalfting mllm!', err=ValueError)
    if mm_token_str is None:
        logger.error('Must contain image_token in data when qalfting mllm!', err=ValueError)
    if mm_token_str not in text:
        logger.error(f'Input text must contains mm token place holder "{mm_token_str}"!', err=ValueError)
    return text.replace(mm_token_str, mm_token_str * num_mm_tokens, 1)


def _generate_and_tokenize_prompt(
    data_point, preformatter=None, tokenizer=None, response_only_loss=None, pipeline=None
):
    """Generate and tokenize prompt."""

    def tokenize(prompt, cutoff_len=2048):
        # Assume tokenizer is not sentencepiece
        tokenizer.add_eos_token = True
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result['input_ids'][-1] != tokenizer.eos_token_id and len(result['input_ids']) < cutoff_len:
            result['input_ids'].append(tokenizer.eos_token_id)
        # remove duplicate bos tokens
        add_bos = pipeline.get_tokenizer_add_bos() if pipeline is not None else tokenizer.add_bos_token
        bos_token_id = pipeline.config.l.bos_token_id if pipeline is not None else tokenizer.bos_token_id
        result['input_ids'] = utils.enforce_add_bos_mode(
            add_bos, np.array(result['input_ids'])[None, ...], bos_token_id
        ).tolist()[0]
        result.pop('attention_mask')
        return result

    preformatter = Preformatter(preformatter)

    cutoff_len = 2048
    if data_point.get('mm_embeddings', None) is not None:
        data_point['text'] = _append_mm_token_strings(
            data_point['text'],
            data_point.get('num_img_tokens', None),
            getattr(pipeline.config.ft, 'image_token', None),
        )
        cutoff_len = 4096
        # FIXME: use format prompt in future
        # data_point['text'], _ = pipeline.format_prompt(data_point['text'], **data_point['mm_kwargs'])
    if preformatter.used:
        full_prompt = preformatter.generate_prompt(
            data_point['text'], data_point.get('input', None), data_point['label']
        )
    else:
        full_prompt = data_point['text'] + data_point['label']
    print_main(f'Full prompt: {full_prompt}')
    tokenized_full_prompt = tokenize(full_prompt, cutoff_len=cutoff_len)
    if response_only_loss:
        if preformatter.used:
            user_prompt = preformatter.generate_prompt(data_point['text'], data_point.get('input', None))
        else:
            user_prompt = data_point['text']
        tokenized_user_prompt = tokenize(user_prompt, cutoff_len=cutoff_len)
        user_prompt_len = len(tokenized_user_prompt['input_ids']) - 1  # avoid counting eos for user prompt
        tokenized_full_prompt['labels'] = [LABEL_PAD_TOKEN_ID] * user_prompt_len + tokenized_full_prompt['input_ids'][
            user_prompt_len:
        ]  # could be sped up, probably

    # Add multimodal related data, if any
    mm_embeddings = data_point.get('mm_embeddings', None)
    if mm_embeddings is not None:
        tokenized_full_prompt['mm_embeddings'] = mm_embeddings
        print_main(f"tokenized_full_prompt['mm_embeddings']: {tokenized_full_prompt['mm_embeddings']}")
        tokenized_full_prompt['num_img_tokens'] = data_point.get('num_img_tokens', None)
        if tokenized_full_prompt['num_img_tokens'] is not None:
            print_main(f"tokenized_full_prompt['num_img_tokens']: {tokenized_full_prompt['num_img_tokens']}")
        tokenized_full_prompt['images'] = data_point.get('images', None)
        if tokenized_full_prompt['images'] is not None:
            print_main(f"tokenized_full_prompt['images']: {tokenized_full_prompt['images']}")
    return tokenized_full_prompt


def _process_multimodal_input(data_point, preformatter=None, tokenizer=None, response_only_loss=None, pipeline=None):
    def tokenize(prompt, cutoff_len=4096):
        tokenizer.add_eos_token = True
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result['input_ids'][-1] != tokenizer.eos_token_id and len(result['input_ids']) < cutoff_len:
            result['input_ids'].append(tokenizer.eos_token_id)
        result.pop('attention_mask')
        # remove duplicate bos tokens
        add_bos = pipeline.get_tokenizer_add_bos() if pipeline is not None else tokenizer.add_bos_token
        bos_token_id = pipeline.config.l.bos_token_id if pipeline is not None else tokenizer.bos_token_id
        result['input_ids'] = utils.enforce_add_bos_mode(
            add_bos, np.array(result['input_ids'])[None, ...], bos_token_id
        ).tolist()[0]
        return result

    preformatter = Preformatter(preformatter)

    multimodal_inputs = utils.get_multimodal_inputs_from_jsonl_line(data_point)
    if multimodal_inputs is None:
        multimodal_inputs = []
    kwargs = {}
    processed_lst = []
    if len(multimodal_inputs) > 0:
        for mm_input_path in multimodal_inputs:
            mm_input = copy.deepcopy(mm_input_path)
            mm_input_processed, kwargs = pipeline.load_multimodal_input(mm_input_path, **kwargs)
            mm_input_processed, kwargs = pipeline.forward_pre_preprocessor_hook(mm_input_processed, **kwargs)
            mm_input_processed, kwargs = pipeline.forward_preprocessor(mm_input_processed, **kwargs)
            processed_lst.append(mm_input)
    full_prompt = data_point['text']
    full_prompt, kwargs = pipeline.format_prompt(full_prompt, **kwargs)
    if preformatter.used:
        full_prompt = preformatter.generate_prompt(full_prompt, data_point.get('input', None))
    else:
        full_prompt = full_prompt
    full_prompt += data_point['label']
    logger.debug(f'Full prompt: {full_prompt}')
    tokenized_full_prompt = tokenize(full_prompt)
    if response_only_loss:
        user_prompt = data_point['text']
        user_prompt, kwargs = pipeline.format_prompt(user_prompt, **kwargs)
        if preformatter.used:
            user_prompt = preformatter.generate_prompt(user_prompt, data_point.get('input', None))
        else:
            user_prompt = user_prompt
        logger.debug(f'Response only loss user prompt: {user_prompt}')
        tokenized_user_prompt = tokenize(user_prompt)
        user_prompt_len = len(tokenized_user_prompt['input_ids']) - 1
        tokenized_full_prompt['labels'] = [LABEL_PAD_TOKEN_ID] * user_prompt_len + tokenized_full_prompt['input_ids'][
            user_prompt_len:
        ]  # could be sped up, probably
    tokenized_full_prompt['mm_inputs'] = processed_lst

    # Get additional kwargs
    full_prompt, kwargs = pipeline.forward_pre_tokenizer_hook(full_prompt, **kwargs)
    kwargs['quiet'] = True
    _, kwargs = (
        (full_prompt, kwargs)
        if pipeline.input_mode == 'tokens'
        else pipeline.forward_tokenizer(
            full_prompt,
            preformatter,
            mm_path=multimodal_inputs,
            **kwargs,
        )
    )
    tokenized_full_prompt['kwargs'] = kwargs
    return tokenized_full_prompt


def _converter_qalft(args, quantized_model_paths, quantized_model_infos, lora_config, existing_lora):
    """Performs Quantization-Aware LoRA FineTuning (QALFT) for the converter backend."""
    exp_name = os.path.basename(args.quantized_model_folder.rstrip('/'))
    quant_config_dir = os.path.join('qalft_configs', f'{exp_name}')

    if is_main_process():
        utils.recursive_remove_if_exist(quant_config_dir, recreate=True)
    barrier()

    # 2. Extract Quant Params from TFLites #########################
    with execute_main_process_first():
        config = PipelineConfig(args.config, verbose=False)
        quant_param_dict = {}
        print_main('Extracting quantization parameters from quantized models')
        for chunk_idx, quantized_model_path in tqdm(enumerate(quantized_model_paths), total=len(quantized_model_paths)):
            quant_param_dict = {
                **quant_param_dict,
                **qat_utils.extract_quant_params(
                    quantized_model_path, quantized_model_infos[chunk_idx], config, chunk_idx
                ),
            }
        # FIXME: temporary workaround for split mask, should refactor alongside input-output
        if quantized_model_infos[chunk_idx]['model_config'].get('use_split_mask', False):
            # find last stub count
            all_stubs_idx = [int(ele.split('.')[-1]) for ele in quant_param_dict if ele.startswith('stubs')]
            quant_param_dict[f'stubs.{max(all_stubs_idx) + 1}'] = quant_param_dict['split_mask_stub']
            quant_param_dict.pop('split_mask_stub')

    # 3. Create LoRA config ########################################
    output_dir = os.path.join(utils.get_dirpath(args.config), args.lora_name)
    lora_config_path = os.path.join(output_dir, 'adapter_config.json')
    out_lora_config = copy.deepcopy(lora_config)
    # Create Lora config for train from scratch case
    if not existing_lora and is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        if config.rotate:
            out_lora_config.update(
                {'rotate': True, 'rotate_seed': config.rotate_seed, 'rotate_mode': config.rotate_mode}
            )
        with open(lora_config_path, 'w') as f:
            f.write(json.dumps(out_lora_config, indent=4))
    barrier()

    pipeline = FloatPipeline(
        args.config,
        lora_config_path,
        task='qalft',
        input_mode='text',
        dtype=PRECISION[args.training_precision],
        debug=args.debug,
        distribute_layers=False if deepspeed_launched else not args.use_main_gpu,
        use_single_bmm_attention=args.use_single_bmm_attention,
        add_bos=args.bos_mode == 'see',
    )
    tokenizer = pipeline.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.l.pad_token_id
        print_main(
            "'pad_token_id' is not found in tokenizer. Using 'pad_token_id' from model config: "
            f'{config.l.pad_token_id}. Please verify that this behavior is intended.',
            level='warning',
        )

    tokenizer.padding_side = PADDING_SIDE  # Allow batched inference

    pipeline.lora_handler.l[0].dropout = out_lora_config['lora_dropout']  # reassign if user uses new dropout value

    if is_main_process():
        example_inputs = pipeline.llm.get_jit_trace_inputs()
    # 4. Generate QAT quant config #################################
    #
    # The bitwidth/symmetric quantizer settings will be updated in the qat_utils.map_quant_params
    # function. Therefore, a default bitwidth/symmetric setting is used here.
    quant_config_filepath = os.path.join(quant_config_dir, f'{args.lora_name}.json')

    # Generate `quant_config` in main process only to prevent inconsistencies.
    if is_main_process():
        gen_quant_config(
            pipeline.llm,
            8,  # weight_bit
            True,  # weight_sym
            16,  # act_bit
            True,  # act_sym
            example_inputs,
            quant_config_filepath,
            per_channel_weight=True,
            weight_quantizer='LastValueQuantizer',
            activation_quantizer='LastValueQuantizer',
        )
        # Delete all lora weights and lora_A act quant inside the quant config and re-export.
        # Delete all quant setting inside `constant_weights` since the weights of base model are
        # frozen during training. We will replace model weights with dequant weights and there is
        # no necessity to use fakequant for them.
        with open(quant_config_filepath) as f:
            data = f.read()
        quant_config = json.loads(data)
        quant_config['quantizer_targets']['constant_weights'] = []
        quant_config['quantizer_targets']['activations'] = [
            item
            for item in quant_config['quantizer_targets']['activations']
            if not item['target_name'].endswith('lora_A')
        ]
        os.remove(quant_config_filepath)
        with open(quant_config_filepath, 'w') as f:
            f.write(json.dumps(quant_config, indent=4))
    barrier()

    # 5. Create FQ Model ###########################################
    print_main('Getting FakeQuant Model')
    fq_model = pipeline.get_fakequant_model(quant_config_filepath, existing_lora)

    # 6. Map Quant Params to FQ Model ##############################
    print_main('Mapping Quant Params')
    qat_utils.map_quant_params(fq_model, quant_param_dict, args.buffer_scale, target_with_fakequant=['activation'])

    # Freeze Base Model and Quant Params
    fq_model.train()
    for name, param in fq_model.named_parameters():
        if 'lora' not in name or ('lora' in name and '_act_quantizer' in name):
            param.requires_grad = False

    # 7. Finetune Flow #############################################
    generate_and_tokenize_prompt = partial(
        _generate_and_tokenize_prompt,
        preformatter=args.preformatter,
        tokenizer=tokenizer,
        response_only_loss=args.response_only_loss,
        pipeline=pipeline,
    )
    process_multimodal_input = partial(
        _process_multimodal_input,
        preformatter=args.preformatter,
        tokenizer=tokenizer,
        response_only_loss=args.response_only_loss,
        pipeline=pipeline,
    )
    with execute_main_process_first():
        data = load_dataset('json', data_files=args.dataset)
        if pipeline.has_encoder():
            train_data = data['train'].shuffle().map(process_multimodal_input)
        else:
            train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)

        # handle tie word embeddings True case
        if len(pipeline.state_dict['llm'].keys()) == 0:
            embedding_layer = utils.get_embedding_layer(config.l, weight_dir=config.l.weight_dir)
        else:
            embedding_layer = utils.get_embedding_layer(config.l, state_dict=pipeline.state_dict['llm'])

    if config.rotate:
        rotate.rotate_qalft(
            fq_model,
            embedding_layer,
            os.path.join(args.quantized_model_folder, 'rotate.safetensors'),
            pipeline,
            rotate_lora=not lora_config.get('rotate', False),
        )

    best_state_dict = train(
        pipeline,
        fq_model,
        train_data,
        embedding_layer,
        args,
        tokenizer,
        output_dir,
        out_lora_config,
    )

    barrier()  # Make sure all the processes are finished and ready to save lora weights.

    # Execute in main process only to prevent redundant operations.
    if is_main_process():
        if existing_lora:
            output_dir = os.path.join(output_dir, f'epoch_{args.num_epoch}')
            os.makedirs(output_dir, exist_ok=True)
            lora_config_path = os.path.join(output_dir, 'adapter_config.json')
            if config.rotate:
                # since the lora has been rotated weights, they are considered rotated
                # thus need to dump rotate config in the QALFT'ed lora output dir
                out_lora_config.update(
                    {'rotate': True, 'rotate_seed': config.rotate_seed, 'rotate_mode': config.rotate_mode}
                )
            with open(lora_config_path, 'w') as f:
                f.write(json.dumps(out_lora_config, indent=4))
        save_lora(output_dir, best_state_dict, safe_serialization=args.safetensors)
        print_main(f'Lora weights saved to {output_dir}')

        # 8. Dump lora bins for cmdline ############################
        print_main('Creating per-layer lora bins for cmdline ...')
        pipeline = FloatPipeline(
            args.config,
            lora_config_path,
            task='export_lora',
            dtype=np.float16,
            debug=args.debug,
        )

        for chunk_idx, quantized_model_path in enumerate(quantized_model_paths):
            if chunk_idx == config.l.num_hidden_layers:
                continue
            if chunk_idx < lora_config['lora_start_layer_idx'] or chunk_idx > lora_config['lora_end_layer_idx']:
                continue
            subgraph = quantized_model_utils.get_subgraph_from_quantized_model(quantized_model_path)
            utils.create_lora_bin_for_cmdline(
                quantized_model_path,
                chunk_idx,
                pipeline,
                'float',
                subgraph=subgraph,
            )
        print_main(
            f'Per-layer LoRA weight bins exported to {os.path.join(args.quantized_model_folder, "llm", args.lora_name)}'
        )
        os.remove(quant_config_filepath)
        os.removedirs(quant_config_dir)


def _mlkits_qalft(args, graph_path, lora_config, existing_lora):
    """Performs Quantization-Aware LoRA FineTuning (QALFT) for the mlkits backend."""
    from mlkits.core.cluster import cluster_constants

    from .utils import mlkits_graph_utils, mlkits_model_utils, mlkits_utils

    with execute_main_process_first():
        config = PipelineConfig(args.config, verbose=False)

    lora_dir = os.path.join(utils.get_dirpath(args.config), args.lora_name)
    lora_config_path = os.path.join(lora_dir, 'adapter_config.json')
    lora_output_dir = os.path.join(args.workspace, args.lora_name)
    if is_main_process():
        os.makedirs(lora_output_dir, exist_ok=True)
        if not existing_lora:
            # Create LoRA config.
            os.makedirs(lora_dir, exist_ok=True)
            with open(lora_config_path, 'w') as f:
                f.write(json.dumps(lora_config, indent=4))
        barrier()

    pipeline = FloatPipeline(
        args.config,
        lora_config_path,
        task='qalft',
        input_mode='text',
        dtype=PRECISION[args.training_precision],
        backend=args.backend,
        debug=args.debug,
        distribute_layers=False if deepspeed_launched else not args.use_main_gpu,
        add_bos=args.bos_mode == 'see',
    )
    if existing_lora:
        lora_weights = pipeline.lora_handler.state_dicts[0]['llm']
        lora_weights = {
            next(iter(pipeline.lora_handler.llm_lora_state_dict_mapping[k].values())): lora_weights[k]
            for k in lora_weights
        }
    else:
        lora_weights = None

    if args.reduce_lora:
        pipeline = FloatPipeline(
            args.config,
            None,
            task='ptq',
            input_mode='text',
            dtype=PRECISION[args.training_precision],
            backend=args.backend,
            debug=args.debug,
            distribute_layers=False if deepspeed_launched else not args.use_main_gpu,
            add_bos=args.bos_mode == 'see',
        )

    tokenizer = pipeline.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.l.pad_token_id
        print_main(
            "'pad_token_id' is not found in tokenizer. Using 'pad_token_id' from model config: "
            f'{config.l.pad_token_id}. Please verify that this behavior is intended.',
            level='warning',
        )

    tokenizer.padding_side = PADDING_SIDE  # Allow batched inference

    print_main('Loading AIR graph.')
    graph = mlkits_graph_utils.load_graph(graph_path)

    enable_cluster = bool([op for op in graph.ops if cluster_constants.CLUSTER_CONFIG in op.meta])
    if args.reduce_lora and enable_cluster:
        logger.error('LoRA reduction does not support the graph optimized with cluster encoding.')

    mlkits_utils.prepare_overture(pipeline)
    if pipeline.overture is not None:
        mlkits_graph_utils.update_overture(pipeline, graph)

    enable_oreo = bool([op for op in graph.ops if op.isa('hadamard')])
    # TODO(zk.huang@mediatek.com): Support LoRA reduction for OREO.
    if args.reduce_lora and enable_oreo:
        logger.error('LoRA reduction does not support the graph optimized with OREO.')

    if enable_oreo and existing_lora:
        lora_offset = 2 + pipeline.llm.expected_num_pos_emb_inputs + pipeline.llm.expected_num_cache_inputs
        if pipeline.config.l.use_split_mask:
            lora_offset += 1
        mlkits_model_utils.MLKitsLoRAHandler(pipeline, len(lora_weights), lora_offset).rotate_lora(graph)

    # Create FQ Model.
    print_main('Getting FakeQuant Model')
    fq_model = mlkits_model_utils.get_fq_model_from_graph(
        pipeline,
        graph,
        lora_config,
        lora_weights,
        args.int8_cache,
        existing_lora=existing_lora,
        buffer_scale=args.buffer_scale,
        reduce_lora=args.reduce_lora,
        opt_config_path=args.opt_config,
    )
    save_reduced_lora = (
        partial(
            mlkits_model_utils.save_reduced_lora_results,
            graph=graph,
            deepspeed=args.deepspeed,
        )
        if args.reduce_lora
        else None
    )

    # Finetune Flow.
    generate_and_tokenize_prompt = partial(
        _generate_and_tokenize_prompt,
        preformatter=args.preformatter,
        tokenizer=tokenizer,
        response_only_loss=args.response_only_loss,
    )
    process_multimodal_input = partial(
        _process_multimodal_input,
        tokenizer=tokenizer,
        response_only_loss=args.response_only_loss,
        pipeline=pipeline,
    )
    with execute_main_process_first():
        data = load_dataset('json', data_files=args.dataset)
        if pipeline.has_encoder():
            train_data = data['train'].shuffle().map(process_multimodal_input)
        else:
            train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)

        if len(pipeline.state_dict['llm'].keys()) == 0:
            embedding_layer = utils.get_embedding_layer(config.l, weight_dir=config.l.weight_dir)
        else:
            embedding_layer = utils.get_embedding_layer(config.l, state_dict=pipeline.state_dict['llm'])

        if enable_oreo:
            logger.debug('Fuse input embedding with rotation matrix.')
            logger.dcheck(
                f'{mlkits_model_utils.INPUT_PREFIX}_0_post_rotation' in graph.meta.rotation
                and 'value' in graph.meta.rotation[f'{mlkits_model_utils.INPUT_PREFIX}_0_post_rotation'],
                'Missing rotation information in the graph.',
            )
            hadamard_weight = torch.from_numpy(
                graph.meta.rotation[f'{mlkits_model_utils.INPUT_PREFIX}_0_post_rotation']['value']
            )
            embedding_layer.weight.data.copy_(embedding_layer.weight.data @ hadamard_weight)

    best_state_dict = train(
        pipeline,
        fq_model,
        train_data,
        embedding_layer,
        args,
        tokenizer,
        lora_output_dir,
        lora_config,
        save_reduced_lora,
    )

    barrier()  # Make sure all the processes are finished and ready to save lora weights.

    # Save the best model state dict and serialize the results.
    if is_main_process():
        fq_model.load_state_dict(best_state_dict)

        if args.reduce_lora:
            mlkits_model_utils.serialize_qalft_results(args, fq_model, graph)
            return

        if not existing_lora:
            save_lora(lora_dir, best_state_dict, safe_serialization=args.safetensors)
            print_main(f'Lora weights saved to {lora_dir}')

        shutil.copyfile(lora_config_path, os.path.join(lora_output_dir, 'adapter_config.json'))
        save_lora(lora_output_dir, best_state_dict, safe_serialization=args.safetensors)
        print_main(f'Lora weights saved to {lora_output_dir}')

        mlkits_model_utils.serialize_qalft_results(
            args, fq_model, graph, os.path.join(lora_output_dir, 'adapter_config.json'), enable_oreo=enable_oreo
        )


@memory_peak_profile
def main(args=None):
    """Main function to perform the training and quantization process.

    This function parses the arguments, performs sanity checks, initializes the DeepSpeed engine if necessary,
    extracts quantization parameters, creates and saves the LoRA configuration,
    generates the quantization configuration, creates the FakeQuant model, maps quantization parameters,
    and performs the fine-tuning.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))
    args.deepspeed = deepspeed_launched
    args.use_main_gpu = args.deepspeed or args.use_main_gpu

    quantized_model_paths, quantized_model_infos, lora_config, existing_lora = args_sanity_checks(args)
    print_args(args, lora_config)

    if args.deepspeed:
        print_main('Initializing DeepSpeed Engine for Training.')
        deepspeed.init_distributed()
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    if args.backend == const.CONVERTER:
        global mtk_converter
        import mtk_converter

        _converter_qalft(args, quantized_model_paths, quantized_model_infos, lora_config, existing_lora)
    else:
        global mlkits
        import mlkits

        mlkits.setup_mlkits(
            {
                'workspace_path': args.workspace.as_posix(),
                'framework': 'pytorch',
                'environment': {'recursion_limit': 10000000},
            }
        )
        _mlkits_qalft(args, quantized_model_paths[0], lora_config, existing_lora)


if __name__ == '__main__':
    main()
