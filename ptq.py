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
"""Script to Post-Training Quantize (PTQ) LLM, generating a dynamic-shaped quantized TFLite or MLIR model."""

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import torch

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import FloatPipeline
from .utils import const, logger, overture_utils, quantized_model_utils, rotate, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.precision_config import PTQPrecisionConfig

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_ptq_llm'


class LastEncoderChunk(torch.nn.Module):
    """Dummy class to combine last encoder chunk and pre-projector hook for PTQ."""

    def __init__(self, encoder, pre_projector):
        """Init Dummy class to combine last encoder chunk and pre-projector hook for PTQ."""
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.pre_projector = pre_projector

    def forward(self, x, x1=None):
        """Forward encoder then pre-projector hook."""
        if x1 is None:
            return self.pre_projector.forward(self.encoder.forward(x))[0]
        return self.pre_projector.forward(self.encoder.forward(x, x1))[0]

    def get_jit_trace_inputs(self):
        """Get encoder jit trace inputs."""
        return self.encoder.get_jit_trace_inputs()

    def get_ptq_inputs(self, args, **kwargs):
        """Get encoder ptq inputs."""
        return self.encoder.get_ptq_inputs(args, **kwargs)

    def pop_remaining_unused_weights(self, state_dict):
        """Pops unused vision weights."""
        return self.encoder.pop_remaining_unused_weights(state_dict)


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    def _get_common_parser(parser):
        parser.add_argument(
            'config',
            type=str,
            help='[Required] Model config json file. '
            'Model config must be in same directory as all model weight bins and tokenizer files.',
        )
        parser.add_argument(
            '-l',
            '--lora_config',
            type=str,
            default=None,
            help='LoRA adapter config json file. Only need to provide if not using calibration dataset.',
        )
        parser.add_argument(
            '-d',
            '--calibration_dataset',
            type=str,
            default=None,
            help='Calibration dataset folder. Use `fake` for fake calibration data (for latency measurement only).',
        )
        parser.add_argument(
            '-e',
            '--evaluation_dataset',
            type=str,
            default=None,
            help='Evaluation dataset folder for mixed precision sensitivity analysis. '
            'Should be different from calibration dataset.',
        )
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
            '--zero_lora_inputs',
            action='store_true',
            help='Flag to force lora inputs to be zeros during calibration.',
        )
        parser.add_argument(
            '-c',
            '--act_clip_range',
            type=int,
            default=None,
            help='Value to clip activation ranges during calibration.',
        )
        parser.add_argument(
            '--use_single_bmm_attention',
            action='store_true',
            help='Whether to use single bmm attention graph. ',
        )
        parser.add_argument('--skip_check', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
        parser.add_argument('--encoder_only', action='store_true', help='Flag to ptq only encoder.')
        parser.add_argument(
            '--get_emb_minmax_from_cal_data',
            action='store_true',
            help='Whether to get the input embedding minmax from the calibration data for first llm layer. '
            'Defaults to False.',
        )
        parser.add_argument(
            '--skip_align_projector_decoder',
            action='store_true',
            help='Whether to skip aligning the dtype and quantized scale of encoder output with decoder input. '
            'Defaults to False.',
        )
        parser.add_argument(
            '--dummy_weights',
            action='store_true',
            help='Whether to use the torch default initialized weights instead of actual weights. Defaults to False.',
        )
        parser.add_argument(
            '--aopt',
            type=int,
            default=None,
            choices=[None, 0, 1, 2, 3],
            help='Optimization levels of activation precision.',
        )
        return parser

    def _get_converter_parser(subparser):
        parser = subparser.add_parser(
            'converter', help='Run PTQ for suppoorted LLM models using mtk_converter backend.'
        )
        parser = _get_common_parser(parser)
        parser.add_argument(
            '-o',
            '--output_folder',
            type=str,
            default=None,
            help='User-specified output folder to save quantized model to. '
            'Will automatically save quantized model to ./quantized_models/dynamic_shape folder using name of weights '
            'folder and other parameters set for this PTQ script by default.',
        )
        parser.add_argument(
            '--partial',
            type=str,
            default=None,
            help='Only PTQ certain layers instead of the full model. String should be in the format: '
            'eA-B,lX,Y,Z. Where ABXYZ are all integers. Integers following `e` indicate encoder layers, '
            'and integers following `l` indicate llm layers. Example: l3,5,7 (Only PTQ llm layers 3, 5, and 7)',
        )
        parser.add_argument(
            '-p',
            '--precision_config',
            type=str,
            default=None,
            help='Precision config json file path containing quantization schema for entire model. '
            'Will generate a default precision config json file and exit if missing. '
            'Refer to document for full list of precision choices.',
        )
        parser.add_argument(
            '-m',
            '--calibration_method',
            type=str,
            default='Overall',
            help='Calibration method to use for PTQ. Defaults to Overall. Refer to document for choices.',
        )
        parser.add_argument(
            '-w',
            '--weight_optimization',
            type=str,
            default=None,
            choices=['hessian', 'gradient'],
            help='Advanced weight optimization strategy to use during PTQ. Defaults to None.',
        )
        parser.add_argument(
            '--per_tensor_weight_quant',
            action='store_true',
            help='Use per-tensor weight quantization instead of per-channel.',
        )
        parser.add_argument(
            '--extra_converter_options',
            type=str,
            default=None,
            help='Path to json file containing additional mtk_converter attributes to set that are not handled by '
            'this script.',
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
        parser.add_argument(
            '--vsq_encoder',
            action='store_true',
            help='Whether to run Vector Scaled Quantization (VSQ) for the Encoder model. Defaults to False.',
        )
        parser.add_argument(
            '--vsq_llm',
            action='store_true',
            help='Whether to run Vector Scaled Quantization (VSQ) for the LLM model. Defaults to False.',
        )
        parser.add_argument(
            '--vsq_vector_size_encoder',
            type=int,
            default=128,
            choices=[16, 32, 64, 128],
            help='Vector Scaled Quantization (VSQ) input channel vector size. Defaults to 128.',
        )
        parser.add_argument(
            '--vsq_vector_size_llm',
            type=int,
            default=128,
            choices=[16, 32, 64, 128],
            help='Vector Scaled Quantization (VSQ) input channel vector size. Defaults to 128.',
        )
        parser.add_argument(
            '--pad_lm_head', type=int, default=None, help='Pad lm head to multiples of provided number.'
        )
        parser.add_argument(
            '--dump_qparams', action='store_true', help='Flag to dump the quantization parameters for customization.'
        )
        parser.add_argument(
            '--load_qparams', action='store_true', help='Flag to load the quantization parameters after customization.'
        )
        parser.add_argument(
            '--force_overwrite',
            action='store_true',
            help='Force overwrite of dynamic shape quantized model forlder if it already exists. Use with caution.',
        )
        parser.add_argument(
            '--use_opt_separate_bmm',
            action='store_true',
            help='Enable optimization for separate BMM case when relevant pattern exists.',
        )
        return parser

    def _get_mlkits_parser(subparser):
        parser = subparser.add_parser('mlkits', help='Run PTQ for suppoorted LLM models using mlkits backend.')
        parser = _get_common_parser(parser)
        parser.add_argument(
            '--workspace', type=pathlib.Path, help='Specifies the workspace directory for the output files.'
        )

        # Optimization.
        parser.add_argument(
            '--preflight', action='store_true', default=False, help='Enables preflight without actual optimization.'
        )
        parser.add_argument(
            '--opt-config',
            type=pathlib.Path,
            default=None,
            help='Specifies the path to a YAML file that contains optimization configurations.',
        )
        parser.add_argument(
            '-i8c', '--int8_cache', action='store_true', help='Flag to force cache inputs/outputs to be int8 data type.'
        )
        parser.add_argument(
            '--oreo',
            type=str,
            default=None,
            choices=[None, 'aggressive', 'auto'],
            help='Specifies the Oreo configuration.',
        )
        parser.add_argument(
            '--data-length',
            type=int,
            default=-1,
            help='Specifies the length of calibration data. -1 means use all calibration data.',
        )
        parser.add_argument(
            '--estimate-bpv',
            action='store_true',
            help='Whether to calculate bpv(bits-per-value).',
        )
        parser.add_argument(
            '--disable-fp16-softmax',
            action='store_true',
            help='Whether to disable FP16 softmax.',
        )
        parser.add_argument(
            '--ana-file',
            type=pathlib.Path,
            default=None,
            help='Path to MLKits ANA file.',
        )
        parser.add_argument(
            '--buffer-scale',
            type=float,
            default=1.0,
            help='Scaling factor for activation minmax as a buffer for those LoRA fakequantize operations. '
            'Defaults to None. Specify only when `calibration_dataset` and `lora_config` are both provided '
            '(Hotplug LoRA scenario).',
        )
        # Old Pruning/Quantization arguments for backward compatibility.
        parser.add_argument(
            '--sparsity',
            type=float,
            default=None,
            help='Specifies the sparsity for weights. None indicating no pruning will be applied to '
            'weights. Defaults to None.',
        )
        parser.add_argument(
            '--act-bits',
            type=int,
            default=None,
            help='Specifies the bit-width for activation quantization. Defaults to None.',
        )
        parser.add_argument(
            '--act-sym', action='store_true', default=False, help='Enables symmetric quantization for activations.'
        )
        parser.add_argument(
            '--weight-bits',
            type=int,
            default=None,
            help='Specifies the bit-width for weight quantization. Defaults to None.',
        )
        parser.add_argument(
            '--weight-sym', action='store_true', default=False, help='Enables symmetric quantization for weights.'
        )
        parser.add_argument(
            '--weight-per-channel',
            action='store_true',
            default=False,
            help='Enables per-channel quantization for weights.',
        )
        parser.add_argument(
            '--dynamic-quant', action='store_true', default=False, help='Enables dyanmic quantization for activation.'
        )

        # MLIR/TFLite Conversion.
        parser.add_argument(
            '--graph', type=pathlib.Path, default=None, help='Specifies the optimized graph for converting TFLite.'
        )
        parser.add_argument(
            '--prompt-token-size', type=int, help='Specifies the token size for the prompt mode in the TFLite model.'
        )
        parser.add_argument(
            '--gen-token-size',
            type=int,
            default=None,
            help='Specifies the token size for the generation mode in the TFLite model.',
        )
        parser.add_argument(
            '--cache-size', type=int, help='Specifies the cache size for the generation mode in the TFLite model.'
        )
        parser.add_argument(
            '--num-chunks', type=int, default=1, help='Specifies the number of chunks for the MLIR/TFLite model.'
        )
        parser.add_argument(
            '--lm-head-token-size',
            type=int,
            default=None,
            help='Specifies the token size for the language model (LM) head in the TFLite model.',
        )
        parser.add_argument(
            '--medusa-head-token-size',
            type=int,
            default=None,
            help='Specifies the token size for the Medusa head in the TFLite model.',
        )
        parser.add_argument(
            '--num-heads',
            type=int,
            default=0,
            help='Specifies the number of heads for the MLIR/TFLite model. For Medusa, specify to '
            '2 for LM head and Medusa head.',
        )
        # Old Conversion arguments for backward compatibility.
        parser.add_argument(
            '--num-tflite-chunks', type=int, default=1, help='Specifies the number of chunks for the TFLite model.'
        )
        parser.add_argument(
            '--num-tflite-heads',
            type=int,
            default=0,
            help='Specifies the number of heads for the TFLite model. For Medusa, specify to 2 for '
            'LM head and Medusa head.',
        )
        return parser

    parser = argparse.ArgumentParser(
        description='Run Post Training Quantization (PTQ) for suppoorted models using the chosen backend.',
        allow_abbrev=False,
    )
    subparser = parser.add_subparsers(dest='backend')
    _get_converter_parser(subparser)
    _get_mlkits_parser(subparser)
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )

    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Raises:
        RuntimeError: If any of the argument checks fail.
        ValueError: If the calibration method is invalid.
    """

    def _common_checks(args):
        sc.check_exist(args.config, 'Config file')
        sc.check_ext(args.config, '.json', 'Config file')
        config = PipelineConfig(args.config, verbose=False)

        if args.encoder_only and config.e is None:
            logger.error('Must provide encoder config when using encoder_only.', err=ValueError)

        if args.dummy_weights and args.calibration_dataset != 'fake':
            logger.error('Dummy weights should not be used if real calibration dataset provided!')

        if (args.dummy_weights or args.calibration_dataset == 'fake') and args.backend == 'mlkits':
            logger.error('Dummy weights or fake calibration dataset should not be used with mlkits backend!')

        if args.skip_align_projector_decoder and config.e is None:
            logger.error('Must provide encoder config when using skip_align_projector_decoder.', err=ValueError)

        if not args.dummy_weights:
            weight_dir = utils.get_dirpath(args.config)
            sc.check_weights_exist(weight_dir)

        sc.check_ptq_format(args.format)

        return config

    def _converter_checks(args, config):
        if args.lora_config is not None:
            if args.calibration_dataset != 'fake' and args.calibration_dataset is not None:
                logger.error(
                    'lora_config is not needed when using calibration dataset. '
                    'Lora configs will be inferred from calibration data.'
                )
            sc.check_exist(args.lora_config, 'Lora config file')
            sc.check_ext(args.lora_config, '.json', 'Lora config file')
            sc.check_lora_config(args.lora_config, config)
            with_lora = True
        else:
            if args.calibration_dataset == 'fake' or args.calibration_dataset is None:
                with_lora = False
            else:
                enc_lora_mapping_file = os.path.join(args.calibration_dataset, 'encoder', 'chunk_0', 'lora_mapper.txt')
                llm_lora_mapping_file = os.path.join(args.calibration_dataset, 'llm', 'chunk_0', 'lora_mapper.txt')
                with_lora = os.path.exists(enc_lora_mapping_file) or os.path.exists(llm_lora_mapping_file)

        if (
            args.output_folder is not None
            and os.path.exists(args.output_folder)
            and args.partial is None
            and not args.force_overwrite
        ):
            logger.error(
                f'Output folder {args.output_folder} already exists. '
                'Please manually delete the existing folder or use --force_overwrite if intending to overwrite.',
                err=FileExistsError,
            )

        if args.partial is not None:
            alphabets = ''
            for char in args.partial:
                if char.isalpha():
                    alphabets += char
            for char in alphabets:
                if char not in ['e', 'l']:
                    logger.error(f'Only expected the letters `e` and `l` in partial but got: {alphabets}')
            if len(alphabets) > 2:
                logger.error('`e` and `l` should only appear once each.')
            if 'e' in alphabets and config.e is None:
                logger.error('Cannot specify `e` in partial if model contains no encoder.')

        if args.precision_config is None and args.aopt is not None:
            logger.error('Expect `--precision_config` to not be None if `--aopt` is not None.')

        if args.precision_config is not None:
            sc.check_exist(args.precision_config, 'Precision Config')
            sc.check_ext(args.precision_config, '.json', 'Precision Config')

        precision_config = PTQPrecisionConfig(config, args.precision_config, args.aopt)

        if not with_lora and args.zero_lora_inputs:
            logger.error('zero_lora_inputs is only for lora models!')

        ee_index = config.l.early_exit_index
        is_ee = ee_index is not None
        num_decoder_layers = (
            config.l.early_exit_index + config.l.early_exit_num_layers if is_ee else config.l.num_hidden_layers
        )

        is_activation_static_quant = any(list(precision_config.is_activation_static_quant().values()))
        if is_activation_static_quant or args.weight_optimization is not None:
            if args.calibration_dataset is None:
                logger.error(
                    'Calibration dataset is required for either weight optimization'
                    ' or at least one precision in precision config.'
                )
            if args.calibration_dataset != 'fake':
                sc.check_exist(args.calibration_dataset, 'Calibration dataset')
                sc.check_isdir(args.calibration_dataset, 'calibration dataset')

                if len(os.listdir(os.path.join(args.calibration_dataset, 'llm'))) != num_decoder_layers + 1:
                    logger.error(
                        f'Expected {num_decoder_layers + 1} folders '
                        'in calibration dataset folder but got '
                        f'{len(os.listdir(args.calibration_dataset))} folders'
                    )

        valid_calibration_methods = utils.get_converter_calibration_methods()
        if args.calibration_method not in valid_calibration_methods:
            logger.error(f'Invalid calibration method. Must be one of {valid_calibration_methods}.', ValueError)

        if args.extra_converter_options is not None:
            sc.check_exist(args.extra_converter_options, 'Extra converter options json file')
            sc.check_ext(args.extra_converter_options, '.json', 'Extra converter options json file')

        if args.pad_lm_head is not None:
            weight, activation = precision_config.converter_to_standalone_precision_mapping(
                precision_config.tail_precision
            )
            if not (weight == 'int4' and activation == 'int16'):
                logger.warning('pad_lm_head is set for non 4W16A tail precision, which is not recommended!')
            if config.l.vocab_size % args.pad_lm_head == 0:
                logger.warning(f'vocab_size is divisible by {args.pad_lm_head}, pad_lm_head will have no effect!')

        cvt_ver = utils.get_converter_version(include_minor=True)
        if args.np_compat is not None and args.np_compat > cvt_ver[0]:
            logger.error(
                f'`np_compat` ({args.np_compat}) cannot be greater than current mtk_converter version ({cvt_ver[0]})'
            )
        if cvt_ver[0] < 9 or (cvt_ver[0] == 9 and cvt_ver[1] < 2):
            logger.error(f'Found converter version {cvt_ver[0]}.{cvt_ver[1]} but >= 9.2.0 is required')

        # check overture option
        if config.l.overture_dict is not None:
            overture_path = overture_utils.get_overture_path(config.l.overture_dict)
            sc.check_exist(overture_path, 'Overture path')

        if args.vsq_encoder:
            if precision_config.max_encoder_weight_bit is None:
                logger.error('Expected Encoder precision config for VSQ but none found.')
            sc.check_between_inclusive(precision_config.max_encoder_weight_bit, 1, 8)
            if args.format == 'tflite':
                logger.error('VSQ can only be used with MLIR and not TFLite.')
        if args.vsq_llm:
            sc.check_between_inclusive(precision_config.max_llm_weight_bit, 1, 8)
            if args.format == 'tflite':
                logger.error('VSQ can only be used with MLIR and not TFLite.')

        if args.dump_qparams and args.load_qparams:
            logger.error('Cannot dump and load qparams in the same PTQ run!')

        if args.use_single_bmm_attention and args.use_opt_separate_bmm:
            logger.error('Cannot use single bmm option and 16b8b separate bmm option together.')

    def _mlkits_checks(args):
        if args.workspace is None:
            logger.error('The `--workspace` argument  must be specified.')

        enc_lora_mapping_file = os.path.join(args.calibration_dataset, 'encoder', 'chunk_0', 'lora_mapper.txt')
        llm_lora_mapping_file = os.path.join(args.calibration_dataset, 'llm', 'chunk_0', 'lora_mapper.txt')
        if args.lora_config is not None:
            if os.path.exists(enc_lora_mapping_file) or os.path.exists(llm_lora_mapping_file):
                logger.error('lora_config is not needed when calibration dataset contains LoRA configs.')
            sc.check_exist(args.lora_config, 'Lora config file')
            sc.check_ext(args.lora_config, '.json', 'Lora config file')
            sc.check_lora_config(args.lora_config, config)
            if args.buffer_scale <= 0:
                logger.error(f'buffer_scale must be greater than 0, but got {args.buffer_scale}.')
            logger.info(f'Hotplug LoRA into the optimized model with buffer scale {args.buffer_scale}.')

        opt_args = [
            args.sparsity,
            args.act_bits,
            args.act_sym,
            args.weight_bits,
            args.weight_sym,
            args.weight_per_channel,
            args.dynamic_quant,
        ]
        if args.ana_file and any(opt_args):
            logger.error('Both `--ana-file` and separate opt arguments are given. Please check.')

        if args.opt_config and any(opt_args):
            logger.error(
                'Both `--opt_config` and separate opt arguments are given. Please Specify all '
                'optimization settings in `--opt_config`.'
            )

        if args.data_length != -1 and args.data_length <= 0:
            logger.error(f'The data length must be a positive integer or -1, but got {args.data_length}.')

        if args.graph is not None and not args.graph.exists():
            logger.error(f'The graph {args.graph} does not exist. Please check the value of input argument `--graph`.')

        if (args.prompt_token_size is not None or args.gen_token_size is not None) and args.cache_size is None:
            logger.error(
                'The `--prompt-token-size` or `--gen-token-size` is given, --cache-size must be '
                'specified too. Please check the input arguments.'
            )

        if args.num_tflite_chunks != 1 and args.num_chunks != 1:
            logger.error(
                f'`--num_tflite_chunks` is given: {args.num_tflite_chunks}. '
                'Please specify number of MLIR/TFLITE chunks by `--num_chunks` only.'
            )

        if args.num_tflite_heads != 0 and args.num_heads != 0:
            logger.error(
                f'`--num_tflite_heads` is given: {args.num_tflite_heads}. '
                'Please specify number of heads for the MLIR/TFLite model by `--num_heads` only.'
            )

        if args.medusa_head_token_size is not None:
            logger.error('Medusa tail are temporarily not supported in the current version.')

    config = _common_checks(args)
    if not args.skip_check:
        if args.backend == 'converter':
            _converter_checks(args, config)
        else:
            _mlkits_checks(args)


def print_converter_args(args, exp_name, precision_config, lora_cfgs):
    """Prints the arguments for verification for the converter backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
        exp_name (str): The experiment name.
        precision_config (PTQPrecisionConfig): The parsed precision config class.
        lora_cfgs (list): The list of LoRA configuration files.
    """
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Config file:                              {args.config}')
    logger.info(f'PTQ Backend:                              {args.backend}')
    logger.info(f'Output format:                            {args.format}')
    logger.info(f'Use Single BMM Attetion Graph:            {args.use_single_bmm_attention}')
    logger.info(f'Force Overwrite existing dynamic models:  {args.force_overwrite}')
    logger.info('Precision summary:')
    precision_config.print_precision_summary()
    if len(lora_cfgs) > 0 and lora_cfgs[0] != 'None':
        if len(lora_cfgs) == 1:
            logger.info(f'Lora config file:              {lora_cfgs[0]}')
        else:
            logger.info(f'Number of Lora config files:   {len(lora_cfgs)}')
        logger.info(f'Force zero lora inputs:                   {args.zero_lora_inputs}')
    if args.act_clip_range is not None:
        logger.info(f'Activation clip range:                    {args.act_clip_range}')
    logger.info(f'Output folder:                            {args.output_folder}')
    if args.calibration_dataset is not None:
        logger.info(f'Calibration dataset:                      {args.calibration_dataset}')
        logger.info(f'Calibration method:                       {args.calibration_method}')
    logger.info(f'Weight optimization:                      {args.weight_optimization}')
    logger.info(
        f'Weight quant granularity:                 {"Per-tensor" if args.per_tensor_weight_quant else "Per-channel"}'
    )
    logger.info(f'VSQ Encoder:                              {args.vsq_encoder}')
    if args.vsq_encoder:
        logger.info(f'VSQ vector size encoder:                  {args.vsq_vector_size_encoder}')
    logger.info(f'VSQ LLM:                                  {args.vsq_llm}')
    if args.vsq_llm:
        logger.info(f'VSQ vector size llm:                      {args.vsq_vector_size_llm}')
    if args.extra_converter_options is not None:
        logger.info(f'Extra converter options file:             {args.extra_converter_options}')
    if args.debug:
        logger.info(f'PTQ debug dir:                            {os.path.join("ptq_debug", exp_name)}')
    if args.pad_lm_head is not None:
        logger.info(f'Pad lm_head:                              {args.pad_lm_head}')
    if args.aopt is not None:
        logger.info(f'AOPT:                                     {args.aopt}')
    logger.info(f'Dump Quant Params:                        {args.dump_qparams}')
    logger.info(f'Load Quant Params:                        {args.load_qparams}')
    logger.info(f'Encoder only PTQ:                         {args.encoder_only}')
    logger.info(f'mtk_llm_sdk version:                     {__version__}')
    logger.info(f'mtk_converter version:                   {mtk_converter.__version__}')


def print_mlkits_args(args, lora_cfgs):
    """Prints the arguments for verification for the MLKits backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
        lora_cfgs (list): The list of LoRA configuration files.
    """
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Config file:                    {args.config}')
    logger.info(f'PTQ Backend:                    {args.backend}')
    logger.info(f'Output format:                  {args.format}')
    logger.info(f'Use Single BMM Attetion Graph:  {args.use_single_bmm_attention}')
    logger.info(f'Disable FP16 softmax:            {args.disable_fp16_softmax}')
    if len(lora_cfgs) > 0 and lora_cfgs[0] != 'None':
        if len(lora_cfgs) == 1:
            logger.info(f'Lora config file:               {lora_cfgs[0]}')
        else:
            logger.info(f'Number of Lora config files:    {len(lora_cfgs)}')
        logger.info(f'Force zero lora inputs:         {args.zero_lora_inputs}')
    if args.buffer_scale is not None:
        logger.info(f'Buffer scale:                   {args.buffer_scale}')
    if args.act_clip_range is not None:
        logger.info(f'Activation clip range:          {args.act_clip_range}')
    logger.info(f'Calibration dataset:            {args.calibration_dataset}')
    logger.info(f'Calibration data length:        {args.data_length}')
    logger.info(f'Workspace:                      {args.workspace}')
    logger.info(f'Preflight mode:                 {args.preflight}')
    if args.aopt is not None:
        logger.info(f'AOPT:                           {args.aopt}')
    if args.opt_config:
        logger.info(f'Optimization config:            {args.opt_config}')
    else:
        logger.info(f'Sparsity:                       {args.sparsity}')
        logger.info(f'Activation bitwidth:            {args.act_bits}')
        logger.info(f'Activation symmetric:           {args.act_sym}')
        logger.info(f'Weight bitwidth:                {args.weight_bits}')
        logger.info(f'Weight bitwidth:                {args.weight_sym}')
        logger.info(f'Weight per-channel:             {args.weight_per_channel}')
        logger.info(f'Dynamic quantization:           {args.dynamic_quant}')
    if args.int8_cache:
        logger.info('Force int8 cache I/O:            True')
    logger.info(f'OREO:                           {args.oreo}')
    logger.info(f'Graph:                          {args.graph}')
    logger.info(f'Prompt token size:              {args.prompt_token_size}')
    logger.info(f'Gen token size:                 {args.gen_token_size}')
    logger.info(f'Cache size:                     {args.cache_size}')
    logger.info(f'Nummber of chunks:              {args.num_chunks}')
    logger.info(f'LM head token size:             {args.lm_head_token_size}')
    logger.info(f'Number of heads:                {args.num_heads}')
    logger.info(f'Encoder only PTQ:               {args.encoder_only}')
    logger.info(f'Estimate BPV(bits-per-value):   {args.estimate_bpv}')
    logger.info(f'ANA file:                       {args.ana_file}')
    logger.info(f'mtk_llm_sdk version:           {__version__}')
    logger.info(f'mlkits version:                {getattr(mlkits, "__version__", "dev")}')


def regularize_mlkits_args(args):
    """Regularizes MLKits arguments."""
    args.evaluation_dataset = None
    args.zero_lora_inputs = False

    # Convert old options usage to the new one.
    args.num_chunks = args.num_tflite_chunks if args.num_tflite_chunks != 1 else args.num_chunks
    args.num_heads = args.num_tflite_heads if args.num_tflite_heads != 0 else args.num_heads

    # Disable fp16 softmax if not using split mask and not using singlebmm.
    config = PipelineConfig(args.config, verbose=False)
    if not config.l.use_split_mask and not args.use_single_bmm_attention:
        args.disable_fp16_softmax = True
        logger.warning('Disable fp16 softmax since not using split mask and singlebmm.')

    # TODO(Pearlie.Lin@mediatek.com): Support AOPT= 3.
    if args.aopt == 3:
        args.aopt = 2
        logger.warning('Setting AOPT to 2 as AOPT = 3 is not supported yet.')


def _get_lora_configs(args):
    """Gets LoRA configurations from arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        lora_cfgs (set): A set of LoRA configurations.
    """
    lora_cfgs = set()
    if args.calibration_dataset is not None:
        lora_mapping_file = os.path.join(args.calibration_dataset, 'llm', 'chunk_0', 'lora_mapper.txt')
        if os.path.exists(lora_mapping_file):
            with open(lora_mapping_file) as f:
                for lora_cfg in f.readlines():
                    if lora_cfg != '':
                        lora_cfgs.add(lora_cfg.rstrip('\n'))
    if args.lora_config is not None:
        lora_cfgs.add(args.lora_config)
    lora_cfgs = list(lora_cfgs)
    logger.debug(f'Lora configs:\n{lora_cfgs}')

    return lora_cfgs


def _get_lora_maps(args, pipeline, lora_cfgs):
    """Gets LoRA maps for encoder layers and decoder layers.

    Args:
        args (argparse.Namespace): The parsed arguments.
        pipeline (FloatPipeline): The Pipeline object.
        lora_cfgs (list): The list of LoRA configurations.

    Returns:
        encoder_lora_maps (list): The list of LoRA maps for encoder layers.
        llm_lora_maps (list): The list of LoRA maps for decoder layers.
    """
    num_encoder_chunks = len(pipeline.encoder_layers_per_chunk)
    encoder_lora_maps = [None for _ in range(num_encoder_chunks)]
    if pipeline.lora_handler.has_encoder_lora() and args.calibration_dataset != 'fake':
        for i in range(num_encoder_chunks):
            encoder_lora_maps[i] = []
            if args.calibration_dataset is not None:
                lora_mapping_file = os.path.join(args.calibration_dataset, 'encoder', f'chunk_{i}', 'lora_mapper.txt')
                with open(lora_mapping_file) as f:
                    for lora_cfg in [x for x in f.readlines() if x != '']:
                        lora_cfg_idx = lora_cfgs.index(lora_cfg.rstrip('\n'))
                        encoder_lora_maps[i].append(lora_cfg_idx)

    num_llm_chunks = len(pipeline.llm_layers_per_chunk)
    llm_lora_maps = [None for _ in range(num_llm_chunks)]
    if pipeline.lora_handler.has_llm_lora() and args.calibration_dataset != 'fake':
        for i in range(num_llm_chunks):
            llm_lora_maps[i] = []
            if args.calibration_dataset is not None:
                lora_mapping_file = os.path.join(args.calibration_dataset, 'llm', f'chunk_{i}', 'lora_mapper.txt')
                if pathlib.Path(lora_mapping_file).exists():
                    with open(lora_mapping_file) as f:
                        for lora_cfg in [x for x in f.readlines() if x != '']:
                            lora_cfg_idx = lora_cfgs.index(lora_cfg.rstrip('\n'))
                            llm_lora_maps[i].append(lora_cfg_idx)
                else:
                    if args.backend != 'mlkits':
                        logger.error(
                            'Simultaneously providing `--calibration_dataset` and `--lora_config` is only '
                            'supported by MLKits backend.'
                        )
                    # If the LoRA mapping file does not exist, it means that the `--lora_config` must be provided.
                    calibration_data_dir = pathlib.Path(args.calibration_dataset) / 'llm' / f'chunk_{i}'
                    llm_lora_maps[i] = [0 for path in calibration_data_dir.iterdir() if path.is_file()]

    return encoder_lora_maps, llm_lora_maps


def _converter_ptq(args):
    """Performs Post-Training Quantization (PTQ) for the converter backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """

    def _encoder_layer_ptq(
        layer,
        args,
        pipeline,
        layer_idx,
        output_folder,
        exp_name,
        config,
        lora_map=None,
        projector=False,
    ):
        lora_inputs = None if projector else pipeline.lora_handler.encoder_lora_inputs[layer_idx]

        example_inputs = layer.get_jit_trace_inputs()

        # JIT trace decoder layer into a temporary pt file
        logger.info(f'JIT tracing {"projector" if projector else "encoder"} layer {"" if projector else layer_idx}')
        temp_trace_filename = utils.jit_trace(layer, example_inputs, output_folder)

        with utils.temp_file(temp_trace_filename):
            # Quantize JIT traced layer
            logger.info(f'PTQ-ing {"projector" if projector else "encoder"} layer {"" if projector else layer_idx}')
            input_shapes, input_value_ranges, calib_data_gen = layer.get_ptq_inputs(
                args,
                exp_name=exp_name,
                lora_inputs=lora_inputs,
                calib_lora_map=lora_map,
            )[:3]  # TODO Support evaluation dataset

            converter = mtk_converter.PyTorchConverter.from_script_module_file(
                temp_trace_filename, input_shapes=input_shapes, experimental_debug_tensor_names=True
            )

            # mtk_converter flags
            converter.quantize = True
            if pipeline.precision_config.is_activation_static_quant()['encoder'] and layer_idx == 0:
                converter.prepend_input_quantize_ops = True
                converter.prepend_input_quantize_ops_indices = [0]
            else:
                converter.prepend_input_quantize_ops = False
                converter.prepend_input_quantize_ops_indices = []
            converter.append_output_dequantize_ops = False
            converter.append_output_dequantize_ops_indices = []
            if len(pipeline.precision_config.bypassed_ops) > 0:
                converter.extra_quantization_bypass_ops = pipeline.precision_config.bypassed_ops

            # Encoder mixed precision not supported yet
            converter.precision_proportion = {pipeline.precision_config.encoder_precision: 1.0}

            converter.use_per_output_channel_quantization = not args.per_tensor_weight_quant

            # Disable encoder hessian and gradient for now

            # calibration-related flags. Not applicable for precision=dynamic_quant or FP
            if args.calibration_dataset is not None:
                converter.calibration_data_gen = calib_data_gen
                converter.calibration_method = args.calibration_method
                if pipeline.precision_config.is_activation_static_quant()['encoder']:
                    converter.input_value_ranges = input_value_ranges
            else:
                converter.allow_missing_quantization_ranges = True

            # Additional user-specified options
            if args.extra_converter_options is not None:
                with open(args.extra_converter_options) as f:
                    options = json.load(f)
                sc.check_converter_options(converter, options)
                for k, v in options.items():
                    setattr(converter, k, v)

            if args.vsq_encoder:
                converter.use_vector_scaled_quantization = True
                converter.vector_scaled_quantization_vector_size = args.vsq_vector_size_encoder

            # Append layer id to filename
            fname = f'{exp_name}_{layer_idx}.{args.format}'
            if args.debug:
                os.makedirs('ptq_debug', exist_ok=True)
                os.makedirs(os.path.join('ptq_debug', exp_name), exist_ok=True)
                converter.quantization_debug_dir = os.path.join('ptq_debug', exp_name, 'encoder', f'chunk_{layer_idx}')

            if args.dump_qparams:
                os.makedirs(os.path.join('quant_params', exp_name, 'encoder'), exist_ok=True)
                converter.export_qparams_config_file = os.path.join(
                    'quant_params', exp_name, 'encoder', f'chunk_{layer_idx}.json'
                )
                converter.export_qparams_model_file = os.path.join(
                    'quant_params', exp_name, 'encoder', f'chunk_{layer_idx}.nrpmodel'
                )

            # Handle cases when the qparam model/config files are provided
            if args.load_qparams:
                qparam_path = os.path.join('quant_params', exp_name, 'encoder')
                logger.info(f'Loading models and quantization parameters for layer{layer_idx} from {qparam_path}')
                model_fn = os.path.join(qparam_path, f'chunk_{layer_idx}.nrpmodel')
                config_fn = os.path.join(qparam_path, f'chunk_{layer_idx}.json')
                converter = mtk_converter.Converter.from_model_proto_file(model_fn)
                converter.quantize = True
                converter.bypass_general_transformation_stage = True
                converter.import_qparams_config_file = config_fn

            output_path = os.path.join(output_folder, fname)
            # Convert to tflite/mlir
            if args.format == 'tflite':
                converter.convert_to_tflite(output_file=output_path)
            else:
                converter.convert_to_mlir(output_file=output_path)

            if projector:
                quantized_model_utils.finalize_projector_quantized_model(
                    output_path,
                    config,
                    pipeline,
                    pipeline.precision_config.encoder_precision,
                    layer_idx,
                    args.np_compat,
                )
            else:
                quantized_model_utils.finalize_encoder_quantized_model(
                    output_path,
                    config,
                    pipeline,
                    pipeline.precision_config.encoder_precision,
                    layer_idx,
                    args.np_compat,
                )
            del calib_data_gen
            logger.info(
                f'PTQ-ed {"projector" if projector else "encoder"} layer{"" if projector else layer_idx} to '
                f'{os.path.join(output_folder, fname)}'
            )

    def _llm_layer_ptq(
        layer,
        args,
        pipeline,
        layer_idx,
        output_folder,
        exp_name,
        config,
        lora_map=None,
        tail=False,
        rotate_handler=None,
    ):
        if not pipeline.config.rotate:
            lora_inputs = None if tail else pipeline.lora_handler.llm_lora_inputs[layer_idx]
        else:
            lora_inputs = rotate_handler.apply_llm(layer, layer_idx, tail)

        example_inputs = layer.get_jit_trace_inputs()

        # JIT trace decoder layer into a temporary pt file
        logger.info(f'JIT tracing LLM layer {layer_idx}')
        temp_trace_filename = utils.jit_trace(layer, example_inputs, output_folder)

        with utils.temp_file(temp_trace_filename):
            # Quantize JIT traced layer
            logger.info(f'PTQ-ing LLM layer {layer_idx}')
            input_shapes, input_value_ranges, calib_data_gen_ = layer.get_ptq_inputs(
                args,
                exp_name=exp_name,
                lora_inputs=lora_inputs,
                calib_lora_map=lora_map,
                has_encoder=pipeline.has_encoder(),
            )[:3]  # TODO Support evaluation dataset

            if pipeline.config.rotate:
                if layer_idx == 0 and input_value_ranges[0] is not None:
                    logger.debug('Update emb_minmax for the first layer with rotated embedding min/max')
                    input_value_ranges[0] = (-config._rot_emb_absmax, config._rot_emb_absmax)  # noqa: SLF001

                logger.debug('Rotate inputs_embeds and past_values from original calibration data')
                r_mat1 = rotate_handler.r_mat['r1'].numpy()
                r_mat2 = None if tail else rotate_handler.r_mat[f'{layer_idx}_r2'].numpy()

                def calib_data_gen():
                    for data_list in calib_data_gen_():
                        inputs_embeds = data_list[0]
                        data_list[0] = np.matmul(inputs_embeds, r_mat1).astype(inputs_embeds.dtype)
                        if r_mat2 is not None:
                            num_lora_inputs = 0 if lora_inputs is None else len(lora_inputs[0])
                            past_values_idx = -1 - num_lora_inputs
                            if pipeline.config.l.infini_attention:
                                past_values_idx -= 2
                            if pipeline.config.l.use_split_mask:
                                past_values_idx -= 1
                            past_values = data_list[past_values_idx]
                            data_list[past_values_idx] = np.matmul(past_values, r_mat2).astype(past_values.dtype)
                        yield data_list
            else:
                calib_data_gen = calib_data_gen_

            converter = mtk_converter.PyTorchConverter.from_script_module_file(
                temp_trace_filename, input_shapes=input_shapes, experimental_debug_tensor_names=True
            )

            # mtk_converter flags
            converter.quantize = True
            converter.prepend_input_quantize_ops = False
            converter.prepend_input_quantize_ops_indices = []
            converter.append_output_dequantize_ops = False
            converter.append_output_dequantize_ops_indices = []

            if (
                pipeline.precision_config.embeds_precision == 'FP' and layer_idx == 0
            ) or pipeline.precision_config.respath_precision == 'FP':
                logger.debug('Prepend input quantize op for index 0')
                converter.prepend_input_quantize_ops = True
                converter.prepend_input_quantize_ops_indices.append(0)
            if pipeline.precision_config.mask_precision == 'FP' and not tail:
                logger.debug('Prepend input quantize op for index 1')
                converter.prepend_input_quantize_ops = True
                converter.prepend_input_quantize_ops_indices.append(1)
            if (
                pipeline.precision_config.logits_precision == 'FP' and tail
            ) or pipeline.precision_config.respath_precision == 'FP':
                logger.debug('Append output dequantize op for index 0')
                converter.append_output_dequantize_ops = True
                converter.append_output_dequantize_ops_indices.append(0)
            if len(pipeline.precision_config.bypassed_ops) > 0:
                converter.extra_quantization_bypass_ops = pipeline.precision_config.bypassed_ops

            if pipeline.precision_config.is_llm_mixed_precision() or args.aopt is not None:
                # Per-FC mixed precision
                if pipeline.precision_config.mode == 'converter_config' and layer_idx < config.num_hidden_layers:
                    quant_config_file = pipeline.precision_config.llm_precision[layer_idx]['precision_config_path']
                    with open(quant_config_file) as f:
                        quant_config = json.load(f)
                else:
                    # Generate `tmp_mp_config_{layer_idx}.json` for (1) `manual` mode (2) tail layer.
                    quant_config_file = os.path.join(output_folder, f'tmp_mp_config_{layer_idx}.json')
                    quant_config = pipeline.precision_config.generate_converter_precision_config(layer_idx)
                    with open(quant_config_file, 'w') as f:
                        f.write(json.dumps(quant_config, indent=4))
                default_precision = quant_config['precision_hints']['default_precision']
                converter.precision_config_file = quant_config_file

                if pipeline.config.l.model_type == 'gecko2' and not tail:
                    converter.prepend_input_quantize_ops = True
                    # input embeds
                    if layer.first_layer_idx == 0:
                        converter.prepend_input_quantize_ops_indices.append(0)

                    is_ee = pipeline.config.l.early_exit_index is not None
                    if is_ee and layer.first_layer_idx >= pipeline.config.l.early_exit_index:
                        if layer.fire_pe:
                            converter.prepend_input_quantize_ops_indices.append(2)
                            converter.prepend_input_quantize_ops_indices.append(3)
                    else:
                        if layer.fire_pe:
                            converter.prepend_input_quantize_ops_indices.append(3)
                            converter.prepend_input_quantize_ops_indices.append(4)
                        # per layer emb
                        converter.prepend_input_quantize_ops_indices.append(2)
            else:
                # Fixed precision
                default_precision = (
                    pipeline.precision_config.tail_precision
                    if tail
                    else next(iter(pipeline.precision_config.llm_unique_precisions))
                )
                converter.precision_proportion = {default_precision: 1.0}

            converter.use_per_output_channel_quantization = not args.per_tensor_weight_quant
            if args.weight_optimization == 'gradient':
                # Set batch_size to 1 since we use dynamic shape calibration dataset
                converter.use_gradient_opt = True
                converter.gradient_opt_batch_size = 1
            elif args.weight_optimization == 'hessian':
                converter.use_hessian_opt = True

            # calibration-related flags.
            # If calib dataset is provided, pass the calib dataset to converter to decide whether it is needed
            if args.calibration_dataset is not None:
                converter.calibration_data_gen = calib_data_gen
                converter.calibration_method = args.calibration_method
                if (not tail and pipeline.precision_config.is_activation_static_quant()['llm']) or (
                    tail and pipeline.precision_config.is_activation_static_quant()['tail']
                ):
                    converter.input_value_ranges = input_value_ranges
            else:
                converter.allow_missing_quantization_ranges = True

            # Additional user-specified options
            if args.extra_converter_options is not None:
                with open(args.extra_converter_options) as f:
                    options = json.load(f)
                sc.check_converter_options(converter, options)
                for k, v in options.items():
                    setattr(converter, k, v)

            if args.vsq_llm:
                converter.use_vector_scaled_quantization = True
                converter.vector_scaled_quantization_vector_size = args.vsq_vector_size_llm

            # Append layer id to filename
            fname = f'{exp_name}_{layer_idx}.{args.format}'
            if args.debug:
                os.makedirs('ptq_debug', exist_ok=True)
                os.makedirs(os.path.join('ptq_debug', exp_name), exist_ok=True)
                converter.quantization_debug_dir = os.path.join('ptq_debug', exp_name, 'llm', f'chunk_{layer_idx}')

            if args.dump_qparams:
                os.makedirs(os.path.join('quant_params', exp_name, 'llm'), exist_ok=True)
                converter.export_qparams_config_file = os.path.join(
                    'quant_params', exp_name, 'llm', f'chunk_{layer_idx}.json'
                )
                converter.export_qparams_model_file = os.path.join(
                    'quant_params', exp_name, 'llm', f'chunk_{layer_idx}.nrpmodel'
                )

            # Handle cases when the qparam model/config files are provided
            if args.load_qparams:
                qparam_path = os.path.join('quant_params', exp_name, 'llm')
                logger.info(f'Loading models and quantization parameters for layer{layer_idx} from {qparam_path}')
                model_fn = os.path.join(qparam_path, f'chunk_{layer_idx}.nrpmodel')
                config_fn = os.path.join(qparam_path, f'chunk_{layer_idx}.json')
                converter = mtk_converter.Converter.from_model_proto_file(model_fn)
                converter.quantize = True
                converter.bypass_general_transformation_stage = True
                converter.import_qparams_config_file = config_fn

            output_path = os.path.join(output_folder, fname)
            # Convert to tflite/mlir
            if args.format == 'tflite':
                converter.convert_to_tflite(output_file=output_path)
            else:
                converter.convert_to_mlir(output_file=output_path)

            quantized_model_utils.finalize_llm_quantized_model(
                output_path, config, pipeline, default_precision, layer_idx, args.np_compat
            )
            del calib_data_gen
            logger.info(f'PTQ-ed LLM layer {layer_idx} to {os.path.join(output_folder, fname)}')

            if (
                pipeline.precision_config.is_llm_mixed_precision() or args.aopt is not None
            ) and pipeline.precision_config.mode == 'manual':
                logger.debug(f'Remove temp mtk_converter precision config: {quant_config_file}')
                os.remove(quant_config_file)

    logger.debug('Running Converter PTQ')

    lora_cfgs = _get_lora_configs(args)
    pipeline = FloatPipeline(
        args.config,
        lora_cfgs,
        task='ptq',
        debug=args.debug,
        use_single_bmm_attention=args.use_single_bmm_attention,
        encoder_only_ptq=args.encoder_only,
        dummy_weights=args.dummy_weights,
    )

    precision_config = PTQPrecisionConfig(
        pipeline.config,
        args.precision_config,
        aopt=args.aopt,
        lora_handler=pipeline.lora_handler,
        use_single_bmm_attention=args.use_single_bmm_attention,
        use_opt_separate_bmm=args.use_opt_separate_bmm,
    )
    pipeline.set_precision_config(precision_config)

    if args.output_folder is None:
        exp_name = utils.get_exp_name(args.config, args.lora_config)
        exp_name += f'_{precision_config.name}'

        if args.calibration_dataset is not None:
            exp_name_without_lora = utils.get_exp_name(args.config)
            ds_name = os.path.basename(args.calibration_dataset.rstrip('/'))
            if ds_name.startswith(exp_name_without_lora):
                ds_name = ds_name.replace(exp_name_without_lora, '')
            if ds_name not in ['', 'fake']:
                exp_name += f'{ds_name}'
            exp_name += f'_{args.calibration_method}'
        if args.weight_optimization is not None:
            exp_name += f'_{args.weight_optimization}'
        vsq_llm_vs_name = f'vs_{args.vsq_vector_size_llm}'
        if pipeline.has_encoder():
            vsq_encoder_vs_name = f'vs_{args.vsq_vector_size_encoder}'
            if args.vsq_encoder and args.vsq_llm:
                exp_name += f'_vsq_encoder_{vsq_encoder_vs_name}_vsq_llm_{vsq_llm_vs_name}'
            elif args.vsq_encoder:
                exp_name += f'_vsq_encoder_{vsq_encoder_vs_name}'
        else:
            if args.vsq_llm:
                exp_name += f'_vsq_llm_{vsq_llm_vs_name}'
        if args.extra_converter_options is not None:
            exp_name += f'_{os.path.splitext(os.path.basename(args.extra_converter_options))[0]}'
        if args.calibration_dataset == 'fake':
            exp_name = 'fake_' + exp_name
        if pipeline.config.rotate:
            exp_name += f'_rotate_{pipeline.config.rotate_mode}_{pipeline.config.rotate_seed}'
        args.output_folder = os.path.join('quantized_models/dynamic_shape', exp_name)
        if os.path.exists(args.output_folder) and args.partial is None and not args.force_overwrite:
            logger.error(
                f'Output folder {args.output_folder} already exists. '
                'Please manually delete the existing folder  or use --force_overwrite if intending to overwrite.',
                err=FileExistsError,
            )
    else:
        exp_name = os.path.basename(args.output_folder.rstrip('/'))

    os.makedirs(args.output_folder, exist_ok=True)
    if args.aopt == 3:
        args.aopt = 2
        logger.warning('Setting AOPT to 2 as AOPT = 3 is not supported yet.')
    print_converter_args(args, exp_name, precision_config, lora_cfgs)

    encoder_lora_maps, llm_lora_maps = _get_lora_maps(args, pipeline, lora_cfgs)

    # dump fp16 embedding bin for float model
    if not args.dummy_weights and pipeline.config.l.model_type != 'gecko2' and not args.encoder_only:
        utils.dump_embedding_lut_for_cmdline(pipeline, quant=False)

    if pipeline.config.rotate:
        rotate_handler = rotate.RotationHandler(pipeline, args.output_folder)
        pipeline.config.rotate_path = pipeline.config.config['rotate_path'] = rotate_handler.rotate_path
    else:
        rotate_handler = None

    # PTQ Encoder
    if pipeline.has_encoder():
        # resolve which encoder layers need to be PTQ-ed
        if args.partial is None:
            encoder_indices_to_ptq = list(range(pipeline.num_encoder_layers + int(pipeline.has_projector())))
        elif 'e' not in args.partial:
            encoder_indices_to_ptq = []
        else:
            encoder_layer_idx_set = set(range(pipeline.num_encoder_layers + int(pipeline.has_projector())))
            encoder_part = args.partial.split('e')[1].split('l')[0]
            encoder_part = encoder_part.strip()
            if '-' in encoder_part:
                if encoder_part.count('-') > 1:
                    logger.error(f'Invalid encoder layer format: {encoder_part}, expect at most 1 `-`', err=ValueError)
                lower, upper = encoder_part.split('-')
                try:
                    encoder_indices_to_ptq = set(range(int(lower), int(upper) + 1))
                except ValueError:
                    logger.error(
                        f'Invalid encoder layer format: {encoder_part}, expect range formats to be of the form: '
                        '`X-Y`, where X and Y are both integers',
                        err=ValueError,
                    )
            elif ',' in encoder_part:
                if encoder_part.endswith(','):
                    encoder_part = encoder_part[:-1]
                try:
                    encoder_indices_to_ptq = {int(x) for x in encoder_part.split(',')}
                except ValueError:
                    logger.error(
                        f'Invalid encoder layer format: {encoder_part}, expect comma formats to be of the form: '
                        '`X,Y,...`, which should only contain commas and integers',
                        err=ValueError,
                    )
            else:
                try:
                    encoder_indices_to_ptq = {
                        int(encoder_part),
                    }
                except ValueError:
                    logger.error(
                        f'Invalid encoder layer format: {encoder_part}, expect integer formats to be of the form: '
                        '`X`, where X must be an integer',
                        err=ValueError,
                    )
            if not encoder_indices_to_ptq.issubset(encoder_layer_idx_set):
                logger.error(
                    f'{encoder_indices_to_ptq} is not a subset of remaining encoder layer indices: '
                    f'{encoder_layer_idx_set}. Ensure your indices are zero-based (From 0 to num_hidden_layers - 1).',
                    err=ValueError,
                )
            encoder_indices_to_ptq = sorted(encoder_indices_to_ptq)

        encoder_output_folder = os.path.join(args.output_folder, 'encoder')
        os.makedirs(encoder_output_folder, exist_ok=True)
        layer = None
        for i in range(pipeline.num_encoder_layers):
            if i not in encoder_indices_to_ptq:
                continue
            layer = pipeline.init_encoder_layer_for_ptq(i)
            if i == pipeline.num_encoder_layers - 1:
                layer = LastEncoderChunk(layer, pipeline.pre_projector_hook)
            _encoder_layer_ptq(
                layer,
                args,
                pipeline,
                i,
                encoder_output_folder,
                exp_name,
                pipeline.config.e,
                encoder_lora_maps[i],
            )
        if pipeline.has_projector() and pipeline.num_encoder_layers in encoder_indices_to_ptq:
            pipeline.init_projector()
            projector = pipeline.projector
            if pipeline.config.rotate:
                rotate_handler.apply_projector(projector)
            _encoder_layer_ptq(
                projector,
                args,
                pipeline,
                pipeline.num_encoder_layers,
                encoder_output_folder,
                exp_name,
                pipeline.config.e,
                None,
                projector=True,
            )

        if args.encoder_only:
            logger.info('`encoder_only` flag is used, only convert/ ptq encoder tflite.')
            return

    # PTQ LLM
    # resolve which llm layers need to be PTQ-ed
    if args.partial is None:
        llm_indices_to_ptq = list(range(pipeline.num_decoder_layers + 1))
    elif 'l' not in args.partial:
        llm_indices_to_ptq = []
    else:
        llm_layer_idx_set = set(range(pipeline.num_decoder_layers + 1))
        llm_part = args.partial.split('l')[1].split('e')[0]
        llm_part = llm_part.strip()
        if '-' in llm_part:
            if llm_part.count('-') > 1:
                logger.error(f'Invalid llm layer format: {llm_part}, expect at most 1 `-`', err=ValueError)
            if ',' in llm_part:
                lower_range, upper_range = llm_part.split(',', 1)
                lower, upper = lower_range.split('-')
                upper_range = (
                    {int(x) for x in upper_range.split(',')} if upper_range.count(',') > 0 else {int(upper_range)}
                )
            else:
                lower, upper = llm_part.split('-')
                upper_range = None
            try:
                llm_indices_to_ptq = set(range(int(lower), int(upper) + 1))
                if upper_range is not None:
                    llm_indices_to_ptq.update(upper_range)
            except ValueError:
                logger.error(
                    f'Invalid llm layer format: {llm_part}, expect range formats to be of the form: '
                    '`X-Y`, where X and Y are both integers',
                    err=ValueError,
                )
        elif ',' in llm_part:
            if llm_part.endswith(','):
                llm_part = llm_part[:-1]
            try:
                llm_indices_to_ptq = {int(x) for x in llm_part.split(',')}
            except ValueError:
                logger.error(
                    f'Invalid llm layer format: {llm_part}, expect comma formats to be of the form: '
                    '`X,Y,...`, which should only contain commas and integers',
                    err=ValueError,
                )
        else:
            try:
                llm_indices_to_ptq = {
                    int(llm_part),
                }
            except ValueError:
                logger.error(
                    f'Invalid llm layer format: {llm_part}, expect integer formats to be of the form: '
                    '`X`, where X must be an integer',
                    err=ValueError,
                )
        if not llm_indices_to_ptq.issubset(llm_layer_idx_set):
            logger.error(
                f'{llm_indices_to_ptq} is not a subset of remaining llm layer indices: '
                f'{llm_layer_idx_set}. Ensure your indices are zero-based (From 0 to num_hidden_layers - 1).',
                err=ValueError,
            )
        llm_indices_to_ptq = sorted(llm_indices_to_ptq)

    if not args.dummy_weights and len(llm_indices_to_ptq) < pipeline.num_decoder_layers:
        logger.info(f'LLM Indices to PTQ: {llm_indices_to_ptq}')
        logger.info('Updating state dict to keep relevant layers only')
        pipeline.update_state_dict(llm_indices_to_ptq)

    llm_output_folder = os.path.join(args.output_folder, 'llm')
    os.makedirs(llm_output_folder, exist_ok=True)
    for i in range(pipeline.num_decoder_layers):
        if i not in llm_indices_to_ptq:
            continue
        layer = pipeline.init_llm_layer_for_ptq(i)
        _llm_layer_ptq(
            layer,
            args,
            pipeline,
            i,
            llm_output_folder,
            exp_name,
            pipeline.config.l,
            llm_lora_maps[i],
            rotate_handler=rotate_handler,
        )
        if i == 0:
            # dump embedding bin for quantized model
            embeds_precision = pipeline.precision_config.embeds_precision
            embeds_precision_name = pipeline.precision_config.get_precision_name(embeds_precision)[1]
            quantize_emb = embeds_precision_name.startswith('int')
            emb_bitwidth = pipeline.precision_config.get_bitwidth(embeds_precision_name)[1] if quantize_emb else 16

            should_open_file = args.get_emb_minmax_from_cal_data or (
                pipeline.has_projector() and not args.skip_align_projector_decoder
            )
            # get quantized info from quantized model folder
            if should_open_file:
                first_quantized_info_path = os.path.join(llm_output_folder, f'{exp_name}_info_{i}.json')
                with open(first_quantized_info_path, encoding='utf-8') as fp:
                    first_quantized_info = json.load(fp)

            embedding_scale_zp = None
            if args.get_emb_minmax_from_cal_data:
                embedding_scale = first_quantized_info['input_scales'][0]
                embedding_zp = first_quantized_info['input_zero_points'][0]
                embedding_scale_zp = (embedding_scale, embedding_zp)

            if pipeline.has_projector() and not args.skip_align_projector_decoder:
                fname = f'{exp_name}_{pipeline.num_encoder_layers}.{args.format}'
                projector_model_path = os.path.join(encoder_output_folder, fname)
                quantized_model_utils.finalize_projector_quantized_model(
                    projector_model_path,
                    pipeline.config.e,
                    pipeline,
                    pipeline.precision_config.encoder_precision,
                    pipeline.num_encoder_layers,
                    args.np_compat,
                    first_quantized_info,
                )

            if pipeline.config.rotate:
                rotate_handler.dump_embedding_bin(
                    emb_bitwidth, quantize_emb, llm_output_folder, embedding_scale_zp=embedding_scale_zp
                )
            else:
                if pipeline.config.l.model_type == 'gecko2':
                    embedder_output_folder = llm_output_folder.replace('llm', 'embedder')
                    utils.dump_embedding_tflite_for_cmdline(
                        pipeline,
                        quant=quantize_emb,
                        quantized_model_folder=embedder_output_folder,
                        dummy_weights=args.dummy_weights,
                    )
                    if quantize_emb:
                        embedder_output_path = os.path.join(
                            embedder_output_folder, f'embedding_{embeds_precision_name}.tflite'
                        )
                    else:
                        embedder_output_path = os.path.join(embedder_output_folder, 'embedding_fp16.tflite')
                    quantized_model_utils.finalize_embedder_quantized_model(
                        embedder_output_path, pipeline.config.l, pipeline
                    )
                else:
                    utils.dump_embedding_lut_for_cmdline(
                        pipeline,
                        quant=quantize_emb,
                        quantized_model_folder=llm_output_folder,
                        dummy_weights=args.dummy_weights,
                        embedding_scale_zp=embedding_scale_zp,
                    )

            # dump infini attention update module
            if pipeline.config.l.infini_attention:
                infini_update_output_folder = llm_output_folder.replace('llm', 'infini_update')

                quantize_infini_update = pipeline.precision_config.infini_update_precision.startswith('int')

                utils.dump_infini_update_tflite_for_cmdline(
                    pipeline, quant=quantize_infini_update, quantized_model_folder=infini_update_output_folder
                )
                if quantize_infini_update:
                    infini_update_output_path = os.path.join(
                        infini_update_output_folder,
                        f'infini_update_{pipeline.precision_config.infini_update_precision}.tflite',
                    )
                else:
                    infini_update_output_path = os.path.join(infini_update_output_folder, 'infini_update_fp.tflite')
                quantized_model_utils.finalize_infini_update_quantized_model(
                    infini_update_output_path, pipeline.config.l, pipeline
                )

        del layer

    if pipeline.num_decoder_layers in llm_indices_to_ptq:
        layer = pipeline.init_tail()
        if args.pad_lm_head is not None and pipeline.config.l.vocab_size % args.pad_lm_head != 0:
            padded_lm_head, pad_size = utils.pad_lm_head_to_any_n(layer.lm_head, pipeline.config.l, args.pad_lm_head)
            layer.lm_head = padded_lm_head
            # dump to original weight/ config.json
            utils.update_config_file_lm_head_pad(args.config, pad_size)
            # update the config.json in PTQ'ed model as well
            pipeline.config.l.lm_head_pad_size = pipeline.config.config['lm_head_pad_size'] = pad_size

        _llm_layer_ptq(
            layer,
            args,
            pipeline,
            pipeline.num_decoder_layers,
            llm_output_folder,
            exp_name,
            pipeline.config.l,
            tail=True,
            rotate_handler=rotate_handler,
        )
        if pipeline.config.t is not None:
            layer.forward = layer.forward_alt
            if hasattr(layer, 'get_jit_trace_inputs_alt'):
                layer.get_jit_trace_inputs = layer.get_jit_trace_inputs_alt
            if hasattr(layer, 'get_jit_trace_inputs_alt'):
                layer.get_ptq_inputs = layer.get_ptq_inputs_alt
            _llm_layer_ptq(
                layer,
                args,
                pipeline,
                pipeline.num_decoder_layers + 1,
                llm_output_folder,
                exp_name,
                pipeline.config.t,
                tail=True,
            )

    with open(os.path.join(args.output_folder, 'config.json'), 'w') as f:
        f.write(json.dumps(pipeline.config.config, indent=4))

    if pipeline.config.l.overture_dict is not None:
        if rotate_handler is not None:
            overture = overture_utils.get_overture(pipeline.config.l.overture_dict)
            rotate_handler.dump_overture(overture, llm_output_folder)
        else:
            overture_path = overture_utils.get_overture_path(pipeline.config.l.overture_dict)
            shutil.copy(overture_path, os.path.join(llm_output_folder, 'overture.npy'))


def _mlkits_ptq(args):
    """Performs Post-Training Quantization (PTQ) for the MLKits backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    from .utils import mlkits_utils

    regularize_mlkits_args(args)

    exp_name = utils.get_exp_name(args.config)
    lora_cfgs = _get_lora_configs(args)

    print_mlkits_args(args, lora_cfgs)

    pipeline = FloatPipeline(
        args.config,
        lora_cfgs,
        task='ptq',
        backend=args.backend,
        debug=args.debug,
        use_single_bmm_attention=args.use_single_bmm_attention,
        encoder_only_ptq=args.encoder_only,
    )

    with open(args.workspace / 'config.json', 'w') as f:
        f.write(json.dumps(pipeline.config.config, indent=4))

    # Dump fp16 embedding bin for float model.
    utils.dump_embedding_lut_for_cmdline(pipeline, quant=False)

    encoder_lora_maps, llm_lora_maps = _get_lora_maps(args, pipeline, lora_cfgs)
    encoder_lora_map = encoder_lora_maps[0] if pipeline.lora_handler.has_encoder_lora() else None
    llm_lora_map = llm_lora_maps[0] if pipeline.lora_handler.has_llm_lora() else None

    def data_generator_wrapper(data_generator):
        """The wrapper is used to wrap the cythonized data generator."""
        yield from data_generator()

    if args.encoder_only:
        logger.info('The `encoder_only` argument is specified, skip LLM optimization.')
    else:
        llm_optimizer = mlkits_utils.MLKitsLLMOptimizer(
            args, exp_name, pipeline, data_generator_wrapper, lora_map=llm_lora_map
        )
        llm_optimizer.run()

    if pipeline.config.e is not None:
        encoder_optimizer = mlkits_utils.MLKitsEncoderOptimizer(
            args, exp_name, pipeline, data_generator_wrapper, lora_map=encoder_lora_map
        )
        encoder_optimizer.run()


@memory_peak_profile
def main(args=None):
    """Main function to perform Post-Training Quantization (PTQ).

    This function parses the arguments, performs sanity checks, and then calls the appropriate PTQ function
    based on the specified backend (converter or MLKits).

    Args:
        args (Namespace): custom argument passing from another python script.

    Raises:
        NotImplementedError: If act_clip_range is used with converter backend with unsupported version of mtk_converter
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))
    args_sanity_checks(args)

    if args.backend == 'converter':
        #####################################################
        # Set up mtk_converter.
        #####################################################
        if args.act_clip_range is not None and not sc.check_converter_version(8, 7, soft=True):
            logger.error(
                f'Please manually set environment variable: MTKCVTR_EXPERIMENTAL_ACT_CLIP_RANGE={args.act_clip_range}. '
                '`act_clip_range` behaviour is fixed from mtk_converter >= 8.7.0.',
                err=NotImplementedError,
            )
        global mtk_converter
        import mtk_converter

        _converter_ptq(args)
    else:
        #####################################################
        # Set up MLKits.
        #####################################################
        global mlkits
        import mlkits

        mlkits.setup_mlkits(
            {
                'workspace_path': args.workspace.as_posix(),
                'framework': 'pytorch',
                'environment': {'recursion_limit': 10000000},
            }
        )
        _mlkits_ptq(args)


if __name__ == '__main__':
    main()
