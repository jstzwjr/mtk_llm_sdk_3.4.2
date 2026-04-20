# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""Script to prepare required files for HPT."""

import argparse
import json
import os
import pathlib
import pickle
import random
import sys

import numpy as np
import torch

from .models.pipeline import FloatPipeline
from .utils import logger, preformatter, utils
from .utils.memory_profiler import memory_peak_profile


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare required files for HPT.')
    parser.add_argument(
        '--config',
        type=str,
        help='[Required] Model config json file. '
        'Model config must be in same directory as all model weight bins and tokenizer files.',
    )
    parser.add_argument(
        '--workspace', type=pathlib.Path, help='Specifies the workspace directory for the output files.'
    )
    parser.add_argument(
        '-d',
        '--calibration-dataset',
        type=str,
        default=None,
        help='Calibration dataset folder. Use `fake` for fake calibration data (for latency measurement only).',
    )
    parser.add_argument(
        '--evaluation-prompt',
        type=str,
        default=None,
        help='Evaluation prompt for HPT proxy evaluation. '
        'Should be different from calibration dataset and evaluation dataset.',
    )
    parser.add_argument(
        '-l',
        '--lora_config',
        type=str,
        default=None,
        help='LoRA adapter config json file. Only need to provide if not using calibration dataset.',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices={'float16', 'float32'},
        help='Datatype to run evaluation.',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument(
        '-i8c',
        '--int8_cache',
        action='store_true',
        help='Flag to force cache inputs/outputs to be int8 data type.',
    )
    parser.add_argument(
        '--data-length',
        type=int,
        default=-1,
        help='Specifies the length of calibration data. -1 means use all calibration data.',
    )
    parser.add_argument(
        '-p',
        '--preformatter',
        type=str,
        default=None,
        help='Preformatter json file path to wrap instructions with for instruction-tuned models. Defaults to None.',
    )
    parser.add_argument(
        '-b',
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
    parser.add_argument(
        '--disable-fp16-softmax',
        action='store_true',
        help='Whether to disable FP16 softmax.',
    )
    parser.add_argument(
        '-t',
        '--token-size',
        type=int,
        default=None,
        help='Token size for the input data. If not set, the maximum valid length of evaluation dataset will be used.',
    )
    parser.add_argument(
        '--opt-config',
        type=pathlib.Path,
        required=True,
        help='Specifies the path to a YAML file that contains optimization configurations.',
    )
    parser.add_argument(
        '--oreo',
        type=str,
        default=None,
        choices=[None, 'aggressive', 'auto'],
        help='Specifies the Oreo configuration.',
    )
    parser.add_argument(
        '--aopt',
        type=int,
        default=None,
        choices=[None, 0, 1, 2, 3],
        help='Optimization levels of activation precision.',
    )
    return parser


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
        llm_lora_maps (list): The list of LoRA maps for decoder layers.
    """
    num_llm_chunks = len(pipeline.llm_layers_per_chunk)
    llm_lora_maps = [None for _ in range(num_llm_chunks)]
    if pipeline.lora_handler.has_llm_lora() and args.calibration_dataset != 'fake':
        for i in range(num_llm_chunks):
            llm_lora_maps[i] = []
            if args.calibration_dataset is not None:
                lora_mapping_file = os.path.join(args.calibration_dataset, 'llm', f'chunk_{i}', 'lora_mapper.txt')
                with open(lora_mapping_file) as f:
                    for lora_cfg in [x for x in f.readlines() if x != '']:
                        lora_cfg_idx = lora_cfgs.index(lora_cfg.rstrip('\n'))
                        llm_lora_maps[i].append(lora_cfg_idx)
    return llm_lora_maps


def _prepare_eval(args):
    """Prepares the evaluation environment for the MLKits backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    from .utils import mlkits_evaluation_utils, mlkits_utils

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    float_dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    pipeline = FloatPipeline(
        args.config,
        args.lora_config,
        task='evaluate',
        input_mode='text',
        add_bos=args.bos_mode == 'see',
        dtype=float_dtype,
        use_single_bmm_attention=args.use_single_bmm_attention,
    )

    # Prepare data generator.
    data_generator = mlkits_evaluation_utils.get_data_generator(
        pipeline=pipeline,
        llm_config=pipeline.config.l,
        dataset=args.evaluation_prompt,
        token_size=args.token_size,
        preformatter=preformatter.Preformatter(args.preformatter),
    )
    logger.info('Data preparation done.')

    if float_dtype == torch.float16:
        # Re-build pipeline with float32 for MLKits frontend.
        pipeline = FloatPipeline(
            args.config,
            args.lora_config,
            task='evaluate',
            input_mode='text',
            add_bos=args.bos_mode == 'see',
            dtype=torch.float32,
            use_single_bmm_attention=args.use_single_bmm_attention,
        )

    # For evaluation, we only expect one lora config.
    llm_lora_map = [[0]] if pipeline.lora_handler.has_llm_lora() else None

    pipeline.load_checkpoints(pipeline.config.l.weight_dir)
    mlkits.setup_mlkits({'framework': 'pytorch', 'workspace_name': 'tmp'})
    optimizer = mlkits_utils.MLKitsLLMOptimizer(
        args, exp_name='eval_pytorch_logits', pipeline=pipeline, lora_map=llm_lora_map
    )

    source_model = mlkits_evaluation_utils.prepare_model_for_mlkits(optimizer.model, data_generator)
    logger.info('Model preparation done.')

    golden = list(mlkits_evaluation_utils.collect_logits(source_model, data_generator, 32))

    eval_data = {'data': list(data_generator()), 'golden': golden}
    with open(args.workspace / 'evaluation_dataset.pickle', 'wb') as f:
        pickle.dump(eval_data, f)


def _prepare_ptq(args):
    """Performs Post-Training Quantization (PTQ) for the MLKits backend.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    from .utils import mlkits_utils

    exp_name = utils.get_exp_name(args.config)
    lora_cfgs = _get_lora_configs(args)

    pipeline = FloatPipeline(
        args.config,
        lora_cfgs,
        task='ptq',
        backend='mlkits',
        debug=args.debug,
        use_single_bmm_attention=args.use_single_bmm_attention,
        encoder_only_ptq=args.encoder_only,
    )

    with open(args.workspace / 'config.json', 'w') as f:
        f.write(json.dumps(pipeline.config.config, indent=4))

    # Dump fp16 embedding bin for float model.
    utils.dump_embedding_lut_for_cmdline(pipeline, quant=False)

    llm_lora_maps = _get_lora_maps(args, pipeline, lora_cfgs)
    llm_lora_map = llm_lora_maps[0] if pipeline.lora_handler.has_llm_lora() else None

    def data_generator_wrapper(data_generator):
        """The wrapper is used to wrap the cythonized data generator."""
        yield from data_generator()

    llm_optimizer = mlkits_utils.MLKitsLLMOptimizer(
        args, exp_name, pipeline, data_generator_wrapper, lora_map=llm_lora_map
    )
    llm_optimizer.prepare_hpt()


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
    from .utils import mlkits_utils

    if args is None:
        parser = get_argument_parser()
        args = mlkits_utils.make_compatible_argument_parser(parser).parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    #####################################################
    # Set up MLKits.
    #####################################################
    global mlkits
    import mlkits

    mlkits.setup_mlkits(
        {
            'workspace_path': args.workspace.as_posix(),
            'framework': 'pytorch',
            'environment': {'recursion_limit': 100000},
        }
    )
    _prepare_ptq(args)
    _prepare_eval(args)


if __name__ == '__main__':
    main()
