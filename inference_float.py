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
"""Script to run prompt and response inference for float LLM models."""

import argparse
import json
import os
import sys

import torch

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import FloatPipeline
from .utils import const, logger, overture_utils, rotate, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.preformatter import Preformatter

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_inference_float_llm'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run PyTorch floating point model inference using mtk_llm_sdk model files.', allow_abbrev=False
    )
    parser.add_argument(
        'config',
        type=str,
        help='[Required] Model config file. Must be in same directory as all model weights and tokenizer files.',
    )
    parser.add_argument('inputs', type=str, help='[Required] Jsonl file of prompts to run inference for.')
    parser.add_argument(
        '-l', '--lora_config', type=str, default=None, help='LoRA adapter config json file. Defaults to None.'
    )
    parser.add_argument(
        '-p',
        '--preformatter',
        type=str,
        default=None,
        help='Preformatter json file path to wrap instructions with for instruction-tuned models. Defaults to None.',
    )
    parser.add_argument(
        '-r',
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='Repetition penalty factor to apply to logits. Defaults to 1.0 (No penalty).',
    )
    parser.add_argument(
        '-t',
        '--num_token',
        type=int,
        default=128,
        help='Fixed shape prompt input token length to simulate. Defaults to 128.',
    )
    parser.add_argument(
        '-c', '--cache_size', type=int, default=1024, help='Fixed shape cache size to simulate. Defaults to 1024.'
    )
    parser.add_argument(
        '-m',
        '--max_output_tokens',
        type=int,
        default=128,
        help='Maximum number of response tokens to generate. Defaults to 128.',
    )
    parser.add_argument(
        '-b',
        '--bos_mode',
        type=str,
        default='see',
        choices=['see', 'skip'],
        help='How BOS token should be handled. Defaults to `see`.',
    )
    parser.add_argument(
        '--streaming_prompt_max_len',
        type=int,
        default=0,
        help='Maximum number of prompt tokens to forward as one prompt. '
        'Chunks prompt into multiple chunks if prompts exceed this length. '
        'Defaults to 0 (forward whole prompt as one prompt).',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32'],
        help='Datatype to run inference in. If response is rubbish, try setting to float32. Defaults to `float16`',
    )
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for random sample search.')
    parser.add_argument('--top_p', type=float, default=None, help='Top-P for random sample search.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-K for random sample search.')
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument('--rotate', action='store_true', help='Flag to rotate floating point model.')
    parser.add_argument(
        '--save_jsonl_path',
        type=str,
        default=None,
        help='Flag to save the float inference output to certain path. Defaults to None',
    )
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    parser.add_argument('--cache_evict_config', type=str, default='', help='Cache Evict config path')
    parser.add_argument('--save_preformatter', action='store_true', help='Whether to dump prompt with preformatter.')
    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        input_mode: The type of input provided in the args.inputs argument

    Raises:
        RuntimeError: If any of the argument checks fail.
        ValueError: If the inputs argument is invalid.
    """
    sc.check_exist(args.config, 'Config file')
    sc.check_ext(args.config, '.json', 'Config file')
    config = PipelineConfig(args.config, verbose=False)
    if args.lora_config is not None:
        sc.check_exist(args.lora_config, 'Lora config file')
        sc.check_ext(args.lora_config, '.json', 'Lora config file')
        sc.check_lora_config(args.lora_config, config)

    sc.check_weights_exist(config.l.weight_dir)
    if config.e and config.e.weight_dir:
        sc.check_weights_exist(config.e.weight_dir)
    if config.t and config.t.weight_dir:
        sc.check_weights_exist(config.t.weight_dir)

    sc.check_supported_tokenizer(config.l)

    input_mode = sc.check_input_jsonl(args.inputs)
    if input_mode in const.INBUILT_DATASETS:
        logger.error('Inbuilt datasets not applicable for inference!', err=ValueError)

    if input_mode != 'text':
        if args.preformatter is not None:
            logger.error('preformatter only supported for text input.')
        if args.streaming_prompt_max_len > 0:
            logger.error('Streaming prompt mode is only implemented for text input.', err=NotImplementedError)
    else:
        sc.check_tokenizer_exist(config.l.weight_dir)
        if args.preformatter is not None:
            sc.check_exist(args.preformatter, 'Preformatter json file')
            sc.check_ext(args.preformatter, '.json', 'preformatter')
        if args.streaming_prompt_max_len != 0:
            sc.check_positive_int(args.streaming_prompt_max_len, 'streaming_prompt_max_len')

    sc.check_between_inclusive(args.repetition_penalty, 1.0, 10.0, 'repetition_penalty')

    sc.check_positive_int(args.num_token, 'num_token')
    sc.check_positive_int(args.cache_size, 'cache_size')
    sc.check_positive_int(args.max_output_tokens, 'max_output_tokens')

    sc.check_search_args(args)

    if config.l.overture_dict is not None:
        overture_path = overture_utils.get_overture_path(config.l.overture_dict)
        sc.check_exist(overture_path, 'Overture')

    if args.cache_evict_config != '':
        with open(args.cache_evict_config) as file:
            evict_config = json.load(file)
        evict_method = evict_config.pop('cache_evict', None)
        if evict_method not in ['GlobalSnapKV', 'LocalSnapKV']:
            logger.error(
                'Must specify cache evict method by setting `cache_evict` in cache_evict_config. '
                f'choices:[`LocalSnapKV`, `GlobalSnapKV`] but get {evict_method}',
                err=RuntimeError,
            )
        logger.info(f'Applying cache evict method:  {evict_method}')
        evict_config['cache_size'] = args.cache_size
    else:
        evict_method = None
        evict_config = {}

    return input_mode, evict_method, evict_config


def print_args(args):
    """Printed arguments for this script."""
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Config file:                      {args.config}')
    if args.lora_config is not None:
        logger.info(f'Lora config file:                 {args.lora_config}')
    logger.info(f'Prompt inputs file:               {args.inputs}')
    logger.info(f'Preformatter json:                {args.preformatter}')
    logger.info(f'Repetition penalty:               {args.repetition_penalty}')
    logger.info(f'Simulated input token length:     {args.num_token}')
    logger.info(f'Simulated cache size:             {args.cache_size}')
    logger.info(f'Maximum output tokens:            {args.max_output_tokens}')
    logger.info(f'BOS mode:                         {args.bos_mode}')
    logger.info(f'Is Rotated:                       {args.rotate}')
    if args.streaming_prompt_max_len > 0:
        logger.info(f'Max streaming prompt length:      {args.streaming_prompt_max_len}')
    if args.temperature is not None:
        logger.info(f'Temperature:                      {args.temperature}')
    if args.top_p is not None:
        logger.info(f'Top-P:                            {args.top_p}')
    if args.top_k is not None:
        logger.info(f'Top-K:                            {args.top_k}')
    logger.info(f'Data type:                        {args.dtype}')
    logger.info(f'Save inference output to jsonl:   {args.save_jsonl_path}')
    logger.info(f'Save inference prompt with preformatter:{args.save_preformatter}')
    if args.cache_evict_config:
        logger.info(f'Cache evict config path:      {args.cache_evict_config}')
    logger.info(f'mtk_llm_sdk version:             {__version__}')


@memory_peak_profile
def main(args=None):
    """Main entrypoint.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    input_mode, evict_method, evict_config = args_sanity_checks(args)
    print_args(args)

    pipeline = FloatPipeline(
        args.config,
        args.lora_config,
        task='inference',
        input_mode=input_mode,
        add_bos=args.bos_mode == 'see',
        dtype=torch.float16 if args.dtype == 'float16' else torch.float32,
        debug=args.debug,
        evict_method=evict_method,
        evict_config=evict_config,
    )

    if args.rotate:
        rotate.rotate_pipeline(pipeline)

    preformatter = Preformatter(args.preformatter)

    with open(args.inputs) as f:
        lines = [json.loads(x) for x in f]

    if args.save_jsonl_path is not None:
        output_file = os.path.expanduser(args.save_jsonl_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    qid_ = -1
    for line in lines:
        qid_ += 1
        prompt = line[input_mode]
        qid = line.get('question_id', qid_)
        llm_output = pipeline.forward(
            prompt=prompt,
            streaming_prompt_max_len=args.streaming_prompt_max_len,
            preformatter=preformatter,
            multimodal_inputs=utils.get_multimodal_inputs_from_jsonl_line(line),
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_output_tokens,
            num_token=args.num_token,
            cache_size=args.cache_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        response = pipeline.get_response(llm_output, preformatter)

        if args.save_jsonl_path is not None:
            output = {}
            output.update(line)  # Update information of input dict
            if args.preformatter is not None and args.save_preformatter:
                prompt = pipeline.prompt_formatted
            output.update({'question_id': qid, 'text': prompt, 'label': response})
            with open(str(output_file), 'a', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False)
                f.write('\n')

        logger.info('-' * 100)


if __name__ == '__main__':
    main()
