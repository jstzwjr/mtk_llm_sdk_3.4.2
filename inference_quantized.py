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
"""Script to run prompt and response inference for quantized LLM models."""

import argparse
import json
import os
import sys

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import QuantizedPipeline
from .utils import const, logger, quantized_model_utils, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.preformatter import Preformatter

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_inference_quantized_llm'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run MTK quantized models inference.', allow_abbrev=False)
    parser.add_argument(
        'quantized_model_folder',
        type=str,
        help='[Required] Fixed shape quantized model folder. '
        'Will automatically detect for encoder models, LLM prompt models, and LLM generative models.',
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
        help='Maximum number of prompt tokens to forward at once. Defaults to 0 (forward all).',
    )
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for random sample search.')
    parser.add_argument('--top_p', type=float, default=None, help='Top-P for random sample search.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-K for random sample search.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic algorithms to execute quantized models (will run significantly slower).',
    )
    parser.add_argument(
        '--use_single_bmm_attention',
        action='store_true',
        help='Whether to use single bmm attention graph. ',
    )
    parser.add_argument(
        '--save_jsonl_path',
        type=str,
        default=None,
        help='Flag to save the float inference output to certain path. Defaults to None',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
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
    sc.check_exist(args.quantized_model_folder, 'quantized model directory')
    sc.check_isdir(args.quantized_model_folder, 'quantized model directory')

    encoder_folder = os.path.join(args.quantized_model_folder, 'encoder')
    llm_folder = os.path.join(args.quantized_model_folder, 'llm')

    if os.path.exists(encoder_folder):
        if len(os.listdir(encoder_folder)) == 1:
            encoder_folder = os.path.join(encoder_folder, os.listdir(encoder_folder)[0])
        else:
            largest_batch = 0
            for enc_model in os.listdir(encoder_folder):
                batch_size = int(enc_model.split('encoder_')[1].split('b')[0])
                if batch_size > largest_batch:
                    largest_batch = batch_size
                    enc_folder = enc_model
            encoder_folder = os.path.join(encoder_folder, enc_folder)
    else:
        encoder_folder = None

    largest_cache = 0
    largest_token = 0
    gen_folder = None
    largest_cache_gen_folder = None
    prompt_folder = None
    gen_models = [x for x in os.listdir(llm_folder) if x.startswith('llm_1t')]
    prompt_models = [x for x in os.listdir(llm_folder) if not x.startswith('llm_1t')]
    if len(gen_models) == 0:
        logger.error(f'Did not find any generative LLM models (1t) in {llm_folder}')
    if len(prompt_models) == 0:
        logger.warning(
            f'Did not find any prompt LLM models (>1t) in {llm_folder}. '
            'Will use 1t model for prompt mode which will be extremely slow.'
        )
    while True:
        for llm_model in gen_models:
            cache_size = int(llm_model.split('llm_1t')[1].split('c')[0])
            if cache_size > largest_cache:
                largest_cache = cache_size
                gen_folder = llm_model
        if largest_cache_gen_folder is None:
            largest_cache_gen_folder = gen_folder

        if len(prompt_models) == 0:
            prompt_folder = gen_folder
        else:
            for llm_model in prompt_models:
                cache_size = int(llm_model.split('t')[1].split('c')[0])
                if cache_size == largest_cache:
                    token_size = int(llm_model.split('t')[0].split('llm_')[1])
                    if token_size > largest_token:
                        largest_token = token_size
                        prompt_folder = llm_model
        if prompt_folder is not None:
            break
        gen_models.remove(gen_folder)
        if len(gen_models) == 0:
            logger.warning(
                'Cound not find a prompt model that has the same cache size as any generative models. Will use '
                f'{largest_cache_gen_folder} for both prompt and generative mode, which is expected to be very slow.'
            )
            gen_folder = largest_cache_gen_folder
            prompt_folder = largest_cache_gen_folder
            break

    quantized_model_folders = {
        'encoder': encoder_folder,
        'prompt': os.path.join(llm_folder, prompt_folder),
        'generative': os.path.join(llm_folder, gen_folder),
    }
    quantized_model_infos = {
        'encoder': [],
        'llm': [],
    }

    if encoder_folder is not None:
        encoder_model_paths = [
            os.path.join(quantized_model_folders['encoder'], x)
            for x in os.listdir(quantized_model_folders['encoder'])
            if not x.endswith('.json')
        ]
        for model_path in encoder_model_paths:
            quantized_model_infos['encoder'].append(
                quantized_model_utils.extract_encoder_quantized_model_info(model_path)
            )

    prompt_model_paths = utils.get_sorted_path_list(quantized_model_folders['prompt'], ext=['.tflite', '.mlir'])
    for model_path in prompt_model_paths:
        quantized_model_infos['llm'].append(quantized_model_utils.extract_llm_quantized_model_info(model_path))

    config = PipelineConfig(
        os.path.join(args.quantized_model_folder, 'config.json'),
        verbose=False,
    )
    has_lora = False
    if len(quantized_model_infos['encoder']) > 0:
        has_lora = has_lora or quantized_model_infos['encoder'][0]['num_lora_inputs'] > 0
    has_lora = has_lora or quantized_model_infos['llm'][0]['num_lora_inputs'] > 0

    if args.lora_config is not None:
        if not has_lora:
            logger.error('Quantized model does not have LoRA inputs.')
        sc.check_exist(args.lora_config, 'Lora config file')
        sc.check_ext(args.lora_config, '.json', 'Lora config file')
        sc.check_lora_config(args.lora_config, config)
        with open(args.lora_config) as f:
            lora_config = json.load(f)
        lora_rotate = lora_config.get('rotate', False)
        if config.rotate != lora_rotate:
            logger.error('Expect quantized model and LoRA to be both rotated or unrotated.')
    else:
        if has_lora:
            logger.error('Quantized model has LoRA inputs but lora_config is not provided.')

    if config.e is not None and len(quantized_model_infos['encoder']) == 0:
        logger.error('Model config has encoder but no encoder quantized models found.')

    sc.check_supported_tokenizer(config.l)

    input_mode = sc.check_input_jsonl(args.inputs)

    prompt_num_token = quantized_model_infos['llm'][0]['t']
    cache_size = quantized_model_infos['llm'][0]['c']
    if prompt_num_token is None or cache_size is None:
        logger.error('Dynamic shape models not supported. Please run shape fixer (mtk_fix_llm_shape) first.')

    if input_mode == 'wikitext':
        logger.error('wikitext not applicable for inference!', err=ValueError)

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
    sc.check_positive_int(args.max_output_tokens, 'max_output_tokens')

    custom_tail = quantized_model_utils.has_separate_tails(quantized_model_folders['prompt'])[1]
    if custom_tail is not None:
        logger.error(
            'Medusa/EAGLE custom tails currently not supported for PC inference. '
            'Please use device cmdline for verification.',
            err=NotImplementedError,
        )

    if prompt_num_token > 1:
        sc.check_exist(quantized_model_folders['generative'], 'Generative quantized model directory')
        sc.check_isdir(quantized_model_folders['generative'], 'Generative quantized model directory')
        sc.check_quantized_models(quantized_model_folders['generative'], quantized_model_folders['prompt'])
        if len(os.listdir(quantized_model_folders['generative'])) != len(os.listdir(quantized_model_folders['prompt'])):
            logger.error(
                "Number of model chunks in prompt and generative model folders don't"
                f' match!\nPrompt={len(os.listdir(quantized_model_folders["prompt"]))} chunks. '
                f'Generative={len(os.listdir(quantized_model_folders["generative"]))} chunks.'
            )
    else:
        sc.check_quantized_models(quantized_model_folders['generative'])

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
        evict_config['cache_size'] = cache_size
    else:
        evict_method = None
        evict_config = {}

    return config, input_mode, quantized_model_folders, quantized_model_infos, evict_method, evict_config


def print_args(args, quantized_model_folders, quantized_model_infos):
    """Printed arguments for this script."""
    logger.info('Please check if all arguments are correct:')
    if quantized_model_folders['encoder'] is not None:
        logger.info(f'Encoder quantized model folder:       {quantized_model_folders["encoder"]}')
    logger.info(f'Prompt quantized model folder:        {quantized_model_folders["prompt"]}')
    logger.info(f'Generative quantized model folder:    {quantized_model_folders["generative"]}')
    logger.info(f'Prompt inputs file:                   {args.inputs}')
    if args.lora_config is not None:
        logger.info(f'Lora config file:                     {args.lora_config}')
    logger.info(f'Preformatter json:                    {args.preformatter}')
    logger.info(f'Repetition penalty:                   {args.repetition_penalty}')
    logger.info(f'Prompt fixed shape input token length:{quantized_model_infos["llm"][0]["t"]}')
    logger.info(f'Fixed shape cache size:               {quantized_model_infos["llm"][0]["c"]}')
    logger.info(f'Number of chunks:                     {len(quantized_model_infos["llm"])}')
    logger.info(f'Separate final FC:                    {quantized_model_infos["llm"][-1]["tail"] in const.TAIL_TYPES}')
    logger.info(f'Maximum output tokens:                {args.max_output_tokens}')
    logger.info(f'BOS mode:                             {args.bos_mode}')
    if args.streaming_prompt_max_len > 0:
        logger.info(f'Max streaming prompt length:          {args.streaming_prompt_max_len}')
    if args.temperature is not None:
        logger.info(f'Temperature:                          {args.temperature}')
    if args.top_p is not None:
        logger.info(f'Top-P:                                {args.top_p}')
    if args.top_k is not None:
        logger.info(f'Top-K:                                {args.top_k}')
    logger.info(f'Use Single BMM Attention Graph:       {args.use_single_bmm_attention}')
    logger.info(f'Save inference output to jsonl:       {args.save_jsonl_path}')
    logger.info(f'Save inference prompt with preformatter:{args.save_preformatter}')
    if args.cache_evict_config:
        logger.info(f'Cache evict config path:      {args.cache_evict_config}')
    logger.info(f'mtk_converter version:                {mtk_converter.__version__}')
    logger.info(f'mtk_llm_sdk version:                  {__version__}')


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

    if args.deterministic:
        os.environ['MTKCVTR_DISABLE_DELEGATOR'] = '1'
        os.environ['MTKCVTR_USE_DETERMINISTIC_ALGORITHMS'] = '1'
    global mtk_converter
    import mtk_converter

    config, input_mode, quantized_model_folders, quantized_model_infos, evict_method, evict_config = args_sanity_checks(
        args
    )
    print_args(args, quantized_model_folders, quantized_model_infos)

    pipeline = QuantizedPipeline(
        config,
        args.lora_config,
        task='inference',
        quantized_model_folders=quantized_model_folders,
        input_mode=input_mode,
        add_bos=args.bos_mode == 'see',
        debug=args.debug,
        use_single_bmm_attention=args.use_single_bmm_attention,
        evict_method=evict_method,
        evict_config=evict_config,
    )

    preformatter = Preformatter(args.preformatter)

    with open(args.inputs) as f:
        lines = [json.loads(x) for x in f]

    if args.save_jsonl_path is not None:
        output_file = os.path.expanduser(args.save_jsonl_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    qid_ = -1
    for i, line in enumerate(lines):
        prompt_name = line.get('name', f'prompt_{i}')

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
            num_token=None,  # Unused
            cache_size=None,  # Unused
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            prompt_name=prompt_name,
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
