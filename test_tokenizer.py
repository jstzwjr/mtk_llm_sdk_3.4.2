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
"""Script to test tokenization of text or detokenization of tokens using provided model configs."""

import argparse
import json
import os
import sys

from sentencepiece import SentencePieceProcessor

from . import __version__
from .models.pipeline import FloatPipeline
from .tokenizers.tokenization_qwen import QWenTokenizer
from .utils import logger, utils
from .utils import sanity_checks as sc
from .utils.preformatter import Preformatter

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_test_llm_tokenizer'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Tests a tokenizer's encoding or decoding on a given set of text or token inputs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        'config', type=str, help='[Required] Model config json file. Must be in same directory as tokenizer files.'
    )
    parser.add_argument(
        'input',
        type=str,
        help='[Required] Input text to encode, or comma- or space-separated token IDs to decode, or jsonl filepath.',
    )
    parser.add_argument(
        '-p',
        '--preformatter',
        type=str,
        default=None,
        help='Preformatter json file path to wrap text with. Defaults to None.',
    )
    parser.add_argument(
        '-eb',
        '--exclude_bos',
        action='store_true',
        help='Flag to disable tokenizer from prepending BOS token at the front of encoding. Ignore when decoding.',
    )
    parser.add_argument(
        '-ie',
        '--include_eos',
        action='store_true',
        help='Flag to force tokenizer to append EOS token at the end of encoding. Ignore when decoding.',
    )
    parser.add_argument(
        '--dump_outputs_path',
        default=None,
        type=str,
        help='Path to file to dump out the output of encode and/or decode.',
    )
    parser.add_argument('--debug', action='store_true', help='Flag to turn on debug mode.')
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the provided arguments.

    Args:
        args (Namespace): The parsed arguments.

    Raises:
        FileNotFoundError: If the config or preformatter file does not exist.
        ValueError: If the config or preformatter file has an incorrect extension.
    """
    sc.check_exist(args.config, 'Config file')
    sc.check_ext(args.config, '.json', 'Config file')
    with open(args.config) as f:
        config_json = json.load(f)
    if 'llm' in config_json:
        config_json = config_json['llm']
    sc.check_tokenizer_exist(utils.get_dirpath(args.config))

    if args.preformatter is not None:
        sc.check_exist(args.preformatter, 'Preformatter json file')
        sc.check_ext(args.preformatter, '.json', 'preformatter')


def main(args=None):
    """Main function to handle tokenization and decoding based on provided arguments.

    This function parses the arguments, performs sanity checks, initializes the tokenizer,
    processes the input (either text or tokens), and prints the encoded or decoded results.

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

    pipeline = FloatPipeline(
        args.config,
        None,  # Lora config
        task='test_tokenizer',
    )
    tokenizer = pipeline.tokenizer
    tokenizer.add_bos_token = not args.exclude_bos
    tokenizer.add_eos_token = args.include_eos
    decode = False

    preformatter = Preformatter(args.preformatter)

    logger.info('TOKENIZER INFO:')
    logger.info(f'Tokenizer folder: {utils.get_dirpath(args.config)}')
    if isinstance(tokenizer, QWenTokenizer):
        logger.info(f'BOS id: {tokenizer.im_start_id}')
        logger.info(f'EOS id: {tokenizer.im_end_id}')
        logger.info('Special tokens:', tokenizer.special_tokens)
    elif isinstance(tokenizer, SentencePieceProcessor):
        pass
    else:
        logger.info(f'BOS token and id: {tokenizer.bos_token} | {tokenizer.bos_token_id}')
        logger.info(f'EOS token and id: {tokenizer.eos_token} | {tokenizer.eos_token_id}')
        logger.info(f'PAD token and id: {tokenizer.pad_token} | {tokenizer.pad_token_id}')
        logger.info(f'UNK token and id: {tokenizer.unk_token} | {tokenizer.unk_token_id}')
    logger.info(f'Include BOS during encode: {not args.exclude_bos}')
    logger.info(f'Include EOS during encode: {args.include_eos}')

    if args.input.endswith('.jsonl'):
        inputs = []
        decode_list = []
        with open(args.input) as f:
            lines = [json.loads(x) for x in f]
        for line in lines:
            if line.get('text', None) is not None:
                inputs.append(preformatter.generate_prompt(line['text']))
                decode_list.append(False)
            elif line.get('tokens', None) is not None:
                inputs.append(utils.tokenized_text_to_array(line['tokens'])[0].tolist())
                decode_list.append(True)
            else:
                logger.error(f'line does not contain either text or tokens key: {line}')
    else:
        try:
            temp = []
            inputs_temp = args.input.strip().split(',') if ',' in args.input else args.input.split(' ')
            for inp in inputs_temp:
                temp.append(int(inp))
            decode_list = [True]
        except ValueError:
            temp = preformatter.generate_prompt(args.input)
            decode_list = [False]
        inputs = [temp]

    output_dump = []
    for input_, decode in zip(inputs, decode_list):
        logger.info(f'Input: {input_}')
        if decode:
            if args.exclude_bos:
                logger.info('--exclude_bos will be ignored as inputs are decoded.')
            if args.include_eos:
                logger.info('--include_eos will be ignored as inputs are decoded.')
            decoded = tokenizer.decode(input_)
            logger.info(f'Decoded ({len(input_)} tokens):\n{decoded}\n')
            input_ = ', '.join(map(str, input_))
            output_dump.append([input_, 'Decoded', decoded])
        else:
            if isinstance(tokenizer, SentencePieceProcessor):
                input_ids = tokenizer.encode(input_.replace('\\n', '\n').replace('\\t', '\t'))
            else:
                input_ids = tokenizer(input_.replace('\\n', '\n').replace('\\t', '\t'))['input_ids']
            logger.info(f'Encoded ({len(input_ids)} tokens):\n{input_ids}\n')
            output_dump.append([input_, 'Encoded', input_ids])

    if args.dump_outputs_path is not None:
        logger.info(f'Dumping encoded/decoded output to {args.dump_outputs_path}')
        if os.path.exists(args.dump_outputs_path):
            logger.warning(f'{args.dump_outputs_path} already exists but it will be overwritten.')
        else:
            os.makedirs(args.dump_outputs_path.rsplit('/', 1)[0], exist_ok=True)

        if args.dump_outputs_path.endswith('jsonl'):
            with open(args.dump_outputs_path, 'w', encoding='utf-8') as f:
                for _, cur_type, out in output_dump:
                    key = 'text' if cur_type == 'Decoded' else 'tokens'
                    dump_result = {key: out}
                    json.dump(dump_result, f, ensure_ascii=False)
                    f.write('\n')
        else:
            with open(args.dump_outputs_path, 'w', encoding='utf-8') as f:
                for inp, cur_type, out in output_dump:
                    f.write(f'Input:\n{inp}\n\n{cur_type} Output:\n{out}\n')
                    f.write(f'{"-" * 100}\n')


if __name__ == '__main__':
    main()
