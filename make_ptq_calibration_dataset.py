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
"""Script to generate calibration dataset for LLM Post-Training Quantization."""

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import FloatPipeline
from .utils import const, datautils, logger, overture_utils, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.preformatter import Preformatter

os.environ['HF_DATASETS_OFFLINE'] = '1'  # Force Huggingface datasets to be offline
os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_make_llm_ptq_calib_dataset'


def get_argument_parser():
    """Argument parser for this script."""

    def _create_common_parser(parser):
        parser.add_argument(
            'config',
            type=str,
            help='[Required] Model config json file. '
            'Model config must be in same directory as all model weight bins and tokenizer files.',
        )
        parser.add_argument(
            'inputs',
            type=str,
            default=None,
            help='[Required] Jsonl file path of prompts (text or tokens) to make calibration dataset '
            'with, or "wikitext" or "c4" to use in-built train sets to generate calibration dataset.',
        )
        parser.add_argument('-l', '--lora_config', type=str, default=None, help='LoRA adapter config json file.')
        parser.add_argument(
            '-b',
            '--max_batches',
            type=int,
            default=-1,
            help='Maximum number of calibration data samples to save. Defaults to -1 (save all).',
        )
        parser.add_argument(
            '-p',
            '--preformatter',
            type=str,
            default=None,
            help='Preformatter json file path to wrap prompts with for instruction-tuned models. Defaults to None.',
        )
        parser.add_argument(
            '-m',
            '--max_output_tokens',
            type=int,
            default=128,
            help='Maximum number of response tokens to generate for each prompt. Defaults to 128.',
        )
        parser.add_argument(
            '-o',
            '--output_folder',
            type=str,
            default=None,
            help='User-specified output folder to save calibration dataset to. '
            'Will automatically generate calibration dataset to ./calibration_datasets folder using name of weights '
            'folder and name of lora folder if any by default.',
        )
        parser.add_argument(
            '--streaming_prompt_max_len',
            type=int,
            default=0,
            help='Maximum number of prompt tokens to forward at once. Defaults to 0 (forward all).',
        )
        parser.add_argument(
            '--bos_mode',
            type=str,
            default='skip',
            choices=['save', 'skip'],
            help='How BOS token should be handled. Defaults to "skip".',
        )
        parser.add_argument(
            '--use_single_bmm_attention',
            action='store_true',
            help='Whether to use single bmm attention graph. ',
        )
        parser.add_argument(
            '--force_overwrite',
            action='store_true',
            help='Force overwrite of calibration_dataset if it already exists. Use with caution.',
        )
        parser.add_argument(
            '--debug',
            action='store_true',
        )
        return parser

    def _create_converter_parser(subparser):
        parser = subparser.add_parser(
            'converter',
            help='Generate PTQ calibration dataset for supported LLM models for mtk_converter backend PTQ.',
        )

        return _create_common_parser(parser)

    def _create_mlkits_parser(subparser):
        parser = subparser.add_parser(
            'mlkits', help='Generate PTQ calibration dataset for supported LLM models for mlkits backend PTQ.'
        )
        parser = _create_common_parser(parser)
        parser.add_argument(
            '-t',
            '--token_size',
            type=int,
            default=1024,
            help='Token size of each calibration data sample. Defaults to 1024. '
            '-1 means use max_position_embeddings value inside config.json.',
        )

        return parser

    parser = argparse.ArgumentParser(
        description='Generates the calibration dataset for supported models for the chosen PTQ backend.',
        allow_abbrev=False,
    )
    subparser = parser.add_subparsers(dest='backend')
    _create_converter_parser(subparser)
    _create_mlkits_parser(subparser)
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )

    return parser


def args_sanity_checks(args):
    """Sanity checks for this script."""

    def _common_checks(args):
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

        if input_mode == 'embeddings' and config.e:
            logger.error(
                'Using embeddings as calibration input is experimental and only for LLM.', err=NotImplementedError
            )
        if input_mode != 'text':
            if args.preformatter is not None:
                logger.error('preformatter only supported for text jsonl input.')
            if args.streaming_prompt_max_len > 0:
                logger.error('Streaming prompt mode is only implemented for text jsonl input.', err=NotImplementedError)
        else:
            sc.check_tokenizer_exist(config.l.weight_dir)
            if args.preformatter is not None:
                sc.check_exist(args.preformatter, 'Preformatter json file')
                sc.check_ext(args.preformatter, '.json', 'preformatter')
            if args.streaming_prompt_max_len != 0:
                sc.check_positive_int(args.streaming_prompt_max_len, 'streaming_prompt_max_len')

        sc.check_positive_int(args.max_output_tokens, 'max_output_tokens')

        if args.output_folder is not None and os.path.exists(args.output_folder) and not args.force_overwrite:
            logger.error(
                f'Output folder {args.output_folder} already exists. '
                'Please manually delete the existing folder  or use --force_overwrite if intending to overwrite.',
                err=FileExistsError,
            )

        # check overture option
        if config.l.overture_dict is not None:
            overture_path = overture_utils.get_overture_path(config.l.overture_dict)
            sc.check_exist(overture_path, 'Overture path')

        # check bita option
        if config.l.bita:
            sc.check_exist(config.l.bita_config, 'BiTA Config')
            try:
                with open(config.l.bita_config) as f:
                    bita_config = json.load(f)

                # Check required keys
                required_keys = ['weight_path', 'prefix_key_name', 'prefix_length', 'bita_inference_draft_length']
                for key in required_keys:
                    if key not in bita_config:
                        logger.error(f'Missing required key in BiTA config: {key}', err=ValueError)

                # Check if weight file exists
                sc.check_exist(bita_config['weight_path'], 'BiTA weight file')
            except json.JSONDecodeError:
                logger.error(f'Invalid JSON format in BiTA config file: {config.l.bita_config}', err=ValueError)

        if config.l.infini_attention and args.streaming_prompt_max_len > 0:
            logger.error('Streaming prompt mode is not implemented for infini attention.', err=NotImplementedError)

        return config, input_mode

    def _converter_checks(args, config):
        if args.max_batches > 0 and args.max_batches % 2 != 0:
            logger.error('max_batches must be an even number!', err=ValueError)

        if config.e and args.inputs in const.INBUILT_DATASETS:
            logger.error('Cannot use in-built datasets for LMM models.')

    def _mlkits_checks(args, config):
        if config.t and config.t.model_type != 'medusa':
            logger.error('Only medusa custom tail is currently supported for mlkits backend', err=NotImplementedError)
        if args.token_size > config.l.max_position_embeddings:
            logger.error("`token_size` cannot be more than config's `max_position_embeddings`.", err=ValueError)

    config, input_mode = _common_checks(args)
    if args.backend == 'converter':
        _converter_checks(args, config)
    else:
        _mlkits_checks(args, config)

    return config, input_mode


def print_args(args):
    """Printed arguments for this script."""

    def _converter_only(args):
        pass

    def _mlkits_only(args):
        logger.info(f'Token size:                                   {args.token_size}')

    logger.info('Please check if all arguments are correct:')
    logger.info(f'Config file:                                  {args.config}')
    if args.lora_config is not None:
        logger.info(f'Lora config file:                             {args.lora_config}')
    logger.info(f'Output dataset folder name:                   {args.output_folder}')
    logger.info(f'Prompt inputs file:                           {args.inputs}')
    if args.max_batches > 0:
        logger.info(f'Maximum calibration data batches to generate: {args.max_batches}')
    logger.info(f'Preformatter json:                            {args.preformatter}')
    logger.info(f'Maximum output tokens:                        {args.max_output_tokens}')
    logger.info(f'BOS mode:                                     {args.bos_mode}')
    if args.streaming_prompt_max_len > 0:
        logger.info(f'Max streaming prompt length:                  {args.streaming_prompt_max_len}')
    logger.info(f'Use Single BMM Attention Graph:               {args.use_single_bmm_attention}')
    logger.info(f'Force overwrite existing calibration dataset: {args.force_overwrite}')
    if args.backend == 'converter':
        _converter_only(args)
    else:
        _mlkits_only(args)
    logger.info(f'mtk_llm_sdk version:                          {__version__}')


def get_lora_mapping_files(lora_config, pipeline, output_dir, mlkits=False):
    """Generates LoRA mapping files for each decoder layer.

    Args:
        lora_config (str): The path to the LoRA configuration file.
        pipeline (Pipeline): The Pipeline object.
        output_dir (str): The output directory.
        mlkits (bool): Whether to use MLKits (default is False).

    Returns:
        list: A list of file objects for the LoRA mapping files.
    """
    lora_mapping_files = {
        'encoder': [],
        'llm': [],
    }
    if lora_config is not None:
        for i in range(pipeline.num_encoder_layers):
            lora_mapping_file = open(os.path.join(output_dir, 'encoder', f'chunk_{i}', 'lora_mapper.txt'), 'w')  # noqa: SIM115
            lora_mapping_files['encoder'].append(lora_mapping_file)
            if mlkits:
                # mlkits only 1 chunk
                break
        if pipeline.has_projector() and not mlkits:
            lora_mapping_file = open(  # noqa: SIM115
                os.path.join(output_dir, 'encoder', f'chunk_{pipeline.num_encoder_layers}', 'lora_mapper.txt'), 'w'
            )
            lora_mapping_files['encoder'].append(lora_mapping_file)

        for i in range(pipeline.num_decoder_layers):
            lora_mapping_file = open(os.path.join(output_dir, 'llm', f'chunk_{i}', 'lora_mapper.txt'), 'w')  # noqa: SIM115
            lora_mapping_files['llm'].append(lora_mapping_file)
            if mlkits:
                # mlkits only 1 chunk
                break
    return lora_mapping_files


def sort_by_token(output_folder):
    """Sort the data by the valid token size.

    The original un-sorted folder will be renamed with a `.source` postfix, and the final sorted
    folder will use symbolic links to the original one.

    Args:
        output_folder (str): the output folder containing the data batches to be sorted.
    """
    output_folder = pathlib.Path(output_folder)
    root_folder = output_folder.parent.parent

    # Rename `<root>/llm` to `<root>/llm.source`.
    source_folder = output_folder.parent
    source_folder = source_folder.rename(source_folder.with_suffix('.source'))

    # Create a new `<root>/llm/chunk_0`.
    output_folder.mkdir(parents=True)

    # Handle lora mapper file.
    if (source_folder / 'chunk_0' / 'lora_mapper.txt').is_file():
        shutil.copy(source_folder / 'chunk_0' / 'lora_mapper.txt', output_folder / 'lora_mapper.txt')

    size_from_source = {}
    for batch in source_folder.glob('chunk_0/*.npz'):
        token_size = np.load(batch)['input_tokens'].size
        size_from_source[batch] = token_size

    sorted_batches = sorted(size_from_source.items(), key=lambda item: item[1], reverse=True)
    for i, (batch, _) in enumerate(sorted_batches):
        dst_batch = (output_folder / f'batch-{i:04d}.npz').as_posix()
        src_batch = pathlib.Path('../..', batch.relative_to(root_folder)).as_posix()
        os.symlink(src_batch, dst_batch)


@memory_peak_profile
def main(args=None):
    """Main function to generate and save calibration data.

    This function parses the arguments, performs sanity checks, initializes the pipeline, and processes the input data
    to generate and save calibration data for the specified backend (converter or mlkits).

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    config, input_mode = args_sanity_checks(args)
    if args.output_folder is None:
        ds_name = utils.get_exp_name(args.config, args.lora_config)
        if config.t:
            ds_name += f'_{config.t.model_type if config.t.model_type == "medusa" else "eagle"}'
        if input_mode in const.INBUILT_DATASETS:
            ds_name += f'_{input_mode}'
        args.output_folder = f'calibration_datasets/{ds_name}'
        if os.path.exists(args.output_folder) and not args.force_overwrite:
            logger.error(
                f'Output folder {args.output_folder} already exists. '
                'Please manually delete the existing folder or use --force_overwrite if intending to overwrite.',
                err=FileExistsError,
            )
    print_args(args)

    pipeline = FloatPipeline(
        args.config,
        args.lora_config,
        task='make_calibration',
        input_mode=input_mode,
        add_bos=args.bos_mode == 'save',
        backend=args.backend,
        debug=args.debug,
        use_single_bmm_attention=args.use_single_bmm_attention,
    )

    preformatter = Preformatter(args.preformatter)

    overture_size = 0
    if config.l.overture_dict is not None:
        overture = overture_utils.get_overture(config.l.overture_dict)
        overture_size = overture.shape[-2]

    bita_draft_length = 0
    if config.l.bita:
        with open(config.l.bita_config) as f:
            bita_config = json.load(f)
        bita_draft_length = bita_config['bita_inference_draft_length']

    # enforce overwrite
    if args.force_overwrite and os.path.isdir(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    output_dirs = {'llm': []}

    if args.backend == const.MLKITS:
        # MLKits needs single chunk only.
        if pipeline.has_encoder():
            output_dirs['encoder'] = [os.path.join(args.output_folder, 'encoder', 'chunk_0')]
        output_dirs['llm'].append(os.path.join(args.output_folder, 'llm', 'chunk_0'))
    else:
        if pipeline.has_encoder():
            output_dirs['encoder'] = []
            for i in range(pipeline.num_encoder_layers):
                output_dirs['encoder'].append(os.path.join(args.output_folder, 'encoder', f'chunk_{i}'))
            if pipeline.has_projector():
                output_dirs['encoder'].append(
                    os.path.join(args.output_folder, 'encoder', f'chunk_{pipeline.num_encoder_layers}')
                )
        for i in range(pipeline.num_decoder_layers + 1):
            output_dirs['llm'].append(os.path.join(args.output_folder, 'llm', f'chunk_{i}'))

    flattened_output_dirs = [dir_ for dirs_ in output_dirs.values() for dir_ in dirs_]
    for output_dir in flattened_output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    lora_mapping_files = get_lora_mapping_files(
        args.lora_config, pipeline, args.output_folder, mlkits=args.backend == const.MLKITS
    )

    if args.backend == 'converter':
        max_len = config.l.max_position_embeddings - overture_size - bita_draft_length
    else:
        max_len = config.l.max_position_embeddings if args.token_size < 0 else args.token_size
    logger.debug(f'max_len={max_len}')

    if input_mode in const.INBUILT_DATASETS:
        lines = datautils.get_dataset(input_mode, pipeline.tokenizer, 'train', max_len, np.int32)
    else:
        with open(args.inputs) as f:
            lines = [json.loads(x) for x in f]

    logger.info('Saving calibration data ...')
    curr_batches = 0
    if args.max_batches > 0:
        max_batches = (
            args.max_batches
            if args.backend == const.MLKITS or args.streaming_prompt_max_len > 0 or config.l.infini_attention
            else min(args.max_batches, 2 * len(lines))
        )
    else:
        max_batches = 99999999 if args.streaming_prompt_max_len > 0 or args.backend == const.MLKITS else 2 * len(lines)
    logger.debug(f'max_batches={max_batches}')

    for i, line in enumerate(lines):
        logger.debug(f'Curr line={line}')
        prompt = line[input_mode]
        label = None

        cache_size = 4096
        if config.l.infini_attention:
            cache_size = config.l.infini_segment_size
        elif config.l.extra_input['sink_rope']:
            cache_size = None

        if input_mode in const.INBUILT_DATASETS or input_mode == 'embeddings':
            combined_input = prompt
        else:
            if line.get('label', None) is None:
                pipeline.task = 'inference'  # Set to inference to generate response without saving calibration data
                logger.debug('No label provided. Set pipeline task to inference to generate label.')
                llm_output = pipeline.forward(
                    prompt=prompt,
                    input_mode=input_mode,
                    streaming_prompt_max_len=args.streaming_prompt_max_len,
                    preformatter=preformatter,
                    multimodal_inputs=utils.get_multimodal_inputs_from_jsonl_line(line),
                    max_new_tokens=args.max_output_tokens,
                    num_token=min(1024, config.l.max_position_embeddings)
                    if not config.l.extra_input['sink_rope']
                    else None,
                    cache_size=cache_size,
                    quiet=True,
                )
                eos_id = [config.l.eos_token_id] if isinstance(config.l.eos_token_id, int) else config.l.eos_token_id
                if llm_output[0][-1] in eos_id:
                    llm_output = llm_output[:, :-1]  # Remove final EOS ID as that is not an expected calibration input
                    logger.debug('Remove EOS token from response')

                if input_mode == 'text':
                    response = pipeline.get_response(llm_output, preformatter, quiet=True)
                    logger.debug(f'Response after get_response: {response}')
                    if config.e is not None:
                        combined_input = prompt
                        label = response
                    else:
                        combined_input = preformatter.generate_prompt(prompt, input_=None, label=response)
                elif input_mode == 'tokens':
                    combined_input = llm_output
                else:
                    combined_input = np.concatenate(
                        (prompt, pipeline.get_embeds(tokens=llm_output[pipeline.input_length :])),
                        axis=1,
                    )
            else:
                logger.debug('Label provided. Directly using it.')
                if input_mode == 'text':
                    if config.e is not None:
                        combined_input = prompt
                        label = line['label']
                    else:
                        combined_input = preformatter.generate_prompt(prompt, input_=None, label=line['label'])
                else:
                    # Works for both tokens and embeddings
                    combined_input = np.concatenate((prompt, pipeline.format_prompt(line['label'])), axis=1)
        logger.debug(f'combined_input={combined_input}')

        pipeline.task = 'make_calibration'  # Set to make_calibration to save calibration data npz
        logger.debug('Set pipeline task to make_calibration')
        make_calib_kwargs = {'label': label}
        llm_output = pipeline.forward(
            prompt=combined_input,
            input_mode=input_mode,
            streaming_prompt_max_len=args.streaming_prompt_max_len,
            preformatter=preformatter,
            multimodal_inputs=utils.get_multimodal_inputs_from_jsonl_line(line),
            max_new_tokens=1,
            num_token=max_len if args.backend == const.MLKITS else None,
            output_dirs=output_dirs,
            max_batches=max_batches,
            lora_mapping_files=lora_mapping_files,
            **make_calib_kwargs,
        )

        pipeline.get_response(llm_output, preformatter, quiet=True)

        curr_batches = len([x for x in os.listdir(output_dirs['llm'][0]) if x.endswith('.npz')])
        if not args.debug:
            logger.info(
                f'Current line: {i + 1} | Current batches = '
                f'{curr_batches}{"/" if max_batches != 99999999 else ""}'
                f'{max_batches if max_batches != 99999999 else ""}'
                '\r'
            )
        logger.debug(f'Current line: {i + 1}, curr_batches={curr_batches}, max_batches={max_batches}')

        if curr_batches >= max_batches:
            break

    print()  # To ensure the ram/vram info is printed on a new line

    for x in lora_mapping_files:
        for f in lora_mapping_files[x]:
            f.close()

    if args.backend == const.MLKITS:
        for key, output_chunk_dirs in output_dirs.items():
            if 'llm' in key:
                sort_by_token(output_chunk_dirs[0])


if __name__ == '__main__':
    main()
