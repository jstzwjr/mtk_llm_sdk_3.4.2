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
"""Script to calculate supported metric scores for LLM models."""

import argparse
import collections
import json
import os
import random
import sys
from datetime import datetime

import mtk_converter
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.pipeline import FloatPipeline, QuantizedPipeline
from .utils import const, datautils, logger, quantized_model_utils, utils
from .utils import sanity_checks as sc
from .utils.memory_profiler import memory_peak_profile
from .utils.preformatter import Preformatter

os.environ['HF_DATASETS_OFFLINE'] = '1'  # Force Huggingface datasets to be offline
os.environ['HF_EVALUATE_OFFLINE'] = '1'  # Force Huggingface evaluate to be offline
os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_evaluate_llm'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run evaluation for selected metrics for LLM models.', allow_abbrev=False
    )
    parser.add_argument(
        'config',
        type=str,
        help='[Required] Model config file. Must be in same directory as all model weight bins and tokenizer files.',
    )
    parser.add_argument(
        'model_type',
        type=str,
        choices=['float', 'quantized'],
        help='[Required] Either float or quantized. float is to evaluate Pytorch model accuracy, '
        'quantized is to evaluate quantized model accuracy.',
    )
    parser.add_argument(
        'metric',
        type=str,
        choices=const.SUPPORTED_METRICS,
        help='[Required] Metric to evaluate model on.',
    )
    parser.add_argument('-l', '--lora_config', type=str, default=None, help='LoRA adapter config json file.')
    parser.add_argument(
        '--quantized_lora_config',
        type=str,
        default=None,
        help='LoRA adapter config json file for quantized model. '
        'If not provided, `args.lora_config` will be used by quantized model as well.',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default=None,
        help='Input dataset name/jsonl file. Defaults to wikitext (for ppl) if not specified. '
        'In built dataset(s): wikitext, c4. If not using in built dataset, provide path to custom '
        'eval dataset jsonl.',
    )
    parser.add_argument(
        '-q',
        '--quantized_model_folder',
        type=str,
        default=None,
        help='Quantized model folder. Required for model_type=quantized.',
    )
    parser.add_argument(
        '-p',
        '--preformatter',
        type=str,
        default=None,
        help='Preformatter json file path to wrap instructions with for instruction-tuned models. '
        'Defaults to None. Not applicable for ppl metric.',
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
        '--dtype',
        type=str,
        default='float16',
        choices={'float16', 'float32'},
        help='Datatype to run evaluation in for float model. Only used when float model is required.',
    )
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed. Defaults to 0.')
    parser.add_argument('--debug', action='store_true', help='Flag to enable debug mode.')
    parser.add_argument('--save', action='store_true', help='Flag to save evalution result into a json file.')
    parser.add_argument(
        '--file', action=utils.PrintFilepathAndExit, file=__file__, help='Prints out absolute filepath and exit'
    )
    parser.add_argument('--cache_evict_config', type=str, default='', help='Cache Evict config path')
    return parser


def args_sanity_checks(args):
    """Performs sanity checks on the arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        tuple: A tuple containing quantized model information dictionaries and input mode string.

    Raises:
        RuntimeError: If any of the argument checks fail.
        FileNotFoundError: If the quantized model folder is not found.
        NotImplementedError: If custom tail such as Medusa or EAGLE are detected in model.
    """
    sc.check_exist(args.config, 'Config file')
    sc.check_ext(args.config, '.json', 'Config file')
    config = PipelineConfig(args.config, verbose=False)
    if (
        args.lora_config is None
        and args.quantized_lora_config is not None
        and (args.model_type == 'float' or args.metric == 'logits')
    ):
        logger.error(
            '`--lora_config` should be provided if `--quantized_lora_config` is provided when `model_type` is `float` '
            'or `metric` is `logits`.'
        )
    if args.lora_config is not None:
        sc.check_exist(args.lora_config, 'LoRA config file')
        sc.check_ext(args.lora_config, '.json', 'Lora config file')
        sc.check_lora_config(args.lora_config, config)
        with open(args.lora_config) as f:
            lora_config = json.load(f)
        lora_rotate = lora_config.get('rotate', False)
        if args.model_type == 'float' and lora_rotate:
            logger.error('Expect unrotated LoRA for float model.')
        if args.quantized_lora_config is not None:
            sc.check_exist(args.quantized_lora_config, 'Quantized LoRA config file')
            sc.check_ext(args.quantized_lora_config, '.json', 'Quantized Lora config file')
            sc.check_lora_config(args.quantized_lora_config, config)
            with open(args.quantized_lora_config) as f:
                quantized_lora_config = json.load(f)
            # If model_type is quantized, we can expect to get rotate info from quantized_lora_config
            lora_rotate = quantized_lora_config.get('rotate', False)
        if args.model_type == 'quantized' and (config.rotate != lora_rotate):
            logger.error('Expect quantized model and LoRA to be both rotated or unrotated.')

    weight_dir = utils.get_dirpath(args.config)

    input_mode = sc.check_input_jsonl(args.dataset)

    if args.metric == 'bleu':
        if input_mode in const.INBUILT_DATASETS:
            logger.error(
                'Invalid dataset. Please specify path to your own eval dataset jsonl to '
                f'calculate bleu score. {args.dataset} is only used for calculating ppl (perplexity) score.'
            )
        if args.preformatter is not None:
            sc.check_exist(args.preformatter, 'Preformatter json file')
            sc.check_ext(args.preformatter, '.json', 'preformatter')
    elif args.metric == 'ppl':
        if args.preformatter is not None:
            logger.error('preformatter is not supported for ppl metric.')
    elif args.metric == 'logits':
        if input_mode in const.INBUILT_DATASETS:
            logger.error(
                'Invalid dataset. Please specify path to your own eval dataset jsonl to '
                f'calculate logits score. {args.dataset} is only used for calculating ppl (perplexity) score.'
            )
        if args.preformatter is not None:
            sc.check_exist(args.preformatter, 'Preformatter json file')
            sc.check_ext(args.preformatter, '.json', 'preformatter')
        if args.model_type == 'float':
            logger.error('metric=logits is only valid for model_type=quantized.')

    quantized_model_folders = {}
    quantized_model_infos = {}
    if args.model_type == 'float':
        sc.check_weights_exist(weight_dir)
        if args.quantized_model_folder is not None:
            logger.error('`quantized_model_folder` is strictly for model_type=quantized')
    elif args.model_type == 'quantized':
        if args.quantized_model_folder is None:
            logger.error('quantized_model_folder is required for model_type=quantized')

        if args.dtype != 'float16' and args.metric != 'logits':
            logger.error('dtype is strictly for model_type=float or metric=logits')

        if not os.path.exists(args.quantized_model_folder):
            logger.error(f'Quantized model folder not found: {args.quantized_model_folder}', err=FileNotFoundError)

        if os.path.exists(os.path.join(args.quantized_model_folder, 'encoder')) and args.metric != 'logits':
            logger.error(
                'Multimodal models currently can only support mtk_evaluate_llm script with logits.',
                err=NotImplementedError,
            )
        llm_folder = os.path.join(args.quantized_model_folder, 'llm')
        encoder_folder = os.path.join(args.quantized_model_folder, 'encoder')

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

        largest_cache_or_token = 0
        gen_folder = None
        prompt_folder = None
        largest_cache_prompt_folder = None
        gen_models = [x for x in os.listdir(llm_folder) if x.startswith('llm_1t')]
        prompt_models = [x for x in os.listdir(llm_folder) if not x.startswith('llm_1t')]
        if len(gen_models) == 0:
            logger.warning(
                f'Did not find any generative LLM models (1t) in {llm_folder}. '
                'This warning is expected when running ppl and logits metrics and can be safely ignored.'
            )
        if len(prompt_models) == 0:
            logger.error(
                f'Did not find any prompt LLM models (>1t) in {llm_folder}. Running evaluation using only generative '
                'model can take days and is not recommended.'
            )

        if args.metric == 'ppl':
            # only run for prompt models (> 1t)
            for llm_model in prompt_models:
                token_size = int(llm_model.split('t')[0].split('_')[1])
                if token_size > largest_cache_or_token:
                    largest_cache_or_token = token_size
                    prompt_folder = llm_model
            gen_folder = None
        else:
            while True:
                for llm_model in prompt_models:
                    cache_size = int(llm_model.split('t')[1].split('c')[0])
                    if cache_size > largest_cache_or_token:
                        logger.info(f'new largest cache size: {cache_size}')
                        largest_cache_or_token = cache_size
                        prompt_folder = llm_model
                if largest_cache_prompt_folder is None:
                    largest_cache_prompt_folder = prompt_folder

                if len(gen_models) > 0 and args.metric == 'bleu':
                    for llm_model in gen_models:
                        cache_size = int(llm_model.split('llm_1t')[1].split('c')[0])
                        if cache_size == largest_cache_or_token:
                            gen_folder = llm_model
                # bleu or logits
                if gen_folder is not None or args.metric == 'logits':
                    break
                prompt_models.remove(prompt_folder)
                if len(prompt_models) == 0:
                    if args.metric == 'bleu':
                        logger.error(
                            'Cound not find a generative model that has the same cache size as any prompt models.'
                        )
                    gen_folder = None
                    prompt_folder = largest_cache_prompt_folder
                    break

        quantized_model_folders = {
            'encoder': encoder_folder,
            'prompt': os.path.join(llm_folder, prompt_folder),
            'generative': None if gen_folder is None else os.path.join(llm_folder, gen_folder),
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

        prompt_num_token = quantized_model_infos['llm'][0]['t']
        cache_size = quantized_model_infos['llm'][0]['c']

        if prompt_num_token is None or cache_size is None:
            logger.error('Dynamic shape quantized model not supported.')
        if args.metric == 'bleu':
            if prompt_num_token > 1:
                sc.check_exist(quantized_model_folders['generative'], 'Generative quantized model directory')
                sc.check_isdir(quantized_model_folders['generative'], 'Generative quantized model directory')
                sc.check_quantized_models(quantized_model_folders['generative'], quantized_model_folders['prompt'])
                if len(os.listdir(quantized_model_folders['generative'])) != len(
                    os.listdir(quantized_model_folders['prompt'])
                ):
                    logger.error(
                        "Number of model chunks in prompt and generative model folders don't"
                        f' match!\nPrompt={len(os.listdir(quantized_model_folders["prompt"]))} chunks. '
                        f'Generative={len(os.listdir(quantized_model_folders["generative"]))} chunks.'
                    )
            else:
                sc.check_quantized_models(quantized_model_folders['generative'])

        custom_tail = quantized_model_utils.has_separate_tails(quantized_model_folders['prompt'])[1]
        if custom_tail:
            logger.error(
                'Medusa/EAGLE custom tails currently not supported for mtk_evaluate_llm.', err=NotImplementedError
            )

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

    return input_mode, quantized_model_folders, quantized_model_infos, evict_method, evict_config


def print_args(args, quantized_model_folders, quantized_model_infos):
    """Prints the arguments for verification."""
    logger.info('Please check if all arguments are correct:')
    logger.info(f'Config file:                          {args.config}')
    if args.lora_config is not None:
        logger.info(f'Lora config file:                     {args.lora_config}')
    if args.quantized_lora_config is not None:
        logger.info(f'Quantized lora config file:           {args.quantized_lora_config}')
    logger.info(f'Model type:                           {args.model_type}')
    logger.info(f'Metric:                               {args.metric}')
    logger.info(f'Dataset:                              {args.dataset}')
    logger.info(f'Use Single BMM Attention Graph:       {args.use_single_bmm_attention}')
    if args.metric != 'ppl':
        logger.info(f'Preformatter json:                    {args.preformatter}')
    if args.model_type == 'quantized' or args.metric == 'logits':
        logger.info(f'Prompt quantized model folder:        {quantized_model_folders["prompt"]}')
        logger.info(f'Generative quantized model folder:    {quantized_model_folders["generative"]}')
        logger.info(f'Prompt fixed shape input token length:{quantized_model_infos["llm"][0]["t"]}')
        logger.info(f'Fixed shape cache size:               {quantized_model_infos["llm"][0]["c"]}')
        logger.info(f'Number of chunks:                     {len(quantized_model_infos["llm"])}')
    if args.model_type == 'float' or args.metric == 'logits':
        logger.info(f'Float dtype:                          {args.dtype}')
    logger.info(f'BOS mode:                             {args.bos_mode}')
    logger.info(f'Random seed:                          {args.seed}')
    logger.info(f'Dump result:                          {args.save}')
    if args.cache_evict_config:
        logger.info(f'Cache evict config path:      {args.cache_evict_config}')
    logger.debug(f'mtk_llm_sdk version:                 {__version__}')


def _encode_text(tokenizer, text):
    if isinstance(tokenizer, SentencePieceProcessor):
        return np.array([tokenizer.encode(text)], dtype=np.int32)
    return tokenizer(text, return_tensors='np')['input_ids'].astype(np.int32)


def evaluate_bleu(
    pipeline,
    input_mode,
    prompts,
    labels,
    dataset_name,
):
    """Evaluates the BLEU score for the given prompts and labels.

    Args:
        pipeline (Pipeline): The model pipeline.
        input_mode (str): The type of input data for prompts and labels (text or tokens).
        prompts (list): The list of prompts to calculate bleu score for.
        labels (list): The list of labels to calculate bleu score against.
        dataset_name (str): The name of the evaluation dataset.

    Returns:
        dict: A dictionary containing the BLEU score and breakdown for the dataset.
    """
    import evaluate

    scores = []
    json_dump = []
    bleu = evaluate.load(os.path.join(os.path.dirname(__file__), 'metrics/bleu.py'))
    eos_id = (
        [pipeline.config.l.eos_token_id]
        if isinstance(pipeline.config.l.eos_token_id, int)
        else pipeline.config.l.eos_token_id
    )
    logger.info('Getting responses ...')
    for i, (prompt, label) in tqdm(enumerate(zip(prompts, labels)), total=len(prompts)):
        if input_mode == 'tokens':
            label = pipeline.tokenizer.decode(label)

        generation_output = pipeline.forward(
            prompt=prompt,
            multimodal_inputs=None,  # Don't support MLLM
            custom_embeds=None,  # Don't support custom embeds
            max_new_tokens=512,  # Cap response length to prevent infinite loop if EOS is never predicted
            num_token=128,  # Only used for float model
            cache_size=4096,  # Only used for float model
            prompt_name=f'prompt_{i}',
            quiet=True,
        )
        input_length = pipeline.input_length

        if generation_output[0][-1] in eos_id:
            output = pipeline.tokenizer.decode(generation_output[0][input_length:-1])
        else:
            output = pipeline.tokenizer.decode(generation_output[0][input_length:])
        score = bleu.compute(predictions=[output], references=[label])['bleu']
        logger.debug(f'pred:  {output}')
        logger.debug(f'label: {label}')
        logger.debug('Score:', score)
        scores.append(score)
        json_dump.append({'Prompt': prompt, 'Pred': output, 'Label': label, 'Score': score})
    bleu_score = np.nanmean(scores)
    return {dataset_name: {'bleu': bleu_score, 'breakdown': json_dump}}


def evaluate_ppl(
    pipeline,
    input_ids,
    dataset_name,
):
    """Evaluates the perplexity for the given test data.

    Args:
        pipeline (Pipeline): The model pipeline.
        input_ids (object): The input token ids.
        dataset_name (str): The name of the evaluation dataset.

    Returns:
        dict: A dictionary containing the perplexity score for the dataset.
    """
    from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

    # for infini attention quantized pipeline, use the same length as float pipeline
    if isinstance(pipeline, QuantizedPipeline) and not pipeline.config.l.infini_attention:
        max_seq_len = pipeline.prompt_num_tokens
    elif pipeline.overture is None:
        max_seq_len = pipeline.config.l.max_position_embeddings
    else:
        max_seq_len = pipeline.config.l.max_position_embeddings - pipeline.overture.shape[-2]

    nsamples = input_ids.numel() // max_seq_len
    logger.info(f'nsamples: {nsamples}')

    input_ids = input_ids.cpu().numpy()

    logger.info('Evaluating perplexity')
    nlls = []
    for i in tqdm(range(nsamples)):
        curr_inputs = input_ids[:, i * max_seq_len : (i + 1) * max_seq_len]
        curr_labels = curr_inputs[:, 1:]
        input_embeds = pipeline.get_embeds(curr_inputs)[0]
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=curr_inputs.shape[1] + 1))

        if pipeline.config.l.infini_attention:
            logits = pipeline.generate_llm_with_memory(
                input_embeds,
                curr_inputs,
                num_token=None,
                cache_size=None,
                stopping_criteria=stopping_criteria,
                return_logits=True,
                prompt_name=f'prompt_{i}',
            )
        else:
            logits = pipeline.generate_llm(
                input_embeds,
                curr_inputs,
                num_token=max_seq_len,  # Only used for float model
                cache_size=1,  # Only used for float model
                stopping_criteria=stopping_criteria,
                return_logits=True,
                dynamic_shape=True,
                prompt_name=f'prompt_{i}',
            )
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        if not isinstance(curr_labels, torch.Tensor):
            curr_labels = torch.from_numpy(curr_labels)
        logits = logits[:, :-1, :].to(pipeline.main_device).contiguous()
        curr_labels = curr_labels.to(pipeline.main_device)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), curr_labels.view(-1))
        neg_log_likelihood = loss.float() * max_seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_seq_len)).item()

    return {dataset_name: {'perplexity': ppl}}


def evaluate_logits(
    args,
    pipeline,
    input_mode,
    prompts,
    labels,
    quantized_model_folders,
    dataset_name,
    use_single_bmm_attention=False,
    preformatter=None,
    evict_method=None,
    evict_config=None,
):
    """Evaluates the top-1 to top-32 logits overlap score between floating and quantized model.

    Args:
        args (argparse.Namespace): The parsed arguments.
        pipeline (Pipeline): The float model pipeline.
        input_mode (str): The type of input data for prompts and labels (text or tokens).
        prompts (list): The list of prompts.
        labels (list): The list of labels.
        quantized_model_folders (dict): The dictionary of lists of quantized model folder paths
        dataset_name (str): The name of the evaluation dataset.
        use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        preformatter (object): The preformatter.
        evict_config : Cache eviction config.
        evict_method : Cache eviction approach.

    Returns:
        dict: A dictionary containing the logits scores for the dataset.
    """
    if not isinstance(pipeline, FloatPipeline):
        logger.error(f'Expected FloatPipeline first when running `evaluate_logits`, but got: {type(pipeline)}')
    from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

    logger.debug(f'quantized_model_folders: {quantized_model_folders}')

    prompt_tokens = []
    response_tokens = []
    kwargs_list = []
    multimodal_embeds = []
    multimodal_inputs = []

    overture_size = 0 if pipeline.overture is None else pipeline.overture.shape[-2]

    if pipeline.lora_handler.rotated:
        logger.info('Loading unrotated LoRA for float model')
        pipeline.lora_handler.load_chunked_lora_inputs(pipeline, rotate=False)

    # Generate reponse if not in dataset
    if len(labels) == 0:
        logger.info('Generating float response tokens as labels...')
        if pipeline.has_encoder():  # Do multimodal forward
            additional_return_dict = {'text_tokens': None, 'multimodal_embeds': None, 'kwargs': None}
            for prompt in tqdm(prompts, total=len(prompts)):
                prompt_input = prompt[input_mode]
                output_tokens, additional_return_dict = pipeline.forward(
                    prompt=prompt_input,
                    streaming_prompt_max_len=0,
                    preformatter=preformatter,
                    multimodal_inputs=utils.get_multimodal_inputs_from_jsonl_line(prompt),
                    repetition_penalty=1.0,
                    max_new_tokens=512,
                    num_token=1024,
                    cache_size=8192,
                    additional_return_dict=additional_return_dict,
                )
                logger.debug(f'output_tokens: {output_tokens.shape}')
                logger.debug(f'Label response : {pipeline.get_response(output_tokens, preformatter)}')
                prompt_tokens.append(additional_return_dict['text_tokens'])
                logger.debug(f'prompt_tokens: {prompt_tokens[-1].shape}')
                response_tokens.append(output_tokens[:, additional_return_dict['text_tokens'].shape[-1] :])
                logger.debug(f'response_tokens: {response_tokens[-1].shape}')
                kwargs_list.append(additional_return_dict['kwargs'])
                multimodal_embeds.append(additional_return_dict['multimodal_embeds'])
                multimodal_inputs.append(utils.get_multimodal_inputs_from_jsonl_line(prompt))
        else:  # LLM text only forward
            pipeline.input_mode = 'tokens'
            for prompt in tqdm(prompts, total=len(prompts)):
                input_ids = prompt if input_mode == 'tokens' else _encode_text(pipeline.tokenizer, prompt)
                input_length = input_ids.shape[-1]
                output_tokens = pipeline.forward(
                    prompt=input_ids,
                    multimodal_inputs=None,  # Don't support MLLM
                    custom_embeds=None,  # Don't support custom embeds
                    max_new_tokens=512 if evict_method is None else const.EVICTION_EVALUATION_MAX_NEW_TOKEN,
                    # Cap response length to prevent infinite loop if EOS is never predicted
                    num_token=input_length,
                    cache_size=input_length + 512,
                    dynamic_shape=evict_method is None,
                )
                prompt_tokens.append(input_ids)
                response_tokens.append(output_tokens[:, input_length:])
    else:
        for prompt in prompts:
            input_ids = prompt if input_mode == 'tokens' else _encode_text(pipeline.tokenizer, prompt)
            prompt_tokens.append(input_ids)
        for label in labels:
            output_ids = label if input_mode == 'tokens' else _encode_text(pipeline.tokenizer, label)
            response_tokens.append(output_ids)

    # Get float logits
    float_output_logits = []
    logger.info('Generating float logits...')
    for idx, (prompt, label) in tqdm(enumerate(zip(prompt_tokens, response_tokens)), total=len(prompt_tokens)):
        input_ids = np.concatenate((prompt, label), axis=1)
        if pipeline.has_encoder():
            logger.debug(f'kwargs_list: {kwargs_list[idx]}')
            input_embeds, kwargs = pipeline.get_embeds.forward(input_ids, multimodal_embeds[idx], **kwargs_list[idx])
        else:
            input_embeds, kwargs = pipeline.get_embeds(input_ids)
        input_length = input_embeds.shape[1]
        if pipeline.config.l.infini_attention:
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(MaxLengthCriteria(max_length=input_length + 1))
            float_logits = pipeline.generate_llm_with_memory(
                input_embeds,
                input_ids,
                num_token=None,
                cache_size=None,
                stopping_criteria=stopping_criteria,
                return_logits=True,
                **kwargs,
            ).squeeze()
        else:
            float_logits = pipeline.generate_llm(
                input_embeds,
                input_ids,
                num_token=input_length,
                cache_size=1 if evict_method is None else max(overture_size, 1),
                stopping_criteria=None,
                dynamic_shape=evict_method is None,
                return_logits=True,
                **kwargs,
            ).squeeze()

        gen_logits = float_logits[-label.shape[-1] :, :].cpu()
        output_logits = []
        if not isinstance(gen_logits, torch.Tensor):
            gen_logits = torch.from_numpy(gen_logits)
        per_token_gen_logits = torch.split(gen_logits, 1)
        for token_logits in per_token_gen_logits:
            topk_gen_values, topk_gen_indices = torch.topk(token_logits.squeeze(), 32)
            topk_token_logits = []
            for v, i in zip(topk_gen_values.tolist(), topk_gen_indices.tolist()):
                topk_token_logits.append([i, v])
            assert len(topk_token_logits) == 32, f'topk_token_logits len is {len(topk_token_logits)}'
            output_logits.append(topk_token_logits)
        float_output_logits.append(output_logits)

    # Clean up
    utils.cleanup_pipeline(pipeline)
    del float_logits

    # Generate tflite logits
    # read config from quantized model folder
    config = PipelineConfig(os.path.join(args.quantized_model_folder, 'config.json'))
    pipeline = QuantizedPipeline(
        config,
        args.quantized_lora_config if args.quantized_lora_config is not None else args.lora_config,
        task='evaluate',
        quantized_model_folders=quantized_model_folders,
        input_mode=input_mode,
        add_bos=args.bos_mode == 'see',
        debug=args.debug,
        use_single_bmm_attention=use_single_bmm_attention,
        evict_method=evict_method,
        evict_config=evict_config,
    )

    quantized_output_logits = []
    logger.info('Generating quantized model logits...')
    for index, (prompt, label) in tqdm(enumerate(zip(prompt_tokens, response_tokens)), total=len(prompt_tokens)):
        input_ids = np.concatenate((prompt, label), axis=1)
        response_length = label.shape[1]
        input_ids = utils.enforce_add_bos_mode(
            pipeline.get_tokenizer_add_bos(), input_ids, pipeline.config.l.bos_token_id
        )
        if pipeline.has_encoder():
            kwargs_list[index]['pipeline_type'] = 'quantized'
            multimodal_embeds_, _, _ = pipeline.get_encoder_output_embedding(
                prompt=None, preformatter=None, multimodal_inputs=multimodal_inputs[index]
            )
            input_embeds = pipeline.get_embeds.forward(input_ids, multimodal_embeds_, **kwargs_list[index])[0]
            input_length = input_embeds.shape[0]
        else:
            input_length = input_ids.shape[-1]
            input_embeds, kwargs = pipeline.get_embeds(input_ids)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=input_length + 1))
        if pipeline.config.l.infini_attention:
            quantized_logits = pipeline.generate_llm_with_memory(
                input_embeds,
                input_ids,
                num_token=None,  # Unused
                cache_size=None,  # Unused
                stopping_criteria=stopping_criteria,
                return_logits=True,
                prompt_name=f'prompt_{index}',
                **kwargs,
            ).squeeze()
        else:
            quantized_logits = pipeline.generate_llm(
                input_embeds,
                input_ids,
                num_token=None,  # Unused
                cache_size=None,  # Unused
                stopping_criteria=stopping_criteria,
                return_logits=True,
                prompt_name=f'prompt_{index}',
                **kwargs,
            ).squeeze()
        quantized_logits = torch.from_numpy(quantized_logits)

        gen_logits = quantized_logits[-response_length:, :].cpu()
        output_logits = []
        per_token_gen_logits = torch.split(gen_logits, 1)
        for token_logits in per_token_gen_logits:
            topk_gen_values, topk_gen_indices = torch.topk(token_logits.squeeze(), 32)
            topk_token_logits = []
            for v, i in zip(topk_gen_values.tolist(), topk_gen_indices.tolist()):
                topk_token_logits.append([i, v])
            assert len(topk_token_logits) == 32
            output_logits.append(topk_token_logits)
        quantized_output_logits.append(output_logits)

    # Clean up
    utils.cleanup_pipeline(pipeline)
    del quantized_logits

    scores = {}
    for k in range(1, 33):
        scores[f'top-{k}'] = topk_overlap(float_output_logits, quantized_output_logits, k)

    return {dataset_name: scores}


def topk_overlap(fp_results, test_results, k):
    """Calculates the top-k overlap between float and test results.

    Args:
        fp_results (list): The float results.
        test_results (list): The test results.
        k (int): The top-k value.

    Returns:
        float: The top-k overlap percentage.
    """
    assert len(fp_results) == len(test_results)

    total_output_logits_num = 0
    total_output_logits_overlap = 0

    for fp_item, test_item in zip(fp_results, test_results):
        num_token = len(fp_item)
        total_output_logits_num += num_token * k

        output_overlap = 0
        for token_idx in range(num_token):
            fp_output_topk_logits = [item[0] for item in fp_item[token_idx][:k]]
            test_output_topk_logits = [item[0] for item in test_item[token_idx][:k]]
            output_overlap += len(list(set(fp_output_topk_logits) & set(test_output_topk_logits)))
        total_output_logits_overlap += output_overlap

    return total_output_logits_overlap / total_output_logits_num * 100


def make_table(result_dict):
    """Generates a table of results.

    Args:
        result_dict (dict): The dictionary containing the results.

    Returns:
        str: The generated table in Markdown format.
    """
    from pytablewriter import MarkdownTableWriter

    md_writer = MarkdownTableWriter()
    md_writer.headers = ['Task', 'Version', 'Metric', 'Value', '', 'Stderr']

    values = []

    for k, dic in result_dict['results'].items():
        version = result_dict['versions'][k]
        for m, v in dic.items():
            if m.endswith('_stderr'):
                continue
            if m == 'breakdown':
                continue

            if m + '_stderr' in dic:
                se = dic[m + '_stderr']
                values.append([k, version, m, '{:.4f}'.format(v), '±', '{:.4f}'.format(se)])
            else:
                values.append([k, version, m, '{:.4f}'.format(v), '', ''])
            k = ''
            version = ''
    md_writer.value_matrix = values

    return md_writer.dumps()


def load_dataset(ds_path, input_mode, preformatter, is_multimodal=False):
    """Loads the dataset from the specified path and applies the preformatter.

    Args:
        ds_path (str): The path to the dataset.
        input_mode (str): type of data in the dataset, whether text or tokens
        preformatter (object): The preformatter object.
        is_multimodal (bool): Whether the processing input is multimodal.

    Returns:
        tuple: A tuple containing the prompts, labels, and a boolean indicating if the dataset is tokenized.

    Raises:
        RuntimeError: If the dataset contains mixed 'text' and 'token' fields.
    """
    with open(ds_path) as f:
        lines = [json.loads(x) for x in f]
    prompts = []
    labels = []
    for line in lines:
        if input_mode == 'text':
            if is_multimodal:
                prompts.append(line)
            else:
                prompts.append(preformatter.generate_prompt(line['text']))
        else:
            assert input_mode == 'tokens'
            prompt_tokens = utils.tokenized_text_to_array(line['tokens'])
            prompts.append(prompt_tokens)
        if line.get('label', None) is not None:
            if input_mode == 'text':
                labels.append(line['label'])
            else:
                label_tokens = utils.tokenized_text_to_array(line['label'])
                labels.append(label_tokens)

    if len(labels) > 0:
        assert len(labels) == len(prompts)

    return prompts, labels


@memory_peak_profile
def main(args=None):
    """Main function to perform evaluation of metrics.

    This function parses the arguments, performs sanity checks, loads the dataset, and evaluates the model outputs
    based on the specified metric (BLEU, PPL, or logits overlap). It saves the results to a file if specified.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))

    # Set defaults
    if args.metric == 'perplexity':
        args.metric = 'ppl'
    if args.metric == 'ppl' and args.dataset is None:
        args.dataset = 'wikitext'
    input_mode, quantized_model_folders, quantized_model_infos, evict_method, evict_config = args_sanity_checks(args)
    print_args(args, quantized_model_folders, quantized_model_infos)

    if args.save:
        os.makedirs('responses', exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    float_dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    preformatter = Preformatter(args.preformatter)

    if args.model_type == 'float' or args.metric == 'logits':
        pipeline = FloatPipeline(
            args.config,
            args.lora_config,
            task='evaluate',
            input_mode=input_mode,
            add_bos=args.bos_mode == 'see',
            dtype=float_dtype,
            debug=args.debug,
            use_single_bmm_attention=args.use_single_bmm_attention,
            evict_method=evict_method,
            evict_config=evict_config,
        )
    else:
        pipeline = QuantizedPipeline(
            args.config,
            args.quantized_lora_config if args.quantized_lora_config is not None else args.lora_config,
            task='evaluate',
            quantized_model_folders=quantized_model_folders,
            input_mode=input_mode,
            add_bos=args.bos_mode == 'see',
            debug=args.debug,
            use_single_bmm_attention=args.use_single_bmm_attention,
            evict_method=evict_method,
            evict_config=evict_config,
        )

    if args.metric == 'ppl':
        if args.dataset == 'wikitext':
            dataset_name = 'wikitext2'
            dataset = datautils.get_dataset(
                dataset_name, pipeline.tokenizer, 'test', max_len=pipeline.config.l.max_position_embeddings
            )
        elif args.dataset == 'c4':
            dataset_name = 'c4'
            dataset = datautils.get_dataset(
                dataset_name, pipeline.tokenizer, 'validation', max_len=pipeline.config.l.max_position_embeddings
            )
        else:
            logger.error('Currently only c4 and wikitext are supported for ppl metric!', err=NotImplementedError)
        labels = []
    else:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        prompts, labels = load_dataset(args.dataset, input_mode, preformatter, is_multimodal=pipeline.has_encoder())
        if len(labels) == 0 and args.metric == 'bleu':
            logger.error('labels are required to calculate bleu score!')

    results = collections.defaultdict(dict)
    if args.metric == 'ppl':
        curr_results = evaluate_ppl(
            pipeline,
            dataset,
            dataset_name,
        )
    elif args.metric == 'bleu':
        curr_results = evaluate_bleu(
            pipeline,
            input_mode,
            prompts,
            labels,
            dataset_name,
        )
    elif args.metric == 'logits':
        curr_results = evaluate_logits(
            args,
            pipeline,
            input_mode,
            prompts,
            labels,
            quantized_model_folders,
            dataset_name,
            args.use_single_bmm_attention,
            preformatter=preformatter,
            evict_method=evict_method,
            evict_config=evict_config,
        )

    results['results'] = curr_results
    results['versions'] = {dataset_name: 1, 'mtk_converter': mtk_converter.__version__}
    results['versions'].update({'mtk_llm_sdk': __version__})
    # add info about the model and few shot config
    results['config'] = {'model_type': args.model_type}

    if args.save:
        if args.model_type == 'float':
            exp_name = utils.get_exp_name(args.config, args.lora_config)
        else:
            exp_name = os.path.basename(args.quantized_model_folder.rstrip('/'))

        dumped = json.dumps(results, indent=2)

        output_path = os.path.join(
            'responses', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.metric}_{exp_name}.json'
        )
        with open(output_path, 'w') as f:
            f.write(dumped)

        logger.info(f'Result dumped to {output_path}')

    logger.info('\n' + make_table(results))


if __name__ == '__main__':
    main()
