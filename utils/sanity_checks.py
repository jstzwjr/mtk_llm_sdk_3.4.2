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
"""Define all sanity checks for mtk_llm_sdk."""

import json
import os
import re

from . import const, logger, quantized_model_utils


def check_between_inclusive(num, min_, max_, message=None):
    """Checks if a number is between two values inclusively.

    Args:
        num (int or float): The number to check.
        min_ (int or float): The minimum value.
        max_ (int or float): The maximum value.
        message (str, optional): The custom error message.

    Raises:
        TypeError: If the types of num, min_, and max_ are not the same.
        ValueError: If num is not between min_ and max_ inclusively.
    """
    if not (type(num) is type(min_) is type(max_)):
        logger.error(
            f'Got different types for num ({type(num)}), min ({type(min_)}), and max ({type(max_)})', err=TypeError
        )
    if not (min_ <= num <= max_):
        if message is None:
            logger.error(f'Expected number between {min_} and {max_} inclusive, but got: {num}', err=ValueError)
        logger.error(f'{message} must be between {min_} and {max_} inclusive, but got: {num}', err=ValueError)


def check_converter_options(converter, options):
    """Checks if the converter options are valid.

    Args:
        converter (object): The converter object.
        options (dict): The options to check.

    Raises:
        ValueError: If an option is invalid or should not be modified explicitly.
    """
    available_options = converter.get_available_options()
    handled_by_ptq_script = [
        'quantize',
        'prepend_input_quantize_ops',
        'append_output_dequantize_ops',
        'input_value_ranges',
        'calibration_data_gen',
        'precision_proportion',
        'precision_config_file',
        'use_gradient_opt',
        'gradient_opt_batch_size',
        'use_hessian_opt',
        'calibration_method',
    ]
    for k in options:
        if k not in available_options:
            logger.error(
                f'{k} is not a valid mtk_converter.PyTorchConverter attribute. '
                f'List of available attributes: {available_options}',
                err=ValueError,
            )
        if k in handled_by_ptq_script:
            logger.error(
                f'Do not explicitly modify converter attribute: {k}, use the API provided by ptq.py instead.',
                err=ValueError,
            )


def check_converter_version(min_major, min_minor=None, soft=False):
    """Checks if the mtk_converter version is at least a provided version number.

    Raises:
        ImportError: If the mtk_converter version is less than the specified major/minor version number.
    """
    import mtk_converter

    if min_minor is None:
        min_minor = 0

    major, minor = mtk_converter.__version__.split('.')[:2]
    if int(major) < min_major or (int(major) == min_major and int(minor) < min_minor):
        if not soft:
            logger.error(f'mtk_converter version needs to be at least {min_major}.{min_minor}', err=ImportError)
        return False
    return True


def check_dynamic_shape(quantized_model_path):
    """Checks if the quantized model file has a dynamic shape.

    Args:
        quantized_model_path (str): The path to the quantized model file.

    Raises:
        RuntimeError: If the quantized model file does not have a dynamic shape.
    """
    from . import quantized_model_utils

    check_ext(quantized_model_path, ['.tflite', '.mlir'])
    model_info = quantized_model_utils.extract_llm_quantized_model_info(quantized_model_path)

    if model_info['t'] is not None or model_info['c'] is not None:
        logger.error(
            f'Expected a dynamic shape quantized model but {quantized_model_path} is a '
            f'{model_info["t"]}t{model_info["c"]}c quantized model.'
        )


def check_exist(file_or_folder, message=None):
    """Checks if a file or folder exists.

    Args:
        file_or_folder (str): The path to the file or folder.
        message (str, optional): The custom error message.

    Raises:
        FileNotFoundError: If the file or folder does not exist.
    """
    if not os.path.exists(file_or_folder):
        if message is None:
            logger.error(f'{file_or_folder} does not exist.', err=FileNotFoundError)
        logger.error(f'{message} does not exist: {file_or_folder}', err=FileNotFoundError)


def check_ext(file, ext, message=None):
    """Checks if a file has the expected extension.

    Args:
        file (str): The path to the file.
        ext (str or list or tuple): The expected extension(s).
        message (str, optional): The custom error message.

    Raises:
        RuntimeError: If the file does not have the expected extension.
    """
    if isinstance(ext, (list, tuple)):
        for e in ext:
            if file.endswith(e):
                return
        if message is None:
            logger.error(f'Expected one of {ext} extensions, but got: {file}')
        logger.error(f'Expected one of {ext} extensions for {message}, but got: {file}')

    if not file.endswith(ext):
        if message is None:
            logger.error(f'Expected {ext} file, but got: {file}')
        logger.error(f'Expected {ext} file for {message}, but got: {file}')


def check_input_jsonl(jsonl_path):
    """Checks if the input JSONL file is valid.

    Args:
        jsonl_path (str): The path to the JSONL file.

    Returns:
        str: The mode of the JSONL file ('text', 'tokens', or 'embeddings').

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        RuntimeError: If the JSONL file is empty or has invalid content.
        KeyError: If the JSONL file has mutually exclusive keys.
    """
    if jsonl_path in const.INBUILT_DATASETS:
        return jsonl_path

    check_exist(jsonl_path, 'Input prompts jsonl file')
    check_ext(jsonl_path, '.jsonl', 'Input prompts jsonl file')

    mode = None

    with open(jsonl_path) as f:
        lines = [json.loads(x) for x in f]
    if len(lines) == 0:
        logger.error(f'{jsonl_path} is an empty jsonl file.')

    for i, line in enumerate(lines):
        if mode is None:
            if 'text' in line:
                mode = 'text'
                if 'tokens' in line or 'embeddings' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )
            elif 'tokens' in line:
                mode = 'tokens'
                if 'text' in line or 'embeddings' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )
            elif 'embeddings' in line:
                mode = 'embeddings'
                if 'text' in line or 'tokens' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )
            else:
                logger.error('Every line in jsonl needs to have either "text", "tokens", or "input_embedding" key.')
        else:
            if mode == 'text':
                if 'text' not in line:
                    logger.error('Expected all lines in jsonl to contain "text" key.')
                if 'tokens' in line or 'embeddings' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )
            elif mode == 'tokens':
                if 'tokens' not in line:
                    logger.error('Expected all lines in jsonl to contain "tokens" key.')
                if 'text' in line or 'embeddings' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )
            else:
                if 'embeddings' not in line:
                    logger.error('Expected all lines in jsonl to contain both "tokens" and "input_embedding" keys.')
                if 'text' in line or 'tokens' in line:
                    logger.error(
                        f'"embeddings", "text" and "tokens" keys are mutually exclusive but two or more are found in line {i}: {line}',  # noqa: E501
                        err=KeyError,
                    )

        if 'image' in line:
            if isinstance(line['image'], str):
                check_ext(line['image'], ['.jpg', '.png', '.npy', 'jpeg'])
                check_exist(line['image'])
            elif isinstance(line['image'], list):
                for i in range(len(line['image'])):
                    check_ext(line['image'][i], ['.jpg', '.png', '.npy', 'jpeg'])
                    check_exist(line['image'][i])
            else:
                logger.error('"image" have to either a list of images or a str of single image.', err=KeyError)

        if 'embeddings' in line:
            check_ext(line['embeddings'], '.npy')
            check_exist(line['embeddings'])

    return mode


def check_isdir(folder, message=None):
    """Checks if a path is a directory.

    Args:
        folder (str): The path to the folder.
        message (str, optional): The custom error message.

    Raises:
        FileNotFoundError: If the path is not a directory.
        RuntimeError: If the path is not a directory and a custom message is provided.
    """
    if not os.path.isdir(folder):
        if message is None:
            logger.error(f'{folder} is not a directory.', err=FileNotFoundError)
        logger.error(f'Expected directory for {message}, but got: {folder}')


def check_lora_config(file, pipeline_config):
    """Checks if the LoRA configuration is valid.

    Check if adapter config is pre-3.0 or post-3.0.
    Only pure LLM configs are accepted for pre-3.0.
    For MLLM lora, must use post-3.0 config style.
    In both cases, normalize first to post-3.0 style.

    Args:
        file (str): The path to the LoRA configuration file.
        pipeline_config (object): The pipeline configuration object.

    Raises:
        KeyError: If the LoRA target modules are invalid.
        RuntimeError: If the LoRA adapter config target modules are ordered incorrectly.
    """
    with open(file) as f:
        config_ = json.load(f)

    if 'llm' in config_:
        # Post-3.0
        if 'r' not in config_['llm']:
            logger.error('`r` (rank) must be one of the keys of LLM lora config', err=KeyError)
        if 'target_modules' not in config_['llm']:
            logger.error('`target_modules` must be one of the keys of LLM lora config', err=KeyError)

    if 'encoder' in config_:
        # Post-3.0
        if 'r' not in config_['encoder']:
            logger.error('`r` (rank) must be one of the keys of encoder lora config', err=KeyError)
        if 'target_modules' not in config_['encoder']:
            logger.error('`target_modules` must be one of the keys of encoder lora config', err=KeyError)

    if 'llm' not in config_ and 'encoder' not in config_:
        # Pre-3.0
        if 'r' not in config_:
            logger.error('`r` (rank) must be one of the keys of lora config', err=KeyError)
        if 'target_modules' not in config_:
            logger.error('`target_modules` must be one of the keys of lora config', err=KeyError)
        config = {'llm': config_}
    else:
        config = config_

    if 'encoder' in config:
        check_positive_int(config['encoder']['r'])
        check_positive_int(config['encoder'].get('lora_alpha', config['encoder']['r']))

        lora_start_layer_idx = config['encoder'].get('lora_start_layer_idx', 0)
        lora_end_layer_idx = config['encoder'].get('lora_end_layer_idx', pipeline_config.e.num_hidden_layers - 1)

        if lora_end_layer_idx == -1:
            lora_end_layer_idx = pipeline_config.e.num_hidden_layers - 1

        if lora_start_layer_idx > lora_end_layer_idx:
            logger.error(
                '[Encoder] '
                f'`lora_start_layer_idx` ({lora_start_layer_idx}) cannot be more than '
                f'`lora_end_layer_idx` ({lora_end_layer_idx})',
                err=ValueError,
            )

        # TODO: Check encoder target_modules

    if 'llm' in config:
        check_positive_int(config['llm']['r'])
        check_positive_int(config['llm'].get('lora_alpha', config['llm']['r']))

        lora_start_layer_idx = config['llm'].get('lora_start_layer_idx', 0)
        lora_end_layer_idx = config['llm'].get('lora_end_layer_idx', pipeline_config.l.num_hidden_layers - 1)

        if lora_end_layer_idx == -1:
            lora_end_layer_idx = pipeline_config.l.num_hidden_layers - 1

        if lora_start_layer_idx > lora_end_layer_idx:
            logger.error(
                '[LLM] '
                f'`lora_start_layer_idx` ({lora_start_layer_idx}) cannot be more than '
                f'`lora_end_layer_idx` ({lora_end_layer_idx})',
                err=ValueError,
            )

        # Check lora modules
        lora_modules = config['llm']['target_modules']
        accepted_modules = [v for k, v in pipeline_config.l.fc_names['attn'].items() if k != 'name'] + [
            v for k, v in pipeline_config.l.fc_names['mlp'].items() if k != 'name'
        ]

        rejected_modules = []
        for module in lora_modules:
            if module not in accepted_modules:
                rejected_modules.append(module)
        if len(rejected_modules) > 0:
            logger.error(
                f'These LoRA target_modules are rejected: {rejected_modules}. '
                f'Full list of acceptable module names: {accepted_modules}',
                err=KeyError,
            )

        curr_idx = 0
        for module in lora_modules:
            j = accepted_modules.index(module)
            if j < curr_idx:
                logger.error(
                    f'LoRA adapter config target_modules is ordered incorrectly: {lora_modules}.'
                    f'\nPlease arrange the target modules to be in this order: {accepted_modules}'
                )
            curr_idx = j


def check_num_chunks(quantized_model_list, num_chunks_or_layers_per_chunk, separate_tail, encoder_only=False):
    """Checks the number of chunks and their sizes for quantized model files.

    Args:
        quantized_model_list (list): List of quantized model file paths.
        num_chunks_or_layers_per_chunk (list): Number of chunks or layers per chunk.
        separate_tail (bool): Whether to separate the tail layer.
        encoder_only (bool): Whether to shape fix only encoder. Default to False.

    Raises:
        RuntimeError: If the expected size for the largest chunk exceeds the maximum supported size.
    """
    if len(quantized_model_list) == 0:
        if encoder_only:
            logger.warning(
                'No dynamic shape llm found, doing encoder only shape fixing, please ensure this is desired behaviour.'
            )
            return
        logger.error('Must provide dynamic shape llm folder when doing shape fixing.', err=RuntimeError)
    if quantized_model_list[0].endswith('.mlir'):
        return
    while not quantized_model_list[0].endswith('_0.tflite'):
        quantized_model_list.pop(0)
    layer_size = os.path.getsize(quantized_model_list[0])
    tail_size = os.path.getsize(quantized_model_list[-1])
    if len(num_chunks_or_layers_per_chunk) == 1:
        num_decoder_layers = len(quantized_model_list) - 1
        num_layers_per_chunk = [
            (num_decoder_layers // num_chunks_or_layers_per_chunk[0])
            + (i < (num_decoder_layers % num_chunks_or_layers_per_chunk[0]))
            for i in range(num_chunks_or_layers_per_chunk[0])
        ]
        max_layers = max(num_layers_per_chunk)
        max_size = max(tail_size, layer_size * max_layers) if separate_tail else (layer_size * max_layers + tail_size)
    else:
        num_layers_per_chunk = num_chunks_or_layers_per_chunk
        max_layers = max(num_layers_per_chunk)
        if separate_tail:
            max_size = max(tail_size, layer_size * max_layers)
        else:
            max_size = max(tail_size + layer_size * num_layers_per_chunk[-1], layer_size * max_layers)

    if max_size > (2 * 1024**3):
        from . import utils

        major, minor = utils.get_converter_version(include_minor=True)
        if major < 8 or (major == 8 and minor < 6):
            logger.error(
                f'Expected size for largest chunk is {max_size / (1024**3):.2f}GB, which '
                'is larger than the maximum that each TFLite can support which is 2GB. Please increase '
                'the number of chunks or reduce the number of layers for the biggest chunk. This '
                'restriction is lifted when using mtk_converter>=8.6.0'
            )


def check_positive_int(num, message=None):
    """Checks if a number is a positive integer.

    Args:
        num (int): The number to check.
        message (str, optional): The custom error message.

    Raises:
        ValueError: If the number is not a positive integer.
    """
    if not isinstance(num, int):
        logger.error(f'Expected int but got: {type(num)}', err=ValueError)
    if num < 1:
        if message is None:
            logger.error(f'Expected positive integer but got: {num}', err=ValueError)
        logger.error(f'{message} must be a positive integer, but got: {num}', err=ValueError)


def check_ptq_format(format_):
    """Checks if the PTQ format is valid.

    Args:
        format_ (str): The PTQ format.

    Raises:
        ValueError: If the PTQ format is invalid.
        ImportError: If the MLIR export is not supported by the mtk_converter version.
    """
    if format_ not in ['tflite', 'mlir']:
        logger.error('PTQ format must be one of: tflite, mlir.', err=ValueError)
    if format_ == 'mlir' and not check_converter_version(9, soft=True):
        logger.error('mlir export is only supported on NP9 versions of mtk_converter (9.X.X)', err=ImportError)


def check_search_args(args):
    """Checks the arguments for randome sample search method.

    Args:
        args (argparse.Namespace): The arguments to check.

    Raises:
        ValueError: If any argument is invalid.
    """
    if args.temperature is not None and not (0 <= args.temperature <= 1.0):
        logger.error('temperature must be between 0 and 1 inclusive.', err=ValueError)
    if args.top_p is not None and not (0 <= args.top_p <= 1.0):
        logger.error('top_p must be between 0 and 1 inclusive.', err=ValueError)
    if args.top_k is not None and args.top_k < 0:
        logger.error('top_k must be zero or a positive integer.', err=ValueError)


def check_shapes(shapes, encoder_only=False):
    """Checks if the provided shapes are valid.

    Args:
        shapes (list): A list of shape strings.
        encoder_only (bool): A flag indicates whether the shape is for encoder only shape fixing.
            Defaults to False.

    Returns:
        tuple: A tuple containing a boolean indicating if permutation mode is used and a boolean indicating
            if rank is used.

    Raises:
        TypeError: If shapes is not a list.
        RuntimeError: If there are duplicate shapes or if the shapes are in the wrong format.
    """
    if not isinstance(shapes, list):
        logger.error(f'Expected shapes to be a list, but got {type(shapes)} instead', err=TypeError)

    ERROR_MESSAGE = (  # noqa: N806
        'Shape {} is in the wrong format. Please refer to document for instructions on how to construct shapes.'
    )
    COMPONENT_ERROR_MESSAGE = (  # noqa: N806
        'Shape {} is in the wrong format due to offending component: {}. '
        'Please refer to document for instructions on how to construct shapes.'
    )

    perm = True

    # Check dupes
    dedupe = list(set(shapes))
    if len(dedupe) < len(shapes):
        logger.error('There are duplicate shapes provided. Please remove all duplicates.')

    # Check if permutation mode
    for shape in shapes:
        if len(re.findall(r'\d+[ebtcr]', shape)) > 1:
            perm = False
        elif len(re.findall(r'\d+[elbtcr]', shape)) == 0:
            logger.error(ERROR_MESSAGE.format(shape))
        if len(re.findall(r'[^elbtcr\d]', shape)) > 0:
            logger.error(COMPONENT_ERROR_MESSAGE.format(shape, re.findall(r'[^elbtcr\d]', shape)))
        if shape.count('l') >= 1 and len(re.findall(r'\d+[elbtcr]', shape)) > 1:
            logger.error(f'Embedder shape (l) cannot be paired with others (tcr) in the same shape: {shape}.')
        for component in re.split(r'\d+[elbtcr]', shape):
            if component != '':
                logger.error(COMPONENT_ERROR_MESSAGE.format(shape, component))

    if perm:
        has_token = has_cache = has_rank = False
    else:
        has_token = has_cache = has_rank = None

    # Rest of the checks
    for shape in shapes:
        if perm:
            if shape.count('t') == 1:
                has_token = True
            elif shape.count('c') == 1:
                has_cache = True
            elif shape.count('r') == 1:
                if int(shape[:-1]) % 16 != 0:
                    logger.warning(
                        'lora rank * lora precision (in bytes) must be multiple of 16 due to hardware limitations.'
                    )
                has_rank = True
        else:
            curr_has_token = curr_has_cache = curr_has_rank = False
            for component in re.findall(r'\d+[btcr]', shape):
                if component.count('b') == 1:
                    logger.error(
                        'LLM multi-batch mode is not supported using the batch dimension, '
                        'but instead, is supported using the token dimension. Please multiply t by b.'
                    )
                elif component.count('t') == 1:
                    if curr_has_token:
                        logger.error(f'Multiple token (t) terms detected in shape: {shape}')
                    curr_has_token = True
                elif component.count('c') == 1:
                    if curr_has_cache:
                        logger.error(f'Multiple cache (c) terms detected in shape: {shape}')
                    curr_has_cache = True
                elif component.count('r') == 1:
                    lora_rank = int(re.findall(r'\d+r', component)[0][:-1])
                    if lora_rank % 16 != 0:
                        logger.warning(
                            'lora rank * lora precision (in bytes) must be multiple of 16 due to hardware limitations.'
                        )
                    if curr_has_rank:
                        logger.error(f'Multiple lora rank (r) terms detected in shape: {shape}')
                    curr_has_rank = True
                    if has_rank is not None and not has_rank:
                        logger.error(
                            "All shapes must have rank or all shapes don't have "
                            'rank, depending on whether the quantized model has dynamic lora inputs or not.'
                        )

            if has_rank is None:
                has_rank = curr_has_rank
            else:
                if has_rank and not curr_has_rank and 'e' not in shape:
                    logger.error(
                        "All shapes must have rank or all shapes don't have "
                        'rank, depending on whether the quantized model has dynamic lora inputs or not.'
                    )
            if shape.count('l') == 0 and 'e' not in shape:
                if not curr_has_token:
                    logger.error('Must provide at least 1 input token size (t) for all shapes')
                if not curr_has_cache:
                    logger.error('Must provide at least 1 cache size (c) for all shapes')

    if perm:
        if encoder_only:
            logger.warning(
                'Encoder only shape fixing, please ensure the provided shape is We, where W = encoder batch size.'
            )
        else:
            if not has_token:
                logger.error('Must provide at least 1 input token size (t) for permutation mode')
            if not has_cache:
                logger.error('Must provide at least 1 cache size (c) for permutation mode')
    else:
        if has_rank is None:
            has_rank = False

    return perm, has_rank


def check_support_quantizerstub():
    """Checks if QuantizerStub is supported in mtk_quantization.

    Returns:
        bool: True if QuantizerStub is supported, False otherwise.
    """
    import mtk_quantization

    return hasattr(mtk_quantization.pytorch.functional, 'QuantizerStub')


def check_supported_tokenizer(config):
    """Checks if the tokenizer is supported.

    Args:
        config (object): The configuration object.

    Raises:
        RuntimeError: If the config class or tokenizer is unsupported.
    """
    from ..models.configuration_base import BaseLLMConfig
    from .const import SUPPORTED_TOKENIZERS

    if not isinstance(config, BaseLLMConfig):
        logger.error(f'Unsupported config class: {type(config)}. config needs to be subclassed from BaseLLMConfig')

    if config.tokenizer not in SUPPORTED_TOKENIZERS:
        logger.error(f'Unsupported tokenizer: {config.tokenizer}. Supported tokenizers: {SUPPORTED_TOKENIZERS}')


def check_quantized_models(generative_model_folder, prompt_model_folder=None):
    """Checks if the quantized model models are valid.

    Args:
        generative_model_folder (str): The folder containing the generative model.
        prompt_model_folder (str, optional): The folder containing the prompt model.

    Raises:
        RuntimeError: If the models are invalid.
    """
    from . import utils

    gen_chunk_0 = utils.get_sorted_path_list(generative_model_folder, ['.tflite', '.mlir'])[0]

    gen_model_info = quantized_model_utils.extract_llm_quantized_model_info(gen_chunk_0)
    gen_num_token = gen_model_info['t']
    gen_cache_size = gen_model_info['c']

    if gen_num_token != 1:
        logger.error(
            'Expected generative_model to have fixed shape input token = 1, but '
            f'provided model has input token shape {gen_num_token}.'
        )

    gen_num_chunks = len(os.listdir(generative_model_folder))

    if prompt_model_folder is not None:
        prompt_chunk_0 = utils.get_sorted_path_list(prompt_model_folder, ['.tflite', '.mlir'])[0]

        prompt_model_info = quantized_model_utils.extract_llm_quantized_model_info(prompt_chunk_0)
        prompt_num_token = prompt_model_info['t']
        prompt_cache_size = prompt_model_info['c']

        if prompt_num_token == 1:
            logger.error(
                'Expected prompt_model to have fixed shape input token > 1, but provided model has input token shape 1.'
            )
        if not prompt_model_info.get('infini_attention', False) and prompt_cache_size != gen_cache_size:
            logger.error(
                'Expected prompt and generative models to have the same cache size,'
                f' but got prompt={prompt_cache_size} and generative= '
                f'{gen_cache_size}'
            )

        prompt_num_chunks = len(os.listdir(prompt_model_folder))

        if prompt_num_chunks != gen_num_chunks:
            logger.error(
                'Expected prompt and generative models to have the same number of '
                f'chunks, but got prompt={prompt_num_chunks} chunks and generative= '
                f'{gen_num_chunks} chunks.'
            )

        gen_separate_tail, gen_custom_tail = quantized_model_utils.has_separate_tails(generative_model_folder)
        prompt_separate_tail, prompt_custom_tail = quantized_model_utils.has_separate_tails(prompt_model_folder)

        if gen_separate_tail != prompt_separate_tail:
            logger.error(
                'Expected prompt and generative models to either both have separate '
                'final FCs or both not have separate final FCs, but got one with and one without.'
            )

        if gen_custom_tail != prompt_custom_tail:
            logger.error(
                'Expected prompt and generative models to either both have separate '
                'custom tails or both not have separate custom tails, but got one with and one without.'
            )


def check_tokenizer_exist(folder):
    """Checks if a tokenizer exists in the specified folder.

    Args:
        folder (str): The folder to check.

    Raises:
        FileNotFoundError: If the tokenizer is not found.
    """
    for f in os.listdir(folder):
        if f in ['tokenizer.model', 'tokenizer.json', 'vocab.json'] or f.endswith('.tiktoken'):
            return
    logger.error(
        f'Tokenizer not found in {folder}. Expected tokenizer.model, tokenizer.json, vocab.json, or tokenizer.tiktoken',
        err=FileNotFoundError,
    )


def check_weights_exist(weight_dir):
    """Checks if weight files exist in the specified directory.

    Args:
        weight_dir (str): The directory containing the weight files.

    Raises:
        FileNotFoundError: If no weight files are found in the directory.
    """
    if (
        len([f for f in os.listdir(weight_dir) if (f.endswith(('.bin', '.safetensors')) and 'training_args' not in f)])
        == 0
    ):
        logger.error(
            f'No weight files found in {weight_dir}! Weight files should be either .bin or .safetensors file types.',
            err=FileNotFoundError,
        )
