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
"""Script to fold `stable_embedding` Layer Norm into Embedding weights for easier rotation."""

import argparse
import glob
import json
import os
import shutil
import sys

import torch
import transformers.tokenization_utils_base as tokenizer_utils

from . import __version__
from .models.configuration_pipeline import PipelineConfig
from .models.llm.modeling_minicpm import convert_weight_to_llama_format
from .utils import logger, utils
from .utils import sanity_checks as sc

os.environ['MTK_LLM_SDK_SCRIPT'] = 'mtk_fuse_embed_layer_norm'


def get_argument_parser():
    """Gets the argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Fuse embed LayerNorm into embedding and save updated weights.',
        allow_abbrev=False,
    )
    parser.add_argument(
        'config',
        type=str,
        help='[Required] Model config json file. '
        'Model config must be in same directory as all model weight bins and tokenizer files.',
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
        RuntimeError: If any of the argument checks fail.
    """
    sc.check_exist(args.config, 'Config file')
    sc.check_ext(args.config, '.json', 'Config file')
    config = PipelineConfig(args.config)
    if config.g.name != 'text_only':
        logger.error(f"Expect get_embeds hook to be 'text_only', but got '{config.g.name}'.")

    with open(args.config) as f:
        config_json = json.load(f)

    return config, config_json


def _find_unique_tensor(state_dict, subkey):
    """Finds a unique tensor in the state dictionary with the specified subkey.

    Args:
        state_dict (dict): The state dictionary.
        subkey (str): The subkey to search for.

    Returns:
        tuple: A tuple containing the tensor and its key.

    Raises:
        RuntimeError: If there is not exactly one tensor with the specified subkey.
    """
    target_key = [key for key in state_dict if subkey in key]
    if len(target_key) != 1:
        logger.error(f'Expect exactly one tensor in the state_dict with subkey `{subkey}`.')

    target_key = target_key[0]
    return state_dict[target_key], target_key


def load_checkpoints(weight_dir, config):
    """Load checkpoints from the specified folders.

    Args:
        weight_dir (str): Directory of LLM weights.
        config (object): LLM configuration object.
    """
    checkpoint_files = [
        os.path.join(weight_dir, f)
        for f in os.listdir(weight_dir)
        if (f.startswith('pytorch_model') and f.endswith('.bin'))
        or (f.startswith('model') and f.endswith('.safetensors'))
    ]

    if len(checkpoint_files) == 0:
        logger.error(f'No checkpoint files found in {weight_dir}', err=FileNotFoundError)

    logger.info(f'Loading weights from: {weight_dir}')

    state_dict = {}
    is_safetensors = checkpoint_files[0].endswith('.safetensors')
    for i in range(len(checkpoint_files)):
        if is_safetensors:
            state_dict = {**state_dict, **utils.load_file(checkpoint_files[i])}
        else:
            state_dict = {**state_dict, **torch.load(checkpoint_files[i], map_location='cpu')}

    if config.early_exit_index is not None:
        prefix = ''
        for k in list(state_dict.keys()):
            if 'layers.' in k:
                if prefix == '':
                    prefix = k.split('layers.')[0]
                layer_idx = int(k.split('layers.')[1].split('.')[0])
                if layer_idx >= config.early_exit_index:
                    del state_dict[k]
            elif 'h.' in k:
                if prefix == '':
                    prefix = k.split('h.')[0]
                layer_idx = int(k.split('layers.')[1].split('.')[0])
                if layer_idx >= config.early_exit_index:
                    del state_dict[k]
            elif k == f'{prefix}norm.weight':
                del state_dict[k]
        expected_ee_checkpoint = os.path.join(weight_dir, f'ee{config.early_exit_index}.bin')
        if os.path.exists(expected_ee_checkpoint):
            logger.info(f'Loading weights from: {expected_ee_checkpoint}')
            state_dict = {**state_dict, **torch.load(expected_ee_checkpoint, map_location='cpu')}
        else:
            expected_ee_checkpoint = os.path.join(weight_dir, f'ee{config.early_exit_index}.safetensors')
            if os.path.exists(expected_ee_checkpoint):
                logger.info(f'Loading weights from: {expected_ee_checkpoint}')
                state_dict = {**state_dict, **utils.load_file(expected_ee_checkpoint)}
            else:
                logger.error(
                    f'Cannot find early exit checkpoint for early index {config.early_exit_index}. '
                    f'Expected either ee{config.early_exit_index}.bin or ee{config.early_exit_index}.safetensors',
                    err=FileNotFoundError,
                )
        state_dict[f'{prefix}norm.weight'] = state_dict.pop('ee.norm.weight')

    if (config.model_type in ['minicpm']) and (not config.llama_format):
        state_dict = convert_weight_to_llama_format(state_dict, config)

    return state_dict


def main(args=None):
    """Main function to fuse embedding and LayerNorm weights.

    This function parses the arguments, performs sanity checks, finds the target embedding and LayerNorm tensors,
    computes the LayerNorm fusion, updates the state dictionary and model configuration, and saves the updated
    state dictionary and model configuration to a new weight folder.

    Args:
        args (Namespace): custom argument passing from another python script.
    """
    if args is None:
        args = get_argument_parser().parse_args()
    if args.debug:
        logger.set_level('DEBUG')
    logger.debug(' '.join(f"'{i}'" if ' ' in i else i for i in sys.argv))
    logger.debug(f'mtk_llm_sdk version: {__version__}')
    config, config_json = args_sanity_checks(args)

    weight_dir = utils.get_dirpath(args.config)
    if not config.l.use_stable_embedding:
        logger.error("There's no need to run this script when `use_stable_embedding` is `False`.")

    # Find target Embedding and LayerNorm tensors
    state_dict = load_checkpoints(weight_dir, config.l)
    emb_weight, emb_weight_key = _find_unique_tensor(state_dict, config.l.embedding_key)
    ln_weight, ln_weight_key = _find_unique_tensor(state_dict, 'embed_layer_norm.weight')
    ln_bias, ln_bias_key = _find_unique_tensor(state_dict, 'embed_layer_norm.bias')

    # Compute LayerNorm fusion
    emb_w = emb_weight.double()
    mean = emb_w.mean(dim=-1, keepdim=True)
    var = torch.var(emb_w, dim=-1, keepdim=True, unbiased=False)
    eps = torch.nn.LayerNorm([]).eps
    scale = ln_weight.double()
    bias = ln_bias.double()
    emb_weight = (((emb_w - mean) / torch.sqrt(var + eps)) * scale + bias).to(emb_weight.dtype)

    # Update state_dict and model config
    state_dict[emb_weight_key] = emb_weight
    del state_dict[ln_weight_key]
    del state_dict[ln_bias_key]
    config_json.get('llm', config_json)['use_stable_embedding'] = False

    # Save updated state_dict and model config to new weight folder
    output_dir = f'{weight_dir}_embed_LN_fused'
    utils.recursive_remove_if_exist(output_dir, recreate=True)
    logger.info(f'Updated weights and model config are saved to `{output_dir}`.')
    torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
    new_config = os.path.join(output_dir, os.path.basename(args.config))
    with open(new_config, 'w') as f:
        f.write(json.dumps(config_json, indent=4))

    # Copy tokenizer to new weight folder
    for f in glob.glob(os.path.join(weight_dir, 'tokenizer*')):
        shutil.copy2(f, output_dir)

    # Copy additional tokenizer-related files to new weight folder
    for n in (tokenizer_utils.SPECIAL_TOKENS_MAP_FILE, tokenizer_utils.ADDED_TOKENS_FILE):
        f = os.path.join(weight_dir, n)
        if os.path.exists(f):
            shutil.copy2(f, output_dir)

    logger.info(f'Please check if all the required files are copied to `{output_dir}`!')


if __name__ == '__main__':
    main()
