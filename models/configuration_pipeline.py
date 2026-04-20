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
"""Define configuration of pipeline."""

import json

from ..utils import const, logger, utils
from ..utils.sanity_checks import check_exist, check_ext


class DummyConfig:
    """Dummy config class just to instantiate fake lora handler to count LoRAs from lora config."""

    def __init__(self, inner=False):
        """Initialize the dummy config class.

        Args:
            inner (bool, optional): flag to determine if to create an inner instance of dummy config within another
                dummy config.
        """
        if inner:
            self.num_hidden_layers = 32
            self.early_exit_index = None
        else:
            self.l = DummyConfig(inner=True)
            self.e = DummyConfig(inner=True)


class PipelineConfig:
    """PipelineConfig class for handling pipeline configuration settings.

    Attributes:
        p (BaseConfig): The preprocessor configuration, if any.
        e (BaseConfig): The encoder configuration, if any.
        l (BaseConfig): The LLM configuration.
        t (BaseConfig): The tail configuration, if any.
        ph (HookConfig): The pre-preprocessor hook configuration, if any.
        th (HookConfig): The pre-tokenizer hook configuration, if any.
        eh (HookConfig): The pre-encoder hook configuration, if any.
        p2h (HookConfig): The pre-projector hook configuration, if any.
        gh (HookConfig): The pre-getembed hook configuration, if any.
        lh (HookConfig): The pre-llm hook configuration, if any.
        t2h (HookConfig): The pre-tail hook configuration, if any.
        num_lora_configs (int): The number of LoRA configurations.
        lora_config_paths (list): The list of LoRA configuration file paths.
        custom_logits_processors (str): Custom logits processors.
        custom_stop_criteria (str): Custom stop criteria.

    Methods:
        __init__(config_or_config_path, verbose):
            Initialize the PipelineConfig.
    """

    def __init__(self, config_path, verbose=True):
        """Initialize the PipelineConfig.

        Args:
            config_path (str or dict): The path to the configuration json file.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        logger.debug(f'[PipelineConfig] Initialize PipelineConfig. config_or_config_path={config_path}')
        self.path = config_path
        check_ext(config_path, '.json')
        check_exist(config_path)
        with open(config_path) as f:
            config_ = json.load(f)

        # For quantized model flow, load model config from quantized_model_info.json
        if config_path.endswith('_info_0.json'):
            config_ = config_['model_config']

        assert isinstance(config_, dict)
        self.num_lora_configs = 0

        """
        Check if config is pre-3.0 or post-3.0.
        Only pure LLM configs are accepted for pre-3.0.
        For MLLM/medusa/eagle flows, must use post-3.0 config style.
        """
        if 'model_type' in config_:
            logger.debug(f'[PipelineConfig] Pre-3.0 config detected. model_type={config_["model_type"]}.')
            if config_['model_type'] in const.SUPPORTED_LLMS:
                config = {'llm': config_}
            else:
                if config_['model_type'] in const.SUPPORTED_COMBINED_CONFIGS:
                    config = parse_combined_config(config_)
                else:
                    logger.error(
                        'Old pre-3.0 config detected. Only pure LLM configs '
                        'or combined MLLM configs are supported. Got '
                        f'model_type={config_["model_type"]}, supported '
                        f'model_types: {const.SUPPORTED_LLMS + const.SUPPORTED_COMBINED_CONFIGS}',
                        err=NotImplementedError,
                    )
        else:
            logger.debug('[PipelineConfig] Post-3.0 config detected.')
            for k in config_:
                acceptable_keys = [
                    *const.ALL_PIPELINE_MODULES,
                    'bita',
                    'rotate',
                    'rotate_seed',
                    'rotate_mode',
                    'rotate_path',
                ]
                if k not in acceptable_keys:
                    logger.error(
                        f'Invalid config key: {k}. Config keys must be one of: {acceptable_keys}',
                        err=KeyError,
                    )
            if 'llm' not in config_:
                logger.error('config must minimally have "llm" key.', err=KeyError)
            config = config_

        if config['llm'].get('weight_dir', None) is None:
            config['llm']['weight_dir'] = utils.get_dirpath(config_path)

        self._p = utils.get_normalized_config(
            config.get('preprocessor', None), 'preprocessor', verbose=verbose
        )  # Preprocessor
        self._e = utils.get_normalized_config(config.get('encoder', None), 'encoder', verbose=verbose)  # Encoder
        self._p2 = utils.get_normalized_config(config.get('projector', None), 'projector', verbose=verbose)  # Projector
        self._l = utils.get_normalized_config(config['llm'], 'llm', verbose=verbose)  # LLM
        self._t = utils.get_normalized_config(config.get('tail', None), 'tail', verbose=verbose)  # Tail

        self._ft = utils.get_hook_config(
            config.get('format_text', None), 'format_text', verbose=verbose
        )  # format text hook
        self._ph = utils.get_hook_config(
            config.get('pre_preprocessor_hook', None), 'pre_preprocessor_hook', verbose=verbose
        )  # pre preprocessor hook
        self._th = utils.get_hook_config(
            config.get('pre_tokenizer_hook', None), 'pre_tokenizer_hook', verbose=verbose
        )  # pre tokenizer hook
        self._tf = utils.get_hook_config(
            config.get('tokenizer_func_hook', {'name': 'default'}), 'tokenizer_func_hook', verbose=verbose
        )  # tokenizer func hook
        self._eh = utils.get_hook_config(
            config.get('pre_encoder_hook', None), 'pre_encoder_hook', verbose=verbose
        )  # pre encoder hook
        self._p2h = utils.get_hook_config(
            config.get('pre_projector_hook', None), 'pre_projector_hook', verbose=verbose
        )  # pre projector hook
        self._gh = utils.get_hook_config(
            config.get('pre_getembed_hook', None), 'pre_getembed_hook', verbose=verbose
        )  # pre getembed hook
        self._g = utils.get_hook_config(
            config.get('get_embeds', {'name': 'text_only'}), 'get_embeds', verbose=verbose
        )  # getembeds hook
        self._lh = utils.get_hook_config(
            config.get('pre_llm_hook', None), 'pre_llm_hook', verbose=verbose
        )  # pre llm hook
        self._t2h = utils.get_hook_config(
            config.get('pre_tail_hook', None), 'pre_tail_hook', verbose=verbose
        )  # pre tail hook
        self._clp = utils.get_hook_config(
            config.get('logits_processor', None), 'logit_processor', verbose=verbose
        )  # custom logit processor
        self._csc = utils.get_hook_config(
            config.get('stop_criteria', None), 'stop_criteria', verbose=verbose
        )  # custom stop criteria

        config['rotate'] = self.rotate = config.get('rotate', config['llm'].get('rotate', False))
        config['rotate_seed'] = self.rotate_seed = config.get('rotate_seed', config['llm'].get('rotate_seed', 0))
        config['rotate_mode'] = self.rotate_mode = config.get(
            'rotate_mode', config['llm'].get('rotate_mode', 'hadamard')
        )
        config['rotate_path'] = self.rotate_path = config.get('rotate_path', None)
        config['bita'] = self.bita = config.get('bita', config['llm'].get('bita', False))
        self.config = config

    @property
    def p(self):
        """Preprocessor config getter."""
        return self._p

    @property
    def e(self):
        """Encoder config getter."""
        return self._e

    @property
    def p2(self):
        """Projector config getter."""
        return self._p2

    @property
    def l(self):  # noqa: E743
        """LLM config getter."""
        return self._l

    @property
    def t(self):
        """Tail config getter."""
        return self._t

    @t.setter
    def t(self, value):
        """Tail config setter."""
        self._t = value

    @property
    def ft(self):
        """format_text hook config getter."""
        return self._ft

    @property
    def ph(self):
        """Pre-preprocessor hook config getter."""
        return self._ph

    @property
    def th(self):
        """Pre-tokenizer hook config getter."""
        return self._th

    @property
    def tf(self):
        """Tokenizer function hook config getter."""
        return self._tf

    @property
    def eh(self):
        """Pre-encoder hook config getter."""
        return self._eh

    @property
    def p2h(self):
        """Pre-projector hook config getter."""
        return self._p2h

    @property
    def gh(self):
        """Pre-getembed hook config getter."""
        return self._gh

    @property
    def g(self):
        """Get embeds config getter."""
        return self._g

    @property
    def lh(self):
        """Pre-llm hook config getter."""
        return self._lh

    @property
    def t2h(self):
        """Pre-tail hook config getter."""
        return self._t2h

    @property
    def clp(self):
        """Custom logits processor config getter."""
        return self._clp

    @property
    def csc(self):
        """Custom stopping criteria config getter."""
        return self._csc

    def override_hooks_for_quantized_pipeline(self):
        """Forcefully set pre-projector hook to use Passthrough, and any numpy_to_torch to Passthrough."""
        logger.debug('Forcefully set pre_projector_hook to use Passthrough hook')
        self._p2h = utils.get_hook_config(None, 'pre_projector_hook', verbose=False)

        if self.ph.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_preprocessor_hook to use Passthrough hook')
            self._ph = utils.get_hook_config(None, 'pre_preprocessor_hook', verbose=False)
        if self.th.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_tokenizer_hook to use Passthrough hook')
            self._th = utils.get_hook_config(None, 'pre_tokenizer_hook', verbose=False)
        if self.eh.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_encoder_hook to use Passthrough hook')
            self._eh = utils.get_hook_config(None, 'pre_encoder_hook', verbose=False)
        if self.gh.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_getembed_hook to use Passthrough hook')
            self._gh = utils.get_hook_config(None, 'pre_getembed_hook', verbose=False)
        if self.lh.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_llm_hook to use Passthrough hook')
            self._lh = utils.get_hook_config(None, 'pre_llm_hook', verbose=False)
        if self.t2h.name == 'numpy_to_torch':
            logger.debug('Forcefully set pre_tail_hook to use Passthrough hook')
            self._t2h = utils.get_hook_config(None, 'pre_tail_hook', verbose=False)


def parse_combined_config(combined_config):
    """Parse combined configuration.

    Args:
        combined_config (dict): The combined configuration.

    Returns:
        dict: The parsed configuration.
    """
    logger.debug(f'Enter parse_combined_config, model_type={combined_config["model_type"]}')
    #                                                             # Hook2 | Hook4 | load_vision_weight |  -: Default O: implemented X: not implemented  # noqa: E501
    config = None
    if combined_config['model_type'] == 'llava':  #   -   |   O   |                    |  # noqa: E262
        config = {
            'format_text': {'name': 'llava'},
            'preprocessor': {
                'model_type': 'clip',
                'crop_size': combined_config.get('crop_size', 336),
                'do_center_crop': combined_config.get('do_center_crop', True),
                'do_normalize': combined_config.get('do_normalize', True),
                'do_resize': combined_config.get('do_resize', True),
                'image_mean': combined_config.get('image_mean', const.OPENAI_CLIP_MEAN),
                'image_std': combined_config.get('image_std', const.OPENAI_CLIP_STD),
                'resample': combined_config.get('resample', 3),
                'size': combined_config.get('size', 336),
                'data_format': combined_config.get('data_format', 'channels_last'),
            },
            'pre_encoder_hook': {
                'name': 'numpy_to_torch',
            },
            'encoder': {
                'model_type': 'clip',
                'layer_norm_eps': combined_config.get('layer_norm_eps', 1e-05),
                'hidden_size': combined_config.get('encoder_hidden_size', 1024),
                'image_size': combined_config.get('image_size', 336),
                'intermediate_size': combined_config.get('encoder_intermediate_size', 4096),
                'num_attention_heads': combined_config.get('encoder_num_attention_heads', 16),
                'num_hidden_layers': combined_config.get('encoder_num_hidden_layers', 24),
                'patch_size': combined_config.get('patch_size', 14),
                'vision_select_layer': combined_config.get('vision_select_layer', -2),
            },
            'pre_projector_hook': {
                'name': 'patch_select',
            },
            'projector': {
                'model_type': 'mlp_gelu',
                'image_size': combined_config.get('image_size', 336),
                'patch_size': combined_config.get('patch_size', 14),
                'projector_type': combined_config.get('projector_type', 'mlp2x_gelu'),
                'hidden_size': combined_config.get('projector_hidden_size', 1024),
                'projection_dim': combined_config.get('projection_dim', 4096),
            },
            'get_embeds': {
                'name': 'llava',
            },
            'llm': {
                'model_type': 'llama',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', 2),
                'hidden_size': combined_config.get('hidden_size', 4096),
                'intermediate_size': combined_config.get('intermediate_size', 11008),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 4096),
                'num_attention_heads': combined_config.get('num_attention_heads', 32),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 32),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 32),
                'pad_token_id': combined_config.get('pad_token_id', 0),
                'rms_norm_eps': combined_config.get('rms_norm_eps', 1e-05),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 32000),
                'norm': combined_config.get('norm', 'RMSNorm'),
            },
        }

    elif combined_config['model_type'] == 'tiny_llava':
        logger.error(
            'tiny_llava not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'llava-ov':
        logger.error(
            'llava-ov not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'llava-next':
        logger.error(
            'llava-next not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'mobilevlm':
        logger.error(
            'mobilevlm not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'paligemma':
        logger.error(
            'paligemma not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'vila':
        logger.error(
            'vila not yet back-ported in the current version of mtk_llm_sdk. Please manually port the model '
            'yourself or wait for official support in future versions of mtk_llm_sdk.',
            err=NotImplementedError,
        )

    elif combined_config['model_type'] == 'internvl2-1b':
        image_size = combined_config.get('image_size', 448)
        patch_size = combined_config.get('patch_size', 14)
        downsample_ratio = combined_config.get('downsample_ratio', 0.5)
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
        config = {
            'format_text': {'name': 'internvl2', 'num_image_token': num_image_token},
            'preprocessor': {
                'model_type': 'intern_vit_6b',
                'force_image_size': combined_config.get('force_image_size', 448),
                'max_dynamic_patch': combined_config.get('max_dynamic_patch', 12),
                'min_dynamic_patch': combined_config.get('min_dynamic_patch', 1),
                'use_thumbnail': combined_config.get('use_thumbnail', True),
            },
            'pre_encoder_hook': {
                'name': 'pad_internvl2',
            },
            'encoder': {
                'model_type': 'intern_vit_6b',
                'hidden_act': combined_config.get('hidden_act', 'gelu'),
                'hidden_size': combined_config.get('encoder_hidden_size', 1024),
                'image_size': image_size,
                'intermediate_size': combined_config.get('encoder_intermediate_size', 4096),
                'layer_norm_eps': combined_config.get('layer_norm_eps', 1e-06),
                'num_attention_heads': combined_config.get('encoder_num_attention_heads', 16),
                'num_channels': combined_config.get('num_channels', 3),
                'num_hidden_layers': combined_config.get('encoder_num_hidden_layers', 24),
                'patch_size': patch_size,
                'qk_normalization': combined_config.get('qk_normalization', False),
                'qkv_bias': combined_config.get('qkv_bias', True),
                'downsample_ratio': downsample_ratio,
            },
            'pre_projector_hook': {
                'name': 'internvl2_pixel_shuffle',
                'select_layer': combined_config.get('select_layer', -1),
                'downsample_ratio': downsample_ratio,
            },
            'projector': {
                'model_type': 'internvl2',
                'encoder_hidden_size': combined_config.get('encoder_hidden_size', 1024),
                'hidden_size': combined_config.get('hidden_size', 2048),
                'downsample_ratio': downsample_ratio,
                'image_size': image_size,
                'patch_size': patch_size,
            },
            'get_embeds': {
                'name': 'internvl2',
            },
            'llm': {
                'model_type': 'qwen2',
                'bos_token_id': combined_config.get('bos_token_id', 151643),
                'eos_token_id': combined_config.get('eos_token_id', 151645),
                'hidden_size': combined_config.get('hidden_size', 896),
                'intermediate_size': combined_config.get('intermediate_size', 4864),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 32768),
                'num_attention_heads': combined_config.get('num_attention_heads', 14),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 24),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 2),
                'rms_norm_eps': combined_config.get('rms_norm_eps', 1e-06),
                'rope_theta': combined_config.get('rope_theta', 1000000.0),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 151655),
                'mask_value': combined_config.get('mask_value', -10000.0),
            },
        }

    elif combined_config['model_type'] == 'internvl2-2b':
        image_size = combined_config.get('image_size', 448)
        patch_size = combined_config.get('patch_size', 14)
        downsample_ratio = combined_config.get('downsample_ratio', 0.5)
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
        config = {
            'format_text': {'name': 'internvl2', 'num_image_token': num_image_token},
            'preprocessor': {
                'model_type': 'intern_vit_6b',
                'force_image_size': combined_config.get('force_image_size', 448),
                'max_dynamic_patch': combined_config.get('max_dynamic_patch', 12),
                'min_dynamic_patch': combined_config.get('min_dynamic_patch', 1),
                'use_thumbnail': combined_config.get('use_thumbnail', True),
            },
            'pre_encoder_hook': {
                'name': 'pad_internvl2',
            },
            'encoder': {
                'model_type': 'intern_vit_6b',
                'hidden_act': combined_config.get('hidden_act', 'gelu'),
                'hidden_size': combined_config.get('encoder_hidden_size', 1024),
                'image_size': combined_config.get('image_size', 448),
                'intermediate_size': combined_config.get('encoder_intermediate_size', 4096),
                'layer_norm_eps': combined_config.get('layer_norm_eps', 1e-06),
                'num_attention_heads': combined_config.get('encoder_num_attention_heads', 16),
                'num_channels': combined_config.get('num_channels', 3),
                'num_hidden_layers': combined_config.get('encoder_num_hidden_layers', 24),
                'patch_size': combined_config.get('patch_size', 14),
                'qk_normalization': combined_config.get('qk_normalization', False),
                'qkv_bias': combined_config.get('qkv_bias', True),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'pre_projector_hook': {
                'name': 'internvl2_pixel_shuffle',
                'select_layer': combined_config.get('select_layer', -1),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'projector': {
                'model_type': 'internvl2',
                'encoder_hidden_size': combined_config.get('encoder_hidden_size', 1024),
                'hidden_size': combined_config.get('hidden_size', 2048),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
                'image_size': combined_config.get('image_size', 448),
                'patch_size': combined_config.get('patch_size', 14),
            },
            'get_embeds': {
                'name': 'internvl2',
            },
            'llm': {
                'model_type': 'internlm2',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', 2),
                'hidden_size': combined_config.get('hidden_size', 2048),
                'intermediate_size': combined_config.get('intermediate_size', 8192),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 32768),
                'num_attention_heads': combined_config.get('num_attention_heads', 16),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 24),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 8),
                'pad_token_id': combined_config.get('pad_token_id', 2),
                'rms_norm_eps': combined_config.get('rms_norm_eps', 1e-05),
                'rope_theta': combined_config.get('rope_theta', 1000000),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 92553),
            },
        }

    elif combined_config['model_type'] == 'internvl2-4b':
        image_size = combined_config.get('image_size', 448)
        patch_size = combined_config.get('patch_size', 14)
        downsample_ratio = combined_config.get('downsample_ratio', 0.5)
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
        config = {
            'format_text': {'name': 'internvl2', 'num_image_token': num_image_token},
            'preprocessor': {
                'model_type': 'intern_vit_6b',
                'force_image_size': combined_config.get('force_image_size', 448),
                'max_dynamic_patch': combined_config.get('max_dynamic_patch', 12),
                'min_dynamic_patch': combined_config.get('min_dynamic_patch', 1),
                'use_thumbnail': combined_config.get('use_thumbnail', True),
            },
            'pre_encoder_hook': {
                'name': 'pad_internvl2',
            },
            'encoder': {
                'model_type': 'intern_vit_6b',
                'hidden_act': combined_config.get('hidden_act', 'gelu'),
                'hidden_size': combined_config.get('hidden_size', 1024),
                'image_size': combined_config.get('image_size', 448),
                'intermediate_size': combined_config.get('intermediate_size', 4096),
                'layer_norm_eps': combined_config.get('layer_norm_eps', 1e-06),
                'num_attention_heads': combined_config.get('num_attention_heads', 16),
                'num_channels': combined_config.get('num_channels', 3),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 24),
                'patch_size': combined_config.get('patch_size', 14),
                'qk_normalization': combined_config.get('qk_normalization', False),
                'qkv_bias': combined_config.get('qkv_bias', True),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'pre_projector_hook': {
                'name': 'internvl2_pixel_shuffle',
                'select_layer': combined_config.get('select_layer', -1),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'projector': {
                'model_type': 'internvl2',
                'hidden_size': combined_config.get('hidden_size', 2048),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
                'image_size': combined_config.get('image_size', 448),
                'patch_size': combined_config.get('patch_size', 14),
            },
            'get_embeds': {
                'name': 'internvl2',
            },
            'llm': {
                'model_type': 'phi3',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', 32000),
                'hidden_size': combined_config.get('hidden_size', 3072),
                'intermediate_size': combined_config.get('intermediate_size', 8192),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 131072),
                'num_attention_heads': combined_config.get('num_attention_heads', 32),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 32),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 32),
                'pad_token_id': combined_config.get('pad_token_id', 32000),
                'rms_norm_eps': combined_config.get('rms_norm_eps', 1e-05),
                'rope_scaling': combined_config.get('rope_scaling', None),
                'rope_theta': combined_config.get('rope_theta', 10000.0),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 32020),
            },
        }

    elif combined_config['model_type'] == 'internvl2-8b':
        image_size = combined_config.get('image_size', 448)
        patch_size = combined_config.get('patch_size', 14)
        downsample_ratio = combined_config.get('downsample_ratio', 0.5)
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
        config = {
            'format_text': {'name': 'internvl2', 'num_image_token': num_image_token},
            'preprocessor': {
                'model_type': 'intern_vit_6b',
                'force_image_size': combined_config.get('force_image_size', 448),
                'max_dynamic_patch': combined_config.get('max_dynamic_patch', 12),
                'min_dynamic_patch': combined_config.get('min_dynamic_patch', 1),
                'use_thumbnail': combined_config.get('use_thumbnail', True),
            },
            'pre_encoder_hook': {
                'name': 'pad_internvl2',
            },
            'encoder': {
                'model_type': 'intern_vit_6b',
                'hidden_act': combined_config.get('hidden_act', 'gelu'),
                'hidden_size': combined_config.get('hidden_size', 1024),
                'image_size': combined_config.get('image_size', 448),
                'intermediate_size': combined_config.get('intermediate_size', 4096),
                'layer_norm_eps': combined_config.get('layer_norm_eps', 1e-06),
                'num_attention_heads': combined_config.get('num_attention_heads', 16),
                'num_channels': combined_config.get('num_channels', 3),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 24),
                'patch_size': combined_config.get('patch_size', 14),
                'qk_normalization': combined_config.get('qk_normalization', False),
                'qkv_bias': combined_config.get('qkv_bias', True),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'pre_projector_hook': {
                'name': 'internvl2_pixel_shuffle',
                'select_layer': combined_config.get('select_layer', -1),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
            },
            'projector': {
                'model_type': 'internvl2',
                'hidden_size': combined_config.get('hidden_size', 2048),
                'downsample_ratio': combined_config.get('downsample_ratio', 0.5),
                'image_size': combined_config.get('image_size', 448),
                'patch_size': combined_config.get('patch_size', 14),
            },
            'get_embeds': {
                'name': 'internvl2',
            },
            'llm': {
                'model_type': 'internlm2',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', 2),
                'hidden_size': combined_config.get('hidden_size', 4096),
                'intermediate_size': combined_config.get('intermediate_size', 14336),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 32768),
                'num_attention_heads': combined_config.get('num_attention_heads', 32),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 32),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 8),
                'pad_token_id': combined_config.get('pad_token_id', 2),
                'rms_norm_eps': combined_config.get('rms_norm_eps', 1e-05),
                'rope_theta': combined_config.get('rope_theta', 1000000),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 92553),
            },
        }

    elif combined_config['model_type'] == 'qwen2_vl':
        preprocessor_config = {
            'model_type': 'qwen2_vl',
            'min_pixels': combined_config.get('min_pixels', 3136),
            'max_pixels': combined_config.get('max_pixels', 28 * 28 * 1280),
            'patch_size': combined_config.get('patch_size', 14),
            'temporal_patch_size': combined_config.get('temporal_patch_size', 2),
            'merge_size': combined_config.get('merge_size', 2),
            'image_mean': combined_config.get('image_mean', const.OPENAI_CLIP_MEAN),
            'image_std': combined_config.get('image_std', const.OPENAI_CLIP_STD),
        }
        config = {
            'format_text': {'name': 'qwen2_vl'},
            'preprocessor': preprocessor_config,
            'pre_encoder_hook': {
                'name': 'qwen2vl_preencoder',
                'spatial_merge_size': combined_config.get('spatial_merge_size', 2),
                'embed_dim': combined_config.get('embed_dim', 1280),
                'num_heads': combined_config.get('encoder_num_head', 16),
            },
            'encoder': {
                'model_type': 'qwen2_vl_vision',
                'num_hidden_layers': combined_config.get('depth', 32),
                'embed_dim': combined_config.get('embed_dim', 1280),
                'mlp_ratio': combined_config.get('mlp_ratio', 4),
                'num_head': combined_config.get('encoder_num_head', 16),
                'in_chans': combined_config.get('in_chans', 3),
                'hidden_size': combined_config.get('encoder_hidden_size', 1536),
                'patch_size': combined_config.get('patch_size', 14),
                'spatial_merge_size': combined_config.get('spatial_merge_size', 2),
                'spatial_patch_size': combined_config.get('spatial_patch_size', 14),
                'temporal_patch_size': combined_config.get('temporal_patch_size', 2),
                'use_conv2d_patch_embed': combined_config.get('use_conv2d_patch_embed', True),
                'preprocessor_config': combined_config.get('preprocessor_config', preprocessor_config),  # For PTQ
                'image_resolution': combined_config.get('image_resolution', [924, 1064]),  # For PTQ
            },
            'pre_projector_hook': {},
            'projector': {
                'model_type': 'qwen2_vl',
                'dim': combined_config.get('hidden_size', 1536),
                'embed_dim': combined_config.get('embed_dim', 1280),
                'preprocessor_config': combined_config.get('preprocessor_config', preprocessor_config),  # For PTQ
                'image_resolution': combined_config.get('image_resolution', [924, 1064]),  # For PTQ
                'spatial_merge_size': combined_config.get('spatial_merge_size', 2),  # For PTQ
                'num_head': combined_config.get('encoder_num_head', 16),  # For PTQ
            },
            'get_embeds': {'name': 'qwen2_vl', 'image_token_ids': combined_config.get('image_token_id', 151655)},
            'pre_llm_hook': {
                'name': 'qwen2vl_prellm',
                'spatial_merge_size': combined_config.get('spatial_merge_size', 2),
                'image_token_id': combined_config.get('image_token_id', 151655),
                'video_token_id': combined_config.get('video_token_id', 151656),
                'vision_start_token_id': combined_config.get('vision_start_token_id', 151652),
            },
            'llm': {
                'model_type': 'qwen2',
                'bos_token_id': combined_config.get('bos_token_id', 151643),
                'eos_token_id': combined_config.get('eos_token_id', 151653),
                'hidden_size': combined_config.get('hidden_size', 1536),
                'intermediate_size': combined_config.get('intermediate_size', 8960),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 32768),
                'num_attention_heads': combined_config.get('num_attention_heads', 12),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 28),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 2),
                'norm_eps': combined_config.get('rms_norm_eps', 1e-6),
                'rope_theta': combined_config.get('rope_theta', 1000000.0),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', True),
                'vocab_size': combined_config.get('vocab_size', 151936),
                'mask_value': combined_config.get('mask_value', -10000),  # Important for Qwen2 based model
                'rope_scaling': combined_config.get('rope_scaling', None),
                'is_vl': True,
            },
        }

    elif combined_config['model_type'] == 'phi3v':
        preprocessor_config = {
            'model_type': 'phi3v',
            'num_crops': combined_config.get('num_crops', 16),
            'do_convert_rgb': combined_config.get('do_convert_rgb', True),
            'image_mean': combined_config.get('image_mean', const.OPENAI_CLIP_MEAN),
            'image_std': combined_config.get('image_std', const.OPENAI_CLIP_STD),
            'use_hd_transform': combined_config.get('use_hd_transform', True),
        }
        config = {
            'format_text': {'name': 'phi3v'},
            'preprocessor': preprocessor_config,
            'pre_encoder_hook': {
                'name': 'pad_phi3v',
            },
            'encoder': {
                'model_type': 'phi3_vision_emb',
                'hidden_size': combined_config.get('hidden_size', 3072),
                'img_processor': combined_config.get('img_processor', None),  # must pass
                'vocab_size': combined_config.get('vocab_size', 32064),
                'use_hd_transform': combined_config.get('use_hd_transform', True),
                'with_learnable_separator': combined_config.get('with_learnable_separator', True),
                'hd_transform_order': combined_config.get('hd_transform_order', 'sub_glb'),
                'fixed_img_size': combined_config.get('fixed_img_size', None),
            },
            'pre_projector_hook': {
                'name': 'phi3v_preprojector',
                'use_hd_transform': combined_config.get('use_hd_transform', True),
                'hd_transform_order': combined_config.get('hd_transform_order', 'sub_glb'),
                'with_learnable_separator': combined_config.get('with_learnable_separator', True),
                'sub_glb_GN_path': combined_config.get('sub_glb_GN_path', None),
                'fixed_img_size': combined_config.get('fixed_img_size', None),
            },
            'projector': {
                'model_type': 'phi3v',
                'projector_cls': combined_config.get('projector_cls', 'mlp'),
                'use_hd_transform': combined_config.get('use_hd_transform', True),
                'img_dim_out': combined_config.get('img_dim_out', 1024),
                'hidden_size': combined_config.get('hidden_size', 3072),
                'fixed_img_size': combined_config.get('fixed_img_size', None),
            },
            'tokenizer_func_hook': {
                'name': 'phi3v',
            },
            'get_embeds': {
                'name': 'phi3v',
                'num_img_tokens': combined_config.get('num_img_tokens', 144),
                'vocab_size': combined_config.get('vocab_size', 32064),
            },
            'llm': {
                'model_type': 'phi3',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', 32000),
                'hidden_size': combined_config.get('hidden_size', 3072),
                'intermediate_size': combined_config.get('intermediate_size', 8192),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 131072),
                'num_attention_heads': combined_config.get('num_attention_heads', 32),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 32),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 32),
                'norm_eps': combined_config.get('rms_norm_eps', 1e-5),
                'original_max_position_embeddings': combined_config.get('original_max_position_embeddings', 4096),
                'rope_theta': combined_config.get('rope_theta', 10000.0),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 32064),
                'tokenizer': combined_config.get('tokenizer', 'llama_fast'),
                'rope_scaling': combined_config.get('rope_scaling', None),  # must pass
            },
        }

    elif combined_config['model_type'] == 'minicpmv':
        config = {
            'format_text': {
                'name': 'minicpmv',
                'query_num': combined_config.get('query_num', 64),
                'use_image_id': combined_config.get('query_num', True),
            },
            'preprocessor': {
                'model_type': 'minicpmv_navit_siglip',
                'do_center_crop': combined_config.get('do_center_crop', False),
                'do_normalize': combined_config.get('do_normalize', True),
                'do_resize': combined_config.get('do_resize', True),
                'do_reshape_by_patch': combined_config.get('do_reshape_by_patch', True),
                'do_slice_mode': combined_config.get('do_slice_mode', False),
                'image_mean': combined_config.get('image_mean', (0.5, 0.5, 0.5)),
                'image_std': combined_config.get('image_std', (0.5, 0.5, 0.5)),
                'resample': combined_config.get('resample', 3),
                'size': combined_config.get('size', 448),
                'patch_size': combined_config.get('patch_size', 14),
                'max_slice_nums': combined_config.get('max_slice_nums', 9),
                'scale_resolution': combined_config.get('scale_resolution', 448),
                'data_format': combined_config.get('data_format', 'channels_last'),
            },
            'pre_encoder_hook': {
                'name': 'numpy_to_torch',
            },
            'encoder': {
                'model_type': 'minicpmv_navit_siglip',
                'layer_norm_eps': combined_config['vision_config'].get('layer_norm_eps', 1e-06),
                'hidden_size': combined_config['vision_config'].get('hidden_size', 1152),
                'image_size': combined_config['vision_config'].get('image_size', 980),
                'intermediate_size': combined_config['vision_config'].get('intermediate_size', 4304),
                'num_attention_heads': combined_config['vision_config'].get('num_attention_heads', 16),
                'num_hidden_layers': combined_config['vision_config'].get('num_hidden_layers', 27),
                'patch_size': combined_config['vision_config'].get('patch_size', 14),
                'vision_select_layer': combined_config['vision_config'].get('vision_select_layer', -1),
                'ptq_image_batch': combined_config['vision_config'].get('ptq_image_width', 1),
                'ptq_image_width': combined_config['vision_config'].get('ptq_image_width', 448),
                'ptq_image_height': combined_config['vision_config'].get('ptq_image_height', 448),
                'image_text': combined_config['vision_config'].get('image_text', '<unk>'),
                'image_token': combined_config['vision_config'].get('image_token_id', 0),
                'mlp_gelu': combined_config['vision_config'].get('mlp_gelu', 'gelu_pytorch_tanh'),
                'do_reshape_by_patch': combined_config.get('do_reshape_by_patch', True),
            },
            'pre_projector_hook': {},
            'projector': {
                'model_type': 'minicpmv',
                'grid_size': int(combined_config.get('query_num', 64) ** 0.5),
                'embed_dim': combined_config.get('hidden_size', 1536),
                'kv_dim': combined_config.get('mm_hidden_size', 1152),
                'use_resampler_embed': combined_config.get('mm_use_resampler_embed', False),
                'patch_size': combined_config['vision_config'].get('patch_size', 14),
                'ptq_image_batch': combined_config['vision_config'].get('ptq_image_width', 1),
                'ptq_image_width': combined_config['vision_config'].get('ptq_image_width', 448),
                'ptq_image_height': combined_config['vision_config'].get('ptq_image_height', 448),
            },
            'get_embeds': {'name': 'minicpmv', 'image_token_ids': combined_config.get('image_token_id', 0)},
            'llm': {
                'model_type': 'minicpm',
                'bos_token_id': combined_config.get('bos_token_id', 1),
                'eos_token_id': combined_config.get('eos_token_id', [2, 73440]),
                'hidden_size': combined_config.get('hidden_size', 1536),
                'intermediate_size': combined_config.get('intermediate_size', 3840),
                'max_position_embeddings': combined_config.get('max_position_embeddings', 32768),
                'num_attention_heads': combined_config.get('num_attention_heads', 24),
                'num_hidden_layers': combined_config.get('num_hidden_layers', 52),
                'num_key_value_heads': combined_config.get('num_key_value_heads', 8),
                'norm_eps': combined_config.get('rms_norm_eps', 1e-5),
                'original_max_position_embeddings': combined_config.get('original_max_position_embeddings', 32768),
                'rope_theta': combined_config.get('rope_theta', 10000.0),
                'tie_word_embeddings': combined_config.get('tie_word_embeddings', False),
                'vocab_size': combined_config.get('vocab_size', 73464),
                'tokenizer': combined_config.get('tokenizer', 'llama'),
                'rope_scaling': combined_config.get('rope_scaling', None),
                'scale_depth': combined_config.get('scale_depth', 1.4),
                'scale_emb': combined_config.get('scale_emb', 12),
                'dim_model_base': combined_config.get('dim_model_base', 256),
            },
        }

    elif combined_config['model_type'] == 'whisper':
        gen_config_path = combined_config.get('gen_config_path', None)
        if gen_config_path is None:
            logger.error('gen_config_path is required but missing from config.json', err=KeyError)
        check_ext(gen_config_path, '.json')
        check_exist(gen_config_path)
        with open(gen_config_path) as f:
            processor_config = json.load(f)
        config = {
            'preprocessor': {
                'model_type': 'whisper',
                'feature_size': combined_config.get('num_mel_bins', None),
                'sampling_rate': combined_config.get('sampling_rate', 16000),
            },
            'pre_encoder_hook': {
                'name': 'numpy_to_torch',
            },
            'encoder': {
                'model_type': 'whisper',
                'decoder_layers': combined_config.get('decoder_layers', None),
                'd_model': combined_config.get('d_model', None),
                'encoder_attention_heads': combined_config.get('encoder_attention_heads', None),
                'encoder_ffn_dim': combined_config.get('encoder_ffn_dim', None),
                'encoder_layers': combined_config.get('encoder_layers', None),
                'max_source_positions': combined_config.get('max_source_positions', None),
                'max_position_embeddings': combined_config.get('max_target_positions', None),
                'num_mel_bins': combined_config.get('num_mel_bins', None),
            },
            'llm': {
                'model_type': 'whisper_decoder',
                'bos_token_id': combined_config.get('bos_token_id', 50257),
                'eos_token_id': combined_config.get('eos_token_id', 50257),
                'hidden_size': combined_config.get('d_model', None),
                'intermediate_size': combined_config.get('decoder_ffn_dim', None),
                'max_source_positions': combined_config.get('max_source_positions', None),
                'max_position_embeddings': combined_config.get('max_target_positions', None),
                'num_attention_heads': combined_config.get('decoder_attention_heads', None),
                'num_key_value_heads': combined_config.get('decoder_attention_heads', None),
                'num_hidden_layers': combined_config.get('decoder_layers', None),
                'num_mel_bins': combined_config.get('num_mel_bins', None),
                'pad_token_id': combined_config.get('pad_token_id', None),
                'vocab_size': combined_config.get('vocab_size', None),
            },
            'logits_processor': {
                'name': 'whisper',
                'eos_token_id': processor_config.get('eos_token_id', None),
                'forced_decoder_ids': processor_config.get('forced_decoder_ids', None),
                'max_initial_timestamp_index': processor_config.get('max_initial_timestamp_index', None),
                'no_timestamps_token_id': processor_config.get('no_timestamps_token_id', None),
                'suppress_tokens': processor_config.get('suppress_tokens', None),
            },
        }
    if config is None:
        logger.error(f'Unsupported combined model: {combined_config["model_type"]}', err=NotImplementedError)

    logger.debug(f'Parsed config={config}')
    return config
