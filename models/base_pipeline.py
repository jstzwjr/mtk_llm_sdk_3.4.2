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
"""Define base pipeline class."""

from abc import ABC, abstractmethod
from multiprocessing import set_start_method
from pathlib import Path

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

from ..utils import const, logger, utils
from ..utils.cache_utils import Memory
from ..utils.precision_config import PTQPrecisionConfig
from .configuration_base import BaseConfig
from .configuration_pipeline import PipelineConfig
from .lora_handler import LoRAHandler


class BasePipeline(ABC):
    """BasePipeline class for handling various stages of a machine learning pipeline.

    Attributes:
        args (tuple): Additional arguments.
        kwargs (dict): Additional keyword arguments.
        config (PipelineConfig): The pipeline configuration.
        task (str): The task to be performed.
        tokenizer (Tokenizer): The tokenizer for text processing.
        preprocessor (object): The preprocessor object.
        encoder (list): The list of encoder layers.
        llm (list): The list of LLM layers.
        tail (object): The tail object.
        pre_preprocessor_hook (object): The pre-preprocessor hook.
        pre_tokenizer_hook (object): The pre-tokenizer hook.
        pre_encoder_hook (object): The pre-encoder hook.
        pre_projector_hook (object): The pre-projector hook.
        pre_getembed_hook (object): The pre-get embed hook.
        pre_llm_hook (object): The pre-llm hook.
        pre_tail_hook (object): The pre-tail hook.
        text_embedding_layer (object): The text embedding layer.
        encoder_lora_inputs (list): The list of encoder LoRA inputs.
        llm_lora_inputs (list): The list of LLM LoRA inputs.
        llm_layers_per_chunk (list): The list of LLM layers per chunk.
        num_encoder_layers (int): The number of encoder layers.
        num_decoder_layers (int): The number of decoder layers.

    Methods:
        __init__(combined_config, lora_configs, task, *args, **kwargs): Initialize the BasePipeline.
        init_preprocessor(): Initialize the preprocessor.
        init_encoder(): Initialize the encoder.
        init_llm(): Initialize the LLM.
        init_tail(): Initialize the tail.
        init_hooks(): Initialize the hooks.
        has_encoder(): Check if the encoder is present.
        has_preprocessor(): Check if the preprocessor is present.
        has_custom_tail(): Check if a custom tail is present.
        load_multimodal_input(*inputs): Load multimodal input.
        hook1(inputs): Forward pass through the first hook.
        forward_preprocessor(*inputs): Forward pass through the preprocessor.
        hook2(inputs): Forward pass through the second hook.
        forward_tokenizer(*inputs): Forward pass through the tokenizer.
        hook3(inputs): Forward pass through the third hook.
        forward_encoder(*inputs): Forward pass through the encoder.
        hook4(inputs): Forward pass through the fourth hook.
        get_embeds(tokens, multimodal_embeds, custom_embeds): Get embeddings.
        hook5(inputs): Forward pass through the fifth hook.
        generate_llm(input_embeds, text_tokens, num_token, cache_size, repetition_penalty, max_new_tokens, temperature,
            top_p, top_k): Forward pass through the LLM.
        hook6(inputs): Forward pass through the sixth hook.
        forward_tail(*inputs): Forward pass through the tail.
        deduce_num_layers_per_chunk(): Deduce the number of blocks per chunk.
        forward(text_tokens, multimodal_inputs, custom_embeds, repetition_penalty, max_new_tokens, num_token,
            cache_size, temperature, top_p, top_k): Forward pass through the pipeline.
        sampler(logits, input_ids, logits_processors, logits_warper): Sample the next token.
        get_response(tokens, preformatter, input_length): Get the response.
        close_response_handler(): Close the response handler.
        get_image_mask(): Calculate the input image mask.
        get_text_embeds(): Process token into text embedding.
    """

    def __init__(self, combined_config, lora_configs, task, *args, **kwargs):
        """Initialize the BasePipeline.

        Args:
            combined_config (dict): The combined configuration.
            lora_configs (dict): The LoRA configurations.
            task (str): The task to be performed.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        set_start_method('spawn', force=True)  # Needed to use multiprocessing with CUDA
        self.args = args
        self.kwargs = kwargs
        self.task = task
        if self.task not in const.ALL_SDK_TASKS:
            logger.error(f'Unsupported task: {self.task}', err=ValueError)

        self.backend = kwargs.pop('backend', const.CONVERTER)
        if self.backend not in const.SUPPORTED_BACKENDS:
            logger.error(f'Unsupported backend: {self.backend}', err=ValueError)

        self.config = (
            combined_config
            if isinstance(combined_config, PipelineConfig)
            else PipelineConfig(combined_config, verbose=self.task != 'test_tokenizer')
        )
        self.lora_handler = LoRAHandler(
            lora_configs,
            self.config,
            quiet=kwargs.get('quiet'),
            dummy_lora=(kwargs.get('dummy_weights', False) and lora_configs != []),
        )

        # Don't need tokenizer: input is tokens/embeddings, or task is export_lora_bins or ptq
        # Need tokenizer: All other tasks/input_modes
        tokenizer = True
        self.input_mode = self.kwargs.get('input_mode', None)
        if self.input_mode == 'embeddings' or self.task in ['export_lora', 'ptq', 'hotplug']:
            tokenizer = False
        # TODO(Ting-An.Chien@mediatek.com): Merge with previous condition.
        if self.backend == const.MLKITS:
            tokenizer = True
        logger.debug(f'input_mode={self.input_mode}, tokenizer={tokenizer}')

        # assume older device if version not given
        self.use_single_bmm_attention = self.kwargs.get('use_single_bmm_attention', False)
        logger.debug(f'use_single_bmm_attention={self.use_single_bmm_attention}')

        self.tokenizer = (
            None
            if not tokenizer
            else utils.get_tokenizer(self.config.l, self.config.l.weight_dir, add_bos=self.kwargs.get('add_bos', True))
        )

        self.preprocessor = None
        self.encoder = []
        self.encoder_inputs = set()
        self.projector = None
        self.llm = []
        self.llm_prompt = None
        self.llm_gen = None
        self.tail = None
        self.format_text = None
        self.pre_preprocessor_hook = None
        self.pre_tokenizer_hook = None
        self.tokenizer_func = None
        self.pre_encoder_hook = None
        self.pre_projector_hook = None
        self.pre_getembed_hook = None
        self.get_embeds = None
        self.pre_llm_hook = None
        self.pre_tail_hook = None
        self.text_embedding_layer = None
        self.main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.encoder_layers_per_chunk = None
        self.llm_layers_per_chunk = None
        self.encoder_layer_ids = None
        self.llm_layer_ids = None

        self.precision_config = None

        self.num_encoder_layers = 0
        self.num_decoder_layers = 0

        self._debug = kwargs.pop('debug', False)

        self._mm_token_ids = []
        self.input_length = None
        self.overture = None
        self.use_bita = False

        self._pipeline_type = None

        self.evict_method = kwargs.get('evict_method')
        self.evict_class = utils.resolve_evictor_class(self.evict_method)

        self.prompt_formatted = None

    @property
    def pipeline_type(self):
        """Pipeline type."""
        return self._pipeline_type

    @property
    def mm_token_ids(self):
        """Multimodel placeholder token ids."""
        return self._mm_token_ids

    @mm_token_ids.setter
    def mm_token_ids(self, i):
        logger.debug(f'Append token id {i} to self._mm_token_ids')
        self._mm_token_ids.append(i)

    def _num_mm_token_ids(self):
        return len(self._mm_token_ids)

    def init_preprocessor(self):
        """Initialize the preprocessor."""
        if self.config.p is None:
            return
        self.preprocessor = self.preprocessor_class(**self.config.p.get())

    @abstractmethod
    def init_encoder(self):
        """Initialize the encoder."""

    @abstractmethod
    def init_projector(self):
        """Initialize the projector."""

    @abstractmethod
    def init_llm(self):
        """Initialize the LLM."""

    @abstractmethod
    def init_tail(self):
        """Initialize the tail."""

    def init_hooks(self):
        """Initialize the hooks."""
        from ..utils.const import (
            SUPPORTED_FORMATTEXT,
            SUPPORTED_GENMODS,
            SUPPORTED_GETEMBEDS,
            SUPPORTED_HOOKS,
            SUPPORTED_TOKENIZER_FUNCS,
        )

        # format_text hook
        if self.config.ft.name in SUPPORTED_FORMATTEXT:
            logger.debug(f'Init format_text hook using {self.config.ft.name}')
            hook_class = utils.resolve_hook_class(self.config.ft.name, special='format_text')
            self.format_text = hook_class(self.config.ft)
        else:
            logger.error(f'Invalid format_text hook name: {self.config.ft.name}', err=NotImplementedError)

        # pre_preprocessor_hook
        if self.config.ph.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-preprocessor hook using {self.config.ph.name}')
            hook_class = utils.resolve_hook_class(self.config.ph.name)
            self.pre_preprocessor_hook = hook_class(self.config.ph)
        else:
            logger.error(f'Invalid pre_preprocessor_hook hook name: {self.config.ph.name}', err=NotImplementedError)

        # pre_tokenizer_hook
        if self.config.th.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-tokenizer hook using {self.config.th.name}')
            hook_class = utils.resolve_hook_class(self.config.th.name)
            self.pre_tokenizer_hook = hook_class(self.config.th)
        else:
            logger.error(f'Invalid pre_tokenizer_hook hook name: {self.config.th.name}', err=NotImplementedError)

        # tokenizer_func_hook
        if self.config.tf.name in SUPPORTED_TOKENIZER_FUNCS:
            logger.debug(f'Init tokenizer function hook using {self.config.tf.name}')
            hook_class = utils.resolve_hook_class(self.config.tf.name, special='tokenizer_func')
            self.tokenizer_func = hook_class(self.config.tf)
        else:
            logger.error(f'Invalid tokenizer_func_hook hook name: {self.config.tf.name}', err=NotImplementedError)

        # pre_encoder_hook
        if self.config.eh.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-encoder hook using {self.config.eh.name}')
            hook_class = utils.resolve_hook_class(self.config.eh.name)
            self.pre_encoder_hook = hook_class(self.config.eh)
        else:
            logger.error(f'Invalid pre_encoder_hook hook name: {self.config.eh.name}', err=NotImplementedError)

        # pre_projector_hook
        # Special case, always use passthrough for QuantizedPipeline as pre_projector_hook should be in encoder.
        if self.config.p2h.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-projector hook using {self.config.p2h.name}')
            hook_class = utils.resolve_hook_class(self.config.p2h.name)
            self.pre_projector_hook = hook_class(self.config.p2h)
        else:
            logger.error(f'Invalid pre_projector_hook hook name: {self.config.p2h.name}', err=NotImplementedError)

        # pre_getembed_hook
        if self.config.gh.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-getembed hook using {self.config.gh.name}')
            hook_class = utils.resolve_hook_class(self.config.gh.name)
            self.pre_getembed_hook = hook_class(self.config.gh)
        else:
            logger.error(f'Invalid pre_getembed_hook hook name: {self.config.gh.name}', err=NotImplementedError)

        # get_embeds_hook
        if self.config.g.name in SUPPORTED_GETEMBEDS:
            logger.debug(f'Init get_embeds using {self.config.g.name}')
            hook_class = utils.resolve_hook_class(self.config.g.name, special='get_embeds')
            self.get_embeds = hook_class(
                self.config.g, text_embedding_layer=self.text_embedding_layer, dtype=self.dtype
            )
        else:
            logger.error(f'Invalid get_embeds_hook hook name: {self.config.g.name}', err=NotImplementedError)

        # pre_llm_hook
        if self.config.lh.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-llm hook using {self.config.lh.name}')
            hook_class = utils.resolve_hook_class(self.config.lh.name)
            self.pre_llm_hook = hook_class(self.config.lh)
        else:
            logger.error(f'Invalid pre_llm_hook hook name: {self.config.lh.name}', err=NotImplementedError)

        # self.pre_tail_hook
        if self.config.t2h.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-tail hook using {self.config.t2h.name}')
            hook_class = utils.resolve_hook_class(self.config.t2h.name)
            self.pre_tail_hook = hook_class(self.config.t2h)
        else:
            logger.error(f'Invalid pre_tail_hook hook name: {self.config.t2h.name}', err=NotImplementedError)

        # generation_modifiers
        if self.config.clp.name in SUPPORTED_GENMODS:
            logger.debug(f'Init generation modifiers using {self.config.clp.name}')
            hook_class = utils.resolve_hook_class(self.config.clp.name, special='generation_modifiers')
            self.clp = hook_class(self.config.clp)
        else:
            logger.error(f'Invalid generation_modifiers hook name: {self.config.clp.name}', err=NotImplementedError)

    def init_mm_qalft_hooks(self):
        """Initialize necessary hooks for mm qalft."""
        from ..utils.const import SUPPORTED_FORMATTEXT, SUPPORTED_GETEMBEDS, SUPPORTED_HOOKS

        if self.text_embedding_layer is None:
            self.text_embedding_layer = utils.get_embedding_layer(self.config.l, state_dict=self.state_dict['llm']).to(
                self.main_device
            )
        # pre_preprocessor_hook
        if self.config.ph.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-preprocessor hook using {self.config.ph.name}')
            hook_class = utils.resolve_hook_class(self.config.ph.name)
            self.pre_preprocessor_hook = hook_class(self.config.ph)
        else:
            logger.error(f'Invalid pre_preprocessor_hook hook name: {self.config.ph.name}', err=NotImplementedError)
        # format_text hook
        if self.config.ft.name in SUPPORTED_FORMATTEXT:
            logger.debug(f'Init format_text hook using {self.config.ft.name}')
            hook_class = utils.resolve_hook_class(self.config.ft.name, special='format_text')
            self.format_text = hook_class(self.config.ft)
        else:
            logger.error(f'Invalid format_text hook name: {self.config.ft.name}', err=NotImplementedError)
        # pre_getembed_hook
        if self.config.gh.name in SUPPORTED_HOOKS:
            logger.debug(f'Init pre-getembed hook using {self.config.gh.name}')
            hook_class = utils.resolve_hook_class(self.config.gh.name)
            self.pre_getembed_hook = hook_class(self.config.gh)
        else:
            logger.error(f'Invalid pre_getembed_hook hook name: {self.config.gh.name}', err=NotImplementedError)
        # get_embeds_hook
        if self.config.g.name in SUPPORTED_GETEMBEDS:
            logger.debug(f'Init get_embeds using {self.config.g.name}')
            hook_class = utils.resolve_hook_class(self.config.g.name, special='get_embeds')
            self.get_embeds = hook_class(
                self.config.g, text_embedding_layer=self.text_embedding_layer, dtype=self.dtype
            )
        else:
            logger.error(f'Invalid get_embeds_hook hook name: {self.config.g.name}', err=NotImplementedError)

    def init_evictor(self, config: BaseConfig, **kwargs):
        """Init Evictor for long context inference.

        Args:
            config (BaseConfig): The model's configuration to get model architecture information.
            kwargs: Additional arguments for initializing the evictor.
        """
        self.evictor = None
        if self.evict_class is not None and (self.task == 'inference' or self.task == 'evaluate'):
            kwargs['llm_layers_per_chunk'] = self.llm_layers_per_chunk.copy()
            self.evictor = self.evict_class(config, **kwargs)

    def has_preprocessor(self):
        """Check if the preprocessor is present.

        Returns:
            bool: True if the preprocessor is present, False otherwise.
        """
        return self.config.p is not None

    def has_encoder(self):
        """Check if the encoder is present.

        Returns:
            bool: True if the encoder is present, False otherwise.
        """
        return self.config.e is not None

    def has_projector(self):
        """Check if the projector is present.

        Returns:
            bool: True if the projector is present, False otherwise.
        """
        return self.config.p2 is not None

    def has_custom_tail(self):
        """Check if a custom tail is present.

        Returns:
            bool: True if a custom tail is present, False otherwise.
        """
        from ..utils import const

        if self.config.t is None:
            return False
        return self.config.t.model_type in const.SUPPORTED_CUSTOM_TAILS

    def load_multimodal_input(self, mm_path, **kwargs):
        """Load multimodal input from given filepath.

        Args:
            mm_path: The input filepath to be loaded.
            kwargs (dict): Additional keyword arguments.
        """
        from datasets import Audio, Dataset
        from PIL import Image

        if mm_path.endswith(('jpg', 'png', 'jpeg')):
            logger.debug(f'Loading image from path: {mm_path}')
            return Image.open(mm_path), kwargs
        if mm_path.endswith('mp3'):  # placeholder audio extension
            logger.debug(f'Loading audio from path: {mm_path}')
            return Dataset.from_dict({'audio': [str(mm_path)]}).cast_column('audio', Audio(sampling_rate=16000))[0][
                'audio'
            ]['array'], kwargs
        if mm_path.endswith('.npy'):
            logger.debug(f'Loading mm input numpy from path: {mm_path}')
            return np.load(mm_path), kwargs
        logger.error(
            'Expect multimodal input filepath to end with either image (jpg, png) or audio (mp3) extensions, but got'
            f'{mm_path}'
        )
        raise

    def forward_pre_preprocessor_hook(self, inputs, **kwargs):
        """Forward pass through the pre_preprocessor hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_preprocessor_hook')
        return self.pre_preprocessor_hook.forward(inputs, **kwargs)

    def forward_preprocessor(self, inputs, **kwargs):
        """Forward pass through the preprocessor.

        Args:
            inputs: The inputs for the preprocessor.
            kwargs: Additional keyword arguments.
        """
        logger.debug('Enter forward_preprocessor')
        if self.config.p is None:
            return inputs, kwargs

        kwargs.update(self.config.p.kwargs)
        outputs, kwargs = self.preprocessor.preprocess(inputs, **kwargs)
        if all(x.shape == outputs['input_features'][0].shape for x in outputs['input_features']):
            outputs['input_features'] = np.stack(outputs['input_features'], axis=0)
        return outputs, kwargs

    def forward_pre_tokenizer_hook(self, inputs, **kwargs):
        """Forward pass through the pre_tokenizer hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_tokenizer_hook')
        return self.pre_tokenizer_hook.forward(inputs, **kwargs)

    def forward_tokenizer(self, prompt, preformatter, **kwargs):
        """Forward pass through the tokenizer.

        Args:
            prompt (str): input prompt or input token.
            preformatter (Preformatter): preformatter.
            kwargs (dict): Additional keyword arguments.
        """
        logger.debug('Forward tokenizer')
        return self.tokenizer_func(self, prompt, preformatter, **kwargs)

    def forward_pre_encoder_hook(self, inputs, **kwargs):
        """Forward pass through the pre_encoder hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_encoder_hook')
        return self.pre_encoder_hook.forward(inputs, **kwargs)

    @abstractmethod
    def forward_encoder(self, *inputs, **kwargs):
        """Forward pass through the encoder.

        Args:
            inputs: The inputs to be encoded.
            kwargs (dict): Additional keyword arguments.
        """

    @abstractmethod
    def forward_projector(self, *inputs, **kwargs):
        """Forward pass through the projector.

        Args:
            inputs: The inputs to be processed.
            kwargs (dict): Additional keyword arguments.
        """

    def forward_pre_projector_hook(self, inputs, **kwargs):
        """Forward pass through the pre_projector hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_projector_hook')
        return self.pre_projector_hook.forward(inputs, **kwargs)

    def forward_pre_getembed_hook(self, inputs, **kwargs):
        """Forward pass through the pre_getembed hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_getembed_hook')
        return self.pre_getembed_hook.forward(inputs, **kwargs)

    def forward_pre_llm_hook(self, inputs, **kwargs):
        """Forward pass through the pre_llm hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_llm_hook')
        return self.pre_llm_hook.forward(inputs, **kwargs)

    @abstractmethod
    def generate_llm(
        self,
        input_embeds,
        input_ids,
        num_token,
        cache_size,
        stopping_criteria,
        logits_processors,
        logits_warper,
        **kwargs,
    ):
        """Generates LLM response tokens.

        Args:
            input_embeds: The input embeddings.
            input_ids: The input IDs.
            num_token: Maximum number of tokens to forward at once during prompt mode.
            cache_size: The cache size.
            stopping_criteria: The stopping criteria.
            logits_processors: The logits processors.
            logits_warper: The logits warper.
            kwargs (optional): Additional keyword arguments.

        Returns:
            numpy.ndarray: Generated token ids concatenated behind input token ids.
        """

    def forward_pre_tail_hook(self, inputs, **kwargs):
        """Forward pass through the pre_tail hook.

        Args:
            inputs (any): The inputs to be processed.
            kwargs (dict): Additional keyword arguments.

        Returns:
            any: The processed inputs.
        """
        logger.debug('Forward forward_pre_tail_hook')
        return self.pre_tail_hook.forward(inputs, **kwargs)

    @abstractmethod
    def forward_tail(self, *inputs, **kwargs):
        """Forward pass through the tail.

        Args:
            inputs: The inputs to be processed.
            kwargs (dict): Additional keyword arguments.
        """

    @abstractmethod
    def deduce_num_layers_per_chunk(self):
        """Deduce the number of blocks per chunk."""

    def format_prompt(self, prompt, **kwargs):
        """Format the prompt based on input_mode.

        For text, don't format unless specifically overriden by encoder/llm.
        For tokens, convert a string array into a numpy array.
        For embeddings, load numpy file path.
        For in-built datasets, prompts should already be formatted, so do nothing.

        Args:
            prompt (str): The prompt to be formatted.
            kwargs: Other kwargs.

        Returns:
            str or np.ndarray: The formatted prompt. str if text, np.ndarray otherwise.
        """
        if self.input_mode == 'text':
            return self.format_text(prompt, **kwargs)
        if self.input_mode == 'tokens':
            return utils.tokenized_text_to_array(prompt), kwargs
        if self.input_mode == 'embeddings':
            return np.load(prompt), kwargs
        return prompt, kwargs

    def set_precision_config(self, precision_config):
        """Set the precision configuration for pipeline."""
        logger.debug(f'Setting pipeline precision config attribute using {type(precision_config)}')
        if not isinstance(precision_config, PTQPrecisionConfig):
            logger.error(f'Only can set PTQPrecisionConfig for precision config, but got {type(precision_config)}')
        self.precision_config = precision_config

    def nhwc_to_nchw(self, inp):
        """Converts NHWC images to NCHW format."""
        from transformers.feature_extraction_utils import BatchFeature

        batch_feature = isinstance(inp, BatchFeature)

        images = inp['input_features'] if batch_feature else inp

        if isinstance(images, (list, tuple)):
            inputs_nchw = []
            for i, inp in enumerate(images):
                if isinstance(inp, torch.Tensor) and inp.ndim == 4:
                    inputs_nchw.append(images[i].permute(0, 3, 1, 2))
                    continue
                if isinstance(inp, np.ndarray) and inp.ndim == 4:
                    inputs_nchw.append(images[i].transpose(0, 3, 1, 2))
                    continue
                inputs_nchw.append(images[i])
            if batch_feature:
                inp['input_features'] = inputs_nchw
                return inp
            return inputs_nchw
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            if batch_feature:
                inp['input_features'] = images.permute(0, 3, 1, 2)
                return inp
            return images.permute(0, 3, 1, 2)
        if isinstance(images, np.ndarray) and images.ndim == 4:
            if batch_feature:
                inp['input_features'] = images.transpose(0, 3, 1, 2)
                return inp
            return images.transpose(0, 3, 1, 2)

        if batch_feature:
            inp['input_features'] = images
            return inp
        return images

    def get_tokenizer_add_bos(self):
        """Gets whether the pipeline's tokenizer prepends BOS token or not.

        Returns:
            bool: whether the pipeline's tokenizer prepends BOS token or not.
        """
        if self.tokenizer is None:
            logger.debug('No tokenizer to get `add_bos` attribute from. Returning None.')
            return None

        if isinstance(self.tokenizer, SentencePieceProcessor):
            logger.debug(f"Getting tokenizer's `add_bos` attribute: {self.tokenizer._add_bos}.")  # noqa: SLF001
            return self.tokenizer._add_bos  # noqa: SLF001

        logger.debug(f"Getting tokenizer's `add_bos` attribute: {self.tokenizer.add_bos_token}.")
        return self.tokenizer.add_bos_token

    def set_tokenizer_add_bos(self, add_bos):
        """Sets whether the pipeline's tokenizer prepends BOS token or not.

        Args:
            add_bos: boolean to set if pipeline's tokenizer prepends BOS token or not.
        """
        logger.debug(f"Setting tokenizer's `add_bos` attribute to {add_bos}.")
        if self.tokenizer is None:
            logger.debug('Tokenizer does not exist. Skipping.')
            return

        if isinstance(self.tokenizer, SentencePieceProcessor):
            self.tokenizer._add_bos = add_bos  # noqa: SLF001
        else:
            self.tokenizer.add_bos_token = add_bos

    def generate_llm_with_memory(
        self,
        input_embeds,
        input_ids,
        num_token,
        cache_size,
        stopping_criteria,
        logits_processors=None,
        logits_warper=None,
        cross_attn=None,
        **kwargs,
    ):
        """LLM Generation with long term memory capability."""
        from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

        input_length = input_embeds.shape[1]

        # dynamic shape is not supported on infini attention
        kwargs['dynamic_shape'] = False

        window_size = self.config.l.infini_window_size
        sink_size = self.config.l.infini_sink_size
        assert window_size >= 0 and sink_size >= 0, 'sink size and window size need to be >= 0'

        infini_max_gen_length = self.config.l.infini_max_gen_length
        assert infini_max_gen_length >= 0, 'infini max gen length should be >= 0, 0 means no max gen length'

        if self.task == 'evaluate':
            # for quantized pipeline
            if hasattr(self, 'cache_size'):
                segment_cache_size = self.cache_size - sink_size - window_size
                num_token = self.prompt_num_tokens
            # for float pipeline
            else:
                segment_cache_size = self.config.l.infini_segment_size
                num_token = segment_cache_size
        elif self.task == 'make_calibration':
            # Note: Make calibration flow similar to inference pipeline,
            # but num_token should be smaller than segment size
            # otherwise, there won't be any kv cache in calib dataset as they are being reset
            if hasattr(self, 'cache_size'):  # for quantized pipeline
                logger.error('Make calibration flow is only implemented in float pipeline.')
                raise
            segment_cache_size = self.config.l.infini_segment_size
            num_token = min(512, self.config.l.infini_segment_size // 3)
        else:
            if hasattr(self, 'cache_size'):
                segment_cache_size = self.cache_size - sink_size - window_size
                num_token = self.prompt_num_tokens
            else:
                segment_cache_size = self.config.l.infini_segment_size
                num_token = segment_cache_size if num_token is None else num_token

        # segment size check for quantized
        assert segment_cache_size == self.config.l.infini_segment_size, (
            f'{segment_cache_size} != {self.config.l.infini_segment_size}'
        )

        if cache_size is not None and cache_size != segment_cache_size:
            logger.info(
                f'Overwriting user defined cache size {cache_size} '
                f'to infini transformer segment size {segment_cache_size}'
            )

        # handle return logits functionality
        return_logits = False
        if kwargs.get('return_logits'):
            return_logits = True
            if isinstance(self.dtype, torch.dtype):
                all_logits = torch.zeros((1, 0, self.config.l.vocab_size), dtype=self.dtype).numpy()
            else:
                all_logits = np.zeros((1, 0, self.config.l.vocab_size), dtype=self.dtype)

        prompt_infini_cache_size = segment_cache_size + sink_size + window_size
        gen_infini_cache_size = segment_cache_size + sink_size + window_size + infini_max_gen_length
        islocal = True  # means position embedding start from 0 for each segment
        memory = Memory(
            self.config.l,
            dtype=self.dtype,
            prompt_cache_size=prompt_infini_cache_size,
            gen_cache_size=gen_infini_cache_size,
            is_local=islocal,
            device=self.main_device,
        )

        # get the max generated tokens
        max_length_criteria = [
            stopping_criteria[i]
            for i in range(len(stopping_criteria))
            if isinstance(stopping_criteria[i], MaxLengthCriteria)
        ]
        if len(max_length_criteria) > 0:
            max_length = max_length_criteria[0].max_length
        else:
            logger.debug('Using max position embeddings as max length.')
            max_length = self.config.l.max_position_embeddings

        max_generated_tokens = max_length - input_length

        if infini_max_gen_length > 0 and max_generated_tokens > infini_max_gen_length:
            logger.info(f'Overwriting max generated tokens using infini_max_gen_length: {infini_max_gen_length}')
            max_generated_tokens = infini_max_gen_length

        if islocal:
            num_full_segments = max(0, (input_length - sink_size - window_size) // segment_cache_size)
            leftover = input_length - num_full_segments * segment_cache_size
            if num_full_segments > 0:
                leftover -= sink_size
            logger.info(
                f'Infini Attention Segment size: {segment_cache_size}, prompt cache size: {prompt_infini_cache_size}, '
                f'gen cache size: {gen_infini_cache_size}'
            )
            logger.info(
                f'Infini Attention Input length: {input_length}, '
                f'Number of full segments: {num_full_segments}, leftover: {leftover}'
            )
            if sink_size > 0:
                sink_ids = input_ids[:, :sink_size]
                sink_embeds = input_embeds[:, :sink_size, :]

            # Part 1: Prompt processing -- handle all full segments
            if num_full_segments > 0:
                cur_cache_size = prompt_infini_cache_size

                for i in range(num_full_segments):
                    cur_input_embeds = input_embeds[
                        :, sink_size + i * segment_cache_size : sink_size + (i + 1) * segment_cache_size, :
                    ]
                    cur_input_ids = input_ids[
                        :, sink_size + i * segment_cache_size : sink_size + (i + 1) * segment_cache_size
                    ]

                    max_length = segment_cache_size + 1

                    if i == 0 and sink_size > 0:
                        # if there is sink, processed them in the first segment
                        if isinstance(self.dtype, torch.dtype):
                            cur_input_embeds = torch.concatenate([sink_embeds, cur_input_embeds], dim=1)
                        else:
                            cur_input_embeds = np.concatenate([sink_embeds, cur_input_embeds], axis=1)
                        cur_input_ids = np.concatenate([sink_ids, cur_input_ids], axis=1)

                        max_length += sink_size

                    # split the input into segments
                    # since it is prompt processing produce one token only
                    cur_stopping_criteria = StoppingCriteriaList()
                    cur_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))

                    memory.set_prompt_full_segment_mode(True)
                    outputs = self.generate_llm(
                        cur_input_embeds,
                        cur_input_ids,
                        num_token,
                        cur_cache_size,
                        cur_stopping_criteria,
                        logits_processors,
                        logits_warper,
                        cross_attn,
                        memory=memory,
                        **kwargs,
                    )
                    if return_logits:
                        logits = outputs
                        if isinstance(logits, torch.Tensor):
                            logits = logits.detach().cpu().numpy()
                        all_logits = np.concatenate([all_logits, logits], axis=1)

                if leftover == 0 and max_generated_tokens == 1:
                    # done generation
                    if return_logits:
                        return all_logits
                    llm_output, kwargs = outputs
                    llm_output = np.concatenate([input_ids, llm_output[:, cur_input_embeds.shape[1] :]], axis=-1)

                    return llm_output, kwargs

            # Part 2: Prompt processing -- handle leftover and Generation
            if leftover > 0:
                cur_input_embeds = input_embeds[:, -leftover:, :]
                cur_input_ids = input_ids[:, -leftover:]
                cur_cache_size = prompt_infini_cache_size
            else:
                # if no leftover, use the first generated logits
                cur_input_ids = outputs[0][:, -1:]
                cur_input_embeds = self.text_embedding_layer(torch.tensor(cur_input_ids).to(self.main_device))
                if not isinstance(self.dtype, torch.dtype):
                    cur_input_embeds = cur_input_embeds.detach().cpu().numpy()

            cur_stopping_criteria = StoppingCriteriaList()
            cur_stopping_criteria.append(MaxLengthCriteria(max_length=cur_input_embeds.shape[1] + max_generated_tokens))

            if return_logits:
                # need to set again as it may be popped in the previous generate_llm
                kwargs['return_logits'] = True

            # let memory knows that full segment prompt processing has ended
            memory.set_prompt_full_segment_mode(False)
            outputs = self.generate_llm(
                cur_input_embeds,
                cur_input_ids,
                num_token,
                cur_cache_size,
                cur_stopping_criteria,
                logits_processors,
                logits_warper,
                cross_attn,
                memory=memory,
                **kwargs,
            )
            if return_logits:
                logits = outputs
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu().numpy()
                return np.concatenate([all_logits, logits], axis=1)

            llm_output, kwargs = outputs
            llm_output = np.concatenate([input_ids, llm_output[:, cur_input_embeds.shape[1] :]], axis=-1)

            return llm_output, kwargs

        logger.error('Global version of infini attention not yet supported.')
        return None

    @torch.no_grad()
    def process_multimodal_input(self, multimodal_paths, **kwargs):
        """Process multimodal input."""
        multimodal_embeds = []
        cross_attentions = []

        for path in multimodal_paths:
            mm_input, kwargs = self.load_multimodal_input(path, **kwargs)

            mm_input, kwargs = self.forward_pre_preprocessor_hook(mm_input, **kwargs)
            processed, kwargs = self.forward_preprocessor(mm_input, **kwargs)

            pre_encoder, kwargs = self.forward_pre_encoder_hook(processed, **kwargs)
            # pre_encoder = self.nhwc_to_nchw(pre_encoder)
            encoded, attn, kwargs = self.forward_encoder(pre_encoder, **kwargs)

            pre_proj, kwargs = self.forward_pre_projector_hook(encoded, **kwargs)
            projected, kwargs = self.forward_projector(pre_proj, **kwargs)

            multimodal_embeds.append(projected)
            if attn is not None:
                cross_attentions.append(attn)

        return multimodal_embeds, cross_attentions, kwargs

    @torch.no_grad()
    def get_encoder_output_embedding(self, prompt=None, preformatter=None, multimodal_inputs=None, **kwargs):
        """Forward pass through the encoder related part of the pipeline."""
        logger.debug('In get_encoder_output_embedding.')
        if multimodal_inputs is None:
            return [], kwargs
        kwargs.update({'pipeline_type': self._pipeline_type})
        if len(multimodal_inputs) > 0:
            multimodal_embeds = []
            mm_file_name = []
            for mm_input_path in multimodal_inputs:
                # Assume there can be more than 1 multimodal inputs per prompt,
                # For example, 2 images, or 2 audio inputs.
                # Mixed multimodality is currently not supported.
                kwargs['dupe'] = mm_input_path in self.encoder_inputs
                if self.task == 'make_calibration':
                    self.encoder_inputs.add(mm_input_path)
                mm_input, kwargs = self.load_multimodal_input(mm_input_path, **kwargs)
                mm_input, kwargs = self.forward_pre_preprocessor_hook(mm_input, **kwargs)
                mm_input_processed, kwargs = self.forward_preprocessor(mm_input, **kwargs)

                # From this point onwards until self.forward_encoder, will use BatchFeature
                logger.debug(f'After forward_preprocessor shape: {mm_input_processed["input_features"].shape}')
                mm_input_processed_pre_tokenizer, kwargs = self.forward_pre_tokenizer_hook(mm_input_processed, **kwargs)
                logger.debug(
                    f'After pre_tokenizer_hook shape: {mm_input_processed_pre_tokenizer["input_features"].shape}'
                )
                logger.debug(f'kwargs before forward_pre_encoder_hook: {kwargs}')
                mm_input_preencoder, kwargs = self.forward_pre_encoder_hook(mm_input_processed_pre_tokenizer, **kwargs)
                logger.debug(f'kwargs after forward_pre_encoder_hook: {kwargs}')
                logger.debug(f'After pre_encoder_hook shape: {mm_input_preencoder["input_features"].shape}')
                mm_input_encoded, _cross_attn, kwargs = self.forward_encoder(mm_input_preencoder, **kwargs)
                logger.debug(f'After forward_encoder shape: {mm_input_encoded.shape}')

                # Not using BatchFeature anymore
                mm_input_preprojector, kwargs = self.forward_pre_projector_hook(mm_input_encoded, **kwargs)
                if len(mm_input_preprojector) > 1:
                    logger.debug(f'After pre_projector_hook shape: {mm_input_preprojector[0].shape}')
                else:
                    logger.debug(f'After pre_projector_hook shape: {mm_input_preprojector.shape}')

                mm_input_projected, kwargs = self.forward_projector(mm_input_preprojector, **kwargs)
                if len(mm_input_projected) > 1:
                    logger.debug(f'After forward_projector shape: {mm_input_projected[0].shape}')
                else:
                    logger.debug(f'After forward_projector shape: {mm_input_projected.shape}')

                multimodal_embeds.append(mm_input_projected)
                mm_file_name.append(Path(mm_input_path).stem)

            # Forward more to get necessary kwargs
            if prompt is not None and preformatter is not None:
                prompt, kwargs = self.format_prompt(prompt, **kwargs)
                prompt, kwargs = self.forward_pre_tokenizer_hook(prompt, **kwargs)
                kwargs['quiet'] = self.task == 'make_calibration' or kwargs.get('quiet', False)
                text_tokens, kwargs = (
                    (prompt, kwargs)
                    if self.input_mode == 'tokens'
                    else self.forward_tokenizer(
                        prompt,
                        preformatter,
                        mm_path=multimodal_inputs,
                        **kwargs,
                    )
                )
                _, kwargs = self.forward_pre_getembed_hook(text_tokens, **kwargs)

        return multimodal_embeds, mm_file_name, kwargs

    @torch.no_grad()
    def forward(
        self,
        prompt=None,
        streaming_prompt_max_len=0,
        preformatter=None,
        multimodal_inputs=None,
        repetition_penalty=1.0,
        max_new_tokens=128,
        num_token=None,
        cache_size=None,
        temperature=None,
        top_p=None,
        top_k=None,
        additional_return_dict=None,
        **kwargs,
    ):
        """Forward pass through the pipeline.

        Args:
            prompt (str, optional): The input prompt string. Defaults to None.
            streaming_prompt_max_len (int, optional): Maximum number of prompt tokens to forward as one prompt.
                Chunks prompt into multiple chunks if prompts exceed this length.
                Defaults to 0 (forward whole prompt as one prompt).
            preformatter (Preformatter, optional): The LLM preformatter.
            img_path (str, optional): The path to image from jsonl.
            multimodal_inputs (list, optional): The multimodal input file paths. Defaults to None.
            custom_embeds (list, optional): The custom embeddings. Defaults to None.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            max_new_tokens (int, optional): The maximum number of new tokens. Defaults to 128.
            num_token (int, optional): The number of prompt tokens. Default depends on task.
            cache_size (int, optional): The cache size. Default depends on task.
            temperature (float, optional): The temperature for sampling. Defaults to None.
            top_p (float, optional): The top-p value for sampling. Defaults to None.
            top_k (int, optional): The top-k value for sampling. Defaults to None.
            additional_return_dict (dict, optional): The variable to be additionally returned. Default to None.
            kwargs (optional): Additional keyword arguments.

        Returns:
            any: The output of the forward pass through the LLM.
        """
        from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
        from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

        from ..utils import generate_utils
        from ..utils.preformatter import Preformatter

        logger.debug('Enter pipeline forward')

        if multimodal_inputs is None:
            multimodal_inputs = []
        if preformatter is None:
            preformatter = Preformatter(None)
        if self.input_mode == 'embeddings' and len(multimodal_inputs) > 0:
            logger.error('`multimodal_inputs` must be empty list if input_mode is `embeddings`.')

        stopping_criteria = StoppingCriteriaList()
        logits_processors = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        logits_processors.append(self.clp)

        logits_warper = generate_utils.get_sample_logits_warper(temperature, top_p, top_k)

        kwargs.update({'pipeline_type': self._pipeline_type})
        logger.debug(
            f'prompt={prompt}, input_mode={self.input_mode}, multimodal_inputs={multimodal_inputs}, kwargs={kwargs}'
        )

        cross_attns = []
        if self.input_mode == 'embeddings':
            prompt, kwargs = self.format_prompt(prompt, **kwargs)
            input_embeds, kwargs = self.get_embeds.forward(custom_embeds=prompt, **kwargs)

            llm_input, kwargs = self.forward_pre_llm_hook(input_embeds, **kwargs)

            self.input_length = llm_input.shape[1]

            stopping_criteria.append(MaxLengthCriteria(max_length=llm_input.shape[1] + max_new_tokens))

            text_tokens = np.zeros(llm_input.shape[:2], dtype=np.int64)

            if self.task == 'make_calibration':
                if self.backend == const.MLKITS:
                    logger.debug('Save calibration data with fixed token/cache size for MLKits.')
                    cache_size = 1
                else:
                    num_token = self.input_length - 1
                    cache_size = self.input_length + 1
                    kwargs['dynamic_shape'] = True

            elif self.task == 'find_overture':
                num_token = llm_input.shape[1]
                cache_size = llm_input.shape[1] + max_new_tokens
                kwargs['dynamic_shape'] = True

            # prevent overwriting of cache_size during inference torch
            elif self.task == 'inference' and self._pipeline_type == 'float' and self.config.l.extra_input['sink_rope']:
                if cache_size is None:
                    cache_size = llm_input.shape[1] + max_new_tokens
                if num_token is None:
                    num_token = llm_input.shape[1]

            if self.config.l.infini_attention:
                llm_output, kwargs = self.generate_llm_with_memory(
                    llm_input,
                    text_tokens,
                    num_token,
                    cache_size,
                    stopping_criteria,
                    logits_processors,
                    logits_warper,
                    **kwargs,
                )
            else:
                llm_output, kwargs = self.generate_llm(
                    llm_input,
                    text_tokens,
                    num_token,
                    cache_size,
                    stopping_criteria,
                    logits_processors,
                    logits_warper,
                    **kwargs,
                )
        else:
            multimodal_embeds = None
            if len(multimodal_inputs) > 0:
                if streaming_prompt_max_len > 0:
                    logger.error('Prompt chunking is not implemented for MLLM flow.', err=NotImplementedError)
                multimodal_embeds = []
                for mm_input_path in multimodal_inputs:
                    # Assume there can be more than 1 multimodal inputs per prompt,
                    # For example, 2 images, or 2 audio inputs.
                    # Mixed multimodality is currently not supported.
                    kwargs['dupe'] = mm_input_path in self.encoder_inputs
                    if self.task == 'make_calibration':
                        self.encoder_inputs.add(mm_input_path)
                    mm_input, kwargs = self.load_multimodal_input(mm_input_path, **kwargs)
                    mm_input, kwargs = self.forward_pre_preprocessor_hook(mm_input, **kwargs)
                    mm_input_processed, kwargs = self.forward_preprocessor(mm_input, **kwargs)

                    # From this point onwards until self.forward_encoder, will use BatchFeature
                    logger.debug(f'After forward_preprocessor shape: {mm_input_processed["input_features"].shape}')
                    mm_input_preencoder, kwargs = self.forward_pre_encoder_hook(mm_input_processed, **kwargs)
                    logger.debug(f'After pre_encoder_hook shape: {mm_input_preencoder["input_features"].shape}')
                    mm_input_encoded, cross_attn, kwargs = self.forward_encoder(mm_input_preencoder, **kwargs)
                    if len(mm_input_encoded) > 1:
                        logger.debug(f'After forward_encoder shape: {mm_input_encoded[0].shape}')
                    else:
                        logger.debug(f'After forward_encoder shape: {mm_input_encoded.shape}')

                    # Not using BatchFeature anymore
                    mm_input_preprojector, kwargs = self.forward_pre_projector_hook(mm_input_encoded, **kwargs)
                    if len(mm_input_preprojector) > 1:
                        logger.debug(f'After pre_projector_hook shape: {mm_input_preprojector[0].shape}')
                    else:
                        logger.debug(f'After pre_projector_hook shape: {mm_input_preprojector.shape}')

                    mm_input_projected, kwargs = self.forward_projector(mm_input_preprojector, **kwargs)
                    if len(mm_input_projected) > 1:
                        logger.debug(f'After forward_projector shape: {mm_input_projected[0].shape}')
                    else:
                        logger.debug(f'After forward_projector shape: {mm_input_projected.shape}')

                    multimodal_embeds.append(mm_input_projected)
                    if cross_attn is not None:
                        cross_attns.append(cross_attn)
                logger.debug(f'kwargs before format_prompt: {kwargs}')
                prompt, kwargs = self.format_prompt(prompt, **kwargs)
                prompt_to_print = kwargs.get('prompt_to_log', prompt)
                logger.debug(f'formatted prompt={prompt_to_print}')
                prompt, kwargs = self.forward_pre_tokenizer_hook(prompt, **kwargs)
                logger.debug(f'Prompt after pre_tokenizer_hook: {prompt}')
                kwargs['quiet'] = self.task == 'make_calibration' or kwargs.get('quiet', False)
                text_tokens, kwargs = (
                    (prompt, kwargs)
                    if self.input_mode == 'tokens'
                    else self.forward_tokenizer(
                        prompt,
                        preformatter,
                        mm_path=multimodal_inputs,
                        **kwargs,
                    )
                )
                logger.debug(f'Text Tokens: {text_tokens}')
            else:
                prompt, kwargs = self.format_prompt(prompt, **kwargs)

            if len(cross_attns) > 1:
                logger.error(f'Maximum of 1 audio input is supported for cross-attention, but found {len(cross_attns)}')
            elif len(cross_attns) == 0:
                cross_attns = [None]

            self.prompt_formatted = preformatter.generate_prompt(prompt)

            if streaming_prompt_max_len == 0:
                logger.debug('No prompt chunking')
                if multimodal_embeds is None:
                    kwargs['quiet'] = self.task == 'make_calibration' or kwargs.get('quiet', False)
                    text_tokens, kwargs = (
                        (prompt, kwargs)
                        if self.input_mode in ['tokens', *const.INBUILT_DATASETS]
                        else self.forward_tokenizer(
                            prompt,
                            preformatter,
                            **kwargs,
                        )
                    )
                if self.use_bita and self.task == 'make_calibration':
                    kwargs['add_bita_draft'] = True
                    kwargs['vocab_size'] = self.config.l.vocab_size

                text_tokens, kwargs = self.forward_pre_getembed_hook(text_tokens, **kwargs)
                self.input_length = text_tokens.shape[1]
                logger.info(f'Input Prompt Length: {self.input_length}')
                input_embeds, kwargs = self.get_embeds.forward(text_tokens, multimodal_embeds, **kwargs)
                # For special handling mllm overture cases which change positional embedding
                # according to multimodality inputs. For example, Qwen2-VL. For these cases,
                # each `pre_llm_hook` should change its implementation by considering overture size.
                # Typically, it can only consider size of overture rather than overture itself.
                if self.overture is not None:
                    kwargs['overture_size'] = self.overture.shape[-2]
                    kwargs['dummy_token_id'] = self.config.l.bos_token_id

                llm_input, kwargs = self.forward_pre_llm_hook(input_embeds, **kwargs)

                stopping_criteria.append(MaxLengthCriteria(max_length=llm_input.shape[1] + max_new_tokens))

                if self.task == 'make_calibration':
                    if self.backend == const.MLKITS:
                        cache_size = 1
                        logger.debug(
                            f'Save calibration data with fixed token/cache {num_token}/{cache_size} size for MLKits.'
                        )
                    else:
                        num_token = llm_input.shape[1] - 1
                        cache_size = llm_input.shape[1] + 1
                        kwargs['dynamic_shape'] = True

                elif self.task == 'find_overture':
                    num_token = llm_input.shape[1]
                    cache_size = llm_input.shape[1] + max_new_tokens
                    kwargs['dynamic_shape'] = True

                # prevent overwriting of cache_size during inference torch
                elif (
                    self.task == 'inference'
                    and self._pipeline_type == 'float'
                    and self.config.l.extra_input['sink_rope']
                ):
                    if cache_size is None:
                        cache_size = llm_input.shape[1] + max_new_tokens
                    if num_token is None:
                        num_token = llm_input.shape[1]

                if self.config.l.infini_attention:
                    llm_output, kwargs = self.generate_llm_with_memory(
                        llm_input,
                        text_tokens,
                        num_token,
                        cache_size,
                        stopping_criteria,
                        logits_processors,
                        logits_warper,
                        cross_attns[0],
                        **kwargs,
                    )
                else:
                    llm_output, kwargs = self.generate_llm(
                        llm_input,
                        text_tokens,
                        num_token,
                        cache_size,
                        stopping_criteria,
                        logits_processors,
                        logits_warper,
                        cross_attns[0],
                        **kwargs,
                    )
            else:
                if preformatter.used:
                    logger.debug('Prompt chunking with preformatter')
                    if self.input_mode != 'text':
                        logger.error(
                            'Prompt chunking with preformatter is only supported for text input.',
                            err=NotImplementedError,
                        )
                    sentences = prompt.split('\n')
                    curr_chunk = ''
                    prev_chunk = None
                    prev_chunk_tokens = None
                    prev_response = None
                    for i, sentence in enumerate(sentences):
                        kwargs['quiet'] = True
                        curr_chunk_tokens, kwargs = self.forward_tokenizer(
                            curr_chunk,
                            preformatter,
                            sub_response=prev_response,
                            **kwargs,
                        )
                        if curr_chunk_tokens.shape[1] < streaming_prompt_max_len:
                            prev_chunk = curr_chunk
                            prev_chunk_tokens = curr_chunk_tokens
                            curr_chunk += sentence + '\n'
                            if i == len(sentences) - 1:
                                prev_chunk = curr_chunk
                            else:
                                continue
                        if prev_chunk_tokens is None:
                            logger.error(
                                f'Length of a single line ({curr_chunk_tokens.shape[1]}) is more than maximum '
                                f'length to chunk text to ({streaming_prompt_max_len}).'
                            )

                        kwargs['quiet'] = self.task == 'make_calibration' or kwargs.get('quiet', False)
                        text_tokens, kwargs = self.forward_tokenizer(
                            prev_chunk,
                            preformatter,
                            sub_response=prev_response,
                            **kwargs,
                        )
                        text_tokens, kwargs = self.forward_pre_getembed_hook(text_tokens, **kwargs)
                        self.input_length = text_tokens.shape[1]
                        # Don't expect MLLM inputs for prompt chunking
                        input_embeds, kwargs = self.get_embeds.forward(text_tokens, **kwargs)

                        llm_input, kwargs = self.forward_pre_llm_hook(input_embeds, **kwargs)

                        stopping_criteria.append(MaxLengthCriteria(max_length=llm_input.shape[1] + max_new_tokens))

                        if self.task == 'make_calibration':
                            if self.backend == const.MLKITS:
                                logger.debug('Save calibration data with fixed token/cache size for MLKits.')
                                cache_size = 1
                            else:
                                kwargs['dynamic_shape'] = True
                        else:
                            num_token = streaming_prompt_max_len
                            cache_size = llm_input.shape[1] + max_new_tokens

                        llm_output, kwargs = self.generate_llm(
                            llm_input,
                            text_tokens,
                            num_token,
                            cache_size,
                            stopping_criteria,
                            logits_processors,
                            logits_warper,
                            cross_attns[0],
                            **kwargs,
                        )

                        response = self.get_response(llm_output, preformatter, quiet=True)
                        prev_response = response
                        curr_chunk = ''
                        prev_chunk = None
                        prev_chunk_tokens = None
                else:
                    logger.debug('Standard prompt chunking')
                    kwargs['quiet'] = self.task == 'make_calibration' or kwargs.get('quiet', False)
                    text_tokens, kwargs = (
                        (prompt, kwargs)
                        if self.input_mode == 'tokens'
                        else self.forward_tokenizer(
                            prompt,
                            preformatter,
                            **kwargs,
                        )
                    )
                    text_tokens, kwargs = self.forward_pre_getembed_hook(text_tokens, **kwargs)
                    self.input_length = text_tokens.shape[1]
                    input_embeds, kwargs = self.get_embeds.forward(text_tokens, **kwargs)

                    llm_input, kwargs = self.forward_pre_llm_hook(input_embeds, **kwargs)

                    if self.input_mode in const.INBUILT_DATASETS:
                        # If in-built dataset, don't use cache
                        num_passes = int(llm_input.shape[1] // streaming_prompt_max_len) + int(
                            llm_input.shape[1] % streaming_prompt_max_len > 0
                        )
                        for i in range(num_passes):
                            stopping_criteria.append(MaxLengthCriteria(max_length=1))

                            if self.task == 'make_calibration':
                                if self.backend == const.MLKITS:
                                    logger.debug('Save calibration data with fixed token/cache size for MLKits.')
                                    cache_size = 1
                                else:
                                    kwargs['dynamic_shape'] = True
                            else:
                                num_token = streaming_prompt_max_len
                                cache_size = streaming_prompt_max_len + 1

                            llm_output, kwargs = self.generate_llm(
                                llm_input[
                                    :,
                                    i * num_token : min(llm_input.shape[1] - 1, (i + 1) * num_token),
                                ],
                                text_tokens[
                                    :,
                                    i * num_token : min(llm_input.shape[1] - 1, (i + 1) * num_token),
                                ],
                                num_token,
                                cache_size,
                                stopping_criteria,
                                logits_processors,
                                logits_warper,
                                cross_attns[0],
                                **kwargs,
                            )
                    else:
                        stopping_criteria.append(MaxLengthCriteria(max_length=llm_input.shape[1] + max_new_tokens))

                        if self.task == 'make_calibration' and self.backend == const.MLKITS:
                            logger.debug('Save calibration data with fixed token/cache size for MLKits.')
                            cache_size = 1
                        else:
                            num_token = streaming_prompt_max_len
                            cache_size = llm_input.shape[1] + max_new_tokens

                        llm_output, kwargs = self.generate_llm(
                            llm_input,
                            text_tokens,
                            num_token,
                            cache_size,
                            stopping_criteria,
                            logits_processors,
                            logits_warper,
                            cross_attns[0],
                            **kwargs,
                        )

        # Return additional values, currently, these values are only used in below scenarios:
        # 1. 'mtk_evaluate_llm' evaluates logits score of MLLM.
        # 2. 'mtk_find_overture' requires text_tokens in forward function.
        if additional_return_dict is not None:
            if 'text_tokens' in additional_return_dict:
                additional_return_dict['text_tokens'] = text_tokens
            if 'multimodal_embeds' in additional_return_dict:
                additional_return_dict['multimodal_embeds'] = multimodal_embeds
            if 'kwargs' in additional_return_dict:
                additional_return_dict['kwargs'] = kwargs
            return self.remove_mm_tokens_if_exist(llm_output), additional_return_dict

        return self.remove_mm_tokens_if_exist(llm_output)

    def sampler(self, logits, input_ids, logits_processors=None, logits_warper=None):
        """Sample the next token.

        Args:
            logits (torch.Tensor): The logits.
            input_ids (numpy.ndarray): The input IDs.
            logits_processors (LogitsProcessorList, optional): The logits processors. Defaults to None.
            logits_warper (LogitsProcessorList, optional): The logits warper. Defaults to None.

        Returns:
            torch.Tensor: The next token.
        """
        logger.debug('Enter sampler')
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        next_token_logits = logits[:, -1, :].to(self.main_device)
        if logits_processors is not None:
            next_token_logits = logits_processors(
                torch.from_numpy(input_ids).to(torch.int64).to(self.main_device), next_token_logits
            )
        if logits_warper is not None and len(logits_warper) > 0:
            logger.debug('Using random sampler')
            next_token_scores = logits_warper(
                torch.from_numpy(input_ids).to(torch.int64).to(self.main_device), next_token_logits
            )
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            logger.debug('Using greedy sampler')
            next_token = torch.argmax(next_token_logits, dim=-1)
        logger.debug(f'Next token: {next_token}')
        return next_token

    def get_response(self, tokens, preformatter, quiet=False):
        """Get the response.

        Args:
            tokens (list): The tokens.
            preformatter (object): The preformatter object.
            input_length (int): The input length.
            quiet (bool): Flag to not print the response.

        Returns:
            any: The response.
        """
        logger.debug('Enter get_response')
        if self.input_length is None:
            logger.error('`pipeline.forward` not called before `pipeline.get_response`')
        llm_output = tokens[0][self.input_length :] if not preformatter.used else tokens[0]
        if self.input_mode in ['text', 'tokens']:
            if isinstance(self.tokenizer, SentencePieceProcessor):
                logger.debug(f'Output Tokens: {llm_output.tolist()}\n')
                output = self.tokenizer.decode(llm_output.tolist())
            else:
                logger.debug(f'Output Tokens: {llm_output}\n')
                output = self.tokenizer.decode(llm_output)
            output = preformatter.get_response(output)
            if not quiet:
                if self.task == 'make_calibration':
                    logger.debug(f'Response:\n{output}')
                else:
                    logger.info(f'Response:\n{output}')
        else:
            output = llm_output
            if not quiet:
                if self.task == 'make_calibration':
                    logger.debug(f'Response:\n{output}')
                else:
                    logger.info(f'Response:\n{output}')
        return output

    def remove_mm_tokens_if_exist(self, output_tokens):
        """Removes placeholder image/audio tokens from generated response if present.

        Also decrements the input_length attribute of Pipeline accordingly.

        Args:
            output_tokens (np.ndarray): The output tokens to remove image/audio tokens from.

        Returns:
            np.ndarray: The output tokens with image/audio tokens removed.
        """
        if self.task == 'evaluate' and self.has_encoder():
            return output_tokens
        mm_token_mask = self.get_embeds.get_mm_mask(output_tokens[0])
        mm_token_indices = [i for i, x in enumerate(mm_token_mask) if x]
        num_tokens = len(mm_token_indices)
        if num_tokens > 0:
            logger.debug(f'Removing {num_tokens} multimodal placeholder tokens from the generated response.')
            output_tokens = np.delete(output_tokens[0], mm_token_indices)[None, :]
            self.input_length -= num_tokens
        return output_tokens

    def pad_input_ids_to_input_embeds_len(self, input_ids, input_length, dummpy_pad_id=-2454):
        """Pad dummy ids onto input ids to the lenth of input embeds.

        This is to ensure the max_new_token is enforced correctly at generation.
        """
        if dummpy_pad_id >= 0:
            logger.error(f'dummpy_pad_id must be smaller than 0, got {dummpy_pad_id}', err=RuntimeError)
        id_length = input_ids.shape[1]
        pad_length = input_length - id_length
        dummy_pad = dummpy_pad_id * np.ones((1, pad_length), dtype=np.int64)
        return np.concatenate([input_ids, dummy_pad], axis=-1)
