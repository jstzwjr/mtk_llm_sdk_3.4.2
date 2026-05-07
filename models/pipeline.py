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
"""Define pipeline classes."""

import json
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional
from transformers.feature_extraction_utils import BatchFeature

from ..utils import cache_utils, const, generate_utils, logger, overture_utils, quantized_model_utils, utils
from .base_pipeline import BasePipeline


class FloatPipeline(BasePipeline):
    """FloatPipeline class for handling the pipeline of a Pytorch floating point model."""

    def __init__(self, combined_config, lora_configs, task, *args, **kwargs):
        """Initialize the FloatPipeline.

        Args:
            combined_config (dict): The combined configuration.
            lora_configs (list): The list of LoRA configurations.
            task (str): The main task this pipeline is created for.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        logger.debug('Initialize FloatPipeline')
        super().__init__(combined_config, lora_configs, task, *args, **kwargs)

        self.state_dict = {}
        if self.task not in const.FLOAT_PIPELINE_TASKS:
            logger.error(
                f'Unsupported float pipeline task: {self.task}. '
                f'All supported float pipeline tasks: {const.FLOAT_PIPELINE_TASKS}',
                err=ValueError,
            )
        if self.task in ['make_calibration', 'hotplug']:
            self.dtype = kwargs.get('dtype', torch.float32)
        elif self.task == 'ptq':
            self.dtype = kwargs.get('dtype', np.float32 if self.backend == const.CONVERTER else torch.float32)
        else:
            self.dtype = kwargs.get('dtype', torch.float16)

        logger.debug(f'task={self.task}, dtype={self.dtype}')

        self.max_threads = int(os.environ.get('MAX_THREADS', 1))

        self.encoder_weight_dir = getattr(self.config.e, 'weight_dir', None)
        logger.debug(f'encoder_weight_dir={self.encoder_weight_dir}')
        self.llm_weight_dir = getattr(self.config.l, 'weight_dir', None)
        logger.debug(f'llm_weight_dir={self.llm_weight_dir}')
        self.tail_weight_dir = getattr(self.config.t, 'weight_dir', None)
        logger.debug(f'tail_weight_dir={self.tail_weight_dir}')

        self.preprocessor_class = utils.resolve_preprocessor_class(self.config.p)
        self.encoder_chunk_class = utils.resolve_encoder_class(self.config.e)
        self.projector_class = utils.resolve_projector_class(self.config.p2)
        self.llm_chunk_class = utils.resolve_llm_class(self.config.l)

        # NEW (Patch E): bridge vision deepstack count -> LLM config so Qwen3ModelChunk knows
        # which chunks (LLM layer < N) accept ds_padded extra input.
        _ds_indexes = getattr(self.config.e, 'deepstack_visual_indexes', None) if self.config.e is not None else None
        self.config.l._num_deepstack_inject = len(_ds_indexes) if _ds_indexes else 0
        self.config.t, self.llm_tail_class, self.llm_tail_decoder_class = utils.resolve_tail_class(
            self.config.t, self.config.l
        )

        self.ee_index = self.config.l.early_exit_index
        self.is_ee = self.ee_index is not None
        self.distribute_layers = kwargs.pop('distribute_layers', True)

        self._pipeline_type = 'float'

        self._encoder_only_ptq = kwargs.pop('encoder_only_ptq', False)
        if not self._encoder_only_ptq and self.llm_weight_dir is None:
            logger.error(f'LLM Weight Dir: {self.llm_weight_dir} cannot be None')

        self._dummy_weights = kwargs.pop('dummy_weights', False)

        # assume encoder weights inside if self.config.e is not None and encoder weight dir is none
        all_weight_dirs = {'llm': self.llm_weight_dir}
        if self.encoder_weight_dir and self.encoder_weight_dir != self.llm_weight_dir:
            all_weight_dirs['encoder'] = self.encoder_weight_dir
        if self.tail_weight_dir and self.tail_weight_dir != self.llm_weight_dir:
            all_weight_dirs['tail'] = self.tail_weight_dir

        if self.task not in ['test_tokenizer', 'hotplug']:  # These tasks do not need to load any model weights
            self.load_checkpoints(all_weight_dirs)

        if self.task in ['ptq']:
            # Delay initialization for PTQ as PTQ saves memory by only instantiating what is needed
            self.barebones_init(encoder_only=self._encoder_only_ptq)
            self.deduce_num_layers_per_chunk()
            self.lora_handler.load_chunked_lora_inputs(self)
        elif self.task == 'qalft':  # Delay weight loading for QALFT as first step is to gen quant config
            self.qalft_init()
            # init mm required hooks
            if self.config.e is not None:
                self.init_hooks()
                self.init_preprocessor()
                self.init_encoder()
                self.init_projector()
            self.deduce_num_layers_per_chunk()
            self.lora_handler.load_as_state_dict(self)
        elif self.task in ['hotplug']:
            self.barebones_init()
            self.deduce_num_layers_per_chunk()
            self.lora_handler.load_as_state_dict(self)
        elif self.task not in ['test_tokenizer', 'fuse_embed_ln']:
            # test_tokenizer and fuse_embed_ln does not need to instantiate any models
            if self.input_mode != 'embeddings':
                self.init_preprocessor()
                self.init_encoder()
                self.init_projector()
            self.init_llm()
            self.init_tail()
            self.init_hooks()
            self.main_device = self.llm[0].device_list[0]
            self.deduce_num_layers_per_chunk()
            self.lora_handler.load_chunked_lora_inputs(self)
            self.evict_config = kwargs.pop('evict_config', {})
            self.init_evictor(self.config.l, **self.evict_config)

        if self.task == 'make_calibration' and hasattr(self.config.l, 'bita') and self.config.l.bita:
            self.bita_init()

        logger.debug(f'main_device={self.main_device}')

    def _init_single_encoder_layer(self, i):
        logger.debug(f'Init single encoder layer: {i}')
        chunk = self.encoder_chunk_class(
            self.config.e,
            None if not self.lora_handler.has_encoder_lora() else self.lora_handler.e[0],
            num_layers=1,
            first_layer_idx=i,
            chunk_idx=i,
            dtype=utils.get_torch_dtype(self.dtype),
            encoder_weight_dir=self.encoder_weight_dir if self.encoder_weight_dir is not None else self.llm_weight_dir,
        )

        # encoder weights in same bin as llm if 'encoder' sd not found
        encoder_state_dict = self.state_dict['encoder'] if self.state_dict.get('encoder') else self.state_dict['llm']
        chunk, _ = chunk.load_weights(encoder_state_dict, i)
        if i == self.num_encoder_layers - 1:
            chunk.pop_remaining_unused_weights(encoder_state_dict)
        return chunk

    def init_encoder(self):
        """Initialize the encoder."""
        logger.debug('Enter init_encoder')
        if self.config.e is None:
            logger.debug('No encoder to init')
            self.num_encoder_layers = 0
            return

        self.num_encoder_layers = getattr(self.config.e, 'select_layer', self.config.e.num_hidden_layers)
        logger.debug(f'num_encoder_layers={self.num_encoder_layers}')

        logger.info('Instantiating encoder layers...')
        if self.max_threads == 1:
            for i in range(self.num_encoder_layers):
                self.encoder.append(self._init_single_encoder_layer(i))
        else:
            with Pool(min(self.max_threads, self.num_encoder_layers)) as pool:
                self.encoder = pool.map(self._init_single_encoder_layer, range(self.num_encoder_layers))

        lora_state_dict_mapping = {}
        for layer in self.encoder:
            chunk_lora_mapping = layer.generate_default_lora_state_dict_mapping()
            lora_state_dict_mapping.update(chunk_lora_mapping)
        self.lora_handler.set_encoder_lora_state_dict_mapping(lora_state_dict_mapping)

    def init_projector(self):
        """Initialize the projector."""
        logger.debug('Enter init_projector')
        if self.config.p2 is None:
            logger.debug('No projector to init')
            return

        logger.info('Instantiating projector model...')
        if self.task == 'ptq':
            proj = self.projector_class(self.config.p2, dtype=torch.float32, jit_trace=True)
        else:
            proj = self.projector_class(self.config.p2, dtype=utils.get_torch_dtype(self.dtype))

        if not self._dummy_weights:
            encoder_state_dict = (
                self.state_dict['encoder'] if self.state_dict.get('encoder') else self.state_dict['llm']
            )
            self.projector, _ = proj.load_weights(encoder_state_dict)  # projector weights with encoder
        else:
            proj.device_list = ['cpu']
            proj.to('cpu')

    def _init_single_llm_layer(self, i):
        logger.debug(f'Init single LLM layer: {i}')
        chunk = self.llm_chunk_class(
            self.config.l,
            None if not self.lora_handler.has_llm_lora() else self.lora_handler.l[0],
            num_layers=1,
            first_layer_idx=i,
            chunk_idx=i,
            dtype=utils.get_torch_dtype(self.dtype),
            include_tail=False,
            distribute_layers=self.distribute_layers,
            use_single_bmm_attention=self.use_single_bmm_attention,
        )

        chunk, _ = chunk.load_weights(self.state_dict['llm'], i)
        return chunk

    def init_text_embedding(self):
        """Initialize the text embedding layer for LLM."""
        self.text_embedding_layer = utils.get_embedding_layer(self.config.l, state_dict=self.state_dict['llm']).to(
            self.main_device
        )

    def init_llm(self):
        """Initialize the LLM."""
        logger.debug('Enter init_llm')
        self.num_decoder_layers = (
            self.config.l.early_exit_index + self.config.l.early_exit_num_layers
            if self.is_ee
            else self.config.l.num_hidden_layers
        )
        logger.debug(f'num_decoder_layers={self.num_decoder_layers}')

        self.init_text_embedding()

        logger.info('Instantiating decoder layers...')
        if self.max_threads == 1:
            for i in range(self.num_decoder_layers):
                self.llm.append(self._init_single_llm_layer(i))
        else:
            with Pool(min(self.max_threads, self.num_decoder_layers)) as pool:
                self.llm = pool.map(self._init_single_llm_layer, range(self.num_decoder_layers))

        lora_state_dict_mapping = {}
        lora_merged_state_dict_mapping = {}
        for layer in self.llm:
            chunk_lora_mapping, chunk_merged_lora_mapping = layer.generate_default_lora_state_dict_mapping()
            lora_state_dict_mapping.update(chunk_lora_mapping)
            lora_merged_state_dict_mapping.update(chunk_merged_lora_mapping)
        self.lora_handler.set_llm_lora_state_dict_mapping(lora_state_dict_mapping, lora_merged_state_dict_mapping)

        if self.config.l.overture_dict is not None:
            self.overture = overture_utils.get_overture(self.config.l.overture_dict)
            if self.overture is None and 'length' in self.config.l.overture_dict:
                logger.error(
                    f'Overture is enabled but Overture file is not found in {self.config.l.overture_dict["path"]}.'
                )

            if self.overture is not None:
                logger.warning("Tokenizer's `add_bos` attribute forcibly set to False for Overture feature.")
                self.set_tokenizer_add_bos(False)

    def init_encoder_layer_for_ptq(self, layer_idx):
        """Initialize single encoder layer for per-layer PTQ.

        Args:
            layer_idx (int): encoder layer index to instantiate.
        """
        logger.debug('Enter init_encoder_layer_for_ptq')
        if self.task != 'ptq':
            logger.error('`init_encoder_layer_for_ptq` function can only be called when task is `ptq`')
        logger.info(f'Instantiating encoder layer {layer_idx}')
        chunk = self.encoder_chunk_class(
            self.config.e,
            None if not self.lora_handler.has_encoder_lora() else self.lora_handler.e[0],
            num_layers=1,
            first_layer_idx=layer_idx,
            chunk_idx=layer_idx,
            dtype=torch.float32,
            jit_trace=True,
            encoder_weight_dir=self.encoder_weight_dir if self.encoder_weight_dir is not None else self.llm_weight_dir,
        )

        if not self._dummy_weights:
            encoder_state_dict = (
                self.state_dict['encoder'] if self.state_dict.get('encoder') else self.state_dict['llm']
            )
            chunk, _ = chunk.load_weights(encoder_state_dict, layer_idx)
            if layer_idx == self.num_encoder_layers - 1:
                chunk.pop_remaining_unused_weights(encoder_state_dict)
        else:
            chunk.device_list = ['cpu']
            chunk.to('cpu')
        return chunk

    def init_llm_layer_for_ptq(self, layer_idx=None):
        """Initialize LLM layer for PTQ.

        Args:
            layer_idx (int): LLM layer index to instantiate. If not provided, initialize the entire LLM.
        """
        logger.debug('Enter init_llm_layer_for_ptq')
        if self.task != 'ptq':
            logger.error('`init_llm_layer_for_ptq` function can only be called when task is `ptq`')

        num_layers = self.num_decoder_layers if layer_idx is None else 1
        layer_idx = 0 if layer_idx is None else layer_idx

        logger.info(f'Instantiating decoder layer {layer_idx}')
        chunk = self.llm_chunk_class(
            self.config.l,
            None if not self.lora_handler.has_llm_lora() else self.lora_handler.l[0],
            num_layers=num_layers,
            first_layer_idx=layer_idx,
            chunk_idx=layer_idx,
            dtype=torch.float32,
            jit_trace=True,
            use_single_bmm_attention=self.use_single_bmm_attention,
        )

        if not self._dummy_weights:
            if self.is_ee and layer_idx >= self.ee_index:
                chunk, _ = chunk.load_weights(self.state_dict['llm'], layer_idx - self.ee_index)
            else:
                chunk, _ = chunk.load_weights(self.state_dict['llm'], layer_idx)
        else:
            chunk.device_list = ['cpu' for _ in range(num_layers)]
            chunk.to('cpu')
        return chunk

    def init_tail(self):
        """Initialize the tail."""
        logger.debug('Enter init_tail')
        if self.task == 'ptq':
            dtype = torch.float32
            jit_trace = True
        else:
            dtype = utils.get_torch_dtype(self.dtype)
            jit_trace = False

        if self.config.t is None:
            logger.debug('Init normal tail')
            chunk = self.llm_tail_class(
                self.config.l, chunk_idx=self.num_decoder_layers, dtype=dtype, jit_trace=jit_trace
            )
        else:
            logger.error(
                'Medusa and EAGLE tails are temporarily not supported in the current version.', err=NotImplementedError
            )
            if self.config.t.model_type == 'eagle':
                logger.debug('Init EAGLE tail')
                chunk = self.llm_tail_class(
                    self.config.t,
                    chunk_idx=self.num_decoder_layers,
                    dtype=dtype,
                    decoder_layer=self.llm_tail_decoder_class,
                    jit_trace=jit_trace,
                )
            else:
                logger.debug('Init Medusa tail')
                chunk = self.llm_tail_class(
                    self.config.t, chunk_idx=self.num_decoder_layers, dtype=dtype, jit_trace=jit_trace
                )

        if not self._dummy_weights:
            tail_state_dict = self.state_dict['tail'] if self.state_dict.get('tail') else self.state_dict['llm']
            chunk, _ = chunk.load_weights(tail_state_dict)
        else:
            # state dict should only be none during fake ptq
            chunk.device_list = ['cpu']
            chunk.to('cpu')

        self.tail = chunk
        return chunk

    def barebones_init(self, encoder_only=False):
        """Modified init function for PTQ, which does not instantiate models yet.

        Args:
            encoder_only (bool): Flag to indicate whether to barebones init encoder model only. Defaults to False.

        """
        logger.debug('Enter barebones_init')

        if self.config.e is not None:
            # Init encoder-related attributes only first
            self.num_encoder_layers = getattr(self.config.e, 'select_layer', self.config.e.num_hidden_layers)
            logger.debug(f'num_encoder_layers={self.num_encoder_layers}')

            # Init the encoder chunks just to get the lora mapping dicts, don't load weights yet
            lora_state_dict_mapping = {}
            for i in range(self.num_encoder_layers):
                chunk = self.encoder_chunk_class(
                    self.config.e,
                    None if not self.lora_handler.has_encoder_lora() else self.lora_handler.e[0],
                    num_layers=1,
                    first_layer_idx=i,
                    chunk_idx=i,
                    dtype=torch.float32,
                    jit_trace=True,
                    encoder_weight_dir=self.encoder_weight_dir
                    if self.encoder_weight_dir is not None
                    else self.llm_weight_dir,
                )
                chunk_lora_mapping = chunk.generate_default_lora_state_dict_mapping()
                lora_state_dict_mapping.update(chunk_lora_mapping)
            self.lora_handler.set_encoder_lora_state_dict_mapping(lora_state_dict_mapping)

            # Init pre_projector_hook
            if self.config.p2h.name in const.SUPPORTED_HOOKS:
                logger.debug(f'Init pre-projector hook using {self.config.p2h.name}')
                hook_class = utils.resolve_hook_class(self.config.p2h.name)
                self.pre_projector_hook = hook_class(self.config.p2h)
            else:
                logger.error('', err=NotImplementedError)

            if encoder_only:
                logger.debug('Encoder only PTQ, only barebone init encoder model chunks.')
                return

        # Init LLM-related attributes only first
        self.num_decoder_layers = (
            self.config.l.early_exit_index + self.config.l.early_exit_num_layers
            if self.is_ee
            else self.config.l.num_hidden_layers
        )
        logger.debug(f'num_decoder_layers={self.num_decoder_layers}')

        # Init the LLM chunks just to get the lora mapping dicts, don't load weights yet
        lora_state_dict_mapping = {}
        lora_merged_state_dict_mapping = {}
        for i in range(self.num_decoder_layers):
            chunk = self.llm_chunk_class(
                self.config.l,
                None if not self.lora_handler.has_llm_lora() else self.lora_handler.l[0],
                num_layers=1,
                first_layer_idx=i,
                chunk_idx=i,
                dtype=torch.float32,
                jit_trace=True,
                use_single_bmm_attention=self.use_single_bmm_attention,
            )
            chunk_lora_mapping, chunk_merged_lora_mapping = chunk.generate_default_lora_state_dict_mapping()
            lora_state_dict_mapping.update(chunk_lora_mapping)
            lora_merged_state_dict_mapping.update(chunk_merged_lora_mapping)
        self.lora_handler.set_llm_lora_state_dict_mapping(lora_state_dict_mapping, lora_merged_state_dict_mapping)

    def bita_init(self):
        """Initialize Bidirectional Token Attention (BiTA) components if enabled."""
        logger.debug('Initializing BiTA components')
        self.use_bita = True

        with open(self.config.l.bita_config) as f:
            bita_config = json.load(f)

        bita_weight = torch.load(bita_config['weight_path'])
        self.bita_prefix = bita_weight[bita_config['prefix_key_name']]
        self.bita_prefix_length = bita_config['prefix_length']
        self.bita_draft_length = bita_config['bita_inference_draft_length']

        # Process the prefix weights for use in attention mechanism
        self._process_bita_prefix()
        logger.info(f'BiTA initialized with {self.bita_prefix_length} prefix tokens')

    def _process_bita_prefix(self):
        """Process BiTA prefix weights for use in attention mechanism."""
        if not self.use_bita:
            return

        # Extract prefix encoder weights
        prefix_weight = self.bita_prefix

        # Reshape if needed to match expected dimensions
        if len(prefix_weight.shape) != 6:
            bita_prefix_dim = self.config.l.hidden_size // self.config.l.num_attention_heads
            prefix_weight = prefix_weight.reshape(
                self.config.l.num_hidden_layers,
                2,  # key and value
                1,  # batch size
                self.config.l.num_key_value_heads,
                self.bita_prefix_length,
                bita_prefix_dim,
            )

        # Convert to appropriate dtype and device
        if isinstance(self.dtype, torch.dtype) and not isinstance(prefix_weight, torch.Tensor):
            prefix_weight = torch.tensor(prefix_weight, dtype=self.dtype).to(self.main_device)
        elif isinstance(self.dtype, np.dtype) and isinstance(prefix_weight, torch.Tensor):
            prefix_weight = prefix_weight.cpu().numpy().astype(self.dtype)

        # Store processed prefix weights
        self.bita_prefix = prefix_weight
        logger.debug(f'Processed BiTA prefix with shape {self.bita_prefix.shape}')

    def qalft_init(self):
        """Modified init function for QALFT, which creates one chunk for all layers and does not load weights yet."""
        logger.debug('Enter qalft_init')

        # Init LLM-related attributes only first
        self.num_decoder_layers = (
            self.config.l.early_exit_index + self.config.l.early_exit_num_layers
            if self.is_ee
            else self.config.l.num_hidden_layers
        )
        logger.debug(f'num_decoder_layers={self.num_decoder_layers}')

        # Init the LLM chunks just to get the lora mapping dicts, don't load weights yet
        lora_state_dict_mapping = {}
        lora_merged_state_dict_mapping = {}
        self.llm = self.llm_chunk_class(
            self.config.l,
            self.lora_handler.l[0],
            num_layers=self.num_decoder_layers,
            first_layer_idx=0,
            chunk_idx=0,
            dtype=utils.get_torch_dtype(self.dtype),
            include_tail=True,
            parallel_lora=True,
            distribute_layers=self.distribute_layers,
            use_single_bmm_attention=self.use_single_bmm_attention,
        )
        chunk_lora_mapping, chunk_merged_lora_mapping = self.llm.generate_default_lora_state_dict_mapping()
        lora_state_dict_mapping.update(chunk_lora_mapping)
        lora_merged_state_dict_mapping.update(chunk_merged_lora_mapping)
        self.lora_handler.set_llm_lora_state_dict_mapping(lora_state_dict_mapping, lora_merged_state_dict_mapping)

    def get_encoder_full_model(self, jit_trace=False, existing_lora=False, parallel_lora=False):
        """Gets the Encoder model with loaded weights.

        Args:
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            existing_lora (bool): A boolean indicating if the LoRA used is pre-trained or not. Default is False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
        """
        logger.debug('Enter get_encoder_full_model')

        if self.config.e is None:
            logger.error('MLLM configuration is not set.')

        self.num_encoder_layers = getattr(self.config.e, 'select_layer', self.config.e.num_hidden_layers)
        logger.debug(f'num_encoder_layers={self.num_encoder_layers}')

        logger.info('Instantiating encoder layers...')
        self.encoder = self.encoder_chunk_class(
            self.config.e,
            None if not self.lora_handler.has_encoder_lora() else self.lora_handler.e[0],
            num_layers=self.num_encoder_layers,
            first_layer_idx=0,
            chunk_idx=0,
            dtype=self.dtype,
            jit_trace=jit_trace,
            parallel_lora=parallel_lora,
            encoder_weight_dir=self.encoder_weight_dir if self.encoder_weight_dir is not None else self.llm_weight_dir,
        )

        lora_state_dict_mapping = self.encoder.generate_default_lora_state_dict_mapping()
        self.lora_handler.set_encoder_lora_state_dict_mapping(lora_state_dict_mapping)
        encoder_state_dict = self.state_dict['encoder'] if self.state_dict.get('encoder') else self.state_dict['llm']
        if existing_lora:
            encoder_state_dict = {**encoder_state_dict, **self.lora_handler.state_dicts[0]['encoder']}
        _, _ = self.encoder.load_weights(encoder_state_dict, 0)

        self.projector = self.projector_class(self.config.p2, dtype=self.dtype, jit_trace=jit_trace)
        _, _ = self.projector.load_weights(encoder_state_dict)

    def get_llm_full_model(self, jit_trace=False, existing_lora=False, parallel_lora=False):
        """Gets the LLM model with loaded weights.

        Args:
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            existing_lora (bool): A boolean indicating if the LoRA used is pre-trained or not. Default is False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
        """
        logger.debug('Enter get_llm_full_model')

        # Init LLM-related attributes only first
        self.num_decoder_layers = (
            self.config.l.early_exit_index + self.config.l.early_exit_num_layers
            if self.is_ee
            else self.config.l.num_hidden_layers
        )
        logger.debug(f'num_decoder_layers={self.num_decoder_layers}')

        self.llm = self.llm_chunk_class(
            self.config.l,
            None if not self.lora_handler.has_llm_lora() else self.lora_handler.l[0],
            num_layers=self.num_decoder_layers,
            first_layer_idx=0,
            chunk_idx=0,
            dtype=self.dtype,
            include_tail=True,
            jit_trace=jit_trace,
            parallel_lora=parallel_lora,
            distribute_layers=self.distribute_layers,
            use_single_bmm_attention=self.use_single_bmm_attention,
        )
        lora_state_dict_mapping, lora_merged_state_dict_mapping = self.llm.generate_default_lora_state_dict_mapping()
        self.lora_handler.set_llm_lora_state_dict_mapping(lora_state_dict_mapping, lora_merged_state_dict_mapping)

        if existing_lora:
            # The key of LoRA dict is different from the name in the model. A correct key mapping is required to load
            # the LoRA weights. For example, `0_q_A` should be mapped to `layers.0.self_attn.q_proj.lora_A.weight`.
            llm_lora_dict = self.lora_handler.state_dicts[0]['llm']
            llm_lora_dict_with_external_key = {
                next(iter(self.lora_handler.llm_lora_state_dict_mapping[k].values())): llm_lora_dict[k]
                for k in llm_lora_dict
            }
            self.state_dict['llm'] = {**self.state_dict['llm'], **llm_lora_dict_with_external_key}

        model, _ = self.llm.load_weights(self.state_dict['llm'], 0)
        return model

    def get_fakequant_model(self, quant_config, existing_lora):
        """Gets the LLM model with fakequant operators inserted.

        Args:
            quant_config (str): The filepath of the mtk_quantization quant config.
            existing_lora (bool): A boolean indicating if the LoRA used is pre-trained or not.
        """
        if existing_lora:
            cur_lora = {}
            for key in self.lora_handler.state_dicts[0]['llm']:
                external_model_key = next(iter(self.lora_handler.llm_lora_state_dict_mapping[key].values()))
                cur_lora[external_model_key] = self.lora_handler.state_dicts[0]['llm'][key]
            self.state_dict['llm'] = {**self.state_dict['llm'], **cur_lora}

        model, _ = self.llm.load_weights(self.state_dict['llm'], 0, quant_config)
        return model

    def check_all_weights_assigned(self):
        """Check if all weights in state dict are assigned."""
        logger.debug('Enter check_all_weights_assigned')
        keys = list(self.state_dict['llm'].keys())
        embed_subkey = self.config.l.embedding_key

        logger.debug(f'Remaining keys after loading weights: {keys}')

        if self.config.l.model_type == 'milm':
            logger.debug('Remove `decoder.version` key for milm model')
            self.state_dict['llm'].pop('decoder.version')

        for k in keys:
            if 'inv_freq' in k:
                logger.debug(f'Remove rot-emb related key: {k}')
                self.state_dict['llm'].pop(k)
            if embed_subkey in k:
                logger.debug(f'Remove token embedding key: {k}')
                self.state_dict['llm'].pop(k)

        if len(self.state_dict) > 0:
            logger.error(f'There are extra state_dict keys that are unassigned:\n{self.state_dict["llm"].keys()}')
        del self.state_dict['llm']

    def deduce_num_layers_per_chunk(self):
        """Deduce the number of decoder layers per chunk."""
        logger.debug('Enter deduce_num_layers_per_chunk')

        self.encoder_layers_per_chunk = [1] * self.num_encoder_layers
        self.llm_layers_per_chunk = [1] * self.num_decoder_layers
        self.encoder_layer_ids = [[x] for x in range(self.num_encoder_layers)]
        self.llm_layer_ids = [[x] for x in range(self.num_decoder_layers)]
        if self.backend == const.MLKITS and self.task == 'ptq':
            self.encoder_layers_per_chunk = [self.num_encoder_layers]
            self.llm_layers_per_chunk = [self.num_decoder_layers]
            self.encoder_layer_ids = [[self.num_encoder_layers]]
            self.llm_layer_ids = [list(range(self.num_decoder_layers))]
        logger.debug(f'encoder_layers_per_chunk = {self.encoder_layers_per_chunk}')
        logger.debug(f'llm_layers_per_chunk = {self.llm_layers_per_chunk}')

    def load_checkpoints(self, folders: dict):
        """Load checkpoints from the specified folders.

        Args:
            folders (list): List of folders to load checkpoints from.
        """
        logger.debug('Enter load_checkpoints')

        checkpoint_files: dict = {}

        for model, weight_dir in folders.items():
            if weight_dir is None:
                continue
            if not self._dummy_weights:
                checkpoint_files[model] = [
                    os.path.join(weight_dir, f)
                    for f in os.listdir(weight_dir)
                    if f.endswith(('.bin', '.safetensors')) and not f.startswith(('training_args', 'embedding'))
                ]

        if len(checkpoint_files) == 0:
            if any(folders[folder] is not None for folder in folders):
                if self._dummy_weights:
                    logger.warning(f'WARNING: No checkpoint files found in folders: {folders}. Using dummy weights.')
                else:
                    logger.error(f'Error: No checkpoint files found in folders: {folders}.')
            return {}

        self.state_dict: dict = {model: {} for model in folders}
        logger.info('Loading weights from:')
        for model in checkpoint_files:
            filelist = checkpoint_files[model]
            for f in filelist:
                logger.info(f)

                if f.endswith('.safetensors'):
                    self.state_dict[model] = {**self.state_dict[model], **utils.load_file(f)}
                else:
                    self.state_dict[model] = {**self.state_dict[model], **torch.load(f, map_location='cpu')}

        if self.config.l.early_exit_index is not None and self.config.l.model_type != 'gecko2':
            prefix = ''
            for k in list(self.state_dict['llm'].keys()):
                if 'layers.' in k:
                    if prefix == '':
                        prefix = k.split('layers.')[0]
                    layer_idx = int(k.split('layers.')[1].split('.')[0])
                    if layer_idx >= self.ee_index:
                        del self.state_dict['llm'][k]
                elif 'h.' in k:
                    if prefix == '':
                        prefix = k.split('h.')[0]
                    layer_idx = int(k.split('layers.')[1].split('.')[0])
                    if layer_idx >= self.ee_index:
                        del self.state_dict['llm'][k]
                elif k == f'{prefix}norm.weight':
                    del self.state_dict['llm'][k]

        if (self.config.l.model_type in ['minicpm']) and (not self.config.l.llama_format):
            from .llm.modeling_minicpm import convert_weight_to_llama_format

            self.state_dict['llm'] = convert_weight_to_llama_format(self.state_dict['llm'], self.config.l)
        return None

    def update_state_dict(self, llm_indices_to_ptq: list):
        """Update State Dict for partial PTQ case."""
        import re

        keys_to_keep = []
        layer_pattern = re.compile(r'(.+\.)?layers\.(\d+)\..*')
        for key in self.state_dict['llm']:
            match = re.match(layer_pattern, key)
            if match is not None:
                match = match.groups()  # match[0] will be None if nothing prefixes layers
                if len(match) > 2:
                    logger.error(f'Found invalid key name in state_dict: {key}')
                layer_idx = int(match[1])
                if self.is_ee and 'ee' in key:
                    logger.debug(f'Found ee layer_idx: {layer_idx}')
                    layer_idx += self.ee_index
                logger.debug(f'Found actual layer_idx: {layer_idx}')
                if layer_idx not in llm_indices_to_ptq:
                    continue
                keys_to_keep.append(key)
                logger.debug(f'Keeping key: {key}')
            else:
                # tail or embed
                if self.num_decoder_layers == llm_indices_to_ptq[-1] or 0 in llm_indices_to_ptq:
                    keys_to_keep.append(key)
                    logger.debug(f'Keeping key: {key}')

        # Patch F6: filter visual.* keys (vision encoder + projector weights) so partial-tail PTQ
        # doesn't mis-load visual.merger.norm.weight as Qwen tail norm.weight (shape mismatch 1024 vs 2560).
        keys_to_keep = [k for k in keys_to_keep if not k.startswith('visual.')]

        self.state_dict['llm'] = {key: self.state_dict['llm'][key] for key in keys_to_keep}
        logger.debug(f'Updated tensor names in state dict:\n{self.state_dict["llm"].keys()}')

    """ Forward passes """

    def forward_encoder(self, inputs, **kwargs):
        """Forward pass through the encoder.

        Args:
            inputs: The inputs for the encoder.
            kwargs: Additional keyword arguments.
        """
        logger.debug('Enter forward_encoder')
        if self.config.e is None:
            return inputs, kwargs

        if isinstance(inputs, BatchFeature):
            input_list = [inputs['input_features']]
            if 'attention_mask_audio' in inputs:
                input_list.append(torch.tensor(inputs['attention_mask_audio']))
            inputs = input_list

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        inputs = [x.to(self.dtype) for x in inputs]

        output_dirs = [] if self.task != 'make_calibration' else kwargs.get('output_dirs')['encoder']
        lora_mapping_files = [] if self.task != 'make_calibration' else kwargs.get('lora_mapping_files')['encoder']

        # Qwen2-VL
        qwen2vl_vision_attn_mask = kwargs.get('qwen2vl_vision_attn_mask')
        qwen2vl_vision_rot_emb = kwargs.get('qwen2vl_vision_rot_emb')

        # Qwen2.5-VL
        qwen2_5vl_vision_attn_mask = kwargs.get('qwen2_5vl_vision_attn_mask')
        qwen2_5vl_vision_attn_mask_window = kwargs.get('qwen2_5vl_vision_attn_mask_window')
        qwen2_5vl_vision_rot_emb = kwargs.get('qwen2_5vl_vision_rot_emb')
        qwen2_5vl_vision_window_index = kwargs.get('qwen2_5vl_vision_window_index')
        qwen2_5vl_vision_fullatt_block_indexes = kwargs.get('qwen2_5vl_vision_fullatt_block_indexes', [])

        if qwen2vl_vision_attn_mask is not None and qwen2_5vl_vision_attn_mask is not None:
            logger.error('Cannot simultaneously pass qwen2-vl attention mask and qwen2.5-vl attention mask')
        if qwen2vl_vision_attn_mask is not None and qwen2_5vl_vision_attn_mask_window is not None:
            logger.error('Cannot simultaneously pass qwen2-vl attention mask and qwen2.5-vl attention mask')

        # For Qwen2-VL, all layer's attention masks are the same,
        # while for Qwen2.5-VL, only the layers in qwen2_5vl_vision_fullatt_block_indexes
        # will use normal attention mask.
        qwen_attn_mask = (
            qwen2vl_vision_attn_mask if qwen2vl_vision_attn_mask is not None else qwen2_5vl_vision_attn_mask
        )
        qwen_attn_mask_window = (
            qwen2vl_vision_attn_mask if qwen2vl_vision_attn_mask is not None else qwen2_5vl_vision_attn_mask_window
        )
        qwen_vision_rot_emb = qwen2vl_vision_rot_emb if qwen2vl_vision_rot_emb is not None else qwen2_5vl_vision_rot_emb

        internvl_navit_rope_vision_rot_emb = kwargs.get('internvl_navit_rope_rotary_pos_emb')
        internvl_navit_rope_cu_seqlens = kwargs.get('internvl_navit_rope_cu_seqlens')

        # For Qwen3-VL: set precomputed positional embeddings and attention mask on encoder chunk 0.
        # During float inference, _precomputed_pos_embed is never set (only set in get_jit_trace_inputs),
        # so the fallback arange(seq_len) would use wrong sequential indices instead of 2D spatial positions.
        qwen3vl_image_grid_thw = kwargs.get('image_grid_thw')
        if qwen3vl_image_grid_thw is not None and len(self.encoder) > 0:
            enc0 = self.encoder[0]
            if hasattr(enc0, '_fast_pos_embed_interpolate'):
                if isinstance(qwen3vl_image_grid_thw, np.ndarray):
                    qwen3vl_image_grid_thw = torch.from_numpy(qwen3vl_image_grid_thw)
                enc0._precomputed_pos_embed = enc0._fast_pos_embed_interpolate(qwen3vl_image_grid_thw).detach()
                if qwen_attn_mask is not None:
                    for enc in self.encoder:
                        enc._vision_attention_mask = qwen_attn_mask
                # NEW: also propagate vision rotary embedding (freqs) to every encoder chunk
                # so each Qwen3VL block can apply RoPE per-layer (HF-aligned).
                if qwen_vision_rot_emb is not None:
                    for enc in self.encoder:
                        enc._vision_rot_emb = qwen_vision_rot_emb

        outputs = []
        cross_attns = []
        hidden_states = inputs[0]
        logger.debug(f'self.encoder_layers_per_chunk: {self.encoder_layers_per_chunk}')
        for i in range(len(self.encoder_layers_per_chunk)):
            extra_inputs = []
            # TODO: refactor all the mllm models to use inner_encoder_hook
            if hasattr(self.encoder[i], 'inner_encoder_hook'):
                self.encoder[i].inner_encoder_hook(hidden_states, kwargs)
            if getattr(self.encoder[i], 'attention_mask', None) is not None:
                if i in qwen2_5vl_vision_fullatt_block_indexes:
                    logger.debug(f'Set normal attention mask of Qwen2.5-VL at layer {i}.')
                    self.encoder[i].attention_mask = qwen_attn_mask
                else:
                    logger.debug(f'Set window attention mask of Qwen2.5-VL at layer {i}.')
                    self.encoder[i].attention_mask = qwen_attn_mask_window
            if getattr(self.encoder[i], 'rot_emb', None) is not None:
                self.encoder[i].rot_emb = qwen_vision_rot_emb
            if getattr(self.encoder[i], 'window_index', None) is not None:
                self.encoder[i].window_index = qwen2_5vl_vision_window_index
            if getattr(self.encoder[i], 'internvl_navit_rope_vision_rot_emb', None) is not None:
                self.encoder[i].internvl_navit_rope_vision_rot_emb = internvl_navit_rope_vision_rot_emb
            if getattr(self.encoder[i], 'internvl_navit_rope_cuseqlen', None) is not None:
                self.encoder[i].internvl_navit_rope_cuseqlen = internvl_navit_rope_cu_seqlens
            logger.debug(f'chunk {i}')
            logger.debug(f'hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}')
            if self.lora_handler.has_encoder_lora():
                logger.debug(f'num_lora_inputs dtype: {self.lora_handler.encoder_lora_inputs[i][0].dtype}')

            save_chunk = self.backend == const.CONVERTER or (self.backend == const.MLKITS and i == 0)
            if self.task == 'make_calibration' and not kwargs.get('dupe') and save_chunk:
                logger.debug(f'Save encoder calibration batch (layer {i})')
                save_dict = {'hidden_states': hidden_states.cpu().numpy().astype(np.float32)}
                if len(inputs) > 1:
                    save_dict.update({'audio_attn_mask': inputs[1].cpu().numpy().astype(np.float32)})

                np.savez(
                    os.path.join(
                        output_dirs[i],
                        'batch-{:04d}.npz'.format(len([x for x in os.listdir(output_dirs[i]) if x.endswith('.npz')])),
                    ),
                    **save_dict,
                )
                if len(lora_mapping_files) > 0:
                    lora_mapping_files[i].write(self.lora_handler.lora_config_paths[0] + '\n')
            if len(inputs) > 1:
                extra_inputs.append(inputs[1])
            extra_inputs.extend(self.lora_handler.encoder_lora_inputs[i])

            # NEW (Patch A): encoder chunk 在 deepstack tap 点会返回 tuple(hs, ds_tensor[, ...]) 给 jit_trace；
            # eager 路径下 deepstack 仍走 _last_deepstack_intermediates side-channel，这里只取 hs[0] 串到下一层。
            _enc_out = self.encoder[i](hidden_states, *extra_inputs)
            hidden_states = _enc_out[0] if isinstance(_enc_out, tuple) else _enc_out
        out = hidden_states

        # NEW: deepstack — aggregate intermediates captured by each encoder chunk
        deepstack_raw_intermediates = []
        for enc in self.encoder:
            chunk_caps = getattr(enc, '_last_deepstack_intermediates', None)
            if chunk_caps:
                deepstack_raw_intermediates.extend(chunk_caps)
        if deepstack_raw_intermediates:
            kwargs['_deepstack_raw_intermediates'] = deepstack_raw_intermediates
            logger.debug(
                f'[deepstack] encoder captured {len(deepstack_raw_intermediates)} intermediate(s); '
                f'first shape={tuple(deepstack_raw_intermediates[0].shape)}'
            )

        if isinstance(out, tuple):
            if len(out) != 2:
                logger.error(
                    f'Expected a tuple of 2 items (encoder_outputs, cross_attn), but got a tuple of len {len(out)}.'
                )
            outputs.append(out[0])
            cross_attns.append(out[1])
        else:
            outputs.append(out)
        if len(outputs) == 1:
            outputs = outputs[0]
        if len(cross_attns) == 0:
            cross_attns = None
        elif len(cross_attns) == 1:
            cross_attns = cross_attns[0]

        kwargs.update(
            {
                'phi3v_sub_GN': getattr(self.encoder[0], 'sub_GN', None),
                'phi3v_glb_GN': getattr(self.encoder[0], 'glb_GN', None),
            }
        )

        return outputs, cross_attns, kwargs

    def forward_projector(self, inputs, **kwargs):
        """Forward pass through the projector.

        Args:
            inputs: The inputs for the projector.
            kwargs: Additional keyword arguments.
        """
        logger.debug(f'Enter {self.projector.__class__.__name__} forward_projector')
        if self.config.p2 is None:
            return inputs, kwargs

        logger.debug(f'projector input shape: {inputs.shape}, dtype: {inputs.dtype}')
        if hasattr(self.projector, 'inner_projector_hook'):
            self.projector.inner_projector_hook(inputs, kwargs)

        # NEW (Patch C): 准备 deepstack 输入（设备/dtype 对齐到 projector 主输入）
        raw_inters = kwargs.pop('_deepstack_raw_intermediates', None) or ()
        if raw_inters:
            try:
                _proj_param = next(self.projector.parameters())
                _t_dev, _t_dtype = _proj_param.device, _proj_param.dtype
            except StopIteration:
                _t_dev, _t_dtype = inputs.device, inputs.dtype
            raw_inters = tuple(r.to(device=_t_dev, dtype=_t_dtype) for r in raw_inters)
            inputs = inputs.to(device=_t_dev, dtype=_t_dtype)

        save_chunk = self.backend == const.CONVERTER
        if self.task == 'make_calibration' and save_chunk:
            logger.debug('Save projector calibration batch')
            output_dir = None if self.task != 'make_calibration' else kwargs.get('output_dirs')['encoder'][-1]

            save_dict = {'hidden_states': inputs.cpu().numpy().astype(np.float32)}
            # NEW (Patch C): 校准 npz 把 deepstack 输入也存进来，键名与 projector.get_ptq_inputs 读取约定一致
            for _j, _r in enumerate(raw_inters):
                save_dict[f'deepstack_{_j}'] = _r.cpu().numpy().astype(np.float32)
            np.savez(
                os.path.join(
                    output_dir,
                    'batch-{:04d}.npz'.format(len([x for x in os.listdir(output_dir) if x.endswith('.npz')])),
                ),
                **save_dict,
            )

        # NEW (Patch C): projector forward 现在自带 deepstack mergers，pipeline 不再做 Python 编排
        proj_out = self.projector(inputs, *raw_inters)
        if isinstance(proj_out, tuple):
            main_out = proj_out[0]
            ds_embeds = list(proj_out[1:])
            if ds_embeds:
                kwargs['deepstack_visual_embeds'] = ds_embeds
                self._deepstack_consumed = 0  # reset per-prompt cursor
                logger.debug(
                    f'[deepstack] projector produced {len(ds_embeds)} ds_embeds; '
                    f'first shape={tuple(ds_embeds[0].shape)} dtype={ds_embeds[0].dtype}'
                )
        else:
            main_out = proj_out
        return main_out, kwargs

    @torch.no_grad()
    def forward_llm_float(
        self,
        input_embeds,
        mask,
        pos_emb,
        cache,
        pad_len=0,
        curr_tokens=None,
        cross_attn=None,
        tail_cache=None,
        memory=None,
        **kwargs,
    ):
        """Forward pass of LLM.

        Args:
            input_embeds: The input embeddings.
            mask: The combined attention+cache mask.
            pos_emb: List of positional embedding components.
            cache: The input cache object.
            pad_len: Length of prompt padding, if any.
            curr_tokens: The current input token ids corresponding to the input_embeds.
            cross_attn: Cross attention from encoder.
            tail_cache: The input cache object for the EAGLE tail if using EAGLE model.
            memory: The memory object for infini attention if using infini attention.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor. logits output of LLM.
            Cache object. Updated cache of LLM.
            None or Cache object. Updated cache of EAGLE tail if using EAGLE model, else None.
        """
        logger.debug('Enter forward_llm_float')
        hidden_states = input_embeds
        original_mask = mask
        curr_k = []
        curr_v = []

        kwargs['attn_logits'] = []
        kwargs['attn_weights'] = []

        output_dirs = [] if self.task != 'make_calibration' else kwargs['output_dirs']['llm']
        lora_mapping_files = [] if self.task != 'make_calibration' else kwargs['lora_mapping_files']['llm']

        cross_attention = []
        if cross_attn is not None:
            for i in range(int(len(cross_attn) / 2)):
                cross_attention.append([cross_attn[i * 2], cross_attn[i * 2 + 1]])

        if self.config.l.model_type == 'gecko2':
            cos_sin = pos_emb[-2:]  # for gecko2 usage
            input_mask = mask  # for gecko2 usage
            all_layer_emb = pos_emb.pop(0) if self.config.l.d_per_layer_embedding else None
            fire_pos_emb = pos_emb.pop(0) if self.config.l.fire_normalizer_threshold else None
        elif self.config.l.model_type == 'gemma3':
            input_mask = mask
            pos_emb_local = pos_emb[-2:]
            pos_emb_global = pos_emb[:2]

        for i in range(sum(self.llm_layers_per_chunk)):
            if self.config.l.model_type == 'gecko' and i >= 1 and len(pos_emb) > 2:
                pos_emb.remove(pos_emb[0])
            elif self.config.l.model_type == 'gecko2':
                pos_emb = []
                if all_layer_emb is not None and (not self.is_ee or (self.is_ee and i < self.ee_index)):
                    pos_emb.append(all_layer_emb)

                if fire_pos_emb is not None and i in self.config.l.fire_pe_index:
                    pos_emb.extend(fire_pos_emb)
                else:
                    pos_emb.extend(cos_sin)

                if (
                    self.config.l.global_local_attention_pattern[i % len(self.config.l.global_local_attention_pattern)]
                    == 'GLOBAL'
                ):
                    mask = input_mask[1]
                else:
                    mask = input_mask[0]
            elif self.config.l.model_type == 'gemma3':
                if (
                    self.config.l.global_local_attention_pattern[i % len(self.config.l.global_local_attention_pattern)]
                    == 'GLOBAL'
                ):
                    mask = input_mask[1]
                    pos_emb = pos_emb_global
                else:
                    mask = input_mask[0]
                    pos_emb = pos_emb_local
                original_mask = mask

            elif self.config.l.model_type == 'whisper_decoder' and i >= 1 and len(pos_emb) > 0:
                pos_emb.remove(pos_emb[0])

            cache_in = cache.get(layer=i)
            cache_in = memory.get(layer=i, kvs=cache_in) if self.config.l.infini_attention else cache_in
            other_masks = []
            if self.config.l.infini_attention:
                bsz, q_len, _ = hidden_states.size()
                infini_mask_shape = (bsz, self.config.l.num_attention_heads, q_len, self.config.l.head_dim)
                other_masks.append(
                    torch.ones(infini_mask_shape, dtype=hidden_states.dtype)
                    if memory.is_first_segment
                    else torch.zeros(infini_mask_shape, dtype=hidden_states.dtype)
                )
            if self.config.l.use_split_mask:
                num_token = input_embeds.shape[1]
                mask = original_mask[:, :, :, -num_token:]
                other_masks.append(original_mask[:, :, :1, :-num_token])

            cross_attn = cross_attention[i] if cross_attention != [] else []

            logger.debug(f'chunk {i}')
            logger.debug(f'hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}')
            logger.debug(f'mask shape: {mask.shape}, dtype: {mask.dtype}')
            if pos_emb != []:
                logger.debug(f'pos_emb len: {len(pos_emb)} shape: {pos_emb[0].shape}, dtype: {pos_emb[0].dtype}')
            logger.debug(f'cache len: {len(cache_in)} shape: {cache_in[0].shape}, dtype: {cache_in[0].dtype}')
            if cross_attn != []:
                logger.debug(
                    f'cross_attn len: {len(cross_attn)} shape: {cross_attn[0].shape}, dtype: {cross_attn[0].dtype}'
                )
            logger.debug(f'num_lora_inputs len: {len(self.lora_handler.llm_lora_inputs[i])}')
            if (
                self.lora_handler.has_llm_lora()
                and i >= self.lora_handler.global_llm_start_idx
                and i <= self.lora_handler.global_llm_end_idx
            ):
                logger.debug(f'num_lora_inputs dtype: {self.lora_handler.llm_lora_inputs[i][0].dtype}')

            # NEW (Patch E-fix): pre-scatter ds_padded BEFORE the calibration save block,
            # so the save_dict can include ds_padded (used by Qwen3ModelChunk.get_ptq_inputs).
            _extra_chunk_inputs = []
            _ds_embeds = kwargs.get('deepstack_visual_embeds')
            if _ds_embeds is not None and i < len(_ds_embeds):
                _img_tok_id = getattr(self.config.l, 'image_token_id', None) or (self.config.l.kwargs.get('image_token_id') if hasattr(self.config.l, 'kwargs') and isinstance(self.config.l.kwargs, dict) else None)
                if _img_tok_id is None:
                    _img_tok_id = getattr(self.config.l.kwargs, 'image_token_id', None) if hasattr(self.config.l, 'kwargs') else None
                ds_padded = torch.zeros_like(hidden_states)
                if _img_tok_id is not None and curr_tokens is not None:
                    _ct = torch.as_tensor(curr_tokens) if not isinstance(curr_tokens, torch.Tensor) else curr_tokens
                    _batch_mask = (_ct == _img_tok_id)
                    _n_in_batch = int(_batch_mask.sum().item())
                    if _n_in_batch > 0:
                        _consumed = getattr(self, '_deepstack_consumed', 0)
                        _ds_full = _ds_embeds[i]
                        _slice = _ds_full[_consumed:_consumed + _n_in_batch]
                        if _slice.shape[0] != _n_in_batch:
                            logger.warning(
                                f'[deepstack] layer {i}: slice shape {_slice.shape} != n_in_batch {_n_in_batch}; '
                                f'consumed={_consumed} ds_full_len={_ds_full.shape[0]}'
                            )
                        _slice = _slice.to(device=ds_padded.device, dtype=ds_padded.dtype)
                        _bm = _batch_mask.to(ds_padded.device)
                        if _bm.dim() == 1:
                            _bm = _bm.unsqueeze(0)
                        _T_full = ds_padded.shape[1]
                        _T_real = _bm.shape[-1]
                        if _T_real < _T_full:
                            _pad = torch.zeros(_bm.shape[0], _T_full - _T_real, dtype=torch.bool, device=_bm.device)
                            _bm = torch.cat([_pad, _bm], dim=-1)
                        elif _T_real > _T_full:
                            _bm = _bm[:, -_T_full:]
                        ds_padded[_bm] = _slice
                        if i == len(_ds_embeds) - 1:
                            self._deepstack_consumed = _consumed + _n_in_batch
                            logger.debug(
                                f'[deepstack] batch advanced cursor: {_consumed} -> {_consumed + _n_in_batch}'
                            )
                _extra_chunk_inputs.append(ds_padded)

            save_chunk = self.backend == const.CONVERTER or (self.backend == const.MLKITS and i == 0)
            if self.task == 'make_calibration' and save_chunk:
                logger.debug(f'Save LLM calibration batch (layer {i})')
                save_dict = {
                    'input_tokens': curr_tokens,
                    'inputs_embeds': hidden_states.cpu().numpy().astype(np.float32),
                    'mask': mask.cpu().numpy().astype(np.float32),
                    'past_keys': cache_in[0].cpu().numpy().astype(np.float32),
                    'past_values': cache_in[1].cpu().numpy().astype(np.float32),
                }
                # NEW (Patch E-fix): save ds_padded for chunks that have deepstack injection
                if _extra_chunk_inputs:
                    save_dict['ds_padded'] = _extra_chunk_inputs[0].cpu().numpy().astype(np.float32)
                if self.config.l.model_type == 'gecko' and i == 0:
                    assert len(pos_emb) == 3
                    save_dict.update(
                        {
                            'pos_emb': pos_emb[0].cpu().numpy().astype(np.float32),
                            'cos': pos_emb[1].cpu().numpy().astype(np.float32),
                            'sin': pos_emb[2].cpu().numpy().astype(np.float32),
                        }
                    )
                elif self.config.l.model_type == 'gecko2':
                    emb_idx = 0
                    if self.config.l.d_per_layer_embedding and (not self.is_ee or (self.is_ee and i < self.ee_index)):
                        save_dict.update(
                            {
                                'per_layer_emb': pos_emb[emb_idx].detach().cpu().numpy().astype(np.float32),
                            }
                        )
                        emb_idx += 1
                    if self.config.l.use_fire_pe_in_global and i in self.config.l.fire_pe_index:
                        save_dict.update(
                            {
                                'fire_pe_relative_pos': pos_emb[emb_idx].detach().cpu().numpy().astype(np.float32),
                                'fire_pe_query_pos': pos_emb[emb_idx + 1].detach().cpu().numpy().astype(np.float32),
                            }
                        )
                        emb_idx += 2
                    else:
                        save_dict.update(
                            {
                                'cos': pos_emb[emb_idx].cpu().numpy().astype(np.float32),
                                'sin': pos_emb[emb_idx + 1].cpu().numpy().astype(np.float32),
                            }
                        )
                        emb_idx += 2
                elif self.config.l.model_type == 'whisper_decoder':
                    if i == 0:
                        save_dict.update(
                            {
                                'pos_emb': pos_emb[0].cpu().numpy().astype(np.float32),
                            }
                        )
                elif self.config.l.extra_input['sink_rope']:
                    assert len(pos_emb) == 4
                    save_dict.update(
                        {
                            'cos': pos_emb[0].cpu().numpy().astype(np.float32),
                            'sin': pos_emb[1].cpu().numpy().astype(np.float32),
                            'k_cos': pos_emb[2].cpu().numpy().astype(np.float32),
                            'k_sin': pos_emb[3].cpu().numpy().astype(np.float32),
                        }
                    )
                else:
                    assert len(pos_emb) == 2
                    save_dict.update(
                        {
                            'cos': pos_emb[0].cpu().numpy().astype(np.float32),
                            'sin': pos_emb[1].cpu().numpy().astype(np.float32),
                        }
                    )
                other_masks_idx = 0
                if self.config.l.infini_attention:
                    assert len(cache_in) == 4
                    save_dict.update(
                        {
                            'mem': cache_in[2].cpu().numpy().astype(np.float32),
                            'infini_mask': other_masks[other_masks_idx].cpu().numpy().astype(np.float32),
                        }
                    )
                    other_masks_idx += 1
                    if memory.is_memory_updated:
                        save_dict.update(
                            {
                                'z': cache_in[3].cpu().numpy().astype(np.float32),
                            }
                        )
                    else:
                        save_dict.update(
                            {
                                # reset z back to zero for make calib
                                'z': np.zeros_like(cache_in[3].cpu().numpy()).astype(np.float32),
                            }
                        )
                if self.config.l.use_split_mask:
                    save_dict.update(
                        {
                            'split_mask': other_masks[other_masks_idx].cpu().numpy().astype(np.float32),
                        }
                    )
                    other_masks_idx += 1
                if cross_attn != []:
                    save_dict.update(
                        {
                            'cross_key': cross_attn[0].cpu().numpy().astype(np.float32),
                            'cross_value': cross_attn[1].cpu().numpy().astype(np.float32),
                        }
                    )
                np.savez(
                    os.path.join(
                        output_dirs[i],
                        'batch-{:04d}.npz'.format(len([x for x in os.listdir(output_dirs[i]) if x.endswith('.npz')])),
                    ),
                    **save_dict,
                )
                if len(lora_mapping_files) > 0:
                    lora_mapping_files[i].write(self.lora_handler.lora_config_paths[0] + '\n')

            logger.debug(f'Forward LLM (chunk {i})')
            model_out = self.llm[i](
                hidden_states,
                mask,
                *pos_emb,
                *cache_in,
                *cross_attn,
                *other_masks,
                *self.lora_handler.llm_lora_inputs[i],
                *_extra_chunk_inputs,
            )
            hidden_states = model_out[0]

            k_cache = model_out[1]
            v_cache = model_out[2]
            if pad_len > 0:
                k_cache = k_cache[:, :, :-pad_len, :]
                v_cache = v_cache[:, :, :-pad_len, :]
            curr_k.append(k_cache)
            curr_v.append(v_cache)

            if self.config.l.extra_output['attn_logits'] and self.config.l.extra_output['attn_weights']:
                kwargs['attn_logits'].append(model_out[-2])
                kwargs['attn_weights'].append(model_out[-1])
            elif self.config.l.extra_output['attn_logits']:
                kwargs['attn_logits'].append(model_out[-1])
            elif self.config.l.extra_output['attn_weights']:
                kwargs['attn_weights'].append(model_out[-1])

        kwargs['cache_evictor'] = cache
        kwargs['curr_k'] = curr_k
        kwargs['curr_v'] = curr_v

        if self.evictor is not None:
            self.evictor.update_num_token(input_embeds.shape[1])
            self.evictor.cache_insert(**kwargs)
        else:
            cache.insert(curr_k, curr_v)

        hidden_states, kwargs = self.forward_pre_tail_hook(hidden_states, **kwargs)
        logits, tail_cache, kwargs = self.forward_tail(
            hidden_states,
            curr_tokens=curr_tokens,
            mask=mask,
            pos_emb=pos_emb,
            input_embeds=input_embeds,
            tail_cache=tail_cache,
            output_dir=None if self.task != 'make_calibration' else output_dirs[-1],
            **kwargs,
        )
        return logits, cache, tail_cache, kwargs

    def generate_llm(
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
        """Generates LLM response tokens.

        Args:
            input_embeds: The input embeddings.
            input_ids: The input IDs.
            num_token: Maximum number of tokens to forward at once during prompt mode.
            cache_size: The cache size.
            stopping_criteria: The stopping criteria.
            logits_processors: The logits processors.
            logits_warper: The logits warper.
            cross_attn: Cross attention from encoder.
            kwargs (optional): Additional keyword arguments.

        Returns:
            numpy.ndarray: Generated token ids concatenated behind input token ids.
        """
        if kwargs.pop('dynamic_shape', False):
            return self.generate_llm_dynamic_shape(
                input_embeds,
                input_ids,
                num_token,
                stopping_criteria,
                logits_processors,
                logits_warper,
                cross_attn,
                **kwargs,
            )
        return self.generate_llm_fixed_shape(
            input_embeds,
            input_ids,
            num_token,
            cache_size,
            stopping_criteria,
            logits_processors,
            logits_warper,
            cross_attn,
            **kwargs,
        )

    def generate_llm_dynamic_shape(
        self,
        input_embeds,
        input_ids,
        num_token,
        stopping_criteria,
        logits_processors=None,
        logits_warper=None,
        cross_attn=None,
        **kwargs,
    ):
        """Generate LLM response using dynamic shape.

        Args:
            input_embeds: The input embeddings.
            input_ids: The input IDs.
            num_token: Maximum number of tokens to forward at once during prompt mode.
            cache_size: The cache size.
            stopping_criteria: The stopping criteria.
            logits_processors: The logits processors.
            logits_warper: The logits warper.
            cross_attn: Cross attention from encoder.
            kwargs (optional): Additional keyword arguments.

        Returns:
            numpy.ndarray: Generated token ids concatenated behind input token ids.
        """
        logger.debug('Enter generate_llm_dynamic_shape')
        input_length = input_embeds.shape[1]

        # Pad input_ids to input_length for stopping_criteria
        if input_ids.shape[1] != input_length:
            logger.debug(f'Pad stopping criteria used input_ids {input_ids.shape[1]} to input_length {input_length}')
            input_ids_for_max_length_criteria = self.pad_input_ids_to_input_embeds_len(input_ids, input_length)
        else:
            input_ids_for_max_length_criteria = input_ids

        # Init cache
        cache = cache_utils.Cache(
            self.config.l,
            1,
            self.dtype,
            mode='dynamic',
            device=self.main_device,
            overture=self.overture if kwargs.pop('use_overture', True) else None,
            bita_prefix=self.bita_prefix if self.use_bita and self.task == 'make_calibration' else None,
        )
        tail_cache = (
            None
            if self.config.t is None or (self.config.t is not None and self.config.t.model_type != 'medusa')
            else cache_utils.Cache(self.config.t, 1, self.dtype, mode='dynamic', device=self.main_device)
        )

        output_dirs = [] if self.task != 'make_calibration' else kwargs.get('output_dirs')['llm']
        calib_max_batches = 0 if self.task != 'make_calibration' else kwargs.get('max_batches')

        seq_length = 0

        master_rot_emb = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)

        if self.config.l.model_type in ['gecko', 'whisper_decoder']:
            master_pos_emb = generate_utils.get_master_pos_emb(
                self.config.l, self.dtype, state_dict=self.state_dict['llm']
            )

        firelite_pos_emb = None
        if self.config.l.model_type == 'gecko2':
            firelite_pos_emb = generate_utils.get_firelite_pos_emb(
                self.config.l,
                self.dtype,
                length=max(self.config.l.max_position_embeddings, cache.cache_size + num_token),
                state_dict=self.state_dict['llm'],
            )
        # local pos emb for gemma3
        elif self.config.l.model_type == 'gemma3':
            self.config.l.rotary_emb_base = self.config.l.rope_local_base_freq
            self.config.l.rope_scaling['factor'] = 1
            master_rot_emb_local = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)
        prompt_mode = True
        num_prompt_passes = int((input_length - seq_length) / num_token) + int(
            (input_length - seq_length) % num_token != 0
        )
        logger.debug(f'num_prompt_passes={num_prompt_passes}')

        eos_id = (
            [self.config.l.eos_token_id] if isinstance(self.config.l.eos_token_id, int) else self.config.l.eos_token_id
        )

        while True:
            if prompt_mode:
                num_token = min(num_token, input_length - seq_length)
                logger.debug(f'Prompt mode, num_token={num_token}')
                curr_input_ids = (
                    input_ids[:, :-1]  # MLLM, input_ids len is shorter than input_embeds
                    if num_token > input_ids.shape[1]
                    else input_ids[:, seq_length : seq_length + num_token]  # LLM only
                )
                curr_input_embeds = input_embeds[:, seq_length : seq_length + num_token, :]
            else:
                logger.debug('Gen mode')
                curr_input_ids = input_ids[:, -num_token:]

            # Calculate BiTA draft length for current step
            current_bita_draft_len = 0
            if self.use_bita and self.task == 'make_calibration':
                current_bita_draft_len = self.bita_draft_length - 1 if seq_length == 0 else 1

            mask = generate_utils.generate_mask(
                cache.cache_size,
                min(cache.overture_size + seq_length, cache.cache_size),
                num_token,
                num_token,
                mask_value=self.config.l.mask_value,
                dtype=self.dtype,
                sliding_window=self.config.l.sliding_window_attention_size != 0,
                sliding_window_size=self.config.l.sliding_window_attention_size,
                bita_prefix_length=self.bita_prefix_length if self.use_bita else 0,
                bita_draft_length=current_bita_draft_len,
            )

            if cache.overture_size + seq_length + num_token > master_rot_emb.shape[2]:
                logger.error('Prompt + Response total length exceeds maximum positional embedding length.')

            pos_emb = (
                [master_pos_emb[:, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :]]
                if self.config.l.model_type in ['gecko', 'whisper_decoder']
                else []
            )

            if self.config.l.model_type == 'gecko2':
                if self.config.l.d_per_layer_embedding:
                    all_layer_emb = kwargs['per_layer_embedding'][:, seq_length : seq_length + num_token, ...]
                    pos_emb.append(all_layer_emb)
                if self.config.l.use_fire_pe_in_global:
                    cur_pos = seq_length + cache.overture_size
                    fire_relative_pos, fire_query_pos = firelite_pos_emb
                    cur_fire_relative_pos = fire_relative_pos[
                        :, :, cache.cache_size : cache.cache_size + num_token, : cache.cache_size + num_token
                    ]
                    cur_fire_query_pos = fire_query_pos[:, :, cur_pos : cur_pos + num_token, :]
                    pos_emb.append(
                        [cur_fire_relative_pos, cur_fire_query_pos]
                    )  # append and handle in forward llm float

                # gecko2 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache.cache_size,
                    min(cache.overture_size + seq_length, cache.cache_size),
                    num_token,
                    num_token,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)
            elif self.config.l.model_type == 'gemma3':
                # gemma3 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache.cache_size,
                    min(cache.overture_size + seq_length, cache.cache_size),
                    num_token,
                    num_token,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)

            if self.config.l.model_type != 'whisper_decoder':
                # dynamic rope
                if self.config.l.extra_input['sink_rope']:
                    start_idx = cache.overture_size + seq_length
                    q_end_idx = cache.overture_size + seq_length + num_token
                    k_end_idx = max(cache.overture_size, 1) + seq_length + num_token
                    q_cos = master_rot_emb[:, :1, start_idx:q_end_idx, :]
                    q_sin = master_rot_emb[:, 1:, start_idx:q_end_idx, :]
                    k_cos = master_rot_emb[:, :1, :k_end_idx, :]
                    k_sin = master_rot_emb[:, 1:, :k_end_idx, :]
                    if self.use_single_bmm_attention:
                        if isinstance(q_cos, np.ndarray):  # noqa: SIM108
                            new_shape = (0, 2, 1, 3)
                        else:
                            new_shape = (2, 1)
                        q_cos = q_cos.transpose(*new_shape)
                        q_sin = q_sin.transpose(*new_shape)
                        k_cos = k_cos.transpose(*new_shape)
                        k_sin = k_sin.transpose(*new_shape)
                    pos_emb.extend([q_cos, q_sin, k_cos, k_sin])
                else:
                    cos = master_rot_emb[
                        :, :1, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :
                    ]
                    sin = master_rot_emb[
                        :, 1:, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :
                    ]
                    if self.use_single_bmm_attention:
                        if isinstance(cos, np.ndarray):  # noqa: SIM108
                            new_shape = (0, 2, 1, 3)
                        else:
                            new_shape = (2, 1)
                        cos = cos.transpose(*new_shape)
                        sin = sin.transpose(*new_shape)
                    pos_emb.extend([cos, sin])

            # gemma3 local pos_emb
            if self.config.l.model_type == 'gemma3':
                cos = master_rot_emb_local[
                    :, :1, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :
                ]
                sin = master_rot_emb_local[
                    :, 1:, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :
                ]
                if self.use_single_bmm_attention:
                    if isinstance(cos, np.ndarray):  # noqa: SIM108
                        new_shape = (0, 2, 1, 3)
                    else:
                        new_shape = (2, 1)
                    cos = cos.transpose(*new_shape)
                    sin = sin.transpose(*new_shape)
                pos_emb.extend([cos, sin])

            if prompt_mode:
                hidden_states = curr_input_embeds
            else:
                if self.config.l.model_type == 'gecko2':
                    # needs previous token
                    hidden_states = self.text_embedding_layer(torch.tensor(input_ids[:, -2:]).to(self.main_device))
                    hidden_states, per_layer_embedding = hidden_states
                    hidden_states = hidden_states[:, -1:, :]
                    if self.config.l.d_per_layer_embedding:
                        pos_emb[0] = per_layer_embedding[:, -1:, ...]
                else:
                    hidden_states = self.text_embedding_layer(torch.tensor(curr_input_ids).to(self.main_device))
                hidden_states = hidden_states.to(self.dtype)
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.from_numpy(hidden_states).to(self.dtype)

            logits, cache, tail_cache, kwargs = self.forward_llm_float(
                hidden_states,
                mask,
                pos_emb,
                cache,
                curr_tokens=curr_input_ids,
                cross_attn=cross_attn,
                tail_cache=tail_cache,
                **kwargs,
            )
            if kwargs.pop('return_logits', False):
                logger.debug('Return logits only')
                return logits
            seq_length = seq_length + num_token
            logger.debug(f'seq_length={seq_length}')

            if seq_length < input_length:
                # Prompt mode not done, discard logits and continue processing prompt
                continue
            # Prompt mode done, switch to generative mode
            num_token = 1
            prompt_mode = False

            next_token = self.sampler(logits, input_ids, logits_processors, logits_warper)

            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1)
            input_ids_for_max_length_criteria = np.concatenate(
                [input_ids_for_max_length_criteria, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1
            )

            # stop when eos or if we exceed the maximum length or if max calibration batches saved
            if next_token[0].item() in eos_id:
                logger.debug('EOS token predicted. Stop generation.')
                break
            if stopping_criteria(torch.tensor(input_ids_for_max_length_criteria), ()):
                logger.debug('Max length reached. Stop generation.')
                break
            if (
                self.task == 'make_calibration'
                and len([x for x in os.listdir(output_dirs[0]) if x.endswith('.npz')]) >= calib_max_batches
            ):
                logger.debug('Max calibration batches generated. Stop generation.')
                break
            if self.task == 'make_calibration' and not prompt_mode:
                logger.debug('2 calibration batches saved for this line. Stop generation.')
                break
        return input_ids, kwargs

    def generate_llm_fixed_shape(
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
        """Generate LLM response using fixed shape.

        Args:
            input_embeds: The input embeddings.
            input_ids: The input IDs.
            num_token: The number of tokens.
            cache_size: The cache size.
            stopping_criteria: The stopping criteria.
            logits_processors: The logits processors.
            logits_warper: The logits warper.
            cross_attn: Cross attention from encoder.
            kwargs (optional): Additional keyword arguments.

        Returns:
            numpy.ndarray: Generated token ids concatenated behind input token ids.
        """
        logger.debug('Enter generate_llm_fixed_shape')
        input_length = input_embeds.shape[1]
        logger.debug(f'Prompt input_length={input_length}')

        memory = None
        if self.config.l.infini_attention:
            memory = kwargs.pop('memory')

        # Pad input_ids to input_length for stopping_criteria
        if input_ids.shape[1] != input_length:
            logger.debug(f'Pad stopping criteria used input_ids {input_ids.shape[1]} to input_length {input_length}')
            input_ids_for_max_length_criteria = self.pad_input_ids_to_input_embeds_len(input_ids, input_length)
        else:
            input_ids_for_max_length_criteria = input_ids

        # Init cache
        cache = cache_utils.Cache(
            self.config.l,
            cache_size,
            self.dtype,
            device=self.main_device,
            overture=self.overture if kwargs.pop('use_overture', True) else None,
        )
        tail_cache = (
            None
            if self.config.t is None or (self.config.t is not None and self.config.t.model_type != 'medusa')
            else cache_utils.Cache(self.config.t, cache_size, self.dtype, device=self.main_device)
        )

        # Init evictor prompt length
        if self.evictor is not None:
            self.evictor.update_prompt_length(input_length, cache.overture_size)

        output_dirs = [] if self.task != 'make_calibration' else kwargs.get('output_dirs')['llm']
        calib_max_batches = 0 if self.task != 'make_calibration' else kwargs.get('max_batches')

        seq_length = 0

        master_rot_emb = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)

        if self.config.l.model_type in ['gecko', 'whisper_decoder']:
            master_pos_emb = generate_utils.get_master_pos_emb(
                self.config.l, self.dtype, state_dict=self.state_dict['llm']
            )
        # local pos emb for gemma3
        elif self.config.l.model_type == 'gemma3':
            self.config.l.rotary_emb_base = self.config.l.rope_local_base_freq
            self.config.l.rope_scaling['factor'] = 1
            master_rot_emb_local = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)

        firelite_pos_emb = None
        if self.config.l.model_type == 'gecko2':
            firelite_pos_emb = generate_utils.get_firelite_pos_emb(
                self.config.l,
                self.dtype,
                length=max(self.config.l.max_position_embeddings, cache.cache_size + num_token),
                state_dict=self.state_dict['llm'],
            )

        prompt_mode = True
        if self.config.l.infini_attention:
            memory.set_prompt_mode(prompt_mode)

        num_prompt_passes = int((input_length - seq_length) / num_token) + int(
            (input_length - seq_length) % num_token != 0
        )
        logger.debug(f'num_prompt_passes={num_prompt_passes}')
        pad_len = (
            int(num_token - (input_length - seq_length) % num_token)
            if (input_length - seq_length) % num_token != 0
            else 0
        )
        logger.debug(f'pad_len={pad_len}')
        if self.config.l.infini_attention and kwargs.get('return_logits'):
            final_logits = torch.zeros((1, 0, self.config.l.vocab_size), dtype=self.dtype)

        eos_id = (
            [self.config.l.eos_token_id] if isinstance(self.config.l.eos_token_id, int) else self.config.l.eos_token_id
        )

        while True:
            if pad_len > 0:
                logger.debug(f'First prompt pass. Right pad curr_input_embeds by {pad_len}')
                curr_input_ids = input_ids[:, : num_token - pad_len]
                curr_input_embeds = input_embeds[:, : num_token - pad_len, :]
                curr_input_embeds = torch.nn.functional.pad(curr_input_embeds, ((0, 0, 0, pad_len)))
            else:
                if prompt_mode:
                    logger.debug('Prompt mode')
                    curr_input_ids = input_ids[:, seq_length : seq_length + num_token]
                    curr_input_embeds = input_embeds[:, seq_length : seq_length + num_token, :]
                else:
                    logger.debug('Gen mode')
                    curr_input_ids = input_ids[:, -num_token:]

            if self.config.l.infini_attention:
                # get from memory
                cur_cache_size = memory.cur_cache_size
                valid_cache_size = memory.get_valid_cache_size(overture_size=cache.overture_size)
            elif self.evictor is not None:
                cur_cache_size = cache_size
                valid_cache_size = min(cache.offset, cache_size) if not cache.is_full else cache_size
            else:
                cur_cache_size = cache_size
                valid_cache_size = min(cache.overture_size + seq_length, cache_size)

            if self.task == 'make_calibration' and self.backend == const.MLKITS:
                logger.debug('Save calibration data without valid cache for MLKits.')
                valid_cache_size = 0

            mask = generate_utils.generate_mask(
                cur_cache_size,
                valid_cache_size,
                num_token,
                num_token - pad_len,
                mask_value=self.config.l.mask_value,
                dtype=self.dtype,
                sliding_window=self.config.l.sliding_window_attention_size != 0,
                sliding_window_size=self.config.l.sliding_window_attention_size,
            )

            if cache.overture_size + seq_length + num_token > master_rot_emb.shape[2]:
                logger.error('Prompt + Response total length exceeds maximum positional embedding length.')

            pos_emb = (
                [master_pos_emb[:, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :]]
                if self.config.l.model_type in ['gecko', 'whisper_decoder']
                else []
            )

            if self.config.l.model_type == 'gecko2':
                if self.config.l.d_per_layer_embedding:
                    if pad_len > 0:
                        all_layer_emb = kwargs['per_layer_embedding'][:, : num_token - pad_len, :, :]
                        all_layer_emb = torch.nn.functional.pad(all_layer_emb, ((0, 0, 0, 0, 0, pad_len)))
                    else:
                        if prompt_mode:
                            all_layer_emb = kwargs['per_layer_embedding'][:, seq_length : seq_length + num_token, :, :]
                    pos_emb.append(all_layer_emb)
                if self.config.l.use_fire_pe_in_global:
                    cur_pos = seq_length + cache.overture_size
                    fire_relative_pos, fire_query_pos = firelite_pos_emb
                    cur_fire_relative_pos = fire_relative_pos[
                        :, :, cache.cache_size : cache.cache_size + num_token, : cache.cache_size + num_token
                    ]
                    cur_fire_query_pos = fire_query_pos[:, :, cur_pos : cur_pos + num_token, :]
                    pos_emb.append(
                        [cur_fire_relative_pos, cur_fire_query_pos]
                    )  # append and handle in forward llm float

                # gecko2 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache_size,
                    min(cache.overture_size + seq_length, cache_size),
                    num_token,
                    num_token - pad_len,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)
            elif self.config.l.model_type == 'gemma3':
                # gemma3 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache.cache_size,
                    min(cache.overture_size + seq_length, cache.cache_size),
                    num_token,
                    num_token,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)

            if self.config.l.model_type != 'whisper_decoder':
                # static rope
                if self.config.l.extra_input['sink_rope']:
                    start_idx = cache.overture_size + seq_length
                    end_idx = cache.overture_size + seq_length + num_token
                    q_cos = master_rot_emb[
                        :,
                        :1,
                        min(start_idx, cache.cache_size) : min(end_idx, cache.cache_size + num_token),
                        :,
                    ]
                    q_sin = master_rot_emb[
                        :,
                        1:,
                        min(start_idx, cache.cache_size) : min(end_idx, cache.cache_size + num_token),
                        :,
                    ]
                    k_rot_emb = master_rot_emb[:, :, : min(start_idx, cache.cache_size) + num_token, :]
                    k_rot_emb = torch.nn.functional.pad(k_rot_emb, (0, 0, max(cache.cache_size - start_idx, 0), 0))
                    k_cos = k_rot_emb[:, :1, :, :]
                    k_sin = k_rot_emb[:, 1:, :, :]
                    if self.use_single_bmm_attention:
                        if isinstance(q_cos, np.ndarray):  # noqa: SIM108
                            new_shape = (0, 2, 1, 3)
                        else:
                            new_shape = (2, 1)
                        q_cos = q_cos.transpose(*new_shape)
                        q_sin = q_sin.transpose(*new_shape)
                        k_cos = k_cos.transpose(*new_shape)
                        k_sin = k_sin.transpose(*new_shape)
                    pos_emb.extend([q_cos, q_sin, k_cos, k_sin])
                else:
                    if self.config.l.infini_attention:
                        # currently infini_attention uses local
                        # get from memory
                        start_position = memory.get_start_position(overture_size=cache.overture_size)
                        end_position = start_position + num_token
                        logger.debug(f'pos emb start position: {start_position}')
                    else:
                        start_position = cache.overture_size + seq_length
                        end_position = cache.overture_size + seq_length + num_token
                    cos = master_rot_emb[:, :1, start_position:end_position, :]
                    sin = master_rot_emb[:, 1:, start_position:end_position, :]
                    if self.use_single_bmm_attention:
                        if isinstance(cos, np.ndarray):  # noqa: SIM108
                            new_shape = (0, 2, 1, 3)
                        else:
                            new_shape = (2, 1)
                        cos = cos.transpose(*new_shape)
                        sin = sin.transpose(*new_shape)
                    pos_emb.extend([cos, sin])
            # gemma3 local pos_emb
            if self.config.l.model_type == 'gemma3':
                start_position = cache.overture_size + seq_length
                end_position = cache.overture_size + seq_length + num_token
                cos = master_rot_emb_local[:, :1, start_position:end_position, :]
                sin = master_rot_emb_local[:, 1:, start_position:end_position, :]
                if self.use_single_bmm_attention:
                    if isinstance(cos, np.ndarray):  # noqa: SIM108
                        new_shape = (0, 2, 1, 3)
                    else:
                        new_shape = (2, 1)
                    cos = cos.transpose(*new_shape)
                    sin = sin.transpose(*new_shape)
                pos_emb.extend([cos, sin])

            if prompt_mode:
                hidden_states = curr_input_embeds
            else:
                if self.config.l.model_type == 'gecko2':
                    # needs previous token
                    hidden_states = self.text_embedding_layer(torch.tensor(input_ids[:, -2:]).to(self.main_device))
                    hidden_states, per_layer_embedding = hidden_states
                    hidden_states = hidden_states[:, -1:, :]
                    if self.config.l.d_per_layer_embedding:
                        pos_emb[0] = per_layer_embedding[:, -1:, ...]
                else:
                    hidden_states = self.text_embedding_layer(torch.tensor(curr_input_ids).to(self.main_device))
                hidden_states = hidden_states.to(self.dtype)
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.from_numpy(hidden_states).to(self.dtype)

            logits, cache, tail_cache, kwargs = self.forward_llm_float(
                hidden_states,
                mask,
                pos_emb,
                cache,
                pad_len,
                curr_tokens=curr_input_ids,
                cross_attn=cross_attn,
                tail_cache=tail_cache,
                memory=memory,
                **kwargs,
            )

            # update memory
            if self.config.l.infini_attention:
                is_updated = False
                cur_kvs = cache.get()
                is_updated = memory.insert_and_update(cur_kvs=cur_kvs, num_token=num_token - pad_len)
                # no need to manually clear cache as can just set the mask to 0
                logger.debug(f'Is memory updated: {is_updated}')

            if pad_len > 0:
                logits = logits[:, :-pad_len, :]

            if not self.config.l.infini_attention and kwargs.pop('return_logits', False):
                logger.debug('Return logits only')
                return logits
            seq_length = seq_length + num_token - pad_len
            logger.debug(f'seq_length={seq_length}')

            if self.config.l.infini_attention and kwargs.get('return_logits'):
                final_logits = torch.concatenate([final_logits, logits.detach().cpu()], dim=1)
            if (
                self.task == 'make_calibration'
                and len([x for x in os.listdir(output_dirs[0]) if x.endswith('.npz')]) >= calib_max_batches
            ):
                logger.debug('Max calibration batches generated. Stop generation.')
                break

            kwargs['seq_length'] = seq_length
            if self.evictor is not None:
                self.evictor.update_attn(**kwargs)
                if self.evictor.trigger(seq_length):
                    # update cache
                    self.evictor.compress(cache, seq_length)

            pad_len = 0  # Only first pass will ever need padding
            if seq_length < input_length:
                # Prompt mode not done, discard logits and continue processing prompt
                continue
            # Prompt mode done, switch to generative mode
            num_token = 1
            prompt_mode = False

            logits_to_sample = (
                input_ids if self.config.l.model_type == 'whisper_decoder' else input_ids[:, input_length:]
            )
            next_token = self.sampler(logits, logits_to_sample, logits_processors, logits_warper)

            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1)
            input_ids_for_max_length_criteria = np.concatenate(
                [input_ids_for_max_length_criteria, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1
            )

            # stop when eos or if we exceed the maximum length or if max calibration batches saved
            if next_token[0].item() in eos_id:
                logger.debug('EOS token predicted. Stop generation.')
                break
            if stopping_criteria(torch.tensor(input_ids_for_max_length_criteria), ()):
                logger.debug('Max length reached. Stop generation.')
                break
            if self.task == 'make_calibration' and self.backend == const.MLKITS:
                logger.debug('Prompt calibration batches saved for this line. Stop generation.')
                break

            # FIXME: change to better way in future for different cache sizes
            # prepare cache again for different cache sizes
            if self.config.l.infini_attention:
                memory.set_prompt_mode(prompt_mode)
                if not memory.update_during_gen and cache.cache_size != memory.gen_cache_size:
                    logger.info('Changing cache due to different sizes between prompt and gen.')
                    cur_cache = cache.get()
                    del cache
                    cur_k = []
                    cur_v = []
                    for cache_idx in range(0, len(cur_cache), 2):
                        cur_k.append(cur_cache[cache_idx])
                        cur_v.append(cur_cache[cache_idx + 1])
                    cache = cache_utils.Cache(
                        self.config.l,
                        memory.gen_cache_size,
                        self.dtype,
                        device=self.main_device,
                        overture=self.overture if kwargs.pop('use_overture', True) else None,
                    )
                    cache.insert(cur_k, cur_v)

        if self.config.l.infini_attention and kwargs.pop('return_logits', False):
            logger.debug('Return logits only for infini attention')
            return final_logits
        return input_ids, kwargs

    def forward_tail(
        self,
        hidden_states,
        curr_tokens=None,
        mask=None,
        pos_emb=None,
        input_embeds=None,
        tail_cache=None,
        output_dir=None,
        **kwargs,
    ):
        """Forward pass of LLM tail.

        Args:
            hidden_states: The input hidden states.
            curr_tokens: The current input token ids corresponding to the input_embeds.
            mask: The combined attention+cache mask.
            pos_emb: List of positional embedding components.
            input_embeds: The input embeddings to the first decoder layer.
            tail_cache: The input cache object for the EAGLE tail if using EAGLE model.
            output_dir: The output directory to save calibration dataset to if making calibration dataset.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor. logits output of LLM.
            None or Cache object. Updated cache of EAGLE tail if using EAGLE model, else None.
        """
        logger.debug('Enter forward_tail')
        save_tail = self.task == 'make_calibration' and self.backend == const.CONVERTER
        if self.config.t is not None and self.config.t.model_type != 'medusa':
            logger.debug('EAGLE tail')
            tail_cache_in = tail_cache.get()
            if save_tail:
                assert output_dir is not None
                if self.config.l.extra_input['sink_rope']:
                    np.savez(
                        os.path.join(
                            output_dir,
                            'batch-{:04d}.npz'.format(len([x for x in os.listdir(output_dir) if x.endswith('.npz')])),
                        ),
                        input_tokens=curr_tokens,
                        inputs_embeds=hidden_states.cpu().numpy().astype(np.float32),
                        mask=mask.cpu().numpy().astype(np.float32),
                        cos=pos_emb[0].cpu().numpy().astype(np.float32),
                        sin=pos_emb[1].cpu().numpy().astype(np.float32),
                        k_cos=pos_emb[2].cpu().numpy().astype(np.float32),
                        k_sin=pos_emb[3].cpu().numpy().astype(np.float32),
                        past_keys=tail_cache_in[0].cpu().numpy().astype(np.float32),
                        past_values=tail_cache_in[1].cpu().numpy().astype(np.float32),
                    )
                else:
                    np.savez(
                        os.path.join(
                            output_dir,
                            'batch-{:04d}.npz'.format(len([x for x in os.listdir(output_dir) if x.endswith('.npz')])),
                        ),
                        input_tokens=curr_tokens,
                        inputs_embeds=hidden_states.cpu().numpy().astype(np.float32),
                        mask=mask.cpu().numpy().astype(np.float32),
                        pos_emb=pos_emb[0].cpu().numpy().astype(np.float32),
                        past_keys=tail_cache_in[0].cpu().numpy().astype(np.float32),
                        past_values=tail_cache_in[1].cpu().numpy().astype(np.float32),
                    )
            model_out = self.tail.forward_alt(
                torch.from_numpy(input_embeds),
                torch.from_numpy(hidden_states),
                torch.from_numpy(mask),
                *[torch.from_numpy(x) for x in pos_emb],
                torch.from_numpy(tail_cache_in[0]),
                torch.from_numpy(tail_cache_in[1]),
            )
            logits = model_out[0]
            tail_cache.insert(model_out[1], model_out[2])
        else:
            logger.debug('No custom tail or Medusa tail')
            if save_tail:
                assert output_dir is not None
                np.savez(
                    os.path.join(output_dir, 'batch-{:04d}.npz'.format(len(os.listdir(output_dir)))),
                    input_tokens=curr_tokens,
                    hidden_states=hidden_states.cpu().numpy().astype(np.float32),
                )
            logits = self.tail(hidden_states)
        return logits[..., : self.config.l.vocab_size], tail_cache, kwargs  # Remove padding for models like minicpm


class QuantizedPipeline(BasePipeline):
    """QuantizedPipeline class for handling the pipeline of a quantized model."""

    def __init__(
        self,
        combined_config,
        lora_configs,
        task,
        quantized_model_folders=None,
        *args,
        **kwargs,
    ):
        """Initialize the QuantizedPipeline.

        Args:
            combined_config (dict): The combined configuration.
            lora_configs (list): The list of LoRA configurations.
            task (str): The main task this pipeline is created for.
            quantized_model_folders (dict): Dictionary containing encoder, LLM prompt, and LLM gen mode folders.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        from ..utils.precision_config import PTQPrecisionConfig

        logger.debug('Initialize QuantizedPipeline')
        super().__init__(combined_config, lora_configs, task, *args, **kwargs)
        self.config.override_hooks_for_quantized_pipeline()  # Force pre-projector hook to use Passthrough

        if self.task not in const.QUANTIZED_PIPELINE_TASKS:
            logger.error(
                f'Unsupported quantized pipeline task: {self.task}. '
                f'All supported quantized pipeline tasks: {const.QUANTIZED_PIPELINE_TASKS}',
                err=ValueError,
            )

        self.encoder_weight_dir = getattr(self.config.e, 'weight_dir', None)
        logger.debug(f'encoder_weight_dir={self.encoder_weight_dir}')
        self.llm_weight_dir = self.config.l.weight_dir
        logger.debug(f'llm_weight_dir={self.llm_weight_dir}')
        self.tail_weight_dir = getattr(self.config.t, 'weight_dir', None)
        logger.debug(f'tail_weight_dir={self.tail_weight_dir}')

        self.preprocessor_class = utils.resolve_preprocessor_class(self.config.p)
        self.encoder_chunk_class = utils.resolve_encoder_class(self.config.e)
        self.llm_chunk_class = utils.resolve_llm_class(self.config.l)

        # NEW (Patch E): same bridge as FloatPipeline
        _ds_indexes = getattr(self.config.e, 'deepstack_visual_indexes', None) if self.config.e is not None else None
        self.config.l._num_deepstack_inject = len(_ds_indexes) if _ds_indexes else 0

        self.ee_index = self.config.l.early_exit_index
        self.is_ee = self.ee_index is not None
        self.exp_name = None
        self.dtype = np.float32

        if quantized_model_folders['prompt'] is None:
            # bleu
            self.modes = 'gen'
        elif quantized_model_folders['generative'] is None:
            # ppl/logits
            self.modes = 'prompt'
        else:
            # bleu/inference
            self.modes = 'both'

        if self.task == 'inference' and self.modes != 'both':
            logger.error('Both prompt and generative models cannot both be None when task=inference.')

        logger.debug(f'task={self.task}, modes={self.modes}, dtype={self.dtype}')

        self.encoder_quantized_model_infos = []
        self.llm_prompt_quantized_model_infos = []
        self.llm_gen_quantized_model_infos = []
        self.prompt_num_tokens = 0
        self.cache_size = 0

        self.separate_tail = False
        self.num_separate_tails = 0

        self.use_json = self.kwargs.pop('use_json', False)

        if self.input_mode != 'embeddings':
            self.init_preprocessor()
            self.init_encoder(quantized_model_folders['encoder'], load=self.task != 'export_lora')
        self.init_llm(
            quantized_model_folders['prompt'], quantized_model_folders['generative'], load=self.task != 'export_lora'
        )

        logger.debug(f'separate_tail: {self.separate_tail}')
        logger.debug(f'num_separate_tails: {self.num_separate_tails}')
        if self.llm_gen is not None:
            logger.debug(f'llm_gen:\n{self.llm_gen}')
        logger.debug(f'llm_prompt:\n{self.llm_prompt}')

        self.init_hooks()
        self.deduce_num_layers_per_chunk()
        self.init_lora()
        self.lora_handler.load_chunked_lora_inputs(self)

        precision_dict = (
            self.llm_gen_quantized_model_infos[0]['precision_config']
            if self.modes != 'prompt'
            else self.llm_prompt_quantized_model_infos[0]['precision_config']
        )
        precision_config = PTQPrecisionConfig(self.config, precision_dict)
        self.set_precision_config(precision_config)

        self._debug_dir = None

        self._pipeline_type = 'quantized'

        self.evict_config = kwargs.pop('evict_config', {})
        self.init_evictor(self.config.l, **self.evict_config)

    def _dump_np_and_bin(self, path, fname, arr):
        logger.debug(f'Dumping numpy and bin to {path}/{fname}.npy/bin')
        np.save(os.path.join(path, f'{fname}.npy'), arr)
        arr.tofile(os.path.join(path, f'{fname}.bin'))

    def init_encoder(self, encoder_folder, load=True):
        """Initialize the encoder.

        Args:
            encoder_folder (str): Quantized encoder folder.
            load (bool): Whether to load the quantized models into Executors.
        """
        logger.debug('Enter init_encoder')
        if encoder_folder is None:
            self.num_encoder_layers = 0
            return

        enc_quantized_model_paths = [
            os.path.join(encoder_folder, x)
            for x in os.listdir(encoder_folder)
            if x.endswith(('.tflite', '.mlir'))
        ]
        for enc_quantized_model_path in enc_quantized_model_paths:
            self.encoder_quantized_model_infos.append(
                quantized_model_utils.extract_encoder_quantized_model_info(enc_quantized_model_path)
            )
        if load:
            logger.info('Loading encoder model:')
            self.encoder = quantized_model_utils.load_quantized_models(encoder_folder)

        self.num_encoder_layers = sum(x['num_layers'] for x in (self.encoder_quantized_model_infos))

    def init_projector(self):
        """Do nothing. Projector should be part of encoder for QuantizedPipeline."""

    def init_llm(self, prompt_folder, gen_folder, load=True):
        """Initialize the quantized LLM models.

        Args:
            prompt_folder (str): Quantized LLM prompt folder.
            gen_folder (str): Quantized LLM generative folder. Must be 1t.
            load (bool): Whether to load the quantized models into Executors.
        """
        if self.modes != 'prompt':
            gen_quantized_model_paths = utils.get_sorted_path_list(
                gen_folder, ext='.json' if self.use_json else ['.tflite', '.mlir']
            )
            for gen_quantized_model_path in gen_quantized_model_paths:
                if self.use_json:
                    self.llm_gen_quantized_model_infos.append(
                        utils.get_quantized_model_info_json(gen_quantized_model_path)
                    )
                else:
                    self.llm_gen_quantized_model_infos.append(
                        quantized_model_utils.extract_llm_quantized_model_info(gen_quantized_model_path)
                    )
            if load:
                logger.info('Loading generative model(s):')
                self.llm_gen = quantized_model_utils.load_quantized_models(gen_folder)
        else:
            self.llm_gen = None

        if self.modes == 'gen':
            self.llm_prompt_quantized_model_infos = self.llm_gen_quantized_model_infos
            self.llm_prompt = self.llm_gen
        else:
            prompt_quantized_model_paths = utils.get_sorted_path_list(
                prompt_folder, ext='.json' if self.use_json else ['.tflite', '.mlir']
            )
            for prompt_quantized_model_path in prompt_quantized_model_paths:
                if self.use_json:
                    self.llm_prompt_quantized_model_infos.append(
                        utils.get_quantized_model_info_json(prompt_quantized_model_path)
                    )
                else:
                    self.llm_prompt_quantized_model_infos.append(
                        quantized_model_utils.extract_llm_quantized_model_info(prompt_quantized_model_path)
                    )
            if load:
                logger.info('Loading prompt model(s):')
                self.llm_prompt = quantized_model_utils.load_quantized_models(prompt_folder)

        self.num_decoder_layers = sum(
            x['num_layers']
            for x in (
                self.llm_prompt_quantized_model_infos if self.modes != 'gen' else self.llm_gen_quantized_model_infos
            )
        )
        logger.debug(f'num_decoder_layers={self.num_decoder_layers}')

        self.exp_name = os.path.basename(prompt_folder.rstrip('/'))
        self.prompt_num_tokens = self.llm_prompt_quantized_model_infos[0]['t']
        self.cache_size = self.llm_prompt_quantized_model_infos[0]['c']

        self.separate_tail = self.llm_prompt_quantized_model_infos[-1]['tail'] is not None
        if self.separate_tail and load:
            self.init_tail()
        logger.debug(f'prompt_num_tokens={self.prompt_num_tokens}, cache_size={self.cache_size}')

        if load:
            if self.config.l.model_type == 'gecko2':
                self.text_embedding_layer = utils.get_embedding_layer(
                    self.config.l, weight_dir=self.config.l.weight_dir
                ).to(self.main_device)
            else:
                self.text_embedding_layer = utils.get_embedding_bin(self.config.l, prompt_folder).to(self.main_device)

        if self.config.l.overture_dict is not None:
            self.overture = overture_utils.get_quantized_model_overture(prompt_folder)
            if self.overture is None:
                logger.error('Overture is enabled but Overture file is not found.')

            logger.warning("Tokenizer's `add_bos` attribute forcibly set to False for Overture feature.")
            self.set_tokenizer_add_bos(False)

    def init_tail(self):
        """Initialize tail if separate tail."""
        logger.debug('Enter init_tail')
        if self.llm_prompt_quantized_model_infos[-1]['tail'] not in [*const.SUPPORTED_CUSTOM_TAILS, 'tail']:
            logger.debug('No separate tail')
            self.tail_prompt = None
            self.tail_gen = None
            return
        if self.llm_prompt_quantized_model_infos[-1]['tail'] != 'tail':
            logger.error(
                'Medusa and EAGLE tails are temporarily not supported in the current version.', err=NotImplementedError
            )
        if self.llm_prompt_quantized_model_infos[-1]['tail'] in const.SUPPORTED_CUSTOM_TAILS:
            self.num_separate_tails = 2
        else:
            self.num_separate_tails = 1
        logger.debug('Init separate tail')
        # llm_gen can be None but llm_prompt will always have value
        if self.task == 'inference':
            logger.debug('Get prompt and gen tail')
            # inference will always have both prompt and gen model
            self.tail_prompt = self.llm_prompt[-self.num_separate_tails :]
            self.tail_gen = self.llm_gen[-self.num_separate_tails :]
            del self.llm_prompt[-self.num_separate_tails :]
            del self.llm_gen[-self.num_separate_tails :]
        elif self.task == 'evaluate':
            if self.modes != 'prompt':
                self.tail_gen = self.llm_gen[-self.num_separate_tails :]  # switch tail to llm_gen tail later
                del self.llm_gen[-self.num_separate_tails :]
                if self.modes == 'both':
                    logger.debug('Get prompt and gen tail')
                    self.tail_prompt = self.llm_prompt[-self.num_separate_tails :]
                    del self.llm_prompt[-self.num_separate_tails :]
                else:
                    logger.debug('Get gen tail only')
                    self.tail_prompt = None
            elif self.modes == 'prompt':
                logger.debug('Get prompt tail only')
                self.tail_prompt = self.llm_prompt[-self.num_separate_tails :]
                del self.llm_prompt[-self.num_separate_tails :]
                self.tail_gen = None
        logger.debug(f'tail_prompt:\n{self.tail_prompt}')
        logger.debug(f'tail_gen:\n{self.tail_gen}')

    def init_lora(self):
        """Use ModelChunk classes to load the lora state dict mappings so that LoRAs can be loaded correctly."""
        logger.debug('Enter init_lora')

        if self.lora_handler.has_encoder_lora():
            # Init the encoder chunks just to get the lora mapping dicts, don't need to load weights
            logger.debug('Setting encoder lora state dict mapping')
            lora_state_dict_mapping = {}
            first_layer_idx = 0
            for i, num_layers in enumerate(self.encoder_layers_per_chunk):
                chunk = self.encoder_chunk_class(
                    self.config.e,
                    None if not self.lora_handler.has_encoder_lora() else self.lora_handler.e[0],
                    num_layers=num_layers,
                    first_layer_idx=first_layer_idx,
                    chunk_idx=i,
                )
                chunk_lora_mapping = chunk.generate_default_lora_state_dict_mapping()
                lora_state_dict_mapping.update(chunk_lora_mapping)
                first_layer_idx += num_layers
            self.lora_handler.set_encoder_lora_state_dict_mapping(lora_state_dict_mapping)

        if self.lora_handler.has_llm_lora():
            # Init the LLM chunks just to get the lora mapping dicts, don't need to load weights
            logger.debug('Setting LLM lora state dict mapping')
            lora_state_dict_mapping = {}
            lora_merged_state_dict_mapping = {}
            first_layer_idx = 0
            for i, num_layers in enumerate(self.llm_layers_per_chunk):
                chunk = self.llm_chunk_class(
                    self.config.l,
                    None if not self.lora_handler.has_llm_lora() else self.lora_handler.l[0],
                    num_layers=num_layers,
                    first_layer_idx=first_layer_idx,
                    chunk_idx=i,
                )
                chunk_lora_mapping, chunk_merged_lora_mapping = chunk.generate_default_lora_state_dict_mapping()
                lora_state_dict_mapping.update(chunk_lora_mapping)
                lora_merged_state_dict_mapping.update(chunk_merged_lora_mapping)
                first_layer_idx += num_layers
            self.lora_handler.set_llm_lora_state_dict_mapping(lora_state_dict_mapping, lora_merged_state_dict_mapping)

    def deduce_num_layers_per_chunk(self):
        """Deduce the number of encoder and decoder layers per chunk."""
        logger.debug('Enter deduce_num_layers_per_chunk')
        self.encoder_layers_per_chunk = [x['num_layers'] for x in self.encoder_quantized_model_infos]
        logger.debug(f'encoder_layers_per_chunk = {self.encoder_layers_per_chunk}')

        self.llm_layers_per_chunk = [
            x['num_layers']
            for x in (
                self.llm_prompt_quantized_model_infos if self.modes != 'gen' else self.llm_gen_quantized_model_infos
            )
        ]
        if self.separate_tail:
            self.llm_layers_per_chunk = self.llm_layers_per_chunk[: -self.num_separate_tails]
        logger.debug(f'llm_layers_per_chunk = {self.llm_layers_per_chunk}')

        self.encoder_layer_ids = [[x] for x in range(self.num_encoder_layers)]
        self.llm_layer_ids = [[x] for x in range(self.num_decoder_layers)]

    """ Forward passes """

    def forward_encoder(self, inputs, **kwargs):
        """Forward pass through the encoder.

        Args:
            inputs: The inputs for the encoder.
            kwargs (optional): Additional keyword arguments.
        """
        logger.debug('Enter forward_encoder')
        if self.config.e is None:
            return inputs, kwargs

        if isinstance(inputs, BatchFeature):
            input_list = [inputs['input_features']]
            if 'attention_mask_audio' in inputs:
                input_list.append(inputs['attention_mask_audio'])
            inputs = input_list

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [x.astype(self.dtype) for x in inputs]

        # qwen2vl_vision_attn_mask = kwargs.pop('qwen2vl_vision_attn_mask', None)
        # qwen2vl_vision_rot_emb = kwargs.pop('qwen2vl_vision_rot_emb', None)

        outputs = []
        cross_attns = []
        hidden_states = inputs[0]
        for i in range(len(self.encoder_layers_per_chunk)):
            extra_inputs = []
            # if getattr(self.encoder[i], 'attention_mask', None) is not None:
            #     self.encoder[i].attention_mask = qwen2vl_vision_attn_mask
            # if getattr(self.encoder[i], 'rot_emb', None) is not None:
            #     self.encoder[i].rot_emb = qwen2vl_vision_rot_emb
            logger.debug(f'chunk {i}')
            logger.debug(f'hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}')
            if self.lora_handler.has_encoder_lora():
                logger.debug(f'num_lora_inputs dtype: {self.lora_handler.encoder_lora_inputs[i][0].dtype}')
            if len(inputs) > 1:
                extra_inputs.append(inputs[1])
            model_in = [hidden_states, *extra_inputs, *self.lora_handler.encoder_lora_inputs[i]]
            executor = self.encoder[i]

            logger.debug('Quantize inputs')
            model_in = executor.run(
                model_in, output_names=executor.model_inputs, quantize_input=True
            )  # Quantize inputs

            # save quantized inputs if debug mode
            if self._debug_dir is not None:
                logger.debug('Saving quantized inputs as part of debug flow')
                in_dir = os.path.join(self._debug_dir, 'encoder', 'quantized_inputs')
                utils.recursive_remove_if_exist(in_dir, recreate=True)

                hidden_states = model_in[0]
                if len(self.lora_handler.encoder_lora_inputs[i]) > 0:
                    quantized_encoder_lora_inputs = model_in[-len(self.lora_handler.encoder_lora_inputs[i]) :]

                self._dump_np_and_bin(in_dir, 'in_hidden_states', hidden_states)
                for j in range(len(self.lora_handler.encoder_lora_inputs[i])):
                    self._dump_np_and_bin(in_dir, f'in_lora_{j}', quantized_encoder_lora_inputs[j])

            logger.debug(f'Forward Encoder (chunk {i})')
            model_out = executor.run(model_in)  # Run model

            # save quantized outputs if debug mode
            if self._debug_dir is not None:
                logger.debug('Saving quantized outputs as part of debug flow')
                output_names = executor.model_outputs
                out_dir = os.path.join(self._debug_dir, 'encoder', 'quantized_outputs')
                utils.recursive_remove_if_exist(out_dir, recreate=True)
                for j in range(len(output_names)):
                    self._dump_np_and_bin(out_dir, 'out_' + output_names[j].replace('/', '.'), model_out[j])

                logger.debug('Dequantize outputs')
                model_out_dequant = executor.run(
                    dict(zip(executor.model_outputs, model_out)), dequantize_output=True
                )  # Dequantize outputs

                # save dequantized outputs if debug mode
                if self._debug_dir is not None:
                    logger.debug('Saving dequantized outputs as part of debug flow')
                    output_names = executor.model_outputs
                    out_dir = os.path.join(self._debug_dir, 'encoder', 'dequantized_outputs')
                    utils.recursive_remove_if_exist(out_dir, recreate=True)
                    for j in range(len(output_names)):
                        self._dump_np_and_bin(out_dir, 'out_' + output_names[j].replace('/', '.'), model_out_dequant[j])
            hidden_states = model_out[0]

        out = model_out

        # Check if model out contains cross attention or not
        if len(out) == 1:
            outputs.append(out[0])
        elif len(out) == 2:  # Minimum 3 output if model contains cross attention
            outputs.append(out)
        else:
            outputs.append(out[0])
            cross_attns.append(out[1:])

        if len(outputs) == 1:
            outputs = outputs[0]
        if len(cross_attns) == 0:
            cross_attns = None
        elif len(cross_attns) == 1:
            cross_attns = cross_attns[0]
        return outputs, cross_attns, kwargs

    def forward_projector(self, inputs, **kwargs):
        """Do Nothing. Projector is expected to be inside encoder."""
        return inputs, kwargs

    @torch.no_grad()
    def forward_llm_quantized(
        self,
        curr_models,
        curr_models_info,
        input_embeds,
        mask,
        pos_emb,
        cache,
        pad_len=0,
        curr_tokens=None,
        cross_attn=None,
        tail_cache=None,
        memory=None,
        **kwargs,
    ):
        """Runs the forward pass for quantized models.

        Args:
            curr_models (list): List of current executors, whether prompt or gen.
            curr_models_info (list): List of current executors info, whether prompt or gen.
            input_embeds (np.ndarray): The input embeddings.
            mask (np.ndarray): The attention mask.
            pos_emb (list): List of positional embeddings.
            cache (list): List of cache tensors.
            pad_len: Length of prompt padding, if any.
            curr_tokens (np.ndarray, optional): The input tokens.
            cross_attn: Cross attention from encoder.
            tail_cache: The input cache object for the EAGLE tail if using EAGLE model.
            memory: The memory object for infini attention if using infini attention.
            kwargs (optional): Additional keyword arguments.

        Returns:
            torch.Tensor. logits output of LLM.
            Cache object. Updated cache of LLM.
            None or Cache object. Updated cache of EAGLE tail if using EAGLE model, else None.
        """
        logger.debug('Enter forward_llm_quantized')
        hidden_states = input_embeds
        original_mask = mask
        curr_k = []
        curr_v = []

        kwargs['attn_logits'] = []
        kwargs['attn_weights'] = []

        cross_attention = []
        if cross_attn is not None:
            for i in range(int(len(cross_attn) / 2)):
                cross_attention.append([cross_attn[i * 2], cross_attn[i * 2 + 1]])

        if self.config.l.model_type == 'gecko2':
            cos_sin = pos_emb[-2:]  # for gecko2
            input_mask = mask  # for gecko2
            all_layer_emb = pos_emb.pop(0) if self.config.l.d_per_layer_embedding else None
            fire_pos_emb = pos_emb.pop(0) if self.config.l.fire_normalizer_threshold else None
        elif self.config.l.model_type == 'gemma3':
            input_mask = mask
            pos_emb_local = pos_emb[-2:]
            pos_emb_global = pos_emb[:2]

        # Only deal with decoder layers and non separate tail here
        for i, num_layer in enumerate(self.llm_layers_per_chunk):
            if (self.config.l.model_type == 'gecko' and i >= 1 and len(pos_emb) > 2) or (
                self.config.l.model_type == 'whisper_decoder' and i >= 1 and len(pos_emb) > 0
            ):
                pos_emb.remove(pos_emb[0])
            elif self.config.l.model_type == 'gecko2':
                # start with cos and sin every chunk
                pos_emb = cos_sin
                if all_layer_emb is not None:
                    # extend per layer emb after cos, sin
                    pos_emb.append(all_layer_emb)

                curr_chunk_layer_ids = [
                    sum(self.llm_layers_per_chunk[:i]) + idx for idx in range(self.llm_layers_per_chunk[i])
                ]
                # if current chunk does not contain global layer
                if set(curr_chunk_layer_ids).isdisjoint(set(self.config.l.fire_pe_index)):
                    pos_emb = pos_emb[2:] + pos_emb[0:2]
                    mask = input_mask[0]
                elif set(curr_chunk_layer_ids).issubset(set(self.config.l.fire_pe_index)):
                    # remove the cos and sin
                    if fire_pos_emb is not None:
                        pos_emb.extend(fire_pos_emb)
                    pos_emb = pos_emb[2:]
                    mask = input_mask[1]
                else:
                    if fire_pos_emb is not None:
                        pos_emb.extend(fire_pos_emb)
                    # move cos, sin to the back
                    pos_emb = pos_emb[2:] + pos_emb[0:2]
                    mask = input_mask[0]
                    # add another mask in front of pos_emb
                    # no need to pop because each chunk will create a fresh pos_emb
                    pos_emb.insert(0, input_mask[1])
                original_mask = mask

            elif self.config.l.model_type == 'gemma3':
                if (
                    self.config.l.global_local_attention_pattern[i % len(self.config.l.global_local_attention_pattern)]
                    == 'GLOBAL'
                ):
                    mask = input_mask[1]
                    pos_emb = pos_emb_global
                else:
                    mask = input_mask[0]
                    pos_emb = pos_emb_local
                original_mask = mask

            cache_in = cache.get(
                layer_from=sum(self.llm_layers_per_chunk[:i]),
                layer_to=sum(self.llm_layers_per_chunk[:i]) + num_layer - 1,
            )
            cache_in = (
                memory.get(
                    layers=list(
                        range(sum(self.llm_layers_per_chunk[:i]), sum(self.llm_layers_per_chunk[:i]) + num_layer)
                    ),
                    kvs=cache_in,
                )
                if self.config.l.infini_attention
                else cache_in
            )
            other_masks = []
            if self.config.l.infini_attention:
                # FIXME: cross attention not supported
                bsz, q_len, _ = hidden_states.shape
                infini_mask_shape = (bsz, self.config.l.num_attention_heads, q_len, self.config.l.head_dim)
                other_masks.append(
                    np.ones(infini_mask_shape, dtype=hidden_states.dtype)
                    if memory.is_first_segment
                    else np.zeros(infini_mask_shape, dtype=hidden_states.dtype)
                )

            # FIXME: split mask is not compatible for the case that model_type is gecko2.
            if self.config.l.use_split_mask:
                num_token = input_embeds.shape[1]
                cache_mask_token_length = curr_models_info[i].get('bita_decode_token', 1)
                mask = [original_mask[:, :, :, -num_token:], original_mask[:, :, :cache_mask_token_length, :-num_token]]
            else:
                mask = [original_mask]
            # weave cross_attn into cache_in
            if len(cross_attention) > 0:
                new_cache = []
                cross_req = cross_attention[
                    sum(self.llm_layers_per_chunk[:i]) : sum(self.llm_layers_per_chunk[:i]) + num_layer
                ]
                for k in range(num_layer):
                    new_cache.append(cache_in[k * 2])
                    new_cache.append(cache_in[k * 2 + 1])
                    new_cache.append(cross_req[k][0])
                    new_cache.append(cross_req[k][1])
                cache_in = new_cache
            logger.debug(f'chunk {i}')
            logger.debug(f'hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}')
            logger.debug(f'mask shape: {mask[0].shape}, dtype: {mask[0].dtype}')
            if len(mask) > 1:
                logger.debug(f'split_mask shape: {mask[1].shape}, dtype: {mask[1].dtype}')
            logger.debug(f'pos_emb len: {len(pos_emb)} shape: {pos_emb[0].shape}, dtype: {pos_emb[0].dtype}')
            logger.debug(f'cache len: {len(cache_in)}')
            if len(cache_in) > 0:
                logger.debug(f'cache shape: {cache_in[0].shape}, dtype: {cache_in[0].dtype}')
            logger.debug(f'num_lora_inputs len: {len(self.lora_handler.llm_lora_inputs[i])}')
            if len(self.lora_handler.llm_lora_inputs[i]) > 0:
                logger.debug(f'num_lora_inputs dtype: {self.lora_handler.llm_lora_inputs[i][0].dtype}')

            model_in = [hidden_states, *mask, *pos_emb, *cache_in, *other_masks, *self.lora_handler.llm_lora_inputs[i]]
            executor = curr_models[i]

            logger.debug('Quantize inputs')
            model_in = executor.run(
                model_in, output_names=executor.model_inputs, quantize_input=True
            )  # Quantize inputs

            # save quantized inputs if debug mode
            if self._debug_dir is not None:
                logger.debug('Saving quantized inputs as part of debug flow')
                in_dir = os.path.join(
                    self._debug_dir, f'first_token_index_{cache.offset}', f'chunk_{i}', 'quantized_inputs'
                )
                utils.recursive_remove_if_exist(in_dir, recreate=True)
                if curr_tokens is not None:
                    np.save(os.path.join(in_dir, 'in_tokens.npy'), curr_tokens)

                input_names = executor.model_inputs
                for j in range(len(input_names)):
                    self._dump_np_and_bin(in_dir, f'in_{j}_' + input_names[j].replace('/', '.'), model_in[j])

            logger.debug(f'Forward LLM (chunk {i})')
            model_out = executor.run(model_in)  # Run model

            # save quantized outputs if debug mode
            if self._debug_dir is not None:
                logger.debug('Saving quantized outputs as part of debug flow')
                output_names = executor.model_outputs
                out_dir = os.path.join(
                    self._debug_dir, f'first_token_index_{cache.offset}', f'chunk_{i}', 'quantized_outputs'
                )
                utils.recursive_remove_if_exist(out_dir, recreate=True)
                for j in range(len(output_names)):
                    self._dump_np_and_bin(out_dir, f'out_{j}_' + output_names[j].replace('/', '.'), model_out[j])

            logger.debug('Dequantize outputs')
            model_out = executor.run(
                dict(zip(executor.model_outputs, model_out)), dequantize_output=True
            )  # Dequantize outputs

            # save dequantized outputs if debug mode
            if self._debug_dir is not None:
                logger.debug('Saving dequantized outputs as part of debug flow')
                output_names = executor.model_outputs
                out_dir = os.path.join(
                    self._debug_dir, f'first_token_index_{cache.offset}', f'chunk_{i}', 'dequantized_outputs'
                )
                utils.recursive_remove_if_exist(out_dir, recreate=True)
                for j in range(len(output_names)):
                    self._dump_np_and_bin(out_dir, f'out_{j}_' + output_names[j].replace('/', '.'), model_out[j])

            hidden_states = model_out[0]
            for j in range(1, (num_layer * 2 + 1), 2):
                k_cache = model_out[j]
                v_cache = model_out[j + 1]
                if pad_len > 0:
                    k_cache = k_cache[:, :, :-pad_len, :]
                    v_cache = v_cache[:, :, :-pad_len, :]
                curr_k.append(k_cache)
                curr_v.append(v_cache)

            if self.config.l.extra_output['attn_logits'] and self.config.l.extra_output['attn_weights']:
                kwargs['attn_logits'] += [model_out[idx] for idx in range(-num_layer, -(num_layer + 1) * 2, -1)]
                kwargs['attn_weights'] += [model_out[idx] for idx in range(-1, -(num_layer + 1), -1)]
            elif self.config.l.extra_output['attn_logits']:
                kwargs['attn_logits'] += [model_out[idx] for idx in range(-1, -(num_layer + 1), -1)]
            elif self.config.l.extra_output['attn_weights']:
                kwargs['attn_weights'] += [model_out[idx] for idx in range(-1, -(num_layer + 1), -1)]

        kwargs['cache_evictor'] = cache
        kwargs['curr_k'] = curr_k
        kwargs['curr_v'] = curr_v

        if self.evictor is not None:
            self.evictor.update_num_token(input_embeds.shape[1])
            self.evictor.cache_insert(**kwargs)
        else:
            cache.insert(curr_k, curr_v)

        # TODO: Double check execution flow for the case of custom tails
        if self.separate_tail:
            # Only need to swap tails during 'both' mode
            if self.modes == 'both':
                tail = self.tail_gen if not kwargs['prompt_mode'] else self.tail_prompt
            hidden_states, kwargs = self.forward_pre_tail_hook(hidden_states, **kwargs)
            for idx in range(1, len(tail) + 1):
                hidden_states, tail_cache, kwargs = self.forward_tail(
                    tail[idx - 1],
                    hidden_states,
                    curr_tokens=curr_tokens,
                    tail_cache=tail_cache,
                    tail_chunk_idx=len(self.llm_layers_per_chunk) + idx,
                    **kwargs,
                )

        logits = hidden_states

        return logits, cache, tail_cache, kwargs

    def generate_llm(
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
        """Generate LLM response using fixed shape.

        Args:
            input_embeds: The input embeddings.
            input_ids: The input IDs.
            num_token: Unused.
            cache_size: Unused.
            stopping_criteria: The stopping criteria.
            logits_processors: The logits processors.
            logits_warper: The logits warper.
            cross_attn: Cross attention from encoder.
            kwargs (optional): Additional keyword arguments.

        Returns:
            numpy.ndarray: Generated token ids concatenated behind input token ids.
        """
        logger.debug('Enter generate_llm')
        if self._debug:
            self._debug_dir = os.path.join('quantized_inference_debug', self.exp_name, kwargs['prompt_name'])

        curr_models = self.llm_prompt
        curr_models_info = self.llm_prompt_quantized_model_infos
        input_length = input_embeds.shape[1]
        num_token = self.prompt_num_tokens
        cache_size = self.cache_size
        logger.debug(f'Prompt input_length={input_length}, t={num_token}, c={cache_size}')

        memory = None
        if self.config.l.infini_attention:
            memory = kwargs.pop('memory')

        # Pad input_ids to input_length for stopping_criteria
        if input_ids.shape[1] != input_length:
            logger.debug(f'Pad stopping criteria used input_ids {input_ids.shape[1]} to input_length {input_length}')
            input_ids_for_max_length_criteria = self.pad_input_ids_to_input_embeds_len(input_ids, input_length)
        else:
            input_ids_for_max_length_criteria = input_ids

        # Init cache
        cache = cache_utils.Cache(
            self.config.l,
            cache_size,
            self.dtype,
            device=self.main_device,
            overture=self.overture if kwargs.pop('use_overture', True) else None,
        )
        tail_cache = (
            None
            if self.config.t is None or (self.config.t is not None and self.config.t.model_type != 'medusa')
            else cache_utils.Cache(self.config.t, cache_size, self.dtype, device=self.main_device)
        )

        # Init evictor prompt length
        if self.evictor is not None:
            self.evictor.update_prompt_length(input_length, cache.overture_size)

        seq_length = 0

        master_rot_emb = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)

        if self.config.l.model_type in ['gecko', 'whisper_decoder']:
            master_pos_emb = generate_utils.get_master_pos_emb(self.config.l, self.dtype)

        firelite_pos_emb = None
        if self.config.l.model_type == 'gecko2':
            firelite_pos_emb = generate_utils.get_firelite_pos_emb(
                self.config.l,
                self.dtype,
                length=max(self.config.l.max_position_embeddings, cache.cache_size + num_token),
            )
        elif self.config.l.model_type == 'gemma3':
            self.config.l.rotary_emb_base = self.config.l.rope_local_base_freq
            self.config.l.rope_scaling['factor'] = 1
            master_rot_emb_local = generate_utils.get_master_rot_emb(self.config, self.dtype, **kwargs)

        prompt_mode = True
        if self.config.l.infini_attention:
            memory.set_prompt_mode(prompt_mode)

        curr_input_ids = None
        num_prompt_passes = int((input_length - seq_length) / num_token) + int(
            (input_length - seq_length) % num_token != 0
        )
        logger.debug(f'num_prompt_passes={num_prompt_passes}')
        pad_len = (
            int(num_token - (input_length - seq_length) % num_token)
            if (input_length - seq_length) % num_token != 0
            else 0
        )
        logger.debug(f'pad_len={pad_len}')

        eos_id = (
            [self.config.l.eos_token_id] if isinstance(self.config.l.eos_token_id, int) else self.config.l.eos_token_id
        )

        final_logits = np.zeros((1, 0, self.config.l.vocab_size), dtype=self.dtype)
        while True:
            if pad_len > 0:
                logger.debug(f'First prompt pass. Right pad curr_input_embeds by {pad_len}')
                curr_input_ids = input_ids[:, : num_token - pad_len]
                # TODO: Got unexpected embedding data type for AndesVL inference. Should be resolved in previous pass.
                if isinstance(input_embeds, torch.Tensor):
                    input_embeds = input_embeds.detach().cpu().numpy()
                curr_input_embeds = input_embeds[:, : num_token - pad_len, :]
                curr_input_embeds = np.pad(curr_input_embeds, ((0, 0), (0, pad_len), (0, 0)))
            else:
                if prompt_mode:
                    logger.debug('Prompt mode')
                    curr_input_ids = input_ids[:, seq_length : seq_length + num_token]
                    curr_input_embeds = input_embeds[:, seq_length : seq_length + num_token, :]
                else:
                    logger.debug('Gen mode')
                    curr_input_ids = input_ids[:, -num_token:]

            if self.config.l.infini_attention:
                # get from memory
                cur_cache_size = memory.cur_cache_size
                valid_cache_size = memory.get_valid_cache_size(overture_size=cache.overture_size)
                logger.debug(f'valid cache size: {valid_cache_size}')
            elif self.evictor is not None:
                cur_cache_size = cache_size
                valid_cache_size = min(cache.offset, cache_size) if not cache.is_full else cache_size
            else:
                cur_cache_size = cache_size
                valid_cache_size = min(cache.overture_size + seq_length, cache_size)

            mask = generate_utils.generate_mask(
                cur_cache_size,
                valid_cache_size,
                num_token,
                num_token - pad_len,
                mask_value=self.config.l.mask_value,
                dtype=self.dtype,
                sliding_window=self.config.l.sliding_window_attention_size != 0,
                sliding_window_size=self.config.l.sliding_window_attention_size,
            )

            if cache.overture_size + seq_length + num_token > master_rot_emb.shape[2]:
                logger.error('Prompt + Response total length exceeds maximum positional embedding length.')

            pos_emb = (
                [master_pos_emb[:, cache.overture_size + seq_length : cache.overture_size + seq_length + num_token, :]]
                if self.config.l.model_type in ['gecko', 'whisper_decoder']
                else []
            )

            if self.config.l.model_type == 'gecko2':
                if self.config.l.d_per_layer_embedding:
                    if pad_len > 0:
                        all_layer_emb = kwargs['per_layer_embedding'][:, : num_token - pad_len, :, :]
                        all_layer_emb = np.pad(all_layer_emb, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
                    else:
                        if prompt_mode:
                            all_layer_emb = kwargs['per_layer_embedding'][:, seq_length : seq_length + num_token, :, :]
                    pos_emb.append(all_layer_emb)
                if self.config.l.use_fire_pe_in_global:
                    cur_pos = seq_length + cache.overture_size
                    fire_relative_pos, fire_query_pos = firelite_pos_emb
                    cur_fire_relative_pos = fire_relative_pos[
                        :, :, cache.cache_size : cache.cache_size + num_token, : cache.cache_size + num_token
                    ]
                    cur_fire_query_pos = fire_query_pos[:, :, cur_pos : cur_pos + num_token, :]
                    pos_emb.append(
                        [cur_fire_relative_pos, cur_fire_query_pos]
                    )  # append and handle in forward llm float

                # gecko2 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache_size,
                    min(cache.overture_size + seq_length, cache_size),
                    num_token,
                    num_token - pad_len,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)
            elif self.config.l.model_type == 'gemma3':
                # gemma3 needs both SWA and global mask
                global_mask = generate_utils.generate_mask(
                    cache.cache_size,
                    min(cache.overture_size + seq_length, cache.cache_size),
                    num_token,
                    num_token,
                    mask_value=self.config.l.mask_value,
                    dtype=self.dtype,
                    sliding_window=False,
                )
                mask = (mask, global_mask)
            if self.config.l.model_type != 'whisper_decoder':
                # static rope
                if self.config.l.extra_input['sink_rope']:
                    start_idx = cache.overture_size + seq_length
                    end_idx = cache.overture_size + seq_length + num_token
                    q_cos = master_rot_emb[:, :1, min(start_idx, cache_size) : min(end_idx, cache_size + num_token), :]
                    q_sin = master_rot_emb[:, 1:, min(start_idx, cache_size) : min(end_idx, cache_size + num_token), :]
                    k_rot_emb = master_rot_emb[:, :, : min(start_idx, cache_size) + num_token, :]
                    k_rot_emb = np.pad(k_rot_emb, ((0, 0), (0, 0), (max(cache_size - start_idx, 0), 0), (0, 0)))
                    k_cos = k_rot_emb[:, :1, :, :]
                    k_sin = k_rot_emb[:, 1:, :, :]
                    if self.use_single_bmm_attention:
                        q_cos = q_cos.transpose(0, 2, 1, 3)
                        q_sin = q_sin.transpose(0, 2, 1, 3)
                        k_cos = k_cos.transpose(0, 2, 1, 3)
                        k_sin = k_sin.transpose(0, 2, 1, 3)
                    pos_emb.extend([q_cos, q_sin, k_cos, k_sin])
                else:
                    if self.config.l.infini_attention:
                        # currently infini_attention uses local
                        # get from memory
                        start_position = memory.get_start_position(overture_size=cache.overture_size)
                        end_position = start_position + num_token
                        logger.debug(f'pos emb start position: {start_position}')
                    else:
                        start_position = cache.overture_size + seq_length
                        end_position = cache.overture_size + seq_length + num_token
                    cos = master_rot_emb[:, :1, start_position:end_position, :]
                    sin = master_rot_emb[:, 1:, start_position:end_position, :]
                    if self.use_single_bmm_attention:
                        cos = cos.transpose(0, 2, 1, 3)
                        sin = sin.transpose(0, 2, 1, 3)
                    pos_emb.extend([cos, sin])
            # gemma3 need cat local pos_emb
            # TODO: Gemma3 RoPE currently can not be with cache eviction.
            if self.config.l.model_type == 'gemma3':
                start_position = cache.overture_size + seq_length
                end_position = cache.overture_size + seq_length + num_token
                cos = master_rot_emb_local[:, :1, start_position:end_position, :]
                sin = master_rot_emb_local[:, 1:, start_position:end_position, :]
                if self.use_single_bmm_attention:
                    cos = cos.transpose(0, 2, 1, 3)
                    sin = sin.transpose(0, 2, 1, 3)
                pos_emb.extend([cos, sin])

            if prompt_mode:
                hidden_states = curr_input_embeds
            else:
                if self.config.l.model_type == 'gecko2':
                    # needs previous token
                    hidden_states = self.text_embedding_layer(torch.tensor(input_ids[:, -2:]).to(self.main_device))
                    hidden_states, per_layer_embedding = hidden_states
                    hidden_states = hidden_states[:, -1:, :].detach().cpu().numpy()
                    if self.config.l.d_per_layer_embedding:
                        pos_emb[0] = per_layer_embedding[:, -1:, ...].detach().cpu().numpy()
                else:
                    hidden_states = (
                        self.text_embedding_layer(torch.tensor(curr_input_ids).to(self.main_device))
                        .detach()
                        .cpu()
                        .numpy()
                    )
            kwargs['prompt_mode'] = prompt_mode
            logits, cache, tail_cache, kwargs = self.forward_llm_quantized(
                curr_models,
                curr_models_info,
                hidden_states,
                mask,
                pos_emb,
                cache,
                pad_len,
                curr_tokens=curr_input_ids,
                cross_attn=cross_attn,
                tail_cache=tail_cache,
                memory=memory,
                **kwargs,
            )
            if len(logits.shape) == 2:
                # Expand the batch size dimension.
                logits = np.expand_dims(logits, axis=0)

            # update memory
            # for infini attention, memory update is guaranteed to occur after insertion
            # as in generate_llm, the segments are processed to guarantee this
            if self.config.l.infini_attention:
                is_updated = False
                cur_kvs = cache.get()
                is_updated = memory.insert_and_update(cur_kvs=cur_kvs, num_token=num_token - pad_len)
                # no need to manually clear cache as can just set the mask to 0
                logger.debug(f'Is memory updated: {is_updated}')

            if pad_len > 0:
                logits = logits[:, :-pad_len, :]

            seq_length = seq_length + num_token - pad_len
            logger.debug(f'seq_length={seq_length}')

            # slice in case of lm_head pad
            logits = logits[:, :, : self.config.l.vocab_size]
            final_logits = np.concatenate([final_logits, logits], axis=1)

            kwargs['seq_length'] = seq_length
            if self.evictor is not None:
                self.evictor.update_attn(**kwargs)
                if self.evictor.trigger(seq_length):
                    # update cache
                    self.evictor.compress(cache, seq_length)

            pad_len = 0  # Only first pass will ever need padding
            if seq_length < input_length:
                # Prompt mode not done, discard logits and continue processing prompt
                continue
            # Prompt mode done, switch to generative mode
            num_token = 1
            prompt_mode = False
            curr_models = self.llm_gen
            curr_models_info = self.llm_gen_quantized_model_infos

            logits_to_sample = (
                input_ids if self.config.l.model_type == 'whisper_decoder' else input_ids[:, input_length:]
            )
            next_token = self.sampler(logits, logits_to_sample, logits_processors, logits_warper)

            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1)
            input_ids_for_max_length_criteria = np.concatenate(
                [input_ids_for_max_length_criteria, next_token[:, None].cpu().numpy().astype(np.int32)], axis=-1
            )

            # stop when eos or if we exceed the maximum length
            if next_token[0].item() in eos_id:
                logger.debug('EOS token predicted. Stop generation.')
                break
            if stopping_criteria(torch.tensor(input_ids_for_max_length_criteria), ()):
                logger.debug('Max length reached. Stop generation.')
                break

            # FIXME: change to better way in future for different cache sizes
            # prepare cache again for different cache sizes
            if self.config.l.infini_attention:
                memory.set_prompt_mode(prompt_mode)
                if not memory.update_during_gen and cache.cache_size != memory.gen_cache_size:
                    logger.info('Changing cache due to different sizes between prompt and gen.')
                    cur_cache = cache.get()
                    del cache
                    cur_k = []
                    cur_v = []
                    for cache_idx in range(0, len(cur_cache), 2):
                        cur_k.append(cur_cache[cache_idx])
                        cur_v.append(cur_cache[cache_idx + 1])
                    cache = cache_utils.Cache(
                        self.config.l,
                        memory.gen_cache_size,
                        self.dtype,
                        device=self.main_device,
                        overture=self.overture if kwargs.pop('use_overture', True) else None,
                    )
                    cache.insert(cur_k, cur_v)

        if kwargs.pop('return_logits', False):
            logger.debug('Return logits only')
            return final_logits
        return input_ids, kwargs

    def forward_tail(self, tail, hidden_states, curr_tokens=None, tail_cache=None, tail_chunk_idx=1, **kwargs):
        """Forward pass through the tail.

        Args:
            tail (Tail): The separate tail executor.
            hidden_states: The hidden_states for the tail.
            curr_tokens (np.ndarray, optional): The input tokens.
            tail_cache: The input cache object for the EAGLE tail if using EAGLE model.
            tail_chunk_idx (int): The tail chunk index meant for saving debug tensors.
            kwargs: Additional keyword arguments.
        """
        logger.debug('Running Separate Tail')
        logger.debug('Quantize inputs')
        (hidden_states,) = tail.run(
            [hidden_states], output_names=tail.model_inputs, quantize_input=True
        )  # Quantize inputs

        # save quantized inputs if debug mode
        if self._debug_dir is not None:
            logger.debug('Saving quantized inputs as part of debug flow')
            in_dir = os.path.join(
                self._debug_dir,
                f'first_token_index_{tail_cache.offset if tail_cache is not None else kwargs["offset"]}',
                f'chunk_{tail_chunk_idx}',
                'inputs',
            )
            utils.recursive_remove_if_exist(in_dir, recreate=True)
            if curr_tokens is not None:
                np.save(os.path.join(in_dir, 'in_tokens.npy'), curr_tokens)
            self._dump_np_and_bin(in_dir, 'in_hidden_states', hidden_states)

        logger.debug('Forward LLM separate tail chunk')
        (logits,) = tail.run([hidden_states])  # Run Tail

        # save quantized outputs if debug mode
        if self._debug_dir is not None:
            logger.debug('Saving quantized outputs as part of debug flow')
            out_dir = os.path.join(
                self._debug_dir,
                f'first_token_index_{tail_cache.offset if tail_cache is not None else kwargs["offset"]}',
                f'chunk_{tail_chunk_idx}',
                'outputs',
            )
            utils.recursive_remove_if_exist(out_dir, recreate=True)
            self._dump_np_and_bin(out_dir, 'out_logits', logits)

        logger.debug('Dequantize outputs')
        (logits,) = tail.run(dict(zip(tail.model_outputs, [logits])), dequantize_output=True)  # Dequantize outputs

        return logits[..., : self.config.l.vocab_size], tail_cache, kwargs  # Remove padding for models like minicpm
