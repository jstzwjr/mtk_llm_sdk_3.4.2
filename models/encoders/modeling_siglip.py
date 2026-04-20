# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Define SigLIP model class."""

import json
import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import image_utils, logger, utils
from ..preprocessors.preprocessor_siglip import SigLIPImageProcessor
from .configuration_siglip import SigLIPConfig
from .modeling_clip import CLIPVisionEncoderChunk


class SigLIPVisionEmbeddings(nn.Module):
    """SigLIPVisionEmbeddings class.

    This class extends the nn.Module class and initializes the SigLIPVisionEmbeddings with specific configurations.

    Attributes:
        config (SigLIPConfig): The configuration for the SigLIPVisionEmbeddings.
        hidden_size (int): The hidden size.
        image_size (int): The image size.
        patch_size (int): The patch size.
        patch_embedding (nn.Conv2d): The patch embedding layer.
        jit_trace (bool): Whether to use JIT tracing.
        num_patches (int): The number of patches.
        num_positions (int): The number of positions.
        position_embedding (nn.Embedding): The position embedding layer.
        position_ids (torch.Tensor): The position IDs.

    Methods:
        forward: Forward pass for the SigLIPVisionEmbeddings.
    """

    def __init__(self, config: SigLIPConfig, jit_trace=False):
        """Initializes the SigLIPVisionEmbeddings class.

        Args:
            config (SigLIPConfig): The configuration for the SigLIPVisionEmbeddings.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.jit_trace = jit_trace
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches  # Different from CLIP: Since no Class embedding
        self.position_embedding = nn.Embedding(self.num_positions, self.hidden_size)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        self.register_buffer('vision_pos_emb', torch.zeros(1, self.num_positions, self.hidden_size), persistent=False)

    def forward(self, pixel_values):
        """Forward pass for the SigLIPVisionEmbeddings.

        Args:
            pixel_values (torch.Tensor): The input pixel values.

        Returns:
            torch.Tensor: The output embeddings.
        """
        pixel_values = self.patch_embedding(pixel_values)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)
        return pixel_values + self.vision_pos_emb


class SigLIPVisionTransformer(CLIPVisionEncoderChunk):
    """SigLIPVisionTransformer class.

    This class extends the CLIPVisionEncoderChunk class and initializes the SigLIPVisionTransformer with
    specific configurations.

    Attributes:
        config (SigLIPConfig): The configuration for the SigLIPVisionTransformer.
        jit_trace (bool): Whether to use JIT tracing.
        dtype (torch.dtype): The data type. Defaults to torch.float32.
        embeddings (SigLIPVisionEmbeddings): The embeddings layer.
        post_layernorm (nn.LayerNorm): The post layer normalization layer.
        image_processor (SigLIPImageProcessor): The image processor.

    Methods:
        forward: Forward pass for the SigLIPVisionTransformer.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        parallel_lora=False,
        distribute_layers=True,
        **kwargs,
    ):
        """Initializes the SigLIPVisionTransformer class.

        Args:
            config (SigLIPConfig): The configuration for the SigLIP model.
            lora (LoRA): The lora object.
            num_layers (int): The number of encoder layers for the current chunk.
            first_layer_idx (int): The index of the first encoder layer in the current chunk.
            chunk_idx (int): The chunk index of the current chunk.
            dtype (torch.dtype): The default dtype to use for this chunk.
            jit_trace (bool): Flag to determine if model is to be run as part of JIT tracing or not.
            parallel_lora (bool): Flag to determine if parallel lora is used in the current chunk.
            distribute_layers (bool): Flag to determine if encoder layers should be evenly distributed among available
                GPUs or not.
            kwargs (dict): Additional keyword arguments.
        """
        super().__init__(
            config,
            lora,
            num_layers,
            first_layer_idx,
            chunk_idx,
            dtype,
            jit_trace,
            parallel_lora,
            distribute_layers,
            **kwargs,
        )
        if self.chunk_idx == 0:
            self.embeddings = SigLIPVisionEmbeddings(config, jit_trace=jit_trace)
            del self.pre_layernorm  # Since SigLIP does not require this
        if self.chunk_idx == self.config.num_hidden_layers - 1:
            self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.image_processor = SigLIPImageProcessor()

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        super()._generate_default_state_dict_mapping()
        if self.chunk_idx == 0:
            self.state_dict_mapping.pop('pre_norm_weight')
            self.state_dict_mapping.pop('pre_norm_bias')
            self.state_dict_mapping.pop('class_embedding')
            self.state_dict_mapping.pop('position_embedding_weight')
            self.state_dict_mapping.pop('patch_embedding_weight')
            self.state_dict_mapping.pop('patch_embedding_bias')
            self.state_dict_mapping.update(
                {
                    'position_embedding_weight': {
                        'embeddings.position_embedding.weight': 'vision_model.embeddings.position_embedding.weight'
                    },
                    'patch_embedding_weight': {
                        'embeddings.patch_embedding.weight': 'vision_model.embeddings.patch_embedding.weight'
                    },
                    'patch_embedding_bias': {
                        'embeddings.patch_embedding.bias': 'vision_model.embeddings.patch_embedding.bias'
                    },
                }
            )
        if self.chunk_idx == self.config.num_hidden_layers - 1:
            self.state_dict_mapping.update(
                {
                    'post_norm_weight': {'post_layernorm.weight': f'vision_model.{self.norm_names["post"]}.weight'},
                    'post_norm_bias': {'post_layernorm.bias': f'vision_model.{self.norm_names["post"]}.bias'},
                }
            )

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter CLIPVisionEncoderChunk load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

        state_dict = self._pop_text_model_weights(state_dict)

        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        for internal_key, mapping_dict in self.state_dict_mapping.items():
            logger.debug(f'internal_key={internal_key}, mapping_dict={mapping_dict}')
            found = False
            # Ensure that weight exist in state
            # mapping_dict values should be a dict with exactly length 1
            if not isinstance(mapping_dict, dict):
                logger.error(f'Expected dict for mapping_dict but got {type(mapping_dict)}', err=TypeError)
            if len(mapping_dict) != 1:
                logger.error(f'Expected exactly 1 key-value pair in mapping_dict but got {len(mapping_dict)}')
            model_key = next(iter(mapping_dict))
            external_key = mapping_dict[model_key]

            dtype = self.dtype if internal_key.split('_')[-2] != 'norm' else torch.float32

            # Check if key with all found prefixes directly matches the state_dict key
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(
                        f'Found {internal_key} weight using prefix, state_dict key={key_to_test}, dtype={dtype}'
                    )
                    weights_to_load.update({model_key: state_dict.pop(key_to_test).to(dtype)})
                    found = True
                    state_dict_keys.remove(key_to_test)
                    break

            if not found:
                for k in state_dict_keys:
                    if k.endswith(external_key):
                        logger.debug(
                            f'Found {internal_key} weight using iteration, state_dict key={k}, dtype={dtype}. '
                            f'Adding prefix: {k[: -len(external_key)]}'
                        )
                        self.prefixes.append(k[: -len(external_key)])
                        weights_to_load.update({model_key: state_dict.pop(k).to(dtype)})
                        found = True
                        state_dict_keys.remove(k)
                        break

            if not found and internal_key.endswith('_bias'):
                # Bias key not found, default to zeros
                # For shape, check corresponding weight shape in both state_dict and weights_to_load
                weight_internal_key = internal_key.replace('_bias', '_weight')
                weight_model_key = next(iter(self.state_dict_mapping[weight_internal_key]))
                shape = None
                if weight_model_key in weights_to_load:
                    logger.debug(
                        f'Init bias {internal_key} to zeros using shape={shape} found from {weight_internal_key} '
                        f'weight shape and dtype={dtype}. Found from weights_to_load.'
                    )
                    weight_shape = weights_to_load[weight_model_key].shape
                    shape = weight_shape[0]
                else:
                    # Could be that the corresponding weight has not been loaded into weights_to_load
                    weight_external_key = external_key.replace('_bias', '_weight')
                    for pre in self.prefixes:
                        key_to_test = pre + weight_external_key
                        if key_to_test in state_dict_keys:
                            logger.debug(
                                f'Init bias {internal_key} to zeros using shape={shape} found from '
                                f'{weight_internal_key} weight shape and dtype={dtype}. Found from state_dict.'
                            )
                            weight_shape = state_dict[key_to_test].shape
                            shape = weight_shape[0]
                            break
                if shape is None:
                    logger.error(
                        'Unable to find weight in both weights_to_load and state_dict associated with bias: '
                        f'{internal_key}'
                    )
                weights_to_load.update({model_key: torch.zeros(shape, dtype=dtype)})
                continue

            if not found:
                logger.debug(f'Cannot find {internal_key} weight')
                missing_keys.append((internal_key, external_key))

        if len(missing_keys) > 0:
            for internal_key, external_key in missing_keys:
                logger.warning(f'Unable to find {internal_key} weight in state_dict. Expected subkey: {external_key}')
            logger.info(f'state dict keys for reference: {state_dict_keys}')
            logger.error('Please modify your state_dict keys according to the expected subkeys.', err=KeyError)

        if len(self.prefixes) > 2:
            logger.warning(
                f'More than 1 prefix found (found {self.prefixes[1:]}). '
                'This is unexpected and will likely cause errors during weight loading.'
            )

        num_gpu = torch.cuda.device_count()
        if num_gpu == 0 or self.jit_trace:
            self.device_list = ['cpu' for _ in range(self.num_layers)]
        else:
            if self.distribute_layers:
                master_gpu_ids = sorted(
                    list(range(num_gpu)) * (self.config.num_hidden_layers // num_gpu)
                    + (
                        list(range(num_gpu))[: self.config.num_hidden_layers % num_gpu]
                        if self.config.num_hidden_layers % num_gpu != 0
                        else []
                    )
                )
            else:
                master_gpu_ids = [os.getenv('LOCAL_RANK', 0)] * self.config.num_hidden_layers
            self.device_list = [f'cuda:{x}' for x in master_gpu_ids][state_dict_start_idx:state_dict_end_idx]

        if weights_to_load.keys() != self.state_dict().keys():
            weights_to_load_only_keys = [x for x in weights_to_load if x not in self.state_dict()]
            model_only_keys = [x for x in self.state_dict() if x not in weights_to_load and 'lora' not in x]
            if self.parallel_lora:
                model_only_keys = [
                    x for x in model_only_keys if '_weight_quantizer' not in x and '_act_quantizer' not in x
                ]
            if model_only_keys != [] or weights_to_load_only_keys != []:
                logger.error(
                    f"model state dict keys don't match with state_dict to load into model.\n"
                    f'Model only keys:{model_only_keys}\nstate_dict only keys:{weights_to_load_only_keys}'
                )

        if quant_config is not None:
            logger.info(f'Quantizing chunk {self.chunk_idx} using quant config: {quant_config}')
            quant_config_chunk = int(quant_config.rsplit('_', 1)[-1].split('.json')[0])
            if quant_config_chunk != self.chunk_idx:
                logger.error(
                    f'chunk_idx={self.chunk_idx} but quant config used {quant_config} is for chunk {quant_config_chunk}'
                )
            quantize_handler = mtk_quantization.pytorch.QuantizeHandler()
            self = quantize_handler.prepare(self, quant_config)
            self._quantizer_dict = quantize_handler._quantizer_dict  # noqa: SLF001
            if not self.parallel_lora:
                with open(quant_config) as f:
                    data = f.read()
                quant_config_dict = json.loads(data)
                weight_targets = quant_config_dict['quantizer_targets']['constant_weights']
                for wgt in weight_targets:
                    weights_to_load = self._add_quantizer_weights(
                        state_dict,
                        weights_to_load,
                        wgt,
                        'weight',
                        prefix=self.prefixes[-1],
                    )
                act_targets = quant_config_dict['quantizer_targets']['activations']
                for act in act_targets:
                    weights_to_load = self._add_quantizer_weights(
                        state_dict,
                        weights_to_load,
                        act,
                        'activation',
                        prefix=self.prefixes[-1],
                    )

        self.load_state_dict(weights_to_load, strict=False)

        # Update static vision position embedding after state dict loaded.
        if self.chunk_idx == 0:
            self.embeddings.vision_pos_emb = self.embeddings.position_embedding(self.embeddings.position_ids)

        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        if self.first_layer_idx == 0:
            self.embeddings.to(self.device_list[0])
        if self.chunk_idx == self.config.num_hidden_layers - 1:
            self.post_layernorm.to(self.device_list[-1])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict

    @image_utils.NHWCWrapper
    def forward(self, pixel_values, use_last_hidden_state=False):
        """Forward pass for the SigLIPVisionTransformer.

        Args:
            pixel_values (torch.Tensor): The input pixel values.
            use_last_hidden_state (bool, optional): Whether to use the last hidden state. Defaults to False.

        Returns:
            tuple: The hidden states and encoder outputs. If use_last_hidden_state is True, also returns the
                last hidden states.
        """
        hidden_states = pixel_values.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.embeddings(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states.to(self.device_list[idx]))

        if use_last_hidden_state:
            last_hidden_states = self.post_layernorm(hidden_states)
            return hidden_states, last_hidden_states

        return hidden_states

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        if self.chunk_idx == 0:
            return torch.randn(
                1,
                self.config.image_size,
                self.config.image_size,
                self.config.num_channels,
                device='cpu',
                dtype=torch.float32,
            )
        feature_length = int(self.config.image_size / self.config.patch_size)
        feature_size = int(feature_length * feature_length)
        return torch.randn(1, feature_size, self.config.hidden_size, device='cpu', dtype=torch.float32)

    def get_ptq_inputs(self, args, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            kwargs: Additional keyword arguments.

        Returns:
            tuple: Tuple containing input shapes, input value ranges, calibration data generator,
            and evaluation data generator.
        """
        if self.chunk_idx == 0:
            input_shapes = [[None, self.config.image_size, self.config.image_size, self.config.num_channels]]
        else:
            feature_length = int(self.config.image_size / self.config.patch_size)
            feature_size = int(feature_length * feature_length)
            input_shapes = [[None, feature_size, self.config.hidden_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [
                            np.random.rand(
                                1, self.config.image_size, self.config.image_size, self.config.num_channels
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(1, feature_size, self.config.hidden_size).astype(np.float32)]
        else:

            def calib_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.calibration_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [
                            np.random.rand(
                                1, self.config.image_size, self.config.image_size, self.config.num_channels
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(1, feature_size, self.config.hidden_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
