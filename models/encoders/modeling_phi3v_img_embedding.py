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
"""Define Phi3-V vision encoder model class."""

import json
import os
from datetime import datetime

import mtk_quantization
import numpy as np
import torch
import torch.nn as nn

from ...utils import logger, utils
from .configuration_clip import CLIPConfig
from .modeling_clip import CLIPVisionEncoderChunk

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPConfig(
    hidden_size=1024,
    image_size=336,
    intermediate_size=4096,
    layer_norm_eps=1e-05,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=24,
    patch_size=14,
    projection_dim=768,
)


class Phi3VImageEmbeddingChunk(CLIPVisionEncoderChunk):
    """Phi3V Visual Encoder.

    This class implements the visual encoder for the Phi3V model, which processes image inputs and generates embeddings.

    Attributes:
        jit_trace (bool): Whether to use JIT tracing.
        _wte (torch.nn.Module): Word token embedding module.
        clip_config (dict): Configuration for the CLIP vision model.
        img_processor (CLIPVisionTransformer): Vision transformer for processing images.
        image_dim_out (int): Output dimension of the image features.
        num_img_tokens (int): Number of image tokens.
        img_sizes (torch.LongTensor): Sizes of the images.
        use_hd_transform (bool): Whether to use high-dimensional transform.
        with_learnable_separator (bool): Whether to use a learnable separator.
        hd_transform_order (str): Order of high-dimensional transform.
        glb_GN (torch.nn.Parameter): Global group normalization parameter.
        sub_GN (torch.nn.Parameter): Sub group normalization parameter.
        img_projection (torch.nn.Module): Module for projecting image features.
        vocab_size (int): Size of the vocabulary.
        img_features (torch.FloatTensor): Image features.
        layer_idx (int): Index of the layer to extract features from.
        type_feature (str): Type of feature to extract.

    Methods:
        set_img_features(img_features):
            Set the image features.
        set_img_sizes(img_sizes):
            Set the image sizes.
        set_text_emb(wte):
            Set the text embedding module.
        get_img_features(img_embeds):
            Get image features from the image embeddings.
        forward_vision(pixel_values):
            Forward pass for the vision model.
        forward_projector(img_features, ndim):
            Forward pass for the image projector.
        forward(input_ids, pixel_values):
            Forward pass for the entire model.
    """

    def __init__(
        self,
        config,
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
        """Initializes the Phi3VImageEmbeddingChunk class.

        Args:
            config (CLIPConfig): The configuration for the CLIP model.
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
        if isinstance(config.img_processor, dict) and config.img_processor.get('name', None) == 'clip_vision_model':
            assert 'model_name' in config.img_processor, 'model_name must be provided for CLIPVisionModel'
            assert 'image_dim_out' in config.img_processor, 'image_dim_out must be provided for CLIPVisionModel'
            assert 'num_img_tokens' in config.img_processor, 'num_img_tokens must be provided for CLIPVisionModel'
            assert config.img_processor['model_name'] == 'openai/clip-vit-large-patch14-336'
            self.clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.image_dim_out = config.img_processor['image_dim_out']  # 1024
            self.num_img_tokens = config.img_processor['num_img_tokens']  # 144
        else:
            logger.error(f'img_processor = {config.img_processor}, not implemented', err=NotImplementedError)

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.use_hd_transform
        self.with_learnable_separator = config.with_learnable_separator
        self.hd_transform_order = config.hd_transform_order
        # with_hd_transform and with_learnable_separator should have same value
        if self.use_hd_transform != self.with_learnable_separator:
            logger.error('use_hd_transform and with_learnable_separator should have same value', err=ValueError)

        self.vocab_size = config.vocab_size
        self.img_features = None

        self.layer_idx = config.select_layer
        self.type_feature = config.type_feature

        # For PTQ, need to save sub_GN and glb_GN to encoder_weight_dir
        self.encoder_weight_dir = kwargs.pop('encoder_weight_dir', None)
        if self.encoder_weight_dir is None:
            logger.error('Must pass encoder_weight_dir for Phi3VImageEmbeddingChunk.', err=ValueError)

        self.fixed_img_sizes = config.fixed_img_sizes

        super().__init__(
            self.clip_config,
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

        if self.with_learnable_separator and self.chunk_idx == 0:
            if not self.use_hd_transform:
                logger.error('learnable separator is only for hd transform', err=ValueError)
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))
            logger.debug(f'learnable separator enabled for hd transform, hd_transform_order={self.hd_transform_order}')

        self.jit_trace = jit_trace

        self._wte = None
        self.img_sizes = None

    def _pop_redundant_keys(self, state_dict):
        state_dict.pop('model.vision_embed_tokens.img_processor.vision_model.post_layernorm.bias', None)
        state_dict.pop('model.vision_embed_tokens.img_processor.vision_model.post_layernorm.weight', None)
        return state_dict

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {}
        if self.chunk_idx == 0:
            state_dict_mapping = {
                'class_embedding': {'embeddings.class_embedding': 'embeddings.class_embedding'},
                'position_embedding_weight': {
                    'embeddings.position_embedding.weight': 'embeddings.position_embedding.weight'
                },
                'patch_embedding_weight': {'embeddings.patch_embedding.weight': 'embeddings.patch_embedding.weight'},
                'patch_embedding_bias': {'embeddings.patch_embedding.bias': 'embeddings.patch_embedding.bias'},
                'pre_norm_weight': {'pre_layernorm.weight': f'{self.norm_names["pre"]}.weight'},
                'pre_norm_bias': {'pre_layernorm.bias': f'{self.norm_names["pre"]}.bias'},
            }

            # sub_GN and glb_GN
            state_dict_mapping.update(
                {
                    'glb_GN': {'glb_GN': 'model.vision_embed_tokens.glb_GN'},
                    'sub_GN': {'sub_GN': 'model.vision_embed_tokens.sub_GN'},
                }
            )

        # fmt: off
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.out_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.out_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_fc1_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc1.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc1"]}.weight'
                    },
                    f'{outer_layer_idx}_fc1_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc1.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc1"]}.bias'
                    },
                    f'{outer_layer_idx}_fc2_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc2.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc2"]}.weight'
                    },
                    f'{outer_layer_idx}_fc2_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc2.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc2"]}.bias'
                    },
                    f'{outer_layer_idx}_1_layer_norm_weight': {
                        f'layers.{inner_layer_idx}.layer_norm1.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm1"]}.weight'
                    },
                    f'{outer_layer_idx}_2_layer_norm_weight': {
                        f'layers.{inner_layer_idx}.layer_norm2.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm2"]}.weight'
                    },
                }
            )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_1_layer_norm_bias': {
                            f'layers.{inner_layer_idx}.layer_norm1.bias':
                            f'encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm1"]}.bias'
                        },
                        f'{outer_layer_idx}_2_layer_norm_bias': {
                            f'layers.{inner_layer_idx}.layer_norm2.bias':
                            f'encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm2"]}.bias'
                        },
                    }
                )
            # fmt: on
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())

        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter Phi3VImgEmbedding load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

        state_dict = self._pop_redundant_keys(state_dict)

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

        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        if self.first_layer_idx == 0:
            self.embeddings.to(self.device_list[0])
            self.pre_layernorm.to(self.device_list[0])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        # Save sub_GN and glb_GN to encoder_weight_dir
        if self.chunk_idx == 0:
            sub_GN_save_path = os.path.join(self.encoder_weight_dir, 'sub_GN.pt')  # noqa: N806
            glb_GN_save_path = os.path.join(self.encoder_weight_dir, 'glb_GN.pt')  # noqa: N806
            torch.save(self.sub_GN, sub_GN_save_path)
            torch.save(self.glb_GN, glb_GN_save_path)
            logger.debug(f'Save sub_GN to {sub_GN_save_path}')
            logger.debug(f'Save glb_GN to {glb_GN_save_path}')

        return self, state_dict

    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        """Set the image features.

        Args:
            img_features (torch.FloatTensor): Image features.
        """
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        """Set the image sizes.

        Args:
            img_sizes (torch.LongTensor): Sizes of the images.
        """
        self.img_sizes = img_sizes

    def set_text_emb(self, wte):
        """Set the text embedding module.

        Args:
            wte (torch.nn.Module): Word token embedding module.
        """
        logger.debug(f'set text emb: {type(wte)}')
        self._wte = wte

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        """Get image features from the image embeddings.

        Args:
            img_embeds (torch.FloatTensor): Image embeddings.

        Returns:
            torch.FloatTensor: Extracted image features.
        """
        layer_idx = self.layer_idx
        type_feature = self.type_feature

        _, img_processor_output = self.img_processor(img_embeds)
        img_feature = img_processor_output[layer_idx]

        if type_feature == 'patch':
            return img_feature[:, 1:]

        if type_feature == 'cls_patch':
            return img_feature

        raise NotImplementedError

    def _calculate_batch_size(self):
        batch_size = 0
        for s in self.fixed_img_sizes:
            h = s[0] // self.config.image_size
            w = s[1] // self.config.image_size
            batch_size += h * w + 1
        logger.debug(f'phi3v vision embedding batch size: {batch_size}')
        return batch_size

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        batch_size = self._calculate_batch_size()
        if self.chunk_idx == 0:
            return torch.randn(
                batch_size,
                self.config.image_size,
                self.config.image_size,
                self.config.num_channels,
                device='cpu',
                dtype=torch.float32,
            )
        feature_length = int(self.config.image_size / self.config.patch_size)
        feature_size = int(feature_length * feature_length)
        return torch.randn(batch_size, 1 + feature_size, self.config.hidden_size, device='cpu', dtype=torch.float32)

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
        batch_size = self._calculate_batch_size()
        logger.debug(f'ptq batch size: {batch_size}')
        if self.chunk_idx == 0:
            input_shapes = [[batch_size, self.config.image_size, self.config.image_size, self.config.num_channels]]
        else:
            feature_length = int(self.config.image_size / self.config.patch_size)
            feature_size = int(feature_length * feature_length)
            input_shapes = [[batch_size, 1 + feature_size, self.config.hidden_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [
                            np.random.rand(
                                batch_size,
                                self.config.image_size,
                                self.config.image_size,
                                self.config.num_channels,
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(batch_size, 1 + feature_size, self.config.hidden_size).astype(np.float32)]
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
                                batch_size, self.config.image_size, self.config.image_size, self.config.num_channels
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(batch_size, 1 + feature_size, self.config.hidden_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    # FIXME: Remove below after verification
    def forward_vision(self, pixel_values):
        """Forward pass for the vision model.

        Args:
            pixel_values (torch.FloatTensor): Pixel values of the images.

        Returns:
            torch.FloatTensor: Projected image features.
        """
        img_embeds = pixel_values.to(self.device_list[0])
        img_sizes = self.img_sizes
        logger.debug(f'Fixed shape image size: {img_sizes}')

        if self.use_hd_transform:
            # We directly use 4D tensor here.
            # img_embeds: (num_images, max_num_crops, 3, H, W)
            # img_sizes: (num_images, 2).view(1, -1)

            # Only supports single images here
            bs = 1
            # Nx(HW)xC
            img_features = self.get_img_features(img_embeds)
            base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)

            assert base_feat_height == 24 and base_feat_width == 24, (
                f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width},'
            )
            ' expect 24x24 features for hd transform'

            # bs x max_num_crops x (24x24) x C
            img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
            c = self.image_dim_out
            H = 24  # base_feat_height  # noqa: N806

            output_imgs = []
            output_len = []
            # training is tensor, inference is list
            if isinstance(img_sizes, torch.Tensor):
                img_sizes = img_sizes.view(-1, 2)
            for _bs in range(bs):
                h, w = img_sizes[_bs]
                h = h // 336
                w = w // 336
                b_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = img_features[_bs, :1]
                logger.debug(f'global_img_feature: {global_img_feature.shape}')

                # 1 x 12 x 12 x 4096
                glb_img_golden = (
                    global_img_feature.reshape(1, H, H, c)
                    .reshape(1, H // 2, 2, H // 2, 2, c)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, H // 2, H // 2, 4 * c)
                    .contiguous()
                )

                glb_img = (
                    global_img_feature.reshape(1, H, H, c)
                    .reshape(H, H, c)
                    .reshape(H, H // 2, 2, c)
                    .contiguous()
                    .reshape(H, H // 2, 2 * c)
                    .reshape(H // 2, 2, H // 2, 2 * c)
                    .permute(0, 2, 1, 3)
                    .reshape(H // 2, H // 2, 4 * c)
                    .contiguous()
                    .reshape(1, H // 2, H // 2, 4 * c)
                    .contiguous()
                )
                assert torch.all(glb_img == glb_img_golden)

                temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)  # noqa: N806

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, 4 * c)

                if img_sizes[_bs][0] != 336 or img_sizes[_bs][1] != 336:  # batch size != 1
                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:b_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img_backup = sub_img
                    sub_img_golden = (
                        sub_img.reshape(b_, H, H, c)
                        .reshape(b_, H // 2, 2, H // 2, 2, c)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(b_, -1, 4 * c)
                        .contiguous()
                    )

                    sub_img_4D = []  # noqa: N806
                    for i in range(b_):
                        single_sub_img = sub_img_backup[i]
                        single_sub_img = (
                            single_sub_img.reshape(H, H // 2, 2, c)
                            .reshape(H, H // 2, 2 * c)
                            .reshape(H // 2, 2, H // 2, 2 * c)
                            .permute(0, 2, 1, 3)
                            .reshape(H // 2, H // 2, 4 * c)
                            .reshape(1, H // 2, H // 2, 4 * c)
                            .contiguous()
                        )
                        sub_img_4D.append(single_sub_img)
                    sub_img = torch.cat(sub_img_4D, dim=0).reshape(b_, -1, 4 * c).contiguous()
                    assert torch.all(sub_img == sub_img_golden)

                    sub_img_golden = (
                        sub_img_golden.reshape(1, h, w, 12, 12, -1)
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(1, h * 12, w * 12, 4 * c)
                    )

                    sub_img = (
                        sub_img.reshape(b_, 12, 12, 4 * c)
                        .reshape(b_, 12, 12 * 4 * c)
                        .reshape(h, w, 12, 12 * 4 * c)
                        .permute(0, 2, 1, 3)
                        .reshape(h * 12, w, 12 * 4 * c)
                        .reshape(h * 12, w, 12, 4 * c)
                        .reshape(h * 12, w * 12, 4 * c)
                        .reshape(1, h * 12, w * 12, 4 * c)
                        .contiguous()
                    )
                    assert torch.all(sub_img == sub_img_golden)
                    temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)  # noqa: N806
                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, 4 * c)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        logger.error(
                            f'hd_transform_order = {self.hd_transform_order}, not implemented', err=NotImplementedError
                        )
                    temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                else:
                    output_imgs.append(glb_img)
                    temp_len = 156
                assert temp_len == output_imgs[-1].shape[1], (
                    f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                )
                output_len.append(temp_len)

            """
            img_set_tensor = []
            for _output_img in output_imgs:
                img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                img_set_tensor.append(img_feature_proj)
            logger.debug(
                f'img_embeds size: {img_embeds.size()}, image sizes: {img_sizes} '
                f'loading time {datetime.now() - start_time}'
            )
            """
        elif not self.use_hd_transform and img_embeds.ndim == 4:
            start_time = datetime.now()
            self.get_img_features(img_embeds).reshape(-1, self.image_dim_out)
            logger.debug(f'img_embeds size: {img_embeds.size()}, loading time {datetime.now() - start_time}')
            # img_set_tensor = self.img_projection(tt)  # adapted visual features.
        elif not self.use_hd_transform and img_embeds.ndim == 3:
            img_embeds.view(-1, self.image_dim_out)
            # img_set_tensor = self.img_projection(tt)  # adapted visual features.
        else:
            raise NotImplementedError
        return output_imgs

    def forward_projector(self, img_features, ndim=5):
        """Forward pass for the image projector.

        Args:
            img_features (torch.FloatTensor): Image features.
            ndim (int): Number of dimensions of the input tensor.

        Returns:
            torch.FloatTensor: Projected image features.
        """
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype
        if self.use_hd_transform:  # and img_sizes is not None and len(img_sizes):
            # We directly use 4D tensor here.
            img_set_tensor = []
            for _output_img in img_features:
                img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                img_set_tensor.append(img_feature_proj)
        elif (not self.use_hd_transform and ndim == 4) or (not self.use_hd_transform and ndim == 3):
            img_set_tensor = self.img_projection(img_features)  # adapted visual features.
        else:
            raise NotImplementedError
        return img_set_tensor

    def forward_dynamic(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass for the model.

        Args:
            input_ids (torch.FloatTensor): Input IDs.
            pixel_values (torch.FloatTensor): Pixel values of the images.

        Returns:
            torch.FloatTensor: image features.
        """
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).to(device=pixel_values.device, dtype=torch.int64)

        MAX_INPUT_ID = int(1e9)  # noqa: N806
        img_embeds = pixel_values

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        img_sizes = self.img_sizes

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        with torch.no_grad():
            positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)

        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        if len(positions.tolist()) > 0:
            # if True:
            with torch.no_grad():
                g_values = abs(input_ids[positions[:, 0], positions[:, 1]])

            if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                assert img_embeds.ndim == 5, f'img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'
                # img_embeds: (num_images, max_num_crops, 3, H, W)
                # img_sizes: (num_images, 2).view(1, -1)

                start_time = datetime.now()
                bs = img_embeds.shape[0]
                # Nx(HW)xC
                img_features = self.get_img_features(img_embeds.flatten(0, 1))
                logger.debug(f'img_features right after CLIP: {img_features.shape}')
                base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)

                assert base_feat_height == 24 and base_feat_width == 24, (
                    f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, '
                )
                'expect 24x24 features for hd transform'

                # bs x max_num_crops x (24x24) x C
                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                logger.debug(f'img_features right after view: {img_features.shape}')
                c = self.image_dim_out
                H = base_feat_height  # noqa: N806

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // 336
                    w = w // 336
                    b_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]
                    logger.debug(f'global_img_feature: {global_img_feature.shape}')

                    # 1 x 12 x 12 x 4096
                    glb_img_golden = (
                        global_img_feature.reshape(1, H, H, c)
                        .reshape(1, H // 2, 2, H // 2, 2, c)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(1, H // 2, H // 2, 4 * c)
                        .contiguous()
                    )

                    glb_img = (
                        global_img_feature.reshape(1, H, H, c)
                        .reshape(H, H, c)
                        .reshape(H, H // 2, 2, c)
                        .contiguous()
                        .reshape(H, H // 2, 2 * c)
                        .reshape(H // 2, 2, H // 2, 2 * c)
                        .permute(0, 2, 1, 3)
                        .reshape(H // 2, H // 2, 4 * c)
                        .contiguous()
                        .reshape(1, H // 2, H // 2, 4 * c)
                        .contiguous()
                    )
                    assert torch.all(glb_img == glb_img_golden)

                    temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)  # noqa: N806

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, 4 * c)

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:b_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img_backup = sub_img
                    sub_img_golden = (
                        sub_img.reshape(b_, H, H, c)
                        .reshape(b_, H // 2, 2, H // 2, 2, c)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(b_, -1, 4 * c)
                        .contiguous()
                    )

                    sub_img_4D = []  # noqa: N806
                    for i in range(b_):
                        single_sub_img = sub_img_backup[i]
                        single_sub_img = (
                            single_sub_img.reshape(H, H // 2, 2, c)
                            .reshape(H, H // 2, 2 * c)
                            .reshape(H // 2, 2, H // 2, 2 * c)
                            .permute(0, 2, 1, 3)
                            .reshape(H // 2, H // 2, 4 * c)
                            .reshape(1, H // 2, H // 2, 4 * c)
                            .contiguous()
                        )
                        sub_img_4D.append(single_sub_img)
                    sub_img = torch.cat(sub_img_4D, dim=0).reshape(b_, -1, 4 * c).contiguous()
                    assert torch.all(sub_img == sub_img_golden)

                    sub_img_golden = (
                        sub_img_golden.reshape(1, h, w, 12, 12, -1)
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(1, h * 12, w * 12, 4 * c)
                    )

                    sub_img = (
                        sub_img.reshape(b_, 12, 12, 4 * c)
                        .reshape(b_, 12, 12 * 4 * c)
                        .reshape(h, w, 12, 12 * 4 * c)
                        .permute(0, 2, 1, 3)
                        .reshape(h * 12, w, 12 * 4 * c)
                        .reshape(h * 12, w, 12, 4 * c)
                        .reshape(h * 12, w * 12, 4 * c)
                        .reshape(1, h * 12, w * 12, 4 * c)
                        .contiguous()
                    )
                    assert torch.all(sub_img == sub_img_golden)
                    temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)  # noqa: N806
                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, 4 * c)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        logger.error(
                            f'hd_transform_order = {self.hd_transform_order}, not implemented', err=NotImplementedError
                        )

                    temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                    assert temp_len == output_imgs[-1].shape[1], (
                        f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                    )
                    output_len.append(temp_len)

                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                    img_set_tensor.append(img_feature_proj)
                logger.debug(
                    f'img_embeds size: {img_embeds.size()}, image sizes: {img_sizes} '
                    f'loading time {datetime.now() - start_time}'
                )
            elif img_embeds.ndim == 4:
                selected_g_values = g_values[:: self.num_img_tokens]
                assert len(img_embeds) == len(selected_g_values), (
                    f'img_embeds size: {img_embeds.size()}, selected_g_values size: {len(selected_g_values)},'
                )
                f' selected_g_value {selected_g_values}'
                start_time = datetime.now()
                tt = (
                    self.get_img_features(img_embeds).to(target_device).to(target_dtype).reshape(-1, self.image_dim_out)
                )
                logger.debug(f'img_embeds size: {img_embeds.size()}, loading time {datetime.now() - start_time}')
                img_set_tensor = self.img_projection(tt)  # adapted visual features.
            elif img_embeds.ndim == 3:
                selected_g_values = g_values[:: self.num_img_tokens]
                assert len(img_embeds) == len(selected_g_values), (
                    f'img_embeds size: {img_embeds.size()}, selected_g_values size: {len(selected_g_values)},'
                )
                f' selected_g_value {selected_g_values}'
                tt = img_embeds.to(target_device).to(target_dtype).view(-1, self.image_dim_out)
                img_set_tensor = self.img_projection(tt)  # adapted visual features.
            else:
                raise NotImplementedError

        """
        with torch.no_grad():
            input_ids.clamp_min_(0).clamp_max_(self.vocab_size)

        if isinstance(self._wte, torch.nn.Module):  # Using uniform quant
            hidden_states = self._wte(input_ids).detach().cpu().numpy()
        else:
            hidden_states = self._wte.run([input_ids.detach().cpu().numpy().astype(np.int32)], dequantize_output=True)[
                0
            ]
        hidden_states = torch.from_numpy(hidden_states).to(input_ids.device)
        """

        # original
        # hidden_states = self._wte(input_ids)

        # Already move to phi3v get embeds
        """
        if select:
            if hd_transform:
                idx = 0
                for i, cnt in enumerate(num_img_tokens):
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                        img_set_tensor[i].to(hidden_states.dtype).to(hidden_states.device)
                    )
                    idx += cnt
            else:
                idx = 0
                assert len(selected_g_values) * self.num_img_tokens == len(img_set_tensor), (
                    f'len(selected_g_values) * self.num_img_tokens = {len(selected_g_values) * self.num_img_tokens},'
                )
                f' len(img_set_tensor) = {len(img_set_tensor)}'
                for i, _g in enumerate(selected_g_values):
                    cnt = self.num_img_tokens
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                        img_set_tensor[i * cnt : (i + 1) * cnt].to(hidden_states.dtype).to(hidden_states.device)
                    )
                    idx += cnt
        """

        return img_set_tensor
