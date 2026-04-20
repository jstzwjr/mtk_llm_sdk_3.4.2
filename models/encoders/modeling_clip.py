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
"""Define CLIP ViT model class."""

import json
import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import image_utils, logger, utils
from ..activations import FastGelu, QuickGelu
from ..modeling_base import BaseVisionEncoderChunk
from .configuration_clip import CLIPConfig


class CLIPVisionEmbeddings(nn.Module):
    """CLIP Vision Embeddings class.

    This class is responsible for creating the vision embeddings for the CLIP model.

    Attributes:
        config (CLIPConfig): The configuration for the CLIP model.
        hidden_size (int): The hidden size of the embeddings.
        image_size (int): The size of the input images.
        patch_size (int): The size of the patches.
        class_embedding (nn.Parameter): The class embedding parameter.
        patch_embedding (nn.Conv2d): The convolutional layer for patch embedding.
        jit_trace (bool): Whether to use JIT tracing.
        num_patches (int): The number of patches.
        num_positions (int): The number of positions.
        position_embedding (nn.Embedding): The position embedding layer.
        position_ids (torch.Tensor): The position IDs.

    Methods:
        __init__: Initializes the CLIPVisionEmbeddings class.
        forward: Performs the forward pass to compute the embeddings.
    """

    def __init__(self, config: CLIPConfig, jit_trace=False):
        """Initializes the CLIPVisionEmbeddings class.

        Args:
            config (CLIPConfig): The configuration for the CLIP model.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.hidden_size))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.jit_trace = jit_trace
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.hidden_size)
        self.position_ids = torch.arange(self.num_positions).expand((1, -1))

    def forward(self, pixel_values):
        """Performs the forward pass to compute the embeddings.

        Args:
            pixel_values (torch.Tensor): The input pixel values.

        Returns:
            torch.Tensor: The computed embeddings.
        """
        batch_size = pixel_values.shape[0]
        pixel_values = self.patch_embedding(pixel_values)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, pixel_values], dim=1)
        return embeddings + self.position_embedding(self.position_ids.to(pixel_values.device))


class CLIPAttention(nn.Module):
    """CLIP Attention class.

    This class implements the attention mechanism for the CLIP model.

    Attributes:
        config: The configuration for the CLIP model.
        hidden_size (int): The hidden size of the attention mechanism.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        scale (float): The scaling factor for the query projection.
        k_proj (nn.Linear): The linear layer for the key projection.
        v_proj (nn.Linear): The linear layer for the value projection.
        q_proj (nn.Linear): The linear layer for the query projection.
        out_proj (nn.Linear): The linear layer for the output projection.

    Methods:
        __init__: Initializes the CLIPAttention class.
        forward: Performs the forward pass to compute the attention output.
    """

    def __init__(self, config):
        """Initializes the CLIPAttention class.

        Args:
            config: The configuration for the CLIP model.

        Raises:
            ValueError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        if self.head_dim * self.num_heads != self.hidden_size:
            logger.error(
                f'hidden_size must be divisible by num_heads (got `hidden_size`={self.hidden_size} and `num_heads`='
                f'{self.num_heads}).',
                err=ValueError,
            )
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states):
        """Performs the forward pass to compute the attention output.

        Input shape: Batch x Time x Channel.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.
            causal_attention_mask (torch.Tensor, optional): The causal attention mask. Defaults to None.

        Returns:
            torch.Tensor: The attention output.

        Raises:
            ValueError: If the size of the attention mask or causal attention mask is incorrect.
        """
        bsz, q_len, hidden_size = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = (
            query_states.contiguous()
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(*proj_shape)
        )
        key_states = key_states.contiguous().view(*proj_shape)
        value_states = value_states.contiguous().view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    """Initializes the CLIPMLP class.

    Args:
        config: The configuration for the CLIP model.
    """

    def __init__(self, config):
        """Initializes the CLIPMLP class.

        Args:
            config: The configuration for the CLIP model.
        """
        super().__init__()
        self.config = config
        act_name = getattr(self.config, 'mlp_gelu', 'quick_gelu')
        if act_name == 'gelu_pytorch_tanh':
            self.gelu = FastGelu()
        elif act_name == 'quick_gelu':
            self.gelu = QuickGelu()
        else:
            logger.error(f'Unsupported CLIP mlp_gelu type: {act_name}')
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass to compute the MLP output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.

        Returns:
            torch.Tensor: The MLP output.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return self.fc2(hidden_states)


class CLIPEncoderLayer(nn.Module):
    """CLIP Encoder Layer class.

    This class implements a single encoder layer for the CLIP model.

    Attributes:
        embed_dim (int): The embedding dimension.
        self_attn (CLIPAttention): The self-attention mechanism.
        layer_norm1 (nn.LayerNorm): The first layer normalization.
        mlp (CLIPMLP): The Multi-Layer Perceptron (MLP).
        layer_norm2 (nn.LayerNorm): The second layer normalization.

    Methods:
        __init__: Initializes the CLIPEncoderLayer class.
        forward: Performs the forward pass to compute the encoder layer output.
    """

    def __init__(self, config: CLIPConfig):
        """Initializes the CLIPEncoderLayer class.

        Args:
            config (CLIPConfig): The configuration for the CLIP model.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.norm_eps)

    def forward(self, hidden_states):
        """Performs the forward pass to compute the encoder layer output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.
            causal_attention_mask (torch.Tensor, optional): The causal attention mask. Defaults to None.

        Returns:
            torch.Tensor: The encoder layer output.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CLIPVisionEncoderChunk(BaseVisionEncoderChunk):
    """CLIP Vision Transformer class.

    This class implements the vision transformer for the CLIP model.

    Attributes:
        config (CLIPConfig): The configuration for the CLIP model.
        jit_trace (bool): Whether to use JIT tracing.
        num_blocks (int): The number of hidden layers.
        main_device: The main device.
        embeddings (CLIPVisionEmbeddings): The vision embeddings.
        pre_layernorm (nn.LayerNorm): The layer normalization before the encoder.
        encoder (CLIPEncoder): The encoder.

    Methods:
        __init__: Initializes the CLIPVisionEncoderChunk class.
        num_patches_per_side: Returns the number of patches per side.
        num_patches: Returns the total number of patches.
        hidden_size: Returns the hidden size.
        forward: Performs the forward pass to compute the vision transformer output.
    """

    def __init__(
        self,
        config: CLIPConfig,
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
        """Initializes the CLIPVisionEncoderChunk class.

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
            self.embeddings = CLIPVisionEmbeddings(config, jit_trace=jit_trace)
            self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(self.num_layers)])

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

    def _pop_text_model_weights(self, state_dict):
        return {k: v for (k, v) in state_dict.items() if not (k.startswith(('text_model.', 'text_projection.')))}

    def pop_remaining_unused_weights(self, state_dict):
        """Remove unused encoder weights AFTER loading encoder weights.

        Args:
            state_dict: loaded state_dict.
        """
        # All unused, so no internal key
        unused_state_dict_mapping = {
            'post_norm_weight': {'_': f'{self.norm_names["post"]}.weight'},
            'post_norm_bias': {'_': f'{self.norm_names["post"]}.bias'},
            'logit_scale': {'_': 'logit_scale'},
            'logit_bias': {'_': 'logit_bias'},
            'position_ids': {'_': 'vision_model.embeddings.position_ids'},
            'visual_projection_weight': {'_': 'visual_projection.weight'},
        }
        state_dict_keys = list(state_dict.keys())
        for internal_key, mapping_dict in unused_state_dict_mapping.items():
            external_key = mapping_dict[next(iter(mapping_dict))]
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(f'Remove {internal_key} weight as unused: {key_to_test}')
                    state_dict.pop(key_to_test)
                    state_dict_keys.remove(key_to_test)
                    break

        # Pop remaining vision model weights containing vision prefixes
        if len(self.prefixes) > 1:
            prefixes = self.prefixes[1:]
            for pre in prefixes:
                for key in state_dict_keys:
                    if key.startswith(pre):
                        state_dict.pop(key)
        return state_dict

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

        return self, state_dict

    @image_utils.NHWCWrapper
    def forward(self, pixel_values, **kwargs):
        """Performs the forward pass to compute the vision transformer output.

        Args:
            pixel_values (torch.Tensor): The input pixel values.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output hidden states.
        """
        hidden_states = pixel_values.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.embeddings(hidden_states)
            hidden_states = self.pre_layernorm(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states.to(self.device_list[idx]))

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
        return torch.randn(1, 1 + feature_size, self.config.hidden_size, device='cpu', dtype=torch.float32)

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
            input_shapes = [[None, 1 + feature_size, self.config.hidden_size]]
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
                        yield [np.random.rand(1, 1 + feature_size, self.config.hidden_size).astype(np.float32)]
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
                        yield [np.random.rand(1, 1 + feature_size, self.config.hidden_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
