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
"""Define InternViT model class."""

import json
import os

import mtk_quantization
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from ...utils import image_utils, logger, utils
from ..modeling_base import BaseVisionEncoderChunk
from ..norm import RMSNorm
from .configuration_intern_vit import InternViTConfig
from .modeling_clip import CLIPMLP


class InternVisionEmbeddings(nn.Module):
    """Embedding layer for the InternVision model, which includes patch embeddings, class embeddings, and position embeddings.

    Attributes:
        config (InternViTConfig): Configuration for the InternVision model.
        jit_trace (bool): Whether to use JIT tracing.
        embed_dim (int): The embedding dimension.
        image_size (int): The size of the input images.
        patch_size (int): The size of the patches.
        class_embedding (torch.nn.Parameter): The class embedding parameter.
        patch_embedding (torch.nn.Conv2d): The patch embedding layer.
        num_patches (int): The number of patches.
        num_positions (int): The number of positions.
        position_embedding (torch.nn.Parameter): The position embedding parameter.
        _bicubic_pos_emb (torch.Tensor): The bicubic position embeddings.

    Methods:
        _get_pos_embed(pos_embed, h, w): Get the position embeddings resized to the target height and width.
        bicubic_pos_emb(): Get the bicubic position embeddings.
        bicubic_pos_emb(pos_emb): Set the bicubic position embeddings.
        forward(pixel_values): Forward pass to generate embeddings from pixel values.
    """  # noqa: E501

    def __init__(self, config: InternViTConfig, jit_trace=False):
        """Initialize the InternVisionEmbeddings.

        Args:
            config (InternViTConfig): Configuration for the InternVision model.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.jit_trace = jit_trace

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
        self._bicubic_pos_emb = None

    def _get_pos_embed(self, pos_embed, h, w):
        """Get the position embeddings resized to the target height and width.

        Args:
            pos_embed (torch.Tensor): The original position embeddings.
            h (int): The target height.
            w (int): The target width.

        Returns:
            torch.Tensor: The resized position embeddings.
        """
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        return (
            F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
            .reshape(1, -1, h * w)
            .permute(0, 2, 1)
            .to(target_dtype)
        )

    @property
    def bicubic_pos_emb(self):
        """Get the bicubic position embeddings.

        Returns:
            torch.Tensor: The bicubic position embeddings.
        """
        return self._bicubic_pos_emb

    @bicubic_pos_emb.setter
    def bicubic_pos_emb(self, pos_emb):
        self._bicubic_pos_emb = pos_emb

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Forward pass to generate embeddings from pixel values.

        Args:
            pixel_values (torch.FloatTensor): The input pixel values.

        Returns:
            torch.Tensor: The generated embeddings.
        """
        target_dtype = self.patch_embedding.weight.dtype
        # shape = [*, channel, width, height]
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, _, _height, _width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._bicubic_pos_emb,
            ],
            dim=1,
        )
        return embeddings + position_embedding.to(target_dtype)


class InternAttention(nn.Module):
    """Multi-headed attention mechanism from the 'Attention Is All You Need' paper.

    Attributes:
        config (InternViTConfig): Configuration for the InternVision model.
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        scale (float): The scaling factor for the attention mechanism.
        qkv (torch.nn.Linear): The linear layer for query, key, and value.
        qk_normalization (bool): Whether to use query-key normalization.
        q_norm (RMSNorm): The normalization layer for query.
        k_norm (RMSNorm): The normalization layer for key.
        proj (torch.nn.Linear): The projection layer.

    Methods:
        _naive_attn(x): Compute the attention mechanism.
        forward(hidden_states): Forward pass to apply attention on hidden states.
    """

    def __init__(self, config: InternViTConfig):
        """Initialize the InternAttention.

        Args:
            config (InternViTConfig): Configuration for the InternVision model.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // self.num_heads)
        if self.head_dim * self.num_heads != self.embed_dim:
            logger.error(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).',
                err=ValueError,
            )

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.norm_eps)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        b, n, c = x.shape
        # (3, B, heads, N, C // heads)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 3, 1)  # (3, B, C, N)
        q, k, v = qkv.unbind(0)  # (B, C, N)
        q = q.reshape(b, self.num_heads, c // self.num_heads, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, n).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, c // self.num_heads, n).permute(0, 1, 3, 2)

        if self.qk_normalization:
            b_, h_, n_, d_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(b_, n_, h_, d_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(b_, n_, h_, d_).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)

    def forward(self, hidden_states):
        """Forward pass to apply attention on hidden states.

        Args:
            hidden_states (torch.Tensor): The input hidden states.

        Returns:
            torch.Tensor: The output hidden states after applying attention.
        """
        return self._naive_attn(hidden_states)


class InternVisionEncoderLayer(nn.Module):
    """Encoder layer for the InternVision model, identical to `CLIPEncoderLayer` with layer scaling.

    Attributes:
        config (InternViTConfig): Configuration for the InternVision model.
        embed_dim (int): The embedding dimension.
        intermediate_size (int): The intermediate size for the MLP.
        attn (InternAttention): The attention layer.
        mlp (CLIPMLP): The MLP layer.
        norm1 (torch.nn.LayerNorm): The first normalization layer.
        norm2 (torch.nn.LayerNorm): The second normalization layer.
        ls1 (torch.nn.Parameter): The layer scaling parameter for the attention layer.
        ls2 (torch.nn.Parameter): The layer scaling parameter for the MLP layer.

    Methods:
        forward(hidden_states): Forward pass for the encoder layer.
    """

    def __init__(self, config: InternViTConfig):
        """Initialize the InternVisionEncoderLayer.

        Args:
            config (InternViTConfig): Configuration for the InternVision model.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternAttention(config)
        self.mlp = CLIPMLP(config)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.norm_eps)

        self.ls1 = nn.Parameter(1.0 * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(1.0 * torch.ones(self.embed_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Forward pass for the encoder layer.

        Args:
            hidden_states (torch.Tensor): The input hidden states.

        Returns:
            torch.Tensor: The output hidden states after processing.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        return hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2


class InternVisionEncoder(nn.Module):
    """Encoder for the InternVision model, consisting of multiple encoder layers.

    Attributes:
        config (InternViTConfig): Configuration for the InternVision model.
        device_list (list): List of devices for each encoder layer.
        layers (torch.nn.ModuleList): List of encoder layers.

    Methods:
        forward(inputs_embeds): Forward pass for the encoder.
    """

    def __init__(self, config: InternViTConfig):
        """Initialize the InternVisionEncoder.

        Args:
            config (InternViTConfig): Configuration for the InternVision model.
        """
        super().__init__()
        self.config = config
        self.device_list = []
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds):
        """Forward pass for the encoder.

        Args:
            inputs_embeds (torch.Tensor): The input embeddings.

        Returns:
            tuple: The final hidden states and all encoder states.
        """
        encoder_states = ()

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = (*encoder_states, hidden_states)

            hidden_states = encoder_layer(hidden_states.to(self.device_list[idx]))

        encoder_states = (*encoder_states, hidden_states)

        return hidden_states, encoder_states


class InternVisionModel(BaseVisionEncoderChunk):
    """InternVision model, which includes embeddings and encoder.

    Attributes:
        config (InternViTConfig): Configuration for the InternVision model.
        jit_trace (bool): Whether to use JIT tracing.
        num_layers (int): The number of hidden layers.
        main_device (str): The main device for the model.
        embeddings (InternVisionEmbeddings): The embedding layer.
        encoder (InternVisionEncoder): The encoder.

    Methods:
        precompute_pos_embeddings(): Precompute the bicubic position embeddings.
        resize_pos_embeddings(old_size, new_size, patch_size): Resize the position embeddings to a new size.
        get_input_embeddings(): Get the input embeddings.
        forward(pixel_values): Forward pass for the InternVision model.
    """

    def __init__(
        self,
        config: InternViTConfig,
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
        """Initializes the InternVisionModel class.

        Args:
            config (InternViTConfig): The configuration for the InternVision model.
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
        torch.set_default_dtype(dtype)

        self.config = config
        self.jit_trace = jit_trace
        self.main_device = 'cuda:0'

        if self.chunk_idx == 0:
            self.embeddings = InternVisionEmbeddings(config, jit_trace=jit_trace)
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for _ in range(self.num_layers)])

        image_size = self.config.image_size
        patch_size = self.config.patch_size
        self._num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio**2))

    @property
    def num_image_token(self):
        """Get the InternViT number of tokens."""
        return self._num_image_token

    def precompute_pos_embeddings(self):
        """Precompute the bicubic position embeddings."""
        logger.debug('Enter InternViT precompute_pos_embeddings.')
        height = self.config.image_size // self.config.patch_size
        width = self.config.image_size // self.config.patch_size
        bicubic_pos_emb = self.embeddings._get_pos_embed(  # noqa: SLF001
            self.embeddings.position_embedding[:, 1:, :], height, width
        ).detach()
        self.embeddings.bicubic_pos_emb = bicubic_pos_emb

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """Resize the position embeddings to a new size.

        Args:
            old_size (int): The old size of the position embeddings.
            new_size (int): The new size of the position embeddings.
            patch_size (int): The patch size.
        """
        pos_emb = self.embeddings.position_embedding
        _, _num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size

    def get_input_embeddings(self):
        """Get the input embeddings.

        Returns:
            InternVisionEmbeddings: The input embeddings.
        """
        return self.embeddings

    def load_vision_weight(self, state_dict, prefix=''):
        """Load internvit model weights.

        Args:
            state_dict (dict): The model state_dict.
            prefix (str): The prefix for the state dictionary keys. Defaults to ''.
        """
        state_dict = self._pop_redundant_prefix(state_dict, prefix='vision_model.')
        temp_state_dict = {}
        mapping_keys_list = []

        # embeddings
        temp_state_dict = {
            **temp_state_dict,
            'embeddings.class_embedding': state_dict.pop('embeddings.class_embedding'),
            'embeddings.patch_embedding.bias': state_dict.pop('embeddings.patch_embedding.bias'),
            'embeddings.patch_embedding.weight': state_dict.pop('embeddings.patch_embedding.weight'),
            'embeddings.position_embedding': state_dict.pop('embeddings.position_embedding'),
        }
        mapping_keys_list = [
            'embeddings.class_embedding',
            'embeddings.patch_embedding.bias',
            'embeddings.patch_embedding.weight',
            'embeddings.position_embedding',
        ]

        # encoder layers
        for idx in range(self.vision_config.num_hidden_layers):
            temp_state_dict = {
                **temp_state_dict,
                f'encoder.layers.{idx}.attn.proj.bias': state_dict.pop(f'encoder.layers.{idx}.attn.proj.bias'),
                f'encoder.layers.{idx}.attn.proj.weight': state_dict.pop(f'encoder.layers.{idx}.attn.proj.weight'),
                f'encoder.layers.{idx}.attn.qkv.bias': state_dict.pop(f'encoder.layers.{idx}.attn.qkv.bias'),
                f'encoder.layers.{idx}.attn.qkv.weight': state_dict.pop(f'encoder.layers.{idx}.attn.qkv.weight'),
                f'encoder.layers.{idx}.ls1': state_dict.pop(f'encoder.layers.{idx}.ls1'),
                f'encoder.layers.{idx}.ls2': state_dict.pop(f'encoder.layers.{idx}.ls2'),
                f'encoder.layers.{idx}.mlp.fc1.bias': state_dict.pop(f'encoder.layers.{idx}.mlp.fc1.bias'),
                f'encoder.layers.{idx}.mlp.fc1.weight': state_dict.pop(f'encoder.layers.{idx}.mlp.fc1.weight'),
                f'encoder.layers.{idx}.mlp.fc2.bias': state_dict.pop(f'encoder.layers.{idx}.mlp.fc2.bias'),
                f'encoder.layers.{idx}.mlp.fc2.weight': state_dict.pop(f'encoder.layers.{idx}.mlp.fc2.weight'),
                f'encoder.layers.{idx}.norm1.bias': state_dict.pop(f'encoder.layers.{idx}.norm1.bias'),
                f'encoder.layers.{idx}.norm1.weight': state_dict.pop(f'encoder.layers.{idx}.norm1.weight'),
                f'encoder.layers.{idx}.norm2.bias': state_dict.pop(f'encoder.layers.{idx}.norm2.bias'),
                f'encoder.layers.{idx}.norm2.weight': state_dict.pop(f'encoder.layers.{idx}.norm2.weight'),
            }
            mapping_keys_list = [
                *mapping_keys_list,
                f'encoder.layers.{idx}.attn.proj.bias',
                f'encoder.layers.{idx}.attn.proj.weight',
                f'encoder.layers.{idx}.attn.qkv.bias',
                f'encoder.layers.{idx}.attn.qkv.weight',
                f'encoder.layers.{idx}.attn.qkv.weight',
                f'encoder.layers.{idx}.ls1',
                f'encoder.layers.{idx}.ls2',
                f'encoder.layers.{idx}.mlp.fc1.bias',
                f'encoder.layers.{idx}.mlp.fc1.weight',
                f'encoder.layers.{idx}.mlp.fc2.bias',
                f'encoder.layers.{idx}.mlp.fc2.weight',
                f'encoder.layers.{idx}.norm1.bias',
                f'encoder.layers.{idx}.norm1.weight',
                f'encoder.layers.{idx}.norm2.bias',
                f'encoder.layers.{idx}.norm2.weight',
            ]
            if torch.cuda.device_count() == 0 or self.jit_trace:
                self.vision_device_list.append('cpu')
            else:
                device_id = idx // (
                    self.vision_config.num_hidden_layers // torch.cuda.device_count()
                    + (self.vision_config.num_hidden_layers % torch.cuda.device_count() != 0)
                )
                self.vision_device_list.append(f'cuda:{device_id}')

        # mm_projector
        mm_projector_state_dict, mm_projector_mapping_keys_list = self.mm_projector.load_mm_projector_state_dict
        temp_state_dict = {**temp_state_dict, **mm_projector_state_dict}
        mapping_keys_list = [*mapping_keys_list, *mm_projector_mapping_keys_list]

        try:
            self.load_state_dict(temp_state_dict, strict=True)
            self.eval()
            return True, self
        except RuntimeError:
            return False, mapping_keys_list

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {}
        if self.chunk_idx == 0:
            state_dict_mapping = {
                'class_embedding': {'embeddings.class_embedding': 'vision_model.embeddings.class_embedding'},
                'patch_embedding_bias': {
                    'embeddings.patch_embedding.bias': 'vision_model.embeddings.patch_embedding.bias'
                },
                'patch_embedding_weight': {
                    'embeddings.patch_embedding.weight': 'vision_model.embeddings.patch_embedding.weight'
                },
                'position_embedding': {'embeddings.position_embedding': 'vision_model.embeddings.position_embedding'},
            }

        # fmt: off
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'layers.{outer_layer_idx}.attn_proj_bias': {
                        f'layers.{inner_layer_idx}.attn.proj.bias':
                        f'encoder.layers.{outer_layer_idx}.attn.proj.bias'
                    },
                    f'layers.{outer_layer_idx}.attn_proj_weight': {
                        f'layers.{inner_layer_idx}.attn.proj.weight':
                        f'encoder.layers.{outer_layer_idx}.attn.proj.weight'
                    },
                    f'layers.{outer_layer_idx}.attn_qkv_bias': {
                        f'layers.{inner_layer_idx}.attn.qkv.bias':
                        f'encoder.layers.{outer_layer_idx}.attn.qkv.bias'
                    },
                    f'layers.{outer_layer_idx}.attn_qkv_weight': {
                        f'layers.{inner_layer_idx}.attn.qkv.weight':
                        f'encoder.layers.{outer_layer_idx}.attn.qkv.weight'
                    },
                    f'layers.{outer_layer_idx}_ls1_weight': {
                        f'layers.{inner_layer_idx}.ls1':
                        f'encoder.layers.{outer_layer_idx}.ls1'
                    },
                    f'layers.{outer_layer_idx}_ls2_weight': {
                        f'layers.{inner_layer_idx}.ls2':
                        f'encoder.layers.{outer_layer_idx}.ls2'
                    },
                    f'layers.{outer_layer_idx}.mlp_fc1_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc1.bias':
                        f'encoder.layers.{outer_layer_idx}.mlp.fc1.bias'
                    },
                    f'layers.{outer_layer_idx}.mlp_fc1_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc1.weight':
                        f'encoder.layers.{outer_layer_idx}.mlp.fc1.weight'
                    },
                    f'layers.{outer_layer_idx}.mlp_fc2_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc2.bias':
                        f'encoder.layers.{outer_layer_idx}.mlp.fc2.bias'
                    },
                    f'layers.{outer_layer_idx}.mlp_fc2_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc2.weight':
                        f'encoder.layers.{outer_layer_idx}.mlp.fc2.weight'
                    },
                    f'layers.{outer_layer_idx}_norm1_bias': {
                        f'layers.{inner_layer_idx}.norm1.bias':
                        f'encoder.layers.{outer_layer_idx}.norm1.bias'
                    },
                    f'layers.{outer_layer_idx}_norm1_weight': {
                        f'layers.{inner_layer_idx}.norm1.weight':
                        f'encoder.layers.{outer_layer_idx}.norm1.weight'
                    },
                    f'layers.{outer_layer_idx}_norm2_bias': {
                        f'layers.{inner_layer_idx}.norm2.bias':
                        f'encoder.layers.{outer_layer_idx}.norm2.bias'
                    },
                    f'layers.{outer_layer_idx}_norm2_weight': {
                        f'layers.{inner_layer_idx}.norm2.weight':
                        f'encoder.layers.{outer_layer_idx}.norm2.weight'
                    },
                }
            )
        # fmt: on
        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter InternVisionModel load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

        # state_dict = self._pop_text_model_weights(state_dict)

        self.device_list = []
        prefixes = ['']
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
            for pre in prefixes:
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
                        prefixes.append(k[: -len(external_key)])
                        weights_to_load.update({model_key: state_dict.pop(k).to(dtype)})
                        found = True
                        state_dict_keys.remove(k)
                        break

            if not found and internal_key.endswith('_bias'):
                # Bias key not found, default to zeros
                # Default shapes:
                # fc1: intermediate_size
                # All others: hidden size
                shape = self.config.intermediate_size if internal_key.endswith('_fc1_bias') else self.config.hidden_size
                logger.debug(f'Init bias {internal_key} to zeros using shape={shape} and dtype={dtype}')
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

        if len(prefixes) > 2:
            logger.warning(
                f'More than 1 prefix found (found {prefixes[1:]}). '
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
                        prefix=prefixes[-1],
                    )
                act_targets = quant_config_dict['quantizer_targets']['activations']
                for act in act_targets:
                    weights_to_load = self._add_quantizer_weights(
                        state_dict,
                        weights_to_load,
                        act,
                        'activation',
                        prefix=prefixes[-1],
                    )

        self.load_state_dict(weights_to_load, strict=False)

        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        if self.first_layer_idx == 0:
            self.embeddings.to(self.device_list[0])
        # self.pre_layernorm.to(self.device_list[0])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        # state_dict = self._pop_unused_layer_weights(state_dict, prefixes)

        # precompute bicubic position embeddings
        if self.chunk_idx == 0:
            self.precompute_pos_embeddings()

        return self, state_dict

    @image_utils.NHWCWrapper
    def forward(self, pixel_values, **kwargs):
        """Forward pass for the InternVision model.

        Args:
            pixel_values (torch.Tensor): The input pixel values.
            kwargs: Other kwargs.

        Returns:
            tuple: The final hidden states and all encoder outputs.
        """
        hidden_states = pixel_values.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.embeddings(hidden_states)

        # hidden_states, encoder_outputs = self.encoder(hidden_states)

        encoder_states = ()
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = (*encoder_states, hidden_states)

            hidden_states = encoder_layer(hidden_states.to(self.device_list[idx]))

        encoder_states = (*encoder_states, hidden_states)
        return hidden_states  # , encoder_states

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
