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
"""Define InternViT Navit Rope model class."""

import math
import os

import mtk_quantization
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn

from ...utils import logger, utils
from ..norm import RMSNorm
from .configuration_internvl_navit_rope import InternViTNavitRopeConfig
from .modeling_clip import CLIPMLP
from .modeling_intern_vit import InternVisionModel

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to the vision tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        freqs (torch.Tensor): The frequency tensor.

    Returns:
        torch.Tensor: The tensor with applied rotary positional embedding.
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Vision Rotary Embedding class for applying rotary positional embeddings in vision models.

    Attributes:
        inv_freq (torch.Tensor): The inverse frequency tensor.

    Methods:
        __init__(dim, theta): Initialize the Vision Rotary Embedding.
        forward(seqlen): Forward pass to compute rotary positional embeddings.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Initialize the Vision Rotary Embedding.

        Args:
            dim (int): The dimension of the embedding.
            theta (float, optional): The base frequency. Defaults to 10000.0.
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen) -> torch.Tensor:
        """Forward pass to compute rotary positional embeddings.

        Args:
            seqlen (int): The sequence length.

        Returns:
            torch.Tensor: The computed rotary positional embeddings.
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def precompute_vision_rotary_embedding(head_dim, rope_theta, grid_hw):
    """Precompute InternViT Navit Rope vision rotary embedding."""
    pos_ids = []
    rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, rope_theta)
    for h, w in grid_hw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // 2,
            2,
            w // 2,
            2,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // 2,
            2,
            w // 2,
            2,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_hw.max()
    rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    rotary_pos_emb_cos = rotary_pos_emb.cos().unsqueeze(1)
    rotary_pos_emb_sin = rotary_pos_emb.sin().unsqueeze(1)
    return torch.cat([rotary_pos_emb_cos, rotary_pos_emb_sin], dim=1)


def precompute_cuseqlen(grid_hw, seq_length):
    """Precompute cumulative sequence length and attention mask."""
    cu_seqlens = (grid_hw[:, 0] * grid_hw[:, 1]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(torch.float32).min)
    for i in range(1, len(cu_seqlens)):
        attention_mask[
            ...,
            cu_seqlens[i - 1] : cu_seqlens[i],
            cu_seqlens[i - 1] : cu_seqlens[i],
        ] = 0
    return attention_mask


class InternVLNavitRopePatchEmbed(nn.Module):
    """Patch embedding of InternViT Navit Rope."""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
        jit_trace=False,
    ) -> None:
        """Initialize the InternVisionEmbeddings.

        Args:
            patch_size: Patch size.
            in_channels: Input feature channels.
            embed_dim: Patch embedding dimension.
            jit_trace: Dummy.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward funtion.

        Args:
            hidden_states (torch.Tensor): input tensor of shape [seq_len, in_channels*patch_size*patch_size].

        Returns:
            torch.Tensor: output tensor of shape [seq_len, embed].
        """
        target_dtype = self.proj.weight.dtype
        return self.proj(hidden_states.to(dtype=target_dtype))


class InternVisionAttention(nn.Module):
    """InternVL Navit Rope attention."""

    def __init__(self, config):
        """Initialize InternVisionAttention."""
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Rotary embedding modules
        self.q_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.q_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.q_add = mtk_quantization.pytorch.functional.Add()
        self.q_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        self.k_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.k_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.k_add = mtk_quantization.pytorch.functional.Add()
        self.k_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding using MTK functions.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            cos (torch.Tensor): The cosine tensor.
            sin (torch.Tensor): The sine tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with applied rotary positional embedding.
        """
        q1, q2 = torch.split(q, self.head_dim // 2, dim=-1)
        # q1 = q[..., : torch.div(q.shape[-1], 2, rounding_mode='floor')]
        # q2 = q[..., torch.div(q.shape[-1], 2, rounding_mode='floor') :]
        q_rotated = self.q_cat((-q2, q1))
        k1, k2 = torch.split(k, self.head_dim // 2, dim=-1)
        # k1 = k[..., : torch.div(k.shape[-1], 2, rounding_mode='floor')]
        # k2 = k[..., torch.div(k.shape[-1], 2, rounding_mode='floor') :]
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function."""
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Official
        # q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        # k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # MTK
        cos, sin = torch.split(pos_emb, 1, dim=1)
        cos = cos.repeat(1, 1, 2).float()
        sin = sin.repeat(1, 1, 2).float()
        q, k = self.apply_rotary_pos_emb_mtk(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class InternVisionEncoderLayer(nn.Module):
    """InternVL Navit Rope encoder layer."""

    def __init__(self, config):
        """Initialize InternVisionEncoderLayer."""
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternVisionAttention(config)
        self.mlp = CLIPMLP(config)
        self.norm1 = NORM2FN[config.norm_type](self.embed_dim, eps=config.norm_eps)
        self.norm2 = NORM2FN[config.norm_type](self.embed_dim, eps=config.norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
    ):
        """Forward function.

        Args:
        hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`):
            input to the layer of shape `(batch, seq_len, embed_dim)`.
        cu_seqlens: cumulative sequence length.
        rotary_pos_emb: vision rotary embedding.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb) * self.ls1
        return hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2


class InternVLNavitRopeModel(InternVisionModel):
    """InternVLNavitRopeModel model, which includes embeddings and encoder.

    Attributes:
        config (InternViTNavitRopeConfig): Configuration for the InternVLNavitRopeModel model.
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
        config: InternViTNavitRopeConfig,
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
        """Initializes the InternVLNavitRopeModel class.

        Args:
            config (InternViTNavitRopeConfig): The configuration for the InternVision model.
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
            self.embeddings = InternVLNavitRopePatchEmbed(
                config.patch_size, config.num_channels, config.hidden_size, jit_trace=jit_trace
            )
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for _ in range(self.num_layers)])
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, self.config.rope_theta)
        self._vision_rot_emb = 1
        self._cuseqlen = 1
        self.image_grid_hw = None

    @property
    def internvl_navit_rope_vision_rot_emb(self):
        """Vision rotary embedding."""
        return self._vision_rot_emb

    @internvl_navit_rope_vision_rot_emb.setter
    def internvl_navit_rope_vision_rot_emb(self, x):
        """Setter of vision rotary embedding."""
        self._vision_rot_emb = x

    @property
    def internvl_navit_rope_cuseqlen(self):
        """Vision cumulative sequence length."""
        return self._cuseqlen

    @internvl_navit_rope_cuseqlen.setter
    def internvl_navit_rope_cuseqlen(self, x):
        """Setter of cumulative sequence length."""
        self._cuseqlen = x

    def precompute_pos_embeddings(self):
        """No need to compute pos embedding anymore."""

    def _rot_pos_emb(self, grid_hw):
        pos_ids = []
        for h, w in grid_hw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // 2,
                2,
                w // 2,
                2,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // 2,
                2,
                w // 2,
                2,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_hw.max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        super()._generate_default_state_dict_mapping()
        if self.chunk_idx == 0:
            self.state_dict_mapping.pop('class_embedding')
            self.state_dict_mapping.pop('patch_embedding_bias')
            self.state_dict_mapping.pop('patch_embedding_weight')
            self.state_dict_mapping.pop('position_embedding')
            self.state_dict_mapping.update(
                {'patch_embedding_weight': {'embeddings.proj.weight': 'vision_model.embeddings.proj.weight'}}
            )

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter InternVLNavitRopeModel load weights.')
        state_dict = self._pop_redundant_prefix(state_dict, prefix='vision_encoder.')
        return super().load_weights(state_dict, state_dict_start_idx, quant_config)

    def forward(self, pixel_values, **kwargs):
        """Forward pass for the InternVision model.

        Args:
            pixel_values (torch.Tensor): The input pixel values.
            kwargs: Other kwargs.

        Returns:
            tuple: The final hidden states and all encoder outputs.
        """
        cu_seqlens = kwargs.pop('internvl_navit_rope_cu_seqlens', self._cuseqlen)
        rotary_pos_emb = kwargs.pop('internvl_navit_rope_rotary_pos_emb', self._vision_rot_emb)
        if cu_seqlens is None:
            logger.error(
                'internvl_navit_rope_cu_seqlens must be either passed or pre-set when forwarding dynamic shape '
                'InternVL-Navit-Rope.',
                err=ValueError,
            )
        if rotary_pos_emb is None:
            logger.error(
                'qwen2_vl_rotary_pos_emb must be passed or pre-set when forwarding dynamic shape Qwen2-VL ViT.',
                err=ValueError,
            )

        hidden_states = pixel_values.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.embeddings(hidden_states)
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states.to(self.device_list[idx]),
                cu_seqlens.to(device=self.device_list[idx], dtype=self.dtype),
                rotary_pos_emb.to(self.device_list[idx], dtype=self.dtype),
            )

        return hidden_states

    def _calculate_ptq_fixed_shape_batch(self):
        from ..preprocessors.configuration_internvl_navit_rope import InternVLNavitRopePreprocessorConfig
        from ..preprocessors.preprocessor_internvl_navit_rope import InternVLNavitRopePreprocessor

        logger.debug('Enter InternVL-Navit-Rope _calculate_ptq_fixed_shape_batch.')
        preprocessor_config = InternVLNavitRopePreprocessorConfig(**self.config.preprocessor_config)
        processor = InternVLNavitRopePreprocessor(**preprocessor_config.get())
        image_hw = self.config.image_resolution
        if image_hw is None:
            logger.error('image_resolution in config must be set when PTQing InternVL-Navit-Rope.', err=ValueError)
        image = np.random.rand(image_hw[0], image_hw[1], 3)
        image = Image.fromarray(image.astype('uint8'))
        return_dict, _ = processor.preprocess([image])
        self.image_grid_hw = torch.tensor(return_dict['image_grid_hw'])
        logger.debug(f'InternVL-Navit-Rope PTQ image_grid_hw: {self.image_grid_hw}')

    def _calulate_ptq_attnmask_rotemb(self):
        seq_length = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        self._cuseqlen = precompute_cuseqlen(self.image_grid_hw, seq_length=seq_length)

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rope_theta = self.config.rope_theta
        self._vision_rot_emb = precompute_vision_rotary_embedding(head_dim, rope_theta, self.image_grid_hw)

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        self._calculate_ptq_fixed_shape_batch()
        self._calulate_ptq_attnmask_rotemb()
        fixed_batch_size = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        if self.chunk_idx == 0:
            return torch.randn(
                fixed_batch_size,
                self.config.num_channels * self.config.patch_size * self.config.patch_size,
                device='cpu',
                dtype=torch.float32,
            )
        return torch.randn(fixed_batch_size, self.config.hidden_size, device='cpu', dtype=torch.float32)

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
        fixed_batch_size = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        if self.chunk_idx == 0:
            input_shapes = [
                [fixed_batch_size, self.config.num_channels * self.config.patch_size * self.config.patch_size]
            ]
        else:
            input_shapes = [[fixed_batch_size, self.config.hidden_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [
                            np.random.rand(
                                fixed_batch_size,
                                self.config.num_channels * self.config.patch_size * self.config.patch_size,
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(fixed_batch_size, self.config.hidden_size).astype(np.float32)]
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
                                fixed_batch_size,
                                self.config.num_channels * self.config.patch_size * self.config.patch_size,
                            ).astype(np.float32)
                        ]
                    else:
                        yield [np.random.rand(fixed_batch_size, self.config.hidden_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
