# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""Define Qwen2.5VL model class."""

import json
import math
import os

import mtk_quantization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from PIL import Image

from ...utils import logger, utils
from ..modeling_base import BaseVisionEncoderChunk
from ..norm import RMSNorm


# Copied from transformers.models.llama.modeling_llama.rotate_half
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


def precompute_qwen2_5vl_vision_rot_emb(grid_thw, config):
    """Precompute rotary positional embeddings for Qwen2.5-VL vision model.

    Args:
        grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).
        config (object): The configuration object containing model parameters.

    Returns:
        torch.Tensor: The precomputed rotary positional embeddings.
    """
    pos_ids = []
    spatial_merge_size = config.spatial_merge_size
    head_dim = config.embed_dim // config.num_heads
    rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
    for t, h, w in grid_thw:
        t = t.to(torch.int32).item()
        h = h.to(torch.int32).item()
        w = w.to(torch.int32).item()
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max().item()
    rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    rotary_pos_emb_cos = rotary_pos_emb.cos().unsqueeze(1)
    rotary_pos_emb_sin = rotary_pos_emb.sin().unsqueeze(1)
    return torch.cat([rotary_pos_emb_cos, rotary_pos_emb_sin], dim=1)


def precompute_qwen2_5vl_vision_attn_mask(seq_length, cu_seqlens, mask_value):
    """Precompute attention mask for Qwen2.5-VL vision model.

    Args:
        seq_length (int): The sequence length.
        cu_seqlens (torch.Tensor): The cumulative sequence lengths.
        mask_value (float): The mask value of vision attention mask.

    Returns:
        torch.Tensor: The precomputed attention mask.
    """
    attention_mask = torch.full([1, seq_length, seq_length], mask_value, dtype=torch.float32)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


def get_window_index(grid_thw, config):
    """Get the window index for a given grid and configuration.

    Args:
        grid_thw (list): A list of tuples representing the grid dimensions (time, height, width).
        config (object): The configuration object.

    Returns:
        tuple: A tuple containing the window index and cumulative window sequence lengths.
    """
    window_index = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = config.window_size // config.spatial_merge_size // config.patch_size
    spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h // config.spatial_merge_size,
            grid_w // config.spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
    window_index = torch.cat(window_index, dim=0)

    return window_index, cu_window_seqlens


class PatchMerger(nn.Module):
    """Patch Merger class for merging patches in vision models.

    Attributes:
        hidden_size (int): The hidden size for the MLP.
        ln_q (LayerNorm): The layer normalization layer.
        mlp (nn.Sequential): The MLP for merging patches.

    Methods:
        __init__(dim, context_dim, spatial_merge_size): Initialize the Patch Merger.
        forward(x): Forward pass for merging patches.
    """

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        """Initialize the Patch Merger.

        Args:
            dim (int): The dimension of the output.
            context_dim (int): The context dimension.
            spatial_merge_size (int, optional): The spatial merge size. Defaults to 2.
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for merging patches.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after merging patches.
        """
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


# For replace conv3d with conv2d
class Conv2dInplaceConv3d(torch.nn.Module):
    """Conv2dInplaceConv3d class."""

    def __init__(self, conv3d):
        """Only stride=1 and bias=False are supported."""
        super().__init__()
        inc, outc, ksize = conv3d.in_channels, conv3d.out_channels, conv3d.kernel_size
        self.conv2d1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, bias=False)
        self.conv2d2 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, bias=False)

        with torch.no_grad():
            self.conv2d1.weight.data = conv3d.weight.data[:, :, 0, :, :]
            self.conv2d2.weight.data = conv3d.weight.data[:, :, 1, :, :]
        self.conv2d1.to(conv3d.weight.data.device)
        self.conv2d2.to(conv3d.weight.data.device)

    def __getattr__(self, attr):
        """Attribute gettter."""
        conv2d1 = self._modules['conv2d1']
        conv2d2 = self._modules['conv2d2']
        if attr == 'conv2d1':
            return conv2d1
        if attr == 'conv2d2':
            return conv2d2
        # Note that since dtype is only obtained here in other places, conv2d1 can be returned.
        return getattr(conv2d1, attr)

    def forward(self, x: torch.Tensor):
        """Forward function."""
        return (self.conv2d1(x[:, 0::2, :, :]) + self.conv2d2(x[:, 1::2, :, :])).unsqueeze(2)


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


class PatchEmbed(nn.Module):
    """Patch Embedding class for converting images to patch embeddings in vision models.

    Attributes:
        patch_size (int): The size of each patch.
        temporal_patch_size (int): The temporal size of each patch.
        in_channels (int): The number of input channels.
        embed_dim (int): The embedding dimension.
        proj (nn.Conv3d): The 3D convolutional layer for projecting patches to embeddings.

    Methods:
        __init__(patch_size, temporal_patch_size, in_channels, embed_dim): Initialize the Patch Embed.
        forward(hidden_states): Forward pass to convert images to patch embeddings.
    """

    def __init__(
        self,
        config,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """Initialize the Patch Embed.

        Args:
            config (Config): The configuration object.
            patch_size (int, optional): The size of each patch. Defaults to 14.
            temporal_patch_size (int, optional): The temporal size of each patch. Defaults to 2.
            in_channels (int, optional): The number of input channels. Defaults to 3.
            embed_dim (int, optional): The embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        if self.config.use_conv2d_patch_embed:
            self.proj = Conv2dInplaceConv3d(
                nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
            )
        else:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass to convert images to patch embeddings.

        Args:
            hidden_states (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with patch embeddings.
        """
        target_dtype = self.proj.weight.dtype
        if self.config.use_conv2d_patch_embed:
            hidden_states = hidden_states.view(
                -1, self.in_channels * self.temporal_patch_size, self.patch_size, self.patch_size
            )
        else:
            hidden_states = hidden_states.view(
                -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen2_5_VLMLP(nn.Module):  # noqa: N801
    """Class for Qwen2_5_VLMLP.

    This class defines a multi-layer perceptron (MLP) for the Qwen2_5_VL model.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        intermediate_size (int): The size of the intermediate layer.
        gate_proj (nn.Linear): Linear layer for gating projection.
        up_proj (nn.Linear): Linear layer for up projection.
        down_proj (nn.Linear): Linear layer for down projection.
        act_fn (nn.Module): Activation function.

    Methods:
        forward: Forward pass through the MLP.
    """

    def __init__(self, config, bias: bool = False):
        """Initialize the Qwen2_5_VLMLP.

        Args:
            config (object): The configuration object.
            bias (bool, optional): Whether to use bias in the linear layers. Default is False.
        """
        super().__init__()
        self.hidden_size = config.embed_dim
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        """Forward pass through the MLP.

        Args:
            hidden_state (torch.Tensor): The input hidden state.

        Returns:
            torch.Tensor: The output of the MLP.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class VisionAttention(nn.Module):
    """Vision Attention class for applying attention mechanisms in vision models.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        qkv (nn.Linear): The linear layer for query, key, and value projections.
        proj (nn.Linear): The linear layer for the output projection.
        q_mul1 (mtk_quantization.pytorch.functional.Mul): The first multiplication function for query.
        q_mul2 (mtk_quantization.pytorch.functional.Mul): The second multiplication function for query.
        q_add (mtk_quantization.pytorch.functional.Add): The addition function for query.
        q_cat (mtk_quantization.pytorch.functional.Cat): The concatenation function for query.
        k_mul1 (mtk_quantization.pytorch.functional.Mul): The first multiplication function for key.
        k_mul2 (mtk_quantization.pytorch.functional.Mul): The second multiplication function for key.
        k_add (mtk_quantization.pytorch.functional.Add): The addition function for key.
        k_cat (mtk_quantization.pytorch.functional.Cat): The concatenation function for key.

    Methods:
        __init__(dim, num_heads): Initialize the Vision Attention.
        apply_rotary_pos_emb_mtk(q, k, cos, sin): Apply rotary positional embedding using MTK functions.
        forward(hidden_states, attention_mask, pos_emb): Forward pass for the attention mechanism.
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        """Initialize the Vision Attention.

        Args:
            dim (int): The dimension of the input.
            num_heads (int, optional): The number of attention heads. Defaults to 16.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

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
        # q1, q2 = torch.split(q, self.head_dim//2, dim=-1)
        q1 = q[..., : torch.div(q.shape[-1], 2, rounding_mode='floor')]
        q2 = q[..., torch.div(q.shape[-1], 2, rounding_mode='floor') :]
        q_rotated = self.q_cat((-q2, q1))
        # k1, k2 = torch.split(k, self.head_dim//2, dim=-1)
        k1 = k[..., : torch.div(k.shape[-1], 2, rounding_mode='floor')]
        k2 = k[..., torch.div(k.shape[-1], 2, rounding_mode='floor') :]
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (torch.Tensor): The attention mask.
            pos_emb (torch.Tensor, optional): The positional embedding. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying attention.
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        cos, sin = pos_emb
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


class Qwen2_5VLVisionBlock(nn.Module):  # noqa: N801
    """Qwen2.5-VL Vision Block class for applying vision blocks in the Qwen2.5-VL model.

    Attributes:
        norm1 (LayerNorm): The first layer normalization layer.
        norm2 (LayerNorm): The second layer normalization layer.
        attn (VisionAttention): The attention layer.
        mlp (VisionMlp): The MLP layer.

    Methods:
        __init__(config): Initialize the Qwen2.5-VL Vision Block.
        forward(hidden_states, attn_mask, rotary_pos_emb): Forward pass for the vision block.
    """

    def __init__(self, config) -> None:
        """Initialize the Qwen2.5-VL Vision Block.

        Args:
            config (object): The configuration object containing model parameters.
        """
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim, eps=1e-6)
        self.norm2 = RMSNorm(config.embed_dim, eps=1e-6)
        int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, attn_mask, rotary_pos_emb) -> torch.Tensor:
        """Forward pass for the vision block.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): The attention mask.
            rotary_pos_emb (torch.Tensor): The rotary positional embedding.

        Returns:
            torch.Tensor: The output tensor after applying the vision block.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attn_mask, rotary_pos_emb)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class CumulativeSeqenceLens(nn.Module):
    """Cumulative Sequence Lens class for computing cumulative sequence lengths.

    Methods:
        __init__(): Initialize the Cumulative Sequence Lens.
        forward(grid_thw): Forward pass to compute cumulative sequence lengths.
    """

    def __init__(self):
        """Initialize the Cumulative Sequence Lens."""
        super().__init__()

    def forward(self, grid_thw):
        """Forward pass to compute cumulative sequence lengths.

        Args:
            grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).

        Returns:
            torch.Tensor: The cumulative sequence lengths.
        """
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen2_5VLVisionModel(BaseVisionEncoderChunk):  # noqa: N801
    """Qwen2.5-VL Vision Model class for the Qwen2.5-VL vision model.

    Attributes:
        spatial_merge_size (int): The spatial merge size.
        patch_embed (PatchEmbed): The patch embedding layer.
        rotary_pos_emb (VisionRotaryEmbedding): The rotary positional embedding layer.
        blocks (nn.ModuleList): The list of vision blocks.
        merger (PatchMerger): The patch merger layer.
        jit_trace (bool): Whether to use JIT tracing.
        main_device (str): The main device for the model.
        cu_seqlens (CumulativeSeqenceLens): The cumulative sequence lens.

    Methods:
        __init__(config, jit_trace): Initialize the Qwen2.5-VL Vision Model.
        get_dtype(): Get the data type of the model.
        get_device(): Get the device of the model.
        rot_pos_emb(grid_thw): Compute rotary positional embeddings.
        forward(hidden_states, attn_mask, rotary_pos_emb): Forward pass for the vision model.
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
    ) -> None:
        """Initialize the Qwen2.5-VL Vision Model.

        Args:
            config (object): The configuration object containing model parameters.
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

        self.spatial_merge_size = config.spatial_merge_size

        if self.chunk_idx == 0:
            self.patch_embed = PatchEmbed(
                config,
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.embed_dim,
            )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.layers = nn.ModuleList([Qwen2_5VLVisionBlock(config) for _ in range(self.num_layers)])

        self.jit_trace = jit_trace
        self.main_device = 'cuda:0'
        # FIXME: This attribute needs to be taken care of during deployment,
        # One solution is to convert a separate graph to do the torch.repeat_interleave and cumsum operation
        # and offload this graph to CPU.
        self.cu_seqlens = CumulativeSeqenceLens()

        self._vision_attention_mask = 1
        self._vision_rot_emb = 1
        self._window_index = 1

        self.image_grid_thw = None

        self.spatial_merge_unit = self.config.spatial_merge_size * self.config.spatial_merge_size

    def get_dtype(self) -> torch.dtype:
        """Get the data type of the model.

        Returns:
            torch.dtype: The data type of the model.
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        """Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return self.blocks[0].mlp.fc2.weight.device

    @property
    def attention_mask(self):
        """Get Qwen2.5-VL ViT vision attention mask."""
        return self._vision_attention_mask

    @attention_mask.setter
    def attention_mask(self, attn_mask):
        logger.debug(f'Set Qwen2.5-VL ViT attention_mask to {attn_mask.shape}')
        self._vision_attention_mask = attn_mask

    @property
    def rot_emb(self):
        """Get Qwen2.5-VL ViT vision rotary embedding."""
        return self._vision_rot_emb

    @rot_emb.setter
    def rot_emb(self, r):
        logger.debug(f'Set Qwen2.5-VL ViT rot_emb to {r.shape}')
        self._vision_rot_emb = r

    @property
    def window_index(self):
        """Get Qwen2.5-VL vision window index."""
        return self._window_index

    @window_index.setter
    def window_index(self, w):
        logger.debug(f'Set Qwen2.5-VL ViT window_index to {w}')
        self._window_index = w

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {}
        if self.chunk_idx == 0:
            if self.config.use_conv2d_patch_embed:
                # Need to handle conv3d weight slicing later
                state_dict_mapping = {
                    'patch_embed_conv2d1_weight': {'patch_embed.proj.conv2d1.weight': 'visual.patch_embed.proj.weight'},
                    'patch_embed_conv2d2_weight': {'patch_embed.proj.conv2d2.weight': 'visual.patch_embed.proj.weight'},
                }
            else:
                state_dict_mapping = {
                    'patch_embed_weight': {'patch_embed.proj.weight': 'visual.patch_embed.proj.weight'}
                }
        # fmt: off
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}.attn.proj_bias': {
                        f'layers.{inner_layer_idx}.attn.proj.bias':
                        f'visual.blocks.{outer_layer_idx}.attn.proj.bias'
                    },
                    f'{outer_layer_idx}.attn.proj_weight': {
                        f'layers.{inner_layer_idx}.attn.proj.weight':
                        f'visual.blocks.{outer_layer_idx}.attn.proj.weight'
                    },
                    f'{outer_layer_idx}.attn.qkv_bias': {
                        f'layers.{inner_layer_idx}.attn.qkv.bias':
                        f'visual.blocks.{outer_layer_idx}.attn.qkv.bias'
                    },
                    f'{outer_layer_idx}.attn.qkv_weight': {
                        f'layers.{inner_layer_idx}.attn.qkv.weight':
                        f'visual.blocks.{outer_layer_idx}.attn.qkv.weight'
                    },
                    f'{outer_layer_idx}.mlp.g_bias': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.bias':
                        f'visual.blocks.{outer_layer_idx}.mlp.gate_proj.bias'
                    },
                    f'{outer_layer_idx}.mlp.g_weight': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.weight':
                        f'visual.blocks.{outer_layer_idx}.mlp.gate_proj.weight'
                    },
                    f'{outer_layer_idx}.mlp.u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias':
                        f'visual.blocks.{outer_layer_idx}.mlp.up_proj.bias'
                    },
                    f'{outer_layer_idx}.mlp.u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight':
                        f'visual.blocks.{outer_layer_idx}.mlp.up_proj.weight'
                    },
                    f'{outer_layer_idx}.mlp.d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias':
                        f'visual.blocks.{outer_layer_idx}.mlp.down_proj.bias'
                    },
                    f'{outer_layer_idx}.mlp.d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight':
                        f'visual.blocks.{outer_layer_idx}.mlp.down_proj.weight'
                    },
                    f'{outer_layer_idx}.norm1_weight': {
                        f'layers.{inner_layer_idx}.norm1.weight':
                        f'visual.blocks.{outer_layer_idx}.norm1.weight'
                    },
                    f'{outer_layer_idx}.norm2_weight': {
                        f'layers.{inner_layer_idx}.norm2.weight':
                        f'visual.blocks.{outer_layer_idx}.norm2.weight'
                    },
                }
            )
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())
        # fmt: on

        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter Qwen2_5VLVisionModel load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

        # state_dict = self._pop_text_model_weights(state_dict)

        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        # Handle use_conv2d_patch_embed
        if self.chunk_idx == 0 and self.config.use_conv2d_patch_embed:
            conv3dweight = state_dict.pop('visual.patch_embed.proj.weight')
            conv2d1 = conv3dweight[:, :, 0, :, :].contiguous()
            conv2d2 = conv3dweight[:, :, 1, :, :].contiguous()
            weights_to_load.update(
                {
                    'patch_embed.proj.conv2d1.weight': conv2d1.to(torch.float32),
                    'patch_embed.proj.conv2d2.weight': conv2d2.to(torch.float32),
                }
            )
            self.state_dict_mapping.pop('patch_embed_conv2d1_weight')
            self.state_dict_mapping.pop('patch_embed_conv2d2_weight')

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
            self.patch_embed.to(self.device_list[0])
            # self.pre_layernorm.to(self.device_list[0])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict

    def rot_pos_emb(self, grid_thw):
        """Compute rotary positional embeddings.

        Args:
            grid_thw (torch.Tensor): The grid tensor with shape (T, H, W).

        Returns:
            torch.Tensor: The computed rotary positional embeddings.
        """
        pos_ids = []
        for t, h, w in grid_thw:
            t = t.to(torch.int32).item()
            h = h.to(torch.int32).item()
            w = w.to(torch.int32).item()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for the vision model.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): The attention mask.
            rotary_pos_emb (torch.Tensor): The rotary positional embedding.
            kwargs: Other kwargs

        Returns:
            torch.Tensor: The output tensor after applying the vision model.
        """
        attn_mask = kwargs.get('qwen2_5_vl_attn_mask')
        rotary_pos_emb = kwargs.get('qwen2_5_vl_rotary_pos_emb')
        window_index = kwargs.get('qwen2_5_vl_window_index')
        if attn_mask is None:
            attn_mask = self._vision_attention_mask
            logger.debug('Vision attention mask is not passed as argument, using the pre-set attribute instead.')
        if rotary_pos_emb is None:
            rotary_pos_emb = self._vision_rot_emb
            logger.debug('Vision rotary embedding is not passed as argument, using the pre-set attribute instead.')
        if window_index is None:
            window_index = self._window_index
            logger.debug('Vision window index is not passed as argument, using the pre-set attribute instead.')
        if attn_mask is None:
            logger.error(
                'qwen2_5_vl_attn_mask must be either passed or pre-set when forwarding dynamic shape Qwen2.5-VL ViT.',
                err=ValueError,
            )
        if rotary_pos_emb is None:
            logger.error(
                'qwen2_5_vl_rotary_pos_emb must be passed or pre-set when forwarding dynamic shape Qwen2.5-VL ViT.',
                err=ValueError,
            )
        if window_index is None:
            logger.error(
                'qwen2_5_vl_window_index must be passed or pre-set when forwarding dynamic shape Qwen2.5-VL ViT.',
                err=ValueError,
            )
        hidden_states = hidden_states.to(self.device_list[0])
        window_index = window_index.to(self.device_list[0])

        seq_len, _ = hidden_states.size()
        logger.debug(f'hidden_states: {hidden_states.shape}')
        logger.debug(f'window_index: {window_index}')
        logger.debug(f'window_index: {window_index.shape}')

        if self.first_layer_idx == 0:
            if self.config.exclude_first_gather and self.image_grid_thw is not None:  # PTQ
                hidden_states_select = hidden_states
            else:
                hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                # hidden_states = hidden_states[window_index, :, :]
                # GATHER OP is not supported, so we need to rewrite it into other equivalent implementation

                # GATHER OP implementation
                hidden_states_select_gather = torch.index_select(hidden_states, 0, window_index)

                # SLICE OP implementation
                hidden_states_select = []
                for w_i in window_index:
                    hidden_states_select.append(hidden_states[w_i, :, :].unsqueeze(0))
                hidden_states_select = torch.cat(hidden_states_select, dim=0)

                assert torch.all(hidden_states_select_gather == hidden_states_select)

                hidden_states_select = hidden_states_select.reshape(seq_len, -1)
            hidden_states = self.patch_embed(hidden_states_select)
        logger.debug(f'hidden_states: {hidden_states.shape}')
        attn_mask = attn_mask.to(device=self.device_list[0])
        rotary_pos_emb = rotary_pos_emb.to(self.device_list[0])
        hidden_states = hidden_states.to(self.device_list[0])
        cos, sin = torch.split(rotary_pos_emb, 1, dim=1)

        cos = cos.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        cos_select_gather = cos[window_index, :, :]
        cos_select = torch.index_select(cos, 0, window_index)
        assert torch.all(cos_select_gather == cos_select)
        cos = cos_select.reshape(seq_len, -1).unsqueeze(1)

        sin = sin.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        sin_select_gather = sin[window_index, :, :]
        sin_select = torch.index_select(sin, 0, window_index)
        assert torch.all(sin_select_gather == sin_select)
        sin = sin_select.reshape(seq_len, -1).unsqueeze(1)

        rotary_pos_emb = (cos, sin)
        for idx, blk in enumerate(self.layers):
            hidden_states = blk(
                hidden_states.to(self.device_list[idx]),
                attn_mask=attn_mask.to(self.device_list[idx]),
                rotary_pos_emb=rotary_pos_emb,
            )

        return hidden_states

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        self._calculate_ptq_fixed_shape_batch()
        self._calulate_ptq_attnmask_rotemb()
        fixed_batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        if self.chunk_idx == 0:
            temp_input = torch.randn(
                fixed_batch_size,
                self.config.patch_size
                * self.config.patch_size
                * self.config.in_channels
                * self.config.temporal_patch_size,
                device='cpu',
                dtype=torch.float32,
            )
            if self.config.exclude_first_gather:
                temp_input = temp_input.reshape(
                    fixed_batch_size // self.spatial_merge_unit, self.spatial_merge_unit, -1
                )
                temp_input = temp_input[self._window_index, :, :]
                temp_input = temp_input.reshape(fixed_batch_size, -1)
            self.chunk0_input_shape = temp_input.shape
            return temp_input
        feature_size = int(self.config.embed_dim)
        return torch.randn(fixed_batch_size, feature_size, device='cpu', dtype=torch.float32)

    def _calculate_ptq_fixed_shape_batch(self):
        from ..preprocessors.configuration_qwen2vl_vision import Qwen2VLPreprocessorConfig
        from ..preprocessors.preprocessor_qwen2vl_vision import Qwen2VLImageProcessor

        logger.debug('Enter Qwen2.5-VL _calculate_ptq_fixed_shape_batch.')
        preprocessor_config = Qwen2VLPreprocessorConfig(**self.config.preprocessor_config)
        processor = Qwen2VLImageProcessor(**preprocessor_config.get())
        image_hw = self.config.image_resolution
        if image_hw is None:
            logger.error('image_resolution in config must be set when PTQing Qwen2.5-VL ViT.', err=ValueError)
        image = np.random.rand(image_hw[0], image_hw[1], 3)
        image = Image.fromarray(image.astype('uint8'))
        return_dict, _ = processor.preprocess([image])
        self.image_grid_thw = torch.tensor(return_dict['image_grid_thw'])
        logger.debug(f'Qwen2.5-VL PTQ image_grid_thw: {self.image_grid_thw}')

    def _calulate_ptq_attnmask_rotemb(self):
        from ..hooks.qwen2_vl_pre_encoder import CumulativeSeqenceLens

        logger.debug('Enter Qwen2.5-VL _calulate_ptq_attnmask_rotemb.')
        cuseqlen = CumulativeSeqenceLens()
        cu_seqlens = cuseqlen(self.image_grid_thw)
        batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        # Calculate window attention cu_seqlens
        window_index, cu_window_seqlens = get_window_index(self.image_grid_thw, self.config)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=self.image_grid_thw.device, dtype=self.image_grid_thw.dtype
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        vision_attn_mask_window = precompute_qwen2_5vl_vision_attn_mask(
            seq_length=batch_size, cu_seqlens=cu_window_seqlens, mask_value=self.config.mask_value
        )
        vision_attn_mask_normal = precompute_qwen2_5vl_vision_attn_mask(
            seq_length=batch_size, cu_seqlens=cu_seqlens, mask_value=self.config.mask_value
        )

        if self.chunk_idx in self.config.fullatt_block_indexes:
            logger.debug(f'Set normal attention mask at layer {self.chunk_idx}.')
            self._vision_attention_mask = vision_attn_mask_normal
        else:
            logger.debug(f'Set window attention mask at layer {self.chunk_idx}.')
            self._vision_attention_mask = vision_attn_mask_window
        self._vision_rot_emb = precompute_qwen2_5vl_vision_rot_emb(self.image_grid_thw, self.config)
        self._window_index = window_index

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
        fixed_batch_size = (self.image_grid_thw[0][1] * self.image_grid_thw[0][2]).item()
        flatten_patch_size = (
            self.config.patch_size * self.config.patch_size * self.config.in_channels * self.config.temporal_patch_size
        )
        if self.chunk_idx == 0:
            input_shapes = [self.chunk0_input_shape]
        else:
            feature_size = int(self.config.embed_dim)
            input_shapes = [[fixed_batch_size, feature_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        yield [np.random.rand(fixed_batch_size, flatten_patch_size).astype(np.float32)]
                    else:
                        yield [np.random.rand(fixed_batch_size, feature_size).astype(np.float32)]
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
                        yield [np.random.rand(fixed_batch_size, flatten_patch_size).astype(np.float32)]
                    else:
                        yield [np.random.rand(fixed_batch_size, feature_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
