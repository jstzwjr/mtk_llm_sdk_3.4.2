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
"""Define Whisper Encoder model class."""

import json
import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger, qat_utils, utils
from ..activations import TorchGelu
from ..modeling_base import BaseEncoderModelChunk
from ..norm import LayerNorm
from .configuration_whisper import WhisperEncoderConfig


class WhisperMLP(nn.Module):
    """Whisper Multi-Layer Perceptron (MLP) class.

    This class implements the Whisper MLP component with optional LoRA (Low-Rank Adaptation) support.

    Attributes:
        config (WhisperEncoderConfig): Configuration for the WhisperMLP.
        lora (LoRA): LoRA object.
        layer_idx (int): Index of the layer.
        jit_trace (bool): Whether to use JIT tracing.
        hidden_size (int): Size of the hidden layer.
        intermediate_size (int): Size of the intermediate layer.
        parallel_lora (bool): Whether to use parallel LoRA.
        up_proj (nn.Linear): Linear layer for up projection.
        down_proj (nn.Linear): Linear layer for down projection.
        mul (mtk_quantization.pytorch.functional.Mul): Multiplication function from mtk_quantization.
        mul2 (mtk_quantization.pytorch.functional.Mul): Second multiplication function from mtk_quantization.
        with_lora (bool): Whether to use LoRA.
    """

    def __init__(self, config: WhisperEncoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initializes the MLP class.

        Args:
            config (BaseLLMConfig): Configuration for the MLP.
            lora (LoRA): LoRA object.
            layer_idx (int): Index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
        """
        super().__init__()
        self.config = config
        self.lora = lora
        self.jit_trace = jit_trace
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx
        self.parallel_lora = parallel_lora

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.gelu = TorchGelu()

        # Lora modules
        self.with_lora = False
        if self.lora is not None:
            self.with_lora = self.layer_idx >= self.lora.start_idx and self.layer_idx <= self.lora.end_idx
            if self.with_lora:
                if self.parallel_lora:
                    for module in self.lora.target_modules:
                        if module == self.config.fc_names['mlp']['up']:
                            self.up_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.up_lora_B = nn.Linear(self.lora.rank, self.intermediate_size, bias=False).apply(
                                qat_utils.init_lora_B
                            )
                            self.up_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.up_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.up_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.up_lora_dropout = nn.Identity()
                        elif module == self.config.fc_names['mlp']['down']:
                            self.down_lora_A = nn.Linear(self.intermediate_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.down_lora_B = nn.Linear(self.lora.rank, self.hidden_size, bias=False).apply(
                                qat_utils.init_lora_B
                            )
                            self.down_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.down_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.down_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.down_lora_dropout = nn.Identity()
                else:
                    for module in self.lora.target_modules:
                        if module == self.config.fc_names['mlp']['up']:
                            self.up_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.up_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.up_lora_add = mtk_quantization.pytorch.functional.Add()
                        elif module == self.config.fc_names['mlp']['down']:
                            self.down_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.down_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.down_lora_add = mtk_quantization.pytorch.functional.Add()

    def forward(self, x, *lora_inputs):
        """Forward pass for the MLP.

        Args:
            x (torch.Tensor): The input tensor.
            *lora_inputs: Additional inputs for LoRA.

        Returns:
            torch.Tensor: The output tensor after applying the MLP and optional LoRA.
        """
        lora_idx = 0

        if self.lora is not None and self.config.fc_names[None]['up'] in self.lora.target_modules and self.with_lora:
            up2 = self.fc1(x)
            if self.parallel_lora:
                up_lora = self.up_lora_B(self.up_lora_A(self.up_lora_dropout(x)))
                if self.lora.scale != 1.0:
                    up_lora = self.up_lora_scale_mul(up_lora, self.lora.scale)
            else:
                up_lora = self.up_lora_B(
                    self.up_lora_A(x, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            up = self.up_lora_add(up2, up_lora)
        else:
            up = self.up_proj(x)

        pre_down = self.gelu(up)
        if self.lora is not None and self.config.fc_names[None]['down'] in self.lora.target_modules and self.with_lora:
            down2 = self.fc2(pre_down)
            if self.parallel_lora:
                down_lora = self.down_lora_B(self.down_lora_A(self.down_lora_dropout(pre_down)))
                if self.lora.scale != 1.0:
                    down_lora = self.down_lora_scale_mul(down_lora, self.lora.scale)
            else:
                down_lora = self.down_lora_B(
                    self.down_lora_A(pre_down, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            down = self.down_lora_add(down2, down_lora)
        else:
            down = self.down_proj(pre_down)

        return down


class WhisperEncoderAttention(nn.Module):
    """Whisper Encoder Attention class.

    This class implements the attention mechanism for the Whisper Encoder model.

    Attributes:
        config: The configuration for the Whisper Encoder model.
        hidden_size (int): The hidden size of the attention mechanism.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        scale (float): The scaling factor for the query projection.
        k_proj (nn.Linear): The linear layer for the key projection.
        v_proj (nn.Linear): The linear layer for the value projection.
        q_proj (nn.Linear): The linear layer for the query projection.
        o_proj (nn.Linear): The linear layer for the output projection.
        add (mtk_quantization.pytorch.functional.Add): Addition function from mtk_quantization.
        matmul1 (mtk_quantization.pytorch.functional.Matmul): Addition function from mtk_quantization.
        matmul2 (mtk_quantization.pytorch.functional.Matmul): Addition function from mtk_quantization.

    Methods:
        __init__: Initializes the WhisperEncoderAttention class.
        forward: Performs the forward pass to compute the attention output.
    """

    def __init__(self, config: WhisperEncoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initializes the WhisperEncoderAttention class.

        Args:
            config (WhisperEncoderConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.

        Raises:
            ValueError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.config = config
        self.lora = lora
        self.jit_trace = jit_trace
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        if self.head_dim * self.num_heads != self.hidden_size:
            logger.error(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:'
                f' {self.num_heads}).',
                err=ValueError,
            )
        self.layer_idx = layer_idx
        self.parallel_lora = parallel_lora
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.add = mtk_quantization.pytorch.functional.Add()
        self.matmul1 = mtk_quantization.pytorch.functional.Matmul()
        self.matmul2 = mtk_quantization.pytorch.functional.Matmul()

        # Lora modules
        self.with_lora = False
        if self.lora is not None:
            self.with_lora = self.layer_idx >= self.lora.start_idx and self.layer_idx <= self.lora.end_idx
            if self.with_lora:
                if self.parallel_lora:
                    for module in self.lora.target_modules:
                        if module == self.config.fc_names['attn']['q']:
                            self.q_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.q_lora_B = nn.Linear(self.lora.rank, self.hidden_size, bias=False).apply(
                                qat_utils.init_lora_B
                            )
                            self.q_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.q_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.q_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.q_lora_dropout = nn.Identity()
                        elif module == self.config.fc_names['attn']['k']:
                            self.k_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.k_lora_B = nn.Linear(
                                self.lora.rank, self.num_key_value_heads * self.head_dim, bias=False
                            ).apply(qat_utils.init_lora_B)
                            self.k_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.k_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.k_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.k_lora_dropout = nn.Identity()
                        elif module == self.config.fc_names['attn']['v']:
                            self.v_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.v_lora_B = nn.Linear(
                                self.lora.rank, self.num_key_value_heads * self.head_dim, bias=False
                            ).apply(qat_utils.init_lora_B)
                            self.v_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.v_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.v_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.v_lora_dropout = nn.Identity()
                        elif module == self.config.fc_names['attn']['o']:
                            self.o_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.o_lora_B = nn.Linear(self.lora.rank, self.hidden_size, bias=False).apply(
                                qat_utils.init_lora_B
                            )
                            self.o_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.o_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.o_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.o_lora_dropout = nn.Identity()
                else:
                    for module in self.lora.target_modules:
                        if module == self.config.fc_names['attn']['q']:
                            self.q_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.q_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.q_lora_add = mtk_quantization.pytorch.functional.Add()
                        elif module == self.config.fc_names['attn']['k']:
                            self.k_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.k_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.k_lora_add = mtk_quantization.pytorch.functional.Add()
                        elif module == self.config.fc_names['attn']['v']:
                            self.v_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.v_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.v_lora_add = mtk_quantization.pytorch.functional.Add()
                        elif module == self.config.fc_names['attn']['o']:
                            self.o_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.o_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.o_lora_add = mtk_quantization.pytorch.functional.Add()

    def forward(self, hidden_states, *lora_inputs):
        """Performs the forward pass to compute the attention output.

        Input shape: Batch x Max Source Position x Hidden Size.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            lora_inputs (tuple): Input lora tuple.

        Returns:
            torch.Tensor: The attention output.

        Raises:
            ValueError: If the size of the attention mask or causal attention mask is incorrect.
        """
        bsz, q_len, hidden_size = hidden_states.size()
        lora_idx = 0
        if self.lora is not None and self.config.fc_names['attn']['q'] in self.lora.target_modules and self.with_lora:
            query_states = self.q_proj(hidden_states) * self.scale
            if self.parallel_lora:
                q_lora = self.q_lora_B(self.q_lora_A(self.q_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    q_lora = self.q_lora_scale_mul(q_lora, self.lora.scale)
            else:
                q_lora = self.q_lora_B(
                    self.q_lora_A(hidden_states, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            query_states = (
                self.q_lora_add(query_states, q_lora).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            )
        else:
            query_states = (
                (self.q_proj(hidden_states) * self.scale)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(self.num_heads, q_len, self.head_dim)
            )

        if self.lora is not None and self.config.fc_names['attn']['k'] in self.lora.target_modules and self.with_lora:
            key_states = self.k_proj(hidden_states)
            if self.parallel_lora:
                k_lora = self.k_lora_B(self.k_lora_A(self.k_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    k_lora = self.k_lora_scale_mul(k_lora, self.lora.scale)
            else:
                k_lora = self.k_lora_B(
                    self.k_lora_A(hidden_states, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            key_states = (
                self.k_lora_add(key_states, k_lora)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            key_states = (
                self.k_proj(hidden_states)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(self.num_heads, q_len, self.head_dim)
            )

        if self.lora is not None and self.config.fc_names['attn']['v'] in self.lora.target_modules and self.with_lora:
            value_states = self.v_proj(hidden_states)
            if self.parallel_lora:
                v_lora = self.v_lora_B(self.v_lora_A(self.v_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    v_lora = self.v_lora_scale_mul(v_lora, self.lora.scale)
            else:
                v_lora = self.v_lora_B(
                    self.v_lora_A(hidden_states, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            value_states = (
                self.v_lora_add(value_states, v_lora)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            value_states = (
                self.v_proj(hidden_states)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(self.num_heads, q_len, self.head_dim)
            )

        attn_weights = self.matmul1(query_states, key_states.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = self.matmul2(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        if self.lora is not None and self.config.fc_names['attn']['o'] in self.lora.target_modules and self.with_lora:
            attn_output2 = self.o_proj(attn_output)
            if self.parallel_lora:
                o_lora = self.o_lora_B(self.o_lora_A(self.o_lora_dropout(attn_output)))
                if self.lora.scale != 1.0:
                    o_lora = self.o_lora_scale_mul(o_lora, self.lora.scale)
            else:
                o_lora = self.o_lora_B(
                    self.o_lora_A(attn_output, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output


class WhisperEncoderLayer(nn.Module):
    """Whisper Encoder Layer class.

    This class implements a single encoder layer for the Whisper model.

    Attributes:
        embed_dim (int): The hidden_size dimension.
        self_attn (WhisperEncoderAttention): The self-attention mechanism.
        intermediate_size (int): The intermediate_size dimension.
        gelu (nn.Module): The activation function used.
        fc1 (nn.Linear): The linear layer for the fc1 projection.
        fc2 (nn.Linear): The linear layer for the fc2 projection.
        input_norm (nn.LayerNorm): The input normalization.
        post_attention_norm (nn.LayerNorm): The post attention normalization.
        add1 (mtk_quantization.pytorch.functional.Add): Addition function from mtk_quantization.
        add2 (mtk_quantization.pytorch.functional.Add): Addition function from mtk_quantization.

    Methods:
        __init__: Initializes the WhisperEncoderLayer class.
        forward: Performs the forward pass to compute the encoder layer output.
    """

    def __init__(
        self,
        config: WhisperEncoderConfig,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=WhisperEncoderAttention,
        mlp_class=WhisperMLP,
        norm_class=LayerNorm,
        parallel_lora=False,
    ):
        """Initializes the WhisperEncoderLayer class.

        Args:
            config (WhisperEncoderConfig): The configuration for the Whisper Encoder model.
            lora (LoRA): LoRA object.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            layer_idx (int, optional): Index of the layer. Default is None.
            attn_class (type, optional): Class for the attention mechanism. Default is Attention.
            mlp_class (type, optional): Class for the MLP mechanism.
            norm_class (type, optional): Class for the normalization. Default is LayerNorm.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            exclude_input_norm (bool, optional): Whether to exclude input normalization. Default is False.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.jit_trace = jit_trace
        self.self_attn = attn_class(config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora)
        self.mlp = mlp_class(config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora)
        self.input_norm = norm_class(self.hidden_size, eps=config.norm_eps)
        self.post_attention_norm = norm_class(self.hidden_size, eps=config.norm_eps)
        self.add = mtk_quantization.pytorch.functional.Add()
        self.add2 = mtk_quantization.pytorch.functional.Add()

        self.expected_attn_lora_inputs = 0
        self.expected_mlp_lora_inputs = 0
        if not parallel_lora and lora is not None:
            for module in lora.target_modules:
                if module in [v for k, v in config.fc_names['attn'].items() if k != 'name']:
                    if module == config.fc_names['attn']['qkv']:
                        logger.error(
                            f'combined QKV FC not supported any longer. Please use '
                            f'{config.fc_names["attn"]["q"]}, '
                            f'{config.fc_names["attn"]["k"]}, '
                            f'{config.fc_names["attn"]["v"]}',
                            err=NotImplementedError,
                        )
                    self.expected_attn_lora_inputs += 2
                elif module in [v for k, v in config.fc_names['mlp'].items() if k != 'name']:
                    self.expected_mlp_lora_inputs += 2

    def forward(self, hidden_states, *lora_inputs):
        """Performs the forward pass to compute the encoder layer output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            lora_inputs (tuple): Tuple of lora inputs.

        Returns:
            torch.Tensor: The encoder layer output.
        """
        residual = hidden_states

        hidden_states = self.input_norm(hidden_states)
        attn_outputs = self.self_attn(hidden_states, *lora_inputs[: self.expected_attn_lora_inputs])
        hidden_states = self.add(residual, attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, *lora_inputs[-self.expected_mlp_lora_inputs :])

        return self.add2(residual, hidden_states)


class WhisperEncoderKVLayer(nn.Module):
    """Whisper Encoder KV Layer class.

    This class implements a single KV layer for the Whisper cross encoder model.

    Attributes:
        config: The configuration for the Whisper Encoder model.
        layer_idx (int): The layer index of the attention mechanism.
        hidden_size (int): The hidden size of the attention mechanism.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        k_proj (nn.Linear): The linear layer for the key projection.
        v_proj (nn.Linear): The linear layer for the value projection.

    Methods:
        __init__: Initializes the WhisperEncoderKVLayer class.
        forward: Performs the forward pass to compute the cross attention KV encoder layer output.
    """

    def __init__(self, config: WhisperEncoderConfig, layer_idx, jit_trace=False):
        """Initializes the WhisperEncoderKVLayer class.

        Args:
            config (WhisperEncoderConfig): The configuration for the Whisper Encoder model.
            layer_idx (int): index of kv cross attention layer
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states):
        """Performs the forward pass to compute the KV cross attention layer output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.

        Returns:
            torch.Tensor: The k cross attention layer output.
            torch.Tensor: The v cross attention layer output.
        """
        bsz, q_len, _ = hidden_states.size()
        k_out = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_out = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        return k_out, v_out


class WhisperEncoderChunk(BaseEncoderModelChunk):
    """Model chunk class for the model.

    This class implements a chunk of the model, which includes multiple encoder layers and optional tail components.

    Attributes:
    config (object): The configuration object.
        layers (nn.ModuleList): List of encoder layers.
        first_layer_idx (int): The chunk index.
        conv1 (nn.Conv1d): Conv layer for Whisper encoder.
        conv2 (nn.Conv1d): Conv layer for Whisper encoder.
        gelu1 (nn.Module): Activation for Whisper encoder.
        gelu2 (nn.Module): Activation for Whisper encoder.
        embed_positions (nn.Module): position embeddings.
        norm (nn.Module): Normalization layer.
    """

    def __init__(
        self,
        config: WhisperEncoderConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        encoder_class=WhisperEncoderLayer,
        norm_class=LayerNorm,
        parallel_lora=False,
        distribute_layers=True,
        **kwargs,
    ):
        """Initializes the WhisperEncoderChunk class.

        Args:
            config (BaseConfig): Configuration for the model chunk.
            lora (LoRA): LoRA object.
            num_layers (int): Number of encoder layers in the chunk.
            first_layer_idx (int): The index of the first encoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            encoder_class (type, optional): Class for the encoder layer. Default is WhisperEncoderLayer.
            norm_class (type, optional): Class for the normalization. Default is None.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            distribute_layers (bool, optional): Whether to distribute layers across devices. Default is True.
            kwargs: Additional keyword arguments.
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
            self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)
            self.gelu1 = TorchGelu()
            self.gelu2 = TorchGelu()
            self.embed_positions = nn.Embedding(config.max_source_positions, config.hidden_size)
        self.num_layers = num_layers
        self.first_layer_idx = first_layer_idx
        self.chunk_idx = chunk_idx
        self.layers = nn.ModuleList(
            [
                encoder_class(
                    config,
                    lora,
                    jit_trace=jit_trace,
                    layer_idx=(chunk_idx * num_layers) + i,
                    norm_class=norm_class,
                    parallel_lora=parallel_lora,
                )
                for i in range(num_layers)
            ]
        )
        if self.first_layer_idx + self.num_layers == self.config.num_hidden_layers:
            self.norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
            self.encoder_tails_kv = nn.ModuleList(
                [
                    WhisperEncoderKVLayer(
                        config,
                        layer_idx=i,
                    )
                    for i in range(config.decoder_num_layers)
                ]
            )

        self.expected_num_pos_emb_inputs = 0
        if not self.parallel_lora and self.with_lora:
            self.expected_num_lora_inputs_per_layer = 0 if self.lora is None else len(self.lora.target_modules) * 2
            self.expected_num_lora_inputs = self.expected_num_lora_inputs_per_layer * num_layers
        else:
            self.expected_num_lora_inputs_per_layer = 0
            self.expected_num_lora_inputs = 0

        self.expected_num_inputs = 1 + self.expected_num_pos_emb_inputs + self.expected_num_lora_inputs

        if self.support_quant_stub:
            self.stubs = nn.ModuleList(
                [mtk_quantization.pytorch.functional.QuantizerStub() for _ in range(self.expected_num_inputs)]
            )

    def forward(self, *inputs):
        """Forward pass for the model chunk.

        Args:
            inputs (list): List of all input tensors.

        Returns:
            tuple: Tuple containing output tensors.
        """
        if len(inputs) != self.expected_num_inputs:
            logger.error(f'Expected {self.expected_num_inputs} inputs but got {len(inputs)}')
        inputs = list(inputs)
        for i in range(self.expected_num_inputs):
            if self.support_quant_stub:
                inputs[i] = self.stubs[i](inputs[i])
            if self.distribute_layers:
                inputs[i] = inputs[i].to(self.device_list[0])

        inputs_embeds = inputs[0]
        lora_inputs = inputs[1:] if self.expected_num_lora_inputs > 0 else []

        if self.chunk_idx == 0:
            inputs_embeds = self.gelu1(self.conv1(inputs_embeds))
            inputs_embeds = self.gelu2(self.conv2(inputs_embeds)).permute(0, 2, 1)
            embed_pos = self.embed_positions.weight
            inputs_embeds = inputs_embeds + embed_pos.reshape((1, embed_pos.shape[0], embed_pos.shape[1]))

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_outputs = encoder_layer(
                hidden_states.to(self.device_list[idx]),
                *[
                    x.to(self.device_list[idx])
                    for x in lora_inputs[
                        idx * self.expected_num_lora_inputs_per_layer : (idx + 1)
                        * self.expected_num_lora_inputs_per_layer
                    ]
                ],
            )
            hidden_states = encoder_outputs

        if self.first_layer_idx + self.num_layers == self.config.num_hidden_layers:
            cross_cache = []
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)
            for _, encoder_tails_kv_layer in enumerate(self.encoder_tails_kv):
                if self.distribute_layers:
                    encoder_outputs = encoder_tails_kv_layer(hidden_states.to(self.device_list[idx]))
                else:
                    encoder_outputs = encoder_layer(hidden_states)
                cross_cache.append(encoder_outputs[0].to(inputs_embeds.device))
                cross_cache.append(encoder_outputs[1].to(inputs_embeds.device))

            return hidden_states, (cross_cache)

        return hidden_states

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        if self.chunk_idx == 0:
            example_inputs = torch.randn(
                1, self.config.num_mel_bins, self.config.max_source_positions * 2, device='cpu', dtype=torch.float32
            )
        else:
            example_inputs = torch.randn(
                1, self.config.max_source_positions, self.config.hidden_size, device='cpu', dtype=torch.float32
            )
        if self.config.sparse_attn and self.chunk_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        lora_inputs = []
        if self.lora is not None and not self.parallel_lora:
            for i in range(self.num_layers):
                if not self.with_lora[i]:
                    continue
                for module in self.lora.target_modules:
                    if module == self.config.fc_names['attn']['q']:
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.lora_rank, self.config.hidden_size, device='cpu', dtype=torch.float32
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.hidden_size, self.config.lora_rank, device='cpu', dtype=torch.float32
                            )
                        )
                    elif module == self.config.fc_names['attn']['k'] or module == self.config.fc_names['attn']['v']:
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.lora_rank, self.config.hidden_size, device='cpu', dtype=torch.float32
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1, num_head * self.head_dim, self.config.lora_rank, device='cpu', dtype=torch.float32
                            )
                        )
                    elif module == self.config.fc_names['attn']['o']:
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.lora_rank, self.config.hidden_size, device='cpu', dtype=torch.float32
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.hidden_size, self.config.lora_rank, device='cpu', dtype=torch.float32
                            )
                        )
                    elif module == module == self.config.fc_names['mlp']['up']:
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.lora_rank, self.config.hidden_size, device='cpu', dtype=torch.float32
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.config.intermediate_size,
                                self.config.lora_rank,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                    elif module == self.config.fc_names['mlp']['down']:
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.config.lora_rank,
                                self.config.intermediate_size,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.hidden_size, self.config.lora_rank, device='cpu', dtype=torch.float32
                            )
                        )

        return (example_inputs, *lora_inputs)

    def get_ptq_inputs(self, args, exp_name, lora_inputs=None, calib_lora_map=None, eval_lora_map=None):
        """Gets inputs for post-training quantization (PTQ).

        This method generates inputs for PTQ, including LoRA inputs if applicable.

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            lora_inputs (list, optional): List of number of lora scenarios of list of LoRA inputs per scenario.
                Default is None.
            calib_lora_map (list, optional): List of LoRA scenario mappings for calibration dataset. Default is None.
            eval_lora_map (list, optional): List of LoRA scenario mappings for evaluation dataset. Default is None.

        Returns:
            Tuple containing input shapes, input value ranges, calibration data generator,
                and evaluation data generator.
        """
        if calib_lora_map is None:
            if args.calibration_dataset is None:
                calib_ds_len = 0
            elif args.calibration_dataset == 'fake':
                calib_ds_len = 10
            else:
                calib_ds_len = len(
                    os.listdir(os.path.join(args.calibration_dataset, 'encoder', f'chunk_{self.chunk_idx}'))
                )
            calib_lora_map = [0 for _ in range(calib_ds_len)]
        if eval_lora_map is None:
            if args.evaluation_dataset is None:
                eval_ds_len = 0
            elif args.evaluation_dataset == 'fake':
                eval_ds_len = 10
            else:
                eval_ds_len = len(
                    os.listdir(os.path.join(args.evaluation_dataset, 'encoder', f'chunk_{self.chunk_idx}'))
                )
            eval_lora_map = [0 for _ in range(eval_ds_len)]
        if lora_inputs is None:
            lora_inputs = [[]]

        if self.config.sparse_attn and self.first_layer_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads
        lora_shapes = []
        if self.lora is not None and not self.parallel_lora:
            for i in range(self.num_layers):
                if not self.with_lora[i]:
                    continue
                for module in self.lora.target_modules:
                    if module == self.config.fc_names['attn']['q']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, self.config.hidden_size, None])
                    elif module == self.config.fc_names['attn']['k'] or module == self.config.fc_names['attn']['v']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, num_head * self.head_dim, None])
                    elif module == self.config.fc_names['attn']['o']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, self.config.hidden_size, None])
                    elif module == self.config.fc_names['mlp']['gate'] or module == self.config.fc_names['mlp']['up']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, self.config.intermediate_size, None])
                    elif module == self.config.fc_names['mlp']['down']:
                        lora_shapes.append([None, None, self.config.intermediate_size])
                        lora_shapes.append([None, self.config.hidden_size, None])

        minmax = (-1.0, 1.0) if self.chunk_idx == 0 else None

        if self.chunk_idx == 0:
            input_shapes = [[1, self.config.num_mel_bins, self.config.max_source_positions * 2], *lora_shapes]
        else:
            input_shapes = [[1, self.config.max_source_positions, self.config.hidden_size], *lora_shapes]
        input_value_ranges = [
            minmax,
            *[None for _ in range(len(lora_shapes))],
        ]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        return_data = [
                            np.random.rand(1, self.config.num_mel_bins, self.config.max_source_positions * 2).astype(
                                np.float32
                            )
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, self.config.max_source_positions, self.config.hidden_size).astype(
                                np.float32
                            )
                        ]

                    yield return_data
        else:

            def calib_data_gen():
                for i, f in enumerate(
                    utils.get_sorted_path_list(
                        os.path.join(os.path.join(args.calibration_dataset + 'encoder'), f'chunk_{self.chunk_idx}'),
                        '.npz',
                        sep='-',
                    )
                ):
                    data = np.load(f)
                    return_data = [data['hidden_states'].astype(np.float32)]

                    return_data.extend([*lora_inputs[calib_lora_map[i]]])

                    yield return_data

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for i in range(10):
                    if self.chunk_idx == 0:
                        return_data = [
                            np.random.rand(1, self.config.num_mel_bins, self.config.max_source_positions * 2).astype(
                                np.float32
                            ),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, self.config.max_source_positions, self.config.hidden_size).astype(
                                np.float32
                            ),
                            *lora_inputs[calib_lora_map[i]],
                        ]

                    yield return_data
        else:

            def eval_data_gen():
                for i, f in enumerate(
                    utils.get_sorted_path_list(
                        os.path.join(os.path.join(args.evaluation_dataset + 'encoder'), f'chunk_{self.chunk_idx}'),
                        '.npz',
                        sep='-',
                    )
                ):
                    data = np.load(f)
                    return_data = [data['hidden_states'].astype(np.float32)]
                    return_data.extend([*lora_inputs[calib_lora_map[i]]])

                    yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        # since the model does not expect to contain merged QKV or UpGate FCs.
        # {
        #     internal_identifying_key: expected_state_dict_key,
        #     ...
        # }
        state_dict_mapping = {}

        if self.chunk_idx == 0:
            state_dict_mapping = {
                'conv1.weight': {'conv1.weight': 'encoder.conv1.weight'},
                'conv1.bias': {'conv1.bias': 'encoder.conv1.bias'},
                'conv2.weight': {'conv2.weight': 'encoder.conv2.weight'},
                'conv2.bias': {'conv2.bias': 'encoder.conv2.bias'},
                'embed_positions.weight': {'embed_positions.weight': 'encoder.embed_positions.weight'},
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
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.weight'
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.bias'
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.weight'
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias':
                        f'encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.bias'
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight':
                        f'encoder.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                    },
                }
            )

            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias':
                            f'encoder.layers.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias':
                            f'encoder.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
                        },
                    }
                )
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())

        if self.first_layer_idx + self.num_layers == self.config.num_hidden_layers:
            state_dict_mapping.update(
                {
                    'final_norm_weight': {'norm.weight': 'encoder.layer_norm.weight'},
                    'final_norm_bias': {'norm.bias': 'encoder.layer_norm.bias'},
                }
            )
            for dec_idx in range(self.config.decoder_num_layers):
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_dec_kv_{dec_idx}_k_weight': {
                            f'encoder_tails_kv.{dec_idx}.k_proj.weight':
                            f'decoder.layers.{dec_idx}.encoder_attn.k_proj.weight'
                        },
                        f'{outer_layer_idx}_dec_kv_{dec_idx}_k_bias': {
                            f'encoder_tails_kv.{dec_idx}.k_proj.bias':
                            f'decoder.layers.{dec_idx}.encoder_attn.k_proj.bias'
                        },
                        f'{outer_layer_idx}_dec_kv_{dec_idx}_v_weight': {
                            f'encoder_tails_kv.{dec_idx}.v_proj.weight':
                            f'decoder.layers.{dec_idx}.encoder_attn.v_proj.weight'
                        },
                        f'{outer_layer_idx}_dec_kv_{dec_idx}_v_bias': {
                            f'encoder_tails_kv.{dec_idx}.v_proj.bias':
                            f'decoder.layers.{dec_idx}.encoder_attn.v_proj.bias'
                        },
                    }
                )
        self.state_dict_mapping = state_dict_mapping
        # fmt: on

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter WhisperEncoderChunk load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')

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

            if len(internal_key.split('_')) > 1:
                dtype = self.dtype if internal_key.split('_')[-2] != 'norm' else torch.float32
            else:
                dtype = torch.float32

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
            if self.chunk_idx == 0:
                self.conv1.to(self.device_list[i])
                self.conv2.to(self.device_list[i])
                self.gelu1.to(self.device_list[i])
                self.gelu2.to(self.device_list[i])
                self.embed_positions.to(self.device_list[i])
            if self.first_layer_idx + self.num_layers == self.config.num_hidden_layers:
                self.norm.to(self.device_list[i])
                for j in range(self.config.decoder_num_layers):
                    self.encoder_tails_kv[j].to(self.device_list[i])

        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict
