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
"""Define Whisper Decoder model class."""

import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger, qat_utils, utils
from ...utils.const import DEFAULT_JIT_TRACE_CACHE_SIZE, DEFAULT_JIT_TRACE_NUM_TOKEN
from ..activations import TorchGelu
from ..modeling_base import BaseModelChunk
from ..norm import LayerNorm, RMSNorm
from .attention import WhisperAttention
from .configuration_whisper import WhisperDecoderConfig
from .modeling_common import Tail


class WhisperMLP(nn.Module):
    """Whisper Multi-Layer Perceptron (MLP) class.

    This class implements the Whisper MLP component with optional LoRA (Low-Rank Adaptation) support.

    Attributes:
        config (WhisperDecoderConfig): Configuration for the WhisperMLP.
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

    def __init__(self, config: WhisperDecoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
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
            up = self.up_proj(x)
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
            up = self.up_lora_add(up, up_lora)
        else:
            up = self.up_proj(x)

        pre_down = self.gelu(up)
        if self.lora is not None and self.config.fc_names[None]['down'] in self.lora.target_modules and self.with_lora:
            down = self.down_proj(pre_down)
            if self.parallel_lora:
                down_lora = self.down_lora_B(self.down_lora_A(self.down_lora_dropout(pre_down)))
                if self.lora.scale != 1.0:
                    down_lora = self.down_lora_scale_mul(down_lora, self.lora.scale)
            else:
                down_lora = self.down_lora_B(
                    self.down_lora_A(pre_down, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            down = self.down_lora_add(down, down_lora)
        else:
            down = self.down_proj(pre_down)

        return down


class WhisperDecoderAttention(WhisperAttention):
    """Llama Attention class.

    This class extends the Attention class for the Whisper Decoder model.
    """

    def __init__(self, config: WhisperDecoderConfig, **kwargs):
        """Initializes the WhisperDecoderAttention class.

        Args:
            config: A WhisperDecoderConfig object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, **kwargs)


class WhisperCrossAttention(nn.Module):
    """Whisper Cross Attention class.

    This class implements the attention mechanism for the Whisper Cross model.

    Attributes:
        config: The configuration for the Whisper Encoder model.
        lora (LoRA): LoRA object.
        layer_idx (int): The index of the layer.
        jit_trace (bool): Whether to use JIT tracing.
        hidden_size (int): The hidden size of the attention mechanism.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        attn_scale (float): The scaling factor for the query projection.
        q_proj (nn.Linear): The linear layer for the query projection.
        o_proj (nn.Linear): The linear layer for the output projection.
        mul (mtk_quantization.pytorch.functional.Mul): Addition function from mtk_quantization.
        matmul1 (mtk_quantization.pytorch.functional.Matmul): Addition function from mtk_quantization.
        matmul2 (mtk_quantization.pytorch.functional.Matmul): Addition function from mtk_quantization.
        with_lora (bool): Whether to use LoRA.

    Methods:
        __init__: Initializes the WhisperCrossAttention class.
        forward: Performs the forward pass to compute the cross attention output.
    """

    def __init__(self, config: WhisperDecoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initializes the WhisperEncoderAttention class.

        Args:
            config: The configuration for the Whisper Encoder model.
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
        self.layer_idx = layer_idx
        self.jit_trace = jit_trace
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.attn_scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.matmul1 = mtk_quantization.pytorch.functional.Matmul()
        self.matmul2 = mtk_quantization.pytorch.functional.Matmul()
        self.mul = mtk_quantization.pytorch.functional.Mul()

        # Lora modules
        self.with_lora = False
        if self.lora is not None:
            self.with_lora = self.layer_idx >= self.lora.start_idx and self.layer_idx <= self.lora.end_idx
            if self.with_lora:
                if self.parallel_lora:
                    for module in self.lora.target_modules:
                        if module == self.config.fc_names['cross_attn']['q']:
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
                        elif module == self.config.fc_names['cross_attn']['o']:
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
                        if module == self.config.fc_names['cross_attn']['q']:
                            self.q_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.q_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.q_lora_add = mtk_quantization.pytorch.functional.Add()
                        elif module == self.config.fc_names['cross_attn']['o']:
                            self.o_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.o_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.o_lora_add = mtk_quantization.pytorch.functional.Add()

    def forward(self, hidden_states, cross_key, cross_value, *lora_inputs):
        """Performs the forward pass to compute the attention output.

        Input shape: Batch x Max Source Position x Hidden Size.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            cross_key (torch.Tensor): The input cross attention key_proj done in encoder hidden states.
            cross_value (torch.Tensor): The input cross attention value_proj done in encoder hidden states.
            lora_inputs (tuple): Lora inputs.

        Returns:
            torch.Tensor: The attention output.

        Raises:
            ValueError: If the size of the attention mask or causal attention mask is incorrect.
        """
        bsz, q_len, _ = hidden_states.size()

        lora_idx = 0

        if self.lora is not None and self.config.fc_names['attn']['q'] in self.lora.target_modules and self.with_lora:
            query_states = self.mul(self.q_proj(hidden_states))
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
                self.mul(self.q_proj(hidden_states), self.attn_scale)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        key_states = cross_key
        value_states = cross_value

        attn_weights = self.matmul1(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = self.matmul2(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

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


class WhisperDecoderLayer(nn.Module):
    """Whisper Decoder layer class for the model.

    This class implements a whisper decoder layer with self-attention and cross-attention components,
    and optional LoRA (Low-Rank Adaptation) support.

    Attributes:
        hidden_size (int): Size of the hidden layer.
        intermediate_size (int): Size of the intermediate layer.
        jit_trace (bool): Whether to use JIT tracing.
        exclude_input_norm (bool): Whether to exclude input normalization.
        self_attn (Attention): Self-attention mechanism.
        cross_attn (WhisperCrossAttention): Cross Attention layer.
        mlp (WhisperMLP): Multi-Layer Perceptron.
        input_norm (nn.Module): Normalization layer for input.
        post_attention_norm (nn.Module): Normalization layer for post-attention.
        add (mtk_quantization.pytorch.functional.Add): Addition function from mtk_quantization.
        add2 (mtk_quantization.pytorch.functional.Add): Second addition function from mtk_quantization.
        expected_attn_lora_inputs (int): Expected number of LoRA inputs for attention.
        expected_cross_attn_lora_inputs (int): Expected number of LoRA inputs for cross attention.
        expected_mlp_lora_inputs (int): Expected number of LoRA inputs for MLP.
    """

    def __init__(
        self,
        config: WhisperDecoderConfig,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=WhisperAttention,
        mlp_class=WhisperMLP,
        norm_class=LayerNorm,
        parallel_lora=False,
        exclude_input_norm=False,
        use_single_bmm_attention=False,
    ):
        """Initializes the DecoderLayer class.

        Args:
            config (BaseConfig): Configuration for the decoder layer.
            lora (LoRA): LoRA object.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            layer_idx (int, optional): Index of the layer. Default is None.
            attn_class (type, optional): Class for the attention mechanism. Default is Attention.
            cross_attn_class (type, optional): Class for the cross attention mechanism.
            mlp_class (type, optional): Class for the MLP mechanism.
            norm_class (type, optional): Class for the normalization. Default is LayerNorm.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            exclude_input_norm (bool, optional): Whether to exclude input normalization. Default is False.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.jit_trace = jit_trace
        self.exclude_input_norm = exclude_input_norm
        self.self_attn = attn_class(
            config,
            lora,
            layer_idx,
            jit_trace=jit_trace,
            parallel_lora=parallel_lora,
            use_single_bmm_attention=use_single_bmm_attention,
        )
        self.cross_attn = WhisperCrossAttention(
            config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora
        )
        self.mlp = mlp_class(config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora)

        self.post_attention_norm = LayerNorm(self.hidden_size, eps=config.norm_eps)

        self.add = mtk_quantization.pytorch.functional.Add()
        self.add2 = mtk_quantization.pytorch.functional.Add()
        self.add3 = mtk_quantization.pytorch.functional.Add()
        if norm_class is None:
            norm_class = RMSNorm if config.norm == 'RMSNorm' else LayerNorm

        if not self.exclude_input_norm:
            self.input_norm = norm_class(config.hidden_size, eps=config.norm_eps).float()

        self.post_attention_norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.post_cross_attention_norm = norm_class(config.hidden_size, eps=config.norm_eps).float()

        self.expected_attn_lora_inputs = 0
        self.expected_cross_attn_lora_inputs = 0
        self.expected_mlp_lora_inputs = 0
        if not parallel_lora and lora is not None and lora.start_idx <= layer_idx and lora.end_idx >= layer_idx:
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
                elif module in [v for k, v in config.fc_names['cross_attn'].items() if k != 'name']:
                    if module == config.fc_names['cross_attn']['k'] or module == config.fc_names['cross_attn']['v']:
                        logger.error(f'KV FC not supported. {module} is unsupported ', err=NotImplementedError)
                    self.expected_cross_attn_lora_inputs += 2
                elif module in [v for k, v in config.fc_names['mlp'].items() if k != 'name']:
                    self.expected_mlp_lora_inputs += 2

    def forward(self, hidden_states, mask, past_key, past_value, cross_key, cross_value, *lora_inputs):
        """Performs the forward pass to compute the decoder layer output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            mask (torch.Tensor): Attention mask.
            past_key (torch.Tensor): Past key tensor.
            past_value (torch.Tensor): Past value tensor.
            cross_key (torch.Tensor): Cross key tensor.
            cross_value (torch.Tensor): Cross value tensor.
            *lora_inputs: Additional inputs for LoRA.

        Returns:
            torch.Tensor: The decoder layer output.
        """
        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.input_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.input_norm(hidden_states.to(torch.float32)).to(dtype)

        attn_outputs = self.self_attn(
            hidden_states, mask, past_key, past_value, *lora_inputs[: self.expected_attn_lora_inputs]
        )
        attn_output, present_key, present_value = attn_outputs
        hidden_states = self.add(residual, attn_output)

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.post_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(hidden_states.to(torch.float32)).to(dtype)
        hidden_states = self.cross_attn(
            hidden_states,
            cross_key,
            cross_value,
            *lora_inputs[
                self.expected_attn_lora_inputs : self.expected_attn_lora_inputs + self.expected_cross_attn_lora_inputs
            ],
        )
        hidden_states = self.add2(residual, hidden_states)

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.post_cross_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_cross_attention_norm(hidden_states.to(torch.float32)).to(dtype)
        hidden_states = self.mlp(hidden_states, *lora_inputs[-self.expected_mlp_lora_inputs :])
        hidden_states = self.add3(residual, hidden_states)

        return hidden_states, present_key, present_value


class WhisperDecoderTail(Tail):
    """Whisper Decoder Tail class.

    This class extends the Tail class for the Whisper decoder model.
    """

    def __init__(self, config: WhisperDecoderConfig, **kwargs):
        """Initializes the WhisperDecoderTail class.

        Args:
            config: A WhisperDecoderConfig object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, **kwargs)

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {
            'final_norm_weight': {'norm.weight': f'{self.norm_names["final"]}.weight'},
            'lm_head_weight': {'lm_head.weight': f'{self.fc_names["tail"]["name"]}.weight'},
            'lm_head_bias': {'lm_head.bias': f'{self.fc_names["tail"]["name"]}.bias'},
        }
        if self.config.norm == 'LayerNorm':
            state_dict_mapping.update({'final_norm_bias': {'norm.bias': f'{self.norm_names["final"]}.bias'}})

        self.state_dict_mapping = state_dict_mapping

    def load_weights(self, state_dict, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter BaseModelTail load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        # Check if merged weights/biases exist. Split them if exist.
        for name, external_key in self.merged_state_dict_mapping.items():
            found_key = None
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(f'Found {name} merged weight/bias using prefix')
                    found_key = key_to_test
                    break

            if found_key is None:
                for k in state_dict_keys:
                    if k.endswith('.' + external_key):
                        logger.debug(
                            f'Found {name} merged weight/bias using iteration. Adding prefix: {k[: -len(external_key)]}'
                        )
                        self.prefixes.append(k[: -len(external_key)])
                        found_key = k
                        break

            if found_key is not None:
                layer_id = int(name.split('_')[0])
                if '_qkv_weight' in name:
                    # Split QKV weight
                    logger.debug(f'Splitting layer {layer_id} {name} merged weight into Q/K/V weights')
                    q_size = self.config.hidden_size
                    kv_size = self.config.num_key_value_heads * self.head_dim
                    qkv_weight = state_dict.pop(found_key)
                    state_dict.update(
                        {
                            next(iter(self.state_dict_mapping[f'{layer_id}_q_weight'].values())): qkv_weight[
                                :q_size, :
                            ],
                            next(iter(self.state_dict_mapping[f'{layer_id}_k_weight'].values())): qkv_weight[
                                q_size : q_size + kv_size, :
                            ],
                            next(iter(self.state_dict_mapping[f'{layer_id}_v_weight'].values())): qkv_weight[
                                q_size + kv_size : q_size + 2 * kv_size, :
                            ],
                        }
                    )
                elif '_qkv_bias' in name:
                    # Split QKV bias
                    logger.debug(f'Splitting layer {layer_id} {name} merged bias into Q/K/V biases')
                    qkv_bias = state_dict.pop(found_key)
                    state_dict.update(
                        {
                            next(iter(self.state_dict_mapping[f'{layer_id}_q_bias'].values())): qkv_bias[:q_size],
                            next(iter(self.state_dict_mapping[f'{layer_id}_k_bias'].values())): qkv_bias[
                                q_size : q_size + kv_size
                            ],
                            next(iter(self.state_dict_mapping[f'{layer_id}_v_bias'].values())): qkv_bias[
                                q_size + kv_size : q_size + 2 * kv_size
                            ],
                        }
                    )
                elif '_gu_weight' in name:
                    # Split GateUp weight
                    logger.debug(f'Splitting layer {layer_id} {name} merged weight into Gate/Up weights')
                    gu_weight = state_dict.pop(found_key)
                    state_dict.update(
                        {
                            next(iter(self.state_dict_mapping[f'{layer_id}_g_weight'].values())): gu_weight[
                                : self.config.intermediate_size, :
                            ],
                            next(iter(self.state_dict_mapping[f'{layer_id}_u_weight'].values())): gu_weight[
                                self.config.intermediate_size :, :
                            ],
                        }
                    )
                else:
                    assert '_gu_bias' in name
                    # Split GateUp bias
                    logger.debug(f'Splitting layer {layer_id} {name} merged bias into Gate/Up biases')
                    gu_bias = state_dict.pop(found_key)
                    state_dict.update(
                        {
                            next(iter(self.state_dict_mapping[f'{layer_id}_g_bias'].values())): gu_bias[
                                : self.config.intermediate_size
                            ],
                            next(iter(self.state_dict_mapping[f'{layer_id}_u_bias'].values())): gu_bias[
                                self.config.intermediate_size :
                            ],
                        }
                    )

        # Renew state_dict_keys as might be modified by splitting of merged weights
        state_dict_keys = list(state_dict.keys())
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

            if not found and internal_key in ['lm_head_weight', 'lm_head_bias']:
                # Check if tie_word_embeddings
                tail_mapping = {
                    'lm_head_weight': 'decoder.embed_tokens.weight',
                    'lm_head_bias': 'decoder.embed_tokens.bias',
                }
                for pre in self.prefixes:
                    key_to_test = pre + tail_mapping[internal_key]
                    if key_to_test in state_dict_keys:
                        logger.debug(
                            f'Found {internal_key} tail weight/bias (tie_word_embeddings) using prefix, '
                            f'state_dict key={key_to_test}, dtype={dtype}'
                        )
                        weights_to_load.update({model_key: state_dict.pop(key_to_test).to(dtype)})
                        found = True
                        state_dict_keys.remove(key_to_test)
                        break
                if not found:
                    for k in state_dict_keys:
                        if k.endswith(tail_mapping[internal_key]):
                            logger.debug(
                                f'Found {internal_key} tail weight/bias (tie_word_embeddings) using iteration, '
                                f'state_dict key={k}, dtype={dtype}. Adding prefix: '
                                f'{k[: -len(tail_mapping[internal_key])]}'
                            )
                            self.prefixes.append(k[: -len(tail_mapping[internal_key])])
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
                logger.warning(f'Cannot find {internal_key} weight')
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
            self.device_list = ['cpu']
        else:
            self.device_list = [
                f'cuda:{torch.cuda.device_count() - 1 if self.distribute_layers else int(os.getenv("LOCAL_RANK", 0))}'
            ]

        if weights_to_load.keys() != self.state_dict().keys():
            weights_to_load_only_keys = [x for x in weights_to_load if x not in self.state_dict()]
            model_only_keys = [x for x in self.state_dict() if x not in weights_to_load and 'lora' not in x]
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

        self.load_state_dict(weights_to_load, strict=False)
        self.to(self.device_list[0])
        self.eval()

        return self, state_dict


class WhisperDecoderModelChunk(BaseModelChunk):
    """Model chunk class for the model.

    This class implements a chunk of the model, which includes multiple decoder layers and optional tail components.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        embed_layer_norm (LayerNorm): Layer normalization for embeddings.
        norm (nn.Module): Normalization layer.
        lm_head (nn.Linear): Linear layer for language modeling.
        expected_num_cache_inputs (int): Expected number of cache inputs.
        expected_num_lora_inputs (int): Expected number of LoRA inputs.
        support_quant_stub (bool): Whether quantization stub is supported.
        inputs_embeds (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for input embeddings.
        mask (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for mask.
        pos_emb (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for position embeddings.
        past_keys (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for past keys.
        past_values (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for past values.
        cache (nn.ModuleList): List of quantizer stubs for cache.
    """

    def __init__(
        self,
        config: WhisperDecoderConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        jit_trace=False,
        decoder_class=WhisperDecoderLayer,
        norm_class=LayerNorm,
        parallel_lora=False,
        distribute_layers=True,
        use_single_bmm_attention=False,
    ):
        """Initializes the WhisperDecoderModelChunk class.

        Args:
            config (BaseConfig): Configuration for the model chunk.
            lora (LoRA): LoRA object.
            num_layers (int): Number of decoder layers in the chunk.
            first_layer_idx (int): The index of the first decoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            include_tail (bool, optional): Whether to include the tail component. Default is False.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            decoder_class (type, optional): Class for the decoder layer. Default is DecoderLayer.
            norm_class (type, optional): Class for the normalization. Default is None.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            distribute_layers (bool, optional): Whether to distribute layers across devices. Default is True.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__(
            config,
            lora,
            num_layers,
            first_layer_idx,
            chunk_idx,
            dtype,
            include_tail,
            jit_trace,
            parallel_lora,
            distribute_layers,
        )
        if norm_class is None:
            norm_class = RMSNorm if config.norm == 'RMSNorm' else LayerNorm

        self.layers = nn.ModuleList(
            [
                decoder_class(
                    config,
                    lora,
                    jit_trace=jit_trace,
                    layer_idx=first_layer_idx + i,
                    norm_class=norm_class,
                    parallel_lora=parallel_lora,
                    use_single_bmm_attention=use_single_bmm_attention,
                )
                for i in range(num_layers)
            ]
        )

        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            self.embed_layer_norm = LayerNorm(config.hidden_size).float()

        if self.include_tail:
            self.norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + config.lm_head_pad_size)

        self.expected_num_pos_emb_inputs = 1 if self.first_layer_idx == 0 else 0
        self.expected_num_cache_inputs_per_layer = 4
        self.expected_num_cache_inputs = self.expected_num_cache_inputs_per_layer * num_layers
        if not self.parallel_lora and self.with_lora:
            self.expected_num_lora_inputs_per_layer = [0] * num_layers
            if self.lora is not None:
                for idx in range(num_layers):
                    cur_layer_idx = first_layer_idx + idx
                    if self.lora.start_idx > cur_layer_idx or self.lora.end_idx < cur_layer_idx:
                        continue
                    self.expected_num_lora_inputs_per_layer[idx] = len(self.lora.target_modules) * 2

            self.expected_num_lora_inputs = sum(self.expected_num_lora_inputs_per_layer)
        else:
            self.expected_num_lora_inputs_per_layer = [0] * num_layers
            self.expected_num_lora_inputs = 0

        self.expected_num_inputs = (
            2 + self.expected_num_pos_emb_inputs + self.expected_num_cache_inputs + self.expected_num_lora_inputs
        )

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
        mask = inputs[1]
        if self.chunk_idx == 0:
            pos_emb = inputs[2]
            i = 3
        else:
            i = 2
        cache = inputs[i : i + self.expected_num_cache_inputs]
        i += self.expected_num_cache_inputs
        lora_inputs = inputs[i:] if self.expected_num_lora_inputs > 0 else []

        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            if self.jit_trace:
                inputs_embeds = self.embed_layer_norm(inputs_embeds)
            else:
                inputs_embeds = self.embed_layer_norm(inputs_embeds.to(torch.float32)).to(self.dtype)
        if self.first_layer_idx == 0:
            inputs_embeds = inputs_embeds + pos_emb.to(self.device_list[0])
        hidden_states = inputs_embeds

        cache_outputs = []
        for idx, decoder_layer in enumerate(self.layers):
            decoder_outputs = decoder_layer(
                hidden_states.to(self.device_list[idx]),
                mask.to(self.device_list[idx]),
                *[
                    x.to(self.device_list[idx])
                    for x in cache[
                        idx * self.expected_num_cache_inputs_per_layer : (idx + 3)
                        * self.expected_num_cache_inputs_per_layer
                    ]
                ],
                *[
                    x.to(self.device_list[idx])
                    for x in lora_inputs[
                        sum(self.expected_num_lora_inputs_per_layer[:idx]) : sum(
                            self.expected_num_lora_inputs_per_layer[: idx + 1]
                        )
                    ]
                ],
            )
            hidden_states = decoder_outputs[0]
            cache_outputs.append(decoder_outputs[1].to(inputs_embeds.device))
            cache_outputs.append(decoder_outputs[2].to(inputs_embeds.device))

        if self.include_tail:
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)
            hidden_states = self.lm_head(hidden_states)
        return hidden_states, *cache_outputs

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        if self.config.sparse_attn and self.chunk_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        num_attention_heads = self.config.num_attention_heads

        pos_emb_inputs = (
            [torch.randn(1, 1, self.config.hidden_size, device='cpu', dtype=torch.float32)]
            if self.chunk_idx == 0
            else []
        )

        cache_inputs = []
        for _ in range(self.num_layers):
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # K cache
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # V cache
            cache_inputs.append(
                torch.randn(
                    self.num_layers,
                    num_head,
                    self.config.max_source_positions,
                    self.head_dim,
                    device='cpu',
                    dtype=torch.float32,
                )
            )  # cross K cache
            cache_inputs.append(
                torch.randn(
                    self.num_layers,
                    num_head,
                    self.config.max_source_positions,
                    self.head_dim,
                    device='cpu',
                    dtype=torch.float32,
                )
            )  # cross V cache

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
                                1,
                                self.head_dim * num_attention_heads,
                                self.config.lora_rank,
                                device='cpu',
                                dtype=torch.float32,
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
                                1,
                                self.config.lora_rank,
                                self.head_dim * num_attention_heads,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                        lora_inputs.append(
                            torch.randn(
                                1, self.config.hidden_size, self.config.lora_rank, device='cpu', dtype=torch.float32
                            )
                        )
                    elif module == self.config.fc_names['mlp']['gate'] or module == self.config.fc_names['mlp']['up']:
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

        return (
            torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size, device='cpu', dtype=torch.float32),
            torch.randn(
                1,
                1,
                DEFAULT_JIT_TRACE_NUM_TOKEN,
                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                device='cpu',
                dtype=torch.float32,
            ),
            *pos_emb_inputs,
            *cache_inputs,
            *lora_inputs,
        )

    def get_ptq_inputs(self, args, exp_name, lora_inputs=None, calib_lora_map=None, eval_lora_map=None, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        This method generates inputs for PTQ, including LoRA inputs if applicable.

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            lora_inputs (list, optional): List of number of lora scenarios of list of LoRA inputs per scenario.
                Default is None.
            calib_lora_map (list, optional): List of LoRA scenario mappings for calibration dataset. Default is None.
            eval_lora_map (list, optional): List of LoRA scenario mappings for evaluation dataset. Default is None.
            kwargs: Other keyword arguments.

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
                calib_ds_len = len(os.listdir(os.path.join(args.calibration_dataset, 'llm', f'chunk_{self.chunk_idx}')))
            calib_lora_map = [0 for _ in range(calib_ds_len)]
        if eval_lora_map is None:
            if args.evaluation_dataset is None:
                eval_ds_len = 0
            elif args.evaluation_dataset == 'fake':
                eval_ds_len = 10
            else:
                eval_ds_len = len(os.listdir(os.path.join(args.evaluation_dataset, 'llm', f'chunk_{self.chunk_idx}')))
            eval_lora_map = [0 for _ in range(eval_ds_len)]
        if lora_inputs is None:
            lora_inputs = [[]]

        if self.config.sparse_attn and self.first_layer_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        num_attention_heads = self.config.num_attention_heads

        if args.zero_lora_inputs:
            assert self.with_lora
            assert not self.parallel_lora
            num_lora_choices = len(lora_inputs)
            num_lora_layer_inputs = len(lora_inputs[0])
            for i in range(num_lora_choices):
                for j in range(num_lora_layer_inputs):
                    lora_inputs[i][j] = np.zeros_like(lora_inputs[i][j])
        if self.parallel_lora:
            lora_inputs = [[]]

        if not args.dummy_weights and self.first_layer_idx == 0 and not args.get_emb_minmax_from_cal_data:
            weight_dir = utils.get_dirpath(args.config)
            emb_minmax = utils.extract_emb_minmax(weight_dir)
        else:
            emb_minmax = None

        rot_emb_minmax = (-1.0, 1.0)

        lora_shapes = []
        if self.lora is not None and not self.parallel_lora:
            for i in range(self.num_layers):
                if not self.with_lora[i]:
                    continue
                for module in self.lora.target_modules:
                    if module == self.config.fc_names['attn']['q']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, self.head_dim * num_attention_heads, None])
                    elif module == self.config.fc_names['attn']['k'] or module == self.config.fc_names['attn']['v']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, num_head * self.head_dim, None])
                    elif module == self.config.fc_names['attn']['o']:
                        lora_shapes.append([None, None, self.head_dim * num_attention_heads])
                        lora_shapes.append([None, self.config.hidden_size, None])
                    elif module == self.config.fc_names['mlp']['gate'] or module == self.config.fc_names['mlp']['up']:
                        lora_shapes.append([None, None, self.config.hidden_size])
                        lora_shapes.append([None, self.config.intermediate_size, None])
                    elif module == self.config.fc_names['mlp']['down']:
                        lora_shapes.append([None, None, self.config.intermediate_size])
                        lora_shapes.append([None, self.config.hidden_size, None])

        pos_emb_shapes = [[None, None, self.config.hidden_size]] if self.chunk_idx == 0 else []

        cache_shapes = [
            [None, num_head, None, self.head_dim],
            [None, num_head, None, self.head_dim],
            [None, num_head, self.config.max_source_positions, self.head_dim],
            [None, num_head, self.config.max_source_positions, self.head_dim],
        ]

        input_shapes = [
            [None, None, self.config.hidden_size],
            [None, 1, None, None],
            *pos_emb_shapes,
            *cache_shapes,
            *lora_shapes,
        ]
        input_value_ranges = [
            emb_minmax,
            (self.config.mask_value, 0.0),
            *[rot_emb_minmax for _ in range(len(pos_emb_shapes))],
            *[None for _ in range(len(cache_shapes))],
            *[None for _ in range(len(lora_shapes))],
        ]

        if args.calibration_dataset is None:
            calib_data_gen = None
        elif args.calibration_dataset == 'fake':

            def calib_data_gen():
                for i in range(10):
                    pos_emb_data = [np.random.rand(1, 1, self.config.hidden_size).astype(np.float32)]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(
                            self.num_layers, num_head, self.config.max_source_positions, self.head_dim
                        ).astype(np.float32),
                        np.random.rand(
                            self.num_layers, num_head, self.config.max_source_positions, self.head_dim
                        ).astype(np.float32),
                    ]
                    if self.chunk_idx == 0:
                        return_data = [
                            np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                            np.random.rand(
                                1,
                                1,
                                DEFAULT_JIT_TRACE_NUM_TOKEN,
                                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                            ).astype(np.float32),
                            *pos_emb_data,
                            *cache_data,
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                            np.random.rand(
                                1,
                                1,
                                DEFAULT_JIT_TRACE_NUM_TOKEN,
                                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                            ).astype(np.float32),
                            *cache_data,
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    yield return_data
        else:

            def calib_data_gen():
                for i, f in enumerate(
                    utils.get_sorted_path_list(
                        os.path.join(args.calibration_dataset, 'llm', f'chunk_{self.chunk_idx}'),
                        '.npz',
                        sep='-',
                    )
                ):
                    data = np.load(f)
                    return_data = [
                        data['inputs_embeds'].astype(np.float32),
                        data['mask'].astype(np.float32),
                    ]
                    if self.chunk_idx == 0:
                        return_data.extend([data['pos_emb'].astype(np.float32)])

                    return_data.extend(
                        [
                            data['past_keys'].astype(np.float32),
                            data['past_values'].astype(np.float32),
                            data['cross_key'].astype(np.float32),
                            data['cross_value'].astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    )
                    yield return_data

        if args.evaluation_dataset is None:
            eval_data_gen = None
        elif args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for i in range(10):
                    pos_emb_data = [np.random.rand(1, 1, self.config.hidden_size).astype(np.float32)]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(
                            self.num_layers, num_head, self.config.max_source_positions, self.head_dim
                        ).astype(np.float32),
                        np.random.rand(
                            self.num_layers, num_head, self.config.max_source_positions, self.head_dim
                        ).astype(np.float32),
                    ]
                    if self.chunk_idx == 0:
                        return_data = [
                            np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                            np.random.rand(
                                1,
                                1,
                                DEFAULT_JIT_TRACE_NUM_TOKEN,
                                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                            ).astype(np.float32),
                            *pos_emb_data,
                            *cache_data,
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                            np.random.rand(
                                1,
                                1,
                                DEFAULT_JIT_TRACE_NUM_TOKEN,
                                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                            ).astype(np.float32),
                            *cache_data,
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    yield return_data
        else:

            def eval_data_gen():
                for i, f in enumerate(
                    utils.get_sorted_path_list(
                        os.path.join(args.calibration_dataset, 'llm', f'chunk_{self.chunk_idx}'),
                        '.npz',
                        sep='-',
                    )
                ):
                    data = np.load(f)
                    return_data = [
                        data['inputs_embeds'].astype(np.float32),
                        data['mask'].astype(np.float32),
                    ]
                    if self.chunk_idx == 0:
                        return_data.extend([data['pos_emb'].astype(np.float32)])

                    return_data.extend(
                        [
                            data['past_keys'].astype(np.float32),
                            data['past_values'].astype(np.float32),
                            data['cross_key'].astype(np.float32),
                            data['cross_value'].astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    )
                    yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    def _add_quantizer_weights(
        self,
        state_dict,
        weights_to_load,
        quantizer_dict,
        quantizer_type,
        prefix,
    ):
        """Adds quantizer to the state dictionary.

        Args:
            state_dict (dict): Original state dictionary containing the model weights.
            weights_to_load (dict): State dictionary to load into model.
            quantizer_dict (dict): The dictionary containing the quantizer information.
            quantizer_type (str): Type of quantizer ('weight' or 'activation').
            prefix (str): Prefix for the state dictionary keys.

        Returns:
            dict: Updated temporary state dictionary.
        """
        logger.debug(f'Adding quantizer weight {quantizer_dict["target_name"]} into weights_to_load')
        assert quantizer_type in ['weight', 'activation']
        assert quantizer_dict['quantizer']['type'] in ['LastValueQuantizer', 'EMAQuantizer']
        model_subkey = (
            quantizer_dict['target_name']
            .replace(self.norm_names['input'], 'input_norm')
            .replace(self.norm_names['post_attn'], 'post_attention_norm')
        )
        model_subkey = quantizer_dict['target_name']
        quantizer_type_subkey = '_weight_quantizer' if quantizer_type == 'weight' else '_act_quantizer'

        # Hacking of name mapping to match state_dict to model
        state_dict_subkey = prefix + quantizer_dict['target_name']
        state_dict_subkey = (
            quantizer_dict['target_name']
            .replace('input_norm', self.norm_names['input'])
            .replace('post_attention_norm', self.norm_names['post_attn'])
        )
        if quantizer_type == 'weight':
            model_subkey = model_subkey.replace('.weight', '')
            state_dict_subkey = state_dict_subkey.replace('.weight', '')
        if 'layers.' in state_dict_subkey:
            inner_layer_idx = int(state_dict_subkey.split('layers.')[1].split('.')[0])
            outer_layer_idx = inner_layer_idx + self.first_layer_idx
            state_dict_subkey = (
                state_dict_subkey.split(f'.{inner_layer_idx}.')[0]
                + f'.{outer_layer_idx}.'
                + state_dict_subkey.split(f'.{inner_layer_idx}.')[1]
            )

        weights_to_load = {
            **weights_to_load,
            model_subkey + f'.{quantizer_type_subkey}._min_vals': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._min_vals'
            ),
            model_subkey + f'.{quantizer_type_subkey}._max_vals': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._max_vals'
            ),
            model_subkey + f'.{quantizer_type_subkey}._shadow_min_vals': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._shadow_min_vals'
            ),
            model_subkey + f'.{quantizer_type_subkey}._shadow_max_vals': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._shadow_max_vals'
            ),
            model_subkey + f'.{quantizer_type_subkey}._bitwidth': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._bitwidth'
            ),
            model_subkey + f'.{quantizer_type_subkey}._is_frozen': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._is_frozen'
            ),
            model_subkey + f'.{quantizer_type_subkey}._is_disabled': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._is_disabled'
            ),
            model_subkey + f'.{quantizer_type_subkey}._symmetric': state_dict.pop(
                state_dict_subkey + f'.{quantizer_type_subkey}._symmetric'
            ),
        }
        if quantizer_dict['quantizer']['type'] == 'EMAQuantizer':
            weights_to_load = {
                **weights_to_load,
                model_subkey + f'.{quantizer_type_subkey}._is_ema_init': state_dict.pop(
                    state_dict_subkey + f'.{quantizer_type_subkey}._is_ema_init'
                ),
            }
        return weights_to_load

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        # merged_state_dict_mapping should just be a dict,
        # since the model does not expect to contain merged QKV or UpGate FCs.
        # {
        #     internal_identifying_key: expected_state_dict_key,
        #     ...
        # }
        state_dict_mapping = {}
        merged_state_dict_mapping = {}

        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            state_dict_mapping = {
                'stable_embedding_weight': {'embed_layer_norm.weight': 'embed_layer_norm.weight'},
                'stable_embedding_bias': {'embed_layer_norm.bias': f'{self.norm_names["stable_embedding"]}.bias'},
            }

        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            merged_state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_qkv_weight': f'layers.{inner_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.weight',
                    f'{outer_layer_idx}_qkv_bias': f'layers.{inner_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.bias',
                }
            )
            # fmt: off
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_cq_weight': {
                        f'layers.{inner_layer_idx}.cross_attn.q_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["cross_attn"]["name"]}.{self.fc_names["cross_attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_cq_bias': {
                        f'layers.{inner_layer_idx}.cross_attn.q_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["cross_attn"]["name"]}.{self.fc_names["cross_attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_co_weight': {
                        f'layers.{inner_layer_idx}.cross_attn.o_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["cross_attn"]["name"]}.{self.fc_names["cross_attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_co_bias': {
                        f'layers.{inner_layer_idx}.cross_attn.o_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["cross_attn"]["name"]}.{self.fc_names["cross_attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.weight'
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.bias'
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.weight'
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias':
                        f'decoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.bias'
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                    },
                    f'{outer_layer_idx}_post_cross_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_cross_attention_norm.weight':
                        f'decoder.layers.{outer_layer_idx}.{self.norm_names["post_cross_attn"]}.weight'
                    },
                }
            )
            if self.config.use_qk_norm:
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_q_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.q_norm.weight':
                            f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["query"]}.weight'
                        },
                        f'{outer_layer_idx}_k_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.k_norm.weight':
                            f'decoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["key"]}.weight'
                        },
                    }
                )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias':
                            f'decoder.layers.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias':
                            f'decoder.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
                        },
                        f'{outer_layer_idx}_post_cross_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_cross_attention_norm.bias':
                            f'decoder.layers.{outer_layer_idx}.{self.norm_names["post_cross_attn"]}.bias'
                        },
                    }
                )
            # fmt: on
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())
        if self.include_tail:
            state_dict_mapping.update(
                {
                    'final_norm_weight': {'norm.weight': f'{self.norm_names["final"]}.weight'},
                    'lm_head_weight': {'lm_head.weight': f'{self.fc_names["tail"]["name"]}.weight'},
                    'lm_head_bias': {'lm_head.bias': f'{self.fc_names["tail"]["name"]}.bias'},
                }
            )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update({'final_norm_bias': {'norm.bias': f'{self.norm_names["final"]}.bias'}})

        self.state_dict_mapping = state_dict_mapping
        self.merged_state_dict_mapping = merged_state_dict_mapping
