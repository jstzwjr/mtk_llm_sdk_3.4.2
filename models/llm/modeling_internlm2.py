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
"""PyTorch InternLM2 model."""

import json
import os

import mtk_quantization
import numpy as np
import torch
from einops import rearrange
from torch import nn

from ...utils import logger
from ..norm import RMSNorm
from .attention import Attention
from .configuration_internlm2 import InternLM2Config
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class InternLM2MLP(MLP):
    """InternLM2 MLP class.

    This class extends the MLP class for the InternLM2 model.
    """

    def __init__(self, config: InternLM2Config, lora, layer_idx, **kwargs):
        """Initializes the InternLM2MLP class.

        Args:
            config: A InternLM2Config object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class InternLM2Attention(Attention):
    """InternLM2 Attention class.

    This class extends the Attention class for the InternLM2 model and changes the way qkv matrix is interleaved.
    """

    def __init__(
        self,
        config: InternLM2Config,
        lora,
        layer_idx,
        jit_trace=False,
        parallel_lora=False,
        use_single_bmm_attention=False,
    ):
        """Initializes the InternLM2Attention class.

        Args:
            config (GeckoConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__(
            config=config,
            lora=lora,
            layer_idx=layer_idx,
            jit_trace=jit_trace,
            parallel_lora=parallel_lora,
            use_single_bmm_attention=use_single_bmm_attention,
        )
        # FIXME split qkv
        del self.q_proj
        del self.k_proj
        del self.v_proj

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=True,
        )

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding to query and key states."""
        # q1, q2 = torch.split(q, self.head_dim // 2, dim=-1)
        q1 = q[..., : torch.div(q.shape[-1], 2, rounding_mode='floor')]
        q2 = q[..., torch.div(q.shape[-1], 2, rounding_mode='floor') :]
        q_rotated = self.q_cat((-q2, q1))
        # k1, k2 = torch.split(k, self.head_dim // 2, dim=-1)
        k1 = k[..., : torch.div(k.shape[-1], 2, rounding_mode='floor')]
        k2 = k[..., torch.div(k.shape[-1], 2, rounding_mode='floor') :]
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed

    def forward(self, hidden_states, mask, cos, sin, past_key, past_value, *lora_inputs):
        """Forward pass for the InternLM2Attention class.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.
            mask (torch.Tensor): Mask tensor.
            cos (torch.Tensor): Positional embedding tensor.
            sin (torch.Tensor): Positional embedding tensor.
            past_key (torch.Tensor): Past key tensor.
            past_value (torch.Tensor): Past value tensor.
            *lora_inputs: Additional LoRA inputs.

        Returns:
            tuple: Tuple containing the attention output, key states, and value states.
        """
        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]
        lora_idx = 0

        if self.use_split_mask:
            mask_current = mask
            mask_cache = lora_inputs[lora_idx]
            lora_idx += 1

        if self.lora is not None and self.config.fc_names['attn']['qkv'] in self.lora.target_modules and self.with_lora:
            qkv_states = self.qkv_proj(hidden_states)
            if self.parallel_lora:
                qkv_lora = self.qkv_proj_lora_B(self.qkv_proj_lora_A(self.qkv_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    qkv_lora = self.qkv_lora_scale_mul(qkv_lora, self.lora.scale)
            else:
                qkv_lora = self.qkv_proj_lora_B(
                    self.qkv_proj_lora_A(hidden_states, lora_inputs[lora_idx].transpose(1, 2).to(hidden_states.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(hidden_states.device),
                )
                lora_idx += 2
            qkv_states = self.v_lora_add(qkv_states, qkv_lora)
            qkv_states = rearrange(qkv_states, 'b q d -> (b q) d')

            qkv_states = rearrange(
                qkv_states,
                'bq (h gs d) -> bq h gs d',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )

            qkv_states = torch.chunk(qkv_states, qkv_states.shape[2], dim=2)
            query_states = torch.cat(qkv_states[: self.num_key_value_groups], dim=2)

            # query_states = qkv_states[..., : self.num_key_value_groups, :]
            query_states = rearrange(query_states, 'bq h gs d -> bq (h gs) d')
            query_states = rearrange(query_states, '(b q) hgs d -> b q hgs d', b=bsz).transpose(1, 2)

            # key_states = qkv_states[..., -2, :]
            # value_states = qkv_states[..., -1, :]
            k0, k1, _, _ = qkv_states[-2].shape
            key_states = qkv_states[-2].reshape(k0, k1, -1)  # .squeeze(-2)
            v0, v1, _, _ = qkv_states[-1].shape
            value_states = qkv_states[-1].reshape(v0, v1, -1)  # .squeeze(-2)

            key_states = rearrange(key_states, '(b q) hgs d -> b q hgs d', b=bsz).transpose(1, 2)
            value_states = rearrange(value_states, '(b q) hgs d -> b q hgs d', b=bsz).transpose(1, 2)
        else:
            qkv_states = self.qkv_proj(hidden_states)
            qkv_states = rearrange(qkv_states, 'b q d -> (b q) d')

            qkv_states = rearrange(
                qkv_states,
                'bq (h gs d) -> bq h gs d',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )

            # To eliminate gather, modify to use chunk
            qkv_states = torch.chunk(qkv_states, qkv_states.shape[2], dim=2)
            query_states = torch.cat(qkv_states[: self.num_key_value_groups], dim=2)

            # query_states = qkv_states[..., : self.num_key_value_groups, :]
            query_states = rearrange(query_states, 'bq h gs d -> bq (h gs) d')
            query_states = rearrange(query_states, '(b q) hgs d -> b q hgs d', b=bsz)

            # key_states = qkv_states[..., -2, :]
            # value_states = qkv_states[..., -1, :]
            k0, k1, _, _ = qkv_states[-2].shape
            key_states = qkv_states[-2].reshape(k0, k1, -1)  # .squeeze(-2)
            v0, v1, _, _ = qkv_states[-1].shape
            value_states = qkv_states[-1].reshape(v0, v1, -1)  # .squeeze(-2)

            key_states = rearrange(key_states, '(b q) hgs d -> b q hgs d', b=bsz)
            value_states = rearrange(value_states, '(b q) hgs d -> b q hgs d', b=bsz).transpose(1, 2)

        if not self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # cos, sin = torch.split(pos_emb, 1, dim=1)
        query_states, key_states = self.apply_rotary_pos_emb_mtk(query_states, key_states, cos, sin)

        if self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        key_states_out = key_states
        value_states_out = value_states
        if not self.use_single_bmm_attention and self.config.ring_buffer:
            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len, self.num_key_value_groups)
                past_key = self.repeat_kv(past_key, bsz, c_len, self.num_key_value_groups)
                past_value = self.repeat_kv(past_value, bsz, c_len, self.num_key_value_groups)
            attn_weight_bmm = self.matmul1(query_states, past_key.transpose(2, 3))
            attn_weight_partial_sum = self.matmul2(query_states, key_states.transpose(2, 3))
            if self.use_split_mask:
                attn_weight_bmm = self.div_cache(attn_weight_bmm, self.attn_scale)
                attn_weight_partial_sum = self.div_current(attn_weight_partial_sum, self.attn_scale)
                attn_weight_bmm = self.add_mask_cache(attn_weight_bmm, mask_cache)
                attn_weight_partial_sum = self.add_mask_current(attn_weight_partial_sum, mask_current)
                attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
            else:
                attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
                attn_weights = self.div(attn_weights, self.attn_scale)
        else:
            key_states = self.cat([past_key, key_states])
            value_states = self.cat2([past_value, value_states])

            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len + c_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len + c_len, self.num_key_value_groups)
            attn_weights = self.div(self.matmul1(query_states, key_states.transpose(2, 3)), self.attn_scale)

        if not self.use_split_mask:
            attn_weights = self.add(attn_weights, mask)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if not self.use_single_bmm_attention and self.config.ring_buffer:
            attn_weights_past = attn_weights[..., :c_len]
            attn_output_bmm = self.matmul3(attn_weights_past, past_value)
            attn_output_partial_sum = self.matmul4(attn_weights[..., c_len:], value_states)
            attn_output = self.add2(attn_output_bmm, attn_output_partial_sum)
        else:
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
                    self.o_lora_A(attn_output, lora_inputs[lora_idx].transpose(1, 2).to(hidden_states.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(hidden_states.device),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states_out


class InternLM2DecoderLayer(DecoderLayer):
    """InternLM2 Decoder Layer class.

    This class extends the DecoderLayer class for the InternLM2 model.
    """

    def __init__(
        self,
        config: InternLM2Config,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=InternLM2Attention,
        mlp_class=MLP,
        norm_class=RMSNorm,
        parallel_lora=False,
        exclude_input_norm=False,
        use_single_bmm_attention=False,
    ):
        """Initializes the InternLM2DecoderLayer class.

        Args:
            config (BaseLLMConfig): Configuration for the decoder layer.
            lora (LoRA): LoRA object.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            layer_idx (int, optional): Index of the layer. Default is None.
            attn_class (type, optional): Class for the attention mechanism. Default is Attention.
            mlp_class (type, optional): Class for the MLP. Default is MLP.
            norm_class (type, optional): Class for the normalization. Default is RMSNorm.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            exclude_input_norm (bool, optional): Whether to exclude input normalization. Default is False.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__(
            config=config,
            lora=lora,
            jit_trace=jit_trace,
            layer_idx=layer_idx,
            attn_class=attn_class,
            mlp_class=mlp_class,
            norm_class=norm_class,
            parallel_lora=parallel_lora,
            exclude_input_norm=exclude_input_norm,
            use_single_bmm_attention=use_single_bmm_attention,
        )


class InternLM2Tail(Tail):
    """InternLM2 Tail class.

    This class extends the Tail class for the InternLM2 model.
    """

    def __init__(self, config, chunk_idx, dtype=torch.float32, jit_trace=False, norm_class=None):
        """Initializes the InternLM2Tail class.

        Args:
            config (BaseLLMConfig): Configuration for the tail component.
            chunk_idx (int): Index of the chunk.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            norm_class (type, optional): Class for the normalization. Default is None.
        """
        super().__init__(
            config=config,
            chunk_idx=chunk_idx,
            dtype=dtype,
            jit_trace=jit_trace,
            norm_class=norm_class,
        )


class InternLM2ModelChunk(ModelChunk):
    """InternLM2 Model Chunk class.

    This class extends the ModelChunk class for the InternLM2 model.
    """

    def __init__(
        self,
        config: InternLM2Config,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        jit_trace=False,
        decoder_class=InternLM2DecoderLayer,
        norm_class=None,
        parallel_lora=False,
        distribute_layers=True,
        use_single_bmm_attention=False,
    ):
        """Initializes the InternLM2ModelChunk class.

        Args:
            config (BaseLLMConfig): Configuration for the model chunk.
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
            decoder_class=decoder_class,
            norm_class=norm_class,
            parallel_lora=parallel_lora,
            distribute_layers=distribute_layers,
            use_single_bmm_attention=use_single_bmm_attention,
        )

    # FIXME: Change to 3.0 weight loading style
    def load_weights_2_x(self, state_dict, state_dict_start_idx, quant_config=None):
        """Loads weights into the model chunk.

        Args:
            state_dict (dict): State dictionary containing the weights.
            state_dict_start_idx (int): Starting index for the state dictionary.
            quant_config (str, optional): Path to the quantization configuration file. Default is None.

        Returns:
            self: The model chunk with loaded weights.
        """
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        expected_subkey = (
            f'layers.{state_dict_start_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
        )
        state_dict_keys = list(state_dict.keys())
        temp_key = None
        input_norm_subkey = None
        post_attention_norm_subkey = None
        parallel_lora_key = None
        for key in state_dict_keys:
            if expected_subkey in key and not key.startswith('tail.'):
                temp_key = key
            if (
                f'layers.{state_dict_start_idx}' in key
                and 'norm' in key
                and 'attention' in key
                and '_weight_quantizer' not in key
                and '_act_quantizer' not in key
            ):
                input_norm_subkey = key.split('.')[-2]
            if (
                f'layers.{state_dict_start_idx}' in key
                and 'norm' in key
                and 'ffn' in key
                and '_weight_quantizer' not in key
                and '_act_quantizer' not in key
            ):
                post_attention_norm_subkey = key.split('.')[-2]
            if f'layers.{state_dict_start_idx}' in key and 'lora' in key:
                parallel_lora_key = key
        if temp_key is None:
            logger.error(
                f"Cannot find layer {state_dict_start_idx}'s {self.fc_names['attn']['o']} weight inside state_dict. "
                f'Please ensure {self.fc_names["attn"]["o"]} weight key contains: {expected_subkey}',
                err=KeyError,
            )
        if input_norm_subkey is None:
            logger.error(
                f"Cannot find layer {state_dict_start_idx}'s input norm weight inside state_dict. "
                f'Please ensure input norm weight key contains: layers.{state_dict_start_idx}, norm, '
                'and attention inside the key string.',
                err=KeyError,
            )
        if post_attention_norm_subkey is None:
            logger.error(
                f"Cannot find layer {state_dict_start_idx}'s post attention norm weight inside state_dict."
                f' Please ensure post attention norm weight key contains: layers.{state_dict_start_idx}, norm, and '
                'ffn inside the key string.',
                err=KeyError,
            )
        prefix = temp_key.split(expected_subkey)[0]
        if self.parallel_lora and parallel_lora_key is not None:
            lora_prefix = parallel_lora_key.split(f'layers.{state_dict_start_idx}.')[0]

        outer_layer_idx = state_dict_start_idx
        self.device_list = []
        if self.config.use_stable_embedding and self.chunk_idx == 0:
            temp_state_dict = {
                'embed_layer_norm.weight': state_dict.pop(f'{prefix}embed_layer_norm.weight').to(torch.float32),
                'embed_layer_norm.bias': state_dict.pop(
                    f'{prefix}embed_layer_norm.bias', torch.zeros(self.config.hidden_size, dtype=self.dtype)
                ).to(torch.float32),
            }
        else:
            temp_state_dict = {}

        for inner_layer_idx in range(self.num_layers):
            if (
                state_dict.get(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.weight',
                    None,
                )
                is not None
            ):
                q_weight = k_weight = v_weight = None
                q_bias = k_bias = v_bias = None
                qkv_weight = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.weight'
                ).to(self.dtype)
                qkv_bias = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.bias',
                    torch.zeros(
                        (2 * self.config.num_key_value_heads * self.head_dim)
                        + (self.config.num_attention_heads * self.head_dim),
                        dtype=self.dtype,
                    ),
                ).to(self.dtype)
            else:
                qkv_weight = qkv_bias = None
                if (
                    state_dict.get(
                        f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight',
                        None,
                    )
                    is None
                    or state_dict.get(
                        f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight',
                        None,
                    )
                    is None
                    or state_dict.get(
                        f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight',
                        None,
                    )
                    is None
                ):
                    logger.error(f'QKV or Q, K, V proj weights not found for layer {outer_layer_idx}!', err=KeyError)
                q_weight = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                ).to(self.dtype)
                k_weight = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                ).to(self.dtype)
                v_weight = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                ).to(self.dtype)
                q_bias = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias',
                    torch.zeros(self.config.hidden_size, dtype=self.dtype),
                ).to(self.dtype)
                k_bias = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias',
                    torch.zeros(self.config.num_key_value_heads * self.head_dim, dtype=self.dtype),
                ).to(self.dtype)
                v_bias = state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias',
                    torch.zeros(self.config.num_key_value_heads * self.head_dim, dtype=self.dtype),
                ).to(self.dtype)

            if qkv_weight is None:
                temp_state_dict = {
                    **temp_state_dict,
                    f'layers.{inner_layer_idx}.self_attn.q_proj.weight': q_weight,
                    f'layers.{inner_layer_idx}.self_attn.k_proj.weight': k_weight,
                    f'layers.{inner_layer_idx}.self_attn.v_proj.weight': v_weight,
                }
                temp_state_dict = {
                    **temp_state_dict,
                    f'layers.{inner_layer_idx}.self_attn.q_proj.bias': q_bias,
                    f'layers.{inner_layer_idx}.self_attn.k_proj.bias': k_bias,
                    f'layers.{inner_layer_idx}.self_attn.v_proj.bias': v_bias,
                }
            else:
                # FIXME split qkv
                temp_state_dict = {
                    **temp_state_dict,
                    f'layers.{inner_layer_idx}.self_attn.wqkv.weight': qkv_weight,
                    f'layers.{inner_layer_idx}.self_attn.wqkv.bias': qkv_bias,
                }

            temp_state_dict = {
                **temp_state_dict,
                f'layers.{inner_layer_idx}.self_attn.o_proj.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.gate_proj.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.weight'
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.up_proj.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.weight'
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.down_proj.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.weight'
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.gate_proj.bias': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.bias',
                    torch.zeros(self.config.intermediate_size, dtype=self.dtype),
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.up_proj.bias': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.bias',
                    torch.zeros(self.config.intermediate_size, dtype=self.dtype),
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.mlp.down_proj.bias': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.bias',
                    torch.zeros(self.config.hidden_size, dtype=self.dtype),
                ).to(self.dtype),
                f'layers.{inner_layer_idx}.input_norm.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.weight'
                ).to(torch.float32),
                f'layers.{inner_layer_idx}.post_attention_norm.weight': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.weight'
                ).to(torch.float32),
                f'layers.{inner_layer_idx}.self_attn.o_proj.bias': state_dict.pop(
                    f'{prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias',
                    torch.zeros(self.config.hidden_size, dtype=self.dtype),
                ).to(self.dtype),
            }

            if self.config.norm == 'LayerNorm':
                temp_state_dict = {
                    **temp_state_dict,
                    f'layers.{inner_layer_idx}.input_norm.bias': state_dict.pop(
                        f'{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.bias',
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ).to(torch.float32),
                    f'layers.{inner_layer_idx}.post_attention_norm.bias': state_dict.pop(
                        f'{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.bias',
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ).to(torch.float32),
                }

            if self.parallel_lora and parallel_lora_key is not None and self.with_lora:
                for module in self.config.lora_modules:
                    if module in [v for k, v in self.config.fc_names['attn'].items() if k != 'name']:
                        temp_state_dict[
                            f'layers.{inner_layer_idx}.attention.{module.replace("proj", "lora")}_A.weight'
                        ] = state_dict.pop(
                            f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.weight'
                        ).to(self.dtype)
                        temp_state_dict[
                            f'layers.{inner_layer_idx}.attention.{module.replace("proj", "lora")}_B.weight'
                        ] = state_dict.pop(
                            f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.weight'
                        ).to(self.dtype)
                    else:
                        temp_state_dict[f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_A.weight'] = (
                            state_dict.pop(
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.weight'
                            ).to(self.dtype)
                        )
                        temp_state_dict[f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_B.weight'] = (
                            state_dict.pop(
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.weight'
                            ).to(self.dtype)
                        )

            num_gpu = torch.cuda.device_count()
            if num_gpu == 0 or self.jit_trace:
                self.device_list.append('cpu')
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
                    device_id = master_gpu_ids[outer_layer_idx]
                else:
                    device_id = 0
                self.device_list.append(f'cuda:{device_id}')
            outer_layer_idx += 1
        if self.include_tail:
            if self.config.tie_word_embeddings:
                lm_head_weight_key = f'{prefix}{self.config.embedding_key}'
                lm_head_bias_key = f'{prefix}{self.config.embedding_key}'.replace('weight', 'bias')
                if state_dict.get(lm_head_weight_key, None) is None:
                    lm_head_weight_key = f'{self.fc_names["tail"]["name"]}.weight'
                    lm_head_bias_key = f'{self.fc_names["tail"]["name"]}.bias'
            else:
                lm_head_weight_key = f'{self.fc_names["tail"]["name"]}.weight'
                lm_head_bias_key = f'{self.fc_names["tail"]["name"]}.bias'
            temp_state_dict = {
                **temp_state_dict,
                'norm.weight': state_dict.pop(f'{prefix}norm.weight').to(torch.float32),
                'lm_head.weight': state_dict.pop(lm_head_weight_key).to(self.dtype),
                'lm_head.bias': state_dict.pop(
                    lm_head_bias_key,
                    torch.zeros(self.config.vocab_size + self.config.lm_head_pad_size, dtype=self.dtype),
                ).to(self.dtype),
            }
            if self.config.norm == 'LayerNorm':
                temp_state_dict['norm.bias'] = state_dict.pop(
                    f'{prefix}norm.bias', torch.zeros(self.config.hidden_size, dtype=self.dtype)
                ).to(torch.float32)

        if quant_config is not None:
            print(f'Quantizing chunk {self.chunk_idx} using quant config: {quant_config}')
            quant_config_chunk = int(quant_config.rsplit('_', 1)[-1].split('.json')[0])
            if quant_config_chunk != self.chunk_idx:
                logger.error(
                    f'chunk_ifx={self.chunk_idx} but quant config used {quant_config} is for chunk {quant_config_chunk}'
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
                    temp_state_dict = self._add_quantizer(
                        state_dict,
                        temp_state_dict,
                        wgt['target_name'],
                        prefix,
                        input_norm_subkey,
                        post_attention_norm_subkey,
                        'weight',
                        wgt['quantizer']['type'],
                    )
                act_targets = quant_config_dict['quantizer_targets']['activations']
                for act in act_targets:
                    temp_state_dict = self._add_quantizer(
                        state_dict,
                        temp_state_dict,
                        act['target_name'],
                        prefix,
                        input_norm_subkey,
                        post_attention_norm_subkey,
                        'activation',
                        act['quantizer']['type'],
                    )

        if temp_state_dict.keys() != self.state_dict().keys():
            temp_state_dict_only_keys = [x for x in temp_state_dict if x not in self.state_dict()]
            model_only_keys = [x for x in self.state_dict() if x not in temp_state_dict and 'lora' not in x]
            if self.parallel_lora:
                model_only_keys = [
                    x for x in model_only_keys if '_weight_quantizer' not in x and '_act_quantizer' not in x
                ]
            if model_only_keys != [] or temp_state_dict_only_keys != []:
                logger.error(
                    f"model state dict keys don't match with state_dict to load into model.\n"
                    f'Model only keys:{model_only_keys}\nstate_dict only keys:{temp_state_dict_only_keys}'
                )
        self.load_state_dict(temp_state_dict, strict=False)
        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        if self.config.use_stable_embedding and self.chunk_idx == 0:
            self.embed_layer_norm.to(self.device_list[0])
        if self.include_tail:
            self.norm.to(self.device_list[-1])
            self.lm_head.to(self.device_list[-1])

        if self.support_quant_stub:
            self.inputs_embeds.to(self.device_list[0])
            self.mask.to(self.device_list[0])
            self.pos_emb.to(self.device_list[0])
            if self.num_layers == 1:
                self.past_keys.to(self.device_list[0])
                self.past_values.to(self.device_list[0])
            else:
                for i in range(self.num_layers):
                    self.cache[i].to(self.device_list[i])
                    self.cache[self.num_layers + i].to(self.device_list[i])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict

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
            # Keep qkv merge for InternLM2
            """
            merged_state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_qkv_weight': f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.weight',
                    f'{outer_layer_idx}_qkv_bias': f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.bias',
                    f'{outer_layer_idx}_gu_weight': f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.'
                    f'{self.fc_names["mlp"]["gateup"]}.weight',
                    f'{outer_layer_idx}_gu_bias': f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.'
                    f'{self.fc_names["mlp"]["gateup"]}.bias',
                }
            )
            """
            # fmt: off
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_qkv_weight': {
                        f'layers.{inner_layer_idx}.self_attn.qkv_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.weight'
                    },
                    f'{outer_layer_idx}_qkv_bias': {
                        f'layers.{inner_layer_idx}.self_attn.qkv_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_g_weight': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.weight'
                    },
                    f'{outer_layer_idx}_g_bias': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.bias'
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.weight'
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.bias'
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.weight'
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.bias'
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                    },
                }
            )
            if self.config.use_qk_norm:
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_q_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.q_norm.weight':
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["query"]}.weight'
                        },
                        f'{outer_layer_idx}_k_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.k_norm.weight':
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["key"]}.weight'
                        },
                    }
                )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias':
                            f'layers.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias':
                            f'layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
                        },
                    }
                )
            # fmt: on
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping()[0])

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

    # Modify from mls_models.modeling_base.BaseModelChunks.load_weights
    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter BaseModelChunk load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)
        state_dict_end_idx = state_dict_start_idx + self.num_layers
        logger.debug(f'state_dict_start_idx={state_dict_start_idx}, state_dict_end_idx={state_dict_end_idx}')
        self.device_list = []
        self.prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        # Check if merged weights/biases exist. Split them if exist.
        for internal_key, external_key in self.merged_state_dict_mapping.items():
            found_key = None
            for pre in self.prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(f'Found {internal_key} merged weight/bias using prefix')
                    found_key = key_to_test
                    break

            if found_key is None:
                for k in state_dict_keys:
                    if k.endswith(external_key):
                        logger.debug(
                            f'Found {internal_key} merged weight/bias using iteration. Adding prefix: '
                            f'{k[: -len(external_key)]}'
                        )
                        self.prefixes.append(k[: -len(external_key)])
                        found_key = k
                        break

            if found_key is not None:
                layer_id = int(internal_key.split('_')[0])
                if '_qkv_weight' in internal_key:
                    # Split QKV weight
                    logger.debug(f'Splitting layer {layer_id} {internal_key} merged weight into Q/K/V weights')
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
                elif '_qkv_bias' in internal_key:
                    # Split QKV bias
                    logger.debug(f'Splitting layer {layer_id} {internal_key} merged bias into Q/K/V biases')
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
                elif '_gu_weight' in internal_key:
                    # Split GateUp weight
                    logger.debug(f'Splitting layer {layer_id} {internal_key} merged weight into Gate/Up weights')
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
                    assert '_gu_bias' in internal_key
                    # Split GateUp bias
                    logger.debug(f'Splitting layer {layer_id} {internal_key} merged bias into Gate/Up biases')
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
            # print(f'internal_key={internal_key}, mapping_dict={mapping_dict}')
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
                    'lm_head_weight': self.config.embedding_key,
                    'lm_head_bias': self.config.embedding_key.replace('weight', 'bias'),
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
                    f'Model only keys:{model_only_keys}\nstate_dict only keys:{weights_to_load_only_keys}\n'
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
        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            self.embed_layer_norm.to(self.device_list[0])
        if self.include_tail:
            self.norm.to(self.device_list[-1])
            self.lm_head.to(self.device_list[-1])
        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict
