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
"""PyTorch Gemma & Gemma2 & Gemma3 model."""

import math

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger
from ..activations import FastGelu
from ..norm import GemmaRMSNorm
from .attention import Attention
from .configuration_gemma import GemmaConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class GemmaMLP(MLP):
    """Gemma MLP class.

    This class extends the MLP class for the Gemma model.
    """

    def __init__(self, config: GemmaConfig, lora, layer_idx, **kwargs):
        """Initializes the GemmaMLP class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)

        del self.mul2
        self.gelu = FastGelu()

    def forward(self, x, *lora_inputs):
        """Forward pass for the GemmaMLP class.

        Args:
            x (torch.Tensor): Input tensor.
            *lora_inputs: Additional inputs for LoRA.

        Returns:
            torch.Tensor: Output tensor after applying the MLP.
        """
        lora_idx = 0
        if self.lora is not None and self.config.fc_names['mlp']['gate'] in self.lora.target_modules and self.with_lora:
            gate2 = self.gate_proj(x)
            if self.parallel_lora:
                gate_lora = self.gate_lora_B(self.gate_lora_A(self.gate_lora_dropout(x)))
                if self.lora.scale != 1.0:
                    gate_lora = self.gate_lora_scale_mul(gate_lora, self.lora.scale)
            else:
                gate_lora = self.gate_lora_B(
                    self.gate_lora_A(x, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
                )
                lora_idx += 2
            gate = self.gate_lora_add(gate2, gate_lora)
        else:
            gate = self.gate_proj(x)

        if self.lora is not None and self.config.fc_names['mlp']['up'] in self.lora.target_modules and self.with_lora:
            up2 = self.up_proj(x)
            if self.parallel_lora:
                up_lora = self.up_lora_B(self.up_lora_A(self.up_lora_dropout(x)))
                if self.lora.scale != 1.0:
                    up_lora = self.up_lora_scale_mul(up_lora, self.lora.scale)
            else:
                up_lora = self.up_lora_B(
                    self.up_lora_A(x, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
                )
                lora_idx += 2
            up = self.up_lora_add(up2, up_lora)
        else:
            up = self.up_proj(x)

        pre_down = self.mul(self.gelu(gate), up)
        if self.lora is not None and self.config.fc_names['mlp']['down'] in self.lora.target_modules and self.with_lora:
            down2 = self.down_proj(pre_down)
            if self.parallel_lora:
                down_lora = self.down_lora_B(self.down_lora_A(self.down_lora_dropout(pre_down)))
                if self.lora.scale != 1.0:
                    down_lora = self.down_lora_scale_mul(down_lora, self.lora.scale)
            else:
                down_lora = self.down_lora_B(
                    self.down_lora_A(pre_down, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
                )
            down = self.down_lora_add(down2, down_lora)
        else:
            down = self.down_proj(pre_down)

        return down


class GemmaAttention(Attention):
    """Gemma Attention class.

    This class extends the Attention class for the Gemma model.
    """

    def __init__(self, config: GemmaConfig, lora, layer_idx, **kwargs):
        """Initializes the GemmaAttention class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class Gemma2Attention(Attention):
    """Gemma2 Attention class.

    This class extends the Attention class for the Gemma2 model.
    """

    def __init__(self, config: GemmaConfig, lora, layer_idx, **kwargs):
        """Initializes the Gemma2Attention class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)
        # NOTE: Change it to non reciprocal so that div can be used instead of mul
        self.query_pre_attn_scalar = torch.tensor(self.config.query_pre_attn_scalar**0.5)

        self.tanh_mul = mtk_quantization.pytorch.functional.Mul()
        self.tanh_div = mtk_quantization.pytorch.functional.Div()

    def scaled_tanh(self, hidden_states, scale):
        """Apply scaled tanh activation.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor.
            scale (float, optional): The scaling factor. Defaults to 50.0.

        Returns:
            torch.Tensor: The tensor with scaled tanh activation applied.
        """
        return self.tanh_mul(scale, torch.tanh(self.tanh_div(hidden_states, scale)))

    def forward(self, hidden_states, *maybe_lora_inputs, **inputs):
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            maybe_lora_inputs (tuple): The LoRA inputs.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                past_key (torch.Tensor): Past key tensor.
                past_value (torch.Tensor): Past value tensor.

        Returns:
            tuple: The attention output, key states, and value states.
        """
        mask = inputs.get('mask')
        cos = inputs.get('cos')
        sin = inputs.get('sin')
        past_key = inputs.get('past_key')
        past_value = inputs.get('past_value')

        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]
        lora_idx = 0

        if self.use_split_mask:
            mask_current = mask
            mask_cache = maybe_lora_inputs[lora_idx]
            lora_idx += 1

        if self.lora is not None and self.config.fc_names['attn']['q'] in self.lora.target_modules and self.with_lora:
            query_states = self.q_proj(hidden_states)
            if self.parallel_lora:
                q_lora = self.q_lora_B(self.q_lora_A(self.q_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    q_lora = self.q_lora_scale_mul(q_lora, self.lora.scale)
            else:
                q_lora = self.q_lora_B(
                    self.q_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            query_states = self.q_lora_add(query_states, q_lora).view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

        if self.lora is not None and self.config.fc_names['attn']['k'] in self.lora.target_modules and self.with_lora:
            key_states = self.k_proj(hidden_states)
            if self.parallel_lora:
                k_lora = self.k_lora_B(self.k_lora_A(self.k_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    k_lora = self.k_lora_scale_mul(k_lora, self.lora.scale)
            else:
                k_lora = self.k_lora_B(
                    self.k_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            key_states = self.k_lora_add(key_states, k_lora).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        else:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.lora is not None and self.config.fc_names['attn']['v'] in self.lora.target_modules and self.with_lora:
            value_states = self.v_proj(hidden_states)
            if self.parallel_lora:
                v_lora = self.v_lora_B(self.v_lora_A(self.v_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    v_lora = self.v_lora_scale_mul(v_lora, self.lora.scale)
            else:
                v_lora = self.v_lora_B(
                    self.v_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            value_states = (
                self.v_lora_add(value_states, v_lora)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            value_states = (
                self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            )

        if not self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        query_states, key_states = self.apply_rotary_pos_emb_mtk(query_states, key_states, cos, sin)

        if self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.apply_q_norm(query_states)
            key_states = self.apply_k_norm(key_states)

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
                # NOTE: gemma2 scales with query pre attn scalar instead of attn scale before adding mask
                attn_weight_bmm = self.div_cache(attn_weight_bmm, self.query_pre_attn_scalar)
                attn_weight_partial_sum = self.div_current(attn_weight_partial_sum, self.query_pre_attn_scalar)

                attn_weight_bmm = self.scaled_tanh(attn_weight_bmm, scale=self.config.attn_logit_softcapping)
                attn_weight_partial_sum = self.scaled_tanh(
                    attn_weight_partial_sum, scale=self.config.attn_logit_softcapping
                )

                attn_weight_bmm = self.add_mask_cache(attn_weight_bmm, mask_cache)
                attn_weight_partial_sum = self.add_mask_current(attn_weight_partial_sum, mask_current)

                attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
            else:
                attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
                # NOTE: gemma2 scales with query pre attn scalar instead of attn scale
                attn_weights = self.div(attn_weights, self.query_pre_attn_scalar)
        else:
            key_states = self.cat([past_key, key_states])
            value_states = self.cat2([past_value, value_states])

            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len + c_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len + c_len, self.num_key_value_groups)

            attn_weights = self.matmul1(query_states, key_states.transpose(2, 3))
            attn_weights = self.div(attn_weights, self.query_pre_attn_scalar)

        if not self.use_split_mask:
            # NOTE: Unique to gemma 2
            attn_weights = self.scaled_tanh(attn_weights, scale=self.config.attn_logit_softcapping)
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
                    self.o_lora_A(attn_output, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states_out


class Gemma3Attention(Attention):
    """Gemma3 Attention class.

    This class extends the Attention class for the Gemma3 model.
    """

    def __init__(self, config: GemmaConfig, lora, layer_idx, **kwargs):
        """Initializes the Gemma3Attention class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)
        # NOTE: Change it to non reciprocal so that div can be used instead of mul
        self.query_pre_attn_scalar = torch.tensor(self.config.query_pre_attn_scalar**0.5)
        norm_class = kwargs.pop('norm_class', GemmaRMSNorm)
        if self.config.use_qk_norm:
            self.q_norm = norm_class(self.head_dim, eps=config.norm_eps).float()
            self.k_norm = norm_class(self.head_dim, eps=config.norm_eps).float()

    def forward(self, hidden_states, *maybe_lora_inputs, **inputs):
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            maybe_lora_inputs (tuple): The LoRA inputs.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                past_key (torch.Tensor): Past key tensor.
                past_value (torch.Tensor): Past value tensor.

        Returns:
            tuple: The attention output, key states, and value states.
        """
        mask = inputs.get('mask')
        cos = inputs.get('cos')
        sin = inputs.get('sin')
        past_key = inputs.get('past_key')
        past_value = inputs.get('past_value')

        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]
        lora_idx = 0

        if self.lora is not None and self.config.fc_names['attn']['q'] in self.lora.target_modules and self.with_lora:
            query_states = self.q_proj(hidden_states)
            if self.parallel_lora:
                q_lora = self.q_lora_B(self.q_lora_A(self.q_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    q_lora = self.q_lora_scale_mul(q_lora, self.lora.scale)
            else:
                q_lora = self.q_lora_B(
                    self.q_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            query_states = self.q_lora_add(query_states, q_lora).view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

        if self.lora is not None and self.config.fc_names['attn']['k'] in self.lora.target_modules and self.with_lora:
            key_states = self.k_proj(hidden_states)
            if self.parallel_lora:
                k_lora = self.k_lora_B(self.k_lora_A(self.k_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    k_lora = self.k_lora_scale_mul(k_lora, self.lora.scale)
            else:
                k_lora = self.k_lora_B(
                    self.k_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            key_states = self.k_lora_add(key_states, k_lora).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        else:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.lora is not None and self.config.fc_names['attn']['v'] in self.lora.target_modules and self.with_lora:
            value_states = self.v_proj(hidden_states)
            if self.parallel_lora:
                v_lora = self.v_lora_B(self.v_lora_A(self.v_lora_dropout(hidden_states)))
                if self.lora.scale != 1.0:
                    v_lora = self.v_lora_scale_mul(v_lora, self.lora.scale)
            else:
                v_lora = self.v_lora_B(
                    self.v_lora_A(hidden_states, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            value_states = (
                self.v_lora_add(value_states, v_lora)
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            value_states = (
                self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            )

        if not self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        if self.config.use_qk_norm:
            # NOTE: gemma3 applies qk norm before applying rot emb
            query_states = self.apply_q_norm(query_states)
            key_states = self.apply_k_norm(key_states)

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

            attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
            attn_weights = self.div(attn_weights, self.query_pre_attn_scalar)
        else:
            key_states = self.cat([past_key, key_states])
            value_states = self.cat2([past_value, value_states])
            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len + c_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len + c_len, self.num_key_value_groups)
            attn_weights = self.matmul1(query_states, key_states.transpose(2, 3))
            attn_weights = self.div(attn_weights, self.query_pre_attn_scalar)

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
                    self.o_lora_A(attn_output, maybe_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states_out


class GemmaDecoderLayer(DecoderLayer):
    """Gemma Decoder Layer class.

    This class extends the DecoderLayer class for the Gemma model.
    """

    def __init__(self, config: GemmaConfig, lora, **kwargs):
        """Initializes the GemmaDecoderLayer class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=GemmaAttention, mlp_class=GemmaMLP, **kwargs)


class Gemma2DecoderLayer(DecoderLayer):
    """Gemma2 Decoder Layer class.

    This class extends the DecoderLayer class for the Gemma2 model.
    """

    def __init__(self, config: GemmaConfig, lora, **kwargs):
        """Initializes the Gemma2DecoderLayer class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=Gemma2Attention, mlp_class=GemmaMLP, **kwargs)
        norm_class = kwargs.pop('norm_class', GemmaRMSNorm)
        self.pre_feedforward_layernorm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.post_feedforward_layernorm = norm_class(config.hidden_size, eps=config.norm_eps).float()

    def forward(self, hidden_states, *maybe_lora_inputs, **inputs):
        """Forward pass for the decoder layer.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.
            maybe_lora_inputs (Tuple): Additional inputs for LoRA.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                past_key (torch.Tensor): Past key tensor.
                past_value (torch.Tensor): Past value tensor.

        Returns:
            tuple: Tuple containing the hidden states, present key, and present value.
        """
        residual = hidden_states
        if not self.exclude_input_norm:
            if self.jit_trace:
                hidden_states = self.input_norm(hidden_states)
            else:
                dtype = hidden_states.dtype
                hidden_states = self.input_norm(hidden_states.to(torch.float32)).to(dtype)

        maybe_lora_inputs_slicing = self.expected_attn_lora_inputs
        if self.use_split_mask:
            maybe_lora_inputs_slicing += 1

        attn_outputs = self.self_attn(hidden_states, *maybe_lora_inputs[:maybe_lora_inputs_slicing], **inputs)

        attn_output, present_key, present_value = attn_outputs

        if self.jit_trace:
            hidden_states = self.post_attention_norm(attn_output)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(attn_output.to(torch.float32)).to(dtype)

        hidden_states = self.add(residual, hidden_states)

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.pre_feedforward_layernorm(hidden_states.to(torch.float32)).to(dtype)
        hidden_states = self.mlp(hidden_states, *maybe_lora_inputs[maybe_lora_inputs_slicing:])
        if self.jit_trace:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_feedforward_layernorm(hidden_states.to(torch.float32)).to(dtype)
        hidden_states = self.add2(residual, hidden_states)

        return hidden_states, present_key, present_value


class Gemma3DecoderLayer(DecoderLayer):
    """Gemma3 Decoder Layer class.

    This class extends the DecoderLayer class for the Gemma2 model.
    """

    def __init__(self, config: GemmaConfig, lora, **kwargs):
        """Initializes the Gemma3DecoderLayer class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=Gemma3Attention, mlp_class=GemmaMLP, **kwargs)
        norm_class = kwargs.pop('norm_class', GemmaRMSNorm)
        self.use_res_clamp = config.use_res_clamp
        self.pre_feedforward_layernorm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.post_feedforward_layernorm = norm_class(config.hidden_size, eps=config.norm_eps).float()

    def forward(self, hidden_states, *maybe_lora_inputs, **inputs):
        """Forward pass for the decoder layer.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.
            maybe_lora_inputs (Tuple): Additional inputs for LoRA.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                past_key (torch.Tensor): Past key tensor.
                past_value (torch.Tensor): Past value tensor.

        Returns:
            tuple: Tuple containing the hidden states, present key, and present value.
        """
        residual = hidden_states
        if not self.exclude_input_norm:
            if self.jit_trace:
                hidden_states = self.input_norm(hidden_states)
            else:
                dtype = hidden_states.dtype
                hidden_states = self.input_norm(hidden_states.to(torch.float32)).to(dtype)

        maybe_lora_inputs_slicing = self.expected_attn_lora_inputs

        attn_outputs = self.self_attn(hidden_states, *maybe_lora_inputs[:maybe_lora_inputs_slicing], **inputs)

        attn_output, present_key, present_value = attn_outputs

        if self.jit_trace:
            hidden_states = self.post_attention_norm(attn_output)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(attn_output.to(torch.float32)).to(dtype)
        # account for res add overflow
        if self.use_res_clamp:
            hidden_states = self.add(residual, hidden_states).clamp(
                torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
            )
        else:
            hidden_states = self.add(residual, hidden_states)

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.pre_feedforward_layernorm(hidden_states.to(torch.float32)).to(dtype)

        hidden_states = self.mlp(hidden_states, *maybe_lora_inputs[maybe_lora_inputs_slicing:])

        if self.jit_trace:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_feedforward_layernorm(hidden_states.to(torch.float32)).to(dtype)
        # account for res add overflow
        if self.use_res_clamp:
            hidden_states = self.add2(residual, hidden_states).clamp(
                torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
            )
        else:
            hidden_states = self.add2(residual, hidden_states)

        return hidden_states, present_key, present_value


class GemmaTail(Tail):
    """Gemma Tail class.

    This class extends the Tail class for the Gemma model.
    """

    def __init__(self, config: GemmaConfig, chunk_idx, **kwargs):
        """Initializes the GemmaTail class.

        Args:
            config: A GemmaConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class Gemma2Tail(Tail):
    """Gemma2 Tail class.

    This class extends the Tail class for the Gemma2 model.
    """

    def __init__(self, config: GemmaConfig, chunk_idx, **kwargs):
        """Initializes the Gemma2Tail class.

        Args:
            config: A GemmaConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, norm_class=GemmaRMSNorm, **kwargs)

        self.tanh_mul = mtk_quantization.pytorch.functional.Mul()
        self.tanh_div = mtk_quantization.pytorch.functional.Div()

    def scaled_tanh(self, hidden_states, scale):
        """Applies a scaled tanh activation function.

        Args:
            hidden_states (torch.Tensor): Input tensor.
            scale (float): Scaling factor.

        Returns:
            torch.Tensor: Scaled tanh output.
        """
        return self.tanh_mul(scale, torch.tanh(self.tanh_div(hidden_states, scale)))

    def forward(
        self,
        hidden_states,  # (b, t, 4096)
    ):
        """Forward pass for the Gemma2Tail class.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after applying the tail layers and scaled tanh activation.
        """
        logits = super().forward(hidden_states)
        return self.scaled_tanh(logits, self.config.final_logit_softcapping)


class Gemma3Tail(Tail):
    """Gemma3 Tail class.

    This class extends the Tail class for the Gemma3 model.
    """

    def __init__(self, config: GemmaConfig, chunk_idx, **kwargs):
        """Initializes the Gemma3Tail class.

        Args:
            config: A GemmaConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, norm_class=GemmaRMSNorm, chunk_idx=chunk_idx, **kwargs)


class GemmaModelChunk(ModelChunk):
    """Gemma Model Chunk class.

    This class extends the ModelChunk class for the Gemma model.
    """

    def __init__(self, config: GemmaConfig, lora, **kwargs):
        """Initializes the GemmaModelChunk class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=GemmaDecoderLayer, **kwargs)

        self.mul = mtk_quantization.pytorch.functional.Mul()

    def forward(self, inputs_embeds, mask, pos_emb, *cache_and_lora):
        """Forward pass for the GemmaModelChunk class.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings tensor of shape (batch_size, sequence_length, hidden_size).
            mask (torch.Tensor): Attention mask tensor.
            pos_emb (torch.Tensor): Positional embeddings tensor.
            *cache_and_lora: Additional inputs for cache and LoRA.

        Returns:
            tuple: A tuple containing:
                - hidden_states (torch.Tensor): Output tensor of shape (batch_size, sequence_length, hidden_size).
                - next_key_cache (list): List of next key cache tensors.
                - next_value_cache (list): List of next value cache tensors.
        """
        batch_size = inputs_embeds.size()[0]
        if not self.jit_trace:
            assert len(cache_and_lora) == self.expected_num_cache_inputs + self.expected_num_lora_inputs, (
                f'Expected {self.expected_num_cache_inputs + self.expected_num_lora_inputs} number of cache+lora '
                f'inputs, but got {len(cache_and_lora)}'
            )

            cache = cache_and_lora[: self.expected_num_cache_inputs]
            lora_inputs = cache_and_lora[self.expected_num_cache_inputs :]

            assert len(cache) == self.expected_num_cache_inputs
            assert len(lora_inputs) == self.expected_num_lora_inputs
            if self.num_blocks == 1:
                past_keys, past_values = cache
                assert past_keys.shape[0] == self.num_blocks * batch_size, (
                    f'key cache wrong first dim: {past_keys.shape[0]} != {self.num_blocks * batch_size}'
                )
                assert past_values.shape[0] == self.num_blocks * batch_size, (
                    f'value cache wrong first dim: {past_values.shape[0]} != {self.num_blocks * batch_size}'
                )
        else:
            cache = cache_and_lora[: self.expected_num_cache_inputs]
            lora_inputs = cache_and_lora[self.expected_num_cache_inputs :]
            if self.num_blocks == 1:
                past_keys, past_values = cache

        if self.support_quant_stub:
            if self.distribute_layers:
                inputs_embeds = self.inputs_embeds(inputs_embeds).to(self.device_list[0])
            else:
                inputs_embeds = self.inputs_embeds(inputs_embeds)
            mask = self.mask(mask)
            pos_emb = self.pos_emb(pos_emb)
            if self.num_blocks == 1:
                past_keys = self.past_keys(past_keys)
                past_values = self.past_values(past_values)
            else:
                cache = [self.cache[i](cache[i]) for i in range(len(self.cache))]

        else:
            if self.distribute_layers:
                inputs_embeds = inputs_embeds.to(self.device_list[0])

        if self.config.use_stable_embedding and self.chunk_idx == 0:
            if self.jit_trace:
                inputs_embeds = self.embed_layer_norm(inputs_embeds)
            else:
                inputs_embeds = self.embed_layer_norm(inputs_embeds.to(torch.float32)).to(self.dtype)

        if self.chunk_idx == 0:
            inputs_embeds = self.mul(inputs_embeds, math.sqrt(self.config.hidden_size))

        hidden_states = inputs_embeds

        if self.num_blocks == 1:
            if self.distribute_layers:
                decoder_outputs = self.layers[0](
                    hidden_states.to(self.device_list[0]),
                    mask.to(self.device_list[0]),
                    pos_emb.to(self.device_list[0]),
                    past_keys.to(self.device_list[0]),
                    past_values.to(self.device_list[0]),
                    *lora_inputs[: len(self.config.lora_modules) * 2],
                )
            else:
                decoder_outputs = self.layers[0](
                    hidden_states,
                    mask,
                    pos_emb,
                    past_keys,
                    past_values,
                    *lora_inputs[: len(self.config.lora_modules) * 2],
                )
            hidden_states = decoder_outputs[0]
            next_key_cache = [decoder_outputs[1].to(inputs_embeds.device)]
            next_value_cache = [decoder_outputs[2].to(inputs_embeds.device)]
        else:
            next_key_cache = []
            next_value_cache = []
            # decoder layers
            for idx, decoder_layer in enumerate(self.layers):
                if self.distribute_layers:
                    decoder_outputs = decoder_layer(
                        hidden_states.to(self.device_list[idx]),
                        mask.to(self.device_list[idx]),
                        pos_emb.to(self.device_list[idx]),
                        cache[idx].to(self.device_list[idx]),
                        cache[self.num_blocks + idx].to(self.device_list[idx]),
                        *lora_inputs[
                            idx * len(self.config.lora_modules) * 2 : (idx + 1) * len(self.config.lora_modules) * 2
                        ],
                    )
                else:
                    decoder_outputs = decoder_layer(
                        hidden_states,
                        mask,
                        pos_emb,
                        cache[idx],
                        cache[self.num_blocks + idx],
                        *lora_inputs[
                            idx * len(self.config.lora_modules) * 2 : (idx + 1) * len(self.config.lora_modules) * 2
                        ],
                    )
                hidden_states = decoder_outputs[0]
                next_key_cache.append(decoder_outputs[1].to(inputs_embeds.device))
                next_value_cache.append(decoder_outputs[2].to(inputs_embeds.device))

        if self.include_tail:
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states, *next_key_cache, *next_value_cache


class Gemma2ModelChunk(ModelChunk):
    """Gemma2 Model Chunk class.

    This class extends the ModelChunk class for the Gemma2 model.
    """

    def __init__(self, config: GemmaConfig, lora, **kwargs):
        """Initializes the Gemma2ModelChunk class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=Gemma2DecoderLayer, norm_class=GemmaRMSNorm, **kwargs)

        self.mul = mtk_quantization.pytorch.functional.Mul()

        if self.include_tail:
            self.tanh_mul = mtk_quantization.pytorch.functional.Mul()
            self.tanh_div = mtk_quantization.pytorch.functional.Div()

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
            # fmt: off
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
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
                    f'{outer_layer_idx}_pre_mlp_weight': {
                        f'layers.{inner_layer_idx}.pre_feedforward_layernorm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["pre_mlp"]}.weight'
                    },
                    f'{outer_layer_idx}_post_mlp_norm_weight': {
                        f'layers.{inner_layer_idx}.post_feedforward_layernorm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["post_mlp"]}.weight'
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

    def scaled_tanh(self, hidden_states, scale):
        """Applies a scaled tanh activation function.

        Args:
            hidden_states (torch.Tensor): Input tensor.
            scale (float): Scaling factor.

        Returns:
            torch.Tensor: Scaled tanh output.
        """
        return self.tanh_mul(scale, torch.tanh(self.tanh_div(hidden_states, scale)))

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
        decoder_inputs = {'mask': inputs[1], 'cos': inputs[2], 'sin': inputs[3]}

        i = 4
        cache = inputs[i : i + self.expected_num_cache_inputs]
        i += self.expected_num_cache_inputs

        if self.use_split_mask:
            split_mask = [inputs[i]]
            i += 1
        else:
            split_mask = []

        lora_inputs = inputs[i:] if self.expected_num_lora_inputs > 0 else []

        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            if self.jit_trace:
                inputs_embeds = self.embed_layer_norm(inputs_embeds)
            else:
                inputs_embeds = self.embed_layer_norm(inputs_embeds.to(torch.float32)).to(self.dtype)

        if self.chunk_idx == 0:
            inputs_embeds = self.mul(inputs_embeds, math.sqrt(self.config.hidden_size))
        hidden_states = inputs_embeds

        cache_outputs = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = list(
                cache[
                    idx * self.expected_num_cache_inputs_per_layer : (idx + 1)
                    * self.expected_num_cache_inputs_per_layer
                ]
            )
            decoder_inputs['past_key'] = past_key_value[0]
            decoder_inputs['past_value'] = past_key_value[1]
            decoder_inputs = {k: v.to(self.device_list[idx]) for k, v in decoder_inputs.items()}

            decoder_outputs = decoder_layer(
                hidden_states.to(self.device_list[idx]),
                *[x.to(self.device_list[idx]) for x in split_mask],
                *[
                    x.to(self.device_list[idx])
                    for x in lora_inputs[
                        sum(self.expected_num_lora_inputs_per_layer[:idx]) : sum(
                            self.expected_num_lora_inputs_per_layer[: idx + 1]
                        )
                    ]
                ],
                **decoder_inputs,
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
            hidden_states = self.scaled_tanh(hidden_states, self.config.final_logit_softcapping)
        return hidden_states, *cache_outputs


class Gemma3ModelChunk(ModelChunk):
    """Gemma3 Model Chunk class.

    This class extends the ModelChunk class for the Gemma3 model.
    """

    def __init__(self, config: GemmaConfig, lora, num_layers, first_layer_idx, **kwargs):
        """Initializes the Gemma3ModelChunk class.

        Args:
            config: A GemmaConfig object.
            lora (LoRA): LoRA object.
            num_layers: Num of layers.
            first_layer_idx: first_layer_idx.
            kwargs: All keyword arguments.
        """
        super().__init__(
            config,
            lora,
            num_layers,
            first_layer_idx,
            decoder_class=Gemma3DecoderLayer,
            norm_class=GemmaRMSNorm,
            **kwargs,
        )
        self.mul = mtk_quantization.pytorch.functional.Mul()

    def _pop_base_layer_suffix(self, state_dict):
        for k in list(state_dict.keys()):
            if 'language_model.' in k:
                k_new = k.replace('language_model.', '')
                state_dict[k_new] = state_dict.pop(k)
        return state_dict

    def _pop_base_layer_vision(self, state_dict):
        for k in list(state_dict.keys()):
            if 'vision_tower.' in k:
                state_dict.pop(k)
            if 'multi_modal' in k:
                state_dict.pop(k)
        return state_dict

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        state_dict = self._pop_base_layer_suffix(state_dict=state_dict)
        if self.config.use_vision is False:
            state_dict = self._pop_base_layer_vision(state_dict=state_dict)
        return super().load_weights(state_dict, state_dict_start_idx, quant_config)

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
            # fmt: off
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
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
                    f'{outer_layer_idx}_pre_mlp_weight': {
                        f'layers.{inner_layer_idx}.pre_feedforward_layernorm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["pre_mlp"]}.weight'
                    },
                    f'{outer_layer_idx}_post_mlp_norm_weight': {
                        f'layers.{inner_layer_idx}.post_feedforward_layernorm.weight':
                        f'layers.{outer_layer_idx}.{self.norm_names["post_mlp"]}.weight'
                    },
                    f'{outer_layer_idx}_q_norm_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_norm.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["query"]}.weight'
                    },
                    f'{outer_layer_idx}_k_norm_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_norm.weight':
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["key"]}.weight'
                    }
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
        decoder_inputs = {'mask': inputs[1], 'cos': inputs[2], 'sin': inputs[3]}

        i = 4

        cache = inputs[i : i + self.expected_num_cache_inputs]
        i += self.expected_num_cache_inputs

        lora_inputs = inputs[i:] if self.expected_num_lora_inputs > 0 else []

        if self.config.use_stable_embedding and self.first_layer_idx == 0:
            if self.jit_trace:
                inputs_embeds = self.embed_layer_norm(inputs_embeds)
            else:
                inputs_embeds = self.embed_layer_norm(inputs_embeds.to(torch.float32)).to(self.dtype)

        if self.chunk_idx == 0:
            inputs_embeds = self.mul(inputs_embeds, math.sqrt(self.config.hidden_size))
        hidden_states = inputs_embeds

        cache_outputs = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = list(
                cache[
                    idx * self.expected_num_cache_inputs_per_layer : (idx + 1)
                    * self.expected_num_cache_inputs_per_layer
                ]
            )
            decoder_inputs['past_key'] = past_key_value[0]
            decoder_inputs['past_value'] = past_key_value[1]
            decoder_inputs = {k: v.to(self.device_list[idx]) for k, v in decoder_inputs.items()}

            decoder_outputs = decoder_layer(
                hidden_states.to(self.device_list[idx]),
                *[
                    x.to(self.device_list[idx])
                    for x in lora_inputs[
                        sum(self.expected_num_lora_inputs_per_layer[:idx]) : sum(
                            self.expected_num_lora_inputs_per_layer[: idx + 1]
                        )
                    ]
                ],
                **decoder_inputs,
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
