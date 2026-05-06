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
"""PyTorch QWen model."""

import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger, utils
from ...utils.const import DEFAULT_JIT_TRACE_CACHE_SIZE, DEFAULT_JIT_TRACE_NUM_TOKEN
from .attention import Attention
from .configuration_qwen import QwenConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class QwenMLP(nn.Module):
    """QwenMLP class for the Qwen model.

    Attributes:
        config (QwenConfig): The configuration object containing model parameters.
        layer_idx (int): The index of the layer.
        jit_trace (bool): Whether to use JIT tracing.
        parallel_lora (bool): Whether to use parallel LoRA.
        hidden_size (int): The hidden size of the model.
        lora_rank (int): The rank of the LoRA.
        lora_alpha (float): The alpha value for the LoRA.
        lora_dropout (float): The dropout rate for the LoRA.
        lora_modules (list): The list of LoRA modules.
        ff_dim_in (int): The input dimension for the feed-forward layer.
        w1 (nn.Linear): The first linear layer.
        w2 (nn.Linear): The second linear layer.
        c_proj (nn.Linear): The projection layer.
        mul (mtk_quantization.pytorch.functional.Mul): The first multiplication layer.
        mul2 (mtk_quantization.pytorch.functional.Mul): The second multiplication layer.
        with_lora (bool): Whether to use LoRA.
        w1_lora_A (mtk_quantization.pytorch.functional.Matmul): The first LoRA matrix multiplication for w1.
        w1_lora_B (mtk_quantization.pytorch.functional.Matmul): The second LoRA matrix multiplication for w1.
        w1_lora_add (mtk_quantization.pytorch.functional.Add): The addition layer for w1 LoRA.
        w2_lora_A (mtk_quantization.pytorch.functional.Matmul): The first LoRA matrix multiplication for w2.
        w2_lora_B (mtk_quantization.pytorch.functional.Matmul): The second LoRA matrix multiplication for w2.
        w2_lora_add (mtk_quantization.pytorch.functional.Add): The addition layer for w2 LoRA.
        w3_lora_A (mtk_quantization.pytorch.functional.Matmul): The first LoRA matrix multiplication for w3.
        w3_lora_B (mtk_quantization.pytorch.functional.Matmul): The second LoRA matrix multiplication for w3.
        w3_lora_add (mtk_quantization.pytorch.functional.Add): The addition layer for w3 LoRA.

    Methods:
        __init__(config, layer_idx, jit_trace, parallel_lora): Initialize the QwenMLP.
        forward(x, *lora_inputs): Forward pass of the QwenMLP.
    """

    def __init__(self, config: QwenConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initialize the QwenMLP.

        Args:
            config (QwenConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.lora = lora
        self.jit_trace = jit_trace
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.parallel_lora = parallel_lora

        self.ff_dim_in = config.intermediate_size // 2
        self.w1 = nn.Linear(self.hidden_size, self.ff_dim_in)
        self.w2 = nn.Linear(self.hidden_size, self.ff_dim_in)
        self.c_proj = nn.Linear(self.ff_dim_in, self.hidden_size)
        self.mul = mtk_quantization.pytorch.functional.Mul()
        self.mul2 = mtk_quantization.pytorch.functional.Mul()

        # Lora modules
        self.with_lora = False
        if self.lora is not None:
            self.with_lora = self.layer_idx >= self.lora.start_idx and self.layer_idx <= self.lora.end_idx
            if self.with_lora:
                if self.parallel_lora:
                    logger.error('', err=NotImplementedError)
                for module in self.lora.target_modules:
                    if module == self.config.fc_names['mlp']['gate']:
                        self.w1_lora_A = mtk_quantization.pytorch.functional.Matmul()
                        self.w1_lora_B = mtk_quantization.pytorch.functional.Matmul()
                        self.w1_lora_add = mtk_quantization.pytorch.functional.Add()
                    elif module == self.config.fc_names['mlp']['up']:
                        self.w2_lora_A = mtk_quantization.pytorch.functional.Matmul()
                        self.w2_lora_B = mtk_quantization.pytorch.functional.Matmul()
                        self.w2_lora_add = mtk_quantization.pytorch.functional.Add()
                    elif module == self.config.fc_names['mlp']['down']:
                        self.w3_lora_A = mtk_quantization.pytorch.functional.Matmul()
                        self.w3_lora_B = mtk_quantization.pytorch.functional.Matmul()
                        self.w3_lora_add = mtk_quantization.pytorch.functional.Add()

    def forward(self, x, *lora_inputs):
        """Forward pass of the QwenMLP.

        Args:
            x (torch.Tensor): The input tensor.
            *lora_inputs: Additional inputs for LoRA.

        Returns:
            torch.Tensor: The output tensor.
        """
        lora_idx = 0
        a1 = self.w1(x)
        if self.lora is not None and self.config.fc_names['mlp']['gate'] in self.lora.target_modules and self.with_lora:
            w1_lora = self.w1_lora_B(
                self.w1_lora_A(x, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
            )
            a1 = self.w1_lora_add(a1, w1_lora)
            lora_idx += 2

        a2 = self.w2(x)
        if self.lora is not None and self.config.fc_names['mlp']['up'] in self.lora.target_modules and self.with_lora:
            w2_lora = self.w2_lora_B(
                self.w2_lora_A(x, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
            )
            a2 = self.w2_lora_add(a2, w2_lora)
            lora_idx += 2

        intermediate_parallel = self.mul(self.mul2(a2, torch.sigmoid(a2)), a1)

        if self.lora is not None and self.config.fc_names['mlp']['down'] in self.lora.target_modules and self.with_lora:
            down2 = self.c_proj(intermediate_parallel)
            if self.parallel_lora:
                w3_lora = self.w3_lora_B(self.w3_lora_A(intermediate_parallel))
            else:
                w3_lora = self.w3_lora_B(
                    self.w3_lora_A(intermediate_parallel, lora_inputs[lora_idx].transpose(1, 2).to(x.device)),
                    lora_inputs[lora_idx + 1].transpose(1, 2).to(x.device),
                )
            output = self.w3_lora_add(down2, w3_lora)
        else:
            output = self.c_proj(intermediate_parallel)

        return output


class QwenAttention(Attention):
    """QwenAttention class for the Qwen model.

    Methods:
        __init__(config, layer_idx, jit_trace, parallel_lora): Initialize the QwenAttention.
    """

    def __init__(self, config: QwenConfig, lora, layer_idx, **kwargs):
        """Initializes the QwenAttention class.

        Args:
            config: A QwenConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class Qwen3Attention(Attention):
    """Qwen3 LLM attention class."""

    def __init__(
        self, config: QwenConfig, lora, layer_idx, jit_trace=False, parallel_lora=False, use_single_bmm_attention=False
    ):
        """Initialize the Qwen3 Attention.

        Args:
            config (QwenConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__(config, lora, layer_idx, jit_trace, parallel_lora, use_single_bmm_attention)

    def forward(self, hidden_states, *maybe_lora_inputs, **inputs):
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            maybe_lora_inputs (tuple): The infini attention and LoRA inputs.
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
        if self.config.extra_input['sink_rope']:
            k_cos = inputs.get('k_cos')
            k_sin = inputs.get('k_sin')
        past_key = inputs.get('past_key')
        past_value = inputs.get('past_value')

        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]
        lora_idx = 0

        if self.mask_scaling_factor != 1:
            mask = self.mask_scaling_mul(mask, self.mask_scaling_factor)

        if self.use_split_mask:
            mask_current = mask
            mask_cache = maybe_lora_inputs[lora_idx]
            if self.mask_scaling_factor != 1:
                mask_cache = self.mask_cache_scaling_mul(mask_cache, self.mask_scaling_factor)
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

        if self.config.extra_input['sink_rope']:
            key_states_out_cc = key_states
            # Qwen3 apply qk_norm before applying rope
            key_states = self.apply_k_norm(key_states)
            query_states = self.apply_q_norm(query_states)

            if self.use_single_bmm_attention:
                past_key = past_key.transpose(1, 2)

            query_states, key_states, past_key = self.apply_sink_rotary_pos_emb_mtk(
                query_states,
                key_states,
                past_key,
                cos,
                sin,
                k_cos=k_cos[:, :, c_len:, :] if not self.use_single_bmm_attention else k_cos[:, c_len:, :, :],
                k_sin=k_sin[:, :, c_len:, :] if not self.use_single_bmm_attention else k_sin[:, c_len:, :, :],
                pk_cos=k_cos[:, :, :c_len, :] if not self.use_single_bmm_attention else k_cos[:, :c_len, :, :],
                pk_sin=k_sin[:, :, :c_len, :] if not self.use_single_bmm_attention else k_sin[:, :c_len, :, :],
            )
        else:
            # Qwen3 apply qk_norm before applying rope
            query_states = self.apply_q_norm(query_states)
            key_states = self.apply_k_norm(key_states)
            query_states, key_states = self.apply_rotary_pos_emb_mtk(query_states, key_states, cos, sin)

        if self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            if self.config.extra_input['sink_rope']:
                key_states_out_cc = key_states_out_cc.transpose(1, 2)
                past_key = past_key.transpose(1, 2)

        key_states_out = key_states_out_cc if self.config.extra_input['sink_rope'] else key_states

        value_states_out = value_states
        if not self.use_single_bmm_attention and self.config.ring_buffer:
            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len, self.num_key_value_groups)
                past_key = self.repeat_kv(past_key, bsz, c_len, self.num_key_value_groups)
                past_value = self.repeat_kv(past_value, bsz, c_len, self.num_key_value_groups)

            if self.layer_idx in self.sdpa_layers or 'all' in self.sdpa_layers:
                scaled_query_states = self.div(query_states, self.attn_scale)
                scaled_query_states = self.div_dummy1(scaled_query_states, self.sdpa_layers_dummy_scale)
                attn_weight_bmm = self.matmul1(scaled_query_states, past_key.transpose(2, 3))
                attn_weight_partial_sum = self.matmul2(scaled_query_states, key_states.transpose(2, 3))
                attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))
                attn_weights = self.div_dummy2(attn_weights, 1.0 / self.sdpa_layers_dummy_scale)
            else:
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

            if self.layer_idx in self.sdpa_layers or 'all' in self.sdpa_layers:
                scaled_query_states = self.div(query_states, self.attn_scale)
                scaled_query_states = self.div_dummy1(scaled_query_states, self.sdpa_layers_dummy_scale)
                attn_weights = self.div(
                    self.matmul1(scaled_query_states, key_states.transpose(2, 3)), 1.0 / self.sdpa_layers_dummy_scale
                )
            else:
                attn_weights = self.div(self.matmul1(query_states, key_states.transpose(2, 3)), self.attn_scale)

        if not self.use_split_mask:
            attn_weights = self.add(attn_weights, mask)

        if self.get_attn_logits:
            logits = attn_weights

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

        attn_score = []
        if self.get_attn_logits:
            attn_score.append(logits)
        if self.get_attn_weights:
            if self.config.cache_evict['in_graph']:
                if self.config.cache_evict['method'] == 'LocalSnapKV':
                    attn_weights = torch.max(attn_weights, dim=-1, keepdim=True)[0].mean(dim=1)
                elif self.config.cache_evict['method'] == 'GlobalSnapKV':
                    # in graph operations
                    start = attn_weights.shape[2] - self.config.cache_evict['obs_window']
                    end = attn_weights.shape[2]
                    attn_weights = attn_weights[:, :, start:end, :].mean(dim=-2)
                    attn_weights = attn_weights.view(
                        attn_weights.shape[0], -1, self.num_key_value_groups, attn_weights.shape[-1]
                    )
                    attn_weights = attn_weights.mean(dim=-2)
            attn_score.append(attn_weights)
        if self.get_attn_logits or self.get_attn_weights:
            return attn_output, key_states_out, value_states_out, *attn_score
        return attn_output, key_states_out, value_states_out


class QwenDecoderLayer(DecoderLayer):
    """QwenDecoderLayer class for the Qwen model."""

    def __init__(self, config: QwenConfig, lora, **kwargs):
        """Initializes the QwenDecoderLayer class.

        Args:
            config: A QwenConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=QwenAttention, mlp_class=QwenMLP, **kwargs)


class Qwen3DecoderLayer(DecoderLayer):
    """Qwen3DecoderLayer class for the Qwen model."""

    def __init__(self, config: QwenConfig, lora, **kwargs):
        """Initializes the Qwen3DecoderLayer class.

        Args:
            config: A QwenConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=Qwen3Attention, mlp_class=MLP, **kwargs)


class QwenTail(Tail):
    """QwenTail class for the Qwen model.

    Methods:
        __init__(config, chunk_idx, dtype, jit_trace, norm_class, distribute_layers): Initialize the QwenTail.
        load_weights(state_dict, state_dict_start_idx, quant_config): Load weights into the QwenTail.
    """

    def __init__(self, config: QwenConfig, chunk_idx, **kwargs):
        """Initializes the QwenTail class.

        Args:
            config: A QwenConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class Qwen2Tail(Tail):
    """Qwen2Tail class for the Qwen model."""

    def __init__(self, config: QwenConfig, chunk_idx, **kwargs):
        """Initializes the Qwen2Tail class.

        Args:
            config: A QwenConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class QwenModelChunk(ModelChunk):
    """QwenModelChunk class for the Qwen model."""

    def __init__(self, config: QwenConfig, lora, **kwargs):
        """Initializes the QwenModelChunk class.

        Args:
            config: A QwenConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=QwenDecoderLayer, **kwargs)

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
                    f'{outer_layer_idx}_qkv_weight': f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.weight',
                    f'{outer_layer_idx}_qkv_bias': f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.'
                    f'{self.fc_names["attn"]["qkv"]}.bias',
                    f'{outer_layer_idx}_gu_weight': f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.'
                    f'{self.fc_names["mlp"]["gateup"]}.weight',
                    f'{outer_layer_idx}_gu_bias': f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.'
                    f'{self.fc_names["mlp"]["gateup"]}.bias',
                }
            )
            # fmt: off
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_g_weight': {
                        f'layers.{inner_layer_idx}.mlp.w1.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.weight'
                    },
                    f'{outer_layer_idx}_g_bias': {
                        f'layers.{inner_layer_idx}.mlp.w1.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.bias'
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.w2.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.weight'
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.w2.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.bias'
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.c_proj.weight':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.weight'
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.c_proj.bias':
                        f'h.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.bias'
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight':
                        f'h.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight':
                        f'h.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                    },
                }
            )
            if self.config.use_qk_norm:
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_q_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.q_norm.weight':
                            f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["query"]}.weight'
                        },
                        f'{outer_layer_idx}_k_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.k_norm.weight':
                            f'h.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["key"]}.weight'
                        },
                    }
                )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias':
                            f'h.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias':
                            f'h.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
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

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            tuple: A tuple containing the JIT trace inputs.
        """
        if self.config.sparse_attn and self.first_layer_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        num_attention_heads = self.config.num_attention_heads

        pos_emb_inputs = [
            torch.randn(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim, device='cpu', dtype=torch.float32)
            if not self.use_single_bmm_attention
            else torch.randn(
                1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim, device='cpu', dtype=torch.float32
            ),  # cos
            torch.randn(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim, device='cpu', dtype=torch.float32)
            if not self.use_single_bmm_attention
            else torch.randn(
                1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim, device='cpu', dtype=torch.float32
            ),  # sin
        ]

        cache_inputs = []
        for _ in range(self.num_layers):
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # K cache
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # V cache

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
                                self.config.intermediate_size // 2,
                                self.config.lora_rank,
                                device='cpu',
                                dtype=torch.float32,
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

    def get_ptq_inputs(
        self, args, exp_name, lora_inputs=None, calib_lora_map=None, eval_lora_map=None, has_encoder=False
    ):
        """Gets inputs for post-training quantization (PTQ).

        This method generates inputs for PTQ, including LoRA inputs if applicable.

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            lora_inputs (list, optional): List of number of lora scenarios of list of LoRA inputs per scenario.
                Default is None.
            calib_lora_map (list, optional): List of LoRA scenario mappings for calibration dataset. Default is None.
            eval_lora_map (list, optional): List of LoRA scenario mappings for evaluation dataset. Default is None.
            has_encoder (bool, optional): Boolean to indicate if pipeline has encoder present. Default is False.

        Returns:
            tuple: A tuple containing input shapes, input value ranges, calibration data generator,
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

        if self.config.sparse_attn and self.chunk_idx not in [0, self.config.num_hidden_layers - 1]:
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

        if (
            not args.dummy_weights
            and self.first_layer_idx == 0
            and not has_encoder
            and not args.get_emb_minmax_from_cal_data
        ):
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
                        lora_shapes.append([None, self.config.intermediate_size // 2, None])

        pos_emb_shapes = [
            [None, 1, None, self.head_dim] if not self.use_single_bmm_attention else [None, None, 1, self.head_dim],
            [None, 1, None, self.head_dim] if not self.use_single_bmm_attention else [None, None, 1, self.head_dim],
        ]

        cache_shapes = [
            [None, num_head, None, self.head_dim],
            [None, num_head, None, self.head_dim],
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
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                    ]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                    ]
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
                        data['cos'].astype(np.float32),
                        data['sin'].astype(np.float32),
                    ]
                    return_data.extend(
                        [
                            data['past_keys'].astype(np.float32),
                            data['past_values'].astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    )
                    yield return_data

        if args.evaluation_dataset is None:
            eval_data_gen = None
        elif args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for i in range(10):
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                    ]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                    ]
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
                    yield return_data
        else:

            def eval_data_gen():
                for i, f in enumerate(
                    utils.get_sorted_path_list(
                        os.path.join(args.evaluation_dataset, 'llm', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                    )
                ):
                    data = np.load(f)
                    return_data = [
                        data['inputs_embeds'].astype(np.float32),
                        data['mask'].astype(np.float32),
                        data['cos'].astype(np.float32),
                        data['sin'].astype(np.float32),
                    ]
                    return_data.extend(
                        [
                            data['past_keys'].astype(np.float32),
                            data['past_values'].astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    )
                    yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen


class Qwen2VLAttention(Attention):
    """Class for Qwen2VLAttention.

    This class extends the Attention class to include functionalities specific to Qwen2VL.

    Attributes:
        config (object): The configuration object.
        lora (object): The LoRA object.
        layer_idx (int): The layer index.
        jit_trace (bool): Whether to use JIT tracing.
        parallel_lora (bool): Whether to use parallel LoRA.
        mrope_section (int): The multimodal rotary position embedding section.
        _position_ids (Any): The position IDs.

    Methods:
        position_ids: Property to get the position IDs.
        position_ids: Setter to set the position IDs.
        apply_rotary_pos_emb_mtk: Apply multimodal rotary position embedding.
    """

    def __init__(self, config, lora, layer_idx, jit_trace=False, parallel_lora=False, **kwargs):
        """Initialize the Qwen2VLAttention.

        Args:
            config (object): The configuration object.
            lora (object): The LoRA object.
            layer_idx (int): The layer index.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora, **kwargs)
        self.mrope_section = self.config.rope_scaling['mrope_section']
        self._position_ids = None

    @property
    def position_ids(self):
        """Property to get the position IDs.

        Returns:
            Any: The position IDs.
        """
        return self._position_ids

    @position_ids.setter
    def position_ids(self, _id):
        """Setter to set the position IDs.

        Args:
            _id (Any): The position IDs.
        """
        self._position_ids = _id

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Qwen2VL use multimodal rotary pos emb.

        Explanation:
            Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
            sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
            vision embedding part, we apply rotary position embedding on temporal, height, width dimension seperately.
            Here we split the channel dimension to 3 chunks for the temporal, height, width rotary position embedding.
            For text embedding part, we just apply 1D rotary position embedding. The three rotary position index
            (temporal, height and width) of text embedding is always the same, so the text embedding rotary position
            embedding has no difference with modern LLMs.
        """
        # mrope_section = self.mrope_section * 2

        # Move to utils.get_master_rot_emb to avoid gather op in graph
        # cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        # sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)

        q1, q2 = torch.split(q, self.head_dim // 2, dim=-1)
        q_rotated = self.q_cat((-q2, q1))

        k1, k2 = torch.split(k, self.head_dim // 2, dim=-1)
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed


class Qwen2VLDecoderLayer(DecoderLayer):
    """Class for Qwen2VLDecoderLayer.

    This class extends the DecoderLayer class to include functionalities specific to Qwen2VL.

    Attributes:
        config (object): The configuration object.
        lora (object): The LoRA object.
        kwargs (dict, optional): Additional keyword arguments.

    Methods:
        set_attn_position_id: Set the attention position ID.
    """

    def __init__(self, config, lora, **kwargs):
        """Initialize the Qwen2VLDecoderLayer.

        Args:
            config (object): The configuration object.
            lora (object): The LoRA object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config, lora, attn_class=Qwen2VLAttention, **kwargs)

    def set_attn_position_id(self, position_id):
        """Set the attention position ID.

        Args:
            position_id (Any): The position ID to set.
        """
        self.self_attn.position_id = position_id


class Qwen2ModelChunk(ModelChunk):
    """Qwen2ModelChunk class for the Qwen2 model."""

    def __init__(self, config: QwenConfig, lora, **kwargs):
        """Initializes the Qwen2ModelChunk class.

        Args:
            config: A QwenConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=Qwen2VLDecoderLayer if config.is_vl else DecoderLayer, **kwargs)

    def get_qwen2vl_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        """Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.

            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.

            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.
        """  # noqa: D214
        spatial_merge_size = self.config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        self.position_ids = position_ids
        self.mrope_position_deltas = mrope_position_deltas

        self.model.set_decoder_layer_position_ids(position_ids)
        return position_ids, mrope_position_deltas


class Qwen3ModelChunk(ModelChunk):
    """Qwen3ModelChunk class for the Qwen3 model.

    Patch E: deepstack 注入。pipeline 通过 config._num_deepstack_inject 告知本次 run
    需在 LLM layer [0, N) 注入 deepstack visual embed。本类的 chunk 若覆盖了这些层，
    forward 多接受 1 个 ds_padded 输入（[1, T, D]，已由 pipeline 端 scatter 完成），
    内部仅做 hidden = hidden + ds_padded（DLA-friendly mul+add）。
    """

    def __init__(self, config: QwenConfig, lora, **kwargs):
        """Initializes the Qwen3ModelChunk class.

        Args:
            config: A QwenConfig object. May carry attribute `_num_deepstack_inject` (int).
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=Qwen3DecoderLayer, **kwargs)
        num_ds_inject = int(getattr(config, '_num_deepstack_inject', 0) or 0)
        # 当前每个 chunk 对应 1 层，first_layer_idx 即 LLM 层号
        self._inject_ds = (num_ds_inject > 0) and (self.first_layer_idx < num_ds_inject)
        # 注：不修改 base.expected_num_inputs；forward 在 super 前 strip ds_padded

    def forward(self, *inputs):
        if not self._inject_ds:
            return super().forward(*inputs)
        ds_padded = inputs[-1]
        base_inputs = inputs[:-1]
        outputs = super().forward(*base_inputs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = ()
        # DLA-friendly：直接 add（pipeline 已把 ds_padded scatter 成同 shape，非 image 位置为 0）
        ds_padded = ds_padded.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + ds_padded
        return (hidden_states, *rest)

    def get_jit_trace_inputs(self):
        base = super().get_jit_trace_inputs()
        if not self._inject_ds:
            return base
        # ds_padded 与 input_embeds 同 shape (1, T, hidden_size)
        if isinstance(base, tuple):
            ds_input = torch.zeros_like(base[0])
            return (*base, ds_input)
        ds_input = torch.zeros_like(base)
        return (base, ds_input)

    def get_ptq_inputs(self, *args, **kwargs):
        result = super().get_ptq_inputs(*args, **kwargs)
        if not self._inject_ds:
            return result
        input_shapes, input_value_ranges, calib_data_gen, eval_data_gen = result
        # ds_padded shape == input_embeds shape
        ds_shape = list(input_shapes[0])
        new_input_shapes = list(input_shapes) + [ds_shape]
        new_input_value_ranges = list(input_value_ranges) + [None]

        def _wrap_gen(orig_gen, ds_shape):
            def _new():
                for batch in orig_gen():
                    ds = np.zeros(ds_shape, dtype=np.float32)
                    yield [*batch, ds]
            return _new

        return (
            new_input_shapes,
            new_input_value_ranges,
            _wrap_gen(calib_data_gen, ds_shape),
            _wrap_gen(eval_data_gen, ds_shape),
        )
