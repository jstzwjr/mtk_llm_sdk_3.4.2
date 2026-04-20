# Copyright (C) 2020 MediaTek Inc. All rights reserved.
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
"""Cythonized module containing MediaTek's Ring Buffer Implementation of LLM Attention."""

import math

import mtk_quantization
import mtk_quantization.pytorch
import numpy as np
import torch
from torch import nn

from ...utils import qat_utils
from ..configuration_base import BaseConfig
from ..norm import RMSNorm, VivoInfiniKVNorm

np.random.seed(42)


class Attention(nn.Module):
    """Common LLM attention class."""

    def __init__(
        self, config: BaseConfig, lora, layer_idx, jit_trace=False, parallel_lora=False, use_single_bmm_attention=False
    ):
        """Initialize the MTK Attention.

        Args:
            config (GeckoConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.lora = lora
        self.jit_trace = jit_trace
        self.use_single_bmm_attention = use_single_bmm_attention
        # FIXME: change to get split mask as input rather than config
        self.use_split_mask = config.use_split_mask
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.sparse_attn = config.sparse_attn
        self.layer_idx = layer_idx
        self.parallel_lora = parallel_lora
        self.attn_scale = math.sqrt(self.head_dim)
        self.infini_attention = config.infini_attention
        self.mask_scaling_factor = config.mask_scaling_factors[layer_idx]
        if self.infini_attention:
            self.infini_kv_norm = config.infini_kv_norm
            self.infini_use_combined_mlp = config.infini_use_combined_mlp
            self.infini_use_alternative_calc = config.infini_use_alternative_calc

        if self.sparse_attn and layer_idx not in [0, config.num_hidden_layers - 1]:
            self.num_key_value_heads = self.config.sparse_attn_num_head
            self.num_heads = self.config.sparse_attn_num_head

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        if self.config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps).float()
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps).float()

        if getattr(self.config, 'rope_scaling', None) is not None and self.config.rope_scaling['type'] == 'yarn':

            def _yarn_get_softmax_scale(config, scaling_factor):
                if scaling_factor <= 1 or config.model_type != 'llama':
                    return 1.0
                return 0.1 * math.log(scaling_factor) + 1.0

            attn_factor = self.config.rope_scaling.get('attn_factor', 1)
            additional_softmax_scale = (
                torch.tensor(_yarn_get_softmax_scale(self.config, self.config.rope_scaling['factor'])) * attn_factor
            )
            self.attn_scale = self.attn_scale / (additional_softmax_scale * additional_softmax_scale)

        # Attention modules
        if self.use_split_mask:
            self.add_mask_current = mtk_quantization.pytorch.functional.Add()
            self.add_mask_cache = mtk_quantization.pytorch.functional.Add()

            if self.use_single_bmm_attention:
                self.div = mtk_quantization.pytorch.functional.Div()
            else:
                self.div_current = mtk_quantization.pytorch.functional.Div()
                self.div_cache = mtk_quantization.pytorch.functional.Div()
        else:
            self.add = mtk_quantization.pytorch.functional.Add()
            self.div = mtk_quantization.pytorch.functional.Div()
        self.matmul1 = mtk_quantization.pytorch.functional.Matmul()
        self.matmul2 = mtk_quantization.pytorch.functional.Matmul()
        if not self.use_single_bmm_attention and self.config.ring_buffer:
            self.add2 = mtk_quantization.pytorch.functional.Add()
            self.matmul3 = mtk_quantization.pytorch.functional.Matmul()
            self.matmul4 = mtk_quantization.pytorch.functional.Matmul()
            self.kq_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        else:
            self.cat = mtk_quantization.pytorch.functional.Cat(dim=2)
            self.cat2 = mtk_quantization.pytorch.functional.Cat(dim=2)

            if self.use_split_mask:
                self.kq_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

        # NOTE: sdpa only for qwen
        self.sdpa_layers = getattr(self.config, 'sdpa_layers', [])
        if self.layer_idx in self.sdpa_layers or 'all' in self.sdpa_layers:
            self.sdpa_layers_dummy_scale = getattr(self.config, 'sdpa_layers_dummy_scale', 0.5)
            self.div_dummy1 = mtk_quantization.pytorch.functional.Div()  # dummy mul for compiler optimization
            self.div_dummy2 = mtk_quantization.pytorch.functional.Div()  # dummy mul for compiler optimization

        if self.infini_attention:
            self.infini_epsilon = torch.tensor(1e-4)
            self.infini_betas = nn.Parameter(torch.randn(1, self.num_heads, 1, 1), requires_grad=False)
            self.infini_q_z_bmm = mtk_quantization.pytorch.functional.Matmul()
            self.infini_q_mem_bmm = mtk_quantization.pytorch.functional.Matmul()
            self.infini_mem_div = mtk_quantization.pytorch.functional.Div()
            self.infini_weighted_add = mtk_quantization.pytorch.functional.Add()
            self.infini_mem_mul = mtk_quantization.pytorch.functional.Mul()
            self.infini_curr_mul = mtk_quantization.pytorch.functional.Mul()
            self.infini_mask_add = mtk_quantization.pytorch.functional.Add()
            self.infini_mask_normal_mul = mtk_quantization.pytorch.functional.Mul()
            self.infini_mask_activate_mul = mtk_quantization.pytorch.functional.Mul()

            if self.infini_kv_norm:
                self.infini_norm_1 = VivoInfiniKVNorm()
                self.infini_norm_2 = VivoInfiniKVNorm()
            if self.infini_use_combined_mlp:
                self.infini_mem_up_proj = nn.Linear(self.head_dim, self.head_dim // 2)
                self.infini_mem_act_func = nn.ReLU()
                self.infini_mem_down_proj = nn.Linear(self.head_dim // 2, self.head_dim)

        # Rotary embedding modules
        self.q_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.q_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.q_add = mtk_quantization.pytorch.functional.Add()
        self.q_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        self.k_mul1 = mtk_quantization.pytorch.functional.Mul()
        self.k_mul2 = mtk_quantization.pytorch.functional.Mul()
        self.k_add = mtk_quantization.pytorch.functional.Add()
        self.k_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        if self.config.extra_input['sink_rope']:
            self.pk_mul1 = mtk_quantization.pytorch.functional.Mul()
            self.pk_mul2 = mtk_quantization.pytorch.functional.Mul()
            self.pk_add = mtk_quantization.pytorch.functional.Add()
            self.pk_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

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
                            self.q_lora_B = nn.Linear(self.lora.rank, self.num_heads * self.head_dim, bias=False).apply(
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

        if self.mask_scaling_factor != 1:
            self.mask_scaling_mul = mtk_quantization.pytorch.functional.Mul()
            if self.use_split_mask:
                self.mask_cache_scaling_mul = mtk_quantization.pytorch.functional.Mul()

        # Extra outputs
        self.get_attn_logits = self.config.extra_output.get('attn_logits')
        self.get_attn_weights = self.config.extra_output.get('attn_weights')

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding to query and key states."""
        q1, q2 = torch.split(q, self.head_dim // 2, dim=-1)
        q_rotated = self.q_cat((-q2, q1))
        k1, k2 = torch.split(k, self.head_dim // 2, dim=-1)
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, cos), self.k_mul2(k_rotated, sin))
        return q_embed, k_embed

    def apply_sink_rotary_pos_emb_mtk(self, q, k, pk, cos, sin, k_cos, k_sin, pk_cos, pk_sin):
        """Apply sink rotary positional embedding to query and key states."""
        q1, q2 = torch.split(q, self.head_dim // 2, dim=-1)
        q_rotated = self.q_cat((-q2, q1))

        k1, k2 = torch.split(k, self.head_dim // 2, dim=-1)
        k_rotated = self.k_cat((-k2, k1))

        pk1, pk2 = torch.split(pk, self.head_dim // 2, dim=-1)
        pk_rotated = self.pk_cat((-pk2, pk1))

        q_embed = self.q_add(self.q_mul1(q, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k, k_cos), self.k_mul2(k_rotated, k_sin))
        pk_embed = self.pk_add(self.pk_mul1(pk, pk_cos), self.pk_mul2(pk_rotated, pk_sin))

        return q_embed, k_embed, pk_embed

    def apply_k_norm(self, k):
        """Apply k norm."""
        if self.config.use_qk_norm:
            if self.jit_trace:
                k = self.k_norm(k)
            else:
                dtype = k.dtype
                k = self.k_norm(k.to(torch.float32)).to(dtype)
        return k

    def apply_q_norm(self, q):
        """Apply q norm."""
        if self.config.use_qk_norm:
            if self.jit_trace:
                q = self.q_norm(q)
            else:
                dtype = q.dtype
                q = self.q_norm(q.to(torch.float32)).to(dtype)
        return q

    def repeat_kv(self, hidden_states, batch, q_len, n_rep):
        """Repeat key/value caches for GQA/MQA models."""
        hidden_states = hidden_states.repeat(1, 1, n_rep, 1)
        return hidden_states.view(batch, self.num_heads, q_len, self.head_dim)

    def forward(self, hidden_states, *maybe_infini_and_lora_inputs, **inputs):
        """Forward pass for the attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            maybe_infini_and_lora_inputs (tuple): The infini attention and LoRA inputs.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                (k_cos) (torch.Tensor): Cosine rotational embedding for sink_rope.
                (k_sin) (torch.Tensor): Sine rotational embedding for sink_rope.
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

        if self.infini_attention:
            mem = maybe_infini_and_lora_inputs[lora_idx]
            z = maybe_infini_and_lora_inputs[lora_idx + 1]
            infini_mask = maybe_infini_and_lora_inputs[lora_idx + 2]
            lora_idx += 3

        if self.use_split_mask:
            mask_current = mask
            mask_cache = maybe_infini_and_lora_inputs[lora_idx]
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
                    self.q_lora_A(hidden_states, maybe_infini_and_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_infini_and_lora_inputs[lora_idx + 1].transpose(1, 2),
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
                    self.k_lora_A(hidden_states, maybe_infini_and_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_infini_and_lora_inputs[lora_idx + 1].transpose(1, 2),
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
                    self.v_lora_A(hidden_states, maybe_infini_and_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_infini_and_lora_inputs[lora_idx + 1].transpose(1, 2),
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
            key_states = self.apply_k_norm(key_states)
            key_states_out_cc = key_states

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
            query_states = self.apply_q_norm(query_states)
        else:
            query_states, key_states = self.apply_rotary_pos_emb_mtk(query_states, key_states, cos, sin)

        if self.use_single_bmm_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            if self.config.extra_input['sink_rope']:
                key_states_out_cc = key_states_out_cc.transpose(1, 2)
                past_key = past_key.transpose(1, 2)

        # for infini attention, retrieve memory
        if self.infini_attention:
            sigma_q = nn.functional.relu(query_states) + 1.0
            denominator = self.infini_q_z_bmm(sigma_q, z.transpose(-2, -1).repeat(1, self.num_key_value_groups, 1, 1))
            if self.infini_use_alternative_calc:
                scaled_sigma_q = self.infini_mem_div(
                    sigma_q, torch.maximum(denominator, self.infini_epsilon.to(hidden_states.device))
                )
                att_mem = self.infini_q_mem_bmm(scaled_sigma_q, mem.repeat(1, self.num_key_value_groups, 1, 1))
            else:
                # the max op may not be useful due to scale difference after quantized
                # FIXME: need to discuss with converter to prevent div by zero
                denominator = torch.maximum(denominator, self.infini_epsilon.to(hidden_states.device))

                att_mem = self.infini_mem_div(
                    self.infini_q_mem_bmm(sigma_q, mem.repeat(1, self.num_key_value_groups, 1, 1)), denominator
                )

            if self.infini_kv_norm:
                if self.jit_trace:
                    key_states = self.infini_norm_1(key_states)
                    value_states = self.infini_norm_2(value_states)
                else:
                    dtype = key_states.dtype
                    key_states = self.infini_norm_1(key_states.to(torch.float32)).to(dtype)
                    value_states = self.infini_norm_2(value_states.to(torch.float32)).to(dtype)

        if self.config.extra_input['sink_rope']:
            key_states_out = key_states_out_cc
        else:
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

                if self.use_split_mask:
                    attn_weight_bmm = attn_weights[..., :c_len]
                    attn_weight_partial_sum = attn_weights[..., c_len:]
                    attn_weight_bmm = self.add_mask_cache(attn_weight_bmm, mask_cache)
                    attn_weight_partial_sum = self.add_mask_current(attn_weight_partial_sum, mask_current)
                    attn_weights = self.kq_cat((attn_weight_bmm, attn_weight_partial_sum))

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

        # infini attention, do weighted add here
        if self.infini_attention:
            if self.infini_use_combined_mlp:
                att_mem = self.infini_mem_up_proj(att_mem)
                att_mem = self.infini_mem_act_func(att_mem)
                att_mem = self.infini_mem_down_proj(att_mem)

            infini_attn_output = self.infini_weighted_add(
                self.infini_mem_mul(torch.sigmoid(self.infini_betas), att_mem),
                self.infini_curr_mul(1.0 - torch.sigmoid(self.infini_betas), attn_output),
            )

            attn_output = self.infini_mask_add(
                self.infini_mask_normal_mul(infini_mask, attn_output),
                self.infini_mask_activate_mul((1 - infini_mask), infini_attn_output),
            )

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
                    self.o_lora_A(attn_output, maybe_infini_and_lora_inputs[lora_idx].transpose(1, 2)),
                    maybe_infini_and_lora_inputs[lora_idx + 1].transpose(1, 2),
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


class WhisperAttention(nn.Module):
    """Whisper Decoder attention class."""

    def __init__(
        self, config: BaseConfig, lora, layer_idx, jit_trace=False, parallel_lora=False, use_single_bmm_attention=False
    ):
        """TODO."""
        super().__init__()
        self.config = config
        self.lora = lora
        self.jit_trace = jit_trace
        self.use_single_bmm_attention = use_single_bmm_attention
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.sparse_attn = config.sparse_attn
        self.layer_idx = layer_idx
        self.parallel_lora = parallel_lora
        self.attn_scale = self.head_dim**-0.5

        if self.sparse_attn and layer_idx not in [0, config.num_hidden_layers - 1]:
            self.num_key_value_heads = self.config.sparse_attn_num_head
            self.num_heads = self.config.sparse_attn_num_head

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        if self.config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps).float()
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps).float()

        # Attention modules
        self.add = mtk_quantization.pytorch.functional.Add()
        self.div = mtk_quantization.pytorch.functional.Div()
        self.q_proj_mul = mtk_quantization.pytorch.functional.Mul()
        self.matmul1 = mtk_quantization.pytorch.functional.Matmul()
        self.matmul2 = mtk_quantization.pytorch.functional.Matmul()
        if not self.use_single_bmm_attention and self.config.ring_buffer:
            self.add2 = mtk_quantization.pytorch.functional.Add()
            self.matmul3 = mtk_quantization.pytorch.functional.Matmul()
            self.matmul4 = mtk_quantization.pytorch.functional.Matmul()
            self.kq_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        else:
            self.cat = mtk_quantization.pytorch.functional.Cat(dim=2)
            self.cat2 = mtk_quantization.pytorch.functional.Cat(dim=2)

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

    def repeat_kv(self, hidden_states, batch, q_len, n_rep):
        """Repeat key/value caches for GQA/MQA models."""
        hidden_states = hidden_states.repeat(1, 1, n_rep, 1)
        return hidden_states.view(batch, self.num_heads, q_len, self.head_dim)

    def forward(self, hidden_states, mask, past_key, past_value, *lora_inputs):
        """Forward function."""
        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]
        lora_idx = 0

        if self.lora is not None and self.config.fc_names['attn']['q'] in self.lora.target_modules and self.with_lora:
            query_states = self.q_proj_mul(self.q_proj(hidden_states), self.attn_scale)
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
                self.q_proj_mul(self.q_proj(hidden_states), self.attn_scale)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
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
                self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
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
                self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            )

        if self.config.use_qk_norm:
            if self.jit_trace:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
            else:
                dtype = query_states.dtype
                query_states = self.q_norm(query_states.to(torch.float32)).to(dtype)
                key_states = self.k_norm(key_states.to(torch.float32)).to(dtype)

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

        else:
            key_states = self.cat([past_key, key_states])
            value_states = self.cat2([past_value, value_states])
            if self.num_key_value_groups > 1:
                key_states = self.repeat_kv(key_states, bsz, q_len + c_len, self.num_key_value_groups)
                value_states = self.repeat_kv(value_states, bsz, q_len + c_len, self.num_key_value_groups)
            attn_weights = self.matmul1(query_states, key_states.transpose(2, 3))

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
                    self.o_lora_A(attn_output, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states_out
