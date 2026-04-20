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
"""PyTorch Tail with EAGLE heads."""

import json
import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger, utils
from ...utils.const import DEFAULT_JIT_TRACE_CACHE_SIZE, DEFAULT_JIT_TRACE_NUM_TOKEN
from .modeling_common import Tail

np.random.seed(42)


class EagleTail(Tail):
    """EagleTail class with EAGLE heads.

    This class extends the Tail class to include EAGLE heads for the model.

    Attributes:
        num_layers (int): Number of layers in the model.
        layers (nn.ModuleList): List of decoder layers.
        fc (nn.Linear): Fully connected layer.
        cat (mtk_quantization.pytorch.functional.Cat): Concatenation layer.
    """

    def __init__(
        self,
        config,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        norm_class=None,
        distribute_layers=True,
        decoder_layer=None,
    ):
        """Initializes the EagleTail class.

        Args:
            config (BaseConfig): Configuration for the model.
            chunk_idx (int): Index of the chunk.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            norm_class (type, optional): Class for the normalization. Default is None.
            distribute_layers (bool, optional): Whether to distribute layers across devices. Default is True.
            decoder_layer (type, optional): Class for the decoder layer. Default is None.

        Raises:
            RuntimeError: If decoder_layer is not provided.
        """
        if decoder_layer is None:
            logger.error('decoder_layer must be provided for EagleTail class')
        super().__init__(config, chunk_idx, dtype, jit_trace, norm_class, distribute_layers)
        self.first_layer_idx = 0
        self.num_layers = self.config.num_hidden_layers

        self.layers = nn.ModuleList(
            [
                decoder_layer(config, jit_trace, layer_idx=chunk_idx, exclude_input_norm=True)
                for _ in range(self.num_layers)
            ]
        )

        self.fc = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size)
        self.cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

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
        super()._generate_default_state_dict_mapping()
        state_dict_mapping = self.state_dict_mapping
        merged_state_dict_mapping = {}

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

        self.state_dict_mapping = state_dict_mapping
        self.merged_state_dict_mapping = merged_state_dict_mapping

    def forward(self, hidden_states):
        """Forward pass for the EagleTail model.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        if self.support_quant_stub:
            if self.distribute_layers:
                hidden_states = self.hidden_states(hidden_states).to(self.device_list[0])
            else:
                hidden_states = self.hidden_states(hidden_states)
        else:
            if self.distribute_layers:
                hidden_states = hidden_states.to(self.device_list[0])

        if self.jit_trace:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)

        return self.lm_head(hidden_states)

    def forward_alt(self, inputs_embeds, hidden_states, mask, pos_emb, past_key, past_value):
        """Alternative forward pass for the EagleTail model.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings tensor.
            hidden_states (torch.Tensor): Hidden states tensor.
            mask (torch.Tensor): Attention mask.
            pos_emb (torch.Tensor): Position embeddings.
            past_key (torch.Tensor): Past key tensor.
            past_value (torch.Tensor): Past value tensor.

        Returns:
            tuple: Tuple containing the hidden states and next key and value caches.
        """
        hidden_states = self.fc(
            self.cat([inputs_embeds.to(self.device_list[0]), hidden_states.to(self.device_list[0])])
        )

        next_key_cache = []
        next_value_cache = []

        for decoder_layer in self.layers:
            hidden_states, curr_key, curr_value = decoder_layer(
                hidden_states,
                mask.to(self.device_list[0]),
                pos_emb.to(self.device_list[0]),
                past_key.to(self.device_list[0]),
                past_value.to(self.device_list[0]),
            )
            next_key_cache.append(curr_key)
            next_value_cache.append(curr_value)

        return hidden_states, *next_key_cache, *next_value_cache

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        logger.debug('Enter EagleTail load_weights')
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
                    if k.startswith('extra.') and k.endswith(external_key):
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
                    if k.startswith('extra.') and k.endswith(external_key):
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

    def get_jit_trace_inputs_alt(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including hidden states, masks, and past key and value
        tensors.

        Returns:
            tuple: Tuple containing the input tensors for JIT tracing.
        """
        head_dim = int(self.config.hidden_size / self.config.num_attention_heads)

        return (
            torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size, device='cpu', dtype=torch.float32),
            torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size, device='cpu', dtype=torch.float32),
            torch.randn(
                1,
                1,
                DEFAULT_JIT_TRACE_NUM_TOKEN,
                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                device='cpu',
                dtype=torch.float32,
            ),
            torch.randn(1, 2, DEFAULT_JIT_TRACE_NUM_TOKEN, head_dim, device='cpu', dtype=torch.float32),
            torch.randn(
                self.num_layers,
                self.config.num_key_value_heads,
                DEFAULT_JIT_TRACE_CACHE_SIZE,
                head_dim,
                device='cpu',
                dtype=torch.float32,
            ),
            torch.randn(
                self.num_layers,
                self.config.num_key_value_heads,
                DEFAULT_JIT_TRACE_CACHE_SIZE,
                head_dim,
                device='cpu',
                dtype=torch.float32,
            ),
        )

    def get_ptq_inputs_alt(self, args, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        This method generates inputs for PTQ, including hidden states, masks, and past key and value tensors.

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            file_format (str, optional): File format for PTQ. Default is 'tflite'.
            kwargs: Other arguments.

        Returns:
            tuple: Tuple containing input shapes, input value ranges, calibration data generator, and evaluation data
                generator.
        """
        head_dim = int(self.config.hidden_size / self.config.num_attention_heads)

        weight_dir = utils.get_dirname(args.config)
        emb_minmax = utils.extract_emb_minmax(weight_dir)

        rot_emb_minmax = (-1.0, 1.0)

        input_shapes = [
            [None, None, self.config.hidden_size],
            [None, None, self.config.hidden_size],
            [None, 1, None, None],
            [None, 2, None, head_dim],
            [None, self.config.num_key_value_heads, None, head_dim],
            [None, self.config.num_key_value_heads, None, head_dim],
        ]
        input_value_ranges = [
            emb_minmax,
            None,
            (self.config.mask_value, 0.0),
            rot_emb_minmax,
            None,
            None,
        ]

        if args.calibration_dataset is None:
            calib_data_gen = None
        elif args.calibration_dataset == 'fake':

            def calib_data_gen():
                """Generates calibration data from the dataset."""
                for _ in range(10):
                    return_data = [
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        np.random.rand(
                            1,
                            1,
                            DEFAULT_JIT_TRACE_NUM_TOKEN,
                            DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        ).astype(np.float32),
                        np.random.rand(1, 2, DEFAULT_JIT_TRACE_NUM_TOKEN, head_dim).astype(np.float32),
                        np.random.rand(
                            self.num_layers, self.config.num_key_value_heads, DEFAULT_JIT_TRACE_CACHE_SIZE, head_dim
                        ).astype(np.float32),
                        np.random.rand(
                            self.num_layers, self.config.num_key_value_heads, DEFAULT_JIT_TRACE_CACHE_SIZE, head_dim
                        ).astype(np.float32),
                    ]
                    yield return_data
        else:

            def calib_data_gen():
                """Generates calibration data from the dataset."""
                for f in utils.get_sorted_path_list(
                    os.path.join(args.calibration_dataset, f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    return_data = [
                        data['inputs_embeds'].astype(np.float32),
                        data['hidden_states'].astype(np.float32),
                        data['mask'].astype(np.float32),
                        data['pos_emb'].astype(np.float32),
                        data['past_keys'].astype(np.float32),
                        data['past_values'].astype(np.float32),
                    ]
                    yield return_data

        if args.evaluation_dataset is None:
            eval_data_gen = None
        elif args.evaluation_dataset == 'fake':

            def eval_data_gen():
                """Generates evaluation data from the dataset."""
                for _ in range(10):
                    return_data = [
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        np.random.rand(
                            1,
                            1,
                            DEFAULT_JIT_TRACE_NUM_TOKEN,
                            DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        ).astype(np.float32),
                        np.random.rand(1, 2, DEFAULT_JIT_TRACE_NUM_TOKEN, head_dim).astype(np.float32),
                        np.random.rand(
                            self.num_layers, self.config.num_key_value_heads, DEFAULT_JIT_TRACE_CACHE_SIZE, head_dim
                        ).astype(np.float32),
                        np.random.rand(
                            self.num_layers, self.config.num_key_value_heads, DEFAULT_JIT_TRACE_CACHE_SIZE, head_dim
                        ).astype(np.float32),
                    ]
                    yield return_data
        else:

            def eval_data_gen():
                """Generates evaluation data from the dataset."""
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    return_data = [
                        data['inputs_embeds'].astype(np.float32),
                        data['hidden_states'].astype(np.float32),
                        data['mask'].astype(np.float32),
                        data['pos_emb'].astype(np.float32),
                        data['past_keys'].astype(np.float32),
                        data['past_values'].astype(np.float32),
                    ]
                    yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
