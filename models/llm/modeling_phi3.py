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
"""PyTorch Phi-mini model."""

import os

import mtk_quantization
import numpy as np
import torch

from ...utils.const import DEFAULT_JIT_TRACE_CACHE_SIZE, DEFAULT_JIT_TRACE_NUM_TOKEN
from ..norm import RMSNorm
from .attention import Attention
from .configuration_phi3 import Phi3Config, Phi4Config
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class Phi3MLP(MLP):
    """Phi3 MLP class.

    This class extends the MLP class for the Phi3-mini model.
    """

    def __init__(self, config: Phi3Config, lora, layer_idx, **kwargs):
        """Initializes the Phi3MLP class.

        Args:
            config: A Phi3Config object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class Phi3Attention(Attention):
    """Phi3 Attention class.

    This class extends the Attention class for the Phi3-mini model.
    """

    def __init__(self, config: Phi3Config, lora, layer_idx, **kwargs):
        """Initializes the Phi3Attention class.

        Args:
            config: A Phi3Config object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class Phi3DecoderLayer(DecoderLayer):
    """Phi3 Decoder Layer class.

    This class extends the DecoderLayer class for the Phi3-mini model.
    """

    def __init__(self, config: Phi3Config, lora, **kwargs):
        """Initializes the Phi3DecoderLayer class.

        Args:
            config: A Phi3Config object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=Phi3Attention, mlp_class=Phi3MLP, **kwargs)


class Phi3Tail(Tail):
    """Phi3 Tail class.

    This class extends the Tail class for the Phi3-mini model.
    """

    def __init__(self, config: Phi3Config, chunk_idx, **kwargs):
        """Initializes the Phi3Tail class.

        Args:
            config: A Phi3Config object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class Phi3ModelChunk(ModelChunk):
    """Phi3 Model Chunk class.

    This class extends the ModelChunk class for the Phi3-mini model.
    """

    def __init__(self, config: Phi3Config, lora, **kwargs):
        """Initializes the Phi3ModelChunk class.

        Args:
            config: A Phi3Config object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=Phi3DecoderLayer, **kwargs)


class Phi4Attention(Attention):
    """Phi4 Attention class.

    This class extends the Attention class for the Phi4-mini model.
    """

    def __init__(
        self,
        config: Phi4Config,
        lora,
        layer_idx,
        **kwargs,
    ):
        """Initializes the Phi4Attention class.

        Args:
            config (Phi4Config): A Phi3Config object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)

        # Additional rotary embedding modules for partial rotary embedding
        self.partial_q_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)
        self.partial_k_cat = mtk_quantization.pytorch.functional.Cat(dim=-1)

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding to query and key states.

        Phi4-mini uses partial rotary embedding, which only rotate some of the q and k.
        """
        rotary_dim = cos.shape[-1]

        # Use split
        q_rot, q_pass = torch.split(q, rotary_dim, dim=-1)
        q1, q2 = torch.split(q_rot, rotary_dim // 2, dim=-1)
        q_rotated = self.q_cat((-q2, q1))

        k_rot, k_pass = torch.split(k, rotary_dim, dim=-1)
        k1, k2 = torch.split(k_rot, rotary_dim // 2, dim=-1)
        k_rotated = self.k_cat((-k2, k1))

        q_embed = self.q_add(self.q_mul1(q_rot, cos), self.q_mul2(q_rotated, sin))
        k_embed = self.k_add(self.k_mul1(k_rot, cos), self.k_mul2(k_rotated, sin))

        q_embed = self.partial_q_cat([q_embed, q_pass])
        k_embed = self.partial_k_cat([k_embed, k_pass])

        return q_embed, k_embed


class Phi4DecoderLayer(DecoderLayer):
    """Phi4 Decoder Layer class.

    This class extends the DecoderLayer class for the Phi4-mini model.
    """

    def __init__(
        self,
        config: Phi4Config,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=Phi4Attention,
        mlp_class=Phi3MLP,
        norm_class=RMSNorm,
        parallel_lora=False,
        exclude_input_norm=False,
        use_single_bmm_attention=False,
    ):
        """Initializes the Phi4DecoderLayer class.

        Attributes:
            hidden_size (int): Size of the hidden layer.
            jit_trace (bool): Whether to use JIT tracing.
            exclude_input_norm (bool): Whether to exclude input normalization.
            self_attn (Attention): Self-attention mechanism.
            mlp (MLP): Multi-Layer Perceptron.
            input_norm (nn.Module): Normalization layer for input.
            post_attention_norm (nn.Module): Normalization layer for post-attention.
            add (mtk_quantization.pytorch.functional.Add): Addition function from mtk_quantization.
            add2 (mtk_quantization.pytorch.functional.Add): Second addition function from mtk_quantization.
            expected_attn_lora_inputs (int): Expected number of LoRA inputs for attention.
            expected_mlp_lora_inputs (int): Expected number of LoRA inputs for MLP.
            use_single_bmm_attention (bool, optional): Use single bmm attention graph. Defaults to False.
        """
        super().__init__(
            config,
            lora,
            jit_trace=jit_trace,
            layer_idx=layer_idx,
            attn_class=attn_class,
            mlp_class=mlp_class,
            norm_class=norm_class,
            parallel_lora=parallel_lora,
            exclude_input_norm=exclude_input_norm,
            use_single_bmm_attention=use_single_bmm_attention,
        )


class Phi4ModelChunk(ModelChunk):
    """Phi4 Model Chunk class.

    This class extends the ModelChunk class for the Phi4-mini model.
    """

    def __init__(
        self,
        config: Phi4Config,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        jit_trace=False,
        decoder_class=Phi4DecoderLayer,
        norm_class=None,
        parallel_lora=False,
        distribute_layers=True,
        use_single_bmm_attention=False,
    ):
        """Initializes the ModelChunk class.

        Args:
            config (Phi4Config): Configuration for the model chunk.
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
            config=config,
            lora=lora,
            num_layers=num_layers,
            first_layer_idx=first_layer_idx,
            chunk_idx=chunk_idx,
            dtype=dtype,
            include_tail=include_tail,
            jit_trace=jit_trace,
            decoder_class=decoder_class,
            norm_class=norm_class,
            parallel_lora=parallel_lora,
            distribute_layers=distribute_layers,
            use_single_bmm_attention=use_single_bmm_attention,
        )

    # To account for Phi4-Omni's both vision and speech lora
    def generate_default_lora_state_dict_mapping(self):
        """Generates default lora state dict mapping for lora handler."""
        if self.lora is None:
            return {}, {}

        state_dict_mapping = {}
        merged_state_dict_mapping = {}

        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            # fmt: off
            for fc_name in self.config.fc_names['attn']:
                if fc_name == 'name':
                    continue
                module = self.config.fc_names['attn'][fc_name]

                suffix = self.config.input_mode + '.' if self.config.input_mode is not None else ''
                lora_prefix = self.config.lora_prefix + '.' if self.config.lora_prefix is not None else ''
                if fc_name == 'qkv':
                    # Remove potential 'base_layer' suffix for LoRA
                    if module.endswith('.base_layer'):
                        module = module.strip('.base_layer')  # noqa: B005

                    merged_state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': (
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.{suffix}weight'
                            ),
                            f'{outer_layer_idx}_{fc_name}_B': (
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.{suffix}weight'
                            ),
                        }
                    )
                else:
                    # Add potential 'base_layer suffix for LoRA
                    if module.endswith('.base_layer'):
                        fc_name = fc_name + '.base_layer'
                    state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': {
                                f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_A.weight': (
                                    f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.{suffix}weight'
                                )
                            },
                            f'{outer_layer_idx}_{fc_name}_B': {
                                f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_B.weight': (
                                    f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.{suffix}weight'
                                )
                            },
                        }
                    )

            for fc_name in self.config.fc_names['mlp']:
                if fc_name == 'name':
                    continue
                module = self.config.fc_names['mlp'][fc_name]

                if fc_name == 'gateup':
                    simplified_fc_name = 'gu'
                    merged_state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{simplified_fc_name}_A': (
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.{suffix}weight'
                            ),
                            f'{outer_layer_idx}_{simplified_fc_name}_B': (
                                f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.{suffix}weight'
                            ),
                        }
                    )
                else:
                    # Remove potential 'base_layer' suffix for LoRA
                    if module.endswith('.base_layer'):
                        module = module.strip('.base_layer')  # noqa: B005
                    if fc_name == 'gate':
                        simplified_fc_name = 'g'
                    elif fc_name == 'up':
                        simplified_fc_name = 'u'
                    else:
                        simplified_fc_name = fc_name
                    state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{simplified_fc_name}_A': {
                                f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_A.weight': (
                                    f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.{suffix}weight'
                                )
                            },
                            f'{outer_layer_idx}_{simplified_fc_name}_B': {
                                f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_B.weight': (
                                    f'{lora_prefix}layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.{suffix}weight'
                                )
                            },
                        }
                    )
            # fmt: on
        return state_dict_mapping, merged_state_dict_mapping

    def _pop_base_layer_suffix(self, state_dict):
        for k in list(state_dict.keys()):
            if '.base_layer' in k:
                k_new = k.replace('.base_layer', '')
                state_dict[k_new] = state_dict.pop(k)
        return state_dict

    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Load model weights.

        Args:
            state_dict: model state_dict.
            state_dict_start_idx (int): The start index of state dict.
            quant_config: QAT quantization config. Defaults to None.
        """
        state_dict = self._pop_base_layer_suffix(state_dict=state_dict)
        return super().load_weights(state_dict, state_dict_start_idx, quant_config)

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        jit_trace_inputs = list(super().get_jit_trace_inputs())
        rot_dim = int(self.head_dim * self.config.partial_rotary_factor)
        pos_emb_inputs = [
            torch.randn(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim, device='cpu', dtype=torch.float32)
            if not self.use_single_bmm_attention
            else torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim, device='cpu', dtype=torch.float32),  # cos
            torch.randn(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim, device='cpu', dtype=torch.float32)
            if not self.use_single_bmm_attention
            else torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim, device='cpu', dtype=torch.float32),  # sin
        ]
        pos_emb_count = 2
        for _ in range(pos_emb_count):
            jit_trace_inputs.pop(2)
        p_idx = 2
        for p in pos_emb_inputs:
            jit_trace_inputs.insert(p_idx, p)
            p_idx += 1
        return jit_trace_inputs

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
        rot_dim = int(self.head_dim * self.config.partial_rotary_factor)
        input_shapes, input_value_ranges, calib_data_gen, eval_data_gen = super().get_ptq_inputs(
            args, exp_name, lora_inputs, calib_lora_map, eval_lora_map, has_encoder
        )
        pos_emb_shapes = [
            [None, 1, None, rot_dim] if not self.use_single_bmm_attention else [None, None, 1, rot_dim],
            [None, 1, None, rot_dim] if not self.use_single_bmm_attention else [None, None, 1, rot_dim],
        ]
        pos_emb_count = 2
        for _ in range(pos_emb_count):
            input_shapes.pop(2)
        p_idx = 2
        for p in pos_emb_shapes:
            input_shapes.insert(p_idx, p)
            p_idx += 1

        if self.config.sparse_attn and self.first_layer_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for i in range(10):
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim).astype(np.float32),
                    ]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                    ]
                    if self.infini_attention:
                        cache_data.extend(
                            [
                                np.random.rand(self.num_layers, num_head, self.head_dim, self.head_dim).astype(
                                    np.float32
                                ),
                                np.random.rand(self.num_layers, num_head, 1, self.head_dim).astype(np.float32),
                            ]
                        )
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

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for i in range(10):
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, rot_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, rot_dim).astype(np.float32),
                    ]
                    cache_data = [
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                        np.random.rand(self.num_layers, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim).astype(
                            np.float32
                        ),
                    ]
                    if self.infini_attention:
                        cache_data.extend(
                            [
                                np.random.rand(self.num_layers, num_head, self.head_dim, self.head_dim).astype(
                                    np.float32
                                ),
                                np.random.rand(self.num_layers, num_head, 1, self.head_dim).astype(np.float32),
                            ]
                        )
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

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
