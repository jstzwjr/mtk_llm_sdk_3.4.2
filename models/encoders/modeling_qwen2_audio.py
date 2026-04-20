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

from ...utils import logger, utils
from ..activations import TorchGelu
from ..modeling_base import BaseEncoderModelChunk
from ..norm import LayerNorm
from .configuration_qwen2_audio import Qwen2AudioEncoderConfig
from .modeling_whisper import WhisperEncoderAttention, WhisperEncoderLayer, WhisperMLP


class Qwen2AudioEncoderMLP(WhisperMLP):
    """Qwen2AudioEncoderMLP MLP class.

    This class extends the WhisperMLP class for the Qwen2AudioEncoderMLP.
    """

    def __init__(self, config: Qwen2AudioEncoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initializes the MLP class.

        Args:
            config (Qwen2AudioEncoderConfig): Configuration for the MLP.
            lora (LoRA): LoRA object.
            layer_idx (int): Index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
        """
        super().__init__(config, lora, layer_idx)


class Qwen2AudioEncoderAttention(WhisperEncoderAttention):
    """Qwen2Audio Encoder Attention class.

    This class extends the WhisperEncoderAttention mechanism for the Qwen2Audio Encoder model.

    """

    def __init__(self, config: Qwen2AudioEncoderConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
        """Initializes the Qwen2AudioEncoderAttention class.

        Args:
            config (Qwen2AudioEncoderConfig): The configuration object containing model parameters.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.

        Raises:
            ValueError: If hidden_size is not divisible by num_heads.
        """
        super().__init__(config, lora, layer_idx, jit_trace, parallel_lora)

    def forward(self, hidden_states, attn_mask, *lora_inputs):
        """Performs the forward pass to compute the attention output.

        Input shape: Batch x Max Source Position x Hidden Size.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): Mask for attention module.
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
        attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = self.matmul2(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)

        if self.lora is not None and self.config.fc_names['attn']['o'] in self.lora.target_modules and self.with_lora:
            attn_output2 = self.o_proj(attn_output)
            if self.parallel_lora:
                o_lora = self.o_lora_B(self.o_lora_A(self.o_lora_dropout(attn_output)))
            else:
                o_lora = self.o_lora_B(
                    self.o_lora_A(attn_output, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
            attn_output = self.o_lora_add(attn_output2, o_lora)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2AudioEncoderLayer(WhisperEncoderLayer):
    """Qwen2Audio Encoder Layer class.

    This class extends the single Whisper Encoder Layer for the Qwen2Audio model.

    """

    def __init__(
        self,
        config: Qwen2AudioEncoderConfig,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=Qwen2AudioEncoderAttention,
        mlp_class=Qwen2AudioEncoderMLP,
        norm_class=LayerNorm,
        parallel_lora=False,
    ):
        """Initializes the Qwen2AudioEncoderLayer class.

        Args:
            config (Qwen2AudioEncoderConfig): The configuration for the Qwen2Audio Encoder model.
            lora (LoRA): LoRA object.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            layer_idx (int, optional): Index of the layer. Default is None.
            attn_class (type, optional): Class for the attention mechanism. Default is Qwen2AudioEncoderAttention.
            mlp_class (type, optional): Class for the MLP mechanism. Default is Qwen2AudioEncoderMLP.
            norm_class (type, optional): Class for the normalization. Default is LayerNorm.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Default is False.
            exclude_input_norm (bool, optional): Whether to exclude input normalization. Default is False.
        """
        super().__init__(config, lora, jit_trace, layer_idx, attn_class, mlp_class, norm_class, parallel_lora)

    def forward(self, hidden_states, attn_mask, *lora_inputs):
        """Performs the forward pass to compute the encoder layer output.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attn_mask (torch.Tensor): Mask for attention module.
            lora_inputs (tuple): Tuple of lora inputs.

        Returns:
            torch.Tensor: The encoder layer output.
        """
        residual = hidden_states

        hidden_states = self.input_norm(hidden_states)
        attn_outputs = self.self_attn(hidden_states, attn_mask, *lora_inputs[: self.expected_attn_lora_inputs])
        hidden_states = self.add(residual, attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, *lora_inputs[-self.expected_mlp_lora_inputs :])

        return self.add2(residual, hidden_states)


class Qwen2AudioEncoderChunk(BaseEncoderModelChunk):
    """Qwen2AudioEncoder chunk class for the model.

    This class implements a chunk of the model, which includes multiple encoder layers and optional tail components.

    Attributes:
        config (object): The configuration object.
        layers (nn.ModuleList): List of encoder layers.
        first_layer_idx (int): The chunk index.
        conv1 (nn.Conv1d): Conv layer for Qwen2Audio encoder.
        conv2 (nn.Conv1d): Conv layer for Qwen2Audio encoder.
        gelu1 (nn.Module): Activation for Qwen2Audio encoder.
        gelu2 (nn.Module): Activation for Qwen2Audio encoder.
        embed_positions (nn.Module): position embeddings.
        norm (nn.Module): Normalization layer.
    """

    def __init__(
        self,
        config: Qwen2AudioEncoderConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        encoder_class=Qwen2AudioEncoderLayer,
        norm_class=LayerNorm,
        parallel_lora=False,
        distribute_layers=True,
        **kwargs,
    ):
        """Initializes the Qwen2AudioEncoderModelChunk class.

        Args:
            config (Qwen2AudioEncoderConfig): Configuration for the model chunk.
            lora (LoRA): LoRA object.
            num_layers (int): Number of encoder layers in the chunk.
            first_layer_idx (int): The index of the first encoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            encoder_class (type, optional): Class for the encoder layer. Default is Qwen2AudioEncoderLayer.
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
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
            self.norm = norm_class(config.hidden_size, eps=config.norm_eps).float()

        self.expected_num_pos_emb_inputs = 0
        if not self.parallel_lora and self.with_lora:
            self.expected_num_lora_inputs_per_layer = 0 if self.lora is None else len(self.lora.target_modules) * 2
            self.expected_num_lora_inputs = self.expected_num_lora_inputs_per_layer * num_layers
        else:
            self.expected_num_lora_inputs_per_layer = 0
            self.expected_num_lora_inputs = 0
        # qwen2-audio uses mask
        self.expected_num_inputs = 2 + self.expected_num_pos_emb_inputs + self.expected_num_lora_inputs

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
        attn_mask = inputs[1]
        lora_inputs = inputs[2:] if self.expected_num_lora_inputs > 0 else []
        if self.chunk_idx == 0:
            inputs_embeds = self.gelu1(self.conv1(inputs_embeds))
            inputs_embeds = self.gelu2(self.conv2(inputs_embeds)).permute(0, 2, 1)
            embed_pos = self.embed_positions.weight

            inputs_embeds = inputs_embeds + embed_pos.reshape((1, embed_pos.shape[0], embed_pos.shape[1]))
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_outputs = encoder_layer(
                hidden_states.to(self.device_list[idx]),
                attn_mask.to(self.device_list[idx]),
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
            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states = self.avg_pooler(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 1)
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)
            return hidden_states
        return hidden_states

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        if self.chunk_idx == 0:
            example_inputs = [
                torch.randn(
                    1, self.config.num_mel_bins, self.config.max_source_positions * 2, device='cpu', dtype=torch.float32
                ),
                torch.randn(1, 1, self.config.max_source_positions, device='cpu', dtype=torch.float32),
            ]
        else:
            example_inputs = [
                torch.randn(
                    1, self.config.max_source_positions, self.config.hidden_size, device='cpu', dtype=torch.float32
                ),
                torch.randn(1, 1, self.config.max_source_positions, device='cpu', dtype=torch.float32),
            ]
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

        return (*example_inputs, *lora_inputs)

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
            input_shapes = [
                [1, self.config.num_mel_bins, self.config.max_source_positions * 2],
                [1, 1, self.config.max_source_positions],
                *lora_shapes,
            ]
        else:
            input_shapes = [
                [1, self.config.max_source_positions, self.config.hidden_size],
                [1, 1, self.config.max_source_positions],
                *lora_shapes,
            ]
        input_value_ranges = [
            minmax,
            (-20.0, 0.0),
            *[None for _ in range(len(lora_shapes))],
        ]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        return_data = [
                            np.random.rand(1, self.config.num_mel_bins, self.config.max_source_positions * 2).astype(
                                np.float32
                            ),
                            np.random.rand(1, 1, self.config.max_source_positions).astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, self.config.max_source_positions, self.config.hidden_size).astype(
                                np.float32
                            ),
                            np.random.rand(1, 1, self.config.max_source_positions).astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
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
                    return_data = [data['hidden_states'].astype(np.float32), data['audio_attn_mask'].astype(np.float32)]
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
                            np.random.rand(1, 1, self.config.max_source_positions).astype(np.float32),
                            *lora_inputs[calib_lora_map[i]],
                        ]
                    else:
                        return_data = [
                            np.random.rand(1, self.config.max_source_positions, self.config.hidden_size).astype(
                                np.float32
                            ),
                            np.random.rand(1, 1, self.config.max_source_positions).astype(np.float32),
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
                    return_data = [data['hidden_states'].astype(np.float32), data['audio_attn_mask'].astype(np.float32)]
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
                'conv1.weight': {'conv1.weight': 'audio_tower.conv1.weight'},
                'conv1.bias': {'conv1.bias': 'audio_tower.conv1.bias'},
                'conv2.weight': {'conv2.weight': 'audio_tower.conv2.weight'},
                'conv2.bias': {'conv2.bias': 'audio_tower.conv2.bias'},
                'embed_positions.weight': {'embed_positions.weight': 'audio_tower.embed_positions.weight'},
            }

        # fmt: off
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.weight'
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["mlp"]["up"]}.bias'
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.weight'
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias':
                        f'audio_tower.layers.{outer_layer_idx}.{self.fc_names["mlp"]["down"]}.bias'
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight':
                        f'audio_tower.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                    },
                }
            )

            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias':
                            f'audio_tower.layers.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias':
                            f'audio_tower.layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
                        },
                    }
                )
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())

        if self.first_layer_idx + self.num_layers == self.config.num_hidden_layers:
            state_dict_mapping.update(
                {
                    'final_norm_weight': {'norm.weight': 'audio_tower.layer_norm.weight'},
                    'final_norm_bias': {'norm.bias': 'audio_tower.layer_norm.bias'},
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

        if self.support_quant_stub:
            for i in range(len(self.stubs)):
                self.stubs[i].to(self.device_list[0])

        if self.parallel_lora:
            self.train()
        else:
            self.eval()

        return self, state_dict
