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
"""Common backbone across multiple models."""

import os

import mtk_quantization
import numpy as np
import torch
from torch import nn

from ...utils import logger, qat_utils, utils
from ...utils.const import DEFAULT_JIT_TRACE_CACHE_SIZE, DEFAULT_JIT_TRACE_NUM_TOKEN
from ..configuration_base import BaseLLMConfig
from ..modeling_base import BaseModelChunk, BaseModelTail
from ..norm import LayerNorm, RMSNorm
from .attention import Attention

np.random.seed(42)


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) class.

    This class implements the MLP component with optional LoRA (Low-Rank Adaptation) support.

    Attributes:
        config (BaseLLMConfig): Configuration for the MLP.
        lora (LoRA): LoRA object.
        layer_idx (int): Index of the layer.
        jit_trace (bool): Whether to use JIT tracing.
        hidden_size (int): Size of the hidden layer.
        intermediate_size (int): Size of the intermediate layer.
        parallel_lora (bool): Whether to use parallel LoRA.
        gate_proj (nn.Linear): Linear layer for gate projection.
        up_proj (nn.Linear): Linear layer for up projection.
        down_proj (nn.Linear): Linear layer for down projection.
        mul (mtk_quantization.pytorch.functional.Mul): Multiplication function from mtk_quantization.
        mul2 (mtk_quantization.pytorch.functional.Mul): Second multiplication function from mtk_quantization.
        with_lora (bool): Whether to use LoRA.
    """

    def __init__(self, config: BaseLLMConfig, lora, layer_idx, jit_trace=False, parallel_lora=False):
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

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.mul = mtk_quantization.pytorch.functional.Mul()
        self.mul2 = mtk_quantization.pytorch.functional.Mul()

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
                        elif module == self.config.fc_names['mlp']['gate']:
                            self.gate_lora_A = nn.Linear(self.hidden_size, self.lora.rank, bias=False).apply(
                                qat_utils.init_lora_A
                            )
                            self.gate_lora_B = nn.Linear(self.lora.rank, self.intermediate_size, bias=False).apply(
                                qat_utils.init_lora_B
                            )
                            self.gate_lora_add = mtk_quantization.pytorch.functional.Add()
                            if self.lora.scale != 1.0:
                                self.gate_lora_scale_mul = mtk_quantization.pytorch.functional.Mul()
                            if self.lora.dropout > 0.0:
                                self.gate_lora_dropout = nn.Dropout(p=self.lora.dropout)
                            else:
                                self.gate_lora_dropout = nn.Identity()
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
                        elif module == self.config.fc_names['mlp']['gate']:
                            self.gate_lora_A = mtk_quantization.pytorch.functional.Matmul()
                            self.gate_lora_B = mtk_quantization.pytorch.functional.Matmul()
                            self.gate_lora_add = mtk_quantization.pytorch.functional.Add()
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
        if self.lora is not None and self.config.fc_names['mlp']['gate'] in self.lora.target_modules and self.with_lora:
            gate2 = self.gate_proj(x)
            if self.parallel_lora:
                gate_lora = self.gate_lora_B(self.gate_lora_A(self.gate_lora_dropout(x)))
                if self.lora.scale != 1.0:
                    gate_lora = self.gate_lora_scale_mul(gate_lora, self.lora.scale)
            else:
                gate_lora = self.gate_lora_B(
                    self.gate_lora_A(x, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
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
                    self.up_lora_A(x, lora_inputs[lora_idx].transpose(1, 2)),
                    lora_inputs[lora_idx + 1].transpose(1, 2),
                )
                lora_idx += 2
            up = self.up_lora_add(up2, up_lora)
        else:
            up = self.up_proj(x)

        pre_down = self.mul(self.mul2(gate, torch.sigmoid(gate)), up)
        if self.lora is not None and self.config.fc_names['mlp']['down'] in self.lora.target_modules and self.with_lora:
            down2 = self.down_proj(pre_down)
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


class DecoderLayer(nn.Module):
    """Decoder layer class for the model.

    This class implements a decoder layer with self-attention and MLP components,
    and optional LoRA (Low-Rank Adaptation) support.

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
    """

    def __init__(
        self,
        config: BaseLLMConfig,
        lora,
        jit_trace=False,
        layer_idx=None,
        attn_class=Attention,
        mlp_class=MLP,
        norm_class=RMSNorm,
        parallel_lora=False,
        exclude_input_norm=False,
        use_single_bmm_attention=False,
    ):
        """Initializes the DecoderLayer class.

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
        super().__init__()
        self.hidden_size = config.hidden_size
        self.jit_trace = jit_trace
        self.exclude_input_norm = exclude_input_norm
        self.infini_attention = config.infini_attention
        self.use_split_mask = config.use_split_mask
        self.self_attn = attn_class(
            config,
            lora,
            layer_idx,
            jit_trace=jit_trace,
            parallel_lora=parallel_lora,
            use_single_bmm_attention=use_single_bmm_attention,
        )
        self.mlp = mlp_class(config, lora, layer_idx, jit_trace=jit_trace, parallel_lora=parallel_lora)

        if norm_class is None:
            norm_class = RMSNorm if config.norm == 'RMSNorm' else LayerNorm

        if not self.exclude_input_norm:
            self.input_norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.post_attention_norm = norm_class(config.hidden_size, eps=config.norm_eps).float()

        self.add = mtk_quantization.pytorch.functional.Add()
        self.add2 = mtk_quantization.pytorch.functional.Add()

        self.expected_attn_lora_inputs = 0
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
                elif module in [v for k, v in config.fc_names['mlp'].items() if k != 'name']:
                    if module == config.fc_names['mlp']['gateup']:
                        logger.error(
                            f'combined GateUp FC not supported any longer. Please use '
                            f'{config.fc_names["mlp"]["gate"]}, '
                            f'{config.fc_names["mlp"]["up"]}',
                            err=NotImplementedError,
                        )
                    self.expected_mlp_lora_inputs += 2

    def forward(self, hidden_states, *maybe_infini_and_lora_inputs, **inputs):
        """Forward pass for the decoder layer.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.
            maybe_infini_and_lora_inputs (Tuple): Additional inputs for infini attention and LoRA.
            inputs (dict): Helpful inputs for attention including.
                mask (torch.Tensor): Attention mask.
                cos (torch.Tensor): Cosine rotational embedding.
                sin (torch.Tensor): Sine rotational embedding.
                (k_cos) (torch.Tensor): Cosine rotational embedding for sink_rope.
                (k_sin) (torch.Tensor): Sine rotational embedding for sink_rope.
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

        maybe_infini_and_lora_inputs_slicing = self.expected_attn_lora_inputs
        if self.infini_attention:
            maybe_infini_and_lora_inputs_slicing += 3
        if self.use_split_mask:
            maybe_infini_and_lora_inputs_slicing += 1

        attn_outputs = self.self_attn(
            hidden_states, *maybe_infini_and_lora_inputs[:maybe_infini_and_lora_inputs_slicing], **inputs
        )

        attn_output, present_key, present_value = attn_outputs[:3]
        hidden_states = self.add(residual, attn_output)

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.post_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(hidden_states.to(torch.float32)).to(dtype)
        hidden_states = self.mlp(hidden_states, *maybe_infini_and_lora_inputs[maybe_infini_and_lora_inputs_slicing:])
        hidden_states = self.add2(residual, hidden_states)

        if len(attn_outputs) > 3:
            return hidden_states, present_key, present_value, *attn_outputs[3:]
        return hidden_states, present_key, present_value


class Tail(BaseModelTail):
    """Tail class for the model.

    This class implements the tail component of the model, which includes normalization and a linear layer
    for language modeling.

    Attributes:
        norm (nn.Module): Normalization layer.
        lm_head (nn.Linear): Linear layer for language modeling.
        support_quant_stub (bool): Whether quantization stub is supported.
        hidden_states (mtk_quantization.pytorch.functional.QuantizerStub): Quantizer stub for hidden states.
    """

    def __init__(self, config, chunk_idx, dtype=torch.float32, jit_trace=False, norm_class=None):
        """Initializes the Tail class.

        Args:
            config (BaseLLMConfig): Configuration for the tail component.
            chunk_idx (int): Index of the chunk.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            norm_class (type, optional): Class for the normalization. Default is None.
        """
        super().__init__(config, chunk_idx, dtype, jit_trace=jit_trace)

        if norm_class is None:
            norm_class = RMSNorm if config.norm == 'RMSNorm' else LayerNorm

        self.norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + config.lm_head_pad_size)

        if self.support_quant_stub:
            self.hidden_states = mtk_quantization.pytorch.functional.QuantizerStub()

    def forward(self, hidden_states):
        """Forward pass for the tail component.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.

        Returns:
            torch.Tensor: Output tensor after applying normalization and linear layer.
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

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        return torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size, device='cpu', dtype=torch.float32)

    def get_ptq_inputs(self, args, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        Args:
            args (Namespace): Arguments for PTQ.
            kwargs: Additional keyword arguments.

        Returns:
            tuple: Tuple containing input shapes, input value ranges, calibration data generator,
            and evaluation data generator.
        """
        input_shapes = [[None, None, self.config.hidden_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    yield [np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32)]
        else:

            def calib_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.calibration_dataset, 'llm', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        if args.calibration_dataset == 'fake':

            def eval_data_gen():
                for _ in range(10):
                    yield [np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32)]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'llm', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen


class ModelChunk(BaseModelChunk):
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
        config: BaseLLMConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        jit_trace=False,
        decoder_class=DecoderLayer,
        norm_class=None,
        parallel_lora=False,
        distribute_layers=True,
        use_single_bmm_attention=False,
    ):
        """Initializes the ModelChunk class.

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
            parallel_lora,
            distribute_layers,
        )
        if norm_class is None:
            norm_class = RMSNorm if config.norm == 'RMSNorm' else LayerNorm

        self.use_single_bmm_attention = use_single_bmm_attention

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

        self.infini_attention = config.infini_attention
        self.use_split_mask = config.use_split_mask

        self.expected_num_pos_emb_inputs = 4 if self.config.extra_input['sink_rope'] else 2
        self.expected_num_cache_inputs_per_layer = 2

        # infini attention has mem and z for each layer, categorize as cache inputs
        if self.infini_attention:
            self.expected_num_cache_inputs_per_layer += 2

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
            2
            + self.expected_num_pos_emb_inputs
            + self.expected_num_cache_inputs
            + self.expected_num_lora_inputs
            + int(self.infini_attention)  # for infini mask
            + int(self.use_split_mask)  # for additional mask
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
        decoder_inputs = {'mask': inputs[1], 'cos': inputs[2], 'sin': inputs[3]}

        i = 4
        if self.config.extra_input['sink_rope']:
            decoder_inputs['k_cos'] = inputs[i]
            decoder_inputs['k_sin'] = inputs[i + 1]
            i += 2

        cache = inputs[i : i + self.expected_num_cache_inputs]
        i += self.expected_num_cache_inputs
        if self.infini_attention:
            infini_mask = [inputs[i]]
            i += 1
        else:
            infini_mask = []

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

        hidden_states = inputs_embeds

        cache_outputs = []
        attn_score = []
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
                *[x.to(self.device_list[idx]) for x in infini_mask],
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
            if len(decoder_outputs) > 3:
                attn_score.extend([out.to(inputs_embeds.device) for out in decoder_outputs[3:]])

        if self.include_tail:
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)
            hidden_states = self.lm_head(hidden_states)
        return hidden_states, *cache_outputs, *attn_score

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.

        Returns:
            Tuple containing the input tensors for JIT tracing.
        """
        if self.config.sparse_attn and self.first_layer_idx not in [0, self.config.num_hidden_layers - 1]:
            num_head = self.config.sparse_attn_num_head
        else:
            num_head = self.config.num_key_value_heads

        num_attention_heads = self.config.num_attention_heads

        normal_mask = (
            torch.randn(
                1,
                1,
                DEFAULT_JIT_TRACE_NUM_TOKEN,
                DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                device='cpu',
                dtype=torch.float32,
            )
            if not self.use_split_mask
            else torch.randn(
                1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, DEFAULT_JIT_TRACE_NUM_TOKEN, device='cpu', dtype=torch.float32
            )
        )
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
        if self.config.extra_input['sink_rope']:
            pos_emb_inputs.extend(
                [
                    torch.randn(
                        1,
                        1,
                        DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        self.head_dim,
                        device='cpu',
                        dtype=torch.float32,
                    )  # k_cos
                    if not self.use_single_bmm_attention
                    else torch.randn(
                        1,
                        DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        1,
                        self.head_dim,
                        device='cpu',
                        dtype=torch.float32,
                    ),
                    torch.randn(
                        1,
                        1,
                        DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        self.head_dim,
                        device='cpu',
                        dtype=torch.float32,
                    )  # k_sin
                    if not self.use_single_bmm_attention
                    else torch.randn(
                        1,
                        DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        1,
                        self.head_dim,
                        device='cpu',
                        dtype=torch.float32,
                    ),
                ]
            )

        cache_inputs = []
        for _ in range(self.num_layers):
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # K cache
            cache_inputs.append(
                torch.randn(1, num_head, DEFAULT_JIT_TRACE_CACHE_SIZE, self.head_dim, device='cpu', dtype=torch.float32)
            )  # V cache
            if self.infini_attention:
                # mem and z
                cache_inputs.append(
                    torch.randn(1, num_head, self.head_dim, self.head_dim, device='cpu', dtype=torch.float32)
                )
                cache_inputs.append(torch.randn(1, num_head, 1, self.head_dim, device='cpu', dtype=torch.float32))

        other_masks_inputs = []
        if self.infini_attention:
            other_masks_inputs.append(
                torch.randn(
                    1,
                    num_attention_heads,
                    DEFAULT_JIT_TRACE_NUM_TOKEN,
                    self.head_dim,
                    device='cpu',
                    dtype=torch.float32,
                )
            )

        if self.use_split_mask:
            other_masks_inputs.append(
                torch.randn(1, 1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE, device='cpu', dtype=torch.float32)
            )

        lora_inputs = []
        if self.lora is not None and not self.parallel_lora:
            for i in range(self.num_layers):
                if not self.with_lora[i]:
                    continue
                for module in self.lora.target_modules:
                    if module == self.config.fc_names['attn']['q']:
                        lora_inputs.append(
                            torch.randn(1, self.lora.rank, self.config.hidden_size, device='cpu', dtype=torch.float32)
                        )
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.head_dim * num_attention_heads,
                                self.lora.rank,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                    elif module == self.config.fc_names['attn']['k'] or module == self.config.fc_names['attn']['v']:
                        lora_inputs.append(
                            torch.randn(1, self.lora.rank, self.config.hidden_size, device='cpu', dtype=torch.float32)
                        )
                        lora_inputs.append(
                            torch.randn(1, num_head * self.head_dim, self.lora.rank, device='cpu', dtype=torch.float32)
                        )
                    elif module == self.config.fc_names['attn']['o']:
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.lora.rank,
                                self.head_dim * num_attention_heads,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                        lora_inputs.append(
                            torch.randn(1, self.config.hidden_size, self.lora.rank, device='cpu', dtype=torch.float32)
                        )
                    elif module == self.config.fc_names['mlp']['gate'] or module == self.config.fc_names['mlp']['up']:
                        lora_inputs.append(
                            torch.randn(1, self.lora.rank, self.config.hidden_size, device='cpu', dtype=torch.float32)
                        )
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.config.intermediate_size,
                                self.lora.rank,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                    elif module == self.config.fc_names['mlp']['down']:
                        lora_inputs.append(
                            torch.randn(
                                1,
                                self.lora.rank,
                                self.config.intermediate_size,
                                device='cpu',
                                dtype=torch.float32,
                            )
                        )
                        lora_inputs.append(
                            torch.randn(1, self.config.hidden_size, self.lora.rank, device='cpu', dtype=torch.float32)
                        )

        return (
            torch.randn(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size, device='cpu', dtype=torch.float32),
            normal_mask,
            *pos_emb_inputs,
            *cache_inputs,
            *other_masks_inputs,
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
            # Note: lora inputs will be None only when the whole model has no lora
            # if it is partial layer lora, then lora inputs will not be None
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
                        lora_shapes.append([None, self.config.intermediate_size, None])
                    elif module == self.config.fc_names['mlp']['down']:
                        lora_shapes.append([None, None, self.config.intermediate_size])
                        lora_shapes.append([None, self.config.hidden_size, None])

        normal_mask_shape = [None, 1, None, None]

        pos_emb_shapes = [
            [None, 1, None, self.head_dim] if not self.use_single_bmm_attention else [None, None, 1, self.head_dim],
            [None, 1, None, self.head_dim] if not self.use_single_bmm_attention else [None, None, 1, self.head_dim],
        ]
        if self.config.extra_input['sink_rope']:
            pos_emb_shapes.extend(
                [
                    [None, 1, None, self.head_dim]
                    if not self.use_single_bmm_attention
                    else [None, None, 1, self.head_dim],
                    [None, 1, None, self.head_dim]
                    if not self.use_single_bmm_attention
                    else [None, None, 1, self.head_dim],
                ]
            )

        cache_shapes = [
            [None, num_head, None, self.head_dim],
            [None, num_head, None, self.head_dim],
        ]

        if self.infini_attention:
            cache_shapes.extend(
                [
                    [None, num_head, self.head_dim, self.head_dim],
                    [None, num_head, 1, self.head_dim],
                ]
            )

        other_mask_shapes = []
        other_mask_ranges = []
        if self.infini_attention:
            other_mask_shapes.append([None, num_attention_heads, None, self.head_dim])
            other_mask_ranges.append((0.0, 1.0))

        if self.use_split_mask:
            other_mask_shapes.append([None, 1, None, None])
            other_mask_ranges.append((self.config.mask_value, 0.0))

        input_shapes = [
            [None, None, self.config.hidden_size],
            normal_mask_shape,
            *pos_emb_shapes,
            *cache_shapes,
            *other_mask_shapes,
            *lora_shapes,
        ]
        input_value_ranges = [
            emb_minmax,
            (self.config.mask_value, 0.0),
            *[rot_emb_minmax for _ in range(len(pos_emb_shapes))],
            *[None for _ in range(len(cache_shapes))],
            *other_mask_ranges,
            *[None for _ in range(len(lora_shapes))],
        ]

        if args.calibration_dataset is None:
            calib_data_gen = None
        elif args.calibration_dataset == 'fake':

            def calib_data_gen():
                for i in range(10):
                    normal_mask_data = (
                        np.random.rand(
                            1,
                            1,
                            DEFAULT_JIT_TRACE_NUM_TOKEN,
                            DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        ).astype(np.float32)
                        if not self.use_split_mask
                        else np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, DEFAULT_JIT_TRACE_NUM_TOKEN).astype(
                            np.float32
                        )
                    )
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                    ]
                    if self.config.extra_input['sink_rope']:
                        pos_emb_data.extend(
                            [
                                np.random.rand(
                                    1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim
                                ).astype(np.float32)  # k_cos
                                if not self.use_single_bmm_attention
                                else np.random.rand(
                                    1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim
                                ).astype(np.float32),
                                np.random.rand(
                                    1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim
                                ).astype(np.float32)  # k_sin
                                if not self.use_single_bmm_attention
                                else np.random.rand(
                                    1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim
                                ).astype(np.float32),
                            ]
                        )
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
                    other_mask_data = []
                    if self.infini_attention:
                        other_mask_data.append(
                            np.random.rand(1, num_attention_heads, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(
                                np.float32
                            ),
                        )

                    if self.use_split_mask:
                        other_mask_data.append(
                            np.random.rand(1, 1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE).astype(np.float32),
                        )

                    return_data = [
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        normal_mask_data,
                        *pos_emb_data,
                        *cache_data,
                        *other_mask_data,
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
                    if self.config.extra_input['sink_rope']:
                        return_data = [
                            data['inputs_embeds'].astype(np.float32),
                            data['mask'].astype(np.float32),
                            data['cos'].astype(np.float32),
                            data['sin'].astype(np.float32),
                            data['k_cos'].astype(np.float32),
                            data['k_sin'].astype(np.float32),
                        ]
                    else:
                        return_data = [
                            data['inputs_embeds'].astype(np.float32),
                            data['mask'].astype(np.float32),
                            data['cos'].astype(np.float32),
                            data['sin'].astype(np.float32),
                        ]

                    return_data.extend([data['past_keys'].astype(np.float32), data['past_values'].astype(np.float32)])
                    if self.infini_attention:
                        return_data.append(data['mem'].astype(np.float32))
                        return_data.append(data['z'].astype(np.float32))
                        return_data.append(data['infini_mask'].astype(np.float32))
                    if self.use_split_mask:
                        return_data.append(data['split_mask'].astype(np.float32))

                    return_data.extend([*lora_inputs[calib_lora_map[i]]])
                    yield return_data

        if args.evaluation_dataset is None:
            eval_data_gen = None
        elif args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for i in range(10):
                    normal_mask_data = (
                        np.random.rand(
                            1,
                            1,
                            DEFAULT_JIT_TRACE_NUM_TOKEN,
                            DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN,
                        ).astype(np.float32)
                        if not self.use_split_mask
                        else np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, DEFAULT_JIT_TRACE_NUM_TOKEN).astype(
                            np.float32
                        )
                    )
                    pos_emb_data = [
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                        np.random.rand(1, 1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(np.float32)
                        if not self.use_single_bmm_attention
                        else np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim).astype(np.float32),
                    ]
                    if self.config.extra_input['sink_rope']:
                        pos_emb_data.extend(
                            [
                                np.random.rand(
                                    1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim
                                ).astype(np.float32)  # k_cos
                                if not self.use_single_bmm_attention
                                else np.random.rand(
                                    1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim
                                ).astype(np.float32),
                                np.random.rand(
                                    1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim
                                ).astype(np.float32)  # k_sin
                                if not self.use_single_bmm_attention
                                else np.random.rand(
                                    1, DEFAULT_JIT_TRACE_CACHE_SIZE + DEFAULT_JIT_TRACE_NUM_TOKEN, 1, self.head_dim
                                ).astype(np.float32),
                            ]
                        )
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
                    other_mask_data = []
                    if self.infini_attention:
                        other_mask_data.append(
                            np.random.rand(1, num_attention_heads, DEFAULT_JIT_TRACE_NUM_TOKEN, self.head_dim).astype(
                                np.float32
                            ),
                        )

                    if self.use_split_mask:
                        other_mask_data.append(
                            np.random.rand(1, 1, 1, DEFAULT_JIT_TRACE_CACHE_SIZE).astype(np.float32),
                        )

                    return_data = [
                        np.random.rand(1, DEFAULT_JIT_TRACE_NUM_TOKEN, self.config.hidden_size).astype(np.float32),
                        normal_mask_data,
                        *pos_emb_data,
                        *cache_data,
                        *other_mask_data,
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
                    if self.config.extra_input['sink_rope']:
                        return_data.extend(
                            [
                                data['k_cos'].astype(np.float32),
                                data['k_sin'].astype(np.float32),
                            ]
                        )
                    return_data.extend([data['past_keys'].astype(np.float32), data['past_values'].astype(np.float32)])
                    if self.infini_attention:
                        return_data.append(data['mem'].astype(np.float32))
                        return_data.append(data['z'].astype(np.float32))
                        return_data.append(data['infini_mask'].astype(np.float32))
                    if self.use_split_mask:
                        return_data.append(data['split_mask'].astype(np.float32))
                    return_data.extend([*lora_inputs[calib_lora_map[i]]])
                    yield return_data

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
