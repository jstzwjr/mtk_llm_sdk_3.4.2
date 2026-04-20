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
"""Common base classes across all models."""

import json
import os
from abc import ABC, abstractmethod

import mtk_quantization
import torch

from ..utils import logger
from ..utils.sanity_checks import check_support_quantizerstub
from .configuration_base import BaseConfig, BaseLLMConfig, BaseProjectorConfig
from .configuration_pipeline import PipelineConfig


class BaseModel(ABC, torch.nn.Module):
    """BaseModel class for handling all models.

    Attributes:
        config (object): The configuration object.
        dtype (torch.dtype): The default data type.
        jit_trace (bool): Whether model is instantiated for JIT tracing or not.

    Methods:
        __init__(config, dtype, jit_trace, **kwargs): Initialize the BaseModel.
        _generate_default_state_dict_mapping(): Abstract method to generate model's default state dict mapping.
        forward(): Abstract method for the forward pass.
        load_weights(state_dict, state_dict_start_idx, quant_config): Abstract method to load weights.
        get_jit_trace_inputs(): Abstract method to get JIT trace inputs.
        get_ptq_inputs(args, args, exp_name, lora_inputs, calib_lora_map, eval_lora_map):
            Abstract method to get PTQ inputs.
    """

    def __init__(
        self,
        config: BaseConfig,
        lora,
        dtype=torch.float32,
        jit_trace: bool = False,
        distribute_layers: bool = True,
    ):
        """Initialize the BaseModel.

        Args:
            config (object): The configuration object.
            lora (LoRA): LoRA object.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
            jit_trace (bool, optional): Whether model is instantiated for JIT tracing or not. Defaults to False.
            distribute_layers (bool): Whether to distribute layers across all available GPUs.
        """
        torch.nn.Module.__init__(self)
        torch.set_default_dtype(dtype)
        self.config = config
        self.lora = lora
        self.dtype = dtype
        self.jit_trace = jit_trace
        self.distribute_layers = distribute_layers
        self.device_list = []
        self.fc_names = config.fc_names
        self.norm_names = config.norm_names
        self.state_dict_mapping = {}
        self.merged_state_dict_mapping = {}
        self.stubs = []
        self.prefixes = ['']
        self.support_quant_stub = check_support_quantizerstub()

    @abstractmethod
    def _generate_default_state_dict_mapping(self):
        """Abstract method to generate model's default state dict mapping."""
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }

    @abstractmethod
    def generate_default_lora_state_dict_mapping(self):
        """Abstract method to generate model's default LoRA state dict mapping."""

    @abstractmethod
    def forward(self):
        """Abstract method for the model forward pass."""

    @abstractmethod
    def load_weights(self, state_dict, state_dict_start_idx, quant_config=None):
        """Abstract method for loading model weights."""

    @abstractmethod
    def get_jit_trace_inputs(self):
        """Abstract method to get inputs for JIT tracing.

        This method generates random inputs for JIT tracing, including LoRA inputs if applicable.
        """

    @abstractmethod
    def get_ptq_inputs(
        self, args, exp_name, lora_inputs=None, calib_lora_map=None, eval_lora_map=None, has_encoder=False
    ):
        """Abstract method to get PTQ inputs.

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            lora_inputs (list, optional): List of number of lora scenarios of list of LoRA inputs per scenario.
                Default is None.
            calib_lora_map (list, optional): List of LoRA scenario mappings for calibration dataset. Default is None.
            eval_lora_map (list, optional): List of LoRA scenario mappings for evaluation dataset. Default is None.
            has_encoder (bool, optional): Boolean to indicate if pipeline has encoder present. Default is False.
        """


class BaseEncoderModelChunk(BaseModel):
    """Base encoder model.

    This is an abstract base class for all encoder models.

    Attributes:
        config (object): Configuration object containing model parameters.
        jit_trace (bool): Whether to use JIT tracing.
        dtype (torch.dtype): Data type for the model.
    """

    def __init__(
        self,
        config: PipelineConfig,
        lora,
        num_layers: int,
        first_layer_idx: int,
        chunk_idx: int,
        dtype=torch.float32,
        jit_trace: bool = False,
        parallel_lora: bool = False,
        distribute_layers: bool = True,
        **kwargs,
    ):
        """Initialize the BaseVisionEncoderChunk class.

        Args:
            config (object): PipelineConfig object containing entire model parameters.
            lora (LoRA): LoRA object.
            num_layers (int): The number of blocks.
            first_layer_idx (int): The index of the first decoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            distribute_layers (bool, optional): Whether to distribute layers. Defaults to True.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            config,
            lora,
            dtype,
            jit_trace,
            distribute_layers,
        )
        self.layers = []
        self.num_layers = num_layers
        self.first_layer_idx = first_layer_idx
        self.chunk_idx = chunk_idx
        self.parallel_lora = parallel_lora
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        if self.parallel_lora and self.lora is None:
            logger.error('Cannot create parallel lora model when lora object is not provided.')

        if lora is None:
            self.with_lora = [False] * self.num_layers
        else:
            self.with_lora = [
                x >= self.lora.start_idx and x <= self.lora.end_idx
                for x in range(self.first_layer_idx, self.first_layer_idx + self.num_layers + 1)
            ]
        self._generate_default_state_dict_mapping()

    def pop_remaining_unused_weights(self, state_dict):
        """Remove unused encoder weights AFTER loading encoder weights.

        Args:
            state_dict: loaded state_dict.
        """
        return state_dict

    def generate_default_lora_state_dict_mapping(self):
        """Generates default lora state dict mapping for lora handler."""
        if self.lora is None:
            return {}

        state_dict_mapping = {}

        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            # fmt: off
            for fc_name in self.config.fc_names['attn']:
                if fc_name == 'name':
                    continue
                module = self.config.fc_names['attn'][fc_name]
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_{fc_name}_A': {
                            f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_A.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.weight'
                            )
                        },
                        f'{outer_layer_idx}_{fc_name}_B': {
                            f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_B.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.weight'
                            )
                        },
                    }
                )

            for fc_name in self.config.fc_names['mlp']:
                if fc_name == 'name':
                    continue
                module = self.config.fc_names['mlp'][fc_name]
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_{fc_name}_A': {
                            f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_A.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.weight'
                            )
                        },
                        f'{outer_layer_idx}_{fc_name}_B': {
                            f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_B.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.weight'
                            )
                        },
                    }
                )
            # fmt: on
        return state_dict_mapping


class BaseVisionEncoderChunk(BaseEncoderModelChunk):
    """Base vision encoder model.

    This is an abstract base class for vision encoder models.

    Attributes:
        config (object): Configuration object containing model parameters.
        jit_trace (bool): Whether to use JIT tracing.
        dtype (torch.dtype): Data type for the model.
    """

    def __init__(
        self,
        config: PipelineConfig,
        lora,
        num_layers: int,
        first_layer_idx: int,
        chunk_idx: int,
        dtype=torch.float32,
        jit_trace: bool = False,
        parallel_lora: bool = False,
        distribute_layers: bool = True,
        **kwargs,
    ):
        """Initialize the BaseVisionEncoderChunk class.

        Args:
            config (object): PipelineConfig object containing entire model parameters.
            lora (LoRA): LoRA object.
            num_layers (int): The number of blocks.
            first_layer_idx (int): The index of the first decoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            distribute_layers (bool, optional): Whether to distribute layers. Defaults to True.
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
        self._num_image_token = None

    @property
    def num_image_token(self):
        """Get the number of image tokens.

        Returns:
            int: The number of image tokens.
        """
        return self._num_image_token

    def _pop_redundant_prefix(self, state_dict, prefix='model.'):
        """Removes redundant prefixes from the state dict keys.

        Args:
            state_dict (dict): The state dict.
            prefix (str, optional): The prefix to remove. Defaults to 'model.'.

        Returns:
            dict: The state dict with redundant prefixes removed.
        """
        for k in list(state_dict.keys()):
            if prefix in k:
                state_dict[k[len(prefix) :]] = state_dict.pop(k)
        return state_dict


class BaseProjector(BaseModel):
    """Base projector model.

    This is an abstract base class for all projector models.

    Attributes:
        config (object): Projector config object containing projector parameters.
        dtype (torch.dtype): Data type for the projector.
    """

    def __init__(self, config: BaseProjectorConfig, dtype=torch.float32, jit_trace: bool = False, **kwargs):
        """Initialize the BaseProjector class.

        Args:
            config (object): Projector config object containing projector parameters.
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float32.
            jit_trace (bool, optional): Flag to determine if running the projector in jit trace mode or not.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            config,
            None,  # LoRA
            dtype,
            jit_trace,
            distribute_layers=False,
        )
        self.projector = None
        self._generate_default_state_dict_mapping()

    def generate_default_lora_state_dict_mapping(self):
        """Generates default lora state dict mapping for lora handler."""
        logger.error('Projector does not support LoRA.', err=NotImplementedError)


class BaseModelChunk(BaseModel):
    """BaseModelChunk class for handling model chunks.

    Attributes:
        config (object): The configuration object.
        num_layers (int): The number of blocks.
        first_layer_idx (int): The chunk index.
        dtype (torch.dtype): The data type.
        include_tail (bool): Whether to include the tail.
        jit_trace (bool): Whether to use JIT tracing.
        parallel_lora (bool): Whether to use parallel LoRA.
        distribute_layers (bool): Whether to distribute layers across all available GPUs..
        device_list (list): The list of devices.
        fc_names (dict): The dictionary of fully connected layer names.
        head_dim (int): The head dimension.
        with_lora (bool): Whether to use LoRA.

    Methods:
        __init__(config, num_layers, first_layer_idx, dtype, include_tail, jit_trace, parallel_lora, distribute_layers):
            Initialize the BaseModelChunk.
        _generate_default_state_dict_mapping(): Generates the model's default state dict mapping.
        load_weights(state_dict, state_dict_start_idx, quant_config): Common method to load most LLM weights.
    """

    def __init__(
        self,
        config: BaseLLMConfig,
        lora,
        num_layers: int,
        first_layer_idx: int,
        chunk_idx: int,
        dtype=torch.float32,
        include_tail: bool = False,
        jit_trace: bool = False,
        parallel_lora: bool = False,
        distribute_layers: bool = True,
    ):
        """Initialize the BaseModelChunk.

        Args:
            config (object): The configuration object.
            lora (LoRA): LoRA object.
            num_layers (int): The number of blocks.
            first_layer_idx (int): The index of the first decoder layer of this chunk.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
            include_tail (bool, optional): Whether to include the tail. Defaults to False.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            parallel_lora (bool, optional): Whether to use parallel LoRA. Defaults to False.
            distribute_layers (bool, optional): Whether to distribute layers. Defaults to True.
        """
        super().__init__(
            config,
            lora,
            dtype,
            jit_trace,
            distribute_layers,
        )
        self.layers = []
        self.num_layers = num_layers
        self.first_layer_idx = first_layer_idx
        self.chunk_idx = chunk_idx
        self.include_tail = include_tail
        self.parallel_lora = parallel_lora
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        if self.parallel_lora and self.lora is None:
            logger.error('Cannot create parallel lora model when lora object is not provided.')

        if lora is None:
            self.with_lora = [False] * self.num_layers
        else:
            self.with_lora = [
                x >= self.lora.start_idx and x <= self.lora.end_idx
                for x in range(self.first_layer_idx, self.first_layer_idx + self.num_layers + 1)
            ]
        self._generate_default_state_dict_mapping()

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
            # fmt: off
            merged_state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_qkv_weight': (
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.weight'
                    ),
                    f'{outer_layer_idx}_qkv_bias': (
                        f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["qkv"]}.bias'
                    ),
                    f'{outer_layer_idx}_gu_weight': (
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gateup"]}.weight'
                    ),
                    f'{outer_layer_idx}_gu_bias': (
                        f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gateup"]}.bias'
                    ),
                }
            )
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.o_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_g_weight': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_g_bias': {
                        f'layers.{inner_layer_idx}.mlp.gate_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["gate"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_u_weight': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_u_bias': {
                        f'layers.{inner_layer_idx}.mlp.up_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["up"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_d_weight': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.weight': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_d_bias': {
                        f'layers.{inner_layer_idx}.mlp.down_proj.bias': (
                            f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["down"]}.bias'
                        )
                    },
                    f'{outer_layer_idx}_input_norm_weight': {
                        f'layers.{inner_layer_idx}.input_norm.weight': (
                            f'layers.{outer_layer_idx}.{self.norm_names["input"]}.weight'
                        )
                    },
                    f'{outer_layer_idx}_post_attn_norm_weight': {
                        f'layers.{inner_layer_idx}.post_attention_norm.weight': (
                            f'layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.weight'
                        )
                    },
                }
            )
            if self.config.use_qk_norm:
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_q_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.q_norm.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["query"]}.weight'
                            )
                        },
                        f'{outer_layer_idx}_k_norm_weight': {
                            f'layers.{inner_layer_idx}.self_attn.k_norm.weight': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.norm_names["key"]}.weight'
                            )
                        },
                    }
                )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_input_norm_bias': {
                            f'layers.{inner_layer_idx}.input_norm.bias': (
                                f'layers.{outer_layer_idx}.{self.norm_names["input"]}.bias'
                            )
                        },
                        f'{outer_layer_idx}_post_attn_norm_bias': {
                            f'layers.{inner_layer_idx}.post_attention_norm.bias': (
                                f'layers.{outer_layer_idx}.{self.norm_names["post_attn"]}.bias'
                            )
                        },
                    }
                )
            if self.config.infini_attention:
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_infini_attn_betas': {
                            f'layers.{inner_layer_idx}.self_attn.infini_betas': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.gate'
                            )
                        },
                    }
                )
                if self.config.infini_use_combined_mlp:
                    state_dict_mapping.update(
                        {
                            # FIXME: hardcoded the '0' and '2' here
                            f'{outer_layer_idx}_infini_mem_up_proj_weight': {
                                f'layers.{inner_layer_idx}.self_attn.infini_mem_up_proj.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.combine_mlp_1.0.weight'
                                )
                            },
                            f'{outer_layer_idx}_infini_mem_up_proj_bias': {
                                f'layers.{inner_layer_idx}.self_attn.infini_mem_up_proj.bias': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.combine_mlp_1.0.bias'
                                )
                            },
                            f'{outer_layer_idx}_infini_mem_down_proj_weight': {
                                f'layers.{inner_layer_idx}.self_attn.infini_mem_down_proj.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.combine_mlp_1.2.weight'
                                )
                            },
                            f'{outer_layer_idx}_infini_mem_down_proj_bias': {
                                f'layers.{inner_layer_idx}.self_attn.infini_mem_down_proj.bias': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.combine_mlp_1.2.bias'
                                )
                            },
                        }
                    )

            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping()[0])
        # fmt: on
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
                if fc_name == 'qkv':
                    merged_state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.weight'
                            ),
                            f'{outer_layer_idx}_{fc_name}_B': (
                                f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.weight'
                            ),
                        }
                    )
                else:
                    state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': {
                                f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_A.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_A.weight'
                                )
                            },
                            f'{outer_layer_idx}_{fc_name}_B': {
                                f'layers.{inner_layer_idx}.self_attn.{module.replace("proj", "lora")}_B.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{module}.lora_B.weight'
                                )
                            },
                        }
                    )

            for fc_name in self.config.fc_names['mlp']:
                if fc_name == 'name':
                    continue
                module = self.config.fc_names['mlp'][fc_name]
                if fc_name == 'gateup':
                    merged_state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': (
                                f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.weight'
                            ),
                            f'{outer_layer_idx}_{fc_name}_B': (
                                f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.weight'
                            ),
                        }
                    )
                else:
                    state_dict_mapping.update(
                        {
                            f'{outer_layer_idx}_{fc_name}_A': {
                                f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_A.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_A.weight'
                                )
                            },
                            f'{outer_layer_idx}_{fc_name}_B': {
                                f'layers.{inner_layer_idx}.mlp.{module.replace("proj", "lora")}_B.weight': (
                                    f'layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{module}.lora_B.weight'
                                )
                            },
                        }
                    )
            # fmt: on
        return state_dict_mapping, merged_state_dict_mapping

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

            if not found and not internal_key.endswith(('_A', '_B')):
                missing_keys.append((internal_key, external_key))

        if len(missing_keys) > 0:
            for internal_key, external_key in missing_keys:
                logger.warning(f'Unable to find {internal_key} weight in state_dict. Expected subkey: {external_key}')
            logger.info(f'state dict keys for reference: {state_dict_keys}')
            logger.error(
                'Missing keys encountered (as listed above). Please modify your state_dict keys according to the '
                'expected subkeys.',
                err=KeyError,
            )

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

        # modify lm head if there is padding
        if self.include_tail and self.config.lm_head_pad_size != 0:
            # both weight and bias will exist
            weights_to_load['lm_head.weight'] = torch.nn.functional.pad(
                weights_to_load['lm_head.weight'], (0, 0, 0, self.config.lm_head_pad_size)
            )
            weights_to_load['lm_head.bias'] = torch.nn.functional.pad(
                weights_to_load['lm_head.bias'], (0, self.config.lm_head_pad_size)
            )

        if quant_config is not None:
            logger.info(f'Quantizing chunk {self.chunk_idx} using quant config: {quant_config}')
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


class BaseModelTail(BaseModel):
    """BaseModelTail class for handling model tails.

    Attributes:
        config (object): The configuration object.
        dtype (torch.dtype): The data type.
        jit_trace (bool): Whether to use JIT tracing.
        device_list (list): The list of devices.

    Methods:
        __init__(config, dtype, jit_trace): Initialize the BaseModelTail.
        _generate_default_state_dict_mapping(): Generates the model tail's default state dict mapping.
        load_weights(state_dict, state_dict_start_idx, quant_config): Common method to load most LLM tail weights.
    """

    def __init__(
        self,
        config: BaseLLMConfig,
        chunk_idx: int,
        dtype=torch.float32,
        jit_trace: bool = False,
        distribute_layers: bool = True,
    ):
        """Initialize the BaseModelTail.

        Args:
            config (object): The configuration object.
            chunk_idx (int): The current chunk index.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
            distribute_layers (bool, optional): Whether to distribute layers. Defaults to True.
        """
        super().__init__(
            config,
            None,  # LoRA
            dtype,
            jit_trace,
            distribute_layers,
        )
        self.chunk_idx = chunk_idx
        self._generate_default_state_dict_mapping()

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

    def generate_default_lora_state_dict_mapping(self):
        """Generates default lora state dict mapping for lora handler."""
        logger.error('Tail does not support LoRA.', err=NotImplementedError)

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
                    if k.endswith('.' + external_key):
                        logger.debug(
                            f'Found {internal_key} weight using iteration, state_dict key={k}, dtype={dtype}. '
                            f'Adding prefix: {k[: -len(external_key)]}'
                        )
                        self.prefixes.append(k[: -len(external_key)])
                        weights_to_load.update({model_key: state_dict.pop(k).to(dtype)})
                        found = True
                        state_dict_keys.remove(k)
                        break

            if not found:
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
                # Default shapes:
                # lm_head: vocab size
                assert internal_key == 'lm_head_bias'
                shape = self.config.vocab_size
                logger.debug(f'Init bias {internal_key} to zeros using shape={shape} and dtype={dtype}')
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

        # pad lm head
        if self.config.lm_head_pad_size != 0:
            # both weight and bias will exist
            weights_to_load['lm_head.weight'] = torch.nn.functional.pad(
                weights_to_load['lm_head.weight'], (0, 0, 0, self.config.lm_head_pad_size)
            )
            weights_to_load['lm_head.bias'] = torch.nn.functional.pad(
                weights_to_load['lm_head.bias'], (0, self.config.lm_head_pad_size)
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
