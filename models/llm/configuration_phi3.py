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
"""Configuration of Phi3."""

from ...utils import logger
from .configuration_common import CommonConfig


class Phi3Config(CommonConfig):
    """Configuration class for the Phi3 model.

    This class inherits from CommonConfig and is used to set up the configuration
    for the Phi3 model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'phi3'.
        fc_names (dict): A dictionary containing the names of the fully connected layers.
        original_max_position_embeddings (int): The original maximum position embeddings.
        rope_scaling (dict): The configuration for rope scaling.
    """

    def __init__(self, **kwargs):
        """Initializes the Phi3Config class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'phi3'.
            KeyError: If required configuration parameters are missing.
            ValueError: If invalid values are provided for certain parameters.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'phi3')
        if self.model_type != 'phi3':
            logger.error(f'Expected model_type to be phi3 but got {self.model_type} instead')

        self.fc_names = {
            'attn': {
                'name': 'self_attn',
                'qkv': 'qkv_proj',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'o_proj',
            },
            'mlp': {'name': 'mlp', 'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj', 'gateup': 'gate_up_proj'},
            'tail': {'name': 'lm_head'},
        }
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': 'input_layernorm',
            'post_attn': 'post_attention_layernorm',
            'final': 'norm',
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }

        # phi-3 attributes
        if self.original_max_position_embeddings is None:
            logger.error(
                'original_max_position_embeddings is required for Phi3 but missing from config.json', err=KeyError
            )
        self._rope_scaling_adjustment()
        self._rope_scaling_validation()

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """logger.info the configuration of Phi3 model."""
        super().print_config()
        logger.info(f'Original max position embeddings:        {self.original_max_position_embeddings}')
        logger.info(f'Rope scaling:                            {self.rope_scaling}')

    def _rope_scaling_adjustment(self):
        """Adjust the `type` of the `rope_scaling` configuration for backward compatibility."""
        if self.rope_scaling is None:
            return

        rope_scaling_type = self.rope_scaling.get('type', None)

        # For backward compatibility if previous version used "su" or "yarn"
        if rope_scaling_type is not None and rope_scaling_type in ['su', 'yarn']:
            self.rope_scaling['type'] = 'longrope'

    def _rope_scaling_validation(self):
        """Validate the `rope_scaling` configuration."""
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            logger.error(
                '`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, '
                f'got {self.rope_scaling}',
                err=ValueError,
            )
        rope_scaling_type = self.rope_scaling.get('type', None)
        rope_scaling_short_factor = self.rope_scaling.get('short_factor', None)
        rope_scaling_long_factor = self.rope_scaling.get('long_factor', None)
        if rope_scaling_type is None or rope_scaling_type not in ['longrope']:
            logger.error(
                f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}", err=ValueError
            )
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            logger.error(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}",
                err=ValueError,
            )
        if len(rope_scaling_short_factor) != self.hidden_size // self.num_attention_heads // 2:
            logger.error(
                f"`rope_scaling`'s short_factor field must have length "
                f'{self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}',
                err=ValueError,
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            logger.error(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}",
                err=ValueError,
            )
        if len(rope_scaling_long_factor) != self.hidden_size // self.num_attention_heads // 2:
            logger.error(
                f"`rope_scaling`'s long_factor field must have length "
                f'{self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}',
                err=ValueError,
            )


class Phi4Config(Phi3Config):
    """Configuration class for the Phi4 model."""

    def __init__(self, **kwargs):
        """Initializes the Phi4Config class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'phi4'.
            KeyError: If required configuration parameters are missing.
            ValueError: If invalid values are provided for certain parameters.
        """
        model_type = kwargs.pop('model_type', 'phi4')
        if model_type != 'phi4':
            logger.error(f'Expected model_type to be phi4 but got {self.model_type} instead.', err=ValueError)
        verbose = kwargs.pop('verbose', True)
        kwargs['model_type'] = 'phi3'
        kwargs['verbose'] = False

        # phi4 attributes
        self.partial_rotary_factor = kwargs.pop('partial_rotary_factor', 1)

        super().__init__(**kwargs)
        self.model_type = model_type
        self.fc_names = {
            'attn': {
                'name': 'self_attn',
                'qkv': 'qkv_proj',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'o_proj',
            },
            'mlp': {
                'name': 'mlp',
                'gate': 'g_proj',  # 'gate_proj', #,
                'up': 'u_proj',  # 'up_proj', #,
                'down': 'down_proj',
                'gateup': 'gate_up_proj',
            },
            'tail': {'name': 'lm_head'},
        }
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': 'input_layernorm',
            'post_attn': 'post_attention_layernorm',
            'final': 'norm',
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }
        self.input_mode = self.kwargs.pop('input_mode', None)
        self.lora_prefix = self.kwargs.pop('lora_prefix', None)
        if self.input_mode is not None and self.input_mode not in ['vision', 'speech']:
            logger.error(f'input_mode must be one of "vision", "speech" or None, got {self.input_mode}', err=ValueError)

        if verbose:
            self.print_config()
            self.print_unused_kwargs()

    def _rope_scaling_validation(self):
        """Validate the `rope_scaling` configuration.

        The main difference of Phi4 rope scaling with Phi3 is it allows 'partial rotary embedding'.
        So the rotary_ndims might be less than original rot_dims.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                '`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, '
                f'got {self.rope_scaling}'
            )
        rope_scaling_type = self.rope_scaling.get('type', None)
        rope_scaling_short_factor = self.rope_scaling.get('short_factor', None)
        rope_scaling_long_factor = self.rope_scaling.get('long_factor', None)
        if rope_scaling_type is None or rope_scaling_type not in ['longrope']:
            logger.error(
                f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}", err=ValueError
            )
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            logger.error(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}",
                err=ValueError,
            )
        rotary_ndims = int(self.hidden_size // self.num_attention_heads * self.partial_rotary_factor)
        if len(rope_scaling_short_factor) != rotary_ndims // 2:
            logger.error(
                f"`rope_scaling`'s short_factor field must have length {rotary_ndims // 2}, "
                f'got {len(rope_scaling_short_factor)}',
                err=ValueError,
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            logger.error(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}",
                err=ValueError,
            )
        if len(rope_scaling_long_factor) != rotary_ndims // 2:
            logger.error(
                f"`rope_scaling`'s long_factor field must have length {rotary_ndims // 2}, "
                f'got {len(rope_scaling_long_factor)}',
                err=ValueError,
            )

    def print_config(self):
        """logger.info the configuration of Phi3 model."""
        super().print_config()
        logger.info(f'Partial rotary factor:        {self.partial_rotary_factor}')
        logger.info(f'Input mode:                   {self.input_mode}')
        logger.info(f'LoRA prefix:                  {self.lora_prefix}')
