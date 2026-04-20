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
"""PyTorch MiniCPM model."""

import math

import numpy as np

from .attention import Attention
from .configuration_minicpm import MinicpmConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


def convert_weight_to_llama_format(state_dict, config):
    """Converts weights to LLaMA format.

    This function scales the weights according to the configuration parameters.

    Args:
        state_dict (dict): State dictionary containing the weights.
        config (MinicpmConfig): Configuration for the model.

    Returns:
        dict: State dictionary with converted weights.
    """
    scale_depth = config.scale_depth / math.sqrt(config.num_hidden_layers)
    scale_emb = config.scale_emb
    scale_head = 1 / (config.hidden_size / config.dim_model_base)

    if 'model.embed_tokens.weight' in state_dict:  # minicpm
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight'].clone()
        state_dict['model.embed_tokens.weight'] *= scale_emb
        state_dict['lm_head.weight'] *= scale_head
    elif 'llm.model.embed_tokens.weight' in state_dict:  # minicpmv
        state_dict['llm.lm_head.weight'] = state_dict['llm.model.embed_tokens.weight'].clone()
        state_dict['llm.model.embed_tokens.weight'] *= scale_emb
        state_dict['llm.lm_head.weight'] *= scale_head

    for key in state_dict:
        if ('self_attn.o_proj.weight' in key) | ('mlp.down_proj.weight' in key):
            state_dict[key] *= scale_depth

    return state_dict


class MinicpmMLP(MLP):
    """Minicpm MLP class.

    This class extends the MLP class for the Minicpm model.
    """

    def __init__(self, config: MinicpmConfig, lora, layer_idx, **kwargs):
        """Initializes the MinicpmMLP class.

        Args:
            config: A MinicpmConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class MinicpmAttention(Attention):
    """Minicpm Attention class.

    This class extends the Attention class for the Minicpm model.
    """

    def __init__(self, config: MinicpmConfig, lora, layer_idx, **kwargs):
        """Initializes the MinicpmAttention class.

        Args:
            config: A MinicpmConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class MinicpmDecoderLayer(DecoderLayer):
    """Minicpm Decoder Layer class.

    This class extends the DecoderLayer class for the Minicpm model.
    """

    def __init__(self, config: MinicpmConfig, lora, **kwargs):
        """Initializes the MinicpmDecoderLayer class.

        Args:
            config: A MinicpmConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=MinicpmAttention, mlp_class=MinicpmMLP, **kwargs)


class MinicpmTail(Tail):
    """Minicpm Tail class.

    This class extends the Tail class for the Minicpm model.
    """

    def __init__(self, config: MinicpmConfig, chunk_idx, **kwargs):
        """Initializes the MinicpmTail class.

        Args:
            config: A MinicpmConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class MinicpmModelChunk(ModelChunk):
    """Minicpm Model Chunk class.

    This class extends the ModelChunk class for the Minicpm model.
    """

    def __init__(self, config: MinicpmConfig, lora, **kwargs):
        """Initializes the MinicpmModelChunk class.

        Args:
            config: A MinicpmConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=MinicpmDecoderLayer, **kwargs)
