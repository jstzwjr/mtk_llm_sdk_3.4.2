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
"""PyTorch LLaMA model."""

import numpy as np

from .attention import Attention
from .configuration_llama import LlamaConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class LlamaMLP(MLP):
    """Llama MLP class.

    This class extends the MLP class for the Llama model.
    """

    def __init__(self, config: LlamaConfig, lora, layer_idx, **kwargs):
        """Initializes the LlamaMLP class.

        Args:
            config: A LlamaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class LlamaAttention(Attention):
    """Llama Attention class.

    This class extends the Attention class for the Llama model.
    """

    def __init__(self, config: LlamaConfig, lora, layer_idx, **kwargs):
        """Initializes the LlamaAttention class.

        Args:
            config: A LlamaConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class LlamaDecoderLayer(DecoderLayer):
    """Llama Decoder Layer class.

    This class extends the DecoderLayer class for the Llama model.
    """

    def __init__(self, config: LlamaConfig, lora, **kwargs):
        """Initializes the LlamaDecoderLayer class.

        Args:
            config: A LlamaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=LlamaAttention, mlp_class=LlamaMLP, **kwargs)


class LlamaTail(Tail):
    """Llama Tail class.

    This class extends the Tail class for the Llama model.
    """

    def __init__(self, config: LlamaConfig, chunk_idx, **kwargs):
        """Initializes the LlamaTail class.

        Args:
            config: A LlamaConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class LlamaModelChunk(ModelChunk):
    """Llama Model Chunk class.

    This class extends the ModelChunk class for the Llama model.
    """

    def __init__(self, config: LlamaConfig, lora, **kwargs):
        """Initializes the LlamaModelChunk class.

        Args:
            config: A LlamaConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=LlamaDecoderLayer, **kwargs)
