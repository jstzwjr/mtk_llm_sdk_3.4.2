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
"""PyTorch Baichuan model."""

import numpy as np

from .attention import Attention
from .configuration_baichuan import BaichuanConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class BaichuanMLP(MLP):
    """Multi-Layer Perceptron (MLP) class for the Baichuan model.

    This class inherits from the MLP class and is used to define the MLP
    component of the Baichuan model.
    """

    def __init__(self, config: BaichuanConfig, lora, layer_idx, **kwargs):
        """Initializes the BaichuanMLP class.

        Args:
            config: A BaichuanConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class BaichuanAttention(Attention):
    """Attention class for the Baichuan model.

    This class inherits from the Attention class and is used to define the attention
    mechanism of the Baichuan model.
    """

    def __init__(self, config: BaichuanConfig, lora, layer_idx, **kwargs):
        """Initializes the BaichuanAttention class.

        Args:
            config: A BaichuanConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class BaichuanDecoderLayer(DecoderLayer):
    """Decoder layer class for the Baichuan model.

    This class inherits from the DecoderLayer class and is used to define the decoder
    layer of the Baichuan model.
    """

    def __init__(self, config: BaichuanConfig, lora, **kwargs):
        """Initializes the BaichuanDecoderLayer class.

        Args:
            config: A BaichuanConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=BaichuanAttention, mlp_class=BaichuanMLP, **kwargs)


class BaichuanTail(Tail):
    """Tail class for the Baichuan model.

    This class inherits from the Tail class and is used to define the tail component
    of the Baichuan model.
    """

    def __init__(self, config: BaichuanConfig, chunk_idx, **kwargs):
        """Initializes the BaichuanTail class.

        Args:
            config: A BaichuanConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class BaichuanModelChunk(ModelChunk):
    """Model chunk class for the Baichuan model.

    This class inherits from the ModelChunk class and is used to define a chunk of the
    Baichuan model.
    """

    def __init__(self, config: BaichuanConfig, lora, **kwargs):
        """Initializes the BaichuanModelChunk class.

        Args:
            config: A BaichuanConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=BaichuanDecoderLayer, **kwargs)
