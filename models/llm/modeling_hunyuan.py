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
"""PyTorch HunYuan model."""

import numpy as np

from .attention import Attention
from .configuration_hunyuan import HunYuanConfig
from .modeling_common import MLP, DecoderLayer, ModelChunk, Tail

np.random.seed(42)


class HunYuanMLP(MLP):
    """HunYuan MLP class.

    This class extends the MLP class for the HunYuan model.
    """

    def __init__(self, config: HunYuanConfig, lora, layer_idx, **kwargs):
        """Initializes the HunYuanMLP class.

        Args:
            config: A HunYuanConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class HunYuanAttention(Attention):
    """HunYuan Attention class.

    This class extends the Attention class for the HunYuan model.
    """

    def __init__(self, config: HunYuanConfig, lora, layer_idx, **kwargs):
        """Initializes the HunYuanAttention class.

        Args:
            config: A HunYuanConfig object.
            lora (LoRA): LoRA object.
            layer_idx (int): The index of the layer.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, layer_idx, **kwargs)


class HunYuanDecoderLayer(DecoderLayer):
    """HunYuan Decoder Layer class.

    This class extends the DecoderLayer class for the HunYuan model.
    """

    def __init__(self, config: HunYuanConfig, lora, **kwargs):
        """Initializes the HunYuanDecoderLayer class.

        Args:
            config: A HunYuanConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, attn_class=HunYuanAttention, mlp_class=HunYuanMLP, **kwargs)


class HunYuanTail(Tail):
    """HunYuan Tail class.

    This class extends the Tail class for the HunYuan model.
    """

    def __init__(self, config: HunYuanConfig, chunk_idx, **kwargs):
        """Initializes the HunYuanTail class.

        Args:
            config: A HunYuanConfig object.
            chunk_idx (int): Index of the chunk.
            kwargs: All keyword arguments.
        """
        super().__init__(config, chunk_idx, **kwargs)


class HunYuanModelChunk(ModelChunk):
    """HunYuan Model Chunk class.

    This class extends the ModelChunk class for the HunYuan model.
    """

    def __init__(self, config: HunYuanConfig, lora, **kwargs):
        """Initializes the HunYuanModelChunk class.

        Args:
            config: A HunYuanConfig object.
            lora (LoRA): LoRA object.
            kwargs: All keyword arguments.
        """
        super().__init__(config, lora, decoder_class=HunYuanDecoderLayer, **kwargs)
