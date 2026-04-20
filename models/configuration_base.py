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
"""Define base config class."""

from abc import ABC, abstractmethod

from ..utils import const, logger


class BaseConfig(ABC):
    """BaseConfig class for handling configuration settings.

    Attributes:
        model_type (str): The type of the model.
        type (str): The type of the configuration.
        kwargs (dict): Additional keyword arguments.
        weight_dir (str): The directory for weights.

    Methods:
        __init__(**kwargs): Initialize the BaseConfig.
        print_config(): Abstract method to print the configuration.
        print_unused_kwargs(): Print unused keyword arguments.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.model_type = 'base'
        self.type = 'base'

        self.kwargs = kwargs
        self.weight_dir = self.kwargs.pop('weight_dir', None)

    @abstractmethod
    def print_config(self):
        """Abstract method to print the configuration."""

    def print_unused_kwargs(self):
        """Print unused keyword arguments."""
        if len(list(self.kwargs.keys())) > 0:
            logger.debug(f'Unused kwargs: {list(self.kwargs.keys())}')


class BasePreprocessorConfig(BaseConfig):
    """BasePreprocessorConfig class for handling preprocessor configuration settings.

    Methods:
        __init__(**kwargs): Initialize the BasePreprocessorConfig.
        get(): Gets the preprocessor config as a dict to pass into a preprocessor class.
    """

    def __init__(self, **kwargs):
        """Initialize the BasePreprocessorConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def get(self):
        """Gets the preprocessor config as a dict to pass into a preprocessor class."""
        return self.kwargs


class BaseEncoderConfig(BaseConfig):
    """BaseEncoderConfig class for handling encoder configuration settings.

    Attributes:
        hidden_size (int): The hidden size.
        intermediate_size (int): The intermediate size.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        norm_eps (float): The epsilon value for normalization.
        mask_value (float): The mask value.

    Methods:
        __init__(**kwargs): Initialize the BaseEncoderConfig.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseEncoderConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = 'base_encoder'
        self.type = 'encoder'

        self.hidden_size = 0
        self.intermediate_size = 0
        self.num_hidden_layers = 0
        self.num_attention_heads = 0
        self.norm_eps = 0
        self.mask_value = -100.0


class BaseVisionEncoderChunkConfig(BaseEncoderConfig):
    """BaseVisionEncoderChunkConfig class for handling vision encoder configuration settings.

    Attributes:
        projection_dim (int): The projection dimension.
        image_size (int): The image size.
        patch_size (int): The patch size.
        num_channels (int): The number of channels.
        mm_projector_type (str): The type of multimodal projector.
        mm_hidden_size (int): The hidden size for multimodal projector.
        mm_vision_select_layer (int): The selected layer for vision.
        mm_vision_select_feature (str): The selected feature for vision.

    Methods:
        __init__(**kwargs): Initialize the BaseVisionEncoderChunkConfig.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseVisionEncoderChunkConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = 'base_vision_encoder'
        self.type = 'vision_encoder'

        self.norm = 'LayerNorm'
        self.image_size = None
        self.patch_size = None
        self.num_channels = None

        self.image_text = const.DEFAULT_IMAGE_TOKEN
        self.image_token = const.DEFAULT_IMAGE_TOKEN_ID


class BaseAudioEncoderConfig(BaseEncoderConfig):
    """BaseAudioEncoderConfig class for handling audio encoder configuration settings.

    Methods:
        __init__(**kwargs): Initialize the BaseAudioEncoderConfig.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseAudioEncoderConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = 'base_audio_encoder'
        self.type = 'audio_encoder'

        self.decoder_hidden_size = 0
        self.decoder_num_layers = 0
        self.num_mel_bins = 0
        self.max_source_positions = 0

        self.sparse_attn = False
        self.sparse_attn_num_head = 0
        self.num_key_value_heads = 0

        self.audio_text = const.DEFAULT_AUDIO_TOKEN
        self.audio_token = const.DEFAULT_AUDIO_TOKEN_ID


class BaseProjectorConfig(BaseConfig):
    """BaseProjectorConfig class for handling encoder projector settings.

    Attributes:
        hidden_size (int): The hidden size.
        intermediate_size (int): The intermediate size.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        norm_eps (float): The epsilon value for normalization.
        mask_value (float): The mask value.

    Methods:
        __init__(**kwargs): Initialize the BaseEncoderConfig.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseEncoderConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = 'base_projector'
        self.type = 'projector'
        self.fc_names = {}
        self.norm_names = {}


class BaseLLMConfig(BaseConfig):
    """BaseLLMConfig class for handling LLM configuration settings.

    Attributes:
        hidden_size (int): The hidden size.
        intermediate_size (int): The intermediate size.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        head_dim (int): The head dimension.
        norm_eps (float): The epsilon value for normalization.
        mask_value (float): The mask value.
        mask_scaling_factors (list): List of scaling factors for per layer mask value.
        vocab_size (int): The vocabulary size.
        num_key_value_heads (int): The number of key-value heads.
        max_position_embeddings (int): The maximum position embeddings.
        rotary_emb_base (int): The base value for rotary embeddings.
        ntk_scaling_factor (float): The scaling factor for NTK.
        norm (str): The normalization type.
        bos_token_id (int): The ID for the beginning of sentence token.
        pad_token_id (int): The ID for the padding token.
        eos_token_id (int or list): The ID for the end of sentence token.
        unk_token_id (int): The ID for the unknown token.
        ring_buffer (bool): Whether to use a ring buffer.
        sliding_window_attention_size (int): The size of the sliding window attention.
        sparse_attn (bool): Whether to use sparse attention.
        sparse_attn_num_head (int): The number of heads for sparse attention.
        use_stable_embedding (bool): Whether to use stable embedding.
        tie_word_embeddings (bool): Whether to tie word embeddings.
        fc_names (dict): The dictionary of fully connected layer names.
        tokenizer (str): The tokenizer type.

    Methods:
        __init__(**kwargs): Initialize the BaseLLMConfig.
    """

    def __init__(self, **kwargs):
        """Initialize the BaseLLMConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_type = 'base_llm'
        self.type = 'llm'

        self.hidden_size = 0
        self.intermediate_size = 0
        self.num_hidden_layers = 0
        self.num_attention_heads = 0
        self.head_dim = 0
        self.norm_eps = 0
        self.mask_value = -100.0
        self.mask_scaling_factors = None

        self.vocab_size = None
        self.num_key_value_heads = 0
        self.max_position_embeddings = 0
        self.rotary_emb_base = 10000
        self.ntk_scaling_factor = 1.0
        self.norm = None
        self.norm_eps = 0

        self.bos_token_id = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None

        self.ring_buffer = True
        self.sliding_window_attention_size = 0
        self.sparse_attn = False
        self.sparse_attn_num_head = 0

        self.use_stable_embedding = False
        self.tie_word_embeddings = False

        self.fc_names = {
            'attn': {'name': None, 'qkv': None, 'q': None, 'k': None, 'v': None, 'o': None},
            'mlp': {'name': None, 'gate': None, 'up': None, 'down': None, 'gateup': None},
            'tail': {'name': None},
        }
        self.norm_names = {
            'stable_embedding': 'embed_layer_norm',
            'input': None,
            'post_attn': None,
            'final': None,
            'query': 'query_layernorm',
            'key': 'key_layernorm',
        }

        self.tokenizer = 'default'
