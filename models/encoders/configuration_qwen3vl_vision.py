"""Qwen3-VL ViT Configuration."""

from ...utils import logger
from ..configuration_base import BaseVisionEncoderChunkConfig


class Qwen3VLVisionConfig(BaseVisionEncoderChunkConfig):
    """Configuration class for the Qwen3-VL Vision Transformer model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'qwen3_vl_vision')
        if self.model_type != 'qwen3_vl_vision':
            raise RuntimeError(f'Expected model_type to be qwen3_vl_vision but got {self.model_type} instead')

        self.depth = self.kwargs.pop('num_hidden_layers', 24)
        self.num_hidden_layers = self.depth
        self.embed_dim = self.kwargs.pop('embed_dim', 1024)
        self.intermediate_size = self.kwargs.pop('intermediate_size', 4096)
        self.hidden_size = self.kwargs.pop('hidden_size', 2560)
        self.hidden_act = self.kwargs.pop('hidden_act', 'gelu_pytorch_tanh')
        self.num_heads = self.kwargs.pop('num_heads', 16)
        self.num_attention_heads = self.num_heads
        self.in_channels = self.kwargs.pop('in_chans', 3)
        self.patch_size = self.kwargs.pop('patch_size', 16)
        self.spatial_merge_size = self.kwargs.pop('spatial_merge_size', 2)
        self.temporal_patch_size = self.kwargs.pop('temporal_patch_size', 2)
        self.use_conv2d_patch_embed = self.kwargs.pop('use_conv2d_patch_embed', False)
        self.out_hidden_size = self.kwargs.pop('out_hidden_size', 2560)
        self.num_position_embeddings = self.kwargs.pop('num_position_embeddings', 2304)
        self.deepstack_visual_indexes = self.kwargs.pop('deepstack_visual_indexes', [5, 11, 17])
        self.mask_value = self.kwargs.pop('mask_value', -10000)

        # For fixed shape PTQ
        self.image_resolution = self.kwargs.pop('image_resolution', None)
        self.preprocessor_config = self.kwargs.pop('preprocessor_config', None)

        self.projector_type = kwargs.pop('projector_type', 'patchmerger')

        self.fc_names = {
            'attn': {
                'name': 'attn',
                'q': 'q_proj',
                'k': 'k_proj',
                'v': 'v_proj',
                'o': 'proj',
            },
            'mlp': {'name': 'mlp', 'fc1': 'linear_fc1', 'fc2': 'linear_fc2'},
        }
        self.norm_names = {
            'layernorm1': 'norm1',
            'layernorm2': 'norm2',
        }

        if self.kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        logger.info(f'depth:                     {self.depth}')
        logger.info(f'embed_dim:                 {self.embed_dim}')
        logger.info(f'intermediate_size:         {self.intermediate_size}')
        logger.info(f'hidden_size (out):         {self.hidden_size}')
        logger.info(f'hidden_act:                {self.hidden_act}')
        logger.info(f'num_heads:                 {self.num_heads}')
        logger.info(f'in_channels:               {self.in_channels}')
        logger.info(f'patch_size:                {self.patch_size}')
        logger.info(f'spatial_merge_size:        {self.spatial_merge_size}')
        logger.info(f'temporal_patch_size:       {self.temporal_patch_size}')
        logger.info(f'num_position_embeddings:   {self.num_position_embeddings}')
        logger.info(f'deepstack_visual_indexes:  {self.deepstack_visual_indexes}')
