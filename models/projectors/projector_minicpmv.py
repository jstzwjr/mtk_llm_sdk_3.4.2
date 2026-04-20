# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""Define Minicpmv vision projector class."""

import os
from functools import partial

import numpy as np
import torch
from torch import nn

from ...utils import logger, utils
from ..modeling_base import BaseProjector
from .configuration_projector_minicpmv import MinicpmVProjectorConfig


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """Generate a 2D sine-cosine positional embedding for an image.

    Args:
        embed_dim : int
            The dimension of the embedding.
        image_size : int or tuple
            The size of the image. If an integer is provided, it is assumed to be the size of both height and width.
            If a tuple is provided, it should be in the format (height, width).

    Returns:
        np.ndarray
            A numpy array containing the 2D sine-cosine positional embedding.
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate a 2D sine-cosine positional embedding from a grid.

    Args:
        embed_dim : int
            The dimension of the embedding. Must be an even number.
        grid : np.ndarray
            A 2D grid array with shape (2, height, width), where the first dimension represents the width and height
            coordinates respectively.

    Returns:
        np.ndarray
            A numpy array containing the 2D sine-cosine positional embedding with shape (height, width, embed_dim).
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H, W, D/2)

    return np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate a 1D sine-cosine positional embedding from a grid position.

    Args:
        embed_dim : int
            The dimension of the embedding. Must be an even number.
        pos : np.ndarray
            A 1D array representing the position grid.

    Returns:
        np.ndarray
            A numpy array containing the 1D sine-cosine positional embedding with shape (height, width, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)


class MinicpmVProjector(BaseProjector):
    """Projector of MiniCPMV.

    A 2D perceiver-resampler network with one cross-attention layer using
    (grid_size**2) learnable queries and 2D sine-cosine positional embeddings.

    Args:
        grid_size : int
            The size of the grid for the learnable queries. The total number of queries will be grid_size**2.
        embed_dim : int
            The dimension of the embeddings.
        num_heads : int
            The number of attention heads in the multi-head attention mechanism.
        kv_dim : int, optional
            The dimension of the key and value projections. If None or equal to embed_dim, no projection is applied.
            Default is None.
        use_resampler_embed : bool, optional
            If True, use sine-cosine positional embeddings for the resampler. Default is True.
        max_size : tuple, optional
            The maximum size of the positional embedding grid. Default is (70, 70).
    """

    def __init__(
        self,
        config: MinicpmVProjectorConfig,
        dtype=torch.float32,
        jit_trace=False,
        max_size=(70, 70),
    ):
        """Init MinicpmV projector."""
        super().__init__(config=config, jit_trace=jit_trace, dtype=dtype)
        self.num_queries = config.grid_size**2
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.use_resampler_embed = config.use_resampler_embed
        self.max_size = max_size
        self.mlp_depth = 1

        if self.use_resampler_embed:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(get_2d_sincos_pos_embed(config.embed_dim, config.grid_size)).float()
            ).requires_grad_(False)
        self.query = nn.Parameter(torch.zeros(self.num_queries, config.embed_dim))

        if config.kv_dim is not None and config.kv_dim != config.embed_dim:
            self.kv_proj = nn.Linear(config.kv_dim, config.embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads)
        self.ln_q = norm_layer(config.embed_dim)
        self.ln_kv = norm_layer(config.embed_dim)

        self.ln_post = norm_layer(config.embed_dim)
        self.projector = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size):
        """Set the 2D positional embedding cache for the given maximum size."""
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self.register_buffer('pos_embed_k', pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, batch_size):
        """Adjust the positional embedding cache based on the target sizes of the images in the batch."""
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

        self._adjust_pos(tgt_sizes, batch_size)

    def _adjust_pos(self, tgt_sizes, batch_size):
        """Adjust the positional embeddings for the current batch based on the target sizes."""
        pos_embed = []
        for i in range(batch_size):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed_k[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D
        self.pos_embed_batch = pos_embed.to(self.device_list[0])

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {
            'query': {'query': 'resampler.query'},
            'projector': {'projector.weight': 'resampler.proj'},
            'kv_proj_weight': {'kv_proj.weight': 'resampler.kv_proj.weight'},
            'attn_in_proj_weight': {'attn.in_proj_weight': 'resampler.attn.in_proj_weight'},
            'attn_in_proj_bias': {'attn.in_proj_bias': 'resampler.attn.in_proj_bias'},
            'attn_out_proj_weight': {'attn.out_proj.weight': 'resampler.attn.out_proj.weight'},
            'attn_out_proj_bias': {'attn.out_proj.bias': 'resampler.attn.out_proj.bias'},
            'ln_q_weight': {'ln_q.weight': 'resampler.ln_q.weight'},
            'ln_q_bias': {'ln_q.bias': 'resampler.ln_q.bias'},
            'ln_kv_weight': {'ln_kv.weight': 'resampler.ln_kv.weight'},
            'ln_kv_bias': {'ln_kv.bias': 'resampler.ln_kv.bias'},
            'ln_post_weight': {'ln_post.weight': 'resampler.ln_post.weight'},
            'ln_post_bias': {'ln_post.bias': 'resampler.ln_post.bias'},
        }

        self.state_dict_mapping = state_dict_mapping

    def inner_projector_hook(self, inputs, kwargs):
        """Inner projector hook before forwarding projector.

        This function calculates and sets the position IDs for patches in a batch of images. The position IDs are used
        to identify the relative positions of patches within each image. The function also creates an attention mask
        for the patches based on the target sizes of the images.

        Parameters:
            inputs: torch.Tensor
                A tensor of shape (batch_size, feature_size, hidden_size).
            kwargs (dict, optional): Additional keyword arguments.
        """
        batch_size = inputs.shape[0]
        tgt_sizes = torch.tensor(kwargs.get('tgt_sizes'))
        if tgt_sizes is None:
            logger.error('tgt_sizes must be passed when forwarding MinicpmVProjector')

        self._adjust_pos_cache(tgt_sizes, batch_size)

    def load_weights(self, state_dict):
        """Load model weights.

        Args:
            state_dict: model state_dict.
        """
        logger.debug('Enter MLPGeluProjector load_weights')
        if state_dict is None:
            logger.error('state_dict cannot be None', err=ValueError)

        self.device_list = []
        prefixes = ['']
        weights_to_load = {}
        state_dict_keys = list(state_dict.keys())
        missing_keys = []

        dtype = self.dtype

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

            # Check if key with all found prefixes directly matches the state_dict key
            for pre in prefixes:
                key_to_test = pre + external_key
                if key_to_test in state_dict_keys:
                    logger.debug(f'Found {internal_key} weight using prefix, state_dict key={key_to_test}')
                    weights_to_load.update({model_key: state_dict.pop(key_to_test).to(dtype)})
                    found = True
                    state_dict_keys.remove(key_to_test)
                    break

            if not found:
                for k in state_dict_keys:
                    if k.endswith(external_key):
                        logger.debug(
                            f'Found {internal_key} weight using iteration, state_dict key={k}. '
                            f'Adding prefix: {k[: -len(external_key)]}'
                        )
                        prefixes.append(k[: -len(external_key)])
                        weights_to_load.update({model_key: state_dict.pop(k).to(dtype)})
                        found = True
                        state_dict_keys.remove(k)
                        break

            if not found and internal_key.endswith('_bias'):
                # Bias key not found, default to zeros
                # Default shapes:
                # All: projection dim
                logger.debug(f'Init bias {internal_key} to zeros using shape={self.projection_dim}')
                weights_to_load.update({model_key: torch.zeros(self.projection_dim, dtype=dtype)})
                continue

            if not found:
                logger.debug(f'Cannot find {internal_key} weight')
                missing_keys.append((internal_key, external_key))

        if len(missing_keys) > 0:
            for internal_key, external_key in missing_keys:
                logger.warning(f'Unable to find {internal_key} weight in state_dict. Expected subkey: {external_key}')
            logger.info(f'state dict keys for reference: {state_dict_keys}')
            logger.error('Please modify your state_dict keys according to the expected subkeys.', err=KeyError)

        if len(prefixes) > 2:
            logger.warning(
                f'More than 1 prefix found (found {prefixes[1:]}). '
                'This is unexpected and will likely cause errors during weight loading.'
            )

        num_gpu = torch.cuda.device_count()
        if num_gpu == 0 or self.jit_trace:
            self.device_list = ['cpu' for _ in range(self.mlp_depth)]
        else:
            if self.distribute_layers:
                master_gpu_ids = sorted(
                    list(range(num_gpu)) * (self.mlp_depth // num_gpu)
                    + (list(range(num_gpu))[: self.mlp_depth % num_gpu] if self.mlp_depth % num_gpu != 0 else [])
                )
            else:
                master_gpu_ids = [os.getenv('LOCAL_RANK', 0)] * self.mlp_depth
            self.device_list = [f'cuda:{x}' for x in master_gpu_ids]

        if weights_to_load.keys() != self.state_dict().keys():
            weights_to_load_only_keys = [x for x in weights_to_load if x not in self.state_dict()]
            model_only_keys = [x for x in self.state_dict() if x not in weights_to_load and 'lora' not in x]
            if model_only_keys != [] or weights_to_load_only_keys != []:
                logger.error(
                    f"model state dict keys don't match with state_dict to load into model.\n"
                    f'Model only keys:{model_only_keys}\nstate_dict only keys:{weights_to_load_only_keys}'
                )

        weights_to_load['projector.weight'] = weights_to_load['projector.weight'].T
        self.load_state_dict(weights_to_load, strict=False)

        self.to(self.device_list[0])

        self.eval()

        return self, state_dict

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        feature_size = int(
            self.config.ptq_image_height
            // self.config.patch_size
            * self.config.ptq_image_width
            // self.config.patch_size
        )
        example_inputs = torch.randn(
            self.config.ptq_image_batch, feature_size, self.config.kv_dim, device='cpu', dtype=torch.float32
        )
        tgt_sizes = []
        for _i in range(self.config.ptq_image_batch):
            tgt_sizes.append(
                np.array(
                    (
                        self.config.ptq_image_height // self.config.patch_size,
                        self.config.ptq_image_width // self.config.patch_size,
                    )
                )
            )
        tgt_sizes = np.vstack(tgt_sizes)
        self.inner_projector_hook(example_inputs, kwargs={'tgt_sizes': tgt_sizes})
        return example_inputs

    def get_ptq_inputs(self, args, **kwargs):
        """Gets inputs for post-training quantization (PTQ).

        Args:
            args (Namespace): Arguments for PTQ.
            exp_name (str): Experiment name.
            kwargs: Additional keyword arguments.

        Returns:
            tuple: Tuple containing input shapes, input value ranges, calibration data generator,
            and evaluation data generator.
        """
        feature_size = int(
            self.config.ptq_image_height
            // self.config.patch_size
            * self.config.ptq_image_width
            // self.config.patch_size
        )
        input_shapes = [[self.config.ptq_image_batch, feature_size, self.config.kv_dim]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    yield [
                        np.random.rand(self.config.ptq_image_batch, feature_size, self.config.kv_dim).astype(np.float32)
                    ]
        else:

            def calib_data_gen():
                encoder_calib_dir = os.path.join(args.calibration_dataset, 'encoder')
                encoder_chunk_dirs = [os.path.join(encoder_calib_dir, x) for x in os.listdir(encoder_calib_dir)]
                projector_calib_dir = sorted(encoder_chunk_dirs, key=lambda f: int(f.rsplit('_', 1)[1]))[-1]
                for f in utils.get_sorted_path_list(projector_calib_dir, '.npz', sep='-'):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for _ in range(10):
                    yield [
                        np.random.rand(self.config.ptq_image_batch, feature_size, self.config.kv_dim).astype(np.float32)
                    ]
        else:

            def eval_data_gen():
                encoder_eval_dir = os.path.join(args.evaluation_dataset, 'encoder')
                encoder_chunk_dirs = [os.path.join(encoder_eval_dir, x) for x in os.listdir(encoder_eval_dir)]
                projector_eval_dir = sorted(encoder_chunk_dirs, key=lambda f: int(f.rsplit('_', 1)[1]))[-1]
                for f in utils.get_sorted_path_list(projector_eval_dir, '.npz', sep='-'):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    def forward(self, x):
        """Forward pass of the resampler network."""
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        n = x.shape[1]
        q = self.ln_q(self.query)

        if self.use_resampler_embed:
            out = self.attn(self._repeat(q, n) + self.pos_embed, x + self.pos_embed_batch, x, attn_mask=None)[0]
        else:
            out = self.attn(self._repeat(q, n), x + self.pos_embed_batch, x, attn_mask=None)[0]

        x = out.permute(1, 0, 2)
        x = self.ln_post(x)
        return self.projector(x)

    def _repeat(self, query, n):
        """Repeat the query tensor N times along the batch dimension."""
        return query.unsqueeze(1).repeat(1, n, 1)
