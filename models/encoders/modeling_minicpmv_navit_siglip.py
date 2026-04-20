# Copyright (C) 2024 MediaTek Inc. All rights reserved.
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
"""Define MinicpmVNaViTSigLIP model class."""

import os

import numpy as np
import torch

from ...utils import image_utils, logger, utils
from ..preprocessors.preprocessor_minicpmv_navit_siglip import MinicpmVNavitSigLIPImageProcessor
from .configuration_minicpmv_navit_siglip import MinicpmVNavitSigLIPConfig
from .modeling_siglip import SigLIPVisionEmbeddings, SigLIPVisionTransformer


class MinicpmVNavitSigLIPVisionEmbeddings(SigLIPVisionEmbeddings):
    """MinicpmVNavitSigLIPVisionEmbeddings class.

    This class extends the nn.Module class and initializes the MinicpmVNavitSigLIPVisionEmbeddings
    with specific configurations.

    Attributes:
        config (MinicpmVNavitSigLIPConfig): The configuration for the MinicpmVNavitSigLIPVisionEmbeddings.
        hidden_size (int): The hidden size.
        image_size (int): The image size.
        patch_size (int): The patch size.
        patch_embedding (nn.Conv2d): The patch embedding layer.
        jit_trace (bool): Whether to use JIT tracing.
        num_patches (int): The number of patches.
        num_positions (int): The number of positions.
        position_embedding (nn.Embedding): The position embedding layer.
        position_ids (torch.Tensor): The position IDs.

    Methods:
        forward: Forward pass for the MinicpmVNavitSigLIPVisionEmbeddings.
    """

    def __init__(self, config: MinicpmVNavitSigLIPConfig, jit_trace=False):
        """Initializes the MinicpmVNavitSigLIPVisionEmbeddings class.

        Args:
            config (MinicpmVNavitSigLIPConfig): The configuration for the MinicpmVNavitSigLIPVisionEmbeddings.
            jit_trace (bool, optional): Whether to use JIT tracing. Defaults to False.
        """
        super().__init__(config)


class MinicpmVNavitSigLIPVisionTransformer(SigLIPVisionTransformer):
    """MinicpmVNavitSigLIPVisionTransformer class.

    This class extends the SigLIPVisionTransformer class and initializes the MinicpmVNavitSigLIPVisionTransformer with
    specific configurations.

    Attributes:
        config (MinicpmVNavitSigLIPConfig): The configuration for the MinicpmVNavitSigLIPVisionTransformer.
        jit_trace (bool): Whether to use JIT tracing.
        dtype (torch.dtype): The data type. Defaults to torch.float32.
        embeddings (MinicpmVNavitSigLIPVisionEmbeddings): The embeddings layer.
        post_layernorm (nn.LayerNorm): The post layer normalization layer.
        image_processor (MinicpmVNavitSigLIPImageProcessor): The image processor.

    Methods:
        forward: Forward pass for the MinicpmVNavitSigLIPVisionTransformer.
    """

    def __init__(
        self,
        config: MinicpmVNavitSigLIPConfig,
        lora,
        num_layers,
        first_layer_idx,
        chunk_idx,
        dtype=torch.float32,
        jit_trace=False,
        parallel_lora=False,
        distribute_layers=True,
        **kwargs,
    ):
        """Initializes the SigLIPVisionTransformer class.

        Args:
            config (MinicpmVNavitSigLIPConfig): The configuration for the MinicpmVNavitSigLIP model.
            lora (LoRA): The lora object.
            num_layers (int): The number of vpm.encoder layers for the current chunk.
            first_layer_idx (int): The index of the first vpm.encoder layer in the current chunk.
            chunk_idx (int): The chunk index of the current chunk.
            dtype (torch.dtype): The default dtype to use for this chunk.
            jit_trace (bool): Flag to determine if model is to be run as part of JIT tracing or not.
            parallel_lora (bool): Flag to determine if parallel lora is used in the current chunk.
            distribute_layers (bool): Flag to determine if vpm.encoder layers should be evenly
                distributed among available GPUs or not.
            kwargs (dict): Additional keyword arguments.
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
        if self.chunk_idx == 0:
            self.embeddings = MinicpmVNavitSigLIPVisionEmbeddings(config, jit_trace=jit_trace)

        self.image_processor = MinicpmVNavitSigLIPImageProcessor()

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        state_dict_mapping = {}
        if self.chunk_idx == 0:
            state_dict_mapping = {
                'position_embedding_weight': {
                    'embeddings.position_embedding.weight': 'vpm.embeddings.position_embedding.weight'
                },
                'patch_embedding_weight': {
                    'embeddings.patch_embedding.weight': 'vpm.embeddings.patch_embedding.weight'
                },
                'patch_embedding_bias': {'embeddings.patch_embedding.bias': 'vpm.embeddings.patch_embedding.bias'},
            }

        # fmt: off
        for inner_layer_idx, outer_layer_idx in enumerate(
            range(self.first_layer_idx, self.first_layer_idx + self.num_layers)
        ):
            state_dict_mapping.update(
                {
                    f'{outer_layer_idx}_q_weight': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.weight'
                    },
                    f'{outer_layer_idx}_q_bias': {
                        f'layers.{inner_layer_idx}.self_attn.q_proj.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["q"]}.bias'
                    },
                    f'{outer_layer_idx}_k_weight': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.weight'
                    },
                    f'{outer_layer_idx}_k_bias': {
                        f'layers.{inner_layer_idx}.self_attn.k_proj.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["k"]}.bias'
                    },
                    f'{outer_layer_idx}_v_weight': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.weight'
                    },
                    f'{outer_layer_idx}_v_bias': {
                        f'layers.{inner_layer_idx}.self_attn.v_proj.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["v"]}.bias'
                    },
                    f'{outer_layer_idx}_o_weight': {
                        f'layers.{inner_layer_idx}.self_attn.out_proj.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.weight'
                    },
                    f'{outer_layer_idx}_o_bias': {
                        f'layers.{inner_layer_idx}.self_attn.out_proj.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["attn"]["name"]}.{self.fc_names["attn"]["o"]}.bias'
                    },
                    f'{outer_layer_idx}_fc1_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc1.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc1"]}.weight'
                    },
                    f'{outer_layer_idx}_fc1_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc1.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc1"]}.bias'
                    },
                    f'{outer_layer_idx}_fc2_weight': {
                        f'layers.{inner_layer_idx}.mlp.fc2.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc2"]}.weight'
                    },
                    f'{outer_layer_idx}_fc2_bias': {
                        f'layers.{inner_layer_idx}.mlp.fc2.bias':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.fc_names["mlp"]["name"]}.{self.fc_names["mlp"]["fc2"]}.bias'
                    },
                    f'{outer_layer_idx}_1_layer_norm_weight': {
                        f'layers.{inner_layer_idx}.layer_norm1.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm1"]}.weight'
                    },
                    f'{outer_layer_idx}_2_layer_norm_weight': {
                        f'layers.{inner_layer_idx}.layer_norm2.weight':
                        f'vpm.encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm2"]}.weight'
                    },
                }
            )
            if self.config.norm == 'LayerNorm':
                state_dict_mapping.update(
                    {
                        f'{outer_layer_idx}_1_layer_norm_bias': {
                            f'layers.{inner_layer_idx}.layer_norm1.bias':
                            f'vpm.encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm1"]}.bias'
                        },
                        f'{outer_layer_idx}_2_layer_norm_bias': {
                            f'layers.{inner_layer_idx}.layer_norm2.bias':
                            f'vpm.encoder.layers.{outer_layer_idx}.{self.norm_names["layernorm2"]}.bias'
                        },
                    }
                )
            # fmt: on
            if self.parallel_lora and self.with_lora[inner_layer_idx]:
                state_dict_mapping.update(self.generate_default_lora_state_dict_mapping())

        if self.chunk_idx == self.config.num_hidden_layers - 1:
            state_dict_mapping.update(
                {
                    'post_norm_weight': {'post_layernorm.weight': f'vpm.{self.norm_names["post"]}.weight'},
                    'post_norm_bias': {'post_layernorm.bias': f'vpm.{self.norm_names["post"]}.bias'},
                }
            )

        self.state_dict_mapping = state_dict_mapping

    def inner_encoder_hook(self, images, kwargs):
        """Inner encoder hook before forwarding encoder.

        This function calculates and sets the position IDs for patches in a batch of images. The position IDs are used
        to identify the relative positions of patches within each image. The function also creates an attention mask
        for the patches based on the target sizes of the images.

        Parameters:
            images: torch.Tensor
                A tensor of shape (batch_size, channels, height, width) representing the batch of images.
            kwargs (dict, optional): Additional keyword arguments.
        """
        if self.chunk_idx != 0:
            return

        tgt_sizes = torch.tensor(kwargs.get('tgt_sizes'))
        if tgt_sizes is None:
            logger.error('tgt_sizes must be passed when forwarding MinicpmVNavitSigLIPVisionTransformer')

        batch_size, max_im_h, max_im_w, _channel = images.shape

        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        patch_attn_mask = torch.zeros((batch_size, 1, max_patches), dtype=torch.bool, device=images.device)
        for i in range(batch_size):
            patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.config.patch_size, max_im_w // self.config.patch_size
        num_patches_per_side = self.config.image_size // self.config.patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attn_mask):
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]

            fractional_coords_h = torch.arange(0, 1 - 1e-6, (1 / nb_patches_h.item()))
            fractional_coords_w = torch.arange(0, 1 - 1e-6, (1 / nb_patches_w.item()))

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        self.embeddings.position_ids = position_ids.to(self.device_list[0])
        self.embeddings.vision_pos_emb = self.embeddings.position_embedding(self.embeddings.position_ids)

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        if self.chunk_idx == 0:
            example_inputs = torch.randn(
                self.config.ptq_image_batch,
                self.config.ptq_image_height,
                self.config.ptq_image_width,
                self.config.num_channels,
                device='cpu',
                dtype=torch.float32,
            )
            tgt_sizes = np.array(
                [
                    [
                        self.config.ptq_image_height // self.config.patch_size,
                        self.config.ptq_image_width // self.config.patch_size,
                    ]
                ]
            )
            self.inner_encoder_hook(example_inputs, kwargs={'tgt_sizes': tgt_sizes})
            return example_inputs

        feature_size = int(
            self.config.ptq_image_height
            // self.config.patch_size
            * self.config.ptq_image_width
            // self.config.patch_size
        )
        return torch.randn(1, feature_size, self.config.hidden_size, device='cpu', dtype=torch.float32)

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
        if self.chunk_idx == 0:
            if self.config.do_reshape_by_patch:
                patch_num = self.config.ptq_image_height * self.config.ptq_image_width // self.config.patch_size
                input_shapes = [
                    [self.config.ptq_image_batch, self.config.patch_size, patch_num, self.config.num_channels]
                ]
            else:
                input_shapes = [
                    [
                        self.config.ptq_image_batch,
                        self.config.ptq_image_height,
                        self.config.ptq_image_width,
                        self.config.num_channels,
                    ]
                ]
        else:
            feature_size = int(
                self.config.ptq_image_height
                // self.config.patch_size
                * self.config.ptq_image_width
                // self.config.patch_size
            )
            input_shapes = [[self.config.ptq_image_batch, feature_size, self.config.hidden_size]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        if self.config.do_reshape_by_patch:
                            yield [
                                np.random.rand(
                                    self.config.ptq_image_batch,
                                    self.config.patch_size,
                                    patch_num,
                                    self.config.num_channels,
                                ).astype(np.float32)
                            ]
                        else:
                            yield [
                                np.random.rand(
                                    self.config.ptq_image_batch,
                                    self.config.ptq_image_height,
                                    self.config.ptq_image_width,
                                    self.config.num_channels,
                                ).astype(np.float32)
                            ]
                    else:
                        yield [
                            np.random.rand(self.config.ptq_image_batch, feature_size, self.config.hidden_size).astype(
                                np.float32
                            )
                        ]
        else:

            def calib_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.calibration_dataset, 'encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        if args.evaluation_dataset == 'fake':

            def eval_data_gen():
                for _ in range(10):
                    if self.chunk_idx == 0:
                        if self.config.do_reshape_by_patch:
                            yield [
                                np.random.rand(
                                    self.config.ptq_image_batch,
                                    self.config.patch_size,
                                    patch_num,
                                    self.config.num_channels,
                                ).astype(np.float32)
                            ]
                        else:
                            yield [
                                np.random.rand(
                                    self.config.ptq_image_batch,
                                    self.config.ptq_image_height,
                                    self.config.ptq_image_width,
                                    self.config.num_channels,
                                ).astype(np.float32)
                            ]
                    else:
                        yield [
                            np.random.rand(self.config.ptq_image_batch, feature_size, self.config.hidden_size).astype(
                                np.float32
                            )
                        ]
        else:

            def eval_data_gen():
                for f in utils.get_sorted_path_list(
                    os.path.join(args.evaluation_dataset, 'vpm.encoder', f'chunk_{self.chunk_idx}'), '.npz', sep='-'
                ):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen

    @image_utils.NHWCWrapper
    def forward(self, pixel_values):
        """Forward pass for the MinicpmVNavitSigLIPVisionTransformer.

        Args:
            pixel_values (torch.Tensor): The input pixel values.
            use_last_hidden_state (bool, optional): Whether to use the last hidden state. Defaults to False.

        Returns:
            tuple: The hidden states and encoder outputs. If use_last_hidden_state is True, also returns the
                last hidden states.
        """
        hidden_states = pixel_values.to(self.device_list[0])
        if self.first_layer_idx == 0:
            hidden_states = self.embeddings(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states.to(self.device_list[idx]))

        if self.chunk_idx == self.config.num_hidden_layers - 1:
            hidden_states = self.post_layernorm(hidden_states)

        return hidden_states
