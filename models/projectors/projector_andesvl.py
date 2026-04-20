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
"""Define andesvl vision projector class."""

import os

import numpy as np
import torch
from PIL import Image

from ...utils import logger, utils
from .configuration_projector_andesvl import AndesVLProjectorConfig
from .projector_internvl2 import InternVL2Projector


class InternVLNavitRopeProjector(InternVL2Projector):
    """InternVLNavitRopeProjector Vision Projector class.

    This class implements the vision projector using MLP+gelu.
    """

    def __init__(self, config: AndesVLProjectorConfig, dtype=torch.float32, jit_trace=False):
        """Initialize InternVLNavitRopeProjector.

        Args:
            config: Projector model config.
            dtype: Data type to load model in.
            jit_trace: Dummy argument.
        """
        super().__init__(config=config, dtype=dtype, jit_trace=jit_trace)

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        self.state_dict_mapping = {
            '0.weight': {'projector.0.weight': 'mlp.0.weight'},
            '0.bias': {'projector.0.bias': 'mlp.0.bias'},
            '1.weight': {'projector.1.weight': 'mlp.1.weight'},
            '1.bias': {'projector.1.bias': 'mlp.1.bias'},
            '3.weight': {'projector.3.weight': 'mlp.3.weight'},
            '3.bias': {'projector.3.bias': 'mlp.3.bias'},
        }

    def _calculate_ptq_fixed_shape_batch(self):
        from ..preprocessors.configuration_internvl_navit_rope import InternVLNavitRopePreprocessorConfig
        from ..preprocessors.preprocessor_internvl_navit_rope import InternVLNavitRopePreprocessor

        logger.debug('Enter InternVL-Navit-Rope _calculate_ptq_fixed_shape_batch.')
        preprocessor_config = InternVLNavitRopePreprocessorConfig(**self.config.preprocessor_config)
        processor = InternVLNavitRopePreprocessor(**preprocessor_config.get())
        image_hw = self.config.image_resolution
        if image_hw is None:
            logger.error('image_resolution in config must be set when PTQing InternVL-Navit-Rope.', err=ValueError)
        image = np.random.rand(image_hw[0], image_hw[1], 3)
        image = Image.fromarray(image.astype('uint8'))
        return_dict, _ = processor.preprocess([image])
        self.image_grid_hw = torch.tensor(return_dict['image_grid_hw'])
        logger.debug(f'InternVL-Navit-Rope PTQ image_grid_hw: {self.image_grid_hw}')

    def _calulate_ptq_attnmask_rotemb(self):
        from ..encoders.modeling_internvl_navit_rope import (
            precompute_cuseqlen,
            precompute_vision_rotary_embedding,
        )

        seq_length = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        self._cuseqlen = precompute_cuseqlen(self.image_grid_hw, seq_length=seq_length)

        head_dim = self.config.encoder_hidden_size // self.config.num_attention_heads
        rope_theta = self.config.rope_theta
        self._vision_rot_emb = precompute_vision_rotary_embedding(head_dim, rope_theta, self.image_grid_hw)

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        self._calculate_ptq_fixed_shape_batch()
        self._calulate_ptq_attnmask_rotemb()
        fixed_batch_size = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        return torch.randn(
            fixed_batch_size // 4,
            self.encoder_hidden_size * 4,
            device='cpu',
            dtype=torch.float32,
        )

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
        fixed_batch_size = (self.image_grid_hw[0][0] * self.image_grid_hw[0][1]).item()
        input_shapes = [[fixed_batch_size // 4, self.encoder_hidden_size * 4]]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    yield [np.random.rand(fixed_batch_size // 4, self.encoder_hidden_size * 4).astype(np.float32)]
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
                    yield [np.random.rand(fixed_batch_size // 4, self.encoder_hidden_size * 4).astype(np.float32)]
        else:

            def eval_data_gen():
                encoder_eval_dir = os.path.join(args.evaluation_dataset, 'encoder')
                encoder_chunk_dirs = [os.path.join(encoder_eval_dir, x) for x in os.listdir(encoder_eval_dir)]
                projector_eval_dir = sorted(encoder_chunk_dirs, key=lambda f: int(f.rsplit('_', 1)[1]))[-1]
                for f in utils.get_sorted_path_list(projector_eval_dir, '.npz', sep='-'):
                    data = np.load(f)
                    yield [data['hidden_states'].astype(np.float32)]

        return input_shapes, input_value_ranges, calib_data_gen, eval_data_gen
