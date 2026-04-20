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
"""Define InternVL2 vision projector class."""

import os

import numpy as np
import torch
from torch import nn

from ...utils import logger, utils
from ..activations import TorchGeluApproximate
from ..modeling_base import BaseProjector
from .configuration_projector_internvl2 import InternVL2ProjectorConfig


class InternVL2Projector(BaseProjector):
    """InternVL2 Vision Projector class.

    This class implements the vision projector using MLP+gelu.
    """

    def __init__(self, config: InternVL2ProjectorConfig, dtype=torch.float32, jit_trace=False):
        """Initialize MLPGeluProjector.

        Args:
            config: Projector model config.
            dtype: Data type to load model in.
            jit_trace: Dummy argument.
        """
        super().__init__(config=config, dtype=dtype, jit_trace=jit_trace)

        self.encoder_hidden_size = config.encoder_hidden_size
        self.hidden_size = config.hidden_size
        self.downsample_ratio = config.downsample_ratio

        self.projector = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(self.encoder_hidden_size * int(1 / self.downsample_ratio) ** 2, self.hidden_size),
            TorchGeluApproximate('tanh'),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.mlp_depth = 3

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        self.state_dict_mapping = {
            '0.weight': {'projector.0.weight': 'mlp1.0.weight'},
            '0.bias': {'projector.0.bias': 'mlp1.0.bias'},
            '1.weight': {'projector.1.weight': 'mlp1.1.weight'},
            '1.bias': {'projector.1.bias': 'mlp1.1.bias'},
            '3.weight': {'projector.3.weight': 'mlp1.3.weight'},
            '3.bias': {'projector.3.bias': 'mlp1.3.bias'},
        }

    def load_weights(self, state_dict):
        """Load model weights.

        Args:
            state_dict: model state_dict.
        """
        logger.debug('Enter InternVL2Projector load_weights')
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

        self.load_state_dict(weights_to_load, strict=False)

        self.projector.to(self.device_list[0])

        self.eval()

        return self, state_dict

    def forward(self, img_embedding):
        """Performs the forward pass to compute the vision transformer output.

        Args:
            img_embedding (torch.Tensor): The output image embeddings from vision encoder.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]: The hidden states and the projector outputs.
        """
        return self.projector(img_embedding.to(self.device_list[0]))

    def get_jit_trace_inputs(self):
        """Gets inputs for JIT tracing.

        Returns:
            torch.Tensor: Input tensor for JIT tracing.
        """
        feature_length = int(self.config.image_size / self.config.patch_size)
        feature_size = int(feature_length * feature_length)
        return torch.randn(
            1,
            feature_size // (int(1 / self.downsample_ratio) ** 2),
            self.encoder_hidden_size * (int(1 / self.downsample_ratio) ** 2),
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
        feature_length = int(self.config.image_size / self.config.patch_size)
        int(feature_length * feature_length)
        input_shapes = [
            [
                None,
                None,
                self.encoder_hidden_size * (int(1 / self.downsample_ratio) ** 2),
            ]
        ]
        input_value_ranges = [None]

        if args.calibration_dataset == 'fake':

            def calib_data_gen():
                for _ in range(10):
                    yield [
                        np.random.rand(
                            1, 1 + int(self.config.image_size / self.config.patch_size), self.config.hidden_size
                        ).astype(np.float32)
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
                        np.random.rand(
                            1, 1 + int(self.config.image_size / self.config.patch_size), self.config.hidden_size
                        ).astype(np.float32)
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
