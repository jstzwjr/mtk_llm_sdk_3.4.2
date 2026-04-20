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
"""Define a function to convert numpy arrays to torch tensors."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from ...utils import logger
from .numpy_to_torch import NumpyToTorch


class PadPhi3V(NumpyToTorch):
    """PadPhi3V class that extends NumpyToTorch.

    This class converts numpy arrays to torch tensors and pad batch size to 17 at
    quantized pipeline.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the PadPhi3V.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the PadPhi3V.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, inp, **kwargs):
        """Forward pass.

        Args:
            inp: The numpy array to convert into a torch tensor.
            kwargs (dict, optional): Additional keyword arguments.
        """
        batch_feature = isinstance(inp, BatchFeature)
        pipeline_type = kwargs.get('pipeline_type')
        out, kwargs = super().forward(inp, **kwargs)
        if pipeline_type == 'float':
            arr = out['input_features'] if batch_feature else out
            batch_size = arr.shape[0]
            kwargs.update({'phi3v_original_batch_size': batch_size})
            return out, kwargs

        arr = out['input_features'] if batch_feature else out
        batch_size = arr.shape[0]
        kwargs.update({'phi3v_original_batch_size': batch_size})
        batch_pad = 17 - batch_size
        if batch_size < 17:  # Need to pad Batch dimenstion to 17
            zero_batch = torch.zeros(batch_pad, arr.shape[1], arr.shape[2], arr.shape[3], dtype=arr.dtype)
            arr = torch.cat([arr, zero_batch], dim=0)
            logger.debug(f'Pad image to {arr.shape}')
        arr = arr.cpu().numpy()
        if batch_feature:
            inp['input_features'] = arr
            return inp, kwargs
        return arr, kwargs
