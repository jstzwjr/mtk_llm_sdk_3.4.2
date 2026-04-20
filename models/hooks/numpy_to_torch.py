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

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from ...utils import logger
from ..modeling_hook_base import BaseHook


class NumpyToTorch(BaseHook):
    """NumpyToTorch class that converts numpy arrays to torch tensors.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the NumpyToTorch.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the NumpyToTorch.

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
        logger.debug('Enter numpy_to_torch.')
        batch_feature = isinstance(inp, BatchFeature)
        logger.debug(f'Is batch_feature: {batch_feature}')
        arr = inp['input_features'] if batch_feature else inp

        if isinstance(arr, (list, tuple)):
            for a in arr:
                if not isinstance(a, np.ndarray):
                    logger.error(f'Expected numpy array but got type: {type(a)}.', err=TypeError)
            return [torch.from_numpy(a) for a in arr], kwargs
        if not isinstance(arr, np.ndarray):
            logger.error(f'Expected numpy array but got type: {type(arr)}.', err=TypeError)

        out = torch.from_numpy(arr)

        if batch_feature:
            inp['input_features'] = out
            return inp, kwargs
        return out, kwargs
