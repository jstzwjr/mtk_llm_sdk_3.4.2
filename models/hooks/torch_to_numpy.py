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
"""Define a function to convert torch tensors to numpy arrays."""

import torch
from transformers.feature_extraction_utils import BatchFeature

from ...utils import logger
from ..modeling_hook_base import BaseHook


class TorchToNumpy(BaseHook):
    """TorchToNumpy class that converts torch tensors to numpy arrays.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the TorchToNumpy.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the TorchToNumpy.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, inp, **kwargs):
        """Forward pass.

        Args:
            inp: The torch tensor to convert into a numpy array.
            kwargs (dict, optional): Additional keyword arguments.
        """
        batch_feature = isinstance(inp, BatchFeature)

        tensor = inp['input_features'] if batch_feature else inp

        if isinstance(tensor, (list, tuple)):
            for tsr in tensor:
                if not isinstance(tsr, torch.Tensor):
                    logger.error(f'Expected torch tensor but got type: {type(tsr)}.', err=TypeError)
            return [tsr.detach().cpu().numpy() for tsr in tensor], kwargs
        if not isinstance(tensor, torch.Tensor):
            logger.error(f'Expected torch tensor but got type: {type(tensor)}.', err=TypeError)

        out = tensor.detach().cpu().numpy()

        if batch_feature:
            inp['input_features'] = out
            return inp, kwargs
        return out, kwargs
