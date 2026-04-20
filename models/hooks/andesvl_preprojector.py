# Copyright (C) 2025 MediaTek Inc. All rights reserved.
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
"""Define a function to reshape the AndesVL encoder output."""

from ..modeling_hook_base import BaseHook


class AndesVLPreprojector(BaseHook):
    """AndesVLPreprojector class that reshape the AndesVL encoder output.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the AndesVLPreprojector.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the AndesVLPreprojector.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, inp, **kwargs):
        """Forward pass."""
        inp = inp.view(-1, inp.shape[-1] * 4)  # (Len_img//4, H_vit*4)
        return inp, kwargs
