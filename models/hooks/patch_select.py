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
"""Define the patch select method between a vision encoder and vision projector."""

from ..modeling_hook_base import BaseHook


class PatchSelect(BaseHook):
    """PatchSelect class that selects a patch from vision encoder output.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the PatchSelect.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the PatchSelect.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, image_features, **kwargs):
        """Forward pass.

        Args:
            image_features: The image features obtained from the output of a vision encoder.
            kwargs (dict, optional): Additional keyword arguments.
        """
        return image_features[:, 1:], kwargs
