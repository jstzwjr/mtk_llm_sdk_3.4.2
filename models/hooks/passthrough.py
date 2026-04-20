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
"""Define of base passthrough class."""

from ..modeling_hook_base import BaseHook


class Passthrough(BaseHook):
    """Passthrough class that returns the inputs as outputs.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Passthrough.
        forward(inputs): Forward pass that returns the inputs.
    """

    def __init__(self, config):
        """Initialize the Passthrough.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, inputs, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            inputs (any): The inputs to be passed through.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            The same inputs that were passed in.
        """
        return inputs, kwargs
