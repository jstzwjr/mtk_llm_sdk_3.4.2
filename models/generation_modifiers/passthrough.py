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

from transformers.generation.logits_process import LogitsProcessor


class Passthrough(LogitsProcessor):
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
        self.config = config

    def __call__(self, input_ids, scores):
        """Forward pass that returns the inputs.

        Args:
            input_ids (torch.Tensor, optional): Input tokens. Defaults to None.
            scores (torch.Tensor, optional): Logits output from model

        Returns:
            The same inputs that were passed in.
        """
        return scores
