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
"""Define mlkits-related model wrpper for mtk_llm_sdk."""

import torch


class EncoderWrapper(torch.nn.Module):
    """An E2E encoder module wrapper for pipeline."""

    def __init__(self, pipeline):
        """Initialize E2E encoder."""
        super().__init__()
        self.encoder = pipeline.encoder
        self.pre_projector = pipeline.pre_projector_hook
        self.projector = pipeline.projector

    def forward(self, x):
        """Forward E2E encoder."""
        return self.projector(self.pre_projector.forward(self.encoder.forward(x))[0])
