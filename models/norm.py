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
"""Common normalization functions used by multiple models."""

import mtk_quantization
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) class.

    This class implements the RMSNorm layer normalization technique.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        eps (float): A small value to avoid division by zero.
        weight (torch.nn.Parameter): The learnable weights of the normalization layer.
        variance_epsilon (float): A small value to avoid division by zero.
        mul (mtk_quantization.pytorch.functional.Mul): Multiplication function from mtk_quantization.
    """

    def __init__(self, hidden_size, eps=1e-6):
        """Initializes the RMSNorm class.

        Args:
            hidden_size (int): The size of the hidden layer.
            eps (float, optional): A small value to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.mul = mtk_quantization.pytorch.functional.Mul()

    def forward(self, hidden_states):
        """Forward pass for the RMSNorm layer.

        Args:
            hidden_states (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.mul(self.weight, hidden_states)


class VivoInfiniKVNorm(nn.Module):
    """KV Norm used in Infini Attention."""

    def __init__(self):
        """Initialize the Infini Attention KV Norm."""
        super().__init__()

    def _calculate_variance(self, input_tensor):
        mean = torch.mean(input_tensor, dim=(1, 2, 3), keepdim=True)
        squared_diffs = (input_tensor - mean) ** 2
        count = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
        return torch.sum(squared_diffs, dim=(1, 2, 3), keepdim=False) / count

    def forward(self, hidden_states):
        """Forward pass for KV Norm."""
        temp = hidden_states - torch.mean(hidden_states, (1, 2, 3))
        return temp / torch.sqrt(self._calculate_variance(hidden_states) + 1e-5)


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm class.

    This class defines a Root Mean Square Layer Normalization (RMSNorm) for the Gemma model.
    """

    def __init__(self, hidden_size, eps=1e-6, with_scale=True, decompose=False):
        """Initializes the GemmaRMSNorm class.

        Args:
            hidden_size (int): Size of the hidden layer.
            eps (float, optional): Epsilon value for numerical stability. Default is 1e-6.
            with_scale (bool, optional): Whether to scale the RMSNorm output. Defaults to True.
            decompose (bool, optional): Whether to decompose the RMSNorm.
        """
        super().__init__()
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.mul = mtk_quantization.pytorch.functional.Mul()
        self.variance_epsilon = eps
        self.decompose = decompose

    def forward(self, hidden_states):
        """Forward pass for the GemmaRMSNorm class.

        Args:
            hidden_states (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        if self.decompose:
            hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)
        else:
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.with_scale:
            return self.mul(1.0 + self.weight, hidden_states)
        return hidden_states


LayerNorm = torch.nn.LayerNorm
