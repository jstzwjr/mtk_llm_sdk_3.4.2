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
"""Common activation functions used by multiple models."""

import torch
from torch import nn


class FastGelu(nn.Module):
    """Fast Gaussian Error Linear Unit (FastGelu) activation function class.

    This class implements the FastGelu activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FastGelu activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the FastGelu activation function.
        """
        return torch.nn.functional.gelu(x, approximate='tanh')


class QuickGelu(nn.Module):
    """Quick Gaussian Error Linear Unit (QuickGelu) activation function class.

    This class implements the QuickGelu activation function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the QuickGelu activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the QuickGelu activation function.
        """
        return x * torch.sigmoid(1.702 * x)


class TorchGelu(nn.Module):
    """Torch Gaussian Error Linear Unit (TorchGelu) activation function class.

    This class implements the TorchGelu activation function.

    Attributes:
        gelu (nn.Module): Gelu Activation function from Pytorch.
    """

    def __init__(self):
        """Initializes the TorchGelu class."""
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TorchGelu activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the TorchGelu activation function.
        """
        return self.gelu(x)


class TorchGeluApproximate(nn.Module):
    """PyTorch Gaussian Error Linear Unit activation function class with tanh approximation (TorchGeluApproximate).

    This class implements the TorchGeluApproximate activation function. Only supported in PyTorch Version
    1.12 onwards.
    """

    def __init__(self, approximate: str = 'none'):
        """Initializes the TorchGeluTanh class.

        Args:
            approximate (str): The approximation to use to GELU. Either `'tanh'` or `'none'`. Defaults to `'none'`.
        """
        super().__init__()
        self.gelu = torch.nn.GELU(approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TorchGeluTanh activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the TorchGeluTanh activation function.
        """
        return self.gelu(x)


class ReLU1p5(nn.Module):
    """RELU1p5 activation function."""

    def __init__(self):
        """Initializes the RELU1p5 module."""
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the RELU1p5 function."""
        x = self.relu(x)
        return x * torch.sqrt(x)
