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
"""PyTorch Tail with Medusa heads."""

import mtk_quantization
import numpy as np
import torch
from torch import nn

from .modeling_common import Tail

np.random.seed(42)


class ResBlock(nn.Module):
    """Residual Block class.

    This class defines a residual block with a linear layer and element-wise operations.
    """

    def __init__(self, hidden_size):
        """Initializes the ResBlock class.

        Args:
            hidden_size (int): Size of the hidden layer.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.mul = mtk_quantization.pytorch.functional.Mul()
        self.add = mtk_quantization.pytorch.functional.Add()

    def forward(self, x):
        """Forward pass for the ResBlock class.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        y = self.linear(x)

        return self.add(x, self.mul(y, torch.sigmoid(y)))


class MedusaTail(Tail):
    """Medusa Tail class.

    This class extends the Tail class and includes Medusa heads for the model.
    """

    def __init__(self, config, chunk_idx, dtype=torch.float32, jit_trace=False, norm_class=None):
        """Initializes the MedusaTail class.

        Args:
            config: Configuration for the model.
            chunk_idx (int): Index of the chunk.
            dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            jit_trace (bool, optional): Whether to use JIT tracing. Default is False.
            norm_class (type, optional): Class for the normalization. Default is None.
            distribute_layers (bool, optional): Whether to distribute layers across devices. Default is True.
        """
        super().__init__(config, chunk_idx, dtype, jit_trace, norm_class)
        self.num_heads = self.config.medusa_num_heads
        self.num_layers = self.config.medusa_num_layers
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.config.hidden_size)] * self.num_layers),
                    nn.Linear(
                        self.config.hidden_size, self.config.vocab_size + self.config.lm_head_pad_size, bias=False
                    ),
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward_alt(self, hidden_states):
        """Alternative forward pass for the MedusaTail class.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.

        Returns:
            list: List of logits from each Medusa head.
        """
        if self.support_quant_stub:
            if self.distribute_layers:
                hidden_states = self.hidden_states(hidden_states).to(self.device_list[0])
            else:
                hidden_states = self.hidden_states(hidden_states)
        else:
            if self.distribute_layers:
                hidden_states = hidden_states.to(self.device_list[0])

        if self.jit_trace:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)

        logits = []
        for i in range(self.num_heads):
            logits.append(self.medusa_heads[i](hidden_states))

        return logits

    def _generate_default_state_dict_mapping(self):
        # state_dict_mapping should be a dict of dicts with length 1.
        # {
        #     internal_identifying_key: {internal_model_key: expected_state_dict_key},
        #     ...
        # }
        super()._generate_default_state_dict_mapping()
        # fmt: off
        for head_idx in range(self.num_heads):
            for layer_idx in range(self.num_layers):
                self.state_dict_mapping.update(
                    {
                        f'medusa_{layer_idx}_head_{head_idx}_linear_weight': {
                            f'medusa_heads.{head_idx}.{layer_idx}.linear.weight':
                            f'{head_idx}.{layer_idx}.linear.weight'
                        },
                        f'medusa_{layer_idx}_head_{head_idx}_linear_bias': {
                            f'medusa_heads.{head_idx}.{layer_idx}.linear.bias':
                            f'{head_idx}.{layer_idx}.linear.bias'
                        },
                        f'medusa_{layer_idx}_head_{head_idx}_weight': {
                            f'medusa_heads.{head_idx}.{layer_idx + 1}.weight':
                            f'{head_idx}.{layer_idx + 1}.weight'
                        },
                    }
                )
        # fmt: on

    def load_weights(self, state_dict, **kwargs):
        """Loads weights into the MedusaTail class.

        Args:
            state_dict (dict): State dictionary containing the weights.
            kwargs: Additional keyword arguments.

        Returns:
            self: The MedusaTail instance with loaded weights.
        """
        state_dict_keys = state_dict.keys()
        self, state_dict = super().load_weights(state_dict)
        medusa_prefix = self.prefixes[-1]

        # Determine the number of heads in the state_dict
        head_indices = set()
        for key in state_dict_keys:
            if key.startswith(medusa_prefix):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    head_indices.add(int(parts[1]))
        num_heads_in_state_dict = len(head_indices)

        # Remove unused heads from state_dict
        for head_idx in range(self.num_heads, num_heads_in_state_dict):
            for layer_idx in range(self.num_layers):
                state_dict.pop(f'{medusa_prefix}{head_idx}.{layer_idx}.linear.weight', None)
                state_dict.pop(f'{medusa_prefix}{head_idx}.{layer_idx}.linear.bias', None)
                state_dict.pop(f'{medusa_prefix}{head_idx}.{layer_idx + 1}.weight', None)

        if medusa_prefix != '':
            # Remove all remaining medusa weights
            for k in list(state_dict.keys()):
                if k.startswith(medusa_prefix):
                    state_dict.pop(k)

        return self
