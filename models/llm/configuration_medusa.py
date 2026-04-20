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
"""Configuration of Medusa."""

import torch

from ...utils import logger
from ..configuration_base import BaseConfig


class MedusaConfig(BaseConfig):
    """Configuration class for the Medusa model.

    This class inherits from BaseConfig and is used to set up the configuration
    for the Medusa model, including handling response, model type, and specific
    parameters related to attention mechanisms and embeddings.

    Attributes:
        model_type (str): The type of the model, expected to be 'medusa'.
        num_attention_heads (int): The number of attention heads, set to 1 to prevent division by zero.
        hidden_size (int): The size of the hidden layers.
        vocab_size (int): The size of the vocabulary.
        norm (str): The type of normalization used ('RMSNorm' or 'LayerNorm').
        embedding_key (str): The key for the embedding.
        tie_word_embeddings (bool): A flag indicating whether to tie word embeddings.
        medusa_num_layers (int): The number of Medusa layers.
        medusa_num_heads (int): The number of Medusa heads.
        medusa_choices (str): The choices for Medusa configuration.
        posterior_threshold (float): The threshold for validation of Medusa output.
        posterior_alpha (float): Another threshold hyperparameter, recommended to be the square
            root of posterior_threshold.
        medusa_len (int): The length of the Medusa structure.
        medusa_mask (torch.Tensor): The attention mask for Medusa.
        tree_indices (torch.Tensor): The tree indices for the Medusa structure.
        medusa_position_ids (torch.Tensor): The position IDs for the Medusa structure.
        retrieve_indices (torch.Tensor): The retrieval indices for Medusa structure verification.
    """

    def __init__(self, **kwargs):
        """Initializes the MedusaConfig class.

        Args:
            kwargs: Additional keyword arguments for configuration.

        Raises:
            RuntimeError: If the model_type is not 'medusa'.
            KeyError: If required configuration parameters are missing.
            ValueError: If invalid values are provided for certain parameters.
        """
        super().__init__(**kwargs)

        self.model_type = self.kwargs.pop('model_type', 'medusa')
        if self.model_type != 'medusa':
            logger.error(f'Expected model_type to be medusa but got {self.model_type} instead')

        self.num_attention_heads = 1  # To prevent div by 0 error when instantiate modeling_base

        self.hidden_size = self.kwargs.pop('hidden_size', None)
        if self.hidden_size is None:
            logger.error('hidden_size is required but missing from config.json', err=KeyError)

        self.vocab_size = self.kwargs.pop('vocab_size', None)
        if self.vocab_size is None:
            logger.error('vocab_size is required but missing from config.json', err=KeyError)

        self.norm = self.kwargs.pop('norm', 'RMSNorm')
        if self.norm not in ['RMSNorm', 'LayerNorm']:
            logger.error('norm must be one of: RMSNorm (default) or LayerNorm', err=ValueError)

        self.embedding_key = self.kwargs.pop('embedding_key', 'embed_tokens.weight')
        self.tie_word_embeddings = self.kwargs.pop('tie_word_embeddings', False)

        self.medusa_num_layers = self.kwargs.pop('medusa_num_layers', None)
        if self.medusa_num_layers is None:
            logger.error('medusa_num_layers is required but missing from config.json', err=KeyError)

        self.medusa_num_heads = self.kwargs.pop('medusa_num_heads', None)
        if self.medusa_num_heads is None:
            logger.error('medusa_num_heads is required but missing from config.json', err=KeyError)

        self.fc_names = {
            'tail': {'name': 'lm_head'},
        }

        self.medusa_choices = self.kwargs.pop('medusa_choices', 'mc_sim_7b_63')
        if self.medusa_choices not in [
            'mc_sim_7b_63',
            'vicuna_7b_stage2',
            'vicuna_7b_stage1',
            'vicuna_7b_stage1_ablation',
        ]:
            logger.error(
                'medusa_choices must be one of: mc_sim_7b_63 (default), vicuna_7b_stage1, vicuna_7b_stage1_ablation, '
                'vicuna_7b_stage2',
                err=ValueError,
            )

        if self.medusa_choices == 'mc_sim_7b_63':
            choices = [
                [0],
                [0, 0],
                [1],
                [0, 1],
                [2],
                [0, 0, 0],
                [1, 0],
                [0, 2],
                [3],
                [0, 3],
                [4],
                [0, 4],
                [2, 0],
                [0, 5],
                [0, 0, 1],
                [5],
                [0, 6],
                [6],
                [0, 7],
                [0, 1, 0],
                [1, 1],
                [7],
                [0, 8],
                [0, 0, 2],
                [3, 0],
                [0, 9],
                [8],
                [9],
                [1, 0, 0],
                [0, 2, 0],
                [1, 2],
                [0, 0, 3],
                [4, 0],
                [2, 1],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 0, 0],
                [0, 1, 1],
                [0, 0, 6],
                [0, 3, 0],
                [5, 0],
                [1, 3],
                [0, 0, 7],
                [0, 0, 8],
                [0, 0, 9],
                [6, 0],
                [0, 4, 0],
                [1, 4],
                [7, 0],
                [0, 1, 2],
                [2, 0, 0],
                [3, 1],
                [2, 2],
                [8, 0],
                [0, 5, 0],
                [1, 5],
                [1, 0, 1],
                [0, 2, 1],
                [9, 0],
                [0, 6, 0],
                [0, 0, 0, 1],
                [1, 6],
                [0, 7, 0],
            ]
        elif self.medusa_choices == 'vicuna_7b_stage2':
            choices = [
                (0,),
                (0, 0),
                (1,),
                (0, 1),
                (0, 0, 0),
                (1, 0),
                (2,),
                (0, 2),
                (0, 0, 1),
                (0, 3),
                (3,),
                (0, 1, 0),
                (2, 0),
                (4,),
                (0, 0, 2),
                (0, 4),
                (1, 1),
                (1, 0, 0),
                (0, 0, 0, 0),
                (5,),
                (0, 0, 3),
                (0, 5),
                (0, 2, 0),
                (3, 0),
                (0, 1, 1),
                (0, 6),
                (6,),
                (0, 7),
                (0, 0, 4),
                (4, 0),
                (1, 2),
                (0, 8),
                (7,),
                (0, 3, 0),
                (0, 0, 0, 1),
                (0, 0, 5),
                (2, 1),
                (0, 0, 6),
                (1, 0, 1),
                (0, 0, 1, 0),
                (2, 0, 0),
                (5, 0),
                (0, 9),
                (0, 1, 2),
                (8,),
                (0, 4, 0),
                (0, 2, 1),
                (1, 3),
                (0, 0, 7),
                (0, 0, 0, 2),
                (0, 0, 8),
                (1, 1, 0),
                (0, 1, 0, 0),
                (6, 0),
                (9,),
                (0, 1, 3),
                (0, 0, 0, 3),
                (1, 0, 2),
                (0, 5, 0),
                (3, 1),
                (0, 0, 2, 0),
                (7, 0),
                (1, 4),
            ]
        elif self.medusa_choices == 'vicuna_7b_stage1_ablation':
            choices = [
                (0,),
                (0, 0),
                (1,),
                (0, 0, 0),
                (0, 1),
                (1, 0),
                (2,),
                (0, 2),
                (0, 0, 1),
                (3,),
                (0, 3),
                (0, 1, 0),
                (2, 0),
                (0, 0, 2),
                (0, 4),
                (4,),
                (0, 0, 0, 0),
                (1, 0, 0),
                (1, 1),
                (0, 0, 3),
                (0, 2, 0),
                (0, 5),
                (5,),
                (3, 0),
                (0, 1, 1),
                (0, 6),
                (6,),
                (0, 0, 4),
                (1, 2),
                (0, 0, 0, 1),
                (4, 0),
                (0, 0, 5),
                (0, 7),
                (0, 8),
                (0, 3, 0),
                (0, 0, 1, 0),
                (1, 0, 1),
                (7,),
                (2, 0, 0),
                (0, 0, 6),
                (2, 1),
                (0, 1, 2),
                (5, 0),
                (0, 2, 1),
                (0, 9),
                (0, 0, 0, 2),
                (0, 4, 0),
                (8,),
                (1, 3),
                (0, 0, 7),
                (0, 1, 0, 0),
                (1, 1, 0),
                (6, 0),
                (9,),
                (0, 0, 8),
                (0, 0, 9),
                (0, 5, 0),
                (0, 0, 2, 0),
                (1, 0, 2),
                (0, 1, 3),
                (0, 0, 0, 3),
                (3, 0, 0),
                (3, 1),
            ]
        elif self.medusa_choices == 'vicuna_7b_stage1':
            choices = [
                (0,),
                (0, 0),
                (1,),
                (2,),
                (0, 1),
                (1, 0),
                (3,),
                (0, 2),
                (4,),
                (0, 0, 0),
                (0, 3),
                (5,),
                (2, 0),
                (0, 4),
                (6,),
                (0, 5),
                (1, 1),
                (0, 0, 1),
                (7,),
                (3, 0),
                (0, 6),
                (8,),
                (9,),
                (0, 1, 0),
                (0, 7),
                (0, 8),
                (4, 0),
                (0, 0, 2),
                (1, 2),
                (0, 9),
                (2, 1),
                (5, 0),
                (1, 0, 0),
                (0, 0, 3),
                (1, 3),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 4),
                (6, 0),
                (1, 4),
                (0, 0, 5),
                (2, 2),
                (0, 3, 0),
                (3, 1),
                (0, 0, 6),
                (7, 0),
                (1, 5),
                (1, 0, 1),
                (2, 0, 0),
                (0, 0, 7),
                (8, 0),
                (0, 0, 0, 0),
                (4, 1),
                (0, 1, 2),
                (0, 4, 0),
                (9, 0),
                (0, 2, 1),
                (2, 3),
                (1, 6),
                (0, 0, 8),
                (0, 5, 0),
                (3, 2),
                (5, 1),
            ]

        self.posterior_threshold = kwargs.pop('posterior_threshold', 0.09)  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        self.posterior_alpha = kwargs.pop('posterior_alpha', 0.3)

        sorted_medusa_choices = sorted(choices, key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1
        self.medusa_len = medusa_len
        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        self.medusa_mask = medusa_attn_mask.unsqueeze(0).unsqueeze(0)

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        topk = 10
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + topk * i + 1
            start += depth_counts[i]
        self.tree_indices = medusa_tree_indices

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1 : start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]
        self.medusa_position_ids = medusa_position_ids

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i - 1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]))
                retrieve_paths.append(cur_medusa_choice[: c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max(len(x) for x in retrieve_indices_nest)
        retrieve_indices = [self.pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        self.retrieve_indices = torch.cat(
            [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1
        )

        if kwargs.pop('verbose', True):
            self.print_config()
            self.print_unused_kwargs()

    def pad_path(self, path, length, pad_value=-2):
        """Pads a path to the specified length with the given pad value.

        Args:
            path (list): The path to be padded.
            length (int): The desired length of the path.
            pad_value (int, optional): The value to use for padding. Default is -2.

        Returns:
            list: The padded path.
        """
        # Calculate the number of padding values needed by subtracting the length
        # of the path from the desired length.
        # Append the padding values to the original path and return the new list.
        return path + [pad_value] * (length - len(path))

    def print_config(self):
        """Prints the configuration of the Medusa model.

        This method prints the model type, hidden size, number of Medusa layers,
        number of Medusa heads, Medusa choices, and vocabulary size.
        """
        logger.info(f'{self.model_type} config:')
        logger.info(f'Hidden size:          {self.hidden_size}')
        logger.info(f'Num medusa layers:    {self.medusa_num_layers}')
        logger.info(f'Num medusa heads:     {self.medusa_num_heads}')
        logger.info(f'Medusa choices:       {self.medusa_choices}')
        logger.info(f'Vocab size:           {self.vocab_size}')
