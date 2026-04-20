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
"""Define LoRA-related classes."""

import copy
import json
import os

import numpy as np
import torch

from ..utils import logger, utils
from ..utils.sanity_checks import check_exist, check_ext


class LoRAHandler:
    """LoRAHandler class for handling one or more LoRAs.

    Attributes:
        e (BaseConfig): The encoder LoRA object, if any.
        l (BaseConfig): The LLM LoRA object.
        llm_lora_indices (int): The number of LoRA configurations.
        lora_config_paths (list): The list of LoRA configuration file paths.

    Methods:
        __init__(lora_config_filepaths):
            Initialize the LoRAHandler.
    """

    def __init__(self, lora_config_filepaths, pipeline_config, quiet=False, dummy_lora=False):
        """Initialize the LoRAHandler.

        Args:
            lora_config_filepaths (str or list of str): The path or list of LoRA configuration file paths.
            pipeline_config (PipelineConfig): The PipelineConfig object.
            quiet (bool, optional): Boolean to turn off info logging. Defaults to False.
            dummy_lora (bool, optional): Boolean to use dummy lora. Defaults to False.
        """
        logger.debug(f'[LoRAHandler] Initialize LoRAHandler. lora_config_filepaths={lora_config_filepaths}')
        self.pipeline_config = pipeline_config
        self.quiet = quiet
        self._dummy_lora = dummy_lora

        self._e = []
        self._l = []
        self.encoder_lora_inputs = []
        self.llm_lora_inputs = []
        self.encoder_lora_indices = []
        self.llm_lora_indices = []
        self.state_dicts = []
        self.global_encoder_min_rank = None
        self.global_encoder_max_rank = None
        self.global_encoder_target_modules = None
        self.global_encoder_start_idx = None
        self.global_encoder_end_idx = None

        self.global_llm_min_rank = None
        self.global_llm_max_rank = None
        self.global_llm_target_modules = None
        self.global_llm_start_idx = None
        self.global_llm_end_idx = None

        self.pad_required = 0
        self.rotated = None
        self.num_lora_configs = 0
        self.lora_config_paths = []
        self.lora_weight_dirs = []
        self.encoder_lora_state_dict_mapping = {}
        self.llm_lora_state_dict_mapping = {}
        self.llm_lora_merged_state_dict_mapping = {}
        self.num_encoder_layers = 0
        self.num_llm_layers = 0

        if lora_config_filepaths is None:
            logger.debug('[LoRAHandler] No loras detected')
            return

        if not isinstance(lora_config_filepaths, (list, tuple)):
            lora_config_filepaths = [lora_config_filepaths]

        self.lora_config_paths = lora_config_filepaths
        self.lora_weight_dirs = [utils.get_dirpath(x) for x in self.lora_config_paths]
        self.num_lora_configs = len(self.lora_config_paths)

        for i, path in enumerate(self.lora_config_paths):
            check_exist(path)
            check_ext(path, '.json')
            with open(path) as f:
                config_ = json.load(f)
            lora_config = utils.get_compatible_lora_config(config_)
            self.rotated = lora_config.get('rotate', False)

            has_encoder = 'encoder' in lora_config
            self._e.append(None if not has_encoder else LoRA(lora_config['encoder'], pipeline_config.e))
            if has_encoder:
                self.encoder_lora_indices.append(i)

            has_llm = 'llm' in lora_config
            self._l.append(None if not has_llm else LoRA(lora_config['llm'], pipeline_config.l))
            if has_llm:
                self.llm_lora_indices.append(i)

        logger.debug(f'[LoRAHandler] {len(self.encoder_lora_indices)} encoder lora(s) detected')
        logger.debug(f'[LoRAHandler] {len(self.llm_lora_indices)} LLM lora(s) detected')

        # Set encoder globals
        for enc_lora in self._e:
            if enc_lora is None:
                continue
            if self.global_encoder_min_rank is None:
                self.global_encoder_min_rank = enc_lora.rank
            else:
                if enc_lora.rank < self.global_encoder_min_rank:
                    self.global_encoder_min_rank = enc_lora.rank
            if self.global_encoder_max_rank is None:
                self.global_encoder_max_rank = enc_lora.rank
            else:
                if enc_lora.rank > self.global_encoder_max_rank:
                    self.global_encoder_max_rank = enc_lora.rank

            # Enforce all target modules must be same
            if self.global_encoder_target_modules is None:
                self.global_encoder_target_modules = enc_lora.target_modules
            else:
                if enc_lora.target_modules != self.global_encoder_target_modules:
                    logger.error(
                        'Multiple encoder LoRAs must strictly have the same `target_modules`.'
                        f'Got: {self.global_encoder_target_modules} and {enc_lora.target_modules}.'
                    )

            # Enforce all start idx must be same
            if self.global_encoder_start_idx is None:
                self.global_encoder_start_idx = enc_lora.start_idx
            else:
                if enc_lora.start_idx != self.global_encoder_start_idx:
                    logger.error(
                        'Multiple encoder LoRAs must have the same `lora_start_layer_idx`'
                        f'Got: {self.global_encoder_start_idx} and {enc_lora.start_idx}.',
                        err=ValueError,
                    )

            # Enforce all end idx must be same
            if self.global_encoder_end_idx is None:
                self.global_encoder_end_idx = enc_lora.end_idx
            else:
                if enc_lora.end_idx != self.global_encoder_end_idx:
                    logger.error(
                        'Multiple encoder LoRAs must have the same `lora_end_layer_idx`'
                        f'Got: {self.global_encoder_end_idx} and {enc_lora.end_idx}.',
                        err=ValueError,
                    )
            if not self.quiet:
                if self.global_encoder_min_rank == self.global_encoder_max_rank:
                    logger.info(f'[LoRAHandler] Encoder lora rank:          {self.global_encoder_min_rank}')
                else:
                    logger.info(f'[LoRAHandler] Encoder min lora rank:      {self.global_encoder_min_rank}')
                    logger.info(f'[LoRAHandler] Encoder max lora rank:      {self.global_encoder_max_rank}')
                logger.info(f'[LoRAHandler] Encoder lora alpha:         {enc_lora.alpha}')
                if enc_lora.dropout > 0:
                    logger.info(f'[LoRAHandler] Encoder lora dropout:       {enc_lora.dropout}')
                logger.info(f'[LoRAHandler] Encoder target modules:     {self.global_encoder_target_modules}')
                logger.info(f'[LoRAHandler] Encoder lora start layer:   {self.global_encoder_start_idx}')
                logger.info(f'[LoRAHandler] Encoder lora end layer:     {self.global_encoder_end_idx}')
                logger.info(f'[LoRAHandler] Encoder lora rotated:       {self.rotated}')

        # Set encoder defaults if no encoders
        if self.global_encoder_target_modules is None:
            self.global_encoder_target_modules = []

        # Set LLM globals
        for llm_lora in self._l:
            if llm_lora is None:
                continue
            if self.global_llm_min_rank is None:
                self.global_llm_min_rank = llm_lora.rank
            else:
                if llm_lora.rank < self.global_llm_min_rank:
                    self.global_llm_min_rank = llm_lora.rank
            if self.global_llm_max_rank is None:
                self.global_llm_max_rank = llm_lora.rank
            else:
                if llm_lora.rank > self.global_llm_max_rank:
                    self.global_llm_max_rank = llm_lora.rank

            # Enforce all target modules must be same
            if self.global_llm_target_modules is None:
                self.global_llm_target_modules = llm_lora.target_modules
            else:
                if llm_lora.target_modules != self.global_llm_target_modules:
                    logger.error(
                        'Multiple llm LoRAs must strictly have the same `target_modules`.'
                        f'Got: {self.global_llm_target_modules} and {llm_lora.target_modules}.'
                    )

            # Enforce all start idx must be same
            if self.global_llm_start_idx is None:
                self.global_llm_start_idx = llm_lora.start_idx
            else:
                if llm_lora.start_idx != self.global_llm_start_idx:
                    logger.error(
                        'Multiple llm LoRAs must have the same `lora_start_layer_idx`'
                        f'Got: {self.global_llm_start_idx} and {llm_lora.start_idx}.',
                        err=ValueError,
                    )

            # Enforce all end idx must be same
            if self.global_llm_end_idx is None:
                self.global_llm_end_idx = llm_lora.end_idx
            else:
                if llm_lora.end_idx != self.global_llm_end_idx:
                    logger.error(
                        'Multiple llm LoRAs must have the same `lora_end_layer_idx`'
                        f'Got: {self.global_llm_end_idx} and {llm_lora.end_idx}.',
                        err=ValueError,
                    )

            if not self.quiet:
                if self.global_llm_min_rank == self.global_llm_max_rank:
                    logger.info(f'[LoRAHandler] LLM lora rank:          {self.global_llm_min_rank}')
                else:
                    logger.info(f'[LoRAHandler] LLM min lora rank:      {self.global_llm_min_rank}')
                    logger.info(f'[LoRAHandler] LLM max lora rank:      {self.global_llm_max_rank}')
                logger.info(f'[LoRAHandler] LLM lora alpha:         {llm_lora.alpha}')
                if llm_lora.dropout > 0:
                    logger.info(f'[LoRAHandler] LLM lora dropout:       {llm_lora.dropout}')
                logger.info(f'[LoRAHandler] LLM target modules:     {self.global_llm_target_modules}')
                logger.info(f'[LoRAHandler] LLM lora start layer:   {self.global_llm_start_idx}')
                logger.info(f'[LoRAHandler] LLM lora end layer:     {self.global_llm_end_idx}')
                logger.info(f'[LoRAHandler] LLM lora rotated:       {self.rotated}')

        # Set encoder defaults if no encoders
        if self.global_llm_target_modules is None:
            self.global_llm_target_modules = []

    @property
    def e(self, idx=None):
        """Encoder LoRA config getter."""
        if idx is not None:
            if idx >= self.encoder_lora_indices:
                logger.error('idx cannot be more than or equal to the total number of encoder LoRA configs.')
            return self._e[idx]
        return self._e

    @property
    def l(self, idx=None):  # noqa: E743
        """LLM LoRA config getter."""
        if idx is not None:
            if idx >= self.llm_lora_indices:
                logger.error('idx cannot be more than or equal to the total number of LLM LoRA configs.')
            return self._l[idx]
        return self._l

    def set_encoder_lora_state_dict_mapping(self, state_dict_mapping):
        """Sets the LLM lora state dict mapping.

        Args:
            state_dict_mapping (dict): The encoder LoRA state dict mapping for all FCs.
        """
        self.encoder_lora_state_dict_mapping = state_dict_mapping
        num_encoder_layers = 0
        for internal_key in self.encoder_lora_state_dict_mapping:
            layer_id = int(internal_key.split('_')[0]) + 1
            if layer_id > num_encoder_layers:
                num_encoder_layers = layer_id
        self.num_encoder_layers = num_encoder_layers
        logger.debug(
            f'[LoRAHandler] self.encoder_lora_state_dict_mapping={self.encoder_lora_state_dict_mapping}, num_encoder_layers={num_encoder_layers}'  # noqa: E501
        )

    def set_llm_lora_state_dict_mapping(self, state_dict_mapping, combined_state_dict_mapping=None):
        """Sets the LLM lora state dict mapping.

        Args:
            state_dict_mapping (dict): The LLM LoRA state dict mapping for all FCs.
            combined_state_dict_mapping (dict): The LLM LoRA state dict mapping for combined FCs.
        """
        self.llm_lora_state_dict_mapping = state_dict_mapping
        self.llm_lora_merged_state_dict_mapping = combined_state_dict_mapping
        num_llm_layers = 0
        for internal_key in self.llm_lora_state_dict_mapping:
            layer_id = int(internal_key.split('_')[0]) + 1
            if layer_id > num_llm_layers:
                num_llm_layers = layer_id
        self.num_llm_layers = num_llm_layers
        logger.debug(
            f'[LoRAHandler] self.llm_lora_state_dict_mapping={self.llm_lora_state_dict_mapping}, num_llm_layers={num_llm_layers}'  # noqa: E501
        )
        logger.debug(f'[LoRAHandler] self.llm_lora_merged_state_dict_mapping={self.llm_lora_merged_state_dict_mapping}')

    def has_encoder_lora(self, idx=None):
        """Checks if LoRAHandler has encoder LoRA objects.

        Args:
            idx (int, optional): The specific LoRA index to check. Defaults to None.
                (checks if any of the LoRAs contain encoder LoRA).
        """
        if idx is not None:
            if idx >= len(self.lora_config_paths):
                logger.error(
                    f'[LoRAHandler] LoRAHandler only has {len(self.lora_config_paths)} LoRAs but idx given is {idx}',
                    err=IndexError,
                )
            return idx in self.encoder_lora_indices
        return not all(x is None for x in self._e)

    def has_llm_lora(self, idx=None):
        """Checks if LoRAHandler has LLM LoRA objects.

        Args:
            idx (int, optional): The specific LoRA index to check. Defaults to None.
                (checks if any of the LoRAs contain LLM LoRA).
        """
        if idx is not None:
            if idx >= len(self.lora_config_paths):
                logger.error(
                    f'[LoRAHandler] LoRAHandler only has {len(self.lora_config_paths)} LoRAs but idx given is {idx}',
                    err=IndexError,
                )
            return idx in self.llm_lora_indices
        return not all(x is None for x in self._l)

    def pad_lora_A(self, lora_inp):  # noqa: N802
        """Pads the lora type A from shape [original_r, ...] to [target_r, ...]."""
        if isinstance(lora_inp, torch.Tensor):
            return torch.cat([lora_inp, torch.zeros([self.pad_required, lora_inp.shape[1]])], dim=0)
        return np.concatenate([lora_inp, np.zeros([self.pad_required, lora_inp.shape[1]])], axis=0)

    def pad_lora_B(self, lora_inp):  # noqa: N802
        """Pads the lora type B from shape [..., original_r] to [..., target_r]."""
        if isinstance(lora_inp, torch.Tensor):
            return torch.cat([lora_inp, torch.zeros([lora_inp.shape[0], self.pad_required])], dim=-1)
        return np.concatenate([lora_inp, np.zeros([lora_inp.shape[0], self.pad_required])], axis=-1)

    def load_as_state_dict(self, pipeline, lora_config_paths=None):
        """Load lora checkpoints into a state dict."""
        from ..utils import utils

        logger.debug('[LoRAHandler] Enter load_as_state_dict')

        if lora_config_paths is None:
            lora_config_paths = self.lora_config_paths

        self.state_dicts = [{'encoder': {}, 'llm': {}} for _ in range(len(lora_config_paths))]

        prefixes = ['']

        lora_files = []
        for lora_id, lora_config_filepath in enumerate(lora_config_paths):
            lora_dir = utils.get_dirpath(lora_config_filepath)
            if not self._dummy_lora:
                lora_files = sorted(
                    [
                        os.path.join(lora_dir, f)
                        for f in os.listdir(lora_dir)
                        if f.endswith(('.bin', '.safetensors')) and not f.startswith(('training_args', 'embedding'))
                    ]
                )
            if len(lora_files) == 0:
                logger.warning(
                    f'No LoRA weights found in `{lora_dir}`. This message is expected for QALFT from scratch.'
                )
                self.encoder_lora_inputs = [[] for _ in range(len(pipeline.encoder_layers_per_chunk))]
                return

            logger.info(f'Loading LoRA weights from: `{lora_files}`')
            is_safetensors = lora_files[0].endswith('.safetensors')
            curr_state_dict = {}
            for i in range(len(lora_files)):
                if is_safetensors:
                    curr_state_dict.update(utils.load_file(lora_files[i]))
                else:
                    curr_state_dict.update(torch.load(lora_files[i], map_location='cpu'))
            state_dict_keys = list(curr_state_dict.keys())

            # Check if merged LLM weights/biases exist. Split them if exist.
            for internal_key, external_key in self.llm_lora_merged_state_dict_mapping.items():
                found_key = None
                for pre in prefixes:
                    key_to_test = pre + external_key
                    if key_to_test in state_dict_keys:
                        logger.debug(
                            f'Found {internal_key} merged LoRA weight using prefix, state_dict key={key_to_test}'
                        )
                        found_key = key_to_test
                        break

                if found_key is None:
                    for k in state_dict_keys:
                        if k.endswith(external_key):
                            logger.debug(
                                f'Found {internal_key} merged LoRA weight using iteration, state_dict key='
                                f'{k}. Adding prefix: {k[: -len(external_key)]}'
                            )
                            prefixes.append(k[: -len(external_key)])
                            found_key = k
                            break

                if found_key is not None:
                    layer_id = int(internal_key.split('_')[0])
                    if '_qkv_A' in internal_key:
                        # Duplicate QKV lora A weight
                        logger.debug(
                            f'Duplicating layer {layer_id} {internal_key} merged QKV LoRA A weight into Q/K/V weights'
                        )
                        qkv_a = curr_state_dict.pop(found_key)
                        curr_state_dict.update(
                            {
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_q_A'].values())): qkv_a,
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_k_A'].values())): qkv_a,
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_v_A'].values())): qkv_a,
                            }
                        )
                    elif '_qkv_B' in internal_key:
                        # Split QKV lora B weight
                        logger.debug(
                            f'Splitting layer {layer_id} {internal_key} merged QKV LoRA B weight into Q/K/V weights'
                        )
                        qkv_b = curr_state_dict.pop(found_key)
                        q_size = self.pipeline_config.l.hidden_size
                        head_dim = self.pipeline_config.l.hidden_size // self.pipeline_config.l.num_attention_heads
                        kv_size = self.pipeline_config.l.num_key_value_heads * head_dim
                        curr_state_dict.update(
                            {
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_q_B'].values())): qkv_b[
                                    :q_size, :
                                ],
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_k_B'].values())): qkv_b[
                                    q_size : q_size + kv_size, :
                                ],
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_v_B'].values())): qkv_b[
                                    q_size + kv_size : q_size + 2 * kv_size, :
                                ],
                            }
                        )
                    elif '_gu_A' in internal_key:
                        # Duplicate GateUp lora A weight
                        logger.debug(
                            f'Duplicating layer {layer_id} {internal_key} merged GateUp LoRA A weight into Gate/Up '
                            'weights'
                        )
                        gu_a = curr_state_dict.pop(found_key)
                        curr_state_dict.update(
                            {
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_g_A'].values())): gu_a,
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_u_A'].values())): gu_a,
                            }
                        )
                    else:
                        assert '_gu_B' in internal_key
                        # Split GateUp lora B weight
                        logger.debug(
                            f'Splitting layer {layer_id} {internal_key} merged GateUp LoRA B weight into Gate/Up '
                            'weights'
                        )
                        gu_b = curr_state_dict.pop(found_key)
                        intermediate_size = self.pipeline_config.l.intermediate_size
                        curr_state_dict.update(
                            {
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_g_B'].values())): gu_b[
                                    :intermediate_size, :
                                ],
                                next(iter(self.llm_lora_state_dict_mapping[f'{layer_id}_u_B'].values())): gu_b[
                                    intermediate_size:, :
                                ],
                            }
                        )

            missing_keys = []
            state_dict_keys = list(curr_state_dict.keys())
            # Only prescale if not qalft and alpha / rank != 1.0
            prescale_enc_lora = (
                self.e[lora_id] is not None and pipeline.task != 'qalft' and self.e[lora_id].scale != 1.0
            )
            enc_lora_alpha = self.e[lora_id].alpha if prescale_enc_lora else 1
            enc_lora_rank = self.e[lora_id].rank if prescale_enc_lora else 1
            # Translate all the state dict keys into internal keys
            for layer_id in range(self.num_encoder_layers):
                for module in self.global_encoder_target_modules:
                    for proj in ['A', 'B']:
                        internal_key = f'{layer_id}_{module.replace("_proj", "")}_{proj}'
                        mapping_dict = self.encoder_lora_state_dict_mapping[internal_key]
                        logger.debug(f'internal_key={internal_key}, mapping_dict={mapping_dict}')
                        found = False
                        # Ensure that weight exist in state
                        # mapping_dict values should be a dict with exactly length 1
                        if not isinstance(mapping_dict, dict):
                            logger.error(f'Expected dict for mapping_dict but got {type(mapping_dict)}', err=TypeError)
                        if len(mapping_dict) != 1:
                            logger.error(
                                f'Expected exactly 1 key-value pair in mapping_dict but got {len(mapping_dict)}'
                            )
                        model_key = next(iter(mapping_dict))
                        external_key = mapping_dict[model_key]

                        # Check if key with all found prefixes directly matches the state_dict key
                        for pre in prefixes:
                            key_to_test = pre + external_key
                            if key_to_test in state_dict_keys:
                                logger.debug(
                                    f'Found encoder {internal_key} weight using prefix, state_dict key={key_to_test}'
                                )
                                if proj == 'A':
                                    self.state_dicts[lora_id]['encoder'][internal_key] = (
                                        curr_state_dict.pop(key_to_test) * enc_lora_alpha
                                    )
                                elif proj == 'B':
                                    self.state_dicts[lora_id]['encoder'][internal_key] = (
                                        curr_state_dict.pop(key_to_test) / enc_lora_rank
                                    )
                                found = True
                                state_dict_keys.remove(key_to_test)
                                break

                        if not found:
                            for k in state_dict_keys:
                                if k.endswith(external_key):
                                    logger.debug(
                                        f'Found encoder {internal_key} weight using iteration, state_dict key={k}, '
                                        f'Adding prefix: {k[: -len(external_key)]}'
                                    )
                                    prefixes.append(k[: -len(external_key)])
                                    if proj == 'A':
                                        self.state_dicts[lora_id]['encoder'][internal_key] = (
                                            curr_state_dict.pop(k) * enc_lora_alpha
                                        )
                                    elif proj == 'B':
                                        self.state_dicts[lora_id]['encoder'][internal_key] = (
                                            curr_state_dict.pop(k) / enc_lora_rank
                                        )
                                    found = True
                                    state_dict_keys.remove(k)
                                    break

                        if not found:
                            logger.warning(f'Cannot find encoder {internal_key} lora {proj} weight')
                            missing_keys.append((internal_key, external_key))

            prescale_llm_lora = (
                self.l[lora_id] is not None and pipeline.task != 'qalft' and self.l[lora_id].scale != 1.0
            )
            llm_lora_alpha = self.l[lora_id].alpha if prescale_llm_lora else 1
            llm_lora_rank = self.l[lora_id].rank if prescale_llm_lora else 1
            for layer_id in range(self.num_llm_layers):
                if layer_id < self.global_llm_start_idx or layer_id > self.global_llm_end_idx:
                    continue
                for module in self.global_llm_target_modules:
                    for proj in ['A', 'B']:
                        internal_key = f'{layer_id}_{module.replace("_proj", "")}_{proj}'
                        mapping_dict = self.llm_lora_state_dict_mapping[internal_key]
                        logger.debug(f'internal_key={internal_key}, mapping_dict={mapping_dict}')
                        found = False
                        # Ensure that weight exist in state
                        # mapping_dict values should be a dict with exactly length 1
                        if not isinstance(mapping_dict, dict):
                            logger.error(f'Expected dict for mapping_dict but got {type(mapping_dict)}', err=TypeError)
                        if len(mapping_dict) != 1:
                            logger.error(
                                f'Expected exactly 1 key-value pair in mapping_dict but got {len(mapping_dict)}'
                            )
                        model_key = next(iter(mapping_dict))
                        external_key = mapping_dict[model_key]

                        # Check if key with all found prefixes directly matches the state_dict key
                        for pre in prefixes:
                            key_to_test = pre + external_key
                            if key_to_test in state_dict_keys:
                                logger.debug(
                                    f'Found llm {internal_key} weight using prefix, state_dict key={key_to_test}'
                                )
                                if proj == 'A':
                                    self.state_dicts[lora_id]['llm'][internal_key] = (
                                        curr_state_dict.pop(key_to_test) * llm_lora_alpha
                                    )
                                elif proj == 'B':
                                    self.state_dicts[lora_id]['llm'][internal_key] = (
                                        curr_state_dict.pop(key_to_test) / llm_lora_rank
                                    )
                                found = True
                                state_dict_keys.remove(key_to_test)
                                break

                        if not found:
                            for k in state_dict_keys:
                                if k.endswith(external_key):
                                    logger.debug(
                                        f'Found llm {internal_key} weight using iteration, state_dict key={k}, '
                                        f'Adding prefix: {k[: -len(external_key)]}'
                                    )
                                    prefixes.append(k[: -len(external_key)])
                                    if proj == 'A':
                                        self.state_dicts[lora_id]['llm'][internal_key] = (
                                            curr_state_dict.pop(k) * llm_lora_alpha
                                        )
                                    elif proj == 'B':
                                        self.state_dicts[lora_id]['llm'][internal_key] = (
                                            curr_state_dict.pop(k) / llm_lora_rank
                                        )
                                    found = True
                                    state_dict_keys.remove(k)
                                    break

                        if not found:
                            logger.warning(f'Cannot find llm {internal_key} lora {proj} weight')
                            missing_keys.append((internal_key, external_key))

            if len(missing_keys) > 0:
                logger.error(
                    f'There are missing LoRA keys that are expected but not found in the LoRA weights: {missing_keys}'
                )

            # FIXME: Need to handle encoder lora when there IS encoder lora at QALFT
            self.encoder_lora_inputs = [[] for _ in range(len(pipeline.encoder_layers_per_chunk))]

    def load_chunked_lora_inputs(self, pipeline, rotate=None):
        """Load lora checkpoints and split them into how the model is chunked.

        Args:
            pipeline (Pipeline): The Pipeline object.
            rotate (bool, optional): Enforce either rotated or unrotated LoRA loading. Defaults to None
                (directly load the provided LoRA config without enforcing LoRA rotation).

        Raises:
            FileNotFoundError: If lora_config provided was unrotated, and rotated LoRA does not exist, and rotate=True.
        """
        from pathlib import Path

        logger.debug(f'[LoRAHandler] Enter load_chunked_lora_inputs, rotate={rotate}')

        if self.rotated and rotate is False:  # rotate can be None, which means default to self.rotated
            logger.debug('Explicitly loading unrotated LLM LoRA even though rotated LoRA was specified in LoRA config.')
            unrotated_paths = []
            for rotated_path in self.lora_config_paths:
                unrotated_path = Path(rotated_path).parent.parent / 'adapter_config.json'
                if not os.path.exists(unrotated_path):
                    logger.error(
                        f'Unrotated LLM LoRA for {rotated_path} cannot be found. Expected file path: {unrotated_path}',
                        err=FileNotFoundError,
                    )
                unrotated_paths.append(unrotated_path)
            lora_config_paths = unrotated_paths
        elif not self.rotated and rotate:
            logger.error(
                'Unable to load rotated LoRA if unrotated LoRA is specified in LoRA config as rotated LoRA path '
                'cannot be implicitly inferred.'
            )
        else:
            logger.debug(f'Loading {"rotated" if self.rotated else "unrotated"} LLM LoRA.')
            lora_config_paths = self.lora_config_paths

        self.load_as_state_dict(pipeline, lora_config_paths)

        if self._dummy_lora:
            default_lora_shapes = self.get_default_lora_shapes()

        temp_state_dict = copy.deepcopy(self.state_dicts)

        if not self.has_encoder_lora():
            # No encoder lora in model
            if pipeline.task == 'ptq':
                self.encoder_lora_inputs = [[[]] for _ in range(len(pipeline.encoder_layers_per_chunk))]
            else:
                self.encoder_lora_inputs = [[] for _ in range(len(pipeline.encoder_layers_per_chunk))]
            logger.debug(f'self.encoder_lora_inputs: {self.encoder_lora_inputs}')
        else:
            # Encoder lora in model
            if pipeline.task != 'ptq':
                if len(lora_config_paths) > 1:
                    logger.error('Only 1 lora config allowed when `for_ptq` is False')
                assert len(lora_config_paths) == 1
                if lora_config_paths[0] is None:
                    logger.error('lora config cannot be None when `for_ptq` is True')

            encoder_lora_inputs = []
            for lora_id, lora_config_filepath in enumerate(lora_config_paths):
                if lora_config_filepath is None or not self.has_encoder_lora(lora_id):
                    curr_lora_inputs = [None] * len(pipeline.encoder_layers_per_chunk)
                else:
                    curr_lora_inputs = []
                    for chunk_id, num_layer in enumerate(pipeline.encoder_layers_per_chunk):
                        chunk_inputs = []
                        for layer_id in range(
                            sum(pipeline.encoder_layers_per_chunk[:chunk_id]),
                            num_layer + sum(pipeline.encoder_layers_per_chunk[:chunk_id]),
                        ):
                            if layer_id < self.global_encoder_start_idx or layer_id > self.global_encoder_end_idx:
                                continue
                            for module in self.global_encoder_target_modules:
                                for proj in ['A', 'B']:
                                    internal_key = f'{module.replace("_proj", "")}_{proj}'
                                    if not self._dummy_lora:
                                        internal_key = f'{layer_id}_{internal_key}'
                                        lora_input = temp_state_dict[lora_id]['encoder'].pop(internal_key)
                                    else:
                                        lora_input = torch.randn(*default_lora_shapes[internal_key])
                                    if isinstance(pipeline.dtype, torch.dtype):
                                        chunk_inputs.append(lora_input.unsqueeze(0).to(pipeline.dtype))
                                    else:
                                        chunk_inputs.append(
                                            lora_input.unsqueeze(0)
                                            .to(torch.float32)
                                            .cpu()
                                            .numpy()
                                            .astype(pipeline.dtype)
                                        )

                        curr_lora_inputs.append(chunk_inputs)
                encoder_lora_inputs.append(curr_lora_inputs)

            if pipeline.task != 'ptq':
                assert len(encoder_lora_inputs) == 1
                # [chunk_id][lora_inputs]
                self.encoder_lora_inputs = encoder_lora_inputs[0]
            else:
                # Transpose from [lora_id][chunk_id][lora_inputs]
                #             to [chunk_id][lora_id][lora_inputs]
                self.encoder_lora_inputs = [list(x) for x in zip(*encoder_lora_inputs)]

        if not self.has_llm_lora():
            # No LLM lora in model
            if pipeline.task == 'ptq':
                self.llm_lora_inputs = [[[]] for _ in range(sum(pipeline.llm_layers_per_chunk))]
            else:
                self.llm_lora_inputs = [[] for _ in range(sum(pipeline.llm_layers_per_chunk))]
        else:
            # LLM lora in model
            if pipeline.task not in ['ptq', 'export_lora']:
                if len(lora_config_paths) > 1:
                    logger.error('Only 1 lora config allowed when `for_ptq` is False')
                assert len(lora_config_paths) == 1
                if lora_config_paths[0] is None:
                    logger.error('lora config cannot be None when `for_ptq` is True')

            from .pipeline import QuantizedPipeline

            if pipeline.task in ['inference', 'evaluate'] and isinstance(pipeline, QuantizedPipeline):
                if len(pipeline.llm_gen_quantized_model_infos):
                    self.pad_required = (
                        pipeline.llm_gen_quantized_model_infos[0]['lora_rank'] - self.global_llm_max_rank
                    )
                elif len(pipeline.llm_prompt_quantized_model_infos):
                    self.pad_required = (
                        pipeline.llm_prompt_quantized_model_infos[0]['lora_rank'] - self.global_llm_max_rank
                    )
                else:
                    logger.error('No quantized model info is found. Please check your quantized model folder.')
                if self.pad_required < 0:
                    logger.error(
                        f'lora_rank of quantized model ({pipeline.llm_gen_quantized_model_infos[0]["lora_rank"]}) '
                        f'cannot be smaller than lora weight ({self.global_llm_max_rank})'
                    )

            llm_lora_inputs = []
            for lora_id, lora_config_filepath in enumerate(lora_config_paths):
                if lora_config_filepath is None or not self.has_llm_lora(lora_id):
                    curr_lora_inputs = [None] * len(pipeline.llm_layers_per_chunk)
                else:
                    curr_lora_inputs = []
                    for chunk_id, num_layer in enumerate(pipeline.llm_layers_per_chunk):
                        chunk_inputs = []
                        for layer_id in range(
                            sum(pipeline.llm_layers_per_chunk[:chunk_id]),
                            num_layer + sum(pipeline.llm_layers_per_chunk[:chunk_id]),
                        ):
                            if layer_id < self.global_llm_start_idx or layer_id > self.global_llm_end_idx:
                                continue
                            for module in self.global_llm_target_modules:
                                for proj in ['A', 'B']:
                                    pad_func = self.pad_lora_A if proj == 'A' else self.pad_lora_B
                                    internal_key = f'{module.replace("_proj", "")}_{proj}'
                                    if not self._dummy_lora:
                                        internal_key = f'{layer_id}_{internal_key}'
                                        lora_input = temp_state_dict[lora_id]['llm'].pop(internal_key)
                                    else:
                                        lora_input = torch.randn(*default_lora_shapes[internal_key])
                                    lora_input = pad_func(lora_input) if self.pad_required > 0 else lora_input

                                    if isinstance(pipeline.dtype, torch.dtype):
                                        chunk_inputs.append(lora_input.unsqueeze(0).to(pipeline.dtype))
                                    else:
                                        chunk_inputs.append(
                                            lora_input.unsqueeze(0)
                                            .to(torch.float32)
                                            .cpu()
                                            .numpy()
                                            .astype(pipeline.dtype)
                                        )

                        curr_lora_inputs.append(chunk_inputs)
                llm_lora_inputs.append(curr_lora_inputs)

            for lora_id in range(len(lora_config_paths)):
                if len(temp_state_dict[lora_id]['encoder']) > 0:
                    logger.error(
                        'There are extra encoder lora state_dict keys that are unassigned for '
                        f'{lora_config_paths[lora_id]["encoder"]}:\n{temp_state_dict[lora_id]["encoder"].keys()}'
                    )
                if len(temp_state_dict[lora_id]['llm']) > 0:
                    logger.error(
                        'There are extra LLM lora state_dict keys that are unassigned for '
                        f'{lora_config_paths[lora_id]["llm"]}:\n{temp_state_dict[lora_id]["llm"].keys()}'
                    )

            if pipeline.task not in ['ptq', 'export_lora']:
                assert len(llm_lora_inputs) == 1
                # [chunk_id][lora_inputs]
                self.llm_lora_inputs = llm_lora_inputs[0]
            else:
                # Transpose from [lora_id][chunk_id][lora_inputs]
                #             to [chunk_id][lora_id][lora_inputs]
                self.llm_lora_inputs = [list(x) for x in zip(*llm_lora_inputs)]

    def get_default_lora_shapes(self):
        """Get default lora shapes for dummy lora."""
        lora_rank = self._l[0].rank
        hidden_size = self.pipeline_config.l.hidden_size
        intermediate_size = self.pipeline_config.l.intermediate_size
        num_heads = self.pipeline_config.l.num_key_value_heads  # assume non sparse
        head_dim = self.pipeline_config.l.head_dim
        num_attention_heads = self.pipeline_config.l.num_attention_heads

        return {
            'q_A': (lora_rank, hidden_size),
            'q_B': (head_dim * num_attention_heads, lora_rank),
            'k_A': (lora_rank, hidden_size),
            'k_B': (num_heads * head_dim, lora_rank),
            'v_A': (lora_rank, hidden_size),
            'v_B': (num_heads * head_dim, lora_rank),
            'o_A': (lora_rank, head_dim * num_attention_heads),
            'o_B': (hidden_size, lora_rank),
            'gate_A': (lora_rank, hidden_size),
            'gate_B': (intermediate_size, lora_rank),
            'up_A': (lora_rank, hidden_size),
            'up_B': (intermediate_size, lora_rank),
            'down_A': (lora_rank, intermediate_size),
            'down_B': (hidden_size, lora_rank),
        }

    def get_from_state_dict(self, names, encoder=False):
        """Retrieve Lora tensors from state_dict."""
        if not isinstance(names, list):
            names = [names]

        return [lora_dict['encoder' if encoder else 'llm'][name] for lora_dict in self.state_dicts for name in names]

    def get_widest_lora_ranges(self):
        """Get the widest prescaled llm lora range.

        Returns:
            final_lora_qparam (dict): Finalized lora min max ranges.
        """
        final_lora_qparam = {k: {'min': v.min(), 'max': v.max()} for k, v in self.state_dicts[0]['llm'].items()}
        for lora_id in range(1, len(self.state_dicts)):
            llm_lora_sd = self.state_dicts[lora_id]['llm']
            logger.debug(f'Lora state dict id: {lora_id}')
            for lora_name, lora_weight in llm_lora_sd.items():
                logger.debug(f'Lora name: {lora_name}')
                lora_min = lora_weight.min()
                lora_max = lora_weight.max()
                if lora_min < final_lora_qparam[lora_name]['min']:
                    logger.debug(f'Replacing {lora_name} min: {final_lora_qparam[lora_name]["min"]} -> {lora_min}')
                    final_lora_qparam[lora_name]['min'] = lora_min
                if lora_max > final_lora_qparam[lora_name]['max']:
                    logger.debug(f'Replacing {lora_name} max: {final_lora_qparam[lora_name]["max"]} -> {lora_max}')
                    final_lora_qparam[lora_name]['max'] = lora_max

        return final_lora_qparam

    def save_rotated(self, layer_idx=None, ori_lora_config_paths=None):
        """Rotate and save rotated LorRAs to a new subdirectory.

        Joint-ptq exports per-layer LoRA and export_lora_bin exports entire LoRA.
        """
        from safetensors.torch import save_file

        logger.debug('[LoRAHandler] Enter save_rotated')
        if not self.rotated:
            logger.error('[LoRAHandler] Expected LoRA to be rotated, but is not.')

        lora_config_paths = ori_lora_config_paths or self.lora_config_paths
        for lora_id, lora_cfg in enumerate(lora_config_paths):
            # TODO: rotate encoder lora
            # 1. Save rotated LoRAs as .safetensors
            llm_lora_dict = self.state_dicts[lora_id]['llm']
            lora_keys = (
                llm_lora_dict.keys()
                if layer_idx is None
                else [k for k in llm_lora_dict if k.startswith(f'{layer_idx}_')]
            )
            llm_lora_dict_with_external_key = {
                next(iter(self.llm_lora_state_dict_mapping[k].values())): llm_lora_dict[k] for k in lora_keys
            }
            lora_dir = os.path.dirname(lora_cfg)
            rot_lora_dir = os.path.join(
                lora_dir, f'rotate_{self.pipeline_config.rotate_mode}_{self.pipeline_config.rotate_seed}'
            )
            os.makedirs(rot_lora_dir, exist_ok=True)
            rot_lora_path = os.path.join(
                rot_lora_dir,
                'adapter_model.safetensors' if layer_idx is None else f'adapter_model_{layer_idx}.safetensors',
            )
            logger.info(f'[LoRAHandler] Saving rotated LoRA to `{rot_lora_path}`')
            save_file(llm_lora_dict_with_external_key, rot_lora_path, metadata={'format': 'pt'})

            # 2. Save rotated LoRA config
            rot_lora_cfg = os.path.join(rot_lora_dir, os.path.basename(lora_cfg))
            if self.lora_config_paths[lora_id] == rot_lora_cfg:
                continue

            with open(lora_cfg) as f:
                lora_config = json.load(f)
            if self.pipeline_config.l.fc_names['attn']['qkv'] in lora_config['target_modules']:
                logger.debug('[LoRAHandler] Replace QKV target_module with Q/K/V target_modules')
                index = lora_config['target_modules'].index(self.pipeline_config.l.fc_names['attn']['qkv'])
                lora_config['target_modules'] = (
                    lora_config['target_modules'][:index]
                    + [
                        self.pipeline_config.l.fc_names['attn']['q'],
                        self.pipeline_config.l.fc_names['attn']['k'],
                        self.pipeline_config.l.fc_names['attn']['v'],
                    ]
                    + lora_config['target_modules'][index + 1 :]
                )
            if self.pipeline_config.l.fc_names['mlp']['gateup'] in lora_config['target_modules']:
                logger.debug('[LoRAHandler] Replace GateUp target_module with Gate/Up target_modules')
                index = lora_config['target_modules'].index(self.pipeline_config.l.fc_names['mlp']['gateup'])
                lora_config['target_modules'] = (
                    lora_config['target_modules'][:index]
                    + [
                        self.pipeline_config.l.fc_names['mlp']['gate'],
                        self.pipeline_config.l.fc_names['mlp']['up'],
                    ]
                    + lora_config['target_modules'][index + 1 :]
                )
            # Add rotation flags in lora config
            lora_config.update(
                {
                    'rotate': True,
                    'rotate_seed': self.pipeline_config.rotate_seed,
                    'rotate_mode': self.pipeline_config.rotate_mode,
                }
            )
            logger.debug(f'[LoRAHandler] Rotated LoRA config: {lora_config}')
            with open(rot_lora_cfg, 'w') as f:
                f.write(json.dumps(lora_config, indent=4))
            logger.info(f'[LoRAHandler] Saving rotated LoRA config to `{rot_lora_cfg}`.')
            logger.debug(f'[LoRAHandler] Replace self.lora_config_paths[{lora_id}] with {rot_lora_cfg}')
            self.lora_config_paths[lora_id] = rot_lora_cfg


class LoRA:
    """LoRA class for handling single LoRAs.

    Methods:
        __init__(lora_config_filepaths):
            Initialize the LoRA object.
    """

    def __init__(self, lora_sub_config, model_config):
        """Initialize the LoRA.

        Args:
            lora_sub_config (dict): The dictionary containing the LoRA config of either an encoder or LLM.
            model_config (BaseConfig): The encoder or LLM config object.
        """
        logger.debug(f'[LoRA] Initialize LoRA object. lora_sub_config={lora_sub_config}')
        if not isinstance(lora_sub_config, dict):
            logger.error(
                f'[LoRA] Expected `lora_sub_config` to be type dict but got type {type(lora_sub_config)}',
                err=TypeError,
            )

        self.rank = lora_sub_config.get('r', lora_sub_config.get('rank', None))
        if self.rank is None:
            logger.error('[LoRA] LoRA rank cannot be None')

        self.target_modules = lora_sub_config.get('target_modules', None)
        if self.target_modules is None:
            logger.error('[LoRA] LoRA target_modules cannot be None')
        self.split_lora_fc_names()

        self.alpha = lora_sub_config.get('lora_alpha', lora_sub_config.get('alpha', self.rank))

        self.scale = self.alpha / self.rank  # only used during qalft (training with parallel lora)

        self.dropout = lora_sub_config.get('lora_dropout', 0.0)

        self.start_idx = lora_sub_config.get('lora_start_layer_idx', 0)

        model_num_hidden_layers = (
            (model_config.early_exit_index + model_config.early_exit_num_layers)
            if model_config.early_exit_index is not None
            else model_config.num_hidden_layers
        )

        self.end_idx = lora_sub_config.get('lora_end_layer_idx', model_num_hidden_layers - 1)

        if self.end_idx < 0:
            self.end_idx += model_config.num_hidden_layers

        if self.start_idx > self.end_idx:
            logger.error(
                f'`lora_start_layer_idx` ({self.start_idx}) cannot be more than `lora_end_layer_idx` ({self.end_idx})',
                err=ValueError,
            )

    def split_lora_fc_names(self):
        """Splits the LoRA target_modules fully connected layer names into individual components."""
        logger.debug('[LoRA] Enter split_lora_fc_names')
        all_qkv_fcs = ['qkv_proj', 'W_pack', 'c_attn']
        all_gateup_fcs = ['gate_up_proj', 'w12']

        logger.debug(f'target_modules before splitting={self.target_modules}')
        final_target_modules = []
        for target in self.target_modules:
            if target in all_qkv_fcs:
                final_target_modules.extend(['q_proj', 'k_proj', 'v_proj'])
            elif target in all_gateup_fcs:
                if target == 'gate_up_proj':
                    final_target_modules.extend(['gate_proj', 'up_proj'])
                else:
                    final_target_modules.extend(['w1', 'w2'])
            else:
                final_target_modules.append(target)
        logger.debug(f'target_modules after splitting={final_target_modules}')
        self.target_modules = final_target_modules
