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
"""Define precision configuration of pipeline."""

import json
import os

from . import const, logger, precision_config_utils, utils
from . import sanity_checks as sc


class PTQPrecisionConfig:
    """PTQPrecisionConfig class for handling pipeline precision settings for PTQ.

    Attributes:
        encoder_precision (str): The encoder precision, if any.
        llm_precision (dict): The per-layer-and-segment LLM precision dict.
        tail_precision (str): The tail precision.
        lora_precision (str): The LoRA inputs precision, defaults to LLM activation precision.
        mask_precision (str): The mask input precision, defaults to LLM activation precision.
        cache_precision (str): The cache input/output precision, defaults to LLM activation precision.
        embeds_precision (str): The embedding input precision, defaults to LLM activation precision.
        logits_precision (str): The logits output precision, defaults to LLM activation precision.
        bypassed_ops (list): The OPs to skip quantization during PTQ, if any.
    """

    all_valid_converter_precisions = utils.get_converter_precisions()
    all_valid_converter_ops = utils.get_converter_ops()
    all_valid_standalone_precisions = ['int4', 'int8', 'int16', 'float']
    # mapped to converter precisions
    converter_precision_mapping = {
        'int4': 'sym4W_sym4A',
        'int8': 'sym4W_sym8A',
        'int16': 'sym4W_sym16A',
        'float': 'FP',
    }

    # map from converter precision back to shorthand precision
    config_precision_mapping = {v: k for k, v in converter_precision_mapping.items()}

    def __init__(
        self,
        pipeline_config,
        precision_config_path_or_dict,
        aopt=None,
        lora_handler=None,
        use_single_bmm_attention=False,
        use_opt_separate_bmm=False,
    ):
        """Initialize the PTQPrecisionConfig.

        Args:
            pipeline_config (PipelineConfig): The Pipeline configuration object.
            precision_config_path_or_dict (str or dict): The path to json file of the pipeline precision config.
                Generates default precision config if None.
            aopt (int, optional): aopt precision option.
            lora_handler (object, optional): Lora handler of current pipeline.
            use_single_bmm_attention (bool, optional): Boolean indicating whether to use single bmm attention.
            use_opt_separate_bmm (bool, optional): Boolean indicating whether to use optimization for separate bmm.
        """
        self.has_lora = lora_handler is not None and (lora_handler.has_llm_lora() or lora_handler.has_encoder_lora())
        logger.debug(
            '[PTQPrecisionConfig] Initialize PTQPrecisionConfig. precision_config_path_or_dict='
            f'{precision_config_path_or_dict}, has_lora={self.has_lora}'
        )
        self.config = pipeline_config
        self.lora_handler = lora_handler
        self.use_single_bmm_attention = use_single_bmm_attention  # only needed for setting kv cache precision
        self.use_opt_separate_bmm = use_opt_separate_bmm
        self.llm_num_hidden_layers = (  # Attr needed for _generate_default_config_and_exit()
            self.config.l.early_exit_index + self.config.l.early_exit_num_layers
            if self.config.l.early_exit_index is not None
            else self.config.l.num_hidden_layers
        )
        if precision_config_path_or_dict is None:
            self._generate_default_config_and_exit()
        if isinstance(precision_config_path_or_dict, dict):
            self._load_from_dict(precision_config_path_or_dict)
            return

        if not isinstance(precision_config_path_or_dict, str):
            logger.error(
                '[PTQPrecisionConfig] Expected dict or str for `precision_config_path_or_dict`, but got '
                f'{type(precision_config_path_or_dict)}',
                err=TypeError,
            )
        sc.check_exist(precision_config_path_or_dict, 'Precision config json')

        self.name = os.path.splitext(os.path.basename(precision_config_path_or_dict))[0]
        if aopt is not None:
            self.name += f'_aopt_{aopt}'
        logger.debug(f'[PTQPrecisionConfig] name={self.name}')

        with open(precision_config_path_or_dict) as f:
            precision_config = json.load(f)

        self.aopt = aopt

        # Set Encoder precision
        encoder_precision = precision_config.pop('encoder', None)
        if self.config.e is None and encoder_precision is not None:
            logger.error(
                '[PTQPrecisionConfig] Pipeline has no encoder but encoder precision is given in precision config.'
            )
        if self.config.e is not None and encoder_precision is None:
            logger.error(
                '[PTQPrecisionConfig] Pipeline has encoder but encoder precision is not given in precision config.'
            )
        if encoder_precision is not None and encoder_precision not in self.all_valid_converter_precisions:
            logger.error(
                f'[PTQPrecisionConfig] {encoder_precision} is not a valid precision. List of all valid precisions: '
                f'{self.all_valid_converter_precisions}',
                err=ValueError,
            )
        self.encoder_precision = encoder_precision
        logger.debug(f'[PTQPrecisionConfig] encoder_precision={self.encoder_precision}')
        if self.encoder_precision is not None:
            encoder_weight_bits = set()
            # encoder currently fixed precision only
            encoder_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                self.encoder_precision, return_bits=True
            )
            encoder_weight_bits.add(encoder_weight_bit)
            self.max_encoder_weight_bit = max(encoder_weight_bits)
            logger.info(f'[PTQPrecisionConfig] max_encoder_weight_bit={self.max_encoder_weight_bit}')
        else:
            self.max_encoder_weight_bit = None

        self.max_llm_weight_bit = 0

        # Set LLM precision
        llm_precision = precision_config.pop('llm', None)
        if llm_precision is None:
            logger.error('[PTQPrecisionConfig] LLM precision is compulsory but is not given in precision config.')
        if isinstance(llm_precision, str):
            llm_precision_dict = {f'0-{self.llm_num_hidden_layers - 1}': llm_precision}
        else:
            llm_precision_dict = llm_precision
        logger.debug(f'[PTQPrecisionConfig] llm_precision_dict={llm_precision_dict}')
        assert isinstance(llm_precision_dict, dict), f'Expected dict or str but got {type(llm_precision_dict)}'
        layer_idx_set = set(range(self.llm_num_hidden_layers))

        # Ensure all LLM layers are accounted for
        llm_precision_dict_perlayer = {}
        seen = set()
        for k, v in llm_precision_dict.items():
            k = k.strip()
            if '-' in k:
                if k.count('-') > 1:
                    logger.error(
                        f'[PTQPrecisionConfig] Invalid layer format: {k}, expect at most 1 `-`', err=ValueError
                    )
                lower, upper = k.split('-')
                try:
                    subrange = set(range(int(lower), int(upper) + 1))
                except ValueError:
                    logger.error(
                        f'[PTQPrecisionConfig] Invalid layer format: {k}, expect range formats to be of the form: '
                        '`X-Y`, where X and Y are both integers',
                        err=ValueError,
                    )
            elif ',' in k:
                if k.endswith(','):
                    k = k[:-1]
                try:
                    subrange = {int(x) for x in k.split(',')}
                except ValueError:
                    logger.error(
                        f'[PTQPrecisionConfig] Invalid layer format: {k}, expect comma formats to be of the form: '
                        '`X,Y,...`, which should only contain commas and integers',
                        err=ValueError,
                    )
            else:
                try:
                    subrange = {
                        int(k),
                    }
                except ValueError:
                    logger.error(
                        f'[PTQPrecisionConfig] Invalid layer format: {k}, expect integer formats to be of the form: '
                        '`X`, where X must be an integer',
                        err=ValueError,
                    )
            if bool(seen.intersection(subrange)):
                logger.error(
                    f'[PTQPrecisionConfig] Repeated layer indices found: {seen.intersection(subrange)}', err=ValueError
                )
            if not subrange.issubset(layer_idx_set):
                logger.error(
                    f'[PTQPrecisionConfig] {subrange} is not a subset of remaining layer indices: {layer_idx_set}. '
                    'Ensure your indices are zero-based (From 0 to num_hidden_layers - 1).',
                    err=ValueError,
                )
            for idx in subrange:
                llm_precision_dict_perlayer[idx] = v
            layer_idx_set = layer_idx_set - subrange
            seen.update(subrange)
        if len(layer_idx_set) > 0:
            logger.error(f'[PTQPrecisionConfig] Not all LLM layers accounted for. Missing layers: {layer_idx_set}')
        del llm_precision_dict

        # Parse per-layer precision dict
        invalid_precisions = set()
        self.llm_unique_precisions = set()
        attn_fcs = [x for x in self.config.l.fc_names['attn'] if x not in ['name', 'qkv']]
        mlp_fcs = [x for x in self.config.l.fc_names['mlp'] if x not in ['name', 'gateup']]
        all_fcs = attn_fcs + mlp_fcs
        self.mode = None
        for i in range(self.llm_num_hidden_layers):
            if isinstance(llm_precision_dict_perlayer[i], str):
                layer_precision = llm_precision_dict_perlayer[i]
                if layer_precision.endswith('.json'):
                    # Path to converter v1/v2 precision config
                    if self.mode == 'manual':
                        logger.error('Cannot mix manual precision settings with mtk_converter precision config json.')
                    self.mode = 'converter_config'
                    sc.check_exist(layer_precision, f'layer {i} mtk_converter precision config json')
                    with open(layer_precision) as f:
                        cvtr_precision_config = json.load(f)
                    if 'precision_specs' in cvtr_precision_config:
                        for spec in cvtr_precision_config['precision_specs']:
                            spec_precision = spec['precision_name']
                            if spec_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                                invalid_precisions.add(spec_precision)
                            else:
                                # not sure how to check for fc for this
                                llm_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                                    spec_precision, return_bits=True
                                )
                                self.max_llm_weight_bit = max(self.max_llm_weight_bit, llm_weight_bit)
                            self.llm_unique_precisions.add(spec_precision)
                    elif 'precision_hints' in cvtr_precision_config:
                        default_precision = cvtr_precision_config['precision_hints']['default_precision']
                        if default_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                            invalid_precisions.add(default_precision)
                        self.llm_unique_precisions.add(default_precision)
                        if cvtr_precision_config['version'] in ['v1', 'v2']:
                            for hint_dict in cvtr_precision_config['precision_hints']['hints'].values():
                                for hint_precision in hint_dict.values():
                                    if hint_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                                        invalid_precisions.add(hint_precision)
                                    else:
                                        # not sure how to check for fc for this
                                        llm_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                                            hint_precision, return_bits=True
                                        )
                                        self.max_llm_weight_bit = max(self.max_llm_weight_bit, llm_weight_bit)
                                    self.llm_unique_precisions.add(hint_precision)
                        else:
                            llm_precision_dict_perlayer[i] = {'precision_config_path': layer_precision}
                            for hint_dict in cvtr_precision_config['precision_hints']['hints']:
                                hint_target_op_type = hint_dict['target_op_types']
                                hint_target_op_name = hint_dict['target_op_output_name']
                                for precision_type in ['input_precision', 'output_precision']:
                                    try:
                                        hint_precision = hint_dict[precision_type]
                                    except KeyError:
                                        logger.debug(f'[PTQPrecisionConfig] missing {precision_type}')
                                        if (
                                            precision_type == 'output_precision'
                                            and hint_target_op_type == 'FullyConnected'
                                        ):
                                            for fc in all_fcs:
                                                if fc + '_proj' in hint_target_op_name:
                                                    logger.debug(
                                                        f'[PTQPrecisionConfig] Adding default precision '
                                                        f'{default_precision} for {fc}_proj output_precision'
                                                    )
                                                    llm_precision_dict_perlayer[i].update({fc: default_precision})
                                                    break
                                            continue

                                    if hint_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                                        invalid_precisions.add(hint_precision)
                                    else:
                                        if hint_target_op_type == 'FullyConnected':
                                            llm_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                                                hint_precision, return_bits=True
                                            )
                                            # we can assume i/o precision for FC op will always have same weight bit
                                            self.max_llm_weight_bit = max(self.max_llm_weight_bit, llm_weight_bit)
                                        if precision_type == 'output_precision':
                                            # only need to save fc output prescision to llm_precision
                                            for fc in all_fcs:
                                                if fc + '_proj' in hint_target_op_name:
                                                    llm_precision_dict_perlayer[i].update({fc: hint_precision})

                                    self.llm_unique_precisions.add(hint_precision)
                    else:
                        logger.error(
                            f'Expected {layer_precision} to contain either "precision_specs" or "precision_hints" key.'
                        )
                else:
                    # NOTE: Currently using this branch
                    # Layer-wide single precision
                    if self.mode == 'converter_config':
                        logger.error('Cannot mix manual precision settings with mtk_converter precision config json.')
                    self.mode = 'manual'
                    llm_precision_dict_perlayer[i] = {}
                    for fc in all_fcs:
                        logger.debug(f'[PTQPrecisionConfig] Layer {i}: Set {fc} FC to {layer_precision}')
                        llm_precision_dict_perlayer[i].update({fc: layer_precision})
                    if layer_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                        invalid_precisions.add(layer_precision)
                    else:
                        llm_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                            layer_precision, return_bits=True
                        )
                        self.max_llm_weight_bit = max(self.max_llm_weight_bit, llm_weight_bit)
                    self.llm_unique_precisions.add(layer_precision)
            else:
                if not isinstance(llm_precision_dict_perlayer[i], dict):
                    logger.error(
                        f'[PTQPrecisionConfig] Layer {i}: Expected dict or str but got '
                        f'{type(llm_precision_dict_perlayer[i])}'
                    )
                self.mode = 'manual'
                # Check if both attn/mlp precision and individual FC precision specified. Should only have either/or.
                accounted_attn_fcs = []
                accounted_mlp_fcs = []
                for key in llm_precision_dict_perlayer[i]:
                    if key == 'attention':
                        if len(accounted_attn_fcs) > 0:
                            logger.error(
                                'Per-layer precision keys cannot overlap. '
                                'Either use "attention" to cover all attention FCs, '
                                'or specify individual FC precisions, not both at the same time.'
                            )
                        accounted_attn_fcs = attn_fcs
                    elif key == 'mlp':
                        if len(accounted_mlp_fcs) > 0:
                            logger.error(
                                'Per-layer precision keys cannot overlap. '
                                'Either use "mlp" to cover all MLP FCs, or specify individual FC precisions, '
                                'not both at the same time.'
                            )
                        accounted_mlp_fcs = mlp_fcs
                    else:
                        if key in attn_fcs:
                            if key in accounted_attn_fcs:
                                logger.error(
                                    'Per-layer precision keys cannot overlap. '
                                    'Either use "attention" to cover all attention FCs, '
                                    'or specify individual FC precisions, not both at the same time.'
                                )
                            accounted_attn_fcs.append(key)
                        elif key in mlp_fcs:
                            if key in accounted_mlp_fcs:
                                logger.error(
                                    'Per-layer precision keys cannot overlap. '
                                    'Either use "mlp" to cover all MLP FCs, or specify individual FC precisions, '
                                    'not both at the same time.'
                                )
                            accounted_mlp_fcs.append(key)
                        else:
                            logger.error(
                                f'Unknown key: {key}. Recognized keys: {all_fcs + const.SUPPORTED_PRECISION_TARGETS}'
                            )

                # Check if all FCs are accounted for in the precision config, else their precisions become ambiguous.
                if accounted_attn_fcs != attn_fcs:
                    logger.error(
                        f'[PTQPrecisionConfig] Layer {i}: Some attention FCs are missing in the precision config: '
                        f'{set(attn_fcs) - set(accounted_attn_fcs)}'
                    )
                if accounted_mlp_fcs != mlp_fcs:
                    logger.error(
                        f'[PTQPrecisionConfig] Layer {i}: Some MLP FCs are missing in the precision config: '
                        f'{set(mlp_fcs) - set(accounted_mlp_fcs)}'
                    )

                curr_layer_keys = list(llm_precision_dict_perlayer[i].keys())
                for k in curr_layer_keys:
                    supported_precision_targets = all_fcs + const.SUPPORTED_PRECISION_TARGETS
                    if k not in supported_precision_targets:
                        logger.error(
                            f'[PTQPrecisionConfig] Layer {i}: {k} is not a supported precision target. '
                            f'Supported precision targets: {supported_precision_targets}',
                            err=KeyError,
                        )
                    attn_or_mlp_precision = None
                    if k in const.SUPPORTED_PRECISION_TARGETS:
                        # Further break down attn/mlp precision into individual FC precisions
                        logger.debug(f'[PTQPrecisionConfig] Layer {i}: Remove {k}')
                        attn_or_mlp_precision = llm_precision_dict_perlayer[i].pop(k)
                        if k == 'attention':
                            for fc in attn_fcs:
                                logger.debug(f'[PTQPrecisionConfig] Layer {i}: Set {fc} FC to {attn_or_mlp_precision}')
                                llm_precision_dict_perlayer[i].update({fc: attn_or_mlp_precision})
                        else:
                            assert k == 'mlp'
                            for fc in mlp_fcs:
                                logger.debug(f'[PTQPrecisionConfig] Layer {i}: Set {fc} FC to {attn_or_mlp_precision}')
                                llm_precision_dict_perlayer[i].update({fc: attn_or_mlp_precision})
                    curr_precision = attn_or_mlp_precision or llm_precision_dict_perlayer[i].get(k, None)
                    if curr_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
                        invalid_precisions.add(curr_precision)
                    else:
                        llm_weight_bit, _ = self.converter_to_standalone_precision_mapping(
                            curr_precision, return_bits=True
                        )
                        self.max_llm_weight_bit = max(self.max_llm_weight_bit, llm_weight_bit)
                    self.llm_unique_precisions.add(curr_precision)

        logger.info(f'Precision Config Mode: {self.mode}')

        if len(invalid_precisions) > 0:
            logger.error(
                f'[PTQPrecisionConfig] {invalid_precisions} are not valid precisions.\nList of all valid precisions: '
                f'{self.all_valid_converter_precisions}',
                err=ValueError,
            )

        logger.info(f'[PTQPrecisionConfig] max_llm_weight_bit={self.max_llm_weight_bit}')

        self.llm_precision = llm_precision_dict_perlayer
        logger.debug(f'[PTQPrecisionConfig] llm_precision={self.llm_precision}')

        # Set Tail precision
        tail_precision = precision_config.pop('tail', None)
        if tail_precision is None:
            logger.error('[PTQPrecisionConfig] Tail precision is compulsory but is not given in precision config.')
        if tail_precision.split('_vsq')[0] not in self.all_valid_converter_precisions:
            logger.error(
                f'[PTQPrecisionConfig] {tail_precision} is not a valid precision. List of all valid precisions: '
                f'{self.all_valid_converter_precisions}',
                err=ValueError,
            )
        self.tail_precision = tail_precision
        logger.debug(f'[PTQPrecisionConfig] tail_precision={self.tail_precision}')

        # Set default activation precision
        default_precision = precision_config.pop('default', None)
        if default_precision is None:
            if 'FP' in self.llm_unique_precisions or any(
                ele.endswith('dynamic_quant') for ele in self.llm_unique_precisions
            ):
                default_precision = 'float'
            else:
                default_precision = 'int16'
        self.default_precision = self._set_valid_precision(default_precision, 'default')
        logger.debug(f'[PTQPrecisionConfig] default activation precision={self.default_precision}')

        # Set LoRA input precision
        lora_precision = precision_config.pop('lora', default_precision)
        self.lora_precision = self._set_valid_precision(lora_precision, 'lora')
        logger.debug(f'[PTQPrecisionConfig] lora_precision={self.lora_precision}')
        if self.default_precision != 'FP' and self.lora_precision == 'FP':
            self.llm_unique_precisions.add(self.lora_precision)

        # Set softmax precision
        softmax_precision = precision_config.pop('softmax', None)
        if softmax_precision is None:
            softmax_precision = default_precision
        self.softmax_precision = self._set_valid_precision(softmax_precision, 'softmax')
        logger.debug(f'[PTQPrecisionConfig] softmax_precision={self.softmax_precision}')
        if self.default_precision != 'FP' and self.softmax_precision == 'FP':
            self.llm_unique_precisions.add(self.softmax_precision)

        # Set Mask input precision
        mask_precision = precision_config.pop('mask', None)
        if mask_precision is None:
            mask_precision = default_precision
        self.mask_precision = self._set_valid_precision(mask_precision, 'mask')
        logger.debug(f'[PTQPrecisionConfig] mask_precision={self.mask_precision}')
        if (
            'FP' not in (self.default_precision, self.softmax_precision)
            and self.mask_precision != self.default_precision
        ):
            logger.info('[PTQPrecisionConfig] Adding mask_precision to llm_unique_precisions')
            # either mask or cache will be added. does not matter if duplicate. Main purpose is to classify as MP
            self.llm_unique_precisions.add(self.mask_precision)

        # Set Cache input/output precision
        cache_precision = precision_config.pop('cache', None)
        if cache_precision is None:
            cache_precision = default_precision
        self.cache_precision = self._set_valid_precision(cache_precision, 'cache')
        logger.debug(f'[PTQPrecisionConfig] cache_precision={self.cache_precision}')
        if (
            'FP' not in (self.default_precision, self.softmax_precision)
            and self.cache_precision != self.default_precision
        ):
            logger.info('[PTQPrecisionConfig] Adding cache_precision to llm_unique_precisions')
            # either mask or cache will be added. does not matter if duplicate. Main purpose is to classify as MP
            self.llm_unique_precisions.add(self.cache_precision)

        # Set input embeddings precision
        embeds_precision = precision_config.pop('embeddings', None)
        if embeds_precision is None:
            embeds_precision = default_precision
        self.embeds_precision = self._set_valid_precision(embeds_precision, 'embeddings')
        logger.debug(f'[PTQPrecisionConfig] embeds_precision={self.embeds_precision}')

        # Set output logits precision
        logits_precision = precision_config.pop('logits', None)
        if logits_precision is None:
            logits_precision = default_precision
        self.logits_precision = self._set_valid_precision(logits_precision, 'logits')
        logger.debug(f'[PTQPrecisionConfig] logits_precision={self.logits_precision}')

        # Set residual path precision
        respath_precision = precision_config.pop('res_path', None)
        if respath_precision is None:
            respath_precision = default_precision
        self.respath_precision = self._set_valid_precision(respath_precision, 'res_path')
        # Respath FP is only unique if default activation is an int
        if (
            self.respath_precision not in self.llm_unique_precisions
            and self.respath_precision == 'FP'
            and self.default_precision != 'FP'
        ):
            logger.info('[PTQPrecisionConfig] Adding respath_precision to llm_unique_precisions')
            self.llm_unique_precisions.add(self.respath_precision)
        if self.embeds_precision == self.logits_precision and self.respath_precision != self.embeds_precision:
            logger.warning(
                f'Residual path precision ({self.respath_precision}) is different from input embeddings and output '
                f'logits precision ({self.embeds_precision}). This is suboptimal.'
            )
        logger.debug(f'[PTQPrecisionConfig] respath_precision={self.respath_precision}')

        # Set quantization-bypassed OPs
        bypassed_ops = precision_config.pop('bypass', [])
        for op in bypassed_ops:
            if op not in self.all_valid_converter_ops:
                logger.error(
                    f'[PTQPrecisionConfig] {op} is not a valid OP name. List of all valid OP names: '
                    f'{self.all_valid_converter_ops}',
                    err=KeyError,
                )
        self.bypassed_ops = bypassed_ops
        logger.debug(f'[PTQPrecisionConfig] bypassed_ops={self.bypassed_ops}')

        logger.debug(f'[PTQPrecisionConfig] llm_unique_precisions={self.llm_unique_precisions}')
        # Set infini update precision
        if self.config.l.infini_attention:
            # FIXME: hardcoded as float for now
            self.infini_update_precision = 'float'

        if self.is_llm_mixed_precision() and not sc.check_converter_version(9, soft=True):
            logger.error('Mixed precision is only supported on mtk_converter >=9.0.0')

    def _generate_default_config_and_exit(self):
        logger.info('No precision config provided. Generating template precision config and exiting.')
        default_config = {}

        if self.config.e is not None:
            default_config['encoder'] = 'FP'
        default_config['llm'] = {f'0-{self.llm_num_hidden_layers - 1}': 'FP'}
        default_config['tail'] = 'FP'
        default_config['default'] = 'float'
        default_config['embeddings'] = 'float'
        default_config['mask'] = 'float'
        default_config['logits'] = 'float'
        default_config['cache'] = 'float'
        if self.has_lora:
            default_config['lora'] = 'float'
        default_config['bypass'] = []

        with open('template_precision_config.json', 'w') as f:
            f.write(json.dumps(default_config, indent=4))
        logger.info(
            'Template precision config generated at: ./template_precision_config.json. '
            'Please refer to document for full list of precision choices.'
        )
        exit()

    @classmethod
    def _set_valid_precision(cls, precision, name):
        logger.debug(f'[PTQPrecisionConfig] Setting precision for {name}')
        if precision is None:
            return None
        return cls._resolve_to_converter_precision(precision)

    def _load_from_dict(self, precision_dict):
        for k, v in precision_dict.items():
            setattr(self, k, v)

    @classmethod
    def _resolve_to_converter_precision(cls, precision):
        if precision in cls.all_valid_standalone_precisions:
            return cls.converter_precision_mapping[precision]
        if precision in cls.all_valid_converter_precisions:
            return precision
        logger.error(
            f'[PTQPrecisionConfig] {precision} is not a valid precision. '
            'List of all valid precisions:\n'
            f'{cls.all_valid_standalone_precisions + cls.all_valid_converter_precisions}',
            err=ValueError,
        )
        return None

    @classmethod
    def get_bitwidth(cls, precision):
        """Returns weight and activation bitwidth.

        Args:
            precision (str): Precision string. Either standalone precision or converter precision

        Returns:
            tuple: The weight and activation bitwidth.
        """
        cvtr_precision = cls._resolve_to_converter_precision(precision)
        return cls.converter_to_standalone_precision_mapping(cvtr_precision, return_bits=True)

    @classmethod
    def get_precision_name(cls, precision):
        """Get the standalone precision name (int<bitwidth>).

        Args:
            precision (str): Precision string. Either standalone precision or converter precision

        Returns:
            tuple: The weight and activation standalone precision name.
        """
        cvtr_precision = cls._resolve_to_converter_precision(precision)
        return cls.converter_to_standalone_precision_mapping(cvtr_precision)

    @classmethod
    def get_symmetric_setting(cls, precision):
        """Returns weight and activation symmetric setting.

        Args:
            precision (str): Precision string. Either standalone precision or converter precision

        Returns:
            tuple: The weight and activation symmetric setting.
        """
        cvtr_precision = cls._resolve_to_converter_precision(precision)
        return cls.converter_to_standalone_precision_mapping(cvtr_precision, return_symmetric_setting=True)

    @classmethod
    def converter_to_standalone_precision_mapping(
        cls, converter_precision, return_bits=False, return_symmetric_setting=False
    ):
        """Converts mtk_converter precision format to standalone weight and activation format.

        Args:
            converter_precision (str): A valid mtk_converter precision string.
            return_bits (bool, optional): Whether to return the number of bits instead. Defaults to False.
            return_symmetric_setting (bool, optional): Whether to return the symmetric setting instead.
            Defaults to False.

        Returns:
        if return_bits:
            wgt_bit (int): Standalone weight bits.
            act_bit (int): Standalone activation bits.
        else:
            weight (str): Standalone weight precision.
            activation (str): Standalone activation precision.
        """
        if converter_precision.split('_vsq')[0] not in cls.all_valid_converter_precisions:
            logger.error(
                f'{converter_precision} is not a valid precision. List of all valid precisions: '
                f'{cls.all_valid_converter_precisions}',
                err=ValueError,
            )
        if converter_precision == 'FP':
            if return_bits:
                return 32, 32
            return 'float', 'float'
        if converter_precision == 'dynamic_quant':
            if return_bits:
                return 8, 32
            if return_symmetric_setting:
                return 'sym', 'asym'
            return 'int8', 'float'

        converter_precision = converter_precision.split('_vsq')[0]

        precision_group = converter_precision.split('_')

        weight_precision = precision_group[0].split('W')[0]
        act_precision = 'dynamic_quant' if len(precision_group[1:]) > 1 else precision_group[1].split('A')[0]

        weight_symmetric_setting = 'sym'
        act_symmetric_setting = 'sym'
        if 'sym' in weight_precision:
            wgt_bit = int(weight_precision.split('sym')[-1])
        elif 'asym' in weight_precision:
            wgt_bit = int(weight_precision.split('asym')[-1])
            weight_symmetric_setting = 'asym'
        else:
            logger.error(f'Invalid symmetric setting for weight: {weight_precision}')

        weight = f'int{wgt_bit}'

        if act_precision == 'dynamic_quant':
            activation = 'float'
            act_bit = 32
        else:
            if 'sym' in act_precision:
                act_bit = int(act_precision.split('sym')[-1])
            elif 'asym' in act_precision:
                act_bit = int(act_precision.split('asym')[-1])
                act_symmetric_setting = 'asym'
            else:
                logger.error(f'Invalid symmetric setting for act: {act_precision}')

            activation = f'int{act_bit}'

        if return_bits:
            return wgt_bit, act_bit
        if return_symmetric_setting:
            return weight_symmetric_setting, act_symmetric_setting
        return weight, activation

    def is_activation_static_quant(self):
        """Determine if any component of pipeline requires calibration dataset.

        Returns:
            Dict containing 3 str:bool pairs indicating whether encoder/llm/tail needs calibration dataset or not.
        """

        def _is_activation_static_quant(precision):
            if precision is None:
                return None
            if precision == 'FP':
                return False
            return not precision.endswith('dynamic_quant')

        # Any OP precision not in ('FP', 'dynamic_quant') needs calibration data for PTQ.
        llm_need_calib = False
        for _ in range(self.llm_num_hidden_layers):
            for v in self.llm_unique_precisions:
                if v != 'FP' and not v.endswith('dynamic_quant'):
                    llm_need_calib = True
                    break
            if llm_need_calib:
                break

        return {
            'encoder': _is_activation_static_quant(self.encoder_precision),
            'llm': llm_need_calib,
            'tail': _is_activation_static_quant(self.tail_precision),
        }

    def is_llm_mixed_precision(self):
        """Returns a boolean whether the LLM has more than 1 unique precision."""
        return len(self.llm_unique_precisions) > 1

    def generate_converter_precision_config(self, global_layer_idx, local_layer_idx=0):
        """Generates mtk_converter precision hint style precision config of a specified LLM layer.

        Args:
            global_layer_idx (int): The layer index to generate the precision config for.
            local_layer_idx (int): The layer index inside the config to start from, should correspond to the index
                within a chunk. Defaults to 0 (per-layer config).

        Returns:
            config (dict): mtk_converter style precision config.
        """
        if global_layer_idx > self.llm_num_hidden_layers:
            logger.error(
                f'`layer_idx` cannot be greater than the number of LLM layers in model ({self.llm_num_hidden_layers})'
            )
        config = {'version': 'v3', 'precision_hints': {'default_precision': self.default_precision, 'hints': []}}
        if global_layer_idx == self.llm_num_hidden_layers:
            config['precision_hints']['default_precision'] = self.tail_precision
            return config

        layer_dict = self.llm_precision[global_layer_idx]
        logger.debug(f'[PTQPrecisionConfig] Layer Dict: {layer_dict}')

        fc_names = self.config.l.fc_names
        attn_name_mappings = {k: v for (k, v) in fc_names['attn'].items() if k not in ['name', 'qkv']}
        mlp_name_mappings = {k: v for (k, v) in fc_names['mlp'].items() if k not in ['name', 'gateup']}

        mask_scaling_factors = self.config.l.mask_scaling_factors

        o_precision = layer_dict['o']
        down_precision = layer_dict['down']

        if self.aopt is not None:
            precision_config_utils.overwrite_precision_hint(layer_dict, self.aopt)

        is_fp_respath = self.respath_precision == self.embeds_precision == self.logits_precision == 'FP'

        if self.config.l.model_type == 'gecko2' and self.config.l.rotqat:
            config['precision_hints']['hints'].extend(
                [
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/mlp/gate_up_proj_rotqat.*',
                        'target_op_types': 'FullyConnected',
                        'input_precision': 'sym16W_sym16A',
                        'output_precision': 'sym16W_sym16A',
                    },
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/mlp/down_proj_rotqat.*',
                        'target_op_types': 'FullyConnected',
                        'input_precision': 'sym16W_sym16A',
                        'output_precision': 'sym16W_sym16A',
                    },
                ]
            )

        # FIXME: Currently force internlm2 qkv to follow o_precision for i/o
        if self.config.l.model_type == 'internlm2':
            config['precision_hints']['hints'].append(
                {
                    'target_op_output_name': f'layers/{local_layer_idx}/self_attn/qkv_proj.*',
                    'target_op_types': 'FullyConnected',
                    'input_precision': o_precision,
                    'output_precision': o_precision,
                }
            )

        # NOTE: Currently does not handle merged fc proj mappings
        for fc, precision in layer_dict.items():
            if fc in attn_name_mappings:
                if fc == 'o':
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/o_proj.*',
                            'target_op_types': 'FullyConnected',
                            'input_precision': precision,
                            'output_precision': self.respath_precision,
                        }
                    )
                elif fc == 'v':
                    # for 16b8b single bmm case
                    if self.get_precision_name(self.cache_precision)[1] == 'int8' and (
                        self.use_single_bmm_attention or self.use_opt_separate_bmm
                    ):
                        if (
                            self.lora_handler.has_llm_lora()
                            and 'v_proj' in self.lora_handler.global_llm_target_modules
                            and global_layer_idx >= self.lora_handler.global_llm_start_idx
                            and global_layer_idx <= self.lora_handler.global_llm_end_idx
                        ):
                            v_out_precision = self.default_precision  # v_lora_add will be the cache precision
                        else:
                            v_out_precision = self.cache_precision
                    else:
                        v_out_precision = o_precision
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/v_proj.*',
                            'target_op_types': 'FullyConnected',
                            'input_precision': precision,
                            'output_precision': v_out_precision,
                        }
                    )
                else:
                    # q, k
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/{fc}_proj.*',
                            'target_op_types': 'FullyConnected',
                            'input_precision': precision,
                            'output_precision': o_precision,
                        }
                    )
            elif fc in mlp_name_mappings:
                if fc == 'down':
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/mlp/down_proj/.*',
                            'target_op_types': 'FullyConnected',
                            'input_precision': precision,
                            'output_precision': self.respath_precision,
                        }
                    )
                elif fc == 'gate':
                    # FIXME: For now, use gate proj FP output for llama model only
                    if self.config.l.model_type == 'llama':
                        # set gate proj to FP
                        config['precision_hints']['hints'].append(
                            {
                                'target_op_output_name': f'layers/{local_layer_idx}/mlp/gate_proj/.*',
                                'target_op_types': 'FullyConnected',
                                'input_precision': precision,
                                'output_precision': 'FP',
                            }
                        )
                        config['precision_hints']['hints'].append(
                            {
                                'target_op_output_name': f'layers/{local_layer_idx}/mlp/mul2/mul',
                                'target_op_types': 'Silu',
                                'output_precision': down_precision,
                            }
                        )
                    else:
                        # Note: .* after gate_proj is for n2v2
                        config['precision_hints']['hints'].append(
                            {
                                'target_op_output_name': f'layers/{local_layer_idx}/mlp/gate_proj.*/.*',
                                'target_op_types': 'FullyConnected',
                                'input_precision': precision,
                                'output_precision': down_precision,
                            }
                        )
                else:
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/mlp/{fc}_proj.*',
                            'target_op_types': 'FullyConnected',
                            'input_precision': precision,
                            'output_precision': down_precision,
                        }
                    )
            else:
                logger.error(
                    f'Unexpected key: {fc}. Expected keys: '
                    f'{list(attn_name_mappings.keys()) + list(mlp_name_mappings.keys())}'
                )

        # Non FC precision hints

        # lora float case
        if (
            self.lora_handler.has_llm_lora()
            and not (
                global_layer_idx < self.lora_handler.global_llm_start_idx
                or global_layer_idx > self.lora_handler.global_llm_end_idx
            )
            and self.lora_precision == 'FP'
        ):
            for module in self.lora_handler.global_llm_target_modules:
                lora_mod = module[:-5]
                parent_mod = 'self_attn' if lora_mod in ['q', 'k', 'v', 'o', 'qkv'] else 'mlp'
                # # o and down lora A will be handled separately for fp respath case

                lora_b_out_precision = (
                    self.default_precision
                    if not is_fp_respath or (is_fp_respath and lora_mod not in ['o', 'down'])
                    else 'FP'
                )
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/{parent_mod}/{lora_mod}_lora_A/matmul',
                        'target_op_types': 'MatMul',
                        'input_precision': self.lora_precision,
                        'output_precision': self.lora_precision,
                    }
                )
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/{parent_mod}/{lora_mod}_lora_B/matmul',
                        'target_op_types': 'MatMul',
                        'input_precision': self.lora_precision,
                        'output_precision': lora_b_out_precision,
                    }
                )

        # FIXME: temp use getattr until split mask PR merged
        use_split_mask = getattr(self.config.l, 'use_split_mask', False)

        # fp softmax + split mask / single bmm
        if self.default_precision != 'FP' and self.softmax_precision == 'FP':
            if self.config.l.cache_evict.get('in_graph') is False:  # If cache evict outgraph
                softmax_target_op_name = 'to'
            else:
                softmax_target_op_name = f'layers/{local_layer_idx}/self_attn/softmax'

            config['precision_hints']['hints'].append(
                {
                    'target_op_output_name': softmax_target_op_name,
                    'target_op_types': 'Softmax',
                    'input_precision': self.softmax_precision,
                    'output_precision': self.default_precision,
                }
            )

            # BMM output
            config['precision_hints']['hints'].append(
                {
                    'target_op_output_name': f'layers/{local_layer_idx}/self_attn/matmul1/matmul',
                    'target_op_types': 'MatMul',
                    'output_precision': self.softmax_precision,
                }
            )

            if not self.use_single_bmm_attention:
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/matmul2/matmul',
                        'target_op_types': 'MatMul',
                        'output_precision': self.softmax_precision,
                    }
                )

            # FIXME: Temp force mask to be float.
            if use_split_mask:
                # mask
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add_mask_current/add',
                        'target_op_types': 'Add',
                        'input_precision': self.softmax_precision,
                        'output_precision': self.softmax_precision,
                    }
                )
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add_mask_cache/add',
                        'target_op_types': 'Add',
                        'input_precision': self.softmax_precision,
                        'output_precision': self.softmax_precision,
                    }
                )

                if self.use_single_bmm_attention:
                    # QK Mul
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/div/div.*',
                            'target_op_types': 'Mul',
                            'input_precision': self.softmax_precision,
                            'output_precision': self.softmax_precision,
                        }
                    )
                else:
                    # QK Mul
                    # Cache
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/div_cache/div.*',
                            'target_op_types': 'Mul',
                            'input_precision': self.softmax_precision,
                            'output_precision': self.softmax_precision,
                        }
                    )

                    # Current
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/div_current/div.*',
                            'target_op_types': 'Mul',
                            'input_precision': self.softmax_precision,
                            'output_precision': self.softmax_precision,
                        }
                    )
            else:
                # mask
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add/add',
                        'target_op_types': 'Add',
                        'input_precision': self.softmax_precision,
                        'output_precision': self.softmax_precision,
                    }
                )

                # QK Mul
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/div/div.*',
                        'target_op_types': 'Mul',
                        'input_precision': self.softmax_precision,
                        'output_precision': self.softmax_precision,
                    }
                )

        elif 'FP' not in (self.default_precision, self.softmax_precision):
            mask_scaling_factor = mask_scaling_factors[global_layer_idx] if mask_scaling_factors is not None else 1
            if mask_scaling_factor != 1:
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/mask_scaling_mul/mul',
                        'target_op_types': 'Mul',
                        'input_precision': self.mask_precision,
                        'output_precision': self.mask_precision,
                    }
                )
            if not self.use_single_bmm_attention and use_split_mask:
                # Non FP non default precision mask
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add_mask_current/add',
                        'target_op_types': 'Add',
                        'input_precision': self.mask_precision,
                    }
                )

                # Non FP non default precision mask
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add_mask_cache/add',
                        'target_op_types': 'Add',
                        'input_precision': self.mask_precision,
                    }
                )
                if mask_scaling_factor != 1:
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': f'layers/{local_layer_idx}/self_attn/mask_cache_scaling_mul/mul',
                            'target_op_types': 'Mul',
                            'input_precision': self.mask_precision,
                            'output_precision': self.mask_precision,
                        }
                    )
            else:
                # Non FP non default precision mask
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/add/add',
                        'target_op_types': 'Add',
                        'input_precision': self.mask_precision,
                    }
                )

        if self.get_precision_name(self.cache_precision)[1] == 'int8':
            # force k and v concat to have input precision of int8
            if self.use_single_bmm_attention:
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/cat.*/cat',
                        'target_op_types': 'Concat',
                        'input_precision': self.cache_precision,
                        'output_precision': self.cache_precision,
                    }
                )
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/k_add/add',
                        'target_op_types': 'Add',
                        'output_precision': self.cache_precision,
                    }
                )
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/matmul2/matmul',
                        'target_op_types': 'MatMul',
                        'output_precision': self.default_precision,
                    }
                )
            else:
                # when using sink rope in cache eviction, the last output OP is NOT Add
                # in this case, no need to use 16b x 8b BMM for matmul1 (past key) and matmul2 (cur_key)
                # as the apply rot embedding part is in default precision (int16)
                if not self.config.l.extra_input.get('sink_rope', False) and self.use_opt_separate_bmm:
                    config['precision_hints']['hints'].append(
                        {
                            'target_op_output_name': 'to..*',
                            'target_op_types': 'Add',
                            'output_precision': self.cache_precision,
                        }
                    )

            if (
                self.lora_handler.has_llm_lora()
                and 'v_proj' in self.lora_handler.global_llm_target_modules
                and global_layer_idx >= self.lora_handler.global_llm_start_idx
                and global_layer_idx <= self.lora_handler.global_llm_end_idx
                and (self.use_single_bmm_attention or self.use_opt_separate_bmm)
            ):
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/v_lora_add/add',
                        'target_op_types': 'Add',
                        'output_precision': self.cache_precision,
                    }
                )

        if not self.use_single_bmm_attention:
            # force matmul4 - BMM(softmax output, cur_v) to always have default precision output
            # when using splitmask without fpsoftmax, the matmul4 will have a requant Op after it, which is undesirable
            config['precision_hints']['hints'].append(
                {
                    'target_op_output_name': f'layers/{local_layer_idx}/self_attn/matmul4/matmul',
                    'target_op_types': 'MatMul',
                    'output_precision': self.default_precision,
                }
            )

        # Handle fp respath + int lora case for o and down proj
        if (
            is_fp_respath
            and self.lora_handler.has_llm_lora()
            and self.lora_handler.global_llm_target_modules in ['o_proj', 'down_proj']
            and self.lora_handler.global_llm_start_idx <= global_layer_idx <= self.lora_handler.global_llm_end_idx
            and self.lora_precision != 'FP'
        ):
            if 'o_proj' in self.lora_handler.global_llm_target_modules:
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/o_lora_B/matmul',
                        'target_op_types': 'MatMul',
                        'output_precision': 'FP',
                    }
                )
            if 'down_proj' in self.lora_handler.global_llm_target_modules:
                config['precision_hints']['hints'].append(
                    {
                        'target_op_output_name': f'layers/{local_layer_idx}/self_attn/down_lora_B/matmul',
                        'target_op_types': 'MatMul',
                        'output_precision': 'FP',
                    }
                )

        logger.debug(f'[PTQPrecisionConfig] generated mtk_converter precision hint config={config}')
        return config

    def print_precision_summary(self):
        """Prints out all the precision settings of the pipeline."""
        if self.encoder_precision is not None:
            logger.info(f'Encoder precision:              {self.encoder_precision}')
        logger.info(f'LLM precision(s):               {self.llm_unique_precisions}')
        logger.info(f'Tail precision:                 {self.tail_precision}')
        logger.info(f'Default activation precision:   {self.get_precision_name(self.default_precision)[1]}')
        logger.info(f'Embedding input precision:      {self.get_precision_name(self.embeds_precision)[1]}')
        logger.info(f'Mask input precision:           {self.get_precision_name(self.mask_precision)[1]}')
        logger.info(f'Softmax input precision:        {self.get_precision_name(self.softmax_precision)[1]}')
        logger.info(f'Logits output precision:        {self.get_precision_name(self.logits_precision)[1]}')
        logger.info(f'Residual path precision:        {self.get_precision_name(self.respath_precision)[1]}')
        logger.info(f'Cache inputs/outputs precision: {self.get_precision_name(self.cache_precision)[1]}')
        if self.has_lora:
            logger.info(f'LoRA inputs precision:          {self.get_precision_name(self.lora_precision)[1]}')
        if len(self.bypassed_ops) > 0:
            logger.info(f'Quantization bypassed OPs:      {self.bypassed_ops}')
