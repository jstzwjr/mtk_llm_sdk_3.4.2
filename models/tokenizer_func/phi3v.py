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
"""Define InternVL2 tokenizer_func class."""

import os
import re

import numpy as np
import torch

from ...utils import const, logger, utils
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class Phi3VFormatTextConfig(HookConfig):
    """Phi3VFormatTextConfig class that extend HookConfig."""

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.image_token = self.kwargs.pop('image_token', const.PHI3V_DEFAULT_IMAGE_TOKEN)


class Phi3VTokenizerFunc(BaseHook):
    """Phi3VTokenizerFunc class that formats text prompts for Phi3V models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Phi3VFormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config: Phi3VFormatTextConfig, **kwargs):
        """Initialize the Phi3VFormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_token = self.config.image_token  # '<|image_1|>'

    def convert_images_texts_to_inputs(
        self,
        images_batch_features,
        tokenizer,
        prompt,
        pipeline,
        preformatter=None,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors='np',
        **kwargs,
    ):
        """Converts images and texts into model inputs suitable for processing.

        Args:
            images_batch_features (dict): A dictionary containing image data and metadata.
            tokenizer (Tokenizer): The tokenizer to use for processing texts.
            prompt (str): The input text containing placeholders for images.
            pipeline (Pipeline): The pipeline object.
            preformatter (Preformatter, optional): An optional preformatter for the texts. Defaults to None.
            image_token (str, optional): The token representing an image placeholder in the text.
                Defaults to '<|image_1|>'.
            image_fname (str, optional): The filename of the image. Defaults to ''.
            padding (bool, optional): Whether to pad the sequences. Defaults to False.
            truncation (bool, optional): Whether to truncate the sequences. Defaults to None.
            max_length (int, optional): The maximum length of the sequences. Defaults to None.
            return_tensors (str, optional): The type of tensors to return. Defaults to None.
            kwargs: Other kwargs.

        Returns:
            BatchFeature: A batch feature containing the processed inputs.

        Raises:
            ValueError: If images are not preprocessed by Phi3VImageProcessor.
        """
        label = kwargs.pop('label', None)
        mm_path = kwargs.pop('mm_path', [])
        sub_response = kwargs.pop('sub_response', None)
        quiet = kwargs.get('quiet', False)
        simplified_prompt = kwargs.pop('prompt_to_log', None)

        logger.debug(f'Forward tokenizer, label={label}, mm_path={mm_path}, sub_response={sub_response}, quiet={quiet}')

        prompt_formatted = preformatter.generate_prompt(prompt, input_=sub_response, label=label)
        if pipeline.has_encoder():
            if len(mm_path) == 0:
                logger.error('Expected multimodal input for multimodal model but did not get any.', err=ValueError)
            prompt_to_print = prompt.replace(self.image_token, os.path.basename(mm_path))
            if simplified_prompt is not None:
                prompt_to_print = simplified_prompt
            else:
                prompt_to_print = prompt.replace(self.image_token, os.path.basename(mm_path))
        else:
            prompt_to_print = prompt_formatted
        if not quiet:
            logger.info(f'Input text (with {preformatter.name} preformatter):\n{prompt_to_print}')

        if not len(images_batch_features):
            return tokenizer(
                prompt_formatted,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )['input_ids'].astype(np.int32)

        pattern = r'<\|image_\d+\|>'
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in re.split(pattern, prompt_formatted)]

        if 'num_img_tokens' in images_batch_features:
            num_img_tokens = images_batch_features['num_img_tokens']
        else:
            logger.error('Images must be preprocessed first by Phi3VImageProcessor.', ValueError)

        images = [images_batch_features['input_features']]

        # image_tags needs to start from 1 to n
        image_tags = re.findall(pattern, prompt_formatted)
        image_ids = [int(s.split('|')[1].split('_')[-1]) for s in image_tags]
        unique_image_ids = sorted(set(image_ids))
        # image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be [1, 4, 5]
        # check the condition
        if unique_image_ids != list(range(1, len(unique_image_ids) + 1)):
            logger.error(
                f'image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], '
                f'cannot be {unique_image_ids}',
                err=ValueError,
            )
        # total images must be the same as the number of image tags
        if len(unique_image_ids) != len(images):
            logger.error(
                f'total images must be the same as the number of image tags, got {len(unique_image_ids)}'
                f' image tags and {len(images)} images',
                err=ValueError,
            )

        image_ids_pad = [[-iid] * num_img_tokens[iid - 1] for iid in image_ids]

        def insert_separator(x, sep_list):
            if len(x) > len(sep_list):
                sep_list.append([])
            return [ele for sublist in zip(x, sep_list) for ele in sublist]

        input_ids = []
        offset = 0
        for x in insert_separator(prompt_chunks, image_ids_pad):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cpu().numpy()

    def forward(self, pipeline, prompt, preformatter, **kwargs):
        """Forward pass through the tokenizer.

        Args:
            pipeline (Pipeline): Pipeline object.
            prompt (str): input prompt or input token.
            preformatter (Preformatter): preformatter.
            kwargs (dict, optional): Additional keyword arguments.
        """
        image_batch_features = kwargs.get('phi3v_image_batch_features')
        if image_batch_features is None:
            logger.error('image_batch_features must be passed when forwarding Phi3-V format text.')

        input_ids = self.convert_images_texts_to_inputs(
            images_batch_features=image_batch_features,
            tokenizer=pipeline.tokenizer,
            pipeline=pipeline,
            preformatter=preformatter,
            prompt=prompt,
            **kwargs,
        )

        # Force add bos_token_id in front if add_bos is true
        return utils.enforce_add_bos_mode(
            pipeline.get_tokenizer_add_bos(), input_ids, pipeline.config.l.bos_token_id
        ), kwargs
