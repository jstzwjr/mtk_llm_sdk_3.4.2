# Copyright (C) 2024 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Define default tokenizer forward class."""

import os

import numpy as np
from sentencepiece import SentencePieceProcessor

from ...utils import logger, utils
from ..modeling_hook_base import BaseHook


class DefaultTokenizerFunc(BaseHook):
    """DefaultTokenizerFunc Hook class."""

    def __init__(self, config: BaseHook, **kwargs):
        """Initialize the DefaultTokenizerFunc class."""
        super().__init__(config)

    def forward(self, pipeline, prompt, preformatter, **kwargs):
        """Forward pass through the tokenizer.

        Args:
            pipeline (Pipeline): Pipeline object.
            prompt (str): input prompt or input token.
            preformatter (Preformatter): preformatter.
            kwargs (dict, optional): Additional keyword arguments.
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
            if hasattr(pipeline.config.e, 'image_text'):
                mm_text = pipeline.config.e.image_text
                mm_token = pipeline.config.e.image_token
            else:
                mm_text = pipeline.config.e.audio_text
                mm_token = pipeline.config.e.audio_token
            for i in range(len(mm_path)):
                prompt = prompt.replace(mm_text, os.path.basename(mm_path[i]), 1)
            prompt_to_print = prompt
            if simplified_prompt is not None:
                prompt_to_print = simplified_prompt
            else:
                for i in range(len(mm_path)):
                    prompt_to_print = prompt.replace(mm_text, os.path.basename(mm_path[i]), 1)
                prompt_to_print = prompt
        else:
            prompt_to_print = prompt_formatted
        if not quiet:
            logger.info(f'Input text (with {preformatter.name} preformatter):\n{prompt_to_print}')

        if pipeline.has_encoder():
            # Tokenize text except placeholder multimodal tokens
            if isinstance(pipeline.tokenizer, SentencePieceProcessor):
                prompt_chunks = [
                    np.array([pipeline.tokenizer.encode(chunk)], dtype=np.int32)
                    for chunk in prompt_formatted.split(mm_text)
                ]
            else:
                prompt_chunks = [
                    pipeline.tokenizer(chunk, return_tensors='np')['input_ids'].astype(np.int32)
                    for chunk in prompt_formatted.split(mm_text)
                ]

            # Remove BOS token for 2nd chunk and later if image_token exists
            prompt_chunks = [
                x if i == 0 else np.delete(x, np.where(x == pipeline.config.l.bos_token_id))[None, :]
                for i, x in enumerate(prompt_chunks)
            ]
        else:
            if isinstance(pipeline.tokenizer, SentencePieceProcessor):
                prompt_chunks = [np.array([pipeline.tokenizer.encode(prompt_formatted)], dtype=np.int32)]
            else:
                prompt_chunks = [
                    pipeline.tokenizer(prompt_formatted, return_tensors='np')['input_ids'].astype(np.int32)
                ]

        if len(prompt_chunks) == 1:
            prompt_tokens = prompt_chunks[0]
        else:
            # Replace image_token with image_token_id, stitch back whole prompt
            token_list = []
            for i in range(len(prompt_chunks) - 1):
                token_list.append(prompt_chunks[i][0])
                token_list.append([mm_token])
            token_list.append(prompt_chunks[-1][0])
            prompt_tokens = np.concatenate(token_list).astype(np.int32)[None, :]

        internvl2_image_token_ = kwargs.pop('internvl2_image_token', None)
        internvl2_image_token_ids_ = (
            pipeline.tokenizer.convert_tokens_to_ids(internvl2_image_token_)
            if internvl2_image_token_ is not None
            else None
        )

        kwargs.update({'internvl2_image_token_id': [internvl2_image_token_ids_]})

        # Force add bos_token_id in front if add_bos is true
        return utils.enforce_add_bos_mode(
            pipeline.get_tokenizer_add_bos(), prompt_tokens, pipeline.config.l.bos_token_id
        ), kwargs
