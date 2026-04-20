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
"""Define Qwen2-VL format_text class."""

from ...utils import const, logger
from ..modeling_hook_base import BaseHook


class Qwen2VLFormatText(BaseHook):
    """Qwen2VLFormatText class that formats text prompts for InternVL2 models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the InternVL2FormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config, **kwargs):
        """Initialize the InternVL2FormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_placeholder = kwargs.pop('image_placeholder', const.QWEN2VL_IMAGE_TOKEN)
        self.num_image_tokens = None

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            torch.Tensor or numpy.ndarray: The formatted Qwen2VL prompt.
        """
        logger.debug('Enter Qwen2VLFormatText forward')
        logger.debug(f'Text before format: {text}')
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        merge_length = kwargs.get('merge_length')
        image_tokens = 0

        text = self.image_placeholder + text

        if not isinstance(text, list):
            prompt_text = [text]
            original_text = [text]
        if image_grid_thw is not None:
            assert merge_length is not None, logger.error(
                'merge_length should be set when image_grid_thw is not None', err=ValueError
            )
            index = 0
            for i in range(len(prompt_text)):
                while '<|image_pad|>' in prompt_text[i]:
                    prompt_text[i] = prompt_text[i].replace(
                        '<|image_pad|>', '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    original_text[i] = original_text[i].replace(
                        '<|image_pad|>', f'<|image_pad|>*{(image_grid_thw[index].prod() // merge_length)}', 1
                    )
                    image_tokens += image_grid_thw[index].prod() // merge_length
                    index += 1
                prompt_text[i] = prompt_text[i].replace('<|placeholder|>', '<|image_pad|>')
            logger.debug(f'Text after format: {original_text[0]}')
        elif video_grid_thw is not None:
            assert merge_length is not None
            index = 0
            for i in range(len(prompt_text)):
                while '<|video_pad|>' in prompt_text[i]:
                    prompt_text[i] = prompt_text[i].replace(
                        '<|video_pad|>', '<|placeholder|>' * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    original_text[i] = original_text[i].replace(
                        '<|image_pad|>', f'<|image_pad|>*{(image_grid_thw[index].prod() // merge_length)}, 1'
                    )
                    image_tokens += image_grid_thw[index].prod() // merge_length
                    index += 1
                prompt_text[i] = prompt_text[i].replace('<|placeholder|>', '<|video_pad|>')
            logger.debug(f'Text after format: {original_text[0]}')
        else:
            logger.info(f'prompt formatted: {prompt_text}')

        image_tokens += 2  # Add <|vision_start|> and <|vision_end|>
        self.num_image_tokens = image_tokens
        kwargs.update({'prompt_to_log': original_text[0]})
        return prompt_text[0], kwargs
