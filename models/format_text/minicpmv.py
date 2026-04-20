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
"""Define MinicpmV format_text class."""

import re

from ...utils import const, logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


def get_image_id_placeholder(token_processor, idx=0):
    """Generate a placeholder string for an image ID using a token processor.

    Args:
        token_processor : object
            An object that contains the start and end tokens for the image ID. It should have the attributes
            `im_id_start` and `im_id_end`.
        idx : int, optional
            The index to be included in the placeholder string. Default is 0.

    Returns:
            str: A placeholder string for the image ID.
    """
    return f'{token_processor.im_id_start}{idx}{token_processor.im_id_end}'


def get_grid_placeholder(token_processor, grid, image_feature_size):
    """Generate a placeholder string for a grid of image slices using a token processor.

    Args:
        token_processor : object
            An object that contains the start, end, and unknown tokens for the image slices. It should have
            the attributes `slice_start`, `slice_end`, and `unk`.
        grid : tuple or None
            The grid dimensions (rows, columns) for the image slices. If None, an empty string is returned.
        image_feature_size : int
            The size of the image features, determining the number of unknown tokens in each slice placeholder.

    Returns:
            str: A placeholder string for the grid of image slices.
    """
    if grid is None:
        return ''
    slice_image_placeholder = (
        token_processor.slice_start + token_processor.unk * image_feature_size + token_processor.slice_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for _i in range(rows):
        lines = []
        for _j in range(cols):
            lines.append(slice_image_placeholder)
        slices.append(''.join(lines))

    return '\n'.join(slices)


def get_slice_image_placeholder(token_processor, grid, image_idx=0, use_image_id=True, image_feature_size=64):
    """Generate a placeholder string for a sliced image using a token processor.

    Args:
        token_processor : object
            An object that contains the start, end, and unknown tokens for the image and slices. It should have
            the attributes `im_start`, `im_end`, `unk`, `slice_start`, and `slice_end`.
        grid : tuple or None
            The grid dimensions (rows, columns) for the image slices. If None, no grid placeholders are added.
        image_idx : int, optional
            The index to be included in the image ID placeholder. Default is 0.
        use_image_id : bool, optional
            If True, include an image ID placeholder. Default is True.
        image_feature_size : int, optional
            The size of the image features, determining the number of unknown tokens in each placeholder. Default is 64.

    Returns:
            str: A placeholder string for the sliced image, including the image ID and grid placeholders if specified.
    """
    image_placeholder = token_processor.im_start + token_processor.unk * image_feature_size + token_processor.im_end

    if use_image_id:
        final_placeholder = get_image_id_placeholder(token_processor, image_idx) + image_placeholder
    else:
        final_placeholder = image_placeholder

    if grid is not None:
        final_placeholder = final_placeholder + get_grid_placeholder(token_processor, grid, image_feature_size)
    return final_placeholder


class MinicpmVTokenProcessor:
    """A token processor for the MinicpmV model that handles special tokens for images and references.

    This class provides methods to convert special tokens to their corresponding IDs using a tokenizer.
    It defines various special tokens used in the MinicpmV model, such as image start and end tokens,
    reference tokens, and slice tokens.

    Args:
        tokenizer: object
            The tokenizer to be used for converting tokens to IDs.
    """

    def __init__(self):
        """Init special token for MinicpmV."""
        self.im_start = '<image>'
        self.im_end = '</image>'
        self.ref_start = '<ref>'
        self.ref_end = '</ref>'
        self.box_start = '<box>'
        self.box_end = '</box>'
        self.quad_start = '<quad>'
        self.quad_end = '</quad>'
        self.slice_start = '<slice>'
        self.slice_end = '</slice>'
        self.im_id_start = '<image_id>'
        self.im_id_end = '</image_id>'
        self.unk = '<unk>'


class MinicpmVFormatTextConfig(HookConfig):
    """MinicpmVFormatTextConfig class that extend HookConfig."""

    def __init__(self, **kwargs):
        """Initialize the HookConfig.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.query_num = self.kwargs.pop('query_num', None)
        if self.query_num is None:
            raise KeyError('query_num is required but missing from config.json')

        self.use_image_id = self.kwargs.pop('use_image_id', None)
        if self.use_image_id is None:
            raise KeyError('use_image_id is required but missing from config.json')


class MinicpmVFormatText(BaseHook):
    """MinicpmVFormatText class that formats text prompts for MinicpmV models.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the MinicpmVFormatText.
        forward(inputs): Forward pass that formats the text.
    """

    def __init__(self, config: MinicpmVFormatTextConfig, **kwargs):
        """Initialize the MinicpmVFormatText.

        Args:
            config (object): The configuration object.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(config)
        self.image_placeholder = kwargs.pop('image_placeholder', const.MINICPMV_IMAGE_TOKEN)
        self.token_processor = MinicpmVTokenProcessor()

    def forward(self, text, **kwargs):
        """Forward pass that returns the inputs.

        Args:
            text: The input text to be formatted.
            kwargs (dict, optional): Additional keyword arguments.

        Returns:
            str: The formatted MinicpmV prompt.
        """
        logger.debug('Enter MinicpmVFormatText forward')
        logger.debug(f'Text before format: {text}')
        grid = kwargs.get('grid')

        text = self.image_placeholder + '\n' + text

        image_tags = re.findall(self.image_placeholder, text)
        text_chunks = text.split(self.image_placeholder)
        final_text = ''
        for i in range(len(image_tags)):
            final_text = (
                final_text
                + text_chunks[i]
                + get_slice_image_placeholder(
                    self.token_processor, grid, i, self.config.use_image_id, self.config.query_num
                )
            )
        final_text += text_chunks[-1]
        logger.info(f'prompt formatted: {final_text}')

        return final_text, kwargs
