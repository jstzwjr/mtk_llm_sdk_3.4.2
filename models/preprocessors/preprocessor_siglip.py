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
"""SigLIP preprocessor class."""

from typing import Dict, Optional, Union

import numpy as np
from transformers.image_transforms import get_resize_output_image_size, resize

from ...utils import image_utils, logger
from .preprocessor_clip import CLIPImageProcessor


class SigLIPImageProcessor(CLIPImageProcessor):
    """SigLIPImageProcessor class.

    This class extends the CLIPImageProcessor class and provides a method for resizing images.

    Attributes:
        None

    Methods:
        resize: Resizes an image.
    """

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample=image_utils.PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, image_utils.ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize an image.

        The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample: (`image_utils.PILImageResampling`, *optional*,
                defaults to `image_utils.PILImageResampling.BICUBIC`)
                Resampling filter to use when resiizing the image.
            data_format (`str` or `image_utils.ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            **kwargs: Other kwargs.
        """
        if 'shortest_edge' not in size:
            logger.error(
                f'The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}', err=ValueError
            )
        output_size = get_resize_output_image_size(
            image, size=size['shortest_edge'], default_to_square=True
        )  # Accepts square only
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)
