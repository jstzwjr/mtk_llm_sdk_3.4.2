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
"""Define the patch select method between a vision encoder and vision projector."""

from ...utils import logger
from ...utils.image_utils import expand2square, process_anyres_image_no_processor
from ..modeling_hook_base import BaseHook


class ImageAspectRatioProcess(BaseHook):
    """ImageAspectRatioProcess class does the necessary image aspect ratio transform before image preprocessor.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the ImageAspectRatioProcess.
        forward(inputs): Forward pass.
    """

    def __init__(self, config):
        """Initialize the ImageAspectRatioProcess.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def forward(self, image, **kwargs):
        """Forward pass.

        Args:
            image: The image input for image preprocessor.
            kwargs (dict, optional): Additional keyword arguments.
        """
        processor = kwargs.pop('processor', None)
        processor_config = kwargs.pop('processor_config', None)
        image_aspect_ratio = kwargs.pop('image_aspect_ratio', '')
        if processor is None:
            logger.error(
                'processor must be passed when using ImageAspectRatioProcess pre-preprocessor hook.', err=ValueError
            )
        if processor_config is None:
            logger.error(
                'processor_config must be passed when using ImageAspectRatioProcess pre-preprocessor hook.',
                err=ValueError,
            )
        if image_aspect_ratio == 'pad':
            if getattr(processor, 'image_mean', None) is None:
                logger.error(
                    "image mean must be set in processor when using image_aspect_ratio as 'pad'.", err=ValueError
                )
            inputs = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        elif image_aspect_ratio == 'anyres' or 'anyres_max' in image_aspect_ratio:
            # FIXME: (Andy) Need to do
            # image_patches = [
            #    processor.preprocess(image_patch, return_tensors='pt', **processor_cfg)['pixel_values'][0]
            #    for image_patch in image_patches
            # ]
            # In processor.preprocess
            if getattr(processor_config, 'image_grid_pinpoints', None) is None:
                logger.error(
                    "image_grid_pinpoints must be set in processor_config when using image_aspect_ratio as 'anyres'.",
                    err=ValueError,
                )
            inputs = process_anyres_image_no_processor(
                image, processor, processor_config.image_grid_pinpoints, processor_config
            )
        else:
            inputs = image
        return inputs, kwargs
