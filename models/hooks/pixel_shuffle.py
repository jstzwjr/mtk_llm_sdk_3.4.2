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
"""Define the pixel shuffle method between a vision encoder and vision projector."""

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class InternVL2PixelShuffleConfig(HookConfig):
    """Configuration class for InternVL2PixelShuffle.

    This class extends the HookConfig to include additional configurations specific to InternVL2PixelShuffle.

    Attributes:
        select_layer (int): The selected layer.
        downsample_ratio (float): The downsample ratio.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the InternVL2PixelShuffleConfig.

        Args:
            kwargs (dict, optional): Additional keyword arguments.
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)
        self.select_layer = self.kwargs.pop('select_layer', -1)
        self.downsample_ratio = self.kwargs.pop('downsample_ratio', 0.5)

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'select_layer:     {self.select_layer}')
        logger.info(f'downsample_ratio: {self.downsample_ratio}')


class InternVL2PixelShuffle(BaseHook):
    """InternVL2PixelShuffle class that performs pixel shuffle output in InternVL2 series.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the PatchSelect.
        forward(inputs): Forward pass.
    """

    def __init__(self, config: InternVL2PixelShuffleConfig):
        """Initialize the PatchSelect.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def pixel_shuffle(self, x, scale_factor=0.5):
        """Perform pixel shuffle operation.

        Args:
            x (torch.Tensor): The input tensor.
            scale_factor (float, optional): The scale factor for pixel shuffle. Defaults to 0.5.

        Returns:
            torch.Tensor: The shuffled tensor.
        """
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, image_features, **kwargs):
        """Forward pass.

        Args:
            image_features: The image features obtained from the output of a vision encoder.
            kwargs (dict, optional): Additional keyword arguments.
        """
        # if self.config.select_layer == -1:
        # last_hidden_state, encoder_state = self.vision_model(images)
        # vit_embeds, _ = image_features
        # else:
        # _, vit_embeds = image_features
        # vit_embeds = vit_embeds[self.config.select_layer]
        vit_embeds = image_features
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.config.downsample_ratio)
        return vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]), kwargs
