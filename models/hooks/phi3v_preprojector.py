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
"""Define the tensor manupulation method between a phi3-v vision encoder and vision projector."""

import os

import torch

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class Phi3vPreprojectorConfig(HookConfig):
    """Configuration class for Phi3vPreProjector.

    This class extends the HookConfig to include additional configurations specific to Phi3vPreProjector.

    Attributes:
        select_layer (int): The selected layer.
        downsample_ratio (float): The downsample ratio.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the Phi3vPreprojectorConfig.

        Args:
            kwargs (dict, optional): Additional keyword arguments.
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)

        self.use_hd_transform = self.kwargs.pop('use_hd_transform', None)
        if self.use_hd_transform is None:
            logger.error(
                'use_hd_transform is required in Phi3VPrePRojectorConfig but missing in config.json.', err=ValueError
            )
        self.with_learnable_separator = self.kwargs.pop('with_learnable_separator', False)
        self.hd_transform_order = self.kwargs.pop('hd_transform_order', 'glb_sub')
        self.image_dim_out = self.kwargs.pop('image_dim_out', 1024)
        self.type_feature = self.kwargs.pop('type_feature', 'patch')
        self.sub_glb_GN_path = self.kwargs.pop('sub_glb_GN_path', None)
        if self.sub_glb_GN_path is None:
            logger.error('sub_glb_GN_path is required but missing in config.json.')

        # For PTQ
        self.img_sizes = self.kwargs.pop('fixed_img_size', None)
        if self.img_sizes is None:
            logger.error('img_sizes is required but missing in config.json.')

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'use_hd_transform:     {self.use_hd_transform}')
        logger.info(f'hd_transform_order:   {self.hd_transform_order}')
        logger.info(f'image_dim_out:        {self.image_dim_out}')
        logger.info(f'type_feature:         {self.type_feature}')
        logger.info(f'sub_glb_GN_path:      {self.sub_glb_GN_path}')
        logger.info(f'img_sizes"            {self.img_sizes}')


class Phi3vPreProjector(BaseHook):
    """Phi3vPreProjector class that performs tensor manipulation at vision encoder output.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the PatchSelect.
        forward(inputs): Forward pass.
    """

    def __init__(self, config: Phi3vPreprojectorConfig):
        """Initialize the PatchSelect.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)
        if self.config.with_learnable_separator:
            if not self.config.use_hd_transform:
                logger.error('learnable separator is only for hd transform', err=ValueError)
            load_path_glb = os.path.join(self.config.sub_glb_GN_path, 'glb_GN.pt')
            load_path_sub = os.path.join(self.config.sub_glb_GN_path, 'sub_GN.pt')
            self.glb_GN = torch.load(load_path_glb)  # shape: [1, 1, self.config.image_dim_out * 4]
            self.sub_GN = torch.load(load_path_sub)  # shape: [1, 1, 1, self.config.image_dim_out * 4]
            logger.debug(f'Load glb_GN from {load_path_glb} with shape: {self.glb_GN.shape}')
            logger.debug(f'Load glb_GN from {load_path_sub} with shape: {self.sub_GN.shape}')
        else:
            self.glb_GN = None
            self.sub_GN = None

    def forward(self, image_features, **kwargs):
        """Forward pass.

        Args:
            image_features: The image features obtained from the output of a vision encoder.
            kwargs (dict, optional): Additional keyword arguments.
        """
        image_dim_out = self.config.image_dim_out

        sub_GN = self.sub_GN.to(image_features.device)  # noqa: N806
        glb_GN = self.glb_GN.to(image_features.device)  # noqa: N806

        phi3v_image_batch_features = kwargs.get('phi3v_image_batch_features')
        if phi3v_image_batch_features is None:
            logger.warning('Must pass phi3v_image_batch_features when forwarding Phi3vPreProjector.')
            img_sizes = None
            num_img_tokens = None
        else:
            img_sizes = phi3v_image_batch_features.get('image_sizes', None)
            num_img_tokens = phi3v_image_batch_features.get('num_img_tokens', None)
        if img_sizes is None:
            logger.warning(
                'phi3v_image_batch_features does not contain img_sizes when forwarding Phi3vPreProjector. '
                'Use attribute `self.config.img_sizes` instead.'
            )
            img_sizes = self.config.img_sizes

        # Select type_feature
        if self.config.type_feature == 'patch':
            image_features = image_features[:, 1:]

        kwargs.update({'phi3v_hd_transform': self.config.use_hd_transform, 'phi3v_num_img_tokens': num_img_tokens})

        # Only supports single images here
        logger.debug(f'image_features: {image_features.shape}')
        bs = 1
        if self.config.use_hd_transform:
            base_feat_height = base_feat_width = int(image_features.shape[1] ** 0.5)

            assert base_feat_height == 24 and base_feat_width == 24, (
                f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width},'
            )
            ' expect 24x24 features for hd transform'

            # bs x max_num_crops x (24x24) x C
            image_features = image_features.view(bs, -1, base_feat_height * base_feat_width, image_dim_out)
            c = image_dim_out
            # H = 24
            H = base_feat_height  # noqa: N806

            output_imgs = []
            output_len = []
            # training is tensor, inference is list
            if isinstance(img_sizes, torch.Tensor):
                img_sizes = img_sizes.view(-1, 2)
            for _bs in range(bs):
                h, w = img_sizes[_bs]
                h = h // 336
                w = w // 336
                b_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = image_features[_bs, :1]
                logger.debug(f'global_img_feature: {global_img_feature.shape}')

                # 1 x 12 x 12 x 4096
                glb_img_golden = (
                    global_img_feature.reshape(1, H, H, c)
                    .reshape(1, H // 2, 2, H // 2, 2, c)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, H // 2, H // 2, 4 * c)
                    .contiguous()
                )

                glb_img = (
                    global_img_feature.reshape(1, H, H, c)
                    .reshape(H, H, c)
                    .reshape(H, H // 2, 2, c)
                    .contiguous()
                    .reshape(H, H // 2, 2 * c)
                    .reshape(H // 2, 2, H // 2, 2 * c)
                    .permute(0, 2, 1, 3)
                    .reshape(H // 2, H // 2, 4 * c)
                    .contiguous()
                    .reshape(1, H // 2, H // 2, 4 * c)
                    .contiguous()
                )
                assert torch.all(glb_img == glb_img_golden)

                temp_glb_GN = sub_GN.repeat(1, H // 2, 1, 1)  # noqa: N806

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, 4 * c)

                if img_sizes[_bs][0] != 336 or img_sizes[_bs][1] != 336:  # batch size != 1
                    # (max_num_crops-1) x (12x12) x C
                    sub_img = image_features[_bs, 1:]
                    logger.debug(f'sub_img before operation: {sub_img.shape}')
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:b_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img_backup = sub_img
                    sub_img_golden = (
                        sub_img.reshape(b_, H, H, c)
                        .reshape(b_, H // 2, 2, H // 2, 2, c)
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(b_, -1, 4 * c)
                        .contiguous()
                    )

                    sub_img_4D = []  # noqa: N806
                    for i in range(b_):
                        single_sub_img = sub_img_backup[i]
                        single_sub_img = (
                            single_sub_img.reshape(H, H // 2, 2, c)
                            .reshape(H, H // 2, 2 * c)
                            .reshape(H // 2, 2, H // 2, 2 * c)
                            .permute(0, 2, 1, 3)
                            .reshape(H // 2, H // 2, 4 * c)
                            .reshape(1, H // 2, H // 2, 4 * c)
                            .contiguous()
                        )
                        sub_img_4D.append(single_sub_img)
                    sub_img = torch.cat(sub_img_4D, dim=0).reshape(b_, -1, 4 * c).contiguous()
                    assert torch.all(sub_img == sub_img_golden)

                    sub_img_golden = (
                        sub_img_golden.reshape(1, h, w, 12, 12, -1)
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(1, h * 12, w * 12, 4 * c)
                    )

                    sub_img = (
                        sub_img.reshape(b_, 12, 12, 4 * c)
                        .reshape(b_, 12, 12 * 4 * c)
                        .reshape(h, w, 12, 12 * 4 * c)
                        .permute(0, 2, 1, 3)
                        .reshape(h * 12, w, 12 * 4 * c)
                        .reshape(h * 12, w, 12, 4 * c)
                        .reshape(h * 12, w * 12, 4 * c)
                        .reshape(1, h * 12, w * 12, 4 * c)
                        .contiguous()
                    )
                    assert torch.all(sub_img == sub_img_golden)
                    temp_sub_GN = sub_GN.repeat(1, h * 12, 1, 1)  # noqa: N806
                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, 4 * c)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.config.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, glb_GN, sub_img], dim=1))
                    elif self.config.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, glb_GN, glb_img], dim=1))
                    else:
                        logger.error(
                            f'hd_transform_order = {self.config.hd_transform_order}, not implemented',
                            err=NotImplementedError,
                        )
                    temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                else:
                    output_imgs.append(glb_img)
                    temp_len = 156
                assert temp_len == output_imgs[-1].shape[1], (
                    f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                )
                output_len.append(temp_len)
                return output_imgs[0], kwargs
        else:
            return image_features, kwargs
        return None
