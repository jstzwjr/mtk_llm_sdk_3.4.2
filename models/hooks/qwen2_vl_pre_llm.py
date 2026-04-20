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
"""Define Qwen2-VL pre-llm hook to precompute multimodal position ids and multimodal position delta."""

import torch

from ...utils import logger
from ..configuration_hook import HookConfig
from ..modeling_hook_base import BaseHook


class Qwen2VLPreLLMConfig(HookConfig):
    """Configuration class for Qwen2VLPreEncoder.

    This class extends the HookConfig to include additional configurations specific to Qwen2VLPreEncoder.

    Attributes:
        spatial_merge_size (int): The spatial merge size.
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of heads.

    Methods:
        print_config: Print the hook configuration.
    """

    def __init__(self, **kwargs):
        """Initialize the Qwen2VLPreEncoderConfig.

        Args:
            kwargs (dict, optional): Additional keyword arguments.

        Raises:
            ValueError: If spatial_merge_size, embed_dim, or num_heads are not provided.
        """
        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = False
        super().__init__(**kwargs)
        self.spatial_merge_size = self.kwargs.pop('spatial_merge_size', None)
        if self.spatial_merge_size is None:
            logger.error('spatial_merge_size must be set when using Qwen2VLPreLLM.', err=ValueError)

        self.image_token_id = self.kwargs.pop('image_token_id', None)
        if self.image_token_id is None:
            logger.error('image_token_id must be set when using Qwen2VLPreLLM.', err=ValueError)

        self.video_token_id = self.kwargs.pop('video_token_id', None)
        if self.video_token_id is None:
            logger.error('video_token_id must be set when using Qwen2VLPreLLM.', err=ValueError)

        self.vision_start_token_id = self.kwargs.pop('vision_start_token_id', None)
        if self.vision_start_token_id is None:
            logger.error('vision_start_token_id must be set when using Qwen2VLPreLLM.', err=ValueError)

        if verbose and self.name != 'passthrough':
            self.print_config()
            self.print_unused_kwargs()

    def print_config(self):
        """Print the hook configuration."""
        logger.info(f'Hook type: {self.type}, hook name: {self.name}')
        if self.custom_path is not None:
            logger.info(f'Hook path: {self.custom_path}')
        logger.info(f'spatial_merge_size:        {self.spatial_merge_size}')
        logger.info(f'image_token_id:            {self.image_token_id}')
        logger.info(f'video_token_id:            {self.video_token_id}')
        logger.info(f'vision_start_token_id:     {self.vision_start_token_id}')


class Qwen2VLPreLLM(BaseHook):
    """Qwen2VLPreLLM class that precomputes necessary input for Qwen2-VL LLM.

    Attributes:
        config (object): The configuration object.

    Methods:
        __init__(config): Initialize the Qwen2VLPreEncoder.
        forward(inputs): Forward pass.
    """

    def __init__(self, config: Qwen2VLPreLLMConfig):
        """Initialize the NumpyToTorch.

        Args:
            config (object): The configuration object.
        """
        super().__init__(config=config)

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        overture_size=0,
        dummy_token_id=None,
    ):
        """Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.

            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.

            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

            If overture_size is given, all position_ids must be shifted accordingly.
        """  # noqa: D214
        spatial_merge_size = self.config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        # Pad dummy ids to represent overture tokens
        if overture_size > 0:
            input_ids = torch.nn.functional.pad(input_ids, (overture_size, 0), value=dummy_token_id)

        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        self.position_ids = position_ids
        self.mrope_position_deltas = mrope_position_deltas

        # self.model.set_decoder_layer_position_ids(position_ids)
        return position_ids, mrope_position_deltas

    def forward(self, input_embeds, **kwargs):
        """Forward pass.

        Args:
            input_embeds: The input_embeds to be passed to LLM. It will just directly be passed through.
            kwargs (dict, optional): Additional keyword arguments.
        """
        image_grid_thw = kwargs.pop('image_grid_thw', None)
        if image_grid_thw is None:
            logger.error('image_grid_thw must be passed when forwarding Qwen2VLPreLLM.', err=ValueError)

        text_tokens = kwargs.pop('text_tokens', None)
        if text_tokens is None:
            logger.error('text_tokens must be passed when forwarding Qwen2VLPreLLM.', err=ValueError)

        overture_size = kwargs.pop('overture_size', 0)
        logger.debug(
            f'overture_size={overture_size} when forwarding Qwen2VLPreLLM. '
            'Corresponding outputs are nudged according to it.'
        )

        dummy_token_id = kwargs.pop('dummy_token_id', None)
        if overture_size > 0 and dummy_token_id is None:
            logger.error('dummy_token_id must be passed when forwarding Qwen2VLPreLLM with overture.', err=ValueError)

        qwen2_vl_position_ids, qwen2_vl_mrope_delta = self.get_rope_index(
            torch.tensor(text_tokens) if not isinstance(text_tokens, torch.Tensor) else text_tokens,
            image_grid_thw=image_grid_thw,
            overture_size=overture_size,
            dummy_token_id=dummy_token_id,
        )

        kwargs.update({'qwen2_vl_position_ids': qwen2_vl_position_ids, 'qwen2_vl_mrope_delta': qwen2_vl_mrope_delta})

        return input_embeds, kwargs
