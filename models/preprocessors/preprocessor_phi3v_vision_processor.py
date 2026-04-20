# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.  # noqa: CPY001
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
"""Phi3-V vision embedding preprocessor class."""

import re
from typing import List, Optional, Union

import torch
import torchvision
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import (
    convert_to_rgb,
)

from ...utils import logger
from ...utils.const import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from ...utils.image_utils import (
    HD_transform,
    TensorType,
    calc_hd_transform_size,
    make_list_of_images,
    pad_to_max_num_crops_tensor,
    valid_images,
)


def convert_images_texts_to_inputs(
    images,
    tokenizer,
    texts,
    preformatter=None,
    image_token='<|image_1|>',
    image_fname='',
    padding=False,
    truncation=None,
    max_length=None,
    return_tensors=None,
):
    """Converts images and texts into model inputs suitable for processing.

    Args:
        images (dict): A dictionary containing image data and metadata.
        tokenizer (Tokenizer): The tokenizer to use for processing texts.
        texts (str): The input text containing placeholders for images.
        preformatter (Preformatter, optional): An optional preformatter for the texts. Defaults to None.
        image_token (str, optional): The token representing an image placeholder in the text. Defaults to '<|image_1|>'.
        image_fname (str, optional): The filename of the image. Defaults to ''.
        padding (bool, optional): Whether to pad the sequences. Defaults to False.
        truncation (bool, optional): Whether to truncate the sequences. Defaults to None.
        max_length (int, optional): The maximum length of the sequences. Defaults to None.
        return_tensors (str, optional): The type of tensors to return. Defaults to None.

    Returns:
        BatchFeature: A batch feature containing the processed inputs.

    Raises:
        ValueError: If images are not preprocessed by Phi3VImageProcessor.
    """
    if preformatter is not None:
        texts = preformatter.generate_prompt(texts, None)
    if not len(images):
        model_inputs = tokenizer(
            texts, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )
        return BatchFeature(data={**model_inputs})

    prompt_to_print = texts.replace(image_token, image_fname)
    if preformatter is not None:
        logger.info(f'Prompt (with {preformatter.name} preformatter):\n{prompt_to_print}')
    else:
        logger.info(f'Prompt:\n{prompt_to_print}')

    pattern = r'<\|image_\d+\|>'
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in re.split(pattern, texts)]

    if 'num_img_tokens' in images:
        num_img_tokens = images['num_img_tokens']
    else:
        # assert 'num_crops' in images, 'num_crops must be provided in images if num_img_tokens is not provided'
        # num_crops = images['num_crops']
        # num_img_tokens = [_num_crops * self.num_img_tokens for _num_crops in num_crops]
        logger.error('Images must be preprocessed first by Phi3VImageProcessor.', err=ValueError)

    images, image_sizes = images['pixel_values'], images['image_sizes']

    # image_tags needs to start from 1 to n
    image_tags = re.findall(pattern, texts)
    # image_ids = [int(s.split("|")[1].split("_")[-1]) * -1 for s in image_tags]
    # image_ids_pad = [[iid]*num_img_tokens[i] for i, iid in enumerate(image_ids)]
    image_ids = [int(s.split('|')[1].split('_')[-1]) for s in image_tags]
    unique_image_ids = sorted(set(image_ids))
    # image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be [1, 4, 5]
    # check the condition
    assert unique_image_ids == list(range(1, len(unique_image_ids) + 1)), (
        f'image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}'
    )
    # total images must be the same as the number of image tags
    assert len(unique_image_ids) == len(images), (
        f'total images must be the same as the number of image tags, got {len(unique_image_ids)}'
    )
    f' image tags and {len(images)} images'

    image_ids_pad = [[-iid] * num_img_tokens[iid - 1] for iid in image_ids]

    def insert_separator(x, sep_list):
        if len(x) > len(sep_list):
            sep_list.append([])
        return [ele for sublist in zip(x, sep_list) for ele in sublist]

    input_ids = []
    offset = 0
    for x in insert_separator(prompt_chunks, image_ids_pad):
        input_ids.extend(x[offset:])

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = (input_ids > -1000000).to(torch.long)

    return BatchFeature(
        data={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': images,
            'image_sizes': image_sizes,
        }
    )


class Phi3VImageProcessor(BaseImageProcessor):
    r"""Constructs a Phi3 image processor.

    Based on [`CLIPImageProcessor`] with incorporation of additional techniques
    for processing high resolution images as explained in the [InternLM-XComposer2-4KHD](https://arxiv.org/abs/2401.16420).

    Args:
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ['pixel_values']

    def __init__(
        self,
        num_crops: int = 1,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the Phi3VImageProcessor with the given parameters.

        Args:
            num_crops (int, optional): Number of crops to generate. Defaults to 1.
            image_mean (Optional[Union[float, List[float]]], optional): Mean to use for normalization. Defaults to None.
            image_std (Optional[Union[float, List[float]]], optional): Standard deviation to use for normalization.
                Defaults to None.
            do_convert_rgb (bool, optional): Whether to convert the image to RGB. Defaults to True.
            kwargs: Other kwargs.
        """
        super().__init__(**kwargs)
        self.num_crops = num_crops
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.use_hd_transform = kwargs.pop('use_hd_transform', None)

    def calc_num_image_tokens(self, images):
        """Calculate the number of image tokens for each image.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        """
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, '
                'torch.Tensor, tf.Tensor or jax.ndarray.'
            )

        images = [image.convert('RGB') for image in images]
        # (H, W, C)
        elems = [HD_transform(im, hd_num=self.num_crops) for im in images]
        shapes = [[im.size[1], im.size[0]] for im in elems]
        return [int((h // 336 * w // 336 + 1) * 144 + 1 + (h // 336 + 1) * 12) for h, w in shapes]

    def calc_num_image_tokens_from_image_size(self, width, height):
        """Calculate the number of image tokens for a given image size.

        Args:
            width (`int`): Width of the image.
            height (`int`): Height of the image.
        """
        new_width, new_height = calc_hd_transform_size(width, height, hd_num=self.num_crops)
        return int((new_height // 336 * new_width // 336 + 1) * 144 + 1 + (new_height // 336 + 1) * 12)

    def preprocess(
        self,
        images,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """Preprocess the images and return a batch feature.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            kwargs: Other kwargs.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, '
                'torch.Tensor, tf.Tensor or jax.ndarray.'
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        image_sizes = []
        img_processor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(image_mean, image_std)]
        )

        # PIL images
        # HD_transform pad images to size of multiiply of 336, 336
        # convert to RGB first
        images = [image.convert('RGB') for image in images]
        elems = [HD_transform(im, hd_num=self.num_crops) for im in images]
        # tensor transform and normalize
        hd_images = [img_processor(im) for im in elems]
        # create global image
        global_image = [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(),
                size=(336, 336),
                mode='bicubic',
            ).to(im.dtype)
            for im in hd_images
        ]

        # [(3, h, w)], where h, w is multiple of 336
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        num_img_tokens = [int((h // 336 * w // 336 + 1) * 144 + 1 + (h // 336 + 1) * 12) for h, w in shapes]
        # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
        # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
        hd_images_reshape = [
            im.reshape(1, 3, h // 336, 336, w // 336, 336)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, 336, 336)
            .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]
        # concat global image and local image
        hd_images_reshape = [
            torch.cat([_global_image, _im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)
        ]

        # pad to max_num_crops
        image_transformed = [pad_to_max_num_crops_tensor(im, self.num_crops + 1) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        image_sizes = [torch.LongTensor(_shapes) for _shapes in shapes]
        padded_images = image_transformed
        image_sizes = shapes

        data = {
            'input_features': padded_images.squeeze(0).permute(0, 2, 3, 1),
            'image_sizes': image_sizes,
            'num_img_tokens': num_img_tokens,
        }
        kwargs.update({'phi3v_image_batch_features': data, 'phi3v_hd_transform': self.use_hd_transform})

        return BatchFeature(data=data, tensor_type=return_tensors), kwargs
