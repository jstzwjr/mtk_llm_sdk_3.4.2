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
"""MinicpmVNavitSigLIP preprocessor class."""

import math
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import to_channel_dimension_format

from ...utils import image_utils
from .preprocessor_siglip import SigLIPImageProcessor


class MinicpmVNavitSigLIPImageProcessor(SigLIPImageProcessor):
    """MinicpmVNavitSigLIPImageProcessor class.

    This class extends the SigLIPImageProcessor class and provides methods for minicpmv.

    Attributes:
        None

    Methods:
        resize: Resizes an image.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Init MinicpmVNavitSigLIPImageProcessor."""
        super().__init__(**kwargs)

    def _ensure_divide(self, length, patch_size):
        """Ensure that a given length is divisible by the patch size.

        Args:
            length : int
                The original length to be adjusted.
            patch_size : int
                The size of the patches to ensure divisibility by.

        Returns:
            int
                The adjusted length that is divisible by the patch size.
        """
        return max(round(length / patch_size) * patch_size, patch_size)

    def _find_best_resize(self, original_size, scale_resolution, patch_size, allow_upscale=False):
        """Calculate the best resize dimensions for an image to ensure it can be evenly divided into patches.

        This function determines the optimal dimensions to resize an image so that it can be evenly divided into patches
        of a specified size. The resizing is based on the target resolution and the original size of the image.
        Optionally, the function can allow upscaling of the image if the original size is smaller than the target
        resolution.

        Args:
            original_size : tuple
                The original size of the input image in the format (height, width, channels).
            scale_resolution : int
                The target resolution to scale the image to.
            patch_size : int
                The size of the patches to extract from the image. Each patch will be of dimensions
                (patch_size, patch_size).
            allow_upscale : bool, optional
                If True, the image can be upscaled to meet the target resolution. Default is False.

        Returns:
            tuple
                A tuple containing the best width and height to resize the image to.
        """
        height, width, _channel = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self._ensure_divide(width, patch_size)
        best_height = self._ensure_divide(height, patch_size)
        return (best_width, best_height)

    def _split_to_patches(self, image, grid):
        """Split an image into smaller patches based on the specified grid dimensions.

        Args:
            image : np.ndarray
                The input image to be split into patches.
            grid : tuple
                The grid dimensions [rows, columns] to divide the image into.

        Returns:
            list of list of np.ndarray
                A nested list containing the image patches. The outer list represents rows of patches,
                and the inner lists represent columns of patches within each row.
        """
        patches = []
        height, width, _channel = image.shape
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])

        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                images.append(image[i : i + grid_y, j : j + grid_x])
            patches.append(images)

        return patches

    def _get_refine_size(self, original_size, grid, scale_resolution, patch_size, allow_upscale=False):
        """Calculate the refined size of an image to ensure it can be evenly divided into a specified grid.

        This function refines the size of an image so that it can be evenly divided into a grid of specified dimensions.
        The refined size is calculated based on the original size of the image, the grid dimensions, the target
        resolution, and the patch size. Optionally, the function can allow upscaling of the image.

        Args:
            original_size : tuple
                The original size of the input image in the format (height, width, channels).
            grid : tuple
                The grid dimensions [rows, columns] to divide the image into.
            scale_resolution : int
                The target resolution to scale the image to.
            patch_size : int
                The size of the patches to extract from the image. Each patch will be of dimensions
                (patch_size, patch_size).
            allow_upscale : bool, optional
                If True, the image can be upscaled to meet the target resolution. Default is False.

        Returns:
            tuple
                A tuple containing the refined width and height of the image.
        """
        height, width, channel = original_size
        grid_y, grid_x = grid

        refine_width = self._ensure_divide(width, grid_x)
        refine_height = self._ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self._find_best_resize(
            (grid_height, grid_width, channel), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        return (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    def _get_sliced_grid(self, image_size, scale_resolution, max_slice_nums, never_split=False):
        """Determine the best grid configuration for slicing an image based on its size and specified parameters.

        This function calculates the optimal grid dimensions for slicing an image into smaller patches. The grid is
        determined based on the image's aspect ratio, the target resolution, and the maximum number of slices allowed.
        If slicing is not required or explicitly disabled, the function returns None.

        Args:
            image_size : tuple
                The size of the input image in the format (height, width, channels).
            scale_resolution : int
                The target resolution to scale the image to.
            max_slice_nums : int
                The maximum number of slices to divide the image into.
            never_split : bool, optional
                If True, the image will never be split into patches and the function will return None. Default is False.

        Returns:
            list or None
                A list containing the best grid dimensions [rows, columns] for slicing the image, or None if slicing is
                not required.
        """
        original_height, original_width, _original_channel = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or never_split:
            return None
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float('inf')
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def slice_image(self, image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
        """Slice an image into smaller patches or resize it based on specified parameters.

        This function takes an input image and either slices it into smaller patches or resizes it, depending on the
        provided parameters and the image's original size. The slicing is done in a way that ensures the patches are
        of a specified size and the image can be divided evenly. If slicing is not needed, the image is upsampled
        to the desired resolution.

        Args:
            image : np.ndarray
                The input image to be processed.
            max_slice_nums : int, optional
                The maximum number of slices to divide the image into. Default is 9.
            scale_resolution : int, optional
                The target resolution to scale the image to. Default is 448.
            patch_size : int, optional
                The size of the patches to extract from the image. Each patch will be of dimensions
                (patch_size, patch_size). Default is 14.
            never_split : bool, optional
                If True, the image will never be split into patches and will only be resized. Default is False.

        Returns:
            tuple
                A tuple containing:
                - slice_images: list of np.ndarray
                    A list of images, where the first image is the resized source image and the rest are
                    the sliced patches.
                - best_grid: tuple or None
                    The grid dimensions used for slicing the image, or None if the image was not sliced.
        """
        original_size = image.shape
        source_image = None
        best_grid = self._get_sliced_grid(original_size, scale_resolution, max_slice_nums, never_split)
        slice_images = []

        if best_grid is None:
            # dont need to slice, upsample
            best_resize = self._find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
            source_image = self.resize(image, {'shortest_edge': best_resize})
        else:
            # source image, down-sampling and ensure divided by patch_size
            best_resize = self._find_best_resize(original_size, scale_resolution, patch_size)
            source_image = self.resize(image, {'shortest_edge': best_resize})
            refine_size = self._get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = self.resize(image, {'shortest_edge': refine_size})
            patches = self._split_to_patches(refine_image, best_grid)

        slice_images = [source_image]
        if best_grid is not None:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

        return slice_images, best_grid

    def reshape_by_patch(
        self,
        image: np.ndarray,
        patch_size=int,
    ):
        """Reshape an image into patches of a specified size.

        This function takes an input image in the form of a NumPy array and reshapes it into smaller patches of
        a given size.

        Args:
            image : np.ndarray
                The input image to be reshaped, expected to be in the format (height, width, channels).
            patch_size : int
                The size of the patches to extract from the image. Each patch will be of dimensions
                (patch_size, patch_size).

        Returns:
            np.ndarray
                A NumPy array containing the reshaped patches. The output array will have the shape
                (number_of_patches, patch_size, patch_size, channels).
        """
        image = torch.from_numpy(image).permute(2, 0, 1)
        patches = nn.functional.unfold(image, (patch_size, patch_size), stride=(patch_size, patch_size))
        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        patches = patches.permute(1, 2, 0)
        return patches.numpy()

    def preprocess(
        self,
        images: image_utils.ImageInput,
        do_resize: Optional[bool] = None,
        size=None,
        resample: image_utils.PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, image_utils.TensorType]] = None,
        data_format: Optional[image_utils.ChannelDimension] = image_utils.ChannelDimension.FIRST,
        do_reshape_by_patch: Optional[bool] = None,
        patch_size=None,
        do_slice_mode: Optional[bool] = None,
        max_slice_nums=None,
        scale_resolution=None,
        **kwargs,
    ) -> PIL.Image.Image:
        """Preprocess an image or batch of images.

        Args:
            images (`image_utils.ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum
                `image_utils.PILImageResampling`. Only has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `image_utils.TensorType`, *optional*):
                The type of tensors to return.
                - Unset: Return a list of `np.ndarray`.
            data_format (`image_utils.ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. Can be one of:
                - `image_utils.ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `image_utils.ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: defaults to the channel dimension format of the input image.
            do_reshape_by_patch (`bool`, *optional*, defaults to `self.do_reshape_by_patch`):
                Whether to unfold the input image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                patch_size for unfolding the input image.
            do_slice_mode (`bool`, *optional*, defaults to `self.do_slice_mode`):
                Whether to slice the input image.
            max_slice_nums (`int`, *optional*, defaults to `self.max_slice_nums`):
                The maximum number of slices for slicing the input image.
            scale_resolution (`int`, *optional*, defaults to `self.scale_resolution`):
                The resolution for the sliced grid.
            **kwargs: Other kwargs.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        if isinstance(size, int):
            size = {'shortest_edge': size}
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        if isinstance(crop_size, int):
            crop_size = {'height': crop_size, 'width': crop_size}
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = image_utils.make_list_of_images(images)

        if not image_utils.valid_images(images):
            raise ValueError(
                'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, '
                'torch.Tensor, tf.Tensor or jax.ndarray.'
            )

        if do_resize and size is None:
            raise ValueError('Size must be specified if do_resize is True.')

        if do_center_crop and crop_size is None:
            raise ValueError('Crop size must be specified if do_center_crop is True.')

        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [image.convert('RGB') for image in images]

        # All transformations expect numpy arrays.
        images = [image_utils.to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_center_crop:
            images = [self.center_crop(image=image, size=crop_size) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        best_grid = None
        if do_slice_mode:
            images, best_grid = next(
                self.slice_image(
                    image=image, max_slice_nums=max_slice_nums, scale_resolution=scale_resolution, patch_size=patch_size
                )
                for image in images
            )

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        tgt_sizes = []
        for slice_image in images:
            tgt_sizes.append(np.array((slice_image.shape[0] // patch_size, slice_image.shape[1] // patch_size)))
        tgt_sizes = np.vstack(tgt_sizes)

        if do_reshape_by_patch:
            images = [self.reshape_by_patch(image=image, patch_size=patch_size) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {'input_features': images}

        kwargs.update({'tgt_sizes': tgt_sizes, 'grid': best_grid})

        return BatchFeature(data=data, tensor_type=return_tensors), kwargs
