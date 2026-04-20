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
"""Define base hook class."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Tuple

import numpy as np
import torch

from ..utils import logger
from .configuration_hook import GetEmbedsHooksConfig, HookConfig


class ModelOutput(OrderedDict):
    """Base class for all model outputs as dataclass.

    Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ModelOutput class.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Raises:
        TypeError: If the subclass of ModelOutput does not use the @dataclass decorator.
        """
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            logger.error(
                f'{self.__module__}.{self.__class__.__name__} is not a dataclasss.'
                ' This is a subclass of ModelOutput and so must use the @dataclass decorator.',
                err=TypeError,
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            logger.error(f'{self.__class__.__name__} has no fields.', err=ValueError)
        if not all(field.default is None for field in class_fields[1:]):
            logger.error(f'{self.__class__.__name__} should not have more than one required field.', err=ValueError)

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not (isinstance(first_field, (torch.Tensor, np.ndarray))):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            logger.error(
                                f'Cannot set key/value for {element}. It needs to be a tuple (key, value).',
                                err=ValueError,
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        """Prevent deletion of items.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Raises:
        Exception: Always raises an exception to prevent deletion.
        """
        logger.error(f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.')

    def setdefault(self, *args, **kwargs):
        """Prevent setting default values.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Raises:
        Exception: Always raises an exception to prevent setting default values.
        """
        logger.error(f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.')

    def pop(self, *args, **kwargs):
        """Prevent popping of items.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Raises:
        Exception: Always raises an exception to prevent popping of items.
        """
        logger.error(f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        """Prevent updating of items.

        Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Raises:
        Exception: Always raises an exception to prevent updating of items.
        """
        logger.error(f'You cannot use ``update`` on a {self.__class__.__name__} instance.')

    def __getitem__(self, k):
        """Get item by key.

        Args:
        k (str or int): Key or index to retrieve the item.

        Returns:
        Any: The item corresponding to the key or index.
        """
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        return self.to_tuple()[k]

    def __setattr__(self, name, value):
        """Set attribute value.

        Args:
        name (str): Name of the attribute.
        value (Any): Value to set for the attribute.
        """
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        """Set item value.

        Args:
        key (str): Key of the item.
        value (Any): Value to set for the item.
        """
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        """Reduce the object for pickling.

        Returns:
        tuple: Reduced representation of the object.
        """
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()  # noqa: A001
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any]:
        """Convert self to a tuple containing all the attributes/keys that are not `None`."""
        return tuple(self[k] for k in self.keys())


@dataclass
class HookOutput(ModelOutput):
    """Dataclass for HookOutput.

    This class extends the ModelOutput class to include additional outputs specific to hooks.

    Attributes:
        forward_out (Any): Output of the forward pass.
        additional_0 (Any): Additional output 0.
        additional_1 (Any): Additional output 1.

    Methods:
        __post_init__: Check the HookOutput dataclass.
    """

    forward_out: Any = None
    additional_0: Any = None
    additional_1: Any = None


class BaseHook(ABC, torch.nn.Module):
    """Abstract base class for hooks.

    This class extends the torch.nn.Module and provides an abstract base for hooks.

    Attributes:
        config (HookConfig): Configuration object for the hook.
    """

    def __init__(self, config: HookConfig, **kwargs):
        """Initialize the BaseHook class.

        Args:
            config (HookConfig): Configuration object for the hook.
            **kwargs: Additional keyword arguments.
        """
        torch.nn.Module.__init__(self)
        self.config = config
        self.name = config.name

    @abstractmethod
    def forward(self):
        """Abstract method for the forward pass."""


class BaseGetEmbedsHook(BaseHook):
    """Abstract base class for get_embeds hooks."""

    def __init__(self, config: GetEmbedsHooksConfig, **kwargs):
        """Initialize the BaseGetEmbedsHook class.

        Args:
            config (GetEmbedsHooksConfig): Configuration object for the hook.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config)
        self.text_embedding_layer = kwargs.pop('text_embedding_layer', None)
        self.dtype = kwargs.pop('dtype', None)
        self.mm_token_ids = []
        self.mm_token_ids.extend(self.config.audio_token_ids)
        self.mm_token_ids.extend(self.config.image_token_ids)

    def get_mm_mask(self, input_ids):
        """Get multimodal token position mask.

        Args:
             input_ids (np.ndarray): Input token IDs
        """
        logger.debug('Enter BaseGetEmbedsHook get_mm_mask')
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]

        return [x in self.mm_token_ids for x in input_ids]
