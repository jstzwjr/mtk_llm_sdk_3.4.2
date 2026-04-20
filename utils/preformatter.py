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
"""Module to define the Preformatter class."""

import json
import os

from . import logger


class Preformatter:
    """A class to handle prompt formatting based on a template.

    Attributes:
        _passthrough (bool): Whether to bypass formatting.
        name (str): The name of the template.
        template (dict): The template loaded from a JSON file.
        applied (bool): Whether preformatter has been applied.
    """

    __slots__ = ('_passthrough', '_verbose', 'name', 'template')

    def __init__(self, template_path):
        """Initializes the Preformatter with a template.

        Args:
            template_path (str or None): The path to the template JSON file. If None, passthrough mode is enabled.

        Raises:
            ValueError: If the template file cannot be read.
        """
        if template_path is None:
            logger.debug('[Preformatter] No preformatter')
            self._passthrough = True
            self.name = 'no'
            self._verbose = False
        else:
            logger.debug(f'[Preformatter] Using preformatter path: {template_path}')
            self._passthrough = False
            self.name = os.path.basename(template_path).rsplit('.', 1)[0]
            logger.debug(f'[Preformatter] name={self.name}')
            if not os.path.exists(template_path):
                logger.error(f"Can't read preformatter template json: {template_path}", err=ValueError)
            with open(template_path) as fp:
                self.template = json.load(fp)

    @property
    def used(self):
        """Property to check if preformatter is actually used or just a passthrough class."""
        return not self._passthrough

    def generate_prompt(
        self,
        instruction,
        input_=None,
        label=None,
    ):
        """Generates a prompt based on the instruction and optional input and label.

        Args:
            instruction (str): The instruction to include in the prompt.
            input_ (str, optional): The additional input to include in the prompt. Default is None.
            label (str, optional): The label (response/output) to append to the prompt. Default is None.

        Returns:
            str: The generated prompt.
        """
        # returns the full prompt from instruction and optional input_
        # if a label (=response, =output) is provided, it's also appended.
        if self._passthrough:
            return instruction + label if label else instruction
        if input_ is not None:
            logger.debug('[Preformatter] Using `prompt_input` template')
            res = self.template['prompt_input']
            template_split = res.split('{input}')
            res = res.replace('{input}', input_)
        else:
            logger.debug('[Preformatter] Using `prompt_no_input` template')
            res = self.template['prompt_no_input']
            template_split = res.split('{instruction}')  # noqa: RUF027
        res = (
            res.replace('{instruction}', instruction)
            if not all(i in instruction for i in template_split)
            else instruction
        )
        if label:
            res += label
        logger.debug(f'Formatted Prompt:\n{res}')
        return res

    def get_response(self, output):
        """Extracts the response from the output based on the template.

        Args:
            output (str): The output to extract the response from.

        Returns:
            str: The extracted response.
        """
        logger.debug(f'[Preformatter] output before extracting response={output}')
        if self._passthrough:
            return output
        return output.rsplit(self.template['response_split'], 1)[-1].strip()
