# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.  # noqa: CPY001
#
# This code is based on transformers/src/transformers/models/llama/tokenization_llama_fast.py
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
"""Tokenization classes for InternLM."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from .tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {'vocab_file': './tokenizer.model'}

PRETRAINED_VOCAB_FILES_MAP = {}


# Modified from transformers.model.llama.tokenization_llama.LlamaTokenizer
class InternLM2Tokenizer(PreTrainedTokenizer):
    """Construct a InternLM2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Attributes:
        vocab_file (`str`): Path to the vocabulary file.
        unk_token (`str`, optional): The unknown token. Defaults to '<unk>'.
        bos_token (`str`, optional): The beginning of sequence token. Defaults to '<s>'.
        eos_token (`str`, optional): The end of sequence token. Defaults to '</s>'.
        pad_token (`str`, optional): The padding token. Defaults to '</s>'.
        sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the SentencePiece model.
            Defaults to None.
        add_bos_token (bool, optional): Whether to add a beginning of sequence token. Defaults to True.
        add_eos_token (bool, optional): Whether to add an end of sequence token. Defaults to False.
        decode_with_prefix_space (bool, optional): Whether to decode with a prefix space. Defaults to False.
        clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Defaults to False.

    Methods:
        no_prefix_space_tokens: Returns the set of tokens that do not have a prefix space.
        vocab_size: Returns the size of the vocabulary.
        bos_token_id: Returns the ID for the beginning of sequence token.
        eos_token_id: Returns the ID for the end of sequence token.
        get_vocab: Returns the vocabulary as a dictionary.
        _tokenize: Tokenizes a given text.
        _convert_token_to_id: Converts a token to its corresponding ID.
        _convert_id_to_token: Converts an ID to its corresponding token.
        _maybe_add_prefix_space: Adds a prefix space to the decoded string if necessary.
        convert_tokens_to_string: Converts a sequence of tokens to a single string.
        save_vocabulary: Saves the vocabulary to a specified directory.
        build_inputs_with_special_tokens: Builds input sequences with special tokens.
        get_special_tokens_mask: Retrieves the special tokens mask for a sequence.
        create_token_type_ids_from_sequences: Creates token type IDs for sequence pairs.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    _auto_class = 'AutoTokenizer'

    def __init__(
        self,
        vocab_file,
        unk_token='<unk>',
        bos_token='<s>',
        eos_token='</s>',
        pad_token='</s>',
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        decode_with_prefix_space=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """Initializes the InternLM2Tokenizer.

        Args:
            vocab_file (`str`): Path to the vocabulary file.
            unk_token (`str`, optional): The unknown token. Defaults to '<unk>'.
            bos_token (`str`, optional): The beginning of sequence token. Defaults to '<s>'.
            eos_token (`str`, optional): The end of sequence token. Defaults to '</s>'.
            pad_token (`str`, optional): The padding token. Defaults to '</s>'.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the SentencePiece model.
                Defaults to None.
            add_bos_token (bool, optional): Whether to add a beginning of sequence token. Defaults to True.
            add_eos_token (bool, optional): Whether to add an end of sequence token. Defaults to False.
            decode_with_prefix_space (bool, optional): Whether to decode with a prefix space. Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self._no_prefix_space_tokens = None
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def no_prefix_space_tokens(self):
        """Returns the set of tokens that do not have a prefix space.

        Returns:
            set: The set of tokens that do not have a prefix space.
        """
        if self._no_prefix_space_tokens is None:
            vocab = self.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {i for i, tok in enumerate(vocab) if not tok.startswith('▁')}
        return self._no_prefix_space_tokens

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        """Returns the ID for the beginning of sequence token.

        Returns:
            Optional[int]: The ID for the beginning of sequence token.
        """
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        """Returns the ID for the end of sequence token.

        Returns:
            Optional[int]: The ID for the end of sequence token.
        """
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns the vocabulary as a dictionary.

        Returns:
            dict: The vocabulary.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def _maybe_add_prefix_space(self, tokens, decoded):
        if tokens and tokens[0] not in self.no_prefix_space_tokens:
            return ' ' + decoded
        return decoded

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ''
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += ' '
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        out_string = self.clean_up_tokenization(out_string)
        out_string = self._maybe_add_prefix_space(tokens=tokens, decoded=out_string)
        return out_string[1:]

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to a specified directory.

        Args:
            save_directory (`str`): The directory to save the vocabulary.
            filename_prefix (Optional[str], optional): The prefix for the filename. Defaults to None.

        Returns:
            Tuple[str]: The paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            raise NotADirectoryError(f'Vocabulary path ({save_directory}) should be a directory')
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file']
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Builds input sequences with special tokens.

        Args:
            token_ids_0 (List[int]): The first sequence of token IDs.
            token_ids_1 (List[int], optional): The second sequence of token IDs. Defaults to None.

        Returns:
            List[int]: The input sequence with special tokens.
        """
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = [*output, self.eos_token_id]

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Retrieve sequence ids from a token list that has no special tokens added.

        This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        T5 does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
