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
"""Define Hunyuan tokenizer class."""

import base64
import os
import unicodedata
from typing import Collection, Dict, List, Optional, Set, Tuple, Union

import tiktoken
from tokenizers import AddedToken

from .tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {'vocab_file': 'hy.tiktoken'}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501
ENDOFTEXT = '<|endoftext|>'
STARTOFTEXT = '<|startoftext|>'
BOSTOKEN = '<|bos|>'
EOSTOKEN = '<|eos|>'
PADTOKEN = '<|pad|>'
EXTRAS = tuple(f'<|extra_{i}|>' for i in range(205))

SPECIAL_START_ID = 127957


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    dic = {}
    rank = 0
    for line in open(tiktoken_bpe_file, 'rb'):  # noqa: SIM115
        if line:
            token, _ = line.split()
            if base64.b64decode(token) in dic:
                continue
            dic[base64.b64decode(token)] = int(rank)
            rank += 1
    global SPECIAL_START_ID
    SPECIAL_START_ID = rank
    return dic


SPECIAL_TOKENS = tuple(
    enumerate(
        ((ENDOFTEXT, STARTOFTEXT, BOSTOKEN, EOSTOKEN, PADTOKEN, *EXTRAS)),
        start=SPECIAL_START_ID,
    )
)
# NOTE: Unused Token ID starts from 127962
SPECIAL_TOKENS_SET = {t for i, t in SPECIAL_TOKENS}


class HunYuanTokenizer(PreTrainedTokenizer):
    """HunYuan tokenizer.

    Attributes:
        vocab_file (`str`): Path to the vocabulary file.
        errors (`str`): How to handle errors in decoding UTF-8 byte sequences.
        extra_vocab_file (`str`, optional): Path to an additional vocabulary file.
        mergeable_ranks (`Dict[bytes, int]`): The mergeable ranks for byte-pair encoding.
        special_tokens (`Dict[str, int]`): The special tokens and their corresponding IDs.
        decoder (`Dict[int, Union[bytes, str]]`): The decoder for token IDs to tokens.
        tokenizer (`tiktoken.Encoding`): The tokenizer instance.
        eod_id (`int`): The end of document token ID.
        bod_id (`int`): The beginning of document token ID.
        bos_id (`int`): The beginning of sequence token ID.
        eos_id (`int`): The end of sequence token ID.
        pad_id (`int`): The padding token ID.
        eod_token (`str`): The end of document token.
        bod_token (`str`): The beginning of document token.
        bos_token (`str`): The beginning of sequence token.
        eos_token (`str`): The end of sequence token.
        pad_token (`str`): The padding token.
        eod_token_id (`int`): The end of document token ID (duplicated for alignment).
        bod_token_id (`int`): The beginning of document token ID (duplicated for alignment).
        bos_token_id (`int`): The beginning of sequence token ID (duplicated for alignment).
        eos_token_id (`int`): The end of sequence token ID (duplicated for alignment).
        pad_token_id (`int`): The padding token ID (duplicated for alignment).

    Methods:
        __getstate__: Returns the state of the tokenizer for pickling.
        __setstate__: Sets the state of the tokenizer from pickling.
        __len__: Returns the size of the vocabulary.
        get_vocab: Returns the vocabulary as a dictionary.
        convert_tokens_to_ids: Converts tokens to their corresponding IDs.
        _add_tokens: Adds new tokens to the tokenizer.
        save_vocabulary: Saves the vocabulary to a specified directory.
        tokenize: Converts a string to a sequence of tokens.
        convert_tokens_to_string: Converts a sequence of tokens to a single string.
        vocab_size: Returns the size of the vocabulary.
        _convert_id_to_token: Converts an ID to its corresponding token.
        _convert_token_to_id: Converts a token to its corresponding ID.
        _tokenize: Tokenizes a given text.
        _decode: Decodes a sequence of token IDs to a string.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors='replace',
        extra_vocab_file=None,
        **kwargs,
    ):
        """Initializes the HunYuanTokenizer.

        Args:
            vocab_file (`str`): Path to the vocabulary file.
            errors (`str`, optional): How to handle errors in decoding UTF-8 byte sequences. Defaults to 'replace'.
            extra_vocab_file (`str`, optional): Path to an additional vocabulary file. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # how to handle errors in decoding UTF-8 byte sequences
        # use ignore if you are in streaming inference
        self.errors = errors

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: Dict[bytes, int]
        self.special_tokens = {token: index for index, token in SPECIAL_TOKENS}

        # try load extra vocab from file
        if extra_vocab_file is not None:
            used_ids = set(self.mergeable_ranks.values()) | set(self.special_tokens.values())
            extra_mergeable_ranks = _load_tiktoken_bpe(extra_vocab_file)
            for token, index in extra_mergeable_ranks.items():
                if token in self.mergeable_ranks:
                    print(f'extra token {token} exists, skipping')
                    continue
                if index in used_ids:
                    print(f'the index {index} for extra token {token} exists, skipping')
                    continue
                self.mergeable_ranks[token] = index
            # the index may be sparse after this, but don't worry tiktoken.Encoding will handle this

        enc = tiktoken.Encoding(
            'HunYuan',
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab, (
            f'{len(self.mergeable_ranks)} + {len(self.special_tokens)} != {enc.n_vocab} in encoding'
        )

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.bod_id = self.special_tokens[STARTOFTEXT]
        self.bos_id = self.special_tokens[BOSTOKEN]
        self.eos_id = self.special_tokens[EOSTOKEN]
        self.pad_id = self.special_tokens[PADTOKEN]

        self.eod_token = ENDOFTEXT
        self.bod_token = STARTOFTEXT
        self.bos_token = BOSTOKEN
        self.eos_token = EOSTOKEN
        self.pad_token = PADTOKEN

        # duplicated for align to 29W
        self.eod_token_id = self.tokenizer.eot_token
        self.bod_token_id = self.special_tokens[STARTOFTEXT]
        self.bos_token_id = self.special_tokens[BOSTOKEN]
        self.eos_token_id = self.special_tokens[EOSTOKEN]
        self.pad_token_id = self.special_tokens[PADTOKEN]

    def __getstate__(self):
        """Returns the state of the tokenizer for pickling.

        Returns:
            dict: The state of the tokenizer.
        """
        # for pickle lovers
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        """Sets the state of the tokenizer from pickling.

        Args:
            state (dict): The state of the tokenizer.
        """
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            'HunYuan',
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        """Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        """Returns the vocabulary as a dictionary.

        Returns:
            Dict[bytes, int]: The vocabulary.
        """
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> List[int]:
        """Converts tokens to their corresponding IDs.

        Args:
            tokens (Union[bytes, str, List[Union[bytes, str]]]): The tokens to convert.

        Returns:
            List[int]: The corresponding IDs.
        """
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError('Adding regular tokens is not supported')
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS_SET:
                raise ValueError('Adding unknown special tokens is not supported')
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, 'hunyuan.tiktoken')
        with open(file_path, 'w', encoding='utf8') as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode('utf8') + ' ' + str(v) + '\n'
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = 'all',
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize('NFC', text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """Converts a sequence of tokens in a single string."""
        text = ''
        temp = b''
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode('utf-8', errors=self.errors)
                    temp = b''
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError('token should only be of type types or str')
        if temp:
            text += temp.decode('utf-8', errors=self.errors)
        return text

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included."""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError('unknown ids')

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError('unknown token')

    def _tokenize(self, text: str, **kwargs):
        """Converts a string in a sequence of tokens (string), using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Decodes a sequence of token IDs to a string.

        Args:
            token_ids (Union[int, List[int]]): The sequence of token IDs.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to False.
            errors (Optional[str], optional): How to handle errors in decoding. Defaults to None.
            **kwargs: Other kwrags.

        Returns:
            str: The decoded string.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
