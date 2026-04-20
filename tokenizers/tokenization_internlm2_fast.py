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
"""Tokenization Fast class for InternLM."""

import os
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

from tokenizers import Tokenizer, decoders, normalizers, processors
from tokenizers.models import BPE
from transformers.convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS,
    SentencePieceExtractor,
    SpmConverter,
)

from .tokenization_internlm2 import InternLM2Tokenizer
from .tokenization_utils_fast import PreTrainedTokenizerFast

VOCAB_FILES_NAMES = {'vocab_file': './tokenizer.model'}


# Modified from transformers.convert_slow_tokenizer.LlamaConverter
class InternLM2Converter(SpmConverter):
    """InternLM2 Converter for handling byte-level BPE tokenization.

    This class provides methods to convert a SentencePiece model to a tokenizer
    that supports byte-level BPE tokenization with special handling for byte fallback.

    Attributes:
        handle_byte_fallback (bool): Indicates if byte fallback is handled.

    Methods:
        vocab(proto): Extracts the vocabulary from the SentencePiece model.
        unk_id(proto): Returns the ID for the unknown token.
        decoder(replacement, add_prefix_space): Creates the decoder sequence for the tokenizer.
        tokenizer(proto): Creates the tokenizer from the SentencePiece model proto.
        normalizer(proto): Creates the normalizer sequence for the tokenizer.
        pre_tokenizer(replacement, add_prefix_space): Creates the pre-tokenizer for the tokenizer.
    """

    handle_byte_fallback = True

    def vocab(self, proto):
        """Extracts the vocabulary from the SentencePiece model.

        Args:
            proto: The SentencePiece model proto.

        Returns:
            List[Tuple[str, float]]: The vocabulary with pieces and their scores.
        """
        vocab = [
            ('<unk>', 0.0),
            ('<s>', 0.0),
            ('</s>', 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        """Returns the ID for the unknown token.

        Args:
            proto: The SentencePiece model proto.

        Returns:
            int: The ID for the unknown token.
        """
        return 0

    def decoder(self, replacement, add_prefix_space):
        """Creates the decoder sequence for the tokenizer.

        Args:
            replacement: The replacement character.
            add_prefix_space: Whether to add a prefix space.

        Returns:
            decoders.Sequence: The decoder sequence.
        """
        decoders_sequence = [
            decoders.Replace('▁', ' '),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if self.proto.normalizer_spec.add_dummy_prefix:
            decoders_sequence.append(decoders.Strip(content=' ', left=1))
        return decoders.Sequence(decoders_sequence)

    def tokenizer(self, proto):
        """Creates the tokenizer from the SentencePiece model proto.

        Args:
            proto: The SentencePiece model proto.

        Returns:
            Tokenizer: The tokenizer instance.

        Raises:
            RuntimeError: If the model type is not BPE.
            Exception: If the model type is not Unigram.
        """
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        # special tokens
        added_tokens = self.original_tokenizer.added_tokens_decoder
        for i in range(len(vocab_scores)):
            _piece, score = vocab_scores[i]
            if i in added_tokens:
                vocab_scores[i] = (added_tokens[i].content, score)
        if model_type == 1:
            raise RuntimeError('InternLM2 is supposed to be a BPE model!')

        if model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens([added_token for index, added_token in added_tokens.items()])
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        """Creates the normalizer sequence for the tokenizer.

        Args:
            proto: The SentencePiece model proto.

        Returns:
            normalizers.Sequence: The normalizer sequence.
        """
        normalizers_list = []
        if proto.normalizer_spec.add_dummy_prefix:
            normalizers_list.append(normalizers.Prepend(prepend='▁'))
        normalizers_list.append(normalizers.Replace(pattern=' ', content='▁'))
        return normalizers.Sequence(normalizers_list)

    def pre_tokenizer(self, replacement, add_prefix_space):
        """Creates the pre-tokenizer for the tokenizer.

        Args:
            replacement: The replacement character.
            add_prefix_space: Whether to add a prefix space.

        Returns:
            None: The pre-tokenizer instance.
        """
        return None  # noqa: RET501


SLOW_TO_FAST_CONVERTERS['InternLM2Tokenizer'] = InternLM2Converter


# Modified from transformers.model.llama.tokenization_llama_fast.LlamaTokenizerFast -> InternLM2TokenizerFast
class InternLM2TokenizerFast(PreTrainedTokenizerFast):
    """InternLM2 Tokenizer Fast implementation.

    This class provides methods for fast tokenization using the InternLM2 model.

    Attributes:
        vocab_files_names: The names of the vocabulary files.
        slow_tokenizer_class: The class for the slow tokenizer.
        padding_side: The side on which to pad sequences.
        model_input_names: The names of the model inputs.
        _auto_class: The auto class for the tokenizer.

    Methods:
        can_save_slow_tokenizer: Checks if the slow tokenizer can be saved.
        update_post_processor: Updates the underlying post processor with the current `bos_token` and `eos_token`.
        save_vocabulary: Saves the vocabulary to a specified directory.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = InternLM2Tokenizer
    padding_side = 'left'
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
        """Initializes the InternLM2TokenizerFast.

        Args:
            vocab_file (str): Path to the vocabulary file.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            bos_token (str, optional): The beginning of sequence token. Defaults to '<s>'.
            eos_token (str, optional): The end of sequence token. Defaults to '</s>'.
            pad_token (str, optional): The padding token. Defaults to '</s>'.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the SentencePiece model.
                Defaults to None.
            add_bos_token (bool, optional): Whether to add a beginning of sequence token. Defaults to True.
            add_eos_token (bool, optional): Whether to add an end of sequence token. Defaults to False.
            decode_with_prefix_space (bool, optional): Whether to decode with a prefix space. Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sp_model_kwargs=sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            decode_with_prefix_space=decode_with_prefix_space,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """Checks if the slow tokenizer can be saved.

        Returns:
            bool: True if the slow tokenizer can be saved, False otherwise.
        """
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """Updates the underlying post processor with the current `bos_token` and `eos_token`."""
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError('add_bos_token = True but bos_token = None')

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError('add_eos_token = True but eos_token = None')

        single = f'{(bos + ":0 ") if self.add_bos_token else ""}$A:0{(" " + eos + ":0") if self.add_eos_token else ""}'
        pair = f'{single}{(" " + bos + ":1") if self.add_bos_token else ""} '
        f'$B:1{(" " + eos + ":1") if self.add_eos_token else ""}'

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        """Returns whether the end of sequence token is added.

        Returns:
            bool: True if the end of sequence token is added, False otherwise.
        """
        return self._add_eos_token

    @property
    def add_bos_token(self):
        """Returns whether the beginning of sequence token is added.

        Returns:
            bool: True if the beginning of sequence token is added, False otherwise.
        """
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to a specified directory.

        Args:
            save_directory (str): The directory to save the vocabulary.
            filename_prefix (Optional[str], optional): The prefix for the filename. Defaults to None.

        Returns:
            Tuple[str]: The paths to the saved files.

        Raises:
            ValueError: If the fast tokenizer does not have the necessary information to save the vocabulary for
                a slow tokenizer.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                'Your fast tokenizer does not have the necessary information to save the vocabulary for a slow '
                'tokenizer.'
            )

        if not os.path.isdir(save_directory):
            raise NotADirectoryError(f'Vocabulary path ({save_directory}) should be a directory')
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file']
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
