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
"""Define functions to load dataset samples.

Functions:
    get_c4: Get C4 dataset samples.
    get_wikitext2: Get WikiText-2 dataset samples.
    get_dataset: Get specific dataset samples.
"""

import os

import torch
from sentencepiece import SentencePieceProcessor

from . import logger


def get_c4(tokenizer, split, max_len, dtype):
    """Get C4 samples."""
    logger.info(f'Loading C4 dataset, {split} split.')
    from datasets import load_dataset

    assert split in ('train', 'validation')
    dataset = load_dataset(
        os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)),
        data_files={
            split: os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, f'datasets/c4/c4-{split}.json.gz'))
        },
        split=split,
    )
    if isinstance(tokenizer, SentencePieceProcessor):
        samples = torch.tensor([tokenizer.encode(' '.join(dataset['text']))], dtype=torch.int64)
    else:
        samples = tokenizer(' '.join(dataset['text']), return_tensors='pt').input_ids

    if split == 'train':
        nsamples = samples.shape[1] // max_len
        lines = []
        for i in range(nsamples):
            curr_inputs = samples[:, i * max_len : (i + 1) * max_len]
            if not isinstance(dtype, torch.dtype):
                curr_inputs = curr_inputs.cpu().numpy().astype(dtype)
            lines.append({'c4': curr_inputs})
        return lines
    if not isinstance(dtype, torch.dtype):
        samples = samples.cpu().numpy().astype(dtype)
    return samples


def get_wikitext2(tokenizer, split, max_len, dtype):
    """Get WikiText-2 samples."""
    logger.info(f'Loading wikitext2 dataset, {split} split.')
    from datasets import Dataset

    assert split in ('train', 'test')
    dataset = Dataset.from_file(
        os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, f'datasets/wikitext/wikitext-{split}.arrow'))
    )
    if isinstance(tokenizer, SentencePieceProcessor):
        samples = torch.tensor([tokenizer.encode('\n\n'.join(dataset['text']))], dtype=torch.int64)
    else:
        samples = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt').input_ids

    if split == 'train':
        nsamples = samples.shape[1] // max_len
        lines = []
        for i in range(nsamples):
            curr_inputs = samples[:, i * max_len : (i + 1) * max_len]
            if not isinstance(dtype, torch.dtype):
                curr_inputs = curr_inputs.cpu().numpy().astype(dtype)
            lines.append({'wikitext': curr_inputs})
        return lines
    if not isinstance(dtype, torch.dtype):
        samples = samples.cpu().numpy().astype(dtype)
    return samples


def get_dataset(name, tokenizer, split, max_len=2048, dtype=torch.int32):
    """Get specific dataset samples."""
    return {'c4': get_c4, 'wikitext': get_wikitext2, 'wikitext2': get_wikitext2}[name](tokenizer, split, max_len, dtype)
