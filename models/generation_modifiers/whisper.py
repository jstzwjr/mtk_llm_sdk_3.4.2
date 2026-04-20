# Copyright (C) 2020 MediaTek Inc. All rights reserved.
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
"""Define Whisper logits processor."""

import torch
from transformers.generation.logits_process import LogitsProcessor


class WhisperLogitsProcessor(LogitsProcessor):
    """Whisper Logits Processor class.

    This class implements the Whisper  Logits Processor component.

    """

    def __init__(self, config):
        """Initialize the WhisperLogitsProcessor class.

        Args:
            config (object): The configuration object.
        """
        self.config = config

    def __call__(self, input_ids, scores):
        """Forward pass to get processed logits.

        Args:
            input_ids (torch.Tensor, optional): Input tokens. Defaults to None.
            scores (torch.Tensor, optional): Logits output from model

        Returns:
            torch.Tensor or tuple: The embeddings.
        """
        if len(input_ids[0]) == 1 or len(input_ids[0]) > 3:
            if len(input_ids[0]) > 3 and (input_ids[0][3] == self.config.kwargs['no_timestamps_token_id'] + 1):
                # do timestamp thingy
                timestamp_begin = self.config.kwargs['no_timestamps_token_id'] + 1
                eos_token_id = self.config.kwargs['eos_token_id']
                num_forced = (
                    len(self.config.kwargs['forced_decoder_ids'])
                    if self.config.kwargs['forced_decoder_ids'] is not None
                    else 0
                )
                begin_index = num_forced + 1
                max_initial_timestamp_index = self.config.kwargs['max_initial_timestamp_index']
                scores_processed = scores.clone()
                scores_processed[:, self.config.kwargs['no_timestamps_token_id']] = -100.0

                # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
                input_ids1 = torch.tensor(input_ids)
                for k in range(input_ids1.shape[0]):
                    sampled_tokens = input_ids1[k, begin_index:]
                    seq = list(sampled_tokens.tolist())

                    last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
                    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin

                    if last_was_timestamp:
                        if penultimate_was_timestamp:  # has to be non-timestamp
                            scores_processed[k, timestamp_begin:] = -100.0
                        else:  # cannot be normal text tokens
                            scores_processed[k, :eos_token_id] = -100.0

                    timestamps = sampled_tokens[sampled_tokens.ge(timestamp_begin)]
                    if timestamps.numel() > 0:
                        # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                        # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                        if last_was_timestamp and not penultimate_was_timestamp:
                            timestamp_last = timestamps[-1]
                        else:
                            # Avoid to emit <|0.00|> again
                            timestamp_last = timestamps[-1] + 1

                        scores_processed[k, timestamp_begin:timestamp_last] = -100.0

                # apply the `max_initial_timestamp` option
                if input_ids.shape[1] == begin_index:
                    scores_processed[:, :timestamp_begin] = -100.0

                    if max_initial_timestamp_index is not None:
                        last_allowed = timestamp_begin + max_initial_timestamp_index
                        scores_processed[:, last_allowed + 1 :] = -100.0

                # if sum of probability over timestamps is above any other token, sample timestamp
                logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
                for k in range(input_ids.shape[0]):
                    timestamp_logprob = logprobs[k, timestamp_begin:].logsumexp(dim=-1)
                    max_text_token_logprob = logprobs[k, :timestamp_begin].max()
                    if timestamp_logprob > max_text_token_logprob:
                        scores_processed[k, :timestamp_begin] = -100.0

                scores = scores_processed

            if len(self.config.kwargs['suppress_tokens']) != 0:
                for i in range(len(self.config.kwargs['suppress_tokens'])):
                    suppressed_token = self.config.kwargs['suppress_tokens'][i]
                    scores[0, int(suppressed_token)] = -100.0  # -float("inf")

        return scores
