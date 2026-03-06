# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Content generator for agentic workload benchmarking.

Generates text with exact token counts by sampling random token IDs
from the tokenizer vocabulary. This matches the existing pattern used
by RandomDataGenerator and ensures token-accurate context accumulation.
"""

import numpy as np

from inference_perf.utils.custom_tokenizer import CustomTokenizer


class ContentGenerator:
    """Generates text with exact token counts using the model tokenizer.

    Uses random token ID sampling from the vocabulary to guarantee
    exact token counts, matching the pattern in RandomDataGenerator.
    """

    def __init__(self, tokenizer: CustomTokenizer) -> None:
        self.tokenizer = tokenizer

        hf_tokenizer = tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError(
                    "Tokenizer does not support vocabulary size detection."
                ) from e

        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

    def generate(self, target_tokens: int) -> str:
        """Generate text with approximately the target number of tokens.

        Args:
            target_tokens: Desired token count.

        Returns:
            Generated text string.
        """
        if target_tokens <= 0:
            return ""

        token_ids = np.random.randint(0, self.vocab_size, size=target_tokens, dtype=np.int64)
        return self.tokenizer.get_tokenizer().decode(token_ids.tolist())
