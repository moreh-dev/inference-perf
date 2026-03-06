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

"""Tests for ContentGenerator.

The one contract that matters: generate(N) samples exactly N token IDs
from [0, vocab_size) and passes them through the tokenizer's decode.
"""

from unittest.mock import Mock

from inference_perf.utils.content_generator import ContentGenerator


def _mock_tokenizer(vocab_size=32000):
    hf_tok = Mock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = lambda ids: " ".join(f"tok{i}" for i in ids)
    custom_tok = Mock()
    custom_tok.get_tokenizer.return_value = hf_tok
    return custom_tok


class TestContentGenerator:

    def test_samples_exact_token_count(self):
        """generate(N) must decode exactly N token IDs."""
        captured = []

        hf_tok = Mock()
        hf_tok.vocab_size = 100
        hf_tok.decode = lambda ids: (captured.extend(ids), "text")[1]

        custom_tok = Mock()
        custom_tok.get_tokenizer.return_value = hf_tok

        gen = ContentGenerator(custom_tok)
        gen.generate(42)
        assert len(captured) == 42

    def test_token_ids_in_vocab_range(self):
        """All sampled IDs must be in [0, vocab_size)."""
        captured = []

        hf_tok = Mock()
        hf_tok.vocab_size = 50
        hf_tok.decode = lambda ids: (captured.extend(ids), "text")[1]

        custom_tok = Mock()
        custom_tok.get_tokenizer.return_value = hf_tok

        gen = ContentGenerator(custom_tok)
        gen.generate(500)
        assert all(0 <= tid < 50 for tid in captured)

    def test_zero_returns_empty(self):
        gen = ContentGenerator(_mock_tokenizer())
        assert gen.generate(0) == ""
        assert gen.generate(-1) == ""
