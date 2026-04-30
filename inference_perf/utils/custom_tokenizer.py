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
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from inference_perf.config import CustomTokenizerConfig


class CustomTokenizer:
    def __init__(self, config: CustomTokenizerConfig) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            config.pretrained_model_name_or_path, token=config.token, trust_remote_code=config.trust_remote_code
        )

    def count_tokens(self, text: str) -> int:
        if text == "":
            return 0
        # Some HF tokenizers (e.g. gpt-oss) report model_max_length as 1e30 to mean "no limit",
        # which overflows the underlying Rust tokenizer's int conversion. Clamp to a value
        # large enough for any realistic request but safe for the C/Rust int type.
        max_length = min(self.tokenizer.model_max_length, 1_000_000)
        return len(self.tokenizer(text, truncation=True, max_length=max_length).input_ids)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer
