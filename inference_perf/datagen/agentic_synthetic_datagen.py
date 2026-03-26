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

"""Agentic synthetic data generator.

Generates synthetic agentic sessions from configurable distributions.
Produces Session objects with realistic token count profiles for
multi-turn agentic workload benchmarking.
"""

import logging
import random
from typing import List
from uuid import uuid4

from inference_perf.config import AgenticSyntheticConfig, DistributionParams
from inference_perf.models import Session, Turn, ToolCall, FinishReason

logger = logging.getLogger(__name__)


def _sample_distribution(params: DistributionParams) -> float:
    """Sample a value from the configured distribution."""
    if params.type == "normal":
        value = random.gauss(params.mean, params.std_dev)
        return max(params.min, min(params.max, value))
    elif params.type == "uniform":
        return random.uniform(params.min, params.max)
    elif params.type == "constant":
        return params.value or params.mean
    else:
        return params.mean


class AgenticSyntheticDataGenerator:
    """Generate synthetic agentic sessions from distribution parameters.

    Creates Session objects with configurable turns, tool calls, and
    token count profiles. Context grows naturally across turns as
    output tokens and tool results accumulate.

    The last turn of each session always has finish_reason=stop to
    ensure sessions are well-formed.
    """

    def __init__(self, config: AgenticSyntheticConfig) -> None:
        self.config = config
        self.sessions: List[Session] = []

        self._generate_sessions()

        avg_turns = sum(s.num_turns for s in self.sessions) / len(self.sessions)
        logger.info(
            f"Generated {len(self.sessions)} synthetic sessions "
            f"with average {avg_turns:.1f} turns"
        )

    def _generate_sessions(self) -> None:
        """Generate all synthetic sessions."""
        for _ in range(self.config.num_sessions):
            self.sessions.append(self._generate_session())

    def _generate_session(self) -> Session:
        """Generate a single synthetic session."""
        session_id = str(uuid4())

        num_turns = max(1, int(_sample_distribution(self.config.turns_per_session)))

        turns: List[Turn] = []
        # Turn 0 input = sampled initial tokens + system prompt
        input_tokens = max(10, int(_sample_distribution(self.config.input_tokens_turn_0)))
        input_tokens += self.config.system_prompt_tokens

        for turn_idx in range(num_turns):
            is_last = turn_idx == num_turns - 1
            prev_input_tokens = turns[-1].input_tokens if turns else 0

            output_tokens = max(1, int(_sample_distribution(self.config.output_tokens_per_turn)))

            # Last turn always stops; otherwise sample tool_call_probability
            if is_last:
                has_tool_calls = False
            else:
                has_tool_calls = random.random() < self.config.tool_call_probability

            finish_reason = FinishReason.TOOL_CALLS if has_tool_calls else FinishReason.STOP

            tool_calls: List[ToolCall] = []
            if has_tool_calls:
                num_tc = max(1, int(_sample_distribution(self.config.tool_calls_per_turn)))
                for i in range(num_tc):
                    result_tokens = max(1, int(_sample_distribution(self.config.tool_result_tokens)))
                    tool_calls.append(ToolCall(
                        name=f"tool_{i}",
                        duration_ms=max(10, result_tokens),
                        result_tokens=result_tokens,
                        tool_call_id=f"tc_{uuid4().hex[:8]}",
                    ))

            new_context_tokens = max(0, input_tokens - prev_input_tokens) if turn_idx > 0 else 0

            turns.append(Turn(
                session_id=session_id,
                turn_index=turn_idx,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                new_context_tokens=new_context_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            ))

            # Next turn's input = current input + output + tool results
            input_tokens = input_tokens + output_tokens + sum(tc.result_tokens for tc in tool_calls)

        return Session(session_id=session_id, turns=turns)

    def get_sessions(self) -> List[Session]:
        """Get all generated sessions."""
        return self.sessions
