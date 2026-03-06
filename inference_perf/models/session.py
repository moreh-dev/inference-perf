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

"""Session models for agentic workload benchmarking.

A Session represents a complete agent execution - a single user request
that triggers a multi-turn agent loop.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, computed_field

from .turn import Turn, FinishReason


class Session(BaseModel):
    """Represents a complete agentic session (multi-turn agent execution).

    A session contains a sequence of turns that represent one complete
    agent execution. The turns are ordered by turn_index and represent
    the causal chain of LLM calls with tool executions in between.

    Attributes:
        session_id: Unique identifier for this session.
        turns: Ordered list of turns in this session.
        original_start_time_ms: Original session start timestamp (for trace replay).
        metadata: Optional metadata (adapter name, tags, etc.).
    """
    session_id: str = Field(..., description="Unique identifier for this session")
    turns: List[Turn] = Field(
        default_factory=list,
        description="Ordered list of turns in this session"
    )
    original_start_time_ms: Optional[int] = Field(
        default=None,
        description="Original session start timestamp for trace replay (ms)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata (adapter name, tags, etc.)"
    )

    @computed_field
    @property
    def num_turns(self) -> int:
        """Number of turns in this session."""
        return len(self.turns)

    @computed_field
    @property
    def num_tool_calls(self) -> int:
        """Total number of tool calls across all turns."""
        return sum(len(turn.tool_calls) for turn in self.turns)

    @computed_field
    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens across all turns."""
        return sum(turn.input_tokens for turn in self.turns)

    @computed_field
    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens across all turns."""
        return sum(turn.output_tokens for turn in self.turns)

    @computed_field
    @property
    def max_context_length(self) -> int:
        """Maximum input tokens in any single turn (peak memory)."""
        if not self.turns:
            return 0
        return max(turn.input_tokens for turn in self.turns)

    @computed_field
    @property
    def total_tool_pause_ms(self) -> int:
        """Sum of all tool execution durations."""
        return sum(turn.total_tool_duration_ms for turn in self.turns)

    @computed_field
    @property
    def total_llm_latency_ms(self) -> int:
        """Sum of LLM call durations from source trace."""
        return sum(turn.llm_latency_ms or 0 for turn in self.turns)

    @computed_field
    @property
    def original_session_latency_ms(self) -> int:
        """Total session duration from source trace (LLM + tool pauses)."""
        return self.total_llm_latency_ms + self.total_tool_pause_ms

    @property
    def context_growth_rate(self) -> float:
        """Average increase in input tokens per turn."""
        if self.num_turns <= 1:
            return 0.0
        total_growth = sum(turn.new_context_tokens for turn in self.turns[1:])
        return total_growth / (self.num_turns - 1)

    @property
    def adapter_name(self) -> Optional[str]:
        """Get the adapter name from metadata if present."""
        return self.metadata.get("adapter_name")

    @adapter_name.setter
    def adapter_name(self, value: str) -> None:
        """Set the adapter name in metadata."""
        self.metadata["adapter_name"] = value

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to the session.

        Automatically computes new_context_tokens based on previous turn.
        """
        if turn.session_id != self.session_id:
            raise ValueError(
                f"Turn session_id '{turn.session_id}' doesn't match "
                f"session '{self.session_id}'"
            )

        if turn.turn_index != len(self.turns):
            raise ValueError(
                f"Turn index {turn.turn_index} doesn't match expected "
                f"index {len(self.turns)}"
            )

        if turn.new_context_tokens == 0 and self.turns:
            prev_turn = self.turns[-1]
            turn.new_context_tokens = turn.input_tokens - prev_turn.input_tokens

        self.turns.append(turn)

    def get_turn(self, turn_index: int) -> Optional[Turn]:
        """Get a turn by index."""
        if 0 <= turn_index < len(self.turns):
            return self.turns[turn_index]
        return None

    def is_complete(self) -> bool:
        """Check if the session is complete (last turn has stop finish_reason)."""
        if not self.turns:
            return False
        return self.turns[-1].finish_reason == FinishReason.STOP

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "original_start_time_ms": self.original_start_time_ms,
            "metadata": self.metadata,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "max_context_length": self.max_context_length,
            "total_tool_pause_ms": self.total_tool_pause_ms,
            "total_llm_latency_ms": self.total_llm_latency_ms,
            "original_session_latency_ms": self.original_session_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create Session from dictionary."""
        turns_data = data.get("turns", [])
        turns = [
            Turn.from_dict(t) if isinstance(t, dict) else t
            for t in turns_data
        ]

        return cls(
            session_id=data["session_id"],
            turns=turns,
            original_start_time_ms=data.get("original_start_time_ms"),
            metadata=data.get("metadata", {}),
        )

    def to_csv_rows(self) -> List[dict]:
        """Convert session to CSV rows (one row per turn)."""
        return [turn.to_csv_row() for turn in self.turns]


class SessionSummary(BaseModel):
    """Lightweight summary of a session for reporting purposes.

    Attributes:
        session_id: Unique identifier for this session.
        num_turns: Number of turns in the session.
        num_tool_calls: Total tool calls across all turns.
        total_input_tokens: Sum of input tokens.
        total_output_tokens: Sum of output tokens.
        max_context_length: Peak input tokens.
        total_tool_pause_ms: Sum of tool execution times.
        total_llm_latency_ms: Sum of LLM call durations.
        original_session_latency_ms: Total session duration from source.
        adapter_name: Optional adapter/LoRA name.
    """
    session_id: str
    num_turns: int = 0
    num_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    max_context_length: int = 0
    total_tool_pause_ms: int = 0
    total_llm_latency_ms: int = 0
    original_session_latency_ms: int = 0
    adapter_name: Optional[str] = None

    @classmethod
    def from_session(cls, session: Session) -> "SessionSummary":
        """Create summary from a full Session object."""
        return cls(
            session_id=session.session_id,
            num_turns=session.num_turns,
            num_tool_calls=session.num_tool_calls,
            total_input_tokens=session.total_input_tokens,
            total_output_tokens=session.total_output_tokens,
            max_context_length=session.max_context_length,
            total_tool_pause_ms=session.total_tool_pause_ms,
            total_llm_latency_ms=session.total_llm_latency_ms,
            original_session_latency_ms=session.original_session_latency_ms,
            adapter_name=session.adapter_name,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "max_context_length": self.max_context_length,
            "total_tool_pause_ms": self.total_tool_pause_ms,
            "total_llm_latency_ms": self.total_llm_latency_ms,
            "original_session_latency_ms": self.original_session_latency_ms,
            "adapter_name": self.adapter_name,
        }
