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

"""Tests for Turn, ToolCall, Session models.

Focuses on CSV round-trips (lossy tool-metric splitting), Session builder
validation, and context growth computation.
"""

import pytest

from inference_perf.models.turn import FinishReason, ToolCall, Turn
from inference_perf.models.session import Session, SessionSummary


class TestTurnCsvRoundtrip:
    """CSV is the trace ingestion path. Tool metrics are summed on write
    and evenly split on read — verify the lossy round-trip is correct."""

    def test_single_tool(self):
        t = Turn(
            session_id="s1", turn_index=0, input_tokens=500, output_tokens=100,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[ToolCall(name="tool_0", duration_ms=400, result_tokens=52)],
            llm_latency_ms=800,
        )
        row = t.to_csv_row()
        t2 = Turn.from_csv_row(row, session_id="s1")
        assert t2.input_tokens == 500
        assert t2.finish_reason == FinishReason.TOOL_CALLS
        assert len(t2.tool_calls) == 1
        assert t2.tool_calls[0].duration_ms == 400

    def test_multiple_tools_split_evenly(self):
        t = Turn(
            session_id="s1", turn_index=0, input_tokens=500, output_tokens=100,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[
                ToolCall(name="a", duration_ms=200, result_tokens=60),
                ToolCall(name="b", duration_ms=200, result_tokens=60),
            ],
        )
        row = t.to_csv_row()
        assert row["tool_duration_ms"] == 400
        assert row["tool_result_tokens"] == 120

        t2 = Turn.from_csv_row(row, session_id="s1")
        assert len(t2.tool_calls) == 2
        assert all(tc.duration_ms == 200 for tc in t2.tool_calls)
        assert all(tc.result_tokens == 60 for tc in t2.tool_calls)

    def test_unknown_finish_reason_falls_back(self):
        row = {
            "turn_index": "0", "input_tokens": "100", "output_tokens": "50",
            "finish_reason": "some_new_reason",
            "num_tool_calls": "0", "tool_duration_ms": "0",
            "tool_result_tokens": "0", "llm_latency_ms": "0",
        }
        assert Turn.from_csv_row(row, session_id="s1").finish_reason == FinishReason.UNKNOWN


class TestSession:

    def test_add_turn_computes_new_context(self):
        s = Session(session_id="s1")
        s.add_turn(Turn(session_id="s1", turn_index=0, input_tokens=300, output_tokens=50))
        s.add_turn(Turn(session_id="s1", turn_index=1, input_tokens=500, output_tokens=80))
        assert s.turns[1].new_context_tokens == 200

    def test_add_turn_rejects_wrong_session_id(self):
        s = Session(session_id="s1")
        with pytest.raises(ValueError, match="doesn't match"):
            s.add_turn(Turn(session_id="wrong", turn_index=0, input_tokens=100, output_tokens=50))

    def test_add_turn_rejects_wrong_index(self):
        s = Session(session_id="s1")
        with pytest.raises(ValueError, match="doesn't match expected"):
            s.add_turn(Turn(session_id="s1", turn_index=5, input_tokens=100, output_tokens=50))

    def test_context_growth_rate(self):
        s = Session(session_id="s1")
        s.add_turn(Turn(session_id="s1", turn_index=0, input_tokens=300, output_tokens=50))
        s.add_turn(Turn(session_id="s1", turn_index=1, input_tokens=500, output_tokens=50))
        s.add_turn(Turn(session_id="s1", turn_index=2, input_tokens=900, output_tokens=50))
        # growth: 200, 400 → avg = 300
        assert s.context_growth_rate == 300.0

    def test_is_complete(self):
        s = Session(session_id="s1", turns=[
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50,
                 finish_reason=FinishReason.TOOL_CALLS),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=50,
                 finish_reason=FinishReason.STOP),
        ])
        assert s.is_complete()

    def test_computed_fields(self):
        s = Session(session_id="s1", turns=[
            Turn(session_id="s1", turn_index=0, input_tokens=300, output_tokens=80,
                 finish_reason=FinishReason.TOOL_CALLS,
                 tool_calls=[ToolCall(name="t", duration_ms=200, result_tokens=50)],
                 llm_latency_ms=500),
            Turn(session_id="s1", turn_index=1, input_tokens=600, output_tokens=100,
                 llm_latency_ms=700),
        ])
        assert s.num_turns == 2
        assert s.num_tool_calls == 1
        assert s.total_input_tokens == 900
        assert s.total_output_tokens == 180
        assert s.max_context_length == 600
        assert s.total_tool_pause_ms == 200
        assert s.total_llm_latency_ms == 1200
        assert s.original_session_latency_ms == 1400
