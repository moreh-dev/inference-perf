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

"""Tests for TurnPositionMetrics.

The platform-relevant invariants:
- TTFT decreases across turn positions (prefix caching diagnostic)
- Input tokens grow across turn positions (context accumulation)
- Mixed session lengths: only sessions reaching turn N contribute to turn N stats
"""

import pytest

from inference_perf.loadgen.session_runner import SessionResult, TurnResult
from inference_perf.metrics.session_metrics import SessionMetrics, TurnPositionMetrics


def _make_session_result(
    session_id="s1", num_turns=3,
    turn_duration=0.5, gap=0.2,
    base_input=300, input_growth=200,
    base_ttft=200.0, ttft_decay=40.0,
):
    start = 100.0
    turn_results, t = [], start
    for i in range(num_turns):
        turn_results.append(TurnResult(
            session_id=session_id, turn_index=i,
            scheduled_time=t, start_time=t, end_time=t + turn_duration,
            input_tokens=base_input + i * input_growth,
            output_tokens=80,
            ttft_ms=base_ttft - i * ttft_decay,
            success=True,
        ))
        t += turn_duration + gap
    return SessionResult(
        session_id=session_id, turn_results=turn_results,
        start_time=start, end_time=t - gap,
    )


def _make_metrics_list(n_sessions=20, n_turns=4):
    return [
        SessionMetrics.from_session_result(
            _make_session_result(
                session_id=f"s{i}", num_turns=n_turns,
                base_ttft=200.0 + i * 5, ttft_decay=40.0,
            )
        )
        for i in range(n_sessions)
    ]


class TestTurnPositionMetrics:

    def test_ttft_decreases_with_turn_position(self):
        """Prefix caching diagnostic: TTFT should drop at later turns."""
        ml = _make_metrics_list(n_sessions=20, n_turns=4)
        ttfts = [
            TurnPositionMetrics.from_session_metrics_list(i, ml).avg_ttft_ms
            for i in range(4)
        ]
        for i in range(1, len(ttfts)):
            assert ttfts[i] < ttfts[i - 1]

    def test_input_tokens_grow_with_turn_position(self):
        """Context accumulation: input tokens should increase per turn."""
        ml = _make_metrics_list(n_sessions=10, n_turns=4)
        inputs = [
            TurnPositionMetrics.from_session_metrics_list(i, ml).avg_input_tokens
            for i in range(4)
        ]
        for i in range(1, len(inputs)):
            assert inputs[i] > inputs[i - 1]

    def test_mixed_session_lengths(self):
        """Only sessions reaching turn N contribute to turn N metrics."""
        short = [
            SessionMetrics.from_session_result(
                _make_session_result(session_id=f"short_{i}", num_turns=2)
            ) for i in range(5)
        ]
        long = [
            SessionMetrics.from_session_result(
                _make_session_result(session_id=f"long_{i}", num_turns=4)
            ) for i in range(5)
        ]
        ml = short + long

        assert TurnPositionMetrics.from_session_metrics_list(0, ml).count == 10
        assert TurnPositionMetrics.from_session_metrics_list(3, ml).count == 5

    def test_missing_turn_index_returns_zero(self):
        ml = _make_metrics_list(n_sessions=5, n_turns=2)
        tpm = TurnPositionMetrics.from_session_metrics_list(99, ml)
        assert tpm.count == 0
        assert tpm.avg_ttft_ms == 0.0

    def test_percentile_ordering(self):
        ml = _make_metrics_list(n_sessions=100, n_turns=2)
        tpm = TurnPositionMetrics.from_session_metrics_list(0, ml)
        assert tpm.p50_ttft_ms <= tpm.p90_ttft_ms <= tpm.p99_ttft_ms
