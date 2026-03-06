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

"""Tests for SessionRunner.

Covers the three behaviors that matter for platform evaluation:
1. Context accumulation — correct message roles and growing history
2. Turn failure handling — session stops cleanly
3. Delay routing — tool_call turns vs stop turns pick the right config
"""

from unittest.mock import AsyncMock, Mock

import pytest

from inference_perf.apis import ChatCompletionAPIData
from inference_perf.config import (
    AgenticDelayConfig,
    APIConfig,
    APIType,
    DelayConfig,
    DelayType,
    DistributionParams,
)
from inference_perf.loadgen.session_runner import SessionRunner
from inference_perf.models import Session, Turn, ToolCall
from inference_perf.models.turn import FinishReason


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mock_content_generator():
    gen = Mock()
    gen.generate = Mock(side_effect=lambda n: "x" * max(n, 0))
    return gen


def _mock_client():
    client = Mock()
    client.process_request = AsyncMock()
    return client


def _make_session(turn_specs):
    """Build session from list of (input_tokens, output_tokens, finish_reason, tool_calls)."""
    turns, prev = [], 0
    for i, (inp, out, fr, tc) in enumerate(turn_specs):
        turns.append(Turn(
            session_id="s1", turn_index=i,
            input_tokens=inp, output_tokens=out,
            new_context_tokens=max(0, inp - prev) if i > 0 else 0,
            finish_reason=fr, tool_calls=tc, llm_latency_ms=500,
        ))
        prev = inp
    return Session(session_id="s1", turns=turns)


def _runner(session, client=None, delay_config=None):
    return SessionRunner(
        session=session,
        client=client or _mock_client(),
        api_config=APIConfig(type=APIType.Chat),
        content_generator=_mock_content_generator(),
        delay_config=delay_config,
    )


# ---------------------------------------------------------------------------
# Context accumulation
# ---------------------------------------------------------------------------


class TestContextAccumulation:

    @pytest.mark.asyncio
    async def test_message_roles_with_tools(self):
        """Turn with tool_calls → user, assistant, tool messages added to context."""
        session = _make_session([
            (300, 50, FinishReason.TOOL_CALLS, [ToolCall(name="t0", duration_ms=100, result_tokens=30)]),
            (500, 80, FinishReason.STOP, []),
        ])
        r = _runner(session)
        await r.run()
        roles = [m.role for m in r.context]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_context_grows_across_turns(self):
        """Each turn's request should include all prior context."""
        client = _mock_client()
        session = _make_session([
            (300, 50, FinishReason.STOP, []),
            (500, 80, FinishReason.STOP, []),
            (800, 60, FinishReason.STOP, []),
        ])
        r = _runner(session, client=client)
        await r.run()

        msg_counts = [
            len(call[0][0].messages)
            for call in client.process_request.call_args_list
        ]
        # Turn 0: 1 msg, Turn 1: 3 msgs (user+asst+user), Turn 2: 5 msgs
        assert msg_counts == [1, 3, 5]

    @pytest.mark.asyncio
    async def test_program_id_and_turn_index(self):
        client = _mock_client()
        session = _make_session([(300, 50, FinishReason.STOP, [])])
        r = _runner(session, client=client)
        await r.run()

        req = client.process_request.call_args[0][0]
        assert req.program_id == "s1"
        assert req.turn_index == 0


# ---------------------------------------------------------------------------
# Turn failure
# ---------------------------------------------------------------------------


class TestTurnFailure:

    @pytest.mark.asyncio
    async def test_failed_turn_stops_session(self):
        client = _mock_client()
        client.process_request.side_effect = [None, Exception("timeout"), None]

        session = _make_session([
            (300, 50, FinishReason.STOP, []),
            (500, 80, FinishReason.STOP, []),
            (800, 60, FinishReason.STOP, []),
        ])
        result = await _runner(session, client=client).run()

        assert len(result.turn_results) == 2
        assert result.turn_results[0].success
        assert not result.turn_results[1].success
        assert result.turns_completed == 1


# ---------------------------------------------------------------------------
# Delay routing
# ---------------------------------------------------------------------------


class TestDelayRouting:

    def test_tool_turn_uses_tool_delay(self):
        dc = AgenticDelayConfig(
            tool_call_delay=DelayConfig(type=DelayType.FIXED, fixed_ms=100),
            user_think_delay=DelayConfig(type=DelayType.FIXED, fixed_ms=999),
        )
        session = _make_session([
            (300, 50, FinishReason.TOOL_CALLS, [ToolCall(name="t", duration_ms=400, result_tokens=30)]),
        ])
        r = _runner(session, delay_config=dc)
        assert r._compute_delay(dc.tool_call_delay, session.turns[0]) == 100.0

    def test_stop_turn_uses_user_delay(self):
        dc = AgenticDelayConfig(
            tool_call_delay=DelayConfig(type=DelayType.FIXED, fixed_ms=999),
            user_think_delay=DelayConfig(type=DelayType.FIXED, fixed_ms=500),
        )
        session = _make_session([(300, 50, FinishReason.STOP, [])])
        r = _runner(session, delay_config=dc)
        assert r._compute_delay(dc.user_think_delay, session.turns[0]) == 500.0

    def test_replay_delay_uses_tool_duration(self):
        dc = AgenticDelayConfig(
            tool_call_delay=DelayConfig(type=DelayType.REPLAY),
        )
        session = _make_session([
            (300, 50, FinishReason.TOOL_CALLS, [ToolCall(name="t", duration_ms=400, result_tokens=30)]),
        ])
        r = _runner(session, delay_config=dc)
        assert r._compute_delay(dc.tool_call_delay, session.turns[0]) == 400.0

    def test_zero_delay(self):
        dc = AgenticDelayConfig(tool_call_delay=DelayConfig(type=DelayType.ZERO))
        session = _make_session([(300, 50, FinishReason.STOP, [])])
        r = _runner(session, delay_config=dc)
        assert r._compute_delay(dc.tool_call_delay, session.turns[0]) == 0.0
