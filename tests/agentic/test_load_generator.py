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

"""Tests for AgenticLoadGenerator.

Covers the scheduling contract: constant arrival spacing,
Poisson rate convergence, and round-robin session selection.
"""

import numpy as np
import pytest

from unittest.mock import Mock

from inference_perf.config import (
    APIConfig,
    APIType,
    LoadConfig,
    LoadType,
    SessionArrivalConfig,
    SessionArrivalType,
    SessionStageConfig,
)
from inference_perf.loadgen.agentic_load_generator import AgenticLoadGenerator
from inference_perf.models import Session, Turn
from inference_perf.models.turn import FinishReason


def _make_sessions(n=3):
    sessions = []
    for i in range(n):
        sessions.append(Session(session_id=f"sess_{i}", turns=[
            Turn(session_id=f"sess_{i}", turn_index=0,
                 input_tokens=300, output_tokens=80, finish_reason=FinishReason.STOP),
        ]))
    return sessions


def _make_generator(sessions=None, rate=10.0, total_sessions=10):
    sessions = sessions or _make_sessions()
    config = LoadConfig(
        type=LoadType.AGENTIC,
        session_arrival=SessionArrivalConfig(
            type=SessionArrivalType.CONSTANT,
            stages=[SessionStageConfig(rate=rate, duration=5, total_sessions=total_sessions)],
        ),
    )
    gen = AgenticLoadGenerator(
        sessions=sessions,
        load_config=config,
        api_config=APIConfig(type=APIType.Chat),
        content_generator=Mock(generate=Mock(return_value="x")),
    )
    return gen


class TestArrivalTimes:

    def test_constant_spacing(self):
        gen = _make_generator(rate=5.0, total_sessions=10)
        times = gen._generate_arrival_times(10, 5.0, poisson=False)
        assert len(times) == 10
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(0.2)
        assert times[9] == pytest.approx(1.8)

    def test_poisson_rate_convergence(self):
        """Average inter-arrival should converge to 1/rate."""
        np.random.seed(42)
        gen = _make_generator(rate=20.0, total_sessions=5000)
        times = gen._generate_arrival_times(5000, 20.0, poisson=True)
        inter = [times[i] - times[i - 1] for i in range(1, len(times))]
        avg = sum(inter) / len(inter)
        assert abs(avg - 0.05) < 0.005

    def test_poisson_monotonically_increasing(self):
        gen = _make_generator()
        times = gen._generate_arrival_times(100, 10.0, poisson=True)
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]


class TestRoundRobin:

    def test_cycles_through_sessions(self):
        gen = _make_generator()
        picked = [gen._get_next_session().session_id for _ in range(7)]
        assert picked == [
            "sess_0", "sess_1", "sess_2",
            "sess_0", "sess_1", "sess_2",
            "sess_0",
        ]
