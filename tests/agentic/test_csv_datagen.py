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

"""Tests for AgenticCsvDataGenerator.

Covers the real failure modes: file errors, schema validation,
tool-metric splitting, context growth computation, and row ordering.
"""

import pytest

from inference_perf.config import AgenticCsvConfig
from inference_perf.datagen.agentic_csv_datagen import AgenticCsvDataGenerator
from inference_perf.models.turn import FinishReason


SAMPLE_TRACE = """\
session_id,turn_index,input_tokens,output_tokens,finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms
sess_001,0,312,28,tool_calls,1,400,52,800
sess_001,1,392,94,stop,0,0,0,1100
sess_002,0,256,150,stop,0,0,0,900
sess_003,0,1024,42,tool_calls,2,350,120,1200
sess_003,1,1186,200,tool_calls,1,180,45,1500
sess_003,2,1431,88,stop,0,0,0,1100
"""


@pytest.fixture
def trace_csv(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text(SAMPLE_TRACE)
    return str(p)


class TestAgenticCsvDataGenerator:

    def test_loads_correct_session_count(self, trace_csv):
        sessions = AgenticCsvDataGenerator(AgenticCsvConfig(path=trace_csv)).get_sessions()
        assert len(sessions) == 3

    def test_tool_duration_split_across_calls(self, trace_csv):
        """sess_003 turn 0: 2 tools, 350ms total → 175ms each."""
        by_id = {s.session_id: s for s in
                 AgenticCsvDataGenerator(AgenticCsvConfig(path=trace_csv)).get_sessions()}
        t0 = by_id["sess_003"].turns[0]
        assert len(t0.tool_calls) == 2
        assert t0.tool_calls[0].duration_ms == 175
        assert t0.tool_calls[1].duration_ms == 175
        assert t0.total_tool_result_tokens == 120

    def test_context_growth_computed(self, trace_csv):
        by_id = {s.session_id: s for s in
                 AgenticCsvDataGenerator(AgenticCsvConfig(path=trace_csv)).get_sessions()}
        s3 = by_id["sess_003"]
        assert s3.turns[0].new_context_tokens == 0
        assert s3.turns[1].new_context_tokens == 1186 - 1024
        assert s3.turns[2].new_context_tokens == 1431 - 1186

    def test_unordered_rows_sorted(self, tmp_path):
        csv_data = (
            "session_id,turn_index,input_tokens,output_tokens,"
            "finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms\n"
            "s1,2,800,50,stop,0,0,0,500\n"
            "s1,0,200,50,tool_calls,1,100,30,500\n"
            "s1,1,500,50,tool_calls,1,80,20,500\n"
        )
        p = tmp_path / "unordered.csv"
        p.write_text(csv_data)
        s = AgenticCsvDataGenerator(AgenticCsvConfig(path=str(p))).get_sessions()[0]
        assert [t.turn_index for t in s.turns] == [0, 1, 2]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgenticCsvDataGenerator(AgenticCsvConfig(path=str(tmp_path / "nope.csv")))

    def test_missing_required_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("session_id,turn_index,input_tokens\ns1,0,100\n")
        with pytest.raises(ValueError, match="missing required columns"):
            AgenticCsvDataGenerator(AgenticCsvConfig(path=str(p)))

    def test_empty_csv_no_sessions(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text(
            "session_id,turn_index,input_tokens,output_tokens,"
            "finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms\n"
        )
        with pytest.raises(ValueError, match="No valid sessions"):
            AgenticCsvDataGenerator(AgenticCsvConfig(path=str(p)))

    def test_unknown_finish_reason(self, tmp_path):
        csv_data = (
            "session_id,turn_index,input_tokens,output_tokens,"
            "finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms\n"
            "s1,0,100,50,weird_reason,0,0,0,500\n"
        )
        p = tmp_path / "unknown_fr.csv"
        p.write_text(csv_data)
        s = AgenticCsvDataGenerator(AgenticCsvConfig(path=str(p))).get_sessions()[0]
        assert s.turns[0].finish_reason == FinishReason.UNKNOWN
