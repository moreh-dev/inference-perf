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

"""Agentic CSV data generator.

Loads agentic sessions from CSV files containing turn-level data.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from inference_perf.config import AgenticCsvConfig
from inference_perf.models import Session, Turn, ToolCall, FinishReason

logger = logging.getLogger(__name__)


# Expected CSV columns
CSV_COLUMNS = [
    "session_id",
    "turn_index",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "num_tool_calls",
    "tool_duration_ms",
    "tool_result_tokens",
    "llm_latency_ms",
]

# Optional CSV columns
OPTIONAL_COLUMNS = [
    "ttft_ms",
    "timestamp_ms",
]


class AgenticCsvDataGenerator:
    """Load agentic sessions from CSV file.

    CSV Format:
    session_id,turn_index,input_tokens,output_tokens,finish_reason,
    num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms

    Optional columns:
    ttft_ms,timestamp_ms
    """

    def __init__(self, config: AgenticCsvConfig) -> None:
        self.csv_config = config
        self.sessions: List[Session] = []

        self._load_csv()

        if not self.sessions:
            raise ValueError(f"No valid sessions found in CSV file: {self.csv_config.path}")

        logger.info(
            f"Loaded {len(self.sessions)} sessions from CSV "
            f"with average {sum(s.num_turns for s in self.sessions) / len(self.sessions):.1f} turns"
        )

    def _load_csv(self) -> None:
        """Load and parse the CSV file into Session objects."""
        csv_path = Path(self.csv_config.path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV and group by session_id
        session_data: Dict[str, List[dict]] = {}

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV file has no header row")

            missing_columns = set(CSV_COLUMNS) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")

            for row in reader:
                session_id = row['session_id']
                if session_id not in session_data:
                    session_data[session_id] = []
                session_data[session_id].append(row)

        # Convert to Session objects
        for session_id, rows in session_data.items():
            try:
                session = self._build_session(session_id, rows)
                self.sessions.append(session)
            except Exception as e:
                logger.warning(f"Failed to parse session {session_id}: {e}")

    def _build_session(self, session_id: str, rows: List[dict]) -> Session:
        """Build a Session object from CSV rows."""
        rows.sort(key=lambda r: int(r['turn_index']))

        turns: List[Turn] = []
        prev_input_tokens = 0

        for row in rows:
            turn = self._build_turn(session_id, row, prev_input_tokens)
            turns.append(turn)
            prev_input_tokens = turn.input_tokens

        # Get earliest timestamp for session start time
        original_start_time_ms = None
        if turns and turns[0].timestamp_ms is not None:
            original_start_time_ms = turns[0].timestamp_ms

        return Session(
            session_id=session_id,
            turns=turns,
            original_start_time_ms=original_start_time_ms,
        )

    def _build_turn(
        self,
        session_id: str,
        row: dict,
        prev_input_tokens: int,
    ) -> Turn:
        """Build a Turn object from a CSV row."""
        turn_index = int(row['turn_index'])
        input_tokens = int(row['input_tokens'])
        output_tokens = int(row['output_tokens'])

        new_context_tokens = max(0, input_tokens - prev_input_tokens) if turn_index > 0 else 0

        finish_reason_str = row.get('finish_reason', 'stop')
        try:
            finish_reason = FinishReason(finish_reason_str)
        except ValueError:
            finish_reason = FinishReason.UNKNOWN

        tool_calls = self._build_tool_calls(row)

        llm_latency_ms = self._parse_optional_int(row.get('llm_latency_ms'))
        ttft_ms = self._parse_optional_int(row.get('ttft_ms'))
        timestamp_ms = self._parse_optional_int(row.get('timestamp_ms'))

        return Turn(
            session_id=session_id,
            turn_index=turn_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            new_context_tokens=new_context_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            llm_latency_ms=llm_latency_ms,
            ttft_ms=ttft_ms,
            timestamp_ms=timestamp_ms,
        )

    def _build_tool_calls(self, row: dict) -> List[ToolCall]:
        """Build tool call list from CSV row."""
        num_tool_calls = int(row.get('num_tool_calls', 0))
        if num_tool_calls == 0:
            return []

        total_duration = int(row.get('tool_duration_ms', 0))
        total_result_tokens = int(row.get('tool_result_tokens', 0))

        tool_calls = []
        for i in range(num_tool_calls):
            tool_calls.append(ToolCall(
                name=f"tool_{i}",
                duration_ms=total_duration // num_tool_calls,
                result_tokens=total_result_tokens // num_tool_calls,
            ))

        return tool_calls

    def _parse_optional_int(self, value: Optional[str]) -> Optional[int]:
        """Parse an optional integer value."""
        if value is None or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def get_sessions(self) -> List[Session]:
        """Get all loaded sessions."""
        return self.sessions
