"""AgenticSyntheticDataGenerator seed reproducibility.

Two generators created with the same seed must produce sessions with identical
turn structure (turn count, input/output/tool tokens, finish reason). UUIDs
differ between runs by design and are excluded from the comparison — only the
workload-affecting fields matter for paired benchmark analysis.
"""

from __future__ import annotations

from typing import Any

from inference_perf.config import AgenticSyntheticConfig
from inference_perf.datagen.agentic_synthetic_datagen import AgenticSyntheticDataGenerator


def _signature(generator: AgenticSyntheticDataGenerator) -> list[list[tuple[Any, ...]]]:
    """Reduce a generator's sessions to comparable workload signatures.

    UUID session_id and runtime fields (latency, ttft, timestamp) are excluded.
    Only fields that affect the request payload to vLLM matter for paired
    benchmark analysis.
    """
    out: list[list[tuple[Any, ...]]] = []
    for session in generator.get_sessions():
        turn_sig = []
        for t in session.turns:
            turn_sig.append(
                (
                    t.input_tokens,
                    t.output_tokens,
                    t.new_context_tokens,
                    t.finish_reason,
                    len(t.tool_calls),
                )
            )
        out.append(turn_sig)
    return out


def _config(seed: int | None) -> AgenticSyntheticConfig:
    return AgenticSyntheticConfig(num_sessions=10, seed=seed)


def test_same_seed_yields_same_sessions() -> None:
    a = AgenticSyntheticDataGenerator(_config(seed=42))
    b = AgenticSyntheticDataGenerator(_config(seed=42))
    assert _signature(a) == _signature(b)


def test_different_seed_yields_different_sessions() -> None:
    a = AgenticSyntheticDataGenerator(_config(seed=42))
    b = AgenticSyntheticDataGenerator(_config(seed=43))
    assert _signature(a) != _signature(b)


def test_no_seed_is_nondeterministic() -> None:
    """When seed is None, two runs MAY produce different sessions.

    We can't strictly assert inequality (random chance of collision is nonzero
    over 10 sessions), but we verify the API accepts seed=None and produces
    well-formed sessions.
    """
    g = AgenticSyntheticDataGenerator(_config(seed=None))
    assert len(g.get_sessions()) == 10
