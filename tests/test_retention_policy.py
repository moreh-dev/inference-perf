"""Tests for KV-cache retention policy implementations."""

from inference_perf.models import Turn, ToolCall, FinishReason
from inference_perf.policies.retention import ContinuumPolicy, NoopPolicy


def _make_turn(
    finish_reason: FinishReason = FinishReason.TOOL_CALLS,
    input_tokens: int = 10000,
    output_tokens: int = 200,
    tool_duration_ms: int = 5000,
    turn_index: int = 3,
) -> Turn:
    """Helper to create a Turn for testing."""
    tool_calls = []
    if finish_reason == FinishReason.TOOL_CALLS:
        tool_calls = [ToolCall(
            name="search",
            duration_ms=tool_duration_ms,
            result_tokens=500,
            tool_call_id="tc_test",
        )]
    return Turn(
        session_id="test-session",
        turn_index=turn_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        new_context_tokens=700,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


class TestNoopPolicy:
    def test_always_returns_none(self) -> None:
        policy = NoopPolicy()
        turn = _make_turn(FinishReason.TOOL_CALLS)
        assert policy.compute_directives(turn, 10000, 1024) is None

    def test_returns_none_on_stop(self) -> None:
        policy = NoopPolicy()
        turn = _make_turn(FinishReason.STOP)
        assert policy.compute_directives(turn, 10000, 1024) is None

    def test_returns_none_with_scope(self) -> None:
        policy = NoopPolicy()
        turn = _make_turn(FinishReason.TOOL_CALLS)
        assert policy.compute_directives(turn, 10000, 1024, scope="s1") is None


class TestContinuumPolicy:
    def test_emits_directives_on_tool_calls(self) -> None:
        policy = ContinuumPolicy()
        turn = _make_turn(FinishReason.TOOL_CALLS)
        result = policy.compute_directives(turn, 10000, 1024, scope="s1")

        assert result is not None
        assert "retention_directives" in result
        assert result["retention_scope"] == "s1"
        directives = result["retention_directives"]
        assert len(directives) == 3  # system + history + tail

    def test_no_directives_on_stop_without_prior_tool_call(self) -> None:
        """Stop turn without a preceding tool_call → no directives."""
        policy = ContinuumPolicy()
        turn = _make_turn(FinishReason.STOP)
        result = policy.compute_directives(turn, 10000, 1024, scope="s1")
        assert result is None

    def test_no_directives_on_length(self) -> None:
        policy = ContinuumPolicy()
        turn = _make_turn(FinishReason.LENGTH)
        result = policy.compute_directives(turn, 10000, 1024)
        assert result is None

    def test_system_prompt_region(self) -> None:
        policy = ContinuumPolicy(system_prompt_priority=90)
        turn = _make_turn()
        result = policy.compute_directives(turn, 10000, 1024)

        directives = result["retention_directives"]
        system_dir = directives[0]
        assert system_dir["start"] == 0
        assert system_dir["end"] == 1024
        assert system_dir["priority"] == 90
        assert "duration" not in system_dir  # persistent

    def test_history_region(self) -> None:
        policy = ContinuumPolicy(
            history_priority=70,
            tail_fraction=0.2,
        )
        turn = _make_turn(input_tokens=10000)
        result = policy.compute_directives(turn, 10000, 1024)

        directives = result["retention_directives"]
        history_dir = directives[1]
        assert history_dir["start"] == 1024
        # tail_start = max(1024, 10000 - 2000) = 8000
        assert history_dir["end"] == 8000
        assert history_dir["priority"] == 70
        assert "duration" not in history_dir  # persistent

    def test_tail_region_with_ttl(self) -> None:
        policy = ContinuumPolicy(
            tail_priority=50,
            tail_fraction=0.2,
            ttl_buffer_s=5.0,
        )
        turn = _make_turn(input_tokens=10000, tool_duration_ms=3000)
        result = policy.compute_directives(turn, 10000, 1024)

        directives = result["retention_directives"]
        tail_dir = directives[2]
        assert tail_dir["start"] == 8000
        assert tail_dir["end"] == 10000
        assert tail_dir["priority"] == 50
        # TTL = tool_duration (3s) + buffer (5s) = 8s
        assert tail_dir["duration"] == 8.0

    def test_custom_priorities(self) -> None:
        policy = ContinuumPolicy(
            system_prompt_priority=95,
            history_priority=60,
            tail_priority=30,
        )
        turn = _make_turn()
        result = policy.compute_directives(turn, 10000, 1024)

        directives = result["retention_directives"]
        assert directives[0]["priority"] == 95
        assert directives[1]["priority"] == 60
        assert directives[2]["priority"] == 30

    def test_no_system_prompt(self) -> None:
        """When system_prompt_tokens=0, only history + tail regions."""
        policy = ContinuumPolicy()
        turn = _make_turn(input_tokens=5000)
        result = policy.compute_directives(turn, 5000, 0)

        directives = result["retention_directives"]
        # No system prompt region, just history + tail
        assert len(directives) == 2
        assert directives[0]["start"] == 0  # history starts at 0
        assert directives[1]["end"] == 5000  # tail goes to end

    def test_small_context_merges_regions(self) -> None:
        """When context is small, tail_start == system_prompt_tokens."""
        policy = ContinuumPolicy(tail_fraction=0.5)
        # context=2000, system_prompt=1024, tail=1000
        # tail_start = max(1024, 2000 - 1000) = 1024
        turn = _make_turn(input_tokens=2000)
        result = policy.compute_directives(turn, 2000, 1024)

        directives = result["retention_directives"]
        # System [0,1024) and tail [1024,2000) — no history region
        assert len(directives) == 2
        assert directives[0]["end"] == 1024  # system
        assert directives[1]["start"] == 1024  # tail (no gap)
        assert directives[1]["end"] == 2000

    def test_zero_context_returns_none(self) -> None:
        policy = ContinuumPolicy()
        turn = _make_turn(input_tokens=0)
        result = policy.compute_directives(turn, 0, 0)
        assert result is None

    def test_directive_structure_valid(self) -> None:
        """Directives should form non-overlapping, contiguous ranges."""
        policy = ContinuumPolicy()
        turn = _make_turn(input_tokens=20000)
        result = policy.compute_directives(turn, 20000, 1024)

        directives = result["retention_directives"]
        # Check contiguous: each starts where previous ends
        for i in range(1, len(directives)):
            assert directives[i]["start"] == directives[i - 1]["end"]

        # Check bounds
        assert directives[0]["start"] == 0
        assert directives[-1]["end"] == 20000

    # --- Scoped clear-on-resume tests ---

    def test_clear_on_resume_after_tool_call(self) -> None:
        """After tool_calls turn, next non-tool turn emits a scoped clear."""
        policy = ContinuumPolicy()
        scope = "session-abc"

        # Turn 0: tool_calls → directives emitted
        t0 = _make_turn(FinishReason.TOOL_CALLS, turn_index=0)
        r0 = policy.compute_directives(t0, 10000, 1024, scope=scope)
        assert r0 is not None
        assert "retention_directives" in r0

        # Turn 1: stop (resuming after tool call) → scoped clear
        t1 = _make_turn(FinishReason.STOP, turn_index=1)
        r1 = policy.compute_directives(t1, 11000, 1024, scope=scope)
        assert r1 is not None
        assert r1["retention_scope"] == scope
        assert "retention_directives" not in r1

    def test_no_clear_on_consecutive_tool_calls(self) -> None:
        """Consecutive tool_calls turns emit directives, not clears."""
        policy = ContinuumPolicy()
        scope = "session-abc"

        t0 = _make_turn(FinishReason.TOOL_CALLS, turn_index=0)
        r0 = policy.compute_directives(t0, 10000, 1024, scope=scope)
        assert "retention_directives" in r0

        t1 = _make_turn(FinishReason.TOOL_CALLS, turn_index=1)
        r1 = policy.compute_directives(t1, 11000, 1024, scope=scope)
        assert "retention_directives" in r1

    def test_no_clear_without_preceding_tool_call(self) -> None:
        """Stop turn without preceding tool_call → None (no clear needed)."""
        policy = ContinuumPolicy()
        scope = "session-abc"

        t0 = _make_turn(FinishReason.STOP, turn_index=0)
        r0 = policy.compute_directives(t0, 10000, 1024, scope=scope)
        assert r0 is None

    def test_clear_requires_scope(self) -> None:
        """Without a scope, resume after tool_call emits nothing."""
        policy = ContinuumPolicy()

        t0 = _make_turn(FinishReason.TOOL_CALLS, turn_index=0)
        policy.compute_directives(t0, 10000, 1024)

        t1 = _make_turn(FinishReason.STOP, turn_index=1)
        r1 = policy.compute_directives(t1, 11000, 1024)
        assert r1 is None

    def test_scope_included_in_tool_call_directives(self) -> None:
        """When scope is provided, tool_calls directives include it."""
        policy = ContinuumPolicy()
        turn = _make_turn(FinishReason.TOOL_CALLS)
        result = policy.compute_directives(turn, 10000, 1024, scope="s1")
        assert result["retention_scope"] == "s1"

    def test_scope_omitted_when_none(self) -> None:
        """When scope is None, directives don't include retention_scope."""
        policy = ContinuumPolicy()
        turn = _make_turn(FinishReason.TOOL_CALLS)
        result = policy.compute_directives(turn, 10000, 1024, scope=None)
        assert "retention_scope" not in result

    def test_cross_session_isolation(self) -> None:
        """tool_call state from one session must not leak to another."""
        policy = ContinuumPolicy()

        # Session A turn 0: tool_calls
        ta0 = Turn(
            session_id="A", turn_index=0,
            input_tokens=1000, output_tokens=50,
            new_context_tokens=100,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[ToolCall(
                name="t", duration_ms=100,
                result_tokens=10, tool_call_id="tc1",
            )],
        )
        ra0 = policy.compute_directives(ta0, 1000, 0, scope="A")
        assert ra0 is not None
        assert "retention_directives" in ra0

        # Session B turn 0: stop — must NOT get a spurious clear
        tb0 = Turn(
            session_id="B", turn_index=0,
            input_tokens=500, output_tokens=80,
            new_context_tokens=80,
            finish_reason=FinishReason.STOP,
            tool_calls=[],
        )
        rb0 = policy.compute_directives(tb0, 500, 0, scope="B")
        assert rb0 is None  # no clear, no directives
