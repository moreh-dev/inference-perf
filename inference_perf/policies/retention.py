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

"""Retention policy implementations for KV-cache retention directives.

Policies compute per-turn retention_directives that instruct the inference
server to protect specific token ranges in its KV-cache with explicit
priorities and TTLs. The directives are injected into the request via
the extra_body mechanism in ChatCompletionAPIData.
"""

import logging
from typing import Any, Optional, Protocol

from inference_perf.models import Turn, FinishReason

logger = logging.getLogger(__name__)


class RetentionPolicy(Protocol):
    """Protocol for retention policy implementations.

    A retention policy decides, per turn, whether to attach
    retention_directives to the inference request and what those
    directives should contain.
    """

    def compute_directives(
        self,
        turn: Turn,
        context_tokens: int,
        system_prompt_tokens: int,
        scope: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Compute extra_body dict with retention_directives, or None.

        Args:
            turn: The current turn being executed.
            context_tokens: Total accumulated input tokens (turn.input_tokens).
            system_prompt_tokens: Number of tokens in the system prompt.
            scope: Opaque scope identifier (e.g. session ID) for
                retention ownership.

        Returns:
            Dict suitable for ChatCompletionAPIData.extra_body
            (containing "retention_directives" and optionally
            "retention_scope" keys), or None if no directives should
            be sent for this turn.
        """
        ...


class NoopPolicy:
    """Baseline policy that never emits retention directives."""

    def compute_directives(
        self,
        turn: Turn,
        context_tokens: int,
        system_prompt_tokens: int,
        scope: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return None


class ContinuumPolicy:
    """Continuum-style differential retention by token region.

    Splits the context into three regions with decreasing priority:
      1. System prompt [0, system_prompt_tokens): highest priority, no TTL.
      2. Conversation history [system_prompt_tokens, context - tail): mid priority, no TTL.
      3. Recent tail [context - tail, context): lower priority, TTL-bounded.

    Emits directives on turns that end with tool_calls (the model is
    about to pause while tools execute, creating eviction risk).

    Emits a scoped clear (scope with empty directives) on the first turn
    after a tool_calls turn — the pause is over and the engine should
    release the retention held by this scope.
    """

    def __init__(
        self,
        system_prompt_priority: int = 90,
        history_priority: int = 70,
        tail_priority: int = 50,
        tail_fraction: float = 0.2,
        ttl_buffer_s: float = 5.0,
    ) -> None:
        self.system_prompt_priority = system_prompt_priority
        self.history_priority = history_priority
        self.tail_priority = tail_priority
        self.tail_fraction = tail_fraction
        self.ttl_buffer_s = ttl_buffer_s
        self._prev_was_tool_call: dict[str, bool] = {}

    def compute_directives(
        self,
        turn: Turn,
        context_tokens: int,
        system_prompt_tokens: int,
        scope: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        sid = turn.session_id
        is_tool_call = turn.finish_reason == FinishReason.TOOL_CALLS
        resuming = self._prev_was_tool_call.get(sid, False) and not is_tool_call
        self._prev_was_tool_call[sid] = is_tool_call

        # Resuming after a tool-call pause: emit a scoped clear so the
        # engine releases blocks owned by this scope.
        if resuming and scope is not None:
            logger.debug(
                "ContinuumPolicy: clearing retention scope=%s "
                "session=%s turn=%d",
                scope, turn.session_id, turn.turn_index,
            )
            return {"retention_scope": scope}

        if not is_tool_call:
            return None

        if context_tokens <= 0:
            return None

        directives: list[dict[str, Any]] = []

        # Compute tail boundary
        tail_size = max(1, int(context_tokens * self.tail_fraction))
        tail_start = max(system_prompt_tokens, context_tokens - tail_size)

        # Estimate TTL: tool execution time + buffer
        tool_duration_s = turn.total_tool_duration_ms / 1000.0
        ttl = tool_duration_s + self.ttl_buffer_s

        # Region 1: System prompt — highest priority, persist
        if system_prompt_tokens > 0:
            directives.append({
                "start": 0,
                "end": system_prompt_tokens,
                "priority": self.system_prompt_priority,
            })

        # Region 2: Conversation history — mid priority, persist
        history_end = tail_start
        if history_end > system_prompt_tokens:
            directives.append({
                "start": system_prompt_tokens,
                "end": history_end,
                "priority": self.history_priority,
            })

        # Region 3: Recent tail — lower priority, TTL-bounded
        if tail_start < context_tokens:
            directives.append({
                "start": tail_start,
                "end": context_tokens,
                "priority": self.tail_priority,
                "duration": ttl,
            })

        if not directives:
            return None

        result: dict[str, Any] = {"retention_directives": directives}
        if scope is not None:
            result["retention_scope"] = scope

        logger.debug(
            "ContinuumPolicy: session=%s turn=%d context=%d directives=%s",
            turn.session_id, turn.turn_index, context_tokens, directives,
        )

        return result
