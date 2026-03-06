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

"""Session runner for agentic workload execution.

The SessionRunner manages the execution of a single agentic session,
handling context accumulation, inter-turn delays, and result collection.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, InferenceAPIData
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import AgenticDelayConfig, DelayConfig, DelayType, APIConfig
from inference_perf.models import Session, Turn, FinishReason
from inference_perf.policies.retention import RetentionPolicy
from inference_perf.utils.content_generator import ContentGenerator

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a single turn execution."""
    session_id: str
    turn_index: int
    scheduled_time: float
    start_time: float
    end_time: float
    input_tokens: int
    output_tokens: int
    ttft_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SessionResult:
    """Result of a complete session execution."""
    session_id: str
    turn_results: List[TurnResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def session_latency_ms(self) -> float:
        """Total wall-clock time including pauses."""
        return (self.end_time - self.start_time) * 1000

    @property
    def session_inference_time_ms(self) -> float:
        """Sum of LLM call durations (excludes pauses)."""
        return sum(
            (tr.end_time - tr.start_time) * 1000
            for tr in self.turn_results
        )

    @property
    def session_pause_time_ms(self) -> float:
        """Total pause time between turns."""
        return self.session_latency_ms - self.session_inference_time_ms

    @property
    def inference_duty_cycle(self) -> float:
        """Ratio of inference time to total latency."""
        if self.session_latency_ms == 0:
            return 0.0
        return self.session_inference_time_ms / self.session_latency_ms

    @property
    def turns_completed(self) -> int:
        """Number of successfully completed turns."""
        return sum(1 for tr in self.turn_results if tr.success)

    @property
    def session_input_tokens(self) -> int:
        """Total input tokens across all turns."""
        return sum(tr.input_tokens for tr in self.turn_results)

    @property
    def session_output_tokens(self) -> int:
        """Total output tokens across all turns."""
        return sum(tr.output_tokens for tr in self.turn_results)

    @property
    def peak_context_length(self) -> int:
        """Maximum input tokens in any single turn."""
        if not self.turn_results:
            return 0
        return max(tr.input_tokens for tr in self.turn_results)


class SessionRunner:
    """Manages execution of a single agentic session.

    The SessionRunner iterates through turns in a session, building
    requests with accumulated context, applying inter-turn delays,
    and collecting results. Uses ContentGenerator for token-accurate
    content in context accumulation.
    """

    def __init__(
        self,
        session: Session,
        client: ModelServerClient,
        api_config: APIConfig,
        content_generator: ContentGenerator,
        delay_config: Optional[AgenticDelayConfig] = None,
        stage_id: int = 0,
        retention_policy: Optional[RetentionPolicy] = None,
        system_prompt_tokens: int = 0,
    ):
        self.session = session
        self.client = client
        self.api_config = api_config
        self.content_generator = content_generator
        self.delay_config = delay_config or AgenticDelayConfig()
        self.stage_id = stage_id
        self.retention_policy = retention_policy
        self.system_prompt_tokens = system_prompt_tokens

        # Context accumulation
        self.context: List[ChatMessage] = []
        self.results: List[TurnResult] = []

    async def run(self) -> SessionResult:
        """Execute all turns in the session."""
        session_result = SessionResult(session_id=self.session.session_id)
        session_result.start_time = time.perf_counter()

        for turn in self.session.turns:
            try:
                turn_result = await self._execute_turn(turn)
                self.results.append(turn_result)
                session_result.turn_results.append(turn_result)

                if not turn_result.success:
                    logger.warning(
                        f"Session {self.session.session_id} turn {turn.turn_index} failed: "
                        f"{turn_result.error_message}"
                    )
                    break

                # Apply inter-turn delay if not the last turn
                if turn.turn_index < len(self.session.turns) - 1:
                    await self._apply_inter_turn_delay(turn)

            except Exception as e:
                logger.error(
                    f"Session {self.session.session_id} turn {turn.turn_index} exception: {e}"
                )
                error_result = TurnResult(
                    session_id=self.session.session_id,
                    turn_index=turn.turn_index,
                    scheduled_time=time.perf_counter(),
                    start_time=time.perf_counter(),
                    end_time=time.perf_counter(),
                    input_tokens=turn.input_tokens,
                    output_tokens=0,
                    success=False,
                    error_message=str(e),
                )
                session_result.turn_results.append(error_result)
                break

        session_result.end_time = time.perf_counter()

        logger.debug(
            f"Session {self.session.session_id} completed: "
            f"{session_result.turns_completed}/{len(self.session.turns)} turns, "
            f"{session_result.session_latency_ms:.0f}ms total"
        )

        return session_result

    async def _execute_turn(self, turn: Turn) -> TurnResult:
        """Execute a single turn."""
        scheduled_time = time.perf_counter()

        # Build request with accumulated context + new user message
        request = self._build_request(turn)

        start_time = time.perf_counter()

        try:
            response_info = await self.client.process_request(
                request,
                self.stage_id,
                scheduled_time,
            )

            end_time = time.perf_counter()

            # Extract TTFT from response output_token_times
            ttft_ms = None
            if response_info and response_info.output_token_times:
                ttft_ms = (response_info.output_token_times[0] - start_time) * 1000

            # Update context with synthetic response matching token counts
            self._update_context(turn)

            return TurnResult(
                session_id=self.session.session_id,
                turn_index=turn.turn_index,
                scheduled_time=scheduled_time,
                start_time=start_time,
                end_time=end_time,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                ttft_ms=ttft_ms,
                success=True,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return TurnResult(
                session_id=self.session.session_id,
                turn_index=turn.turn_index,
                scheduled_time=scheduled_time,
                start_time=start_time,
                end_time=end_time,
                input_tokens=turn.input_tokens,
                output_tokens=0,
                success=False,
                error_message=str(e),
            )

    def _build_request(self, turn: Turn) -> InferenceAPIData:
        """Build an inference request for a turn.

        Constructs a ChatCompletionAPIData with the accumulated context
        plus a new user message with token-accurate generated content.
        """
        messages: List[ChatMessage] = []

        # Add accumulated context from prior turns
        messages.extend(self.context)

        # Add new user message for this turn with content matching
        # the token count delta (new tokens introduced in this turn)
        new_tokens = turn.new_context_tokens if turn.new_context_tokens > 0 else turn.input_tokens
        user_content = self.content_generator.generate(new_tokens)
        messages.append(ChatMessage(role="user", content=user_content))

        request = ChatCompletionAPIData(
            messages=messages,
            program_id=self.session.session_id,
            turn_index=turn.turn_index,
        )

        # Inject retention directives if policy is configured
        if self.retention_policy is not None:
            extra = self.retention_policy.compute_directives(
                turn=turn,
                context_tokens=turn.input_tokens,
                system_prompt_tokens=self.system_prompt_tokens,
            )
            if extra:
                request.extra_body = extra

        return request

    def _update_context(self, turn: Turn) -> None:
        """Update context with turn results for the next turn.

        Adds the user message (from build), a synthetic assistant
        response matching output_tokens, and tool result messages
        matching tool_result_tokens. All content is generated via
        ContentGenerator for token accuracy.
        """
        # Add the user message we just sent
        new_tokens = turn.new_context_tokens if turn.new_context_tokens > 0 else turn.input_tokens
        user_content = self.content_generator.generate(new_tokens)
        self.context.append(ChatMessage(role="user", content=user_content))

        # Add synthetic assistant response matching output_tokens
        assistant_content = self.content_generator.generate(turn.output_tokens)
        self.context.append(ChatMessage(role="assistant", content=assistant_content))

        # Add tool result messages if turn had tool calls
        if turn.has_tool_calls:
            for tool_call in turn.tool_calls:
                tool_content = self.content_generator.generate(tool_call.result_tokens)
                self.context.append(ChatMessage(
                    role="tool",
                    content=tool_content,
                    id=tool_call.tool_call_id,
                ))

    async def _apply_inter_turn_delay(self, turn: Turn) -> None:
        """Apply delay between turns."""
        if turn.finish_reason == FinishReason.TOOL_CALLS:
            delay_ms = self._compute_delay(
                self.delay_config.tool_call_delay,
                turn
            )
        else:
            delay_ms = self._compute_delay(
                self.delay_config.user_think_delay,
                turn
            )

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

    def _compute_delay(self, config: DelayConfig, turn: Turn) -> float:
        """Compute delay in milliseconds based on configuration."""
        if config.type == DelayType.ZERO:
            return 0.0

        elif config.type == DelayType.FIXED:
            return float(config.fixed_ms or 0)

        elif config.type == DelayType.REPLAY:
            return float(turn.total_tool_duration_ms)

        elif config.type == DelayType.DISTRIBUTION:
            if config.distribution is None:
                return 0.0

            dist = config.distribution
            if dist.type == "normal":
                value = random.gauss(dist.mean, dist.std_dev)
                return max(dist.min, min(dist.max, value))
            elif dist.type == "uniform":
                return random.uniform(dist.min, dist.max)
            elif dist.type == "constant":
                return dist.value or 0.0
            else:
                return 0.0

        return 0.0
