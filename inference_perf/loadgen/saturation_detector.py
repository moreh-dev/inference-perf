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

"""Saturation detection for agentic workloads.

Detects the saturation point of an inference system under agentic workloads
by monitoring session inference time degradation as session arrival rate increases.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from inference_perf.config import AgenticSweepConfig, StageGenType

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single probe stage at a specific session arrival rate."""
    rate: float
    session_inference_times_ms: List[float] = field(default_factory=list)
    completed_sessions: int = 0
    total_sessions: int = 0

    @property
    def p50(self) -> float:
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 50))

    @property
    def p90(self) -> float:
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 90))

    @property
    def p95(self) -> float:
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 95))

    @property
    def p99(self) -> float:
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 99))

    @property
    def mean(self) -> float:
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.mean(self.session_inference_times_ms))

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "completed_sessions": self.completed_sessions,
            "total_sessions": self.total_sessions,
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p90_ms": self.p90,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
        }


@dataclass
class SaturationResult:
    """Result of saturation detection."""
    saturation_rate: float
    probe_results: List[ProbeResult]
    baseline_p95: float
    saturation_p95: float
    degradation_detected: bool

    def to_dict(self) -> dict:
        return {
            "saturation_rate": self.saturation_rate,
            "baseline_p95_ms": self.baseline_p95,
            "saturation_p95_ms": self.saturation_p95,
            "degradation_detected": self.degradation_detected,
            "probe_results": [pr.to_dict() for pr in self.probe_results],
        }


class SaturationDetector:
    """Detects saturation point for agentic workloads.

    Runs probe stages at increasing session arrival rates and monitors
    session_inference_time_p95 for degradation beyond the configured threshold.
    """

    def __init__(self, config: AgenticSweepConfig):
        self.config = config
        self.probe_results: List[ProbeResult] = []

    def get_probe_rates(self) -> List[float]:
        """Generate probe rates to test."""
        if self.config.probe_rates:
            return sorted(self.config.probe_rates)

        if self.config.type == StageGenType.LINEAR:
            rates = np.linspace(
                self.config.min_probe_rate,
                self.config.max_probe_rate,
                self.config.num_probes
            )
        else:  # GEOMETRIC
            rates = np.geomspace(
                self.config.min_probe_rate,
                self.config.max_probe_rate,
                self.config.num_probes
            )

        return [float(r) for r in rates]

    def add_probe_result(
        self,
        rate: float,
        inference_times_ms: List[float],
        completed_sessions: int,
        total_sessions: int,
    ) -> None:
        """Add results from a probe stage."""
        probe = ProbeResult(
            rate=rate,
            session_inference_times_ms=inference_times_ms,
            completed_sessions=completed_sessions,
            total_sessions=total_sessions,
        )
        self.probe_results.append(probe)

        logger.info(
            f"Probe at rate={rate:.1f}: {completed_sessions}/{total_sessions} sessions, "
            f"p95={probe.p95:.0f}ms, mean={probe.mean:.0f}ms"
        )

    def detect_saturation(self) -> SaturationResult:
        """Analyze probe results to detect saturation point."""
        if not self.probe_results:
            raise ValueError("No probe results to analyze")

        sorted_probes = sorted(self.probe_results, key=lambda p: p.rate)

        baseline_p95 = 0.0
        for probe in sorted_probes:
            if probe.p95 > 0:
                baseline_p95 = probe.p95
                break

        if baseline_p95 <= 0:
            return SaturationResult(
                saturation_rate=sorted_probes[-1].rate,
                probe_results=self.probe_results,
                baseline_p95=0.0,
                saturation_p95=sorted_probes[-1].p95,
                degradation_detected=False,
            )

        threshold_multiplier = 1.0 + self.config.degradation_threshold
        saturation_rate = sorted_probes[-1].rate
        saturation_p95 = sorted_probes[-1].p95
        degradation_detected = False

        for probe in sorted_probes[1:]:
            if probe.p95 > baseline_p95 * threshold_multiplier:
                saturation_rate = probe.rate
                saturation_p95 = probe.p95
                degradation_detected = True
                break

        return SaturationResult(
            saturation_rate=saturation_rate,
            probe_results=self.probe_results,
            baseline_p95=baseline_p95,
            saturation_p95=saturation_p95,
            degradation_detected=degradation_detected,
        )

    def generate_stages(self, saturation_rate: float) -> List[Tuple[float, int]]:
        """Generate session arrival rate stages based on detected saturation."""
        num_stages = self.config.num_stages
        stage_duration = self.config.stage_duration

        if self.config.type == StageGenType.LINEAR:
            rates = np.linspace(self.config.min_probe_rate, saturation_rate, num_stages)
        else:
            rates = np.geomspace(self.config.min_probe_rate, saturation_rate, num_stages)

        return [(float(rate), stage_duration) for rate in rates]

    def reset(self) -> None:
        """Clear all probe results."""
        self.probe_results.clear()
