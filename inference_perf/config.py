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
import logging
import os
import re
from datetime import datetime
from enum import Enum
from os import cpu_count
from typing import Any, Optional, List, Union

import yaml
from pydantic import BaseModel, HttpUrl, model_validator, Field

from inference_perf.circuit_breaker import CircuitBreakerConfig


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class APIConfig(BaseModel):
    type: APIType = APIType.Completion
    streaming: bool = False
    headers: Optional[dict[str, str]] = None


class TraceFormat(Enum):
    AZURE_PUBLIC_DATASET = "AzurePublicDataset"


class TraceConfig(BaseModel):
    file: str
    format: TraceFormat = TraceFormat.AZURE_PUBLIC_DATASET


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"
    Random = "random"
    SharedPrefix = "shared_prefix"
    CNNDailyMail = "cnn_dailymail"
    InfinityInstruct = "infinity_instruct"
    BillsumConversations = "billsum_conversations"
    # Agentic data types
    AgenticSynthetic = "agentic_synthetic"
    AgenticCsv = "agentic_csv"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200
    total_count: Optional[int] = None


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    num_groups: int = 10
    num_prompts_per_group: int = 10
    system_prompt_len: int = 100
    question_len: int = 50
    output_len: int = 50


# --- Agentic workload config models ---


class DistributionParams(BaseModel):
    """Parameters for distribution sampling."""
    type: str = Field(default="normal", description="Distribution type: normal, uniform, constant")
    mean: float = Field(default=0, description="Mean value (for normal distribution)")
    std_dev: float = Field(default=1, description="Standard deviation (for normal distribution)")
    min: float = Field(default=0, description="Minimum value")
    max: float = Field(default=0, description="Maximum value (for uniform distribution)")
    value: Optional[float] = Field(default=None, description="Constant value (for constant distribution)")


class AgenticSyntheticConfig(BaseModel):
    """Configuration for synthetic agentic session generation."""
    num_sessions: int = Field(default=100, gt=0, description="Number of sessions to generate")
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible session generation. None = nondeterministic.",
    )
    turns_per_session: DistributionParams = Field(
        default_factory=lambda: DistributionParams(type="normal", mean=5, std_dev=2, min=1, max=20),
        description="Distribution for number of turns per session"
    )
    tool_call_probability: float = Field(
        default=0.5, ge=0, le=1,
        description="Probability that a turn ends with tool calls"
    )
    tool_calls_per_turn: DistributionParams = Field(
        default_factory=lambda: DistributionParams(type="normal", mean=2, std_dev=1, min=1, max=5),
        description="Distribution for number of tool calls per turn"
    )
    input_tokens_turn_0: DistributionParams = Field(
        default_factory=lambda: DistributionParams(type="normal", mean=500, std_dev=100, min=100, max=2000),
        description="Distribution for input tokens in first turn"
    )
    output_tokens_per_turn: DistributionParams = Field(
        default_factory=lambda: DistributionParams(type="normal", mean=150, std_dev=50, min=10, max=500),
        description="Distribution for output tokens per turn"
    )
    tool_result_tokens: DistributionParams = Field(
        default_factory=lambda: DistributionParams(type="normal", mean=100, std_dev=30, min=10, max=500),
        description="Distribution for tool result tokens"
    )
    system_prompt_tokens: int = Field(default=0, ge=0, description="Fixed system prompt tokens")


class AgenticCsvConfig(BaseModel):
    """Configuration for loading agentic sessions from CSV."""
    path: str = Field(..., description="Path to CSV file containing session data")


class DelayType(str, Enum):
    """Types of inter-turn delays."""
    REPLAY = "replay"
    FIXED = "fixed"
    DISTRIBUTION = "distribution"
    ZERO = "zero"


class DelayConfig(BaseModel):
    """Configuration for inter-turn delays."""
    type: DelayType = Field(default=DelayType.ZERO, description="Delay type")
    fixed_ms: Optional[int] = Field(default=None, ge=0, description="Fixed delay in milliseconds")
    distribution: Optional[DistributionParams] = Field(
        default=None, description="Distribution for delay sampling"
    )


class AgenticDelayConfig(BaseModel):
    """Configuration for delays in agentic workloads."""
    tool_call_delay: DelayConfig = Field(
        default_factory=lambda: DelayConfig(type=DelayType.ZERO),
        description="Delay after tool calls"
    )
    user_think_delay: DelayConfig = Field(
        default_factory=lambda: DelayConfig(type=DelayType.ZERO),
        description="Delay between user turns"
    )


class SessionArrivalType(str, Enum):
    """Types of session arrival patterns."""
    CONSTANT = "constant"
    POISSON = "poisson"
    TRACE = "trace"


class SessionStageConfig(BaseModel):
    """Configuration for a single session load stage."""
    rate: Optional[float] = Field(default=None, gt=0, description="Session arrival rate (sessions/sec)")
    duration: Optional[int] = Field(default=None, gt=0, description="Stage duration in seconds")
    active_sessions: Optional[int] = Field(default=None, gt=0, description="Number of concurrent sessions")
    total_sessions: Optional[int] = Field(default=None, gt=0, description="Total sessions to run in stage")


class SessionArrivalConfig(BaseModel):
    """Configuration for session arrival patterns."""
    type: SessionArrivalType = Field(
        default=SessionArrivalType.CONSTANT,
        description="Session arrival type"
    )
    time_scale: float = Field(
        default=1.0, gt=0,
        description="Time scale for trace replay (1.0 = real-time, 2.0 = 2x speed)"
    )
    stages: List[SessionStageConfig] = Field(
        default_factory=list,
        description="Session load stages"
    )


class RetentionPolicyConfig(BaseModel):
    """Configuration for KV-cache retention directives.

    Controls whether and how retention_directives are injected into
    inference requests during agentic workloads. Used to instruct the
    server to protect KV-cache blocks from eviction during tool-call pauses.
    """
    type: str = Field(
        default="none",
        description="Policy type: 'none' (baseline) or 'continuum' (differential priority)"
    )
    system_prompt_priority: int = Field(default=90, ge=0, le=100)
    history_priority: int = Field(default=70, ge=0, le=100)
    tail_priority: int = Field(default=50, ge=0, le=100)
    tail_fraction: float = Field(
        default=0.2, ge=0, le=1,
        description="Fraction of context to treat as 'recent tail' (lower priority)"
    )
    ttl_buffer_s: float = Field(
        default=5.0, ge=0,
        description="Extra seconds added to tool-call duration for TTL"
    )


# --- End agentic config models ---


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock

    # Valid only for shareGPT type at this moment
    path: Optional[str] = None  # path to the downloaded shareGPT dataset

    # Distributions are only supported for synthetic/random dataset at this moment
    input_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None
    shared_prefix: Optional[SharedPrefix] = None

    # Agentic workload configurations
    agentic_synthetic: Optional[AgenticSyntheticConfig] = None
    agentic_csv: Optional[AgenticCsvConfig] = None

    # Trace file is only supported for random dataset at this moment
    trace: Optional[TraceConfig] = None


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"
    TRACE_REPLAY = "trace_replay"
    CONCURRENT = "concurrent"
    # Agentic load types
    AGENTIC = "agentic"
    AGENTIC_CONCURRENT = "agentic_concurrent"
    AGENTIC_TRACE_REPLAY = "agentic_trace_replay"


class MetricsClientType(Enum):
    PROMETHEUS = "prometheus"
    DEFAULT = "default"


class LoadStage(BaseModel):
    """Base class for load stages. Use specific subclasses for different load types."""

    pass


class StandardLoadStage(LoadStage):
    """Load stage for CONSTANT and POISSON load types."""

    rate: float = Field(..., gt=0, description="Request rate (QPS)")
    duration: int = Field(..., gt=0, description="Duration in seconds")

    # These fields should not be set for standard load types
    num_requests: Optional[int] = Field(default=None, description="Not used for standard load types")
    concurrency_level: Optional[int] = Field(default=None, description="Not used for standard load types")

    @model_validator(mode="after")
    def validate_standard_fields(self) -> "StandardLoadStage":
        if self.num_requests is not None:
            raise ValueError("num_requests should not be set for CONSTANT/POISSON load types")
        if self.concurrency_level is not None:
            raise ValueError("concurrency_level should not be set for CONSTANT/POISSON load types")
        return self


class ConcurrentLoadStage(LoadStage):
    """Load stage for CONCURRENT load type."""

    num_requests: int = Field(..., gt=0, description="Number of requests to send")
    concurrency_level: int = Field(..., gt=0, description="Concurrency level")

    # These fields are set at runtime for load generation but should not be configured
    rate: Optional[float] = Field(None, description="Set at runtime for load generation")
    duration: Optional[int] = Field(None, description="Set at runtime for load generation")

    @model_validator(mode="after")
    def validate_concurrent_fields(self) -> "ConcurrentLoadStage":
        # Allow rate and duration to be set at runtime, but they should start as None
        # No validation needed here since they're set dynamically
        return self


class StageGenType(Enum):
    GEOM = "geometric"
    LINEAR = "linear"


class SweepConfig(BaseModel):
    type: StageGenType
    num_requests: int = 2000
    timeout: float = 60
    num_stages: int = 5
    stage_duration: int = 180
    saturation_percentile: float = 95


class AgenticSweepConfig(BaseModel):
    """Configuration for agentic workload sweep/saturation detection."""
    type: StageGenType = Field(
        default=StageGenType.LINEAR,
        description="Stage generation type: linear or geometric"
    )
    num_sessions: int = Field(default=200, gt=0, description="Sessions per probe stage")
    timeout: float = Field(default=120, gt=0, description="Timeout per probe stage (seconds)")
    num_stages: int = Field(default=5, gt=0, description="Stages to generate after detection")
    stage_duration: int = Field(default=180, gt=0, description="Duration of each generated stage (seconds)")
    saturation_metric: str = Field(
        default="session_inference_time_p95",
        description="Metric to monitor for saturation detection"
    )
    degradation_threshold: float = Field(
        default=0.2, ge=0, le=1,
        description="Relative increase that indicates saturation (0.2 = 20%)"
    )
    probe_rates: Optional[List[float]] = Field(
        default=None,
        description="Explicit probe rates to test (if None, auto-generates)"
    )
    min_probe_rate: float = Field(default=1.0, gt=0, description="Minimum session arrival rate to probe")
    max_probe_rate: float = Field(default=50.0, gt=0, description="Maximum session arrival rate to probe")
    num_probes: int = Field(default=5, gt=0, description="Number of probe rate levels")


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: Union[List[StandardLoadStage], List[ConcurrentLoadStage]] = []
    sweep: Optional[SweepConfig] = None
    num_workers: int = max(1, cpu_count())  # type: ignore
    worker_max_concurrency: int = 100
    worker_max_tcp_connections: int = 2500
    trace: Optional[TraceConfig] = None
    circuit_breakers: List[str] = []
    request_timeout: Optional[float] = None

    # Agentic workload configurations
    session_arrival: Optional[SessionArrivalConfig] = None
    agentic: Optional[AgenticDelayConfig] = None
    agentic_sweep: Optional[AgenticSweepConfig] = None
    retention: Optional[RetentionPolicyConfig] = None

    @model_validator(mode="after")
    def validate_load_config(self) -> "LoadConfig":
        is_agentic = self.type in (
            LoadType.AGENTIC,
            LoadType.AGENTIC_CONCURRENT,
            LoadType.AGENTIC_TRACE_REPLAY,
        )

        # Validate that sweep is not used with concurrent or agentic load types
        if self.type == LoadType.CONCURRENT and self.sweep is not None:
            raise ValueError("Cannot have sweep config with CONCURRENT load type")
        if is_agentic and self.sweep is not None:
            raise ValueError("Cannot have sweep config with agentic load types (use agentic_sweep instead)")

        # Validate agentic_sweep is only used with agentic load types
        if self.agentic_sweep is not None:
            if not is_agentic:
                raise ValueError("agentic_sweep can only be used with agentic load types")
            if self.type == LoadType.AGENTIC_TRACE_REPLAY:
                raise ValueError("agentic_sweep cannot be used with AGENTIC_TRACE_REPLAY load type")

        # Agentic load types get default session_arrival if not set
        if is_agentic:
            if self.session_arrival is None:
                self.session_arrival = SessionArrivalConfig()
            if self.type == LoadType.AGENTIC_TRACE_REPLAY:
                self.session_arrival.type = SessionArrivalType.TRACE
            return self

        # Validate stage types match load type (for non-agentic types)
        if self.type == LoadType.CONCURRENT:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, ConcurrentLoadStage):
                    raise ValueError(
                        f"Stage {i}: CONCURRENT load type requires ConcurrentLoadStage, got {type(stage).__name__}"
                    )
        else:  # CONSTANT or POISSON
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, StandardLoadStage):
                    raise ValueError(
                        f"Stage {i}: {self.type.value.upper()} load type requires StandardLoadStage, got {type(stage).__name__}"
                    )

        return self


class StorageConfigBase(BaseModel):
    path: str = f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_file_prefix: Optional[str] = None


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str


class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str


class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = StorageConfigBase()
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = None
    simple_storage_service: Optional[SimpleStorageServiceConfig] = None


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_request: Optional[bool] = False
    # Agentic-specific report options
    per_session: Optional[bool] = False
    per_turn_position: Optional[bool] = False
    session_summary: Optional[bool] = False


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = False


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = RequestLifecycleMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = PrometheusMetricsReportConfig()


class PrometheusClientConfig(BaseModel):
    scrape_interval: int = 15
    url: Optional[HttpUrl] = None
    filters: List[str] = []
    google_managed: bool = False

    @model_validator(mode="after")
    def check_exclusive_fields(self) -> "PrometheusClientConfig":
        if bool(self.url) == bool(self.google_managed):
            raise ValueError("Exactly one of 'url' or 'google_managed' must be set.")
        return self


class MetricsClientConfig(BaseModel):
    type: MetricsClientType
    prometheus: Optional[PrometheusClientConfig] = None


class ModelServerClientConfig(BaseModel):
    type: ModelServerType = ModelServerType.VLLM
    model_name: Optional[str] = None
    base_url: str
    ignore_eos: bool = True
    api_key: Optional[str] = None


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class Config(BaseModel):
    api: APIConfig = APIConfig()
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    metrics: Optional[MetricsClientConfig] = None
    report: ReportConfig = ReportConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    server: Optional[ModelServerClientConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand ${VAR_NAME} environment variables in config values."""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config)
        result = config
        for var_name in matches:
            env_value = os.environ.get(var_name, '')
            if not env_value:
                logger = logging.getLogger(__name__)
                logger.warning(f"Environment variable '{var_name}' not set, using empty string")
            result = result.replace(f'${{{var_name}}}', env_value)
        return result
    else:
        return config


def read_config(config_file: str) -> Config:
    logger = logging.getLogger(__name__)
    logger.info("Using configuration from: %s", config_file)
    with open(config_file, "r") as stream:
        cfg = yaml.safe_load(stream)

    # Expand environment variables
    cfg = expand_env_vars(cfg)

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

    # Handle stage type conversion based on load type
    if "load" in merged_cfg and "stages" in merged_cfg["load"] and merged_cfg["load"]["stages"]:
        load_type = merged_cfg["load"].get("type", "constant")
        stages = merged_cfg["load"]["stages"]

        # Agentic load types use session_arrival.stages, not load.stages
        if load_type not in ("agentic", "agentic_concurrent", "agentic_trace_replay"):
            if load_type == "concurrent":
                concurrent_stages = []
                for stage in stages:
                    concurrent_stages.append(ConcurrentLoadStage(**stage))
                merged_cfg["load"]["stages"] = concurrent_stages
            else:
                standard_stages = []
                for stage in stages:
                    standard_stages.append(StandardLoadStage(**stage))
                merged_cfg["load"]["stages"] = standard_stages

    logger.info(
        "Benchmarking with the following config:\n\n%s\n", yaml.dump(merged_cfg, sort_keys=False, default_flow_style=False)
    )
    return Config(**merged_cfg)
