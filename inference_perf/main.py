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
from argparse import ArgumentParser
from inference_perf.analysis.analyze import analyze_reports
from typing import Any, Dict, List, Optional
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.loadgen import LoadGenerator, AgenticLoadGenerator
from inference_perf.loadgen.session_runner import SessionResult
from inference_perf.config import (
    DataGenType,
    LoadType,
    MetricsClientType,
    ModelServerType,
    ReportConfig,
    StandardLoadStage,
    ConcurrentLoadStage,
    RetentionPolicyConfig,
    read_config,
)
from inference_perf.datagen import (
    DataGenerator,
    MockDataGenerator,
    HFShareGPTDataGenerator,
    SyntheticDataGenerator,
    RandomDataGenerator,
    SharedPrefixDataGenerator,
    CNNDailyMailDataGenerator,
    InfinityInstructDataGenerator,
    BillsumConversationsDataGenerator,
    AgenticCsvDataGenerator,
    AgenticSyntheticDataGenerator,
)
from inference_perf.client.modelserver import (
    ModelServerClient,
    vLLMModelServerClient,
    SGlangModelServerClient,
    MockModelServerClient,
)
from inference_perf.client.metricsclient.base import MetricsClient, PerfRuntimeParameters
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient, GoogleManagedPrometheusMetricsClient
from inference_perf.client.filestorage import (
    StorageClient,
    GoogleCloudStorageClient,
    LocalStorageClient,
    SimpleStorageServiceClient,
)
from inference_perf.client.requestdatacollector import (
    RequestDataCollector,
    LocalRequestDataCollector,
    MultiprocessRequestDataCollector,
)
from inference_perf.circuit_breaker import init_circuit_breakers
from inference_perf.reportgen import ReportGenerator
from inference_perf.reportgen.base import summarize
from inference_perf.policies import ContinuumPolicy, NoopPolicy
from inference_perf.utils import CustomTokenizer, ReportFile
from inference_perf.utils.content_generator import ContentGenerator
from inference_perf.metrics import SessionMetrics, TurnPositionMetrics
from inference_perf.logger import setup_logging
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

AGENTIC_LOAD_TYPES = {LoadType.AGENTIC, LoadType.AGENTIC_CONCURRENT, LoadType.AGENTIC_TRACE_REPLAY}


class InferencePerfRunner:
    def __init__(
        self,
        client: ModelServerClient,
        loadgen: LoadGenerator,
        reportgen: ReportGenerator,
        storage_clients: List[StorageClient],
    ) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen
        self.storage_clients = storage_clients

    def run(self) -> None:
        async def _run() -> None:
            # Start the collector, which will gather metrics from the model_server_client
            async with self.reportgen.get_metrics_collector().start():
                # Generate load that is sent to inference endpoint
                await self.loadgen.run(self.client)

        asyncio.run(_run())

    def generate_reports(self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters) -> List[ReportFile]:
        return asyncio.run(self.reportgen.generate_reports(report_config=report_config, runtime_parameters=runtime_parameters))

    def save_reports(self, reports: List[ReportFile]) -> None:
        for storage_client in self.storage_clients:
            storage_client.save_report(reports)

    def stop(self) -> None:
        asyncio.run(self.loadgen.stop())


def generate_session_reports(
    session_results: List[SessionResult],
    report_config: ReportConfig,
) -> List[ReportFile]:
    """Generate session-level reports from agentic workload results."""
    reports: List[ReportFile] = []

    # Build SessionMetrics from results
    session_metrics_list = [
        SessionMetrics.from_session_result(sr)
        for sr in session_results
    ]

    # Session summary report
    if report_config.request_lifecycle.session_summary:
        latencies = [m.session_latency_ms for m in session_metrics_list]
        inference_times = [m.session_inference_time_ms for m in session_metrics_list]
        duty_cycles = [m.inference_duty_cycle for m in session_metrics_list]
        turns = [float(m.turns_completed) for m in session_metrics_list]
        peak_contexts = [float(m.peak_context_length) for m in session_metrics_list]

        summary: Dict[str, Any] = {
            "total_sessions": len(session_metrics_list),
            "session_latency_ms": summarize(latencies),
            "session_inference_time_ms": summarize(inference_times),
            "inference_duty_cycle": summarize(duty_cycles),
            "turns_completed": summarize(turns),
            "peak_context_length": summarize(peak_contexts),
        }

        # Throughput
        if session_results:
            total_wall_time = max(sr.end_time for sr in session_results) - min(sr.start_time for sr in session_results)
            if total_wall_time > 0:
                summary["session_throughput"] = len(session_results) / total_wall_time
                total_turns = sum(sr.turns_completed for sr in session_results)
                summary["effective_request_rate"] = total_turns / total_wall_time

        reports.append(ReportFile(
            name="session_summary_metrics",
            contents=summary,
        ))

    # Per-turn-position report
    if report_config.request_lifecycle.per_turn_position:
        max_turns = max((m.turns_total for m in session_metrics_list), default=0)
        turn_position_data = []

        for turn_idx in range(max_turns):
            tpm = TurnPositionMetrics.from_session_metrics_list(turn_idx, session_metrics_list)
            if tpm.count > 0:
                turn_position_data.append(tpm.to_dict())

        reports.append(ReportFile(
            name="per_turn_position_metrics",
            contents=turn_position_data,
        ))

    # Per-session report
    if report_config.request_lifecycle.per_session:
        reports.append(ReportFile(
            name="per_session_metrics",
            contents=[m.to_dict() for m in session_metrics_list],
        ))

    return reports


def main_cli() -> None:
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config File", required=False)
    parser.add_argument("-a", "--analyze", help="Path to a report directory to analyze.", required=False)
    parser.add_argument(
        "--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.analyze:
        analyze_reports(args.analyze)
        return

    if not args.config_file:
        parser.error("argument -c/--config_file is required when not using --analyze")

    config = read_config(args.config_file)

    is_agentic = config.load.type in AGENTIC_LOAD_TYPES

    # Set stage rates to high values if using concurrent load type
    if config.load.type == LoadType.CONCURRENT:
        for stage in config.load.stages:
            if isinstance(stage, ConcurrentLoadStage):
                stage.duration = 1
                stage.rate = stage.num_requests
        config.load.worker_max_concurrency = 0

    # Define Circuit Breakers
    if config.circuit_breakers:
        init_circuit_breakers(config.circuit_breakers)

    # Define Metrics Client
    metrics_client: Optional[MetricsClient] = None
    if config.metrics:
        if config.metrics.type == MetricsClientType.PROMETHEUS and config.metrics.prometheus:
            if config.metrics.prometheus.google_managed:
                metrics_client = GoogleManagedPrometheusMetricsClient(config.metrics.prometheus)
            else:
                metrics_client = PrometheusMetricsClient(config=config.metrics.prometheus)

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.local_storage:
            storage_clients.append(LocalStorageClient(config=config.storage.local_storage))
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))
        if config.storage.simple_storage_service:
            storage_clients.append(SimpleStorageServiceClient(config=config.storage.simple_storage_service))

    # Define Report Generator.
    # Agentic path runs entirely in one process (asyncio tasks within main_cli),
    # so use the in-process collector. The multiprocess one spawns a non-daemon
    # mp.JoinableQueue feeder thread; in agentic path nobody drains the queue,
    # so the feeder thread blocks process exit forever after the bench ends.
    collector: RequestDataCollector
    if is_agentic or config.load.num_workers <= 0:
        collector = LocalRequestDataCollector()
    else:
        collector = MultiprocessRequestDataCollector()
    reportgen = ReportGenerator(metrics_client, collector, config=config)

    # Create tokenizer based on tokenizer config
    tokenizer: Optional[CustomTokenizer] = None
    if config.tokenizer and config.tokenizer.pretrained_model_name_or_path:
        try:
            tokenizer = CustomTokenizer(config.tokenizer)
        except Exception as e:
            raise Exception("Tokenizer initialization failed") from e

    # Define Model Server Client
    model_server_client: ModelServerClient
    if config.server:
        if config.server.type == ModelServerType.VLLM:
            model_server_client = vLLMModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
            )
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.SGLANG:
            model_server_client = SGlangModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
            )
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.TGI:
            model_server_client = TGImodelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
            )
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.MOCK:
            model_server_client = MockModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                timeout=config.load.request_timeout,
            )
            tokenizer = model_server_client.tokenizer
    else:
        raise Exception("model server client config missing")

    # --- Agentic workload path ---
    if is_agentic:
        if tokenizer is None:
            raise Exception("Agentic workloads require a configured tokenizer")

        # Load sessions from data source
        if config.data.type == DataGenType.AgenticCsv:
            if config.data.agentic_csv is None:
                raise Exception("agentic_csv configuration is required for agentic_csv data type")
            agentic_datagen = AgenticCsvDataGenerator(config.data.agentic_csv)
        elif config.data.type == DataGenType.AgenticSynthetic:
            if config.data.agentic_synthetic is None:
                raise Exception("agentic_synthetic configuration is required for agentic_synthetic data type")
            agentic_datagen = AgenticSyntheticDataGenerator(config.data.agentic_synthetic)
        else:
            raise Exception(f"Unsupported agentic data type: {config.data.type.value}")

        sessions = agentic_datagen.get_sessions()
        content_generator = ContentGenerator(tokenizer)

        # Build retention policy from config
        retention_policy = None
        system_prompt_tokens = 0
        if config.load.retention and config.load.retention.type == "continuum":
            rc = config.load.retention
            retention_policy = ContinuumPolicy(
                system_prompt_priority=rc.system_prompt_priority,
                history_priority=rc.history_priority,
                tail_priority=rc.tail_priority,
                tail_fraction=rc.tail_fraction,
                ttl_buffer_s=rc.ttl_buffer_s,
            )
            # Get system_prompt_tokens from data config
            if config.data.agentic_synthetic:
                system_prompt_tokens = config.data.agentic_synthetic.system_prompt_tokens
            logger.info(
                f"Retention policy: continuum (priorities={rc.system_prompt_priority}/"
                f"{rc.history_priority}/{rc.tail_priority}, "
                f"tail_fraction={rc.tail_fraction}, ttl_buffer={rc.ttl_buffer_s}s)"
            )

        agentic_loadgen = AgenticLoadGenerator(
            sessions=sessions,
            load_config=config.load,
            api_config=config.api,
            content_generator=content_generator,
            retention_policy=retention_policy,
            system_prompt_tokens=system_prompt_tokens,
        )

        start_time = time.time()

        # Run agentic load test
        asyncio.run(agentic_loadgen.run(model_server_client))

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Agentic benchmark completed in {duration:.1f}s")

        # Generate session-level reports
        session_results = agentic_loadgen.get_all_session_results()
        reports = generate_session_reports(session_results, config.report)

        # Also generate standard request lifecycle reports from the collector
        # (only if the collector was started — agentic path may skip it)
        try:
            request_metrics = [
                metric for metric in reportgen.get_metrics_collector().get_metrics()
                if metric.stage_id is not None and metric.stage_id >= 0
            ]
            if request_metrics:
                from inference_perf.reportgen.base import summarize_requests
                reports.append(ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=summarize_requests(request_metrics).model_dump(),
                ))
        except AttributeError:
            logger.debug("Skipping lifecycle metrics — collector not started in agentic path")

        # Add config report
        reports.append(ReportFile(
            name="config",
            file_type="yaml",
            contents=config.model_dump(mode="json"),
        ))

        # Save reports
        for storage_client in storage_clients:
            storage_client.save_report(reports)

        return

    # --- Standard (non-agentic) path ---

    if config.load is None:
        raise Exception("load config missing")

    if len(config.load.stages) == 0 and config.load.sweep is None:
        raise Exception("Load stages must be configured, or sweep must be configured")

    # Define DataGenerator
    datagen: DataGenerator
    if config.data:
        if config.data.type in set(
            {
                DataGenType.ShareGPT,
                DataGenType.Synthetic,
                DataGenType.Random,
                DataGenType.CNNDailyMail,
                DataGenType.InfinityInstruct,
                DataGenType.BillsumConversations,
            }
        ):
            if tokenizer is None:
                raise Exception(
                    f"{config.data.type.value} data generator requires a configured tokenizer. "
                    "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
                )

        if config.data.type in [DataGenType.Synthetic, DataGenType.Random]:
            if config.data.trace is None:
                if config.data.input_distribution is None:
                    raise Exception(
                        f"{config.data.type.value} data generator requires 'input_distribution' to be configured if no trace config is provided"
                    )
                if config.data.output_distribution is None:
                    raise Exception(
                        f"{config.data.type.value} data generator requires 'output_distribution' to be configured if no trace config is provided"
                    )

                max_requests = 0
                for stage in config.load.stages:
                    if isinstance(stage, StandardLoadStage):
                        max_requests = max(max_requests, int(stage.rate * stage.duration))
                    elif isinstance(stage, ConcurrentLoadStage):
                        max_requests = max(max_requests, stage.num_requests)
                total_count = max_requests + 1
                if config.data.input_distribution.total_count is None:
                    config.data.input_distribution.total_count = total_count
                if config.data.output_distribution.total_count is None:
                    config.data.output_distribution.total_count = total_count

        if config.data.type == DataGenType.SharedPrefix and config.data.shared_prefix is None:
            raise Exception(f"{config.data.type.value} data generator requires 'shared_prefix' to be configured")

        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.CNNDailyMail:
            datagen = CNNDailyMailDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Synthetic:
            datagen = SyntheticDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Random:
            datagen = RandomDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.SharedPrefix:
            datagen = SharedPrefixDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.InfinityInstruct:
            datagen = InfinityInstructDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.BillsumConversations:
            datagen = BillsumConversationsDataGenerator(config.api, config.data, tokenizer)
        else:
            datagen = MockDataGenerator(config.api, config.data, tokenizer)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if isinstance(metrics_client, PrometheusMetricsClient) and config.report.prometheus and config.report.prometheus.per_stage:
        config.load.interval = max(config.load.interval, metrics_client.scrape_interval)
    loadgen = LoadGenerator(datagen, config.load)

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(model_server_client, loadgen, reportgen, storage_clients)

    start_time = time.time()

    # Run Perf Test
    perfrunner.run()

    end_time = time.time()
    duration = end_time - start_time

    # Generate Reports after the tests
    reports = perfrunner.generate_reports(
        report_config=config.report,
        runtime_parameters=PerfRuntimeParameters(
            start_time=start_time,
            duration=duration,
            model_server_metrics=model_server_client.get_prometheus_metric_metadata(),
            stages=loadgen.stage_runtime_info,
        ),
    )

    # Save Reports
    perfrunner.save_reports(reports=reports)

    perfrunner.stop()


if __name__ == "__main__":
    main_cli()
