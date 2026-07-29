"""Experiment runner for reproducible SKB-Q infrastructure runs.

The runner orchestrates configuration, metadata, seeds, metrics, and result
serialization. It does not execute real benchmarks or implement model logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from skbq.config import (
    ExperimentConfig,
    SeedRegistry,
    capture_experiment_metadata,
    load_experiment_config,
)
from skbq.experiments.context import ExperimentContext
from skbq.experiments.metrics import EvaluationPipeline, MetricResult
from skbq.experiments.results import (
    ExperimentResult,
    ResultWriter,
    RunDirectoryManager,
    runtime_seconds_since,
)

Workload = Callable[[ExperimentContext], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """In-memory summary returned after one runner execution."""

    context: ExperimentContext
    result: ExperimentResult
    metrics: tuple[MetricResult, ...]
    runtime_seconds: float
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentRunner:
    """Run one reproducible experiment infrastructure pass."""

    results_root: Path = Path("results")
    repo_path: Path = Path(".")
    evaluation_pipeline: EvaluationPipeline = field(default_factory=EvaluationPipeline)
    result_writer: ResultWriter = field(default_factory=ResultWriter)
    package_names: Sequence[str] | None = None

    def run(
        self,
        config: ExperimentConfig | str | Path,
        workload: Workload | None = None,
    ) -> ExperimentRunResult:
        """Execute one dependency-free experiment runner pass."""

        start_time = perf_counter()
        experiment_config = _coerce_config(config)
        seed_registry = SeedRegistry.from_random_seeds(experiment_config.random_seeds)
        seed_registry.apply_global(_global_seed_name(seed_registry))

        run_id, run_directory = RunDirectoryManager(self.results_root).create_run_directory(
            experiment_config.experiment_id
        )
        warnings: list[str] = []
        if run_id != experiment_config.experiment_id:
            warnings.append(
                f"experiment_id {experiment_config.experiment_id!r} already existed; "
                f"using unique run id {run_id!r}"
            )

        metadata = capture_experiment_metadata(
            repo_path=self.repo_path,
            package_names=self.package_names,
            experiment_id=run_id,
        )
        context = ExperimentContext(
            config=experiment_config,
            metadata=metadata,
            seed_registry=seed_registry,
            output_directory=run_directory,
            experiment_id=run_id,
            config_hash=experiment_config.config_hash(),
            warnings=tuple(warnings),
        )

        observations = _execute_workload(context, workload, warnings)
        runtime_seconds = runtime_seconds_since(start_time)
        metrics = self.evaluation_pipeline.evaluate(context, observations)
        warnings.extend(_metric_warnings(metrics))

        result = self.result_writer.write(
            context=context,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            warnings=tuple(warnings),
        )
        return ExperimentRunResult(
            context=context,
            result=result,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            warnings=tuple(warnings),
        )


def _coerce_config(config: ExperimentConfig | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    return load_experiment_config(config)


def _global_seed_name(seed_registry: SeedRegistry) -> str:
    if "python" in seed_registry.values:
        return "python"
    return sorted(seed_registry.values)[0]


def _execute_workload(
    context: ExperimentContext,
    workload: Workload | None,
    warnings: list[str],
) -> Mapping[str, object]:
    if workload is None:
        warnings.append("no experiment workload configured; metrics are not computed")
        return {}

    observations = workload(context)
    if not isinstance(observations, Mapping):
        raise TypeError("experiment workload must return a mapping of metric observations")
    return observations


def _metric_warnings(metrics: Sequence[MetricResult]) -> tuple[str, ...]:
    return tuple(
        warning
        for metric in metrics
        for warning in metric.warnings
    )
