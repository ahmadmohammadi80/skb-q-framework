"""Evaluation pipeline and metric registry for SKB-Q experiments.

Metric classes are framework-only interfaces. They do not fabricate benchmark
values; a metric reports ``not_computed`` unless a future workload supplies an
explicit observation for that metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol

from skbq.experiments.context import ExperimentContext


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One deterministic metric result."""

    name: str
    value: float | None
    status: str
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable metric result."""

        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "warnings": list(self.warnings),
            "metadata": dict(sorted(self.metadata.items())),
        }


class Metric(Protocol):
    """Plugin metric protocol for evaluation pipelines."""

    @property
    def name(self) -> str:
        """Return deterministic metric name."""

    def compute(
        self,
        context: ExperimentContext,
        observations: Mapping[str, object],
    ) -> MetricResult:
        """Compute a metric from explicit observations."""


@dataclass(frozen=True, slots=True)
class ObservationMetric:
    """Metric that reads an explicit observation by name."""

    name: str

    def compute(
        self,
        context: ExperimentContext,
        observations: Mapping[str, object],
    ) -> MetricResult:
        """Return supplied observation or an explicit not-computed status."""

        if self.name not in observations:
            return MetricResult(
                name=self.name,
                value=None,
                status="not_computed",
                warnings=(f"metric {self.name!r} was not supplied by a workload",),
            )

        value = _metric_value(observations[self.name], self.name)
        return MetricResult(name=self.name, value=value, status="computed")


class PerplexityMetric(ObservationMetric):
    """Framework metric interface for perplexity."""

    def __init__(self) -> None:
        super().__init__("perplexity")


class AccuracyMetric(ObservationMetric):
    """Framework metric interface for accuracy."""

    def __init__(self) -> None:
        super().__init__("accuracy")


class RegretMetric(ObservationMetric):
    """Framework metric interface for regret."""

    def __init__(self) -> None:
        super().__init__("regret")


class RuntimeMetric(ObservationMetric):
    """Framework metric interface for runtime observations."""

    def __init__(self) -> None:
        super().__init__("runtime")


class CompressionRatioMetric(ObservationMetric):
    """Framework metric interface for compression ratio observations."""

    def __init__(self) -> None:
        super().__init__("compression_ratio")


class MemoryEstimationMetric(ObservationMetric):
    """Framework metric interface for estimated quantized memory usage."""

    def __init__(self) -> None:
        super().__init__("memory_estimation_bytes")


@dataclass(frozen=True, slots=True)
class MetricRegistry:
    """Deterministic metric plugin registry."""

    _metrics: Mapping[str, Metric] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> MetricRegistry:
        """Return registry with default SKB-Q framework metrics."""

        registry = cls()
        return registry.register_many(
            (
                PerplexityMetric(),
                AccuracyMetric(),
                RegretMetric(),
                RuntimeMetric(),
            )
        )

    @classmethod
    def with_agentq_experiment_metrics(cls) -> MetricRegistry:
        """Return registry with metrics used by AgentQ experiment workloads."""

        return cls.with_defaults().register_many(
            (
                CompressionRatioMetric(),
                MemoryEstimationMetric(),
            )
        )

    def register(self, metric: Metric) -> MetricRegistry:
        """Return a new registry with one metric registered."""

        if not metric.name.strip():
            raise ValueError("metric name cannot be empty")
        metrics = dict(self._metrics)
        if metric.name in metrics:
            raise ValueError(f"metric already registered: {metric.name}")
        metrics[metric.name] = metric
        return MetricRegistry(dict(sorted(metrics.items())))

    def register_many(self, metrics: Sequence[Metric]) -> MetricRegistry:
        """Return a new registry with multiple metrics registered."""

        registry = self
        for metric in metrics:
            registry = registry.register(metric)
        return registry

    def names(self) -> tuple[str, ...]:
        """Return metric names in deterministic order."""

        return tuple(sorted(self._metrics))

    def metrics(self) -> tuple[Metric, ...]:
        """Return metrics in deterministic order."""

        return tuple(self._metrics[name] for name in self.names())


@dataclass(frozen=True, slots=True)
class EvaluationPipeline:
    """Evaluate registered metrics from explicit workload observations."""

    registry: MetricRegistry = field(default_factory=MetricRegistry.with_defaults)

    def evaluate(
        self,
        context: ExperimentContext,
        observations: Mapping[str, object] | None = None,
    ) -> tuple[MetricResult, ...]:
        """Evaluate all registered metrics deterministically."""

        metric_inputs = observations or {}
        return tuple(
            metric.compute(context, metric_inputs)
            for metric in self.registry.metrics()
        )


def _metric_value(value: object, metric_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"metric {metric_name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {metric_name!r} must be finite")
    return result
