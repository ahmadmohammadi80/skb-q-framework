"""Result directory management and serialization for SKB-Q experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

from skbq.experiments.context import ExperimentContext
from skbq.experiments.metrics import MetricResult


RESULT_FILENAMES = (
    "config.json",
    "metadata.json",
    "metrics.json",
    "warnings.json",
)

OPTIONAL_ARTIFACT_FILENAMES = (
    "graph_manifest.json",
    "allocation_plan.json",
)


@dataclass(frozen=True, slots=True)
class RunDirectoryManager:
    """Create unique run directories without overwriting prior results."""

    results_root: Path = Path("results")

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_root", Path(self.results_root))

    def create_run_directory(self, experiment_id: str) -> tuple[str, Path]:
        """Create a unique ``results/<experiment_id>`` directory."""

        normalized_id = _safe_experiment_id(experiment_id)
        self.results_root.mkdir(parents=True, exist_ok=True)

        for suffix in range(10000):
            run_id = normalized_id if suffix == 0 else f"{normalized_id}-{suffix:03d}"
            run_directory = self.results_root / run_id
            try:
                run_directory.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return run_id, run_directory

        raise RuntimeError(f"could not create unique run directory for {normalized_id}")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Loaded or written experiment result artifact bundle."""

    experiment_id: str
    run_directory: Path
    config: dict[str, object]
    metadata: dict[str, object]
    metrics: dict[str, object]
    warnings: dict[str, object]


@dataclass(frozen=True, slots=True)
class ResultWriter:
    """Deterministic JSON result writer with no-overwrite guarantees."""

    def write(
        self,
        context: ExperimentContext,
        metrics: tuple[MetricResult, ...],
        runtime_seconds: float,
        warnings: tuple[str, ...],
        additional_artifacts: dict[str, dict[str, object]] | None = None,
    ) -> ExperimentResult:
        """Write result artifacts for one experiment run."""

        run_directory = context.output_directory
        run_directory.mkdir(parents=True, exist_ok=True)

        metric_payload = {
            "experiment_id": context.experiment_id,
            "runtime_seconds": runtime_seconds,
            "metrics": {
                metric.name: metric.to_mapping()
                for metric in sorted(metrics, key=lambda item: item.name)
            },
        }
        metadata_payload = {
            **context.metadata.to_mapping(),
            "experiment_id": context.experiment_id,
            "runtime_seconds": runtime_seconds,
            "config_hash": context.config_hash,
            "seed_values": context.seed_registry.to_mapping(),
        }
        warning_payload = {
            "experiment_id": context.experiment_id,
            "warnings": list(warnings),
        }

        config_payload = context.config.to_mapping()
        _write_json_new(run_directory / "config.json", config_payload)
        _write_json_new(run_directory / "metadata.json", metadata_payload)
        _write_json_new(run_directory / "metrics.json", metric_payload)
        _write_json_new(run_directory / "warnings.json", warning_payload)
        for filename in OPTIONAL_ARTIFACT_FILENAMES:
            if additional_artifacts and filename in additional_artifacts:
                _write_json_new(run_directory / filename, additional_artifacts[filename])

        return self.load(run_directory, optional_artifacts=additional_artifacts)

    def load(
        self,
        run_directory: str | Path,
        optional_artifacts: dict[str, dict[str, object]] | None = None,
    ) -> ExperimentResult:
        """Load result artifacts from a run directory."""

        path = Path(run_directory)
        payloads = {
            filename: _read_json(path / filename)
            for filename in RESULT_FILENAMES
        }
        metadata = payloads["metadata.json"]
        experiment_id = str(metadata["experiment_id"])
        return ExperimentResult(
            experiment_id=experiment_id,
            run_directory=path,
            config=payloads["config.json"],
            metadata=metadata,
            metrics=payloads["metrics.json"],
            warnings=payloads["warnings.json"],
        )


def runtime_seconds_since(start_time: float) -> float:
    """Return elapsed monotonic runtime seconds."""

    return perf_counter() - start_time


def write_json_artifact(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON artifact without overwriting an existing file."""

    _write_json_new(path, payload)


def _write_json_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result file: {path}")
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_experiment_id(experiment_id: str) -> str:
    normalized = experiment_id.strip()
    if not normalized:
        raise ValueError("experiment_id cannot be empty")
    forbidden = {"/", "\\", "\0"}
    if any(character in normalized for character in forbidden):
        raise ValueError("experiment_id cannot contain path separators")
    return normalized
