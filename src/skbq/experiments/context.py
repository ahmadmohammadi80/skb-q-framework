"""Experiment context for reproducible SKB-Q runner execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skbq.config import ExperimentConfig, ExperimentMetadata, SeedRegistry


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Immutable runtime context shared by evaluation and result serialization."""

    config: ExperimentConfig
    metadata: ExperimentMetadata
    seed_registry: SeedRegistry
    output_directory: Path
    experiment_id: str
    config_hash: str
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable context metadata."""

        return {
            "experiment_id": self.experiment_id,
            "output_directory": str(self.output_directory),
            "config_hash": self.config_hash,
            "seed_values": self.seed_registry.to_mapping(),
            "warnings": list(self.warnings),
        }
