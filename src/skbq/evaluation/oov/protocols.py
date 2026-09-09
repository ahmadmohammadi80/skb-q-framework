"""OOV evaluation protocol and baseline hook interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol

from skbq.config import SeedRegistry
from skbq.evaluation.oov.oov_dataset import OOVEvaluationRecord


@dataclass(frozen=True, slots=True)
class OOVProtocolResult:
    """Deterministic protocol or baseline hook result."""

    protocol_name: str
    record_id: str
    status: str
    selected_operator_type: str | None = None
    score: float | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable result mapping."""

        return {
            "protocol_name": self.protocol_name,
            "record_id": self.record_id,
            "status": self.status,
            "selected_operator_type": self.selected_operator_type,
            "score": self.score,
            "warnings": list(self.warnings),
            "metadata": dict(sorted(self.metadata.items())),
        }


class OOVEvaluationProtocol(Protocol):
    """Interface for OOV evaluation protocols."""

    name: str

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        """Evaluate one OOV record against known vocabulary records."""


@dataclass(frozen=True, slots=True)
class SeenOperatorReconstructionProtocol:
    """Interface for seen-operator reconstruction evaluation."""

    name: str = "seen_operator_reconstruction"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="interface_only",
            warnings=("seen reconstruction workload is not implemented",),
        )


@dataclass(frozen=True, slots=True)
class UnseenOperatorBridgingProtocol:
    """Interface for unseen-operator bridging evaluation."""

    name: str = "unseen_operator_bridging"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="interface_only",
            warnings=("unseen bridging workload is not implemented",),
        )


@dataclass(frozen=True, slots=True)
class OracleComparisonProtocol:
    """Interface for oracle comparison evaluation."""

    name: str = "oracle_comparison"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="interface_only",
            warnings=("oracle comparison workload is not implemented",),
        )


@dataclass(frozen=True, slots=True)
class StructuralNearestNeighborHook:
    """Deterministic structural nearest-neighbor OOV baseline hook."""

    name: str = "structural_nearest_neighbor"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        if not known_records:
            return OOVProtocolResult(
                protocol_name=self.name,
                record_id=record.record_id,
                status="not_computed",
                warnings=("no known records available for nearest-neighbor baseline",),
            )

        ranked = sorted(
            (
                (_structural_distance(record, candidate), candidate.operator_type)
                for candidate in known_records
            ),
            key=lambda item: (item[0], item[1]),
        )
        distance, operator_type = ranked[0]
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="computed",
            selected_operator_type=operator_type,
            score=1.0 / (1.0 + distance),
            metadata={"distance": distance},
        )


@dataclass(frozen=True, slots=True)
class RandomFallbackHook:
    """Seeded deterministic random fallback OOV baseline hook."""

    seed_registry: SeedRegistry = field(default_factory=lambda: SeedRegistry({"oov": 0}))
    name: str = "random_fallback"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        if not known_records:
            return OOVProtocolResult(
                protocol_name=self.name,
                record_id=record.record_id,
                status="not_computed",
                warnings=("no known records available for random fallback baseline",),
            )

        sorted_records = tuple(sorted(known_records, key=lambda item: item.record_id))
        rng = self.seed_registry.random_for(self.name, record.record_id, base_seed_name="oov")
        selected = sorted_records[rng.randrange(len(sorted_records))]
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="computed",
            selected_operator_type=selected.operator_type,
            metadata={"seed_registry": self.seed_registry.to_mapping()},
        )


@dataclass(frozen=True, slots=True)
class GNNBaselineHook:
    """Interface placeholder for a future GNN OOV baseline."""

    name: str = "gnn_baseline"

    def evaluate(
        self,
        record: OOVEvaluationRecord,
        known_records: Sequence[OOVEvaluationRecord],
    ) -> OOVProtocolResult:
        return OOVProtocolResult(
            protocol_name=self.name,
            record_id=record.record_id,
            status="not_implemented",
            warnings=("GNN baseline interface is defined, but training/inference is not implemented",),
        )


def _structural_distance(left: OOVEvaluationRecord, right: OOVEvaluationRecord) -> float:
    left_vector = _summary_vector(left)
    right_vector = _summary_vector(right)
    return math.sqrt(sum((left_vector[index] - right_vector[index]) ** 2 for index in range(len(left_vector))))


def _summary_vector(record: OOVEvaluationRecord) -> tuple[float, ...]:
    metadata = record.structural_metadata
    return (
        float(metadata["mean_parameter_count"]),
        float(metadata["mean_input_degree"]),
        float(metadata["mean_output_degree"]),
        float(metadata["mean_depth_index"]),
        float(bool(metadata["has_nonlinearity"])),
        float(bool(metadata["has_multi_branch_routing"])),
    )
