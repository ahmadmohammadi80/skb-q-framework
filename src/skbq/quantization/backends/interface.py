"""Quantization backend interfaces for applying allocation plans to models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from skbq.graph.operator_graph import OperatorGraph
from skbq.quantization.operator_allocation import OperatorAllocationPlan


@dataclass(frozen=True, slots=True)
class QuantizationBackendResult:
    """Result of applying an allocation plan with a quantization backend."""

    backend_id: str
    applied_operator_ids: tuple[str, ...]
    skipped_operator_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    def to_mapping(self) -> dict[str, object]:
        """Return JSON-serializable backend result."""

        return {
            "backend_id": self.backend_id,
            "applied_operator_ids": list(self.applied_operator_ids),
            "skipped_operator_ids": list(self.skipped_operator_ids),
            "metadata": dict(self.metadata),
        }


class QuantizationBackend(Protocol):
    """Protocol for backends that apply ``OperatorAllocationPlan`` to a model."""

    @property
    def backend_id(self) -> str:
        """Return stable backend identifier."""

    def apply(
        self,
        model: object,
        graph: OperatorGraph,
        plan: OperatorAllocationPlan,
    ) -> QuantizationBackendResult:
        """Apply bit-width decisions to ``model`` according to ``plan``."""
