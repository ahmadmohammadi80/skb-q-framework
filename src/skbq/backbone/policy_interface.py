"""Frozen policy interface for SKB-Q allocation experiments.

The interface defines how a frozen policy consumes a graph embedding and budget.
It intentionally omits RAMP/GAMMA implementations and benchmark logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
import math

from skbq.backbone.encoder import GraphEmbedding, validate_embedding


@dataclass(frozen=True, slots=True)
class PolicyAllocation:
    """Immutable allocation returned by a frozen policy."""

    allocations: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_allocations = {
            key: _finite_non_negative(value, f"allocation[{key}]")
            for key, value in self.allocations.items()
        }
        object.__setattr__(self, "allocations", MappingProxyType(normalized_allocations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class FrozenPolicy(ABC):
    """Abstract interface for frozen allocation policies."""

    @abstractmethod
    def allocate(
        self,
        graph_embedding: Sequence[float],
        budget: float,
    ) -> PolicyAllocation:
        """Allocate a finite budget from an immutable graph embedding."""


def validate_policy_inputs(
    graph_embedding: Sequence[float],
    budget: float,
) -> tuple[GraphEmbedding, float]:
    """Validate common frozen-policy inputs for concrete implementations."""

    embedding = validate_embedding(graph_embedding)
    budget_value = _finite_non_negative(budget, "budget")
    return embedding, budget_value


def _finite_non_negative(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result
