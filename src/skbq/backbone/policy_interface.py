"""Frozen policy interface for SKB-Q allocation experiments.

The interface defines how a frozen policy consumes a graph embedding and budget.
It intentionally omits RAMP/GAMMA/BAQ implementations and benchmark logic. In
Phase 4 notation, the policy is ``pi_theta`` and returns a deterministic
allocation distribution ``P(G)`` for graph embedding ``G``.
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
    """Immutable allocation distribution ``P(G)`` returned by a frozen policy."""

    allocations: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)
    policy_name: str = "pi_theta"

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("policy_name cannot be empty")
        normalized_allocations = {
            _allocation_key(key): _finite_non_negative(value, f"allocation[{key}]")
            for key, value in sorted(self.allocations.items(), key=lambda item: str(item[0]))
        }
        object.__setattr__(self, "allocations", MappingProxyType(normalized_allocations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def total_budget(self) -> float:
        """Return total allocated budget mass."""

        return sum(self.allocations.values())


class FrozenPolicy(ABC):
    """Abstract interface for frozen quantization policies ``pi_theta``."""

    @abstractmethod
    def allocate(
        self,
        graph_embedding: Sequence[float],
        budget: float,
    ) -> PolicyAllocation:
        """Return allocation distribution ``P(G)`` for a graph embedding and budget."""


@dataclass(frozen=True, slots=True)
class UniformFrozenPolicy(FrozenPolicy):
    """Deterministic reference ``pi_theta`` distributing budget uniformly."""

    name: str = "uniform_pi_theta"

    def allocate(
        self,
        graph_embedding: Sequence[float],
        budget: float,
    ) -> PolicyAllocation:
        """Return uniform ``P(G)`` over graph-embedding dimensions."""

        embedding, budget_value = validate_policy_inputs(graph_embedding, budget)
        per_dimension_budget = budget_value / len(embedding)
        return PolicyAllocation(
            allocations={
                f"dimension_{index}": per_dimension_budget
                for index in range(len(embedding))
            },
            metadata={
                "notation": "P(G)",
                "graph_embedding_dim": len(embedding),
                "budget": budget_value,
            },
            policy_name=self.name,
        )


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


def _allocation_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("allocation keys must be non-empty strings")
    return value
