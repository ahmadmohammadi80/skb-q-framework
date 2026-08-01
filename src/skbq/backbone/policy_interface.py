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
import hashlib
import json
from types import MappingProxyType
import math

from skbq.backbone.encoder import GraphEmbedding, validate_embedding
from skbq.backbone.provenance import PolicyProvenance


@dataclass(frozen=True, slots=True)
class AllocationTarget:
    """Typed target receiving budget from frozen policy distribution ``P(G)``."""

    target_id: str
    target_type: str
    operator_id: str | None = None
    layer_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _allocation_key(self.target_id)
        _allocation_key(self.target_type)
        if self.operator_id is not None:
            _allocation_key(self.operator_id)
        if self.layer_id is not None:
            _allocation_key(self.layer_id)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable target mapping."""

        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "operator_id": self.operator_id,
            "layer_id": self.layer_id,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class TargetAllocation:
    """Budget assigned to one allocation target."""

    target: AllocationTarget
    budget: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "budget",
            _finite_non_negative(self.budget, f"allocation[{self.target.target_id}]"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable target allocation."""

        return {"target": self.target.to_mapping(), "budget": self.budget}


@dataclass(frozen=True, slots=True, init=False)
class PolicyAllocation:
    """Immutable allocation distribution ``P(G)`` returned by a frozen policy."""

    target_allocations: tuple[TargetAllocation, ...]
    metadata: Mapping[str, object]
    policy_name: str
    provenance: PolicyProvenance | None

    def __init__(
        self,
        target_allocations: Sequence[TargetAllocation] | None = None,
        allocations: Mapping[str, float] | None = None,
        metadata: Mapping[str, object] | None = None,
        policy_name: str = "pi_theta",
        provenance: PolicyProvenance | None = None,
    ) -> None:
        if not policy_name.strip():
            raise ValueError("policy_name cannot be empty")
        if target_allocations is not None and allocations is not None:
            raise ValueError("provide either target_allocations or allocations, not both")

        if target_allocations is None:
            if allocations is None:
                raise ValueError("PolicyAllocation requires at least one allocation target")
            target_allocations = tuple(
                TargetAllocation(
                    target=AllocationTarget(
                        target_id=_allocation_key(key),
                        target_type="quantization_target",
                    ),
                    budget=value,
                )
                for key, value in sorted(allocations.items(), key=lambda item: str(item[0]))
            )

        normalized_target_allocations = tuple(
            sorted(target_allocations, key=lambda item: item.target.target_id)
        )
        if not normalized_target_allocations:
            raise ValueError("PolicyAllocation requires at least one allocation target")

        object.__setattr__(self, "target_allocations", normalized_target_allocations)
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
        object.__setattr__(self, "policy_name", policy_name)
        object.__setattr__(self, "provenance", provenance)

    @property
    def allocations(self) -> Mapping[str, float]:
        """Return compatibility mapping from target id to allocated budget."""

        return MappingProxyType(
            {
                allocation.target.target_id: allocation.budget
                for allocation in self.target_allocations
            }
        )

    @property
    def total_budget(self) -> float:
        """Return total allocated budget mass."""

        return sum(allocation.budget for allocation in self.target_allocations)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable policy allocation mapping."""

        return {
            "policy_name": self.policy_name,
            "provenance": None if self.provenance is None else self.provenance.to_mapping(),
            "target_allocations": [
                allocation.to_mapping() for allocation in self.target_allocations
            ],
            "metadata": dict(sorted(self.metadata.items())),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def allocation_hash(self) -> str:
        """Return stable SHA-256 hash for this policy allocation."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class FrozenPolicy(ABC):
    """Abstract interface for frozen quantization policies ``pi_theta``."""

    @abstractmethod
    def allocate(
        self,
        graph_embedding: Sequence[float],
        budget: float,
        targets: Sequence[AllocationTarget] | None = None,
    ) -> PolicyAllocation:
        """Return allocation distribution ``P(G)`` for a graph embedding and budget."""


@dataclass(frozen=True, slots=True)
class UniformFrozenPolicy(FrozenPolicy):
    """Deterministic reference ``pi_theta`` distributing budget uniformly."""

    name: str = "uniform_pi_theta"

    @property
    def provenance(self) -> PolicyProvenance:
        """Return deterministic provenance for this reference policy."""

        return PolicyProvenance(
            policy_id=self.name,
            config={"allocation_rule": "uniform_over_targets"},
        )

    def allocate(
        self,
        graph_embedding: Sequence[float],
        budget: float,
        targets: Sequence[AllocationTarget] | None = None,
    ) -> PolicyAllocation:
        """Return uniform ``P(G)`` over allocation targets."""

        embedding, budget_value = validate_policy_inputs(graph_embedding, budget)
        allocation_targets = tuple(targets or (AllocationTarget("graph", "graph"),))
        per_target_budget = budget_value / len(allocation_targets)
        return PolicyAllocation(
            target_allocations=tuple(
                TargetAllocation(target=target, budget=per_target_budget)
                for target in allocation_targets
            ),
            metadata={
                "notation": "P(G)",
                "graph_embedding_dim": len(embedding),
                "target_count": len(allocation_targets),
                "budget": budget_value,
            },
            policy_name=self.name,
            provenance=self.provenance,
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
