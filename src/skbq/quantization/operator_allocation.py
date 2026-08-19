"""Operator-level quantization allocation records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from skbq.quantization.candidates import BitWidthCandidate
from skbq.quantization.provenance import AllocationProvenance


@dataclass(frozen=True, slots=True)
class OperatorAllocation:
    """Bit-width assignment for one operator, layer, or quantization group target."""

    target_id: str
    target_type: str
    bit_width_candidate: BitWidthCandidate
    parameter_count: int
    policy_mass: float
    operator_id: str | None = None
    layer_id: str | None = None
    quantization_group_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_target_key(self.target_id, "target_id")
        _validate_target_key(self.target_type, "target_type")
        if self.operator_id is not None:
            _validate_target_key(self.operator_id, "operator_id")
        if self.layer_id is not None:
            _validate_target_key(self.layer_id, "layer_id")
        if self.quantization_group_id is not None:
            _validate_target_key(self.quantization_group_id, "quantization_group_id")
        if not isinstance(self.parameter_count, int) or isinstance(self.parameter_count, bool):
            raise TypeError("parameter_count must be an integer")
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        policy_mass = float(self.policy_mass)
        if policy_mass < 0.0 or policy_mass != policy_mass:
            raise ValueError("policy_mass must be finite and non-negative")
        object.__setattr__(self, "policy_mass", policy_mass)
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    @property
    def storage_bits(self) -> int:
        """Return storage bits consumed by this operator allocation."""

        return self.bit_width_candidate.storage_bits(self.parameter_count)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable operator allocation."""

        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "operator_id": self.operator_id,
            "layer_id": self.layer_id,
            "quantization_group_id": self.quantization_group_id,
            "bit_width_candidate": self.bit_width_candidate.to_mapping(),
            "parameter_count": self.parameter_count,
            "policy_mass": self.policy_mass,
            "storage_bits": self.storage_bits,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OperatorAllocationPlan:
    """Concrete quantization plan derived from policy distribution ``P(G)``."""

    allocations: tuple[OperatorAllocation, ...]
    provenance: AllocationProvenance | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_allocations = tuple(
            sorted(self.allocations, key=lambda item: item.target_id)
        )
        if not normalized_allocations:
            raise ValueError("OperatorAllocationPlan requires at least one allocation")
        seen_ids = {allocation.target_id for allocation in normalized_allocations}
        if len(seen_ids) != len(normalized_allocations):
            raise ValueError("duplicate target_id in OperatorAllocationPlan")
        object.__setattr__(self, "allocations", normalized_allocations)
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    @property
    def total_storage_bits(self) -> int:
        """Return total storage bits across all operator allocations."""

        return sum(allocation.storage_bits for allocation in self.allocations)

    @property
    def total_policy_mass(self) -> float:
        """Return total policy mass referenced by this plan."""

        return sum(allocation.policy_mass for allocation in self.allocations)

    def allocation_for(self, target_id: str) -> OperatorAllocation:
        """Return allocation for one target id."""

        for allocation in self.allocations:
            if allocation.target_id == target_id:
                return allocation
        raise KeyError(f"unknown allocation target: {target_id}")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable allocation plan."""

        return {
            "allocations": [allocation.to_mapping() for allocation in self.allocations],
            "provenance": None if self.provenance is None else self.provenance.to_mapping(),
            "metadata": dict(self.metadata),
            "total_storage_bits": self.total_storage_bits,
            "total_policy_mass": self.total_policy_mass,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def allocation_hash(self) -> str:
        """Return stable SHA-256 hash for this allocation plan."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _validate_target_key(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
