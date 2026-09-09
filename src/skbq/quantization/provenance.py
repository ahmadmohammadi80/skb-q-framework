"""Deterministic provenance records for quantization allocation decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class AllocationProvenance:
    """Deterministic provenance for a quantization allocation plan."""

    allocator_id: str
    allocator_version: str = "reference"
    policy_name: str = "pi_theta"
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.allocator_id, "allocator_id")
        _validate_identifier(self.allocator_version, "allocator_version")
        _validate_identifier(self.policy_name, "policy_name")
        object.__setattr__(self, "config", MappingProxyType(dict(sorted(self.config.items()))))

    @property
    def deterministic_id(self) -> str:
        """Return stable SHA-256 identifier for this allocation provenance."""

        return _sha256(self.canonical_json())

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable mapping."""

        return {
            "allocator_id": self.allocator_id,
            "allocator_version": self.allocator_version,
            "policy_name": self.policy_name,
            "config": dict(self.config),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return _canonical_json(self.to_mapping())


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
