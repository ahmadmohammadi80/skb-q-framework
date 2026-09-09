"""Policy action space for quantization allocation over graph targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.backbone.policy_interface import AllocationTarget
from skbq.quantization.candidates import (
    BitWidthCandidate,
    default_bit_width_candidates,
    normalize_candidates,
)


@dataclass(frozen=True, slots=True)
class PolicyActionSpace:
    """Discrete bit-width actions available to frozen policy ``pi_theta`` per target."""

    default_candidates: tuple[BitWidthCandidate, ...] = field(
        default_factory=default_bit_width_candidates
    )
    target_candidates: Mapping[str, tuple[BitWidthCandidate, ...]] = field(default_factory=dict)
    target_type_candidates: Mapping[str, tuple[BitWidthCandidate, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_candidates",
            normalize_candidates(self.default_candidates),
        )
        normalized_target_candidates = {
            str(target_id): normalize_candidates(candidates)
            for target_id, candidates in sorted(
                self.target_candidates.items(),
                key=lambda item: str(item[0]),
            )
        }
        normalized_type_candidates = {
            str(target_type): normalize_candidates(candidates)
            for target_type, candidates in sorted(
                self.target_type_candidates.items(),
                key=lambda item: str(item[0]),
            )
        }
        object.__setattr__(self, "target_candidates", MappingProxyType(normalized_target_candidates))
        object.__setattr__(
            self,
            "target_type_candidates",
            MappingProxyType(normalized_type_candidates),
        )

    def candidates_for(self, target: AllocationTarget) -> tuple[BitWidthCandidate, ...]:
        """Return deterministic candidate ordering for one allocation target."""

        if target.target_id in self.target_candidates:
            return self.target_candidates[target.target_id]
        if target.target_type in self.target_type_candidates:
            return self.target_type_candidates[target.target_type]
        return self.default_candidates

    def candidates_for_id(self, target_id: str, target_type: str = "graph_operator") -> tuple[BitWidthCandidate, ...]:
        """Return candidates for a target id without constructing ``AllocationTarget``."""

        return self.candidates_for(
            AllocationTarget(target_id=target_id, target_type=target_type)
        )

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable action-space mapping."""

        return {
            "default_candidates": [candidate.to_mapping() for candidate in self.default_candidates],
            "target_candidates": {
                target_id: [candidate.to_mapping() for candidate in candidates]
                for target_id, candidates in sorted(self.target_candidates.items())
            },
            "target_type_candidates": {
                target_type: [candidate.to_mapping() for candidate in candidates]
                for target_type, candidates in sorted(self.target_type_candidates.items())
            },
        }
