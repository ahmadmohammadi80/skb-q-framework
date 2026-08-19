"""Discrete bit-width candidates for quantization allocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BitWidthCandidate:
    """One discrete bit-width action available to policy ``pi_theta``."""

    bit_width: int
    candidate_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bit_width, int) or isinstance(self.bit_width, bool):
            raise TypeError("bit_width must be an integer")
        if self.bit_width < 0:
            raise ValueError("bit_width must be non-negative")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    def storage_bits(self, parameter_count: int) -> int:
        """Return storage cost for ``parameter_count`` parameters at this width."""

        if parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        return parameter_count * self.bit_width

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable mapping."""

        return {
            "bit_width": self.bit_width,
            "candidate_id": self.candidate_id,
            "metadata": dict(self.metadata),
        }


def zero_bit_width_candidate() -> BitWidthCandidate:
    """Return deterministic zero-width candidate for budget-exhausted targets."""

    return BitWidthCandidate(bit_width=0, candidate_id="bw0")


def default_bit_width_candidates() -> tuple[BitWidthCandidate, ...]:
    """Return deterministic default candidate set for research scaffolding."""

    return tuple(
        BitWidthCandidate(bit_width=width, candidate_id=f"bw{width}")
        for width in (1, 2, 4, 8)
    )


def normalize_candidates(
    candidates: Sequence[BitWidthCandidate],
) -> tuple[BitWidthCandidate, ...]:
    """Return candidates sorted deterministically by bit width then id."""

    normalized = tuple(candidates)
    if not normalized:
        raise ValueError("at least one BitWidthCandidate is required")
    seen_ids: set[str] = set()
    for candidate in normalized:
        if candidate.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate.candidate_id!r}")
        seen_ids.add(candidate.candidate_id)
    return tuple(sorted(normalized, key=lambda item: (item.bit_width, item.candidate_id)))
