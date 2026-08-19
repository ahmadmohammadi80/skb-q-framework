"""Known operator vocabulary for reproducible SKB-Q experiments.

The registry stores operator names, structural feature vectors, explicit
embedding placeholders, and metadata. The included entries are schematic
vocabulary examples for infrastructure and unit tests; they are not experimental
results or benchmark measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.bridge.candidate_filter import BridgeCandidate, EmbeddingVector
from skbq.bridge.structural_features import FeatureVector, as_feature_vector


@dataclass(frozen=True, slots=True)
class OperatorVocabularyEntry:
    """One operator entry in the SKB-Q vocabulary."""

    name: str
    structural_features: FeatureVector
    embedding: EmbeddingVector | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_features", as_feature_vector(self.structural_features))
        if self.embedding is not None:
            object.__setattr__(self, "embedding", tuple(float(value) for value in self.embedding))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_bridge_candidate(self) -> BridgeCandidate:
        """Convert the vocabulary entry into a bridge candidate."""

        return BridgeCandidate(
            identifier=self.name,
            structural_features=self.structural_features,
            embedding=self.embedding,
            metadata=dict(self.metadata),
        )


class OperatorRegistry:
    """Case-insensitive registry for known SKB-Q operator vocabulary entries."""

    def __init__(self, entries: Iterable[OperatorVocabularyEntry]) -> None:
        self._entries = {_normalize_name(entry.name): entry for entry in entries}
        if not self._entries:
            raise ValueError("operator registry requires at least one entry")

    def get(self, name: str) -> OperatorVocabularyEntry:
        """Return one operator entry by case-insensitive name."""

        key = _normalize_name(name)
        if key not in self._entries:
            raise KeyError(f"unknown operator: {name}")
        return self._entries[key]

    def names(self) -> tuple[str, ...]:
        """Return registered operator names in deterministic order."""

        return tuple(sorted(entry.name for entry in self._entries.values()))

    def entries(self) -> tuple[OperatorVocabularyEntry, ...]:
        """Return all registry entries in deterministic name order."""

        return tuple(self.get(name) for name in self.names())

    def candidates(self) -> tuple[BridgeCandidate, ...]:
        """Return all entries as bridge candidates."""

        return tuple(entry.to_bridge_candidate() for entry in self.entries())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return _normalize_name(name) in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def default_operator_registry() -> OperatorRegistry:
    """Return the default known-operator registry for reproducible experiments."""

    return OperatorRegistry(KNOWN_OPERATOR_ENTRIES)


def _entry(
    name: str,
    structural_features: Sequence[float],
    family: str,
    notes: str,
) -> OperatorVocabularyEntry:
    return OperatorVocabularyEntry(
        name=name,
        structural_features=as_feature_vector(structural_features),
        embedding=None,
        metadata={
            "family": family,
            "feature_source": "schematic_registry_entry",
            "notes": notes,
        },
    )


def _normalize_name(name: str) -> str:
    normalized = " ".join(name.strip().casefold().split())
    if not normalized:
        raise ValueError("operator name cannot be empty")
    return normalized


KNOWN_OPERATOR_ENTRIES: tuple[OperatorVocabularyEntry, ...] = (
    _entry(
        "Attention",
        (0.0, 1.0, 1.0, 0.40, 0.0, 0.0),
        "attention",
        "Canonical attention block vocabulary entry.",
    ),
    _entry(
        "GQA",
        (-0.2876820724517809, 1.0, 1.0, 0.40, 0.0, 0.0),
        "attention",
        "Grouped-query attention vocabulary entry.",
    ),
    _entry(
        "SwiGLU",
        (0.4054651081081644, 1.0, 1.0, 0.65, 1.0, 0.0),
        "feed_forward",
        "Gated feed-forward activation vocabulary entry.",
    ),
    _entry(
        "RMSNorm",
        (-2.302585092994046, 1.0, 1.0, 0.20, 0.0, 0.0),
        "normalization",
        "Root-mean-square normalization vocabulary entry.",
    ),
    _entry(
        "MoE Router",
        (-1.3862943611198906, 1.0, 1.0, 0.55, 0.0, 1.0),
        "mixture_of_experts",
        "Router vocabulary entry with multi-branch routing.",
    ),
    _entry(
        "Expert FFN",
        (0.6931471805599453, 1.0, 1.0, 0.70, 1.0, 0.0),
        "mixture_of_experts",
        "Expert feed-forward network vocabulary entry.",
    ),
)
