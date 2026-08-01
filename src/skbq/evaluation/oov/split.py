"""Deterministic vocabulary split protocols for OOV evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.vocabulary import VocabularyStore


@dataclass(frozen=True, slots=True)
class VocabularySplit:
    """Train/test vocabulary partition for OOV evaluation."""

    train_operator_types: tuple[str, ...]
    test_operator_types: tuple[str, ...]
    protocol: str
    heldout: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train = tuple(sorted(set(self.train_operator_types)))
        test = tuple(sorted(set(self.test_operator_types)))
        overlap = set(train) & set(test)
        if overlap:
            raise ValueError(f"train/test vocabulary overlap is invalid: {sorted(overlap)}")
        if not train:
            raise ValueError("V_train cannot be empty")
        if not test:
            raise ValueError("V_test cannot be empty")
        if not self.protocol.strip():
            raise ValueError("split protocol cannot be empty")
        object.__setattr__(self, "train_operator_types", train)
        object.__setattr__(self, "test_operator_types", test)
        object.__setattr__(self, "heldout", tuple(sorted(set(self.heldout))))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def label_for(self, operator_type: str) -> str:
        """Return known/unknown label for an operator type."""

        if operator_type in self.train_operator_types:
            return "known"
        if operator_type in self.test_operator_types:
            return "unknown"
        raise KeyError(f"operator type {operator_type!r} is not present in this split")

    def membership_for(self, operator_type: str) -> str:
        """Return vocabulary membership name for an operator type."""

        return "V_train" if self.label_for(operator_type) == "known" else "V_test"

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable split mapping."""

        return {
            "protocol": self.protocol,
            "heldout": list(self.heldout),
            "V_train": list(self.train_operator_types),
            "V_test": list(self.test_operator_types),
            "metadata": dict(sorted(self.metadata.items())),
        }


def operator_class_holdout(
    vocabulary: VocabularyStore,
    held_out_operator_types: tuple[str, ...],
) -> VocabularySplit:
    """Hold out explicit operator classes from the known vocabulary."""

    all_types = set(vocabulary.operator_types())
    heldout = set(held_out_operator_types)
    unknown = sorted(heldout - all_types)
    if unknown:
        raise ValueError(f"held-out operator types are absent from vocabulary: {unknown}")
    return VocabularySplit(
        train_operator_types=tuple(sorted(all_types - heldout)),
        test_operator_types=tuple(sorted(heldout)),
        protocol="operator_class_holdout",
        heldout=tuple(sorted(heldout)),
        metadata={"vocabulary_size": len(all_types)},
    )


def architecture_family_holdout(
    vocabulary: VocabularyStore,
    held_out_architectures: tuple[str, ...],
) -> VocabularySplit:
    """Hold out operator types observed in selected architecture families."""

    heldout_architectures = set(held_out_architectures)
    if not heldout_architectures:
        raise ValueError("at least one architecture family must be held out")

    test_types = set()
    for entry in vocabulary.entries:
        architectures = set(entry.graph_statistics["architectures"])
        if architectures & heldout_architectures:
            test_types.add(entry.operator_type)

    all_types = set(vocabulary.operator_types())
    missing_architectures = heldout_architectures - {
        architecture
        for entry in vocabulary.entries
        for architecture in entry.graph_statistics["architectures"]
    }
    if missing_architectures:
        raise ValueError(f"held-out architectures are absent from vocabulary: {sorted(missing_architectures)}")

    return VocabularySplit(
        train_operator_types=tuple(sorted(all_types - test_types)),
        test_operator_types=tuple(sorted(test_types)),
        protocol="architecture_family_holdout",
        heldout=tuple(sorted(heldout_architectures)),
        metadata={"vocabulary_size": len(all_types)},
    )
