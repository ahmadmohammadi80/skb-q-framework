"""OOV evaluation records derived from vocabulary splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from skbq.evaluation.oov.split import VocabularySplit
from skbq.vocabulary import VocabularyEntry, VocabularyStore


@dataclass(frozen=True, slots=True)
class OOVEvaluationRecord:
    """One deterministic record for OOV operator evaluation."""

    source_model: str
    operator_type: str
    structural_metadata: Mapping[str, object]
    label: str
    vocabulary_membership: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.label not in {"known", "unknown"}:
            raise ValueError("OOV label must be 'known' or 'unknown'")
        if self.vocabulary_membership not in {"V_train", "V_test"}:
            raise ValueError("vocabulary_membership must be V_train or V_test")
        object.__setattr__(self, "structural_metadata", MappingProxyType(dict(self.structural_metadata)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def record_id(self) -> str:
        """Return deterministic record id."""

        return f"{self.source_model}::{self.operator_type}::{self.vocabulary_membership}"

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable record mapping."""

        return {
            "record_id": self.record_id,
            "source_model": self.source_model,
            "operator_type": self.operator_type,
            "structural_metadata": dict(sorted(self.structural_metadata.items())),
            "label": self.label,
            "vocabulary_membership": self.vocabulary_membership,
            "provenance": dict(sorted(self.provenance.items())),
        }


@dataclass(frozen=True, slots=True)
class OOVDataset:
    """Deterministic OOV dataset generated from a vocabulary split."""

    records: tuple[OOVEvaluationRecord, ...]
    split: VocabularySplit

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: VocabularyStore,
        split: VocabularySplit,
    ) -> OOVDataset:
        """Create records from all vocabulary entries and source provenance."""

        records = []
        for entry in vocabulary.entries:
            for provenance in entry.provenance:
                records.append(_record_from_entry(entry, provenance.to_mapping(), split))
        return cls(
            records=tuple(sorted(records, key=lambda record: record.record_id)),
            split=split,
        )

    def known_records(self) -> tuple[OOVEvaluationRecord, ...]:
        """Return records belonging to ``V_train``."""

        return tuple(record for record in self.records if record.label == "known")

    def unknown_records(self) -> tuple[OOVEvaluationRecord, ...]:
        """Return records belonging to ``V_test``."""

        return tuple(record for record in self.records if record.label == "unknown")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable dataset mapping."""

        return {
            "split": self.split.to_mapping(),
            "records": [record.to_mapping() for record in self.records],
        }


def _record_from_entry(
    entry: VocabularyEntry,
    provenance: Mapping[str, object],
    split: VocabularySplit,
) -> OOVEvaluationRecord:
    label = split.label_for(entry.operator_type)
    return OOVEvaluationRecord(
        source_model=str(provenance["source_model"]),
        operator_type=entry.operator_type,
        structural_metadata=entry.structural_metadata,
        label=label,
        vocabulary_membership=split.membership_for(entry.operator_type),
        provenance=provenance,
    )
