"""Analysis helpers for deterministic OOV evaluation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from skbq.evaluation.oov.oov_dataset import OOVDataset, OOVEvaluationRecord
from skbq.evaluation.oov.split import VocabularySplit


def oov_coverage(dataset: OOVDataset) -> dict[str, object]:
    """Return known/unknown coverage statistics for an OOV dataset."""

    total = len(dataset.records)
    known = len(dataset.known_records())
    unknown = len(dataset.unknown_records())
    return {
        "total_records": total,
        "known_records": known,
        "unknown_records": unknown,
        "oov_coverage": 0.0 if total == 0 else unknown / total,
    }


def candidate_recall(
    dataset: OOVDataset,
    candidate_sets: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Return candidate recall from explicit candidate sets."""

    evaluated = 0
    hits = 0
    missing_records: list[str] = []
    for record in dataset.unknown_records():
        if record.record_id not in candidate_sets:
            missing_records.append(record.record_id)
            continue
        evaluated += 1
        if record.operator_type in set(candidate_sets[record.record_id]):
            hits += 1

    return {
        "evaluated_records": evaluated,
        "hits": hits,
        "recall": None if evaluated == 0 else hits / evaluated,
        "missing_records": sorted(missing_records),
    }


def operator_distance_statistics(dataset: OOVDataset) -> dict[str, object]:
    """Return nearest known-operator distance statistics for unknown records."""

    known = dataset.known_records()
    unknown = dataset.unknown_records()
    if not known or not unknown:
        return {
            "status": "not_computed",
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    distances = tuple(
        min(_distance(record, candidate) for candidate in known)
        for record in unknown
    )
    return {
        "status": "computed",
        "count": len(distances),
        "min": min(distances),
        "max": max(distances),
        "mean": sum(distances) / len(distances),
    }


def split_reproducibility_report(split: VocabularySplit) -> dict[str, object]:
    """Return deterministic split reproducibility metadata."""

    return {
        "protocol": split.protocol,
        "heldout": list(split.heldout),
        "train_count": len(split.train_operator_types),
        "test_count": len(split.test_operator_types),
        "train_operator_types": list(split.train_operator_types),
        "test_operator_types": list(split.test_operator_types),
        "has_overlap": bool(set(split.train_operator_types) & set(split.test_operator_types)),
        "metadata": dict(sorted(split.metadata.items())),
    }


def _distance(left: OOVEvaluationRecord, right: OOVEvaluationRecord) -> float:
    left_vector = _summary_vector(left)
    right_vector = _summary_vector(right)
    return math.sqrt(sum((left_vector[index] - right_vector[index]) ** 2 for index in range(len(left_vector))))


def _summary_vector(record: OOVEvaluationRecord) -> tuple[float, ...]:
    metadata = record.structural_metadata
    return (
        float(metadata["mean_parameter_count"]),
        float(metadata["mean_input_degree"]),
        float(metadata["mean_output_degree"]),
        float(metadata["mean_depth_index"]),
        float(bool(metadata["has_nonlinearity"])),
        float(bool(metadata["has_multi_branch_routing"])),
    )
