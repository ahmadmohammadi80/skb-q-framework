"""Immutable vocabulary store for extracted SKB-Q operator vocabulary entries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.bridge.structural_features import OperatorStructuralMetadata
from skbq.graph.operator_graph import OperatorGraph, OperatorNode


VOCABULARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class VocabularySourceProvenance:
    """Provenance for one source model contributing to a vocabulary."""

    source_model: str
    model_version: str | None
    extraction_timestamp: str
    git_commit: str | None
    framework_version: str

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable provenance."""

        return {
            "source_model": self.source_model,
            "model_version": self.model_version,
            "extraction_timestamp": self.extraction_timestamp,
            "git_commit": self.git_commit,
            "framework_version": self.framework_version,
        }


@dataclass(frozen=True, slots=True)
class VocabularyEntry:
    """Unified vocabulary entry for one operator type."""

    operator_type: str
    structural_metadata: Mapping[str, object]
    parameter_statistics: Mapping[str, object]
    graph_statistics: Mapping[str, object]
    provenance: tuple[VocabularySourceProvenance, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.operator_type.strip():
            raise ValueError("operator_type cannot be empty")
        object.__setattr__(self, "structural_metadata", MappingProxyType(dict(self.structural_metadata)))
        object.__setattr__(self, "parameter_statistics", MappingProxyType(dict(self.parameter_statistics)))
        object.__setattr__(self, "graph_statistics", MappingProxyType(dict(self.graph_statistics)))
        object.__setattr__(
            self,
            "provenance",
            tuple(sorted(self.provenance, key=lambda item: (item.source_model, item.model_version or ""))),
        )

    @classmethod
    def from_graph_nodes(
        cls,
        operator_type: str,
        graph: OperatorGraph,
        nodes: Sequence[OperatorNode],
        provenance: VocabularySourceProvenance,
    ) -> VocabularyEntry:
        """Create one vocabulary entry from graph nodes of the same operator type."""

        metadata = tuple(graph.structural_metadata_for(node.operator_id) for node in nodes)
        return cls(
            operator_type=operator_type,
            structural_metadata=_structural_summary(metadata),
            parameter_statistics=_parameter_statistics(nodes),
            graph_statistics=_graph_statistics(graph, nodes),
            provenance=(provenance,),
        )

    def merge(self, other: VocabularyEntry) -> VocabularyEntry:
        """Merge duplicate operator-type entries from multiple source models."""

        if self.operator_type != other.operator_type:
            raise ValueError("cannot merge entries with different operator types")

        left_count = int(self.graph_statistics["occurrence_count"])
        right_count = int(other.graph_statistics["occurrence_count"])
        total_count = left_count + right_count
        return VocabularyEntry(
            operator_type=self.operator_type,
            structural_metadata=_merge_structural_metadata(
                self.structural_metadata,
                other.structural_metadata,
                left_count,
                right_count,
            ),
            parameter_statistics=_merge_parameter_statistics(
                self.parameter_statistics,
                other.parameter_statistics,
            ),
            graph_statistics=_merge_graph_statistics(self.graph_statistics, other.graph_statistics),
            provenance=_unique_provenance((*self.provenance, *other.provenance)),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable vocabulary entry."""

        return {
            "operator_type": self.operator_type,
            "structural_metadata": dict(sorted(self.structural_metadata.items())),
            "parameter_statistics": dict(sorted(self.parameter_statistics.items())),
            "graph_statistics": dict(sorted(self.graph_statistics.items())),
            "provenance": [item.to_mapping() for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class VocabularyStore:
    """Unified vocabulary ``V`` built from one or more supported models."""

    entries: tuple[VocabularyEntry, ...]
    schema_version: str = VOCABULARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        merged: dict[str, VocabularyEntry] = {}
        for entry in self.entries:
            merged[entry.operator_type] = (
                entry
                if entry.operator_type not in merged
                else merged[entry.operator_type].merge(entry)
            )
        object.__setattr__(
            self,
            "entries",
            tuple(merged[name] for name in sorted(merged)),
        )

    @classmethod
    def from_graph(
        cls,
        graph: OperatorGraph,
        provenance: VocabularySourceProvenance,
    ) -> VocabularyStore:
        """Build a vocabulary store from one extracted operator graph."""

        nodes_by_type: dict[str, list[OperatorNode]] = {}
        for node in graph.nodes:
            nodes_by_type.setdefault(node.operator_type, []).append(node)
        return cls(
            tuple(
                VocabularyEntry.from_graph_nodes(operator_type, graph, tuple(nodes), provenance)
                for operator_type, nodes in sorted(nodes_by_type.items())
            )
        )

    @classmethod
    def merge_many(cls, stores: Iterable[VocabularyStore]) -> VocabularyStore:
        """Merge multiple vocabulary stores into one unified vocabulary ``V``."""

        entries: list[VocabularyEntry] = []
        for store in stores:
            entries.extend(store.entries)
        if not entries:
            raise ValueError("cannot merge an empty vocabulary store sequence")
        return cls(tuple(entries))

    def get(self, operator_type: str) -> VocabularyEntry:
        """Return a vocabulary entry by operator type."""

        for entry in self.entries:
            if entry.operator_type == operator_type:
                return entry
        raise KeyError(f"unknown operator type: {operator_type}")

    def operator_types(self) -> tuple[str, ...]:
        """Return operator types in deterministic order."""

        return tuple(entry.operator_type for entry in self.entries)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable vocabulary mapping."""

        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_mapping() for entry in self.entries],
        }


def _structural_summary(metadata: Sequence[OperatorStructuralMetadata]) -> dict[str, object]:
    return {
        "mean_parameter_count": _mean(item.parameter_count for item in metadata),
        "mean_reference_parameter_count": _mean(item.reference_parameter_count for item in metadata),
        "mean_input_degree": _mean(item.input_degree for item in metadata),
        "mean_output_degree": _mean(item.output_degree for item in metadata),
        "mean_depth_index": _mean(item.depth_index for item in metadata),
        "max_input_degree": max(item.max_input_degree for item in metadata),
        "max_output_degree": max(item.max_output_degree for item in metadata),
        "max_depth_index": max(item.max_depth_index for item in metadata),
        "has_nonlinearity": any(item.has_nonlinearity for item in metadata),
        "has_multi_branch_routing": any(item.has_multi_branch_routing for item in metadata),
    }


def _parameter_statistics(nodes: Sequence[OperatorNode]) -> dict[str, object]:
    counts = tuple(float(node.parameter_count) for node in nodes)
    return {
        "count": len(counts),
        "total": sum(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": _mean(counts),
    }


def _graph_statistics(graph: OperatorGraph, nodes: Sequence[OperatorNode]) -> dict[str, object]:
    return {
        "occurrence_count": len(nodes),
        "source_graph_count": 1,
        "architectures": [graph.architecture],
        "operator_ids": sorted(node.operator_id for node in nodes),
        "max_depth_position": max(node.depth_position for node in nodes),
        "max_input_degree": max(node.input_degree for node in nodes),
        "max_output_degree": max(node.output_degree for node in nodes),
    }


def _merge_structural_metadata(
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_count: int,
    right_count: int,
) -> dict[str, object]:
    total = left_count + right_count
    merged = {}
    for key in (
        "mean_parameter_count",
        "mean_reference_parameter_count",
        "mean_input_degree",
        "mean_output_degree",
        "mean_depth_index",
    ):
        merged[key] = (float(left[key]) * left_count + float(right[key]) * right_count) / total
    merged["max_input_degree"] = max(float(left["max_input_degree"]), float(right["max_input_degree"]))
    merged["max_output_degree"] = max(float(left["max_output_degree"]), float(right["max_output_degree"]))
    merged["max_depth_index"] = max(float(left["max_depth_index"]), float(right["max_depth_index"]))
    merged["has_nonlinearity"] = bool(left["has_nonlinearity"]) or bool(right["has_nonlinearity"])
    merged["has_multi_branch_routing"] = bool(left["has_multi_branch_routing"]) or bool(
        right["has_multi_branch_routing"]
    )
    return merged


def _merge_parameter_statistics(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, object]:
    count = int(left["count"]) + int(right["count"])
    total = float(left["total"]) + float(right["total"])
    return {
        "count": count,
        "total": total,
        "min": min(float(left["min"]), float(right["min"])),
        "max": max(float(left["max"]), float(right["max"])),
        "mean": total / count,
    }


def _merge_graph_statistics(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, object]:
    architectures = sorted({*left["architectures"], *right["architectures"]})
    operator_ids = sorted({*left["operator_ids"], *right["operator_ids"]})
    return {
        "occurrence_count": int(left["occurrence_count"]) + int(right["occurrence_count"]),
        "source_graph_count": int(left["source_graph_count"]) + int(right["source_graph_count"]),
        "architectures": architectures,
        "operator_ids": operator_ids,
        "max_depth_position": max(int(left["max_depth_position"]), int(right["max_depth_position"])),
        "max_input_degree": max(int(left["max_input_degree"]), int(right["max_input_degree"])),
        "max_output_degree": max(int(left["max_output_degree"]), int(right["max_output_degree"])),
    }


def _unique_provenance(
    provenance: Sequence[VocabularySourceProvenance],
) -> tuple[VocabularySourceProvenance, ...]:
    by_key = {
        (
            item.source_model,
            item.model_version,
            item.extraction_timestamp,
            item.git_commit,
            item.framework_version,
        ): item
        for item in provenance
    }
    return tuple(by_key[key] for key in sorted(by_key))


def _mean(values: Iterable[float]) -> float:
    values_tuple = tuple(float(value) for value in values)
    return sum(values_tuple) / len(values_tuple)
