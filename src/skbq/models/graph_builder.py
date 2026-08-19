"""Build SKB-Q OperatorGraph objects from Hugging Face transformer modules."""

from __future__ import annotations

from dataclasses import dataclass

from skbq.bridge.structural_features import extract_operator_features
from skbq.graph.operator_graph import OperatorGraph, OperatorNode
from skbq.models.hf_adapter import HFModuleRecord, iter_hf_module_records
from skbq.models.operator_mapping import detect_supported_architecture, map_operator_type


@dataclass(frozen=True, slots=True)
class HuggingFaceGraphBuilder:
    """Convert supported Hugging Face transformer module trees into OperatorGraphs."""

    include_root: bool = True

    def build(self, model: object, config: object | None = None) -> OperatorGraph:
        """Build an OperatorGraph from a loaded Hugging Face model."""

        architecture = detect_supported_architecture(config or getattr(model, "config", model))
        records = iter_hf_module_records(model)
        if not self.include_root:
            records = tuple(record for record in records if record.module_path)
        if not records:
            raise ValueError("cannot build OperatorGraph from an empty module tree")

        record_by_path = {record.module_path: record for record in records}
        nodes = tuple(
            self._node_from_record(record, record_by_path)
            for record in records
        )
        graph = OperatorGraph(nodes=nodes, architecture=architecture)
        _validate_feature_generation(graph)
        return graph

    def _node_from_record(
        self,
        record: HFModuleRecord,
        record_by_path: dict[str, HFModuleRecord],
    ) -> OperatorNode:
        operator_id = _operator_id(record.module_path)
        parent_ids = (
            ()
            if record.parent_path is None or record.parent_path not in record_by_path
            else (_operator_id(record.parent_path),)
        )
        child_ids = tuple(
            _operator_id(child_path)
            for child_path in record.child_paths
            if child_path in record_by_path
        )
        return OperatorNode(
            operator_id=operator_id,
            operator_type=map_operator_type(record.module_path, record.module),
            parent_ids=parent_ids,
            child_ids=child_ids,
            parameter_count=record.parameter_count,
            tensor_shapes=record.tensor_shapes,
            depth_position=record.depth,
            branch_count=max(len(child_ids), 1),
            has_nonlinearity=_has_nonlinearity(record),
        )


def _operator_id(module_path: str) -> str:
    return module_path if module_path else "__model__"


def _has_nonlinearity(record: HFModuleRecord) -> bool:
    lowered = f"{record.module_path} {record.class_name}".casefold()
    return any(marker in lowered for marker in ("silu", "gelu", "relu", "activation", "swiglu", "mlp"))


def _validate_feature_generation(graph: OperatorGraph) -> None:
    for metadata in graph.structural_metadata().values():
        extract_operator_features(metadata)
