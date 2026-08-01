"""AgentQ state representation built from existing SKB-Q graph and encoder APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from skbq.backbone.encoder import FrozenEncoder, StructuralFeatureFrozenEncoder
from skbq.bridge.structural_features import FeatureVector, extract_operator_features
from skbq.graph.operator_graph import OperatorGraph


@dataclass(frozen=True, slots=True)
class OperatorState:
    """Per-operator state for AgentQ policy inference."""

    operator_id: str
    layer_id: str
    operator_type: str
    structural_features: FeatureVector
    parameter_count: int
    encoder_embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_key(self.operator_id, "operator_id")
        _validate_key(self.layer_id, "layer_id")
        _validate_key(self.operator_type, "operator_type")
        if not isinstance(self.parameter_count, int) or isinstance(self.parameter_count, bool):
            raise TypeError("parameter_count must be an integer")
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if self.encoder_embedding is not None:
            object.__setattr__(
                self,
                "encoder_embedding",
                tuple(float(value) for value in self.encoder_embedding),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable operator state."""

        return {
            "operator_id": self.operator_id,
            "layer_id": self.layer_id,
            "operator_type": self.operator_type,
            "structural_features": list(self.structural_features),
            "parameter_count": self.parameter_count,
            "encoder_embedding": (
                None if self.encoder_embedding is None else list(self.encoder_embedding)
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GraphState:
    """Collection of operator states with graph-level metadata."""

    operators: tuple[OperatorState, ...]
    architecture: str
    graph_identifier: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("GraphState requires at least one operator")
        normalized_operators = tuple(sorted(self.operators, key=lambda item: item.operator_id))
        operator_ids = [operator.operator_id for operator in normalized_operators]
        if len(set(operator_ids)) != len(operator_ids):
            raise ValueError("operator ids must be unique within GraphState")
        if not self.architecture.strip():
            raise ValueError("architecture cannot be empty")
        if not self.graph_identifier.strip():
            raise ValueError("graph_identifier cannot be empty")
        object.__setattr__(self, "operators", normalized_operators)
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    @property
    def operator_ids(self) -> tuple[str, ...]:
        """Return operator ids in deterministic sorted order."""

        return tuple(operator.operator_id for operator in self.operators)

    @property
    def node_count(self) -> int:
        """Return number of operators in this graph state."""

        return len(self.operators)

    def get(self, operator_id: str) -> OperatorState:
        """Return operator state by id."""

        for operator in self.operators:
            if operator.operator_id == operator_id:
                return operator
        raise KeyError(f"unknown operator id: {operator_id}")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable graph state."""

        return {
            "operators": [operator.to_mapping() for operator in self.operators],
            "architecture": self.architecture,
            "graph_identifier": self.graph_identifier,
            "metadata": dict(self.metadata),
            "node_count": self.node_count,
            "operator_ids": list(self.operator_ids),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def state_hash(self) -> str:
        """Return stable SHA-256 hash for this graph state."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StateBuilder:
    """Build AgentQ graph state from ``OperatorGraph`` via existing SKB-Q APIs."""

    encoder: FrozenEncoder = StructuralFeatureFrozenEncoder()
    include_encoder_embeddings: bool = True

    def build(self, graph: OperatorGraph) -> GraphState:
        """Convert an operator graph into AgentQ state representation."""

        structural_metadata = graph.structural_metadata()
        operators: list[OperatorState] = []
        for node in graph.nodes:
            metadata = structural_metadata[node.operator_id]
            structural_features = extract_operator_features(metadata)
            encoder_embedding: tuple[float, ...] | None = None
            if self.include_encoder_embeddings:
                encoder_embedding = self.encoder.encode(metadata)
            operators.append(
                OperatorState(
                    operator_id=node.operator_id,
                    layer_id=str(node.depth_position),
                    operator_type=node.operator_type,
                    structural_features=structural_features,
                    parameter_count=node.parameter_count,
                    encoder_embedding=encoder_embedding,
                    metadata={
                        "depth_position": node.depth_position,
                        "input_degree": node.input_degree,
                        "output_degree": node.output_degree,
                        "branch_group": node.branch_group,
                        "has_nonlinearity": node.has_nonlinearity,
                        "has_multi_branch_routing": node.has_multi_branch_routing,
                    },
                )
            )

        return GraphState(
            operators=tuple(operators),
            architecture=graph.architecture,
            graph_identifier=_graph_identifier(graph),
            metadata={
                "max_depth_position": graph.max_depth_position,
                "max_input_degree": graph.max_input_degree,
                "max_output_degree": graph.max_output_degree,
                "reference_parameter_count": graph.reference_parameter_count,
            },
        )


def _graph_identifier(graph: OperatorGraph) -> str:
    payload = json.dumps(
        {
            "architecture": graph.architecture,
            "node_ids": list(graph.node_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{graph.architecture}:{digest}"


def _validate_key(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
