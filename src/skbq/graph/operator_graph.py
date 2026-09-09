"""Model-agnostic operator graph representation for SKB-Q extraction.

The graph layer records topology, structural metadata, tensor shape annotations,
and branching information without depending on any model framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.bridge.structural_features import OperatorStructuralMetadata

Shape = tuple[int | str, ...]
TensorShapeMetadata = Mapping[str, Shape]


@dataclass(frozen=True, slots=True)
class OperatorNode:
    """One operator in a model-agnostic computational graph."""

    operator_id: str
    operator_type: str
    parent_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    parameter_count: int = 1
    tensor_shapes: TensorShapeMetadata = field(default_factory=dict)
    depth_position: int = 0
    branch_group: str | None = None
    branch_index: int | None = None
    branch_count: int = 1
    is_branch_merge: bool = False
    has_nonlinearity: bool = False

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ValueError("operator_id cannot be empty")
        if not self.operator_type.strip():
            raise ValueError("operator_type cannot be empty")
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if self.depth_position < 0:
            raise ValueError("depth_position must be non-negative")
        if self.branch_count <= 0:
            raise ValueError("branch_count must be positive")
        if self.branch_index is not None and not 0 <= self.branch_index < self.branch_count:
            raise ValueError("branch_index must be within branch_count")

        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "child_ids", tuple(self.child_ids))
        object.__setattr__(self, "tensor_shapes", MappingProxyType(_normalize_shapes(self.tensor_shapes)))

    @property
    def input_degree(self) -> int:
        """Return the number of parent operators."""

        return len(self.parent_ids)

    @property
    def output_degree(self) -> int:
        """Return the number of child operators."""

        return len(self.child_ids)

    @property
    def has_multi_branch_routing(self) -> bool:
        """Return whether this node participates in multi-branch routing."""

        return self.branch_count > 1 or self.output_degree > 1 or self.is_branch_merge


@dataclass(frozen=True, slots=True)
class OperatorGraph:
    """Immutable operator graph with structural metadata extraction helpers."""

    nodes: tuple[OperatorNode, ...]
    architecture: str = "unknown"

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("OperatorGraph requires at least one node")
        node_map = {node.operator_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise ValueError("operator ids must be unique")

        for node in self.nodes:
            self._validate_edges(node, node_map)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return operator ids in graph order."""

        return tuple(node.operator_id for node in self.nodes)

    def get(self, operator_id: str) -> OperatorNode:
        """Return a node by operator id."""

        for node in self.nodes:
            if node.operator_id == operator_id:
                return node
        raise KeyError(f"unknown operator id: {operator_id}")

    def roots(self) -> tuple[OperatorNode, ...]:
        """Return graph nodes with no parents."""

        return tuple(node for node in self.nodes if not node.parent_ids)

    def leaves(self) -> tuple[OperatorNode, ...]:
        """Return graph nodes with no children."""

        return tuple(node for node in self.nodes if not node.child_ids)

    def structural_metadata(self) -> dict[str, OperatorStructuralMetadata]:
        """Return SKB-Q structural metadata for every operator."""

        return {
            node.operator_id: self.structural_metadata_for(node.operator_id)
            for node in self.nodes
        }

    def structural_metadata_for(self, operator_id: str) -> OperatorStructuralMetadata:
        """Return SKB-Q structural metadata for one operator."""

        node = self.get(operator_id)
        return OperatorStructuralMetadata(
            parameter_count=max(float(node.parameter_count), 1.0),
            reference_parameter_count=self.reference_parameter_count,
            input_degree=float(node.input_degree),
            output_degree=float(node.output_degree),
            depth_index=float(node.depth_position),
            max_input_degree=float(self.max_input_degree),
            max_output_degree=float(self.max_output_degree),
            max_depth_index=float(self.max_depth_position),
            has_nonlinearity=node.has_nonlinearity,
            has_multi_branch_routing=node.has_multi_branch_routing,
        )

    @property
    def max_input_degree(self) -> int:
        """Return maximum input degree across graph nodes."""

        return max(node.input_degree for node in self.nodes)

    @property
    def max_output_degree(self) -> int:
        """Return maximum output degree across graph nodes."""

        return max(node.output_degree for node in self.nodes)

    @property
    def max_depth_position(self) -> int:
        """Return maximum depth position across graph nodes."""

        return max(node.depth_position for node in self.nodes)

    @property
    def reference_parameter_count(self) -> float:
        """Return positive graph-level reference count for log parameter ratios."""

        return max(max(float(node.parameter_count) for node in self.nodes), 1.0)

    @staticmethod
    def _validate_edges(
        node: OperatorNode,
        node_map: Mapping[str, OperatorNode],
    ) -> None:
        for parent_id in node.parent_ids:
            if parent_id not in node_map:
                raise KeyError(f"unknown parent {parent_id!r} for node {node.operator_id!r}")
            if node.operator_id not in node_map[parent_id].child_ids:
                raise ValueError(
                    f"parent {parent_id!r} does not reference child {node.operator_id!r}"
                )

        for child_id in node.child_ids:
            if child_id not in node_map:
                raise KeyError(f"unknown child {child_id!r} for node {node.operator_id!r}")
            if node.operator_id not in node_map[child_id].parent_ids:
                raise ValueError(
                    f"child {child_id!r} does not reference parent {node.operator_id!r}"
                )


def _normalize_shapes(shapes: TensorShapeMetadata) -> dict[str, Shape]:
    normalized: dict[str, Shape] = {}
    for name, shape in shapes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tensor shape names must be non-empty strings")
        normalized[name] = _normalize_shape(shape)
    return normalized


def _normalize_shape(shape: Sequence[int | str]) -> Shape:
    normalized = tuple(shape)
    for dimension in normalized:
        if isinstance(dimension, bool) or not isinstance(dimension, int | str):
            raise TypeError("shape dimensions must be integers or symbolic strings")
        if isinstance(dimension, int) and dimension <= 0:
            raise ValueError("integer shape dimensions must be positive")
        if isinstance(dimension, str) and not dimension.strip():
            raise ValueError("symbolic shape dimensions cannot be empty")
    return normalized
