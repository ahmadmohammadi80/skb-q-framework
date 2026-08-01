"""Integration helpers connecting frozen backbones to the SKB-Q bridge.

This module wires frozen encoder outputs ``phi(v)`` into bridge candidates and
applies frozen policies ``pi_theta`` to pooled graph embeddings. It does not
implement model internals, quantization algorithms, or experimental execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from skbq.backbone.encoder import (
    FrozenEncoder,
    GraphEmbedding,
    StructuralFeatureFrozenEncoder,
    mean_pool_embeddings,
)
from skbq.backbone.policy_interface import (
    AllocationTarget,
    FrozenPolicy,
    PolicyAllocation,
    UniformFrozenPolicy,
)
from skbq.backbone.provenance import EncoderProvenance
from skbq.bridge.candidate_filter import BridgeCandidate
from skbq.graph.operator_graph import OperatorGraph


@dataclass(frozen=True, slots=True)
class GraphEncodingResult:
    """Deterministic graph encoding result for pooled operator embeddings."""

    embedding: GraphEmbedding
    encoder_provenance: EncoderProvenance
    pooling_metadata: Mapping[str, object]
    graph_identifier: str
    node_count: int

    def __post_init__(self) -> None:
        if not self.graph_identifier.strip():
            raise ValueError("graph_identifier cannot be empty")
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        object.__setattr__(self, "pooling_metadata", MappingProxyType(dict(self.pooling_metadata)))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable graph encoding mapping."""

        return {
            "embedding": list(self.embedding),
            "encoder_provenance": self.encoder_provenance.to_mapping(),
            "pooling_metadata": dict(sorted(self.pooling_metadata.items())),
            "graph_identifier": self.graph_identifier,
            "node_count": self.node_count,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def encoding_hash(self) -> str:
        """Return stable SHA-256 hash for this graph encoding."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenBackbone:
    """Frozen backbone bundle exposing encoder ``phi`` and policy ``pi_theta``."""

    encoder: FrozenEncoder = field(default_factory=StructuralFeatureFrozenEncoder)
    policy: FrozenPolicy = field(default_factory=UniformFrozenPolicy)

    def encode_candidate(self, candidate: BridgeCandidate) -> BridgeCandidate:
        """Return a bridge candidate with immutable ``phi(v)`` embedding attached."""

        embedding = self.encoder.encode(candidate)
        metadata = dict(candidate.metadata)
        metadata["encoder"] = self.encoder.__class__.__name__
        metadata["embedding_notation"] = "phi(v)"
        return BridgeCandidate(
            identifier=candidate.identifier,
            structural_features=candidate.structural_features,
            embedding=embedding,
            semantic_embedding=candidate.semantic_embedding,
            functional_embedding=candidate.functional_embedding,
            metadata=metadata,
        )

    def encode_candidates(
        self,
        candidates: Sequence[BridgeCandidate],
    ) -> tuple[BridgeCandidate, ...]:
        """Return bridge candidates with frozen vocabulary embeddings attached."""

        return tuple(self.encode_candidate(candidate) for candidate in candidates)

    def encode_graph(self, graph: OperatorGraph) -> GraphEncodingResult:
        """Return graph encoding result ``G`` from operator-level ``phi(v)`` values."""

        metadata_by_operator = graph.structural_metadata()
        embeddings = tuple(
            self.encoder.encode(metadata_by_operator[operator_id])
            for operator_id in graph.node_ids
        )
        embedding = mean_pool_embeddings(embeddings)
        return GraphEncodingResult(
            embedding=embedding,
            encoder_provenance=_encoder_provenance(self.encoder),
            pooling_metadata={
                "pooling": "mean",
                "node_ids": list(graph.node_ids),
                "embedding_count": len(embeddings),
                "embedding_dim": len(embedding),
            },
            graph_identifier=_graph_identifier(graph),
            node_count=len(graph.nodes),
        )

    def allocate_graph(self, graph: OperatorGraph, budget: float) -> PolicyAllocation:
        """Apply frozen policy ``pi_theta`` to graph embedding ``G``."""

        encoding = self.encode_graph(graph)
        return self.policy.allocate(
            encoding.embedding,
            budget,
            targets=allocation_targets_for_graph(graph),
        )


def allocation_targets_for_graph(graph: OperatorGraph) -> tuple[AllocationTarget, ...]:
    """Return deterministic operator-level allocation targets for a graph."""

    return tuple(
        AllocationTarget(
            target_id=node.operator_id,
            target_type="graph_operator",
            operator_id=node.operator_id,
            layer_id=str(node.depth_position),
            metadata={
                "operator_type": node.operator_type,
                "depth_position": node.depth_position,
            },
        )
        for node in graph.nodes
    )


def _encoder_provenance(encoder: FrozenEncoder) -> EncoderProvenance:
    provenance = getattr(encoder, "provenance", None)
    if isinstance(provenance, EncoderProvenance):
        return provenance
    return EncoderProvenance(encoder_id=encoder.__class__.__name__)


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
