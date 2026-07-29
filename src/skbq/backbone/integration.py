"""Integration helpers connecting frozen backbones to the SKB-Q bridge.

This module wires frozen encoder outputs ``phi(v)`` into bridge candidates and
applies frozen policies ``pi_theta`` to pooled graph embeddings. It does not
implement model internals, quantization algorithms, or experimental execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from skbq.backbone.encoder import (
    FrozenEncoder,
    GraphEmbedding,
    StructuralFeatureFrozenEncoder,
    mean_pool_embeddings,
)
from skbq.backbone.policy_interface import FrozenPolicy, PolicyAllocation, UniformFrozenPolicy
from skbq.bridge.candidate_filter import BridgeCandidate
from skbq.graph.operator_graph import OperatorGraph


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

    def encode_graph(self, graph: OperatorGraph) -> GraphEmbedding:
        """Return pooled graph embedding ``G`` from operator-level ``phi(v)`` values."""

        metadata_by_operator = graph.structural_metadata()
        embeddings = tuple(
            self.encoder.encode(metadata_by_operator[operator_id])
            for operator_id in graph.node_ids
        )
        return mean_pool_embeddings(embeddings)

    def allocate_graph(self, graph: OperatorGraph, budget: float) -> PolicyAllocation:
        """Apply frozen policy ``pi_theta`` to graph embedding ``G``."""

        return self.policy.allocate(self.encode_graph(graph), budget)
