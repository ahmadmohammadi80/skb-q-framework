"""Backbone abstractions for studying encoder and policy interfaces in SKB-Q."""

from skbq.backbone.encoder import (
    EncodedOperator,
    FrozenEncoder,
    GraphEmbedding,
    StructuralFeatureFrozenEncoder,
    VocabularyEmbedding,
    mean_pool_embeddings,
    validate_embedding,
)
from skbq.backbone.integration import (
    FrozenBackbone,
    GraphEncodingResult,
    allocation_targets_for_graph,
)
from skbq.backbone.policy_interface import (
    AllocationTarget,
    FrozenPolicy,
    PolicyAllocation,
    TargetAllocation,
    UniformFrozenPolicy,
    validate_policy_inputs,
)
from skbq.backbone.provenance import EncoderProvenance, PolicyProvenance

__all__ = [
    "AllocationTarget",
    "EncodedOperator",
    "EncoderProvenance",
    "FrozenBackbone",
    "FrozenEncoder",
    "FrozenPolicy",
    "GraphEmbedding",
    "GraphEncodingResult",
    "PolicyAllocation",
    "PolicyProvenance",
    "StructuralFeatureFrozenEncoder",
    "TargetAllocation",
    "UniformFrozenPolicy",
    "VocabularyEmbedding",
    "allocation_targets_for_graph",
    "mean_pool_embeddings",
    "validate_embedding",
    "validate_policy_inputs",
]
