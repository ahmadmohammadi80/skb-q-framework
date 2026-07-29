"""Frozen encoder interface for SKB-Q backbone experiments.

Concrete encoders are expected to be frozen during SKB-Q evaluation. This module
defines only the contract; it does not implement RAMP, GAMMA, or any trainable
encoding behavior. In Phase 4 notation, a frozen encoder returns immutable
vocabulary embeddings ``phi(v)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math

from skbq.bridge.structural_features import (
    OperatorStructuralMetadata,
    as_feature_vector,
    extract_operator_features,
)

VocabularyEmbedding = tuple[float, ...]
GraphEmbedding = tuple[float, ...]


class FrozenEncoder(ABC):
    """Abstract interface for frozen encoders producing ``phi(v)``."""

    @abstractmethod
    def encode(self, operator: object) -> VocabularyEmbedding:
        """Return a deterministic immutable vocabulary embedding ``phi(v)``."""


@dataclass(frozen=True, slots=True)
class EncodedOperator:
    """Operator paired with its immutable frozen vocabulary embedding."""

    operator_id: str
    embedding: VocabularyEmbedding
    encoder_name: str


@dataclass(frozen=True, slots=True)
class StructuralFeatureFrozenEncoder(FrozenEncoder):
    """Deterministic reference encoder using SKB-Q structural features as ``phi(v)``."""

    name: str = "structural_feature_phi"

    def encode(self, operator: object) -> VocabularyEmbedding:
        """Encode an operator from structural features or structural metadata."""

        if isinstance(operator, OperatorStructuralMetadata):
            return validate_embedding(extract_operator_features(operator))

        if hasattr(operator, "structural_features"):
            return validate_embedding(as_feature_vector(getattr(operator, "structural_features")))

        if hasattr(operator, "structural_metadata"):
            return validate_embedding(extract_operator_features(getattr(operator, "structural_metadata")))

        return validate_embedding(extract_operator_features(operator))

    def encode_operator(self, operator_id: str, operator: object) -> EncodedOperator:
        """Return an encoded operator record for traceable tests and integrations."""

        return EncodedOperator(
            operator_id=operator_id,
            embedding=self.encode(operator),
            encoder_name=self.name,
        )


def validate_embedding(embedding: Sequence[float]) -> GraphEmbedding:
    """Coerce an encoder output into the canonical immutable embedding type."""

    if len(embedding) == 0:
        raise ValueError("embedding cannot be empty")
    result = tuple(float(value) for value in embedding)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding values must be finite")
    return result


def mean_pool_embeddings(embeddings: Sequence[Sequence[float]]) -> GraphEmbedding:
    """Return deterministic mean-pooled graph embedding from ``phi(v)`` vectors."""

    if not embeddings:
        raise ValueError("cannot pool an empty embedding sequence")
    normalized = tuple(validate_embedding(embedding) for embedding in embeddings)
    dimension = len(normalized[0])
    if any(len(embedding) != dimension for embedding in normalized):
        raise ValueError("all embeddings must have equal dimensionality")
    return tuple(
        sum(embedding[index] for embedding in normalized) / len(normalized)
        for index in range(dimension)
    )
