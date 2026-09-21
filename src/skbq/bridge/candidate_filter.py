"""Candidate filtering for structural nearest-neighbor selection in SKB-Q.

This module normalizes vocabulary-side structural features with z-scores,
computes Euclidean distance in the normalized feature space, selects top-k
nearest candidates, and exposes a deterministic nearest-neighbor fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from skbq.bridge.structural_features import FeatureVector, as_feature_vector

EmbeddingVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    """A vocabulary candidate available to the structural knowledge bridge."""

    identifier: str
    structural_features: FeatureVector
    embedding: EmbeddingVector | None = None
    semantic_embedding: EmbeddingVector | None = None
    functional_embedding: EmbeddingVector | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_features", as_feature_vector(self.structural_features))
        object.__setattr__(self, "embedding", _as_embedding(self.embedding, "embedding"))
        object.__setattr__(
            self,
            "semantic_embedding",
            _as_embedding(self.semantic_embedding, "semantic_embedding"),
        )
        object.__setattr__(
            self,
            "functional_embedding",
            _as_embedding(self.functional_embedding, "functional_embedding"),
        )


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """A candidate and its normalized structural distance from the source."""

    candidate: BridgeCandidate
    distance: float
    normalized_query: FeatureVector
    normalized_candidate: FeatureVector
    vocabulary_index: int


@dataclass(frozen=True, slots=True)
class ZScoreNormalizer:
    """Feature-wise z-score normalizer fitted on vocabulary candidates."""

    means: FeatureVector
    standard_deviations: FeatureVector

    @classmethod
    def fit(cls, feature_vectors: Sequence[FeatureVector]) -> ZScoreNormalizer:
        """Fit normalization statistics from vocabulary structural features."""

        if not feature_vectors:
            raise ValueError("cannot fit z-score normalizer without candidates")

        vectors = [as_feature_vector(vector) for vector in feature_vectors]
        means = tuple(
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        )
        variances = tuple(
            sum((vector[index] - means[index]) ** 2 for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        )
        standard_deviations = tuple(math.sqrt(variance) for variance in variances)

        return cls(
            means=as_feature_vector(means),
            standard_deviations=as_feature_vector(standard_deviations),
        )

    def transform(self, feature_vector: Sequence[float]) -> FeatureVector:
        """Normalize one feature vector with fitted vocabulary statistics."""

        vector = as_feature_vector(feature_vector)
        normalized = tuple(
            0.0
            if self.standard_deviations[index] == 0.0
            else (vector[index] - self.means[index]) / self.standard_deviations[index]
            for index in range(len(vector))
        )
        return as_feature_vector(normalized)


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Euclidean distance between two structural feature vectors."""

    left_vector = as_feature_vector(left)
    right_vector = as_feature_vector(right)
    return math.sqrt(
        sum((left_vector[index] - right_vector[index]) ** 2 for index in range(len(left_vector)))
    )


def select_top_k_nearest(
    query_features: Sequence[float],
    candidates: Sequence[BridgeCandidate],
    k: int,
) -> tuple[CandidateMatch, ...]:
    """Select the top-k nearest candidates after vocabulary z-score normalization."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not candidates:
        return ()

    normalizer = ZScoreNormalizer.fit([candidate.structural_features for candidate in candidates])
    normalized_query = normalizer.transform(query_features)
    matches = tuple(
        CandidateMatch(
            candidate=candidate,
            distance=euclidean_distance(
                normalized_query,
                normalizer.transform(candidate.structural_features),
            ),
            normalized_query=normalized_query,
            normalized_candidate=normalizer.transform(candidate.structural_features),
            vocabulary_index=index,
        )
        for index, candidate in enumerate(candidates)
    )

    ranked = sorted(
        matches,
        key=lambda match: (match.distance, match.candidate.identifier, match.vocabulary_index),
    )
    return tuple(ranked[:k])


def deterministic_nearest_neighbor_fallback(
    query_features: Sequence[float],
    candidates: Sequence[BridgeCandidate],
) -> CandidateMatch:
    """Return the deterministic nearest structural neighbor for fallback use."""

    matches = select_top_k_nearest(query_features=query_features, candidates=candidates, k=1)
    if not matches:
        raise ValueError("cannot select fallback without candidates")
    return matches[0]


def _as_embedding(values: Sequence[float] | None, field_name: str) -> EmbeddingVector | None:
    if values is None:
        return None
    if len(values) == 0:
        raise ValueError(f"{field_name} cannot be empty")

    embedding = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError(f"{field_name} values must be finite")
    return embedding
