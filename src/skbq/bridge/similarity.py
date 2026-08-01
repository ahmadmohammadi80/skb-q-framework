"""Similarity channels for SKB-Q bridge scoring.

The bridge combines three channels:

* semantic similarity through an interface that can later be backed by an LLM;
* structural similarity derived from normalized structural distance;
* functional similarity through a separate interface for behavior-level evidence.

Only deterministic local baselines are implemented here. No external model or API
is contacted by these classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol

from skbq.bridge.candidate_filter import BridgeCandidate, CandidateMatch, EmbeddingVector


@dataclass(frozen=True, slots=True)
class SimilarityContext:
    """Source-side evidence available to similarity channels."""

    embedding: EmbeddingVector | None = None
    semantic_embedding: EmbeddingVector | None = None
    functional_embedding: EmbeddingVector | None = None


@dataclass(frozen=True, slots=True)
class CandidateSimilarity:
    """Per-channel and aggregate similarity scores for one candidate."""

    candidate: BridgeCandidate
    semantic: float
    structural: float
    functional: float
    aggregate: float
    structural_distance: float


class SemanticSimilarity(Protocol):
    """Interface for semantic similarity providers, including future LLM adapters."""

    def score(self, context: SimilarityContext, candidate: BridgeCandidate) -> float:
        """Return a semantic score in ``[0, 1]``."""


class FunctionalSimilarity(Protocol):
    """Interface for functional similarity providers."""

    def score(self, context: SimilarityContext, candidate: BridgeCandidate) -> float:
        """Return a functional score in ``[0, 1]``."""


@dataclass(frozen=True, slots=True)
class SimilarityWeights:
    """Aggregation weights for semantic, structural, and functional channels."""

    semantic: float = 1.0
    structural: float = 1.0
    functional: float = 1.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("semantic", self.semantic),
            ("structural", self.structural),
            ("functional", self.functional),
        ):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} similarity weight must be finite and non-negative")
        if self.total == 0.0:
            raise ValueError("at least one similarity weight must be positive")

    @property
    def total(self) -> float:
        """Return the unnormalized total channel weight."""

        return self.semantic + self.structural + self.functional


@dataclass(frozen=True, slots=True)
class DeterministicEmbeddingSimilarity:
    """Deterministic cosine-similarity baseline for embedding comparisons."""

    normalize_to_unit_interval: bool = True

    def score_embeddings(
        self,
        left: Sequence[float] | None,
        right: Sequence[float] | None,
    ) -> float:
        """Score two embeddings deterministically without external dependencies."""

        if left is None or right is None:
            return 0.0

        cosine = cosine_similarity(left, right)
        if not self.normalize_to_unit_interval:
            return cosine
        return (cosine + 1.0) / 2.0


@dataclass(frozen=True, slots=True)
class EmbeddingSemanticSimilarity:
    """Semantic channel using deterministic embeddings until external adapters exist."""

    baseline: DeterministicEmbeddingSimilarity = field(
        default_factory=DeterministicEmbeddingSimilarity
    )

    def score(self, context: SimilarityContext, candidate: BridgeCandidate) -> float:
        """Score source and candidate semantic evidence."""

        source_embedding = context.semantic_embedding or context.embedding
        candidate_embedding = candidate.semantic_embedding or candidate.embedding
        return self.baseline.score_embeddings(source_embedding, candidate_embedding)


@dataclass(frozen=True, slots=True)
class EmbeddingFunctionalSimilarity:
    """Functional channel using deterministic embeddings until richer evidence exists."""

    baseline: DeterministicEmbeddingSimilarity = field(
        default_factory=DeterministicEmbeddingSimilarity
    )

    def score(self, context: SimilarityContext, candidate: BridgeCandidate) -> float:
        """Score source and candidate functional evidence."""

        source_embedding = context.functional_embedding or context.embedding
        candidate_embedding = candidate.functional_embedding or candidate.embedding
        return self.baseline.score_embeddings(source_embedding, candidate_embedding)


@dataclass(frozen=True, slots=True)
class SimilarityScorer:
    """Combine semantic, structural, and functional channels for candidates."""

    semantic_channel: SemanticSimilarity = field(default_factory=EmbeddingSemanticSimilarity)
    functional_channel: FunctionalSimilarity = field(default_factory=EmbeddingFunctionalSimilarity)
    weights: SimilarityWeights = field(default_factory=SimilarityWeights)

    def score_candidate(
        self,
        context: SimilarityContext,
        match: CandidateMatch,
    ) -> CandidateSimilarity:
        """Return per-channel and aggregate similarity for one candidate match."""

        semantic = _clamp_score(self.semantic_channel.score(context, match.candidate), "semantic")
        structural = structural_similarity(match.distance)
        functional = _clamp_score(
            self.functional_channel.score(context, match.candidate),
            "functional",
        )
        aggregate = (
            self.weights.semantic * semantic
            + self.weights.structural * structural
            + self.weights.functional * functional
        ) / self.weights.total

        return CandidateSimilarity(
            candidate=match.candidate,
            semantic=semantic,
            structural=structural,
            functional=functional,
            aggregate=aggregate,
            structural_distance=match.distance,
        )

    def score_candidates(
        self,
        context: SimilarityContext,
        matches: Sequence[CandidateMatch],
    ) -> tuple[CandidateSimilarity, ...]:
        """Score candidate matches in a deterministic order."""

        return tuple(self.score_candidate(context, match) for match in matches)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for equal-length, finite vectors."""

    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal dimensionality")
    if not left:
        raise ValueError("embedding vectors cannot be empty")

    left_vector = _coerce_vector(left, "left")
    right_vector = _coerce_vector(right, "right")
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(
        left_vector[index] * right_vector[index] for index in range(len(left_vector))
    )
    return dot_product / (left_norm * right_norm)


def structural_similarity(distance: float) -> float:
    """Convert a non-negative structural distance into a bounded similarity."""

    if distance < 0.0 or not math.isfinite(distance):
        raise ValueError("structural distance must be finite and non-negative")
    return 1.0 / (1.0 + distance)


def _coerce_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} embedding values must be finite")
    return vector


def _clamp_score(score: float, channel_name: str) -> float:
    if not math.isfinite(score):
        raise ValueError(f"{channel_name} similarity score must be finite")
    return min(1.0, max(0.0, score))
