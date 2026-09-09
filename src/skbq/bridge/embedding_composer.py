"""Embedding composition for SKB-Q bridge candidates.

The bridge composition follows ``psi(o) = sum_i w_i phi(v_i)`` where candidate
weights are produced by temperature-scaled softmax over similarity scores. The
module also exposes entropy and confidence-gate calculations for reproducible
decision analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from skbq.bridge.candidate_filter import BridgeCandidate, EmbeddingVector
from skbq.bridge.similarity import CandidateSimilarity


@dataclass(frozen=True, slots=True)
class CandidateWeight:
    """Softmax weight assigned to a candidate embedding."""

    candidate: BridgeCandidate
    weight: float
    score: float


@dataclass(frozen=True, slots=True)
class CompositionResult:
    """Result of weighted embedding composition and confidence evaluation."""

    embedding: EmbeddingVector
    candidate_weights: tuple[CandidateWeight, ...]
    entropy: float
    confidence: float
    passed_confidence_gate: bool

    @property
    def selected_candidate(self) -> BridgeCandidate:
        """Return the highest-weighted candidate with deterministic tie ordering."""

        return max(
            self.candidate_weights,
            key=lambda item: (item.weight, item.score, item.candidate.identifier),
        ).candidate


@dataclass(frozen=True, slots=True)
class EmbeddingComposer:
    """Compose candidate embeddings with softmax weights and confidence gating."""

    temperature: float = 1.0
    confidence_threshold: float = 0.5
    max_entropy: float | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_entropy is not None and (
            self.max_entropy < 0.0 or not math.isfinite(self.max_entropy)
        ):
            raise ValueError("max_entropy must be finite and non-negative")

    def compose(self, similarities: Sequence[CandidateSimilarity]) -> CompositionResult:
        """Compute ``psi(o)`` from candidate similarities and embeddings."""

        if not similarities:
            raise ValueError("cannot compose embeddings without candidate similarities")

        scores = tuple(similarity.aggregate for similarity in similarities)
        weights = softmax(scores, temperature=self.temperature)
        embeddings = tuple(
            _require_embedding(similarity.candidate) for similarity in similarities
        )
        composed_embedding = weighted_embedding_sum(embeddings, weights)
        uncertainty = entropy(weights)
        confidence = max(weights)
        passed_gate = passes_confidence_gate(
            weights=weights,
            confidence_threshold=self.confidence_threshold,
            max_entropy=self.max_entropy,
        )

        return CompositionResult(
            embedding=composed_embedding,
            candidate_weights=tuple(
                CandidateWeight(
                    candidate=similarities[index].candidate,
                    weight=weights[index],
                    score=scores[index],
                )
                for index in range(len(similarities))
            ),
            entropy=uncertainty,
            confidence=confidence,
            passed_confidence_gate=passed_gate,
        )


def softmax(scores: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Return temperature-scaled softmax weights for finite scores."""

    if not scores:
        raise ValueError("softmax requires at least one score")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")

    scaled_scores = tuple(_finite_float(score, "score") / temperature for score in scores)
    max_score = max(scaled_scores)
    exponentials = tuple(math.exp(score - max_score) for score in scaled_scores)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy for a probability vector using natural logarithms."""

    if not probabilities:
        raise ValueError("entropy requires at least one probability")

    total = sum(_finite_float(probability, "probability") for probability in probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("probabilities cannot be negative")

    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def passes_confidence_gate(
    weights: Sequence[float],
    confidence_threshold: float,
    max_entropy: float | None = None,
) -> bool:
    """Evaluate the confidence gate from softmax weights and optional entropy cap."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")

    confidence = max(weights)
    entropy_value = entropy(weights)
    entropy_ok = max_entropy is None or entropy_value <= max_entropy
    return confidence >= confidence_threshold and entropy_ok


def weighted_embedding_sum(
    embeddings: Sequence[EmbeddingVector],
    weights: Sequence[float],
) -> EmbeddingVector:
    """Return ``sum_i w_i phi(v_i)`` for equal-dimensional embeddings."""

    if len(embeddings) != len(weights):
        raise ValueError("embedding and weight counts must match")
    if not embeddings:
        raise ValueError("weighted sum requires at least one embedding")

    dimension = len(embeddings[0])
    if dimension == 0:
        raise ValueError("embeddings cannot be empty")
    if any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("all embeddings must have equal dimensionality")

    return tuple(
        sum(weights[index] * embeddings[index][dimension_index] for index in range(len(embeddings)))
        for dimension_index in range(dimension)
    )


def _require_embedding(candidate: BridgeCandidate) -> EmbeddingVector:
    if candidate.embedding is None:
        raise ValueError(f"candidate {candidate.identifier!r} is missing phi(v_i) embedding")
    return candidate.embedding


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
