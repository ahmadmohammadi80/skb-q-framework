"""Main orchestration interface for Algorithm 1 in SKB-Q research.

The :class:`SKBQBridge` coordinates structural feature extraction, candidate
filtering, similarity scoring, confidence gating, embedding composition, and
deterministic fallback without relying on external services.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
import math

from skbq.bridge.candidate_filter import (
    BridgeCandidate,
    CandidateMatch,
    EmbeddingVector,
    deterministic_nearest_neighbor_fallback,
    select_top_k_nearest,
)
from skbq.bridge.embedding_composer import CompositionResult, EmbeddingComposer
from skbq.bridge.similarity import CandidateSimilarity, SimilarityContext, SimilarityScorer
from skbq.bridge.structural_features import (
    FeatureVector,
    OperatorStructuralMetadata,
    extract_operator_features,
)


@dataclass(frozen=True, slots=True)
class BridgeSource:
    """Source operator evidence consumed by the SKB-Q bridge."""

    identifier: str
    structural_metadata: OperatorStructuralMetadata | Mapping[str, object] | object
    embedding: EmbeddingVector | None = None
    semantic_embedding: EmbeddingVector | None = None
    functional_embedding: EmbeddingVector | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _coerce_embedding(self.embedding, "embedding"))
        object.__setattr__(
            self,
            "semantic_embedding",
            _coerce_embedding(self.semantic_embedding, "semantic_embedding"),
        )
        object.__setattr__(
            self,
            "functional_embedding",
            _coerce_embedding(self.functional_embedding, "functional_embedding"),
        )


@dataclass(frozen=True, slots=True)
class BridgeDecision:
    """Complete deterministic decision record emitted by the bridge."""

    source_identifier: str
    structural_features: FeatureVector
    candidate_matches: tuple[CandidateMatch, ...]
    similarities: tuple[CandidateSimilarity, ...]
    composition: CompositionResult
    selected_candidate: BridgeCandidate
    used_fallback: bool
    fallback_match: CandidateMatch | None = None


@dataclass(frozen=True, slots=True)
class SKBQBridge:
    """Orchestrate the deterministic SKB-Q bridge pipeline."""

    top_k: int = 5
    similarity_scorer: SimilarityScorer = field(default_factory=SimilarityScorer)
    embedding_composer: EmbeddingComposer = field(default_factory=EmbeddingComposer)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BridgeDecision:
        """Coordinate the full bridge workflow for a source item and candidates."""

        structural_features = self.extract_structural_features(source)
        candidate_matches = self.filter_candidates(structural_features, candidates)
        if not candidate_matches:
            raise ValueError("SKBQBridge requires at least one bridge candidate")

        similarities = self.score_similarity(source, candidate_matches)
        composition = self.compose_embedding(similarities)

        if self.passes_confidence_gate(composition):
            selected_candidate = composition.selected_candidate
            self.logger.info(
                "SKB-Q bridge selected candidate %s for source %s with confidence %.6f",
                selected_candidate.identifier,
                source.identifier,
                composition.confidence,
            )
            return BridgeDecision(
                source_identifier=source.identifier,
                structural_features=structural_features,
                candidate_matches=candidate_matches,
                similarities=similarities,
                composition=composition,
                selected_candidate=selected_candidate,
                used_fallback=False,
            )

        fallback_match = self.deterministic_fallback(structural_features, candidates)
        self.logger.info(
            "SKB-Q bridge used deterministic fallback candidate %s for source %s",
            fallback_match.candidate.identifier,
            source.identifier,
        )
        return BridgeDecision(
            source_identifier=source.identifier,
            structural_features=structural_features,
            candidate_matches=candidate_matches,
            similarities=similarities,
            composition=composition,
            selected_candidate=fallback_match.candidate,
            used_fallback=True,
            fallback_match=fallback_match,
        )

    def extract_structural_features(
        self,
        source: BridgeSource | OperatorStructuralMetadata | Mapping[str, object] | object,
    ) -> FeatureVector:
        """Prepare structural features for downstream bridge operations."""

        metadata = source.structural_metadata if isinstance(source, BridgeSource) else source
        features = extract_operator_features(metadata)
        self.logger.debug("extracted structural features: %s", features)
        return features

    def filter_candidates(
        self,
        structural_features: Sequence[float],
        candidates: Sequence[BridgeCandidate],
    ) -> tuple[CandidateMatch, ...]:
        """Select candidate items eligible for similarity scoring."""

        matches = select_top_k_nearest(
            query_features=structural_features,
            candidates=candidates,
            k=self.top_k,
        )
        self.logger.debug(
            "filtered candidates: %s",
            [(match.candidate.identifier, match.distance) for match in matches],
        )
        return matches

    def score_similarity(
        self,
        source: BridgeSource,
        candidates: Sequence[CandidateMatch],
    ) -> tuple[CandidateSimilarity, ...]:
        """Score filtered candidates against structural features."""

        context = SimilarityContext(
            embedding=source.embedding,
            semantic_embedding=source.semantic_embedding,
            functional_embedding=source.functional_embedding,
        )
        similarities = self.similarity_scorer.score_candidates(context, candidates)
        self.logger.debug(
            "scored candidates: %s",
            [
                (
                    similarity.candidate.identifier,
                    similarity.semantic,
                    similarity.structural,
                    similarity.functional,
                    similarity.aggregate,
                )
                for similarity in similarities
            ],
        )
        return similarities

    def passes_confidence_gate(
        self,
        composition: CompositionResult,
    ) -> bool:
        """Evaluate whether similarity scores satisfy the confidence gate."""

        passed = composition.passed_confidence_gate
        self.logger.debug(
            "confidence gate passed=%s confidence=%.6f entropy=%.6f",
            passed,
            composition.confidence,
            composition.entropy,
        )
        return passed

    def compose_embedding(
        self,
        similarities: Sequence[CandidateSimilarity],
    ) -> CompositionResult:
        """Compose the bridge embedding from structure and selected evidence."""

        composition = self.embedding_composer.compose(similarities)
        self.logger.debug(
            "composed embedding from candidates: %s",
            [
                (weight.candidate.identifier, weight.weight, weight.score)
                for weight in composition.candidate_weights
            ],
        )
        return composition

    def deterministic_fallback(
        self,
        structural_features: Sequence[float],
        candidates: Sequence[BridgeCandidate],
    ) -> CandidateMatch:
        """Provide deterministic fallback behavior when gating is not satisfied."""

        return deterministic_nearest_neighbor_fallback(structural_features, candidates)


def _coerce_embedding(values: Sequence[float] | None, field_name: str) -> EmbeddingVector | None:
    if values is None:
        return None
    if len(values) == 0:
        raise ValueError(f"{field_name} cannot be empty")

    embedding = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError(f"{field_name} values must be finite")
    return embedding
