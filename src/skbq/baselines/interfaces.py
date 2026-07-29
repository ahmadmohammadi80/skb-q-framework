"""Deterministic baseline interfaces for SKB-Q experimental infrastructure.

The baselines return selection decisions only. They do not compute benchmark
metrics, report results, or call external services.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import random
from typing import Protocol

from skbq.bridge.candidate_filter import (
    BridgeCandidate,
    deterministic_nearest_neighbor_fallback,
)
from skbq.bridge.similarity import EmbeddingSemanticSimilarity, SimilarityContext
from skbq.bridge.skbq_bridge import BridgeDecision, BridgeSource, SKBQBridge


@dataclass(frozen=True, slots=True)
class BaselineDecision:
    """Candidate-selection decision emitted by a baseline."""

    baseline_name: str
    selected_candidate: BridgeCandidate
    score: float | None = None
    details: Mapping[str, object] = field(default_factory=dict)


class BaselineRunner(Protocol):
    """Protocol shared by SKB-Q comparison baselines."""

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BaselineDecision:
        """Return a deterministic candidate-selection decision."""


@dataclass(frozen=True, slots=True)
class RandomFallbackBaseline:
    """Seeded random fallback baseline for reproducible comparison controls."""

    seed: int = 0
    name: str = "random_fallback"

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BaselineDecision:
        """Select a candidate by seeded pseudo-random index."""

        if not candidates:
            raise ValueError("RandomFallbackBaseline requires at least one candidate")

        rng = random.Random(f"{self.seed}:{source.identifier}:{len(candidates)}")
        index = rng.randrange(len(candidates))
        return BaselineDecision(
            baseline_name=self.name,
            selected_candidate=candidates[index],
            score=None,
            details={"seed": self.seed, "index": index},
        )


@dataclass(frozen=True, slots=True)
class StructuralNearestNeighborBaseline:
    """Baseline that selects the nearest structural vocabulary candidate."""

    name: str = "structural_nearest_neighbor"

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BaselineDecision:
        """Select by deterministic structural nearest-neighbor fallback."""

        match = deterministic_nearest_neighbor_fallback(
            query_features=source_features(source),
            candidates=candidates,
        )
        return BaselineDecision(
            baseline_name=self.name,
            selected_candidate=match.candidate,
            score=1.0 / (1.0 + match.distance),
            details={"distance": match.distance},
        )


@dataclass(frozen=True, slots=True)
class NonLLMSemanticBaseline:
    """Semantic baseline backed only by deterministic embedding similarity."""

    name: str = "non_llm_semantic"
    semantic_channel: EmbeddingSemanticSimilarity = field(
        default_factory=EmbeddingSemanticSimilarity
    )

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BaselineDecision:
        """Select the highest deterministic embedding-similarity candidate."""

        if not candidates:
            raise ValueError("NonLLMSemanticBaseline requires at least one candidate")

        context = SimilarityContext(
            embedding=source.embedding,
            semantic_embedding=source.semantic_embedding,
            functional_embedding=source.functional_embedding,
        )
        scored = tuple(
            (candidate, self.semantic_channel.score(context, candidate))
            for candidate in candidates
        )
        selected_candidate, selected_score = sorted(
            scored,
            key=lambda item: (-item[1], item[0].identifier),
        )[0]

        return BaselineDecision(
            baseline_name=self.name,
            selected_candidate=selected_candidate,
            score=selected_score,
            details={"scored_candidates": tuple((item[0].identifier, item[1]) for item in scored)},
        )


@dataclass(frozen=True, slots=True)
class SKBQFullBaseline:
    """Baseline adapter around the full deterministic SKB-Q bridge."""

    bridge: SKBQBridge = field(default_factory=SKBQBridge)
    name: str = "skbq_full"

    def run(
        self,
        source: BridgeSource,
        candidates: Sequence[BridgeCandidate],
    ) -> BaselineDecision:
        """Run the full bridge and expose it through the baseline interface."""

        decision = self.bridge.run(source, candidates)
        return BaselineDecision(
            baseline_name=self.name,
            selected_candidate=decision.selected_candidate,
            score=decision.composition.confidence,
            details=_bridge_details(decision),
        )


def source_features(source: BridgeSource) -> tuple[float, float, float, float, float, float]:
    """Extract source structural features for baselines without constructing a bridge."""

    return SKBQBridge().extract_structural_features(source)


def _bridge_details(decision: BridgeDecision) -> Mapping[str, object]:
    return {
        "used_fallback": decision.used_fallback,
        "confidence": decision.composition.confidence,
        "entropy": decision.composition.entropy,
        "selected_candidate": decision.selected_candidate.identifier,
    }
