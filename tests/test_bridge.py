"""Unit tests for the deterministic SKB-Q bridge implementation."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.bridge.candidate_filter import (  # noqa: E402
    BridgeCandidate,
    deterministic_nearest_neighbor_fallback,
    select_top_k_nearest,
)
from skbq.bridge.embedding_composer import EmbeddingComposer, entropy, softmax  # noqa: E402
from skbq.bridge.similarity import (  # noqa: E402
    CandidateSimilarity,
    DeterministicEmbeddingSimilarity,
    structural_similarity,
)
from skbq.bridge.skbq_bridge import BridgeSource, SKBQBridge  # noqa: E402
from skbq.bridge.structural_features import (  # noqa: E402
    OperatorStructuralMetadata,
    extract_operator_features,
)


class StructuralFeatureTests(unittest.TestCase):
    def test_extract_operator_features_returns_six_reproducible_values(self) -> None:
        metadata = OperatorStructuralMetadata(
            parameter_count=200.0,
            reference_parameter_count=100.0,
            input_degree=2.0,
            output_degree=1.0,
            depth_index=3.0,
            max_input_degree=4.0,
            max_output_degree=2.0,
            max_depth_index=6.0,
            has_nonlinearity=True,
            has_multi_branch_routing=False,
        )

        features = extract_operator_features(metadata)

        self.assertEqual(len(features), 6)
        self.assertAlmostEqual(features[0], math.log(2.0))
        self.assertEqual(features[1:], (0.5, 0.5, 0.5, 1.0, 0.0))


class CandidateFilterTests(unittest.TestCase):
    def test_select_top_k_uses_normalized_nearest_neighbor(self) -> None:
        candidates = (
            BridgeCandidate("far", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), embedding=(0.0, 1.0)),
            BridgeCandidate("near", (2.0, 2.0, 2.0, 2.0, 1.0, 1.0), embedding=(1.0, 0.0)),
        )

        matches = select_top_k_nearest(
            query_features=(2.0, 2.0, 2.0, 2.0, 1.0, 1.0),
            candidates=candidates,
            k=1,
        )

        self.assertEqual(matches[0].candidate.identifier, "near")
        self.assertAlmostEqual(matches[0].distance, 0.0)

    def test_fallback_breaks_ties_by_identifier(self) -> None:
        candidates = (
            BridgeCandidate("b", (1.0, 1.0, 1.0, 1.0, 0.0, 0.0), embedding=(1.0, 0.0)),
            BridgeCandidate("a", (1.0, 1.0, 1.0, 1.0, 0.0, 0.0), embedding=(0.0, 1.0)),
        )

        fallback = deterministic_nearest_neighbor_fallback(
            query_features=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            candidates=candidates,
        )

        self.assertEqual(fallback.candidate.identifier, "a")


class SimilarityTests(unittest.TestCase):
    def test_embedding_similarity_is_deterministic_and_bounded(self) -> None:
        baseline = DeterministicEmbeddingSimilarity()

        self.assertAlmostEqual(baseline.score_embeddings((1.0, 0.0), (1.0, 0.0)), 1.0)
        self.assertAlmostEqual(baseline.score_embeddings((1.0, 0.0), (-1.0, 0.0)), 0.0)
        self.assertAlmostEqual(structural_similarity(3.0), 0.25)


class EmbeddingComposerTests(unittest.TestCase):
    def test_softmax_entropy_and_composition(self) -> None:
        first = BridgeCandidate("first", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), embedding=(1.0, 0.0))
        second = BridgeCandidate("second", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), embedding=(0.0, 1.0))
        similarities = (
            CandidateSimilarity(first, 1.0, 1.0, 1.0, 2.0, 0.0),
            CandidateSimilarity(second, 0.5, 0.5, 0.5, 1.0, 1.0),
        )
        composer = EmbeddingComposer(temperature=1.0, confidence_threshold=0.6)

        weights = softmax((2.0, 1.0))
        result = composer.compose(similarities)

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(entropy(weights), 0.0)
        self.assertTrue(result.passed_confidence_gate)
        self.assertAlmostEqual(result.embedding[0], weights[0])
        self.assertAlmostEqual(result.embedding[1], weights[1])


class BridgeOrchestrationTests(unittest.TestCase):
    def test_bridge_selects_confident_candidate_without_fallback(self) -> None:
        metadata = OperatorStructuralMetadata(
            parameter_count=100.0,
            reference_parameter_count=100.0,
            input_degree=1.0,
            output_degree=1.0,
            depth_index=1.0,
            max_input_degree=2.0,
            max_output_degree=2.0,
            max_depth_index=2.0,
            has_nonlinearity=True,
            has_multi_branch_routing=False,
        )
        features = extract_operator_features(metadata)
        source = BridgeSource(
            identifier="source",
            structural_metadata=metadata,
            embedding=(1.0, 0.0),
            semantic_embedding=(1.0, 0.0),
            functional_embedding=(1.0, 0.0),
        )
        candidates = (
            BridgeCandidate(
                "best",
                features,
                embedding=(1.0, 0.0),
                semantic_embedding=(1.0, 0.0),
                functional_embedding=(1.0, 0.0),
            ),
            BridgeCandidate(
                "other",
                (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
                embedding=(0.0, 1.0),
                semantic_embedding=(0.0, 1.0),
                functional_embedding=(0.0, 1.0),
            ),
        )
        bridge = SKBQBridge(
            top_k=1,
            embedding_composer=EmbeddingComposer(confidence_threshold=0.9),
        )

        decision = bridge.run(source, candidates)

        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.selected_candidate.identifier, "best")
        self.assertAlmostEqual(decision.composition.confidence, 1.0)

    def test_bridge_uses_deterministic_fallback_when_gate_fails(self) -> None:
        metadata = OperatorStructuralMetadata(
            parameter_count=100.0,
            reference_parameter_count=100.0,
            input_degree=0.0,
            output_degree=0.0,
            depth_index=0.0,
            max_input_degree=0.0,
            max_output_degree=0.0,
            max_depth_index=0.0,
        )
        source = BridgeSource(identifier="source", structural_metadata=metadata)
        features = extract_operator_features(metadata)
        candidates = (
            BridgeCandidate("b", features, embedding=(1.0, 0.0)),
            BridgeCandidate("a", features, embedding=(0.0, 1.0)),
        )
        bridge = SKBQBridge(
            top_k=2,
            embedding_composer=EmbeddingComposer(confidence_threshold=0.9),
        )

        decision = bridge.run(source, candidates)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.selected_candidate.identifier, "a")
        self.assertEqual(decision.fallback_match.candidate.identifier, "a")


if __name__ == "__main__":
    unittest.main()
