"""Tests for frozen backbone integration with SKB-Q bridge components."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.backbone import (  # noqa: E402
    FrozenBackbone,
    StructuralFeatureFrozenEncoder,
    UniformFrozenPolicy,
    mean_pool_embeddings,
)
from skbq.bridge.embedding_composer import EmbeddingComposer  # noqa: E402
from skbq.bridge.skbq_bridge import BridgeSource, SKBQBridge  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402
from skbq.vocabulary import default_operator_registry  # noqa: E402


class FrozenEncoderReferenceTests(unittest.TestCase):
    def test_structural_feature_encoder_returns_immutable_phi_embedding(self) -> None:
        candidate = default_operator_registry().get("Attention").to_bridge_candidate()
        encoder = StructuralFeatureFrozenEncoder()

        embedding = encoder.encode(candidate)
        encoded = encoder.encode_operator(candidate.identifier, candidate)

        self.assertIsInstance(embedding, tuple)
        self.assertEqual(embedding, candidate.structural_features)
        self.assertEqual(encoded.embedding, embedding)
        self.assertEqual(encoded.encoder_name, "structural_feature_phi")

    def test_mean_pool_embeddings_is_deterministic(self) -> None:
        first = mean_pool_embeddings(((1.0, 3.0), (3.0, 5.0)))
        second = mean_pool_embeddings(((1.0, 3.0), (3.0, 5.0)))

        self.assertEqual(first, (2.0, 4.0))
        self.assertEqual(first, second)


class FrozenPolicyReferenceTests(unittest.TestCase):
    def test_uniform_policy_returns_p_g_distribution(self) -> None:
        policy = UniformFrozenPolicy()

        allocation = policy.allocate((1.0, 2.0, 3.0), budget=9.0)

        self.assertEqual(allocation.policy_name, "uniform_pi_theta")
        self.assertEqual(allocation.total_budget, 9.0)
        self.assertEqual(allocation.metadata["notation"], "P(G)")
        self.assertEqual(
            dict(allocation.allocations),
            {"dimension_0": 3.0, "dimension_1": 3.0, "dimension_2": 3.0},
        )

    def test_uniform_policy_rejects_invalid_budget(self) -> None:
        with self.assertRaises(ValueError):
            UniformFrozenPolicy().allocate((1.0, 2.0), budget=-1.0)


class FrozenBackboneIntegrationTests(unittest.TestCase):
    def test_backbone_encodes_bridge_candidates_without_mutation(self) -> None:
        raw_candidate = default_operator_registry().get("SwiGLU").to_bridge_candidate()
        backbone = FrozenBackbone()

        encoded_candidate = backbone.encode_candidate(raw_candidate)

        self.assertIsNone(raw_candidate.embedding)
        self.assertEqual(encoded_candidate.embedding, raw_candidate.structural_features)
        self.assertEqual(encoded_candidate.metadata["embedding_notation"], "phi(v)")

    def test_backbone_allocates_graph_deterministically(self) -> None:
        graph = build_transformer_graph(
            SyntheticArchitectureSpec("Transformer", num_layers=1, hidden_size=8)
        )
        backbone = FrozenBackbone()

        first = backbone.allocate_graph(graph, budget=6.0)
        second = backbone.allocate_graph(graph, budget=6.0)

        self.assertEqual(dict(first.allocations), dict(second.allocations))
        self.assertEqual(first.total_budget, 6.0)
        self.assertEqual(len(first.allocations), 6)

    def test_encoded_candidates_run_through_skbq_bridge(self) -> None:
        graph = build_transformer_graph(
            SyntheticArchitectureSpec("Transformer", num_layers=1, hidden_size=8)
        )
        source = BridgeSource(
            identifier="source_attention",
            structural_metadata=graph.structural_metadata_for("layer0.attention"),
        )
        registry = default_operator_registry()
        raw_candidates = (
            registry.get("Attention").to_bridge_candidate(),
            registry.get("GQA").to_bridge_candidate(),
        )
        encoded_candidates = FrozenBackbone().encode_candidates(raw_candidates)
        bridge = SKBQBridge(
            top_k=1,
            embedding_composer=EmbeddingComposer(confidence_threshold=0.9),
        )

        decision = bridge.run(source, encoded_candidates)

        self.assertFalse(decision.used_fallback)
        self.assertIsNotNone(decision.composition.embedding)
        self.assertEqual(decision.composition.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
