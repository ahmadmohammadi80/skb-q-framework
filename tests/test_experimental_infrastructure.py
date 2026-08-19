"""Tests for SKB-Q experimental infrastructure components."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.backbone.encoder import FrozenEncoder, validate_embedding  # noqa: E402
from skbq.backbone.policy_interface import (  # noqa: E402
    FrozenPolicy,
    PolicyAllocation,
    validate_policy_inputs,
)
from skbq.baselines import (  # noqa: E402
    NonLLMSemanticBaseline,
    RandomFallbackBaseline,
    SKBQFullBaseline,
    StructuralNearestNeighborBaseline,
)
from skbq.bridge.candidate_filter import BridgeCandidate  # noqa: E402
from skbq.bridge.embedding_composer import EmbeddingComposer  # noqa: E402
from skbq.bridge.skbq_bridge import BridgeSource, SKBQBridge  # noqa: E402
from skbq.bridge.structural_features import (  # noqa: E402
    OperatorStructuralMetadata,
    extract_operator_features,
)
from skbq.config import SeedRegistry  # noqa: E402
from skbq.vocabulary import default_operator_registry  # noqa: E402


class VocabularyRegistryTests(unittest.TestCase):
    def test_default_registry_contains_known_operator_examples(self) -> None:
        registry = default_operator_registry()

        self.assertEqual(len(registry), 6)
        self.assertIn("Attention", registry.names())
        self.assertIn("MoE Router", registry.names())
        self.assertEqual(registry.get("gqa").name, "GQA")
        self.assertIsNone(registry.get("SwiGLU").embedding)
        self.assertEqual(len(registry.get("RMSNorm").structural_features), 6)

    def test_registry_entries_convert_to_bridge_candidates(self) -> None:
        candidate = default_operator_registry().get("Expert FFN").to_bridge_candidate()

        self.assertIsInstance(candidate, BridgeCandidate)
        self.assertEqual(candidate.identifier, "Expert FFN")
        self.assertEqual(candidate.metadata["feature_source"], "schematic_registry_entry")


class BaselineComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        metadata = OperatorStructuralMetadata(
            parameter_count=100.0,
            reference_parameter_count=100.0,
            input_degree=1.0,
            output_degree=1.0,
            depth_index=1.0,
            max_input_degree=2.0,
            max_output_degree=2.0,
            max_depth_index=2.0,
        )
        features = extract_operator_features(metadata)
        self.source = BridgeSource(
            identifier="source",
            structural_metadata=metadata,
            embedding=(1.0, 0.0),
            semantic_embedding=(1.0, 0.0),
            functional_embedding=(1.0, 0.0),
        )
        self.candidates = (
            BridgeCandidate(
                "near",
                features,
                embedding=(1.0, 0.0),
                semantic_embedding=(1.0, 0.0),
                functional_embedding=(1.0, 0.0),
            ),
            BridgeCandidate(
                "far",
                (2.0, 2.0, 2.0, 2.0, 1.0, 1.0),
                embedding=(0.0, 1.0),
                semantic_embedding=(0.0, 1.0),
                functional_embedding=(0.0, 1.0),
            ),
        )

    def test_random_fallback_is_seed_reproducible(self) -> None:
        seed_registry = SeedRegistry({"baseline": 42, "python": 0})
        baseline = RandomFallbackBaseline(seed_registry=seed_registry)

        first = baseline.run(self.source, self.candidates)
        second = baseline.run(self.source, self.candidates)

        self.assertEqual(first.selected_candidate.identifier, second.selected_candidate.identifier)
        self.assertEqual(first.baseline_name, "random_fallback")
        self.assertEqual(first.details["seed_registry"], {"baseline": 42, "python": 0})

    def test_structural_and_semantic_baselines_select_expected_candidate(self) -> None:
        structural = StructuralNearestNeighborBaseline().run(self.source, self.candidates)
        semantic = NonLLMSemanticBaseline().run(self.source, self.candidates)

        self.assertEqual(structural.selected_candidate.identifier, "near")
        self.assertEqual(semantic.selected_candidate.identifier, "near")
        self.assertGreaterEqual(structural.score, semantic.score)

    def test_full_skbq_baseline_wraps_bridge_decision(self) -> None:
        baseline = SKBQFullBaseline(
            bridge=SKBQBridge(
                top_k=1,
                embedding_composer=EmbeddingComposer(confidence_threshold=0.9),
            )
        )

        decision = baseline.run(self.source, self.candidates)

        self.assertEqual(decision.baseline_name, "skbq_full")
        self.assertEqual(decision.selected_candidate.identifier, "near")
        self.assertFalse(decision.details["used_fallback"])


class BackboneInterfaceTests(unittest.TestCase):
    def test_frozen_encoder_is_abstract_and_validates_embeddings(self) -> None:
        class ConstantEncoder(FrozenEncoder):
            def encode(self, operator: object) -> tuple[float, ...]:
                return validate_embedding((1.0, 2.0, 3.0))

        with self.assertRaises(TypeError):
            FrozenEncoder()

        self.assertEqual(ConstantEncoder().encode(object()), (1.0, 2.0, 3.0))

    def test_frozen_policy_is_abstract_and_validates_inputs(self) -> None:
        class BudgetPolicy(FrozenPolicy):
            def allocate(
                self,
                graph_embedding: tuple[float, ...],
                budget: float,
            ) -> PolicyAllocation:
                embedding, validated_budget = validate_policy_inputs(graph_embedding, budget)
                return PolicyAllocation(
                    allocations={"graph": validated_budget},
                    metadata={"embedding_dim": len(embedding)},
                )

        with self.assertRaises(TypeError):
            FrozenPolicy()

        allocation = BudgetPolicy().allocate((1.0, 0.0), 8.0)

        self.assertEqual(allocation.allocations["graph"], 8.0)
        self.assertEqual(allocation.metadata["embedding_dim"], 2)


if __name__ == "__main__":
    unittest.main()
