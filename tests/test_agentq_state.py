"""Tests for AgentQ state construction from operator graphs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.agentq import GraphState, OperatorState, StateBuilder  # noqa: E402
from skbq.backbone import StructuralFeatureFrozenEncoder  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402


class StateBuilderTests(unittest.TestCase):
    def test_synthetic_graph_converts_to_valid_graph_state(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=2))
        state = StateBuilder().build(graph)

        self.assertIsInstance(state, GraphState)
        self.assertEqual(state.node_count, len(graph.nodes))
        self.assertEqual(state.architecture, graph.architecture)
        self.assertTrue(state.graph_identifier.startswith(graph.architecture))

    def test_operator_ids_are_preserved(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        state = StateBuilder().build(graph)

        self.assertEqual(set(state.operator_ids), set(graph.node_ids))
        for operator_id in graph.node_ids:
            operator_state = state.get(operator_id)
            self.assertEqual(operator_state.operator_id, operator_id)

    def test_operator_state_contains_structural_features_and_optional_embedding(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        encoder = StructuralFeatureFrozenEncoder()
        state = StateBuilder(encoder=encoder, include_encoder_embeddings=True).build(graph)
        operator = state.get("embedding")
        metadata = graph.structural_metadata_for("embedding")

        self.assertIsInstance(operator, OperatorState)
        self.assertEqual(len(operator.structural_features), 6)
        self.assertIsNotNone(operator.encoder_embedding)
        self.assertEqual(operator.encoder_embedding, encoder.encode(metadata))
        self.assertGreater(operator.parameter_count, 0)

    def test_state_builder_without_embeddings(self) -> None:
        state = StateBuilder(include_encoder_embeddings=False).build(
            build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        )

        for operator in state.operators:
            self.assertIsNone(operator.encoder_embedding)

    def test_graph_state_serialization_is_deterministic(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        builder = StateBuilder()

        first = builder.build(graph)
        second = builder.build(graph)

        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(len(first.state_hash()), 64)
        self.assertIn("embedding", first.canonical_json())

    def test_graph_state_sorts_operators_deterministically(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        state = StateBuilder().build(graph)

        self.assertEqual(state.operator_ids, tuple(sorted(graph.node_ids)))


if __name__ == "__main__":
    unittest.main()
