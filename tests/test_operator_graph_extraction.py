"""Tests for the model-agnostic SKB-Q operator graph extraction layer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.bridge.structural_features import extract_operator_features  # noqa: E402
from skbq.graph import (  # noqa: E402
    GraphExtractionPipeline,
    OperatorGraph,
    OperatorNode,
    SyntheticArchitectureSpec,
    SyntheticGraphExtractor,
    build_mamba_graph,
    build_moe_transformer_graph,
    build_rwkv_graph,
    build_transformer_graph,
)


class OperatorGraphRepresentationTests(unittest.TestCase):
    def test_graph_rejects_inconsistent_edges(self) -> None:
        parent = OperatorNode(
            operator_id="parent",
            operator_type="Linear",
            child_ids=("child",),
        )
        child = OperatorNode(
            operator_id="child",
            operator_type="Linear",
        )

        with self.assertRaises(ValueError):
            OperatorGraph(nodes=(parent, child))

    def test_zero_parameter_node_emits_positive_structural_metadata(self) -> None:
        node = OperatorNode(
            operator_id="zero_param",
            operator_type="Identity",
            parameter_count=0,
            tensor_shapes={"activation": ("seq", 8)},
        )
        graph = OperatorGraph(nodes=(node,), architecture="unit")

        metadata = graph.structural_metadata_for("zero_param")
        features = extract_operator_features(metadata)

        self.assertEqual(metadata.parameter_count, 1.0)
        self.assertEqual(metadata.reference_parameter_count, 1.0)
        self.assertEqual(len(features), 6)


class SyntheticGraphBuilderTests(unittest.TestCase):
    def test_transformer_graph_produces_metadata_for_every_operator(self) -> None:
        graph = build_transformer_graph(
            SyntheticArchitectureSpec("Transformer", num_layers=2, hidden_size=16)
        )
        metadata = graph.structural_metadata()

        self.assertEqual(graph.architecture, "Transformer")
        self.assertEqual(graph.roots()[0].operator_id, "embedding")
        self.assertEqual(graph.leaves()[0].operator_id, "output")
        self.assertEqual(set(metadata), set(graph.node_ids))
        self.assertIn("layer0.attention", graph.node_ids)
        self.assertEqual(len(extract_operator_features(metadata["layer0.attention"])), 6)

    def test_moe_transformer_graph_records_branching_information(self) -> None:
        graph = build_moe_transformer_graph(
            SyntheticArchitectureSpec(
                "MoE Transformer",
                num_layers=1,
                hidden_size=8,
                num_experts=3,
            )
        )
        router = graph.get("layer0.moe_router")
        merge = graph.get("layer0.expert_merge")
        metadata = graph.structural_metadata_for("layer0.moe_router")

        self.assertEqual(router.output_degree, 3)
        self.assertTrue(router.has_multi_branch_routing)
        self.assertEqual(len(merge.parent_ids), 3)
        self.assertTrue(merge.is_branch_merge)
        self.assertTrue(metadata.has_multi_branch_routing)

    def test_mamba_and_rwkv_builders_are_deterministic(self) -> None:
        spec = SyntheticArchitectureSpec("Mamba-style", num_layers=2, hidden_size=12)
        first = build_mamba_graph(spec)
        second = build_mamba_graph(spec)
        rwkv = build_rwkv_graph(SyntheticArchitectureSpec("RWKV-style", num_layers=1))

        self.assertEqual(first.node_ids, second.node_ids)
        self.assertEqual(first.structural_metadata(), second.structural_metadata())
        self.assertIn("layer0.time_mix", rwkv.node_ids)
        self.assertEqual(rwkv.get("layer0.time_mix").tensor_shapes["input"], ("seq", 128))


class GraphExtractionPipelineTests(unittest.TestCase):
    def test_pipeline_supports_all_synthetic_architectures(self) -> None:
        pipeline = GraphExtractionPipeline(SyntheticGraphExtractor())

        for architecture in (
            "Transformer",
            "MoE Transformer",
            "Mamba-style",
            "RWKV-style",
        ):
            with self.subTest(architecture=architecture):
                graph = pipeline.extract_graph(
                    SyntheticArchitectureSpec(architecture, num_layers=1, hidden_size=8)
                )
                metadata = pipeline.extract_structural_metadata(
                    SyntheticArchitectureSpec(architecture, num_layers=1, hidden_size=8)
                )

                self.assertGreater(len(graph.nodes), 0)
                self.assertEqual(set(metadata), set(graph.node_ids))

    def test_extractor_accepts_mapping_specs(self) -> None:
        graph = SyntheticGraphExtractor().extract(
            {
                "architecture": "transformer",
                "num_layers": 1,
                "hidden_size": 8,
            }
        )

        self.assertEqual(graph.architecture, "Transformer")
        self.assertIn("layer0.swiglu", graph.node_ids)


if __name__ == "__main__":
    unittest.main()
