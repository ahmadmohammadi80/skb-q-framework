"""Tests for Hugging Face real-model integration layer contracts."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.bridge.candidate_filter import BridgeCandidate  # noqa: E402
from skbq.bridge.embedding_composer import EmbeddingComposer  # noqa: E402
from skbq.bridge.skbq_bridge import BridgeSource, SKBQBridge  # noqa: E402
from skbq.bridge.structural_features import extract_operator_features  # noqa: E402
from skbq.models import (  # noqa: E402
    HuggingFaceGraphBuilder,
    HuggingFaceGraphExtractor,
    HuggingFaceModelLoader,
    MissingDependencyError,
    UnsupportedArchitectureError,
    detect_supported_architecture,
)
from skbq.models.operator_mapping import map_operator_type  # noqa: E402


class StubParameter:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def numel(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


class StubModule:
    def __init__(
        self,
        children: list[tuple[str, "StubModule"]] | None = None,
        parameters: list[tuple[str, StubParameter]] | None = None,
        config: object | None = None,
    ) -> None:
        self._children = tuple(children or ())
        self._parameters = tuple(parameters or ())
        if config is not None:
            self.config = config

    def named_children(self):
        return self._children

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for child_name, child in self._children:
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from child.named_modules(child_prefix)

    def named_parameters(self, recurse: bool = False):
        if not recurse:
            return self._parameters
        parameters = list(self._parameters)
        for child_name, child in self._children:
            for parameter_name, parameter in child.named_parameters(recurse=True):
                parameters.append((f"{child_name}.{parameter_name}", parameter))
        return tuple(parameters)


class LlamaForCausalLM(StubModule):
    pass


class LlamaModel(StubModule):
    pass


class ModuleList(StubModule):
    pass


class LlamaDecoderLayer(StubModule):
    pass


class LlamaAttention(StubModule):
    pass


class LlamaMLP(StubModule):
    pass


class LlamaRMSNorm(StubModule):
    pass


class Linear(StubModule):
    pass


def linear() -> Linear:
    return Linear(parameters=[("weight", StubParameter((8, 8)))])


def rms_norm() -> LlamaRMSNorm:
    return LlamaRMSNorm(parameters=[("weight", StubParameter((8,)))])


def llama_fixture(model_type: str = "llama") -> LlamaForCausalLM:
    attention = LlamaAttention(
        children=[
            ("q_proj", linear()),
            ("k_proj", linear()),
            ("v_proj", linear()),
            ("o_proj", linear()),
        ]
    )
    mlp = LlamaMLP(
        children=[
            ("gate_proj", linear()),
            ("up_proj", linear()),
            ("down_proj", linear()),
        ]
    )
    layer = LlamaDecoderLayer(
        children=[
            ("input_layernorm", rms_norm()),
            ("self_attn", attention),
            ("post_attention_layernorm", rms_norm()),
            ("mlp", mlp),
        ]
    )
    model = LlamaModel(children=[("layers", ModuleList(children=[("0", layer)]))])
    return LlamaForCausalLM(
        children=[
            ("model", model),
            ("lm_head", linear()),
        ],
        config=SimpleNamespace(model_type=model_type),
    )


class HuggingFaceModelIntegrationTests(unittest.TestCase):
    def test_loader_uses_hugging_face_auto_classes(self) -> None:
        model = llama_fixture()
        config = model.config
        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda model_name, **kwargs: config
            ),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda model_name, **kwargs: model
            ),
        )

        with patch("skbq.models.loader._import_transformers", return_value=fake_transformers):
            loaded = HuggingFaceModelLoader().load("local-llama")

        self.assertEqual(loaded.model_name_or_path, "local-llama")
        self.assertEqual(loaded.architecture, "Llama")
        self.assertIs(loaded.model, model)

    def test_loader_reports_missing_transformers_dependency(self) -> None:
        with patch(
            "skbq.models.loader._import_transformers",
            side_effect=MissingDependencyError("missing"),
        ):
            with self.assertRaises(MissingDependencyError):
                HuggingFaceModelLoader().load("local-llama")

    def test_graph_builder_preserves_topology_and_metadata(self) -> None:
        graph = HuggingFaceGraphBuilder().build(llama_fixture())

        self.assertEqual(graph.architecture, "Llama")
        self.assertIn("__model__", graph.node_ids)
        self.assertIn("model.layers.0.self_attn.q_proj", graph.node_ids)
        attention = graph.get("model.layers.0.self_attn")
        q_proj = graph.get("model.layers.0.self_attn.q_proj")
        self.assertIn(q_proj.operator_id, attention.child_ids)
        self.assertEqual(q_proj.parent_ids, (attention.operator_id,))
        self.assertEqual(q_proj.parameter_count, 64)
        self.assertEqual(q_proj.tensor_shapes["parameter:weight"], (8, 8))

        metadata = graph.structural_metadata()
        self.assertEqual(set(metadata), set(graph.node_ids))
        self.assertEqual(len(extract_operator_features(metadata[q_proj.operator_id])), 6)

    def test_graph_extraction_is_deterministic(self) -> None:
        builder = HuggingFaceGraphBuilder()

        first = builder.build(llama_fixture())
        second = builder.build(llama_fixture())

        self.assertEqual(first.node_ids, second.node_ids)
        self.assertEqual(first.structural_metadata(), second.structural_metadata())

    def test_graph_extractor_accepts_loaded_model_bundle(self) -> None:
        model = llama_fixture()
        loaded = SimpleNamespace(model=model, config=model.config)

        graph = HuggingFaceGraphExtractor().extract(loaded)

        self.assertEqual(graph.architecture, "Llama")

    def test_supported_and_unsupported_architecture_detection(self) -> None:
        self.assertEqual(detect_supported_architecture(SimpleNamespace(model_type="qwen2")), "Qwen")
        self.assertEqual(detect_supported_architecture(SimpleNamespace(model_type="mistral")), "Mistral")

        with self.assertRaises(UnsupportedArchitectureError):
            detect_supported_architecture(SimpleNamespace(model_type="mamba"))
        with self.assertRaises(UnsupportedArchitectureError):
            HuggingFaceGraphBuilder().build(llama_fixture(model_type="mixtral"))

    def test_operator_mapping_does_not_skip_unknown_modules(self) -> None:
        unknown = StubModule()

        self.assertEqual(map_operator_type("model.unknown", unknown), "HF::StubModule")

    def test_graph_output_is_bridge_compatible(self) -> None:
        graph = HuggingFaceGraphBuilder().build(llama_fixture())
        source_id = "model.layers.0.self_attn.q_proj"
        source = BridgeSource(
            identifier=source_id,
            structural_metadata=graph.structural_metadata_for(source_id),
        )
        candidates = []
        for node in (graph.get(source_id), graph.get("model.layers.0.self_attn.k_proj")):
            features = extract_operator_features(graph.structural_metadata_for(node.operator_id))
            candidates.append(
                BridgeCandidate(
                    identifier=node.operator_id,
                    structural_features=features,
                    embedding=features,
                )
            )
        bridge = SKBQBridge(
            top_k=1,
            embedding_composer=EmbeddingComposer(confidence_threshold=0.9),
        )

        decision = bridge.run(source, tuple(candidates))

        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.composition.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
