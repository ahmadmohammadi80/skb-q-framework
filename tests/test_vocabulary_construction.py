"""Tests for reproducible SKB-Q vocabulary construction."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.graph import OperatorGraph, OperatorNode  # noqa: E402
from skbq.vocabulary import (  # noqa: E402
    VocabularyBuilder,
    VocabularyBuildRequest,
    canonical_vocabulary_json,
    read_vocabulary_json,
    vocabulary_hash,
    write_vocabulary_json,
)


FIXED_TIMESTAMP = "2026-01-01T00:00:00+00:00"


class StaticExtractor:
    def __init__(self, graphs: dict[str, OperatorGraph]) -> None:
        self.graphs = graphs
        self.seen_specs: list[str] = []

    def extract(self, model_spec: object) -> OperatorGraph:
        key = str(model_spec)
        self.seen_specs.append(key)
        return self.graphs[key]


def graph_fixture(architecture: str, parameter_offset: int = 0) -> OperatorGraph:
    attention_params = 64 + parameter_offset
    mlp_params = 128 + parameter_offset
    nodes = (
        OperatorNode(
            operator_id="__model__",
            operator_type="HF::Root",
            child_ids=("layer.attention_a", "layer.attention_b", "layer.mlp", "layer.norm"),
            depth_position=0,
        ),
        OperatorNode(
            operator_id="layer.attention_a",
            operator_type="Attention",
            parent_ids=("__model__",),
            parameter_count=attention_params,
            tensor_shapes={"parameter:weight": (8, 8)},
            depth_position=1,
        ),
        OperatorNode(
            operator_id="layer.attention_b",
            operator_type="Attention",
            parent_ids=("__model__",),
            parameter_count=attention_params + 8,
            tensor_shapes={"parameter:weight": (8, 9)},
            depth_position=1,
        ),
        OperatorNode(
            operator_id="layer.mlp",
            operator_type="SwiGLU",
            parent_ids=("__model__",),
            parameter_count=mlp_params,
            tensor_shapes={"parameter:weight": (8, 16)},
            depth_position=1,
            has_nonlinearity=True,
        ),
        OperatorNode(
            operator_id="layer.norm",
            operator_type="RMSNorm",
            parent_ids=("__model__",),
            parameter_count=8,
            tensor_shapes={"parameter:weight": (8,)},
            depth_position=1,
        ),
    )
    return OperatorGraph(nodes=nodes, architecture=architecture)


class VocabularyConstructionTests(unittest.TestCase):
    def test_vocabulary_creation_from_extracted_graph(self) -> None:
        extractor = StaticExtractor({"llama": graph_fixture("Llama")})
        builder = VocabularyBuilder(extractor=extractor, repo_path=Path(__file__).resolve().parents[1])

        store = builder.build_from_model(
            "llama",
            source_model="Llama-Unit",
            model_version="v1",
            extraction_timestamp=FIXED_TIMESTAMP,
        )

        self.assertEqual(extractor.seen_specs, ["llama"])
        self.assertIn("Attention", store.operator_types())
        attention = store.get("Attention")
        self.assertEqual(attention.graph_statistics["occurrence_count"], 2)
        self.assertEqual(attention.parameter_statistics["count"], 2)
        self.assertEqual(attention.provenance[0].source_model, "Llama-Unit")
        self.assertEqual(attention.provenance[0].model_version, "v1")
        self.assertEqual(attention.provenance[0].extraction_timestamp, FIXED_TIMESTAMP)

    def test_vocabulary_output_is_deterministic(self) -> None:
        extractor = StaticExtractor({"qwen": graph_fixture("Qwen", parameter_offset=4)})
        builder = VocabularyBuilder(extractor=extractor, repo_path=Path(__file__).resolve().parents[1])

        first = builder.build_from_model(
            "qwen",
            source_model="Qwen-Unit",
            model_version="v2",
            extraction_timestamp=FIXED_TIMESTAMP,
        )
        second = builder.build_from_model(
            "qwen",
            source_model="Qwen-Unit",
            model_version="v2",
            extraction_timestamp=FIXED_TIMESTAMP,
        )

        self.assertEqual(canonical_vocabulary_json(first), canonical_vocabulary_json(second))
        self.assertEqual(vocabulary_hash(first), vocabulary_hash(second))

    def test_multi_model_merging_preserves_provenance(self) -> None:
        extractor = StaticExtractor(
            {
                "llama": graph_fixture("Llama"),
                "qwen": graph_fixture("Qwen", parameter_offset=4),
                "mistral": graph_fixture("Mistral", parameter_offset=8),
            }
        )
        builder = VocabularyBuilder(extractor=extractor, repo_path=Path(__file__).resolve().parents[1])

        store = builder.build_from_models(
            (
                VocabularyBuildRequest("llama", "Llama-Unit", "v1", FIXED_TIMESTAMP),
                VocabularyBuildRequest("qwen", "Qwen-Unit", "v2", FIXED_TIMESTAMP),
                VocabularyBuildRequest("mistral", "Mistral-Unit", "v3", FIXED_TIMESTAMP),
            )
        )

        attention = store.get("Attention")
        self.assertEqual(attention.graph_statistics["occurrence_count"], 6)
        self.assertEqual(attention.graph_statistics["source_graph_count"], 3)
        self.assertEqual(
            [item.source_model for item in attention.provenance],
            ["Llama-Unit", "Mistral-Unit", "Qwen-Unit"],
        )
        self.assertEqual(
            attention.graph_statistics["architectures"],
            ["Llama", "Mistral", "Qwen"],
        )

    def test_duplicate_operator_handling_merges_entries(self) -> None:
        store = VocabularyBuilder(
            extractor=StaticExtractor({"llama": graph_fixture("Llama")}),
            repo_path=Path(__file__).resolve().parents[1],
        ).build_from_model(
            "llama",
            source_model="Llama-Unit",
            model_version="v1",
            extraction_timestamp=FIXED_TIMESTAMP,
        )

        self.assertEqual(store.operator_types().count("Attention"), 1)
        attention = store.get("Attention")
        self.assertEqual(attention.parameter_statistics["min"], 64.0)
        self.assertEqual(attention.parameter_statistics["max"], 72.0)
        self.assertEqual(attention.parameter_statistics["mean"], 68.0)

    def test_vocabulary_serialization_round_trip_and_no_overwrite(self) -> None:
        store = VocabularyBuilder(
            extractor=StaticExtractor({"llama": graph_fixture("Llama")}),
            repo_path=Path(__file__).resolve().parents[1],
        ).build_from_model(
            "llama",
            source_model="Llama-Unit",
            model_version="v1",
            extraction_timestamp=FIXED_TIMESTAMP,
        )

        with TemporaryDirectory() as directory:
            output_path = write_vocabulary_json(store, Path(directory))
            loaded = read_vocabulary_json(output_path)

            self.assertEqual(output_path.name, "vocabulary.json")
            self.assertEqual(vocabulary_hash(store), vocabulary_hash(loaded))
            with self.assertRaises(FileExistsError):
                write_vocabulary_json(store, output_path)


if __name__ == "__main__":
    unittest.main()
