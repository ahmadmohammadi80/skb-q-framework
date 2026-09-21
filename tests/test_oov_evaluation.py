"""Tests for SKB-Q out-of-vocabulary evaluation framework."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.config import SeedRegistry  # noqa: E402
from skbq.evaluation.oov import (  # noqa: E402
    GNNBaselineHook,
    OOVDataset,
    RandomFallbackHook,
    StructuralNearestNeighborHook,
    architecture_family_holdout,
    candidate_recall,
    oov_coverage,
    operator_class_holdout,
    operator_distance_statistics,
    split_reproducibility_report,
)
from skbq.vocabulary import (  # noqa: E402
    VocabularyEntry,
    VocabularySourceProvenance,
    VocabularyStore,
)


def provenance(source_model: str) -> VocabularySourceProvenance:
    return VocabularySourceProvenance(
        source_model=source_model,
        model_version="unit",
        extraction_timestamp="2026-01-01T00:00:00+00:00",
        git_commit="abc123",
        framework_version="0.1.0",
    )


def entry(
    operator_type: str,
    mean_parameter_count: float,
    architectures: list[str],
    source_model: str,
) -> VocabularyEntry:
    return VocabularyEntry(
        operator_type=operator_type,
        structural_metadata={
            "mean_parameter_count": mean_parameter_count,
            "mean_reference_parameter_count": 128.0,
            "mean_input_degree": 1.0,
            "mean_output_degree": 1.0,
            "mean_depth_index": 1.0,
            "max_input_degree": 1.0,
            "max_output_degree": 1.0,
            "max_depth_index": 4.0,
            "has_nonlinearity": operator_type == "SwiGLU",
            "has_multi_branch_routing": False,
        },
        parameter_statistics={
            "count": 1,
            "total": mean_parameter_count,
            "min": mean_parameter_count,
            "max": mean_parameter_count,
            "mean": mean_parameter_count,
        },
        graph_statistics={
            "occurrence_count": 1,
            "source_graph_count": 1,
            "architectures": architectures,
            "operator_ids": [operator_type],
            "max_depth_position": 1,
            "max_input_degree": 1,
            "max_output_degree": 1,
        },
        provenance=(provenance(source_model),),
    )


def vocabulary_fixture() -> VocabularyStore:
    return VocabularyStore(
        (
            entry("Attention", 64.0, ["Llama", "Qwen"], "llama"),
            entry("RMSNorm", 8.0, ["Llama"], "llama"),
            entry("SwiGLU", 128.0, ["Mistral"], "mistral"),
        )
    )


class OOVSplitTests(unittest.TestCase):
    def test_operator_class_holdout_is_deterministic_and_disjoint(self) -> None:
        split = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))
        repeated = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))

        self.assertEqual(split.to_mapping(), repeated.to_mapping())
        self.assertEqual(split.test_operator_types, ("SwiGLU",))
        self.assertEqual(set(split.train_operator_types) & set(split.test_operator_types), set())

    def test_architecture_family_holdout(self) -> None:
        split = architecture_family_holdout(vocabulary_fixture(), ("Mistral",))

        self.assertEqual(split.test_operator_types, ("SwiGLU",))
        self.assertIn("Attention", split.train_operator_types)

    def test_invalid_splits_raise(self) -> None:
        vocabulary = vocabulary_fixture()

        with self.assertRaises(ValueError):
            operator_class_holdout(vocabulary, ("Unknown",))
        with self.assertRaises(ValueError):
            operator_class_holdout(vocabulary, vocabulary.operator_types())
        with self.assertRaises(ValueError):
            architecture_family_holdout(vocabulary, ("RWKV",))


class OOVDatasetAndAnalysisTests(unittest.TestCase):
    def test_dataset_preserves_provenance_and_membership(self) -> None:
        split = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))
        dataset = OOVDataset.from_vocabulary(vocabulary_fixture(), split)

        unknown = dataset.unknown_records()[0]

        self.assertEqual(unknown.operator_type, "SwiGLU")
        self.assertEqual(unknown.label, "unknown")
        self.assertEqual(unknown.vocabulary_membership, "V_test")
        self.assertEqual(unknown.provenance["source_model"], "mistral")

    def test_dataset_and_analysis_are_reproducible(self) -> None:
        split = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))
        first = OOVDataset.from_vocabulary(vocabulary_fixture(), split)
        second = OOVDataset.from_vocabulary(vocabulary_fixture(), split)

        self.assertEqual(first.to_mapping(), second.to_mapping())
        self.assertEqual(oov_coverage(first), oov_coverage(second))
        self.assertEqual(split_reproducibility_report(split)["has_overlap"], False)

    def test_candidate_recall_and_distance_statistics(self) -> None:
        split = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))
        dataset = OOVDataset.from_vocabulary(vocabulary_fixture(), split)
        unknown = dataset.unknown_records()[0]

        recall = candidate_recall(dataset, {unknown.record_id: ("Attention", "SwiGLU")})
        distances = operator_distance_statistics(dataset)

        self.assertEqual(recall["recall"], 1.0)
        self.assertEqual(distances["status"], "computed")
        self.assertEqual(distances["count"], 1)


class OOVProtocolHookTests(unittest.TestCase):
    def test_baseline_hooks_are_deterministic(self) -> None:
        split = operator_class_holdout(vocabulary_fixture(), ("SwiGLU",))
        dataset = OOVDataset.from_vocabulary(vocabulary_fixture(), split)
        unknown = dataset.unknown_records()[0]
        known = dataset.known_records()

        structural = StructuralNearestNeighborHook().evaluate(unknown, known)
        first_random = RandomFallbackHook(SeedRegistry({"oov": 7})).evaluate(unknown, known)
        second_random = RandomFallbackHook(SeedRegistry({"oov": 7})).evaluate(unknown, known)
        gnn = GNNBaselineHook().evaluate(unknown, known)

        self.assertEqual(structural.status, "computed")
        self.assertIsNotNone(structural.selected_operator_type)
        self.assertEqual(first_random.selected_operator_type, second_random.selected_operator_type)
        self.assertEqual(gnn.status, "not_implemented")


if __name__ == "__main__":
    unittest.main()
