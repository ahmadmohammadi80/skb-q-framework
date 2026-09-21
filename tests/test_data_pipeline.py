"""Tests for real vocabulary and OOV data preparation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.data import (  # noqa: E402
    DataPreparationConfig,
    DataPreparationPipeline,
    FutureHeldOutModelRequest,
    OOVPreparationInterface,
)
from skbq.vocabulary import (  # noqa: E402
    VocabularyEntry,
    VocabularySourceProvenance,
    VocabularyStore,
    read_vocabulary_json,
    vocabulary_hash,
)


FIXED_TIMESTAMP = "2026-01-01T00:00:00+00:00"


class StaticVocabularyBuilder:
    def __init__(self) -> None:
        self.requests = ()

    def build_from_models(self, requests):
        self.requests = tuple(requests)
        entries = []
        for request in self.requests:
            if request.source_model == "llama":
                entries.append(_entry("Attention", "Llama", request))
            elif request.source_model == "qwen":
                entries.append(_entry("RMSNorm", "Qwen", request))
            elif request.source_model == "mistral":
                entries.append(_entry("SwiGLU", "Mistral", request))
        return VocabularyStore(tuple(entries))


def _entry(operator_type: str, architecture: str, request) -> VocabularyEntry:
    parameter_count = {"Attention": 64.0, "RMSNorm": 8.0, "SwiGLU": 128.0}[operator_type]
    return VocabularyEntry(
        operator_type=operator_type,
        structural_metadata={
            "mean_parameter_count": parameter_count,
            "mean_reference_parameter_count": 128.0,
            "mean_input_degree": 1.0,
            "mean_output_degree": 1.0,
            "mean_depth_index": 1.0,
            "max_input_degree": 1.0,
            "max_output_degree": 1.0,
            "max_depth_index": 3.0,
            "has_nonlinearity": operator_type == "SwiGLU",
            "has_multi_branch_routing": False,
        },
        parameter_statistics={
            "count": 1,
            "total": parameter_count,
            "min": parameter_count,
            "max": parameter_count,
            "mean": parameter_count,
        },
        graph_statistics={
            "occurrence_count": 1,
            "source_graph_count": 1,
            "architectures": [architecture],
            "operator_ids": [operator_type],
            "max_depth_position": 1,
            "max_input_degree": 1,
            "max_output_degree": 1,
        },
        provenance=(
            VocabularySourceProvenance(
                source_model=request.source_model,
                model_version=request.model_version,
                extraction_timestamp=request.extraction_timestamp,
                git_commit="fixture",
                framework_version="0.1.0",
            ),
        ),
    )


def config(output_dir: Path) -> DataPreparationConfig:
    return DataPreparationConfig(
        input_model_paths={
            "llama": "/models/llama",
            "qwen": "/models/qwen",
            "mistral": "/models/mistral",
        },
        model_versions={
            "llama": "llama-v1",
            "qwen": "qwen-v1",
            "mistral": "mistral-v1",
        },
        output_artifact_dir=output_dir,
        split_protocol="operator_class_holdout",
        heldout_operator_types=("SwiGLU",),
        random_seed=123,
        extraction_timestamp=FIXED_TIMESTAMP,
    )


class DataPreparationPipelineTests(unittest.TestCase):
    def test_pipeline_generates_vocabulary_artifacts_and_oov_records(self) -> None:
        with TemporaryDirectory() as directory:
            builder = StaticVocabularyBuilder()
            pipeline = DataPreparationPipeline(
                vocabulary_builder=builder,
                repo_path=Path(__file__).resolve().parents[1],
            )

            result = pipeline.run(config(Path(directory)))

            artifact_dir = Path(directory) / "vocabulary"
            self.assertEqual(result.artifact_manifest.artifact_directory, artifact_dir)
            self.assertTrue((artifact_dir / "vocabulary.json").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertTrue((artifact_dir / "hash").exists())
            self.assertEqual(result.oov_dataset.unknown_records()[0].operator_type, "SwiGLU")
            self.assertEqual(
                [request.source_model for request in builder.requests],
                ["llama", "qwen", "mistral"],
            )

    def test_vocabulary_hash_is_stable(self) -> None:
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first = DataPreparationPipeline(vocabulary_builder=StaticVocabularyBuilder()).run(
                config(Path(first_dir))
            )
            second = DataPreparationPipeline(vocabulary_builder=StaticVocabularyBuilder()).run(
                config(Path(second_dir))
            )

            self.assertEqual(first.artifact_manifest.vocabulary_hash, second.artifact_manifest.vocabulary_hash)
            self.assertEqual(
                first.artifact_manifest.vocabulary_hash,
                vocabulary_hash(read_vocabulary_json(first.artifact_manifest.vocabulary_path)),
            )

    def test_provenance_and_metadata_are_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            result = DataPreparationPipeline(vocabulary_builder=StaticVocabularyBuilder()).run(
                config(Path(directory))
            )
            attention = result.vocabulary.get("Attention")

            self.assertEqual(attention.provenance[0].source_model, "llama")
            self.assertEqual(attention.provenance[0].model_version, "llama-v1")
            self.assertEqual(attention.provenance[0].extraction_timestamp, FIXED_TIMESTAMP)
            self.assertEqual(result.metadata["seed_values"], {"data_preparation": 123})
            self.assertEqual(result.metadata["split"]["V_test"], ["SwiGLU"])

    def test_artifact_writer_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline = DataPreparationPipeline(vocabulary_builder=StaticVocabularyBuilder())
            pipeline.run(config(Path(directory)))

            with self.assertRaises(FileExistsError):
                pipeline.run(config(Path(directory)))

    def test_invalid_configuration_and_future_oov_interface(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                DataPreparationConfig(
                    input_model_paths={"llama": "/models/llama"},
                    output_artifact_dir=Path(directory),
                    split_protocol="operator_class_holdout",
                    heldout_operator_types=("SwiGLU",),
                    random_seed=0,
                )
            with self.assertRaises(ValueError):
                DataPreparationConfig(
                    input_model_paths={
                        "llama": "/models/llama",
                        "qwen": "/models/qwen",
                        "mistral": "/models/mistral",
                    },
                    output_artifact_dir=Path(directory),
                    split_protocol="invalid",
                    heldout_operator_types=("SwiGLU",),
                    random_seed=0,
                )
            request = FutureHeldOutModelRequest("mamba", "/models/mamba", "future")
            with self.assertRaises(NotImplementedError):
                OOVPreparationInterface().prepare(request)


if __name__ == "__main__":
    unittest.main()
