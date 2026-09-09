"""Executable data preparation pipeline for SKB-Q vocabulary and OOV records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skbq import __version__
from skbq.config import SeedRegistry
from skbq.config.metadata import capture_experiment_metadata
from skbq.data.artifacts import VocabularyArtifactManifest, VocabularyArtifactWriter
from skbq.data.preparation import DataPreparationConfig, REQUIRED_TRAIN_MODEL_FAMILIES
from skbq.evaluation.oov import (
    OOVDataset,
    architecture_family_holdout,
    operator_class_holdout,
)
from skbq.vocabulary import VocabularyBuildRequest, VocabularyBuilder, VocabularyStore


@dataclass(frozen=True, slots=True)
class DataPreparationResult:
    """Result of one deterministic data preparation run."""

    config: DataPreparationConfig
    vocabulary: VocabularyStore
    oov_dataset: OOVDataset
    artifact_manifest: VocabularyArtifactManifest
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class DataPreparationPipeline:
    """Build real vocabulary artifacts and compatible OOV records."""

    vocabulary_builder: VocabularyBuilder = field(default_factory=VocabularyBuilder)
    artifact_writer: VocabularyArtifactWriter = field(default_factory=VocabularyArtifactWriter)
    repo_path: Path = Path(".")

    def run(self, config: DataPreparationConfig) -> DataPreparationResult:
        """Run data preparation for supported transformer checkpoint paths."""

        seed_registry = SeedRegistry({"data_preparation": config.random_seed})
        requests = tuple(
            VocabularyBuildRequest(
                model_spec=config.model_path(family),
                source_model=family,
                model_version=config.model_version(family),
                extraction_timestamp=config.extraction_timestamp,
            )
            for family in REQUIRED_TRAIN_MODEL_FAMILIES
        )
        vocabulary = self.vocabulary_builder.build_from_models(requests)
        split = _build_split(config, vocabulary)
        oov_dataset = OOVDataset.from_vocabulary(vocabulary, split)
        metadata = self._metadata(config, vocabulary, oov_dataset, seed_registry)
        manifest = self.artifact_writer.write(
            vocabulary=vocabulary,
            output_artifact_dir=config.output_artifact_dir,
            metadata=metadata,
        )
        return DataPreparationResult(
            config=config,
            vocabulary=vocabulary,
            oov_dataset=oov_dataset,
            artifact_manifest=manifest,
            metadata=metadata,
        )

    def _metadata(
        self,
        config: DataPreparationConfig,
        vocabulary: VocabularyStore,
        oov_dataset: OOVDataset,
        seed_registry: SeedRegistry,
    ) -> dict[str, object]:
        captured = capture_experiment_metadata(
            repo_path=self.repo_path,
            package_names=("skb-q-framework",),
            experiment_id="data_preparation",
        )
        return {
            "pipeline": "real_vocabulary_oov_data_preparation",
            "framework_version": __version__,
            "config": config.to_mapping(),
            "metadata": captured.to_mapping(),
            "seed_values": seed_registry.to_mapping(),
            "vocabulary_operator_types": list(vocabulary.operator_types()),
            "oov_record_count": len(oov_dataset.records),
            "split": oov_dataset.split.to_mapping(),
        }


def _build_split(config: DataPreparationConfig, vocabulary: VocabularyStore):
    if config.split_protocol == "operator_class_holdout":
        return operator_class_holdout(vocabulary, config.heldout_operator_types)
    if config.split_protocol == "architecture_family_holdout":
        return architecture_family_holdout(vocabulary, config.heldout_architectures)
    raise ValueError(f"unsupported split protocol: {config.split_protocol}")
