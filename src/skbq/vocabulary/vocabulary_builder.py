"""Build reproducible SKB-Q operator vocabularies from supported models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from skbq import __version__
from skbq.config.metadata import current_git_commit_hash
from skbq.graph.operator_graph import OperatorGraph
from skbq.models.hf_adapter import HuggingFaceGraphExtractor
from skbq.vocabulary.vocabulary_store import (
    VocabularySourceProvenance,
    VocabularyStore,
)


@dataclass(frozen=True, slots=True)
class VocabularyBuildRequest:
    """One model vocabulary build request with optional provenance overrides."""

    model_spec: object
    source_model: str | None = None
    model_version: str | None = None
    extraction_timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class VocabularyBuilder:
    """Build unified vocabulary ``V`` from supported transformer models."""

    extractor: object = field(default_factory=HuggingFaceGraphExtractor)
    repo_path: Path = Path(".")
    framework_version: str = __version__

    def build_from_model(
        self,
        model_spec: object,
        source_model: str | None = None,
        model_version: str | None = None,
        extraction_timestamp: str | None = None,
    ) -> VocabularyStore:
        """Load/extract one model graph and convert operators into vocabulary entries."""

        graph = self.extractor.extract(model_spec)
        if not isinstance(graph, OperatorGraph):
            raise TypeError("vocabulary extractor must return an OperatorGraph")

        provenance = VocabularySourceProvenance(
            source_model=source_model or _source_model(model_spec, graph),
            model_version=model_version or _model_version(model_spec),
            extraction_timestamp=extraction_timestamp or _timestamp(),
            git_commit=current_git_commit_hash(self.repo_path),
            framework_version=self.framework_version,
        )
        return VocabularyStore.from_graph(graph, provenance)

    def build_from_models(
        self,
        model_specs: tuple[object, ...],
    ) -> VocabularyStore:
        """Build and merge vocabularies from multiple models."""

        if not model_specs:
            raise ValueError("at least one model spec is required")
        stores = tuple(
            self._build_from_item(model_spec)
            for model_spec in model_specs
        )
        return VocabularyStore.merge_many(stores)

    def _build_from_item(self, item: object) -> VocabularyStore:
        if isinstance(item, VocabularyBuildRequest):
            return self.build_from_model(
                item.model_spec,
                source_model=item.source_model,
                model_version=item.model_version,
                extraction_timestamp=item.extraction_timestamp,
            )
        return self.build_from_model(item)


def _source_model(model_spec: object, graph: OperatorGraph) -> str:
    if isinstance(model_spec, str):
        return model_spec
    for attribute in ("model_name_or_path", "name_or_path", "_name_or_path"):
        value = getattr(model_spec, attribute, None)
        if isinstance(value, str) and value:
            return value
    config = getattr(model_spec, "config", None)
    if config is not None:
        for attribute in ("name_or_path", "_name_or_path"):
            value = getattr(config, attribute, None)
            if isinstance(value, str) and value:
                return value
    return graph.architecture


def _model_version(model_spec: object) -> str | None:
    for attribute in ("model_version", "revision", "commit_hash"):
        value = getattr(model_spec, attribute, None)
        if isinstance(value, str) and value:
            return value
    config = getattr(model_spec, "config", None)
    if config is not None:
        for attribute in ("revision", "_commit_hash", "commit_hash"):
            value = getattr(config, attribute, None)
            if isinstance(value, str) and value:
                return value
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
