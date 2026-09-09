"""Model-agnostic graph extraction pipeline for SKB-Q."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from skbq.bridge.structural_features import OperatorStructuralMetadata
from skbq.graph.operator_graph import OperatorGraph


class GraphExtractor(Protocol):
    """Protocol for graph extractors over arbitrary model specifications."""

    def extract(self, model_spec: object) -> OperatorGraph:
        """Return an operator graph for a model specification."""


@dataclass(frozen=True, slots=True)
class GraphExtractionPipeline:
    """Small pipeline wrapper around a model-agnostic graph extractor."""

    extractor: GraphExtractor

    def extract_graph(self, model_spec: object) -> OperatorGraph:
        """Extract and validate an operator graph."""

        return self.extractor.extract(model_spec)

    def extract_structural_metadata(
        self,
        model_spec: object,
    ) -> Mapping[str, OperatorStructuralMetadata]:
        """Extract structural metadata for every operator in a model spec."""

        return self.extract_graph(model_spec).structural_metadata()
