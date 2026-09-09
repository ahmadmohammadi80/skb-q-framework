"""Deterministic trace objects for SKB-Q bridge decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from skbq.bridge.candidate_filter import BridgeCandidate, CandidateMatch
from skbq.bridge.embedding_composer import CompositionResult
from skbq.bridge.similarity import CandidateSimilarity


@dataclass(frozen=True, slots=True)
class BridgeTrace:
    """Complete deterministic path from operator evidence to final ``psi(o)``."""

    source_identifier: str
    structural_features: tuple[float, ...]
    candidate_set: tuple[Mapping[str, object], ...]
    similarity_scores: tuple[Mapping[str, object], ...]
    softmax_weights: tuple[Mapping[str, object], ...]
    confidence_gate: Mapping[str, object]
    final_embedding: tuple[float, ...]
    selected_candidate_id: str
    used_fallback: bool
    policy_decision: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_features", tuple(self.structural_features))
        object.__setattr__(self, "candidate_set", tuple(MappingProxyType(dict(item)) for item in self.candidate_set))
        object.__setattr__(
            self,
            "similarity_scores",
            tuple(MappingProxyType(dict(item)) for item in self.similarity_scores),
        )
        object.__setattr__(
            self,
            "softmax_weights",
            tuple(MappingProxyType(dict(item)) for item in self.softmax_weights),
        )
        object.__setattr__(self, "confidence_gate", MappingProxyType(dict(self.confidence_gate)))
        object.__setattr__(self, "final_embedding", tuple(self.final_embedding))
        if self.policy_decision is not None:
            object.__setattr__(self, "policy_decision", MappingProxyType(dict(self.policy_decision)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_components(
        cls,
        source_identifier: str,
        structural_features: Sequence[float],
        candidate_matches: Sequence[CandidateMatch],
        similarities: Sequence[CandidateSimilarity],
        composition: CompositionResult,
        selected_candidate: BridgeCandidate,
        used_fallback: bool,
        policy_decision: Mapping[str, object] | None = None,
    ) -> BridgeTrace:
        """Build a bridge trace from deterministic bridge components."""

        return cls(
            source_identifier=source_identifier,
            structural_features=tuple(float(value) for value in structural_features),
            candidate_set=tuple(
                {
                    "candidate_id": match.candidate.identifier,
                    "structural_distance": match.distance,
                    "vocabulary_index": match.vocabulary_index,
                    "normalized_query": list(match.normalized_query),
                    "normalized_candidate": list(match.normalized_candidate),
                }
                for match in candidate_matches
            ),
            similarity_scores=tuple(
                {
                    "candidate_id": similarity.candidate.identifier,
                    "semantic": similarity.semantic,
                    "structural": similarity.structural,
                    "functional": similarity.functional,
                    "aggregate": similarity.aggregate,
                    "structural_distance": similarity.structural_distance,
                }
                for similarity in similarities
            ),
            softmax_weights=tuple(
                {
                    "candidate_id": weight.candidate.identifier,
                    "weight": weight.weight,
                    "score": weight.score,
                }
                for weight in composition.candidate_weights
            ),
            confidence_gate={
                "confidence": composition.confidence,
                "entropy": composition.entropy,
                "passed": composition.passed_confidence_gate,
            },
            final_embedding=composition.embedding,
            selected_candidate_id=selected_candidate.identifier,
            used_fallback=used_fallback,
            policy_decision=policy_decision,
            metadata={"notation": "Operator -> g(t) -> C_k -> s -> softmax -> psi(o)"},
        )

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable trace mapping."""

        return {
            "source_identifier": self.source_identifier,
            "structural_features": list(self.structural_features),
            "candidate_set": [dict(item) for item in self.candidate_set],
            "similarity_scores": [dict(item) for item in self.similarity_scores],
            "softmax_weights": [dict(item) for item in self.softmax_weights],
            "confidence_gate": dict(self.confidence_gate),
            "final_embedding": list(self.final_embedding),
            "selected_candidate_id": self.selected_candidate_id,
            "used_fallback": self.used_fallback,
            "policy_decision": None if self.policy_decision is None else dict(self.policy_decision),
            "metadata": dict(self.metadata),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def trace_hash(self) -> str:
        """Return stable SHA-256 hash for this bridge trace."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
