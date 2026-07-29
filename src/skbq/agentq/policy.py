"""AgentQ policy interface for graph-conditioned mixed-precision allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from skbq.agentq.provenance import AgentQProvenance
from skbq.agentq.state import GraphState


@dataclass(frozen=True, slots=True)
class AgentQPrediction:
    """Deterministic policy output scaffold for AgentQ inference."""

    graph_identifier: str
    operator_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph_identifier.strip():
            raise ValueError("graph_identifier cannot be empty")
        normalized_ids = tuple(str(operator_id) for operator_id in self.operator_ids)
        object.__setattr__(self, "operator_ids", normalized_ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable prediction."""

        return {
            "graph_identifier": self.graph_identifier,
            "operator_ids": list(self.operator_ids),
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class AgentQPolicy(Protocol):
    """Protocol for AgentQ policies ``pi_theta`` over graph state."""

    def predict(self, graph_state: GraphState) -> AgentQPrediction:
        """Return a policy prediction for the given graph state."""

    def get_provenance(self) -> AgentQProvenance:
        """Return deterministic provenance for this policy."""


class AbstractAgentQPolicy(ABC):
    """Abstract base class sharing validation helpers for concrete AgentQ policies."""

    @abstractmethod
    def predict(self, graph_state: GraphState) -> AgentQPrediction:
        """Return a policy prediction for the given graph state."""

    @abstractmethod
    def get_provenance(self) -> AgentQProvenance:
        """Return deterministic provenance for this policy."""

    def _validate_graph_state(self, graph_state: GraphState) -> GraphState:
        if not isinstance(graph_state, GraphState):
            raise TypeError("graph_state must be a GraphState")
        return graph_state


@dataclass(frozen=True, slots=True)
class StructuralReferenceAgentQPolicy(AbstractAgentQPolicy):
    """Reference AgentQ policy using structural state only (no learned parameters)."""

    policy_id: str = "structural_reference_pi_theta"

    def predict(self, graph_state: GraphState) -> AgentQPrediction:
        """Return a deterministic structural reference prediction."""

        state = self._validate_graph_state(graph_state)
        return AgentQPrediction(
            graph_identifier=state.graph_identifier,
            operator_ids=state.operator_ids,
            metadata={
                "notation": "pi_theta(G)",
                "node_count": state.node_count,
                "state_hash": state.state_hash(),
            },
        )

    def get_provenance(self) -> AgentQProvenance:
        """Return provenance for the structural reference policy."""

        return AgentQProvenance(
            policy_id=self.policy_id,
            config={"policy_type": "structural_reference"},
        )
