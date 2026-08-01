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


class LearnedAgentQPolicy(AbstractAgentQPolicy):
    """Inference-only AgentQ policy connecting network logits to allocation decisions."""

    def __init__(
        self,
        network: object,
        bit_budget: object,
        decoder: object | None = None,
        state_builder: object | None = None,
        action_space: object | None = None,
        policy_id: str = "learned_agentq_pi_theta",
    ) -> None:
        from skbq.agentq.action_decoder import ActionDecoder
        from skbq.agentq.state import StateBuilder
        from skbq.quantization.action_space import PolicyActionSpace
        from skbq.quantization.budget import BitBudget

        if not policy_id.strip():
            raise ValueError("policy_id cannot be empty")
        if not isinstance(bit_budget, BitBudget):
            raise TypeError("bit_budget must be a BitBudget")

        self._network = network
        self._bit_budget = bit_budget
        self._state_builder = state_builder or StateBuilder()
        self._policy_id = policy_id
        selected_action_space = action_space or PolicyActionSpace()
        self._decoder = decoder or ActionDecoder(action_space=selected_action_space)

    def predict(self, graph_state: GraphState) -> AgentQPrediction:
        """Run graph policy inference and decode allocation decisions."""

        state = self._validate_graph_state(graph_state)
        network = self._network
        network.eval()
        output = network(state)
        return self._decoder.decode_to_prediction(
            graph_state=state,
            logits=output.logits,
            bit_budget=self._bit_budget,
            deterministic=True,
        )

    def predict_graph(self, graph: object) -> AgentQPrediction:
        """Build graph state from an operator graph and run inference."""

        from skbq.graph.operator_graph import OperatorGraph

        if not isinstance(graph, OperatorGraph):
            raise TypeError("graph must be an OperatorGraph")
        return self.predict(self._state_builder.build(graph))

    def get_provenance(self) -> AgentQProvenance:
        """Return provenance for this learned inference policy."""

        network = self._network
        return AgentQProvenance(
            policy_id=self._policy_id,
            config={
                "policy_type": "learned_inference",
                "decoder_id": self._decoder.decoder_id,
                "network_num_actions": getattr(network, "num_actions", None),
                "network_hidden_dim": getattr(network, "hidden_dim", None),
                "bit_budget": self._bit_budget.to_mapping(),
            },
        )
