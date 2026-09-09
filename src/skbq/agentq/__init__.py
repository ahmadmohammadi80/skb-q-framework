"""AgentQ learned policy layer for adaptive mixed-precision quantization."""

from skbq.agentq.policy import (
    AbstractAgentQPolicy,
    AgentQPolicy,
    AgentQPrediction,
    LearnedAgentQPolicy,
    StructuralReferenceAgentQPolicy,
)
from skbq.agentq.provenance import AgentQProvenance
from skbq.agentq.state import GraphState, OperatorState, StateBuilder

__all__ = [
    "AbstractAgentQPolicy",
    "ActionDecoder",
    "AgentQPolicy",
    "AgentQPrediction",
    "AgentQProvenance",
    "GraphPolicyNetwork",
    "GraphPolicyOutput",
    "GraphState",
    "LearnedAgentQPolicy",
    "OperatorState",
    "StateBuilder",
    "StructuralReferenceAgentQPolicy",
]


def __getattr__(name: str) -> object:
    if name == "ActionDecoder":
        from skbq.agentq.action_decoder import ActionDecoder

        return ActionDecoder
    if name == "GraphPolicyNetwork":
        from skbq.agentq.network import GraphPolicyNetwork

        return GraphPolicyNetwork
    if name == "GraphPolicyOutput":
        from skbq.agentq.network import GraphPolicyOutput

        return GraphPolicyOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
