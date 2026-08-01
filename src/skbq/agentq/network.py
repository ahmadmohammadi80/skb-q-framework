"""Inference-only graph policy network scaffold for AgentQ."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from skbq.agentq.state import GraphState, OperatorState


@dataclass(frozen=True, slots=True)
class GraphPolicyOutput:
    """Per-operator logits produced by ``GraphPolicyNetwork``."""

    operator_ids: tuple[str, ...]
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if self.logits.ndim != 2:
            raise ValueError("logits must have shape [num_operators, num_actions]")
        if self.logits.shape[0] != len(self.operator_ids):
            raise ValueError("logits row count must match operator_ids length")


class GraphPolicyNetwork(nn.Module):
    """Lightweight inference-only graph policy network over ``GraphState``."""

    def __init__(
        self,
        num_actions: int = 4,
        hidden_dim: int = 64,
        structural_dim: int = 6,
        embedding_dim: int = 6,
    ) -> None:
        super().__init__()
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.structural_dim = structural_dim
        self.embedding_dim = embedding_dim
        self.metadata_dim = 4
        self.input_dim = structural_dim + embedding_dim + self.metadata_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, graph_state: GraphState) -> GraphPolicyOutput:
        """Return per-operator logits for operators in ``graph_state`` order."""

        if not isinstance(graph_state, GraphState):
            raise TypeError("graph_state must be a GraphState")

        operator_ids = graph_state.operator_ids
        features = self._operator_features(graph_state)
        depths = self._depth_positions(graph_state)
        neighbor_features = self._aggregate_neighbors(features, depths)
        hidden = features + neighbor_features
        logits = self.mlp(hidden)

        return GraphPolicyOutput(operator_ids=operator_ids, logits=logits)

    def _operator_features(self, graph_state: GraphState) -> torch.Tensor:
        rows = [
            self._features_for_operator(operator, graph_state.metadata)
            for operator in graph_state.operators
        ]
        return torch.stack(rows, dim=0)

    def _features_for_operator(
        self,
        operator: OperatorState,
        graph_metadata: object,
    ) -> torch.Tensor:
        structural = torch.tensor(operator.structural_features, dtype=torch.float32)
        if operator.encoder_embedding is None:
            embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
        else:
            embedding = torch.tensor(operator.encoder_embedding, dtype=torch.float32)

        max_depth = _metadata_float(graph_metadata, "max_depth_position", default=1.0)
        depth = _operator_metadata_float(operator, "depth_position", default=0.0)
        normalized_depth = depth / max(max_depth, 1.0)
        input_degree = _operator_metadata_float(operator, "input_degree", default=0.0)
        output_degree = _operator_metadata_float(operator, "output_degree", default=0.0)
        log_parameter_count = math.log1p(float(operator.parameter_count))
        reference_count = _metadata_float(
            graph_metadata,
            "reference_parameter_count",
            default=max(float(operator.parameter_count), 1.0),
        )
        normalized_log_parameter = log_parameter_count / math.log1p(max(reference_count, 1.0))

        metadata = torch.tensor(
            [
                normalized_depth,
                input_degree,
                output_degree,
                normalized_log_parameter,
            ],
            dtype=torch.float32,
        )
        return torch.cat((structural, embedding, metadata), dim=0)

    def _depth_positions(self, graph_state: GraphState) -> torch.Tensor:
        depths = [
            _operator_metadata_float(operator, "depth_position", default=0.0)
            for operator in graph_state.operators
        ]
        return torch.tensor(depths, dtype=torch.float32)

    def _aggregate_neighbors(
        self,
        features: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate parent/child depth neighbors using metadata depth positions."""

        neighbor_features = torch.zeros_like(features)
        for index, depth in enumerate(depths):
            parent_mask = depths == (depth - 1.0)
            child_mask = depths == (depth + 1.0)
            neighbor_parts: list[torch.Tensor] = []
            if bool(parent_mask.any()):
                neighbor_parts.append(features[parent_mask].mean(dim=0))
            if bool(child_mask.any()):
                neighbor_parts.append(features[child_mask].mean(dim=0))
            if neighbor_parts:
                neighbor_features[index] = torch.stack(neighbor_parts, dim=0).mean(dim=0)
        return neighbor_features


def _metadata_float(metadata: object, key: str, default: float) -> float:
    if not isinstance(metadata, dict):
        return default
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _operator_metadata_float(operator: OperatorState, key: str, default: float) -> float:
    value = operator.metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)
