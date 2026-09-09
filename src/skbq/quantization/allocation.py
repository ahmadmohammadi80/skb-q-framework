"""Quantization allocation layer connecting frozen policy ``pi_theta`` to operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from skbq.backbone.integration import allocation_targets_for_graph
from skbq.backbone.policy_interface import FrozenPolicy, PolicyAllocation, UniformFrozenPolicy
from skbq.graph.operator_graph import OperatorGraph
from skbq.quantization.action_space import PolicyActionSpace
from skbq.quantization.budget import BitBudget
from skbq.quantization.constraints import ConstraintSet, default_constraint_set
from skbq.quantization.operator_allocation import OperatorAllocationPlan
from skbq.quantization.resolver import DeterministicAllocationResolver


@dataclass(frozen=True, slots=True)
class QuantizationAllocationLayer:
    """Research layer mapping ``pi_theta`` output ``P(G)`` to operator bit-width plans."""

    policy: FrozenPolicy = field(default_factory=UniformFrozenPolicy)
    action_space: PolicyActionSpace = field(default_factory=PolicyActionSpace)
    constraints: ConstraintSet = field(default_factory=default_constraint_set)
    resolver: DeterministicAllocationResolver | None = None

    def allocate_graph(
        self,
        graph: OperatorGraph,
        bit_budget: BitBudget,
        graph_embedding: tuple[float, ...] | None = None,
        quantization_groups: Mapping[str, str] | None = None,
    ) -> OperatorAllocationPlan:
        """Run ``pi_theta`` on graph embedding ``G`` and resolve operator allocations."""

        parameter_counts = _parameter_counts_for_graph(graph)
        policy_allocation = self._policy_allocation(
            graph,
            bit_budget,
            graph_embedding,
        )
        resolver = self.resolver or DeterministicAllocationResolver(
            action_space=self.action_space,
            constraints=self.constraints,
        )
        return resolver.resolve(
            policy_allocation=policy_allocation,
            bit_budget=bit_budget,
            parameter_counts=parameter_counts,
            quantization_groups=quantization_groups,
        )

    def resolve_policy_allocation(
        self,
        policy_allocation: PolicyAllocation,
        bit_budget: BitBudget,
        parameter_counts: Mapping[str, int],
        quantization_groups: Mapping[str, str] | None = None,
    ) -> OperatorAllocationPlan:
        """Resolve an existing ``P(G)`` distribution without re-running ``pi_theta``."""

        resolver = self.resolver or DeterministicAllocationResolver(
            action_space=self.action_space,
            constraints=self.constraints,
        )
        return resolver.resolve(
            policy_allocation=policy_allocation,
            bit_budget=bit_budget,
            parameter_counts=parameter_counts,
            quantization_groups=quantization_groups,
        )

    def _policy_allocation(
        self,
        graph: OperatorGraph,
        bit_budget: BitBudget,
        graph_embedding: tuple[float, ...] | None,
    ) -> PolicyAllocation:
        targets = allocation_targets_for_graph(graph)
        budget_mass = float(bit_budget.total_bits)
        if graph_embedding is None:
            graph_embedding = _structural_graph_embedding(graph)
        return self.policy.allocate(
            graph_embedding,
            budget_mass,
            targets=targets,
        )


def _parameter_counts_for_graph(graph: OperatorGraph) -> dict[str, int]:
    return {node.operator_id: node.parameter_count for node in graph.nodes}


def _structural_graph_embedding(graph: OperatorGraph) -> tuple[float, ...]:
    """Return deterministic structural embedding without invoking a frozen encoder."""

    if not graph.nodes:
        return (0.0,)
    total_parameters = float(sum(node.parameter_count for node in graph.nodes))
    max_depth = float(graph.max_depth_position)
    return (
        total_parameters,
        float(len(graph.nodes)),
        float(graph.max_input_degree),
        float(graph.max_output_degree),
        max_depth,
    )
