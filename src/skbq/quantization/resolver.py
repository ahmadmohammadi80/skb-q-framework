"""Deterministic resolver from policy distribution ``P(G)`` to operator allocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from skbq.backbone.policy_interface import PolicyAllocation, TargetAllocation
from skbq.quantization.action_space import PolicyActionSpace
from skbq.quantization.budget import BitBudget
from skbq.quantization.candidates import BitWidthCandidate, zero_bit_width_candidate
from skbq.quantization.constraints import ConstraintSet, default_constraint_set
from skbq.quantization.operator_allocation import OperatorAllocation, OperatorAllocationPlan
from skbq.quantization.provenance import AllocationProvenance


@dataclass(frozen=True, slots=True)
class DeterministicAllocationResolver:
    """Reference resolver mapping ``P(G)`` mass to discrete operator bit widths."""

    action_space: PolicyActionSpace = PolicyActionSpace()
    constraints: ConstraintSet = default_constraint_set()
    allocator_id: str = "deterministic_mass_resolver"
    allocator_version: str = "reference"

    def resolve(
        self,
        policy_allocation: PolicyAllocation,
        bit_budget: BitBudget,
        parameter_counts: Mapping[str, int],
        quantization_groups: Mapping[str, str] | None = None,
    ) -> OperatorAllocationPlan:
        """Resolve ``P(G)`` into operator-level bit-width assignments."""

        groups = dict(quantization_groups or {})
        initial_allocations = self._greedy_initial_allocations(
            policy_allocation,
            bit_budget,
            parameter_counts,
            groups,
        )
        upgraded = self._upgrade_allocations(
            initial_allocations,
            bit_budget,
            groups,
        )
        plan = OperatorAllocationPlan(
            allocations=upgraded,
            provenance=self._provenance(policy_allocation),
            metadata={
                "notation": "P(G)->operator_allocation",
                "policy_name": policy_allocation.policy_name,
                "policy_allocation_hash": policy_allocation.allocation_hash(),
                "bit_budget": bit_budget.to_mapping(),
            },
        )
        self.constraints.validate(plan, bit_budget)
        return plan

    def _greedy_initial_allocations(
        self,
        policy_allocation: PolicyAllocation,
        bit_budget: BitBudget,
        parameter_counts: Mapping[str, int],
        quantization_groups: Mapping[str, str],
    ) -> list[OperatorAllocation]:
        remaining_bits = bit_budget.total_bits
        assigned_groups: dict[str, BitWidthCandidate] = {}
        allocations: list[OperatorAllocation] = []
        sorted_targets = sorted(
            policy_allocation.target_allocations,
            key=lambda item: (-item.budget, item.target.target_id),
        )

        for target_allocation in sorted_targets:
            target = target_allocation.target
            parameter_count = _parameter_count_for(target.target_id, parameter_counts)
            group_id = quantization_groups.get(target.target_id)

            if group_id is not None and group_id in assigned_groups:
                candidate = assigned_groups[group_id]
            else:
                candidates = self.action_space.candidates_for(target)
                candidate = _best_candidate_within_budget(
                    candidates,
                    parameter_count,
                    remaining_bits,
                )
                if group_id is not None:
                    assigned_groups[group_id] = candidate

            storage_bits = candidate.storage_bits(parameter_count)
            if storage_bits > remaining_bits:
                candidate = zero_bit_width_candidate()
                storage_bits = 0

            remaining_bits -= storage_bits
            allocations.append(
                _build_allocation(
                    target_allocation,
                    candidate,
                    parameter_count,
                    group_id,
                )
            )
        return allocations

    def _upgrade_allocations(
        self,
        allocations: Sequence[OperatorAllocation],
        bit_budget: BitBudget,
        quantization_groups: Mapping[str, str],
    ) -> tuple[OperatorAllocation, ...]:
        current = list(allocations)
        upgraded = True
        while upgraded:
            upgraded = False
            priority = sorted(
                current,
                key=lambda item: (-item.policy_mass, item.target_id),
            )
            for allocation in priority:
                candidates = self.action_space.candidates_for_id(
                    allocation.target_id,
                    allocation.target_type,
                )
                current_width = allocation.bit_width_candidate.bit_width
                next_candidate = _next_candidate(candidates, current_width)
                if next_candidate is None:
                    continue
                candidate_allocations = _apply_candidate_upgrade(
                    current,
                    allocation.target_id,
                    next_candidate,
                    quantization_groups,
                )
                total_bits = sum(item.storage_bits for item in candidate_allocations)
                if total_bits <= bit_budget.total_bits:
                    current = candidate_allocations
                    upgraded = True
        return tuple(current)

    def _provenance(self, policy_allocation: PolicyAllocation) -> AllocationProvenance:
        return AllocationProvenance(
            allocator_id=self.allocator_id,
            allocator_version=self.allocator_version,
            policy_name=policy_allocation.policy_name,
            config={
                "resolution_rule": "greedy_mass_priority_upgrade",
                "constraint_ids": [
                    constraint.constraint_id for constraint in self.constraints.constraints
                ],
            },
        )


def _build_allocation(
    target_allocation: TargetAllocation,
    candidate: BitWidthCandidate,
    parameter_count: int,
    quantization_group_id: str | None,
) -> OperatorAllocation:
    target = target_allocation.target
    return OperatorAllocation(
        target_id=target.target_id,
        target_type=target.target_type,
        operator_id=target.operator_id,
        layer_id=target.layer_id,
        quantization_group_id=quantization_group_id,
        bit_width_candidate=candidate,
        parameter_count=parameter_count,
        policy_mass=target_allocation.budget,
        metadata=dict(target.metadata),
    )


def _parameter_count_for(target_id: str, parameter_counts: Mapping[str, int]) -> int:
    if target_id not in parameter_counts:
        raise KeyError(f"missing parameter_count for target {target_id!r}")
    value = parameter_counts[target_id]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"parameter_count for {target_id!r} must be a non-negative integer")
    return value


def _best_candidate_within_budget(
    candidates: Sequence[BitWidthCandidate],
    parameter_count: int,
    remaining_bits: int,
) -> BitWidthCandidate:
    feasible = tuple(
        candidate
        for candidate in candidates
        if candidate.storage_bits(parameter_count) <= remaining_bits
    )
    if feasible:
        return feasible[-1]
    return zero_bit_width_candidate()


def _next_candidate(
    candidates: Sequence[BitWidthCandidate],
    current_width: int,
) -> BitWidthCandidate | None:
    for candidate in candidates:
        if candidate.bit_width > current_width:
            return candidate
    return None


def _apply_candidate_upgrade(
    allocations: Sequence[OperatorAllocation],
    target_id: str,
    candidate: BitWidthCandidate,
    quantization_groups: Mapping[str, str],
) -> list[OperatorAllocation]:
    upgraded_group = quantization_groups.get(target_id)
    updated: list[OperatorAllocation] = []
    for allocation in allocations:
        same_group = (
            upgraded_group is not None
            and quantization_groups.get(allocation.target_id) == upgraded_group
        )
        if allocation.target_id == target_id or same_group:
            updated.append(
                OperatorAllocation(
                    target_id=allocation.target_id,
                    target_type=allocation.target_type,
                    operator_id=allocation.operator_id,
                    layer_id=allocation.layer_id,
                    quantization_group_id=allocation.quantization_group_id,
                    bit_width_candidate=candidate,
                    parameter_count=allocation.parameter_count,
                    policy_mass=allocation.policy_mass,
                    metadata=dict(allocation.metadata),
                )
            )
        else:
            updated.append(allocation)
    return updated
