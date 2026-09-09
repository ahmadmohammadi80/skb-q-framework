"""Decode graph policy logits into AgentQ allocation decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from skbq.agentq.provenance import AgentQProvenance
from skbq.agentq.state import GraphState, OperatorState
from skbq.quantization.action_space import PolicyActionSpace
from skbq.quantization.budget import BitBudget
from skbq.quantization.candidates import BitWidthCandidate, default_bit_width_candidates, zero_bit_width_candidate
from skbq.quantization.constraints import ConstraintSet, default_constraint_set
from skbq.quantization.operator_allocation import OperatorAllocation, OperatorAllocationPlan
from skbq.quantization.provenance import AllocationProvenance

if TYPE_CHECKING:
    import torch


CANONICAL_BIT_WIDTHS: tuple[int, ...] = tuple(
    candidate.bit_width for candidate in default_bit_width_candidates()
)


@dataclass(frozen=True, slots=True)
class ActionDecoder:
    """Decode per-operator logits into ``OperatorAllocationPlan`` decisions."""

    action_space: PolicyActionSpace = field(default_factory=PolicyActionSpace)
    constraints: ConstraintSet = field(default_factory=default_constraint_set)
    decoder_id: str = "agentq_action_decoder"
    decoder_version: str = "reference"
    canonical_bit_widths: tuple[int, ...] = CANONICAL_BIT_WIDTHS

    def decode(
        self,
        graph_state: GraphState,
        logits: torch.Tensor,
        bit_budget: BitBudget,
        temperature: float = 1.0,
        deterministic: bool = True,
        quantization_groups: Mapping[str, str] | None = None,
    ) -> OperatorAllocationPlan:
        """Decode logits into a validated operator allocation plan."""

        decoded_actions = self._decode_actions(
            graph_state=graph_state,
            logits=logits,
            temperature=temperature,
            deterministic=deterministic,
        )
        allocations = self._build_allocations(
            graph_state=graph_state,
            decoded_actions=decoded_actions,
            quantization_groups=quantization_groups,
        )
        projected = self._project_to_budget(allocations, bit_budget)
        plan = OperatorAllocationPlan(
            allocations=projected,
            provenance=self._allocation_provenance(graph_state),
            metadata={
                "notation": "pi_theta(G)->allocation",
                "decoder_id": self.decoder_id,
                "temperature": float(temperature),
                "deterministic": deterministic,
                "bit_budget": bit_budget.to_mapping(),
                "graph_state_hash": graph_state.state_hash(),
            },
        )
        self.constraints.validate(plan, bit_budget)
        return plan

    def decode_to_prediction(
        self,
        graph_state: GraphState,
        logits: torch.Tensor,
        bit_budget: BitBudget,
        temperature: float = 1.0,
        deterministic: bool = True,
        quantization_groups: Mapping[str, str] | None = None,
    ) -> object:
        """Decode logits into ``AgentQPrediction`` with allocation metadata."""

        from skbq.agentq.policy import AgentQPrediction

        plan = self.decode(
            graph_state=graph_state,
            logits=logits,
            bit_budget=bit_budget,
            temperature=temperature,
            deterministic=deterministic,
            quantization_groups=quantization_groups,
        )
        return _prediction_from_plan(
            prediction_type=AgentQPrediction,
            graph_state=graph_state,
            plan=plan,
            temperature=temperature,
            deterministic=deterministic,
        )

    def _decode_actions(
        self,
        graph_state: GraphState,
        logits: torch.Tensor,
        temperature: float,
        deterministic: bool,
    ) -> list[tuple[OperatorState, BitWidthCandidate, float, int]]:
        torch = _import_torch()

        if not isinstance(graph_state, GraphState):
            raise TypeError("graph_state must be a GraphState")
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim != 2:
            raise ValueError("logits must have shape [num_operators, num_actions]")
        if logits.shape[0] != graph_state.node_count:
            raise ValueError("logits row count must match graph_state node count")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("temperature must be a positive finite number")

        decoded: list[tuple[OperatorState, BitWidthCandidate, float, int]] = []
        for row_index, operator in enumerate(graph_state.operators):
            candidates = self.action_space.candidates_for_id(
                operator.operator_id,
                "graph_operator",
            )
            masked_logits = self._mask_logits(
                logits[row_index],
                allowed_widths={candidate.bit_width for candidate in candidates},
            )
            action_index, policy_mass = self._select_action(
                masked_logits,
                temperature=temperature,
                deterministic=deterministic,
            )
            candidate = self._candidate_for_index(action_index, candidates)
            decoded.append((operator, candidate, policy_mass, action_index))
        return decoded

    def _mask_logits(
        self,
        logits_row: torch.Tensor,
        allowed_widths: set[int],
    ) -> torch.Tensor:
        masked = logits_row.clone()
        action_count = min(masked.shape[0], len(self.canonical_bit_widths))
        for action_index in range(action_count):
            bit_width = self.canonical_bit_widths[action_index]
            if bit_width not in allowed_widths:
                masked[action_index] = float("-inf")
        return masked

    def _select_action(
        self,
        masked_logits: torch.Tensor,
        temperature: float,
        deterministic: bool,
    ) -> tuple[int, float]:
        torch = _import_torch()

        scaled_logits = masked_logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        if deterministic:
            action_index = int(torch.argmax(probabilities).item())
        else:
            action_index = int(torch.multinomial(probabilities, num_samples=1).item())
        policy_mass = float(probabilities[action_index].item())
        return action_index, policy_mass

    def _candidate_for_index(
        self,
        action_index: int,
        candidates: Sequence[BitWidthCandidate],
    ) -> BitWidthCandidate:
        if action_index < 0 or action_index >= len(self.canonical_bit_widths):
            raise ValueError("action_index out of range for canonical bit widths")
        target_width = self.canonical_bit_widths[action_index]
        for candidate in candidates:
            if candidate.bit_width == target_width:
                return candidate
        raise ValueError(f"no candidate matches canonical action width {target_width}")

    def _build_allocations(
        self,
        graph_state: GraphState,
        decoded_actions: Sequence[tuple[OperatorState, BitWidthCandidate, float, int]],
        quantization_groups: Mapping[str, str] | None,
    ) -> list[OperatorAllocation]:
        groups = dict(quantization_groups or {})
        allocations: list[OperatorAllocation] = []
        for operator, candidate, policy_mass, action_index in decoded_actions:
            allocations.append(
                OperatorAllocation(
                    target_id=operator.operator_id,
                    target_type="graph_operator",
                    operator_id=operator.operator_id,
                    layer_id=operator.layer_id,
                    quantization_group_id=groups.get(operator.operator_id),
                    bit_width_candidate=candidate,
                    parameter_count=operator.parameter_count,
                    policy_mass=policy_mass,
                    metadata={
                        "operator_type": operator.operator_type,
                        "action_index": action_index,
                        "depth_position": operator.metadata.get("depth_position"),
                    },
                )
            )
        return allocations

    def _project_to_budget(
        self,
        allocations: Sequence[OperatorAllocation],
        bit_budget: BitBudget,
    ) -> tuple[OperatorAllocation, ...]:
        current = list(allocations)
        total_bits = sum(item.storage_bits for item in current)
        if total_bits <= bit_budget.total_bits:
            return tuple(current)

        downgrade_order = sorted(
            current,
            key=lambda item: (item.policy_mass, item.target_id),
        )
        zero_candidate = zero_bit_width_candidate()
        for allocation in downgrade_order:
            if sum(item.storage_bits for item in current) <= bit_budget.total_bits:
                break
            current = [
                _replace_candidate(item, zero_candidate)
                if item.target_id == allocation.target_id
                else item
                for item in current
            ]
        return tuple(current)

    def _allocation_provenance(self, graph_state: GraphState) -> AllocationProvenance:
        return AllocationProvenance(
            allocator_id=self.decoder_id,
            allocator_version=self.decoder_version,
            policy_name="pi_theta",
            config={
                "decoding_rule": "masked_softmax_argmax",
                "canonical_bit_widths": list(self.canonical_bit_widths),
                "graph_state_hash": graph_state.state_hash(),
            },
        )


def _prediction_from_plan(
    prediction_type: type,
    graph_state: GraphState,
    plan: OperatorAllocationPlan,
    temperature: float,
    deterministic: bool,
) -> object:
    bit_widths = {
        allocation.target_id: allocation.bit_width_candidate.bit_width
        for allocation in plan.allocations
    }
    return prediction_type(
        graph_identifier=graph_state.graph_identifier,
        operator_ids=graph_state.operator_ids,
        metadata={
            "notation": "pi_theta(G)",
            "state_hash": graph_state.state_hash(),
            "allocation_hash": plan.allocation_hash(),
            "bit_widths": bit_widths,
            "temperature": float(temperature),
            "deterministic": deterministic,
            "total_storage_bits": plan.total_storage_bits,
        },
    )


def _replace_candidate(
    allocation: OperatorAllocation,
    candidate: BitWidthCandidate,
) -> OperatorAllocation:
    return OperatorAllocation(
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


def _import_torch() -> object:
    import torch

    return torch
