"""Quantization constraints for operator-level allocation plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skbq.quantization.budget import BitBudget
    from skbq.quantization.operator_allocation import OperatorAllocationPlan


class QuantizationConstraint(ABC):
    """Abstract constraint evaluated against an operator allocation plan."""

    @property
    @abstractmethod
    def constraint_id(self) -> str:
        """Return stable identifier for this constraint."""

    @abstractmethod
    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        """Raise ``ValueError`` when the plan violates this constraint."""

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable constraint metadata."""

        return {"constraint_id": self.constraint_id, "constraint_type": self.__class__.__name__}


@dataclass(frozen=True, slots=True)
class TotalBudgetConstraint(QuantizationConstraint):
    """Require total allocated storage bits to stay within ``BitBudget``."""

    @property
    def constraint_id(self) -> str:
        return "total_budget"

    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        if plan.total_storage_bits > bit_budget.total_bits:
            raise ValueError(
                f"allocation uses {plan.total_storage_bits} bits, "
                f"exceeding budget of {bit_budget.total_bits}"
            )


@dataclass(frozen=True, slots=True)
class MinBitWidthConstraint(QuantizationConstraint):
    """Require every allocation to use at least ``min_bit_width``."""

    min_bit_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.min_bit_width, int) or isinstance(self.min_bit_width, bool):
            raise TypeError("min_bit_width must be an integer")
        if self.min_bit_width <= 0:
            raise ValueError("min_bit_width must be positive")

    @property
    def constraint_id(self) -> str:
        return f"min_bit_width_{self.min_bit_width}"

    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        for allocation in plan.allocations:
            if allocation.bit_width_candidate.bit_width < self.min_bit_width:
                raise ValueError(
                    f"target {allocation.target_id!r} uses bit width "
                    f"{allocation.bit_width_candidate.bit_width}, "
                    f"below minimum {self.min_bit_width}"
                )

    def to_mapping(self) -> dict[str, object]:
        payload = super().to_mapping()
        payload["min_bit_width"] = self.min_bit_width
        return payload


@dataclass(frozen=True, slots=True)
class MaxBitWidthConstraint(QuantizationConstraint):
    """Require every allocation to use at most ``max_bit_width``."""

    max_bit_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_bit_width, int) or isinstance(self.max_bit_width, bool):
            raise TypeError("max_bit_width must be an integer")
        if self.max_bit_width <= 0:
            raise ValueError("max_bit_width must be positive")

    @property
    def constraint_id(self) -> str:
        return f"max_bit_width_{self.max_bit_width}"

    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        for allocation in plan.allocations:
            if allocation.bit_width_candidate.bit_width > self.max_bit_width:
                raise ValueError(
                    f"target {allocation.target_id!r} uses bit width "
                    f"{allocation.bit_width_candidate.bit_width}, "
                    f"above maximum {self.max_bit_width}"
                )

    def to_mapping(self) -> dict[str, object]:
        payload = super().to_mapping()
        payload["max_bit_width"] = self.max_bit_width
        return payload


@dataclass(frozen=True, slots=True)
class GroupUniformityConstraint(QuantizationConstraint):
    """Require identical bit width within each quantization group."""

    group_members: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized = {
            str(target_id): str(group_id)
            for target_id, group_id in sorted(self.group_members.items(), key=lambda item: item[0])
        }
        object.__setattr__(self, "group_members", normalized)

    @property
    def constraint_id(self) -> str:
        return "group_uniformity"

    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        widths_by_group: dict[str, set[int]] = {}
        for allocation in plan.allocations:
            group_id = self.group_members.get(allocation.target_id)
            if group_id is None:
                continue
            widths_by_group.setdefault(group_id, set()).add(
                allocation.bit_width_candidate.bit_width
            )
        for group_id, widths in sorted(widths_by_group.items()):
            if len(widths) > 1:
                raise ValueError(
                    f"quantization group {group_id!r} has mixed bit widths: {sorted(widths)}"
                )

    def to_mapping(self) -> dict[str, object]:
        payload = super().to_mapping()
        payload["group_members"] = dict(self.group_members)
        return payload


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    """Ordered collection of quantization constraints."""

    constraints: tuple[QuantizationConstraint, ...] = ()

    def validate(self, plan: OperatorAllocationPlan, bit_budget: BitBudget) -> None:
        """Validate a plan against every registered constraint."""

        for constraint in self.constraints:
            constraint.validate(plan, bit_budget)

    def to_mapping(self) -> list[dict[str, object]]:
        """Return deterministic JSON-serializable constraint metadata."""

        return [constraint.to_mapping() for constraint in self.constraints]


def default_constraint_set() -> ConstraintSet:
    """Return default constraints for research scaffolding."""

    return ConstraintSet(constraints=(TotalBudgetConstraint(),))
