"""Bit budget abstractions for quantization allocation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

from skbq.config.schema import BudgetConfig


@dataclass(frozen=True, slots=True)
class BitBudget:
    """Total bit budget for operator-level quantization allocation."""

    total_bits: int
    unit: str = "bits"

    def __post_init__(self) -> None:
        if not isinstance(self.total_bits, int) or isinstance(self.total_bits, bool):
            raise TypeError("total_bits must be an integer")
        if self.total_bits < 0:
            raise ValueError("total_bits must be non-negative")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be a non-empty string")

    @classmethod
    def from_budget_config(cls, config: BudgetConfig) -> BitBudget:
        """Build a bit budget from validated experiment budget configuration."""

        if config.unit != "bits" and config.unit != "generic":
            raise ValueError(f"unsupported budget unit for bit allocation: {config.unit}")
        total = _integral_bits(config.total, "budget.total")
        return cls(total_bits=total, unit="bits")

    def consumed_bits(self, used_bits: int) -> BitBudget:
        """Return remaining budget after consuming ``used_bits``."""

        if used_bits < 0:
            raise ValueError("used_bits must be non-negative")
        remaining = self.total_bits - used_bits
        if remaining < 0:
            raise ValueError("used_bits exceeds total_bits")
        return BitBudget(total_bits=remaining, unit=self.unit)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable mapping."""

        return {"total_bits": self.total_bits, "unit": self.unit}

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON serialization."""

        import json

        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))


def _integral_bits(value: float, field_name: str) -> int:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return int(result)
