"""Validated experiment configuration schema for SKB-Q research runs.

The schema is intentionally model-agnostic: it captures reproducibility-critical
parameters without instantiating models, running experiments, or producing
results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    tomllib = None  # type: ignore[assignment]


ConfigMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VocabularyConfig:
    """Vocabulary configuration for a reproducible experiment."""

    registry: str
    operators: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> VocabularyConfig:
        """Build vocabulary config from a mapping."""

        registry = _required_str(data, "registry")
        operators = tuple(_string_sequence(data.get("operators", ()), "operators"))
        return cls(registry=registry, operators=operators)

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"registry": self.registry, "operators": list(self.operators)}


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    """Backbone encoder/policy configuration without model instantiation."""

    encoder: str
    policy: str
    frozen: bool = True

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> BackboneConfig:
        """Build backbone config from a mapping."""

        return cls(
            encoder=_required_str(data, "encoder"),
            policy=_required_str(data, "policy"),
            frozen=_optional_bool(data, "frozen", True),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"encoder": self.encoder, "policy": self.policy, "frozen": self.frozen}


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """Budget configuration for allocation experiments."""

    total: float
    unit: str = "generic"

    @classmethod
    def from_value(cls, value: object) -> BudgetConfig:
        """Build budget config from either a number or a mapping."""

        if isinstance(value, Mapping):
            total = _required_float(value, "total")
            unit = _optional_str(value, "unit", "generic")
        else:
            total = _finite_float(value, "budget")
            unit = "generic"

        if total < 0.0:
            raise ValueError("budget total must be non-negative")
        return cls(total=total, unit=unit)

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"total": self.total, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class LambdaWeights:
    """Similarity-channel lambda weights from the mathematical formulation."""

    semantic: float
    structural: float
    functional: float

    def __post_init__(self) -> None:
        for name, value in (
            ("semantic", self.semantic),
            ("structural", self.structural),
            ("functional", self.functional),
        ):
            if value < 0.0:
                raise ValueError(f"lambda weight {name} must be non-negative")
        if self.total == 0.0:
            raise ValueError("at least one lambda weight must be positive")

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> LambdaWeights:
        """Build lambda weights from a mapping."""

        return cls(
            semantic=_required_float(data, "semantic"),
            structural=_required_float(data, "structural"),
            functional=_required_float(data, "functional"),
        )

    @property
    def total(self) -> float:
        """Return the unnormalized total lambda mass."""

        return self.semantic + self.structural + self.functional

    def normalized(self) -> tuple[float, float, float]:
        """Return lambda weights normalized to sum to one."""

        return (
            self.semantic / self.total,
            self.structural / self.total,
            self.functional / self.total,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "semantic": self.semantic,
            "structural": self.structural,
            "functional": self.functional,
        }


@dataclass(frozen=True, slots=True)
class RandomSeeds:
    """Named random seeds required for reproducible experiments."""

    values: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("at least one random seed is required")
        validated = {
            str(name): _seed_value(seed, f"random_seeds[{name}]")
            for name, seed in self.values.items()
        }
        object.__setattr__(self, "values", MappingProxyType(validated))

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> RandomSeeds:
        """Build random seed config from a mapping."""

        return cls(values=dict(data))

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level reproducible SKB-Q experiment configuration."""

    vocabulary: VocabularyConfig
    backbone: BackboneConfig
    budget: BudgetConfig
    tau: float
    k_prime: int
    confidence_threshold: float
    lambda_weights: LambdaWeights
    random_seeds: RandomSeeds

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("temperature tau must be positive")
        if self.k_prime <= 0:
            raise ValueError("k_prime must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

    @classmethod
    def from_mapping(cls, data: ConfigMapping) -> ExperimentConfig:
        """Build the full experiment config from a mapping."""

        return cls(
            vocabulary=VocabularyConfig.from_mapping(_required_mapping(data, "vocabulary")),
            backbone=BackboneConfig.from_mapping(_required_mapping(data, "backbone")),
            budget=BudgetConfig.from_value(_required_value(data, "budget")),
            tau=_required_float(data, "tau"),
            k_prime=_required_int(data, "k_prime"),
            confidence_threshold=_required_float(data, "confidence_threshold"),
            lambda_weights=LambdaWeights.from_mapping(
                _required_mapping(data, "lambda_weights")
            ),
            random_seeds=RandomSeeds.from_mapping(_required_mapping(data, "random_seeds")),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "vocabulary": self.vocabulary.to_mapping(),
            "backbone": self.backbone.to_mapping(),
            "budget": self.budget.to_mapping(),
            "tau": self.tau,
            "k_prime": self.k_prime,
            "confidence_threshold": self.confidence_threshold,
            "lambda_weights": self.lambda_weights.to_mapping(),
            "random_seeds": self.random_seeds.to_mapping(),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from JSON or TOML and validate it."""

    config_path = Path(path)
    if config_path.suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config loading requires Python 3.11 or newer")
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("experiment config must be .json or .toml")

    if not isinstance(data, Mapping):
        raise TypeError("experiment config root must be a mapping")

    experiment_data = data.get("experiment", data)
    if not isinstance(experiment_data, Mapping):
        raise TypeError("experiment config must contain a mapping")

    return ExperimentConfig.from_mapping(experiment_data)


def _required_value(data: ConfigMapping, key: str) -> object:
    if key not in data:
        raise KeyError(f"missing required config field: {key}")
    return data[key]


def _required_mapping(data: ConfigMapping, key: str) -> ConfigMapping:
    value = _required_value(data, key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _required_str(data: ConfigMapping, key: str) -> str:
    return _non_empty_str(_required_value(data, key), key)


def _optional_str(data: ConfigMapping, key: str, default: str) -> str:
    if key not in data:
        return default
    return _non_empty_str(data[key], key)


def _optional_bool(data: ConfigMapping, key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool")
    return value


def _required_float(data: ConfigMapping, key: str) -> float:
    return _finite_float(_required_value(data, key), key)


def _required_int(data: ConfigMapping, key: str) -> int:
    value = _required_value(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return result


def _seed_value(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer seed")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return tuple(_non_empty_str(item, field_name) for item in value)
