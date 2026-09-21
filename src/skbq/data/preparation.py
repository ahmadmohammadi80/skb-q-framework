"""Configuration and OOV preparation interfaces for SKB-Q data artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


REQUIRED_TRAIN_MODEL_FAMILIES = ("llama", "qwen", "mistral")
FUTURE_HELD_OUT_MODEL_FAMILIES = ("mamba", "rwkv", "jamba")
SUPPORTED_SPLIT_PROTOCOLS = ("operator_class_holdout", "architecture_family_holdout")


@dataclass(frozen=True, slots=True)
class DataPreparationConfig:
    """Configuration for real vocabulary and OOV dataset preparation."""

    input_model_paths: Mapping[str, str]
    output_artifact_dir: Path
    split_protocol: str
    random_seed: int
    heldout_operator_types: tuple[str, ...] = ()
    heldout_architectures: tuple[str, ...] = ()
    model_versions: Mapping[str, str | None] = field(default_factory=dict)
    extraction_timestamp: str | None = None

    def __post_init__(self) -> None:
        model_paths = {_normalize_family(key): str(value) for key, value in self.input_model_paths.items()}
        missing = sorted(set(REQUIRED_TRAIN_MODEL_FAMILIES) - set(model_paths))
        if missing:
            raise ValueError(f"missing required training model path(s): {missing}")
        if self.split_protocol not in SUPPORTED_SPLIT_PROTOCOLS:
            raise ValueError(f"unsupported split protocol: {self.split_protocol}")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.split_protocol == "operator_class_holdout" and not self.heldout_operator_types:
            raise ValueError("operator_class_holdout requires heldout_operator_types")
        if self.split_protocol == "architecture_family_holdout" and not self.heldout_architectures:
            raise ValueError("architecture_family_holdout requires heldout_architectures")

        versions = {
            _normalize_family(key): value
            for key, value in self.model_versions.items()
        }
        object.__setattr__(self, "input_model_paths", MappingProxyType(dict(sorted(model_paths.items()))))
        object.__setattr__(self, "output_artifact_dir", Path(self.output_artifact_dir))
        object.__setattr__(self, "heldout_operator_types", tuple(sorted(set(self.heldout_operator_types))))
        object.__setattr__(self, "heldout_architectures", tuple(sorted(set(self.heldout_architectures))))
        object.__setattr__(self, "model_versions", MappingProxyType(dict(sorted(versions.items()))))

    def model_path(self, family: str) -> str:
        """Return model path for a normalized family."""

        return self.input_model_paths[_normalize_family(family)]

    def model_version(self, family: str) -> str | None:
        """Return optional model version for a normalized family."""

        return self.model_versions.get(_normalize_family(family))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable configuration."""

        return {
            "input_model_paths": dict(self.input_model_paths),
            "output_artifact_dir": str(self.output_artifact_dir),
            "split_protocol": self.split_protocol,
            "random_seed": self.random_seed,
            "heldout_operator_types": list(self.heldout_operator_types),
            "heldout_architectures": list(self.heldout_architectures),
            "model_versions": dict(self.model_versions),
            "extraction_timestamp": self.extraction_timestamp,
        }


@dataclass(frozen=True, slots=True)
class FutureHeldOutModelRequest:
    """Interface record for future held-out OOV model families."""

    model_family: str
    model_path: str | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        family = _normalize_family(self.model_family)
        if family not in FUTURE_HELD_OUT_MODEL_FAMILIES:
            raise ValueError(
                f"future held-out model must be one of {FUTURE_HELD_OUT_MODEL_FAMILIES}"
            )
        object.__setattr__(self, "model_family", family)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable request."""

        return {
            "model_family": self.model_family,
            "model_path": self.model_path,
            "model_version": self.model_version,
            "status": "interface_only",
        }


@dataclass(frozen=True, slots=True)
class OOVPreparationInterface:
    """Interface placeholder for future Mamba/RWKV/Jamba held-out data preparation."""

    supported_future_families: tuple[str, ...] = FUTURE_HELD_OUT_MODEL_FAMILIES

    def prepare(self, request: FutureHeldOutModelRequest) -> None:
        """Declare but do not execute future OOV held-out model preparation."""

        raise NotImplementedError(
            f"{request.model_family} OOV model preparation is reserved for a later milestone"
        )


def _normalize_family(value: str) -> str:
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("model family cannot be empty")
    return normalized
