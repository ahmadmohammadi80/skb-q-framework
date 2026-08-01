"""Structural feature extraction for SKB-Q operator bridge research.

The bridge represents an operator ``t`` with a six-dimensional vector ``g(t)``:

1. ``log(parameter_ratio)``
2. normalized input degree
3. normalized output degree
4. normalized depth position
5. nonlinearity indicator
6. multi-branch routing indicator

The functions in this module are deterministic and validate inputs explicitly so
experiments can reproduce feature construction without hidden defaults.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


FEATURE_NAMES: tuple[str, str, str, str, str, str] = (
    "log_parameter_ratio",
    "normalized_input_degree",
    "normalized_output_degree",
    "normalized_depth_position",
    "has_nonlinearity",
    "has_multi_branch_routing",
)

FeatureVector = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class OperatorStructuralMetadata:
    """Raw structural metadata required to compute ``g(t)`` for one operator."""

    parameter_count: float
    reference_parameter_count: float
    input_degree: float
    output_degree: float
    depth_index: float
    max_input_degree: float
    max_output_degree: float
    max_depth_index: float
    has_nonlinearity: bool = False
    has_multi_branch_routing: bool = False


def extract_operator_features(
    operator: OperatorStructuralMetadata | Mapping[str, object] | object,
) -> FeatureVector:
    """Return the six-dimensional structural feature vector ``g(t)``.

    ``operator`` may be an :class:`OperatorStructuralMetadata`, a mapping with
    matching field names, or an object exposing those names as attributes.
    """

    metadata = coerce_operator_metadata(operator)
    parameter_ratio = metadata.parameter_count / metadata.reference_parameter_count

    return (
        math.log(parameter_ratio),
        _normalize(metadata.input_degree, metadata.max_input_degree, "input_degree"),
        _normalize(metadata.output_degree, metadata.max_output_degree, "output_degree"),
        _normalize(metadata.depth_index, metadata.max_depth_index, "depth_index"),
        float(metadata.has_nonlinearity),
        float(metadata.has_multi_branch_routing),
    )


def coerce_operator_metadata(
    operator: OperatorStructuralMetadata | Mapping[str, object] | object,
) -> OperatorStructuralMetadata:
    """Convert supported operator metadata containers into a validated dataclass."""

    if isinstance(operator, OperatorStructuralMetadata):
        metadata = operator
    else:
        metadata = OperatorStructuralMetadata(
            parameter_count=_read_float(operator, "parameter_count"),
            reference_parameter_count=_read_float(operator, "reference_parameter_count"),
            input_degree=_read_float(operator, "input_degree"),
            output_degree=_read_float(operator, "output_degree"),
            depth_index=_read_float(operator, "depth_index"),
            max_input_degree=_read_float(operator, "max_input_degree"),
            max_output_degree=_read_float(operator, "max_output_degree"),
            max_depth_index=_read_float(operator, "max_depth_index"),
            has_nonlinearity=_read_bool(operator, "has_nonlinearity"),
            has_multi_branch_routing=_read_bool(operator, "has_multi_branch_routing"),
        )

    _validate_metadata(metadata)
    return metadata


def as_feature_vector(values: Sequence[float]) -> FeatureVector:
    """Validate and coerce a sequence into the canonical SKB-Q feature vector."""

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} structural features, received {len(values)}"
        )

    vector = tuple(_coerce_finite_float(value, FEATURE_NAMES[index]) for index, value in enumerate(values))
    return vector  # type: ignore[return-value]


def parameter_ratio(metadata: OperatorStructuralMetadata) -> float:
    """Return the positive parameter ratio used by ``log(parameter_ratio)``."""

    validated = coerce_operator_metadata(metadata)
    return validated.parameter_count / validated.reference_parameter_count


def _validate_metadata(metadata: OperatorStructuralMetadata) -> None:
    for field_name in (
        "parameter_count",
        "reference_parameter_count",
        "input_degree",
        "output_degree",
        "depth_index",
        "max_input_degree",
        "max_output_degree",
        "max_depth_index",
    ):
        _coerce_finite_float(getattr(metadata, field_name), field_name)

    if metadata.parameter_count <= 0.0:
        raise ValueError("parameter_count must be positive for log(parameter_ratio)")
    if metadata.reference_parameter_count <= 0.0:
        raise ValueError("reference_parameter_count must be positive")

    _validate_non_negative(metadata.input_degree, "input_degree")
    _validate_non_negative(metadata.output_degree, "output_degree")
    _validate_non_negative(metadata.depth_index, "depth_index")
    _validate_non_negative(metadata.max_input_degree, "max_input_degree")
    _validate_non_negative(metadata.max_output_degree, "max_output_degree")
    _validate_non_negative(metadata.max_depth_index, "max_depth_index")


def _normalize(value: float, maximum: float, field_name: str) -> float:
    if maximum == 0.0:
        if value == 0.0:
            return 0.0
        raise ValueError(f"{field_name} cannot be positive when its maximum is zero")
    if value > maximum:
        raise ValueError(f"{field_name} cannot exceed its declared maximum")
    return value / maximum


def _read_float(source: Mapping[str, object] | object, field_name: str) -> float:
    value = _read_value(source, field_name)
    return _coerce_finite_float(value, field_name)


def _read_bool(source: Mapping[str, object] | object, field_name: str) -> bool:
    value = _read_value(source, field_name)
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a bool")


def _read_value(source: Mapping[str, object] | object, field_name: str) -> object:
    if isinstance(source, Mapping):
        if field_name not in source:
            raise KeyError(f"missing required structural metadata field: {field_name}")
        return source[field_name]

    if hasattr(source, field_name):
        return getattr(source, field_name)

    raise AttributeError(f"missing required structural metadata attribute: {field_name}")


def _coerce_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a finite number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _validate_non_negative(value: float, field_name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
