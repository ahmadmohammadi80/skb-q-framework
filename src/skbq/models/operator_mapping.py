"""Architecture and operator mapping for Hugging Face transformer models."""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """Raised when optional Hugging Face dependencies are unavailable."""


class UnsupportedArchitectureError(ValueError):
    """Raised when a model architecture is outside the current integration scope."""


SUPPORTED_ARCHITECTURE_ALIASES: dict[str, str] = {
    "llama": "Llama",
    "qwen": "Qwen",
    "qwen2": "Qwen",
    "mistral": "Mistral",
}

UNSUPPORTED_ARCHITECTURE_MARKERS = (
    "mamba",
    "rwkv",
    "jamba",
    "mixtral",
    "moe",
)


def detect_supported_architecture(config_or_model: object) -> str:
    """Return supported architecture name or raise a clear exception."""

    architecture_key = _architecture_key(config_or_model)
    lowered = architecture_key.casefold()
    if any(marker in lowered for marker in UNSUPPORTED_ARCHITECTURE_MARKERS):
        raise UnsupportedArchitectureError(
            f"architecture {architecture_key!r} is not supported in this milestone"
        )

    if lowered in SUPPORTED_ARCHITECTURE_ALIASES:
        return SUPPORTED_ARCHITECTURE_ALIASES[lowered]

    raise UnsupportedArchitectureError(
        "supported Hugging Face architectures are Llama, Qwen, and Mistral; "
        f"received {architecture_key!r}"
    )


def map_operator_type(module_path: str, module: object) -> str:
    """Map a Hugging Face module to an SKB-Q operator type without skipping it."""

    class_name = module.__class__.__name__
    lowered_path = module_path.casefold()
    lowered_class = class_name.casefold()

    if "self_attn" in lowered_path or "attention" in lowered_class:
        return "Attention"
    if "mlp" in lowered_path or "feedforward" in lowered_class:
        return "SwiGLU"
    if "norm" in lowered_path or "rmsnorm" in lowered_class or "layernorm" in lowered_class:
        return "RMSNorm"
    if "embed" in lowered_path or "embedding" in lowered_class:
        return "Token Embedding"
    if "lm_head" in lowered_path or "output" in lowered_path:
        return "Output Projection"
    if "linear" in lowered_class or lowered_class == "linear":
        return "Linear"

    # Unknown classes are represented explicitly instead of being skipped.
    return f"HF::{class_name}"


def _architecture_key(config_or_model: object) -> str:
    config = getattr(config_or_model, "config", config_or_model)
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type.strip():
        return model_type.strip()

    architectures = getattr(config, "architectures", None)
    if architectures:
        first_architecture = str(tuple(architectures)[0])
        return _class_to_architecture(first_architecture)

    return _class_to_architecture(config_or_model.__class__.__name__)


def _class_to_architecture(class_name: str) -> str:
    lowered = class_name.casefold()
    for alias in SUPPORTED_ARCHITECTURE_ALIASES:
        if alias in lowered:
            return alias
    for marker in UNSUPPORTED_ARCHITECTURE_MARKERS:
        if marker in lowered:
            return marker
    return class_name
