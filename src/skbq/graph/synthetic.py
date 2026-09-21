"""Deterministic synthetic operator graphs for SKB-Q unit testing.

These builders create schematic graphs for exercising extraction logic. They do
not represent benchmark results, trained models, or framework-specific modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from skbq.graph.operator_graph import OperatorGraph, OperatorNode, Shape, TensorShapeMetadata


@dataclass(frozen=True, slots=True)
class SyntheticArchitectureSpec:
    """Specification for deterministic synthetic graph generation."""

    architecture: str
    num_layers: int = 2
    hidden_size: int = 128
    intermediate_size: int | None = None
    num_experts: int = 4

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")

    @property
    def ffn_size(self) -> int:
        """Return deterministic feed-forward width for synthetic blocks."""

        return self.intermediate_size or self.hidden_size * 4


@dataclass(frozen=True, slots=True)
class SyntheticGraphExtractor:
    """Graph extractor for deterministic synthetic architecture specifications."""

    def extract(self, model_spec: object) -> OperatorGraph:
        """Extract a synthetic graph from a spec, mapping, or architecture name."""

        spec = _coerce_spec(model_spec)
        architecture = _normalize_architecture(spec.architecture)
        if architecture == "transformer":
            return build_transformer_graph(spec)
        if architecture == "moe_transformer":
            return build_moe_transformer_graph(spec)
        if architecture == "mamba":
            return build_mamba_graph(spec)
        if architecture == "rwkv":
            return build_rwkv_graph(spec)
        raise ValueError(f"unsupported synthetic architecture: {spec.architecture}")


def build_transformer_graph(
    spec: SyntheticArchitectureSpec | None = None,
) -> OperatorGraph:
    """Build a deterministic synthetic Transformer operator graph."""

    spec = spec or SyntheticArchitectureSpec("Transformer")
    nodes = [_embedding_spec(spec)]
    previous_id = "embedding"

    for layer_index in range(spec.num_layers):
        prefix = f"layer{layer_index}"
        attention_norm = f"{prefix}.attention_norm"
        attention = f"{prefix}.attention"
        ffn_norm = f"{prefix}.ffn_norm"
        swiglu = f"{prefix}.swiglu"
        nodes.extend(
            [
                _node(attention_norm, "RMSNorm", (previous_id,), spec.hidden_size, spec, layer_index * 4 + 1),
                _node(attention, "Attention", (attention_norm,), 4 * spec.hidden_size**2, spec, layer_index * 4 + 2),
                _node(ffn_norm, "RMSNorm", (attention,), spec.hidden_size, spec, layer_index * 4 + 3),
                _node(
                    swiglu,
                    "SwiGLU",
                    (ffn_norm,),
                    3 * spec.hidden_size * spec.ffn_size,
                    spec,
                    layer_index * 4 + 4,
                    has_nonlinearity=True,
                ),
            ]
        )
        previous_id = swiglu

    nodes.append(_output_spec(previous_id, spec, spec.num_layers * 4 + 1))
    return _graph_from_specs("Transformer", nodes)


def build_moe_transformer_graph(
    spec: SyntheticArchitectureSpec | None = None,
) -> OperatorGraph:
    """Build a deterministic synthetic MoE Transformer operator graph."""

    spec = spec or SyntheticArchitectureSpec("MoE Transformer")
    nodes = [_embedding_spec(spec)]
    previous_id = "embedding"

    for layer_index in range(spec.num_layers):
        prefix = f"layer{layer_index}"
        attention_norm = f"{prefix}.attention_norm"
        attention = f"{prefix}.attention"
        router_norm = f"{prefix}.router_norm"
        router = f"{prefix}.moe_router"
        expert_ids = tuple(f"{prefix}.expert{expert_index}" for expert_index in range(spec.num_experts))
        merge = f"{prefix}.expert_merge"

        nodes.extend(
            [
                _node(attention_norm, "RMSNorm", (previous_id,), spec.hidden_size, spec, layer_index * 6 + 1),
                _node(attention, "GQA", (attention_norm,), 3 * spec.hidden_size**2, spec, layer_index * 6 + 2),
                _node(router_norm, "RMSNorm", (attention,), spec.hidden_size, spec, layer_index * 6 + 3),
                _node(
                    router,
                    "MoE Router",
                    (router_norm,),
                    spec.hidden_size * spec.num_experts,
                    spec,
                    layer_index * 6 + 4,
                    branch_group=f"{prefix}.experts",
                    branch_count=spec.num_experts,
                    tensor_shapes={"input": _activation_shape(spec), "logits": ("seq", spec.num_experts)},
                ),
            ]
        )
        nodes.extend(
            _node(
                expert_id,
                "Expert FFN",
                (router,),
                2 * spec.hidden_size * spec.ffn_size,
                spec,
                layer_index * 6 + 5,
                branch_group=f"{prefix}.experts",
                branch_index=expert_index,
                branch_count=spec.num_experts,
                has_nonlinearity=True,
            )
            for expert_index, expert_id in enumerate(expert_ids)
        )
        nodes.append(
            _node(
                merge,
                "Expert Merge",
                expert_ids,
                1,
                spec,
                layer_index * 6 + 6,
                branch_group=f"{prefix}.experts",
                branch_count=spec.num_experts,
                is_branch_merge=True,
            )
        )
        previous_id = merge

    nodes.append(_output_spec(previous_id, spec, spec.num_layers * 6 + 1))
    return _graph_from_specs("MoE Transformer", nodes)


def build_mamba_graph(
    spec: SyntheticArchitectureSpec | None = None,
) -> OperatorGraph:
    """Build a deterministic synthetic Mamba-style operator graph."""

    spec = spec or SyntheticArchitectureSpec("Mamba-style")
    nodes = [_embedding_spec(spec)]
    previous_id = "embedding"

    for layer_index in range(spec.num_layers):
        prefix = f"layer{layer_index}"
        norm = f"{prefix}.norm"
        scan = f"{prefix}.selective_scan"
        gate = f"{prefix}.gate"
        projection = f"{prefix}.projection"
        nodes.extend(
            [
                _node(norm, "RMSNorm", (previous_id,), spec.hidden_size, spec, layer_index * 4 + 1),
                _node(scan, "Selective Scan", (norm,), 3 * spec.hidden_size**2, spec, layer_index * 4 + 2),
                _node(
                    gate,
                    "Mamba Gate",
                    (scan,),
                    spec.hidden_size**2,
                    spec,
                    layer_index * 4 + 3,
                    has_nonlinearity=True,
                ),
                _node(projection, "Linear Projection", (gate,), spec.hidden_size**2, spec, layer_index * 4 + 4),
            ]
        )
        previous_id = projection

    nodes.append(_output_spec(previous_id, spec, spec.num_layers * 4 + 1))
    return _graph_from_specs("Mamba-style", nodes)


def build_rwkv_graph(
    spec: SyntheticArchitectureSpec | None = None,
) -> OperatorGraph:
    """Build a deterministic synthetic RWKV-style operator graph."""

    spec = spec or SyntheticArchitectureSpec("RWKV-style")
    nodes = [_embedding_spec(spec)]
    previous_id = "embedding"

    for layer_index in range(spec.num_layers):
        prefix = f"layer{layer_index}"
        time_mix = f"{prefix}.time_mix"
        channel_mix = f"{prefix}.channel_mix"
        nodes.extend(
            [
                _node(time_mix, "RWKV TimeMix", (previous_id,), 3 * spec.hidden_size**2, spec, layer_index * 2 + 1),
                _node(
                    channel_mix,
                    "RWKV ChannelMix",
                    (time_mix,),
                    3 * spec.hidden_size * spec.ffn_size,
                    spec,
                    layer_index * 2 + 2,
                    has_nonlinearity=True,
                ),
            ]
        )
        previous_id = channel_mix

    nodes.append(_output_spec(previous_id, spec, spec.num_layers * 2 + 1))
    return _graph_from_specs("RWKV-style", nodes)


@dataclass(frozen=True, slots=True)
class _NodeSpec:
    operator_id: str
    operator_type: str
    parent_ids: tuple[str, ...]
    parameter_count: int
    tensor_shapes: TensorShapeMetadata
    depth_position: int
    branch_group: str | None = None
    branch_index: int | None = None
    branch_count: int = 1
    is_branch_merge: bool = False
    has_nonlinearity: bool = False


def _graph_from_specs(architecture: str, specs: Sequence[_NodeSpec]) -> OperatorGraph:
    child_ids: dict[str, list[str]] = {spec.operator_id: [] for spec in specs}
    for spec in specs:
        for parent_id in spec.parent_ids:
            child_ids[parent_id].append(spec.operator_id)

    return OperatorGraph(
        architecture=architecture,
        nodes=tuple(
            OperatorNode(
                operator_id=spec.operator_id,
                operator_type=spec.operator_type,
                parent_ids=spec.parent_ids,
                child_ids=tuple(child_ids[spec.operator_id]),
                parameter_count=spec.parameter_count,
                tensor_shapes=spec.tensor_shapes,
                depth_position=spec.depth_position,
                branch_group=spec.branch_group,
                branch_index=spec.branch_index,
                branch_count=spec.branch_count,
                is_branch_merge=spec.is_branch_merge,
                has_nonlinearity=spec.has_nonlinearity,
            )
            for spec in specs
        ),
    )


def _embedding_spec(spec: SyntheticArchitectureSpec) -> _NodeSpec:
    return _node(
        "embedding",
        "Token Embedding",
        (),
        spec.hidden_size**2,
        spec,
        0,
    )


def _output_spec(parent_id: str, spec: SyntheticArchitectureSpec, depth: int) -> _NodeSpec:
    return _node("output", "Output Projection", (parent_id,), spec.hidden_size**2, spec, depth)


def _node(
    operator_id: str,
    operator_type: str,
    parent_ids: Sequence[str],
    parameter_count: int,
    spec: SyntheticArchitectureSpec,
    depth_position: int,
    tensor_shapes: TensorShapeMetadata | None = None,
    branch_group: str | None = None,
    branch_index: int | None = None,
    branch_count: int = 1,
    is_branch_merge: bool = False,
    has_nonlinearity: bool = False,
) -> _NodeSpec:
    return _NodeSpec(
        operator_id=operator_id,
        operator_type=operator_type,
        parent_ids=tuple(parent_ids),
        parameter_count=parameter_count,
        tensor_shapes=tensor_shapes or {"input": _activation_shape(spec), "output": _activation_shape(spec)},
        depth_position=depth_position,
        branch_group=branch_group,
        branch_index=branch_index,
        branch_count=branch_count,
        is_branch_merge=is_branch_merge,
        has_nonlinearity=has_nonlinearity,
    )


def _activation_shape(spec: SyntheticArchitectureSpec) -> Shape:
    return ("seq", spec.hidden_size)


def _coerce_spec(model_spec: object) -> SyntheticArchitectureSpec:
    if isinstance(model_spec, SyntheticArchitectureSpec):
        return model_spec
    if isinstance(model_spec, str):
        return SyntheticArchitectureSpec(model_spec)
    if isinstance(model_spec, Mapping):
        return SyntheticArchitectureSpec(
            architecture=str(model_spec["architecture"]),
            num_layers=int(model_spec.get("num_layers", 2)),
            hidden_size=int(model_spec.get("hidden_size", 128)),
            intermediate_size=(
                None
                if model_spec.get("intermediate_size") is None
                else int(model_spec["intermediate_size"])
            ),
            num_experts=int(model_spec.get("num_experts", 4)),
        )
    if hasattr(model_spec, "architecture"):
        intermediate_size = getattr(model_spec, "intermediate_size", None)
        return SyntheticArchitectureSpec(
            architecture=str(getattr(model_spec, "architecture")),
            num_layers=int(getattr(model_spec, "num_layers", 2)),
            hidden_size=int(getattr(model_spec, "hidden_size", 128)),
            intermediate_size=None if intermediate_size is None else int(intermediate_size),
            num_experts=int(getattr(model_spec, "num_experts", 4)),
        )
    raise TypeError("synthetic graph extraction requires a spec, mapping, or architecture name")


def _normalize_architecture(name: str) -> str:
    normalized = " ".join(name.strip().casefold().replace("_", " ").split())
    aliases = {
        "transformer": "transformer",
        "moe transformer": "moe_transformer",
        "moe-transformer": "moe_transformer",
        "mamba": "mamba",
        "mamba style": "mamba",
        "mamba-style": "mamba",
        "rwkv": "rwkv",
        "rwkv style": "rwkv",
        "rwkv-style": "rwkv",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unsupported synthetic architecture: {name}")
