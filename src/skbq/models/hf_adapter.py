"""Hugging Face module adapter for deterministic graph extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from skbq.graph.operator_graph import Shape, TensorShapeMetadata
from skbq.models.loader import HuggingFaceModelLoader, LoadedHuggingFaceModel


@dataclass(frozen=True, slots=True)
class HFModuleRecord:
    """Framework-neutral record for one Hugging Face module."""

    module_path: str
    module: object
    class_name: str
    parent_path: str | None
    child_paths: tuple[str, ...]
    parameter_count: int
    tensor_shapes: TensorShapeMetadata = field(default_factory=dict)
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_paths", tuple(self.child_paths))
        object.__setattr__(self, "tensor_shapes", MappingProxyType(dict(self.tensor_shapes)))


@dataclass(frozen=True, slots=True)
class HuggingFaceGraphExtractor:
    """Graph extractor that loads or accepts Hugging Face transformer models."""

    loader: HuggingFaceModelLoader = field(default_factory=HuggingFaceModelLoader)
    builder: object | None = None

    def extract(self, model_spec: object):
        """Return an OperatorGraph from a model name, loaded model, or loaded bundle."""

        builder = self._builder()
        if isinstance(model_spec, str):
            loaded = self.loader.load(model_spec)
            return builder.build(loaded.model, loaded.config)
        if isinstance(model_spec, LoadedHuggingFaceModel):
            return builder.build(model_spec.model, model_spec.config)
        if hasattr(model_spec, "model") and hasattr(model_spec, "config"):
            return builder.build(getattr(model_spec, "model"), getattr(model_spec, "config"))
        return builder.build(model_spec, getattr(model_spec, "config", None))

    def _builder(self):
        if self.builder is not None:
            return self.builder
        from skbq.models.graph_builder import HuggingFaceGraphBuilder

        return HuggingFaceGraphBuilder()


def iter_hf_module_records(model: object) -> tuple[HFModuleRecord, ...]:
    """Walk every Hugging Face module deterministically."""

    if not hasattr(model, "named_modules"):
        raise TypeError("Hugging Face model object must expose named_modules()")

    module_items = tuple(model.named_modules())
    module_by_path = {path: module for path, module in module_items}
    child_paths_by_parent = {path: _child_paths(path, module, module_by_path) for path, module in module_items}

    return tuple(
        HFModuleRecord(
            module_path=path,
            module=module,
            class_name=module.__class__.__name__,
            parent_path=_parent_path(path),
            child_paths=child_paths_by_parent[path],
            parameter_count=count_direct_parameters(module),
            tensor_shapes=tensor_shape_metadata(module),
            depth=_module_depth(path),
        )
        for path, module in module_items
    )


def count_direct_parameters(module: object) -> int:
    """Return direct parameter count without recursively double-counting children."""

    parameters = _direct_named_parameters(module)
    return sum(_numel(parameter) for _, parameter in parameters)


def tensor_shape_metadata(module: object) -> TensorShapeMetadata:
    """Return direct parameter shape metadata for a module."""

    shapes: dict[str, Shape] = {}
    for name, parameter in _direct_named_parameters(module):
        shape = _shape(parameter)
        if shape:
            shapes[f"parameter:{name}"] = shape
    return shapes


def _direct_named_parameters(module: object) -> tuple[tuple[str, object], ...]:
    if hasattr(module, "named_parameters"):
        try:
            return tuple(module.named_parameters(recurse=False))
        except TypeError:
            return tuple(module.named_parameters())
    if hasattr(module, "parameters"):
        try:
            return tuple((str(index), parameter) for index, parameter in enumerate(module.parameters(recurse=False)))
        except TypeError:
            return tuple((str(index), parameter) for index, parameter in enumerate(module.parameters()))
    return ()


def _child_paths(path: str, module: object, module_by_path: Mapping[str, object]) -> tuple[str, ...]:
    if not hasattr(module, "named_children"):
        return ()
    children = []
    for child_name, _child in module.named_children():
        child_path = f"{path}.{child_name}" if path else child_name
        if child_path not in module_by_path:
            raise ValueError(f"child module {child_path!r} was not present in named_modules()")
        children.append(child_path)
    return tuple(children)


def _parent_path(path: str) -> str | None:
    if not path:
        return None
    if "." not in path:
        return ""
    return path.rsplit(".", 1)[0]


def _module_depth(path: str) -> int:
    if not path:
        return 0
    return len(path.split("."))


def _numel(parameter: object) -> int:
    if hasattr(parameter, "numel"):
        return int(parameter.numel())
    shape = _shape(parameter)
    if not shape:
        return 0
    total = 1
    for dimension in shape:
        if isinstance(dimension, int):
            total *= dimension
    return total


def _shape(parameter: object) -> Shape:
    shape = getattr(parameter, "shape", None)
    if shape is None and hasattr(parameter, "size"):
        shape = parameter.size()
    if shape is None:
        return ()
    return tuple(int(dimension) for dimension in shape)
