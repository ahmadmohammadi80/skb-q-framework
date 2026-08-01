"""Minimal PyTorch fake-quantization backend for mixed-precision allocation plans."""

from __future__ import annotations

from dataclasses import dataclass

from skbq.graph.operator_graph import OperatorGraph
from skbq.quantization.backends.interface import QuantizationBackendResult
from skbq.quantization.operator_allocation import OperatorAllocationPlan


@dataclass(frozen=True, slots=True)
class PyTorchFakeQuantBackend:
    """Apply per-operator fake quantization using PyTorch tensors in-place."""

    backend_id: str = "pytorch_fake_quant"

    def apply(
        self,
        model: object,
        graph: OperatorGraph,
        plan: OperatorAllocationPlan,
    ) -> QuantizationBackendResult:
        """Fake-quantize module parameters mapped from the operator graph."""

        torch = _import_torch()
        nn = _import_torch_nn()

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        allocation_by_id = {
            allocation.target_id: allocation for allocation in plan.allocations
        }
        applied: list[str] = []
        skipped: list[str] = []
        parameter_count = 0

        for node in graph.nodes:
            allocation = allocation_by_id.get(node.operator_id)
            if allocation is None:
                skipped.append(node.operator_id)
                continue

            bit_width = allocation.bit_width_candidate.bit_width
            if bit_width <= 0:
                skipped.append(node.operator_id)
                continue

            module = _resolve_module(model, node.operator_id)
            if module is None:
                skipped.append(node.operator_id)
                continue

            updated = _fake_quantize_module(module, bit_width, torch)
            if updated == 0:
                skipped.append(node.operator_id)
            else:
                applied.append(node.operator_id)
                parameter_count += updated

        return QuantizationBackendResult(
            backend_id=self.backend_id,
            applied_operator_ids=tuple(sorted(applied)),
            skipped_operator_ids=tuple(sorted(skipped)),
            metadata={
                "quantization_mode": "fake_quant_per_tensor",
                "updated_parameter_tensors": parameter_count,
                "allocation_hash": plan.allocation_hash(),
            },
        )


def _fake_quantize_module(module: object, bit_width: int, torch: object) -> int:
    if not hasattr(module, "parameters"):
        return 0

    updated = 0
    for parameter in module.parameters(recurse=False):
        if parameter is None or parameter.data is None:
            continue
        parameter.data = _fake_quantize_tensor(parameter.data, bit_width, torch)
        updated += 1
    return updated


def _fake_quantize_tensor(tensor: object, bit_width: int, torch: object) -> object:
    if bit_width >= 16:
        return tensor
    qmax = (2 ** bit_width) - 1
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if float(max_val - min_val) == 0.0:
        return tensor
    scale = (max_val - min_val) / float(qmax)
    quantized = torch.round((tensor - min_val) / scale).clamp(0, qmax)
    return quantized * scale + min_val


def _resolve_module(model: object, operator_id: str) -> object | None:
    if operator_id == "__model__":
        return model

    module_path = operator_id
    if module_path.startswith("__model__."):
        module_path = module_path[len("__model__.") :]

    try:
        return model.get_submodule(module_path)
    except (AttributeError, ModuleNotFoundError, ValueError):
        return None


def _import_torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorchFakeQuantBackend requires the optional 'torch' package"
        ) from error
    return torch


def _import_torch_nn() -> object:
    torch = _import_torch()
    return torch.nn
