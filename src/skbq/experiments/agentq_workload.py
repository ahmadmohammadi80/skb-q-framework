"""AgentQ end-to-end experiment workload for real model evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from skbq.agentq.action_decoder import ActionDecoder
from skbq.agentq.network import GraphPolicyNetwork
from skbq.agentq.state import StateBuilder
from skbq.experiments.context import ExperimentContext
from skbq.experiments.results import write_json_artifact
from skbq.graph.operator_graph import OperatorGraph
from skbq.models import HuggingFaceGraphBuilder, HuggingFaceModelLoader
from skbq.quantization.backends.pytorch_backend import PyTorchFakeQuantBackend
from skbq.quantization.budget import BitBudget


FP_REFERENCE_BITS = 16


@dataclass(frozen=True, slots=True)
class AgentQExperimentSettings:
    """Runtime settings for one AgentQ experiment execution."""

    model_path: str | None = None
    dataset_name: str = "wikitext"
    dataset_split: str = "test"
    max_eval_samples: int = 32
    max_eval_tokens: int = 512
    bit_budget_total_bits: int = 1_000_000_000
    local_files_only: bool = True
    trust_remote_code: bool = False


def build_agentq_workload(settings: AgentQExperimentSettings) -> object:
    """Build an experiment workload closure for AgentQ allocation experiments."""

    def workload(context: ExperimentContext) -> Mapping[str, object]:
        return _run_agentq_workload(context, settings)

    return workload


def _run_agentq_workload(
    context: ExperimentContext,
    settings: AgentQExperimentSettings,
) -> Mapping[str, object]:
    observations: dict[str, object] = {}
    warnings: list[str] = []

    if settings.model_path is None or not str(settings.model_path).strip():
        warnings.append("model_path was not provided; experiment artifacts are not generated")
        observations["workload_warnings"] = tuple(warnings)
        return observations

    model_path = str(settings.model_path).strip()
    bit_budget = BitBudget(total_bits=settings.bit_budget_total_bits)

    try:
        loaded = _load_model_bundle(
            model_path=model_path,
            local_files_only=settings.local_files_only,
            trust_remote_code=settings.trust_remote_code,
        )
        graph = HuggingFaceGraphBuilder().build(loaded.model, loaded.config)
        write_json_artifact(
            context.output_directory / "graph_manifest.json",
            _graph_manifest(graph, loaded.architecture, model_path),
        )

        state_builder = StateBuilder(include_encoder_embeddings=True)
        graph_state = state_builder.build(graph)
        network = GraphPolicyNetwork()
        network.eval()
        policy_output = network(graph_state)
        plan = ActionDecoder().decode(
            graph_state=graph_state,
            logits=policy_output.logits,
            bit_budget=bit_budget,
            deterministic=True,
        )
        write_json_artifact(
            context.output_directory / "allocation_plan.json",
            plan.to_mapping(),
        )

        backend_result = PyTorchFakeQuantBackend().apply(
            model=loaded.model,
            graph=graph,
            plan=plan,
        )
        observations["compression_ratio"] = _compression_ratio(graph, plan)
        observations["memory_estimation_bytes"] = float(plan.total_storage_bits) / 8.0
        observations["backend_result"] = backend_result.to_mapping()

        perplexity = _evaluate_perplexity(
            model=loaded.model,
            model_path=model_path,
            dataset_name=settings.dataset_name,
            dataset_split=settings.dataset_split,
            max_eval_samples=settings.max_eval_samples,
            max_eval_tokens=settings.max_eval_tokens,
            local_files_only=settings.local_files_only,
            trust_remote_code=settings.trust_remote_code,
        )
        if perplexity is not None:
            observations["perplexity"] = perplexity
        else:
            warnings.append("perplexity was not computed; optional dataset/tokenizer dependencies missing")

    except ImportError as error:
        warnings.append(f"optional dependency missing: {error}")
    except Exception as error:  # pragma: no cover - surfaced to warnings for real runs
        warnings.append(f"agentq workload failed: {error}")

    if warnings:
        observations["workload_warnings"] = tuple(warnings)
    return observations


def _load_model_bundle(
    model_path: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> object:
    loader = HuggingFaceModelLoader(
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return loader.load(model_path)


def _graph_manifest(
    graph: OperatorGraph,
    architecture: str,
    model_path: str,
) -> dict[str, object]:
    return {
        "architecture": graph.architecture,
        "model_path": model_path,
        "detected_architecture": architecture,
        "node_count": len(graph.nodes),
        "operator_ids": list(graph.node_ids),
        "parameter_counts": {
            node.operator_id: node.parameter_count for node in graph.nodes
        },
    }


def _compression_ratio(graph: OperatorGraph, plan: object) -> float:
    fp_bits = sum(node.parameter_count * FP_REFERENCE_BITS for node in graph.nodes)
    allocated_bits = float(plan.total_storage_bits)
    if allocated_bits <= 0.0:
        raise ValueError("allocated bits must be positive for compression ratio")
    return fp_bits / allocated_bits


def _evaluate_perplexity(
    model: object,
    model_path: str,
    dataset_name: str,
    dataset_split: str,
    max_eval_samples: int,
    max_eval_tokens: int,
    local_files_only: bool,
    trust_remote_code: bool,
) -> float | None:
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        return None

    if not isinstance(model, torch.nn.Module):
        return None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_id, dataset_config = _resolve_dataset(dataset_name)
    dataset = load_dataset(dataset_id, dataset_config, split=dataset_split)
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for index, row in enumerate(dataset):
            if index >= max_eval_samples:
                break
            text = str(row[text_column]).strip()
            if not text:
                continue
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_eval_tokens,
            )
            input_ids = tokens["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            token_count = int(input_ids.shape[1] - 1)
            total_loss += float(outputs.loss) * token_count
            total_tokens += token_count

    if total_tokens == 0:
        return None
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


def _resolve_dataset(dataset_name: str) -> tuple[str, str | None]:
    normalized = dataset_name.strip().casefold()
    if normalized in {"wikitext", "wikitext-2", "wikitext2"}:
        return "wikitext", "wikitext-2-raw-v1"
    if normalized in {"c4", "c4-subset"}:
        return "c4", "en"
    return dataset_name, None
