"""Tests for AgentQ experiment workload and artifact generation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.config import ExperimentConfig, capture_experiment_metadata  # noqa: E402
from skbq.config.seeds import SeedRegistry  # noqa: E402
from skbq.experiments.agentq_workload import AgentQExperimentSettings, build_agentq_workload  # noqa: E402
from skbq.experiments.context import ExperimentContext  # noqa: E402
from skbq.experiments.metrics import EvaluationPipeline, MetricRegistry  # noqa: E402
from skbq.experiments.runner import ExperimentRunner  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, input_ids=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids required")
        hidden = self.linear(input_ids.float())
        if labels is None:
            return hidden
        loss = (hidden - labels.float()).pow(2).mean()
        return type("Output", (), {"loss": loss})()


class AgentQExperimentWorkloadTests(unittest.TestCase):
    def test_missing_model_path_returns_not_computed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._context(temp_dir, "missing-model")
            observations = build_agentq_workload(
                AgentQExperimentSettings(model_path=None),
            )(context)

            self.assertNotIn("perplexity", observations)
            self.assertFalse((Path(temp_dir) / "graph_manifest.json").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for AgentQ experiment tests")
    def test_workload_writes_artifacts_and_metrics_with_stub_model(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        loaded = type(
            "Loaded",
            (),
            {
                "model": TinyLinearModel(),
                "config": type("Config", (), {"model_type": "llama"})(),
                "architecture": "Llama",
            },
        )()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AgentQExperimentSettings(
                model_path="/tmp/local-model",
                bit_budget_total_bits=10_000_000,
                max_eval_samples=2,
                max_eval_tokens=8,
            )
            context = self._context(temp_dir, "stub-run")

            with patch(
                "skbq.experiments.agentq_workload._load_model_bundle",
                return_value=loaded,
            ):
                with patch(
                    "skbq.experiments.agentq_workload.HuggingFaceGraphBuilder.build",
                    return_value=graph,
                ):
                    with patch(
                        "skbq.experiments.agentq_workload._evaluate_perplexity",
                        return_value=None,
                    ):
                        observations = build_agentq_workload(settings)(context)

            self.assertIn("compression_ratio", observations)
            self.assertIn("memory_estimation_bytes", observations)
            self.assertTrue((Path(temp_dir) / "graph_manifest.json").exists())
            self.assertTrue((Path(temp_dir) / "allocation_plan.json").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for AgentQ experiment tests")
    def test_runner_writes_required_artifacts(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        loaded = type(
            "Loaded",
            (),
            {
                "model": TinyLinearModel(),
                "config": type("Config", (), {"model_type": "llama"})(),
                "architecture": "Llama",
            },
        )()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AgentQExperimentSettings(
                model_path="/tmp/local-model",
                bit_budget_total_bits=10_000_000,
            )
            runner = ExperimentRunner(
                results_root=Path(temp_dir) / "results",
                repo_path=Path(temp_dir),
                evaluation_pipeline=EvaluationPipeline(
                    registry=MetricRegistry.with_agentq_experiment_metrics(),
                ),
            )
            config = ExperimentConfig.from_mapping(
                {
                    "schema_version": "1.0",
                    "experiment_id": "agentq-runner-test",
                    "vocabulary": {"registry": "default", "operators": []},
                    "backbone": {
                        "encoder": "structural_feature_phi",
                        "policy": "learned_agentq_pi_theta",
                        "frozen": True,
                    },
                    "budget": {"total": 10_000_000, "unit": "bits"},
                    "tau": 1.0,
                    "k_prime": 3,
                    "confidence_threshold": 0.7,
                    "lambda_weights": {
                        "semantic": 1.0,
                        "structural": 1.0,
                        "functional": 1.0,
                    },
                    "random_seeds": {"python": 0},
                }
            )

            with patch(
                "skbq.experiments.agentq_workload._load_model_bundle",
                return_value=loaded,
            ):
                with patch(
                    "skbq.experiments.agentq_workload.HuggingFaceGraphBuilder.build",
                    return_value=graph,
                ):
                    with patch(
                        "skbq.experiments.agentq_workload._evaluate_perplexity",
                        return_value=12.5,
                    ):
                        result = runner.run(
                            config,
                            workload=build_agentq_workload(settings),
                        )

            run_dir = result.context.output_directory
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "graph_manifest.json").exists())
            self.assertTrue((run_dir / "allocation_plan.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            perplexity_metric = next(
                metric for metric in result.metrics if metric.name == "perplexity"
            )
            self.assertEqual(perplexity_metric.status, "computed")
            self.assertEqual(perplexity_metric.value, 12.5)

    def _context(self, temp_dir: str, experiment_id: str) -> ExperimentContext:
        config = ExperimentConfig.from_mapping(
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                "vocabulary": {"registry": "default", "operators": []},
                "backbone": {
                    "encoder": "structural_feature_phi",
                    "policy": "learned_agentq_pi_theta",
                    "frozen": True,
                },
                "budget": {"total": 1_000_000, "unit": "bits"},
                "tau": 1.0,
                "k_prime": 3,
                "confidence_threshold": 0.7,
                "lambda_weights": {
                    "semantic": 1.0,
                    "structural": 1.0,
                    "functional": 1.0,
                },
                "random_seeds": {"python": 0},
            }
        )
        return ExperimentContext(
            config=config,
            metadata=capture_experiment_metadata(repo_path=Path(temp_dir), experiment_id=experiment_id),
            seed_registry=SeedRegistry.from_random_seeds(config.random_seeds),
            output_directory=Path(temp_dir),
            experiment_id=experiment_id,
            config_hash=config.config_hash(),
        )


if __name__ == "__main__":
    unittest.main()
