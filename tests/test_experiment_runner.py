"""Tests for SKB-Q experiment runner and evaluation infrastructure."""

from __future__ import annotations

from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.config import ExperimentConfig  # noqa: E402
from skbq.experiments import (  # noqa: E402
    EvaluationPipeline,
    ExperimentContext,
    ExperimentRunner,
    MetricRegistry,
    MetricResult,
    ResultWriter,
    RunDirectoryManager,
)


def sample_config(experiment_id: str = "runner-test") -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "vocabulary": {
                "registry": "default",
                "operators": ["Attention"],
            },
            "backbone": {
                "encoder": "structural_feature_phi",
                "policy": "uniform_pi_theta",
                "frozen": True,
            },
            "budget": {
                "total": 8,
                "unit": "bits",
            },
            "tau": 1.0,
            "k_prime": 2,
            "confidence_threshold": 0.7,
            "lambda_weights": {
                "semantic": 1.0,
                "structural": 1.0,
                "functional": 1.0,
            },
            "random_seeds": {
                "python": 123,
                "baseline": 456,
            },
        }
    )


class ExplicitMetric:
    """Plugin metric used to verify metric registration."""

    @property
    def name(self) -> str:
        return "explicit_metric"

    def compute(
        self,
        context: ExperimentContext,
        observations: dict[str, object],
    ) -> MetricResult:
        value = observations.get(self.name)
        if value is None:
            return MetricResult(self.name, None, "not_computed")
        return MetricResult(self.name, float(value), "computed")


class ExperimentRunnerTests(unittest.TestCase):
    def test_runner_writes_and_loads_result_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = sample_config()
            runner = ExperimentRunner(results_root=root, repo_path=Path(__file__).resolve().parents[1])

            run = runner.run(config)
            loaded = ResultWriter().load(run.context.output_directory)

            self.assertEqual(run.result.experiment_id, "runner-test")
            self.assertEqual(loaded.experiment_id, run.result.experiment_id)
            for filename in ("config.json", "metadata.json", "metrics.json", "warnings.json"):
                self.assertTrue((run.context.output_directory / filename).exists())
            self.assertEqual(loaded.config, config.to_mapping())
            self.assertEqual(loaded.metadata["config_hash"], config.config_hash())
            self.assertEqual(loaded.metadata["seed_values"], {"baseline": 456, "python": 123})
            self.assertIn("package_versions", loaded.metadata)
            self.assertGreaterEqual(loaded.metadata["runtime_seconds"], 0.0)

    def test_repeated_execution_is_deterministic_but_uses_unique_directories(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = sample_config("repeat-test")
            runner = ExperimentRunner(results_root=root, repo_path=Path(__file__).resolve().parents[1])

            first = runner.run(config)
            second = runner.run(config)

            self.assertEqual(first.context.experiment_id, "repeat-test")
            self.assertEqual(second.context.experiment_id, "repeat-test-001")
            self.assertNotEqual(first.context.output_directory, second.context.output_directory)
            self.assertEqual(first.result.config, second.result.config)
            self.assertEqual(
                first.result.metrics["metrics"]["accuracy"]["status"],
                second.result.metrics["metrics"]["accuracy"]["status"],
            )
            self.assertTrue((root / "repeat-test").exists())
            self.assertTrue((root / "repeat-test-001").exists())

    def test_metric_registration_and_workload_observations(self) -> None:
        with TemporaryDirectory() as directory:
            registry = MetricRegistry.with_defaults().register(ExplicitMetric())
            runner = ExperimentRunner(
                results_root=Path(directory),
                repo_path=Path(__file__).resolve().parents[1],
                evaluation_pipeline=EvaluationPipeline(registry),
            )

            run = runner.run(
                sample_config("metric-test"),
                workload=lambda context: {"explicit_metric": 2.5},
            )

            self.assertEqual(
                run.result.metrics["metrics"]["explicit_metric"]["value"],
                2.5,
            )
            self.assertEqual(
                run.result.metrics["metrics"]["perplexity"]["status"],
                "not_computed",
            )

    def test_seed_propagation_reseeds_python_random(self) -> None:
        with TemporaryDirectory() as directory:
            runner = ExperimentRunner(results_root=Path(directory), repo_path=Path(__file__).resolve().parents[1])
            config = sample_config("seed-test")

            runner.run(config)
            first = random.random()
            runner.run(config)
            second = random.random()

            self.assertEqual(first, second)


class RunDirectoryManagerTests(unittest.TestCase):
    def test_run_directory_manager_never_overwrites(self) -> None:
        with TemporaryDirectory() as directory:
            manager = RunDirectoryManager(Path(directory))

            first_id, first_path = manager.create_run_directory("no-overwrite")
            second_id, second_path = manager.create_run_directory("no-overwrite")

            self.assertEqual(first_id, "no-overwrite")
            self.assertEqual(second_id, "no-overwrite-001")
            self.assertNotEqual(first_path, second_path)


if __name__ == "__main__":
    unittest.main()
