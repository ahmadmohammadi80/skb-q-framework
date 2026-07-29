"""Tests for SKB-Q reproducibility foundation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import skbq  # noqa: E402
from skbq.config import (  # noqa: E402
    CURRENT_SCHEMA_VERSION,
    ExperimentConfig,
    SeedRegistry,
    capture_experiment_metadata,
    deterministic_config_hash,
    load_experiment_config,
)
from skbq.config.metadata import (  # noqa: E402
    current_git_branch,
    current_git_commit_hash,
    is_git_dirty,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only relevant on Python 3.10
    tomllib = None


def sample_config_mapping() -> dict[str, object]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "experiment_id": "unit-test-experiment",
        "vocabulary": {
            "registry": "default",
            "operators": ["Attention", "GQA"],
        },
        "backbone": {
            "encoder": "frozen-placeholder",
            "policy": "frozen-placeholder",
            "frozen": True,
        },
        "budget": {
            "total": 8,
            "unit": "bits",
        },
        "tau": 1.0,
        "k_prime": 3,
        "confidence_threshold": 0.7,
        "lambda_weights": {
            "semantic": 1.0,
            "structural": 2.0,
            "functional": 1.0,
        },
        "random_seeds": {
            "python": 0,
            "baseline": 42,
        },
    }


class PackageMetadataTests(unittest.TestCase):
    def test_package_has_version_and_pyproject_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = root / "pyproject.toml"

        self.assertEqual(skbq.__version__, "0.1.0")
        self.assertTrue(pyproject.exists())

        if tomllib is not None:
            parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            self.assertEqual(parsed["project"]["name"], "skb-q-framework")


class ExperimentConfigTests(unittest.TestCase):
    def test_experiment_config_validates_required_reproducibility_fields(self) -> None:
        config = ExperimentConfig.from_mapping(sample_config_mapping())

        self.assertEqual(config.vocabulary.registry, "default")
        self.assertEqual(config.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(config.experiment_id, "unit-test-experiment")
        self.assertEqual(config.backbone.encoder, "frozen-placeholder")
        self.assertEqual(config.budget.total, 8.0)
        self.assertEqual(config.k_prime, 3)
        self.assertEqual(config.lambda_weights.normalized(), (0.25, 0.5, 0.25))
        self.assertEqual(config.random_seeds.values["baseline"], 42)

    def test_experiment_config_rejects_invalid_tau(self) -> None:
        data = sample_config_mapping()
        data["tau"] = 0.0

        with self.assertRaises(ValueError):
            ExperimentConfig.from_mapping(data)

    def test_experiment_config_keeps_backward_compatible_defaults(self) -> None:
        data = sample_config_mapping()
        del data["schema_version"]
        del data["experiment_id"]

        config = ExperimentConfig.from_mapping(data)

        self.assertEqual(config.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(config.experiment_id, "default")

    def test_experiment_config_rejects_unknown_top_level_fields(self) -> None:
        data = sample_config_mapping()
        data["unexpected"] = True

        with self.assertRaises(KeyError):
            ExperimentConfig.from_mapping(data)

    def test_config_hash_is_deterministic(self) -> None:
        first = sample_config_mapping()
        second = sample_config_mapping()
        second["random_seeds"] = {"baseline": 42, "python": 0}

        self.assertEqual(deterministic_config_hash(first), deterministic_config_hash(second))
        self.assertEqual(
            len(ExperimentConfig.from_mapping(first).config_hash()),
            64,
        )

    def test_load_experiment_config_from_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(sample_config_mapping()), encoding="utf-8")

            config = load_experiment_config(path)

        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.budget.unit, "bits")


class ExperimentMetadataTests(unittest.TestCase):
    def test_capture_metadata_records_environment_facts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = capture_experiment_metadata(
            repo_path=root,
            package_names=("definitely-missing-skbq-test-package",),
            experiment_id="unit-test-experiment",
        )

        self.assertEqual(metadata.experiment_id, "unit-test-experiment")
        self.assertEqual(metadata.git_commit_hash, current_git_commit_hash(root))
        self.assertEqual(metadata.git_branch, current_git_branch(root))
        self.assertEqual(metadata.git_is_dirty, is_git_dirty(root))
        self.assertIsNotNone(metadata.python_version)
        self.assertIn("definitely-missing-skbq-test-package", metadata.package_versions)
        self.assertIsNone(metadata.package_versions["definitely-missing-skbq-test-package"])
        self.assertIn("+00:00", metadata.timestamp_utc)


class SeedRegistryTests(unittest.TestCase):
    def test_seed_registry_derives_stable_component_seed(self) -> None:
        registry = SeedRegistry({"python": 0, "baseline": 42})

        first = registry.derive_seed("component", "stream")
        second = registry.derive_seed("component", "stream")

        self.assertEqual(first, second)
        self.assertNotEqual(first, registry.derive_seed("component", "other-stream"))


if __name__ == "__main__":
    unittest.main()
