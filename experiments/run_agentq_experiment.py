"""Run a local AgentQ allocation and evaluation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skbq.config import ExperimentConfig  # noqa: E402
from skbq.experiments.agentq_workload import AgentQExperimentSettings, build_agentq_workload  # noqa: E402
from skbq.experiments.metrics import EvaluationPipeline, MetricRegistry  # noqa: E402
from skbq.experiments.runner import ExperimentRunner  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local AgentQ experiment.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face model directory or identifier (local_files_only=True).",
    )
    parser.add_argument(
        "--experiment-id",
        default="agentq-local-run",
        help="Experiment identifier used for results/<run_id>/.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Evaluation dataset name (wikitext or c4).",
    )
    parser.add_argument(
        "--dataset-split",
        default="test",
        help="Dataset split used for perplexity evaluation.",
    )
    parser.add_argument(
        "--bit-budget",
        type=int,
        default=1_000_000_000,
        help="Total bit budget for AgentQ allocation.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=32,
        help="Maximum dataset rows evaluated for perplexity.",
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=512,
        help="Maximum tokens per evaluated sample.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow remote Hugging Face downloads (default is local_files_only).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow trust_remote_code when loading models/tokenizers.",
    )
    return parser


def _default_experiment_config(experiment_id: str, bit_budget: int) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "vocabulary": {"registry": "default", "operators": []},
            "backbone": {
                "encoder": "structural_feature_phi",
                "policy": "learned_agentq_pi_theta",
                "frozen": True,
            },
            "budget": {"total": bit_budget, "unit": "bits"},
            "tau": 1.0,
            "k_prime": 3,
            "confidence_threshold": 0.7,
            "lambda_weights": {
                "semantic": 1.0,
                "structural": 1.0,
                "functional": 1.0,
            },
            "random_seeds": {"python": 0, "agentq": 0},
        }
    )


def main() -> int:
    args = _build_parser().parse_args()
    settings = AgentQExperimentSettings(
        model_path=args.model_path,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        max_eval_samples=args.max_eval_samples,
        max_eval_tokens=args.max_eval_tokens,
        bit_budget_total_bits=args.bit_budget,
        local_files_only=not args.allow_remote,
        trust_remote_code=args.trust_remote_code,
    )
    runner = ExperimentRunner(
        results_root=Path(args.results_root),
        repo_path=ROOT,
        evaluation_pipeline=EvaluationPipeline(
            registry=MetricRegistry.with_agentq_experiment_metrics(),
        ),
    )
    result = runner.run(
        _default_experiment_config(args.experiment_id, args.bit_budget),
        workload=build_agentq_workload(settings),
    )
    print(f"Run completed: {result.context.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
