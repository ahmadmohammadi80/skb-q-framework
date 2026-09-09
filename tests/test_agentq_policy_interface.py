"""Tests for AgentQ policy interface contract."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.agentq import (  # noqa: E402
    AbstractAgentQPolicy,
    AgentQPolicy,
    AgentQPrediction,
    AgentQProvenance,
    StateBuilder,
    StructuralReferenceAgentQPolicy,
)
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402


class MinimalConcreteAgentQPolicy(AbstractAgentQPolicy):
    """Minimal concrete policy for interface contract tests."""

    def predict(self, graph_state) -> AgentQPrediction:
        state = self._validate_graph_state(graph_state)
        return AgentQPrediction(
            graph_identifier=state.graph_identifier,
            operator_ids=state.operator_ids,
            metadata={"policy": "minimal"},
        )

    def get_provenance(self) -> AgentQProvenance:
        return AgentQProvenance(policy_id="minimal_policy")


class AgentQPolicyInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph_state = StateBuilder().build(
            build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        )

    def test_protocol_is_satisfied_by_concrete_policy(self) -> None:
        policy = MinimalConcreteAgentQPolicy()

        self.assertIsInstance(policy, AgentQPolicy)
        prediction = policy.predict(self.graph_state)
        provenance = policy.get_provenance()

        self.assertIsInstance(prediction, AgentQPrediction)
        self.assertEqual(prediction.graph_identifier, self.graph_state.graph_identifier)
        self.assertIsInstance(provenance, AgentQProvenance)
        self.assertEqual(len(provenance.deterministic_id), 64)

    def test_structural_reference_policy_is_deterministic(self) -> None:
        policy = StructuralReferenceAgentQPolicy()

        first = policy.predict(self.graph_state)
        second = policy.predict(self.graph_state)

        self.assertEqual(first.to_mapping(), second.to_mapping())
        self.assertEqual(first.metadata["state_hash"], self.graph_state.state_hash())

    def test_no_neural_network_framework_imports(self) -> None:
        forbidden_roots = ("torch", "tensorflow", "jax", "flax", "keras")

        for module_name in sys.modules:
            if module_name.split(".")[0] in forbidden_roots:
                self.fail(f"unexpected neural network dependency loaded: {module_name}")

        policy_module_path = Path(__file__).resolve().parents[1] / "src" / "skbq" / "agentq" / "policy.py"
        policy_source = policy_module_path.read_text(encoding="utf-8")

        for root in forbidden_roots:
            self.assertNotIn(root, policy_source)


if __name__ == "__main__":
    unittest.main()
