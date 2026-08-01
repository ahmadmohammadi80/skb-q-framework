"""Tests for AgentQ ActionDecoder and LearnedAgentQPolicy inference pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

from skbq.agentq.action_decoder import ActionDecoder  # noqa: E402
from skbq.agentq.network import GraphPolicyNetwork  # noqa: E402
from skbq.agentq.policy import AgentQPrediction, LearnedAgentQPolicy  # noqa: E402
from skbq.agentq.state import StateBuilder  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402
from skbq.quantization import BitBudget, BitWidthCandidate, PolicyActionSpace  # noqa: E402


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for ActionDecoder tests")
class ActionDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = StateBuilder(include_encoder_embeddings=True)
        self.graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        self.state = self.builder.build(self.graph)
        self.bit_budget = BitBudget(total_bits=10_000_000)

    def _logits(self, network: GraphPolicyNetwork) -> torch.Tensor:
        return network(self.state).logits

    def test_logits_decode_to_operator_allocation_plan(self) -> None:
        network = GraphPolicyNetwork()
        decoder = ActionDecoder()
        plan = decoder.decode(self.state, self._logits(network), self.bit_budget)

        self.assertEqual(len(plan.allocations), self.state.node_count)
        self.assertLessEqual(plan.total_storage_bits, self.bit_budget.total_bits)
        self.assertEqual(len(plan.allocation_hash()), 64)

    def test_deterministic_argmax_decoding(self) -> None:
        network = GraphPolicyNetwork()
        decoder = ActionDecoder()
        logits = self._logits(network)

        first = decoder.decode_to_prediction(self.state, logits, self.bit_budget, deterministic=True)
        second = decoder.decode_to_prediction(self.state, logits, self.bit_budget, deterministic=True)

        self.assertEqual(first.to_mapping(), second.to_mapping())

    def test_temperature_mode_changes_policy_mass(self) -> None:
        network = GraphPolicyNetwork()
        decoder = ActionDecoder()
        logits = self._logits(network)

        cold = decoder.decode_to_prediction(
            self.state,
            logits,
            self.bit_budget,
            temperature=0.1,
            deterministic=True,
        )
        hot = decoder.decode_to_prediction(
            self.state,
            logits,
            self.bit_budget,
            temperature=5.0,
            deterministic=True,
        )

        self.assertIn("bit_widths", cold.metadata)
        self.assertIn("bit_widths", hot.metadata)
        self.assertNotEqual(cold.metadata["temperature"], hot.metadata["temperature"])

    def test_invalid_action_masking_restricts_bit_widths(self) -> None:
        narrow = BitWidthCandidate(bit_width=2, candidate_id="bw2")
        decoder = ActionDecoder(
            action_space=PolicyActionSpace(
                target_type_candidates={"graph_operator": (narrow,)},
            ),
        )
        logits = torch.zeros((self.state.node_count, 4))
        logits[:, 3] = 100.0

        prediction = decoder.decode_to_prediction(self.state, logits, self.bit_budget)
        bit_widths = prediction.metadata["bit_widths"]

        self.assertTrue(all(width == 2 for width in bit_widths.values()))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for LearnedAgentQPolicy tests")
class LearnedAgentQPolicyTests(unittest.TestCase):
    def test_works_with_synthetic_graph(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        policy = LearnedAgentQPolicy(
            network=GraphPolicyNetwork(),
            bit_budget=BitBudget(total_bits=10_000_000),
        )

        prediction = policy.predict_graph(graph)

        self.assertIsInstance(prediction, AgentQPrediction)
        self.assertEqual(len(prediction.operator_ids), len(graph.nodes))
        self.assertIn("allocation_hash", prediction.metadata)

    def test_deterministic_output_with_fixed_seed(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))

        torch.manual_seed(11)
        first_policy = LearnedAgentQPolicy(
            network=GraphPolicyNetwork(),
            bit_budget=BitBudget(total_bits=10_000_000),
        )
        first = first_policy.predict_graph(graph)

        torch.manual_seed(11)
        second_policy = LearnedAgentQPolicy(
            network=GraphPolicyNetwork(),
            bit_budget=BitBudget(total_bits=10_000_000),
        )
        second = second_policy.predict_graph(graph)

        self.assertEqual(first.to_mapping(), second.to_mapping())

    def test_get_provenance_is_stable(self) -> None:
        policy = LearnedAgentQPolicy(
            network=GraphPolicyNetwork(),
            bit_budget=BitBudget(total_bits=1_000),
        )

        provenance = policy.get_provenance()

        self.assertEqual(provenance.policy_id, "learned_agentq_pi_theta")
        self.assertEqual(len(provenance.deterministic_id), 64)


if __name__ == "__main__":
    unittest.main()
