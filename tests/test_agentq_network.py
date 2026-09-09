"""Tests for AgentQ GraphPolicyNetwork inference scaffold."""

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

from skbq.agentq.network import GraphPolicyNetwork, GraphPolicyOutput  # noqa: E402
from skbq.agentq.state import StateBuilder  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402


def tearDownModule() -> None:
    """Remove torch from ``sys.modules`` so other AgentQ tests stay isolated."""

    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            del sys.modules[module_name]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for GraphPolicyNetwork tests")
class GraphPolicyNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = StateBuilder(include_encoder_embeddings=True)

    def _state(self, num_layers: int) -> object:
        graph = build_transformer_graph(
            SyntheticArchitectureSpec("transformer", num_layers=num_layers)
        )
        return self.builder.build(graph)

    def test_forward_pass_succeeds(self) -> None:
        network = GraphPolicyNetwork()
        output = network(self._state(num_layers=1))

        self.assertIsInstance(output, GraphPolicyOutput)
        self.assertEqual(output.logits.ndim, 2)
        self.assertFalse(torch.isnan(output.logits).any())

    def test_variable_graph_sizes_work(self) -> None:
        network = GraphPolicyNetwork()
        small = network(self._state(num_layers=1))
        large = network(self._state(num_layers=2))

        self.assertEqual(small.logits.shape[0], small.operator_ids.__len__())
        self.assertEqual(large.logits.shape[0], large.operator_ids.__len__())
        self.assertNotEqual(small.logits.shape[0], large.logits.shape[0])

    def test_fixed_random_seed_produces_deterministic_output(self) -> None:
        state = self._state(num_layers=1)

        torch.manual_seed(7)
        first_network = GraphPolicyNetwork()
        first_output = first_network(state)

        torch.manual_seed(7)
        second_network = GraphPolicyNetwork()
        second_output = second_network(state)

        self.assertTrue(torch.equal(first_output.logits, second_output.logits))

    def test_output_tensor_shape_is_correct(self) -> None:
        num_actions = 5
        network = GraphPolicyNetwork(num_actions=num_actions)
        state = self._state(num_layers=1)
        output = network(state)

        self.assertEqual(output.logits.shape, (len(state.operator_ids), num_actions))
        self.assertEqual(output.operator_ids, state.operator_ids)


if __name__ == "__main__":
    unittest.main()
