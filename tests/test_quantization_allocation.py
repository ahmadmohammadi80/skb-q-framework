"""Tests for the quantization allocation layer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skbq.backbone import (  # noqa: E402
    AllocationTarget,
    FrozenBackbone,
    UniformFrozenPolicy,
)
from skbq.config.schema import BudgetConfig  # noqa: E402
from skbq.graph import SyntheticArchitectureSpec, build_transformer_graph  # noqa: E402
from skbq.quantization import (  # noqa: E402
    BitBudget,
    BitWidthCandidate,
    ConstraintSet,
    DeterministicAllocationResolver,
    GroupUniformityConstraint,
    MaxBitWidthConstraint,
    MinBitWidthConstraint,
    OperatorAllocation,
    OperatorAllocationPlan,
    PolicyActionSpace,
    QuantizationAllocationLayer,
    TotalBudgetConstraint,
    default_bit_width_candidates,
)


class BitBudgetTests(unittest.TestCase):
    def test_bit_budget_from_experiment_config(self) -> None:
        budget = BitBudget.from_budget_config(BudgetConfig(total=128, unit="bits"))

        self.assertEqual(budget.total_bits, 128)
        self.assertEqual(budget.unit, "bits")
        self.assertEqual(budget.consumed_bits(64).total_bits, 64)

    def test_bit_budget_rejects_negative_total(self) -> None:
        with self.assertRaises(ValueError):
            BitBudget(total_bits=-1)


class BitWidthCandidateTests(unittest.TestCase):
    def test_default_candidates_are_deterministic(self) -> None:
        first = default_bit_width_candidates()
        second = default_bit_width_candidates()

        self.assertEqual(first, second)
        self.assertEqual(tuple(item.bit_width for item in first), (1, 2, 4, 8))

    def test_storage_bits_multiplies_parameter_count(self) -> None:
        candidate = BitWidthCandidate(bit_width=4, candidate_id="bw4")

        self.assertEqual(candidate.storage_bits(10), 40)
        self.assertEqual(BitWidthCandidate(bit_width=0, candidate_id="bw0").storage_bits(100), 0)



class PolicyActionSpaceTests(unittest.TestCase):
    def test_candidates_resolve_by_target_type(self) -> None:
        narrow = BitWidthCandidate(bit_width=2, candidate_id="bw2")
        action_space = PolicyActionSpace(
            target_type_candidates={"graph_operator": (narrow,)},
        )
        target = AllocationTarget("layer0.attn", "graph_operator")

        candidates = action_space.candidates_for(target)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, "bw2")


class DeterministicResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        self.backbone = FrozenBackbone(policy=UniformFrozenPolicy())
        self.policy_allocation = self.backbone.allocate_graph(self.graph, budget=100.0)
        self.parameter_counts = {
            node.operator_id: node.parameter_count for node in self.graph.nodes
        }

    def test_resolver_maps_p_g_to_operator_allocations(self) -> None:
        resolver = DeterministicAllocationResolver()
        plan = resolver.resolve(
            policy_allocation=self.policy_allocation,
            bit_budget=BitBudget(total_bits=10_000),
            parameter_counts=self.parameter_counts,
        )

        self.assertEqual(len(plan.allocations), len(self.graph.nodes))
        self.assertGreater(plan.total_storage_bits, 0)
        self.assertLessEqual(plan.total_storage_bits, 10_000)
        self.assertEqual(len(plan.allocation_hash()), 64)
        for allocation in plan.allocations:
            self.assertIn(allocation.target_type, {"graph_operator"})
            self.assertIsNotNone(allocation.operator_id)

    def test_resolver_is_deterministic(self) -> None:
        resolver = DeterministicAllocationResolver()
        kwargs = {
            "policy_allocation": self.policy_allocation,
            "bit_budget": BitBudget(total_bits=10_000),
            "parameter_counts": self.parameter_counts,
        }

        first = resolver.resolve(**kwargs)
        second = resolver.resolve(**kwargs)

        self.assertEqual(first.allocation_hash(), second.allocation_hash())

    def test_group_uniformity_upgrades_entire_group(self) -> None:
        targets = tuple(self.policy_allocation.target_allocations)
        groups = {
            targets[0].target.target_id: "block0",
            targets[1].target.target_id: "block0",
        }
        resolver = DeterministicAllocationResolver(
            constraints=ConstraintSet(
                constraints=(TotalBudgetConstraint(), GroupUniformityConstraint(groups)),
            ),
        )
        plan = resolver.resolve(
            policy_allocation=self.policy_allocation,
            bit_budget=BitBudget(total_bits=1_000_000),
            parameter_counts=self.parameter_counts,
            quantization_groups=groups,
        )

        group_widths = {
            allocation.bit_width_candidate.bit_width
            for allocation in plan.allocations
            if allocation.quantization_group_id == "block0"
        }
        self.assertEqual(len(group_widths), 1)


class QuantizationAllocationLayerTests(unittest.TestCase):
    def test_allocate_graph_runs_pi_theta_and_resolves_operators(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=2))
        layer = QuantizationAllocationLayer()
        plan = layer.allocate_graph(graph, BitBudget(total_bits=50_000))

        self.assertEqual(len(plan.allocations), len(graph.nodes))
        self.assertEqual(plan.metadata["notation"], "P(G)->operator_allocation")
        self.assertEqual(plan.provenance.policy_name, "uniform_pi_theta")
        self.assertLessEqual(plan.total_storage_bits, 50_000)

    def test_constraints_reject_exceeded_budget(self) -> None:
        graph = build_transformer_graph(SyntheticArchitectureSpec("transformer", num_layers=1))
        layer = QuantizationAllocationLayer(
            constraints=ConstraintSet(
                constraints=(
                    TotalBudgetConstraint(),
                    MinBitWidthConstraint(min_bit_width=8),
                    MaxBitWidthConstraint(max_bit_width=8),
                ),
            ),
        )

        with self.assertRaises(ValueError):
            layer.allocate_graph(graph, BitBudget(total_bits=1))


class OperatorAllocationSerializationTests(unittest.TestCase):
    def test_operator_allocation_plan_serializes_deterministically(self) -> None:
        allocation = OperatorAllocation(
            target_id="layer0.attn",
            target_type="graph_operator",
            operator_id="layer0.attn",
            layer_id="0",
            bit_width_candidate=BitWidthCandidate(bit_width=4, candidate_id="bw4"),
            parameter_count=100,
            policy_mass=2.5,
        )
        plan = OperatorAllocationPlan(allocations=(allocation,))

        first_json = plan.canonical_json()
        second_json = OperatorAllocationPlan(
            allocations=(allocation,),
        ).canonical_json()

        self.assertEqual(first_json, second_json)
        self.assertIn("layer0.attn", first_json)


if __name__ == "__main__":
    unittest.main()
