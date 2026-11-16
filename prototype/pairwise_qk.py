"""Exact allocation for disjoint Q/K pairs using weight-only logit distortion."""
from itertools import product

import numpy as np


def quantize(weight, bits):
    """Symmetric rowwise quantization with a stored float32 scale per row."""
    limit = (1 << (bits - 1)) - 1
    scale = (np.max(np.abs(weight), axis=1) / limit).astype(np.float32)
    divisor = np.where(scale > 0, scale, 1)
    codes = np.clip(np.rint(weight / divisor[:, None]), -limit, limit)
    row_bytes = (weight.shape[1] * bits + 7) // 8
    return codes * scale[:, None], weight.shape[0] * (row_bytes + 4)


def pair_options(q, k, rotations, *, interaction=True):
    """Use a fixed uniform weighting over supplied relative rotations."""
    rotations = tuple(rotations)
    if not rotations:
        raise ValueError("Supply relative rotations or a single identity matrix")
    q_weights = {bits: quantize(q, bits) for bits in (4, 8)}
    k_weights = {bits: quantize(k, bits) for bits in (4, 8)}
    reference = [q.T @ rotation @ k for rotation in rotations]
    costs, errors = {}, {}
    for a, b in product((4, 8), repeat=2):
        q_hat, q_bytes = q_weights[a]
        k_hat, k_bytes = k_weights[b]
        costs[a, b] = q_bytes + k_bytes
        errors[a, b] = float(np.mean([
            np.linalg.norm(q_hat.T @ rotation @ k_hat - target, "fro") ** 2
            for rotation, target in zip(rotations, reference)
        ]))
    if not interaction:
        errors[8, 8] = errors[8, 4] + errors[4, 8] - errors[4, 4]
    return [(costs[bits], error, bits) for bits, error in errors.items()]


def allocate_exact(groups, budget_bytes):
    """Minimize total distortion at an exact storage budget, or reject it."""
    states = {0: (0.0, ())}
    for options in groups:
        next_states = {}
        for spent, (error, allocation) in states.items():
            for cost, distortion, bits in options:
                total = spent + cost
                candidate = (error + distortion, allocation + (bits,))
                if total <= budget_bytes and (
                    total not in next_states or candidate < next_states[total]
                ):
                    next_states[total] = candidate
        states = next_states
    if budget_bytes not in states:
        raise ValueError("Exact byte budget is infeasible")
    return states[budget_bytes]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    disagreements = 0
    for _ in range(40):
        pairs = [(rng.normal(size=(4, 8)), rng.normal(size=(4, 8))) for _ in range(3)]
        full = [pair_options(q, k, [np.eye(4)]) for q, k in pairs]
        additive = [pair_options(q, k, [np.eye(4)], interaction=False) for q, k in pairs]
        budget = 224
        for groups in (full, additive):
            brute = min(
                (sum(o[1] for o in choices), tuple(o[2] for o in choices))
                for choices in product(*groups) if sum(o[0] for o in choices) == budget
            )
            assert np.isclose(allocate_exact(groups, budget)[0], brute[0], rtol=1e-12, atol=1e-12)
        disagreements += allocate_exact(full, budget)[1] != allocate_exact(additive, budget)[1]
    print(f"80 exact allocations matched enumeration; {disagreements}/40 policy differences")
