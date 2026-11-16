"""Per-bit Q/K basis search with commuting RoPE rotations."""
import numpy as np
from pairwise_qk import allocate_exact, pair_options


def basis_options(q, k, rotations, bases):
    rotations = tuple(rotations)
    best = {bits: (cost, error, 0) for cost, error, bits in
            pair_options(q, k, rotations)}
    for index, basis in enumerate(bases, start=1):
        if not np.allclose(basis.T @ basis, np.eye(q.shape[0]), atol=1e-12, rtol=0):
            raise ValueError("Basis must be orthogonal")
        if any(not np.allclose(basis @ r, r @ basis, atol=1e-12, rtol=0)
               for r in rotations):
            raise ValueError("Basis must commute with the tested rotations")
        for cost, error, bits in pair_options(basis @ q, basis @ k, rotations):
            if error < best[bits][1]:
                best[bits] = cost, error, index
    return [(cost, error, (bits, index))
            for bits, (cost, error, index) in best.items()]


def planar(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rotations = [planar(offset * .21) for offset in (0, 1, 3, 8)]
    bases = [planar(i * np.pi / 16) for i in range(1, 8)]
    wins = 0
    for _ in range(40):
        pairs = [(rng.normal(size=(2, 8)), rng.normal(size=(2, 8))) for _ in range(3)]
        for q, k in pairs:
            for basis in bases:
                for r in rotations:
                    assert np.allclose((basis @ q).T @ r @ (basis @ k), q.T @ r @ k)
        baseline = [pair_options(q, k, rotations) for q, k in pairs]
        candidate = [basis_options(q, k, rotations, bases) for q, k in pairs]
        old = allocate_exact(baseline, 112)[0]
        new = allocate_exact(candidate, 112)[0]
        assert new <= old + 1e-12
        wins += new < old - 1e-12
    print(f"{wins}/40 strict distortion improvements at identical byte cost")
