"""Centralized seed propagation utilities for reproducible SKB-Q runtime code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import random
from types import MappingProxyType

from skbq.config.schema import RandomSeeds


@dataclass(frozen=True, slots=True)
class SeedRegistry:
    """Immutable seed registry with deterministic per-component derivation."""

    values: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SeedRegistry requires at least one seed")
        validated = {
            str(name): _seed_value(seed, f"seed[{name}]")
            for name, seed in sorted(self.values.items(), key=lambda item: str(item[0]))
        }
        object.__setattr__(self, "values", MappingProxyType(validated))

    @classmethod
    def from_random_seeds(cls, random_seeds: RandomSeeds) -> SeedRegistry:
        """Create a seed registry from validated experiment random seeds."""

        return cls(values=random_seeds.values)

    def get(self, name: str, default: int | None = None) -> int:
        """Return a named seed, optionally falling back to ``default``."""

        if name in self.values:
            return self.values[name]
        if default is not None:
            return _seed_value(default, f"default seed for {name}")
        raise KeyError(f"unknown seed name: {name}")

    def derive_seed(
        self,
        component: str,
        stream: str = "default",
        base_seed_name: str = "python",
        default_seed: int | None = None,
    ) -> int:
        """Derive a stable 32-bit seed for a component and stream."""

        base_seed = self.get(base_seed_name, default_seed)
        payload = f"{base_seed}:{component}:{stream}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:16], 16) % (2**32)

    def random_for(
        self,
        component: str,
        stream: str = "default",
        base_seed_name: str = "python",
        default_seed: int | None = None,
    ) -> random.Random:
        """Return an isolated random generator for a component stream."""

        return random.Random(
            self.derive_seed(
                component=component,
                stream=stream,
                base_seed_name=base_seed_name,
                default_seed=default_seed,
            )
        )

    def apply_global(self, seed_name: str = "python") -> None:
        """Seed Python's global random module when a runtime explicitly opts in."""

        random.seed(self.get(seed_name))

    def to_mapping(self) -> dict[str, int]:
        """Return a deterministic mapping representation."""

        return dict(sorted(self.values.items()))


def _seed_value(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer seed")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value
