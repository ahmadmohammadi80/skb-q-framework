"""Frozen encoder interface for SKB-Q backbone experiments.

Concrete encoders are expected to be frozen during SKB-Q evaluation. This module
defines only the contract; it does not implement RAMP, GAMMA, or any trainable
encoding behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import math

GraphEmbedding = tuple[float, ...]


class FrozenEncoder(ABC):
    """Abstract interface for frozen operator or graph encoders."""

    @abstractmethod
    def encode(self, operator: object) -> GraphEmbedding:
        """Return a deterministic embedding for an operator-like object."""


def validate_embedding(embedding: Sequence[float]) -> GraphEmbedding:
    """Coerce an encoder output into the canonical immutable embedding type."""

    if len(embedding) == 0:
        raise ValueError("embedding cannot be empty")
    result = tuple(float(value) for value in embedding)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding values must be finite")
    return result
