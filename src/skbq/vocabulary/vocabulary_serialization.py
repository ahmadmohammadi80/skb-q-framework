"""Deterministic serialization for SKB-Q vocabulary stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from skbq.vocabulary.vocabulary_store import VocabularyStore


def vocabulary_to_mapping(vocabulary: VocabularyStore | dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic mapping for a vocabulary store or mapping."""

    if isinstance(vocabulary, VocabularyStore):
        return vocabulary.to_mapping()
    return vocabulary


def canonical_vocabulary_json(vocabulary: VocabularyStore | dict[str, Any]) -> str:
    """Return canonical deterministic JSON for a vocabulary."""

    return json.dumps(
        vocabulary_to_mapping(vocabulary),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def vocabulary_hash(vocabulary: VocabularyStore | dict[str, Any]) -> str:
    """Return SHA-256 hash for canonical vocabulary JSON."""

    return hashlib.sha256(canonical_vocabulary_json(vocabulary).encode("utf-8")).hexdigest()


def write_vocabulary_json(
    vocabulary: VocabularyStore,
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write deterministic ``vocabulary.json`` content."""

    output_path = Path(path)
    if output_path.is_dir():
        output_path = output_path / "vocabulary.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing vocabulary file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(vocabulary.to_mapping(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_vocabulary_json(path: str | Path) -> dict[str, Any]:
    """Read a serialized vocabulary JSON file as a mapping."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
