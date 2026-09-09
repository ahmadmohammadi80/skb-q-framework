"""Artifact writing for SKB-Q vocabulary data preparation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from skbq.vocabulary import VocabularyStore, vocabulary_hash, write_vocabulary_json


@dataclass(frozen=True, slots=True)
class VocabularyArtifactManifest:
    """Manifest for generated vocabulary artifacts."""

    artifact_directory: Path
    vocabulary_path: Path
    metadata_path: Path
    hash_path: Path
    vocabulary_hash: str

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-serializable manifest."""

        return {
            "artifact_directory": str(self.artifact_directory),
            "vocabulary_path": str(self.vocabulary_path),
            "metadata_path": str(self.metadata_path),
            "hash_path": str(self.hash_path),
            "vocabulary_hash": self.vocabulary_hash,
        }


@dataclass(frozen=True, slots=True)
class VocabularyArtifactWriter:
    """Write deterministic vocabulary artifacts with overwrite protection."""

    overwrite: bool = False

    def write(
        self,
        vocabulary: VocabularyStore,
        output_artifact_dir: str | Path,
        metadata: dict[str, object],
    ) -> VocabularyArtifactManifest:
        """Write vocabulary, metadata, and hash artifacts."""

        artifact_directory = Path(output_artifact_dir) / "vocabulary"
        artifact_directory.mkdir(parents=True, exist_ok=True)
        vocabulary_path = artifact_directory / "vocabulary.json"
        metadata_path = artifact_directory / "metadata.json"
        hash_path = artifact_directory / "hash"

        if not self.overwrite:
            for path in (vocabulary_path, metadata_path, hash_path):
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

        digest = vocabulary_hash(vocabulary)
        write_vocabulary_json(vocabulary, vocabulary_path, overwrite=self.overwrite)
        _write_json(metadata_path, metadata, overwrite=self.overwrite)
        _write_text(hash_path, digest + "\n", overwrite=self.overwrite)

        return VocabularyArtifactManifest(
            artifact_directory=artifact_directory,
            vocabulary_path=vocabulary_path,
            metadata_path=metadata_path,
            hash_path=hash_path,
            vocabulary_hash=digest,
        )


def _write_json(path: Path, payload: dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(payload, encoding="utf-8")
