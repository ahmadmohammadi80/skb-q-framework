"""Experiment metadata capture for reproducible SKB-Q runs.

Metadata capture records environment facts needed to reproduce a run. It does
not execute experiments, evaluate models, or generate result files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import subprocess
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Immutable metadata captured at experiment setup time."""

    git_commit_hash: str | None
    python_version: str
    package_versions: Mapping[str, str | None] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "package_versions",
            MappingProxyType(dict(sorted(self.package_versions.items()))),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable metadata representation."""

        return {
            "git_commit_hash": self.git_commit_hash,
            "python_version": self.python_version,
            "package_versions": dict(self.package_versions),
            "timestamp_utc": self.timestamp_utc,
        }


def capture_experiment_metadata(
    repo_path: str | Path = ".",
    package_names: Sequence[str] | None = None,
) -> ExperimentMetadata:
    """Capture reproducibility metadata for the current environment."""

    return ExperimentMetadata(
        git_commit_hash=current_git_commit_hash(repo_path),
        python_version=platform.python_version(),
        package_versions=installed_package_versions(package_names),
    )


def current_git_commit_hash(repo_path: str | Path = ".") -> str | None:
    """Return the current git commit hash, or ``None`` outside a git checkout."""

    completed = subprocess.run(
        ["git", "-C", str(Path(repo_path)), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    commit_hash = completed.stdout.strip()
    return commit_hash or None


def installed_package_versions(
    package_names: Sequence[str] | None = None,
) -> Mapping[str, str | None]:
    """Return installed package versions for named packages or all distributions."""

    if package_names is None:
        versions: dict[str, str | None] = {}
        for distribution in importlib_metadata.distributions():
            name = distribution.metadata.get("Name")
            if name:
                versions[name] = distribution.version
        return dict(sorted(versions.items()))

    requested_versions: dict[str, str | None] = {}
    for package_name in package_names:
        requested_versions[package_name] = _package_version(package_name)
    return dict(sorted(requested_versions.items()))


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None
