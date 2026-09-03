"""Central read-only source and repository-local write path policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathPolicyError(ValueError):
    """Raised when a path crosses the legacy-read/new-write boundary."""


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RepositoryPathPolicy:
    repository_root: Path
    source_root: Path

    @classmethod
    def create(
        cls, repository_root: str | Path, source_root: str | Path
    ) -> "RepositoryPathPolicy":
        repository = Path(repository_root).resolve()
        source = Path(source_root).resolve()
        if repository == source or _within(repository, source) or _within(source, repository):
            raise PathPolicyError("Source and official repository must be independent trees")
        return cls(repository_root=repository, source_root=source)

    def resolve_source(self, configured_path: str | Path) -> Path:
        path = Path(configured_path)
        resolved = (path if path.is_absolute() else self.source_root / path).resolve()
        if not _within(resolved, self.source_root):
            raise PathPolicyError(f"Source path escapes configured source_root: {resolved}")
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    @property
    def allowed_write_roots(self) -> tuple[Path, ...]:
        return tuple(
            (self.repository_root / name).resolve()
            for name in ("outputs", "splits", "artifacts")
        )

    def resolve_write(self, configured_path: str | Path) -> Path:
        path = Path(configured_path)
        resolved = (path if path.is_absolute() else self.repository_root / path).resolve()
        if _within(resolved, self.source_root):
            raise PathPolicyError(f"Writes to the legacy source tree are forbidden: {resolved}")
        if not any(resolved == root or _within(resolved, root) for root in self.allowed_write_roots):
            raise PathPolicyError(
                f"Write path must be inside outputs/, splits/ or artifacts/: {resolved}"
            )
        return resolved

