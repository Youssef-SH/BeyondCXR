"""Dependency-light operational source and lock provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

from beyondcxr.data.hashing import sha256_file

PROJECT_ROOT_FILES = frozenset(
    {
        ".dockerignore",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".python-version",
        "CHANGELOG.md",
        "CITATION.cff",
        "Dockerfile",
        "LICENSE",
        "Makefile",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "uv.lock",
    }
)
_PROJECT_DIRECTORIES = frozenset(
    {"src", "tests", "scripts", "configs", "docs", "examples", "data", "results", ".github"}
)


def project_untracked_paths(repository_root: str | Path) -> set[str]:
    """Return non-ignored untracked paths within the project's source surface."""
    output = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    ).stdout
    root_files = {name.encode("utf-8") for name in PROJECT_ROOT_FILES}
    directories = {name.encode("utf-8") for name in _PROJECT_DIRECTORIES}
    return {
        name.decode("utf-8")
        for name in output.split(b"\0")
        if name in root_files or name.split(b"/", 1)[0] in directories
    }


def git_revision(repository_root: str | Path = ".") -> tuple[str, bool]:
    """Return the commit and project-owned dirty-worktree flag."""
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty or bool(project_untracked_paths(repository_root))


def uv_lock_sha256(path: str | Path = "uv.lock") -> str:
    """Return the SHA-256 of the exact dependency lock."""
    lock = Path(path)
    if not lock.is_file():
        raise FileNotFoundError(f"Dependency lock is missing: {lock}")
    return sha256_file(lock)
