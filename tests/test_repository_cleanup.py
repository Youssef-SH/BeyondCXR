from __future__ import annotations

import subprocess
from pathlib import Path


def test_clean_and_purge_generated_own_distinct_reproducible_scopes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    generated = (
        "reports/keep.txt",
        "models/keep.txt",
        "private/predictions/rsna/prediction-test/predictions.parquet",
        "private/localization/localization-test/summary.json",
        "mlartifacts/keep.txt",
        "mlruns/keep.txt",
        "mlflow.db",
        "mlflow.db-wal",
        "mlflow.db-shm",
        "data/cache/rsna/cache-test/images.npy",
        "data/manifests/rsna/bundles/bundle-test/bundle.txt",
        "data/manifests/rsna/CURRENT",
        "data/manifests/symile/bundles/bundle-test/bundle.txt",
        "data/manifests/symile/cv/cv-assignment-test/assignments.parquet",
        "data/manifests/symile/CURRENT",
        "outbox/results.tar.gz",
    )
    transient = (
        ".pytest_cache/cache.txt",
        ".ruff_cache/cache.txt",
        ".mypy_cache/cache.txt",
        "src/__pycache__/module.pyc",
        "reports/.staging-test/partial.txt",
        "reports/.comparison.tmp",
    )
    preserved = (
        ".git/keep.txt",
        ".venv/__pycache__/keep.pyc",
        "data/raw/rsna/__pycache__/keep.pyc",
        "data/raw/rsna/source.dcm",
    )
    for path in (*generated, *transient, *preserved):
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content\n", encoding="utf-8")
    makefile = Path("Makefile").resolve()

    clean = subprocess.run(
        ["make", "-f", str(makefile), "-C", str(workspace), "clean"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr
    assert all((workspace / path).is_file() for path in (*generated, *preserved))
    assert all(not (workspace / path).exists() for path in transient)
    assert not (workspace / ".pytest_cache").exists()
    assert not (workspace / ".ruff_cache").exists()
    assert not (workspace / ".mypy_cache").exists()
    assert not (workspace / "src/__pycache__").exists()
    assert not (workspace / "reports/.staging-test").exists()

    purge = subprocess.run(
        ["make", "-f", str(makefile), "-C", str(workspace), "purge-generated"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert purge.returncode == 0, purge.stderr
    assert all(not (workspace / path).exists() for path in generated)
    assert all((workspace / path).is_file() for path in preserved)
    for path in (
        "reports",
        "models",
        "private/predictions",
        "private/localization",
        "mlartifacts",
        "mlruns",
        "data/cache",
        "outbox",
        "data/manifests/rsna/bundles/bundle-test",
        "data/manifests/symile/bundles/bundle-test",
        "data/manifests/symile/cv/cv-assignment-test",
    ):
        assert not (workspace / path).exists()
    assert (workspace / "data/raw").is_dir()
    assert (workspace / ".git").is_dir()
    assert (workspace / ".venv").is_dir()
