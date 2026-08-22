from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_purge_generated_removes_abandoned_preopen_symile_control(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    freeze = workspace / "private/control/symile/freezes/pretest-freeze-synthetic/manifest.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text('{"pretest_freeze_schema_version":1}\n', encoding="utf-8")

    purge = subprocess.run(
        [
            "make",
            "-f",
            str(Path("Makefile").resolve()),
            "-C",
            str(workspace),
            "purge-generated",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert purge.returncode == 0, purge.stderr
    assert not (workspace / "private/control/symile").exists()


def test_symile_make_requires_and_forwards_backup_root(tmp_path: Path) -> None:
    arguments = tmp_path / "arguments.json"
    launcher = tmp_path / "uv"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "from pathlib import Path\n"
        f"Path({str(arguments)!r}).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    environment = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    command = [
        "make",
        "-f",
        str(Path("Makefile").resolve()),
        "-C",
        str(tmp_path),
        "symile-campaign",
    ]
    for values, missing in (
        (["SOURCE_ROOT=", "BACKUP_ROOT="], "SOURCE_ROOT"),
        (["SOURCE_ROOT=source", "BACKUP_ROOT="], "BACKUP_ROOT"),
    ):
        result = subprocess.run(
            [*command, *values], env=environment, capture_output=True, text=True, check=False
        )
        assert result.returncode != 0
        assert missing in result.stdout
        assert not arguments.exists()
    backup = tmp_path / "persistent backup"
    result = subprocess.run(
        [*command, "SOURCE_ROOT=source", f"BACKUP_ROOT={backup}"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    forwarded = json.loads(arguments.read_text(encoding="utf-8"))
    assert forwarded[forwarded.index("--backup-root") + 1] == str(backup)
    assert forwarded[forwarded.index("--source-root") + 1] == "source"


def test_clean_and_purge_generated_own_distinct_reproducible_scopes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    generated = (
        "reports/keep.txt",
        "models/keep.txt",
        "private/predictions/rsna/prediction-test/predictions.parquet",
        "private/localization/localization-test/summary.json",
        "mlartifacts/keep.txt",
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
    backup = tmp_path / "external-backup" / ".staging-preserved" / "evidence.tmp"
    backup.parent.mkdir(parents=True)
    backup.write_text("content\n", encoding="utf-8")

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

    test_open = workspace / "private/control/symile/test-open.json"
    test_open.parent.mkdir(parents=True)
    test_open.write_text("synthetic opened-record sentinel\n", encoding="utf-8")
    blocked = subprocess.run(
        ["make", "-f", str(makefile), "-C", str(workspace), "purge-generated"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "Refusing to purge" in blocked.stdout
    assert all((workspace / path).is_file() for path in (*generated, *preserved))
    assert test_open.is_file()
    test_open.unlink()

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
    assert backup.read_text(encoding="utf-8") == "content\n"
