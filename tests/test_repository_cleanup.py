from __future__ import annotations

import re
import subprocess
from pathlib import Path

PUBLIC_ACTIONS = {"setup", "rsna", "symile", "symile-test", "release", "serve", "check"}


def test_make_default_displays_discoverable_help() -> None:
    default_recipe = subprocess.run(
        ["make", "--no-print-directory", "-n"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_recipe = subprocess.run(
        ["make", "--no-print-directory", "-n", "help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert default_recipe.stdout == help_recipe.stdout
    assert "uv run" not in default_recipe.stdout
    default = subprocess.run(
        ["make", "--no-print-directory"], check=True, capture_output=True, text=True
    )
    explicit = subprocess.run(
        ["make", "--no-print-directory", "help"], check=True, capture_output=True, text=True
    )
    assert default.stdout == explicit.stdout
    assert set(re.findall(r"\bmake\s+([a-z][a-z-]*)", explicit.stdout)) == PUBLIC_ACTIONS


def test_make_exposes_only_the_public_surface() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    phony = re.findall(r"^\.PHONY:\s+(.+)$", text, re.MULTILINE)
    assert set(" ".join(phony).split()) == PUBLIC_ACTIONS | {"help"}
    targets = re.findall(r"^([a-z][a-z-]*):", text, re.MULTILINE)
    assert set(targets) == PUBLIC_ACTIONS | {"help"}


def test_make_lifecycle_recipes_are_thin_workflow_forwarders() -> None:
    expected = {
        "rsna": "uv run python -m beyondcxr.workflow rsna",
        "symile": "uv run python -m beyondcxr.workflow symile",
        "symile-test": "uv run python -m beyondcxr.workflow symile-test",
        "release": "uv run python -m beyondcxr.workflow release",
        "serve": "uv run --extra serving python -m beyondcxr.workflow serve",
    }
    for target, command in expected.items():
        result = subprocess.run(
            ["make", "--no-print-directory", "-n", target],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == command


def test_make_setup_and_check_have_the_exact_owned_commands() -> None:
    setup = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-n",
            "setup",
            "BACKUP_ROOT=/backup path",
            "RSNA_SOURCE_ROOT=/rsna path",
            "SYMILE_SOURCE_ROOT=/symile path",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert setup.returncode == 0, setup.stderr
    assert setup.stdout.splitlines() == [
        "uv sync --locked --group dev --group acquisition --extra serving",
        'uv run python -m beyondcxr.workflow setup --backup-root "/backup path" '
        '--rsna-source-root "/rsna path" --symile-source-root "/symile path"',
    ]
    check = subprocess.run(
        ["make", "--no-print-directory", "-n", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.splitlines() == [
        "uv lock --check",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest",
    ]
