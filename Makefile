.DEFAULT_GOAL := help

.PHONY: help setup rsna symile symile-test release serve check

help:
	@printf '%s\n' \
		'BeyondCXR' \
		'' \
		'  make setup        Set up this machine' \
		'  make rsna         Run/resume the complete RSNA study' \
		'  make symile       Run/resume Symile through the pre-test freeze' \
		'  make symile-test  Open/resume the official held-out Symile test phase' \
		'  make release      Reproduce and validate the final release' \
		'  make serve        Run the validated inference service' \
		'  make check        Run developer checks'

setup:
	uv sync --locked --group dev --group acquisition --extra serving
	uv run python -m beyondcxr.workflow setup $(if $(BACKUP_ROOT),--backup-root "$(BACKUP_ROOT)") $(if $(RSNA_SOURCE_ROOT),--rsna-source-root "$(RSNA_SOURCE_ROOT)") $(if $(SYMILE_SOURCE_ROOT),--symile-source-root "$(SYMILE_SOURCE_ROOT)")

rsna:
	uv run python -m beyondcxr.workflow rsna

symile:
	uv run python -m beyondcxr.workflow symile

symile-test:
	uv run python -m beyondcxr.workflow symile-test

release:
	uv run python -m beyondcxr.workflow release

serve:
	uv run --extra serving python -m beyondcxr.workflow serve

check:
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest
