"""Publish with rollback or exactly reproduce public derivatives from a campaign export."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.binding import ResultBinding, load_result_binding
from beyondcxr.release.filesystem import directory_bytes
from beyondcxr.release.inputs import (
    derive_result_binding,
    export_manifest_sha256,
    validate_release_inputs,
)
from beyondcxr.release.projection import PublicResultProjection
from beyondcxr.release.render import (
    parse_result_region,
    public_result_surface,
    render_public_results,
    replace_result_region,
)


@dataclass
class _Regenerated:
    root: Path
    readme: bytes
    model_card: bytes
    temporary: tempfile.TemporaryDirectory[str]


def publish_results(
    *,
    artifact_root: str | Path,
    output_root: str | Path,
    readme_path: str | Path,
    model_card_path: str | Path,
) -> None:
    """Derive the binding, prepare every public byte, then install as one rollback unit."""
    regenerated = _regenerate(
        artifact_root=artifact_root,
        binding=None,
        results_readme=Path(output_root) / "README.md",
        readme_path=readme_path,
        model_card_path=model_card_path,
    )
    try:
        _install_with_rollback(
            candidate_results=regenerated.root,
            output_root=Path(output_root),
            readme_path=Path(readme_path),
            readme=regenerated.readme,
            model_card_path=Path(model_card_path),
            model_card=regenerated.model_card,
        )
    finally:
        regenerated.temporary.cleanup()


def reproduce_results(
    *,
    binding_path: str | Path,
    artifact_root: str | Path,
    output_root: str | Path,
    readme_path: str | Path,
    model_card_path: str | Path,
) -> None:
    """Regenerate privately and compare the complete committed public surface."""
    binding = load_result_binding(binding_path)
    regenerated = _regenerate(
        artifact_root=artifact_root,
        binding=binding,
        results_readme=Path(output_root) / "README.md",
        readme_path=readme_path,
        model_card_path=model_card_path,
    )
    try:
        expected = directory_bytes(regenerated.root)
        observed = directory_bytes(Path(output_root))
        if observed != expected:
            raise ManifestBuildError("Committed public results do not reproduce exactly")
        observed_readme = Path(readme_path).read_bytes()
        parse_result_region(observed_readme.decode())
        parse_result_region(regenerated.readme.decode())
        if observed_readme != regenerated.readme:
            raise ManifestBuildError("README result region does not reproduce exactly")
        observed_model_card = Path(model_card_path).read_bytes()
        parse_result_region(observed_model_card.decode())
        parse_result_region(regenerated.model_card.decode())
        if observed_model_card != regenerated.model_card:
            raise ManifestBuildError("Model-card result region does not reproduce exactly")
    finally:
        regenerated.temporary.cleanup()


def _regenerate(
    *,
    artifact_root: str | Path,
    binding: ResultBinding | None,
    results_readme: Path,
    readme_path: str | Path,
    model_card_path: str | Path,
) -> _Regenerated:
    from beyondcxr.training.preservation import restore_and_validate_export
    from beyondcxr.training.symile_campaign import validate_restored_campaign

    archive = Path(artifact_root)
    witness = export_manifest_sha256(archive)
    temporary = tempfile.TemporaryDirectory()
    generated = Path(temporary.name) / "results"
    generated_symile = generated / "symile"

    def validate_and_render(
        restored_root: Path,
    ) -> tuple[ResultBinding, PublicResultProjection]:
        campaign = validate_restored_campaign(restored_root)
        derived = derive_result_binding(campaign)
        if binding is not None and binding != derived:
            raise ManifestBuildError("Committed binding differs from the validated campaign")
        projection = validate_release_inputs(
            derived, campaign, restored_root, export_manifest_sha256=witness
        )
        render_public_results(projection, generated_symile)
        observed = {
            path.relative_to(generated_symile).as_posix()
            for path in generated_symile.rglob("*")
            if path.is_file()
        }
        if observed != public_result_surface():
            raise ManifestBuildError("Generated public result membership is invalid")
        return derived, projection

    try:
        derived, projection = restore_and_validate_export(
            archive, restoration_validator=validate_and_render
        )
        if not results_readme.is_file():
            raise ManifestBuildError("Public result README template is unavailable")
        shutil.copyfile(results_readme, generated / "README.md")
        (generated_symile / "binding.json").write_bytes(derived.canonical_bytes())
        readme = replace_result_region(
            Path(readme_path).read_text(encoding="utf-8"), _readme_fragment(projection)
        ).encode()
        model_card = replace_result_region(
            Path(model_card_path).read_text(encoding="utf-8"), _model_card_fragment(projection)
        ).encode()
        return _Regenerated(generated, readme, model_card, temporary)
    except Exception:
        temporary.cleanup()
        raise


def _readme_fragment(projection: PublicResultProjection) -> str:
    development = projection.development
    held = projection.held_out
    development_cxr = development["ensemble_metrics"]["cxr_densenet"]
    development_primary = development["ensemble_metrics"]["cxr_labs_gated"]
    development_effect = {
        metric: development_primary[metric] - development_cxr[metric]
        for metric in ("roc_auc", "average_precision", "brier_score")
    }
    held_cxr = held["predictor_views"]["cxr_densenet"]
    primary = held["predictor_views"]["cxr_labs_gated"]
    effect = held["paired_effects"]["primary"]["roc_auc"]
    return "\n".join(
        (
            "### Development OOF evidence",
            "",
            f"- CXR AUROC / AP / Brier: {development_cxr['roc_auc']:.3f} / "
            f"{development_cxr['average_precision']:.3f} / {development_cxr['brier_score']:.3f}",
            f"- Primary gated AUROC / AP / Brier: {development_primary['roc_auc']:.3f} / "
            f"{development_primary['average_precision']:.3f} / "
            f"{development_primary['brier_score']:.3f}",
            f"- Gated minus CXR ΔAUROC / ΔAP / ΔBrier: {development_effect['roc_auc']:.3f} / "
            f"{development_effect['average_precision']:.3f} / "
            f"{development_effect['brier_score']:.3f}",
            "",
            "### Held-out test evidence",
            "",
            f"- CXR AUROC / AP / Brier: {held_cxr['roc_auc']:.3f} / "
            f"{held_cxr['average_precision']:.3f} / {held_cxr['brier_score']:.3f}",
            f"- Primary gated AUROC / AP / Brier: {primary['roc_auc']:.3f} / "
            f"{primary['average_precision']:.3f} / {primary['brier_score']:.3f}",
            (
                "- Gated minus CXR ΔAUROC: "
                f"{effect['point']:.3f} (95% CI {effect['lower']:.3f} to {effect['upper']:.3f})"
            ),
            "",
            "See [aggregate tables and figures]"
            "(https://github.com/Youssef-SH/BeyondCXR/tree/main/results/symile/).",
        )
    )


def _model_card_fragment(projection: PublicResultProjection) -> str:
    development = projection.development
    held = projection.held_out
    provenance = projection.provenance
    development_cxr = development["ensemble_metrics"]["cxr_densenet"]
    development_primary = development["ensemble_metrics"]["cxr_labs_gated"]
    development_effect = development_primary["roc_auc"] - development_cxr["roc_auc"]
    held_cxr = held["predictor_views"]["cxr_densenet"]
    primary = held["predictor_views"]["cxr_labs_gated"]
    effect = held["paired_effects"]["primary"]["roc_auc"]
    thresholds = projection.thresholds
    packages = provenance["primary_package_ids"]
    pretrained = provenance["pretrained_scientific_identity"]
    materializations = provenance["pretrained_weight_materializations"]
    subgroup_policy = projection.subgroups["policy"]
    subgroup_rows = [
        row
        for row in projection.subgroups["rows"]
        if not row["supported"] or row["estimate"] == "Mean-logit OOF ensemble"
    ]
    lines = [
        "### Development OOF evidence",
        "",
        f"- CXR AUROC: {development_cxr['roc_auc']:.3f}",
        f"- Primary gated AUROC: {development_primary['roc_auc']:.3f}",
        f"- Gated minus CXR ΔAUROC: {development_effect:.3f}",
        "",
        "### Development subgroup characterization",
        "",
        (
            "- Prespecified support requires "
            f"n ≥ {subgroup_policy['minimum_samples']}, "
            f"positives ≥ {subgroup_policy['minimum_positives']}, and "
            f"negatives ≥ {subgroup_policy['minimum_negatives']}."
        ),
        *(
            (
                f"- {row['stratum']}: n={row['n']}, positives={row['positives']}, "
                f"negatives={row['negatives']}; gated minus CXR ΔAUROC="
                f"{row['effect_roc_auc']:.3f}, ΔAP={row['effect_average_precision']:.3f}, "
                f"ΔBrier={row['effect_brier_score']:.3f}."
                if row["supported"]
                else f"- {row['stratum']}: unavailable under the prespecified support rule."
            )
            for row in subgroup_rows
        ),
        "",
        "### Held-out test evidence",
        "",
        f"- CXR AUROC: {held_cxr['roc_auc']:.3f}",
        f"- Primary AUROC: {primary['roc_auc']:.3f}",
        f"- Primary Average Precision: {primary['average_precision']:.3f}",
        f"- Primary Brier score: {primary['brier_score']:.3f}",
        f"- Calibration slope: {primary['calibration_slope']:.3f}",
        f"- Calibration intercept: {primary['calibration_intercept']:.3f}",
        (
            "- Gated minus CXR ΔAUROC: "
            f"{effect['point']:.3f} (95% CI {effect['lower']:.3f} to {effect['upper']:.3f})"
        ),
        f"- Development Youden-J threshold (metadata): {thresholds['youden_j']:.6g}",
        (
            "- Development 90%-sensitivity threshold (metadata): "
            f"{thresholds['target_sensitivity']:.6g}"
        ),
        "",
        "### Validated scientific coordinates",
        "",
        "- Ordered primary packages: " + ", ".join(f"`{item}`" for item in packages),
        f"- Bundle: `{provenance['bundle_id']}`",
        f"- Split assignment: `{provenance['split_assignment_id']}`",
        f"- CV assignment: `{provenance['cv_assignment_id']}`",
        f"- Pre-test freeze: `{provenance['pretest_freeze_id']}`",
        f"- Global result: `{provenance['global_result_id']}`",
        f"- Science Git commit: `{provenance['science_git_commit']}`",
        "",
        "### Pretrained scientific weight",
        "",
        f"- Declared identity: `{pretrained['declared_name']}`",
        f"- Stable identifier: `{pretrained['stable_identifier']}`",
        f"- Scientific content SHA-256: `{pretrained['sha256']}`",
        *(
            f"- Seed {item['seed']} materialization: `{item['cache_filename']}`, "
            f"{item['byte_size']} bytes, SHA-256 `{item['sha256']}`"
            for item in materializations
        ),
        "",
        "### Reproducibility and fixity witnesses",
        "",
        f"- Science lock SHA-256: `{provenance['science_dependency_lock_sha256']}`",
        f"- Bundle-manifest SHA-256: `{provenance['bundle_manifest_sha256']}`",
        f"- Export-manifest SHA-256: `{provenance['export_manifest_sha256']}`",
        "",
        (
            "Serving uses the ordered three-member mean-logit ensemble. Thresholds are research "
            "metadata and are not emitted as decisions by `/predict`."
        ),
    ]
    return "\n".join(lines)


def _install_with_rollback(
    *,
    candidate_results: Path,
    output_root: Path,
    readme_path: Path,
    readme: bytes,
    model_card_path: Path,
    model_card: bytes,
) -> None:
    parent = output_root.parent.resolve()
    with tempfile.TemporaryDirectory(dir=parent, prefix=".beyondcxr-release-") as temporary_name:
        temporary = Path(temporary_name)
        prepared_results = temporary / "results"
        shutil.copytree(candidate_results, prepared_results)
        prepared_readme = temporary / "README.md"
        prepared_model_card = temporary / "model_card.md"
        prepared_readme.write_bytes(readme)
        prepared_model_card.write_bytes(model_card)
        backups = temporary / "backups"
        backups.mkdir()
        targets = (
            (output_root, prepared_results, backups / "results"),
            (readme_path, prepared_readme, backups / "README.md"),
            (model_card_path, prepared_model_card, backups / "model_card.md"),
        )
        installed: list[tuple[Path, Path, bool]] = []
        try:
            for target, prepared, backup in targets:
                had_original = target.exists()
                installed.append((target, backup, had_original))
                if target.exists():
                    os.replace(target, backup)
                os.replace(prepared, target)
        except BaseException as installation_failure:
            rollback_failures: list[BaseException] = []
            retry: list[tuple[Path, Path, bool]] = []
            for target, backup, had_original in reversed(installed):
                try:
                    _restore_install_target(target, backup, had_original)
                except BaseException as rollback_failure:
                    rollback_failures.append(rollback_failure)
                    retry.append((target, backup, had_original))
            for target, backup, had_original in retry:
                try:
                    _restore_install_target(target, backup, had_original)
                except BaseException as rollback_failure:
                    rollback_failures.append(rollback_failure)
            if rollback_failures:
                raise BaseExceptionGroup(
                    "Public result installation failed and rollback encountered failures",
                    [installation_failure, *rollback_failures],
                ) from installation_failure
            raise


def _restore_install_target(target: Path, backup: Path, had_original: bool) -> None:
    if backup.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        os.replace(backup, target)
    elif not had_original:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
