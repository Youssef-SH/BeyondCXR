"""Repository checks and full final-release acceptance for BeyondCXR."""

from __future__ import annotations

import ast
import csv
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tokenize
import tomllib
import zipfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path, PurePosixPath

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.binding import load_result_binding
from beyondcxr.release.filesystem import directory_bytes
from beyondcxr.release.render import (
    RESULT_END,
    RESULT_START,
    parse_result_region,
    public_result_surface,
)

_LINK = re.compile(r"\[[^]]+\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]+)?\)")
_MAKE_COMMAND = re.compile(r"^\s*make\s+([a-z][a-z0-9-]+)", re.MULTILINE)
_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
_PATH_SEPARATOR = "/"
_HOME_DIRECTORY = "home"
_USERS_DIRECTORY = "Users"
_WINDOWS_SEPARATOR = "\\"
_PRIVATE_PATH = re.compile(
    rf"(?:{re.escape(_PATH_SEPARATOR + _HOME_DIRECTORY + _PATH_SEPARATOR)}[^/\s]+"
    rf"|{re.escape(_PATH_SEPARATOR + _USERS_DIRECTORY + _PATH_SEPARATOR)}[^/\s]+"
    rf"|[A-Za-z]:{re.escape(_WINDOWS_SEPARATOR + _USERS_DIRECTORY + _WINDOWS_SEPARATOR)}"
    rf"[^\\\s]+)"
)
_SECRET = re.compile(
    r"(?i)(?:['\"]?(?:api[_-]?key|secret|token|password)['\"]?)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+-]{16,}"
)
_SUBJECT_ID_FIELDS = frozenset({"subject_id", "patient_id", "patientid"})
_ADMISSION_ID_FIELDS = frozenset({"hadm_id", "label_hadm_id"})
_CXR_STUDY_ID_FIELDS = frozenset({"study_id", "cxr_study_id"})
_ECG_STUDY_ID_FIELDS = frozenset({"ecg_study_id", "ecg_adm", "ecg_file_name"})
_RSNA_UUID_FIELDS = frozenset({"patient_id", "patientid", "image_id", "dicom_id"})
_CXR_DICOM_ID_FIELDS = frozenset({"dicom_id", "cxr_dicom_id", "cxr_24_72_hr"})
_SOURCE_IDENTIFIER_FIELDS = frozenset(
    {"sample_id", "image_id"}
    | _SUBJECT_ID_FIELDS
    | _ADMISSION_ID_FIELDS
    | _CXR_STUDY_ID_FIELDS
    | _ECG_STUDY_ID_FIELDS
    | _CXR_DICOM_ID_FIELDS
)
_UUID_IDENTIFIER = re.compile(r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_MIMIC_CXR_DICOM_IDENTIFIER = re.compile(r"(?i)[0-9a-f]{8}(?:-[0-9a-f]{8}){4}")
_SYMILE_SAMPLE_IDENTIFIER = re.compile(r"symile:(2[0-9]{7})")
_IDENTIFIER_FIELD_PATTERN = "|".join(
    re.escape(field) for field in sorted(_SOURCE_IDENTIFIER_FIELDS, key=len, reverse=True)
)
_IDENTIFIER_KEY_VALUE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_])(?P<key_quote>['\"]?)"
    rf"(?P<field>{_IDENTIFIER_FIELD_PATTERN})(?P=key_quote)(?![A-Za-z0-9_])"
    r"\s*[:=]\s*(?P<value>['\"][^'\"\r\n]*['\"]|[A-Za-z0-9_.:-]+)"
)
_ROOT_FILE_ALLOWLIST = {
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
_RESTRICTED_SUFFIXES = {
    ".dcm",
    ".dicom",
    ".npy",
    ".npz",
    ".parquet",
    ".feather",
    ".joblib",
    ".pkl",
    ".skops",
    ".pt",
    ".pth",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
}
_OPAQUE_BINARY_SUFFIXES = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".ttf",
    ".webp",
    ".woff",
    ".woff2",
}
_MAX_TRACKED_TEXT_BYTES = 8 * 1024 * 1024
_GENERATED_ROOTS = {"models", "mlartifacts", "outbox", "private", "reports"}
_GENERATED_DATA_ROOTS = {"cache", "interim", "manifests", "processed", "raw"}
_GENERATED_ROOT_FILES = {"mlflow.db", "mlflow.db-shm", "mlflow.db-wal"}


def check_repository(root: str | Path, *, final: bool = False) -> None:
    """Validate documentation, privacy, packaging, result-surface, and release invariants."""
    repository = Path(root).resolve()
    tracked = _release_candidate_paths(repository)
    unexpected_root_files = [
        path
        for path in tracked
        if path.parent == repository and path.name not in _ROOT_FILE_ALLOWLIST
    ]
    restricted = [path for path in tracked if path.suffix.lower() in _RESTRICTED_SUFFIXES]
    if restricted:
        raise ManifestBuildError(f"Restricted artifact is tracked: {restricted[0].as_posix()}")
    markdown = [path for path in tracked if path.suffix.lower() == ".md"]
    make_targets = _make_targets(repository / "Makefile")
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for target in _LINK.findall(text):
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(repository) or not resolved.exists():
                raise ManifestBuildError(f"Broken documentation link in {path.as_posix()}")
        for command in _MAKE_COMMAND.findall(text):
            if command not in make_targets:
                raise ManifestBuildError(f"Unknown Make command in {path.as_posix()}: {command}")
        for command in _BASH_BLOCK.findall(text):
            if subprocess.run(
                ["bash", "-n"], input=command, text=True, capture_output=True, check=False
            ).returncode:
                raise ManifestBuildError(
                    f"Shell-invalid documented Bash command in {path.as_posix()}"
                )
    for path in tracked:
        if _path_contains_source_identifier(path.relative_to(repository)):
            raise ManifestBuildError(
                f"Tracked path contains row-level material: {path.relative_to(repository)}"
            )
        text = _tracked_text(path)
        if _PRIVATE_PATH.search(text) or _SECRET.search(text):
            raise ManifestBuildError(f"Tracked text contains private material: {path.as_posix()}")
        patient_material = (
            _check_python_source(path, text)
            if path.suffix.lower() == ".py"
            else (_text_contains_source_identifier(text) or _table_contains_source_identifier(text))
        )
        if patient_material:
            raise ManifestBuildError(f"Tracked text contains row-level material: {path.as_posix()}")
    generated = [path for path in tracked if _is_generated_repository_path(repository, path)]
    if generated:
        raise ManifestBuildError(
            f"Generated or private repository state is tracked: {generated[0].as_posix()}"
        )
    if unexpected_root_files:
        raise ManifestBuildError(
            f"Unexpected tracked root file: {unexpected_root_files[0].as_posix()}"
        )
    _check_docker_context(repository)
    _check_package_metadata(repository, final=final)
    _check_result_markers(repository, final=final)
    _check_result_surface(repository, final=final)
    if final:
        _check_release_delta(repository)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ManifestBuildError("Final release checks require a clean Git tree")


def inspect_distribution_archives(paths: list[str | Path], *, repository_root: str | Path) -> None:
    """Reject restricted or repository-private members from built wheel and sdist archives."""
    candidates = [Path(path) for path in paths]
    archives = [
        path for path in candidates if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    ]
    wheels = [path for path in archives if path.suffix == ".whl"]
    sdists = [path for path in archives if path.name.endswith(".tar.gz")]
    if len(candidates) != 2 or len(wheels) != 1 or len(sdists) != 1 or len(archives) != 2:
        raise ManifestBuildError("Distribution inspection requires one wheel and one sdist")
    expected_package = _expected_package_files(Path(repository_root))
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as stream:
                members = stream.infolist()
                names = _canonical_archive_names(member.filename for member in members)
                metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
                if len(metadata_names) != 1:
                    raise ManifestBuildError("Wheel distribution metadata is missing or ambiguous")
                metadata = stream.read(metadata_names[0]).decode("utf-8")
                if not re.search(r"(?m)^Name: beyondcxr$", metadata) or not re.search(
                    r"(?m)^Version: 1\.0\.0$", metadata
                ):
                    raise ManifestBuildError("Wheel metadata does not declare beyondcxr 1.0.0")
                license_root = "beyondcxr-1.0.0.dist-info/licenses/"
                if not {f"{license_root}LICENSE", f"{license_root}THIRD_PARTY_NOTICES.md"} <= set(
                    names
                ):
                    raise ManifestBuildError("Wheel licensing files are incomplete")
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as stream:
                members = stream.getmembers()
                if any(not (member.isfile() or member.isdir()) for member in members):
                    raise ManifestBuildError("Source distribution contains a special member")
                names = _canonical_archive_names(member.name for member in members)
                root = "beyondcxr-1.0.0"
                if not names or any(
                    name != root and not name.startswith(f"{root}/") for name in names
                ):
                    raise ManifestBuildError("Source distribution has an unexpected top-level root")
                project_file = f"{root}/pyproject.toml"
                if names.count(project_file) != 1:
                    raise ManifestBuildError("Source distribution metadata is missing or ambiguous")
                project_member = next(
                    member
                    for member, name in zip(members, names, strict=True)
                    if name == project_file
                )
                member = stream.extractfile(project_member)
                if member is None:
                    raise ManifestBuildError("Source distribution metadata is unreadable")
                project = tomllib.loads(member.read().decode("utf-8")).get("project", {})
                if project.get("name") != "beyondcxr" or project.get("version") != "1.0.0":
                    raise ManifestBuildError("Source distribution does not declare beyondcxr 1.0.0")
        else:
            raise ManifestBuildError("Unexpected distribution archive type")
        normalized = names
        if archive.suffix == ".whl":
            required_package = {f"beyondcxr/{name}" for name in expected_package}
            if not required_package <= set(normalized):
                raise ManifestBuildError("Wheel does not contain the beyondcxr package")
            if any(
                name not in {"beyondcxr", "beyondcxr-1.0.0.dist-info"}
                and not name.startswith(("beyondcxr/", "beyondcxr-1.0.0.dist-info/"))
                for name in normalized
            ):
                raise ManifestBuildError("Distribution contains an unexpected package namespace")
        else:
            required_package = {
                f"beyondcxr-1.0.0/src/beyondcxr/{name}" for name in expected_package
            }
            if not required_package <= set(normalized):
                raise ManifestBuildError("Source distribution does not contain src/beyondcxr")
        required_sdist = {
            f"beyondcxr-1.0.0/{name}"
            for name in (
                "pyproject.toml",
                "README.md",
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "CITATION.cff",
            )
        }
        if archive.name.endswith(".tar.gz") and not required_sdist <= set(normalized):
            raise ManifestBuildError("Source distribution release files are incomplete")
        if archive.name.endswith(".tar.gz"):
            for member in normalized:
                relative = Path(member).parts[1:]
                if len(relative) >= 2 and relative[0] == "src" and relative[1] != "beyondcxr":
                    raise ManifestBuildError(
                        "Distribution contains an unexpected package namespace"
                    )
        for name in normalized:
            path = Path(name)
            relative_parts = path.parts[1:] if archive.name.endswith(".tar.gz") else path.parts
            if path.suffix.lower() in _RESTRICTED_SUFFIXES or (
                relative_parts
                and relative_parts[0] in {"private", "models", "reports", "outbox", "data"}
            ):
                raise ManifestBuildError(f"Distribution contains restricted member: {name}")


def _expected_package_files(repository_root: Path) -> set[str]:
    root = repository_root.resolve()
    package_root = root / "src" / "beyondcxr"
    expected = {
        path.relative_to(package_root).as_posix()
        for path in _release_candidate_paths(root)
        if path.is_relative_to(package_root)
    }
    if not expected:
        raise ManifestBuildError("Checkout contains no distributable beyondcxr package files")
    return expected


def _canonical_archive_names(names: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    observed: set[str] = set()
    for value in names:
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ManifestBuildError("Distribution contains a malformed member path")
        candidate = value[:-1] if value.endswith("/") else value
        if (
            not candidate
            or candidate.startswith("/")
            or re.match(r"^[A-Za-z]:", candidate)
            or any(part in {"", ".", ".."} for part in candidate.split("/"))
        ):
            raise ManifestBuildError("Distribution contains a malformed member path")
        path = PurePosixPath(candidate)
        normalized = path.as_posix()
        if normalized != candidate:
            raise ManifestBuildError("Distribution contains a malformed member path")
        if normalized in observed:
            raise ManifestBuildError("Distribution contains a duplicate member path")
        observed.add(normalized)
        canonical.append(normalized)
    return canonical


def run_final_acceptance(
    root: str | Path,
    *,
    artifact_root: str | Path,
    authority_root: str | Path,
) -> None:
    """Run full release acceptance in a fresh local clone."""
    repository = Path(root).resolve()
    artifact = Path(artifact_root).resolve()
    authority = Path(authority_root).resolve()
    check_repository(repository, final=True)
    if authority.is_relative_to(repository):
        raise ManifestBuildError("Serving authorities must be published outside the repository")
    authority.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=authority.parent, prefix=".beyondcxr-release-"
    ) as temporary:
        checkout = Path(temporary) / "checkout"
        _run(["git", "clone", "--no-hardlinks", str(repository), str(checkout)], repository)
        commands = (
            ["uv", "sync", "--locked", "--group", "dev", "--extra", "serving"],
            ["make", "lock-check"],
            ["make", "lint"],
            ["make", "format-check"],
            ["make", "test"],
            ["make", "pre-commit"],
            ["uv", "build"],
        )
        for command in commands:
            _run(command, checkout)
        inspect_distribution_archives(
            sorted((checkout / "dist").iterdir()), repository_root=checkout
        )
        _run(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "beyondcxr.release.cli",
                "reproduce",
                "--artifact-root",
                str(artifact),
            ],
            checkout,
        )
        staged_authorities = Path(temporary) / "authorities"
        authority_name = _run_capture(
            [
                "uv",
                "run",
                "--locked",
                "--extra",
                "serving",
                "python",
                "-m",
                "beyondcxr.release.cli",
                "serving-authority",
                "--artifact-root",
                str(artifact),
                "--authority-root",
                str(staged_authorities),
                "--repository-root",
                str(checkout),
            ],
            checkout,
        ).strip()
        candidate_authority = staged_authorities / authority_name
        docker = shutil.which("docker")
        if docker is None:
            raise ManifestBuildError("Final acceptance requires docker")
        image = f"beyondcxr-release-check:{os.getpid()}"
        try:
            _run([docker, "build", "--tag", image, "."], checkout)
            _run(
                [
                    docker,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "/bin/sh",
                    image,
                    "-c",
                    _docker_verification_command(),
                ],
                checkout,
            )
            _run_container_serving_acceptance(
                artifact=artifact,
                authority=candidate_authority,
                image=image,
                cwd=checkout,
                docker=docker,
            )
        finally:
            try:
                subprocess.run(
                    [docker, "image", "rm", "--force", image],
                    cwd=checkout,
                    check=False,
                    capture_output=True,
                )
            except OSError:
                pass
        check_repository(checkout, final=True)
        _publish_accepted_authority(candidate_authority, authority)


def _publish_accepted_authority(candidate: Path, authority_root: Path) -> Path:
    """Atomically install an accepted authority without replacing an existing authority."""
    authority_root.mkdir(parents=True, exist_ok=True)
    final = authority_root / candidate.name
    if final.exists() or final.is_symlink():
        if directory_bytes(final) != directory_bytes(candidate):
            raise ManifestBuildError("Existing serving authority differs from accepted candidate")
        return final
    os.replace(candidate, final)
    return final


def _run(command: list[str], cwd: Path) -> None:
    executable = shutil.which(command[0])
    if executable is None:
        raise ManifestBuildError(f"Final acceptance requires {command[0]}")
    try:
        subprocess.run([executable, *command[1:]], cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise ManifestBuildError(f"Final acceptance command failed: {command[0]}") from exc


def _run_capture(command: list[str], cwd: Path) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        raise ManifestBuildError(f"Final acceptance requires {command[0]}")
    try:
        return subprocess.run(
            [executable, *command[1:]], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ManifestBuildError(f"Final acceptance command failed: {command[0]}") from exc


def _docker_verification_command() -> str:
    """Return the container checks executed by final acceptance."""
    return (
        "test -f /app/LICENSE && "
        "test -f /app/THIRD_PARTY_NOTICES.md && "
        "/app/.venv/bin/python -c 'import beyondcxr' && "
        "test ! -e /app/private && test ! -e /app/models && "
        "test ! -e /app/reports && test ! -e /app/outbox && test ! -e /app/data"
    )


def _container_serving_command(
    *, image: str, authority: Path, package_root: Path, docker: str = "docker"
) -> list[str]:
    """Construct the read-only containerized serving acceptance command."""
    return [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=bind,src={authority},dst=/artifacts/authority,readonly",
        "--mount",
        f"type=bind,src={package_root},dst=/artifacts/packages,readonly",
        "--entrypoint",
        "/app/.venv/bin/python",
        image,
        "-m",
        "beyondcxr.release.container_acceptance",
        "--authority",
        "/artifacts/authority",
        "--package-root",
        "/artifacts/packages",
    ]


def _run_container_serving_acceptance(
    *, artifact: Path, authority: Path, image: str, cwd: Path, docker: str
) -> None:
    """Restore packages and exercise all serving endpoints inside the built image."""
    from beyondcxr.training.preservation import restore_and_validate_export
    from beyondcxr.training.symile_campaign import validate_restored_campaign

    def exercise(restored_root: Path) -> None:
        campaign = validate_restored_campaign(restored_root)
        package_root = campaign.packages[0].directory.parent
        _run(
            _container_serving_command(
                image=image,
                authority=authority,
                package_root=package_root,
                docker=docker,
            ),
            cwd,
        )

    restore_and_validate_export(artifact, restoration_validator=exercise)


def _tracked_text(path: Path) -> str:
    """Decode one bounded tracked UTF-8 text candidate."""
    if path.suffix.lower() in _OPAQUE_BINARY_SUFFIXES:
        raise ManifestBuildError(f"Opaque tracked binary is not allowed: {path.as_posix()}")
    if path.stat().st_size > _MAX_TRACKED_TEXT_BYTES:
        raise ManifestBuildError(f"Tracked text candidate is too large to scan: {path.as_posix()}")
    raw = path.read_bytes()
    if b"\0" in raw:
        raise ManifestBuildError(f"Tracked text candidate contains NUL bytes: {path.as_posix()}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestBuildError(f"Tracked text candidate is not UTF-8: {path.as_posix()}") from exc


def _check_python_source(path: Path, text: str) -> bool:
    """Detect private material and explicit source identifiers in Python source."""
    try:
        tree = ast.parse(text, filename=path.as_posix())
    except SyntaxError as exc:
        raise ManifestBuildError(f"Tracked Python source is invalid: {path.as_posix()}") from exc
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT and _text_contains_source_identifier(token.string):
            return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _PRIVATE_PATH.search(node.value) or _SECRET.search(node.value):
                raise ManifestBuildError(
                    f"Tracked text contains private material: {path.as_posix()}"
                )
            if _text_contains_source_identifier(node.value):
                return True
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.lower() in _SOURCE_IDENTIFIER_FIELDS
                    and isinstance(value, ast.Constant)
                    and _is_source_identifier(key.value.lower(), value.value)
                ):
                    return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, ast.Constant):
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.lower() in _SOURCE_IDENTIFIER_FIELDS
                        and _is_source_identifier(target.id.lower(), node.value.value)
                    ):
                        return True
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                        and target.slice.value.lower() in _SOURCE_IDENTIFIER_FIELDS
                        and _is_source_identifier(target.slice.value.lower(), node.value.value)
                    ):
                        return True
        if (
            isinstance(node, ast.keyword)
            and node.arg is not None
            and node.arg.lower() in _SOURCE_IDENTIFIER_FIELDS
            and isinstance(node.value, ast.Constant)
            and _is_source_identifier(node.arg.lower(), node.value.value)
        ):
            return True
    return False


def _is_source_identifier(field: str, value: object) -> bool:
    """Classify one literal value using the current field-specific source grammar."""
    field = field.lower()
    if field not in _SOURCE_IDENTIFIER_FIELDS:
        return False
    if field == "sample_id" and isinstance(value, str):
        return _SYMILE_SAMPLE_IDENTIFIER.fullmatch(value) is not None or (
            value.startswith("rsna:") and _UUID_IDENTIFIER.fullmatch(value.removeprefix("rsna:"))
        )
    if isinstance(value, str):
        if field in _RSNA_UUID_FIELDS and _UUID_IDENTIFIER.fullmatch(value):
            return True
        if field in _CXR_DICOM_ID_FIELDS and _MIMIC_CXR_DICOM_IDENTIFIER.fullmatch(value):
            return True
        if value.isdecimal():
            value = int(value)
        elif field in _ECG_STUDY_ID_FIELDS and re.fullmatch(r"[0-9]+\.0", value):
            value = int(value[:-2])
    elif field in _ECG_STUDY_ID_FIELDS and isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int:
        return False
    if field in _SUBJECT_ID_FIELDS:
        return 10_000_000 <= value < 20_000_000
    if field in _ADMISSION_ID_FIELDS:
        return 20_000_000 <= value < 30_000_000
    if field in _CXR_STUDY_ID_FIELDS:
        return 50_000_000 <= value < 60_000_000
    if field in _ECG_STUDY_ID_FIELDS:
        return 40_000_000 <= value < 50_000_000
    return False


def _text_contains_source_identifier(text: str) -> bool:
    """Extract textual identifier candidates and apply the current source grammar."""
    for match in _IDENTIFIER_KEY_VALUE.finditer(text):
        value = match.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if _is_source_identifier(match.group("field"), value):
            return True
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9:-]*", text)
    return any(_is_unambiguous_source_identifier(token) for token in tokens)


def _is_unambiguous_source_identifier(value: str, *, include_rsna_uuid: bool = False) -> bool:
    """Recognize source formats that are meaningful without a surrounding field name."""
    return (
        _is_source_identifier("sample_id", value)
        or _is_source_identifier("cxr_dicom_id", value)
        or (include_rsna_uuid and _is_source_identifier("image_id", value))
    )


def _path_contains_source_identifier(path: Path) -> bool:
    """Inspect repository-relative path components for unambiguous source identifiers."""
    for component in path.parts:
        candidates = {component, Path(component).stem}
        if any(
            _is_unambiguous_source_identifier(candidate, include_rsna_uuid=True)
            for candidate in candidates
        ):
            return True
    return False


def _is_generated_repository_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if len(relative.parts) == 1 and relative.name in _GENERATED_ROOT_FILES:
        return True
    if relative.parts[0] in _GENERATED_ROOTS:
        return True
    return (
        len(relative.parts) > 2
        and relative.parts[0] == "data"
        and relative.parts[1] in _GENERATED_DATA_ROOTS
        and path.name != ".gitkeep"
    )


def _release_candidate_paths(root: Path) -> list[Path]:
    """Return tracked-present and untracked-nonignored files in the candidate tree.

    Tracked deletions have no releasable bytes and are omitted. Git-ignored generated state is
    outside this repository-release scan. Staged additions and rename destinations are included.
    """
    try:
        cached = subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        others = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestBuildError("Release checks require a Git checkout") from exc
    try:
        cached_names = {item.decode() for item in cached.stdout.split(b"\0") if item}
        other_names = {item.decode() for item in others.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise ManifestBuildError("Release-candidate path is not UTF-8") from exc
    paths = []
    for name in sorted(cached_names | other_names):
        path = root / name
        if name in cached_names and not path.exists() and not path.is_symlink():
            continue
        paths.append(path)
    for path in paths:
        if path.is_symlink():
            raise ManifestBuildError(f"Release-candidate symlink is not allowed: {path.as_posix()}")
        if not path.is_file():
            raise ManifestBuildError(f"Release-candidate file is unavailable: {path.as_posix()}")
    return paths


def _make_targets(path: Path) -> set[str]:
    targets = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-z][a-z0-9-]*):(?:\s.*)?", line)
        if match:
            targets.add(match.group(1))
    return targets


def _table_contains_source_identifier(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        parsed = _tabular_cells(line)
        if parsed is None:
            continue
        delimiter, columns = parsed
        normalized_columns = [column.lower() for column in columns]
        if not columns or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item) is None for item in columns
        ):
            continue
        identifier_indices = [
            offset
            for offset, column in enumerate(normalized_columns)
            if column in _SOURCE_IDENTIFIER_FIELDS
        ]
        if not identifier_indices:
            continue
        for row in lines[index + 1 :]:
            if not row.strip():
                continue
            candidate = _tabular_cells(row, delimiter=delimiter)
            if candidate is None:
                break
            _, values = candidate
            if (
                delimiter == "|"
                and values
                and all(re.fullmatch(r":?-{3,}:?", value) is not None for value in values)
            ):
                continue
            if len(values) != len(columns):
                break
            if any(
                _is_source_identifier(normalized_columns[offset], values[offset])
                for offset in identifier_indices
            ):
                return True
    return False


def _tabular_cells(line: str, *, delimiter: str | None = None) -> tuple[str, list[str]] | None:
    stripped = line.strip()
    if not stripped:
        return None
    selected = delimiter
    if selected is None:
        if "|" in stripped:
            selected = "|"
        elif "\t" in stripped:
            selected = "\t"
        elif "," in stripped:
            selected = ","
        else:
            selected = " "
    if selected == " ":
        cells = re.split(r"\s+", stripped)
    else:
        cells = next(csv.reader([stripped], delimiter=selected, skipinitialspace=True))
        if selected == "|" and cells and not cells[0].strip():
            cells.pop(0)
        if selected == "|" and cells and not cells[-1].strip():
            cells.pop()
    return selected, [cell.strip() for cell in cells]


def _check_docker_context(root: Path) -> None:
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    if not dockerignore or dockerignore[0] != "*":
        raise ManifestBuildError("Docker context must deny files by default")
    allowed = {line for line in dockerignore if line.startswith("!")}
    if allowed != {
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!LICENSE",
        "!THIRD_PARTY_NOTICES.md",
        "!src/",
        "!src/**",
    }:
        raise ManifestBuildError("Docker context allowlist differs from the serving build contract")


def _check_package_metadata(root: Path, *, final: bool = False) -> None:
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = document.get("project", {})
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    version = project.get("version")
    if (
        project.get("name") != "beyondcxr"
        or not isinstance(version, str)
        or not re.fullmatch(r"\d+\.\d+\.\d+", version)
        or not re.search(rf"(?m)^version:\s*{re.escape(version)}\s*$", citation)
        or not re.search(r"(?m)^title:\s*[\"']?BeyondCXR[\"']?\s*$", citation)
        or not re.search(r"(?m)^type:\s*software\s*$", citation)
        or not re.search(r"(?m)^license:\s*Apache-2\.0\s*$", citation)
        or not re.search(
            r"(?m)^repository-code:\s*[\"\']?https://github\.com/Youssef-SH/BeyondCXR[\"\']?\s*$",
            citation,
        )
        or project.get("license") != "Apache-2.0"
        or project.get("license-files") != ["LICENSE", "THIRD_PARTY_NOTICES.md"]
        or not (root / "LICENSE").is_file()
        or not (root / "THIRD_PARTY_NOTICES.md").is_file()
        or not (root / "CITATION.cff").is_file()
    ):
        raise ManifestBuildError("Package metadata differs from the release contract")
    if final and version != "1.0.0":
        raise ManifestBuildError("Final release metadata must declare version 1.0.0")


def _check_result_markers(root: Path, *, final: bool) -> None:
    for document in (root / "README.md", root / "docs" / "model_card.md"):
        text = document.read_text(encoding="utf-8")
        parse_result_region(text)
        if final and "Awaiting formal Symile execution" in text:
            raise ManifestBuildError(f"Unresolved result placeholder in {document.as_posix()}")


def _check_result_surface(root: Path, *, final: bool) -> None:
    results = root / "results"
    if results.is_symlink() or any(path.is_symlink() for path in results.rglob("*")):
        raise ManifestBuildError("Public result tree must not contain symlinks")
    binding = results / "symile" / "binding.json"
    if not binding.exists():
        if final:
            raise ManifestBuildError("Final release is missing results/symile/binding.json")
        observed = {
            path.relative_to(results).as_posix() for path in results.rglob("*") if path.is_file()
        }
        if observed != {"README.md"}:
            raise ManifestBuildError("Pre-binding results must contain exactly README.md")
        return
    load_result_binding(binding)
    expected = {"README.md", "symile/binding.json"}
    expected.update(f"symile/{name}" for name in public_result_surface())
    observed = {
        path.relative_to(results).as_posix() for path in results.rglob("*") if path.is_file()
    }
    if observed != expected:
        raise ManifestBuildError("Public result file membership is incomplete or unexpected")


def _check_release_delta(root: Path) -> None:
    """Validate the permitted post-execution release delta."""
    binding = load_result_binding(root / "results" / "symile" / "binding.json")
    science = binding.science_execution.git_commit
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{science}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", science, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        changed_output = subprocess.run(
            ["git", "diff", "--name-status", "-z", "--diff-filter=ACDMRT", f"{science}..HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestBuildError(
            "Science-execution commit is unavailable or not an ancestor"
        ) from exc

    bounded = {"README.md", "docs/model_card.md"}
    metadata = {"pyproject.toml", "CITATION.cff", "uv.lock"}
    _check_changelog_finalization(root, science)
    for relative in _changed_paths(changed_output):
        if relative.startswith("results/symile/"):
            continue
        if relative == "CHANGELOG.md":
            continue
        if relative in bounded:
            _check_bounded_release_document(root, science, relative)
            continue
        if relative in metadata:
            _check_version_only_delta(root, science, relative)
            continue
        raise ManifestBuildError(f"Unauthorized post-science change: {relative}")


def _changed_paths(output: bytes) -> tuple[str, ...]:
    fields = output.rstrip(b"\0").split(b"\0") if output else []
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode()
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if index + count > len(fields):
            raise ManifestBuildError("Git release delta is malformed")
        paths.extend(field.decode() for field in fields[index : index + count])
        index += count
    return tuple(paths)


def _check_changelog_finalization(root: Path, science: str) -> None:
    before = _git_file(root, science, "CHANGELOG.md")
    after = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^## Unreleased$")
    if len(pattern.findall(before)) != 1:
        raise ManifestBuildError("Science-commit changelog has no unique Unreleased heading")
    heading = re.compile(r"(?m)^## 1\.0\.0 — (\d{4}-\d{2}-\d{2})$")
    match = heading.search(after)
    if match is None or pattern.sub(match.group(0), before, count=1) != after:
        raise ManifestBuildError("Post-science changelog change is not heading-only")
    try:
        date.fromisoformat(match.group(1))
    except ValueError as exc:
        raise ManifestBuildError("Post-science changelog release date is invalid") from exc


def _git_file(root: Path, revision: str, relative: str) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestBuildError(
            f"Required science-commit file is unavailable: {relative}"
        ) from exc


def _result_shell(text: str) -> str:
    before, _, after = parse_result_region(text)
    return f"{before}{RESULT_START}\n<generated-result-region>\n{RESULT_END}{after}"


def _check_bounded_release_document(root: Path, science: str, relative: str) -> None:
    before = _git_file(root, science, relative)
    after = (root / relative).read_text(encoding="utf-8")
    if _result_shell(before) != _result_shell(after):
        raise ManifestBuildError(f"Unauthorized prose change outside result region: {relative}")


def _check_version_only_delta(root: Path, science: str, relative: str) -> None:
    before = _git_file(root, science, relative)
    after = (root / relative).read_text(encoding="utf-8")
    if relative == "pyproject.toml":
        pattern = re.compile(r'(?m)^(version\s*=\s*)"([^"]+)"\s*$')
    elif relative == "CITATION.cff":
        pattern = re.compile(r"(?m)^(version:\s*)([^\s]+)\s*$")
    else:
        pattern = re.compile(r'(?ms)(\[\[package\]\]\nname = "beyondcxr"\nversion = ")([^"]+)(")')
    before_match = pattern.search(before)
    after_match = pattern.search(after)
    if (
        before_match is None
        or after_match is None
        or before_match.group(2) != "0.1.0"
        or after_match.group(2) != "1.0.0"
        or pattern.sub(
            r"\g<1><release-version>\g<3>" if relative == "uv.lock" else r"\g<1><release-version>",
            before,
            count=1,
        )
        != pattern.sub(
            r"\g<1><release-version>\g<3>" if relative == "uv.lock" else r"\g<1><release-version>",
            after,
            count=1,
        )
    ):
        raise ManifestBuildError(f"Post-science metadata change is not version-only: {relative}")
