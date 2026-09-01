from pathlib import Path

import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.utils.validation_context import (
    load_authenticated_file,
    memoized_directory_validation,
    operation_scoped_validation,
    validation_operation,
)


def test_operation_scoped_reuse_rejects_changed_underlying_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")

    @operation_scoped_validation
    def validate() -> None:
        assert load_authenticated_file("test", artifact, Path.read_bytes) == b"original"
        artifact.write_bytes(b"replaced")
        assert load_authenticated_file("test", artifact, Path.read_bytes) == b"original"

    with pytest.raises(ManifestBuildError, match="changed during the operation"):
        validate()


def test_loaded_values_do_not_escape_the_operation_scope(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")

    @operation_scoped_validation
    def validate() -> object:
        return load_authenticated_file("test", artifact, lambda _path: object())

    assert validate() is not validate()


def test_unknown_keyword_owner_does_not_inherit_nested_scope(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")

    @operation_scoped_validation
    def validate_inner(*, root: Path) -> object:
        del root
        return load_authenticated_file("test", artifact, lambda _path: object())

    @operation_scoped_validation
    def validate_outer(*, root: Path) -> tuple[object, object]:
        first = load_authenticated_file("test", artifact, lambda _path: object())
        return first, validate_inner(root=root / "nested")

    outer, nested = validate_outer(root=tmp_path / "outer")
    assert outer is not nested


def test_reused_parent_rejects_changed_transitive_dependency(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "manifest.json").write_bytes(b"parent")
    child_manifest = child / "manifest.json"
    child_manifest.write_bytes(b"child")

    @memoized_directory_validation("test-child")
    def validate_child(directory: Path) -> bytes:
        return (directory / "manifest.json").read_bytes()

    @memoized_directory_validation("test-parent")
    def validate_parent(directory: Path) -> bytes:
        assert (directory / "manifest.json").read_bytes() == b"parent"
        return validate_child(child)

    @operation_scoped_validation
    def validate() -> None:
        assert validate_parent(parent) == b"child"
        child_manifest.write_bytes(b"other")
        assert validate_parent(parent) == b"child"

    with pytest.raises(ManifestBuildError, match="changed during the operation"):
        validate()


def test_reused_directory_rejects_membership_change(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_bytes(b"content")

    @memoized_directory_validation("test-membership")
    def validate_artifact(directory: Path) -> bytes:
        return (directory / "manifest.json").read_bytes()

    @operation_scoped_validation
    def validate() -> None:
        assert validate_artifact(artifact) == b"content"
        (artifact / "unexpected.bin").write_bytes(b"unexpected")
        assert validate_artifact(artifact) == b"content"

    with pytest.raises(ManifestBuildError, match="changed during the operation"):
        validate()


def test_explicit_validation_lifetime_is_root_bound_and_revoked(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    external = tmp_path / "external"
    owned.mkdir()
    external.mkdir()
    (owned / "manifest.json").write_bytes(b"owned")
    (external / "manifest.json").write_bytes(b"external")

    @memoized_directory_validation("test-explicit-owner")
    def validate(directory: Path) -> object:
        (directory / "manifest.json").read_bytes()
        return object()

    with validation_operation(owned):
        retained = validate(owned)
        assert validate(owned) is retained
        assert validate(external) is not validate(external)
    assert validate(owned) is not retained


def test_explicit_validation_lifetime_detects_late_owned_mutation(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    manifest = owned / "manifest.json"
    manifest.write_bytes(b"owned")

    @memoized_directory_validation("test-explicit-mutation")
    def validate(directory: Path) -> bytes:
        return (directory / "manifest.json").read_bytes()

    with pytest.raises(ManifestBuildError, match="changed during the operation"):
        with validation_operation(owned):
            assert validate(owned) == b"owned"
            manifest.write_bytes(b"changed")
