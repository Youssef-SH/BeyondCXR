"""Operation-scoped reuse for validation of authenticated artifact bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import cast

from beyondcxr.data.errors import ManifestBuildError


@dataclass
class _ValidationState:
    owner: Path | None
    loaded: dict[tuple[str, Path, str], object] = field(default_factory=dict)
    validated: dict[tuple[str, Path, str], tuple[dict[Path, str], object]] = field(
        default_factory=dict
    )
    collectors: list[set[Path]] = field(default_factory=list)
    baseline: dict[Path, str] = field(default_factory=dict)
    directories: dict[Path, frozenset[Path]] = field(default_factory=dict)


_STATE: ContextVar[_ValidationState | None] = ContextVar("beyondcxr_validation_state", default=None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register(state: _ValidationState, paths: set[Path], hashes: Mapping[Path, str]) -> None:
    for collector in state.collectors:
        collector.update(paths)
    for path in paths:
        state.baseline.setdefault(path, hashes[path])


def operation_scoped_validation[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    """Share validation only inside one call tree and revoke it on return."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        owner = Path(args[0]).resolve() if args and isinstance(args[0], str | Path) else None
        current = _STATE.get()
        if current is not None and owner is not None and current.owner == owner:
            return function(*args, **kwargs)
        token = _STATE.set(_ValidationState(owner=owner))
        try:
            result = function(*args, **kwargs)
            state = _STATE.get()
            try:
                changed = state is None or any(
                    frozenset(path.resolve() for path in root.iterdir()) != members
                    for root, members in state.directories.items()
                )
                changed = (
                    changed
                    or state is None
                    or any(_sha256(path) != digest for path, digest in state.baseline.items())
                )
            except OSError:
                changed = True
            if changed:
                raise ManifestBuildError("Validated artifact bytes changed during the operation")
            return result
        finally:
            _STATE.reset(token)

    return wrapped


def load_authenticated_file[T](namespace: str, path: str | Path, loader: Callable[[Path], T]) -> T:
    """Load once per exact file content inside an active validation operation."""

    source = Path(path).resolve()
    state = _STATE.get()
    if state is None:
        return loader(source)
    digest = state.baseline.get(source)
    if digest is None:
        digest = _sha256(source)
    _register(state, {source}, {source: digest})
    key = (namespace, source, digest)
    if key not in state.loaded:
        state.loaded[key] = loader(source)
    return cast(T, state.loaded[key])


def memoized_directory_validation(namespace: str, *, resolver: Callable[..., Path] | None = None):
    """Reuse validation while all direct and transitive dependency bytes match."""

    def decorate[**P, T](function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            state = _STATE.get()
            if state is None or not args:
                return function(*args, **kwargs)
            root = resolver(*args, **kwargs) if resolver is not None else Path(args[0])
            try:
                members = sorted(root.iterdir(), key=lambda item: item.name)
                if (
                    root.is_symlink()
                    or not root.is_dir()
                    or any(item.is_symlink() or not item.is_file() for item in members)
                ):
                    return function(*args, **kwargs)
                direct = {member.resolve() for member in members}
            except OSError:
                return function(*args, **kwargs)
            options = _validation_options(args[1:], kwargs)
            resolved_root = root.resolve()
            state.directories.setdefault(resolved_root, frozenset(direct))
            key = (f"{namespace}:{options}", resolved_root, "validation")
            cached = state.validated.get(key)
            if cached is not None:
                snapshot, value = cached
                _register(state, set(snapshot), snapshot)
                return cast(T, value)

            try:
                direct_hashes = {path: _sha256(path) for path in direct}
            except OSError:
                return function(*args, **kwargs)
            _register(state, direct, direct_hashes)
            dependencies = set(direct)
            state.collectors.append(dependencies)
            try:
                value = function(*args, **kwargs)
            finally:
                state.collectors.pop()
            snapshot = {path: state.baseline[path] for path in dependencies}
            state.validated[key] = (snapshot, value)
            _register(state, dependencies, snapshot)
            return value

        return wrapped

    return decorate


def _validation_options(args: tuple[object, ...], kwargs: Mapping[str, object]) -> str:
    """Include every relationship argument in a process-local cache coordinate."""

    def coordinate(value: object) -> object:
        if isinstance(value, Path):
            return ("path", value.resolve())
        if isinstance(value, (str, int, bool, type(None))):
            return value
        if isinstance(value, (tuple, list)):
            return tuple(coordinate(item) for item in value)
        for attribute in ("package_id", "decision_id", "development_id", "prediction_id"):
            identity = getattr(value, attribute, None)
            if isinstance(identity, str):
                return (type(value).__qualname__, attribute, identity, id(value))
        return (type(value).__qualname__, id(value))

    return repr(
        (
            tuple(coordinate(value) for value in args),
            tuple((key, coordinate(value)) for key, value in sorted(kwargs.items())),
        )
    )
