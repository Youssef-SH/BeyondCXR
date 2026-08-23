"""Define lifecycle-specific execution policy for neural DataLoaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

LoaderLifecycle = Literal["reused", "one_shot"]


@dataclass(frozen=True)
class LoaderExecutionPolicy:
    """Operational DataLoader settings excluded from scientific identity."""

    lifecycle: LoaderLifecycle
    num_workers: int
    pin_memory: bool

    def __post_init__(self) -> None:
        if (
            self.lifecycle not in ("reused", "one_shot")
            or isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
            or not isinstance(self.pin_memory, bool)
            or (self.lifecycle == "one_shot" and self.num_workers != 0)
        ):
            raise ValueError("DataLoader execution policy fields are invalid")

    @property
    def persistent_workers(self) -> bool:
        """Return whether worker processes persist across loader iterations."""
        return self.lifecycle == "reused" and self.num_workers > 0

    @property
    def prefetch_factor(self) -> int | None:
        """Return the fixed multiprocessing prefetch factor when applicable."""
        return 2 if self.num_workers > 0 else None

    def provenance(self) -> dict[str, object]:
        """Return compact actual loader execution provenance."""
        return {
            "lifecycle": self.lifecycle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_provenance(cls, value: object) -> LoaderExecutionPolicy:
        """Validate and reconstruct serialized loader execution provenance."""
        fields = {"lifecycle", "num_workers", "pin_memory"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("DataLoader execution provenance has an unexpected field set")
        try:
            result = cls(
                lifecycle=cast(LoaderLifecycle, value["lifecycle"]),
                num_workers=cast(int, value["num_workers"]),
                pin_memory=cast(bool, value["pin_memory"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("DataLoader execution provenance is invalid") from exc
        return result


def reused_loader_policy(*, num_workers: int, pin_memory: bool) -> LoaderExecutionPolicy:
    """Return the policy for train/validation loaders reused across epochs."""
    return LoaderExecutionPolicy("reused", num_workers, pin_memory)


def one_shot_loader_policy(*, pin_memory: bool) -> LoaderExecutionPolicy:
    """Return the synchronous policy for a single evaluation pass."""
    return LoaderExecutionPolicy("one_shot", 0, pin_memory)
