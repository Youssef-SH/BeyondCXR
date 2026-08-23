"""Errors shared by dataset construction and artifact validation."""


class ManifestBuildError(ValueError):
    """Raised when source data or generated artifacts violate their contracts."""
