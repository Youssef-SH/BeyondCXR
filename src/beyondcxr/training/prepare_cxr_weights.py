"""Materialize and report the exact standard CXR initialization weights."""

from __future__ import annotations

import json

from beyondcxr.models.cxr_baseline import ensure_pretrained_weights


def main() -> int:
    identity = ensure_pretrained_weights()
    print(json.dumps(identity.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
