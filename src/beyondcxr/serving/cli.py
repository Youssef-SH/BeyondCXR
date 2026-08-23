"""Run the controlled Symile research-serving API."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from beyondcxr.serving.api import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    application = create_app(
        authority_path=args.authority,
        package_root=args.package_root,
        device=args.device,
    )
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
