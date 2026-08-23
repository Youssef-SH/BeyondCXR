"""Generate an obviously artificial JPEG and complete laboratory JSON example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from beyondcxr.serving.authority import LAB_KEYS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic-serving-request"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, columns = np.indices((360, 480))
    pattern = (((rows // 24) + (columns // 24)) % 2 * 255).astype(np.uint8)
    Image.fromarray(pattern, mode="L").save(
        args.output / "synthetic-checkerboard.jpg",
        format="JPEG",
        quality=90,
    )
    labs = {key: (None if index % 11 == 0 else float(index)) for index, key in enumerate(LAB_KEYS)}
    (args.output / "synthetic-labs.json").write_text(
        json.dumps(labs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Generated explicitly synthetic schema examples; they are not clinical data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
