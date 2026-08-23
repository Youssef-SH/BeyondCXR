"""Inspect a DICOM image and its selected metadata."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from beyondcxr.data.dicom_loader import read_dicom, record_as_dict


def main() -> None:
    """Load and display the DICOM image supplied on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/inspect_dicom.py IMAGE.dcm")

    path = Path(sys.argv[1])
    pixels, record = read_dicom(path)

    print(record_as_dict(record))

    plt.imshow(pixels, cmap="gray")
    plt.axis("off")
    plt.title(path.name)
    plt.show()


if __name__ == "__main__":
    main()
