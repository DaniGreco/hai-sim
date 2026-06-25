"""Download the NIH ChestX-ray14 *sample* (~5.6k images) via kagglehub.

Prereq — a Kaggle API token (one-time, ~1 min):
  1. https://www.kaggle.com/settings  ->  "Create New Token"  (downloads kaggle.json)
  2. put it at  %USERPROFILE%\.kaggle\kaggle.json
     (or export KAGGLE_USERNAME / KAGGLE_KEY)

Then:
  uv run python scripts/download_nih_sample.py

It prints the imgpath / csvpath to paste into the scenario YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        print('kagglehub missing — run: uv pip install -e ".[models]"', file=sys.stderr)
        return 1

    print("downloading nih-chest-xrays/sample (~2 GB) ...")
    root = Path(kagglehub.dataset_download("nih-chest-xrays/sample"))
    print(f"\ndownloaded to: {root}")

    # The sample ships images under sample/images and labels in sample_labels.csv.
    imgpath = next((p for p in [root / "sample" / "images", root / "images"] if p.is_dir()), None)
    csvpath = next(
        (p for p in [root / "sample_labels.csv", root / "sample" / "sample_labels.csv"] if p.is_file()),
        None,
    )
    print("\nPaste these into the scenario YAML (patient_generator):")
    print(f"  imgpath: {imgpath if imgpath else '<images dir under ' + str(root) + '>'}")
    print(f"  csvpath: {csvpath if csvpath else '<sample_labels.csv under ' + str(root) + '>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
