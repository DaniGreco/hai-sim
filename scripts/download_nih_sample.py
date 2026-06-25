"""Download the NIH ChestX-ray14 *sample* (~5.6k images) into data/nih_sample/.

Prereq — a Kaggle API token (one-time, ~1 min):
  - new-style token: set the KAGGLE_API_TOKEN env var to your `KGAT_...` token, or
  - classic token: put kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json
    (or export KAGGLE_USERNAME / KAGGLE_KEY)

Then, from the repo root:
  uv run python scripts/download_nih_sample.py

Idempotent: if data/nih_sample/images is already populated it does nothing. The
scenarios in configs/examples/v1_nih_*.yaml point at data/nih_sample/.

Tip (Windows): export PYTHONUTF8=1 first so the download progress bar does not
crash the cp1252 console.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "nih_sample"


def _find(root: Path, candidates: list[Path]) -> Path | None:
    return next((p for p in candidates if p.exists()), None)


def main() -> int:
    images_dest = DEST / "images"
    if images_dest.is_dir() and any(images_dest.iterdir()):
        print(f"already present: {images_dest} — nothing to do")
        return 0

    try:
        import kagglehub
    except ImportError:
        print('kagglehub missing — run: uv pip install -e ".[models]"', file=sys.stderr)
        return 1

    print("downloading nih-chest-xrays/sample (~2 GB) ...")
    root = Path(kagglehub.dataset_download("nih-chest-xrays/sample"))
    print(f"downloaded to cache: {root}")

    src_images = _find(root, [root / "sample" / "images", root / "images"])
    src_csv = _find(root, [root / "sample_labels.csv", root / "sample" / "sample_labels.csv"])
    if src_images is None or src_csv is None:
        print(f"unexpected dataset layout under {root}", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"copying images -> {images_dest}")
    shutil.copytree(src_images, images_dest, dirs_exist_ok=True)
    shutil.copy2(src_csv, DEST / "sample_labels.csv")

    print(f"\ndone. data is at: {DEST}")
    print("the scenarios already use this path; you can delete the kagglehub cache")
    print(f"  ({root}) to reclaim space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
