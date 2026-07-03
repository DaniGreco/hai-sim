"""Download the ChestX-Det dataset (images + annotations) into data/chestxdet/.

ChestX-Det = ~3.5k NIH ChestX-ray14 images re-annotated by three board-certified
radiologists with polygons + boxes for 13 findings (Apache-2.0). It is the expert
dual ground truth (labels + segmentation) for the localization-validation harness.

From the repo root:
  uv run python scripts/download_chestxdet.py            # test split only (~212 MB)
  uv run python scripts/download_chestxdet.py --train    # also the train split (~850 MB)

Idempotent: skips any file already downloaded and any split already extracted.
Prints the image directory to point ChestXDetDataset / the validation scenario at.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "chestxdet"

RAW = "https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/master"
IMG = "http://resource.deepwise.com/ChestX-Det"

ANNOTATIONS = {
    "ChestX_Det_test.json": f"{RAW}/ChestX_Det_test.json",
    "ChestX_Det_train.json": f"{RAW}/ChestX_Det_train.json",
}
SPLITS = {
    "test": f"{IMG}/test_data.zip",
    "train": f"{IMG}/train_data.zip",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name} <- {url}")
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)  # trusted dataset host
    tmp.replace(dest)


def _extract(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEST)


def _png_dirs() -> list[Path]:
    dirs = {p.parent for p in DEST.rglob("*.png")}
    return sorted(dirs)


def main() -> int:
    parser = argparse.ArgumentParser(description="download ChestX-Det into data/chestxdet/")
    parser.add_argument("--train", action="store_true", help="also fetch the ~850 MB train split")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    print("annotations:")
    for fname, url in ANNOTATIONS.items():
        _download(url, DEST / fname)

    wanted = ["test"] + (["train"] if args.train else [])
    for split in wanted:
        zip_path = DEST / f"{split}_data.zip"
        print(f"{split} images:")
        _download(SPLITS[split], zip_path)
        # Heuristic: if we already have many PNGs, assume this split is extracted.
        if len(_png_dirs()) == 0 or not any(DEST.rglob("*.png")):
            print(f"  extracting {zip_path.name} ...")
            _extract(zip_path)
        else:
            print("  images already extracted")

    png_dirs = _png_dirs()
    if not png_dirs:
        print("no PNGs found after extraction — check the archive layout", file=sys.stderr)
        return 1
    print("\ndone. image directory(ies):")
    for d in png_dirs:
        print(f"  {d}  ({sum(1 for _ in d.glob('*.png'))} images)")
    print("point ChestXDetDataset(image_dir=...) / the validation scenario at the right one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
