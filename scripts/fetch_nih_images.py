"""Download NIH ChestX-ray14 and place the 30,805 images the NIH caches were built on.

Only needed to build new cache cells or to run a scenario that reads the pixels: the
studies replay ``data/cache`` and never open an image.

    uv run --with kagglehub python scripts/fetch_nih_images.py           # ~42 GB download
    uv run --with kagglehub python scripts/fetch_nih_images.py --link    # hard links, no copy

The images come from the Kaggle dataset ``nih-chest-xrays/data`` (version 3, all 112,120
images); a Kaggle API token is needed (``~/.kaggle/kaggle.json`` or ``KAGGLE_API_TOKEN``).
The script writes ``data/nih_full/Data_Entry_2017.csv`` and ``data/nih_full/images/`` and,
unless ``--no-check``, verifies every file against ``data/provenance/nih_sha256sums.txt``.
``--source-root`` uses an already extracted copy instead of downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

CSV_NAME = "Data_Entry_2017.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # another file system: fall back to a copy
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=Path, default=Path("data/provenance/nih_image_ids.txt"))
    ap.add_argument("--sums", type=Path, default=Path("data/provenance/nih_sha256sums.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/nih_full"))
    ap.add_argument("--source-root", type=Path, default=None, help="an already extracted dataset root")
    ap.add_argument("--link", action="store_true", help="hard-link instead of copying")
    ap.add_argument("--no-check", action="store_true", help="skip the SHA-256 check")
    args = ap.parse_args()

    wanted = [ln.strip() for ln in args.ids.read_text(encoding="utf-8").splitlines() if ln.strip()]
    root = args.source_root
    if root is None:
        try:
            import kagglehub
        except ImportError:
            sys.exit("kagglehub is missing: run with `uv run --with kagglehub ...` or pass --source-root")
        print("downloading nih-chest-xrays/data (~42 GB) with kagglehub ...", flush=True)
        root = Path(kagglehub.dataset_download("nih-chest-xrays/data"))
    root = root.resolve()

    csv_hits = list(root.rglob(CSV_NAME))
    if not csv_hits:
        sys.exit(f"{CSV_NAME} not found under {root}")
    index = {p.name: p for p in root.rglob("*.png")}  # images_001/images ... images_012/images
    missing = [n for n in wanted if n not in index]
    if missing:
        sys.exit(f"{len(missing)} images of the pool are missing under {root}, e.g. {missing[:3]}")

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_hits[0], args.out / CSV_NAME)
    for i, name in enumerate(wanted, 1):
        place(index[name], images / name, args.link)
        if i % 5000 == 0 or i == len(wanted):
            print(f"  placed {i}/{len(wanted)}", flush=True)

    if not args.no_check:
        print("checking SHA-256 ...", flush=True)
        base = args.out.parent  # the sums are relative to data/
        bad = []
        for line in args.sums.read_text(encoding="utf-8").splitlines():
            digest, rel = line.split(maxsplit=1)
            if sha256(base / rel) != digest:
                bad.append(rel)
        if bad:
            sys.exit(f"{len(bad)} files differ from the shipped checksums, e.g. {bad[:3]}")
        print("all files match", flush=True)
    print(f"done: {len(wanted)} images in {images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
