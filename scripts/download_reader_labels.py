"""Download open multi-reader CXR label sets for grounding doctor personas.

Two sources, both WITHOUT a Google-Form / PhysioNet gate — so this runs unattended:

1. NIH ChestX-ray14 radiologist-**adjudicated** labels (Majkowska et al. 2020, 4
   findings) — shipped inside ``torchxrayvision`` as
   ``google2019_nih-chest-xray-labels.csv.gz`` (the adjudicated CONSENSUS ground
   truth over 4,376 images). Exported here to ``data/nih_rad_labels/``.

   NOTE: the per-reader ``individual_readers.csv`` (needed to compute per-radiologist
   sensitivity/specificity = ``FixedAccuracyDoctor`` persona parameters) is NOT
   bundled and is Google-Form-gated: https://forms.gle/bmYPtLy4qsJahVKFA — drop it
   into ``data/nih_rad_labels/`` once obtained.

2. CheXpert test-set **per-reader** labels (github.com/rajpurkarlab/cheXpert-test-set-labels)
   — the 5-vote-majority ``groundtruth.csv`` plus every individual radiologist read
   under ``radiologists/`` (open on GitHub). Gives per-reader + consensus immediately;
   error = a benchmark read disagreeing with the majority GT. (Images themselves need
   Stanford AIMI registration — only the labels are fetched here.)

Other shortlisted sets (VinDr-CXR, REFLACX) are PhysioNet-credentialed (DUA + CITI
training) and cannot be auto-downloaded — obtain them manually.

Run:  uv run python scripts/download_reader_labels.py
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

NIH_OUT = Path("data/nih_rad_labels")
CHEX_OUT = Path("data/chexpert_test_labels")
CHEX_REPO = "rajpurkarlab/cheXpert-test-set-labels"
FORM_URL = "https://forms.gle/bmYPtLy4qsJahVKFA"
_UA = {"User-Agent": "hai_sim-reader-labels"}


def export_nih_adjudicated() -> None:
    """Export the xrv-bundled NIH Google adjudicated consensus labels."""
    import torchxrayvision as xrv

    src = Path(os.path.dirname(xrv.__file__)) / "data" / "google2019_nih-chest-xray-labels.csv.gz"
    if not src.exists():
        print(f"[nih]  SKIP: bundled labels not found at {src}")
        return
    NIH_OUT.mkdir(parents=True, exist_ok=True)
    out = NIH_OUT / "nih_google_adjudicated.csv"
    with gzip.open(src, "rt", encoding="utf-8") as fi:
        data = fi.read()
    out.write_text(data, encoding="utf-8")
    n_rows = max(data.count("\n") - 1, 0)
    (NIH_OUT / "README.txt").write_text(
        "nih_google_adjudicated.csv = Majkowska et al. 2020 adjudicated CONSENSUS labels\n"
        "(4 findings: Fracture, Pneumothorax, Airspace opacity, Nodule or mass), bundled in\n"
        "torchxrayvision. This is the reference-standard ground truth, NOT per-reader.\n\n"
        f"Per-reader reads (individual_readers.csv) are Google-Form-gated:\n  {FORM_URL}\n"
        "Complete the form, then place individual_readers.csv in this folder.\n",
        encoding="utf-8",
    )
    print(f"[nih]  adjudicated GT -> {out}  ({n_rows} rows, 4 findings)")


def _gh_contents(path: str) -> list[dict]:
    url = f"https://api.github.com/repos/{CHEX_REPO}/contents/{path}"
    req = urllib.request.Request(url, headers={**_UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return list(json.load(r))


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        dst.write_bytes(r.read())


def _walk_csv(path: str) -> int:
    """Recursively download every .csv under a repo path, preserving structure."""
    count = 0
    for item in _gh_contents(path):
        if item["type"] == "dir":
            count += _walk_csv(item["path"])
        elif item["type"] == "file" and item["name"].endswith(".csv"):
            _download(item["download_url"], CHEX_OUT / item["path"])
            print(f"[chex] {item['path']}")
            count += 1
    return count


def download_chexpert_labels() -> int:
    """Grab the majority groundtruth + all per-reader reads (open, no gate)."""
    count = 0
    for item in _gh_contents(""):
        if item["type"] == "file" and item["name"] == "groundtruth.csv":
            _download(item["download_url"], CHEX_OUT / item["name"])
            print(f"[chex] {item['name']}")
            count += 1
    count += _walk_csv("radiologists")
    return count


if __name__ == "__main__":
    export_nih_adjudicated()
    n = download_chexpert_labels()
    print(f"[chex] {n} label CSVs -> {CHEX_OUT}/")
    print("\nManual follow-ups:")
    print(f"  - NIH per-reader individual_readers.csv: {FORM_URL} -> {NIH_OUT}/")
    print("  - VinDr-CXR / REFLACX: PhysioNet credentialed (DUA + CITI).")
