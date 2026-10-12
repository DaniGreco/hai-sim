"""Turn the raw VinDr / VinBigData ``train.csv`` (per-radiologist boxes) into the tidy
tables the framework consumes.

Reads one row per box (columns ``image_id, class_name, rad_id, ...``) and writes, in the
raw 14-finding VinDr vocabulary:

  labels.csv      ``image_id`` + one column per VinDr finding, presence at ``k``-of-N readers
  difficulty.csv  ``example_id`` + inter-reader disagreement columns (the delta of thesis §5.7)
  readers.csv     ``image_id, rad_id`` + per-finding presence for each reader (used later by
                  reader-calibrated personas / a ``reader_panel`` source)

The mapping from these VinDr findings into the model's ``xrv`` schema is deliberately NOT
applied here — it is a runtime configuration line (``label_sources[].map: vindr14->xrv``),
so the raw vocabulary stays inspectable and the alignment shows up in a scenario's diff.
Presence at ``k`` is a modelling choice, hence a flag: regenerate with a different ``k``
rather than baking one threshold in.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

from hai_sim.core.types import LabelSchema

VINDR = LabelSchema.vindr().labels  # 14 findings, class-id order


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read train.csv, transparently unwrapping the Kaggle single-file ZIP."""
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".csv"))
            text = zf.read(member).decode("utf-8")
    else:
        text = raw.decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _delta(count: int, n_readers: int) -> float:
    """Normalised inter-reader disagreement 4p(1-p): 0 unanimous, 1 at an even split."""
    p = count / n_readers if n_readers else 0.0
    return 4.0 * p * (1.0 - p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=Path.home()
        / ".cache/kagglehub/competitions"
        / "vinbigdata-chest-xray-abnormalities-detection/train.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/vindr"))
    ap.add_argument(
        "--k",
        type=int,
        default=2,
        help="presence threshold: a finding is present if >= k of the readers boxed it",
    )
    args = ap.parse_args()

    rows = read_rows(args.train_csv)
    readers_of: dict[str, set[str]] = defaultdict(set)  # image -> {rad_id}
    boxed: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        iid, rad, cls = r["image_id"], r["rad_id"], r["class_name"]
        readers_of[iid].add(rad)
        if cls in VINDR:  # "No finding" only identifies a reader; it is not a finding
            boxed[iid][cls].add(rad)

    images = sorted(readers_of)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def n_boxed(iid: str, finding: str) -> int:
        return len(boxed[iid].get(finding, ()))

    labels_path = args.out_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", *VINDR])
        for iid in images:
            w.writerow([iid, *(1 if n_boxed(iid, f) >= args.k else 0 for f in VINDR)])

    diff_path = args.out_dir / "difficulty.csv"
    with diff_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "example_id",
                "mean_delta_all",
                "mean_delta_active",
                "frac_contested",
                "max_delta",
                "n_contested",
                "n_readers",
            ]
        )
        for iid in images:
            n = len(readers_of[iid])
            deltas = [_delta(n_boxed(iid, f), n) for f in VINDR]
            active = [d for f, d in zip(VINDR, deltas, strict=True) if n_boxed(iid, f) > 0]
            n_contested = sum(1 for f in VINDR if 0 < n_boxed(iid, f) < n)
            mean_all = sum(deltas) / len(deltas)
            mean_active = sum(active) / len(active) if active else 0.0
            w.writerow(
                [
                    iid,
                    f"{mean_all:.6f}",
                    f"{mean_active:.6f}",
                    f"{n_contested / len(VINDR):.6f}",
                    f"{max(deltas):.6f}",
                    n_contested,
                    n,
                ]
            )

    readers_path = args.out_dir / "readers.csv"
    with readers_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "rad_id", *VINDR])
        for iid in images:
            for rad in sorted(readers_of[iid]):
                w.writerow([iid, rad, *(1 if rad in boxed[iid].get(f, ()) else 0 for f in VINDR)])

    print(f"images: {len(images)} | readers/image (min..max): "
          f"{min(len(v) for v in readers_of.values())}.."
          f"{max(len(v) for v in readers_of.values())}")
    print(f"wrote {labels_path} (presence at k>={args.k})")
    print(f"wrote {diff_path}")
    print(f"wrote {readers_path}")


if __name__ == "__main__":
    main()
