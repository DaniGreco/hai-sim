"""Build a case-difficulty table — the column `difficulty.kind: table` consumes.

Two sources, one output format. The output is always a CSV of
``example_id,difficulty``, which is deliberately the *least* committal interface it
could have: a future learned complexity model plugs into the same file, and so does
anything else that scores images.

**Source 1: inter-reader disagreement** (``--readers``). A case is hard exactly when
expert readers disagree about it. This is the most defensible operational definition
available and it needs no model — ``../TESI.md`` §4.8 already settled on divergence from
a consensus as the definition of *error*; this points the same idea at difficulty and
asks how much the panel split rather than who was wrong. The CheXpert test set ships
eight radiologists' reads of 500 studies, which is on disk today.

**Source 2: AI predictive uncertainty** (``--cache``). §8.1 names this the natural
observable proxy for difficulty, and §10 names it as the escape from the arbitrary
``Beta(2, 5)`` prior — with it the human-AI error correlation is *induced* from real
model behaviour rather than imposed.

Producing both is the point. They are two independent estimates of the same latent
quantity, so **correlating them is a research question that costs nothing**: does the
model find the same cases hard that radiologists disagree about? Pass ``--compare`` with
both sources to get the answer on whatever images they share.

Usage:

    python scripts/build_difficulty_table.py \\
        --readers data/chexpert_test_labels/radiologists \\
        --out data/difficulty/chexpert_disagreement.csv

    python scripts/build_difficulty_table.py \\
        --cache data/cache/nih_ensemble --measure mutual_information \\
        --out data/difficulty/nih_ensemble_uncertainty.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from hai_sim.difficulty.models import disagreement_table


def _read_reader_csv(path: Path) -> dict[str, tuple[int, ...]]:
    """One reader's reads: first column is the study id, the rest are {1,0,-1} labels."""
    out: dict[str, tuple[int, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return out
        for row in reader:
            if not row:
                continue
            values: list[int] = []
            for cell in row[1:]:
                try:
                    values.append(int(float(cell)))
                except ValueError:
                    values.append(-1)
            out[row[0]] = tuple(values)
    return out


def _from_readers(root: Path) -> dict[str, float]:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"[error] no reader CSVs under {root}")
    per_image: dict[str, dict[str, tuple[int, ...]]] = {}
    widths: set[int] = set()
    for path in csvs:
        reader_id = path.stem
        for example_id, labels in _read_reader_csv(path).items():
            per_image.setdefault(example_id, {})[reader_id] = labels
            widths.add(len(labels))
    if len(widths) > 1:
        raise SystemExit(
            f"[error] reader files disagree on the label count: {sorted(widths)}. They "
            "must share one label schema, or the columns being compared are not the "
            "same findings."
        )
    counts = [len(v) for v in per_image.values()]
    print(
        f"{len(csvs)} reader files, {len(per_image)} studies, "
        f"{min(counts)}-{max(counts)} readers per study, {widths.pop()} labels"
    )
    if min(counts) < 2:
        print(
            "note: some studies have a single reader; disagreement is undefined there "
            "and comes back as 0.0 (easy), which understates their difficulty."
        )
    return disagreement_table(per_image)


def _from_cache(stem: str, measure: str) -> dict[str, float]:
    from hai_sim.analysis.inference_cache import InferenceCache
    from hai_sim.uq.methods import EnsembleUQ

    cache = InferenceCache.load(stem)
    uq = EnsembleUQ(measure=measure)
    span = float(np.log(2.0)) if measure != "std" else 0.5
    out: dict[str, float] = {}
    for i, image_id in enumerate(cache.image_ids):
        _probs, unc, _extra = uq.estimate(np.asarray(cache.member_probs[i], dtype=float))
        finite = unc[np.isfinite(unc)]
        out[str(image_id)] = float(np.clip(np.mean(finite) / span, 0.0, 1.0)) if finite.size else 0.0
    print(f"cache {stem}: {cache.n_images} images, measure={measure}")
    if cache.member_probs.shape[1] == 1:
        print(
            "WARNING: this cache has ONE member, so there is no disagreement to measure and "
            "an epistemic measure is identically zero. Use predictive_entropy, or build "
            "a cache with several members (ensemble weights / prompt paraphrases)."
        )
    return out


def _write(table: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["example_id", "difficulty"])
        for example_id, value in sorted(table.items()):
            writer.writerow([example_id, f"{value:.6f}"])
    values = np.asarray(list(table.values()), dtype=float)
    print(
        f"wrote {len(table)} rows -> {path}\n"
        f"  difficulty: mean={values.mean():.3f} median={np.median(values):.3f} "
        f"p90={np.quantile(values, 0.9):.3f} max={values.max():.3f} "
        f"zeros={int(np.sum(values == 0))}"
    )


def _compare(a: dict[str, float], b: dict[str, float]) -> None:
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        print(
            f"\n[compare] only {len(shared)} shared image ids — the two sources index "
            "different datasets, so they cannot be correlated. This is expected for "
            "CheXpert reads against an NIH cache."
        )
        return
    x = np.asarray([a[k] for k in shared])
    y = np.asarray([b[k] for k in shared])
    if x.std() == 0 or y.std() == 0:
        print("\n[compare] one source is constant; correlation undefined.")
        return
    r = float(np.corrcoef(x, y)[0, 1])
    print(
        f"\n[compare] {len(shared)} shared images, Pearson r = {r:+.3f}\n"
        "  This is the open question in one number: does the model find hard what the "
        "radiologists disagreed about? A high r means the AI's uncertainty is a usable "
        "stand-in for reader difficulty (and the shared-difficulty model can be "
        "grounded); a low one means they are measuring different things, which is a "
        "result in itself and should be reported rather than assumed away."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a case-difficulty table")
    p.add_argument("--readers", help="directory of per-reader label CSVs")
    p.add_argument("--cache", help="inference cache stem")
    p.add_argument(
        "--measure",
        default="mutual_information",
        choices=["mutual_information", "predictive_entropy", "mean_entropy", "std"],
    )
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument(
        "--compare",
        action="store_true",
        help="with both sources: report how well they agree (see the module docstring)",
    )
    args = p.parse_args(argv)

    if not args.readers and not args.cache:
        print("[error] pass --readers, --cache, or both")
        return 2

    tables: dict[str, dict[str, float]] = {}
    if args.readers:
        tables["disagreement"] = _from_readers(Path(args.readers))
    if args.cache:
        tables["uncertainty"] = _from_cache(args.cache, args.measure)

    if args.compare and len(tables) == 2:
        _compare(tables["disagreement"], tables["uncertainty"])
    elif args.compare:
        print("[compare] needs both --readers and --cache; skipped")

    out = Path(args.out)
    if len(tables) == 2:
        # Two sources, two files: they are different estimates of difficulty and
        # averaging them would hide exactly the disagreement worth reporting.
        for name, table in tables.items():
            _write(table, out.with_name(f"{out.stem}_{name}{out.suffix}"))
    else:
        _write(next(iter(tables.values())), out)

    print(
        "\nUse it in a scenario:\n"
        "  difficulty:\n"
        "    kind: table\n"
        f"    path: {out}\n"
        "    strict: true   # fail on an uncovered image rather than flattening it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
