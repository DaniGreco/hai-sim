"""Build the resumable localization cache (probs + per-member Grad-CAM + masks).

Runs the model from a validation scenario over ChestX-Det train + test *once* and
stores only raw model output; validation then runs offline from the cache
(`hai-sim validate --from-cache`) with any metric / threshold / aggregation / split.

Prereq: uv run python scripts/download_chestxdet.py --train   (all 3578 images)
Run (Windows: PYTHONUTF8=1, from the repo root):
  uv run python scripts/build_localization_cache.py \
    --scenario configs/examples/val_chestxdet_ensemble_clean.yaml \
    --out data/cache/chestxdet_localization --checkpoint-every 25

Auto-resume: re-run the same line; images already in the cache (matched by
split + dataset index) are skipped, so a sudden stop loses at most one checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from hai_sim.analysis.localization.cache import LocalizationCache, build_localization_cache
from hai_sim.analysis.localization.chestxdet import ChestXDetDataset
from hai_sim.analysis.localization.saliency import GradCAMSaliency
from hai_sim.config.scenario import load_validation_scenario
from hai_sim.factory import build_ai, build_label_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="build the ChestX-Det localization cache")
    parser.add_argument("--scenario", required=True, help="validation scenario YAML (model/saliency)")
    parser.add_argument("--out", required=True, help="cache stem (writes <out>.npz + <out>.json)")
    parser.add_argument("--data-dir", default="data/chestxdet", help="ChestX-Det root")
    parser.add_argument("--splits", default="train,test", help="comma list of splits to include")
    parser.add_argument("--mask-resolution", type=int, default=112)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0, help="cap images PER split (0 = all)")
    args = parser.parse_args()

    scenario = load_validation_scenario(Path(args.scenario))
    schema = build_label_schema(scenario.label_schema)
    ai = build_ai(scenario.ai, schema, np.random.default_rng(scenario.seed))
    saliency = GradCAMSaliency(ai)  # type: ignore[arg-type]  # scenario guarantees torchxrv

    out = Path(args.out)
    cache: LocalizationCache | None = LocalizationCache.load(out) if out.with_suffix(".npz").exists() else None
    done = (
        {(s, int(i)) for s, i in zip(cache.split, cache.dataset_indices, strict=True)}
        if cache is not None
        else set()
    )
    data_dir = Path(args.data_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    def _save(c: LocalizationCache) -> None:
        c.save(out)

    t0 = time.time()
    for split in splits:
        dataset = ChestXDetDataset(
            data_dir / f"ChestX_Det_{split}.json", data_dir / split, schema=schema
        )
        remaining = [i for i in range(len(dataset)) if (split, i) not in done]
        if args.limit:
            remaining = remaining[: args.limit]
        if not remaining:
            print(f"[{split}] nothing to do ({len(dataset)} images already cached)")
            continue
        print(f"[{split}] {len(remaining)} images to process (of {len(dataset)})")

        def _progress(done_n: int, total_n: int, _split: str = split) -> None:
            step = max(1, total_n // 50)
            if done_n % step == 0 or done_n == total_n:
                secs = time.time() - t0
                print(f"  [{_split}] {done_n}/{total_n}  ({secs:.0f}s)", flush=True)

        cache = build_localization_cache(
            saliency,
            dataset,
            indices=remaining,
            split_tags=[split] * len(remaining),
            mask_resolution=args.mask_resolution,
            checkpoint_every=args.checkpoint_every,
            on_checkpoint=_save,
            prior=cache,
            dataset_fingerprint=f"chestxdet|{','.join(splits)}",
            progress=_progress,
        )
        _save(cache)

    if cache is None:
        print("no images cached", file=sys.stderr)
        return 1
    print(
        f"\ndone. cache: {out}.npz  images={cache.n_images}  instances={cache.n_instances}  "
        f"members={cache.n_members}  ({time.time() - t0:.0f}s)"
    )
    print("validate offline with: hai-sim validate --from-cache", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
