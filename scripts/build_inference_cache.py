"""Build a reusable offline inference cache over a sample of the dataset.

Runs the cross-domain Deep Ensemble (the weight sets from a scenario YAML) over a
deterministic sample of images, persisting the raw per-member probabilities,
ground truth and per-image inference timing (see ``analysis.inference_cache``).
While it runs it checkpoints the partial cache and re-renders the cumulative
error-capture curves, so the figure builds up incrementally during the (long) job.

Example (run from the repo root so the scenario's relative data paths resolve):

    PYTHONUTF8=1 uv run python scripts/build_inference_cache.py \
        --scenario configs/examples/v1_nih_ensemble.yaml \
        --out data/cache/nih_ensemble_2000 \
        --limit 2000 --batch-size 1 --checkpoint-every 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from hai_sim.analysis.inference_cache import InferenceCache, build_inference_cache
from hai_sim.config.scenario import load_scenario
from hai_sim.factory import build_label_schema, build_xrv_dataset
from hai_sim.models.torchxrv import TorchXRayVisionAI


def _select_indices(n_dataset: int, limit: int, sample: str, seed: int) -> list[int]:
    n = n_dataset if limit <= 0 else min(limit, n_dataset)
    if sample == "random":
        # Deterministic for a given (n_dataset, seed): a larger --limit is a prefix
        # superset of a smaller one, so resuming with a bigger limit just extends.
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.permutation(n_dataset)[:n]]
    return list(range(n))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build an offline ensemble inference cache")
    p.add_argument("--scenario", required=True, help="scenario YAML (dataset + ensemble weights)")
    p.add_argument("--out", default="data/cache/inference_cache", help="output stem (no extension)")
    p.add_argument("--limit", type=int, default=2000, help="max images to cache (<=0 = all)")
    p.add_argument("--batch-size", type=int, default=1, help="1 → true per-image latency timing")
    p.add_argument("--sample", choices=["random", "sequential"], default="random")
    p.add_argument("--seed", type=int, default=None, help="sampling seed (default: scenario.seed)")
    p.add_argument("--device", default=None, help="torch device (default: scenario.ai.device)")
    p.add_argument("--checkpoint-every", type=int, default=50, help="0 disables checkpoints")
    p.add_argument("--plot-dir", default="out/analysis", help="where to write the evolving figure")
    p.add_argument("--no-plot", action="store_true", help="skip rendering the curves")
    p.add_argument("--fresh", action="store_true", help="ignore any existing cache and start over")
    args = p.parse_args(argv)

    scenario = load_scenario(Path(args.scenario))
    schema = build_label_schema(scenario.label_schema)
    pg = scenario.patient_generator
    if pg.kind != "xrv_dataset":
        print(f"[error] scenario patient_generator.kind must be xrv_dataset, got {pg.kind}")
        return 2

    print(f"building dataset ({pg.dataset}) from {pg.imgpath} ...", flush=True)
    dataset = build_xrv_dataset(pg)
    device = args.device or scenario.ai.device
    print(f"loading ensemble {scenario.ai.weights} on {device} ...", flush=True)
    model = TorchXRayVisionAI(schema=schema, weights=scenario.ai.weights, device=device)

    seed = args.seed if args.seed is not None else scenario.seed
    target = _select_indices(len(dataset), args.limit, args.sample, seed)
    out_stem = Path(args.out)
    plot_path = Path(args.plot_dir) / "error_curves.png"

    # Resume: reuse a compatible cache at the same stem unless --fresh was passed.
    prior: InferenceCache | None = None
    if not args.fresh and out_stem.with_suffix(".npz").exists():
        prior = InferenceCache.load(out_stem)
        if (
            prior.weights != tuple(model.weight_list)
            or prior.pathologies != tuple(schema.labels)
            or prior.dataset != pg.dataset
        ):
            print(
                f"[error] existing cache {out_stem}.npz is incompatible "
                "(weights/pathologies/dataset differ). Use --fresh or a different --out."
            )
            return 2

    done = {int(i) for i in prior.dataset_indices} if prior is not None else set()
    remaining = [i for i in target if i not in done]
    target_total = len(set(target) | done)
    print(
        f"target {len(target)} images (sample={args.sample}, seed={seed}); "
        f"resume: {len(done)} already cached, {len(remaining)} to process "
        f"(batch_size={args.batch_size})",
        flush=True,
    )

    def _render(cache: InferenceCache, *, final: bool) -> None:
        if args.no_plot or cache.n_images == 0:
            return
        from hai_sim.analysis.error_curve import render_error_curves

        tag = "final" if final else f"N={cache.n_images}/{target_total}"
        render_error_curves(cache, plot_path, title_extra=f"[{tag}]")

    def on_checkpoint(cache: InferenceCache) -> None:
        cache.save(out_stem)
        _render(cache, final=False)
        print(
            f"  [checkpoint] {cache.n_images}/{target_total} cached  "
            f"mean_infer={float(np.mean(cache.inference_seconds)):.3f}s/img  -> {out_stem}.npz",
            flush=True,
        )

    t0 = time.time()
    cache = build_inference_cache(
        model,
        dataset,
        indices=remaining,
        dataset_name=pg.dataset,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=on_checkpoint,
        prior=prior,
    )
    cache.save(out_stem)
    _render(cache, final=True)
    elapsed = time.time() - t0
    mean_infer = float(np.mean(cache.inference_seconds)) if cache.n_images else 0.0
    print(
        f"DONE {cache.n_images} images in {elapsed:.1f}s wall  "
        f"({mean_infer:.3f}s/img model inference)\n"
        f"  cache  -> {out_stem}.npz (+ .json manifest)\n"
        f"  figure -> {plot_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
