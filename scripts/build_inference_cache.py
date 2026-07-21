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

from hai_sim.analysis.inference_cache import (
    CacheableModel,
    InferenceCache,
    build_inference_cache,
)
from hai_sim.config.scenario import load_scenario
from hai_sim.factory import build_ai, build_label_schema, build_xrv_dataset


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
    p.add_argument(
        "--measure",
        default="predictive_entropy",
        choices=[
            "mutual_information",
            "predictive_entropy",
            "mean_entropy",
            "std",
            "least_confidence",
        ],
        help="primary UQ measure for the rendered curves (default: predictive_entropy)",
    )
    args = p.parse_args(argv)

    scenario = load_scenario(Path(args.scenario))
    schema = build_label_schema(scenario.label_schema)
    pg = scenario.patient_generator
    if pg.kind != "xrv_dataset":
        print(f"[error] scenario patient_generator.kind must be xrv_dataset, got {pg.kind}")
        return 2

    print(
        f"building dataset ({pg.dataset}, views={list(pg.views)}, "
        f"unique_patients={pg.unique_patients}) from {pg.imgpath} ...",
        flush=True,
    )
    dataset = build_xrv_dataset(pg)
    device = args.device or scenario.ai.device
    # Any backend that can score an image several times over: a Deep Ensemble's weight
    # sets, or a generative model's prompt paraphrases. Built through the factory so the
    # scenario's `ai.kind` is the only thing that decides which one, exactly as in a run.
    ai_cfg = scenario.ai.model_copy(update={"device": device})
    print(f"loading AI backend (kind={ai_cfg.kind}) on {device} ...", flush=True)
    model = build_ai(ai_cfg, schema, np.random.default_rng(scenario.seed))
    if not isinstance(model, CacheableModel):
        print(
            f"[error] ai.kind={ai_cfg.kind} cannot be cached: it exposes no member "
            "probabilities. Use torchxrv or medgemma."
        )
        return 2

    seed = args.seed if args.seed is not None else scenario.seed
    target = _select_indices(len(dataset), args.limit, args.sample, seed)
    out_stem = Path(args.out)
    # Composition signature: dataset indices are only meaningful for this exact
    # filtering, so a differently-filtered cache must never be resumed into.
    fingerprint = (
        f"{pg.dataset}|views={','.join(pg.views)}|unique={pg.unique_patients}|ndata={len(dataset)}"
    )

    # Resume: reuse a compatible cache at the same stem unless --fresh was passed.
    prior: InferenceCache | None = None
    if not args.fresh and out_stem.with_suffix(".npz").exists():
        prior = InferenceCache.load(out_stem)
        if (
            prior.weights != tuple(model.member_names)
            or prior.pathologies != tuple(schema.labels)
            or prior.dataset != pg.dataset
            or (prior.dataset_fingerprint and prior.dataset_fingerprint != fingerprint)
        ):
            print(
                f"[error] existing cache {out_stem}.npz is incompatible "
                f"(weights/pathologies/dataset/views differ).\n"
                f"  existing: {prior.dataset_fingerprint!r}\n"
                f"  current : {fingerprint!r}\n"
                "  Use --fresh or a different --out."
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
        from hai_sim.analysis.error_curve import render_dashboard

        tag = "final" if final else f"{cache.n_images}/{target_total}"
        # Primary snapshot lands in the same layout as plot_dashboard.py: this builder
        # produces the cross-domain ensemble cache at the default threshold / aggregator.
        base = Path(args.plot_dir) / "ensemble" / args.measure / "thr0.5"
        render_dashboard(cache, base, measure=args.measure, title_extra=f"[{tag}]")

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
        dataset_fingerprint=fingerprint,
    )
    cache.save(out_stem)
    _render(cache, final=True)
    elapsed = time.time() - t0
    mean_infer = float(np.mean(cache.inference_seconds)) if cache.n_images else 0.0
    print(
        f"DONE {cache.n_images} images in {elapsed:.1f}s wall  "
        f"({mean_infer:.3f}s/img model inference)\n"
        f"  cache   -> {out_stem}.npz (+ .json manifest)\n"
        f"  figures -> {args.plot_dir}/ensemble/{args.measure}/thr0.5/ "
        f"(mean/error_curves.png, per_pathology_*.png)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
