"""Materialize cache-matrix cells for a (model, dataset) outside a study run.

The standalone build command for the cache matrix (`hai_sim.analysis.cache_matrix`): it
takes a scenario YAML (which already declares the model and the dataset), selects a
sample of images, and — per ``--mode`` — reports the plan, requires it complete, or
**fills** the gaps by running each atomic model only over the ids it is still missing.
This is the natural place to run on the GPU server; a study on the laptop then replays
the cells with ``ai.cache_mode: require``.

    # Build every member of the VinDr ensemble over 15k images, on GPU 0:
    PYTHONUTF8=1 uv run python scripts/materialize_cache.py \
        --scenario configs/examples/v4_vindr_ensemble.yaml \
        --limit 0 --device cuda:0 --checkpoint-every 200

    # Just one member, and just inspect the plan:
    uv run python scripts/materialize_cache.py --scenario ... --only chex --mode report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.analysis.cache_build import _id_at, materialize_cell
from hai_sim.analysis.cache_matrix import (
    apply_cache_mode,
    cells_for_ai,
    dataset_cell_fingerprint,
    dataset_id,
    plan_summary,
    resolve_matrix,
)
from hai_sim.config.scenario import load_scenario
from hai_sim.factory import build_cacheable_dataset, build_label_schema

if TYPE_CHECKING:
    from hai_sim.analysis.cache_build import ModelBuilder
    from hai_sim.analysis.cache_matrix import CellSpec
    from hai_sim.config.scenario import AIModelConfig, PatientGenConfig
    from hai_sim.core.types import LabelSchema


def select_indices(n_dataset: int, limit: int, sample: str, seed: int) -> list[int]:
    """Deterministic image selection: a larger ``--limit`` is a prefix superset of a
    smaller one (same seed), so a bigger run just extends the accumulating cell."""
    n = n_dataset if limit <= 0 else min(limit, n_dataset)
    if sample == "random":
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.permutation(n_dataset)[:n]]
    return list(range(n))


def run_materialize(
    ai: AIModelConfig,
    pg: PatientGenConfig,
    schema: LabelSchema,
    *,
    limit: int,
    sample: str,
    seed: int,
    device: str | None,
    cache_root: str,
    only: tuple[str, ...] | None,
    mode: str,
    batch_size: int,
    checkpoint_every: int,
    dataset: Any | None = None,
    model_builder: ModelBuilder | None = None,
) -> None:
    """Resolve the model's cells for the dataset and apply ``mode`` (report/require/fill).

    ``dataset`` / ``model_builder`` are injectable so the orchestration is testable with
    a fake model over a tiny folder; production leaves them ``None`` (built via factory).
    """
    cells = list(cells_for_ai(ai))
    if only is not None:
        wanted = set(only)
        cells = [c for c in cells if c.alias in wanted]
        missing = wanted - {c.alias for c in cells}
        if missing:
            raise ValueError(f"--only names cells not in this model: {sorted(missing)}")
    if not cells:
        raise ValueError("no cells selected to materialize")

    ds = dataset if dataset is not None else build_cacheable_dataset(pg, schema)
    indices = select_indices(len(ds), limit, sample, seed)
    id_of = _id_at(ds)
    requested_ids = [id_of(int(i)) for i in indices]

    ds_id = dataset_id(pg)
    ds_fp = dataset_cell_fingerprint(pg)
    root = Path(cache_root)

    plan = resolve_matrix(cells, ds_id, ds_fp, requested_ids, root)
    print(plan_summary(plan), flush=True)
    if mode == "report":
        return

    def fill(spec: CellSpec) -> None:
        print(f"  building {spec.alias} on {device or 'cpu'} ...", flush=True)
        materialize_cell(
            spec,
            pg,
            schema,
            indices=indices,
            root=root,
            device=device,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            dataset=ds,
            model_builder=model_builder,
        )
        print(f"  done {spec.alias}", flush=True)

    apply_cache_mode(plan, mode, fill)  # 'require' raises on a gap; 'fill' builds it
    if mode == "fill":
        print("final:", flush=True)
        print(plan_summary(resolve_matrix(cells, ds_id, ds_fp, requested_ids, root)), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Materialize cache-matrix cells for a model x dataset")
    p.add_argument("--scenario", required=True, help="scenario YAML (declares ai + patient_generator)")
    p.add_argument("--limit", type=int, default=0, help="max images (<=0 = all)")
    p.add_argument("--sample", choices=["random", "sequential"], default="sequential")
    p.add_argument("--seed", type=int, default=None, help="sampling seed (default: scenario.seed)")
    p.add_argument("--device", default=None, help="torch device (default: scenario.ai.device)")
    p.add_argument("--cache-root", default=None, help="matrix root (default: scenario.ai.cache_root)")
    p.add_argument("--only", default=None, help="comma-separated cell aliases to build (default: all)")
    p.add_argument("--mode", choices=["report", "require", "fill"], default="fill")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=200, help="0 disables checkpoints")
    args = p.parse_args(argv)

    scenario = load_scenario(Path(args.scenario))
    schema = build_label_schema(scenario.label_schema)
    only = tuple(a.strip() for a in args.only.split(",")) if args.only else None
    try:
        run_materialize(
            scenario.ai,
            scenario.patient_generator,
            schema,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed if args.seed is not None else scenario.seed,
            device=args.device or scenario.ai.device,
            cache_root=args.cache_root or scenario.ai.cache_root,
            only=only,
            mode=args.mode,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
