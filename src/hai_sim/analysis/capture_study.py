"""Capture study: does an ordering of the worklist put the right cases first?

Read the images of one dataset in the order a model proposes and count, as you go, how many
of the cases you care about you have already met. Two targets, per pathology:

- ``positive`` — the images that truly show the finding. The clinical question: do sick
  patients get read earlier?
- ``error``    — the images on which the model's own call is wrong. The uncertainty
  question: is the reader's attention sent where the model fails?

The curve is framed by two references (`hai_sim.analysis.error_curve.compute_curve`): the
**ideal** ordering meets every target first, so it climbs straight and then stays flat — a
right trapezoid; a **random** ordering is the diagonal. ``skill`` is the area between the
curve and the diagonal as a share of the area between the ideal and the diagonal
(1 = ideal, 0 = random, < 0 = worse than random); ``auroc`` is the probability that a target
case is ranked ahead of a non-target one.

Orderings (all "highest first"):

- ``probability``        — the model's probability of the finding: most suspicious first.
- ``predictive_entropy`` — entropy of that probability: most undecided first. Computed on
                           the raw or on the calibrated probability (``probabilities``);
                           it is the one ordering calibration can change, because an entropy
                           peaks at one half and calibration moves which cases sit there.
- ``mean_entropy``       — mean entropy of the ensemble members (aleatoric part).
- ``mutual_information`` — disagreement between members (epistemic part). Defined only
                           where at least two members cover the finding.

The member-level measures use the raw members. A model with one member (an LLM read-out)
has only the first two.

One YAML drives everything: which datasets, which models (a model is a list of cache-matrix
cells, composed by image id — an ensemble is declared here, not pre-built), which orderings.
The study evaluates on the cache rows the calibration was *not* fitted on.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from hai_sim.analysis.error_curve import CurveResult, compute_curve
from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache
from hai_sim.uq.calibration import load_platt_file
from hai_sim.uq.methods import binary_entropy

Ordering = Literal["probability", "predictive_entropy", "mean_entropy", "mutual_information"]
Target = Literal["positive", "error"]

ORDERING_LABEL: dict[str, str] = {
    "probability": "Probability (most suspicious first)",
    "predictive_entropy": "Predictive entropy",
    "mean_entropy": "Mean member entropy",
    "mutual_information": "Member disagreement",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaptureModel(_Strict):
    """One AI: the cache-matrix cells whose members are averaged into its probability."""

    cells: tuple[str, ...] = Field(min_length=1)
    label: str | None = None  # how the model is named inside a figure; default = its key
    calibration: str | None = None  # file stem under calibration_dir; default = model key


class CaptureStudyConfig(_Strict):
    name: str
    cache_dir: str = "data/cache"
    calibration_dir: str = "data/cache/calibration"
    datasets: tuple[str, ...]
    models: dict[str, CaptureModel]
    orderings: tuple[Ordering, ...] = (
        "probability",
        "predictive_entropy",
        "mean_entropy",
        "mutual_information",
    )
    probabilities: tuple[Literal["raw", "calibrated"], ...] = ("raw", "calibrated")
    targets: tuple[Target, ...] = ("positive", "error")
    # Evaluate on these cache rows; the calibration files are fitted on the other parity.
    split: Literal["odd", "even", "all"] = "odd"
    # A pathology enters the study only with this many target cases on the evaluated rows.
    min_targets: int = Field(default=20, ge=1)


def load_config(path: str | Path) -> CaptureStudyConfig:
    return CaptureStudyConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


# --- data ----------------------------------------------------------------------------------


def compose_members(caches: list[InferenceCache]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Stack the members of several cells by image id → ``(member_probs, ground_truth, ids)``.

    Rows follow the first cell's order restricted to the ids every cell holds, which is how
    `CompositeCachedAI` composes the same cells at replay time.
    """
    later = [set(c.image_ids) for c in caches[1:]]
    common = tuple(i for i in caches[0].image_ids if all(i in s for s in later))
    blocks = []
    for c in caches:
        pos = {img: r for r, img in enumerate(c.image_ids)}
        blocks.append(np.asarray(c.member_probs, dtype=float)[[pos[i] for i in common]])
    pos0 = {img: r for r, img in enumerate(caches[0].image_ids)}
    gt = np.asarray(caches[0].ground_truth)[[pos0[i] for i in common]]
    return np.concatenate(blocks, axis=1), gt, common


def _rows(n: int, split: str) -> np.ndarray:
    rows = np.arange(n)
    return rows if split == "all" else rows[rows % 2 == (0 if split == "even" else 1)]


def _prevalence_matched_threshold(p_fit: np.ndarray, y_fit: np.ndarray) -> float:
    """The cut at which the model calls as many positives as there are. It needs no cost
    and is unchanged by any monotone calibration, so raw and calibrated share their errors."""
    prevalence = float(np.mean(y_fit))
    return float(np.quantile(p_fit, 1.0 - prevalence))


def _scores(
    ordering: str, members: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray | None, kind: str
) -> np.ndarray | None:
    """The ranking score of one ordering for one pathology, or None when it does not exist."""
    if ordering == "probability":
        return p_raw  # a monotone calibration cannot change this order
    if ordering == "predictive_entropy":
        p = p_cal if kind == "calibrated" else p_raw
        return None if p is None else np.asarray(binary_entropy(p))
    covering = int(np.isfinite(members).any(axis=0).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_h = np.nanmean(binary_entropy(members), axis=1)
    if ordering == "mean_entropy":
        return None if covering < 2 else mean_h  # with one member it IS predictive entropy
    if ordering == "mutual_information":
        if covering < 2:
            return None
        return np.clip(np.asarray(binary_entropy(p_raw)) - mean_h, 0.0, None)
    raise ValueError(f"unknown ordering {ordering!r}")


# --- run -----------------------------------------------------------------------------------


def run_capture_study(cfg: CaptureStudyConfig) -> dict[str, Any]:
    """Every curve of the study → ``{"rows": [...], "curves": {key: CurveResult}}``."""
    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, str, str, str, str], CurveResult] = {}
    for dataset in cfg.datasets:
        for model_key, model in cfg.models.items():
            caches = [
                InferenceCache.load(str(Path(cfg.cache_dir) / f"{dataset}__{cell}"))
                for cell in model.cells
            ]
            members, gt, _ = compose_members(caches)
            labels = tuple(caches[0].pathologies)
            p_all = measures_from_cache(members)["mean_p"]
            cal_path = Path(cfg.calibration_dir) / f"{dataset}__{model.calibration or model_key}.json"
            p_cal_all = (
                load_platt_file(cal_path, labels).calibrate(p_all) if cal_path.exists() else None
            )
            eval_rows = _rows(members.shape[0], cfg.split)
            fit_rows = np.setdiff1d(np.arange(members.shape[0]), eval_rows)
            if fit_rows.size == 0:
                fit_rows = eval_rows
            for k, pathology in enumerate(labels):
                ok = np.isfinite(p_all[:, k]) & (gt[:, k] >= 0)
                ev = eval_rows[ok[eval_rows]]
                ft = fit_rows[ok[fit_rows]]
                if ev.size == 0 or ft.size == 0 or gt[ft, k].sum() == 0:
                    continue
                y = gt[ev, k] == 1
                threshold = _prevalence_matched_threshold(p_all[ft, k], gt[ft, k])
                is_target = {"positive": y, "error": (p_all[ev, k] >= threshold) != y}
                n_cover = int(np.isfinite(members[:, :, k]).any(axis=0).sum())
                for target in cfg.targets:
                    t = is_target[target]
                    if int(t.sum()) < cfg.min_targets or int((~t).sum()) == 0:
                        continue
                    for ordering in cfg.orderings:
                        kinds = cfg.probabilities if ordering == "predictive_entropy" else ("raw",)
                        for kind in kinds:
                            score = _scores(
                                ordering,
                                members[ev][:, :, k],
                                p_all[ev, k],
                                None if p_cal_all is None else p_cal_all[ev, k],
                                kind,
                            )
                            if score is None or float(np.nanstd(score)) == 0.0:
                                continue
                            curve = compute_curve(
                                score,
                                t,
                                error_type="any",
                                mode="entry",
                                measure=ordering,
                                threshold=threshold,
                            )
                            curves[(dataset, model_key, pathology, target, ordering, kind)] = curve
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "model": model_key,
                                    "pathology": pathology,
                                    "n_members": n_cover,
                                    "target": target,
                                    "ordering": ordering,
                                    "probabilities": kind,
                                    "n_images": curve.n_units,
                                    "n_targets": curve.n_errors,
                                    "skill": round(curve.skill, 4),
                                    "auroc": round(curve.auroc, 4),
                                }
                            )
    return {"rows": rows, "curves": curves}


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_and_write(cfg: CaptureStudyConfig, out_dir: str | Path = "out/capture") -> Path:
    """Run the study, write the tidy table, the config and the figures. Returns the folder."""
    from hai_sim.analysis.capture_figures import render_all

    out = Path(out_dir) / cfg.name
    result = run_capture_study(cfg)
    if not result["rows"]:
        raise ValueError("the study produced no curve: check datasets, cells and min_targets")
    write_table(result["rows"], out / "capture.csv")
    (out / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
    render_all(cfg, result, out)
    return out
