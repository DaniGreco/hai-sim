"""Figures of the capture study. English only, one look for every figure.

Three kinds, each answering one question:

- ``curves_<dataset>_<model>_<target>``  one panel per pathology, one line per ordering:
  *which ordering puts the target cases first?*
- ``models_<dataset>_<target>``           one panel per pathology, one line per model, all
  ordered by probability: *which model puts them first?*
- ``skill_<dataset>_<target>``            pathology x (model, ordering) table of the skill:
  the whole study at a glance, including the orderings the curve panels leave out.

Axes are shares (images read, targets met), so every panel has the same frame: the ideal
ordering is the right trapezoid, the random one the diagonal.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.analysis.capture_study import ORDERING_LABEL

if TYPE_CHECKING:
    from hai_sim.analysis.capture_study import CaptureStudyConfig
    from hai_sim.analysis.error_curve import CurveResult

# Categorical slots 1-3 of the validated palette (the only set that separates on every
# pair, which small multiples need), plus neutral inks for the two references and text.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DASHES = ("-", (0, (5, 1.5)), (0, (1.5, 1.2)))  # identity never rests on colour alone
INK, INK_2, RULE, IDEAL = "#0b0b0b", "#52514e", "#d9d8d2", "#8a8983"
SURFACE = "#ffffff"

TARGET_TITLE = {
    "positive": "share of positive cases already read",
    "error": "share of the model's errors already read",
}
CURVE_ORDERINGS = (
    ("probability", "raw"),
    ("predictive_entropy", "calibrated"),
    ("mean_entropy", "raw"),
)
PRETTY = {"Pleural_Thickening": "Pleural thickening"}


def _pretty(name: str) -> str:
    return PRETTY.get(name, name)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
        }
    )
    return plt


def _thin(curve: CurveResult, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """The curve as shares, thinned for plotting (denser at the start, where it bends)."""
    n = curve.n_units
    idx = np.unique(
        np.concatenate(
            [np.geomspace(1, n, points // 2).astype(int), np.linspace(1, n, points // 2).astype(int)]
        )
    ) - 1
    x = np.concatenate([[0.0], curve.x[idx] / n])
    y = np.concatenate([[0.0], curve.actual[idx] / max(curve.n_errors, 1)])
    return x, y


def _frame(ax: Any, curve: CurveResult) -> None:
    share = curve.n_errors / curve.n_units
    ax.plot([0, share, 1], [0, 1, 1], color=IDEAL, lw=1.0, zorder=1)
    ax.plot([0, 1], [0, 1], color=IDEAL, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.grid(color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label(ax: Any, pathology: str, n: int) -> None:
    ax.set_title(_pretty(pathology), fontsize=7.8, color=INK, loc="left", pad=3)
    ax.text(0.97, 0.06, f"n = {n}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color=INK_2)


def _grid(plt: Any, n: int, cols: int = 4) -> tuple[Any, list[Any]]:
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.62 * rows + 1.15), squeeze=False)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]


def _finish(fig: Any, handles: list[Any], title: str, subtitle: str, ylab: str, out: Path) -> None:
    from matplotlib.lines import Line2D

    refs = [
        Line2D([], [], color=IDEAL, lw=1.0, label="Ideal ordering"),
        Line2D([], [], color=IDEAL, lw=1.0, ls=(0, (2, 2)), label="Random ordering"),
    ]
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.text(0.01, 1 - 0.27 / h, subtitle, ha="left", va="top", fontsize=8, color=INK_2)
    fig.legend(
        handles=handles + refs,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        labelcolor=INK_2,
        handlelength=2.6,
        columnspacing=1.6,
    )
    fig.supxlabel("share of images read, in the proposed order", fontsize=8, color=INK_2, y=0.52 / h)
    fig.supylabel(ylab, fontsize=8, color=INK_2, x=0.005)
    fig.tight_layout(rect=(0.015, 0.62 / h, 1, 1 - 0.5 / h), h_pad=1.1, w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))


def _pathologies(curves: dict[Any, CurveResult], dataset: str, target: str) -> list[str]:
    """Pathologies of a dataset, most frequent target first (the order of every figure)."""
    size: dict[str, int] = {}
    for (ds, _m, pathology, tg, _o, _k), c in curves.items():
        if ds == dataset and tg == target:
            size[pathology] = max(size.get(pathology, 0), c.n_errors)
    return sorted(size, key=lambda p: -size[p])


def render_orderings(curves: dict[Any, CurveResult], dataset: str, model: str, target: str, out: Path) -> None:
    from matplotlib.lines import Line2D

    plt = _plt()
    names = [p for p in _pathologies(curves, dataset, target)
             if (dataset, model, p, target, "probability", "raw") in curves]
    if not names:
        return
    fig, axes = _grid(plt, len(names))
    for ax, pathology in zip(axes, names, strict=True):
        first = curves[(dataset, model, pathology, target, "probability", "raw")]
        _frame(ax, first)
        for (ordering, kind), color, dash in zip(CURVE_ORDERINGS, SERIES, DASHES, strict=True):
            c = curves.get((dataset, model, pathology, target, ordering, kind))
            if c is None:
                continue
            ax.plot(*_thin(c), color=color, lw=1.5, ls=dash, zorder=3)
        _label(ax, pathology, first.n_errors)
    handles = [
        Line2D([], [], color=col, lw=1.5, ls=dash,
               label=ORDERING_LABEL[o] + (" (calibrated)" if k == "calibrated" else ""))
        for (o, k), col, dash in zip(CURVE_ORDERINGS, SERIES, DASHES, strict=True)
    ]
    what = "positive cases" if target == "positive" else "model errors"
    _finish(
        fig, handles,
        f"Which ordering meets the {what} first? {model}, {dataset.upper()}",
        "One panel per finding. The closer a line is to the ideal ordering, the earlier the cases are read.",
        TARGET_TITLE[target], out,
    )
    plt.close(fig)


def render_models(curves: dict[Any, CurveResult], models: list[str], dataset: str, target: str, out: Path) -> None:
    from matplotlib.lines import Line2D

    plt = _plt()
    names = [p for p in _pathologies(curves, dataset, target)
             if all((dataset, m, p, target, "probability", "raw") in curves for m in models)]
    if not names or len(models) < 2:
        return
    shown = models[: len(SERIES)]
    fig, axes = _grid(plt, len(names))
    for ax, pathology in zip(axes, names, strict=True):
        first = curves[(dataset, shown[0], pathology, target, "probability", "raw")]
        _frame(ax, first)
        for m, color, dash in zip(shown, SERIES, DASHES, strict=False):
            ax.plot(*_thin(curves[(dataset, m, pathology, target, "probability", "raw")]),
                    color=color, lw=1.5, ls=dash, zorder=3)
        _label(ax, pathology, first.n_errors)
    handles = [Line2D([], [], color=c, lw=1.5, ls=d, label=m) for m, c, d in zip(shown, SERIES, DASHES, strict=False)]
    _finish(
        fig, handles,
        f"Which model puts the positive cases first? {dataset.upper()}",
        "Every model orders the worklist by its own probability of the finding, most suspicious first.",
        TARGET_TITLE[target], out,
    )
    plt.close(fig)


def render_skill_table(rows: list[dict[str, Any]], models: list[str], dataset: str, target: str, out: Path) -> None:
    """Pathology x (model, ordering) skill, on a diverging scale centred on random."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt = _plt()
    sel = [r for r in rows if r["dataset"] == dataset and r["target"] == target]
    if not sel:
        return
    columns: list[tuple[str, str, str]] = []
    for m in models:
        for o, k in (("probability", "raw"), ("predictive_entropy", "raw"),
                     ("predictive_entropy", "calibrated"), ("mean_entropy", "raw"),
                     ("mutual_information", "raw")):
            if any(r["model"] == m and r["ordering"] == o and r["probabilities"] == k for r in sel):
                columns.append((m, o, k))
    size = {r["pathology"]: r["n_targets"] for r in sel}
    names = sorted(size, key=lambda p: -size[p])
    grid = np.full((len(names), len(columns)), np.nan)
    for r in sel:
        key = (r["model"], r["ordering"], r["probabilities"])
        if key in columns:
            grid[names.index(r["pathology"]), columns.index(key)] = r["skill"]

    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f4b08a", "#ecebe6", "#8fb8ea", "#1d5fb5"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(names) + 1.65))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=TwoSlopeNorm(0.0, -1.0, 1.0), aspect="auto")
    for i in range(len(names)):
        for j in range(len(columns)):
            v = grid[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.6, color=INK_2 if np.isnan(v) else (SURFACE if abs(v) > 0.6 else INK))
    short = {("probability", "raw"): "Proba-\nbility", ("predictive_entropy", "raw"): "Entropy\nraw",
             ("predictive_entropy", "calibrated"): "Entropy\ncalibr.",
             ("mean_entropy", "raw"): "Member\nentropy", ("mutual_information", "raw"): "Member\ndisagr."}
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([short[(o, k)] for _m, o, k in columns], fontsize=6.2, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{_pretty(p)}  (n = {size[p]})" for p in names], fontsize=7.2, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    start = 0
    for m in models:
        width = sum(1 for c in columns if c[0] == m)
        if width:
            ax.text(start + (width - 1) / 2, -1.55, m, ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
            if start:
                ax.axvline(start - 0.5, color=SURFACE, lw=4)
            start += width
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    what = "positive cases" if target == "positive" else "model errors"
    h = fig.get_figheight()
    fig.suptitle(f"Skill of each ordering at meeting the {what} first, {dataset.upper()}",
                 x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=10, color=INK, weight="bold")
    fig.text(0.01, 0.02 / h, "Skill: +1 = ideal ordering, 0 = random ordering, negative = worse than random.\n"
             "n/a: the ordering does not exist for that finding (fewer than two ensemble members cover it).",
             ha="left", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.34 / h, 1, 1 - 0.3 / h))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def render_all(cfg: CaptureStudyConfig, result: dict[str, Any], out: Path) -> None:
    curves, rows, models = result["curves"], result["rows"], list(cfg.models)
    for dataset in cfg.datasets:
        for target in cfg.targets:
            for model in models:
                render_orderings(curves, dataset, model, target,
                                 out / "figures" / f"curves_{dataset}_{model}_{target}")
            render_skill_table(rows, models, dataset, target, out / "figures" / f"skill_{dataset}_{target}")
        if "positive" in cfg.targets:
            render_models(curves, models, dataset, "positive", out / "figures" / f"models_{dataset}_positive")
