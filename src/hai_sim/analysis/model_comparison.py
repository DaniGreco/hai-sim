"""Comparing AI readers of different natures on exactly the same images.

A discriminative classifier and a generative vision-language model answer the same
question by completely different means, so "which is better?" is not one number. This
module puts several `InferenceCache`s side by side — matched image by image — and asks
four questions that a single accuracy figure cannot separate:

1. **Does it discriminate?** Per-pathology AUROC (area under the receiver operating
   characteristic curve): the probability the model scores a random positive above a
   random negative. Threshold-free, so it measures ranking rather than a decision rule.
2. **Do its probabilities mean anything?** A reliability curve: of the cases it called
   0.8, did ~80% turn out positive? Discrimination and calibration are independent — a
   model can rank perfectly and still state wildly wrong probabilities, and a generative
   model is a prime suspect for exactly that.
3. **Do two models fail on the same cases?** The error-overlap composition and its
   agreement coefficient κ. This is the one that matters most here: a pair only beats
   its members where their mistakes *differ*, so the overlap is the substrate of the
   whole human-AI complementarity argument (``../TESI.md`` §5). With two real models it
   can be measured instead of assumed.
4. **Does its uncertainty find its errors?** Cumulative error capture: walk the entries
   from most to least uncertain and count mistakes. A curve above the diagonal means the
   model knows where it is weak — the premise triage rests on.

Everything is computed from cached predictions, so a comparison costs no inference.
Models are matched on ``image_ids`` and on the pathologies they share; anything only one
of them scored is excluded rather than compared against a blank.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.analysis.error_curve import _auroc
from hai_sim.analysis.inference_cache import InferenceCache, measures_from_cache

# Categorical series colours, in fixed slot order — a model keeps its colour no matter
# how many others are in the figure. Validated for colour-vision deficiency separation
# on a light surface; three of them fall below 3:1 contrast, which is why every mark in
# this module carries a visible label rather than relying on hue alone.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
)
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class ModelRun:
    """One model's cached predictions over a set of images."""

    name: str
    image_ids: tuple[str, ...]
    pathologies: tuple[str, ...]
    probs: np.ndarray
    """``(N, L)`` fused probability of each pathology being present."""
    uncertainty: np.ndarray
    """``(N, L)`` the model's own doubt, in the requested measure."""
    ground_truth: np.ndarray
    """``(N, L)`` in ``{1, 0, -1}``; -1 = not annotated, excluded from every metric."""

    def errors(self, threshold: float = 0.5) -> np.ndarray:
        """``(N, L)`` boolean: the call at this threshold disagrees with the truth.

        Unannotated entries come back ``False`` — they are not mistakes, they are
        unmeasurable, and callers pair this with `scorable` rather than reading it alone.
        """
        called = self.probs >= threshold
        truth = self.ground_truth == 1
        return np.asarray((called != truth) & self.scorable(), dtype=bool)

    def scorable(self) -> np.ndarray:
        """``(N, L)`` boolean: entries this dataset actually annotated."""
        return np.asarray(np.isin(self.ground_truth, (0, 1)), dtype=bool)


def parse_cache_spec(spec: str) -> tuple[str, list[int] | None]:
    """``"path/to/stem"`` or ``"path/to/stem#0"`` / ``"#0,2"`` to select members.

    The suffix is how one cache supplies more than one model to a comparison — most
    usefully "this single ensemble member" against "the whole ensemble".
    """
    stem, sep, members = spec.partition("#")
    if not sep:
        return stem, None
    try:
        picked = [int(part) for part in members.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"bad member selection in {spec!r}; expected e.g. stem#0,2") from None
    if not picked:
        raise ValueError(f"bad member selection in {spec!r}; expected e.g. stem#0,2")
    return stem, picked


def load_runs(
    specs: Sequence[str | Path],
    names: Sequence[str] | None = None,
    measure: str = "predictive_entropy",
) -> list[ModelRun]:
    """Load one `ModelRun` per cache spec, defaulting each name to its file name."""
    if names is not None and len(names) != len(specs):
        raise ValueError(f"got {len(names)} names for {len(specs)} caches")
    runs: list[ModelRun] = []
    for i, spec in enumerate(specs):
        stem, members = parse_cache_spec(str(spec))
        cache = InferenceCache.load(stem)
        if names is not None:
            label = names[i]
        else:
            label = Path(stem).name + (f"#{','.join(map(str, members))}" if members else "")
        runs.append(run_from_cache(cache, label, measure, members))
    return runs


def run_from_cache(
    cache: InferenceCache,
    name: str,
    measure: str,
    members: Sequence[int] | None = None,
) -> ModelRun:
    """Fuse a cache's members into the probability and uncertainty a comparison needs.

    ``members`` selects a subset by index, which is how one cache yields several models:
    a single ensemble member is a different reader from the ensemble it belongs to, and
    comparing the two is the cleanest available demonstration of what ensembling buys.
    """
    stack = cache.member_probs
    if members is not None:
        picked = list(members)
        if not picked:
            raise ValueError("members must not be empty")
        if any(m < 0 or m >= stack.shape[1] for m in picked):
            raise ValueError(
                f"member index out of range; the cache has {stack.shape[1]} members"
            )
        stack = stack[:, picked, :]
    measures = measures_from_cache(stack)
    if measure not in measures:
        raise ValueError(f"unknown measure {measure!r}; available: {sorted(measures)}")
    return ModelRun(
        name=name,
        image_ids=tuple(cache.image_ids),
        pathologies=tuple(cache.pathologies),
        probs=np.asarray(measures["mean_p"], dtype=float),
        uncertainty=np.asarray(measures[measure], dtype=float),
        ground_truth=np.asarray(cache.ground_truth, dtype=int),
    )


def align_runs(runs: Sequence[ModelRun]) -> list[ModelRun]:
    """Restrict every run to the images and pathologies they all share.

    Comparing models on different images, or on pathologies only one of them scored,
    would attribute a dataset difference to the model. Order follows the first run, so
    the result is deterministic.
    """
    if not runs:
        return []
    shared_ids = [
        img for img in runs[0].image_ids if all(img in set(r.image_ids) for r in runs[1:])
    ]
    shared_labels = [
        name
        for name in runs[0].pathologies
        if all(name in set(r.pathologies) for r in runs[1:])
    ]
    if not shared_ids:
        raise ValueError("the caches share no images; are they over the same dataset?")
    if not shared_labels:
        raise ValueError("the caches share no pathologies; are they on the same schema?")

    aligned: list[ModelRun] = []
    for run in runs:
        row_of = {img: i for i, img in enumerate(run.image_ids)}
        rows = np.array([row_of[img] for img in shared_ids], dtype=int)
        cols = np.array([run.pathologies.index(name) for name in shared_labels], dtype=int)
        picker = np.ix_(rows, cols)
        aligned.append(
            ModelRun(
                name=run.name,
                image_ids=tuple(shared_ids),
                pathologies=tuple(shared_labels),
                probs=run.probs[picker],
                uncertainty=run.uncertainty[picker],
                ground_truth=run.ground_truth[picker],
            )
        )
    return aligned


# ── 1. Discrimination ───────────────────────────────────────────────────────────


def auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank-based area under the ROC curve; ``None`` when one class is absent.

    Returning ``None`` rather than 0.5 for a degenerate split matters: "we could not
    measure this" and "the model was no better than chance" are different statements and
    a chart must not merge them. The Mann-Whitney computation (mid-rank tie handling and
    all) is shared with `error_curve._auroc`, which returns NaN on the degenerate split;
    this only maps that NaN to the ``None`` the discrimination chart wants.
    """
    value = _auroc(np.asarray(scores, dtype=float), positive)
    return None if np.isnan(value) else float(value)


def per_pathology_auroc(run: ModelRun) -> dict[str, float | None]:
    """AUROC per pathology, over the entries this dataset annotated."""
    out: dict[str, float | None] = {}
    for k, name in enumerate(run.pathologies):
        keep = run.scorable()[:, k]
        out[name] = (
            auroc(run.probs[keep, k], run.ground_truth[keep, k] == 1)
            if keep.any()
            else None
        )
    return out


# ── 2. Calibration ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reliability:
    """Observed positive rate against stated probability, in equal-width bins."""

    bin_centers: np.ndarray
    observed: np.ndarray
    """Fraction actually positive in each bin; NaN where the bin is empty."""
    counts: np.ndarray
    expected_calibration_error: float
    """Mean |stated - observed| weighted by bin population. 0 = perfectly calibrated."""


def reliability(run: ModelRun, bins: int = 10) -> Reliability:
    keep = run.scorable()
    stated = run.probs[keep]
    actual = run.ground_truth[keep] == 1
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Rightmost bin is closed so a stated probability of exactly 1.0 is not dropped.
    index = np.clip(np.digitize(stated, edges[1:-1], right=False), 0, bins - 1)

    observed = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    gap = 0.0
    for b in range(bins):
        in_bin = index == b
        counts[b] = int(in_bin.sum())
        if counts[b]:
            observed[b] = float(actual[in_bin].mean())
            gap += counts[b] * abs(float(stated[in_bin].mean()) - observed[b])
    total = int(counts.sum())
    return Reliability(
        bin_centers=(edges[:-1] + edges[1:]) / 2.0,
        observed=observed,
        counts=counts,
        expected_calibration_error=gap / total if total else 0.0,
    )


# ── 3. Do they fail together? ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorOverlap:
    """How two models' mistakes coincide, over the entries both could be scored on."""

    a: str
    b: str
    both: int
    only_a: int
    only_b: int
    neither: int
    kappa: float | None
    """Cohen's κ on the two error indicators: 0 = mistakes fall independently, 1 = they
    err on exactly the same entries, < 0 = they err on complementary ones. The team
    argument lives here — complementarity needs κ low, and κ is a property of the pair,
    not of either model's accuracy."""

    @property
    def total(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither


def error_overlap(a: ModelRun, b: ModelRun, threshold: float = 0.5) -> ErrorOverlap:
    keep = a.scorable() & b.scorable()
    err_a = a.errors(threshold)[keep]
    err_b = b.errors(threshold)[keep]

    both = int((err_a & err_b).sum())
    only_a = int((err_a & ~err_b).sum())
    only_b = int((~err_a & err_b).sum())
    neither = int((~err_a & ~err_b).sum())
    n = both + only_a + only_b + neither

    kappa: float | None = None
    if n:
        observed = (both + neither) / n
        rate_a, rate_b = (both + only_a) / n, (both + only_b) / n
        chance = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
        # chance == 1 means at least one model never erred (or always did): agreement is
        # then guaranteed and κ says nothing, so report "not measurable" instead of ±inf.
        kappa = (observed - chance) / (1 - chance) if chance < 1.0 else None
    return ErrorOverlap(a.name, b.name, both, only_a, only_b, neither, kappa)


def all_overlaps(runs: Sequence[ModelRun], threshold: float = 0.5) -> list[ErrorOverlap]:
    return [
        error_overlap(runs[i], runs[j], threshold)
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]


# ── 4. Does uncertainty find the errors? ────────────────────────────────────────


def error_capture(run: ModelRun, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative errors found while reviewing entries from most to least uncertain.

    Returns the fraction reviewed and the fraction of all errors caught by then, so
    curves from models with different error counts are directly comparable. The diagonal
    is what an uninformative ordering achieves.
    """
    keep = run.scorable()
    doubt = run.uncertainty[keep]
    wrong = run.errors(threshold)[keep]
    n = int(keep.sum())
    if n == 0 or not wrong.any():
        return np.zeros(1), np.zeros(1)
    order = np.argsort(-doubt, kind="mergesort")
    caught = np.cumsum(wrong[order])
    reviewed = np.arange(1, n + 1, dtype=float) / n
    return reviewed, caught / caught[-1]


# ── Rendering ───────────────────────────────────────────────────────────────────


def _figure_header(fig: Any, title: str, subtitle: str) -> None:
    """Title and one line of context, spaced in inches rather than figure fractions.

    A fractional offset collapses on a short figure and the two lines collide; fixed
    physical spacing keeps the header identical whatever the figure's height.
    """
    height = fig.get_size_inches()[1]
    fig.text(0.01, 1 - 0.30 / height, title, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.01, 1 - 0.58 / height, subtitle, ha="left", va="top", fontsize=9, color=_MUTED)


def _frame(ax: Any) -> None:
    """Recede the chrome so the data is the only thing with weight."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, color=_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=_SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def render_discrimination(runs: Sequence[ModelRun], out_path: Path) -> Path:
    """Grouped bars: per-pathology AUROC, one bar per model."""
    import matplotlib.pyplot as plt

    scores = [per_pathology_auroc(r) for r in runs]
    labels = [
        name
        for name in runs[0].pathologies
        if any(s[name] is not None for s in scores)  # drop pathologies nobody could score
    ]
    # Width follows the number of pathologies, not the number of models: adding a model
    # should make the bars thinner, not stretch the figure off the page.
    fig, ax = plt.subplots(figsize=(max(9.0, 0.78 * len(labels)), 5.4))
    fig.patch.set_facecolor(_SURFACE)
    _frame(ax)

    width = 0.78 / max(len(runs), 1)
    positions = np.arange(len(labels), dtype=float)
    for m, (run, score) in enumerate(zip(runs, scores, strict=True)):
        values = [score[name] for name in labels]
        heights = [0.0 if v is None else v for v in values]
        offset = positions + (m - (len(runs) - 1) / 2) * width
        ax.bar(
            offset, heights, width * 0.88, label=run.name,
            color=SERIES_COLORS[m % len(SERIES_COLORS)], zorder=3,
        )
        # The palette's lighter slots sit under 3:1 on this surface, so every bar states
        # its own value rather than leaving hue to carry it.
        for x, value in zip(offset, values, strict=True):
            ax.text(
                x, (value or 0.0) + 0.012, "n/a" if value is None else f"{value:.2f}",
                ha="center", va="bottom", fontsize=6.2, rotation=90, color=_INK,
            )

    ax.axhline(0.5, color=_MUTED, ls=":", lw=1.2, zorder=2)
    # Anchored to the left edge in axes coordinates, so it never lands on a bar.
    ax.text(
        0.004, 0.5, "chance", transform=ax.get_yaxis_transform(),
        fontsize=7, color=_MUTED, va="bottom", ha="left",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.legend(fontsize=8, ncol=len(runs), loc="upper left", frameon=False)
    _figure_header(
        fig,
        "Discrimination per pathology",
        f"area under the ROC curve · {len(runs[0].image_ids)} shared images · "
        "'n/a' = one class absent",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out_path)


def render_reliability(runs: Sequence[ModelRun], out_path: Path, bins: int = 10) -> Path:
    """Stated probability against observed frequency — is a 0.8 really an 0.8?"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    fig.patch.set_facecolor(_SURFACE)
    _frame(ax)
    ax.plot([0, 1], [0, 1], color=_MUTED, ls=":", lw=1.3, zorder=2, label="perfect")

    for m, run in enumerate(runs):
        rel = reliability(run, bins)
        color = SERIES_COLORS[m % len(SERIES_COLORS)]
        seen = rel.counts > 0
        ax.plot(
            rel.bin_centers[seen], rel.observed[seen], color=color, lw=2.0,
            marker="o", ms=5.5, zorder=3,
            label=f"{run.name} (ECE {rel.expected_calibration_error:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("probability the model stated", fontsize=9)
    ax.set_ylabel("fraction actually positive", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    _figure_header(
        fig,
        "Calibration",
        "below the diagonal = over-confident · ECE = mean gap",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out_path)


def render_error_overlap(
    runs: Sequence[ModelRun], out_path: Path, threshold: float = 0.5
) -> Path:
    """Where two models' mistakes fall: together, separately, or not at all."""
    import matplotlib.pyplot as plt

    overlaps = all_overlaps(runs, threshold)
    if not overlaps:
        raise ValueError("comparing error overlap needs at least two models")
    index_of = {run.name: i for i, run in enumerate(runs)}

    fig, ax = plt.subplots(figsize=(10.5, 1.35 * len(overlaps) + 1.9))
    fig.patch.set_facecolor(_SURFACE)
    _frame(ax)
    ax.grid(False)

    for row, ov in enumerate(overlaps):
        total = max(ov.total, 1)
        # Each "only X" slice wears X's own series colour so a model is trackable across
        # the whole figure; the shared failure takes neutral ink because it belongs to
        # neither. Order is fixed — only-first, both, only-second, neither — so the
        # identity of a slice is carried by its POSITION as well as its colour, and the
        # figure stays readable in greyscale or with colour-vision deficiency.
        segments = [
            (ov.only_a / total, SERIES_COLORS[index_of[ov.a] % len(SERIES_COLORS)], ov.a),
            (ov.both / total, "#52514e", None),
            (ov.only_b / total, SERIES_COLORS[index_of[ov.b] % len(SERIES_COLORS)], ov.b),
            (ov.neither / total, "#e1e0d9", None),
        ]
        left = 0.0
        for fraction, color, owner in segments:
            ax.barh(row, fraction, left=left, height=0.42, color=color, zorder=3)
            middle = left + fraction / 2
            if fraction > 0.05:
                ax.text(
                    middle, row, f"{fraction:.0%}", ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if color == "#52514e" else _INK,
                )
            # Name the model that owns this slice above it, so a reader never has to
            # match a colour to a legend to know whose mistakes these are.
            if owner is not None:
                ax.text(
                    middle, row - 0.30, f"only {owner}", ha="center", va="bottom",
                    fontsize=7, color=_MUTED,
                )
            left += fraction
        kappa = "n/a" if ov.kappa is None else f"{ov.kappa:+.3f}"
        ax.text(1.015, row, f"κ {kappa}", va="center", fontsize=9, color=_INK)

    ax.set_yticks(range(len(overlaps)))
    ax.set_yticklabels([f"{o.a}\nvs {o.b}" for o in overlaps], fontsize=8)
    # Explicit, so a single pair renders as a bar rather than filling the frame.
    ax.set_ylim(len(overlaps) - 0.55, -0.62)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of scorable (image, pathology) entries", fontsize=9)
    from matplotlib.patches import Rectangle

    handles = [
        Rectangle((0, 0), 1, 1, color="#52514e"),
        Rectangle((0, 0), 1, 1, color="#e1e0d9"),
    ]
    ax.legend(
        handles, ["both wrong", "both right"],
        fontsize=8, loc="upper right", frameon=False, ncol=2,
        bbox_to_anchor=(1.0, 1.14),
    )
    _figure_header(
        fig,
        "Do they fail on the same cases?",
        "κ near 0 = mistakes fall independently, so pairing can help · "
        "κ high = they fail together, so it cannot",
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.86))
    return _save(fig, out_path)


def render_error_capture(
    runs: Sequence[ModelRun], out_path: Path, threshold: float = 0.5
) -> Path:
    """Does the model's own uncertainty rank its mistakes to the front?"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    fig.patch.set_facecolor(_SURFACE)
    _frame(ax)
    ax.plot([0, 1], [0, 1], color=_MUTED, ls=":", lw=1.3, zorder=2, label="uninformative")

    for m, run in enumerate(runs):
        reviewed, caught = error_capture(run, threshold)
        color = SERIES_COLORS[m % len(SERIES_COLORS)]
        # Area above the diagonal, normalised: 0 = no better than reviewing at random.
        skill = float(np.trapezoid(caught, reviewed) * 2.0 - 1.0) if len(reviewed) > 1 else 0.0
        ax.plot(reviewed, caught, color=color, lw=2.0, zorder=3,
                label=f"{run.name} (skill {skill:+.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of entries reviewed (most uncertain first)", fontsize=9)
    ax.set_ylabel("fraction of the model's errors found", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _figure_header(
        fig,
        "Does uncertainty find the errors?",
        "above the diagonal = the model knows where it is weak",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out_path)


def render_all(
    runs: Sequence[ModelRun], out_dir: Path, threshold: float = 0.5
) -> list[Path]:
    """Every comparison figure, on aligned runs. Returns the paths written."""
    aligned = align_runs(runs)
    written = [
        render_discrimination(aligned, out_dir / "discrimination.png"),
        render_reliability(aligned, out_dir / "calibration.png"),
        render_error_capture(aligned, out_dir / "error_capture.png", threshold),
    ]
    if len(aligned) > 1:
        written.append(
            render_error_overlap(aligned, out_dir / "error_overlap.png", threshold)
        )
    return written
