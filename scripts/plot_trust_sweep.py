"""Figures for the SDT trust sweep (study_sdt_trust_sweep.yaml).

Reads the study's tidy CSV and renders two publication figures:

    fig_trust_curves   the inverted-U reliance curve — team F1 vs how much the doctor
                       leans on the AI, one small multiple per AI quality, one line per
                       doctor skill. The headline: an interior optimum, whose height
                       scales inversely with the doctor's own skill, collapsing to the
                       AI's own accuracy under blind trust.
    fig_failure_modes  the two ways it goes wrong — a junior's under-reliance falling and
                       a senior's over-reliance rising as trust grows (they cross), and
                       the clinical cost as missed findings vs trust by skill.

Usage:
    python scripts/plot_trust_sweep.py [tidy_csv] [--out-dir out]

Colour follows the data's job (dataviz method): doctor skill is an *ordered* factor, so
it rides a single blue ramp (light = junior, dark = senior); the two failure-mode rates
are distinct categories, so they take the first two categorical hues. Both palettes were
validated against the light surface.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── palette (dataviz reference, light surface; validated) ────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# doctor skill = ordered → one blue ramp, light→dark = junior→senior (ordinal steps 250/400/550)
SKILL_COLOR = {"junior": "#86b6ef", "mid": "#3987e5", "senior": "#1c5cab"}
# failure-mode rates = distinct categories → categorical slots 1 (blue) and 2 (orange)
C_UNDER = "#2a78d6"
C_OVER = "#eb6834"

AI_LEVELS = [(2.56, "Good AI\n(acc ≈ 0.90)"), (3.29, "Very good AI\n(≈ 0.95)"),
             (4.0, "Excellent AI\n(≈ 0.98)")]
DOC_LEVELS = [(1.68, "junior"), (2.56, "mid"), (3.29, "senior")]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "text.color": INK,
        "axes.titlecolor": INK, "font.size": 11,
    })


def load(path: Path) -> dict:
    """Mean over seeds, keyed (ai_dprime, doc_dprime, weight) → {metric: value}."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(path.open(encoding="utf-8")):
        key = (round(float(r["ai.d_prime"]), 2), round(float(r["persona.capability.d_prime"]), 2),
               round(float(r["persona.trust.weight_ai"]), 2))
        for m, col in [("f1", "actors.hybrid.f1_macro"), ("ai_f1", "actors.ai.f1_macro"),
                       ("n_fn", "actors.hybrid.n_fn"), ("over", "team.over_reliance"),
                       ("under", "team.under_reliance")]:
            v = r.get(col, "")
            if v not in ("", "None"):
                acc[key][m].append(float(v))
    return {k: {m: sum(xs) / len(xs) for m, xs in d.items()} for k, d in acc.items()}


def _grid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def fig_trust_curves(data: dict, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, (ai_d, ai_lab) in zip(axes, AI_LEVELS, strict=True):
        _grid(ax)
        ai_f1 = data[(ai_d, 2.56, 0.5)]["ai_f1"]  # AI-only F1 (doctor/trust-invariant)
        ax.axhline(ai_f1, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
        ax.text(1.0, ai_f1, "  AI alone", color=MUTED, va="center", ha="left", fontsize=9)
        for doc_d, doc_lab in DOC_LEVELS:
            ys = [data[(ai_d, doc_d, w)]["f1"] for w in WEIGHTS]
            c = SKILL_COLOR[doc_lab]
            ax.plot(WEIGHTS, ys, color=c, lw=2.3, marker="o", ms=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
            peak_i = max(range(len(ys)), key=lambda i: ys[i])  # ring the interior optimum
            ax.plot(WEIGHTS[peak_i], ys[peak_i], marker="o", ms=9, mfc="none",
                    mec=c, mew=1.8, zorder=4)
            ax.annotate(doc_lab, (WEIGHTS[0], ys[0]), textcoords="offset points",
                        xytext=(-6, 0), ha="right", va="center", color=c, fontsize=9.5,
                        fontweight="bold")
        ax.set_title(ai_lab, fontsize=11, pad=8)
        ax.set_xlabel("Weight given to the AI", fontsize=10)
        ax.set_xticks(WEIGHTS)
        ax.set_xlim(-0.16, 1.06)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Team accuracy  (F1, macro)", fontsize=10)
    axes[0].set_ylim(0.62, 1.0)
    fig.suptitle("A little trust helps, blind trust hurts — and the gain shrinks as the doctor gets better",
                 fontsize=13, fontweight="bold", x=0.5, y=1.0)
    fig.text(0.5, 0.905, "team accuracy vs. how much the doctor leans on the AI · circle = optimum · "
             "at full trust every doctor collapses onto the AI's own accuracy",
             ha="center", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = out / "fig_trust_curves.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def fig_failure_modes(data: dict, out: Path) -> Path:
    _style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ai_d = 2.56  # the "good AI" slice

    # Left: opposite failure modes — junior under-reliance vs senior over-reliance.
    _grid(axL)
    under = [data[(ai_d, 1.68, w)]["under"] for w in WEIGHTS]  # junior ignores a right AI
    over = [data[(ai_d, 3.29, w)]["over"] for w in WEIGHTS]    # senior follows a wrong AI
    axL.plot(WEIGHTS, under, color=C_UNDER, lw=2.3, marker="o", ms=5.5,
             markeredgecolor=SURFACE, markeredgewidth=1.1)
    axL.plot(WEIGHTS, over, color=C_OVER, lw=2.3, marker="o", ms=5.5,
             markeredgecolor=SURFACE, markeredgewidth=1.1)
    axL.annotate("Under-reliance\n(junior ignores a correct AI)", (WEIGHTS[1], under[1]),
                 textcoords="offset points", xytext=(10, 14), color=C_UNDER, fontsize=9.5, fontweight="bold")
    axL.annotate("Over-reliance\n(senior follows a wrong AI)", (WEIGHTS[3], over[3]),
                 textcoords="offset points", xytext=(-6, -34), ha="right", color=C_OVER,
                 fontsize=9.5, fontweight="bold")
    axL.set_title("Trust fixes one reader and breaks the other", fontsize=11.5, fontweight="bold", pad=8)
    axL.set_ylabel("Share of chances handled wrong", fontsize=10)
    axL.set_xlabel("Weight given to the AI", fontsize=10)
    axL.set_xticks(WEIGHTS)
    axL.set_xlim(-0.05, 1.05)
    axL.set_ylim(-0.03, 1.03)
    axL.tick_params(length=0)

    # Right: clinical cost — missed findings vs trust, by skill.
    _grid(axR)
    for doc_d, doc_lab in DOC_LEVELS:
        ys = [data[(ai_d, doc_d, w)]["n_fn"] for w in WEIGHTS]
        c = SKILL_COLOR[doc_lab]
        axR.plot(WEIGHTS, ys, color=c, lw=2.3, marker="o", ms=5.5,
                 markeredgecolor=SURFACE, markeredgewidth=1.1)
        # Label at the w=0 end, where the three skills are far apart — at full trust the
        # lines converge on the AI's own FN count and any label there would collide.
        axR.annotate(doc_lab, (WEIGHTS[0], ys[0]), textcoords="offset points",
                     xytext=(-8, 0), ha="right", va="center", color=c, fontsize=9.5,
                     fontweight="bold")
    axR.set_title("Over-trust makes a good doctor miss more", fontsize=11.5, fontweight="bold", pad=8)
    axR.set_ylabel("Missed findings  (false negatives / 2000 cases)", fontsize=10)
    axR.set_xlabel("Weight given to the AI", fontsize=10)
    axR.set_xticks(WEIGHTS)
    axR.set_xlim(-0.22, 1.05)
    axR.tick_params(length=0)

    fig.suptitle("Two failure modes of trust, with a good AI (accuracy ≈ 0.90)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "fig_failure_modes.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", help="tidy study CSV (default: latest in --out-dir)")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.csv:
        csv_path = Path(args.csv)
    else:
        cands = sorted(out.glob("study_sdt_trust_sweep_*[0-9].csv"))
        if not cands:
            raise SystemExit("no study_sdt_trust_sweep_*.csv in out/; run the study first")
        csv_path = cands[-1]
    data = load(csv_path)
    print(f"loaded {csv_path.name}")
    print("wrote", fig_trust_curves(data, out))
    print("wrote", fig_failure_modes(data, out))


if __name__ == "__main__":
    main()
