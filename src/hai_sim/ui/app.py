"""The result browser, as a Streamlit page. Widgets and layout only.

Launch it with ``hai-sim ui`` (or ``streamlit run src/hai_sim/ui/app.py``). Every
reshaping decision lives in `hai_sim.ui.results`, which has no Streamlit import and is
tested in the normal suite; this file should stay boring enough that nothing in it is
worth asserting on.

Four tabs, and each answers a question the terminal answers badly:

**Explore**   what did this sweep do? — pick a metric and two factors, get a chart
**Contrasts** did an arm actually beat the baseline? — a forest plot with intervals,
              paired at the seed, so the eye reads "does the interval clear zero"
**Signals**   what did the new signal plane produce? — per-read distributions of
              difficulty, fatigue and load, which nothing else visualises
**Compare**   two runs disagree — *what was different about them?* — a config diff
              beside the metric deltas

There is no Run button. The browser reads; it never simulates and never writes into a
run's output. A result is reproducible because a YAML and a seed produced it, and this
page's job is to keep showing you that YAML next to its numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hai_sim.ui.results import (
    Artifact,
    common_metrics,
    config_diff,
    contrast_rows,
    discover,
    factor_names,
    load_records,
    metric_names,
    numeric_column,
    observation_keys,
    run_group_rows,
    run_kpi_rows,
    study_rows,
    summary_rows,
)

st.set_page_config(page_title="hai_sim results", layout="wide")


@st.cache_data(show_spinner=False)
def _discover(out_dir: str, _token: float) -> list[Artifact]:
    """``_token`` is a cache buster: pass the directory mtime so new runs appear."""
    return discover(out_dir)


def _artifacts(out_dir: str) -> list[Artifact]:
    root = Path(out_dir)
    token = root.stat().st_mtime if root.is_dir() else 0.0
    return _discover(out_dir, token)


def _pick(label: str, artifacts: list[Artifact], key: str) -> Artifact | None:
    if not artifacts:
        return None
    chosen = st.selectbox(label, artifacts, format_func=lambda a: a.label, key=key)
    return chosen


# --- sidebar ----------------------------------------------------------------

st.sidebar.title("hai_sim")
st.sidebar.caption("Result browser — reads only, never runs.")
out_dir = st.sidebar.text_input("Output directory", value="out")
artifacts = _artifacts(out_dir)
studies = [a for a in artifacts if a.kind == "study"]
runs = [a for a in artifacts if a.kind == "run"]
st.sidebar.metric("Studies", len(studies))
st.sidebar.metric("Single runs", len(runs))
if st.sidebar.button("Rescan"):
    _discover.clear()
    st.rerun()

if not artifacts:
    st.warning(
        f"No readable results under `{out_dir}`. Produce one with "
        "`hai-sim run configs/examples/v0_smoke_hybrid.yaml`, or point at another "
        "directory in the sidebar."
    )
    st.stop()

explore_tab, contrast_tab, signal_tab, compare_tab = st.tabs(
    ["Explore", "Contrasts", "Signals", "Compare"]
)

# --- Explore ----------------------------------------------------------------

with explore_tab:
    st.subheader("What did this sweep do?")
    if not studies:
        st.info("No study artefacts yet. Run one with `hai-sim study <study.yaml>`.")
    else:
        study = _pick("Study", studies, "explore_study")
        assert study is not None
        rows = study_rows(study)
        factors = factor_names(study)
        metrics = metric_names(study)
        frame = pd.DataFrame(rows)

        if not metrics:
            st.warning("This study reported no scalar metrics.")
        else:
            c1, c2, c3 = st.columns(3)
            metric = c1.selectbox("Metric", metrics, key="explore_metric")
            x_axis = c2.selectbox("X axis", factors or ["seed"], key="explore_x")
            colour_options = ["(none)"] + [f for f in factors if f != x_axis]
            colour = c3.selectbox("Colour by", colour_options, key="explore_colour")

            groups = sorted({str(r.get("group", "")) for r in rows})
            if len(groups) > 1:
                group = st.selectbox("Conditioning subgroup", groups, key="explore_group")
                frame = frame[frame["group"].astype(str) == group]

            # Mean over replications: one point per cell, which is what a factor plot
            # is asking for. The spread across seeds is the Contrasts tab's job.
            index = [x_axis] + ([colour] if colour != "(none)" else [])
            pivot = frame.groupby(index)[metric].mean().reset_index()
            chart = (
                pivot.pivot(index=x_axis, columns=colour, values=metric)
                if colour != "(none)"
                else pivot.set_index(x_axis)[metric]
            )
            st.bar_chart(chart)
            st.caption(
                f"Mean of `{metric}` over {len(set(frame.get('seed', [0])))} replication(s). "
                "Averaging seeds here is deliberate — the per-seed spread is what the "
                "Contrasts tab uses, and mixing the two views hides the pairing."
            )

            with st.expander("Rows"):
                st.dataframe(frame, width="stretch")
            with st.expander("Per-cell summary (mean, sd, 95% interval)"):
                st.dataframe(pd.DataFrame(summary_rows(study)), width="stretch")
            with st.expander("The study configuration that produced this"):
                st.code(json.dumps(study.payload.get("study", {}), indent=2), language="json")

# --- Contrasts --------------------------------------------------------------

with contrast_tab:
    st.subheader("Did the arm actually beat the baseline?")
    st.caption(
        "Paired at the seed: each delta is the mean *per-seed* difference, not the "
        "difference of the means. That is what the common random numbers buy — the "
        "sampling noise cancels rather than being averaged over, so the interval is "
        "narrow enough to mean something. Read it as: does the bar clear zero?"
    )
    contrast_studies = [s for s in studies if contrast_rows(s)]
    if not contrast_studies:
        st.info(
            "No study here declares a `baseline:`. Add one to the study YAML and the "
            "runner computes paired contrasts against that arm."
        )
    else:
        study = _pick("Study", contrast_studies, "contrast_study")
        assert study is not None
        rows = contrast_rows(study)
        frame = pd.DataFrame(rows)
        metrics = sorted({str(r["metric"]) for r in rows})
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Metric", metrics, key="contrast_metric")
        groups = sorted({str(r.get("group", "")) for r in rows})
        group = c2.selectbox("Subgroup", groups, key="contrast_group") if len(groups) > 1 else groups[0]

        subset = frame[(frame["metric"] == metric) & (frame["group"].astype(str) == group)]
        if subset.empty:
            st.warning("Nothing for that combination.")
        else:
            st.caption(f"Baseline: **{subset['baseline'].iloc[0]}**")
            for _, row in subset.iterrows():
                delta, low, high = row["delta_mean"], row["delta_ci_low"], row["delta_ci_high"]
                clears = bool(row["excludes_zero"])
                mark = "🟩" if clears and delta > 0 else "🟥" if clears else "⬜"
                arm_col, delta_col = st.columns([1, 3])
                arm_col.markdown(f"{mark} **{row['arm']}**")
                delta_col.markdown(
                    f"Δ = `{delta:+.4f}`  [{low:+.4f}, {high:+.4f}]  "
                    f"· n_pairs = {row['n_pairs']}"
                    + ("" if clears else "  — interval contains zero")
                )
            st.bar_chart(subset.set_index("arm")["delta_mean"])
            with st.expander("All contrasts"):
                st.dataframe(frame, width="stretch")

# --- Signals ----------------------------------------------------------------

with signal_tab:
    st.subheader("What did the signal plane produce?")
    st.caption(
        "Per-read scalars the engine computes between the two stations: how hard the "
        "case was, how worn the reader was, how deep the queue got. Nothing else "
        "visualises these, and a difficulty distribution that turns out to be nearly "
        "constant means a coupling study measured nothing."
    )
    if not runs:
        st.info("No single-run artefacts yet.")
    else:
        run = _pick("Run", runs, "signal_run")
        assert run is not None
        meta = run.payload.get("metadata", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Difficulty model", str(meta.get("difficulty_model", "none")))
        c2.metric("Fatigue curve", str(meta.get("fatigue_curve", "none")))
        c3.metric("Severity model", str(meta.get("severity_model", "none")))

        records = load_records(run)
        if not records:
            st.warning(
                "The per-patient records CSV is not next to this summary, so there is "
                "nothing per-read to plot. The summary below is still complete."
            )
        else:
            keys = observation_keys(records, prefix="sig.") + observation_keys(
                records, prefix="load."
            )
            if not keys:
                st.info(
                    "This run produced no signal-plane or load columns — it predates "
                    "them, or configured neither a difficulty model nor a fatigue curve."
                )
            else:
                chosen = st.multiselect(
                    "Columns", keys, default=keys[: min(2, len(keys))], key="signal_keys"
                )
                for key in chosen:
                    values = numeric_column(records, key)
                    if not values:
                        continue
                    series = pd.Series(values, name=key)
                    st.markdown(f"**{key}** — n={len(values)}")
                    counts, edges = pd.cut(series, bins=20, retbins=True)
                    st.bar_chart(
                        pd.DataFrame(
                            {key: counts.value_counts(sort=False).to_numpy()},
                            index=[f"{e:.3f}" for e in edges[:-1]],
                        )
                    )
                    st.caption(
                        f"mean {series.mean():.4f} · median {series.median():.4f} · "
                        f"min {series.min():.4f} · max {series.max():.4f} · "
                        f"distinct {series.nunique()}"
                        + (
                            "  ⚠️ one distinct value — this signal is constant, so "
                            "anything conditioned on it is conditioned on nothing."
                            if series.nunique() == 1
                            else ""
                        )
                    )

        with st.expander("KPI report"):
            st.dataframe(
                pd.DataFrame(run_kpi_rows(run), columns=["metric", "value"]),
                width="stretch",
            )
        by_group = run_group_rows(run)
        if by_group:
            with st.expander(f"KPIs by subgroup ({len(by_group)} groups)"):
                table = pd.DataFrame(
                    {name: dict(pairs) for name, pairs in by_group.items()}
                )
                st.dataframe(table, width="stretch")

# --- Compare ----------------------------------------------------------------

with compare_tab:
    st.subheader("Two runs disagree — what was different about them?")
    st.caption(
        "Reading two YAML files side by side answers this badly: they are long and "
        "mostly identical. This lists only the keys that differ, including blocks "
        "present in one and absent in the other."
    )
    if len(artifacts) < 2:
        st.info("Need at least two artefacts to compare.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            left = _pick("Left", artifacts, "cmp_left")
        with c2:
            right = _pick("Right", artifacts, "cmp_right")
        assert left is not None and right is not None

        if left.path == right.path:
            st.info("Pick two different artefacts.")
        else:
            deltas = config_diff(left.scenario, right.scenario)
            st.markdown(f"### Configuration — {len(deltas)} key(s) differ")
            if not deltas:
                st.success(
                    "Identical configurations. Any difference in the numbers is the "
                    "seed, or something outside the scenario file."
                )
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "path": d.path,
                                "status": d.status,
                                "left": str(d.left),
                                "right": str(d.right),
                            }
                            for d in deltas
                        ]
                    ),
                    width="stretch",
                )

            shared = common_metrics([left, right])
            st.markdown(f"### Metrics — {len(shared)} in common")
            if not shared:
                st.warning(
                    "No metric names in common: these two reported different things, "
                    "so there is nothing to line up."
                )
            elif left.kind == right.kind == "run":
                lhs, rhs = dict(run_kpi_rows(left)), dict(run_kpi_rows(right))
                table = pd.DataFrame(
                    [
                        {
                            "metric": m,
                            "left": lhs[m],
                            "right": rhs[m],
                            "delta": rhs[m] - lhs[m],
                        }
                        for m in shared
                    ]
                )
                st.dataframe(
                    table.sort_values("delta", key=abs, ascending=False), width="stretch"
                )
                st.caption(
                    "Single runs at one seed each: a delta here includes sampling noise "
                    "and is not a paired contrast. Use a study with a `baseline:` when "
                    "the size of a difference matters."
                )
            else:
                st.info(
                    "Metric-level comparison is implemented for two single runs. For "
                    "sweeps, the Contrasts tab compares arms *within* a study, which is "
                    "the paired comparison worth making."
                )
