"""Tests for the RQ1 uncertainty-study harness: composition and threshold policies."""

from __future__ import annotations

import numpy as np

from hai_sim.analysis.inference_cache import InferenceCache
from hai_sim.analysis.uncertainty_study import (
    _cost_optimal_threshold,
    _prevalence_threshold,
    compose_member_caches,
    confusion_at,
    per_pathology_thresholds,
)


def _single_member_cache(probs_1col: np.ndarray, gt_1col: np.ndarray, weight: str) -> InferenceCache:
    n = probs_1col.shape[0]
    return InferenceCache(
        member_probs=probs_1col.reshape(n, 1, 1).astype(np.float32),
        ground_truth=gt_1col.reshape(n, 1).astype(np.int8),
        inference_seconds=np.zeros(n),
        image_ids=tuple(f"img{i}" for i in range(n)),
        dataset_indices=np.arange(n),
        weights=(weight,),
        pathologies=("Effusion",),
        dataset="toy",
        batch_size=1,
    )


def test_compose_stacks_members_and_aligns_ids(tmp_path) -> None:
    a = _single_member_cache(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 1]), "densenet121-res224-chex")
    b = _single_member_cache(np.array([0.4, 0.5, 0.6]), np.array([0, 1, 1]), "densenet121-res224-pc")
    a.save(tmp_path / "toy__chex")
    b.save(tmp_path / "toy__pc")

    ens = compose_member_caches(
        [str(tmp_path / "toy__chex"), str(tmp_path / "toy__pc")], dataset_name="toy"
    )
    assert ens.member_probs.shape == (3, 2, 1)
    assert np.allclose(ens.member_probs[:, 0, 0], [0.1, 0.2, 0.3])
    assert np.allclose(ens.member_probs[:, 1, 0], [0.4, 0.5, 0.6])
    assert ens.weights == ("densenet121-res224-chex", "densenet121-res224-pc")
    assert np.array_equal(ens.ground_truth[:, 0], [0, 1, 1])


def test_compose_keeps_only_common_ids(tmp_path) -> None:
    a = _single_member_cache(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 1]), "w-a")
    b = _single_member_cache(np.array([0.4, 0.5]), np.array([0, 1]), "w-b")  # only img0, img1
    a.save(tmp_path / "d__a")
    b.save(tmp_path / "d__b")
    ens = compose_member_caches([str(tmp_path / "d__a"), str(tmp_path / "d__b")])
    assert ens.n_images == 2  # img2 dropped (not in member b)
    assert ens.image_ids == ("img0", "img1")


def test_cost_optimal_threshold_matches_worked_example() -> None:
    # The 10-lastre example from the design discussion: with FN:FP = 100:10 the optimum is
    # "call the top 6 positive" (catch every positive, tolerate 3 false alarms).
    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.40, 0.25, 0.15, 0.05])
    truth = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    tau = _cost_optimal_threshold(scores, truth, c_fp=10.0, c_fn=100.0)
    pred = scores >= tau
    assert int(pred.sum()) == 6  # top 6 called positive
    assert set(np.where(pred)[0]) == {0, 1, 2, 3, 4, 5}
    fn = int(((~pred) & (truth == 1)).sum())
    assert fn == 0  # every real positive is caught


def test_prevalence_threshold_gives_fp_equals_fn() -> None:
    # Matching the predicted-positive count to the prevalence forces false alarms == misses.
    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.40, 0.25, 0.15, 0.05])
    truth = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # prevalence 3/10
    tau = _prevalence_threshold(scores, truth)
    pred = scores >= tau
    assert int(pred.sum()) == 3  # predicted-positive count == number of real positives
    fp = int((pred & (truth == 0)).sum())
    fn = int(((~pred) & (truth == 1)).sum())
    assert fp == fn


def test_per_pathology_thresholds_fixed_is_constant() -> None:
    mean_p = np.random.default_rng(0).random((20, 3))
    gt = np.random.default_rng(1).integers(0, 2, size=(20, 3)).astype(np.int8)
    tau = per_pathology_thresholds(mean_p, gt, policy="fixed", value=0.5)
    assert tau.shape == (3,)
    assert np.allclose(tau, 0.5)


def test_confusion_counts_and_cost() -> None:
    mean_p = np.array([[0.9], [0.4], [0.8], [0.2]])
    gt = np.array([[1], [1], [0], [0]], dtype=np.int8)  # entries: TP, FN, FP, TN at thr 0.5
    conf = confusion_at(mean_p, gt, np.array([0.5]), c_fp=10.0, c_fn=100.0)
    assert (conf.tp, conf.fn, conf.fp, conf.tn) == (1, 1, 1, 1)
    assert conf.cost == 10.0 * 1 + 100.0 * 1
    assert conf.n == 4


def test_confusion_ignores_unannotated_entries() -> None:
    mean_p = np.array([[0.9], [0.9], [0.2]])
    gt = np.array([[1], [-1], [0]], dtype=np.int8)  # middle entry not annotated -> excluded
    conf = confusion_at(mean_p, gt, np.array([0.5]))
    assert conf.n == 2
    assert (conf.tp, conf.tn) == (1, 1)


# ── Per-pathology validation views ───────────────────────────────────────────────


def _toy_ensemble_cache(n: int = 200, seed: int = 0) -> InferenceCache:
    """Two members, two pathologies; member probs correlated with truth + noise."""
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    probs = np.empty((n, 2, 2), dtype=np.float32)
    for m in range(2):
        noise = rng.normal(0.0, 0.25, size=(n, 2))
        probs[:, m, :] = np.clip(0.2 + 0.6 * gt + noise, 0.0, 1.0)
    return InferenceCache(
        member_probs=probs,
        ground_truth=gt,
        inference_seconds=np.zeros(n),
        image_ids=tuple(f"img{i}" for i in range(n)),
        dataset_indices=np.arange(n),
        weights=("w-a", "w-b"),
        pathologies=("Effusion", "Nodule"),
        dataset="toy",
        batch_size=1,
    )


def test_per_pathology_table_shape_and_fields() -> None:
    from hai_sim.analysis.uncertainty_study import per_pathology_table

    cache = _toy_ensemble_cache()
    rows = per_pathology_table(
        cache, thresholds=np.array([0.5, 0.5]), measures=("predictive_entropy", "std")
    )
    # 2 pathologies x 2 measures x 3 error types
    assert len(rows) == 12
    row = rows[0]
    for field in ("dataset", "pathology", "measure", "error_type", "skill",
                  "auroc_unc", "auroc_disc", "prevalence", "n_errors", "low_support"):
        assert field in row
    # discrimination should be well above chance on this construction
    assert row["auroc_disc"] > 0.7


def test_calibration_panel_platt_reduces_ece_on_biased_probs() -> None:
    from hai_sim.analysis.uncertainty_study import calibration_panel

    cache = _toy_ensemble_cache(n=400)
    # Bias every member upward: raw ECE becomes large, Platt should fix most of it.
    cache.member_probs[:] = np.clip(cache.member_probs * 0.4 + 0.55, 0.0, 1.0)
    rows, pooled = calibration_panel(cache)
    assert len(rows) == 2
    for r in rows:
        assert r["ece_platt"] <= r["ece_raw"] + 0.02
    assert pooled["raw"]["confidence"] and pooled["platt"]["confidence"]


def test_cost_ratio_sweep_thresholds_fall_as_misses_get_expensive() -> None:
    from hai_sim.analysis.uncertainty_study import cost_ratio_sweep

    cache = _toy_ensemble_cache(n=400)
    rows = cost_ratio_sweep(cache, ratios=(1.0, 10.0, 50.0))
    pooled = {r["ratio"]: r for r in rows if r["pathology"] == "__pooled__"}
    assert set(pooled) == {1.0, 10.0, 50.0}
    # More expensive misses -> cast a wider net -> sensitivity must not decrease.
    assert pooled[50.0]["sensitivity"] >= pooled[1.0]["sensitivity"] - 1e-9
    per_label = [r for r in rows if r["pathology"] == "Effusion"]
    taus = {r["ratio"]: r["threshold"] for r in per_label}
    assert taus[50.0] <= taus[1.0] + 1e-9


def test_member_error_overlap_kappa_bounds_and_self_consistency() -> None:
    from hai_sim.analysis.uncertainty_study import member_error_overlap

    cache = _toy_ensemble_cache(n=400)
    rows = member_error_overlap(cache)
    assert rows, "two members over 400 images must produce at least one pair row"
    for r in rows:
        assert -1.0 <= r["kappa"] <= 1.0
        assert r["pair"] == "w-axw-b"
    # Identical members must err identically: kappa == 1 on every pathology.
    cache.member_probs[:, 1, :] = cache.member_probs[:, 0, :]
    rows_same = member_error_overlap(cache)
    assert all(abs(r["kappa"] - 1.0) < 1e-9 for r in rows_same)


def test_difficulty_alignment_recovers_planted_correlation(tmp_path) -> None:
    from hai_sim.analysis.uncertainty_study import difficulty_alignment

    cache = _toy_ensemble_cache(n=300, seed=3)
    # Plant difficulty proportional to the model's own predictive entropy -> the
    # Spearman correlation must come out strongly positive.
    from hai_sim.analysis.inference_cache import measures_from_cache

    ent = measures_from_cache(cache.member_probs)["predictive_entropy"].mean(axis=1)
    table = tmp_path / "difficulty.csv"
    lines = ["example_id,delta"]
    lines += [f"{img},{ent[i]:.6f}" for i, img in enumerate(cache.image_ids)]
    table.write_text("\n".join(lines), encoding="utf-8")

    rows, extras = difficulty_alignment(cache, table, value_column="delta")
    assert extras["coverage"]["n_covered"] == 300
    assert len(rows) == 2
    for r in rows:
        assert r["spearman_predictive_entropy"] > 0.5


def test_run_study_writes_per_pathology_artifacts(tmp_path) -> None:
    from hai_sim.analysis.uncertainty_study import run_study

    a = _toy_ensemble_cache(n=120, seed=1)
    b = _toy_ensemble_cache(n=120, seed=2)
    for ds, cache in (("dsa", a), ("dsb", b)):
        cache.dataset = ds
        # split members into two single-member cells, as on disk
        for j, w in enumerate(cache.weights):
            cell = InferenceCache(
                member_probs=cache.member_probs[:, j : j + 1, :],
                ground_truth=cache.ground_truth,
                inference_seconds=cache.inference_seconds,
                image_ids=cache.image_ids,
                dataset_indices=cache.dataset_indices,
                weights=(w,),
                pathologies=cache.pathologies,
                dataset=ds,
                batch_size=1,
            )
            cell.save(tmp_path / f"{ds}__{w}")

    out = tmp_path / "out"
    cells = {
        "dsa": [str(tmp_path / "dsa__w-a"), str(tmp_path / "dsa__w-b")],
        "dsb": [str(tmp_path / "dsb__w-a"), str(tmp_path / "dsb__w-b")],
    }
    run_study(cells, out, measures=("predictive_entropy", "std"))
    for name in ("per_pathology_measures.csv", "calibration_panel.csv",
                 "cost_ratio_sweep.csv", "member_overlap.csv",
                 "pathology_in_vs_ood.csv"):
        assert (out / name).exists(), name
    for ds in ("dsa", "dsb"):
        for fig in ("per_pathology_measures.png", "calibration_panel.png",
                    "cost_ratio_sweep.png", "member_overlap.png"):
            assert (out / ds / fig).exists(), f"{ds}/{fig}"
    assert (out / "pathology_in_vs_ood.png").exists()
