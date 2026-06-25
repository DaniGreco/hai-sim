"""UQ methods: entropy and the deep-ensemble decomposition."""

from __future__ import annotations

import numpy as np

from hai_sim.uq.methods import EnsembleUQ, EntropyUQ


def test_entropy_single_pass() -> None:
    probs, unc, extra = EntropyUQ().estimate(np.array([[0.5, 0.1, 0.9]]))
    assert np.allclose(probs, [0.5, 0.1, 0.9])
    # p=0.5 is maximally uncertain; 0.1 and 0.9 are symmetric and lower.
    assert unc[0] > unc[1]
    assert np.isclose(unc[1], unc[2])
    assert extra == {}


def test_entropy_accepts_1d() -> None:
    probs, unc, _ = EntropyUQ().estimate(np.array([0.5, 0.5]))
    assert probs.shape == (2,)
    assert unc.shape == (2,)


def test_ensemble_mi_zero_when_members_agree() -> None:
    members = np.array([[0.7, 0.2], [0.7, 0.2], [0.7, 0.2]])
    probs, mi, extra = EnsembleUQ("mutual_information").estimate(members)
    assert np.allclose(probs, [0.7, 0.2])
    assert np.allclose(mi, 0.0, atol=1e-9)
    assert extra["n_members"] == 3


def test_ensemble_mi_positive_when_members_disagree() -> None:
    members = np.array([[0.99, 0.01], [0.01, 0.99]])
    _, mi, _ = EnsembleUQ("mutual_information").estimate(members)
    assert np.all(np.asarray(mi) > 0.0)


def test_ensemble_decomposition_identity() -> None:
    members = np.array([[0.8, 0.3, 0.5], [0.4, 0.6, 0.5]])
    _, _, extra = EnsembleUQ("predictive_entropy").estimate(members)
    pe = np.array(extra["predictive_entropy"])
    me = np.array(extra["mean_entropy"])
    mi = np.array(extra["mutual_information"])
    # predictive entropy = aleatoric (mean entropy) + epistemic (mutual info)
    assert np.allclose(pe - me, mi, atol=1e-9)
    assert np.all(mi >= -1e-12)
