"""How "how sure are you?" is computed. UQ = uncertainty quantification.

The number these produce is what triage sorts by, so what it *means* decides what
"hardest case" means in a study. Both take a ``(n_members, n_labels)`` stack of
probabilities — one row per model, one column per pathology — and return a probability
and an uncertainty per label.

``EntropyUQ`` — one model, one forward pass
    Uncertainty = the entropy of the predicted probability: highest at p = 0.5, zero at
    p = 0 or 1. It measures only *how close to the fence this model sat*, so a single
    confidently wrong model reports low uncertainty. Cheap, and blind to its own
    ignorance.

``EnsembleUQ`` — several independently trained models (a **Deep Ensemble**)
    Uncertainty comes from **disagreement between models**, which lets it separate two
    things that are not the same problem:

        predictive entropy   = total uncertainty (entropy of the averaged probability)
        mean entropy         = ALEATORIC: ambiguity in the image itself. Every model is
                               individually unsure, and they all agree about it. More
                               training data would not help.
        mutual information   = EPISTEMIC: the models *disagree*. The image is unlike
                               what they were trained on. This is the one that should
                               flag the cases where the model is out of its depth —
                               = predictive entropy - mean entropy.
        std                  = raw spread of the members' probabilities.

    One of the four is selected as *the* uncertainty for triage; all four are returned
    in ``extra`` and reach the telemetry, so a study can compare them as triage keys
    without rerunning anything.

Both are NaN-aware: a pathology no ensemble member was trained on comes back NaN rather
than as a fabricated zero, and downstream code keeps it out of the metrics.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Literal

import numpy as np

_EPS = 1e-12


@contextmanager
def _silence_allnan():
    """Suppress numpy's 'Mean of empty slice' / all-NaN RuntimeWarnings.

    A pathology covered by no ensemble member yields an all-NaN column; nanmean
    /nanstd correctly return NaN there (the wrapper maps it to a benign 0). The
    warning is expected, not a problem.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield

EnsembleMeasure = Literal[
    "mutual_information", "predictive_entropy", "mean_entropy", "std"
]


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Per-element Shannon entropy (nats) of independent Bernoulli(p) labels."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.asarray(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def _as_members(member_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(member_probs, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"member_probs must be (n_members, n_labels), got shape {arr.shape}")
    return arr


class EntropyUQ:
    """Single-pass uncertainty: binary entropy of the (mean) probability."""

    name = "entropy"

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        members = _as_members(member_probs)
        with _silence_allnan():
            probs = np.nanmean(members, axis=0)
        unc = binary_entropy(probs)
        return probs, unc, {}


class EnsembleUQ:
    """Deep Ensemble uncertainty with the full entropy decomposition.

    measure selects which per-label vector becomes the scalar-able uncertainty:
      - ``predictive_entropy`` = H(mean_p)               (total)
      - ``mean_entropy``       = mean_m H(p_m)           (aleatoric)
      - ``mutual_information`` = predictive_entropy - mean_entropy  (epistemic)
      - ``std``                = std_m p_m               (raw disagreement)
    """

    def __init__(self, measure: EnsembleMeasure = "mutual_information") -> None:
        self.measure: EnsembleMeasure = measure
        self.name = f"ensemble:{measure}"

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        members = _as_members(member_probs)
        with _silence_allnan():
            mean_p = np.nanmean(members, axis=0)
            mean_entropy = np.nanmean(binary_entropy(members), axis=0)
            std = np.nanstd(members, axis=0)
        predictive_entropy = binary_entropy(mean_p)
        mutual_information = np.clip(predictive_entropy - mean_entropy, 0.0, None)

        measures: dict[str, np.ndarray] = {
            "predictive_entropy": predictive_entropy,
            "mean_entropy": mean_entropy,
            "mutual_information": mutual_information,
            "std": std,
        }
        coverage = np.sum(~np.isnan(members), axis=0)
        unc = measures[self.measure]
        extra: dict[str, object] = {
            "n_members": int(members.shape[0]),
            "coverage": tuple(int(c) for c in coverage),
            **{k: tuple(float(x) for x in v) for k, v in measures.items()},
        }
        return mean_p, unc, extra
