"""Concrete `UQMethod` implementations.

Two strategies ship here, both operating on a ``(n_members, n_labels)`` stack of
per-member positive-class probabilities:

- ``EntropyUQ``: single forward pass (n_members may be 1). Uncertainty = the
  binary predictive entropy of each label's probability. This is a cheap
  *aleatoric-flavoured* proxy derived from the model's own confidence.

- ``EnsembleUQ``: a Deep Ensemble over several model members (e.g. the different
  TorchXRayVision weight sets). It decomposes total uncertainty into the three
  classic measures — predictive entropy (total), mean entropy (aleatoric) and
  their difference, the mutual information (epistemic disagreement) — matching
  the ENTROPY / MEAN_ENTROPY / MUTUAL_INFORMATION trio used in the prior "Roby"
  framework. Any of them (or the member std) can be selected as the scalar
  uncertainty; the rest are returned in ``extra``.
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
