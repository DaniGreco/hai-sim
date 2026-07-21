"""Example KPI bundles that live OUTSIDE the library.

Reference either of these from a scenario or study YAML by file path and attribute::

    kpis:
      - system
      - actors
      - configs/examples/custom_metrics.py:CostBundle
      - configs/examples/custom_metrics.py:UncertaintySplitBundle

Nothing here is imported by `hai_sim`; the registry loads the file on demand. Two rules
are all a bundle must respect: expose a ``name`` and a ``compute(records)``, and be
constructible with no arguments (so parameters need defaults — set them by subclassing
if you want a second variant under a different name).

The report is keyed by the bundle's own ``name``, so the study layer reaches these with
the same dotted paths as the built-ins: ``cost.hybrid_total``, ``uncertainty_split.n``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hai_sim.metrics.collector import PatientRecord

_ACTORS = ("ai", "doctor", "hybrid")


def _labels(record: PatientRecord, actor: str) -> tuple[int, ...]:
    """The label vector of one actor. ``doctor`` = unaided, ``hybrid`` = final call."""
    if actor == "ai":
        return record.ai_labels
    if actor == "hybrid":
        return record.doc_labels
    return record.doctor_solo_labels or record.doc_labels


class CostBundle:
    """Clinical cost with false negatives weighted far above false positives.

    A missed pathology and a needless alarm are not the same mistake, so macro-F1 —
    which treats them alike — is the wrong lens for the decision this simulator exists
    to inform. This bundle collapses the confusion counts into one currency and reports
    it per actor, plus the per-patient average that makes runs of different size
    comparable. The default 10:1 is a placeholder: the ratio is exactly the modelling
    choice a sensitivity analysis should sweep, not a constant to be believed.
    """

    name = "cost"

    def __init__(self, fn_cost: float = 10.0, fp_cost: float = 1.0) -> None:
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def compute(self, records: Sequence[PatientRecord]) -> dict[str, float]:
        out: dict[str, float] = {"fn_cost": self.fn_cost, "fp_cost": self.fp_cost}
        n_patients = max(len(records), 1)
        for actor in _ACTORS:
            n_fn = n_fp = 0
            for record in records:
                for pred, truth in zip(_labels(record, actor), record.ground_truth, strict=True):
                    if truth not in (0, 1):
                        continue  # not annotated for this label — excluded everywhere
                    if pred == 0 and truth == 1:
                        n_fn += 1
                    elif pred == 1 and truth == 0:
                        n_fp += 1
            total = self.fn_cost * n_fn + self.fp_cost * n_fp
            out[f"{actor}_total"] = total
            out[f"{actor}_per_patient"] = total / n_patients
        return out


@dataclass(frozen=True)
class UncertaintySplit:
    """Mean epistemic / aleatoric uncertainty, split by whether the AI erred."""

    n: int
    n_with_error: int
    epistemic_on_error: float | None
    epistemic_on_correct: float | None
    aleatoric_on_error: float | None
    aleatoric_on_correct: float | None

    @property
    def epistemic_separation(self) -> float | None:
        """How much higher epistemic uncertainty runs on the AI's mistakes.

        Positive ⇒ the epistemic signal points at the errors, which is the premise
        the whole uncertainty-first triage rests on.
        """
        if self.epistemic_on_error is None or self.epistemic_on_correct is None:
            return None
        return self.epistemic_on_error - self.epistemic_on_correct


class UncertaintySplitBundle:
    """Does epistemic uncertainty single out the AI's errors better than aleatoric?

    This metric was **not expressible before the telemetry carried observations**: the
    ensemble computes the full decomposition for every patient, but the collector used
    to drop it, so only the single aggregated uncertainty survived. It reads
    ``ai.mutual_information`` (epistemic) and ``ai.mean_entropy`` (aleatoric) straight
    out of the bag — no change to `hai_sim` needed to ask a new question of it.

    Returns ``None`` fields when the run's AI does not publish a decomposition (a mock,
    or a single model rather than an ensemble), rather than pretending to a number.
    """

    name = "uncertainty_split"

    def compute(self, records: Sequence[PatientRecord]) -> UncertaintySplit:
        epi_err: list[float] = []
        epi_ok: list[float] = []
        ale_err: list[float] = []
        ale_ok: list[float] = []
        n_with_error = 0

        for record in records:
            epistemic = record.observation("ai.mutual_information")
            aleatoric = record.observation("ai.mean_entropy")
            errs = any(
                pred != truth
                for pred, truth in zip(record.ai_labels, record.ground_truth, strict=True)
                if truth in (0, 1)
            )
            n_with_error += int(errs)
            if epistemic is not None:
                (epi_err if errs else epi_ok).append(_mean(epistemic))
            if aleatoric is not None:
                (ale_err if errs else ale_ok).append(_mean(aleatoric))

        return UncertaintySplit(
            n=len(records),
            n_with_error=n_with_error,
            epistemic_on_error=_mean_or_none(epi_err),
            epistemic_on_correct=_mean_or_none(epi_ok),
            aleatoric_on_error=_mean_or_none(ale_err),
            aleatoric_on_correct=_mean_or_none(ale_ok),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values) if len(values) else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
