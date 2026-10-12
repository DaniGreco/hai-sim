from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.types import Diagnosis, ModelOutput, Patient, PatientContext
from hai_sim.sort.uncertainty import UncertaintyDescSort


def _output(uncertainty: tuple[float, ...]) -> ModelOutput:
    n = len(uncertainty)
    return ModelOutput(
        diagnosis=Diagnosis(labels=tuple([0] * n), confidence=tuple([0.0] * n)),
        uncertainty=uncertainty,
    )


def _patient() -> Patient:
    return Patient(
        id=0,
        images=(np.zeros((1, 1), dtype=np.uint8),),
        ground_truth=(0, 0, 0, 0, 0),
        context=PatientContext(patient_id=0, arrival_time=0.0),
    )


def test_uncertainty_desc_higher_uncertainty_gets_lower_priority_value():
    sort = UncertaintyDescSort(aggregator="mean")
    low = sort.priority(_output((0.1, 0.1, 0.1, 0.1, 0.1)), _patient())
    high = sort.priority(_output((0.9, 0.9, 0.9, 0.9, 0.9)), _patient())
    # PriorityStore: lower key → earlier, so high-uncertainty must have lower key
    assert high < low


def test_uncertainty_max_aggregator():
    sort = UncertaintyDescSort(aggregator="max")
    out = _output((0.1, 0.9, 0.1, 0.1, 0.1))
    assert sort.priority(out, _patient()) == pytest.approx(-0.9)


def test_uncertainty_weighted_requires_weights():
    sort = UncertaintyDescSort(aggregator="weighted")
    with pytest.raises(ValueError):
        sort.priority(_output((0.5,) * 5), _patient())


def test_probability_desc_serves_the_most_suspicious_patient_first():
    from hai_sim.sort.uncertainty import ProbabilityDescSort

    def out(conf: tuple[float, ...]) -> ModelOutput:
        n = len(conf)
        return ModelOutput(
            diagnosis=Diagnosis(labels=tuple([0] * n), confidence=conf),
            uncertainty=tuple([0.0] * n),
        )

    sort = ProbabilityDescSort(aggregator="max")
    calm = sort.priority(out((0.01, 0.02, 0.01, 0.01, 0.01)), _patient())
    alarming = sort.priority(out((0.01, 0.40, 0.01, 0.01, 0.01)), _patient())
    assert alarming < calm, "the queue serves the lowest priority value first"
