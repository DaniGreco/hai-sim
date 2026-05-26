from __future__ import annotations

import pytest

from hai_sim.core.types import Diagnosis, LabelSchema, ModelOutput


def test_diagnosis_length_mismatch_raises():
    with pytest.raises(ValueError):
        Diagnosis(labels=(1, 0), confidence=(0.9,))


def test_model_output_length_mismatch_raises():
    diag = Diagnosis(labels=(1, 0), confidence=(0.9, 0.1))
    with pytest.raises(ValueError):
        ModelOutput(diagnosis=diag, uncertainty=(0.1,))


def test_label_schemas():
    assert LabelSchema.default_v0().n == 5
    assert LabelSchema.chexpert().n == 14
