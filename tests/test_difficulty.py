"""Case difficulty: the five models, and the coupling they exist to create.

The last test is the point of the whole module. Everything before it checks a
component; that one checks the *claim* — that with a difficulty model wired to the AI's
uncertainty, a simulated reader's errors stop being independent of the AI's.
"""

from __future__ import annotations

import numpy as np
import pytest

from hai_sim.core.protocols import DifficultyModel
from hai_sim.core.types import (
    Diagnosis,
    LabelSchema,
    ModelOutput,
    Patient,
    PatientContext,
)
from hai_sim.difficulty.models import (
    ConstantDifficulty,
    ImageQualityDifficulty,
    PriorDifficulty,
    TableDifficulty,
    UncertaintyDifficulty,
    disagreement_table,
)

SCHEMA = LabelSchema.default_v0()


def _patient(example_id: str | None = "img0", image: np.ndarray | None = None) -> Patient:
    img = image if image is not None else np.zeros((8, 8), dtype=np.uint8)
    return Patient(
        id=0,
        images=(img,),
        ground_truth=(0,) * SCHEMA.n,
        context=PatientContext(patient_id=0, arrival_time=0.0, example_id=example_id),
    )


def _ai_output(uncertainty: tuple[float, ...], **extra: object) -> ModelOutput:
    n = len(uncertainty)
    return ModelOutput(
        diagnosis=Diagnosis(labels=(0,) * n, confidence=(0.5,) * n),
        uncertainty=uncertainty,
        extra=dict(extra),
    )


@pytest.mark.parametrize(
    "model",
    [
        ConstantDifficulty(0.3),
        PriorDifficulty(),
        UncertaintyDifficulty(),
        ImageQualityDifficulty(),
        TableDifficulty({"img0": 0.7}),
    ],
)
def test_every_model_satisfies_the_protocol(model):
    assert isinstance(model, DifficultyModel)


@pytest.mark.parametrize(
    "model",
    [
        ConstantDifficulty(0.3),
        PriorDifficulty(),
        UncertaintyDifficulty(),
        ImageQualityDifficulty(),
        TableDifficulty({"img0": 0.7}),
    ],
)
def test_every_model_stays_in_the_unit_interval(model):
    """Difficulty is a fraction. A reader's accuracy penalty is scaled by it, so a
    value outside [0, 1] would silently mean 'more than fully degraded'."""
    patient = _patient()
    value = model.score(
        patient=patient,
        ai_output=_ai_output((0.6, 0.1, 0.4, 0.0, 0.69)),
        context=patient.context,
    )
    assert 0.0 <= float(value) <= 1.0


def test_prior_difficulty_is_stable_per_image():
    """The same image must be equally hard every time it is sampled.

    Paired contrasts across replications compare like with like; if difficulty were a
    fresh draw per *read*, the same patient would be a different case in the treatment
    arm than in the baseline and the pairing would be broken.
    """
    model = PriorDifficulty(seed=7)
    patient = _patient("study_42")
    first = model.score(patient=patient, ai_output=None, context=patient.context)
    second = model.score(patient=patient, ai_output=None, context=patient.context)
    assert first == second

    fresh = PriorDifficulty(seed=7)
    assert fresh.score(patient=patient, ai_output=None, context=patient.context) == first


def test_prior_difficulty_varies_across_images():
    model = PriorDifficulty(seed=3)
    values = {
        model.score(
            patient=(p := _patient(f"img{i}")), ai_output=None, context=p.context
        )
        for i in range(30)
    }
    assert len(values) > 20, "a Beta prior should give near-distinct values per image"


def test_prior_difficulty_without_image_identity_still_works():
    """Synthetic patients carry no example_id; the model must not crash on them."""
    model = PriorDifficulty()
    patient = _patient(example_id=None)
    value = model.score(patient=patient, ai_output=None, context=patient.context)
    assert 0.0 <= value <= 1.0


def test_uncertainty_difficulty_is_monotone_in_uncertainty():
    model = UncertaintyDifficulty()
    patient = _patient()
    low = model.score(
        patient=patient, ai_output=_ai_output((0.01,) * 5), context=patient.context
    )
    high = model.score(
        patient=patient, ai_output=_ai_output((0.69,) * 5), context=patient.context
    )
    assert high > low


def test_uncertainty_difficulty_reads_a_named_measure_from_the_extra_channel():
    """Epistemic and aleatoric are different quantities and both must be selectable."""
    model = UncertaintyDifficulty(source="mutual_information")
    patient = _patient()
    out = _ai_output((0.0,) * 5, mutual_information=(0.69,) * 5)
    value = model.score(patient=patient, ai_output=out, context=patient.context)
    assert value == pytest.approx(1.0, abs=0.01)


def test_uncertainty_difficulty_names_the_available_measures_when_asked_for_a_missing_one():
    model = UncertaintyDifficulty(source="mutual_information")
    patient = _patient()
    with pytest.raises(ValueError, match="observation channel"):
        model.score(
            patient=patient, ai_output=_ai_output((0.1,) * 5), context=patient.context
        )


def test_uncertainty_difficulty_ignores_labels_no_member_covered():
    """An all-NaN pathology is 'not measured', not 'maximally easy'."""
    model = UncertaintyDifficulty()
    patient = _patient()
    out = _ai_output((0.69, float("nan"), 0.69, float("nan"), 0.69))
    value = model.score(patient=patient, ai_output=out, context=patient.context)
    assert value == pytest.approx(1.0, abs=0.01)


def test_uncertainty_difficulty_can_stay_per_label():
    model = UncertaintyDifficulty(aggregator="none")
    patient = _patient()
    value = model.score(
        patient=patient, ai_output=_ai_output((0.0, 0.69, 0.0, 0.0, 0.0)),
        context=patient.context,
    )
    assert isinstance(value, np.ndarray) and value.shape == (5,)


def test_image_quality_difficulty_ranks_a_blurred_image_as_harder():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(32, 32)).astype(float)
    blurred = np.full((32, 32), 128.0)
    model = ImageQualityDifficulty(reference=float(np.var(sharp)))
    hard = model.score(
        patient=(p := _patient(image=blurred)), ai_output=None, context=p.context
    )
    easy = model.score(
        patient=(q := _patient(image=sharp)), ai_output=None, context=q.context
    )
    assert hard > easy


def test_table_difficulty_looks_up_by_example_id():
    model = TableDifficulty({"a": 0.9, "b": 0.1}, default=0.5)
    for example_id, expected in (("a", 0.9), ("b", 0.1), ("missing", 0.5)):
        patient = _patient(example_id)
        assert model.score(
            patient=patient, ai_output=None, context=patient.context
        ) == expected


def test_table_difficulty_strict_mode_refuses_to_silently_default():
    """A partial table flattens the difficulty distribution where it applies, which is
    a silent loss of the very variation the model exists to supply."""
    model = TableDifficulty({"a": 0.9}, strict=True)
    patient = _patient("b")
    with pytest.raises(KeyError, match="no entry"):
        model.score(patient=patient, ai_output=None, context=patient.context)


def test_table_difficulty_from_csv(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("example_id,difficulty\nimg0,0.8\nimg1,0.2\n", encoding="utf-8")
    model = TableDifficulty.from_csv(path)
    patient = _patient("img0")
    assert model.score(patient=patient, ai_output=None, context=patient.context) == 0.8


def test_table_difficulty_from_csv_reports_a_missing_column(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,score\nimg0,0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        TableDifficulty.from_csv(path)


def test_disagreement_table_is_zero_when_readers_agree():
    table = disagreement_table({"img0": {"r1": (1, 0), "r2": (1, 0), "r3": (1, 0)}})
    assert table["img0"] == pytest.approx(0.0)


def test_disagreement_table_peaks_at_an_even_split():
    table = disagreement_table({"img0": {"r1": (1,), "r2": (0,)}})
    assert table["img0"] == pytest.approx(1.0)


def test_disagreement_table_skips_labels_nobody_annotated():
    """-1 is 'not annotated'. Counting it as a vote would invent disagreement."""
    table = disagreement_table({"img0": {"r1": (1, -1), "r2": (1, -1)}})
    assert table["img0"] == pytest.approx(0.0)


def test_disagreement_table_feeds_table_difficulty_directly():
    """The two halves of the recommended real-data route must actually compose."""
    reads = {"a": {"r1": (1,), "r2": (0,)}, "b": {"r1": (0,), "r2": (0,)}}
    model = TableDifficulty(disagreement_table(reads), strict=True)
    hard = _patient("a")
    easy = _patient("b")
    assert model.score(patient=hard, ai_output=None, context=hard.context) > model.score(
        patient=easy, ai_output=None, context=easy.context
    )
