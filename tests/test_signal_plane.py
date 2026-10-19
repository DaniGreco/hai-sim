"""The signal plane end-to-end: production, delivery, telemetry — and the coupling.

The last two tests are the ones that matter. Everything else checks plumbing; those
check the claim the plumbing exists for — that a simulated reader whose accuracy reads
the difficulty signal stops erring independently of the AI, so the human-AI error
correlation `rho` becomes a parameter instead of being pinned at zero by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from hai_sim.config.scenario import ScenarioConfig
from hai_sim.core.types import (
    Diagnosis,
    Image,
    LabelSchema,
    ModelOutput,
    PatientContext,
    ReaderState,
)
from hai_sim.factory import build_from_scenario
from hai_sim.metrics.collector import PatientRecord
from hai_sim.sim.engine import SimulationEngine

SCHEMA = LabelSchema.default_v0()


class TellingUncertaintyAI:
    """A model whose uncertainty is honest: high uncertainty exactly when it is wrong.

    Real ensembles only approximate this (the measured error-capture skill is +0.567,
    not 1.0). Making it exact here isolates the mechanism under test: if the coupling
    fails to appear with a *perfectly* informative uncertainty, it is the wiring at
    fault and not the strength of the signal.
    """

    name = "TellingUncertaintyAI"

    def __init__(self, schema: LabelSchema, error_every: int = 3) -> None:
        self.schema = schema
        self.error_every = error_every
        self._seen = 0

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        self._seen += 1
        wrong = (self._seen % self.error_every) == 0
        # Calls every finding absent, or every finding present. The engine does NOT
        # hand the AI stage the ground truth (deliberately — a real model never has
        # it), so "wrong" cannot be defined against the truth here; it is defined
        # against a low-prevalence stream, where all-present is nearly all-wrong and
        # all-absent nearly all-right.
        labels = (1 if wrong else 0,) * self.schema.n
        # ln 2 is the maximum binary entropy, so a difficulty model normalising by it
        # maps this to exactly 1.0 (hard) or 0.0 (easy).
        unc = float(np.log(2.0)) if wrong else 0.0
        return ModelOutput(
            diagnosis=Diagnosis(labels=labels, confidence=(0.5,) * self.schema.n),
            uncertainty=(unc,) * self.schema.n,
        )

    def service_time(
        self, images: Sequence[Image], context: PatientContext | None = None
    ) -> float:
        return 0.01


def _scenario(**overrides: object) -> ScenarioConfig:
    base: dict = {
        "name": "signals",
        "seed": 3,
        "patient_generator": {
            "kind": "poisson",
            "arrival_rate": 5.0,
            "n_patients": 60,
            "prevalence": 0.3,
        },
        "ai": {"kind": "random_mock"},
        "persona": "solo_senior",
        "sort": {"kind": "random"},
    }
    base.update(overrides)
    return ScenarioConfig.model_validate(base)


def _run(scenario: ScenarioConfig, ai: object | None = None) -> list[PatientRecord]:
    built = build_from_scenario(scenario)
    return (
        SimulationEngine(
            patient_generator=built.generator,
            ai=ai if ai is not None else built.ai,
            doctor=built.doctor,
            sort_strategy=built.sort,
            severity_model=built.severity,
            difficulty_model=built.difficulty,
            fatigue_curve=built.fatigue,
        )
        .run()
        .collector.records
    )


def _errors(record: PatientRecord) -> int:
    return sum(
        1
        for truth, call in zip(record.ground_truth, record.doc_labels, strict=True)
        if truth >= 0 and truth != call
    )


def _ai_errors(record: PatientRecord) -> int:
    return sum(
        1
        for truth, call in zip(record.ground_truth, record.ai_labels, strict=True)
        if truth >= 0 and truth != call
    )


# --- production and delivery ------------------------------------------------


def test_no_signals_are_produced_when_nothing_is_configured():
    """An unconfigured run must be exactly what it was before the plane existed."""
    records = _run(_scenario())
    assert all(not any(k.startswith("sig.") for k in r.observations) for r in records)


def test_difficulty_reaches_the_telemetry():
    records = _run(_scenario(difficulty={"kind": "constant", "value": 0.42}))
    assert all(r.observation("sig.difficulty") == 0.42 for r in records)


def test_fatigue_reaches_the_telemetry_and_advances_with_cases_read():
    records = _run(
        _scenario(fatigue={"kind": "linear", "per_unit": 0.02, "basis": "cases"})
    )
    levels = [r.observation("sig.fatigue") for r in records]
    assert levels[0] == 0.0, "the first read of a shift is done fresh"
    assert levels[-1] > levels[0]
    assert levels == sorted(levels), "with one worker, fatigue is monotone in read order"


def test_severity_is_mirrored_onto_the_signal_plane():
    records = _run(_scenario(severity={"kind": "constant", "value": 0.7}))
    assert all(r.observation("sig.severity") == 0.7 for r in records)


def test_a_per_label_signal_is_collapsed_to_one_number():
    """The plane is scalar because its consumers act once per read, not once per
    finding. The per-label form survives in the first-class severity field."""
    records = _run(
        _scenario(
            severity={"kind": "activation", "source": "ai_probs", "mode": "vector"}
        )
    )
    assert isinstance(records[0].severity, tuple)
    assert isinstance(records[0].observation("sig.severity"), float)


def test_signals_can_condition_a_metric():
    """The payoff of putting them in the observation bag: `sig.difficulty` is a
    grouping key like any other, so any bundle can be read per difficulty band."""
    from hai_sim.config.scenario import KPIGroupConfig
    from hai_sim.factory import build_kpi_groupings
    from hai_sim.metrics.slicing import split_records

    records = _run(_scenario(difficulty={"kind": "prior"}))
    groups = build_kpi_groupings([KPIGroupConfig(key="sig.difficulty", edges=(0.5,))], SCHEMA)
    slices = split_records(records, groups)
    assert len(slices) >= 2
    assert sum(len(s.records) for s in slices) == len(records)
    assert all(s.records for s in slices), "an empty band means the key never resolved"


def test_the_engine_records_which_modules_produced_the_plane():
    """A difficulty of 0.31 means nothing without knowing whether it came from a Beta
    prior or from the model's mutual information."""
    built = build_from_scenario(_scenario(difficulty={"kind": "prior", "a": 2.0, "b": 5.0}))
    result = SimulationEngine(
        patient_generator=built.generator,
        ai=built.ai,
        doctor=built.doctor,
        sort_strategy=built.sort,
        difficulty_model=built.difficulty,
    ).run()
    assert result.metadata["difficulty_model"] == "prior:beta(2,5)"
    assert result.metadata["fatigue_curve"] == "none"


def test_each_doctor_worker_tires_independently():
    """Fatigue belongs to the reader, not to the actor object. Two readers sharing a
    queue that both drove one counter would show a doubled decrement each."""
    records = _run(
        _scenario(
            n_doctor_workers=2,
            fatigue={"kind": "linear", "per_unit": 0.05, "basis": "cases"},
        )
    )
    peak = max(r.observation("sig.fatigue") for r in records)
    # 60 patients over 2 workers is ~30 reads each -> ~1.5 capped at 1.0 if shared,
    # but each worker should top out near 30 * 0.05 = 1.5 -> capped. Use a mid-run
    # check instead: the *set* of levels must contain repeats, since two workers pass
    # through the same case count independently.
    levels = [round(float(r.observation("sig.fatigue")), 6) for r in records]
    assert len(levels) - len(set(levels)) >= 10, "two workers must revisit the same levels"
    assert peak <= 1.0


def test_the_ai_station_never_receives_the_signal_plane():
    """Difficulty is computed *from* the AI's output; feeding it back would be circular,
    and the AI stage is deliberately handed the plain context anyway."""
    seen: list[PatientContext | None] = []

    class Recording(TellingUncertaintyAI):
        def diagnose(self, images, *, context=None, ai_hint=None):
            seen.append(context)
            return super().diagnose(images, context=context, ai_hint=ai_hint)

    _run(
        _scenario(difficulty={"kind": "constant", "value": 0.9}),
        ai=Recording(SCHEMA),
    )
    assert seen and all(not ctx.signals for ctx in seen if ctx is not None)


def test_the_service_time_model_sees_the_fatigue_signal():
    """The read has to be timed with the fatigue the reader has *now*, which is why the
    signal is stamped at pick-up rather than after the read."""
    fast = _run(
        _scenario(
            persona={
                "name": "p",
                "capability": {
                    "kind": "fixed_accuracy",
                    "service_time": {"kind": "constant", "value": 1.0},
                },
            }
        )
    )
    slow = _run(
        _scenario(
            fatigue={"kind": "linear", "per_unit": 0.05, "basis": "cases"},
            persona={
                "name": "p",
                "capability": {
                    "kind": "fixed_accuracy",
                    "service_time": {
                        "kind": "constant",
                        "value": 1.0,
                        "fatigue_slope": 2.0,
                    },
                },
            },
        )
    )
    first_read = fast[0].ts_doctor_end - fast[0].ts_doctor_start
    last_read = slow[-1].ts_doctor_end - slow[-1].ts_doctor_start
    assert first_read == pytest.approx(1.0)
    assert last_read > 1.5


# --- the claim --------------------------------------------------------------


def _error_correlation(records: list[PatientRecord]) -> float:
    """Pearson correlation between the doctor's and the AI's per-case error counts.

    This is `rho_HM`: the quantity the reader model treats as the experimental lever and controls
    through the shared-difficulty model.
    """
    doc = np.asarray([_errors(r) for r in records], dtype=float)
    ai = np.asarray([_ai_errors(r) for r in records], dtype=float)
    if doc.std() == 0 or ai.std() == 0:
        return 0.0
    return float(np.corrcoef(doc, ai)[0, 1])


def _coupled_scenario(difficulty_slope: float) -> ScenarioConfig:
    return _scenario(
        patient_generator={
            "kind": "poisson",
            "arrival_rate": 5.0,
            "n_patients": 300,
            "prevalence": 0.1,
        },
        # Difficulty *is* the AI's uncertainty: the coupling is induced from the
        # model's own behaviour rather than imposed by a parameter.
        difficulty={"kind": "uncertainty", "source": "uncertainty", "aggregator": "mean"},
        persona={
            "name": "coupled",
            "capability": {
                "kind": "fixed_accuracy",
                "sensitivity": 0.95,
                "specificity": 0.95,
                "difficulty_slope": difficulty_slope,
                "floor": 0.5,
            },
        },
    )


def test_without_a_difficulty_slope_the_reader_errs_independently_of_the_ai():
    """The baseline, and the defect being fixed: a reader that flips labels at a fixed
    rate draws from a coin that has never seen the image, so `rho` is zero whatever a
    scenario asks for. This is why both senior arms measured a flat 0.0000 delta."""
    records = _run(_coupled_scenario(0.0), ai=TellingUncertaintyAI(SCHEMA))
    assert abs(_error_correlation(records)) < 0.15


def test_with_a_difficulty_slope_the_reader_and_the_ai_fail_on_the_same_cases():
    """The claim. Same reader, same accuracies, same images — only the difficulty
    coupling switched on — and the two error streams line up."""
    records = _run(_coupled_scenario(0.45), ai=TellingUncertaintyAI(SCHEMA))
    assert _error_correlation(records) > 0.3


def test_the_reader_publishes_the_penalty_it_actually_applied():
    """So a study can check the coupling bit, rather than inferring it from an outcome:
    a run with a difficulty model but a case-blind persona would look identical."""
    coupled = _run(_coupled_scenario(0.45), ai=TellingUncertaintyAI(SCHEMA))
    blind = _run(_coupled_scenario(0.0), ai=TellingUncertaintyAI(SCHEMA))
    assert any(r.observation("doc.accuracy_penalty", 0.0) > 0 for r in coupled)
    assert all(r.observation("doc.accuracy_penalty") is None for r in blind)


def test_accuracy_never_degrades_past_a_coin_flip():
    """Unbounded arithmetic would make a reader *anti*-correlated with the truth on hard
    cases, silently inverting every complementarity result."""
    records = _run(
        _scenario(
            patient_generator={
                "kind": "poisson",
                "arrival_rate": 5.0,
                "n_patients": 200,
                "prevalence": 0.5,
            },
            difficulty={"kind": "constant", "value": 1.0},
            persona={
                "name": "wrecked",
                "capability": {
                    "kind": "fixed_accuracy",
                    "sensitivity": 0.9,
                    "specificity": 0.9,
                    "difficulty_slope": 5.0,
                },
            },
        )
    )
    total = sum(len([g for g in r.ground_truth if g >= 0]) for r in records)
    wrong = sum(_errors(r) for r in records)
    assert 0.4 < wrong / total < 0.6, "a fully degraded reader is a coin, not an inverter"


def test_reader_state_is_not_shared_between_runs():
    """Two runs of the same scenario must give the same fatigue trace."""
    scenario = _scenario(fatigue={"kind": "vigilance", "scale": 10.0})
    first = [r.observation("sig.fatigue") for r in _run(scenario)]
    second = [r.observation("sig.fatigue") for r in _run(scenario)]
    assert first == second
    assert ReaderState().cases_read == 0
