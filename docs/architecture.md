# Architecture

How the code is organised, where each part of the workflow can be replaced, and how to add
your own.

- [The contract](#the-contract) · [Module map](#module-map) ·
  [Uncertainty measures](#uncertainty-measures) · [Scripts](#scripts)
- [Extending HaiSIM](#extending-haisim): [a new model](#a-new-ai-model) ·
  [a new worklist order](#a-new-worklist-order) ·
  [a new integration policy](#a-new-integration-policy) · [a new metric](#a-new-metric) ·
  [a new dataset](#a-new-dataset)
- [Limitations](#limitations)

## The contract

Every part of the workflow a study may want to vary is a `@runtime_checkable` `Protocol` in
[`src/hai_sim/core/protocols.py`](../src/hai_sim/core/protocols.py). The protocols are
structural: a class satisfies one by having the right attributes and methods, with no
inheritance and no registration. `BaseDiagnosticActor` (`core/base.py`) is the only abstract
base class, and inheriting it is optional; it supplies a default `name` and a constant
`service_time`.

Where each protocol sits in a run:

```
ArrivalProcess ─→ Dataset ─→ [ DiagnosticActor: the AI ] ─→ SortStrategy ─→
   (when)         (who)          │  UQMethod + Calibrator     (triage)
                                 │  SeverityModel                    │
                                 │  DifficultyModel  ─── signals ────┤
                                 │  FatigueCurve     ─── signals ────┤
                                 └──────────────────────────→ [ DiagnosticActor:
                                                ServiceTimeModel   the reader ]
                                                IntegrationPolicy      │
                                                        KPIBundle · Grouping ──→
```

| protocol | method | implementations |
|---|---|---|
| `DiagnosticActor` | `diagnose(images, *, context, ai_hint) -> ModelOutput`, `service_time(images, context) -> float` | `RandomMockAI`, `FixedAccuracyAI` (synthetic), `TorchXRayVisionAI`, `MedGemmaAI`, `VLLMMedGemmaAI`, `CachedEnsembleAI`, `CompositeCachedAI`, `RandomMockDoctor`, `FixedAccuracyDoctor`, `SDTDoctor`, `AssistedDoctor`, `ProjectedActor` |
| `IntegrationPolicy` | `combine(doctor_out, ai_hint, context) -> ModelOutput` | `LinearPoolPolicy`, `ConfidenceGatedPoolPolicy`, `LogOddsFusionPolicy` |
| `SortStrategy` | `priority(output, patient) -> float` (lower is served first) | `RandomSort`, `UncertaintyDescSort`, `ProbabilityDescSort` |
| `UQMethod` | `estimate(member_probs) -> (probs, uncertainty, extra)` | `EntropyUQ`, `EnsembleUQ`, `CalibratedUQ` |
| `Calibrator` | `calibrate(probs) -> probs` | `IdentityCalibrator`, `TemperatureScaling`, `PlattCalibrator`, `IsotonicCalibrator` |
| `ArrivalProcess` | `next_delay(now) -> float` | `PoissonArrival`, `DeterministicArrival`, `NHPPArrival` (with `PiecewiseConstantRate`) |
| `SeverityModel` | `score(*, patient, ai_output, context)` | `ConstantSeverity`, `ActivationSeverity`, `ImageQualitySeverity`, `EmbeddingSeverity` |
| `DifficultyModel` | `score(*, patient, ai_output, context)` | `ConstantDifficulty`, `PriorDifficulty`, `UncertaintyDifficulty`, `ImageQualityDifficulty`, `TableDifficulty` |
| `FatigueCurve` | `level(state) -> float` | `NoFatigue`, `LinearFatigue`, `VigilanceFatigue`, `CircadianFatigue`, `CompositeFatigue` |
| `ServiceTimeModel` | `sample(images, context) -> float` | `ConstantServiceTime`, `LognormalServiceTime`, `ScaledServiceTime`, `FatigueModulatedServiceTime` |
| `Dataset` | `__len__`, `__getitem__ -> DatasetExample`, plus `name`, `schema`, `capabilities` | `MockDataset`, `XRVDataset`, `ImageFolderDataset` |

Two more live next to what they serve, in `metrics/`:

| protocol | method | implementations |
|---|---|---|
| `KPIBundle` (`metrics/registry.py`) | `compute(records) -> Any` | `SystemBundle`, `ActorsBundle`, `TeamBundle`, `JointTableBundle`, plus any bundle referenced from a scenario as `file.py:Attribute` |
| `Grouping` (`metrics/slicing.py`) | `split(records) -> [RecordSlice]` | `ValueGrouping`, `LabelGrouping`, `TruthGrouping` |

Smaller protocols serve one subsystem: `LabelSource` (`data/sources.py`) for label files,
`LabelReadout` and `VisionLanguageBackend` (`models/readout.py`) for the MedGemma readouts,
and `CacheableModel` (`analysis/inference_cache.py`) for anything that can fill a cache cell.

`DiagnosticActor` is one contract for **both** readers. A model and a reader differ in what
they do, not in what they are asked for, so a study can put either at either station or
compare them on identical terms. Two rules apply to an AI model: it must accept `ai_hint`
and ignore it, and it must never read `context.ground_truth`, which the engine fills only
for the reader stage, as the device that makes a simulated reader err at a chosen rate.

A part of the workflow becomes a protocol when there are three implementations one would
actually run **and compare in the same study**. That is why there is no cost-model protocol
(a cost matrix is already a `KPIBundle`) and no protocol for the station layout (only one
arrangement is under test: the AI in series, before the reader).

## Module map

Everything under [`src/hai_sim/`](../src/hai_sim):

```
core/protocols.py            the protocols above: the public contract
core/types.py                frozen dataclasses: LabelSchema, Diagnosis, ModelOutput, Patient,
                             PatientContext, ReaderState, DatasetExample, DatasetCapabilities
core/base.py                 BaseDiagnosticActor, the optional base class
cli.py                       the `hai-sim` entry point: run | study | capture | plot
factory.py                   scenario config -> runtime objects; every `kind:` is resolved here
config/scenario.py           strict pydantic models of a scenario, and load_scenario
labels.py                    label projection (label_view) and label maps (vindr14->xrv)
personas.py                  PERSONA_LIBRARY: the senior/junior x solo/skeptical/trusting presets
sdt.py                       signal-detection core shared by the SDT reader and the synthetic AI
_numerics.py                 logit and sigmoid
models/                      AI backends: mock, synthetic (skill as a dial), torchxrv,
                             cached (one cell, or several composed), medgemma, readout,
                             vllm_backend; _finalize.py turns UQ output into a ModelOutput
doctors/                     readers: mock, simple (FixedAccuracyDoctor), sdt (SDTDoctor),
                             assisted (AssistedDoctor and the three integration policies)
sort/                        worklist orders: random, uncertainty (uncertainty and probability)
uq/methods.py                EntropyUQ, EnsembleUQ
uq/calibration.py            the four calibrators, CalibratedUQ, the fitters, the ECE
arrivals/processes.py        Poisson, deterministic, NHPP by thinning
severity/, difficulty/       per-patient signals computed between the two stations
fatigue/                     fatigue curves (curves.py) and reading-time models (service.py)
data/                        MockDataset, XRVDataset, ImageFolderDataset, CSV label sources
sim/engine.py                the SimPy engine
sim/patient_generator.py     synthetic patient stream (poisson)
sim/dataset_generator.py     patient stream from a Dataset
sim/cache_generator.py       patient stream from a cache cell (inference_cache)
metrics/collector.py         PatientRecord and MetricsCollector: fixed core plus the
                             namespaced observation bag (ai. doc. ctx. load. sig.)
metrics/kpi.py               the four built-in bundles, compute_report, compute_grouped_report
metrics/registry.py          bundle registry: name | module:Attribute | file.py:Attribute
metrics/slicing.py           conditioned metrics: split the records, rerun the bundle
metrics/report.py            flatten a report to dotted path -> scalar
study/spec.py                study config, factor grid -> runs
study/runner.py              run the cells (optionally in parallel), write the tables
study/stats.py               replication statistics and paired contrasts
analysis/inference_cache.py  the cache cell format (.npz + .json manifest), resumable build
analysis/cache_matrix.py     one cell per (model, dataset): naming, fingerprints, build plan
analysis/cache_build.py      fill the missing cells of a scenario
analysis/capture_study.py    the capture study (`hai-sim capture`); figures in capture_figures.py
analysis/error_curve.py      capture curves, their skill and AUROC
analysis/study_figures.py    the figures of the simulation studies (`hai-sim plot`)
```

To read the code in pipeline order: `sim/patient_generator.py` → `models/` → `sort/` →
`doctors/` → `metrics/`, with `sim/engine.py` as the loop that runs them and `factory.py`
as the bridge from YAML to objects.

## Uncertainty measures

In `uq/methods.py`. Every measure is per finding, in nats (natural-log binary entropy), and
NaN-aware: not every ensemble member covers every finding, so a finding no member covers
comes back NaN and the number of covering members is reported as `coverage`.

| measure | formula | reads as |
|---|---|---|
| `predictive_entropy` | H(mean_m p_m) | total uncertainty |
| `mean_entropy` | mean_m H(p_m) | aleatoric: ambiguity in the image itself |
| `mutual_information` | predictive entropy − mean entropy, clipped at 0 | epistemic: the members disagree (the default of `ensemble_measure`) |
| `std` | std_m p_m | raw spread of the members |

`EnsembleUQ` exposes one of the four as the uncertainty the worklist sorts on, and returns
all four in `extra`, so they all reach the telemetry and can be compared as triage keys
without rerunning inference. `EntropyUQ` is the entropy of a single probability; it is what
a one-member model such as MedGemma has. `CalibratedUQ` wraps either one with a
[calibrator](calibration-and-fusion.md).

## Scripts

In [`scripts/`](../scripts). Run them from the repository root with `uv run python`.

| script | purpose |
|---|---|
| `build_vindr_labels.py` | Download the VinDr-CXR competition `train.csv` from Kaggle and write `data/vindr/labels.csv` (a finding is present when at least `--k` readers, default 2 of 3, marked it), `difficulty.csv` (inter-reader disagreement per finding) and `readers.csv`. `--train-csv` uses a local copy. Needed once before any study on VinDr-CXR. |
| `fetch_nih_images.py` | Download NIH ChestX-ray14 from Kaggle (about 42 GB) and place the 30,805 images of the NIH cells in `data/nih_full/`, checked against the SHA-256 list in `data/provenance/`. Only needed to build new cells. |
| `materialize_cache.py` | Build (`--mode fill`) or check (`--mode report`, `require`) the cache cells a scenario declares ([usage](usage.md#inference-caches)). |
| `fit_calibration.py` | Fit a calibration on one or more cache cells. `--method platt_per_label --split even --out <file>` writes the per-finding Platt files of `data/cache/calibration/`; the other methods (`temperature`, `temperature_per_label`, `platt`, `isotonic`) print a YAML block. |
| `paper_figures.py` | Figures, table and numbers of the paper (RQ1 to RQ4), from the capture study and the RQ3 and RQ4 study tables ([usage](usage.md#reproduce-the-paper)). |

Examples of the calibration fits that produced the shipped files:

```bash
uv run python scripts/fit_calibration.py --method platt_per_label --split even \
    --cache data/cache/vindr__chex data/cache/vindr__mimic_ch \
            data/cache/vindr__pc data/cache/vindr__rsna \
    --out data/cache/calibration/vindr__ens4.json
uv run python scripts/fit_calibration.py --method platt_per_label --split even \
    --cache data/cache/vindr__medgemma-27b \
    --out data/cache/calibration/vindr__medgemma-27b.json
```

Several `--cache` stems are composed by image id into one ensemble, and the fit is on the
fused probability, the number the reader is shown.

## Extending HaiSIM

Adding an implementation of a protocol always follows the same steps (see also
[`CONTRIBUTING.md`](../CONTRIBUTING.md)):

1. write the class against the protocol, in the module of its kind;
2. add its `kind` value and its fields to
   [`config/scenario.py`](../src/hai_sim/config/scenario.py), with defaults that leave
   existing scenarios unchanged;
3. add its branch to the matching `build_*` function in
   [`factory.py`](../src/hai_sim/factory.py); import heavy dependencies inside the branch;
4. add tests in `tests/`, and document the new field in [scenarios](scenarios.md).

A new metric is the exception: it needs no change to the package.

### A new AI model

An AI model is a `DiagnosticActor`. Run the per-member probabilities through the
configured `UQMethod`, so that the uncertainty measure and the calibration of the scenario
apply to it like to the shipped models:

```python
import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import LabelSchema, ModelOutput


class MyModel(BaseDiagnosticActor):
    def __init__(self, schema: LabelSchema, uq, decision_threshold=0.5, service_time=0.05):
        super().__init__(service_time=service_time)
        self.schema, self.uq, self.decision_threshold = schema, uq, decision_threshold

    def member_probabilities(self, image) -> np.ndarray:
        """(n_members, n_labels): P(finding present) for each member."""
        ...

    def diagnose(self, images, *, context=None, ai_hint=None) -> ModelOutput:
        # ai_hint is ignored; context.ground_truth must never be read here.
        probs, uncertainty, extra = self.uq.estimate(self.member_probabilities(images[0]))
        return ModelOutput.from_arrays(
            probs >= self.decision_threshold, probs, uncertainty, extra=extra
        )
```

Then add `my_model` to `AIModelConfig.kind` and a branch to `build_ai`:

```python
if cfg.kind == "my_model":
    from my_package.models import MyModel

    return MyModel(
        schema=schema,
        uq=_build_uq(cfg, schema.labels),                         # UQ + calibration
        decision_threshold=_decision_threshold(cfg, schema.labels),
        service_time=cfg.service_time,
    )
```

Whatever the model puts in `extra` appears in the telemetry as `ai.<key>`. To make the
model cacheable through the cache matrix, give it `member_names`, `member_probabilities` and
`member_probabilities_batch` (the `CacheableModel` protocol) and add a branch to
`cells_for_ai` in `analysis/cache_matrix.py` that names its cell and fingerprint.

### A new worklist order

A `SortStrategy` has a `name` and `priority(output, patient) -> float`; the lowest value is
served first, so negate a "highest first" score. Add the `kind` to `SortConfig` and a branch
to `build_sort`, which also receives the worklist random stream for orders that need one.

### A new integration policy

An `IntegrationPolicy` has a `name` and `combine(doctor_out, ai_hint, context)`, and returns
the team's `ModelOutput`. It must keep the reader's unaided call in
`extra["doctor_solo_labels"]`, which is what the `actors.doctor` and `team` KPIs read; see
`_team_output` in `doctors/assisted.py`. Add the `kind` to `TrustConfig` and `FusionConfig`
and a branch to `build_trust_policy`.

### A new metric

A `KPIBundle` is any object with a `name` and `compute(records)`, constructible with no
arguments. It can live in your own file:

```python
# my_metrics.py
from collections.abc import Sequence

from hai_sim.metrics.collector import PatientRecord


class MissedPatients:
    """Patients with at least one finding the team missed."""

    name = "missed"

    def compute(self, records: Sequence[PatientRecord]) -> dict[str, float]:
        missed = sum(
            any(call == 0 and truth == 1 for call, truth in zip(r.doc_labels, r.ground_truth))
            for r in records
        )
        return {"n_patients": len(records), "patients_with_a_miss": missed}
```

```yaml
kpis: [system, actors, my_metrics.py:MissedPatients]
```

The report is keyed by the bundle's `name`, so a study reaches the metric as
`missed.patients_with_a_miss`, and any [`kpi_groups`](scenarios.md#kpi_groups) axis splits it
like a built-in bundle. A record exposes the fixed fields (`ground_truth`, `ai_labels`,
`ai_confidence`, `ai_uncertainty`, `doc_labels`, `doctor_solo_labels`, the timestamps,
`severity`) and every observation through `record.observation("ai.mutual_information")`.
A bundle can also be registered under a short name in Python with
`hai_sim.metrics.registry.register_bundle`.

[`configs/examples/custom_metrics.py`](../configs/examples/custom_metrics.py) holds two
worked examples: `CostBundle` (misses weighted 10:1 against false alarms, per actor) and
`UncertaintySplitBundle` (mean epistemic and aleatoric uncertainty on the AI's errors and on
its correct calls, read from the observation bag; it needs `ai.uq: ensemble`).

### A new dataset

A folder of images with a CSV of labels needs no code: use
`patient_generator.kind: image_folder` with one or more `label_sources`, and a `map` from the
CSV's label space to the model's ([scenarios](scenarios.md#patient_generatorlabel_sources)).
Anything else implements the `Dataset` protocol: indexable `DatasetExample`s (image and
ground truth, with −1 for a finding the dataset does not annotate) and a
`DatasetCapabilities` that declares what else it provides.

## Limitations

- **Simulated readers.** The readers are signal-detection observers with a fixed operating
  point, blind to the image. Their errors are independent of the AI's, so the benefit of
  the advice they show is an upper bound. Replaying the reads of real physicians
  (`capability.source: reader`) is reserved and raises.
- **Weights of advice are set, not measured.** Only the ordering senior < junior is anchored
  in the literature; no study has measured the weight of advice for radiological reads, or
  in log-odds space ([calibration and fusion](calibration-and-fusion.md#assumptions-without-a-source)).
- **Fatigue and case difficulty** are implemented and tested, but no shipped study uses
  them. The fatigue defaults assume one time unit is one hour.
- **Simulated time has no unit.** Reading times are a log-normal modelling choice; waits are
  best read in multiples of the mean reading time.
- **One arrangement.** The AI reads in series, before one reader per station. A parallel
  arrangement, or a population of readers, is not modelled.
- **Small calibration samples** for a few findings (fewer than 40 positives in the fitting
  half), listed in [calibration and fusion](calibration-and-fusion.md#low-support-calibrations).
- **The two MedGemma models differ in generation as well as size**: the 4B is MedGemma 1.5,
  the 27B is the first generation.
