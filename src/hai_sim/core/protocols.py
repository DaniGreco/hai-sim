"""The seams — every place you can substitute your own piece.

These Protocols are the framework's public contract. They are *structural*: a class
satisfies one by having the right methods, with no inheritance and no registration. If
you want the default ``name``/``service_time`` behaviour for free, subclass
`BaseDiagnosticActor` in `hai_sim.core.base` — but you never have to.

Where each seam sits in the run, which is what makes them independently swappable:

    ArrivalProcess ─→ Dataset ─→ [ DiagnosticActor: the AI ] ─→ SortStrategy ─→
       (when)         (who)          │  UQMethod + Calibrator     (triage)
                                     │  SeverityModel                    │
                                     │  DifficultyModel  ─── signals ────┤
                                     │  FatigueCurve     ─── signals ────┤
                                     └──────────────────────────→ [ DiagnosticActor:
                                                    ServiceTimeModel   the doctor ]
                                                    IntegrationPolicy      │
                                                            KPIBundle · Grouping ──→

`DiagnosticActor` is deliberately one contract for **both** readers. A model and a
doctor differ in what they do, not in what they are asked for, and sharing the contract
is what lets a study put either of them at either station — or compare a doctor against
the model on identical terms.

**Simulation plane** — `DiagnosticActor` (reads images) · `IntegrationPolicy` (how much
the doctor trusts the model) · `SortStrategy` (what order the queue is worked in) ·
`UQMethod` (where the uncertainty number comes from) · `Calibrator` (whether that
probability can be believed) · `SeverityModel` (per-patient clinical weight) ·
`DifficultyModel` (how hard this case is) · `FatigueCurve` (how worn the reader is) ·
`ServiceTimeModel` (how long a read takes) · `ArrivalProcess` (when patients arrive) ·
`Dataset` (images and ground truth).

Two more live next to what they serve, in `hai_sim.metrics`: `KPIBundle` (a metric) and
`Grouping` (a conditioning axis).

*When does something here deserve a Protocol?* When you can name three implementations
you would actually run **and compare in the same study**. Every contract below clears
that bar; a cost model does not, because a cost matrix is already expressible as a
`KPIBundle`, and station topology does not, because there is only ever one arrangement
under test here (AI in series, before the doctor).

Acronyms: AI = artificial intelligence (the automated reader) · UQ = uncertainty
quantification · KPI = key performance indicator · NHPP = non-homogeneous Poisson
process. Full list in ``GLOSSARY.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hai_sim.core.types import (
    DatasetCapabilities,
    DatasetExample,
    Image,
    LabelSchema,
    ModelOutput,
    Patient,
    PatientContext,
    ReaderState,
)

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class DiagnosticActor(Protocol):
    """Anything that turns images into a diagnosis: an AI model, or a doctor.

    ``diagnose`` returns the per-label verdict; ``service_time`` says how long the actor
    occupies its station for this patient, which is what creates queues.

    ``ai_hint`` carries an upstream reader's opinion. Only the doctor stage is given one
    — it is how the AI's output reaches the doctor in the serial arrangement. **An AI
    model must accept the argument and ignore it**, both because nothing upstream feeds
    it one and because a model reading another model's answer is not what is being
    studied here.

    ``context`` carries the patient. Note that its ``ground_truth`` field is populated
    *only* for the doctor stage, as the mechanism that makes a simulated reader err at a
    chosen rate; an actor that reads real pixels must never consult it.
    """

    name: str

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput: ...

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float: ...


@runtime_checkable
class IntegrationPolicy(Protocol):
    """How a doctor integrates the AI's output into its own decision.

    The modular seam of the human-AI study: given the doctor's *own* (unaided)
    `ModelOutput` and the AI ``ai_hint``, return the doctor's *final* (aided)
    `ModelOutput`. Swap the policy — fixed-weight pool, uncertainty-weighted
    log-odds fusion, trust dynamics — without touching the doctor or the engine.
    Normative fusion rules and descriptive advice-taking rules are both just
    implementations of this one contract. Implementations should stash the
    unaided doctor labels in ``extra["doctor_solo_labels"]`` so the KPI layer can
    keep the solo counterfactual alongside the team verdict.
    """

    name: str

    def combine(
        self,
        doctor_out: ModelOutput,
        ai_hint: ModelOutput,
        context: PatientContext | None = None,
    ) -> ModelOutput: ...


@runtime_checkable
class SortStrategy(Protocol):
    """Computes the priority key used by the triaged PriorityStore.

    Lower priority value → served first (SimPy `PriorityStore` convention).
    Implementations decide how to collapse the per-label uncertainty vector
    into a scalar (mean, max, weighted, etc.).
    """

    name: str

    def priority(self, output: ModelOutput, patient: Patient) -> float: ...


@runtime_checkable
class UQMethod(Protocol):
    """Turns a stack of per-member probabilities into (probs, uncertainty).

    ``member_probs`` has shape ``(n_members, n_labels)``: one row per ensemble
    member (or one row for a single forward pass). The method returns the fused
    per-label probability vector and a per-label uncertainty vector, plus an
    ``extra`` dict of any auxiliary measures (e.g. the full entropy / mutual
    information decomposition). Keeping uncertainty per-label is deliberate: the
    `SortStrategy` is what collapses it to a scalar.
    """

    name: str

    def estimate(
        self, member_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]: ...


@runtime_checkable
class SeverityModel(Protocol):
    """Computes a per-patient severity signal, decoupled from its consumers.

    Runs after AI inference; the engine records the result in telemetry whether
    or not anything downstream (triage, KPI conditioning, cost) uses it — the
    signal is *produced*, its consumption is a separate concern. Returns a scalar
    (image-level severity) or a per-label ``np.ndarray`` in vector mode.
    Implementations may read the image (quality), the AI output (probabilities /
    uncertainty / embedding) and the ground truth (an oracle severity).
    """

    name: str

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray: ...


@runtime_checkable
class DifficultyModel(Protocol):
    """How hard this case is to read — the latent ``delta`` of the reader model.

    Same shape as `SeverityModel` and a *different quantity*: severity is how much
    it matters clinically, difficulty is how easy it is to get wrong. They are
    orthogonal — a large pneumothorax is severe and easy, a subtle nodule is mild
    and hard — so they travel as two signals, never one.

    Difficulty is the only route by which a *simulated* reader's errors can be made
    to correlate with the AI's. Without it a reader that flips labels at a fixed
    rate errs independently of the image by construction, which pins the human-AI
    error correlation ``rho`` at zero no matter what a scenario asks for. That is
    why this seam exists and why several implementations matter: the framework
    should not privilege one definition of "hard case".

    Runs after AI inference, so an implementation may read the model's predictive
    uncertainty and thereby *induce* the coupling from real data rather than impose
    it from a hand-chosen prior. Result reaches the
    doctor through ``PatientContext.signals["difficulty"]`` and the telemetry as
    ``sig.difficulty``, whether or not any reader consults it.

    Returns a scalar, or a per-label array when a case is hard for one finding and
    easy for another.
    """

    name: str

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray: ...


@runtime_checkable
class Calibrator(Protocol):
    """Maps raw model probabilities to calibrated ones — probabilities in, out.

    A model that assigns 0.95 to cases that are positive 4% of the time is not
    wrong about the *ranking*, only about the *number*; discrimination and
    calibration are independent, and both models measured so far are badly
    over-confident. Any rule that combines the doctor's belief with the AI's on a
    common scale — the log-odds fusion above all — is only as sound as that number,
    which is why this is a first-class seam rather than a preprocessing detail.

    Applied to the ``(n_members, n_labels)`` stack **before** fusion, so the
    uncertainty the `UQMethod` derives describes the probabilities the run actually
    uses. Calibrating after fusion would leave the two disagreeing.

    Implementations must be NaN-preserving: a label no ensemble member covers stays
    NaN rather than being mapped to a fabricated value.
    """

    name: str

    def calibrate(self, probs: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class FatigueCurve(Protocol):
    """How worn a reading station is, given what it has done — 0.0 fresh, 1.0 spent.

    Reads a `ReaderState` (cases read, time on shift, utilisation, wall clock) and
    returns a level in ``[0, 1]``. What the level *does* is a separate concern: it
    reaches the reader as ``PatientContext.signals["fatigue"]``, where a service-time
    model can stretch the read and a reader can degrade its own accuracy.

    This seam is not optional polish — no analytic fatigue curve exists for
    radiology (only isolated pre/post points), so the
    **shape of the curve is a mandatory sensitivity axis**. A mandatory
    sensitivity analysis over shapes is precisely a demand that the shape be
    swappable.
    """

    name: str

    def level(self, state: ReaderState) -> float: ...


@runtime_checkable
class ServiceTimeModel(Protocol):
    """How long one read occupies a station — the thing that creates the queue.

    Split out of the actor because reading time is where three separate open
    questions meet: whether AI assistance speeds a reader up or costs them
    verification time (the literature supports both signs, so the sign
    is a sensitivity axis), whether acute fatigue slows reading as well as degrading
    accuracy, and what an empirical reading-time
    distribution looks like. All three are answered by swapping this, not by editing
    the doctor.

    An actor without one keeps its own built-in timing, so this is additive.
    """

    name: str

    def sample(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float: ...


@runtime_checkable
class ArrivalProcess(Protocol):
    """When the next patient arrives — separated from *which* patient arrives.

    ``next_delay(now)`` returns the gap until the next arrival given the current
    arrival-clock time. Time-dependence is the whole point of the argument: a
    homogeneous process ignores it, while a non-homogeneous Poisson process (NHPP)
    needs it to know which hour's rate applies.

    Splitting *when* from *who* is what lets the same dataset stream be driven by a
    flat rate, a 24x7 piecewise-constant rate, or a deterministic metronome without
    touching the generator that samples images. Those are the three options, and
    the choice among them is conditional on a statistical test —
    a decision that has to be *testable*, i.e. swappable.
    """

    name: str

    def next_delay(self, now: float) -> float: ...


@runtime_checkable
class Dataset(Protocol):
    """Unified, capability-aware dataset contract (dataset-agnostic).

    A dataset provides indexable `DatasetExample`s (image + native ground truth,
    plus optional masks / per-reader reads / metadata) and *declares* what it
    provides via `DatasetCapabilities`. The simulation patient stream and persona
    calibration both read this one contract and gate
    their features on the declared capabilities — so adding a dataset means
    implementing this contract and declaring its provisions, nothing else.
    """

    name: str
    schema: LabelSchema
    capabilities: DatasetCapabilities

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> DatasetExample: ...
