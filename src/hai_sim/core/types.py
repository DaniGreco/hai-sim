"""Core data types shared across the framework.

All actors (AI model, doctor digital twin) communicate through these types.
Multi-label by default: every per-patient output is a vector of length
``len(schema.labels)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

Image: TypeAlias = np.ndarray
"""An image is an ``np.ndarray``. Shape/dtype are implementation-defined.

The mocks use ``np.zeros((1, 1), dtype=np.uint8)`` as placeholder. Real wrappers
(TorchXRayVision, CheXNet, LLM vision SDKs) load and preprocess internally.
"""


@dataclass(frozen=True)
class LabelSchema:
    """The pathologies this run reasons about, in a fixed order.

    Every per-patient vector in the framework — predictions, confidences, ground truth —
    is indexed by this order, so the schema is what keeps the AI, the doctor, the dataset
    and the metrics talking about the same findings.

    Pick the one matching your model: `xrv` for the pre-trained chest X-ray classifier
    (18 pathologies), `chexpert` for the 14-finding CheXpert convention, `default_v0` for
    the 5 fake labels the mocks use in tests and smoke runs.
    """

    labels: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.labels)

    @classmethod
    def default_v0(cls) -> LabelSchema:
        """Five nameless labels for mocks, tests and smoke runs. Not clinical."""
        return cls(labels=("L0", "L1", "L2", "L3", "L4"))

    @classmethod
    def single(cls) -> LabelSchema:
        """One label — the single-target study frame.

        A synthetic single-finding task, aligned with classical signal detection theory
        (one binary target) and the cleanest frame for a trust study, where a
        multi-label doctor only muddies whether the AI was heeded on *this* call. Named
        ``Finding`` rather than a real pathology because it is a controlled abstraction,
        not a clinical claim."""
        return cls(labels=("Finding",))

    @classmethod
    def chexpert(cls) -> LabelSchema:
        return cls(
            labels=(
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            )
        )

    @classmethod
    def xrv(cls) -> LabelSchema:
        """The 18 pathologies of a TorchXRayVision DenseNet, in model-output order.

        Order MUST match ``xrv.models.DenseNet().pathologies`` so the model's
        output vector indexes line up with this schema 1:1. Do not reorder.
        """
        return cls(
            labels=(
                "Atelectasis",
                "Consolidation",
                "Infiltration",
                "Pneumothorax",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Effusion",
                "Pneumonia",
                "Pleural_Thickening",
                "Cardiomegaly",
                "Nodule",
                "Mass",
                "Hernia",
                "Lung Lesion",
                "Fracture",
                "Lung Opacity",
                "Enlarged Cardiomediastinum",
            )
        )

    @classmethod
    def vindr(cls) -> LabelSchema:
        """The 14 findings of the VinDr-CXR / VinBigData competition, in class-id order.

        A *source* label space, not a model one: no model here predicts in it. It exists
        so a VinDr label CSV can be read in its own vocabulary and then mapped into the
        model schema (`xrv`) by a `LabelMap`, rather than being silently pre-aligned. Four
        of these have no `xrv` counterpart (Aortic enlargement, Calcification, ILD, Other
        lesion) and are simply dropped by the map; the rest rename or split into it.
        """
        return cls(
            labels=(
                "Aortic enlargement",
                "Atelectasis",
                "Calcification",
                "Cardiomegaly",
                "Consolidation",
                "ILD",
                "Infiltration",
                "Lung Opacity",
                "Nodule/Mass",
                "Other lesion",
                "Pleural effusion",
                "Pleural thickening",
                "Pneumothorax",
                "Pulmonary fibrosis",
            )
        )


@dataclass(frozen=True)
class Diagnosis:
    """The 'classification + confidence' pair from an actor.

    ``labels[i] == 1`` means the actor positively classifies pathology ``i``.
    ``confidence[i]`` is the actor's P(label_i = 1 | input), in [0, 1].
    """

    labels: tuple[int, ...]
    confidence: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.confidence):
            raise ValueError(
                f"labels (n={len(self.labels)}) and confidence (n={len(self.confidence)}) "
                f"must have the same length"
            )


@dataclass(frozen=True)
class ModelOutput:
    """The full per-patient output from a `DiagnosticActor`.

    ``uncertainty`` stays a **per-label vector**: collapsing it to one number is the
    `SortStrategy`'s job, not this type's, because different triage rules want to
    collapse it differently (mean across findings, worst finding, clinically weighted).

    What the number means depends on the `UQMethod` that produced it. A Deep Ensemble
    can separate *aleatoric* uncertainty (irreducible ambiguity in the image) from
    *epistemic* (the model's own ignorance); this field carries whichever measure was
    selected, and the full decomposition travels in ``extra``.

    ``extra`` is the actor's **observation channel**. Whatever a module puts
    here reaches the telemetry, namespaced by the producing stage (``ai.*`` for
    the AI station, ``doc.*`` for the doctor station), and is therefore readable
    by any KPI bundle *without* changing the record schema. That is how a new
    module surfaces a new observable — the aleatoric/epistemic decomposition, a
    cache index, a latent difficulty draw — with no edit to the library. Keys
    should be short and stable; values should be scalars, strings or per-label
    sequences, so they flatten into CSV columns cleanly. Anything else is still
    accepted and stored as JSON.

    ``embedding`` is the model's penultimate feature vector (the input to the
    final linear head), or ``None`` for actors that don't expose one. It is the
    natural substrate for feature-space severity and feature-space uncertainty
    (Mahalanobis / DDU / kNN), which read the geometry of the representation
    rather than the output probabilities.
    """

    diagnosis: Diagnosis
    uncertainty: tuple[float, ...]
    extra: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.uncertainty) != len(self.diagnosis.labels):
            raise ValueError(
                f"uncertainty (n={len(self.uncertainty)}) must match diagnosis "
                f"labels (n={len(self.diagnosis.labels)})"
            )

    @classmethod
    def from_arrays(
        cls,
        labels: Sequence[int] | np.ndarray,
        confidence: Sequence[float] | np.ndarray,
        uncertainty: Sequence[float] | np.ndarray,
        *,
        extra: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
    ) -> ModelOutput:
        """Build a `ModelOutput` from raw per-label arrays.

        The single home of the ``tuple(int(...))`` / ``tuple(float(...))`` coercion
        every actor performed by hand: an int-valued label vector, a float confidence
        vector and a float uncertainty vector, in the canonical immutable form. Keeping
        it here means the model and doctor wrappers share one int/float contract instead
        of restating it at each construction site.
        """
        return cls(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in confidence),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra=extra if extra is not None else {},
            embedding=embedding,
        )


@dataclass(frozen=True)
class PatientContext:
    """Optional metadata an actor may use to modulate its decision.

    ``ground_truth`` is a *simulation device*, not clinical information: it is
    injected by the engine only into the context handed to the doctor stage so
    that *simulated* readers (e.g. ``FixedAccuracyDoctor``) can corrupt the true
    label at a configured error rate. Real image-reading actors (the CNN
    wrapper) must ignore it. It is ``None`` for actors that should not see it.

    ``patient_id`` and ``example_id`` are deliberately **two different things**:
    ``patient_id`` counts arrivals in this run (unique per patient, even when the
    same image is sampled twice), while ``example_id`` names the underlying
    dataset item (stable across runs). Keeping them apart is what lets telemetry
    be joined back to the image — and what lets a cache-backed actor look up the
    predictions for the image the patient is carrying.

    ``signals`` is the **signal plane**: the delivery channel for per-read scalars
    computed by the engine between the two stations and handed to the doctor. It
    exists because the interesting modulators of a reader are not properties of
    the patient record — they are computed, and they have several competing
    implementations each. Well-known keys:

    - ``difficulty`` — how hard this case is to read, from a `DifficultyModel`.
      The latent ``delta`` of thesis §5.7, and the only route by which a
      simulated reader's error can be made to correlate with the AI's.
    - ``fatigue`` — how worn the reader is *at the moment it picks this patient
      up*, from a `FatigueCurve` over the station's `ReaderState`. A property of
      the reader and the clock, not of the case, which is why it is stamped per
      read rather than carried by the patient.
    - ``severity`` — the clinical weight from a `SeverityModel`, mirrored here so
      a reader can consult it (it is also a first-class telemetry field).

    Consumers must treat every key as optional: a scenario that configures no
    difficulty model simply leaves it out, and a reader written against it has to
    keep working. The whole bag reaches the telemetry under the ``sig.*``
    namespace, so any of it can condition a metric.

    Excluded from equality and hashing on purpose: signals are *derived from* a
    case-and-reader pairing, not part of the case's identity, and keeping a
    mutable mapping out of ``__hash__`` keeps the context usable as a dict key.
    """

    patient_id: int
    arrival_time: float
    age: float | None = None
    sex: str | None = None
    priority: int = 0
    ground_truth: tuple[int, ...] | None = None
    example_id: str | None = None
    signals: Mapping[str, float] = field(default_factory=dict, compare=False)

    def signal(self, key: str, default: float = 0.0) -> float:
        """Read one signal, or ``default`` when this run does not produce it.

        Always use this rather than indexing ``signals``: which signals exist is a
        scenario choice, so a reader that hard-indexes breaks the moment someone
        runs it without a difficulty model configured.
        """
        value = self.signals.get(key)
        return default if value is None else float(value)


@dataclass(frozen=True)
class ReaderState:
    """What a reading station has done so far — the substrate a `FatigueCurve` reads.

    Owned and advanced by the engine (one instance per worker), never by the actor:
    two doctors working the same queue tire independently, and an actor that counted
    its own reads would conflate them. Immutable; the engine replaces it after each
    completed read.

    Three different clocks, deliberately kept apart because different fatigue
    literatures use different ones:

    - ``clock`` — absolute simulation time. What a **circadian** curve needs.
    - ``time_on_shift`` — elapsed time since this worker started, derived from
      ``clock - shift_start``. What a **time-on-task** curve needs.
    - ``busy_time`` — time actually spent reading, so ``busy_time /
      time_on_shift`` is the worker's utilisation. What a curve driven by
      *workload* rather than by the clock needs.

    ``cases_read`` is the fourth, and the one the radiology literature usually
    reports against (accuracy after N studies).
    """

    cases_read: int = 0
    clock: float = 0.0
    shift_start: float = 0.0
    busy_time: float = 0.0

    @property
    def time_on_shift(self) -> float:
        return max(0.0, self.clock - self.shift_start)

    @property
    def utilisation(self) -> float:
        """Fraction of the shift spent reading; 0.0 before any time has passed."""
        elapsed = self.time_on_shift
        return self.busy_time / elapsed if elapsed > 0.0 else 0.0

    def advanced(self, *, now: float, service_time: float) -> ReaderState:
        """Return the state after one completed read of length ``service_time``."""
        return ReaderState(
            cases_read=self.cases_read + 1,
            clock=now,
            shift_start=self.shift_start,
            busy_time=self.busy_time + service_time,
        )


@dataclass(frozen=True)
class Patient:
    """A patient in the simulation pipeline.

    ``ground_truth`` is the dataset label, one entry per schema label. Entries
    are ``1`` (present), ``0`` (absent), or ``-1`` (not annotated by this
    dataset for this patient — real CXR datasets only label a subset of the
    model's pathologies). Metrics skip ``-1`` entries.
    """

    id: int
    images: tuple[Image, ...]
    ground_truth: tuple[int, ...]
    context: PatientContext


@dataclass(frozen=True)
class LocalizationExample:
    """One image for spatial + label validation: model-ready pixels, expert
    image-level labels, and expert per-label localization masks, all aligned to a
    `LabelSchema`.

    ``image`` is preprocessed to the model's input geometry (e.g. ``(1, 224, 224)``
    for the xrv DenseNet). ``masks[i]`` is a boolean array **co-registered with
    that geometry** (same HxW as the model's saliency map) marking where label
    ``i`` is localized, or ``None`` when this dataset does not localize label ``i``
    for this image. ``labels[i]`` is the expert image-level call in the
    ``{1, 0, -1}`` convention (``-1`` = not annotated for label ``i`` by this
    dataset). Keeping masks in model geometry is deliberate: the concrete dataset
    owns the fiddly crop/resize co-registration, so the validator can overlay
    heatmap and mask without knowing the transform.
    """

    image_id: str
    image: Image
    labels: tuple[int, ...]
    masks: tuple[np.ndarray | None, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.masks):
            raise ValueError(
                f"labels (n={len(self.labels)}) and masks (n={len(self.masks)}) "
                f"must have the same length (one entry per schema label)"
            )


@dataclass(frozen=True)
class DatasetCapabilities:
    """Declarative descriptor of what a dataset provides.

    The framework *gates* features on this descriptor instead of hardcoding
    dataset names: which ground-truth sources are selectable (``gt_sources``),
    whether a persona can be calibrated on per-reader reads (``has_per_reader`` /
    ``reader_ids``), whether localization can be validated (``has_masks``), and
    what extra per-example metadata is available (``metadata_fields``). A scenario
    requesting a capability the dataset lacks fails fast with a clear error.
    """

    gt_sources: tuple[str, ...] = ("native",)
    has_per_reader: bool = False
    reader_ids: tuple[str, ...] = ()
    has_masks: bool = False
    metadata_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetExample:
    """One dataset item: image + native ground truth, plus optional provisions.

    ``masks`` / ``reader_labels`` / ``label_sources`` / ``metadata`` are populated
    only when the dataset advertises the matching capability; consumers check
    `DatasetCapabilities` before relying on them. ``ground_truth`` uses the
    ``{1, 0, -1}`` convention (``-1`` = not annotated), aligned to the dataset
    schema. ``reader_labels`` maps a reader id to that reader's label vector.

    ``label_sources`` is the general form: a name → label-vector map for the
    several *alternative* ground-truth sources one image can carry (e.g. a
    dataset's own labels, an adjudicated consensus, a mapped external CSV), each
    already aligned to the model schema. It is what lets ``gt_source`` select
    which annotations a run believes, and ``reader_labels`` is its per-reader
    special case. A dataset advertises the available names in
    ``DatasetCapabilities.gt_sources``.
    """

    example_id: str
    image: Image
    ground_truth: tuple[int, ...]
    masks: tuple[np.ndarray | None, ...] | None = None
    reader_labels: dict[str, tuple[int, ...]] | None = None
    label_sources: dict[str, tuple[int, ...]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
