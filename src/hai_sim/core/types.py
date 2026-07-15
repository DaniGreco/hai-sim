"""Core data types shared across the framework.

All actors (AI model, doctor digital twin) communicate through these types.
Multi-label by default: every per-patient output is a vector of length
``len(schema.labels)``.
"""

from __future__ import annotations

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
    """Names of the labels (pathologies) the simulation reasons about.

    For v0 the mocks use 5 fake labels. Switch to the 14 CheXpert names when
    a real model is wired in.
    """

    labels: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.labels)

    @classmethod
    def default_v0(cls) -> LabelSchema:
        return cls(labels=("L0", "L1", "L2", "L3", "L4"))

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

    ``uncertainty`` is a per-label vector. The aggregation (mean, max,
    weighted) is the SortStrategy's job: don't collapse it here. For v0 the
    uncertainty value mixes aleatoric and epistemic — splitting them is
    deferred to a later iteration.

    ``extra`` is for implementation-specific debug info (token logprobs from
    an LLM, ensemble disagreement matrix, etc.). Stable consumers should not
    rely on it.

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


@dataclass(frozen=True)
class PatientContext:
    """Optional metadata an actor may use to modulate its decision.

    Kept deliberately minimal in v0 — the doctor twin will grow this when
    fatigue and case-complexity modifiers come in.

    ``ground_truth`` is a *simulation device*, not clinical information: it is
    injected by the engine only into the context handed to the doctor stage so
    that *simulated* readers (e.g. ``FixedAccuracyDoctor``) can corrupt the true
    label at a configured error rate. Real image-reading actors (the CNN
    wrapper) must ignore it. It is ``None`` for actors that should not see it.
    """

    patient_id: int
    arrival_time: float
    age: float | None = None
    sex: str | None = None
    priority: int = 0
    ground_truth: tuple[int, ...] | None = None


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
