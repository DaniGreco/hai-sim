"""Structural contracts for swappable modules.

Protocols are the *public contract* of the framework. Anything that walks like
a `DiagnosticActor` is one — no inheritance required. Use the `BaseDiagnosticActor`
ABC in `hai_sim.core.base` if you want a base class with default behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hai_sim.core.types import (
    Image,
    LabelSchema,
    LocalizationExample,
    ModelOutput,
    Patient,
    PatientContext,
)

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class DiagnosticActor(Protocol):
    """Unified contract for any entity that produces a diagnosis from images.

    Implemented by AI models and by doctor digital twins. The `ai_hint` kwarg
    is only meaningful for doctors operating in assisted mode; AI models
    should accept it and ignore it.
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


# --- Validation plane -------------------------------------------------------
# The three contracts below make *offline component validation* (heatmap /
# label fidelity against expert ground truth) as swappable as the simulation
# modules above. They are the "lightweight spine": a model, a saliency method, a
# ground-truth dataset, and a metric compose into a validation run the same way
# an actor, a sort strategy and a queue compose into a simulation run.


@runtime_checkable
class SaliencyProvider(Protocol):
    """Optional model capability: a per-label spatial saliency map.

    Implemented *alongside* `DiagnosticActor` by models that can explain *where*
    they looked (e.g. Grad-CAM on a CNN). ``heatmap`` returns a ``(H, W)`` float
    array in the model's own input geometry, larger = more attributed to
    ``label_index`` (a position into the model's `LabelSchema`). A model that does
    not implement this stays label-validatable but not localization-validatable.
    """

    name: str

    def heatmap(self, image: Image, label_index: int) -> np.ndarray: ...


@runtime_checkable
class LocalizationDataset(Protocol):
    """A ground-truth source for spatial + label validation.

    Yields `LocalizationExample`s whose image-level labels and per-label masks are
    aligned to ``schema`` and co-registered with the model input geometry. Swap
    the concrete dataset (ChestX-Det, VinDr-CXR, CheXlocalize) without touching
    the validator.
    """

    name: str
    schema: LabelSchema

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> LocalizationExample: ...


@runtime_checkable
class LocalizationMetric(Protocol):
    """Scores how well a heatmap agrees with an expert mask (both in the same
    geometry): pointing-game hit-rate, IoU, Dice, energy-in-mask, ...

    Returns a scalar; higher = better agreement (``0..1`` for the built-ins). The
    metric owns any thresholding of the raw heatmap it needs.
    """

    name: str

    def score(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float: ...
