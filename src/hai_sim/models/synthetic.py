"""A synthetic AI reader with a dial for how good it is — the controllable "good model".

AI = artificial intelligence (the automated reader). The studies this serves ask what a
doctor does when the AI *performs well*: whether it is trusted, and how that varies with the
doctor's own capability (``../TESI.md`` §3, RQ3). Answering that cleanly wants the AI's
skill to be a **controlled variable**, not whatever a particular pre-trained network happens
to score on a particular sample — so the AI here is a signal-detection observer
(`hai_sim.sdt`) whose discriminability ``d'`` the scenario sets, and can sweep. A large
``d'`` is a strong reader; sweeping it maps out the whole "how good does the AI have to be
before trusting it pays / before over-trust hurts" surface that a fixed model cannot.

Being an SDT observer, it emits a **genuine, calibrated probability**, not a decision with a
detached confidence — which is what an `IntegrationPolicy` needs from the AI side and what a
log-odds fusion assumes. Its errors are drawn from its own noise, independent of the
doctor's, which is the low-``rho`` regime where a team can genuinely beat its members
(§5, §8.4): the complementarity a good real model showed by measurement, here by
construction.

**How it sees the truth without cheating.** A *real* image model must never read the ground
truth, and the engine enforces that by handing the AI stage a context with none. A synthetic
reader still needs to know the case to realize its error rate — so it reads it from the
*image*, which in a synthetic run legitimately carries the case's true state (the patient
generator's ``embed_truth`` writes the label vector into the pixels). The AI reads pixels,
as it must; those pixels just happen to be the diagnostic signal itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim import sdt
from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.uq.methods import binary_entropy


class FixedAccuracyAI(BaseDiagnosticActor):
    """A signal-detection AI reader whose skill is a configured ``d'`` (or sens/spec).

    Reads the true label vector from ``images[0]`` (written there by a patient generator
    with ``embed_truth`` set), draws one SDT read of it, and reports the emitted call plus
    a genuine belief and its entropy. ``service_time`` is a small constant: the AI is not
    the queue's bottleneck, the doctor is.
    """

    def __init__(
        self,
        schema: LabelSchema,
        rng: np.random.Generator | None = None,
        d_prime: float | Sequence[float] | None = None,
        criterion: float | Sequence[float] | None = None,
        sensitivity: float | Sequence[float] = 0.9,
        specificity: float | Sequence[float] = 0.9,
        prior: float | Sequence[float] = 0.5,
        service_time: float = 0.05,
        name: str = "FixedAccuracyAI",
    ) -> None:
        super().__init__(name=name, service_time=service_time)
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.d_prime, self.criterion, self.prior = sdt.resolve_observer(
            schema.n,
            d_prime=d_prime,
            criterion=criterion,
            sensitivity=sensitivity,
            specificity=specificity,
            prior=prior,
        )

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        if not images:
            raise ValueError("FixedAccuracyAI needs an image carrying the embedded truth")
        y = np.rint(np.asarray(images[0], dtype=float)).astype(int).ravel()
        if y.shape != (self.schema.n,):
            raise ValueError(
                f"embedded-truth image has {y.shape[0]} entries != schema.n "
                f"{self.schema.n}; set patient_generator.embed_truth and match the schema"
            )
        call, belief = sdt.observe(
            y,
            d_prime=self.d_prime,
            criterion=self.criterion,
            prior=self.prior,
            rng=self.rng,
        )
        return ModelOutput.from_arrays(
            call, belief, binary_entropy(belief), extra={"_synthetic_ai": True}
        )
