"""The signal-detection doctor: a genuine observer, not a coin flip.

This is the SDT doctor of thesis §5.7 — the successor to
`FixedAccuracyDoctor`. Both are *simulated readers* that derive their decision from the
ground truth the engine injects into the doctor stage; the difference is what happens in
between. `FixedAccuracyDoctor` keeps or flips each label on a coin whose bias is a fixed
sensitivity/specificity, so its reported confidence is a constant decoupled from the
realized call. `SDTDoctor` instead draws noisy evidence for the case and forms a real
belief from it (`hai_sim.sdt`), so:

- its confidence is a probability that **varies case by case** — the property that makes an
  `IntegrationPolicy` a smooth reliance curve instead of a step (a Δ=0 measured in development);
- a strong reader (large ``d'``) more often lands on confident evidence and so *resists* a
  wrong AI, while a weak reader sits near the fence and is *moved* by it — appropriate
  reliance that scales with capability emerges from the model rather than being asserted;
- it reads the same ``sensitivity``/``specificity`` the persona library already states,
  which are inverted to ``(d', criterion)`` once at construction, so it is a drop-in for
  the existing personas.

**Difficulty and fatigue** enter as a loss of discriminability. ``difficulty_slope`` and
``fatigue_slope`` subtract from ``d'`` in proportion to the signal-plane values the engine
supplies (``sig.difficulty`` / ``sig.fatigue``), lowering the *generating* ``d'`` while the
belief keeps using the base one — so a reader worn down by a hard case stays as confident
as if it were easy, the over-confidence thesis §5.7 lets emerge. With both slopes at 0 (the
default) the reader is a pure observer whose errors are independent of the AI's; a scenario
that couples them through a shared `DifficultyModel` is what turns the human-AI error
correlation ``rho`` into a tunable lever (thesis §5.7).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim import sdt
from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.doctors._shared import lognormal_service_time, signal_penalty
from hai_sim.uq.methods import binary_entropy


class SDTDoctor(BaseDiagnosticActor):
    """An equal-variance Gaussian signal-detection reader over the ground truth.

    Capability is set either by ``d_prime`` directly or by a ``sensitivity``/
    ``specificity`` operating point (inverted to ``d'`` and criterion). ``prior`` is the
    base rate the belief is calibrated to; leave it at the prevalence to get a reader that
    is neither over- nor under-confident on average.
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
        difficulty_slope: float = 0.0,
        fatigue_slope: float = 0.0,
        service_time_mu: float = 2.5,
        service_time_sigma: float = 0.4,
        name: str = "SDTDoctor",
        service_time_model: object | None = None,
    ) -> None:
        super().__init__(name=name)
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
        if difficulty_slope < 0 or fatigue_slope < 0:
            raise ValueError("difficulty_slope and fatigue_slope must be >= 0")
        self.difficulty_slope = float(difficulty_slope)
        self.fatigue_slope = float(fatigue_slope)
        self.service_time_mu = service_time_mu
        self.service_time_sigma = service_time_sigma
        self.service_time_model = service_time_model

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        gt = context.ground_truth if context is not None else None
        if gt is None:
            raise ValueError(
                "SDTDoctor requires ground_truth in the context. The engine injects it "
                "into the doctor stage; check the wiring."
            )
        if len(gt) != self.schema.n:
            raise ValueError(f"ground_truth length {len(gt)} != schema.n {self.schema.n}")

        d_eff, penalty = self._effective_dprime(context)
        call, belief = sdt.observe(
            np.asarray(gt),
            d_prime=self.d_prime,
            criterion=self.criterion,
            prior=self.prior,
            rng=self.rng,
            d_prime_eff=d_eff,
        )
        extra: dict[str, object] = {"_sdt_doctor": True}
        if penalty > 0.0:
            # Published so a study can confirm the coupling actually bit rather than infer
            # it from an outcome: a run where a difficulty model is configured but this
            # stays 0 has a reader that ignores the signal.
            extra["dprime_penalty"] = penalty
        return ModelOutput.from_arrays(
            call, belief, binary_entropy(belief), extra=extra
        )

    def _effective_dprime(
        self, context: PatientContext | None
    ) -> tuple[np.ndarray | None, float]:
        """Lower ``d'`` by the signal plane's difficulty and fatigue, holding criterion."""
        penalty = signal_penalty(context, self.difficulty_slope, self.fatigue_slope)
        if penalty <= 0.0:
            return None, 0.0
        # Discriminability floors at 0 (chance): a reader that read *worse than noise* on
        # hard cases would be an artefact of unbounded arithmetic, not a model.
        return np.maximum(self.d_prime - penalty, 0.0), float(penalty)

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return lognormal_service_time(
            self.rng,
            self.service_time_mu,
            self.service_time_sigma,
            model=self.service_time_model,
            images=images,
            context=context,
        )
