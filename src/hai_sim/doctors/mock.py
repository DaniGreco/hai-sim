"""Mock doctor digital twin.

Two modes for v0:
- ``blind``: ignores any AI hint.
- ``noisy_oracle``: peeks at the patient's ground truth (passed via context not
  yet — for v0 we keep it blind; the noisy_oracle variant lands when the
  experiment harness needs a 'doctor better than the AI mock' baseline).

Service time ~ lognormal. Confidence skewed toward correctness via Beta(5, 2).
No fatigue model yet — that's the next iteration.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext


class RandomMockDoctor(BaseDiagnosticActor):
    """Doctor mock: slightly better than uniform random.

    Confidence ~ Beta(5, 2) → mean ≈ 0.71, biased toward decisive calls.
    Decision threshold at 0.5. Service time ~ lognormal(mu, sigma).
    Accepts but ignores ``ai_hint`` in v0; the hook is there for the assisted
    variant.
    """

    def __init__(
        self,
        schema: LabelSchema,
        rng: np.random.Generator | None = None,
        service_time_mu: float = 2.5,
        service_time_sigma: float = 0.4,
        alpha_conf: float = 5.0,
        beta_conf: float = 2.0,
        alpha_unc: float = 2.0,
        beta_unc: float = 5.0,
        use_ai_hint: bool = False,
        name: str = "RandomMockDoctor",
    ) -> None:
        super().__init__(name=name)
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.service_time_mu = service_time_mu
        self.service_time_sigma = service_time_sigma
        self.alpha_conf = alpha_conf
        self.beta_conf = beta_conf
        self.alpha_unc = alpha_unc
        self.beta_unc = beta_unc
        self.use_ai_hint = use_ai_hint

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        n = self.schema.n
        confidence = self.rng.beta(self.alpha_conf, self.beta_conf, size=n)
        uncertainty = self.rng.beta(self.alpha_unc, self.beta_unc, size=n)
        labels = (confidence >= 0.5).astype(int)
        extra: dict[str, object] = {"_mock": True}
        if self.use_ai_hint and ai_hint is not None:
            # Hook for assisted mode; v0 just records that the hint was seen.
            extra["ai_hint_seen"] = True
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in confidence),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra=extra,
        )

    def service_time(
        self,
        images: Sequence[Image],
        context: PatientContext | None = None,
    ) -> float:
        return float(self.rng.lognormal(self.service_time_mu, self.service_time_sigma))
