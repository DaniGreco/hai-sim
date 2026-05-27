"""Mock doctor digital twin.

Service time ~ lognormal. Confidence skewed toward correctness via Beta(5, 2).
No fatigue model yet — that's the next iteration.

Hybrid mode: when ``use_ai_hint=True`` and an `ai_hint` is provided, an
`AssistedPolicy` decides whether to defer to the AI on a per-label basis. v0
ships one minimal policy:

- ``noop``: the hint is recorded in `extra` but the doctor's decision is
  unchanged. Useful to verify wiring without affecting KPIs.
- ``follow_when_certain``: on each label, if the AI's uncertainty is below
  ``ai_hint_uncertainty_threshold`` the doctor copies the AI's label;
  otherwise the doctor uses its own sampled decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext

AssistedPolicy = Literal["noop", "follow_when_certain"]


class RandomMockDoctor(BaseDiagnosticActor):
    """Doctor mock: slightly better than uniform random, optionally AI-assisted.

    Confidence ~ Beta(5, 2) → mean ≈ 0.71, biased toward decisive calls.
    Decision threshold at 0.5. Service time ~ lognormal(mu, sigma).
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
        assisted_policy: AssistedPolicy = "noop",
        ai_hint_uncertainty_threshold: float = 0.3,
        name: str = "RandomMockDoctor",
    ) -> None:
        super().__init__(name=name)
        if not 0.0 <= ai_hint_uncertainty_threshold <= 1.0:
            raise ValueError(
                f"ai_hint_uncertainty_threshold must be in [0,1], got {ai_hint_uncertainty_threshold}"
            )
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.service_time_mu = service_time_mu
        self.service_time_sigma = service_time_sigma
        self.alpha_conf = alpha_conf
        self.beta_conf = beta_conf
        self.alpha_unc = alpha_unc
        self.beta_unc = beta_unc
        self.use_ai_hint = use_ai_hint
        self.assisted_policy: AssistedPolicy = assisted_policy
        self.ai_hint_uncertainty_threshold = ai_hint_uncertainty_threshold

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
            extra["ai_hint_seen"] = True
            if self.assisted_policy == "follow_when_certain":
                # Per-label deferral: on labels where the AI is confident
                # (uncertainty below threshold), adopt the AI's call.
                ai_labels = ai_hint.diagnosis.labels
                ai_unc = ai_hint.uncertainty
                followed: list[int] = []
                merged = list(labels.tolist())
                for i, u in enumerate(ai_unc):
                    if u < self.ai_hint_uncertainty_threshold:
                        merged[i] = ai_labels[i]
                        followed.append(i)
                labels = np.array(merged, dtype=int)
                extra["ai_hint_followed_labels"] = tuple(followed)
            # "noop" → hint seen, no behavior change.

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
