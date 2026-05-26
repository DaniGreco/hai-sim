"""Mock AI model: random confidence + random uncertainty per label."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext


class RandomMockAI(BaseDiagnosticActor):
    """Confidence ~ Beta(alpha_conf, beta_conf), uncertainty ~ Beta(alpha_unc, beta_unc).

    Independent per label. Decision threshold at 0.5 on confidence. Useful as a
    sanity-check actor to validate the pipeline; replace with a real wrapper
    (TorchXRayVision, CheXNet, LLM) when the inference layer lands.
    """

    def __init__(
        self,
        schema: LabelSchema,
        rng: np.random.Generator | None = None,
        service_time: float = 0.05,
        alpha_conf: float = 2.0,
        beta_conf: float = 2.0,
        alpha_unc: float = 2.0,
        beta_unc: float = 5.0,
        name: str = "RandomMockAI",
    ) -> None:
        super().__init__(name=name, service_time=service_time)
        self.schema = schema
        self.rng = rng if rng is not None else np.random.default_rng()
        self.alpha_conf = alpha_conf
        self.beta_conf = beta_conf
        self.alpha_unc = alpha_unc
        self.beta_unc = beta_unc

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
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in confidence),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra={"_mock": True},
        )
