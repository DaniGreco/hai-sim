"""A model that guesses — the null AI, for checking that a pipeline is wired correctly.

AI = artificial intelligence (the automated reader). This one does not look at the
image at all: it draws a confidence and an uncertainty per pathology at random. Its
purpose is to make the plumbing exercisable without installing a deep-learning stack,
so any *result* obtained with it is about the plumbing, never about reading chest X-rays.

For real predictions use `TorchXRayVisionAI` (runs a pre-trained classifier) or
`CachedEnsembleAI` (replays predictions computed earlier, which is what makes large
studies finish in seconds).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext


class RandomMockAI(BaseDiagnosticActor):
    """Confidence ~ Beta(alpha_conf, beta_conf), uncertainty ~ Beta(alpha_unc, beta_unc).

    Drawn independently per pathology, with the positive/negative call taken at a 0.5
    threshold on the confidence. Nothing here depends on the image.
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
        return ModelOutput.from_arrays(
            labels, confidence, uncertainty, extra={"_mock": True}
        )
