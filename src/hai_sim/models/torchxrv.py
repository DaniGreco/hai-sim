"""Real pre-trained CXR model wrapper around TorchXRayVision (inference only).

Wraps one or more TorchXRayVision DenseNet121 weight sets behind the framework's
`DiagnosticActor` protocol. No training ever happens here — only ``forward()``.

- A single weight set + ``EntropyUQ`` gives a fast, single-pass baseline.
- Several weight sets + ``EnsembleUQ`` form a cross-domain Deep Ensemble whose
  member disagreement is a genuine epistemic uncertainty signal.

``torch`` / ``torchxrayvision`` are imported lazily so the core package stays
importable without the heavy ``[models]`` extra installed.

The output pathology vector is aligned to ``LabelSchema.xrv()`` (18 pathologies,
model-output order). Build the scenario with ``label_schema: xrv`` so model
indices and schema indices line up 1:1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.uq.methods import EntropyUQ

if TYPE_CHECKING:
    from hai_sim.core.protocols import UQMethod

# Default cross-domain ensemble: weight sets trained on different source datasets.
DEFAULT_ENSEMBLE_WEIGHTS = (
    "densenet121-res224-nih",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-pc",
    "densenet121-res224-rsna",
)


class TorchXRayVisionAI(BaseDiagnosticActor):
    """Inference-only TorchXRayVision DenseNet121 actor with pluggable UQ."""

    def __init__(
        self,
        schema: LabelSchema,
        weights: str | Sequence[str] = "densenet121-res224-all",
        uq: UQMethod | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 0.05,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        weight_list = [weights] if isinstance(weights, str) else list(weights)
        if not weight_list:
            raise ValueError("at least one weight set is required")
        resolved_name = name if name is not None else f"TorchXRayVisionAI({'+'.join(weight_list)})"
        super().__init__(name=resolved_name, service_time=service_time)

        if schema.n != 18:
            raise ValueError(
                f"TorchXRayVisionAI expects the 18-pathology xrv schema "
                f"(LabelSchema.xrv()), got schema.n={schema.n}. Use label_schema: xrv."
            )
        self.schema = schema
        self.weight_list = weight_list
        self.uq: UQMethod = uq if uq is not None else EntropyUQ()
        self.decision_threshold = decision_threshold
        self.device = device

        self._torch: Any = None
        self._transform: Any = None
        self._models: list[Any] = []
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torchvision
            import torchxrayvision as xrv
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "TorchXRayVisionAI needs the [models] extra: "
                'uv pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        for w in self.weight_list:
            model = xrv.models.DenseNet(weights=w)
            model.eval()
            model.to(self.device)
            # Sanity: every member must share the canonical 18-pathology order.
            if tuple(model.pathologies) != self.schema.labels:
                raise ValueError(
                    f"weight set {w!r} pathologies do not match LabelSchema.xrv()"
                )
            self._models.append(model)

    def _to_model_input(self, image: Image) -> np.ndarray:
        """Return a (1, 224, 224) float array in xrv's expected range.

        If the image is already xrv-preprocessed (shape (1, 224, 224)) it is
        passed through; otherwise raw grayscale is normalized, single-channeled,
        center-cropped and resized.
        """
        img = np.asarray(image)
        if img.shape == (1, 224, 224):
            return img.astype(np.float32)

        import torchxrayvision as xrv

        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.ndim != 2:
            raise ValueError(f"expected 2D grayscale image, got shape {img.shape}")
        # Heuristic: values clearly above 1 look like 0..255 pixel intensities.
        max_val = float(img.max()) if img.size else 0.0
        img = xrv.datasets.normalize(img, 255 if max_val > 1.5 else 1.0)
        img = img[None, ...]  # add channel dim → (1, H, W)
        img = self._transform(img)
        return np.asarray(img, dtype=np.float32)

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        torch = self._torch
        model_input = self._to_model_input(images[0])
        tensor = torch.from_numpy(model_input)[None, ...].to(self.device)  # (1,1,224,224)

        member_probs = np.empty((len(self._models), self.schema.n), dtype=np.float64)
        with torch.no_grad():
            for m, model in enumerate(self._models):
                out = model(tensor)  # xrv DenseNet returns sigmoid probabilities
                member_probs[m] = out[0].detach().cpu().numpy()

        probs, uncertainty, uq_extra = self.uq.estimate(member_probs)
        labels = (probs >= self.decision_threshold).astype(int)

        extra: dict[str, Any] = {"uq_method": self.uq.name, **uq_extra}
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in probs),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra=extra,
        )
