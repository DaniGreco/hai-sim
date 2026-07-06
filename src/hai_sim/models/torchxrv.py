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
        # Per-member boolean mask over the 18 canonical positions: True where the
        # member was trained on that pathology (xrv leaves untrained slots blank).
        self._member_masks: list[np.ndarray] = []
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
            # Every member outputs an 18-vector in canonical order, but dataset-
            # specific members blank out the pathologies they weren't trained on
            # (e.g. RSNA covers only Pneumonia + Lung Opacity). Build a validity
            # mask and verify the non-blank slots line up with the schema order.
            paths = list(model.pathologies)
            if len(paths) != self.schema.n:
                raise ValueError(
                    f"weight set {w!r} has {len(paths)} pathologies, expected {self.schema.n}"
                )
            mask = np.zeros(self.schema.n, dtype=bool)
            for i, p in enumerate(paths):
                if p == "":
                    continue
                if p != self.schema.labels[i]:
                    raise ValueError(
                        f"weight set {w!r} pathology at index {i} is {p!r}; expected "
                        f"blank or {self.schema.labels[i]!r} (output order not aligned "
                        f"to LabelSchema.xrv())"
                    )
                mask[i] = True
            if not mask.any():
                raise ValueError(f"weight set {w!r} has no usable pathologies")
            self._models.append(model)
            self._member_masks.append(mask)

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

    def member_probabilities_batch(self, images: Sequence[Image]) -> np.ndarray:
        """Raw per-member positive-class probabilities for a batch of images.

        Returns an array of shape ``(n_images, n_members, n_labels)``. Entry
        ``[i, m, k]`` is member ``m``'s ``P(label_k = 1 | image_i)``, or ``NaN``
        where member ``m`` was not trained on pathology ``k`` (xrv blanks those
        slots). This is the inference output *before* any UQ reduction: cache it
        to recover every UQ measure / threshold / aggregation offline without
        re-running the model. Batching amortizes Python overhead across images.
        """
        torch = self._torch
        if len(images) == 0:
            return np.empty((0, len(self._models), self.schema.n), dtype=np.float64)
        # _to_model_input yields (1, 224, 224); stack to (B, 1, 224, 224).
        batch = np.stack([self._to_model_input(im) for im in images])
        tensor = torch.from_numpy(batch).to(self.device)
        n = len(images)
        # NaN marks "this member does not cover this pathology" → nan-aware UQ.
        member_probs = np.full((n, len(self._models), self.schema.n), np.nan, dtype=np.float64)
        with torch.no_grad():
            for m, model in enumerate(self._models):
                out = model(tensor)  # xrv DenseNet returns sigmoid probabilities, (B, n_labels)
                vals = out.detach().cpu().numpy()
                mask = self._member_masks[m]
                member_probs[:, m, mask] = vals[:, mask]
        return member_probs

    def member_probabilities(self, image: Image) -> np.ndarray:
        """Single-image convenience wrapper. Returns ``(n_members, n_labels)``."""
        return np.asarray(self.member_probabilities_batch([image])[0])

    @property
    def members(self) -> list[Any]:
        """The loaded ensemble member torch modules, in weight-list order.

        Exposed for explainability tooling (e.g. Grad-CAM) that needs the raw
        modules; inference itself goes through `member_probabilities_batch`.
        """
        return self._models

    @property
    def member_masks(self) -> list[np.ndarray]:
        """Per-member boolean coverage over schema labels (True = member trained on it)."""
        return self._member_masks

    def member_feature_maps_and_probs(self, image: Image) -> tuple[np.ndarray, np.ndarray]:
        """One forward per member: ``(probs (M, L), feature_maps (M, C, h, w))``.

        ``probs`` is the usual per-member probability vector (``NaN`` for uncovered
        pathologies); ``feature_maps`` is the spatial output of the DenseNet's last
        conv block, captured for free during the same forward. Forward only, no
        autograd: together with `member_classifier_weights` this is the raw material
        to reconstruct *any* label's Grad-CAM offline (the head is linear, so the
        Grad-CAM weights are analytic — see `cam_from_features`).
        """
        torch = self._torch
        arr = self._to_model_input(image)  # (1, 224, 224)
        tensor = torch.from_numpy(arr[None, ...]).to(self.device)  # (1, 1, 224, 224)
        n = len(self._models)
        probs = np.full((n, self.schema.n), np.nan, dtype=np.float64)
        maps: np.ndarray | None = None
        for m, model in enumerate(self._models):
            captured: dict[str, Any] = {}

            def _hook(_mod: Any, _inp: Any, out: Any, _c: dict[str, Any] = captured) -> None:
                _c["A"] = out

            handle = model.features.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    out = model(tensor)  # (1, L) probabilities
            finally:
                handle.remove()
            feature_map = captured["A"].detach().cpu().numpy()[0].astype(np.float32)  # (C, h, w)
            if maps is None:
                maps = np.full((n, *feature_map.shape), np.nan, dtype=np.float32)
            maps[m] = feature_map
            vals = out.detach().cpu().numpy()[0]
            mask = self._member_masks[m]
            probs[m, mask] = vals[mask]
        assert maps is not None  # at least one member always runs
        return probs, maps

    def member_classifier_weights(self) -> np.ndarray:
        """Per-member final linear-head weights, shape ``(M, L, C)`` (for offline CAMs)."""
        return np.stack(
            [
                np.asarray(m.classifier.weight.detach().cpu().numpy(), dtype=np.float32)
                for m in self._models
            ]
        )

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        member_probs = self.member_probabilities(images[0])
        probs, uncertainty, uq_extra = self.uq.estimate(member_probs)
        # A pathology no member covers comes back NaN → emit a benign negative.
        probs = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
        uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0)
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
