"""ChestX-Det ground-truth loader for spatial + label validation.

ChestX-Det (Lian et al.) re-annotates ~3.5k NIH ChestX-ray14 images with
instance-level **polygons + boxes** for 13 thoracic findings, drawn by three
board-certified radiologists. That gives us both halves of the dual ground truth
the framework needs: expert *localization* (masks) and expert *image-level
labels* (a finding is present iff a radiologist annotated it, absent otherwise) —
on the very images our NIH pipeline already uses.

This loader implements the `LocalizationDataset` protocol against the xrv label
schema. It owns the fiddly bit: expert geometry lives in the native image
resolution (1024x1024), while a saliency map comes out in the model's input
geometry (224x224 after xrv center-crop + resize). We push **both** the image and
each rasterized mask through the *same* ``XRayCenterCrop + XRayResizer`` transform
so heatmap and mask are pixel-co-registered and the metrics can overlay them
directly.

Label coverage: 11 of the 13 ChestX-Det categories map onto xrv pathologies
(below). ``Calcification`` and ``Diffuse Nodule`` have no clean xrv counterpart
and are left unmapped — they neither set a label nor a mask (mapping diffuse
miliary nodularity onto focal ``Nodule`` would be clinically wrong). xrv
pathologies ChestX-Det never annotates (Infiltration, Edema, Pneumonia, Hernia,
Lung Lesion, Lung Opacity, Enlarged Cardiomediastinum) are marked ``-1`` (not
annotated) so the label metrics skip them rather than count them as true
negatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hai_sim.core.types import Image, LabelSchema, LocalizationExample

# ChestX-Det category name -> xrv pathology name. Categories absent here
# (Calcification, Diffuse Nodule) are intentionally not validated.
CHESTXDET_TO_XRV: dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Fracture": "Fracture",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural Thickening": "Pleural_Thickening",
    "Pneumothorax": "Pneumothorax",
}


class ChestXDetDataset:
    """Expert masks + labels from ChestX-Det, aligned to a `LabelSchema`.

    Parameters
    ----------
    annotation_path:
        Path to ``ChestX_Det_train.json`` / ``ChestX_Det_test.json``.
    image_dir:
        Directory containing the ``file_name`` images referenced by the JSON.
    schema:
        Label schema to align to (default: the 18-label xrv schema). Only
        categories whose mapped xrv name exists in the schema are validated.
    resize:
        Model input side length; images and masks are co-registered to
        ``(resize, resize)``. Must match the model whose heatmaps you validate
        (224 for the xrv DenseNet).
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        schema: LabelSchema | None = None,
        resize: int = 224,
        name: str = "ChestX-Det",
    ) -> None:
        import json

        self.schema = schema if schema is not None else LabelSchema.xrv()
        self.image_dir = Path(image_dir)
        self.resize = resize
        self.name = name

        # xrv-index -> ChestX-Det category, for the categories this schema can hold.
        self._covered: dict[int, str] = {}
        for cat, xrv_name in CHESTXDET_TO_XRV.items():
            if xrv_name in self.schema.labels:
                self._covered[self.schema.labels.index(xrv_name)] = cat
        if not self._covered:
            raise ValueError(
                f"schema {self.schema.labels!r} shares no pathology with ChestX-Det; "
                "use LabelSchema.xrv()"
            )

        with open(annotation_path, encoding="utf-8") as fh:
            self._records: list[dict[str, Any]] = json.load(fh)

        self._xrv: Any = None
        self._transform: Any = None

    def _ensure_backends(self) -> None:
        if self._transform is not None:
            return
        try:
            import torchvision
            import torchxrayvision as xrv
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'ChestXDetDataset needs the [models] extra: uv pip install -e ".[models]"'
            ) from exc
        self._xrv = xrv
        self._transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(self.resize)]
        )

    def __len__(self) -> int:
        return len(self._records)

    def _load_gray(self, file_name: str) -> np.ndarray:
        from skimage.io import imread

        img = np.asarray(imread(self.image_dir / file_name))
        if img.ndim == 3:  # some NIH PNGs are stored 3-channel
            img = img[..., :3].mean(axis=2)
        if img.ndim != 2:
            raise ValueError(f"{file_name}: expected 2D grayscale, got shape {img.shape}")
        return img.astype(np.float32)

    def _preprocess_image(self, gray: np.ndarray) -> Image:
        """Grayscale -> model-ready ``(1, resize, resize)`` float32 (xrv geometry)."""
        maxval = 255.0 if float(gray.max()) <= 255.0 else float(gray.max())
        img = self._xrv.datasets.normalize(gray, maxval)
        img = img[None, ...]  # (1, H, W)
        out: Image = np.asarray(self._transform(img), dtype=np.float32)
        return out

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        """Push a native-resolution bool mask through the same crop+resize."""
        m = mask.astype(np.float32)[None, ...]  # (1, H, W)
        resized = np.asarray(self._transform(m), dtype=np.float32)[0]  # (resize, resize)
        binary: np.ndarray = resized >= 0.5
        return binary

    def _rasterize(self, record: dict[str, Any], category: str, h: int, w: int) -> np.ndarray:
        """Union of a category's instances (polygon if present, else box)."""
        from skimage.draw import polygon as sk_polygon

        mask = np.zeros((h, w), dtype=bool)
        syms = record["syms"]
        polys = record.get("polygons") or [None] * len(syms)
        boxes = record.get("boxes") or [None] * len(syms)
        for j, sym in enumerate(syms):
            if sym != category:
                continue
            poly = polys[j]
            if poly:
                pts = np.asarray(poly, dtype=float)
                # [x, y] points -> (row=y, col=x); skimage.draw is untyped.
                rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(h, w))  # type: ignore[no-untyped-call]
                mask[rr, cc] = True
            elif boxes[j]:
                x1, y1, x2, y2 = (round(v) for v in boxes[j])
                mask[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)] = True
        return mask

    def __getitem__(self, index: int) -> LocalizationExample:
        self._ensure_backends()
        record = self._records[index]
        file_name = str(record["file_name"])
        gray = self._load_gray(file_name)
        h, w = gray.shape
        syms = set(record["syms"])

        n = self.schema.n
        labels: list[int] = [-1] * n
        masks: list[np.ndarray | None] = [None] * n
        for idx, category in self._covered.items():
            present = category in syms
            labels[idx] = 1 if present else 0
            if present:
                native = self._rasterize(record, category, h, w)
                masks[idx] = self._transform_mask(native)

        return LocalizationExample(
            image_id=file_name,
            image=self._preprocess_image(gray),
            labels=tuple(labels),
            masks=tuple(masks),
        )
