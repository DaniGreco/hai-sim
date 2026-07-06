"""Reusable offline cache of raw model output for localization + label validation.

The expensive part of validating a model against ChestX-Det is the forward passes
*and* the per-member Grad-CAM backward passes. We run them **once** over the image
set and persist only the raw model artifacts; every metric, threshold, member
aggregation, pathology slice and error definition is then derived offline without
touching the model — the same "cache raw, derive everything" split the uncertainty
track already uses (`inference_cache.py`), extended to the spatial side.

What is stored (``N`` images, ``M`` members, ``L`` labels, ``K`` localized
instances = ``(image, label-with-mask)`` pairs):

- image level: ``member_probs (N, M, L)``, ``ground_truth (N, L)``, ``image_ids``,
  ``split`` (train/test tag), ``dataset_indices``;
- instance level: ``inst_image (K,)``, ``inst_label (K,)``, the per-member native
  Grad-CAM ``inst_cam (K, M, h7, w7)`` (NaN where a member does not cover the
  label), and the co-registered expert mask ``inst_mask (K, R, R)`` at
  ``mask_resolution``.

Per-member CAMs (not the fused heatmap) are stored on purpose: the member
aggregation stays a free offline knob, exactly like per-member ``member_probs``
keep the UQ aggregation free. Writes are atomic (temp + ``os.replace``) so an
interrupted checkpoint never leaves half a cache behind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hai_sim.analysis.localization.saliency import GradCAMSaliency
    from hai_sim.core.protocols import LocalizationDataset


@dataclass
class LocalizationCache:
    """A frozen-in-time snapshot of ensemble inference + Grad-CAM over an image set."""

    # --- image level -------------------------------------------------------
    member_probs: np.ndarray
    """``(N, M, L)`` float32. Per-member ``P(label=1 | image)``; ``NaN`` where a
    member was not trained on that pathology."""

    ground_truth: np.ndarray
    """``(N, L)`` int8 in ``{1, 0, -1}`` (expert image-level labels; -1 = not
    annotated by this dataset for that pathology)."""

    image_ids: tuple[str, ...]
    """``(N,)`` dataset-native image identifiers (the ChestX-Det file name)."""

    split: tuple[str, ...]
    """``(N,)`` per-image split tag (``"train"`` / ``"test"``) for offline slicing."""

    dataset_indices: np.ndarray
    """``(N,)`` int. Index into the underlying dataset for each cached row."""

    # --- instance level (localized findings) -------------------------------
    inst_image: np.ndarray
    """``(K,)`` int. Row in the image-level arrays this instance belongs to."""

    inst_label: np.ndarray
    """``(K,)`` int. Schema label index of this localized finding."""

    inst_cam: np.ndarray
    """``(K, M, h7, w7)`` float32. Per-member native Grad-CAM, min-max normalized;
    ``NaN`` for members that do not cover ``inst_label``."""

    inst_mask: np.ndarray
    """``(K, R, R)`` bool. Expert mask co-registered to the model geometry, stored
    at ``mask_resolution`` R."""

    # --- metadata ----------------------------------------------------------
    weights: tuple[str, ...]
    pathologies: tuple[str, ...]
    dataset: str
    mask_resolution: int
    dataset_fingerprint: str = ""

    @property
    def n_images(self) -> int:
        return int(self.member_probs.shape[0])

    @property
    def n_members(self) -> int:
        return int(self.member_probs.shape[1])

    @property
    def n_labels(self) -> int:
        return int(self.member_probs.shape[2])

    @property
    def n_instances(self) -> int:
        return int(self.inst_image.shape[0])

    def save(self, stem: str | Path) -> None:
        """Persist to ``<stem>.npz`` + ``<stem>.json`` atomically."""
        stem = Path(stem)
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(
                fh,
                member_probs=self.member_probs,
                ground_truth=self.ground_truth,
                dataset_indices=self.dataset_indices,
                image_ids=np.asarray(self.image_ids),
                split=np.asarray(self.split),
                inst_image=self.inst_image,
                inst_label=self.inst_label,
                inst_cam=self.inst_cam,
                inst_mask=self.inst_mask,
            )
        os.replace(tmp_npz, npz_path)

        manifest: dict[str, Any] = {
            "n_images": self.n_images,
            "n_members": self.n_members,
            "n_labels": self.n_labels,
            "n_instances": self.n_instances,
            "weights": list(self.weights),
            "pathologies": list(self.pathologies),
            "dataset": self.dataset,
            "mask_resolution": self.mask_resolution,
            "dataset_fingerprint": self.dataset_fingerprint,
            "cam_shape": list(self.inst_cam.shape[2:]) if self.n_instances else [7, 7],
            "split_counts": {
                s: int(np.sum(np.asarray(self.split) == s)) for s in sorted(set(self.split))
            },
        }
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        tmp_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_json, json_path)

    @classmethod
    def load(cls, stem: str | Path) -> LocalizationCache:
        stem = Path(stem)
        with np.load(stem.with_suffix(".npz")) as data:
            member_probs = np.asarray(data["member_probs"])
            ground_truth = np.asarray(data["ground_truth"])
            dataset_indices = np.asarray(data["dataset_indices"])
            image_ids = tuple(str(x) for x in data["image_ids"])
            split = tuple(str(x) for x in data["split"])
            inst_image = np.asarray(data["inst_image"])
            inst_label = np.asarray(data["inst_label"])
            inst_cam = np.asarray(data["inst_cam"])
            inst_mask = np.asarray(data["inst_mask"])
        manifest = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            member_probs=member_probs,
            ground_truth=ground_truth,
            image_ids=image_ids,
            split=split,
            dataset_indices=dataset_indices,
            inst_image=inst_image,
            inst_label=inst_label,
            inst_cam=inst_cam,
            inst_mask=inst_mask,
            weights=tuple(manifest["weights"]),
            pathologies=tuple(manifest["pathologies"]),
            dataset=str(manifest["dataset"]),
            mask_resolution=int(manifest["mask_resolution"]),
            dataset_fingerprint=str(manifest.get("dataset_fingerprint", "")),
        )


def _downsample_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a boolean mask to ``(resolution, resolution)`` (bilinear + 0.5 cut)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (resolution, resolution):
        return mask
    from skimage.transform import resize as sk_resize

    resized = sk_resize(  # type: ignore[no-untyped-call]
        mask.astype(np.float32), (resolution, resolution), order=1, anti_aliasing=False
    )
    return np.asarray(resized) >= 0.5


def build_localization_cache(
    saliency: GradCAMSaliency,
    dataset: LocalizationDataset,
    *,
    indices: Sequence[int],
    split_tags: Sequence[str],
    mask_resolution: int = 112,
    dataset_name: str = "chestxdet",
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[LocalizationCache], None] | None = None,
    prior: LocalizationCache | None = None,
    dataset_fingerprint: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> LocalizationCache:
    """Run probs + per-member Grad-CAM over ``indices`` and collect the raw cache.

    ``indices`` are the images still to process (the caller excludes those already
    in ``prior``); ``split_tags`` is the per-index split label. Every snapshot
    handed to ``on_checkpoint`` (and the final return) is the concatenation
    ``prior + freshly-processed``, so each atomic save is a complete, resumable
    cache. Only labels the dataset localizes (a non-``None`` mask) get a Grad-CAM.
    """
    model = saliency.model
    schema = dataset.schema
    n_labels = schema.n
    n_members = len(model.weight_list)
    if len(split_tags) != len(indices):
        raise ValueError("split_tags must align 1:1 with indices")
    base_images = prior.n_images if prior is not None else 0

    img_probs: list[np.ndarray] = []
    img_gt: list[np.ndarray] = []
    img_ids: list[str] = []
    img_split: list[str] = []
    img_dsidx: list[int] = []
    inst_image: list[int] = []
    inst_label: list[int] = []
    inst_cam: list[np.ndarray] = []
    inst_mask: list[np.ndarray] = []

    def _snapshot() -> LocalizationCache:
        mp = (
            np.stack(img_probs).astype(np.float32)
            if img_probs
            else np.empty((0, n_members, n_labels), np.float32)
        )
        gt = (
            np.stack(img_gt).astype(np.int8)
            if img_gt
            else np.empty((0, n_labels), np.int8)
        )
        cam_hw = inst_cam[0].shape[1:] if inst_cam else (7, 7)
        ic = (
            np.stack(inst_cam).astype(np.float32)
            if inst_cam
            else np.empty((0, n_members, *cam_hw), np.float32)
        )
        im = (
            np.stack(inst_mask).astype(bool)
            if inst_mask
            else np.empty((0, mask_resolution, mask_resolution), bool)
        )
        ii = np.asarray(inst_image, dtype=np.int64)
        il = np.asarray(inst_label, dtype=np.int64)
        di = np.asarray(img_dsidx, dtype=np.int64)
        ids = tuple(img_ids)
        spl = tuple(img_split)

        if prior is not None:
            mp = np.concatenate([prior.member_probs, mp], axis=0)
            gt = np.concatenate([prior.ground_truth, gt], axis=0)
            di = np.concatenate([prior.dataset_indices, di], axis=0)
            ids = tuple(prior.image_ids) + ids
            spl = tuple(prior.split) + spl
            ic = np.concatenate([prior.inst_cam, ic], axis=0)
            im = np.concatenate([prior.inst_mask, im], axis=0)
            ii = np.concatenate([prior.inst_image, ii], axis=0)
            il = np.concatenate([prior.inst_label, il], axis=0)

        return LocalizationCache(
            member_probs=mp,
            ground_truth=gt,
            image_ids=ids,
            split=spl,
            dataset_indices=di,
            inst_image=ii,
            inst_label=il,
            inst_cam=ic,
            inst_mask=im,
            weights=tuple(model.weight_list),
            pathologies=tuple(schema.labels),
            dataset=dataset_name,
            mask_resolution=mask_resolution,
            dataset_fingerprint=dataset_fingerprint,
        )

    processed = 0
    last_ckpt = 0
    for idx, split in zip(indices, split_tags, strict=True):
        ex = dataset[int(idx)]
        img_probs.append(np.asarray(model.member_probabilities(ex.image), dtype=np.float32))
        img_gt.append(np.asarray(ex.labels, dtype=np.int8))
        img_ids.append(ex.image_id)
        img_split.append(split)
        img_dsidx.append(int(idx))
        row = base_images + len(img_probs) - 1
        for label_index, mask in enumerate(ex.masks):
            if mask is None:
                continue
            inst_image.append(row)
            inst_label.append(label_index)
            inst_cam.append(np.asarray(saliency.member_cams(ex.image, label_index), np.float32))
            inst_mask.append(_downsample_mask(mask, mask_resolution))
        processed += 1
        if progress is not None:
            progress(base_images + processed, base_images + len(indices))
        if checkpoint_every and on_checkpoint and processed - last_ckpt >= checkpoint_every:
            on_checkpoint(_snapshot())
            last_ckpt = processed

    return _snapshot()
