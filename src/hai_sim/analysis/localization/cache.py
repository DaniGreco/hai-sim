"""Reusable offline cache of raw model output for localization + label validation.

The cache stores the model's **richest reusable artifact** — the per-member spatial
feature maps (the last conv block's output) — plus the linear head weights, the
per-member probabilities, timing, and the expert references (labels + masks). From
this, *everything* is derived offline without touching the model: any label's
Grad-CAM (the head is linear, so the CAM is analytic — see `cam_from_features`),
logits, probabilities, embeddings, calibration. Storing feature maps also makes the
build **forward-only** (no per-label backprop).

What is stored (``N`` images, ``M`` members, ``C`` channels, ``h``x``w`` = 7x7
spatial, ``L`` labels, ``K`` localized instances):

- image level: ``feature_maps (N, M, C, h, w)`` float16 (the raw artifact),
  ``member_probs (N, M, L)``, ``ground_truth (N, L)``, ``inference_seconds (N,)``,
  ``image_ids``, ``split`` (train/test), ``dataset_indices``;
- head / metadata: ``classifier_weights (M, L, C)``, ``member_covers (M, L)``;
- expert references: ``inst_image (K,)``, ``inst_label (K,)``, ``inst_mask
  (K, R, R)`` at ``mask_resolution`` R.

Writes are atomic (temp + ``os.replace``). Feature maps dominate size (~1.4 GB for
3578 images x 4 members at fp16), so the arrays are stored **uncompressed** for fast
checkpoints; use a larger ``checkpoint_every`` for the full build.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hai_sim.core.protocols import LocalizationDataset


@dataclass
class LocalizationCache:
    """A frozen-in-time snapshot of forward-only ensemble output over an image set."""

    # --- image level -------------------------------------------------------
    feature_maps: np.ndarray
    """``(N, M, C, h, w)`` float16. Per-member last-conv feature maps from the real
    forward (fix_resolution applied) — consistent with ``member_probs``."""

    member_probs: np.ndarray
    """``(N, M, L)`` float32. Per-member ``P(label=1)``; ``NaN`` for uncovered."""

    ground_truth: np.ndarray
    """``(N, L)`` int8 in ``{1, 0, -1}`` (expert image-level labels)."""

    inference_seconds: np.ndarray
    """``(N,)`` float64. Per-image forward time (single image), for service_time."""

    image_ids: tuple[str, ...]
    split: tuple[str, ...]
    dataset_indices: np.ndarray

    # --- head / metadata for offline derivation ----------------------------
    classifier_weights: np.ndarray
    """``(M, L, C)`` float32. Per-member linear-head weights → offline Grad-CAM/logits."""

    member_covers: np.ndarray
    """``(M, L)`` bool. Whether member ``m`` was trained on pathology ``l``."""

    # --- expert references (localized findings) ----------------------------
    inst_image: np.ndarray
    inst_label: np.ndarray
    inst_mask: np.ndarray
    """``(K, R, R)`` bool. Expert mask co-registered to model geometry at R."""

    weights: tuple[str, ...]
    pathologies: tuple[str, ...]
    dataset: str
    mask_resolution: int
    dataset_fingerprint: str = ""

    @property
    def n_images(self) -> int:
        return int(self.feature_maps.shape[0])

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
        """Persist to ``<stem>.npz`` + ``<stem>.json`` atomically (uncompressed)."""
        stem = Path(stem)
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp_npz, "wb") as fh:
            np.savez(
                fh,
                feature_maps=self.feature_maps,
                member_probs=self.member_probs,
                ground_truth=self.ground_truth,
                inference_seconds=self.inference_seconds,
                dataset_indices=self.dataset_indices,
                image_ids=np.asarray(self.image_ids),
                split=np.asarray(self.split),
                classifier_weights=self.classifier_weights,
                member_covers=self.member_covers,
                inst_image=self.inst_image,
                inst_label=self.inst_label,
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
            "feature_shape": list(self.feature_maps.shape[1:]) if self.n_images else [],
            "split_counts": {
                s: int(np.sum(np.asarray(self.split) == s)) for s in sorted(set(self.split))
            },
            "inference_seconds_per_image": {
                "mean": float(np.mean(self.inference_seconds)) if self.n_images else 0.0,
                "total": float(np.sum(self.inference_seconds)),
            },
        }
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        tmp_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_json, json_path)

    @classmethod
    def load(cls, stem: str | Path) -> LocalizationCache:
        stem = Path(stem)
        with np.load(stem.with_suffix(".npz")) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files if k not in ("image_ids", "split")}
            image_ids = tuple(str(x) for x in data["image_ids"])
            split = tuple(str(x) for x in data["split"])
        manifest = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            feature_maps=arrays["feature_maps"],
            member_probs=arrays["member_probs"],
            ground_truth=arrays["ground_truth"],
            inference_seconds=arrays["inference_seconds"],
            image_ids=image_ids,
            split=split,
            dataset_indices=arrays["dataset_indices"],
            classifier_weights=arrays["classifier_weights"],
            member_covers=arrays["member_covers"],
            inst_image=arrays["inst_image"],
            inst_label=arrays["inst_label"],
            inst_mask=arrays["inst_mask"],
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
    model: Any,
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
    """Forward-only pass: capture per-member feature maps + probs + timing + masks.

    ``model`` must expose ``member_feature_maps_and_probs``, ``member_classifier_weights``,
    ``member_masks`` and ``weight_list`` (a `TorchXRayVisionAI`). No backprop happens:
    Grad-CAMs are reconstructed offline from the stored feature maps + head weights.
    """
    schema = dataset.schema
    n_labels = schema.n
    n_members = len(model.weight_list)
    if len(split_tags) != len(indices):
        raise ValueError("split_tags must align 1:1 with indices")
    base_images = prior.n_images if prior is not None else 0

    classifier_weights = np.asarray(model.member_classifier_weights(), dtype=np.float32)
    member_covers = np.stack([np.asarray(m, dtype=bool) for m in model.member_masks])

    img_maps: list[np.ndarray] = []
    img_probs: list[np.ndarray] = []
    img_gt: list[np.ndarray] = []
    img_secs: list[float] = []
    img_ids: list[str] = []
    img_split: list[str] = []
    img_dsidx: list[int] = []
    inst_image: list[int] = []
    inst_label: list[int] = []
    inst_mask: list[np.ndarray] = []

    def _snapshot() -> LocalizationCache:
        feat_shape = img_maps[0].shape if img_maps else (n_members, 1024, 7, 7)
        fm = (
            np.stack(img_maps).astype(np.float16)
            if img_maps
            else np.empty((0, *feat_shape), np.float16)
        )
        mp = (
            np.stack(img_probs).astype(np.float32)
            if img_probs
            else np.empty((0, n_members, n_labels), np.float32)
        )
        gt = np.stack(img_gt).astype(np.int8) if img_gt else np.empty((0, n_labels), np.int8)
        secs = np.asarray(img_secs, dtype=np.float64)
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
            fm = np.concatenate([prior.feature_maps, fm], axis=0)
            mp = np.concatenate([prior.member_probs, mp], axis=0)
            gt = np.concatenate([prior.ground_truth, gt], axis=0)
            secs = np.concatenate([prior.inference_seconds, secs], axis=0)
            di = np.concatenate([prior.dataset_indices, di], axis=0)
            ids = tuple(prior.image_ids) + ids
            spl = tuple(prior.split) + spl
            im = np.concatenate([prior.inst_mask, im], axis=0)
            ii = np.concatenate([prior.inst_image, ii], axis=0)
            il = np.concatenate([prior.inst_label, il], axis=0)

        return LocalizationCache(
            feature_maps=fm,
            member_probs=mp,
            ground_truth=gt,
            inference_seconds=secs,
            image_ids=ids,
            split=spl,
            dataset_indices=di,
            classifier_weights=classifier_weights,
            member_covers=member_covers,
            inst_image=ii,
            inst_label=il,
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
        t0 = perf_counter()
        probs, maps = model.member_feature_maps_and_probs(ex.image)  # (M,L), (M,C,h,w)
        img_secs.append(perf_counter() - t0)
        img_maps.append(np.asarray(maps, dtype=np.float16))
        img_probs.append(np.asarray(probs, dtype=np.float32))
        img_gt.append(np.asarray(ex.labels, dtype=np.int8))
        img_ids.append(ex.image_id)
        img_split.append(split)
        img_dsidx.append(int(idx))
        row = base_images + len(img_maps) - 1
        for label_index, mask in enumerate(ex.masks):
            if mask is None:
                continue
            inst_image.append(row)
            inst_label.append(label_index)
            inst_mask.append(_downsample_mask(mask, mask_resolution))
        processed += 1
        if progress is not None:
            progress(base_images + processed, base_images + len(indices))
        if checkpoint_every and on_checkpoint and processed - last_ckpt >= checkpoint_every:
            on_checkpoint(_snapshot())
            last_ckpt = processed

    return _snapshot()
