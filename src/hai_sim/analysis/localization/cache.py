"""Reusable offline cache of raw model output for localization + label validation.

The cache stores the model's **richest reusable artifact** — the per-member spatial
feature maps (the last conv block's output) — plus the linear head weights, the
per-member probabilities, timing, and the expert references (labels + masks). From
this, *everything* is derived offline without touching the model: any label's
Grad-CAM (the head is linear, so the CAM is analytic — see `cam.cam_from_features`),
logits, probabilities, embeddings, calibration. Storing feature maps also makes the
build **forward-only** (no per-label backprop).

Storage is split so the big array never gets rewritten:

- ``<stem>.features.npy`` : the ``(N, M, C, h, w)`` float16 feature maps, written
  **incrementally into a preallocated memory-mapped array** during the build (no
  in-RAM accumulation, no re-stacking, no per-checkpoint rewrite) and read back
  ``mmap_mode='r'`` (so an evaluator only pages in the rows it touches);
- ``<stem>.npz``  : the small arrays (probs, gt, timing, head weights, masks, ids);
- ``<stem>.json`` : a human-readable manifest.

Checkpoints flush the memmap and rewrite only the small ``.npz`` (~tens of MB), so
a 3578-image build writes the 1.4 GB feature file once, not dozens of times.
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


def _features_path(stem: Path) -> Path:
    return stem.with_name(stem.name + ".features.npy")


@dataclass
class LocalizationCache:
    """A forward-only snapshot of ensemble feature maps + probs over an image set."""

    feature_maps: np.ndarray
    """``(N, M, C, h, w)`` float16 (may be a ``np.memmap``). Per-member last-conv
    feature maps from the real forward — the raw material for any label's Grad-CAM."""

    member_probs: np.ndarray
    """``(N, M, L)`` float32. Per-member ``P(label=1)``; ``NaN`` for uncovered."""

    ground_truth: np.ndarray
    """``(N, L)`` int8 in ``{1, 0, -1}`` (expert image-level labels)."""

    inference_seconds: np.ndarray
    """``(N,)`` float64. Per-image forward time."""

    image_ids: tuple[str, ...]
    split: tuple[str, ...]
    dataset_indices: np.ndarray

    classifier_weights: np.ndarray
    """``(M, L, C)`` float32. Per-member linear-head weights → offline Grad-CAM/logits."""

    member_covers: np.ndarray
    """``(M, L)`` bool. Whether member ``m`` was trained on pathology ``l``."""

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
        # Metadata count, not feature_maps.shape[0]: the memmap may be preallocated
        # to the full planned size while only the processed rows are valid.
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

    def _manifest(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_members": self.n_members,
            "n_labels": self.n_labels,
            "n_instances": self.n_instances,
            "weights": list(self.weights),
            "pathologies": list(self.pathologies),
            "dataset": self.dataset,
            "mask_resolution": self.mask_resolution,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_shape": list(self.feature_maps.shape[1:]),
            "split_counts": {
                s: int(np.sum(np.asarray(self.split) == s)) for s in sorted(set(self.split))
            },
            "inference_seconds_per_image": {
                "mean": float(np.mean(self.inference_seconds)) if self.n_images else 0.0,
                "total": float(np.sum(self.inference_seconds)),
            },
        }

    def save_metadata(self, stem: str | Path) -> None:
        """Persist the small arrays + manifest (not the feature maps) atomically."""
        stem = Path(stem)
        npz_path = stem.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp_npz, "wb") as fh:
            np.savez(
                fh,
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
        tmp_json = stem.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(self._manifest(), indent=2), encoding="utf-8")
        os.replace(tmp_json, stem.with_suffix(".json"))

    def save(self, stem: str | Path) -> None:
        """Persist feature maps (``.features.npy``) + metadata + manifest atomically.

        If ``feature_maps`` is already the memmap backing ``<stem>.features.npy`` it
        is just flushed; otherwise (an in-memory cache) it is written out.
        """
        stem = Path(stem)
        feat = _features_path(stem)
        fm = self.feature_maps
        backing = getattr(fm, "filename", None)
        same_file = backing is not None and Path(backing).resolve() == feat.resolve()
        if isinstance(fm, np.memmap) and same_file:
            fm.flush()  # already the file we would write; just flush (and it's open)
        else:
            feat.parent.mkdir(parents=True, exist_ok=True)
            tmp = feat.with_name(feat.name + ".tmp")
            with open(tmp, "wb") as fh:  # file handle → np.save won't append ".npy"
                np.save(fh, np.asarray(fm))
            os.replace(tmp, feat)
        self.save_metadata(stem)

    @classmethod
    def load(cls, stem: str | Path, mmap: bool = True) -> LocalizationCache:
        stem = Path(stem)
        feat = _features_path(stem)
        if not feat.exists():
            raise FileNotFoundError(
                f"no feature file at {feat} — this cache is missing its feature maps "
                "(a pre-feature-map cache?); rebuild it with build_localization_cache"
            )
        feature_maps = np.load(feat, mmap_mode="r" if mmap else None)
        with np.load(stem.with_suffix(".npz")) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files if k not in ("image_ids", "split")}
            image_ids = tuple(str(x) for x in data["image_ids"])
            split = tuple(str(x) for x in data["split"])
        manifest = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            feature_maps=feature_maps,
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
    out_stem: str | Path,
    total_images: int,
    mask_resolution: int = 112,
    dataset_name: str = "chestxdet",
    checkpoint_every: int = 0,
    prior: LocalizationCache | None = None,
    dataset_fingerprint: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> LocalizationCache:
    """Forward-only pass writing feature maps straight into a preallocated memmap.

    ``model`` must expose ``member_feature_maps_and_probs``, ``member_classifier_weights``,
    ``member_masks`` and ``weight_list``. ``total_images`` is the full planned image
    count (this call's images plus any already in ``prior``); the ``.features.npy``
    memmap is sized to it once and rows are written in place — no backprop, no
    per-checkpoint rewrite of the big array.
    """
    stem = Path(out_stem)
    feat = _features_path(stem)
    schema = dataset.schema
    n_labels = schema.n
    n_members = len(model.weight_list)
    if len(split_tags) != len(indices):
        raise ValueError("split_tags must align 1:1 with indices")
    base = prior.n_images if prior is not None else 0

    classifier_weights = np.asarray(model.member_classifier_weights(), dtype=np.float32)
    member_covers = np.stack([np.asarray(m, dtype=bool) for m in model.member_masks])

    feat_mmap: Any = None  # np.memmap (has .flush); Any keeps mypy off the subclass
    if feat.exists():
        feat_mmap = np.load(feat, mmap_mode="r+")
        if int(feat_mmap.shape[0]) != total_images:
            raise ValueError(
                f"feature cache at {feat} is sized for {feat_mmap.shape[0]} images but this "
                f"run plans {total_images}; resume with the same scope or use a fresh --out"
            )

    img_probs: list[np.ndarray] = []
    img_gt: list[np.ndarray] = []
    img_secs: list[float] = []
    img_ids: list[str] = []
    img_split: list[str] = []
    img_dsidx: list[int] = []
    inst_image: list[int] = []
    inst_label: list[int] = []
    inst_mask: list[np.ndarray] = []

    def _make_cache() -> LocalizationCache:
        assert feat_mmap is not None
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
            feature_maps=feat_mmap,
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
    for local, (idx, split) in enumerate(zip(indices, split_tags, strict=True)):
        ex = dataset[int(idx)]
        t0 = perf_counter()
        probs, maps = model.member_feature_maps_and_probs(ex.image)  # (M,L), (M,C,h,w)
        img_secs.append(perf_counter() - t0)
        if feat_mmap is None:
            feat.parent.mkdir(parents=True, exist_ok=True)
            feat_mmap = np.lib.format.open_memmap(
                feat, mode="w+", dtype=np.float16, shape=(total_images, *maps.shape)
            )
        feat_mmap[base + local] = np.asarray(maps, dtype=np.float16)
        img_probs.append(np.asarray(probs, dtype=np.float32))
        img_gt.append(np.asarray(ex.labels, dtype=np.int8))
        img_ids.append(ex.image_id)
        img_split.append(split)
        img_dsidx.append(int(idx))
        for label_index, mask in enumerate(ex.masks):
            if mask is None:
                continue
            inst_image.append(base + local)
            inst_label.append(label_index)
            inst_mask.append(_downsample_mask(mask, mask_resolution))
        processed += 1
        if progress is not None:
            progress(base + processed, total_images)
        if checkpoint_every and processed - last_ckpt >= checkpoint_every:
            feat_mmap.flush()
            _make_cache().save_metadata(stem)
            last_ckpt = processed

    if feat_mmap is not None:
        feat_mmap.flush()
    return _make_cache()
