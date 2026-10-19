"""Reusable offline cache of raw ensemble inference output.

The expensive part of the uncertainty/error analysis is the forward passes of the
cross-domain Deep Ensemble (one DenseNet121 per weight set, per image). We run
them *once* over a fixed sample of the dataset and persist the **raw** per-member
probability matrix together with the ground truth and per-image inference timing.

Everything downstream is derived from this cache without ever touching the model
again: the UQ measure (predictive / mean entropy, mutual information, std), the
decision threshold, per-label vs per-image aggregation, and the error definition
(any / FN / FP). To stay bit-for-bit consistent with the simulator, the cached
``member_probs`` are exactly what ``TorchXRayVisionAI.member_probabilities``
returns, so feeding them back to ``EnsembleUQ.estimate`` reproduces the simulator's
uncertainty vector exactly.

Storage layout — one cache is two sibling files sharing a stem:

- ``<stem>.npz``  : the arrays (member_probs, ground_truth, inference_seconds, ids).
- ``<stem>.json`` : a human-readable manifest (weights, pathologies, dataset, timing).

Writes are atomic (temp file + ``os.replace``) so an interrupted checkpoint never
leaves a half-written cache behind — important because the build loop checkpoints
periodically while it runs.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from hai_sim.core.types import DatasetExample

if TYPE_CHECKING:
    from hai_sim.core.types import LabelSchema


@dataclass
class InferenceCache:
    """A frozen-in-time snapshot of ensemble inference over a sample of images.

    Shapes use ``N`` = images, ``M`` = ensemble members, ``L`` = labels.
    """

    member_probs: np.ndarray
    """``(N, M, L)`` float32. ``[i, m, k]`` = member ``m``'s P(label_k=1 | image_i),
    or ``NaN`` where member ``m`` was not trained on pathology ``k``."""

    ground_truth: np.ndarray
    """``(N, L)`` int8 in ``{1, 0, -1}`` (-1 = not annotated by this dataset)."""

    inference_seconds: np.ndarray
    """``(N,)`` float64. Wall-clock model-inference time per image. With
    ``batch_size == 1`` this is the true single-image latency (preprocessing of an
    already-resized image is a passthrough, so it is essentially pure forward time);
    with batching it is the per-image amortized time (batch wall / batch size)."""

    image_ids: tuple[str, ...]
    """``(N,)`` dataset-native identifiers (e.g. the NIH filename), best-effort."""

    dataset_indices: np.ndarray
    """``(N,)`` int. Index into the underlying dataset for each cached row."""

    weights: tuple[str, ...]
    """``(M,)`` ensemble member weight-set names, in member (axis-1) order."""

    pathologies: tuple[str, ...]
    """``(L,)`` label names, in column (axis-2) order (== ``LabelSchema.xrv()``)."""

    dataset: str
    """Dataset name the images were sampled from (``nih`` / ``rsna`` / ``chexpert``)."""

    batch_size: int
    """Batch size used during inference; 1 ⇒ ``inference_seconds`` is true latency."""

    dataset_fingerprint: str = ""
    """Identity of what the model saw: dataset | views | unique_patients | resize.
    Guards resume: a cache built over a differently-filtered/-preprocessed dataset must
    not be silently extended. The image *count* is deliberately absent — a cell is an
    accumulating pool keyed by image id (`hai_sim.analysis.cache_matrix`), so a study
    completes only the ids it lacks rather than being pinned to one fixed-N sample."""

    model_fingerprint: str = ""
    """Exact identity of the model config that produced the probabilities (weights for
    xrv; model_id + prompt_set + readout for MedGemma). The decision threshold, UQ
    measure and calibration are applied downstream, so they are NOT part of it."""

    @property
    def n_images(self) -> int:
        return int(self.member_probs.shape[0])

    @property
    def n_members(self) -> int:
        return int(self.member_probs.shape[1])

    @property
    def n_labels(self) -> int:
        return int(self.member_probs.shape[2])

    def save(self, stem: str | Path) -> None:
        """Persist to ``<stem>.npz`` + ``<stem>.json`` atomically."""
        stem = Path(stem)
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
        # Pass a file handle so np.savez does NOT append its own ".npz" suffix.
        with open(tmp_npz, "wb") as fh:
            np.savez(
                fh,
                member_probs=self.member_probs,
                ground_truth=self.ground_truth,
                inference_seconds=self.inference_seconds,
                dataset_indices=self.dataset_indices,
                image_ids=np.asarray(self.image_ids),  # unicode array, no pickle needed
            )
        os.replace(tmp_npz, npz_path)

        manifest: dict[str, Any] = {
            "n_images": self.n_images,
            "n_members": self.n_members,
            "n_labels": self.n_labels,
            "weights": list(self.weights),
            "pathologies": list(self.pathologies),
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "batch_size": self.batch_size,
            "inference_seconds_per_image": {
                "mean": float(np.mean(self.inference_seconds)) if self.n_images else 0.0,
                "median": float(np.median(self.inference_seconds)) if self.n_images else 0.0,
                "min": float(np.min(self.inference_seconds)) if self.n_images else 0.0,
                "max": float(np.max(self.inference_seconds)) if self.n_images else 0.0,
                "total": float(np.sum(self.inference_seconds)),
            },
        }
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        tmp_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_json, json_path)

    @classmethod
    def load(cls, stem: str | Path) -> InferenceCache:
        """Load a cache previously written by :meth:`save`."""
        stem = Path(stem)
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")
        with np.load(npz_path) as data:
            member_probs = np.asarray(data["member_probs"])
            ground_truth = np.asarray(data["ground_truth"])
            inference_seconds = np.asarray(data["inference_seconds"])
            dataset_indices = np.asarray(data["dataset_indices"])
            image_ids = tuple(str(x) for x in data["image_ids"])
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        return cls(
            member_probs=member_probs,
            ground_truth=ground_truth,
            inference_seconds=inference_seconds,
            image_ids=image_ids,
            dataset_indices=dataset_indices,
            weights=tuple(manifest["weights"]),
            pathologies=tuple(manifest["pathologies"]),
            dataset=str(manifest["dataset"]),
            batch_size=int(manifest["batch_size"]),
            dataset_fingerprint=str(manifest.get("dataset_fingerprint", "")),
            model_fingerprint=str(manifest.get("model_fingerprint", "")),
        )


def _ground_truth_row(lab: np.ndarray) -> np.ndarray:
    """Map a dataset label vector to the ``{1, 0, -1}`` convention.

    Handles both conventions the builder is fed: a raw TorchXRayVision vector
    (``0`` / ``1`` / ``NaN`` = not annotated) and a `DatasetExample` vector already in
    ``{1, 0, -1}`` (``-1`` = not annotated). Both ``NaN`` and an existing ``-1`` map to
    ``-1``, so a "not annotated" label is never silently turned into an annotated
    negative — labels the model scores but the dataset does not annotate must stay
    unknown, not zero.
    """
    lab = np.asarray(lab, dtype=float).ravel()
    out = np.full(lab.shape, -1, dtype=np.int8)
    annotated = ~np.isnan(lab) & (lab != -1)
    out[annotated] = (lab[annotated] >= 0.5).astype(np.int8)
    return out


def _image_id(dataset: Any, idx: int) -> str:
    """Best-effort dataset-native id (filename / path), falling back to the index."""
    csv = getattr(dataset, "csv", None)
    if csv is not None:
        for col in ("Image Index", "Path", "path", "ImageID", "image_id"):
            if col in getattr(csv, "columns", ()):
                return str(csv.iloc[idx][col])
    return str(idx)


def _extract_item(item: Any, dataset: Any, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Normalize one dataset item to (model-ready image, label vector, image id).

    Supports the two dataset shapes a cache is built over: a raw TorchXRayVision
    dataset (a ``dict`` with ``img`` / ``lab``, id read from its csv), and any framework
    `Dataset` yielding a `DatasetExample` (image / ground_truth / example_id) — which is
    what lets one builder cache both an xrv dataset and an `ImageFolderDataset` (VinDr).
    """
    if isinstance(item, DatasetExample):
        return (
            np.asarray(item.image, dtype=np.float32),
            np.asarray(item.ground_truth, dtype=float),
            item.example_id,
        )
    return (
        np.asarray(item["img"], dtype=np.float32),
        np.asarray(item["lab"], dtype=float),
        _image_id(dataset, idx),
    )


@runtime_checkable
class CacheableModel(Protocol):
    """What building a cache needs of a model — deliberately less than a full actor.

    Any model that can score an image several times over (a Deep Ensemble's weight sets,
    a generative model's prompt paraphrases) satisfies this, which is what lets one cache
    format and one builder serve backends of completely different natures.
    """

    schema: LabelSchema

    @property
    def member_names(self) -> tuple[str, ...]:
        """Identifier per member, in the order `member_probabilities_batch` returns them."""

    def member_probabilities(self, image: Any) -> np.ndarray:
        """``(n_members, n_labels)`` probabilities for one image."""

    def member_probabilities_batch(self, images: Sequence[Any]) -> np.ndarray:
        """``(len(images), n_members, n_labels)``.

        A backend with no real batching may loop; the builder still calls this so the
        timing it records covers whatever the backend actually does per image.
        """


def build_inference_cache(
    model: CacheableModel,
    dataset: Any,
    *,
    indices: Sequence[int],
    dataset_name: str,
    batch_size: int = 1,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[InferenceCache], None] | None = None,
    prior: InferenceCache | None = None,
    dataset_fingerprint: str = "",
    model_fingerprint: str = "",
) -> InferenceCache:
    """Run the ensemble over ``indices`` and collect raw probabilities + timing.

    ``indices`` are the images still to process — the caller is responsible for
    excluding any already in ``prior`` (the resume cache). Every snapshot handed
    to ``on_checkpoint`` (and the final return) is the concatenation
    ``prior + freshly-processed``, so each atomic save is a complete, resumable
    cache: a sudden power-off loses at most the work since the last checkpoint.

    The model forward passes (and *only* those) are timed per image: dataset
    loading / preprocessing is excluded so ``inference_seconds`` reflects model
    inference cost, usable to calibrate the simulator's ``ai.service_time``.
    """
    n_total = len(indices)
    n_members = len(model.member_names)
    n_labels = model.schema.n

    member_probs = np.full((n_total, n_members, n_labels), np.nan, dtype=np.float32)
    ground_truth = np.full((n_total, n_labels), -1, dtype=np.int8)
    inference_seconds = np.zeros(n_total, dtype=np.float64)
    image_ids: list[str] = []
    dataset_indices = np.asarray(list(indices), dtype=np.int64)

    def _snapshot(k: int) -> InferenceCache:
        new_ids = tuple(image_ids[:k])
        if prior is None:
            mp = member_probs[:k].copy()
            gt = ground_truth[:k].copy()
            inf = inference_seconds[:k].copy()
            ids = new_ids
            di = dataset_indices[:k].copy()
        else:
            mp = np.concatenate([prior.member_probs, member_probs[:k]], axis=0)
            gt = np.concatenate([prior.ground_truth, ground_truth[:k]], axis=0)
            inf = np.concatenate([prior.inference_seconds, inference_seconds[:k]], axis=0)
            ids = tuple(prior.image_ids) + new_ids
            di = np.concatenate([prior.dataset_indices, dataset_indices[:k]], axis=0)
        return InferenceCache(
            member_probs=mp,
            ground_truth=gt,
            inference_seconds=inf,
            image_ids=ids,
            dataset_indices=di,
            weights=tuple(model.member_names),
            pathologies=tuple(model.schema.labels),
            dataset=dataset_name,
            batch_size=batch_size,
            dataset_fingerprint=dataset_fingerprint,
            model_fingerprint=model_fingerprint,
        )

    processed = 0
    last_ckpt = 0
    for start in range(0, n_total, batch_size):
        batch_idx = list(indices[start : start + batch_size])
        extracted = [_extract_item(dataset[i], dataset, i) for i in batch_idx]
        imgs = [img for img, _, _ in extracted]

        t0 = perf_counter()
        probs = model.member_probabilities_batch(imgs)  # (b, M, L)
        per_img = (perf_counter() - t0) / max(len(imgs), 1)

        b = len(batch_idx)
        member_probs[start : start + b] = probs.astype(np.float32)
        for j, (_, lab, img_id) in enumerate(extracted):
            ground_truth[start + j] = _ground_truth_row(lab)
            inference_seconds[start + j] = per_img
            image_ids.append(img_id)
        processed += b

        if checkpoint_every and on_checkpoint and processed - last_ckpt >= checkpoint_every:
            on_checkpoint(_snapshot(processed))
            last_ckpt = processed

    return _snapshot(processed)


def measures_from_cache(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized UQ decomposition over a ``(N, M, L)`` member-probability stack.

    Returns the per-image ``(N, L)`` arrays mirroring ``EnsembleUQ``:
    ``mean_p`` (the ensemble mean probability), ``predictive_entropy`` (total,
    == single-pass ``EntropyUQ``), ``mean_entropy`` (aleatoric), their difference
    ``mutual_information`` (epistemic), and member ``std``. NaN member slots (an
    uncovered pathology) are ignored via the nan-aware reductions.
    """
    from hai_sim.uq.methods import binary_entropy

    probs = np.asarray(member_probs, dtype=float)
    with warnings.catch_warnings():
        # All-NaN columns (a pathology no member covers) yield NaN here; expected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_p = np.nanmean(probs, axis=1)
        mean_entropy = np.nanmean(binary_entropy(probs), axis=1)
        std = np.nanstd(probs, axis=1)
    predictive_entropy = binary_entropy(mean_p)
    mutual_information = np.clip(predictive_entropy - mean_entropy, 0.0, None)
    # Confidence-based: distance of the mean prob from a confident call. High near
    # the 0.5 boundary, ~0 near 0/1. Per label it is rank-equivalent to
    # predictive_entropy (binary entropy is monotone in min(p, 1-p)), so it gives
    # the same per-pathology AUROC; it differs once aggregated across labels.
    least_confidence = np.minimum(mean_p, 1.0 - mean_p)
    return {
        "mean_p": mean_p,
        "predictive_entropy": predictive_entropy,
        "mean_entropy": mean_entropy,
        "mutual_information": mutual_information,
        "std": std,
        "least_confidence": least_confidence,
    }
