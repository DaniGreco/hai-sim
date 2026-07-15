"""Capability-aware `Dataset` adapter over a TorchXRayVision dataset object.

Wraps any TorchXRayVision-style dataset (exposes ``pathologies`` and
``[i] -> {"img": ndarray, "lab": ndarray}``) in the framework's `Dataset`
contract. No torch is imported here: it only adapts an already-built object, so
the core package stays importable without the heavy ``[models]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hai_sim.core.types import DatasetCapabilities, DatasetExample, Image, LabelSchema


def to_sentinel(v: float) -> int:
    """Map a dataset label cell (1.0 / 0.0 / NaN) to the ``{1, 0, -1}`` convention."""
    if v != v:  # NaN → not annotated
        return -1
    return 1 if float(v) >= 0.5 else 0


class XRVDataset:
    """A `Dataset` backed by a TorchXRayVision dataset object.

    Native ground truth only for now; adjudicated / per-reader sources are added
    to ``capabilities.gt_sources`` when that plumbing lands with the real-data work.
    """

    def __init__(self, dataset: Any, schema: LabelSchema, name: str = "xrv") -> None:
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        if tuple(dataset.pathologies) != schema.labels:
            raise ValueError(
                "dataset.pathologies do not match the active LabelSchema; "
                "align the schema to the dataset (use label_schema: xrv)."
            )
        self._ds = dataset
        self.schema = schema
        self.name = name
        self.capabilities = DatasetCapabilities(gt_sources=("native",))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> DatasetExample:
        item = self._ds[index]
        image: Image = np.asarray(item["img"], dtype=np.float32)
        gt = tuple(to_sentinel(v) for v in np.asarray(item["lab"]).ravel())
        if len(gt) != self.schema.n:
            raise ValueError(f"dataset label length {len(gt)} != schema.n {self.schema.n}")
        return DatasetExample(example_id=str(index), image=image, ground_truth=gt)
