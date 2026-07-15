"""Synthetic capability-aware `Dataset`: random multi-label GT at a set prevalence.

A dependency-free `Dataset` for wiring tests and the dataset-agnostic patient
stream without real images / torch. Native ground truth only.
"""

from __future__ import annotations

import numpy as np

from hai_sim.core.types import DatasetCapabilities, DatasetExample, Image, LabelSchema


class MockDataset:
    """A synthetic dataset with native GT only (no masks / per-reader reads).

    Ground truth is drawn once at construction so ``dataset[i]`` is stable
    regardless of access order — an indexable pool, not a stream.
    """

    def __init__(
        self,
        schema: LabelSchema,
        n_examples: int,
        prevalence: float | tuple[float, ...] = 0.2,
        rng: np.random.Generator | None = None,
        image_shape: tuple[int, ...] = (1, 1),
        name: str = "MockDataset",
    ) -> None:
        if n_examples <= 0:
            raise ValueError(f"n_examples must be > 0, got {n_examples}")
        r = rng if rng is not None else np.random.default_rng()
        if isinstance(prevalence, (int, float)):
            prev = tuple([float(prevalence)] * schema.n)
        else:
            if len(prevalence) != schema.n:
                raise ValueError(f"prevalence length {len(prevalence)} != schema.n {schema.n}")
            prev = tuple(float(p) for p in prevalence)
        self.schema = schema
        self.name = name
        self._image_shape = image_shape
        self._gt: list[tuple[int, ...]] = [
            tuple(int(r.random() < p) for p in prev) for _ in range(n_examples)
        ]
        self.capabilities = DatasetCapabilities(gt_sources=("native",))

    def __len__(self) -> int:
        return len(self._gt)

    def __getitem__(self, index: int) -> DatasetExample:
        image: Image = np.zeros(self._image_shape, dtype=np.uint8)
        return DatasetExample(example_id=str(index), image=image, ground_truth=self._gt[index])
