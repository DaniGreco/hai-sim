"""An image-folder `Dataset`: pixels from a directory, labels from `LabelSource`s.

The generic real-image stream. Where `XRVDataset` wraps a TorchXRayVision dataset
object (which carries its own labels), `ImageFolderDataset` starts from a plain
folder of images and attaches one or more named `LabelSource`s — the shape a dataset
released as "images + annotation CSVs" (VinDr, and most others) actually takes.
``gt_source`` selects which attached source is the run's ground truth; the rest ride
along in ``DatasetExample.label_sources`` so a paired-contrast study can swap the
believed truth.

No torch here: the image is read as a plain array and handed to an injected
``preprocess`` callable (the factory builds the model-appropriate one), keeping the
core package importable without the heavy models extra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hai_sim.core.types import DatasetCapabilities, DatasetExample, Image, LabelSchema

if TYPE_CHECKING:
    from hai_sim.data.sources import LabelSource


def _load_grayscale(path: Path) -> np.ndarray:
    """Read an image file as a 2-D float array in [0, 255], via Pillow."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


class ImageFolderDataset:
    """A `Dataset` over a folder of images plus named `LabelSource`s.

    ``sources`` are attached in order; the first (or ``primary``) is what ``native``
    and ``ground_truth`` resolve to, and every source name is advertised in
    ``capabilities.gt_sources`` so a scenario asking for one that is not attached
    fails fast. All sources must already be aligned to ``schema`` (the model space).
    """

    def __init__(
        self,
        image_dir: str | Path,
        schema: LabelSchema,
        sources: Sequence[LabelSource],
        *,
        preprocess: Callable[[np.ndarray], Image] | None = None,
        primary: str | None = None,
        extension: str = ".png",
        name: str = "image_folder",
    ) -> None:
        self.schema = schema
        self.name = name
        self._preprocess = preprocess
        self._sources = list(sources)
        if not self._sources:
            raise ValueError("ImageFolderDataset needs at least one label source")
        for src in self._sources:
            if src.schema.labels != schema.labels:
                raise ValueError(
                    f"label source {src.name!r} is in schema {src.schema.labels}, not "
                    f"the dataset schema {schema.labels}"
                )
        names = [src.name for src in self._sources]
        if len(set(names)) != len(names):
            raise ValueError(f"label source names must be unique; got {names}")
        self._primary = primary if primary is not None else names[0]
        if self._primary not in names:
            raise ValueError(f"primary source {self._primary!r} is not among {names}")
        self._paths = sorted(Path(image_dir).glob(f"*{extension}"))
        if not self._paths:
            raise ValueError(f"no *{extension} images in {image_dir}")
        self.capabilities = DatasetCapabilities(gt_sources=("native", *names))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> DatasetExample:
        path = self._paths[index]
        example_id = path.stem
        image: Image = _load_grayscale(path)
        if self._preprocess is not None:
            image = self._preprocess(image)
        n = self.schema.n
        label_sources: dict[str, tuple[int, ...]] = {}
        for src in self._sources:
            vec = src.labels_for(example_id)
            label_sources[src.name] = vec if vec is not None else (-1,) * n
        return DatasetExample(
            example_id=example_id,
            image=image,
            ground_truth=label_sources[self._primary],
            label_sources=label_sources,
        )
