"""Named label sources — the general "one image, several label sets" seam.

The framework already lets a scenario pick *which* ground truth to believe via
``patient_generator.gt_source``; a `LabelSource` is where those alternatives come
from when they live in CSV files rather than baked into a dataset object. Each
source reads a table keyed by ``example_id`` in some *source* label space and maps
it, through a `LabelMap`, into the model schema — so a VinDr consensus CSV, a NIH
adjudicated CSV and a model's native labels are the same kind of thing, selected the
same way.

Why a seam (the framework's bar — three implementations you would run **and compare
in one study**): ``csv`` (a mapped per-image table), ``reader_panel`` (a
per-radiologist table reduced to a consensus, added with the persona work) and a
dataset's own ``native`` labels. A ground-truth-robustness study is exactly a paired
contrast that swaps this one knob and re-runs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hai_sim.core.types import LabelSchema
    from hai_sim.labels import LabelMap


@runtime_checkable
class LabelSource(Protocol):
    """A named provider of per-image labels, aligned to the model schema.

    ``labels_for`` returns the ``{1, 0, -1}`` vector for an image, or ``None`` when
    this source does not cover it (the dataset then fills ``-1``, so an uncovered
    image reads as "not annotated" rather than raising).
    """

    name: str
    schema: LabelSchema

    def labels_for(self, example_id: str) -> tuple[int, ...] | None: ...


def _cell_to_sentinel(raw: str | None) -> int:
    """A CSV cell -> the ``{1, 0, -1}`` convention. Blank / missing / ``-1`` -> ``-1``."""
    if raw is None:
        return -1
    text = raw.strip()
    if text == "" or text.lower() in ("nan", "na", "-1"):
        return -1
    return 1 if float(text) >= 0.5 else 0


class CsvLabelSource:
    """Per-image labels from a CSV in a source space, mapped to the model schema.

    The CSV has an id column plus one column per source-schema label it carries (any
    it omits are read as ``-1``). Each row becomes a source-space vector; a `LabelMap`
    renames/splits/drops it into the model schema. Rows are read once at construction
    and held in memory — these tables are one row per image, tens of thousands at most.
    """

    def __init__(
        self,
        path: str | Path,
        label_map: LabelMap,
        *,
        id_column: str = "example_id",
        name: str = "csv",
    ) -> None:
        self.name = name
        self._map = label_map
        self.schema = label_map.target
        source_labels = label_map.source.labels
        table: dict[str, tuple[int, ...]] = {}
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or ())
            if id_column not in fields:
                raise ValueError(
                    f"{path}: label source needs an id column {id_column!r}; "
                    f"found {reader.fieldnames}"
                )
            present = [lbl for lbl in source_labels if lbl in fields]
            if not present:
                raise ValueError(
                    f"{path}: none of the source-schema labels {source_labels} are "
                    f"columns here; found {reader.fieldnames}. Check the source `space`."
                )
            for row in reader:
                vec = tuple(
                    _cell_to_sentinel(row.get(lbl)) if lbl in fields else -1
                    for lbl in source_labels
                )
                table[row[id_column]] = vec
        self._table = table

    def labels_for(self, example_id: str) -> tuple[int, ...] | None:
        source_vec = self._table.get(example_id)
        if source_vec is None:
            return None
        return self._map.map_gt(source_vec)

    def __len__(self) -> int:
        return len(self._table)
