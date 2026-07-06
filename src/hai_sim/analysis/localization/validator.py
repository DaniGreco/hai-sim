"""Drive a model over a localization dataset and produce a dual-GT report.

`LocalizationValidator` is the composition point of the validation plane: hand it
an actor (any `DiagnosticActor`), a `LocalizationDataset`, and — optionally — a
`SaliencyProvider` and a set of `LocalizationMetric`s, and it streams every image
through both checks:

- **label**: the actor's predicted labels vs the expert image-level labels
  (per-pathology TP/FN/FP/TN, ``-1`` entries skipped);
- **localization**: for every pathology the dataset localizes in the image, the
  saliency heatmap scored against the expert mask by each metric.

Grad-CAM is expensive (a backward pass per member per label), so the run is
**resumable**: per-image records are persisted atomically every
``checkpoint_every`` images, and a re-run with the same ``records_path`` skips the
indices already done. Heatmaps themselves are never stored — they are scored and
discarded — so the on-disk footprint stays tiny. Without a `SaliencyProvider` the
run degrades cleanly to label-only validation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hai_sim.analysis.localization.metrics import default_metrics
from hai_sim.analysis.localization.report import ComponentValidationReport, assemble_report

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hai_sim.core.protocols import (
        DiagnosticActor,
        LocalizationDataset,
        LocalizationMetric,
        SaliencyProvider,
    )


def _save_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(records), encoding="utf-8")
    os.replace(tmp, path)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(json.loads(Path(path).read_text(encoding="utf-8")))


class LocalizationValidator:
    """Compose actor + dataset + saliency + metrics into a `ComponentValidationReport`."""

    def __init__(
        self,
        actor: DiagnosticActor,
        dataset: LocalizationDataset,
        saliency: SaliencyProvider | None = None,
        metrics: Sequence[LocalizationMetric] | None = None,
    ) -> None:
        self.actor = actor
        self.dataset = dataset
        self.saliency = saliency
        self.metrics = tuple(metrics) if metrics is not None else default_metrics()

    def run(
        self,
        indices: Sequence[int] | None = None,
        records_path: str | Path | None = None,
        checkpoint_every: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> ComponentValidationReport:
        all_indices = list(indices) if indices is not None else list(range(len(self.dataset)))

        records: list[dict[str, Any]] = []
        if records_path is not None and Path(records_path).exists():
            records = _load_records(records_path)
        done = {int(r["index"]) for r in records}

        since_ckpt = 0
        for idx in all_indices:
            if int(idx) in done:
                continue
            records.append(self._process(int(idx)))
            since_ckpt += 1
            if progress is not None:
                progress(len(records), len(all_indices))
            if checkpoint_every and records_path is not None and since_ckpt >= checkpoint_every:
                _save_records(records_path, records)
                since_ckpt = 0

        if records_path is not None:
            _save_records(records_path, records)
        return self._build_report(records)

    def _process(self, idx: int) -> dict[str, Any]:
        ex = self.dataset[idx]
        out = self.actor.diagnose([ex.image])
        pred = [int(x) for x in out.diagnosis.labels]

        loc: list[dict[str, Any]] = []
        if self.saliency is not None:
            for i, mask in enumerate(ex.masks):
                if mask is None:
                    continue
                heatmap = self.saliency.heatmap(ex.image, i)
                scores: dict[str, float | None] = {}
                for metric in self.metrics:
                    val = float(metric.score(heatmap, mask))
                    scores[metric.name] = None if math.isnan(val) else val
                loc.append({"label": i, "predicted": pred[i], "scores": scores})

        return {
            "index": idx,
            "image_id": ex.image_id,
            "gt": [int(x) for x in ex.labels],
            "pred": pred,
            "loc": loc,
        }

    def _build_report(self, records: list[dict[str, Any]]) -> ComponentValidationReport:
        return assemble_report(
            records,
            self.dataset.schema.labels,
            [m.name for m in self.metrics],
            model=getattr(self.actor, "name", type(self.actor).__name__),
            dataset=getattr(self.dataset, "name", type(self.dataset).__name__),
            saliency=getattr(self.saliency, "name", None) if self.saliency is not None else None,
            decision_threshold=getattr(self.actor, "decision_threshold", None),
        )
