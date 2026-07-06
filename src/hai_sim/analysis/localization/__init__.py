"""Offline spatial + label validation of a model against expert ground truth.

This subpackage is the concrete implementation behind the three validation-plane
protocols in :mod:`hai_sim.core.protocols` (`SaliencyProvider`,
`LocalizationDataset`, `LocalizationMetric`). It answers "does the model look at
the right place?" by comparing a model's saliency heatmap to expert masks, and
"does the model call the right label?" against expert image-level labels — for
any model plugged into the framework.
"""

from __future__ import annotations

from hai_sim.analysis.localization.cache import (
    LocalizationCache,
    build_localization_cache,
)
from hai_sim.analysis.localization.chestxdet import ChestXDetDataset
from hai_sim.analysis.localization.evaluate import evaluate_from_cache
from hai_sim.analysis.localization.metrics import (
    Dice,
    EnergyInMask,
    IoU,
    PointingGame,
    default_metrics,
)
from hai_sim.analysis.localization.report import ComponentValidationReport
from hai_sim.analysis.localization.saliency import GradCAMSaliency
from hai_sim.analysis.localization.validator import LocalizationValidator

__all__ = [
    "ChestXDetDataset",
    "ComponentValidationReport",
    "Dice",
    "EnergyInMask",
    "GradCAMSaliency",
    "IoU",
    "LocalizationCache",
    "LocalizationValidator",
    "PointingGame",
    "build_localization_cache",
    "default_metrics",
    "evaluate_from_cache",
]
