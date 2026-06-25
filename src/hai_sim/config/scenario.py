"""Scenario configuration: validated YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIModelConfig(_Strict):
    kind: Literal["random_mock", "torchxrv"] = "random_mock"
    service_time: float = 0.05
    # random_mock params
    alpha_conf: float = 2.0
    beta_conf: float = 2.0
    alpha_unc: float = 2.0
    beta_unc: float = 5.0
    # torchxrv params
    weights: str | tuple[str, ...] = "densenet121-res224-all"
    uq: Literal["entropy", "ensemble"] = "entropy"
    ensemble_measure: Literal[
        "mutual_information", "predictive_entropy", "mean_entropy", "std"
    ] = "mutual_information"
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "cpu"


class DoctorConfig(_Strict):
    kind: Literal["random_mock", "fixed_accuracy"] = "random_mock"
    service_time_mu: float = 2.5
    service_time_sigma: float = 0.4
    # random_mock params
    alpha_conf: float = 5.0
    beta_conf: float = 2.0
    alpha_unc: float = 2.0
    beta_unc: float = 5.0
    use_ai_hint: bool = False
    assisted_policy: Literal["noop", "follow_when_certain"] = "noop"
    ai_hint_uncertainty_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # fixed_accuracy params (scalar applied to all labels, or per-label vector)
    sensitivity: float | tuple[float, ...] = 0.9
    specificity: float | tuple[float, ...] = 0.9


class SortConfig(_Strict):
    kind: Literal["random", "uncertainty_desc"]
    aggregator: Literal["mean", "max", "weighted"] = "mean"
    weights: tuple[float, ...] | None = None


class PatientGenConfig(_Strict):
    kind: Literal["poisson", "xrv_dataset"] = "poisson"
    arrival_rate: float = Field(gt=0)
    n_patients: int = Field(gt=0)
    # poisson params
    prevalence: float | tuple[float, ...] = 0.2
    image_shape: tuple[int, ...] = (1, 1)
    # xrv_dataset params
    dataset: Literal["nih", "rsna", "chexpert"] = "nih"
    imgpath: str | None = None
    csvpath: str | None = None
    sample_mode: Literal["sequential", "random"] = "sequential"


class ScenarioConfig(_Strict):
    name: str = "unnamed"
    seed: int = 0
    label_schema: Literal["default_v0", "chexpert", "xrv"] = "default_v0"
    patient_generator: PatientGenConfig
    ai: AIModelConfig
    doctor: DoctorConfig
    sort: SortConfig
    n_ai_workers: int = 1
    n_doctor_workers: int = 1

    @field_validator("n_ai_workers", "n_doctor_workers")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v


def load_scenario(path: Path) -> ScenarioConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)
