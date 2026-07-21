"""Scenario configuration: validated YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIModelConfig(_Strict):
    kind: Literal["random_mock", "torchxrv", "cached", "medgemma"] = "random_mock"
    service_time: float = 0.05
    # cached params: replay an inference cache (real ensemble predictions, no re-inference).
    # Reuses uq / ensemble_measure / decision_threshold below.
    cache_stem: str | None = None
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
    # medgemma params — a generative vision-language model as the AI reader.
    # `readout` is how a probability is elicited from a model that emits text:
    #   independent_question   → one question per pathology off a shared reading of the
    #                            image. THE DEFAULT: the alternatives below let the
    #                            answers contaminate each other.
    #   teacher_forced_logprob → one forced checklist. Superseded — measured to be
    #                            dominated by its own filler (see readout.py). Diagnostic
    #                            only; do not produce results with it.
    #   generated_json         → parse the confidences the model writes down (NOT the
    #                            same quantity; kept as a deliberate contrast)
    # `prompt_set` names a set of paraphrases used as ensemble members: `single` is one
    # pass, `triple` gives member disagreement for an epistemic estimate at 3x the cost.
    model_id: str = "google/medgemma-1.5-4b-it"
    readout: Literal[
        "independent_question", "teacher_forced_logprob", "generated_json"
    ] = "independent_question"
    prompt_set: Literal["single", "triple"] = "single"
    both_polarities: bool = False
    dtype: str = "auto"


class CapabilityConfig(_Strict):
    """A persona's capability = the reader's own, AI-unaided diagnostic ability."""

    kind: Literal["fixed_accuracy", "random_mock"] = "fixed_accuracy"
    service_time_mu: float = 2.5
    service_time_sigma: float = 0.4
    # fixed_accuracy params (scalar applied to all labels, or per-label vector)
    sensitivity: float | tuple[float, ...] = 0.9
    specificity: float | tuple[float, ...] = 0.9
    # random_mock params
    alpha_conf: float = 5.0
    beta_conf: float = 2.0
    alpha_unc: float = 2.0
    beta_unc: float = 5.0
    # Calibration source. 'fixed' = the sens/spec above. 'reader' = calibrate on a real
    # radiologist's per-finding reads — gated on the dataset advertising per-reader reads;
    # wired with the real-data work (raises for now).
    source: Literal["fixed", "reader"] = "fixed"
    reader_id: str | None = None


class TrustConfig(_Strict):
    """A persona's trust = a swappable AI-integration policy (used only when sees_ai).

    `linear_pool` (default) mixes probabilities with a fixed Weight of Advice
    (`weight_ai`, the advice-taking literature's mean is ~0.39). `confidence_gated`
    applies that weight only where the AI is confident enough (`confidence_threshold`).
    `log_odds_fusion` is the normative, uncertainty-weighted Bayesian pool.
    """

    kind: Literal["linear_pool", "confidence_gated", "log_odds_fusion"] = "linear_pool"
    # linear_pool / confidence_gated: Weight of Advice
    weight_ai: float = Field(default=0.39, ge=0.0, le=1.0)
    # confidence_gated: min AI confidence (max(p, 1-p)) to listen to the AI
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # log_odds_fusion
    beta: float = Field(default=1.0, ge=0.0)  # AI down-weight sharpness on uncertainty
    weight_doctor: float = Field(default=1.0, ge=0.0)
    prior: float = Field(default=0.5, gt=0.0, lt=1.0)


class PersonaConfig(_Strict):
    """A doctor persona = capability x visibility x trust — the doctor model itself.

    ``sees_ai`` is visibility; ``trust`` (an integration policy) is used only when the
    persona sees the AI. In a scenario, ``persona`` may instead be a library name
    (e.g. ``senior_trusting``) that expands to one of these.
    """

    name: str | None = None
    capability: CapabilityConfig = Field(default_factory=CapabilityConfig)
    sees_ai: bool = False
    trust: TrustConfig | None = None


class SortConfig(_Strict):
    kind: Literal["random", "uncertainty_desc"]
    aggregator: Literal["mean", "max", "weighted"] = "mean"
    weights: tuple[float, ...] | None = None


class SeverityConfig(_Strict):
    """Optional per-patient severity model. Omit the block → no severity computed.

    `activation` is the workhorse: ``severity = agg(criticality * activation)``,
    where `criticality` is a pre-assigned per-label clinical weight (length =
    n_labels) and `activation` comes from the AI output or the ground truth.
    """

    kind: Literal["constant", "activation", "image_quality", "embedding"] = "constant"
    # constant
    value: float = 0.0
    # activation
    source: Literal["ai_probs", "ai_binary", "ai_uncertainty", "gt"] = "ai_probs"
    criticality: tuple[float, ...] | None = None
    aggregator: Literal["mean", "max", "sum"] = "max"
    mode: Literal["scalar", "vector"] = "scalar"
    # image_quality
    quality_metric: Literal["laplacian_var", "intensity_std"] = "laplacian_var"
    # embedding
    embedding_reduce: Literal["l2_norm", "mean"] = "l2_norm"


class LabelViewConfig(_Strict):
    """Projects the model's native labels to the active study label set.

    ``full`` (default) = no projection (multi-label over the whole schema).
    ``subset`` = keep only ``targets`` (one target → a single-label study).
    ``binary_abnormal`` = collapse ``targets`` (or all labels) to a single
    "Abnormal" label (any finding), combining probabilities via ``combine``.
    """

    mode: Literal["full", "subset", "binary_abnormal"] = "full"
    targets: tuple[str, ...] | None = None
    combine: Literal["noisy_or", "max"] = "noisy_or"


class PatientGenConfig(_Strict):
    kind: Literal["poisson", "mock_dataset", "xrv_dataset", "inference_cache"] = "poisson"
    arrival_rate: float = Field(gt=0)
    n_patients: int = Field(gt=0)
    # inference_cache params: replay a prebuilt inference cache as the patient stream.
    cache_stem: str | None = None
    # poisson / mock_dataset params (synthetic)
    prevalence: float | tuple[float, ...] = 0.2
    image_shape: tuple[int, ...] = (1, 1)
    # mock_dataset params: size of the synthetic Dataset pool (n_patients are drawn from it)
    n_examples: int = Field(default=100, gt=0)
    # xrv_dataset params
    dataset: Literal["nih", "rsna", "chexpert"] = "nih"
    # Ground truth is owned by the dataset: selecting a dataset selects its GT.
    # 'native' = the dataset's own labels. Alternative sources (e.g. NIH adjudicated
    # consensus, per-reader reads) are a dataset sub-knob wired with the real-data work.
    gt_source: str = "native"
    imgpath: str | None = None
    csvpath: str | None = None
    sample_mode: Literal["sequential", "random"] = "sequential"
    # View filtering (applied for dataset=nih). Default ("PA",) matches xrv's own
    # default; use ("PA", "AP") to include antero-posterior (portable/ICU) frontals.
    views: tuple[str, ...] = ("PA",)
    # Keep one image per patient (xrv default) → statistically independent samples.
    unique_patients: bool = True
    # Square side the images are resized to after centre-cropping. **This is a per-model
    # property, not a dataset constant**: the TorchXRayVision DenseNet was trained at 224,
    # MedGemma encodes at 896. Feeding one model the other's resolution measures it
    # outside its training distribution, so a comparison of *capability* gives each
    # backend its own value and says so.
    resize: int = Field(default=224, gt=0)


class KPIGroupConfig(_Strict):
    """One conditioning axis for the KPI bundles: every bundle is computed per subgroup.

    ``key`` is either the literal ``"label"`` — one subgroup per pathology, which turns
    a macro-averaged bundle into a per-pathology one — or something readable off a
    record: a plain field (``severity``, ``priority_key``, ``ts_arrival``) or an
    observation (``ctx.sex``, ``load.in_system_at_arrival``, ``ai.uq_method``).

    ``edges`` bins a continuous quantity into ordered groups: ``[1, 5]`` gives ``<1``,
    ``1-5``, ``>=5``. Leave it out for categoricals. Records missing the key are
    dropped unless ``include_missing`` — otherwise the denominator changes silently.
    """

    key: str
    edges: tuple[float, ...] = ()
    include_missing: bool = False


class ScenarioConfig(_Strict):
    name: str = "unnamed"
    seed: int = 0
    label_schema: Literal["default_v0", "chexpert", "xrv"] = "default_v0"
    label_view: LabelViewConfig | None = None
    patient_generator: PatientGenConfig
    ai: AIModelConfig
    persona: str | PersonaConfig
    sort: SortConfig
    severity: SeverityConfig | None = None
    # Which swappable KPI bundles to compute for this run's end-to-end test. Each entry
    # is a registered name or a reference to your own bundle, e.g.
    # "configs/examples/custom_metrics.py:CostBundle".
    # None → the default set (system, actors, team, joint_table).
    kpis: tuple[str, ...] | None = None
    # Optional conditioning axes: every selected bundle is computed once per subgroup
    # (per severity band, per load regime, per pathology, ...). Several axes cross.
    kpi_groups: tuple[KPIGroupConfig, ...] = ()
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


# --- Validation scenarios (offline heatmap + label validation) ---------------


class LocalizationDatasetConfig(_Strict):
    kind: Literal["chestxdet"] = "chestxdet"
    annotation_path: str
    image_dir: str
    resize: int = 224


class ValidationMetricsConfig(_Strict):
    pointing_game: bool = True
    # Chebyshev radius (px) for a pointing-game hit; 0 = strict exact pixel. A coarse
    # Grad-CAM (7x7 upsampled to 224) quantizes its peak to ~32 px cells, so ~16 is a
    # principled default that scores the metric at the map's real resolution.
    pointing_tolerance: int = 0
    energy_in_mask: bool = True
    iou_thresholds: tuple[float, ...] = (0.5,)
    dice_thresholds: tuple[float, ...] = (0.5,)


class ValidationScenarioConfig(_Strict):
    """Declarative validation run: a model + a dual-GT dataset + a saliency method.

    Mirrors `ScenarioConfig` for simulations, so a validation is configured,
    versioned and reproduced exactly like a simulation. Reuses `AIModelConfig`
    so the very model wired into a simulation is the one validated here.
    """

    name: str = "unnamed_validation"
    seed: int = 0
    label_schema: Literal["xrv"] = "xrv"
    dataset: LocalizationDatasetConfig
    ai: AIModelConfig
    saliency: Literal["none", "grad_cam"] = "grad_cam"
    metrics: ValidationMetricsConfig = Field(default_factory=ValidationMetricsConfig)
    limit: int = 0  # 0 = all images


def load_validation_scenario(path: Path) -> ValidationScenarioConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ValidationScenarioConfig.model_validate(data)
