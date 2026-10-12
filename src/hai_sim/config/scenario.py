"""Scenario configuration: validated YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

QualityMetric = Literal["laplacian_var", "intensity_std"]
"""Image-quality proxy shared by the severity and difficulty image-quality models."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationConfig(_Strict):
    """Post-hoc calibration of the AI's probabilities. See `hai_sim.uq.calibration`.

    ``temperature`` is the standard first choice (Guo et al. 2017): one parameter,
    monotone, so it changes the numbers without touching any ranking. ``platt`` adds an
    intercept, which is what a model biased toward calling findings present needs.
    ``isotonic`` fits any monotone shape and overfits small samples happily.

    Parameters are *fitted*, not guessed: use ``scripts/fit_calibration.py`` against an
    inference cache. Per-pathology Platt is 2 x L numbers per (model, dataset), so it
    lives in a file next to the cache (``path``) rather than in the scenario. Fit on data
    the study does not score (``patient_generator.split``), and remember calibration
    belongs to (model + readout + prompt + population), not to a model in the abstract.

    ``stage`` says what gets calibrated: ``output`` (default) the fused probability the
    reader is shown, leaving the uncertainty on the raw members; ``members`` each member
    before fusion, so the uncertainty describes calibrated numbers too. See
    `hai_sim.uq.calibration`.
    """

    kind: Literal["identity", "temperature", "platt", "isotonic"] = "identity"
    stage: Literal["output", "members"] = "output"
    # platt: a per-label calibration file written by scripts/fit_calibration.py; when
    # set it replaces the scalar a / b below.
    path: str | None = None
    # temperature: scalar, or one value per label (len must equal the schema).
    temperature: float | tuple[float, ...] = 1.0
    # platt: sigmoid(a * logit(p) + b)
    a: float = 1.0
    b: float = 0.0
    # isotonic: fitted knots, x ascending in [0, 1]
    knots_x: tuple[float, ...] = ()
    knots_y: tuple[float, ...] = ()


class AIModelConfig(_Strict):
    kind: Literal["random_mock", "torchxrv", "cached", "medgemma", "fixed_accuracy"] = (
        "random_mock"
    )
    # A label for study tables when a whole `ai` block is a factor level. No effect on a run.
    name: str | None = None
    service_time: float = 0.05
    # fixed_accuracy params: a synthetic signal-detection AI whose skill is a dial, for the
    # "AI performs well — is it trusted?" study (thesis §5.5). Give `d_prime` to set skill
    # directly (the natural quality-sweep axis) or a `sensitivity`/`specificity` operating
    # point; `prior` is the base rate its calibrated belief assumes. Needs the patient
    # generator's `embed_truth` so the synthetic reader can see the case it is scoring.
    d_prime: float | tuple[float, ...] | None = None
    criterion: float | tuple[float, ...] | None = None
    sensitivity: float | tuple[float, ...] = 0.9
    specificity: float | tuple[float, ...] = 0.9
    prior: float | tuple[float, ...] = 0.5
    # cached params: replay an inference cache (real ensemble predictions, no re-inference).
    # Reuses uq / ensemble_measure / decision_threshold below.
    cache_stem: str | None = None
    # Cache matrix — transparent memoization of torchxrv/medgemma inference, addressed per
    # atomic model as `<cache_root>/<dataset>__<model>` (see `hai_sim.analysis.cache_matrix`).
    # Only torchxrv/medgemma are cacheable; every other AI kind (and validation, which builds
    # the model directly) always runs live regardless of this field. `require` (default) =
    # replay the cells, error if one is missing; `fill` = build missing cells inline (on the
    # GPU server); `report` = print the plan and stop; `off` = skip the cache and run the live
    # model (a quick smoke, or feature-space severity which the cache does not store).
    cache_mode: Literal["off", "require", "fill", "report"] = "require"
    cache_root: str = "data/cache"
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
    # The AI's own call, used for its stand-alone errors and the reliance metrics (the reader
    # is shown the probability, not this call). A number, or "prevalence_matched": per
    # finding, the cut at which the model calls as many positives as there are, read from
    # the calibration file. A calibrated probability of a rare finding almost never reaches
    # 0.5, so a fixed 0.5 would make a calibrated AI call nothing at all.
    decision_threshold: float | Literal["prevalence_matched"] = 0.5
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
    # vLLM HTTP backend: serve medgemma from a running vLLM OpenAI-compatible endpoint
    # instead of loading the weights in this process (e.g. the k3s node's pods — which also
    # lets the 27B run on one 96 GB card with no in-process sharding). `endpoint` is the base
    # URL up to and including /v1; `served_model_name` is the server's --served-model-name,
    # which the request's `model` field must match. Only the `independent_question` readout
    # is supported over HTTP (teacher-forcing needs per-position logits the API hides). See
    # models/vllm_backend.py. `concurrency` is in-flight requests per image (<= vLLM's
    # --max-num-seqs); keep it modest when the GPU is shared with other workloads.
    backend: Literal["transformers", "vllm"] = "transformers"
    endpoint: str | None = None
    served_model_name: str | None = None
    concurrency: int = 8
    top_logprobs: int = 20
    # Optional post-hoc probability calibration, applied to member probabilities before
    # they are fused. Omit → raw model output, which for both backends measured so far
    # is badly over-confident (measured during development; thesis §2.4).
    calibration: CalibrationConfig | None = None


class ServiceTimeConfig(_Strict):
    """How long one read takes. See `hai_sim.fatigue.service` for why this is a seam.

    ``lognormal`` (default) is the conventional right-skewed shape. The two wrappers
    compose on top of it:

    - ``scale`` multiplies by a constant — the knob for the open question (thesis §5.7) of
      whether AI assistance speeds a reader up (``factor < 1``, Seah 2021 measured
      81 s → 56 s) or costs them verification time (``factor > 1``). Sweep it; the
      literature does not settle it.
    - ``fatigue_slope`` stretches the read in proportion to the ``sig.fatigue`` signal,
      which is the time dimension of the doctor model (thesis §5.7).

    Both may be set at once: the scale is applied to the fatigue-modulated draw.
    """

    kind: Literal["lognormal", "constant"] = "lognormal"
    mu: float = 2.5
    sigma: float = Field(default=0.4, ge=0.0)
    value: float = Field(default=1.0, ge=0.0)
    factor: float = Field(default=1.0, ge=0.0)
    fatigue_slope: float = Field(default=0.0, ge=0.0)


class CapabilityConfig(_Strict):
    """A persona's capability = the reader's own, AI-unaided diagnostic ability."""

    kind: Literal["fixed_accuracy", "random_mock", "sdt"] = "fixed_accuracy"
    service_time_mu: float = 2.5
    service_time_sigma: float = 0.4
    # fixed_accuracy / sdt params (scalar applied to all labels, or per-label vector).
    # `fixed_accuracy` keeps or flips each label on a fixed-bias coin; `sdt` is the
    # signal-detection observer of thesis §5.7, whose belief varies case by case (so an
    # integration policy becomes a curve, not a step). For `sdt`, capability is set either
    # by `d_prime` (the SDT-native skill axis) or by the sens/spec operating point below,
    # and `prior` is the base rate its belief is calibrated to.
    sensitivity: float | tuple[float, ...] = 0.9
    specificity: float | tuple[float, ...] = 0.9
    d_prime: float | tuple[float, ...] | None = None
    criterion: float | tuple[float, ...] | None = None
    prior: float | tuple[float, ...] = 0.5
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
    # How much of its accuracy this reader loses at difficulty 1.0 / fatigue 1.0.
    # Both default to 0, so a persona is case-blind unless a scenario says otherwise —
    # and a case-blind reader errs independently of the AI by construction, which is
    # what pins the human-AI error correlation at zero (thesis §5.7).
    difficulty_slope: float = Field(default=0.0, ge=0.0)
    fatigue_slope: float = Field(default=0.0, ge=0.0)
    # Accuracy floor: degradation stops at a coin flip rather than inverting.
    floor: float = Field(default=0.5, ge=0.0, le=1.0)
    # Optional swappable reading-time model; omit → the lognormal above.
    service_time: ServiceTimeConfig | None = None


class TrustConfig(_Strict):
    """A persona's trust = a swappable AI-integration policy (used only when sees_ai).

    `linear_pool` (default) mixes probabilities with a fixed Weight of Advice
    (`weight_ai`, the advice-taking literature's mean is ~0.39). `confidence_gated`
    applies that weight only where the AI is confident enough (`confidence_threshold`).
    `log_odds_fusion` is the normative, uncertainty-weighted Bayesian pool.
    """

    kind: Literal["linear_pool", "confidence_gated", "log_odds_fusion"] = "linear_pool"
    # The fused belief at or above which the team calls a finding present. 0.5 keeps the
    # team on the same footing as the unaided reader, so a paired contrast measures the
    # advice and not a change of operating point; 1/(1 + c_FN/c_FP) is the cost-optimal
    # cut, meaningful only on calibrated probabilities.
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    # linear_pool / confidence_gated: Weight of Advice
    weight_ai: float = Field(default=0.39, ge=0.0, le=1.0)
    # confidence_gated: min AI confidence (max(p, 1-p)) to listen to the AI
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # log_odds_fusion
    beta: float = Field(default=1.0, ge=0.0)  # AI down-weight sharpness on uncertainty
    prior: float = Field(default=0.5, gt=0.0, lt=1.0)


class FusionConfig(_Strict):
    """Scenario-level choice of *how* the doctor's and the AI's opinions are combined.

    A persona says who the reader is and how much it trusts the AI (``weight_ai``); this
    block says by which rule that trust is exercised, so one study can run the same
    personas under several rules. Any field left unset keeps the persona's own value.
    """

    kind: Literal["linear_pool", "confidence_gated", "log_odds_fusion"] | None = None
    beta: float | None = Field(default=None, ge=0.0)
    decision_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)


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
    kind: Literal["random", "uncertainty_desc", "probability_desc"]
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
    quality_metric: QualityMetric = "laplacian_var"
    # embedding
    embedding_reduce: Literal["l2_norm", "mean"] = "l2_norm"


class DifficultyConfig(_Strict):
    """Optional per-case difficulty model. Omit the block → no difficulty computed.

    The latent ``delta`` of thesis §5.7, and the only route by which a simulated
    reader's errors can correlate with the AI's. Thesis §8.3 states that there is no
    canonical difficulty model and that the ``Beta(2, 5)`` prior is chosen by hand; the
    response is to make it a configuration line and report the sensitivity.

    - ``constant``      — the null model. Run it to show an effect vanishes without
                          difficulty.
    - ``prior``         — a fresh ``Beta(a, b)`` per case, keyed by image so it is
                          stable across replications. Unrelated to the image, hence the
                          ``rho = 0`` control.
    - ``uncertainty``   — the AI's own predictive uncertainty: the coupling is then
                          *induced* from real model behaviour instead of imposed.
                          ``source`` may be ``uncertainty`` (whatever the run's UQ
                          selected) or a named measure (``mutual_information`` =
                          epistemic, ``mean_entropy`` = aleatoric), which needs
                          ``ai.uq: ensemble`` to exist at all.
    - ``image_quality`` — blur/contrast only; the acquisition share of difficulty.
    - ``table``         — a CSV of ``example_id,difficulty``. This is how inter-reader
                          disagreement and any future learned complexity model plug in,
                          without new library code.
    """

    kind: Literal["constant", "prior", "uncertainty", "image_quality", "table"] = "prior"
    # constant
    value: float = Field(default=0.5, ge=0.0, le=1.0)
    # prior: Beta(a, b) — Beta(2, 5) is the default and an explicit assumption (thesis §8.3)
    a: float = Field(default=2.0, gt=0.0)
    b: float = Field(default=5.0, gt=0.0)
    # uncertainty
    source: str = "uncertainty"
    aggregator: Literal["mean", "max", "none"] = "mean"
    scale: float | None = None
    # image_quality
    quality_metric: QualityMetric = "laplacian_var"
    reference: float | None = None
    # table
    path: str | None = None
    id_column: str = "example_id"
    value_column: str = "difficulty"
    default: float = Field(default=0.5, ge=0.0, le=1.0)
    strict: bool = False


class FatigueConfig(_Strict):
    """Optional reader fatigue curve. Omit the block → readers never tire.

    Thesis §5.7 records that radiology has no analytic fatigue curve — only
    isolated pre/post points — and treats the shape as **a mandatory sensitivity
    axis**. Hence four shapes rather than one, and a ``composite`` that
    combines them, since a night shift is circadian *and* time-on-task.

    ⚠️ Time has no units: parameters are in simulation time units and the defaults
    assume one unit = one hour. Say which in any write-up.
    """

    kind: Literal["none", "linear", "vigilance", "circadian", "composite"] = "none"
    basis: Literal["cases", "time_on_shift", "busy_time", "utilisation"] = "cases"
    # linear
    per_unit: float = Field(default=0.01, ge=0.0)
    cap: float = Field(default=1.0, ge=0.0, le=1.0)
    # vigilance: asymptote * (1 - exp(-x / scale))
    asymptote: float = Field(default=1.0, ge=0.0, le=1.0)
    scale: float = Field(default=40.0, gt=0.0)
    # circadian: mean + amplitude * cos(2*pi*(clock - acrophase) / period)
    amplitude: float = Field(default=0.3, ge=0.0)
    mean: float = Field(default=0.3)
    period: float = Field(default=24.0, gt=0.0)
    acrophase: float = 5.0
    # composite: which of the above to combine, and how
    parts: tuple[Literal["linear", "vigilance", "circadian"], ...] = ()
    combine: Literal["max", "sum", "mean"] = "max"


class ArrivalConfig(_Strict):
    """How arrival times are generated. Omit → homogeneous Poisson at ``arrival_rate``.

    Thesis §5.4 makes the escalation conditional on a test rather than on
    taste: start homogeneous, move to a non-homogeneous Poisson process (NHPP) with a
    piecewise-constant rate, and go further only if a conditional Kolmogorov-Smirnov
    test rejects. Kim & Whitt (2014) explain the caution — much of the reported burstiness in
    hospital arrivals is an artefact of pooling hours across weeks of differing
    intensity, so an overdispersed default would bake in a known artefact.

    For ``nhpp``, ``rates`` is one rate per slot and ``period`` the length of the cycle;
    168 hourly rates over a period of 168 gives the day-of-week stratification of De
    Santis et al. (2022), necessary rather than optional.
    """

    kind: Literal["poisson", "deterministic", "nhpp"] = "poisson"
    # nhpp: piecewise-constant rate table repeating with `period`
    rates: tuple[float, ...] = ()
    period: float | None = None


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


class LabelSourceConfig(_Strict):
    """One named label source attached to an image set (``kind: image_folder``).

    A CSV of per-image labels in a declared source ``space``, mapped into the model
    schema by ``map``. Several may be attached to the same images; the run's
    ``patient_generator.gt_source`` picks which one is the scoring ground truth, and
    the rest ride along for a paired-contrast study that swaps the believed truth.

    - ``space`` — the label space the CSV columns live in (a `build_label_schema`
      name such as ``vindr`` or ``xrv``); its labels are the columns read.
    - ``map`` — how that space aligns to the model schema: a registered name
      (``vindr14->xrv``), an inline ``{source_label: target_label(s)}`` mapping (the
      escape hatch — a value may be a list to split one finding across several), or
      omitted for the by-name identity when the CSV is already in the model schema.
    """

    name: str
    kind: Literal["csv"] = "csv"
    path: str
    space: str = "xrv"
    map: str | dict[str, str | list[str]] | None = None
    id_column: str = "example_id"


class PatientGenConfig(_Strict):
    kind: Literal[
        "poisson", "mock_dataset", "xrv_dataset", "inference_cache", "image_folder"
    ] = "poisson"
    arrival_rate: float = Field(gt=0)
    n_patients: int = Field(gt=0)
    # *When* patients arrive, separated from *which* patient arrives. Omit → a
    # homogeneous Poisson process at `arrival_rate`, the baseline of thesis §5.4. Note that load
    # is not background: order effects only exist under saturation, so a triage study
    # is a study of what this block produces.
    arrivals: ArrivalConfig | None = None
    # inference_cache params: replay a prebuilt inference cache as the patient stream.
    cache_stem: str | None = None
    # poisson / mock_dataset params (synthetic)
    prevalence: float | tuple[float, ...] = 0.2
    image_shape: tuple[int, ...] = (1, 1)
    # poisson only: write the case's true label vector into the "image" so a synthetic AI
    # (ai.kind=fixed_accuracy) can read the case it scores without the engine handing the
    # AI stage a ground truth a real model must never see.
    embed_truth: bool = False
    # mock_dataset params: size of the synthetic Dataset pool (n_patients are drawn from it)
    n_examples: int = Field(default=100, gt=0)
    # xrv_dataset params
    dataset: Literal["nih", "rsna", "chexpert"] = "nih"
    # Stable identifier for this dataset in the cache matrix (the `<dataset>` half of a
    # cache stem `<dataset>__<model>`, see `hai_sim.analysis.cache_matrix`). Defaults to
    # `dataset` for kind=xrv_dataset and to the image folder's parent directory name for
    # kind=image_folder — set it explicitly when neither is meaningful (e.g. `vindr`).
    dataset_id: str | None = None
    # Ground truth is owned by the dataset: selecting a dataset selects its GT.
    # 'native' = the dataset's own labels. For kind=image_folder, gt_source instead
    # names one of `label_sources` below ('native' = the first of them).
    gt_source: str = "native"
    imgpath: str | None = None
    csvpath: str | None = None
    # image_folder params: pixels from `imgpath`/*`image_ext`, labels from one or more
    # CSV `label_sources` mapped into the model schema. The generic "images +
    # annotation CSVs" dataset (VinDr, ...); see `LabelSourceConfig`.
    label_sources: tuple[LabelSourceConfig, ...] = ()
    image_ext: str = ".png"
    # "permutation" (inference_cache only): every image once, in a random order.
    sample_mode: Literal["sequential", "random", "permutation"] = "sequential"
    # inference_cache only: restrict the patient pool to the even- or odd-indexed cache
    # rows. A calibration fitted on one half must be studied on the other
    # (scripts/fit_calibration.py --split even  ->  split: odd).
    split: Literal["all", "even", "odd"] = "all"
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
    a macro-averaged bundle into a per-pathology one — the literal ``"truth"`` — for every
    pathology, the patients who have it and the patients who do not, which is how a time
    metric is asked "do the sick wait less?" — or something readable off a
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
    label_schema: Literal["default_v0", "single", "chexpert", "xrv"] = "default_v0"
    label_view: LabelViewConfig | None = None
    patient_generator: PatientGenConfig
    ai: AIModelConfig
    persona: str | PersonaConfig
    # How opinions are combined, overriding the persona's own rule. See `FusionConfig`.
    fusion: FusionConfig | None = None
    sort: SortConfig
    severity: SeverityConfig | None = None
    # The signal plane: per-read scalars the engine computes between the two stations
    # and hands to the doctor. Both are optional and both reach the telemetry as
    # ``sig.*`` whether or not any reader consults them.
    difficulty: DifficultyConfig | None = None
    fatigue: FatigueConfig | None = None
    # Which swappable KPI bundles to compute for this run's end-to-end test. Each entry
    # is a registered name or a reference to your own bundle, e.g.
    # "configs/examples/custom_metrics.py:CostBundle".
    # None → the default set (system, actors, team, joint_table).
    kpis: tuple[str, ...] | None = None
    # Optional conditioning axes: every selected bundle is computed once per subgroup
    # (per severity band, per load regime, per pathology, ...). Several axes cross.
    kpi_groups: tuple[KPIGroupConfig, ...] = ()
    n_ai_workers: int = Field(default=1, ge=1)
    n_doctor_workers: int = Field(default=1, ge=1)


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
