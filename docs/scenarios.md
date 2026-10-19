# Scenario reference

A scenario is one YAML file that describes one run: which patients arrive and when, which
AI reads them first, which reader reads them second, in what order, and which metrics are
computed. It is validated by `pydantic` models in
[`src/hai_sim/config/scenario.py`](../src/hai_sim/config/scenario.py) with
`extra = forbid`, so **any unknown key raises** and a typo fails loudly.

How to read the tables:

- *required* means the field has no default;
- *omitted* means the field is optional, and the default column says what happens when it
  is absent;
- *number or per label* means one number, or one number per finding of the active label set;
- a field that applies to one `kind` only says so; a field of another kind is accepted and
  ignored.

For clean comparisons: **change one block at a time and keep `seed` fixed.**
[`configs/examples/quickstart.yaml`](../configs/examples/quickstart.yaml) is a complete,
commented scenario to start from.

```yaml
name: my_scenario
seed: 42
label_schema: xrv
patient_generator: { ... }   # required: which patients arrive, and when
ai: { ... }                  # required: the AI reader
persona: senior_trusting     # required: a library name, or a capability/visibility/trust block
fusion: { ... }              # optional: override the persona's integration rule
sort: { ... }                # required: the order of the worklist
severity: { ... }            # optional signals, computed between the two stations
difficulty: { ... }
fatigue: { ... }
label_view: { ... }          # optional: project the labels (subset / binary abnormal)
kpis: [system, actors, team, joint_table]
kpi_groups: [ ... ]          # optional: every metric also per subgroup
n_ai_workers: 1
n_doctor_workers: 1
```

## Contents

- [Quick map](#quick-map) · [Top-level fields](#top-level-fields)
- [`patient_generator`](#patient_generator) · [`ai`](#ai) · [`persona`](#persona) ·
  [`fusion`](#fusion) · [`sort`](#sort)
- [`severity`](#severity) · [`difficulty`](#difficulty) · [`fatigue`](#fatigue) ·
  [`label_view`](#label_view)
- [`kpis`](#kpis) · [`kpi_groups`](#kpi_groups)
- [Common errors](#common-errors) · [Recipes](#recipes)

## Quick map

Which knob answers which question:

| question | block or field to vary | keep fixed |
|---|---|---|
| Does the AI help the reader? | `persona.sees_ai`, or a persona with and without the AI | seed, patients |
| Does the order of the worklist matter? | `sort.kind`, `sort.aggregator` | seed, ai, persona, patients |
| How does the kind of reader change the team? | `persona` | seed, ai, patients |
| Which rule should combine the two opinions? | `fusion.kind` | seed, ai, persona |
| Which uncertainty measure? | `ai.uq`, `ai.ensemble_measure`, `ai.weights` | seed, everything else |
| Operating point of the AI | `ai.decision_threshold`, `ai.calibration` | seed, everything else |
| Load and saturation | `patient_generator.arrival_rate`, `patient_generator.arrivals`, `n_doctor_workers`, reading time | seed, ai, reader quality |

## Top-level fields

<!-- fields: ScenarioConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | `unnamed` | Name of the run, written in the output file names. |
| `seed` | integer | `0` | Seed of the run. Five independent random streams (patients, AI, doctor, worklist, arrivals) are derived from it, so changing one module does not move the others. See [determinism](usage.md#determinism-and-seeds). |
| `label_schema` | `default_v0`, `single`, `chexpert`, `xrv` | `default_v0` | The finding set every per-patient vector is indexed by. `xrv` is the 18-finding TorchXRayVision set used by every real model (`ai.kind: torchxrv` raises without it); `chexpert` is the 14-finding CheXpert convention; `single` is one label; `default_v0` is five unnamed labels for tests. |
| `label_view` | block | omitted, no projection | Projection to the study label set, see [`label_view`](#label_view). |
| `patient_generator` | block | required | See [`patient_generator`](#patient_generator). |
| `ai` | block | required | See [`ai`](#ai). |
| `persona` | library name or block | required | See [`persona`](#persona). |
| `fusion` | block | omitted, the persona's own rule | See [`fusion`](#fusion). |
| `sort` | block | required | See [`sort`](#sort). |
| `severity` | block | omitted, not computed | See [`severity`](#severity). |
| `difficulty` | block | omitted, not computed | See [`difficulty`](#difficulty). |
| `fatigue` | block | omitted, readers never tire | See [`fatigue`](#fatigue). |
| `kpis` | list of names | omitted, all four built-in bundles | Metric bundles to compute, see [`kpis`](#kpis). |
| `kpi_groups` | list of blocks | empty | Conditioning axes, see [`kpi_groups`](#kpi_groups). |
| `n_ai_workers` | integer ≥ 1 | `1` | Parallel servers of the AI station. |
| `n_doctor_workers` | integer ≥ 1 | `1` | Parallel servers of the doctor station. More readers means less saturation and shorter waits, and the order of the worklist matters less. |

## `patient_generator`

`kind` selects the source of the patients:

- `poisson` and `mock_dataset` generate synthetic cases;
- `xrv_dataset` reads a dataset through the TorchXRayVision loaders;
- `image_folder` reads a folder of images with one or more label files (used for VinDr-CXR);
- `inference_cache` replays the image ids and labels of a cache cell without touching the
  pixels. Every study of the paper uses it.

<!-- fields: PatientGenConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `poisson`, `mock_dataset`, `xrv_dataset`, `inference_cache`, `image_folder` | `poisson` | Source of the patients. |
| `arrival_rate` | number > 0 | required | Mean arrivals per time unit. Raise it and the queue saturates, which is when the order of service starts to matter. |
| `n_patients` | integer > 0 | required | Patients generated in the run. |
| `arrivals` | block | omitted, homogeneous Poisson at `arrival_rate` | Arrival process, see [below](#patient_generatorarrivals). |
| `cache_stem` | path | none | `inference_cache` only (required). Stem of the cache cell replayed, without extension, e.g. `data/cache/composed/vindr__ens4`. |
| `prevalence` | number or per label | `0.2` | `poisson`, `mock_dataset`. Probability that each finding is present. |
| `image_shape` | list of integers | `[1, 1]` | `poisson`, `mock_dataset`. Shape of the placeholder image. |
| `embed_truth` | true, false | `false` | `poisson` only. Writes the true label vector into the placeholder image, so that a `fixed_accuracy` AI can score the case without the engine handing ground truth to the AI stage. |
| `n_examples` | integer > 0 | `100` | `mock_dataset` only. Size of the synthetic pool the patients are drawn from. |
| `dataset` | `nih`, `rsna`, `chexpert` | `nih` | `xrv_dataset` only. Dataset loaded. |
| `dataset_id` | text | none: the `dataset` name, or the image folder's parent name | Identifier of the dataset in the cache matrix, the first half of a cache stem `<dataset>__<model>`. Set it when neither default is meaningful (e.g. `vindr`). |
| `gt_source` | text | `native` | Which labels are the scoring ground truth. For `image_folder`, the `name` of one of the `label_sources` (`native` = the first). Other kinds accept only `native`. |
| `imgpath` | path | none | Image directory, e.g. `data/nih_full/images`. Required for `xrv_dataset` and `image_folder`. |
| `csvpath` | path | none | `xrv_dataset`. Annotation file of the dataset; required for `chexpert`. |
| `label_sources` | list of blocks | empty | `image_folder` only (at least one). Label files attached to the images, see [below](#patient_generatorlabel_sources). |
| `image_ext` | text | `.png` | `image_folder` only. Extension of the image files. |
| `sample_mode` | `sequential`, `random`, `permutation` | `sequential` | How patients are drawn from the pool: in order (wrapping around), at random with replacement, or every image at most once in a random order (`inference_cache` only). |
| `split` | `all`, `even`, `odd` | `all` | `inference_cache` only. Keep the even- or odd-indexed cache rows, so that a calibration fitted on one half (`fit_calibration.py --split even`) is studied on the other (`split: odd`). |
| `views` | list of text | `[PA]` | `nih` only. Projections kept; add `AP` for portable frontal images. |
| `unique_patients` | true, false | `true` | Keep one image per patient, so that the samples are independent. |
| `resize` | integer > 0 | `224` | Side of the square the images are resized to. A property of the model, not of the dataset: 224 for the DenseNet-121 ensemble, 896 for MedGemma. |

### `patient_generator.arrivals`

<!-- fields: ArrivalConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `poisson`, `deterministic`, `nhpp` | `poisson` | Homogeneous Poisson process, equally spaced arrivals, or a non-homogeneous Poisson process (NHPP) with a piecewise-constant rate, sampled by thinning (Lewis & Shedler 1979). |
| `rates` | list of numbers | empty | `nhpp` only (required). One rate per slot of the cycle; 168 hourly rates over a period of 168 give a day-of-week profile (De Santis et al. 2022). |
| `period` | number | none | `nhpp` only. Length of the cycle after which `rates` repeats. |

Start with the homogeneous process and move to `nhpp` only when the data call for it: much of
the burstiness reported in hospital arrivals is an artefact of pooling hours across weeks of
different intensity (Kim & Whitt 2014).

### `patient_generator.label_sources`

<!-- fields: LabelSourceConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | required | Name by which `gt_source` selects this source. |
| `kind` | `csv` | `csv` | Format of the label file. |
| `path` | path | required | The label file, one row per image. |
| `space` | text | `xrv` | Label space of the columns of the file, such as `vindr` or `xrv`. |
| `map` | registered name, inline mapping, or none | none, identity by name | How the source space aligns to the model schema. `vindr14->xrv` is the registered map used for VinDr-CXR. An inline mapping is `{source_label: target_label}`, and a target may be a list. |
| `id_column` | text | `example_id` | Column holding the image identifier. |

## `ai`

`kind` selects the AI reader:

- `torchxrv` and `medgemma` run the real models and, through the cache matrix, replay them
  (see [inference caches](usage.md#inference-caches));
- `cached` replays one cache cell directly;
- `fixed_accuracy` is a synthetic signal-detection reader whose skill is a dial;
- `random_mock` draws confidences and uncertainties from Beta distributions, for tests.

`service_time` is the *simulated* cost of one inference. It changes the queue, never the
predictions.

<!-- fields: AIModelConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `random_mock`, `torchxrv`, `cached`, `medgemma`, `fixed_accuracy` | `random_mock` | Which AI reader is built. |
| `name` | text | none | Label of this block in the study tables when a whole `ai` block is a factor level. No effect on the run. |
| `service_time` | number | `0.05` | Simulated time of one AI read. |
| `d_prime` | number or per label | none | `fixed_accuracy`. Sensitivity index d′ of the synthetic reader; when set it replaces the operating point below. |
| `criterion` | number or per label | none | `fixed_accuracy`. Decision criterion. |
| `sensitivity` | number or per label | `0.9` | `fixed_accuracy`. Operating point. |
| `specificity` | number or per label | `0.9` | `fixed_accuracy`. Operating point. |
| `prior` | number or per label | `0.5` | `fixed_accuracy`. Base rate its calibrated belief assumes. |
| `cache_stem` | path | none | `cached` only (required). Stem of the cache cell replayed. |
| `cache_mode` | `off`, `require`, `fill`, `report` | `require` | `torchxrv`, `medgemma`. `require` replays the cache cells and fails if one is missing; `fill` builds the missing cells; `report` prints the plan and stops; `off` runs the live model. |
| `cache_root` | path | `data/cache` | Root of the cache matrix. |
| `alpha_conf`, `beta_conf` | numbers | `2.0`, `2.0` | `random_mock`. Beta parameters of the confidence. |
| `alpha_unc`, `beta_unc` | numbers | `2.0`, `5.0` | `random_mock`. Beta parameters of the uncertainty. |
| `weights` | name or list of names | `densenet121-res224-all` | `torchxrv`. Pre-trained weights. A list builds a Deep Ensemble, one member per name. The ensemble of the paper uses `-chex`, `-mimic_ch`, `-pc`, `-rsna`; the `-nih` member is left out because it was trained on the NIH images. |
| `uq` | `entropy`, `ensemble` | `entropy` | Uncertainty method: the entropy of one probability, or the disagreement of the members. `ensemble` needs more than one member. |
| `ensemble_measure` | `mutual_information`, `predictive_entropy`, `mean_entropy`, `std` | `mutual_information` | Which disagreement measure `ensemble` exposes, see [uncertainty measures](architecture.md#uncertainty-measures). This is the number `uncertainty_desc` sorts on. |
| `decision_threshold` | number in [0, 1], or `prevalence_matched` | `0.5` | Cut of the AI's own call, used for its stand-alone errors and the reliance metrics (the reader is shown the probability, not this call). `prevalence_matched` reads from the calibration file, per finding, the cut at which the model calls as many positives as there are; it needs `calibration.path`. |
| `device` | text | `cpu` | Device of the live model. Does not change results. |
| `model_id` | text | `google/medgemma-1.5-4b-it` | `medgemma`. Model identifier. |
| `readout` | `independent_question`, `teacher_forced_logprob`, `generated_json` | `independent_question` | `medgemma`. How a probability is read out of a model that emits text, see [below](#medgemma-reading-a-probability-out-of-a-generative-model). |
| `prompt_set` | `single`, `triple` | `single` | `medgemma`. One prompt, or three paraphrases used as ensemble members. |
| `both_polarities` | true, false | `false` | `medgemma`. Ask each question in both polarities. |
| `dtype` | text | `auto` | `medgemma`. Weight precision of the in-process backend. |
| `backend` | `transformers`, `vllm` | `transformers` | `medgemma`. Load the weights in this process, or call a running vLLM server. |
| `endpoint` | URL | none | `vllm` (required). Base URL of the server, up to and including `/v1`. |
| `served_model_name` | text | none | `vllm` (required). The server's `--served-model-name`. |
| `concurrency` | integer | `8` | `vllm`. Requests in flight per image. |
| `top_logprobs` | integer | `20` | `vllm`. Log-probabilities requested per token. |
| `calibration` | block | omitted, raw probabilities | Post-hoc calibration, see [below](#aicalibration). |

Available DenseNet-121 weights in TorchXRayVision: `densenet121-res224-all`, `-nih`,
`-chex`, `-mimic_ch`, `-mimic_nb`, `-pc`, `-rsna`.

### `ai.calibration`

Calibration maps are monotone: they change the numbers and every decision taken at a fixed
threshold, never the ranking of the cases. Fit them with `scripts/fit_calibration.py` on
rows the study does not score (`patient_generator.split`).
[Calibration and fusion](calibration-and-fusion.md) explains the full chain.

<!-- fields: CalibrationConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `identity`, `temperature`, `platt`, `isotonic` | `identity` | Calibration map. The studies of the paper use per-finding `platt`. |
| `stage` | `output`, `members` | `output` | Calibrate the fused probability the reader is shown, or each member before fusion (then the uncertainty also describes calibrated numbers). |
| `path` | path | none | `platt`. Per-finding calibration file written by `fit_calibration.py`; when set it replaces `a` and `b`. |
| `temperature` | number or per label | `1.0` | `temperature`. Divisor of the logits (Guo et al. 2017). |
| `a`, `b` | numbers | `1.0`, `0.0` | `platt`. Slope and intercept of `sigmoid(a · logit(p) + b)`. |
| `knots_x`, `knots_y` | lists of numbers | empty, empty | `isotonic` (required). Fitted knots, `knots_x` ascending in [0, 1]. |

### MedGemma: reading a probability out of a generative model

MedGemma emits *text*, not one probability per finding, so a **readout** elicits one. The
choice of readout is a modelling choice, not a parsing detail:

| readout | how the number is obtained | use |
|---|---|---|
| `independent_question` (default) | one yes/no question per finding, each branching off a *shared* reading of the image, so no answer can see another; P(yes) is read from the token log-probabilities | use this |
| `teacher_forced_logprob` | one forced yes/no checklist covering every finding | superseded, kept as a diagnostic only |
| `generated_json` | parses the confidences the model writes down | a number the model found plausible to write, *not* a probability it holds; kept as a deliberate contrast |

**Why the checklist is superseded.** A checklist is one sequence, so the answer at position
*k* is conditioned on the answers forced before it. Measured during development on real NIH
ChestX-ray14 images, that dependence dominated the image: forcing all-yes versus all-no
moved the mean answer from 0.875 to 0.099 (mean absolute gap 0.78 on a 0–1 scale), the gap
correlated +0.48 with position, and position 0, the only slot with nothing before it, came
out identical under both fillers.

`independent_question` removes the dependence by construction. Encoding the image
(896×896 into 256 tokens) and the instruction is nearly all the compute and is identical for
every question, so the in-process backend runs that prefix once and rewinds its key-value
cache between questions. Only this readout works over the vLLM backend.

**The weights are gated.** Accept the Health AI Developer Foundations terms on Hugging Face
and log in (`huggingface-cli login`) first. One forward pass of the 4B model takes tens of
seconds on a CPU, so build the cache cells once and replay them; the build configurations
are `configs/cache/*_medgemma_*.yaml`. A MedGemma cell is an ordinary cache cell, so every
study and metric works on it unchanged.

## `persona`

The doctor **is** a persona: capability × visibility × trust. `persona` is either the name
of a library preset or a block. A preset is a signal-detection reader (`sdt`) with the same
sensitivity and specificity on every finding, blind to the image, and a linear pool with the
listed weight of advice when it sees the AI.

| preset | sensitivity | specificity | sees AI | weight of the AI |
|---|---|---|---|---|
| `solo_senior` | 0.95 | 0.95 | no | |
| `solo_junior` | 0.85 | 0.88 | no | |
| `senior_skeptical` | 0.95 | 0.95 | yes | 0.20 |
| `senior_trusting` | 0.95 | 0.95 | yes | 0.45 |
| `junior_skeptical` | 0.85 | 0.88 | yes | 0.40 |
| `junior_trusting` | 0.85 | 0.88 | yes | 0.65 |

What is anchored and what is not:

- the default weight of advice, 0.39, is the pooled mean of the advice-taking
  meta-analysis of Bailey et al. (2023), measured on generic numeric estimates, not on
  clinical reading;
- only the ordering senior < junior is anchored in the clinical literature (Gaube et al.
  2021), and on a related construct (susceptibility to wrong advice), not on a measured
  weight of advice;
- the four magnitudes and the sensitivity/specificity values are modelling choices.

Because the preset readers are blind to the image, their errors are independent of the AI's,
so the benefit of the advice they show is an upper bound. See
[`src/hai_sim/personas.py`](../src/hai_sim/personas.py) for the sources.

<!-- fields: PersonaConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | none | Label of the persona in the outputs. |
| `capability` | block | all defaults below | The reader's own, unaided diagnostic ability. |
| `sees_ai` | true, false | `false` | Visibility: whether the reader is shown the AI's probabilities. `false` is the solo reader. |
| `trust` | block | none, a `linear_pool` with defaults | Integration policy, used only when `sees_ai` is true. |

### `persona.capability`

<!-- fields: CapabilityConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `fixed_accuracy`, `random_mock`, `sdt` | `fixed_accuracy` | `sdt` is the signal-detection observer whose belief varies case by case (the reader of every library preset); `fixed_accuracy` keeps or flips each true label with a fixed probability, so its confidence is a constant; `random_mock` draws from Beta distributions, for tests. |
| `service_time_mu` | number | `2.5` | Location of the log-normal reading time (median e^mu ≈ 12.2, mean ≈ 13.2 with the default spread), used when `service_time` is omitted. The reader is usually the bottleneck, so raising it lengthens the queue. |
| `service_time_sigma` | number | `0.4` | Spread of the log-normal reading time, used when `service_time` is omitted. |
| `sensitivity` | number or per label | `0.9` | `fixed_accuracy`, `sdt`. Operating point: lower means more missed findings. |
| `specificity` | number or per label | `0.9` | `fixed_accuracy`, `sdt`. Operating point: lower means more false alarms. |
| `d_prime` | number or per label | none | `sdt`. Sensitivity index d′; when set it replaces the operating point. |
| `criterion` | number or per label | none | `sdt`. Decision criterion. |
| `prior` | number or per label | `0.5` | `sdt`. Base rate the belief is calibrated to. |
| `alpha_conf`, `beta_conf` | numbers | `5.0`, `2.0` | `random_mock`. Beta parameters of the confidence. |
| `alpha_unc`, `beta_unc` | numbers | `2.0`, `5.0` | `random_mock`. Beta parameters of the uncertainty. |
| `source` | `fixed`, `reader` | `fixed` | Where the operating point comes from. `reader` is reserved for a real reader's reads and raises. |
| `reader_id` | text | none | Reserved, with `source: reader`. |
| `difficulty_slope` | number ≥ 0 | `0.0` | Loss at difficulty 1: a share of accuracy for `fixed_accuracy`, a loss of d′ for `sdt`. At zero the reader is blind to difficulty, and its errors are independent of the AI's. |
| `fatigue_slope` | number ≥ 0 | `0.0` | The same, for the fatigue signal. At zero the reader never tires. |
| `floor` | number in [0, 1] | `0.5` | `fixed_accuracy`. Accuracy below which degradation stops, so a worn reader becomes a coin flip and never inverts. |
| `service_time` | block | omitted, log-normal with the two fields above | Reading-time model, see below. |

### `persona.capability.service_time`

<!-- fields: ServiceTimeConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `lognormal`, `constant` | `lognormal` | Shape of the reading time. |
| `mu` | number | `2.5` | `lognormal`. Location parameter. |
| `sigma` | number ≥ 0 | `0.4` | `lognormal`. Spread parameter. |
| `value` | number ≥ 0 | `1.0` | `constant`. The reading time. |
| `factor` | number ≥ 0 | `1.0` | Constant multiplier of the draw: below 1 the AI speeds the reader up, above 1 it costs verification time. The literature reports both, so sweep it. |
| `fatigue_slope` | number ≥ 0 | `0.0` | Stretch of the read in proportion to the fatigue signal. Applied before `factor`. |

### `persona.trust`

<!-- fields: TrustConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `linear_pool`, `confidence_gated`, `log_odds_fusion` | `linear_pool` | Integration rule. `linear_pool` averages the two probabilities; `confidence_gated` listens to the AI only where it is confident; `log_odds_fusion` adds the two opinions as evidence, after moving the AI's probability to the reader's base rate. See [calibration and fusion](calibration-and-fusion.md). |
| `decision_threshold` | number in (0, 1) | `0.5` | Fused belief at or above which the team calls a finding present. |
| `weight_ai` | number in [0, 1] | `0.39` | The weight of advice: the AI's share of the pool, under every rule. |
| `confidence_threshold` | number in [0, 1] | `0.7` | `confidence_gated`. Minimum AI confidence, max(p, 1 − p), below which the advice is ignored. |
| `beta` | number ≥ 0 | `1.0` | `log_odds_fusion`. Sharpness of the down-weighting of the AI by its own uncertainty, exp(−beta · H). The studies of the paper set it to 0. |
| `prior` | number in (0, 1) | `0.5` | `log_odds_fusion`. The reader's base rate: the prior removed from its log-odds and the one the team posterior is rebuilt on. |

## `fusion`

Changes the integration rule of any persona, preset or block, and leaves its other fields
untouched. This is how one study runs the same personas under several rules
(`configs/studies/team_effect.yaml`). It cannot change the weight of advice; that belongs to
the persona.

<!-- fields: FusionConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `linear_pool`, `confidence_gated`, `log_odds_fusion` | none, the persona's rule | Integration rule imposed on the persona. |
| `beta` | number ≥ 0 | none | Sharpness of the uncertainty discount of the logarithmic pool. |
| `decision_threshold` | number in (0, 1) | none | Team decision threshold. |

## `sort`

<!-- fields: SortConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `random`, `uncertainty_desc`, `probability_desc` | required | Order in which the reader takes the waiting patients: a uniform random priority (the baseline), most uncertain first, or most probably abnormal first. |
| `aggregator` | `mean`, `max`, `weighted` | `mean` | How the per-finding values are reduced to one priority. `mean` dilutes one critical finding among many confident ones; `max` escalates a patient with any one high value; `weighted` weights by clinical importance. Ignored by `random`. |
| `weights` | list of numbers | none | `weighted` only (required). One weight per finding. |

**The mean wait is the same under every order.** Reordering a queue changes *who* waits, not
how long patients wait on average. Order effects exist only under saturation, so measure
them with metrics split by ground truth or severity (`kpi_groups: [{key: truth}]`), at a load
that saturates the reader. To sort on probabilities of different findings with `max`, put
them on one scale first with a calibration.

## `severity`

A per-patient severity signal. It is **recorded in the telemetry whether or not anything
uses it**.

<!-- fields: SeverityConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `constant`, `activation`, `image_quality`, `embedding` | `constant` | Severity model. `activation` reduces criticality × activation over the findings. |
| `value` | number | `0.0` | `constant`. The severity of every patient. |
| `source` | `ai_probs`, `ai_binary`, `ai_uncertainty`, `gt` | `ai_probs` | `activation`. The activation vector. |
| `criticality` | list of numbers | none | `activation`. Clinical weight of each finding, one per label. |
| `aggregator` | `mean`, `max`, `sum` | `max` | `activation`. Reduction of the weighted vector. |
| `mode` | `scalar`, `vector` | `scalar` | `activation`. One severity per patient, or one per finding. |
| `quality_metric` | `laplacian_var`, `intensity_std` | `laplacian_var` | `image_quality`. Sharpness or contrast proxy. |
| `embedding_reduce` | `l2_norm`, `mean` | `l2_norm` | `embedding`. Reduction of the feature vector of the live model (needs `ai.cache_mode: off`). |

## `difficulty`

The per-case difficulty δ in [0, 1]. It is computed after the AI has read the image and
handed to the reader as `sig.difficulty`. It is the only route by which a simulated reader's
errors can correlate with the AI's, and it acts only on a reader with
`capability.difficulty_slope > 0`. No study in `configs/studies/` uses it.

<!-- fields: DifficultyConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `constant`, `prior`, `uncertainty`, `image_quality`, `table` | `prior` | `constant` is the null model; `prior` draws a Beta value per image, unrelated to the image (the zero-correlation control); `uncertainty` reads the AI's own uncertainty, so the coupling is induced from the model; `image_quality` reads blur or contrast; `table` reads a file. |
| `value` | number in [0, 1] | `0.5` | `constant`. The difficulty of every case. |
| `a`, `b` | numbers > 0 | `2.0`, `5.0` | `prior`. Parameters of the Beta distribution, a modelling choice. |
| `source` | `uncertainty` or a measure name | `uncertainty` | `uncertainty`. The run's selected uncertainty, or a named ensemble measure such as `mutual_information` or `mean_entropy` (needs `ai.uq: ensemble`). |
| `aggregator` | `mean`, `max`, `none` | `mean` | `uncertainty`. Reduction over findings. |
| `scale` | number | none, the theoretical maximum of the measure | `uncertainty`. Value mapped to difficulty 1. |
| `quality_metric` | `laplacian_var`, `intensity_std` | `laplacian_var` | `image_quality`. Sharpness or contrast proxy. |
| `reference` | number | none, the running median of the quality seen so far | `image_quality`. Quality value at which difficulty is 0.5. |
| `path` | path | none | `table` (required). CSV file with one row per image, e.g. the inter-reader disagreement written by `scripts/build_vindr_labels.py` to `data/vindr/difficulty.csv`. |
| `id_column` | text | `example_id` | `table`. Column holding the image identifier. |
| `value_column` | text | `difficulty` | `table`. Column holding the difficulty. |
| `default` | number in [0, 1] | `0.5` | `table`. Difficulty of an image missing from the file. |
| `strict` | true, false | `false` | `table`. Fail on a missing image instead of using `default`. |

## `fatigue`

How worn the reader is when it picks up each patient, handed to the reader as `sig.fatigue`.
Radiology has no analytic fatigue curve, so the shape is a sensitivity axis. Time has no
unit: the defaults assume one time unit is one hour. Fatigue acts only on a reader with
`capability.fatigue_slope > 0` or a `service_time.fatigue_slope > 0`. No study in
`configs/studies/` uses it.

<!-- fields: FatigueConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `none`, `linear`, `vigilance`, `circadian`, `composite` | `none` | Shape of the fatigue curve. |
| `basis` | `cases`, `time_on_shift`, `busy_time`, `utilisation` | `cases` | Quantity the curve is a function of. |
| `per_unit` | number ≥ 0 | `0.01` | `linear`. Fatigue gained per unit of the basis. |
| `cap` | number in [0, 1] | `1.0` | `linear`. Ceiling of the curve. |
| `asymptote` | number in [0, 1] | `1.0` | `vigilance`. Limit of asymptote · (1 − e^(−x / scale)). |
| `scale` | number > 0 | `40.0` | `vigilance`. Time constant of the exponential. |
| `amplitude` | number ≥ 0 | `0.3` | `circadian`. Amplitude of mean + amplitude · cos(2π (clock − acrophase) / period). |
| `mean` | number | `0.3` | `circadian`. Mean level. |
| `period` | number > 0 | `24.0` | `circadian`. Length of the cycle in time units. |
| `acrophase` | number | `5.0` | `circadian`. Clock time of the peak. |
| `parts` | list of `linear`, `vigilance`, `circadian` | empty | `composite` (required). Curves combined. |
| `combine` | `max`, `sum`, `mean` | `max` | `composite`. How the parts are combined. |

## `label_view`

Projects the model's labels onto the study label set; the AI, the reader, the ground truth
and the metrics all follow it.

<!-- fields: LabelViewConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `mode` | `full`, `subset`, `binary_abnormal` | `full` | No projection, keep only `targets`, or collapse them into one "abnormal" label. |
| `targets` | list of label names | none; required for `subset`, every label for `binary_abnormal` | Findings kept or collapsed. One target under `subset` gives a single-finding study. |
| `combine` | `noisy_or`, `max` | `noisy_or` | `binary_abnormal`. How the probabilities of the targets are combined. |

The quickstart and the RQ3 and RQ4 studies of the paper keep the ten findings annotated on
both NIH ChestX-ray14 and VinDr-CXR: Cardiomegaly, Effusion, Consolidation, Atelectasis,
Infiltration, Nodule, Mass, Pleural_Thickening, Pneumothorax, Fibrosis. `queue_load.yaml`
also keeps Lung Opacity, which only VinDr-CXR annotates. Without a view, findings the dataset
does not annotate are −1 and skipped by every metric.

## `kpis`

```yaml
kpis: [system, actors, team, joint_table]   # the default: all four
```

Each entry is either a **registered name** or a reference to your own bundle, so a new
metric never requires a change to the package:

```yaml
kpis:
  - system
  - configs/examples/custom_metrics.py:CostBundle   # path/to/file.py:Attribute
  - my_package.my_metrics:CostBundle                # or an importable module
```

A bundle needs only a `name` and a `compute(records)` method, and must be constructible
with no arguments. The report is keyed by the bundle's own `name`, so a custom bundle is
reachable by the study layer's dotted paths (`cost.hybrid_total`) exactly like a built-in
one. The built-in bundles are defined in [outputs](outputs.md#kpi-definitions); writing your
own is described in [architecture](architecture.md#a-new-metric).

> Resolving a reference imports the target, which runs its module-level code. A scenario
> file is therefore as trusted as a script.

## `kpi_groups`

Every selected bundle is computed once more **per subgroup**. An entry of `kpi_groups`, and
of the `by` list of a study, has these fields:

<!-- fields: KPIGroupConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `key` | `label`, `truth`, or a record field | required | The axis. `label` gives one subgroup per finding (`label=Effusion`), turning a macro-averaged bundle into a per-finding one. `truth` gives, for every finding, the patients who have it and those who do not (`truth=Effusion:positive`, `truth=Effusion:negative`): this is how a time metric is asked "do the sick wait less?". Any other key names a record field (`severity`, `priority_key`, `ts_arrival`) or an observation (`ctx.sex`, `load.in_system_at_arrival`, `ai.*`, `doc.*`, `sig.*`). |
| `edges` | list of numbers | empty | Bin edges for a continuous key: `[1, 5]` gives `<1`, `1-5` and `>=5`. Leave empty for a categorical key; `label` and `truth` take none. |
| `include_missing` | true, false | `false` | Keep the records that lack the key instead of dropping them, so the denominator never changes silently. |

```yaml
kpi_groups:
  - key: severity
    edges: [1.0, 2.5]                    # severity <1, 1-2.5, >=2.5
  - key: load.in_system_at_arrival
    edges: [5.0, 20.0]                   # quiet / busy / saturated at arrival
  - key: label                           # one subgroup per finding
```

Several entries **cross**: `severity × label` yields one subgroup per (band, finding). The
console prints a preview with one column per axis; the full grid goes to the run JSON under
`kpis_by_group`. A grouping only splits the records and reruns the unchanged bundle, so
*any* bundle can be conditioned, including your own.

## Common errors

These raise an error when the scenario is loaded, built, or at the first read:

- `ai.kind: torchxrv` needs `label_schema: xrv`; `ai.kind: cached` needs `ai.cache_stem`.
- `ai.backend: vllm` needs `ai.endpoint` and `ai.served_model_name`, and supports only
  `readout: independent_question`.
- `ai.decision_threshold: prevalence_matched` needs `ai.calibration.path`.
- For `torchxrv` and `medgemma`, a `cache_mode` other than `off` needs a dataset-backed
  patient generator (`xrv_dataset` or `image_folder`), and feature-space severity
  (`severity.kind: embedding`) needs `cache_mode: off`.
- `patient_generator.kind: xrv_dataset` and `image_folder` need `imgpath`; `dataset: chexpert`
  needs `csvpath`; `image_folder` needs at least one `label_sources` entry; `inference_cache`
  needs `cache_stem`; `sample_mode: permutation` works only with `inference_cache`.
- `sort.aggregator: weighted` needs `sort.weights`, one per finding; `severity.criticality`
  needs one weight per label.
- `label_view.mode: subset` needs `targets`, and every target must be a label of the schema.
- `arrivals.kind: nhpp` needs `rates`; `fatigue.kind: composite` needs `parts`;
  `difficulty.kind: table` needs `path`; `calibration.kind: isotonic` needs knots of equal
  length.
- A named `difficulty.source` (e.g. `mutual_information`) needs `ai.uq: ensemble`.
- `persona.capability.source: reader` raises, and so does an unknown persona name.
- `kpi_groups` keys `label` and `truth` take no `edges`.
- Unknown keys, out-of-range values, and `arrival_rate <= 0` raise.
- A VinDr-CXR cache cell needs `data/vindr/labels.csv`, built once by
  `uv run python scripts/build_vindr_labels.py` (see [`data/README.md`](../data/README.md)).

## Recipes

Minimal diffs on a base scenario.

**Worklist order**, one field:

```yaml
sort: {kind: random}                                  # A, the baseline
sort: {kind: probability_desc, aggregator: max}       # B, the most suspicious first
```

Expect the same mean wait and a different wait for the patients who have a finding, and only
when the reader is saturated.

**With and without the AI**, same reader, visibility flipped:

```yaml
persona: solo_senior
persona: senior_trusting
```

**Two rules for the same opinions:**

```yaml
fusion: {kind: linear_pool}
fusion: {kind: log_odds_fusion, beta: 0.0}
```

**Uncertainty from the members of the ensemble:**

```yaml
ai:
  kind: torchxrv
  weights: [densenet121-res224-chex, densenet121-res224-mimic_ch,
            densenet121-res224-pc, densenet121-res224-rsna]
  uq: ensemble
  ensemble_measure: mean_entropy
```

**Replay a cache cell instead of running the model** (no images, no `models` extra):

```yaml
patient_generator:
  kind: inference_cache
  cache_stem: data/cache/composed/vindr__ens4
  arrival_rate: 0.075
  n_patients: 7500
  sample_mode: permutation
  split: odd
ai:
  kind: cached
  cache_stem: data/cache/composed/vindr__ens4
  uq: ensemble
  ensemble_measure: mean_entropy
  calibration: {kind: platt, path: data/cache/calibration/vindr__ens4.json}
```
