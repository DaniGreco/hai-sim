# hai_sim — a playground for testing an AI placed *in series* with a doctor

A simulated reading room where an AI model and a simulated radiologist read the same
chest X-rays (CXR), **one after the other**, and everything that happens is measured.
Change one piece — the model, the doctor's skill, how much the doctor trusts the model,
the order the queue is worked through — rerun, and see what moved. Master thesis of
Daniele Greco (UniBg = Università degli Studi di Bergamo).

**"In series" is the whole point.** The AI reads first and does two separable jobs: its
uncertainty **reorders the queue**, and its opinion **reaches the doctor as a hint**. The
doctor always forms its own opinion *before* the hint is folded in, so every patient
leaves behind three verdicts — model alone, doctor alone, and the team — which is what
makes it possible to ask not just "was the team right?" but "did the doctor take the
advice when it should have?".

> **This is the technical reference**: how to install it, how to run it, every knob,
> every KPI (key performance indicator), and everything currently implemented.
>
> **Every acronym and term of art is defined once in [`GLOSSARY.md`](GLOSSARY.md)** —
> start there if a word here is unfamiliar.
>
> **The *why* — design, experiments and results — lives in the thesis**: D. Greco,
> *A Framework to Study the Cooperation Between Human and Machine Learning Algorithms
> in Diagnostic Decisions*, Master's thesis, University of Bergamo, 2026. A "thesis
> §x.y" in the code or in this README points to a section of it.

**Founding constraint: no training from scratch.** Pre-trained models, inference only.
The CXR dataset is the simulation's *patient generator*, not a training set.

---

## Contents

- [**GLOSSARY.md**](GLOSSARY.md) — every acronym and term, defined once
- [Install](#install) · [Quickstart](#quickstart) · [Pipeline](#pipeline) · [CLI](#cli)
- [Data & caches](#data--caches)
- [Scenario reference](#scenario-reference) — the experiment control panel
- [Study reference](#study-reference) — factor grids, paired contrasts
- [Output](#output) · [KPI definitions](#kpi-definitions)
- [What is implemented](#what-is-implemented) — protocols, implementations, scripts
- [Determinism](#determinism) · [Tests](#tests) · [Future work](#future-work)

---

## Install

```powershell
# from C:\Projects\tesi\hai_sim
uv venv
uv pip install -e ".[dev]"
```

For the real models (`[models]` extra — CPU torch keeps it light):

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[models]"
```

On Windows set `PYTHONUTF8=1`, otherwise the TorchXRayVision weight-download progress bar
crashes the console (`cp1252` cannot render its block characters).

## Quickstart

```powershell
uv run pytest

# the four experiments of the thesis (they replay the caches in data/cache, see below)
uv run hai-sim capture configs/studies/capture.yaml                 # E1 + E2
uv run hai-sim study configs/studies/queue_load.yaml --workers 8    # E3
uv run hai-sim study configs/studies/team_effect.yaml --workers 8   # E4

# the figures of E3 and E4, from the tables the studies wrote
uv run hai-sim plot queue_load out/study_queue_load_<stamp>_summary.csv
uv run hai-sim plot team_rules out/study_team_effect_<stamp>_long.csv
uv run hai-sim plot actor_errors out/study_team_effect_<stamp>_long.csv --ai medgemma-27b
```

Always run from the repo root so the repo-relative paths in the scenarios resolve.

---

## Pipeline

```
PatientGenerator → input_queue (Store, first-in-first-out)
                 → ModelStation (× n_ai_workers)          ← the AI reads FIRST
                 → triaged_store (PriorityStore, key = SortStrategy)   ← triage happens here
                 → DoctorStation (× n_doctor_workers)     ← the doctor reads SECOND,
                 → MetricsCollector                          seeing the AI's opinion
```

Every actor — AI model and doctor alike — implements the same `DiagnosticActor`
protocol, `diagnose(images, *, context, ai_hint) → ModelOutput`, so any implementation is
swappable from the scenario YAML. The doctor may consult the AI's `ModelOutput` as a hint.

Why the AI sits *upstream of the queue* rather than beside it: having already read
everyone, it can say which cases it found hardest **before** the doctor picks the next
patient. That is what makes triage by uncertainty possible at all — and it only changes
outcomes when patients are actually waiting, so any order study has to be run at a load
that saturates the doctor.

Two design seams worth knowing:

- **Uncertainty stays per-label** all the way through `UQMethod`. Only `SortStrategy`
  collapses it to a scalar priority key. Because the full per-label vector is written to
  the CSV, you can **re-triage offline with any aggregator without re-running inference**.
- **The doctor's own opinion stays AI-independent.** `AssistedDoctor` stashes the solo
  counterfactual in `extra["doctor_solo_labels"]`, which is what makes the team/reliance
  KPIs computable by construction.

---

## CLI

```
hai-sim run <scenario.yaml>   [--out-dir out]
hai-sim study <study.yaml>    [--out-dir out] [--workers N]   # N processes for the distinct scenarios
hai-sim capture <capture.yaml> [--out-dir out/capture]
hai-sim plot <kind> <study.csv> [--out stem] [--persona P] [--fusion F] [--ai A] [--mean-read T]
hai-sim validate <scenario.yaml> | --from-cache <stem>
```

**`run`** — one scenario → per-patient telemetry CSV + summary JSON + a printed KPI table.

**`study`** — expand a factor grid → one run per cell per seed → a tidy KPI table
(CSV + JSON).

**`capture`** — the offline capture study of experiments E1 and E2: read the images of a
cache in the order a model proposes and count the positive cases (or the model's own
errors) met along the way, per pathology. No simulation; the cache cells only.

**`plot`** — the figure of a simulation study from the CSV it wrote: `queue_load` (E3,
reads the `*_summary.csv`), `team_effect`, `team_rules` and `actor_errors` (E4, read the
`*_long.csv`). Figures go to `out/figures/<kind>.{png,pdf}` unless `--out` says otherwise.

**`validate`** — score a model's labels and Grad-CAM heatmaps against expert ground truth
(ChestX-Det masks). Two modes: a validation scenario YAML (runs inference), or
`--from-cache <stem>` (offline, from a prebuilt localization cache).

| flag | default | meaning |
|---|---|---|
| `--limit N` | none | cap images (direct mode) |
| `--checkpoint-every N` | 25 | persist records every N images |
| `--threshold F` | 0.5 | decision threshold |
| `--aggregation mean\|max` | mean | how ensemble member CAMs are combined |
| `--split train\|test` | none | restrict to a split |
| `--pointing-tolerance PX` | 16 | pointing-game radius (`--from-cache`) |

---

## Data & caches

Everything under `data/` is gitignored. Large datasets live there so scenarios can use
portable, repo-relative paths instead of machine-specific absolute ones.

```
data/
├── nih_full/                        # NIH ChestX-ray14: one image per patient, 30,805 images
│   ├── images/                      # not kept, see "Getting the images back"
│   ├── Data_Entry_2017.csv
│   └── provenance/                  # id list, checksums, the script that built the pool
├── vindr/                           # VinDr-CXR: the 15,000 training images
│   ├── images                       # link to the kagglehub download, not kept
│   ├── labels.csv, difficulty.csv, readers.csv
│   └── provenance/                  # id list, checksums, the Kaggle annotation files
└── cache/                           # inference caches (build once, replay forever)
    ├── <dataset>__<cell>.{npz,json} # one model on one dataset: chex, mimic_ch, pc, rsna,
    │                                #   medgemma-1 (MedGemma 1.5 4B), medgemma-27b
    ├── composed/<dataset>__ens4.*   # the four DenseNet members in one cache (E3, E4)
    ├── calibration/<dataset>__*.json  # per-pathology Platt maps, fitted on the even rows
    └── provenance/                  # how the MedGemma cells were built (server, logs)
```

**Model weights are not here.** TorchXRayVision manages them; they auto-download to
`~/.torchxrayvision/models_data/` on first use (~80 MB each) and are shared across projects.

**Caches are the reason real studies are fast.** Each cache cell holds the probabilities
of one model over one dataset, keyed by image id. `scripts/materialize_cache.py` builds
the cells of a scenario (`--mode fill`, on a GPU machine) or checks that they are complete
(`--mode report`); afterwards `ai.kind: cached` + `patient_generator.kind: inference_cache`
replay them, so a study over thousands of real images finishes in seconds instead of
hours. A cell is resumable and guarded by a model fingerprint, so a stale cache cannot be
silently replayed against different weights. The scenarios that built the cells of the
thesis are `configs/examples/nih_full_ensemble.yaml`, `v4_vindr_ensemble.yaml` and the
four `v5_medgemma_*.yaml`.

`scripts/build_vindr_labels.py` turns the raw VinDr-CXR annotations into the label files
above, and `scripts/fit_calibration.py` fits the calibration files.

### Getting the images back

The experiments of the thesis read only `data/cache`, so the images were deleted once the
last cell was built. They are needed again only to build new cells or to run a scenario
that reads the pixels. The cells hold exactly the image ids listed in the two
`provenance/` folders.

| dataset | source | how | check |
|---|---|---|---|
| NIH ChestX-ray14, Wang et al., CVPR 2017, [doi:10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369) | Kaggle dataset `nih-chest-xrays/data`, version 3 (all 112,120 images, ~42 GB) | `python data/nih_full/provenance/build_nih_full_pool.py --ids data/nih_full/provenance/nih_pool_ids.txt --out nih_full_pool` downloads it with kagglehub and copies the 30,805 images of the pool; move `nih_full_pool/nih_full/images` to `data/nih_full/images` | from `data/`: `sha256sum -c nih_full/provenance/SHA256SUMS.txt` |
| VinDr-CXR, Nguyen et al., Scientific Data 2022, [doi:10.1038/s41597-022-01498-w](https://doi.org/10.1038/s41597-022-01498-w) | Kaggle dataset `awsaf49/vinbigdata-1024-image-dataset`, version 1 (the training images as 1024 px PNG) | `kagglehub.dataset_download("awsaf49/vinbigdata-1024-image-dataset")`; `data/vindr/images` points to its `vinbigdata/train` folder | from the download's `vinbigdata/`: `sha256sum -c <repo>/data/vindr/provenance/SHA256SUMS.txt` |

## Scenario reference

A scenario is a single YAML file validated by `pydantic`
(`src/hai_sim/config/scenario.py`) with `extra = forbid`: **any unknown key raises**, so
typos fail loudly. In every table a field marked *required* has no default, a field marked
*omitted* is optional and the default column says what happens when it is absent, and
*number or per label* means one number, or one number per finding of the active label set.
Fields that apply to one `kind` only say so; a field of another kind is accepted and ignored.

This file is the experiment control panel. The golden rule for clean comparisons:
**change ONE block at a time, keep `seed` fixed.**

```yaml
name: my_scenario
seed: 42
label_schema: xrv
patient_generator: { ... }   # required — which patients arrive, and when
ai: { ... }                  # required — the AI reader
persona: senior_trusting     # required — a library name, or a capability/visibility/trust block
fusion: { ... }              # optional — override the persona's integration rule
sort: { ... }                # required — the order of the worklist
severity: { ... }            # optional signals, computed between the two stations
difficulty: { ... }
fatigue: { ... }
label_view: { ... }          # optional — project the labels (subset / binary abnormal)
kpis: [system, actors, team, joint_table]
kpi_groups: [ ... ]          # optional — every metric also per subgroup
n_ai_workers: 1
n_doctor_workers: 1
```

### Quick map — which knob answers which question

| Experimental question | Block / field to vary | Keep fixed |
|---|---|---|
| Does the AI help the reader? | `persona.sees_ai`, or a persona with and without the AI | seed, patients |
| Does the order of the worklist matter? | `sort.kind`, `sort.aggregator` | seed, ai, persona, patients |
| How does the kind of reader change the team? | `persona` | seed, ai, patients |
| Which rule should combine the two opinions? | `fusion.kind` | seed, ai, persona |
| Which uncertainty measure? | `ai.uq`, `ai.ensemble_measure`, `ai.weights` | seed, everything else |
| Operating point of the AI | `ai.decision_threshold`, `ai.calibration` | seed, everything else |
| Load / saturation | `patient_generator.arrival_rate`, `patient_generator.arrivals`, `n_doctor_workers`, reading time | seed, ai, doctor quality |

### Top-level fields

<!-- fields: ScenarioConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | `unnamed` | Name of the run, written in the output file names. |
| `seed` | integer | `0` | Master seed. Five independent substreams (patients, AI, doctor, worklist, arrivals) are spawned from it, so changing one module does not move the others. |
| `label_schema` | `default_v0`, `single`, `chexpert`, `xrv` | `default_v0` | The finding set every per-patient vector is indexed by. `xrv` is the 18-label TorchXRayVision set used by every real model (`torchxrv` raises without it), `chexpert` the 14-finding convention, `single` one label, `default_v0` five unnamed labels for tests. |
| `label_view` | block | omitted, no projection | Projection to the study label set, see [`label_view`](#label_view). |
| `patient_generator` | block | required | See [`patient_generator`](#patient_generator). |
| `ai` | block | required | See [`ai`](#ai). |
| `persona` | library name or block | required | See [`persona`](#persona). |
| `fusion` | block | omitted, the persona's own rule | See [`fusion`](#fusion). |
| `sort` | block | required | See [`sort`](#sort). |
| `severity` | block | omitted, not computed | See [`severity`](#severity). |
| `difficulty` | block | omitted, not computed | See [`difficulty`](#difficulty). |
| `fatigue` | block | omitted, readers never tire | See [`fatigue`](#fatigue). |
| `kpis` | list of names | none, all four built-in bundles | Metric bundles to compute, see [`kpis`](#kpis). |
| `kpi_groups` | list of blocks | empty | Conditioning axes, see [`kpi_groups`](#kpi_groups). |
| `n_ai_workers` | integer ≥ 1 | `1` | Parallel servers of the AI station. |
| `n_doctor_workers` | integer ≥ 1 | `1` | Parallel servers of the doctor station. **More doctors → less saturation → shorter waits**, and the order of the worklist matters less. |

### `patient_generator`

`kind` selects the source of the patients. `poisson` and `mock_dataset` generate synthetic
cases, `xrv_dataset` reads a dataset through the TorchXRayVision loaders, `image_folder`
reads a folder of images with one or more label files (VinDr-CXR), and `inference_cache`
replays the images and labels of a cache cell without touching the pixels (every
experiment of the thesis).

<!-- fields: PatientGenConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `poisson`, `mock_dataset`, `xrv_dataset`, `inference_cache`, `image_folder` | `poisson` | Source of the patients. |
| `arrival_rate` | number > 0 | required | Mean arrivals per time unit. **Raise it → the queue saturates** and the order of service starts to matter. |
| `n_patients` | integer > 0 | required | Patients generated in the run. |
| `arrivals` | block | omitted, homogeneous Poisson at `arrival_rate` | Arrival process, see below. |
| `cache_stem` | path | none | `inference_cache` only. Stem of the cache cell replayed, without extension, e.g. `data/cache/composed/vindr__ens4`. |
| `prevalence` | number or per label | `0.2` | `poisson`, `mock_dataset`. Probability that each finding is present. |
| `image_shape` | list of integers | `[1, 1]` | `poisson`, `mock_dataset`. Shape of the placeholder image. |
| `embed_truth` | true, false | `false` | `poisson` only. Writes the true label vector into the image, so that a `fixed_accuracy` AI can read the case it scores without the engine handing ground truth to the AI stage. |
| `n_examples` | integer > 0 | `100` | `mock_dataset` only. Size of the synthetic pool the patients are drawn from. |
| `dataset` | `nih`, `rsna`, `chexpert` | `nih` | `xrv_dataset` only. Dataset loaded. |
| `dataset_id` | text | none, the dataset name or the image folder's parent name | Identifier of the dataset in the cache matrix, the first half of a cache stem `<dataset>__<model>`. |
| `gt_source` | text | `native` | Which labels are the scoring ground truth. For `image_folder`, the name of one of the `label_sources` (`native` = the first). Other kinds accept only `native`. |
| `imgpath` | path | none | Image directory, e.g. `data/nih_full/images`. Required for `xrv_dataset` and `image_folder`. |
| `csvpath` | path | none | `xrv_dataset`. Annotation file of the dataset; required for `chexpert`. |
| `label_sources` | list of blocks | empty | `image_folder` only (at least one). Label files attached to the images, see below. |
| `image_ext` | text | `.png` | `image_folder` only. Extension of the image files. |
| `sample_mode` | `sequential`, `random`, `permutation` | `sequential` | How patients are drawn from the pool: in order (wrapping around), at random with replacement, or (`inference_cache` only) every image once in a random order. |
| `split` | `all`, `even`, `odd` | `all` | `inference_cache` only. Keep the even- or odd-indexed cache rows, so that a calibration fitted on one half (`fit_calibration.py --split even`) is studied on the other (`split: odd`). |
| `views` | list of text | `[PA]` | `nih` only. Projections kept; add `AP` for portable frontals. |
| `unique_patients` | true, false | `true` | Keep one image per patient, so that the samples are independent. |
| `resize` | integer > 0 | `224` | Side of the square the images are resized to. **A property of the model, not of the dataset**: 224 for the DenseNet ensemble, 896 for MedGemma. |

#### `patient_generator.arrivals`

<!-- fields: ArrivalConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `poisson`, `deterministic`, `nhpp` | `poisson` | Homogeneous Poisson, equally spaced arrivals, or a non-homogeneous Poisson process (NHPP) with a piecewise-constant rate, sampled by thinning. |
| `rates` | list of numbers | empty | `nhpp` only (required). One rate per slot of the cycle; 168 hourly rates over a period of 168 give a weekly profile. |
| `period` | number | none | `nhpp` only. Length of the cycle after which `rates` repeats. |

#### `patient_generator.label_sources`

<!-- fields: LabelSourceConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | required | Name by which `gt_source` selects this source. |
| `kind` | `csv` | `csv` | Format of the label file. |
| `path` | path | required | The label file, one row per image. |
| `space` | text | `xrv` | Label space of the columns of the file, such as `vindr` or `xrv`. |
| `map` | registered name, inline mapping, or none | none, identity by name | How the source space aligns to the model schema. `vindr14->xrv` is the registered map used for VinDr-CXR. An inline mapping is `{source_label: target_label}`, and a target may be a list. |
| `id_column` | text | `example_id` | Column holding the image identifier. |

### `ai`

`kind` selects the AI reader. `torchxrv` and `medgemma` run the real models (and, through
the cache matrix, replay them), `cached` replays one cache cell, `fixed_accuracy` is a
synthetic signal-detection reader whose skill is a dial, and `random_mock` draws
confidences and uncertainties from Beta distributions for tests. `service_time` is the
*simulated* cost of one inference: it changes the queue, never the predictions.

<!-- fields: AIModelConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `random_mock`, `torchxrv`, `cached`, `medgemma`, `fixed_accuracy` | `random_mock` | Which AI reader is built. |
| `name` | text | none | Label of this block in the study tables when a whole `ai` block is a factor level. No effect on the run. |
| `service_time` | number | `0.05` | Simulation time of one AI read. |
| `d_prime` | number or per label | none | `fixed_accuracy`. Sensitivity index of the synthetic reader; when set it replaces the operating point below. |
| `criterion` | number or per label | none | `fixed_accuracy`. Decision criterion. |
| `sensitivity` | number or per label | `0.9` | `fixed_accuracy`. Operating point. |
| `specificity` | number or per label | `0.9` | `fixed_accuracy`. Operating point. |
| `prior` | number or per label | `0.5` | `fixed_accuracy`. Base rate its calibrated belief assumes. |
| `cache_stem` | path | none | `cached` only (required). Stem of the cache cell replayed. |
| `cache_mode` | `off`, `require`, `fill`, `report` | `require` | `torchxrv`, `medgemma`. `require` replays the cache cells and fails if one is missing, `fill` builds the missing cells, `report` prints the plan and stops, `off` runs the live model. |
| `cache_root` | path | `data/cache` | Root of the cache matrix. |
| `alpha_conf`, `beta_conf` | numbers | `2.0`, `2.0` | `random_mock`. Beta parameters of the confidence. |
| `alpha_unc`, `beta_unc` | numbers | `2.0`, `5.0` | `random_mock`. Beta parameters of the uncertainty. |
| `weights` | name or list of names | `densenet121-res224-all` | `torchxrv`. Pre-trained weights. A list builds a Deep Ensemble, one member per name; the thesis uses `-chex`, `-mimic_ch`, `-pc`, `-rsna` (the `-nih` member would leak on NIH). |
| `uq` | `entropy`, `ensemble` | `entropy` | Uncertainty method: the entropy of one probability, or the disagreement of the members. `ensemble` needs more than one member. |
| `ensemble_measure` | `mutual_information`, `predictive_entropy`, `mean_entropy`, `std` | `mutual_information` | Which disagreement measure `ensemble` exposes, see [Uncertainty measures](#uncertainty-measures--uqmethodspy). **This is the number `uncertainty_desc` sorts on.** |
| `decision_threshold` | number in [0, 1], or `prevalence_matched` | `0.5` | Cut of the AI's own call, used for its stand-alone errors and the reliance metrics (the reader is shown the probability, not this call). `prevalence_matched` reads from the calibration file, per finding, the cut at which the model calls as many positives as there are; it needs `calibration.path`. |
| `device` | text | `cpu` | Device of the live model. Does not change results. |
| `model_id` | text | `google/medgemma-1.5-4b-it` | `medgemma`. Model identifier. |
| `readout` | `independent_question`, `teacher_forced_logprob`, `generated_json` | `independent_question` | `medgemma`. How a probability is read out of a model that emits text, see below. |
| `prompt_set` | `single`, `triple` | `single` | `medgemma`. One prompt, or three paraphrases used as ensemble members. |
| `both_polarities` | true, false | `false` | `medgemma`. Ask each question in both polarities. |
| `dtype` | text | `auto` | `medgemma`. Weight precision of the in-process backend. |
| `backend` | `transformers`, `vllm` | `transformers` | `medgemma`. Load the weights in this process, or call a running vLLM server. |
| `endpoint` | URL | none | `vllm` (required). Base URL of the server, up to and including `/v1`. |
| `served_model_name` | text | none | `vllm` (required). The server's `--served-model-name`. |
| `concurrency` | integer | `8` | `vllm`. In-flight requests per image. |
| `top_logprobs` | integer | `20` | `vllm`. Log-probabilities requested per token. |
| `calibration` | block | omitted, raw probabilities | Post-hoc calibration, see below. |

> Available DenseNet121 weights: `densenet121-res224-all`, `-nih`, `-chex`, `-mimic_ch`,
> `-mimic_nb`, `-pc`, `-rsna`.

#### `ai.calibration`

Calibration maps are monotone: they change the numbers and every decision at a fixed
threshold, never a ranking. Fit them with `scripts/fit_calibration.py` on data the study
does not score (`patient_generator.split`).

<!-- fields: CalibrationConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `identity`, `temperature`, `platt`, `isotonic` | `identity` | Calibration map. The thesis uses per-pathology `platt`. |
| `stage` | `output`, `members` | `output` | Calibrate the fused probability the reader is shown, or each member before fusion. |
| `path` | path | none | `platt`. Per-pathology calibration file written by `fit_calibration.py`; when set it replaces `a` and `b`. |
| `temperature` | number or per label | `1.0` | `temperature`. Divisor of the logits. |
| `a`, `b` | numbers | `1.0`, `0.0` | `platt`. Slope and intercept of `sigmoid(a · logit(p) + b)`. |
| `knots_x`, `knots_y` | lists of numbers | empty, empty | `isotonic` (required). Fitted knots, `knots_x` ascending in [0, 1]. |

#### MedGemma — how a probability is read out of a generative model

MedGemma emits *text*, not a probability per pathology, so a **readout** elicits one.
Which readout you pick is a modelling choice, not a parsing detail:

| readout | how the number is obtained | verdict |
|---|---|---|
| `independent_question` **(default)** | one question per pathology, each branching off a *shared* reading of the image, so no answer can see another | **use this** |
| `teacher_forced_logprob` | one forced yes/no checklist covering every pathology | **superseded — do not produce results with it**, see below |
| `generated_json` | parses the confidences the model writes down | a number the model found plausible to emit, *not* a probability it holds — a deliberate contrast |

⚠️ **Why the checklist was abandoned, measured rather than assumed.** A checklist is one
sequence, so the answer at position *k* is conditioned on the answers forced before it.
On real NIH chest X-rays that dependence dominated the image: forcing all-yes versus
all-no moved the mean answer from **0.875 to 0.099** (mean absolute gap 0.78 on a 0–1
scale), the gap correlated **+0.48 with position**, and position 0 — the only slot with
nothing before it — came out **identical** under both fillers.

`independent_question` removes the dependence by construction. Encoding the image
(896×896 → 256 tokens) and the instruction is nearly all the compute and is identical for
every question, so the backend runs that prefix **once** and rewinds its key-value cache
between questions. Only this readout works over the vLLM backend.

⚠️ **Gated, and slow on CPU.** Accept the Health AI Developer Foundations terms on
Hugging Face and `huggingface-cli login` first. One forward pass of the 4B model takes
tens of seconds on a CPU, so **build the cache cells once and replay them** — see
`configs/examples/v5_medgemma_*.yaml`. A MedGemma cell is an ordinary cache cell, so every
study and metric works on it unchanged.

### `persona`

The doctor **is** a persona: capability × visibility × trust. `persona` is either the name
of a preset of the library or a block. A preset is a signal-detection reader (`sdt`) with
the same sensitivity and specificity on every finding, blind to the image, and a linear
pool with the listed weight when it sees the AI.

| preset | sensitivity | specificity | sees AI | weight of the AI |
|---|---|---|---|---|
| `solo_senior` | 0.95 | 0.95 | no | |
| `solo_junior` | 0.85 | 0.88 | no | |
| `senior_skeptical` | 0.95 | 0.95 | yes | 0.20 |
| `senior_trusting` | 0.95 | 0.95 | yes | 0.45 |
| `junior_skeptical` | 0.85 | 0.88 | yes | 0.40 |
| `junior_trusting` | 0.85 | 0.88 | yes | 0.65 |

Only the ordering senior < junior of the weights is anchored in the literature; the
numbers themselves are modelling choices (thesis §5.8, §8.3).

<!-- fields: PersonaConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | none | Label of the persona in the outputs. |
| `capability` | block | all defaults below | The reader's own, unaided diagnostic ability. |
| `sees_ai` | true, false | `false` | Visibility: whether the reader is shown the AI's probabilities. `false` is the solo reader. |
| `trust` | block | none | Integration policy, used only when `sees_ai` is true. |

#### `persona.capability`

<!-- fields: CapabilityConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `fixed_accuracy`, `random_mock`, `sdt` | `fixed_accuracy` | `sdt` is the signal-detection observer whose belief varies case by case (the reader of every experiment); `fixed_accuracy` keeps or flips each true label with a fixed probability, so its confidence is a constant; `random_mock` draws from Beta distributions for tests. |
| `service_time_mu` | number | `2.5` | Location of the log-normal reading time (median ≈ e^mu), used when `service_time` is omitted. The doctor is usually the bottleneck, so raising it lengthens the queue. |
| `service_time_sigma` | number | `0.4` | Spread of the log-normal reading time, used when `service_time` is omitted. |
| `sensitivity` | number or per label | `0.9` | `fixed_accuracy`, `sdt`. Operating point: lower → more missed findings. |
| `specificity` | number or per label | `0.9` | `fixed_accuracy`, `sdt`. Operating point: lower → more false alarms. |
| `d_prime` | number or per label | none | `sdt`. Sensitivity index; when set it replaces the operating point. |
| `criterion` | number or per label | none | `sdt`. Decision criterion. |
| `prior` | number or per label | `0.5` | `sdt`. Base rate the belief is calibrated to. |
| `alpha_conf`, `beta_conf` | numbers | `5.0`, `2.0` | `random_mock`. Beta parameters of the confidence. |
| `alpha_unc`, `beta_unc` | numbers | `2.0`, `5.0` | `random_mock`. Beta parameters of the uncertainty. |
| `source` | `fixed`, `reader` | `fixed` | Where the operating point comes from. `reader` is reserved for a real reader's reads and raises. |
| `reader_id` | text | none | Reserved, with `source: reader`. |
| `difficulty_slope` | number ≥ 0 | `0.0` | Share of accuracy lost at difficulty 1. At zero the reader is blind to difficulty, and its errors are independent of the AI's. |
| `fatigue_slope` | number ≥ 0 | `0.0` | Share of accuracy lost at fatigue 1. At zero the reader never tires. |
| `floor` | number in [0, 1] | `0.5` | Accuracy below which degradation stops, so a worn reader becomes a coin flip and never inverts. |
| `service_time` | block | omitted, log-normal with the two fields above | Reading-time model, see below. |

#### `persona.capability.service_time`

<!-- fields: ServiceTimeConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `lognormal`, `constant` | `lognormal` | Shape of the reading time. |
| `mu` | number | `2.5` | `lognormal`. Location parameter. |
| `sigma` | number ≥ 0 | `0.4` | `lognormal`. Spread parameter. |
| `value` | number ≥ 0 | `1.0` | `constant`. The reading time. |
| `factor` | number ≥ 0 | `1.0` | Constant multiplier of the draw: below 1 the AI speeds the reader up, above 1 it costs verification time. The literature supports both, so sweep it. |
| `fatigue_slope` | number ≥ 0 | `0.0` | Stretch of the read in proportion to the fatigue signal. Applied before `factor`. |

#### `persona.trust`

<!-- fields: TrustConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `linear_pool`, `confidence_gated`, `log_odds_fusion` | `linear_pool` | Integration rule. `linear_pool` averages the two probabilities; `confidence_gated` listens to the AI only where it is confident; `log_odds_fusion` adds the two opinions as evidence, after moving the AI's probability to the reader's base rate. |
| `decision_threshold` | number in (0, 1) | `0.5` | Fused belief at or above which the team calls a finding present. |
| `weight_ai` | number in [0, 1] | `0.39` | `linear_pool`, `confidence_gated`. Weight of advice, the AI's share of the pool. 0.39 is the pooled mean of the advice-taking literature. |
| `confidence_threshold` | number in [0, 1] | `0.7` | `confidence_gated`. Minimum AI confidence, max(p, 1 − p), below which the advice is ignored. |
| `beta` | number ≥ 0 | `1.0` | `log_odds_fusion`. Sharpness of the down-weighting of the AI by its own uncertainty, exp(−beta · H). |
| `prior` | number in (0, 1) | `0.5` | `log_odds_fusion`. Prior removed from the two log-odds before they are added. |

### `fusion`

Changes the integration rule of any persona, preset or block, leaving its other fields
untouched — this is how one study runs the same personas under several rules (E4).

<!-- fields: FusionConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `linear_pool`, `confidence_gated`, `log_odds_fusion` | none | Integration rule imposed on the persona. |
| `beta` | number ≥ 0 | none | Sharpness of the logarithmic pool. |
| `decision_threshold` | number in (0, 1) | none | Team decision threshold. |

### `sort`

<!-- fields: SortConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `random`, `uncertainty_desc`, `probability_desc` | required | Order in which the doctor takes the waiting patients: a uniform random priority (the baseline), most uncertain first, or most probably abnormal first. |
| `aggregator` | `mean`, `max`, `weighted` | `mean` | How the per-finding values are reduced to one priority. `mean` dilutes one critical finding among many confident ones, `max` escalates a patient with any one high value, `weighted` weights by clinical importance. |
| `weights` | list of numbers | none | `weighted` only (required). One weight per finding. |

**The mean wait is identical under every order (permutation-invariant)**: reordering
changes *who* waits, not how long on average. Order effects exist only under saturation,
so measure them with metrics split by ground truth or severity, at a load that saturates
the doctor.

### `severity`

A per-patient severity signal, **logged in the telemetry whether or not anything consumes
it**.

<!-- fields: SeverityConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `constant`, `activation`, `image_quality`, `embedding` | `constant` | Severity model. `activation` = aggregate over findings of criticality × activation. |
| `value` | number | `0.0` | `constant`. The severity of every patient. |
| `source` | `ai_probs`, `ai_binary`, `ai_uncertainty`, `gt` | `ai_probs` | `activation`. The activation vector. |
| `criticality` | list of numbers | none | `activation`. Clinical weight of each finding (one per label). |
| `aggregator` | `mean`, `max`, `sum` | `max` | `activation`. Reduction of the weighted vector. |
| `mode` | `scalar`, `vector` | `scalar` | `activation`. One severity per patient, or one per finding. |
| `quality_metric` | `laplacian_var`, `intensity_std` | `laplacian_var` | `image_quality`. Sharpness or contrast proxy. |
| `embedding_reduce` | `l2_norm`, `mean` | `l2_norm` | `embedding`. Reduction of the feature vector of the live model (needs `ai.cache_mode: off`). |

### `difficulty`

The per-case difficulty δ in [0, 1], computed after the AI has read the image and handed to
the doctor as `sig.difficulty`. It is the only route by which a reader's errors can
correlate with the AI's (thesis §5.7). No experiment of the thesis uses it (§8.3).

<!-- fields: DifficultyConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `kind` | `constant`, `prior`, `uncertainty`, `image_quality`, `table` | `prior` | `prior` draws a fresh Beta value per image (the ρ = 0 control), `uncertainty` reads the AI's own uncertainty (the coupling is induced from the model), `image_quality` reads blur or contrast, `table` reads a file. |
| `value` | number in [0, 1] | `0.5` | `constant`. The difficulty of every case. |
| `a`, `b` | numbers > 0 | `2.0`, `5.0` | `prior`. Parameters of the Beta distribution, a modelling choice. |
| `source` | `uncertainty` or a measure name | `uncertainty` | `uncertainty`. The run's selected uncertainty, or a named ensemble measure such as `mutual_information` or `mean_entropy`. |
| `aggregator` | `mean`, `max`, `none` | `mean` | `uncertainty`. Reduction over findings. |
| `scale` | number | none, the theoretical maximum of the measure | `uncertainty`. Value mapped to difficulty 1. |
| `quality_metric` | `laplacian_var`, `intensity_std` | `laplacian_var` | `image_quality`. Sharpness or contrast proxy. |
| `reference` | number | none, the running median of the quality seen so far | `image_quality`. Quality value at which difficulty is 0.5. |
| `path` | path | none | `table` (required). CSV file with one row per image. |
| `id_column` | text | `example_id` | `table`. Column holding the image identifier. |
| `value_column` | text | `difficulty` | `table`. Column holding the difficulty. |
| `default` | number in [0, 1] | `0.5` | `table`. Difficulty of an image missing from the file. |
| `strict` | true, false | `false` | `table`. Fail on a missing image instead of using `default`. |

### `fatigue`

How worn the reader is when it picks up each patient, handed to the doctor as
`sig.fatigue`. Radiology has no analytic fatigue curve, so the shape is a sensitivity
axis. ⚠️ Time has no unit: the defaults assume one time unit = one hour. No experiment of
the thesis uses it (§8.3).

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

### `label_view`

Projects the model's labels onto the study label set; AI, doctor, ground truth and metrics
all follow it.

<!-- fields: LabelViewConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `mode` | `full`, `subset`, `binary_abnormal` | `full` | No projection, keep only `targets`, or collapse them into one abnormal label. |
| `targets` | list of label names | none, required for `subset`, every label for `binary_abnormal` | Findings kept or collapsed. One target under `subset` gives a single-finding study. |
| `combine` | `noisy_or`, `max` | `noisy_or` | `binary_abnormal`. How the probabilities of the targets are combined. |

### `kpis`

```yaml
kpis: [system, actors, team, joint_table]   # the default: all four
```

Each entry is either a **registered name** or a reference to your own bundle, so a new
metric never requires patching the library:

```yaml
kpis:
  - system
  - configs/examples/custom_metrics.py:CostBundle   # path/to/file.py:Attribute
  - my_package.my_metrics:CostBundle                # or an importable module
```

A bundle needs only a `name` and a `compute(records)`, and must be constructible with no
arguments. The report is keyed by the bundle's own `name`, so a custom bundle is reachable
by the study layer's dotted paths (`cost.hybrid_total`) exactly like a built-in one. See
`configs/examples/custom_metrics.py` for two worked examples and
`src/hai_sim/metrics/registry.py` for the contract.

> Resolving a reference imports the target, which runs its module-level code. A scenario
> file is therefore as trusted as a script.

### `kpi_groups`

Every selected bundle is computed once more **per subgroup**. An entry of `kpi_groups`, and
of the `by` list of a study, has these fields:

<!-- fields: KPIGroupConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `key` | `label`, `truth`, or a record field | required | The axis. `label` gives one subgroup per finding, turning a macro-averaged bundle into a per-finding one. `truth` gives, for every finding, the patients who have it and those who do not — how a time metric is asked "do the sick wait less?". Any other key names a record field (`severity`, `priority_key`, `ts_arrival`) or an observation (`ctx.sex`, `load.in_system_at_arrival`, `ai.*`, `doc.*`, `sig.*`). |
| `edges` | list of numbers | empty | Bin edges for a continuous key: `[1, 5]` gives below 1, from 1 to 5, and 5 or more. Leave empty for a categorical key (`label` and `truth` take none). |
| `include_missing` | true, false | `false` | Keep the records that lack the key instead of dropping them, so the denominator never changes silently. |

```yaml
kpi_groups:
  - key: severity
    edges: [1.0, 2.5]                    # → severity <1, 1–2.5, >=2.5
  - key: load.in_system_at_arrival
    edges: [5.0, 20.0]                   # quiet / busy / saturated at arrival
  - key: label                           # one subgroup per pathology
```

Several entries **cross**: `severity × label` yields one subgroup per (band, pathology).
The console prints one column per axis; the full grid goes to the JSON under
`kpis_by_group`. Because a grouping just splits the records and re-runs the unchanged
bundle, *any* bundle is conditionable — including your own.

### Validation scenario (`hai-sim validate`)

A separate, smaller schema: one model, a dataset with expert masks, and a saliency method.
It reuses the `ai` block, so the model validated is the one a simulation would use.

<!-- fields: ValidationScenarioConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | `unnamed_validation` | Name of the run. |
| `seed` | integer | `0` | Seed. |
| `label_schema` | `xrv` | `xrv` | Label set; only `xrv`. |
| `dataset` | block | required | Dataset with expert localisation masks, see below. |
| `ai` | block | required | The model, as in a scenario. |
| `saliency` | `none`, `grad_cam` | `grad_cam` | Heatmap method. `grad_cam` needs `ai.kind: torchxrv`; `none` validates the labels only. |
| `metrics` | block | all defaults below | Localisation metrics. |
| `limit` | integer | `0` | Images validated; 0 = all. |

<!-- fields: LocalizationDatasetConfig -->
| `dataset` field | values | default | meaning |
|---|---|---|---|
| `kind` | `chestxdet` | `chestxdet` | Dataset format (ChestX-Det). |
| `annotation_path` | path | required | The annotation JSON. |
| `image_dir` | path | required | Image directory. |
| `resize` | integer | `224` | Side of the square the images are resized to. |

<!-- fields: ValidationMetricsConfig -->
| `metrics` field | values | default | meaning |
|---|---|---|---|
| `pointing_game` | true, false | `true` | Does the heatmap's peak fall inside the mask? |
| `pointing_tolerance` | integer | `0` | Radius in pixels of a pointing-game hit; 0 = the exact pixel. |
| `energy_in_mask` | true, false | `true` | Share of the heatmap's energy inside the mask. |
| `iou_thresholds` | list of numbers | `[0.5]` | Thresholds at which the binarised heatmap's IoU with the mask is computed. |
| `dice_thresholds` | list of numbers | `[0.5]` | The same for the Dice coefficient. |

### Gotchas (validation rejects these)

- `ai.kind: torchxrv` needs `label_schema: xrv`; `ai.kind: cached` needs `ai.cache_stem`.
- `ai.backend: vllm` needs `ai.endpoint` and `ai.served_model_name`.
- `ai.decision_threshold: prevalence_matched` needs `ai.calibration.path`.
- For `torchxrv` and `medgemma`, a `cache_mode` other than `off` needs a dataset-backed
  patient generator (`xrv_dataset` or `image_folder`), and feature-space severity
  (`severity.kind: embedding`) needs `cache_mode: off`.
- `patient_generator.kind: xrv_dataset` and `image_folder` need `imgpath`; `chexpert` needs
  `csvpath`; `image_folder` needs at least one `label_sources` entry;
  `inference_cache` needs `cache_stem`; `sample_mode: permutation` works only with
  `inference_cache`.
- `sort.aggregator: weighted` needs `sort.weights`; `severity.criticality` needs one weight
  per label.
- `arrivals.kind: nhpp` needs `rates`; `fatigue.kind: composite` needs `parts`;
  `difficulty.kind: table` needs `path`; `calibration.kind: isotonic` needs knots.
- `persona.capability.source: reader` raises, and so does an unknown persona name.
- `kpi_groups` keys `label` and `truth` take no `edges`.
- Unknown keys, out-of-range values, or `arrival_rate <= 0` raise.

### Recipes — minimal diffs

**Worklist order** — same scenario, one field:
```yaml
sort: {kind: random}                                  # A, the baseline
sort: {kind: probability_desc, aggregator: max}       # B, the most suspicious first
```
Expect: the same mean wait, a different wait for the patients who have a finding — and
only when the reader is saturated.

**With and without the AI** — same reader, flip visibility:
```yaml
persona: solo_senior
persona: senior_trusting
```

**Two rules for the same opinions**:
```yaml
fusion: {kind: linear_pool}
fusion: {kind: log_odds_fusion}
```

**Uncertainty from the members of the ensemble**:
```yaml
ai: {kind: torchxrv, weights: [densenet121-res224-chex, densenet121-res224-mimic_ch,
     densenet121-res224-pc, densenet121-res224-rsna], uq: ensemble, ensemble_measure: mean_entropy}
```

---

## Study reference

A study expands a **factor grid** over a base scenario into one run per cell per seed, then
tabulates chosen KPI paths into one tidy table.

```yaml
name: persona_grid
base:                    # inline scenario, OR a path to a scenario YAML
  name: base
  seed: 0
  label_schema: default_v0
  patient_generator: {kind: mock_dataset, arrival_rate: 0.5, n_patients: 50, n_examples: 80}
  ai: {kind: random_mock}
  persona: solo_senior
  sort: {kind: uncertainty_desc, aggregator: mean}

factors:                 # dotted config-paths → levels
  persona: [solo_senior, senior_trusting, senior_skeptical, junior_trusting, junior_skeptical]
  sort.aggregator: [mean, max]

design: grid
replications: [1, 2, 3]  # seeds — common random numbers give each arm the same stream

baseline:                # the reference arm → paired contrasts
  persona: solo_senior

by:                      # same shape as a scenario's `kpi_groups`
  - key: load.in_system_at_arrival
    edges: [5.0, 20.0]

kpis: [system, actors, team]

report:                  # omit to tabulate EVERY scalar in the report
  - team.complementarity_gap_f1
  - team.rair
  - actors.hybrid.recall_macro
  - actors.hybrid.n_fn
```

A factor is a dotted path into the scenario with a list of levels. A level may be a whole
block, so `persona` can take a list of preset names and `ai` a list of complete AI blocks
told apart by their `name`. Every cell runs once per seed in `replications`, and the seed
replaces the `seed` of the base scenario.

<!-- fields: StudyConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | `unnamed_study` | Name of the study, used in the output file names. |
| `base` | path or inline scenario | required | The scenario every cell starts from. |
| `factors` | mapping of dotted path to list of levels | required | The varied fields. Every factor needs at least one level. |
| `design` | `grid`, `ofat`, `random` | `grid` | How the levels are combined into cells, see below. |
| `n_samples` | integer | none, required for `random` | Cells drawn from the grid. |
| `design_seed` | integer | `0` | `random`. Seed of the draw. |
| `replications` | list of integers | `[0]` | Seeds. One run per cell per seed; the same seed gives every cell the same patient stream. |
| `report` | list of metric paths | empty, every scalar | Metrics tabulated, written as `bundle.metric`, e.g. `team.rair` or `system.mean_wait_to_doctor`. |
| `kpis` | list of names | none, derived from `report` or taken from the base scenario | Bundles computed; a registered name or `path/to/file.py:Attribute`. |
| `by` | list of blocks | empty | Conditioning axes, as [`kpi_groups`](#kpi_groups). |
| `baseline` | mapping of factor to level | none | The reference cell of the paired contrasts. Every named factor must exist and every level must be among its levels. |

| design | what it runs |
|---|---|
| `grid` | the full cross-product |
| `ofat` | a baseline (first level of each factor) plus one run per remaining level — linear, not exponential, in the number of factors |
| `random` | `n_samples` cells drawn from the grid (`design_seed` fixes which). Not a Latin hypercube: factors here are lists of levels, with no interval to stratify. The `baseline` cell is always kept |

#### Paired contrasts — what `baseline` buys you

The engine splits each run's seed into independent per-module substreams, so **two arms at
the same seed read the same patients in the same order**. Declaring `baseline` turns that
into a number: the reported delta is the mean of the *per-seed differences*, with a 95%
interval on those differences. The shared sampling noise cancels rather than being averaged
over, which is why the interval is far tighter than comparing two independent means. Seeds
present in only one arm are skipped. A partial baseline (naming some factors) contrasts
each arm against the cell that differs in *those* factors alone, so a contrast never
confounds two changes at once.

#### Output files

| file | shape |
|---|---|
| `study_<name>_<stamp>.csv` | wide — one row per (run, subgroup) |
| `study_<name>_<stamp>_long.csv` | **canonical** — one row per (cell, seed, subgroup, metric); the shape to group and plot |
| `study_<name>_<stamp>_summary.csv` | per-cell `n / mean / sd / ci_low / ci_high`, plus `vs_baseline_delta` and its interval |
| `study_<name>_<stamp>.json` | everything above, plus the validated study config |

Confidence intervals are two-sided 95% on Student's t (`study/stats.py` carries its own
critical-value table, so no scipy; normal approximation past 30 degrees of freedom).

`hai-sim study --workers N` spreads the distinct scenarios over N processes; every run
carries its own seed, so the result does not depend on the order of execution.

#### Examples

- `configs/studies/queue_load.yaml` — experiment E3: three arrival rates × three worklist
  orders, 20 seeds, every time metric split by ground truth and paired against the random
  order.
- `configs/studies/team_effect.yaml` — experiment E4: four personas × two fusion rules ×
  three AIs, 20 seeds, every KPI per finding.
- `configs/studies/team_effect_extra.yaml` — E4 with the weights of advice swapped (a
  senior at 0.65, a junior at 0.20), same reads; not in the thesis, it is the figure of the
  defense slide "E4 · WHO GAINS".

---

## Output

### CSV — per-patient telemetry

One row per patient. Any downstream KPI can be re-derived from this file alone.

| Column | Meaning |
|---|---|
| `patient_id` | sequential id assigned at generation |
| `ts_arrival` | sim time the patient entered `input_queue` |
| `ts_model_start` / `ts_model_end` | AI worker pick-up / `diagnose` return |
| `ts_doctor_start` / `ts_doctor_end` | doctor pick-up / verdict recorded |
| `priority_key` | key used by the `PriorityStore` (lower → served earlier) |
| `gt__<label>` | ground truth bit per label (0/1; `-1` = unannotated, skipped) |
| `ai_label__`, `ai_conf__`, `ai_unc__` | AI prediction / confidence / uncertainty per label |
| `doc_label__`, `doc_conf__`, `doc_unc__` | doctor's, per label |
| `severity` / `severity__<label>` | present when a `severity` model is configured |

#### Observations — the open part of the record

Beyond the fixed columns above, the record carries a namespaced **observation bag**:
whatever a module puts in `ModelOutput.extra` reaches the telemetry, so a new module
surfaces a new observable with no change to the collector. Per-label values expand into
one column per label (`ai.mutual_information__Effusion`); scalars take one column.

| namespace | source | examples |
|---|---|---|
| `ai.*` | the AI station's `ModelOutput.extra` | `ai.uq_method`, `ai.n_members`, `ai.image_id`, `ai.cache_index`, and from `uq: ensemble` the **full decomposition**: `ai.predictive_entropy__*` (total), `ai.mean_entropy__*` (aleatoric), `ai.mutual_information__*` (epistemic), `ai.std__*`, `ai.coverage__*` |
| `doc.*` | the doctor station's `ModelOutput.extra` | `doc.policy`, `doc.assisted`, `doc.ai_hint_used`, `doc.fusion_weight_ai__*` (the weight the trust policy actually applied, per label), `doc.doctor_solo_confidence__*` |
| `ctx.*` | `PatientContext` | `ctx.example_id` (the dataset item / image filename — this is what joins telemetry back to the image), `ctx.age`, `ctx.sex`, `ctx.priority` |
| `load.*` | queue occupancy sampled by the engine | `load.input_queue_at_arrival`, `load.in_system_at_arrival`, `load.triaged_at_enqueue` (how deep the triaged queue was when this patient joined it — i.e. whether triage had room to act), `load.triaged_at_read` |

`patient_id` counts arrivals and is unique even when the same image is sampled twice;
`ctx.example_id` names the image. They are deliberately different things.

Every observation key is also a valid `kpi_groups` key and readable from a custom bundle
via `record.observation("ai.mutual_information")`.

### JSON — run summary

```jsonc
{
  "scenario": { /* validated config dump */ },
  "ai": "TorchXRayVisionAI",
  "doctor": "AssistedDoctor(FixedAccuracyDoctor, LogOddsFusionPolicy)",
  "sort": "UncertaintyDescSort(mean)",
  "wall_clock_s": 0.07,            // Python wall-clock; NOT simulated time
  "metadata": { "sim_end_time": 1285.24, "n_ai_workers": 1, "n_doctor_workers": 1 },
  "kpis": { /* the enabled bundles */ }
}
```

---

## KPI definitions

Computed in `src/hai_sim/metrics/kpi.py`. Four swappable bundles, selected by `kpis:`.

### `system` — run-level (`RunKPIs`)

| KPI | Definition |
|---|---|
| `n_patients` | patients fully processed end-to-end |
| `sim_duration` | `max(ts_doctor_end)` |
| `throughput` | `n_patients / sim_duration` — compare against λ to detect saturation |
| `mean_wait_to_model` | mean `ts_model_start − ts_arrival` |
| `mean_wait_to_doctor` | mean `ts_doctor_start − ts_model_end`. ⚠️ **The sum is invariant under permutation of the queue**, so the *mean* does not change with the sort policy — only the per-patient distribution does |
| `mean_time_in_system` | mean `ts_doctor_end − ts_arrival` |

### `actors` — per-actor (`ActorKPIs`), computed for `ai`, `doctor`, `hybrid`

| KPI | Definition | Clinical reading |
|---|---|---|
| `f1_macro` | mean per-label F1 | primary classification KPI; every pathology weighted equally |
| `precision_macro` | mean per-label TP/(TP+FP) | "when it says positive, how often is it right?" |
| `recall_macro` | mean per-label TP/(TP+FN) = sensitivity | "of truly positive labels, how many are caught?" Low recall ⇒ many FN |
| `accuracy` | (ΣTP + ΣTN) / total entries | **misleading at low prevalence** (all-zero scores high). Reported for completeness |
| `n_fn` | (patient, label) pairs with gt=1, pred=0 | **clinically the worst error** — missed pathology |
| `n_fp` | pred=1, gt=0 | over-call |
| `ttfe` | Time To First Error: timestamp of the first patient mispredicted on *any* label | how long before the system slips |
| `mttf` | Mean Time To Failure: mean gap between error events | large ⇒ errors well spread; small ⇒ bursty. `None` if < 2 errors |
| `mttfn` | mean gap between consecutive **false-negative** events | the FN-focused twin of MTTF — the one radiology cares about |

**"Error"** for `ttfe`/`mttf` = the predicted label vector differs from GT on **at least one
label** (hamming-style, not exact match); the event time is `ts_doctor_end`.

**The three actor blocks**: `ai` = the model alone (what would a solo deployment do?);
`doctor` = the **unaided** call (for an `AssistedDoctor`, the stashed counterfactual);
`hybrid` = the doctor stage's **actual** output — the team call when AI-assisted.

### `team` — human-AI team quality (`TeamKPIs`)

Because the simulator knows **both** counterfactuals for every case, the appropriate-reliance
metrics — which normally need a two-stage human study — are computed directly.

| KPI | Definition | Reading |
|---|---|---|
| `complementarity_gap_f1` / `_accuracy` | team − max(AI-solo, doctor-solo) | > 0 ⇒ **the whole exceeds its parts** |
| `rair` | Relative AI Reliance: of entries where doctor-solo wrong & AI right, the fraction the team got right | did the team *harvest* correct AI? |
| `rsr` | Relative Self-Reliance: of entries where doctor-solo right & AI wrong, the fraction the team got right | did the team *resist* wrong AI? |
| `over_reliance` / `under_reliance` | `1 − rsr` / `1 − rair` | agency lost to wrong AI / correct AI ignored |
| `n_ai_reliance_opportunities` / `n_self_reliance_opportunities` | entries where doctor-solo wrong & AI right / doctor-solo right & AI wrong | the denominators of `rair` / `rsr`; a ratio over a handful of opportunities is noise |

RAIR/RSR follow Schemmer et al. (2023), *IUI*.

### `joint_table` — the AI × Doctor × GT contingency

Two confusion matrices plus 8 GT-signed cells — where exactly the two actors agree,
disagree, and who is right when they do.

### Reading a result

- **`recall_macro` / `n_fn`** → the clinical axis (missed disease). Watch these first.
- **`precision_macro` / `n_fp`** → the workload axis (over-calls).
- **`mttfn` / `ttfe`** → *when* errors accumulate — the axis triage order moves.
- **`mean_wait_to_doctor` / `throughput`** → saturation; tells you if the load knobs bit.

**Never judge a change by `f1_macro` or `accuracy` alone — always decompose into recall (FN)
vs precision (FP).**

---

## What is implemented

### The contract — `src/hai_sim/core/protocols.py`

Fourteen `@runtime_checkable` Protocols. `BaseDiagnosticActor` (`core/base.py`) is the only
ABC, and inheriting it is optional.

A seam earns its place when you can name three implementations you would actually run
**and compare in the same study**. That rule is why there is no `CostModel` (a cost matrix
is already a `KPIBundle`) and no workflow-topology seam (only one arrangement is under
test: the AI in series, before the doctor).

| Protocol | Method | Implementations |
|---|---|---|
| `DiagnosticActor` | `diagnose(images, *, context, ai_hint) → ModelOutput` | `RandomMockAI`, `FixedAccuracyAI` (synthetic), `TorchXRayVisionAI`, `MedGemmaAI`, `VLLMMedGemmaAI`, `CachedEnsembleAI`, `CompositeCachedAI`, `RandomMockDoctor`, `FixedAccuracyDoctor`, `SDTDoctor`, `AssistedDoctor`, `ProjectedActor` |
| `IntegrationPolicy` | `combine(doctor_out, ai_hint, context)` | `LinearPoolPolicy`, `ConfidenceGatedPoolPolicy`, `LogOddsFusionPolicy` |
| `SortStrategy` | `priority(output, patient) → float` | `RandomSort`, `UncertaintyDescSort`, `ProbabilityDescSort` |
| `UQMethod` | `estimate(member_probs) → (probs, unc, extra)` | `EntropyUQ`, `EnsembleUQ`, `CalibratedUQ` |
| `Calibrator` | `calibrate(probs) → probs` | `IdentityCalibrator`, `TemperatureScaling`, `PlattCalibrator`, `IsotonicCalibrator` |
| `ArrivalProcess` | `next_delay(now) → float` | `PoissonArrival`, `DeterministicArrival`, `NHPPArrival` (+ `PiecewiseConstantRate`) |
| `SeverityModel` | `score(*, patient, ai_output, context)` | `ConstantSeverity`, `ActivationSeverity`, `ImageQualitySeverity`, `EmbeddingSeverity` |
| `DifficultyModel` | `score(*, patient, ai_output, context)` | `ConstantDifficulty`, `PriorDifficulty`, `UncertaintyDifficulty`, `ImageQualityDifficulty`, `TableDifficulty` |
| `FatigueCurve` | `level(state) → float` | `NoFatigue`, `LinearFatigue`, `VigilanceFatigue`, `CircadianFatigue`, `CompositeFatigue` |
| `ServiceTimeModel` | `sample(images, context) → float` | `ConstantServiceTime`, `LognormalServiceTime`, `ScaledServiceTime`, `FatigueModulatedServiceTime` |
| `Dataset` | `__len__` / `__getitem__` + `capabilities` | `MockDataset`, `XRVDataset`, `ImageFolderDataset` |
| `SaliencyProvider` | `heatmap(image, label_index)` | `GradCAMSaliency` (all-NaN = not localizable) |
| `LocalizationDataset` | | `ChestXDetDataset` |
| `LocalizationMetric` | `score(heatmap, gt_mask)` | `PointingGame`, `EnergyInMask`, `IoU`, `Dice` |

Two more Protocols live in `metrics/`:

| Protocol | Method | Implementations |
|---|---|---|
| `KPIBundle` (`metrics/registry.py`) | `compute(records) → Any` | `SystemBundle`, `ActorsBundle`, `TeamBundle`, `JointTableBundle` — **plus anything you reference from a scenario as `file.py:Attribute`** |
| `Grouping` (`metrics/slicing.py`) | `split(records) → [RecordSlice]` | `ValueGrouping`, `LabelGrouping` |

### Module map — `src/hai_sim/`

```
core/protocols.py           the 14 Protocols = the public contract (KPIBundle and
                            Grouping live in metrics/)
core/types.py               frozen dataclasses: LabelSchema, Diagnosis, ModelOutput,
                            Patient(Context), Localization/DatasetExample, DatasetCapabilities
core/base.py                BaseDiagnosticActor ABC
cli.py                      `hai-sim` entrypoint → run | study | capture | plot | validate
factory.py                  ScenarioConfig → runtime objects (all the registries live here)
labels.py                   LabelProjection / ProjectedActor — remap actor schema → dataset schema
personas.py                 PERSONA_LIBRARY: the 2×3 capability × attitude grid
sdt.py                      the signal-detection core shared by the SDT reader and the synthetic AI
config/scenario.py          strict pydantic models + load_scenario / load_validation_scenario
models/                     AI backends: mock, synthetic (SDT, skill as a dial), torchxrv,
                            cached (single and composite), medgemma + readout + vllm_backend
doctors/                    mock, simple (FixedAccuracyDoctor), sdt (SDTDoctor), assisted
                            (AssistedDoctor + the 3 integration policies)
sort/{random,uncertainty}.py  the triage policies (random, uncertainty, probability)
uq/methods.py               EntropyUQ, EnsembleUQ
uq/calibration.py           the four calibrators + CalibratedUQ
arrivals/processes.py       Poisson, deterministic, NHPP by thinning
severity/, difficulty/      per-patient signals computed between the two stations
fatigue/                    fatigue curves and reading-time models
data/                       MockDataset | XRVDataset | ImageFolderDataset, CSV label sources
sim/engine.py               the SimPy engine
sim/{patient,dataset,cache}_generator.py   the three patient streams
metrics/collector.py        PatientRecord + MetricsCollector — fixed core plus the
                            open, namespaced observation bag (ai./doc./ctx./load./sig.)
metrics/kpi.py              the four built-in bundles + compute_(grouped_)report
metrics/registry.py         open bundle registry: name | module:Attr | file.py:Attr
metrics/slicing.py          conditioned metrics: split the records, rerun the bundle
metrics/report.py           flatten a report to dotted path → scalar
study/{spec,runner}.py      factor grid → tidy long table (+ wide, + summary)
study/stats.py              replication statistics and paired contrasts
analysis/inference_cache.py resumable per-member probability cache
analysis/cache_matrix.py    one cell per (model, dataset), composed into ensembles by image id
analysis/cache_build.py     fill the missing cells of a scenario
analysis/capture_study.py   the capture study of E1 and E2 (+ capture_figures.py)
analysis/study_figures.py   the figures of E3 and E4 (`hai-sim plot`)
analysis/error_curve.py     error-capture curves and their skill
analysis/localization/*     the validation harness of `hai-sim validate`: cache, cam,
                            chestxdet, evaluate, metrics, report, saliency, validator
```

### Uncertainty measures — `uq/methods.py`

All NaN-aware (`coverage` reported in `extra`), because not every ensemble member covers
every pathology. Binary entropy in nats.

| measure | formula | reads as |
|---|---|---|
| `predictive_entropy` | H(mean_p) | total |
| `mean_entropy` | mean_m H(p_m) | aleatoric |
| `mutual_information` | PE − ME, clipped ≥ 0 | **epistemic** (default) |
| `std` | std_m p_m | raw member disagreement |

`EntropyUQ` is the single-pass binary entropy (an aleatoric proxy).

### Scripts — `scripts/`

| script | purpose |
|---|---|
| `materialize_cache.py` | build (`--mode fill`) or check (`--mode report`) the cache cells of a scenario |
| `fit_calibration.py` | fit the per-pathology Platt maps on the even rows of a cache (`--split even`) |
| `build_vindr_labels.py` | VinDr-CXR raw annotations → `labels.csv`, `difficulty.csv`, `readers.csv` |
| `download_chestxdet.py` | ChestX-Det images and masks, the ground truth of `hai-sim validate` |
| `build_localization_cache.py` | forward-only feature maps (~2.8 GB fp32) for offline Grad-CAM (`validate --from-cache`) |

---

## Determinism

A single `seed` drives the whole run. Internally `np.random.SeedSequence(seed)` is spawned
into 5 independent sub-streams (patient generation, AI, doctor, sort, and arrival)
so that **swapping one module does not perturb the others**. Same seed + same scenario file
= byte-identical CSV/JSON. In studies, `replications` are seeds, giving each arm common
random numbers.

## Tests

485 tests across 48 files. `uv run pytest`.

`test_protocol_compliance.py` asserts runtime conformance to the Protocols.
`test_real_models.py` is gated behind the `models` extra.

## Future work

Thesis §8.4 lists the future work: replay the reads of real physicians, measure the weight
of advice instead of setting it, and test which uncertainty measures shorten the time to
diagnosis of the positive cases. Fatigue and case difficulty are implemented and tested
but no experiment of the thesis uses them (thesis §8.3).
