# hai_sim — Human-AI Diagnostic Workflow Simulator

Modular SimPy simulator of a hybrid human-AI chest X-ray (CXR) reading department.
Master thesis of Daniele Greco (UniBg).

> **This is the technical reference**: how to install it, how to run it, every knob,
> every KPI, and everything currently implemented.
>
> **The *why* — research questions, state of the art, decisions, roadmap and open
> questions — lives in [`../TESI.md`](../TESI.md)**, the project's single descriptive
> document. Raw literature reviews are archived in `../literature/`.

**Founding constraint: no training from scratch.** Pre-trained models, inference only.
The CXR dataset is the simulation's *patient generator*, not a training set.

---

## Contents

- [Install](#install) · [Quickstart](#quickstart) · [Pipeline](#pipeline) · [CLI](#cli)
- [Data & caches](#data--caches)
- [Scenario reference](#scenario-reference) — the experiment control panel
- [Study reference](#study-reference) — factor grids
- [Output](#output) · [KPI definitions](#kpi-definitions)
- [What is implemented](#what-is-implemented) — protocols, implementations, scripts
- [Determinism](#determinism) · [Tests](#tests) · [Not implemented yet](#not-implemented-yet)

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

# three scenarios, same seed — directly comparable
uv run hai-sim run configs/examples/v0_smoke_random.yaml
uv run hai-sim run configs/examples/v0_smoke_uncertainty.yaml
uv run hai-sim run configs/examples/v0_smoke_hybrid.yaml

# real NIH images through the pre-trained DenseNet121 (needs the [models] extra + data)
uv run hai-sim run configs/examples/v1_nih_torchxrv.yaml
uv run hai-sim run configs/examples/v1_nih_ensemble.yaml

# a full persona sweep (30 runs → one tidy table)
uv run hai-sim study configs/examples/study_persona_grid.yaml
```

Always run from the repo root so the repo-relative paths in the scenarios resolve.

---

## Pipeline

```
PatientGenerator → input_queue (Store, FIFO)
                 → ModelStation (× n_ai_workers)
                 → triaged_store (PriorityStore, key = SortStrategy)
                 → DoctorStation (× n_doctor_workers)
                 → MetricsCollector
```

Every actor — AI model and doctor alike — implements the same `DiagnosticActor`
protocol, `diagnose(images, *, context, ai_hint) → ModelOutput`, so any implementation is
swappable from the scenario YAML. The doctor may consult the AI's `ModelOutput` as a hint.

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
hai-sim study <study.yaml>    [--out-dir out]
hai-sim validate <scenario.yaml> | --from-cache <stem>
```

**`run`** — one scenario → per-patient telemetry CSV + summary JSON + a printed KPI table.

**`study`** — expand a factor grid → one run per cell per seed → a tidy KPI table
(CSV + JSON).

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
├── nih_sample/                      # NIH ChestX-ray14 sample (~5.6k images)
│   ├── images/
│   ├── sample_labels.csv            # original Kaggle labels
│   └── sample_labels.xrvclean.csv   # auto-generated (numeric Patient Age) — safe to delete
├── cache/                           # inference caches (build once, replay forever)
│   └── nih_ensemble.{npz,json}
├── nih_rad_labels/                  # NIH adjudicated consensus GT (4,375 rows, 4 findings)
└── chexpert_test_labels/            # CheXpert majority GT + 8 per-reader reads (500 studies)
```

**Get the images** — needs a Kaggle API token (`%USERPROFILE%\.kaggle\kaggle.json`, or the
`KAGGLE_API_TOKEN` env var):

```bash
uv run python scripts/download_nih_sample.py      # NIH sample → data/nih_sample/
uv run python scripts/download_chestxdet.py       # ChestX-Det masks (localization GT)
uv run python scripts/download_reader_labels.py   # NIH adjudicated + CheXpert reader labels (no gate)
```

**Model weights are not here.** TorchXRayVision manages them; they auto-download to
`~/.torchxrayvision/models_data/` on first use (~80 MB each) and are shared across projects.

**Caches are the reason real studies are fast.** `scripts/build_inference_cache.py` runs the
5-member cross-domain ensemble once over a dataset and stores the per-member probabilities;
afterwards `ai.kind: cached` + `patient_generator.kind: inference_cache` replay them, so a
30-run persona study over 1,000 real images finishes in seconds instead of hours. The cache
is resumable and guarded by a model fingerprint, so a stale cache cannot be silently
replayed against different weights.

---

## Scenario reference

A scenario is a single YAML file validated by `pydantic`
(`src/hai_sim/config/scenario.py`) with `extra = forbid`: **any unknown key raises**, so
typos fail loudly. Every field is optional unless marked *required*; defaults are shown.

This file is the experiment control panel. The golden rule for clean comparisons:
**change ONE block at a time, keep `seed` fixed.**

```yaml
name: my_scenario          # str  — free text, appears in output filenames
seed: 42                   # int  — drives every stochastic stream; fix it across a comparison
label_schema: xrv          # "default_v0" (5 fake) | "chexpert" (14) | "xrv" (18, real model)
patient_generator: { ... }
ai: { ... }
persona: senior_trusting   # a persona library name, OR an inline capability/visibility/trust block
sort: { ... }
severity: { ... }          # optional — a per-patient severity model (produced & logged)
label_view: { ... }        # optional — project labels (subset / binary-abnormal)
kpis: [system, actors, team, joint_table]   # optional — which KPI bundles (default = all)
n_ai_workers: 1            # int >= 1 — concurrent AI inference workers
n_doctor_workers: 1        # int >= 1 — concurrent doctors
```

### Quick map — which knob answers which question

| Experimental question | Block / field to vary | Keep fixed |
|---|---|---|
| **Q-Model**: does the AI help? | `persona.sees_ai` (solo vs assisted) | seed, persona, patients |
| **Q-Order**: does triage order matter? | `sort.kind`, `sort.aggregator` | seed, ai, persona, patients |
| **Q-Doctor / RQ3**: how do personas change team outcome & reliance? | `persona` | seed, ai, patients |
| **UQ comparison** | `ai.uq`, `ai.ensemble_measure`, `ai.weights` | seed, everything else |
| **Operating point** | `ai.decision_threshold` | seed, everything else |
| **Load / saturation** | `patient_generator.arrival_rate`, `n_doctor_workers`, `service_time*` | seed, ai, doctor quality |

### `label_schema`

| value | n labels | use |
|---|---|---|
| `default_v0` | 5 (`L0..L4`) | mock-only smoke tests |
| `chexpert` | 14 | reserved for CheXpert-native models |
| `xrv` | 18 | **required for `ai.kind: torchxrv`** (the model outputs 18 pathologies in this exact order) |

> `torchxrv` raises unless `label_schema: xrv`. The mocks work with any schema.

### `patient_generator` — the patient stream

```yaml
patient_generator:
  kind: poisson | mock_dataset | xrv_dataset | inference_cache   # default poisson
  arrival_rate: 0.5                # REQUIRED, float > 0
  n_patients: 200                  # REQUIRED, int > 0
```

| param | type | what it controls | raising it |
|---|---|---|---|
| `arrival_rate` (λ) | float > 0 | Poisson inter-arrival rate (patients per sim-time unit) | **queue saturates faster** → longer `mean_wait_to_doctor`; the *order* of service matters more |
| `n_patients` | int > 0 | total patients emitted, then stop | tighter metric estimates, longer wall-clock |

**`kind: poisson`** — synthetic, no model needed (wiring/load tests):

| param | default | effect |
|---|---|---|
| `prevalence` | `0.2` (scalar) or per-label list | P(label = 1) in synthetic GT. Higher → more positives → recall easier to read |
| `image_shape` | `[1, 1]` | placeholder array shape (mocks ignore pixels) |

**`kind: mock_dataset`** — synthetic images through the full `Dataset` contract; exercises
the dataset-agnostic path without torch:

| param | default | effect |
|---|---|---|
| `prevalence` | `0.2` | P(label = 1) in the synthetic pool |
| `n_examples` | `100` | pool size; `n_patients` are drawn from it |
| `image_shape` | `[1, 1]` | placeholder array shape |

**`kind: xrv_dataset`** — real CXR images:

| param | default | effect |
|---|---|---|
| `dataset` | `nih` \| `rsna` \| `chexpert` | which TorchXRayVision loader; sets the available GT labels (unlabeled → `-1`, skipped by metrics) |
| `imgpath` | **required** | image folder, e.g. `data/nih_sample/images` |
| `csvpath` | `null` | labels csv. Required for `chexpert`; for `nih` the sample's `Patient Age` is auto-cleaned |
| `sample_mode` | `sequential` \| `random` | `random` draws with replacement (use when `n_patients` > dataset size); `sequential` wraps around in order |
| `gt_source` | `native` | which GT source to use. **Capability-gated**: only sources the dataset *advertises* are accepted |

**`kind: inference_cache`** — replay a prebuilt cache (pair with `ai.kind: cached`):

| param | effect |
|---|---|
| `cache_stem` | path stem, e.g. `data/cache/nih_ensemble` (reads `.npz` + `.json`) |
| `sample_mode` | as above |

### `ai` — the model under test

```yaml
ai:
  kind: random_mock | torchxrv | cached    # default random_mock
  service_time: 0.05                       # float >= 0, SIM-time per inference (not wall-clock)
```

`service_time` is the *simulated* cost of one inference. It does not change predictions —
only queueing dynamics.

**`kind: random_mock`** — sanity actor, predictions ≈ random (wiring tests only):

| param | default | effect |
|---|---|---|
| `alpha_conf`, `beta_conf` | `2.0`, `2.0` | Beta(α,β) for per-label confidence; threshold at 0.5 → label |
| `alpha_unc`, `beta_unc` | `2.0`, `5.0` | Beta(α,β) for per-label uncertainty (mean ≈ 0.29) |

**`kind: torchxrv`** — the real pre-trained DenseNet121, inference only:

| param | default | values | what to expect |
|---|---|---|---|
| `weights` | `densenet121-res224-all` | one name, **or a list** | a list of dataset-specific weights (`-nih`, `-chex`, `-mimic_ch`, `-pc`, `-rsna`) forms a cross-domain Deep Ensemble: better calibrated, fewer FP, **×N slower** |
| `uq` | `entropy` | `entropy` \| `ensemble` | `entropy` = 1 pass, uncertainty from the model's own confidence (aleatoric proxy). `ensemble` = real epistemic disagreement. Use `ensemble` only with multiple `weights` |
| `ensemble_measure` | `mutual_information` | `mutual_information` \| `predictive_entropy` \| `mean_entropy` \| `std` | which per-label uncertainty is exposed — **this is the number the sort triages on**. MI = epistemic (member disagreement); predictive entropy = total; mean entropy = aleatoric; std = raw spread |
| `decision_threshold` | `0.5` | [0, 1] | probability cutoff → binary label. **Raise → fewer positives → fewer FP, more FN.** The single biggest lever on the FP/FN trade-off; 0.5 over-predicts on xrv models |
| `device` | `cpu` | `cpu` \| `cuda` | inference device; does not change results |

> Available DenseNet121 weights: `densenet121-res224-all`, `-nih`, `-chex`, `-mimic_ch`,
> `-mimic_nb`, `-pc`, `-rsna`.

**`kind: cached`** — replay a prebuilt inference cache. Same `uq` / `ensemble_measure` /
`decision_threshold` knobs as `torchxrv` (they are applied at replay time, so you can sweep
them for free), plus `cache_stem`.

### `persona` — the doctor (capability × visibility × trust)

The doctor **is** a persona. Give a **library name** or an inline block.

```yaml
persona: senior_trusting            # library: solo_senior, solo_junior,
                                    #   {senior,junior}_{trusting,skeptical}
```

```yaml
persona:                            # inline
  capability:                       # the reader's own, AI-unaided ability
    kind: fixed_accuracy            # "fixed_accuracy" | "random_mock"
    sensitivity: 0.95               # scalar in [0,1] or per-label list
    specificity: 0.95
    service_time_mu: 2.5            # lognormal mu — reading time; median ≈ exp(mu)
    service_time_sigma: 0.4
    source: fixed                   # "fixed" | "reader" (calibrate on real reads — gated)
  sees_ai: true                     # visibility: does the reader consult the AI?
  trust:                            # used only when sees_ai — a swappable IntegrationPolicy
    kind: linear_pool               # "linear_pool" | "confidence_gated" | "log_odds_fusion"
    weight_ai: 0.45
```

**capability** — `sensitivity` ↓ → more FN (missed disease); `specificity` ↓ → more FP
(over-call); both scalar or per-label. `service_time_*` set the reading-time lognormal; the
doctor is usually the bottleneck, so raising `mu` lengthens queues and makes sort order
matter more. `source: reader` would calibrate capability on a real radiologist's reads —
**currently raises**; see [Not implemented yet](#not-implemented-yet).

**visibility** — `sees_ai: false` = solo reader (the medico-solo baseline); `true` =
consults the AI.

**trust** — the swappable integration policy (`sees_ai: true` only):

| `trust.kind` | key params | behaviour |
|---|---|---|
| `linear_pool` | `weight_ai` ∈ [0,1] | weighted average `(1-w)·doctor + w·AI`. `weight_ai` **is** the Weight of Advice; the advice-taking literature's pooled mean is ≈0.39 (egocentric discounting) |
| `confidence_gated` | `weight_ai`, `confidence_threshold` | apply the WoA only where AI confidence `max(p, 1-p)` ≥ threshold; below it, ignore the AI |
| `log_odds_fusion` | `beta`, `weight_doctor`, `prior` | the normative Bayesian pool; agreeing opinions reinforce; the AI is down-weighted by its own uncertainty (`exp(-beta·H)`) |

> The library grid *{senior, junior} × {solo, trusting, skeptical}* is the RQ3 sweep:
> capability senior 0.95/0.95 vs junior 0.85/0.88; trust WoA senior < junior at each
> attitude (experts discount advice more). Read the effect off the `team` KPI block and
> the `joint_table`.

### `sort` — the triage policy

```yaml
sort:
  kind: random | uncertainty_desc    # REQUIRED (no default)
  aggregator: mean                   # only for uncertainty_desc
  weights: null                      # only for aggregator=weighted
```

| param | values | what to expect |
|---|---|---|
| `kind` | `random` \| `uncertainty_desc` | `random` = baseline. `uncertainty_desc` = serve the highest-uncertainty patients first. **The mean queue wait is identical either way (permutation-invariant)**; what changes is *which* patients are seen before a saturated system runs out of doctor time |
| `aggregator` | `mean` \| `max` \| `weighted` | collapses the per-label uncertainty vector to one scalar key. **`mean`** dilutes a single critical finding among 17 confident ones; **`max`** escalates a patient with any one very-uncertain pathology; **`weighted`** weights by clinical importance. Different aggregators reorder the queue substantially (often <20% overlap in the top-served) |
| `weights` | list of `n_labels` floats | required when `aggregator: weighted` |

### `severity` (optional)

A swappable per-patient severity signal, **logged in telemetry whether or not anything
consumes it** (triage/cost consumers are opt-in). Omit the block → no severity.

```yaml
severity:
  kind: activation            # "constant" | "activation" | "image_quality" | "embedding"
  source: ai_probs            # activation source: ai_probs | ai_binary | ai_uncertainty | gt
  criticality: [ ... ]        # optional per-label clinical weights (length n_labels)
  aggregator: max             # max | sum | mean
  mode: scalar                # scalar | vector (per-label)
```

`activation` = `aggregate_l(criticality_l × activation_l)`: criticality is a pre-assigned
per-label weight, activation a per-image signal. `image_quality` = blur/intensity proxy;
`embedding` = a reduction of the model's feature vector (auto-enables the AI's embedding).

### `label_view` (optional) — single- vs multi-label studies

Projects the model's native labels onto the **active** study schema (AI, doctor, GT and the
KPIs all follow it). Omit → full multi-label.

```yaml
label_view:
  mode: binary_abnormal       # "full" | "subset" | "binary_abnormal"
  targets: [Pneumothorax]     # subset: which labels (one target = a single-label study)
  combine: noisy_or           # binary_abnormal: noisy_or | max (how AI probs collapse)
```

### `kpis` (optional)

```yaml
kpis: [system, actors, team, joint_table]   # default = all four
```

### `n_ai_workers` / `n_doctor_workers`

| param | default | effect |
|---|---|---|
| `n_ai_workers` | 1 | concurrent AI stations. Raise to clear an AI bottleneck (rarely needed) |
| `n_doctor_workers` | 1 | concurrent doctors. **Raise → less saturation → shorter `mean_wait_to_doctor`**, and triage order matters *less* |

### Gotchas (validation rejects these)

- `ai.kind: torchxrv` **requires** `label_schema: xrv`.
- `patient_generator.kind: xrv_dataset` **requires** `imgpath` (and `csvpath` for `chexpert`).
- `sort.aggregator: weighted` **requires** `sort.weights` of length `n_labels`.
- `ai.uq: ensemble` is only meaningful with **multiple** `ai.weights`.
- `persona.capability.source: reader`, and any non-`native` `gt_source`, raise unless the
  dataset advertises that capability.
- An unknown `persona` library name, a `kpis` bundle outside the four, or
  `severity.criticality` of the wrong length, raise.
- Unknown keys, out-of-range thresholds, or `arrival_rate <= 0` raise.

### Recipes — minimal diffs

**Q-Order (mean vs max triage)** — same scenario, one field:
```yaml
sort: {kind: uncertainty_desc, aggregator: mean}   # A
sort: {kind: uncertainty_desc, aggregator: max}    # B
```
Expect: same global accuracy/F1, different patients served before saturation → differences
in *time-to-FN* and in which FN are caught early.

**Operating point (FP/FN trade-off)**:
```yaml
ai: {kind: torchxrv, ..., decision_threshold: 0.5}   # many FP, low precision
ai: {kind: torchxrv, ..., decision_threshold: 0.8}   # fewer FP, more FN, higher precision
```

**UQ method — single vs ensemble, same patients**:
```yaml
ai: {kind: torchxrv, weights: densenet121-res224-all, uq: entropy}
ai: {kind: torchxrv, weights: [..nih, ..chex, ..mimic_ch, ..pc, ..rsna], uq: ensemble, ensemble_measure: mutual_information}
```

**Q-Model (with vs without AI)** — same capability, flip visibility:
```yaml
persona: {capability: {kind: fixed_accuracy, sensitivity: 0.85, specificity: 0.9}, sees_ai: false}
persona: {capability: {kind: fixed_accuracy, sensitivity: 0.85, specificity: 0.9}, sees_ai: true, trust: {kind: linear_pool, weight_ai: 0.4}}
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

design: grid             # "grid" = full cross-product | "ofat" = one-factor-at-a-time
replications: [1, 2, 3]  # seeds — common random numbers give each arm the same stream

report:                  # KPI-paths to tabulate (one column each)
  - team.complementarity_gap_f1
  - team.rair
  - actors.hybrid.recall_macro
  - actors.hybrid.n_fn
```

`design: ofat` runs a baseline (first level of each factor) plus, for each other factor,
one run per remaining level — linear instead of exponential in the number of factors.

`configs/examples/study_nih_persona_grid.yaml` is the same sweep on **real NIH images via
cache replay**: 5 personas × 2 aggregators × 3 seeds = 30 runs in seconds, no inference.

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

Nine `@runtime_checkable` Protocols. `BaseDiagnosticActor` (`core/base.py`) is the only ABC,
and inheriting it is optional.

| Protocol | Method | Implementations |
|---|---|---|
| `DiagnosticActor` | `diagnose(images, *, context, ai_hint) → ModelOutput` | `RandomMockAI`, `TorchXRayVisionAI`, `CachedEnsembleAI`, `RandomMockDoctor`, `FixedAccuracyDoctor`, `AssistedDoctor`, `ProjectedActor` |
| `IntegrationPolicy` | `combine(doctor_out, ai_hint, context)` | `LinearPoolPolicy`, `ConfidenceGatedPoolPolicy`, `LogOddsFusionPolicy` |
| `SortStrategy` | `priority(output, patient) → float` | `RandomSort`, `UncertaintyDescSort` |
| `UQMethod` | `estimate(member_probs) → (probs, unc, extra)` | `EntropyUQ`, `EnsembleUQ` |
| `SeverityModel` | `score(*, patient, ai_output, context)` | `ConstantSeverity`, `ActivationSeverity`, `ImageQualitySeverity`, `EmbeddingSeverity` |
| `Dataset` | `__len__` / `__getitem__` + `capabilities` | `MockDataset`, `XRVDataset` |
| `SaliencyProvider` | `heatmap(image, label_index)` | `GradCAMSaliency` (all-NaN = not localizable) |
| `LocalizationDataset` | | `ChestXDetDataset` |
| `LocalizationMetric` | `score(heatmap, gt_mask)` | `PointingGame`, `EnergyInMask`, `IoU`, `Dice` |

`KPIBundle` (`metrics/kpi.py`) is a second Protocol: `SystemBundle`, `ActorsBundle`,
`TeamBundle`, `JointTableBundle`.

### Module map — `src/hai_sim/`

```
core/protocols.py           the 9 Protocols = the public contract
core/types.py               frozen dataclasses: LabelSchema, Diagnosis, ModelOutput,
                            Patient(Context), Localization/DatasetExample, DatasetCapabilities
core/base.py                BaseDiagnosticActor ABC
cli.py                      `hai-sim` entrypoint → run | study | validate
factory.py                  ScenarioConfig → runtime objects (all the registries live here)
labels.py                   LabelProjection / ProjectedActor — remap actor schema → dataset schema
personas.py                 PERSONA_LIBRARY: the 2×3 capability × attitude grid
config/scenario.py          strict pydantic models + load_scenario / load_validation_scenario
models/{mock,torchxrv,cached}.py     the three AI backends
doctors/{mock,simple,assisted}.py    the three doctors + the 3 integration policies
sort/{random,uncertainty}.py         the two triage policies
uq/methods.py                        EntropyUQ, EnsembleUQ
severity/models.py                   the four severity models
data/{mock,xrv}.py                   MockDataset | XRVDataset ({1,0,-1} sentinel convention)
sim/engine.py                        the SimPy engine
sim/{patient,dataset,cache}_generator.py   the three patient streams
metrics/collector.py                 PatientRecord + MetricsCollector (CSV telemetry)
metrics/kpi.py                       the four KPI bundles
study/{spec,runner}.py               factor grid → tidy table
analysis/inference_cache.py          resumable per-member probability cache
analysis/calibration.py              per-pathology threshold fitting
analysis/error_curve.py              error-capture curves, AUROC, dashboards (the RQ1 apparatus)
analysis/localization/*              cache, cam, chestxdet, evaluate, metrics, report,
                                     saliency, validator
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
| `build_inference_cache.py` | run the ensemble once over a dataset → per-member probabilities (checkpointed, resumable) |
| `build_localization_cache.py` | forward-only feature maps (~2.8 GB fp32) for offline Grad-CAM |
| `calibrate.py` | per-pathology thresholds (recall-target and cost-weighted), held-out evaluation |
| `compare_aggregator.py` | mean vs max triage collapse |
| `compare_single_vs_ensemble.py` | single model vs Deep Ensemble |
| `plot_dashboard.py` | the RQ1 figures |
| `download_nih_sample.py` / `download_chestxdet.py` / `download_reader_labels.py` | data |

---

## Determinism

A single `seed` drives the whole run. Internally `np.random.SeedSequence(seed)` is spawned
into 5 independent sub-streams (patient generation, AI, doctor, sort, and a reserved fifth)
so that **swapping one module does not perturb the others**. Same seed + same scenario file
= byte-identical CSV/JSON. In studies, `replications` are seeds, giving each arm common
random numbers.

## Tests

188 test functions across 30 files. `uv run pytest`.

Largest: `test_severity` (20), `test_labels` (16), `test_localization_metrics` (11),
`test_personas` (10), `test_integration_policy` (10), `test_study` / `test_dataset` (9),
`test_uq` / `test_kpi_report` / `test_error_curve` (8). `test_protocol_compliance.py`
asserts runtime conformance to the Protocols. `test_real_models.py` is gated behind the
`models` extra.

**Coverage gaps**: no direct tests for `cli.py`, `factory.py`, or `config/scenario.py`;
`sim/engine.py` is covered only by `test_engine_smoke.py`.

## Not implemented yet

See [`../TESI.md`](../TESI.md) §8 for the roadmap and the reasoning.

| Missing | Note |
|---|---|
| **`SDTDoctor`** + shared case difficulty `δ` / coupling `ρ` | the priority; today the doctor is `FixedAccuracyDoctor` (fixed sens/spec, Bernoulli on GT) and the headline `ρ`-sweep experiment cannot run |
| **Temperature scaling** | `analysis/calibration.py` only fits thresholds. **Blocking for `log_odds_fusion`** — without calibration the fusion can be worse than the doctor alone |
| **Reader-calibrated personas** | `build_capability` **raises** on `source="reader"`; `DatasetCapabilities` advertises `has_per_reader` but nothing is wired. The NIH `individual_readers.csv` is Google-Form-gated, so only adjudicated consensus labels are obtainable unattended |
| **Feature-space UQ** | Mahalanobis / DDU / deep k-NN / energy on the penultimate embedding — derivable from the localization cache at zero inference cost (RQ2) |
| **MedGemma-4B backend** | the planned second AI actor |
| Fatigue, trust dynamics, MRMC reader population | optional realism modulators, after the core |
| NHPP arrivals | today homogeneous Poisson only |
| MC Dropout, X-ray machine model | future comparison / scope extension |
