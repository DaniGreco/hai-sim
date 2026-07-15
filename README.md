# hai_sim — Human-AI Diagnostic Workflow Simulator

Modular SimPy-based simulator of a hybrid human-AI diagnostic department.
Part of the master thesis of Daniele Greco (UniBg).

## Pipeline

```
PatientGenerator → input_queue (Store, FIFO)
                 → ModelStation (× n_ai_workers)
                 → triaged_store (PriorityStore, key=SortStrategy)
                 → DoctorStation (× n_doctor_workers)
                 → MetricsCollector
```

All actors (AI model, doctor digital twin) implement the same `DiagnosticActor`
protocol — `diagnose(images, *, context, ai_hint) → ModelOutput` — so any
implementation can be swapped via the YAML scenario config. The doctor may
optionally consult the AI's `ModelOutput` as a hint (see *Hybrid mode* below).

## Scope

**v0 — mocks (no extra deps):**
- `RandomMockAI`, `RandomMockDoctor`, `PoissonPatientGenerator`
- Sort strategies: `RandomSort`, `UncertaintyDescSort(mean|max|weighted)`
- One assisted policy: `follow_when_certain` (per-label deferral on AI uncertainty)
- Metrics: TTFE, MTTF, mean-time-to-FN, F1 macro per actor (AI / doctor / hybrid)

**v1 — real models (`[models]` extra):**
- `TorchXRayVisionAI`: inference-only DenseNet121 (18 pathologies), single weight
  set or a cross-domain Deep Ensemble. AUC > 0.90 on several pathologies
  (Edema, Effusion, Cardiomegaly, Pneumothorax, ...).
- Swappable `UQMethod`: `EntropyUQ` (single pass) and `EnsembleUQ` (predictive
  entropy / mean entropy / mutual information decomposition).
- `FixedAccuracyDoctor`: trivial ground-truth-based reader with fixed
  sensitivity/specificity — the controllable baseline twin.
- `XRVDatasetPatientGenerator`: real CXR images as the patient stream
  (NIH / RSNA / CheXpert via TorchXRayVision loaders). Unannotated labels → `-1`,
  skipped by the metrics.

**NOT yet**: fatigue/SDT composite doctor, MC Dropout, X-ray machine model,
multi-doctor calibration, AI+doctor fusion policy.

## Real models (v1)

```powershell
# install the heavy extra (CPU torch keeps it light)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[models]"

# 1. get real images into data/nih_sample/ (needs a Kaggle API token; see data/README.md)
uv run python scripts/download_nih_sample.py

# 2. run from the repo root — single-model entropy UQ, or the cross-domain Deep Ensemble
uv run hai-sim run configs/examples/v1_nih_torchxrv.yaml
uv run hai-sim run configs/examples/v1_nih_ensemble.yaml
```

Model weights (~80 MB each) auto-download to `~/.torchxrayvision/` on first use.
On Windows set `PYTHONUTF8=1` so the download progress bar doesn't crash the
console (`cp1252` can't render its block characters).

## Quickstart

```powershell
# from C:\Projects\tesi\hai_sim
uv venv
uv pip install -e ".[dev]"

# Tests
uv run pytest

# Smoke runs — three scenarios with the same seed for direct comparison
uv run hai-sim run configs/examples/v0_smoke_random.yaml
uv run hai-sim run configs/examples/v0_smoke_uncertainty.yaml
uv run hai-sim run configs/examples/v0_smoke_hybrid.yaml
```

Output: `out/records_<scenario>_<ts>.csv` (per-patient telemetry) and
`out/summary_<scenario>_<ts>.json` (KPI block + scenario dump). Both are
gitignored.

---

## Output

### CSV — per-patient telemetry

One row per patient. Columns:

| Column | Meaning |
|---|---|
| `patient_id` | sequential id assigned at generation |
| `ts_arrival` | simulated time when the patient entered `input_queue` |
| `ts_model_start` | simulated time when the AI worker picked it up |
| `ts_model_end` | simulated time when the AI's `diagnose` returned |
| `ts_doctor_start` | simulated time when the doctor picked it from `triaged_store` |
| `ts_doctor_end` | simulated time when the doctor's verdict was recorded |
| `priority_key` | key used by the `PriorityStore` (lower → served earlier) |
| `gt__<label>` (× N labels) | ground truth bit per label (0/1) |
| `ai_label__<label>`, `ai_conf__<label>`, `ai_unc__<label>` | AI prediction, confidence, uncertainty per label |
| `doc_label__<label>`, `doc_conf__<label>`, `doc_unc__<label>` | doctor's prediction, confidence, uncertainty per label |

Any downstream KPI can be re-derived from this CSV alone.

### JSON — run summary

```jsonc
{
  "scenario": { /* validated scenario config dump */ },
  "ai": "RandomMockAI",
  "doctor": "RandomMockDoctor",
  "sort": "UncertaintyDescSort(mean)",
  "wall_clock_s": 0.07,           // Python wall-clock; NOT simulated time
  "metadata": {
    "sim_end_time": 1285.24,
    "n_ai_workers": 1,
    "n_doctor_workers": 1
  },
  "kpis": { /* see below */ }
}
```

### KPI definitions

Computed in `src/hai_sim/metrics/kpi.py` and printed by the CLI.

**Run-level (`RunKPIs`)** — system metrics, independent of which actor produced the verdict:

| KPI | Definition | Units / range |
|---|---|---|
| `n_patients` | patients fully processed end-to-end | int |
| `sim_duration` | `max(ts_doctor_end)` — when the last verdict was recorded | sim time |
| `throughput` | `n_patients / sim_duration` — compare against arrival rate λ to detect saturation | patients / sim-time |
| `mean_wait_to_model` | mean of `ts_model_start − ts_arrival` — time spent in input queue | sim time |
| `mean_wait_to_doctor` | mean of `ts_doctor_start − ts_model_end` — time spent in triaged queue. **Sum is invariant under permutation of the queue**, so the *mean* doesn't change with the sort policy (per-patient distribution does) | sim time |
| `mean_time_in_system` | mean of `ts_doctor_end − ts_arrival` — end-to-end latency | sim time |

**Per-actor (`ActorKPIs`)** — computed three times, once each for `ai`, `doctor`, `hybrid`:

| KPI | Definition | Range / units | Clinical reading |
|---|---|---|---|
| `f1_macro` | mean over labels of F1 = 2·P·R/(P+R), per-label | [0, 1] | primary classification KPI; macro avg gives every pathology equal weight |
| `precision_macro` | mean over labels of TP/(TP+FP) | [0, 1] | "when the actor says positive, how often is it right?" |
| `recall_macro` | mean over labels of TP/(TP+FN) — aka sensitivity | [0, 1] | "of truly positive labels, how many does the actor catch?" Low recall ⇒ many FN |
| `accuracy` | (ΣTP + ΣTN) / total (patient × label) entries | [0, 1] | misleading at low prevalence (predicting all-zero scores high). Reported for completeness |
| `n_fn` | total FN events = (patient, label) pairs where gt=1, pred=0 | int | **clinically the worst kind of error** — missed pathology |
| `n_fp` | total FP events = pred=1, gt=0 | int | over-call / false alarm |
| `ttfe` | Time To First Error: simulated timestamp of the first patient on which the actor mispredicts *any* label | sim time / `None` | how long before the system slips up; `None` means flawless run |
| `mttf` | Mean Time To Failure: mean gap between consecutive error events on the timeline | sim time / `None` | reliability-engineering: large ⇒ errors well spread; small ⇒ bursty. `None` if < 2 errors |
| `mttfn` | mean gap between consecutive **false-negative** events | sim time / `None` | the FN-focused twin of MTTF; the one radiology actually cares about |

**Definition of "error" used by `ttfe`/`mttf`**: a patient counts as an error if the actor's predicted label vector differs from the ground truth on **at least one label** (hamming-style, not exact-match). The event time used is `ts_doctor_end` — the moment the system's verdict on that patient became known.

**The three actor blocks**:
- **`ai`** — score the AI's label vector against ground truth. Baseline: "what would the model deployed alone do?"
- **`doctor`** — the *unaided* (solo) doctor call. When the doctor is an `AssistedDoctor`, this is the counterfactual it stashes (`doctor_solo_labels`); otherwise it is simply the doctor's output.
- **`hybrid`** — the doctor stage's *actual* output = the *aided* (team) call when AI-assisted, else identical to `doctor`.

**Team block (`TeamKPIs`)** — human-AI team quality vs its two members. Because the simulator knows both counterfactuals (doctor-solo, AI-solo) for every case, the appropriate-reliance metrics — normally needing a two-stage user study — are computed directly:

| KPI | Definition | Reading |
|---|---|---|
| `complementarity_gap_f1` / `_accuracy` | team − max(AI-solo, doctor-solo) | > 0 ⇒ the team beats the better of its two members ("the whole exceeds its parts") |
| `rair` | Relative AI Reliance: of entries where doctor-solo wrong & AI right, fraction the team got right | did the team *harvest* correct AI? (Schemmer et al. 2023) |
| `rsr` | Relative Self-Reliance: of entries where doctor-solo right & AI wrong, fraction the team got right | did the team *resist* wrong AI? |
| `over_reliance` / `under_reliance` | `1 − rsr` / `1 − rair` | agency lost to wrong AI / correct AI ignored |

See `LITERATURE_doctor_ai_integration.md` for the model + policy design behind these.

---

## Writing a scenario YAML

A scenario is a single YAML file validated by `pydantic` (`extra = forbid`, so
unknown keys raise). All optional fields can be omitted; defaults are listed.

### Top-level skeleton

```yaml
name: my_scenario                 # str, free text; appears in output filenames
seed: 42                          # int; controls all stochastic streams
label_schema: default_v0          # "default_v0" (5 fake labels) | "chexpert" (14)
patient_generator: { ... }        # see below
ai: { ... }
doctor: { ... }
sort: { ... }
n_ai_workers: 1                   # int ≥ 1; concurrent AI inference workers
n_doctor_workers: 1               # int ≥ 1; concurrent doctors
```

### `patient_generator`

```yaml
patient_generator:
  kind: poisson                   # only "poisson" supported in v0
  arrival_rate: 0.5               # float > 0; Poisson λ in patients per sim-time unit
  n_patients: 100                 # int > 0; total patients to emit, then stop
  prevalence: 0.2                 # float in [0,1] OR a list of length N_labels
                                  #   scalar → same Bernoulli prob for every label
                                  #   list   → per-label prior probability of gt=1
  image_shape: [1, 1]             # tuple of ints; placeholder array shape for mocks
```

### `ai`

```yaml
ai:
  kind: random_mock               # only "random_mock" in v0
  service_time: 0.05              # float ≥ 0; constant inference time
  alpha_conf: 2.0                 # float > 0; Beta(α, β) for per-label confidence
  beta_conf: 2.0                  # float > 0
  alpha_unc: 2.0                  # float > 0; Beta(α, β) for per-label uncertainty
  beta_unc: 5.0                   # float > 0; default (2,5) → mean ≈ 0.29
```

The mock samples confidence and uncertainty from independent Beta distributions
per label and thresholds confidence at 0.5 to produce the binary label. Decision
quality is therefore close to uniform random by design — this is a wiring test,
not a clinical baseline.

### `doctor`

```yaml
doctor:
  kind: random_mock               # only "random_mock" in v0
  service_time_mu: 2.5            # float; lognormal μ (log-space)
  service_time_sigma: 0.4         # float > 0; lognormal σ (log-space)
                                  #   median = exp(μ), mean = exp(μ + σ²/2)
                                  #   defaults → median ≈ 12.2 sim-time, mean ≈ 13.2
  alpha_conf: 5.0                 # Beta α for confidence (default mean ≈ 0.71)
  beta_conf: 2.0
  alpha_unc: 2.0                  # Beta for uncertainty (default mean ≈ 0.29)
  beta_unc: 5.0

  # Hybrid mode — see "Hybrid mode" section below
  use_ai_hint: false              # bool; if true, the AI's ModelOutput is passed to diagnose()
  assisted_policy: noop           # "noop" | "follow_when_certain"
  ai_hint_uncertainty_threshold: 0.3   # float in [0,1]; only used by "follow_when_certain"
```

### `sort`

```yaml
sort:
  kind: random                    # "random" | "uncertainty_desc"
  aggregator: mean                # only used when kind=uncertainty_desc:
                                  #   "mean" | "max" | "weighted"
  weights: null                   # required when aggregator=weighted; list of N floats
                                  # ignored otherwise
```

- `random`: assigns uniform-random priority keys; baseline against which any
  triage policy should beat.
- `uncertainty_desc`: collapses the per-label uncertainty vector to a scalar
  via the chosen aggregator and serves high-uncertainty patients first. The
  underlying hypothesis (high uncertainty ↔ clinical complexity) is precisely
  what the experiment harness will test.

### Determinism

A single `seed` drives the whole run. Internally `np.random.SeedSequence(seed)`
is spawned into 5 independent sub-streams (one each for patient generation, AI,
doctor, sort, and a reserved fifth) so that **swapping one module does not
perturb the others**. Same seed + same scenario file = byte-identical CSV/JSON.

---

## Hybrid mode (Human-AI assisted)

By default the doctor sees only the images. To make the doctor consult the AI's
output, set:

```yaml
doctor:
  use_ai_hint: true               # enables passing ai_hint=ModelOutput into doctor.diagnose
  assisted_policy: follow_when_certain
  ai_hint_uncertainty_threshold: 0.3
```

### What each policy does

- **`noop`** *(default)* — the AI's `ModelOutput` is recorded in the doctor's
  output `extra` dict (`ai_hint_seen=True`) but the doctor's prediction is
  unchanged. Useful for verifying the wiring without altering KPIs.
- **`follow_when_certain`** — per-label deferral. On each label `i`, if the
  AI's `uncertainty[i] < ai_hint_uncertainty_threshold`, the doctor copies the
  AI's label for that label; otherwise the doctor uses its own sampled call.
  The list of labels where the AI was followed lands in `extra["ai_hint_followed_labels"]`.

This is intentionally a **minimal** policy: enough to make `use_ai_hint=true`
move the KPI needle, not enough to be clinically realistic. Richer fusion
strategies (full deferral, weighted vote, defer-on-doctor-uncertainty, etc.)
will arrive as standalone components when the experiment campaign requires them.

### Example: comparing baseline vs hybrid

`configs/examples/v0_smoke_uncertainty.yaml` and `configs/examples/v0_smoke_hybrid.yaml`
share the same seed, sort, and arrival pattern; the only differences are
`use_ai_hint` and `assisted_policy`. Run both and compare the `doctor` KPI block
to see exactly how much consulting the AI moves the needle (with a random-mock
AI, usually downward — that's expected because the mock is bad).

### Note on the `hybrid` KPI block in v0

When `use_ai_hint=true`, the doctor's recorded `doc_labels` *already* reflect
the assisted decision. The `hybrid` KPI block continues to mirror the `doctor`
block in v0 — they will diverge once a `FusionPolicy` component is introduced
that combines AI and doctor outputs *after* both have been recorded
independently. Until then, the cleanest comparison is:
`use_ai_hint=false` doctor block  vs  `use_ai_hint=true` doctor block.
