# Scenario parameters — what to touch, what changes, what to expect

A scenario is a single YAML file validated by `pydantic` (`src/hai_sim/config/scenario.py`)
with `extra = forbid`: **any unknown key raises**, so typos fail loudly. Every
field below is optional unless marked *required*; defaults are shown.

This file is the experiment control panel. The golden rule for clean comparisons:
**change ONE block at a time, keep `seed` fixed**. The seed is split into
independent sub-streams (patient / ai / doctor / sort), so changing the sort
policy does not perturb which patients or images are drawn — only the ordering.

```yaml
name: my_scenario          # str  — free text, appears in output filenames
seed: 42                   # int  — drives every stochastic stream; fix it across a comparison
label_schema: xrv          # "default_v0" (5 fake) | "chexpert" (14) | "xrv" (18, real model)
patient_generator: { ... }
ai: { ... }
doctor: { ... }
sort: { ... }
n_ai_workers: 1            # int >= 1 — concurrent AI inference workers
n_doctor_workers: 1        # int >= 1 — concurrent doctors
```

---

## Quick map — which knob answers which question

| Experimental question | Block / field to vary | Keep fixed |
|---|---|---|
| **Q-Model**: does the AI help? | `ai.kind` (mock vs torchxrv), `doctor.use_ai_hint` | seed, doctor, patients |
| **Q-Order**: does triage order matter? | `sort.kind`, `sort.aggregator` | seed, ai, doctor, patients |
| **Q-Doctor**: how does the reader model change outcomes? | `doctor.*` (sensitivity/specificity, kind) | seed, ai, patients |
| **UQ comparison (Q3)** | `ai.uq`, `ai.ensemble_measure`, `ai.weights` | seed, everything else |
| **Operating point** | `ai.decision_threshold` | seed, everything else |
| **Load / saturation** | `patient_generator.arrival_rate`, `n_doctor_workers`, `service_time*` | seed, ai, doctor quality |

---

## `label_schema` (top level)

| value | n labels | use |
|---|---|---|
| `default_v0` | 5 (`L0..L4`) | mock-only smoke tests |
| `chexpert` | 14 | reserved for CheXpert-native models |
| `xrv` | 18 | **required for `ai.kind: torchxrv`** (model outputs 18 pathologies in this exact order) |

> Constraint: `torchxrv` raises unless `label_schema: xrv`. The mock model/doctor
> work with any schema.

---

## `patient_generator` — the patient stream (the "input")

```yaml
patient_generator:
  kind: poisson | xrv_dataset      # default poisson
  arrival_rate: 0.5                # REQUIRED, float > 0
  n_patients: 200                  # REQUIRED, int > 0
```

| param | type / default | what it controls | what to expect when you raise it |
|---|---|---|---|
| `arrival_rate` (λ) | float > 0 | Poisson inter-arrival rate (patients per sim-time unit) | more patients arrive per unit time → **queue saturates faster** → longer `mean_wait_to_doctor`; the *order* of service matters more |
| `n_patients` | int > 0 | total patients emitted, then stop | more → tighter metric estimates, longer wall-clock (1 forward pass each, ×5 for ensemble) |

**`kind: poisson`** (synthetic, no model needed — for wiring/load tests):

| param | default | effect |
|---|---|---|
| `prevalence` | `0.2` (scalar) or per-label list | P(label = 1) in synthetic ground truth. Higher → more positives → recall easier to read, accuracy less inflated |
| `image_shape` | `[1, 1]` | placeholder array shape (mocks ignore pixels) |

**`kind: xrv_dataset`** (real CXR images — the realistic patient generator):

| param | default | effect |
|---|---|---|
| `dataset` | `nih` \| `rsna` \| `chexpert` | which TorchXRayVision loader; sets the available ground-truth labels (unlabeled → `-1`, skipped by metrics) |
| `imgpath` | `null` (**required**) | folder of images, e.g. `data/nih_sample/images` (repo-relative; run from repo root) |
| `csvpath` | `null` | labels csv. Required for `chexpert`; for `nih` the sample's `Patient Age` is auto-cleaned |
| `sample_mode` | `sequential` \| `random` | `random` draws with replacement (good when `n_patients` > dataset size); `sequential` wraps around in order |

---

## `ai` — the AI model (the actor under test)

```yaml
ai:
  kind: random_mock | torchxrv     # default random_mock
  service_time: 0.05               # float >= 0, sim-time per inference (NOT wall-clock)
```

`service_time` is the **simulated** cost of one inference. It does not change
predictions — only queueing dynamics. Raise it and the AI station becomes a
bottleneck (rare unless many patients / few workers).

**`kind: random_mock`** — sanity actor, predictions ≈ random (wiring tests only):

| param | default | effect |
|---|---|---|
| `alpha_conf`, `beta_conf` | `2.0`, `2.0` | Beta(α,β) for per-label confidence; threshold at 0.5 → label |
| `alpha_unc`, `beta_unc` | `2.0`, `5.0` | Beta(α,β) for per-label uncertainty (mean ≈ 0.29) |

**`kind: torchxrv`** — the real pre-trained DenseNet121 (inference only):

| param | default | values / range | what it controls — what to expect |
|---|---|---|---|
| `weights` | `densenet121-res224-all` | one name (single model) **or a list** (Deep Ensemble) | a list of dataset-specific weights (`-nih`, `-chex`, `-mimic_ch`, `-pc`, `-rsna`) forms a cross-domain ensemble: better calibrated, fewer FP, **but ×N slower** (N forward passes) |
| `uq` | `entropy` | `entropy` \| `ensemble` | `entropy` = 1 pass, uncertainty from the model's own confidence (aleatoric proxy). `ensemble` = real epistemic disagreement across members. Use `ensemble` only with multiple `weights` |
| `ensemble_measure` | `mutual_information` | `mutual_information` \| `predictive_entropy` \| `mean_entropy` \| `std` | which per-label uncertainty the ensemble exposes. `mutual_information` = epistemic (member disagreement); `predictive_entropy` = total; `mean_entropy` = aleatoric; `std` = raw spread. This is the number the sort triages on |
| `decision_threshold` | `0.5` | [0, 1] | probability cutoff → binary label. **Raise it → fewer positives → fewer FP, more FN** (precision up, recall down). The single biggest lever on the FP/FN trade-off; 0.5 over-predicts on xrv models |
| `service_time` | `0.05` | float >= 0 | sim-time per inference (queueing only) |
| `device` | `cpu` | `cpu` \| `cuda` | inference device; does not change results |

> Available DenseNet121 weight names: `densenet121-res224-all`, `-nih`, `-chex`,
> `-mimic_ch`, `-mimic_nb`, `-pc`, `-rsna`. Weights auto-download to
> `~/.torchxrayvision/` on first use.

---

## `doctor` — the digital twin (the reader)

```yaml
doctor:
  kind: random_mock | fixed_accuracy   # default random_mock
  service_time_mu: 2.5                  # lognormal mu (log-space)
  service_time_sigma: 0.4               # lognormal sigma (log-space)
```

`service_time_*` set the **reading time** distribution: median ≈ `exp(mu)`, mean
≈ `exp(mu + sigma^2/2)`. The doctor is usually the system bottleneck, so raising
`mu` lengthens queues and makes the sort order matter more.

**`kind: fixed_accuracy`** — the trivial controllable reader (recommended baseline):

| param | default | values | what it controls — what to expect |
|---|---|---|---|
| `sensitivity` | `0.9` | scalar in [0,1] **or** a per-label list | P(reports positive \| truly positive). **Lower → more FN** (missed disease). Set per-label to make some pathologies harder |
| `specificity` | `0.9` | scalar in [0,1] **or** per-label list | P(reports negative \| truly negative). **Lower → more FP** (over-call) |

> The engine injects ground truth into the doctor's context as a *simulation
> device*; `fixed_accuracy` flips the truth at these rates. It ignores `ai_hint`.

**`kind: random_mock`** — random reader, optionally AI-assisted (for Q-Model wiring):

| param | default | effect |
|---|---|---|
| `alpha_conf`, `beta_conf` | `5.0`, `2.0` | Beta for confidence (mean ≈ 0.71, decisive) |
| `alpha_unc`, `beta_unc` | `2.0`, `5.0` | Beta for uncertainty |
| `use_ai_hint` | `false` | if true, the AI's `ModelOutput` is passed to the doctor |
| `assisted_policy` | `noop` \| `follow_when_certain` | `follow_when_certain` copies the AI's label per pathology when AI uncertainty < threshold |
| `ai_hint_uncertainty_threshold` | `0.3` (in [0,1]) | below this AI uncertainty the doctor defers to the AI |

---

## `sort` — the triage policy (how the queue is ordered)

```yaml
sort:
  kind: random | uncertainty_desc    # REQUIRED (no default)
  aggregator: mean                   # only for uncertainty_desc
  weights: null                      # only for aggregator=weighted
```

| param | values | what it controls — what to expect |
|---|---|---|
| `kind` | `random` \| `uncertainty_desc` | `random` = baseline (uniform key). `uncertainty_desc` = serve highest-uncertainty patients first. The **mean** queue wait is identical either way (permutation-invariant); what changes is *which* patients are seen before the (saturated) system runs out of doctor time |
| `aggregator` | `mean` \| `max` \| `weighted` | collapses the 18-label uncertainty vector to one scalar key. **`mean`** dilutes a single critical finding among 17 confident ones; **`max`** escalates a patient with any one very-uncertain pathology; **`weighted`** weights by clinical importance. Different aggregators reorder the queue substantially (often <20% overlap in the top-served) |
| `weights` | list of `n_labels` floats | required when `aggregator: weighted`; per-pathology importance (e.g. Pneumothorax >> Nodule) |

> Because the full per-label uncertainty vector is saved to the CSV, you can
> re-triage offline with any aggregator **without re-running inference**.

---

## `n_ai_workers` / `n_doctor_workers` (top level)

| param | default | effect |
|---|---|---|
| `n_ai_workers` | 1 | concurrent AI inference stations. Raise to clear an AI bottleneck (rarely needed; mock/cpu AI is fast) |
| `n_doctor_workers` | 1 | concurrent doctors. **Raise → less saturation → shorter `mean_wait_to_doctor`**, and the triage order matters *less* (more patients get seen anyway) |

---

## Gotchas (validation will reject these)

- `ai.kind: torchxrv` **requires** `label_schema: xrv`.
- `patient_generator.kind: xrv_dataset` **requires** `imgpath` (and `csvpath` for `chexpert`).
- `sort.aggregator: weighted` **requires** `sort.weights` of length `n_labels`.
- `ai.uq: ensemble` is only meaningful with **multiple** `ai.weights`.
- Unknown keys, out-of-range thresholds (`[0,1]`), or `arrival_rate <= 0` raise.

---

## Recipes — minimal diffs for each comparison

**Q-Order (mean vs max triage)** — copy a scenario twice, change only:
```yaml
# A:                      # B:
sort: {kind: uncertainty_desc, aggregator: mean}
sort: {kind: uncertainty_desc, aggregator: max}
```
Expect: same global accuracy/F1, but a different set of patients served before
saturation → differences in *time-to-FN* and which FN are caught early.

**Operating point (FP/FN trade-off)** — sweep one field:
```yaml
ai: {kind: torchxrv, ..., decision_threshold: 0.5}   # -> many FP, low precision
ai: {kind: torchxrv, ..., decision_threshold: 0.8}   # -> fewer FP, more FN, higher precision
```

**UQ method (Q3)** — single vs ensemble, same patients:
```yaml
ai: {kind: torchxrv, weights: densenet121-res224-all, uq: entropy}
ai: {kind: torchxrv, weights: [..nih, ..chex, ..mimic_ch, ..pc, ..rsna], uq: ensemble, ensemble_measure: mutual_information}
```

**Q-Doctor (junior vs senior reader)**:
```yaml
doctor: {kind: fixed_accuracy, sensitivity: 0.75, specificity: 0.85}   # junior
doctor: {kind: fixed_accuracy, sensitivity: 0.92, specificity: 0.95}   # senior
```

**Q-Model (with vs without AI assist)** — random_mock doctor:
```yaml
doctor: {kind: random_mock, use_ai_hint: false}
doctor: {kind: random_mock, use_ai_hint: true, assisted_policy: follow_when_certain}
```

---

## Reading the result of a change

After each run, compare in `out/summary_*.json` (see the main `README.md` for
full KPI definitions):
- **`recall_macro` / `n_fn`** → the clinical axis (missed disease). Watch these first.
- **`precision_macro` / `n_fp`** → the workload axis (over-calls).
- **`mttfn` / `ttfe`** → *when* errors accumulate — the axis triage order moves.
- **`mean_wait_to_doctor` / `throughput`** → saturation; tells you if load knobs bit.

Do not judge a change by `f1_macro` or `accuracy` alone — always decompose into
recall (FN) vs precision (FP).
