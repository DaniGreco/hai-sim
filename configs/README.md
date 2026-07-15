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
persona: senior_trusting   # a persona library name, OR an inline capability/visibility/trust block
sort: { ... }
severity: { ... }          # optional — a per-patient severity model (produced & logged)
label_view: { ... }        # optional — project labels (subset / binary-abnormal)
kpis: [system, actors, team, joint_table]  # optional — which KPI bundles to compute (default = all)
n_ai_workers: 1            # int >= 1 — concurrent AI inference workers
n_doctor_workers: 1        # int >= 1 — concurrent doctors
```

---

## Quick map — which knob answers which question

| Experimental question | Block / field to vary | Keep fixed |
|---|---|---|
| **Q-Model**: does the AI help? | `persona.sees_ai` (solo vs assisted) | seed, persona, patients |
| **Q-Order**: does triage order matter? | `sort.kind`, `sort.aggregator` | seed, ai, persona, patients |
| **Q-Doctor / RQ3**: how do personas change team outcome & reliance? | `persona` (library name, or capability×visibility×trust) | seed, ai, patients |
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
  kind: poisson | mock_dataset | xrv_dataset   # default poisson
  arrival_rate: 0.5                # REQUIRED, float > 0
  n_patients: 200                  # REQUIRED, int > 0
  gt_source: native                # ground truth is owned by the dataset (default 'native')
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

**`kind: mock_dataset`** (synthetic images through the full `Dataset` contract — like
`poisson` but a capability-aware dataset pool; exercises the dataset-agnostic path, no torch):

| param | default | effect |
|---|---|---|
| `prevalence` | `0.2` (scalar) or per-label list | P(label = 1) in the synthetic pool |
| `n_examples` | `100` | size of the dataset pool; `n_patients` are drawn from it (with `sample_mode`) |
| `image_shape` | `[1, 1]` | placeholder array shape |

**`kind: xrv_dataset`** (real CXR images — the realistic patient generator):

| param | default | effect |
|---|---|---|
| `dataset` | `nih` \| `rsna` \| `chexpert` | which TorchXRayVision loader; sets the available ground-truth labels (unlabeled → `-1`, skipped by metrics) |
| `imgpath` | `null` (**required**) | folder of images, e.g. `data/nih_sample/images` (repo-relative; run from repo root) |
| `csvpath` | `null` | labels csv. Required for `chexpert`; for `nih` the sample's `Patient Age` is auto-cleaned |
| `sample_mode` | `sequential` \| `random` | `random` draws with replacement (good when `n_patients` > dataset size); `sequential` wraps around in order |
| `gt_source` | `native` | which ground-truth source the dataset provides; only sources the dataset *advertises* are accepted (**capability-gated**). Alternatives (NIH adjudicated consensus, per-reader) land with the real-data work |

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

## `persona` — the doctor model (capability × visibility × trust)

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
    source: fixed                   # "fixed" | "reader" (calibrate on real reads — gated on dataset)
  sees_ai: true                     # visibility: does the reader consult the AI?
  trust:                            # used only when sees_ai — a swappable IntegrationPolicy
    kind: linear_pool               # "linear_pool" | "confidence_gated" | "log_odds_fusion"
    weight_ai: 0.45
```

**capability** (`fixed_accuracy`): `sensitivity` ↓ → more FN (missed disease);
`specificity` ↓ → more FP (over-call); both scalar or per-label. `random_mock` is the
random wiring reader. `service_time_*` set the reading-time lognormal (the doctor is
usually the bottleneck → raising `mu` lengthens queues, makes sort order matter more).
`source: reader` calibrates capability on a real radiologist's reads — **gated** on the
dataset advertising per-reader reads (raises until wired).

**visibility** (`sees_ai`): `false` = solo reader (medico-solo baseline); `true` = consults the AI.

**trust** (the swappable integration policy, `sees_ai: true` only):

| `trust.kind` | key params | behaviour |
|---|---|---|
| `linear_pool` | `weight_ai` (Weight of Advice, [0,1]) | weighted average `(1-w)·doctor + w·AI`. Lit mean WOA ≈0.39 |
| `confidence_gated` | `weight_ai`, `confidence_threshold` | apply WOA only where AI confidence `max(p,1-p)` ≥ threshold; else ignore the AI |
| `log_odds_fusion` | `beta`, `weight_doctor`, `prior` | normative Bayesian pool; agreeing opinions reinforce; AI down-weighted by its uncertainty (`exp(-beta·H)`) |

> The library grid *{senior, junior} × {solo, trusting, skeptical}* is the RQ3 sweep:
> capability senior 0.95/0.95 vs junior 0.85/0.88; trust WOA senior < junior at each
> attitude (experts discount advice more). Read the effect off the `team` KPI block and
> the `joint_table`. See `LITERATURE_doctor_ai_integration.md`.

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

## `severity` (optional) — a per-patient severity signal

A swappable module that computes a per-patient severity, **logged in telemetry whether or
not anything uses it** (triage/cost consumers are opt-in). Omit the block → no severity.

```yaml
severity:
  kind: activation            # "constant" | "activation" | "image_quality" | "embedding"
  source: ai_probs            # activation source: ai_probs | ai_binary | ai_uncertainty | gt
  criticality: [ ... ]        # optional per-label clinical weights (length n_labels)
  aggregator: max             # max | sum | mean → scalar   (or mode: vector → per-label)
  mode: scalar
```

`activation` = `aggregate_l(criticality_l × activation_l)`: criticality is a pre-assigned
per-label weight, activation a per-image signal. `image_quality` = blur/intensity proxy;
`embedding` = a reduction of the model's feature vector (auto-enables the AI's embedding).

---

## `label_view` (optional) — single- vs multi-label studies

Projects the model's native labels onto the **active** study schema (AI, doctor, GT and the
KPIs all follow it). Omit → full multi-label.

```yaml
label_view:
  mode: binary_abnormal       # "full" | "subset" | "binary_abnormal"
  targets: [Pneumothorax]     # subset: which labels (one target = a single-label study)
  combine: noisy_or           # binary_abnormal: noisy_or | max (how AI probs collapse to "Abnormal")
```

---

## `kpis` (optional) — which KPI bundles to compute

```yaml
kpis: [system, actors, team, joint_table]   # default = all four
```

Swappable bundles: `system` (throughput/waits), `actors` (per-actor AI/doctor/hybrid
F1/sens/spec/FN/FP), `team` (complementarity gap, RAIR/RSR), `joint_table` (the AI×Doctor×GT
contingency: 2 confusion matrices + 8 GT-signed cells).

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
- `persona.capability.source: reader`, and any non-`native` `gt_source`, raise unless the dataset advertises that capability.
- An unknown `persona` library name, a `kpis` bundle outside {system, actors, team, joint_table}, or `severity.criticality` of the wrong length, raise.
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

**Q-Doctor / RQ3 (persona grid)** — vary only the persona:
```yaml
persona: solo_senior          # senior, no AI (the medico-solo baseline)
persona: senior_trusting      # senior + AI, higher Weight of Advice
persona: senior_skeptical     # senior + AI, egocentric discounting
```

**Q-Model (with vs without AI)** — same capability, flip visibility:
```yaml
persona: {capability: {kind: fixed_accuracy, sensitivity: 0.85, specificity: 0.9}, sees_ai: false}
persona: {capability: {kind: fixed_accuracy, sensitivity: 0.85, specificity: 0.9}, sees_ai: true, trust: {kind: linear_pool, weight_ai: 0.4}}
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
