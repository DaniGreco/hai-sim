# hai_sim — Project Scope & Gap Tracker

Living document. Baseline scope from the **2026-07-08** meeting ("Summary of the
planned work"), with context from 2026-07-02 and 2026-06-29. Refined progressively
as the thesis advances — we dissect one item at a time.

Legend: ✅ done · ◐ partial · ❌ missing · ⭐ built beyond the summary (asset).

## 0. One-line status
The **AI + uncertainty-prioritisation + RQ1** half is mature (even ahead of the
summary). The **doctor-persona + real-data grounding + LLM-comparison** half — the
core of the 07-08 scope — is largely greenfield. We have a strong *AI side*; the
*doctor side* exists only as unassembled primitives.

## 1. Overall goal
> "A framework to simulate the interaction of AI and doctor."

✅ The SimPy engine is this framework: `PatientGenerator → AI (ModelStation) →
PriorityStore (sort) → Doctor → Metrics`, with swappable `Protocol`-based
components, YAML scenarios, KPIs and reproducible seeding.

## 2. Planned contributions vs status

### 2.1 Prioritisation techniques ("different ways to give images to the doctor")
| Technique | Status | Exists / Gap |
|---|---|---|
| Uncertainty-based | ✅ | `UncertaintyDescSort` (mean/max/weighted) over 5 UQ measures + full RQ1 validation apparatus |
| Random baseline | ✅ | `RandomSort` |
| **Feature / last-layer UQ ("Ziwei's approach")** | ❌ | see 2.1.a |
| Complexity / outlier-based (pixel-diff, image-diff, edge-cases, ask-doctor) | ❌ | not implemented (06-29 idea) |

**2.1.a — "Ziwei's approach": uncertainty from the last pre-output layer.**
Idea (Ziwei, colleague): measure uncertainty from the **penultimate embedding**
(just before the output linear layer) instead of from output probabilities. This is
a known family of *post-hoc, feature-space* UQ/OOD methods — ideal for our
inference-only setup (no retraining):
- **Mahalanobis distance** (Lee et al., NeurIPS 2018): class-conditional Gaussians
  on training embeddings; score = distance to the (predicted) class mean.
- **Deep Deterministic Uncertainty / DDU** (Mukhoti et al., CVPR 2023): GMM/GDA
  density on features = epistemic; softmax entropy = aleatoric.
- **Deep k-NN** (Sun et al., ICML 2022): distance to k-th nearest training embedding.
- **Energy score** (Liu et al., 2020): `−logsumexp(logits)` — a lighter logit-layer variant.
- (SNGP / DUQ need a modified/retrained last layer → less suitable given inference-only.)

**How to implement in hai_sim (high reuse):**
- The xrv DenseNet head is `ReLU → global-avg-pool → linear`. The penultimate
  embedding = GAP of the stored `(C=1024, 7, 7)` feature map → **already in the
  localization feature cache**, so embeddings are derivable offline with zero extra
  inference.
- Add a `FeatureUQ` (small new protocol, or extend `UQMethod`) taking embeddings;
  fit its statistics on a reference (train) split — fits the existing cache +
  offline-analysis pattern.
- Wire its scalar into `UncertaintyDescSort` → this *is* a new prioritisation
  technique, directly answering "different prioritisation techniques".
- Bonus: for MedGemma (generative), token-probability UQ is unreliable → the
  embedding approach is the natural UQ there too.

### 2.2 Doctor personas ("a way to build personas of doctors")
Dimensions (07-08 + 07-02): **Trust × Visibility × Capability**.
| Piece | Status | Exists / Gap |
|---|---|---|
| Capability (experienced 99% / weak 95%) | ✅ | `FixedAccuracyDoctor(sensitivity, specificity)`, per-label |
| Visibility (sees / doesn't see ML) | ✅ | `use_ai_hint` / `AssistedDoctor` vs solo doctor |
| Trust (trusts / distrusts ML) | ◐ | static fusion weights (`LinearPool` w_ai; `LogOddsFusion` w_doctor/β). **Missing:** trust as a persona parameter, trust *dynamics*, calibration to literature |
| **Persona abstraction** (named bundle + library + paper links) | ❌ | primitives exist but not assembled into a `Persona` composing capability×visibility×trust; no library; no scenario generator over a persona grid |
| Grounding personas in real doctor-error data | ❌ | doctors are synthetic (from ground truth); no real-data ingestion/calibration |
| SDT / fatigue composite doctor | ❌ | README "NOT yet" |

The **Persona layer is the central missing contribution.** Mostly *assembly* + one
new concept (`PersonaConfig` = base doctor + integration policy + visibility +
trust) + a scenario generator that sweeps the grid.

### 2.3 Reusability for researchers ("with real data or real doctors")
◐ Modular protocols + YAML + CLI make it extensible. **Missing:** real-data hooks, a
human-in-the-loop mode (a real doctor answering), packaging/docs for external medics.

### 2.4 LLM comparison → **MedGemma** as an AI backend
❌ today (no LLM actor), but the design anticipates it (`types.py` mentions LLM
vision SDKs / token logprobs). **Decision (07-08):** use **MedGemma-4B**
(multimodal: SigLIP vision + Gemma-3 4B).
- **Local feasibility on this PC — YES.** Intel Core Ultra 7 155H, **32 GB RAM**,
  Intel Arc iGPU, **no NVIDIA/CUDA**, 607 GB free. 32 GB loads 4B comfortably
  (bf16 ≈ 8 GB, 4-bit ≈ 3 GB). Paths:
  - Simplest: **4-bit quantized via Ollama / llama.cpp on CPU** (a few sec/image).
  - Accelerated: **IPEX-LLM or OpenVINO** on the Arc iGPU / NPU.
  - hai_sim **caches inference** (build-once, analyze-offline, like the DenseNet
    ensemble) → per-image speed is not a blocker.
- Access: gated on Hugging Face (Health AI Developer Foundations license) + token.
- Integration: a new `DiagnosticActor` wrapping MedGemma → prompt for per-label
  present/absent → map to the schema. Calibrated per-label UQ is hard for a
  generative model → pair with 2.1.a (embedding UQ).

## 3. Research questions
- **RQ1 — Is sorting by uncertainty effective (does it surface wrong images first)?**
  ⭐ Machinery done: cumulative error-capture curves, `skill` (AUSE/EXAM analogue),
  AUROC, per-pathology. Needs: run at scale + write up.
- **RQ2 (candidate)** — Does feature-space UQ (2.1.a) beat output-entropy UQ at
  surfacing errors?
- **RQ3 (candidate)** — How do personas (trust/visibility/capability) change team
  outcome & appropriate reliance (RAIR/RSR)?

## 4. Datasets — surveyed, decided, partially downloaded (2026-07-13)
Deep-research survey (25 claims verified 3-0). **Key finding:** no public "real
doctor-error catalogue" (RADPEER/discrepancy corpus) exists for CXR; every viable
option defines **error = a per-reader read disagreeing with an adjudicated/majority
consensus** on the same images.

Ranked shortlist:
| # | Dataset | Per-reader | Adjudicated GT | Error = | Schema | Access (reality) |
|---|---|---|---|---|---|---|
| 1 | **NIH ChestX-ray14 re-read** (Majkowska 2020) | ⚠ form-gated `individual_readers.csv` | ✅ **bundled in torchxrayvision** | read ≠ adjudicated | **4 findings** | GT: FREE (xrv); per-reader: Google Form |
| 2 | **CheXpert test** | ✅ 8 rad/study, **open on GitHub** | ✅ 5-vote majority | benchmark read (bc4/6/8) ≠ majority | CheXpert-14 → xrv-18 | labels FREE (GitHub); images: AIMI reg. |
| 3 | VinDr-CXR | ✅ 15k train (3/img, rad_id) | ✅ 3k test (5-rad, 2-stage) | inter-reader disagreement (train) | 28-class | PhysioNet (DUA+CITI) |
| 4 | REFLACX | ✅ 5 rad, graded 0-5 certainty | derive | disagreement (109-img overlap) | REFLACX | PhysioNet |
| — | EGD-CXR | ❌ single reader | — | eye-gaze only (visibility axis) | — | PhysioNet |
| — | CorBenchX | ❌ synthetic (LLM-injected) | — | reusable error taxonomy only | — | arXiv |
| CT | LIDC-IDRI | ✅ 4 rad/nodule | 2-stage | disagreement | CT nodules | TCIA open |

**DOWNLOADED** (`scripts/download_reader_labels.py`, no gate):
- `data/nih_rad_labels/nih_google_adjudicated.csv` — NIH adjudicated CONSENSUS GT,
  4,375 rows, 4 findings (Fracture / Pneumothorax / Airspace-opacity / Nodule-or-mass).
- `data/chexpert_test_labels/` — CheXpert majority `groundtruth.csv` + **8 per-reader
  reads** (bc1/2/3/5/7 GT-contributors + bc4/6/8 benchmark), 500 studies × CheXpert-14.

**DECISION — primary: NIH adjudicated re-read; CheXpert as the open per-reader source.**
Bridge to §2.2: per-reader **sensitivity/specificity per finding** (CheXpert reads vs
majority, or NIH `individual_readers.csv` vs adjudicated) become the
`FixedAccuracyDoctor` persona parameters → a persona library calibrated on real radiologists.

Error operationalisation (exact):
- NIH: `individual_readers.csv`[read] ≠ `nih_google_adjudicated.csv`[label] (4 findings).
- CheXpert: benchmark read (bc4|bc6|bc8) ≠ `groundtruth.csv` (5-vote majority) — non-circular.
- VinDr train: one of 3 blind reads ≠ majority of the other two.

Pending / manual: NIH `individual_readers.csv` (Google Form https://forms.gle/bmYPtLy4qsJahVKFA);
VinDr-CXR + REFLACX (PhysioNet DUA + CITI). Open Qs: (a) are per-reader IDs stable per-individual
(CheXpert samples 2/5 GT readers per study → partly no)? (b) do RSNA/ChestX-Det/PadChest expose
per-annotator IDs? (unconfirmed). Realistic prevalence: population-adjust (as Majkowska did).
Caveat: to *use* the NIH re-read labels we need the matching full-res NIH images (our
`data/nih_sample` is only a subset) — check image coverage of the 4,375 re-read IDs.

## 5. AI backends
| Backend | Status |
|---|---|
| TorchXRayVision DenseNet-121, single + cross-domain Deep Ensemble | ✅ |
| MedGemma-4B (multimodal) | ❌ planned — see 2.4 |

## 6. ⭐ Assets already built (beyond the summary)
- AI→doctor **integration policies** (LinearPool, LogOddsFusion) + **TeamKPIs**
  (complementarity gap, RAIR/RSR appropriate reliance) — backbone for persona/trust work.
- **UQ decomposition** (predictive/mean entropy, mutual information, std),
  epistemic/aleatoric split, cross-domain Deep Ensemble.
- **RQ1 apparatus** (error-capture curves, skill, AUROC, per-pathology).
- **Calibration** (per-pathology thresholds: recall-target & cost-weighted "Borsani-style").
- **Localization validation** (Grad-CAM vs ChestX-Det masks); live & cache paths now
  aligned to max fidelity (224 / float32).
- Reproducible seeding, resumable inference/feature caches, CLI, 98+ tests.

## 7. Task backlog (rough priority)
1. ~~Name & obtain the doctor-error dataset(s)~~ **DONE** (§4): surveyed + decided;
   NIH adjudicated GT + CheXpert 8-reader labels downloaded. **Next:** compute
   per-reader sensitivity/specificity per finding → `FixedAccuracyDoctor` persona
   params (feeds §2.2); verify NIH image coverage; optionally fetch NIH `individual_readers.csv` (form).
2. **Persona layer** (§2.2): `PersonaConfig` + library + scenario-grid generator.
3. **Feature-space UQ** (§2.1.a): embedding extraction from feature cache +
   Mahalanobis/DDU/kNN + wire to sort.
4. **MedGemma actor** (§2.4): quantized local run + `DiagnosticActor` wrapper + inference cache.
5. Trust dynamics (§2.2) + link personas to papers.
6. Run RQ1 at scale + write up; define RQ2/RQ3.
7. LLM related-work positioning (nature triage, craft-md, PatientSim).

## 8. Open decisions
- Application scenario: theoretical/simulation framework (current) vs full-ward tool
  (needs real doctors). 07-08 leans simulation + real-data-calibrated personas. ← confirm.
- Ziwei's exact method reference (a specific paper?), if one exists.
- Error-cost model: formalise 0→1 (false-negative) weighting as a cost matrix?
  (currently partial via the weighted sort aggregator).

---
_Last updated: 2026-07-13._
