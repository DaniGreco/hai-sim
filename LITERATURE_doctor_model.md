# Modeling the doctor (simulated reader) — literature review & design

**Scope.** How should the **doctor's own decision process** be modeled in `hai_sim`
— the reader whose opinion the AI hint is then integrated into? This is the
companion to `LITERATURE_doctor_ai_integration.md` (which covered *how AI info
enters* the doctor); here we design *the reader itself*. Four-angle survey (SDT
single-reader, case-difficulty & human–AI error correlation, empirical
modulators, MRMC reader-population simulation), then a unified generative model,
a complexity ladder, and how it plugs into the existing `AssistedDoctor` spine.

Acronyms expanded on first use. SDT = Signal Detection Theory; ROC = Receiver
Operating Characteristic; AUC = Area Under the ROC Curve; CXR = chest X-ray; d′ =
detectability index; MRMC = multi-reader multi-case; DBM = Dorfman-Berbaum-Metz;
OR = Obuchowski-Rockette; RM = Roe-Metz; IRT = Item Response Theory; SoS =
satisfaction of search; Φ = standard-normal CDF, Φ⁻¹ its inverse (probit), σ(·) =
logistic; ρ_HM = human–machine correlation; pp = percentage points.

---

## 0. The convergence (the key insight)

The four literatures are **one model** seen from four angles:

- **SDT** (angle A) gives one reader: evidence `x`, discriminability `d′`,
  criterion `c`, and `d′ = √2·Φ⁻¹(AUC)` — the scale that pins doctor competence to
  the AI's measured per-pathology AUC.
- **MRMC / Roe-Metz** (angle D) is that same SDT observer written as a
  variance-component decomposition `Z = μ + R[reader] + C[case] + RC + E`, used to
  *simulate reader populations*.
- The **shared case effect `C[case]`** in Roe-Metz is exactly the **latent case
  difficulty `δ`** of the difficulty/IRT literature (angle B) — and it is the
  **engine of human–AI error correlation**: readers (and an AI reading the same
  images) fail together on hard cases.
- The **modulators** (angle C — expertise, fatigue, prevalence, calibration, SoS)
  are just *what sets or drifts* `d′` and `c`.

So modeling the doctor well = **an SDT observer whose discriminability is eroded
by a shared per-case difficulty, with a tunable coupling `ρ` to the AI's
difficulty**. That single `ρ` is the complementarity lever the whole thesis turns
on: the team beats its members only when doctor and AI fail on *different* cases.

---

## 1. The unified doctor model (the recommendation)

Per case `k`, per label `ℓ`, truth `y_ℓ ∈ {0,1}`, agent `g ∈ {doctor, AI}`:

```
# once per case: a shared latent difficulty (bimodal, or a real instance-hardness score)
δ_k ~ Difficulty()

# per agent, per label:
δ_g   = √ρ · δ_k + √(1−ρ) · η_g            η_g ~ N(0,1)     # shared difficulty; corr(δ_D, δ_A) = ρ
d'eff = max(0, d'_{g,ℓ} − a · δ_g)                           # difficulty erodes discriminability
x     = d'eff · y_ℓ + ε                    ε ~ N(0,1)        # SDT evidence (reader's internal signal)
call  = 1[ x ≥ k_ℓ ]                                         # binary decision at criterion k_ℓ
belief= σ( logit π_ℓ + d'_{g,ℓ}·x − ½ d'_{g,ℓ}² )            # calibrated posterior P(disease | x)
```

Every parameter is literature-anchored:

| Parameter | Meaning | Set from |
|---|---|---|
| `d'_{g,ℓ}` | per-label discriminability (competence) | `d′ = √2·Φ⁻¹(AUC_ℓ)`; AUC from the model's/tier's measured per-pathology AUC (A, C, D) |
| `k_ℓ` / `c_ℓ` | criterion (operating point / caller aggressiveness) | target sensitivity, or cost/prevalence Bayes-optimal `k = ½d′ + ln(β*)/d′` (A, C) |
| `δ_k` | shared case difficulty | **bimodal** (mostly trivial/impossible, Meding 2022) or a real instance-hardness / IRT / soft-label-entropy score (B) |
| `a` | how strongly difficulty erodes `d′` | tunes case-to-case error spread (B) |
| **`ρ`** | **doctor↔AI difficulty coupling** | **THE lever**: baseline ≈0.35 (Steyvers ρ_HM), sweep [0, 0.75] (B) |
| `π_ℓ` | prevalence used for the belief readout | true prevalence → calibrated; mismatch → mis-calibrated (A) |

**Three properties that make this the right model:**

1. **Genuine belief probability.** `belief` is a real calibrated posterior — so the
   `LogOddsFusionPolicy` we already built consumes it directly; the
   *belief-transform hack* needed for `FixedAccuracyDoctor` is gone.
2. **Same scale as the AI.** `d′_ℓ ↔ AUC_ℓ` puts doctor and AI competence on one
   axis, per pathology.
3. **Emergent hard-case overconfidence.** The reader computes `belief` with its
   *nominal* `d'_{g,ℓ}`, not knowing the case was hard (`d'eff` was reduced). So on
   hard cases it is systematically overconfident — reproducing the measured
   **+40 pp hard-case overconfidence** (Bercovich 2025, angle C) as an *emergent*
   effect, not a bolted-on offset.

**The `ρ` sweep is the headline experiment.** `ρ=0` → independent errors → maximum
complementarity; `ρ=1` → both driven by the same `δ` → correlated failures, team
collapses to its best member. Sweeping `ρ ∈ {0, .2, .4, .6, .8}` traces the whole
complementarity curve and locates the Steyvers boundary `min(a_M/a_H, a_H/a_M) > ρ`.

---

## 2. Complexity ladder

| Level | Model | Adds | Purpose |
|---|---|---|---|
| Current | `FixedAccuracyDoctor` | per-label sens/spec Bernoulli flip | trivial baseline; confidence decoupled from label |
| **L1** | **SDT observer** | per-label `d′` + criterion `c`, calibrated posterior | fixes belief representation; `d′↔AUC` scale |
| **→ L2 (build now)** | **+ shared case difficulty `δ` + coupling `ρ`** | the error-correlation lever | makes complementarity meaningful & sweepable — the thesis core |
| L3 | + modulators | expertise tier (`d′`), fatigue drift `d′(t)`, prevalence→`c`, calibration, SoS | realism knobs (angle C) |
| L4 | + reader population | Roe-Metz variance components (`σ²_R` small, `σ²_crit` large) | inter-reader variability studies (angle D) |

**Recommendation: build L1+L2 together.** They are the minimal model that (a) fixes
the belief, (b) shares the AUC scale, (c) makes the complementarity sweep possible.
L3/L4 are progressive add-ons the architecture already supports.

---

## 3. How it plugs into hai_sim

- **`SDTDoctor`** = a new *base* doctor (own-opinion generator) implementing the L1/L2
  model, dropped in under the existing `AssistedDoctor` exactly like
  `FixedAccuracyDoctor`. Its `diagnose` returns a `ModelOutput` whose `confidence`
  is the genuine posterior `belief` and `labels` the SDT `call`.
- **Removes the belief hack.** `doctors/assisted.py::doctor_belief` exists only
  because `FixedAccuracyDoctor`'s confidence is decoupled from its label; the
  `SDTDoctor` emits a real posterior, so fusion uses it as-is. Keep the transform
  as a fallback for the trivial doctor.
- **Where `δ` lives.** The shared per-case difficulty must be attached to the
  **case/patient** (e.g. a field on `PatientContext` or drawn by the patient
  generator, seeded reproducibly), because *both* the doctor's and the AI's error
  model read it. For a mock AI this is a direct draw; for the **real
  TorchXRayVision AI**, its **own predictive uncertainty is the natural observable
  proxy for `δ`** (angle B: instance hardness ≈ model uncertainty) — so the coupling
  can be induced rather than imposed.
- **Same-scale calibration.** Set the doctor's per-label `d′` from the AUCs already
  measured in `analysis/calibration.py` (per-pathology ROC), offset by a tier Δ for
  expertise.

---

## 4. Angle A — SDT / ROC single-reader model

An SDT reader turns a hidden truth into a continuous latent **evidence** `x`, then
reads out a belief and/or a binary call. Two knobs: **d′** (separation of
diseased/non-diseased evidence = competence) and **c** (yes/no threshold = bias).

| Author (year) | Title | Venue | Model / key formula |
|---|---|---|---|
| Tanner & Swets (1954) | A decision-making theory of visual detection | Psych. Review 61(6) | founding SDT observer; respond yes iff `x ≥ k`; ROC point `(Φ(−k), Φ(d′−k))` |
| Green & Swets (1966) | Signal Detection Theory and Psychophysics | Wiley | ideal LR observer; optimal cut on likelihood ratio `β` |
| Dorfman & Alf (1969) | MLE of SDT parameters — rating data | J. Math. Psych. 6 | binormal rating-scale ROC by maximum likelihood (RSCORE) |
| Metz (1978) | Basic principles of ROC analysis | Sem. Nucl. Med. 8(4) | binormal ROC in radiology; movable decision threshold; area `A_z` |
| Hanley & McNeil (1982) | The meaning and use of the area under a ROC curve | Radiology 143(1) | `AUC = P(x_signal > x_noise)` (Wilcoxon) |
| Pan & Metz (1997) | The "proper" binormal model | Acad. Radiology 4(5) | decision var = monotone transform of LR → concave, no "hook" |
| Macmillan & Creelman (2005) | Detection Theory: A User's Guide (2nd) | Erlbaum | `c = −½(z_H + z_F)`; unequal-variance `d_a = Δμ/σ_rms` |

**Equal-variance model (default):** `x|y=0 ~ N(0,1)`, `x|y=1 ~ N(d′,1)`.
- log-LR is linear: `λ(x) = d′·x − ½d′²`
- **posterior (the belief):** `P(disease|x) = σ(logit π + λ(x))` — perfectly calibrated
  when `π` = true prevalence and the Gaussian model holds.
- **call:** `ŷ = 1[x ≥ k]`, `k = ½d′ + c`; `FPR = Φ(−k)`, `TPR = Φ(d′−k)`.
- **d′ ↔ AUC:** `AUC = Φ(d′/√2)` ⇒ `d′ = √2·Φ⁻¹(AUC)` (AUC .80→d′1.19, .90→1.81, .95→2.33).

**Unequal-variance / binormal (opt-in, asymmetric ROC):** `x|y=1 ~ N(μ,σ²)`, `a=μ/σ`,
`b=1/σ`; `A_z = Φ(a/√(1+b²))`; `d_a = √2·Φ⁻¹(A_z)`. Radiology reader ROCs commonly
sit at `b ≈ 0.7–1.0` (diseased evidence more spread out). Caveat: raw LR is
non-monotone when `σ≠1` (the ROC "hook"); use Pan-Metz proper model if you need a
strictly LR-consistent belief.

**Criterion from cost/prevalence (Bayes-optimal):** `β* = [(1−π)/π]·[C_FP/C_FN]`,
`k = ½d′ + ln(β*)/d′` ⇒ higher prevalence or miss-cost pushes the reader liberal.

**Heterogeneous reader = parameter vectors** `d′ = (d′_1..d′_L)` (competence) and
`c = (c_1..c_L)` (per-label bias), optionally `b` per label. Generalist ≈ flat `d′`;
subspecialist ≈ high `d′` on a few labels. Skill and bias are orthogonal — slide `c`
at fixed `d′` to trace one reader's whole ROC. Label-to-label error coupling via a
multivariate-normal evidence draw with correlation `R` (marginals unchanged).

---

## 5. Angle B — case difficulty & human–AI error correlation

The single most important lever: **how much a shared case difficulty drives both
agents**. Template = GLAD (Whitehill 2009): a shared per-item difficulty couples
multiple decision-makers.

| Author (year) | Title | Venue | Contribution |
|---|---|---|---|
| Steyvers, Tejeda, Kerrigan & Smyth (2022) | Bayesian modeling of human–AI complementarity | PNAS 119(11) | latent `ρ_HM`; complementarity shrinks as `ρ_HM`↑; measured `ρ_HM ≈ 0.4` (vs ≈0.7 within-kind) |
| Yáñez et al. (2024) | Confidence-weighted integration of human & machine judgments | arXiv:2408.08083 | gain largest when judgments weakly correlated |
| Bansal et al. (2021) | Does the Whole Exceed its Parts? | CHI 2021 | team gain needs *complementary error boundaries*, not matched accuracy |
| Geirhos, Meding & Wichmann (2020) | Error consistency (κ) between CNNs and humans | NeurIPS 2020 | standard metric for "same-image errors"; human–CNN κ low |
| Meding et al. (2022) | Trivial or impossible — dichotomous data difficulty | ICLR 2022 | per-image difficulty **bimodal** (46% trivial, 11.5% impossible) |
| Whitehill et al. (2009) | GLAD: optimal integration of labels of unknown expertise | NIPS 2009 | **shared latent difficulty** template: `logit P(correct) = α_j·β_i` |
| Smith, Martinez & Giraud-Carrier (2014) | An instance level analysis of data complexity | Machine Learning 95(2) | **instance hardness** = 1 − mean p(correct); kDN proxy |
| Martínez-Plumed et al. (2019) | IRT in AI: classifiers at the instance level | Artificial Intelligence 271 | fit IRT with models as respondents → per-instance difficulty δ |
| Lalor, Wu & Yu (2016) | Building an evaluation scale using IRT | EMNLP 2016 | 2PL/3PL IRT on human responses → item difficulty |
| Kundel, Nodine & Carmody (1978) | Visual scanning in nodule detection | Invest. Radiol. 13(3) | miss = 30% search / 25% recognition / 45% decision |
| Berbaum et al. (1990) | Satisfaction of search | Invest. Radiol. 25(2) | detecting one finding lowers detection of a second |

**Case-difficulty model (2PL-IRT ⊕ SDT):** draw `δ_k` once per case (bimodal for
realism); effective discriminability `d'eff = d'_0 − a·δ_k`; `P(correct) = Φ(d'eff/2)`.
`δ` bundles perceptual difficulty (low conspicuity/contrast — Kundel) and cognitive
difficulty (SoS, ambiguity — Berbaum). **Estimate `δ` from data** instead of a
synthetic draw: instance hardness (Smith), IRT-on-models (Martínez-Plumed), or
human soft-label entropy (Peterson CIFAR-10H).

**Coupling (the sweepable `ρ`):** split each agent's *experienced* difficulty into a
shared and a private part:
```
z_D = √ρ·δ_k + √(1−ρ)·ε_D ,   z_A = √ρ·δ_k + √(1−ρ)·ε_A   ⇒   corr(z_D, z_A) = ρ  exactly
```
Because the error indicators are monotone thresholdings of `z`, the latent error
correlation (and Geirhos' κ) rises monotonically with `ρ`.

**Realistic `ρ` anchors:** baseline **ρ ≈ 0.35** (Steyvers human–machine posterior
mean ≈0.4, observed ≈0.33); upper bound **≈0.7** (within-kind); sweep **[0, 0.75]**.
Radiology direction of travel: AI liberal / readers conservative → partially
decorrelated errors, ρ in the 0.2–0.4 band (Seah 2021).

---

## 6. Angle C — empirical modulators (what sets/drifts d′ and c)

| Author (year) | Venue | Factor | Effect |
|---|---|---|---|
| Kundel & Nodine (1975) | Radiology 116(3) | expertise/gist | experts detect ~70% of abnormalities from a 200 ms flash |
| Evans et al. (2016) | PNAS 113(37) | gist | above-chance abnormal/normal at 500 ms (AUC≈0.6), scales with expertise |
| Potchen et al. (2000) | Radiology 217(2) | between-reader | top-20 vs bottom-20 radiologists: 95% vs 75% on identical CXRs |
| Krupinski et al. (2010) | JACR 7(9) | fatigue (day) | fracture AUC 0.885→0.852 early vs late in workday |
| Ruutiainen et al. (2013) | Acad. Radiol. 20(3) | fatigue (overnight) | major-error rate ~×2 in final 2 h of a 12-h shift |
| Wolfe et al. (2007) | JEP:Gen 136(4) | prevalence | miss fraction 0.11(44%)→0.25(10%)→0.52(1%), mostly a criterion shift |
| Berbaum et al. (2015) | Acad. Radiol. 22(11) | SoS | second-target TP −7.2 pp (or a conservative criterion shift) |
| Sokolovskaya et al. (2015) | JACR 12(7) | speed | ~2× reading speed → major-miss 10%→27% |
| Bercovich et al. (2025) | Radiology | calibration | radiologists overconfident by **>40 pp on hard cases** (hard–easy effect) |
| Waite et al. (2017) | AJR 208(4) | error types | ~60–80% of errors perceptual vs cognitive |

**Mappings onto the model:**
- **Expertise → baseline `d′` per tier.** Expert AUC ≈0.85–0.90 (`d′≈2.2–2.6`),
  general ≈0.80–0.86 (`1.7–2.0`), junior ≈0.75–0.82 (`1.3–1.6`). Expert−novice gap
  `Δd′ ≈ 0.3–0.6` / ΔAUC 0.05–0.10; much larger on subtle targets.
- **Fatigue → `d′(t)` drift** ≈ −0.02…−0.05 per hour (−0.03 AUC/day; ×2 error at end
  of overnight shift), + a small conservative `c` drift.
- **Prevalence → `c`.** Rare targets ⇒ raise `c` (stricter), `d′` intact (Wolfe).
- **Calibration → confidence-temperature + difficulty-scaled overconfidence** (up to
  +40 pp on hard cases) — **emerges for free** in §1 if the reader reads out belief
  with its nominal `d′`.
- **SoS → state-dependent `c` bump** after a first detection (multi-finding cases).
- **Speed → saturating `d′(t_read) = d′_max·(1−e^{−t/τ})`.**

---

## 7. Angle D — MRMC reader-population simulation

The radiology-stats standard for *simulating* reader studies: a three-factor
random-effects model (random readers, random cases) on a latent Gaussian decision
variable — the Roe-Metz model.

| Author (year) | Venue | Contribution |
|---|---|---|
| Dorfman, Berbaum & Metz (1992) | Invest. Radiol. 27 | DBM: jackknife pseudovalue ANOVA over reader/case/interaction |
| Obuchowski & Rockette (1995) | Comm. Stat. 24(2) | OR: ANOVA on AUC directly; error correlations r1/r2/r3 |
| Roe & Metz (1997) | Acad. Radiol. 4 | the RM simulation model (latent decision variable + random effects) |
| Hillis (2012) | Acad. Radiol. 19(12) | unequal-variance RM extension (current standard simulator) |
| Hillis (2018) | Stat. Med. 37(13) | analytic RM↔OR parameter map |
| Beam, Layde & Sullivan (1996) | Arch. Intern. Med. 156 | empirical: 108 radiologists → Se spread ≥40 pp, AUC spread ~11 pp |

**Generative decomposition:** `Z[j,k,t] = μ_t + R[j,t] + C[k,t] + RC[j,k,t] + E[j,k,t]`
with `AUC = Φ(μ/√2)`. Which component controls what:

| Want | Parameter | Empirical magnitude |
|---|---|---|
| between-reader `d′`/AUC spread | `σ²_R` | **small** (0.01–0.03 → AUC SD ≈0.03–0.05) |
| between-reader operating-point spread | `σ²_crit` | **large** (Beam: Se spread ≥40 pp) |
| **shared case difficulty → correlated failures (incl. AI)** | `σ²_C` | **dominant** (≈0.2) — the correlation engine (= `δ`) |
| idiosyncratic "this reader, this case" | `σ²_RC` | ≈0.2 |
| intra-reader inconsistency | `σ²_ε` | ≈0.2–0.6 |

**AI as an extra reader** reuses the same `C[k]` draws (same images → same hard
cases) with its own `R_AI`, `RC_AI`, criterion; share a **fraction `ρ` of `C[k]`** to
dial the human–AI correlation — the same `ρ` as §5. Between-reader same-case
correlation `r2 ≈ 0.3–0.5` (OR). **Tools:** iMRMC (FDA/DIDSR), RJafroc, MRMCaov —
use to *validate* the induced reader/case correlations. **Defaults:** `σ²_C≈0.2`,
`σ²_RC≈0.2`, `σ²_ε≈0.2–0.6`, `σ²_R≈0.01–0.03`, `σ²_crit` large, `μ` from target AUC,
`b` from Hillis-2012 for unequal variance.

---

## 8. Implementation next steps

1. **`SDTDoctor`** (new base under `AssistedDoctor`): §1 model at L1+L2 — per-label
   `d′` (from AUC) + criterion, shared `δ`, coupling `ρ`, calibrated posterior belief.
2. **Attach `δ` to the case** — a reproducibly-seeded difficulty per patient
   (`PatientContext` field or patient-generator draw) that both doctor and AI read;
   for the real AI, derive/couple via its predictive uncertainty.
3. **Drop the belief-transform** for the SDT path (genuine posterior); keep it as the
   `FixedAccuracyDoctor` fallback.
4. **`ρ` sweep harness** — report team accuracy, complementarity gap, RAIR/RSR, and
   realized error-consistency κ across `ρ ∈ [0, 0.75]`. The headline experiment.
5. **Modulators (L3)** as opt-in: expertise tier, fatigue `d′(t)`, prevalence→`c`.
6. **Reader population (L4)** as opt-in: Roe-Metz variance components; validate with
   iMRMC/RJafroc.

### Primary sources to read first
- SDT: Macmillan & Creelman 2005; Metz 1978; Green & Swets 1966.
- Difficulty/correlation: Steyvers et al. 2022 (PNAS); Whitehill et al. 2009 (GLAD);
  Meding et al. 2022.
- Modulators: Krupinski et al. 2010; Wolfe et al. 2007; Bercovich et al. 2025.
- Population: Hillis 2012; Roe & Metz 1997; Beam et al. 1996.
