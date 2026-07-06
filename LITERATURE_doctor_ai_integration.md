# Integrating AI information into the doctor model — literature review & design

**Scope.** How should the *AI model's output* be integrated into the *doctor
model* inside `hai_sim`, as a **swappable, modular** component? This document
surveys the reference literature across four angles, then recommends a concrete
doctor model, a starting complexity level, and an experimental design for the
thesis. It is the companion to `LITERATURE_arrival_modeling.md` for the
human–AI-integration axis of the simulator.

Acronyms are expanded on first use. AI = artificial intelligence; CXR = chest
X-ray; WOA = Weight of Advice; JAS = Judge–Advisor System; LLR = log-likelihood
ratio; L2D = Learning to Defer; RAIR = Relative AI Reliance; RSR = Relative
Self-Reliance; Grad-CAM = Gradient-weighted Class Activation Mapping; SDT =
Signal Detection Theory; EM = Expectation–Maximization; ECE = Expected
Calibration Error; OOD = out-of-distribution; pp = percentage points; MI =
mutual information; MRMC = multi-reader multi-case; RCT = randomized controlled
trial.

---

## 0. The framing that reorganizes everything

Our "doctor" (`FixedAccuracyDoctor`, `RandomMockDoctor`) is a **behavioral
simulated reader**, not an image model: it derives a decision from the patient's
ground truth and corrupts it at a configured sensitivity/specificity. So
"integrating AI information into the doctor" does **not** mean feeding AI features
into a second network. It means **letting the AI output modulate the doctor's
decision process** — shift its effective sensitivity/specificity, or its per-label
probability, given the AI's classification / uncertainty / localization.

The literature splits into two families that answer different questions. **We
need both.**

| Philosophy | Question answered | Model family | Role in the thesis |
|---|---|---|---|
| **Normative** (optimal fusion) | "What is the *best* team performance achievable?" | Log-odds / naïve-Bayes pooling; Bayesian confusion-matrix (Kerrigan–Steyvers) | **Upper bound / reference** — the ceiling against which behavioral sub-optimality is measured |
| **Descriptive** (advice-taking) | "How does a *real clinician* who weighs advice behave?" | Weight-of-Advice; trust dynamics (Judge–Advisor System) | **Realism** — reproduces egocentric discounting, algorithm aversion, automation bias, over-reliance |

**Both live behind the same seam** already in the code:
`doctor.diagnose(..., ai_hint: ModelOutput) -> ModelOutput` (`engine.py:123`),
and both produce a **per-label posterior**. So the modular design is a single
swappable protocol:

```
IntegrationPolicy.combine(doctor_opinion, ai_hint, context) -> ModelOutput
```

Normative rules and behavioral rules are just different implementations of it.

**Work in logit space** (log-odds of the probability). There the fusion is
linear, "confidence" maps to distance-from-0.5 = precision, and the AI's
**predictive entropy** (total uncertainty) and **mutual information** (epistemic
uncertainty) — which the pipeline already emits — enter naturally as a *weight*,
not as extra class probabilities.

---

## 1. The three integration modalities = the signal-richness ladder

This maps the user's "information = any aspect of the AI output" onto three
strictly-richer signal levels. Each has literature-backed mechanisms and *real
clinical effect sizes* (§4).

| Modality | AI signal consumed | Integration mechanism | Expected empirical effect (source) | Status in code |
|---|---|---|---|---|
| **M1 — Classification** | label / probability `p_M` | fixed-weight or log-odds pool; AI as a second reader | **+6…+26 pp** sensitivity when correct (Seah 2021); when **wrong**, reader follows it **25–36 %** (experts) → **42–80 %** (novices) of the time (Gaube 2021; Dratsch 2023) | ✅ `ai_hint` carries `labels`+`confidence` |
| **M2 — + Uncertainty** | `p_M` + entropy `H` / MI `I` | instance-adaptive weight `w_M(x)=e^{−βH}`: down-weight the AI where it is uncertain | calibrated confidence improves *appropriate reliance* but **not** guaranteed accuracy; **miscalibrated** confidence *worsens* decisions (Zhang 2020; He 2024) | ✅ `uncertainty` already in the hint |
| **M3 — + Localization** | Grad-CAM heatmap / bounding box | trust gate `g(x)`: veto the AI when the heatmap is anatomically implausible (Clever-Hans catch) | heatmaps **halve** automation bias (36 %→18 %, Pesapane 2026) **but** with a ~9–11 % floor and **no protection** against a systematically biased AI (Jabbour 2023; Bernstein 2023) | ⚠️ **missing**: localization is only in the offline `SaliencyProvider` plane, not in the runtime `ModelOutput` |

**The M3 gap is the only architectural addition.** Surface a compact
localization summary into `ModelOutput.extra` at runtime — the minimal shape is a
**per-label plausibility scalar** `g_i ∈ [0,1]` (e.g. fraction of heatmap energy
inside the lung fields), which reuses the CAM math already in
`analysis/localization/cam.py`.

Orthogonal to *which signal*, there is *how* it is integrated (the policy). The
two axes together turn "Model A (AI) × Model B (doctor) × integration modality"
into a clean factorial experiment (§5).

---

## 2. Recommendation — which doctor model, at what complexity, now

**Build an `AssistedDoctor` = own opinion (reuse `FixedAccuracyDoctor`) + a
pluggable `IntegrationPolicy` that fuses it with `ai_hint`.**

`FixedAccuracyDoctor` already emits per-label confidence + uncertainty, and its
confidence is **calibrated by construction** (you set it via sensitivity/
specificity) — so the doctor's precision `c_D` is a *controllable design
parameter*, ideal for a controlled study. Its opinion is formed independently of
the AI, so the "form-your-own-opinion-first" ordering (which the advice-taking
literature shows increases discounting) is naturally representable by whether the
policy anchors on `x_D`.

**Starting complexity: confidence-weighted log-odds fusion** ("M2 / Rule C"):

```
logit(p_team) = logit(π) + w_D·LLR_D + w_M(x)·LLR_M
  LLR_D = logit(p_D) − logit(π)          (doctor evidence)
  LLR_M = logit(p_M) − logit(π)          (AI evidence)
  w_M(x) = exp(−β·H[p_M(x)])             (down-weight uncertain AI)
```

`π` = per-label prevalence/prior; `H` = predictive entropy. **Why this is the
sweet spot** (not simpler, not more complex to start):

- **One interpretable parameter** (`β` = trust sharpness; or `λ` for egocentric
  discounting).
- **Consumes exactly the signals already emitted** (probability + entropy + MI).
- **It is both philosophies at once**: with the Bayes-optimal weight it is the
  normative fusion; with `w<0.5` it is behavioral egocentric discounting
  (empirical mean WOA ≈ **0.39**, meta-analysis Bailey et al. 2023, N≈17 000).

### Complexity ladder

| Level | Rule | Requires | When to use |
|---|---|---|---|
| Trivial baseline | **Fixed weight** (linear pool) `p=(1−w)p_D+w·p_M` | probabilities only | minimal reference; calibration-tolerant |
| **→ DEFAULT** | **Uncertainty-weighted log-odds** (M2 / Rule C) | **calibrated** prob. + entropy/MI | the workhorse — start here |
| Advanced (dynamics) | **Trust dynamics** (M3-behavioral): `w` rises/falls with AI track record, asymmetric drop after an error (`α_down ≫ α_up`) | case stream + feedback | "algorithm aversion / trust over time" chapter |
| Advanced (explanation) | **Localization-gated verify** (M4 / P5): gate `g(x)` vetoes the AI when the heatmap is implausible | localization in the hint | "bounding-box modality" chapter |

---

## 3. The four seams already in the code (and the one gap)

| Seam | Location | Today | Fill with |
|---|---|---|---|
| AI→doctor channel | `doctor.diagnose(..., ai_hint)` wired at `engine.py:123` | carries AI `labels`+`confidence`+`uncertainty` | add a localization summary to `ModelOutput.extra` |
| Integration policy | `AssistedPolicy` in `doctors/mock.py` (`noop`/`follow_when_certain`) | trivial threshold copy | generalize to swappable `IntegrationPolicy` implementations |
| Team KPI | `hybrid` block in `metrics/kpi.py:133` | **placeholder `== doctor`** (comment: "later: fusion policy") | complementarity gap + RAIR/RSR + over-/under-reliance (§5) |
| Probability calibration | — (only threshold calibration exists) | `analysis/calibration.py` picks the *decision threshold* τ | add **temperature scaling** of `p_M` for the log-odds/Bayesian rules (§6) |

---

## 4. Literature — Angle A: cognitive & behavioral advice-taking

The descriptive family: how a human imperfectly weights advice. Operates
**per label in logit space**; the observable is the standard JAS metric
**WOA = (x_D\* − x_D)/(x_A − x_D)** (0 = ignore, 1 = fully adopt, 0.5 = equal
weight). Egocentric discounting = WOA < 0.5 (pooled ≈ 0.39).

| Author (year) | Title | Venue | Core finding | Mechanism I can implement |
|---|---|---|---|---|
| Harvey & Fischer (1997) | Taking advice: Accepting help, improving judgment, and sharing responsibility | OBHDP 70(2) | Judges take *some* advice even from novices; more when advisor expert / decision important | WOA rises with (advisor − own expertise) and error cost |
| Yaniv & Kleinberger (2000) | Advice taking in decision making: Egocentric discounting and reputation formation | OBHDP 83(2) | Advice discounted vs own opinion; reputation forms fast, **lost faster than gained** | discount factor λ<1; **asymmetric trust update** (down-fast, up-slow) |
| Sniezek & Van Swol (2001) | Trust, confidence, and expertise in a JAS | OBHDP 84(2) | **Higher advisor confidence ⇒ more trust & uptake** | weight `w` increasing in *displayed advisor confidence* |
| Gino & Moore (2007) | Effects of task difficulty on use of advice | JBDM 20(1) | **Overweight advice on hard tasks, underweight on easy** | baseline `w` increasing in task difficulty / own uncertainty |
| Yaniv (2004) | Receiving other people's advice: Influence and benefit | OBHDP 93(1) | WOA ≈ 0.2–0.3; **distant advice discounted more** | distance-dependent weight (outlier advice rejected) |
| Soll & Larrick (2009) | Strategies for revising judgment | JEP:LMC 35(3) | Two strategies — **averaging vs choosing**; averaging usually better but choosing common | mixture: with prob `p` choose (keep own OR fully adopt), else average |
| Bonaccio & Dalal (2006) | Advice taking and decision-making: an integrative review | OBHDP 101(2) | **Canonical review** of all knobs and their signs | master map of parameters |
| Bailey et al. (2023) | A meta-analysis of the weight of advice | Current Psychology (346 effects) | **Pooled WOA ≈ 0.39**; advisor-quality is strongest predictor (0.32/0.37/0.48 low/neutral/high) | empirical defaults for `w` and reputation modulation |
| Dietvorst, Simmons & Massey (2015) | Algorithm aversion | JEP:General 144(1) | People **abandon an algorithm faster than a human after the same error** | visible-error trigger sharply cutting `w` for the *algorithmic* advisor |
| Dietvorst et al. (2018) | Overcoming algorithm aversion | Management Science 64(3) | **Even slight ability to adjust the model restores adoption** | adjustability flag raising baseline `w` |
| Logg, Minson & Moore (2019) | Algorithm appreciation | OBHDP 151 | Lay people weight advice **more** when labeled "algorithm"; shrinks with own expertise | source-label offset `δ_algo`, moderated by expertise |
| Lee & See (2004) | Trust in automation: designing for appropriate reliance | Human Factors 46(1) | **Trust-calibration** framework; over-trust→over-reliance, under-trust→disuse | reliance = f(trust vs true reliability); calibration gap knob |
| Parasuraman & Riley (1997) | Humans and automation: use, misuse, disuse, abuse | Human Factors 39(2) | Names **disuse** (under-reliance) & **misuse** (automation bias) | two-sided miscalibration parameter |
| Tversky & Kahneman (1974) | Judgment under uncertainty: heuristics and biases | Science 185 | **Anchoring-and-adjustment**; insufficient adjustment | basis for "adjust from own toward advice" + order effects |

### Instantiable mechanisms (simplest → most complex)

Let `x_D,x_A` = doctor/AI estimates (logit), `c_D,c_A` = confidences
(e.g. `c=1/H`), `w∈[0,1]` = WOA.

- **M1 — Fixed-weight linear pool.** `x_D* = (1−w0)·x_D + w0·x_A`.
  `w0` default **0.3–0.4** (egocentric discounting); 0.5 = rational equal-weight;
  →0 fully algorithm-averse; →1 automation bias. This one parameter reproduces
  the seminal anchoring-and-adjustment result.
- **M2 — Confidence-weighted (precision-weighted) + egocentric bias.**
  `w = (λ·c_A)/(c_D + λ·c_A)`, `x_D* = (1−w)x_D + w·x_A`. With `λ=1` this is
  exact inverse-variance/Bayesian fusion; `λ<1` = egocentric discount (tune so
  simulated mean WOA ≈ 0.39). Since `c_D` drops on hard cases, it yields
  "overweight advice on hard tasks" for free (Gino & Moore 2007). **Recommended
  default.**
- **M3 — Reputation/track-record-modulated (trust dynamics).** Add trust state
  `T∈[0,1]`: `w = clip(λ·T·g(source,expertise)·c_A/(c_D+c_A), 0, 1)`; update
  `T ← T+α_up(1−T)` if AI correct, `T ← T−α_down·T` if wrong, with
  **α_down ≫ α_up** (asymmetric revision; sharper for algorithms → aversion).
  Reporting `T` vs the AI's true reliability gives the calibration gap directly.
- **M4 — Choose-vs-average mixture + localization salience.** With prob
  `p_choose` adopt one estimate wholesale, else average; adoption probability
  `p_adopt = σ(a + b(c_A−c_D) + c(T−0.5) + d·s)` where salience
  `s = base + β·1[heatmap] + γ·overlap(heatmap, doctor_ROI)`. `β,γ` capture the
  documented **over-reliance-via-explanation** effect (Gaube 2023). The natural
  home for the Grad-CAM channel.

**Progression:** ship M1 (baseline), make M2 the default, add M3 for trust
dynamics, reserve M4 for the localization/explanation experiments.

### Knobs & documented direction of effect

- Baseline WOA `w0` ≈ 0.39 (< 0.5 = discounting) — Bailey 2023, Yaniv 2004.
- Advisor confidence `c_A` ↑ ⇒ weight ↑ — Sniezek & Van Swol 2001.
- Own confidence `c_D` ↑ ⇒ weight ↓ — Bonaccio & Dalal 2006.
- Task difficulty ↑ ⇒ advice-taking ↑ — Gino & Moore 2007.
- Reputation `T` ↑ ⇒ weight ↑; forms fast, lost faster — Yaniv & Kleinberger 2000.
- Visible AI error ⇒ reliance cut, **steeper for algorithms** — Dietvorst 2015.
- Model adjustability ⇒ adoption ↑ — Dietvorst 2018.
- "Algorithm" label: lay ↑ / expert ≈ or ↓ — Logg 2019.
- Doctor expertise ↑ ⇒ reliance ↓ (novices gain most) — Logg 2019, Tschandl 2020.
- Localization salience `β,γ` ⇒ reliance ↑ **including on wrong advice** — Gaube 2023.
- Advice distance ↑ ⇒ discounting ↑ — Yaniv 2004.
- Opinion-before-advice ⇒ more discounting (anchor effect) — Tversky & Kahneman 1974.
- Choose-vs-average `p_choose` ⇒ built-in sub-optimality — Soll & Larrick 2009.
- Trust-vs-reliability gap ⇒ over-/under-reliance — Lee & See 2004.

---

## 5. Literature — Angle B: normative / probabilistic fusion

Per-label binary fusion (multi-label = L parallel binary fusions). Notation:
prior `π = p(Y=1)`; AI posterior `p_M`; doctor posterior `p_D`; reliability
weights `w_D, w_M ∈ [0,1]`.

| Author (year) | Title | Venue | Combines | Formula |
|---|---|---|---|---|
| Steyvers, Tejeda, Kerrigan, Smyth (2022) | Bayesian modeling of human–AI complementarity | PNAS 119(11) | human ordinal confidence + machine softmax | latent logits (λ_H,λ_M)~N(μ,Σ), Σ carries human–machine correlation ρ_HM; posterior via MCMC. Complementarity grows as ρ_HM ↓ (measured ≈0.4 vs ≈0.7 for same-type pairs) |
| Kerrigan, Smyth, Steyvers (2021) | Combining human predictions with model probabilities via confusion matrices and calibration | NeurIPS 34 | human hard label + calibrated model prob | **`p(y=j|h=i,m) = φ_ij·m_j^θ / Σ_k φ_ik·m_k^θ`**, φ_ij = p(h=i|y=j) human confusion matrix, m^θ temperature-scaled |
| Dawid & Skene (1979) | ML estimation of observer error-rates using EM | JRSS-C 28(1) | several noisy raters w/ own confusion matrices | `p(y=k|·) ∝ π_k Π_r Π_j (π^{(r)}_{kj})^{[l=j]}`, error-rates + prior via EM (no gold labels) |
| Kittler, Hatef, Duin, Matas (1998) | On combining classifiers | IEEE TPAMI 20(3) | classifier posterior vectors | **Product rule** `∝ p(y)^{-(R-1)}Π_i p_i`; **Sum rule** `∝ Σ_i p_i` (robust, best empirically) |
| Genest & Zidek (1986) | Combining probability distributions | Statistical Science 1(1) | experts' distributions | **Linear pool** `Σ_i w_i p_i`; **log pool** `∝ Π_i p_i^{w_i}` (externally Bayesian) |
| Sorkin, Hays & West (2001) | Signal-detection analysis of group decision making | Psych. Review 108(1) | readers' SDT decision variables | optimal weights `w ∝ R⁻¹d′`, group sensitivity `d′_group = √(d′ᵀR⁻¹d′)`; OR/AND are sub-optimal special cases |
| Guo, Pleiss, Sun, Weinberger (2017) | On calibration of modern neural networks | ICML | recalibrates one classifier | **Temperature scaling** `p̂=softmax(z/T)`, T>1 for over-confident nets; prerequisite for fusion |

Foundational: **Green & Swets (1966)** — optimal SDT statistic is the likelihood
ratio; independent evidence combines by **adding log-likelihood ratios**.

### Fusion rules (simplest → most principled)

- **Rule A — OR / AND (labels only).**
  OR: `Se = 1−(1−Se_D)(1−Se_M)`, `Sp = Sp_D·Sp_M` (raises sensitivity, the
  "second-reader safety net"). AND: dual (raises specificity). Cheapest baseline;
  assumes conditional independence.
- **Rule B — Linear opinion pool (probabilities).**
  `p = w_D·p_D + w_M·p_M`, `w_D+w_M=1`. Fused confidence **cannot exceed** the
  more confident reader ⇒ conservative, robust to one over-confident member
  (miscalibration hurts least). Less statistically efficient. Good default when
  AI calibration is uncertain.
- **Rule C — Log-odds pool with reliability weights = naïve-Bayes/SDT.**
  `logit(p_post) = logit(π) + w_D·LLR_D + w_M·LLR_M`. With `w=1` this is exactly
  `logit(posterior) = logit(prior) + Σ LLR_i`, optimal fusion of conditionally
  independent readers. Weights either **fixed** (SDT-optimal `w ∝ R⁻¹d′`,
  discounting correlated readers) or **instance-adaptive from AI uncertainty**:
  `w_M(x) = exp(−β·H[p_M(x)])` or `1 − H/log2`, so the AI is down-weighted
  precisely where its entropy/MI is high. **Calibration caveat:** the log pool is
  under-dispersed — an over-confident AI *compounds error*; you MUST
  temperature-scale `p_M` first, else fusion is worse than the doctor alone.
- **Rule D — Bayesian confusion-matrix (Kerrigan/Steyvers), most principled.**
  `p(Y=j|h=i,m) = φ_ij·m_j^θ / Σ_k φ_ik·m_k^θ`, doctor confusion matrix
  `φ_ij = p(h=i|Y=j)` (from validation data or EM), calibrated AI vector `m^θ`.
  Replaces the doctor's scalar weight with a full per-class error profile
  (handles a doctor reliable on Cardiomegaly but confusing Effusion↔Edema).

**Ladder for the simulator:** ship A (baseline) and C (log-odds, uncertainty
weights) as the two defaults, B as the robust fallback, D once a per-class doctor
confusion matrix exists.

### What each rule needs from the AI

| Rule | hard label | prob `p_M` | uncertainty `H/I` | localization |
|---|---|---|---|---|
| A OR/AND | required | — | — | — |
| B linear | — | required (robust to mild miscal.) | optional weight | optional gate |
| C log-odds | derivable | **required, calibrated** | **natural fit** as `w_M(x)` | optional plausibility gate |
| D confusion-matrix | — | **required, calibrated** | optional | optional gate |

Uncertainty enters as the **reliability/weight term**, not as extra classes.
Localization enters as a **trust gate** `g(x)∈[0,1]` (heatmap plausibility)
multiplied into `w_M`, or — if localization ground truth is available — as an
extra conditionally-independent LLR term.

---

## 6. Literature — Angle C: formal ML human–AI teaming (policies + metrics)

| Author (year) | Title | Venue | Framework | Decision rule |
|---|---|---|---|---|
| Madras, Pitassi & Zemel (2018) | Predict Responsibly | NeurIPS | learning to defer | jointly train predictor + deferral gate on a **team** objective |
| Mozannar & Sontag (2020) | Consistent estimators for learning to defer to an expert | ICML | L2D as cost-sensitive classification | add a "defer" class, consistent surrogate loss; defer iff expert's expected loss < model's |
| Raghu et al. (2019) | The algorithmic automation problem | arXiv:1903.12220 | triage / automation-as-optimization | route by per-instance human error vs model error |
| De et al. (2020) | Regression under human assistance | AAAI 2020 | budgeted human assistance | greedy subset selection (submodular) |
| De et al. (2021) | Classification under human assistance | AAAI 2021 | margin-based deferral | defer hardest instances at a target automation level |
| Okati, De & Gomez-Rodriguez (2021) | Differentiable learning under triage | NeurIPS | triage-policy learning | **optimal triage = deterministic threshold** on (model error − human error) |
| Wilder, Horvitz & Kamar (2020) | Learning to complement humans | IJCAI | complementarity, end-to-end | spend capacity on human-hard instances; integrated vs discretionary modes |
| Bansal et al. (2021) | Does the whole exceed its parts? | CHI | empirical complementarity + explanations | explanations raise acceptance of **correct AND incorrect** AI (over-reliance) |
| Rastogi et al. (2022) | A taxonomy of human & ML strengths | HCOMP | enabling conditions for complementarity | aggregation model of when team > best member |
| Buçinca, Malaya & Gajos (2021) | To trust or to think | CSCW | cognitive forcing | forcing engagement reduces over-reliance (trust-vs-think tradeoff) |
| Schemmer et al. (2023) | Appropriate reliance on AI advice | IUI | appropriate-reliance measurement | decompose into **RAIR** (switch to correct AI) & **RSR** (keep own correct vs wrong AI) |

### Integration policies (swappable, each keying on a different AI channel)

- **P1 — Always-defer-to-AI** (full-automation baseline; classification).
- **P2 — Defer-when-AI-confident** (AI-initiated; keys on uncertainty): accept AI
  if `u(x)<τ`, else doctor reads. Threshold τ, or a learned rejector (L2D).
- **P3 — Defer-when-doctor-uncertain** (human-initiated; doctor forms opinion
  first, adopts AI if `c_doc<c*`). "Decide-first, then reveal AI" (cognitive
  forcing).
- **P4 — Weighted fusion / calibrated trust** (both channels): `p_team =
  w(x)p_AI + (1−w(x))p_doc`, `w(x)` increasing in AI relative reliability. Reduces
  to P1/P3 at extremes. = Rule C.
- **P5 — Verify-with-localization** (classification + localization + judgment):
  show AI + Grad-CAM; accept if it agrees or the heatmap is plausible; **reject
  when the heatmap is off-target**. Exercises the "know-when-to-override" gap
  explanations alone don't close.
- **P6 — Cost-sensitive triage under a workload budget**: defer only a fraction
  `b` of cases (human capacity), chosen by thresholding per-instance
  (model error − human error) — ties directly to the simulator's arrival/queue
  model, turning policy comparison into an **accuracy-vs-workload frontier**.

Two orthogonal axes: **who initiates** (AI-gated P2 vs doctor-gated P3) and **how
signals combine** (hard hand-off vs soft fusion P4 vs verification P5).

### Team evaluation metrics — *directly computable in a simulator*

Because both counterfactuals (doctor-solo, AI-solo) are known for every case, the
whole reliance apparatus is computable **without a user study**. Build from the
per-instance 2×2 outcome table (doctor-solo correct? × AI correct?):

- **Team accuracy** — final decisions under the policy (per-pathology AUROC, or
  accuracy/sensitivity/specificity at an operating point).
- **Complementarity gap** = `Team − max(AI-solo, Doctor-solo)`; > 0 ⇒
  complementarity (Bansal 2021; Rastogi 2022). Primary ranking quantity.
- **RAIR** = (switched from wrong doctor to correct AI) / (AI-reliance
  opportunities) — harvesting good AI.
- **RSR** = (kept correct answer vs wrong AI) / (self-reliance opportunities) —
  resisting bad AI. Both must be high (P1 exposes: RAIR high, RSR≈0).
- **Over-reliance** = `1−RSR`; **under-reliance** = `1−RAIR`.
- **Accuracy–coverage (risk–coverage) curve** and deferral rate / workload for
  P2/P3/P6 → the **accuracy-vs-workload frontier**.
- **Expected team cost** under asymmetric costs (a missed pathology = FN costs
  more than a FP), per Mozannar & Sontag 2020 / Okati 2021.
- **Team calibration** (Brier / ECE on `p_team`) — fusion can miscalibrate.
- **Localization-grounded reliance** (P5): correlate override decisions with
  heatmap off-target rate — "right for the right reasons".

This fills the `hybrid` KPI block (`metrics/kpi.py:133`), currently a placeholder.

---

## 7. Literature — Angle D: empirical clinical human–AI radiology (realistic parameters)

| Author (year) | Venue | Setting | Key measured effect |
|---|---|---|---|
| Tschandl et al. (2020) | Nature Medicine | dermoscopy, 302 readers | multiclass-prob display **63.6 %→77.0 % (+13.4 pp)**; binary score → **no** benefit; **faulty/OOD AI dragged even experts below unaided** |
| Jabbour et al. (2023) | JAMA | 457 clinicians, respiratory-failure vignettes | standard AI+explanation **+4.4 pp**; **biased AI −11.3 pp**; **saliency did NOT rescue** (−9.1 pp) |
| Gaube et al. (2021) | npj Digital Medicine | 265 physicians, CXR | inaccurate advice worsened accuracy **regardless of source label**; **41.7 %** of non-experts vs **27.5 %** of radiologists always followed wrong advice |
| Dratsch et al. (2023) | Radiology | 27 radiologists, mammography BI-RADS | with **incorrect** AI, correct-rating share collapsed: inexperienced **79.7 %→19.8 %**, very experienced **82.3 %→45.5 %** |
| Bernstein et al. (2023) | European Radiology | 6 radiologists, 90 CXR, sham AI | incorrect AI: **FN 2.7 %→33 %**, **FP 51.4 %→86 %**; bounding boxes did not reliably protect |
| Pesapane et al. (2026)† | European Radiology | 6 radiologists, 200 mammograms, Grad-CAM | heatmaps **halved bias**: automation 36.1 %→17.8 %, anchoring 33.9 %→17.2 %; accuracy 86.2 %→90.1 %; ~1-in-6 bias persists |
| Seah et al. (2021) | Lancet Digital Health | MRMC, 20 radiologists, 2568 CXR | sensitivity **+6…+26 pp**; reading time **81 s→56 s (−31 %)** |
| Nam et al. (2023) | Radiology (RCT) | 10 476 patients, nodule detection | detection **0.59 % vs 0.25 %**, no rise in false referrals |
| Kim et al. (2022) | Korean J Radiol | 24 readers, mammography | sensitivity specialists 74.6 %→88.6 %, generalists 52.1 %→79.4 %; time fell for specialists, **rose** for generalists |
| Lång/Hernström (2023/2025) | Lancet Oncol / Lancet | MASAI RCT, >105 000 women | cancer detection **+29 %**, screen-reading workload **−44 %** (triage mode) |
| Patel et al. (2019) | npj Digital Medicine | pneumonia CXR | human+AI **beat either alone** (direct complementarity when AI reliable) |

† Pesapane 2026 captured via trade-press summary — verify DOI/authors/numbers
against the journal before citing in the thesis. The other headline numbers were
cross-verified across independent sources.

HCI-side (confidence/uncertainty modality): Zhang, Liao & Bellamy (2020, FAT*);
Rechkemmer & Yin (2022); He et al. (2024).

### Modality effects (summary)

- **(a) Classification/score only.** Gains when correct (+6…+26 pp sensitivity),
  **but only if information-rich** (a binary score gave *no* benefit — Tschandl).
  Specificity ≈ flat. Automation bias substantial: readers follow wrong AI
  ~¼–⅓ (experts) to ~⅘ (novices).
- **(b) + Confidence/uncertainty.** Imaging evidence thin; well-calibrated
  confidence improves *appropriate reliance* but often **not** team accuracy;
  **miscalibrated/overconfident** AI *increases* over-reliance. Treat confidence
  as a reliance modulator effective only insofar as it is calibrated.
- **(c) + Localization/heatmap.** Genuinely two-sided: protective (Pesapane —
  a visibly wrong heatmap lets readers catch a wrong AI) vs harmful (Jabbour,
  Bernstein — plausible-but-wrong explanations deepen anchoring). No free lunch;
  depends on whether the reader interrogates the box. Novices benefit most.
- **Complementarity is conditional, not automatic:** team > best-of-two only when
  AI is accurate and the display is information-rich; with faulty/biased/OOD AI
  the team falls *below* the unaided human.

### Defensible simulator parameters

- **Correct-AI benefit:** sensitivity +6…+26 pp (central ~+8…+12 pp); accuracy
  +4…+13 pp (~+5 terse score, ~+10–13 rich); specificity ≈ N(0, small); AUC
  +0.02…+0.05; reading-time ×0.69–0.90 experts, ×1.10–1.25 novices.
- **Follow-wrong-AI probability:** experts score-only **0.25–0.36**; novices
  **0.42–0.80**; with heatmap **halve** (→~0.17–0.18) but floor ~0.09–0.11 and
  **no reduction** for systematically biased AI.
- **Wrong-AI accuracy cost:** −9…−11 pp; FN error can rise ~5–12×, FP ~1.5–2×,
  strongest in novices.
- **Complementarity gate:** allow team > max only in the correct-AI, rich-display
  branch; cap below unaided human in the faulty-AI branch.

---

## 8. Experimental design (the thesis study)

Factorial grid the modular framework already supports:

```
{M1, M2, M3}                 × which AI signal (classification / +uncertainty / +localization)
  × {normative, behavioral}  × how it integrates (optimal fusion vs advice-taking)
  × {AI model A}             × swappable AI (RandomMockAI, TorchXRayVision ensemble, …)
  × {doctor model B}         × swappable doctor (sensitivity/specificity, expertise level)
```

Each cell → a `RunKPIs` with the filled-in `hybrid` block: complementarity gap,
RAIR/RSR, over-/under-reliance, and (with deferral) the accuracy-vs-workload
frontier. The normative cells give the ceiling; the behavioral cells (calibrated
to §7 effect sizes) give realistic human behavior; the gap between them is a
thesis result.

---

## 9. Prerequisite: probability calibration (precise)

The log-odds/Bayesian rules (Rule C/D) need **calibrated probabilities**. A CXR
net is systematically over-confident (Guo 2017); an inflated `p_M` dominates
fusion and makes the team **worse** than the doctor alone.

- ✅ Present: **threshold calibration** (`analysis/calibration.py` picks the
  decision cut τ, recall-target / cost-weighted). Needed for the *decision* and
  for OR/AND.
- ⚠️ Missing: **probability calibration** (temperature scaling — adjusts the
  *value* of `p_M`). Small addition (one `T` per pathology, fit on the
  calibration split the `InferenceCache` already exposes). Rules A/B are
  calibration-tolerant, so M1 can ship without it; add it when M2-normative goes
  in.

---

## 10. Implementation next steps

1. **`IntegrationPolicy` protocol** in `core/protocols.py`:
   `combine(doctor_out, ai_hint, context) -> ModelOutput`. Generalizes the
   current `AssistedPolicy`.
2. **`AssistedDoctor`** wrapping `FixedAccuracyDoctor`'s own opinion + a policy.
3. **Default policy**: uncertainty-weighted log-odds fusion (M2 / Rule C).
4. **Baseline + behavioral policies**: linear pool (M1), trust-dynamics (M3).
5. **Fill the `hybrid` KPI block**: complementarity gap, RAIR/RSR,
   over-/under-reliance.
6. **Surface localization into `ModelOutput.extra`** (per-label plausibility
   scalar) to enable M3.
7. **Add temperature scaling** to the calibration module for Rule C/D.
8. Config plumbing in `config/scenario.py` (`DoctorConfig`) + `factory.py`.

---

### Primary sources to read first

- Advice-taking: Bonaccio & Dalal 2006 (review); Bailey et al. 2023 (WOA
  meta-analysis); Logg et al. 2019.
- Fusion: Kerrigan et al. 2021 (NeurIPS) + Steyvers et al. 2022 (PNAS);
  Kittler et al. 1998.
- Teaming/deferral: Mozannar & Sontag 2020; Bansal et al. 2021; Schemmer et al.
  2023.
- Clinical: Gaube et al. 2021; Jabbour et al. 2023 (JAMA); Dratsch et al. 2023;
  Tschandl et al. 2020.
