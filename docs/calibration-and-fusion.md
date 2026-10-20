# Calibration and fusion

What the AI shows the reader, how the two opinions are combined, and where every formula
comes from. A formula without a checked source is listed under
[assumptions](#assumptions-without-a-source).

- [The chain in three steps](#the-chain-in-three-steps) ·
  [Why calibrate at all](#why-calibrate-at-all) ·
  [Why the prevalence has to be removed](#why-the-prevalence-has-to-be-removed)
- [Calibration files](#calibration-files) · [Sources](#sources) ·
  [Assumptions without a source](#assumptions-without-a-source) ·
  [Low-support calibrations](#low-support-calibrations)

## The chain in three steps

1. **Pool.** The model's probability `p` of a finding is the mean over its members (one
   member for MedGemma). Members that do not cover a finding are left out of the mean.

2. **Calibrate the pooled probability**, per finding *f* and per (model, dataset):

   ```
   p_cal = sigmoid(a_f · logit(p) + b_f)
   ```

   `(a_f, b_f)` minimise the negative log-likelihood on the **even-indexed** cache rows. The
   studies draw their patients from the **odd** rows (`patient_generator.split: odd`), so a
   calibration is never scored on the images it was fitted on. In a scenario:

   ```yaml
   ai:
     calibration: {kind: platt, path: data/cache/calibration/vindr__ens4.json}
   ```

   With `stage: output` (the default) the pooled probability is calibrated and the
   uncertainty stays the one of the raw members; `stage: members` calibrates each member
   before pooling, so the uncertainty also describes calibrated numbers.

3. **Fuse with the reader** (`fusion.kind: log_odds_fusion`):

   ```
   logit(p_cal') = logit(p_cal) − logit(prevalence_f) + logit(prior_reader)   # same prior
   logit(p_team) = (1 − w) · logit(p_reader) + w · logit(p_cal')              # log pool
   the team calls "present" when p_team ≥ decision_threshold (default 0.5)
   ```

   `w` is the persona's weight of advice (`trust.weight_ai`); `prior_reader` is the base rate
   the reader's belief is formed under (`trust.prior`, 0.5 for the library personas);
   `prevalence_f` is the prevalence of finding *f* in the fitting half, stored in the
   calibration file. The comparison arm is the linear pool (`fusion.kind: linear_pool`):

   ```
   p_team = (1 − w) · p_reader + w · p_cal
   ```

The general form implemented by `LogOddsFusionPolicy` (`src/hai_sim/doctors/assisted.py`),
with π = `trust.prior`, H_AI the AI's uncertainty on the finding and β = `trust.beta`:

```
logit(p_team) = logit(π) + (1 − w) · [logit(p_reader) − logit(π)]
                         + w · exp(−β · H_AI) · [logit(p_cal) − logit(prevalence_f)]
```

With β = 0, as in every study of the paper, it reduces to step 3. The prevalence is
subtracted only when the scenario uses a per-finding Platt file (`kind: platt` with `path`);
otherwise the AI's evidence is taken relative to π. A finding the AI does not cover gets
weight 0, and the reader decides alone.

The third rule, `confidence_gated`, is a linear pool that uses the weight `w` only where the
AI's confidence max(p, 1 − p) reaches `trust.confidence_threshold`, and ignores the AI
elsewhere. It follows the advice-taking finding that uptake rises with how confident an
advisor appears (Sniezek & Van Swol 2001), which is only as sound as the model's
calibration.

The AI's own call, used for its stand-alone errors and for the reliance metrics, is separate
from the team's: with `ai.decision_threshold: prevalence_matched` it is the per-finding cut,
read from the calibration file, at which the model calls as many positives as there are.

## Why calibrate at all

Raw, both model families are badly over-confident. Measured during development on 200 NIH
ChestX-ray14 images, the expected calibration error (ECE) was 0.27 for the DenseNet-121
ensemble and 0.80 for MedGemma, while MedGemma ranked the cases of several findings better
than the ensemble (by AUROC, the area under the receiver operating characteristic curve). Ranking and calibration are independent: a monotone map changes
the numbers, never the order of the cases. But any rule that adds the AI's evidence to the reader's, as the logarithmic pool
does, is only as sound as the number: an over-confident AI contributes a huge log-odds term
and overrides the reader.

A calibration belongs to (model + readout + prompt + population), not to a model in the
abstract: a map fitted on NIH ChestX-ray14 does not hold on VinDr-CXR, and for a generative
model not even for a different prompt. That is why there is one file per (model, dataset).

## Why the prevalence has to be removed

A calibrated probability contains the base rate. On NIH ChestX-ray14 (odd rows, DenseNet-121
ensemble) the median calibrated probability among the *true positives* is 0.01 to 0.12
depending on the finding, and on most findings none of them reaches 0.5: no image justifies
"more likely present than absent" for a finding with a 1% prevalence, which is the honest
answer. A linear pool averages that number with the
reader's belief and cuts at 0.5, so a calibrated AI almost always votes "absent". Subtracting
the prevalence turns the calibrated probability back into the evidence the image carries (a
likelihood ratio), which the logarithmic pool can add to the reader's.

Measured during development on NIH ChestX-ray14 with the DenseNet-121 ensemble
(`senior_trusting` against the unaided senior, sum over 14 findings, 1,000 patients,
5 seeds):

| advice | fusion | change in missed findings | change in false alarms |
|---|---|---|---|
| raw | linear pool | +10.4 | −298 |
| calibrated | linear pool | +35.4 | −511 |
| calibrated | logarithmic pool on a common prior | **−0.4** | **−105** |

With the logarithmic pool the false-alarm gain is concentrated where the ensemble
discriminates (Hernia, Edema, Cardiomegaly, Effusion, Mass) and is nil where it is near
chance (Fibrosis, Infiltration, Nodule).

A caveat. By a raw 10:1 cost of misses against false alarms the linear pool still scores
better, because a reader at specificity 0.95 on rare findings errs mostly by false alarms,
and an AI that always votes "absent" acts like a stricter reader threshold. That is a change
of operating point, not information from the AI. If cost is the target, the lever is
`trust.decision_threshold` (or `fusion.decision_threshold`), declared as such.

## Calibration files

`data/cache/calibration/<dataset>__<model>.json`, one per dataset (`nih`, `vindr`) and model
(`ens4` for the four-member DenseNet-121 ensemble, `medgemma-4b`, `medgemma-27b`). Each
holds, per finding, the slope `a`, the intercept `b`, the number of positives in the fitting
half, the prevalence, and the prevalence-matched thresholds on the raw and on the calibrated
scale. A `provenance` block records the caches, their fingerprints, the split, and the
held-out ECE before and after calibration.

They are written by `scripts/fit_calibration.py`:

```bash
uv run python scripts/fit_calibration.py --method platt_per_label --split even \
    --cache data/cache/vindr__chex data/cache/vindr__mimic_ch \
            data/cache/vindr__pc data/cache/vindr__rsna \
    --out data/cache/calibration/vindr__ens4.json
```

Several `--cache` stems are composed by image id into one ensemble, and the fit is on the
pooled probability. `--split even` fits on the even rows and reports the error on the odd
ones.

## Sources

| formula | source | where in the paper |
|---|---|---|
| Platt map `1 / (1 + exp(A·f + B))`, fitted by minimum negative log-likelihood | Platt 1999, *Advances in Large Margin Classifiers* | eq. (11)–(12), sec. 2.2, p. 4 |
| calibrate after pooling, not the members (the mean of calibrated members is under-confident) | Rahaman & Thiery 2021, NeurIPS 34 | "Pool-Then-Calibrate", list of contributions p. 2, and sec. 5 |
| re-expressing a posterior on a new prior: `p(i|x) ∝ [p(i) / p_t(i)] · p_t(i|x)` | Saerens, Latinne & Decaestecker 2002, *Neural Computation* 14(1) | eq. (3)–(4), sec. 2.2 |
| logarithmic opinion pool `∝ Π p_i^{w_i}` with `Σ w_i = 1`; linear opinion pool `Σ w_i p_i` | Genest & Zidek 1986, *Statistical Science* 1(1), 114–148 | eq. (3.11) and (3.10), p. 119 |
| product rule for conditionally independent classifiers `P^{−(R−1)}(ω) Π P(ω|x_i)` | Kittler, Hatef, Duin & Matas 1998, *IEEE TPAMI* 20(3) | eq. (7), p. 228 |
| pooled weight of advice ≈ 0.39 | Bailey et al. 2023, *Current Psychology* | pooled estimate of the meta-analysis |

Differences from the sources:

- Platt fits the sigmoid on the score of a support vector machine; here the input is
  `logit(p)` of a network or of a language-model readout, the usage popularised by Guo et
  al. 2017. Platt also smooths the 0/1 targets for small samples; that is not done here
  (thousands of images per fit).
- Rahaman & Thiery pool then calibrate with a *single temperature*, in a low-data, mixup
  setting; here the same order is used with a per-finding Platt map.
- Saerens et al. and Kittler et al. write the multi-class form; the binary log-odds form in
  the code is the same identity for two classes.
- With `w = 0.5` and a common prior, the decision of the logarithmic pool coincides with
  Kittler's product rule.

## Assumptions without a source

1. **The weight of advice as the weight of a logarithmic pool.** Bailey et al. 2023 measure
   it on numerical estimates, as a linear shift toward the advice, in non-clinical tasks.
   Nobody has measured it in log-odds space or for radiological reads. The same caveat,
   weaker, holds for the linear pool (see `src/hai_sim/personas.py`).
2. **The uncertainty discount `exp(−β · H)`** of `LogOddsFusionPolicy` is a design of this
   framework. The studies of the paper run with `beta: 0`; switched on, it is a hypothesis
   to test. Note that the default of `trust.beta` is 1.0, so set `beta: 0` explicitly to
   reproduce the studies.
3. **`prior_reader = 0.5`**: the base rate the signal-detection reader's belief is formed
   under, chosen so that its call and its belief agree (see the persona library).

## Low-support calibrations

Findings with fewer than about 40 positives in the fitting half. The counts depend on the
dataset rows, so they are the same for all three models:

| dataset | finding | positives in the fitting half |
|---|---|---|
| VinDr-CXR | Atelectasis | 33 |
| VinDr-CXR | Pneumothorax | 31 |
| NIH ChestX-ray14 | Edema | 39 |
| NIH ChestX-ray14 | Hernia | 43 (borderline) |

Treat any result that depends on the calibrated value of these findings with care.
