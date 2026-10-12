# Calibrated advice and how it is fused with the reader

Status 2026-09-17. What the AI shows the reader, how the two opinions are combined, and
where every formula comes from. Rule of the project: **a formula without a checked source is
either replaced by one that has it, or listed below as an assumption.**

## The pipeline, in three steps

1. **Pool.** The model's probability for a finding is the mean over its members
   (`p`; one member for MedGemma).
2. **Calibrate the pooled probability**, per finding and per (model, dataset):
   `p_cal = sigmoid(a_f * logit(p) + b_f)`, `(a_f, b_f)` by minimum negative log-likelihood
   on the even-indexed cache rows. Files: `data/cache/calibration/<dataset>__<model>.json`
   (written by `scripts/fit_calibration.py --method platt_per_label --split even`), which
   also store the per-finding prevalence of the fitting half. Studies draw patients from
   the odd rows (`patient_generator.split: odd`).
3. **Fuse with the reader** (`fusion.kind: log_odds_fusion`):

   ```
   logit(p_cal') = logit(p_cal) - logit(prevalence_f) + logit(prior_reader)    # same prior
   logit(p_team) = (1 - w) * logit(p_reader) + w * logit(p_cal')                # log pool
   team calls "present" when p_team >= decision_threshold (default 0.5)
   ```

   `w` is the persona's weight of advice; `prior_reader` is 0.5 for the library personas.
   The comparison arm is the linear pool `p_team = (1 - w) * p_reader + w * p_cal`.

## Why the prevalence has to be removed

A calibrated probability contains the base rate. On NIH (odd rows) the median calibrated
probability among the *true positives* is 0.01 to 0.12 and it reaches 0.5 for about 0% of
them: no image justifies "more likely present than absent" for a 1% finding, which is the
honest answer. A linear pool averages that number with the reader's belief and cuts at
0.5, so a calibrated AI always votes "absent". Measured (a development study on NIH, since removed,
senior_trusting vs the unaided senior, sum over 14 findings, 1000 patients, 5 seeds):

| advice | fusion | delta missed findings | delta false alarms |
|---|---|---|---|
| raw | linear pool | +10.4 | -298 |
| calibrated | linear pool | +35.4 | -511 |
| calibrated | logarithmic pool on a common prior | **-0.4** | **-105** |

With the logarithmic pool the false-alarm gain is concentrated where the ensemble
discriminates (Hernia, Edema, Cardiomegaly, Effusion, Mass) and is nil where it is near
chance (Fibrosis, Infiltration, Nodule).

Caveat to keep in the thesis: by raw 10:1 cost the linear pool still scores better, because
a reader at specificity 0.95 on rare findings errs mostly by false alarms, and an
always-"absent" voter is equivalent to a stricter reader threshold. That is a change of
operating point, not information from the AI; if cost is the target the lever is
`decision_threshold`, declared as such.

## Formula -> source (each checked against the PDF in `../papers/`)

| formula | source | where in the paper | file |
|---|---|---|---|
| Platt map `1 / (1 + exp(A f + B))`, fitted by minimum negative log-likelihood | Platt 1999, *Advances in Large Margin Classifiers* | eq. (11)-(12), sec. 2.2, p. 4 | `platt1999probabilistic.pdf` |
| calibrate after pooling, not the members (averaging calibrated members is under-confident) | Rahaman & Thiery 2021, NeurIPS 34 | "Pool-Then-Calibrate", contributions list p. 2 and sec. 5 | `rahaman2021ensembles.pdf` |
| re-expressing a posterior on a new prior: `p(i|x) ∝ [p(i) / p_t(i)] * p_t(i|x)` | Saerens, Latinne & Decaestecker 2002, *Neural Computation* 14(1) | eq. (3)-(4), sec. 2.2 | `saerens2002priors.pdf` |
| logarithmic opinion pool `∝ Π p_i^{w_i}`, `Σ w_i = 1`; linear opinion pool `Σ w_i p_i` | Genest & Zidek 1986, *Statistical Science* 1(1), 114-148 | eq. (3.11) and (3.10), p. 119 | `genest1986combining.pdf` (scan) |
| product rule for conditionally independent classifiers `P^{-(R-1)}(ω) Π P(ω|x_i)` | Kittler, Hatef, Duin & Matas 1998, IEEE TPAMI 20(3) | eq. (7), p. 228 | `kittler1998combining.pdf` |
| pooled weight of advice ≈ 0.39 | Bailey et al. 2023, *Current Psychology* | already verified in `../papers/README.md` | paywalled |

Differences from the sources, stated:

- Platt fits the sigmoid on an SVM score; here the input is `logit(p)` of a network or of an
  LLM read-out, the usage popularised by Guo et al. 2017. Platt also smooths the 0/1 targets
  for small samples; not done here (thousands of images per fit).
- Rahaman & Thiery pool-then-calibrate with a *single temperature* in a low-data, mixup
  setting; here the same order is used with a per-finding Platt map.
- Saerens et al. and Kittler et al. write the multi-class form; the binary log-odds form used
  in the code is the same identity for two classes.

## Assumptions with no source

1. **The weight of advice as the weight of a logarithmic pool.** Bailey 2023 measures it on
   numerical estimates, as a linear shift toward the advice, in non-clinical tasks. Nobody
   measured it in log-odds space or for radiological reads. Same caveat, weaker, for the
   linear pool (already declared in `personas.py`).
2. **The uncertainty discount `exp(-beta * H)`** of `LogOddsFusionPolicy` is a design of this
   project. The studies run with `beta: 0`; if it is ever switched on it must be presented as
   a hypothesis.
3. **`prior_reader = 0.5`**: the base rate the SDT reader's belief is formed under, chosen so
   that its call and its belief agree (see the persona library).

## Low-support calibrations (fewer than 40 positives in the fitting half)

VinDr: Atelectasis (33), Pneumothorax (31). NIH: Edema (39), Hernia (43, borderline). Flag them
wherever a result depends on the calibrated value.
