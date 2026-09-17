# Measured reader–AI error coupling (rho) on VinDr-CXR

Status 2026-09-17: measured, exploratory. Results in
`out/analysis/reader_ai_coupling/reader_ai_coupling.json`, produced by the `measure.py`
next to it. That script is a provisional holding place and must be folded into the
framework (it is the same kind of object as `difficulty_alignment` in the offline
uncertainty study); do not let it become another one-off like the old `rho_probe`.

## Why this exists

The coupling sweep used `difficulty_slope` as a free knob and then measured the error
correlation it produced on a *simulated* reader. That number is a manipulation check, not
a finding: the correlation exists because we put it there. VinDr-CXR ships the individual
reads of its radiologists (`data/vindr/readers.csv`, three readers per image, 17 reader
ids), so the correlation between a *real* reader's errors and the AI's errors can be
measured directly. The simulator is then calibrated to that value instead of sweeping an
arbitrary range.

## How rho is computed

Unit of analysis: one (image, reader, finding) triple, for the 10 VinDr findings that map
into the model's label space (`vindr14->xrv`; Nodule/Mass uses the max of the two model
outputs).

1. **Leave-one-out reference.** For reader *r*, the reference label is the common answer
   of the *other two* readers, and the unit is kept only when those two agree. The reader
   therefore never votes on the truth they are scored against (scoring against the
   2-of-3 majority would contaminate the reader's error with their own vote).
2. **Reader error** = reader's call differs from the reference.
3. **AI error** = AI's binary call differs from the *same* reference. The AI call is the
   ensemble mean probability cut at a per-finding threshold. Three policies are reported
   (fixed 0.5, prevalence-matched, cost-optimal at FN:FP = 10:1); thresholds are fit on
   even-indexed images, everything is reported on odd-indexed ones.
4. **phi** = Pearson correlation of the two binary error columns, per finding. Reported
   three ways: marginal, and conditional on the reference class
   (`phi|+`: among reference-positive units, i.e. do reader and AI *miss* the same cases;
   `phi|-`: among reference-negative units, i.e. do they *false-alarm* on the same cases).
5. Confidence intervals: 300 image-level bootstrap resamples (units from one image are
   not independent).

### Formulas

`a[i,r,f] ∈ {0,1}`: annotation of reader `r` on image `i`, finding `f`. `r'`, `r''`: the
other two readers of image `i`. `p^(m)[i,f]`: probability from ensemble member `m`.

```
unit u = (i, r, f), kept iff a[i,r',f] == a[i,r'',f]
reference      y_u      = a[i,r',f]                                  (= a[i,r'',f])
AI score       p[i,f]   = mean over the M_f members covering f of p^(m)[i,f]   (NaN-safe;
                          M_f = 1 for MedGemma; Nodule/Mass = max(p[Nodule], p[Mass]))
AI call        yhat[i,f]= 1[ p[i,f] >= tau_f ]
reader error   eH_u     = 1[ a[i,r,f]  != y_u ]
AI error       eA_u     = 1[ yhat[i,f] != y_u ]

threshold      tau_f    = argmin_tau  sum over fit-half units of
                          10 * 1[y_u=1, p<tau]  +  1 * 1[y_u=0, p>=tau]
                          (candidates: 401 quantiles of p on the fit half)
               other policies: tau_f = 0.5 ; tau_f = quantile(p, 1 - mean(y)) (prevalence)

phi   = (n11*n00 - n10*n01) / sqrt( (n11+n10)(n01+n00)(n11+n01)(n10+n00) )
        with n_jk = #{u : eH_u = j, eA_u = k}     (== Pearson corr of eH, eA)
phi|+ = phi restricted to units with y_u = 1      (all errors are misses)
phi|- = phi restricted to units with y_u = 0      (all errors are false alarms)

reader sens = mean(a[i,r,f] | y_u = 1) ; reader spec = 1 - mean(a[i,r,f] | y_u = 0)
CI          = 2.5 / 97.5 percentiles over 300 resamples of IMAGES (with replacement)
split       = fit on even cache row index, report on odd
```

## What came out (ensemble, cost-optimal policy, held-out half)

| finding | ref. positives | reader sens / spec | phi | phi\|+ | phi\|- |
|---|---|---|---|---|---|
| Cardiomegaly | 2174 | 0.885 / 0.988 | +0.11 | +0.13 [+0.08, +0.19] | +0.14 [+0.13, +0.16] |
| Pulmonary fibrosis | 1100 | 0.824 / 0.985 | +0.10 | -0.02 [-0.08, +0.05] | +0.08 [+0.06, +0.10] |
| Pleural thickening | 799 | 0.665 / 0.975 | +0.18 | +0.05 [-0.02, +0.14] | +0.14 [+0.12, +0.16] |
| Pleural effusion | 764 | 0.884 / 0.991 | +0.16 | +0.28 [+0.19, +0.38] | +0.08 [+0.05, +0.11] |
| Lung Opacity | 442 | 0.597 / 0.982 | +0.24 | +0.24 [+0.12, +0.35] | +0.16 [+0.14, +0.20] |
| Nodule/Mass | 358 | 0.662 / 0.991 | +0.21 | +0.09 [-0.05, +0.20] | +0.06 [+0.03, +0.08] |
| Infiltration | 202 | 0.624 / 0.991 | +0.23 | -0.06 [-0.29, +0.12] | +0.10 [+0.06, +0.14] |
| Consolidation | 98 | 0.551 / 0.995 | +0.34 | n/a | 0.00 |
| Pneumothorax | 67 | 0.896 / 0.999 | +0.17 | n/a | n/a |
| Atelectasis | 45 | 0.533 / 0.998 | +0.37 | n/a | +0.14 |

n/a: at the cost-optimal cut the AI calls no positive for that finding, so its error
column is constant inside the class.

Reading:

- **The marginal phi is not a usable parameter.** It moves with the AI operating point
  (Atelectasis: +0.04 at 0.5, +0.24 prevalence-matched, +0.37 cost-optimal) because most of
  it is class structure: both the reader and the AI err far more on positives than on
  negatives, so any two imperfect raters correlate marginally. The simulated reader already
  reproduces that part as soon as it has realistic per-finding sensitivity/specificity.
- **The class-conditional phi is the quantity `difficulty_slope` has to reproduce.** It is
  stable across AI policies. False-alarm coupling `phi|-` is small and consistently
  positive (+0.06 … +0.16). Miss coupling `phi|+` is clearly positive for Pleural effusion
  (+0.28), Lung Opacity (+0.24) and Cardiomegaly (+0.13), and indistinguishable from zero
  for Pulmonary fibrosis, Pleural thickening, Nodule/Mass and Infiltration.
- **MedGemma is not decoupled from the readers.** Both MedGemma cells give conditional
  values in the same range (4B: `phi|-` +0.04…+0.17, Effusion `phi|+` +0.29; 27B: `phi|-`
  +0.08…+0.27, Effusion `phi|+` +0.24). Consistent with a shared, image-driven difficulty.
- By-product: per-finding reader sensitivity/specificity (0.53–0.90 / 0.975–0.999), far
  from the 0.95/0.95 persona used so far. `SDTDoctor` accepts per-label sequences, so
  these can be used directly.

## How it is meant to be fixed in the simulator (not done yet)

1. Reader operating point per finding from the table above (per-label `sensitivity` /
   `specificity` on the SDT capability).
2. `difficulty_slope` calibrated once so that the *simulated* class-conditional phi matches
   the measured one; then frozen. Subsequent studies (which uncertainty measure helps
   most, personas, ordering) run at that fixed coupling.
3. Open point: one scalar slope cannot match ten per-finding targets that range from 0 to
   +0.28. Either accept a single slope fit to the support-weighted mean and report the
   per-finding residual, or make the slope per-label.

## Limitations (to be carried into the thesis)

1. **The reference is a two-reader consensus, not clinical truth.** A reader who is right
   against two colleagues who both missed a subtle finding is scored as a false alarm. If
   the AI also sees that finding, this inflates `phi|-`. The positive false-alarm coupling
   is therefore an upper bound on true false-alarm coupling.
2. **Contested cases are excluded.** Units where the other two readers disagree are
   dropped (0.2–7% of units per finding). Those are the hardest cases, where coupling is
   plausibly strongest, so the measured values are likely underestimates of miss coupling.
3. **Readers read without AI.** This measures the correlation of *independent* errors,
   which is the correct definition of rho. It says nothing about how a reader reacts to
   advice; weight of advice and trust remain literature-anchored assumptions.
4. **Presence means "drew a box".** VinDr labels are localisation annotations; a reader
   who saw a finding but did not box it counts as negative.
5. **Low support.** Atelectasis (45), Pneumothorax (67), Consolidation (98) positives in
   the held-out half: `phi|+` is not estimable or not quotable. Only 10 of the 18 model
   labels are covered at all.
6. **Pooled over readers.** 17 reader ids with very uneven workloads are pooled; rho per
   reader was not examined. Reader identity may confound with case mix.
7. **Out-of-domain only.** NIH has no per-reader labels, so rho is measured on VinDr only
   and is *assumed* to transfer when NIH studies use it.
8. **phi on binary errors depends on base rates.** Its attainable maximum is below 1 when
   the two error rates differ; values are comparable across findings only loosely.
9. **Thresholds are fit against the same kind of reference** (other-two consensus) on the
   fit half; the cost ratio 10:1 is the thesis' declared modelling choice.
