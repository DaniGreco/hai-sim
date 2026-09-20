# The experiments of the thesis, in plain words

The thesis is the framework. These experiments show it being used and returning sensible
answers; each one exercises a different part of it. Everything is reported **per pathology**
(prevalences span two orders of magnitude, a pooled number hides more than it shows), every
figure is in English, and every study evaluates on the cache rows the calibrations were not
fitted on (odd rows; calibrations are fitted on the even ones).

| | question | command | framework part |
|---|---|---|---|
| E1 | Which reading order meets the sick patients first? | `hai-sim capture configs/studies/capture.yaml` | uncertainty + ordering |
| E2 | Which AI orders the worklist best? | same command, same table | swappable models, cache matrix |
| E3 | Under load, do sick patients really wait less? | `hai-sim study configs/studies/queue_load.yaml --workers 20` then `hai-sim plot queue_load <summary.csv>` | the queue simulator |
| E4 | What does a reader gain from seeing the AI? | `hai-sim study configs/studies/team_effect.yaml --workers 20` then `hai-sim plot team_effect <long.csv>` | personas, calibration, fusion |
| G | Do readers and the AI fail on the same images? | see `STUDY_reader_ai_coupling.md` | justifies independent personas |

Preparation, once: `scripts/fit_calibration.py --method platt_per_label --split even ...`
(six files, see `STUDY_calibrated_advice.md`).

## E1 — which reading order meets the sick patients first?

Take all the images of a dataset and read them in the order a model proposes. After each
image, count how many of the patients who really have the finding you have already met. Draw
that count against the number of images read.

- A perfect order meets all the sick patients first, so the line climbs straight and then
  stays flat: a right trapezoid.
- A random order is the diagonal.
- A real order lies in between. **Skill** is how much of the way from the diagonal to the
  trapezoid it covers: 1 = perfect, 0 = random, negative = worse than random.

Four orders are compared: by **probability** (most suspicious first), by **entropy** of that
probability (most undecided first), by the **mean entropy of the ensemble members**, and by
the **disagreement between members**. A second panel counts the model's own errors instead of
the sick patients.

What comes out: ordering by probability is the best or tied-best order for every finding.
Entropy of the *calibrated* probability gives the same order (a rare finding never reaches a
probability of one half, and below one half entropy grows with the probability), while
entropy of the *raw* probability is arbitrary, down to skill -0.85. Member entropy is worse
than probability in 12 of 13 findings; member disagreement is below random almost everywhere.
So for rare findings "uncertainty triage" is "suspicion triage", and calibration is what makes
an entropy meaningful.

## E2 — which AI orders the worklist best?

The same curves, one line per model, each ordering by its own probability. MedGemma 27B is
the best model for 9 findings of 14 on NIH and 9 of 11 on VinDr-CXR (median skill 0.72 on
VinDr-CXR against 0.66 for the ensemble and 0.53 for MedGemma 4B). An LLM read-out has a single
member, so its only uncertainty is the entropy of its probability: for it the comparison is
about who puts the sick first, not about uncertainty measures.

## E3 — under load, do sick patients really wait less?

E1 is a count on a fixed list. E3 runs the department: patients arrive over time, one reader
serves them, and the waiting list is served in random order, most-uncertain-first or
most-suspicious-first. Reordering a queue cannot shorten the average wait, it only changes who
waits, so the wait is read separately for the patients who have each finding. Three loads:
half idle, at capacity, overloaded.

What comes out: half idle there is almost no queue, so little to gain (about 9 time units down
to 5 to 7). At capacity sick patients wait roughly a tenth of what they wait in random order;
overloaded they gain thousands of time units (for example Cardiomegaly 6570 -> 717) while the
patients without the finding lose a few hundred. Most-suspicious-first beats
most-uncertain-first, as E1 predicts.

## E4 — what does a reader gain from seeing the AI?

Four personas (senior or junior, skeptical or trusting) read 2000 patients. Every read keeps
the reader's own call next to the team's call, so the effect of the advice is measured on the
same read. Two rules combine the opinions: the linear pool averages the probabilities, the
logarithmic pool adds the evidence after putting both opinions on the same prior.

What comes out: with the linear pool the result is the same whichever AI gives the advice
(senior trusting: +52, +55, +47 missed findings and -809, -810, -810 false alarms). A calibrated
probability of a rare finding is always small, so the AI always votes "absent": the team is
simply a stricter reader, and no information from the AI is used. With the logarithmic pool
missed findings do not increase (they fall slightly), false alarms fall where the AI
discriminates, the better AI helps more (27B: -298, ensemble: -263, 4B: -256 false alarms for
the trusting senior) and the weaker reader gains more.

## Assumptions to keep in view

Personas err independently of the AI (the check G shows the real coupling is small, so the
benefit is an upper bound); the team calls a finding present at 0.5, an arbitrary declared
operating point; the weight of advice comes from non-clinical tasks. Formula sources:
`STUDY_calibrated_advice.md`.
