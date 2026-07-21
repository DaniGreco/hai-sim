# Glossary

Every acronym and term of art used in this codebase, defined once. The code expands each
acronym at its first use in a module; this file is the single place to look one up.

Ordered by what you meet first when reading the pipeline, not alphabetically.

---

## The arrangement

| Term | Meaning |
|---|---|
| **AI** — artificial intelligence | In this codebase, always *the automated reader*: a chest X-ray classifier that outputs a probability per pathology. Never a person, never a chatbot. |
| **In series** | The arrangement this framework exists to test: the AI reads **first**, its output is used to **reorder the queue**, and the doctor reads **second**, seeing the AI's opinion as a hint. Contrast with *in parallel* (both read independently and a rule arbitrates) — not modelled here. |
| **Reader** | Anyone or anything that turns an image into a diagnosis. The AI is a reader; so is the doctor. Both implement the same `DiagnosticActor` contract, which is what lets a study swap one for the other. |
| **Doctor / persona** | The simulated radiologist. A **persona** is a doctor specified along three axes: **capability** (how good it is alone), **visibility** (whether it sees the AI at all), **trust** (how much weight it gives the AI when it does). |
| **Solo / aided / team** | *Solo* = the doctor's own opinion, formed before the AI's is folded in. *Aided* (= *team* = *hybrid* in the KPI tables) = the doctor's final call after integrating the AI. Keeping both is what makes reliance measurable. |
| **Playground** | The intended use: change one thing, rerun, see what moved. Every stage is swappable from a YAML file, and nothing downstream needs to know what you swapped. |
| **Discrete-event simulation** | Time advances by jumping to the next event (an arrival, a read finishing), not in fixed ticks. Built on **SimPy**. "Simulated time" is therefore unrelated to how long the Python process runs. |
| **Service time** | How long a reader occupies a station for one patient, in simulated time units. |
| **Saturation** | Patients arriving faster than they can be read, so a backlog builds. **Triage order only matters under saturation** — with an empty queue there is nothing to reorder. |

## The image and the truth

| Term | Meaning |
|---|---|
| **CXR** — chest X-ray | A chest radiograph. The imaging modality this simulator is built around. |
| **GT** — ground truth | The reference answer a prediction is scored against. Owned by the dataset, not by the simulator. |
| **Label / pathology** | One finding that may be present or absent (Effusion, Pneumothorax, …). Predictions are vectors: one entry per label. |
| **Multi-label** | A patient may have several findings at once, so every actor emits a *vector*, not a single class. |
| **`-1` (not annotated)** | Third ground-truth state, besides 1 (present) and 0 (absent): this dataset never labelled this pathology for this image. Every metric **skips** these entries rather than assuming absence. |
| **Prevalence** | The prior probability that a label is positive. Low prevalence is why accuracy is a misleading headline metric here. |
| **`patient_id` vs `example_id`** | `patient_id` counts arrivals in one run (unique even if the same image is drawn twice). `example_id` names the underlying dataset item (the image filename). Deliberately different: one is a position in a queue, the other is an identity. |

## Scoring a reader

| Term | Meaning |
|---|---|
| **TP / FP / FN / TN** | True positive (flagged, and present) · false positive (flagged, absent — a false alarm) · **false negative** (missed, and present — the clinically dangerous one) · true negative (cleared, absent). |
| **Sensitivity** = **recall** | TP / (TP + FN). Of the findings that were there, how many did the reader catch? Low recall = missed disease. |
| **Specificity** | TN / (TN + FP). Of the healthy cases, how many were correctly cleared? |
| **Precision** | TP / (TP + FP). When the reader flags something, how often is it really there? |
| **F1** | Harmonic mean of precision and recall. One number, but it hides *which* of the two moved — always decompose. |
| **Macro average** | Compute per label, then average the labels with equal weight, so a rare pathology is not drowned out by a common one. (Contrast *micro*: pool all entries first.) |
| **Accuracy** | Fraction of all (patient, label) entries called correctly. **Misleading under low prevalence**: predicting "nothing is ever present" scores high while catching nothing. Reported for completeness only. |
| **AUROC** — area under the receiver operating characteristic curve | Probability that a random positive gets a higher score than a random negative. Threshold-free, so it measures *ranking* quality rather than decisions. |
| **TTFE / MTTF / MTTFN** | Time to first error · mean time to failure (mean gap between consecutive errors) · mean time to a false negative. Reliability framing: not just *how many* mistakes, but *how they arrive over time*. |

## Uncertainty

| Term | Meaning |
|---|---|
| **UQ** — uncertainty quantification | Attaching "how sure are you?" to each prediction. Here it drives triage: the queue is reordered by it. |
| **Deep Ensemble** | Several independently trained models predicting the same image. Their **disagreement** is the uncertainty signal. |
| **Aleatoric** uncertainty | Irreducible noise in the data itself (a genuinely ambiguous image). More data would not help. |
| **Epistemic** uncertainty | The model's own ignorance (an image unlike its training data). More data *would* help. This is the one that ought to flag the errors. |
| **Predictive entropy / mean entropy / mutual information** | The decomposition of an ensemble's uncertainty: predictive entropy = **total**; mean entropy = **aleatoric**; mutual information = total − aleatoric = **epistemic**. All three are recorded per label for every patient. |
| **Calibration** | Whether a stated probability means what it says (of the cases called 0.8, do ~80% turn out positive?). **Temperature scaling** is the standard post-hoc fix. Not yet implemented — noted because log-odds fusion assumes it. |

## Combining the two readers

| Term | Meaning |
|---|---|
| **Integration policy** | The swappable rule that fuses the doctor's own opinion with the AI's into the final call. The one seam where "how much does this doctor trust the AI" lives. |
| **WOA** — weight of advice | How much of the advisor's opinion a judge adopts: 0 = ignored entirely, 1 = adopted wholesale. From the human advice-taking literature, where the pooled average is ≈ 0.39. |
| **Linear pool** | Fusion by mixing probabilities: `p_team = (1−w)·p_doctor + w·p_AI`. Cannot become more confident than the more confident reader, so it tolerates a badly calibrated AI. |
| **Log-odds fusion** | Fusion by adding evidence on the log-odds scale. Optimal for two independent readers — but an over-confident AI dominates the sum, which is why it needs calibration first. |
| **LLR** — log-likelihood ratio | The evidence one reader contributes, on the log-odds scale: `logit(p) − logit(prior)`. |
| **SDT** — signal detection theory | The standard model of a human observer: noisy evidence plus a decision threshold, summarised by **d′** (discriminability — how well the two cases separate) and **criterion** (how much evidence is demanded before saying "yes"). Not yet implemented; the planned successor to the fixed-accuracy doctor. |
| **Reliance** | Whether the doctor takes the AI's advice *when it should*. **RAIR** (relative AI reliance) = of the cases where the doctor alone was wrong and the AI right, how often did the team end up right? **RSR** (relative self-reliance) = of the cases where the doctor alone was right and the AI wrong, how often did the team hold its ground? Over-reliance = 1 − RSR; under-reliance = 1 − RAIR. |
| **Complementarity gap** | Team score − the better of (AI alone, doctor alone). Positive means the pair genuinely beats both its members; ≤ 0 means the pairing bought nothing. |
| **Joint table** | The AI × doctor × ground-truth contingency: for each ground-truth sign, the 2×2 count of who said what. Shows *where* the two readers agree and disagree, which the summary ratios compress away. |

## Measuring a run

| Term | Meaning |
|---|---|
| **KPI** — key performance indicator | Any number summarising a run. |
| **Bundle** | A named, swappable group of KPIs computed from the run's records. Bundles are plain functions over the telemetry, so you can write one in your own file and name it in the YAML. |
| **Telemetry / record** | One row per patient: timestamps, every actor's labels and confidences, the ground truth, plus the open **observation bag**. |
| **Observation bag** | The extensible part of a record. Anything a module publishes reaches it, namespaced: `ai.*` (from the AI), `doc.*` (from the doctor), `ctx.*` (patient context), `load.*` (queue occupancy). This is how a new module exposes a new measurable without changing the record. |
| **Grouping** | A conditioning axis. It splits the records and reruns the *unchanged* bundle on each piece, which is how any metric becomes "per severity band", "per pathology", "under saturation". |
| **Severity** | A per-patient clinical-weight signal, produced by a swappable model and always recorded. Deliberately **not** wired into triage by default: it is available to condition on, and consuming it is a separate choice. |
| **Label view** | A projection of the label space for a study: keep all labels, keep a subset, or collapse to a single "abnormal / not". Lets one model serve single-label and multi-label studies without retraining. |

## Running an experiment

| Term | Meaning |
|---|---|
| **Scenario** | One fully specified run, as a YAML file: which dataset, which AI, which persona, which triage rule, which metrics. |
| **Study** | A set of scenarios generated by varying **factors**. |
| **Factor / level / cell** | A factor is a knob (`persona`); its **levels** are the values tried; a **cell** is one combination of levels. |
| **Replication / seed** | The same cell rerun with a different random seed. |
| **Arm / baseline** | An **arm** is one cell being compared. The **baseline** is the reference arm everything else is measured against. |
| **CRN** — common random numbers | Splitting one seed into independent per-module streams, so two arms at the same seed meet *the same patients in the same order*. |
| **Paired contrast** | Comparing two arms **seed by seed** and averaging the differences, rather than comparing two averages. Because common random numbers make the shared noise identical in both arms, it cancels in the subtraction — which is why the resulting interval is far tighter. |
| **Confidence interval** | Reported at 95%, two-sided, on Student's *t*. A single replication reports `None`, not zero: one observation has no spread. |

## Beyond the simulation

| Term | Meaning |
|---|---|
| **MedGemma** | A medical vision-language model (Google): an image encoder bolted to a Gemma language model, trained on medical images and text. It answers in prose, which is why it needs a *readout*. Version 1.5 exists only at 4 billion parameters; the 27B is the earlier MedGemma 1, so comparing them varies generation as well as size. |
| **Vision-language model** | A model that takes an image and text together and produces text. Compare a **classifier**, which produces one number per class and nothing else. |
| **Readout** | How a probability per pathology is extracted from a model that emits text. Not a parsing detail — different readouts elicit genuinely different quantities from the same model. |
| **Teacher forcing** | Feeding a model the answer you want it to have written and asking what probability it assigned there, instead of letting it generate. One forward pass, and an answer for every question. |
| **Logit / log-probability** | The model's raw score for a token before, and after, normalising into a probability. Reading logits is how a yes/no answer becomes a number. |
| **Prompt ensemble** | Asking the same image several differently-worded questions and treating the answers as ensemble members. Where the wordings agree the model is genuinely sure; where a paraphrase flips the answer it is not. The generative stand-in for a Deep Ensemble. |
| **ECE** — expected calibration error | Average gap between stated probability and observed frequency, weighted by how many cases fall in each band. 0 = perfectly calibrated. |
| **κ (kappa)** — Cohen's kappa | Agreement between two judgements, corrected for the agreement expected by chance. Used here on two models' *error* indicators: κ near 0 = they fail on unrelated cases (so pairing them can help), κ high = they fail together (so it cannot). |
| **Grad-CAM / CAM** — (gradient-weighted) class activation map | A heatmap of where in the image a network looked when predicting a label. Used to check the model attends to the right anatomy. |
| **IoU / Dice** | Two ways to score overlap between a predicted region and an expert's outline. IoU = intersection / union; Dice = 2·intersection / (sum of areas). |
| **Pointing game** | A laxer localization check: does the heatmap's hottest point fall inside the expert's outline at all? |
| **Inference cache** | Stored model outputs for a fixed image set, so a study can replay real predictions without rerunning the network. Turns hours into seconds. |
| **NHPP** — non-homogeneous Poisson process | Arrivals whose rate varies with time of day and day of week. Not yet implemented; arrivals are currently a constant-rate Poisson process. |
| **MRMC** — multi-reader multi-case | The standard study design for comparing readers, where both readers and cases vary. Named here as the eventual way to model a *population* of doctors rather than one. |
