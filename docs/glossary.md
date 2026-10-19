# Glossary

Every acronym and term of art used in this codebase, defined once. The code expands each
acronym at its first use in a module; this page is the single place to look one up.

Ordered by what you meet first when reading the pipeline, not alphabetically.

- [The arrangement](#the-arrangement) · [The image and the truth](#the-image-and-the-truth) ·
  [Scoring a reader](#scoring-a-reader) · [Uncertainty](#uncertainty)
- [The signal plane](#the-signal-plane) · [When patients arrive](#when-patients-arrive) ·
  [Combining the two readers](#combining-the-two-readers)
- [Measuring a run](#measuring-a-run) · [Running an experiment](#running-an-experiment) ·
  [Models and data](#models-and-data)

## The arrangement

| term | meaning |
|---|---|
| **AI**: artificial intelligence | In this codebase, always *the automated reader*: a chest X-ray model that outputs a probability per finding. Never a person, never a chatbot. |
| **In series** | The arrangement this framework exists to test: the AI reads **first**, its output is used to **reorder the queue**, and the reader reads **second**, seeing the AI's opinion as a hint. Contrast with *in parallel* (both read independently and a rule arbitrates), which is not modelled. |
| **Reader** | Anyone or anything that turns an image into a diagnosis. The AI is a reader; so is the doctor. Both implement the same `DiagnosticActor` contract, which is what lets a study swap one for the other. |
| **Doctor / persona** | The simulated radiologist. A **persona** is a doctor described along three axes: **capability** (how good it is alone), **visibility** (whether it sees the AI at all), **trust** (how much weight it gives the AI when it does). |
| **Solo / aided / team** | *Solo*: the reader's own opinion, formed before the AI's is folded in. *Aided* (= *team* = *hybrid* in the KPI tables): the reader's final call after integrating the AI. Keeping both is what makes reliance measurable. |
| **Playground** | The intended use: change one thing, rerun, see what moved. Every stage is swappable from a YAML file, and nothing downstream needs to know what was swapped. |
| **Discrete-event simulation** | Time advances by jumping to the next event (an arrival, a read finishing), not in fixed ticks. Built on **SimPy**. "Simulated time" is therefore unrelated to how long the Python process runs. |
| **Service time** | How long a reader occupies a station for one patient, in simulated time units. |
| **Saturation** | Patients arriving faster than they can be read, so a backlog builds. **Triage order only matters under saturation**: with an empty queue there is nothing to reorder. |

## The image and the truth

| term | meaning |
|---|---|
| **CXR**: chest X-ray | A chest radiograph, the imaging modality this simulator is built around. |
| **GT**: ground truth | The reference answer a prediction is scored against. Owned by the dataset, not by the simulator. |
| **Label / finding / pathology** | One finding that may be present or absent (Effusion, Pneumothorax, …). Predictions are vectors, one entry per label. |
| **Multi-label** | A patient may have several findings at once, so every actor emits a *vector*, not a single class. |
| **`-1` (not annotated)** | Third ground-truth state, besides 1 (present) and 0 (absent): the dataset never labelled this finding for this image. Every metric **skips** these entries rather than assuming absence. |
| **Prevalence** | The prior probability that a label is positive. Low prevalence is why accuracy is a misleading headline metric here. |
| **`patient_id` vs `example_id`** | `patient_id` counts arrivals in one run (unique even if the same image is drawn twice). `example_id` names the underlying dataset item (the image). Deliberately different: one is a position in a queue, the other an identity. |

## Scoring a reader

| term | meaning |
|---|---|
| **TP / FP / FN / TN** | True positive (flagged, and present) · false positive (flagged, absent: a false alarm) · **false negative** (missed, and present: the clinically dangerous one) · true negative (cleared, absent). |
| **Sensitivity** = **recall** | TP / (TP + FN). Of the findings that were there, how many did the reader catch? Low recall means missed disease. |
| **Specificity** | TN / (TN + FP). Of the absent findings, how many were correctly cleared? |
| **Precision** | TP / (TP + FP). When the reader flags something, how often is it really there? |
| **F1** | Harmonic mean of precision and recall. One number, but it hides *which* of the two moved: always decompose. |
| **Macro average** | Compute per label, then average the labels with equal weight, so a rare finding is not drowned out by a common one. (Contrast *micro*: pool all entries first.) |
| **Accuracy** | Fraction of all (patient, label) entries called correctly. **Misleading under low prevalence**: predicting "nothing is ever present" scores high while catching nothing. Reported for completeness only. |
| **AUROC**: area under the receiver operating characteristic curve | Probability that a random positive gets a higher score than a random negative. Threshold-free, so it measures *ranking* quality rather than decisions. |
| **TTFE / MTTF / MTTFN** | Time to first error · mean time to failure (mean gap between consecutive errors) · the same for false negatives only. Reliability framing: not just *how many* mistakes, but *how they arrive over time*. |
| **Capture curve / skill** | Read the cases in the order a model proposes and plot the share of target cases (positives, or the model's own errors) met so far. An ideal order climbs straight to the top (a right trapezoid), a random one follows the diagonal. *Skill* is the area between the curve and the diagonal as a share of the ideal's: 1 ideal, 0 random. |

## Uncertainty

| term | meaning |
|---|---|
| **UQ**: uncertainty quantification | Attaching "how sure are you?" to each prediction. Here it can drive triage: the queue is reordered by it. |
| **Deep Ensemble** | Several independently trained models predicting the same image. Their **disagreement** is the uncertainty signal. |
| **Aleatoric** uncertainty | Irreducible noise in the data itself (a genuinely ambiguous image). More data would not help. |
| **Epistemic** uncertainty | The model's own ignorance (an image unlike its training data). More data *would* help. |
| **Predictive entropy / mean entropy / mutual information** | The decomposition of an ensemble's uncertainty: predictive entropy = **total**; mean entropy = **aleatoric**; mutual information = total − aleatoric = **epistemic**. All three are recorded per label for every patient. |
| **Calibration** | Whether a stated probability means what it says (of the cases called 0.8, do about 80% turn out positive?). Independent of ranking quality: a model can order cases well and still state meaningless numbers. Fixed after the fact by a swappable `Calibrator`; see [calibration and fusion](calibration-and-fusion.md). |
| **ECE**: expected calibration error | Average gap between stated probability and observed frequency, weighted by how many cases fall in each band of probability. 0 means perfectly calibrated. Measured during development: DenseNet-121 ensemble 0.27, MedGemma 0.80. |
| **Temperature scaling** | The one-parameter calibration (Guo et al. 2017): divide the log-odds by `T`. **Monotone**, so it cannot change any ranking: AUROC and the triage order are untouched, only the numbers move. |
| **Platt scaling** | Two-parameter calibration: `sigmoid(a·logit(p) + b)` (Platt 1999). The intercept is what temperature lacks, and what a model biased toward calling findings *present* needs. Fitted per finding in the studies. |
| **Prevalence-matched threshold** | The cut at which a model calls as many positives as there are, fitted on the calibration half. Used as the AI's own operating point (`ai.decision_threshold: prevalence_matched`), because a calibrated probability of a rare finding almost never reaches 0.5. |

## The signal plane

Three per-read quantities the engine computes between the AI station and the reader, hands
to the reader in `context.signals`, and records as `sig.*`. Neither reader produces them,
and each has several competing definitions, which is why each is a protocol.

| term | meaning |
|---|---|
| **Signal plane** | The channel that delivers these quantities. It lets a per-case quantity reach a reader without every module knowing about every other. |
| **Severity** | How much this case *matters clinically*. Always recorded when configured, and deliberately **not** wired into triage by default: it is available to condition metrics on, and using it for triage is a separate choice. |
| **Difficulty** (`δ`, delta) | How easy this case is to *get wrong*. **Independent of severity**: a large pneumothorax is severe and easy, a subtle nodule mild and hard. The latent quantity that can drive both readers' errors, and so the only route to a non-zero `ρ`. |
| **ρ** (rho): error correlation | How much the reader's errors and the AI's land on the same cases. `ρ = 0` means independent errors, the regime where a team can beat both its members. **Zero unless a reader reads the difficulty signal**, because a simulated reader that never sees the image cannot fail on the same images as the AI. |
| **Induced vs imposed coupling** | *Imposed*: `ρ` is a parameter typed in. *Induced*: difficulty comes from the AI's own predictive uncertainty, so the coupling is a measurement of real model behaviour. The second is why the difficulty model runs *after* inference. |
| **Inter-reader disagreement** | A data-based definition of difficulty: a case is hard when expert readers split on it. VinDr-CXR has three radiologists per training image; `scripts/build_vindr_labels.py` writes their disagreement to `data/vindr/difficulty.csv`, which `difficulty.kind: table` reads. |
| **Fatigue** | How worn the *station* is when it picks a case up. A property of the reader and the clock, not of the case, so two reader workers on one queue tire independently. |
| **Vigilance decrement** | The shape fatigue takes in the sustained-attention literature: a fast early loss that then levels off, not a linear ramp. Modelled as an exponential approach to an asymptote. |
| **Circadian** | Fatigue as a function of the *hour*, independent of workload. A reader who has done nothing all night is still impaired, which is why it combines with a time-on-task curve rather than replacing it. |

## When patients arrive

| term | meaning |
|---|---|
| **Arrival process** | The rule generating the gaps between arrivals, kept separate from *which* patient arrives, so comparing two schedules does not also compare two case mixes. |
| **Homogeneous Poisson process** | Constant rate, exponential gaps, memoryless. The baseline. |
| **NHPP**: non-homogeneous Poisson process | A Poisson process whose rate varies over time: a reading room is busier by day than by night, and on weekdays than at weekends. |
| **Thinning** (Lewis & Shedler 1979) | How an NHPP is sampled exactly: propose arrivals at the peak rate, keep each with probability `λ(t)/λ_max`. No numerical integration, and it works with a tabulated rate as easily as with a formula. |
| **Aggregation artefact** | Why overdispersed arrival models are *not* the default: pooling the same hour across weeks of different intensity creates burstiness that was never in the data (Kim & Whitt 2014). Test before escalating. |

## Combining the two readers

| term | meaning |
|---|---|
| **Integration policy** | The swappable rule that fuses the reader's own opinion with the AI's into the final call. The one place where "how much does this reader trust the AI" lives. |
| **WOA**: weight of advice | How much of an advisor's opinion a judge adopts: 0 means ignored, 1 means adopted wholesale. From the human advice-taking literature, where the pooled average is about 0.39 (Bailey et al. 2023). |
| **Linear pool** | Fusion by mixing probabilities: `p_team = (1−w)·p_reader + w·p_AI`. Cannot become more confident than the more confident reader, so it tolerates a badly calibrated AI. |
| **Log-odds fusion / logarithmic pool** | Fusion by adding evidence on the log-odds scale. Optimal for two independent readers, but an over-confident AI dominates the sum, which is why it needs calibration first. |
| **LLR**: log-likelihood ratio | The evidence one reader contributes, on the log-odds scale: `logit(p) − logit(prior)`. |
| **SDT**: signal detection theory | The standard model of a human observer: noisy evidence plus a decision threshold, summarised by **d′** (discriminability: how well present and absent separate) and the **criterion** (how much evidence is demanded before saying "yes"). Implemented by `SDTDoctor`, the reader of the library personas, and by the synthetic AI (`ai.kind: fixed_accuracy`). |
| **Reliance** | Whether the reader takes the AI's advice *when it should*. **RAIR** (relative AI reliance): of the cases where the reader alone was wrong and the AI right, how often was the team right? **RSR** (relative self-reliance): of the cases where the reader alone was right and the AI wrong, how often did the team hold its ground? Over-reliance = 1 − RSR; under-reliance = 1 − RAIR (Schemmer et al. 2023). |
| **Complementarity gap** | Team score − the better of (AI alone, reader alone). Positive means the pair beats both its members; zero or below means the pairing bought nothing. |
| **Joint table** | The AI × reader × ground-truth contingency: for each sign of the ground truth, the 2×2 count of who said what. Shows *where* the two readers agree and disagree, which the summary ratios compress. |

## Measuring a run

| term | meaning |
|---|---|
| **KPI**: key performance indicator | Any number summarising a run. |
| **Bundle** | A named, swappable group of KPIs computed from the run's records. Bundles are plain functions of the telemetry, so you can write one in your own file and name it in the YAML. |
| **Telemetry / record** | One row per patient: timestamps, every actor's labels and probabilities, the ground truth, plus the open **observation bag**. |
| **Observation bag** | The extensible part of a record. Anything a module publishes reaches it, namespaced: `ai.*` (from the AI), `doc.*` (from the reader), `ctx.*` (patient context), `load.*` (queue occupancy), `sig.*` (the signal plane). This is how a new module exposes a new quantity without changing the record. |
| **Grouping** | A conditioning axis. It splits the records and reruns the *unchanged* bundle on each piece, which is how any metric becomes "per finding", "per severity band" or "under saturation". |
| **Label view** | A projection of the label space for a study: keep all labels, keep a subset, or collapse them into one "abnormal / not". Lets one model serve single-label and multi-label studies without retraining. |

## Running an experiment

| term | meaning |
|---|---|
| **Scenario** | One fully specified run, as a YAML file: which dataset, which AI, which persona, which triage rule, which metrics. |
| **Study** | A set of scenarios generated by varying **factors**. |
| **Factor / level / cell** | A factor is a knob (`persona`); its **levels** are the values tried; a **cell** is one combination of levels. |
| **Replication / seed** | The same cell rerun with a different random seed. |
| **Arm / baseline** | An **arm** is one cell being compared. The **baseline** is the reference arm the others are measured against. |
| **CRN**: common random numbers | Splitting one seed into independent per-module streams, so two arms at the same seed meet *the same patients in the same order*. |
| **Paired contrast** | Comparing two arms **seed by seed** and averaging the differences, rather than comparing two averages. With common random numbers the shared noise is identical in both arms and cancels in the subtraction, so the interval is much tighter. |
| **Confidence interval** | Reported at 95%, two-sided, on Student's *t*. A single replication reports none: one observation has no spread. |

## Models and data

| term | meaning |
|---|---|
| **DenseNet-121 ensemble** | Four pre-trained TorchXRayVision DenseNet-121 classifiers (weights trained on CheXpert, MIMIC-CXR, PadChest and RSNA), averaged into one probability. The NIH-trained weights are left out because they were trained on the NIH ChestX-ray14 images. |
| **MedGemma** | A medical vision-language model (Google): an image encoder joined to a Gemma language model, trained on medical images and text. It answers in text, which is why it needs a *readout*. Version 1.5 exists only at 4 billion parameters; the 27B is the earlier MedGemma 1, so comparing them varies generation as well as size. |
| **Vision-language model** | A model that takes an image and text together and produces text. Compare a **classifier**, which produces one number per class and nothing else. |
| **Readout** | How a probability per finding is extracted from a model that emits text. Not a parsing detail: different readouts elicit different quantities from the same model. |
| **Teacher forcing** | Feeding a model the answer you want it to have written and reading the probability it assigned there, instead of letting it generate. One forward pass gives an answer for every question, but everything forced earlier in the sequence conditions everything after it, which is why the checklist readout is superseded. |
| **Key-value (KV) cache** | What a language model keeps about the tokens it has already read, so it need not re-read them. Reusing it across several questions about the same image is what makes asking them one at a time affordable; **rewinding** it between questions keeps the answers independent. |
| **Shared prefix** | The part of a prompt identical across many queries: here the image plus the instruction, which is nearly all the compute. Run once, reused by every question. |
| **Logit / log-probability** | The model's raw score for a token before, and after, normalising into a probability. Reading them is how a yes/no answer becomes a number. |
| **Prompt ensemble** | Asking the same image several differently worded questions and treating the answers as ensemble members. Where the wordings agree the model is sure; where a paraphrase flips the answer it is not. The generative counterpart of a Deep Ensemble (`prompt_set: triple`). |
| **vLLM** | An open-source inference server for language models. The shipped MedGemma cells were built through its OpenAI-compatible API (`ai.backend: vllm`). |
| **Inference cache / cache cell** | Stored model outputs for a fixed image set, so a study can replay real predictions without rerunning the network. One cell per (model, dataset), named `<dataset>__<model>`; an ensemble is several cells composed by image id. Turns hours into seconds. |
| **NIH ChestX-ray14 / VinDr-CXR** | The two datasets of the shipped caches: 30,805 NIH ChestX-ray14 images (one per patient; in domain for some ensemble members) and the 15,000 VinDr-CXR training images (out of domain for every member). See [`data/README.md`](../data/README.md). |
