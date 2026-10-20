# Outputs and metrics

What `hai-sim run` writes, and how every KPI (key performance indicator) is defined. The
files of a study are described in [studies](studies.md#output-files).

- [Per-patient CSV](#per-patient-csv) · [Observations](#observations) ·
  [Run JSON](#run-json)
- [KPI definitions](#kpi-definitions): [`system`](#system-run-level) ·
  [`actors`](#actors-one-verdict-at-a-time) · [`team`](#team-the-human-ai-team) ·
  [`joint_table`](#joint_table-ai--reader--ground-truth)
- [Reading a result](#reading-a-result)

## Per-patient CSV

`out/records_<name>_<stamp>.csv` has one row per patient.

| column | meaning |
|---|---|
| `patient_id` | sequential id assigned on arrival |
| `ts_arrival` | simulated time the patient entered the input queue |
| `ts_model_start`, `ts_model_end` | the AI station picks the patient up / returns its output |
| `ts_doctor_start`, `ts_doctor_end` | the reader picks the patient up / the verdict is recorded |
| `priority_key` | key of the triaged priority queue; lower is served earlier |
| `gt__<label>` | ground truth per finding: 1 present, 0 absent, −1 not annotated (skipped by every metric) |
| `ai_label__`, `ai_conf__`, `ai_unc__` | the AI's call, probability and uncertainty per finding |
| `doc_label__`, `doc_conf__`, `doc_unc__` | the reader's final call, probability and uncertainty per finding (the team's, when the reader sees the AI) |
| `severity`, `severity__<label>` | present when a `severity` model is configured (scalar or per finding) |
| `ai.*`, `doc.*`, `ctx.*`, `load.*`, `sig.*` | the [observations](#observations) |

Columns ending in `__<label>` repeat once per finding of the active label set.

The unaided reader's call, which an assisted reader keeps for the team and reliance KPIs, is
held in memory and is not a CSV column; its probability is in
`doc.doctor_solo_confidence__<label>`.

### Observations

Beyond the fixed columns, each record carries a namespaced **observation bag**. Whatever a
module puts in `ModelOutput.extra` reaches the telemetry, so a new module can expose a new
quantity with no change to the collector. A per-finding value expands into one column per
finding (`ai.mutual_information__Effusion`); a scalar takes one column.

| namespace | source | examples |
|---|---|---|
| `ai.*` | the AI station's `ModelOutput.extra` | `ai.uq_method`, `ai.image_id`, `ai.cache_index`, `ai.calibrator`, `ai.calibration_stage`; with `uq: ensemble` the full decomposition `ai.predictive_entropy__*` (total), `ai.mean_entropy__*` (aleatoric), `ai.mutual_information__*` (epistemic), `ai.std__*`, plus `ai.n_members` and `ai.coverage__*` |
| `doc.*` | the reader station's `ModelOutput.extra` | `doc.policy`, `doc.assisted`, `doc.ai_hint_used`, `doc.fusion_weight_ai__*` (the weight the policy actually applied, per finding), `doc.doctor_solo_confidence__*`, `doc.team_decision_threshold` |
| `ctx.*` | the patient context | `ctx.example_id` (the image identifier, which joins a record back to its image), `ctx.age`, `ctx.sex`, `ctx.priority` |
| `load.*` | queue occupancy sampled by the engine | `load.input_queue_at_arrival`, `load.in_system_at_arrival`, `load.triaged_at_enqueue` (how deep the triaged queue was when the patient joined it, i.e. whether triage had room to act), `load.triaged_at_read` |
| `sig.*` | the signal plane | `sig.severity`, `sig.difficulty`, `sig.fatigue`, when configured |

`patient_id` counts arrivals and is unique even when the same image is drawn twice;
`ctx.example_id` names the image. They are deliberately different.

Every observation key is also a valid [`kpi_groups`](scenarios.md#kpi_groups) key, and a
custom bundle reads one with `record.observation("ai.mutual_information")`.

## Run JSON

`out/summary_<name>_<stamp>.json`:

```jsonc
{
  "scenario": { /* the validated scenario, every default filled in */ },
  "ai": "CachedEnsembleAI(single#0)",
  "doctor": "Persona[senior_trusting]",
  "sort": "ProbabilityDescSort(max)",
  "wall_clock_s": 2.7,              // Python wall-clock time, not simulated time
  "metadata": {
    "sim_end_time": 27296.2, "n_ai_workers": 1, "n_doctor_workers": 1,
    "severity_model": "none", "difficulty_model": "none", "fatigue_curve": "none"
  },
  "kpis": { /* one entry per bundle */ },
  "kpis_by_group": { /* the same, per subgroup; only with kpi_groups */ }
}
```

## KPI definitions

Computed in [`src/hai_sim/metrics/kpi.py`](../src/hai_sim/metrics/kpi.py). Four built-in
bundles, selected by [`kpis`](scenarios.md#kpis); the default is all four. In a study, a
metric is addressed by its dotted path: `system.mean_wait_to_doctor`,
`actors.hybrid.recall_macro`, `team.rair`, `joint_table.joint.gt1_ai1_doc0`.

Conventions:

- Entries whose ground truth is −1 (not annotated) are skipped everywhere, never counted as
  absent.
- An **error** on a patient means the predicted label vector differs from the ground truth
  on at least one annotated finding (a per-label notion, not exact match).
- Time metrics use `ts_doctor_end` as a patient's event time: the moment the verdict exists.

### `system`: run level

| KPI | definition |
|---|---|
| `n_patients` | patients fully processed |
| `sim_duration` | `max(ts_doctor_end)`, in simulated time units |
| `throughput` | `n_patients / sim_duration`; compare it with the arrival rate to detect saturation |
| `mean_wait_to_model` | mean of `ts_model_start − ts_arrival` |
| `mean_wait_to_doctor` | mean of `ts_doctor_start − ts_model_end`, the wait in the queue the sort order acts on. The sum of the waits does not change when the queue is reordered, so the **mean** does not depend on the sort order; only who waits does |
| `mean_time_in_system` | mean of `ts_doctor_end − ts_arrival` |

### `actors`: one verdict at a time

Every patient leaves three verdicts, and the same metrics are computed for each:

- `ai`: the model alone, i.e. what a deployment without a reader would do;
- `doctor`: the reader's **unaided** call (for an assisted reader, the call it formed before
  seeing the AI);
- `hybrid`: the reader stage's actual output, i.e. the team's call when the reader sees the
  AI. When it does not, `hybrid` and `doctor` coincide.

| KPI | definition | reading |
|---|---|---|
| `f1_macro` | mean over findings of the per-finding F1 | one number; every finding weighted equally |
| `precision_macro` | mean over findings of TP / (TP + FP) | when it says positive, how often is it right? |
| `recall_macro` | mean over findings of TP / (TP + FN), the sensitivity | of the findings present, how many are caught? Low recall means many misses |
| `accuracy` | (ΣTP + ΣTN) / all annotated entries | misleading at low prevalence: calling everything absent scores high. Reported for completeness |
| `n_fn` | (patient, finding) pairs with truth 1 and call 0 | missed findings, the clinically worst error |
| `n_fp` | (patient, finding) pairs with truth 0 and call 1 | false alarms |
| `ttfe` | time to first error: `ts_doctor_end` of the first patient with an error | how long before the system slips; `None` if it never errs |
| `mttf` | mean time to failure: mean gap between consecutive error events | large means errors are spread out, small means bursts; `None` with fewer than 2 errors |
| `mttfn` | the same, counting only patients with at least one false negative | the miss-focused version; `None` with fewer than 2 such patients |

TP, FP, FN and TN are true positives, false positives, false negatives and true negatives.
To get one row per finding instead of a macro average, add `kpi_groups: [{key: label}]`.

### `team`: the human-AI team

Because the simulator knows both the unaided call and the team call for every case, the
appropriate-reliance metrics, which normally need a two-stage reader study, are computed
directly. They follow Schemmer et al. (2023). Each annotated (patient, finding) entry is
sorted by *was the reader alone right?* × *was the AI right?*.

| KPI | definition | reading |
|---|---|---|
| `complementarity_gap_f1` | team macro-F1 − max(AI macro-F1, unaided macro-F1) | above 0, the pair beats both its members |
| `complementarity_gap_accuracy` | the same gap on accuracy | prefer the F1 gap at low prevalence |
| `rair` | relative AI reliance: of the entries where the reader alone was wrong and the AI right, the fraction the team got right | did the team take correct advice? |
| `rsr` | relative self-reliance: of the entries where the reader alone was right and the AI wrong, the fraction the team got right | did the team resist wrong advice? |
| `over_reliance` | 1 − `rsr` | own correct answers lost to a wrong AI |
| `under_reliance` | 1 − `rair` | correct advice ignored |
| `n_ai_reliance_opportunities` | entries where the reader alone was wrong and the AI right | denominator of `rair` |
| `n_self_reliance_opportunities` | entries where the reader alone was right and the AI wrong | denominator of `rsr` |

A ratio over a handful of opportunities is noise: check the denominators. With a reader who
does not see the AI, the team is the reader alone: `rair` is 0, `rsr` is 1, and the gap is 0
or below.

### `joint_table`: AI × reader × ground truth

The contingency behind the ratios above, over the annotated (patient, finding) entries:

- `n_entries`: entries counted;
- `ai_confusion`, `doctor_confusion`: the two confusion matrices (`tp`, `fp`, `fn`, `tn`),
  the second for the reader's final (team) call;
- `joint`: eight cells `gt{g}_ai{a}_doc{d}` with g, a, d in {1, 0}: for each sign of the
  ground truth, how often the AI and the reader said positive or negative. It shows where
  the two agree, where they disagree, and who is right when they do.

## Reading a result

- `recall_macro` and `n_fn`: the clinical axis (missed disease). Look at these first.
- `precision_macro` and `n_fp`: the workload axis (false alarms).
- `mttfn` and `ttfe`: *when* errors accumulate, which is what the worklist order can move.
- `mean_wait_to_doctor` and `throughput`: saturation; they tell whether the load settings
  had an effect. For order effects, split the waits by ground truth
  (`kpi_groups: [{key: truth}]`).

Never judge a change by `f1_macro` or `accuracy` alone: always split it into misses (FN) and
false alarms (FP), per finding.
