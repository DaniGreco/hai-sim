# Study reference

A study expands a **factor grid** over a base [scenario](scenarios.md) into one run per cell
per seed, then tabulates chosen KPI paths into one tidy table. It is validated by
[`src/hai_sim/study/spec.py`](../src/hai_sim/study/spec.py) (unknown keys raise) and run with
`hai-sim study <study.yaml> [--workers N]`.

- [Example](#example) · [Fields](#fields) · [Designs](#designs)
- [Paired contrasts](#paired-contrasts) · [Output files](#output-files)
- [The studies in `configs/studies/`](#the-studies-in-configsstudies) ·
  [Capture study](#capture-study)

## Example

```yaml
name: persona_grid
base:                    # an inline scenario, or a path to a scenario YAML
  name: base
  seed: 0
  label_schema: default_v0
  patient_generator: {kind: mock_dataset, arrival_rate: 0.5, n_patients: 50, n_examples: 80}
  ai: {kind: random_mock}
  persona: solo_senior
  sort: {kind: uncertainty_desc, aggregator: mean}

factors:                 # dotted paths into the scenario -> levels
  persona: [solo_senior, senior_trusting, senior_skeptical, junior_trusting, junior_skeptical]
  sort.aggregator: [mean, max]

design: grid
replications: [1, 2, 3]  # seeds: each arm reads the same patient stream at the same seed

baseline:                # the reference arm -> paired contrasts
  persona: solo_senior

by:                      # same shape as a scenario's kpi_groups
  - key: load.in_system_at_arrival
    edges: [5.0, 20.0]

kpis: [system, actors, team]

report:                  # omit to tabulate every scalar in the report
  - team.complementarity_gap_f1
  - team.rair
  - actors.hybrid.recall_macro
  - actors.hybrid.n_fn
```

A factor is a dotted path into the scenario with a list of levels. A level may be a whole
block, so `persona` can take a list of preset names and `ai` a list of complete AI blocks
told apart by their `name` (see `configs/studies/team_effect.yaml`). Every cell runs once per
seed in `replications`, and that seed replaces the `seed` of the base scenario.

## Fields

<!-- fields: StudyConfig -->
| field | values | default | meaning |
|---|---|---|---|
| `name` | text | `unnamed_study` | Name of the study, used in the output file names. |
| `base` | path or inline scenario | required | The scenario every cell starts from. |
| `factors` | mapping of dotted path to list of levels | required | The varied fields. Every factor needs at least one level. |
| `design` | `grid`, `ofat`, `random` | `grid` | How the levels are combined into cells, see [designs](#designs). |
| `n_samples` | integer | none; required for `random` | Cells drawn from the grid. |
| `design_seed` | integer | `0` | `random`. Seed of the draw. |
| `replications` | list of integers | `[0]` | Seeds. One run per cell per seed; the same seed gives every cell the same patient stream. |
| `report` | list of metric paths | empty, every scalar | Metrics tabulated, written as `bundle.metric`, e.g. `team.rair` or `system.mean_wait_to_doctor`. |
| `kpis` | list of names | none: derived from `report`, else taken from the base scenario | Bundles computed; a registered name or `path/to/file.py:Attribute` ([`kpis`](scenarios.md#kpis)). |
| `by` | list of blocks | empty | Conditioning axes, as [`kpi_groups`](scenarios.md#kpi_groups). Every metric is also reported per subgroup, in a `group` column. |
| `baseline` | mapping of factor to level | none | The reference cell of the [paired contrasts](#paired-contrasts). Every named factor must exist, and every level must be among its levels. |

Cells whose built scenario is identical (a repeated level, two paths that resolve to the
same configuration) are simulated once and share the result. A factor that is inert but
still changes the configuration (an aggregator under a random sort) still counts as a
distinct scenario.

## Designs

| design | what it runs |
|---|---|
| `grid` | the full cross-product of the levels |
| `ofat` | one factor at a time: a baseline cell (the first level of each factor) plus one cell per remaining level, so the number of cells grows linearly, not exponentially, with the number of factors |
| `random` | `n_samples` cells drawn from the grid (`design_seed` fixes which). Not a Latin hypercube: factors are lists of levels, with no interval to stratify. The `baseline` cell is always kept |

## Paired contrasts

The engine splits each run's seed into independent per-module streams, so **two arms at the
same seed read the same patients in the same order** (common random numbers). Declaring
`baseline` turns that into a number: the reported delta is the mean of the *per-seed
differences* between an arm and its baseline, with a 95% interval on those differences. The
shared sampling noise cancels instead of being averaged over, so the interval is much tighter
than a comparison of two independent means. Seeds present in only one arm are skipped.

A partial baseline, which names only some factors, contrasts each arm with the cell that
differs in *those* factors alone. For example, with `baseline: {sort.kind: random}` in a
grid of arrival rate × sort order, every order is compared with the random order *at the
same arrival rate*, so a contrast never mixes two changes.

Confidence intervals are two-sided 95% intervals on Student's *t* with n − 1 degrees of
freedom; `study/stats.py` carries its own table of critical values, so no SciPy is needed,
and uses the normal value 1.96 past 30 degrees of freedom. A single replication reports no
interval.

## Output files

All written to `--out-dir` (default `out/`):

| file | shape |
|---|---|
| `study_<name>_<stamp>.csv` | wide: one row per (run, subgroup), one column per metric path |
| `study_<name>_<stamp>_long.csv` | **canonical**: one row per (cell, seed, subgroup, metric), columns `<factors>, seed, group, metric, value`; the shape to group and plot |
| `study_<name>_<stamp>_summary.csv` | per (cell, subgroup, metric): `n, mean, sd, ci_low, ci_high`, and against the baseline `vs_baseline_delta, vs_baseline_ci_low, vs_baseline_ci_high, vs_baseline_n_pairs` |
| `study_<name>_<stamp>.json` | everything above, plus the validated study configuration |

A factor level that is a whole block is written as its `name`. The subgroup keys are those of
[`kpi_groups`](scenarios.md#kpi_groups), e.g. `label=Effusion` or `truth=Effusion:positive`.
`hai-sim plot` draws the figures of the shipped studies from these files
([usage](usage.md#plot)).

## The studies in `configs/studies/`

All of them replay the cache cells and read the odd cache rows (the calibration files were
fitted on the even ones). The four simulation studies read every VinDr-CXR image of those
rows once per run (`sample_mode: permutation`, 7,500 patients) and use 20 seeds.

| file | design | paper |
|---|---|---|
| `capture.yaml` | Capture study: for each dataset, model and finding, how early an ordering meets the positive cases and the model's own false negatives and false positives. Not a simulation, see [below](#capture-study). | RQ1 and RQ2 |
| `queue_load.yaml` | Waiting times on VinDr-CXR with the DenseNet-121 ensemble ordering the worklist: three arrival rates (0.04, 0.075, 0.15; about half idle, at capacity and overloaded for one reader) × three orders (random, most uncertain first, most suspicious first). Every time metric is split by ground truth (`by: truth`) and paired against the random order. | |
| `queue_load_medgemma27b.yaml` | The same design with MedGemma 27B ordering the worklist, on the ten findings both datasets annotate. | RQ3 |
| `team_effect.yaml` | The human-AI team on VinDr-CXR: four personas (senior or junior, skeptical or trusting) × two fusion rules (linear pool, logarithmic pool) × three AIs (DenseNet-121 ensemble, MedGemma 1.5 4B, MedGemma 27B), every KPI per finding. | |
| `team_effect_medgemma27b.yaml` | The same design with MedGemma 27B as the only AI, on the ten findings both datasets annotate. | RQ4 |

The VinDr-CXR cells need their labels rebuilt once with
`uv run python scripts/build_vindr_labels.py` ([`data/README.md`](../data/README.md)).

## Capture study

`hai-sim capture <capture.yaml>` reads a different file, validated by
[`src/hai_sim/analysis/capture_study.py`](../src/hai_sim/analysis/capture_study.py). It reads
the images of a cache in the order a model proposes and counts, per finding, how many target
cases have been met so far. An ideal order meets every target first (a right trapezoid); a
random order is the diagonal. `skill` is the area between the curve and the diagonal as a
share of the area between the ideal and the diagonal (1 = ideal, 0 = random, below 0 = worse
than random); `auroc` is the probability that a target is ranked ahead of a non-target.

```yaml
name: capture
datasets: [nih, vindr]
models:
  ensemble:
    label: DenseNet-121 ensemble (4 members)
    cells: [chex, mimic_ch, pc, rsna]     # cells averaged into one probability
    calibration: ens4                     # data/cache/calibration/<dataset>__ens4.json
  medgemma-27b:
    label: MedGemma 27B
    cells: [medgemma-27b]
orderings: [probability, predictive_entropy, mean_entropy, mutual_information]
probabilities: [raw, calibrated]
targets: [positive, fn, fp]
split: odd
min_targets: 20
```

| field | values | default | meaning |
|---|---|---|---|
| `name` | text | required | Name of the output folder, `<out-dir>/<name>/`. |
| `cache_dir` | path | `data/cache` | Where the cells are. |
| `calibration_dir` | path | `data/cache/calibration` | Where the calibration files are. |
| `datasets` | list of dataset ids | required | The `<dataset>` half of the cell stems. |
| `models` | mapping of key to model | required | Each model has `cells` (at least one cell alias, composed by image id), an optional `label` for the figures and an optional `calibration` file stem (default: the model key). |
| `orderings` | `probability`, `predictive_entropy`, `mean_entropy`, `mutual_information` | all four | Orders tried, highest first. `mean_entropy` and `mutual_information` need at least two members covering the finding. |
| `probabilities` | `raw`, `calibrated` | both | For `predictive_entropy`, the probability its entropy is computed on. Calibration does not change the `probability` order, but it changes which cases sit near 0.5. |
| `targets` | `positive`, `fn`, `fp`, `error` | `positive`, `fn`, `fp` | What is counted: images with the finding, the model's false negatives, its false positives, or both errors. The model's call is its probability at the prevalence-matched threshold, fitted on the other rows. |
| `split` | `odd`, `even`, `all` | `odd` | Cache rows evaluated. |
| `min_targets` | integer ≥ 1 | `20` | A finding enters the study only with at least this many targets on the evaluated rows. |

The output folder holds `capture.csv` (one row per dataset, model, finding, target, ordering
and probability kind, with `n_images`, `n_targets`, `skill` and `auroc`), `config.json` and
`figures/`.
