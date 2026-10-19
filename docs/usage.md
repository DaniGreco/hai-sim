# Usage

How to install HaiSIM, run it, and reproduce the studies of the paper.

- [Install](#install) · [First run](#first-run) · [Reproduce the paper](#reproduce-the-paper)
- [Command line](#command-line) · [Pipeline](#pipeline) · [Inference caches](#inference-caches)
- [Determinism and seeds](#determinism-and-seeds) · [Tests](#tests) · [Windows notes](#windows-notes)

## Install

HaiSIM needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From the repository
root:

```bash
uv sync --extra dev --extra analysis
```

The core package (SimPy, NumPy, pydantic, PyYAML, rich) is enough to run scenarios and
studies on the shipped caches. The optional extras, declared in `pyproject.toml`:

| extra | adds | needed for |
|---|---|---|
| `analysis` | matplotlib, SciPy, kagglehub | `hai-sim capture`, `hai-sim plot`, `scripts/paper_figures.py`, and the label download of `scripts/build_vindr_labels.py` |
| `dev` | pytest, pytest-cov, mypy, ruff | tests, type check, lint |
| `models` | PyTorch, torchvision, TorchXRayVision, scikit-image, kagglehub, transformers, accelerate, Pillow | running the models live: building new cache cells, or a scenario that reads the pixels |

The `models` extra is heavy and is not needed to reproduce the paper:

```bash
uv sync --extra dev --extra analysis --extra models
```

This installs the default PyTorch build from PyPI, which on Linux includes CUDA. For a
CPU-only build, install into the environment with the PyTorch CPU index instead:

```bash
uv venv
uv pip install -e ".[dev,analysis,models]" --extra-index-url https://download.pytorch.org/whl/cpu
```

and then run commands from the activated environment, or with `uv run --no-sync`, so that
`uv run` does not re-sync the environment to the lock file.

TorchXRayVision downloads its weights on first use (about 80 MB per weight set) to
`~/.torchxrayvision/models_data/`. The MedGemma weights are gated: accept the Health AI
Developer Foundations terms on Hugging Face and log in (`huggingface-cli login`) before the
first run. The shipped MedGemma cells were built with a vLLM server; the commands are at the
top of `configs/cache/*_medgemma_*.yaml`.

## First run

```bash
uv run hai-sim run configs/examples/quickstart.yaml
```

[`quickstart.yaml`](../configs/examples/quickstart.yaml) replays MedGemma 27B on 2,000 NIH
ChestX-ray14 patients from the shipped cache: no image and no download is needed, and the
run takes a few seconds. It prints how often the AI alone, the reader alone and the two
together miss a finding or raise a false alarm, and how long patients wait. The per-patient
records and the run summary go to `out/` ([outputs](outputs.md)). Change one line, run again,
and compare.

Always run from the repository root, so that the repository-relative paths in the YAML files
resolve.

## Reproduce the paper

The studies replay the inference caches in `data/cache` and never open an image. The
VinDr-CXR cells ship without their labels, which must be rebuilt once (this needs a Kaggle
API token; see [`data/README.md`](../data/README.md)):

```bash
uv run python scripts/build_vindr_labels.py                                   # once

uv run hai-sim capture configs/studies/capture.yaml                           # RQ1 and RQ2
uv run hai-sim study configs/studies/queue_load_medgemma27b.yaml --workers 8  # RQ3
uv run hai-sim study configs/studies/team_effect_medgemma27b.yaml --workers 8 # RQ4
uv run python scripts/paper_figures.py                 # figures, table and numbers, to out/paper/
```

`paper_figures.py` reruns the capture study and reads the newest RQ3 and RQ4 tables in
`out/` (`--queue-summary` and `--team-long` pick others).

Two more studies run the same designs with other AIs: `queue_load.yaml` orders the worklist
with the DenseNet-121 ensemble, and `team_effect.yaml` compares the advice of all three AIs.
Their figures come from `hai-sim plot`:

```bash
uv run hai-sim study configs/studies/queue_load.yaml --workers 8
uv run hai-sim plot queue_load out/study_queue_load_<stamp>_summary.csv

uv run hai-sim study configs/studies/team_effect.yaml --workers 8
uv run hai-sim plot team_rules out/study_team_effect_<stamp>_long.csv
uv run hai-sim plot actor_errors out/study_team_effect_<stamp>_long.csv --ai medgemma-27b
```

The study files are described in [studies](studies.md#the-studies-in-configsstudies).

## Command line

```
hai-sim run <scenario.yaml>     [--out-dir out]
hai-sim study <study.yaml>      [--out-dir out] [--workers N]
hai-sim capture <capture.yaml>  [--out-dir out/capture]
hai-sim plot <kind> <csv>       [--out STEM] [--persona P] [--fusion F] [--ai A] [--mean-read T]
```

### `run`

One [scenario](scenarios.md): one simulation, a printed KPI table, and two files:

- `<out-dir>/records_<name>_<stamp>.csv`, one row per patient;
- `<out-dir>/summary_<name>_<stamp>.json`, the validated scenario and every KPI.

See [outputs](outputs.md) for both.

### `study`

One [study](studies.md): a factor grid expanded into one run per cell per seed, then a table
of the chosen KPIs. Writes `study_<name>_<stamp>.csv`, `_long.csv`, `_summary.csv` and
`.json` to `--out-dir`, prints a preview and, when the study declares a `baseline`, the
paired contrasts.

| flag | default | meaning |
|---|---|---|
| `--out-dir` | `out` | Output folder. |
| `--workers` | `1` | Processes for the distinct scenarios. Every run carries its own seed, so the result does not depend on the order of execution. |

### `capture`

The offline capture study (RQ1 and RQ2 of the paper): read the images of a cache in the
order a model proposes and count the positive cases, or the model's own errors, met along the
way, per finding. No simulation; only the cache cells are read. Writes
`<out-dir>/<name>/capture.csv`, `config.json` and `figures/`. Needs the `analysis` extra. The
configuration is described in [studies](studies.md#capture-study).

### `plot`

The figure of a simulation study, from the CSV the study wrote. The plot also reads the
study JSON written next to the CSV, to label the figure. Figures go to
`out/figures/<kind>.png` and `.pdf` unless `--out` says otherwise. Needs the `analysis` extra.

| kind | reads | shows |
|---|---|---|
| `queue_load` | `*_summary.csv` | mean wait of the patients who have each finding, per worklist order and load |
| `team_effect` | `*_long.csv` | per finding, the change in the reader's misses and false alarms when one reader sees each AI |
| `team_rules` | `*_long.csv` | all findings together, every persona under both fusion rules, one marker per AI |
| `actor_errors` | `*_long.csv` | per finding, the miss and false-alarm rates of the AI alone, the reader alone and the team; needs a study with `sample_mode: permutation` |

| flag | default | applies to | meaning |
|---|---|---|---|
| `--out` | `out/figures/<kind>` | all | Output stem, without extension. |
| `--persona` | `senior_trusting` | `team_effect`, `actor_errors` | Which reader. |
| `--fusion` | `log_odds_fusion` | `team_effect`, `actor_errors` | Which integration rule. |
| `--ai` | `ensemble` | `actor_errors` | Which AI gives the advice (the `name` of its `ai` block). |
| `--mean-read` | `13.2` | `queue_load` | Mean reading time of the reader; waits are shown in multiples of it. |

## Pipeline

```
PatientGenerator
    │  arrivals, one at a time
    ▼
input_queue          first-in-first-out store
    │
    ▼
model_worker  × n_ai_workers         ← the AI reads first
    │  per-finding probability and uncertainty;
    │  the SortStrategy turns them into one priority
    ▼
triaged_store        priority store: this is where triage happens
    │  lower priority value is served first
    ▼
doctor_worker × n_doctor_workers     ← the reader reads second,
    │                                  with the AI's output as a hint
    ▼
MetricsCollector     one record per patient
```

Built on SimPy, a discrete-event simulator: time jumps from one event to the next (an
arrival, a read finishing), so "simulated time" has nothing to do with how long the process
runs.

Every actor, AI model and reader alike, implements the same `DiagnosticActor` protocol,
`diagnose(images, *, context, ai_hint) -> ModelOutput`, so any implementation can be swapped
from the scenario YAML ([architecture](architecture.md)).

The AI sits *upstream of the queue*, not beside it. Having read every patient, it can say
which cases it found hardest **before** the reader picks the next one. That is what makes
triage by uncertainty possible, and it changes outcomes only when patients are waiting: an
order study has to run at a load that saturates the reader.

Three properties of the design matter for the metrics:

- **Uncertainty stays per finding** through the `UQMethod`. Only the `SortStrategy`
  collapses it to one priority. The full per-finding vector is written to the CSV, so the
  queue can be re-triaged offline with another aggregator without rerunning inference.
- **The reader's own opinion is formed before the AI's is folded in.** `AssistedDoctor`
  keeps the unaided call in `extra["doctor_solo_labels"]`, so every patient has three
  verdicts: the AI alone, the reader alone and the team. The team and reliance KPIs follow
  from that.
- **Only the reader stage sees the ground truth**, as the device that makes a simulated
  reader err at a chosen rate. An image-reading model never receives it.

Between the two stations the engine computes the **signal plane**: severity, difficulty and
fatigue. They reach the reader in `context.signals` and the telemetry as `sig.*`
([scenarios](scenarios.md#severity)).

## Inference caches

Running a DenseNet ensemble or a 27-billion-parameter language model on tens of thousands of
images takes hours; replaying its stored output takes seconds. A **cache cell** holds the raw
probabilities of one model over one dataset, keyed by image id, in two files:
`data/cache/<dataset>__<model>.npz` and `.json`. An ensemble is not a cell: it is several
member cells composed by image id at replay time. The cells shipped with the repository, and
how to get the labels and images back, are described in
[`data/README.md`](../data/README.md).

Two ways to replay:

- `ai.kind: cached` with `ai.cache_stem` replays one cell directly, and
  `patient_generator.kind: inference_cache` draws the patients (image ids and labels) from a
  cell. This is what every study of the paper does.
- `ai.kind: torchxrv` or `medgemma` with `cache_mode: require` (the default) declares the
  real model and replays its cells transparently; `fill` builds the missing ones, `report`
  prints the plan, `off` runs the live model.

A cell is guarded by a model fingerprint (the weight set, or the MedGemma model, prompt set
and readout) and a dataset fingerprint (dataset, views, one image per patient, resize), so a
stale cell is never
silently replayed against different weights. Decision threshold, uncertainty measure and
calibration are applied after the cache, so one cell serves all of them. A cell is an
accumulating pool: a build completes only the image ids it is missing, and checkpoints as it
goes.

To build or check cells outside a study, give `scripts/materialize_cache.py` a scenario that
declares the model and the dataset (the ones in [`configs/cache/`](../configs/cache) built
the shipped cells):

```bash
uv run python scripts/materialize_cache.py --scenario configs/cache/vindr_ensemble.yaml \
    --limit 0 --device cuda:0 --checkpoint-every 200          # build every member
uv run python scripts/materialize_cache.py --scenario configs/cache/vindr_ensemble.yaml \
    --only chex --mode report                                 # inspect the plan only
```

| flag | default | meaning |
|---|---|---|
| `--scenario` | required | Scenario YAML that declares `ai` and `patient_generator`. |
| `--limit` | `0` | Maximum images; 0 or less means all. |
| `--sample` | `sequential` | `sequential` or `random` selection. |
| `--seed` | the scenario's seed | Sampling seed. |
| `--device` | the scenario's `ai.device` | PyTorch device. |
| `--cache-root` | the scenario's `ai.cache_root` | Root of the cache matrix. |
| `--only` | all | Comma-separated cell aliases to build. |
| `--mode` | `fill` | `report`, `require` or `fill`. |
| `--batch-size` | `1` | Images per forward pass. |
| `--checkpoint-every` | `200` | Images between checkpoints; 0 disables them. |

The calibration files in `data/cache/calibration/` are fitted with
`scripts/fit_calibration.py` ([calibration and fusion](calibration-and-fusion.md)).

## Determinism and seeds

A single `seed` drives the whole run. Internally `np.random.SeedSequence(seed)` is split into
five independent streams (patient sampling, AI, doctor, worklist, arrivals), so **swapping
one module does not change the random draws of the others**. Arrivals have their own stream,
separate from patient sampling, so comparing two arrival processes does not also compare two
case mixes. The same seed and the same scenario file give the same records and KPIs.

In a study, `replications` are seeds. Two arms at the same seed read the same patients in the
same order (common random numbers), which is what makes the
[paired contrasts](studies.md#paired-contrasts) tight.

## Tests

```bash
uv run pytest                                 # about 450 tests
uv run ruff check src tests scripts configs   # lint
uv run mypy src                               # strict type check
```

`tests/test_protocol_compliance.py` checks that every implementation satisfies its protocol
at run time. `tests/test_real_models.py` runs the real TorchXRayVision models and is skipped
unless the `models` extra is installed; it downloads the weights on first use.
[`CONTRIBUTING.md`](../CONTRIBUTING.md) lists what a pull request must pass.

## Windows notes

- Set `PYTHONUTF8=1` before running the TorchXRayVision models (`$env:PYTHONUTF8 = "1"` in
  PowerShell). Otherwise the progress bar of the weight download crashes the console, because
  the `cp1252` code page cannot render its block characters.
- Paths in the YAML files use forward slashes and work unchanged on Windows.
- `hai-sim study --workers` is capped at 61 processes, the limit of a Windows process pool
  (the cap applies on every platform).
