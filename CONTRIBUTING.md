# Contributing

Thank you for helping. Bug reports, new reader models, new integration policies and new
datasets are all welcome.

## Set up

```bash
uv sync --extra dev --extra analysis      # add --extra models for the live models
```

## Before opening a pull request

```bash
uv run ruff check src tests scripts configs   # lint
uv run mypy src                               # strict type check
uv run pytest -q                              # about 450 tests, a few minutes
```

All three must pass; the CI runs them on Python 3.11 and 3.12.

## How the code is organised

Every part of the workflow that a study might want to vary (the AI, the worklist order,
the reader, the integration policy, arrivals, fatigue, difficulty, metrics) is a
`Protocol` in [`src/hai_sim/core/protocols.py`](src/hai_sim/core/protocols.py), built from
YAML by [`src/hai_sim/factory.py`](src/hai_sim/factory.py). Adding an implementation means:

1. write the class against the protocol, in the module of its kind;
2. add its configuration fields to [`src/hai_sim/config/scenario.py`](src/hai_sim/config/scenario.py)
   and its branch in the factory;
3. add tests next to the existing ones in `tests/`;
4. document the new knob in [`docs/scenarios.md`](docs/scenarios.md).

A new metric does not need a code change in the package: a scenario can point to your own
class (`kpis: ["path/to/file.py:MyBundle"]`), as
[`configs/examples/custom_metrics.py`](configs/examples/custom_metrics.py) shows.
[`docs/architecture.md`](docs/architecture.md) explains the design in more detail.

## Conventions

- Python 3.11+, type hints everywhere (`mypy --strict`), line length 100.
- One seed per run, split into one random stream per module: a change to one module must
  not change the random draws of another.
- A new configuration field gets a default that keeps every existing scenario unchanged.
- Commit messages in the imperative mood ("add", "fix"), one topic per commit.
