"""The result browser: read what a run left on disk, and make it comparable.

Deliberately split in two, and the split is the point:

- `hai_sim.ui.results` — **pure**. Discovers artefacts, reshapes them, diffs two
  configurations. No Streamlit import, so it runs in the normal test suite whether or
  not the ``[ui]`` extra is installed, and so the reshaping can be reused by a script,
  a notebook, or a different front end later.
- `hai_sim.ui.app` — the Streamlit page. Widgets and layout only; it must contain no
  logic that a test would want to assert on.

**This browser never runs a simulation and never writes into a run's output.** It is a
reader. A run is defined by its YAML and its seed (thesis §5.10: determinism is a
requirement, not a nicety), so anything that produced a result already left the file
that reproduces it — the browser's job is to show you that file next to its numbers,
not to become an alternative way of getting numbers without one.
"""

from hai_sim.ui.results import (
    Artifact,
    config_diff,
    contrast_rows,
    describe,
    discover,
    factor_names,
    load_records,
    metric_names,
    run_kpi_rows,
    study_rows,
)

__all__ = [
    "Artifact",
    "config_diff",
    "contrast_rows",
    "describe",
    "discover",
    "factor_names",
    "load_records",
    "metric_names",
    "run_kpi_rows",
    "study_rows",
]
