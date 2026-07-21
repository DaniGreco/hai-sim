"""Study layer: expand a factor grid into scenarios and tabulate KPIs (tidy results)."""

from hai_sim.study.runner import (
    StudyResult,
    run_study,
    write_long_csv,
    write_summary_csv,
    write_tidy_csv,
    write_tidy_json,
)
from hai_sim.study.spec import Run, StudyConfig, baseline_of, expand_runs, load_study
from hai_sim.study.stats import (
    Aggregate,
    CellSummary,
    Contrast,
    TidyRow,
    aggregate,
    paired_contrasts,
    summarize_cells,
)

__all__ = [
    "Aggregate",
    "CellSummary",
    "Contrast",
    "Run",
    "StudyConfig",
    "StudyResult",
    "TidyRow",
    "aggregate",
    "baseline_of",
    "expand_runs",
    "load_study",
    "paired_contrasts",
    "run_study",
    "summarize_cells",
    "write_long_csv",
    "write_summary_csv",
    "write_tidy_csv",
    "write_tidy_json",
]
