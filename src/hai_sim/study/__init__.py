"""Study layer: expand a factor grid into scenarios and tabulate KPIs (tidy results)."""

from hai_sim.study.runner import StudyResult, run_study, write_tidy_csv, write_tidy_json
from hai_sim.study.spec import Run, StudyConfig, expand_runs, load_study

__all__ = [
    "Run",
    "StudyConfig",
    "StudyResult",
    "expand_runs",
    "load_study",
    "run_study",
    "write_tidy_csv",
    "write_tidy_json",
]
