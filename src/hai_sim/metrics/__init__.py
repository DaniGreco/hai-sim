from hai_sim.metrics.collector import MetricsCollector, PatientRecord
from hai_sim.metrics.kpi import compute_grouped_report, compute_kpis, compute_report
from hai_sim.metrics.registry import (
    KPI_BUNDLES,
    KPIBundle,
    register_bundle,
    resolve_bundle,
)
from hai_sim.metrics.report import flatten_report
from hai_sim.metrics.slicing import (
    Grouping,
    LabelGrouping,
    RecordSlice,
    ValueGrouping,
    split_records,
)

__all__ = [
    "KPI_BUNDLES",
    "Grouping",
    "KPIBundle",
    "LabelGrouping",
    "MetricsCollector",
    "PatientRecord",
    "RecordSlice",
    "ValueGrouping",
    "compute_grouped_report",
    "compute_kpis",
    "compute_report",
    "flatten_report",
    "register_bundle",
    "resolve_bundle",
    "split_records",
]
