"""Capability-aware, swappable datasets (dataset-agnostic contract)."""

from hai_sim.data.mock import MockDataset
from hai_sim.data.xrv import XRVDataset

__all__ = ["MockDataset", "XRVDataset"]
