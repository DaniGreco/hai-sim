"""Study spec: expand a factor grid into concrete scenarios.

A `StudyConfig` is a base scenario + a set of factors (dotted config-paths, each with a
list of levels) + a design (full grid or one-factor-at-a-time) + replication seeds. It
expands into a list of `Run`s — one concrete, validated `ScenarioConfig` per cell x seed.
Same seed across the arms of a comparison → the common-random-numbers substreams give each
arm the *same* patient stream, so differences are the factor's effect, not sampling noise.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from hai_sim.config.scenario import KPIGroupConfig, ScenarioConfig


class StudyConfig(BaseModel):
    """A parameter study over a base scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed_study"
    base: str | dict[str, Any]  # path to a scenario YAML, or an inline scenario mapping
    factors: dict[str, list[Any]]  # dotted config-path -> list of levels
    design: Literal["grid", "ofat", "random"] = "grid"
    n_samples: int | None = None  # design 'random': how many cells to draw from the grid
    design_seed: int = 0  # design 'random': which cells get drawn
    replications: tuple[int, ...] = (0,)  # seeds; one run per cell per seed
    report: tuple[str, ...] = ()
    """KPI-paths to tabulate, e.g. ``team.rair``. Empty → every scalar the report
    flattens to, so a study need not know the path names up front."""
    kpis: tuple[str, ...] | None = None
    """Bundles to compute. None → derived from `report` when it names paths, else the
    base scenario's own selection. Name your own bundle here as
    ``path/to/file.py:Attribute`` to tabulate a metric that lives outside the library."""
    by: tuple[KPIGroupConfig, ...] = ()
    """Conditioning axes: every metric is also produced per subgroup (per severity
    band, per load regime, per pathology). Several axes cross."""
    baseline: dict[str, Any] | None = None
    """Factor assignment naming the reference arm, e.g. ``{persona: solo_senior}``. Set
    it to get **paired** contrasts: because the engine's seed substreams are independent
    per module, two arms at the same seed see the same patient stream, so the per-seed
    difference removes sampling noise instead of averaging over it. Partial assignments
    are allowed — the baseline for each arm is the cell that matches on these factors
    and agrees with the arm on all the others."""

    @field_validator("factors")
    @classmethod
    def _factors_nonempty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for name, levels in v.items():
            if not levels:
                raise ValueError(f"factor {name!r} has no levels")
        return v

    @model_validator(mode="after")
    def _check_design(self) -> StudyConfig:
        if self.design == "random" and not self.n_samples:
            raise ValueError("design 'random' requires n_samples")
        if self.baseline:
            unknown = set(self.baseline) - set(self.factors)
            if unknown:
                raise ValueError(
                    f"baseline names non-factors {sorted(unknown)}; "
                    f"factors are {sorted(self.factors)}"
                )
            for name, level in self.baseline.items():
                if level not in self.factors[name]:
                    raise ValueError(
                        f"baseline {name}={level!r} is not among that factor's levels "
                        f"{self.factors[name]}"
                    )
        return self


def load_study(path: Path) -> StudyConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return StudyConfig.model_validate(data)


@dataclass(frozen=True)
class Run:
    """One expanded run: the factor assignment, the seed, and the built scenario."""

    factors: dict[str, Any]
    seed: int
    scenario: ScenarioConfig


def _load_base(base: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(base, str):
        with Path(base).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"base scenario {base!r} did not parse to a mapping")
        return loaded
    return copy.deepcopy(base)


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` for a dotted ``path='a.b.c'``, creating dicts as needed."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _assignments(study: StudyConfig) -> list[dict[str, Any]]:
    names = list(study.factors)
    if not names:
        return [{}]
    if study.design in ("grid", "random"):
        cells = [
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*(study.factors[n] for n in names))
        ]
        if study.design == "grid":
            return cells
        # 'random': draw a subset of the full grid, for a space too large to enumerate.
        # (Not a Latin hypercube: factors here are lists of levels, not continuous
        # ranges, so there is no interval to stratify.) The baseline arm is always kept
        # — a contrast against a cell that was not run is not a contrast.
        n_samples = min(int(study.n_samples or 0), len(cells))
        forced = [c for c in cells if study.baseline and _matches(c, study.baseline)]
        pool = [c for c in cells if c not in forced]
        rng = np.random.default_rng(study.design_seed)
        take = max(n_samples - len(forced), 0)
        picked = rng.choice(len(pool), size=min(take, len(pool)), replace=False)
        return forced + [pool[int(i)] for i in sorted(picked)]
    # one-factor-at-a-time: a baseline (first level of each factor) plus, for every other
    # level of every factor, one run that varies only that factor.
    baseline = {n: study.factors[n][0] for n in names}
    out = [dict(baseline)]
    for n in names:
        for level in study.factors[n]:
            if level == baseline[n]:
                continue
            a = dict(baseline)
            a[n] = level
            out.append(a)
    return out


def _matches(assignment: dict[str, Any], pattern: dict[str, Any]) -> bool:
    """True when ``assignment`` agrees with every factor named in ``pattern``."""
    return all(assignment.get(name) == level for name, level in pattern.items())


def baseline_of(
    assignment: dict[str, Any], study: StudyConfig
) -> dict[str, Any] | None:
    """The reference cell for ``assignment``: the baseline levels on the factors the
    baseline names, and ``assignment``'s own levels everywhere else.

    Returns ``None`` for the baseline cell itself (nothing to contrast against) and when
    the study declares no baseline.
    """
    if not study.baseline or _matches(assignment, study.baseline):
        return None
    return {**assignment, **study.baseline}


def expand_runs(study: StudyConfig) -> list[Run]:
    """Expand the study into concrete, validated `Run`s (cells x replication seeds)."""
    base_dict = _load_base(study.base)
    seeds = study.replications or (int(base_dict.get("seed", 0)),)
    runs: list[Run] = []
    for assignment in _assignments(study):
        for seed in seeds:
            d = copy.deepcopy(base_dict)
            for path, level in assignment.items():
                _set_path(d, path, level)
            d["seed"] = seed
            runs.append(
                Run(factors=assignment, seed=seed, scenario=ScenarioConfig.model_validate(d))
            )
    return runs
