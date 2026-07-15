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

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hai_sim.config.scenario import ScenarioConfig


class StudyConfig(BaseModel):
    """A parameter study over a base scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed_study"
    base: str | dict[str, Any]  # path to a scenario YAML, or an inline scenario mapping
    factors: dict[str, list[Any]]  # dotted config-path -> list of levels
    design: Literal["grid", "ofat"] = "grid"
    replications: tuple[int, ...] = (0,)  # seeds; one run per cell per seed
    report: tuple[str, ...] = Field(min_length=1)  # KPI-paths to tabulate, e.g. team.rair

    @field_validator("factors")
    @classmethod
    def _factors_nonempty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for name, levels in v.items():
            if not levels:
                raise ValueError(f"factor {name!r} has no levels")
        return v


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
    if study.design == "grid":
        return [
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*(study.factors[n] for n in names))
        ]
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
