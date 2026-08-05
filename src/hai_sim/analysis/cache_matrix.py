"""The cache matrix: address and identify per-model inference caches on disk.

The disk layout is a matrix — one cache file per *(atomic model, dataset)*, named
``<cache_root>/<dataset>__<alias>``. An **atomic model** is the smallest independently
buildable predictor: a single TorchXRayVision weight set, or one MedGemma
configuration. An ensemble is *not* a file — it is several member cells composed at
analysis time (see `hai_sim.models.cached`), so dropping or adding a member is a
choice of which files to load, never a rebuild.

A cell's **identity** is what makes two computations "the same", split in two:

- ``model_fingerprint`` — the exact model config that produced the raw probabilities
  (the weight set for xrv; ``model_id`` + ``prompt_set`` + ``readout`` for MedGemma).
  The decision threshold, the uncertainty measure and calibration are all applied
  *downstream* of the cache, so they are deliberately **not** part of it: one cell
  serves every threshold and UQ measure.
- ``dataset_fingerprint`` — what the model was shown (dataset, views, unique-patients,
  resize). The image **count** is deliberately absent: a cell is an *accumulating pool*
  keyed by image id, so a study needing images the cell lacks completes only the gap
  rather than being pinned to one fixed-N sample.

This module is pure addressing + identity (it imports no torch): the resolver that
turns a study's needs into a build plan, and the actor that composes member cells,
build on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hai_sim.analysis.inference_cache import InferenceCache

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from hai_sim.config.scenario import AIModelConfig, PatientGenConfig

DEFAULT_CACHE_ROOT = Path("data/cache")

# Characters that are unsafe in a file name on the platforms we target.
_UNSAFE = '/\\:*?"<>| \t'


def _sanitize(name: str) -> str:
    """Make an alias safe as a file-name component (aliases are already tidy; this is a
    safety net for an unusual weight set or model id)."""
    return "".join("-" if c in _UNSAFE else c for c in name)


def _xrv_alias(weight: str) -> str:
    """``densenet121-res224-chex`` -> ``chex``; leave a non-standard name as-is."""
    if weight.startswith("densenet121"):
        return weight.rsplit("-", 1)[-1]
    return weight


def _medgemma_alias(model_id: str) -> str:
    """``google/medgemma-1.5-4b-it`` -> ``medgemma-1.5-4b``."""
    tail = model_id.rsplit("/", 1)[-1]
    return tail[:-3] if tail.endswith("-it") else tail


@dataclass(frozen=True)
class CellSpec:
    """One atomic model in the matrix: how to name it, identify it, and build it.

    ``ai_cfg`` is a single-model config that builds *just* this predictor — for an
    ensemble member it is the parent config narrowed to that one weight set.
    """

    alias: str
    """Human-readable name; the ``<model>`` half of the ``<dataset>__<model>`` stem."""
    fingerprint: str
    """Exact model identity, checked against a cell's manifest before it counts as a hit."""
    ai_cfg: AIModelConfig
    """Config that builds this atomic model alone."""

    def stem(self, dataset_id: str, root: Path = DEFAULT_CACHE_ROOT) -> Path:
        return Path(root) / f"{dataset_id}__{self.alias}"


def cells_for_ai(cfg: AIModelConfig) -> list[CellSpec]:
    """Expand an AI config into its atomic cells.

    A TorchXRayVision ensemble becomes one cell per weight set (so a member can be
    dropped or added by loading a different set of files); a MedGemma config is a
    single cell (its several prompt paraphrases are *inside* the one model, not
    independently droppable). Synthetic / mock / manual-``cached`` kinds are not
    cacheable through the matrix and raise.
    """
    if cfg.kind == "torchxrv":
        weights = [cfg.weights] if isinstance(cfg.weights, str) else list(cfg.weights)
        if not weights:
            raise ValueError("torchxrv config has no weight sets to cache")
        cells = [
            CellSpec(
                alias=_sanitize(_xrv_alias(w)),
                fingerprint=f"torchxrv|{w}",
                ai_cfg=cfg.model_copy(update={"weights": w}),
            )
            for w in weights
        ]
        aliases = [c.alias for c in cells]
        if len(set(aliases)) != len(aliases):
            raise ValueError(
                f"ensemble weight sets collapse to duplicate cache aliases {aliases}; "
                "give distinct weight sets"
            )
        return cells
    if cfg.kind == "medgemma":
        fingerprint = (
            f"medgemma|{cfg.model_id}|prompt_set={cfg.prompt_set}|"
            f"readout={cfg.readout}|both_polarities={cfg.both_polarities}"
        )
        return [
            CellSpec(
                alias=_sanitize(_medgemma_alias(cfg.model_id)),
                fingerprint=fingerprint,
                ai_cfg=cfg,
            )
        ]
    raise ValueError(
        f"ai.kind={cfg.kind!r} is not cacheable through the matrix; "
        "only 'torchxrv' and 'medgemma' produce member probabilities"
    )


def dataset_id(pg: PatientGenConfig) -> str:
    """The ``<dataset>`` half of a cache stem for this patient generator."""
    if pg.dataset_id:
        return pg.dataset_id
    if pg.kind == "xrv_dataset":
        return pg.dataset
    if pg.kind == "image_folder":
        if pg.imgpath is None:
            raise ValueError("image_folder needs imgpath (or an explicit dataset_id)")
        return Path(pg.imgpath).parent.name or "image_folder"
    raise ValueError(
        f"patient_generator.kind={pg.kind!r} has no cache dataset id; use xrv_dataset "
        "or image_folder"
    )


def dataset_cell_fingerprint(pg: PatientGenConfig) -> str:
    """Identity of what the model saw — dataset + preprocessing, *without* image count."""
    did = dataset_id(pg)
    if pg.kind == "xrv_dataset":
        return (
            f"xrv:{did}|views={','.join(pg.views)}|"
            f"unique={pg.unique_patients}|resize={pg.resize}"
        )
    if pg.kind == "image_folder":
        return f"image_folder:{did}|resize={pg.resize}|ext={pg.image_ext}"
    raise ValueError(f"patient_generator.kind={pg.kind!r} cannot be cached")


def present_ids(cache: InferenceCache) -> frozenset[str]:
    """The image ids a cell already covers."""
    return frozenset(cache.image_ids)


@dataclass(frozen=True)
class CellStatus:
    """A single cell resolved against disk: does it exist, does it match, what does it hold."""

    spec: CellSpec
    stem: Path
    exists: bool
    conflict: str | None
    """Set when a cell exists at the stem but its stored identity does not match what is
    wanted — the caller must not silently reuse or extend it."""
    present_ids: frozenset[str]

    def missing(self, requested: Iterable[str]) -> list[str]:
        """Requested ids this cell does not yet cover, in order and de-duplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for i in requested:
            if i not in self.present_ids and i not in seen:
                seen.add(i)
                out.append(i)
        return out


def inspect_cell(
    spec: CellSpec,
    dataset_id_: str,
    expected_dataset_fingerprint: str,
    root: Path = DEFAULT_CACHE_ROOT,
) -> CellStatus:
    """Resolve one cell against disk: missing, matching (with coverage), or conflicting.

    A stored cell counts as usable only if both fingerprints match what is wanted;
    otherwise ``conflict`` is set and the caller decides (rename, rebuild, or error).
    An empty stored fingerprint (a cache written before this field existed) is treated
    as "unknown, do not block" rather than a mismatch.
    """
    stem = spec.stem(dataset_id_, root)
    if not stem.with_suffix(".npz").exists():
        return CellStatus(spec, stem, exists=False, conflict=None, present_ids=frozenset())
    cache = InferenceCache.load(stem)
    conflict: str | None = None
    if cache.model_fingerprint and cache.model_fingerprint != spec.fingerprint:
        conflict = (
            f"model identity differs: on disk {cache.model_fingerprint!r}, "
            f"wanted {spec.fingerprint!r}"
        )
    elif (
        expected_dataset_fingerprint
        and cache.dataset_fingerprint
        and cache.dataset_fingerprint != expected_dataset_fingerprint
    ):
        conflict = (
            f"dataset identity differs: on disk {cache.dataset_fingerprint!r}, "
            f"wanted {expected_dataset_fingerprint!r}"
        )
    return CellStatus(
        spec, stem, exists=True, conflict=conflict, present_ids=present_ids(cache)
    )


# --- resolving a whole study's cells, and the build policy -------------------


@dataclass(frozen=True)
class MatrixPlan:
    """Every cell a study needs, resolved against disk, plus the ids it requested.

    This is the study-governed view: the run declares its model and dataset, the plan
    says for each atomic cell whether it is a hit, a partial (some ids cached), a miss,
    or a conflict — so the `cache_mode` policy can decide to serve, complete or build.
    """

    dataset_id: str
    requested_ids: tuple[str, ...]
    cells: tuple[CellStatus, ...]

    @property
    def conflicts(self) -> list[CellStatus]:
        return [c for c in self.cells if c.conflict is not None]

    @property
    def incomplete(self) -> list[CellStatus]:
        """Cells that do not yet cover every requested id (missing or wholly absent)."""
        return [c for c in self.cells if c.missing(self.requested_ids)]

    @property
    def is_complete(self) -> bool:
        return not self.incomplete and not self.conflicts


def resolve_matrix(
    cells: Iterable[CellSpec],
    dataset_id_: str,
    dataset_fingerprint: str,
    requested_ids: Iterable[str],
    root: Path = DEFAULT_CACHE_ROOT,
) -> MatrixPlan:
    """Resolve every cell of a study against disk for the requested image ids."""
    reqs = tuple(dict.fromkeys(requested_ids))  # de-duplicate, keep order
    statuses = tuple(
        inspect_cell(c, dataset_id_, dataset_fingerprint, root) for c in cells
    )
    return MatrixPlan(dataset_id_, reqs, statuses)


def plan_summary(plan: MatrixPlan) -> str:
    """A human-readable per-cell status table, for `report` and error messages."""
    lines = [
        f"cache matrix for dataset {plan.dataset_id!r} "
        f"({len(plan.requested_ids)} images requested):"
    ]
    for st in plan.cells:
        miss = st.missing(plan.requested_ids)
        if st.conflict is not None:
            state = f"CONFLICT ({st.conflict})"
        elif not st.exists:
            state = f"miss (build {len(miss)})"
        elif miss:
            state = f"partial ({len(st.present_ids)} cached, build {len(miss)})"
        else:
            state = f"hit ({len(st.present_ids)} cached)"
        lines.append(f"  {st.spec.alias:<18} {state}  -> {st.stem}")
    return "\n".join(lines)


def apply_cache_mode(
    plan: MatrixPlan,
    mode: str,
    fill: Callable[[CellSpec], None] | None = None,
) -> MatrixPlan:
    """Enforce the study's cache policy against a resolved plan.

    A **conflict** always raises, whatever the mode — two experiments must never share a
    file. Otherwise:

    - ``require`` — replay only: raise if anything is missing (the safe default, so a
      study on a laptop never silently kicks off a multi-day build).
    - ``fill`` — build the gaps by calling ``fill`` once per incomplete cell (this is
      where torch lives; run it on the GPU server).
    - ``report`` — do nothing; the caller prints `plan_summary` and stops.
    """
    if plan.conflicts:
        raise ValueError("cache matrix has conflicting cells:\n" + plan_summary(plan))
    if mode == "report":
        return plan
    if mode == "require":
        if plan.incomplete:
            raise ValueError(
                "cache_mode 'require' but the cache matrix is incomplete. Build the gaps "
                "(cache_mode: fill on the GPU server, or the materialize command):\n"
                + plan_summary(plan)
            )
        return plan
    if mode == "fill":
        if fill is None:
            raise ValueError("cache_mode 'fill' needs a fill callback")
        for st in plan.incomplete:
            fill(st.spec)
        return plan
    raise ValueError(f"unknown cache_mode: {mode!r} (use require / fill / report)")
