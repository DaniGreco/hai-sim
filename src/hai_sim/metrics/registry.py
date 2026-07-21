"""Open registry for swappable KPI bundles.

Every other axis of the framework is selectable from a scenario YAML — the AI, the
sort strategy, the persona, the severity model, the dataset. The metric axis was the
exception: bundles lived in a tuple baked into the library, so inventing a metric meant
patching `hai_sim` itself. This module makes it symmetric. A bundle reference is either
a registered name or a pointer to *the researcher's own code*, resolved at run time.

Forms accepted by `resolve_bundle`:

``"team"``
    a name in the registry — the built-ins, plus anything `register_bundle` added.
``"my_pkg.my_metrics:CostBundle"``
    an importable module, then an attribute on it.
``"analysis/my_metrics.py:CostBundle"``
    a **path to a .py file**, then an attribute. This is the form that works when the
    metric lives next to the scenario instead of inside an installed package.

The attribute may be a class (instantiated with no arguments), a ready-made instance,
or a zero-argument factory. Whatever comes out must satisfy `KPIBundle`.

⚠️ Resolving a reference imports the target, which runs its module-level code. That is
inherent — a metric *is* code — but it means a scenario file is as trusted as a script.
Do not run scenarios you would not run as Python.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hai_sim.metrics.collector import PatientRecord


@runtime_checkable
class KPIBundle(Protocol):
    """A named, swappable KPI bundle: a pure function over the run telemetry.

    Bundles are the unit of KPI selection: a scenario picks which to compute for its
    end-to-end test. Custom bundles satisfy this contract structurally — no inheritance.
    `compute` may return anything; a mapping or a dataclass keeps it reachable by the
    dotted report paths the study layer uses (e.g. ``team.rair``).
    """

    name: str

    def compute(self, records: Sequence[PatientRecord]) -> Any: ...


KPI_BUNDLES: dict[str, KPIBundle] = {}
"""Name → bundle. Mutable on purpose: `register_bundle` is the plugin seam."""

# Modules loaded from a file path, keyed by resolved path so two plugins sharing a
# file name do not shadow each other and re-referencing one does not re-execute it.
_FILE_MODULES: dict[Path, ModuleType] = {}


def register_bundle(bundle: KPIBundle, *, replace: bool = False) -> KPIBundle:
    """Add a bundle to the registry under its own ``name``; returns it for chaining.

    Refuses to shadow an existing name unless ``replace=True`` — a silent override
    would make a report mean something different from what its name says.
    """
    if not isinstance(bundle, KPIBundle):
        raise TypeError(
            f"{bundle!r} is not a KPIBundle: it needs a 'name' attribute and a "
            "'compute(records)' method"
        )
    name = bundle.name
    if not name:
        raise ValueError("a KPI bundle needs a non-empty name")
    if name in KPI_BUNDLES and not replace and KPI_BUNDLES[name] is not bundle:
        raise ValueError(
            f"KPI bundle {name!r} is already registered; pass replace=True to override it"
        )
    KPI_BUNDLES[name] = bundle
    return bundle


def resolve_bundle(reference: str) -> KPIBundle:
    """Resolve a registered name or a ``target:attribute`` reference to a bundle."""
    if ":" not in reference:
        bundle = KPI_BUNDLES.get(reference)
        if bundle is None:
            raise ValueError(
                f"unknown KPI bundle {reference!r}; registered: "
                f"{tuple(sorted(KPI_BUNDLES))}. To use your own, reference it as "
                "'module:Attribute' or 'path/to/file.py:Attribute'."
            )
        return bundle

    # rpartition, not split: a Windows path carries a drive-letter colon of its own.
    target, _, attribute = reference.rpartition(":")
    if not target or not attribute:
        raise ValueError(
            f"malformed KPI bundle reference {reference!r}; expected 'target:Attribute'"
        )
    module = _load_module(target, reference)
    try:
        obj = getattr(module, attribute)
    except AttributeError:
        raise ValueError(
            f"KPI bundle reference {reference!r}: {target!r} has no attribute "
            f"{attribute!r}"
        ) from None

    # Order matters: a bundle *class* also carries `name` and `compute` attributes, so
    # it would pass the protocol check while still needing instantiation.
    if isinstance(obj, type):
        bundle = obj()
    elif isinstance(obj, KPIBundle):
        bundle = obj
    elif callable(obj):
        bundle = obj()
    else:
        raise TypeError(
            f"KPI bundle reference {reference!r} resolved to {obj!r}, which is neither "
            "a class, a bundle instance nor a factory"
        )
    if not isinstance(bundle, KPIBundle):
        raise TypeError(
            f"KPI bundle reference {reference!r} produced {bundle!r}, which is not a "
            "KPIBundle: it needs a 'name' attribute and a 'compute(records)' method"
        )
    return bundle


def _load_module(target: str, reference: str) -> ModuleType:
    """Import ``target`` — a dotted module path, or a path to a ``.py`` file."""
    if not target.endswith(".py"):
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            raise ValueError(
                f"KPI bundle reference {reference!r}: cannot import module "
                f"{target!r} ({exc}). If it is a plain file, reference it by path "
                "ending in .py."
            ) from exc

    path = Path(target).resolve()
    cached = _FILE_MODULES.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise ValueError(f"KPI bundle reference {reference!r}: no such file {path}")
    module_name = f"_hai_sim_kpi_plugin_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"KPI bundle reference {reference!r}: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Publish before executing: a plugin defining a dataclass under
    # `from __future__ import annotations` resolves its own annotations through
    # sys.modules, and would fail on a module that is not there yet.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)  # never leave a half-executed module behind
        raise
    _FILE_MODULES[path] = module
    return module
