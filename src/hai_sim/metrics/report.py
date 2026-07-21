"""Flatten a KPI report into ``dotted path -> scalar``.

A bundle may return anything — a mapping, a dataclass, a nesting of both — which is
right for reading one run, and wrong for tabulating many. This turns whatever came back
into the flat, addressable form the study layer tabulates and the console renders, using
the *same* dotted paths a study already names in its ``report:`` list (``team.rair``,
``actors.hybrid.recall_macro``, ``joint_table.joint.gt1_ai1_doc1``).

Only scalars are emitted. A per-label vector has no single value to put in a cell, and
inventing one (a mean) would quietly answer a different question than the one asked —
use a ``label`` KPI grouping to get one row per pathology instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

Scalar = int | float | str | bool | None


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool))


def flatten_report(report: Mapping[str, Any], *, max_depth: int = 6) -> dict[str, Scalar]:
    """Every scalar reachable in ``report``, keyed by its dotted path."""
    out: dict[str, Scalar] = {}
    _walk(report, prefix="", depth=0, max_depth=max_depth, out=out)
    return out


def _walk(node: Any, *, prefix: str, depth: int, max_depth: int, out: dict[str, Scalar]) -> None:
    if depth > max_depth:
        return
    if _is_scalar(node):
        if prefix:
            out[prefix] = node
        return
    if isinstance(node, Mapping):
        items: list[tuple[str, Any]] = [(str(k), v) for k, v in node.items()]
    elif is_dataclass(node) and not isinstance(node, type):
        items = [(f.name, getattr(node, f.name)) for f in fields(node)]
    else:
        return  # sequences and anything else: no single cell to put them in
    for name, value in items:
        _walk(
            value,
            prefix=f"{prefix}.{name}" if prefix else name,
            depth=depth + 1,
            max_depth=max_depth,
            out=out,
        )
